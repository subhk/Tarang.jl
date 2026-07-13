# Fluid Dynamics Examples

Collection of fluid dynamics simulations with Tarang.jl. Each example is a complete, self-contained
script that runs as it stands; the resolutions and step counts are kept small so they finish in
seconds to a couple of minutes, and the numbers quoted after each one are what it prints. Raise `N`
and the stop criterion for production runs.

## Incompressible Navier-Stokes

Which formulation you write is decided by the geometry, because the incompressibility constraint
is solved through the tau method on a *coupled* (non-periodic) axis:

| geometry | formulation |
|----------|-------------|
| walls in one direction, periodic in the others | primitive variables `u`, `p` + Chebyshev tau method |
| fully periodic | vorticity–streamfunction (2D) |

Two configurations that look natural but do **not** work:

- **Two bounded axes** (Chebyshev × Chebyshev — the classic lid-driven cavity) cannot be built.
  Tarang couples exactly one non-periodic axis; the remaining axes must be separable (Fourier).
  Even with correct tau variables and lifts, the solve fails with
  `DimensionMismatch: second dimension of A, 12, does not match the first dimension of B, 144`.
- **Primitive variables in a fully periodic box.** `div(u) + tau_p = 0` is accepted, but with no
  coupled axis the constraint is never enforced: starting from a random (non-solenoidal) field,
  `max|div u|` was 2.918 before and 2.920 after 20 `RK222()` steps, and `SBDF2()` throws
  `SingularException`. Use the vorticity–streamfunction form in a periodic box, where
  `u = skew(∇ψ)` is divergence-free by construction.

### 2D Channel Flow (Poiseuille)

Wall-bounded flow driven by a constant body force: Fourier in the streamwise direction, Chebyshev
between the walls, no-slip at both walls.

```julia
using Tarang, Printf

Lx, Lz = 4.0, 1.0
Nx, Nz = 16, 16
nu     = 1.0          # kinematic viscosity
fx     = 1.0          # constant streamwise body force (the pressure gradient)

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
u = VectorField(domain, "u")

# tau variables carry the Fourier basis: one tau per streamwise mode
tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u     = grad(u) + ez * τ_lift(tau_u1)      # first-order reduction

problem = IVP([p, u, tau_p, tau_u1, tau_u2])
add_parameters!(problem, nu=nu, fx=fx, ex=ex, grad_u=grad_u, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u) + fx*ex")

add_bc!(problem, "u(z=0) = 0")        # no-slip walls
add_bc!(problem, "u(z=$Lz) = 0")
add_bc!(problem, "integ(p) = 0")      # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=2e-2)

cfl = CFL(solver; initial_dt=2e-2, cadence=10, safety=0.5, max_dt=2e-2)
add_velocity!(cfl, u)

run!(solver; stop_iteration=250, cfl=cfl, progress=false)   # t = 5 viscous times

# Steady state is the analytic Poiseuille parabola ux(z) = fx z (Lz - z) / (2ν)
ux = u.components[1]
ensure_layout!(ux, :g)
x, z    = local_grids(dist, xbasis, zbasis)
profile = get_grid_data(ux)[1, :]                 # ux is x-independent
exact   = @. fx * z * (Lz - z) / (2 * nu)
@printf("t = %.2f   max|ux| = %.6f (exact %.6f)   rel. error = %.2e\n",
        solver.sim_time, maximum(profile), fx * Lz^2 / (8 * nu),
        maximum(abs, profile .- exact) / maximum(exact))
```

After 250 steps (`t = 5`, five viscous times) the flow has converged to the analytic parabola:
the profile matches `fx z (Lz - z) / 2ν` to a relative error of `6.3e-15`. The printed
`max|ux| = 0.123634` is the largest value *on the grid* — the Chebyshev nodes do not include the
centreline where the exact peak `fx Lz²/8ν = 0.125` sits.

Note that the boundary-condition strings are parsed by Tarang, not Julia: they cannot see Julia
globals, so `Lz` is interpolated into the string by Julia before `add_bc!` sees it. Boundary
conditions must go through `add_bc!` — declaring them with `add_equation!` silently fails to
enforce anything but a constant.

### Kelvin-Helmholtz Instability

Shear-layer instability in a doubly periodic domain, in vorticity–streamfunction form. A double
shear layer keeps the base state periodic.

```julia
using Tarang, Printf

N      = 64
δ, A   = 0.05, 1e-2      # shear-layer thickness, perturbation amplitude
nu     = 1e-4

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 1.0), dealias=3/2)
zbasis = RealFourier(coords["z"]; size=N, bounds=(0.0, 1.0), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

ζ     = ScalarField(domain, "zeta")     # vorticity
ψ     = ScalarField(domain, "psi")      # streamfunction
u     = VectorField(domain, "u")        # velocity, divergence-free by construction
tau_ψ = ScalarField(dist, "tau_psi", (), Float64)

problem = IVP([ζ, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu)
add_equation!(problem, "∂t(zeta) - nu*Δ(zeta) = -u⋅∇(zeta)")   # vorticity transport
add_equation!(problem, "Δ(psi) + tau_psi - zeta = 0")          # Δψ = ζ
add_equation!(problem, "u - skew(grad(psi)) = 0")              # u = ∇⊥ψ
add_bc!(problem, "integ(psi) = 0")                             # gauge

solver = InitialValueSolver(problem, RK222(); dt=2e-3)

# Double shear layer (periodic): ux = tanh((z-1/4)/δ) below, tanh((3/4-z)/δ) above,
# so ζ = -∂z(ux); seeded with a k=1 perturbation.
x, z = local_grids(dist, xbasis, zbasis)
zz   = reshape(z, 1, :)
ensure_layout!(ζ, :g)
get_grid_data(ζ) .= @. ifelse(zz < 0.5,
                              -(1 - tanh((zz - 0.25) / δ)^2) / δ,
                               (1 - tanh((0.75 - zz) / δ)^2) / δ) +
                       A * sin(2π * x) * cos(2π * zz)
ensure_layout!(ζ, :c)

uz = u.components[2]
uz_max() = (ensure_layout!(uz, :g); maximum(abs, get_grid_data(uz)))

run!(solver; stop_iteration=1, progress=false)      # one step populates u from ψ
uz0 = uz_max()
run!(solver; stop_iteration=500, progress=false)
ensure_layout!(ζ, :g)
@printf("t = %.2f   max|uz|: %.2e -> %.2e (×%.0f)   max|zeta| = %.2f   E = %.4f\n",
        solver.sim_time, uz0, uz_max(), uz_max() / uz0,
        maximum(abs, get_grid_data(ζ)), total_kinetic_energy(u))
```

`ψ` and `u` are algebraic (no `∂t`), so they are only filled once the solver takes a step — hence
the one-step warm-up before reading the initial `max|uz|`. Over `t = 1` the perturbation grows
from `7.96e-4` to `6.64e-3` (a factor of 8): the shear layer is unstable, as expected.

## Thermal Convection

Both examples below are Rayleigh-Bénard-type: Fourier in `x`, Chebyshev in `z`, tau variables on
the Fourier basis, first-order reduction (`grad_X = ∇X + ẑ·lift(τ)`) for the Chebyshev tau method.

### Double-Diffusive Convection

Convection driven by two scalars with very different diffusivities (temperature and salinity).

```julia
using Tarang, Printf

Lx, Lz = 2.0, 1.0
Nx, Nz = 32, 24
Pr     = 7.0      # Prandtl number
tau_d  = 0.01     # diffusivity ratio κ_S/κ_T
Ra_T   = 1e5      # thermal Rayleigh number
Ra_S   = 1e4      # solutal Rayleigh number

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")      # temperature
S = ScalarField(domain, "S")      # salinity
u = VectorField(domain, "u")

tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_S1 = ScalarField(dist, "tau_S1", (xbasis,), Float64)
tau_S2 = ScalarField(dist, "tau_S2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)
grad_S = grad(S) + ez * τ_lift(tau_S1)

problem = IVP([p, T, S, u, tau_p, tau_T1, tau_T2, tau_S1, tau_S2, tau_u1, tau_u2])
add_parameters!(problem, Pr=Pr, taud=tau_d, buoyT=Ra_T*Pr, buoyS=Ra_S*Pr, ez=ez,
                grad_u=grad_u, grad_T=grad_T, grad_S=grad_S, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(S) - taud*div(grad_S) + τ_lift(tau_S2) = -u⋅∇(S)")
add_equation!(problem,
    "∂t(u) - Pr*div(grad_u) + ∇(p) - buoyT*T*ez + buoyS*S*ez + τ_lift(tau_u2) = -u⋅∇(u)")

add_bc!(problem, "T(z=0) = 1"); add_bc!(problem, "T(z=$Lz) = 0")
add_bc!(problem, "S(z=0) = 1"); add_bc!(problem, "S(z=$Lz) = 0")
add_bc!(problem, "u(z=0) = 0"); add_bc!(problem, "u(z=$Lz) = 0")
add_bc!(problem, "integ(p) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-4)

# Linear conduction profiles + damped noise on both scalars
x, z = local_grids(dist, xbasis, zbasis)
for (field, seed) in ((T, 42), (S, 43))
    fill_random!(field, "g"; seed=seed, distribution="normal", scale=1e-3)
    get_grid_data(field) .*= z' .* (Lz .- z')
    get_grid_data(field) .+= 1.0 .- z' ./ Lz
    ensure_layout!(field, :c)
end

run!(solver; stop_iteration=100, progress=false)

uz = u.components[2]
ensure_layout!(T, :g); ensure_layout!(S, :g); ensure_layout!(uz, :g)
Tg, Sg = get_grid_data(T), get_grid_data(S)
@printf("t = %.4f   max|T| = %.4f   max|S| = %.4f   max|uz| = %.3e\n",
        solver.sim_time, maximum(abs, Tg), maximum(abs, Sg), maximum(abs, get_grid_data(uz)))
@printf("BC residuals: |T(z=0)-1| = %.1e   |S(z=Lz)| = %.1e\n",
        maximum(abs, Tg[:, 1] .- 1), maximum(abs, Sg[:, end]))
```

After 100 steps the scalars stay in `[0, 1]`, convection is starting (`max|uz| = 2.4`), and the
boundary conditions hold to machine precision (`|T(z=0) − 1| ≈ 1e-15`, `|S(z=Lz)| = 1.2e-16`).
The buoyancy signs are what set the regime: `-buoyT*T*ez + buoyS*S*ez` makes temperature
destabilizing and salinity stabilizing.

### Rotating Convection

Rotation about the vertical axis couples the in-plane flow to an out-of-plane velocity, so the
2D plane needs a third velocity component: a scalar field `v`. The Coriolis terms `+f v x̂` and
`−f ux` are the two halves of `−2Ω × u`.

```julia
using Tarang, Printf

Lx, Lz = 2.0, 1.0
Nx, Nz = 32, 24
Pr     = 1.0
Ra     = 2e4
Ta     = 1e4                 # Taylor number
fcor   = sqrt(Ta) * Pr       # Coriolis coefficient in these units

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
v = ScalarField(domain, "v")      # out-of-plane (spanwise) velocity
u = VectorField(domain, "u")      # in-plane velocity (ux, uz)

tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_v1 = ScalarField(dist, "tau_v1", (xbasis,), Float64)
tau_v2 = ScalarField(dist, "tau_v2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)
grad_v = grad(v) + ez * τ_lift(tau_v1)
ux     = u.components[1]

problem = IVP([p, T, v, u, tau_p, tau_T1, tau_T2, tau_v1, tau_v2, tau_u1, tau_u2])
add_parameters!(problem, Pr=Pr, buoy=Ra*Pr, fcor=fcor, ex=ex, ez=ez, ux=ux,
                grad_u=grad_u, grad_T=grad_T, grad_v=grad_v, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
# Spanwise momentum: Coriolis couples v to ux
add_equation!(problem, "∂t(v) - Pr*div(grad_v) + τ_lift(tau_v2) = -u⋅∇(v) - fcor*ux")
# In-plane momentum: buoyancy along ẑ, Coriolis along x̂
add_equation!(problem,
    "∂t(u) - Pr*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u) + fcor*v*ex")

add_bc!(problem, "T(z=0) = 1"); add_bc!(problem, "T(z=$Lz) = 0")
add_bc!(problem, "u(z=0) = 0"); add_bc!(problem, "u(z=$Lz) = 0")
add_bc!(problem, "v(z=0) = 0"); add_bc!(problem, "v(z=$Lz) = 0")
add_bc!(problem, "integ(p) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-4)

x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (Lz .- z')
get_grid_data(T) .+= 1.0 .- z' ./ Lz
ensure_layout!(T, :c)

run!(solver; stop_iteration=100, progress=false)

uz = u.components[2]
ensure_layout!(v, :g); ensure_layout!(uz, :g)
@printf("t = %.4f   max|uz| = %.3e   max|v| = %.3e\n",
        solver.sim_time, maximum(abs, get_grid_data(uz)), maximum(abs, get_grid_data(v)))
```

After 100 steps: `max|uz| = 9.8e-3` and `max|v| = 2.5e-3` — the spanwise velocity is generated
entirely by rotation (it is zero at `Ta = 0`).

There is no parseable cross-product in equation strings, so write the Coriolis force with unit
vectors and components, as above. A vector times a scalar field (`fcor*v*ex`) is fine on either
side of the equation.

## Stratified Flows

### Internal Gravity Waves

Boussinesq flow with a stable background stratification, doubly periodic, in
vorticity–streamfunction form. With `u = ∇⊥ψ` the vertical velocity is `uz = ∂x(ψ)`, which is how
the stratification term is written.

```julia
using Tarang, Printf

N     = 32
N2    = 1.0        # Brunt-Väisälä frequency squared
nu    = 1e-6
kappa = 1e-6
A     = 1e-3       # wave amplitude (linear regime)

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
zbasis = RealFourier(coords["z"]; size=N, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

ζ     = ScalarField(domain, "zeta")
ψ     = ScalarField(domain, "psi")
b     = ScalarField(domain, "b")        # buoyancy
u     = VectorField(domain, "u")
tau_ψ = ScalarField(dist, "tau_psi", (), Float64)

problem = IVP([ζ, ψ, b, u, tau_ψ])
add_parameters!(problem, nu=nu, kappa=kappa, N2=N2)

add_equation!(problem, "∂t(zeta) - nu*Δ(zeta) - ∂x(b) = -u⋅∇(zeta)")
add_equation!(problem, "∂t(b) - kappa*Δ(b) + N2*∂x(psi) = -u⋅∇(b)")
add_equation!(problem, "Δ(psi) + tau_psi - zeta = 0")
add_equation!(problem, "u - skew(grad(psi)) = 0")
add_bc!(problem, "integ(psi) = 0")

# Plane wave with kx = kz = 1: theory gives ω = N*kx/|k| = 1/√2
kx, kz = 1.0, 1.0
ω      = sqrt(N2) * kx / hypot(kx, kz)
period = 2π / ω

solver = InitialValueSolver(problem, RK222(); dt=period / 2000)

x, z = local_grids(dist, xbasis, zbasis)
ensure_layout!(b, :g)
get_grid_data(b) .= A .* cos.(kx .* x .+ kz .* reshape(z, 1, :))
ensure_layout!(b, :c)
b0 = copy(get_grid_data(b))

# Starting from rest, b(t) = b(0)*cos(ωt): the correlation with b(0) traces the wave
corr() = (ensure_layout!(b, :g); sum(get_grid_data(b) .* b0) / sum(b0 .^ 2))

for (frac, steps) in ((0.5, 1000), (1.0, 2000))
    run!(solver; stop_iteration=steps, progress=false)
    @printf("t = %.3f (%.2f period)  <b(t),b(0)>/<b(0),b(0)> = %+.4f  (cos(ωt) = %+.4f)\n",
            solver.sim_time, frac, corr(), cos(ω * solver.sim_time))
end
```

This reproduces the internal-wave dispersion relation `ω = N kx/|k|` exactly: the buoyancy field
anti-correlates with its initial state after half a period (`−1.0000`) and returns to it after a
full period (`+1.0000`, theoretical period `8.8858`).

## Turbulence

### Decaying 2D Turbulence

Band-limited random vorticity in a doubly periodic box, then let it evolve: the enstrophy cascades
to small scales and is dissipated, while the energy stays at large scales.

```julia
using Tarang, Printf, Random
using Statistics: mean

N  = 64
ν  = 1e-3
dt = 2e-3

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (xbasis, ybasis))

ζ     = ScalarField(domain, "zeta")
ψ     = ScalarField(domain, "psi")
u     = VectorField(domain, "u")
tau_ψ = ScalarField(dist, "tau_psi", (), Float64)

problem = IVP([ζ, ψ, u, tau_ψ])
add_parameters!(problem, nu=ν)
add_equation!(problem, "∂t(zeta) - nu*Δ(zeta) = -u⋅∇(zeta)")
add_equation!(problem, "Δ(psi) + tau_psi - zeta = 0")
add_equation!(problem, "u - skew(grad(psi)) = 0")
add_bc!(problem, "integ(psi) = 0")

solver = InitialValueSolver(problem, RK222(); dt=dt)

# Band-limited random vorticity: energy initially in 4 ≤ |k| ≤ 8
Random.seed!(42)
x, y = local_grids(dist, xbasis, ybasis)
ζ0 = zeros(length(x), length(y))
for kx in 1:10, ky in 1:10
    4 <= hypot(kx, ky) <= 8 || continue
    a, φ = randn(), 2π * rand()
    @. ζ0 += a * sin(kx * x + φ) * cos(ky * y' + φ)
end
ζ0 .*= 10.0 / maximum(abs, ζ0)
ensure_layout!(ζ, :g)
get_grid_data(ζ) .= ζ0
ensure_layout!(ζ, :c)

enstrophy_of(ζ) = (ensure_layout!(ζ, :g); 0.5 * mean(get_grid_data(ζ) .^ 2))

# ψ and u are algebraic: one step of the solver populates them from ζ
run!(solver; stop_iteration=1, progress=false)
E0, Z0 = total_kinetic_energy(u), enstrophy_of(ζ)

run!(solver; stop_iteration=500, progress=false)

E1, Z1 = total_kinetic_energy(u), enstrophy_of(ζ)
@printf("t = %.2f   E: %.3f -> %.3f (%.0f%%)   Z: %.2f -> %.2f (%.0f%%)\n",
        solver.sim_time, E0, E1, 100 * E1 / E0, Z0, Z1, 100 * Z1 / Z0)

spec = energy_spectrum(u)                       # NamedTuple (k, power, bin_counts, bin_edges)
kpeak = spec.k[argmax(spec.power)]
@printf("energy spectrum peaks at k = %.1f\n", kpeak)
```

Over `t = 1` at `ν = 1e-3`: `E` falls from `5.225` to `4.902` and `Z` from `4.26` to `3.95`, and
the energy spectrum still peaks at `k = 5.5`, inside the band it was seeded in. Push `N`, lower
`ν` and integrate for longer to see the inverse cascade move that peak to smaller `k`.

Two things are deliberately not shown here. **3D homogeneous isotropic turbulence** has no supported
formulation today: a triply periodic box has no pressure projection (see above), and the
vorticity–streamfunction trick is 2D-only. **Stochastically forced turbulence** via
`StochasticForcing` builds and steps, but the energy it actually injects is many orders of magnitude
below its `energy_injection_rate` setting (at `N = 64`, `ε = 1`, `k_f = 8`, the kinetic energy after
`t = 1` was `1.4e-6`, not `≈ 1`), so the knob does not currently mean what it says.

## Tips

### Resolution Requirements

| Problem | Typical Resolution |
|---------|-------------------|
| Laminar | 64-128 |
| Transitional | 128-256 |
| Turbulent (moderate Re) | 256-512 |
| Turbulent (high Re) | 512-2048+ |

### Timestepping

- Use `RK222` or `RK443` for advection-dominated flows.
- Use `SBDF2`/`SBDF3` for diffusion-dominated or stiff problems.
- Drive the loop with a CFL controller (as in the channel example above): construct it from the
  **solver**, register a `VectorField` velocity, and hand it to `run!`.

```julia
cfl = CFL(solver; initial_dt=1e-3, cadence=10, safety=0.5, max_dt=1e-2)
add_velocity!(cfl, u)                 # VectorField only
run!(solver; stop_iteration=1000, cfl=cfl)
```

### Diagnostics

`total_kinetic_energy` and `total_enstrophy` return domain integrals (`Float64`); `kinetic_energy`
and `enstrophy` return the corresponding `ScalarField`s. `energy_spectrum` needs every axis to be
Fourier. `global_max` is MPI-reduced, so it returns the same value on every rank.

```julia
E    = total_kinetic_energy(u)      # Float64: domain integral of ½|u|²
Z    = total_enstrophy(u)           # Float64: domain integral of ½ω²  (2D only)
spec = energy_spectrum(u)           # NamedTuple: (k, power, bin_counts, bin_edges)

ux = u.components[1]
ensure_layout!(ux, :g)
umax = global_max(dist, abs.(get_grid_data(ux)))   # MPI-reduced: same on every rank

@printf("E = %.4e   Z = %.4e   max|ux| = %.4f   spectrum peak at k = %.1f\n",
        E, Z, umax, spec.k[argmax(spec.power)])
```

### Running in parallel

The fully periodic examples (Kelvin-Helmholtz, internal waves, decaying turbulence) run under MPI
unchanged — they use `local_grids` for initial conditions, which returns each rank's own slice:

```
mpiexec -n 2 julia --project=. kelvin_helmholtz.jl
```

The Kelvin-Helmholtz script above gives bit-identical output at 1 and 2 ranks
(`max|zeta| = 18.62`, `E = 0.3948`).

The wall-bounded examples (channel, double-diffusive, rotating convection) are **serial**: their
right-hand sides contain `-u⋅∇(u)`, a Chebyshev derivative in the explicit RHS, which a
distributed run rejects (each rank owns only part of the Chebyshev axis). MPI also requires the
Chebyshev axis to come *first* in the domain. See [Running with MPI](../getting_started/running_with_mpi.md).

## See Also

- [Rayleigh-Bénard Tutorial](../tutorials/ivp_2d_rbc.md)
- [Heat Transfer Examples](heat_transfer.md)
- [Example Gallery](gallery.md)
