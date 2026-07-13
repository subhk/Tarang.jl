# Heat Transfer Examples

Collection of heat transfer simulations with Tarang.jl.

A bounded (Chebyshev) direction carries boundary conditions with the **tau method**:
one `tau` variable per boundary condition, lifted into the bulk equation and declared
with `add_bc!`. Boundary conditions are *never* declared with `add_equation!` —
a constant BC happens to come out right that way, but anything space- or time-dependent
is silently ignored.

## Pure Diffusion

### 1D Heat Equation

A hot wall at `x=0`, a cold wall at `x=1`. The steady state is the linear conduction
profile `T = 1 - x`.

```julia
using Tarang

coords = CartesianCoordinates("x")
dist   = Distributor(coords; dtype=Float64, device=CPU())
basis  = ChebyshevT(coords["x"]; size=32, bounds=(0.0, 1.0))
domain = Domain(dist, (basis,))

T    = ScalarField(domain, "T")
tau1 = ScalarField(dist, "tau1", (), Float64)   # one tau per boundary condition
tau2 = ScalarField(dist, "tau2", (), Float64)

# first-order reduction: grad_T = ∇T + x̂·lift(τ₁)
ex, = unit_vector_fields(coords, dist)
lift_basis  = derivative_basis(basis, 1)
tau_lift(A) = lift(A, lift_basis, -1)
grad_T      = grad(T) + ex * tau_lift(tau1)

problem = IVP([T, tau1, tau2])
add_parameters!(problem; kappa=0.5, grad_T=grad_T, tau_lift=tau_lift)
add_equation!(problem, "∂t(T) - kappa*div(grad_T) + tau_lift(tau2) = 0")

add_bc!(problem, "T(x=0) = 1")   # hot
add_bc!(problem, "T(x=1) = 0")   # cold

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Initial condition: step function
ensure_layout!(T, :g)
x, = local_grids(dist, basis)
get_grid_data(T) .= Float64.(x .< 0.5)
ensure_layout!(T, :c)

run!(solver; stop_iteration=2000, progress=false)   # to t = 2.0, effectively steady
ensure_layout!(T, :g)
```

The field relaxes from the step to the linear profile. Note that the equation string
refers to fields by their **field name** (the `"T"`, `"tau1"`, `"tau2"` you passed to the
constructor) — if the Julia variable and the field name disagree, the parser warns
`Unknown variable` and silently substitutes zero.

!!! warning "1-D pure-Chebyshev IVP: BCs are enforced to O(1/N), not to machine precision"
    With no periodic direction, the tau variables must live on `()`, and the
    time-stepping path enforces the Dirichlet values only to first order in `N`.
    Measured `max|T - (1-x)|` at steady state: `3.3e-2` (N=16), `1.6e-2` (N=32),
    `7.9e-3` (N=64) — i.e. `1/(2N-2)`. The *same* steady problem solved as an LBVP
    (the Laplace Equation section below) hits `2.5e-16`, and adding a periodic
    direction (next section) also gives machine precision. Prefer either of those if
    you need the boundary values to be exact.

### 2D Diffusion

One periodic direction (`x`, Fourier) and one bounded direction (`z`, Chebyshev). The
tau variables carry the Fourier basis, giving one tau per `x`-mode — this enforces the
boundary conditions to machine precision (measured `max|T(z=0) - 1| < 1e-15`).

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
zbasis = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
domain = Domain(dist, (xbasis, zbasis))

T    = ScalarField(domain, "T")
tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)   # one tau per x-mode
tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)

ex, ez      = unit_vector_fields(coords, dist)
lift_basis  = derivative_basis(zbasis, 1)
tau_lift(A) = lift(A, lift_basis, -1)
grad_T      = grad(T) + ez * tau_lift(tau1)

problem = IVP([T, tau1, tau2])
add_parameters!(problem; kappa=0.1, grad_T=grad_T, tau_lift=tau_lift)
add_equation!(problem, "∂t(T) - kappa*div(grad_T) + tau_lift(tau2) = 0")

add_bc!(problem, "T(z=0) = 1")   # hot bottom
add_bc!(problem, "T(z=1) = 0")   # cold top

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

ensure_layout!(T, :g)
x, z = local_grids(dist, xbasis, zbasis)
get_grid_data(T) .= (1 .- z') .+ 0.1 .* sin.(x) .* sin.(π .* z')
ensure_layout!(T, :c)

run!(solver; stop_iteration=100, progress=false)
```

!!! note "Only one bounded direction per domain"
    Tarang couples a single non-periodic axis. A domain with **two** Chebyshev
    directions (a box with Dirichlet walls on all four sides) does not build — the tau
    subproblem is not square and the solve fails with a `DimensionMismatch`. Keep at
    most one Chebyshev axis; make the others Fourier.

## Convection-Diffusion

### Advection by Known Velocity

A constant advecting velocity is just a parameter, and `U*∂x(T)` is linear, so it goes on
the implicit (left) side with the diffusion term.

```julia
problem = IVP([T, tau1, tau2])
add_parameters!(problem; kappa=0.01, U=1.0, grad_T=grad_T, tau_lift=tau_lift)
add_equation!(problem, "∂t(T) + U*∂x(T) - kappa*div(grad_T) + tau_lift(tau2) = 0")

add_bc!(problem, "T(z=0) = 0")
add_bc!(problem, "T(z=1) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

ensure_layout!(T, :g)
get_grid_data(T) .= sin.(x) .* sin.(π .* z')
ensure_layout!(T, :c)

run!(solver; stop_iteration=100, progress=false)
```

(`T`, `tau1`, `tau2`, `grad_T`, `tau_lift`, `x`, `z` are the ones built in the 2D
diffusion example above.)

### Natural Convection

Buoyancy-driven flow with temperature coupling — the Rayleigh-Bénard system. Velocity,
pressure and temperature are solved together; the nonlinear advection terms sit on the
right-hand (explicit) side, everything linear on the left.

```julia
using Tarang

Lx, Lz = 4.0, 1.0
Nx, Nz = 16, 12
Rayleigh, Prandtl = 2e4, 1.0

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

tau_p  = ScalarField(dist, "tau_p",  (), Float64)          # pressure gauge
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

ex, ez      = unit_vector_fields(coords, dist)
lift_basis  = derivative_basis(zbasis, 1)
tau_lift(A) = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * tau_lift(tau_u1)
grad_T = grad(T) + ez * tau_lift(tau_T1)

problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem; nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez,
                grad_u=grad_u, grad_T=grad_T, tau_lift=tau_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + tau_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + tau_lift(tau_u2) = -u⋅∇(u)")

add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "T(z=1) = 0")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")      # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-4)

# Conduction profile + damped noise
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (1.0 .- z')
get_grid_data(T) .+= 1.0 .- z'
ensure_layout!(T, :c)

run!(solver; stop_iteration=20, progress=false)
```

See the [Rayleigh-Bénard tutorial](../tutorials/ivp_2d_rbc.md) for the full-resolution
version with output.

!!! warning "Not available under MPI"
    This example is serial-only, and it fails twice over. First, MPI decomposes the
    *trailing* axes, and a Chebyshev axis cannot be decomposed (its DCT needs the whole
    axis on each rank), so the `(x_fourier, z_chebyshev)` order above errors at `Domain`
    construction: *"Reorder your bases so the Chebyshev axis comes BEFORE the Fourier
    axes."* Second, even after reordering to `(z_chebyshev, x_fourier)`, the `-u⋅∇(T)` /
    `-u⋅∇(u)` advection terms expand to a `∂z` derivative on the Chebyshev axis in the
    *explicit* right-hand side; no rank owns the whole Chebyshev axis, so the solver
    builds and then dies on the first step: *"cannot differentiate along the non-Fourier
    axis of a DISTRIBUTED field ... Move the term to the implicit (L) side, or run in
    serial."* What does run distributed is a Chebyshev-first domain whose explicit
    right-hand side differentiates only along **Fourier** axes.

## Boundary Conditions

These slot into the 2D problem above, in place of its two Dirichlet conditions. They must
be declared with `add_bc!`; the tau variables are what make them enforceable.

### Convective (Robin) BC

Heat transfer to an environment at `T_amb`: `h*T + k*∂T/∂n = h*T_amb`.

```julia
add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "10*T(z=1) + 1*∂z(T)(z=1) = 0")   # h=10, k=1, T_amb=0
```

The steady solution is `T = 1 - (10/11)·z`, giving `T(z=1) = 1/11`. Relaxed from `T = 0`
with `kappa=0.5`, `dt=1e-3` to `t = 10`: `T(z=1) = 0.09090909` and
`max|T - (1 - 10z/11)| = 3.8e-15`. The Robin condition is enforced to machine precision
once the transient has decayed — at `t = 3` the residual is still `2.9e-6`, which is the
slowest mode, not a tau error.

Coefficients may be literals (as above) or names registered with `add_parameters!`. A BC
string cannot see plain Julia globals — an unregistered name warns `Unknown variable` and
is silently treated as zero.

### Insulated (Neumann) BC

Zero heat flux.

```julia
add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "∂z(T)(z=1) = 0")   # ∂T/∂z = 0 at the top
```

With a hot bottom and an insulated top the steady state is uniform, `T = 1` everywhere.
The bottom value is held to machine precision (`max|T(z=0) - 1| < 1e-15`) while the
interior relaxes at the expected rate: started from `T = 0` with `kappa=0.5`,
`max|T - 1| = 3.14e-2` at `t = 3`, against the slowest decaying mode
`(4/π)·exp(-kappa·(π/2)²·t) = 3.14e-2`.

### Time-Varying BC

A BC value may depend on `t` (and on the spatial coordinates). It is re-evaluated every
step — no callback needed.

```julia
add_bc!(problem, "T(z=0) = sin(2*pi*t)")
add_bc!(problem, "T(z=1) = 0")
```

Measured at `t = 0.1`: `T(z=0) = 0.587785`, against `sin(2π·0.1) = 0.587785`.

## Steady-State Problems

### Laplace Equation

Steady heat conduction. This is a linear boundary value problem (LBVP): boundary
conditions are enforced with the **tau method** (one `tau` variable per BC, lifted
into the bulk equation and declared with `add_bc!`). The bounded direction here is
`z` (Chebyshev); the periodic `x` direction is separable.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 1.0))   # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))   # bounded  (coupled)
dom = Domain(dist, (xb, zb))

T    = ScalarField(dom, "T")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)            # one tau per z-BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([T, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
Tarang.add_equation!(problem, "Δ(T) + l1 + l2 = 0")

# Boundary conditions on the bounded (z) direction, via the tau method
Tarang.add_bc!(problem, "T(z=0)   = 1")
Tarang.add_bc!(problem, "T(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
Tarang.ensure_layout!(T, :g)            # scatter writes coefficients; switch to grid
```

Recovers the exact conduction profile: `max|T - (1-z)| = 2.5e-16`.

!!! note "1D pure-Chebyshev BVP"
    The example keeps a periodic `x` direction, but a pure single-axis Chebyshev
    BVP (no Fourier) also works — drop the `x` axis and put the `tau` variables on
    `()`. The solver builds one coupled tau subproblem over the Chebyshev spectrum,
    and it is machine-precision accurate (`max|T - (1-x)| = 2.5e-16`).

### Poisson Equation

With heat source. Like the Laplace example this is a steady LBVP, so the wall
boundary conditions use the **tau method** (`tau` variables + `lift` + `add_bc!`).
The source must sit on the right-hand side.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 1.0))   # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))   # bounded  (coupled)
dom = Domain(dist, (xb, zb))

T    = ScalarField(dom, "T")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)            # one tau per z-BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

# Heat source (e.g., volumetric heating)
q  = ScalarField(dom, "q")
Tarang.ensure_layout!(q, :g)
zg = create_meshgrid(dom; on_device=false)["z"]
get_grid_data(q) .= sin.(π .* zg)                           # source distribution

problem = LBVP([T, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), q=q)
Tarang.add_equation!(problem, "Δ(T) + l1 + l2 = -q")
Tarang.add_bc!(problem, "T(z=0)   = 0")
Tarang.add_bc!(problem, "T(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
Tarang.ensure_layout!(T, :g)
```

Recovers the exact solution `T = sin(πz)/π²` to `4.7e-16`.

The build prints a warning that a "boundary condition right-hand side" of type
`NegateOperator{ScalarField}` is "being enforced as ZERO". For the `-q` *equation* source
this is a false alarm — the source is applied and the boundary conditions are enforced.
Checked by giving the same problem a nonzero wall value (`T(z=0) = 1`, exact solution
`sin(πz)/π² + 1 - z`): the solve reproduces it to `5.6e-16`, holds `T(z=0) = 1.0000000000`,
and differs from the source-free `1 - z` by `1/π²`, so the source is demonstrably there.
The warning is real and worth heeding when it names a genuine *boundary* right-hand side.

## Heat Transfer Analysis

### Nusselt Number

The wall heat flux needs `∂T/∂z` as data. Build the derivative operator with
`d(field, coordinate)` and evaluate it into a field with `evaluate(op, :g)`.

```julia
function compute_nusselt(T, coord, dist, xbasis, zbasis; kappa=1.0, H=1.0, Delta_T=1.0)
    # Vertical temperature gradient as a grid-space field
    dT_dz = get_grid_data(evaluate(d(T, coord), :g))

    # Wall heat flux: horizontally averaged ∂T/∂z at the bottom wall
    _, z  = local_grids(dist, xbasis, zbasis)
    wall  = argmin(Array(z))
    q_wall = -kappa * sum(Array(dT_dz)[:, wall]) / size(dT_dz, 1)

    # Conductive reference
    q_cond = kappa * Delta_T / H

    return q_wall / q_cond
end

# Check against the pure conduction state T = 1 - z, for which Nu must be exactly 1
ensure_layout!(T, :g)
get_grid_data(T) .= 1 .- z'
ensure_layout!(T, :c)

Nu = compute_nusselt(T, coords["z"], dist, xbasis, zbasis)
```

This returns `Nu = 1.0000000000`, as it must.

### Thermal Boundary Layer

The x-averaged profile is a plain reduction over the grid data. Chebyshev grid points are
not ordered monotonically in general, so sort by `z` first.

```julia
function boundary_layer_thickness(T, dist, xbasis, zbasis; frac=0.01)
    ensure_layout!(T, :g)
    data = Array(get_grid_data(T))
    _, z = local_grids(dist, xbasis, zbasis)

    profile = vec(sum(data, dims=1) ./ size(data, 1))   # x-averaged T(z)
    perm    = sortperm(Array(z))
    zs, ps  = Array(z)[perm], profile[perm]

    # δ_T: first height where the profile has decayed to `frac` of its wall value
    idx = findfirst(<(frac * ps[1]), ps)
    return idx === nothing ? nothing : zs[idx]
end
```

For `T = exp(-z/0.1)` on a 24-point Chebyshev grid this gives `δ_T = 0.466`, against the
analytic `-0.1·log(0.01) = 0.461`.

## Multi-Physics

Tarang solves one domain at a time. There is **no** multi-domain or interface coupling,
so conjugate heat transfer (a solid and a fluid region exchanging flux across a shared
boundary) cannot be expressed: two `Domain`s cannot be coupled by a boundary condition.
Likewise there is no enthalpy / phase-change formulation. Both would have to be built on
top of the framework — for example as a single domain with spatially varying
properties — and neither is provided.

## Tips

### Numerical Stability

- Put diffusion on the implicit (left) side — that is what lets the IMEX steppers take a
  timestep unconstrained by the diffusive limit.
- Advection by a *known* velocity is linear (`U*∂x(T)`), so it goes implicit too.
- Nonlinear advection (`-u⋅∇(T)`) must be explicit, and the timestep is then CFL-limited.
  Drive it with the `CFL` controller: `cfl = CFL(solver; ...)`, `add_velocity!(cfl, u)`,
  `run!(solver; cfl=cfl)`.
- At high Peclet number the only remedies are resolution and dealiasing (`dealias=3/2` on
  the Fourier bases, which is the default). Tarang has no upwinding — it is a spectral
  code.

### Convergence

- Exponential convergence for smooth solutions.
- Check spectral coefficient decay.
- There is no local mesh refinement. A Chebyshev grid already clusters points at the
  walls, so a boundary layer is resolved by raising the Chebyshev `size`, not by
  refining a region.

## See Also

- [Rayleigh-Bénard Tutorial](../tutorials/ivp_2d_rbc.md)
- [Fluid Dynamics Examples](fluid_dynamics.md)
- [Example Gallery](gallery.md)
