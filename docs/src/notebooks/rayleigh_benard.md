# Notebook: Rayleigh-Bénard Convection

This notebook demonstrates 2D Rayleigh-Bénard convection simulation.

## Setup

```julia
using Tarang
using Printf
using Plots
```

## Problem Definition

Rayleigh-Bénard convection simulates fluid heated from below in a gravity field.
Non-dimensionalizing with the box height and the thermal diffusion time, the
equations are

```math
\partial_t u + u \cdot \nabla u = -\nabla p + \mathrm{Pr}\, \nabla^2 u + \mathrm{Ra}\,\mathrm{Pr}\, T \hat{z},
\qquad
\partial_t T + u \cdot \nabla T = \nabla^2 T,
\qquad
\nabla \cdot u = 0 .
```

### Parameters

```julia
Ra = 2e4    # Rayleigh number
Pr = 1.0    # Prandtl number
Lx = 4.0    # Domain width
Lz = 1.0    # Domain height

Nx, Nz = 64, 16
```

The resolution is deliberately small so the notebook runs in under a minute.
Production runs use `Nx, Nz = 256, 64` and `Ra = 2e6` (see
`examples/ivp/rayleigh_benard_2d.jl`).

### Domain

The wall-bounded vertical direction is Chebyshev; the periodic horizontal
direction is Fourier. In serial the Fourier axis comes first.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
z_basis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)

domain = Domain(dist, (x_basis, z_basis))
```

`dealias=3/2` is a *padding factor*: it applies the 3/2 (a.k.a. 2/3) rule.
Passing `dealias=2/3` would switch dealiasing off.

### Fields

Velocity is a single `VectorField`, not two scalars — the parser's `∇`, `div`
and `u⋅∇(u)` all act on vectors, and `CFL` only accepts vector velocities.

```julia
p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")
```

### Tau fields and the first-order reduction

A Chebyshev problem needs tau terms to carry the boundary conditions. Each tau
variable is a rank-reduced field living on the *Fourier* basis only — one tau
per horizontal mode — except the pressure gauge `tau_p`, which is a single
scalar.

```julia
tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), Float64)

ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(z_basis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)

grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)
```

### Equations

Parameters and substitutions are registered with `add_parameters!`; equation
strings are parsed by Tarang and cannot see Julia globals otherwise.

```julia
problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])

add_parameters!(problem, nu=Pr, buoy=Ra*Pr, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")
```

### Boundary Conditions

Boundary conditions go through `add_bc!`, never `add_equation!`. A BC declared
with `add_equation!` is not registered with the boundary-condition manager and
is silently *not enforced* (a constant BC may still come out right by accident,
which is what makes the mistake so easy to miss).

```julia
# No-slip walls
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=$Lz) = 0")

# Fixed temperatures
add_bc!(problem, "T(z=0) = 1")     # Hot
add_bc!(problem, "T(z=$Lz) = 0")   # Cold

# Pressure gauge (the incompressible pressure is defined only up to a constant)
add_bc!(problem, "integ(p) = 0")
```

BC strings can only refer to literals and names registered with
`add_parameters!` — a bare Julia global such as `Lz` inside the string would
warn `Unknown variable: Lz` and be applied as `0`. Interpolating the value with
`$Lz` sidesteps that.

## Initial Conditions

The conduction profile `T = 1 - z` plus a small random perturbation, damped at
the walls so it does not fight the boundary conditions.

```julia
x, z = local_grids(dist, x_basis, z_basis)

fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (1.0 .- z')   # damp noise at the walls
get_grid_data(T) .+= 1.0 .- z'           # add the conduction profile
ensure_layout!(T, :c)
```

## Solver

`CFL` is built from the **solver**, and velocities are added as `VectorField`s.

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-4)

cfl = CFL(solver; initial_dt=1e-4, cadence=10, safety=0.5, max_dt=1e-3)
add_velocity!(cfl, u)
```

The full keyword list is `initial_dt`, `cadence`, `safety`, `threshold`,
`max_change`, `min_change`, `max_dt`. The current step is `cfl.current_dt`;
`min_change`/`max_change` are the per-update *ratio* limits.

## Simulation

```julia
stop_time = 0.02

times    = Float64[]
energies = Float64[]

while solver.sim_time < stop_time
    dt = compute_timestep(cfl)
    step!(solver, dt)

    push!(times, solver.sim_time)
    push!(energies, total_kinetic_energy(u))

    if solver.iteration % 10 == 0
        @printf("iter=%d  t=%.4f  dt=%.2e  KE=%.4e\n",
                solver.iteration, solver.sim_time, dt, energies[end])
    end
end
```

`total_kinetic_energy(u)` returns the domain integral as a `Float64`.
(`kinetic_energy(u)` returns the *field* ½ρ|u|², which is what you want for a
snapshot, not for a time series.) Over the 29 steps above the kinetic energy
grows from `1.5e-7` to `8.6e-5` as the buoyancy instability takes off.

Passing the controller to `run!` is equivalent and shorter:

```julia
run!(solver; stop_time=stop_time, cfl=cfl, progress=false)
```

## Visualization

```julia
ensure_layout!(T, :g)
Tg = Array(get_grid_data(T))

heatmap(x, z, Tg',
    xlabel="x", ylabel="z",
    title="Temperature at t=$(round(solver.sim_time, digits=3))",
    aspect_ratio=:equal, color=:inferno
)
```

`get_grid_data` returns the coefficient array unless the field is in grid
layout, so `ensure_layout!(T, :g)` first. The array is `(Nx, Nz)`, hence the
transpose for `heatmap`.

## Analysis

```julia
plot(times, energies,
    xlabel="Time",
    ylabel="Kinetic Energy",
    title="Energy Evolution"
)
```

A quick check that the tau method did its job — the boundary values are exact
to roundoff (`max|T(z=0) - 1| ≈ 2e-15`):

```julia
@printf("max|T(z=0) - 1| = %.2e\n", maximum(abs, Tg[:, 1]   .- 1.0))
@printf("max|T(z=Lz)|    = %.2e\n", maximum(abs, Tg[:, end]))
```

## Running under MPI

This problem is **serial-only**. The advection terms `-u⋅∇(T)` and `-u⋅∇(u)`
put a Chebyshev derivative on the explicit right-hand side, and a distributed
run cannot take it: each rank owns only part of the Chebyshev axis, so the
first step errors with

> Lazy RHS: cannot differentiate along the non-Fourier axis 1 (ChebyshevT) of a
> DISTRIBUTED field …

Distributed Chebyshev-Fourier runs are supported when every explicit derivative
is along a Fourier axis (and then the Chebyshev axis must come *first* in the
domain). See [Running with MPI](../getting_started/running_with_mpi.md).

## Exercises

1. **Vary Rayleigh number**: Try Ra = 10^4, 10^5, 10^7
2. **Change aspect ratio**: Lx = 2, 8, 16
3. **Compute Nusselt number**: Track heat transfer efficiency,
   `Nu = 1 + integrate(w T) / (Lx * Lz)`
4. **Resolution study**: How do results change with Nx, Nz?

## References

- Chandrasekhar, S. (1961). Hydrodynamic and Hydromagnetic Stability
- Spectral methods tutorials
