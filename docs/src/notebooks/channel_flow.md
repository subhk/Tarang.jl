# Notebook: Channel Flow

This notebook demonstrates pressure-driven channel flow simulation.

## Overview

Channel flow (plane Poiseuille flow) is a fundamental benchmark for viscous flow simulations. Fluid flows between two parallel plates driven by a pressure gradient.

The walls are resolved with a Chebyshev basis, the streamwise direction with a Fourier basis. Because the momentum equation carries the advection term `-u⋅∇(u)` on the explicit side, and that term contains a Chebyshev derivative, **this example is serial only** — a non-Fourier derivative in the explicit right-hand side is not supported under MPI (the solver raises a clear error on the first step). See [Running with MPI](../getting_started/running_with_mpi.md).

## Setup

```julia
using Tarang
using Statistics   # for mean()
```

## Parameters

```julia
Re   = 100.0            # Reynolds number (centreline velocity, half-height)
Lx   = 4π               # Streamwise period
Lz   = 2.0              # Channel height
nu   = 1.0 / Re         # Kinematic viscosity
dpdx = -8 * nu / Lz^2   # Driving pressure gradient (chosen so that u_max = 1)
```

## Domain

The Fourier axis comes first, the Chebyshev axis last.

```julia
Nx, Nz = 32, 24

coords  = CartesianCoordinates("x", "z")
dist    = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain  = Domain(dist, (x_basis, z_basis))
```

## Fields

The velocity is a `VectorField`; the pressure is a `ScalarField`. The tau fields carry
the boundary-condition unknowns: one tau per streamwise mode (hence the `(x_basis,)`
basis tuple), plus a scalar gauge unknown `tau_p` for the pressure.

```julia
p = ScalarField(domain, "p")
u = VectorField(domain, "u")

tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), Float64)
```

## Problem Definition

The wall-normal derivative is reduced to first order with a lift term, which is what makes
the tau method work: `grad_u = ∇u + ẑ·lift(τ₁)`.

```julia
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(z_basis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u     = grad(u) + ez * τ_lift(tau_u1)

problem = IVP([p, u, tau_p, tau_u1, tau_u2])
add_parameters!(problem, nu=nu, dpdx=dpdx, ex=ex, ez=ez,
                grad_u=grad_u, τ_lift=τ_lift)

# Continuity (with pressure gauge)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Momentum, driven by the imposed pressure gradient
add_equation!(problem,
    "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u) - dpdx*ex")
```

Names used inside an equation string must be registered with `add_parameters!` — a string
expression cannot see plain Julia globals (an unregistered name is silently substituted
with 0).

## Boundary Conditions

Boundary conditions are declared with `add_bc!`, never with `add_equation!`: only `add_bc!`
registers the condition with the boundary-condition manager. `$Lz` is interpolated by Julia
before parsing, so the string the parser sees contains a numeric literal.

```julia
# No-slip at both walls
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=$Lz) = 0")

# Pressure gauge (the pressure is only defined up to a constant)
add_bc!(problem, "integ(p) = 0")
```

## Analytical Solution

For laminar flow, the exact solution is parabolic:

```julia
poiseuille(z) = -dpdx / (2 * nu) * z * (Lz - z)

u_max = -dpdx * Lz^2 / (8 * nu)
println("Expected u_max = $u_max")     # 1.0
```

## Initial Conditions

`local_grids` returns this rank's grid vectors for each axis — use it instead of a global
meshgrid.

```julia
xg, zg = local_grids(dist, x_basis, z_basis)
ux = u.components[1]

fill_random!(ux, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(ux) .*= zg' .* (Lz .- zg')   # damp the perturbation at the walls
get_grid_data(ux) .+= poiseuille.(zg')     # laminar base flow
ensure_layout!(u, :c)
```

## Solver

`CFL` is constructed from the **solver**, and velocities are added as `VectorField`s. The
controller is then handed to `run!`, which applies it every `cadence` iterations.

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

cfl = CFL(solver; initial_dt=1e-3, cadence=10, safety=0.4, max_dt=0.01)
add_velocity!(cfl, u)
```

## Simulation

Callbacks are `(interval, function)` tuples; the function receives the solver.

```julia
report = s -> begin
    ensure_layout!(u, :g)
    uc = mean(get_grid_data(ux)[:, Nz ÷ 2])
    println("t = $(round(s.sim_time, digits=4)), u_center = $(round(uc, digits=6))")
end

run!(solver; stop_iteration=50, cfl=cfl, callbacks=[(25, report)], progress=false)
```

Output of the run above:

```
t = 0.169, u_center = 0.995211
t = 0.419, u_center = 0.995268
```

## Results

### Velocity Profile

```julia
ensure_layout!(u, :g)

# Average over x
u_profile    = mean(get_grid_data(ux), dims=1)[:]
u_analytical = poiseuille.(zg)
```

`u_profile` and `zg` are ordinary vectors, ready for the plotting package of your choice.

### Error Analysis

```julia
error = maximum(abs.(u_profile .- u_analytical))
println("Maximum error: $error")     # 0.000124…
```

The residual is the wall-damped perturbation decaying back to the laminar state. Starting
from the exact Poiseuille profile with no perturbation, the same run reproduces it to
`5.6e-16`, and the no-slip conditions hold to `1e-16` — the discrete steady state is the
analytical one.

## Turbulent Channel (Higher Re)

Turbulent flow needs a much larger Reynolds number (`Re = 5000`, `nu = 1/Re`), and with it a
finer grid (`Nx = 256`, `Nz = 128`), a longer integration, and statistical averaging over the
homogeneous `x` direction and over time. That is a production-scale run, not a notebook demo.

## Exercises

1. **Vary Reynolds number**: Compare Re = 100, 500, 1000
2. **Convergence study**: How does error scale with Nz?
3. **Add turbulence**: Re = 5000 with statistics
4. **Couette flow**: Moving top wall instead of pressure gradient

## References

- Pope, S.B. (2000). Turbulent Flows
- Kim, J., Moin, P., Moser, R. (1987). Turbulence statistics in channel flow
