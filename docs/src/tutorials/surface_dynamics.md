# Surface and Boundary Dynamics

This tutorial covers solving advection-diffusion equations on domain boundaries, including Surface Quasi-Geostrophic (SQG) dynamics and general boundary-coupled systems.

## Overview

Many physical systems involve dynamics confined to surfaces or boundaries:

- **Surface Quasi-Geostrophic (SQG)**: Ocean/atmosphere surface temperature dynamics
- **Passive tracers**: Concentration advected by prescribed flow
- **Reactive surfaces**: Chemical species on catalytic surfaces

Tarang provides a flexible `BoundaryAdvectionDiffusion` framework for these cases.

!!! warning "Interior coupling is not functional"
    The framework also declares an *interior-coupled* mode (`InteriorDerivedVelocity` +
    `interior_coupling=`), intended for full QG with a 3D streamfunction inversion, along
    with a dedicated `QGSystem`. **Neither currently runs**; the sections below give the
    exact failures. Everything else on this page is verified working, in serial and
    under MPI.

## The Advection-Diffusion Equation

The general equation solved on each boundary is:

```
∂c/∂t + u·∇c = D(c) + S
```

where:
- `c` is the scalar field (buoyancy, concentration, temperature)
- `u` is the advection velocity
- `D(c)` is the diffusion operator
- `S` is an optional source term

The whole system is stepped with **explicit** timesteppers (`bad_step!`); there is no
implicit solve in this module.

## Velocity Sources

The key difference between problems is how the velocity `u` is obtained.

### Self-Derived Velocity (SQG-like)

Velocity computed from the boundary field itself through an inversion:

```
ψ = (-Δ)^α c      (fractional Laplacian inversion)
u = ∇⊥ψ           (perpendicular gradient for incompressibility)
```

For SQG, `α = -1/2`, giving `ψ = (-Δ)^(-1/2) θ`.

```julia
using Tarang

# SQG setup: velocity from buoyancy inversion
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=64, Ny=64,
    boundaries=[BoundarySpec("theta", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(
        inversion_exponent=-0.5,  # SQG: ψ = (-Δ)^(-1/2) θ
        use_perp_grad=true        # u = ∇⊥ψ (incompressible)
    ),
    diffusion=DiffusionSpec(
        type=:fractional,
        coefficient=1e-4,
        exponent=0.5              # Physical SQG dissipation
    )
)
```

This is the path exercised by the complete SQG example below.

### Interior-Derived Velocity

The intent of `InteriorDerivedVelocity(:perp_grad)` together with `interior_coupling=` is
a full QG system: the boundary fields provide Neumann boundary conditions for a 3D
elliptic problem, and the velocity is `∇⊥ψ` evaluated at each surface.

**This mode does not currently work.** `boundary_advection_diffusion_setup` builds the
interior LBVP *without installing any boundary conditions on it*, so the first
`bad_step!` — which solves that LBVP — fails validation:

```
ArgumentError: Problem validation failed:
Boundary value problem requires boundary conditions
```

Do not use `interior_coupling=` until that is fixed. For surface-only QG dynamics, use
`SelfDerivedVelocity` (the SQG limit), which is fully functional.

### Prescribed Velocity (Passive Tracer)

Velocity set externally by the user each timestep:

```julia
# Passive tracer in prescribed flow
tracer = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=64, Ny=64,
    boundaries=[BoundarySpec("concentration", :z, 0.0)],
    velocity_source=PrescribedVelocity(),
    diffusion=DiffusionSpec(type=:laplacian, coefficient=0.01)
)

# Set the velocity before each step. `local_grids` returns this rank's slice of each
# axis, so the same code is correct in serial and under MPI.
function set_velocity!(bad, t)
    x, y = local_grids(bad.dist, bad.bases...)
    u = bad.velocities["concentration"]
    ensure_layout!(u.components[1], :g)
    ensure_layout!(u.components[2], :g)
    get_grid_data(u.components[1]) .= -sin.(y') .* ones(length(x))
    get_grid_data(u.components[2]) .=  sin.(x)  .* ones(length(y))'
end

c = tracer.fields["concentration"]
ensure_layout!(c, :g)
x, y = local_grids(tracer.dist, tracer.bases...)
get_grid_data(c) .= exp.(-((x .- π).^2 .+ (y' .- π).^2))

for step in 1:10
    set_velocity!(tracer, tracer.time)
    bad_step!(tracer, 0.005)
end
```

After 10 steps this gives `bad_energy(tracer) = 0.0397`, `bad_max_velocity(tracer) = √2`
(the rotating flow above has `|u| ≤ √2`).

## Field access

A `BoundaryAdvectionDiffusion` exposes its state through plain dictionaries:

| accessor | contents |
|---|---|
| `bad.fields[name]` | the `ScalarField` `c` on boundary `name` |
| `bad.velocities[name]` | the `VectorField` `u` on boundary `name` |
| `bad.dist`, `bad.bases`, `bad.coords` | distributor, `(x_basis, y_basis)`, coordinates |
| `bad.params` | `"Lx"`, `"Ly"`, `"κ"`, `"α"` |
| `bad.time`, `bad.iteration` | simulation time and step count |

Grid coordinates come from `local_grids(bad.dist, bad.bases...)`, which returns each
rank's *local* x and y vectors. Broadcast them (`x` against `y'`) rather than building a
global meshgrid — that keeps the code correct under MPI.

## Diffusion Types

### Standard Laplacian

```julia
DiffusionSpec(type=:laplacian, coefficient=0.01)
# Gives: κΔc
```

### Fractional Laplacian

```julia
DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
# Gives: -κ(-Δ)^α c
```

Common exponents:
- `α = 0.5`: Physical SQG dissipation
- `α = 1.0`: Standard Laplacian (equivalent to `:laplacian`)
- `α = 2.0`: Biharmonic (hyperviscosity)

### Hyperdiffusion

```julia
DiffusionSpec(type=:hyperdiffusion, coefficient=1e-8, exponent=2.0)
# Gives: -κ(-Δ)^n c (typically n=2 for biharmonic)
```

`:hyperdiffusion` and `:fractional` share the same implementation; the two names exist
only to document intent.

### No Diffusion

```julia
DiffusionSpec(type=:none)
# Inviscid dynamics
```

!!! note "`implicit=true` is applied explicitly"
    `DiffusionSpec` accepts an `implicit` flag, but this module has only explicit
    timesteppers. Setting `implicit=true` emits a warning and integrates the diffusion
    term explicitly anyway. Choose `dt` accordingly, or use the main solver framework
    (`IVP` + `InitialValueSolver`) if you need a true implicit diffusion solve.

## Complete SQG Example

```julia
using Tarang

# Domain and resolution
Lx, Ly = 2π, 2π
Nx, Ny = 64, 64

# Create SQG system
sqg = boundary_advection_diffusion_setup(
    Lx=Lx, Ly=Ly,
    Nx=Nx, Ny=Ny,
    boundaries=[BoundarySpec("theta", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
)

# Initial condition: an elliptical vortex
θ = sqg.fields["theta"]
ensure_layout!(θ, :g)
x, y = local_grids(sqg.dist, sqg.bases...)
get_grid_data(θ) .= exp.(-((x .- π).^2 ./ 0.5 .+ (y' .- π).^2 ./ 0.3))

# Time stepping
dt = 0.001
nsteps = 200
output_interval = 100

for step in 1:nsteps
    bad_step!(sqg, dt; timestepper=:RK4)

    if step % output_interval == 0
        energy = bad_energy(sqg)
        max_vel = bad_max_velocity(sqg)
        println("Step $step: E = $energy, max|u| = $max_vel")
    end
end
```

Output:

```
Step 100: E = 0.015409487302486865, max|u| = 0.47549001329457397
Step 200: E = 0.015408863536076695, max|u| = 0.4755762567027247
```

Raise `Nx`, `Ny` and `nsteps` for production runs; the numbers above are just a
quick, reproducible check that the vortex is being advected and slowly dissipated.

## Multiple Boundaries

A setup may carry several boundaries. Each gets its own field, its own velocity, and its
own advection-diffusion equation, and `bad_step!` advances them all together:

```julia
two = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=32, Ny=32,
    boundaries=[
        BoundarySpec("theta_bot", :z, 0.0),
        BoundarySpec("theta_top", :z, 1.0)
    ],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
)

# Different initial condition on each surface
x, y = local_grids(two.dist, two.bases...)
for (name, amp) in (("theta_bot", 1.0), ("theta_top", 0.5))
    f = two.fields[name]
    ensure_layout!(f, :g)
    get_grid_data(f) .= amp .* exp.(-((x .- π).^2 .+ (y' .- π).^2) ./ 0.3)
end

for step in 1:50
    bad_step!(two, 0.001)
end

bad_energy(two)                      # 0.00746 — pooled over both surfaces
bad_enstrophy(two, "theta_bot")      # 0.0796 — one named surface at a time
bad_enstrophy(two, "theta_top")      # 0.0199
```

With `SelfDerivedVelocity` each surface inverts **its own** field, so the two surfaces
evolve independently — they are two SQG problems sharing a timestepper. Coupling them
through a shared interior is what `interior_coupling=` is for, and that path is not
functional (see above).

`BoundarySpec` also takes a `field_name` keyword, which renames the underlying field
without changing the dictionary key: `BoundarySpec("theta_bot", :z, 0.0;
field_name="θ_bottom")` is still reached as `bad.fields["theta_bot"]`.

## Adding Source Terms

A source function receives the system and the boundary name, and returns an array shaped
like that rank's grid slab:

```julia
# Gaussian forcing centered at (π, π)
function my_forcing(bad, boundary_name)
    x, y = local_grids(bad.dist, bad.bases...)
    return 0.1 .* exp.(-((x .- π).^2 .+ (y' .- π).^2) ./ 0.5)
end

bad_add_source!(sqg, "theta", my_forcing)
```

The array is added to the RHS of that boundary's equation on every stage of every step.
The function is re-evaluated each time, so it may depend on `bad.time` for time-dependent
forcing — but note that `bad.time` is only advanced *after* the step completes, so all
stages within one step see the same `t`. The forcing is first-order accurate in time.

## Available Timesteppers

| Method | Order | Description |
|--------|-------|-------------|
| `:Euler` | 1 | Forward Euler |
| `:RK2` | 2 | Midpoint method |
| `:RK4` | 4 | Classical Runge-Kutta |
| `:SSPRK3` | 3 | Strong Stability Preserving RK3 |

```julia
bad_step!(sqg, dt; timestepper=:SSPRK3)
```

`:RK4` is the default. Anything else throws `ArgumentError: Unknown timestepper`.

## Diagnostics

All four reduce correctly under MPI (they perform their own `Allreduce`) — do not wrap
them in a collective yourself. The corollary is that they are **collective calls**: every
rank must reach them, so never hide one behind a rank guard.

```julia
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# WRONG — rank 0 enters the Allreduce alone. The ranks desynchronize: the job either
# hangs forever, or aborts once another rank reaches a different collective
#   (Fatal error in internal_Allreduce: Message truncated).
if rank == 0
    println("E = ", bad_energy(sqg))
end

# RIGHT — all ranks compute, one rank prints
E = bad_energy(sqg)
rank == 0 && println("E = ", E)
```

### Energy

```julia
E = bad_energy(sqg)  # mean square of the boundary fields, averaged over all boundaries
```

### Enstrophy

```julia
Z = bad_enstrophy(sqg, "theta")  # mean squared gradient of one named field
```

### Maximum Velocity

```julia
umax = bad_max_velocity(sqg)
```

### CFL Timestep

```julia
dt_cfl = bad_cfl_dt(sqg; safety=0.5)   # safety * min(dx, dy) / max|u|
```

`bad_cfl_dt` returns `Inf` when the velocity is identically zero, so cap it before use.

## Adaptive Timestepping

```julia
dt_max = 0.01

for step in 1:nsteps
    # CFL-limited timestep, capped from above
    dt_next = min(bad_cfl_dt(sqg; safety=0.4), dt_max)

    bad_step!(sqg, dt_next)
end

sqg.time        # current simulation time
sqg.iteration   # number of completed steps
```

## MPI Parallelism

The SQG / prescribed-velocity paths run under MPI unchanged. The boundary domain is
2-dimensional, so it takes a **slab** decomposition: pass a mesh with a unit factor, such
as `(nprocs, 1)`. A genuinely 2-D mesh like `(2, 2)` decomposes *both* axes, leaves no
local axis for the FFT, and fails hard with
`PencilFFT plan creation failed with 4 MPI processes`.

```julia
using Tarang
using MPI
MPI.Initialized() || MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)

sqg = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=64, Ny=64,
    boundaries=[BoundarySpec("theta", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5),
    mesh=(np, 1)          # slab decomposition over `np` processes
)

θ = sqg.fields["theta"]
ensure_layout!(θ, :g)
x, y = local_grids(sqg.dist, sqg.bases...)   # this rank's slice of each axis
get_grid_data(θ) .= exp.(-((x .- π).^2 ./ 0.5 .+ (y' .- π).^2 ./ 0.3))

for step in 1:100
    bad_step!(sqg, 0.001; timestepper=:RK4)
end

# bad_energy / bad_max_velocity do the MPI reduction internally
println("E = ", bad_energy(sqg), "  max|u| = ", bad_max_velocity(sqg))
```

Run with:
```bash
mpiexec -n 4 julia --project=. my_sqg_simulation.jl
```

The energy is bit-identical at 1, 2 and 4 processes
(`E = 0.015409487302486865`), matching the serial run above.

Two rules to keep the distributed version correct:

- Build initial conditions from `local_grids(...)`, never from a global `range`/meshgrid.
- Never wrap `bad_energy` / `bad_enstrophy` / `bad_max_velocity` / `bad_cfl_dt` in an
  `MPI.Allreduce`; they already reduce globally.

## The dedicated `QGSystem`

`qg_system_setup`, `qg_step!`, `qg_invert!` and `qg_energy` are exported and are intended
to provide full QG with interior PV inversion. **They do not currently work.**
`qg_system_setup` builds the interior inversion problem with parameters written directly
into `problem.parameters` (invisible to the equation parser, which warns
`Unknown variable: S / N / f0`) and declares the surface Neumann conditions with
`add_equation!` instead of `add_bc!`. The result is a non-square operator
(1440×1152 at `Nx=Ny=16, Nz=8`) and `qg_step!` throws `DimensionMismatch`.

Use the `BoundaryAdvectionDiffusion` framework with `SelfDerivedVelocity` — the SQG limit
of QG — until this is repaired.

## Summary

| Use Case | Velocity Source | Status |
|----------|-----------------|--------|
| SQG (surface buoyancy) | `SelfDerivedVelocity` | works, serial and MPI |
| Passive tracer in a known flow | `PrescribedVelocity` | works, serial and MPI |
| QG (coupled surface + interior) | `InteriorDerivedVelocity` | **not functional** |

## See Also

- [Operators](../pages/operators.md): Fractional Laplacian details
- [Problems](../pages/problems.md): IVP and LBVP setup
- [Solvers](../pages/solvers.md): Timestepping and BVP solvers
