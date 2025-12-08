# Surface and Boundary Dynamics

This tutorial covers solving advection-diffusion equations on domain boundaries, including Surface Quasi-Geostrophic (SQG) dynamics and general boundary-coupled systems.

## Overview

Many physical systems involve dynamics confined to surfaces or boundaries:

- **Surface Quasi-Geostrophic (SQG)**: Ocean/atmosphere surface temperature dynamics
- **Quasi-Geostrophic (QG)**: Coupled surface buoyancy with interior PV
- **Passive tracers**: Concentration advected by prescribed flow
- **Reactive surfaces**: Chemical species on catalytic surfaces

Tarang provides a flexible `BoundaryAdvectionDiffusion` framework for all these cases.

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

## Velocity Sources

The key difference between problems is how the velocity `u` is obtained:

### 1. Self-Derived Velocity (SQG-like)

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
    Nx=256, Ny=256,
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

### 2. Interior-Derived Velocity (QG-like)

Velocity extracted from a 3D interior field at the boundary:

```julia
# QG setup: velocity from interior streamfunction
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=128, Ny=128,
    boundaries=[
        BoundarySpec("bottom", :z, 0.0),
        BoundarySpec("top", :z, 1.0)
    ],
    velocity_source=InteriorDerivedVelocity(:perp_grad),  # u = ∇⊥ψ|_{surface}
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5),
    interior_coupling=(
        Nz=32,
        H=1.0,
        equation="Δ(ψ) + S*∂z(∂z(ψ)) = q",  # QG elliptic operator
        params=Dict("S" => 0.01, "q" => 0.0)
    )
)
```

### 3. Prescribed Velocity (Passive Tracer)

Velocity set externally by the user each timestep:

```julia
# Passive tracer in prescribed flow
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=128, Ny=128,
    boundaries=[BoundarySpec("concentration", :z, 0.0)],
    velocity_source=PrescribedVelocity(),
    diffusion=DiffusionSpec(type=:laplacian, coefficient=0.01)
)

# In time loop, set velocity before stepping
function set_velocity!(bad, t)
    # Example: rotating flow
    x, y = get_coordinates(bad)
    bad.velocities["concentration"].components[1].data_g .= -sin.(y)
    bad.velocities["concentration"].components[2].data_g .= sin.(x)
end
```

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

### No Diffusion

```julia
DiffusionSpec(type=:none)
# Inviscid dynamics
```

## Complete SQG Example

```julia
using Tarang

# Domain and resolution
Lx, Ly = 2π, 2π
Nx, Ny = 256, 256

# Create SQG system
sqg = boundary_advection_diffusion_setup(
    Lx=Lx, Ly=Ly,
    Nx=Nx, Ny=Ny,
    boundaries=[BoundarySpec("theta", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
)

# Initial condition: random perturbation
θ = sqg.fields["theta"]
ensure_layout!(θ, :g)

# Create elliptical vortex
x = range(0, Lx, length=Nx)
y = range(0, Ly, length=Ny)
X = [xi for xi in x, _ in y]
Y = [yj for _ in x, yj in y]

θ.data_g .= exp.(-((X .- π).^2 ./ 0.5 .+ (Y .- π).^2 ./ 0.3))

# Time stepping
dt = 0.001
nsteps = 10000
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

## Multiple Boundaries (QG-like)

For problems with dynamics at multiple surfaces:

```julia
# Two-surface QG system
qg = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=128, Ny=128,
    boundaries=[
        BoundarySpec("theta_bot", :z, 0.0; field_name="θ_bottom"),
        BoundarySpec("theta_top", :z, 1.0; field_name="θ_top")
    ],
    velocity_source=InteriorDerivedVelocity(:perp_grad),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5),
    interior_coupling=(
        Nz=32,
        H=1.0,
        equation="Δ(ψ) + S*∂z(∂z(ψ)) = 0",
        params=Dict("S" => (1.0/10.0)^2)  # (f₀/N)²
    )
)

# Set different initial conditions at each surface
qg.fields["theta_bot"].data_g .= initial_bottom
qg.fields["theta_top"].data_g .= initial_top

# Both surfaces evolve together
for step in 1:nsteps
    bad_step!(qg, dt)
end
```

## Adding Source Terms

Custom source terms can be added:

```julia
# Gaussian forcing centered at (π, π)
function my_forcing(bad, boundary_name)
    θ = bad.fields[boundary_name]
    Nx, Ny = size(θ.data_g)

    x = range(0, bad.params["Lx"], length=Nx)
    y = range(0, bad.params["Ly"], length=Ny)

    forcing = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        r2 = (x[i] - π)^2 + (y[j] - π)^2
        forcing[i, j] = 0.1 * exp(-r2 / 0.5)
    end

    return forcing
end

bad_add_source!(sqg, "theta", my_forcing)
```

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

## Diagnostics

### Energy

```julia
E = bad_energy(sqg)  # Total L² energy across all boundaries
```

### Enstrophy

```julia
Z = bad_enstrophy(sqg, "theta")  # Squared gradient of specific field
```

### Maximum Velocity

```julia
umax = bad_max_velocity(sqg)
```

### CFL Timestep

```julia
dt_cfl = bad_cfl_dt(sqg; safety=0.5)
```

## Adaptive Timestepping

```julia
for step in 1:nsteps
    # Compute CFL-limited timestep
    dt = bad_cfl_dt(sqg; safety=0.4)
    dt = min(dt, dt_max)  # Cap maximum timestep

    bad_step!(sqg, dt)

    sqg.time  # Current simulation time
end
```

## MPI Parallelism

The framework supports MPI domain decomposition:

```julia
sqg = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=512, Ny=512,
    boundaries=[BoundarySpec("theta", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5),
    mesh=(4, 4)  # 4×4 = 16 MPI processes
)
```

Run with:
```bash
mpiexec -n 16 julia my_sqg_simulation.jl
```

## Comparison with Dedicated QGSystem

For full QG with interior PV dynamics, you can also use the specialized `QGSystem`:

```julia
# Using dedicated QGSystem (more features for QG-specific problems)
qg = qg_system_setup(
    Lx=2π, Ly=2π, H=1.0,
    Nx=128, Ny=128, Nz=32,
    f0=1.0, N=10.0,
    κ=1e-4, α=0.5
)

qg_step!(qg, dt; timestepper=:RK4)
E = qg_energy(qg)
```

The `BoundaryAdvectionDiffusion` framework is more general and flexible, while `QGSystem` provides QG-specific functionality.

## Summary

| Use Case | Velocity Source | Example |
|----------|-----------------|---------|
| SQG | `SelfDerivedVelocity` | Surface buoyancy dynamics |
| QG | `InteriorDerivedVelocity` | Coupled surface-interior |
| Passive tracer | `PrescribedVelocity` | Tracer in known flow |

## See Also

- [Operators](../pages/operators.md): Fractional Laplacian details
- [Problems](../pages/problems.md): IVP and LBVP setup
- [Solvers](../pages/solvers.md): Timestepping and BVP solvers
