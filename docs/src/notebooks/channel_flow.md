# Notebook: Channel Flow

This notebook demonstrates pressure-driven channel flow simulation.

## Overview

Channel flow (plane Poiseuille flow) is a fundamental benchmark for viscous flow simulations. Fluid flows between two parallel plates driven by a pressure gradient.

## Setup

```julia
using Tarang
using MPI
using Plots

MPI.Init()
```

## Parameters

```julia
Re = 1000.0     # Reynolds number
dpdx = -1.0     # Pressure gradient (driving force)
nu = 1.0 / Re   # Kinematic viscosity
Lx = 4π         # Domain length
Lz = 2.0        # Channel height
```

## Domain

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)

Nx, Nz = 128, 64
x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=1.5)
z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))

domain = Domain(dist, (x_basis, z_basis))
```

## Fields

```julia
ux = ScalarField(dist, "ux", (x_basis, z_basis), Float64)
uz = ScalarField(dist, "uz", (x_basis, z_basis), Float64)
p = ScalarField(dist, "p", (x_basis, z_basis), Float64)
```

## Problem Definition

```julia
problem = IVP([ux, uz, p])
problem.parameters["nu"] = nu
problem.parameters["dpdx"] = dpdx

# Momentum equations
Tarang.add_equation!(problem,
    "∂t(ux) + ux*∂x(ux) + uz*∂z(ux) + ∂x(p) = nu*Δ(ux) - dpdx")
Tarang.add_equation!(problem,
    "∂t(uz) + ux*∂x(uz) + uz*∂z(uz) + ∂z(p) = nu*Δ(uz)")

# Continuity
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
```

## Boundary Conditions

```julia
# No-slip at walls
Tarang.add_equation!(problem, "ux(z=0) = 0")    # Bottom
Tarang.add_equation!(problem, "ux(z=$Lz) = 0")  # Top
Tarang.add_equation!(problem, "uz(z=0) = 0")
Tarang.add_equation!(problem, "uz(z=$Lz) = 0")
```

## Analytical Solution

For laminar flow, the exact solution is parabolic:

```julia
# Poiseuille profile
function poiseuille_profile(z, H, dpdx, nu)
    return -dpdx / (2*nu) * z * (H - z)
end

# Maximum velocity
u_max = -dpdx * Lz^2 / (8*nu)
println("Expected u_max = $u_max")
```

## Initial Conditions

```julia
# Start from Poiseuille profile with perturbation
z_grid = get_grid(z_basis)
Tarang.ensure_layout!(ux, :g)

for i in 1:Nx, j in 1:Nz
    ux.data_g[i, j] = poiseuille_profile(z_grid[j], Lz, dpdx, nu)
end

# Add small perturbation
ux.data_g .+= 0.01 .* randn(size(ux.data_g))
Tarang.ensure_layout!(ux, :c)
```

## Solver

```julia
solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)

cfl = CFL(problem; safety=0.4)
add_velocity!(cfl, ux)
add_velocity!(cfl, uz)
```

## Simulation

```julia
t_end = 10.0

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)

    if solver.iteration % 100 == 0
        Tarang.ensure_layout!(ux, :g)
        u_centerline = mean(ux.data_g[:, Nz÷2])
        println("t = $(solver.sim_time), u_center = $u_centerline")
    end
end
```

## Results

### Velocity Profile

```julia
Tarang.ensure_layout!(ux, :g)

# Average over x
u_profile = mean(ux.data_g, dims=1)[:]
z_points = get_grid(z_basis)

# Analytical
u_analytical = [poiseuille_profile(z, Lz, dpdx, nu) for z in z_points]

plot(u_profile, z_points, label="Numerical")
plot!(u_analytical, z_points, label="Analytical", linestyle=:dash)
xlabel!("u")
ylabel!("z")
title!("Velocity Profile")
```

### Error Analysis

```julia
error = maximum(abs.(u_profile .- u_analytical))
println("Maximum error: $error")
```

## Turbulent Channel (Higher Re)

For turbulent flow, increase Reynolds number:

```julia
Re_turb = 5000
nu_turb = 1.0 / Re_turb

# Will need:
# - Higher resolution (Nx=256, Nz=128)
# - Longer integration time
# - Statistical averaging
```

## Exercises

1. **Vary Reynolds number**: Compare Re = 100, 500, 1000
2. **Convergence study**: How does error scale with Nz?
3. **Add turbulence**: Re = 5000 with statistics
4. **Couette flow**: Moving top wall instead of pressure gradient

## Cleanup

```julia
MPI.Finalize()
```

## References

- Pope, S.B. (2000). Turbulent Flows
- Kim, J., Moin, P., Moser, R. (1987). Turbulence statistics in channel flow
