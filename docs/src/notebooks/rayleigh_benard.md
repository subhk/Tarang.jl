# Notebook: Rayleigh-Bénard Convection

This notebook demonstrates 2D Rayleigh-Bénard convection simulation.

## Setup

```julia
using Tarang
using MPI
using Plots

MPI.Init()
```

## Problem Definition

Rayleigh-Bénard convection simulates fluid heated from below in a gravity field.

### Parameters

```julia
Ra = 1e6    # Rayleigh number
Pr = 1.0    # Prandtl number
Lx = 4.0    # Domain width
Lz = 1.0    # Domain height
```

### Domain

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)

Nx, Nz = 256, 64
x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=1.5)
z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))

domain = Domain(dist, (x_basis, z_basis))
```

### Fields

```julia
ux = ScalarField(dist, "ux", (x_basis, z_basis), Float64)
uz = ScalarField(dist, "uz", (x_basis, z_basis), Float64)
p = ScalarField(dist, "p", (x_basis, z_basis), Float64)
T = ScalarField(dist, "T", (x_basis, z_basis), Float64)
```

### Equations

```julia
problem = IVP([ux, uz, p, T])
problem.parameters["Ra"] = Ra
problem.parameters["Pr"] = Pr

Tarang.add_equation!(problem,
    "∂t(ux) + ux*∂x(ux) + uz*∂z(ux) + ∂x(p) = Pr*Δ(ux)")
Tarang.add_equation!(problem,
    "∂t(uz) + ux*∂x(uz) + uz*∂z(uz) + ∂z(p) = Pr*Δ(uz) + Ra*Pr*T")
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
Tarang.add_equation!(problem,
    "∂t(T) + ux*∂x(T) + uz*∂z(T) = Δ(T)")
```

### Boundary Conditions

```julia
# No-slip walls
Tarang.add_equation!(problem, "ux(z=0) = 0")
Tarang.add_equation!(problem, "ux(z=1) = 0")
Tarang.add_equation!(problem, "uz(z=0) = 0")
Tarang.add_equation!(problem, "uz(z=1) = 0")

# Fixed temperatures
Tarang.add_equation!(problem, "T(z=0) = 1")  # Hot
Tarang.add_equation!(problem, "T(z=1) = 0")  # Cold
```

## Initial Conditions

```julia
# Random perturbations to trigger instability
Tarang.ensure_layout!(T, :g)
T.data_g .= 0.5 .+ 0.01 .* randn(size(T.data_g))
Tarang.ensure_layout!(T, :c)
```

## Solver

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-4)

# CFL condition
cfl = CFL(problem; safety=0.5)
add_velocity!(cfl, ux)
add_velocity!(cfl, uz)
```

## Simulation

```julia
t_end = 1.0
dt_output = 0.1

times = Float64[]
energies = Float64[]

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)

    push!(times, solver.sim_time)
    push!(energies, compute_kinetic_energy([ux, uz]))

    if solver.iteration % 100 == 0
        println("t = $(solver.sim_time)")
    end
end
```

## Visualization

```julia
# Temperature field
Tarang.ensure_layout!(T, :g)
heatmap(T.data_g',
    xlabel="x", ylabel="z",
    title="Temperature at t=$(solver.sim_time)",
    colorbar=true
)
```

## Analysis

```julia
# Kinetic energy evolution
plot(times, energies,
    xlabel="Time",
    ylabel="Kinetic Energy",
    title="Energy Evolution"
)
```

## Cleanup

```julia
MPI.Finalize()
```

## Exercises

1. **Vary Rayleigh number**: Try Ra = 10^4, 10^5, 10^7
2. **Change aspect ratio**: Lx = 2, 8, 16
3. **Compute Nusselt number**: Track heat transfer efficiency
4. **Resolution study**: How do results change with Nx, Nz?

## References

- Chandrasekhar, S. (1961). Hydrodynamic and Hydromagnetic Stability
- Spectral methods tutorials
