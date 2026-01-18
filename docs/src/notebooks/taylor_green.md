# Notebook: Taylor-Green Vortex

This notebook demonstrates the Taylor-Green vortex, a classic benchmark for DNS codes.

## Overview

The Taylor-Green vortex is an unsteady flow with an exact analytical solution that decays in time. It's commonly used to validate spectral codes and study the transition to turbulence.

## Setup

```julia
using Tarang
using MPI
using Plots

MPI.Init()
```

## Parameters

```julia
Re = 100.0       # Reynolds number
nu = 1.0 / Re    # Kinematic viscosity
L = 2π           # Domain size
N = 64           # Resolution
```

## Domain (Triply Periodic)

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)

# All Fourier bases for periodic boundaries
x_basis = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dealias=1.5)
y_basis = RealFourier(coords["y"]; size=N, bounds=(0.0, L), dealias=1.5)
z_basis = RealFourier(coords["z"]; size=N, bounds=(0.0, L), dealias=1.5)

domain = Domain(dist, (x_basis, y_basis, z_basis))
```

## Fields

```julia
ux = ScalarField(dist, "ux", (x_basis, y_basis, z_basis), Float64)
uy = ScalarField(dist, "uy", (x_basis, y_basis, z_basis), Float64)
uz = ScalarField(dist, "uz", (x_basis, y_basis, z_basis), Float64)
p = ScalarField(dist, "p", (x_basis, y_basis, z_basis), Float64)
```

## Problem Definition

```julia
problem = IVP([ux, uy, uz, p])
problem.parameters["nu"] = nu

# Momentum equations
Tarang.add_equation!(problem,
    "∂t(ux) + ux*∂x(ux) + uy*∂y(ux) + uz*∂z(ux) + ∂x(p) = nu*Δ(ux)")
Tarang.add_equation!(problem,
    "∂t(uy) + ux*∂x(uy) + uy*∂y(uy) + uz*∂z(uy) + ∂y(p) = nu*Δ(uy)")
Tarang.add_equation!(problem,
    "∂t(uz) + ux*∂x(uz) + uy*∂y(uz) + uz*∂z(uz) + ∂z(p) = nu*Δ(uz)")

# Continuity
Tarang.add_equation!(problem, "∂x(ux) + ∂y(uy) + ∂z(uz) = 0")
```

## Analytical Solution

Taylor-Green initial condition:

```math
\begin{aligned}
u &= \cos(x) \sin(y) \cos(z) \\
v &= -\sin(x) \cos(y) \cos(z) \\
w &= 0 \\
p &= \frac{1}{16}[\cos(2x) + \cos(2y)][\cos(2z) + 2]
\end{aligned}
```

Time evolution (viscous decay):

```math
u(x,t) = u_0(x) e^{-2\nu t}
```

## Initial Conditions

```julia
function taylor_green_ic!(ux, uy, uz)
    x = get_grid(ux.bases[1])
    y = get_grid(ux.bases[2])
    z = get_grid(ux.bases[3])

    Tarang.ensure_layout!(ux, :g)
    Tarang.ensure_layout!(uy, :g)
    Tarang.ensure_layout!(uz, :g)

    for i in eachindex(x), j in eachindex(y), k in eachindex(z)
        ux.data_g[i,j,k] =  cos(x[i]) * sin(y[j]) * cos(z[k])
        uy.data_g[i,j,k] = -sin(x[i]) * cos(y[j]) * cos(z[k])
        uz.data_g[i,j,k] = 0.0
    end

    Tarang.ensure_layout!(ux, :c)
    Tarang.ensure_layout!(uy, :c)
    Tarang.ensure_layout!(uz, :c)
end

taylor_green_ic!(ux, uy, uz)
```

## Diagnostic Functions

```julia
function kinetic_energy(ux, uy, uz)
    Tarang.ensure_layout!(ux, :g)
    Tarang.ensure_layout!(uy, :g)
    Tarang.ensure_layout!(uz, :g)

    return 0.5 * mean(ux.data_g.^2 + uy.data_g.^2 + uz.data_g.^2)
end

function enstrophy(ux, uy, uz)
    # Simplified: full implementation needs curl
    return 0.0  # Placeholder
end

# Analytical kinetic energy decay
KE_analytical(t, nu) = 0.125 * exp(-4*nu*t)
```

## Solver

```julia
solver = InitialValueSolver(problem, RK443(); dt=0.01)

cfl = CFL(problem; safety=0.5)
add_velocity!(cfl, ux)
add_velocity!(cfl, uy)
add_velocity!(cfl, uz)
```

## Simulation

```julia
t_end = 10.0

times = Float64[]
KE_numerical = Float64[]
KE_exact = Float64[]

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)

    if solver.iteration % 10 == 0
        t = solver.sim_time
        KE = kinetic_energy(ux, uy, uz)

        push!(times, t)
        push!(KE_numerical, KE)
        push!(KE_exact, KE_analytical(t, nu))

        println("t = $t, KE = $KE")
    end
end
```

## Results

### Energy Decay

```julia
plot(times, KE_numerical, label="Numerical", marker=:circle)
plot!(times, KE_exact, label="Analytical", linestyle=:dash)
xlabel!("Time")
ylabel!("Kinetic Energy")
title!("Taylor-Green Vortex Energy Decay (Re=$Re)")
```

### Error

```julia
relative_error = abs.(KE_numerical .- KE_exact) ./ KE_exact
plot(times, relative_error,
    xlabel="Time",
    ylabel="Relative Error",
    title="Energy Error",
    yscale=:log10
)
```

### Vorticity Visualization

```julia
# Slice at z = π
Tarang.ensure_layout!(ux, :g)
slice_idx = N÷2

ux_slice = ux.data_g[:, :, slice_idx]
heatmap(ux_slice',
    xlabel="x", ylabel="y",
    title="ux at z=π, t=$(solver.sim_time)"
)
```

## High Reynolds Number

For transition to turbulence:

```julia
# Re = 1600 is classic benchmark
Re_high = 1600
N_high = 256  # Need more resolution

# At high Re:
# - Initial structure breaks down
# - Energy cascades to small scales
# - Enstrophy peaks then decays
```

## Exercises

1. **Convergence study**: N = 32, 64, 128, 256
2. **Reynolds number scan**: Re = 100, 400, 1600, 3000
3. **Energy spectrum**: Compute at different times
4. **Enstrophy evolution**: Track vorticity magnitude

## Cleanup

```julia
MPI.Finalize()
```

## References

- Taylor, G.I. & Green, A.E. (1937). Mechanism of the production of small eddies from large ones
- Brachet, M.E. et al. (1983). Small-scale structure of the Taylor-Green vortex
