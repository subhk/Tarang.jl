# Tutorial: 3D Turbulence Simulation

This tutorial demonstrates setting up and running a 3D turbulent flow simulation using Tarang.jl.

## Physical Problem

We simulate decaying turbulence in a triply-periodic box, governed by the incompressible Navier-Stokes equations:

```math
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} &= -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f} \\
\nabla \cdot \mathbf{u} &= 0
\end{aligned}
```

where:
- $\mathbf{u} = (u, v, w)$ is the velocity field
- $p$ is the pressure (divided by density)
- $\nu$ is the kinematic viscosity
- $\mathbf{f}$ is an optional forcing term

## Domain Setup

### Coordinates and Distribution

```julia
using Tarang
using MPI

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# 3D Cartesian coordinates
coords = CartesianCoordinates("x", "y", "z")

# Distribute across MPI processes
# For 8 processes: mesh=(2, 2, 2)
# For 16 processes: mesh=(4, 2, 2) or (2, 4, 2)
dist = Distributor(coords, mesh=(2, 2, 2))
```

### Spectral Bases

All directions are periodic, so we use Fourier bases:

```julia
# Domain size (typically 2π for convenience)
L = 2π

# Resolution (power of 2 for optimal FFT)
N = 128  # 128^3 grid points

# Fourier bases for all directions
x_basis = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dealias=2/3)
y_basis = RealFourier(coords["y"]; size=N, bounds=(0.0, L), dealias=2/3)
z_basis = RealFourier(coords["z"]; size=N, bounds=(0.0, L), dealias=2/3)

domain = Domain(dist, (x_basis, y_basis, z_basis))
```

!!! note "Dealiasing"
    The `dealias=2/3` parameter enables the 3/2 rule for dealiasing nonlinear terms, essential for accurate turbulence simulations.

!!! info "No Tau Fields Needed"
    Since all directions use periodic Fourier bases, **no tau fields or boundary conditions are required**. The Fourier representation automatically enforces periodicity. Tau fields and the `lift()` operator are only needed for non-periodic directions (e.g., Chebyshev bases with walls). See [Boundary Conditions Tutorial](boundary_conditions.md) for problems with walls.

## Fields and Problem Definition

```julia
# Velocity field (3 components)
u = VectorField(dist, coords, "u", (x_basis, y_basis, z_basis))

# Pressure field
p = ScalarField(dist, "p", (x_basis, y_basis, z_basis))

# Problem definition (use the vector field directly)
problem = IVP([u, p])

# Parameters
Re = 1000.0  # Reynolds number (based on domain scale)
nu = 1.0 / Re

# Add parameter substitutions (Dedalus-style)
add_substitution!(problem, "nu", nu)

# Momentum equation (single vector equation)
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u)")

# Continuity equation
add_equation!(problem, "div(u) = 0")
```

!!! note "Vector Equation Syntax"
    Tarang.jl uses Dedalus-style string equations with full vector support:
    - `∂t(u)` - time derivative of vector field
    - `Δ(u)` - Laplacian of vector field (component-wise)
    - `∇(p)` - gradient of scalar (returns vector)
    - `div(u)` - divergence of vector (returns scalar)
    - `u⋅∇(u)` - advection term (u·∇)u

## Initial Conditions

### Taylor-Green Vortex

A classic test case for turbulence codes:

```julia
function initialize_taylor_green!(u, L)
    ux, uy, uz = u.components

    # Get grid data
    ux_grid = get_grid_data(ux)
    uy_grid = get_grid_data(uy)
    uz_grid = get_grid_data(uz)

    # Get local grid coordinates
    x = get_grid(ux.bases[1])
    y = get_grid(ux.bases[2])
    z = get_grid(ux.bases[3])

    # Taylor-Green initial condition
    for i in eachindex(x), j in eachindex(y), k in eachindex(z)
        ux_grid[i,j,k] =  sin(x[i]) * cos(y[j]) * cos(z[k])
        uy_grid[i,j,k] = -cos(x[i]) * sin(y[j]) * cos(z[k])
        uz_grid[i,j,k] = 0.0
    end

    # Transform to spectral space
    to_spectral!(ux)
    to_spectral!(uy)
    to_spectral!(uz)
end

initialize_taylor_green!(u, L)
```

### Random Initial Conditions

For studying developed turbulence:

```julia
using Random

function initialize_random!(u, energy_spectrum; seed=42)
    Random.seed!(seed + MPI.Comm_rank(MPI.COMM_WORLD))

    for component in u.components
        data = get_grid_data(component)
        data .= randn(size(data))
        to_spectral!(component)
    end

    # Project to divergence-free (optional, done by solver)
    # Scale to desired energy level
    # ... energy scaling code ...
end
```

## Solver Setup

```julia
# Timestepper selection
# - RK443 for IMEX Runge-Kutta integration
# - SBDF2/SBDF3 for very stiff problems at high Re
timestepper = RK443()

# Create solver
solver = InitialValueSolver(problem, timestepper, dt=1e-3)

# CFL condition for adaptive timestep
cfl = CFL(problem, safety=0.5, max_change=1.5)
add_velocity!(cfl, u)
cfl.max_dt = 0.01
```

## Analysis and Output

### Energy Spectrum

```julia
function compute_energy_spectrum(u)
    # Compute kinetic energy in spectral space
    E_k = zeros(N÷2)

    for component in u.components
        data = get_spectral_data(component)
        # Bin by wavenumber shell
        # ... spectral binning code ...
    end

    return E_k
end
```

### Enstrophy and Dissipation

```julia
function compute_enstrophy(u)
    # ω = ∇ × u
    omega = curl(u)

    # Ω = ∫ |ω|² dV
    enstrophy = 0.0
    for component in omega.components
        data = get_grid_data(component)
        enstrophy += sum(data.^2)
    end

    return enstrophy / (N^3)
end

function compute_dissipation(u, nu)
    # ε = 2ν Ω
    return 2 * nu * compute_enstrophy(u)
end
```

### File Output

```julia
# NetCDF output
output = add_netcdf_handler(
    solver, "turbulence_output",
    fields=[ux, uy, uz, p],
    write_interval=0.5
)

# Analysis output (scalars)
analysis = add_netcdf_handler(
    solver, "turbulence_analysis",
    write_interval=0.1
)

# Add computed quantities
add_task!(analysis, () -> compute_kinetic_energy(u), name="kinetic_energy")
add_task!(analysis, () -> compute_enstrophy(u), name="enstrophy")
add_task!(analysis, () -> compute_dissipation(u, nu), name="dissipation")
```

## Time Integration

```julia
# Simulation parameters
t_end = 10.0
output_interval = 0.5
next_output = output_interval

# Diagnostic output
if rank == 0
    println("=" ^ 60)
    println("3D Turbulence Simulation")
    println("Re = $Re, N = $N")
    println("=" ^ 60)
end

# Main loop
while solver.sim_time < t_end
    # Adaptive timestep
    dt = compute_timestep(cfl)

    # Advance solution
    step!(solver, dt)

    # Periodic output
    if solver.sim_time >= next_output
        KE = compute_kinetic_energy(u)
        eps = compute_dissipation(u, nu)

        if rank == 0
            @printf("t = %.3f, KE = %.6e, ε = %.6e, dt = %.2e\n",
                    solver.sim_time, KE, eps, dt)
        end

        next_output += output_interval
    end
end

MPI.Finalize()
```

## Performance Considerations

### Resolution Requirements

| Reynolds Number | Minimum Grid | Recommended Grid |
|----------------|--------------|------------------|
| Re ~ 100       | 32³          | 64³              |
| Re ~ 1000      | 64³          | 128³             |
| Re ~ 10000     | 128³         | 256³             |
| Re ~ 100000    | 256³         | 512³+            |

### MPI Scaling

For 3D simulations:

| Processes | Recommended Mesh | Notes |
|-----------|------------------|-------|
| 8         | (2, 2, 2)        | Basic |
| 64        | (4, 4, 4)        | Good scaling |
| 512       | (8, 8, 8)        | Large scale |

### Memory Usage

Estimate: ~100 bytes per grid point (double precision, 3D velocity + pressure + work arrays)

```julia
# Memory estimate
N = 256
bytes_per_point = 100
total_memory = N^3 * bytes_per_point / 1e9  # GB
println("Estimated memory: $(total_memory) GB")
```

## Running the Simulation

```bash
# Set environment
export OMP_NUM_THREADS=1

# Run with MPI
mpiexec -n 8 julia --project turbulence_3d.jl
```

## Visualization

### ParaView Export

```julia
# Write VTK files for ParaView
write_vtk(solver, "turbulence", iteration=solver.iteration)
```

### In-situ Visualization

```julia
using GLMakie

# Extract slice for visualization
function visualize_slice(u, plane="xy", position=0.5)
    data = get_grid_data(u.components[1])
    # ... slice extraction ...
    heatmap(slice_data)
end
```

## References

1. Taylor, G. I., & Green, A. E. (1937). Mechanism of the production of small eddies from large ones.
2. Pope, S. B. (2000). Turbulent Flows. Cambridge University Press.
3. Canuto, C., et al. (2007). Spectral Methods: Fundamentals in Single Domains.
