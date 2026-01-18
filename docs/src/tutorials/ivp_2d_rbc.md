# Tutorial: 2D Rayleigh-Bénard Convection

This tutorial demonstrates solving a classic fluid dynamics problem: Rayleigh-Bénard convection. We'll set up a complete simulation including equations, boundary conditions, adaptive time stepping, and output.

## Physical Problem

Rayleigh-Bénard convection occurs when a fluid layer is heated from below. The setup:

- Horizontal layer of fluid between two parallel plates
- Bottom plate at temperature T_hot
- Top plate at temperature T_cold
- Gravity acts downward

When the temperature difference is large enough (high Rayleigh number), the fluid becomes unstable and convection cells form.

### Governing Equations

The Boussinesq equations for thermal convection:

```math
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} &= -\nabla p + \text{Pr} \nabla^2 \mathbf{u} + \text{Ra} \cdot \text{Pr} \cdot T \, \hat{\mathbf{z}} \\
\nabla \cdot \mathbf{u} &= 0 \\
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T &= \nabla^2 T
\end{aligned}
```

**Dimensionless parameters**:
- **Ra** (Rayleigh number): $\text{Ra} = \frac{g \alpha \Delta T H^3}{\nu \kappa}$ - ratio of buoyancy to viscous forces
- **Pr** (Prandtl number): $\text{Pr} = \frac{\nu}{\kappa}$ - ratio of momentum to thermal diffusivity

**Variables**:
- $\mathbf{u} = (u, w)$: velocity field (horizontal u, vertical w)
- $p$: pressure
- $T$: temperature (deviation from linear conduction profile)

## Domain and Discretization

### Coordinate System

2D Cartesian domain:
- **x** (horizontal): periodic, length $L_x$
- **z** (vertical): bounded by plates, height $H = 1$

### Spectral Bases

```julia
# Periodic horizontal direction → Fourier
x_basis = RealFourier(coords["x"]; size=256, bounds=(0.0, 4.0))

# Bounded vertical direction → Chebyshev
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
```

**Resolution guidelines**:
- Horizontal: 128-512 modes (depends on aspect ratio and Ra)
- Vertical: 32-128 modes (more for higher Ra)
- Aspect ratio $L_x/H$: typically 2-4 for 2D

## Complete Implementation

### Setup

```julia
using Tarang
using MPI
using Printf

# Initialize MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

# Problem parameters
Ra = 1e6  # Rayleigh number
Pr = 1.0  # Prandtl number
Lx = 4.0  # Horizontal extent
H = 1.0   # Vertical extent

# Domain setup
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

# Spectral bases
Nx, Nz = 256, 64
x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx))
z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, H))

domain = Domain(dist, (x_basis, z_basis))
```

### Fields

```julia
# Create velocity field (vector with 2 components)
u = VectorField(dist, coords, "u", (x_basis, z_basis))
ux = u.components[1]  # Horizontal velocity
uz = u.components[2]  # Vertical velocity

# Pressure field
p = ScalarField(dist, "p", (x_basis, z_basis))

# Temperature field
T = ScalarField(dist, "T", (x_basis, z_basis))
```

### Tau Fields (for Boundary Conditions)

Tarang.jl follows the [Dedalus](https://dedalus-project.readthedocs.io/) approach: users must explicitly create tau fields and add them to equations using the `lift()` operator.

```julia
# Tau field for pressure (removes degeneracy in continuity equation)
tau_p = ScalarField(dist, "tau_p", (), dtype)

# Tau fields for temperature/buoyancy BCs (one per boundary)
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,), dtype)   # T at z=0
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,), dtype)   # T at z=1

# Vector tau fields for velocity BCs (uses VectorField for compact notation)
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), dtype)  # Velocity at z=0
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), dtype)  # Velocity at z=1
```

!!! tip "Vector Tau Fields"
    Using `VectorField` for tau fields allows compact vector notation in equations. The components are accessed via `tau_u1.components[1]` (x) and `tau_u1.components[2]` (z).

### Problem Definition

```julia
# Collect all state variables (vector fields passed directly)
problem = IVP([u, p, T, tau_u1, tau_u2, tau_p, tau_T1, tau_T2])

# Add substitutions for parameters (Dedalus-style)
add_substitution!(problem, "Ra", Ra)
add_substitution!(problem, "Pr", Pr)
add_substitution!(problem, "Lz", H)

# Equations use vector notation (Dedalus-style)
# Continuity: ∇·u = 0
add_equation!(problem, "div(u) + tau_p = 0")

# Temperature equation: ∂T/∂t - ∇²T = -u·∇T
add_equation!(problem, "∂t(T) - Δ(T) + lift(tau_T2) = -u⋅∇(T)")

# Momentum (vector equation): ∂u/∂t - Pr∇²u + ∇p = -u·∇u + Ra*Pr*T*ez
# ez is the unit vector in z-direction (buoyancy acts vertically)
add_equation!(problem, "∂t(u) - Pr*Δ(u) + ∇(p) + lift(tau_u2) = -u⋅∇(u) + Ra*Pr*T*ez")
```

### Boundary Conditions

No-slip walls (velocity = 0) and fixed temperatures, using Dedalus-style string format:

```julia
# Temperature BCs
add_bc!(problem, "T(z=0) = 1")      # Hot bottom
add_bc!(problem, "T(z=Lz) = 0")     # Cold top

# Velocity BCs (no-slip at both walls) - vector notation
add_bc!(problem, "u(z=0) = 0")      # No-slip bottom (sets all components)
add_bc!(problem, "u(z=Lz) = 0")     # No-slip top
```

!!! note "Equation String Syntax"
    Tarang.jl uses Dedalus-style string equations:
    - Unicode operators: `∂t` (time derivative), `Δ` (Laplacian), `∇` (gradient)
    - Vector advection: `u⋅∇(u)` for nonlinear term
    - Scalar advection: `u⋅∇(T)` for temperature advection
    - Unit vectors: `ex`, `ey`, `ez` for directions (e.g., `Ra*Pr*T*ez` for buoyancy)
    - Vector BCs: `u(z=0) = 0` sets all velocity components at the boundary

### Initial Conditions

Add random perturbations to trigger convection:

```julia
# Get grid-space data
T_grid = get_grid_data(T)

# Add small random perturbations
Random.seed!(42 + rank)  # Different seed per rank
T_grid .= 0.5 .+ 0.01 .* (rand(size(T_grid)...) .- 0.5)

# Transform to spectral space
to_spectral!(T)

# Velocity starts at zero (already initialized)
```

### Time Stepper and Solver

```julia
# Choose timestepper
# RK222: Good for moderate Ra
# CNAB2 or SBDF2: Better for high Ra (more stable for stiff problems)
timestepper = RK222()

# Create solver
solver = InitialValueSolver(problem, timestepper, dt=1e-4)
```

**Timestepper selection**:
- Ra < 10⁵: RK222 or RK443 (IMEX Runge-Kutta)
- Ra > 10⁵: CNAB2 or SBDF2 (IMEX multistep)
- Very high Ra: SBDF3 or SBDF4

### Adaptive Time Stepping (CFL)

```julia
# Create CFL condition
cfl = CFL(problem, safety=0.4, max_change=1.5, min_change=0.5)

# Add velocity field for CFL calculation
add_velocity!(cfl, u)

# Set maximum timestep
cfl.max_dt = 0.01
```

**CFL parameters**:
- `safety`: Safety factor (0.2-0.5, lower = more stable)
- `max_change`: Maximum timestep increase per step
- `min_change`: Maximum timestep decrease per step

### Output and Analysis

```julia
# File output handler
output_handler = add_netcdf_handler(
    solver,
    "rayleigh_benard_output",
    fields=[ux, uz, p, T],
    write_interval=0.1  # Write every 0.1 time units
)

# Analysis: Compute Nusselt number (heat flux)
function compute_nusselt(T, uz)
    # Nu = 1 + <uz*T> (horizontally averaged)
    T_grid = get_grid_data(T)
    uz_grid = get_grid_data(uz)

    flux = mean(uz_grid .* T_grid, dims=1)  # Average over x
    Nu = 1.0 .+ flux

    return mean(Nu)  # Average over z as well
end
```

### Time-Stepping Loop

```julia
# Simulation parameters
t_end = 10.0
output_dt = 0.1
next_output = output_dt

# Diagnostics
iteration = 0
start_time = time()

if rank == 0
    println("Starting Rayleigh-Bénard simulation")
    println("Ra = $Ra, Pr = $Pr")
    println("Resolution: $Nx × $Nz")
    println("MPI processes: $size")
    println("=" ^ 60)
end

# Main time loop
while solver.sim_time < t_end
    # Compute adaptive timestep
    dt = compute_timestep(cfl)

    # Take a step
    step!(solver, dt)

    iteration += 1

    # Output and diagnostics
    if solver.sim_time >= next_output
        # Compute Nusselt number
        Nu = compute_nusselt(T, uz)

        if rank == 0
            elapsed = time() - start_time
            @printf("t = %.3f, dt = %.2e, Nu = %.3f, wall_time = %.1fs\n",
                    solver.sim_time, dt, Nu, elapsed)
        end

        next_output += output_dt
    end
end

if rank == 0
    total_time = time() - start_time
    @printf("\nSimulation complete!\n")
    @printf("Total iterations: %d\n", iteration)
    @printf("Total wall time: %.2f s\n", total_time)
    @printf("Time per iteration: %.3f ms\n", 1000 * total_time / iteration)
end

# Cleanup
MPI.Finalize()
```

## Running the Simulation

Save the complete script as `rayleigh_benard_2d.jl` and run:

```bash
# Set threads
export OMP_NUM_THREADS=1

# Run with 4 MPI processes
mpiexec -n 4 julia --project rayleigh_benard_2d.jl
```

**Expected output**:
```
Starting Rayleigh-Bénard simulation
Ra = 1000000.0, Pr = 1.0
Resolution: 256 × 64
MPI processes: 4
============================================================
t = 0.100, dt = 2.45e-04, Nu = 1.023, wall_time = 3.2s
t = 0.200, dt = 2.31e-04, Nu = 1.156, wall_time = 6.5s
...
```

## Physical Interpretation

### Flow Regimes

Different Rayleigh numbers produce different behaviors:

| Ra | Regime | Characteristics |
|----|--------|-----------------|
| < 1708 | Conduction | No flow, pure diffusion |
| 10³-10⁴ | Steady rolls | Regular convection cells |
| 10⁵-10⁶ | Transitional | Time-dependent, irregular |
| > 10⁶ | Turbulent | Chaotic, small-scale structures |

### Nusselt Number

The Nusselt number measures heat transfer enhancement:

```math
\text{Nu} = \frac{\text{Total heat flux}}{\text{Conductive heat flux}}
```

- Nu = 1: Pure conduction (no convection)
- Nu > 1: Enhanced heat transfer due to convection
- Higher Ra → Higher Nu

For Rayleigh-Bénard convection:
- Nu ≈ 1.0 for Ra < 1708
- Nu ∝ Ra^(1/3) approximately for turbulent regime

## Visualization

### Basic Plotting

```julia
using Plots

# Extract data from one field
T_grid = get_grid_data(T)

# Plot temperature field
heatmap(T_grid', xlabel="x", ylabel="z", title="Temperature")
savefig("temperature.png")
```

### Advanced Analysis

```julia
# Compute kinetic energy spectrum
function compute_spectrum(u)
    u_spectral = get_spectral_data(u)
    energy = abs2.(u_spectral)

    # Bin by wavenumber magnitude
    # ... (see Analysis tutorial for details)

    return k, E_k
end

k, E_k = compute_spectrum(ux)
plot(k, E_k, xscale=:log10, yscale=:log10,
     xlabel="k", ylabel="E(k)", label="Kinetic Energy")
```

## Parameter Studies

### Exploring Rayleigh Number

Study the transition to turbulence:

```julia
Ra_values = [1e4, 1e5, 1e6, 1e7]

for Ra in Ra_values
    problem.parameters["Ra"] = Ra
    # ... run simulation and save Nu
end
```

### Aspect Ratio Effects

Try different domain sizes:

```julia
aspect_ratios = [1.0, 2.0, 4.0, 8.0]

for ar in aspect_ratios
    Lx = ar * H
    x_basis = RealFourier(coords["x"]; size=Int(128*ar), bounds=(0.0, Lx))
    # ... run simulation
end
```

## Performance Optimization

### Resolution Requirements

For Ra = 10⁶:
- Minimum: 128 × 32
- Recommended: 256 × 64
- High-resolution: 512 × 128

**Scaling**: For Ra × 10, increase resolution by ~1.5×

### MPI Process Mesh

| Processes | Recommended Mesh | Domain |
|-----------|------------------|--------|
| 4 | (2, 2) | Square/moderate aspect |
| 8 | (4, 2) or (2, 4) | Match domain aspect ratio |
| 16 | (4, 4) | Square mesh works well |

### Timestep Control

```julia
# For stability at high Ra
cfl.safety = 0.3  # Reduce if getting NaN

# For efficiency
cfl.max_dt = 0.01  # Don't let dt grow too large
cfl.max_change = 1.2  # Smooth timestep changes
```

## Troubleshooting

### NaN in Solution

**Causes**:
- Timestep too large
- Ra too high for resolution
- Insufficient damping

**Solutions**:
```julia
# Reduce CFL safety factor
cfl.safety = 0.2

# Use more stable timestepper
timestepper = CNAB2()  # instead of RK222

# Increase resolution
Nz = 128  # instead of 64
```

### Simulation Not Converging

**For steady state**:
- Run longer (convection takes time to develop)
- Check initial conditions have perturbations
- Verify Ra > 1708 (critical value)

### Memory Issues

**Solutions**:
```julia
# Reduce resolution
Nx, Nz = 128, 32

# Use more MPI processes
mesh=(4, 4)  # instead of (2, 2)
```

## Complete Script

The full script is available in `examples/rayleigh_benard_2d.jl`. Key points:

1. Proper MPI initialization and finalization
2. Appropriate bases for boundary conditions
3. Explicit tau fields for each boundary condition
4. lift() operators in all equations with non-periodic BCs
5. Complete equation system with nonlinear terms
6. All boundary conditions linked to tau fields
7. Adaptive time stepping with CFL
8. Output and analysis
9. Performance monitoring

!!! info "Dedalus-Style Approach"
    This tutorial follows the Dedalus approach for boundary conditions. Users must explicitly:
    - Create tau fields (one per BC)
    - Add tau fields to the problem
    - Include lift() terms in equations
    - Link each BC to its tau field

    See [Boundary Conditions Tutorial](boundary_conditions.md) for more details.

## Next Steps

- **[3D Extension](ivp_3d_turbulence.md)**: Add third dimension
- **[Eigenvalue Analysis](eigenvalue_problems.md)**: Compute stability
- **[Custom Analysis](analysis_and_output.md)**: Advanced diagnostics

## References

1. Chandrasekhar, S. (1961). *Hydrodynamic and Hydromagnetic Stability*
2. Tritton, D. J. (1988). *Physical Fluid Dynamics*
3. Spectral Methods Resources
