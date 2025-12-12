# First Steps with Tarang.jl

This tutorial introduces the basic workflow for setting up and solving PDEs with Tarang.jl. We'll build a simple 2D diffusion problem step by step.

## The Tarang.jl Workflow

Every Tarang.jl simulation follows these steps:

1. **Initialize MPI** for parallel computing
2. **Define coordinates** and create a distributor for MPI processes
3. **Choose spectral bases** for each coordinate direction
4. **Create a domain** combining the bases
5. **Define fields** (scalar, vector, or tensor)
6. **Set up a problem** (IVP, BVP, or EVP)
7. **Add equations** using symbolic syntax
8. **Specify boundary conditions**
9. **Create a solver** with a timestepper
10. **Run the simulation** with time stepping
11. **Analyze and output results**

Let's walk through each step with a concrete example.

## Example: 2D Heat Diffusion

We'll solve the 2D heat equation:

```math
\frac{\partial T}{\partial t} = \kappa \nabla^2 T
```

on a rectangular domain with Dirichlet boundary conditions.

### Step 1: Initialize MPI

Every Tarang.jl script starts by initializing MPI:

```julia
using Tarang
using MPI

MPI.Init()
```

### Step 2: Define Coordinates and Distributor

Coordinates define the dimension names and the MPI process distribution:

```julia
# Create 2D Cartesian coordinates
coords = CartesianCoordinates("x", "z")

# Create distributor with 2×2 process mesh (4 MPI processes total)
dist = Distributor(coords, mesh=(2, 2))
```

The `mesh=(2, 2)` means we'll use 4 MPI processes arranged in a 2×2 grid. This enables parallelization in both horizontal (x) and vertical (z) directions.

!!! tip "Choosing Process Mesh"
    The product of mesh dimensions should equal your MPI process count:
    - `mesh=(2, 2)` → 4 processes
    - `mesh=(4, 2)` → 8 processes
    - `mesh=(4, 4)` → 16 processes

### Step 3: Choose Spectral Bases

Bases define the spectral representation in each direction:

```julia
# Periodic direction (x) - use Fourier basis
x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))

# Bounded direction (z) - use Chebyshev basis
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))
```

**Basis selection guide:**
- **Fourier**: Periodic boundaries (e.g., horizontal directions in atmospheric flows)
- **Chebyshev**: Non-periodic with high accuracy (e.g., vertical direction with walls)
- **Legendre**: Non-periodic, alternative to Chebyshev

The `size` parameter determines the spectral resolution.

### Step 4: Create Domain

Combine the bases into a domain:

```julia
domain = Domain(dist, (x_basis, z_basis))
```

The domain handles the spatial discretization and MPI distribution.

### Step 5: Define Fields

Create a scalar field for temperature:

```julia
T = ScalarField(dist, "T", (x_basis, z_basis))
```

For vector fields (like velocity):
```julia
u = VectorField(dist, coords, "u", (x_basis, z_basis))
```

### Step 6: Set Up Problem

Create an Initial Value Problem (IVP):

```julia
problem = IVP([T])

# Add the heat equation
add_equation!(problem, "∂t(T) = kappa*lap(T)")

# Set diffusion coefficient
problem.parameters["kappa"] = 1.0
```

The equation uses symbolic notation:
- `∂t(T)`: time derivative ∂T/∂t
- `lap(T)`: Laplacian ∇²T
- `kappa`: a parameter we can easily modify

### Step 7: Add Boundary Conditions

Specify boundary conditions at the domain edges:

```julia
# Bottom wall (z=0): hot, T=1
add_equation!(problem, "T(z=0) = 1")

# Top wall (z=1): cold, T=0
add_equation!(problem, "T(z=1) = 0")
```

The x-direction is periodic (RealFourier basis), so no boundary conditions are needed there.

### Step 8: Create Solver

Choose a timestepper and create the solver:

```julia
# RK222 is a good general-purpose IMEX timestepper
timestepper = RK222()
solver = InitialValueSolver(problem, timestepper, dt=0.001)
```

Popular timesteppers:
- **RK222, RK443**: IMEX Runge-Kutta (good for most problems)
- **CNAB2, SBDF2**: IMEX multistep methods (for stiff problems)

### Step 9: Set Initial Conditions

Initialize the temperature field:

```julia
# Get the grid space representation
T_grid = get_grid_data(T)

# Set a Gaussian perturbation
for i in 1:size(T_grid, 1), j in 1:size(T_grid, 2)
    x = (i-1) * 2π / 128
    z = (j-1) / 64
    T_grid[i, j] = 0.5 + 0.1 * exp(-((x-π)^2 + (z-0.5)^2) / 0.1)
end

# Transform to spectral space
to_spectral!(T)
```

### Step 10: Run Simulation

Time-step the solver:

```julia
t_end = 1.0
iteration = 0

while solver.sim_time < t_end
    step!(solver)
    iteration += 1

    # Print progress every 100 steps
    if iteration % 100 == 0 && MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("Iteration: $iteration, Time: $(solver.sim_time)")
    end
end
```

### Step 11: Finalize MPI

Always finalize MPI at the end:

```julia
MPI.Finalize()
```

## Complete Example

Here's the full script:

```julia
using Tarang, MPI

MPI.Init()

# Setup
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

domain = Domain(dist, (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))

# Problem
problem = IVP([T])
add_equation!(problem, "∂t(T) = kappa*lap(T)")
problem.parameters["kappa"] = 0.01

# Boundary conditions (Dedalus-style syntax auto-detected)
add_equation!(problem, "T(z=0) = 1")
add_equation!(problem, "T(z=1) = 0")

# Solver
solver = InitialValueSolver(problem, RK222(), dt=0.001)

# Initial conditions
T_grid = get_grid_data(T)
for i in 1:size(T_grid, 1), j in 1:size(T_grid, 2)
    x = (i-1) * 2π / 128
    z = (j-1) / 64
    T_grid[i, j] = 0.5 + 0.1 * exp(-((x-π)^2 + (z-0.5)^2) / 0.1)
end
to_spectral!(T)

# Time stepping
while solver.sim_time < 1.0
    step!(solver)
end

MPI.Finalize()
```

Save this as `heat_diffusion.jl` and run:

```bash
mpiexec -n 4 julia heat_diffusion.jl
```

## What's Next?

Now that you understand the basic workflow, explore:

- [Running with MPI](running_with_mpi.md): Details on parallel execution
- [2D Rayleigh-Bénard Tutorial](../tutorials/ivp_2d_rbc.md): More complex fluid dynamics example
- [Boundary Conditions](../tutorials/boundary_conditions.md): Advanced boundary condition types
- [Analysis and Output](../tutorials/analysis_and_output.md): Saving data and computing diagnostics

## Common Patterns

### Multiple Fields

For coupled PDEs with multiple fields:

```julia
u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))

problem = IVP([u.components[1], u.components[2], p, T])
```

### Parameters

Add and modify parameters easily:

```julia
problem.parameters["Ra"] = 1e6  # Rayleigh number
problem.parameters["Pr"] = 0.7  # Prandtl number
problem.parameters["kappa"] = 0.01  # Thermal diffusivity
```

### Adaptive Time Stepping

Use CFL condition for adaptive time steps:

```julia
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

## Key Takeaways

1. **MPI must be initialized** before creating any Tarang objects
2. **Choose bases** appropriate for your boundary conditions (Fourier for periodic, Chebyshev/Legendre for bounded)
3. **Process mesh** should match your MPI process count
4. **Symbolic equations** use natural mathematical notation
5. **Boundary conditions** are specified separately from equations
6. **Always finalize MPI** at the end of your script
