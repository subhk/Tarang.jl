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
# Put the bounded (Chebyshev) coordinate first for distributed mixed transforms
coords = CartesianCoordinates("z", "x")

# A 2D domain uses a 1D (slab) process mesh
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, device=CPU())
```

For a 2D domain, Tarang keeps one transform direction local and distributes the
other over a one-dimensional process mesh. With four MPI ranks, the mesh is
therefore `(4,)`. In a mixed bounded-periodic domain, list the bounded coordinate
and basis first so its Chebyshev transform remains local. For a 3D domain,
Tarang can use a two-dimensional pencil mesh.

!!! tip "Choosing Process Mesh"
    The product of mesh dimensions should equal your MPI process count:
    - 2D domain: `mesh=(4,)` → 4 processes
    - 3D domain: `mesh=(2, 2)` → 4 processes
    - Omit `mesh` to let Tarang choose a compatible layout automatically

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
domain = Domain(dist, (z_basis, x_basis))
```

The domain handles the spatial discretization and MPI distribution.

### Step 5: Define Fields

Create a scalar field for temperature:

```julia
T = ScalarField(domain, "T")

# Two tau fields enforce the two Chebyshev-wall boundary conditions
tau_T1 = ScalarField(dist, "tau_T1", (), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (), Float64)
```

For vector fields (like velocity):
```julia
u = VectorField(domain, "u")
```

### Step 6: Set Up Problem

Create an Initial Value Problem (IVP). A bounded second-order equation uses the
tau method: one lifted tau enters the gradient and the other enters the bulk
equation.

```julia
ez, ex = unit_vector_fields(coords, dist)  # coords are ("z", "x")
τ_lift(A) = lift(A, derivative_basis(z_basis, 1), -1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([T, tau_T1, tau_T2])
add_parameters!(problem; kappa=0.01, grad_T=grad_T, τ_lift=τ_lift)

# Add the heat equation with its tau corrections
add_equation!(problem, "∂t(T) - kappa*div(grad_T) + τ_lift(tau_T2) = 0")
```

The equation uses symbolic notation:
- `∂t(T)`: time derivative ∂T/∂t
- `div(grad_T)`: Laplacian plus the first tau correction
- `kappa`: a parameter we can easily modify

### Step 7: Add Boundary Conditions

Specify boundary conditions at the domain edges:

```julia
# Bottom wall (z=0): hot, T=1
add_bc!(problem, "T(z=0) = 1")

# Top wall (z=1): cold, T=0
add_bc!(problem, "T(z=1) = 0")
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
# `local_grids` returns this rank's slab, so this works in serial and under MPI
z, x = local_grids(dist, z_basis, x_basis)
ensure_layout!(T, :g)
get_grid_data(T) .= 0.5 .+ 0.1 .* exp.(-((z .- 0.5).^2 .+ (x' .- π).^2) ./ 0.1)
ensure_layout!(T, :c)
```

### Step 10: Run Simulation

Time-step the solver:

```julia
for iteration in 1:20
    step!(solver)

    if iteration % 10 == 0 && MPI.Comm_rank(MPI.COMM_WORLD) == 0
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
coords = CartesianCoordinates("z", "x")
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, device=CPU())

x_basis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 1.0))

domain = Domain(dist, (z_basis, x_basis))
T = ScalarField(domain, "T")
tau_T1 = ScalarField(dist, "tau_T1", (), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (), Float64)

# Problem
ez, ex = unit_vector_fields(coords, dist)
τ_lift(A) = lift(A, derivative_basis(z_basis, 1), -1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([T, tau_T1, tau_T2])
add_parameters!(problem; kappa=0.01, grad_T=grad_T, τ_lift=τ_lift)
add_equation!(problem, "∂t(T) - kappa*div(grad_T) + τ_lift(tau_T2) = 0")

# Boundary conditions
add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "T(z=1) = 0")

# Solver
solver = InitialValueSolver(problem, RK222(), dt=0.001)

# Initial conditions on this rank's local grid
z, x = local_grids(dist, z_basis, x_basis)
ensure_layout!(T, :g)
get_grid_data(T) .= 0.5 .+ 0.1 .* exp.(-((z .- 0.5).^2 .+ (x' .- π).^2) ./ 0.1)
ensure_layout!(T, :c)

# Short smoke run; increase the resolution and step count for a simulation
for _ in 1:20
    step!(solver)
end

MPI.Finalize()
```

Save this as `heat_diffusion.jl` and run:

```bash
mpiexecjl --project=. -n 4 julia heat_diffusion.jl
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
u = VectorField(domain, "u")
p = ScalarField(domain, "p")
T = ScalarField(domain, "T")

problem = IVP([u.components[1], u.components[2], p, T])
```

### Parameters

Add and modify parameters easily:

```julia
problem.namespace["Ra"] = 1e6  # Rayleigh number
problem.namespace["Pr"] = 0.7  # Prandtl number
problem.namespace["kappa"] = 0.01  # Thermal diffusivity
```

### Adaptive Time Stepping

Use CFL condition for adaptive time steps:

```julia
cfl = CFL(solver; safety=0.5)
add_velocity!(cfl, u)
# The returned dt covers advection only. If you treat diffusion explicitly
# (e.g. an LES eddy viscosity), also register it:
#   add_diffusivity!(cfl, nu_e)

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
