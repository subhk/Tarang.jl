# Tutorials Overview

This section contains comprehensive tutorials that guide you through solving various types of PDEs with Tarang.jl. Each tutorial builds on concepts from previous ones, so we recommend following them in order if you're new to spectral methods or Tarang.jl.

## Tutorial Path

### For Beginners

If you're new to spectral methods or Tarang.jl, start here:

1. **[First Steps](../getting_started/first_steps.md)** - Basic workflow and simple diffusion problem
2. **[2D Rayleigh-Bénard Convection](ivp_2d_rbc.md)** - Complete fluid dynamics example
3. **[Boundary Conditions](boundary_conditions.md)** - Master the boundary condition system

### Intermediate Topics

Once comfortable with basics:

4. **[3D Turbulent Flow](ivp_3d_turbulence.md)** - 3D problems with advanced parallelization
5. **[Analysis and Output](analysis_and_output.md)** - Data management and visualization
6. **[Eigenvalue Problems](eigenvalue_problems.md)** - Linear stability analysis

### Advanced Topics

For experienced users:

- **[Custom Operators](../pages/custom_operators.md)** - Define new differential operators
- **[Optimization Guide](../pages/optimization.md)** - Performance tuning

## Tutorial List

### Initial Value Problems (IVP)

Time-evolution problems where you integrate PDEs forward in time.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [2D Rayleigh-Bénard](ivp_2d_rbc.md) | Thermal convection in 2D | Intermediate | Navier-Stokes, buoyancy, CFL |
| [3D Taylor-Green Vortex](ivp_3d_turbulence.md) | 3D turbulence simulation | Advanced | 3D FFTs, energy spectra |
| [Channel Flow](../notebooks/channel_flow.md) | Turbulent channel flow | Advanced | Wall-bounded, statistics |

### Boundary Value Problems (BVP)

Steady-state problems with boundary conditions.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| Steady Convection | Fixed temperature Rayleigh-Bénard | Intermediate | LBVP, sparse linear solve |
| Stokes Flow | Low Reynolds number flow | Beginner | Simple BVP example |

### Eigenvalue Problems (EVP)

Linear stability analysis and normal modes.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Stability Analysis](eigenvalue_problems.md) | Eigenvalue problem setup | Advanced | EVP, eigensolvers |

### Surface and Boundary Dynamics

Problems with dynamics confined to surfaces or boundaries.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Surface Dynamics](surface_dynamics.md) | SQG, QG, boundary advection-diffusion | Advanced | Fractional Laplacian, coupled systems |

### Wave-Mean Flow Analysis

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Rotating Shallow Water](rotating_shallow_water.md) | Inertia-gravity wave filtering | Intermediate | Temporal filters, wave-mean separation |
| [GQL Approximation](../pages/gql_approximation.md) | Generalized Quasi-Linear methods | Advanced | Wavenumber cutoff, zonal jets |

### Special Topics

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Boundary Conditions](boundary_conditions.md) | All BC types and usage | Intermediate | Dirichlet, Neumann, Robin |
| [Analysis & Output](analysis_and_output.md) | Data management | Intermediate | NetCDF, HDF5, analysis |

## Problem Types Explained

### Initial Value Problems (IVP)

**When to use**: Time-dependent PDEs where you know the initial state and want to evolve forward in time.

**Examples**:
- Fluid dynamics (Navier-Stokes)
- Heat diffusion
- Wave propagation
- Reaction-diffusion systems

**Typical structure**:
```julia
problem = IVP(fields)
add_equation!(problem, "∂t(u) = -u*∂x(u) + nu*Δ(u)")
solver = InitialValueSolver(problem, RK222())

while solver.sim_time < t_end
    step!(solver, dt)
end
```

### Boundary Value Problems (BVP)

**When to use**: Steady-state problems where you solve for the spatial distribution given boundary conditions.

**Types**:
- **LBVP**: Linear boundary value problems
- **NLBVP**: Nonlinear boundary value problems (require iteration)

**Examples**:
- Steady-state heat conduction
- Poisson equation
- Steady Stokes flow

**Typical structure**:
```julia
problem = LBVP(fields)
add_equation!(problem, "lap(T) = f")
add_bc!(problem, "T(z=0) = 0")
add_bc!(problem, "T(z=1) = 1")

solver = BoundaryValueSolver(problem)
solve!(solver)
```

### Eigenvalue Problems (EVP)

**When to use**: Linear stability analysis, computing normal modes, or finding eigenvalues of differential operators.

**Examples**:
- Hydrodynamic stability
- Normal mode analysis
- Resonance frequencies

**Typical structure**:
```julia
problem = EVP(fields, eigenvalue=:sigma)
add_equation!(problem, "sigma*u = -lap(u) + f")
add_bc!(problem, "u(z=0) = 0")

solver = EigenvalueSolver(problem)
eigenvalues = solve!(solver, nev=10)
```

## Choosing a Tutorial

### By Physics

**Fluid Dynamics**:
- Start: [2D Rayleigh-Bénard](ivp_2d_rbc.md)
- Advanced: [3D Turbulence](ivp_3d_turbulence.md), [Channel Flow](../notebooks/channel_flow.md)

**Geophysical Flows**:
- Start: [Surface Dynamics](surface_dynamics.md) (SQG, QG)
- Intermediate: [Rotating Shallow Water](rotating_shallow_water.md) (wave-mean separation)
- Advanced: [GQL Approximation](../pages/gql_approximation.md) (zonal jets, turbulence closure)

**Heat Transfer**:
- Start: [First Steps](../getting_started/first_steps.md) (diffusion)
- Advanced: [2D Rayleigh-Bénard](ivp_2d_rbc.md) (convection)

**Stability Analysis**:
- Start: [Eigenvalue Problems](eigenvalue_problems.md)

### By Technique

**Want to learn**:
- **MPI parallelization** → [Running with MPI](../getting_started/running_with_mpi.md)
- **Boundary conditions** → [Boundary Conditions](boundary_conditions.md)
- **Output and analysis** → [Analysis and Output](analysis_and_output.md)
- **3D problems** → [3D Turbulence](ivp_3d_turbulence.md)

### By Complexity

**Beginner**: Basic concepts, single field
- [First Steps](../getting_started/first_steps.md)

**Intermediate**: Multiple fields, coupled equations
- [2D Rayleigh-Bénard](ivp_2d_rbc.md)
- [Boundary Conditions](boundary_conditions.md)
- [Analysis and Output](analysis_and_output.md)

**Advanced**: 3D problems, advanced analysis
- [3D Turbulence](ivp_3d_turbulence.md)
- [Eigenvalue Problems](eigenvalue_problems.md)

## Common Patterns

### Setting Up a Simulation

Every tutorial follows this pattern:

```julia
# 1. MPI initialization
using Tarang, MPI
MPI.Init()

# 2. Domain setup
coords = CartesianCoordinates(...)
dist = Distributor(coords, mesh=...)
bases = (basis1, basis2, ...)
domain = Domain(dist, bases)

# 3. Fields
field1 = ScalarField(...)
field2 = VectorField(...)

# 4. Problem
problem = IVP([field1, field2, ...])
add_equation!(problem, "...")
add_bc!(problem, "...")

# 5. Solver
solver = InitialValueSolver(problem, timestepper)

# 6. Time loop
while solver.sim_time < t_end
    step!(solver, dt)
end

# 7. Cleanup
MPI.Finalize()
```

### Adding Analysis

Common analysis tasks:

```julia
# CFL condition
cfl = CFL(problem)
add_velocity!(cfl, u)

# File output
handler = add_netcdf_handler(
    solver,
    "outputs",
    fields=[u, p, T],
    write_interval=0.1
)

# Custom diagnostics
function compute_diagnostics(solver, u, T)
    ke = 0.5 * mean(u.data .^ 2)
    temp_mean = mean(T.data)
    return (ke=ke, temp=temp_mean)
end
```

## Getting Help

If you're stuck on a tutorial:

1. **Check the complete example** at the end of each tutorial
2. **Look at the source code** in the `examples/` directory
3. **Search the API reference** for function documentation
4. **Ask on GitHub Discussions** for community help

## Contributing Tutorials

We welcome tutorial contributions! See the [Contributing Guide](../pages/contributing.md) for details on:
- Tutorial format and style
- Adding Jupyter notebooks
- Including visualizations
- Testing tutorial code

## Next Steps

Ready to start? Pick a tutorial from the list above or continue with:
- [2D Rayleigh-Bénard Convection](ivp_2d_rbc.md) for a complete fluid dynamics example
- [Boundary Conditions](boundary_conditions.md) to master BC specification
- [Running with MPI](../getting_started/running_with_mpi.md) for parallel computing details
