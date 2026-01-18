# Solvers API

Solvers integrate PDEs in time or solve for steady states. Tarang.jl provides specialized solvers for different problem types.

## Solver Types

### InitialValueSolver

Time-stepping solver for Initial Value Problems (IVP).

**Constructor**:
```julia
InitialValueSolver(
    problem::IVP,
    timestepper::TimeStepper;
    dt::Float64=0.001,
    device::String="cpu"
)
```

**Arguments**:
- `problem`: IVP problem definition
- `timestepper`: Time integration scheme (RK222, CNAB2, etc.)
- `dt`: Initial timestep

**Examples**:

```julia
# IMEX Runge-Kutta
solver = InitialValueSolver(problem, RK222(), dt=0.001)

# IMEX timestepper
solver = InitialValueSolver(problem, CNAB2(), dt=0.01)

# Higher-order timestepper
solver = InitialValueSolver(problem, RK443(), dt=0.001)
```

**Properties**:
```julia
solver.problem          # IVP problem
solver.timestepper      # Time integration scheme
solver.dt               # Current timestep
solver.sim_time         # Current simulation time
solver.iteration        # Current iteration number
solver.wall_time        # Elapsed wall clock time
solver.state            # Current state vector
```

**Methods**:

#### step!

Advance solution by one timestep.

```julia
step!(solver)           # Use solver.dt
step!(solver, dt)       # Use specified dt
```

**Example**:

```julia
# Fixed timestep
while solver.sim_time < t_end
    step!(solver, 0.001)
end

# Adaptive timestep (with CFL)
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

#### proceed

Check if simulation should continue.

```julia
# Set stop conditions
solver.stop_sim_time = 10.0
solver.stop_wall_time = 3600.0  # 1 hour
solver.stop_iteration = 10000

while proceed(solver)
    step!(solver)
end
```

---

### BoundaryValueSolver

Solver for Linear Boundary Value Problems (LBVP).

**Constructor**:
```julia
BoundaryValueSolver(
    problem::LBVP;
    solver_type::String="direct",
    tolerance::Float64=1e-10
)
```

**Arguments**:
- `problem`: LBVP problem definition
- `solver_type`: Solution method ("direct", "iterative", "multigrid")
- `tolerance`: Convergence tolerance for iterative solvers

**Examples**:

```julia
# Direct solver (for small problems)
solver = BoundaryValueSolver(problem, solver_type="direct")

# Iterative solver (for large problems)
solver = BoundaryValueSolver(problem, solver_type="iterative", tolerance=1e-8)
```

**Methods**:

#### solve!

Solve the boundary value problem.

```julia
solve!(solver)
```

**Returns**: Solution fields are updated in place

**Example**:

```julia
# Poisson equation
problem = LBVP([phi])
add_equation!(problem, "Δ(phi) = rho")
add_equation!(problem, "phi(z=0) = 0")
add_equation!(problem, "phi(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)

# Solution is now in phi field
phi_grid = get_grid_data(phi)
```

---

### NonlinearBoundaryValueSolver

Newton-Raphson solver for Nonlinear Boundary Value Problems (NLBVP).

**Constructor**:
```julia
NonlinearBoundaryValueSolver(
    problem::NLBVP;
    tolerance::Float64=1e-8,
    max_iterations::Int=100,
    damping::Float64=1.0
)
```

**Arguments**:
- `problem`: NLBVP problem definition
- `tolerance`: Convergence tolerance
- `max_iterations`: Maximum Newton iterations
- `damping`: Damping factor for Newton steps (0 < damping ≤ 1)

**Examples**:

```julia
# Standard Newton solver
solver = NonlinearBoundaryValueSolver(problem, tolerance=1e-8)

# With damping for stability
solver = NonlinearBoundaryValueSolver(problem, damping=0.5, max_iterations=50)
```

**Methods**:

#### solve!

Solve using Newton-Raphson iteration.

```julia
success = solve!(solver)
```

**Returns**: `true` if converged, `false` otherwise

**Example**:

```julia
# Steady Navier-Stokes
problem = NLBVP([u, v, p])
add_equation!(problem, "u*∂x(u) + v*∂z(u) + ∂x(p) = nu*Δ(u)")
add_equation!(problem, "u*∂x(v) + v*∂z(v) + ∂z(p) = nu*Δ(v)")
add_equation!(problem, "∂x(u) + ∂z(v) = 0")

solver = NonlinearBoundaryValueSolver(problem)

if solve!(solver)
    println("Converged in $(solver.iterations) iterations")
else
    println("Failed to converge")
end
```

**Properties**:
```julia
solver.iterations       # Number of iterations performed
solver.residual         # Final residual norm
solver.converged        # Convergence status
```

---

### EigenvalueSolver

Solver for Eigenvalue Problems (EVP).

**Constructor**:
```julia
EigenvalueSolver(
    problem::EVP;
    nev::Int=10,
    target::Complex{Float64}=0.0+0.0im,
    which::String="LM"
)
```

**Arguments**:
- `problem`: EVP problem definition
- `nev`: Number of eigenvalues to compute
- `target`: Target eigenvalue for shift-invert
- `which`: Which eigenvalues to find ("LM", "SM", "LR", "SR", "LI", "SI")

**Which options**:
- "LM": Largest magnitude
- "SM": Smallest magnitude
- "LR": Largest real part
- "SR": Smallest real part
- "LI": Largest imaginary part
- "SI": Smallest imaginary part

**Examples**:

```julia
# Find 10 eigenvalues with largest growth rate
solver = EigenvalueSolver(problem, nev=10, which="LR")

# Find eigenvalues near target
solver = EigenvalueSolver(problem, nev=5, target=0.1+1.5im)

# Most unstable modes
solver = EigenvalueSolver(problem, nev=20, which="LR")
```

**Methods**:

#### solve!

Compute eigenvalues and eigenvectors.

```julia
eigenvalues, eigenvectors = solve!(solver)
```

**Returns**:
- `eigenvalues`: Array of complex eigenvalues
- `eigenvectors`: Array of eigenvector fields

**Example**:

```julia
# Rayleigh-Bénard stability
problem = EVP([u, v, p, T], eigenvalue=:sigma)
# ... add equations and BCs ...

solver = EigenvalueSolver(problem, nev=10, which="LR")
eigenvalues, eigenvectors = solve!(solver)

# Find critical mode
max_idx = argmax(real.(eigenvalues))
growth_rate = real(eigenvalues[max_idx])
frequency = imag(eigenvalues[max_idx])

println("Maximum growth rate: $growth_rate")
println("Frequency: $frequency")

# Extract critical mode
critical_mode = eigenvectors[max_idx]
```

---

## Solver State Management

### Saving and Loading State

```julia
# Save solver state
save_state(solver, "checkpoint.h5")

# Load solver state
load_state!(solver, "checkpoint.h5")

# Resume simulation
while proceed(solver)
    step!(solver)
end
```

### State Vector Access

```julia
# Get state vector
state = get_state(solver)

# Set state vector
set_state!(solver, new_state)

# Useful for custom initialization or analysis
```

---

## Time Integration

### Time Stepping Loop

Basic pattern for time integration:

```julia
# Create solver
solver = InitialValueSolver(problem, RK222(), dt=0.001)

# Set stop conditions
t_end = 10.0

# Time loop
while solver.sim_time < t_end
    step!(solver)

    # Optional: output and diagnostics
    if solver.iteration % 100 == 0
        println("t = $(solver.sim_time), iteration = $(solver.iteration)")
    end
end
```

### Adaptive Time Stepping

With CFL condition:

```julia
# Create CFL calculator
cfl = CFL(problem, safety=0.5, max_change=1.5, min_change=0.5)
add_velocity!(cfl, u)

# Adaptive loop
while solver.sim_time < t_end
    # Compute adaptive timestep
    dt = compute_timestep(cfl)

    # Take step
    step!(solver, dt)
end
```

### Output During Integration

```julia
# Setup output handler
output = add_netcdf_handler(
    solver,
    "output",
    fields=[u, p, T],
    write_interval=0.1
)

# Time loop with automatic output
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
    # Output written automatically when t > next_write_time
end
```

---

## Solver Options

### Convergence Criteria

For iterative and nonlinear solvers:

```julia
solver.tolerance = 1e-10           # Residual tolerance
solver.max_iterations = 1000       # Maximum iterations
solver.relative_tolerance = 1e-8   # Relative change tolerance
```

### Performance Options

```julia
# Parallel options
solver.use_threading = true
solver.num_threads = 4

# Memory options
solver.preallocate_work = true
solver.work_array_count = 5
```

---

## Advanced Solver Features

### Preconditioners

For iterative solvers:

```julia
# Set preconditioner type
solver.preconditioner = "jacobi"  # or "ilu", "multigrid"

# Custom preconditioner
function my_preconditioner(A, b)
    # Custom preconditioning operation
    return P \ b
end

solver.preconditioner = my_preconditioner
```

### Matrix-Free Methods

For large problems:

```julia
# Use matrix-free Krylov methods
solver.matrix_free = true
solver.krylov_method = "gmres"  # or "bicgstab", "cg"
```

### Continuation Methods

For nonlinear problems:

```julia
# Parameter continuation
Ra_values = [1e4, 5e4, 1e5, 5e5, 1e6]

solution = nothing
for Ra in Ra_values
    problem.parameters["Ra"] = Ra

    if solution !== nothing
        # Use previous solution as initial guess
        set_state!(solver, solution)
    end

    solve!(solver)
    solution = get_state(solver)

    println("Solved for Ra = $Ra")
end
```

---

## Solver Diagnostics

### Convergence Monitoring

```julia
# Enable convergence monitoring
solver.verbose = true
solver.print_interval = 10

# Custom convergence callback
function convergence_callback(solver, iteration, residual)
    if iteration % 10 == 0
        println("Iteration $iteration: residual = $residual")
    end
end

solver.convergence_callback = convergence_callback
```

### Performance Profiling

```julia
# Enable profiling
solver.profile = true

# After solving
print_performance_summary(solver)
# Outputs:
# - Time per iteration
# - Time in different solver phases
# - Memory usage
# - Communication time (MPI)
```

---

## Error Handling

### Solver Failures

```julia
try
    solve!(solver)
catch e
    if isa(e, ConvergenceError)
        println("Failed to converge: $(e.message)")
        println("Residual: $(e.residual)")
    elseif isa(e, LinearAlgebraError)
        println("Linear algebra error: $(e.message)")
    else
        rethrow(e)
    end
end
```

### Recovery Strategies

```julia
# For nonlinear solvers
if !solver.converged
    # Try with damping
    solver.damping = 0.5
    solve!(solver)

    if !solver.converged
        # Try with better initial guess
        # ... provide better initialization
        solve!(solver)
    end
end
```

---

## Complete Example

### Time-Dependent Simulation

```julia
using Tarang, MPI

MPI.Init()

# Setup problem (2D Rayleigh-Bénard)
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

domain = Domain(dist, (x_basis, z_basis))

u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))

# Define problem
problem = IVP([u.components[1], u.components[2], p, T])
problem.parameters["Ra"] = 1e6
problem.parameters["Pr"] = 1.0

add_equation!(problem, "∂t(u) + u*∂x(u) + w*∂z(u) + ∂x(p) = Pr*Δ(u)")
add_equation!(problem, "∂t(w) + u*∂x(w) + w*∂z(w) + ∂z(p) = Pr*Δ(w) + Ra*Pr*T")
add_equation!(problem, "∂x(u) + ∂z(w) = 0")
add_equation!(problem, "∂t(T) + u*∂x(T) + w*∂z(T) = Δ(T)")

# Boundary conditions
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "w(z=0) = 0")
add_equation!(problem, "T(z=0) = 1")
add_equation!(problem, "u(z=1) = 0")
add_equation!(problem, "w(z=1) = 0")
add_equation!(problem, "T(z=1) = 0")

# Initialize fields
# ... set initial conditions ...

# Create solver
solver = InitialValueSolver(problem, RK222(), dt=1e-4)

# CFL condition
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

# Output
output = add_netcdf_handler(solver, "output", fields=[u, p, T], write_interval=0.1)

# Time integration
t_end = 10.0
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)

    if solver.iteration % 100 == 0 && MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("t = $(solver.sim_time), dt = $dt")
    end
end

MPI.Finalize()
```

---

## See Also

- [Problems](problems.md): Problem definition
- [Timesteppers](timesteppers.md): Time integration schemes
- [Analysis](analysis.md): CFL conditions and diagnostics
- [Tutorial: IVP](../tutorials/ivp_2d_rbc.md): Complete example
