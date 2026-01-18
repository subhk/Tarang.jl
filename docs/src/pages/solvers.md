# Solvers

Solvers integrate PDEs in time or solve for steady states.

## InitialValueSolver

For time-dependent problems (IVP).

```julia
using Tarang

# Create solver
solver = InitialValueSolver(problem, RK222(); dt=0.001)

# Time stepping
while solver.sim_time < t_end
    step!(solver)
end
```

### Properties

```julia
solver.problem       # The IVP problem
solver.timestepper   # Time integration scheme
solver.dt            # Current timestep
solver.sim_time      # Current simulation time
solver.iteration     # Iteration count
```

### Methods

```julia
# Advance one step
step!(solver)           # Use solver.dt
step!(solver, dt)       # Use specified dt

# Check stopping conditions
proceed(solver)         # Returns true if should continue
```

## BoundaryValueSolver

For steady linear problems (LBVP).

```julia
# Create solver
solver = BoundaryValueSolver(problem)

# Solve
solve!(solver)

# Solution is in the field variables
```

## NonlinearBoundaryValueSolver

For steady nonlinear problems (NLBVP).

```julia
# Create solver with Newton-Raphson
solver = NonlinearBoundaryValueSolver(problem;
    tolerance=1e-8,
    max_iterations=100,
    damping=1.0
)

# Solve
success = solve!(solver)

if success
    println("Converged in $(solver.iterations) iterations")
end
```

## EigenvalueSolver

For eigenvalue problems (EVP).

```julia
# Create solver
solver = EigenvalueSolver(problem;
    nev=10,           # Number of eigenvalues
    which="LR",       # Largest real part
    target=0.0+0.0im  # Target for shift-invert
)

# Solve
eigenvalues, eigenvectors = solve!(solver)
```

### Which Eigenvalues

- `"LM"`: Largest magnitude
- `"SM"`: Smallest magnitude
- `"LR"`: Largest real part (most unstable)
- `"SR"`: Smallest real part
- `"LI"`: Largest imaginary part
- `"SI"`: Smallest imaginary part

## Time Steppers

### IMEX Runge-Kutta

```julia
RK111()  # 1st order, 1 stage
RK222()  # 2nd order, 2 stages (recommended)
RK443()  # 3rd order, 4 stages (higher accuracy)
```

### IMEX Multistep Methods

For problems with stiff linear terms:

```julia
CNAB1()  # Crank-Nicolson Adams-Bashforth, 1st order
CNAB2()  # Crank-Nicolson Adams-Bashforth, 2nd order

SBDF1()  # Semi-implicit BDF, 1st order
SBDF2()  # Semi-implicit BDF, 2nd order
SBDF3()  # Semi-implicit BDF, 3rd order
SBDF4()  # Semi-implicit BDF, 4th order
```

### Choosing a Timestepper

| Problem Type | Recommended |
|--------------|-------------|
| General purpose | RK222, RK443 |
| Mildly stiff | CNAB2, SBDF2 |
| Very stiff | SBDF3, SBDF4 |

## Adaptive Time Stepping

### CFL Condition

```julia
using Tarang

# Create CFL calculator
cfl = CFL(problem;
    safety=0.5,      # Safety factor
    max_change=1.5,  # Max dt increase per step
    min_change=0.5   # Max dt decrease per step
)

# Register velocity field
add_velocity!(cfl, u)

# Set limits
cfl.max_dt = 0.01
cfl.min_dt = 1e-8

# In time loop
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

### CFL Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| safety | 0.3-0.5 | Lower = more stable |
| max_change | 1.2-2.0 | Smooth dt changes |
| min_change | 0.5 | Prevent sudden drops |

## Stopping Conditions

```julia
# Set stop conditions
solver.stop_sim_time = 10.0      # Stop at t=10
solver.stop_wall_time = 3600.0   # Stop after 1 hour
solver.stop_iteration = 10000    # Stop after 10000 steps

# Use proceed() to check
while proceed(solver)
    step!(solver)
end
```

## Complete Example

```julia
using Tarang
using MPI

MPI.Init()

# Setup (abbreviated)
# ... create domain, fields, problem ...

# Solver
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# CFL
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

# Stop conditions
solver.stop_sim_time = 10.0

# Main loop
while proceed(solver)
    dt = compute_timestep(cfl)
    step!(solver, dt)

    if solver.iteration % 100 == 0
        println("t = $(solver.sim_time), dt = $dt")
    end
end

MPI.Finalize()
```

## See Also

- [Problems](problems.md): Problem definition
- [Timesteppers](timesteppers.md): Time integration details
- [API: Solvers](../api/solvers.md): Complete reference
