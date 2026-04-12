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

# Print a tree-style summary of the solver configuration
diagnose(solver)
```

### Internal solver build path

At solver construction, `InitialValueSolver(problem, timestepper; dt=...)` runs several build stages in order:

1. **`build_solver_matrices!(solver)`** — parses the equation strings and assembles the **global** `L_matrix`, `M_matrix`, and `F_vector` for legacy fall-back paths. These are stored in `problem.parameters`.
2. **`_try_build_subproblems!(solver)`** — this is the fast path. It decomposes the problem into per-Fourier-mode subproblems via `build_subproblems` in `src/core/subsystems.jl`, building small sparse `L_min` / `M_min` matrices per mode, applying left/right permutations, and running valid-mode filtering to drop trivially-satisfied constraint rows (like `integ(p) = 0` at non-DC modes). The resulting `Tuple{Vararg{Subproblem}}` is stored in `problem.parameters["subproblems"]` and drives the modern IMEX stepper.
3. **`build_lazy_rhs_plan!(solver)`** — walks each equation's `F` expression and translates it into a type-parametric `LazyFuture` tree (in `src/core/solvers/lazy_rhs.jl`). Each node (`LazyAdd`, `LazyMul`, `LazyDiff`, `LazyStateField`, `LazyParamField`, `LazyConst`) has a specialized `evaluate_lazy!` method. Julia's JIT then specializes the whole `evaluate_lazy!` call chain on first use — eliminating dynamic dispatch and fusing broadcast operations. If translation fails for any equation (unsupported operator type), the plan's `is_compiled` flag stays `false` and `evaluate_rhs` transparently falls back to the interpreted expression path. You can check whether the plan compiled in the `diagnose(solver)` output.
4. **`_apply_bc_values_to_equations!(solver, 0.0)`** — only runs if there are time- or space-dependent BCs. Populates the initial `equation_data[eq_idx]["F"]` slots with the BC values evaluated at `t=0`, and auto-registers global coordinate arrays in `problem.bc_manager.coordinate_fields` so user BC expressions referencing `x`, `y`, `z`, `t` can be resolved at runtime.

All four steps are transparent to the user. You only interact with the resulting `solver` object.

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
