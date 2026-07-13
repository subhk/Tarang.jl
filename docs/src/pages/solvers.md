# Solvers

Solvers integrate PDEs in time or solve for steady states.

## InitialValueSolver

For time-dependent problems (IVP).

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
domain = Domain(dist, (xbasis, ybasis))

u = ScalarField(domain, "u")
problem = IVP([u])
add_parameters!(problem, nu=0.1)
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")
set!(u, (x, y) -> sin(x) * cos(y))

# Create solver
solver = InitialValueSolver(problem, RK222(); dt=0.001)

# Time stepping
t_end = 0.01
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
solver.state         # Vector of the problem's ScalarFields
solver.rhs_plan      # Compiled RHS plan; `solver.rhs_plan.is_compiled` is the fast-path flag
```

### Methods

```julia
# Advance one step
step!(solver)           # Use solver.dt
step!(solver, 5e-4)     # Step with the given dt, and adopt it as solver.dt

# Check stopping conditions
proceed(solver)         # Returns true if should continue

# Run the whole loop (handles stopping, CFL, outputs, callbacks)
run!(solver; stop_iteration=100, progress=false)

# Print a tree-style summary of the solver configuration
diagnose(solver)
```

### Internal solver build path

At solver construction, `InitialValueSolver(problem, timestepper; dt=...)` runs several build stages in order:

1. **`build_solver_matrices!(solver)`** — parses the equation strings and assembles the **global** `L_matrix`, `M_matrix`, and `F_vector` for legacy fall-back paths. These are stored in `problem.parameters`.
2. **`_try_build_subproblems!(solver)`** — this is the fast path. It decomposes the problem into per-Fourier-mode subproblems via `build_subproblems` in `src/core/subsystems/`, building small sparse `L_min` / `M_min` matrices per mode, applying left/right permutations, and running valid-mode filtering to drop trivially-satisfied constraint rows (like `integ(p) = 0` at non-DC modes). The resulting `Tuple{Vararg{Subproblem}}` is stored in `problem.parameters["subproblems"]` and drives the modern IMEX stepper.
3. **`build_lazy_rhs_plan!(solver)`** — walks each equation's `F` expression and translates it into a type-parametric `LazyFuture` tree (in `src/core/solvers/lazy_rhs.jl`). Each node (`LazyAdd`, `LazySub`, `LazyMul`, `LazyDiv`, `LazyPow`, `LazyUnaryFunc`, `LazyDiff`, `LazyMultiDiff`, `LazyStateField`, `LazyParamField`, `LazyConst`, …) has a specialized `evaluate_lazy!` method. Julia's JIT then specializes the whole `evaluate_lazy!` call chain on first use — eliminating dynamic dispatch and fusing broadcast operations. If translation fails for any equation, the plan's `is_compiled` flag stays `false` and `evaluate_rhs` falls back to the interpreted expression path **for the entire solver**. See [the RHS compilation section](@ref rhs-compilation) — the fallback is ~100× slower, so it is worth checking (`solver.rhs_plan.is_compiled`, or the `diagnose(solver)` output).
4. **`_apply_bc_values_to_equations!(solver, 0.0)`** — only runs if there are time- or space-dependent BCs. Populates the initial `equation_data[eq_idx]["F"]` slots with the BC values evaluated at `t=0`, and auto-registers global coordinate arrays in `problem.bc_manager.coordinate_fields` so user BC expressions referencing `x`, `y`, `z`, `t` can be resolved at runtime.

All four steps are transparent to the user. You only interact with the resulting `solver` object.

## BoundaryValueSolver

`BoundaryValueSolver` solves both steady problem types: it dispatches on the problem,
solving an `LBVP` with a single linear solve and an `NLBVP` with Newton iteration.
There is no separate nonlinear solver type.

### Linear (LBVP)

Manufactured Poisson problem `Δu = -2` on `z ∈ [0, 1]` with `u(0) = u(1) = 0`,
whose exact solution is `u(z) = z(1-z)`:

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))
zbasis = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
domain = Domain(dist, (xbasis, zbasis))

u    = ScalarField(domain, "u")
tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
lb   = derivative_basis(zbasis, 2)

problem = LBVP([u, tau1, tau2])
add_parameters!(problem; Lz=1.0, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=Lz) = 0")

# Create solver and solve
solver = BoundaryValueSolver(problem)
solve!(solver)

# Solution is in the field variables
ensure_layout!(u, :g)
get_grid_data(u)          # matches z(1-z) to 1.4e-16
```

### Nonlinear (NLBVP)

Declare the problem as an `NLBVP` and put the nonlinear terms on the right-hand side.
The same `BoundaryValueSolver` then runs a per-Fourier-mode Newton iteration, rebuilding
the Frechet Jacobian at the current state each iteration.

```julia
# Δu + lift(τ) = u² + g   with   g = -2 - (z(1-z))²
# chosen so the equation reduces to Δu = -2 and the exact solution is again z(1-z).
g = ScalarField(domain, "g")
ensure_layout!(g, :g)
zg = Tarang.create_meshgrid(domain; on_device=false)["z"]
get_grid_data(g) .= -2 .- (zg .* (1 .- zg)) .^ 2

problem = NLBVP([u, tau1, tau2])
add_parameters!(problem; Lz=1.0, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2), g=g)
add_equation!(problem, "Δ(u) + l1 + l2 = u*u + g")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=Lz) = 0")

ensure_layout!(u, :g)
get_grid_data(u) .= 0.0   # Newton initial guess

solver = BoundaryValueSolver(problem; tolerance=1e-10, max_iterations=50)
solve!(solver)            # Newton; returns the solver, warns if it does not converge
```

Only `tolerance` and `max_iterations` are Newton knobs (defaults `1e-10` and `100`);
they can also be set after construction (`solver.tolerance = 1e-8`). `solve!` always
returns the solver and leaves the solution in the field variables — it does not return a
convergence flag. Non-convergence is reported as a warning naming the final residual:

```
┌ Warning: NLBVP per-mode Newton did not reach tolerance 1.0e-10 in 100 iters (final |F|=…)
```

## EigenvalueSolver

For eigenvalue problems (EVP). The eigenvalue symbol declared with `eigenvalue=` replaces
`dt(...)` in the equations, so `dt(u) - Δ(u) = 0` becomes the generalized problem
`σ M u + L u = 0` and the returned values are the growth rates `σ`.

```julia
using Tarang

coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
zbasis = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 1.0))
domain = Domain(dist, (zbasis,))

u    = ScalarField(domain, "u")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lb   = derivative_basis(zbasis, 2)

problem = EVP([u, tau1, tau2]; eigenvalue=:σ)
add_parameters!(problem; Lz=1.0, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
add_equation!(problem, "dt(u) - Δ(u) - l1 - l2 = 0")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=Lz) = 0")

# Create solver
solver = EigenvalueSolver(problem;
    nev=4,        # Number of eigenvalues
    which="SM"    # Smallest magnitude
)

# Solve
eigenvalues, eigenvectors = solve!(solver)
# eigenvalues ≈ [-9.8696, -39.4784, -88.8264, -157.9137] = -(nπ)², the Dirichlet Laplacian
```

`nev`, `which` and `target` may be given to the constructor or overridden at solve time
(`solve!(solver; nev=8, which="LR")`); the values used are stored back on the solver.
Passing `target=0.0+0.0im` selects the `nev` eigenvalues closest to the target instead of
using `which`.

Eigenvectors come back only when the problem has exactly one active subproblem — as above,
where there is no separable Fourier axis. A Fourier axis makes every mode its own
subproblem, and `solve!` then returns the eigenvalues pooled over all modes together with
an empty `0×0` eigenvector matrix.

### Which Eigenvalues

- `"LM"`: Largest magnitude (default)
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

`CFL` is built from an **`InitialValueSolver`** (not from the problem), and velocities are
registered as `VectorField`s. The simplest way to use it is to hand it to `run!`, which
recomputes `solver.dt` before each step:

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (xbasis, ybasis))

s = ScalarField(domain, "s")
u = VectorField(domain, "u")

problem = IVP([s, u])
add_parameters!(problem, nu=0.05)
add_equation!(problem, "∂t(s) - nu*lap(s) = -u⋅∇(s)")
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")
set!(s, (x, y) -> sin(x) * cos(y))
set!(u.components[1], (x, y) -> 0.5)

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Create CFL controller from the SOLVER
cfl = CFL(solver;
    initial_dt=1e-3,  # Starting timestep
    cadence=5,        # Recompute dt every 5 iterations
    safety=0.4,       # Safety factor
    threshold=0.1,    # Only commit a dt change larger than 10% (avoids LU rebuilds)
    max_change=1.5,   # Max dt increase per update
    min_change=0.5,   # Max dt decrease per update
    max_dt=0.01       # Upper limit on dt
)

# Register velocity field (VectorField only)
add_velocity!(cfl, u)

# run! drives the controller
run!(solver; stop_iteration=10, cfl=cfl, progress=false)
solver.dt          # 0.01 — updated by the controller
```

If you write the loop yourself, ask the controller for the new step and assign it:

```julia
solver.stop_iteration = 20
while proceed(solver)
    solver.dt = compute_timestep(cfl)
    step!(solver)
end
```

### CFL Parameters

All are keyword arguments to `CFL(solver; …)` and are also live fields of the returned
object (`cfl.max_dt = 0.02` after construction is fine).

| Parameter | Default | Typical Value | Description |
|-----------|---------|---------------|-------------|
| initial_dt | 0.01 | your starting dt | dt used before the first recomputation |
| cadence | 1 | 1-10 | Recompute dt every N iterations |
| safety | 0.4 | 0.3-0.5 | Lower = more stable |
| threshold | 0.1 | 0.0-0.2 | Relative change below which dt is kept (0 = commit every change) |
| max_change | 2.0 | 1.2-2.0 | Smooth dt increases |
| min_change | 0.5 | 0.5 | Prevent sudden drops |
| max_dt | Inf | your output cadence | Hard ceiling on dt |

There is no `min_dt`: the floor on the timestep is expressed as the *ratio* `min_change`,
not as an absolute value. The current step is `cfl.current_dt`.

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

`run!` sets the same three fields from its `stop_time`, `stop_wall_time` and
`stop_iteration` keywords, so `run!(solver; stop_time=10.0)` is equivalent to the loop
above.

## Complete Example

```julia
using Tarang

# Domain
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (xbasis, ybasis))

# Fields and problem: a scalar advected by a fixed velocity, with diffusion
s = ScalarField(domain, "s")
u = VectorField(domain, "u")

problem = IVP([s, u])
add_parameters!(problem, nu=0.01)
add_equation!(problem, "∂t(s) - nu*lap(s) = -u⋅∇(s)")
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")

set!(s, (x, y) -> sin(x) * cos(y))
set!(u.components[1], (x, y) -> 1.0)

# Solver
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# CFL
cfl = CFL(solver; initial_dt=1e-3, cadence=10, safety=0.5, max_dt=0.01)
add_velocity!(cfl, u)

# Run: stopping, adaptive dt and logging in one call
run!(solver; stop_time=0.2, cfl=cfl,
     callbacks=[(20, sol -> println("t = $(sol.sim_time), dt = $(sol.dt)"))],
     progress=false)
```

The same script runs under MPI without modification — launch it with `mpiexec` and see
[Parallelism](parallelism.md).

## See Also

- [Problems](problems.md): Problem definition
- [Timesteppers](timesteppers.md): Time integration details
- [API: Solvers](../api/solvers.md): Complete reference
