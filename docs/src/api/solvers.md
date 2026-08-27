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
    dt::Real=1e-3,
    device::String="cpu",
    matsolver=:sparse
)
```

**Arguments**:
- `problem`: IVP problem definition
- `timestepper`: Time integration scheme instance (`RK222()`, `CNAB2()`, …)
- `dt`: Initial timestep
- `matsolver`: Backend used for the implicit per-mode solves (see [Solver options](#Solver-options))

**Examples**:

```julia
# IMEX Runge-Kutta
solver = InitialValueSolver(problem, RK222(); dt=0.001)

# IMEX multistep
solver = InitialValueSolver(problem, CNAB2(); dt=0.01)

# Higher-order Runge-Kutta
solver = InitialValueSolver(problem, RK443(); dt=0.001)
```

The timestepper is passed as an **instance** — `RK222()`, not `RK222`.

**Properties**:
```julia
solver.problem            # IVP problem
solver.timestepper        # Time integration scheme
solver.dt                 # Current timestep
solver.sim_time           # Current simulation time
solver.iteration          # Current iteration number
solver.state              # Vector{ScalarField} — the solution fields
solver.rhs_plan           # Compiled RHS plan (see below)
solver.performance_stats  # total_time, total_steps, total_solves, avg_step_time
```

**Methods**:

#### run!

The recommended way to run a simulation. Drives the timestep loop, CFL-adaptive
dt, output handlers, progress, and callbacks — so a typical script needs no manual
`step!`/`process!`/`close!`:

```julia
# Handlers created with the SOLVER auto-register; run! writes them every step
# (at their own sim_dt/iter cadence) and closes them at the end.
snap = add_file_handler("output/snap", solver; sim_dt=0.5)
add_task!(snap, u; name="u")

run!(solver;
     stop_time=10.0,
     cfl=cfl,                       # adaptive dt = compute_timestep(cfl) each step
     log_interval=100,              # built-in progress line every 100 steps
     callbacks=Pair[
         100 => (s -> @printf("t=%.3f dt=%.2e max|u_x|=%.3e\n",
                              s.sim_time, s.dt, global_max(u.components[1]))),
         0.5 => (s -> println("KE = ", total_kinetic_energy(u))),   # every 0.5 sim-time units
     ])
```

**Arguments**:
- `stop_time`: stop when `sim_time >= stop_time`
- `stop_iteration`: stop after N iterations
- `stop_wall_time`: stop after N seconds of wall time
- `cfl`: a `CFL` controller — if given, `dt = compute_timestep(cfl)` each step
- `outputs`: extra output handlers to process each step (handlers created with the
  solver are already auto-registered; use this for ones created from a `dist`)
- `callbacks`: vector of `interval => function` pairs; the function receives the solver
- `log_interval`: print a built-in progress line every N iterations (0 to disable)
- `progress`: print start/finish `@info` lines (default `true`)

Callback (and output) intervals can be `Int` (every N iterations) or `Float64`
(every T sim-time units). Custom diagnostics — time, timestep, and field maxima
via `global_max`/`global_min`/`global_sum` — go in a callback. Those three reducers
take a **`ScalarField`**; for a `VectorField` pass a component (`u.components[1]`)
or use an aggregate such as `total_kinetic_energy(u)`.

!!! warning "Annotate the callback vector with `Pair[...]` when mixing interval types"
    An `Int` interval means *iterations*, a `Float64` interval means *sim-time*. In a
    plain `[…]` literal Julia promotes the element type, so
    `[100 => f, 0.5 => g]` becomes a `Vector{Pair{Float64,Function}}` and the `100`
    silently turns into `100.0` — "every 100 iterations" quietly becomes "every 100
    time units", and the callback never fires in a short run. Writing the literal as
    `Pair[100 => f, 0.5 => g]` keeps each interval's type intact.

#### step!

Advance solution by one timestep (for custom loops):

```julia
step!(solver)           # Use solver.dt
step!(solver, dt)       # Use specified dt
```

#### proceed

Check if simulation should continue — tests `sim_time`, `iteration` and
`wall_time` against the solver's stop conditions:

```julia
solver.stop_sim_time = 10.0

while proceed(solver)
    step!(solver)
end
```

#### diagnose

Print a formatted summary of solver state:

```julia
diagnose(solver)
```

Shows: timestepper, `dt`, sim time, iteration, architecture, MPI ranks, bases,
state fields and DOF, equations, compiled-RHS status, boundary conditions,
subproblem decomposition, stop conditions, and performance stats.

### [Compiled RHS](@id rhs-compilation)

During solver construction, Tarang.jl compiles each equation's explicit right-hand side into a type-specialized expression tree (the "lazy" RHS). This eliminates runtime type dispatch and per-timestep allocation.

If **any** term fails to translate, the compiled plan is unavailable for the
whole solver. Serial CPU solvers retain the interpreted compatibility path by
default. MPI and GPU solvers fail during construction instead of selecting an
unsafe fallback.

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

solver.rhs_plan.is_compiled   # true  ⇒ fast path
                              # false ⇒ interpreted fallback
diagnose(solver)              # reports the same, with the rest of the setup
```

!!! warning "Interpreted execution is an explicit compatibility mode"
    The interpreted path is roughly **100× slower** per RHS evaluation. Use
    `rhs_fallback=:strict` to require compilation on a serial CPU solver. A
    verified CPU/MPI compatibility case can opt in with
    `rhs_fallback=:interpreted`. GPU execution rejects that option: it never
    enters the interpreted path. Distributed all-Fourier fallback is known to
    be incorrect and is also rejected even when compatibility mode is requested.

**What compiles.** Arithmetic (`+ - * / ^`), scalar functions (`sin`, `exp`, `tanh`, …), field and parameter references, derivatives (`∂x(u)`, `d(u,x)`), and the vector/tensor operators that reduce to a scalar: `lap(u)`, `div(v)`, `div(grad(u))`.

**What does not, and why.** `curl` (and anything built on it) has no scalar form
and is not translated. Serial CPU solvers may use the compatibility evaluator;
strict solvers reject it at construction.

Two further cases deliberately decline compilation for every spelling,
including bare derivatives (`∂z(u)`, `d(u,z)`, or advection):

| Case | Behavior |
|---|---|
| Derivative along a **distributed non-Fourier (Chebyshev) axis** | Strict by default. The verified grid-space evaluator can be requested with `rhs_fallback=:interpreted`; moving the term to the implicit side is faster. |
| **Legendre** / non-Chebyshev Jacobi bases | Declined because the compiled differentiation matrix uses a different normalization. Serial CPU compatibility evaluation remains correct. |

!!! note "Behavior change"
    `lap(u*v)` and `div(grad(u))` in an **explicit** RHS used to raise an error inside the interpreted evaluator, which was caught and swallowed — silently zeroing that equation's *entire* right-hand side. Scripts written that way were integrating `F ≡ 0`. They now compile and produce correct results, so their output will differ from earlier versions.

---

### BoundaryValueSolver

Steady-state solver. It handles **both** linear (`LBVP`) and nonlinear (`NLBVP`)
boundary value problems — there is no separate nonlinear solver type; the problem
type selects the algorithm (direct per-mode solve for an LBVP, per-mode Newton
iteration with a symbolic Frechet Jacobian for an NLBVP).

**Constructor**:
```julia
BoundaryValueSolver(
    problem::Union{LBVP, NLBVP};
    device::String="cpu",
    matsolver=:sparse,
    solver_type=nothing,
    tolerance::Real=1e-10,
    max_iterations::Int=100
)
```

**Arguments**:
- `problem`: LBVP or NLBVP problem definition
- `matsolver`: matrix-solver backend for the per-mode tau systems (see [Solver options](#Solver-options))
- `solver_type`: alias for `matsolver`; when it is `nothing` (the default), `matsolver` is used
- `tolerance`: Newton convergence tolerance (NLBVP only; ignored for an LBVP)
- `max_iterations`: maximum Newton iterations (NLBVP only)

**Methods**:

#### solve!

Solve the boundary value problem. The solution is written **in place** into the
problem's fields (in coefficient layout); `solve!` returns the solver.

```julia
solve!(solver)
```

**Example** — 2D Poisson `Δφ = -2` with `φ = 0` on both `z` walls (exact solution
`φ = z(Lz - z)`; the run below reproduces it to 1.4e-16):

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb  = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))
zb  = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (xb, zb))

phi  = ScalarField(dom, "phi")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)   # one tau variable per BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([phi, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(phi) + l1 + l2 = -2")
add_bc!(problem, "phi(z=0)   = 0")     # BCs use add_bc!, never add_equation!
add_bc!(problem, "phi(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)

ensure_layout!(phi, :g)                # the solve writes coefficients
phi_grid = get_grid_data(phi)
```

The bounded (Chebyshev) direction needs explicit `tau` variables lifted into the
bulk equation — see [Problems](problems.md) for the tau method.

#### Nonlinear problems (NLBVP)

An `NLBVP` is passed to the same `BoundaryValueSolver`. Put the nonlinear terms on
the right-hand side; the solver linearizes them symbolically and runs a
per-Fourier-mode Newton iteration, rebuilding the Jacobian at the current state
each iteration. The current field values are the **initial guess**.

```julia
# same domain/tau setup as above, plus a forcing field g
problem = NLBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), g=g)
add_equation!(problem, "Δ(u) + l1 + l2 = u*u + g")   # nonlinearity on the RHS
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solver.tolerance = 1e-10          # Newton residual tolerance
solver.max_iterations = 100

ensure_layout!(u, :g); get_grid_data(u) .= 0.0       # initial guess
solve!(solver)
ensure_layout!(u, :g)
```

With `g = -2 - u_exact²` this manufactured problem has the exact solution
`u = z(Lz - z)`, which the Newton solve recovers to ~1.9e-12.

`solve!` returns the solver, not a convergence flag. If Newton does not reach
`tolerance` within `max_iterations` the per-mode path **warns** and leaves the last
iterate in the fields; check the residual yourself if you need a hard failure.

!!! warning "One solver per problem object"
    Solver construction merges the problem's `add_bc!` conditions into its equation
    list. Doing it twice appends them twice (1 equation → 3 → 5), so building a
    **second** solver from a `Problem` that already has one produces an
    over-determined system: the next `solve!` dies with `DimensionMismatch` (BVP) or
    the build itself with `Number of equations (5) does not match number of variables
    (3)` (EVP). To sweep a parameter, rebuild the problem from scratch — the fields
    persist, so the previous solution carries over as the next initial guess.

---

### EigenvalueSolver

Solver for Eigenvalue Problems (EVP).

**Constructor**:
```julia
EigenvalueSolver(
    problem::EVP;
    nev::Int=10,
    which::Union{String,Symbol}=:LM,
    target::Union{Nothing,ComplexF64}=nothing,
    matsolver=:sparse
)
```

**Arguments**:
- `problem`: EVP problem definition
- `nev`: Number of eigenvalues to return
- `which`: Which eigenvalues to keep. Symbol or String — `:LM` and `"LM"` are equivalent
- `target`: Order by proximity to this shift instead of by `which`. The default,
  `nothing`, means **no shift** (not a zero shift)
- `matsolver`: matrix-solver backend (see [Solver options](#Solver-options))

**Which options**:
- `:LM`: Largest magnitude
- `:SM`: Smallest magnitude
- `:LR`: Largest real part
- `:SR`: Smallest real part
- `:LI`: Largest imaginary part
- `:SI`: Smallest imaginary part

**Eigenvalue convention**: the eigenvalue replaces the time derivative
(`dt(u) → σu`), so keep the `dt(·)` term in the equation — it is what builds the
mass matrix `M`. Returned eigenvalues are the growth rates `σ` of `u ~ e^{σt}`.

**Methods**:

#### solve!

Compute eigenvalues and eigenvectors.

```julia
eigenvalues, eigenvectors = solve!(solver)
```

**Returns**:
- `eigenvalues`: `Vector{ComplexF64}`, ordered by `which` (or by distance to `target`)
- `eigenvectors`: `Matrix{ComplexF64}` whose **columns** are the modes, in coefficient
  space. It is returned only when the problem has a single subproblem (no Fourier
  direction, serial); otherwise it comes back empty (`0×0`), because a per-mode
  eigenvector is defined per Fourier mode.

**Example** — 1D diffusion eigenproblem `σu = Δu` with Dirichlet walls. The exact
eigenvalues are `σ_n = -(nπ/Lz)²`; the run below returns
`-9.8696, -39.4784, -88.8264, …`, matching to ~1e-12:

```julia
coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
zb     = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))
dom    = Domain(dist, (zb,))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lb2  = derivative_basis(zb, 2)

problem = EVP([u, tau1, tau2]; eigenvalue=:σ)
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "dt(u) - Δ(u) - l1 - l2 = 0")   # dt(u) → σu marks M
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = EigenvalueSolver(problem; nev=5, which=:LR)   # 5 fastest-growing modes
eigenvalues, eigenvectors = solve!(solver)

# Most unstable mode
max_idx     = argmax(real.(eigenvalues))
growth_rate = real(eigenvalues[max_idx])
frequency   = imag(eigenvalues[max_idx])
println("Maximum growth rate: $growth_rate")
println("Frequency: $frequency")

critical_mode = eigenvectors[:, max_idx]   # columns are modes
```

To search near a shift instead, pass `target`:

```julia
solver = EigenvalueSolver(problem; nev=3, target=-40.0 + 0.0im)
eigenvalues, _ = solve!(solver)   # -> -39.48, -9.87, -88.83  (ordered by |σ - target|)
```

---

## Solver State

The solution lives in the fields themselves; `solver.state` is the vector of
`ScalarField`s the solver advances (a `VectorField` contributes one entry per
component). There is no separate state-vector object to get or set — initialize
and inspect the fields directly:

```julia
solver.state                       # Vector{ScalarField}: e.g. [s, u_x, u_y]

f = solver.state[1]
ensure_layout!(f, :g)              # switch to grid layout
get_grid_data(f)                   # the local grid array
```

Set initial conditions on the fields **before** stepping (`set!`, `fill_random!`,
or by writing into `get_grid_data`), and read them back the same way afterwards.

---

## Time Integration

`run!` is the recommended driver. The patterns below are for custom loops.

### Time Stepping Loop

```julia
solver = InitialValueSolver(problem, RK222(); dt=0.001)

solver.stop_sim_time = 10.0

while proceed(solver)
    step!(solver)

    if solver.iteration % 100 == 0
        println("t = $(solver.sim_time), iteration = $(solver.iteration)")
    end
end
```

### Adaptive Time Stepping

The `CFL` controller is constructed from the **solver** (not the problem), and
velocities are added as `VectorField`s:

```julia
cfl = CFL(solver; initial_dt=1e-3, cadence=1, safety=0.4,
          max_change=1.5, min_change=0.5, max_dt=0.01)
add_velocity!(cfl, u)          # u::VectorField

# Adaptive loop
while solver.sim_time < t_end
    solver.dt = compute_timestep(cfl)
    step!(solver)
end
```

Or — the usual form — hand the controller to `run!` and let it do the same thing:

```julia
run!(solver; stop_time=t_end, cfl=cfl)
```

See [Analysis](analysis.md) for the full `CFL` kwarg list.

### Output During Integration

Create the handler **with the solver**: it then registers itself, and `run!`
processes it at its own cadence and closes it at the end.

```julia
output = add_file_handler("output/snap", solver; sim_dt=0.1, max_writes=10)
add_task!(output, u; name="u")        # pass field/operator OBJECTS, not strings

run!(solver; stop_time=t_end, cfl=cfl)
```

A handler created from a `dist` instead of a solver does **not** auto-register — and
its constructor needs a third positional `vars` namespace, which the solver form
supplies for you. Pass such a handler to `run!` explicitly, or it never writes:

```julia
h = add_file_handler("output/snap", dist, Dict{String,Any}(); sim_dt=0.1)
add_task!(h, u; name="u")

run!(solver; stop_time=t_end, outputs=[h])   # without outputs=, h is never processed
```

See [I/O](io.md) for the file layout and how to read the results back.

---

## Solver options

### Matrix solver backend

Every solver takes `matsolver`, which selects the backend used to factor the
per-mode tau systems. Registered names:

| name | backend |
|---|---|
| `:sparse`, `:lu`, `:direct` | sparse LU (default) |
| `:dense` | dense LU |
| `:banded` | banded LU |
| `:block`, `:blockdiagonal` | block-diagonal |
| `:spqr`, `:qr` | sparse rank-revealing QR |

```julia
solver = BoundaryValueSolver(problem; matsolver=:dense)
```

`matsolver=:iterative` is accepted but warns and falls back to the sparse direct
solver — there is no iterative or multigrid backend.

### Batched Fourier-mode solve

On a 2-D mixed Fourier-Chebyshev problem every Fourier mode is its own tau
system, and all of them share a shape and a sparsity pattern. `batched_modes`
solves them as one `(n, nmodes)` batch inside the IMEX Runge-Kutta stage loop —
one kernel launch per operation instead of one per mode.

`batched_modes` accepts `nothing` (the default), `true` or `false`. The default
means **batch on GPU, not on CPU**: what batching buys is a lower launch count,
which is a GPU cost, so no existing CPU run changes behavior unless it asks.
`true` opts a CPU run in (the test suite uses it to exercise the path without a
GPU); `false` turns batching off everywhere, GPU included.

`batched_modes_max_bytes` caps the memory one batch may allocate and defaults to
`1 << 30` (1 GiB). A batch holds a dense stage matrix per mode plus the shared
operator values, so its cost grows as `n^2 * nmodes` in the tau-system order `n`
and the mode count; a bucket whose batch would exceed the cap logs an `@info`
and stays on the per-mode loop rather than allocating.

Batching engages only when the run is serial (`nprocs == 1`) and the problem has
exactly one Fourier axis, i.e. is 2-D. MPI runs, 3-D runs, and buckets holding a
single mode stay on the per-mode path silently.

```julia
# force the batched path on a CPU run
solver = InitialValueSolver(problem, RK222(); dt=1e-3, batched_modes=true)

# raise the cap for a large-nz GPU run
solver = InitialValueSolver(problem, RK222(); dt=1e-3,
                            batched_modes_max_bytes=4 << 30)
```

### Convergence criteria

Only the nonlinear (NLBVP) path iterates:

```julia
solver.tolerance      = 1e-10   # Newton residual tolerance
solver.max_iterations = 100     # maximum Newton iterations
```

### Performance

```julia
solver.performance_stats   # total_time, total_steps, total_solves, avg_step_time
diagnose(solver)           # prints the configuration + the same stats
```

---

## Complete Example

### Time-Dependent Simulation

2D Rayleigh-Bénard convection: Fourier in `x`, Chebyshev in `z`, tau variables
for the no-slip and fixed-temperature walls.

```julia
using Tarang

Lx, Lz = 4.0, 1.0
Nx, Nz = 16, 12                 # smoke-test resolution
Rayleigh, Prandtl = 2e4, 1.0

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

# tau variables: one per boundary condition, carrying the Fourier basis
tau_p  = ScalarField(dist, "tau_p",  (), Float64)          # pressure gauge
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# first-order reduction: grad_X = ∇X + ẑ·lift(τ)
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem, nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "T(z=1) = 0")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")       # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-4)

# Initial condition: conduction profile + damped noise
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (1.0 .- z')
get_grid_data(T) .+= 1.0 .- z'
ensure_layout!(T, :c)

# Adaptive timestep
cfl = CFL(solver; initial_dt=1e-4, cadence=10, safety=0.4, max_dt=1e-3)
add_velocity!(cfl, u)

# Output — created with the solver, so run! writes and closes it
output = add_file_handler("output/rbc", solver; iter=10)
add_task!(output, T; name="T")
add_task!(output, u; name="u")

run!(solver; stop_iteration=20, cfl=cfl, log_interval=10,
     callbacks=Pair[
         10 => (s -> println("t = $(s.sim_time), dt = $(s.dt), max|T| = $(global_max(T))")),
     ])
```

The published settings are a fast 20-step smoke run. For a developed convection
run, use `Nx, Nz = 64, 32` and replace `stop_iteration=20` with `stop_time=0.5`;
that takes about 21,800 steps (a few minutes on one core) and ends with
`max|T| = 1.0`, `max|u_z| ≈ 56.5`, and the walls held to ~1e-14.

!!! warning "Choose `max_dt` conservatively"
    While the fluid is still at rest the CFL estimate is unbounded, so the controller
    jumps straight to `max_dt` on its first update — `max_dt` *is* the timestep for
    the early transient. At `max_dt=1e-2` this problem goes unstable and every field
    becomes `NaN` (silently — nothing throws). `1e-3` is stable; the CFL then pulls
    `dt` down to ~2e-5 once convection sets in.

!!! warning "This example is serial-only"
    Under MPI the Chebyshev axis must come **first**, and a Chebyshev derivative
    cannot appear in the explicit RHS — which the `-u⋅∇(u)` and `-u⋅∇(T)` advection
    terms above require. Distributed runs on a Chebyshev-Fourier domain need every
    explicit derivative to be along a Fourier axis. See
    [Parallelism](../pages/parallelism.md).

---

## See Also

- [Problems](problems.md): Problem definition
- [Timesteppers](timesteppers.md): Time integration schemes
- [Analysis](analysis.md): CFL conditions and diagnostics
- [Tutorial: IVP](../tutorials/ivp_2d_rbc.md): Complete example
