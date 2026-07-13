# Performance Optimization

Guide to optimizing Tarang.jl simulations.

## Profiling

### Julia Profiler

```julia
using Profile

step!(solver)      # warm up: the first step pays compilation

Profile.clear()
@profile begin
    for i in 1:50
        step!(solver)
    end
end

Profile.print(maxdepth=6)
```

### Timing

```julia
step!(solver)                                    # warm up
elapsed = @elapsed for i in 1:50
    step!(solver)
end
println("time per step: $(elapsed/50 * 1000) ms")
```

`@btime` from BenchmarkTools works too, but BenchmarkTools is not a dependency of
Tarang — add it to your own environment first.

### Counting Spectral Transforms

Spectral transforms dominate the cost of a step, and a *redundant* one is invisible to an
allocation profiler: `mul!`/`ldiv!` are in-place, so a wasted FFT allocates nothing.
Tarang ships an opt-in counter for exactly this. The three functions are not exported, so
they need the `Tarang.` prefix:

```julia
Tarang.enable_transform_counts!(true)   # off by default (one Ref load when off)
Tarang.reset_transform_counts!()
step!(solver)
Tarang.transform_counts()               # (forward = 18, backward = 27)
Tarang.enable_transform_counts!(false)
```

The numbers above are for one RK222 step of a 2D scalar + vector system. What matters is
that the count is *stable*: if it grows after a change to your equations or to Tarang, you
have added a transform.

## Memory Optimization

### Minimize Allocations

Broadcast assignment (`.=` with a dotted right-hand side) is fused and allocates nothing.
The trap is an **undotted** operator, which materializes a temporary array:

```julia
# Bad: `+` builds a whole temporary array
get_grid_data(field) .= get_grid_data(field) + get_grid_data(other)

# Good: fully fused, no temporary
get_grid_data(field) .+= get_grid_data(other)
```

Measured inside a function on a 16×16 field: the undotted form allocates 2128 bytes per
call, the fused form 0. `a .= a .+ b` is *also* fused and also costs 0 — it is the missing
dot, not the `.=`, that allocates. Measure allocations inside a function, not at global
scope: untyped globals box on their own and swamp the number you are looking for.

### Pre-allocate Work Arrays

```julia
work = similar(get_grid_data(field))     # once, at setup

function loop!(field, other, work, dt, nsteps)
    ensure_layout!(field, :g)
    ensure_layout!(other, :g)
    fd = get_grid_data(field)
    od = get_grid_data(other)
    for step in 1:nsteps
        work .= od
        fd .+= dt .* work
    end
end
```

Hoisting `get_grid_data` out of the loop and reusing `work` makes the loop body allocation
free (measured: 0 bytes for 100 iterations).

### Minimize Transforms

Every `ensure_layout!` that actually changes layout is an FFT. Batch all grid-space work
between one pair of transforms:

```julia
# Bad: a transform for every operation
ensure_layout!(field, :g)
operation1(field)
ensure_layout!(field, :c)
ensure_layout!(field, :g)
operation2(field)
ensure_layout!(field, :c)

# Good: batch operations
ensure_layout!(field, :g)
operation1(field)
operation2(field)
ensure_layout!(field, :c)
```

Counted with `Tarang.transform_counts()`, both starting from grid layout: the first form
costs `(forward = 2, backward = 1)` — three transforms — the second `(forward = 1,
backward = 0)`.

## FFT Optimization

### Grid Sizes

FFT cost is very sensitive to the factorization of `N`. Measured here for a 2D real FFT
(`FFTW.plan_rfft`, `FFTW.MEASURE`, single thread):

| N | time per 2D rfft |
|---|---|
| 256 (2⁸) | 0.11 ms |
| 250 (2·5³) | 0.14 ms |
| 251 (prime) | 1.09 ms |

Prefer powers of two; small-prime factorizations (2, 3, 5) cost a little more but are
perfectly acceptable; a large prime factor is a factor-of-ten cliff.

### FFTW Planning

Planning rigor is **not** user-tunable at present: every FFTW plan in Tarang is created
with `flags=FFTW.MEASURE` (`src/core/transforms/transform_planning.jl`). The
`TARANG_FFTW_RIGOR` environment variable sets a config value that nothing reads, and there
is no `FFTW_PLANNING_RIGOR` knob at all. Do not expect to change planning rigor from
outside the code.

What *is* tunable is the FFTW thread count — see [Threading](#Threading) below.

### FFTW Wisdom

FFTW wisdom is global to the FFTW library, so plans Tarang builds later reuse it. Import it
**before** you create your first `Domain` (that is when the plans are built):

```julia
using FFTW

FFTW.import_wisdom("fftw_wisdom.txt")    # at startup, if the file exists
# ... build the domain, run the simulation ...
FFTW.export_wisdom("fftw_wisdom.txt")    # after the run
```

Tarang does not load or save wisdom for you; the `WISDOM_FILE` config key is currently
inert.

## MPI Optimization

### Process Mesh

The mesh rank must be **at most one less than the domain dimension**: PencilFFTs needs at
least one axis to stay local for the FFT. A 2D domain therefore takes a 1D mesh; a 3D domain
takes a 2D (pencil) mesh, or a 1D (slab) mesh if you prefer.

```julia
# 2D domain: 1D mesh                       (np=4: local grid 32×8 of a 32×32 domain)
dist = Distributor(coords; mesh=(4,))

# 3D domain: 2D mesh — square              (np=4: local grid 16×8×8 of 16³)
dist = Distributor(coords; mesh=(2, 2))

# 3D domain: 2D mesh — anisotropic         (np=4: local grid 16×4×16)
dist = Distributor(coords; mesh=(4, 1))

# 3D domain: 1D slab mesh — also valid     (np=4: local grid 16×16×4)
dist = Distributor(coords; mesh=(4,))
```

A mesh whose rank *equals* the domain dimension decomposes every axis and leaves nothing
local to transform. On a 2D domain, `mesh=(2, 2)` at np=4 is a hard error:

> `PencilFFT plan creation failed with 4 MPI processes. Local FFTW fallback would produce
> incorrect results.`

Unless you have a specific reason, **omit `mesh=`**: the auto mesh is `(nprocs,)` for a 2D
domain and a balanced 2D mesh for a 3D domain (at np=4: `(2, 2)`).

### Load Balance

PencilArrays spreads the remainder across ranks, so a non-divisible `N` is only mildly
uneven — it is not a correctness problem:

```julia
N = 256   # np=4: per-rank extents 64, 64, 64, 64
N = 250   # np=4: per-rank extents 62, 63, 62, 63
```

`N = 250` also costs a little more in the FFT than `N = 256` (see [Grid Sizes](#Grid-Sizes)
above), but neither the one-plane imbalance nor the factorization is worth worrying about.
What actually hurts is a large prime factor, such as `N = 251`.

### Communication

Transposes are the dominant communication cost. Keep synchronization points out of the
inner loop: reduce over `parent(get_grid_data(field))` and issue a single explicit
`MPI.Allreduce` (or use `global_sum` / `global_max` / `integrate`, which already do that),
and gate output and progress reporting on a cadence rather than doing it every step.

## Threading

### FFTW Threads

Tarang sets the FFTW thread count at startup: all Julia threads in a serial run, and one
thread per rank under MPI (ranks already provide the parallelism). Override with an
environment variable:

```bash
export TARANG_FFTW_THREADS=4
```

BLAS threads are separate and are worth pinning to 1 under MPI:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

### Julia Threads

```julia
Threads.nthreads()          # threads available (set with `julia -t N`)

Threads.@threads for i in 1:N
    # ...
end
```

## Resolution Guidelines

### Minimum Resolution

| Feature | Modes Needed |
|---------|-------------|
| Smooth profile | ~10 |
| Moderate gradient | ~30 |
| Sharp gradient | ~100 |
| Boundary layer | ~50-100 |

### Check Convergence

Run the same problem at two resolutions and compare. Spectral convergence is exponential:
once the solution is resolved, refining changes nothing.

```julia
function heat_max(N, nsteps; nu=0.1, dt=1e-3)
    coords = CartesianCoordinates("x", "y")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xb     = RealFourier(coords["x"]; size=N, bounds=(0.0, 2pi))
    yb     = RealFourier(coords["y"]; size=N, bounds=(0.0, 2pi))
    domain = Domain(dist, (xb, yb))

    v = ScalarField(domain, "v")
    problem = IVP([v])
    add_parameters!(problem, nu=nu)
    add_equation!(problem, "∂t(v) - nu*lap(v) = 0")
    set!(v, (x, y) -> sin(x) * cos(y))

    solver = InitialValueSolver(problem, RK222(); dt=dt)
    run!(solver; stop_iteration=nsteps, progress=false)
    ensure_layout!(v, :g)
    return maximum(abs, get_grid_data(v))
end

heat_max(16, 100)    # 0.9801986733
heat_max(32, 100)    # 0.9801986733  (difference 1.6e-15 — already resolved at N=16)
```

## Timestepping

### CFL Optimization

`CFL` takes the **solver**, not the problem, and velocities are added as `VectorField`s.
Pass the controller to `run!`:

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

cfl = CFL(solver; safety=0.4, max_change=1.2, cadence=5, max_dt=0.01)
add_velocity!(cfl, u)

run!(solver; stop_iteration=1000, cfl=cfl)
```

Defaults: `initial_dt=0.01`, `cadence=1`, `safety=0.4`, `threshold=0.1`, `max_change=2.0`,
`min_change=0.5`, `max_dt=Inf`.

- `safety` — lower is conservative, higher is aggressive (0.4 is the default).
- `max_change` / `min_change` — bound the *ratio* between successive timesteps. Tightening
  `max_change` to ~1.2 keeps `dt` smooth.
- `threshold` — sticky-dt hysteresis: a new `dt` is only committed when it differs from the
  current one by more than this relative amount. This matters for implicit solvers, whose
  LHS factorization is rebuilt whenever `dt` changes.
- `cadence` — recompute the timestep every `cadence` calls rather than every step.

`CFL` is a mutable struct, so the knobs can also be changed mid-run (`cfl.max_change = 1.2`).
The current step is `cfl.current_dt` (there is no `min_dt` field).

### Implicit Methods

For stiff problems, IMEX methods allow larger timesteps:

```julia
# Explicit-ish (small dt for diffusion)
solver = InitialValueSolver(problem, RK222(); dt=1e-5)

# IMEX multistep (diffusion implicit, larger dt)
solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)
```

## Benchmarking

### Standard Benchmark

```julia
function benchmark_solver(solver, nsteps)
    step!(solver)                    # warm up (compilation)
    start = time()
    for i in 1:nsteps
        step!(solver)
    end
    elapsed = time() - start

    ngrid = prod(b.meta.size for b in solver.problem.domain.bases)
    println("Time per step: $(elapsed/nsteps * 1000) ms")
    println("Grid points/sec: $(ngrid * nsteps / elapsed)")
end
```

`ngrid` is the *global* grid size (`b.meta.size` is the global size of each basis). For the
per-rank count use `length(get_grid_data(solver.state[1]))`.

### Scaling Test

Strong scaling holds the problem fixed and varies the process count; weak scaling holds the
work per process fixed. Both are driven from the shell, one `mpiexec` per point:

```bash
for np in 1 2 4 8 16; do
    mpiexec -n $np julia --project=. run_benchmark.jl
done
```

## Common Bottlenecks

1. **FFT transforms**: use FFT-friendly sizes, batch layout changes, watch
   `Tarang.transform_counts()`
2. **MPI communication**: valid mesh, minimize syncs
3. **Memory allocation**: pre-allocate, use in-place ops
4. **I/O**: write less frequently, use parallel I/O

## See Also

- [Parallelism](parallelism.md): MPI optimization
- [Timesteppers](timesteppers.md): Choosing methods
- [Domains](domains.md): Resolution guidelines
