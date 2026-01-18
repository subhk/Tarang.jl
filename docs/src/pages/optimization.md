# Performance Optimization

Guide to optimizing Tarang.jl simulations.

## Profiling

### Julia Profiler

```julia
using Profile

@profile begin
    for i in 1:100
        step!(solver)
    end
end

Profile.print()
```

### Timing

```julia
using BenchmarkTools

@btime step!($solver)
```

## Memory Optimization

### Minimize Allocations

```julia
# Bad: Creates new arrays
field.data_g = field.data_g .+ other.data_g

# Good: In-place operation
field.data_g .+= other.data_g
```

### Pre-allocate Work Arrays

```julia
# Once at setup
work = similar(field.data_g)

# Reuse in loop
for step in 1:nsteps
    work .= compute_rhs(field)
    field.data_g .+= dt .* work
end
```

### Minimize Transforms

```julia
# Bad: Transform for each operation
Tarang.ensure_layout!(field, :g)
operation1(field)
Tarang.ensure_layout!(field, :c)
Tarang.ensure_layout!(field, :g)
operation2(field)

# Good: Batch operations
Tarang.ensure_layout!(field, :g)
operation1(field)
operation2(field)
Tarang.ensure_layout!(field, :c)
```

## FFT Optimization

### Power-of-2 Sizes

```julia
# Fast
N = 256  # 2^8

# Slower
N = 250

# Much slower
N = 251  # Prime
```

### FFTW Planning

```julia
# Set planning rigor (before first transform)
ENV["FFTW_PLANNING_RIGOR"] = "FFTW_PATIENT"

# Options:
# FFTW_ESTIMATE - fast planning, may be suboptimal
# FFTW_MEASURE - balanced (default)
# FFTW_PATIENT - thorough search
# FFTW_EXHAUSTIVE - very thorough (slow startup)
```

### FFTW Wisdom

Save and reuse FFT plans:

```julia
using FFTW

# Save wisdom after first run
FFTW.export_wisdom("fftw_wisdom.txt")

# Load on subsequent runs
FFTW.import_wisdom("fftw_wisdom.txt")
```

## MPI Optimization

### Process Mesh

Match mesh to domain:

```julia
# Square domain
dist = Distributor(coords; mesh=(4, 4))

# Wide domain (Lx > Lz)
dist = Distributor(coords; mesh=(8, 2))
```

### Load Balance

```julia
# Good: Even division
N = 256  # 256 / 4 = 64 per process

# Bad: Uneven
N = 250  # Some processes get more
```

### Communication

```julia
# Minimize synchronization points
# Batch MPI operations
# Use non-blocking where possible
```

## Threading

### Set Threads

```bash
# For FFTW (usually 1 is best with MPI)
export OMP_NUM_THREADS=1

# Or in Julia
BLAS.set_num_threads(1)
```

### Julia Threads

```julia
# Check available
Threads.nthreads()

# Threaded loops
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

```julia
# Compare solutions at different N
# Should converge exponentially for spectral methods
```

## Timestepping

### CFL Optimization

```julia
# Tune safety factor
cfl = CFL(problem; safety=0.4)  # Conservative
cfl = CFL(problem; safety=0.6)  # Aggressive

# Limit timestep changes
cfl.max_change = 1.2  # Smooth
```

### Implicit Methods

For stiff problems, IMEX methods allow larger timesteps:

```julia
# Explicit (requires small dt for diffusion)
solver = InitialValueSolver(problem, RK222(); dt=1e-5)

# IMEX (diffusion implicit, larger dt)
solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)
```

## Benchmarking

### Standard Benchmark

```julia
function benchmark_solver(solver, nsteps)
    start = time()
    for i in 1:nsteps
        step!(solver)
    end
    elapsed = time() - start

    println("Time per step: $(elapsed/nsteps * 1000) ms")
    println("Grid points/sec: $(prod(solver.problem.shape) * nsteps / elapsed)")
end
```

### Scaling Test

```julia
# Strong scaling: fixed problem, vary processes
# Weak scaling: fixed work per process

for np in [1, 2, 4, 8, 16]
    # Run with mpiexec -n $np
    # Record time
end
```

## Common Bottlenecks

1. **FFT transforms**: Use optimal sizes, FFTW wisdom
2. **MPI communication**: Balance mesh, minimize syncs
3. **Memory allocation**: Pre-allocate, use in-place ops
4. **I/O**: Write less frequently, use parallel I/O

## See Also

- [Parallelism](parallelism.md): MPI optimization
- [Timesteppers](timesteppers.md): Choosing methods
- [Domains](domains.md): Resolution guidelines
