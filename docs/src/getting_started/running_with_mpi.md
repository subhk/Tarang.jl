# [Running with MPI](@id running-with-mpi)

Tarang.jl is designed for efficient parallel computing using MPI (Message Passing Interface). This guide covers everything you need to know about running Tarang.jl simulations in parallel.

## Basic MPI Execution

### Running a Script

Execute a Tarang.jl script with MPI using `mpiexec`:

```bash
mpiexec -n 4 julia --project=. your_script.jl
```

The `-n 4` flag specifies 4 MPI processes. The process count must match your process mesh configuration in the script.

### Process Mesh Configuration

The process mesh determines how MPI processes are arranged. The one rule that governs every choice below:

!!! warning "The mesh must have exactly one fewer dimension than the domain"
    A parallel FFT needs at least one array axis to stay **local** on each rank. PencilArrays
    decomposes the trailing `length(mesh)` axes, so a mesh whose rank equals the domain
    dimension decomposes *every* axis and leaves nothing for the transform. A 2-D domain takes
    a **1-D** mesh; a 3-D domain takes a **2-D** mesh.

```julia
# 2D domain, 4 processes: slab decomposition (one axis split, one axis local)
dist = Distributor(coords; mesh=(4,))

# 3D domain, 4 processes: pencil decomposition (two axes split, one axis local)
dist = Distributor(coords; mesh=(2, 2))

# Best default: omit `mesh` entirely and let Tarang pick
dist = Distributor(coords)     # 2D -> (nprocs,) ;  3D -> a 2D pencil mesh
```

A mesh containing a unit factor — `(4, 1)` or `(1, 4)` on a 2-D domain — is silently normalized
to the equivalent slab mesh `(4,)`, so those forms work too. A genuine 2-D mesh on a 2-D domain
does **not**:

```julia
dist = Distributor(coords; mesh=(2, 2))   # 2D domain, 4 ranks -> HARD ERROR
# ERROR: PencilFFT plan creation failed with 4 MPI processes.
```

!!! tip "Matching Process Count"
    `prod(mesh)` must equal the number of MPI processes:
    ```julia
    mesh=(4,)     # 2D domain, requires: mpiexec -n 4
    mesh=(16,)    # 2D domain, requires: mpiexec -n 16
    mesh=(4, 2)   # 3D domain, requires: mpiexec -n 8
    mesh=(4, 4)   # 3D domain, requires: mpiexec -n 16
    ```

## Process Mesh Strategies

### 2D Problems

A 2-D domain is decomposed as a **slab**: the last axis is split across all ranks, the first
axis stays local. There is only one shape available, so the only decision is the process count:

```julia
mesh=(8,)     # 8 processes, last axis split 8 ways
mesh=(16,)    # 16 processes
```

Each rank owns `N_last / nprocs` planes of the decomposed axis, so pick a process count that
divides the decomposed axis size evenly. Uneven splits work — PencilArrays spreads the remainder
so that some ranks get one extra plane (a 10-point axis over 4 ranks gives local sizes
`2, 3, 2, 3`) — but they cost you load balance.

### 3D Problems

For 3D problems, Tarang.jl uses **pencil decomposition**: two axes are distributed while the
third remains contiguous (local) on each rank, which is what makes the parallel FFT efficient.

```julia
# Square pencil mesh (for isotropic domains)
mesh=(8, 8)     # 64 processes

# Anisotropic mesh
mesh=(16, 4)    # 64 processes
```

**Rule of thumb**: match the mesh aspect ratio to the sizes of the two decomposed (trailing)
axes, so each rank's block is roughly cubic.

## Which Basis Layouts Work Under MPI

### Pure Fourier — any dimension ≥ 2D

```julia
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
yb = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))
dom = Domain(dist, (xb, yb))          # decomposed on y
```

### Mixed Chebyshev–Fourier — supported, but Chebyshev must come FIRST

PencilArrays decomposes the *trailing* axes, and a Chebyshev axis **cannot** be decomposed: its
DCT is a local FFTW r2r that needs the entire axis on every rank. So under MPI the Chebyshev
axis must be listed **before** the Fourier axes — the reverse of the conventional
`(x Fourier, z Chebyshev)` channel layout used in serial.

```julia
coords = CartesianCoordinates("z", "x")            # note the coordinate order too
dist   = Distributor(coords; dtype=Float64, device=CPU())
zb = ChebyshevT(coords["z"];  size=12, bounds=(0.0, 1.0))
xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
domain = Domain(dist, (zb, xb))                    # Chebyshev FIRST
```

Getting this wrong is a loud error, not a silent wrong answer:

> `MPI mixed Fourier-Chebyshev: the decomposed (trailing) axis/axes [2] are non-Fourier (e.g.
> Chebyshev), but PencilArrays decomposes the LAST 1 dimension(s) and a Chebyshev axis cannot be
> decomposed ... Reorder your bases so the Chebyshev axis comes BEFORE the Fourier axes`

Because the Chebyshev axis must stay local, a 2-D Chebyshev–Fourier domain has exactly one
decomposable axis and therefore only ever supports a 1-D mesh.

!!! warning "`ChannelDomain` / `ChannelDomain3D` are serial-only"
    The `ChannelDomain` helpers build `(x Fourier, z Chebyshev)` — Chebyshev last — and so hit
    the error above under MPI. For a distributed channel, build the `Domain` yourself with the
    Chebyshev axis first.

A complete distributed Chebyshev–Fourier diffusion problem (this runs identically at `-n 1`,
`-n 2` and `-n 4`):

```julia
b    = ScalarField(domain, "b")
tau1 = ScalarField(dist, "tau1", (), Float64)   # under MPI, tau fields carry NO bases
tau2 = ScalarField(dist, "tau2", (), Float64)

ez, ex   = unit_vector_fields(coords, dist)     # coords are ("z","x") => ez is first
τ_lift(A) = lift(A, derivative_basis(zb, 1), -1)
grad_b    = grad(b) + ez * τ_lift(tau1)

problem = IVP([b, tau1, tau2])
add_parameters!(problem, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau2) = 0")
add_bc!(problem, "b(z=0) = 0")
add_bc!(problem, "b(z=1) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

zg, xg = local_grids(dist, zb, xb)              # per-rank LOCAL grids
ensure_layout!(b, :g)
get_grid_data(b) .= sin.(π .* zg) .* (1 .+ 0.5 .* cos.(2 .* xg'))
ensure_layout!(b, :c)

run!(solver; stop_iteration=20, progress=false)
```

Per-Fourier-mode tau fields (`ScalarField(dist, "tau", (xbasis,), Float64)`) only work with the
Fourier axis first, which MPI forbids — under MPI use bases-free `()` tau fields and accept
boundary enforcement at ~1e-6 instead of machine precision.

## What Is Not Supported Under MPI

These all raise clear errors; none of them silently produce wrong answers.

| Not supported | Message / behaviour | Do this instead |
|---|---|---|
| **1D domains** | `MPI parallelization is not supported for 1D problems. 1D FFT requires global data access.` | Run 1D problems serially. |
| **Pure-Chebyshev domains** | `MPI parallelization is not supported for pure Chebyshev domains.` | Needs at least one Fourier axis to decompose; run serially. |
| **Chebyshev axis not first** | The "Reorder your bases" error above. | Put the Chebyshev axis first. |
| **Mesh rank == domain rank** | `PencilFFT plan creation failed with N MPI processes.` | Use a mesh with one fewer dimension than the domain. |
| **`set!(field, ::Function)`** | `DimensionMismatch` (it builds the *global* meshgrid and writes it into the *local* slab). | Use `local_grids(dist, bases...)` and broadcast into `get_grid_data(field)`. |
| **Grid-space `set_scales!` / `change_scales!`** | `Grid-space resampling of a distributed field ... is not supported under MPI` | Resample in coefficient space, or set the scales before distributing. |
| **Chebyshev derivative on the explicit (RHS) side** | First step errors: `Lazy RHS: cannot differentiate along the non-Fourier axis 1 (ChebyshevT) of a DISTRIBUTED field` | Move the term to the implicit (L) side. RHS derivatives along **Fourier** axes are fine. |

The last one is worth spelling out: on a Chebyshev–Fourier domain, `-u⋅∇(u)` and `-u⋅∇(T)`
expand to include `∂z`, so the classic 2D Rayleigh–Bénard / channel setup with advection runs
in **serial only**. A nonlinearity whose derivatives are all along Fourier axes — e.g.
`-b*∂x(b)` — runs distributed and matches serial to roundoff.

## Setting Initial Conditions in Parallel

`set!(field, ::Function)` is serial-only. The distributed-safe idiom uses the rank's *local*
grid vectors:

```julia
xg, yg = local_grids(dist, xb, yb)
ensure_layout!(u, :g)
get_grid_data(u) .= sin.(xg) .* cos.(yg')     # correct at any process count
ensure_layout!(u, :c)
```

`set!(field, ::Number)` and `fill_random!(field, "g"; seed=..., scale=...)` are safe distributed.

## Environment Variables

### Thread Control

Under MPI, one rank per core is the intended model: when `MPI.Comm_size > 1` Tarang forces FFTW
to a **single thread**, regardless of `JULIA_NUM_THREADS`. Keep other libraries (BLAS, OpenMP)
from oversubscribing the same cores:

```bash
export OMP_NUM_THREADS=1
mpiexec -n 8 julia --project=. script.jl
```

To override Tarang's FFTW thread count deliberately — e.g. a few fat ranks on a many-core node —
set `TARANG_FFTW_THREADS`:

```bash
export TARANG_FFTW_THREADS=4     # 4 FFTW threads per rank
mpiexec -n 2 julia --project=. script.jl
```

### Other Useful Variables

```bash
# Tarang logging, read at `using Tarang` time
export TARANG_LOG_LEVEL=DEBUG      # TRACE, DEBUG, INFO, NOTICE, WARN, ERROR
export TARANG_LOG_FILE=tarang.log  # optional log file
```

## HPC Cluster Execution

### SLURM

Example SLURM submission script:

```bash
#!/bin/bash
#SBATCH --job-name=tarang_sim
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Load modules
module load julia/1.9
module load openmpi/4.1

# Set environment
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Calculate total tasks
NTASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Run simulation
srun -n $NTASKS julia --project=. simulation.jl
```

Submit with:
```bash
sbatch submit_tarang.sh
```

### PBS/Torque

Example PBS script:

```bash
#!/bin/bash
#PBS -N tarang_sim
#PBS -l nodes=4:ppn=32
#PBS -l walltime=24:00:00
#PBS -q batch

cd $PBS_O_WORKDIR

module load julia/1.9
module load openmpi/4.1

export OMP_NUM_THREADS=1

mpiexec julia --project=. simulation.jl
```

Submit with:
```bash
qsub submit_tarang.pbs
```

## Load Balancing

### Automatic Load Balancing

Tarang.jl automatically distributes work across MPI processes using PencilArrays. For balanced performance:

1. **Divide the decomposed axes evenly**: `nprocs` should divide the size of the trailing axis
   (2D) or of both trailing axes (3D); otherwise some ranks carry an extra plane.
2. **Use power-of-2 process counts** when possible (4, 8, 16, 32, ...) — spectral sizes usually are.
3. **Consider memory**: each process needs enough RAM for its subdomain.

### Checking Load Distribution

Add diagnostics to your script. `Tarang` does not re-export `MPI`, so import it yourself:

```julia
using Tarang, MPI

rank  = MPI.Comm_rank(MPI.COMM_WORLD)
nproc = MPI.Comm_size(MPI.COMM_WORLD)

local_size = size(parent(get_grid_data(T)))   # this rank's slab

println("Rank $rank/$nproc: local grid size = $local_size")
```

Note `parent(...)`: `get_grid_data` returns a `PencilArray` under MPI, and `parent` gets the raw
local storage. `PencilArrays.range_local(get_grid_data(T))` gives this rank's global index range.

## Communication Patterns

### PencilArrays Transposes

Tarang.jl uses PencilArrays for efficient data distribution. Key operations:

- **FFTs**: May require transposes between different pencil orientations
- **Derivatives**: Computed in spectral space (minimal communication)
- **Nonlinear terms**: Evaluated in grid space (may require transforms)

### Reductions Are Already Collective

A reduction over a whole `PencilArray` (`sum`, `maximum`, `mapreduce`, ...) performs its own
`Allreduce`: it is a **collective**, so every rank must reach it, and its result is already
global. Wrapping it in another `MPI.Allreduce` double-reduces — an `nprocs`× error that is
silent on x86 (on Apple Silicon the PencilArray reduction throws instead: *"User-defined
reduction operators are currently not supported on non-Intel architectures"*). Reduce the
**local** storage and do one explicit collective, or use Tarang's helpers:

```julia
comm = MPI.COMM_WORLD
gd   = get_grid_data(u)

# Correct: reduce the local slab, then one collective
lmax = maximum(abs, parent(gd));  gmax = MPI.Allreduce(lmax, MPI.MAX, comm)

# Correct: Tarang's helpers already do exactly that
global_max(u)     # Float64
global_sum(u)     # Float64
integrate(u)      # Float64, quadrature-weighted

# WRONG: sum(gd) is itself an Allreduce -> nprocs× too large
MPI.Allreduce(sum(gd), MPI.SUM, comm)
```

The same rule applies to `global_max` / `global_sum` / `integrate`: they are collective, so call
them on **every** rank and only guard the `println` with `if rank == 0`, never the call itself.

### Minimizing Communication

To reduce communication overhead:

1. **Group operations**: Batch multiple operations before synchronizing
2. **Keep derivatives spectral**: they are local in coefficient space
3. **Optimize process mesh**: align with the dominant communication direction

## Performance Monitoring

### MPI Profiling

Use MPI profiling tools to identify bottlenecks:

```bash
# With mpiP
mpiexec -n 8 julia --project=. script.jl

# With Intel MPI
export I_MPI_STATS=20
mpiexec -n 8 julia --project=. script.jl
```

### Tarang.jl Built-in Profiling

Enable performance logging:

```julia
using Tarang

# Setup logging with MPI awareness
Tarang.setup_tarang_logging(
    level="INFO",
    filename="performance.log",
    mpi_aware=true,
    console=true
)
```

### Timing Critical Sections

Add timing to your simulation:

```julia
using Printf, MPI

t_start = time()

# ... simulation code ...

t_end = time()
elapsed = t_end - t_start

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @printf "Total time: %.2f seconds\n" elapsed
end
```

## Debugging MPI Applications

### Running in Serial

Test your code without MPI first:

```bash
julia --project=. script.jl  # No mpiexec
```

The same script runs serially and in parallel — just omit `mesh=` and let the `Distributor`
size itself from the communicator:

```julia
dist = Distributor(coords)   # mesh = (1,) on 1 process, (nprocs,) under mpiexec
```

### MPI Debugging Tools

Use parallel debuggers:

```bash
# With TotalView
mpiexec -tv -n 4 julia script.jl

# With DDT (ARM Forge)
ddt mpiexec -n 4 julia script.jl

# With gdb (attach to rank 0)
mpiexec -n 4 xterm -e gdb julia script.jl
```

### Rank-Specific Output

Debug specific MPI ranks. Reduce the rank's **local** slab (`parent(...)`) — a reduction over
the whole `PencilArray` is collective and must be called by every rank:

```julia
using Tarang, MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)
lmax = maximum(abs, parent(get_grid_data(u)))   # this rank's slab only

if rank == 0
    println("Debug: Rank 0 max|u| = ", lmax)
end

# Or print from every rank
println("Rank $rank: max|u| = $lmax")
```

## Common Issues and Solutions

### Deadlocks

**Symptom**: Program hangs without error

**Causes**:
- Mismatched collective operations
- Ranks waiting for different communications
- Incorrect synchronization

**Solution**: Ensure all ranks participate in collective operations:

```julia
# All ranks must call collective operations
MPI.Barrier(MPI.COMM_WORLD)  # All ranks must call this
```

### Load Imbalance

**Symptom**: Some ranks finish much faster than others

**Cause**: the decomposed axis does not divide evenly by the process count, so some ranks carry
an extra plane of the trailing axis.

**Solution**: choose a process count that divides the decomposed axis size:

```julia
# 2D domain, trailing axis size 100
mesh=(8,)    # 100 = 12×8 + 4  -> four ranks carry 13 planes, four carry 12
mesh=(4,)    # 100 = 25×4      -> balanced
```

### Memory Issues

**Symptom**: Out of memory errors on some ranks

**Solution**: Reduce resolution or increase process count:

```julia
# Reduce resolution
x_basis = RealFourier(coords["x"]; size=512, bounds=(0.0, 2π))  # was 1024

# Or use more processes to distribute memory (2D domain -> 1D mesh)
mesh=(16,)   # was (4,)
```

### Wrong Number of Processes

**Symptom**:

```
ERROR: ArgumentError: Mesh size 9 does not match number of processes 4
```

**Solution**: Match `mpiexec -n` to `prod(mesh)`:

```julia
mesh=(4, 2)  # 3D domain, requires mpiexec -n 8
```

### PencilFFT Plan Creation Failed

**Symptom**:

```
ERROR: PencilFFT plan creation failed with 4 MPI processes.
```

**Cause**: the mesh has as many dimensions as the domain, so no axis is left local for the FFT.

**Solution**: drop a mesh dimension — `(2, 2)` → `(4,)` on a 2-D domain — or omit `mesh=`.

## Performance Tips

### Optimal Process Count

1. **Divide the decomposed axis evenly**: `nprocs | N_trailing` (2D), or both trailing axes (3D)
2. **Profile with different counts**: Try 4, 8, 16, 32 processes
3. **Check scaling**: Plot speedup vs. process count
4. **Consider communication**: More processes = more transpose traffic per FFT

### Node-Level Optimization

On multi-socket nodes:

```bash
# Bind processes to cores
mpiexec -n 16 --bind-to core julia script.jl

# Use one process per socket
mpiexec -n 2 --map-by socket julia script.jl
```

### Network Optimization

For InfiniBand or high-speed networks:

```bash
# Enable UCX (if available)
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx

mpiexec -n 32 julia script.jl
```

## Benchmarking

### Weak Scaling

Keep the number of points per process constant while the total grows. With a slab mesh only the
trailing axis is split, so grow both axes and let the split absorb the extra ranks:

```julia
# 4 processes:  local slab 256×32   = 8192 points
mesh=(4,);  x_size=256;  z_size=128

# 16 processes: local slab 512×16   = 8192 points
mesh=(16,); x_size=512;  z_size=256

# 64 processes: local slab 1024×8   = 8192 points
mesh=(64,); x_size=1024; z_size=512
```

Ideal weak scaling: time remains constant.

### Strong Scaling

Keep total problem size constant, increase processes:

```julia
# All use: x_size=1024, z_size=512  (2D domain -> 1D mesh)
mesh=(4,)    # 4 processes,  local 1024×128
mesh=(8,)    # 8 processes,  local 1024×64
mesh=(16,)   # 16 processes, local 1024×32
```

Ideal strong scaling: time decreases linearly with processes.

## Next Steps

- [First Steps](first_steps.md): Basic Tarang.jl workflow
- [Tutorials](../tutorials/overview.md): Example simulations with MPI
- [Configuration](../pages/configuration.md): Advanced MPI configuration options
- [Parallelism Guide](../pages/parallelism.md): Deep dive into parallel algorithms
