# Running with MPI

Tarang.jl is designed for efficient parallel computing using MPI (Message Passing Interface). This guide covers everything you need to know about running Tarang.jl simulations in parallel.

## Basic MPI Execution

### Running a Script

Execute a Tarang.jl script with MPI using `mpiexec`:

```bash
mpiexec -n 4 julia your_script.jl
```

The `-n 4` flag specifies 4 MPI processes. This must match your process mesh configuration in the script.

### Process Mesh Configuration

The process mesh determines how MPI processes are arranged:

```julia
# 2D process mesh (4 processes: 2×2)
dist = Distributor(coords, mesh=(2, 2))

# 1D process mesh (4 processes: 4×1)
dist = Distributor(coords, mesh=(4, 1))

# 3D process mesh (8 processes: 2×2×2)
dist = Distributor(coords, mesh=(2, 2, 2))
```

!!! tip "Matching Process Count"
    Ensure `product(mesh) == number of MPI processes`:
    ```julia
    mesh=(2, 2)  # requires: mpiexec -n 4
    mesh=(4, 2)  # requires: mpiexec -n 8
    mesh=(4, 4)  # requires: mpiexec -n 16
    ```

## Process Mesh Strategies

### 2D Problems

For 2D problems, you can parallelize in both directions:

```julia
# Balanced 2D decomposition (recommended)
mesh=(4, 4)  # 16 processes, good for most 2D problems

# Horizontal decomposition (for thin domains)
mesh=(8, 2)  # 16 processes, more processes in x-direction

# Vertical decomposition (for tall domains)
mesh=(2, 8)  # 16 processes, more processes in z-direction
```

**Rule of thumb**: Match the mesh aspect ratio to your domain aspect ratio.

### 3D Problems

For 3D problems, Tarang.jl uses pencil decomposition:

```julia
# Cubic mesh (for isotropic domains)
mesh=(4, 4, 4)  # 64 processes

# Anisotropic mesh (for stratified flows)
mesh=(8, 8, 2)  # 128 processes, fewer in vertical direction
```

**Pencil decomposition** means data is distributed in two dimensions while remaining contiguous in the third. This enables efficient parallel FFTs.

## Environment Variables

### Thread Control

Set OpenMP thread count to avoid oversubscription:

```bash
export OMP_NUM_THREADS=1
mpiexec -n 8 julia script.jl
```

!!! warning "Performance Impact"
    Not setting `OMP_NUM_THREADS=1` can cause significant performance degradation. Tarang.jl will warn you if this is not set correctly.

### Julia Threads

Julia's multithreading can work alongside MPI:

```bash
export JULIA_NUM_THREADS=4
export OMP_NUM_THREADS=1
mpiexec -n 4 julia script.jl
```

This gives you 4 MPI processes × 4 Julia threads = 16 parallel tasks.

### Other Useful Variables

```bash
# FFTW optimization
export FFTW_PLANNING_RIGOR=FFTW_MEASURE

# Tarang logging
export TARANG_LOG_LEVEL=DEBUG

# MPI debugging
export OMPI_MCA_mpi_show_mca_params=1
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

1. **Use power-of-2 process counts** when possible (4, 8, 16, 32, ...)
2. **Match mesh to domain**: Aspect ratio of mesh should match domain
3. **Consider memory**: Each process needs enough RAM for its subdomain

### Checking Load Distribution

Add diagnostics to your script:

```julia
rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

local_size = size(T.data)  # Size of local data on this rank

println("Rank $rank: Local array size = $local_size")
```

## Communication Patterns

### PencilArrays Transposes

Tarang.jl uses PencilArrays for efficient data distribution. Key operations:

- **FFTs**: May require transposes between different pencil orientations
- **Derivatives**: Computed in spectral space (minimal communication)
- **Nonlinear terms**: Evaluated in grid space (may require transforms)

### Minimizing Communication

To reduce communication overhead:

1. **Group operations**: Batch multiple operations before synchronizing
2. **Use larger process counts**: More processes = smaller messages
3. **Optimize process mesh**: Align with dominant communication direction

## Performance Monitoring

### MPI Profiling

Use MPI profiling tools to identify bottlenecks:

```bash
# With mpiP
mpiexec -n 8 julia --project=. script.jl

# With Intel MPI
export I_MPI_STATS=20
mpiexec -n 8 julia script.jl
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
using Printf

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
julia script.jl  # No mpiexec
```

Modify your script to use 1 process:

```julia
dist = Distributor(coords, mesh=(1, 1))
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

Debug specific MPI ranks:

```julia
rank = MPI.Comm_rank(MPI.COMM_WORLD)

if rank == 0
    println("Debug: Rank 0 data = ", data)
end

# Or debug all ranks
println("Rank $rank: data = $data")
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

**Solution**: Check domain decomposition and adjust mesh:

```julia
# Before (imbalanced for tall domain)
mesh=(8, 2)

# After (better for tall domain)
mesh=(4, 4)
```

### Memory Issues

**Symptom**: Out of memory errors on some ranks

**Solution**: Reduce resolution or increase process count:

```julia
# Reduce resolution
x_basis = RealFourier(coords["x"], size=512, ...)  # was 1024

# Or use more processes to distribute memory
mesh=(8, 8)  # was (4, 4)
```

### Wrong Number of Processes

**Symptom**: `ERROR: Process count mismatch`

**Solution**: Match `mpiexec -n` to `mesh` product:

```julia
mesh=(4, 2)  # requires mpiexec -n 8
```

## Performance Tips

### Optimal Process Count

1. **Start with square meshes**: `mesh=(4,4)`, `(8,8)`, etc.
2. **Profile with different counts**: Try 4, 8, 16, 32 processes
3. **Check scaling**: Plot speedup vs. process count
4. **Consider communication**: More processes = more communication overhead

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

Keep local problem size constant, increase total size:

```julia
# 4 processes: 128×64 per process
mesh=(2, 2); x_size=256; z_size=128

# 16 processes: 128×64 per process
mesh=(4, 4); x_size=512; z_size=256

# 64 processes: 128×64 per process
mesh=(8, 8); x_size=1024; z_size=512
```

Ideal weak scaling: time remains constant.

### Strong Scaling

Keep total problem size constant, increase processes:

```julia
# All use: x_size=1024, z_size=512
mesh=(2, 2)  # 4 processes
mesh=(4, 4)  # 16 processes
mesh=(8, 8)  # 64 processes
```

Ideal strong scaling: time decreases linearly with processes.

## Next Steps

- [First Steps](first_steps.md): Basic Tarang.jl workflow
- [Tutorials](../tutorials/overview.md): Example simulations with MPI
- [Configuration](../pages/configuration.md): Advanced MPI configuration options
- [Parallelism Guide](../pages/parallelism.md): Deep dive into parallel algorithms
