# Parallelism

Tarang.jl uses MPI for distributed-memory parallelism.

## MPI Basics

### Initialization

```julia
using MPI
using Tarang

# Always initialize MPI first
MPI.Init()

# Get process info
rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

# Your simulation code here
# ...

# Always finalize MPI
MPI.Finalize()
```

### Running with MPI

```bash
# Run with 4 processes
mpiexec -n 4 julia --project simulation.jl

# With thread control
export OMP_NUM_THREADS=1
mpiexec -n 4 julia --project simulation.jl
```

## Process Mesh

The distributor organizes MPI processes into a mesh:

```julia
# 2D domain with 4 processes
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(2, 2))

# 3D domain with 8 processes
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; mesh=(2, 2, 2))
```

### Mesh Guidelines

| Processes | 2D Mesh | 3D Mesh |
|-----------|---------|---------|
| 4 | (2, 2) | - |
| 8 | (4, 2) or (2, 4) | (2, 2, 2) |
| 16 | (4, 4) | (4, 2, 2) |
| 64 | (8, 8) | (4, 4, 4) |

**Tips:**
- Match mesh to domain aspect ratio
- Product of mesh = number of processes
- Balance communication vs. computation

## Domain Decomposition

### Pencil Decomposition

Tarang uses pencil (slab) decomposition for efficient parallel FFTs:

```
2D domain decomposed across 4 processes:

Full Domain:        Decomposed:
┌─────────────┐     ┌──────┬──────┐
│             │     │ P0   │ P1   │
│             │     ├──────┼──────┤
│             │     │ P2   │ P3   │
└─────────────┘     └──────┴──────┘
```

### Local vs Global

```julia
# Local data on this process
local_data = field.data_g
local_size = size(local_data)

# Global size
global_size = (basis1.size, basis2.size)

# Local index range (internal function)
start_idx = Tarang.local_indices(dist, axis, global_size[axis])
```

## Communication Patterns

### Automatic Communication

Communication happens automatically during:

1. **Layout transforms**: Grid ↔ Spectral
2. **Global reductions**: sum, max, min
3. **File I/O**: Gather to root or parallel write

### Manual Communication

```julia
# Global reduction using MPI directly
local_max = maximum(field.data_g)
global_max = MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD)

# Global sum
local_sum = sum(field.data_g)
global_sum = MPI.Allreduce(local_sum, MPI.SUM, MPI.COMM_WORLD)
```

## Output Strategies

### Gather Mode

All data collected to rank 0 for writing:

```julia
handler = add_file_handler(path, dist, fields; parallel="gather")
```

**Pros**: Simple, standard file format
**Cons**: Memory limited by rank 0, serial I/O

### Virtual Mode

Each process writes its own file:

```julia
handler = add_file_handler(path, dist, fields; parallel="virtual")
```

**Pros**: Scalable, parallel I/O
**Cons**: Post-processing needed to merge

## Performance Optimization

### Load Balance

Ensure even distribution:

```julia
# Good: Global size divisible by mesh
N = 256  # 256 / 4 = 64 per process
mesh = (4,)

# Bad: Uneven distribution
N = 250  # Processes get different amounts
```

### Communication Minimization

```julia
# Minimize layout transforms
Tarang.ensure_layout!(field, :g)
# Do all grid-space operations
# Then transform once
Tarang.ensure_layout!(field, :c)
```

### FFTW Wisdom

```julia
# Set FFTW planning
ENV["FFTW_PLANNING_RIGOR"] = "FFTW_PATIENT"

# Saves optimal FFT plans
# Costs startup time, saves runtime
```

## Troubleshooting

### Common Issues

**Deadlock**: All processes must call MPI functions together

```julia
# Bad - only rank 0 calls
if rank == 0
    MPI.Barrier(MPI.COMM_WORLD)
end

# Good - all ranks call
MPI.Barrier(MPI.COMM_WORLD)
if rank == 0
    # Only rank 0 does work
end
```

**Memory errors**: Each process memory limited

```julia
# Estimate memory per process
memory = total_points * bytes_per_point / num_processes
```

### Debugging

```julia
# Print from specific rank
if rank == 0
    println("Debug info...")
end

# Print from all ranks (with ordering)
for r in 0:size-1
    if rank == r
        println("Rank $r: ...")
    end
    MPI.Barrier(MPI.COMM_WORLD)
end
```

## See Also

- [Coordinates](coordinates.md): Distributor setup
- [Domains](domains.md): Domain decomposition
- [Running with MPI](../getting_started/running_with_mpi.md): Getting started
