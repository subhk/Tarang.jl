# GPU Computing

GPU acceleration in Tarang.jl (experimental).

## Status

GPU support in Tarang.jl is experimental. Current capabilities:

- CUDA support via CUDA.jl
- Basic field operations
- FFT transforms via cuFFT

## Setup

### NVIDIA GPUs (CUDA)

```julia
using Pkg
Pkg.add("CUDA")

using CUDA
CUDA.versioninfo()
```

### AMD GPUs (ROCm)

```julia
Pkg.add("AMDGPU")

using AMDGPU
AMDGPU.versioninfo()
```

### Apple Silicon (Metal)

```julia
Pkg.add("Metal")

using Metal
Metal.versioninfo()
```

## Basic Usage

### Creating GPU Fields

```julia
# Specify device in distributor
dist = Distributor(coords; mesh=(1,), dtype=Float64, device="cuda")

# Fields automatically use GPU arrays
field = ScalarField(dist, "T", bases, Float64)
```

### Manual Device Control

```julia
using CUDA

# Move to GPU
field_gpu = CuArray(field.data_g)

# Compute on GPU
result_gpu = some_kernel(field_gpu)

# Move back
field.data_g .= Array(result_gpu)
```

## FFT on GPU

```julia
using CUDA
using CUDA.CUFFT

# Plan
plan = CUFFT.plan_fft!(field_gpu)

# Transform
plan * field_gpu
```

## Limitations

### Current Limitations

1. **MPI + GPU**: Multi-GPU support limited
2. **Memory**: GPU memory smaller than CPU
3. **Transfers**: CPU-GPU copies can bottleneck
4. **Not all operations**: Some features CPU-only

### Best Practices

- Keep data on GPU
- Minimize CPU-GPU transfers
- Use GPU-native operations
- Profile to find bottlenecks

## Performance

### When GPUs Help

- Large grids (N > 256 per dimension)
- Compute-bound problems
- Many FFTs

### When GPUs Don't Help

- Small problems
- Memory-bound
- Heavy MPI communication

## Example

```julia
using Tarang
using CUDA

# Check GPU
if CUDA.functional()
    device = "cuda"
else
    device = "cpu"
end

# Setup
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(1,), dtype=Float64, device=device)

# Create fields (automatically on GPU if available)
T = ScalarField(dist, "T", bases, Float64)

# Operations work the same way
Tarang.ensure_layout!(T, :g)
T.data_g .= rand(size(T.data_g)...)
Tarang.ensure_layout!(T, :c)
```

## Multi-GPU

### Single Node, Multiple GPUs

```julia
# Each MPI rank uses one GPU
gpu_id = MPI.Comm_rank(MPI.COMM_WORLD) % CUDA.ndevices()
CUDA.device!(gpu_id)
```

### Multi-Node

Requires GPU-aware MPI:

```bash
# Build MPI.jl with GPU support
export JULIA_MPI_BINARY="system"
export JULIA_CUDA_USE_BINARYBUILDER="false"
```

## Future Work

Planned improvements:

- Full multi-GPU MPI support
- Optimized kernels
- AMD and Apple Silicon parity
- Automatic CPU/GPU selection

## See Also

- [Optimization](optimization.md): General performance
- [Parallelism](parallelism.md): MPI parallelism
- [CUDA.jl Documentation](https://cuda.juliagpu.org/)
