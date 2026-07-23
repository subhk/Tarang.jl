# GPU Computing

Tarang.jl provides comprehensive GPU acceleration through CUDA.jl, enabling significant speedups for spectral simulations on NVIDIA GPUs.

## Overview

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Requirements</p>
<p>GPU support requires an NVIDIA GPU with CUDA capability 5.0+ and the CUDA.jl package.</p>
</div>
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Automatic Dispatch** | Arrays automatically use GPU kernels when on GPU memory |
| **CUFFT Integration** | Optimized FFT plans via NVIDIA's cuFFT library |
| **Custom Kernels** | KernelAbstractions.jl for portable CPU/GPU code |
| **Memory Pools** | Efficient GPU memory management with pooling |
| **Multi-GPU** | MPI + CUDA for distributed GPU computing |
| **Mixed Precision** | Float32 support for memory-bound problems |

## Quick Start

### Basic GPU Setup

```julia
using Tarang
using CUDA

# Check GPU availability
@assert CUDA.functional() "CUDA not available"

# Create distributor with GPU architecture
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(1,), dtype=Float64, device=GPU())

# Create bases and fields (automatically on GPU)
xbasis = Fourier(coords, "x", 256)
ybasis = Fourier(coords, "y", 256)
field = ScalarField(dist, "u", (xbasis, ybasis))

# Field data is a CuArray
@assert field["g"] isa CuArray
```

### CPU vs GPU Architecture

```julia
# CPU execution (default)
dist_cpu = Distributor(coords; device=CPU())

# GPU execution
dist_gpu = Distributor(coords; device=GPU())

# Check architecture
arch = dist_gpu.architecture  # GPU()
```

## GPU Transforms

### Automatic FFT Acceleration

When fields are on GPU, transforms automatically use CUFFT:

```julia
using Tarang, CUDA

dist = Distributor(coords; device=GPU())
field = ScalarField(dist, "u", (xbasis, ybasis))

# Initialize with GPU data
field["g"] .= CUDA.rand(Float64, size(field["g"])...)

# Forward transform (uses CUFFT automatically)
forward_transform!(field)

# Backward transform
backward_transform!(field)
```

### FFT Mode Control

GPU fields always use GPU transforms, including small arrays:

```julia
# Per-field control
set_gpu_fft_mode!(field, :gpu)   # Always use GPU FFT
set_gpu_fft_mode!(field, :auto)  # Default; also always on-device
```

`:cpu` is rejected for a GPU field. Unsupported basis/layout combinations fail
with an error instead of downloading the field and running FFTW.

### Mixed Fourier-Chebyshev Transforms

GPU DCT for Chebyshev bases:

```julia
# Mixed basis domain
xbasis = Fourier(coords, "x", 256)      # FFT
zbasis = ChebyshevT(coords, "z", 64)    # DCT

dist = Distributor(coords; device=GPU())
field = ScalarField(dist, "T", (xbasis, zbasis))

# Transforms automatically select FFT or DCT per dimension
forward_transform!(field)   # FFT in x, DCT in z
backward_transform!(field)  # IFFT in x, IDCT in z
```

## GPU Memory Management

Tarang relies on CUDA.jl's built-in memory pool (following the Oceananigans.jl approach).
No custom pooling is needed — CUDA.jl handles allocation efficiently.

Configure the pool via environment variable:
```bash
export JULIA_CUDA_MEMORY_POOL=binned  # default, efficient for repeated allocations
```

### Data Transfers

```julia
# CPU → GPU
async_copy_to_gpu!(gpu_array, cpu_array)

# GPU → CPU
async_copy_to_cpu!(cpu_array, gpu_array)
```

For pinned memory (faster MPI transfers), use CUDA.jl directly:
```julia
CUDA.Mem.pin(cpu_array)  # page-lock for faster DMA transfers
```

## Custom GPU Kernels

### KernelAbstractions Integration

Write portable kernels that run on both CPU and GPU:

```julia
using Tarang, KernelAbstractions

# Define a kernel
@kernel function add_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

# Wrap as KernelOperation
add_op = KernelOperation(add_kernel!) do c, a, b
    length(c)  # ndrange
end

# Use on any architecture
arch = GPU()
a = ones(arch, Float64, 1024)
b = ones(arch, Float64, 1024)
c = zeros(arch, Float64, 1024)

add_op(arch, c, a, b)  # Runs on GPU
```

### Built-in GPU Kernels

Tarang provides optimized kernels for common operations:

```julia
using TarangCUDAExt

# Element-wise operations
gpu_add!(c, a, b)           # c = a + b
gpu_sub!(c, a, b)           # c = a - b
gpu_mul!(c, a, b)           # c = a * b
gpu_scale!(y, α, x)         # y = α * x
gpu_axpy!(y, α, x)          # y = y + α * x
gpu_linear_combination!(y, α, a, β, b)  # y = α*a + β*b

# Fused operations for timestepping
gpu_rk_stage!(u_new, u, k, dt, coeff)
gpu_axpby!(y, α, x, β)      # y = α*x + β*y

# Physics kernels
gpu_kinetic_energy_2d!(ke, ux, uy)
gpu_kinetic_energy_3d!(ke, ux, uy, uz)
gpu_viscous_damping!(field, ν, k2)
```

## Distributed GPU Computing

### MPI + CUDA

For multi-GPU simulations:

```julia
using Tarang, MPI, CUDA

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Assign GPU to MPI rank
if CUDA.ndevices() >= MPI.Comm_size(comm)
    CUDA.device!(rank)
end

# Create distributed GPU setup
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; mesh=(2, 2), device=GPU())

# Each rank has its own GPU memory
field = ScalarField(dist, "u", bases)
```

### CUDA-Aware MPI

For direct GPU-to-GPU communication:

```julia
using TarangCUDAExt: check_cuda_aware_mpi

if check_cuda_aware_mpi()
    println("CUDA-aware MPI available - using direct GPU transfers")
else
    error("CUDA-aware MPI is required; implicit CPU staging is disabled")
end
```

GPU LBVPs require a CUDA matrix solver such as `:cuda_sparse`, `:cuda_cg`, or
`:cuda_gmres`; CPU-only choices are rejected. GPU NLBVP and EVP solves are not
yet device-native and raise an explicit unsupported-operation error.

### TransposableField for GPU+MPI

Efficient distributed FFTs with GPU:

```julia
# Create transposable field for distributed transforms
field = ScalarField(dist, "u", bases)
tf = TransposableField(field)

# Distributed forward transform (handles GPU transposes)
distributed_forward_transform!(tf)

# Distributed backward transform
distributed_backward_transform!(tf)
```

## Performance Optimization

### Best Practices

1. **Use Float32 when possible** - 2x memory bandwidth, often sufficient accuracy
   ```julia
   dist = Distributor(coords; dtype=Float32, device=GPU())
   ```

2. **Batch operations** - Minimize kernel launches
   ```julia
   # Bad: many small operations
   for i in 1:n
       gpu_scale!(field, α)
   end

   # Good: fused operations
   gpu_rk_stage!(u_new, u, k, dt, coeff)
   ```

3. **Preallocate buffers** - Avoid allocation in hot loops
   ```julia
   # Preallocate work arrays
   work = zeros(GPU(), Float64, size(field["g"]))

   for step in 1:nsteps
       # Reuse work array
       compute!(work, field)
   end
   ```

4. **Synchronize when needed** - Ensure GPU operations complete before CPU access
   ```julia
   CUDA.synchronize()  # Wait for all GPU operations to finish
   ```

### Profiling

```julia
using CUDA

# Profile a section
CUDA.@profile begin
    forward_transform!(field)
    backward_transform!(field)
end

# Time with synchronization
CUDA.@elapsed begin
    forward_transform!(field)
    CUDA.synchronize()
end
```

### Memory Bandwidth Optimization

```julia
# Check if transform is memory-bound
n = prod(size(field["g"]))
bytes_transferred = n * sizeof(eltype(field["g"])) * 4  # rough estimate

# Theoretical bandwidth (e.g., A100 = 2 TB/s)
theoretical_time = bytes_transferred / 2e12

# Compare to actual time
actual_time = CUDA.@elapsed forward_transform!(field)
efficiency = theoretical_time / actual_time
println("Memory bandwidth efficiency: $(efficiency * 100)%")
```

## Tensor Core Support

For supported operations on Ampere+ GPUs:

```julia
using TarangCUDAExt: enable_tensor_cores!, disable_tensor_cores!

# Enable tensor cores (requires compatible data types)
enable_tensor_cores!()

# Disable if numerical precision is critical
disable_tensor_cores!()
```

## Troubleshooting

### Common Issues

**Out of Memory**
```julia
# Check available memory before large allocations
info = gpu_memory_info()
if info.free_bytes < required_bytes
    clear_memory_pool!()  # Free cached allocations
    GC.gc()               # Trigger garbage collection
    CUDA.reclaim()        # Reclaim CUDA memory
end
```

**Slow Performance**
```julia
# Ensure synchronization isn't killing performance
CUDA.allowscalar(false)  # Disable slow scalar indexing

# Strict dispatch also checks this before every transform
@assert field["g"] isa CuArray "Data not on GPU!"
```

GPU transforms, resampling, polynomial derivatives, coupled IVP solves, and
stochastic phase generation do not fall back to CPU. Unsupported operations
raise an error naming the missing device path.

**MPI + CUDA Issues**
```julia
# Ensure correct GPU assignment
println("Rank $rank using GPU $(CUDA.device())")

# Force synchronization before MPI calls
CUDA.synchronize()
MPI.Barrier(comm)
```

## API Reference

See the [GPU API documentation](../api/gpu.md) for complete function references.

### Key Functions

| Function | Description |
|----------|-------------|
| `GPU()` | GPU architecture singleton |
| `on_architecture(GPU(), array)` | Move array to GPU |
| `forward_transform!(field)` | GPU-accelerated forward FFT |
| `backward_transform!(field)` | GPU-accelerated inverse FFT |
| `set_gpu_fft_mode!(field, mode)` | Control FFT backend |
| `TransposableField(field)` | Distributed GPU transforms |
