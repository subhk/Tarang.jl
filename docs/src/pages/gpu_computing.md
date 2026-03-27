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
dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=GPU())

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
dist_cpu = Distributor(coords; architecture=CPU())

# GPU execution
dist_gpu = Distributor(coords; architecture=GPU())

# Check architecture
arch = dist_gpu.architecture  # GPU()
```

## GPU Transforms

### Automatic FFT Acceleration

When fields are on GPU, transforms automatically use CUFFT:

```julia
using Tarang, CUDA

dist = Distributor(coords; architecture=GPU())
field = ScalarField(dist, "u", (xbasis, ybasis))

# Initialize with GPU data
field["g"] .= CUDA.rand(Float64, size(field["g"])...)

# Forward transform (uses CUFFT automatically)
forward_transform!(field)

# Backward transform
backward_transform!(field)
```

### FFT Mode Control

Control when GPU FFTs are used:

```julia
# Per-field control
set_gpu_fft_mode!(field, :gpu)   # Always use GPU FFT
set_gpu_fft_mode!(field, :cpu)   # Always use CPU FFT
set_gpu_fft_mode!(field, :auto)  # Heuristic-based (default)

# Global threshold for :auto mode
# Use GPU FFT only if array has >= N elements
set_gpu_fft_min_elements!(64_000)
```

### Mixed Fourier-Chebyshev Transforms

GPU DCT for Chebyshev bases:

```julia
# Mixed basis domain
xbasis = Fourier(coords, "x", 256)      # FFT
zbasis = ChebyshevT(coords, "z", 64)    # DCT

dist = Distributor(coords; architecture=GPU())
field = ScalarField(dist, "T", (xbasis, zbasis))

# Transforms automatically select FFT or DCT per dimension
forward_transform!(field)   # FFT in x, DCT in z
backward_transform!(field)  # IFFT in x, IDCT in z
```

## GPU Memory Management

### Memory Pools

Tarang uses memory pooling to reduce allocation overhead:

```julia
using TarangCUDAExt: GPUMemoryPool, pool_allocate, pool_release!

# Get memory pool statistics
stats = memory_pool_stats()
println("Allocated: $(stats.allocated_bytes) bytes")
println("Cached: $(stats.cached_bytes) bytes")

# Clear the memory pool (frees cached memory)
clear_memory_pool!()
```

### Pinned Memory for MPI

For efficient GPU-MPI transfers:

```julia
using TarangCUDAExt: get_pinned_buffer, async_copy_to_gpu!, async_copy_to_cpu!

# Get a pinned CPU buffer for async transfers
buffer = get_pinned_buffer(Float64, 1024)

# Async copy operations
async_copy_to_gpu!(gpu_array, buffer)
async_copy_to_cpu!(buffer, gpu_array)
```

### Memory Monitoring

```julia
using TarangCUDAExt: gpu_memory_info, check_gpu_memory

# Get current memory usage
info = gpu_memory_info()
println("Free: $(info.free_bytes / 1e9) GB")
println("Total: $(info.total_bytes / 1e9) GB")

# Check if allocation will fit
can_allocate = check_gpu_memory(required_bytes)
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
dist = Distributor(coords; mesh=(2, 2), architecture=GPU())

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
    println("Staging through CPU for MPI transfers")
end
```

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
   dist = Distributor(coords; dtype=Float32, architecture=GPU())
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

4. **Use streams for overlap** - Overlap computation and communication
   ```julia
   using TarangCUDAExt: get_compute_stream, get_transfer_stream, sync_streams!

   compute_stream = get_compute_stream()
   transfer_stream = get_transfer_stream()
   # ... overlap operations
   sync_streams!()
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

# Check for CPU fallbacks
@assert field["g"] isa CuArray "Data not on GPU!"
```

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
