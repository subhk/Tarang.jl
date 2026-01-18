# GPU API Reference

This page documents the GPU-specific functions and types in Tarang.jl.

## Architecture Types

```@docs
GPU
CPU
AbstractArchitecture
```

## Array Operations

### Data Movement

```julia
# Move array to GPU
gpu_array = on_architecture(GPU(), cpu_array)

# Move array to CPU
cpu_array = on_architecture(CPU(), gpu_array)

# Check if array is on GPU
is_gpu_array(array)
```

### Allocation

```julia
# Allocate on specific architecture
zeros(GPU(), Float64, 128, 128)
ones(GPU(), Float32, 256)
similar(gpu_array)
```

## Transform Functions

### FFT Control

| Function | Description |
|----------|-------------|
| `set_gpu_fft_mode!(field, mode)` | Set FFT backend (`:auto`, `:cpu`, `:gpu`) |
| `set_gpu_fft_min_elements!(n)` | Threshold for auto mode |
| `should_use_gpu_fft(field)` | Check if GPU FFT will be used |

### Transform Execution

| Function | Description |
|----------|-------------|
| `forward_transform!(field)` | Forward spectral transform |
| `backward_transform!(field)` | Backward spectral transform |
| `gpu_forward_transform!(field)` | Force GPU transform |
| `gpu_backward_transform!(field)` | Force GPU transform |

## GPU Kernels

### Element-wise Operations

```julia
# Basic operations
gpu_add!(c, a, b)           # c = a + b
gpu_sub!(c, a, b)           # c = a - b
gpu_mul!(c, a, b)           # c = a .* b
gpu_scale!(y, α, x)         # y = α * x
gpu_axpy!(y, α, x)          # y = y + α * x

# Linear combinations
gpu_linear_combination!(y, α, a, β, b)  # y = α*a + β*b
gpu_axpby!(y, α, x, β)                  # y = α*x + β*y
```

### Timestepping Kernels

```julia
# Runge-Kutta stage update
gpu_rk_stage!(u_new, u, k, dt, coeff)

# Fused multiply-add
gpu_fma!(y, a, b, c)  # y = a*b + c
```

### Physics Kernels

```julia
# Kinetic energy
gpu_kinetic_energy_2d!(ke, ux, uy)
gpu_kinetic_energy_3d!(ke, ux, uy, uz)

# Gradient magnitude squared
gpu_grad_mag_sq_2d!(result, fx, fy)

# Viscous damping in spectral space
gpu_viscous_damping!(field, ν, k2)

# Dealiasing
gpu_dealias_multiply!(result, a, b, mask)
gpu_triple_product!(result, a, b, c)
```

### Complex Operations

```julia
# Complex conjugate multiply
gpu_conj_multiply!(c, a, b)  # c = conj(a) * b

# Squared magnitude
gpu_squared_magnitude!(result, z)  # result = |z|²
```

## Memory Management

### Memory Pool

```julia
# Pool statistics
stats = memory_pool_stats()

# Clear cached memory
clear_memory_pool!()

# Allocate from pool
buffer = pool_allocate(Float64, 1024)
pool_release!(buffer)
```

### Pinned Memory

```julia
# Get pinned buffer for async transfers
buffer = get_pinned_buffer(Float64, size)

# Async copy operations
async_copy_to_gpu!(gpu_dest, pinned_src)
async_copy_to_cpu!(pinned_dest, gpu_src)
```

### Memory Info

```julia
# GPU memory status
info = gpu_memory_info()
# Returns: (free_bytes, total_bytes, used_bytes)

# Check if allocation is possible
can_alloc = check_gpu_memory(required_bytes)
```

## Stream Management

```julia
# Get dedicated streams
compute_stream = get_compute_stream()
transfer_stream = get_transfer_stream()

# Synchronize all streams
sync_streams!()
```

## FFT Plans

### Standard FFT Plans

```julia
# Create GPU FFT plan
plan = plan_gpu_fft(arch, size, eltype)

# Execute transforms
gpu_forward_fft!(output, input, plan)
gpu_backward_fft!(output, input, plan)

# Async FFT
gpu_fft_async!(output, input, plan, stream)
```

### Batched FFT

```julia
# Create batched plan
plan = plan_batched_gpu_fft(arch, batch_size, fft_size, eltype)

# Execute batched transforms
batched_fft!(output, input, plan)
batched_ifft!(output, input, plan)
```

### DCT Plans (Chebyshev)

```julia
# Create DCT plan for specific dimension
plan = plan_gpu_dct_dim(arch, full_size, eltype, dim)

# Execute DCT
gpu_dct_dim!(output, input, plan, Val(:forward))
gpu_dct_dim!(output, input, plan, Val(:backward))
```

### Mixed Transform Plans

```julia
# Create mixed Fourier-Chebyshev plan
plan = plan_gpu_mixed_transform(arch, bases, size)

# Execute mixed transforms
gpu_mixed_forward_transform!(output, input, plan)
gpu_mixed_backward_transform!(output, input, plan)
```

## KernelOperation

Portable kernel wrapper for CPU/GPU:

```julia
using KernelAbstractions

# Define kernel
@kernel function my_kernel!(y, @Const(x), α)
    i = @index(Global)
    @inbounds y[i] = α * x[i]
end

# Create operation
my_op = KernelOperation(my_kernel!) do y, x, α
    length(y)  # ndrange
end

# Execute on any architecture
my_op(arch, y, x, α)
```

### Built-in Operations

| Operation | Description |
|-----------|-------------|
| `GPU_ADD_OP` | Element-wise addition |
| `GPU_SUB_OP` | Element-wise subtraction |
| `GPU_MUL_OP` | Element-wise multiplication |
| `GPU_SCALE_OP` | Scalar multiplication |
| `GPU_AXPY_OP` | AXPY operation |

## Distributed GPU

### TransposableField

```julia
# Create transposable field for distributed transforms
tf = TransposableField(field)

# Distributed transforms with GPU support
distributed_forward_transform!(tf)
distributed_backward_transform!(tf)

# Access current layout
layout = active_layout(tf)  # XLocal, YLocal, or ZLocal
data = current_data(tf)
```

### CUDA-Aware MPI

```julia
# Check for CUDA-aware MPI
is_cuda_aware = check_cuda_aware_mpi()

# GPU pack/unpack for transposes
gpu_pack_for_transpose!(buffer, data, counts, displs, dim, nranks)
gpu_unpack_from_transpose!(data, buffer, counts, displs, dim, nranks)
```

## Configuration

### GPU Config

```julia
# Initialize GPU configuration
init_gpu_config!()

# Get/set configuration
config = GPU_CONFIG

# Tensor core support (Ampere+)
enable_tensor_cores!()
disable_tensor_cores!()
```

### Device Management

```julia
# Ensure correct device context
ensure_device!(arch)

# For multi-GPU
CUDA.device!(rank % CUDA.ndevices())
```

## Cache Management

```julia
# Clear transform caches
clear_gpu_transform_cache!()
clear_batched_fft_cache!()
clear_gpu_mixed_transform_cache!()

# Get cached plan
plan = get_gpu_fft_plan(size, eltype)
plan = get_batched_fft_plan(batch_size, fft_size, eltype)
```
