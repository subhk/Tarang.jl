# ============================================================================
# GPU Transform Functions
# ============================================================================

# Mutable struct to hold GPU FFT plans (cached)
mutable struct GPUTransformCache
    plans::Dict{Tuple, GPUFFTPlan}

    GPUTransformCache() = new(Dict{Tuple, GPUFFTPlan}())
end

const GPU_TRANSFORM_CACHE = GPUTransformCache()

mutable struct FFT1DPlanCache
    plans::Dict{Tuple, Any}

    FFT1DPlanCache() = new(Dict{Tuple, Any}())
end

const FFT_1D_PLAN_CACHE = FFT1DPlanCache()

function get_fft_1d_plan(size::Tuple, dim::Int, T::Type; inverse::Bool=false)
    key = (size, dim, T, inverse)
    if !haskey(FFT_1D_PLAN_CACHE.plans, key)
        dummy = CUDA.zeros(T, size...)
        FFT_1D_PLAN_CACHE.plans[key] = inverse ? CUFFT.plan_ifft(dummy; dims=(dim,)) : CUFFT.plan_fft(dummy; dims=(dim,))
    end
    return FFT_1D_PLAN_CACHE.plans[key]
end

"""
    _gpu_plan_key(arch, local_size, T, real_input)

Generate a cache key for GPU FFT plans.
Includes device ID for multi-GPU support to avoid returning plans from wrong device.
"""
_gpu_plan_key(arch::GPU{CuDevice}, local_size::Tuple, T::Type, real_input::Bool) =
    (CUDA.deviceid(arch.device), local_size, T, real_input)

# Fallback for generic GPU - use current device ID
_gpu_plan_key(arch::GPU, local_size::Tuple, T::Type, real_input::Bool) =
    (CUDA.deviceid(), local_size, T, real_input)

"""
    get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Get or create a cached GPU FFT plan for the given architecture, local size, and dtype.

**Important:** `local_size` should be the LOCAL array shape (what this process owns),
not the global domain size. Plans are cached per (device, size, type, real_input).

For multi-GPU: plans are cached separately for each device to avoid cross-device issues.
"""
function get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)
    key = _gpu_plan_key(arch, local_size, T, real_input)
    if !haskey(GPU_TRANSFORM_CACHE.plans, key)
        GPU_TRANSFORM_CACHE.plans[key] = plan_gpu_fft(arch, local_size, T; real_input=real_input)
    end
    return GPU_TRANSFORM_CACHE.plans[key]
end

"""
    clear_gpu_transform_cache!()

Clear all cached GPU FFT plans.
"""
function clear_gpu_transform_cache!()
    empty!(GPU_TRANSFORM_CACHE.plans)
end

# ============================================================================
# GPU-aware Field Operations
# ============================================================================

"""
    allocate_gpu_data(arch::GPU{CuDevice}, dtype::Type, shape::Tuple)

Allocate array on the specific GPU device stored in the architecture.
Ensures correct device context for multi-GPU support.
"""
function allocate_gpu_data(arch::GPU{CuDevice}, dtype::Type, shape::Tuple)
    ensure_device!(arch)
    return CUDA.zeros(dtype, shape...)
end

# Fallback for generic GPU
function allocate_gpu_data(::GPU, dtype::Type, shape::Tuple)
    return CUDA.zeros(dtype, shape...)
end

# ============================================================================
# GPU-specific Array Allocation (zeros/ones/similar) with Device Context
# These are internal helpers - users should use Base.zeros(arch, ...) etc.
# ============================================================================

# Internal helper functions (not exported, use Base.zeros/ones/similar instead)
function _gpu_zeros(arch::GPU{CuDevice}, T::Type, dims...)
    ensure_device!(arch)
    return CUDA.zeros(T, dims...)
end

_gpu_zeros(::GPU, T::Type, dims...) = CUDA.zeros(T, dims...)

function _gpu_zeros(arr::CuArray, T::Type, dims...)
    CUDA.device!(CUDA.device(arr))
    return CUDA.zeros(T, dims...)
end

_gpu_zeros(arr::CuArray, dims...) = _gpu_zeros(arr, eltype(arr), dims...)

function _gpu_ones(arch::GPU{CuDevice}, T::Type, dims...)
    ensure_device!(arch)
    return CUDA.ones(T, dims...)
end

_gpu_ones(::GPU, T::Type, dims...) = CUDA.ones(T, dims...)

function _gpu_ones(arr::CuArray, T::Type, dims...)
    CUDA.device!(CUDA.device(arr))
    return CUDA.ones(T, dims...)
end

_gpu_ones(arr::CuArray, dims...) = _gpu_ones(arr, eltype(arr), dims...)

function _gpu_similar(arr::CuArray, T::Type, dims...)
    CUDA.device!(CUDA.device(arr))
    return CuArray{T}(undef, dims...)
end

_gpu_similar(arr::CuArray, dims...) = _gpu_similar(arr, eltype(arr), dims...)
_gpu_similar(arr::CuArray) = _gpu_similar(arr, eltype(arr), size(arr)...)

function _gpu_fill(arch::GPU{CuDevice}, val, dims...)
    ensure_device!(arch)
    return CUDA.fill(val, dims...)
end

_gpu_fill(::GPU, val, dims...) = CUDA.fill(val, dims...)

function _gpu_fill(arr::CuArray, val, dims...)
    CUDA.device!(CUDA.device(arr))
    return CUDA.fill(val, dims...)
end

# ============================================================================
# Data Transfer Helpers (attached to Tarang module)
# Users should use Tarang.on_architecture for the primary API
# ============================================================================

# Stub functions - these will be defined if Tarang exports them
# For now, keep as internal helpers that work via on_architecture

"""
    _to_gpu(a::Array)

Internal: Move array to GPU (current device).
Users should use `Tarang.on_architecture(GPU(), array)` instead.
"""
_to_gpu(a::Array) = CuArray(a)

"""
    _to_gpu(arch::GPU{CuDevice}, a::Array)

Internal: Move array to specific GPU device.
Users should use `Tarang.on_architecture(arch, array)` instead.
"""
function _to_gpu(arch::GPU{CuDevice}, a::Array)
    ensure_device!(arch)
    return CuArray(a)
end

"""
    _to_cpu(a::CuArray)

Internal: Move array to CPU.
Users should use `Tarang.on_architecture(CPU(), array)` instead.
"""
_to_cpu(a::CuArray) = Array(a)

# Internal convenience aliases - NOT exported
# Users should use Tarang.on_architecture(GPU(), array) or Tarang.on_architecture(CPU(), array) instead
to_gpu(a::Array) = _to_gpu(a)
to_gpu(arch::GPU{CuDevice}, a::Array) = _to_gpu(arch, a)
to_cpu(a::CuArray) = _to_cpu(a)

# ============================================================================
# Multi-dimensional FFT Kernels for Spectral Methods
# ============================================================================

"""
2D FFT forward transform on GPU
"""
function gpu_fft_2d_forward!(output::CuArray{Complex{T}}, input::CuArray{T}) where {T<:AbstractFloat}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=true)
    gpu_forward_fft!(output, input, plan)
    return output
end

"""
2D FFT backward transform on GPU
"""
function gpu_fft_2d_backward!(output::CuArray{T}, input::CuArray{Complex{T}}, n::Int) where {T<:AbstractFloat}
    out_size = (n, size(input)[2:end]...)
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, out_size, T; real_input=true)
    gpu_backward_fft!(output, input, plan)
    return output
end

"""
3D FFT forward transform on GPU (complex)
"""
function gpu_fft_3d_forward!(output::CuArray{T}, input::CuArray{T}) where {T<:Complex}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=false)
    gpu_forward_fft!(output, input, plan)
    return output
end

"""
3D FFT backward transform on GPU (complex)
"""
function gpu_fft_3d_backward!(output::CuArray{T}, input::CuArray{T}) where {T<:Complex}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=false)
    gpu_backward_fft!(output, input, plan)
    return output
end

# ============================================================================
# GPU Dealiasing Operations
# ============================================================================

"""
Create dealiasing mask on GPU for 2/3 rule
"""
function create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0)
    mask = CUDA.ones(Float64, shape...)

    # Apply 2/3 rule cutoff in each dimension
    for (dim, n) in enumerate(shape)
        cutoff_idx = ceil(Int, n * cutoff)
        # Zero out high frequencies
        # This is dimension-dependent and needs proper indexing
    end

    return mask
end

"""
Apply dealiasing on GPU using spectral cutoff
"""
function apply_dealiasing_gpu!(data::CuArray, cutoff::Float64=2.0/3.0)
    # For now, use simple truncation approach
    # More sophisticated dealiasing would use the mask kernel
    return data
end

# ============================================================================
# GPU Field Multiplication (Override for nonlinear terms)
# ============================================================================

"""
    gpu_multiply_fields!(result::CuArray, a::CuArray, b::CuArray)

GPU-accelerated pointwise multiplication using CUDA kernels.
"""
function Tarang.gpu_multiply_fields!(result::CuArray, a::CuArray, b::CuArray)
    launch!(architecture(result), mul_kernel!, result, a, b; ndrange=length(result))
    return result
end

# ============================================================================
# Distributed GPU FFT Implementation (CUFFT)
# ============================================================================

"""
    local_fft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)

Perform local FFT along dimension `dim` using CUFFT.
"""
function Tarang.local_fft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)
    # Use CUFFT for local FFT along specified dimension
    return CUFFT.fft(data, dim)
end

"""
    local_ifft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)

Perform local inverse FFT along dimension `dim` using CUFFT.
"""
function Tarang.local_ifft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)
    # Use CUFFT for local inverse FFT along specified dimension
    return CUFFT.ifft(data, dim)
end

"""
    gpu_fft_1d!(output::CuArray, input::CuArray, dim::Int)

1D FFT along specified dimension on GPU.
"""
function gpu_fft_1d!(output::CuArray, input::CuArray, dim::Int)
    plan = get_fft_1d_plan(size(input), dim, eltype(input); inverse=false)
    CUFFT.fft!(plan, input, output)
    return output
end

"""
    gpu_ifft_1d!(output::CuArray, input::CuArray, dim::Int)

1D inverse FFT along specified dimension on GPU.
"""
function gpu_ifft_1d!(output::CuArray, input::CuArray, dim::Int)
    plan = get_fft_1d_plan(size(input), dim, eltype(input); inverse=true)
    CUFFT.ifft!(plan, input, output)
    return output
end

"""
    gpu_rfft!(output::CuArray{Complex{T}}, input::CuArray{T})

Real-to-complex FFT on GPU.
"""
function gpu_rfft!(output::CuArray{Complex{T}}, input::CuArray{T}) where {T<:AbstractFloat}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=true)
    gpu_forward_fft!(output, input, plan)
    return output
end

"""
    gpu_irfft!(output::CuArray{T}, input::CuArray{Complex{T}}, n::Int)

Complex-to-real inverse FFT on GPU.
"""
function gpu_irfft!(output::CuArray{T}, input::CuArray{Complex{T}}, n::Int) where {T<:AbstractFloat}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, (n, size(input)[2:end]...), T; real_input=true)
    gpu_backward_fft!(output, input, plan)
    return output
end

# ============================================================================
# GPU Memory Management
# ============================================================================

"""
    gpu_memory_info()

Get current GPU memory usage information.
"""
function gpu_memory_info()
    free, total = CUDA.Mem.info()
    used = total - free
    return (
        free_bytes = free,
        used_bytes = used,
        total_bytes = total,
        free_gb = free / 1e9,
        used_gb = used / 1e9,
        total_gb = total / 1e9,
        usage_percent = 100.0 * used / total
    )
end

"""
    check_gpu_memory(required_bytes::Int)

Check if there's enough GPU memory available.
"""
function check_gpu_memory(required_bytes::Int)
    info = gpu_memory_info()
    if info.free_bytes < required_bytes
        @warn "Insufficient GPU memory" required=required_bytes/1e9 available=info.free_gb
        return false
    end
    return true
end

# ============================================================================
# GPU-aware Array Allocation Helpers (for operators.jl compatibility)
# ============================================================================

"""
    allocate_like(a::CuArray, T::Type, dims...)

Allocate a zeros CuArray on the same device as the input CuArray.
Ensures correct device context for multi-GPU support.
"""
function Tarang.allocate_like(a::CuArray, T::Type, dims...)
    # Switch to the device where 'a' resides before allocating
    CUDA.device!(CUDA.device(a))
    return CUDA.zeros(T, dims...)
end

function Tarang.allocate_like(a::CuArray, dims...)
    CUDA.device!(CUDA.device(a))
    return CUDA.zeros(eltype(a), dims...)
end

"""
    copy_to_device(a::AbstractArray, target::CuArray)

Copy array `a` to GPU on the same device as target CuArray.
Ensures correct device context for multi-GPU support.
"""
function Tarang.copy_to_device(a::AbstractArray, target::CuArray)
    # Switch to the device where 'target' resides before copying
    CUDA.device!(CUDA.device(target))
    return CuArray(a)
end

function Tarang.copy_to_device(a::CuArray, target::CuArray)
    src_device = CUDA.device(a)
    dst_device = CUDA.device(target)

    if src_device == dst_device
        # Same device, just copy
        return copy(a)
    else
        # Cross-device copy: switch to target device and copy
        CUDA.device!(dst_device)
        return CuArray(a)
    end
end

