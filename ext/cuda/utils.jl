# ============================================================================
# GPU Transform Functions
# ============================================================================

# GPUTransformCache and GPU_TRANSFORM_CACHE are defined in transforms.jl

mutable struct FFT1DPlanCache
    plans::Dict{Tuple, Any}

    FFT1DPlanCache() = new(Dict{Tuple, Any}())
end

const FFT_1D_PLAN_CACHE = FFT1DPlanCache()

function get_fft_1d_plan(size::Tuple, dim::Int, T::Type; inverse::Bool=false)
    # Include device ID in cache key for multi-GPU correctness
    device_id = CUDA.deviceid()
    key = (device_id, size, dim, T, inverse)
    if !haskey(FFT_1D_PLAN_CACHE.plans, key)
        dummy = CUDA.zeros(T, size...)
        FFT_1D_PLAN_CACHE.plans[key] = inverse ? CUFFT.plan_ifft(dummy; dims=(dim,)) : CUFFT.plan_fft(dummy; dims=(dim,))
    end
    return FFT_1D_PLAN_CACHE.plans[key]
end

# get_gpu_fft_plan and clear_gpu_transform_cache! are defined in transforms.jl

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
Dealiasing kernel for 2D arrays: zero out modes beyond cutoff in each dimension.
For spectral truncation (2/3 rule), modes with index > cutoff_dim are zeroed.
"""
@kernel function dealiasing_2d_kernel!(data, cutoff_x::Int, cutoff_y::Int, nx::Int, ny::Int)
    idx = @index(Global)
    j = ((idx - 1) ÷ nx) + 1
    i = ((idx - 1) % nx) + 1
    @inbounds if i > cutoff_x || j > cutoff_y
        data[i, j] = zero(eltype(data))
    end
end

"""
Dealiasing kernel for 3D arrays: zero out modes beyond cutoff in each dimension.
"""
@kernel function dealiasing_3d_kernel!(data, cutoff_x::Int, cutoff_y::Int, cutoff_z::Int,
                                        nx::Int, ny::Int, nz::Int)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    j = (((idx - 1) ÷ nx) % ny) + 1
    k = ((idx - 1) ÷ (nx * ny)) + 1
    @inbounds if i > cutoff_x || j > cutoff_y || k > cutoff_z
        data[i, j, k] = zero(eltype(data))
    end
end

"""
    create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0)

Create dealiasing mask on GPU for the 2/3 rule.
Returns a CuArray where entries within the cutoff are 1.0 and entries
beyond the cutoff in any dimension are 0.0.
"""
function create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0)
    mask = CUDA.ones(Float64, shape...)
    cutoffs = map(n -> floor(Int, n * cutoff), shape)

    if length(shape) == 2
        nx, ny = shape
        arch = Tarang.architecture(mask)
        launch!(arch, dealiasing_2d_kernel!, mask, cutoffs[1], cutoffs[2], nx, ny;
                ndrange=nx*ny)
    elseif length(shape) == 3
        nx, ny, nz = shape
        arch = Tarang.architecture(mask)
        launch!(arch, dealiasing_3d_kernel!, mask, cutoffs[1], cutoffs[2], cutoffs[3],
                nx, ny, nz; ndrange=nx*ny*nz)
    else
        # 1D fallback
        n = shape[1]
        cutoff_idx = floor(Int, n * cutoff)
        if cutoff_idx < n
            fill!(view(mask, cutoff_idx+1:n), 0.0)
        end
    end

    return mask
end

"""
    apply_dealiasing_gpu!(data::CuArray, cutoff::Float64=2.0/3.0)

Apply 2/3-rule dealiasing on GPU by zeroing spectral modes beyond the cutoff
in each dimension. Modifies `data` in-place.
"""
function apply_dealiasing_gpu!(data::CuArray{T, 2}, cutoff::Float64=2.0/3.0) where T
    nx, ny = size(data)
    cutoff_x = floor(Int, nx * cutoff)
    cutoff_y = floor(Int, ny * cutoff)
    arch = Tarang.architecture(data)
    launch!(arch, dealiasing_2d_kernel!, data, cutoff_x, cutoff_y, nx, ny;
            ndrange=nx*ny)
    return data
end

function apply_dealiasing_gpu!(data::CuArray{T, 3}, cutoff::Float64=2.0/3.0) where T
    nx, ny, nz = size(data)
    cutoff_x = floor(Int, nx * cutoff)
    cutoff_y = floor(Int, ny * cutoff)
    cutoff_z = floor(Int, nz * cutoff)
    arch = Tarang.architecture(data)
    launch!(arch, dealiasing_3d_kernel!, data, cutoff_x, cutoff_y, cutoff_z,
            nx, ny, nz; ndrange=nx*ny*nz)
    return data
end

function apply_dealiasing_gpu!(data::CuArray{T, 1}, cutoff::Float64=2.0/3.0) where T
    n = length(data)
    cutoff_idx = floor(Int, n * cutoff)
    if cutoff_idx < n
        fill!(view(data, cutoff_idx+1:n), zero(T))
    end
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
        # Cross-device copy: explicitly go through host memory to avoid
        # requiring P2P access between devices.
        # 1. Set source device context and download to host
        CUDA.device!(src_device)
        host_data = Array(a)
        # 2. Set destination device context and upload from host
        CUDA.device!(dst_device)
        return CuArray(host_data)
    end
end

