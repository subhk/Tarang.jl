# ============================================================================
# GPU Transform Functions
# ============================================================================

# GPUTransformCache and GPU_TRANSFORM_CACHE are defined in transforms.jl

mutable struct FFT1DPlanCache
    plans::Dict{Tuple, Any}
    lock::ReentrantLock

    FFT1DPlanCache() = new(Dict{Tuple, Any}(), ReentrantLock())
end

const FFT_1D_PLAN_CACHE = FFT1DPlanCache()

function get_fft_1d_plan(size::Tuple, dim::Int, T::Type; inverse::Bool=false, device_id::Int=_current_device_id())
    # Include device ID in cache key for multi-GPU correctness
    key = (device_id, size, dim, T, inverse)
    lock(FFT_1D_PLAN_CACHE.lock)
    try
        if !haskey(FFT_1D_PLAN_CACHE.plans, key)
            # Ensure plan is created on the correct device
            prev_device = CUDA.device()
            CUDA.device!(CuDevice(device_id))
            try
                dummy = CUDA.zeros(T, size...)
                FFT_1D_PLAN_CACHE.plans[key] = inverse ? CUFFT.plan_ifft(dummy, (dim,)) : CUFFT.plan_fft(dummy, (dim,))
            finally
                CUDA.device!(prev_device)
            end
        end
        return FFT_1D_PLAN_CACHE.plans[key]
    finally
        unlock(FFT_1D_PLAN_CACHE.lock)
    end
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
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CUDA.zeros(T, dims...)
    finally
        # try/finally: an exception mid-allocation (e.g. OOM) must not strand the
        # task on the wrong device (matches Tarang.allocate_like below).
        CUDA.device!(prev_device)
    end
end

_gpu_zeros(arr::CuArray, dims...) = _gpu_zeros(arr, eltype(arr), dims...)

function _gpu_ones(arch::GPU{CuDevice}, T::Type, dims...)
    ensure_device!(arch)
    return CUDA.ones(T, dims...)
end

_gpu_ones(::GPU, T::Type, dims...) = CUDA.ones(T, dims...)

function _gpu_ones(arr::CuArray, T::Type, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CUDA.ones(T, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

_gpu_ones(arr::CuArray, dims...) = _gpu_ones(arr, eltype(arr), dims...)

function _gpu_similar(arr::CuArray, T::Type, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CuArray{T}(undef, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

_gpu_similar(arr::CuArray, dims...) = _gpu_similar(arr, eltype(arr), dims...)
_gpu_similar(arr::CuArray) = _gpu_similar(arr, eltype(arr), size(arr)...)

function _gpu_fill(arch::GPU{CuDevice}, val, dims...)
    ensure_device!(arch)
    return CUDA.fill(val, dims...)
end

_gpu_fill(::GPU, val, dims...) = CUDA.fill(val, dims...)

function _gpu_fill(arr::CuArray, val, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CUDA.fill(val, dims...)
    finally
        CUDA.device!(prev_device)
    end
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
    _dealias_kmax(N, keep_fraction) -> Int

Max retained |mode| for an axis of a two-sided full FFT spectrum, using the SAME
rule as the CPU solver's `Tarang._axis_dealias_cutoff(basis, factor)` with
`factor = 1/keep_fraction`:

    kmax = min(floor(N / (2·factor)), (N − 1) ÷ 3)

The `(N − 1) ÷ 3` cap enforces `3·kmax < N`, so quadratic products (modes up to
`2·kmax`) cannot alias back into `[−kmax, kmax]` — see nonlinear_dealiasing.jl.
`keep_fraction ≥ 1` disables dealiasing (all modes kept), mirroring
`_axis_dealias_cutoff` returning `nothing` for factor ≤ 1; here we return `N`
(no index has |mode| > N, so nothing is zeroed).
"""
@inline function _dealias_kmax(N::Int, keep_fraction::Float64)
    keep_fraction >= 1 && return N   # factor ≤ 1 → keep every mode
    factor = 1.0 / keep_fraction
    return min(floor(Int, N / (2 * factor)), (N - 1) ÷ 3)
end

"""
Dealiasing kernel for 2D arrays (two-sided full FFT spectrum on BOTH axes):
zero every mode with |k| > kmax on either axis.

Index ↔ mode map per axis of length n: index i is mode i−1 on the positive side
or −(n−i+1) on the negative side. An entry is killed only when BOTH
interpretations exceed kmax, so the retained band is |k| ≤ kmax on each side
(conjugate-symmetric; DC at i==1 is always kept since kmax ≥ 0). The even-N
Nyquist index n÷2+1 has |k| = n/2 > kmax (kmax ≤ (n−1)÷3) and is always zeroed.
"""
@kernel function dealiasing_2d_kernel!(data, kmax_x::Int, kmax_y::Int, nx::Int, ny::Int)
    idx = @index(Global)
    j = ((idx - 1) ÷ nx) + 1
    i = ((idx - 1) % nx) + 1
    kill_x = (i - 1 > kmax_x) && (nx - i + 1 > kmax_x)
    kill_y = (j - 1 > kmax_y) && (ny - j + 1 > kmax_y)
    @inbounds if kill_x || kill_y
        data[i, j] = zero(eltype(data))
    end
end

"""
Dealiasing kernel for 3D arrays (two-sided full FFT spectrum on all axes):
zero every mode with |k| > kmax on any axis. See dealiasing_2d_kernel! for the
index ↔ mode map.
"""
@kernel function dealiasing_3d_kernel!(data, kmax_x::Int, kmax_y::Int, kmax_z::Int,
                                        nx::Int, ny::Int, nz::Int)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    j = (((idx - 1) ÷ nx) % ny) + 1
    k = ((idx - 1) ÷ (nx * ny)) + 1
    kill_x = (i - 1 > kmax_x) && (nx - i + 1 > kmax_x)
    kill_y = (j - 1 > kmax_y) && (ny - j + 1 > kmax_y)
    kill_z = (k - 1 > kmax_z) && (nz - k + 1 > kmax_z)
    @inbounds if kill_x || kill_y || kill_z
        data[i, j, k] = zero(eltype(data))
    end
end

"""
    create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0; eltype::Type=Float64)

Create a dealiasing mask on GPU: entries with |k| ≤ kmax on every axis are one,
all others zero, where kmax per axis follows the solver's
`Tarang._axis_dealias_cutoff` rule (`min(floor(N·cutoff/2), (N−1)÷3)` for the
default `cutoff = 2/3` retained fraction).

Every axis is treated as a TWO-SIDED full FFT spectrum (layout: DC, positive
freqs, [Nyquist,] negative freqs). rfft half-spectrum layouts are NOT supported
— a half-spectrum axis is indistinguishable from a short full spectrum by shape
alone, so it cannot be detected here; do not pass rfft output to this helper.

Supports 1–3 dimensions; other ranks throw an `ArgumentError`. `eltype` selects
the mask element type (default `Float64`).
"""
function create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0; eltype::Type=Float64)
    T = eltype
    nd = length(shape)
    1 <= nd <= 3 || throw(ArgumentError(
        "create_dealiasing_mask_gpu supports 1-, 2-, or 3-dimensional shapes, got ndims=$nd"))

    mask = CUDA.ones(T, shape...)
    kmaxs = map(n -> _dealias_kmax(n, cutoff), shape)

    if nd == 2
        nx, ny = shape
        arch = Tarang.architecture(mask)
        launch!(arch, dealiasing_2d_kernel!, mask, kmaxs[1], kmaxs[2], nx, ny;
                ndrange=nx*ny)
    elseif nd == 3
        nx, ny, nz = shape
        arch = Tarang.architecture(mask)
        launch!(arch, dealiasing_3d_kernel!, mask, kmaxs[1], kmaxs[2], kmaxs[3],
                nx, ny, nz; ndrange=nx*ny*nz)
    else
        # 1D: two-sided band |k| ≤ kmax → zero indices kmax+2 : n-kmax
        # (i−1 > kmax on the positive side AND n−i+1 > kmax on the negative side).
        n = shape[1]
        kmax = kmaxs[1]
        lo, hi = kmax + 2, n - kmax
        if lo <= hi
            fill!(view(mask, lo:hi), zero(T))
        end
    end

    return mask
end

"""
    apply_dealiasing_gpu!(data::CuArray, cutoff::Float64=2.0/3.0)

Zero spectral modes with |k| > kmax on any axis, in place, where kmax per axis
follows the solver's `Tarang._axis_dealias_cutoff` rule (see
`create_dealiasing_mask_gpu`). All axes are treated as two-sided full FFT
spectra; rfft half-spectrum layouts are NOT supported (undetectable from the
array shape — do not pass rfft output here).
"""
function apply_dealiasing_gpu!(data::CuArray{T, 2}, cutoff::Float64=2.0/3.0) where T
    nx, ny = size(data)
    kmax_x = _dealias_kmax(nx, cutoff)
    kmax_y = _dealias_kmax(ny, cutoff)
    arch = Tarang.architecture(data)
    launch!(arch, dealiasing_2d_kernel!, data, kmax_x, kmax_y, nx, ny;
            ndrange=nx*ny)
    return data
end

function apply_dealiasing_gpu!(data::CuArray{T, 3}, cutoff::Float64=2.0/3.0) where T
    nx, ny, nz = size(data)
    kmax_x = _dealias_kmax(nx, cutoff)
    kmax_y = _dealias_kmax(ny, cutoff)
    kmax_z = _dealias_kmax(nz, cutoff)
    arch = Tarang.architecture(data)
    launch!(arch, dealiasing_3d_kernel!, data, kmax_x, kmax_y, kmax_z,
            nx, ny, nz; ndrange=nx*ny*nz)
    return data
end

function apply_dealiasing_gpu!(data::CuArray{T, 1}, cutoff::Float64=2.0/3.0) where T
    n = length(data)
    kmax = _dealias_kmax(n, cutoff)
    # Two-sided band |k| ≤ kmax → zero indices kmax+2 : n-kmax (see mask helper).
    lo, hi = kmax + 2, n - kmax
    if lo <= hi
        fill!(view(data, lo:hi), zero(T))
    end
    return data
end

# ============================================================================
# GPU-Native Spectral Resampling
# ============================================================================

"""
    gpu_resample_grid_data!(new_data::CuArray, old_data::CuArray, old_shape, new_shape)

GPU-native spectral resampling using cuFFT: FFT → pad/truncate → IFFT entirely on GPU.
Avoids CPU round-trip for scale changes and dealiasing.
"""
function Tarang.gpu_resample_grid_data!(new_data::CuArray, old_data::CuArray,
                                        old_shape::Tuple, new_shape::Tuple)
    # Pin the current CUDA device to the array's device before any CUFFT plan /
    # CUDA.zeros below — every sibling GPU entry point does this; without it a
    # multi-GPU run whose current device != the data's device would allocate and
    # plan on the wrong device.
    ensure_device!(Tarang.architecture(new_data))
    old_size = size(old_data)
    new_size = size(new_data)

    if old_size == new_size
        copyto!(new_data, old_data)
        return true
    end

    ndims_data = length(old_size)
    if length(new_size) != ndims_data
        return false  # Unsupported; core raises instead of staging through CPU.
    end

    gpu_spectral_resample!(new_data, old_data)
    return true
end

"""
    gpu_spectral_resample!(new_data::CuArray{T}, old_data::CuArray{T})

Perform spectral interpolation on GPU: forward FFT, pad/truncate coefficients,
inverse FFT. Handles real and complex element types.
"""
function gpu_spectral_resample!(new_data::CuArray{T}, old_data::CuArray{T}) where T
    # Pin the device (also called directly, not only via gpu_resample_grid_data!).
    ensure_device!(Tarang.architecture(new_data))
    CT = T <: Real ? Complex{T} : T
    RT = T <: Real ? T : real(T)

    # Convert to complex for FFT if needed
    old_complex = T <: Real ? CuArray{CT}(old_data) : old_data

    # Forward FFT on GPU
    old_fft = CUFFT.fft(old_complex)

    # Create new spectral array and copy appropriate frequency components
    new_fft = CUDA.zeros(CT, size(new_data)...)
    spectral_pad_truncate_gpu!(new_fft, old_fft)

    # Scale for energy conservation
    scale_factor = RT(prod(size(new_data))) / RT(prod(size(old_data)))
    new_fft .*= scale_factor

    # Inverse FFT on GPU
    result = CUFFT.ifft(new_fft)
    if T <: Real
        new_data .= real.(result)
    else
        copyto!(new_data, result)
    end
end

"""
    _resample_axis_pairs(n_old::Int, n_new::Int) -> Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}

Per-axis (src_range, dst_range) copy list for spectral pad/truncate, replicating
EXACTLY the Nyquist conventions of the CPU `Tarang.resample_1d!`
(src/core/field/field_data/field_data_scales.jl):

- Upsampling from even n_old: the old Nyquist bin (index n_old÷2+1) is ZEROED
  (not copied) — copying it one-sidedly would leave a non-Hermitian spectrum and
  an aliased half-period oscillation on the fine grid.
- Downsampling to even n_new: the new Nyquist bin (index n_new÷2+1) receives
  old[+Nyq] PLUS the −Nyq image old[n_old−n_new÷2+1] (conjugate partners for a
  real field) — without the fold the new Nyquist mode comes out at half amplitude.

Destination ranges may overlap (the fold targets a bin inside the positive-copy
range), so consumers must ACCUMULATE (`.+=`) into a zeroed destination.
"""
function _resample_axis_pairs(n_old::Int, n_new::Int)
    pairs = Tuple{UnitRange{Int}, UnitRange{Int}}[]
    if n_old == n_new
        push!(pairs, (1:n_old, 1:n_new))
    elseif n_new > n_old
        # Upsample: zero-pad high frequencies.
        h_old = div(n_old, 2)
        if iseven(n_old)
            # Even n_old: copy positive freqs EXCLUDING the Nyquist (index h_old+1).
            push!(pairs, (1:h_old, 1:h_old))
        else
            # Odd n_old: no Nyquist bin; copy all positive freqs.
            push!(pairs, (1:h_old+1, 1:h_old+1))
        end
        n_neg = n_old - h_old - 1
        n_neg > 0 && push!(pairs, (n_old-n_neg+1:n_old, n_new-n_neg+1:n_new))
    else
        # Downsample: truncate high frequencies.
        h_new = div(n_new, 2)
        # Positive freqs including the new Nyquist (index h_new+1).
        push!(pairs, (1:h_new+1, 1:h_new+1))
        # Even n_new: fold the −Nyq image into the new Nyquist bin (accumulates
        # on top of the positive copy above).
        iseven(n_new) && push!(pairs, (n_old-h_new+1:n_old-h_new+1, h_new+1:h_new+1))
        n_neg = n_new - h_new - 1
        n_neg > 0 && push!(pairs, (n_old-n_neg+1:n_old, n_new-n_neg+1:n_new))
    end
    return pairs
end

"""
    spectral_pad_truncate_gpu!(new_fft::CuArray, old_fft::CuArray)

Copy FFT coefficients from old_fft into the ZEROED new_fft, handling the standard
FFT frequency layout (DC, positive freqs, [Nyquist,] negative freqs) per axis.
Pads with zeros for upsampling, truncates for downsampling, with the same even-N
Nyquist conventions as the CPU `Tarang.resample_1d!` (zero the old Nyquist on
upsample; fold the −Nyq image into the new Nyquist on downsample) — see
`_resample_axis_pairs`. Accumulates (`.+=`), so `new_fft` MUST be zero on entry
(the caller `gpu_spectral_resample!` allocates it with `CUDA.zeros`).
"""
function spectral_pad_truncate_gpu!(new_fft::CuArray{T,1}, old_fft::CuArray{T,1}) where T
    for (s1, d1) in _resample_axis_pairs(size(old_fft, 1), size(new_fft, 1))
        @views new_fft[d1] .+= old_fft[s1]
    end
    return new_fft
end

function spectral_pad_truncate_gpu!(new_fft::CuArray{T,2}, old_fft::CuArray{T,2}) where T
    pairs1 = _resample_axis_pairs(size(old_fft, 1), size(new_fft, 1))
    pairs2 = _resample_axis_pairs(size(old_fft, 2), size(new_fft, 2))
    # Tensor product of the per-axis maps == sequential per-axis resample_1d!.
    for (s1, d1) in pairs1, (s2, d2) in pairs2
        @views new_fft[d1, d2] .+= old_fft[s1, s2]
    end
    return new_fft
end

function spectral_pad_truncate_gpu!(new_fft::CuArray{T,3}, old_fft::CuArray{T,3}) where T
    pairs1 = _resample_axis_pairs(size(old_fft, 1), size(new_fft, 1))
    pairs2 = _resample_axis_pairs(size(old_fft, 2), size(new_fft, 2))
    pairs3 = _resample_axis_pairs(size(old_fft, 3), size(new_fft, 3))
    for (s1, d1) in pairs1, (s2, d2) in pairs2, (s3, d3) in pairs3
        @views new_fft[d1, d2, d3] .+= old_fft[s1, s2, s3]
    end
    return new_fft
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
    # Ensure correct device context for multi-GPU (CUFFT plans are device-specific)
    prev_device = CUDA.device()
    try
        CUDA.device!(CuDevice(dfft.config.device_id))
        result = CUFFT.fft(data, (dim,))
        CUDA.synchronize()
        return result
    finally
        CUDA.device!(prev_device)
    end
end

"""
    local_ifft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)

Perform local inverse FFT along dimension `dim` using CUFFT.
"""
function Tarang.local_ifft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)
    # Ensure correct device context for multi-GPU (CUFFT plans are device-specific)
    prev_device = CUDA.device()
    try
        CUDA.device!(CuDevice(dfft.config.device_id))
        result = CUFFT.ifft(data, (dim,))
        CUDA.synchronize()
        return result
    finally
        CUDA.device!(prev_device)
    end
end

"""
    gpu_fft_1d!(output::CuArray, input::CuArray, dim::Int)

1D FFT along specified dimension on GPU.
"""
function gpu_fft_1d!(output::CuArray, input::CuArray, dim::Int)
    # Derive device from input array for multi-GPU correctness
    input_device = CUDA.device(input)
    device_id = CUDA.deviceid(input_device)
    prev_device = CUDA.device()
    try
        CUDA.device!(input_device)
        plan = get_fft_1d_plan(size(input), dim, eltype(input); inverse=false, device_id=device_id)
        mul!(output, plan, input)
        return output
    finally
        CUDA.device!(prev_device)
    end
end

"""
    gpu_ifft_1d!(output::CuArray, input::CuArray, dim::Int)

1D inverse FFT along specified dimension on GPU.
"""
function gpu_ifft_1d!(output::CuArray, input::CuArray, dim::Int)
    # Derive device from input array for multi-GPU correctness
    input_device = CUDA.device(input)
    device_id = CUDA.deviceid(input_device)
    prev_device = CUDA.device()
    try
        CUDA.device!(input_device)
        plan = get_fft_1d_plan(size(input), dim, eltype(input); inverse=true, device_id=device_id)
        mul!(output, plan, input)
        return output
    finally
        CUDA.device!(prev_device)
    end
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
    # CUDA.jl v5 API (the old `CUDA.Mem.info()` was restructured out).
    free  = CUDA.available_memory()
    total = CUDA.total_memory()
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
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(a))
        return CUDA.zeros(T, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

function Tarang.allocate_like(a::CuArray, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(a))
        return CUDA.zeros(eltype(a), dims...)
    finally
        CUDA.device!(prev_device)
    end
end

"""
    copy_to_device(a::AbstractArray, target::CuArray)

Copy array `a` to GPU on the same device as target CuArray.
Ensures correct device context for multi-GPU support.
"""
function Tarang.copy_to_device(a::AbstractArray, target::CuArray)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(target))
        return CuArray(a)
    finally
        CUDA.device!(prev_device)
    end
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
        prev_device = CUDA.device()
        try
            # 1. Set source device context and download to host
            CUDA.device!(src_device)
            host_data = Array(a)
            # 2. Set destination device context and upload from host
            CUDA.device!(dst_device)
            return CuArray(host_data)
        finally
            # 3. Restore caller's device context — even on exception (e.g. OOM
            #    during the upload), so the task is not stranded on the wrong
            #    device (matches Tarang.allocate_like).
            CUDA.device!(prev_device)
        end
    end
end

# ============================================================================
# GPU-native Random Number Generation for Stochastic Forcing
# ============================================================================

"""
    Tarang._try_gpu_rand!(phases::CuArray{T}) -> Bool

Legacy compatibility hook. Stochastic forcing now uses Tarang's seeded
counter-based device kernel and no longer calls this function.
"""
Tarang._try_gpu_rand!(phases::CuArray{T}) where {T<:AbstractFloat} = false

# ============================================================================
# GPU-Native Spectral Padding/Truncation for 3/2-Rule Dealiasing
# ============================================================================

"""
Override _pad_spectral! for CuArray: uses fused GPU kernel instead of
multiple slice-based copies, reducing kernel launch overhead.
"""
function Tarang._pad_spectral!(padded::CuArray{Complex{T}}, spec_data::CuArray{Complex{T}},
                                original_shape::Tuple, padded_shape::Tuple,
                                fourier_dims::Vector{Int}) where T
    # Pin the current CUDA device to the array's device — these raw KA launches
    # (get_backend + manual kernel call) don't go through `launch!`, which is the
    # only place ensure_device! is otherwise called. Without this, a multi-GPU run
    # whose current device != spec_data's device would hit an illegal access.
    ensure_device!(Tarang.architecture(spec_data))
    fill!(padded, zero(Complex{T}))
    ndim = length(original_shape)
    backend = KernelAbstractions.get_backend(spec_data)

    if ndim == 2
        N1, N2 = original_shape
        M1, M2 = padded_shape
        kernel = pad_spectral_2d_kernel!(backend, 256)
        kernel(padded, spec_data, N1, N2, M1, M2,
               1 in fourier_dims, 2 in fourier_dims;
               ndrange=(N1, N2))
        KernelAbstractions.synchronize(backend)
    elseif ndim == 3
        N1, N2, N3 = original_shape
        M1, M2, M3 = padded_shape
        kernel = pad_spectral_3d_kernel!(backend, 256)
        kernel(padded, spec_data, N1, N2, N3, M1, M2, M3,
               1 in fourier_dims, 2 in fourier_dims, 3 in fourier_dims;
               ndrange=(N1, N2, N3))
        KernelAbstractions.synchronize(backend)
    else
        # 1D: fall back to slice-based (fast enough for 1D)
        N = original_shape[1]
        M = padded_shape[1]
        if 1 in fourier_dims
            # Fourier: split the half-spectrum (low positive freqs + high-index negatives).
            Nh = N ÷ 2
            n_neg = N - Nh - 1
            padded[1:Nh+1] .= spec_data[1:Nh+1]
            if n_neg > 0
                padded[M-n_neg+1:M] .= spec_data[N-n_neg+1:N]
            end
        else
            # Non-Fourier (e.g. Chebyshev): coefficients run low→high order, so the
            # correct zero-padded embedding is a full leading copy (`padded` is zero-filled).
            padded[1:N] .= spec_data[1:N]
        end
    end

    # Split the even-N Nyquist symmetrically across ±N/2 so the padded spectrum
    # stays Hermitian for real fields — EXACTLY as the CPU `_pad_spectral!` does
    # after its copy step. The index maps above (kernel `_gpu_padded_idx` and the
    # 1D slices) copy the full Nyquist bin to +N/2 and leave −N/2 zero; without
    # this split the padded grid picks up a spurious imaginary part that
    # contaminates every dealiased product. The helper is pure selectdim views +
    # broadcast (GPU-compatible; no-op for odd N, which has no Nyquist bin).
    Tarang._split_nyquist_symmetric!(padded, original_shape, padded_shape, fourier_dims)
    return padded
end

"""
Override _truncate_spectral! for CuArray: uses fused GPU kernel.
"""
function Tarang._truncate_spectral!(result::CuArray{Complex{T}}, padded_spec::CuArray{Complex{T}},
                                     original_shape::Tuple, padded_shape::Tuple,
                                     fourier_dims::Vector{Int}) where T
    # Pin the current CUDA device to the array's device (see _pad_spectral! note).
    ensure_device!(Tarang.architecture(result))
    ndim = length(original_shape)
    backend = KernelAbstractions.get_backend(result)

    # Fold the dropped −N/2 image into the +N/2 plane (in place, BEFORE the copy)
    # along each even-N Fourier axis — EXACTLY as the CPU `_truncate_spectral!`
    # does — so the copy below picks up the FULL Nyquist coefficient instead of
    # only the +N/2 half. Pure selectdim views + broadcast (GPU-compatible;
    # no-op for odd N).
    Tarang._fold_nyquist_into_positive!(padded_spec, original_shape, padded_shape, fourier_dims)

    if ndim == 2
        N1, N2 = original_shape
        M1, M2 = padded_shape
        kernel = truncate_spectral_2d_kernel!(backend, 256)
        kernel(result, padded_spec, N1, N2, M1, M2,
               1 in fourier_dims, 2 in fourier_dims;
               ndrange=(N1, N2))
        KernelAbstractions.synchronize(backend)
    elseif ndim == 3
        N1, N2, N3 = original_shape
        M1, M2, M3 = padded_shape
        kernel = truncate_spectral_3d_kernel!(backend, 256)
        kernel(result, padded_spec, N1, N2, N3, M1, M2, M3,
               1 in fourier_dims, 2 in fourier_dims, 3 in fourier_dims;
               ndrange=(N1, N2, N3))
        KernelAbstractions.synchronize(backend)
    else
        # 1D: slice-based
        N = original_shape[1]
        M = padded_shape[1]
        if 1 in fourier_dims
            # Fourier: take the low positive freqs + the high-index negative-freq mirror.
            Nh = N ÷ 2
            n_neg = N - Nh - 1
            result[1:Nh+1] .= padded_spec[1:Nh+1]
            if n_neg > 0
                result[N-n_neg+1:N] .= padded_spec[M-n_neg+1:M]
            end
        else
            # Non-Fourier (e.g. Chebyshev): keep the leading N low-order coefficients.
            result[1:N] .= padded_spec[1:N]
        end
    end
end

# ============================================================================
# GPU MatSolvers Helper Implementations
# ============================================================================
# These extend the abstract function stubs defined in Tarang/src/tools/gpu_matsolvers.jl
# to provide CUDA-backed implementations for GPU matrix solvers.

using SparseArrays: SparseMatrixCSC

Tarang._gpu_zeros(T::Type, dims...) = CUDA.zeros(T, dims...)
Tarang._gpu_array(data::AbstractArray, T::Type) = CuVector{T}(data)
Tarang._gpu_array(data::AbstractMatrix, T::Type) = CuMatrix{T}(data)
Tarang._gpu_sparse_csr(A::SparseMatrixCSC, T::Type) = CUDA.CUSPARSE.CuSparseMatrixCSR(SparseMatrixCSC{T, Int32}(A))
Tarang._gpu_axpy!(α, x, y) = CUDA.axpy!(α, x, y)
Tarang._is_gpu_array(a::CuArray) = true
Tarang._is_gpu_array(::Any) = false
Tarang._gpu_ilu0(A_csr) = CUDA.CUSPARSE.ilu02(A_csr)
Tarang._gpu_ic0(A_csr) = CUDA.CUSPARSE.ic02(A_csr)
