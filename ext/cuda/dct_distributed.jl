# ============================================================================
# Distributed DCT for Multi-GPU Chebyshev Transforms
# ============================================================================

"""
    DistributedDCTPlan{T}

Plan for distributed 3D DCT across multiple GPUs using pencil decomposition.

Uses memory-efficient optimized DCT (R2C FFT based) for local transforms
and NCCL all-to-all for pencil transposes.

Transform sequence (forward, starting from Z-pencil):
1. DCT in Z (local, Z is full)
2. Transpose Z→Y
3. DCT in Y (local, Y is full)
4. Transpose Y→X
5. DCT in X (local, X is full)
Result: spectral coefficients in X-pencil layout
"""
struct DistributedDCTPlan{T}
    # Pencil decomposition
    pencil::PencilDecomposition

    # Local optimized DCT plans for each dimension
    local_dct_plans::NTuple{3, OptimizedGPUDCTPlan{T}}

    # NCCL transpose buffer
    transpose_buffer::NCCLTransposeBuffer{T}

    # Work arrays for local DCT operations
    work_arrays::Vector{CuArray{T, 3}}
end

"""
    DistributedDCTPlan(pencil::PencilDecomposition, T::Type)

Create a distributed DCT plan for the given pencil decomposition.

# Arguments
- `pencil`: Pencil decomposition describing the domain layout
- `T`: Element type for the transform (e.g., Float64, Float32)

# Returns
- `DistributedDCTPlan{T}`: Plan for distributed DCT operations

# Example
```julia
pencil = PencilDecomposition(global_shape, proc_grid, rank, comm)
plan = DistributedDCTPlan(pencil, Float64)
```
"""
function DistributedDCTPlan(pencil::PencilDecomposition, ::Type{T}) where T
    # Preserve current device instead of resetting to device 0
    arch = GPU(device_id = CUDA.deviceid())

    # Create local DCT plans for each dimension
    Nx, Ny, Nz = pencil.global_shape

    # When in X-pencil, transform along X (full dimension)
    plan_x = plan_optimized_gpu_dct(arch, Nx, T)
    # When in Y-pencil, transform along Y (full dimension)
    plan_y = plan_optimized_gpu_dct(arch, Ny, T)
    # When in Z-pencil, transform along Z (full dimension)
    plan_z = plan_optimized_gpu_dct(arch, Nz, T)

    local_plans = (plan_x, plan_y, plan_z)

    # Create transpose buffer
    transpose_buffer = NCCLTransposeBuffer(pencil, T)

    # Allocate work arrays for each pencil orientation
    work_arrays = [
        CUDA.zeros(T, pencil.x_pencil_shape...),
        CUDA.zeros(T, pencil.y_pencil_shape...),
        CUDA.zeros(T, pencil.z_pencil_shape...)
    ]

    return DistributedDCTPlan{T}(pencil, local_plans, transpose_buffer, work_arrays)
end

# ============================================================================
# FFT-based Batched DCT for 3D Arrays
# ============================================================================

# Cache for batched R2C / C2R plans (keyed by device, shape, dim)
const _BATCHED_RFFT_CACHE = Dict{Tuple{Int, Tuple, Int}, Any}()
const _BATCHED_IRFFT_CACHE = Dict{Tuple{Int, Tuple, Int, Int}, Any}()
const _BATCHED_DCT_CACHE_LOCK = ReentrantLock()

function _get_batched_rfft_plan(shape::Tuple, dim::Int, ::Type{T}) where T
    device_id = CUDA.deviceid()
    key = (device_id, shape, dim)
    lock(_BATCHED_DCT_CACHE_LOCK)
    try
        if !haskey(_BATCHED_RFFT_CACHE, key)
            dummy = CUDA.zeros(T, shape...)
            _BATCHED_RFFT_CACHE[key] = CUFFT.plan_rfft(dummy, (dim,))
        end
        return _BATCHED_RFFT_CACHE[key]
    finally
        unlock(_BATCHED_DCT_CACHE_LOCK)
    end
end

function _get_batched_irfft_plan(shape::Tuple, dim::Int, n_out::Int, ::Type{T}) where T
    device_id = CUDA.deviceid()
    key = (device_id, shape, dim, n_out)
    lock(_BATCHED_DCT_CACHE_LOCK)
    try
        if !haskey(_BATCHED_IRFFT_CACHE, key)
            dummy = CUDA.zeros(Complex{T}, shape...)
            _BATCHED_IRFFT_CACHE[key] = CUFFT.plan_irfft(dummy, n_out, (dim,))
        end
        return _BATCHED_IRFFT_CACHE[key]
    finally
        unlock(_BATCHED_DCT_CACHE_LOCK)
    end
end

"""
3D forward twiddle kernel along dimension 1.
Applies twiddle factors to R2C FFT output along dim 1 to produce DCT coefficients.
Each thread handles one (k_freq, j, k_z) element of the FFT output.
"""
@kernel function twiddle_3d_dim1_forward_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                  scale_zero, scale_pos, N, Ny, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    total = half_N_plus1 * Ny * Nz
    if idx > total
        return
    end
    # Map to (freq, j, k)
    freq = ((idx - 1) % half_N_plus1) + 1  # 1-indexed frequency
    j = (((idx - 1) ÷ half_N_plus1) % Ny) + 1
    k = ((idx - 1) ÷ (half_N_plus1 * Ny)) + 1

    if freq <= half_N_plus1 && j <= Ny && k <= Nz
        @inbounds begin
            twiddled = twiddle[freq] * fft_out[freq, j, k]
            if freq == 1
                output[1, j, k] = real(twiddled) * scale_zero
            elseif freq == half_N_plus1
                output[freq, j, k] = real(twiddled) * scale_pos
            else
                output[freq, j, k] = real(twiddled) * scale_pos
                output[N - freq + 2, j, k] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
3D forward twiddle kernel along dimension 2.
"""
@kernel function twiddle_3d_dim2_forward_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                  scale_zero, scale_pos, N, Nx, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    total = Nx * half_N_plus1 * Nz
    if idx > total
        return
    end
    i = ((idx - 1) % Nx) + 1
    freq = (((idx - 1) ÷ Nx) % half_N_plus1) + 1
    k = ((idx - 1) ÷ (Nx * half_N_plus1)) + 1

    if i <= Nx && freq <= half_N_plus1 && k <= Nz
        @inbounds begin
            twiddled = twiddle[freq] * fft_out[i, freq, k]
            if freq == 1
                output[i, 1, k] = real(twiddled) * scale_zero
            elseif freq == half_N_plus1
                output[i, freq, k] = real(twiddled) * scale_pos
            else
                output[i, freq, k] = real(twiddled) * scale_pos
                output[i, N - freq + 2, k] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
3D forward twiddle kernel along dimension 3.
"""
@kernel function twiddle_3d_dim3_forward_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                  scale_zero, scale_pos, N, Nx, Ny)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    total = Nx * Ny * half_N_plus1
    if idx > total
        return
    end
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    freq = ((idx - 1) ÷ (Nx * Ny)) + 1

    if i <= Nx && j <= Ny && freq <= half_N_plus1
        @inbounds begin
            twiddled = twiddle[freq] * fft_out[i, j, freq]
            if freq == 1
                output[i, j, 1] = real(twiddled) * scale_zero
            elseif freq == half_N_plus1
                output[i, j, freq] = real(twiddled) * scale_pos
            else
                output[i, j, freq] = real(twiddled) * scale_pos
                output[i, j, N - freq + 2] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
3D backward (inverse) twiddle kernel along dimension 1.
Prepares complex array for C2R IFFT from DCT coefficients.
"""
@kernel function twiddle_3d_dim1_backward_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N, Ny, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    total = half_N_plus1 * Ny * Nz
    if idx > total
        return
    end
    freq = ((idx - 1) % half_N_plus1) + 1
    j = (((idx - 1) ÷ half_N_plus1) % Ny) + 1
    k = ((idx - 1) ÷ (half_N_plus1 * Ny)) + 1

    if freq <= half_N_plus1 && j <= Ny && k <= Nz
        @inbounds begin
            if freq == 1
                complex_out[1, j, k] = coeffs[1, j, k] * scale_zero * twiddle_inv[1]
            elseif freq == half_N_plus1
                complex_out[freq, j, k] = coeffs[freq, j, k] * scale_pos * twiddle_inv[freq]
            else
                sr = coeffs[freq, j, k] * scale_pos
                si = -coeffs[N - freq + 2, j, k] * scale_pos
                complex_out[freq, j, k] = Complex(sr, si) * twiddle_inv[freq]
            end
        end
    end
end

"""
3D backward (inverse) twiddle kernel along dimension 2.
"""
@kernel function twiddle_3d_dim2_backward_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N, Nx, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    total = Nx * half_N_plus1 * Nz
    if idx > total
        return
    end
    i = ((idx - 1) % Nx) + 1
    freq = (((idx - 1) ÷ Nx) % half_N_plus1) + 1
    k = ((idx - 1) ÷ (Nx * half_N_plus1)) + 1

    if i <= Nx && freq <= half_N_plus1 && k <= Nz
        @inbounds begin
            if freq == 1
                complex_out[i, 1, k] = coeffs[i, 1, k] * scale_zero * twiddle_inv[1]
            elseif freq == half_N_plus1
                complex_out[i, freq, k] = coeffs[i, freq, k] * scale_pos * twiddle_inv[freq]
            else
                sr = coeffs[i, freq, k] * scale_pos
                si = -coeffs[i, N - freq + 2, k] * scale_pos
                complex_out[i, freq, k] = Complex(sr, si) * twiddle_inv[freq]
            end
        end
    end
end

"""
3D backward (inverse) twiddle kernel along dimension 3.
"""
@kernel function twiddle_3d_dim3_backward_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N, Nx, Ny)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    total = Nx * Ny * half_N_plus1
    if idx > total
        return
    end
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    freq = ((idx - 1) ÷ (Nx * Ny)) + 1

    if i <= Nx && j <= Ny && freq <= half_N_plus1
        @inbounds begin
            if freq == 1
                complex_out[i, j, 1] = coeffs[i, j, 1] * scale_zero * twiddle_inv[1]
            elseif freq == half_N_plus1
                complex_out[i, j, freq] = coeffs[i, j, freq] * scale_pos * twiddle_inv[freq]
            else
                sr = coeffs[i, j, freq] * scale_pos
                si = -coeffs[i, j, N - freq + 2] * scale_pos
                complex_out[i, j, freq] = Complex(sr, si) * twiddle_inv[freq]
            end
        end
    end
end

"""
    local_dct_along_dim!(output, input, dct_plan, dim, direction)

Apply local 1D DCT/IDCT along specified dimension of 3D array.
Uses FFT-based O(N log N) algorithm via:
  Forward: even-odd reorder → batched R2C FFT → twiddle factors
  Backward: inverse twiddle → batched C2R IFFT → inverse reorder

# Arguments
- `output`: Output 3D array for DCT coefficients or grid values
- `input`: Input 3D array
- `dct_plan`: OptimizedGPUDCTPlan for the transform dimension
- `dim`: Dimension along which to transform (1, 2, or 3)
- `direction`: Transform direction (:forward or :backward)

# Returns
- Output array with transform applied along specified dimension
"""
function local_dct_along_dim!(output::CuArray{T, 3}, input::CuArray{T, 3},
                               dct_plan::OptimizedGPUDCTPlan{T}, dim::Int,
                               direction::Symbol) where T
    Nx, Ny, Nz = size(input)
    N = size(input, dim)
    @assert size(output) == size(input) "Output and input must have same size"

    arch = Tarang.architecture(input)
    half_N_plus1 = N ÷ 2 + 1

    forward_scale_zero = dct_plan.forward_scale_zero
    forward_scale_pos = dct_plan.forward_scale_pos
    backward_scale_zero = dct_plan.backward_scale_zero
    backward_scale_pos = dct_plan.backward_scale_pos

    if direction == :forward
        # Step 1: Even-odd reorder
        work = similar(input)
        reorder_for_dct_dim!(work, input, dim)

        # Step 2: Batched R2C FFT along dim
        rfft_plan = _get_batched_rfft_plan(size(work), dim, T)
        fft_out = rfft_plan * work

        # Step 3: Apply twiddle factors to extract DCT coefficients
        fft_total = prod(size(fft_out))
        if dim == 1
            launch!(arch, twiddle_3d_dim1_forward_kernel!, output, fft_out, dct_plan.twiddle,
                    forward_scale_zero, forward_scale_pos, N, Ny, Nz;
                    ndrange=fft_total)
        elseif dim == 2
            launch!(arch, twiddle_3d_dim2_forward_kernel!, output, fft_out, dct_plan.twiddle,
                    forward_scale_zero, forward_scale_pos, N, Nx, Nz;
                    ndrange=fft_total)
        else
            launch!(arch, twiddle_3d_dim3_forward_kernel!, output, fft_out, dct_plan.twiddle,
                    forward_scale_zero, forward_scale_pos, N, Nx, Ny;
                    ndrange=fft_total)
        end

    else  # :backward
        # Step 1: Allocate complex array for C2R input
        complex_shape = ntuple(i -> i == dim ? half_N_plus1 : size(input, i), 3)
        complex_buf = CUDA.zeros(Complex{T}, complex_shape...)
        complex_total = prod(complex_shape)

        # Step 2: Apply inverse twiddle factors
        if dim == 1
            launch!(arch, twiddle_3d_dim1_backward_kernel!, complex_buf, input, dct_plan.twiddle_inv,
                    backward_scale_zero, backward_scale_pos, N, Ny, Nz;
                    ndrange=complex_total)
        elseif dim == 2
            launch!(arch, twiddle_3d_dim2_backward_kernel!, complex_buf, input, dct_plan.twiddle_inv,
                    backward_scale_zero, backward_scale_pos, N, Nx, Nz;
                    ndrange=complex_total)
        else
            launch!(arch, twiddle_3d_dim3_backward_kernel!, complex_buf, input, dct_plan.twiddle_inv,
                    backward_scale_zero, backward_scale_pos, N, Nx, Ny;
                    ndrange=complex_total)
        end

        # Step 3: Batched C2R IFFT along dim
        irfft_plan = _get_batched_irfft_plan(complex_shape, dim, N, T)
        work = irfft_plan * complex_buf

        # Step 4: Inverse reorder
        inverse_reorder_for_dct_dim!(output, work, dim)
    end

    CUDA.synchronize()
    return output
end

# ============================================================================
# Distributed DCT Operations
# ============================================================================

"""
    distributed_forward_dct!(coeffs, data, plan::DistributedDCTPlan)

Perform forward distributed 3D DCT.

Transform order (starting from Z-pencil):
1. DCT in Z (local, Z is full)
2. Transpose Z→Y
3. DCT in Y (local, Y is full)
4. Transpose Y→X
5. DCT in X (local, X is full)

Input: grid values in Z-pencil layout
Output: spectral coefficients in X-pencil layout

# Arguments
- `coeffs`: Output array for spectral coefficients (X-pencil shape)
- `data`: Input array of grid values (Z-pencil shape)
- `plan`: DistributedDCTPlan created for this decomposition

# Returns
- `coeffs` array filled with spectral coefficients

# Example
```julia
pencil = PencilDecomposition(global_shape, proc_grid, rank, comm)
plan = DistributedDCTPlan(pencil, Float64)

data = CUDA.rand(Float64, pencil.z_pencil_shape...)
coeffs = CUDA.zeros(Float64, pencil.x_pencil_shape...)

distributed_forward_dct!(coeffs, data, plan)
```
"""
function distributed_forward_dct!(coeffs::CuArray{T, 3}, data::CuArray{T, 3},
                                   plan::DistributedDCTPlan{T}) where T
    pencil = plan.pencil

    # Ensure starting in Z-pencil
    @assert current_orientation(pencil) == :z_pencil "Must start in Z-pencil layout"
    @assert size(data) == pencil.z_pencil_shape "Data shape must match Z-pencil shape"

    # Step 1: DCT in Z (local - Z is full in Z-pencil)
    z_work = plan.work_arrays[3]
    local_dct_along_dim!(z_work, data, plan.local_dct_plans[3], 3, :forward)

    # Step 2: Transpose Z→Y
    y_data = transpose_z_to_y!(plan.transpose_buffer, z_work, pencil)

    # Step 3: DCT in Y (local - Y is now full)
    y_work = plan.work_arrays[2]
    local_dct_along_dim!(y_work, y_data, plan.local_dct_plans[2], 2, :forward)

    # Step 4: Transpose Y→X
    x_data = transpose_y_to_x!(plan.transpose_buffer, y_work, pencil)

    # Step 5: DCT in X (local - X is now full)
    @assert size(coeffs) == pencil.x_pencil_shape "Coeffs shape must match X-pencil shape"
    local_dct_along_dim!(coeffs, x_data, plan.local_dct_plans[1], 1, :forward)

    return coeffs
end

"""
    distributed_backward_dct!(data, coeffs, plan::DistributedDCTPlan)

Perform backward distributed 3D DCT (inverse transform).

Transform order (starting from X-pencil):
1. Inverse DCT in X (local)
2. Transpose X→Y
3. Inverse DCT in Y (local)
4. Transpose Y→Z
5. Inverse DCT in Z (local)

Input: spectral coefficients in X-pencil layout
Output: grid values in Z-pencil layout

# Arguments
- `data`: Output array for grid values (Z-pencil shape)
- `coeffs`: Input array of spectral coefficients (X-pencil shape)
- `plan`: DistributedDCTPlan created for this decomposition

# Returns
- `data` array filled with grid values

# Example
```julia
pencil = PencilDecomposition(global_shape, proc_grid, rank, comm)
plan = DistributedDCTPlan(pencil, Float64)

# After forward transform, coeffs is in X-pencil layout
set_orientation!(pencil, :x_pencil)
data = CUDA.zeros(Float64, pencil.z_pencil_shape...)

distributed_backward_dct!(data, coeffs, plan)
```
"""
function distributed_backward_dct!(data::CuArray{T, 3}, coeffs::CuArray{T, 3},
                                    plan::DistributedDCTPlan{T}) where T
    pencil = plan.pencil

    # Ensure starting in X-pencil (where coefficients live)
    @assert current_orientation(pencil) == :x_pencil "Must start in X-pencil layout"
    @assert size(coeffs) == pencil.x_pencil_shape "Coeffs shape must match X-pencil shape"

    # Step 1: Inverse DCT in X (local)
    x_work = plan.work_arrays[1]
    local_dct_along_dim!(x_work, coeffs, plan.local_dct_plans[1], 1, :backward)

    # Step 2: Transpose X→Y
    y_data = transpose_x_to_y!(plan.transpose_buffer, x_work, pencil)

    # Step 3: Inverse DCT in Y (local)
    y_work = plan.work_arrays[2]
    local_dct_along_dim!(y_work, y_data, plan.local_dct_plans[2], 2, :backward)

    # Step 4: Transpose Y→Z
    z_data = transpose_y_to_z!(plan.transpose_buffer, y_work, pencil)

    # Step 5: Inverse DCT in Z (local)
    @assert size(data) == pencil.z_pencil_shape "Data shape must match Z-pencil shape"
    local_dct_along_dim!(data, z_data, plan.local_dct_plans[3], 3, :backward)

    return data
end

"""
    finalize_distributed_dct_plan!(plan::DistributedDCTPlan)

Clean up resources used by the distributed DCT plan.

This function releases NCCL sub-communicators and allows
work arrays to be garbage collected.

# Arguments
- `plan`: DistributedDCTPlan to finalize

# Example
```julia
plan = DistributedDCTPlan(pencil, Float64)
# ... use plan for transforms ...
finalize_distributed_dct_plan!(plan)
```
"""
function finalize_distributed_dct_plan!(plan::DistributedDCTPlan)
    finalize_nccl_transpose!(plan.transpose_buffer)
    # Work arrays will be garbage collected
    return nothing
end
