# ============================================================================
# GPU DCT Transforms for Chebyshev Basis (using FFT-based approach)
# ============================================================================

"""
    GPUDCTPlan

GPU plan for Discrete Cosine Transform (DCT) used in Chebyshev spectral methods.
Uses FFT-based approach: DCT can be computed via FFT with O(N) pre/post processing.

DCT-II (forward): coefficients from grid values
DCT-III (backward): grid values from coefficients
"""
struct GPUDCTPlan{T}
    size::Int
    axis::Int
    # Precomputed twiddle factors for FFT-based DCT
    twiddle_forward::CuVector{Complex{T}}
    twiddle_backward::CuVector{Complex{T}}
    # Scaling factors (Tarang convention)
    forward_scale_zero::T
    forward_scale_pos::T
    backward_scale_zero::T
    backward_scale_pos::T
    # Work arrays
    work_complex::CuVector{Complex{T}}
    # FFT plan for the extended array
    fft_plan::Any
    ifft_plan::Any
end

"""
    plan_gpu_dct(arch::GPU{CuDevice}, n::Int, T::Type, axis::Int)

Create a GPU DCT plan for Chebyshev transforms along specified axis.

Uses the FFT-based DCT algorithm:
- Forward (DCT-II): Y_k = 2 * Σ_{j=0}^{N-1} X_j * cos(π*k*(2j+1)/(2N))
- Backward (DCT-III): X_j = X_0 + 2 * Σ_{k=1}^{N-1} X_k * cos(π*k*(2j+1)/(2N))

The FFT-based approach uses a 2N-point FFT on symmetrically extended data.
"""
function plan_gpu_dct(arch::GPU{CuDevice}, n::Int, T::Type, axis::Int)
    ensure_device!(arch)

    real_T = T <: Complex ? real(T) : T
    complex_T = Complex{real_T}

    # Twiddle factors for FFT-based DCT
    # Forward: exp(-i*π*k/(2N)) for k = 0..N-1
    twiddle_forward = CuArray(complex_T[exp(complex_T(-im * π * k / (2 * n))) for k in 0:n-1])

    # Backward: exp(i*π*k/(2N)) for k = 0..N-1
    twiddle_backward = CuArray(complex_T[exp(complex_T(im * π * k / (2 * n))) for k in 0:n-1])

    # Scaling factors following Tarang FastCosineTransform convention.
    # Forward: DC gets 1/(2N), AC gets 1/N.
    # Backward: DCT-III formula has an explicit factor of 2 on AC terms (see kernels),
    # so backward_scale_pos=0.5 cancels it, giving effective weight 1.0 on AC terms.
    # This makes backward(forward(x)) == x (the pair forms an exact inverse).
    forward_scale_zero = real_T(1.0 / n / 2.0)   # DC component: 1/(2N)
    forward_scale_pos = real_T(1.0 / n)          # AC components: 1/N
    backward_scale_zero = real_T(1.0)            # DC component: 1.0
    backward_scale_pos = real_T(0.5)             # AC components: 0.5 (×2 in kernel = 1.0)

    # Work array for 2N-point FFT
    work_complex = CUDA.zeros(complex_T, 2 * n)

    # FFT plans for the extended array
    fft_plan = CUFFT.plan_fft(work_complex)
    ifft_plan = CUFFT.plan_ifft(work_complex)

    return GPUDCTPlan{real_T}(
        n, axis,
        twiddle_forward, twiddle_backward,
        forward_scale_zero, forward_scale_pos,
        backward_scale_zero, backward_scale_pos,
        work_complex,
        fft_plan, ifft_plan
    )
end

# Fallback for generic GPU
plan_gpu_dct(arch::GPU, n::Int, T::Type, axis::Int) = plan_gpu_dct(GPU{CuDevice}(CUDA.device()), n, T, axis)

"""
DCT forward scaling kernel: applies Tarang normalization to DCT coefficients.
"""
@kernel function dct_forward_scale_kernel!(output, @Const(input), scale_zero, scale_pos, n)
    i = @index(Global)
    if i <= n
        @inbounds if i == 1
            output[i] = input[i] * scale_zero
        else
            output[i] = input[i] * scale_pos
        end
    end
end

"""
DCT backward pre-scale kernel: applies scaling before inverse DCT.
"""
@kernel function dct_backward_prescale_kernel!(output, @Const(input), scale_zero, scale_pos, n)
    i = @index(Global)
    if i <= n
        @inbounds if i == 1
            output[i] = input[i] * scale_zero
        else
            output[i] = input[i] * scale_pos
        end
    end
end

# ============================================================================
# DCT Reorder Kernels (Even-Odd Interleaving for Memory-Efficient DCT)
# ============================================================================
#
# These kernels implement the even-odd reordering pattern used in memory-efficient
# DCT implementations that avoid the 2N symmetric extension.
#
# The reorder pattern (0-indexed):
#   work[k] = x[2k] for k=0..N/2-1 (even indices to first half)
#   work[N-1-k] = x[2k+1] for k=0..N/2-1 (odd indices reversed to second half)
#
# Example: [1,2,3,4,5,6,7,8] -> [1,3,5,7,8,6,4,2]
#
# This reduces memory usage by ~60% compared to the symmetric extension approach.
# ============================================================================

"""
1D reorder kernel for DCT: reorders input array using even-odd interleaving pattern.
Each thread handles one element.
"""
@kernel function reorder_for_dct_kernel!(work, @Const(x), N)
    k = @index(Global) - 1  # Convert to 0-indexed
    half_N = N ÷ 2

    if k < half_N
        @inbounds begin
            # Even indices (0, 2, 4, ...) go to first half
            work[k + 1] = x[2k + 1]  # 1-indexed Julia arrays
            # Odd indices (1, 3, 5, ...) go to second half reversed
            work[N - k] = x[2k + 2]  # 1-indexed Julia arrays
        end
    end
end

"""
1D inverse reorder kernel for DCT: recovers original array from reordered form.
Each thread handles one element.
"""
@kernel function inverse_reorder_for_dct_kernel!(x, @Const(work), N)
    k = @index(Global) - 1  # Convert to 0-indexed
    half_N = N ÷ 2

    if k < half_N
        @inbounds begin
            # Reconstruct even indices from first half
            x[2k + 1] = work[k + 1]  # 1-indexed Julia arrays
            # Reconstruct odd indices from second half (reversed)
            x[2k + 2] = work[N - k]  # 1-indexed Julia arrays
        end
    end
end

"""
    reorder_for_dct!(work::CuVector{T}, x::CuVector{T}) where T

Reorder a 1D GPU array for memory-efficient DCT using even-odd interleaving.

The reorder pattern (0-indexed):
- work[k] = x[2k] for k=0..N/2-1 (even indices to first half)
- work[N-1-k] = x[2k+1] for k=0..N/2-1 (odd indices reversed to second half)

Example: [1,2,3,4,5,6,7,8] -> [1,3,5,7,8,6,4,2]
"""
function reorder_for_dct!(work::CuVector{T}, x::CuVector{T}) where T
    N = length(x)
    @assert length(work) == N "work and x must have the same length"
    @assert iseven(N) "N must be even for reorder"

    arch = Tarang.architecture(x)
    launch!(arch, reorder_for_dct_kernel!, work, x, N; ndrange=N÷2)

    return work
end

"""
    inverse_reorder_for_dct!(x::CuVector{T}, work::CuVector{T}) where T

Inverse reorder a 1D GPU array from reordered form back to original ordering.

This is the inverse of `reorder_for_dct!`.
"""
function inverse_reorder_for_dct!(x::CuVector{T}, work::CuVector{T}) where T
    N = length(work)
    @assert length(x) == N "x and work must have the same length"
    @assert iseven(N) "N must be even for reorder"

    arch = Tarang.architecture(work)
    launch!(arch, inverse_reorder_for_dct_kernel!, x, work, N; ndrange=N÷2)

    return x
end

"""
3D reorder kernel for DCT along dimension 1.
Each thread handles one (j, k) pair - a "fiber" along dimension 1.
"""
@kernel function reorder_for_dct_3d_dim1_kernel!(work, @Const(x), Nx, Ny, Nz)
    idx = @index(Global)
    # Map linear index to (j, k) pair (1-indexed)
    j = ((idx - 1) % Ny) + 1
    k = ((idx - 1) ÷ Ny) + 1

    half_Nx = Nx ÷ 2

    if j <= Ny && k <= Nz
        @inbounds for m in 0:(half_Nx - 1)
            # Even indices to first half
            work[m + 1, j, k] = x[2m + 1, j, k]
            # Odd indices to second half reversed
            work[Nx - m, j, k] = x[2m + 2, j, k]
        end
    end
end

"""
3D reorder kernel for DCT along dimension 2.
Each thread handles one (i, k) pair - a "fiber" along dimension 2.
"""
@kernel function reorder_for_dct_3d_dim2_kernel!(work, @Const(x), Nx, Ny, Nz)
    idx = @index(Global)
    # Map linear index to (i, k) pair (1-indexed)
    i = ((idx - 1) % Nx) + 1
    k = ((idx - 1) ÷ Nx) + 1

    half_Ny = Ny ÷ 2

    if i <= Nx && k <= Nz
        @inbounds for m in 0:(half_Ny - 1)
            # Even indices to first half
            work[i, m + 1, k] = x[i, 2m + 1, k]
            # Odd indices to second half reversed
            work[i, Ny - m, k] = x[i, 2m + 2, k]
        end
    end
end

"""
3D reorder kernel for DCT along dimension 3.
Each thread handles one (i, j) pair - a "fiber" along dimension 3.
"""
@kernel function reorder_for_dct_3d_dim3_kernel!(work, @Const(x), Nx, Ny, Nz)
    idx = @index(Global)
    # Map linear index to (i, j) pair (1-indexed)
    i = ((idx - 1) % Nx) + 1
    j = ((idx - 1) ÷ Nx) + 1

    half_Nz = Nz ÷ 2

    if i <= Nx && j <= Ny
        @inbounds for m in 0:(half_Nz - 1)
            # Even indices to first half
            work[i, j, m + 1] = x[i, j, 2m + 1]
            # Odd indices to second half reversed
            work[i, j, Nz - m] = x[i, j, 2m + 2]
        end
    end
end

"""
3D inverse reorder kernel for DCT along dimension 1.
Each thread handles one (j, k) pair - a "fiber" along dimension 1.
"""
@kernel function inverse_reorder_for_dct_3d_dim1_kernel!(x, @Const(work), Nx, Ny, Nz)
    idx = @index(Global)
    j = ((idx - 1) % Ny) + 1
    k = ((idx - 1) ÷ Ny) + 1

    half_Nx = Nx ÷ 2

    if j <= Ny && k <= Nz
        @inbounds for m in 0:(half_Nx - 1)
            # Reconstruct even indices from first half
            x[2m + 1, j, k] = work[m + 1, j, k]
            # Reconstruct odd indices from second half (reversed)
            x[2m + 2, j, k] = work[Nx - m, j, k]
        end
    end
end

"""
3D inverse reorder kernel for DCT along dimension 2.
Each thread handles one (i, k) pair - a "fiber" along dimension 2.
"""
@kernel function inverse_reorder_for_dct_3d_dim2_kernel!(x, @Const(work), Nx, Ny, Nz)
    idx = @index(Global)
    i = ((idx - 1) % Nx) + 1
    k = ((idx - 1) ÷ Nx) + 1

    half_Ny = Ny ÷ 2

    if i <= Nx && k <= Nz
        @inbounds for m in 0:(half_Ny - 1)
            # Reconstruct even indices from first half
            x[i, 2m + 1, k] = work[i, m + 1, k]
            # Reconstruct odd indices from second half (reversed)
            x[i, 2m + 2, k] = work[i, Ny - m, k]
        end
    end
end

"""
3D inverse reorder kernel for DCT along dimension 3.
Each thread handles one (i, j) pair - a "fiber" along dimension 3.
"""
@kernel function inverse_reorder_for_dct_3d_dim3_kernel!(x, @Const(work), Nx, Ny, Nz)
    idx = @index(Global)
    i = ((idx - 1) % Nx) + 1
    j = ((idx - 1) ÷ Nx) + 1

    half_Nz = Nz ÷ 2

    if i <= Nx && j <= Ny
        @inbounds for m in 0:(half_Nz - 1)
            # Reconstruct even indices from first half
            x[i, j, 2m + 1] = work[i, j, m + 1]
            # Reconstruct odd indices from second half (reversed)
            x[i, j, 2m + 2] = work[i, j, Nz - m]
        end
    end
end

"""
    reorder_for_dct_dim!(work::CuArray{T,3}, x::CuArray{T,3}, dim::Int) where T

Reorder a 3D GPU array for memory-efficient DCT along the specified dimension.

# Arguments
- `work`: Output array (same size as `x`)
- `x`: Input array
- `dim`: Dimension along which to reorder (1, 2, or 3)
"""
function reorder_for_dct_dim!(work::CuArray{T,3}, x::CuArray{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(x)
    @assert size(work) == size(x) "work and x must have the same size"
    @assert 1 <= dim <= 3 "dim must be 1, 2, or 3"

    if dim == 1
        @assert iseven(Nx) "Nx must be even for dim=1 reorder"
    elseif dim == 2
        @assert iseven(Ny) "Ny must be even for dim=2 reorder"
    else
        @assert iseven(Nz) "Nz must be even for dim=3 reorder"
    end

    arch = Tarang.architecture(x)

    if dim == 1
        ndrange = Ny * Nz
        launch!(arch, reorder_for_dct_3d_dim1_kernel!, work, x, Nx, Ny, Nz; ndrange=ndrange)
    elseif dim == 2
        ndrange = Nx * Nz
        launch!(arch, reorder_for_dct_3d_dim2_kernel!, work, x, Nx, Ny, Nz; ndrange=ndrange)
    else
        ndrange = Nx * Ny
        launch!(arch, reorder_for_dct_3d_dim3_kernel!, work, x, Nx, Ny, Nz; ndrange=ndrange)
    end

    return work
end

"""
    inverse_reorder_for_dct_dim!(x::CuArray{T,3}, work::CuArray{T,3}, dim::Int) where T

Inverse reorder a 3D GPU array from reordered form back to original ordering.

This is the inverse of `reorder_for_dct_dim!`.

# Arguments
- `x`: Output array (original ordering)
- `work`: Input array (reordered form)
- `dim`: Dimension along which to inverse reorder (1, 2, or 3)
"""
function inverse_reorder_for_dct_dim!(x::CuArray{T,3}, work::CuArray{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(work)
    @assert size(x) == size(work) "x and work must have the same size"
    @assert 1 <= dim <= 3 "dim must be 1, 2, or 3"

    if dim == 1
        @assert iseven(Nx) "Nx must be even for dim=1 reorder"
    elseif dim == 2
        @assert iseven(Ny) "Ny must be even for dim=2 reorder"
    else
        @assert iseven(Nz) "Nz must be even for dim=3 reorder"
    end

    arch = Tarang.architecture(work)

    if dim == 1
        ndrange = Ny * Nz
        launch!(arch, inverse_reorder_for_dct_3d_dim1_kernel!, x, work, Nx, Ny, Nz; ndrange=ndrange)
    elseif dim == 2
        ndrange = Nx * Nz
        launch!(arch, inverse_reorder_for_dct_3d_dim2_kernel!, x, work, Nx, Ny, Nz; ndrange=ndrange)
    else
        ndrange = Nx * Ny
        launch!(arch, inverse_reorder_for_dct_3d_dim3_kernel!, x, work, Nx, Ny, Nz; ndrange=ndrange)
    end

    return x
end

# ============================================================================
# Optimized DCT Plan (R2C FFT based, no 2N extension)
# ============================================================================

"""
    OptimizedGPUDCTPlan

Memory-efficient GPU DCT plan using R2C FFT instead of 2N symmetric extension.

This implementation uses the fact that DCT-II can be computed via:
1. Even-odd reordering of input
2. Real-to-complex FFT of length N
3. Twiddle factor multiplication

Memory usage: N real + (N/2+1) complex ≈ 2N floats
vs standard: 2N complex = 4N floats
Savings: ~50-60%

The algorithm is based on the relation between DCT-II and DFT through
the even-odd reordering pattern. After reordering x[n] into y[n] where:
- y[k] = x[2k] for k = 0, ..., N/2-1
- y[N-1-k] = x[2k+1] for k = 0, ..., N/2-1

The DCT coefficients are obtained by:
X[k] = Re(W[k] * FFT(y)[k])

where W[k] = 2 * exp(-i*π*k/(2N)) are the twiddle factors.
"""
struct OptimizedGPUDCTPlan{T}
    size::Int
    # Work arrays (shared — guard with _dct_plan_lock for thread safety)
    work_real::CuVector{T}
    work_complex::CuVector{Complex{T}}
    # Twiddle factors: 2 * exp(-iπk/(2N)) for k=0..N/2 (forward)
    twiddle::CuVector{Complex{T}}
    # Inverse twiddle factors for backward transform
    twiddle_inv::CuVector{Complex{T}}
    # FFT plans
    rfft_plan::Any  # Real-to-complex FFT
    irfft_plan::Any # Complex-to-real IFFT
    # Scaling factors (Tarang convention)
    forward_scale_zero::T
    forward_scale_pos::T
    backward_scale_zero::T
    backward_scale_pos::T
end

# Per-plan lock to prevent concurrent mutation of shared work arrays
const _dct_plan_locks = Dict{UInt, ReentrantLock}()
const _dct_plan_locks_lock = ReentrantLock()

function _get_plan_lock(plan::OptimizedGPUDCTPlan)
    id = objectid(plan)
    lock(_dct_plan_locks_lock)
    try
        get!(() -> ReentrantLock(), _dct_plan_locks, id)
    finally
        unlock(_dct_plan_locks_lock)
    end
end

"""
    plan_optimized_gpu_dct(arch::GPU{CuDevice}, n::Int, T::Type)

Create a memory-efficient GPU DCT plan using R2C FFT.

This plan uses approximately 50-60% less memory than the standard
2N symmetric extension approach while maintaining the same accuracy.

# Arguments
- `arch`: GPU architecture
- `n`: Size of the transform (must be even)
- `T`: Element type (Float32 or Float64)

# Returns
- `OptimizedGPUDCTPlan{T}`: Plan for optimized DCT operations
"""
function plan_optimized_gpu_dct(arch::GPU{CuDevice}, n::Int, T::Type)
    ensure_device!(arch)

    @assert iseven(n) "N must be even for optimized DCT"

    real_T = T <: Complex ? real(T) : T
    complex_T = Complex{real_T}

    # Work arrays: N real for reordered data, N/2+1 complex for R2C output
    work_real = CUDA.zeros(real_T, n)
    work_complex = CUDA.zeros(complex_T, n ÷ 2 + 1)

    # Twiddle factors for forward DCT: 2 * exp(-i*π*k/(2N)) for k=0..N/2
    # The factor of 2 is part of the DCT-II formula
    # Note: We only need N/2+1 twiddle factors for R2C output
    twiddle = CuArray(complex_T[2 * exp(complex_T(-im * π * k / (2 * n))) for k in 0:(n÷2)])

    # Inverse twiddle factors for backward DCT: exp(i*π*k/(2N)) / 2 for k=0..N/2
    # The division by 2 cancels the factor in forward twiddle
    twiddle_inv = CuArray(complex_T[exp(complex_T(im * π * k / (2 * n))) / 2 for k in 0:(n÷2)])

    # FFT plans: R2C for forward, C2R for backward
    rfft_plan = CUFFT.plan_rfft(work_real)
    irfft_plan = CUFFT.plan_irfft(work_complex, n)

    # Scaling factors following Tarang FastCosineTransform convention
    # Forward: DC gets 1/(2N), AC gets 1/N
    # Backward: DC gets 1.0, AC gets 0.5 (which multiplied by 2 in formula gives 1.0)
    forward_scale_zero = real_T(1.0 / n / 2.0)   # DC component: 1/(2N)
    forward_scale_pos = real_T(1.0 / n)          # AC components: 1/N
    backward_scale_zero = real_T(1.0)            # DC component: 1.0
    backward_scale_pos = real_T(0.5)             # AC components: 0.5 (×2 in kernel = 1.0)

    return OptimizedGPUDCTPlan{real_T}(
        n,
        work_real, work_complex,
        twiddle, twiddle_inv,
        rfft_plan, irfft_plan,
        forward_scale_zero, forward_scale_pos,
        backward_scale_zero, backward_scale_pos
    )
end

# Fallback for generic GPU
plan_optimized_gpu_dct(arch::GPU, n::Int, T::Type) =
    plan_optimized_gpu_dct(GPU{CuDevice}(CUDA.device()), n, T)

"""
Kernel to apply twiddle factors after R2C FFT and extract DCT coefficients.

For the optimized DCT algorithm:
- DCT[k] = Re(twiddle[k] * FFT[k]) for k = 0, ..., N/2
- DCT[N-k] = -Im(twiddle[k] * FFT[k]) for k = 1, ..., N/2-1

This kernel handles both halves of the DCT output.
"""
@kernel function optimized_dct_twiddle_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                scale_zero, scale_pos, N)
    k = @index(Global)  # 1-indexed, k = 1 to N/2+1

    if k <= N ÷ 2 + 1
        @inbounds begin
            # Apply twiddle factor
            twiddled = twiddle[k] * fft_out[k]

            # Extract real part for DCT[k-1] (0-indexed: k-1)
            if k == 1
                # DC component (k=0)
                output[1] = real(twiddled) * scale_zero
            elseif k == N ÷ 2 + 1
                # Nyquist component (k=N/2) - only contributes to one output
                output[k] = real(twiddled) * scale_pos
            else
                # Regular component: contributes to both k and N-k
                output[k] = real(twiddled) * scale_pos
                # DCT[N-k+2] in 1-indexed = DCT[N-(k-1)] in 0-indexed
                output[N - k + 2] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
Kernel to prepare complex array for C2R IFFT in backward DCT.

For the inverse optimized DCT algorithm:
- complex[k] = (DCT[k] + i*DCT[N-k]) * twiddle_inv[k] for k = 0, ..., N/2

This sets up the complex array so that C2R IFFT produces the reordered array.
"""
@kernel function optimized_dct_inv_twiddle_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N)
    k = @index(Global)  # 1-indexed, k = 1 to N/2+1

    if k <= N ÷ 2 + 1
        @inbounds begin
            if k == 1
                # DC component (k=0): purely real, no imaginary part from N-0=N
                scaled_coeff = coeffs[1] * scale_zero
                complex_out[1] = scaled_coeff * twiddle_inv[1]
            elseif k == N ÷ 2 + 1
                # Nyquist component (k=N/2): purely real
                scaled_coeff = coeffs[k] * scale_pos
                complex_out[k] = scaled_coeff * twiddle_inv[k]
            else
                # Regular component: combine DCT[k-1] and DCT[N-(k-1)]
                # In 1-indexed: combine coeffs[k] and coeffs[N-k+2]
                scaled_real = coeffs[k] * scale_pos
                scaled_imag = -coeffs[N - k + 2] * scale_pos  # Negative because of twiddle relation
                complex_val = Complex(scaled_real, scaled_imag)
                complex_out[k] = complex_val * twiddle_inv[k]
            end
        end
    end
end

"""
    optimized_forward_dct_1d!(output::CuVector{T}, input::CuVector{T}, plan::OptimizedGPUDCTPlan{T})

Execute memory-efficient forward DCT (DCT-II) using R2C FFT.

Algorithm:
1. Reorder input using even-odd interleaving pattern
2. Apply R2C FFT to reordered data
3. Apply twiddle factors and extract real/imaginary parts for DCT coefficients

# Arguments
- `output`: Output array for DCT coefficients (length N)
- `input`: Input array of grid values (length N)
- `plan`: Optimized DCT plan created by `plan_optimized_gpu_dct`
"""
function optimized_forward_dct_1d!(output::CuVector{T}, input::CuVector{T},
                                    plan::OptimizedGPUDCTPlan{T}) where T
    n = plan.size
    @assert length(input) == n "Input size must match plan size"
    @assert length(output) == n "Output size must match plan size"

    lk = _get_plan_lock(plan)
    lock(lk)
    try
        # Step 1: Reorder input using even-odd interleaving
        reorder_for_dct!(plan.work_real, input)

        # Step 2: Apply R2C FFT
        mul!(plan.work_complex, plan.rfft_plan, plan.work_real)

        # Step 3: Apply twiddle factors and extract DCT coefficients
        arch = Tarang.architecture(input)
        launch!(arch, optimized_dct_twiddle_kernel!, output, plan.work_complex, plan.twiddle,
                plan.forward_scale_zero, plan.forward_scale_pos, n;
                ndrange=n÷2+1)
    finally
        unlock(lk)
    end

    return output
end

"""
    optimized_backward_dct_1d!(output::CuVector{T}, coeffs::CuVector{T}, plan::OptimizedGPUDCTPlan{T})

Execute memory-efficient backward DCT (DCT-III) using C2R IFFT.

Algorithm:
1. Apply inverse twiddle factors to prepare complex array
2. Apply C2R IFFT
3. Inverse reorder to recover original grid ordering

# Arguments
- `output`: Output array for grid values (length N)
- `coeffs`: Input array of DCT coefficients (length N)
- `plan`: Optimized DCT plan created by `plan_optimized_gpu_dct`
"""
function optimized_backward_dct_1d!(output::CuVector{T}, coeffs::CuVector{T},
                                     plan::OptimizedGPUDCTPlan{T}) where T
    n = plan.size
    @assert length(coeffs) == n "Coefficients size must match plan size"
    @assert length(output) == n "Output size must match plan size"

    lk = _get_plan_lock(plan)
    lock(lk)
    try
        # Step 1: Apply inverse twiddle factors
        arch = Tarang.architecture(coeffs)
        launch!(arch, optimized_dct_inv_twiddle_kernel!, plan.work_complex, coeffs, plan.twiddle_inv,
                plan.backward_scale_zero, plan.backward_scale_pos, n;
                ndrange=n÷2+1)

        # Step 2: Apply C2R IFFT
        mul!(plan.work_real, plan.irfft_plan, plan.work_complex)

        # Step 3: Inverse reorder to recover original ordering
        inverse_reorder_for_dct!(output, plan.work_real)
    finally
        unlock(lk)
    end

    return output
end

"""
    gpu_forward_dct_1d!(output::CuVector, input::CuVector, plan::GPUDCTPlan)

Execute forward DCT (DCT-II) on GPU for 1D data.
Transforms grid values to Chebyshev coefficients.
"""
function gpu_forward_dct_1d!(output::CuVector{T}, input::CuVector{T}, plan::GPUDCTPlan{T}) where T
    n = plan.size
    @assert length(input) == n "Input size must match plan size"
    @assert length(output) == n "Output size must match plan size"

    # FFT-based DCT-II algorithm:
    # 1. Create symmetric extension: [x_0, x_1, ..., x_{N-1}, x_{N-1}, ..., x_1, x_0] (length 2N)
    # 2. Compute FFT
    # 3. Extract first N elements and apply twiddle factors
    # 4. Take real part and scale

    # Use local work buffers to avoid shared-state races between concurrent calls
    # Separate input/output buffers required: CUFFT out-of-place plans do not support aliased mul!
    work_in = CUDA.zeros(Complex{T}, 2 * n)
    work_out = CUDA.zeros(Complex{T}, 2 * n)

    # Step 1: Create symmetric extension (vectorized, no scalar indexing)
    copyto!(view(work_in, 1:n), Complex{T}.(input))
    # reverse() is well-supported in GPUArrays.jl for contiguous CuArrays
    copyto!(view(work_in, n+1:2*n), Complex{T}.(reverse(input)))

    # Step 2: FFT of extended array (non-aliased: work_in → work_out)
    mul!(work_out, plan.fft_plan, work_in)

    # Step 3: Extract and apply twiddle factors, then scale (vectorized)
    # Compute real part of (work_out[1:N] .* twiddle_forward)
    raw_coeffs = real.(view(work_out, 1:n) .* plan.twiddle_forward)
    # Apply scaling: first element uses scale_zero, rest use scale_pos
    output .= raw_coeffs .* plan.forward_scale_pos
    view(output, 1:1) .= view(raw_coeffs, 1:1) .* plan.forward_scale_zero

    return output
end

"""
    gpu_backward_dct_1d!(output::CuVector, input::CuVector, plan::GPUDCTPlan)

Execute backward DCT (DCT-III) on GPU for 1D data.
Transforms Chebyshev coefficients to grid values.
"""
function gpu_backward_dct_1d!(output::CuVector{T}, input::CuVector{T}, plan::GPUDCTPlan{T}) where T
    n = plan.size
    @assert length(input) == n "Input size must match plan size"
    @assert length(output) == n "Output size must match plan size"

    # FFT-based DCT-III algorithm (inverse of DCT-II):
    # 1. Pre-scale coefficients
    # 2. Apply inverse twiddle factors
    # 3. Create Hermitian-symmetric extension
    # 4. Compute IFFT
    # 5. Extract first N real values

    # Use local work buffers to avoid shared-state races between concurrent calls
    # Separate input/output buffers required: CUFFT out-of-place plans do not support aliased mul!
    work_in = CUDA.zeros(Complex{T}, 2 * n)
    work_out = CUDA.zeros(Complex{T}, 2 * n)

    # Step 1 & 2: Pre-scale and apply inverse twiddle (vectorized)
    # First element: scale_zero, rest: scale_pos
    view(work_in, 1:1) .= Complex{T}.(view(input, 1:1)) .* view(plan.twiddle_backward, 1:1) .* plan.backward_scale_zero
    if n > 1
        view(work_in, 2:n) .= Complex{T}.(view(input, 2:n)) .* view(plan.twiddle_backward, 2:n) .* plan.backward_scale_pos
    end

    # Step 3: Create Hermitian-symmetric extension for real output (vectorized)
    # work_in[N+1] = 0 (Nyquist)
    # work_in[N+2:2N] = conj(work_in[N:-1:2])
    fill!(view(work_in, n+1:n+1), zero(Complex{T}))
    # reverse() is well-supported in GPUArrays.jl for contiguous CuArrays
    copyto!(view(work_in, n+2:2*n), conj.(reverse(work_in[2:n])))

    # Step 4: IFFT (non-aliased: work_in → work_out)
    mul!(work_out, plan.ifft_plan, work_in)

    # Step 5: Extract real values (first N elements, vectorized)
    output .= real.(view(work_out, 1:n))

    return output
end

# ============================================================================
# Dimension-by-Dimension FFT Plans (for mixed basis transforms)
# ============================================================================

"""
    GPUFFTPlanDim

GPU FFT plan for a specific dimension of a multi-dimensional array.
Used for mixed Fourier-Chebyshev transforms where we need separate
FFT along Fourier dimensions and DCT along Chebyshev dimensions.
"""
struct GPUFFTPlanDim{P, IP}
    plan::P
    iplan::IP
    full_size::Tuple{Vararg{Int}}
    transform_dim::Int
    is_real::Bool
end

"""
    plan_gpu_fft_dim(arch::GPU{CuDevice}, full_size::Tuple, T::Type, dim::Int; real_input::Bool=false)

Create a GPU FFT plan for a specific dimension of a multi-dimensional array.

# Arguments
- `arch`: GPU architecture
- `full_size`: Full array dimensions
- `T`: Element type
- `dim`: Dimension along which to transform (1-indexed)
- `real_input`: If true, create R2C/C2R plans for this dimension
"""
function plan_gpu_fft_dim(arch::GPU{CuDevice}, full_size::Tuple, T::Type, dim::Int; real_input::Bool=false)
    ensure_device!(arch)

    complex_T = T <: Complex ? T : Complex{T}
    ndims = length(full_size)

    @assert 1 <= dim <= ndims "Dimension $dim out of range for $(ndims)D array"

    if real_input
        # Real-to-complex FFT along specified dimension
        dummy_in = CUDA.zeros(T, full_size...)
        plan = CUFFT.plan_rfft(dummy_in, (dim,))

        # Output size: dimension `dim` becomes N/2 + 1
        out_size = ntuple(i -> i == dim ? div(full_size[i], 2) + 1 : full_size[i], ndims)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, full_size[dim], (dim,))

        return GPUFFTPlanDim(plan, iplan, full_size, dim, true)
    else
        # Complex-to-complex FFT along specified dimension
        dummy = CUDA.zeros(complex_T, full_size...)
        plan = CUFFT.plan_fft(dummy, (dim,))
        iplan = CUFFT.plan_ifft(dummy, (dim,))

        return GPUFFTPlanDim(plan, iplan, full_size, dim, false)
    end
end

# Fallback
plan_gpu_fft_dim(arch::GPU, full_size::Tuple, T::Type, dim::Int; real_input::Bool=false) =
    plan_gpu_fft_dim(GPU{CuDevice}(CUDA.device()), full_size, T, dim; real_input=real_input)

"""
    gpu_fft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim)

Execute forward FFT along a specific dimension.
"""
function gpu_fft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim)
    mul!(output, plan.plan, input)
    return output
end

"""
    gpu_ifft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim)

Execute inverse FFT along a specific dimension.
"""
function gpu_ifft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim)
    mul!(output, plan.iplan, input)
    return output
end

# ============================================================================
# GPU DCT for Multi-dimensional Arrays (along specific dimension)
# ============================================================================

"""
    GPUDCTPlanDim

GPU DCT plan for a specific dimension of a multi-dimensional array.
Uses batched 1D DCTs implemented via FFT.
"""
struct GPUDCTPlanDim{T}
    full_size::Tuple{Vararg{Int}}
    transform_dim::Int
    n::Int  # Size along transform dimension
    # Twiddle factors
    twiddle_forward::CuVector{Complex{T}}
    twiddle_backward::CuVector{Complex{T}}
    # Scaling factors
    forward_scale_zero::T
    forward_scale_pos::T
    backward_scale_zero::T
    backward_scale_pos::T
end

"""
    plan_gpu_dct_dim(arch::GPU{CuDevice}, full_size::Tuple, T::Type, dim::Int)

Create a GPU DCT plan for a specific dimension of a multi-dimensional array.
"""
function plan_gpu_dct_dim(arch::GPU{CuDevice}, full_size::Tuple, T::Type, dim::Int)
    ensure_device!(arch)

    real_T = T <: Complex ? real(T) : T
    n = full_size[dim]

    # Twiddle factors (use correct precision Complex{real_T})
    complex_T = Complex{real_T}
    twiddle_forward = CuArray(complex_T[exp(complex_T(-im * π * k / (2 * n))) for k in 0:n-1])
    twiddle_backward = CuArray(complex_T[exp(complex_T(im * π * k / (2 * n))) for k in 0:n-1])

    # Scaling factors (Tarang convention, see plan_gpu_dct for detailed explanation).
    # backward_scale_pos=0.5 cancels the ×2 in the DCT-III kernel formula.
    forward_scale_zero = real_T(1.0 / n / 2.0)
    forward_scale_pos = real_T(1.0 / n)
    backward_scale_zero = real_T(1.0)
    backward_scale_pos = real_T(0.5)

    return GPUDCTPlanDim{real_T}(
        full_size, dim, n,
        twiddle_forward, twiddle_backward,
        forward_scale_zero, forward_scale_pos,
        backward_scale_zero, backward_scale_pos
    )
end

plan_gpu_dct_dim(arch::GPU, full_size::Tuple, T::Type, dim::Int) =
    plan_gpu_dct_dim(GPU{CuDevice}(CUDA.device()), full_size, T, dim)

"""
DCT forward kernel for 2D arrays along dimension 2 (columns).
Each thread handles one column.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
"""
@kernel function dct_forward_2d_dim2_kernel!(output, @Const(input), @Const(twiddle),
                                              scale_zero, scale_pos, pi_val, nx, ny)
    i = @index(Global)  # Column index
    if i <= nx
        # Process column i: apply DCT along dimension 2
        # Using direct DCT-II formula (simpler for GPU, reasonably efficient for moderate N)
        @inbounds for k in 1:ny
            sum_val = zero(eltype(output))
            for j in 1:ny
                # DCT-II: cos(π * (k-1) * (2*(j-1) + 1) / (2*ny))
                angle = pi_val * (k - 1) * (2 * (j - 1) + 1) / (2 * ny)
                sum_val += input[i, j] * cos(angle)
            end
            # Apply scaling
            if k == 1
                output[i, k] = sum_val * scale_zero
            else
                output[i, k] = sum_val * scale_pos
            end
        end
    end
end

"""
DCT forward kernel for 2D arrays along dimension 1 (rows).
Each thread handles one row.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
"""
@kernel function dct_forward_2d_dim1_kernel!(output, @Const(input), @Const(twiddle),
                                              scale_zero, scale_pos, pi_val, nx, ny)
    j = @index(Global)  # Row index
    if j <= ny
        # Process row j: apply DCT along dimension 1
        @inbounds for k in 1:nx
            sum_val = zero(eltype(output))
            for i in 1:nx
                angle = pi_val * (k - 1) * (2 * (i - 1) + 1) / (2 * nx)
                sum_val += input[i, j] * cos(angle)
            end
            if k == 1
                output[k, j] = sum_val * scale_zero
            else
                output[k, j] = sum_val * scale_pos
            end
        end
    end
end

"""
DCT backward kernel for 2D arrays along dimension 2 (columns).
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.

Scaling convention (Tarang): `scale_zero=1.0` for DC, `scale_pos=0.5` for AC.
The factor of 2 in `scale_pos * 2` comes from the DCT-III formula and cancels with
scale_pos=0.5, yielding an effective weight of 1.0 for AC terms. This ensures the
backward transform is the exact inverse of the forward (which uses 1/(2N) for DC, 1/N for AC).
"""
@kernel function dct_backward_2d_dim2_kernel!(output, @Const(input), @Const(twiddle),
                                               scale_zero, scale_pos, pi_val, nx, ny)
    i = @index(Global)  # Column index
    if i <= nx
        # Process column i: apply DCT-III (inverse) along dimension 2
        @inbounds for j in 1:ny
            # DCT-III: x_j = c_0*scale_zero + Σ_{k=1}^{N-1} c_k * (2*scale_pos) * cos(...)
            sum_val = input[i, 1] * scale_zero  # DC component
            for k in 2:ny
                angle = pi_val * (k - 1) * (2 * (j - 1) + 1) / (2 * ny)
                sum_val += input[i, k] * scale_pos * 2 * cos(angle)
            end
            output[i, j] = sum_val
        end
    end
end

"""
DCT backward kernel for 2D arrays along dimension 1 (rows).
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
See `dct_backward_2d_dim2_kernel!` for scaling convention explanation.
"""
@kernel function dct_backward_2d_dim1_kernel!(output, @Const(input), @Const(twiddle),
                                               scale_zero, scale_pos, pi_val, nx, ny)
    j = @index(Global)  # Row index
    if j <= ny
        @inbounds for i in 1:nx
            sum_val = input[1, j] * scale_zero
            for k in 2:nx
                angle = pi_val * (k - 1) * (2 * (i - 1) + 1) / (2 * nx)
                sum_val += input[k, j] * scale_pos * 2 * cos(angle)
            end
            output[i, j] = sum_val
        end
    end
end

"""
    gpu_dct_dim!(output::CuArray, input::CuArray, plan::GPUDCTPlanDim, ::Val{:forward})

Execute forward DCT along a specific dimension of a multi-dimensional array.
"""
function gpu_dct_dim!(output::CuArray{T, 2}, input::CuArray{T, 2},
                      plan::GPUDCTPlanDim, ::Val{:forward}) where T
    nx, ny = size(input)
    arch = Tarang.architecture(input)
    pi_val = T(π)

    if plan.transform_dim == 1
        # DCT along rows (dimension 1)
        launch!(arch, dct_forward_2d_dim1_kernel!, output, input, plan.twiddle_forward,
                plan.forward_scale_zero, plan.forward_scale_pos, pi_val, nx, ny;
                ndrange=ny)
    else
        # DCT along columns (dimension 2)
        launch!(arch, dct_forward_2d_dim2_kernel!, output, input, plan.twiddle_forward,
                plan.forward_scale_zero, plan.forward_scale_pos, pi_val, nx, ny;
                ndrange=nx)
    end
    return output
end

"""
    gpu_dct_dim!(output::CuArray, input::CuArray, plan::GPUDCTPlanDim, ::Val{:backward})

Execute backward DCT along a specific dimension of a multi-dimensional array.
"""
function gpu_dct_dim!(output::CuArray{T, 2}, input::CuArray{T, 2},
                      plan::GPUDCTPlanDim, ::Val{:backward}) where T
    nx, ny = size(input)
    arch = Tarang.architecture(input)
    pi_val = T(π)

    if plan.transform_dim == 1
        launch!(arch, dct_backward_2d_dim1_kernel!, output, input, plan.twiddle_backward,
                plan.backward_scale_zero, plan.backward_scale_pos, pi_val, nx, ny;
                ndrange=ny)
    else
        launch!(arch, dct_backward_2d_dim2_kernel!, output, input, plan.twiddle_backward,
                plan.backward_scale_zero, plan.backward_scale_pos, pi_val, nx, ny;
                ndrange=nx)
    end
    return output
end

# ============================================================================
# 3D DCT Kernels
# ============================================================================

"""
DCT forward kernel for 3D arrays along dimension 1.
Each thread handles one (j, k) pair - a "fiber" along dimension 1.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
"""
@kernel function dct_forward_3d_dim1_kernel!(output, @Const(input),
                                              scale_zero, scale_pos, pi_val, nx, ny, nz)
    idx = @index(Global)
    # Map linear index to (j, k) pair
    j = ((idx - 1) % ny) + 1
    k = ((idx - 1) ÷ ny) + 1

    if j <= ny && k <= nz
        @inbounds for m in 1:nx  # output index along dim 1
            sum_val = zero(eltype(output))
            for i in 1:nx  # input index along dim 1
                angle = pi_val * (m - 1) * (2 * (i - 1) + 1) / (2 * nx)
                sum_val += input[i, j, k] * cos(angle)
            end
            if m == 1
                output[m, j, k] = sum_val * scale_zero
            else
                output[m, j, k] = sum_val * scale_pos
            end
        end
    end
end

"""
DCT forward kernel for 3D arrays along dimension 2.
Each thread handles one (i, k) pair - a "fiber" along dimension 2.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
"""
@kernel function dct_forward_3d_dim2_kernel!(output, @Const(input),
                                              scale_zero, scale_pos, pi_val, nx, ny, nz)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    k = ((idx - 1) ÷ nx) + 1

    if i <= nx && k <= nz
        @inbounds for m in 1:ny  # output index along dim 2
            sum_val = zero(eltype(output))
            for j in 1:ny  # input index along dim 2
                angle = pi_val * (m - 1) * (2 * (j - 1) + 1) / (2 * ny)
                sum_val += input[i, j, k] * cos(angle)
            end
            if m == 1
                output[i, m, k] = sum_val * scale_zero
            else
                output[i, m, k] = sum_val * scale_pos
            end
        end
    end
end

"""
DCT forward kernel for 3D arrays along dimension 3.
Each thread handles one (i, j) pair - a "fiber" along dimension 3.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
"""
@kernel function dct_forward_3d_dim3_kernel!(output, @Const(input),
                                              scale_zero, scale_pos, pi_val, nx, ny, nz)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    j = ((idx - 1) ÷ nx) + 1

    if i <= nx && j <= ny
        @inbounds for m in 1:nz  # output index along dim 3
            sum_val = zero(eltype(output))
            for k in 1:nz  # input index along dim 3
                angle = pi_val * (m - 1) * (2 * (k - 1) + 1) / (2 * nz)
                sum_val += input[i, j, k] * cos(angle)
            end
            if m == 1
                output[i, j, m] = sum_val * scale_zero
            else
                output[i, j, m] = sum_val * scale_pos
            end
        end
    end
end

"""
DCT backward kernel for 3D arrays along dimension 1.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
See `dct_backward_2d_dim2_kernel!` for scaling convention explanation.
"""
@kernel function dct_backward_3d_dim1_kernel!(output, @Const(input),
                                               scale_zero, scale_pos, pi_val, nx, ny, nz)
    idx = @index(Global)
    j = ((idx - 1) % ny) + 1
    k = ((idx - 1) ÷ ny) + 1

    if j <= ny && k <= nz
        @inbounds for i in 1:nx  # output index along dim 1
            sum_val = input[1, j, k] * scale_zero
            for m in 2:nx  # input index along dim 1
                angle = pi_val * (m - 1) * (2 * (i - 1) + 1) / (2 * nx)
                sum_val += input[m, j, k] * scale_pos * 2 * cos(angle)
            end
            output[i, j, k] = sum_val
        end
    end
end

"""
DCT backward kernel for 3D arrays along dimension 2.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
See `dct_backward_2d_dim2_kernel!` for scaling convention explanation.
"""
@kernel function dct_backward_3d_dim2_kernel!(output, @Const(input),
                                               scale_zero, scale_pos, pi_val, nx, ny, nz)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    k = ((idx - 1) ÷ nx) + 1

    if i <= nx && k <= nz
        @inbounds for j in 1:ny  # output index along dim 2
            sum_val = input[i, 1, k] * scale_zero
            for m in 2:ny  # input index along dim 2
                angle = pi_val * (m - 1) * (2 * (j - 1) + 1) / (2 * ny)
                sum_val += input[i, m, k] * scale_pos * 2 * cos(angle)
            end
            output[i, j, k] = sum_val
        end
    end
end

"""
DCT backward kernel for 3D arrays along dimension 3.
`pi_val` must be passed as `T(π)` to avoid Float64 promotion on GPU.
See `dct_backward_2d_dim2_kernel!` for scaling convention explanation.
"""
@kernel function dct_backward_3d_dim3_kernel!(output, @Const(input),
                                               scale_zero, scale_pos, pi_val, nx, ny, nz)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    j = ((idx - 1) ÷ nx) + 1

    if i <= nx && j <= ny
        @inbounds for k in 1:nz  # output index along dim 3
            sum_val = input[i, j, 1] * scale_zero
            for m in 2:nz  # input index along dim 3
                angle = pi_val * (m - 1) * (2 * (k - 1) + 1) / (2 * nz)
                sum_val += input[i, j, m] * scale_pos * 2 * cos(angle)
            end
            output[i, j, k] = sum_val
        end
    end
end

"""
    gpu_dct_dim!(output::CuArray{T,3}, input::CuArray{T,3}, plan::GPUDCTPlanDim, ::Val{:forward})

Execute forward DCT along a specific dimension of a 3D array.
"""
function gpu_dct_dim!(output::CuArray{T, 3}, input::CuArray{T, 3},
                      plan::GPUDCTPlanDim, ::Val{:forward}) where T
    nx, ny, nz = size(input)
    arch = Tarang.architecture(input)
    pi_val = T(π)

    if plan.transform_dim == 1
        # DCT along dimension 1: each thread handles one (j,k) fiber
        ndrange = ny * nz
        launch!(arch, dct_forward_3d_dim1_kernel!, output, input,
                plan.forward_scale_zero, plan.forward_scale_pos, pi_val, nx, ny, nz;
                ndrange=ndrange)
    elseif plan.transform_dim == 2
        # DCT along dimension 2: each thread handles one (i,k) fiber
        ndrange = nx * nz
        launch!(arch, dct_forward_3d_dim2_kernel!, output, input,
                plan.forward_scale_zero, plan.forward_scale_pos, pi_val, nx, ny, nz;
                ndrange=ndrange)
    else
        # DCT along dimension 3: each thread handles one (i,j) fiber
        ndrange = nx * ny
        launch!(arch, dct_forward_3d_dim3_kernel!, output, input,
                plan.forward_scale_zero, plan.forward_scale_pos, pi_val, nx, ny, nz;
                ndrange=ndrange)
    end
    return output
end

"""
    gpu_dct_dim!(output::CuArray{T,3}, input::CuArray{T,3}, plan::GPUDCTPlanDim, ::Val{:backward})

Execute backward DCT along a specific dimension of a 3D array.
"""
function gpu_dct_dim!(output::CuArray{T, 3}, input::CuArray{T, 3},
                      plan::GPUDCTPlanDim, ::Val{:backward}) where T
    nx, ny, nz = size(input)
    arch = Tarang.architecture(input)
    pi_val = T(π)

    if plan.transform_dim == 1
        ndrange = ny * nz
        launch!(arch, dct_backward_3d_dim1_kernel!, output, input,
                plan.backward_scale_zero, plan.backward_scale_pos, pi_val, nx, ny, nz;
                ndrange=ndrange)
    elseif plan.transform_dim == 2
        ndrange = nx * nz
        launch!(arch, dct_backward_3d_dim2_kernel!, output, input,
                plan.backward_scale_zero, plan.backward_scale_pos, pi_val, nx, ny, nz;
                ndrange=ndrange)
    else
        ndrange = nx * ny
        launch!(arch, dct_backward_3d_dim3_kernel!, output, input,
                plan.backward_scale_zero, plan.backward_scale_pos, pi_val, nx, ny, nz;
                ndrange=ndrange)
    end
    return output
end
