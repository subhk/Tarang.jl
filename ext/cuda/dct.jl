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
    # Work arrays (shared across calls, protected by lock for thread safety)
    work_complex::CuVector{Complex{T}}
    work_complex_out::CuVector{Complex{T}}
    # FFT plan for the extended array
    fft_plan::Any
    ifft_plan::Any
    # Lock for thread-safe access to shared work arrays
    lock::ReentrantLock
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

    # Scaling factors following the Tarang FastCosineTransform convention
    # (see src/core/transforms/transform_fft_dct.jl, which the N-D path uses):
    #   forward:  X_0 = S_0/(2N),  X_k = S_k/N,   S_k = Σ_j x_j cos(πk(2j+1)/(2N))
    #   backward: x_j = 2 Σ_{k=0}^{N-1} X_k cos(πk(2j+1)/(2N))   (exact inverse)
    # The forward scales below are applied to S_k by `gpu_forward_dct_1d!` (which
    # halves the raw extension values, since Re(twiddle·FFT([x; rev x])) = 2·S_k).
    forward_scale_zero = real_T(1.0 / n / 2.0)   # DC: 1/(2N) on S_0
    forward_scale_pos = real_T(1.0 / n)          # AC: 1/N on S_k
    # Backward pre-scales feed the Hermitian 2N-spectrum W_k = c_k·e^{iπk/(2N)},
    # whose NORMALIZED ifft (CUFFT plan_ifft carries 1/(2N)) gives
    #   y_j = (1/(2N)) [c_0 + 2 Σ_{k≥1} c_k cos(πk(2j+1)/(2N))].
    # Matching y_j to the convention above requires c_0 = 4N·X_0, c_k = 2N·X_k
    # (this is where the ifft's 1/(2N) is cancelled).
    backward_scale_zero = real_T(4 * n)          # DC: 4N
    backward_scale_pos = real_T(2 * n)           # AC: 2N

    # Work arrays for 2N-point FFT (pre-allocated to avoid per-call GPU allocation)
    work_complex = CUDA.zeros(complex_T, 2 * n)
    work_complex_out = CuVector{complex_T}(undef, 2 * n)

    # FFT plans for the extended array
    fft_plan = CUFFT.plan_fft(work_complex)
    ifft_plan = CUFFT.plan_ifft(work_complex)

    return GPUDCTPlan{real_T}(
        n, axis,
        twiddle_forward, twiddle_backward,
        forward_scale_zero, forward_scale_pos,
        backward_scale_zero, backward_scale_pos,
        work_complex, work_complex_out,
        fft_plan, ifft_plan,
        ReentrantLock()
    )
end

# Fallback for generic GPU
plan_gpu_dct(arch::GPU, n::Int, T::Type, axis::Int) = plan_gpu_dct(GPU{CuDevice}(CUDA.device()), n, T, axis)

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
Fully parallelized: one thread per (m, j, k) element.
"""
@kernel function reorder_for_dct_3d_dim1_kernel!(work, @Const(x), Nx, Ny, Nz)
    idx = @index(Global)
    half_Nx = Nx ÷ 2
    m = ((idx - 1) % half_Nx)
    jk = (idx - 1) ÷ half_Nx
    j = (jk % Ny) + 1
    k = (jk ÷ Ny) + 1

    if j <= Ny && k <= Nz
        @inbounds work[m + 1, j, k] = x[2m + 1, j, k]
        @inbounds work[Nx - m, j, k] = x[2m + 2, j, k]
    end
end

"""
3D reorder kernel for DCT along dimension 2.
Fully parallelized: one thread per (i, m, k) element.
"""
@kernel function reorder_for_dct_3d_dim2_kernel!(work, @Const(x), Nx, Ny, Nz)
    idx = @index(Global)
    half_Ny = Ny ÷ 2
    i = ((idx - 1) % Nx) + 1
    mk = (idx - 1) ÷ Nx
    m = mk % half_Ny
    k = (mk ÷ half_Ny) + 1

    if i <= Nx && k <= Nz
        @inbounds work[i, m + 1, k] = x[i, 2m + 1, k]
        @inbounds work[i, Ny - m, k] = x[i, 2m + 2, k]
    end
end

"""
3D reorder kernel for DCT along dimension 3.
Fully parallelized: one thread per (i, j, m) element.
"""
@kernel function reorder_for_dct_3d_dim3_kernel!(work, @Const(x), Nx, Ny, Nz)
    idx = @index(Global)
    half_Nz = Nz ÷ 2
    i = ((idx - 1) % Nx) + 1
    jm = (idx - 1) ÷ Nx
    j = (jm % Ny) + 1
    m = jm ÷ Ny

    if j <= Ny && m < half_Nz
        @inbounds work[i, j, m + 1] = x[i, j, 2m + 1]
        @inbounds work[i, j, Nz - m] = x[i, j, 2m + 2]
    end
end

"""
3D inverse reorder kernel for DCT along dimension 1.
Fully parallelized: one thread per (m, j, k) element.
"""
@kernel function inverse_reorder_for_dct_3d_dim1_kernel!(x, @Const(work), Nx, Ny, Nz)
    idx = @index(Global)
    half_Nx = Nx ÷ 2
    m = ((idx - 1) % half_Nx)
    jk = (idx - 1) ÷ half_Nx
    j = (jk % Ny) + 1
    k = (jk ÷ Ny) + 1

    if j <= Ny && k <= Nz
        @inbounds x[2m + 1, j, k] = work[m + 1, j, k]
        @inbounds x[2m + 2, j, k] = work[Nx - m, j, k]
    end
end

"""
3D inverse reorder kernel for DCT along dimension 2.
Fully parallelized: one thread per (i, m, k) element.
"""
@kernel function inverse_reorder_for_dct_3d_dim2_kernel!(x, @Const(work), Nx, Ny, Nz)
    idx = @index(Global)
    half_Ny = Ny ÷ 2
    i = ((idx - 1) % Nx) + 1
    mk = (idx - 1) ÷ Nx
    m = mk % half_Ny
    k = (mk ÷ half_Ny) + 1

    if i <= Nx && k <= Nz
        @inbounds x[i, 2m + 1, k] = work[i, m + 1, k]
        @inbounds x[i, 2m + 2, k] = work[i, Ny - m, k]
    end
end

"""
3D inverse reorder kernel for DCT along dimension 3.
Fully parallelized: one thread per (i, j, m) element.
"""
@kernel function inverse_reorder_for_dct_3d_dim3_kernel!(x, @Const(work), Nx, Ny, Nz)
    idx = @index(Global)
    half_Nz = Nz ÷ 2
    i = ((idx - 1) % Nx) + 1
    jm = (idx - 1) ÷ Nx
    j = (jm % Ny) + 1
    m = jm ÷ Ny

    if j <= Ny && m < half_Nz
        @inbounds x[i, j, 2m + 1] = work[i, j, m + 1]
        @inbounds x[i, j, 2m + 2] = work[i, j, Nz - m]
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
        ndrange = (Nx ÷ 2) * Ny * Nz
        launch!(arch, reorder_for_dct_3d_dim1_kernel!, work, x, Nx, Ny, Nz; ndrange=ndrange)
    elseif dim == 2
        ndrange = Nx * (Ny ÷ 2) * Nz
        launch!(arch, reorder_for_dct_3d_dim2_kernel!, work, x, Nx, Ny, Nz; ndrange=ndrange)
    else
        ndrange = Nx * Ny * (Nz ÷ 2)
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
        ndrange = (Nx ÷ 2) * Ny * Nz
        launch!(arch, inverse_reorder_for_dct_3d_dim1_kernel!, x, work, Nx, Ny, Nz; ndrange=ndrange)
    elseif dim == 2
        ndrange = Nx * (Ny ÷ 2) * Nz
        launch!(arch, inverse_reorder_for_dct_3d_dim2_kernel!, x, work, Nx, Ny, Nz; ndrange=ndrange)
    else
        ndrange = Nx * Ny * (Nz ÷ 2)
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

where W[k] = exp(-i*π*k/(2N)) are the twiddle factors.
"""
struct OptimizedGPUDCTPlan{T}
    size::Int
    # Work arrays (shared — guard with _dct_plan_lock for thread safety)
    work_real::CuVector{T}
    work_complex::CuVector{Complex{T}}
    # Twiddle factors: exp(-iπk/(2N)) for k=0..N/2 (forward)
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

    # Twiddle factors for forward DCT: exp(-i*π*k/(2N)) for k=0..N/2.
    # With the even–odd reorder v and V = rfft(v), the plain cosine sum is
    #   S_k = Re(e^{-iπk/(2N)}·V_k)   and   S_{N-k} = -Im(e^{-iπk/(2N)}·V_k),
    # so the twiddle carries NO extra factor (the forward scales below apply the
    # Tarang convention directly to S_k).
    # Note: We only need N/2+1 twiddle factors for R2C output
    twiddle = CuArray(complex_T[exp(complex_T(-im * π * k / (2 * n))) for k in 0:(n÷2)])

    # Inverse twiddle factors for backward DCT: exp(i*π*k/(2N)) for k=0..N/2
    # (rebuilds V_k = e^{iπk/(2N)}·(S_k - i·S_{N-k}) for the C2R IFFT).
    twiddle_inv = CuArray(complex_T[exp(complex_T(im * π * k / (2 * n))) for k in 0:(n÷2)])

    # FFT plans: R2C for forward, C2R for backward
    rfft_plan = CUFFT.plan_rfft(work_real)
    irfft_plan = CUFFT.plan_irfft(work_complex, n)

    # Scaling factors following the Tarang FastCosineTransform convention
    # (identical to src/core/transforms/transform_fft_dct.jl):
    #   forward:  X_0 = S_0/(2N),  X_k = S_k/N
    #   backward: x_j = 2 Σ_{k=0}^{N-1} X_k cos(πk(2j+1)/(2N))   (exact inverse)
    # The backward pre-scales convert X back to the plain sums S (S_0 = 2N·X_0,
    # S_k = N·X_k); the normalized plan_irfft (carries 1/N) then reconstructs the
    # reordered grid values exactly.
    forward_scale_zero = real_T(1.0 / n / 2.0)   # DC: 1/(2N) on S_0
    forward_scale_pos = real_T(1.0 / n)          # AC: 1/N on S_k
    backward_scale_zero = real_T(2 * n)          # DC: X_0 → S_0 = 2N·X_0
    backward_scale_pos = real_T(n)               # AC: X_k → S_k = N·X_k

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
                # Nyquist component (k0=N/2): the C2R input bin must be REAL-valued
                # √2·S_{N/2}. This is the regular branch with the mirror partner
                # being itself: (S - i·S)·e^{iπ/4} = √2·S. (Using the plain
                # `S·twiddle_inv` here would leave a complex bin whose imaginary
                # part the C2R transform drops — halving the Nyquist amplitude.)
                scaled_coeff = coeffs[k] * scale_pos
                complex_out[k] = Complex(scaled_coeff, -scaled_coeff) * twiddle_inv[k]
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

    # Lock to prevent concurrent access to shared work buffers
    lock(plan.lock)
    try
        work_in = plan.work_complex
        work_out = plan.work_complex_out
        fill!(work_in, zero(Complex{T}))

        # Step 1: Create symmetric extension (vectorized, no scalar indexing)
        copyto!(view(work_in, 1:n), Complex{T}.(input))
        copyto!(view(work_in, n+1:2*n), Complex{T}.(reverse(input)))

        # Step 2: FFT of extended array (non-aliased: work_in → work_out)
        mul!(work_out, plan.fft_plan, work_in)

        # Step 3: Extract and apply twiddle factors, then scale (vectorized).
        # Identity: Re(twiddle_forward[k] · FFT([x; reverse(x)])[k]) = 2·S_k with
        # S_k = Σ_j x_j cos(πk(2j+1)/(2N)), so halve the raw values before applying
        # the convention scales (X_0 = S_0/(2N), X_k = S_k/N).
        half = T(0.5)
        raw_coeffs = real.(view(work_out, 1:n) .* plan.twiddle_forward)
        output .= raw_coeffs .* (plan.forward_scale_pos * half)
        view(output, 1:1) .= view(raw_coeffs, 1:1) .* (plan.forward_scale_zero * half)
    finally
        unlock(plan.lock)
    end

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
    # 1. Pre-scale coefficients (plan.backward_scale_* = 4N / 2N — these cancel
    #    the 1/(2N) carried by the normalized plan_ifft; see plan_gpu_dct)
    # 2. Apply inverse twiddle factors
    # 3. Create Hermitian-symmetric extension
    # 4. Compute IFFT
    # 5. Extract first N real values

    # Lock to prevent concurrent access to shared work buffers
    lock(plan.lock)
    try
    work_in = plan.work_complex
    work_out = plan.work_complex_out
    fill!(work_in, zero(Complex{T}))

    # Step 1 & 2: Pre-scale and apply inverse twiddle (vectorized)
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

    finally
        unlock(plan.lock)
    end

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
    # NOTE: these fields (and the twiddles above) are currently UNUSED — the former
    # O(N²) cos-sum kernels were removed and `gpu_dct_dim!` delegates to
    # `Tarang.fft_dct_{forward,backward}_dim`, which applies the convention
    # internally. Values kept self-consistent with that convention:
    #   forward:  X_0 = S_0/(2N), X_k = S_k/N
    #   backward: x_j = 2 Σ_k X_k cos(πk(2j+1)/(2N)); in the old kernel structure
    #   `X_0·scale_zero + Σ_{k≥1} X_k·scale_pos·2·cos(...)` the exact-inverse
    #   values are scale_zero = 2, scale_pos = 1 (the old 1.0/0.5 round-tripped to x/2).
    forward_scale_zero = real_T(1.0 / n / 2.0)
    forward_scale_pos = real_T(1.0 / n)
    backward_scale_zero = real_T(2.0)
    backward_scale_pos = real_T(1.0)

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
    gpu_dct_dim!(output::CuArray, input::CuArray, plan::GPUDCTPlanDim, ::Val{:forward})

Execute forward DCT along a specific dimension of a multi-dimensional array.
"""
# O(N log N) FFT-based multi-dim DCT (replaces the former O(N²) cos-sum kernels).
# Delegates to the device-agnostic `fft_dct_*_dim` in src/core/transforms (validated
# on CPU in test_fft_dct.jl; runs on GPU via CUFFT). Handles 2D and 3D. Real input
# only — complex fields are split into real/imag parts upstream in transforms.jl.
function gpu_dct_dim!(output::CuArray{T}, input::CuArray{T},
                      plan::GPUDCTPlanDim, ::Val{:forward}) where {T<:Real}
    output .= Tarang.fft_dct_forward_dim(input, plan.transform_dim)
    return output
end

"""
    gpu_dct_dim!(output::CuArray, input::CuArray, plan::GPUDCTPlanDim, ::Val{:backward})

Execute backward DCT along a specific dimension of a multi-dimensional array.
"""
function gpu_dct_dim!(output::CuArray{T}, input::CuArray{T},
                      plan::GPUDCTPlanDim, ::Val{:backward}) where {T<:Real}
    output .= Tarang.fft_dct_backward_dim(input, plan.transform_dim)
    return output
end


# ── Cached DCT plan getters ─────────────────────────────────────────────────
# CUFFT-backed DCT plans are expensive to build (CUFFT planning + work-buffer
# allocation), and `GPUDCTPlan` deliberately reuses its internal work arrays
# across calls. The pure-Chebyshev transform path used to call `plan_gpu_dct`/
# `plan_gpu_dct_dim` on EVERY forward/backward transform, rebuilding the plan
# (and its buffers) each step. Cache plans per (device, size, eltype, axis) so
# the hot path reuses one plan — mirroring `get_gpu_fft_plan` for the FFT path.
const GPU_DCT_PLAN_CACHE = Dict{Tuple, GPUDCTPlan}()
const GPU_DCT_DIM_PLAN_CACHE = Dict{Tuple, GPUDCTPlanDim}()
const _GPU_DCT_PLAN_CACHE_LOCK = ReentrantLock()

# Key caches by the device the plan is actually built for. When `arch.device` is
# a concrete `CuDevice` use it directly (direct multi-GPU calls may target a
# device other than the current one); otherwise fall back to the current device.
# Mirrors `get_gpu_fft_plan`'s `GPU{CuDevice}` vs generic `GPU` split so the DCT
# and FFT caches agree on device identity.
_dct_cache_device_id(arch::GPU{CuDevice}) = CUDA.deviceid(arch.device)
_dct_cache_device_id(arch::GPU) = _current_device_id()

"""Get or create a cached 1D GPU DCT plan (thread-safe)."""
function get_gpu_dct_plan(arch::GPU, n::Int, T::Type, axis::Int)
    key = (_dct_cache_device_id(arch), n, T, axis)
    lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        get!(() -> plan_gpu_dct(arch, n, T, axis), GPU_DCT_PLAN_CACHE, key)
    end
end

"""Get or create a cached per-dimension GPU DCT plan (thread-safe)."""
function get_gpu_dct_dim_plan(arch::GPU, full_size::Tuple, T::Type, dim::Int)
    key = (_dct_cache_device_id(arch), full_size, T, dim)
    lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        get!(() -> plan_gpu_dct_dim(arch, full_size, T, dim), GPU_DCT_DIM_PLAN_CACHE, key)
    end
end

"""Clear all cached GPU DCT plans (thread-safe)."""
function clear_gpu_dct_plan_cache!()
    lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        empty!(GPU_DCT_PLAN_CACHE)
        empty!(GPU_DCT_DIM_PLAN_CACHE)
    end
end

# ── Reusable scratch buffers for the complex / multi-dim DCT paths ───────────
# The pure-Chebyshev transform branches allocated fresh CuArrays every call:
# `real.(x)`/`imag.(x)` splits for complex data (DCT kernels need real input),
# and a new output array per dimension in the multi-dim ping-pong. For a field
# transformed every timestep this is 4–8 device allocations per transform. Since
# the DCT preserves array shape, we cache `count` reusable buffers per
# (device, shape, eltype, count) and the multi-dim loop alternates between two of
# them. Safe for serial single-GPU use: each transform copies its result into the
# field's coeff/grid array before returning, so the scratch is free for the next
# call (same assumption the plan work-buffer reuse already makes).
const _GPU_DCT_SCRATCH_CACHE = Dict{Tuple, Any}()

"""Get `count` cached, reusable `(shape, T)` GPU scratch buffers (thread-safe)."""
function get_gpu_dct_scratch(arch::GPU, shape::NTuple{N,Int}, ::Type{T}, count::Int) where {N,T}
    key = (_dct_cache_device_id(arch), shape, T, count)
    buffers = lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        get!(() -> CuArray{T,N}[CUDA.zeros(T, shape...) for _ in 1:count],
             _GPU_DCT_SCRATCH_CACHE, key)
    end
    return buffers::Vector{CuArray{T,N}}   # function barrier: type-stable downstream
end

"""Clear all cached GPU DCT scratch buffers (thread-safe)."""
function clear_gpu_dct_scratch_cache!()
    lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        empty!(_GPU_DCT_SCRATCH_CACHE)
    end
end
