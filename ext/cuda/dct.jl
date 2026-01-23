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

    work = plan.work_complex

    # Step 1: Create symmetric extension (vectorized, no scalar indexing)
    copyto!(view(work, 1:n), Complex{T}.(input))
    # Use explicit reversed indexing instead of reverse() for GPU safety
    copyto!(view(work, n+1:2*n), Complex{T}.(input[n:-1:1]))

    # Step 2: FFT of extended array
    mul!(work, plan.fft_plan, work)

    # Step 3: Extract and apply twiddle factors, then scale (vectorized)
    # Compute real part of (work[1:N] .* twiddle_forward)
    raw_coeffs = real.(view(work, 1:n) .* plan.twiddle_forward)
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

    work = plan.work_complex

    # Step 1 & 2: Pre-scale and apply inverse twiddle (vectorized)
    # First element: scale_zero, rest: scale_pos
    view(work, 1:1) .= Complex{T}.(view(input, 1:1)) .* view(plan.twiddle_backward, 1:1) .* plan.backward_scale_zero
    if n > 1
        view(work, 2:n) .= Complex{T}.(view(input, 2:n)) .* view(plan.twiddle_backward, 2:n) .* plan.backward_scale_pos
    end

    # Step 3: Create Hermitian-symmetric extension for real output (vectorized)
    # work[N+1] = 0 (Nyquist)
    # work[N+2:2N] = conj(work[N:-1:2])
    fill!(view(work, n+1:n+1), zero(Complex{T}))
    # Use explicit reversed indexing instead of reverse() for GPU safety
    copyto!(view(work, n+2:2*n), conj.(work[n:-1:2]))

    # Step 4: IFFT
    mul!(work, plan.ifft_plan, work)

    # Step 5: Extract real values (first N elements, vectorized)
    output .= real.(view(work, 1:n))

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
        plan = CUFFT.plan_rfft(dummy_in, dim)

        # Output size: dimension `dim` becomes N/2 + 1
        out_size = ntuple(i -> i == dim ? div(full_size[i], 2) + 1 : full_size[i], ndims)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, full_size[dim], dim)

        return GPUFFTPlanDim(plan, iplan, full_size, dim, true)
    else
        # Complex-to-complex FFT along specified dimension
        dummy = CUDA.zeros(complex_T, full_size...)
        plan = CUFFT.plan_fft(dummy, dim)
        iplan = CUFFT.plan_ifft(dummy, dim)

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
