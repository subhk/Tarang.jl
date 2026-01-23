# ============================================================================
# GPU Transform Implementations
# ============================================================================

# Field data accessors (internal helpers)
get_grid_data(field::ScalarField) = field.grid_data
get_coeff_data(field::ScalarField) = field.coeff_data
set_grid_data!(field::ScalarField, data) = (field.grid_data = data)
set_coeff_data!(field::ScalarField, data) = (field.coeff_data = data)

"""
    gpu_forward_transform!(field::ScalarField)

GPU-specific forward transform using CUFFT.
Supports:
- Pure Fourier (RealFourier, ComplexFourier) - uses cuFFT
- Pure Chebyshev - uses GPU DCT
- Mixed Fourier-Chebyshev - dimension-by-dimension transforms
"""
function Tarang.gpu_forward_transform!(field::ScalarField)
    arch = field.dist.architecture
    if !Tarang.is_gpu(arch)
        return false
    end

    data_g = get_grid_data(field)
    if !isa(data_g, CuArray)
        return false
    end

    gpu_arch = arch::GPU

    # Ensure correct device is active for multi-GPU support
    ensure_device!(gpu_arch)

    # Determine transform type based on bases
    bases = field.bases
    if isempty(bases)
        return false
    end

    # Use LOCAL array size from actual data (not global domain size)
    # This is critical for distributed computing where each rank owns a portion
    local_grid_shape = size(data_g)

    if !Tarang.should_use_gpu_fft(field, local_grid_shape)
        return false
    end

    # Classify bases
    all_fourier = all(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    all_chebyshev = all(b -> isa(b, ChebyshevT), bases)
    has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    has_chebyshev = any(b -> isa(b, ChebyshevT), bases)

    input_T = eltype(data_g)
    coeff_T = Tarang.coefficient_eltype(field.dtype)

    if all_fourier
        # Pure Fourier case - use optimized multi-dimensional FFT
        # CUFFT's plan_rfft convention: R2C on first dimension, C2C on rest.
        # Only valid when bases[1] is RealFourier.
        has_real = any(b -> isa(b, RealFourier), bases)
        first_is_real = isa(bases[1], RealFourier)

        if has_real && !first_is_real
            # RealFourier on a non-first dimension: can't use multi-dim rfft.
            # Fall through to dimension-by-dimension approach (mixed transform handles this).
            plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)
            local_coeff_shape = plan.coeff_shape

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != coeff_T ||
                          size(existing_coeff) != local_coeff_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
            end

            gpu_mixed_forward_transform!(get_coeff_data(field), data_g, plan)
        elseif first_is_real
            # First dim is RealFourier: use multi-dim rfft (R2C on dim 1, C2C on rest)
            # BUT if input is already complex, rfft is invalid — use C2C instead
            # (matches CPU path which falls back to fft when data is complex)
            if input_T <: Complex
                plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, input_T; real_input=false)
                local_coeff_shape = local_grid_shape  # C2C preserves shape
            else
                plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, input_T; real_input=true)
                local_coeff_shape = (div(local_grid_shape[1], 2) + 1, local_grid_shape[2:end]...)
            end

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != coeff_T ||
                          size(existing_coeff) != local_coeff_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
            end

            gpu_forward_fft!(get_coeff_data(field), data_g, plan)
        else
            # All ComplexFourier: use multi-dim cfft
            # C2C requires complex input; promote real data if needed
            if input_T <: Real
                fft_input = Complex{input_T}.(data_g)
                plan_T = Complex{input_T}
            else
                fft_input = data_g
                plan_T = input_T
            end
            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, plan_T; real_input=false)

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != plan_T ||
                          size(existing_coeff) != local_grid_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(plan_T, local_grid_shape...))
            end

            gpu_forward_fft!(get_coeff_data(field), fft_input, plan)
        end

        return true

    elseif all_chebyshev && length(bases) == 1
        # Pure Chebyshev 1D case - use GPU DCT
        n = local_grid_shape[1]
        local_coeff_shape = local_grid_shape  # Chebyshev: same shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != input_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(input_T, local_coeff_shape...))
        end

        if input_T <: Complex
            # Complex Chebyshev: apply DCT to real and imaginary parts separately
            real_T = real(input_T)
            dct_plan = plan_gpu_dct(gpu_arch, n, real_T, 1)
            real_part = real.(data_g)
            imag_part = imag.(data_g)
            real_out = similar(real_part)
            imag_out = similar(imag_part)
            gpu_forward_dct_1d!(real_out, real_part, dct_plan)
            gpu_forward_dct_1d!(imag_out, imag_part, dct_plan)
            get_coeff_data(field) .= complex.(real_out, imag_out)
        else
            dct_plan = plan_gpu_dct(gpu_arch, n, input_T, 1)
            gpu_forward_dct_1d!(get_coeff_data(field), data_g, dct_plan)
        end
        return true

    elseif all_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # Pure Chebyshev 2D/3D case - use GPU DCT on all dimensions
        # Coefficient shape equals grid shape for Chebyshev
        local_coeff_shape = local_grid_shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != input_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(input_T, local_coeff_shape...))
        end

        # Apply DCT along each dimension
        temp_data = copy(data_g)
        for dim in 1:length(bases)
            dct_plan = plan_gpu_dct_dim(gpu_arch, size(temp_data), input_T, dim)
            output = CUDA.zeros(input_T, size(temp_data)...)
            gpu_dct_dim!(output, temp_data, dct_plan, Val(:forward))
            temp_data = output
        end

        copyto!(get_coeff_data(field), temp_data)
        return true

    elseif has_fourier && has_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # Mixed Fourier-Chebyshev 2D/3D case
        # Use the mixed transform plan for dimension-by-dimension transforms
        # Supports: Fourier-Chebyshev, Fourier-Fourier-Chebyshev, Fourier-Chebyshev-Chebyshev, etc.

        # Get or create mixed transform plan (determines correct coeff_shape)
        plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)
        local_coeff_shape = plan.coeff_shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != coeff_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
        end

        # Execute mixed transform
        gpu_mixed_forward_transform!(get_coeff_data(field), data_g, plan)

        return true
    end

    # For unsupported combinations (e.g., Legendre), fall back to CPU
    return false
end

"""
    gpu_backward_transform!(field::ScalarField)

GPU-specific backward transform using CUFFT.
Supports:
- Pure Fourier (RealFourier, ComplexFourier) - uses cuFFT
- Pure Chebyshev - uses GPU DCT
- Mixed Fourier-Chebyshev - dimension-by-dimension transforms
"""
function Tarang.gpu_backward_transform!(field::ScalarField)
    arch = field.dist.architecture
    if !Tarang.is_gpu(arch)
        return false
    end

    data_c = get_coeff_data(field)
    if !isa(data_c, CuArray)
        return false
    end

    gpu_arch = arch::GPU

    # Ensure correct device is active for multi-GPU support
    ensure_device!(gpu_arch)

    bases = field.bases
    if isempty(bases)
        return false
    end

    # Use LOCAL coefficient array size to determine grid shape
    # This is critical for distributed computing where each rank owns a portion
    local_coeff_shape = size(data_c)

    # Classify bases
    all_fourier = all(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    all_chebyshev = all(b -> isa(b, ChebyshevT), bases)
    has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    has_chebyshev = any(b -> isa(b, ChebyshevT), bases)

    if all_fourier
        # Pure Fourier case - same logic as forward:
        # Only use multi-dim irfft when bases[1] is RealFourier.
        has_real = any(b -> isa(b, RealFourier), bases)
        first_is_real = isa(bases[1], RealFourier)

        if has_real && !first_is_real
            # RealFourier on non-first dimension: use dimension-by-dimension approach
            existing_grid = get_grid_data(field)
            if existing_grid isa CuArray
                local_grid_shape = size(existing_grid)
            else
                # Estimate from coeff shape: only the first RealFourier dim (in order) was R2C'd
                first_real_found = false
                local_grid_shape = ntuple(length(bases)) do dim
                    if isa(bases[dim], RealFourier) && !first_real_found
                        first_real_found = true
                        return 2 * (local_coeff_shape[dim] - 1)
                    else
                        return local_coeff_shape[dim]
                    end
                end
            end

            real_T = field.dtype
            needs_alloc = existing_grid === nothing ||
                          !(existing_grid isa CuArray) ||
                          eltype(existing_grid) != real_T ||
                          size(existing_grid) != local_grid_shape
            if needs_alloc
                set_grid_data!(field, CUDA.zeros(real_T, local_grid_shape...))
            end

            plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, real_T)
            gpu_mixed_backward_transform!(get_grid_data(field), data_c, plan)

        elseif first_is_real
            # First dim is RealFourier: normally use irfft (C2R on dim 1, C2C on rest).
            # BUT if forward used C2C (complex input), coeff shape == grid shape (no halving),
            # so we must detect this and use C2C inverse instead.
            existing_grid = get_grid_data(field)
            if existing_grid isa CuArray
                local_grid_shape = size(existing_grid)
            else
                local_grid_shape = (2 * (local_coeff_shape[1] - 1), local_coeff_shape[2:end]...)
            end

            # Determine if R2C or C2C was used: if coeff dim 1 == grid dim 1, C2C was used
            used_r2c = (local_coeff_shape[1] != local_grid_shape[1])
            real_T = field.dtype

            if used_r2c
                plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, real_T; real_input=true)
                output_T = real_T
            else
                # C2C inverse: coeff and grid have same shape
                coeff_T_bk = eltype(data_c)
                plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, coeff_T_bk; real_input=false)
                output_T = coeff_T_bk
            end

            if !Tarang.should_use_gpu_fft(field, local_grid_shape)
                return false
            end

            needs_alloc = existing_grid === nothing ||
                          !(existing_grid isa CuArray) ||
                          eltype(existing_grid) != output_T ||
                          size(existing_grid) != local_grid_shape
            if needs_alloc
                set_grid_data!(field, CUDA.zeros(output_T, local_grid_shape...))
            end

            gpu_backward_fft!(get_grid_data(field), data_c, plan)
        else
            # All ComplexFourier: use multi-dim icfft
            local_grid_shape = local_coeff_shape

            if !Tarang.should_use_gpu_fft(field, local_grid_shape)
                return false
            end

            coeff_T = eltype(data_c)
            # C2C inverse requires complex input; promote if needed (shouldn't normally happen)
            if coeff_T <: Real
                fft_input = Complex{coeff_T}.(data_c)
                plan_T = Complex{coeff_T}
            else
                fft_input = data_c
                plan_T = coeff_T
            end
            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, plan_T; real_input=false)

            existing_grid = get_grid_data(field)
            needs_alloc = existing_grid === nothing ||
                          !(existing_grid isa CuArray) ||
                          eltype(existing_grid) != plan_T ||
                          size(existing_grid) != local_grid_shape
            if needs_alloc
                set_grid_data!(field, CUDA.zeros(plan_T, local_grid_shape...))
            end

            gpu_backward_fft!(get_grid_data(field), fft_input, plan)
        end

        return true

    elseif all_chebyshev && length(bases) == 1
        # Pure Chebyshev 1D case - use GPU inverse DCT
        local_grid_shape = local_coeff_shape  # Chebyshev: same shape
        input_T = eltype(data_c)
        n = local_coeff_shape[1]

        existing_grid = get_grid_data(field)
        needs_alloc = existing_grid === nothing ||
                      !(existing_grid isa CuArray) ||
                      eltype(existing_grid) != input_T ||
                      size(existing_grid) != local_grid_shape
        if needs_alloc
            set_grid_data!(field, CUDA.zeros(input_T, local_grid_shape...))
        end

        if input_T <: Complex
            # Complex Chebyshev: apply inverse DCT to real and imaginary parts separately
            real_T = real(input_T)
            dct_plan = plan_gpu_dct(gpu_arch, n, real_T, 1)
            real_part = real.(data_c)
            imag_part = imag.(data_c)
            real_out = similar(real_part)
            imag_out = similar(imag_part)
            gpu_backward_dct_1d!(real_out, real_part, dct_plan)
            gpu_backward_dct_1d!(imag_out, imag_part, dct_plan)
            get_grid_data(field) .= complex.(real_out, imag_out)
        else
            dct_plan = plan_gpu_dct(gpu_arch, n, input_T, 1)
            gpu_backward_dct_1d!(get_grid_data(field), data_c, dct_plan)
        end
        return true

    elseif all_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # Pure Chebyshev 2D/3D case - use GPU DCT on all dimensions (in reverse order)
        local_grid_shape = local_coeff_shape  # For Chebyshev, shapes are equal
        input_T = eltype(data_c)

        existing_grid = get_grid_data(field)
        needs_alloc = existing_grid === nothing ||
                      !(existing_grid isa CuArray) ||
                      eltype(existing_grid) != input_T ||
                      size(existing_grid) != local_grid_shape
        if needs_alloc
            set_grid_data!(field, CUDA.zeros(input_T, local_grid_shape...))
        end

        # Apply inverse DCT along each dimension (reverse order for consistency)
        temp_data = copy(data_c)
        for dim in reverse(1:length(bases))
            dct_plan = plan_gpu_dct_dim(gpu_arch, size(temp_data), input_T, dim)
            output = CUDA.zeros(input_T, size(temp_data)...)
            gpu_dct_dim!(output, temp_data, dct_plan, Val(:backward))
            temp_data = output
        end

        copyto!(get_grid_data(field), temp_data)
        return true

    elseif has_fourier && has_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # Mixed Fourier-Chebyshev 2D/3D case
        # Supports: Fourier-Chebyshev, Fourier-Fourier-Chebyshev, Fourier-Chebyshev-Chebyshev, etc.

        # Determine grid shape: use existing grid data if available,
        # otherwise estimate (only the FIRST RealFourier dim uses R2C).
        existing_grid = get_grid_data(field)
        if existing_grid isa CuArray
            local_grid_shape = size(existing_grid)
        else
            # Estimate: only the first RealFourier dimension was R2C'd
            first_real_found = false
            local_grid_shape = ntuple(length(bases)) do dim
                basis = bases[dim]
                if isa(basis, RealFourier) && !first_real_found
                    first_real_found = true
                    return 2 * (local_coeff_shape[dim] - 1)
                else
                    return local_coeff_shape[dim]
                end
            end
        end

        real_T = field.dtype
        needs_alloc = existing_grid === nothing ||
                      !(existing_grid isa CuArray) ||
                      eltype(existing_grid) != real_T ||
                      size(existing_grid) != local_grid_shape
        if needs_alloc
            set_grid_data!(field, CUDA.zeros(real_T, local_grid_shape...))
        end

        # Get or create mixed transform plan (uses grid_shape as canonical reference)
        input_T = real_T <: Complex ? real_T : real_T
        plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)

        # Execute mixed backward transform
        gpu_mixed_backward_transform!(get_grid_data(field), data_c, plan)

        return true
    end

    # For unsupported combinations (e.g., Legendre), fall back to CPU
    return false
end

# ============================================================================
# GPU FFT Transforms using CUFFT
# ============================================================================

"""
    GPUFFTPlan

Wrapper for CUFFT plans that work with CuArrays.
"""
struct GPUFFTPlan{P, IP}
    plan::P
    iplan::IP
    size::Tuple{Vararg{Int}}
    is_real::Bool
end

"""
    plan_gpu_fft(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Create a GPU FFT plan using CUFFT for element type `T`.

**Important:** `local_size` should be the LOCAL array shape (what this process owns),
not the global domain size. In distributed computing, each rank creates plans
for its local data portion.

For multi-GPU: ensures the plan is created on the correct device.
"""
function plan_gpu_fft(arch::GPU{CuDevice}, local_size::Tuple, T::Type; real_input::Bool=false)
    # Ensure we're on the correct device for plan creation
    ensure_device!(arch)

    complex_T = T <: Complex ? T : Complex{T}

    if real_input
        # Real-to-complex FFT (like rfft)
        # Plan is created based on local array dimensions
        dummy_in = CUDA.zeros(T, local_size...)
        plan = CUFFT.plan_rfft(dummy_in)

        # Complex-to-real inverse FFT (like irfft)
        out_size = (div(local_size[1], 2) + 1, local_size[2:end]...)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, local_size[1])

        return GPUFFTPlan(plan, iplan, local_size, true)
    else
        # Complex-to-complex FFT
        dummy = CUDA.zeros(complex_T, local_size...)
        plan = CUFFT.plan_fft(dummy)
        iplan = CUFFT.plan_ifft(dummy)

        return GPUFFTPlan(plan, iplan, local_size, false)
    end
end

# Fallback for generic GPU: delegates to device-specific version using current device,
# ensuring proper device context via ensure_device!
plan_gpu_fft(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false) =
    plan_gpu_fft(GPU{CuDevice}(CUDA.device()), local_size, T; real_input=real_input)

"""
    gpu_forward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)

Execute forward FFT on GPU.
"""
function gpu_forward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    mul!(output, plan.plan, input)
    return output
end

"""
    gpu_backward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)

Execute backward (inverse) FFT on GPU.
"""
function gpu_backward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    if plan.is_real
        # irfft returns real output
        mul!(output, plan.iplan, input)
    else
        mul!(output, plan.iplan, input)
    end
    return output
end

# ============================================================================
# GPU Transform Plan Cache
# ============================================================================

"""
    GPUTransformCache

Cache for GPU FFT plans to avoid recreation overhead.
Keys include device ID for multi-GPU safety.
"""
struct GPUTransformCache
    plans::Dict{Tuple, GPUFFTPlan}
end

const GPU_TRANSFORM_CACHE = GPUTransformCache(Dict{Tuple, GPUFFTPlan}())

"""
    get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Get or create a cached GPU FFT plan.
Plans are cached by (device_id, size, element_type, real_input).
"""
function get_gpu_fft_plan(arch::GPU{CuDevice}, local_size::Tuple, T::Type; real_input::Bool=false)
    device_id = CUDA.deviceid(arch.device)
    key = (device_id, local_size, T, real_input)

    if !haskey(GPU_TRANSFORM_CACHE.plans, key)
        GPU_TRANSFORM_CACHE.plans[key] = plan_gpu_fft(arch, local_size, T; real_input=real_input)
    end
    return GPU_TRANSFORM_CACHE.plans[key]
end

# Fallback for generic GPU
function get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)
    device_id = CUDA.deviceid()
    key = (device_id, local_size, T, real_input)

    if !haskey(GPU_TRANSFORM_CACHE.plans, key)
        GPU_TRANSFORM_CACHE.plans[key] = plan_gpu_fft(arch, local_size, T; real_input=real_input)
    end
    return GPU_TRANSFORM_CACHE.plans[key]
end

"""
    clear_gpu_transform_cache!()

Clear all cached GPU transform plans.
"""
function clear_gpu_transform_cache!()
    empty!(GPU_TRANSFORM_CACHE.plans)
end
