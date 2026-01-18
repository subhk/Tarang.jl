# ============================================================================
# Mixed Fourier-Chebyshev Transform Plans
# ============================================================================

"""
    GPUMixedTransformPlan

Plan for mixed Fourier-Chebyshev transforms on GPU.
Stores per-dimension transform information for sequential dimension-by-dimension transforms.
"""
struct GPUMixedTransformPlan
    basis_types::Vector{Symbol}  # :fourier_real, :fourier_complex, :chebyshev
    transform_order::Vector{Int}  # Order to apply transforms (Fourier first recommended)
    fft_plans::Dict{Int, GPUFFTPlanDim}  # FFT plans by dimension
    dct_plans::Dict{Int, GPUDCTPlanDim}  # DCT plans by dimension
    grid_shape::Tuple{Vararg{Int}}
    coeff_shape::Tuple{Vararg{Int}}
end

"""
    plan_gpu_mixed_transform(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type)

Create a GPU plan for mixed Fourier-Chebyshev transforms.

# Arguments
- `arch`: GPU architecture
- `bases`: Tuple of basis objects (RealFourier, ComplexFourier, ChebyshevT)
- `local_grid_shape`: LOCAL grid shape (for this MPI rank)
- `T`: Element type

# Strategy
For Fourier(x) × Chebyshev(z), we decompose along x (Fourier) only.
Transform order: Fourier dimensions first, then Chebyshev.
This minimizes data movement since Chebyshev is local.
"""
function plan_gpu_mixed_transform(arch::GPU{CuDevice}, bases::Tuple, local_grid_shape::Tuple, T::Type)
    ensure_device!(arch)

    ndims = length(bases)
    @assert ndims == length(local_grid_shape) "Basis count must match grid dimensions"

    basis_types = Symbol[]
    fourier_dims = Int[]
    chebyshev_dims = Int[]

    # Classify each dimension
    for (dim, basis) in enumerate(bases)
        if isa(basis, RealFourier)
            push!(basis_types, :fourier_real)
            push!(fourier_dims, dim)
        elseif isa(basis, ComplexFourier)
            push!(basis_types, :fourier_complex)
            push!(fourier_dims, dim)
        elseif isa(basis, ChebyshevT)
            push!(basis_types, :chebyshev)
            push!(chebyshev_dims, dim)
        else
            error("Unsupported basis type: $(typeof(basis))")
        end
    end

    # Transform order: Fourier first (they may change array size), then Chebyshev
    transform_order = vcat(fourier_dims, chebyshev_dims)

    # Create FFT plans for Fourier dimensions
    fft_plans = Dict{Int, GPUFFTPlanDim}()
    current_shape = local_grid_shape

    for dim in fourier_dims
        is_real = basis_types[dim] == :fourier_real
        # Plan for current shape (may change as we apply transforms)
        fft_plans[dim] = plan_gpu_fft_dim(arch, current_shape, T, dim; real_input=is_real)

        # Update shape for next dimension's plan if this is a real FFT
        if is_real
            current_shape = ntuple(i -> i == dim ? div(current_shape[i], 2) + 1 : current_shape[i], ndims)
        end
    end

    # Compute coefficient shape
    coeff_shape = ntuple(ndims) do dim
        if basis_types[dim] == :fourier_real
            div(local_grid_shape[dim], 2) + 1
        else
            local_grid_shape[dim]
        end
    end

    # Create DCT plans for Chebyshev dimensions (use coefficient shape)
    dct_plans = Dict{Int, GPUDCTPlanDim}()
    for dim in chebyshev_dims
        dct_plans[dim] = plan_gpu_dct_dim(arch, coeff_shape, T, dim)
    end

    return GPUMixedTransformPlan(
        basis_types, transform_order,
        fft_plans, dct_plans,
        local_grid_shape, coeff_shape
    )
end

plan_gpu_mixed_transform(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type) =
    plan_gpu_mixed_transform(GPU{CuDevice}(CUDA.device()), bases, local_grid_shape, T)

# ============================================================================
# Mixed Transform Caching
# ============================================================================

const GPU_MIXED_TRANSFORM_CACHE = Dict{Tuple, GPUMixedTransformPlan}()

"""
    _mixed_plan_key(arch, bases, local_grid_shape, T)

Generate cache key for mixed transform plans.
"""
function _mixed_plan_key(arch::GPU{CuDevice}, bases::Tuple, local_grid_shape::Tuple, T::Type)
    basis_types = tuple([typeof(b) for b in bases]...)
    return (CUDA.deviceid(arch.device), basis_types, local_grid_shape, T)
end

_mixed_plan_key(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type) =
    (CUDA.deviceid(), tuple([typeof(b) for b in bases]...), local_grid_shape, T)

"""
    get_gpu_mixed_transform_plan(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type)

Get or create a cached GPU mixed transform plan.
"""
function get_gpu_mixed_transform_plan(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type)
    key = _mixed_plan_key(arch, bases, local_grid_shape, T)
    if !haskey(GPU_MIXED_TRANSFORM_CACHE, key)
        GPU_MIXED_TRANSFORM_CACHE[key] = plan_gpu_mixed_transform(arch, bases, local_grid_shape, T)
    end
    return GPU_MIXED_TRANSFORM_CACHE[key]
end

"""
    clear_gpu_mixed_transform_cache!()

Clear all cached GPU mixed transform plans.
"""
function clear_gpu_mixed_transform_cache!()
    empty!(GPU_MIXED_TRANSFORM_CACHE)
end

# ============================================================================
# Mixed Transform Execution (N-dimensional)
# ============================================================================

"""
    gpu_mixed_forward_transform!(coeff_data::CuArray, grid_data::CuArray, plan::GPUMixedTransformPlan)

Execute forward mixed Fourier-Chebyshev transform on GPU.
Supports both 2D and 3D arrays.

Transforms grid-space data to spectral coefficients:
1. Apply FFT along Fourier dimensions (may change array size for real FFT)
2. Apply DCT along Chebyshev dimensions
"""
function gpu_mixed_forward_transform!(coeff_data::CuArray{T, N}, grid_data::CuArray{S, N},
                                      plan::GPUMixedTransformPlan) where {T, S, N}
    # Determine complex type for intermediate results
    complex_T = T <: Complex ? T : Complex{T}

    # Work through dimensions in order
    current_data = grid_data

    for dim in plan.transform_order
        basis_type = plan.basis_types[dim]

        if basis_type == :fourier_real || basis_type == :fourier_complex
            # FFT along this dimension
            fft_plan = plan.fft_plans[dim]

            if basis_type == :fourier_real
                # R2C transform changes size along this dimension
                out_shape = ntuple(i -> i == dim ? div(size(current_data, i), 2) + 1 : size(current_data, i), N)
                output = CUDA.zeros(complex_T, out_shape...)
            else
                output = CUDA.zeros(complex_T, size(current_data)...)
            end

            gpu_fft_dim!(output, current_data, fft_plan)
            current_data = output

        elseif basis_type == :chebyshev
            # DCT along this dimension
            dct_plan = plan.dct_plans[dim]

            # DCT works on real data - if we have complex from FFT, process real and imag separately
            if eltype(current_data) <: Complex
                output_real = CUDA.zeros(real(eltype(current_data)), size(current_data)...)
                output_imag = CUDA.zeros(real(eltype(current_data)), size(current_data)...)

                # Extract real and imaginary parts
                input_real = real.(current_data)
                input_imag = imag.(current_data)

                # DCT on each
                gpu_dct_dim!(output_real, input_real, dct_plan, Val(:forward))
                gpu_dct_dim!(output_imag, input_imag, dct_plan, Val(:forward))

                # Combine back to complex
                current_data = complex.(output_real, output_imag)
            else
                output = CUDA.zeros(eltype(current_data), size(current_data)...)
                gpu_dct_dim!(output, current_data, dct_plan, Val(:forward))
                current_data = output
            end
        end
    end

    # Copy to output
    copyto!(coeff_data, current_data)
    return coeff_data
end

"""
    gpu_mixed_backward_transform!(grid_data::CuArray, coeff_data::CuArray, plan::GPUMixedTransformPlan)

Execute backward mixed Fourier-Chebyshev transform on GPU.
Supports both 2D and 3D arrays.

Transforms spectral coefficients to grid-space data:
1. Apply inverse DCT along Chebyshev dimensions
2. Apply inverse FFT along Fourier dimensions
"""
function gpu_mixed_backward_transform!(grid_data::CuArray{T, N}, coeff_data::CuArray{S, N},
                                       plan::GPUMixedTransformPlan) where {T, S, N}
    # Work through dimensions in REVERSE order
    current_data = coeff_data

    for dim in reverse(plan.transform_order)
        basis_type = plan.basis_types[dim]

        if basis_type == :chebyshev
            # Inverse DCT along this dimension
            dct_plan = plan.dct_plans[dim]

            if eltype(current_data) <: Complex
                output_real = CUDA.zeros(real(eltype(current_data)), size(current_data)...)
                output_imag = CUDA.zeros(real(eltype(current_data)), size(current_data)...)

                input_real = real.(current_data)
                input_imag = imag.(current_data)

                gpu_dct_dim!(output_real, input_real, dct_plan, Val(:backward))
                gpu_dct_dim!(output_imag, input_imag, dct_plan, Val(:backward))

                current_data = complex.(output_real, output_imag)
            else
                output = CUDA.zeros(eltype(current_data), size(current_data)...)
                gpu_dct_dim!(output, current_data, dct_plan, Val(:backward))
                current_data = output
            end

        elseif basis_type == :fourier_real || basis_type == :fourier_complex
            # Inverse FFT along this dimension
            fft_plan = plan.fft_plans[dim]

            if basis_type == :fourier_real
                # C2R transform changes size back
                out_shape = plan.grid_shape
                output = CUDA.zeros(T, out_shape...)
            else
                output = CUDA.zeros(eltype(current_data), size(current_data)...)
            end

            gpu_ifft_dim!(output, current_data, fft_plan)
            current_data = output
        end
    end

    # Copy to output
    copyto!(grid_data, real.(current_data))
    return grid_data
end
