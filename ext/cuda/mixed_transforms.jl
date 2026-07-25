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

"""
    _mixed_stage_eltype(ops, dim, T) -> Type

Element type the plan for axis `dim` must be built for: real only while no
earlier stage has produced complex data (the shared `AxisOp` chain records that),
complex from the first FFT onward.
"""
function _mixed_stage_eltype(ops, dim::Int, ::Type{T}) where {T}
    complex_in = T <: Complex
    for ax in 1:(dim - 1)
        complex_in |= ops[ax].out_complex
    end
    RT = T <: Complex ? real(T) : T
    return complex_in ? Complex{RT} : T
end

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

    # Transform order: Fourier first (they may change array size), then Chebyshev.
    # Chebyshev stages preserve shape and realness here (a SCALED Chebyshev axis is
    # refused upstream), so ordering them last does not affect shape bookkeeping.
    sort!(fourier_dims)
    transform_order = vcat(fourier_dims, chebyshev_dims)

    # Which axis is R2C, and the resulting coefficient shape, come from the SHARED
    # layout rules (src/core/transforms/transform_layout.jl) — the same code
    # `coefficient_shape` and the CPU stage specs use. Deriving them here instead
    # is what let this plan disagree with the rest of the framework: it sized the
    # coefficient buffer from the GRID shape, so a scaled Chebyshev axis silently
    # kept every grid mode the CPU chain truncates.
    ops, coeff_shape, _ = Tarang.forward_layout(bases, local_grid_shape, T)

    fft_plans = Dict{Int, GPUFFTPlanDim}()
    current_shape = local_grid_shape
    for dim in fourier_dims
        op = ops[dim]
        use_real = op.op === :rfft
        plan_T = _mixed_stage_eltype(ops, dim, T)
        fft_plans[dim] = plan_gpu_fft_dim(arch, current_shape, plan_T, dim; real_input=use_real)
        current_shape = ntuple(i -> i == dim ? op.out_len : current_shape[i], ndims)
    end

    # Mixed transforms use the cached DCT-I implementation from cheb_deriv.jl.
    # Keep this field for plan compatibility, but do not allocate the obsolete
    # DCT-II/III twiddle plans.
    dct_plans = Dict{Int, GPUDCTPlanDim}()

    return GPUMixedTransformPlan(
        basis_types, transform_order,
        fft_plans, dct_plans,
        local_grid_shape, coeff_shape
    )
end

plan_gpu_mixed_transform(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type) =
    plan_gpu_mixed_transform(GPU{CuDevice}(CUDA.device()), bases, local_grid_shape, T)

# ============================================================================
# Mixed Transform Caching (Thread-Safe)
# ============================================================================

"""
Thread-safe cache for GPU mixed transform plans.
Uses a ReentrantLock to protect concurrent access from multiple Julia threads.
"""
struct GPUMixedTransformCache
    plans::Dict{Tuple, GPUMixedTransformPlan}
    lock::ReentrantLock
end

const GPU_MIXED_TRANSFORM_CACHE = GPUMixedTransformCache(Dict{Tuple, GPUMixedTransformPlan}(), ReentrantLock())

struct GPUMixedTransformScratch{CA,RA}
    complex_a::CA
    complex_b::CA
    real_input::RA
    real_output::RA
    imag_output::RA
end

const GPU_MIXED_SCRATCH_CACHE = Dict{Tuple,Any}()

function get_gpu_mixed_transform_scratch(plan::GPUMixedTransformPlan,
                                         ::Type{CT}) where {CT<:Complex}
    RT = real(CT)
    key = (_current_device_id(), plan.grid_shape, plan.coeff_shape, CT)
    scratch = lock(GPU_MIXED_TRANSFORM_CACHE.lock) do
        get!(GPU_MIXED_SCRATCH_CACHE, key) do
            complex_a = CUDA.zeros(CT, plan.coeff_shape...)
            complex_b = CUDA.zeros(CT, plan.coeff_shape...)
            GPUMixedTransformScratch(
                complex_a,
                complex_b,
                CUDA.zeros(RT, plan.coeff_shape...),
                CUDA.zeros(RT, plan.coeff_shape...),
                CUDA.zeros(RT, plan.coeff_shape...),
            )
        end
    end
    return scratch::GPUMixedTransformScratch
end

@inline function _next_mixed_complex_buffer(current, scratch::GPUMixedTransformScratch)
    return current === scratch.complex_a ? scratch.complex_b : scratch.complex_a
end

"""
    _mixed_plan_key(arch, bases, local_grid_shape, T)

Generate cache key for mixed transform plans.
"""
function _mixed_plan_key(arch::GPU{CuDevice}, bases::Tuple, local_grid_shape::Tuple, T::Type)
    basis_types = tuple([typeof(b) for b in bases]...)
    return (CUDA.deviceid(arch.device), basis_types, local_grid_shape, T)
end

_mixed_plan_key(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type) =
    (_current_device_id(), tuple([typeof(b) for b in bases]...), local_grid_shape, T)

"""
    get_gpu_mixed_transform_plan(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type)

Get or create a cached GPU mixed transform plan (thread-safe).
"""
function get_gpu_mixed_transform_plan(arch::GPU, bases::Tuple, local_grid_shape::Tuple, T::Type)
    key = _mixed_plan_key(arch, bases, local_grid_shape, T)
    lock(GPU_MIXED_TRANSFORM_CACHE.lock) do
        if !haskey(GPU_MIXED_TRANSFORM_CACHE.plans, key)
            GPU_MIXED_TRANSFORM_CACHE.plans[key] = plan_gpu_mixed_transform(arch, bases, local_grid_shape, T)
        end
        return GPU_MIXED_TRANSFORM_CACHE.plans[key]
    end
end

"""
    clear_gpu_mixed_transform_cache!()

Clear all cached GPU mixed transform plans (thread-safe).
"""
function clear_gpu_mixed_transform_cache!()
    lock(GPU_MIXED_TRANSFORM_CACHE.lock) do
        empty!(GPU_MIXED_TRANSFORM_CACHE.plans)
        empty!(GPU_MIXED_SCRATCH_CACHE)
    end
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
    complex_T = T <: Complex ? T : Complex{T}
    scratch = get_gpu_mixed_transform_scratch(plan, complex_T)

    current_data = grid_data

    for dim in plan.transform_order
        basis_type = plan.basis_types[dim]

        if basis_type == :fourier_real || basis_type == :fourier_complex
            fft_plan = plan.fft_plans[dim]
            output = _next_mixed_complex_buffer(current_data, scratch)

            if fft_plan.is_real
                gpu_fft_dim!(output, current_data, fft_plan)
            else
                if eltype(current_data) <: Real
                    input = output === scratch.complex_b ? scratch.complex_a : scratch.complex_b
                    input .= complex.(current_data)
                    gpu_fft_dim!(output, input, fft_plan)
                else
                    gpu_fft_dim!(output, current_data, fft_plan)
                end
            end

            current_data = output

        elseif basis_type == :chebyshev
            if eltype(current_data) <: Complex
                scratch.real_input .= real.(current_data)
                gpu_dct1_along_dim!(scratch.real_output, scratch.real_input, dim, :forward)
                scratch.real_input .= imag.(current_data)
                gpu_dct1_along_dim!(scratch.imag_output, scratch.real_input, dim, :forward)
                output = _next_mixed_complex_buffer(current_data, scratch)
                output .= complex.(scratch.real_output, scratch.imag_output)
                current_data = output
            else
                scratch.real_input .= current_data
                gpu_dct1_along_dim!(scratch.real_output, scratch.real_input, dim, :forward)
                current_data = scratch.real_output
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
    complex_T = S <: Complex ? S : Complex{S}
    scratch = get_gpu_mixed_transform_scratch(plan, complex_T)
    current_data = coeff_data

    for dim in reverse(plan.transform_order)
        basis_type = plan.basis_types[dim]

        if basis_type == :chebyshev
            if eltype(current_data) <: Complex
                scratch.real_input .= real.(current_data)
                gpu_dct1_along_dim!(scratch.real_output, scratch.real_input, dim, :backward)
                scratch.real_input .= imag.(current_data)
                gpu_dct1_along_dim!(scratch.imag_output, scratch.real_input, dim, :backward)
                output = _next_mixed_complex_buffer(current_data, scratch)
                output .= complex.(scratch.real_output, scratch.imag_output)
                current_data = output
            else
                scratch.real_input .= current_data
                gpu_dct1_along_dim!(scratch.real_output, scratch.real_input, dim, :backward)
                current_data = scratch.real_output
            end

        elseif basis_type == :fourier_real || basis_type == :fourier_complex
            fft_plan = plan.fft_plans[dim]

            if fft_plan.is_real
                gpu_ifft_dim!(grid_data, current_data, fft_plan)
                current_data = grid_data
            else
                output = _next_mixed_complex_buffer(current_data, scratch)
                if eltype(current_data) <: Real
                    input = output === scratch.complex_b ? scratch.complex_a : scratch.complex_b
                    input .= complex.(current_data)
                    gpu_ifft_dim!(output, input, fft_plan)
                else
                    gpu_ifft_dim!(output, current_data, fft_plan)
                end
                current_data = output
            end
        end
    end

    if current_data === grid_data
        return grid_data
    elseif T <: Real
        copyto!(grid_data, real.(current_data))
    else
        copyto!(grid_data, current_data)
    end
    return grid_data
end
