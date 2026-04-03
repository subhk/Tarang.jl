"""
Nonlinear term evaluation using PencilArrays and PencilFFTs

This module implements efficient evaluation of nonlinear terms in spectral methods,
designed for Julia with PencilArrays/PencilFFTs.
Supports both 2D and 3D parallelization with proper dealiasing.

Key features:
- Transform-based multiplication for nonlinear terms (u·∇u, etc.)
- Automatic dealiasing using 3/2 rule
- MPI parallelization through PencilArrays
- Efficient memory management and reuse
- Support for various nonlinear operators
"""

# Note: PencilArrays, PencilFFTs, MPI, LinearAlgebra, FFTW are already imported in Tarang.jl

# Performance monitoring (defined first as it's used by NonlinearEvaluator)
mutable struct NonlinearPerformanceStats
    total_evaluations::Int
    total_time::Float64
    dealiasing_time::Float64
    transform_time::Float64

    function NonlinearPerformanceStats()
        new(0, 0.0, 0.0, 0.0)
    end
end

# Nonlinear operator types
abstract type NonlinearOperator <: Operator end

struct AdvectionOperator <: NonlinearOperator
    velocity::VectorField
    scalar::ScalarField
    name::String
    
    function AdvectionOperator(velocity::VectorField, scalar::ScalarField, name::String="advection")
        new(velocity, scalar, name)
    end
end

struct NonlinearAdvectionOperator <: NonlinearOperator
    velocity::VectorField
    name::String
    
    function NonlinearAdvectionOperator(velocity::VectorField, name::String="nonlinear_advection")
        new(velocity, name)
    end
end

struct ConvectiveOperator <: NonlinearOperator
    field1::Union{ScalarField, VectorField}
    field2::Union{ScalarField, VectorField}
    operation::Symbol  # :multiply, :dot_product, :cross_product
    name::String
    
    function ConvectiveOperator(field1, field2, operation::Symbol, name::String="convective")
        new(field1, field2, operation, name)
    end
end

# Nonlinear evaluation engine
mutable struct NonlinearEvaluator <: AbstractNonlinearEvaluator
    dist::Distributor
    pencil_transforms::Dict{String, Any}
    dealiasing_factor::Float64
    temp_fields::Dict{String, ScalarField}
    memory_pool::Vector{PencilArrays.PencilArray}
    scratch_arrays::Vector{AbstractArray}
    performance_stats::NonlinearPerformanceStats

    function NonlinearEvaluator(dist::Distributor; dealiasing_factor::Float64=3.0/2.0)
        evaluator = new(dist, Dict{String, Any}(), dealiasing_factor, Dict{String, ScalarField}(), PencilArrays.PencilArray[],
                       AbstractArray[], NonlinearPerformanceStats())
        setup_nonlinear_transforms!(evaluator)
        return evaluator
    end
end

# Architecture helper functions for NonlinearEvaluator
"""
    architecture(evaluator::NonlinearEvaluator)

Get the architecture (CPU or GPU) for the nonlinear evaluator.
"""
architecture(evaluator::NonlinearEvaluator) = evaluator.dist.architecture

"""
    is_gpu(evaluator::NonlinearEvaluator)

Check if the nonlinear evaluator is using GPU architecture.
"""
is_gpu(evaluator::NonlinearEvaluator) = is_gpu(evaluator.dist.architecture)

# ============================================================================
# Padded Dealiasing (3/2-rule) — GPU and MPI compatible
# ============================================================================

"""
    PaddedDealiasingWorkspace{T, A}

Pre-allocated workspace for proper 3/2-rule dealiasing (Orszag 1971).
Works on CPU, GPU, and MPI-distributed (PencilArray) data.

Type parameters:
- `T`: Float type (Float32 or Float64)
- `A`: Array type (Array{Complex{T}} for CPU, CuArray{Complex{T}} for GPU)

For MPI-distributed data, the workspace operates on each rank's LOCAL array
(parent of PencilArray), padding only non-decomposed Fourier dimensions.
"""
mutable struct PaddedDealiasingWorkspace{T<:AbstractFloat, A<:AbstractArray{Complex{T}}}
    original_shape::Tuple{Vararg{Int}}
    padded_shape::Tuple{Vararg{Int}}
    fourier_dims::Vector{Int}

    # Pre-allocated padded arrays (on same architecture as input data)
    padded1::A
    padded2::A
    padded_product::A

    # FFT plans: FFTW.MEASURE for CPU, plain plan_fft for GPU
    plan_forward::Any
    plan_backward::Any

    # Architecture for dispatch
    arch::AbstractArchitecture
end

"""
    _get_padded_workspace!(evaluator, bases, dtype; local_shape=nothing,
                           local_fourier_dims=nothing, arch=nothing)

Get or create a cached padded dealiasing workspace.

For serial fields, uses the full basis shape and all Fourier dimensions.
For MPI-distributed fields, pass `local_shape` (the per-rank array shape) and
`local_fourier_dims` (only the non-decomposed Fourier dimensions to pad).
"""
function _get_padded_workspace!(evaluator::NonlinearEvaluator, bases::Tuple, dtype::Type{T};
                                local_shape::Union{Nothing, Tuple}=nothing,
                                local_fourier_dims::Union{Nothing, Vector{Int}}=nothing,
                                arch::Union{Nothing, AbstractArchitecture}=nothing) where T
    _arch = arch !== nothing ? arch : evaluator.dist.architecture
    key = "padded_$(hash(bases))_$(dtype)_$(hash(local_shape))_$(is_gpu(_arch))"
    if haskey(evaluator.pencil_transforms, key)
        return evaluator.pencil_transforms[key]
    end

    factor = evaluator.dealiasing_factor

    # Use local_shape if provided (MPI), otherwise compute from bases
    if local_shape !== nothing
        original_shape = collect(local_shape)
        fourier_dims = local_fourier_dims !== nothing ? local_fourier_dims : Int[]
    else
        fourier_dims = Int[]
        original_shape = Int[]
        for (i, basis) in enumerate(bases)
            N = basis.meta.size
            push!(original_shape, N)
            if isa(basis, Union{RealFourier, ComplexFourier})
                push!(fourier_dims, i)
            end
        end
    end

    if isempty(fourier_dims)
        return nothing
    end

    # Skip padded dealiasing for grids too small to benefit from it.
    # With N ≤ 4, there are at most 2 independent modes (DC + Nyquist) —
    # the 3/2-rule padding would produce incorrect results.
    min_fourier_size = minimum(original_shape[d] for d in fourier_dims)
    if min_fourier_size <= 4
        return nothing
    end

    # Compute padded shape: pad only Fourier dimensions
    padded_shape = copy(original_shape)
    for d in fourier_dims
        M = Int(ceil(factor * original_shape[d]))
        if isodd(M); M += 1; end
        padded_shape[d] = M
    end

    orig_t = Tuple(original_shape)
    pad_t = Tuple(padded_shape)

    # Allocate padded arrays on the correct architecture
    padded1 = zeros(_arch, Complex{T}, pad_t...)
    padded2 = zeros(_arch, Complex{T}, pad_t...)
    padded_product = zeros(_arch, Complex{T}, pad_t...)

    # Create FFT plans — CPU gets FFTW.MEASURE, GPU gets plain plan_fft
    if is_gpu(_arch)
        # For GPU: AbstractFFTs.plan_fft dispatches to CUFFT (no flags arg)
        plan_forward = plan_fft(padded1, fourier_dims)
        plan_backward = plan_ifft(padded1, fourier_dims)
    else
        plan_forward = FFTW.plan_fft(padded1, fourier_dims; flags=FFTW.MEASURE)
        plan_backward = FFTW.plan_ifft(padded1, fourier_dims; flags=FFTW.MEASURE)
    end

    ws = PaddedDealiasingWorkspace{T, typeof(padded1)}(
        orig_t, pad_t, fourier_dims,
        padded1, padded2, padded_product,
        plan_forward, plan_backward, _arch
    )
    evaluator.pencil_transforms[key] = ws
    return ws
end

# ============================================================================
# Spectral Padding / Truncation — slice-based for GPU compatibility
# ============================================================================

"""Compute positive and negative frequency index ranges for padding/truncation."""
function _freq_ranges(N::Int, M::Int, is_fourier::Bool)
    if !is_fourier
        return (1:N, 1:N, 1:0, 1:0)  # (orig_pos, pad_pos, orig_neg, pad_neg) — full copy, no neg
    end
    Nh = N ÷ 2
    pos_orig = 1:Nh+1
    pos_pad = 1:Nh+1
    n_neg = N - Nh - 1  # number of negative frequencies
    if n_neg > 0
        neg_orig = N-n_neg+1:N
        neg_pad = M-n_neg+1:M
    else
        neg_orig = 1:0  # empty
        neg_pad = 1:0
    end
    return (pos_orig, pos_pad, neg_orig, neg_pad)
end

"""
    _pad_spectral!(padded, spec_data, original_shape, padded_shape, fourier_dims)

Zero-pad spectral coefficients. Uses slice-based operations for GPU compatibility.
"""
function _pad_spectral!(padded::AbstractArray{Complex{T}}, spec_data::AbstractArray{Complex{T}},
                        original_shape::Tuple, padded_shape::Tuple,
                        fourier_dims::Vector{Int}) where T
    fill!(padded, zero(Complex{T}))
    ndim = length(original_shape)

    if ndim == 1
        r = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        padded[r[2]] .= spec_data[r[1]]
        if length(r[3]) > 0
            padded[r[4]] .= spec_data[r[3]]
        end
    elseif ndim == 2
        _pad_spectral_sliced_2d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    elseif ndim == 3
        _pad_spectral_sliced_3d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    end
end

function _pad_spectral_sliced_2d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
    r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)

    # Copy positive-positive block
    padded[r1[2], r2[2]] .= spec_data[r1[1], r2[1]]
    # Copy positive-negative block
    if length(r2[3]) > 0
        padded[r1[2], r2[4]] .= spec_data[r1[1], r2[3]]
    end
    # Copy negative-positive block
    if length(r1[3]) > 0
        padded[r1[4], r2[2]] .= spec_data[r1[3], r2[1]]
    end
    # Copy negative-negative block
    if length(r1[3]) > 0 && length(r2[3]) > 0
        padded[r1[4], r2[4]] .= spec_data[r1[3], r2[3]]
    end
end

function _pad_spectral_sliced_3d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
    r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)
    r3 = _freq_ranges(original_shape[3], padded_shape[3], 3 in fourier_dims)

    # Copy all 8 quadrant combinations (pos/neg for each of 3 dims)
    for (s1, d1) in ((r1[1], r1[2]), (r1[3], r1[4]))
        length(s1) == 0 && continue
        for (s2, d2) in ((r2[1], r2[2]), (r2[3], r2[4]))
            length(s2) == 0 && continue
            for (s3, d3) in ((r3[1], r3[2]), (r3[3], r3[4]))
                length(s3) == 0 && continue
                padded[d1, d2, d3] .= spec_data[s1, s2, s3]
            end
        end
    end
end

"""
    _truncate_spectral!(result, padded_spec, original_shape, padded_shape, fourier_dims)

Truncate padded spectral coefficients back to original size.
Uses slice-based operations for GPU compatibility.
"""
function _truncate_spectral!(result::AbstractArray{Complex{T}}, padded_spec::AbstractArray{Complex{T}},
                             original_shape::Tuple, padded_shape::Tuple,
                             fourier_dims::Vector{Int}) where T
    ndim = length(original_shape)

    if ndim == 1
        r = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        result[r[1]] .= padded_spec[r[2]]
        if length(r[3]) > 0
            result[r[3]] .= padded_spec[r[4]]
        end
    elseif ndim == 2
        r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)
        result[r1[1], r2[1]] .= padded_spec[r1[2], r2[2]]
        if length(r2[3]) > 0
            result[r1[1], r2[3]] .= padded_spec[r1[2], r2[4]]
        end
        if length(r1[3]) > 0
            result[r1[3], r2[1]] .= padded_spec[r1[4], r2[2]]
        end
        if length(r1[3]) > 0 && length(r2[3]) > 0
            result[r1[3], r2[3]] .= padded_spec[r1[4], r2[4]]
        end
    elseif ndim == 3
        r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)
        r3 = _freq_ranges(original_shape[3], padded_shape[3], 3 in fourier_dims)
        for (d1, s1) in ((r1[1], r1[2]), (r1[3], r1[4]))
            length(d1) == 0 && continue
            for (d2, s2) in ((r2[1], r2[2]), (r2[3], r2[4]))
                length(d2) == 0 && continue
                for (d3, s3) in ((r3[1], r3[2]), (r3[3], r3[4]))
                    length(d3) == 0 && continue
                    result[d1, d2, d3] .= padded_spec[s1, s2, s3]
                end
            end
        end
    end
end

# ============================================================================
# Padded Multiply — architecture-aware
# ============================================================================

"""
    evaluate_padded_multiply(field1, field2, evaluator, ws)

Multiply two fields with proper 3/2-rule padded dealiasing.
Works on CPU and GPU. For MPI data, operates on the local array.
"""
function evaluate_padded_multiply(field1::ScalarField, field2::ScalarField,
                                  evaluator::NonlinearEvaluator,
                                  ws::PaddedDealiasingWorkspace{T}) where T
    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)

    data1 = get_grid_data(field1)
    data2 = get_grid_data(field2)

    # Handle PencilArray: extract local data for the padded operation
    is_pencil = isa(data1, PencilArrays.PencilArray)
    raw1 = is_pencil ? parent(data1) : data1
    raw2 = is_pencil ? parent(data2) : data2

    # Move to workspace architecture if needed
    raw1_ws = on_architecture(ws.arch, raw1)
    raw2_ws = on_architecture(ws.arch, raw2)

    # Step 1: FFT to spectral along Fourier dimensions
    # fft/ifft dispatch to CUFFT for GPU arrays via AbstractFFTs
    spec1 = fft(Complex{T}.(raw1_ws), ws.fourier_dims)
    spec2 = fft(Complex{T}.(raw2_ws), ws.fourier_dims)

    # Step 2: Pad spectral coefficients
    _pad_spectral!(ws.padded1, spec1, ws.original_shape, ws.padded_shape, ws.fourier_dims)
    _pad_spectral!(ws.padded2, spec2, ws.original_shape, ws.padded_shape, ws.fourier_dims)

    # Step 3: IFFT to padded grid (using pre-computed plan)
    ws.padded1 .= ws.plan_backward * ws.padded1
    ws.padded2 .= ws.plan_backward * ws.padded2

    # Step 4: Multiply on padded grid
    ws.padded_product .= ws.padded1 .* ws.padded2

    # Step 5: FFT product back
    ws.padded_product .= ws.plan_forward * ws.padded_product

    # Step 6: Truncate to original coefficients
    spec_result = zeros(ws.arch, Complex{T}, ws.original_shape...)
    _truncate_spectral!(spec_result, ws.padded_product, ws.original_shape, ws.padded_shape, ws.fourier_dims)

    # Step 7: IFFT to grid and normalize
    # Normalization: padded IFFT divides by M, but we want result on N-grid.
    # Scale by M/N per padded Fourier dimension.
    scale = one(T)
    for d in ws.fourier_dims
        scale *= T(ws.padded_shape[d]) / T(ws.original_shape[d])
    end
    grid_result = ifft(spec_result, ws.fourier_dims)

    # Write result to output field
    result = get_temp_field(evaluator, field1, "product_$(field1.name)_$(field2.name)")
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)

    # Extract the correctly-typed grid values
    scaled_result = if field1.dtype <: Real
        real.(grid_result) .* scale
    else
        grid_result .* scale
    end

    # Write back — handle PencilArray wrapper
    if is_pencil
        parent(result_data) .= on_architecture(ws.arch, scaled_result)
    else
        result_data .= scaled_result
    end

    return result
end

"""
    _get_local_fourier_dims(evaluator, bases)

For MPI-distributed fields, return only the Fourier dimensions that are
fully local (not decomposed across MPI ranks). These can be safely padded.
Non-local Fourier dimensions fall back to truncation-after-multiply.
"""
function _get_local_fourier_dims(evaluator::NonlinearEvaluator, bases::Tuple)
    dist = evaluator.dist
    ndim = length(bases)
    fourier_dims = Int[]

    # Determine which dimensions are decomposed
    # For PencilArrays with slab decomposition, the last ndims(mesh) dimensions are decomposed
    mesh = dist.mesh
    n_decomposed = mesh === nothing ? 0 : length(mesh)

    for (i, basis) in enumerate(bases)
        if isa(basis, Union{RealFourier, ComplexFourier})
            # Dimension i is Fourier. Is it also decomposed?
            # Convention: dimensions (ndim - n_decomposed + 1) : ndim are decomposed
            is_decomposed = (n_decomposed > 0) && (i > ndim - n_decomposed)
            if !is_decomposed
                push!(fourier_dims, i)
            end
        end
    end

    return fourier_dims
end

"""
    PaddedPencilFFTWorkspace

Workspace for proper 3/2-rule dealiasing on MPI-distributed PencilArray data.
Creates a padded-size PencilFFT plan and intermediate PencilArrays.

This enables correct dealiasing on ALL dimensions (including distributed ones),
unlike the local-only approach which can only pad non-decomposed dimensions.
"""
mutable struct PaddedPencilFFTWorkspace
    original_shape::Tuple{Vararg{Int}}
    padded_shape::Tuple{Vararg{Int}}
    padded_plan::Any  # PencilFFTs.PencilFFTPlan for padded size
    # Pre-allocated PencilArrays on padded grid
    padded_grid1::Any  # PencilArray for padded physical space
    padded_grid2::Any
    padded_product::Any
    padded_spec1::Any  # PencilArray for padded spectral space
    padded_spec2::Any
    padded_spec_product::Any
end

"""
    _get_padded_pencil_workspace!(evaluator, bases, dist) -> Union{PaddedPencilFFTWorkspace, Nothing}

Create or retrieve a cached PaddedPencilFFTWorkspace for distributed padded dealiasing.
Returns nothing if PencilFFTs cannot be created for the padded shape.
"""
function _get_padded_pencil_workspace!(evaluator::NonlinearEvaluator, bases::Tuple, dist::Distributor)
    key = "padded_pencil_$(hash(bases))"
    if haskey(evaluator.pencil_transforms, key)
        return evaluator.pencil_transforms[key]
    end

    factor = evaluator.dealiasing_factor

    # Compute padded global shape
    original_shape = Int[]
    padded_shape = Int[]
    has_fourier = false
    for basis in bases
        N = basis.meta.size
        push!(original_shape, N)
        if isa(basis, Union{RealFourier, ComplexFourier})
            M = Int(ceil(factor * N))
            if isodd(M); M += 1; end
            push!(padded_shape, M)
            has_fourier = true
        else
            push!(padded_shape, N)
        end
    end

    if !has_fourier
        return nothing
    end

    orig_t = Tuple(original_shape)
    pad_t = Tuple(padded_shape)

    try
        # Create PencilFFT plan for the padded global shape.
        # This is a collective MPI operation — all ranks must call it.
        # Use the same transform types as the original domain's plan.
        original_plan = nothing
        for tr in dist.transforms
            if isa(tr, PencilFFTs.PencilFFTPlan)
                original_plan = tr
                break
            end
        end

        if original_plan === nothing
            @warn "No existing PencilFFT plan found — cannot create padded plan for distributed dealiasing" maxlog=1
            return nothing
        end

        # Create padded plan using the same pencil configuration pattern
        input_pencil = first(original_plan.plans).pencil_in
        padded_pencil = PencilArrays.Pencil(input_pencil.topology, pad_t, input_pencil.decomp_dims)
        transforms = Tuple(PencilFFTs.Transforms.FFT() for _ in 1:length(pad_t))
        padded_plan = PencilFFTs.PencilFFTPlan(padded_pencil, transforms)

        # Pre-allocate PencilArrays for both physical and spectral space
        padded_grid1 = PencilFFTs.allocate_input(padded_plan)
        padded_grid2 = PencilFFTs.allocate_input(padded_plan)
        padded_product = PencilFFTs.allocate_input(padded_plan)
        padded_spec1 = PencilFFTs.allocate_output(padded_plan)
        padded_spec2 = PencilFFTs.allocate_output(padded_plan)
        padded_spec_product = PencilFFTs.allocate_output(padded_plan)

        ws = PaddedPencilFFTWorkspace(
            orig_t, pad_t, padded_plan,
            padded_grid1, padded_grid2, padded_product,
            padded_spec1, padded_spec2, padded_spec_product
        )
        evaluator.pencil_transforms[key] = ws
        @info "Created padded PencilFFT workspace: original=$orig_t, padded=$pad_t"
        return ws
    catch e
        @warn "Failed to create padded PencilFFT workspace: $e — falling back to local-only padding" maxlog=1
        return nothing
    end
end

"""
    evaluate_distributed_padded_multiply(field1, field2, evaluator, ws)

Full 3/2-rule padded dealiasing for MPI-distributed PencilArray data.
Uses a padded-size PencilFFT plan for correct dealiasing on ALL dimensions.

Algorithm:
1. Forward PencilFFT both fields (original size)
2. Pad spectral data into padded-size PencilArrays (zero-fill high modes)
3. Backward padded PencilFFT to padded physical grid
4. Multiply pointwise on padded grid
5. Forward padded PencilFFT
6. Truncate padded spectral data back to original size
7. Backward PencilFFT to original physical grid
"""
function evaluate_distributed_padded_multiply(field1::ScalarField, field2::ScalarField,
                                               evaluator::NonlinearEvaluator,
                                               ws::PaddedPencilFFTWorkspace)
    # Step 1: Forward transform both fields to spectral space (original PencilFFT)
    ensure_layout!(field1, :c)
    ensure_layout!(field2, :c)

    spec1 = get_coeff_data(field1)
    spec2 = get_coeff_data(field2)

    # Step 2: Pad spectral data into padded PencilArrays
    # Zero the padded arrays first
    fill!(parent(ws.padded_spec1), zero(eltype(ws.padded_spec1)))
    fill!(parent(ws.padded_spec2), zero(eltype(ws.padded_spec2)))

    # Copy low-frequency spectral data from original to padded pencils.
    # Both are PencilArrays with different global shapes but the same MPI decomposition.
    # The local portion of each rank holds a subset of the global spectral modes.
    # We copy the modes that fit in the original grid (low frequencies) and
    # leave the rest as zeros (high-frequency padding).
    _copy_pencil_spectral_to_padded!(ws.padded_spec1, spec1, ws.original_shape, ws.padded_shape)
    _copy_pencil_spectral_to_padded!(ws.padded_spec2, spec2, ws.original_shape, ws.padded_shape)

    # Step 3: Backward padded PencilFFT → padded physical grid
    ws.padded_grid1 .= ws.padded_plan \ ws.padded_spec1
    ws.padded_grid2 .= ws.padded_plan \ ws.padded_spec2

    # Step 4: Multiply on padded grid
    parent(ws.padded_product) .= parent(ws.padded_grid1) .* parent(ws.padded_grid2)

    # Step 5: Forward padded PencilFFT → padded spectral
    ws.padded_spec_product .= ws.padded_plan * ws.padded_product

    # Step 6: Truncate padded spectral data back to original spectral PencilArrays
    result = get_temp_field(evaluator, field1, "product_$(field1.name)_$(field2.name)")
    ensure_layout!(result, :c)
    result_spec = get_coeff_data(result)
    fill!(parent(result_spec), zero(eltype(result_spec)))
    _copy_padded_spectral_to_pencil!(result_spec, ws.padded_spec_product, ws.original_shape, ws.padded_shape)

    # Step 7: Normalize — padded FFT/IFFT pair introduces M/N scale per dim
    # Normalization: The padded FFT/IFFT roundtrip produces result scaled by N/M
    # per padded dimension (because IFFT divides by M instead of N for the same
    # spectral coefficients). Multiply by M/N to correct.
    T = real(eltype(result_spec))
    scale = one(T)
    for (N, M) in zip(ws.original_shape, ws.padded_shape)
        if N != M
            scale *= T(M) / T(N)
        end
    end
    if !isapprox(scale, 1)
        parent(result_spec) .*= scale
    end

    # Transform back to grid space
    ensure_layout!(result, :g)
    return result
end

"""
    _copy_pencil_spectral_to_padded!(padded_pencil, orig_pencil, orig_shape, pad_shape)

Copy low-frequency spectral modes from original-size PencilArray to padded-size PencilArray.
Works on local data only (each rank copies its own portion).
"""
function _copy_pencil_spectral_to_padded!(padded_pencil, orig_pencil, orig_shape, pad_shape)
    # Get local data arrays
    orig_local = parent(orig_pencil)
    pad_local = parent(padded_pencil)

    # For each local element, check if its global spectral index falls within
    # the low-frequency region of the padded grid. If so, copy it.
    # Since both PencilArrays share the same MPI decomposition pattern,
    # the local indices correspond to the same global mode indices.
    # The local shapes may differ (padded is larger), so we copy the
    # overlapping region.
    ndim = length(orig_shape)
    copy_ranges = ntuple(ndim) do d
        1:min(size(orig_local, d), size(pad_local, d))
    end
    pad_local[copy_ranges...] .= orig_local[copy_ranges...]
end

"""
    _copy_padded_spectral_to_pencil!(orig_pencil, padded_pencil, orig_shape, pad_shape)

Copy low-frequency spectral modes from padded PencilArray back to original-size PencilArray.
"""
function _copy_padded_spectral_to_pencil!(orig_pencil, padded_pencil, orig_shape, pad_shape)
    orig_local = parent(orig_pencil)
    pad_local = parent(padded_pencil)
    ndim = length(orig_shape)
    copy_ranges = ntuple(ndim) do d
        1:min(size(orig_local, d), size(pad_local, d))
    end
    orig_local[copy_ranges...] .= pad_local[copy_ranges...]
end

function setup_nonlinear_transforms!(evaluator::NonlinearEvaluator)
    """Setup PencilFFT transforms for nonlinear term evaluation.

    Transforms are created lazily on first use and cached for reuse.
    This avoids pre-computing transforms for shapes that may never be used
    and ensures the correct shape is always available (including dealiased sizes
    like 3/2-rule padding which depend on the actual domain).

    The lazy creation path uses local FFTW plans (no MPI collectives), so it is
    safe to call from within the evaluation loop.
    """

    dist = evaluator.dist

    if length(dist.mesh) >= 2
        @info "Nonlinear evaluator configured for 2D+ parallelization (lazy transform creation)"
        @info "  Process mesh: $(dist.mesh)"
        @info "  Dealiasing factor: $(evaluator.dealiasing_factor)"
    else
        @info "Setting up nonlinear transforms for 1D parallelization (fallback)"
        # 1D parallelization fallback
        setup_1d_nonlinear_transforms!(evaluator)
    end
end

function setup_pencil_transforms_for_shape!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})
    """Setup PencilFFT transforms for specific 2D shape"""

    dist = evaluator.dist
    shape_key = "$(shape[1])x$(shape[2])"

    # Create pencil configuration for this shape
    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Create pencil configuration for this shape
        config = PencilConfig(shape, dist.mesh, comm=dist.comm)

        # For serial execution, use simple arrays and FFTW plans
        if MPI.Comm_size(dist.comm) == 1
            # Serial execution - use regular arrays
            forward_data_1 = zeros(ComplexF64, shape...)
            forward_data_2 = zeros(ComplexF64, shape...)

            # Create FFTW plans
            fft_plan_1 = FFTW.plan_fft(forward_data_1)
            fft_plan_2 = FFTW.plan_fft(forward_data_2)

            evaluator.pencil_transforms[shape_key] = Dict(
                "config" => config,
                "forward_pencil_1" => forward_data_1,
                "forward_pencil_2" => forward_data_2,
                "fft_plan_1" => fft_plan_1,
                "fft_plan_2" => fft_plan_2,
                "shape" => shape,
                "serial" => true
            )
        else
            # Parallel execution - use PencilArrays/PencilFFTs with PROPER TOPOLOGY
            # CRITICAL: Must use same decomposition convention as Distributor
            if !dist.use_pencil_arrays
                # GPU+MPI mode uses TransposableField with FIRST dimensions decomposed
                # NonlinearEvaluator with PencilArrays would create layout mismatch
                error("NonlinearEvaluator requires PencilArrays but Distributor has use_pencil_arrays=false (GPU+MPI mode). " *
                      "For GPU+MPI, use TransposableField-based nonlinear evaluation instead, or " *
                      "set use_pencil_arrays=true for CPU+MPI execution.")
            end

            try
                # CRITICAL FIX: Use Distributor's MPI topology and decomposition
                # to ensure consistency with field data layout
                ndim = length(shape)
                ndims_mesh = length(dist.mesh)

                # Decompose LAST dimensions (PencilArrays convention, matches dist.use_pencil_arrays=true)
                decomp_dims = if ndim >= ndims_mesh
                    ntuple(i -> ndim - ndims_mesh + i, ndims_mesh)
                else
                    ntuple(identity, ndim)
                end

                # Use existing MPI topology from Distributor if available
                if dist.mpi_topology !== nothing
                    pencil = PencilArrays.Pencil(dist.mpi_topology, shape, decomp_dims)
                else
                    # Create topology matching Distributor's mesh
                    temp_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
                    pencil = PencilArrays.Pencil(temp_topology, shape, decomp_dims)
                end

                # Create PencilArrays with proper pencil configuration
                forward_data_1 = PencilArrays.PencilArray{ComplexF64}(undef, pencil)
                forward_data_2 = PencilArrays.PencilArray{ComplexF64}(undef, pencil)

                # Create FFT plans for pencil arrays
                # PencilFFTPlan expects a tuple of transforms, one per dimension
                ndims_shape = length(shape)
                transforms = ntuple(_ -> PencilFFTs.Transforms.FFT(), ndims_shape)
                fft_plan_1 = PencilFFTs.PencilFFTPlan(pencil, transforms)
                fft_plan_2 = PencilFFTs.PencilFFTPlan(pencil, transforms)

                evaluator.pencil_transforms[shape_key] = Dict(
                    "config" => config,
                    "forward_pencil_1" => forward_data_1,
                    "forward_pencil_2" => forward_data_2,
                    "fft_plan_1" => fft_plan_1,
                    "fft_plan_2" => fft_plan_2,
                    "shape" => shape,
                    "serial" => false,
                    "pencil" => pencil,
                    "decomp_dims" => decomp_dims
                )
            catch pe
                # CRITICAL: In MPI mode, falling back to serial FFTW produces incorrect results
                if MPI.Comm_size(dist.comm) > 1
                    @error "PencilArrays setup failed in MPI mode - local FFTW will produce incorrect results" exception=pe
                    error("Nonlinear evaluator requires PencilArrays for MPI execution. " *
                          "Check your PencilArrays/PencilFFTs installation.")
                end

                # Serial fallback is only safe for single process
                @warn "PencilArrays setup failed, falling back to serial FFTW" exception=pe
                forward_data_1 = zeros(ComplexF64, shape...)
                forward_data_2 = zeros(ComplexF64, shape...)
                fft_plan_1 = FFTW.plan_fft(forward_data_1)
                fft_plan_2 = FFTW.plan_fft(forward_data_2)

                evaluator.pencil_transforms[shape_key] = Dict(
                    "config" => config,
                    "forward_pencil_1" => forward_data_1,
                    "forward_pencil_2" => forward_data_2,
                    "fft_plan_1" => fft_plan_1,
                    "fft_plan_2" => fft_plan_2,
                    "shape" => shape,
                    "serial" => true
                )
            end
        end

        @debug "Created FFT transforms for shape $shape"

    catch e
        @error "Failed to create FFT transforms for shape $shape: $e\n" *
               "Dealiasing will be disabled for this shape — results may contain aliasing errors."
    end
end

function setup_1d_nonlinear_transforms!(evaluator::NonlinearEvaluator)
    """
    Setup 1D FFT transforms for nonlinear term evaluation.

    This is the fallback for when only 1D domain decomposition is used
    (single process or 1D process mesh). Uses FFTW directly instead of
    PencilFFTs since there's no need for pencil transposes.

    For 1D parallelization:
    - The domain is split along one dimension only
    - FFTs along the local (non-decomposed) dimensions use FFTW
    - FFTs along the decomposed dimension require MPI communication

    This setup creates:
    - Local FFTW plans for each common array size
    - Scratch arrays for in-place transforms
    - Dealiased array configurations
    """

    dist = evaluator.dist
    @info "Setting up 1D nonlinear transforms"

    # Common 1D sizes for spectral methods
    common_1d_sizes = [32, 64, 128, 256, 512, 1024]

    # Common 2D shapes (for 2D problems with 1D decomposition)
    common_2d_shapes = [(64, 64), (128, 64), (128, 128), (256, 128), (256, 256), (512, 256)]

    # Common 3D shapes (for 3D problems with 1D decomposition)
    common_3d_shapes = [(64, 64, 64), (128, 64, 64), (128, 128, 64), (128, 128, 128)]

    # Setup 1D transforms
    for n in common_1d_sizes
        setup_1d_fftw_plans!(evaluator, n)
    end

    # Setup 2D transforms (1D decomposition means one axis is fully local)
    for shape in common_2d_shapes
        setup_2d_fftw_plans!(evaluator, shape)
    end

    # Setup 3D transforms
    for shape in common_3d_shapes
        setup_3d_fftw_plans!(evaluator, shape)
    end

    @info "1D nonlinear transform setup complete"
    @info "  MPI size: $(dist.size)"
    @info "  Dealiasing factor: $(evaluator.dealiasing_factor)"
end

function setup_1d_fftw_plans!(evaluator::NonlinearEvaluator, n::Int)
    """Setup FFTW plans for 1D transforms of size n."""

    shape_key = "1d_$n"

    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Dealiased size using 3/2 rule
        n_dealias = ceil(Int, n * evaluator.dealiasing_factor)

        # Create scratch arrays
        scratch_real = zeros(Float64, n_dealias)
        scratch_complex = zeros(ComplexF64, div(n_dealias, 2) + 1)

        # Create FFTW plans
        # Real-to-complex forward transform
        forward_plan = FFTW.plan_rfft(scratch_real; flags=FFTW.MEASURE)

        # Complex-to-real backward transform
        backward_plan = FFTW.plan_brfft(scratch_complex, n_dealias; flags=FFTW.MEASURE)

        evaluator.pencil_transforms[shape_key] = Dict(
            "type" => :fftw_1d,
            "size" => n,
            "dealiased_size" => n_dealias,
            "forward_plan" => forward_plan,
            "backward_plan" => backward_plan,
            "scratch_real" => scratch_real,
            "scratch_complex" => scratch_complex
        )

        @debug "Created 1D FFTW plans for size $n (dealiased: $n_dealias)"

    catch e
        @warn "Failed to create 1D FFTW plans for size $n: $e"
    end
end

function setup_2d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})
    """Setup FFTW plans for 2D transforms."""

    shape_key = "2d_$(shape[1])x$(shape[2])"

    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Dealiased sizes
        nx_dealias = ceil(Int, shape[1] * evaluator.dealiasing_factor)
        ny_dealias = ceil(Int, shape[2] * evaluator.dealiasing_factor)
        dealias_shape = (nx_dealias, ny_dealias)

        # Create scratch arrays
        scratch_real = zeros(Float64, dealias_shape)
        scratch_complex = zeros(ComplexF64, div(nx_dealias, 2) + 1, ny_dealias)

        # Create FFTW plans for 2D real-to-complex transforms
        forward_plan = FFTW.plan_rfft(scratch_real; flags=FFTW.MEASURE)
        backward_plan = FFTW.plan_brfft(scratch_complex, nx_dealias; flags=FFTW.MEASURE)

        evaluator.pencil_transforms[shape_key] = Dict(
            "type" => :fftw_2d,
            "shape" => shape,
            "dealiased_shape" => dealias_shape,
            "forward_plan" => forward_plan,
            "backward_plan" => backward_plan,
            "scratch_real" => scratch_real,
            "scratch_complex" => scratch_complex
        )

        @debug "Created 2D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 2D FFTW plans for shape $shape: $e"
    end
end

function setup_3d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int, Int})
    """Setup FFTW plans for 3D transforms."""

    shape_key = "3d_$(shape[1])x$(shape[2])x$(shape[3])"

    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Dealiased sizes
        nx_dealias = ceil(Int, shape[1] * evaluator.dealiasing_factor)
        ny_dealias = ceil(Int, shape[2] * evaluator.dealiasing_factor)
        nz_dealias = ceil(Int, shape[3] * evaluator.dealiasing_factor)
        dealias_shape = (nx_dealias, ny_dealias, nz_dealias)

        # Create scratch arrays
        scratch_real = zeros(Float64, dealias_shape)
        scratch_complex = zeros(ComplexF64, div(nx_dealias, 2) + 1, ny_dealias, nz_dealias)

        # Create FFTW plans for 3D real-to-complex transforms
        forward_plan = FFTW.plan_rfft(scratch_real; flags=FFTW.MEASURE)
        backward_plan = FFTW.plan_brfft(scratch_complex, nx_dealias; flags=FFTW.MEASURE)

        evaluator.pencil_transforms[shape_key] = Dict(
            "type" => :fftw_3d,
            "shape" => shape,
            "dealiased_shape" => dealias_shape,
            "forward_plan" => forward_plan,
            "backward_plan" => backward_plan,
            "scratch_real" => scratch_real,
            "scratch_complex" => scratch_complex
        )

        @debug "Created 3D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 3D FFTW plans for shape $shape: $e"
    end
end

function get_nonlinear_transform(evaluator::NonlinearEvaluator, shape::Tuple)
    """
    Get the appropriate transform configuration for a given shape.

    Automatically selects between PencilFFT (for multi-D parallelization)
    and FFTW (for 1D parallelization or serial) based on what's available.
    """
    ndims_shape = length(shape)

    # Try to find exact match first
    if ndims_shape == 1
        shape_key = "1d_$(shape[1])"
    elseif ndims_shape == 2
        shape_key = "2d_$(shape[1])x$(shape[2])"
        # Also try PencilFFT format
        pencil_key = "$(shape[1])x$(shape[2])"
    elseif ndims_shape == 3
        shape_key = "3d_$(shape[1])x$(shape[2])x$(shape[3])"
        pencil_key = "$(shape[1])x$(shape[2])x$(shape[3])"
    else
        @warn "Unsupported shape dimension: $ndims_shape"
        return nothing
    end

    # Check for FFTW-based transform
    if haskey(evaluator.pencil_transforms, shape_key)
        return evaluator.pencil_transforms[shape_key]
    end

    # Check for PencilFFT-based transform (2D/3D only)
    if ndims_shape >= 2 && haskey(evaluator.pencil_transforms, pencil_key)
        return evaluator.pencil_transforms[pencil_key]
    end

    # No exact match - try to create one on the fly
    @debug "Creating transform on-the-fly for shape $shape"

    if ndims_shape == 1
        setup_1d_fftw_plans!(evaluator, shape[1])
    elseif ndims_shape == 2
        setup_2d_fftw_plans!(evaluator, shape)
    elseif ndims_shape == 3
        setup_3d_fftw_plans!(evaluator, shape)
    end

    # Return the newly created transform
    return get(evaluator.pencil_transforms, shape_key, nothing)
end

# Main nonlinear evaluation functions
function evaluate_nonlinear_term(op::AdvectionOperator, layout::Symbol=:g)
    """Evaluate u·∇φ nonlinear term using transform method"""
    
    velocity = op.velocity
    scalar = op.scalar
    
    # Get distributor for transform operations
    dist = velocity.dist

    # Create nonlinear evaluator if not exists
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    evaluator = dist.nonlinear_evaluator
    
    # Compute gradient of scalar field: ∇φ
    grad_scalar = evaluate_gradient(Gradient(scalar, dist.coordsys), :g)
    
    # Evaluate u·∇φ = u_x ∂φ/∂x + u_y ∂φ/∂y (+ u_z ∂φ/∂z in 3D)
    # Use in-place accumulation to avoid lazy Add trees and per-iteration allocation
    result = ScalarField(dist, "$(op.name)_$(scalar.name)", scalar.bases, scalar.dtype)
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)
    fill!(result_data, zero(scalar.dtype))

    # Sum velocity components times gradient components (in-place)
    for i in 1:length(velocity.components)
        product = evaluate_transform_multiply(velocity.components[i], grad_scalar.components[i], evaluator)
        ensure_layout!(product, :g)
        product_data = get_grid_data(product)
        if isa(result_data, PencilArrays.PencilArray) && isa(product_data, PencilArrays.PencilArray)
            parent(result_data) .+= parent(product_data)
        else
            result_data .+= product_data
        end
    end

    return result
end

function evaluate_nonlinear_term(op::NonlinearAdvectionOperator, layout::Symbol=:g)
    """Evaluate (u·∇)u nonlinear momentum term"""

    velocity = op.velocity
    dist = velocity.dist

    # Create nonlinear evaluator if needed
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    evaluator = dist.nonlinear_evaluator

    # Result is a vector field
    result = VectorField(dist, dist.coordsys, "$(op.name)_$(velocity.name)", velocity.bases, velocity.dtype)

    # For each component: (u·∇)u_i = u_j ∂u_i/∂x_j (summed over j)
    # Use in-place accumulation to avoid lazy Add trees and allocation per iteration
    for i in 1:length(velocity.components)
        ensure_layout!(result.components[i], :g)
        comp_data = get_grid_data(result.components[i])
        fill!(comp_data, zero(velocity.dtype))

        # Sum over all spatial directions (in-place)
        for j in 1:length(velocity.components)
            coord = dist.coordsys[j]
            du_i_dx_j = evaluate_differentiate(Differentiate(velocity.components[i], coord, 1), :g)

            product = evaluate_transform_multiply(velocity.components[j], du_i_dx_j, evaluator)
            ensure_layout!(product, :g)
            product_data = get_grid_data(product)
            if isa(comp_data, PencilArrays.PencilArray) && isa(product_data, PencilArrays.PencilArray)
                parent(comp_data) .+= parent(product_data)
            else
                comp_data .+= product_data
            end
        end
    end

    return result
end

function evaluate_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)
    """Efficiently multiply two fields using transform method and proper dealiasing.

    Uses proper 3/2-rule padded dealiasing (Orszag 1971) when possible:
    - CPU serial: full padded dealiasing on all Fourier dimensions
    - GPU serial: full padded dealiasing using GPU FFTs (CUFFT)
    - MPI distributed: padded dealiasing on LOCAL (non-decomposed) Fourier
      dimensions; truncation-after-multiply on distributed dimensions

    Falls back to truncation-after-multiply only when no Fourier bases exist.
    """

    start_time = time()

    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)

    if field1.bases != field2.bases
        throw(ArgumentError(
            "Cannot multiply fields with different bases: " *
            "'$(field1.name)' has bases=$(field1.bases) but " *
            "'$(field2.name)' has bases=$(field2.bases). " *
            "Both fields must be defined on the same domain."))
    end

    # Try proper 3/2-rule padded dealiasing
    if evaluator.dealiasing_factor > 1.0
        T = field1.dtype <: Complex ? real(field1.dtype) : field1.dtype
        data1 = get_grid_data(field1)
        is_pencil = isa(data1, PencilArrays.PencilArray)

        if is_pencil && evaluator.dist.size > 1
            # MPI path: try full distributed padded dealiasing first (all dimensions),
            # then fall back to local-only padding (non-decomposed dimensions only)
            pencil_ws = _get_padded_pencil_workspace!(evaluator, field1.bases, evaluator.dist)
            if pencil_ws !== nothing
                result = evaluate_distributed_padded_multiply(field1, field2, evaluator, pencil_ws)
                evaluator.performance_stats.total_evaluations += 1
                evaluator.performance_stats.total_time += (time() - start_time)
                evaluator.performance_stats.dealiasing_time += (time() - start_time)
                return result
            end

            # Fallback: pad only local (non-decomposed) Fourier dimensions
            local_data = parent(data1)
            local_fdims = _get_local_fourier_dims(evaluator, field1.bases)
            if !isempty(local_fdims)
                ws = _get_padded_workspace!(evaluator, field1.bases, T;
                        local_shape=size(local_data),
                        local_fourier_dims=local_fdims,
                        arch=CPU())
                if ws !== nothing
                    result = evaluate_padded_multiply(field1, field2, evaluator, ws)
                    evaluator.performance_stats.total_evaluations += 1
                    evaluator.performance_stats.total_time += (time() - start_time)
                    evaluator.performance_stats.dealiasing_time += (time() - start_time)
                    return result
                end
            end
        else
            # Serial path (CPU or GPU): pad all Fourier dimensions
            ws = _get_padded_workspace!(evaluator, field1.bases, T)
            if ws !== nothing
                result = evaluate_padded_multiply(field1, field2, evaluator, ws)
                evaluator.performance_stats.total_evaluations += 1
                evaluator.performance_stats.total_time += (time() - start_time)
                evaluator.performance_stats.dealiasing_time += (time() - start_time)
                return result
            end
        end
    end

    # Fallback: direct multiplication + truncation-after-multiply dealiasing
    # (only reached for fields with no Fourier bases, or dealiasing_factor <= 1)
    result = get_temp_field(evaluator, field1, "product_$(field1.name)_$(field2.name)")
    ensure_layout!(result, :g)

    result_data = get_grid_data(result)
    field1_data = get_grid_data(field1)
    field2_data = get_grid_data(field2)

    if isa(result_data, PencilArrays.PencilArray)
        parent(result_data) .= parent(field1_data) .* parent(field2_data)
    else
        gpu_multiply_fields!(result["g"], field1["g"], field2["g"])
    end

    # Only apply truncation-after-multiply dealiasing if the field has Fourier bases.
    # For pure Chebyshev/Legendre fields, the forward/backward transform roundtrip
    # in apply_basic_dealiasing! can introduce significant numerical errors.
    if evaluator.dealiasing_factor > 1.0
        has_fourier = any(isa(b, Union{RealFourier, ComplexFourier}) for b in field1.bases)
        if has_fourier
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end

    evaluator.performance_stats.total_evaluations += 1
    evaluator.performance_stats.total_time += (time() - start_time)

    return result
end

"""
    gpu_multiply_fields!(result_data, data1, data2)

GPU-accelerated pointwise multiplication.
This function is overridden by the CUDA extension to use GPU kernels.
"""
function gpu_multiply_fields!(result_data::AbstractArray, data1::AbstractArray, data2::AbstractArray)
    # Default CPU implementation using broadcasting
    result_data .= data1 .* data2
    return result_data
end

function evaluate_2d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)
    """2D transform-based multiplication using PencilFFTs"""
    
    # Create result field
    result = ScalarField(field1.dist, "product_$(field1.name)_$(field2.name)", field1.bases, field1.dtype)
    ensure_layout!(result, :g)
    
    # Try to find matching transform configuration
    shape_key = "$(shape[1])x$(shape[2])"
    
    if haskey(evaluator.pencil_transforms, shape_key)
        # Use precomputed PencilFFT transforms
        transform_info = evaluator.pencil_transforms[shape_key]
        
        # Get data in appropriate format for PencilArrays
        data1 = get_pencil_compatible_data(field1, transform_info["config"])
        data2 = get_pencil_compatible_data(field2, transform_info["config"])
        
        # Pointwise multiplication in grid space
        # This is where the actual nonlinear interaction happens
        result_data = data1 .* data2
        
        # Apply dealiasing by transforming to spectral space and back
        if evaluator.dealiasing_factor > 1.0
            result_data = apply_2d_dealiasing(result_data, transform_info, evaluator.dealiasing_factor)
        end
        
        # Set result data
        set_pencil_compatible_data!(result, result_data, transform_info["config"])
        
    else
        # Fallback to direct multiplication without PencilFFT optimization
        @debug "No precomputed transforms for shape $shape, using fallback"
        result["g"] .= field1["g"] .* field2["g"]
        
        # Apply basic dealiasing if possible
        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end
    
    return result
end

function evaluate_3d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)
    """3D transform-based multiplication using 3D PencilFFTs"""
    
    result = ScalarField(field1.dist, "product_$(field1.name)_$(field2.name)", field1.bases, field1.dtype)
    ensure_layout!(result, :g)
    
    # For 3D, we need more sophisticated pencil management
    # This would involve 3D pencil decomposition across all three spatial dimensions
    
    if length(evaluator.dist.mesh) >= 3
        # Use 3D PencilFFT approach
        @warn "3D dealiasing via padding is not yet implemented; using undealiased multiplication" maxlog=1

        # Direct multiplication for now - would implement full 3D PencilFFT logic
        result["g"] .= field1["g"] .* field2["g"]
        
        # Apply 3D dealiasing
        if evaluator.dealiasing_factor > 1.0
            apply_3d_dealiasing!(result, evaluator.dealiasing_factor)
        end
        
    else
        # Fallback for insufficient parallelization
        @warn "3D nonlinear multiplication falling back to undealiased pointwise multiply" maxlog=1
        result["g"] .= field1["g"] .* field2["g"]

        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end

    return result
end

function evaluate_fallback_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)
    """Fallback multiplication for unsupported dimensions"""

    result = ScalarField(field1.dist, "product_$(field1.name)_$(field2.name)", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Simple pointwise multiplication
    result["g"] .= field1["g"] .* field2["g"]
    
    # Apply dealiasing if requested
    if evaluator.dealiasing_factor > 1.0
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end
    
    return result
end

# Dealiasing functions
function apply_2d_dealiasing(data::AbstractArray, transform_info::Dict, dealiasing_factor::Float64)
    """Apply 2D dealiasing using PencilFFT transforms"""

    # Transform to spectral space
    fft_plan = transform_info["fft_plan_1"]
    spectral_data = fft_plan * data

    # Zero out high-frequency modes (2/3 rule: keep |k| <= N/(2*factor))
    shape = transform_info["shape"]
    cutoff_x = Int(floor(shape[1] / (2 * dealiasing_factor)))
    cutoff_y = Int(floor(shape[2] / (2 * dealiasing_factor)))

    # Apply dealiasing cutoff
    apply_spectral_cutoff!(spectral_data, (cutoff_x, cutoff_y))

    # Transform back to grid space using backward FFT with normalization
    # FFTW's ifft = bfft / N, so we use bfft and divide by the total size
    dealiased_data = FFTW.bfft(spectral_data) / length(spectral_data)

    return dealiased_data
end

"""
    apply_3d_dealiasing!(field::ScalarField, dealiasing_factor::Float64)

Apply 3D dealiasing.
GPU-compatible: uses appropriate implementation based on field's architecture.
"""
function apply_3d_dealiasing!(field::ScalarField, dealiasing_factor::Float64)
    # Compute cutoffs: keep modes with |k| <= N/(2*dealiasing_factor)
    cutoffs_check = [begin
        if isa(basis, Union{RealFourier, ComplexFourier})
            Int(floor(basis.meta.size / (2 * dealiasing_factor)))
        else
            typemax(Int) >> 1
        end
    end for basis in field.bases]

    # Skip if grid too small for meaningful dealiasing
    any_zero_cutoff = any(c == 0 for (c, basis) in zip(cutoffs_check, field.bases)
                          if isa(basis, Union{RealFourier, ComplexFourier}))
    if any_zero_cutoff
        return
    end

    # Transform to coefficient space
    ensure_layout!(field, :c)

    coeff_data = get_coeff_data(field)

    cutoffs = tuple([begin
        if isa(basis, Union{RealFourier, ComplexFourier})
            Int(floor(basis.meta.size / (2 * dealiasing_factor)))
        else
            size(coeff_data, i)
        end
    end for (i, basis) in enumerate(field.bases)]...)

    # Detect rfft dimensions
    rfft_dims = tuple([begin
        isa(basis, RealFourier) && size(coeff_data, i) == div(basis.meta.size, 2) + 1
    end for (i, basis) in enumerate(field.bases)]...)

    # Apply spectral cutoff - this function handles GPU arrays automatically
    apply_spectral_cutoff!(coeff_data, cutoffs, rfft_dims)

    # Transform back to grid space
    backward_transform!(field)
end

"""
    apply_basic_dealiasing!(field::ScalarField, dealiasing_factor::Float64)

Basic dealiasing for fields without PencilFFT support.
GPU-compatible: uses appropriate implementation based on field's architecture.

Applies the 2/3 rule: after pointwise multiplication in grid space, the product
contains aliased high-frequency modes. This function removes them by:
1. Forward FFT to spectral space
2. Zero modes with |k| > N/(2*dealiasing_factor) (= N/3 for standard 3/2 rule)
3. Inverse FFT back to grid space

Note: This is "truncation-after-multiply" which removes aliased energy in
high modes but cannot undo aliasing contamination of low modes. For exact
dealiasing, use the padding approach (pad to 3N/2, multiply, truncate).
"""
function apply_basic_dealiasing!(field::ScalarField, dealiasing_factor::Float64)
    # Compute cutoff wavenumbers for each Fourier basis dimension.
    # For the 2/3 rule (dealiasing_factor=1.5): keep modes with |k| <= N/3.
    # The maximum resolved wavenumber is N/2, and we keep the bottom 1/dealiasing_factor
    # fraction: cutoff = (N/2) / dealiasing_factor = N / (2 * dealiasing_factor).
    cutoffs = tuple([begin
        if isa(basis, Union{RealFourier, ComplexFourier})
            Int(floor(basis.meta.size / (2 * dealiasing_factor)))
        else
            # Non-Fourier bases (Chebyshev, Legendre): no dealiasing cutoff
            # Use a large value so these dimensions are never filtered
            typemax(Int) >> 1
        end
    end for (i, basis) in enumerate(field.bases)]...)

    # Skip dealiasing if any Fourier cutoff is 0 (grid too small for meaningful dealiasing).
    # A cutoff of 0 would zero ALL non-DC modes, destroying the field's spatial variation.
    # This happens for grids with N <= 2*dealiasing_factor (e.g., N<=3 for 3/2 rule).
    any_zero_cutoff = any(c == 0 for (c, basis) in zip(cutoffs, field.bases)
                          if isa(basis, Union{RealFourier, ComplexFourier}))
    if any_zero_cutoff
        return
    end

    # Transform to spectral space
    forward_transform!(field)

    coeff_data = get_coeff_data(field)

    # Recompute cutoffs using actual coeff array sizes for non-Fourier bases
    cutoffs = tuple([begin
        if isa(basis, Union{RealFourier, ComplexFourier})
            Int(floor(basis.meta.size / (2 * dealiasing_factor)))
        else
            size(coeff_data, i)
        end
    end for (i, basis) in enumerate(field.bases)]...)

    # Detect which dimensions are rfft output (positive frequencies only)
    rfft_dims = tuple([begin
        isa(basis, RealFourier) && size(coeff_data, i) == div(basis.meta.size, 2) + 1
    end for (i, basis) in enumerate(field.bases)]...)

    # Apply spectral cutoff - this function handles GPU arrays automatically
    apply_spectral_cutoff!(coeff_data, cutoffs, rfft_dims)

    # Transform back to grid space
    backward_transform!(field)
end


"""
    apply_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=())

Apply spectral cutoff to remove high-frequency modes (dealiasing).

For spectral data stored in standard FFT layout:
- Positive frequencies: indices 1 to N/2+1
- Negative frequencies: indices N/2+2 to N (for complex FFT)

For rfft output dimensions (indicated by rfft_dims):
- All indices are positive frequencies: k = 0, 1, ..., N/2
- No negative frequency region

This function zeros out modes beyond the cutoff wavenumber in each dimension.
Used for dealiasing in nonlinear term evaluation.

GPU-compatible: Uses broadcasting-based implementation for GPU arrays.

Arguments:
- data: Complex spectral coefficient array
- cutoffs: Tuple of cutoff wavenumbers for each dimension
- rfft_dims: Tuple of Bool indicating which dimensions are rfft output
             (all positive frequencies, no negative frequency region)

The cutoff is applied symmetrically: modes with |k| > cutoff are zeroed.
"""
function apply_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    ndims_data = ndims(data)

    # Check if data is on GPU - use broadcasting-based implementation
    if is_gpu_array(data)
        apply_spectral_cutoff_gpu!(data, cutoffs, rfft_dims)
        return
    end

    # CPU implementation with loops
    if ndims_data == 1
        if length(rfft_dims) >= 1 && rfft_dims[1]
            apply_rfft_spectral_cutoff!(data, cutoffs[1])
        else
            apply_1d_spectral_cutoff!(data, 1, cutoffs[1])
        end
    elseif ndims_data == 2
        apply_2d_spectral_cutoff!(data, cutoffs, rfft_dims)
    elseif ndims_data == 3
        apply_3d_spectral_cutoff!(data, cutoffs, rfft_dims)
    else
        # General N-dimensional case
        apply_nd_spectral_cutoff!(data, cutoffs, rfft_dims)
    end
end

"""
    apply_spectral_cutoff_gpu!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=())

GPU-compatible spectral cutoff using broadcasting with a mask.
Creates a dealiasing mask and applies it element-wise via broadcasting.
"""
function apply_spectral_cutoff_gpu!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    shape = size(data)

    # Build the dealiasing mask using broadcasting
    # The mask is 1.0 where modes should be kept, 0.0 where they should be zeroed
    mask = create_dealiasing_mask(shape, cutoffs, eltype(data), rfft_dims)

    # Move mask to same device as data
    # Use architecture(data) to infer the correct architecture from the array
    arch = architecture(data)
    mask_device = on_architecture(arch, mask)

    # Apply mask using broadcasting (GPU-compatible)
    data .*= mask_device
end

"""
    create_dealiasing_mask(shape::Tuple, cutoffs::Tuple, T::Type, rfft_dims::Tuple=())

Create a dealiasing mask array for the given shape and cutoffs.
Returns a CPU array; caller should move to GPU if needed.

The mask is 1 where |k_i| <= cutoff_i for all dimensions, 0 otherwise.

For rfft dimensions (rfft_dims[d] == true), all indices represent positive
frequencies (k = 0, 1, ..., N/2) and the mask zeros k > cutoff.
For standard FFT dimensions, the standard layout with negative frequencies is used.
"""
function create_dealiasing_mask(shape::Tuple, cutoffs::Tuple, T::Type, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    ndims_data = length(shape)

    # Build 1D masks for each dimension, then combine with outer product
    masks_1d = Vector{Vector{real(T)}}(undef, ndims_data)

    for d in 1:ndims_data
        n = shape[d]
        cutoff = d <= length(cutoffs) ? cutoffs[d] : div(n, 2)
        is_rfft = d <= length(rfft_dims) && rfft_dims[d]

        mask_d = zeros(real(T), n)
        for i in 1:n
            if is_rfft
                # rfft output: all indices are positive frequencies k = i-1
                k = i - 1
                mask_d[i] = k <= cutoff ? one(real(T)) : zero(real(T))
            else
                # Standard complex FFT layout:
                # indices 1 to N/2+1: k = 0, 1, ..., N/2
                # indices N/2+2 to N: k = -(N/2-1), ..., -1
                half_n = div(n, 2)
                k = i <= half_n + 1 ? i - 1 : i - n - 1
                mask_d[i] = abs(k) <= cutoff ? one(real(T)) : zero(real(T))
            end
        end
        masks_1d[d] = mask_d
    end

    # Combine 1D masks into N-D mask using broadcasting
    if ndims_data == 1
        return masks_1d[1]
    elseif ndims_data == 2
        # Outer product of two 1D masks
        return masks_1d[1] .* masks_1d[2]'
    elseif ndims_data == 3
        # 3D: reshape and broadcast
        mask_x = reshape(masks_1d[1], :, 1, 1)
        mask_y = reshape(masks_1d[2], 1, :, 1)
        mask_z = reshape(masks_1d[3], 1, 1, :)
        return mask_x .* mask_y .* mask_z
    else
        # General N-D case: use recursive broadcasting
        mask = ones(real(T), shape...)
        for d in 1:ndims_data
            # Create shape for broadcasting: all 1s except dimension d
            broadcast_shape = ntuple(i -> i == d ? shape[d] : 1, ndims_data)
            mask_d = reshape(masks_1d[d], broadcast_shape...)
            mask .*= mask_d
        end
        return mask
    end
end

"""
    apply_rfft_spectral_cutoff!(data::AbstractVector, cutoff::Int)

Apply spectral cutoff to rfft output (all positive frequencies).

For rfft output with N/2+1 points:
- Index 1: k=0 (DC component)
- Index i: k=i-1 (all positive frequencies)

Modes with k > cutoff are set to zero.
"""
function apply_rfft_spectral_cutoff!(data::AbstractVector, cutoff::Int)
    n = length(data)
    # rfft: all indices are positive frequencies k = i-1
    # Zero indices where k > cutoff, i.e., i > cutoff+1
    for i in (cutoff + 2):n
        data[i] = zero(eltype(data))
    end
end

function apply_1d_spectral_cutoff!(data::AbstractVector, axis::Int, cutoff::Int)
    """
    Apply 1D spectral cutoff along a vector.

    For FFT layout with N points:
    - Index 1: k=0 (DC component)
    - Indices 2 to N/2+1: positive frequencies k=1 to N/2
    - Indices N/2+2 to N: negative frequencies k=-(N/2-1) to -1

    Modes with |k| > cutoff are set to zero.
    """
    n = length(data)
    if cutoff >= div(n, 2)
        return  # No cutoff needed
    end

    # Zero positive high frequencies: indices cutoff+2 to N/2+1
    # (index 1 is k=0, index 2 is k=1, ..., index cutoff+1 is k=cutoff)
    half_n = div(n, 2)
    for i in (cutoff + 2):(half_n + 1)
        if i <= n
            data[i] = zero(eltype(data))
        end
    end

    # Zero negative high frequencies: indices N/2+2 to N-cutoff
    # Negative frequencies are stored in reverse order at the end
    for i in (half_n + 2):(n - cutoff)
        if i <= n
            data[i] = zero(eltype(data))
        end
    end
end

function apply_1d_spectral_cutoff!(data::AbstractArray, axis::Int, cutoff::Int)
    """
    Apply 1D spectral cutoff along specified axis of multi-dimensional array.
    """
    shape = size(data)
    n = shape[axis]

    if cutoff >= div(n, 2)
        return  # No cutoff needed
    end

    half_n = div(n, 2)

    # Create index ranges for slicing
    # Zero out positive high frequencies
    for k in (cutoff + 2):(half_n + 1)
        if k <= n
            indices = ntuple(ndims(data)) do d
                d == axis ? k : Colon()
            end
            data[indices...] .= zero(eltype(data))
        end
    end

    # Zero out negative high frequencies
    for k in (half_n + 2):(n - cutoff)
        if k <= n
            indices = ntuple(ndims(data)) do d
                d == axis ? k : Colon()
            end
            data[indices...] .= zero(eltype(data))
        end
    end
end

function apply_2d_spectral_cutoff!(data::AbstractMatrix, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, 2))
    """
    Apply 2D spectral cutoff for dealiasing.

    Zeros out modes where |kx| > cutoffs[1] or |ky| > cutoffs[2].
    For rfft dimensions, all indices are positive frequencies (k=i-1).
    """
    nx, ny = size(data)
    kx_cut = cutoffs[1]
    ky_cut = length(cutoffs) >= 2 ? cutoffs[2] : div(ny, 2)

    x_is_rfft = length(rfft_dims) >= 1 && rfft_dims[1]
    y_is_rfft = length(rfft_dims) >= 2 && rfft_dims[2]

    half_nx = div(nx, 2)
    half_ny = div(ny, 2)

    for j in 1:ny
        # Determine wavenumber for y-dimension
        if y_is_rfft
            ky = j - 1  # rfft: all positive
            y_in_range = ky <= ky_cut
        else
            ky = j <= half_ny + 1 ? j - 1 : j - ny - 1
            y_in_range = abs(ky) <= ky_cut
        end

        for i in 1:nx
            # Determine wavenumber for x-dimension
            if x_is_rfft
                kx = i - 1  # rfft: all positive
                x_in_range = kx <= kx_cut
            else
                kx = i <= half_nx + 1 ? i - 1 : i - nx - 1
                x_in_range = abs(kx) <= kx_cut
            end

            # Zero out if either frequency is outside cutoff
            if !x_in_range || !y_in_range
                data[i, j] = zero(eltype(data))
            end
        end
    end
end

function apply_3d_spectral_cutoff!(data::AbstractArray{T, 3}, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, 3)) where T
    """
    Apply 3D spectral cutoff for dealiasing.

    Zeros out modes where |kx| > cutoffs[1], |ky| > cutoffs[2], or |kz| > cutoffs[3].
    For rfft dimensions, all indices are positive frequencies (k=i-1).
    """
    nx, ny, nz = size(data)
    kx_cut = cutoffs[1]
    ky_cut = length(cutoffs) >= 2 ? cutoffs[2] : div(ny, 2)
    kz_cut = length(cutoffs) >= 3 ? cutoffs[3] : div(nz, 2)

    x_is_rfft = length(rfft_dims) >= 1 && rfft_dims[1]
    y_is_rfft = length(rfft_dims) >= 2 && rfft_dims[2]
    z_is_rfft = length(rfft_dims) >= 3 && rfft_dims[3]

    half_nx = div(nx, 2)
    half_ny = div(ny, 2)
    half_nz = div(nz, 2)

    for k in 1:nz
        if z_is_rfft
            kz_val = k - 1
            z_in_range = kz_val <= kz_cut
        else
            kz_val = k <= half_nz + 1 ? k - 1 : k - nz - 1
            z_in_range = abs(kz_val) <= kz_cut
        end

        for j in 1:ny
            if y_is_rfft
                ky_val = j - 1
                y_in_range = ky_val <= ky_cut
            else
                ky_val = j <= half_ny + 1 ? j - 1 : j - ny - 1
                y_in_range = abs(ky_val) <= ky_cut
            end

            for i in 1:nx
                if x_is_rfft
                    kx_val = i - 1
                    x_in_range = kx_val <= kx_cut
                else
                    kx_val = i <= half_nx + 1 ? i - 1 : i - nx - 1
                    x_in_range = abs(kx_val) <= kx_cut
                end

                if !x_in_range || !y_in_range || !z_in_range
                    data[i, j, k] = zero(T)
                end
            end
        end
    end
end

function apply_nd_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    """
    Apply N-dimensional spectral cutoff (general case).
    """
    shape = size(data)
    ndims_data = ndims(data)

    # Extend cutoffs to match dimensions
    actual_cutoffs = ntuple(ndims_data) do d
        d <= length(cutoffs) ? cutoffs[d] : div(shape[d], 2)
    end

    half_shape = div.(shape, 2)

    for I in CartesianIndices(data)
        # Check if any frequency is outside cutoff
        outside_cutoff = false

        for d in 1:ndims_data
            idx = I[d]
            n = shape[d]
            is_rfft = d <= length(rfft_dims) && rfft_dims[d]

            if is_rfft
                # rfft: all indices are positive frequencies
                k = idx - 1
                if k > actual_cutoffs[d]
                    outside_cutoff = true
                    break
                end
            else
                # Standard FFT layout
                half_n = half_shape[d]
                k = idx <= half_n + 1 ? idx - 1 : idx - n - 1
                if abs(k) > actual_cutoffs[d]
                    outside_cutoff = true
                    break
                end
            end
        end

        if outside_cutoff
            data[I] = zero(eltype(data))
        end
    end
end

"""
    apply_spherical_spectral_cutoff!(data::AbstractArray, k_max::Int)

Apply spherical spectral cutoff: zero modes with |k| > k_max.

This is useful for isotropic dealiasing where the cutoff is based
on the magnitude of the wavevector rather than individual components.

|k|² = kx² + ky² + kz² (for 3D)

GPU-compatible: Uses broadcasting-based implementation for GPU arrays.
"""
function apply_spherical_spectral_cutoff!(data::AbstractArray, k_max::Int)
    # Check if data is on GPU - use broadcasting-based implementation
    if is_gpu_array(data)
        apply_spherical_spectral_cutoff_gpu!(data, k_max)
        return
    end

    # CPU implementation with loops
    shape = size(data)
    ndims_data = ndims(data)
    half_shape = div.(shape, 2)
    k_max_sq = k_max^2

    for I in CartesianIndices(data)
        # Compute |k|²
        k_sq = 0
        for d in 1:ndims_data
            idx = I[d]
            n = shape[d]
            half_n = half_shape[d]
            k = idx <= half_n + 1 ? idx - 1 : idx - n - 1
            k_sq += k^2
        end

        if k_sq > k_max_sq
            data[I] = zero(eltype(data))
        end
    end
end

"""
    apply_spherical_spectral_cutoff_gpu!(data::AbstractArray, k_max::Int)

GPU-compatible spherical spectral cutoff using broadcasting with a mask.
"""
function apply_spherical_spectral_cutoff_gpu!(data::AbstractArray, k_max::Int)
    shape = size(data)
    ndims_data = ndims(data)

    # Create spherical mask on CPU, then move to device
    mask = create_spherical_mask(shape, k_max, eltype(data))

    # Use architecture(data) to infer the correct architecture from the array
    arch = architecture(data)
    mask_device = on_architecture(arch, mask)

    # Apply mask using broadcasting
    data .*= mask_device
end

"""
    create_spherical_mask(shape::Tuple, k_max::Int, T::Type)

Create a spherical dealiasing mask for the given shape and k_max.
Mask is 1 where |k| <= k_max, 0 otherwise.
"""
function create_spherical_mask(shape::Tuple, k_max::Int, T::Type)
    ndims_data = length(shape)
    k_max_sq = k_max^2

    # Create wavenumber arrays for each dimension
    k_arrays = Vector{Vector{Int}}(undef, ndims_data)
    for d in 1:ndims_data
        n = shape[d]
        half_n = div(n, 2)
        k_arrays[d] = [i <= half_n + 1 ? i - 1 : i - n - 1 for i in 1:n]
    end

    # Build mask based on |k|² <= k_max²
    if ndims_data == 1
        return real(T).([abs(k) <= k_max ? 1.0 : 0.0 for k in k_arrays[1]])
    elseif ndims_data == 2
        return real(T).([k_arrays[1][i]^2 + k_arrays[2][j]^2 <= k_max_sq ? 1.0 : 0.0
                         for i in 1:shape[1], j in 1:shape[2]])
    elseif ndims_data == 3
        return real(T).([k_arrays[1][i]^2 + k_arrays[2][j]^2 + k_arrays[3][k]^2 <= k_max_sq ? 1.0 : 0.0
                         for i in 1:shape[1], j in 1:shape[2], k in 1:shape[3]])
    else
        # General N-D case
        mask = zeros(real(T), shape...)
        for I in CartesianIndices(mask)
            k_sq = sum(k_arrays[d][I[d]]^2 for d in 1:ndims_data)
            mask[I] = k_sq <= k_max_sq ? one(real(T)) : zero(real(T))
        end
        return mask
    end
end

function get_dealiasing_cutoffs(shape::Tuple, dealiasing_factor::Float64=1.5)
    """
    Compute spectral cutoffs for dealiasing.

    For the 2/3 rule (dealiasing_factor=1.5):
    cutoff = N / (2 * dealiasing_factor) = N/3

    This keeps modes with |k| <= N/3. When combined with proper padding
    (or used as a post-multiply filter), modes above this cutoff are zeroed
    to suppress aliasing from quadratic nonlinear interactions.
    """
    return tuple([floor(Int, n / (2 * dealiasing_factor)) for n in shape]...)
end

# Utility functions for PencilArray compatibility
function get_pencil_compatible_data(field::ScalarField, config::PencilConfig)
    """
    Convert field data to PencilArray format compatible with the given PencilConfig.

    This function ensures that the field's data is:
    1. In grid space layout (for nonlinear operations)
    2. Compatible with the PencilConfig's global shape
    3. Uses the correct data type
    4. Properly distributed according to the mesh configuration

    Returns the field's grid space data as a PencilArray or compatible array.
    """

    # Ensure field is in grid space layout for nonlinear operations
    ensure_layout!(field, :g)

    # Verify field has allocated data
    if get_grid_data(field) === nothing
        throw(ArgumentError("Field $(field.name) has no grid space data allocated"))
    end

    field_data = get_grid_data(field)
    field_shape = size(field_data)

    # Pencil transforms operate on CPU arrays; copy from GPU if needed
    if is_gpu_array(field_data)
        field_pencil = Array(field_data)
    else
        field_pencil = field_data
    end

    # Verify shape compatibility
    # The field's local shape should be consistent with the global shape and mesh decomposition
    if !is_shape_compatible(field_shape, config.global_shape, config.mesh, config.comm;
                            use_pencil_arrays=field.dist.use_pencil_arrays)
        @warn "Shape mismatch: field local shape $(field_shape) may not be compatible with " *
              "global shape $(config.global_shape) and mesh $(config.mesh)"
    end

    # Verify data type compatibility and convert if needed
    if eltype(field_pencil) != config.dtype
        @debug "Converting field data type from $(eltype(field_pencil)) to $(config.dtype)"
        # Create converted copy if types don't match
        # Preserve PencilArray wrapper if present (for MPI parallelization)
        if isa(field_pencil, PencilArrays.PencilArray)
            # Create a new PencilArray with the same structure but new dtype
            # Use similar() to preserve the pencil configuration, then copyto! converted data
            pencil = PencilArrays.pencil(field_pencil)
            converted_pencil = PencilArrays.PencilArray{config.dtype}(undef, pencil)
            # Convert and copy data
            src_data = parent(field_pencil)
            dest_data = parent(converted_pencil)
            copyto!(dest_data, convert.(config.dtype, src_data))
            return converted_pencil
        else
            converted_data = convert.(config.dtype, field_pencil)
            return converted_data
        end
    end

    # Verify MPI communicator compatibility
    if field.dist.use_pencil_arrays && field.dist.pencil_config !== nothing
        if field.dist.pencil_config.comm != config.comm
            @warn "MPI communicator mismatch between field and config"
        end
    end

    @debug "Retrieved pencil compatible data for field $(field.name)" size=field_shape eltype=eltype(field_pencil)

    return field_pencil
end

function is_shape_compatible(local_shape::Tuple, global_shape::Tuple, mesh::Tuple, comm::MPI.Comm;
                             use_pencil_arrays::Bool=true)
    """
    Check if the local shape is compatible with the global shape given the mesh decomposition.

    For a valid pencil decomposition:
    - The product of local shapes across all ranks should equal the global shape
    - The local shape should be approximately global_shape / mesh for distributed dimensions

    The `use_pencil_arrays` flag controls which dimensions are expected to be decomposed:
    - true (default): PencilArrays convention - decompose LAST dimensions
    - false: TransposableField ZLocal convention - decompose FIRST dimensions
    """

    if length(local_shape) != length(global_shape)
        return false
    end

    # For serial execution (single rank), local shape should match global shape
    if MPI.Comm_size(comm) == 1
        return local_shape == global_shape
    end

    # For parallel execution, check that local shape is reasonable
    # (within expected range given the mesh decomposition)
    num_dims = length(global_shape)
    mesh_dims = length(mesh)

    for i in 1:num_dims
        # Determine if this dimension is decomposed based on convention
        is_decomposed = if use_pencil_arrays
            # PencilArrays: decompose LAST mesh_dims dimensions
            # For 3D with 2D mesh: dims 2,3 decomposed; dim 1 local
            decomp_start = num_dims - mesh_dims + 1
            mesh_idx = i - decomp_start + 1
            i >= decomp_start && mesh_idx >= 1 && mesh_idx <= mesh_dims && mesh[mesh_idx] > 1
        else
            # TransposableField ZLocal: decompose FIRST mesh_dims dimensions
            # mesh[1] (Rx) decomposes dim 1, mesh[2] (Ry) decomposes dim 2, etc.
            i <= mesh_dims && mesh[i] > 1
        end

        if is_decomposed
            # This dimension is distributed - get the correct mesh divisor
            mesh_idx = if use_pencil_arrays
                decomp_start = num_dims - mesh_dims + 1
                i - decomp_start + 1
            else
                i
            end

            expected_local = ceil(Int, global_shape[i] / mesh[mesh_idx])
            min_local = floor(Int, global_shape[i] / mesh[mesh_idx])

            if local_shape[i] < min_local || local_shape[i] > expected_local
                return false
            end
        else
            # This dimension is not distributed, should match global
            if local_shape[i] != global_shape[i]
                return false
            end
        end
    end

    return true
end

function get_pencil_config_from_field(field::ScalarField)
    """
    Extract a PencilConfig from a ScalarField's distributor configuration.
    """
    dist = field.dist

    if dist.pencil_config !== nothing
        return dist.pencil_config
    end

    # Build a config from field properties
    if field.domain === nothing
        throw(ArgumentError("Field $(field.name) has no domain set - cannot create PencilConfig"))
    end
    gshape = global_shape(field.domain)
    mesh_config = dist.mesh !== nothing ? dist.mesh : (1,)

    return PencilConfig(
        gshape,
        mesh_config;
        comm=dist.comm,
        dtype=field.dtype
    )
end

function ensure_pencil_compatibility!(field::ScalarField, config::PencilConfig)
    """
    Ensure field is compatible with the given PencilConfig, reallocating if necessary.

    This function modifies the field in-place to ensure compatibility with the config.
    Returns true if the field was modified, false otherwise.
    """

    # Ensure field is in grid space
    ensure_layout!(field, :g)

    if get_grid_data(field) === nothing
        # Allocate new data with the correct configuration
        allocate_field_data!(field, config)
        return true
    end

    current_shape = size(get_grid_data(field))

    # Check if reallocation is needed
    needs_realloc = false

    # Check shape compatibility
    if !is_shape_compatible(current_shape, config.global_shape, config.mesh, config.comm;
                            use_pencil_arrays=field.dist.use_pencil_arrays)
        needs_realloc = true
    end

    # Check dtype compatibility
    if eltype(get_grid_data(field)) != config.dtype
        needs_realloc = true
    end

    if needs_realloc
        # Store old data for potential interpolation
        old_data = copy(get_grid_data(field))

        # Allocate new data
        allocate_field_data!(field, config)

        # Attempt to interpolate/copy data if shapes are compatible enough
        try
            interpolate_field_data!(get_grid_data(field), old_data)
        catch e
            @warn "Could not preserve field data during reallocation: $e"
            fill!(get_grid_data(field), zero(config.dtype))
        end

        return true
    end

    return false
end

"""
    _compute_coeff_shape(global_shape::Tuple, bases::Tuple; use_pencil_fft::Bool=false)

Compute the coefficient space shape based on the bases.

IMPORTANT: In MPI mode with PencilFFTs (use_pencil_fft=true), only the FIRST RealFourier
axis uses RFFT (size N/2+1). Subsequent RealFourier axes use FFT (full size N) because
PencilFFTs can only apply RFFT to the first transform dimension.

In serial mode (use_pencil_fft=false), all RealFourier axes use RFFT (N/2+1).
"""
function _compute_coeff_shape(global_shape::Tuple, bases::Tuple; use_pencil_fft::Bool=false)
    shape = Int[]

    # Find the first Fourier axis for PencilFFT mode
    first_fourier_idx = nothing
    if use_pencil_fft
        for (i, basis) in enumerate(bases)
            if isa(basis, RealFourier) || isa(basis, ComplexFourier)
                first_fourier_idx = i
                break
            end
        end
    end

    for (i, basis) in enumerate(bases)
        if i <= length(global_shape)
            if isa(basis, RealFourier)
                if use_pencil_fft && first_fourier_idx !== nothing && i != first_fourier_idx
                    # PencilFFT mode: subsequent RealFourier uses FFT (full size)
                    push!(shape, global_shape[i])
                else
                    # Serial mode OR first RealFourier in PencilFFT: use RFFT (N/2+1)
                    push!(shape, div(global_shape[i], 2) + 1)
                end
            else
                # Other bases: same size in coefficient space
                push!(shape, global_shape[i])
            end
        end
    end
    return Tuple(shape)
end

"""
    allocate_field_data!(field::ScalarField, config::PencilConfig)

Allocate field data according to the PencilConfig.
Architecture-aware: allocates on GPU if the field's distributor uses GPU architecture.

Decomposition conventions:
- CPU+MPI with PencilArrays: decompose LAST dimensions (PencilArrays convention)
- GPU+MPI (TransposableField): decompose FIRST dimensions (ZLocal convention)
- Serial: no decomposition, use global shape
"""
function allocate_field_data!(field::ScalarField, config::PencilConfig)
    dist = field.dist
    arch = dist.architecture
    nprocs = MPI.Comm_size(config.comm)

    if nprocs == 1
        # Serial execution: use global shapes with standard coefficient sizing
        coeff_global_shape = _compute_coeff_shape(config.global_shape, field.bases; use_pencil_fft=false)
        if is_gpu(arch)
            set_grid_data!(field, create_array(arch, config.dtype, config.global_shape...))
            set_coeff_data!(field, create_array(arch, Complex{real(config.dtype)}, coeff_global_shape...))
        else
            set_grid_data!(field, zeros(config.dtype, config.global_shape...))
            set_coeff_data!(field, zeros(Complex{real(config.dtype)}, coeff_global_shape...))
        end
    elseif dist.use_pencil_arrays
        # CPU+MPI with PencilArrays: use PencilFFT plan's allocators for compatibility
        # CRITICAL: PencilFFTs requires arrays allocated from the plan's pencils
        pencil_plan = nothing
        for transform in dist.transforms
            if isa(transform, PencilFFTs.PencilFFTPlan)
                pencil_plan = transform
                break
            end
        end

        # Only reuse the existing plan's allocators if its global shape matches the config.
        # ensure_pencil_compatibility! may request a different global shape.
        plan_matches = pencil_plan !== nothing &&
            first(pencil_plan.plans).pencil_in.size_global == config.global_shape
        if plan_matches
            # Use PencilFFTs' official allocators - guaranteed compatible with mul!/ldiv!
            set_grid_data!(field, PencilFFTs.allocate_input(pencil_plan))
            set_coeff_data!(field, PencilFFTs.allocate_output(pencil_plan))
        else
            # Fallback: create pencils (may not be compatible with PencilFFTs)
            coeff_global_shape = _compute_coeff_shape(config.global_shape, field.bases; use_pencil_fft=true)
            grid_pencil = create_pencil(dist, config.global_shape, nothing, dtype=config.dtype)
            coeff_pencil = create_pencil(dist, coeff_global_shape, nothing, dtype=Complex{real(config.dtype)})
            set_grid_data!(field, grid_pencil)
            set_coeff_data!(field, coeff_pencil)
        end
    else
        # GPU+MPI (or MPI without PencilArrays): use local shapes with TransposableField convention
        # TransposableField ZLocal convention: decompose FIRST dimensions (x by Rx, y by Ry, z local)
        # For GPU+MPI, local FFTs are used per rank, so standard coefficient sizing applies
        coeff_global_shape = _compute_coeff_shape(config.global_shape, field.bases; use_pencil_fft=false)
        # Use get_local_array_size which respects use_pencil_arrays flag for correct convention
        local_grid_shape = get_local_array_size(dist, config.global_shape)
        local_coeff_shape = get_local_array_size(dist, coeff_global_shape)

        if is_gpu(arch)
            set_grid_data!(field, create_array(arch, config.dtype, local_grid_shape...))
            set_coeff_data!(field, create_array(arch, Complex{real(config.dtype)}, local_coeff_shape...))
        else
            set_grid_data!(field, zeros(config.dtype, local_grid_shape...))
            set_coeff_data!(field, zeros(Complex{real(config.dtype)}, local_coeff_shape...))
        end
    end

    field.current_layout = :g
end

"""
    compute_local_shape(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)

Compute the local shape for this rank given the global shape and mesh decomposition.

The mesh defines how processes are arranged in a Cartesian grid. For example,
mesh=(2,4) means 2 processes in dimension 1 and 4 in dimension 2, for 8 total.

The global array is decomposed such that each dimension is split among the
processes in that mesh dimension. Load balancing distributes remainders
to the first ranks in each dimension.

# Arguments
- `global_shape`: Total size of the array in each dimension
- `mesh`: Number of processes in each decomposition dimension
- `comm`: MPI communicator

# Returns
- Tuple of local sizes for each dimension on this rank

# Example
```julia
# 4 processes with mesh (2, 2), global shape (100, 100)
# Each process gets local shape (50, 50)
local_shape = compute_local_shape((100, 100), (2, 2), comm)
```
"""
function compute_local_shape(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)
    mpi_rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    num_dims = length(global_shape)
    mesh_dims = length(mesh)

    # Validate mesh configuration
    mesh_size = prod(mesh)
    if mesh_size != nprocs
        @warn "Mesh size ($mesh_size) doesn't match number of processes ($nprocs)"
    end

    local_shape = collect(global_shape)

    # Compute mesh coordinates for this rank using row-major (C-style) ordering
    # mpi_rank = coord[1] + coord[2]*mesh[1] + coord[3]*mesh[1]*mesh[2] + ...
    mesh_coords = zeros(Int, mesh_dims)
    remaining_rank = mpi_rank
    for i in 1:mesh_dims
        mesh_coords[i] = remaining_rank % mesh[i]
        remaining_rank = remaining_rank ÷ mesh[i]
    end

    # CRITICAL: Decompose LAST dimensions to match Distributor convention
    # Distributor uses: decomp_dims = ntuple(i -> ndim - ndims_mesh + i, ndims_mesh)
    # This means for 3D data with 2D mesh: decompose dims (2, 3), keep dim 1 local
    # For 2D data with 1D mesh: decompose dim 2, keep dim 1 local

    # Compute local sizes for each decomposed dimension
    for i in 1:min(num_dims, mesh_dims)
        # Map mesh dimension i to global dimension (last dimensions)
        global_dim_idx = num_dims - mesh_dims + i

        if mesh[i] > 1 && global_dim_idx >= 1
            n = global_shape[global_dim_idx]  # Global size in this dimension
            p = mesh[i]                        # Number of processes in this dimension
            coord = mesh_coords[i]             # This rank's position in mesh dimension i

            # Compute local size with load balancing
            # First (remainder) ranks get one extra element
            base_size = n ÷ p
            remainder = n % p

            if coord < remainder
                # First 'remainder' ranks get base_size + 1
                local_shape[global_dim_idx] = base_size + 1
            else
                local_shape[global_dim_idx] = base_size
            end
        end
        # Dimensions before decomp_dims keep their global size (local)
    end

    return tuple(local_shape...)
end

"""
    compute_local_range(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)

Compute the global index ranges owned by this rank for each dimension.

# Returns
- Vector of (start, stop) tuples for each dimension (1-based indices)
"""
function compute_local_range(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)
    mpi_rank = MPI.Comm_rank(comm)
    num_dims = length(global_shape)
    mesh_dims = length(mesh)

    # Compute mesh coordinates
    mesh_coords = zeros(Int, mesh_dims)
    remaining_rank = mpi_rank
    for i in 1:mesh_dims
        mesh_coords[i] = remaining_rank % mesh[i]
        remaining_rank = remaining_rank ÷ mesh[i]
    end

    # Determine which dimensions are decomposed
    # Convention: mesh decomposes the LAST dimensions
    # For 3D data with 2D mesh: mesh[1] -> dim 2, mesh[2] -> dim 3
    # For 3D data with 1D mesh: mesh[1] -> dim 3
    decomp_start = num_dims - mesh_dims + 1

    ranges = Vector{Tuple{Int,Int}}(undef, num_dims)

    for i in 1:num_dims
        # Map dimension i to mesh dimension (if decomposed)
        mesh_idx = i - decomp_start + 1

        if mesh_idx >= 1 && mesh_idx <= mesh_dims && mesh[mesh_idx] > 1
            n = global_shape[i]
            p = mesh[mesh_idx]
            coord = mesh_coords[mesh_idx]

            base_size = n ÷ p
            remainder = n % p

            if coord < remainder
                local_size = base_size + 1
                start_idx = coord * (base_size + 1) + 1
            else
                local_size = base_size
                start_idx = remainder * (base_size + 1) + (coord - remainder) * base_size + 1
            end

            ranges[i] = (start_idx, start_idx + local_size - 1)
        else
            # Not decomposed in this dimension
            ranges[i] = (1, global_shape[i])
        end
    end

    return ranges
end

function interpolate_field_data!(dest::AbstractArray, src::AbstractArray)
    """
    Interpolate source data into destination array.
    Uses nearest-neighbor or linear interpolation depending on relative sizes.
    """
    src_shape = size(src)
    dest_shape = size(dest)

    if src_shape == dest_shape
        copyto!(dest, src)
        return
    end

    # Use nearest-neighbor interpolation for simplicity
    num_dims = length(dest_shape)

    for I in CartesianIndices(dest)
        src_indices = ntuple(num_dims) do d
            # Map destination index to source index
            src_idx = round(Int, (I[d] - 1) * (src_shape[d] - 1) / max(dest_shape[d] - 1, 1)) + 1
            clamp(src_idx, 1, src_shape[d])
        end
        dest[I] = src[src_indices...]
    end
end

function set_pencil_compatible_data!(field::ScalarField, data, config::PencilConfig)
    """
    Set field data from PencilArray format.
    Since ScalarField stores data as PencilArrays, this mainly ensures
    proper layout and copies the data.
    """
    
    # Ensure field is in grid space layout
    ensure_layout!(field, :g)
    
    # Verify that field has allocated data
    if get_grid_data(field) === nothing
        throw(ArgumentError("Field $(field.name) has no grid space data allocated"))
    end
    
    # Verify data compatibility
    if size(data) != size(get_grid_data(field))
        throw(DimensionMismatch("Data size $(size(data)) does not match field size $(size(get_grid_data(field)))"))
    end

    if eltype(data) != eltype(get_grid_data(field))
        @warn "Data type mismatch during set: incoming $(eltype(data)), field $(eltype(get_grid_data(field)))"
    end

    # Copy data into the field's PencilArray, respecting architecture
    arr = data
    if eltype(get_grid_data(field)) != eltype(arr)
        arr = convert.(eltype(get_grid_data(field)), arr)
    end

    arch = field.dist.architecture
    if is_gpu_array(get_grid_data(field))
        arr = on_architecture(arch, arr)
    end

    copyto!(get_grid_data(field), arr)
    
    # Mark field as having valid grid space data
    field.current_layout = :g
    
    @debug "Set pencil compatible data for field $(field.name)" size(data) eltype(data)
end

# Memory management
function get_temp_field(evaluator::NonlinearEvaluator, template::ScalarField, name::String)
    """Get temporary field for intermediate calculations """
    
    key = "$(name)_$(hash(template.bases))"
    
    if !haskey(evaluator.temp_fields, key)
        temp_field = ScalarField(template.dist, name, template.bases, template.dtype)

        # Ensure temporary field has allocated data
        ensure_layout!(temp_field, :g)

        evaluator.temp_fields[key] = temp_field
    end
    
    return evaluator.temp_fields[key]
end

function clear_temp_fields!(evaluator::NonlinearEvaluator)
    """Clear temporary fields to free memory"""
    empty!(evaluator.temp_fields)
    GC.gc()
end

"""
    get_temp_array(evaluator::NonlinearEvaluator, shape::Tuple, dtype::Type)

Get temporary array for intermediate calculations.
Architecture-aware: allocates on GPU if evaluator uses GPU architecture.
"""
function get_temp_array(evaluator::NonlinearEvaluator, shape::Tuple, dtype::Type)
    arch = architecture(evaluator)
    if is_gpu(arch)
        # GPU: use architecture-aware allocation
        return create_array(arch, dtype, shape...)
    else
        # CPU: standard zeros allocation
        return zeros(dtype, shape...)
    end
end

"""
    return_temp_array!(evaluator::NonlinearEvaluator, arr::AbstractArray)

Return temporary array to pool or free GPU memory.
For CPU, this is a no-op (GC handles memory).
For GPU, this can free memory explicitly if needed.
"""
function return_temp_array!(evaluator::NonlinearEvaluator, arr::AbstractArray)
    if is_gpu(evaluator) && is_gpu_array(arr)
        # For GPU arrays, we can explicitly free if needed
        # unsafe_free!(arr)  # Uncomment if explicit memory management is needed
    end
    return nothing
end

# Integration with existing operator evaluation
function evaluate_operator(op::NonlinearOperator)
    """Evaluate nonlinear operator"""
    
    if isa(op, AdvectionOperator)
        return evaluate_nonlinear_term(op)
    elseif isa(op, NonlinearAdvectionOperator)
        return evaluate_nonlinear_term(op)
    elseif isa(op, ConvectiveOperator)
        return evaluate_convective_operator(op)
    else
        throw(ArgumentError("Nonlinear operator evaluation not implemented for $(typeof(op))"))
    end
end

"""Get the cached NonlinearEvaluator for a distributor, creating one if needed."""
function _get_evaluator(dist::Distributor)
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    return dist.nonlinear_evaluator
end

function evaluate_convective_operator(op::ConvectiveOperator)
    """Evaluate general convective operator"""

    field1, field2 = op.field1, op.field2

    if op.operation == :multiply
        # Simple multiplication
        if isa(field1, ScalarField) && isa(field2, ScalarField)
            evaluator = _get_evaluator(field1.dist)
            return evaluate_transform_multiply(field1, field2, evaluator)
        else
            throw(ArgumentError("Multiplication not implemented for field types $(typeof(field1)), $(typeof(field2))"))
        end
        
    elseif op.operation == :dot_product
        # Dot product between vectors
        if isa(field1, VectorField) && isa(field2, VectorField)
            return evaluate_vector_dot_product(field1, field2)
        else
            throw(ArgumentError("Dot product requires two vector fields"))
        end
        
    elseif op.operation == :cross_product
        # Cross product between vectors
        if isa(field1, VectorField) && isa(field2, VectorField)
            return evaluate_vector_cross_product(field1, field2)
        else
            throw(ArgumentError("Cross product requires two vector fields"))
        end
        
    else
        throw(ArgumentError("Unknown convective operation: $(op.operation)"))
    end
end

function evaluate_vector_dot_product(v1::VectorField, v2::VectorField)
    """Evaluate v1·v2 dot product"""
    
    if length(v1.components) != length(v2.components)
        throw(ArgumentError("Vector fields must have same number of components"))
    end
    
    evaluator = _get_evaluator(v1.dist)

    # Sum products of components
    result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator)
    
    for i in 2:length(v1.components)
        product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator)
        result = result + product
    end
    
    return result
end

function evaluate_vector_cross_product(v1::VectorField, v2::VectorField)
    """Evaluate v1×v2 cross product (3D only)"""
    
    if length(v1.components) != 3 || length(v2.components) != 3
        throw(ArgumentError("Cross product requires 3D vector fields"))
    end
    if v1.coordsys !== v2.coordsys
        throw(ArgumentError("Cross product requires identical coordinate systems"))
    end
    if v1.bases != v2.bases
        throw(ArgumentError("Cannot compute cross product of VectorFields with different bases"))
    end
    
    evaluator = _get_evaluator(v1.dist)

    handedness = (hasproperty(v1.coordsys, :right_handed) && v1.coordsys.right_handed === false) ? -1 : 1
    
    # Cross product: (a×b)_x = a_y*b_z - a_z*b_y
    #                (a×b)_y = a_z*b_x - a_x*b_z  
    #                (a×b)_z = a_x*b_y - a_y*b_x
    
    result = VectorField(v1.dist, v1.coordsys, "cross_$(v1.name)_$(v2.name)", v1.bases, v1.dtype)
    
    # x-component
    term1 = evaluate_transform_multiply(v1.components[2], v2.components[3], evaluator)
    term2 = evaluate_transform_multiply(v1.components[3], v2.components[2], evaluator)
    result.components[1] = (term1 - term2) * handedness
    
    # y-component  
    term1 = evaluate_transform_multiply(v1.components[3], v2.components[1], evaluator)
    term2 = evaluate_transform_multiply(v1.components[1], v2.components[3], evaluator)
    result.components[2] = (term1 - term2) * handedness
    
    # z-component
    term1 = evaluate_transform_multiply(v1.components[1], v2.components[2], evaluator)
    term2 = evaluate_transform_multiply(v1.components[2], v2.components[1], evaluator)
    result.components[3] = (term1 - term2) * handedness

    return result
end

# Helper to evaluate any operand (Future, Operator, or Field)
function _evaluate_any_operand(arg, layout::Symbol)
    if isa(arg, Future)
        return evaluate(arg; force=true)
    elseif isa(arg, Operator)
        return evaluate(arg, layout)
    else
        return arg
    end
end

# Evaluate methods for DotProduct and CrossProduct from arithmetic.jl
function evaluate(op::DotProduct, layout::Symbol=:g)
    """Evaluate DotProduct of two VectorFields"""
    args = future_args(op)
    if length(args) != 2
        throw(ArgumentError("DotProduct expects exactly two operands"))
    end

    # Evaluate operands first - they might be operators that return VectorFields
    v1 = _evaluate_any_operand(args[1], layout)
    v2 = _evaluate_any_operand(args[2], layout)

    if !isa(v1, VectorField) || !isa(v2, VectorField)
        throw(ArgumentError("DotProduct requires two VectorField operands, got $(typeof(v1)) and $(typeof(v2))"))
    end

    result = evaluate_vector_dot_product(v1, v2)
    ensure_layout!(result, layout)
    return result
end

function evaluate(op::CrossProduct, layout::Symbol=:g)
    """Evaluate CrossProduct of two VectorFields"""
    args = future_args(op)
    if length(args) != 2
        throw(ArgumentError("CrossProduct expects exactly two operands"))
    end

    # Evaluate operands first - they might be operators that return VectorFields
    v1 = _evaluate_any_operand(args[1], layout)
    v2 = _evaluate_any_operand(args[2], layout)

    if !isa(v1, VectorField) || !isa(v2, VectorField)
        throw(ArgumentError("CrossProduct requires two VectorField operands, got $(typeof(v1)) and $(typeof(v2))"))
    end

    result = evaluate_vector_cross_product(v1, v2)
    for comp in result.components
        ensure_layout!(comp, layout)
    end
    return result
end

# Convenience constructors
function advection(u::VectorField, φ::ScalarField)
    """Create advection operator u·∇φ"""
    return AdvectionOperator(u, φ)
end

function nonlinear_momentum(u::VectorField)
    """Create nonlinear momentum operator (u·∇)u"""
    return NonlinearAdvectionOperator(u)
end

function convection(f1, f2, op::Symbol)
    """Create convective operator"""
    return ConvectiveOperator(f1, f2, op)
end

function log_nonlinear_performance(stats::NonlinearPerformanceStats)
    """Log nonlinear evaluation performance statistics"""

    if MPI.Initialized()
        mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if mpi_rank == 0
            @info "Nonlinear evaluation performance:"
            @info "  Total evaluations: $(stats.total_evaluations)"
            @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
            @info "  Average time per evaluation: $(round(stats.total_time/max(stats.total_evaluations, 1), digits=6)) seconds"
            @info "  Dealiasing time: $(round(stats.dealiasing_time, digits=3)) seconds ($(round(100*stats.dealiasing_time/max(stats.total_time, 1e-10), digits=1))%)"
            @info "  Transform time: $(round(stats.transform_time, digits=3)) seconds ($(round(100*stats.transform_time/max(stats.total_time, 1e-10), digits=1))%)"
        end
    else
        @info "Nonlinear evaluation performance:"
        @info "  Total evaluations: $(stats.total_evaluations)"
        @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
        @info "  Average time per evaluation: $(round(stats.total_time/max(stats.total_evaluations, 1), digits=6)) seconds"
    end
end

# ============================================================================
# Exports
# ============================================================================

# Export types
export NonlinearOperator, AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator
export NonlinearEvaluator, NonlinearPerformanceStats

# Export constructor functions
export advection, nonlinear_momentum, convection

# Export evaluation functions
export evaluate_nonlinear_term, evaluate_transform_multiply, evaluate_operator
export evaluate_vector_dot_product, evaluate_vector_cross_product

# Export dealiasing functions
export apply_basic_dealiasing!, apply_spectral_cutoff!, get_dealiasing_cutoffs
export apply_spherical_spectral_cutoff!

# Export utility functions
export get_nonlinear_transform, setup_nonlinear_transforms!
export get_temp_field, clear_temp_fields!, get_temp_array
export log_nonlinear_performance

# Export GPU helper functions for masks (useful for custom dealiasing)
export create_dealiasing_mask, create_spherical_mask

# Export pencil compatibility functions
export get_pencil_compatible_data, set_pencil_compatible_data!
export compute_local_shape, compute_local_range, is_shape_compatible
