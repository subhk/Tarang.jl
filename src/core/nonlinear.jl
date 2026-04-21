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

include("nonlinear/nonlinear_core.jl")

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

    # Pre-allocated spectral buffers (original size) — avoids fft() allocation
    spec1::A
    spec2::A
    spec_result::A

    # FFT plans: FFTW.MEASURE for CPU, plain plan_fft for GPU
    plan_forward::AbstractFFTPlan
    plan_backward::AbstractFFTPlan

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
    # Use tuple key to avoid string allocation on every call
    key = (hash(bases), dtype, hash(local_shape), is_gpu(_arch))
    if haskey(evaluator.pencil_transforms, key)
        return evaluator.pencil_transforms[key]::PaddedDealiasingWorkspace{T}
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

    # Allocate original-size spectral buffers
    spec1 = zeros(_arch, Complex{T}, orig_t...)
    spec2 = zeros(_arch, Complex{T}, orig_t...)
    spec_result = zeros(_arch, Complex{T}, orig_t...)

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
        spec1, spec2, spec_result,
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
    ws.spec1 .= Complex{T}.(raw1_ws)
    ws.spec1 .= fft(ws.spec1, ws.fourier_dims)
    ws.spec2 .= Complex{T}.(raw2_ws)
    ws.spec2 .= fft(ws.spec2, ws.fourier_dims)

    # Step 2: Pad spectral coefficients
    _pad_spectral!(ws.padded1, ws.spec1, ws.original_shape, ws.padded_shape, ws.fourier_dims)
    _pad_spectral!(ws.padded2, ws.spec2, ws.original_shape, ws.padded_shape, ws.fourier_dims)

    # Step 3: IFFT to padded grid (using pre-computed plan)
    ws.padded1 .= ws.plan_backward * ws.padded1
    ws.padded2 .= ws.plan_backward * ws.padded2

    # Step 4: Multiply on padded grid
    ws.padded_product .= ws.padded1 .* ws.padded2

    # Step 5: FFT product back
    ws.padded_product .= ws.plan_forward * ws.padded_product

    # Step 6: Truncate to original coefficients
    fill!(ws.spec_result, zero(Complex{T}))
    _truncate_spectral!(ws.spec_result, ws.padded_product, ws.original_shape, ws.padded_shape, ws.fourier_dims)

    # Step 7: IFFT to grid and normalize
    # Normalization: padded IFFT divides by M, but we want result on N-grid.
    # Scale by M/N per padded Fourier dimension.
    scale = one(T)
    for d in ws.fourier_dims
        scale *= T(ws.padded_shape[d]) / T(ws.original_shape[d])
    end
    ws.spec_result .= ifft(ws.spec_result, ws.fourier_dims)

    # Write result to output field
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)

    # Extract the correctly-typed grid values
    scaled_result = if field1.dtype <: Real
        real.(ws.spec_result) .* scale
    else
        ws.spec_result .* scale
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
mutable struct PaddedPencilFFTWorkspace{P, GA<:AbstractArray, SA<:AbstractArray}
    original_shape::Tuple{Vararg{Int}}
    padded_shape::Tuple{Vararg{Int}}
    padded_plan::P       # PencilFFTs.PencilFFTPlan for padded size
    # Pre-allocated PencilArrays on padded grid
    padded_grid1::GA     # PencilArray for padded physical space
    padded_grid2::GA
    padded_product::GA
    padded_spec1::SA     # PencilArray for padded spectral space
    padded_spec2::SA
    padded_spec_product::SA
end

"""
    _get_padded_pencil_workspace!(evaluator, bases, dist) -> Union{PaddedPencilFFTWorkspace, Nothing}

Create or retrieve a cached PaddedPencilFFTWorkspace for distributed padded dealiasing.
Returns nothing if PencilFFTs cannot be created for the padded shape.
"""
function _get_padded_pencil_workspace!(evaluator::NonlinearEvaluator, bases::Tuple, dist::Distributor)
    key = "padded_pencil_$(hash(bases))"
    if haskey(evaluator.pencil_transforms, key)
        return evaluator.pencil_transforms[key]::PaddedPencilFFTWorkspace
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
        original_plan = _find_pencil_plan(dist)

        if original_plan === nothing
            @warn "No existing PencilFFT plan found — cannot create padded plan for distributed dealiasing" maxlog=1
            return nothing
        end

        # Create padded plan using the stored input pencil (avoids internal PencilFFTs API)
        input_pencil = dist.pencil_fft_input
        padded_pencil = PencilArrays.Pencil(dist.mpi_topology, pad_t, input_pencil.decomp_dims)

        # Build transforms matching the original domain's basis types.
        # RFFT on the first RealFourier axis produces half-spectrum (N/2+1);
        # using FFT instead would create a full-spectrum (N) shape mismatch
        # when copying spectral data between original and padded arrays.
        transform_list = Any[]
        first_real_fourier = true
        for basis in bases
            if isa(basis, RealFourier)
                if first_real_fourier
                    push!(transform_list, PencilFFTs.Transforms.RFFT())
                    first_real_fourier = false
                else
                    push!(transform_list, PencilFFTs.Transforms.FFT())
                end
            elseif isa(basis, ComplexFourier)
                push!(transform_list, PencilFFTs.Transforms.FFT())
            else
                push!(transform_list, PencilFFTs.Transforms.NoTransform())
            end
        end
        transforms = Tuple(transform_list)
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
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
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

"""Setup PencilFFT transforms for nonlinear term evaluation.

    Transforms are created lazily on first use and cached for reuse.
    This avoids pre-computing transforms for shapes that may never be used
    and ensures the correct shape is always available (including dealiased sizes
    like 3/2-rule padding which depend on the actual domain).

    The lazy creation path uses local FFTW plans (no MPI collectives), so it is
    safe to call from within the evaluation loop.
    """
function setup_nonlinear_transforms!(evaluator::NonlinearEvaluator)

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

"""Setup PencilFFT transforms for specific 2D shape"""
function setup_pencil_transforms_for_shape!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})

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

            _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
                PencilTransformConfig(
                    config,
                    forward_data_1,
                    forward_data_2,
                    fft_plan_1,
                    fft_plan_2,
                    shape,
                    true,
                    nothing,
                    nothing,
                ),
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

                _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
                    PencilTransformConfig(
                        config,
                        forward_data_1,
                        forward_data_2,
                        fft_plan_1,
                        fft_plan_2,
                        shape,
                        false,
                        pencil,
                        decomp_dims,
                    ),
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

                _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
                    PencilTransformConfig(
                        config,
                        forward_data_1,
                        forward_data_2,
                        fft_plan_1,
                        fft_plan_2,
                        shape,
                        true,
                        nothing,
                        nothing,
                    ),
                )
            end
        end

        @debug "Created FFT transforms for shape $shape"

    catch e
        @error "Failed to create FFT transforms for shape $shape: $e\n" *
               "Dealiasing will be disabled for this shape — results may contain aliasing errors."
    end
end

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
function setup_1d_nonlinear_transforms!(evaluator::NonlinearEvaluator)

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

"""Setup FFTW plans for 1D transforms of size n."""
function setup_1d_fftw_plans!(evaluator::NonlinearEvaluator, n::Int)

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

        _cache_shape_transform!(evaluator.pencil_transforms, (n,), shape_key,
            FFTWTransformConfig(
                :fftw_1d,
                (n,),
                (n_dealias,),
                forward_plan,
                backward_plan,
                scratch_real,
                scratch_complex,
            ),
        )

        @debug "Created 1D FFTW plans for size $n (dealiased: $n_dealias)"

    catch e
        @warn "Failed to create 1D FFTW plans for size $n: $e"
    end
end

"""Setup FFTW plans for 2D transforms."""
function setup_2d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})

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

        _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
            FFTWTransformConfig(
                :fftw_2d,
                shape,
                dealias_shape,
                forward_plan,
                backward_plan,
                scratch_real,
                scratch_complex,
            ),
        )

        @debug "Created 2D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 2D FFTW plans for shape $shape: $e"
    end
end

"""Setup FFTW plans for 3D transforms."""
function setup_3d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int, Int})

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

        _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
            FFTWTransformConfig(
                :fftw_3d,
                shape,
                dealias_shape,
                forward_plan,
                backward_plan,
                scratch_real,
                scratch_complex,
            ),
        )

        @debug "Created 3D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 3D FFTW plans for shape $shape: $e"
    end
end

"""
    Get the appropriate transform configuration for a given shape.

    Automatically selects between PencilFFT (for multi-D parallelization)
    and FFTW (for 1D parallelization or serial) based on what's available.
    """
function get_nonlinear_transform(evaluator::NonlinearEvaluator, shape::Tuple)
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

include("nonlinear/nonlinear_dealiasing.jl")

# Runtime allocation and pencil-compatibility helpers live out of line so
# the remaining code reads as the nonlinear execution path.
include("nonlinear/nonlinear_pencil_utils.jl")

include("nonlinear/nonlinear_evaluation.jl")

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
