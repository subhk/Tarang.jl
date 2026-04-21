"""Padding, truncation, and padded multiplication helpers for nonlinear evaluation."""

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
