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
mutable struct PaddedDealiasingWorkspace{T<:AbstractFloat, A<:AbstractArray{Complex{T}}} <: AbstractNonlinearTransformConfig
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

    # In-place plans for the ORIGINAL-size spectral buffers (spec1/spec2/spec_result),
    # so the per-call forward/inverse FFTs reuse buffers instead of allocating.
    plan_spec_forward::AbstractFFTPlan
    plan_spec_backward::AbstractFFTPlan

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

    # Compute padded shape: pad only Fourier dimensions. Each axis uses its own
    # basis `dealias` strength when set (>1), falling back to the global factor,
    # so per-basis dealias=... controls the padding (3/2-rule) resolution.
    padded_shape = copy(original_shape)
    for d in fourier_dims
        axis_factor = d <= length(bases) ? _axis_dealias_factor(bases[d], factor) : factor
        M = Int(ceil(axis_factor * original_shape[d]))
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

    # Create FFT plans — CPU gets FFTW.MEASURE, GPU gets plain plan_fft.
    # `plan_*!` are in-place (mutate their argument); the spec plans operate on the
    # original-size buffers. The forward spec plan is built on spec1 and reused on
    # spec2 (identical size/alignment).
    if is_gpu(_arch)
        # For GPU: AbstractFFTs.plan_fft dispatches to CUFFT (no flags arg)
        plan_forward = plan_fft(padded1, fourier_dims)
        plan_backward = plan_ifft(padded1, fourier_dims)
        plan_spec_forward = plan_fft!(spec1, fourier_dims)
        plan_spec_backward = plan_ifft!(spec_result, fourier_dims)
    else
        # In-place padded plans (UNALIGNED — applied to padded1/padded2/padded_product):
        # the per-call `ws.padded .= plan * ws.padded` then transforms in place instead
        # of allocating a fresh padded array each FFT/IFFT (~one padded array per call).
        plan_forward = FFTW.plan_fft!(padded1, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
        plan_backward = FFTW.plan_ifft!(padded1, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
        # UNALIGNED: the forward plan is built on spec1 but also applied to spec2,
        # so it must not bake in a single buffer's memory alignment.
        plan_spec_forward = FFTW.plan_fft!(spec1, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
        plan_spec_backward = FFTW.plan_ifft!(spec_result, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
    end

    ws = PaddedDealiasingWorkspace{T, typeof(padded1)}(
        orig_t, pad_t, fourier_dims,
        padded1, padded2, padded_product,
        spec1, spec2, spec_result,
        plan_forward, plan_backward,
        plan_spec_forward, plan_spec_backward, _arch
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

    # @views: range-indexed RHS slices are views, not materialized copies (no per-call alloc).
    @views padded[r1[2], r2[2]] .= spec_data[r1[1], r2[1]]
    if length(r2[3]) > 0
        @views padded[r1[2], r2[4]] .= spec_data[r1[1], r2[3]]
    end
    if length(r1[3]) > 0
        @views padded[r1[4], r2[2]] .= spec_data[r1[3], r2[1]]
    end
    if length(r1[3]) > 0 && length(r2[3]) > 0
        @views padded[r1[4], r2[4]] .= spec_data[r1[3], r2[3]]
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
        @views result[r1[1], r2[1]] .= padded_spec[r1[2], r2[2]]
        if length(r2[3]) > 0
            @views result[r1[1], r2[3]] .= padded_spec[r1[2], r2[4]]
        end
        if length(r1[3]) > 0
            @views result[r1[3], r2[1]] .= padded_spec[r1[4], r2[2]]
        end
        if length(r1[3]) > 0 && length(r2[3]) > 0
            @views result[r1[3], r2[3]] .= padded_spec[r1[4], r2[4]]
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

    # Step 1: FFT to spectral along Fourier dimensions, in place via pre-built
    # plans (no per-call allocation). Plans dispatch to CUFFT for GPU arrays.
    ws.spec1 .= Complex{T}.(raw1_ws)
    ws.plan_spec_forward * ws.spec1
    ws.spec2 .= Complex{T}.(raw2_ws)
    ws.plan_spec_forward * ws.spec2

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
    ws.plan_spec_backward * ws.spec_result

    # Write result to a pooled output field (rotating buffers — distinct for
    # consecutively-held products like cross product; no per-call allocation).
    result = _checkout_nl_result!(evaluator, field1)
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

# ============================================================================
# Distributed (MPI) dealiasing via 2/3-rule truncation
# ============================================================================

"""
    _dealias_truncate_field!(field, dealiasing_factor)

Zero all Fourier modes with |k| > N/(2·dealiasing_factor) on `field`, in place.
Forward-transforms, applies the per-rank global-wavenumber cutoff (distributed)
or the standard cutoff (serial), then transforms back to grid space.
"""
function _dealias_truncate_field!(field::ScalarField, dealiasing_factor::Float64; final_layout::Symbol=:g)
    forward_transform!(field)
    coeff_data = get_coeff_data(field)

    if isa(coeff_data, PencilArrays.PencilArray)
        _apply_spectral_cutoff_distributed!(coeff_data, field.bases, dealiasing_factor)
    else
        nb = length(field.bases)
        cutoffs = ntuple(nb) do i
            b = field.bases[i]
            isa(b, Union{RealFourier, ComplexFourier}) ?
                Int(floor(b.meta.size / (2 * dealiasing_factor))) : size(coeff_data, i)
        end
        rfft_dims = ntuple(nb) do i
            b = field.bases[i]
            isa(b, RealFourier) && size(coeff_data, i) == div(b.meta.size, 2) + 1
        end
        apply_spectral_cutoff!(coeff_data, cutoffs, rfft_dims)
    end

    # Leave in coeff when the caller will consume/sum in coeff (saves a backward
    # transpose — e.g. dot-product terms summed in coeff with one final backward).
    final_layout === :g && backward_transform!(field)
    return field
end

# ============================================================================
# Batched distributed backward transform (MPI message coalescing)
#
# A per-field backward issues its own all-to-all transpose. Stacking `k`
# same-shaped fields and transforming once issues ONE transpose for all of them.
# We use PencilFFTs' `extra_dims`: the batch is a trailing LOCAL dimension that
# rides through the SAME transpose/permutation sequence as the field's own plan,
# so each slice's layout is identical to the field's coeff array (verified) and
# stacking is a plain copy. Latency win that grows with rank count (transpose
# count is otherwise independent of nprocs under slab decomposition).
# ============================================================================

const _BATCHED_PENCIL_PLAN_CACHE = Dict{Tuple, Any}()

"""Transforms tuple matching `setup_pencil_fft_transforms!`: RFFT on the first
Fourier axis, FFT on later Fourier axes, NoTransform on non-Fourier axes."""
function _field_pencil_transforms(bases)
    first_fourier = 0
    for (i, b) in enumerate(bases)
        isa(b, Union{RealFourier, ComplexFourier}) && first_fourier == 0 && (first_fourier = i)
    end
    tlist = Any[]
    for (i, b) in enumerate(bases)
        if isa(b, RealFourier)
            push!(tlist, i == first_fourier ? PencilFFTs.Transforms.RFFT() : PencilFFTs.Transforms.FFT())
        elseif isa(b, ComplexFourier)
            push!(tlist, PencilFFTs.Transforms.FFT())
        else
            push!(tlist, PencilFFTs.Transforms.NoTransform())
        end
    end
    return Tuple(tlist)
end

"""Get or build (cached) a batched PencilFFTPlan for `B` stacked fields. Uses the
same global shape / transforms / process mesh as the per-field plan plus
`extra_dims=(B,)`, so the per-slice layout matches each field's coeff array."""
function _get_batched_backward_plan!(dist, bases, B::Int)
    key = (objectid(dist), B, map(b -> (nameof(typeof(b)), b.meta.size), bases))
    cached = get(_BATCHED_PENCIL_PLAN_CACHE, key, nothing)
    cached !== nothing && return cached

    transforms = _field_pencil_transforms(bases)
    global_shape = Tuple(b.meta.size for b in bases)
    plan = PencilFFTs.PencilFFTPlan(global_shape, transforms, dist.mesh, dist.comm; extra_dims=(B,))
    # Reused scratch buffers (coeff-side input, grid-side output) — consumed within
    # each call, so caching is safe and removes the two per-call allocations.
    cstack = PencilFFTs.allocate_output(plan)
    gstack = PencilFFTs.allocate_input(plan)
    entry = (plan, cstack, gstack)
    _BATCHED_PENCIL_PLAN_CACHE[key] = entry
    return entry
end

"""
    _pencil_batched_backward!(fields::Vector{<:ScalarField})

Backward-transform `k` same-shaped distributed fields (currently coefficient
layout) using ONE batched PencilFFTs transpose instead of `k` separate ones.
Stacks each field's coeff PencilArray along the trailing batch (extra) axis,
applies the single batched `\\`, and unstacks the grid result into each field.

Fast path requires CPU-backed PencilArray coeff data sharing `dist`/`bases`;
otherwise falls back to per-field `backward_transform!`.
"""
function _pencil_batched_backward!(fields::Vector{<:ScalarField})
    k = length(fields)
    k == 0 && return
    f0 = fields[1]
    ensure_layout!(f0, :c)
    cd0 = get_coeff_data(f0)
    if k == 1 || !isa(cd0, PencilArrays.PencilArray) || is_gpu_array(cd0)
        for f in fields
            backward_transform!(f)
        end
        return
    end

    plan, cstack, gstack = _get_batched_backward_plan!(f0.dist, f0.bases, k)
    cp = parent(cstack)
    nd = ndims(cp)
    for (i, f) in enumerate(fields)
        ensure_layout!(f, :c)
        cf = parent(get_coeff_data(f))
        dst = selectdim(cp, nd, i)
        size(dst) == size(cf) ||
            error("_pencil_batched_backward!: coeff slice $(size(dst)) != field coeff $(size(cf)) — batched-plan layout mismatch")
        copyto!(dst, cf)
    end

    ldiv!(gstack, plan, cstack)                 # ONE transpose for all k slices, in place
    gp = parent(gstack)
    for (i, f) in enumerate(fields)
        gdp = parent(get_grid_data(f))
        slice = selectdim(gp, nd, i)
        if eltype(gdp) <: Real && eltype(slice) <: Complex
            gdp .= real.(slice)
        else
            gdp .= slice
        end
        f.current_layout = :g
    end
    return
end

"""
    _truncate_coeff_into_grid!(dst, src, dealiasing_factor)

Copy `src` into scratch `dst` and band-limit it to |k| ≤ N/(2·factor), leaving
`dst` in GRID layout ready to multiply. The cutoff is applied in COEFFICIENT
space, so when `src` already lives in coefficient space (the usual case —
operands come from spectral derivatives) only a single backward transform is
needed. The old path forced `src` to grid, copied, then forward-transformed the
copy just to truncate it and backward-transformed again — a redundant c→g→c
round trip this avoids.
"""
function _truncate_coeff_into_grid!(dst::ScalarField, src::ScalarField, dealiasing_factor::Float64)
    _truncate_coeff_only!(dst, src, dealiasing_factor)
    backward_transform!(dst)  # coeff → grid: the single required transform
    return dst
end

"""
    _truncate_coeff_only!(dst, src, dealiasing_factor)

The coefficient-space half of `_truncate_coeff_into_grid!`: copy `src`'s coeff
into `dst` and apply the spectral cutoff, leaving `dst` in COEFFICIENT layout
(no backward transform). Split out so several inputs can be truncated, then
backward-transformed together in one batched transpose (see
`_pencil_batched_backward!`).
"""
function _truncate_coeff_only!(dst::ScalarField, src::ScalarField, dealiasing_factor::Float64)
    ensure_layout!(src, :c)   # forward only if src was in grid; no-op if already coeff
    sc = get_coeff_data(src)
    dc = get_coeff_data(dst)  # coeff buffer; contents overwritten below, so no transform needed
    if isa(dc, PencilArrays.PencilArray) && isa(sc, PencilArrays.PencilArray)
        copyto!(parent(dc), parent(sc))
    else
        copyto!(dc, sc)
    end
    dst.current_layout = :c

    if isa(dc, PencilArrays.PencilArray)
        _apply_spectral_cutoff_distributed!(dc, dst.bases, dealiasing_factor)
    else
        nb = length(dst.bases)
        cutoffs = ntuple(nb) do i
            b = dst.bases[i]
            isa(b, Union{RealFourier, ComplexFourier}) ?
                Int(floor(b.meta.size / (2 * dealiasing_factor))) : size(dc, i)
        end
        rfft_dims = ntuple(nb) do i
            b = dst.bases[i]
            isa(b, RealFourier) && size(dc, i) == div(b.meta.size, 2) + 1
        end
        apply_spectral_cutoff!(dc, cutoffs, rfft_dims)
    end
    return dst
end

"""
    evaluate_truncated_multiply_distributed(field1, field2, evaluator)

Multiply two MPI-distributed fields with 2/3-rule dealiasing.

Exact 3/2 zero-padding is not used under MPI because embedding the N-mode
spectrum into the 3N/2 padded spectrum requires cross-rank redistribution
(the original and padded PencilArrays decompose differently). The 2/3 rule is
purely local per rank: truncate both inputs to |k| ≤ N/3, multiply on the grid,
then truncate the product to |k| ≤ N/3. The product of two N/3-band fields has
support up to 2N/3; on the N grid those modes alias only into |k| ≥ N/3, so the
retained |k| ≤ N/3 band is alias-free — i.e. quadratic terms are dealiased
exactly within the retained band.
"""
# Rotating pool of product-result buffers: avoids a fresh ScalarField per product
# while still handing distinct buffers to callers that hold several products at
# once (e.g. cross product). 8 buffers ⇒ safe for realistic nested product depth;
# each buffer is fully overwritten before reuse.
const _NL_RESULT_POOL_SIZE = 8
const _NL_RESULT_IDX = Ref(0)

function _checkout_nl_result!(evaluator::NonlinearEvaluator, field1::ScalarField)
    i = _NL_RESULT_IDX[] % _NL_RESULT_POOL_SIZE
    _NL_RESULT_IDX[] += 1
    key = string("_nl_result_", i, "_", hash(field1.bases), "_", field1.dtype)
    return get!(() -> ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype),
                evaluator.temp_fields, key)
end

function evaluate_truncated_multiply_distributed(field1::ScalarField, field2::ScalarField,
                                                  evaluator::NonlinearEvaluator;
                                                  result_layout::Symbol=:g)
    factor = evaluator.dealiasing_factor

    # Truncate the inputs into reusable scratch fields (consumed within this call,
    # so sharing them across calls is safe — unlike `result`, which the caller may
    # hold alongside a second product, e.g. CrossProduct). This avoids the two
    # per-call `copy(field)` allocations.
    bkey = string(hash(field1.bases))
    f1 = get!(() -> ScalarField(field1.dist, "_nl_trunc_f1", field1.bases, field1.dtype),
              evaluator.temp_fields, "_nl_trunc_f1_" * bkey)
    f2 = get!(() -> ScalarField(field1.dist, "_nl_trunc_f2", field1.bases, field1.dtype),
              evaluator.temp_fields, "_nl_trunc_f2_" * bkey)
    # Band-limit both inputs in coeff space, then bring them back to grid with a
    # SINGLE batched transpose instead of one per input (2 backwards → 1).
    _truncate_coeff_only!(f1, field1, factor)
    _truncate_coeff_only!(f2, field2, factor)
    _pencil_batched_backward!([f1, f2])

    # Multiply on the grid. `result` comes from a rotating buffer pool — distinct
    # buffers for consecutively-held products (e.g. cross product), no per-call alloc.
    result = _checkout_nl_result!(evaluator, field1)
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)
    d1 = get_grid_data(f1)
    d2 = get_grid_data(f2)
    if isa(result_data, PencilArrays.PencilArray)
        parent(result_data) .= parent(d1) .* parent(d2)
    else
        result_data .= d1 .* d2
    end

    # Truncate the product to remove the aliased high modes. When the caller will
    # sum products in coeff (dot-product terms), keep the result in coeff and skip
    # the backward — the sum's single backward replaces the per-product ones.
    _dealias_truncate_field!(result, factor; final_layout=result_layout)
    return result
end

