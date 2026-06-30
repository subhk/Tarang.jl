# Set true to collect wall-clock timing stats; false = zero overhead (dead-code eliminated).
const _TRACK_NL_TIMING = false

# Main nonlinear evaluation functions
"""Evaluate u·∇φ nonlinear term using transform method"""
function evaluate_nonlinear_term(op::AdvectionOperator, layout::Symbol=:g)

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

"""Evaluate (u·∇)u nonlinear momentum term"""
function evaluate_nonlinear_term(op::NonlinearAdvectionOperator, layout::Symbol=:g)

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

"""Efficiently multiply two fields using transform method and proper dealiasing.

    Uses proper 3/2-rule padded dealiasing (Orszag 1971) when possible:
    - CPU serial: full padded dealiasing on all Fourier dimensions
    - GPU serial: full padded dealiasing using GPU FFTs (CUFFT)
    - MPI distributed: full padded dealiasing on ALL Fourier dimensions
      (including the decomposed one) via PencilArrays transpose-pad — matches
      the serial result to roundoff for 2D + 3D pure-Fourier and mixed
      Fourier×(Chebyshev/Jacobi) fields (`evaluate_padded_multiply_distributed`).

    Falls back to local 2/3-rule truncation-after-multiply when padded dealiasing
    does not apply: no Fourier bases, or (distributed only) >3D, a decomposed
    non-Fourier axis, or a Fourier axis with N ≤ 4 (matching serial's small-grid
    skip).
    """
function evaluate_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator;
                                     result_layout::Symbol=:g)

    _TRACK_NL_TIMING && (start_time = time())

    if field1.bases != field2.bases
        throw(ArgumentError(
            "Cannot multiply fields with different bases: " *
            "'$(field1.name)' has bases=$(field1.bases) but " *
            "'$(field2.name)' has bases=$(field2.bases). " *
            "Both fields must be defined on the same domain."))
    end

    # Try proper 3/2-rule padded dealiasing. Gate on the per-axis basis `dealias`
    # settings (factor > 1 on any Fourier axis) rather than the evaluator's global
    # default, so bases set to dealias ≤ 1 are computed without dealiasing.
    if _any_axis_dealias(field1.bases, evaluator.dealiasing_factor)
        T = field1.dtype <: Complex ? real(field1.dtype) : field1.dtype
        # Detect the MPI pencil path WITHOUT forcing a layout first, so the
        # distributed routine can truncate inputs in coefficient space (avoids a
        # c→g→c round trip when operands arrive in coeff space — the usual case).
        if field1.dist.use_pencil_arrays && evaluator.dist.size > 1
            # MPI path. Primary: FULL 3/2-rule padded dealiasing via the transpose-pad
            # distributed FFT (evaluate_padded_multiply_distributed) — the cross-rank
            # redistribution needed to embed the N-mode spectrum into the 3N/2 padded
            # spectrum is done with PencilArrays transposes, so the result matches the
            # serial padded multiply to roundoff (round-7 audit 2026-06). Covers 2D + 3D
            # (incl. 2D process mesh) pure-Fourier and mixed Fourier×(Cheb/Jacobi).
            # Fallback: it returns `nothing` for the cases it does not cover (>3D, a
            # decomposed non-Fourier axis, or a Fourier axis with N ≤ 4 — the last
            # mirroring serial's own small-grid skip), routing to local 2/3-rule
            # truncation-after-multiply (evaluate_truncated_multiply_distributed).
            padded = evaluate_padded_multiply_distributed(field1, field2, evaluator;
                                                          result_layout=result_layout)
            result = padded !== nothing ? padded :
                     evaluate_truncated_multiply_distributed(field1, field2, evaluator;
                                                             result_layout=result_layout)
            evaluator.performance_stats.total_evaluations += 1
            if _TRACK_NL_TIMING
                elapsed = time() - start_time
                evaluator.performance_stats.total_time += elapsed
                evaluator.performance_stats.dealiasing_time += elapsed
            end
            return result
        else
            # Serial path (CPU or GPU): pad all Fourier dimensions (needs grid inputs)
            ensure_layout!(field1, :g)
            ensure_layout!(field2, :g)
            ws = _get_padded_workspace!(evaluator, field1.bases, T)
            if ws !== nothing
                result = evaluate_padded_multiply(field1, field2, evaluator, ws)
                ensure_layout!(result, result_layout)  # serial: cheap FFT, no transpose
                evaluator.performance_stats.total_evaluations += 1
                if _TRACK_NL_TIMING
                    elapsed = time() - start_time
                    evaluator.performance_stats.total_time += elapsed
                    evaluator.performance_stats.dealiasing_time += elapsed
                end
                return result
            end
        end
    end

    # Fallback: direct multiplication + truncation-after-multiply dealiasing
    # (only reached for fields with no Fourier bases, or dealiasing_factor <= 1)
    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    result_data = get_grid_data(result)
    field1_data = get_grid_data(field1)
    field2_data = get_grid_data(field2)

    if isa(result_data, PencilArrays.PencilArray)
        parent(result_data) .= parent(field1_data) .* parent(field2_data)
    else
        gpu_multiply_fields!(result["g"], field1["g"], field2["g"])
    end

    # Only apply truncation-after-multiply dealiasing if a Fourier axis requests it
    # (dealias > 1). For pure Chebyshev/Legendre fields, or bases with dealias ≤ 1,
    # the forward/backward roundtrip in apply_basic_dealiasing! is skipped.
    if _any_axis_dealias(field1.bases, evaluator.dealiasing_factor)
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    ensure_layout!(result, result_layout)
    evaluator.performance_stats.total_evaluations += 1
    _TRACK_NL_TIMING && (evaluator.performance_stats.total_time += time() - start_time)

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

# ── Cached workspace for the distributed padded multiply ─────────────────────
# evaluate_padded_multiply_distributed runs an IDENTICAL transpose / pad / FFT
# SEQUENCE on every call for a fixed problem (topology, global size and the
# decomposition are stable). This workspace lets the routine REUSE across calls:
#   (a) the intermediate Pencils — immutable metadata that PencilArrays.transpose!
#       only reads, so sharing is safe (kills the ~14 fresh Pencils/call);
#   (b) the complex N- and Mp-sized scratch PencilArray buffers (kills the ~14
#       full-grid PencilArray allocations/call);
#   (c) plan-backed in-place local-axis FFTs replacing out-of-place fft()/ifft()
#       (kills the ~12 full-grid FFT-output allocations/call).
# The result field comes from the shared rotating pool via _checkout_nl_result!
# (kills the fresh ScalarField/call). Mirrors the cached-plan+scratch pattern of
# _get_batched_backward_plan!/_BATCHED_PENCIL_PLAN_CACHE and the in-place-plan +
# CPU/GPU split of _get_padded_workspace!/PaddedDealiasingWorkspace (serial sibling).
#
# Buffers are handed out by a per-call BUMP index (`idx`, reset to 0 at the start
# of each call). Because the control flow is data-independent, the i-th _buf!
# request asks for the same shape on every call, so each logical "role" (e.g. up1
# vs up2, which are alive simultaneously when their product is formed) maps to a
# DISTINCT, stable slot and never aliases. Every buffer is FULLY overwritten
# (transpose! destination, a `.=` copy, or a full-coverage pad/truncate) before it
# is read, so stale data from a previous call is never observed — exactly the
# property the original relied on when it allocated `undef` buffers each call.
mutable struct PaddedDistDealiasWorkspace{CT}
    topo::Any
    pencils::Dict{Tuple{Tuple, Tuple}, Any}     # (global size, decomp dims) → memoized NoPermutation Pencil
    buffers::Vector{Any}                         # per-call bump pool of complex scratch PencilArrays
    idx::Base.RefValue{Int}
    fwd_plans::Dict{Tuple{Tuple, Int}, Any}      # (local array size, axis) → in-place forward FFT plan
    inv_plans::Dict{Tuple{Tuple, Int}, Any}      # (local array size, axis) → in-place inverse FFT plan
end

const _PADDED_DIST_WS_CACHE = Dict{Tuple, Any}()

# Keyed (like _get_batched_backward_plan!) on the problem identity — topology
# object, basis kinds/sizes, complex eltype, and the global/padded sizes +
# decomposition that fully determine the transpose sequence. `topo` is the exact
# object the cached pencils are built from, so reused pencils stay compatible with
# the field's grid pencil in transpose! (the same assumption the original makes by
# taking `topo = topology(pencil(gd1))`).
function _get_padded_dist_workspace!(dist, topo, bases, ::Type{CT}, N, Mp, decomp0, fourier_axes) where {CT}
    key = (objectid(dist), objectid(topo), CT,
           map(b -> (nameof(typeof(b)), b.meta.size), bases), N, Mp, decomp0, fourier_axes)
    cached = get(_PADDED_DIST_WS_CACHE, key, nothing)
    cached === nothing || return cached::PaddedDistDealiasWorkspace{CT}
    ws = PaddedDistDealiasWorkspace{CT}(topo,
        Dict{Tuple{Tuple, Tuple}, Any}(), Any[], Ref(0),
        Dict{Tuple{Tuple, Int}, Any}(), Dict{Tuple{Tuple, Int}, Any}())
    _PADDED_DIST_WS_CACHE[key] = ws
    return ws
end

# Memoized NoPermutation Pencil for (size, decomp). Identity-stable: the same
# (size, decomp) returns the SAME object, so _padded_dist_buf! hits the `===` path.
@inline function _padded_dist_pencil!(ws::PaddedDistDealiasWorkspace, sz, dd)
    key = (Tuple(sz), Tuple(dd))
    p = get(ws.pencils, key, nothing)
    p === nothing || return p
    p = PencilArrays.Pencil(ws.topo, Tuple(sz), Tuple(dd); permute=PencilArrays.NoPermutation())
    ws.pencils[key] = p
    return p
end

# Two pencils may share a scratch buffer iff their geometry matches — same global
# size + decomposition + permutation fully determines the local memory layout and
# transpose! validity. `===` (memoized mkpen pencils) is the fast path; the
# structural branch runs only for grid pencils pencil(gd)/pencil(rg), whose
# geometry is fixed per problem (so it is always true on reuse). Permutation lives
# in the Pencil type parameters, so typeof(...) === typeof(...) compares it exactly.
@inline function _pen_compatible(a, b)
    a === b && return true
    return PencilArrays.size_global(a) == PencilArrays.size_global(b) &&
           Tuple(PencilArrays.decomposition(a)) == Tuple(PencilArrays.decomposition(b)) &&
           typeof(PencilArrays.permutation(a)) === typeof(PencilArrays.permutation(b))
end

# Next per-call scratch buffer for pencil `pen`. Bump index ⇒ each role gets a
# distinct, stable slot (no aliasing of simultaneously-live buffers).
function _padded_dist_buf!(ws::PaddedDistDealiasWorkspace{CT}, pen) where {CT}
    i = (ws.idx[] += 1)
    if i <= length(ws.buffers)
        b = ws.buffers[i]::PencilArrays.PencilArray
        _pen_compatible(PencilArrays.pencil(b), pen) && return b
        nb = PencilArrays.PencilArray{CT}(undef, pen)       # shape changed: replace slot
        ws.buffers[i] = nb
        return nb
    end
    nb = PencilArrays.PencilArray{CT}(undef, pen)
    push!(ws.buffers, nb)
    return nb
end

# In-place single-local-axis FFT plans, cached by (local size, axis). CPU: FFTW
# MEASURE on a throwaway (MEASURE scribbles its planning input) + UNALIGNED so one
# plan serves every same-size buffer (the shape recurs across up1/up2/res). GPU /
# generic: AbstractFFTs.plan_fft! dispatches to CUFFT. This is the same concrete-
# CPU-branch / generic-fallback split used by _get_padded_workspace!.
function _padded_dist_fwd_plan!(ws::PaddedDistDealiasWorkspace, arr, a::Int)
    key = (size(arr), a)
    p = get(ws.fwd_plans, key, nothing)
    p === nothing || return p
    p = arr isa Array ? FFTW.plan_fft!(similar(arr), a; flags=FFTW.MEASURE | FFTW.UNALIGNED) :
                        plan_fft!(similar(arr), a)
    ws.fwd_plans[key] = p
    return p
end

function _padded_dist_inv_plan!(ws::PaddedDistDealiasWorkspace, arr, a::Int)
    key = (size(arr), a)
    p = get(ws.inv_plans, key, nothing)
    p === nothing || return p
    p = arr isa Array ? FFTW.plan_ifft!(similar(arr), a; flags=FFTW.MEASURE | FFTW.UNALIGNED) :
                        plan_ifft!(similar(arr), a)
    ws.inv_plans[key] = p
    return p
end

# ── Full 3/2-rule padded dealiasing under MPI (N-D all-Fourier) ──────────────
# Re-implements the serial padded product (evaluate_padded_multiply) for a
# DISTRIBUTED field: pad each Fourier axis WHEN that axis is local, transposing to
# bring each decomposed axis local in turn (PencilFFTs decomposes D-1 axes, keeping
# one local; single-swap transposes cycle the local axis through all D). Reuses the
# SAME per-axis pad/truncate helpers as serial (incl. even-N Nyquist split/fold) +
# scale ∏(M_d/N_d) once, so it matches serial to ROUNDOFF (verified Real+Complex
# Fourier, 2D np=2/4 + 3D np=2/4). Returns `nothing` for cases not covered (>3D,
# mixed bases, not pencil-decomposed) → caller falls back to truncation-after-
# multiply. Round-7 audit 2026-06-23.
function evaluate_padded_multiply_distributed(field1::ScalarField, field2::ScalarField,
                                              evaluator::NonlinearEvaluator; result_layout::Symbol=:g)
    bases = field1.bases
    D = length(bases)
    (D == 2 || D == 3) || return nothing
    isfourier(b) = isa(b, Union{RealFourier, ComplexFourier})
    fourier_axes = Tuple(d for d in 1:D if isfourier(bases[d]))
    isempty(fourier_axes) && return nothing
    dist = field1.dist
    factor = evaluator.dealiasing_factor
    T = field1.dtype <: Complex ? real(field1.dtype) : field1.dtype
    CT = Complex{T}
    N = ntuple(d -> bases[d].meta.size, D)
    # Only Fourier axes are 3/2-padded (matching serial evaluate_padded_multiply,
    # which pads only fourier_dims); non-Fourier (Chebyshev/Jacobi) axes keep their
    # nodal grid and are left untouched. Mp[d] == N[d] for non-Fourier axes.
    Mp = ntuple(D) do d
        isfourier(bases[d]) || return N[d]
        af = _axis_dealias_factor(bases[d], factor); m = Int(ceil(af * N[d])); isodd(m) && (m += 1); m
    end
    any(d -> Mp[d] > N[d], fourier_axes) || return nothing
    # Match serial _get_padded_workspace!: the 3/2-rule is invalid for a Fourier axis
    # with N ≤ 4 (≤2 independent modes), where serial falls back to truncation. Mirror
    # that here so distributed == serial on tiny Fourier axes too.
    minimum(N[d] for d in fourier_axes) <= 4 && return nothing

    ensure_layout!(field1, :g); ensure_layout!(field2, :g)
    gd1 = get_grid_data(field1); gd2 = get_grid_data(field2)
    (gd1 isa PencilArrays.PencilArray && gd2 isa PencilArrays.PencilArray) || return nothing
    pen = PencilArrays.pencil(gd1)
    topo = PencilArrays.topology(pen)
    decomp0 = Tuple(PencilArrays.decomposition(pen))
    length(decomp0) == D - 1 || return nothing        # exactly one local axis (slab/pencil)
    all(d -> isfourier(bases[d]), decomp0) || return nothing   # only pad/transpose decomposed FOURIER axes
    ws = _get_padded_dist_workspace!(dist, topo, bases, CT, N, Mp, decomp0, fourier_axes)
    ws.idx[] = 0
    mkpen(sz, dd) = _padded_dist_pencil!(ws, sz, dd)

    tolog(gd) = begin
        gc = _padded_dist_buf!(ws, PencilArrays.pencil(gd)); parent(gc) .= CT.(parent(gd))
        a = _padded_dist_buf!(ws, mkpen(N, decomp0)); PencilArrays.transpose!(a, gc); a
    end
    # bring axis `a` into the (single) local slot by swapping it with the current
    # local axis — a one-decomposed-dim change, which PencilArrays.transpose! allows.
    makelocal(cur, csz, dec, loc, a) = a == loc ? (cur, dec, loc) : begin
        ndec = [d == a ? loc : d for d in dec]
        nxt = _padded_dist_buf!(ws, mkpen(csz, ndec)); PencilArrays.transpose!(nxt, cur); (nxt, ndec, a)
    end
    # Local-axis pad/truncate: forward FFT in place on the SPENT input buffer `pin`
    # (never read again — see the workspace bump-pool note), pad/truncate into the
    # fresh `pout` (a distinct slot/size), then inverse FFT in place. Plan-backed
    # (no per-call fft()/ifft() allocation); identical FFTW/CUFFT routines + 1/N
    # normalization as the original out-of-place fft/ifft, so the values match to
    # roundoff. `_pad_spectral!` zero-fills pout; `_truncate_spectral!` writes full
    # coverage — both overwrite pout completely before the inverse FFT reads it.
    pad_local!(pout, pin, a) = begin
        pp = parent(pin); _padded_dist_fwd_plan!(ws, pp, a) * pp
        orig = collect(size(pp)); padl = copy(orig); padl[a] = Mp[a]
        _pad_spectral!(parent(pout), pp, Tuple(orig), Tuple(padl), [a])
        po = parent(pout); _padded_dist_inv_plan!(ws, po, a) * po
    end
    trunc_local!(pout, pin, a) = begin
        pp = parent(pin); _padded_dist_fwd_plan!(ws, pp, a) * pp
        orig = collect(size(pp)); trl = copy(orig); trl[a] = N[a]
        _truncate_spectral!(parent(pout), pp, Tuple(trl), Tuple(orig), [a])
        po = parent(pout); _padded_dist_inv_plan!(ws, po, a) * po
    end
    sweep(start, sdec, ssz, tgt, op) = begin
        cur = start; dec = collect(sdec); loc = only(setdiff(1:D, collect(sdec))); csz = collect(ssz)
        for a in fourier_axes   # pad/truncate only Fourier axes; non-Fourier stay nodal/local
            cur, dec, loc = makelocal(cur, csz, dec, loc, a)
            csz[a] = tgt[a]
            nxt = _padded_dist_buf!(ws, mkpen(csz, dec)); op(nxt, cur, a); cur = nxt
        end
        cur
    end

    up1 = sweep(tolog(gd1), decomp0, N, Mp, pad_local!)
    up2 = sweep(tolog(gd2), decomp0, N, Mp, pad_local!)
    pdec = Tuple(PencilArrays.decomposition(PencilArrays.pencil(up1)))
    P = _padded_dist_buf!(ws, mkpen(Mp, pdec)); parent(P) .= parent(up1) .* parent(up2)
    res = sweep(P, pdec, Mp, N, trunc_local!)
    parent(res) .*= (T(prod(Mp)) / T(prod(N)))

    result = _checkout_nl_result!(evaluator, field1)
    ensure_layout!(result, :g)
    rg = get_grid_data(result)
    # Align `res` (NoPermutation) to the result grid pencil's EXACT ordered
    # decomposition so the final transpose is perm-only. Matching just the local
    # axis is NOT enough: res and rg can share the decomposed-axis SET but in a
    # different ORDER (e.g. res (2,3) vs rg (3,2) for a 3D Cheb-Fourier-Fourier
    # field), and PencilArrays.transpose! forbids a >1-slot decomp difference. Walk
    # the decomp tuple slot by slot, each fix being one valid single-swap.
    rgdec = collect(PencilArrays.decomposition(PencilArrays.pencil(rg)))
    rdec = collect(PencilArrays.decomposition(PencilArrays.pencil(res)))
    rloc = only(setdiff(1:D, rdec))
    for i in eachindex(rgdec)
        rdec[i] == rgdec[i] && continue
        if rloc != rgdec[i]                                   # bring target axis local
            res, rdec, rloc = makelocal(res, collect(N), rdec, rloc, rgdec[i])
        end
        res, rdec, rloc = makelocal(res, collect(N), rdec, rloc, rdec[i])  # swap it into slot i
    end
    rc = _padded_dist_buf!(ws, PencilArrays.pencil(rg))
    PencilArrays.transpose!(rc, res)
    if field1.dtype <: Complex
        parent(rg) .= parent(rc)
    else
        parent(rg) .= real.(parent(rc))
    end
    ensure_layout!(result, result_layout)
    return result
end

"""2D transform-based multiplication using PencilFFTs"""
function evaluate_2d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)

    # Create result field (static name avoids string allocation per call)
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Try to find matching transform configuration
    shape_key = shape  # Use tuple directly as key (no string allocation)

    if haskey(evaluator.pencil_transforms, shape_key)
        # Use precomputed transform config. PencilFFT configs keep the old
        # optimized path; FFTW configs fall back to direct multiply plus the
        # standard spectral truncation pass.
        transform_info = evaluator.pencil_transforms[shape_key]
        if transform_info isa PencilTransformConfig
            data1 = get_pencil_compatible_data(field1, transform_info.config)
            data2 = get_pencil_compatible_data(field2, transform_info.config)
            result_data = data1 .* data2

            if evaluator.dealiasing_factor > 1.0
                result_data = apply_2d_dealiasing(result_data, transform_info, evaluator.dealiasing_factor)
            end

            set_pencil_compatible_data!(result, result_data, transform_info.config)
        else
            result["g"] .= field1["g"] .* field2["g"]

            if evaluator.dealiasing_factor > 1.0
                apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
            end
        end

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

"""3D transform-based multiplication using 3D PencilFFTs"""
function evaluate_3d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)

    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
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

"""Fallback multiplication for unsupported dimensions"""
function evaluate_fallback_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)

    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Simple pointwise multiplication
    result["g"] .= field1["g"] .* field2["g"]

    # Apply dealiasing if requested
    if evaluator.dealiasing_factor > 1.0
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    return result
end

# Integration with existing operator evaluation
evaluate_operator(op::AdvectionOperator) = evaluate_nonlinear_term(op)
evaluate_operator(op::NonlinearAdvectionOperator) = evaluate_nonlinear_term(op)
evaluate_operator(op::ConvectiveOperator) = evaluate_convective_operator(op)
function evaluate_operator(op::NonlinearOperator)
    throw(ArgumentError("Nonlinear operator evaluation not implemented for $(typeof(op))"))
end

"""Get the cached NonlinearEvaluator for a distributor, creating one if needed."""
function _get_evaluator(dist::Distributor)
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    return dist.nonlinear_evaluator
end

"""Evaluate general convective operator"""
function evaluate_convective_operator(op::ConvectiveOperator)

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

"""Evaluate v1·v2 dot product"""
function evaluate_vector_dot_product(v1::VectorField, v2::VectorField)

    if length(v1.components) != length(v2.components)
        throw(ArgumentError("Vector fields must have same number of components"))
    end

    evaluator = _get_evaluator(v1.dist)
    n = length(v1.components)
    c1 = v1.components[1]

    # Distributed pure-Fourier: sum the component products in COEFFICIENT space so
    # the `n` per-product backward transposes collapse into ONE (performed by the
    # caller's `ensure_layout!(result, layout)`). Valid because the inverse
    # transform is linear: Σ backward(pᵢ) == backward(Σ pᵢ). Each product is
    # returned truncated-in-coeff via `result_layout=:c` (skips its own backward).
    if n >= 2 && c1.dist.use_pencil_arrays && c1.dist.size > 1 &&
       all(b -> isa(b, Union{RealFourier, ComplexFourier}), c1.bases)
        result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator; result_layout=:c)
        ensure_layout!(result, :c)
        rc = get_coeff_data(result)
        for i in 2:n
            product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator; result_layout=:c)
            ensure_layout!(product, :c)
            pc = get_coeff_data(product)
            if isa(rc, PencilArrays.PencilArray) && isa(pc, PencilArrays.PencilArray)
                parent(rc) .+= parent(pc)
            else
                rc .+= pc
            end
        end
        return result  # coeff; caller's ensure_layout! does the single backward
    end

    # Serial / non-Fourier: sum products in grid (original path).
    result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator)
    for i in 2:n
        product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator)
        result = result + product
    end

    return result
end

"""Evaluate v1×v2 cross product (3D only)"""
function evaluate_vector_cross_product(v1::VectorField, v2::VectorField)

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
"""Evaluate DotProduct of two VectorFields"""
function evaluate(op::DotProduct, layout::Symbol=:g)
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

"""Evaluate CrossProduct of two VectorFields"""
function evaluate(op::CrossProduct, layout::Symbol=:g)
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
"""Create advection operator u·∇φ"""
function advection(u::VectorField, φ::ScalarField)
    return AdvectionOperator(u, φ)
end

"""Create nonlinear momentum operator (u·∇)u"""
function nonlinear_momentum(u::VectorField)
    return NonlinearAdvectionOperator(u)
end

"""Create convective operator"""
function convection(f1, f2, op::Symbol)
    return ConvectiveOperator(f1, f2, op)
end

"""Log nonlinear evaluation performance statistics"""
function log_nonlinear_performance(stats::NonlinearPerformanceStats)

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
