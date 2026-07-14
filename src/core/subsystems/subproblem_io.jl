# Per-subproblem gather/scatter
# ---------------------------------------------------------------------------

"""Return the global 1-based Fourier mode index for this subproblem."""
function _kx_index_global(sp::Subproblem)
    for g in sp.group
        if g isa Integer
            return g + 1  # 0-based → 1-based
        end
    end
    return 1
end

"""
Convert a global 1-based Fourier mode index to a local index within the
coefficient data array. For serial runs, local == global. For MPI with
PencilArrays, the local buffer only holds this rank's modes.
"""
function _global_to_local_kx(kx_global::Int, field::ScalarField, sp::Subproblem)
    dist = sp.dist
    if dist === nothing || dist.size <= 1
        return kx_global
    end
    # Find the separable (Fourier) axis and its global size
    for (axis, g) in enumerate(sp.group)
        if g isa Integer && axis <= length(field.bases) && field.bases[axis] !== nothing
            basis = field.bases[axis]
            if isa(basis, FourierBasis)
                global_size = isa(basis, RealFourier) ? div(basis.meta.size, 2) + 1 : basis.meta.size
                local_range = local_indices(dist, axis, global_size)
                return kx_global - first(local_range) + 1
            end
        end
    end
    return kx_global
end

"""
Get the local coefficient data array. For PencilArrays (MPI), returns the
underlying local buffer via `parent()`. For regular arrays, returns as-is.
"""
_local_coeff_data(cd::AbstractArray) = get_local_data(cd)
_local_coeff_data(::Nothing) = nothing

# ── Distributed mixed Fourier–Chebyshev solve-layout transpose ───────────────
#
# In coeff space a mixed Fourier–Chebyshev field's PencilArray (the
# `pencil_fft_output` layout) is decomposed along the CHEBYSHEV axis with the
# Fourier axis LOCAL (and a memory permutation). The per-Fourier-mode tau solve,
# however, needs each rank to own a subset of Fourier modes plus FULL Chebyshev
# columns — i.e. the Fourier axis DECOMPOSED and the Chebyshev axis LOCAL. That
# is exactly the `dist.pencil_solve` layout. The gather/scatter index logic
# (_subproblem_coeff_index, _global_to_local_kx, …) already matches the solve
# layout (`local_indices` reports the x-mode split); the ONLY defect is data
# placement. So before gather we transpose the coeff data into the solve pencil,
# and after scatter we transpose it back into the FFT pencil.
#
# Collective-correctness: `to_solve_layout!`/`from_solve_layout!` iterate the
# GLOBAL deterministic field list with a rank-independent predicate, and
# `PencilArrays.transpose!` is ONE collective over the whole comm independent of
# how many subproblems each rank iterates. They MUST be called OUTSIDE any
# `for sp in subproblems` loop so every rank issues the same transpose count in
# the same order.

"""
    _needs_solve_transpose(field::ScalarField, dist) -> Bool

True when `field`'s coeff data must be transposed into the Chebyshev-local
solve pencil before a per-mode gather/scatter: a distributed (`size>1`) run with
a built `pencil_solve`, a non-Fourier (Chebyshev/Jacobi) basis present, and the
coeff storage actually a `PencilArray`.
"""
function _needs_solve_transpose(field::ScalarField, dist)
    return dist !== nothing && dist.size > 1 && dist.pencil_solve !== nothing &&
           !isempty(field.bases) &&
           any(b -> b !== nothing && !isa(b, FourierBasis), field.bases) &&
           get_coeff_data(field) isa PencilArrays.PencilArray
end

_needs_solve_transpose(::Any, ::Any) = false

"""
    _iter_scalar_fields(fields) -> Vector{ScalarField}

Flatten a state/RHS field list to its `ScalarField` leaves (Vector/Tensor field
components expanded), mirroring the `_gather_field_raw!(::VectorField, …)`
component iteration so the transpose set matches the gather set exactly.
"""
function _iter_scalar_fields(fields)
    out = ScalarField[]
    for f in fields
        if f isa ScalarField
            push!(out, f)
        elseif f isa VectorField
            for comp in f.components
                push!(out, comp)
            end
        elseif f isa TensorField
            for comp in vec(f.components)
                push!(out, comp)
            end
        end
    end
    return out
end

# Shared empty stash returned by `to_solve_layout!` on the no-op (serial /
# pure-Fourier / no-pencil_solve) path so the common case never allocates.
# `from_solve_layout!` only iterates a stash (read-only), so sharing one const
# across all no-op calls is safe. NEVER push! onto this.
const _EMPTY_SOLVE_STASH = Pair{ScalarField, Any}[]

# The distributed PencilFFT plan applies ONLY the Fourier transform (the
# Chebyshev/Jacobi axis is `NoTransform`, kept local), so `pencil_fft_output`
# holds Fourier-coeffs along the Fourier axis but GRID values along the
# Chebyshev axis. The per-mode matrices are built in Chebyshev-COEFF space, so
# once we are in the solve layout (Chebyshev axis LOCAL) we must apply the local
# Chebyshev (or Legendre/Jacobi) forward transform along that axis before the
# gather, and undo it after the scatter. `_apply_forward`/`_apply_backward` are
# the SAME per-axis dispatchers the serial transform chain uses (they read
# `transform.axis` and handle complex Fourier coefficients), so the coefficient
# convention matches the matrices exactly. The Fourier transform is skipped here
# (already done by the PencilFFT); non-`Transform` entries (the `PencilFFTPlan`
# itself) are skipped too.

"""
    _solve_layout_forward_transform!(solve_pa, dist)

Apply the local non-Fourier (Chebyshev/Legendre/Jacobi) forward transform along
each such axis of `solve_pa` (NoPermutation ⇒ `parent` is in logical order, with
the coupled axis local). Converts the coupled axis from grid to coefficient space
so the per-mode gather sees spectral coefficients.
"""
function _solve_layout_forward_transform!(solve_pa::PencilArrays.PencilArray, dist)
    cd = parent(solve_pa)
    for transform in dist.transforms
        (transform isa Transform) || continue          # skip PencilFFTPlan
        (transform isa FourierTransform) && continue    # Fourier done by PencilFFT
        # In-place via the transform's CACHED scratch+plan (zero-alloc once warm) —
        # same path the serial chain uses. The allocating `_apply_forward` re-planned
        # FFTW + allocated ~6 full-size arrays per call (≈3.5 MB/step under MPI). The
        # solve pencil is square along the coupled axis (grid==coeff), so out===in is
        # safe (the kernel reads `in` into scratch before writing `out`).
        _apply_forward!(cd, cd, transform)
    end
    return solve_pa
end

"""
    _solve_layout_backward_transform!(solve_pa, dist)

Inverse of `_solve_layout_forward_transform!`: apply the local non-Fourier
backward transform along each coupled axis, returning the coupled axis to grid
space before the transpose back to the FFT pencil.
"""
function _solve_layout_backward_transform!(solve_pa::PencilArrays.PencilArray, dist)
    cd = parent(solve_pa)
    for transform in dist.transforms
        (transform isa Transform) || continue
        (transform isa FourierTransform) && continue
        # In-place via cached scratch+plan (zero-alloc once warm). 3-arg form ⇒
        # out_n = transform.grid_size; square coupled axis ⇒ out===in is safe.
        _apply_backward!(cd, cd, transform)
    end
    return solve_pa
end

"""
    to_solve_layout!(fields, dist) -> Vector{Pair}

Transpose every eligible scalar field's coeff data from the FFT pencil
(`pencil_fft_output`) into the Chebyshev-local solve pencil (`dist.pencil_solve`)
and swap the field's coeff storage to the solve PencilArray. `:c` is already
Chebyshev-SPECTRAL along the coupled axis (`forward_transform!` applies the coupled
DCT via `_apply_distributed_coupled_dct!`), so the transpose alone lands the per-mode
tau solve in coefficient space. Returns a stash of `field => fft_pencil_array` pairs
to be undone by `from_solve_layout!`.

No-op (returns an empty stash) for serial runs or when `pencil_solve` is unset.

`fuse_from_grid=true` (state fields only) enables the fused grid→solve fast path:
when a field is in grid space (`:g`), the coupled (Chebyshev) DCT is applied IN the
solve pencil — where the single fft→solve transpose already lands the data — instead
of `forward_transform!` transposing fft→solve→fft (a round-trip whose solve→fft leg
this function would immediately undo with another fft→solve). This removes that pure
`solve→fft→solve` round-trip (2 collective transposes / field / stage). It is safe
ONLY for fields restored by `from_solve_layout!` (a real transpose back): the fused
path leaves the stashed fft pencil array in Chebyshev-GRID (not spectral), which
`from_solve_layout!`'s transpose overwrites, but a POINTER-SWAP restore (as used for
read-only F fields) would leave that field `:c`-flagged yet coupled-axis-grid → its
next `backward_transform!` would wrongly re-invert the DCT. So callers pass
`fuse_from_grid=true` for state only, and leave it `false` (default) for F.
"""
function to_solve_layout!(fields, dist; fuse_from_grid::Bool=false)
    # No-op (serial / pure-Fourier / no pencil_solve): return the SHARED empty
    # stash WITHOUT allocating. This guard runs ~8–14×/step even on serial runs;
    # allocating a fresh empty `Pair[]` here was pure waste. `from_solve_layout!`
    # only iterates the stash (never mutates), so a shared const is safe.
    (dist === nothing || dist.size <= 1 || dist.pencil_solve === nothing) && return _EMPTY_SOLVE_STASH
    stash = Pair{ScalarField, Any}[]
    cache = get_transpose_cache()
    # `collect_state_fields`/buffered-RHS lists are ALREADY flat `ScalarField`
    # vectors, so iterate them directly — `_iter_scalar_fields` would rebuild an
    # identical `ScalarField[]` every call. Only the Vector/Tensor-component case
    # needs the flatten. The list is iterated read-only here.
    leaves = eltype(fields) <: ScalarField ? fields : _iter_scalar_fields(fields)
    for f in leaves
        _needs_solve_transpose(f, dist) || continue
        key = (:solve, objectid(f), objectid(dist.pencil_solve))
        if fuse_from_grid && f.current_layout === :g
            # FUSED grid→solve: apply ONLY the Fourier transform here (skip the
            # coupled DCT's fft→solve→fft round-trip), transpose fft→solve ONCE,
            # then apply the local coupled DCT directly in the solve pencil — where
            # the tau solve needs it — via the same `_solve_layout_forward_transform!`
            # the solver uses. Net: one transpose instead of three for this field.
            forward_transform!(f, :c; apply_coupled_dct=false)   # Fourier only; coupled axis stays GRID
            fft_pa = get_coeff_data(f)
            (fft_pa isa PencilArrays.PencilArray) || continue
            dtype = eltype(fft_pa)
            solve_pa = get_transpose_buffer!(cache, dist.pencil_solve, dtype, key)
            PencilArrays.transpose!(solve_pa, fft_pa)
            _solve_layout_forward_transform!(solve_pa, dist)     # coupled DCT, local, in solve pencil
            set_coeff_data!(f, solve_pa)
            push!(stash, f => fft_pa)
        else
            ensure_layout!(f, :c)
            fft_pa = get_coeff_data(f)
            (fft_pa isa PencilArrays.PencilArray) || continue
            dtype = eltype(fft_pa)
            solve_pa = get_transpose_buffer!(cache, dist.pencil_solve, dtype, key)
            PencilArrays.transpose!(solve_pa, fft_pa)
            # `:c` is already Chebyshev-SPECTRAL (forward_transform! applied the coupled
            # DCT via _apply_distributed_coupled_dct!), so the fft→solve transpose alone
            # lands the per-mode tau solve in coefficient space. No DCT here (was double).
            set_coeff_data!(f, solve_pa)
            push!(stash, f => fft_pa)
        end
    end
    return stash
end

"""
    from_solve_layout!(stash, dist)

Undo `to_solve_layout!`: transpose each field's solve PencilArray back into its FFT
PencilArray and restore the field's coeff storage. The coupled axis stays
Chebyshev-SPECTRAL (the DCT is now owned by `forward_transform!`/`backward_transform!`),
so after this the field is back in the `pencil_fft_output` layout holding full spectral
coefficients — exactly what `backward_transform!` inverts (coupled inverse-DCT, then the
PencilFFT Fourier `ldiv!`).
"""
function from_solve_layout!(stash, dist; to_grid::Bool=false)
    (dist === nothing || dist.size <= 1) && return
    for (f, fft_pa) in stash
        solve_pa = get_coeff_data(f)
        (solve_pa isa PencilArrays.PencilArray) || continue
        if to_grid
            # FUSED solve→grid: the field is ALREADY in the solve pencil, so invert the
            # coupled (Chebyshev) DCT HERE — locally, no transpose — instead of leaving
            # `:c` and letting the next `backward_transform!` transpose fft→solve to do
            # it (which, paired with this function's solve→fft, is a redundant round-trip).
            # Then transpose solve→fft ONCE and finish with the Fourier `ldiv!` only. Net
            # for this field: one transpose instead of three. The transient
            # "Fourier-spectral, coupled-axis GRID, :c-flagged" state exists ONLY between
            # the transpose and `backward_transform!` below — entirely inside this
            # function — so it never escapes and needs no persistent hybrid-state flag.
            # Symmetric with `to_solve_layout!`'s fused grid→solve (`fuse_from_grid`).
            # Safe ONLY for fields the caller next consumes at `:g` with no intervening
            # coeff read (the per-stage state pop before `evaluate_rhs_buffered`).
            # PRECONDITION: the field must be `:c`-flagged (Chebyshev-SPECTRAL in the
            # solve pencil) on entry — the sole caller guarantees this. If a future
            # caller passed `to_grid=true` for a `:g`-flagged field, the transpose below
            # would store coupled-axis-GRID data that `backward_transform!`'s leading
            # `ensure_layout!(:c)` would then re-derive from the STALE grid, silently
            # discarding the solve result. Refuse loudly (rank-uniform: layout is
            # replicated, so this errors on all ranks together or none).
            f.current_layout === :c || error("from_solve_layout!(to_grid=true) requires " *
                ":c-flagged fields; got :$(f.current_layout) for '$(f.name)'")
            _solve_layout_backward_transform!(solve_pa, dist)     # inverse coupled DCT, local, in solve pencil
            PencilArrays.transpose!(fft_pa, solve_pa)             # solve→fft (coupled axis now GRID)
            set_coeff_data!(f, fft_pa)
            backward_transform!(f, :g; apply_coupled_dct=false)   # Fourier ldiv! only → :g
        else
            # The coupled (Chebyshev) DCT is applied/inverted by
            # forward_transform!/backward_transform! (see _apply_distributed_coupled_dct!),
            # so `:c` is Chebyshev-SPECTRAL in both the fft and solve pencils. The
            # solve↔fft transpose preserves those coefficients; no DCT here.
            PencilArrays.transpose!(fft_pa, solve_pa)
            set_coeff_data!(f, fft_pa)
        end
    end
    return
end

"""
    _apply_distributed_coupled_dct!(field::ScalarField, forward::Bool) -> field

Apply (forward=true) or invert (forward=false) the local non-Fourier
(Chebyshev/Legendre/Jacobi) transform along each coupled axis of a DISTRIBUTED
mixed Fourier–Chebyshev field's coeff data, so that `:c` holds true spectral
coefficients along the coupled axis rather than grid values.

The distributed PencilFFT plan transforms ONLY the Fourier axes (the coupled axis is
`NoTransform`, and is the DECOMPOSED axis of `pencil_fft_output`), so after `mul!`
the coupled axis is still in GRID space — the long-standing "distributed mixed `:c`
returns un-DCT'd data" bug. The coupled DCT needs that axis FULLY LOCAL, which is the
`pencil_solve` layout, so we transpose `fft → solve`, apply the local forward/backward
coupled transform (the SAME `_solve_layout_*_transform!` the solver uses), and transpose
back — leaving the coeff data in the SAME `pencil_fft_output` layout but Chebyshev-spectral.

No-op for serial runs, pure-Fourier fields, GPU coeff arrays, or when `pencil_solve` is
unset. `forward_transform!`/`backward_transform!` run symmetrically on every rank, so the
two `PencilArrays.transpose!` collectives here are always rank-uniform.
"""
function _apply_distributed_coupled_dct!(field::ScalarField, forward::Bool)
    dist = field.dist
    (dist !== nothing && dist.size > 1 && dist.pencil_solve !== nothing) || return field
    isempty(field.bases) && return field
    any(b -> b !== nothing && !isa(b, FourierBasis), field.bases) || return field
    fft_pa = get_coeff_data(field)
    (fft_pa isa PencilArrays.PencilArray) || return field
    _count_transform!(:coupled_dct)   # 2 collective transposes below

    cache = get_transpose_cache()
    dtype = eltype(fft_pa)
    # Per-pencil key (drops objectid(field) but KEEPS objectid(dist.pencil_solve)):
    # the coupled-DCT solve_pa is pure transient scratch — fully overwritten by the
    # fft→solve transpose on entry and never stored back into the field — so all fields
    # sharing this pencil_solve safely share ONE bounded buffer (only one field is live
    # at a time within a single forward/backward transform). The buffer is ALLOCATED from
    # dist.pencil_solve, so the key must carry that pencil's identity — a shape-only key
    # would alias a DIFFERENT problem's pencil_solve of equal global size but incompatible
    # topology/permutation (→ "pencil topologies must be the same"). objectid(pencil_solve)
    # is stable per problem (only field objectids churn on the interpreted-RHS path), so
    # this still bounds the cache to one buffer per problem. `dtype` is appended by
    # `get_transpose_buffer!`.
    key = (:coupled_dct, objectid(dist.pencil_solve))
    solve_pa = get_transpose_buffer!(cache, dist.pencil_solve, dtype, key)
    PencilArrays.transpose!(solve_pa, fft_pa)
    forward ? _solve_layout_forward_transform!(solve_pa, dist) :
              _solve_layout_backward_transform!(solve_pa, dist)
    PencilArrays.transpose!(fft_pa, solve_pa)
    return field
end

@inline function _subproblem_backend_field(which::Symbol)
    if which === :M
        return :M_backend
    elseif which === :L
        return :L_backend
    elseif which === :pre_right_pinv
        return :pre_right_pinv_backend
    elseif which === :pre_right
        return :pre_right_backend
    elseif which === :pre_left
        return :pre_left_backend
    end
    throw(ArgumentError("Unknown backend cache key: $which"))
end

@inline function _subproblem_selection_field(which::Symbol)
    if which === :pre_right_pinv
        return :pre_right_pinv_indices
    elseif which === :pre_left
        return :pre_left_indices
    end
    throw(ArgumentError("Unknown selection-index cache key: $which"))
end

@inline function _subproblem_vector_field(which::Symbol)
    if which === :compress_variable_space
        return :compress_variable_space
    elseif which === :expand_variable_space
        return :expand_variable_space
    elseif which === :compress_equation_space
        return :compress_equation_space
    elseif which === :gather_inputs_raw
        return :gather_inputs_raw
    elseif which === :scatter_inputs_expanded
        return :scatter_inputs_expanded
    elseif which === :gather_outputs_raw
        return :gather_outputs_raw
    elseif which === :gather_eqn_F_raw
        return :gather_eqn_F_raw
    elseif which === :gather_alg_F_raw
        return :gather_alg_F_raw
    end
    throw(ArgumentError("Unknown vector cache key: $which"))
end

function _subproblem_backend_matrix!(sp::Subproblem, matrix, which::Symbol, data::AbstractVector)
    matrix === nothing && return nothing
    if !is_gpu_array(data)
        return matrix
    end
    field = _subproblem_backend_field(which)
    cached = getfield(sp.runtime, field)
    cached !== nothing && return cached
    backend = _gpu_sparse_csr(matrix, eltype(matrix))
    setfield!(sp.runtime, field, backend)
    return backend
end

"""
    _bc_rows_device(sp, reference)

Return `sp.bc_rows` as an index vector on the same device as `reference`.

For CPU `reference`, returns the CPU `Vector{Int}` as-is. For a GPU
`reference`, returns (and caches) a device-side `Int` array so that vectorized
`rhs[bc_rows] .= coeff .* alg_f[bc_rows]` stays on-device and does not trigger
scalar indexing when `CUDA.allowscalar(false)` is set.
"""
function _bc_rows_device(sp::Subproblem, reference::AbstractVector)
    if isempty(sp.bc_rows) || !is_gpu_array(reference)
        return sp.bc_rows
    end
    cached = sp.runtime.bc_rows_device
    if cached !== nothing && length(cached) == length(sp.bc_rows)
        return cached
    end
    device_rows = similar(reference, Int, length(sp.bc_rows))
    copyto!(device_rows, sp.bc_rows)
    sp.runtime.bc_rows_device = device_rows
    return device_rows
end

"""
    apply_bc_override!(rhs, alg_f, sp, coeff)

Override the algebraic-constraint rows of `rhs` with `coeff * alg_f[bc_rows]`.

Pipeline:

    tmp .= alg_f[bc_rows]     # gather into cached scratch buffer
    tmp .*= coeff             # in-place scale
    rhs[bc_rows] = tmp        # scatter

All three operations are vectorized and stay on the same device as `rhs`, so
this is safe under `CUDA.allowscalar(false)`. The `bc_rows` index vector is
materialized on the correct device by `_bc_rows_device` and cached on the
subproblem to avoid per-step host→device transfers. The gather target
`tmp` is also a cached scratch buffer (`_bc_override_scratch!`), so this
function performs zero per-call allocation after the first call.
"""
function apply_bc_override!(rhs::AbstractVector, alg_f::AbstractVector,
                            sp::Subproblem, coeff)
    isempty(sp.bc_rows) && return rhs
    c = ComplexF64(coeff)
    if is_gpu_array(rhs)
        # GPU path — vectorized scatter via on-device index array. The
        # `alg_f[bc_rows]` gather allocates a small CuArray (length =
        # length(sp.bc_rows), typically ≤ 16), but the allocation is cheap
        # on modern device runtimes and cannot be avoided without a custom
        # kernel. The cached `_bc_rows_device` avoids per-call H2D.
        bc_rows = _bc_rows_device(sp, rhs)
        rhs[bc_rows] = c .* alg_f[bc_rows]
    else
        # CPU path — scalar loop, zero allocation. For small bc_rows counts
        # (typical: 5–20) this is faster than broadcast-gather anyway.
        bc_rows = sp.bc_rows  # CPU Vector{Int}
        @inbounds @simd for i in eachindex(bc_rows)
            r = bc_rows[i]
            rhs[r] = c * alg_f[r]
        end
    end
    return rhs
end

function _subproblem_selection_indices!(sp::Subproblem, matrix::Union{Nothing, SparseMatrixCSC},
                                        which::Symbol)
    matrix === nothing && return nothing
    field = _subproblem_selection_field(which)
    cached = getfield(sp.runtime, field)
    cached !== nothing && return cached

    m, n = size(matrix)
    if nnz(matrix) == 0
        indices = Int[]
        setfield!(sp.runtime, field, indices)
        return indices
    end

    rows, cols, vals = findnz(matrix)
    length(rows) == m || return nothing

    indices = Vector{Int}(undef, m)
    seen = falses(m)
    oneval = one(eltype(matrix))
    @inbounds for k in eachindex(rows)
        r = rows[k]
        c = cols[k]
        if r < 1 || r > m || c < 1 || c > n || seen[r] || vals[k] != oneval
            return nothing
        end
        seen[r] = true
        indices[r] = c
    end
    all(seen) || return nothing

    setfield!(sp.runtime, field, indices)
    return indices
end

function _assign_to_buffer!(dest::AbstractVector{ComplexF64}, src)
    if is_gpu_array(dest)
        src_dev = is_gpu_array(src) ? src : on_architecture(architecture(dest), Array(src))
        if eltype(src_dev) <: Complex
            dest .= src_dev
        else
            dest .= ComplexF64.(src_dev)
        end
    else
        src_cpu = is_gpu_array(src) ? Array(src) : src
        if eltype(src_cpu) <: Complex
            copyto!(dest, src_cpu)
        else
            copyto!(dest, ComplexF64.(src_cpu))
        end
    end
    return dest
end

function _assign_from_buffer!(dest, src)
    if is_gpu_array(dest)
        src_dev = is_gpu_array(src) ? src : on_architecture(architecture(dest), Array(src))
        if eltype(dest) <: Real
            dest .= real.(src_dev)
        else
            dest .= src_dev
        end
    else
        src_cpu = is_gpu_array(src) ? Array(src) : src
        if eltype(dest) <: Real
            dest .= real.(src_cpu)
        else
            dest .= src_cpu
        end
    end
    return dest
end

function _subproblem_cached_vector!(sp::Subproblem, which::Symbol, n::Int;
                                    like::Union{Nothing, AbstractVector}=nothing)
    field = _subproblem_vector_field(which)
    cached = getfield(sp.runtime, field)
    if cached !== nothing && length(cached) == n
        if like === nothing || is_gpu_array(cached) == is_gpu_array(like)
            return cached
        end
    end

    buffer = if like === nothing
        zeros(sp.dist.architecture, ComplexF64, n)
    else
        similar_zeros(like, ComplexF64, n)
    end
    setfield!(sp.runtime, field, buffer)
    return buffer
end

"""Extract per-mode coefficients from all fields into a flat vector (no permutation)."""
function _gather_subproblem_raw!(buffer::AbstractVector{ComplexF64}, sp::Subproblem, fields::Vector)
    kx_global = _kx_index_global(sp)
    offset = 0
    for field in fields
        offset = _gather_field_raw!(buffer, offset, field, kx_global, sp)
    end
    return buffer
end

function _gather_subproblem_raw(sp::Subproblem, fields::Vector)
    total = sum(subproblem_field_size(sp, field) for field in fields)
    buffer = zeros(sp.dist.architecture, ComplexF64, total)
    return _gather_subproblem_raw!(buffer, sp, fields)
end

"""
Index tuple selecting this subproblem's mode from a multi-Fourier coefficient
array `cd`: each Fourier axis → its local mode index (an `Int`, which drops that
dimension); each coupled (non-Fourier, e.g. Chebyshev) axis → `Colon()` (kept in
full). Returns `nothing` when a selected Fourier mode is not on this rank (MPI).

Used for ≥3-D coefficient arrays — i.e. ≥2 Fourier axes plus a coupled axis (e.g.
a 3D x,y-Fourier + z-Chebyshev field). The 0-D / 1-D / 2-D paths are handled
inline; before this, the multi-D branch assumed a single Fourier axis (dim 1) and
treated dim 2 as the coupled axis, so for 3D it selected only kx, never ky, and
mis-sized the slice (BoundsError) — 3D mixed-basis solves never worked.
"""
function _subproblem_coeff_index(cd::AbstractArray, field::ScalarField, sp::Subproblem)
    # Memo: the index depends only on field.bases + sp.group + this rank's local mode
    # range + the coeff layout (size(cd)) — all step-invariant. Key by field identity
    # AND layout so a relayout (different cd) can't return a stale index. Caches the
    # off-rank `nothing` too, and avoids the per-call Vector{Any} allocation on hits.
    cache = sp.runtime.coeff_index_cache
    key = hash(size(cd), objectid(field))
    hit = get(cache, key, missing)
    hit === missing || return hit

    dist = sp.dist
    nd = ndims(cd)
    idx = Vector{Any}(undef, nd)
    valid = true
    for axis in 1:nd
        basis = axis <= length(field.bases) ? field.bases[axis] : nothing
        g = axis <= length(sp.group) ? sp.group[axis] : nothing
        if basis !== nothing && isa(basis, FourierBasis) && g isa Integer
            kx_global = g + 1   # 0-based mode → 1-based index
            kx_local = if dist === nothing || dist.size <= 1
                kx_global
            else
                global_size = isa(basis, RealFourier) ? div(basis.meta.size, 2) + 1 : basis.meta.size
                lr = local_indices(dist, axis, global_size)
                kx_global - first(lr) + 1
            end
            if kx_local < 1 || kx_local > size(cd, axis)
                valid = false; break    # selected mode not on this rank (MPI)
            end
            idx[axis] = kx_local
        else
            idx[axis] = Colon()   # coupled axis (Chebyshev/Jacobi): keep all modes
        end
    end
    result = valid ? Tuple(idx) : nothing
    length(cache) >= _SP_MEMO_CAP && empty!(cache)
    cache[key] = result
    return result
end

function _gather_field_raw!(buffer::AbstractVector{ComplexF64}, offset::Int, field::ScalarField, kx_global::Int, sp::Subproblem)
    ensure_layout!(field, :c)
    cd_raw = get_coeff_data(field)
    if cd_raw === nothing || isempty(cd_raw)
        n = subproblem_field_size(sp, field)
        # Vectorized zero-fill: works on CPU and GPU without scalar indexing.
        if n > 0
            fill!(view(buffer, offset + 1 : offset + n), ComplexF64(0))
        end
        return offset + n
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        dest = view(buffer, offset + 1:offset + 1)
        _assign_to_buffer!(dest, view(cd, 1:1))
        return offset + 1
    elseif ndims(cd) == 1
        if any(b -> b !== nothing && !isa(b, FourierBasis), field.bases)
            # Pure 1D coupled field (e.g. a single Chebyshev/Jacobi axis, no
            # separable Fourier direction): there is no Fourier mode to select,
            # so gather the ENTIRE coefficient spectrum for this subproblem.
            n = length(cd)
            dest = view(buffer, offset + 1:offset + n)
            _assign_to_buffer!(dest, cd)
            return offset + n
        end
        # 1D Fourier field (tau on a Fourier basis): select the single kx mode.
        kx_local = _global_to_local_kx(kx_global, field, sp)
        dest = view(buffer, offset + 1:offset + 1)
        if kx_local >= 1 && kx_local <= length(cd)
            _assign_to_buffer!(dest, view(cd, kx_local:kx_local))
        else
            fill!(dest, ComplexF64(0))
        end
        return offset + 1
    else
        # Multi-D field: select each Fourier axis's local mode (drops that axis)
        # and keep every coupled (non-Fourier) axis in full. Handles 1 Fourier + 1
        # Chebyshev (the old 2-D case: idx = (kx, :) ≡ selectdim(cd,1,kx)), 2
        # Fourier axes (a 0-D-in-z tau on (x,y): idx = (kx, ky) → 1 DOF), and 3D
        # x,y-Fourier + z-Chebyshev (idx = (kx, ky, :)) uniformly. `n` comes from
        # subproblem_field_size, which already counts 1 DOF per Fourier axis and
        # the full coeff size per coupled axis.
        n = subproblem_field_size(sp, field)
        idxt = _subproblem_coeff_index(cd, field, sp)
        if idxt === nothing
            fill!(view(buffer, offset + 1:offset + n), ComplexF64(0))   # mode not on this rank (MPI)
        elseif ndims(cd) == 2 && !is_gpu_array(cd) &&
               ((idxt[1] isa Colon) ⊻ (idxt[2] isa Colon))
            # 2-D mixed (1 Fourier mode × 1 full coupled axis): strided copy avoids the
            # vec(view(cd, idxt...)) wrapper alloc (~96 B/call). GPU/other shapes → below.
            _gather_select_2d!(buffer, offset, cd, idxt)
        else
            _assign_to_buffer!(view(buffer, offset + 1:offset + n), vec(view(cd, idxt...)))
        end
        return offset + n
    end
end

function _gather_field_raw!(buffer::AbstractVector{ComplexF64}, offset::Int, field::VectorField, kx_global::Int, sp::Subproblem)
    for comp in field.components
        offset = _gather_field_raw!(buffer, offset, comp, kx_global, sp)
    end
    return offset
end

"""Write per-mode coefficients back to fields from a flat vector (no permutation)."""
function _scatter_subproblem_raw(sp::Subproblem, data::AbstractVector, fields::Vector)
    kx_global = _kx_index_global(sp)
    offset = 0
    for field in fields
        offset = _scatter_field_raw!(field, data, offset, kx_global, sp)
    end
end

function _scatter_field_raw!(field::ScalarField, data::AbstractVector, offset::Int, kx_global::Int, sp::Subproblem)
    ensure_layout!(field, :c)
    cd_raw = get_coeff_data(field)
    if cd_raw === nothing || isempty(cd_raw)
        return offset + subproblem_field_size(sp, field)
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        _assign_from_buffer!(view(cd, 1:1), view(data, offset + 1:offset + 1))
        return offset + 1
    elseif ndims(cd) == 1
        if any(b -> b !== nothing && !isa(b, FourierBasis), field.bases)
            # Pure 1D coupled field: write back the ENTIRE coefficient spectrum
            # (mirror of the gather path above).
            n = length(cd)
            _assign_from_buffer!(view(cd, 1:n), view(data, offset + 1:offset + n))
            return offset + n
        end
        kx_local = _global_to_local_kx(kx_global, field, sp)
        if kx_local >= 1 && kx_local <= length(cd)
            _assign_from_buffer!(view(cd, kx_local:kx_local), view(data, offset + 1:offset + 1))
        end
        return offset + 1
    else
        # Mirror of the gather selector: write the solution back into the selected
        # (kx[, ky, …]) mode across all coupled-axis coefficients. Covers 1F+1Cheb,
        # 2 Fourier (0-D-in-z tau), and 3D x,y-Fourier + z-Cheb uniformly.
        n = subproblem_field_size(sp, field)
        idxt = _subproblem_coeff_index(cd, field, sp)
        if idxt !== nothing
            if ndims(cd) == 2 && !is_gpu_array(cd) && !is_gpu_array(data) &&
               ((idxt[1] isa Colon) ⊻ (idxt[2] isa Colon))
                # 2-D mixed (1 Fourier mode × 1 full coupled axis): strided write avoids
                # the view(cd, idxt...) wrapper alloc (~96 B/call). GPU/other shapes → below.
                _scatter_select_2d!(cd, idxt, data, offset)
            else
                _assign_from_buffer!(view(cd, idxt...), view(data, offset + 1:offset + n))
            end
        end
        return offset + n
    end
end

function _scatter_field_raw!(field::VectorField, data::AbstractVector, offset::Int, kx_global::Int, sp::Subproblem)
    for comp in field.components
        offset = _scatter_field_raw!(comp, data, offset, kx_global, sp)
    end
    return offset
end

# Concrete-typed gather/scatter inner loops. The cached subproblem buffers are stored
# as abstract `AbstractVector{ComplexF64}` (so a CuArray fits the same slot), so an
# inline index-copy loop over them dynamic-dispatches getindex/setindex! EVERY element
# → boxes a ComplexF64 + the dispatch tuple per element (~3 MB/step under MPI, profiled
# via Profile.Allocs). Narrow to the concrete CPU `Vector{ComplexF64}` once via a union
# split so the hot branch compiles statically (zero-alloc); any other array type keeps
# the generic — still correct — branch.
@inline function _select_copy!(dest::AbstractVector, raw::AbstractVector, indices)
    if dest isa Vector{ComplexF64} && raw isa Vector{ComplexF64}
        @inbounds for i in eachindex(indices)
            dest[i] = raw[indices[i]]
        end
    else
        @inbounds for i in eachindex(indices)
            dest[i] = raw[indices[i]]
        end
    end
    return dest
end

@inline function _scatter_copy!(dest::AbstractVector, raw::AbstractVector, indices)
    if dest isa Vector{ComplexF64} && raw isa Vector{ComplexF64}
        @inbounds for i in eachindex(indices)
            dest[indices[i]] = raw[i]
        end
    else
        @inbounds for i in eachindex(indices)
            dest[indices[i]] = raw[i]
        end
    end
    return dest
end

# Strided gather/scatter for the 2-D mixed subproblem layout: exactly 1 Fourier mode
# (an Int in `idxt`) × 1 full coupled/Chebyshev axis (a Colon in `idxt`). The previous
# `vec(view(cd, idxt...))` / `view(cd, idxt...)` wrapper alloc'd ~96 B/call regardless
# of typing; an explicit strided loop into/out of the flat buffer is zero-alloc. Mirrors
# `_select_copy!` (above): a union split narrows to the concrete CPU `Matrix{ComplexF64}`
# /`Vector{ComplexF64}` for a static hot loop, with a generic (other CPU eltype) fallback.
# CPU-only — callers guard with `!is_gpu_array(...)` and route GPU/other shapes to the
# unchanged `_assign_to_buffer!`/`_assign_from_buffer!` generic path.
@inline function _gather_select_2d!(buffer::AbstractVector{ComplexF64}, offset::Int,
                                    cd::AbstractMatrix, idxt::Tuple)
    if idxt[1] isa Colon
        col = idxt[2]::Int
        n = size(cd, 1)
        if buffer isa Vector{ComplexF64} && cd isa Matrix{ComplexF64}
            @inbounds for i in 1:n
                buffer[offset + i] = cd[i, col]
            end
        else
            @inbounds for i in 1:n
                buffer[offset + i] = ComplexF64(cd[i, col])
            end
        end
    else
        row = idxt[1]::Int
        n = size(cd, 2)
        if buffer isa Vector{ComplexF64} && cd isa Matrix{ComplexF64}
            @inbounds for j in 1:n
                buffer[offset + j] = cd[row, j]
            end
        else
            @inbounds for j in 1:n
                buffer[offset + j] = ComplexF64(cd[row, j])
            end
        end
    end
    return buffer
end

@inline function _scatter_select_2d!(cd::AbstractMatrix, idxt::Tuple,
                                     data::AbstractVector, offset::Int)
    if idxt[1] isa Colon
        col = idxt[2]::Int
        n = size(cd, 1)
        if cd isa Matrix{ComplexF64} && data isa Vector{ComplexF64}
            @inbounds for i in 1:n
                cd[i, col] = data[offset + i]
            end
        elseif eltype(cd) <: Real
            @inbounds for i in 1:n
                cd[i, col] = real(data[offset + i])
            end
        else
            @inbounds for i in 1:n
                cd[i, col] = data[offset + i]
            end
        end
    else
        row = idxt[1]::Int
        n = size(cd, 2)
        if cd isa Matrix{ComplexF64} && data isa Vector{ComplexF64}
            @inbounds for j in 1:n
                cd[row, j] = data[offset + j]
            end
        elseif eltype(cd) <: Real
            @inbounds for j in 1:n
                cd[row, j] = real(data[offset + j])
            end
        else
            @inbounds for j in 1:n
                cd[row, j] = data[offset + j]
            end
        end
    end
    return cd
end

function compress_variable_space!(dest::AbstractVector, sp::Subproblem, raw::AbstractVector)
    if sp.pre_right_pinv !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(raw)) ?
                  _subproblem_selection_indices!(sp, sp.pre_right_pinv, :pre_right_pinv) :
                  nothing
        if indices !== nothing
            _select_copy!(dest, raw, indices)
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_right_pinv, :pre_right_pinv, raw)
            if !is_gpu_array(dest) && !is_gpu_array(raw) && pre isa AbstractMatrix
                mul!(dest, pre, raw)
            else
                _assign_to_buffer!(dest, pre * raw)
            end
        end
    else
        _assign_to_buffer!(dest, raw)
    end
    return dest
end

function compress_variable_space(sp::Subproblem, raw::AbstractVector)
    n = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 1) : length(raw)
    dest = _subproblem_cached_vector!(sp, :compress_variable_space, n; like=raw)
    return compress_variable_space!(dest, sp, raw)
end

function expand_variable_space!(dest::AbstractVector, sp::Subproblem, data::AbstractVector)
    if sp.pre_right !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(data)) ?
                  _subproblem_selection_indices!(sp, sp.pre_right_pinv, :pre_right_pinv) :
                  nothing
        if indices !== nothing
            fill!(dest, zero(eltype(dest)))
            _scatter_copy!(dest, data, indices)
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_right, :pre_right, data)
            if !is_gpu_array(dest) && !is_gpu_array(data) && pre isa AbstractMatrix
                mul!(dest, pre, data)
            else
                _assign_to_buffer!(dest, pre * data)
            end
        end
    else
        _assign_to_buffer!(dest, data)
    end
    return dest
end

function expand_variable_space(sp::Subproblem, data::AbstractVector)
    n = sp.pre_right !== nothing ? size(sp.pre_right, 1) : length(data)
    dest = _subproblem_cached_vector!(sp, :expand_variable_space, n; like=data)
    return expand_variable_space!(dest, sp, data)
end

function compress_equation_space!(dest::AbstractVector, sp::Subproblem, raw::AbstractVector)
    if sp.pre_left !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(raw)) ?
                  _subproblem_selection_indices!(sp, sp.pre_left, :pre_left) :
                  nothing
        if indices !== nothing
            _select_copy!(dest, raw, indices)
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_left, :pre_left, raw)
            if !is_gpu_array(dest) && !is_gpu_array(raw) && pre isa AbstractMatrix
                mul!(dest, pre, raw)
            else
                _assign_to_buffer!(dest, pre * raw)
            end
        end
    else
        _assign_to_buffer!(dest, raw)
    end
    return dest
end

function compress_equation_space(sp::Subproblem, raw::AbstractVector)
    n = sp.pre_left !== nothing ? size(sp.pre_left, 1) : length(raw)
    dest = _subproblem_cached_vector!(sp, :compress_equation_space, n; like=raw)
    return compress_equation_space!(dest, sp, raw)
end

"""Gather per-mode coefficients and compress them into variable space."""
function gather_inputs(sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_inputs_raw, raw_len)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space(sp, raw)
end

function gather_inputs!(dest::AbstractVector, sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_inputs_raw, raw_len; like=dest)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space!(dest, sp, raw)
end

"""Expand from variable space and scatter back to fields."""
function scatter_inputs(sp::Subproblem, data::AbstractVector, fields::Vector)
    expanded_len = sp.pre_right !== nothing ? size(sp.pre_right, 1) : length(data)
    expanded = _subproblem_cached_vector!(sp, :scatter_inputs_expanded, expanded_len; like=data)
    expand_variable_space!(expanded, sp, data)
    _scatter_subproblem_raw(sp, expanded, fields)
end

"""
Gather per-mode RHS coefficients in state/variable ordering.

`evaluate_rhs` returns one field per state variable, so the explicit RHS must be
compressed with the variable-side preconditioner, not the equation-side one.
"""
function gather_outputs(sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_outputs_raw, raw_len)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space(sp, raw)
end

function gather_outputs!(dest::AbstractVector, sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_outputs_raw, raw_len; like=dest)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space!(dest, sp, raw)
end
