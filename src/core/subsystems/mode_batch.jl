# ── Batched Fourier-mode solve: structural bucketing ─────────────────────────
#
# For a 2-D mixed Fourier–Chebyshev problem every Fourier mode gets its own
# `Subproblem`, and the per-mode matrices are measurably identical in SHAPE and
# SPARSITY PATTERN, differing only in `nzval`. That makes the modes a perfect
# batch. But "measurably" is the operative word: a problem whose kx=0 mode
# carries a gauge constraint or a different BC would break the assumption
# silently. So the batchability of a set of subproblems is OBSERVED here from
# the matrices as actually built, never inferred from `nz`/`nvars` arithmetic.

"""
    batch_signature(sp::Subproblem) -> UInt64

Hash of everything that must match for two subproblems to share a batched
factorization and a batched gather/scatter: the LHS shape and pattern, the mass
matrix shape and pattern, the BC/bulk row and column partitions, and the
preconditioner patterns.

Deliberately EXCLUDES every `nzval`: differing values are the entire point of
batching. Returns `0x0` for a subproblem whose matrices were never built, which
callers must treat as "not batchable".
"""
function batch_signature(sp::Subproblem)
    (sp.M_min === nothing || sp.L_min === nothing || sp.LHS === nothing) && return 0x0
    (sp.M_exp === nothing || sp.L_exp === nothing) && return 0x0

    # `L_exp`, NOT `L_min`. Measured on the channel problem at nx=64/nz=32:
    # `L_min` stores 353 nonzeros at every mode except kx=0, which stores 321 —
    # the ∂xx term is the literal zero operator there, so those entries are
    # never created. Hashing `L_min` would exile kx=0 to its own bucket on every
    # problem containing a second derivative, i.e. nearly all of them.
    # `L_exp` is `expand_pattern(L_min, LHS)`: numerically identical to `L_min`
    # (verified exactly equal), carried in LHS's union pattern, and uniform
    # across all modes including kx=0. The batched L·X product must use `L_exp`
    # for the same reason (Task 6).
    h = hash(:tarang_mode_batch_v1)
    for A in (sp.LHS, sp.M_min, sp.L_exp)
        h = hash(size(A), h)
        h = hash(A.colptr, h)
        h = hash(A.rowval, h)
    end
    h = hash(sp.bc_rows, h)
    h = hash(sp.bulk_rows, h)
    h = hash(sp.bc_cols, h)
    h = hash(sp.bulk_cols, h)
    for P in (sp.pre_left, sp.pre_right, sp.pre_left_pinv, sp.pre_right_pinv)
        if P === nothing
            h = hash(nothing, h)
        else
            h = hash(size(P), h)
            h = hash(P.colptr, h)
            h = hash(P.rowval, h)
        end
    end
    # Never collide with the "not batchable" sentinel. `zero(UInt64)`/`one(UInt64)`
    # rather than `0x0`/`0x1`: the latter are UInt8 literals, which would infer
    # this function as `Union{UInt64, UInt8}` — a type instability in a package
    # whose JET ratchet has zero headroom.
    return h == zero(UInt64) ? one(UInt64) : h
end

"""
    bucket_subproblems(sps) -> Dict{UInt64, Vector{Int}}

Group subproblem INDICES by `batch_signature`. Subproblems with signature `0x0`
are omitted entirely — they have no built matrices and must stay on the per-mode
path. Index vectors come back in ascending order so batch column `m` maps to a
deterministic mode.
"""
function bucket_subproblems(sps)
    buckets = Dict{UInt64, Vector{Int}}()
    for (i, sp) in enumerate(sps)
        sig = batch_signature(sp)
        sig == 0x0 && continue
        push!(get!(buckets, sig, Int[]), i)
    end
    for v in values(buckets)
        sort!(v)
    end
    return buckets
end

"""
    ModeBatch

Every Fourier mode in one structural bucket, laid out so each batched operation
touches one array instead of `nmodes` of them. Column `m` is mode
`sp_indices[m]` throughout.

The sparsity pattern is stored ONCE (`colptr`/`rowval`); only values are
per-mode. `M_exp_nzval` and `L_exp_nzval` are resident for the batch's lifetime,
so `batched_assemble_lhs!` rebuilds every mode's LHS on-device from
`M_exp + dt*a_ii*L_exp` with no host work and no upload — which is also why the
old per-mode host `LHS.nzval` rebuild under adaptive dt disappears.

`factored_key` records the `(dt, a_ii)` a factorization is valid for, alongside
an explicit `dirty` bit. Both, not one: a bare flag plus a reallocated buffer is
how a stale factorization silently serves zeros.
"""
struct ModeBatch
    sp_indices::Vector{Int}
    n::Int
    nmodes::Int

    colptr::AbstractVector{Int}
    rowval::AbstractVector{Int}
    M_exp_nzval::AbstractMatrix{ComplexF64}
    L_exp_nzval::AbstractMatrix{ComplexF64}

    M_min_colptr::AbstractVector{Int}
    M_min_rowval::AbstractVector{Int}
    M_min_nzval::AbstractMatrix{ComplexF64}

    lhs_dense::AbstractArray{ComplexF64, 3}
    bc_rows::AbstractVector{Int}

    factored_key::Ref{Tuple{Float64, Float64}}
    dirty::Ref{Bool}
end

"""Bytes the dense LHS workspace will occupy for `nmodes` matrices of order `n`."""
mode_batch_bytes(n::Int, nmodes::Int) = n * n * nmodes * sizeof(ComplexF64)

# `like` selects the array backend: pass an existing device vector to get device
# storage, or a plain `ComplexF64[]` for host storage. Mirrors the `like=`
# convention already used by `_subproblem_cached_vector!`.
_batch_similar(like::AbstractVector, ::Type{T}, dims...) where {T} =
    similar(like, T, dims...)

"""
    build_mode_batch(sps, indices; like) -> ModeBatch

Pack the subproblems at `indices` into one batch. All of them must share a
`batch_signature`; the caller (`bucket_subproblems`) guarantees that.
"""
function build_mode_batch(sps, indices::Vector{Int}; like::AbstractVector)
    sp1 = sps[indices[1]]
    n = size(sp1.LHS, 1)
    nmodes = length(indices)

    nnz_exp = length(sp1.M_exp.nzval)
    nnz_m = length(sp1.M_min.nzval)

    M_exp = _batch_similar(like, ComplexF64, nnz_exp, nmodes)
    L_exp = _batch_similar(like, ComplexF64, nnz_exp, nmodes)
    M_min = _batch_similar(like, ComplexF64, nnz_m, nmodes)

    # Stage on the host, then upload each block once. Column m == mode
    # sp_indices[m], fixed for the batch's lifetime.
    host_M_exp = Matrix{ComplexF64}(undef, nnz_exp, nmodes)
    host_L_exp = Matrix{ComplexF64}(undef, nnz_exp, nmodes)
    host_M_min = Matrix{ComplexF64}(undef, nnz_m, nmodes)
    for (m, i) in enumerate(indices)
        sp = sps[i]
        @views host_M_exp[:, m] .= sp.M_exp.nzval
        @views host_L_exp[:, m] .= sp.L_exp.nzval
        @views host_M_min[:, m] .= sp.M_min.nzval
    end
    copyto!(M_exp, host_M_exp)
    copyto!(L_exp, host_L_exp)
    copyto!(M_min, host_M_min)

    int_like = _batch_similar(like, Int, 0)
    colptr = _batch_similar(int_like, Int, length(sp1.LHS.colptr))
    rowval = _batch_similar(int_like, Int, length(sp1.LHS.rowval))
    copyto!(colptr, sp1.LHS.colptr)
    copyto!(rowval, sp1.LHS.rowval)

    m_colptr = _batch_similar(int_like, Int, length(sp1.M_min.colptr))
    m_rowval = _batch_similar(int_like, Int, length(sp1.M_min.rowval))
    copyto!(m_colptr, sp1.M_min.colptr)
    copyto!(m_rowval, sp1.M_min.rowval)

    bc_rows = _batch_similar(int_like, Int, length(sp1.bc_rows))
    copyto!(bc_rows, sp1.bc_rows)

    lhs_dense = _batch_similar(like, ComplexF64, n, n, nmodes)

    return ModeBatch(copy(indices), n, nmodes,
                     colptr, rowval, M_exp, L_exp,
                     m_colptr, m_rowval, M_min,
                     lhs_dense, bc_rows,
                     Ref((NaN, NaN)), Ref(true))
end

"""
    csr_pattern(A::SparseMatrixCSC) -> (rowptr, colval, perm)

The CSR view of `A`'s sparsity pattern, plus the permutation that carries a
CSC-ordered `nzval` into CSR order.

`batched_spmv!` assigns one thread per (row, mode) and accumulates that row's
dot product in a register, which needs row-major access. A column-major kernel
would instead have to accumulate into `Y[row, m]` across iterations — the
same-slot read-modify-write shape the KA CPU backend miscompiles.

`perm` is shared by every mode in a batch, which is legal precisely because the
bucket signature guarantees an identical pattern: permuting mode `m`'s values is
`nzval[perm, m]` for every `m`.
"""
function csr_pattern(A::SparseMatrixCSC)
    n = size(A, 1)
    # Transposing a matrix whose values are 1:nnz yields, in CSC order of the
    # transpose (== CSR order of A), the original CSC index of each entry.
    tagged = SparseMatrixCSC(size(A, 1), size(A, 2), copy(A.colptr),
                             copy(A.rowval), collect(1:nnz(A)))
    tagged_t = sparse(transpose(tagged))
    return (copy(tagged_t.colptr), copy(tagged_t.rowval), copy(tagged_t.nzval))
end

"""
    should_batch_modes(base, sps, indices; is_gpu, nprocs) -> Bool

All four conditions from the design must hold:

1. `nprocs == 1` — distributed batching is out of scope, and the per-rank mode
   partitioning plus solve-layout bracket would need MPI verification.
2. `base.batched_modes` resolves true for this device — `nothing` means GPU yes,
   CPU no, so no existing CPU run changes behavior.
3. the bucket holds at least two modes — one mode has nothing to batch.
4. the dense workspace fits under `base.batched_modes_max_bytes`.

Condition 4 emits `@info maxlog=1` when it declines, because a silent
performance cliff at large `nz` is exactly what goes unnoticed for months.
Condition 3 declines silently: warning about it would be noise on every small
problem.
"""
function should_batch_modes(base, sps, indices::Vector{Int};
                            is_gpu::Bool, nprocs::Int)
    nprocs == 1 || return false

    setting = base.batched_modes
    enabled = setting === nothing ? is_gpu : setting
    enabled || return false

    length(indices) >= 2 || return false

    sp1 = sps[indices[1]]
    sp1.LHS === nothing && return false
    n = size(sp1.LHS, 1)
    bytes = mode_batch_bytes(n, length(indices))
    if bytes > base.batched_modes_max_bytes
        @info("Batched mode solve declined: dense workspace needs $bytes bytes " *
              "for $(length(indices)) modes of order $n, over the " *
              "$(base.batched_modes_max_bytes)-byte cap. Falling back to the " *
              "per-mode loop. Raise `batched_modes_max_bytes` to enable.",
              maxlog=1)
        return false
    end
    return true
end

"""
    build_mode_batches!(base, sps; is_gpu, nprocs, like) -> Vector{ModeBatch}

Bucket `sps` and build a `ModeBatch` for every bucket that passes
`should_batch_modes`. Buckets that decline are simply absent from the result,
and their subproblems stay on the per-mode path.
"""
function build_mode_batches!(base, sps; is_gpu::Bool, nprocs::Int,
                             like::AbstractVector)
    batches = ModeBatch[]
    for indices in values(bucket_subproblems(sps))
        should_batch_modes(base, sps, indices; is_gpu, nprocs) || continue
        push!(batches, build_mode_batch(sps, indices; like))
    end
    sort!(batches; by=b -> b.sp_indices[1])
    return batches
end
