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
    return h == 0x0 ? 0x1 : h   # never collide with the "not batchable" sentinel
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
