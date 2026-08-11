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
    BatchedSparseOp

One sparse operator shared by every mode in a batch: a single CSR pattern
(`rowptr`/`colval`) plus one column of values per mode. Built for the operators
the batched stage loop applies with `batched_spmv!` — which iterates ROWS, so
the pattern must be CSR and nothing else (see `ModeBatch` below).

`nrows`/`ncols` are the operator's shape, kept so callers can size the `(nrows,
nmodes)` output and check the `(ncols, nmodes)` input without touching the
device arrays.
"""
struct BatchedSparseOp
    rowptr::AbstractVector{Int}
    colval::AbstractVector{Int}
    nzval::AbstractMatrix{ComplexF64}
    nrows::Int
    ncols::Int
end

"""
    ModeBatch

Every Fourier mode in one structural bucket, laid out so each batched operation
touches one array instead of `nmodes` of them. Column `m` is mode
`sp_indices[m]` throughout.

The sparsity pattern is stored ONCE; only values are per-mode. `M_exp_nzval` and
`L_exp_nzval` are resident for the batch's lifetime, so `batched_assemble_lhs!`
rebuilds every mode's LHS on-device from `M_exp + dt*a_ii*L_exp` with no host
work and no upload — which is also why the old per-mode host `LHS.nzval` rebuild
under adaptive dt disappears.

### Two pattern encodings, deliberately not interchangeable

`lhs_colptr`/`lhs_rowval` are the **CSC** pattern of `LHS`, and they exist for
exactly one consumer: `batched_assemble_lhs!`, which walks columns to place
stored values into a dense buffer. They are named for that consumer so they are
never mistaken for a matvec pattern.

Every operator that gets multiplied by a vector — `M_min`, `L_exp`, and the
three preconditioner projections — is stored in **CSR**, because
`batched_spmv!` assigns one thread per (row, mode) and iterates that row.
Handing it a CSC pattern computes `transpose(A)*x` silently, and none of these
matrices is symmetric. Task 2 shipped `M_min_colptr`/`M_min_rowval` (CSC) with
no caller; they are gone rather than merely deprecated, so the mistake is
unrepresentable. `M_min_nzval` and `L_nzval` are therefore in CSR order too (the
`perm` from `csr_pattern` applied to each mode's `nzval`), NOT in the CSC order
the source `SparseMatrixCSC` stores.

`L_rowptr`/`L_colval`/`L_nzval` come from **`L_exp`, never `L_min`** — `L_min`'s
pattern is not uniform across modes (kx=0 stores fewer nonzeros because ∂xx
vanishes there), while `L_exp` holds the same values in `LHS`'s union pattern.
Same reason `batch_signature` hashes `L_exp`.

### Preconditioner projections

`compress_var` (`pre_right_pinv`), `expand_var` (`pre_right`) and
`compress_eqn` (`pre_left`) are the batched forms of
`compress_variable_space!` / `expand_variable_space!` /
`compress_equation_space!`. `nothing` means the subproblem has no such
projection, in which case those helpers are a plain copy.

### Factorization state

`factored_key` records the `(dt, a_ii)` a factorization is valid for, alongside
an explicit `dirty` bit. Both, not one: a bare flag plus a reallocated buffer is
how a stale factorization silently serves zeros.

`lu` holds the `BatchedDenseLU` over `lhs_dense`, created on first use. It is a
`Ref{Any}` and not a typed field because `BatchedDenseLU` lives in
`src/tools/batched_matsolvers.jl`, which the package loads AFTER this file. It
is kept on the batch, next to the buffer it factors and the key that validates
it, so a workspace rebuild can never leave a live key pointing at an absent
factorization.
"""
struct ModeBatch
    sp_indices::Vector{Int}
    n::Int
    nmodes::Int

    # CSC — `batched_assemble_lhs!` only. Never a matvec pattern.
    lhs_colptr::AbstractVector{Int}
    lhs_rowval::AbstractVector{Int}
    M_exp_nzval::AbstractMatrix{ComplexF64}   # CSC order, matches lhs_colptr
    L_exp_nzval::AbstractMatrix{ComplexF64}   # CSC order, matches lhs_colptr

    # CSR — `batched_spmv!` only.
    M_min_rowptr::AbstractVector{Int}
    M_min_colval::AbstractVector{Int}
    M_min_nzval::AbstractMatrix{ComplexF64}   # CSR order
    L_rowptr::AbstractVector{Int}
    L_colval::AbstractVector{Int}
    L_nzval::AbstractMatrix{ComplexF64}       # CSR order, from L_exp

    compress_var::Union{Nothing, BatchedSparseOp}
    expand_var::Union{Nothing, BatchedSparseOp}
    compress_eqn::Union{Nothing, BatchedSparseOp}

    lhs_dense::AbstractArray{ComplexF64, 3}
    bc_rows::AbstractVector{Int}

    lu::Base.RefValue{Any}
    factored_key::Ref{Tuple{Float64, Float64}}
    dirty::Ref{Bool}
end

# Device bytes one `_build_batched_op` result occupies: the CSR pattern, stored
# once, plus one `ComplexF64` column per mode. `nothing` is an absent
# preconditioner, which allocates nothing at all.
_batched_op_bytes(::Nothing, ::Int) = 0
_batched_op_bytes(A::SparseMatrixCSC, nmodes::Int) =
    nnz(A) * nmodes * sizeof(ComplexF64) +      # nzval, (nnz, nmodes)
    (size(A, 1) + 1) * sizeof(Int) +            # rowptr
    nnz(A) * sizeof(Int)                        # colval

"""
    mode_batch_bytes(sp::Subproblem, nmodes::Int) -> Int

Bytes `build_mode_batch` will allocate for a bucket of `nmodes` modes shaped
like the representative subproblem `sp` — **every** array the resulting
`ModeBatch` holds, not just the dense LHS.

Counting `lhs_dense` alone (`n^2 * nmodes * 16`) under-counts by roughly
`1 + 3*density(LHS)`: `M_exp_nzval`, `L_exp_nzval` and the CSR `L_nzval` are
each `nnz(LHS) x nmodes`, and `M_min_nzval` plus the three projection operators
add more on top. Measured on the channel problem, dense-only versus this total:

| nx, nz | n  | nmodes | density(LHS) | dense-only | true    | ratio |
|--------|----|--------|--------------|------------|---------|-------|
| 32, 32 | 34 | 17     | 0.305        | 314432     | 647480  | 2.06  |
| 32, 64 | 66 | 17     | 0.279        | 1184832    | 2274104 | 1.92  |
| 64, 64 | 66 | 33     | 0.279        | 2299968    | 4391096 | 1.91  |

The density does not fall off with `n` (the tau rows and the Chebyshev
derivative blocks stay dense), so ~1.9x is what to expect at production sizes,
not a small-problem artifact. This number is the only thing standing between
`batched_modes_max_bytes` and a device OOM: a user who caps at 8 GiB on a 12 GB
card would otherwise get ~15 GB of residency and an out-of-memory failure that
the cap's `@info` never warned about, because the counted number passed the cap.

The `Int` index arrays (`lhs_colptr`, `lhs_rowval`, the CSR patterns,
`bc_rows`, `sp_indices`) are counted too. They do not scale with `nmodes` and
are negligible at production sizes, but they are allocated, and a counter that
sums *everything* is one a test can pin exactly against `sizeof`.

Deliberately excluded: the `BatchedDenseLU` pivot/info arrays, allocated lazily
by `_ensure_batch_factored!` on the first factorization. On GPU they are
`O(n*nmodes)` — 1/(4n) of the dense buffer they accompany, since
`getrf_strided_batched!` factors in place. On CPU `lu` copies each mode, so peak
residency there is about twice what this returns; the cap exists to protect the
device, where it does not.

Raises rather than returning 0 for a subproblem whose matrices were never built:
a 0 here would pass any cap silently, which is precisely the failure mode this
function exists to prevent. Callers reject those subproblems first —
`batch_signature` returns `0x0` for them.
"""
function mode_batch_bytes(sp::Subproblem, nmodes::Int)
    LHS = sp.LHS
    M_exp = sp.M_exp
    (LHS === nothing || M_exp === nothing) && error(
        "mode_batch_bytes: subproblem has no built matrices, so the batch it " *
        "would produce cannot be sized. Callers must reject it first " *
        "(`batch_signature` returns 0x0).")
    n = size(LHS, 1)

    bytes = n * n * nmodes * sizeof(ComplexF64)          # lhs_dense
    bytes += length(LHS.colptr) * sizeof(Int)            # lhs_colptr
    bytes += length(LHS.rowval) * sizeof(Int)            # lhs_rowval
    # `build_mode_batch` sizes BOTH CSC value blocks from `M_exp.nzval`
    # (`nnz_exp`), so this mirrors that rather than adding `L_exp`'s own count.
    bytes += 2 * length(M_exp.nzval) * nmodes * sizeof(ComplexF64)
    bytes += _batched_op_bytes(sp.M_min, nmodes)           # CSR M_min
    bytes += _batched_op_bytes(sp.L_exp, nmodes)           # CSR L_exp
    bytes += _batched_op_bytes(sp.pre_right_pinv, nmodes)  # compress_var
    bytes += _batched_op_bytes(sp.pre_right, nmodes)       # expand_var
    bytes += _batched_op_bytes(sp.pre_left, nmodes)        # compress_eqn
    bytes += length(sp.bc_rows) * sizeof(Int)              # bc_rows
    bytes += nmodes * sizeof(Int)                          # sp_indices
    return bytes
end

"""
    _mode_batch_fourier_axes(sp::Subproblem) -> Int

How many separable axes this subproblem's mode group pins: one `Int` entry per
Fourier axis, `nothing` for every coupled (Chebyshev) axis. `(3, nothing)` in
2-D counts 1; `(2, 3, nothing)` in 3-D counts 2.
"""
function _mode_batch_fourier_axes(sp::Subproblem)
    n = 0
    for g in sp.group
        g isa Int && (n += 1)
    end
    return n
end

# `like` selects the array backend: pass an existing device vector to get device
# storage, or a plain `ComplexF64[]` for host storage. Mirrors the `like=`
# convention already used by `_subproblem_cached_vector!`.
_batch_similar(like::AbstractVector, ::Type{T}, dims...) where {T} =
    similar(like, T, dims...)

"""
    _build_batched_op(sps, indices, select; like) -> BatchedSparseOp or nothing

Pack `select(sp)` for every mode into one CSR pattern plus a per-mode value
matrix. Returns `nothing` when the first mode has no such matrix (a legitimately
absent preconditioner); throws when the modes disagree structurally, because
`bucket_subproblems` promised they would not and a silent mismatch here would
apply mode `m`'s values through mode 1's pattern.

The single `perm` from `csr_pattern` reorders every mode's `nzval` — legal
precisely because the shared pattern is what the bucket signature guarantees.
"""
function _build_batched_op(sps, indices::Vector{Int}, select; like::AbstractVector)
    A1 = select(sps[indices[1]])
    A1 === nothing && return nothing

    rowptr_h, colval_h, perm = csr_pattern(A1)
    nnz_A = length(perm)
    nmodes = length(indices)

    host = Matrix{ComplexF64}(undef, nnz_A, nmodes)
    for (m, i) in enumerate(indices)
        A = select(sps[i])
        (A !== nothing && size(A) == size(A1) &&
         A.colptr == A1.colptr && A.rowval == A1.rowval) || error(
            "build_mode_batch: subproblem $i (batch column $m) has a different " *
            "sparsity pattern from the bucket representative; bucket_subproblems " *
            "must not have grouped them.")
        @inbounds @views host[:, m] .= A.nzval[perm]
    end

    int_like = _batch_similar(like, Int, 0)
    rowptr = _batch_similar(int_like, Int, length(rowptr_h))
    colval = _batch_similar(int_like, Int, length(colval_h))
    nzval = _batch_similar(like, ComplexF64, nnz_A, nmodes)
    copyto!(rowptr, rowptr_h)
    copyto!(colval, colval_h)
    copyto!(nzval, host)

    return BatchedSparseOp(rowptr, colval, nzval, size(A1, 1), size(A1, 2))
end

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

    M_exp = _batch_similar(like, ComplexF64, nnz_exp, nmodes)
    L_exp = _batch_similar(like, ComplexF64, nnz_exp, nmodes)

    # Stage on the host, then upload each block once. Column m == mode
    # sp_indices[m], fixed for the batch's lifetime. These two stay in CSC
    # order: their only consumer is `batched_assemble_lhs!`.
    host_M_exp = Matrix{ComplexF64}(undef, nnz_exp, nmodes)
    host_L_exp = Matrix{ComplexF64}(undef, nnz_exp, nmodes)
    for (m, i) in enumerate(indices)
        sp = sps[i]
        @views host_M_exp[:, m] .= sp.M_exp.nzval
        @views host_L_exp[:, m] .= sp.L_exp.nzval
    end
    copyto!(M_exp, host_M_exp)
    copyto!(L_exp, host_L_exp)

    int_like = _batch_similar(like, Int, 0)
    lhs_colptr = _batch_similar(int_like, Int, length(sp1.LHS.colptr))
    lhs_rowval = _batch_similar(int_like, Int, length(sp1.LHS.rowval))
    copyto!(lhs_colptr, sp1.LHS.colptr)
    copyto!(lhs_rowval, sp1.LHS.rowval)

    # CSR for everything that is multiplied by a vector.
    M_op = _build_batched_op(sps, indices, sp -> sp.M_min; like)
    L_op = _build_batched_op(sps, indices, sp -> sp.L_exp; like)
    (M_op === nothing || L_op === nothing) && error(
        "build_mode_batch: M_min/L_exp missing on the bucket representative; " *
        "batch_signature should have rejected this subproblem.")

    compress_var = _build_batched_op(sps, indices, sp -> sp.pre_right_pinv; like)
    expand_var = _build_batched_op(sps, indices, sp -> sp.pre_right; like)
    compress_eqn = _build_batched_op(sps, indices, sp -> sp.pre_left; like)

    bc_rows = _batch_similar(int_like, Int, length(sp1.bc_rows))
    copyto!(bc_rows, sp1.bc_rows)

    lhs_dense = _batch_similar(like, ComplexF64, n, n, nmodes)

    return ModeBatch(copy(indices), n, nmodes,
                     lhs_colptr, lhs_rowval, M_exp, L_exp,
                     M_op.rowptr, M_op.colval, M_op.nzval,
                     L_op.rowptr, L_op.colval, L_op.nzval,
                     compress_var, expand_var, compress_eqn,
                     lhs_dense, bc_rows,
                     Ref{Any}(nothing), Ref((NaN, NaN)), Ref(true))
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

All of the following must hold:

1. `nprocs == 1` — distributed batching is out of scope, and the per-rank mode
   partitioning plus solve-layout bracket would need MPI verification.
2. `base.batched_modes` resolves true for this device — `nothing` means GPU yes,
   CPU no, so no existing CPU run changes behavior.
3. the bucket holds at least two modes — one mode has nothing to batch.
4. the problem has EXACTLY ONE Fourier axis, i.e. it is 2-D mixed
   Fourier-coupled. That is the declared scope of the batched path and the only
   shape any test on it exercises. A 3-D `(x, y)` Fourier + `z` Chebyshev run
   otherwise qualifies on every other condition — at `nx=ny=nz=64` its 4096
   modes sit well under the default cap — and would engage by default a gather
   path (`_batch_field_plan` over a `(kx, ky, :)` selection) that has never
   executed. `_subproblem_strided_index` may well express that selection
   correctly; "may well" is not a basis for a default-on numerical path.
5. the batch's workspace fits under `base.batched_modes_max_bytes` —
   `mode_batch_bytes`, which counts every array the batch allocates, not just
   the dense LHS.

Condition 5 emits `@info maxlog=1` when it declines, because a silent
performance cliff at large `nz` is exactly what goes unnoticed for months.
Conditions 3 and 4 decline SILENTLY: both are structural non-qualifications, and
an unsupported dimensionality is not a surprise the way a performance cliff is.
Condition 4 is checked BEFORE the cap so a 3-D run never emits the cap's `@info`
either.
"""
function should_batch_modes(base, sps, indices::Vector{Int};
                            is_gpu::Bool, nprocs::Int)
    nprocs == 1 || return false

    setting = base.batched_modes
    enabled = setting === nothing ? is_gpu : setting
    enabled || return false

    length(indices) >= 2 || return false

    sp1 = sps[indices[1]]
    # `mode_batch_bytes` needs the built matrices to size the batch, and
    # `build_mode_batch` needs all four. A bucket from `bucket_subproblems`
    # always has them (`batch_signature` returns `0x0` otherwise); a
    # hand-assembled `indices` might not.
    LHS = sp1.LHS
    LHS === nothing && return false
    (sp1.M_exp === nothing || sp1.M_min === nothing ||
     sp1.L_exp === nothing) && return false

    _mode_batch_fourier_axes(sp1) == 1 || return false

    n = size(LHS, 1)
    bytes = mode_batch_bytes(sp1, length(indices))
    if bytes > base.batched_modes_max_bytes
        @info("Batched mode solve declined: the batch workspace needs $bytes " *
              "bytes for $(length(indices)) modes of order $n, over the " *
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

"""
    mass_selection_plan(M::SparseMatrixCSC) -> Union{Nothing, Tuple{Vector{Int}, Vector{ComplexF64}}}

Decide whether `M` is a scaled partial permutation — at most one nonzero per
column AND per row — and if so return `(src, scale)` describing it: column `j`
draws from row `src[j]` with value `scale[j]`, or `src[j] == 0` for a column
that is entirely empty.

### Why this matters

`M_min` is rank-deficient in every tau/BC formulation (its tau rows and columns
are empty), so `M x = b` is solved per-mode with a sparse least-squares. But the
measured `M_min` is a 0/1 partial permutation, and for such a matrix the
minimum-norm least-squares solution is `x = M⁺b` (the pseudo-inverse — for a
scaled partial permutation, the reciprocal-scaled transpose) for ANY `b` —
matching `b` exactly on the image rows and taking the free null columns to
zero. That is one kernel instead of `nmodes` sparse solves.

The shortcut is only valid for this structure. Applying it to a genuine mass
matrix would produce a plausible wrong answer with no error, so the structure is
VERIFIED here rather than assumed, and callers fall back to the per-mode solver
on `nothing`.

An explicitly stored zero, `NaN`, or `Inf` is treated as disqualifying rather
than as a mapping. `scale[j]` is divided by, so it must be a usable, finite
number: a stored zero would be divided by zero, and an `Inf` scale divides
`b[src[j]]` silently down to a plausible `0` — the same silent-wrong-answer
failure this function exists to prevent, just reached without the least-squares
solve. A non-finite or zero entry also means the true structure is not what the
sparsity pattern suggests, whether or not the shortcut ever divides by it.
"""
function mass_selection_plan(M::SparseMatrixCSC)
    n = size(M, 2)
    size(M, 1) == n || return nothing

    src = zeros(Int, n)
    scale = ones(ComplexF64, n)
    row_used = falses(size(M, 1))

    rows = rowvals(M)
    vals = nonzeros(M)
    for j in 1:n
        r = nzrange(M, j)
        length(r) == 0 && continue          # empty column: src stays 0
        length(r) == 1 || return nothing    # two entries in a column
        k = first(r)
        # stored zero/NaN/Inf: not a real mapping. `isfinite` on a ComplexF64
        # checks both the real and imaginary parts, so this also catches a
        # NaN/Inf hiding in the imaginary part of an otherwise-finite value.
        (iszero(vals[k]) || !isfinite(vals[k])) && return nothing
        i = rows[k]
        row_used[i] && return nothing       # two entries in a row
        row_used[i] = true
        src[j] = i
        scale[j] = vals[k]
    end
    return (src, scale)
end
