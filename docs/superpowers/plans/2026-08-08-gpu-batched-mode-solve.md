# Batched Fourier-Mode Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-Fourier-mode host loop in the 2D GPU coupled solve with one batched pipeline over all modes, taking roughly 7,700 kernel launches per step down to roughly 42.

**Architecture:** Subproblems are bucketed at build time by an observed structural signature. Each bucket with two or more members becomes a `ModeBatch` holding every mode's data as `(n, nmodes)` device matrices with one column per mode. Gather, `M·X`, RHS accumulation, BC override, LHS assembly, and the LU solve each become one batched operation. Buckets that cannot batch keep the existing per-mode loop untouched.

**Tech Stack:** Julia, KernelAbstractions (device-generic kernels), CUDA.jl `CUBLAS.getrf_batched!` / `getrs_batched!`, SparseArrays, Test.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-gpu-batched-mode-solve-design.md`. Read it before starting.
- Julia launcher on this machine: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=.` The `julia`/`juliaup` launcher is broken here.
- Full suite is `Pkg.test()`, **not** `test/runtests.jl` (the latter omits Aqua and JET). It takes about 25 minutes.
- Never run `--project=test`; it creates `test/Project.toml` and breaks the suite.
- Every new `test_*.jl` file MUST be registered in `test/file_lists.jl` or `test_test_inventory.jl` fails.
- Scope is serial single-GPU only. `nprocs > 1` must never construct a batch.
- Every KernelAbstractions kernel writes each element exactly once and never re-reads it. The KA CPU backend miscompiles same-slot read-modify-write around inner loops.
- No silent CPU fallback on GPU (contract #74). GPU-side failures raise.
- Default behavior on CPU and under MPI must be byte-for-byte unchanged.
- The JET ratchet in `test/test_jet.jl` sits at exactly 975 against `<=975`. Zero headroom. Check it after every task that adds a file.
- This is a shared machine with load 8 to 25. Never use wall-clock timing as evidence; use counters and value assertions.
- Do not commit unless the user explicitly asks.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/core/subsystems/mode_batch.jl` (create) | `batch_signature`, `bucket_subproblems`, `ModeBatch`, `build_mode_batches!`, engagement predicate |
| `src/core/subsystems/mode_batch_kernels.jl` (create) | The five KernelAbstractions kernels and their host wrappers |
| `src/tools/batched_matsolvers.jl` (create) | `BatchedDenseLU`: `factor!` / `solve!` only |
| `src/core/timesteppers/step_subproblem_rk_batched.jl` (create) | The batched stage loop |
| `src/core/subsystems/subproblem_runtime.jl` (modify) | Add two `include`s |
| `src/tools/load_matsolvers.jl` (modify) | Add one `include` |
| `src/core/load_solver_stack.jl` (modify) | Add one `include` |
| `src/core/solvers/solver_types.jl` (modify) | Two new `SolverBaseData` fields |
| `src/core/timesteppers/step_subproblem_rk.jl` (modify) | Dispatch to the batched loop |
| `test/test_mode_batch_signature.jl` (create) | Bucketing |
| `test/test_mode_batch_kernels_cpu.jl` (create) | Kernel bit-exactness |
| `test/test_batched_dense_lu.jl` (create) | Linear algebra and singular-mode raise |
| `test/test_mode_batch_parity.jl` (create) | End-to-end parity and guards |
| `test/file_lists.jl` (modify) | Register the four test files |
| `test/run_gpu_fc_2d.jl` (modify) | Cluster GPU coverage |

---

### Task 1: Structural signature and bucketing

The entire bucket-and-loop-leftovers policy rests on this. Uniformity must be **observed from the built matrices**, never inferred from `nz` or `nvars`. Assuming uniformity is how a wrong-but-plausible batch gets built and silently produces garbage.

**Files:**
- Create: `src/core/subsystems/mode_batch.jl`
- Modify: `src/core/subsystems/subproblem_runtime.jl` (add `include("mode_batch.jl")` after `include("subproblem_io.jl")`)
- Test: `test/test_mode_batch_signature.jl`

**Interfaces:**
- Consumes: `Subproblem` from `subproblem_types.jl`, fields `LHS`, `M_min`, `L_min`, `bc_rows`, `bulk_cols`, `bc_cols`, `pre_left`, `pre_right`, `pre_left_pinv`, `pre_right_pinv`.
- Produces:
  - `batch_signature(sp::Subproblem) -> UInt64` — `0x0` means "not batchable at all" (a `nothing` `M_min` or `LHS`).
  - `bucket_subproblems(sps) -> Dict{UInt64, Vector{Int}}` — signature to subproblem indices, in ascending index order.

- [ ] **Step 1: Write the failing test**

Create `test/test_mode_batch_signature.jl`:

```julia
"""
Bucketing tests for the batched Fourier-mode solve.

`batch_signature` must be computed from the matrices as actually built. If it
were derived from `nz`/`nvars` arithmetic instead, a problem whose kx=0 mode
carries a different BC or gauge constraint would be batched together with the
rest and silently solve the wrong system.
"""

using Test
using Tarang
using SparseArrays

function _channel_solver(; nx=16, nz=8, dt=1e-3)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π), dealias=3 / 2)
    zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xbasis, zbasis))

    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
    tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    tau_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * tau_lift(tau1)

    problem = IVP([b, tau1, tau2])
    add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
    add_equation!(problem,
                  "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt)
    step!(solver)   # forces build_matrices!
    return solver
end

@testset "mode batch signature" begin
    solver = _channel_solver()
    sps = collect(solver.problem.compiled.subproblems)
    live = [sp for sp in sps if sp.M_min !== nothing]
    @test length(live) > 1

    @testset "uniform problem yields exactly one bucket" begin
        buckets = Tarang.bucket_subproblems(sps)
        @test length(buckets) == 1
        only_bucket = first(values(buckets))
        @test length(only_bucket) == length(live)
        @test issorted(only_bucket)
    end

    @testset "signature is stable and value-independent" begin
        sig1 = Tarang.batch_signature(live[1])
        sig2 = Tarang.batch_signature(live[2])
        @test sig1 == sig2
        @test sig1 != 0x0
        # nzval differs across modes but must NOT change the signature
        @test live[1].LHS.nzval != live[2].LHS.nzval
    end

    @testset "a perturbed pattern splits into its own bucket" begin
        # Give one mode a structurally different LHS. Signature must change,
        # and bucketing must isolate it rather than batching it with the rest.
        odd = live[end]
        original = odd.LHS
        perturbed = copy(original)
        # Add a structural nonzero where there was none.
        target_row = findfirst(r -> perturbed[r, 1] == 0, 1:size(perturbed, 1))
        @test target_row !== nothing
        perturbed[target_row, 1] = 1.0 + 0.0im
        odd.LHS = perturbed

        @test Tarang.batch_signature(odd) != Tarang.batch_signature(live[1])
        buckets = Tarang.bucket_subproblems(sps)
        @test length(buckets) == 2
        sizes = sort!(collect(length.(values(buckets))))
        @test sizes == [1, length(live) - 1]

        odd.LHS = original
    end

    @testset "kx=0 batches with everyone else" begin
        # Regression pin. `L_min` at kx=0 stores FEWER nonzeros than at other
        # modes (the ∂xx term is the zero operator there), so a signature built
        # over `L_min` splits kx=0 into its own bucket on essentially every
        # problem with a second derivative. The signature uses `L_exp` — same
        # values, LHS's union pattern, uniform across all modes.
        zero_mode = findfirst(sp -> sp.group[1] == 0, live)
        @test zero_mode !== nothing
        other = findfirst(sp -> sp.group[1] != 0, live)

        @test nnz(live[zero_mode].L_min) != nnz(live[other].L_min)   # they DO differ
        @test nnz(live[zero_mode].L_exp) == nnz(live[other].L_exp)   # L_exp does not
        @test Matrix(live[zero_mode].L_exp) == Matrix(live[zero_mode].L_min)

        @test Tarang.batch_signature(live[zero_mode]) ==
              Tarang.batch_signature(live[other])
    end

    @testset "an unbuilt subproblem is not batchable" begin
        sp = live[1]
        saved = sp.M_min
        sp.M_min = nothing
        @test Tarang.batch_signature(sp) == 0x0
        sp.M_min = saved
    end
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_signature.jl")'
```
Expected: FAIL with `UndefVarError: bucket_subproblems not defined`.

- [ ] **Step 3: Write the implementation**

Create `src/core/subsystems/mode_batch.jl`:

```julia
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
```

Then add the include to `src/core/subsystems/subproblem_runtime.jl`, after the `subproblem_io.jl` line:

```julia
include("mode_batch.jl")
```

and extend that file's docstring list with:

```
- mode_batch.jl: structural bucketing and the batched per-mode working set
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_signature.jl")'
```
Expected: PASS, all four `@testset` blocks green.

- [ ] **Step 5: Register the test file**

In `test/file_lists.jl`, add to `TEST_FILES`, next to the other subsystem ratchets:

```julia
    "test_mode_batch_signature.jl",   # batchability must be OBSERVED from built matrices, never inferred from nz/nvars — a gauge-constrained kx=0 mode batched with the rest solves the wrong system silently
```

- [ ] **Step 6: Confirm the inventory guard passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_test_inventory.jl")'
```
Expected: PASS.

---

### Task 2: `ModeBatch` construction

Holds every batched mode's data. Pure shape and storage; no timestepper knowledge.

**Files:**
- Modify: `src/core/subsystems/mode_batch.jl`
- Test: `test/test_mode_batch_signature.jl` (append a testset)

**Interfaces:**
- Consumes: `batch_signature`, `bucket_subproblems` from Task 1; `_subproblem_strided_index(cd, field, sp) -> Union{Nothing,Tuple{Int,Int,Int}}` from `subproblem_io.jl`.
- Produces:
  - `struct ModeBatch` with fields `sp_indices::Vector{Int}`, `n::Int`, `nmodes::Int`, `colptr::AbstractVector{Int}`, `rowval::AbstractVector{Int}`, `M_exp_nzval::AbstractMatrix{ComplexF64}`, `L_exp_nzval::AbstractMatrix{ComplexF64}`, `M_min_nzval::AbstractMatrix{ComplexF64}`, `M_min_colptr::AbstractVector{Int}`, `M_min_rowval::AbstractVector{Int}`, `lhs_dense::AbstractArray{ComplexF64,3}`, `bc_rows::AbstractVector{Int}`, `factored_key::Ref{Tuple{Float64,Float64}}`, `dirty::Ref{Bool}`
  - `build_mode_batch(sps, indices; like) -> ModeBatch`
  - `mode_batch_bytes(n, nmodes) -> Int`
  - `csr_pattern(A::SparseMatrixCSC) -> (rowptr::Vector{Int}, colval::Vector{Int}, perm::Vector{Int})` — the CSR view of `A`'s pattern plus the permutation carrying any mode's CSC `nzval` into CSR order. Used by `batched_spmv!` in Task 3, which iterates rows.

- [ ] **Step 1: Write the failing test**

Append to `test/test_mode_batch_signature.jl`:

```julia
@testset "ModeBatch construction" begin
    solver = _channel_solver()
    sps = collect(solver.problem.compiled.subproblems)
    buckets = Tarang.bucket_subproblems(sps)
    indices = first(values(buckets))

    batch = Tarang.build_mode_batch(sps, indices; like=ComplexF64[])

    sp1 = sps[indices[1]]
    n = size(sp1.LHS, 1)

    @test batch.n == n
    @test batch.nmodes == length(indices)
    @test batch.sp_indices == indices

    @testset "pattern stored once, not per mode" begin
        @test length(batch.colptr) == n + 1
        @test batch.colptr == sp1.LHS.colptr
        @test batch.rowval == sp1.LHS.rowval
    end

    @testset "values stored per mode, column-major by mode" begin
        @test size(batch.M_exp_nzval) == (length(sp1.M_exp.nzval), length(indices))
        @test size(batch.L_exp_nzval) == (length(sp1.L_exp.nzval), length(indices))
        for (m, i) in enumerate(indices)
            @test batch.M_exp_nzval[:, m] == sps[i].M_exp.nzval
            @test batch.L_exp_nzval[:, m] == sps[i].L_exp.nzval
            @test batch.M_min_nzval[:, m] == sps[i].M_min.nzval
        end
    end

    @testset "dense LHS workspace is allocated but not yet valid" begin
        @test size(batch.lhs_dense) == (n, n, length(indices))
        @test batch.dirty[]
    end

    @testset "byte accounting matches the allocated buffer" begin
        @test Tarang.mode_batch_bytes(n, length(indices)) ==
              n * n * length(indices) * sizeof(ComplexF64)
    end

    @testset "csr_pattern inverts CSC and carries nzval across" begin
        A = sparse([1, 3, 2, 3], [1, 1, 2, 3], ComplexF64[5, 7, 11, 13], 3, 3)
        rowptr, colval, perm = Tarang.csr_pattern(A)

        @test length(rowptr) == size(A, 1) + 1
        @test rowptr[1] == 1
        @test rowptr[end] == nnz(A) + 1
        @test length(colval) == nnz(A)
        @test length(perm) == nnz(A)

        # Walking the CSR arrays with the permuted values must reproduce A.
        csr_vals = A.nzval[perm]
        rebuilt = zeros(ComplexF64, 3, 3)
        for r in 1:3, k in rowptr[r]:(rowptr[r + 1] - 1)
            rebuilt[r, colval[k]] = csr_vals[k]
        end
        @test rebuilt == Matrix(A)
    end
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_signature.jl")'
```
Expected: FAIL with `UndefVarError: build_mode_batch not defined`. The Task 1 testsets still pass.

- [ ] **Step 3: Write the implementation**

Append to `src/core/subsystems/mode_batch.jl`:

```julia
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
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_signature.jl")'
```
Expected: PASS, all testsets from Tasks 1 and 2.

---

### Task 3: Batched kernels

Five KernelAbstractions kernels plus host wrappers. Each must be **bit-exact** against the per-mode function it replaces, except `batched_assemble_lhs!`, which is spec'd at `1e-15` relative because a kernel may contract `M + c*L` into an FMA where the host expression does not.

Every kernel writes each element exactly once and never re-reads it.

**Files:**
- Create: `src/core/subsystems/mode_batch_kernels.jl`
- Modify: `src/core/subsystems/subproblem_runtime.jl` (add `include("mode_batch_kernels.jl")` after `mode_batch.jl`)
- Test: `test/test_mode_batch_kernels_cpu.jl`

**Interfaces:**
- Consumes: `ModeBatch` from Task 2; `_gather_strided!` / `_scatter_strided!` from `subproblem_io.jl` as the bit-exactness reference.
- Produces, all returning their first argument:
  - `batched_gather!(X, cd, starts, step, len, row_offset)`
  - `batched_scatter!(cd, X, starts, step, len, row_offset)`
  - `batched_spmv!(Y, colptr, rowval, nzval, X)`
  - `batched_bc_override!(RHS, ALG_F, bc_rows, coeff)`
  - `batched_assemble_lhs!(lhs_dense, colptr, rowval, M_nzval, L_nzval, coeff)`

- [ ] **Step 1: Write the failing test**

Create `test/test_mode_batch_kernels_cpu.jl`:

```julia
"""
Value tests for the batched Fourier-mode kernels — on CPU arrays, through the
REAL kernel objects.

These are KernelAbstractions kernels, so the objects the CUDA path launches on a
`CUDABackend()` also run on `KernelAbstractions.CPU()` over plain `Array`s. That
matters more than usual here: the KA CPU backend miscompiles same-slot
read-modify-write around inner loops (found in `_cheb_coeff_to_deriv_kernel!`
during the PR #105 work), so running the real objects is the only way to catch
that class without hardware.

Each kernel is checked against the per-mode function it replaces. Everything is
bit-exact except `batched_assemble_lhs!`, which computes `M + c*L` and may be
FMA-contracted by the backend.
"""

using Test
using Tarang
using SparseArrays
using LinearAlgebra
using KernelAbstractions

@testset "batched mode kernels (CPU backend)" begin
    n, nmodes, nrows = 9, 5, 7
    rng_vals(k) = ComplexF64[(i * 0.37 + k) + (i * 0.11 - k) * im for i in 1:k]

    @testset "batched_gather! matches _gather_strided! bit-for-bit" begin
        # Emulate a coeff array with one Fourier axis and one coupled axis.
        cd = reshape(ComplexF64[(i + 0.25) + (i - 0.5) * im for i in 1:(nrows * nmodes)],
                     nrows, nmodes)
        step_ = stride(cd, 1)
        starts = [1 + (m - 1) * stride(cd, 2) for m in 1:nmodes]

        X = zeros(ComplexF64, nrows, nmodes)
        Tarang.batched_gather!(X, cd, starts, step_, nrows, 0)

        expected = zeros(ComplexF64, nrows, nmodes)
        for m in 1:nmodes
            buf = zeros(ComplexF64, nrows)
            Tarang._gather_strided!(buf, 0, cd, starts[m], step_, nrows)
            expected[:, m] .= buf
        end
        @test X == expected            # bit-exact, not approx
    end

    @testset "batched_scatter! matches _scatter_strided! bit-for-bit" begin
        cd_batched = zeros(ComplexF64, nrows, nmodes)
        cd_ref = zeros(ComplexF64, nrows, nmodes)
        step_ = stride(cd_batched, 1)
        starts = [1 + (m - 1) * stride(cd_batched, 2) for m in 1:nmodes]

        X = reshape(ComplexF64[(i * 0.5) + (i * 0.25) * im for i in 1:(nrows * nmodes)],
                    nrows, nmodes)

        Tarang.batched_scatter!(cd_batched, X, starts, step_, nrows, 0)
        for m in 1:nmodes
            Tarang._scatter_strided!(cd_ref, X[:, m], 0, starts[m], step_, nrows)
        end
        @test cd_batched == cd_ref
    end

    @testset "batched_spmv! matches per-mode mul! bit-for-bit" begin
        # NOTE: batched_spmv! iterates ROWS, so it takes the CSR pattern.
        # Passing A.colptr/A.rowval (CSC) would silently compute transpose(A)*x
        # and only agree when A is symmetric. Go through csr_pattern.
        A = sprand(ComplexF64, n, n, 0.4)
        rowptr, colval, perm = Tarang.csr_pattern(A)

        nzv_csc = zeros(ComplexF64, nnz(A), nmodes)
        X = zeros(ComplexF64, n, nmodes)
        for m in 1:nmodes
            nzv_csc[:, m] .= A.nzval .* (1 + 0.1m)
            X[:, m] .= rng_vals(n) .* (1 - 0.05m)
        end
        nzv_csr = nzv_csc[perm, :]

        Y = zeros(ComplexF64, n, nmodes)
        Tarang.batched_spmv!(Y, rowptr, colval, nzv_csr, X)

        for m in 1:nmodes
            Am = SparseMatrixCSC(n, n, copy(A.colptr), copy(A.rowval),
                                 nzv_csc[:, m])
            expected = zeros(ComplexF64, n)
            mul!(expected, Am, X[:, m])
            @test Y[:, m] == expected
        end
    end

    @testset "an asymmetric matrix distinguishes CSR from CSC" begin
        # Pins the bug the previous testset would otherwise hide: with a
        # symmetric A, feeding the CSC pattern to a row-iterating kernel gives
        # the right answer by accident.
        A = sparse([1, 2], [2, 1], ComplexF64[3.0, 0.0], 2, 2)
        rowptr, colval, perm = Tarang.csr_pattern(A)
        x = reshape(ComplexF64[1.0, 1.0], 2, 1)
        y = zeros(ComplexF64, 2, 1)
        Tarang.batched_spmv!(y, rowptr, colval, reshape(A.nzval[perm], :, 1), x)
        @test y[1, 1] == 3.0        # A[1,2]*x[2]
        @test y[2, 1] == 0.0
    end

    @testset "batched_bc_override! writes only bc rows" begin
        RHS = reshape(ComplexF64[i + 0.0im for i in 1:(n * nmodes)], n, nmodes)
        ALG = reshape(ComplexF64[100i + 0.0im for i in 1:(n * nmodes)], n, nmodes)
        bc = [2, 5]
        coeff = 0.375
        before = copy(RHS)

        Tarang.batched_bc_override!(RHS, ALG, bc, coeff)

        for m in 1:nmodes, r in 1:n
            if r in bc
                @test RHS[r, m] == coeff * ALG[r, m]
            else
                @test RHS[r, m] == before[r, m]   # untouched, bit-for-bit
            end
        end
    end

    @testset "batched_assemble_lhs! reproduces M + c*L densely" begin
        pattern = sprand(ComplexF64, n, n, 0.5)
        nnzp = nnz(pattern)
        Mv = zeros(ComplexF64, nnzp, nmodes)
        Lv = zeros(ComplexF64, nnzp, nmodes)
        for m in 1:nmodes
            Mv[:, m] .= pattern.nzval .* (0.5 + 0.1m)
            Lv[:, m] .= pattern.nzval .* (2.0 - 0.2m)
        end
        coeff = ComplexF64(0.25, 0.0)

        dense = zeros(ComplexF64, n, n, nmodes)
        Tarang.batched_assemble_lhs!(dense, pattern.colptr, pattern.rowval,
                                     Mv, Lv, coeff)

        for m in 1:nmodes
            expected = Matrix(SparseMatrixCSC(n, n, copy(pattern.colptr),
                                              copy(pattern.rowval),
                                              Mv[:, m] .+ coeff .* Lv[:, m]))
            # 1e-15, not bit-exact: the kernel may FMA-contract M + c*L.
            @test isapprox(dense[:, :, m], expected; rtol=1e-15, atol=1e-15)
        end
    end

    @testset "structural zeros are written, not left stale" begin
        # A dense workspace reused across dt changes must be fully overwritten;
        # a kernel that only touched stored nonzeros would leave the previous
        # factorization's values in the structural-zero slots.
        pattern = sparse([1, 3], [1, 2], ComplexF64[1.0, 2.0], 3, 3)
        dense = fill(ComplexF64(9999.0), 3, 3, 1)
        Mv = reshape(copy(pattern.nzval), :, 1)
        Lv = zeros(ComplexF64, nnz(pattern), 1)

        Tarang.batched_assemble_lhs!(dense, pattern.colptr, pattern.rowval,
                                     Mv, Lv, ComplexF64(1.0))

        @test dense[2, 1, 1] == 0        # structural zero, must be cleared
        @test dense[1, 3, 1] == 0
        @test dense[1, 1, 1] == 1.0
        @test dense[3, 2, 1] == 2.0
    end
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_kernels_cpu.jl")'
```
Expected: FAIL with `UndefVarError: batched_gather! not defined`.

- [ ] **Step 3: Write the implementation**

Create `src/core/subsystems/mode_batch_kernels.jl`:

```julia
# ── Batched Fourier-mode kernels ─────────────────────────────────────────────
#
# Each kernel replaces one per-mode operation in the coupled stage loop with a
# single launch over all modes. Column `m` of every `(n, nmodes)` argument is
# mode `batch.sp_indices[m]`.
#
# WRITE-ONCE CONTRACT: every kernel below writes each output element exactly
# once and never reads it back. The KernelAbstractions CPU backend wraps kernel
# bodies in an ivdep/no-alias workitem loop, which licenses reordering of
# same-slot read-modify-writes around inner loops — that miscompiled the
# Chebyshev recurrence kernel in `ext/cuda/cheb_deriv.jl`. Accumulation is done
# into a register and stored once, never into the output slot in a loop.

using KernelAbstractions

@kernel function _batched_gather_kernel!(X, @Const(cd), @Const(starts),
                                         step_, row_offset)
    i, m = @index(Global, NTuple)
    @inbounds X[row_offset + i, m] = cd[starts[m] + (i - 1) * step_]
end

"""
    batched_gather!(X, cd, starts, step_, len, row_offset) -> X

Gather one strided run per mode out of the coefficient array `cd` into rows
`row_offset+1 : row_offset+len` of `X`. `starts[m]` is mode `m`'s linear start
offset; `step_` and `len` are shared, since every mode selects the same coupled
axis. This is `_gather_strided!` for all modes at once.
"""
function batched_gather!(X::AbstractMatrix{ComplexF64}, cd::AbstractArray,
                         starts::AbstractVector{Int}, step_::Int, len::Int,
                         row_offset::Int)
    backend = get_backend(X)
    _batched_gather_kernel!(backend)(X, cd, starts, step_, row_offset;
                                     ndrange=(len, size(X, 2)))
    KernelAbstractions.synchronize(backend)
    return X
end

@kernel function _batched_scatter_kernel!(cd, @Const(X), @Const(starts),
                                          step_, row_offset)
    i, m = @index(Global, NTuple)
    @inbounds cd[starts[m] + (i - 1) * step_] = X[row_offset + i, m]
end

"""
    batched_scatter!(cd, X, starts, step_, len, row_offset) -> cd

The mirror of `batched_gather!`. Writes rows `row_offset+1 : row_offset+len` of
`X` back into each mode's strided run of `cd`.
"""
function batched_scatter!(cd::AbstractArray, X::AbstractMatrix{ComplexF64},
                          starts::AbstractVector{Int}, step_::Int, len::Int,
                          row_offset::Int)
    backend = get_backend(X)
    _batched_scatter_kernel!(backend)(cd, X, starts, step_, row_offset;
                                      ndrange=(len, size(X, 2)))
    KernelAbstractions.synchronize(backend)
    return cd
end

# One thread per (row, mode). Each thread accumulates that row's dot product in
# a REGISTER and stores once — a CSC-column loop writing into Y[row, m]
# repeatedly is exactly the same-slot RMW shape that miscompiled before.
# Iterating rows requires the CSR view of the pattern, so callers pass the
# TRANSPOSED CSC pattern (equivalently, the CSR pattern of the original).
@kernel function _batched_spmv_kernel!(Y, @Const(rowptr), @Const(colval),
                                       @Const(nzval), @Const(X))
    r, m = @index(Global, NTuple)
    acc = zero(ComplexF64)
    @inbounds for k in rowptr[r]:(rowptr[r + 1] - 1)
        acc += nzval[k, m] * X[colval[k], m]
    end
    @inbounds Y[r, m] = acc
end

"""
    batched_spmv!(Y, rowptr, colval, nzval, X) -> Y

`Y[:, m] = A_m * X[:, m]` for every mode, where all `A_m` share the CSR pattern
`(rowptr, colval)` and `nzval[:, m]` holds mode `m`'s values in that order.
"""
function batched_spmv!(Y::AbstractMatrix{ComplexF64},
                       rowptr::AbstractVector{Int}, colval::AbstractVector{Int},
                       nzval::AbstractMatrix{ComplexF64},
                       X::AbstractMatrix{ComplexF64})
    backend = get_backend(Y)
    _batched_spmv_kernel!(backend)(Y, rowptr, colval, nzval, X;
                                   ndrange=size(Y))
    KernelAbstractions.synchronize(backend)
    return Y
end

@kernel function _batched_bc_override_kernel!(RHS, @Const(ALG_F),
                                              @Const(bc_rows), coeff)
    b, m = @index(Global, NTuple)
    @inbounds r = bc_rows[b]
    @inbounds RHS[r, m] = coeff * ALG_F[r, m]
end

"""
    batched_bc_override!(RHS, ALG_F, bc_rows, coeff) -> RHS

Overwrite the algebraic/BC rows of every mode's stage RHS with
`coeff * ALG_F`, enforcing `L_row * X = F_alg` at each stage. `bc_rows` is
shared across the batch — the bucket signature guarantees it.
"""
function batched_bc_override!(RHS::AbstractMatrix{ComplexF64},
                              ALG_F::AbstractMatrix{ComplexF64},
                              bc_rows::AbstractVector{Int}, coeff::Number)
    isempty(bc_rows) && return RHS
    backend = get_backend(RHS)
    _batched_bc_override_kernel!(backend)(RHS, ALG_F, bc_rows,
                                          ComplexF64(coeff);
                                          ndrange=(length(bc_rows), size(RHS, 2)))
    KernelAbstractions.synchronize(backend)
    return RHS
end

# Two passes, each write-once: zero the dense workspace, then place the stored
# values. A single pass cannot do both without reading back what it wrote.
# Zeroing is mandatory — the workspace is reused across dt changes, and touching
# only the stored nonzeros would leave the previous factorization's values
# sitting in the structural-zero slots.
@kernel function _batched_lhs_zero_kernel!(dense)
    i, j, m = @index(Global, NTuple)
    @inbounds dense[i, j, m] = zero(ComplexF64)
end

@kernel function _batched_lhs_place_kernel!(dense, @Const(colptr), @Const(rowval),
                                            @Const(M_nzval), @Const(L_nzval),
                                            coeff, ncols)
    k, m = @index(Global, NTuple)
    # Locate the column owning stored index k by binary search over colptr.
    lo, hi = 1, ncols
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        if colptr[mid] <= k
            lo = mid
        else
            hi = mid - 1
        end
    end
    @inbounds dense[rowval[k], lo, m] = M_nzval[k, m] + coeff * L_nzval[k, m]
end

"""
    batched_assemble_lhs!(dense, colptr, rowval, M_nzval, L_nzval, coeff) -> dense

Build every mode's dense stage LHS as `M_exp + coeff * L_exp`, on-device, from
values that live on the device permanently. This is what removes the per-mode
host `LHS.nzval` rebuild and its upload under adaptive dt.

Not bit-exact against the host expression: the backend may contract
`M + coeff*L` into an FMA.
"""
function batched_assemble_lhs!(dense::AbstractArray{ComplexF64, 3},
                               colptr::AbstractVector{Int},
                               rowval::AbstractVector{Int},
                               M_nzval::AbstractMatrix{ComplexF64},
                               L_nzval::AbstractMatrix{ComplexF64},
                               coeff::Number)
    backend = get_backend(dense)
    n, _, nmodes = size(dense)
    _batched_lhs_zero_kernel!(backend)(dense; ndrange=(n, n, nmodes))
    KernelAbstractions.synchronize(backend)
    _batched_lhs_place_kernel!(backend)(dense, colptr, rowval, M_nzval, L_nzval,
                                        ComplexF64(coeff), n;
                                        ndrange=(size(M_nzval, 1), nmodes))
    KernelAbstractions.synchronize(backend)
    return dense
end
```

Add to `src/core/subsystems/subproblem_runtime.jl`, after the `mode_batch.jl` line:

```julia
include("mode_batch_kernels.jl")
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_kernels_cpu.jl")'
```
Expected: PASS. If `batched_spmv!` fails, the most likely cause is passing a CSC rather than CSR pattern — the kernel iterates rows, so callers must transpose. Fix the caller, not the kernel; a column-loop kernel would reintroduce the same-slot RMW.

- [ ] **Step 5: Register the test file**

In `test/file_lists.jl`, add to `TEST_FILES`:

```julia
    "test_mode_batch_kernels_cpu.jl",  # the real KA kernel objects on the CPU backend — the KA CPU miscompile of same-slot RMW is invisible to a reimplement-and-compare test
```

- [ ] **Step 6: Confirm the inventory guard passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_test_inventory.jl")'
```
Expected: PASS.

---

### Task 4: `BatchedDenseLU`

Linear algebra only. The `info` check is the single most dangerous line in this whole change: an unchecked singular mode returns garbage that looks like a plausible answer.

**Files:**
- Create: `src/tools/batched_matsolvers.jl`
- Modify: `src/tools/load_matsolvers.jl` (add `include("batched_matsolvers.jl")` after `gpu_matsolvers.jl`)
- Test: `test/test_batched_dense_lu.jl`

**Interfaces:**
- Consumes: nothing from earlier tasks. Standalone.
- Produces:
  - `mutable struct BatchedDenseLU` with `A::AbstractArray{ComplexF64,3}`, `pivots`, `info`, `factored::Bool`
  - `BatchedDenseLU(A::AbstractArray{ComplexF64,3})`
  - `batched_factor!(s::BatchedDenseLU) -> s` — raises on any singular mode
  - `batched_solve!(X, s::BatchedDenseLU, B) -> X`

- [ ] **Step 1: Verify the CUDA.jl batched API signature before writing against it**

I could not check this locally; the plan must not guess. Use the temp-env trick from the PR #105 work, which loads and compiles CUDA.jl on a machine with no driver:

```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia -e '
using Pkg
Pkg.activate(mktempdir())
Pkg.add(["CUDA"])
using CUDA
for f in (:getrf_batched!, :getrs_batched!)
    isdefined(CUDA.CUBLAS, f) || error("CUDA.CUBLAS.$f is absent")
    println(f, ":")
    foreach(m -> println("  ", m), methods(getfield(CUDA.CUBLAS, f)))
end'
```

Record the exact signatures. If they differ from what Step 3 assumes, adjust Step 3 rather than the test. If either function is absent in the installed CUDA.jl, stop and report — the whole approach depends on them.

- [ ] **Step 2: Write the failing test**

Create `test/test_batched_dense_lu.jl`:

```julia
"""
`BatchedDenseLU` — factor and solve every Fourier mode's stage matrix in one
call instead of one call per mode.

The singular-mode test is the important one. `getrf_batched` reports per-matrix
status in an `info` array; if that array goes unchecked, a singular mode returns
whatever happened to be in the buffer, which looks like a plausible answer and
propagates silently through the timestep. It must raise.
"""

using Test
using Tarang
using LinearAlgebra

@testset "BatchedDenseLU" begin
    n, nmodes = 6, 4

    function _well_conditioned_batch(n, nmodes)
        A = zeros(ComplexF64, n, n, nmodes)
        for m in 1:nmodes, j in 1:n, i in 1:n
            A[i, j, m] = (i == j) ? ComplexF64(n + m, 0.5) :
                                    ComplexF64(0.1 * (i - j), 0.05 * m)
        end
        return A
    end

    @testset "solve matches per-slice lu()" begin
        A = _well_conditioned_batch(n, nmodes)
        B = reshape(ComplexF64[(i * 0.3) + (i * 0.7) * im
                               for i in 1:(n * nmodes)], n, nmodes)

        expected = zeros(ComplexF64, n, nmodes)
        for m in 1:nmodes
            expected[:, m] .= lu(A[:, :, m]) \ B[:, m]
        end

        s = Tarang.BatchedDenseLU(copy(A))
        Tarang.batched_factor!(s)
        X = zeros(ComplexF64, n, nmodes)
        Tarang.batched_solve!(X, s, B)

        @test isapprox(X, expected; rtol=1e-12)
    end

    @testset "a singular mode raises, naming the mode" begin
        A = _well_conditioned_batch(n, nmodes)
        A[:, :, 3] .= 0            # mode 3 is exactly singular

        s = Tarang.BatchedDenseLU(A)
        err = try
            Tarang.batched_factor!(s)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("3", sprint(showerror, err))
    end

    @testset "solving before factoring raises" begin
        A = _well_conditioned_batch(n, nmodes)
        s = Tarang.BatchedDenseLU(A)
        X = zeros(ComplexF64, n, nmodes)
        B = ones(ComplexF64, n, nmodes)
        @test_throws Exception Tarang.batched_solve!(X, s, B)
    end

    @testset "refactoring after the matrix changes gives the new answer" begin
        A = _well_conditioned_batch(n, nmodes)
        s = Tarang.BatchedDenseLU(A)
        Tarang.batched_factor!(s)

        B = ones(ComplexF64, n, nmodes)
        X1 = zeros(ComplexF64, n, nmodes)
        Tarang.batched_solve!(X1, s, B)

        # Mutate in place, as batched_assemble_lhs! will, then refactor.
        s.A .*= 2
        Tarang.batched_factor!(s)
        X2 = zeros(ComplexF64, n, nmodes)
        Tarang.batched_solve!(X2, s, B)

        @test isapprox(X2, X1 ./ 2; rtol=1e-12)
    end
end
```

- [ ] **Step 3: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_batched_dense_lu.jl")'
```
Expected: FAIL with `UndefVarError: BatchedDenseLU not defined`.

- [ ] **Step 4: Write the implementation**

Create `src/tools/batched_matsolvers.jl`:

```julia
# ── Batched dense LU over Fourier modes ──────────────────────────────────────
#
# Every mode's stage matrix `(M + dt*a_ii*L)` factored and solved in one call
# instead of one per mode. Dense rather than sparse because the per-mode
# matrices measure ~28% dense with full bandwidth (Chebyshev tau rows), so the
# sparse structure buys almost nothing while costing a per-mode launch.
#
# Sparse-with-shared-symbolic was considered and rejected: the sparsity pattern
# IS identical across modes, but partial pivoting diverges per mode, which
# breaks the shared-symbolic premise.

"""
    BatchedDenseLU(A)

Factor and solve `A[:, :, m] * x = b` for every `m` in one call. `A` is mutated
in place by the factorization, and callers are expected to overwrite it (see
`batched_assemble_lhs!`) and call `batched_factor!` again when `dt` changes.
"""
mutable struct BatchedDenseLU
    A::AbstractArray{ComplexF64, 3}
    pivots::Any
    info::Any
    factored::Bool
end

BatchedDenseLU(A::AbstractArray{ComplexF64, 3}) =
    BatchedDenseLU(A, nothing, nothing, false)

"""
    batched_factor!(s::BatchedDenseLU) -> s

LU-factor every mode in place.

Raises if any mode is singular, naming it. This check is not optional: the
batched LAPACK/CUBLAS entry points report per-matrix status in an `info` array
and return normally regardless, so an unchecked singular mode yields buffer
contents that read as a plausible solution and propagate through the timestep
undetected.
"""
function batched_factor!(s::BatchedDenseLU)
    return _batched_factor_impl!(s)
end

# CPU reference path. The GPU method is added by the CUDA extension.
function _batched_factor_impl!(s::BatchedDenseLU)
    A = s.A
    n, _, nmodes = size(A)
    facts = Vector{Any}(undef, nmodes)
    for m in 1:nmodes
        F = lu(view(A, :, :, m); check=false)
        if !issuccess(F)
            error("BatchedDenseLU: mode $m of $nmodes is singular " *
                  "(order $n). A singular stage matrix usually means the " *
                  "problem is under-constrained at this Fourier mode — check " *
                  "the tau/BC rows for that mode.")
        end
        facts[m] = F
    end
    s.pivots = facts
    s.info = zeros(Int, nmodes)
    s.factored = true
    return s
end

"""
    batched_solve!(X, s::BatchedDenseLU, B) -> X

Solve every mode against the stored factorization. `X` and `B` are
`(n, nmodes)`, column `m` being mode `m`. `X` may alias `B`.
"""
function batched_solve!(X::AbstractMatrix{ComplexF64}, s::BatchedDenseLU,
                        B::AbstractMatrix{ComplexF64})
    s.factored || error("BatchedDenseLU: batched_solve! called before " *
                        "batched_factor!; the factorization is stale or absent")
    return _batched_solve_impl!(X, s, B)
end

function _batched_solve_impl!(X::AbstractMatrix{ComplexF64}, s::BatchedDenseLU,
                              B::AbstractMatrix{ComplexF64})
    X === B || copyto!(X, B)
    for m in axes(X, 2)
        ldiv!(s.pivots[m], view(X, :, m))
    end
    return X
end
```

Add to `src/tools/load_matsolvers.jl`:

```julia
include("batched_matsolvers.jl")
```

- [ ] **Step 5: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_batched_dense_lu.jl")'
```
Expected: PASS, all four testsets.

- [ ] **Step 6: Add the GPU method in the CUDA extension**

In `ext/cuda/`, add methods specialized on `CuArray` storage, using the signatures recorded in Step 1. The shape is:

```julia
function Tarang._batched_factor_impl!(s::Tarang.BatchedDenseLU)
    A = s.A
    A isa CuArray || return invoke(Tarang._batched_factor_impl!,
                                   Tuple{Tarang.BatchedDenseLU}, s)
    n, _, nmodes = size(A)
    ptrs = [view(A, :, :, m) for m in 1:nmodes]
    pivots, info = CUDA.CUBLAS.getrf_batched!(ptrs, true)
    host_info = Array(info)
    bad = findall(!iszero, host_info)
    if !isempty(bad)
        error("BatchedDenseLU (GPU): singular stage matrix at mode(s) " *
              "$(bad) of $nmodes; cuBLAS info = $(host_info[bad]). " *
              "No CPU fallback is attempted — see the no-silent-fallback " *
              "contract (#74).")
    end
    s.pivots = pivots
    s.info = info
    s.factored = true
    return s
end
```

with the matching `_batched_solve_impl!` calling `CUDA.CUBLAS.getrs_batched!`. Do not add a `try`/`catch` that reroutes to the CPU path: a GPU failure must raise.

- [ ] **Step 7: Register the test file and confirm the inventory guard**

In `test/file_lists.jl`, add to `TEST_FILES`:

```julia
    "test_batched_dense_lu.jl",        # getrf_batched reports singularity in an info ARRAY and returns normally — an unchecked singular mode returns buffer contents that read as a plausible solution
```

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_test_inventory.jl")'
```
Expected: PASS.

---

### Task 5: Solver options and the engagement predicate

Four conditions gate batching. Getting the *defaults* right is what keeps every existing CPU and MPI run byte-for-byte unchanged.

**Files:**
- Modify: `src/core/solvers/solver_types.jl:89-95` (struct) and `:121-138` (keyword constructor)
- Modify: `src/core/subsystems/mode_batch.jl` (append the predicate)
- Test: `test/test_mode_batch_parity.jl` (guard testsets; parity comes in Task 6)

**Interfaces:**
- Consumes: `mode_batch_bytes` from Task 2; `bucket_subproblems` from Task 1.
- Produces:
  - `SolverBaseData` gains `batched_modes::Union{Nothing,Bool}` and `batched_modes_max_bytes::Int`.
  - `should_batch_modes(base, sps, indices; is_gpu, nprocs) -> Bool`
  - `build_mode_batches!(solver, sps) -> Vector{ModeBatch}`

- [ ] **Step 1: Write the failing test**

Create `test/test_mode_batch_parity.jl` with the guards first (reuse `_channel_solver` by including the signature test's helper via a local copy — do not `include` another test file, as the inventory guard treats every `test_*.jl` as an entry point).

**`_parity_channel_solver` MUST take a `device` keyword defaulting to `CPU()`**, and pass it straight to `Distributor`. Without it the whole file is hard-wired to `CPU()`, and Task 7's cluster runner would include this file on a real GPU, run the CPU path, exercise no CUBLAS code at all, and report a pass — coverage that looks real and is not. The local suite still runs everything on `CPU()`; the parameter exists so the GPU runner can drive the same assertions on device.

```julia
"""
Engagement guards and end-to-end parity for the batched Fourier-mode solve.

The guards matter as much as the parity: batching must be OFF by default on CPU
and must never construct under MPI, so that every existing run is byte-for-byte
unchanged. `batched_modes=true` is what the suite uses to exercise the real
device-generic code path without a GPU.
"""

using Test
using Tarang

function _parity_channel_solver(; nx=16, nz=8, dt=1e-3, kwargs...)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π), dealias=3 / 2)
    zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xbasis, zbasis))

    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
    tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    tau_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * tau_lift(tau1)

    problem = IVP([b, tau1, tau2])
    add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
    add_equation!(problem,
                  "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt, kwargs...)
    return solver, b
end

@testset "batched mode engagement guards" begin
    @testset "default on CPU is OFF" begin
        solver, _ = _parity_channel_solver()
        @test solver.base.batched_modes === nothing
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=false, nprocs=1)
    end

    @testset "opt-in turns it on for CPU" begin
        solver, _ = _parity_channel_solver(; batched_modes=true)
        @test solver.base.batched_modes === true
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test Tarang.should_batch_modes(solver.base, sps, indices;
                                        is_gpu=false, nprocs=1)
    end

    @testset "batched_modes=false overrides even GPU" begin
        solver, _ = _parity_channel_solver(; batched_modes=false)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=true, nprocs=1)
    end

    @testset "MPI never batches, whatever the flag says" begin
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=true, nprocs=2)
    end

    @testset "a one-mode bucket declines silently" begin
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices[1:1];
                                         is_gpu=false, nprocs=1)
    end

    @testset "exceeding the byte cap declines and says so" begin
        solver, _ = _parity_channel_solver(; batched_modes=true,
                                             batched_modes_max_bytes=1)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=false, nprocs=1)
    end

    @testset "the byte counter matches what is actually allocated" begin
        # Task 2's byte test restated the formula instead of measuring the
        # buffer, so the counter and the allocation could drift apart and the
        # memory cap would guard a number nothing allocates. Measure the real
        # thing: this is the gate's whole purpose.
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        batch = Tarang.build_mode_batch(sps, indices; like=ComplexF64[])

        @test Tarang.mode_batch_bytes(batch.n, batch.nmodes) ==
              sizeof(batch.lhs_dense)
    end
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_parity.jl")'
```
Expected: FAIL — `batched_modes` is not an accepted keyword.

- [ ] **Step 3: Extend `SolverBaseData`**

In `src/core/solvers/solver_types.jl`, add two fields to the struct at line 89:

```julia
mutable struct SolverBaseData
    problem::Problem
    matrix_coupling::Vector{Bool}
    entry_cutoff::Float64
    matsolver::Any         # Solver choice (Symbol, Tuple, or concrete solver)
    evaluator::Union{Nothing, AbstractEvaluator}
    # Batched per-Fourier-mode solve. `nothing` means automatic: batch on GPU,
    # do not batch on CPU. `true` forces it on wherever the structural
    # conditions allow (this is what the test suite uses to exercise the
    # device-generic path without a GPU); `false` forces it off everywhere.
    batched_modes::Union{Nothing, Bool}
    batched_modes_max_bytes::Int
end
```

Update the keyword constructor at line 121 and both inner calls at 135 and 138:

```julia
function SolverBaseData(problem::Problem; matrix_coupling=nothing,
                        entry_cutoff::Real=1e-12, matsolver=:sparse,
                        batched_modes::Union{Nothing, Bool}=nothing,
                        batched_modes_max_bytes::Int=1 << 30)
```

and pass `batched_modes, batched_modes_max_bytes` as the two new trailing positional arguments at both construction sites. Then thread the two keywords through the three call sites at lines 473, 915, and 1052, each of which currently reads `SolverBaseData(problem; matsolver=...)`.

- [ ] **Step 4: Write the predicate**

Append to `src/core/subsystems/mode_batch.jl`:

```julia
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
```

- [ ] **Step 5: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_parity.jl")'
```
Expected: PASS, all six guard testsets.

- [ ] **Step 6: Confirm no existing solver test regressed**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_solvers.jl")'
```
Expected: PASS. `test/test_solvers.jl:673` constructs `SolverBaseData(problem)` positionally-free, so the new keywords must have defaults — if this fails, the defaults are missing.

- [ ] **Step 7: Register the test file and confirm the inventory guard**

In `test/file_lists.jl`, add to `TEST_FILES`:

```julia
    "test_mode_batch_parity.jl",       # batching must be OFF by default on CPU and never construct under MPI, or every existing run silently changes numerics
```

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_test_inventory.jl")'
```
Expected: PASS.

---

### Task 6: The batched stage loop

Same mathematics as `step_subproblem_rk.jl:487-599`, run on `(n, nmodes)` matrices.

**Files:**
- Create: `src/core/timesteppers/step_subproblem_rk_batched.jl`
- Modify: `src/core/load_solver_stack.jl` (add the include after `step_subproblem_rk.jl`)
- Modify: `src/core/timesteppers/step_subproblem_rk.jl` (dispatch)
- Test: `test/test_mode_batch_parity.jl` (append the parity testset)

**Interfaces:**
- Consumes, with the signatures AS SHIPPED by Tasks 1-5 (an earlier draft of this plan listed a shorter `build_mode_batches!(solver, sps)` — that form does not exist):
  - `should_batch_modes(base, sps, indices; is_gpu::Bool, nprocs::Int) -> Bool`
  - `build_mode_batches!(base, sps; is_gpu::Bool, nprocs::Int, like::AbstractVector) -> Vector{ModeBatch}` (empty when nothing qualifies — a normal outcome, not an error)
  - `batched_gather!`, `batched_scatter!`, `batched_spmv!`, `batched_bc_override!`, `batched_assemble_lhs!`
  - `BatchedDenseLU{AT}`, `batched_factor!`, `batched_solve!`
  - `csr_pattern(A) -> (rowptr, colval, perm)`
- Produces: `step_subproblem_rk_batched!(solver, state, dt, t, batches, leftover_indices)` and `active_mode_batches(solver) -> Vector{ModeBatch}`.

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_mode_batch_parity.jl`:

```julia
@testset "batched stage loop reproduces the per-mode loop" begin
    nsteps = 5

    ref_solver, ref_b = _parity_channel_solver(; batched_modes=false)
    bat_solver, bat_b = _parity_channel_solver(; batched_modes=true)

    # Identical, non-trivial initial condition on both.
    for (solver, b) in ((ref_solver, ref_b), (bat_solver, bat_b))
        ensure_layout!(b, :g)
        gd = get_grid_data(b)
        for idx in CartesianIndices(gd)
            gd[idx] = sin(0.3 * sum(Tuple(idx))) + 0.1 * prod(Tuple(idx)) % 1
        end
    end

    for _ in 1:nsteps
        step!(ref_solver)
        step!(bat_solver)
    end

    ensure_layout!(ref_b, :g)
    ensure_layout!(bat_b, :g)
    ref_g = Array(get_grid_data(ref_b))
    bat_g = Array(get_grid_data(bat_b))

    scale = maximum(abs, ref_g)
    @test scale > 1e-8                       # guard against comparing zeros
    @test maximum(abs, bat_g .- ref_g) / scale < 1e-12
end

@testset "batches actually engaged during the run" begin
    solver, _ = _parity_channel_solver(; batched_modes=true)
    step!(solver)
    batches = Tarang.active_mode_batches(solver)
    @test !isempty(batches)
    @test sum(b -> b.nmodes, batches) ==
          count(sp -> sp.M_min !== nothing,
                solver.problem.compiled.subproblems)
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_parity.jl")'
```
Expected: the guard testsets still pass; the two new ones FAIL (`active_mode_batches` undefined, and the parity test showing the batched solver taking the per-mode path so the two agree trivially — treat a *passing* parity test at this stage as a failure to engage, which is what the second testset pins).

- [ ] **Step 3: Write the batched stage loop**

Create `src/core/timesteppers/step_subproblem_rk_batched.jl`. Mirror `step_subproblem_rk!` exactly, substituting batched calls. The structure:

```julia
# ── Batched per-mode IMEX Runge-Kutta hot path ───────────────────────────────
#
# Mathematically identical to the per-mode loop in `step_subproblem_rk.jl`; the
# only change is that every mode is a COLUMN of an `(n, nmodes)` matrix instead
# of its own vector, so each operation is one launch rather than `nmodes` of
# them. At Nx=256 with RK222 that is ~42 launches per step against ~7,700.
#
# Sign convention, stage formula, and BC override semantics are unchanged — see
# the header of `step_subproblem_rk.jl`.

function step_subproblem_rk_batched!(solver, state, dt::Float64, t::Float64,
                                     batch::ModeBatch, sps, state_fields,
                                     tableau)
    stages = tableau.stages
    A_exp, A_imp, c = tableau.A_exp, tableau.A_imp, tableau.c
    n, nmodes = batch.n, batch.nmodes

    ws = _batch_workspace!(state, batch, stages)   # X0, MX0, RHS, F[j], LX[j], ALG_F

    # Pre-stage: gather X_n and form M*X_n, both batched.
    _batched_gather_state!(ws.X0, batch, sps, state_fields)
    batched_spmv!(ws.MX0, batch.M_min_rowptr, batch.M_min_colval,
                  batch.M_min_nzval, ws.X0)
    _batched_gather_alg_F!(ws.ALG_F, batch, sps)   # identity-gated, as before

    for i in 1:stages
        state.current_substep = i

        copyto!(ws.RHS, ws.MX0)
        for j in 1:(i - 1)
            a_ej = dt * A_exp[i, j]
            a_ij = dt * A_imp[i, j]
            abs(a_ej) > 1e-14 && (ws.RHS .+= a_ej .* ws.F[j])
            abs(a_ij) > 1e-14 && (ws.RHS .-= a_ij .* ws.LX[j])
        end

        a_ii = A_imp[i, i]
        if abs(a_ii) > 1e-14
            batched_bc_override!(ws.RHS, ws.ALG_F, batch.bc_rows, dt * a_ii)
            _ensure_batch_factored!(batch, dt, a_ii)
            batched_solve!(ws.X, batch.lu, ws.RHS)
        else
            _ensure_mass_factored!(batch)
            batched_solve!(ws.X, batch.mass_lu, ws.RHS)
        end

        _batched_scatter_state!(batch, sps, state_fields, ws.X)
        # evaluate_rhs is whole-field and already batched — unchanged.
        ...
        _batched_gather_outputs!(ws.F[i], batch, sps, F_fields)
        batched_spmv!(ws.LX[i], batch.L_rowptr, batch.L_colval,
                      batch.L_nzval, ws.X)
    end
end
```

Three details that must be right:

1. `_ensure_batch_factored!` compares `batch.factored_key[] == (dt, a_ii)` **and** checks `batch.dirty[]`. Both, not either. It calls `batched_assemble_lhs!` then `batched_factor!`, then sets the key and clears the dirty bit. The existing dt-change handler at `step_subproblem_rk.jl:430-441` must also set `batch.dirty[] = true`.

   **`batched_factor!` must NEVER run without an immediately preceding `batched_assemble_lhs!` over the same buffer.** Task 4's review established a CPU/GPU lifecycle divergence: the CPU path's `lu(view(A,:,:,m); check=false)` *copies*, so `s.A` survives factorization intact, while the GPU path's `getrf_strided_batched!` *overwrites* `A` with the LU factors. Any code that factors a buffer twice without re-assembling it in between therefore works on CPU and silently factors the factors on GPU — a plausible wrong answer with no error, this repository's signature bug class. Put the two calls in that order inside `_ensure_batch_factored!` and nowhere else, and add a test asserting `batched_factor!` is unreachable except through it (e.g. count assemble and factor calls over a multi-step run with a `dt` change and assert they are equal).
2. The batched `L·X` product uses **`L_exp`, never `L_min`** — `L_min`'s pattern is not uniform across modes (kx=0 stores fewer nonzeros because `∂xx` vanishes there), while `L_exp` holds the same values in `LHS`'s union pattern and is uniform. Multiplying by the extra stored zeros changes nothing numerically. Then: `batched_spmv!` iterates rows, so `ModeBatch` must carry the **CSR** pattern for `M_min` and `L_exp`.

Task 2 shipped `ModeBatch` holding `M_min_colptr`/`M_min_rowval` — the **CSC** pattern — and left `csr_pattern` with no caller. That combination is a loaded gun: handing those fields to `batched_spmv!` computes `transpose(M_min)·x` silently, and `M_min` is not symmetric. **REPLACE those two fields, do not add CSR fields alongside them.** No CSC pattern field may survive on `ModeBatch` once this task is done, so that the mistake is unrepresentable rather than merely undocumented.

Extend `build_mode_batch` to call `csr_pattern` (Task 2) once per matrix at build time, store `M_min_rowptr`/`M_min_colval`/`L_rowptr`/`L_colval`, and permute every mode's `nzval` with the single returned `perm` — legal precisely because the bucket signature guarantees a shared pattern.

Add a test asserting `ModeBatch` exposes no CSC pattern field (`!hasfield(Tarang.ModeBatch, :M_min_colptr)`), and one asserting `batched_spmv!` over a batch built from a real solver reproduces per-mode `mul!` with `M_min` — which fails if CSR and CSC are confused, because `M_min` is asymmetric.
3. The gather/scatter `starts` vector: compute each mode's `(start, step, len)` once at build time via `_subproblem_strided_index(cd, field, sp)` for each field, assert `step` and `len` agree across modes, and store `starts` as a device vector. Do **not** assume `starts[m] = starts[1] + (m-1)*stride`; store the measured values.

- [ ] **Step 4: Wire the dispatch**

In `src/core/timesteppers/step_subproblem_rk.jl`, near the top of `step_subproblem_rk!` after `subproblems` is available, build or fetch the cached batches and route the batched subproblems to the new loop while leaving the leftovers on the existing loop. Cache the batch vector on `state.timestepper_data` under `:_sp_rk_mode_batches`, keyed on the identity of the subproblem tuple, matching the existing `_sp_slots!` idiom. Add `active_mode_batches(solver)` returning that cached vector (empty when batching declined) so tests can assert engagement.

Add to `src/core/load_solver_stack.jl` after the `step_subproblem_rk.jl` line:

```julia
include("timesteppers/step_subproblem_rk_batched.jl")
```

- [ ] **Step 5: Run the parity test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_mode_batch_parity.jl")'
```
Expected: PASS, all eight testsets, including `sum(b -> b.nmodes, batches)` covering every live subproblem.

- [ ] **Step 6: Confirm the per-mode path is untouched**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_timesteppers.jl"); include("test/test_solvers.jl")'
```
Expected: PASS. Any failure here means the dispatch changed default behavior, which the design forbids.

---

### Task 7: Suite, ratchets, and GPU runner

**Files:**
- Modify: `test/run_gpu_fc_2d.jl`
- Modify: `test/test_jet.jl` only if the ratchet must move, and only with justification

- [ ] **Step 1: Run the JET ratchet**

JET is a **test-only dependency**: it resolves only inside `Pkg.test()`, so
`julia --project=. -e 'include("test/test_jet.jl")'` fails with
`ArgumentError: Package JET not found in current path`. Same trap as
`test_test_inventory.jl` needing `file_lists.jl` included first. There is no
cheap standalone JET check — it runs as part of Step 3's full suite.

Run the full suite (Step 3) and read the JET result out of it:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Pkg; Pkg.test()' 2>&1 | tee /tmp/tarang-suite.log
grep -A5 -i "jet" /tmp/tarang-suite.log | head -40
```
Expected: PASS at `<=975`. If it fails, the count is reported in the failure message. Fix by improving type stability in the new files before considering a bound change: the usual culprits here are `AbstractMatrix{ComplexF64}` fields in `ModeBatch` and `pivots::Any` / `info::Any` in `BatchedDenseLU`. Narrowing `ModeBatch` to a parametric struct over its array types is the first thing to try. Only raise the bound with an explicit note saying which files added which entries and why they are irreducible.

- [ ] **Step 2: Read the Aqua result out of the same suite log**

Aqua is a test-only dependency too — it sits in `Project.toml`'s `[extras]` and
`[targets]`, exactly like JET. `julia --project=. -e 'include("test/test_aqua.jl")'`
fails the same way. Do not run it standalone.

```bash
grep -B2 -A8 -i "aqua" /tmp/tarang-suite.log | head -40
```
Expected: PASS. New exported names, if any, need docstrings.

- [ ] **Step 3: (folded into Step 1)**

Steps 1 and 2 both read `/tmp/tarang-suite.log`, which Step 1's `Pkg.test()` run
produces. Run the suite **once**, not three times — it takes about 25 minutes per
run on this shared machine.

Run it in the FOREGROUND with a long timeout, or background it and poll the log
file with `tail`. Do **not** hand it to a monitor and wait for a callback: two
implementers on this plan stalled doing exactly that and had to be nudged.

Expected in the log: `Testing Tarang tests passed`.

- [ ] **Step 4: Run the MPI suite**

Batching must never construct under MPI, so this must be unchanged.

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/run_mpi_ci.jl
```
Expected: 49/49 at np=4, matching the pre-change baseline.

- [ ] **Step 5: Extend the GPU cluster runner**

The point of this step is to make a cluster run actually execute
`CUBLAS.getrf_strided_batched!` / `getrs_strided_batched!` — the only code in this
whole change that no local test can reach. (Note: `getrf_strided_batched!`, not
`getrf_batched!`; Task 4 switched entry points after investigating the real API.)

**Simply including the existing test files does NOT achieve that.** Both
`test_batched_dense_lu.jl` and `test_mode_batch_parity.jl` construct
`device=CPU()` solvers and plain `Array`s, so on a GPU machine they would run the
CPU path, exercise no CUBLAS code, and report a pass. That is worse than no
coverage, because it reads as coverage.

Add a GPU-specific block to `test/run_gpu_fc_2d.jl` that drives the SAME assertions
on device:

```julia
# Batched Fourier-mode solve on real hardware. This is the ONLY place the
# CUBLAS strided-batched factor/solve is ever executed — every local test runs
# the device-generic CPU path. Reuses the parity helper's `device` keyword
# rather than duplicating the problem setup.
include(joinpath(@__DIR__, "test_mode_batch_parity.jl"))

@testset "batched mode solve on GPU" begin
    ref_solver,  ref_b  = _parity_channel_solver(; device=GPU(), batched_modes=false)
    bat_solver,  bat_b  = _parity_channel_solver(; device=GPU(), batched_modes=true)

    @test !isempty(Tarang.active_mode_batches(bat_solver))   # it really engaged
    @test  isempty(Tarang.active_mode_batches(ref_solver))   # and really did not

    for _ in 1:5
        step!(ref_solver); step!(bat_solver)
    end
    ensure_layout!(ref_b, :g); ensure_layout!(bat_b, :g)
    r = Array(get_grid_data(ref_b)); b = Array(get_grid_data(bat_b))
    scale = maximum(abs, r)
    @test scale > 1e-8
    @test maximum(abs, b .- r) / scale < 1e-12
end

# Singular-mode detection on device: the `info` array is a DEVICE array and its
# check has never run against real hardware. Assert it raises rather than
# returning buffer contents.
@testset "batched LU singular mode raises on GPU" begin
    A = CUDA.zeros(ComplexF64, 4, 4, 3)
    A[:, :, 1] .= CUDA.CuArray(Matrix{ComplexF64}(I, 4, 4))
    A[:, :, 3] .= CUDA.CuArray(Matrix{ComplexF64}(I, 4, 4))
    s = Tarang.BatchedDenseLU(A)                  # mode 2 is exactly singular
    @test_throws Exception Tarang.batched_factor!(s)
end
```

The two `active_mode_batches` assertions are the load-bearing ones: without them a
regression that silently disables batching on GPU would leave this testset passing.

- [ ] **Step 6: Report results**

Report the JET count before and after, the full-suite result, the MPI count, and state plainly that GPU hardware execution and the launch-count reduction remain **unverified locally** and need a cluster run. Do not describe the change as complete on the strength of CPU tests alone.

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: signature and bucketing to Task 1; `ModeBatch` to Task 2; the five kernels to Task 3; `BatchedDenseLU` and the `info` check to Task 4; the engagement predicate, both solver keywords, and the `@info` fallback to Task 5; the batched stage loop and data flow to Task 6; registration, ratchets, and the GPU runner to Task 7. The spec's three numerical tolerance classes appear as: bit-exact assertions in Task 3, `1e-15` for `batched_assemble_lhs!` in Task 3, and `1e-12` for the solve and trajectory in Tasks 4 and 6.

**Type consistency.** `batch_signature -> UInt64` with `0x0` as the sentinel is used consistently in Tasks 1 and 5. `build_mode_batch(sps, indices; like)` keeps the same signature in Tasks 2, 5, and 6. `batched_factor!` / `batched_solve!` are named identically in Tasks 4 and 6. `should_batch_modes(base, sps, indices; is_gpu, nprocs)` matches between its definition in Task 5 and its uses in Task 5's tests and Task 6.

**Known gap, deliberately left.** Task 6 Step 3 specifies the batched loop as structure plus three must-get-right details rather than complete literal code. The stage loop is a 110-line transliteration of an existing 110-line loop whose surrounding context (`solve_stash` bracketing, `evaluate_rhs_buffered` call, `_refresh_bcs_for_stage!`) must be read from `step_subproblem_rk.jl:463-711` at implementation time. Writing it out here would duplicate that file into the plan and go stale against it. The three details called out are the ones a careful reader would otherwise get wrong: the two-part factorization gate, the CSR-versus-CSC pattern for `batched_spmv!`, and storing measured `starts` rather than assuming a uniform mode stride.
