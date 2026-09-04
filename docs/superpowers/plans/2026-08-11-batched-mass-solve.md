# Batched Mass Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the last per-mode bottleneck from the batched Fourier-mode solve, taking it from ~9.7x to near the original target, by recognising that the mass matrix is a 0/1 partial permutation whose pseudo-inverse is its transpose.

**Architecture:** At batch-build time, verify that `M_min` is structurally a scaled partial permutation (at most one nonzero per row AND per column). When it is, the minimum-norm least-squares solution of `M x = b` is `x[j] = b[row_of(j)] / v_j`, with zeros on null columns — a single write-once kernel over `(n, nmodes)`. When it is not, fall back to the existing per-mode SPQR path, unchanged.

**Tech Stack:** Julia, KernelAbstractions, SparseArrays.

## Background — the measurement this rests on

`step_subproblem_rk_batched.jl` currently batches everything except the mass
solve, which runs per-mode via `SPQRSolver` because `M_min` is rank-deficient.
That costs `2 x nmodes` sparse least-squares solves per RK222 step (the
`a_ii == 0` first stage, and the final update — RK222 is implicitly but not
explicitly stiffly accurate), and is why the branch measured 8,127 -> 839
operations per step rather than the ~65 originally projected.

Measured on the channel problem at two resolutions:

| quantity | `nx=16, nz=8` | `nx=32, nz=16` |
| --- | --- | --- |
| `size(M_min)` | 10 x 10 | 18 x 18 |
| `nnz(M_min)` | 8 | 16 |
| `rank(M_min)` | 8 | 16 |
| every nonzero value | `1.0` | `1.0` |
| zero rows | `[2, 3]` | `[2, 3]` |
| zero columns | `[1, 2]` | `[1, 2]` |
| identical across all modes | yes | yes |

So `M_min` is a 0/1 partial permutation: the tau columns are empty, and every
other column maps to exactly one row. It is **not** diagonal — the row and
column index sets differ — so a masked divide in place is wrong; the values
must be permuted.

For such a matrix the minimum-norm least-squares solution of `M x = b` is
`x = M' b`, for **any** `b`, not only for `b` in the range of `M`: minimising
`||Mx - b||` matches `b` exactly on the image rows and leaves the null columns
free, which the minimum-norm choice sets to zero.

Confirmed against the real solver: for `b = M * rhs`, `SPQRSolver` returns
`[0, 0, rhs[3], rhs[4], ...]` — the input with the null columns zeroed, which
is exactly `M' b`.

## Global Constraints

- Julia launcher: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=.` The plain `julia`/`juliaup` launcher is BROKEN on this machine.
- NEVER run with `--project=test`; it creates `test/Project.toml` and breaks the suite.
- `test/test_jet.jl` and `test/test_aqua.jl` cannot run standalone — both are test-only dependencies resolving only under `Pkg.test()`.
- Every new `test_*.jl` must be registered in `test/file_lists.jl`.
- Every KernelAbstractions kernel writes each output element exactly once and never re-reads it. The KA CPU backend miscompiles same-slot read-modify-write around inner loops.
- Run tests in the FOREGROUND. Do not background them and do not wait on a monitor.
- SHARED machine under heavy load; never use wall-clock timing as evidence.
- Branch: `perf/gpu-batched-mode-solve`, worktree `/Users/subha/Documents/GitHub/Tarang.jl/.worktrees/gpu-batched-mode-solve`.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/core/subsystems/mode_batch.jl` (modify) | `mass_selection_plan(M)` — the structural check and the permutation/scale vectors |
| `src/core/subsystems/mode_batch_kernels.jl` (modify) | `batched_mass_apply!` — the write-once kernel |
| `src/core/timesteppers/step_subproblem_rk_batched.jl` (modify) | Use the fast path when planned; keep the per-mode fallback otherwise |
| `test/test_batched_mass_solve.jl` (create) | Structural check, kernel value equivalence vs SPQR, guard behaviour |
| `test/test_mode_batch_parity.jl` (modify) | Trajectory parity and an operation-count assertion |
| `test/file_lists.jl` (modify) | Register the new test file |

---

### Task 1: Structural check and plan

**Files:**
- Modify: `src/core/subsystems/mode_batch.jl`
- Test: `test/test_batched_mass_solve.jl`

**Interfaces:**
- Produces: `mass_selection_plan(M::SparseMatrixCSC) -> Union{Nothing, Tuple{Vector{Int}, Vector{ComplexF64}}}` — `(src, scale)` where `src[j]` is the row index feeding column `j` (`0` for a null column) and `scale[j]` its value (`1` for a null column). Returns `nothing` when `M` is not a scaled partial permutation.

- [ ] **Step 1: Write the failing test**

Create `test/test_batched_mass_solve.jl`:

```julia
"""
The batched mass solve.

`M_min` is rank-deficient in every tau/BC formulation, so the per-mode path
solves `M x = b` with a sparse least-squares (`SPQRSolver`). Measured, though,
`M_min` is a 0/1 PARTIAL PERMUTATION — at most one nonzero per row and per
column, every value exactly 1, the tau columns empty. For such a matrix the
minimum-norm least-squares solution is `x = M' b` for ANY `b`, which is one
kernel rather than `nmodes` sparse solves.

This file pins the structural check that decides whether that shortcut is
legal, because applying it to a genuine mass matrix would be silently wrong
rather than an error.
"""

using Test
using Tarang
using SparseArrays
using LinearAlgebra

@testset "mass_selection_plan" begin
    @testset "accepts a 0/1 partial permutation" begin
        # cols 1,2 empty; col 3 -> row 1, col 4 -> row 4, col 5 -> row 5
        M = sparse([1, 4, 5], [3, 4, 5], ComplexF64[1, 1, 1], 5, 5)
        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing
        src, scale = plan
        @test src == [0, 0, 1, 4, 5]
        @test scale == ComplexF64[1, 1, 1, 1, 1]
    end

    @testset "accepts a SCALED partial permutation" begin
        M = sparse([1, 4], [3, 4], ComplexF64[2.0, 0.5], 4, 4)
        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing
        src, scale = plan
        @test src == [0, 0, 1, 4]
        @test scale[3] == 2.0
        @test scale[4] == 0.5
    end

    @testset "rejects two nonzeros in one column" begin
        M = sparse([1, 2], [1, 1], ComplexF64[1, 1], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects two nonzeros in one row" begin
        M = sparse([1, 1], [1, 2], ComplexF64[1, 1], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects a genuine mass matrix" begin
        # tridiagonal — the shape a non-identity basis normalisation produces
        M = spdiagm(-1 => ComplexF64[1, 1], 0 => ComplexF64[4, 4, 4],
                    1 => ComplexF64[1, 1])
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "an explicitly stored zero is not a nonzero" begin
        # A stored zero must not be treated as a mapping, or the plan would
        # divide by it.
        M = SparseMatrixCSC(3, 3, [1, 2, 2, 2], [1], ComplexF64[0.0])
        @test Tarang.mass_selection_plan(M) === nothing
    end
end
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Test; include("test/test_batched_mass_solve.jl")'
```
Expected: FAIL with `UndefVarError: mass_selection_plan not defined`.

- [ ] **Step 3: Implement**

Append to `src/core/subsystems/mode_batch.jl`:

```julia
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
minimum-norm least-squares solution is `x = M' b` for ANY `b` — matching `b`
exactly on the image rows and taking the free null columns to zero. That is one
kernel instead of `nmodes` sparse solves.

The shortcut is only valid for this structure. Applying it to a genuine mass
matrix would produce a plausible wrong answer with no error, so the structure is
VERIFIED here rather than assumed, and callers fall back to the per-mode solver
on `nothing`.

An explicitly stored zero is treated as disqualifying rather than as a mapping:
`scale[j]` is divided by, and a stored zero also means the true structure is not
what the pattern suggests.
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
        iszero(vals[k]) && return nothing   # stored zero: not a real mapping
        i = rows[k]
        row_used[i] && return nothing       # two entries in a row
        row_used[i] = true
        src[j] = i
        scale[j] = vals[k]
    end
    return (src, scale)
end
```

- [ ] **Step 4: Run the test and confirm it passes**

Same command as Step 2. Expected: PASS, six testsets.

- [ ] **Step 5: Register and check the inventory guard**

Add to `test/file_lists.jl`'s `TEST_FILES`:

```julia
    "test_batched_mass_solve.jl",      # M_min is a 0/1 partial permutation so its pseudo-inverse is its transpose — but applying that to a genuine mass matrix is silently wrong, so the structure is verified, not assumed
```

Then:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'include("test/file_lists.jl"); using Test; include("test/test_test_inventory.jl")'
```
Expected: PASS.

- [ ] **Step 6: Commit**

---

### Task 2: The kernel, and equivalence to SPQR

**Files:**
- Modify: `src/core/subsystems/mode_batch_kernels.jl`
- Test: `test/test_batched_mass_solve.jl` (append)

**Interfaces:**
- Consumes: `mass_selection_plan` from Task 1.
- Produces: `batched_mass_apply!(X, B, src, scale) -> X` — `X[j, m] = src[j] == 0 ? 0 : B[src[j], m] / scale[j]`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_batched_mass_solve.jl`:

```julia
@testset "batched_mass_apply! equals the per-mode least-squares solve" begin
    using Tarang: MatSolvers

    @testset "against SPQR on the real M_min" begin
        solver = _mass_channel_solver()
        sps = collect(solver.problem.compiled.subproblems)
        live = [sp for sp in sps if sp.M_min !== nothing]
        M = live[1].M_min
        n = size(M, 1)
        nmodes = length(live)

        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing          # the premise of this whole task
        src, scale = plan

        B = ComplexF64[(0.37i + 0.11j) + (0.5i - 0.2j) * im
                       for i in 1:n, j in 1:nmodes]
        X = zeros(ComplexF64, n, nmodes)
        Tarang.batched_mass_apply!(X, B, src, scale)

        # The reference: exactly what the per-mode path does today.
        ref_solver = MatSolvers.solver_instance(MatSolvers.SPQRSolver, M)
        for m in 1:nmodes
            expected = zeros(ComplexF64, n)
            MatSolvers.solve!(expected, ref_solver, B[:, m])
            @test isapprox(X[:, m], expected; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "a scaled permutation divides, not just permutes" begin
        # Pins that `scale` is actually applied: with all-ones data a kernel
        # that ignored `scale` would agree by accident.
        M = sparse([1, 3], [2, 3], ComplexF64[4.0, 0.25], 3, 3)
        src, scale = Tarang.mass_selection_plan(M)
        B = reshape(ComplexF64[8, 12, 16], 3, 1)
        X = zeros(ComplexF64, 3, 1)
        Tarang.batched_mass_apply!(X, B, src, scale)
        @test X[1, 1] == 0            # null column
        @test X[2, 1] == 8 / 4.0      # draws row 1, divided by 4
        @test X[3, 1] == 16 / 0.25    # draws row 3, divided by 0.25
    end

    @testset "null columns are written, not left stale" begin
        M = sparse([2], [2], ComplexF64[1.0], 2, 2)
        src, scale = Tarang.mass_selection_plan(M)
        X = fill(ComplexF64(9999), 2, 1)     # reused buffer
        B = reshape(ComplexF64[5, 7], 2, 1)
        Tarang.batched_mass_apply!(X, B, src, scale)
        @test X[1, 1] == 0                   # must be cleared, not stale
        @test X[2, 1] == 7
    end
end
```

Add the shared helper near the top of the file (a local copy — do not `include`
another test file, the inventory guard treats every `test_*.jl` as an entry
point):

```julia
function _mass_channel_solver(; nx=16, nz=8, dt=1e-3)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
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
    add_bc!(problem, "b(z=0) = 1")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt)
    step!(solver)
    return solver
end
```

- [ ] **Step 2: Run and confirm failure**

Expected: `UndefVarError: batched_mass_apply! not defined`.

- [ ] **Step 3: Implement**

Append to `src/core/subsystems/mode_batch_kernels.jl`:

```julia
# One thread per (column, mode). Each output element is written exactly once
# and never read back — including the null columns, which must be WRITTEN to
# zero rather than skipped, because the destination buffer is reused across
# stages and steps.
@kernel function _batched_mass_apply_kernel!(X, @Const(B), @Const(src),
                                             @Const(scale))
    j, m = @index(Global, NTuple)
    @inbounds begin
        s = src[j]
        X[j, m] = s == 0 ? zero(ComplexF64) : B[s, m] / scale[j]
    end
end

"""
    batched_mass_apply!(X, B, src, scale) -> X

Apply the pseudo-inverse of a scaled partial-permutation mass matrix to every
mode at once: `X[j, m] = B[src[j], m] / scale[j]`, and zero where `src[j] == 0`.

This is the minimum-norm least-squares solution of `M x = b` when `M` is a
scaled partial permutation — see `mass_selection_plan`, which verifies that
structure and produces `src`/`scale`. Callers must not reach here without a
plan from it.
"""
function batched_mass_apply!(X::AbstractMatrix{ComplexF64},
                             B::AbstractMatrix{ComplexF64},
                             src::AbstractVector{Int},
                             scale::AbstractVector{ComplexF64})
    backend = get_backend(X)
    _batched_mass_apply_kernel!(backend)(X, B, src, scale; ndrange=size(X))
    KernelAbstractions.synchronize(backend)
    return X
end
```

- [ ] **Step 4: Run and confirm it passes**

- [ ] **Step 5: Commit**

---

### Task 3: Wire it into the stage loop

**Files:**
- Modify: `src/core/subsystems/mode_batch.jl` (carry `src`/`scale` on `ModeBatch`)
- Modify: `src/core/timesteppers/step_subproblem_rk_batched.jl`
- Modify: `test/test_mode_batch_parity.jl`

**Interfaces:**
- Consumes: `mass_selection_plan`, `batched_mass_apply!`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_mode_batch_parity.jl` two testsets:

```julia
@testset "batched mass path engages and matches the per-mode trajectory" begin
    # Inhomogeneous BC so the ALG_F/BC path is live — a homogeneous problem
    # makes it numerically inert and this comparison vacuous.
    ref_solver, ref_b = _parity_channel_solver(; batched_modes=false, bc_low="1")
    bat_solver, bat_b = _parity_channel_solver(; batched_modes=true,  bc_low="1")

    _seed_parity_ic!(ref_b); _seed_parity_ic!(bat_b)
    for _ in 1:5
        step!(ref_solver); step!(bat_solver)
    end

    batches = Tarang.active_mode_batches(bat_solver)
    @test !isempty(batches)
    @test all(b -> b.mass_src !== nothing, batches)   # the fast path is ON

    ensure_layout!(ref_b, :g); ensure_layout!(bat_b, :g)
    r = Array(get_grid_data(ref_b)); v = Array(get_grid_data(bat_b))
    scale = maximum(abs, r)
    @test scale > 1e-8
    @test maximum(abs, v .- r) / scale < 1e-12
end

@testset "a non-selection mass matrix falls back to the per-mode solve" begin
    solver, _ = _parity_channel_solver(; batched_modes=true, bc_low="1")
    step!(solver)
    batch = first(Tarang.active_mode_batches(solver))
    @test batch.mass_src !== nothing          # baseline: it engaged

    # Perturb M_min so it is no longer a partial permutation, rebuild, and
    # confirm the plan declines rather than silently applying the transpose.
    sps = collect(solver.problem.compiled.subproblems)
    live = [sp for sp in sps if sp.M_min !== nothing]
    M = live[1].M_min
    bad = copy(M)
    j = findfirst(c -> length(nzrange(M, c)) == 1, 1:size(M, 2))
    i = setdiff(1:size(M, 1), rowvals(M))[1]
    bad[i, j] = 1.0 + 0.0im                    # two nonzeros in column j
    @test Tarang.mass_selection_plan(bad) === nothing
end
```

- [ ] **Step 2: Run and confirm failure**

Expected: `mass_src` is not a field of `ModeBatch`.

- [ ] **Step 3: Implement**

Add `mass_src::Union{Nothing, AbstractVector{Int}}` and
`mass_scale::Union{Nothing, AbstractVector{ComplexF64}}` to `ModeBatch`. In
`build_mode_batch`, call `mass_selection_plan(sp1.M_min)`; when it returns a
plan, **verify it holds for EVERY mode in the batch** (the measured problem has
`M_min` identical across modes, but that is a measurement, not a guarantee) and
upload `src`/`scale` via `_batch_similar`; otherwise store `nothing` in both.

In `step_subproblem_rk_batched.jl` there is exactly ONE place to change, not
two. `_batch_mass_solve!` (`:621`) is the per-mode helper; it is called from both
the `a_ii == 0` stage (`:862`) and the final update (`:921`). Branch INSIDE it on
`batch.mass_src !== nothing`: take the `batched_mass_apply!` fast path when a plan
exists, otherwise run the existing per-mode loop unchanged. Do not delete the
per-mode path and do not touch either call site.

Preserve the `ws.mass_ok::Vector{Bool}` contract exactly. Callers read it at
`:923` (`all(ws.mass_ok)`) and `:948` (`ws.mass_ok[m] || continue`), so the fast
path must `fill!(ws.mass_ok, true)` — the plan cannot fail per-mode, since its
validity was established at build time for every mode in the batch. Returning a
stale or partially-written `mass_ok` would silently skip scatters for some modes,
which is the exact silent-wrong shape this plan is guarding against.

- [ ] **Step 4: Run the parity file and confirm it passes**

- [ ] **Step 5: Mutation-verify both new guards**

The session's dominant defect has been tests that pass with the thing they guard
deleted. Confirm, and paste the output:

1. Make `mass_selection_plan` return `nothing` unconditionally — the
   `mass_src !== nothing` assertion must fail.
2. Make `batched_mass_apply!` ignore `scale` (drop the division) — the scaled
   testset in `test_batched_mass_solve.jl` must fail.
3. Make it skip null columns instead of zeroing them — the stale-buffer testset
   must fail.

Revert each mutation and confirm `git diff` is empty before committing.

- [ ] **Step 6: Measure the result**

Report the per-step operation count before and after, the same way the 8,127 ->
839 figure was derived, so the claimed improvement is measured rather than
projected.

- [ ] **Step 7: Full verification**

```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  -e 'using Pkg; Pkg.test()' 2>&1 | tee /tmp/tarang-mass-suite.log
```
Expected: `Testing Tarang tests passed`. Read the JET count out of the same log
(bound is `<= 975`; the branch sat at 954 before this change).

Then the MPI suite:
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/run_mpi_ci.jl
```
Expected: 49 passed, 0 failed at 4 ranks.

- [ ] **Step 8: Commit**

---

## Self-Review

**Spec coverage.** The structural check is Task 1; the kernel and its equivalence
to the existing SPQR path are Task 2; wiring, parity, fallback, mutation
verification, and measurement are Task 3.

**Type consistency.** `mass_selection_plan(M) -> Union{Nothing, Tuple{Vector{Int}, Vector{ComplexF64}}}`
is used identically in Tasks 1, 2, and 3. `batched_mass_apply!(X, B, src, scale)`
keeps one signature across Tasks 2 and 3. `ModeBatch.mass_src` / `.mass_scale`
are named the same in Task 3's tests and implementation.

**The risk this plan is built around.** The shortcut is only valid for a scaled
partial permutation. Applying it to a genuine mass matrix yields a plausible
wrong answer with no error — this repository's dominant historical bug class. The
structure is therefore verified per batch AND per mode, the fallback is retained
rather than replaced, and Task 3 Step 5 mutation-verifies that the guard actually
fires.
