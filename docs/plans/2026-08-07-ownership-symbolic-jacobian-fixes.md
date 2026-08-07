# Ownership and Symbolic Jacobian Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate four silent-aliasing and wrong-Jacobian defects in public differentiation, Frechet differentiation, coefficient-space Jacobian assembly, and `FieldPool` lifecycle tracking.

**Architecture:** Public derivative evaluation will own rotating-pool results by default while internal immediate consumers may explicitly borrow. Frechet differentiation will construct a directional derivative expression using the supplied perturbation, so differential operators act on the perturbation instead of disappearing. Jacobian coefficient blocks will be assembled by applying the real spectral operation to both quadratures of every coefficient basis vector, producing a doubled-real matrix that preserves RealFourier conjugate coupling. `FieldPool` will separately track whether each owned field is actively checked out.

**Tech Stack:** Julia, Tarang spectral fields/operators, SparseArrays, Test.

### Task 1: Public derivative ownership

**Files:**
- Modify: `test/test_deriv_pool_ownership.jl`
- Modify: `src/core/operators/derivatives/derivatives_eval.jl`

**Step 1: Write the failing test**

Add a test that retains the result of both `evaluate(Differentiate(...))` and the default `evaluate_differentiate(...)`, advances the rotating pool beyond its capacity, and verifies both retained values are unchanged. Keep a separate assertion that `own=false` still reuses at most `_DERIV_RESULT_POOL_SIZE` objects.

**Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_deriv_pool_ownership.jl`

Expected: retained derivative values change after pool wraparound.

**Step 3: Write minimal implementation**

Add `own::Bool=true` to `evaluate_differentiate`; copy the rotating-pool result with `_own_borrowed_field` when `own` is true. Mark recursive/internal callers that immediately copy or explicitly take ownership with `own=false`.

**Step 4: Run test to verify it passes**

Run the same test and expect all ownership assertions to pass.

### Task 2: Directional Frechet differentiation

**Files:**
- Modify: `test/test_symbolic_diff.jl`
- Modify: `src/core/operators/symbolic_diff.jl`

**Step 1: Write the failing test**

Add numerical tests for `Differentiate(u)`, `Laplacian(u)`, `FractionalLaplacian(u)`, and `Laplacian(u*u)`, comparing the returned directional expression against the operator applied to the supplied perturbation.

**Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_symbolic_diff.jl`

Expected: the linear differential cases return zero.

**Step 3: Write minimal implementation**

Implement directional differentiation rules that substitute the perturbation at matching fields and apply arithmetic, chain, and differential-operator rules recursively. Route `frechet_differential` through these rules. Make two-argument `sym_diff` reject operator-valued derivatives it cannot represent rather than silently returning an incorrect scalar expression.

**Step 4: Run test to verify it passes**

Run the same test and expect the analytical directional comparisons to pass.

### Task 3: Correct coefficient-space Jacobian blocks

**Files:**
- Modify: `test/test_symbolic_diff.jl`
- Modify: `src/core/operators/symbolic_diff.jl`

**Step 1: Write the failing test**

For a constant physical field, verify its multiplication block acts as the constant on every valid spectral degree of freedom. For `lap(u) - u*u`, verify `J*v` against a directly evaluated Frechet action, including cosine and sine perturbations whose variable-coefficient coupling crosses through zero wavenumber.

**Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_symbolic_diff.jl`

Expected: non-DC modes are zero because the current implementation diagonally inserts coefficient values; a cosine-only basis probe also gives the wrong sign for a coupled sine mode.

**Step 3: Write minimal implementation**

Build field-valued multiplication blocks column-by-column using the actual spectral transform/product path. Probe both the real and imaginary coefficient quadratures and assemble a doubled-real sparse matrix acting on `[real(x); imag(x)]`, so RFFT half-spectrum maps need not pretend to be complex-linear. Assemble each problem Jacobian block from the directional Frechet expression so linear differential operators and nonlinear coefficient actions are both included, and teach the global Newton fallback to solve the doubled-real system.

**Step 4: Run test to verify it passes**

Run the same test and expect numerical `J*v` agreement for both coefficient quadratures and the recovered Newton correction.

### Task 4: Reject duplicate `FieldPool` returns

**Files:**
- Modify: `test/test_field_pool.jl`
- Modify: `src/core/field_pool.jl`

**Step 1: Write the failing test**

Return one checked-out field twice, require the second return to throw `ArgumentError`, then check out two fields and require distinct identities.

**Step 2: Run test to verify it fails**

Run: `julia --project=. test/test_field_pool.jl`

Expected: the second return currently succeeds and the two checkouts alias.

**Step 3: Write minimal implementation**

Track active checkout state in a weak-key dictionary. Set it on allocation/reuse, clear it on return, initialize prewarmed fields as inactive, and reject inactive returns before modifying pool state.

**Step 4: Run test to verify it passes**

Run the same test and expect all pool lifecycle assertions to pass.

### Task 5: Combined verification

**Files:**
- Verify all modified source and test files.

**Step 1:** Run the focused ownership, symbolic differentiation, pool, solver, matrix, and nonlinear-product tests.

**Step 2:** Run a clean source load with compiled modules disabled.

**Step 3:** Run `git diff --check` and inspect the complete diff.

**Step 4:** Run every available registered CPU test, documenting unavailable Aqua/JET, CUDA, and multi-rank MPI coverage separately.
