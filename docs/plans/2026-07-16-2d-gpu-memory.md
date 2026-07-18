# 2D GPU Memory Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate full-field device allocations from the warmed 2D pure-Fourier GPU RK timestep while preserving the stochastic forcing, Poisson, velocity, and Runge--Kutta numerics.

**Architecture:** Keep the generic explicit field RK driver, but bind its host containers to preallocated timestepper workspace fields: one reusable stage state and one retained derivative state per RK stage. Alternate two output states through history rather than deep-copying every step. For the 2D vorticity formulation, recognize the Fourier Poisson and `u = skew(grad(ψ))` constraints and write their coefficients directly into the existing target fields; retain the interpreted allocator-heavy path for unsupported constraints.

**Tech Stack:** Julia 1.10--1.12, Tarang scalar/vector fields, CUDA.jl extension arrays, Fourier spectral operators, Julia `Test` and `@allocated`.

### Task 1: Reproduce stage aliasing and warmed allocation cost

**Files:**
- Create: `test/test_gpu_field_rk_allocations.jl`
- Modify: `test/runtests.jl` only if this repository enumerates the file explicitly

**Step 1: Write the failing tests**

- Advance `dt(q) = q` through `_step_explicit_rk_gpu!` and compare against the RK222 stability polynomial, proving every stage derivative remains independent.
- Warm the exact 2D vorticity/streamfunction/velocity field path and assert a bounded steady-state allocation count.
- Record the identity of cached stage arrays across two steps.

**Step 2: Run the test to verify RED**

Run:

```bash
julia --project=. test/test_gpu_field_rk_allocations.jl
```

Expected: RK222 value mismatch and/or allocation budget failure on the current implementation.

### Task 2: Cache RK stage storage and recycle output states

**Files:**
- Modify: `src/core/timesteppers/state.jl`
- Modify: `src/core/timesteppers/state_utils.jl`
- Modify: `src/core/timesteppers/step_rk.jl`
- Test: `test/test_gpu_field_rk_allocations.jl`

**Step 1:** Add helpers that return host vectors referencing existing workspace field sets and retained per-stage derivative sets.

**Step 2:** Copy each reusable lazy-RHS result into its retained stage buffer before the next RHS evaluation.

**Step 3:** Alternate a cached output field-set with the old one-entry history state instead of calling `copy_state` every step.

**Step 4: Run the focused test to verify GREEN**

Run the Task 1 command. Expected: RK oracle passes, cache identities remain stable, and allocation count drops.

### Task 3: Refresh 2D Fourier constraints in place

**Files:**
- Modify: `src/core/timesteppers/state_utils.jl`
- Modify: `src/core/solvers/lazy_rhs.jl`
- Test: `test/test_gpu_field_rk_allocations.jl`
- Test: `test/test_fourier_algebraic_constraints.jl`

**Step 1: Write the failing allocation regression**

Warm `_refresh_algebraic_state!` for `Δψ + τ - ζ = 0`, `u - skew(grad(ψ)) = 0` and assert that another refresh does not allocate spatial-sized fields.

**Step 2:** Add a direct Fourier Poisson solve that writes `ψ̂` from `ζ̂` into the existing target coefficient buffer.

**Step 3:** Add a direct 2D skew-gradient solve that copies `ψ̂` into the existing velocity buffers and applies cached device-aware Fourier derivative multipliers.

**Step 4:** Retain the general expression evaluator as fallback for every nonmatching constraint.

**Step 5: Verify GREEN**

Run both test files listed above. Expected: analytic constraint tests pass and warmed allocation falls below the new budget.

### Task 4: Reduce persistent GPU workspace and test CUDA when available

**Files:**
- Modify: `src/core/timesteppers/state.jl`
- Modify: `test/test_stochastic_forcing.jl`

**Step 1: Write the failing workspace-count assertion**

Assert RK111/RK222/RK443 allocate only `stages + 1` workspace field sets for the generic field path.

**Step 2:** Change `_workspace_count` to the minimum needed by the reusable stage-state plus retained derivative buffers.

**Step 3:** Extend the conditional CUDA IVP test to warm the solver, check cached device-array identities, and measure CUDA allocations if the installed CUDA.jl exposes a reliable allocation macro.

### Task 5: Verification and delivery

Run:

```bash
julia --project=. test/test_gpu_field_rk_allocations.jl
julia --project=. test/test_stochastic_forcing.jl
julia --project=. test/test_lazy_rhs_fourier.jl
julia --project=. test/test_fourier_algebraic_constraints.jl
julia --project=. test/test_compatibility.jl
git diff --check
```

Then run the exact 8×8 forced 2D example smoke, commit the changes, push `fix/stochastic-forcing-correctness`, and verify that PR #57 points to the new head commit.
