# Webdocs Numerical Content Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the tau-method, boundary-condition, timestepper, and solver-selection documentation agree with the current Tarang.jl implementation.

**Architecture:** Treat source constructors, dispatch tables, stepping kernels, shipped examples, and regression tests as the source of truth. Correct factual claims and examples without changing numerical code or expanding the public API.

**Tech Stack:** Julia, Documenter.jl, Markdown, Tarang.jl timestepper and subproblem implementations.

### Task 1: Correct tau and boundary-condition content

**Files:**
- Modify: `docs/src/pages/tau_method.md`
- Modify: `docs/src/tutorials/boundary_conditions.md`

**Step 1:** Add failing content assertions for the known scalar-constraint miscount, unused first tau fields, undefined unit vectors, and nonexistent automatic missing-tau error.

**Step 2:** Correct the pressure-gauge explanation and scalar constraint/tau-DOF count.

**Step 3:** Replace stale vector-flow examples with the shipped first-order `grad_u`/`grad_T` tau pattern and define every referenced symbol.

**Step 4:** Correct troubleshooting text to describe actual square/rank/factorization failures rather than an unused validator.

### Task 2: Correct timestepper and solver-selection content

**Files:**
- Modify: `docs/src/pages/timesteppers.md`
- Modify: `docs/src/api/timesteppers.md`
- Modify: `docs/src/pages/solvers.md`

**Step 1:** Add failing assertions for RK222's stage count, stale ETD formulas, non-exported public methods, and zero-linear-solve claims.

**Step 2:** Document the implemented RK111/RK222/RK443/RKSMR and multistep behavior, including startup and path limitations.

**Step 3:** Replace ETD formulas with the implemented φ₁/φ₂ variable-step forms.

**Step 4:** Restrict the API guide to exported steppers and replace unsupported CFL/cost/memory claims with implementation-backed guidance.

### Task 3: Verify content and rendered webdocs

**Files:**
- Verify: `docs/src/pages/tau_method.md`
- Verify: `docs/src/tutorials/boundary_conditions.md`
- Verify: `docs/src/pages/timesteppers.md`
- Verify: `docs/src/api/timesteppers.md`
- Verify: `docs/src/pages/solvers.md`

**Step 1:** Run the corrected content assertions and exported-timestepper coverage check.

**Step 2:** Run focused tau/BC/timestepper regression tests.

**Step 3:** Build the complete Documenter site with Tarang loaded and doctests enabled.

**Step 4:** Run `git diff --check`, review the diff, commit, and push the update to PR #34.
