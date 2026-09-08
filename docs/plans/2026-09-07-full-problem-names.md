# Full Problem Names Implementation Plan

> **For Claude:** Use `superpowers:executing-plans` to implement this plan.

**Goal:** Use descriptive problem type names consistently throughout Tarang.

**Architecture:** Rename the concrete types to `InitialValueProblem`,
`LinearBoundaryValueProblem`, `NonlinearBoundaryValueProblem`, and
`EigenvalueProblem`. Remove the old abbreviated bindings and exports as explicitly
requested by the user. Export the full names from Tarang and Tarang.Problems,
and use them throughout the public API and all current examples. Solver behavior
and keyword handling stay the same; existing callers must update their names.

**Tech Stack:** Julia and the existing problem/solver/documentation tests.

Keep the current checkout and all existing uncommitted work. File baselines for
this rename are saved under `/private/tmp/tarang-problem-names-baseline`.

## 1. Establish the API regression

- [x] Exercise canonical names, facade exports, and absence of old bindings
  in `test/test_problems.jl` and observe failure before implementation.

## 2. Rename consistently

- [x] Rename concrete types, builders, signatures, and internal references in
  `src/core/problems/problem_types.jl` and the rest of `src/`.
- [x] Export only the canonical names from
  `src/api/problems.jl`, `src/api/public/quick_start.jl`, and the core exports.
- [x] Update tests, examples, scripts, README, and current documentation.
  Preserve historical plans and mathematical abbreviations that are not APIs.
- [x] Document the required name migration and check for unintended substitutions.

## 3. Verify

- [x] Run problem construction, solver, facade, public API, and docs-code tests.
- [x] Review the rename against saved baselines and run `git diff --check`.
- [x] Run the package suite and representative MPI problem construction/solve.
- [x] Record outcomes and leave changes uncommitted.

Independent review found no actionable rename issues or stale links to renamed
documentation headings. Two-rank initial-value and Chebyshev/Fourier boundary-value solves passed.
The full package suite passed: 10,936 assertions across 169 files, 19
reported broken/skipped, zero failures or errors. The constructor test file
passed all 141 assertions, including absence checks for the removed names.
`git diff --check` passed; dependency manifests are unchanged and edits remain
uncommitted. The pre-existing CUDA extension method-overwrite precompilation
error was emitted again; it did not prevent the package tests from passing.
CUDA hardware was not tested.
