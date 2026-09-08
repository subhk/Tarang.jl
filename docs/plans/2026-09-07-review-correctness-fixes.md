# Review Correctness Fixes Implementation Plan

> **For Claude:** Use `superpowers:executing-plans` to implement these tasks.

**Goal:** Preserve public field edits, select stable spectral diffusion timesteps,
and preserve RK accuracy for time-dependent deterministic forcing.

**Architecture:** Keep buffer arrays and their authoritative layout together in
field storage. Bind problem handles to the active RHS stage without overwriting
retained history. Derive the Fourier diffusion bound from global wavenumbers and
refresh deterministic forcing at RHS evaluation times; stochastic realizations
remain fixed for each step.

**Tech Stack:** Julia, FFTW, MPI, existing Tarang regression suite.

Work in the reviewed checkout to preserve existing uncommitted changes. Do not
commit unrelated edits. The three fixes are authorized by the review follow-up.

## 1. Shared field storage

- [x] Append regressions to `test/test_state_arith_layout.jl`: public grid and
  coefficient edits after stepping, array replacement, copy independence,
  stage evaluation preserving history, and multistep history independence.
- [x] Run the file and verify the new assertions fail for stale layouts/data.
- [x] Move `current_layout` into both concrete storage types in
  `src/core/field/field_types.jl` and `src/core/transposable_field.jl`; preserve
  the existing public property and three-argument storage constructors.
- [x] Update the ScalarField property shim and the direct parser layout read.
- [x] Update solver synchronization to share complete storage and bind stage
  handles without copying into the previous state.
- [x] Update structural assertions in `test/test_hasfield_ratchet.jl` and run
  field, arithmetic, solver, checkpoint, and multistep regression tests.

## 2. Spectral diffusion CFL

- [x] Add and run the failing Fourier mode decay test in
  `test/test_cfl_diffusive.jl`.
- [x] Replace the Fourier finite-difference frequency in
  `src/extras/flow_tools/flow_tools_cfl.jl` with half the spectral Laplacian
  radius times diffusivity; retain the conservative bounded-axis estimate.
- [x] Update analytical CPU/MPI expectations and relevant documentation.
- [x] Run the focused CPU and available MPI CFL tests.

## 3. Deterministic forcing stages

- [x] Add and run failing stage-time/integration tests in
  `test/test_solver_review_regressions.jl`.
- [x] Separate deterministic stage updates from stochastic step updates in
  `src/core/timesteppers/{dispatch,state,state_utils}.jl`.
- [x] Cover lazy, interpreted, and buffered RHS paths and stochastic draw count.
- [x] Run solver and forcing regressions and update relevant documentation.

## 4. Integration verification

- [x] Review only the changes made for these fixes against saved file baselines.
- [x] Run the focused regression files together, then the CPU suite; report any
  environmental or pre-existing failures separately with evidence.
- [x] Run applicable MPI checks if the local MPI runtime is usable.

Validation results:

- Shared-storage tests: 178 passing assertions; broader field/transform/solver/
  checkpoint tests: 443 passing, one optional JLArrays check skipped outside the
  package test environment.
- Combined regression files: 198 passing assertions.
- Focused CFL tests: 115 passing assertions; distributed CFL tests passed with
  two and four ranks.
- Forcing tests: 305 passing assertions; seven CUDA checks skipped.
- Two-rank implicit advection, transposable coefficient parity, and checkpoint
  restart tests passed.
- Independent storage review found no actionable fixed-resolution issues.
  Changing a live solver's resolution is outside this fix: cached matrices and
  histories still assume the construction-time resolution.
- Full `Pkg.test(; allow_reresolve=false)` passed: 10,899 assertions across
  169 test files, 19 reported broken/skipped, zero failures or errors. The
  pre-existing CUDA extension method-overwrite precompilation error is still
  emitted by unchanged `distributor_core.jl` and `ext/cuda/transforms.jl`;
  it did not prevent the package tests from passing. CUDA hardware was not tested.
- `git diff --check` passed. All changes remain uncommitted.

Runtime used for validation (the juliaup launcher has a root-owned config):

```sh
/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using Test, Tarang; include("test/test_state_arith_layout.jl")'
```
