# Timestepper Correctness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the five numerical/runtime defects confirmed by the CPU MPI and GPU reference-array audit.

**Architecture:** Preserve the existing runtime paths. Compose supported diagonal Fourier operators completely and reject unsupported mass/coupling operators before advancing. Give MCNAB2/CNLF2 their own subproblem startup/history flow, accept compatible RHS vector types, and use matrix phi evaluation valid for defective singular operators.

**Tech Stack:** Julia, MPI/PencilArrays, KernelAbstractions, JLArrays, CUDA extension, Test.

The user authorized implementation after reviewing the findings. Work continues in the current tree with its existing changes. Tests and source edits are divided by function ownership; no commit or branch operation is part of this task.

## 1. Regression tests

- [x] Add and run failing tests for nested/cross-field Laplacians in `test/test_diagonal_operator_composition.jl`.
- [x] Add and run failing mass-operator refusal tests for all affected CPU/MPI/GPU-reference paths, including unchanged solution state on refusal.
- [x] Add and run failing Chebyshev MCNAB2/CNLF2 convergence tests and the public MPI interpreted-RHS reproduction.
- [x] Add and run failing ETD nilpotent/Jordan matrix-function and coupled PDE tests.

## 2. Implement fixes

- [x] `step_diagonal_imex.jl`: validate Laplacian operands and compose nested Fourier multipliers. Preserve device array allocation and MPI-local shapes.
- [x] `dispatch.jl` / timestepper helpers: validate parsed mass expressions on paths that assume identity mass, including GPU construction without global matrices. Preserve global/subproblem mass solves.
- [x] `step_global_matrix.jl` / `step_subproblem_multistep.jl`: seed and advance MCNAB2/CNLF2 using their intended coefficients and per-mode history.
- [x] `step_multistep_field.jl`: allow the interpreted RHS vector to copy into the concrete history buffer.
- [x] `phi_functions.jl`: replace the invalid eigenvector fallback with evaluation valid for non-diagonalizable singular matrices.

## 3. Integrate and verify

- [x] Register new test files in `test/file_lists.jl` and document any explicit support limitations in `docs/src/pages/timesteppers.md`.
- [x] Run the new regressions and appropriate existing timestepper tests.
- [x] Run CPU MPI regression checks on two and four ranks, including coupled Fourier/Chebyshev layouts.
- [x] Run the all-method JLArray suite and new GPU-reference regressions; actual CUDA remains unavailable locally.
- [x] Run the full package suite after integration and inspect all failures.
- [x] Review the final diff, record verification results, and report material limitations.

Use `/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.` for CPU tests. The temporary GPU-reference environment is `/private/tmp/tarang-timestep-audit/gpu-env`. Store command logs in `/private/tmp/tarang-timestep-fixes/`. Regression probes from the preceding audit remain in `/private/tmp`.

## Verification results

- CPU focused regressions: mass validation 123/123; diagonal composition 62/62; MCNAB2/CNLF2 subproblems 5/5; matrix phi functions and coupled ETD equations 98/98.
- MPI, two and four ranks: mass validation 126 assertions per rank; diagonal composition 66 assertions per rank; interpreted multistep RHS 6 assertions per rank. The four-rank 3D Chebyshev/Fourier pencil and slab integration checks also pass.
- The ordinary all-method MPI probes retain all 66 numerical convergence records from the audit, with the same 14 expected refusals across the two rank counts.
- GPU reference arrays: diagonal composition 16/16; expanded all-method JLArray suite 220/220, including 126 mass-operator checks.
- Full `Pkg.test(; julia_args=["--startup-file=no", "--threads=4"])`: all 174 default test files completed, with 11,366 passing assertions and 19 skipped/broken cases; no failed assertions or test errors. JET reported 927 findings, below the existing ceiling of 975. Log: `/private/tmp/tarang-timestep-fixes/full-suite-final.log`.
- Final source diff review and `git diff --check` passed. Independent review of the mass-path selection and guard found no additional issues.
- The original coupled PDE `dt(u) - lap(v) = 0`, `dt(v) = 0` now agrees with RK222 and the analytical solution for all three ETD schemes at Fourier modes 1 and 2 (maximum error 1.25e-16).
- CUDA device checks are unavailable on this machine. CUDA regression cases were added for GPU CI; the CUDA file parses and its no-device skip path exits successfully. Local device validation uses JLArrays. The pre-existing CUDA extension method-overwrite precompilation issue remains outside these timestepper fixes.
