# Boundary Condition Fixes Implementation Plan

**Goal:** Repair all eight boundary-condition defects reproduced in the CPU/GPU audit, preserving the public APIs and existing working-tree changes.

**Architecture:** Prepare boundary values through shared context/evaluation helpers for initial-value and steady solvers. Bind the normal coordinate to its boundary position and resolve registered parameters without replacing tangential coordinates. Preserve scalar component selection in both shape inference and matrix assembly. Allocate GPU field/solve buffers using the field backend. Periodic markers remain metadata because Fourier bases enforce periodicity.

**Tech Stack:** Julia, Test, FFTW, MPI, GPUArrays/JLArrays, CUDA extension.

The user authorized all eight fixes. Work continues in the current tree with no commits, resets, or worktree changes. The prior audit supplies numerical reproductions; each implementation group first adds and runs focused regressions. Matrix-changing Robin coefficients remain unsupported as already documented; the repair concerns supported RHS values. GPU emulation checks array/device behavior with scalar indexing disabled and explicit host FFT/LU stand-ins; native CUDA hardware is unavailable.

Alternatives considered: rejecting all nonconstant boundary values would remove advertised behavior; separate steady and transient evaluators would duplicate the bug-prone context logic. Shared preparation keeps supported expressions consistent. CPU staging at GPU scatter would conceal the device-buffer defect, so GPU solves instead allocate device storage at their source.

## 1. Boundary context and refresh

Files: `src/core/boundary_conditions.jl`, `src/core/boundary_conditions/types.jl` if needed, `src/core/solvers/solver_types.jl`, `src/core/problems/problem_types.jl`; new focused boundary-value regression file(s).

- [x] Add/run failing numerical tests for spatial Dirichlet/Neumann values in both linear and nonlinear BVPs (audit error 1.0 / 0.761594).
- [x] Add/run tests comparing raw and structured moving Robin conditions (raw currently returns 0 instead of 0.06).
- [x] Add/run parameterized time/space expressions, including a named boundary position and a normal-coordinate reference on a shifted interval and 3D plane.
- [x] Register Robin strings through the existing parser and preserve their equation mapping.
- [x] Supply registered parameters to boundary evaluation, with time/actual coordinates taking precedence; bind the normal coordinate to the resolved position without mutating shared coordinate grids.
- [x] Reuse boundary preparation for steady constructors so spatial RHS operators are materialized before solving; preserve callback failure propagation and stage-time refresh.
- [x] Run targeted existing BC, callback, BVP, and timestepper tests; rerun original audit probes.

## 2. Stress-free component assembly

Files: `src/core/subsystems/subproblem_types.jl`, `src/core/problems/problem_matrices/problem_matrices_expr_analysis.jl`, `src/core/operators/matrices/matrices_expression.jl`, `src/core/operators/matrices/matrices_subproblem_operators.jl` and related component matrix helpers as required; new component boundary regression file.

- [x] Add/run a failing public `StressFreeBC` solve with square-system and value assertions; use nontrivial manufactured vector fields to test normal/tangential behavior.
- [x] Make generic `Component` output sizing scalar where appropriate in global and subproblem assembly.
- [x] Preserve component selection through interpolation/derivatives and matrix construction; a row-count-only repair is insufficient.
- [x] Check generated 2D/3D conditions, no-slip controls, and existing operator/matrix tests.

## 3. GPU buffer ownership

Files: `src/core/solvers/solver_stepping.jl`, `src/core/field/field_types.jl`, `src/core/field/field_data/field_data_copy_alloc.jl` and `src/core/field/field_layout/field_layout_vectorized.jl` as needed; new JLArray boundary regression and CUDA coverage in `test/test_gpu_fc_2d_complete.jl`.

- [x] Add/run failing JLArray tests for public GPU BVP solve/scatter and `unit_vector_fields` construction with scalar indexing disabled.
- [x] Allocate steady-solve RHS, algebraic RHS, and solution buffers on the correct backend. The nonlinear GPU path already raises an explicit unsupported-operation error; preserve that guard.
- [x] Ensure empty-basis field storage agrees with device allocations, preserving scalar/tau/unit-vector behavior and existing CPU storage contracts.
- [x] Run device boundary gather/override and moving-wall RK checks; add equivalent CUDA-gated tests for actual hardware CI.

## 4. Periodic markers and integration

Files: `src/core/problems/problem_parsing.jl`, new `test/test_periodic_bc_marker.jl`, `test/file_lists.jl`, relevant boundary/GPU documentation.

- [x] Add/run a failing test for `add_bc!(problem, periodic_bc(...))`; verify no equation is emitted and the Fourier solve remains correct.
- [x] Skip equation conversion for periodic metadata without duplicating constraints. Regression: 8/8 assertions pass.
- [x] Register all new tests and update documentation to remove obsolete limitations addressed by these fixes.
- [x] Independently review all diffs and ownership/error handling.
- [x] Rerun the valid original audit cases, CPU boundary and GPU-emulation tests, and two-/four-rank MPI checks including 3D decomposition.
- [x] Run the full default package suite with `--threads=4,1`; resolve regressions and inspect allocation/static-analysis checks.
- [x] Run `git diff --check` and record final results and CUDA hardware limits.

Julia: `/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.`. GPU extras environment: `/private/tmp/tarang-timestep-audit/gpu-env`. Baseline and logs: `/private/tmp/tarang-boundary-fixes/`. Prior reproductions: `/private/tmp/tarang-boundary-audit/`.

## Verification record

- Periodic regression: 8/8 after reproducing the original MethodError.
- CPU boundary integration: 991/991 assertions across 14 files; expanded Robin parameter tests subsequently pass (45 context assertions).
- Final GPU emulation: 131/131 assertions with scalar indexing disabled, including spatial and moving boundaries and nonzero stress-free vector modes. Native CUDA test entry point correctly skips because CUDA is not functional here.
- MPI: all five two-/four-rank invocations pass, including four new spatial steady-wall cases on each rank count and the existing 3D four-rank decomposition test.
- One original combined audit probe incorrectly replaced a required Chebyshev wall with a Fourier periodic marker. Its valid wall cases are rerun unchanged in a temporary copy; periodicity is covered by the dedicated complete Fourier solve and original marker-only probe.
- Native CUDA execution remains unverified. The pre-existing CUDA extension method-overwrite precompilation warning is unchanged.
- Original audit rechecks pass: spatial steady errors at most 2.78e-16, moving Robin residual error 2.78e-17, parameterized moving Dirichlet exact in the probe, shifted 3D normal-coordinate wall error 1.78e-15, square stress-free systems, and periodic marker registration without equations.
- Full-suite static analysis passes at 928 reports; the ceiling remains 975.
- Full default package suite: 181 files, 11,962 passing assertions, 19 pre-existing skips/broken tests; process exits successfully. Allocation checks and existing static-analysis budgets pass without loosening limits. Log: `/private/tmp/tarang-boundary-fixes/full-suite.log`.
- Final `git diff --check` passes. All eight audited defects are repaired; no commits were created.
