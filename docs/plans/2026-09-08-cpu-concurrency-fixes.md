# CPU Concurrency Fixes Implementation Plan

**Goal:** Make concurrent independent CPU Fourier derivative evaluations and shared-factor matrix solves numerically correct across Julia thread pools.

**Architecture:** Give each active operation exclusive ownership of reusable scratch storage until it finishes. Protect only workspace checkout/return and cache growth with locks; release storage in `finally` blocks. Preserve concurrent FFT/matrix computation and remove dependence on global thread IDs. Existing solver and field APIs remain unchanged.

**Tech Stack:** Julia, FFTW, LinearAlgebra/SuiteSparse, MPI, Test.

The user authorized fixes for the two reproduced concurrency defects. Work proceeds in the current tree, preserving previous changes; no commits, resets, or worktree operations are part of this task. A whole-operation lock would fix corruption but serialize callers; thread-indexed arrays would still depend on pool numbering and task scheduling. Exclusive workspace checkout avoids both limitations.

## 1. Fourier derivative scratch

Files: `src/core/operators/derivatives/derivatives_fourier.jl`, new `test/test_cpu_fourier_concurrency.jl`. The adjacent borrowed-result pool in `derivatives_eval.jl` and its regression `test/test_cpu_derivative_result_concurrency.jl` also need task isolation so same-basis public callers remain independent throughout evaluation.

- [x] Add regression coverage for concurrent public derivative calls on independent same-shaped fields and cold/warm cache behavior. Observe numerical failure before source changes.
- [x] Replace shared mutable scratch reuse with exclusive checkout and exception-safe return. Keep FFT plans/buffers reusable and preserve parallel computation.
- [x] Isolate borrowed derivative result pools between tasks while preserving within-task rotation and public result ownership; synchronize shared-basis derivative multiplier caching.
- [x] Run the regression with four threads and the prior audit reproduction; verify sequential/concurrent accuracy and input preservation.

## 2. Matrix-solver workspaces

Files: `src/tools/matsolvers.jl`, `test/test_solvers.jl` if needed, new `test/test_cpu_matsolver_concurrency.jl`.

- [x] Add regression coverage for shared BlockDiagonalSolver/SPQRSolver factors with independent RHS/output arrays, including `--threads=4,1`. Observe the existing BoundsError before changes.
- [x] Make workspaces exclusive to active solves, independent of global thread IDs; return them on errors. Preserve existing matrix/RHS behavior and reuse in sequential solves.
- [x] Verify numerical results on `--threads=1,0`, `4,0`, and `4,1`, including oversubscribed/yielding tasks where meaningful.

## 3. Integrate and verify

- [x] Register all three tests in `test/file_lists.jl` and set default CPU CI to `JULIA_NUM_THREADS='2,1'` so both thread pools are exercised on every supported Julia version.
- [x] Review both diffs and exception/resource handling; run existing solver and derivative tests and relevant allocation checks.
- [x] Rerun original audit probes, four-thread FFT/CPU kernel checks, and two-/four-rank hybrid MPI checks.
- [x] Run the full package suite with four default threads and an interactive pool; inspect all failures and document verification results.
- [x] Run `git diff --check` and report the completed fixes with any remaining limitations.

Julia executable: `/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.`. Logs and baseline snapshots: `/private/tmp/tarang-cpu-parallel-fixes/`. Original probes: `/private/tmp/tarang-cpu-parallel-audit/`.

## Verification results

The regressions were observed failing before their respective fixes:

- Concurrent Fourier scratch: both cold and warm numerical checks failed.
- Borrowed derivative results: 45 failed assertions, including overwritten live results, cross-distributor reuse, and global retention.
- Matrix workspaces: 24 failed assertions with `--threads=4,1`, including the worker-thread BoundsError and corruption when RHS reads yield.

After the fixes:

- Fourier regression passes with `--threads=1,0`, `2,1`, `4,0`, and `4,1`. The original audit now reports zero incorrect/nonfinite values across all six 240-call groups; maximum error is `1.853e-11`, matching sequential accuracy, and inputs remain unchanged.
- Result-pool regression passes 131 assertions with `--threads=1,0`, `2,1`, and `4,1`; the existing ownership tests add 63 passing assertions.
- Matrix regression passes 80 assertions with `--threads=1,0` and 164 with each of `4,0` and `4,1`. The original public solve probe succeeds on every worker with maximum relative error below `1.6e-15`.
- Existing solver tests pass 167 assertions, including in-place allocation limits. Existing tensor/fractional-Laplacian tests pass 65 assertions.
- Large CPU kernels, mode batches, threaded BLAS blocks, and focused existing CPU checks pass 456 assertions with `--threads=4,1`.
- A fresh process verifies four FFTW threads and passes 186 transform/complex Fourier-Chebyshev/mode-batch assertions.
- All ten hybrid MPI invocations pass: reproducible random fill, reductions, transposable parity, padded 3D dealiasing, and nonlinear Chebyshev-Fourier stepping, each with two and four ranks, `--threads=2,1`, and two FFTW threads. Nonlinear squared norm is `33.39084552380445` on two ranks and `33.390845523804444` on four, matching the serial reference to roundoff.
- Independent reviews found no actionable defects in workspace ownership, concurrent pool growth, exception cleanup, or task lifetime handling.
- Full `Pkg.test(; julia_args=["--startup-file=no", "--threads=4,1"])` passes all 177 registered test files: 11,677 assertions pass, 19 remain skipped/broken, and none fail. JET reports remain at 927, below the unchanged ceiling of 975. The pre-existing CUDA extension method-overwrite precompilation warning still appears; it does not prevent the suite from completing successfully.
- Final `git diff --check` passes. No commits were created.

These fixes cover independent eager CPU derivative evaluations and shared `BlockDiagonalSolver`/`SPQRSolver` solves. Concurrent mutation of the same field or factor, lazy RHS evaluation sharing a basis, and GPU task concurrency are outside this verification. CPU timestepper subproblem loops remain sequential within each MPI rank; the optimization guide states that scope explicitly.
