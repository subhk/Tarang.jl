# CUDA Extension Load Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Tarang's CUDA extension load and precompile cleanly on Julia 1.10--1.12 so the existing single-GPU 3D Fourier and Fourier--Chebyshev IVP tests can execute on NVIDIA hardware.

**Architecture:** Keep one CUDA extension and preserve both single- and multi-GPU functionality. Core owns public fallback methods; CUDA adds more-specific methods or hook implementations without overwriting core definitions. Order extension includes by type dependency, and rely on each launch's exact `ndrange` instead of invalid early-return bounds guards inside KernelAbstractions kernels.

**Tech Stack:** Julia 1.10--1.12, CUDA.jl 5, KernelAbstractions 0.9, Tarang package extensions, GitHub Actions, JuliaGPU Buildkite.

### Task 1: Pin extension completion with a no-driver regression

**Files:**
- Modify: `test/test_gpu_transform_correctness.jl`

**Step 1: Add the failing test**

When CUDA.jl imports, assert `Base.get_extension(Tarang, :TarangCUDAExt)` is non-`nothing` and that the final included transpose operation `GPU_UNPACK_3D_OP` is defined.

**Step 2: Run the test to verify RED**

Run the test in a temporary environment containing CUDA.jl on a host without an NVIDIA driver.

Expected: FAIL because the extension only partially loads.

### Task 2: Remove package-extension method overwrites

**Files:**
- Modify: `src/core/architectures.jl`
- Modify: `src/core/transforms/transform_gpu.jl`
- Modify: `ext/TarangCUDAExt.jl`
- Modify: `ext/cuda/architecture.jl`

**Step 1: Keep public constructors and fallbacks in core**

Route `GPU()` and `has_cuda()` through more-specific CUDA hooks. Keep core transform fallbacks less specific than CUDA's `ScalarField` methods.

**Step 2: Make CUDA methods additive**

Import `ensure_device!` before extending it, specialize `device`/`array_type` on `GPU{CuDevice}`, and remove exact generic replacements.

**Step 3: Re-run the extension-load regression**

Expected: no method-overwrite precompile errors.

### Task 3: Correct extension dependency order

**Files:**
- Modify: `ext/TarangCUDAExt.jl`

**Step 1: Order type providers before consumers**

Include DCT, pencil, NCCL, and distributed-DCT definitions before `transforms.jl`; include `batched_fft.jl` after `transforms.jl` provides `GPUFFTPlan`.

**Step 2: Re-run the extension-load regression**

Expected: no `UndefVarError` during extension precompilation.

### Task 4: Make KernelAbstractions kernels valid

**Files:**
- Modify: `ext/cuda/nccl_transpose.jl`
- Modify: `ext/cuda/dct_distributed.jl`
- Modify: `ext/cuda/transpose_kernels.jl`

**Step 1: Remove redundant early exits from every `@kernel`**

Remove `if idx > total; return; end` guards without changing index or data-movement formulas. Every caller supplies an `ndrange` equal to the logical element count, and KernelAbstractions masks execution to that range.

**Step 2: Re-run the extension-load regression**

Expected: PASS with the CUDA extension completely loaded; the hardware assertions skip because `CUDA.functional()` is false.

### Task 5: Add active CI coverage for extension loading

**Files:**
- Modify: `.github/workflows/CI.yml`

**Step 1: Add one no-driver CUDA extension job**

On Ubuntu and Julia 1.12, instantiate Tarang, add CUDA.jl to the job environment, and run `test/test_gpu_transform_correctness.jl`. Require the extension-load test to pass while live device tests skip.

**Step 2: Validate workflow syntax and local command parity**

Run the equivalent temporary-environment command locally and parse the workflow.

### Task 6: Regression verification and delivery

**Files:**
- Verify all modified files

**Step 1: Run focused CPU and CUDA-enabled suites**

Run `test/test_gpu_transform_correctness.jl` with CUDA installed, plus stochastic forcing, separable forcing, solver, subproblem RK, and field-RK allocation tests in the normal project.

**Step 2: Run static checks**

Run `git diff --check`, inspect the final diff, and confirm the worktree contains only scoped changes.

**Step 3: Commit, push, and update PR #58**

Commit the repair, push `fix/stochastic-forcing-correctness`, and document that live numerical/device-allocation assertions still require the configured NVIDIA runner.
