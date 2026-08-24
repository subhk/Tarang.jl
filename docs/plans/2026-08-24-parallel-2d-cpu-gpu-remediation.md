# Parallel 2D CPU/GPU Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate stale-layout data loss and broken distributed-transform wrappers, while keeping Buildkite and the GitHub trigger focused on the available single-GPU box.

**Architecture:** Keep `TransposableField` as the owner of transpose buffers and make its distributed transform API strict about the authoritative source layout. Route optional plan dictionaries through the existing keyword interface, rebuild a workspace when a transform is used with another field, and use MPI.jl's supported `MPI.has_cuda()` capability query. Buildkite will run `GPU_TEST_FILES` on one CUDA device; distributed CUDA/MPI and NCCL files remain registered and parse-checked without being scheduled on the current agent.

**Tech Stack:** Julia, FFTW, MPI.jl, CUDA.jl extension, Buildkite YAML, GitHub Actions.

### Task 1: Lock down source-layout ownership

**Files:**
- Modify: `test/test_transposable_field.jl`
- Modify: `src/core/transpose/transpose_transforms.jl`

**Step 1: Write the failing test**

After a successful round trip, assert that backward transform from `:g` throws and preserves the grid buffer. Transform forward again, then assert that forward transform from `:c` throws and preserves the coefficient buffer.

**Step 2: Run it to verify failure**

Run the file with two MPI ranks. Expected: both `@test_throws` assertions fail because the current implementation performs another transform.

**Step 3: Implement the minimal guard**

Add a helper that throws `ArgumentError` unless `field.current_layout` equals the required source layout, and call it before fetching raw buffers in both distributed transform directions.

**Step 4: Verify green**

Run the same two-rank test and expect all assertions to pass.

### Task 2: Repair the exported wrapper and workspace binding

**Files:**
- Modify: `test/test_transposable_field.jl`
- Modify: `src/core/gpu_distributed.jl`
- Modify: `src/core/transpose/transpose_transforms.jl`

**Step 1: Write the failing tests**

Construct a `DistributedGPUTransform` around a CPU-backed ComplexFourier field, call its exported forward/backward wrappers, and assert a round trip. Construct a second compatible field and assert `setup_transposable_workspace!` returns a workspace bound to that second field.

**Step 2: Verify red**

Expected failures: `MethodError` from the positional plan argument and `workspace.field !== second_field`.

**Step 3: Implement the minimal fixes**

Call distributed transforms with `; plans=transform.plans`, select the supplied dictionary as the plan cache inside each transform, and recreate the `TransposableField` workspace whenever its bound field is not identical to the requested field.

**Step 4: Verify green**

Run the one-, two-, and four-rank TransposableField tests.

### Task 3: Use MPI.jl's CUDA capability query

**Files:**
- Modify: `test/test_transposable_field.jl`
- Modify: `src/core/gpu_distributed.jl`

**Step 1: Write the failing test**

Temporarily clear `TARANG_CUDA_AWARE_MPI`, set `JULIA_MPI_HAS_CUDA=true`, and assert `check_cuda_aware_mpi()` returns true. Also assert Tarang's explicit `0` override wins.

**Step 2: Verify red**

Expected: the positive MPI.jl capability assertion fails because Tarang currently checks only legacy environment variables.

**Step 3: Implement the minimal capability check**

After Tarang's explicit override, call `MPI.has_cuda()` when available, catching probe errors before checking legacy vendor variables.

**Step 4: Verify green**

Run the focused test and restore both environment variables in `finally`.

### Task 4: Enforce single-GPU CI and retain the GitHub bridge

**Files:**
- Modify: `test/test_gpu_test_files_reachable.jl`
- Modify: `.buildkite/pipeline.yml`
- Modify: `.github/workflows/gpu-buildkite.yml`

**Step 1: Write failing static assertions**

Assert that uncommented Buildkite configuration runs `test/run_gpu_ci.jl`, identifies the job as single-GPU, does not run `test_distributed_gpu_transpose.jl`, and that the GitHub workflow defaults to the actual `subhajit-kar/tarang-dot-jl` pipeline.

**Step 2: Verify red**

Expected during remediation: the active two-rank CUDA/MPI command violates the single-GPU policy; the GitHub workflow initially also used the wrong organization fallback.

**Step 3: Implement CI configuration**

Keep only the existing `run_gpu_ci.jl` matrix on the CUDA queue and remove the active two-rank CUDA/MPI step. Continue to parse-check the distributed GPU files. Update the GitHub workflow fallback organization and retain its explicit missing-secret failure.

**Step 4: Verify green**

Run the static reachability test, inspect both YAML files, and run `git diff --check`.

### Task 5: Full verification

**Files:**
- Verify all modified files.

**Step 1:** Run `test/test_transposable_field.jl` with one, two, and four ranks.

**Step 2:** Run the coefficient-level 4-rank reference comparison for `(4,1)`, `(1,4)`, and `(2,2)` meshes.

**Step 3:** Run GPU test-file parse/reachability checks without requiring a physical GPU.

**Step 4:** Inspect the final diff and confirm unrelated untracked design documents are unchanged.
