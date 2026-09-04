# 2D Backend Correctness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct the reproduced 2D MPI and CUDA backend failures without changing unrelated serial behavior or silently accepting unsupported distributed rescaling.

**Architecture:** MPI transform state will be cached per domain and numeric type so fields from multiple domains remain usable without consuming another domain's plan. Coefficient geometry will come from the owning PencilFFT output pencil. Explicit distributed `set_scales!` and `preset_scales!` will reject resolution changes before mutation until scaled distributed plans exist. CUDA transpose helpers will treat empty local partitions as valid no-ops and restore the prior device on every exit.

**Tech Stack:** Julia 1.12, MPI.jl, PencilArrays.jl, PencilFFTs.jl, CUDA.jl extension, KernelAbstractions.jl, Test stdlib.

### Task 1: Add MPI regression tests

**Files:**
- Create: `test/test_mpi_2d_backend_regressions.jl`
- Modify: `test/file_lists.jl`

**Step 1: Write failing tests**

Add independent testsets that assert:

- `local_shape(domain, :c)` equals the logical local size and axes of `get_coeff_data(field)` for an `8×10` RealFourier domain at two and four ranks.
- `SpectralLinearOperator(field, :laplacian)` constructs coefficients with exactly the field's local coefficient shape and correct rank-local wavenumber values.
- A `Float32` MPI field stores `Float32` grid data and `ComplexF32` coefficients and round-trips within Float32 tolerance.
- A `ComplexF64` field on RealFourier bases keeps complex grid storage and a full-size coefficient spectrum, and round-trips complex data.
- Creating and transforming `8×8` and `10×10` fields alternately on one distributor does not invalidate either field.
- `set_scales!(field, 1.5)` throws before changing scales or grid storage in grid and coefficient layouts, including mixed Chebyshev–Fourier.

**Step 2: Verify RED**

Run:

```bash
/usr/bin/env JULIA_BINDIR=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin \
  /Users/subha/.julia/packages/MPI/pvbg6/bin/mpiexecjl --project=. -n 2 \
  /Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia \
  --project=. test/test_mpi_2d_backend_regressions.jl
```

Expected: failures matching the reproduced wrong local shape, Float64 promotion, real storage for complex input, old-plan dimension error, and post-scale metadata/storage mismatch.

### Task 2: Correct coefficient geometry

**Files:**
- Modify: `src/core/domain.jl`
- Modify: `src/core/timesteppers/spectral_operators.jl`

**Step 1: Implement the minimum geometry fix**

For MPI PencilFFT domains, resolve the domain/type-specific bundle and derive local coefficient shape and global ranges from its output pencil through the public logical-order range API. Retain the current layout-derived shape for serial and non-pencil backends. Legacy distributor fields are introspection state only, never a correctness dependency.

**Step 2: Verify GREEN**

Run the new MPI regression file at two and four ranks, plus `test/test_mpi_diagonal_imex_alloc.jl` and `test/test_mpi_configuration_matrix.jl`.

### Task 3: Make MPI transform planning domain- and dtype-aware

**Files:**
- Modify: `src/core/distributor/distributor_core.jl`
- Modify: `src/core/transforms/transform_planning.jl`
- Modify: `src/core/transforms/transform_gpu.jl`
- Modify: `src/core/transforms/transform_fourier.jl`
- Modify: `src/core/field/field_data/field_data_copy_alloc.jl`

**Step 1: Add cached plan state**

Key MPI transform state by domain identity and field dtype. Cache the canonical forward operations, plan, input/output pencils, solve pencil, pencil configuration, and transform list. Activating a cached entry may update the distributor's legacy active fields for compatibility, but field allocation, transforms, derivatives, and mixed solves must consume the selected bundle directly.

**Step 2: Select transforms from the input dtype**

Use RFFT only when the first Fourier transform sees real input. Complex input uses FFT for every RealFourier axis. Construct PencilFFTs with the real component type explicitly:

```julia
fft_plan = PencilFFTs.PencilFFTPlan(pencil, transforms, real_dtype)
```

Allow fields of different concrete dtypes to share a distributor by resolving a separate plan bundle for each dtype.

**Step 3: Activate before allocation and execution**

Before allocating a field or executing a scalar forward/backward transform, resolve the state matching `field.domain` and `field.dtype`. Migrate mixed-solve and other active plan consumers so correctness never depends on whichever bundle was most recently reflected into the distributor.

**Step 4: Verify GREEN**

Run the new regression file at two/four ranks and the existing transform-planning, field-initialization, mixed-basis, diagonal-IMEX, and configuration-matrix MPI files.

### Task 4: Reject unsupported distributed scaling before mutation

**Files:**
- Modify: `src/core/field/field_data/field_data_scales.jl`
- Modify: `docs/src/getting_started/running_with_mpi.md`

**Step 1: Move the distributed resize guard before every mutation**

When `dist.size > 1` and the requested grid shape changes, both `set_scales!` and `preset_scales!` must throw regardless of current layout, basis composition, GPU/CPU architecture, or `use_pencil_arrays`. The error must explain that padded dealiasing uses its separate supported distributed path.

**Step 2: Verify GREEN**

Run the new regression test, `test/test_mpi_decomp_forcing_audit.jl`, all padded-dealiasing MPI files, and serial `test/test_cov_field_data_scales.jl`.

### Task 5: Handle empty CUDA partitions and TensorField transforms

**Files:**
- Modify: `ext/cuda/transpose_kernels.jl`
- Modify: `test/test_gpu_transpose_kernels_cpu.jl`
- Modify: `src/core/field/field_layout/field_layout_access.jl`
- Modify: `test/test_transform_inplace.jl`

**Step 1: Write failing tests**

- Exercise a pure chunk-size helper with `(count, divisor) == (0, 0)` and require a zero chunk rather than `ArgumentError`.
- On CUDA hardware, require zero-length pack/unpack to return without launching and restore the previous CUDA device after an injected validation error.
- Require direct TensorField forward/backward transforms to update every component and round-trip data.

**Step 2: Implement the minimum fixes**

- Return early for empty CUDA pack/unpack arrays.
- Compute validated chunk sizes with `0/0 -> 0`, while retaining errors for nonzero counts with a zero divisor and non-divisible counts.
- Wrap device selection in `try/finally`.
- Add TensorField component-wise direct transform methods matching VectorField.

**Step 3: Verify GREEN**

Run the CUDA extension tests in the temporary CUDA-enabled environment, KA CPU transpose tests, TensorField serial tests, and the four-rank zero-partition CPU oracle.

### Task 6: Review and full verification

**Files:** all changed files above.

**Step 1: Review**

Run a spec-compliance review, then a defect-first code-quality review. Resolve every critical or important finding and re-review.

**Step 2: Full verification**

Run:

- the complete MPI driver at two and four ranks;
- GPU/JLArray 2D domain, device stack, implicit guard, solver, DCT, transpose, mode-batch, and dealias suites;
- relevant serial transform, field, scaling, and configuration tests;
- `git diff --check` and a final diff audit preserving all pre-existing user changes.
