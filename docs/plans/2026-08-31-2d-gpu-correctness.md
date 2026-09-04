# 2D GPU Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the serial 2D GPU path preserve CPU-equivalent values, axes, and layouts across field construction, scaled transforms, deterministic forcing, polynomial derivatives, and DiagonalIMEX step completion.

**Architecture:** Keep the established field and transform dispatch intact. Canonicalize every field to its domain's coordinate-axis order. During an explicit scale change, resize the stored spectral tensor while both the old and new physical grid shapes are known, then let the existing CPU/CUDA inverse plans consume the canonical target layout. Correct the independent state/layout bugs in shared core code instead of adding GPU-only workarounds.

**Tech Stack:** Julia 1.12, Tarang field layouts, FFTW, CUDA.jl/cuFFT extension, GPUArrays/JLArrays, KernelAbstractions, Test stdlib.

**Workspace constraint:** The checkout contains user-owned staged and unstaged work. Preserve it, make only the scoped edits below, and do not commit without an explicit user request.

### Task 1: Canonicalize 2D field basis order

**Files:**
- Modify: `test/test_field_typestability.jl`
- Modify: `test/test_gpu_transform_correctness.jl`
- Modify: `src/core/field/field_types.jl`

**Steps:**

- Construct a field with caller order `ChebyshevT(z) × RealFourier(x)` while the coordinate system order is `(x,z)`. Require `field.bases == field.domain.bases`, matching layout/grid/coefficient shapes, and an analytic forward/backward round trip.
- Observe the pre-fix shape/axis mismatch.
- In both `ScalarField` inner constructors, replace the caller tuple with `domain.bases` before layout lookup, scale initialization, transform ownership, and storage in the field.
- Repeat on CPU and CUDA under scalar indexing prohibition.

### Task 2: Pin coefficient-layout scaling

**Files:**
- Modify: `test/test_cov_field_data_scales.jl`
- Modify: `test/test_gpu_transform_correctness.jl`
- Modify: `src/core/transforms/transform_layout.jl`
- Modify: `src/core/transforms/transform_fourier.jl`
- Modify: `ext/cuda/transforms.jl`

**Step 1: Write failing tests**

- Build a 2D `RealFourier × RealFourier` field at base resolution, transform it to coefficients, call `set_scales!(..., 3/2)`, inverse-transform, and require the full scaled `(Mx, My)` grid plus analytic low-mode values.
- On CUDA, run the same coefficient-current case and the existing `RealFourier × ChebyshevT` case under `CUDA.allowscalar(false)`, checking CPU/GPU parity and preservation of the represented low-mode field.
- Require ambiguous coefficient-current downscaling to throw before scales or buffers mutate.

**Step 2: Verify RED**

Run:

```bash
julia --startup-file=no --project=. test/test_cov_field_data_scales.jl
```

Expected: the 2D inverse returns the old trailing Fourier length instead of the requested scaled length.

**Step 3: Implement the minimum correction**

- Add a shared coefficient-resize helper that receives both physical grid shapes, validates canonical source/target layouts, zero-pads Fourier modes with the correct positive/negative frequency placement, removes an even-grid source Nyquist on upsampling, and applies the product of Fourier grid-size ratios.
- Replace the field's coefficient buffer with the target-layout tensor before changing scale metadata. This makes the existing CPU inverse chain and CUDA pure/mixed inverse plans consume the same canonical shape without backend-specific inference.
- Reject any Fourier downscale that would pass through coefficient remapping (coefficient-current fields and grid-current mixed-basis fields) before a transform, metadata update, or buffer mutation, until multidimensional RFFT Nyquist folding is implemented. Preflight every vector component before changing the first one.

**Step 4: Verify GREEN**

Run the CPU scaling test, CUDA transform suite on hardware, and the CUDA-extension kernel tests on their CPU backend.

### Task 3: Refresh 2D algebraic variables after DiagonalIMEX steps

**Files:**
- Modify: `test/test_diagonal_imex.jl`
- Modify: `test/test_gpu_timesteppers.jl`
- Modify: `src/core/timesteppers/step_diagonal_imex.jl`

**Step 1: Write a failing test**

Advance `q` with each of `DiagonalIMEX_RK222`, `DiagonalIMEX_RK443`, and `DiagonalIMEX_SBDF2` while constraining `lap(psi)-q=0` and `u-skew(grad(psi))=0`. After each of two steps require `psi=-q/2` for the chosen mode and velocity equal to `skew(grad(psi))`, exercising RK recycling plus both SBDF2 branches. Repeat as CPU/GPU parity on CUDA. Add a lightweight no-implicit-term CPU case for the reachable explicit-vector fallback.

**Step 2: Verify RED**

Run:

```bash
julia --startup-file=no --project=. test/test_diagonal_imex.jl
```

Expected: `psi` and `u` remain at their pre-step zero values.

**Step 3: Implement and verify**

Call `_refresh_algebraic_state!` after the final stepped fields are computed and before every serial DiagonalIMEX history insertion. Refresh the state returned by the serial explicit-vector fallback as well. Run the focused CPU and GPU timestepper suites.

### Task 4: Inject and update deterministic forcing in its declared grid layout

**Files:**
- Modify: `test/test_stochastic_forcing.jl`
- Modify: `test/test_gpu_transform_correctness.jl`
- Modify: `src/core/timesteppers/state_utils.jl`
- Modify: `src/core/solvers/lazy_rhs.jl`

**Step 1: Write a failing test**

For `dt(u)=0` on a 2D Fourier field, register nodal forcing `sin(x)cos(y)`, take one RK222 step, and require `u=dt*sin(x)cos(y)`. Also use spatially constant forcing `f(t)=t` from `t=0` and require the RK222 result `dt²/2`, proving the cache is regenerated at stage time. Cover the compiled and interpreted RHS routes and CUDA parity.

**Step 2: Verify RED**

Run the focused stochastic-forcing test. Expected: the first grid rows are treated as Fourier coefficients, producing a nonphysical result.

**Step 3: Implement and verify**

Factor field-aware forcing injection: deterministic forcing adds to an exact-shape `grid_data!(rhs_field)` view, while stochastic forcing retains coefficient-space injection. The once-per-step updater handles stochastic registrations only; regenerate deterministic registrations at both `evaluate_rhs(..., time)` entry points so stage callbacks occur exactly once at stage time. Reset lazy zero fields in the forcing's native layout, keep buffer-release authority honest, and copy interpreted field-valued expressions into the owned RHS buffer before injection so forcing cannot mutate a state/parameter field returned by identity. Use the injection helper from both RHS engines, then run forcing, RK, and GPU transform tests.

### Task 5: Synchronize coefficient-current Chebyshev operands

**Files:**
- Modify: `test/test_derivatives_polynomial.jl`
- Modify: `test/test_gpu_transform_correctness.jl`
- Modify: `src/core/operators/derivatives/derivatives_polynomial.jl`

**Step 1: Write a failing test**

Make a mixed 2D field's grid buffer stale, replace its authoritative coefficients, differentiate along the Chebyshev axis, and require the derivative of the coefficient-current function for both basis orders. Repeat on CUDA.

**Step 2: Verify RED**

Run the polynomial derivative test. Expected: it differentiates the stale grid function.

**Step 3: Implement and verify**

Use `grid_data!(operand)` at the local Chebyshev derivative boundary, matching the Fourier derivative contract, then run polynomial, mixed-transform, and CUDA tests.

### Task 6: Make single-GPU CI non-vacuous

**Files:**
- Modify: `test/run_gpu_ci.jl`

**Steps:**

- Add a driver-level CUDA import and `CUDA.functional()` preflight before spawning any self-guarding test file.
- Keep per-file subprocess isolation, but make a missing driver/device a failing CI condition rather than a zero-test pass.
- Exercise the preflight's failure behavior on this non-CUDA host and syntax/load-check the runner in the CUDA-enabled temporary environment.

### Task 7: Integrated verification and review

**Step 1: Run non-hardware gates**

- Focused scaling, forcing, derivative, DiagonalIMEX, field-layout, and transform tests.
- JLArray device-safety suites with `GPUArrays.allowscalar(false)`.
- CUDA extension DCT, elementwise, transpose, and mode-batch kernels on the KernelAbstractions CPU backend.

**Step 2: Run hardware gates when available**

Run `test/run_gpu_ci.jl` on a functional NVIDIA CUDA host. A non-CUDA host must report this gate as unavailable, not as a pass.

**Step 3: Review**

Run a defect-first independent review, address all actionable findings, run `git diff --check`, and audit the final diff against the pre-existing dirty worktree.
