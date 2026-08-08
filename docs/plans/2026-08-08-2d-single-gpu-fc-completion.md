# Complete 2D Single-GPU Fourier--Chebyshev Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete and value-test the two-dimensional single-GPU Fourier--Chebyshev transform, nonlinear, derivative, and IVP paths, including scaled Chebyshev grids and both basis orders.

**Architecture:** Keep `forward_layout` as the canonical operation source and add a pure stage-shape walk for the CUDA Fourier-first execution order. Make the mixed CUDA driver use shape-keyed persistent scratch so a Chebyshev DCT-I can run at scaled grid length before forward truncation, or after backward zero-padding, without CPU staging. Verify the complete path against CPU and provide a strict CUDA cluster runner.

**Tech Stack:** Julia 1.10--1.12, Tarang spectral fields and subproblem RK, CUDA.jl/CUFFT, KernelAbstractions, CUDA sparse solvers, Julia `Test`.

### Task 1: Make mixed-transform intermediate shapes explicit

**Files:**
- Create: `test/test_transform_stage_shapes.jl`
- Modify: `test/file_lists.jl`
- Modify: `src/core/transforms/transform_layout.jl`

**Step 1: Write the failing shape-contract tests**

Build `ChebyshevT(9) x RealFourier(8)` bases with a scaled grid shape `(14, 12)` and assert:

```julia
ops, coeff_shape, _ = Tarang.forward_layout(bases, (14, 12), Float64)
order = [2, 1] # Fourier first, then Chebyshev
stages = Tarang.transform_stage_shapes(ops, (14, 12), order)

@test coeff_shape == (9, 7)
@test stages == [(14, 12), (14, 7), (9, 7)]
@test reverse(stages) == [(9, 7), (14, 7), (14, 12)]
```

Add cases for `RealFourier x ChebyshevT`, `ComplexFourier x ChebyshevT`, unscaled axes, and invalid/duplicate transform orders. Register the test in `TEST_FILES`.

**Step 2: Run the test to verify RED**

Run:

```bash
julia --project=. test/test_transform_stage_shapes.jl
```

Expected: `UndefVarError: transform_stage_shapes not defined`.

**Step 3: Implement the pure stage-shape walk**

Add:

```julia
function transform_stage_shapes(ops, input_shape::Tuple, order)
    length(ops) == length(input_shape) || throw(DimensionMismatch(...))
    sort(collect(order)) == collect(eachindex(ops)) || throw(ArgumentError(...))
    shape = collect(input_shape)
    stages = Tuple[Tuple(shape)]
    for axis in order
        shape[axis] = ops[axis].out_len
        push!(stages, Tuple(shape))
    end
    return stages
end
```

The helper is deliberately backend-independent and allocation is acceptable because it runs only while constructing a transform plan.

**Step 4: Run GREEN and regression tests**

Run:

```bash
julia --project=. test/test_transform_stage_shapes.jl
julia --project=. test/test_transforms.jl
```

Expected: both files pass.

**Step 5: Commit**

```bash
git add src/core/transforms/transform_layout.jl test/test_transform_stage_shapes.jl test/file_lists.jl
git commit -m "refactor: expose mixed transform stage shapes"
```

### Task 2: Add device-only Chebyshev axis resizing

**Files:**
- Create: `test/test_gpu_fc_2d_complete.jl`
- Modify: `ext/cuda/mixed_transforms.jl`
- Modify: `test/file_lists.jl`

**Step 1: Write conditional CUDA RED tests for resize primitives**

Under `CUDA.functional()` with `CUDA.allowscalar(false)`, create a `(6, 9)` `CuArray`, truncate axis 2 into `(6, 5)`, and zero-pad it back into `(6, 9)`. Compare with host slicing and assert the padded tail is zero. Repeat along axis 1 and with `ComplexF64`.

Call extension-local helpers through:

```julia
cuda_ext = Base.get_extension(Tarang, :TarangCUDAExt)
cuda_ext._copy_axis_prefix!(small, large, 2)
cuda_ext._zero_pad_axis_prefix!(restored, small, 2)
```

If CUDA is unavailable, retain one explicit `@test_skip`; the strict cluster runner added later will turn absence of CUDA into an error.

**Step 2: Run RED on a CUDA host**

Run:

```bash
TARANG_REQUIRE_CUDA=true julia --project=. test/test_gpu_fc_2d_complete.jl
```

Expected: missing resize-helper methods.

**Step 3: Implement generic CuArray resize helpers**

Add helpers that validate equal dimensionality and equal non-resized axes, then use `ntuple` indices and views:

```julia
function _copy_axis_prefix!(dst, src, axis)
    n = size(dst, axis)
    idx = ntuple(d -> d == axis ? (1:n) : Colon(), ndims(src))
    @views dst .= src[idx...]
    return dst
end

function _zero_pad_axis_prefix!(dst, src, axis)
    fill!(dst, zero(eltype(dst)))
    n = size(src, axis)
    idx = ntuple(d -> d == axis ? (1:n) : Colon(), ndims(dst))
    @views dst[idx...] .= src
    return dst
end
```

Reject truncation in the wrong direction rather than silently clipping.

**Step 4: Run GREEN where CUDA is available and parse/skip locally**

Run the Task 2 test command on CUDA. Locally run it without `TARANG_REQUIRE_CUDA` and expect only the documented skip.

**Step 5: Commit**

```bash
git add ext/cuda/mixed_transforms.jl test/test_gpu_fc_2d_complete.jl test/file_lists.jl
git commit -m "feat: resize Chebyshev axes on device"
```

### Task 3: Execute scaled FC transforms with shape-keyed scratch

**Files:**
- Modify: `ext/cuda/mixed_transforms.jl`
- Modify: `ext/cuda/transforms.jl`
- Test: `test/test_gpu_fc_2d_complete.jl`
- Test: `test/test_gpu_transform_correctness.jl`

**Step 1: Add failing CPU-versus-GPU transform tests**

Create matched CPU/GPU fields and compare forward coefficients, backward grids, and coefficient preservation for:

- scaled `RealFourier x ChebyshevT`;
- scaled `ChebyshevT x RealFourier`;
- scaled `ComplexFourier x ChebyshevT` with complex grid data;
- independently scaled Fourier and Chebyshev axes; and
- the existing unscaled FC layout.

Use `preset_scales!` before filling the grid so the test controls the exact scaled shape. Assert the GPU coefficient shape equals `Tarang.forward_layout(...)[2]` and round-trip errors stay below `1e-10` for `Float64`/`ComplexF64`.

**Step 2: Run RED on CUDA**

Expected: the current dispatcher raises `GPUTransformUnsupported` for the scaled Chebyshev axis.

**Step 3: Extend the plan and scratch cache**

Store the shared `AxisOp` vector and `transform_stage_shapes` result in `GPUMixedTransformPlan`. Change `get_gpu_mixed_transform_scratch` to accept an exact intermediate shape and key cached scratch by:

```julia
(device_id, plan.grid_shape, plan.coeff_shape, stage_shape, ComplexType)
```

Each shape entry retains two complex ping-pong arrays and three real arrays. Preserve the documented serial-task cache contract.

**Step 4: Implement forward truncation and backward zero-padding**

For each forward Chebyshev stage:

1. run DCT-I in scratch at the current full shape;
2. if `AxisOp.out_len` is shorter, copy the leading coefficients into scratch at the stage output shape;
3. continue with that smaller array.

For backward Chebyshev stages, walk `reverse(plan.stage_shapes)`:

1. zero-pad the leading stored coefficients into scratch at the preceding forward shape;
2. run inverse DCT-I at the padded grid length;
3. continue with the expanded array.

Remove `_scaled_chebyshev_axis` refusal only from the mixed FC branches. Keep pure-Chebyshev scaled fields rejected until they use the same driver contract deliberately. Validate actual input/output shapes against the plan before mutation.

**Step 5: Run focused GREEN tests**

Run on CUDA:

```bash
TARANG_REQUIRE_CUDA=true julia --project=. test/test_gpu_fc_2d_complete.jl
julia --project=. test/test_gpu_transform_correctness.jl
```

Expected: all scaled/unscaled comparisons pass, no scalar indexing, and stored coefficients remain unchanged by inverse transforms.

**Step 6: Commit**

```bash
git add ext/cuda/mixed_transforms.jl ext/cuda/transforms.jl test/test_gpu_fc_2d_complete.jl test/test_gpu_transform_correctness.jl
git commit -m "feat: complete scaled 2D FC GPU transforms"
```

### Task 4: Validate derivatives and nonlinear FC dealiasing

**Files:**
- Modify: `test/test_gpu_fc_2d_complete.jl`
- Modify only if a value mismatch is demonstrated: `src/core/nonlinear/nonlinear_padding.jl`
- Modify only if a value mismatch is demonstrated: `ext/cuda/utils.jl`

**Step 1: Write derivative RED tests**

For both FC basis orders, sample a smooth function such as
`sin(2x) * z^2 * (1-z)` and compare CPU/GPU `Differentiate` results along the Fourier and Chebyshev axes. Include a complex-Fourier field. Check both grid and coefficient layouts.

**Step 2: Write nonlinear padded-product RED tests**

Use a Fourier basis with `dealias=3/2` and a Chebyshev basis with no scaling. Compare `evaluate_transform_multiply(u, v, evaluator)` on CPU and GPU for both basis orders. Include:

- smooth resolved modes;
- an even-size Fourier Nyquist-sensitive mode;
- `result_layout=:g` and `:c`; and
- preservation of both input fields.

**Step 3: Run RED/GREEN diagnosis on CUDA**

Run the focused CUDA file. If existing kernels pass, make no production change. If they fail, use systematic debugging to isolate transform, padding, Nyquist folding, or derivative convention before editing.

**Step 4: Commit tests and any demonstrated fix**

```bash
git add test/test_gpu_fc_2d_complete.jl src/core/nonlinear/nonlinear_padding.jl ext/cuda/utils.jl
git commit -m "test: validate 2D FC GPU derivatives and dealiasing"
```

Omit unchanged production files from `git add`.

### Task 5: Add a nonlinear wall-bounded GPU IVP oracle

**Files:**
- Modify: `test/test_gpu_fc_2d_complete.jl`

**Step 1: Write the end-to-end CUDA test**

Adapt `test/test_mpi_cheb_fourier_ivp_nonlinear.jl` to matched serial CPU and single-GPU solvers:

```julia
∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)
b(z=0) = 0
b(z=1) = 0
```

Use `Nx=8`, `Nz=10`, `dealias=3/2`, RK222, homogeneous Dirichlet conditions, and the same smooth initial condition on both devices. Run at least five steps.

Assert:

- the GPU path selects `CuSparseLU` and builds nonempty coupled subproblems;
- field, tau, RK, sparse matrix, RHS, solution, and scratch arrays stay on CUDA;
- CPU/GPU coefficients and grids agree within `rtol=5e-7`;
- wall values remain within the solver tolerance;
- all values remain finite; and
- a warmed additional step performs no full-field device allocation according to the CUDA allocation counter already used by `test_gpu_transform_correctness.jl`.

**Step 2: Run RED/GREEN on CUDA**

Run:

```bash
TARANG_REQUIRE_CUDA=true julia --project=. test/test_gpu_fc_2d_complete.jl
```

If the test exposes a defect, follow systematic debugging and add the smallest production fix with a focused regression assertion before rerunning the end-to-end case.

**Step 3: Commit**

```bash
git add test/test_gpu_fc_2d_complete.jl <only-demonstrated-production-files>
git commit -m "test: cover nonlinear 2D FC GPU IVP"
```

### Task 6: Add a strict cluster entry point and documentation

**Files:**
- Create: `test/run_gpu_fc_2d.jl`
- Modify: `test/file_lists.jl`
- Modify: `docs/src/pages/gpu_computing.md`
- Modify: `docs/src/pages/testing.md`

**Step 1: Write the strict runner**

The runner must:

```julia
using CUDA
CUDA.functional() || error("2D FC validation requires a functional CUDA device")
CUDA.allowscalar(false)
CUDA.versioninfo()
ENV["TARANG_REQUIRE_CUDA"] = "true"
include("test_gpu_fc_2d_complete.jl")
```

Use `joinpath(@__DIR__, ...)` for the include. A failed assertion or missing CUDA must produce a nonzero exit status.

**Step 2: Register and document**

Add `test_gpu_fc_2d_complete.jl` to `GPU_TEST_FILES`. Document supported single-GPU FC configurations, the remaining multi-GPU/concurrent-transform limits, and the cluster command:

```bash
julia --project=. test/run_gpu_fc_2d.jl
```

**Step 3: Verify local failure semantics without CUDA**

Run the strict runner locally and expect a clear nonzero CUDA-required error. Run the test file directly and expect a clean skip so ordinary CPU CI remains usable.

**Step 4: Commit**

```bash
git add test/run_gpu_fc_2d.jl test/file_lists.jl docs/src/pages/gpu_computing.md docs/src/pages/testing.md
git commit -m "test: add strict 2D FC GPU cluster runner"
```

### Task 7: Final verification and review

**Files:**
- Review all files changed by Tasks 1--6

**Step 1: Run the local regression gates**

```bash
julia --project=. test/test_transform_stage_shapes.jl
julia --project=. test/test_transforms.jl
julia --project=. test/test_transform_inplace.jl
julia --project=. test/test_gpu_solver_cpu.jl
julia --project=. test/test_gpu_implicit_guard.jl
julia --project=. test/test_nl_product_ownership.jl
julia --project=. test/test_gpu_fc_2d_complete.jl
git diff --check
```

Expected: every CPU test passes; only the direct CUDA-focused test reports its documented skip on this host.

**Step 2: Run the cluster gate on NVIDIA hardware**

```bash
julia --project=. -e 'using Pkg; Pkg.add("CUDA"); Pkg.instantiate()'
julia --project=. test/run_gpu_fc_2d.jl
```

Expected: no skips, no scalar-indexing fallback, and all value/allocation assertions pass.

**Step 3: Review the final diff**

Use `superpowers:requesting-code-review`, address every actionable finding with tests, then use `superpowers:verification-before-completion` before claiming completion.

**Step 4: Commit final review-only adjustments**

If review required changes, commit them separately as:

```bash
git commit -m "fix: address 2D FC GPU review findings"
```

Do not amend earlier commits; preserve the TDD history for cluster diagnosis.
