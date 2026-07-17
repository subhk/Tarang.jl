# Fourier--Chebyshev GPU Forcing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add separable stochastic forcing for Fourier--Chebyshev domains and a GPU-resident mixed-basis RK222 subproblem path with bounded warmed device allocations.

**Architecture:** A new `SeparableStochasticForcing` generates random Fourier coefficients and forms their outer product with a one-time-normalized Chebyshev profile. Mixed GPU IVPs automatically select `CuSparseLU` only when the matrix solver was left at its default, while the CUDA sparse wrapper caches RHS, solution, and scratch buffers for in-place solves.

**Tech Stack:** Julia 1.10--1.12, Tarang spectral fields and subproblems, FFTW/CUDA transforms, CUDA.CUSOLVER sparse direct solvers, Julia `Test`.

### Task 1: Constructor, profile conversion, and validation

**Files:**
- Create: `test/test_separable_stochastic_forcing.jl`
- Modify: `test/file_lists.jl`
- Modify: `src/core/stochastic_forcing.jl`

**Step 1: Write the failing constructor tests**

Create a 1D `ChebyshevT` basis and assert that:

```julia
forcing = SeparableStochasticForcing(
    fourier_size=(8,), chebyshev_basis=zb,
    chebyshev_profile=z -> z * (1 - z),
    domain_size=(2pi,), energy_injection_rate=0.2,
    k_forcing=2.0, dk_forcing=0.5, dt=1e-2,
    architecture=CPU(), rng=MersenneTwister(42),
)
@test forcing isa StochasticForcingType
@test size(forcing.cached_forcing) == (8, 10)
@test forcing.injection_metric === :direct
@test_throws ArgumentError SeparableStochasticForcing(...; injection_metric=:vorticity_kinetic)
@test_throws ArgumentError SeparableStochasticForcing(...; chebyshev_profile=zeros(10))
@test_throws DimensionMismatch SeparableStochasticForcing(...; chebyshev_profile=ones(9))
```

Register the new file in `TEST_FILES`.

**Step 2: Run the test to verify RED**

Run: `julia --project=. test/test_separable_stochastic_forcing.jl`

Expected: `UndefVarError: SeparableStochasticForcing not defined`.

**Step 3: Implement the minimal type and constructor**

Add `SeparableStochasticForcing <: StochasticForcingType` with persistent fields for the Fourier spectrum, phases, Fourier realization, normalized Chebyshev coefficients, mixed forcing, previous solution, RNG, timing, and architecture.

For a function profile:

```julia
zref = _native_grid(chebyshev_basis, 1.0)
lo, hi = chebyshev_basis.meta.bounds
z = @. (hi - lo) / 2 * zref + (hi + lo) / 2
values = T[chebyshev_profile(zi) for zi in z]
transform = ChebyshevTransform(chebyshev_basis)
setup_chebyshev_cpu_transform!(transform, length(values), length(values), 1)
coeffs = vec(_chebyshev_forward(values, transform))
```

For coefficient input, transform it backward once to obtain grid values. Normalize both representations by
`sqrt(sum(weights .* abs2.(values)) / (hi - lo))`, using `get_integration_weights`.

Reject non-`ChebyshevT` bases, non-`:direct` metrics, non-finite data, zero norm, and length mismatches. Allocate all persistent arrays on the requested architecture.

**Step 4: Run the test to verify GREEN**

Run the Task 1 command. Expected: constructor and validation tests pass.

**Step 5: Commit**

```bash
git add src/core/stochastic_forcing.jl test/file_lists.jl test/test_separable_stochastic_forcing.jl
git commit -m "feat: add separable mixed-basis forcing"
```

### Task 2: Allocation-free separable forcing generation

**Files:**
- Modify: `src/core/stochastic_forcing.jl`
- Test: `test/test_separable_stochastic_forcing.jl`

**Step 1: Write failing generation tests**

Verify that one generated realization equals the explicit reshaped product,
the Chebyshev coefficient ratio is identical at every forced Fourier mode,
the Fourier zero mode is zero, Hermitian symmetry holds along the Fourier
axis, the result is cached for repeated `(time, substep)` calls, and a new time
changes it. Assert a warmed CPU call stays under a small fixed allocation budget.

**Step 2: Run RED**

Run the focused file. Expected: missing `generate_forcing!` method.

**Step 3: Implement generation and lifecycle methods**

Implement `generate_forcing!`, `_generate_separable_forcing!`, `set_dt!`,
`reset_forcing!`, `get_forcing_spectrum`, `get_cached_forcing`,
`mean_energy_injection_rate`, and `energy_injection_rate`.

Use only persistent arrays in the hot path:

```julia
_fill_random_phases!(forcing.architecture, forcing.random_phases, forcing.rng)
@. forcing.fourier_realization = forcing.forcing_spectrum *
    exp(im * forcing.random_phases) / sqrt(forcing.dt)
_enforce_hermitian_symmetry!(forcing.fourier_realization, forcing.architecture)
forcing.cached_forcing .= reshape(forcing.fourier_realization, fourier_shape) .*
                          reshape(forcing.chebyshev_profile, profile_shape)
```

Add `Base.getproperty` compatibility for `forcing_rate`, `spectrum`,
`is_stochastic`, and `is_gpu`.

**Step 4: Run GREEN and commit**

Run the focused test, then commit as `feat: generate separable forcing in place`.

### Task 3: Registered mixed-field forcing integration

**Files:**
- Modify: `src/core/stochastic_forcing.jl`
- Modify: `src/core/timesteppers/state_utils.jl` only if generic dispatch is required
- Test: `test/test_separable_stochastic_forcing.jl`

**Step 1: Write the failing integration test**

Build a serial `RealFourier(x) x ChebyshevT(z)` scalar field, register the
separable forcing on `dt(q) = 0`, update registered forcing, evaluate the
compiled RHS, and assert that the coefficient RHS equals the real-Fourier view
of the mixed forcing with no Chebyshev truncation or reinterpretation.

**Step 2: Run RED**

Expected: `_matched_forcing_view` has no separable-forcing method.

**Step 3: Add shape/view dispatch**

Factor the existing real-FFT-compatible shape matching into a shared helper and
add serial-array and `PencilArray` methods for `SeparableStochasticForcing`.
Only Fourier dimensions may be half-spectrum truncated; the Chebyshev dimension
must match exactly or return `nothing`/raise at registration.

**Step 4: Run GREEN and commit**

Run the focused forcing tests and `test/test_stochastic_forcing.jl`. Commit as
`feat: register mixed-basis stochastic forcing`.

### Task 4: GPU-aware default mixed subproblem solver

**Files:**
- Modify: `src/core/solvers/solver_types.jl`
- Test: `test/test_solvers.jl`
- Test: `test/test_gpu_transform_correctness.jl`

**Step 1: Write failing selection tests**

Add a pure helper test showing that default/`:auto` selects `:sparse` for CPU
and pure Fourier GPU states, but `:cuda_sparse` for GPU states containing a
Jacobi/Chebyshev basis. Explicit `:cuda_sparse` remains unchanged; explicit
`:sparse` on a coupled GPU state throws an actionable `ArgumentError`.

**Step 2: Run RED**

Expected: helper/default sentinel not defined.

**Step 3: Implement selection**

Change the IVP constructor default from `matsolver=:sparse` to
`matsolver=:auto`, resolve it after collecting state fields, and add
`_gpu_coupled_state` plus `_select_ivp_matsolver`. Do not change BVP/EVP defaults.
Require the CUDA solver registry entry only on the live GPU-coupled path.

**Step 4: Run GREEN and commit**

Run `test/test_solvers.jl` and relevant compatibility tests. Commit as
`fix: select CUDA solver for mixed GPU IVPs`.

### Task 5: Reuse CUDA sparse-solve buffers

**Files:**
- Modify: `src/tools/gpu_matsolvers.jl`
- Test: `test/test_ilu0_preconditioner.jl` or a new CUDA solver test registered in `GPU_TEST_FILES`

**Step 1: Write a conditional CUDA RED test**

Construct a small nonsingular complex sparse matrix and `CuSparseLU`, warm one
solve, then assert `MatSolvers.solve!` reuses the same RHS, solution, and scratch
buffers and introduces no device allocation on the next solve. Compare with the
CPU solution.

**Step 2: Implement persistent buffers**

Extend `CuSparseLU` with `rhs_buffer`, `solution_buffer`, and `temp_buffer`.
Add `_cusparse_buffers!` and a specialized `MatSolvers.solve!` that copies into
those buffers, calls RF or sparse QR, and copies into the caller's destination.
Keep allocating `solve` as a convenience wrapper around `solve!`. Preserve the
buffers across numeric `refactor!` when the dimension/type is unchanged.

**Step 3: Run conditional GPU tests and commit**

CPU-only hosts must skip cleanly. Commit as `perf: reuse CUDA sparse solve buffers`.

### Task 6: End-to-end forced Fourier--Chebyshev GPU IVP

**Files:**
- Modify: `test/test_gpu_transform_correctness.jl`
- Test: `test/test_separable_stochastic_forcing.jl`

**Step 1: Add the conditional end-to-end test**

Use the channel-like scalar problem from
`test/test_mpi_cheb_fourier_ivp_nonlinear.jl` at `Nx=8`, `Nz=10`, with
homogeneous Dirichlet BCs and a separable profile `z -> z*(1-z)`. Run the same
seeded problem on CPU and GPU for several RK222 steps.

Assert:

- GPU construction auto-selects `CuSparseLU`;
- forcing, field, RK, matrix, and subproblem work buffers are GPU arrays;
- `CUDA.allowscalar(false)` remains enabled;
- CPU/GPU coefficients agree within the transform/solver tolerance;
- forcing is constant across stages and changes across steps; and
- after warmup, another step has zero full-field device allocations. If CUDA
  reports unavoidable internal bytes, enforce a fixed resolution-independent
  ceiling using two resolutions.

**Step 2: Run CPU and conditional GPU suites**

Run:

```bash
julia --project=. test/test_separable_stochastic_forcing.jl
TARANG_RUN_GPU_TESTS=true julia --project=. test/test_gpu_transform_correctness.jl
```

Expected: CPU passes; CUDA test passes on a CUDA host or skips locally.

**Step 3: Commit**

Commit as `test: cover forced mixed-basis GPU IVP`.

### Task 7: Documentation and final verification

**Files:**
- Modify: `docs/src/api/stochastic_forcing.md`
- Modify: `docs/src/pages/stochastic_forcing.md`
- Modify: exports at the bottom of `src/core/stochastic_forcing.jl`

**Step 1: Document the API and limitation**

Add the constructor, profile semantics, GPU solver selection, direct-injection
normalization, and explicit rejection of mixed `:vorticity_kinetic`.

**Step 2: Run verification**

```bash
julia --project=. test/test_separable_stochastic_forcing.jl
julia --project=. test/test_stochastic_forcing.jl
julia --project=. test/test_subproblem_rk.jl
julia --project=. test/test_solvers.jl
julia --project=. test/test_lazy_rhs_fourier.jl
julia --project=. test/test_fourier_algebraic_constraints.jl
julia --project=. -e 'include("test/file_lists.jl"); include("test/test_test_inventory.jl")'
git diff --check
```

Run conditional CUDA tests when hardware is available. Push the branch and
update PR #58 with the mixed-basis behavior, verification counts, and CUDA
coverage status.
