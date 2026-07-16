# Stochastic Forcing Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct stochastic forcing normalization, registration, diagnostics, spectrum statistics, and unsafe timestepper behavior.

**Architecture:** Store an explicit injection metric on each forcing and centralize its spectral weights. Resolve registration against flattened solver fields. Keep one-step forcing behavior, but fail fast on RHS-extrapolating multistep schemes.

**Tech Stack:** Julia, FFTW/RFFT coefficient layouts, MPI/PencilArrays, KernelAbstractions/CUDA extensions, Test stdlib.

### Task 1: Flattened forcing registration

**Files:**
- Modify: `test/test_stochastic_forcing.jl`
- Modify: `src/core/problems/problem_types.jl`

1. Add a failing test with `IVP([u::VectorField, q::ScalarField])` proving `:q` maps to flattened index 3 and `:u_x` maps to index 1.
2. Add a failing ambiguity test for `:u`.
3. Run the focused file and confirm failure.
4. Implement flattened component traversal and validation.
5. Run the focused tests and confirm they pass.

### Task 2: Metric-aware spectrum normalization

**Files:**
- Modify: `test/test_stochastic_forcing.jl`
- Modify: `src/core/stochastic_forcing.jl`

1. Add failing tests for resolution-independent direct energy, vorticity `1/k^2` normalization, covariance ring width, invalid metric, negative rate, and empty positive-rate bands.
2. Run the focused file and confirm failure.
3. Add `injection_metric` to `StochasticForcing`, centralize full-spectrum metric weights, normalize with unnormalised-FFT Parseval scaling, and correct the ring amplitude exponent.
4. Run the focused tests and confirm they pass.

### Task 3: Hermitian and work diagnostics

**Files:**
- Modify: `test/test_stochastic_forcing.jl`
- Modify: `src/core/stochastic_forcing.jl`
- Modify if needed: `ext/cuda/utils.jl`

1. Add failing tests for Nyquist expected variance and RFFT-weighted direct/vorticity work.
2. Run the focused file and confirm failure.
3. Preserve self-conjugate variance and implement metric-aware Parseval/RFFT reductions without GPU scalar indexing.
4. Run focused CPU tests; run CUDA tests when available.

### Task 4: Guard unsafe multistep schemes

**Files:**
- Modify: `test/test_stochastic_forcing.jl`
- Modify: `src/core/timesteppers/dispatch.jl`

1. Add failing tests that CNAB2/SBDF2 reject registered stochastic forcing and RK222 remains supported.
2. Run and confirm failure.
3. Add a dispatch-time compatibility guard with an actionable error.
4. Run and confirm pass.

### Task 5: Documentation and examples

**Files:**
- Modify: `docs/src/pages/stochastic_forcing.md`
- Modify: `docs/src/api/stochastic_forcing.md`
- Modify: `examples/ivp/forced_2d_turbulence.jl`

1. Document metric semantics and supported timesteppers.
2. Set `injection_metric=:vorticity_kinetic` in vorticity examples.
3. Correct work, spectrum, RNG, and normalization formulas.

### Task 6: Verification

1. Run `julia --project=. test/test_stochastic_forcing.jl`.
2. Run the two-rank MPI stochastic forcing test through `MPI.mpiexec()`.
3. Run `git diff --check` and inspect the final diff.
4. Confirm the original physical-energy, multistep, and flattened-index reproductions no longer silently return incorrect results.
