# Backward-side coupled-DCT fusion (distributed mixed Cheb-Fourier IMEX)

**Date:** 2026-07-10
**Status:** approved (per-stage RK scope)
**Area:** `src/core/subsystems/subproblem_io.jl`, `src/core/transforms/transform_fourier.jl`, `src/core/timesteppers/step_subproblem_rk.jl`

## Problem

On the distributed mixed Fourier–Chebyshev IMEX path, each RK stage restores state
from the solve pencil then re-evaluates the RHS:

- `step_subproblem_rk.jl:599` `from_solve_layout!` transposes state **solve→fft**.
- `evaluate_rhs_buffered` (line 610) pulls state to `:g` via `backward_transform!`,
  whose first action is `_apply_distributed_coupled_dct!(field, false)`:
  transpose **fft→solve**, local inverse coupled-DCT, transpose **solve→fft**; then
  the Fourier `ldiv`→`:g`.

The state was *already in the solve pencil* before `from_solve`. So the `solve→fft`
leg (599) followed immediately by the coupled-DCT's `fft→solve` leg is a pure
`solve→fft→solve` round-trip with nothing reading the fft-pencil state in between
(lines 600–609 are comments only; the next coeff read is `evaluate_rhs` at 610).

Cost: **2 redundant transpose collectives per state field per stage** (RK222 4/step,
RK443 8/step). This is the backward analogue of the forward-side round-trip already
fused (`to_solve_layout!(...; fuse_from_grid=true)`, committed).

## Design — flag-free fused backward

Prior notes assumed this needs a persistent "coupled-axis-grid despite `:c`"
hybrid-state flag surviving the `from_solve → evaluate_rhs` boundary (broad blast
radius, high risk). It does not. Because the state is already in the solve pencil,
fuse the entire backward transform into `from_solve`, landing state directly at `:g`:

1. `_solve_layout_backward_transform!(solve_pa, dist)` — inverse coupled-DCT
   **locally in the solve pencil** (0 transpose; coupled axis → grid).
2. `PencilArrays.transpose!(fft_pa, solve_pa)` — solve→fft **once**.
3. `backward_transform!(f, :g; apply_coupled_dct=false)` — Fourier `ldiv`→`:g`,
   skipping the (already-applied) coupled-DCT.

`evaluate_rhs`'s `ensure_layout!(:g)` then becomes a no-op. Coupled-DCT-related
transposes drop **3 → 1** (saves 2/field/stage). This pairs symmetrically with the
committed forward fusion (`to_solve; fuse_from_grid`, which expects `:g` — line 622).

The transient hybrid state (Fourier-spectral, coupled-axis grid, `:c`-flagged)
exists only *between steps 2 and 3, inside `from_solve_layout!`* — it never escapes
the function, so no external code observes a mislabeled field. **No persistent
flag.** This is the load-bearing safety difference from the assumed-risky design.

## Mechanism (2 code changes + 1 call site)

1. **`backward_transform!(field, target=:g; apply_coupled_dct::Bool=true)`**
   (`transform_fourier.jl`) — new kwarg mirroring the existing `forward_transform!`
   one; when `false`, skip `_apply_distributed_coupled_dct!(field, false)` and do
   only the Fourier `ldiv`. Default `true` → all existing callers unchanged.
2. **`from_solve_layout!(stash, dist; to_grid::Bool=false)`**
   (`subproblem_io.jl`) — when `true`, for each fused field run steps 1–3 above
   instead of the plain `solve→fft` transpose + `set_coeff_data!`. Default `false`
   → all other callers (multistep line 329, steady solvers) unchanged.
3. **`step_subproblem_rk.jl:599`** — pass `to_grid=true` (this call site only).

## Scope

Per-stage RK `from_solve` at `step_subproblem_rk.jl:599` **only**. The end-of-step
RK `from_solve` (line 691) and the multistep `from_solve` (line 329) are left
unchanged — they precede a history push whose stored layout would change to `:g`
(safe but broader). Out of scope: F-field forward fusion (separate deferred item).

## Verification / acceptance

- **Bit-identical** (primary): existing distributed Cheb-Fourier guards reproduce
  their references to roundoff at np2 AND np4 — `test_mpi_cheb_fourier_ivp`
  (RK222 + SBDF2), `test_mpi_cheb_fourier_ivp_nonlinear`, `test_mpi_bvp_cheb_fourier`,
  `test_mpi_algebraic_constraints`, `test_mpi_padded_dealiasing`. A rank-divergent
  transpose count would deadlock/diverge → these passing at np2 AND np4 is the
  correctness net.
- **Transpose-count reduction** (the actual win): instrument/verify the per-stage
  coupled transpose count drops from 3 to 1 for a mixed Cheb-Fourier RK222 field.
- **Adversarial review**: aliasing/hybrid-state-escape, rank-uniformity/deadlock,
  bit-identical numerics.

**Acceptance = bit-identical correctness + verified-lower transpose count.** The
single-node wall-clock win is expected **below the noise floor** (transposes ~4% of
wall; the forward fusion measured "within noise"); the real benefit is fewer
collectives at higher rank-count / slower interconnect. **Revert if** it is not
bit-identical, or cannot be shown to reduce the transpose count.
