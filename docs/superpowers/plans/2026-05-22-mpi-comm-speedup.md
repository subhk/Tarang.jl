# MPI Communication Speedup — Phased Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans for Phase 1. Phases 2–4 are investigation/roadmap, not yet bite-sized. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cut the MPI communication cost of the distributed spectral transforms, which dominate strong scaling.

**Tech Stack:** Julia, PencilArrays/PencilFFTs (slab decomposition), MPI.jl (MPICH), FFTW.

**Verification binary / MPI launcher:** see project memory and `2026-05-22-parallel-cpu-perf.md`. 2-rank/4-rank runs work via the MPICH bundled launcher with `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib`.

**Do NOT commit without explicit user permission.**

---

## Profile (measured this session, 2D RealFourier, coeff-space operands)

Per distributed nonlinear product, transposes **per rank** (1 forward + 3 backward = 4), and they are **independent of rank count**:

| N (per dim) | nprocs | transposes/rank | local coeff/rank | per all-to-all (≈) |
|-------------|--------|-----------------|------------------|--------------------|
| 128 | 2 | 4 | 64 KiB | 65 KiB |
| 256 | 2 | 4 | 256 KiB | 258 KiB |
| 512 | 2 | 4 | 1024 KiB | 1028 KiB |
| 128 | 4 | 4 | 32 KiB | 32 KiB |
| 256 | 4 | 4 | 128 KiB | 129 KiB |
| 512 | 4 | 4 | 512 KiB | 514 KiB |

**Interpretation:**
- Transpose **count is constant** in rank count (slab/1D decomposition). So scaling ranks shrinks per-message bytes but never reduces the *number* of all-to-alls per product.
- At 2–8 ranks, messages are ~0.1–1 MB → bandwidth-leaning. At hundreds of ranks, per-message bytes → KB → **latency-bound**, where the constant count hurts most.
- Each field transforms **independently** — `transform_grouped.jl` has a batch path but it is **orphaned** (no caller in timesteppers/solvers/nonlinear) and even it applies "per-field", so there is currently no message coalescing anywhere.

This makes **field batching (Phase 1)** the highest-leverage lever: it reduces the constant all-to-all *count* by coalescing the N independent field transposes into one, a pure latency win that compounds with rank count.

---

## Phase 1 — Coalesce per-field transposes into one batched transpose (IMPLEMENTABLE)

**Idea:** When several same-shaped fields must transform in the same direction (the state vector each stage; the operands of an RHS), stack them along a trailing batch dimension into a single PencilArray and issue **one** `MPI_Alltoallv` instead of N. PencilFFTs supports batched/extra-dimension plans, so the FFT + transpose run once over the batch.

**Design spike — INVESTIGATED 2026-05-22, findings below:**
- [x] **Existing `transform_grouped.jl` batch infra is ineffective for MPI.** `_pencil_batch_forward_transform!`/`_pencil_batch_backward_transform!` loop per-field (`for field in fields; plan * grid_data`) — its own docstring says "applied per-field". So wiring it up yields ZERO transpose coalescing. Only the `_stacked_*` (FFTW/serial) path truly batches, and that path has no MPI transpose. Net: the orphaned code does not solve this; real batching must be built.
- [x] **Batched `PencilFFTPlan` works (verified, 2 ranks).** A plan over `(Nx, Ny, B)` with `RFFT, FFT, NoTransform` (B = batch axis) round-trips at err 8.9e-16, preserves the batch dim, and one `plan * u` moves all B slices in a single transpose-set. `NoTransform` on a trailing axis is already the codebase pattern (transform_planning.jl:219 uses it for non-Fourier axes). ⇒ k fields in one batched transpose = 1 transpose-set instead of k.
- [ ] **Open design detail:** in the probe, PencilFFTs' default decomposition *also split the batch dim* (input decomp dims `(2,3)`). For batching we want the batch axis kept LOCAL so all slices ride one transpose — build the `Pencil` with an explicit decomposition over the Fourier dims only, matching the existing per-field plan's layout so stacked/unstacked buffers align.
- [ ] **Still to trace:** the exact state-transform call sites to intercept (`state_utils.jl`, `lazy_rhs.jl`, `nonlinear_evaluation.jl`) — where ≥2 same-shaped fields transform in the same direction (state vector at stage start; RHS component backwards).

**Files (expected — confirm in spike):**
- Modify: `src/core/transforms/transform_grouped.jl` (real batched transpose over a stacked buffer)
- Modify: `src/core/timesteppers/state_utils.jl` and/or the RHS path to call the batched transform for the state vector
- Test: `test/test_mpi_batched_transform.jl` (new) — correctness vs per-field, 2 and 4 ranks

**Acceptance criteria:**
- [ ] A batched forward+backward of `k` same-shaped distributed fields issues **1** all-to-all per direction, not `k` (verify with the transform counter technique used in profiling — re-add temporarily, remove after)
- [ ] Batched result is bit-identical (≤1e-12) to the per-field path at 2 and 4 ranks
- [ ] Full serial + MPI regression green (`test_nonlinear.jl`, `test_dealiasing_math.jl`, `test_solvers.jl`, `test_mpi_dealiasing_product.jl`)

**Verification gate:** transpose-count test shows `k`→`1` coalescing; product/solve values unchanged at 2 and 4 ranks.

**Risk:** medium. Batched PencilArray layout must match field storage; rfft half-complex axis handling must be preserved across the batch.

---

## Phase 2 — Overlap communication with computation (INVESTIGATE)

**Idea:** Transposes are blocking (no `Isend`/`Irecv`/`Ialltoall` anywhere in the tree). Use non-blocking `MPI_Ialltoallv` to overlap the transpose of one field/stage with local FFT work of another, hiding comms behind compute.

**Investigation gate — INVESTIGATED 2026-05-22:**
- [x] **No public non-blocking API.** `PencilArrays.Transpositions` exposes only blocking `transpose!`. `MPI.Ialltoallv!` is **not defined** in the installed MPI.jl (only `MPI.Alltoallv!`). However, the internal split `transpose_send!`/`transpose_recv!`/`transpose_send_self!` suggests PencilArrays drives the transpose with point-to-point `Isend`/`Irecv` rather than a single collective — so a fork could in principle interleave compute between the send and the wait. There is no supported entry point for that today.
- [ ] Is there independent local compute to overlap with (e.g. batched fields, or stage `j+1` setup)?

**Risk:** high. Requires patching/forking PencilArrays' transpose internals (no public async API). Only pursue if Phase 1 proves insufficient and the target scale is clearly latency-bound.

---

## Phase 3 — GPU-aware MPI for the GPU distributed path (DEFER — needs GPU cluster)

**Idea:** `src/core/gpu_distributed.jl` stages every transpose GPU→host→`Alltoall`→host→GPU (see `mpi_alltoall_transpose`). A CUDA-aware MPI build lets `MPI_Alltoallv` operate directly on device buffers, removing the host copies.

**Why deferred:** cannot verify in this environment (no CUDA, no GPU-aware MPI). Requires a CUDA-aware MPI build and a GPU cluster.

**Direction:** detect CUDA-aware MPI (`MPI.has_cuda()`); when true, pass device arrays straight to `MPI.Alltoallv!` and drop the `Array(...)`/`copy_to_device` staging in `mpi_alltoall_transpose`/`_reverse`.

---

## Phase 4 — Decomposition strategy at scale (REVISIT ONLY IF NEEDED)

**Observation:** the code runs slab/1D decomposition ("1D parallelization (fallback)"), which caps usable ranks at `min(N_dims)` and uses the minimum transpose count. This is optimal at the current scale.

**Direction (only when rank count approaches `min(N)`):** 2D pencil decomposition scales to `N²` ranks at the cost of 2 transposes/transform instead of 1. A crossover study (slab vs pencil wall-time at the target rank count) decides. No action until scaling past the slab ceiling.

---

## Self-Review

**Coverage:** the four levers from the discussion map to Phases 1–4. Phase 1 is the only one implementable + locally verifiable now; it is specified to the point a design spike can complete it (full bite-sized steps deferred to post-spike, since the exact batched-transform code depends on confirming the PencilFFTs batched-plan API — writing fabricated code now would violate the no-placeholder rule).

**Placeholder note:** Phases 2–4 are intentionally investigation/roadmap, each with an explicit gate that must be answered before code. They are not padded with fake implementation.

**Grounding:** all recommendations trace to the measured profile table above (rank-independent count of 4 transposes/product; per-message bytes shrinking with rank count → latency-bound at scale → batching first).
