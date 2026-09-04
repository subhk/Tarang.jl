# Batched Fourier-mode solve for the 2D GPU hot path

Date: 2026-08-08
Status: approved design, not yet implemented
Follows: PR #105 (`perf/gpu-2d-hotpath`), which fixed the transform and
timestepper hot paths but explicitly deferred this item.

## Problem

For a 2D mixed Fourier-Chebyshev IVP, `step_subproblem_rk!` loops over one
`Subproblem` per Fourier mode and issues roughly twenty device operations for
each. At `Nx = 256` with RK222 that is about `129 x 3 x 20 ~ 7,700` kernel
launches per step, on vectors of length `n = nz + 2`. Launch overhead dominates;
the GPU is idle almost the whole step. This is the largest remaining cost on the
coupled GPU path.

## Measurements

Taken on the wall-bounded channel problem from
`test/test_gpu_fc_2d_complete.jl` ("Nonlinear wall-bounded 2D FC IVP"), built on
CPU at two resolutions. Script preserved in the session scratchpad.

| quantity | `nx=64, nz=32` | `nx=128, nz=64` |
| --- | --- | --- |
| subproblems (all live) | 33 | 65 |
| `n = size(LHS, 1)` | 34 | 66 |
| `nnz(LHS)` | 353 | 1217 |
| density | 0.305 | 0.279 |
| bandwidth | `(32, 32)` | full |
| `nnz(M_min)` | 32 | 64 |

Uniformity across modes, measured rather than assumed:

- `size(LHS)` identical: yes
- `LHS` sparsity pattern (`colptr`, `rowval`) identical: yes
- `M_exp` / `L_exp` patterns identical: yes
- `M_min` pattern identical: yes
- **`L_min` pattern identical: NO.** At `nx=64, nz=32`, `L_min` stores 353
  nonzeros at every mode except `kx=0`, which stores 321 — the `∂xx` term is
  the literal zero operator there, so those entries are never created. This is
  the one place uniformity fails, and it fails on essentially every problem
  containing a second derivative. `L_exp` is `expand_pattern(L_min, LHS)`:
  verified exactly equal to `L_min` numerically, carried in `LHS`'s union
  pattern, and uniform across all modes including `kx=0`. The batch signature
  and the batched `L·X` product therefore both use `L_exp`, never `L_min`.
  Hashing `L_min` would exile `kx=0` to its own bucket for no benefit.
- `bc_rows` identical (`[2, 3]`): yes
- `bulk_cols` / `bc_cols` identical: yes
- `pre_left`, `pre_right`, `pre_left_pinv`, `pre_right_pinv` identical, each with
  `nnz == n` (a permutation or diagonal): yes
- `LHS.nzval` identical: no, as expected

Gather and scatter already reduce to a single strided run per field,
`(start, step, len)`, via `sp.runtime.strided_index_cache`.

Two conclusions follow. The modes form a perfect batch: same shape, same
pattern, differing only in values. And the matrices are effectively small dense
ones, not banded and not usefully sparse.

## Approaches considered

**Batched banded solve (`gtsv2StridedBatch`).** Rejected. This is what the
original deferral note proposed, but the measured bandwidth is `(32, 32)` at
`n = 34` and full at `n = 66`. Chebyshev tau rows are dense. The matrices are not
banded.

**Batched sparse with a shared symbolic factorization.** Rejected. The identical
sparsity pattern does make one symbolic factorization tempting, with `nzval`
stored as `(nnz, nmodes)`. But partial pivoting diverges per mode, which breaks
the shared-symbolic premise. The escapes are pivot-free factorization, unsafe on
a stiff Chebyshev operator, or per-mode symbolic analysis, which defeats the
purpose. Hand-written batched sparse LU with fill-in is a research project.

**Batched dense LU. Chosen.** At density 0.28 to 0.31 with full bandwidth, dense
storage costs about 3.5x the `nzval` storage and buys `CUBLAS.getrf_batched!`
and `getrs_batched!`, already wrapped in CUDA.jl, with real partial pivoting and
one launch for all modes. The dense buffer is 4.5 MB at `nx=128, nz=64`. It
scales as `n^2 * nmodes`, so it needs an explicit size guard: `nz=256, nx=512`
would reach 546 MB.

## Scope

Decided with the user before design:

- Batch the **whole stage pipeline**, not just the solve. The solve is one
  operation in twenty; batching it alone would take 7,700 launches to about
  7,400.
- On non-uniform modes, **bucket by signature and loop the leftovers**. Buckets
  with two or more members are batched; the rest keep the existing per-mode path.
- **GPU by default, CPU behind an explicit opt-in flag** (`batched_modes`, see
  Engagement predicate). The implementation is device-generic so the test suite
  exercises the real code path on CPU arrays. No CPU behavior changes by
  default.
- **Serial single-GPU only.** `nprocs > 1` never constructs a batch. Distributed
  batching is a follow-up.

## Architecture

Five units, each with one responsibility and testable in isolation. New code
lands in new files; `step_subproblem_rk.jl` (875 lines) and `subproblem_io.jl`
(1119 lines) do not grow.

### 1. `batch_signature(sp)` — `src/core/subsystems/mode_batch.jl`

A pure function producing a hashable key from the matrices as actually built:
`size(LHS)`, `LHS.colptr`, `LHS.rowval`, `size(M_min)`, `M_min.colptr`,
`M_min.rowval`, `bc_rows`, `bulk_cols`, `bc_cols`, and the preconditioner
patterns. Uniformity is observed, never assumed. This function is the entire
basis of the bucket-and-loop-leftovers policy.

### 2. `ModeBatch` — same file

Immutable-shaped data built once after `build_matrices!`:

- `sp_indices::Vector{Int}`, `n::Int`, `nmodes::Int`
- `M_exp_nzval`, `L_exp_nzval`, each `(nnz, nmodes)` on device, uploaded once
- shared pattern vectors `colptr`, `rowval`: one copy, not `nmodes` copies
- `lhs_dense::(n, n, nmodes)`, `pivots::(n, nmodes)`, `info::(nmodes,)`
- `bc_rows_device`
- per-field gather and scatter descriptors `(start, step, len, mode_stride)`
- factorization state: the `a_ii` and `dt` a factorization is valid for, plus a
  dirty bit

`ModeBatch` knows nothing about timesteppers or RK tableaus.

Keeping `M_exp_nzval` and `L_exp_nzval` resident on device means
`batched_assemble_lhs!` rebuilds every mode's LHS on the GPU from
`M_exp + dt*a_ii*L_exp`. That retires a separate deferred item from PR #105,
"adaptive-dt rebuilds LHS.nzval on host and re-uploads per mode", at no extra
cost.

### 3. Batched kernels — `src/core/subsystems/mode_batch_kernels.jl`

KernelAbstractions, device-generic: `batched_gather!`, `batched_scatter!`,
`batched_spmv!` (shared pattern), `batched_bc_override!`,
`batched_assemble_lhs!`.

Every kernel writes each element exactly once and never re-reads it. The KA CPU
backend miscompiles same-slot read-modify-write around inner loops; PR #105
found this the hard way in `_cheb_coeff_to_deriv_kernel!`.

### 4. `BatchedDenseLU` — `src/tools/batched_matsolvers.jl`

`factor!` and `solve!` only, no knowledge of subproblems. GPU dispatches to
`CUBLAS.getrf_batched!` and `getrs_batched!`; CPU dispatches to per-slice `lu!`
and `ldiv!`.

### 5. Batched stage loop — `src/core/timesteppers/step_subproblem_rk_batched.jl`

Orchestration only. It runs the same mathematical sequence as the existing loop,
on `(n, nmodes)` matrices. `step_subproblem_rk!` dispatches to it or to the
existing per-mode loop.

## Data flow

At build time, subproblems are bucketed by `batch_signature`. Buckets with two or
more members that pass the memory guard become `ModeBatch`es; everything else
keeps a per-mode `Subproblem`. The device uploads happen once here.

Per step, `X`, `RHS`, `MX0`, `F[j]`, `LX[j]`, and `ALG_F` are all `(n, nmodes)`
device matrices with one column per mode:

```
dt changed          -> mark batch dirty  (reuses the existing lhs_dirty handler)

pre-stage:  batched_gather!(X0, fields)                            ~3 kernels
            batched_spmv!(MX0, M_nzval, X0)                          1
            ALG_F: static BCs uploaded once, identity-gated           0

stage i:    RHS .= MX0                                                1
            RHS .+= dt*A_e[i,j].*F[j]                          2 per j
            RHS .-= dt*A_i[i,j].*LX[j]
            batched_bc_override!(RHS, ALG_F, bc_rows, dt*a_ii)        1
            if dirty: batched_assemble_lhs! then getrf_batched!       2
            getrs_batched!(X, RHS)                                    1
            batched_scatter!(fields, X)                             ~3
            evaluate_rhs                                     (unchanged)
            batched_gather!(F[i]); batched_spmv!(LX[i])               2
```

**Measured outcome, superseding the estimate below.** The 42-launch figure assumed
every stage could be batched. It cannot. `M_min` is structurally singular in any
tau/BC formulation — its tau rows are identically zero, so `nnz(M_min) == nz` in
an `(nz+2)x(nz+2)` matrix — and the mass solve is therefore a rank-deficient SPQR
least-squares that a dense batched LU cannot reproduce. RK222 needs it twice per
step: the `a_ii == 0` first stage, and the final update (RK222 is implicitly but
not explicitly stiffly accurate). Those two solves stay per-mode.

Measured at `nmodes = 129`:

| | operations per step |
| --- | --- |
| before | 63 per mode x 129 = 8,127 |
| after | 65 batched + 6 x 129 = 839 |

**About 9.7x, not the 180x this section originally claimed.** What remains
per-mode is only the solve plus two column-staging copies; the gathers, scatters,
RHS accumulation, and stage scatter are batched even in the `a_ii == 0` stages.
Going beyond 9.7x requires a batched rank-deficient least-squares solver, which is
a substantially larger piece of work than this one.

The original estimate follows, for the record: about 42 launches per step against
roughly 7,700.

`evaluate_rhs` is deliberately untouched: it is already whole-field, not
per-mode. The solve-layout bracketing is likewise untouched, since serial scope
makes it a no-op.

## Engagement predicate

Control is a single solver keyword, `batched_modes`:

- `nothing` (default): automatic. Batch on GPU, do not batch on CPU.
- `true`: batch wherever the structural conditions below allow, including CPU.
  This is what the test suite uses.
- `false`: never batch. The escape hatch if the batched path is ever suspect.

The memory cap is a second keyword, `batched_modes_max_bytes`, defaulting to
1 GiB.

All four conditions are then required:

1. `nprocs == 1`
2. `batched_modes` resolves to true for this device
3. the bucket holds at least two modes
4. `n^2 * nmodes * 16 bytes <= batched_modes_max_bytes`

Failing condition 4 drops that bucket alone to the per-mode loop and emits
`@info maxlog=1` reporting the bucket size and the cap, because a silent perf
cliff at large `nz` is exactly the kind of thing that goes unnoticed for months.
Failing condition 3 is silent: a one-mode bucket has nothing to batch, and
warning about it would be noise on every small problem.

Failing condition 1 or 2 means no batch is ever constructed, so every existing
CPU, MPI, and non-GPU run is byte-for-byte unaffected.

## Numerical expectations

Switching the GPU solve from `CuSparseLU` to batched dense LU changes results at
roundoff level, because the pivot order differs. Everything else in the pipeline
is the same arithmetic re-laid-out and stays bit-exact.

The acceptance criterion is therefore split three ways:

- gather, scatter, spmv, BC override, and the RHS copy and axpy: **bit-exact**
  against the per-mode implementations. These are the same operations on the
  same values, only re-laid-out.
- `batched_assemble_lhs!`: agreement to `1e-15` relative, not bit-exact. It
  computes `M_exp + dt*a_ii*L_exp`, which a kernel may contract into an FMA
  where the host expression does not. Asserting bit-exactness here would make
  the test hostage to backend codegen.
- the solve: agreement to `1e-12` relative, reflecting the pivot-order change.
- a five-step channel trajectory: agreement to `1e-12` relative.

## Error handling

Each item corresponds to a specific past failure in this repository.

- **`getrf_batched` `info` is checked, per mode.** A singular mode otherwise
  returns garbage that looks like a plausible answer. Nonzero `info` raises,
  naming the mode index and the bucket. This is the silent-zero class and it is
  the most dangerous line in the change.
- **No silent CPU fallback on GPU.** CUBLAS unavailable or failing raises,
  mirroring the existing GPU branch in `_get_or_build_lhs!`. Contract #74.
- **Factorization validity is gated on identity plus an explicit dirty bit**,
  never a bare flag. A stale flag with a reallocated buffer silently serves
  zeros; that is exactly how the ALG_F bug behaved.
- **Signatures are computed from built matrices**, never from `nz` or `nvars`
  arithmetic. Assuming uniformity is how a wrong-but-plausible batch gets built.
- **Kernels write each element exactly once.** KA CPU same-slot RMW miscompile.

## Testing

No CUDA is available locally, so all of the following run on CPU arrays through
the real code path, using the CPU opt-in flag.

1. `test_mode_batch_signature.jl` — a uniform problem yields one bucket;
   perturbing one mode's pattern yields two buckets plus the correct leftovers.
2. `test_mode_batch_kernels_cpu.jl` — each batched kernel against the existing
   per-mode function, bit-exact.
3. `test_batched_dense_lu.jl` — factor and solve against per-slice `lu()`; a
   deliberately singular mode must raise rather than return.
4. `test_mode_batch_parity.jl` — the channel IVP with `batched_modes=true`
   versus `batched_modes=false`, five steps, against the tolerances above.
5. Guard tests — `nprocs > 1` never constructs a batch; `batched_modes=false`
   never constructs a batch; a bucket exceeding `batched_modes_max_bytes` falls
   back and emits its `@info`; a one-mode bucket falls back silently.
6. Registration in `test/file_lists.jl`, and an extension to
   `test/run_gpu_fc_2d.jl` so the cluster run covers this on real hardware.

## Known risks

**The JET ratchet has zero headroom.** Main sits at exactly 975 against a
`<=975` bound. Five new files will move that number. The intent is to meet the
existing bound through type stability rather than raise it, but this may take
real effort and the bound may have to move.

**Speedup is unverifiable locally.** Semantics, parity, and guards can all be
verified without hardware. The launch-count reduction and any wall-clock gain
cannot; those require the cluster GPU.

**`origin/main` moves during sessions.** PR #104 landed mid-session during the
PR #105 work and forced a rebase of a whole-file rewrite. Fetch and check
`origin/main` before opening the PR.
