# PR 126 Review Remediation

> Execution: verify each report against the current branch, reproduce before
> changing behavior, and record focused verification before closing an item.

**Goal:** Resolve or explicitly qualify all 37 external review findings.

**Architecture:** Preserve the public equation and timestepper APIs. Prefer typed
operator dispatch and existing runtime ownership boundaries. Unsupported numerical
or distributed operations must fail explicitly rather than silently contribute zero.

**Tech Stack:** Julia, Test, MPI.jl, PencilArrays/PencilFFTs, FFTW, CUDA, NetCDF.

## Work Order

1. Expose the complete CPU failure inventory in `test/runtests.jl`; repair stale
   regression expectations without weakening correctness or allocation checks.
2. Reproduce operator traversal, implicit NCC, forcing, mass, and initialization
   failures; fix ownership/dispatch at the source and add registered regressions.
3. Verify RK/DAE, startup convergence, GPU/MPI routing, and decomposition guards.
4. Verify concurrency and checkpoint/output lifecycle, scaling, and merge behavior.
5. Audit remaining dead APIs, compatibility decisions, bounds, and performance claims.
6. Run focused CPU/MPI tests, the full CPU suite, and available device checks.
   Distinguish unavailable GPU hardware checks from passes.

## Finding Ledger

All items begin open; source inspection alone does not establish a numerical result.

| # | Topic | Status |
| --- | --- | --- |
| 1 | Nonlinear convenience operator traversal/evaluation | Fixed; numerical evaluation and LHS refusal regressions pass |
| 2 | Identity Jacobian fallback | Fixed in the global nonlinear solve; unregistered derivative errors propagate with and without a linear matrix |
| 3 | Component NCC refusal bypass | Fixed; derived scalar leaf retained and NCC refusal regression passes |
| 4 | Stiffly accurate RK constraint targets | Open |
| 5 | RKGFY stiff damping | Qualified method limitation: numerical decay matches Crank-Nicolson amplification, which approaches -1; documented as not L-stable |
| 6 | GPU MPI serial routing | Open |
| 7 | SBDF2 history key regression | Fixed; recycling identity test passes with current key |
| 8 | Checkpoint warning assertion | Fixed; current public-name warning assertion passes |
| 9 | Chebyshev shared scratch race | Fixed; task-owned cache, including inherited TLS isolation |
| 10 | Detached forcing destination | Open |
| 11 | SBDF3/4 startup convergence | Qualified documented limitation: single-state lower-order startup gives approximately second-order overall convergence on CPU, for explicit and implicit decay |
| 12 | CNLF2 explicit stability | Qualified method limitation: numerical explicit decay matches the leapfrog recurrence with an unstable parasitic root; documented implicit treatment of dissipation |
| 13 | Inert NCC infrastructure | Open |
| 14 | Algebraic initialization before IR | Open |
| 15 | Fourier over-decomposition | Open |
| 16 | Partial merge transforms and zero-field dtype | Open |
| 17 | Coefficient output scales metadata | Open |
| 18 | Distributed output scaling/validation | Open |
| 19 | State storage/metadata aliasing | Open |
| 20 | EVP boundary namespace | Open |
| 21 | Tensor/component identity mass | Open |
| 22 | Constant nonunit mass on diagonal paths | Open |
| 23 | Singleton parameter conversion/device access | Zero-imaginary CPU conversion claim disproved locally; device paths open |
| 24 | Dead field guards | Open |
| 25 | Test runner early abort | Fixed outer testset; focused run continued through failures and errors |
| 26 | Merge committed-write markers | Open |
| 27 | Polynomial diffusion CFL | Open |
| 28 | LHS parse error diagnostics | Open |
| 29 | Distributed GPU communicator cleanup | CUDA hook method overwrite fixed via backend dispatch; extension loads and cleanup is idempotent on CPU; device collective teardown remains unverified |
| 30 | Dropped forcing on mismatch | Fixed for incompatible coefficient shapes in lazy and interpreted RHS evaluation; both now throw ArgumentError |
| 31 | Public removals without deprecation | Intentional user-requested removal; current docs/examples contain no retired variant/startup keyword usage, and API rejection regressions pass |
| 32 | Exception swallowing in dependency detection | Open |
| 33 | Duplicate transpose workspace cleanup | Open |
| 34 | Unused exports | Open; absence of internal callers alone is not a public-API defect |
| 35 | Mixed numeric Fourier bounds | Open |
| 36 | Long functions | Open; refactor only where needed for verified defects |
| 37 | Stage work, forcing allocation, cache locks | Open; measure before changing synchronization |

## Verification Protocol

For each behavioral fix: run the smallest reproduction before the edit; add a
regression to an existing registered test file; implement the smallest coherent
fix; rerun that file and related tests. Preserve unrelated untracked files.

CPU: `julia --project=. --startup-file=no test/runtests.jl` (test dependencies
must be available; use the package test environment when required).

MPI: launch focused files with `MPI.mpiexec()` at 1, 2, and 4 ranks as applicable.
Use exact solutions, independent stability functions, or order checks instead of
refreshing golden constants from the implementation under test.

First batch: Julia 1.13, four threads, `--compiled-modules=no`, equation IR,
polynomial derivatives, variable-coefficient LHS, diagonal coverage, and checkpoint
tests: 306/306 passed. Before fixes: 299 passed, 5 failed, 1 error (after the
separately reproduced nonlinear failure was repaired). Full-suite execution was
interrupted during conflicting precompilation; rerun serially. Julia LTS 1.10.12
is also installed. No GPU hardware tests have been run for this audit.

Resume verification (2026-09-12): on the working tree at `0378a5126` with the
pending edits, Julia 1.13, four threads, `--compiled-modules=no`, the registered
`test_state_arith_layout.jl`, `test_timestepper_boundaries.jl`, and
`test_mode_batch_parity.jl` files passed 345/345 assertions (test time 1m26.5s).
Log: `/private/tmp/tarang-resume-regressions.log`. `git diff --check` also passed.
This verifies the pending state-name and RK changes against those regressions;
it does not close the broader finding ledger or establish GPU hardware coverage.
The older `/private/tmp/pr126-cpu-verification-2.log` is from another checkout
and must not be counted as verification of this working tree.
Next: obtain the complete CPU failure inventory, then continue the open findings
with focused reproductions and MPI verification.

Continuation verification (2026-09-13): reproduced the swallowed Jacobian errors
(2 failed assertions), ignored forcing shape mismatches (2 failed assertions),
and missing backend cleanup dispatch (1 failed assertion). Each focused
regression passed after its fix. A fresh Julia process then ran
`test_solver_review_regressions.jl`, `test_symbolic_diff.jl`,
`test_stochastic_forcing.jl`, `test_cov_solver_compiled_rhs.jl`, and
`test_cuda_dct_cache_context.jl`: 554 passed, 7 broken/skipped, no failures or
errors (1m15.1s). Log: `/private/tmp/tarang-review-fresh-green.log`.

The full `Pkg.test()` launch reproduced a CUDA extension precompilation error:
the extension replaced the core `_close_backend_plan_caches!(::Distributor)`
method. Backend dispatch fixes this without disabling precompilation. A fresh
process using the package-test environment successfully loaded Tarang and CUDA,
then closed a CPU Distributor twice. Log:
`/private/tmp/tarang-cuda-extension-load.log`. This is extension-load coverage,
not GPU numerical or distributed GPU teardown coverage.

MPI verification passed at both 2 and 4 ranks for
`test_mpi_checkpoint_restart.jl`, `test_mpi_forcing_diag.jl`, and
`test_mpi_distributor.jl` (all six runs, launcher exit 0). This covers collective
CPU lifecycle and checkpoint/forcing behavior; CUDA hardware cases were skipped.
Log: `/private/tmp/tarang-resume-mpi.log`.

Full CPU inventory run: `/private/tmp/tarang-resume-pkg-cpu.log`; launched before
the continuation source fixes, so any failures must be checked against the
fresh source.

The inventory completed: 12,219 passed, 8 failed, 20 broken/skipped, no errors
(13m48.9s). Five failures were the new Jacobian, forcing, and backend-cleanup
regressions against source loaded before their fixes. The other three were two
attached-operator assertions using an array 2-norm tolerance instead of the
neighboring pointwise tolerance, and one NetCDF scalar-versus-one-element-array
attribute assertion. Corrected the norm assertion and added an independent
SBDF2 discrete recurrence check; normalized only the NetCDF attribute's scalar
representation while retaining its exact expected value. Full updated-source
run: `/private/tmp/tarang-final-full-cpu.log`. Preserved package-test environment:
`/private/tmp/tarang-resume-test-env`.

Stability qualification: `test_reference_timesteppers.jl` passed 66 assertions,
including 10 new checks against independent scalar amplification/recurrence
formulas for RKGFY and CNLF2. These expose method limitations without changing
the schemes. Updated API documentation and corrected the CNLF2 docstring's
two-step implicit factors. Log: `/private/tmp/tarang-stability-limits.log`.

The corrected GPU-reference and NetCDF files passed 376/376 assertions in a
fresh process (1m10.2s). Log:
`/private/tmp/tarang-inventory-assertions-green.log`.

SBDF startup qualification: on CPU, SBDF3 and SBDF4 with both explicit and
implicit `u'=-u`, integrating to t=1 from a single state with dt=0.02, 0.01,
0.005, have measured convergence orders 1.988–2.007 against exp(-1). This agrees
with the existing documented second-order startup limit; it is not evidence of
third-/fourth-order convergence from a single initial state. Four checks passed.
Probe: `/private/tmp/tarang-sbdf-startup-order.jl`; log:
`/private/tmp/tarang-sbdf-startup-order.log`.

Independent read-only review of the continuation's Jacobian/forcing/cleanup
fixes, assertion repairs, and stability documentation found no actionable
regressions. The reviewer confirmed that actual multi-GPU DCT teardown remains
a hardware-validation gap.

The updated full-suite setup emitted an MPI precompile-cache availability
error before continuing with compiled modules disabled. Separately, normal
strict precompilation of MPI and Tarang in the preserved test environment,
followed by loading Tarang and CUDA, passed with exit 0 and no method-overwrite
error. Log: `/private/tmp/tarang-final-precompile.log`.

Final validation of this continuation batch: the updated-source full CPU suite
passed with 12,239 passed assertions, 20 broken/skipped, zero failures/errors
(11m55.8s), and `Pkg.test()` exited 0. Log:
`/private/tmp/tarang-final-full-cpu.log`. All six focused 2-/4-rank MPI runs also
passed, as recorded above. `git diff --check` passed. Verification was completed
before committing this batch.

Next batch: continue the open ledger items, starting with forcing destination
ownership (#10), algebraic initialization ordering (#14), and component/nonunit
mass handling (#21–22). A passing existing suite does not close those reports;
each still needs a focused reproduction or an explicit qualification.
