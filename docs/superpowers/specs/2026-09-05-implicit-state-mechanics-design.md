# Implicit-State Mechanics — Design

Date: 2026-09-05. Source: the 2026-09-05 review of `codex/code-review-hardening-source`
and the sync of `main`, read against the audit history in `docs/plans/` and the
contracts in `src/core/module_contracts.jl`.

## Diagnosis

The architecture (Distributor → Domain/Basis → Field → Problem → Solver →
Timestepper) has held through ~40 audit rounds. The recurring bugs are not
architectural; they come from three kinds of state that the code carries
implicitly and that every call site must therefore get right by hand:

1. **Layout.** `current_layout` is a mutable `Symbol`; `get_grid_data` /
   `get_coeff_data` return whichever buffer the field holds without consulting
   it. A stale read is a plausible number, not an error ("silent zero"). Encoding
   layout in the type was judged a dead end on 2026-07-28; the working lever is
   `grid_data!` / `coeff_data!`, whose adoption is tracked by
   `test/test_layout_discipline_ratchet.jl` (277 manual sites at the start of
   this work).

2. **Collectiveness.** Nothing in a signature says "every rank must call this".
   Deadlocks come from a metadata query that builds a plan (collective), a
   constructor that Allreduces, or a per-rank branch that skips a collective.
   The 2026-09-05 sync fixed three instances (`local_shape(:c)`,
   `NetCDFFileHandler.process!`, `create_current_file!`).

3. **Ownership.** Rotating result pools reissue slots regardless of who still
   holds them (two live wrong-answer bugs, both silent); caches keyed on
   `objectid`/shape go stale; and after the GC finalizers were correctly
   removed, MPI communicators had three owners — the Distributor's
   `transpose_workspace_cache`, `DistributedGPUTransform.workspace`, and the
   CUDA extension's `DISTRIBUTED_DCT_PLAN_CACHE` — each with its own close
   policy.

## Decisions

- **Layout:** fold every adjacent `ensure_layout!(x, :L)` + `get_*_data(x)`
  pair into the accessor (mechanical, scripted, gated by the full suite) and
  lower the ratchet. Non-adjacent sites stay; they are the residue the ratchet
  exists to count.
- **Collectiveness:** three rules, each enforced by code not prose:
  1. An accessor never plans. `_field_transform_bundle` refuses when the field
     carries no bundle instead of calling the collective planner.
  2. A per-rank decision gates the body of a collective, never the call
     (already applied to NetCDF creation; documented as the contract).
  3. The contract file lists which entry points are collective.
- **Ownership:** the Distributor is the single owner of MPI communicators.
  `DistributedGPUTransform` borrows the Distributor's cached workspace instead
  of constructing its own; `close(dist)` also releases backend plan caches
  through a hook the CUDA extension implements. The rotating field pools are
  left as they are: they are documented, ratcheted
  (`test_buffer_ownership_ratchet.jl`) and the public producers already default
  to `own=true`; replacing them is a hot-path allocation project, out of scope
  here.
- **Structure:** split `src/core/gpu_distributed.jl` (1594 lines, eight marked
  sections) into one file per section under `src/core/gpu_distributed/`. The
  NetCDF handler file (2857 lines, no section markers) needs its own design
  pass and is deferred.
- **Process:** the test inventory also checks that every registered test file
  is tracked by git, closing a failure that has recurred three times.

## Out of scope

Rotating pool replacement; `netcdf_output.jl` split; equation-pipeline
changes (the single-parse path already landed on 2026-09-05).
