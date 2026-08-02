# Parallel NetCDF slab I/O and solver checkpoint/restart

**Date:** 2026-07-31
**Status:** approved design, not yet implemented

## Problem

Tarang can write distributed output but cannot read it back.

The `NetCDFFileHandler` already writes Dedalus-shaped per-rank files —
`name_s1/name_s1_p0.nc` — and stamps each variable with `start`, `count`, and
`global_shape` attributes (`src/tools/netcdf_output.jl:2255-2257`). Nothing reads
those attributes except the post-hoc merge.

There is no reader counterpart and no restart:

- `load_field!` has rank 0 `ncread` the **entire global array**, then scatters
  (`src/core/field/field_layout/field_layout_arithmetic_io.jl:114-170`). Rank 0
  must hold the whole field. It cannot read the per-rank files the writer produces.
- There is no solver-level `save_state`/`load_state!`. `docs/src/api/io.md` says so
  outright: "Tarang has no built-in checkpoint type. Write a small helper."
- Both exported field-I/O functions are broken, and neither appears anywhere in
  `test/`:

  ```
  save_field CPU FAILED: NetCDF error code 2: No such file or directory
  load_field! CPU FAILED: rethrow(exc) not allowed outside a catch block
  ```

  `save_field` calls `ncwrite` with no preceding `nccreate`, so it throws on every
  call. `load_field!` calls `rethrow(load_error)` outside a `catch` block
  (`:138`), which Julia rejects — destroying the real NetCDF error it meant to
  surface.

## Goals

1. Fix `save_field` and `load_field!`, with tests.
2. Read a distributed checkpoint **at any rank count**, not just the one that wrote it.
3. Solver-level `save_state` / `load_state!` restoring fields plus the simulation clock.
4. Make the GPU path explicit rather than accidental.

## Non-goals

- JLD2 (would add a dependency; separate decision).
- Coefficient-space checkpoints. Grid space is real-valued and round-trips losslessly.
- True MPI-IO (`nc_create_par` / PnetCDF). NetCDF.jl does not expose it.
- Serializing timestepper history — see "Restart fidelity" below.

## Decisions

| Question | Decision |
|---|---|
| Restart at a different rank count? | **Yes.** Each rank computes its own local range and reads the overlapping hyperslabs from whichever files cover it. |
| Checkpoint contents? | **State + clock** (`sim_time`, `iteration`, `dt`). Multistep restarts re-seed and emit a warning naming the cost. |
| Write layout under MPI? | **Per-rank slabs when `np > 1`, single file when `np == 1`** — the rule `current_file` already uses (`netcdf_output.jl:452`). |

## Architecture

Three layers, each usable and testable without the ones above it.

```
src/tools/netcdf_slab_io.jl                       NEW
    Knows only about files and index ranges. No fields, no solvers, no MPI calls.

src/core/field/field_layout/field_layout_arithmetic_io.jl   FIXED
    save_field / load_field! delegate to the slab layer. Signatures unchanged.

src/core/solvers/solver_checkpoint.jl             NEW
    save_state / load_state!. Knows about solvers; delegates fields downward.
```

Rejected alternatives: growing `field_layout_arithmetic_io.jl` (already mixes field
arithmetic with I/O) and putting the solver layer in `solver_stepping.jl` (already
large); adding a read mode to `NetCDFFileHandler` (a scheduling object — restart
needs none of its cadences, tasks, or evaluator wiring).

### Layer 1 — `netcdf_slab_io.jl`

```julia
slab_overlap(src_start::Vector{Int}, src_count::Vector{Int},
             dst_start::Vector{Int}, dst_count::Vector{Int})
    -> Union{Nothing, NamedTuple{(:src_offset, :dst_offset, :extent)}}
```

Pure index math: no I/O, no MPI, no globals. Per dimension the source occupies
`[src_start, src_start + src_count)` and the destination `[dst_start, dst_start +
dst_count)`; the intersection gives an offset into each and a shared extent.
Returns `nothing` when any dimension fails to intersect. All offsets 0-based, to
match the on-disk attributes.

```julia
struct SlabSource
    files::Vector{String}
    layout::Dict{String, Vector{NamedTuple{(:file, :start, :count)}}}  # per variable
    global_shape::Dict{String, Vector{Int}}
end

open_slab_source(path::AbstractString) -> SlabSource
```

Accepts either a single `.nc` file or a directory of `*_p*.nc`. Reads each
variable's `start`/`count`/`global_shape` attributes. A single file written by
serial `save_state` carries the same attributes with `start = 0` and
`count == global_shape`, so both cases take one code path.

```julia
read_local_slab!(dest::Array, src::SlabSource, var::AbstractString,
                 dst_start::Vector{Int}) -> Array
```

For each source slab, `slab_overlap` against `dst_start`/`size(dest)`; on a hit,
`ncread(file, var; start=..., count=...)` and copy into `dest`. Reads only the
intersecting region, never a whole file, never the whole global array.

```julia
write_local_slab(path::AbstractString, var::AbstractString, data::Array,
                 local_start::Vector{Int}, global_shape::Vector{Int})
```

`nccreate` then `ncwrite`, then `ncputatt` with `start`/`count`/`global_shape`.
`nccreate` is the call whose absence is the `save_field` bug.

### Layer 2 — field I/O

`save_field` and `load_field!` keep their signatures. `load_field!` gains the
rank-count-portable path; the rank-0-gather-and-scatter implementation goes away.

Each rank obtains its own `local_start` / `local_shape` from the existing helpers
`get_local_start` / `get_local_shape` (`netcdf_output.jl:1648-1700`), which already
serve the writer, so reader and writer cannot drift apart.

**`save_field`'s on-disk format changes under MPI**: it gathered to one file on
rank 0, and will now write per-rank slabs under the same rule as `save_state`
(below). No working behaviour is lost — `save_field` throws on every call today —
but the change is deliberate and should be noted in the release notes.

### Layer 3 — solver checkpoint

```julia
save_state(solver, path::AbstractString) -> String
load_state!(solver, path::AbstractString) -> solver
```

Writes every field in `solver.state` under its `.name` (a vector variable appears
as its components `u_x`, `u_z`, …), plus global attributes `sim_time`,
`iteration`, `dt` (`solver_types.jl:211,212,219`). `load_state!` restores fields
and clock, then re-syncs the solver's field handles.

On-disk naming, used by both `save_state` and `save_field`:

```
np == 1     chk.nc
np  > 1     chk/chk_p0.nc, chk/chk_p1.nc, …      one file per rank
```

`load_state!` accepts either, and accepts a directory written at any rank count.
`path` is given without the `.nc` suffix or with it; the suffix is normalised.

Writes into `solver.state`, **not** the problem-variable handles: those are separate
objects, and writing to them does not restore the integrator.

## Data flow

**Write.** `ensure_layout!(:g)` — grid data is real, so the round trip is lossless
— then each rank computes its own `local_start`/`local_shape` and writes its slab.
No gather, no collective, no rank-0 memory ceiling.

**Read.** Each rank computes the range it needs **from its own field**, not from
the file. `open_slab_source` scans the file set; `slab_overlap` selects the
intersecting regions; the hyperslabs assemble into a host buffer.

**GPU.** NetCDF cannot read into device memory, so the host buffer is inherent to
file I/O. The device field is then filled by **one explicit `copyto!` upload**,
commented as deliberate one-shot I/O staging — the same precedent as the G1
`gather_alg_F` fix. This is not a silent CPU fallback and must not be read as one
by a future no-fallback audit.

## Error handling

- **Rank-uniform collectives.** Every failure decision is `Allreduce`d before any
  rank throws. The current `load_field!` is the cautionary example twice over: it
  broadcasts an ok-flag and then calls `rethrow` outside a catch block, destroying
  the error it meant to propagate. Use `throw(err)`, not `rethrow(err)`.
- **Coverage assertion.** After the overlap pass every destination element must be
  covered by exactly one source region. Partial coverage silently leaves zeros —
  this repo's dominant bug class. Hard error naming the uncovered index range.
- **Shape mismatch.** File `global_shape` ≠ field global shape → error printing both.
- **Missing variable.** Error naming the variable and the files scanned.

## Restart fidelity

A one-step scheme (RK111/222/443, DiagonalIMEX) restarts exactly.

A multistep scheme (CNAB1/2, SBDF1-4) stores history the checkpoint does not carry,
so on load it re-seeds: SBDF4 falls back to RK443 seeding for 3 steps. The result is
correct but not bit-identical to an uninterrupted run. `load_state!` warns, naming
the scheme and the number of reduced-order steps.

Serializing the history was considered and rejected: the three multistep paths store
it in three different shapes — global `ComplexF64` vectors, per-subproblem buffers,
and field-state deques — so the format would have to cover and version all three.

## Testing

| Level | Assertion |
|---|---|
| unit | `slab_overlap`: even splits, uneven splits, non-overlapping, full containment, single rank. No I/O, no MPI. |
| serial | `save_field` / `load_field!` round-trip. Both currently throw. |
| serial | `save_state` / `load_state!` round-trip; `sim_time`, `iteration`, `dt` restored. |
| serial | Error paths: missing variable, shape mismatch, partial coverage — each throws with its own message. |
| MPI | Write @4 ranks, read back @4, @2, @1. Every result must equal the serial reference bit-for-bit. **This is the assertion that proves rank-count portability.** |
| MPI | Read a real `NetCDFFileHandler` output directory, not only our own writer — pins the attribute contract against the live writer. |
| GPU | JLArray device stack: load into a device field, assert the data lands on-device and matches. Grid layout only, so no FFT is required. |

New test files register in `test/file_lists.jl`: serial tests in `SERIAL_TEST_FILES`,
MPI tests in the MPI list.

## Files touched

| File | Change |
|---|---|
| `src/tools/netcdf_slab_io.jl` | new |
| `src/tools/load_output.jl` | include the new module |
| `src/core/solvers/solver_checkpoint.jl` | new |
| `src/core/solvers.jl` | include the new module |
| `src/core/field/field_layout/field_layout_arithmetic_io.jl` | fix both functions, delegate to the slab layer |
| `src/core/solvers/solver_utils.jl` | export `save_state`, `load_state!` alongside `solve!`, `run!` |
| `test/test_slab_io.jl` | new — unit + serial |
| `test/test_checkpoint_restart.jl` | new — serial |
| `test/test_mpi_checkpoint_restart.jl` | new — MPI, the rank-count matrix |
| `test/test_gpu_checkpoint_staging.jl` | new — JLArray |
| `test/file_lists.jl` | register the four |
| `docs/src/api/io.md` | replace the hand-rolled recipe with the real API |
