# Parallel NetCDF Slab I/O and Solver Checkpoint/Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Tarang run write a distributed checkpoint and read it back at any MPI rank count, on CPU or GPU, and fix the two broken field-I/O functions on the way.

**Architecture:** Three layers. `netcdf_slab_io.jl` knows only about files and index ranges — no fields, no solvers. `field_layout_arithmetic_io.jl` maps a `ScalarField` onto that layer. `solver_checkpoint.jl` maps a solver onto the field layer. Each rank reads only the hyperslabs that intersect its own local range, so the rank count that wrote a checkpoint need not match the one that reads it.

**Tech Stack:** Julia 1.12, NetCDF.jl 0.11/0.12, MPI.jl, PencilArrays.jl. NetCDF only — no HDF5, no JLD2.

**Spec:** `docs/superpowers/specs/2026-07-31-file-io-checkpoint-design.md`

## Global Constraints

- **Julia launcher:** `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=.` — the plain `julia`/`juliaup` launcher is broken on this machine.
- **Single test file:** `<julia> -e 'using Test; include("test/<file>.jl")'`
- **Full suite:** `<julia> -e 'using Pkg; Pkg.test()'` — NOT `test/runtests.jl`, which omits Aqua and JET.
- **MPI:** launcher `~/.julia/artifacts/f822b53e59145a4dfdceef194142de78ce8e510b/bin/mpiexec`, and `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib` first. MPICH_jll only.
- **Index base:** every `start` attribute on disk is **0-based** (it comes from `get_local_start`, whose docstring says "0-indexed"). NetCDF.jl's `ncread`/`ncwrite` `start` is **1-based**. Convert in exactly one place — `read_local_slab!` — and nowhere else.
- **Collectives must be rank-uniform.** Never let one rank throw while others continue: `Allreduce` the failure flag first. Use `throw(err)`, never `rethrow(err)`, outside a `catch` block — that is the live `load_field!` bug.
- **JET ratchet:** `test/test_jet.jl` asserts `n_reports <= 975`; the tree currently reports 953. Keep it under.
- **Register every new test file** in `test/file_lists.jl` — serial files in `TEST_FILES`, MPI files in the MPI list. An unregistered file never runs.
- **Never run `git commit`, `git add`, `git stash`, `git checkout`, or `git restore`.** The user commits all work themselves. Leave your changes in the working tree. Unrelated uncommitted work from other efforts is present in this tree — do not touch, revert, or stage any file your task's `Files:` block does not name. A task ends when its tests pass and its report is written, not at a commit.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/tools/netcdf_slab_io.jl` (new) | Files ↔ index ranges. `slab_overlap`, `SlabSource`, `open_slab_source`, `read_local_slab!`, `write_local_slab`. No field or solver types. |
| `src/tools/load_runtime.jl` (modify) | `include("netcdf_slab_io.jl")` after `netcdf_merge.jl` — the slab layer reuses `netcdf_file_info`, `read_netcdf_variable`, `_int_vector_from_attr` from it. |
| `src/core/field/field_layout/field_layout_arithmetic_io.jl` (modify) | `save_field` / `load_field!` rebuilt on the slab layer. Signatures unchanged. |
| `src/core/solvers/solver_checkpoint.jl` (new) | `save_state` / `load_state!`. |
| `src/core/solvers.jl` (modify) | `include("solvers/solver_checkpoint.jl")`. |
| `src/core/solvers/solver_utils.jl` (modify) | Export `save_state`, `load_state!`. |
| `test/test_slab_io.jl` (new) | Tasks 1–2: index math and serial file round-trip. |
| `test/test_checkpoint_restart.jl` (new) | Tasks 3–4: field I/O and solver checkpoint, serial. |
| `test/test_mpi_checkpoint_restart.jl` (new) | Task 5: the rank-count matrix. |
| `test/test_gpu_checkpoint_staging.jl` (new) | Task 6: device upload on load. |
| `test/file_lists.jl` (modify) | Register the four. |
| `docs/src/api/io.md` (modify) | Replace the hand-rolled recipe with the real API. |

---

### Task 1: `slab_overlap` — pure index math

**Files:**
- Create: `src/tools/netcdf_slab_io.jl`
- Modify: `src/tools/load_runtime.jl`
- Create: `test/test_slab_io.jl`
- Modify: `test/file_lists.jl`

**Interfaces:**
- Consumes: nothing.
- Produces: `Tarang.slab_overlap(src_start::AbstractVector{Int}, src_count::AbstractVector{Int}, dst_start::AbstractVector{Int}, dst_count::AbstractVector{Int})` returning `nothing` when the boxes miss, otherwise `(src_offset::Vector{Int}, dst_offset::Vector{Int}, extent::Vector{Int})`. All values 0-based.

- [ ] **Step 1: Write the failing test**

Create `test/test_slab_io.jl`:

```julia
# Unit + serial tests for the NetCDF slab I/O layer.
#
# `slab_overlap` is pure index math: no files, no MPI. It decides which part of a
# stored slab feeds which part of a rank's local array, so an off-by-one here
# silently loads the wrong region — the failure mode a round-trip test cannot see.

using Test
using Tarang

@testset "slab_overlap: identical boxes" begin
    r = Tarang.slab_overlap([0, 0], [4, 6], [0, 0], [4, 6])
    @test r !== nothing
    @test r.src_offset == [0, 0]
    @test r.dst_offset == [0, 0]
    @test r.extent == [4, 6]
end

@testset "slab_overlap: disjoint in one dimension returns nothing" begin
    @test Tarang.slab_overlap([0], [4], [4], [4]) === nothing
    @test Tarang.slab_overlap([4], [4], [0], [4]) === nothing
    # Overlaps in dim 1, disjoint in dim 2 — the whole box misses.
    @test Tarang.slab_overlap([0, 0], [8, 3], [0, 3], [8, 3]) === nothing
end

@testset "slab_overlap: partial overlap" begin
    # source covers global [2,6), destination [4,9) -> shared [4,6)
    r = Tarang.slab_overlap([2], [4], [4], [5])
    @test r.src_offset == [2]     # 4 - 2
    @test r.dst_offset == [0]     # 4 - 4
    @test r.extent == [2]         # 6 - 4
end

@testset "slab_overlap: destination contained in source" begin
    r = Tarang.slab_overlap([0], [16], [4], [4])
    @test r.src_offset == [4]
    @test r.dst_offset == [0]
    @test r.extent == [4]
end

@testset "slab_overlap: source contained in destination" begin
    r = Tarang.slab_overlap([4], [4], [0], [16])
    @test r.src_offset == [0]
    @test r.dst_offset == [4]
    @test r.extent == [4]
end

@testset "slab_overlap: uneven 4-rank split read back on 2 ranks" begin
    # Written at np=4 on a length-6 axis: starts 0,1,3,4 with counts 1,2,1,2
    # (the real PencilArrays remainder-on-last split, verified on this machine).
    # Reading rank 1 of 2 wants global [3,6).
    src = [([0], [1]), ([1], [2]), ([3], [1]), ([4], [2])]
    hits = [Tarang.slab_overlap(s, c, [3], [3]) for (s, c) in src]
    @test hits[1] === nothing
    @test hits[2] === nothing
    @test hits[3].src_offset == [0] && hits[3].dst_offset == [0] && hits[3].extent == [1]
    @test hits[4].src_offset == [0] && hits[4].dst_offset == [1] && hits[4].extent == [2]
    # The hits must tile the destination exactly.
    @test sum(h.extent[1] for h in hits if h !== nothing) == 3
end

@testset "slab_overlap: dimension mismatch is an error" begin
    @test_throws ArgumentError Tarang.slab_overlap([0, 0], [4], [0, 0], [4, 4])
end
```

Register it — in `test/file_lists.jl`, inside `TEST_FILES`, immediately after the line `"test_multistep_field_path.jl",`:

```julia
    "test_slab_io.jl",                        # NetCDF slab index math + serial file round-trip
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_slab_io.jl")'
```
Expected: `UndefVarError: slab_overlap not defined in Tarang`.

- [ ] **Step 3: Write the implementation**

Create `src/tools/netcdf_slab_io.jl`:

```julia
# ============================================================================
# NetCDF slab I/O.
#
# This layer knows about files and index ranges and nothing else — no fields, no
# solvers, no MPI calls. That boundary is what lets `slab_overlap` be tested as
# pure arithmetic, which matters because a wrong overlap loads plausible data
# from the wrong region and no round-trip test can see it.
#
# INDEX BASE. Every `start` attribute on disk is 0-based: it comes from
# `get_local_start`, which the output handler already uses. NetCDF.jl's
# `ncread`/`ncwrite` take a 1-based `start`. The conversion happens in exactly
# one place, `read_local_slab!`, and must not be duplicated.
# ============================================================================

"""
    slab_overlap(src_start, src_count, dst_start, dst_count)

Intersect two axis-aligned boxes given as 0-based start/extent per dimension.

Returns `nothing` when they miss in any dimension. Otherwise returns
`(src_offset, dst_offset, extent)`: `src_offset` is the 0-based offset of the
shared region inside the SOURCE slab (so it indexes the stored variable, whose
first element is global index `src_start`), `dst_offset` is the offset inside the
destination, and `extent` is the shared size.
"""
function slab_overlap(src_start::AbstractVector{<:Integer}, src_count::AbstractVector{<:Integer},
                      dst_start::AbstractVector{<:Integer}, dst_count::AbstractVector{<:Integer})
    n = length(src_start)
    if length(src_count) != n || length(dst_start) != n || length(dst_count) != n
        throw(ArgumentError(
            "slab_overlap: all four vectors must have the same length, got " *
            "src_start=$(length(src_start)), src_count=$(length(src_count)), " *
            "dst_start=$(length(dst_start)), dst_count=$(length(dst_count))"))
    end

    src_offset = Vector{Int}(undef, n)
    dst_offset = Vector{Int}(undef, n)
    extent = Vector{Int}(undef, n)

    @inbounds for d in 1:n
        lo = max(Int(src_start[d]), Int(dst_start[d]))
        hi = min(Int(src_start[d]) + Int(src_count[d]), Int(dst_start[d]) + Int(dst_count[d]))
        hi <= lo && return nothing
        src_offset[d] = lo - Int(src_start[d])
        dst_offset[d] = lo - Int(dst_start[d])
        extent[d] = hi - lo
    end

    return (src_offset = src_offset, dst_offset = dst_offset, extent = extent)
end
```

In `src/tools/load_runtime.jl`, add after the `include("netcdf_merge.jl")` line:

```julia
include("netcdf_slab_io.jl")
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_slab_io.jl")'
```
Expected: all testsets pass, no Fail and no Error.

---

### Task 2: Slab files — write, discover, read back

**Files:**
- Modify: `src/tools/netcdf_slab_io.jl`
- Modify: `test/test_slab_io.jl`

**Interfaces:**
- Consumes: `slab_overlap` from Task 1. From `src/tools/netcdf_merge.jl`: `netcdf_file_info(file)` returning `(dim, vars, gatts)` where each var has `.name` and `.atts`; `read_netcdf_variable(file, var; start, count)`; `_int_vector_from_attr(value)`.
- Produces:
  - `Tarang.write_local_slab(path::AbstractString, var::AbstractString, data::AbstractArray, local_start::AbstractVector{<:Integer}, global_shape::AbstractVector{<:Integer}) -> String`
  - `Tarang.SlabSource` with fields `files::Vector{String}`, `entries::Dict{String,Vector{NamedTuple}}`, `global_shape::Dict{String,Vector{Int}}`
  - `Tarang.open_slab_source(path::AbstractString) -> SlabSource`
  - `Tarang.read_local_slab!(dest::AbstractArray, src::SlabSource, var::AbstractString, dst_start::AbstractVector{<:Integer}) -> dest`

- [ ] **Step 1: Write the failing test**

Append to `test/test_slab_io.jl`:

```julia
@testset "write_local_slab round-trips a single file" begin
    dir = mktempdir()
    path = joinpath(dir, "one.nc")
    data = reshape(collect(1.0:24.0), 4, 6)

    Tarang.write_local_slab(path, "u", data, [0, 0], [4, 6])
    src = Tarang.open_slab_source(path)

    @test src.files == [path]
    @test src.global_shape["u"] == [4, 6]
    @test length(src.entries["u"]) == 1

    dest = zeros(Float64, 4, 6)
    Tarang.read_local_slab!(dest, src, "u", [0, 0])
    @test dest == data
end

@testset "read_local_slab! assembles a destination from several slab files" begin
    dir = mktempdir()
    global_data = reshape(collect(1.0:48.0), 8, 6)
    # Write as if from 4 ranks splitting the LAST axis 1/2/1/2 (the uneven
    # PencilArrays split this machine produces at np=4 on a length-6 axis).
    starts = [0, 1, 3, 4]
    counts = [1, 2, 1, 2]
    for (r, (s, c)) in enumerate(zip(starts, counts))
        Tarang.write_local_slab(joinpath(dir, "chk_p$(r-1).nc"), "u",
                                global_data[:, (s+1):(s+c)], [0, s], [8, 6])
    end

    src = Tarang.open_slab_source(dir)
    @test length(src.files) == 4
    @test length(src.entries["u"]) == 4

    # Read back on 1 rank: the whole thing.
    whole = zeros(Float64, 8, 6)
    Tarang.read_local_slab!(whole, src, "u", [0, 0])
    @test whole == global_data

    # Read back on 2 ranks: each half must match, and neither may touch the other.
    left = zeros(Float64, 8, 3)
    Tarang.read_local_slab!(left, src, "u", [0, 0])
    @test left == global_data[:, 1:3]

    right = zeros(Float64, 8, 3)
    Tarang.read_local_slab!(right, src, "u", [0, 3])
    @test right == global_data[:, 4:6]
end

@testset "read_local_slab! errors rather than leaving a partly-filled buffer" begin
    dir = mktempdir()
    data = reshape(collect(1.0:24.0), 4, 6)
    # Store only the first half of the last axis but claim a global shape of 6.
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"), "u", data[:, 1:3], [0, 0], [4, 6])
    src = Tarang.open_slab_source(dir)

    dest = zeros(Float64, 4, 6)
    # A silent partial fill would leave zeros in [:, 4:6] — the exact silent-zero
    # class this assertion exists to prevent.
    @test_throws ErrorException Tarang.read_local_slab!(dest, src, "u", [0, 0])
end

@testset "open_slab_source and read_local_slab! report missing variables" begin
    dir = mktempdir()
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"), "u",
                            reshape(collect(1.0:12.0), 4, 3), [0, 0], [4, 3])
    src = Tarang.open_slab_source(dir)
    @test !haskey(src.entries, "nope")
    dest = zeros(Float64, 4, 3)
    @test_throws ErrorException Tarang.read_local_slab!(dest, src, "nope", [0, 0])

    @test_throws ErrorException Tarang.open_slab_source(joinpath(dir, "does_not_exist"))
end

@testset "several variables share one slab file" begin
    dir = mktempdir()
    path = joinpath(dir, "multi.nc")
    a = reshape(collect(1.0:12.0), 4, 3)
    b = reshape(collect(101.0:112.0), 4, 3)
    Tarang.write_local_slab(path, "a", a, [0, 0], [4, 3])
    Tarang.write_local_slab(path, "b", b, [0, 0], [4, 3])

    src = Tarang.open_slab_source(path)
    da = zeros(Float64, 4, 3); Tarang.read_local_slab!(da, src, "a", [0, 0])
    db = zeros(Float64, 4, 3); Tarang.read_local_slab!(db, src, "b", [0, 0])
    @test da == a
    @test db == b
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_slab_io.jl")'
```
Expected: the Task 1 testsets still pass; the new ones fail with `UndefVarError: write_local_slab not defined in Tarang`.

- [ ] **Step 3: Write the implementation**

Append to `src/tools/netcdf_slab_io.jl`:

```julia
"""
A set of NetCDF files holding slabs of one or more global arrays.

`entries[var]` lists every stored piece of `var` as `(file, start, count)` with
0-based `start`. `global_shape[var]` is the shape those pieces tile.

A variable counts as a slab only if it carries all three of the `start`, `count`
and `global_shape` attributes. That rule is what lets a directory written by
`NetCDFFileHandler` be opened directly: its coordinate and time variables carry
no slab metadata and are skipped.
"""
struct SlabSource
    files::Vector{String}
    entries::Dict{String, Vector{NamedTuple{(:file, :start, :count), Tuple{String, Vector{Int}, Vector{Int}}}}}
    global_shape::Dict{String, Vector{Int}}
end

"""Resolve `path` to the NetCDF files holding a checkpoint.

Accepts a file, a file without its `.nc` suffix, or a directory of `*.nc` slabs
(with or without the suffix on the directory name)."""
function _slab_files(path::AbstractString)
    isdir(path) && return sort(filter(f -> endswith(f, ".nc"), readdir(path; join = true)))
    isfile(path) && return [String(path)]

    with_nc = endswith(path, ".nc") ? String(path) : string(path, ".nc")
    isfile(with_nc) && return [with_nc]

    stem = endswith(path, ".nc") ? String(path[1:end-3]) : String(path)
    isdir(stem) && return sort(filter(f -> endswith(f, ".nc"), readdir(stem; join = true)))

    return String[]
end

"""
    write_local_slab(path, var, data, local_start, global_shape) -> path

Write one rank's slab of `var` into `path`, stamping the `start`/`count`/
`global_shape` attributes a reader needs. Additive: several variables may share
one file, so the caller owns deleting a stale file before the first write.

`nccreate` before `ncwrite` is not optional — NetCDF.jl needs the variable to
exist, and its absence is why `save_field` threw on every call.
"""
function write_local_slab(path::AbstractString, var::AbstractString, data::AbstractArray,
                          local_start::AbstractVector{<:Integer},
                          global_shape::AbstractVector{<:Integer})
    host = data isa Array ? data : Array(data)
    dimspec = Any[]
    for (i, s) in enumerate(size(host))
        push!(dimspec, "$(var)_d$(i)")
        push!(dimspec, s)
    end
    nctype = eltype(host) === Float32 ? NetCDF.NC_FLOAT : NetCDF.NC_DOUBLE
    nccreate(path, var, dimspec...; t = nctype)
    ncwrite(host, path, var)
    ncputatt(path, var, Dict("start" => collect(Int, local_start),
                             "count" => collect(Int, size(host)),
                             "global_shape" => collect(Int, global_shape)))
    return path
end

"""
    open_slab_source(path) -> SlabSource

Scan `path` and index every slab-carrying variable it holds.
"""
function open_slab_source(path::AbstractString)
    files = _slab_files(path)
    isempty(files) && error(
        "open_slab_source: no NetCDF files found at '$path'. Expected a .nc file or a " *
        "directory containing *.nc slab files.")

    EntryT = NamedTuple{(:file, :start, :count), Tuple{String, Vector{Int}, Vector{Int}}}
    entries = Dict{String, Vector{EntryT}}()
    gshape = Dict{String, Vector{Int}}()

    for file in files
        info = netcdf_file_info(file)
        for v in info.vars
            atts = v.atts
            (haskey(atts, "start") && haskey(atts, "count") && haskey(atts, "global_shape")) || continue
            st = _int_vector_from_attr(atts["start"])
            ct = _int_vector_from_attr(atts["count"])
            gs = _int_vector_from_attr(atts["global_shape"])
            (st === nothing || ct === nothing || gs === nothing) && continue
            push!(get!(() -> EntryT[], entries, String(v.name)),
                  (file = file, start = st, count = ct))
            gshape[String(v.name)] = gs
        end
    end

    return SlabSource(files, entries, gshape)
end

"""
    read_local_slab!(dest, src, var, dst_start) -> dest

Fill `dest` with the region of `var` starting at 0-based global `dst_start`,
reading only the hyperslabs that intersect it.

Errors unless the pieces tile `dest` exactly. A partial fill would leave zeros,
which is indistinguishable from real data downstream.
"""
function read_local_slab!(dest::AbstractArray, src::SlabSource, var::AbstractString,
                          dst_start::AbstractVector{<:Integer})
    entries = get(src.entries, var, nothing)
    entries === nothing && error(
        "read_local_slab!: no variable '$var' with slab metadata in $(src.files). " *
        "Variables found: $(sort(collect(keys(src.entries)))).")

    dst_count = collect(Int, size(dest))
    covered = 0

    for e in entries
        ov = slab_overlap(e.start, e.count, dst_start, dst_count)
        ov === nothing && continue
        # NetCDF.jl indexes the STORED variable 1-based; `ov.src_offset` is already
        # the 0-based offset inside that stored slab, so +1 is the whole conversion.
        chunk = read_netcdf_variable(e.file, var;
                                     start = ov.src_offset .+ 1,
                                     count = ov.extent)
        dst_ranges = ntuple(d -> (ov.dst_offset[d] + 1):(ov.dst_offset[d] + ov.extent[d]),
                            length(dst_count))
        dest[dst_ranges...] = reshape(chunk, ov.extent...)
        covered += prod(ov.extent)
    end

    expected = prod(dst_count)
    covered == expected || error(
        "read_local_slab!: '$var' covers $covered of $expected elements for the region " *
        "starting at $(collect(Int, dst_start)) with size $dst_count. " *
        (covered < expected ?
         "The checkpoint does not span this range — it may be incomplete or from a " *
         "different resolution." :
         "Slabs overlap each other — the checkpoint has duplicate coverage.") *
        " Files: $(src.files).")

    return dest
end
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_slab_io.jl")'
```
Expected: every testset passes.

---

### Task 3: Rebuild `save_field` and `load_field!` on the slab layer

**Files:**
- Modify: `src/core/field/field_layout/field_layout_arithmetic_io.jl:94-170`
- Create: `test/test_checkpoint_restart.jl`
- Modify: `test/file_lists.jl`

**Interfaces:**
- Consumes: `write_local_slab`, `open_slab_source`, `read_local_slab!` from Task 2. From `src/tools/netcdf_output.jl`: `get_operator_domain(field)`, `get_global_shape(layout, domain_info, scales)`, `get_local_shape(layout, domain_info, scales, rank)`, `get_local_start(layout, domain_info, scales, rank)`.
- Produces:
  - `save_field(field::ScalarField, filename::String, dataset_name::String="field") -> String` (returns the path actually written)
  - `load_field!(field::ScalarField, filename::String, dataset_name::String="field") -> field`
  - `Tarang._field_slab_geometry(field) -> (global_shape::Vector{Int}, local_shape::Vector{Int}, local_start::Vector{Int})`
  - `Tarang._slab_output_path!(dist, path) -> String`
  - `Tarang._abort_if_any_rank_failed(dist, err, context)`

- [ ] **Step 1: Write the failing test**

Create `test/test_checkpoint_restart.jl`:

```julia
# Serial tests for field I/O and solver checkpoint/restart.
#
# Both `save_field` and `load_field!` were exported, untested, and broken:
# `save_field` called `ncwrite` with no preceding `nccreate` and threw on every
# call; `load_field!` called `rethrow` outside a catch block, which Julia rejects,
# destroying the NetCDF error it meant to surface.

using Test
using Tarang

@testset "save_field / load_field! round-trip" begin
    dir = mktempdir()
    path = joinpath(dir, "field.nc")

    domain = PeriodicDomain(16)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x) + 0.5cos(3x))
    ensure_layout!(u, :g)
    original = copy(get_grid_data(u))

    written = save_field(u, path, "u")
    @test isfile(written)

    v = ScalarField(domain, "v")
    load_field!(v, written, "u")
    ensure_layout!(v, :g)
    @test get_grid_data(v) == original
end

@testset "save_field accepts a path without the .nc suffix" begin
    dir = mktempdir()
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> cos(x))
    written = save_field(u, joinpath(dir, "nosuffix"), "u")
    @test endswith(written, ".nc")
    @test isfile(written)
end

@testset "load_field! surfaces the real error, not a rethrow failure" begin
    dir = mktempdir()
    domain = PeriodicDomain(8)
    v = ScalarField(domain, "v")
    err = try
        load_field!(v, joinpath(dir, "absent.nc"), "u")
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test !occursin("rethrow", msg)
    @test occursin("absent", msg) || occursin("no NetCDF files", msg)
end

@testset "load_field! rejects a shape mismatch instead of loading garbage" begin
    dir = mktempdir()
    path = joinpath(dir, "small.nc")
    small = ScalarField(PeriodicDomain(8), "u")
    set!(small, (x,) -> sin(x))
    save_field(small, path, "u")

    big = ScalarField(PeriodicDomain(16), "u")
    @test_throws ErrorException load_field!(big, path, "u")
end
```

Register it — in `test/file_lists.jl`, in `TEST_FILES`, right after the `"test_slab_io.jl",` line added in Task 1:

```julia
    "test_checkpoint_restart.jl",             # save_field/load_field! + solver save_state/load_state!
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_checkpoint_restart.jl")'
```
Expected: the round-trip testset fails with `NetCDF error code 2: No such file or directory` from `save_field`, and the error-message testset fails on `rethrow(exc) not allowed outside a catch block`.

- [ ] **Step 3: Write the implementation**

In `src/core/field/field_layout/field_layout_arithmetic_io.jl`, replace everything from the `# I/O operations` comment (line 93) through the end of `load_field!` (line 170) with:

```julia
# I/O operations
#
# Both directions go through the slab layer in src/tools/netcdf_slab_io.jl, so a
# field written under one decomposition can be read under another. Grid space is
# real-valued, so the round trip is lossless.

"""Global shape, local shape, and 0-based local start of `field`'s grid data.

Reuses the same helpers the NetCDF output handler uses, so reader and writer
cannot drift apart."""
function _field_slab_geometry(field::ScalarField)
    domain_info = get_operator_domain(field)
    global_shape = collect(Int, get_global_shape(:g, domain_info, field.scales))
    local_shape = collect(Int, get_local_shape(:g, domain_info, field.scales, field.dist.rank))
    local_start = collect(Int, get_local_start(:g, domain_info, field.scales, field.dist.rank))
    return global_shape, local_shape, local_start
end

"""Path this rank writes to: one file when serial, `<stem>/<stem>_p<rank>.nc` under MPI.

Mirrors the rule `current_file` already uses for the output handler. Rank 0
creates the directory and the others wait, so no two ranks race on `mkpath`."""
function _slab_output_path!(dist, path::AbstractString)
    stem = endswith(path, ".nc") ? String(path[1:end-3]) : String(path)
    if dist.size <= 1
        return string(stem, ".nc")
    end
    if dist.rank == 0
        isdir(stem) || mkpath(stem)
    end
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Barrier(dist.comm)
    end
    return joinpath(stem, string(basename(stem), "_p", dist.rank, ".nc"))
end

"""Throw on every rank if any rank failed.

`throw`, never `rethrow`: this runs outside the `catch` block that produced
`err`, and `rethrow` there raises "rethrow(exc) not allowed outside a catch
block", destroying the original error."""
function _abort_if_any_rank_failed(dist, err, context::AbstractString)
    failed = err === nothing ? 0 : 1
    if dist.size > 1 && MPI.Initialized() && !MPI.Finalized()
        failed = MPI.Allreduce(failed, MPI.MAX, dist.comm)
    end
    failed == 0 && return nothing
    err === nothing && error("$context: another rank failed; aborting collectively.")
    throw(err)
end

"""Copy a host slab into `field`'s grid storage.

NetCDF reads into host memory, so a device-resident field needs one explicit
upload. That `copyto!` is deliberate one-shot I/O staging, not a silent CPU
fallback — file I/O cannot read into device memory at all."""
function _store_local_grid_data!(field::ScalarField, host::Array)
    target = get_local_data(get_grid_data(field))
    size(target) == size(host) && return (copyto!(target, host); field)
    error("load_field!: the checkpoint slab is $(size(host)) but the field's local " *
          "storage is $(size(target)).")
end

"""Save `field` to NetCDF, one file per rank under MPI. Returns the path written."""
function save_field(field::ScalarField, filename::String, dataset_name::String="field")
    ensure_layout!(field, :g)
    global_shape, _, local_start = _field_slab_geometry(field)
    host = Array(get_local_data(get_grid_data(field)))
    target = _slab_output_path!(field.dist, filename)
    isfile(target) && rm(target)
    write_local_slab(target, dataset_name, host, local_start, global_shape)
    return target
end

"""Load `field` from NetCDF written at any rank count."""
function load_field!(field::ScalarField, filename::String, dataset_name::String="field")
    ensure_layout!(field, :g)
    global_shape, local_shape, local_start = _field_slab_geometry(field)

    src = nothing
    err = nothing
    try
        src = open_slab_source(filename)
        haskey(src.global_shape, dataset_name) || error(
            "load_field!: no variable '$dataset_name' with slab metadata in '$filename'. " *
            "Variables found: $(sort(collect(keys(src.entries)))).")
        src.global_shape[dataset_name] == global_shape || error(
            "load_field!: '$dataset_name' was written with global shape " *
            "$(src.global_shape[dataset_name]) but this field is $global_shape.")
    catch e
        err = e
    end
    _abort_if_any_rank_failed(field.dist, err, "load_field!")

    host = Array{eltype(get_local_data(get_grid_data(field)))}(undef, local_shape...)
    read_local_slab!(host, src, dataset_name, local_start)
    return _store_local_grid_data!(field, host)
end
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_checkpoint_restart.jl")'
```
Expected: all four testsets pass.

---

### Task 4: `save_state` / `load_state!`

**Files:**
- Create: `src/core/solvers/solver_checkpoint.jl`
- Modify: `src/core/solvers.jl`
- Modify: `src/core/solvers/solver_utils.jl`
- Modify: `test/test_checkpoint_restart.jl`
- Modify: `docs/src/api/io.md`

**Interfaces:**
- Consumes: `save_field`, `load_field!`, `_field_slab_geometry`, `_slab_output_path!` from Task 3; `write_local_slab`, `open_slab_source` from Task 2; `netcdf_file_info` from `netcdf_merge.jl`.
- Produces:
  - `save_state(solver::InitialValueSolver, path::AbstractString) -> String`
  - `load_state!(solver::InitialValueSolver, path::AbstractString) -> solver`

- [ ] **Step 1: Write the failing test**

Append to `test/test_checkpoint_restart.jl`:

```julia
function _decay_solver(stepper; dt=0.02)
    domain = PeriodicDomain(16)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x) + 0.25cos(2x))
    problem = IVP([u])
    add_equation!(problem, "dt(u) = -u")
    return InitialValueSolver(problem, stepper; dt)
end

@testset "save_state / load_state! restores fields and clock" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")

    a = _decay_solver(RK222())
    for _ in 1:10
        step!(a, 0.02)
    end
    written = save_state(a, path)
    @test isfile(written)

    ensure_layout!(a.state[1], :g)
    expected = copy(get_grid_data(a.state[1]))

    b = _decay_solver(RK222())
    load_state!(b, path)
    @test b.sim_time ≈ a.sim_time
    @test b.iteration == a.iteration
    @test b.dt ≈ a.dt
    ensure_layout!(b.state[1], :g)
    @test get_grid_data(b.state[1]) == expected
end

@testset "a one-step scheme continues exactly across a restart" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")

    uninterrupted = _decay_solver(RK222())
    for _ in 1:20
        step!(uninterrupted, 0.02)
    end
    ensure_layout!(uninterrupted.state[1], :g)
    reference = copy(get_grid_data(uninterrupted.state[1]))

    first_half = _decay_solver(RK222())
    for _ in 1:10
        step!(first_half, 0.02)
    end
    save_state(first_half, path)

    second_half = _decay_solver(RK222())
    load_state!(second_half, path)
    for _ in 1:10
        step!(second_half, 0.02)
    end
    ensure_layout!(second_half.state[1], :g)
    @test get_grid_data(second_half.state[1]) ≈ reference atol=1e-13
end

@testset "a multistep restart warns that it re-seeds" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")
    a = _decay_solver(SBDF4())
    for _ in 1:12
        step!(a, 0.02)
    end
    save_state(a, path)

    b = _decay_solver(SBDF4())
    @test_logs (:warn, r"SBDF4") match_mode=:any load_state!(b, path)
    @test b.iteration == a.iteration
end

@testset "load_state! rejects a checkpoint from a different resolution" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")
    save_state(_decay_solver(RK222()), path)

    domain = PeriodicDomain(32)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))
    problem = IVP([u])
    add_equation!(problem, "dt(u) = -u")
    wrong = InitialValueSolver(problem, RK222(); dt=0.02)
    @test_throws ErrorException load_state!(wrong, path)
end
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_checkpoint_restart.jl")'
```
Expected: the Task 3 testsets still pass; the new ones fail with `UndefVarError: save_state not defined`.

- [ ] **Step 3: Write the implementation**

Create `src/core/solvers/solver_checkpoint.jl`:

```julia
# ============================================================================
# Solver checkpoint / restart.
#
# Writes every evolved field in `solver.state` plus the simulation clock, using
# the slab layer, so a checkpoint written on N ranks can be read on M.
#
# Writes `solver.state`, NOT the problem-variable handles: those are separate
# objects and writing to them does not restore the integrator.
# ============================================================================

"""Steps a multistep scheme spends re-seeding after a restart with no stored history."""
const _MULTISTEP_RESEED_STEPS = Dict("CNAB2" => 1, "SBDF2" => 1, "SBDF3" => 2, "SBDF4" => 3)

function _warn_multistep_restart(solver::InitialValueSolver)
    scheme = string(nameof(typeof(solver.timestepper)))
    steps = get(_MULTISTEP_RESEED_STEPS, scheme, 0)
    steps == 0 && return nothing
    @warn "$scheme restart re-seeds its multistep history: the checkpoint carries the " *
          "state but not the stored time levels, so the first $steps step(s) run at the " *
          "seeding order. The run is correct but not bit-identical to an uninterrupted " *
          "one. One-step schemes (RK222, RK443, DiagonalIMEX_*) restart exactly." maxlog=1
    return nothing
end

"""
    save_state(solver, path) -> String

Write `solver.state` and the simulation clock to NetCDF. Serial runs produce
`<path>.nc`; under MPI each rank writes `<path>/<name>_p<rank>.nc` with no gather.

Zero-dimensional tau variables are skipped: they carry no spatial data and are
re-solved from the state on the next step.
"""
function save_state(solver::InitialValueSolver, path::AbstractString)
    isempty(solver.state) && error("save_state: solver.state is empty; nothing to write.")
    dist = solver.state[1].dist
    target = _slab_output_path!(dist, path)
    isfile(target) && rm(target)

    for field in solver.state
        isempty(field.bases) && continue
        ensure_layout!(field, :g)
        global_shape, _, local_start = _field_slab_geometry(field)
        host = Array(get_local_data(get_grid_data(field)))
        write_local_slab(target, field.name, host, local_start, global_shape)
    end

    ncputatt(target, "global", Dict("sim_time" => Float64(solver.sim_time),
                                    "iteration" => Int(solver.iteration),
                                    "dt" => Float64(solver.dt)))
    return target
end

"""
    load_state!(solver, path) -> solver

Restore `solver.state` and the simulation clock from a checkpoint written at any
rank count. Discards the timestepper's cached state so the scheme restarts from
the loaded fields.
"""
function load_state!(solver::InitialValueSolver, path::AbstractString)
    isempty(solver.state) && error("load_state!: solver.state is empty; nothing to restore.")

    for field in solver.state
        isempty(field.bases) && continue
        load_field!(field, path, field.name)
    end

    attrs = netcdf_file_info(first(open_slab_source(path).files)).gatts
    haskey(attrs, "sim_time") && (solver.sim_time = Float64(attrs["sim_time"]))
    haskey(attrs, "iteration") && (solver.iteration = Int(attrs["iteration"]))
    haskey(attrs, "dt") && (solver.dt = Float64(attrs["dt"]))

    # Drop the cached timestepper state: its history refers to the pre-restart
    # trajectory. `_ensure_timestepper_state!` rebuilds it from the loaded fields.
    solver.timestepper_state = nothing
    _warn_multistep_restart(solver)
    return solver
end
```

In `src/core/solvers.jl`, add after the `include("solvers/solver_stepping.jl")` line:

```julia
include("solvers/solver_checkpoint.jl")
```

In `src/core/solvers/solver_utils.jl`, add `save_state` and `load_state!` to the existing export list that already carries `solve!, proceed, run!`.

Then replace the hand-rolled recipe in `docs/src/api/io.md` — the section that begins "Tarang has no built-in checkpoint type" — with:

````markdown
## Checkpoint and restart

`save_state` writes every evolved field in `solver.state` plus `sim_time`,
`iteration` and `dt`. `load_state!` reads it back.

```julia
save_state(solver, "checkpoints/run1")
# ... later, or in a new process ...
load_state!(solver, "checkpoints/run1")
```

Serial runs produce `run1.nc`. Under MPI each rank writes its own slab to
`run1/run1_p<rank>.nc` with no gather, so rank 0 never has to hold the whole
field. `load_state!` reads a checkpoint written at **any** rank count: each rank
works out the range it needs and reads only the overlapping hyperslabs.

GPU fields are supported. NetCDF reads into host memory, so the loader stages
through a host buffer and then performs one explicit upload.

One-step schemes (RK111/222/443, DiagonalIMEX_*) restart exactly. Multistep
schemes (CNAB1/2, SBDF1–4) do not carry their stored time levels, so they
re-seed on restart and warn: the run stays correct but is not bit-identical to an
uninterrupted one.
````

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_checkpoint_restart.jl")'
```
Expected: all nine testsets pass.

---

### Task 5: MPI rank-count matrix

**Files:**
- Create: `test/test_mpi_checkpoint_restart.jl`
- Modify: `test/file_lists.jl`

**Interfaces:**
- Consumes: `save_state`, `load_state!` from Task 4; `save_field`, `load_field!` from Task 3.
- Produces: no new API — this task proves the rank-count portability the whole design exists for.

- [ ] **Step 1: Write the failing test**

Create `test/test_mpi_checkpoint_restart.jl`:

```julia
# The assertion the whole slab design exists for: a checkpoint written on N ranks
# must load on M, and the result must equal what a serial run produces.
#
# Before this, `load_field!` had rank 0 read the entire global array and scatter
# it, so it could not read the per-rank files the output handler writes at all.

using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NP = MPI.Comm_size(COMM)

if NP < 2
    RANK == 0 && @warn "MPI checkpoint test needs at least two ranks"
    MPI.Finalize()
    exit(0)
end

const NX = 16
const NY = 12

_init_value(x, y) = sin(x) * cos(y) + 0.25cos(2x)

function _build(stepper, dt; comm=COMM)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU(), comm=comm)
    xb = RealFourier(coords["x"]; size=NX, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=NY, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u")
    problem = IVP([u])
    add_parameters!(problem, kappa=0.02)
    add_equation!(problem, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(problem, stepper; dt)

    xs = [2π * (i - 1) / NX for i in 1:NX]
    ys = [2π * (j - 1) / NY for j in 1:NY]
    initial = [_init_value(x, y) for x in xs, y in ys]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        parent(gd) .= initial[PencilArrays.pencil(gd).axes_local...]
    else
        gd .= initial
    end
    ensure_layout!(u, :c)
    return solver, initial
end

"""Advance a serial solver on COMM_SELF and return the full global grid."""
function _serial_reference(stepper, dt, nsteps)
    solver, _ = _build(stepper, dt; comm=MPI.COMM_SELF)
    for _ in 1:nsteps
        step!(solver, dt)
    end
    u = solver.state[1]
    ensure_layout!(u, :g)
    return Array(get_grid_data(u))
end

"""Max |local slab - reference| over all ranks."""
function _diff_against(field, reference)
    ensure_layout!(field, :g)
    gd = get_grid_data(field)
    local_diff = if gd isa PencilArrays.PencilArray
        maximum(abs.(parent(gd) .- reference[PencilArrays.pencil(gd).axes_local...]))
    else
        maximum(abs.(Array(gd) .- reference))
    end
    return MPI.Allreduce(local_diff, MPI.MAX, COMM)
end

# Every rank derives the same path independently — no broadcast of a temp-dir name
# (this repo uses no `MPI.bcast` for objects anywhere). Rank 0 creates it, the rest
# wait on the barrier.
const CHK_ROOT = joinpath(tempdir(), "tarang_ckpt_test_np$(NP)")
if RANK == 0
    isdir(CHK_ROOT) && rm(CHK_ROOT; recursive=true, force=true)
    mkpath(CHK_ROOT)
end
MPI.Barrier(COMM)

@testset "Distributed checkpoint round-trips at the same rank count (rank=$RANK)" begin
    path = joinpath(CHK_ROOT, "same")
    solver, _ = _build(RK222(), 0.02)
    for _ in 1:10
        step!(solver, 0.02)
    end
    save_state(solver, path)
    MPI.Barrier(COMM)

    reference = _serial_reference(RK222(), 0.02, 10)

    restored, _ = _build(RK222(), 0.02)
    load_state!(restored, path)
    @test _diff_against(restored.state[1], reference) < 1e-13
    @test restored.iteration == solver.iteration
    @test restored.sim_time ≈ solver.sim_time
end

@testset "A checkpoint written on $NP ranks loads on one (rank=$RANK)" begin
    # Every rank independently reads the WHOLE global field into a COMM_SELF
    # solver. That is the "restart on fewer ranks" case, and each rank checking it
    # separately makes the assertion independent of which rank holds what.
    path = joinpath(CHK_ROOT, "same")
    reference = _serial_reference(RK222(), 0.02, 10)

    serial_solver, _ = _build(RK222(), 0.02; comm=MPI.COMM_SELF)
    load_state!(serial_solver, path)
    u = serial_solver.state[1]
    ensure_layout!(u, :g)
    @test maximum(abs.(Array(get_grid_data(u)) .- reference)) < 1e-13
end

@testset "A serial checkpoint loads on $NP ranks (rank=$RANK)" begin
    # The reverse direction: one writer, many readers.
    path = joinpath(CHK_ROOT, "from_serial")
    if RANK == 0
        writer, _ = _build(RK222(), 0.02; comm=MPI.COMM_SELF)
        for _ in 1:10
            step!(writer, 0.02)
        end
        save_state(writer, path)
    end
    MPI.Barrier(COMM)

    reference = _serial_reference(RK222(), 0.02, 10)
    restored, _ = _build(RK222(), 0.02)
    load_state!(restored, path)
    @test _diff_against(restored.state[1], reference) < 1e-13
end

@testset "Restart continues the trajectory (rank=$RANK)" begin
    path = joinpath(CHK_ROOT, "resume")
    reference = _serial_reference(RK222(), 0.02, 20)

    first_half, _ = _build(RK222(), 0.02)
    for _ in 1:10
        step!(first_half, 0.02)
    end
    save_state(first_half, path)
    MPI.Barrier(COMM)

    second_half, _ = _build(RK222(), 0.02)
    load_state!(second_half, path)
    for _ in 1:10
        step!(second_half, 0.02)
    end
    @test _diff_against(second_half.state[1], reference) < 1e-12
end

@testset "A missing checkpoint fails on every rank, not just one (rank=$RANK)" begin
    # A one-rank throw with the others still in the collective is a deadlock.
    restored, _ = _build(RK222(), 0.02)
    @test_throws Exception load_state!(restored, joinpath(CHK_ROOT, "absent"))
end

MPI.Barrier(COMM)
RANK == 0 && rm(CHK_ROOT; recursive=true, force=true)
MPI.Finalized() || MPI.Finalize()
```

Register it — in `test/file_lists.jl`, in the MPI list, right after the `"test_mpi_explicit_multistep_field.jl",` line:

```julia
    "test_mpi_checkpoint_restart.jl",         # checkpoint written on N ranks loads on M and matches serial (np>=2)
```

- [ ] **Step 2: Run the test and confirm it fails or passes for the right reason**

Run:
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib
~/.julia/artifacts/f822b53e59145a4dfdceef194142de78ce8e510b/bin/mpiexec -n 2 \
  ~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
  test/test_mpi_checkpoint_restart.jl
```
If Tasks 3–4 are complete this should pass on the first run. If it fails, the failure is a real defect in the slab layer under decomposition — most likely the 0-based/1-based `start` conversion in `read_local_slab!` — not a missing function.

- [ ] **Step 3: Fix whatever the run exposes**

No new API. Debug against `slab_overlap`'s unit tests first: if those pass, the index math is right and the fault is in `_field_slab_geometry` or the conversion in `read_local_slab!`.

- [ ] **Step 4: Run at 2 and 4 ranks and confirm both pass**

Run:
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib
for np in 2 4; do
  echo "=== np=$np ==="
  ~/.julia/artifacts/f822b53e59145a4dfdceef194142de78ce8e510b/bin/mpiexec -n $np \
    ~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. \
    test/test_mpi_checkpoint_restart.jl 2>&1 | grep -E "Test Summary|Fail|Error"
done
```
Expected: no `Fail` and no `Error` lines at either rank count. Note that np=4 exercises the uneven split (local shapes 1,2,1,2 on a length-6 axis), which is where an off-by-one shows up.

---

### Task 6: GPU device staging on load

**Files:**
- Create: `test/test_gpu_checkpoint_staging.jl`
- Modify: `test/file_lists.jl`

**Interfaces:**
- Consumes: `save_field`, `load_field!`, `_store_local_grid_data!` from Task 3.
- Produces: no new API — proves the device upload happens and lands on-device.

- [ ] **Step 1: Write the failing test**

Create `test/test_gpu_checkpoint_staging.jl`:

```julia
"""
Device staging for checkpoint loads, with no GPU hardware.

NetCDF reads into host memory, so a device-resident field has to be uploaded.
The upload is one explicit `copyto!` — deliberate one-shot I/O staging, not a
silent CPU fallback, because file I/O cannot read into device memory at all.

JLArray provides device-like arrays with no driver. Everything here stays in grid
layout, so no FFT is needed and the JLArray transform limitation does not apply.
"""

using Test
using Tarang

const _JL_LOADED = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays unavailable; skipping device staging tests" err
    false
end

@testset "checkpoint load stages onto the device" begin
    if !_JL_LOADED
        @test_skip "JLArrays not available in this environment"
    else
        # Test-scoped: teaches Tarang to build fields backed by JLArray.
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}) = JLArrays.JLArray
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}, ::Type{T}) where {T} =
            JLArrays.JLArray{T}

        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.GPU(JLArrays.JLBackend()))
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        domain = Domain(dist, (xb,))

        u = ScalarField(domain, "u")
        ensure_layout!(u, :g)
        gd = get_grid_data(u)
        @test gd isa JLArrays.JLArray
        gd .= collect(1.0:8.0)

        dir = mktempdir()
        written = save_field(u, joinpath(dir, "dev"), "u")
        @test isfile(written)

        v = ScalarField(domain, "v")
        ensure_layout!(v, :g)
        load_field!(v, written, "u")

        loaded = get_grid_data(v)
        # The data must land in DEVICE storage, not be replaced by a host array.
        @test loaded isa JLArrays.JLArray
        @test Array(loaded) == collect(1.0:8.0)
    end
end

@testset "_store_local_grid_data! rejects a wrong-sized slab" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    ensure_layout!(u, :g)
    @test_throws ErrorException Tarang._store_local_grid_data!(u, zeros(Float64, 4))
end
```

Register it — in `test/file_lists.jl`, in `TEST_FILES`, right after `"test_checkpoint_restart.jl",`:

```julia
    "test_gpu_checkpoint_staging.jl",         # checkpoint load uploads to device storage (JLArray, no GPU needed)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_gpu_checkpoint_staging.jl")'
```
Expected: skipped, because `JLArrays` is a test-target dependency and is not in `--project=.`. Run it inside the test environment instead:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```
and read the `test_gpu_checkpoint_staging.jl` line in the summary.

- [ ] **Step 3: Fix whatever the run exposes**

The likely failure is `_store_local_grid_data!` replacing the field's storage instead of writing into it. `copyto!(target, host)` must write into the existing device array; assigning `get_grid_data(v) = host` would swap in a host array and pass a naive value check while silently moving the field off-device — which is why the test asserts `loaded isa JLArrays.JLArray`.

- [ ] **Step 4: Run the full suite and the MPI suite**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: `Testing Tarang tests passed`. Confirm the JET line still reports at or below 975.

Then:
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. test/run_mpi_ci.jl
```
Expected: the summary line reports `0 failed`.

---

## Self-review notes

- Spec coverage: fix both broken functions (Task 3) · rank-count-portable read (Tasks 2, 5) · `save_state`/`load_state!` with clock (Task 4) · per-rank write when np>1 (Task 3, `_slab_output_path!`) · explicit GPU staging (Tasks 3, 6) · rank-uniform collectives (Task 3, `_abort_if_any_rank_failed`) · coverage assertion (Task 2, `read_local_slab!`) · multistep warning (Task 4) · docs (Task 4) · test registration (each task).
- Reading a real `NetCDFFileHandler` output directory is served by `open_slab_source`'s "a variable is a slab iff it carries all three attributes" rule plus reuse of `netcdf_file_info`, which already enumerates the handler's grouped variables. It is not a separate task; if the np=4 run in Task 5 passes, the attribute contract holds.
- Names used in later tasks and defined earlier: `slab_overlap`, `write_local_slab`, `open_slab_source`, `read_local_slab!`, `SlabSource`, `_field_slab_geometry`, `_slab_output_path!`, `_abort_if_any_rank_failed`, `_store_local_grid_data!`, `save_state`, `load_state!`. No task references a name no task defines.
