"""
    Field NetCDF I/O

`save_field` / `load_field!`, plus the slab-geometry and collective-exit helpers
they share with `solver_checkpoint.jl`.

WHY THIS IS NOT UNDER `core/`. It was, as the second half of
`field/field_layout/field_layout_arithmetic_io.jl` — a file whose own name
admitted it held two concerns. That put `core` code in front of the NetCDF slab
layer (`netcdf_slab_io.jl`, `netcdf_merge.jl`), both of which load AFTER the
whole core stack. The calls worked only because Julia resolves them at run time,
not at include time, so the inverted dependency was invisible: nothing fails, and
`core` simply cannot be lifted into a module without dragging the output layer
with it.

The fix is placement, not indirection. This code reads and writes NetCDF; it
belongs in the layer that owns NetCDF. `test/test_layering.jl` pins the direction
so the next such file is caught when it is written rather than years later.

Both directions go through the slab layer in `netcdf_slab_io.jl`, so a field
written under one decomposition can be read under another. Grid space is
real-valued, so the round trip is lossless.
"""


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
creates the directory and the others wait, so no two ranks race on `mkpath`.

Rank 0's `mkpath` can throw (no permission, disk full, `stem` already exists as
a plain file). If only rank 0 learned about that, it would propagate the
exception and never reach `MPI.Barrier` below, while every other rank is
already blocked inside that barrier — forever. `_abort_if_any_rank_failed`
turns this into every rank throwing together instead of a deadlock. It is
itself a collective (`MPI.Allreduce`), so it must run on EVERY rank — success
or failure — before any rank reaches the `MPI.Barrier` collective: two
different collectives, same order on every rank, so no rank ever reaches
`Barrier` while another is still waiting in `Allreduce`. On failure,
`_abort_if_any_rank_failed` throws on every rank before any of them gets to
`Barrier`, so the barrier is simply never reached in that case — nobody waits
on it.

Rank 0 also removes stale `<stem>_p<r>.nc` for `r >= dist.size`. Writing to a
path a LARGER run already used otherwise leaves that run's high-rank files
behind: a np=4 run writes `p0..p3`, a later np=2 run to the same path rewrites
`p0`/`p1` and leaves `p2`/`p3`, and the checkpoint is ruined at write time
while the failure only surfaces hours later at restart as "has overlapping
slabs". Only `r >= dist.size` is removed, never a blanket wipe: several
variables legitimately share one rank's file (`save_field` called once per
`VectorField` component, `save_state` looping over `solver.state`), and each of
those calls comes back through here."""
function _slab_output_path!(dist, path::AbstractString, context::AbstractString)
    stem = endswith(path, ".nc") ? String(path[1:end-3]) : String(path)
    if dist.size <= 1
        return string(stem, ".nc")
    end

    err = nothing
    if dist.rank == 0
        try
            isdir(stem) || mkpath(stem)
            _remove_stale_rank_files(stem, dist.size)
        catch e
            err = e
        end
    end
    _abort_if_any_rank_failed(dist, err, context)

    if MPI.Initialized() && !MPI.Finalized()
        MPI.Barrier(dist.comm)
    end
    return joinpath(stem, string(basename(stem), "_p", dist.rank, ".nc"))
end

"""Delete `<stem>/<name>_p<r>.nc` for every `r >= nranks`.

Files for `r < nranks` belong to a rank in the current run and are that rank's
own business (`save_state` rewrites its file; `save_field` appends to it). Only
the ranks that no longer exist leave orphans, and an orphan is indistinguishable
from a live slab to a reader."""
function _remove_stale_rank_files(stem::AbstractString, nranks::Integer)
    isdir(stem) || return nothing
    prefix = string(basename(stem), "_p")
    suffix = ".nc"
    for entry in readdir(stem)
        (startswith(entry, prefix) && endswith(entry, suffix)) || continue
        rank_index = tryparse(Int, chop(entry; head = length(prefix), tail = length(suffix)))
        rank_index === nothing && continue
        rank_index >= nranks && rm(joinpath(stem, entry); force = true)
    end
    return nothing
end

"""Throw on every rank if any rank failed.

`throw`, never `rethrow`: this runs outside the `catch` block that produced `err`,
and `rethrow` there raises "rethrow(exc) not allowed outside a catch block",
destroying the original error."""
function _abort_if_any_rank_failed(dist, err, context::AbstractString)
    failed = err === nothing ? 0 : 1
    if dist.size > 1 && MPI.Initialized() && !MPI.Finalized()
        failed = MPI.Allreduce(failed, MPI.MAX, dist.comm)
    end
    failed == 0 && return nothing
    err === nothing && error("$context: another rank failed; aborting collectively.")
    throw(err)
end

"""Run `f()` and settle its outcome collectively; return `f()`'s value on success.

The rank-uniformity invariant belongs to the ENTRY POINT, not to the individual
throw sites inside it. Guarding one throw site at a time is what let three
regions stay exposed through two review rounds: a per-rank filesystem or NetCDF
failure that escapes an unguarded statement leaves that rank unwinding while its
peers walk into the caller's next collective, and the run hangs rather than
fails. `_collectively` wraps a whole fallible body in one `try`, so the exit is
rank-uniform no matter where inside it the failure happened.

COLLECTIVE ORDERING. Exactly one `_abort_if_any_rank_failed` (one
`MPI.Allreduce`) runs per call, on the success path and on the failure path
alike — the body cannot return or throw past it. So every rank performs the same
number of `Allreduce`s in the same order, which is the property that makes the
guard safe rather than merely well-intentioned.

NESTING. `load_state!` wraps its body while calling `load_field!`, which wraps
its own. That nests safely, and safely for the reason that matters: because the
inner `_collectively` makes its own exit rank-uniform, a rank cannot leave the
inner region while another rank is still inside it. All ranks therefore reach
the outer guard having performed the identical sequence of inner `Allreduce`s.
(An "outermost guard only" scheme that suppressed the inner one would be worse,
not better: the rank that threw would race ahead into the NEXT field's
`ensure_layout!` transpose while its peers were still in the failed one.)

SERIAL. With `dist.size <= 1` `_abort_if_any_rank_failed` performs no MPI call
at all, so this degrades to a plain try/throw and needs no initialised MPI.
"""
function _collectively(f, dist, context::AbstractString)
    result = nothing
    err = nothing
    try
        result = f()
    catch e
        err = e
    end
    _abort_if_any_rank_failed(dist, err, context)
    return result
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

"""Save `field` to NetCDF, one file per rank under MPI. Returns the path written.

`write_local_slab` is additive — several variables can share one file (a
`VectorField`'s components, saved one call per component into the same path
under different `dataset_name`s, is exactly this case). `save_field` only
clears the target when clearing is actually required:
- target absent -> write it.
- target present but does not hold `dataset_name` yet -> append; whatever else
  the file already holds is untouched.
- target present and already holds `dataset_name` -> this call is a rewrite of
  that one variable. NetCDF cannot replace a variable in place, so the file
  has to be deleted and rewritten from scratch — safe only when `dataset_name`
  is the only variable it holds. If the file holds others too, deleting it
  would destroy their data, so this throws instead, naming `dataset_name` and
  the variable(s) that would have been lost.

`_slab_output_path!` runs FIRST, before anything else that can fail. It owns
the only collectives inside `save_field` other than the guard's own
(`Allreduce` then `Barrier`, in that fixed order on every rank), so resolving
the path before any per-rank work means no rank can throw its way past them
while its peers are still inside. Everything after it — reading the existing
file's variable list, the "would discard other variables" refusal, and the
write itself (ENOSPC on one node, a corrupt leftover `.nc`) — is per-rank
fallible and sits inside `_collectively`."""
function save_field(field::ScalarField, filename::String, dataset_name::String="field")
    target = _slab_output_path!(field.dist, filename, "save_field")

    return _collectively(field.dist, "save_field") do
        ensure_layout!(field, :g)
        global_shape, _, local_start = _field_slab_geometry(field)
        host = Array(get_local_data(get_grid_data(field)))

        if isfile(target)
            existing = [v.name for v in netcdf_file_info(target).vars]
            if dataset_name in existing
                others = filter(!=(dataset_name), existing)
                if isempty(others)
                    rm(target)
                else
                    error("save_field: '$dataset_name' already exists in '$target' together " *
                          "with other variable(s) $(join(others, ", ")); NetCDF cannot replace " *
                          "a variable in place, and deleting the file would discard those too. " *
                          "Remove '$target' first if you intend to overwrite '$dataset_name'.")
                end
            end
        end

        write_local_slab(target, dataset_name, host, local_start, global_shape)
        target
    end
end

"""Load `field` from NetCDF written at any rank count.

The whole body is one `_collectively` region. `_store_local_grid_data!` is the
reason that has to include the tail and not just the read: it compares
`get_local_shape`'s analytic split against the field's actual storage — two
independent computations of the same quantity, which this repo's 2026-06-23
audit records diverging on non-divisible splits and >=2-D meshes, i.e.
rank-dependently by construction. A throw there used to escape after the last
`Allreduce`, so `load_state!` would move on to the next field and open its
`load_field!` with a collective the failed rank never reaches."""
function load_field!(field::ScalarField, filename::String, dataset_name::String="field")
    return _collectively(field.dist, "load_field!") do
        ensure_layout!(field, :g)
        global_shape, local_shape, local_start = _field_slab_geometry(field)

        src = open_slab_source(filename)
        haskey(src.global_shape, dataset_name) || error(
            "load_field!: no variable '$dataset_name' with slab metadata in '$filename'. " *
            "Variables found: $(sort(collect(keys(src.entries)))).")
        src.global_shape[dataset_name] == global_shape || error(
            "load_field!: '$dataset_name' was written with global shape " *
            "$(src.global_shape[dataset_name]) but this field is $global_shape.")

        host = Array{eltype(get_local_data(get_grid_data(field)))}(undef, local_shape...)
        read_local_slab!(host, src, dataset_name, local_start)
        _store_local_grid_data!(field, host)
    end
end
