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

    for d in 1:n
        lo = max(Int(src_start[d]), Int(dst_start[d]))
        hi = min(Int(src_start[d]) + Int(src_count[d]), Int(dst_start[d]) + Int(dst_count[d]))
        hi <= lo && return nothing
        src_offset[d] = lo - Int(src_start[d])
        dst_offset[d] = lo - Int(dst_start[d])
        extent[d] = hi - lo
    end

    return (src_offset = src_offset, dst_offset = dst_offset, extent = extent)
end

"""
A set of NetCDF files holding slabs of one or more global arrays.

`entries[var]` lists every stored piece of `var` as `(file, start, count,
dim_lengths)` with 0-based `start`. `dim_lengths` is the full shape of the
variable AS STORED, in Julia (column-major) order; it can be longer than
`count` — see `read_local_slab!`. `global_shape[var]` is the shape the pieces
tile.

A variable counts as a slab only if it carries all three of the `start`, `count`
and `global_shape` attributes. That rule is what lets a directory written by
`NetCDFFileHandler` be opened directly: its coordinate and time variables carry
no slab metadata and are skipped.
"""
struct SlabSource
    files::Vector{String}
    entries::Dict{String, Vector{NamedTuple{(:file, :start, :count, :dim_lengths), Tuple{String, Vector{Int}, Vector{Int}, Vector{Int}}}}}
    global_shape::Dict{String, Vector{Int}}
end

"""Resolve `path` to the NetCDF files holding a checkpoint.

Accepts a file, a file without its `.nc` suffix, or a directory of `*.nc` slabs
(with or without the suffix on the directory name)."""
function _slab_files(path::AbstractString)
    stem = endswith(path, ".nc") ? String(path[1:end-3]) : String(path)
    dir = isdir(path) ? String(path) : (isdir(stem) ? stem : nothing)
    with_nc = endswith(path, ".nc") ? String(path) : string(path, ".nc")
    file = (isfile(path) && !isdir(path)) ? String(path) :
           (isfile(with_nc) ? with_nc : nothing)

    dir_files = dir === nothing ? String[] :
                sort(filter(f -> endswith(f, ".nc"), readdir(dir; join = true)))

    if dir !== nothing && file !== nothing
        # Both a slab directory and a single-file checkpoint exist for this
        # stem — one of them is stale (the save paths now clean the other form
        # up, so this state predates that). Pick the newer write, loudly.
        dir_mtime = isempty(dir_files) ? 0.0 : maximum(mtime, dir_files)
        newer = mtime(file) > dir_mtime ? "file" : "directory"
        @warn "Both a slab directory `$dir` and a checkpoint file `$file` exist " *
              "for this path; loading the newer $newer. Remove the stale one to " *
              "silence this warning."
        return mtime(file) > dir_mtime ? [file] : dir_files
    end

    dir !== nothing && return dir_files
    file !== nothing && return [file]
    return String[]
end

"""
    write_local_slab(path, var, data, local_start, global_shape) -> path

Write one rank's slab of `var` into `path`, stamping the `start`/`count`/
`global_shape` attributes a reader needs. Additive: several variables may share
one file, so the caller owns deleting a stale file before the first write.

`nccreate` before `ncwrite` is not optional — NetCDF.jl needs the variable to
exist, and its absence is why `save_field` threw on every call.

Grid-space real data only: NetCDF has no complex type, and this layer does not
invent an on-disk complex layout of its own (unlike `write_task_data!`, which
splits complex data into a leading real/imaginary dimension for a different
file format). A complex `data` throws rather than silently truncating to its
real part or reinterpreting its bytes — the caller must split it first.
"""
function write_local_slab(path::AbstractString, var::AbstractString, data::AbstractArray,
                          local_start::AbstractVector{<:Integer},
                          global_shape::AbstractVector{<:Integer})
    eltype(data) <: Complex && error(
        "write_local_slab: variable '$var' has eltype $(eltype(data)). Slab I/O writes " *
        "grid-space real data only; complex data must be split into real/imaginary " *
        "parts by the caller before calling write_local_slab.")
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

    EntryT = NamedTuple{(:file, :start, :count, :dim_lengths), Tuple{String, Vector{Int}, Vector{Int}, Vector{Int}}}
    entries = Dict{String, Vector{EntryT}}()
    gshape = Dict{String, Vector{Int}}()
    gshape_file = Dict{String, String}()

    for file in files
        info = netcdf_file_info(file)
        for v in info.vars
            atts = v.atts
            (haskey(atts, "start") && haskey(atts, "count") && haskey(atts, "global_shape")) || continue
            st = _int_vector_from_attr(atts["start"])
            ct = _int_vector_from_attr(atts["count"])
            gs = _int_vector_from_attr(atts["global_shape"])
            (st === nothing || ct === nothing || gs === nothing) && continue
            name = String(v.name)
            # Every slab of one variable must agree on the shape they tile.
            # Last-writer-wins would silently reconcile a genuine disagreement
            # (e.g. a leftover file from a run at a different resolution sharing
            # the directory) to whichever file happened to sort last, and the
            # coverage check downstream would then measure against the wrong
            # global array.
            prev = get(gshape, name, nothing)
            if prev !== nothing && prev != gs
                error("open_slab_source: '$name' declares global_shape $prev in " *
                      "'$(gshape_file[name])' but $gs in '$file'. Slabs of one variable " *
                      "must agree on the shape they tile; the directory mixes files from " *
                      "different resolutions or different runs.")
            end
            push!(get!(() -> EntryT[], entries, name),
                  (file = file, start = st, count = ct,
                   dim_lengths = collect(Int, v.dim_lengths)))
            gshape[name] = gs
            gshape_file[name] = file
        end
    end

    return SlabSource(files, entries, gshape)
end

"""1-based `start`/`count` for reading overlap `ov` out of slab entry `e`.

The slab metadata describes the trailing dimensions of the stored variable. Any
leading dimensions it does not describe (`NetCDFFileHandler`'s unlimited
`sim_time` axis) are read at their LAST index with extent 1, so the caller gets
the most recent write and can reshape the singleton away."""
function _slab_read_indices(e, ov, varname::AbstractString)
    lead = length(e.dim_lengths) - length(e.start)
    lead == 0 && return (ov.src_offset .+ 1, collect(Int, ov.extent))
    lead < 0 && error(
        "read_local_slab!: '$varname' in '$(e.file)' is stored with " *
        "$(length(e.dim_lengths)) dimension(s) but its slab metadata describes " *
        "$(length(e.start)). The start/count attributes do not match the variable.")
    return (vcat(e.dim_lengths[1:lead], ov.src_offset .+ 1),
            vcat(fill(1, lead), collect(Int, ov.extent)))
end

"""
    read_local_slab!(dest, src, var, dst_start) -> dest

Fill `dest` with the region of `var` starting at 0-based global `dst_start`,
reading only the hyperslabs that intersect it.

Errors unless the pieces tile `dest` exactly — each element written by exactly
one slab. A boolean coverage mask (not a running element count) is what
enforces this: two slabs that overlap each other inflate a running count by
their overlap, and when that inflation happens to equal a genuine gap
elsewhere, a count-based check passes while part of `dest` is silently left
unwritten. The mask catches both failure modes separately: any element never
marked means the checkpoint does not span the requested range; any element
marked twice means two slabs claim the same region.

TIME-SERIES VARIABLES. `NetCDFFileHandler` creates its data variables with a
leading unlimited `sim_time` dimension, while `build_layout_metadata` stamps
`start`/`count`/`global_shape` covering only the component and spatial dims
(`netcdf_output.jl`). The stored variable is then one rank higher than its own
slab metadata describes. Those extra dimensions are always LEADING in Julia
order (`write_task_data!` writes `reshape(data, 1, size(data)...)`), so this
reads the LAST index along each of them — the most recent write in the file —
and drops the resulting singleton dimensions. Without that padding the
`start`/`count` vectors handed to NetCDF are shorter than the variable's rank,
which the C layer reads past the end of: sometimes `NetCDF: Index exceeds
dimension bound`, sometimes plausible garbage.
"""
function read_local_slab!(dest::AbstractArray, src::SlabSource, var::AbstractString,
                          dst_start::AbstractVector{<:Integer})
    entries = get(src.entries, var, nothing)
    entries === nothing && error(
        "read_local_slab!: no variable '$var' with slab metadata in $(src.files). " *
        "Variables found: $(sort(collect(keys(src.entries)))).")

    dst_count = collect(Int, size(dest))
    coverage_mask = falses(size(dest))
    # `read_netcdf_variable` dispatches on `var_name::String`; `var` may be any
    # `AbstractString` (e.g. a `SubString`), which would otherwise `MethodError` below.
    varname = String(var)

    for e in entries
        ov = slab_overlap(e.start, e.count, dst_start, dst_count)
        ov === nothing && continue
        dst_ranges = ntuple(d -> (ov.dst_offset[d] + 1):(ov.dst_offset[d] + ov.extent[d]),
                            length(dst_count))
        region = view(coverage_mask, dst_ranges...)
        # Checked at the moment of writing, while `ov` still identifies exactly which
        # sub-region is claimed twice — a running element count can't localize this.
        any(region) && error(
            "read_local_slab!: '$var' has overlapping slabs at the region starting at " *
            "$(collect(Int, dst_start) .+ ov.dst_offset) with size $(ov.extent), inside the " *
            "requested region starting at $(collect(Int, dst_start)) with size $dst_count. " *
            "Slabs overlap each other — the checkpoint has duplicate coverage. " *
            "Files: $(src.files).")
        # NetCDF.jl indexes the STORED variable 1-based; `ov.src_offset` is already
        # the 0-based offset inside that stored slab, so +1 is the whole conversion.
        # `_slab_read_indices` then pads for any leading dimension the slab metadata
        # does not describe (the handler's unlimited `sim_time` axis).
        rd_start, rd_count = _slab_read_indices(e, ov, varname)
        chunk = read_netcdf_variable(e.file, varname; start = rd_start, count = rd_count)
        dest[dst_ranges...] = reshape(chunk, ov.extent...)
        region .= true
    end

    all(coverage_mask) || error(
        "read_local_slab!: '$var' covers $(count(coverage_mask)) of $(length(coverage_mask)) " *
        "elements for the requested region starting at $(collect(Int, dst_start)) with size " *
        "$dst_count. The checkpoint does not span this range — it may be incomplete or " *
        "from a different resolution. Files: $(src.files).")

    return dest
end
