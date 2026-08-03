# ---------------------------------------------------------------------------
# NetCDF group C-API layer
#
# Everything here talks to netCDF-4 groups through NetCDF.jl's thin C wrappers:
# opening a file, creating groups and dimensions, defining variables, and reading
# or writing a hyperslab. It knows nothing about `NetCDFFileHandler`, output
# scheduling, or Tarang fields — verified self-contained when it was split out of
# `netcdf_output.jl`, which had grown past 3000 lines.
#
# The split is not cosmetic. This layer is where `group_ncread` was found reading
# PAST THE END of a short `start`/`count` vector: `nc_get_vara_*` reads exactly
# `ndims(variable)` entries from the pointer regardless of how long the Julia array
# behind it is, so a rank-2 `start` against a rank-3 variable returned a plausible
# array built partly from adjacent memory, with no error. `_validate_vara_indices`
# now guards every such call. Keeping this boundary explicit keeps that guard, and
# the ccalls it protects, reviewable as one unit.
# ---------------------------------------------------------------------------

function create_empty_netcdf4_file!(filename::String)
    ncid = Int32[0]
    NetCDF.nc_create(filename, NetCDF.NC_NETCDF4 | NetCDF.NC_CLOBBER, ncid)
    NetCDF.nc_close(ncid[1])
    return nothing
end

"""
    _netcdf_absence(err) -> Bool

True when `err` means a NetCDF lookup found no such file, group, or variable —
as opposed to a real failure while reading one.

NetCDF.jl is inconsistent about this: the C-API wrappers raise a typed
`NetCDF.NetCDFError`, but its higher-level readers raise an untyped
`ErrorException` via `error("NetCDF file ... does not have a variable named ...")`.
A type test alone therefore cannot express "absent", which is why this predicate
also matches that message. Every caller that treats absence as an expected miss
goes through here, so the string match lives in exactly one place and re-raises
anything it does not recognise.
"""
function _netcdf_absence(err)
    err isa NetCDF.NetCDFError && return true
    err isa ErrorException || return false
    return occursin("does not have a variable named", err.msg) ||
           occursin("does not have a dimension named", err.msg)
end

function with_netcdf_file(filename::String, mode::Integer, f::Function)
    ncid = Int32[0]
    NetCDF.nc_open(filename, mode, ncid)
    try
        return f(ncid[1])
    finally
        NetCDF.nc_close(ncid[1])
    end
end

with_netcdf_file(f::Function, filename::String, mode::Integer) = with_netcdf_file(filename, mode, f)

function enter_define_mode!(ncid::Integer)
    try
        NetCDF.nc_redef(ncid)
    catch err
        # NC_EINDEFINE simply means the file is already in define mode, which is the
        # whole point of the probe. Any other NetCDF failure is real.
        err isa NetCDF.NetCDFError || rethrow()
    end
    return nothing
end

function group_id(ncid::Integer, group::String; create::Bool=false)
    isempty(group) && return Int32(ncid)

    gid = Int32[0]
    try
        NetCDF.nc_inq_grp_ncid(ncid, group, gid)
    catch err
        # "no such group" is the expected miss; a different NetCDF fault, or any
        # non-NetCDF exception, must not be mistaken for an absent group.
        err isa NetCDF.NetCDFError || rethrow()
        create || rethrow()
        NetCDF.nc_def_grp(ncid, group, gid)
    end
    return gid[1]
end

function root_dim_id(ncid::Integer, dim_name::String, dim_len)
    dimid = Int32[0]
    try
        NetCDF.nc_inq_dimid(ncid, dim_name, dimid)
    catch err
        # "no such dimension" is the expected miss and means we define it here.
        err isa NetCDF.NetCDFError || rethrow()
        len = dim_len == Inf ? NetCDF.NC_UNLIMITED : Csize_t(dim_len)
        NetCDF.nc_def_dim(ncid, dim_name, len, dimid)
    end
    return dimid[1]
end

function group_var_id(ncid::Integer, group::String, var_name::String)
    gid = group_id(ncid, group; create=false)
    varid = Int32[0]
    NetCDF.nc_inq_varid(gid, var_name, varid)
    return gid, varid[1]
end

function group_var_exists(filename::String, group::String, var_name::String)
    isfile(filename) || return false
    try
        with_netcdf_file(filename, NetCDF.NC_NOWRITE) do ncid
            group_var_id(ncid, group, var_name)
        end
        return true
    catch err
        # A missing file, group or variable answers "no". A non-NetCDF exception is a
        # real fault and must not be reported as absence.
        _netcdf_absence(err) || rethrow()
        return false
    end
end

function netcdf_type_code(::Type{Float64})
    return NetCDF.NC_DOUBLE
end

function netcdf_type_code(::Type{Float32})
    return NetCDF.NC_FLOAT
end

function netcdf_type_code(::Type{Int64})
    return NetCDF.NC_INT64
end

function netcdf_type_code(::Type{Int32})
    return NetCDF.NC_INT
end

function netcdf_type_code(T::Type)
    return eltype(T) <: Float32 ? NetCDF.NC_FLOAT : NetCDF.NC_DOUBLE
end

function group_nccreate(filename::String, group::String, var_name::String, dims...;
                        atts::Dict=Dict{Any, Any}(), t::Type=Float64)
    iseven(length(dims)) || throw(ArgumentError("Dimensions must be name/length pairs"))

    with_netcdf_file(filename, NetCDF.NC_WRITE) do ncid
        enter_define_mode!(ncid)
        gid = group_id(ncid, group; create=true)

        dimids = Int32[]
        for i in 1:2:length(dims)
            dim_name = String(dims[i])
            dim_len = dims[i + 1]
            push!(dimids, root_dim_id(ncid, dim_name, dim_len))
        end

        exists = true
        varid = Int32[0]
        try
            NetCDF.nc_inq_varid(gid, var_name, varid)
        catch err
            err isa NetCDF.NetCDFError || rethrow()
            exists = false
        end

        if !exists
            c_dimids = reverse(dimids)
            NetCDF.nc_def_var(gid, var_name, netcdf_type_code(t), length(c_dimids), c_dimids, varid)
        end

        if !isempty(atts)
            for (att_name, att_value) in atts
                NetCDF.nc_put_att(gid, varid[1], string(att_name), att_value)
            end
        end

        NetCDF.nc_enddef(ncid)
    end

    return nothing
end

function _group_put_vara(gid::Integer, varid::Integer, start, count, data::Array{Float64})
    NetCDF.nc_put_vara_double(gid, varid, start, count, data)
end

function _group_put_vara(gid::Integer, varid::Integer, start, count, data::Array{Float32})
    NetCDF.nc_put_vara_float(gid, varid, start, count, data)
end

function _group_put_vara(gid::Integer, varid::Integer, start, count, data::Array{Int64})
    NetCDF.nc_put_vara_longlong(gid, varid, start, count, data)
end

function _group_put_vara(gid::Integer, varid::Integer, start, count, data::Array{Int32})
    NetCDF.nc_put_vara_int(gid, varid, start, count, data)
end

function group_ncwrite(data::AbstractArray, filename::String, group::String, var_name::String; start=nothing)
    array = Array(data)
    start_indices = start === nothing ? ones(Int, ndims(array)) : collect(Int, start)
    count_indices = collect(Int, size(array))

    with_netcdf_file(filename, NetCDF.NC_WRITE) do ncid
        gid, varid = group_var_id(ncid, group, var_name)
        # Validate against the variable's rank ON DISK, not `ndims(array)`. The old
        # check compared the two Julia vectors to each other, so a rank-2 array
        # written into a rank-3 variable still handed a 2-entry vector to a C call
        # that reads 3 — the same read-past-the-end hazard as the read path.
        _, shape, unlimited = _group_var_layout(gid, varid)
        _validate_vara_indices(start_indices, count_indices, shape,
                               "group_ncwrite($(repr(group)), $(repr(var_name)))";
                               unlimited = unlimited)

        c_start = Csize_t.(reverse(start_indices .- 1))
        c_count = Csize_t.(reverse(count_indices))
        _group_put_vara(gid, varid, c_start, c_count, array)
    end

    return nothing
end

function _group_var_type_and_shape(gid::Integer, varid::Integer)
    typep, shape, _ = _group_var_layout(gid, varid)
    return typep, shape
end

"""Element type, current shape, and per-dimension unlimited flags for a group variable.

The unlimited flags matter because an unlimited dimension's *current* length is not
a bound: writing past it is how the dimension grows. A `sim_time` axis sits at
length 0 until the first write."""
function _group_var_layout(gid::Integer, varid::Integer)
    typep = Int32[0]
    ndimsp = Int32[0]
    dimids = zeros(Int32, NetCDF.NC_MAX_VAR_DIMS)
    natts = Int32[0]
    NetCDF.nc_inq_var(gid, varid, C_NULL, typep, ndimsp, dimids, natts)

    var_dimids = dimids[1:ndimsp[1]]
    unlim = _group_unlimited_dimids(gid)

    c_shape = Int[]
    c_unlimited = Bool[]
    for dimid in var_dimids
        len = Csize_t[0]
        NetCDF.nc_inq_dimlen(gid, dimid, len)
        push!(c_shape, Int(len[1]))
        push!(c_unlimited, dimid in unlim)
    end

    return typep[1], reverse(c_shape), reverse(c_unlimited)
end

"""Dimension ids that are unlimited and visible from `gid` — its own and its ancestors'.

`nc_inq_unlimdims` reports only the dimensions defined IN the group it is asked
about, not the ones a group inherits. `root_dim_id` defines every dimension at the
root, so asking a `"vars"`/`"time"` group alone reports none and an unlimited
`sim_time` axis looks like a fixed axis of length 0. Walk up to the root."""
function _group_unlimited_dimids(gid::Integer)
    ids = Int32[]
    current = Int32(gid)
    while true
        nunlim = Int32[0]
        NetCDF.nc_inq_unlimdims(current, nunlim, C_NULL)
        n = Int(nunlim[1])
        if n > 0
            level = zeros(Int32, n)
            NetCDF.nc_inq_unlimdims(current, nunlim, level)
            append!(ids, level)
        end
        parent = Int32[0]
        at_root = false
        try
            NetCDF.nc_inq_grp_parent(current, parent)
        catch err
            # NC_ENOGRP means `current` IS the root, which is the loop's exit
            # condition. Anything else is a real failure and must not be swallowed —
            # silently treating it as "no parent" would drop the root's unlimited
            # dimensions and turn an append into a spurious bounds error.
            err isa NetCDF.NetCDFError && err.code == NetCDF.NC_ENOGRP || rethrow()
            at_root = true
        end
        at_root && break
        current = parent[1]
    end
    return ids
end

function group_variable_names(filename::String, group::String)
    isfile(filename) || return String[]

    try
        with_netcdf_file(filename, NetCDF.NC_NOWRITE) do ncid
            gid = group_id(ncid, group; create=false)
            nvars = Int32[0]
            NetCDF.nc_inq_varids(gid, nvars, C_NULL)
            varids = zeros(Int32, nvars[1])
            NetCDF.nc_inq_varids(gid, nvars, varids)

            names = String[]
            for varid in varids
                name_buf = zeros(UInt8, NetCDF.NC_MAX_NAME + 1)
                NetCDF.nc_inq_varname(gid, varid, name_buf)
                push!(names, unsafe_string(pointer(name_buf)))
            end
            return names
        end
    catch err
        # A file or group that cannot be opened has no variables to list. Anything
        # other than a NetCDF failure is real.
        err isa NetCDF.NetCDFError || rethrow()
        return String[]
    end
end

function group_variable_metadata(filename::String, group::String, var_name::String)
    with_netcdf_file(filename, NetCDF.NC_NOWRITE) do ncid
        gid, varid = group_var_id(ncid, group, var_name)

        typep = Int32[0]
        ndimsp = Int32[0]
        dimids = zeros(Int32, NetCDF.NC_MAX_VAR_DIMS)
        natts = Int32[0]
        NetCDF.nc_inq_var(gid, varid, C_NULL, typep, ndimsp, dimids, natts)

        dim_names = String[]
        dim_lengths = Int[]
        for dimid in dimids[1:ndimsp[1]]
            dim_name, dim_len = NetCDF.nc_inq_dim(gid, dimid)
            push!(dim_names, dim_name)
            push!(dim_lengths, Int(dim_len))
        end

        return (
            name=var_name,
            atts=Dict{Any, Any}(NetCDF.getatts_all(gid, varid, natts[1])),
            dim_names=reverse(dim_names),
            dim_lengths=reverse(dim_lengths),
        )
    end
end

function _group_get_vara!(gid::Integer, varid::Integer, start, count, data::Array{Float64})
    NetCDF.nc_get_vara_double(gid, varid, start, count, data)
end

function _group_get_vara!(gid::Integer, varid::Integer, start, count, data::Array{Float32})
    NetCDF.nc_get_vara_float(gid, varid, start, count, data)
end

function _group_get_vara!(gid::Integer, varid::Integer, start, count, data::Array{Int64})
    NetCDF.nc_get_vara_longlong(gid, varid, start, count, data)
end

function _group_get_vara!(gid::Integer, varid::Integer, start, count, data::Array{Int32})
    NetCDF.nc_get_vara_int(gid, varid, start, count, data)
end

"""
    _validate_vara_indices(start_indices, count_indices, shape, context)

Check a 1-based `start`/`count` hyperslab against a variable's on-disk `shape`.

`nc_get_vara_*` and `nc_put_vara_*` read exactly `ndims(variable)` entries from the
`start` and `count` pointers, regardless of how long the Julia vectors behind them
are. A vector shorter than the variable's rank therefore makes the C library read
past the end of Julia-owned memory. That is not a theoretical hazard: a rank-2
`start` against a rank-3 variable returned a plausible array with **no error at
all**, its shape and part of its contents taken from whatever followed the vector
in memory. Silent wrong values are this codebase's dominant failure mode, so every
index vector is checked here before it reaches a `ccall`.

`count[i] == -1` means "to the end of dimension i" and must already have been
resolved by the caller.
"""
function _validate_vara_indices(start_indices::Vector{Int}, count_indices::Vector{Int},
                                shape::Vector{Int}, context::AbstractString;
                                unlimited::Union{Nothing, Vector{Bool}}=nothing)
    n = length(shape)
    if length(start_indices) != n || length(count_indices) != n
        throw(DimensionMismatch(
            "$context: the variable has $n dimension(s), so start and count must each " *
            "have $n entries; got start=$(start_indices) ($(length(start_indices))) and " *
            "count=$(count_indices) ($(length(count_indices))). A shorter vector would be " *
            "read past its end by NetCDF's C API."))
    end

    for i in 1:n
        s, c = start_indices[i], count_indices[i]
        if s < 1
            throw(ArgumentError(
                "$context: start[$i] = $s; these indices are 1-based, so it must be >= 1."))
        end
        if c < 0
            throw(ArgumentError(
                "$context: count[$i] = $c; a count must be non-negative (use -1 before " *
                "resolution to mean 'to the end of this dimension')."))
        end
        # An unlimited dimension's current length is not a bound — writing past it is
        # how it grows, and a `sim_time` axis sits at length 0 until the first write.
        # Callers that pass `unlimited` (writes) exempt those axes; callers that do
        # not (reads) get the full extent check, because reading past the current
        # length of an unlimited dimension is genuinely out of bounds.
        is_unlimited = unlimited !== nothing && unlimited[i]
        if !is_unlimited && s + c - 1 > shape[i]
            throw(ArgumentError(
                "$context: start[$i] = $s with count[$i] = $c reads through index " *
                "$(s + c - 1) of dimension $i, which has length $(shape[i])."))
        end
    end
    return nothing
end

function group_ncread(filename::String, group::String, var_name::String; start=nothing, count=nothing)
    with_netcdf_file(filename, NetCDF.NC_NOWRITE) do ncid
        gid, varid = group_var_id(ncid, group, var_name)
        nctype, shape = _group_var_type_and_shape(gid, varid)
        T = NetCDF.nctype2jltype[nctype]

        start_indices = start === nothing ? ones(Int, length(shape)) : collect(Int, start)
        count_indices = count === nothing ? collect(shape) : collect(Int, count)
        # Resolve the "to the end" sentinel before validating, but only where a
        # matching start exists — a length mismatch is the validator's to report.
        if length(count_indices) == length(start_indices) == length(shape)
            for i in eachindex(count_indices)
                if count_indices[i] == -1
                    count_indices[i] = shape[i] - start_indices[i] + 1
                end
            end
        end
        _validate_vara_indices(start_indices, count_indices, shape,
                               "group_ncread($(repr(group)), $(repr(var_name)))")

        data = Array{T}(undef, count_indices...)
        c_start = Csize_t.(reverse(start_indices .- 1))
        c_count = Csize_t.(reverse(count_indices))
        _group_get_vara!(gid, varid, c_start, c_count, data)
        return data
    end
end
