"""
NetCDF File Merging Utility for Tarang.jl

This module provides comprehensive utilities for merging per-processor NetCDF files
created by Tarang.jl's distributed output system.

Key Features:
- Merge per-processor files (handler_s1_p*.nc) into single files
- Reconstruct global field data from distributed pieces  
- Preserve all metadata and coordinate information
- Handle different data layouts and field distributions
- Optional cleanup of source files after merging
- Support for both 2D and 3D field merging
- Parallel and serial merging modes

Usage:
```julia
# Basic merging
merge_netcdf_files("snapshots")

# Advanced options
merge_netcdf_files("snapshots", 
                   output_name="snapshots_merged.nc",
                   cleanup=true, 
                   merge_mode="reconstruct")

# Batch merge multiple handlers
batch_merge_netcdf(["snapshots", "analysis", "checkpoints"])
```
"""

using NetCDF
using MPI
using Printf
using Dates
using FFTW  # For fft/ifft in layout transformations

# Merging modes
@enum MergeMode begin
    SIMPLE_CONCAT    # Simply concatenate data from all processors
    RECONSTRUCT      # Reconstruct global field from distributed data  
    DOMAIN_DECOMP    # Handle domain decomposition layouts
end

"""
NetCDF File Merger - handles merging of per-processor files
"""
struct NetCDFMerger
    base_name::String
    set_number::Int
    processor_files::Vector{String}
    output_file::String
    merge_mode::MergeMode
    cleanup::Bool
    verbose::Bool
    
    function NetCDFMerger(base_name::String;
                         set_number::Int=1,
                         output_name::String="",
                         merge_mode::MergeMode=RECONSTRUCT,
                         cleanup::Bool=false,
                         verbose::Bool=true)

        # Validate parameters
        if isempty(base_name)
            throw(ArgumentError("NetCDFMerger: base_name cannot be empty"))
        end
        if set_number < 1
            throw(ArgumentError("NetCDFMerger: set_number must be positive, got $set_number"))
        end

        # Find all processor files for this handler/set
        set_pattern = "$(base_name)_s$(set_number)"
        search_dir = "."
        
        # Look for files in set directory
        set_dir = joinpath(search_dir, set_pattern)
        if isdir(set_dir)
            search_dir = set_dir
        end
        
        processor_files = String[]
        for file in readdir(search_dir, join=true)
            # Match pattern more precisely: handler_s#_p#.nc
            m = match(r"_p(\d+)\.nc$", basename(file))
            if occursin("$(set_pattern)_p", basename(file)) && m !== nothing
                push!(processor_files, file)
            end
        end

        # Sort by processor number (safe: we already verified match exists)
        sort!(processor_files, by=f -> begin
            m = match(r"_p(\d+)\.nc$", basename(f))
            m !== nothing ? parse(Int, m.captures[1]) : 0
        end)
        
        # Determine output filename (matching Tarang convention)
        if isempty(output_name)
            if isdir(set_dir)
                output_file = joinpath(set_dir, "$(set_pattern).nc")
            else
                output_file = "$(set_pattern).nc"
            end
        else
            output_file = output_name
        end
        
        new(base_name, set_number, processor_files, output_file, 
            merge_mode, cleanup, verbose)
    end
end

function _processor_rank_from_path(path::String)
    m = match(r"_p(\d+)\.nc$", basename(path))
    m === nothing && error("Processor filename does not end in _p<rank>.nc: '$path'")
    return parse(Int, m.captures[1])
end

function _global_integer_attribute(info, name::String)
    value = get(info.gatts, name, get(info.gatts, Symbol(name), nothing))
    value === nothing && return nothing
    value isa AbstractArray && (value = only(value))
    return Int(value)
end

function _data_variable_names(info)
    names = Set{String}()
    for var_info in info.vars
        is_time_coordinate(var_info.name) && continue
        is_coordinate_variable(var_info) && continue
        push!(names, var_info.name)
    end
    return names
end

"""Validate every source before the output path is touched."""
function validate_merger_inputs!(merger::NetCDFMerger)
    isempty(merger.processor_files) &&
        error("No processor files found for $(merger.base_name)_s$(merger.set_number)")

    output_path = abspath(normpath(merger.output_file))
    source_paths = abspath.(normpath.(merger.processor_files))
    output_path in source_paths && error(
        "Refusing to merge in place: output '$output_path' aliases a processor source")
    if ispath(output_path)
        for source_path in source_paths
            Base.Filesystem.samefile(output_path, source_path) && error(
                "Refusing to merge in place: output '$output_path' is a filesystem " *
                "alias of processor source '$source_path'")
        end
    end

    ranks = _processor_rank_from_path.(merger.processor_files)
    length(unique(ranks)) == length(ranks) ||
        error("Duplicate processor ranks in merge input: $ranks")
    sort(ranks) == collect(0:maximum(ranks)) ||
        error("Processor rank set is incomplete: found $(sort(ranks))")

    infos = Any[]
    expected_sizes = Int[]
    for (file, filename_rank) in zip(merger.processor_files, ranks)
        isfile(file) || error("Processor source disappeared before merge: '$file'")
        info = netcdf_file_info(file) # deliberately fails on corrupt/unreadable input
        push!(infos, info)
        mpi_size = _global_integer_attribute(info, "mpi_size")
        mpi_size === nothing || push!(expected_sizes, mpi_size)
        declared_rank = _global_integer_attribute(info, "processor_rank")
        if declared_rank !== nothing
            declared_rank == filename_rank || error(
                "Processor rank metadata $declared_rank disagrees with '$file'")
        end
    end

    if length(infos) > 1 && length(expected_sizes) != length(infos)
        error("Every source in a multi-rank merge must declare mpi_size; " *
              "found it on $(length(expected_sizes)) of $(length(infos)) files")
    end
    if !isempty(expected_sizes)
        length(unique(expected_sizes)) == 1 ||
            error("Processor files disagree on mpi_size: $expected_sizes")
        expected = only(unique(expected_sizes))
        sort(ranks) == collect(0:(expected - 1)) || error(
            "Processor set is incomplete: expected ranks 0:$(expected - 1), found $(sort(ranks))")
    end

    reference_vars = _data_variable_names(first(infos))
    for (file, info) in zip(merger.processor_files[2:end], infos[2:end])
        vars = _data_variable_names(info)
        vars == reference_vars || error(
            "Processor variable schema mismatch in '$file': expected " *
            "$(sort!(collect(reference_vars))), found $(sort!(collect(vars)))")
    end
    return infos
end

function netcdf_file_info(file::String)
    NetCDF.open(file) do nc
        dims = [
            (name=dim.name, dimlen=Int(dim.dimlen), unlim=dim.unlim)
            for dim in values(nc.dim)
        ]
        vars = Any[
            (
                name=var.name,
                atts=Dict{Any, Any}(var.atts),
                dim_names=[dim.name for dim in var.dim],
                dim_lengths=[Int(dim.dimlen) for dim in var.dim],
            )
            for var in values(nc.vars)
        ]

        known_names = Set(v.name for v in vars)
        for group in (NETCDF_TIME_GROUP, NETCDF_GRIDS_GROUP, NETCDF_VARS_GROUP)
            for var_name in group_variable_names(file, group)
                var_name in known_names && continue
                push!(vars, group_variable_metadata(file, group, var_name))
                push!(known_names, var_name)
            end
        end

        return (dim=dims, vars=vars, gatts=Dict{Any, Any}(nc.gatts))
    end
end

function read_netcdf_variable(file::String, var_name::String; start=nothing, count=nothing)
    for group in (NETCDF_VARS_GROUP, NETCDF_GRIDS_GROUP, NETCDF_TIME_GROUP)
        if group_var_exists(file, group, var_name)
            if start === nothing && count === nothing
                return group_ncread(file, group, var_name)
            elseif count === nothing
                return group_ncread(file, group, var_name; start=start)
            elseif start === nothing
                return group_ncread(file, group, var_name; count=count)
            else
                return group_ncread(file, group, var_name; start=start, count=count)
            end
        end
    end

    if start === nothing && count === nothing
        return ncread(file, var_name)
    elseif count === nothing
        return ncread(file, var_name; start=start)
    elseif start === nothing
        return ncread(file, var_name; count=count)
    else
        return ncread(file, var_name; start=start, count=count)
    end
end

function _int_vector_from_attr(value)
    if value === nothing
        return nothing
    elseif value isa Number
        return [Int(value)]
    elseif value isa AbstractArray || value isa Tuple
        return Int.(collect(value))
    elseif value isa AbstractString
        matches = collect(eachmatch(r"-?\d+", value))
        return isempty(matches) ? nothing : [parse(Int, m.match) for m in matches]
    else
        return nothing
    end
end

function normalize_global_shape(value, data_shape::Tuple)
    shape = _int_vector_from_attr(value)
    shape === nothing && return nothing

    if length(shape) == length(data_shape)
        return tuple(shape...)
    elseif length(shape) == length(data_shape) - 1
        return tuple(data_shape[1], shape...)
    else
        return tuple(shape...)
    end
end

function spatial_dim_index(coord_var::String)
    m = match(r"_dim(\d+)$", coord_var)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function is_time_coordinate(var_name::String)
    return var_name in ("sim_time", "wall_time", "timestep", "iteration", "write_number")
end

function is_coordinate_variable(var_info)
    dim_names = getproperty(var_info, :dim_names)
    return startswith(var_info.name, "dim_") ||
           occursin(r"_dim\d+$", var_info.name) ||
           (length(dim_names) == 1 && var_info.name == dim_names[1])
end

function metadata_for_variable(file::String, var_name::String)
    info = netcdf_file_info(file)
    for var_info in info.vars
        if var_info.name == var_name
            return var_info
        end
    end
    return nothing
end

function coordinate_source_variable(file::String, coord_var::String, file_info::Dict)
    prefix = replace(coord_var, r"_dim\d+$" => "")
    if prefix != coord_var && prefix in file_info["data_vars"]
        return prefix
    end

    for source_var in file_info["data_vars"]
        var_info = metadata_for_variable(file, source_var)
        var_info === nothing && continue
        if coord_var in var_info.dim_names
            return source_var
        end
    end

    return isempty(file_info["data_vars"]) ? "" : first(file_info["data_vars"])
end

function coordinate_indices(var_info, coord_var::String)
    dim_pos = findfirst(==(coord_var), var_info.dim_names)
    if dim_pos !== nothing
        has_time_dim = !isempty(var_info.dim_names) && is_time_coordinate(var_info.dim_names[1])
        decomp_index = has_time_dim ? dim_pos - 1 : dim_pos
        return decomp_index < 1 ? nothing : (decomp_index=decomp_index, global_dim_index=dim_pos)
    end

    coord_index = spatial_dim_index(coord_var)
    coord_index === nothing && return nothing
    has_time_dim = !isempty(var_info.dim_names) && is_time_coordinate(var_info.dim_names[1])
    return (decomp_index=coord_index,
            global_dim_index=has_time_dim ? coord_index + 1 : coord_index)
end

function reconstruct_spatial_coordinate(merger::NetCDFMerger, coord_var::String, file_info::Dict)
    reconstructed = nothing
    covered = falses(0)

    for file in merger.processor_files
        # A processor file that does not carry this coordinate is skipped; that is
        # the expected miss. A non-NetCDF exception is a real fault and must not be
        # silently turned into "this file contributes nothing".
        coord_data = try
            read_netcdf_variable(file, coord_var)
        catch err
            _netcdf_absence(err) || rethrow()
            continue
        end

        source_var = coordinate_source_variable(file, coord_var, file_info)
        isempty(source_var) && return nothing

        var_info = metadata_for_variable(file, source_var)
        var_info === nothing && continue

        indices = coordinate_indices(var_info, coord_var)
        indices === nothing && continue

        data_shape = try
            size(read_netcdf_variable(file, source_var))
        catch err
            _netcdf_absence(err) || rethrow()
            continue
        end
        global_shape = normalize_global_shape(get(var_info.atts, "global_shape", nothing), data_shape)
        global_shape === nothing && continue
        indices.global_dim_index <= length(global_shape) || continue

        global_len = global_shape[indices.global_dim_index]
        if length(coord_data) == global_len
            return coord_data
        end

        start_indices = _int_vector_from_attr(get(var_info.atts, "start", nothing))
        count_indices = _int_vector_from_attr(get(var_info.atts, "count", nothing))
        start_indices === nothing && continue
        count_indices === nothing && continue
        indices.decomp_index <= length(start_indices) || continue
        indices.decomp_index <= length(count_indices) || continue

        if reconstructed === nothing
            reconstructed = Vector{eltype(coord_data)}(undef, global_len)
            covered = falses(global_len)
        end

        range = (start_indices[indices.decomp_index] + 1):(start_indices[indices.decomp_index] + count_indices[indices.decomp_index])
        if length(range) == length(coord_data) && last(range) <= global_len
            reconstructed[range] = coord_data
            covered[range] .= true
        end
    end

    return reconstructed !== nothing && all(covered) ? reconstructed : nothing
end

"""
Get metadata from all processor files to understand data structure
"""
function analyze_processor_files(merger::NetCDFMerger)
    if isempty(merger.processor_files)
        error("No processor files found for $(merger.base_name)_s$(merger.set_number)")
    end
    
    merger.verbose && println("Analyzing $(length(merger.processor_files)) processor files...")
    
    file_info = Dict{String, Any}()
    
    # Analyze first file to get structure
    first_file = merger.processor_files[1]
    info = netcdf_file_info(first_file)
    
    if merger.verbose
        println("  Sample file: $(basename(first_file))")
        println("  Dimensions: $(length(info.dim)) | Variables: $(length(info.vars))")
    end
    
    # Get global attributes
    global_attrs = Dict{String, Any}()
    try
        # Read global attributes (this varies by NetCDF.jl version)
        # We'll collect from the sample file
        global_attrs["processor_count"] = length(merger.processor_files)
        global_attrs["merge_timestamp"] = string(Dates.now())
        global_attrs["source_files"] = join(basename.(merger.processor_files), ", ")
    catch e
        merger.verbose && println("  Warning: Could not read all global attributes: $e")
    end
    
    # Analyze time coordinates
    time_coords = ["sim_time", "wall_time", "timestep", "iteration", "write_number"]
    time_info = Dict{String, Int}()
    
    for coord in time_coords
        try
            data = read_netcdf_variable(first_file, coord)
            time_info[coord] = length(data)
        catch err
            # An absent coordinate has length 0; any other failure is real.
            _netcdf_absence(err) || rethrow()
            time_info[coord] = 0
        end
    end
    
    # Find data variables (exclude coordinate variables)
    data_vars = String[]
    coord_vars = String[]
    
    for var_info in info.vars
        var_name = var_info.name
        if is_time_coordinate(var_name)
            continue
        elseif is_coordinate_variable(var_info)
            push!(coord_vars, var_name)  
        else
            push!(data_vars, var_name)
        end
    end


    expected_output_shapes = Dict{String, Tuple}()
    for var_name in data_vars
        var_info = only(filter(v -> v.name == var_name, info.vars))
        source_shape = Tuple(var_info.dim_lengths)
        if merger.merge_mode == SIMPLE_CONCAT
            expected_output_shapes[var_name] = (source_shape..., length(merger.processor_files))
        else
            declared = normalize_global_shape(
                get(var_info.atts, "global_shape", nothing), source_shape)
            expected_output_shapes[var_name] =
                declared === nothing ? source_shape : Tuple(declared)
        end
    end
    
    file_info["global_attrs"] = global_attrs
    file_info["time_info"] = time_info
    file_info["data_vars"] = data_vars
    file_info["coord_vars"] = coord_vars
    file_info["expected_output_shapes"] = expected_output_shapes
    file_info["first_file"] = first_file
    
    merger.verbose && println("  Found $(length(data_vars)) data variables: $(join(data_vars, ", "))")
    merger.verbose && println("  Time steps: $(get(time_info, "sim_time", 0))")
    
    return file_info
end

"""
Merge time coordinate data from all processors
"""
function merge_time_coordinates!(merger::NetCDFMerger, output_file::String, file_info::Dict)
    merger.verbose && println("Merging time coordinates...")
    
    time_coords = ["sim_time", "wall_time", "timestep", "iteration", "write_number"]
    
    for coord_name in time_coords
        if get(file_info["time_info"], coord_name, 0) > 0
            # Read time data from first processor (should be identical across all)
            time_data = read_netcdf_variable(merger.processor_files[1], coord_name)

            # Create time coordinate (nccreate must be called before ncwrite)
            group_nccreate(output_file, NETCDF_TIME_GROUP, coord_name, "sim_time", length(time_data),
                           t=eltype(time_data),
                           atts=Dict("long_name" => coord_name,
                                     "units" => coord_name == "sim_time" ? "dimensionless time" : "seconds",
                                     "axis" => "T"))

            # Write time data
            group_ncwrite(time_data, output_file, NETCDF_TIME_GROUP, coord_name)
        end
    end
end

"""
Merge spatial coordinate data
"""  
function merge_spatial_coordinates!(merger::NetCDFMerger, output_file::String, file_info::Dict)
    merger.verbose && println("Merging spatial coordinates...")
    
    # Get all spatial coordinate variables from all files
    all_coord_vars = Set{String}()
    for file in merger.processor_files
        info = netcdf_file_info(file)
        for var in info.vars
            if var.name in file_info["coord_vars"]
                push!(all_coord_vars, var.name)
            end
        end
    end
    
    # Process each coordinate variable
    for coord_var in all_coord_vars
        merger.verbose && println("  Processing coordinate: $coord_var")
        
        # Strategy: Take coordinate data from first file that has it
        # In a real domain decomposition, coordinates might need reconstruction
        coord_data = reconstruct_spatial_coordinate(merger, coord_var, file_info)
        coord_attrs = Dict{String, Any}()
        
        if coord_data === nothing
            for file in merger.processor_files
                try
                    coord_data = read_netcdf_variable(file, coord_var)
                    break
                catch err
                    _netcdf_absence(err) || rethrow()
                    continue
                end
            end
        end
        
        if coord_data !== nothing
            coord_attrs["long_name"] = coord_var
            coord_attrs["axis"] = occursin("dim1", coord_var) ? "X" : (occursin("dim2", coord_var) ? "Y" : "Z")

            # Create coordinate variable in output file
            group_nccreate(output_file, NETCDF_GRIDS_GROUP, coord_var, coord_var, length(coord_data),
                           t=eltype(coord_data),
                           atts=coord_attrs)
            group_ncwrite(coord_data, output_file, NETCDF_GRIDS_GROUP, coord_var)
        end
    end
end

"""
Merge data variables using specified merge mode
"""
function merge_data_variables!(merger::NetCDFMerger, output_file::String, file_info::Dict)
    merger.verbose && println("Merging data variables (mode: $(merger.merge_mode))...")
    
    for var_name in file_info["data_vars"]
        merger.verbose && println("  Merging variable: $var_name")
        
        if merger.merge_mode == SIMPLE_CONCAT
            merge_variable_concat!(merger, output_file, var_name, file_info)
        elseif merger.merge_mode == RECONSTRUCT  
            merge_variable_reconstruct!(merger, output_file, var_name, file_info)
        else
            merge_variable_domain_decomp!(merger, output_file, var_name, file_info)
        end
    end
end

function verify_merged_output!(output_file::String, file_info::Dict)
    isfile(output_file) || error("Merged output was not created: '$output_file'")
    expected = Set{String}(file_info["data_vars"])
    actual = Set(group_variable_names(output_file, NETCDF_VARS_GROUP))
    actual == expected || error(
        "Merged output variable set is incomplete: expected $(sort!(collect(expected))), " *
        "found $(sort!(collect(actual)))")
    for (coord_name, expected_count) in file_info["time_info"]
        expected_count > 0 || continue
        actual_count = length(vec(read_netcdf_variable(output_file, coord_name)))
        actual_count == expected_count || error(
            "Merged '$coord_name' has $actual_count records; expected $expected_count")
    end
    sim_records = get(file_info["time_info"], "sim_time", 0)
    for var_name in expected
        data = read_netcdf_variable(output_file, var_name)
        expected_shape = file_info["expected_output_shapes"][var_name]
        size(data) == expected_shape || error(
            "Merged '$var_name' has shape $(size(data)); expected $expected_shape")
        sim_records == 0 || size(data, 1) == sim_records || error(
            "Merged '$var_name' has $(size(data, 1)) records; expected $sim_records")
    end
    return true
end

"""
Simple concatenation merge: combine data along processor dimension
"""
function ensure_processor_coordinate!(output_file::String, n_procs::Int)
    exists = false
    try
        read_netcdf_variable(output_file, "processor", start=[1], count=[1])
        exists = true
    catch err
        # Absence is the answer being probed for; anything else is a real fault.
        _netcdf_absence(err) || rethrow()
        exists = false
    end

    if !exists
        proc_coord = collect(0:(n_procs - 1))
        group_nccreate(output_file, NETCDF_GRIDS_GROUP, "processor", "processor", length(proc_coord),
                       t=eltype(proc_coord),
                       atts=Dict("long_name" => "MPI processor rank"))
        group_ncwrite(proc_coord, output_file, NETCDF_GRIDS_GROUP, "processor")
    end
end

function merge_variable_concat!(merger::NetCDFMerger, output_file::String, var_name::String, file_info::Dict)
    # Read data from all processor files
    all_data = Any[]
    var_attrs = Dict{String, Any}()
    data_type = Float64
    dim_names = String[]
    
    for (i, file) in enumerate(merger.processor_files)
        data = read_netcdf_variable(file, var_name)
        push!(all_data, data)

        if i == 1
            # Get metadata from first file
            var_attrs["long_name"] = var_name
            var_attrs["standard_name"] = var_name
            var_attrs["merged_from"] = "$(length(merger.processor_files)) processors"
            data_type = eltype(data)

            # Get dimension structure
            data_shape = size(data)
            dim_names = ["sim_time"]
            for j in 2:length(data_shape)
                push!(dim_names, "$(var_name)_dim$(j-1)")
            end
        end
    end
    
    if isempty(all_data)
        merger.verbose && println("    No data found for $var_name")
        return
    end
    
    # For simple concat, we add a processor dimension
    # Combined data shape: [spatial/time dims..., processor]
    first_data = all_data[1]
    n_dims = ndims(first_data)
    n_procs = length(all_data)
    combined_shape = (size(first_data)..., n_procs)

    # Stack data along new processor dimension
    combined_data = zeros(data_type, combined_shape)
    for (i, data) in enumerate(all_data)
        size(data) == size(first_data) || error(
            "Cannot concatenate '$var_name': processor $(i - 1) has shape " *
            "$(size(data)), expected $(size(first_data))")
        # Build proper index tuple: (:, :, ..., :, i) for the i-th processor slice
        indices = ntuple(d -> d <= n_dims ? Colon() : i, n_dims + 1)
        combined_data[indices...] = data
    end
    
    # Create processor coordinate once
    ensure_processor_coordinate!(output_file, length(all_data))

    # Create variable in output file
    dim_names_with_proc = [dim_names..., "processor"]
    dim_sizes = [size(combined_data)...]

    # Build alternating dim_name, dim_size pairs for nccreate
    dim_args = Any[]
    for (dn, ds) in zip(dim_names_with_proc, dim_sizes)
        push!(dim_args, dn)
        push!(dim_args, ds)
    end
    group_nccreate(output_file, NETCDF_VARS_GROUP, var_name, dim_args..., t=data_type, atts=var_attrs)
    
    # Write combined data
    group_ncwrite(combined_data, output_file, NETCDF_VARS_GROUP, var_name)
    
    merger.verbose && println("    Concatenated $(length(all_data)) processor datasets")
end

"""
Reconstruction merge: reconstruct global field from distributed data following Tarang patterns.
This reconstructs the global field using spatial domain decomposition information from each processor.
Based on Tarang post:merge_data() function (lines 317-342).
"""
function merge_variable_reconstruct!(merger::NetCDFMerger, output_file::String, var_name::String, file_info::Dict)
    merger.verbose && println("    Reconstructing global field for $var_name")
    
    # Determine global field shape and collect processor data with domain info
    processor_data = Any[]
    var_attrs = Dict{String, Any}()
    data_type = Float64
    global_shape = nothing
    dim_names = String[]
    
    # First pass: collect metadata and determine global shape
    for (i, file) in enumerate(merger.processor_files)
        try
            # Read variable data
            data = read_netcdf_variable(file, var_name)
            
            # Try to read domain decomposition information (Tarang style)
            start_indices = nothing
            count_indices = nothing
            source_dim_names = nothing
            source_global_shape = nothing
            
            try
                # Look for Tarang-style attributes: 'start' and 'count'
                info = netcdf_file_info(file)
                for var_info in info.vars
                    if var_info.name == var_name
                        source_dim_names = var_info.dim_names
                        # Try different attribute name conventions
                        start_indices = _int_vector_from_attr(get(var_info.atts, "start", nothing))
                        count_indices = _int_vector_from_attr(get(var_info.atts, "count", nothing))
                        
                        if start_indices === nothing
                            start_indices = _int_vector_from_attr(get(var_info.atts, "domain_start", nothing))
                            count_indices = _int_vector_from_attr(get(var_info.atts, "domain_count", nothing))
                        end
                        
                        # If we have global shape info, use it
                        if haskey(var_info.atts, "global_shape")
                            source_global_shape = normalize_global_shape(
                                var_info.atts["global_shape"], size(data))
                            global_shape = source_global_shape
                        end
                        break
                    end
                end
            catch e
                merger.verbose && println("      Warning: Could not read domain info from $(basename(file)): $e")
            end
            
            # Store processor data with domain information
            proc_info = Dict(
                "data" => data,
                "start" => start_indices,
                "count" => count_indices,
                "global_shape" => source_global_shape,
                "file" => file
            )
            push!(processor_data, proc_info)
            
            # Set up metadata from first file
            if i == 1
                var_attrs["long_name"] = var_name
                var_attrs["standard_name"] = var_name
                var_attrs["reconstruction_method"] = "spatial_domain_decomposition"
                data_type = eltype(data)
                
                # Create dimension names following NetCDF conventions
                data_shape = size(data)
                if source_dim_names !== nothing && length(source_dim_names) == length(data_shape)
                    dim_names = String.(source_dim_names)
                else
                    dim_names = ["sim_time"]  # First dimension is usually time
                    for j in 2:length(data_shape)
                        push!(dim_names, "$(var_name)_dim$(j-1)")
                    end
                end
                
                # If no global shape found, warn and fall back to first processor's shape.
                # This likely means only 1/N of the domain is captured.
                if global_shape === nothing
                    global_shape = data_shape
                    @warn "No global_shape metadata found for '$var_name'. " *
                          "Using single-processor shape $data_shape — the merged file " *
                          "may contain only a fraction of the full domain." maxlog=1
                end
            end
            
        catch e
            error("Could not read '$var_name' from '$(basename(file))': " *
                  sprint(showerror, e))
        end
    end
    
    if isempty(processor_data)
        merger.verbose && println("    No data found for $var_name")
        return
    end

    if length(processor_data) > 1
        all(p -> p["start"] !== nothing && p["count"] !== nothing,
            processor_data) || error(
                "Cannot reconstruct '$var_name' from multiple processors without " *
                "explicit start/count slab metadata; use SIMPLE_CONCAT for rank-wise data")
        all(p -> p["global_shape"] !== nothing, processor_data) || error(
                "Cannot verify reconstruction of '$var_name': one or more processors " *
                "lack global_shape metadata")
        length(unique([p["global_shape"] for p in processor_data])) == 1 || error(
                "Processor files disagree on global_shape for '$var_name'")
    end

    # Attempt to infer start/count metadata if missing
    has_domain_info = any(p["start"] !== nothing && p["count"] !== nothing for p in processor_data)
    if !has_domain_info
        inferred_shape = infer_slab_decomposition!(processor_data, merger)
        if inferred_shape !== nothing
            merger.verbose && println("      Inferred slab decomposition for missing domain info")
            if global_shape === nothing
                global_shape = inferred_shape
            end
        end
    end
    
    # Determine final global shape
    if global_shape === nothing
        # Fall back to estimating global shape
        merger.verbose && println("      No global shape info found, estimating...")
        sample_shape = size(processor_data[1]["data"])
        
        # For spectral methods, spatial dimensions are often distributed
        # Estimate by checking if we have spatial domain info
        has_domain_info = any(p["start"] !== nothing && p["count"] !== nothing for p in processor_data)
        
        if has_domain_info
            # Try to determine global shape from domain decomposition
            max_extents = collect(sample_shape)  # Convert tuple to mutable array
            for proc_info in processor_data
                if proc_info["start"] !== nothing && proc_info["count"] !== nothing
                    start = proc_info["start"]
                    count = proc_info["count"]
                    # Update maximum extents (skip time dimension)
                    for i in 2:length(max_extents)
                        if i-1 <= length(start)
                            max_extents[i] = max(max_extents[i], start[i-1] + count[i-1])
                        end
                    end
                end
            end
            global_shape = tuple(max_extents...)
        else
            global_shape = sample_shape
        end
    end
    
    # Initialize global reconstructed field
    reconstructed_data = zeros(data_type, global_shape)
    coverage_mask = falses(global_shape)
    
    merger.verbose && println("      Reconstructing to global shape: $global_shape")
    
    # Reconstruct global field following Tarang merge_data pattern
    for proc_info in processor_data
        data = proc_info["data"]
        start_indices = proc_info["start"]
        count_indices = proc_info["count"]
        
        if start_indices !== nothing && count_indices !== nothing
            # Use Tarang-style spatial slicing (post:339)
            try
                # Skip time dimension (index 1), apply to spatial dimensions
                spatial_slices = Any[]
                push!(spatial_slices, Colon())  # Time dimension - take all
                
                for (s, c) in zip(start_indices, count_indices)
                    push!(spatial_slices, (s+1):(s+c))  # Convert to 1-based indexing
                end
                
                # Fill global array at correct spatial location
                slices = tuple(spatial_slices...)
                if size(data) == size(reconstructed_data[slices...])
                    any(coverage_mask[slices...]) && error(
                        "Overlapping slabs while reconstructing '$var_name': " *
                        "$(basename(proc_info["file"])) overlaps prior coverage at $slices")
                    reconstructed_data[slices...] = data
                    coverage_mask[slices...] .= true
                    merger.verbose && println("        Placed data from $(basename(proc_info["file"])) at $slices")
                else
                    error("Slab shape mismatch for '$var_name' in " *
                          "$(basename(proc_info["file"])): expected " *
                          "$(size(reconstructed_data[slices...])), got $(size(data))")
                end
            catch e
                error("Could not place '$var_name' slab from " *
                      "'$(basename(proc_info["file"]))': $(sprint(showerror, e))")
            end
        else
            # No domain decomposition info - fall back to simple overlay/averaging
            merger.verbose && println("        No spatial decomposition info for $(basename(proc_info["file"])), using fallback")
            if size(data) == size(reconstructed_data)
                # Add data where we don't have coverage
                mask = .!coverage_mask
                reconstructed_data[mask] = data[mask]
                coverage_mask[mask] .= true
            end
        end
    end
    
    # Handle uncovered regions
    uncovered_count = count(!, coverage_mask)
    if uncovered_count > 0
        error("Incomplete reconstruction of '$var_name': $uncovered_count of " *
              "$(length(coverage_mask)) points are not covered by any processor")
    end

    # Create variable in output file
    var_attrs["reconstruction_coverage"] = "$(count(coverage_mask))/$(length(coverage_mask)) points covered"
    dim_sizes = [size(reconstructed_data)...]

    # Build alternating dim_name, dim_size pairs for nccreate
    dim_args = Any[]
    for (dn, ds) in zip(dim_names, dim_sizes)
        push!(dim_args, dn)
        push!(dim_args, ds)
    end
    group_nccreate(output_file, NETCDF_VARS_GROUP, var_name, dim_args..., t=data_type, atts=var_attrs)
    
    group_ncwrite(reconstructed_data, output_file, NETCDF_VARS_GROUP, var_name)
    
    merger.verbose && println("    Reconstructed global field from $(length(processor_data)) processors")
end

"""
Domain decomposition merge: handle spatial domain decomposition based on field layout.
This implements layout-aware merging following Tarang distributor patterns.

Different field layouts (grid space vs coefficient space) have different distribution 
patterns that require specialized merging strategies:
- Grid space: Often distributed in spatial dimensions  
- Coefficient space: Often distributed in spectral mode dimensions
- Mixed layouts: May require transpose-like operations during merging

Based on Tarang distributor concepts and post merge patterns.
"""
function merge_variable_domain_decomp!(merger::NetCDFMerger, output_file::String, var_name::String, file_info::Dict)
    merger.verbose && println("    Domain decomposition merge for $var_name")
    
    # Analyze field layout and distribution pattern from processor files
    processor_data = Any[]
    var_attrs = Dict{String, Any}()
    data_type = Float64
    grid_space_flags = nothing
    field_layout = :unknown
    dim_names = String[]
    
    # First pass: determine field layout and distribution pattern
    for (i, file) in enumerate(merger.processor_files)
        try
            # Read variable data
            data = read_netcdf_variable(file, var_name)
            
            # Read layout information (following Tarang post:281)
            layout_info = Dict{String, Any}()
            try
                info = netcdf_file_info(file)
                for var_info in info.vars
                    if var_info.name == var_name
                        # Key Tarang attributes that determine merging strategy
                        layout_info["grid_space"] = get(var_info.atts, "grid_space", nothing)
                        layout_info["layout"] = get(var_info.atts, "layout", nothing)
                        layout_info["scales"] = get(var_info.atts, "scales", nothing)
                        layout_info["constant"] = get(var_info.atts, "constant", false)
                        layout_info["start"] = _int_vector_from_attr(get(var_info.atts, "start", nothing))
                        layout_info["count"] = _int_vector_from_attr(get(var_info.atts, "count", nothing))
                        layout_info["global_shape"] = normalize_global_shape(get(var_info.atts, "global_shape", nothing), size(data))
                        break
                    end
                end
            catch e
                merger.verbose && println("      Warning: Could not read layout info from $(basename(file)): $e")
            end
            
            # Normalize grid_space flags using layout attribute if needed
            if layout_info["grid_space"] === nothing && layout_info["layout"] !== nothing
                layout_str = lowercase(string(layout_info["layout"]))
                if layout_str in ("g", "grid", "grid_space")
                    layout_info["grid_space"] = true
                elseif layout_str in ("c", "coeff", "coeff_space")
                    layout_info["grid_space"] = false
                end
            end
            layout_info["grid_space"] = normalize_grid_space_flags(layout_info["grid_space"], ndims(data))

            # Store processor data with layout information
            proc_info = Dict(
                "data" => data,
                "layout_info" => layout_info,
                "file" => file
            )
            push!(processor_data, proc_info)
            
            # Determine field layout from first file
            if i == 1
                var_attrs["long_name"] = var_name
                var_attrs["standard_name"] = var_name
                data_type = eltype(data)
                
                # Analyze grid_space flags (Tarang layout indicator)
                grid_space_flags = layout_info["grid_space"]
                if grid_space_flags !== nothing
                    if isa(grid_space_flags, AbstractArray)
                        # Array of boolean flags for each dimension
                        field_layout = determine_layout_type(grid_space_flags)
                        var_attrs["layout_type"] = string(field_layout)
                        var_attrs["grid_space_flags"] = string(grid_space_flags)
                    else
                        field_layout = _bool_from_value(grid_space_flags) ? :grid_space : :coeff_space
                        var_attrs["layout_type"] = string(field_layout)
                    end
                end
                
                # Create dimension names
                data_shape = size(data)
                dim_names = ["sim_time"]
                for j in 2:length(data_shape)
                    push!(dim_names, "$(var_name)_dim$(j-1)")
                end
            end
            
        catch e
            error("Could not read '$var_name' layout data from '$(basename(file))': " *
                  sprint(showerror, e))
        end
    end
    
    if isempty(processor_data)
        merger.verbose && println("    No data found for $var_name")
        return
    end

    if length(processor_data) > 1
        all(p -> p["layout_info"]["start"] !== nothing &&
                 p["layout_info"]["count"] !== nothing,
            processor_data) || error(
                "Cannot domain-reconstruct '$var_name' from multiple processors " *
                "without explicit start/count metadata; use SIMPLE_CONCAT instead")
        all(p -> p["layout_info"]["global_shape"] !== nothing,
            processor_data) || error(
                "Cannot verify domain reconstruction of '$var_name' without " *
                "global_shape metadata on every processor")
    end

    merger.verbose && println("      Field layout: $field_layout")
    
    # Apply layout-specific merging strategy
    if field_layout == :grid_space
        merge_grid_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    elseif field_layout == :coeff_space  
        merge_coeff_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    elseif field_layout == :mixed_layout
        merge_mixed_layout_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    else
        merger.verbose && println("      Unknown layout, using standard reconstruction")
        # Fall back to standard reconstruction
        merge_variable_reconstruct!(merger, output_file, var_name, file_info)
    end
end

"""
Main merge function - orchestrates the entire merging process
"""
function merge_files!(merger::NetCDFMerger)
    merger.verbose && println("Starting NetCDF file merge...")
    merger.verbose && println("   Input: $(length(merger.processor_files)) processor files")
    merger.verbose && println("   Output: $(merger.output_file)")
    merger.verbose && println("   Mode: $(merger.merge_mode)")
    
    if isempty(merger.processor_files)
        @warn "No processor files found for merging"
        return false
    end

    # Validate every source, the declared rank set, and output/source separation
    # before removing or creating anything. A failed preflight must be read-only.
    try
        validate_merger_inputs!(merger)
    catch e
        @error "Refusing unsafe/incomplete merge: $e"
        return false
    end
    
    try
        # Analyze input files
        file_info = analyze_processor_files(merger)
        
        # Remove existing output file
        if isfile(merger.output_file)
            rm(merger.output_file)
        end
        
        # Create output directory if needed
        output_dir = dirname(merger.output_file)
        if !isempty(output_dir) && !isdir(output_dir)
            mkpath(output_dir)
        end
        create_empty_netcdf4_file!(merger.output_file)
        
        # Merge time coordinates
        merge_time_coordinates!(merger, merger.output_file, file_info)
        
        # Merge spatial coordinates
        merge_spatial_coordinates!(merger, merger.output_file, file_info)
        
        # Merge data variables
        merge_data_variables!(merger, merger.output_file, file_info)
        
        # Add global attributes to merged file. Failure means the artifact is not
        # complete and therefore must not authorize source cleanup.
        string_attrs = Dict{String, Any}()
        for (att_name, att_value) in file_info["global_attrs"]
            string_attrs[string(att_name)] = string(att_value)
        end
        ncputatt(merger.output_file, "global", string_attrs)

        verify_merged_output!(merger.output_file, file_info)
        
        merger.verbose && println("Merge completed successfully!")
        merger.verbose && println("   Output file size: $(round(filesize(merger.output_file)/1024/1024, digits=2)) MB")
        
        # Cleanup source files if requested
        if merger.cleanup
            cleanup_source_files!(merger)
        end
        
        return true
        
    catch e
        @error "Failed to merge files: $e"
        # Clean up partial output file
        if isfile(merger.output_file)
            rm(merger.output_file)
        end
        return false
    end
end

"""
Clean up source processor files after successful merge
"""
function cleanup_source_files!(merger::NetCDFMerger)
    merger.verbose && println("Cleaning up source files...")
    
    files_removed = 0
    for file in merger.processor_files
        if isfile(file)
            try
                rm(file)
                files_removed += 1
                merger.verbose && println("   Removed: $(basename(file))")
            catch e
                @warn "Could not remove $file: $e"
            end
        end
    end
    
    # Try to remove empty set directory
    if !isempty(merger.processor_files)
        set_dir = dirname(merger.processor_files[1])
        if isdir(set_dir) && isempty(readdir(set_dir))
            try
                rm(set_dir)
                merger.verbose && println("   Removed empty directory: $(basename(set_dir))")
            catch e
                merger.verbose && println("   Warning: Could not remove directory $set_dir: $e")
            end
        end
    end
    
    merger.verbose && println("   Cleaned up $files_removed files")
end

# Convenience functions matching Tarang post-processing style

"""
    merge_netcdf_files(base_name; kwargs...)

Merge per-processor NetCDF files into a single merged file.

# Arguments
- `base_name::String`: Base name of the handler (e.g., "snapshots", "analysis")
- `set_number::Int=1`: Set number to merge (default: 1)
- `output_name::String=""`: Output filename (default: auto-generated)
- `merge_mode::MergeMode=RECONSTRUCT`: How to combine processor data
- `cleanup::Bool=false`: Delete source files after successful merge
- `verbose::Bool=true`: Print progress information

# Examples
```julia
# Basic merge
merge_netcdf_files("snapshots")

# Advanced options  
merge_netcdf_files("analysis", 
                   set_number=2,
                   output_name="analysis_complete.nc", 
                   cleanup=true,
                   merge_mode=SIMPLE_CONCAT)
```
"""
function merge_netcdf_files(base_name::String; 
                           set_number::Int=1,
                           output_name::String="",
                           merge_mode::MergeMode=RECONSTRUCT,
                           cleanup::Bool=false,
                           verbose::Bool=true)
    
    merger = NetCDFMerger(base_name, 
                         set_number=set_number,
                         output_name=output_name,
                         merge_mode=merge_mode, 
                         cleanup=cleanup,
                         verbose=verbose)
    
    return merge_files!(merger)
end

"""
    batch_merge_netcdf(handlers; kwargs...)

Merge multiple handlers in batch mode.

# Examples
```julia
# Merge multiple handlers
batch_merge_netcdf(["snapshots", "analysis", "checkpoints"])

# With cleanup
batch_merge_netcdf(["snapshots", "analysis"], cleanup=true)
```
"""
function batch_merge_netcdf(handlers::Vector{String}; 
                           set_number::Int=1,
                           merge_mode::MergeMode=RECONSTRUCT,
                           cleanup::Bool=false,
                           verbose::Bool=true)
    
    results = Dict{String, Bool}()
    
    verbose && println("Starting batch merge of $(length(handlers)) handlers...")
    
    for handler in handlers
        verbose && println("\n" * "="^50)
        verbose && println("Processing handler: $handler")
        verbose && println("="^50)
        
        success = merge_netcdf_files(handler,
                                   set_number=set_number,
                                   merge_mode=merge_mode,
                                   cleanup=cleanup,
                                   verbose=verbose)
        results[handler] = success
        
        if success
            verbose && println("Successfully merged $handler")
        else
            verbose && println("Failed to merge $handler")  
        end
    end
    
    # Summary
    successful = count(values(results))
    verbose && println("\n" * "="^50)
    verbose && println("Batch merge complete: $successful/$(length(handlers)) successful")
    verbose && println("="^50)
    
    return results
end

"""
    find_mergeable_handlers(directory=".")

Find all handlers with processor files ready for merging.

Returns a dictionary mapping handler names to available set numbers.
"""
function find_mergeable_handlers(directory::String=".")
    handlers = Dict{String, Vector{Int}}()
    
    # Look for handler directories and files
    for entry in readdir(directory, join=true)
        entry_name = basename(entry)
        
        # Look for set directories (handler_s1, handler_s2, etc.)
        # Use regex to match the last _s followed by digits to avoid ambiguity
        # when handler names contain "_s" (e.g., "my_simulation_s1")
        if isdir(entry)
            match_result = match(r"^(.+)_s(\d+)$", entry_name)
            if match_result !== nothing && !isempty(readdir(entry))
                handler_name = match_result.captures[1]
                set_number = parse(Int, match_result.captures[2])
                if !haskey(handlers, handler_name)
                    handlers[handler_name] = Int[]
                end
                push!(handlers[handler_name], set_number)
            end
        end
        
        # Also look for direct processor files (handler_s1_p0.nc format)
        if isfile(entry) && occursin("_s", entry_name) && occursin("_p", entry_name) && endswith(entry_name, ".nc")
            # Extract handler name and set number
            match_result = match(r"^(.+)_s(\d+)_p\d+\.nc$", entry_name)
            if match_result !== nothing
                handler_name = match_result.captures[1]
                set_number = parse(Int, match_result.captures[2])
                if !haskey(handlers, handler_name)
                    handlers[handler_name] = Int[]
                end
                if !(set_number in handlers[handler_name])
                    push!(handlers[handler_name], set_number)
                end
            end
        end
    end
    
    # Sort set numbers for each handler
    for (handler, sets) in handlers
        sort!(sets)
    end
    
    return handlers
end

# Export main functions
export NetCDFMerger, MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP
export merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers
export merge_files!, cleanup_source_files!
