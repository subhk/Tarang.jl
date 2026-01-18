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
    info = ncinfo(first_file)
    
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
            data = ncread(first_file, coord)
            time_info[coord] = length(data)
        catch
            # Coordinate doesn't exist
            time_info[coord] = 0
        end
    end
    
    # Find data variables (exclude coordinate variables)
    data_vars = String[]
    coord_vars = String[]
    
    for var_name in [v.name for v in info.vars]
        if var_name in time_coords
            push!(coord_vars, var_name)
        elseif startswith(var_name, "dim_") || occursin(r"_dim\d+$", var_name)
            push!(coord_vars, var_name)  
        else
            push!(data_vars, var_name)
        end
    end
    
    file_info["global_attrs"] = global_attrs
    file_info["time_info"] = time_info
    file_info["data_vars"] = data_vars
    file_info["coord_vars"] = coord_vars
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
            time_data = ncread(merger.processor_files[1], coord_name)

            # Create time coordinate (nccreate must be called before ncwrite)
            nccreate(output_file, coord_name, "sim_time", length(time_data),
                    t=eltype(time_data),
                    atts=Dict("long_name" => coord_name,
                             "units" => coord_name == "sim_time" ? "dimensionless time" : "seconds",
                             "axis" => "T"))

            # Write time data
            ncwrite(time_data, output_file, coord_name)
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
        info = ncinfo(file)
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
        coord_data = nothing
        coord_attrs = Dict{String, Any}()
        
        for file in merger.processor_files
            try
                coord_data = ncread(file, coord_var)
                # Set standard NetCDF coordinate attributes
                coord_attrs["long_name"] = coord_var
                coord_attrs["axis"] = occursin("dim1", coord_var) ? "X" : (occursin("dim2", coord_var) ? "Y" : "Z")
                break
            catch
                continue
            end
        end
        
        if coord_data !== nothing
            # Create coordinate variable in output file
            nccreate(output_file, coord_var, coord_var, length(coord_data),
                    t=eltype(coord_data),
                    atts=coord_attrs)
            ncwrite(coord_data, output_file, coord_var)
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

"""
Simple concatenation merge: combine data along processor dimension
"""
function ensure_processor_coordinate!(output_file::String, n_procs::Int)
    exists = false
    try
        ncread(output_file, "processor", start=[1], count=[1])
        exists = true
    catch
        exists = false
    end

    if !exists
        proc_coord = collect(0:(n_procs - 1))
        nccreate(output_file, "processor", "processor", length(proc_coord),
                t=eltype(proc_coord),
                atts=Dict("long_name" => "MPI processor rank"))
        ncwrite(proc_coord, output_file, "processor")
    end
end

function merge_variable_concat!(merger::NetCDFMerger, output_file::String, var_name::String, file_info::Dict)
    # Read data from all processor files
    all_data = Any[]
    var_attrs = Dict{String, Any}()
    data_type = Float64
    dim_names = String[]
    
    for (i, file) in enumerate(merger.processor_files)
        try
            data = ncread(file, var_name)
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
        catch e
            merger.verbose && println("    Warning: Could not read $var_name from $(basename(file)): $e")
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
    skipped_count = 0

    for (i, data) in enumerate(all_data)
        if size(data) == size(first_data)
            # Build proper index tuple: (:, :, ..., :, i) for the i-th processor slice
            # Use selectdim-style indexing for dimension-agnostic assignment
            indices = ntuple(d -> d <= n_dims ? Colon() : i, n_dims + 1)
            combined_data[indices...] = data
        else
            # Data shape mismatch - log warning and skip
            skipped_count += 1
            if merger.verbose && skipped_count <= 3
                println("    Warning: Processor $i data shape $(size(data)) != expected $(size(first_data)), skipping")
            end
        end
    end

    if skipped_count > 0
        merger.verbose && println("    Skipped $skipped_count processors due to shape mismatch")
    end

    if skipped_count == n_procs
        merger.verbose && println("    Error: All processor data had mismatched shapes")
        return
    end
    
    # Create processor coordinate once
    ensure_processor_coordinate!(output_file, length(all_data))

    # Create variable in output file
    dim_names_with_proc = [dim_names..., "processor"]
    dim_sizes = [size(combined_data)...]

    # Use Julia types directly for NetCDF.jl
    nccreate(output_file, var_name, dim_names_with_proc, dim_sizes,
            t=data_type, atts=var_attrs)
    
    # Write combined data
    ncwrite(combined_data, output_file, var_name)
    
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
            data = ncread(file, var_name)
            
            # Try to read domain decomposition information (Tarang style)
            start_indices = nothing
            count_indices = nothing
            
            try
                # Look for Tarang-style attributes: 'start' and 'count'
                info = ncinfo(file)
                for var_info in info.vars
                    if var_info.name == var_name
                        # Try different attribute name conventions
                        start_indices = get(var_info.atts, "start", nothing)
                        count_indices = get(var_info.atts, "count", nothing)
                        
                        if start_indices === nothing
                            start_indices = get(var_info.atts, "domain_start", nothing)
                            count_indices = get(var_info.atts, "domain_count", nothing)
                        end
                        
                        # If we have global shape info, use it
                        if haskey(var_info.atts, "global_shape")
                            global_shape = var_info.atts["global_shape"]
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
                dim_names = ["sim_time"]  # First dimension is always time
                for j in 2:length(data_shape)
                    push!(dim_names, "$(var_name)_dim$(j-1)")
                end
                
                # If no global shape found, estimate from first processor
                if global_shape === nothing
                    global_shape = data_shape
                end
            end
            
        catch e
            merger.verbose && println("    Warning: Could not read $var_name from $(basename(file)): $e")
        end
    end
    
    if isempty(processor_data)
        merger.verbose && println("    No data found for $var_name")
        return
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
                    reconstructed_data[slices...] = data
                    coverage_mask[slices...] .= true
                    merger.verbose && println("        Placed data from $(basename(proc_info["file"])) at $slices")
                else
                    merger.verbose && println("        Size mismatch for $(basename(proc_info["file"])): expected $(size(reconstructed_data[slices...])), got $(size(data))")
                end
            catch e
                merger.verbose && println("        Error placing data from $(basename(proc_info["file"])): $e")
                # Fall back to overlaying at origin
                try
                    data_size = size(data)
                    origin_slices = tuple([1:s for s in data_size]...)
                    if size(data) == size(reconstructed_data[origin_slices...])
                        reconstructed_data[origin_slices...] = data
                        coverage_mask[origin_slices...] .= true
                    end
                catch
                    merger.verbose && println("        Could not place data from $(basename(proc_info["file"]))")
                end
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
        merger.verbose && println("        Warning: $uncovered_count grid points not covered by any processor")
        reconstructed_data[.!coverage_mask] .= NaN
    end

    # Create variable in output file
    var_attrs["reconstruction_coverage"] = "$(count(coverage_mask))/$(length(coverage_mask)) points covered"
    dim_sizes = [size(reconstructed_data)...]

    # Use Julia types directly for NetCDF.jl
    nccreate(output_file, var_name, dim_names, dim_sizes,
            t=data_type, atts=var_attrs)
    
    ncwrite(reconstructed_data, output_file, var_name)
    
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
            data = ncread(file, var_name)
            
            # Read layout information (following Tarang post:281)
            layout_info = Dict{String, Any}()
            try
                info = ncinfo(file)
                for var_info in info.vars
                    if var_info.name == var_name
                        # Key Tarang attributes that determine merging strategy
                        layout_info["grid_space"] = get(var_info.atts, "grid_space", nothing)
                        layout_info["layout"] = get(var_info.atts, "layout", nothing)
                        layout_info["scales"] = get(var_info.atts, "scales", nothing)
                        layout_info["constant"] = get(var_info.atts, "constant", false)
                        layout_info["start"] = get(var_info.atts, "start", nothing)
                        layout_info["count"] = get(var_info.atts, "count", nothing)
                        layout_info["global_shape"] = get(var_info.atts, "global_shape", nothing)
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
            merger.verbose && println("    Warning: Could not read $var_name from $(basename(file)): $e")
        end
    end
    
    if isempty(processor_data)
        merger.verbose && println("    No data found for $var_name")
        return
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

"""Determine field layout type from grid_space flags array."""
function determine_layout_type(grid_space_flags)
    if isa(grid_space_flags, AbstractArray) && length(grid_space_flags) > 0
        flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
        all_grid = all(flags)
        all_coeff = all(.!flags)
        
        if all_grid
            return :grid_space
        elseif all_coeff
            return :coeff_space
        else
            return :mixed_layout
        end
    else
        return :unknown
    end
end

function _bool_from_value(value)
    if value isa Bool
        return value
    elseif value isa Number
        return value != 0
    elseif value isa AbstractString
        lower = lowercase(strip(value))
        return lower in ("true", "t", "1", "yes", "y", "grid", "g")
    else
        return false
    end
end

"""
Normalize grid_space flags to a Bool vector matching data dimensions.
Assumes the first dimension is time and should not be transformed.
"""
function normalize_grid_space_flags(flags, ndims_data::Int)
    if flags === nothing
        return nothing
    end

    if flags isa AbstractArray
        bools = Bool[_bool_from_value(f) for f in flags]
    else
        bools = Bool[_bool_from_value(flags)]
    end

    if ndims_data <= 0
        return bools
    end

    if isempty(bools)
        return fill(true, ndims_data)
    end

    if length(bools) == ndims_data - 1
        return vcat(true, bools)
    elseif length(bools) == ndims_data
        return bools
    else
        # Fallback: assume all spatial dims are grid space
        return vcat(true, fill(bools[1], ndims_data - 1))
    end
end

"""
Merge field in grid space (physical space).
Grid space fields are typically distributed spatially across processors.
"""
function merge_grid_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    merger.verbose && println("        Merging grid space field")
    var_attrs["merge_strategy"] = "grid_space_spatial_reconstruction"
    
    # Use spatial reconstruction (like merge_variable_reconstruct!)
    data_type = eltype(processor_data[1]["data"])
    global_shape = nothing
    
    # Try to determine global shape from layout info
    for proc_info in processor_data
        layout_info = proc_info["layout_info"]
        if layout_info["global_shape"] !== nothing
            global_shape = layout_info["global_shape"]
            break
        end
    end
    
    # If no global shape, estimate from domain decomposition
    if global_shape === nothing
        inferred_shape = infer_slab_decomposition!(processor_data, merger)
        if inferred_shape !== nothing
            global_shape = inferred_shape
        else
            global_shape = estimate_global_shape_from_decomposition(processor_data)
        end
    end
    
    # Infer missing start/count metadata if needed
    has_domain_info = any(p["layout_info"]["start"] !== nothing && p["layout_info"]["count"] !== nothing for p in processor_data)
    if !has_domain_info
        inferred_shape = infer_slab_decomposition!(processor_data, merger)
        if inferred_shape !== nothing
            merger.verbose && println("        Inferred slab decomposition for grid space field")
            if global_shape === nothing
                global_shape = inferred_shape
            end
        end
    end

    # Initialize global field
    reconstructed_data = zeros(data_type, global_shape)
    coverage_mask = falses(global_shape)
    
    # Reconstruct using spatial domain information
    for proc_info in processor_data
        data = proc_info["data"]
        layout_info = proc_info["layout_info"]
        
        start_indices = layout_info["start"]
        count_indices = layout_info["count"]
        
        if start_indices !== nothing && count_indices !== nothing
            try
                # Create spatial slices (skip time dimension)
                spatial_slices = Any[Colon()]  # Time dimension
                for (s, c) in zip(start_indices, count_indices)
                    push!(spatial_slices, (s+1):(s+c))  # Convert to 1-based
                end
                
                slices = tuple(spatial_slices...)
                if size(data) == size(reconstructed_data[slices...])
                    reconstructed_data[slices...] = data
                    coverage_mask[slices...] .= true
                end
            catch e
                merger.verbose && println("        Error placing grid space data: $e")
            end
        end
    end
    
    # Write reconstructed field
    write_reconstructed_field(reconstructed_data, var_attrs, output_file, var_name, dim_names, data_type)
    merger.verbose && println("        Grid space field merged")
end

"""
Merge field in coefficient space (spectral space).
Coefficient space fields are typically distributed across spectral modes.
"""
function merge_coeff_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    merger.verbose && println("        Merging coefficient space field")  
    var_attrs["merge_strategy"] = "coeff_space_spectral_reconstruction"
    
    # Coefficient space often requires different handling than grid space
    # Modes may be distributed differently than spatial points
    
    data_type = eltype(processor_data[1]["data"])
    
    # Try to use mode-based reconstruction
    reconstructed_data = reconstruct_spectral_modes(processor_data, data_type, merger)
    
    if reconstructed_data !== nothing
        write_reconstructed_field(reconstructed_data, var_attrs, output_file, var_name, dim_names, data_type)
        merger.verbose && println("        Coefficient space field merged")
    else
        merger.verbose && println("        Falling back to spatial reconstruction for coefficient field")
        merge_grid_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    end
end

"""
Merge field with mixed layout (some dimensions in grid space, others in coefficient space).
Following Tarang patterns, mixed layouts are transformed to pure layouts before merging.

Based on Tarang post and field - mixed layout fields cannot be directly merged
and must be transformed to either pure grid space or pure coefficient space first.
"""
function merge_mixed_layout_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    merger.verbose && println("        Merging mixed layout field")
    var_attrs["merge_strategy"] = "mixed_layout_transform_and_reconstruct"
    
    # Analyze grid_space flags to understand layout pattern
    grid_space_flags = nothing
    sample_layout_info = processor_data[1]["layout_info"]
    
    if sample_layout_info["grid_space"] !== nothing
        grid_space_flags = sample_layout_info["grid_space"]
        merger.verbose && println("          Grid space pattern: $grid_space_flags")
    else
        merger.verbose && println("          No grid space info found, falling back to grid space merge")
        merge_grid_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
        return
    end
    
    # Determine target layout based on predominant layout and data characteristics
    target_layout = determine_optimal_target_layout(grid_space_flags, processor_data, merger)
    merger.verbose && println("          Target layout: $target_layout")
    
    var_attrs["original_layout"] = string(grid_space_flags)
    var_attrs["target_layout"] = string(target_layout)
    
    # Transform processor data to target layout
    transformed_data = transform_to_target_layout!(processor_data, grid_space_flags, target_layout, merger)
    
    if transformed_data !== nothing
        # Update layout info to reflect pure target layout
        for proc_info in transformed_data
            if target_layout == :grid_space
                proc_info["layout_info"]["grid_space"] = trues(length(grid_space_flags))
            else  # :coeff_space
                proc_info["layout_info"]["grid_space"] = falses(length(grid_space_flags))
            end
        end
        
        # Apply appropriate pure layout merging strategy
        if target_layout == :grid_space
            merger.verbose && println("          Applying grid space merge to transformed data")
            merge_grid_space_field!(transformed_data, var_attrs, output_file, var_name, dim_names, merger)
        else  # :coeff_space
            merger.verbose && println("          Applying coefficient space merge to transformed data")
            merge_coeff_space_field!(transformed_data, var_attrs, output_file, var_name, dim_names, merger)
        end
        
        var_attrs["layout_transformation"] = "mixed_to_$(target_layout)"
        merger.verbose && println("        Mixed layout field transformed and merged")
    else
        # Transformation failed, fall back to grid space merge with original data
        merger.verbose && println("          Layout transformation failed, using grid space fallback")
        var_attrs["layout_transformation"] = "failed_fallback_to_grid"
        merge_grid_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    end
end

"""
Determine optimal target layout for mixed layout transformation.
Following Tarang patterns - prefer grid space for most cases unless
coefficient space is clearly more appropriate.
"""
function determine_optimal_target_layout(grid_space_flags, processor_data, merger)
    if !isa(grid_space_flags, AbstractArray)
        return :grid_space
    end
    
    flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
    grid_dims = count(flags)
    coeff_dims = count(!, flags)
    
    merger.verbose && println("            Layout analysis: $grid_dims grid dims, $coeff_dims coeff dims")
    
    # Decision logic based on Tarang patterns:
    # 1. If majority are in grid space, transform to grid space
    # 2. If field is primarily spectral (more coeff dims), prefer coefficient space
    # 3. For tie cases, prefer grid space (Tarang default for output)
    
    if grid_dims >= coeff_dims
        return :grid_space
    else
        # More coefficient dimensions - check if this looks like a spectral field
        sample_data = processor_data[1]["data"]
        
        # Heuristic: if data is complex or has spectral characteristics, prefer coefficient
        if eltype(sample_data) <: Complex
            merger.verbose && println("            Complex data detected, preferring coefficient space")
            return :coeff_space
        else
            # For real data, still prefer grid space for easier interpretation
            return :grid_space
        end
    end
end

"""
Transform mixed layout processor data to target pure layout.
This implements the equivalent of Tarang layout transformation operations.
"""
function transform_to_target_layout!(processor_data, grid_space_flags, target_layout, merger)
    merger.verbose && println("            Transforming $(length(processor_data)) processor datasets")

    transformed_data = Any[]
    
    for proc_info in processor_data
        try
            original_data = proc_info["data"]
            layout_info = proc_info["layout_info"]
            
            # Apply layout transformation 
            transformed_field = apply_layout_transformation(original_data, grid_space_flags, target_layout, merger)
            
            if transformed_field !== nothing
                # Create new processor info with transformed data
                new_proc_info = Dict(
                    "data" => transformed_field,
                    "layout_info" => deepcopy(layout_info),
                    "file" => proc_info["file"]
                )
                push!(transformed_data, new_proc_info)
            else
                merger.verbose && println("              Failed to transform data from $(basename(proc_info["file"]))")
                return nothing
            end
            
        catch e
            merger.verbose && println("              Error transforming $(basename(proc_info["file"])): $e")
            return nothing
        end
    end
    
    merger.verbose && println("            Successfully transformed $(length(transformed_data)) datasets")
    return transformed_data
end

"""
Apply layout transformation to individual field data.

Transforms data between grid space and coefficient space layouts using FFT/IFFT.

Arguments:
- field_data: Array of field values (can be real or complex)
- grid_space_flags: Tuple/Vector of booleans indicating which dimensions are in grid space
                   (true = grid space, false = coefficient space)
- target_layout: :grid_space or :coeff_space
- merger: NetCDFMerger instance for configuration

Returns:
- Transformed array, or nothing if transformation fails
"""
function apply_layout_transformation(field_data, grid_space_flags, target_layout, merger)
    try
        # Validate inputs
        if field_data === nothing || isempty(field_data)
            return nothing
        end

        ndims_data = ndims(field_data)

        grid_space_flags = normalize_grid_space_flags(grid_space_flags, ndims_data)
        if grid_space_flags === nothing
            merger.verbose && println("              Warning: grid_space_flags missing, using default")
            grid_space_flags = fill(true, ndims_data)
        end

        # Check if transformation is needed
        flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
        all_grid = all(flags)
        all_coeff = all(.!flags)

        if target_layout == :grid_space && all_grid
            # Already in grid space
            merger.verbose && println("              Data already in grid space")
            return field_data
        elseif target_layout == :coeff_space && all_coeff
            # Already in coefficient space
            merger.verbose && println("              Data already in coefficient space")
            return field_data
        end

        # Perform transformation
        if target_layout == :grid_space
            return transform_to_grid_space(field_data, grid_space_flags, merger)
        else  # target_layout == :coeff_space
            return transform_to_coeff_space(field_data, grid_space_flags, merger)
        end

    catch e
        merger.verbose && println("              Layout transformation failed: $e")
        @debug "Layout transformation error" exception=(e, catch_backtrace())
        return nothing
    end
end

"""Transform field data to grid space by applying inverse FFT to coefficient dimensions."""
function transform_to_grid_space(field_data, grid_space_flags, merger)
    result = copy(field_data)
    input_real = eltype(field_data) <: Real
    ndims_data = ndims(result)

    # Convert to complex if needed for FFT operations
    if eltype(result) <: Real
        result = complex(result)
    end

    transforms_applied = 0

    # Apply inverse FFT to each dimension that is in coefficient space
    for dim in 1:ndims_data
        if dim == 1
            continue  # Skip time dimension
        end
        if !grid_space_flags[dim]
            # This dimension is in coefficient space - apply inverse FFT
            try
                # Use FFTW for the inverse transform along this dimension
                result = apply_ifft_along_dim(result, dim)
                transforms_applied += 1
            catch e
                merger.verbose && println("              IFFT failed for dimension $dim: $e")
                # Continue with other dimensions
            end
        end
    end

    merger.verbose && println("              Applied $transforms_applied inverse transforms to grid space")

    # Return real part if the result should be real-valued
    # (for physical fields, the imaginary part should be negligible)
    if input_real && !isempty(result)
        # Check if imaginary part is negligible
        max_imag = maximum(abs.(imag(result)))
        max_real = maximum(abs.(real(result)))
        if max_real > 0 && max_imag / max_real < 1e-10
            return real(result)
        end
    end

    return result
end

"""Transform field data to coefficient space by applying forward FFT to grid dimensions."""
function transform_to_coeff_space(field_data, grid_space_flags, merger)
    result = copy(field_data)
    ndims_data = ndims(result)

    # Convert to complex for FFT operations
    if eltype(result) <: Real
        result = complex(result)
    end

    transforms_applied = 0

    # Apply forward FFT to each dimension that is in grid space
    for dim in 1:ndims_data
        if dim == 1
            continue  # Skip time dimension
        end
        if grid_space_flags[dim]
            # This dimension is in grid space - apply forward FFT
            try
                result = apply_fft_along_dim(result, dim)
                transforms_applied += 1
            catch e
                merger.verbose && println("              FFT failed for dimension $dim: $e")
                # Continue with other dimensions
            end
        end
    end

    merger.verbose && println("              Applied $transforms_applied forward transforms to coefficient space")

    return result
end

"""
Apply forward FFT along a specific dimension.
Uses normalized FFT (1/N factor applied).
"""
function apply_fft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Complex
    n = size(data, dim)

    # Create FFT plan for this dimension
    # We use fft with the dims keyword to transform along a specific axis
    result = fft(data, dim)

    # Normalize by 1/N for proper spectral coefficients
    result ./= n

    return result
end

"""Apply FFT to real data along a specific dimension."""
function apply_fft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Real
    return apply_fft_along_dim(complex(data), dim)
end

"""
Apply inverse FFT along a specific dimension.
Uses unnormalized IFFT (multiply by N to invert the forward normalization).
"""
function apply_ifft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Complex
    n = size(data, dim)

    # Apply inverse FFT along specified dimension
    result = ifft(data, dim)

    # IFFT in Julia is already normalized by 1/N, but we used 1/N in forward FFT
    # So we need to multiply by N to get back the original values
    result .*= n

    return result
end

"""Apply IFFT to real data along a specific dimension."""
function apply_ifft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Real
    return apply_ifft_along_dim(complex(data), dim)
end

"""
Attempt to detect whether data is in grid space or coefficient space
based on data characteristics.

Heuristics:
1. Complex data with significant imaginary parts likely in coefficient space
2. Data with values concentrated near zero indices likely in coefficient space
3. Smooth real data likely in grid space

Returns a tuple of booleans (grid_space_flags) for each dimension.
"""
function detect_layout_from_data(field_data, field_name::String="")
    ndims_data = ndims(field_data)

    # Default: assume grid space
    grid_space_flags = fill(true, ndims_data)

    # Check if data is complex
    if eltype(field_data) <: Complex && !isempty(field_data)
        # Check imaginary content
        total_mag = sum(abs.(field_data))
        imag_mag = sum(abs.(imag(field_data)))

        if total_mag > 0 && imag_mag / total_mag > 0.01
            # Significant imaginary content - likely coefficient space
            # For Fourier dimensions, mark as coefficient space
            for dim in 1:ndims_data
                # Check if energy is concentrated at low wavenumbers
                if is_spectral_dimension(field_data, dim)
                    grid_space_flags[dim] = false
                end
            end
        end
    end

    return tuple(grid_space_flags...)
end

"""
Check if a dimension appears to be in spectral (coefficient) space
by examining the energy distribution.

In coefficient space, energy is typically concentrated at low wavenumbers.
"""
function is_spectral_dimension(data::AbstractArray, dim::Int)
    n = size(data, dim)
    if n < 4
        return false  # Too small to determine
    end

    # Sum absolute values along this dimension
    # Move the target dimension to first position for easier slicing
    perm = collect(1:ndims(data))
    perm[1], perm[dim] = perm[dim], perm[1]
    permuted = permutedims(data, perm)

    # Compute energy in low vs high wavenumber regions
    quarter_n = max(1, n ÷ 4)

    # Low wavenumber region (first and last quarter for symmetric spectra)
    low_k_energy = sum(abs.(selectdim(permuted, 1, 1:quarter_n))) +
                   sum(abs.(selectdim(permuted, 1, (n - quarter_n + 1):n)))

    # High wavenumber region (middle half)
    mid_start = quarter_n + 1
    mid_end = n - quarter_n
    if mid_end >= mid_start
        high_k_energy = sum(abs.(selectdim(permuted, 1, mid_start:mid_end)))
    else
        high_k_energy = 0.0
    end

    total_energy = low_k_energy + high_k_energy

    if total_energy == 0
        return false
    end

    # If more than 80% of energy is in low wavenumbers, likely spectral
    return low_k_energy / total_energy > 0.8
end

"""Convert grid_space_flags to a readable string."""
function get_layout_string(grid_space_flags)
    parts = [flag ? "G" : "C" for flag in grid_space_flags]
    return join(parts, "-")
end

"""Parse layout string like 'G-C-G' to grid_space_flags tuple."""
function parse_layout_string(layout_str::String)
    parts = split(layout_str, "-")
    return tuple([uppercase(strip(p)) == "G" for p in parts]...)
end

"""
Infer start/count indices by assuming a slab decomposition along the last spatial dimension.
Returns inferred global shape, or nothing if inference is not possible.
"""
function infer_slab_decomposition!(processor_data, merger)
    if isempty(processor_data)
        return nothing
    end

    sample = processor_data[1]["data"]
    ndims_data = ndims(sample)
    if ndims_data < 2
        return nothing
    end

    spatial_ndims = ndims_data - 1
    shapes = [size(proc["data"])[2:end] for proc in processor_data]
    base = shapes[1]

    if spatial_ndims > 1
        for dim in 1:(spatial_ndims - 1)
            if any(shape[dim] != base[dim] for shape in shapes)
                merger.verbose && println("        Cannot infer slab decomposition: spatial dims vary beyond last axis")
                return nothing
            end
        end
    end

    offset = 0
    for (proc, shape) in zip(processor_data, shapes)
        start = zeros(Int, spatial_ndims)
        count = collect(shape)
        start[end] = offset
        proc["start"] = tuple(start...)
        proc["count"] = tuple(count...)
        if haskey(proc, "layout_info")
            proc["layout_info"]["start"] = tuple(start...)
            proc["layout_info"]["count"] = tuple(count...)
        end
        offset += shape[end]
    end

    prefix = spatial_ndims > 1 ? base[1:end-1] : ()
    return (size(sample, 1), prefix..., offset)
end

"""Estimate global shape from domain decomposition info."""
function estimate_global_shape_from_decomposition(processor_data)
    sample_data = processor_data[1]["data"]
    sample_shape = collect(size(sample_data))
    
    # Try to find maximum extents from start+count info
    for proc_info in processor_data
        layout_info = proc_info["layout_info"]
        start_indices = layout_info["start"]
        count_indices = layout_info["count"]
        
        if start_indices !== nothing && count_indices !== nothing
            for (i, (s, c)) in enumerate(zip(start_indices, count_indices))
                dim_index = i + 1  # Skip time dimension
                if dim_index <= length(sample_shape)
                    sample_shape[dim_index] = max(sample_shape[dim_index], s + c)
                end
            end
        end
    end
    
    return tuple(sample_shape...)
end

"""
Reconstruct spectral coefficient field from distributed modes.

Based on Tarang distributor Layout class and post merge_data function.
Handles block distribution of spectral coefficients across processors.
"""
function reconstruct_spectral_modes(processor_data, data_type, merger)
    merger.verbose && println("          Reconstructing spectral coefficient field")

    if isempty(processor_data)
        merger.verbose && println("            No processor data available")
        return nothing
    end
    
    # Extract metadata from first processor to determine global structure
    first_proc_info = processor_data[1]
    first_layout_info = first_proc_info["layout_info"]
    
    # Check if this is actually a coefficient space field
    grid_space_flags = first_layout_info["grid_space"]
    if grid_space_flags === nothing
        merger.verbose && println("            No grid_space flags found")
        return nothing
    end
    
    # For pure coefficient space, all grid_space flags should be false
    if isa(grid_space_flags, AbstractArray)
        flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
        if any(flags)
            merger.verbose && println("            Mixed or grid space field, not pure coefficient")
            return nothing
        end
    elseif grid_space_flags != false
        merger.verbose && println("            Not coefficient space field")
        return nothing
    end
    
    # Get global shape from processor metadata
    global_shape = first_layout_info["global_shape"]
    if global_shape === nothing
        merger.verbose && println("            No global shape information available")
        return nothing
    end
    
    merger.verbose && println("            Global coefficient array shape: $global_shape")
    
    # Initialize global coefficient array
    global_coeffs = zeros(data_type, global_shape...)
    filled_mask = zeros(Bool, global_shape...)
    
    merger.verbose && println("            Processing $(length(processor_data)) processor datasets")
    
    # Reconstruct by placing each processor's data at correct indices
    processors_placed = 0
    for proc_info in processor_data
        layout_info = proc_info["layout_info"]
        start_indices = layout_info["start"]
        count_sizes = layout_info["count"]
        local_data = proc_info["data"]
        proc_file = basename(proc_info["file"])
        
        if start_indices === nothing || count_sizes === nothing
            merger.verbose && println("            Missing start/count metadata for $proc_file")
            continue
        end
        
        try
            # Convert to Julia 1-based indexing and create slices
            # Skip time dimension (index 1), work on spatial/spectral dimensions
            spatial_slices = Any[Colon()]  # Time dimension - take all
            
            for (start_idx, count_size) in zip(start_indices, count_sizes)
                # Convert from 0-based to 1-based indexing
                julia_start = start_idx + 1
                julia_end = start_idx + count_size
                push!(spatial_slices, julia_start:julia_end)
            end
            
            slices = tuple(spatial_slices...)
            
            # Verify dimensions match
            expected_size = size(global_coeffs[slices...])
            actual_size = size(local_data)
            
            if expected_size != actual_size
                merger.verbose && println("            Size mismatch for $proc_file: expected $expected_size, got $actual_size")
                continue
            end
            
            # Check for overlap (should not happen with proper distribution)
            if any(filled_mask[slices...])
                merger.verbose && println("            Detected overlap for $proc_file at $slices")
                return nothing
            end
            
            # Place data and mark as filled
            global_coeffs[slices...] = local_data
            filled_mask[slices...] .= true
            processors_placed += 1
            
            merger.verbose && println("              Placed coefficients from $proc_file at $slices")
            
        catch e
            merger.verbose && println("            Error placing data from $proc_file: $e")
            continue
        end
    end
    
    # Verify complete reconstruction
    uncovered_points = count(!, filled_mask)
    total_points = length(filled_mask)
    coverage_fraction = count(filled_mask) / total_points
    
    merger.verbose && println("            Reconstruction coverage: $(processors_placed)/$(length(processor_data)) processors")
    merger.verbose && println("            Coverage fraction: $(round(coverage_fraction * 100, digits=1))% ($uncovered_points uncovered points)")
    
    if uncovered_points > 0
        if coverage_fraction < 0.9
            merger.verbose && println("            Incomplete coefficient reconstruction (< 90% coverage)")
            return nothing
        else
            merger.verbose && println("            Minor gaps detected, filling with zeros")
            global_coeffs[.!filled_mask] .= 0.0
        end
    end
    
    # Verify spectral field characteristics
    if validate_spectral_coefficients(global_coeffs, merger)
        merger.verbose && println("            Successfully reconstructed spectral coefficient field")
        return global_coeffs
    else
        merger.verbose && println("            Reconstructed data failed spectral validation")
        return nothing
    end
end

"""
Validate reconstructed spectral coefficients for basic sanity checks.
Based on typical spectral field characteristics.
"""
function validate_spectral_coefficients(coeffs, merger)
    try
        # Basic checks for spectral coefficient arrays
        
        # 1. Check for reasonable coefficient magnitudes
        max_coeff = maximum(abs.(coeffs))
        if max_coeff == 0.0
            merger.verbose && println("              Warning: All coefficients are zero")
            return false
        end
        
        # 2. Check for numerical stability (no infinities or NaNs)
        if !all(isfinite.(coeffs))
            merger.verbose && println("              Error: Non-finite coefficients detected")
            return false
        end
        
        # 3. For complex coefficients, check Hermitian symmetry where applicable
        if eltype(coeffs) <: Complex
            merger.verbose && println("              Complex spectral coefficients detected")
            # Could add Hermitian symmetry checks for Fourier coefficients here
        end
        
        # 4. Check for reasonable dynamic range
        nonzero_coeffs = coeffs[abs.(coeffs) .> 1e-12 * max_coeff]
        if length(nonzero_coeffs) == 0
            merger.verbose && println("              Warning: No significant coefficients found")
            return false
        end
        
        dynamic_range = log10(max_coeff / minimum(abs.(nonzero_coeffs)))
        if dynamic_range > 15  # More than 15 orders of magnitude might indicate numerical issues
            merger.verbose && println("              Warning: Very large dynamic range ($dynamic_range orders)")
        end
        
        merger.verbose && println("              Spectral validation passed ($(length(nonzero_coeffs)) significant modes)")
        return true
        
    catch e
        merger.verbose && println("              Spectral validation failed: $e")
        return false
    end
end

"""Write reconstructed field data to NetCDF file."""
function write_reconstructed_field(data, var_attrs, output_file, var_name, dim_names, data_type)
    dim_sizes = [size(data)...]

    # Use Julia types directly for NetCDF.jl
    nccreate(output_file, var_name, dim_names, dim_sizes,
            t=data_type, atts=var_attrs)

    ncwrite(data, output_file, var_name)
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
        
        # Merge time coordinates
        merge_time_coordinates!(merger, merger.output_file, file_info)
        
        # Merge spatial coordinates
        merge_spatial_coordinates!(merger, merger.output_file, file_info)
        
        # Merge data variables
        merge_data_variables!(merger, merger.output_file, file_info)
        
        # Add global attributes to merged file
        try
            # ncputatt expects (filename, varname, Dict)
            # Convert all values to strings for global attributes
            string_attrs = Dict{String, Any}()
            for (att_name, att_value) in file_info["global_attrs"]
                string_attrs[string(att_name)] = string(att_value)
            end
            ncputatt(merger.output_file, "global", string_attrs)
        catch e
            merger.verbose && println("  Warning: Could not write global attributes: $e")
        end
        
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
