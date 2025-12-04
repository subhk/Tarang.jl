"""
NetCDF Output Support for Tarang.jl

This module provides NetCDF output capabilities matching Dedalus file handling style:
- Per-processor NetCDF files following Dedalus naming: handler_s1/handler_s1_p0.nc
- User-selectable precision (Float32/Float64)  
- Rich metadata and coordinate information following Dedalus structure
- CF conventions compliance
- Set-based file organization with automatic file management
- Task-based field output system

Matches Dedalus evaluator.add_file_handler() API pattern.
"""

using NetCDF
using MPI
using Dates
using Printf

# Precision types for user selection
const NetCDFPrecision = Union{Type{Float32}, Type{Float64}}

"""
NetCDF File Handler matching Dedalus H5FileHandler structure

Follows Dedalus pattern:
- base_path/handler_name_s1/handler_name_s1_p0.nc for processor files
- base_path/handler_name_s1.nc for gathered files
- /scales/ group with time coordinates
- /tasks/ group with field data
"""
mutable struct NetCDFFileHandler
    # Base attributes (matching Dedalus)
    base_path::String
    name::String
    dist::Any  # Domain distributor
    vars::Dict{String, Any}  # Variables for parsing
    
    # Scheduling (matching Dedalus Handler)
    group::Union{String, Nothing}
    wall_dt::Union{Float64, Nothing}
    sim_dt::Union{Float64, Nothing}
    iter::Union{Int, Nothing}
    max_writes::Union{Int, Nothing}
    
    # File management
    set_num::Int
    total_write_num::Int
    file_write_num::Int
    mode::String  # 'overwrite' or 'append'
    
    # NetCDF specific
    precision::NetCDFPrecision
    parallel::String  # 'gather', 'virtual', or 'mpio'
    
    # Tasks and metadata
    tasks::Vector{Dict{String, Any}}
    
    # MPI info
    comm::Any
    rank::Int
    size::Int
    
    function NetCDFFileHandler(base_path::String, dist, vars;
                              group=nothing, wall_dt=nothing, sim_dt=nothing, iter=nothing,
                              max_writes=nothing, mode="overwrite", 
                              precision::NetCDFPrecision=Float64,
                              parallel="gather")
        
        # MPI setup - defer until MPI is initialized
        comm = nothing
        rank, size = 0, 1
        
        # Base path handling (matching Dedalus)
        if endswith(base_path, ".nc")
            base_path = base_path[1:end-3]  # Remove .nc extension
        end
        
        # Extract handler name from path
        name = splitpath(base_path)[end]
        
        # Initialize file numbering 
        set_num = 1
        total_write_num = 0
        file_write_num = 0
        
        # Mode handling (matching Dedalus logic)
        if rank == 0 && mode == "overwrite"
            # Clean up existing files matching pattern
            for file in readdir(".", join=true)
                if startswith(basename(file), "$(name)_s") && (endswith(file, ".nc") || isdir(file))
                    if isdir(file)
                        rm(file, recursive=true)
                    else
                        rm(file)
                    end
                end
            end
        end
        
        handler = new(base_path, name, dist, vars,
                     group, wall_dt, sim_dt, iter, max_writes,
                     set_num, total_write_num, file_write_num, mode,
                     precision, parallel,
                     Vector{Dict{String, Any}}(),
                     comm, rank, size)
        
        # Create output directory structure (defer to when first used)
        # This is done later to avoid calling current_path too early
        
        # MPI setup will be done when first needed
        
        return handler
    end
end

"""
Initialize MPI information for handler
"""
function init_mpi!(handler::NetCDFFileHandler)
    if handler.comm === nothing && MPI.Initialized()
        handler.comm = MPI.COMM_WORLD
        handler.rank = MPI.Comm_rank(handler.comm)
        handler.size = MPI.Comm_size(handler.comm)
    end
end

"""
Get current set path following Dedalus naming: handler_name_s1/
"""
function current_path(handler::NetCDFFileHandler)
    set_name = "$(handler.name)_s$(handler.set_num)"
    return joinpath(dirname(handler.base_path), set_name)
end

"""
Get current file path following Dedalus naming: handler_name_s1/handler_name_s1_p0.nc
"""
function current_file(handler::NetCDFFileHandler)
    init_mpi!(handler)  # Ensure MPI info is available
    set_name = "$(handler.name)_s$(handler.set_num)"
    if handler.parallel == "gather" && handler.size == 1
        # Single file for serial runs
        return joinpath(current_path(handler), "$(set_name).nc")
    else
        # Per-processor files
        proc_name = @sprintf("%s_p%d.nc", set_name, handler.rank)
        return joinpath(current_path(handler), proc_name)
    end
end

"""
Add task to handler (matching Dedalus API)

# Example usage (matching Dedalus style):
snapshots = evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
"""
function add_task!(handler::NetCDFFileHandler, task; layout="g", name=nothing, scales=nothing, postprocess=nothing)
    # Default name
    if name === nothing
        name = string(task)
    end
    
    # Create operator following Dedalus patterns
    operator = create_operator(task, handler.vars, handler.dist)
    
    # Check and remedy scales following Dedalus logic
    if is_locked_field(operator)
        # For locked fields, use domain dealias scales
        if scales === nothing
            scales = get_domain_dealias(operator)
        else
            scales = remedy_scales(handler.dist, scales)
            if scales != get_domain_dealias(operator)
                scales = get_domain_dealias(operator)
                @warn "Cannot specify non-dealias scales for LockedFields"
            end
        end
    else
        scales = remedy_scales(handler.dist, scales)
    end
    
    # Get layout object
    layout_obj = get_layout_object(handler.dist, layout)
    
    # Create task dictionary (matching Dedalus structure)
    task_dict = Dict{String, Any}(
        "operator" => operator,
        "layout" => layout_obj,
        "scales" => scales,
        "name" => name,
        "dtype" => get_operator_dtype(operator, handler.precision),
        "postprocess" => postprocess
    )
    
    # Add data distribution information following Dedalus get_data_distribution
    global_shape, local_start, local_shape = get_data_distribution(handler, task_dict)
    task_dict["global_shape"] = global_shape
    task_dict["local_start"] = local_start
    task_dict["local_shape"] = local_shape
    task_dict["local_size"] = prod(local_shape)
    task_dict["local_slices"] = tuple(i:j for (i,j) in zip(local_start .+ 1, local_start .+ local_shape))
    
    push!(handler.tasks, task_dict)
    return handler
end

"""
Create operator from different input types following Dedalus patterns
"""
function create_operator(task, vars::Dict, dist)
    if isa(task, AbstractString)
        # Parse string expression like "u*v" or "d(u,x)" using vars namespace
        return parse_field_expression(task, vars, dist)
    elseif isa(task, ScalarField) || isa(task, VectorField) || isa(task, TensorField)
        # Existing field/operator
        return task
    elseif isa(task, Dict)
        return task
    else
        # Assume it's already an operator
        return task
    end
end

"""
Parse field expression from string (placeholder - needs proper implementation)
"""
function parse_field_expression(expr_str::String, vars::Dict, dist)
    # This would use the parse_expression function from problems.jl
    # and create appropriate field operators
    # For now, return a placeholder that represents the parsed expression
    return Dict("type" => "parsed_expression", "expr" => expr_str, "vars" => vars)
end

"""
Create copy operator for existing fields
"""
function create_copy_operator(field)
    return Dict("type" => "copy", "operand" => field)
end

"""
Check if operator represents a locked field.
Locked fields have fixed scales and cannot be rescaled.
"""
function is_locked_field(operator)
    # Check for Dict-based operators
    if isa(operator, Dict)
        return get(operator, "locked", false) == true
    end

    # Check for field types with locked property
    if hasproperty(operator, :locked)
        return getproperty(operator, :locked) == true
    end

    # Check for LockedField type (if defined)
    type_name = string(typeof(operator))
    if occursin("Locked", type_name)
        return true
    end

    return false
end

"""
Check if input is a field object
"""
function is_field(obj)
    return isa(obj, ScalarField) || isa(obj, VectorField) || isa(obj, TensorField)
end

"""
Get domain dealias scales from operator
"""
function get_domain_dealias(operator)
    # Placeholder - would extract dealias scales from operator's domain
    return nothing
end

"""
Remedy scales using distributor
"""
function remedy_scales(dist, scales)
    # Placeholder - would call dist.remedy_scales equivalent
    return scales
end

"""
Get layout object from distributor
"""
function get_layout_object(dist, layout)
    # Placeholder - would call dist.get_layout_object equivalent
    return layout
end

"""
Get operator data type.
"""
function get_operator_dtype(operator, default_precision)
    # Check for Dict-based operators
    if isa(operator, Dict)
        return get(operator, "dtype", default_precision)
    end

    # Check for field types with dtype property
    if isa(operator, ScalarField)
        return operator.dtype
    elseif isa(operator, VectorField)
        return operator.components[1].dtype
    elseif isa(operator, TensorField)
        return operator.components[1, 1].dtype
    end

    # Check for generic dtype property
    if hasproperty(operator, :dtype)
        return getproperty(operator, :dtype)
    end

    return default_precision
end

"""
Get data distribution information following Dedalus patterns
"""
function get_data_distribution(handler::NetCDFFileHandler, task::Dict, rank=nothing)
    init_mpi!(handler)  # Ensure MPI info is available
    
    if rank === nothing
        rank = handler.rank
    end
    
    layout = task["layout"]
    scales = task["scales"]
    operator = task["operator"]
    
    # Get domain and tensor signature from operator
    # For now, use placeholder values - real implementation would query operator
    domain = get_operator_domain(operator)
    tensorsig = get_operator_tensorsig(operator)
    
    # Calculate shapes using layout - placeholder implementation
    global_shape = get_global_shape(layout, domain, scales)
    local_shape = get_local_shape(layout, domain, scales, rank)
    local_start = get_local_start(layout, domain, scales, rank)
    
    return global_shape, local_start, local_shape
end

"""
Get domain from operator (placeholder)
"""
function get_operator_domain(operator)
    if isa(operator, Dict) && haskey(operator, "shape")
        return Dict("dims" => length(operator["shape"]), "shape" => operator["shape"])
    elseif isa(operator, ScalarField)
        if operator.data_g === nothing
            ensure_layout!(operator, :g)
        end
        return Dict("dims" => ndims(operator.data_g), "shape" => size(operator.data_g))
    elseif isa(operator, VectorField)
        first_comp = operator.components[1]
        if first_comp.data_g === nothing
            ensure_layout!(first_comp, :g)
        end
        return Dict("dims" => ndims(first_comp.data_g), "shape" => size(first_comp.data_g))
    elseif isa(operator, TensorField)
        first_comp = operator.components[1, 1]
        if first_comp.data_g === nothing
            ensure_layout!(first_comp, :g)
        end
        return Dict("dims" => ndims(first_comp.data_g), "shape" => size(first_comp.data_g))
    end
    return Dict("dims" => 0, "shape" => ())
end

"""
Get tensor signature from operator (placeholder)
"""
function get_operator_tensorsig(operator)
    if isa(operator, ScalarField)
        return ()
    elseif isa(operator, VectorField)
        return (:component,)
    elseif isa(operator, TensorField)
        return (:component_i, :component_j)
    else
        return ()
    end
end

"""
Get global shape from layout (placeholder)
"""
function get_global_shape(layout, domain, scales)
    return domain["shape"]
end

"""
Get local shape from layout (placeholder)
"""
function get_local_shape(layout, domain, scales, rank)
    return domain["shape"]
end

"""
Get local start from layout (placeholder)
"""
function get_local_start(layout, domain, scales, rank)
    return ntuple(_ -> 0, length(domain["shape"]))
end

"""
Check if handler should process based on schedule (matching Dedalus logic)
"""
function check_schedule(handler::NetCDFFileHandler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.0)
    scheduled = false
    
    # Iteration cadence
    if handler.iter !== nothing
        if iteration % handler.iter == 0
            scheduled = true
        end
    end
    
    # Simulation time cadence
    if handler.sim_dt !== nothing
        # Simplified logic - in real Dedalus this is more complex
        sim_div = floor(Int, sim_time / handler.sim_dt)
        if sim_div > handler.total_write_num
            scheduled = true
        end
    end
    
    # Wall time cadence 
    if handler.wall_dt !== nothing
        wall_div = floor(Int, wall_time / handler.wall_dt) 
        if wall_div > handler.total_write_num
            scheduled = true
        end
    end
    
    return scheduled
end

"""
Create NetCDF file with Dedalus-style structure
"""
function create_current_file!(handler::NetCDFFileHandler)
    filename = current_file(handler)
    
    # Create file directory if needed
    dir = dirname(filename)
    if !isdir(dir)
        mkpath(dir)
    end
    
    # Create basic structure matching Dedalus HDF5 layout
    
    # Create time coordinate (unlimited)
    nccreate(filename, "sim_time", "sim_time", 0,  # 0 = unlimited
            atts=Dict("long_name" => "simulation time",
                     "units" => "dimensionless time",
                     "axis" => "T"))
                     
    # Create other time-related scales
    nccreate(filename, "wall_time", "sim_time", 0,
            atts=Dict("long_name" => "wall clock time",
                     "units" => "seconds"))
                     
    nccreate(filename, "timestep", "sim_time", 0,  
            atts=Dict("long_name" => "timestep",
                     "units" => "dimensionless time"))
                     
    nccreate(filename, "iteration", "sim_time", 0,
            atts=Dict("long_name" => "iteration number"))
            
    nccreate(filename, "write_number", "sim_time", 0,
            atts=Dict("long_name" => "write number"))
    
    # Add global attributes (matching Dedalus metadata)
    # Note: NetCDF.jl doesn't support Bool attributes, so we use Int (0/1)
    global_attrs = Dict(
        "set_number" => handler.set_num,
        "handler_name" => handler.name,
        "writes" => handler.file_write_num,
        "title" => "Tarang.jl simulation output",
        "institution" => "Generated by Tarang.jl",
        "source" => "Tarang.jl - Julia implementation of Dedalus",
        "history" => "Created on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))",
        "Conventions" => "CF-1.8",
        "tarang_version" => "0.1.0",
        "software" => "Tarang",
        "software_repository" => "https://github.com/subhajitkar/Tarang.jl",
        "restart_compatible" => 0  # 0 = false, 1 = true (NetCDF doesn't support Bool)
    )
    
    # Add MPI information
    if handler.size > 1
        global_attrs["mpi_size"] = handler.size
        global_attrs["processor_rank"] = handler.rank
    end
    
    ncputatt(filename, "NC_GLOBAL", global_attrs)
    
    return true
end

"""
Process handler: write all tasks to NetCDF (matching Dedalus process method)
"""
function process!(handler::NetCDFFileHandler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.0)
    # Update write counts
    handler.total_write_num += 1
    handler.file_write_num += 1
    
    # Move to next set if necessary
    if handler.max_writes !== nothing && handler.file_write_num > handler.max_writes
        handler.set_num += 1
        handler.file_write_num = 1
    end
    
    # Ensure current file exists
    filename = current_file(handler)
    if !isfile(filename)
        create_current_file!(handler)
    end
    
    # Write time metadata
    write_index = handler.file_write_num
    ncwrite([sim_time], filename, "sim_time", start=[write_index])
    ncwrite([wall_time], filename, "wall_time", start=[write_index])  
    ncwrite([timestep], filename, "timestep", start=[write_index])
    ncwrite([iteration], filename, "iteration", start=[write_index])
    ncwrite([handler.total_write_num], filename, "write_number", start=[write_index])
    
    # Write tasks
    for task in handler.tasks
        write_task_data!(handler, filename, task, write_index)
    end
    
    return true
end

"""
Write individual task data to NetCDF file
"""
function write_task_data!(handler::NetCDFFileHandler, filename::String, task::Dict, write_index::Int)
    init_mpi!(handler)  # Ensure MPI info is available
    task_name = task["name"]
    operator = task["operator"]
    
    # Generate data from operator/field when possible; otherwise fallback to zeros
    data = nothing
    if isa(operator, ScalarField)
        ensure_layout!(operator, :g)
        data = Array(operator.data_g)
    elseif isa(operator, VectorField)
        comps = operator.components
        for comp in comps
            ensure_layout!(comp, :g)
        end
        comp_data = [Array(c.data_g) for c in comps]
        data_shape = (length(comp_data), size(comp_data[1])...)
        data_arr = zeros(eltype(comp_data[1]), data_shape...)
        for (i, arr) in enumerate(comp_data)
            data_arr[i, :, :] .= arr
        end
        data = data_arr
    elseif isa(operator, TensorField)
        comps = operator.components
        for comp in vec(comps)
            ensure_layout!(comp, :g)
        end
        comp_data = [Array(c.data_g) for c in vec(comps)]
        data_shape = (size(comps, 1), size(comps, 2), size(comp_data[1])...)
        data_arr = zeros(eltype(comp_data[1]), data_shape...)
        for i in 1:size(comps, 1), j in 1:size(comps, 2)
            data_arr[i, j, :, :] .= comp_data[(i - 1) * size(comps, 2) + j]
        end
        data = data_arr
    else
        data_shape = task["local_shape"]
        if handler.precision == Float32
            data = zeros(Float32, data_shape...)
        else
            data = zeros(Float64, data_shape...)
        end
    end
    # Apply optional postprocess (slices/reductions)
    if task["postprocess"] !== nothing
        data = task["postprocess"](data)
    end
    data = Array(data)  # Ensure plain array for NetCDF
    nc_type = eltype(data) == Float32 ? NC_FLOAT : NC_DOUBLE
    
    # Check if variable exists
    variable_exists = false
    try
        ncread(filename, task_name, start=[1], count=[1])  # Test if exists
        variable_exists = true
    catch
        variable_exists = false
    end
    
    if !variable_exists
        # Variable doesn't exist, create it
        # Dimensions: (time, [components], spatial dims...)
        data_shape = size(data)
        dim_names = ["sim_time"]
        for i in 1:length(data_shape)
            push!(dim_names, "$(task_name)_dim$i")  # Use task-specific dimension names
        end
        
        # Create spatial coordinate variables if they don't exist
        for (i, dim_name) in enumerate(dim_names[2:end])
            coord_exists = false
            try
                ncread(filename, dim_name, start=[1], count=[1])
                coord_exists = true
            catch
                coord_exists = false
            end
            
            if !coord_exists
                coord_length = data_shape[i]
                coord_data = collect(range(0.0, 1.0, length=coord_length))
                nccreate(filename, dim_name, dim_name, coord_length,
                        atts=Dict("long_name" => dim_name,
                                 "axis" => "XYZ"[min(i,3):min(i,3)]))
                ncwrite(coord_data, filename, dim_name)
            end
        end
        
        # Create main data variable using NetCDF.jl syntax
        # First collect dimension names and sizes
        dims_for_var = copy(dim_names)  # ["sim_time", "task_dim1", ...]
        sizes_for_var = [0; collect(data_shape)]  # 0=unlimited for time
        
        # Create main data variable
        # Note: NetCDF.jl doesn't support Bool attributes, so we use Int (0/1)
        var_atts = Dict(
            "long_name" => task_name,
            "standard_name" => task_name,
            "_FillValue" => handler.precision(NaN),
            "grid_space" => 1,  # 1 = true (NetCDF doesn't support Bool)
            "layout" => task["layout"]
        )
        if task["scales"] !== nothing
            var_atts["scales"] = string(task["scales"])
        end
        nccreate(filename, task_name, dims_for_var, sizes_for_var,
                t=nc_type,
                atts=var_atts)
    end
    
    # Write data with time index
    start_indices = [write_index; ones(Int, length(size(data)))]
    ncwrite(data, filename, task_name, start=start_indices)
    
    return true
end

"""
Convenience function to create NetCDF file handler (matching Dedalus API)
Usage: handler = add_file_handler("snapshots", dist, vars, sim_dt=0.25, max_writes=50)
"""
function add_netcdf_handler(base_path::String, dist, vars; kwargs...)
    return NetCDFFileHandler(base_path, dist, vars; kwargs...)
end

# Export main functionality matching Dedalus interface
export NetCDFFileHandler, add_netcdf_handler

"""
Dedalus-style helper to create a NetCDF file handler.
Matches evaluator.add_file_handler(...) usage in Dedalus.
"""
function add_file_handler(base_path::String, dist, vars; kwargs...)
    return NetCDFFileHandler(base_path, dist, vars; kwargs...)
end

"""
Alias without bang for Dedalus-style task addition.
"""
function add_task(handler::NetCDFFileHandler, task; kwargs...)
    return add_task!(handler, task; kwargs...)
end

"""
Add a task that computes the mean over specified dimensions.

# Arguments
- `handler`: NetCDFFileHandler instance
- `field`: Field to compute mean of
- `dims`: Dimensions to average over (e.g., (:x, :y) or (1, 2))
- `name`: Optional name for the output variable

# Example
```julia
add_mean_task!(handler, u, dims=(:x, :y), name="u_mean_z")
```
"""
function add_mean_task!(handler::NetCDFFileHandler, field; dims=nothing, name=nothing)
    if name === nothing
        field_name = get_field_name(field)
        if dims !== nothing
            dims_str = join(string.(dims), "_")
            name = "$(field_name)_mean_$(dims_str)"
        else
            name = "$(field_name)_mean"
        end
    end

    # Create postprocess function for mean computation
    if dims === nothing
        # Global mean
        postprocess = data -> [netcdf_mean(data)]
    else
        # Mean over specified dimensions
        dim_indices = resolve_dimension_indices(field, dims)
        postprocess = data -> dropdims(netcdf_mean(data, dims=dim_indices), dims=dim_indices)
    end

    return add_task!(handler, field; name=name, postprocess=postprocess)
end

"""
Add a task that extracts a slice of a field.

# Arguments
- `handler`: NetCDFFileHandler instance
- `field`: Field to slice
- `slices`: Dictionary or named tuple specifying slice positions
          e.g., Dict(:z => 0.5) or (z=0.5,) for midplane slice
- `name`: Optional name for the output variable

# Example
```julia
add_slice_task!(handler, u, slices=Dict(:z => 0.0), name="u_bottom")
add_slice_task!(handler, T, slices=(x=0.5, y=0.5), name="T_centerline")
```
"""
function add_slice_task!(handler::NetCDFFileHandler, field; slices=Dict(), name=nothing)
    if name === nothing
        field_name = get_field_name(field)
        slice_parts = ["$(k)=$(v)" for (k, v) in pairs(slices)]
        slice_str = join(slice_parts, "_")
        name = isempty(slice_str) ? field_name : "$(field_name)_$(slice_str)"
    end

    # Create postprocess function for slicing
    postprocess = data -> apply_field_slices(data, field, slices)

    return add_task!(handler, field; name=name, postprocess=postprocess)
end

"""
Add a task that computes a 1D profile (mean over all but one dimension).

# Arguments
- `handler`: NetCDFFileHandler instance
- `field`: Field to compute profile of
- `dim`: The dimension to keep (profile along this dimension)
- `name`: Optional name for the output variable

# Example
```julia
add_profile_task!(handler, u, dim=:z, name="u_profile_z")
```
"""
function add_profile_task!(handler::NetCDFFileHandler, field; dim=nothing, name=nothing)
    if dim === nothing
        throw(ArgumentError("Must specify 'dim' for profile task"))
    end

    if name === nothing
        field_name = get_field_name(field)
        name = "$(field_name)_profile_$(dim)"
    end

    # Create postprocess function for profile computation
    keep_dim = resolve_dimension_index(field, dim)

    postprocess = function(data)
        ndims_data = ndims(data)
        # Average over all dimensions except keep_dim
        result = data
        for d in ndims_data:-1:1
            if d != keep_dim
                result = dropdims(netcdf_mean(result, dims=d), dims=d)
            end
        end
        return result
    end

    return add_task!(handler, field; name=name, postprocess=postprocess)
end

"""
Add a task that computes the variance over specified dimensions.
"""
function add_variance_task!(handler::NetCDFFileHandler, field; dims=nothing, name=nothing)
    if name === nothing
        field_name = get_field_name(field)
        if dims !== nothing
            dims_str = join(string.(dims), "_")
            name = "$(field_name)_var_$(dims_str)"
        else
            name = "$(field_name)_var"
        end
    end

    if dims === nothing
        postprocess = data -> [var(data)]
    else
        dim_indices = resolve_dimension_indices(field, dims)
        postprocess = data -> dropdims(var(data, dims=dim_indices), dims=dim_indices)
    end

    return add_task!(handler, field; name=name, postprocess=postprocess)
end

"""
Add a task that computes the RMS (root mean square) over specified dimensions.
"""
function add_rms_task!(handler::NetCDFFileHandler, field; dims=nothing, name=nothing)
    if name === nothing
        field_name = get_field_name(field)
        if dims !== nothing
            dims_str = join(string.(dims), "_")
            name = "$(field_name)_rms_$(dims_str)"
        else
            name = "$(field_name)_rms"
        end
    end

    if dims === nothing
        postprocess = data -> [sqrt(netcdf_mean(data .^ 2))]
    else
        dim_indices = resolve_dimension_indices(field, dims)
        postprocess = data -> sqrt.(dropdims(netcdf_mean(data .^ 2, dims=dim_indices), dims=dim_indices))
    end

    return add_task!(handler, field; name=name, postprocess=postprocess)
end

"""
Add a task that computes min/max over the domain.
"""
function add_extrema_task!(handler::NetCDFFileHandler, field; name=nothing)
    field_name = get_field_name(field)

    # Add min task
    min_name = name !== nothing ? "$(name)_min" : "$(field_name)_min"
    add_task!(handler, field; name=min_name, postprocess=data -> [minimum(data)])

    # Add max task
    max_name = name !== nothing ? "$(name)_max" : "$(field_name)_max"
    add_task!(handler, field; name=max_name, postprocess=data -> [maximum(data)])

    return handler
end

# ============================================================================
# Helper functions
# ============================================================================

"""
Get field name from various field types.
"""
function get_field_name(field)
    if isa(field, ScalarField)
        return field.name
    elseif isa(field, VectorField)
        return field.name
    elseif isa(field, TensorField)
        return field.name
    elseif isa(field, Dict) && haskey(field, "name")
        return field["name"]
    elseif isa(field, AbstractString)
        return field
    else
        return "field"
    end
end

"""
Resolve dimension specification to index.
Handles both symbolic (:x, :y, :z) and integer specifications.
"""
function resolve_dimension_index(field, dim)
    if isa(dim, Integer)
        return dim
    elseif isa(dim, Symbol)
        # Map symbolic dimensions to indices
        dim_map = Dict(:x => 1, :y => 2, :z => 3, :t => 0)
        if haskey(dim_map, dim)
            return dim_map[dim]
        else
            # Try to get from field's domain
            return get_dimension_index_from_field(field, dim)
        end
    else
        throw(ArgumentError("Unsupported dimension specification: $dim"))
    end
end

"""
Resolve multiple dimension specifications to indices.
"""
function resolve_dimension_indices(field, dims)
    if isa(dims, Tuple) || isa(dims, Vector)
        return Tuple(resolve_dimension_index(field, d) for d in dims)
    else
        return (resolve_dimension_index(field, dims),)
    end
end

"""
Get dimension index from field's domain information.
"""
function get_dimension_index_from_field(field, dim_symbol)
    if isa(field, ScalarField) && field.domain !== nothing
        # Try to match against coordinate names
        for (i, basis) in enumerate(field.bases)
            if hasproperty(basis, :coord) && hasproperty(basis.coord, :name)
                if Symbol(basis.coord.name) == dim_symbol
                    return i
                end
            end
        end
    end
    # Default mapping
    dim_map = Dict(:x => 1, :y => 2, :z => 3)
    return get(dim_map, dim_symbol, 1)
end

"""
Apply slices to field data based on slice specification.
"""
function apply_field_slices(data, field, slices)
    if isempty(slices)
        return data
    end

    # Build slice indices
    ndims_data = ndims(data)
    indices = [Colon() for _ in 1:ndims_data]

    for (dim, value) in pairs(slices)
        dim_idx = resolve_dimension_index(field, dim)
        if dim_idx >= 1 && dim_idx <= ndims_data
            # Convert value to index
            n = size(data, dim_idx)
            if isa(value, Integer)
                indices[dim_idx] = value
            elseif isa(value, AbstractFloat)
                # Interpret as fractional position (0.0 to 1.0)
                idx = clamp(round(Int, value * (n - 1)) + 1, 1, n)
                indices[dim_idx] = idx
            end
        end
    end

    # Apply slicing
    result = data[indices...]

    # Ensure result is at least 1D
    if ndims(result) == 0
        return [result]
    end

    return result
end

"""
Close handler and finalize all files.
"""
function close!(handler::NetCDFFileHandler)
    # Update global attributes with final write count
    try
        filename = current_file(handler)
        if isfile(filename)
            ncputatt(filename, "NC_GLOBAL", Dict(
                "writes" => handler.file_write_num,
                "total_writes" => handler.total_write_num,
                "closed" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            ))
        end
    catch e
        @warn "Failed to update final attributes" exception=e
    end

    return nothing
end

"""
Get list of all output files created by this handler.
"""
function get_output_files(handler::NetCDFFileHandler)
    files = String[]

    for set_num in 1:handler.set_num
        set_name = "$(handler.name)_s$(set_num)"
        set_path = joinpath(dirname(handler.base_path), set_name)

        if isdir(set_path)
            for f in readdir(set_path, join=true)
                if endswith(f, ".nc")
                    push!(files, f)
                end
            end
        end
    end

    return files
end

"""
Get metadata about the handler's output.
"""
function get_handler_info(handler::NetCDFFileHandler)
    return Dict(
        "name" => handler.name,
        "base_path" => handler.base_path,
        "set_num" => handler.set_num,
        "total_writes" => handler.total_write_num,
        "file_writes" => handler.file_write_num,
        "max_writes" => handler.max_writes,
        "precision" => string(handler.precision),
        "parallel" => handler.parallel,
        "num_tasks" => length(handler.tasks),
        "task_names" => [t["name"] for t in handler.tasks],
        "mpi_size" => handler.size,
        "mpi_rank" => handler.rank
    )
end

"""
Reset handler for a new simulation run.
"""
function reset!(handler::NetCDFFileHandler; keep_tasks::Bool=true)
    handler.set_num = 1
    handler.total_write_num = 0
    handler.file_write_num = 0

    if !keep_tasks
        empty!(handler.tasks)
    end

    return handler
end

# ============================================================================
# Statistics helpers (using Base.mean would require Statistics)
# ============================================================================

# Simple mean implementation to avoid dependency
function netcdf_mean(x; dims=nothing)
    if dims === nothing
        return sum(x) / length(x)
    else
        return sum(x, dims=dims) ./ size(x, dims...)
    end
end

# Simple variance implementation
function netcdf_var(x; dims=nothing)
    if dims === nothing
        m = netcdf_mean(x)
        return sum((x .- m) .^ 2) / (length(x) - 1)
    else
        m = netcdf_mean(x, dims=dims)
        n = prod(size(x, d) for d in dims)
        return sum((x .- m) .^ 2, dims=dims) ./ (n - 1)
    end
end

# ============================================================================
# Exports
# ============================================================================

export add_file_handler, add_task
export add_task!, check_schedule, process!, close!
export current_path, current_file, create_current_file!
export add_mean_task!, add_slice_task!, add_profile_task!
export add_variance_task!, add_rms_task!, add_extrema_task!
export get_output_files, get_handler_info, reset!
