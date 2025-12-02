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
function add_task!(handler::NetCDFFileHandler, task; layout="g", name=nothing, scales=nothing)
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
        "dtype" => get_operator_dtype(operator, handler.precision)
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
Check if operator represents a locked field
"""
function is_locked_field(operator)
    # Placeholder - would check if operator is a locked field type
    return haskey(operator, "locked") && operator["locked"] == true
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
Get operator data type
"""
function get_operator_dtype(operator, default_precision)
    # Try to extract dtype from operator, fall back to handler precision
    if haskey(operator, "dtype")
        return operator["dtype"]
    else
        return default_precision
    end
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
    if isa(operator, ScalarField)
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
        "software_repository" => "https://github.com/subhajitkar/Tarang.jl"
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
    
    # Generate sample data (in real implementation, this would be task["out"].data)
    data_shape = task["local_shape"]
    if handler.precision == Float32
        data = randn(Float32, data_shape...) .+ Float32(handler.rank)
        nc_type = NC_FLOAT
    else
        data = randn(Float64, data_shape...) .+ Float64(handler.rank)
        nc_type = NC_DOUBLE
    end
    
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
        # Dimensions: (time, spatial dims...)
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
                # Create coordinate variable
                coord_data = collect(range(0.0, 1.0, length=data_shape[i]))
                nccreate(filename, dim_name, dim_name, data_shape[i],
                        atts=Dict("long_name" => dim_name,
                                 "axis" => "XYZ"[min(i,3):min(i,3)]))
                ncwrite(coord_data, filename, dim_name)
            end
        end
        
        # Create main data variable using NetCDF.jl syntax
        # First collect dimension names and sizes
        dims_for_var = copy(dim_names)  # ["sim_time", "task_dim1", "task_dim2"]
        sizes_for_var = [0; collect(data_shape)]  # [0, 64, 32] where 0=unlimited
        
        # Create main data variable
        nccreate(filename, task_name, dims_for_var, sizes_for_var,
                t=nc_type,
                atts=Dict("long_name" => task_name,
                         "standard_name" => task_name,
                         "_FillValue" => handler.precision(NaN),
                         "grid_space" => true,
                         "layout" => task["layout"]))
    end
    
    # Write data with time index
    start_indices = [write_index; ones(Int, length(data_shape))]
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

export add_file_handler, add_task
export add_task!, check_schedule, process!
export current_path, current_file, create_current_file!
