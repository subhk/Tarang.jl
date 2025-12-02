"""
Evaluator for analysis and output

Translated from dedalus/core/evaluator.py
"""

using HDF5
using MPI

# GPU support
include("gpu_manager.jl")
using .GPUManager

# Include NetCDF support
include("../tools/netcdf_output.jl")

mutable struct FileHandler
    filename::String
    datasets::Dict{String, Any}
    cadence::Union{Int, Nothing}
    sim_dt::Union{Float64, Nothing}
    wall_dt::Union{Float64, Nothing}
    max_writes::Union{Int, Nothing}
    
    # State tracking
    write_count::Int
    last_write_time::Float64
    last_write_sim_time::Float64
    
    function FileHandler(filename::String; cadence::Union{Int, Nothing}=nothing, 
                        sim_dt::Union{Float64, Nothing}=nothing,
                        wall_dt::Union{Float64, Nothing}=nothing,
                        max_writes::Union{Int, Nothing}=nothing)
        new(filename, Dict{String, Any}(), cadence, sim_dt, wall_dt, max_writes, 0, 0.0, 0.0)
    end
end

mutable struct Evaluator
    solver::InitialValueSolver
    handlers::Vector{FileHandler}
    device_config::DeviceConfig
    gpu_workspace::Dict{String, AbstractArray}
    performance_stats::EvaluatorPerformanceStats
    
    function Evaluator(solver::InitialValueSolver)
        # Use same device configuration as solver
        device_config = hasfield(typeof(solver), :device_config) ? solver.device_config : select_device("cpu")
        gpu_workspace = Dict{String, AbstractArray}()
        perf_stats = EvaluatorPerformanceStats()
        new(solver, FileHandler[], device_config, gpu_workspace, perf_stats)
    end
end

# File handler management
function add_file_handler(evaluator::Evaluator, filename::String; kwargs...)
    """Add file handler for output"""
    handler = FileHandler(filename; kwargs...)
    push!(evaluator.handlers, handler)
    return handler
end

function add_task!(handler::FileHandler, field::Union{ScalarField, VectorField, TensorField, Operator}; name::String="")
    """Add field or operator to file handler"""
    if isempty(name)
        if hasfield(typeof(field), :name)
            name = field.name
        else
            name = "field_$(length(handler.datasets)+1)"
        end
    end
    
    handler.datasets[name] = field
end

function add_task!(handler::FileHandler, field, name::String)
    """Add field or operator to file handler with explicit name"""
    handler.datasets[name] = field
end

# Evaluation and output
function evaluate_handlers!(evaluator::Evaluator, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Evaluate all handlers and write output if conditions are met"""
    
    for handler in evaluator.handlers
        if should_write(handler, wall_time, sim_time, iteration)
            write_handler!(handler, evaluator.solver, wall_time, sim_time, iteration)
        end
    end
end

function should_write(handler::FileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Check if handler should write output"""
    
    # Check max writes
    if handler.max_writes !== nothing && handler.write_count >= handler.max_writes
        return false
    end
    
    # Check cadence
    if handler.cadence !== nothing
        if iteration % handler.cadence != 0
            return false
        end
    end
    
    # Check simulation time interval
    if handler.sim_dt !== nothing
        if sim_time - handler.last_write_sim_time < handler.sim_dt
            return false
        end
    end
    
    # Check wall time interval
    if handler.wall_dt !== nothing
        if wall_time - handler.last_write_time < handler.wall_dt
            return false
        end
    end
    
    return true
end

function write_handler!(handler::FileHandler, solver::InitialValueSolver, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Write handler output to file"""
    
    # Create filename with iteration number
    base_name = splitext(handler.filename)[1]
    extension = splitext(handler.filename)[2]
    if isempty(extension)
        extension = ".h5"
    end
    
    filename = "$(base_name)_$(iteration)$(extension)"
    
    # Only rank 0 creates the file initially
    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
    else
        rank = 0
    end
    
    # Evaluate all tasks and gather data
    task_data = Dict{String, Any}()
    
    for (name, task) in handler.datasets
        data = evaluate_task(task, solver)
        task_data[name] = data
    end
    
    # Write to HDF5 file
    write_hdf5_output(filename, task_data, sim_time, iteration, rank)
    
    # Update handler state
    handler.write_count += 1
    handler.last_write_time = wall_time
    handler.last_write_sim_time = sim_time
    
    if rank == 0
        @info "Written output to $filename at t=$(sim_time), iteration=$iteration"
    end
end

function evaluate_task(task, solver::InitialValueSolver)
    """Evaluate a task (field or operator) and return data with GPU support"""
    
    # Get device configuration from solver or evaluator
    device_config = hasfield(typeof(solver), :device_config) ? solver.device_config : select_device("cpu")
    
    gpu_transfer_start = time()
    
    if isa(task, ScalarField)
        # Return field data in grid layout (GPU-aware)
        ensure_layout!(task, :g)
        # Ensure data is moved to GPU if needed for evaluation, then to CPU for output
        task.data_g = ensure_device!(task.data_g, device_config)
        result = Array(task.data_g)  # Always return CPU array for file I/O
        
    elseif isa(task, VectorField)
        # Return vector components (GPU-aware)
        data = []
        for component in task.components
            ensure_layout!(component, :g)
            component.data_g = ensure_device!(component.data_g, device_config)
            push!(data, Array(component.data_g))  # Always return CPU array for file I/O
        end
        result = data
        
    elseif isa(task, TensorField)
        # Return tensor components (GPU-aware)
        data = []
        for i in 1:size(task.components, 1), j in 1:size(task.components, 2)
            component = task.components[i, j]
            ensure_layout!(component, :g)
            component.data_g = ensure_device!(component.data_g, device_config)
            push!(data, Array(component.data_g))  # Always return CPU array for file I/O
        end
        result = data
        
    elseif isa(task, Operator)
        # Evaluate operator (GPU-aware)
        operator_result = evaluate_operator_gpu(task, device_config)
        result = evaluate_task(operator_result, solver)
        
    else
        throw(ArgumentError("Unknown task type: $(typeof(task))"))
    end
    
    # Track GPU transfer time if evaluator has performance stats
    if hasfield(typeof(solver), :evaluator) && solver.evaluator !== nothing && hasfield(typeof(solver.evaluator), :performance_stats)
        solver.evaluator.performance_stats.gpu_transfer_time += time() - gpu_transfer_start
    end
    
    return result
end

function evaluate_operator(op::Operator)
    """Evaluate operator and return result field"""
    
    if isa(op, Gradient)
        return evaluate_gradient(op)
    elseif isa(op, Divergence)
        return evaluate_divergence(op)
    elseif isa(op, Curl)
        return evaluate_curl(op)
    elseif isa(op, Laplacian)
        return evaluate_laplacian(op)
    elseif isa(op, Differentiate)
        return evaluate_differentiate(op)
    else
        throw(ArgumentError("Operator evaluation not implemented for $(typeof(op))"))
    end
end

function evaluate_operator_gpu(op::Operator, device_config::DeviceConfig)
    """Evaluate operator with GPU acceleration"""
    
    # Ensure operand data is on correct device before evaluation
    if hasfield(typeof(op), :operand)
        operand = op.operand
        if isa(operand, ScalarField)
            operand.data_g = ensure_device!(operand.data_g, device_config)
            operand.data_c = ensure_device!(operand.data_c, device_config)
        elseif isa(operand, VectorField)
            for comp in operand.components
                comp.data_g = ensure_device!(comp.data_g, device_config)
                comp.data_c = ensure_device!(comp.data_c, device_config)
            end
        end
    end
    
    # Evaluate operator using existing functions (they will work on GPU data)
    if isa(op, Gradient)
        return evaluate_gradient_gpu(op, device_config)
    elseif isa(op, Divergence)
        return evaluate_divergence_gpu(op, device_config)
    elseif isa(op, Curl)
        return evaluate_curl_gpu(op, device_config)
    elseif isa(op, Laplacian)
        return evaluate_laplacian_gpu(op, device_config)
    elseif isa(op, Differentiate)
        return evaluate_differentiate_gpu(op, device_config)
    else
        # Fallback to CPU evaluation
        @warn "GPU evaluation not available for $(typeof(op)), using CPU fallback"
        return evaluate_operator(op)
    end
end

function evaluate_curl(curl_op::Curl, layout::Symbol=:g)
    """Evaluate curl operator"""
    operand = curl_op.operand
    coordsys = curl_op.coordsys
    
    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires vector field"))
    end
    
    if coordsys.dim == 2
        # 2D curl: ∇ × u = ∂u_y/∂x - ∂u_x/∂y (scalar result)
        u_x = operand.components[1]
        u_y = operand.components[2]
        
        # Get coordinates
        x_coord = coordsys[1]
        y_coord = coordsys[2]
        
        # Compute partial derivatives
        du_y_dx = evaluate_differentiate(Differentiate(u_y, x_coord, 1), layout)
        du_x_dy = evaluate_differentiate(Differentiate(u_x, y_coord, 1), layout)
        
        # Compute curl
        result = ScalarField(operand.dist, "curl_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(result, layout)
        ensure_layout!(du_y_dx, layout)
        ensure_layout!(du_x_dy, layout)
        
        result.data_g .= du_y_dx.data_g .- du_x_dy.data_g
        return result
        
    elseif coordsys.dim == 3
        # 3D curl: ∇ × u = (∂u_z/∂y - ∂u_y/∂z, ∂u_x/∂z - ∂u_z/∂x, ∂u_y/∂x - ∂u_x/∂y)
        u_x = operand.components[1]
        u_y = operand.components[2] 
        u_z = operand.components[3]
        
        # Get coordinates
        x_coord = coordsys[1]
        y_coord = coordsys[2]
        z_coord = coordsys[3]
        
        # Compute partial derivatives for curl_x = ∂u_z/∂y - ∂u_y/∂z
        du_z_dy = evaluate_differentiate(Differentiate(u_z, y_coord, 1), layout)
        du_y_dz = evaluate_differentiate(Differentiate(u_y, z_coord, 1), layout)
        
        curl_x = ScalarField(operand.dist, "curl_x_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(curl_x, layout)
        ensure_layout!(du_z_dy, layout)
        ensure_layout!(du_y_dz, layout)
        curl_x.data_g .= du_z_dy.data_g .- du_y_dz.data_g
        
        # Compute partial derivatives for curl_y = ∂u_x/∂z - ∂u_z/∂x
        du_x_dz = evaluate_differentiate(Differentiate(u_x, z_coord, 1), layout)
        du_z_dx = evaluate_differentiate(Differentiate(u_z, x_coord, 1), layout)
        
        curl_y = ScalarField(operand.dist, "curl_y_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(curl_y, layout)
        ensure_layout!(du_x_dz, layout)
        ensure_layout!(du_z_dx, layout)
        curl_y.data_g .= du_x_dz.data_g .- du_z_dx.data_g
        
        # Compute partial derivatives for curl_z = ∂u_y/∂x - ∂u_x/∂y
        du_y_dx = evaluate_differentiate(Differentiate(u_y, x_coord, 1), layout)
        du_x_dy = evaluate_differentiate(Differentiate(u_x, y_coord, 1), layout)
        
        curl_z = ScalarField(operand.dist, "curl_z_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(curl_z, layout)
        ensure_layout!(du_y_dx, layout)
        ensure_layout!(du_x_dy, layout)
        curl_z.data_g .= du_y_dx.data_g .- du_x_dy.data_g
        
        # Return vector field
        result = VectorField(operand.dist, coordsys, "curl_$(operand.name)", operand.bases, operand.dtype)
        result.components[1] = curl_x
        result.components[2] = curl_y
        result.components[3] = curl_z
        
        return result
        
    else
        throw(ArgumentError("Curl not defined for $(coordsys.dim)D"))
    end
end

function evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)
    """Evaluate Laplacian operator"""
    operand = lap_op.operand
    
    if isa(operand, ScalarField)
        # Scalar Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...
        result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(result, layout)
        fill!(result[string(layout)], 0)
        
        # Sum second derivatives in all directions
        for (i, basis) in enumerate(operand.bases)
            coord_name = basis.meta.element_label
            coord = operand.domain.dist.coordsys[coord_name]
            
            # Second derivative
            d2u = evaluate_differentiate(Differentiate(operand, coord, 2), layout)
            result = result + d2u
        end
        
        return result
    else
        throw(ArgumentError("Laplacian not implemented for operand type $(typeof(operand))"))
    end
end

function write_hdf5_output(filename::String, data::Dict{String, Any}, sim_time::Float64, iteration::Int, rank::Int)
    """Write data to HDF5 file"""
    
    # For MPI, gather all data to rank 0
    if MPI.Initialized() && MPI.Comm_size(MPI.COMM_WORLD) > 1
        comm = MPI.COMM_WORLD
        
        # Gather data from all processes
        global_data = Dict{String, Any}()
        for (name, local_data) in data
            gathered = MPI.Allgather(local_data, comm)
            
            # Concatenate along first dimension (assuming pencil distribution)
            if isa(gathered[1], AbstractArray) && ndims(gathered[1]) > 0
                global_data[name] = cat(gathered..., dims=1)
            else
                global_data[name] = gathered
            end
        end
        data = global_data
    end
    
    # Only rank 0 writes to file
    if rank == 0
        h5open(filename, "w") do file
            # Write metadata
            attrs(file)["sim_time"] = sim_time
            attrs(file)["iteration"] = iteration
            attrs(file)["timestamp"] = string(now())
            
            # Write datasets
            for (name, dataset) in data
                if isa(dataset, AbstractArray)
                    write(file, name, dataset)
                else
                    # Handle non-array data
                    write(file, name, [dataset])
                end
            end
        end
    end
end

# Analysis utilities
mutable struct GlobalFlowProperty
    solver::InitialValueSolver
    cadence::Int
    properties::Dict{String, Any}  # Stores evaluated field data
    reducer::GlobalArrayReducer
    evaluator_handler::Union{Nothing, FileHandler}  # Dictionary-like handler for evaluation
    device_config::DeviceConfig
    gpu_workspace::Dict{String, AbstractArray}
    performance_stats::EvaluatorPerformanceStats
    
    function GlobalFlowProperty(solver::InitialValueSolver; cadence::Int=1)
        reducer = GlobalArrayReducer(solver.dist.comm)
        # Use same device configuration as solver
        device_config = hasfield(typeof(solver), :device_config) ? solver.device_config : select_device("cpu")
        gpu_workspace = Dict{String, AbstractArray}()
        perf_stats = EvaluatorPerformanceStats()
        new(solver, cadence, Dict{String, Any}(), reducer, nothing, device_config, gpu_workspace, perf_stats)
    end
end

struct GlobalArrayReducer
    comm::MPI.Comm
    scalar_buffer::Vector{Float64}
    
    function GlobalArrayReducer(comm::MPI.Comm)
        new(comm, zeros(Float64, 1))
    end
end

function reduce_scalar(reducer::GlobalArrayReducer, local_scalar::Float64, mpi_op)
    """Compute global reduction of a scalar from each process."""
    reducer.scalar_buffer[1] = local_scalar
    if MPI.Initialized()
        MPI.Allreduce!(reducer.scalar_buffer, mpi_op, reducer.comm)
    end
    return reducer.scalar_buffer[1]
end

function global_min(reducer::GlobalArrayReducer, data::AbstractArray; empty::Float64=Inf)
    """Compute global min of all array data."""
    local_min = isempty(data) ? empty : minimum(data)
    return reduce_scalar(reducer, local_min, MPI.MIN)
end

function global_max(reducer::GlobalArrayReducer, data::AbstractArray; empty::Float64=-Inf)
    """Compute global max of all array data."""
    local_max = isempty(data) ? empty : maximum(data)
    return reduce_scalar(reducer, local_max, MPI.MAX)
end

function global_mean(reducer::GlobalArrayReducer, data::AbstractArray)
    """Compute global mean of all array data."""
    local_sum = sum(data)
    local_size = Float64(length(data))
    global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)
    global_size = reduce_scalar(reducer, local_size, MPI.SUM)
    return global_size > 0 ? global_sum / global_size : 0.0
end

function add_property!(flow::GlobalFlowProperty, field::Union{ScalarField, VectorField}, name::String)
    """
    Add property to track.
    Following Dedalus pattern: properties.add_task(property, layout='g', name=name)
    """
    # Store field reference for evaluation
    # In Dedalus, this gets added to the dictionary handler as a task
    flow.properties[name] = field
end

function evaluate_property(flow::GlobalFlowProperty, name::String)
    """
    Get grid data for property evaluation with GPU support.
    Following Dedalus pattern: gdata = self.properties[name]['g']
    Returns the grid data array for the named property.
    """
    if !haskey(flow.properties, name)
        throw(KeyError("Property '$name' not found"))
    end
    
    field = flow.properties[name]
    
    gpu_transfer_start = time()
    
    # Extract grid data similar to Dedalus self.properties[name]['g']
    if isa(field, ScalarField)
        # Ensure field is on correct device for evaluation
        field.data_g = ensure_device!(field.data_g, flow.device_config)
        result = Array(field["g"])  # Always return CPU array for reductions
    elseif isa(field, VectorField)
        # For vector fields, could return magnitude or individual components
        # For now, return first component
        field.components[1].data_g = ensure_device!(field.components[1].data_g, flow.device_config)
        result = Array(field.components[1]["g"])
    elseif isa(field, AbstractArray)
        # Ensure array is on correct device
        field_device = ensure_device!(field, flow.device_config)
        result = Array(field_device)
    else
        error("Unsupported property type: $(typeof(field))")
    end
    
    # Track GPU transfer time
    flow.performance_stats.gpu_transfer_time += time() - gpu_transfer_start
    
    return result
end

function max(flow::GlobalFlowProperty, name::String)
    """
    Compute global max of a property on the grid.
    Following Dedalus implementation in flow_tools.py:107-110
    """
    gdata = evaluate_property(flow, name)
    return global_max(flow.reducer, gdata)
end

function min(flow::GlobalFlowProperty, name::String)
    """
    Compute global min of a property on the grid.
    Following Dedalus implementation in flow_tools.py:102-105
    """
    gdata = evaluate_property(flow, name)
    return global_min(flow.reducer, gdata)
end

function grid_average(flow::GlobalFlowProperty, name::String)
    """
    Compute global mean of a property on the grid.
    Following Dedalus implementation in flow_tools.py:112-115
    """
    gdata = evaluate_property(flow, name)
    return global_mean(flow.reducer, gdata)
end

function volume_integral(flow::GlobalFlowProperty, name::String)
    """
    Compute volume integral of a property.
    Following Dedalus implementation in flow_tools.py:117-130
    """
    # Check for precomputed integral
    integral_name = "_$(name)_integral"
    if haskey(flow.properties, integral_name)
        integral_field = flow.properties[integral_name]
        integral_data = evaluate_property(flow, integral_name)
    else
        # Compute volume integral
        field = flow.properties[name]
        # This would need proper integration operator implementation
        # For now, return sum (approximation of integral)
        gdata = evaluate_property(flow, name)
        integral_data = [sum(gdata)]  # Simple sum approximation
    end
    
    # Communicate integral value to all processes  
    integral_value = global_max(flow.reducer, integral_data)
    return integral_value
end

function volume_average(flow::GlobalFlowProperty, name::String)
    """
    Compute volume average of a property.
    Following Dedalus implementation in flow_tools.py:132-137
    """
    # TODO: missing hypervolume definition (same as Dedalus comment)
    # For now, use grid average as approximation
    return grid_average(flow, name)
    
    # When hypervolume is implemented:
    # average_value = volume_integral(flow, name) / solver.domain.hypervolume
    # return average_value
end

# Enhanced evaluator functions with NetCDF support
function create_evaluator(solver::InitialValueSolver)
    """Create evaluator for solver"""
    return Evaluator(solver)
end

function create_netcdf_evaluator(solver::InitialValueSolver)
    """Create NetCDF evaluator for solver"""
    return NetCDFEvaluator(solver)
end

# Unified evaluator interface that supports both HDF5 and NetCDF
mutable struct UnifiedEvaluator
    solver::InitialValueSolver
    hdf5_handlers::Vector{FileHandler}
    netcdf_handlers::Vector{NetCDFFileHandler}
    device_config::DeviceConfig
    gpu_workspace::Dict{String, AbstractArray}
    performance_stats::EvaluatorPerformanceStats
    
    function UnifiedEvaluator(solver::InitialValueSolver)
        # Use same device configuration as solver
        device_config = hasfield(typeof(solver), :device_config) ? solver.device_config : select_device("cpu")
        gpu_workspace = Dict{String, AbstractArray}()
        perf_stats = EvaluatorPerformanceStats()
        new(solver, FileHandler[], NetCDFFileHandler[], device_config, gpu_workspace, perf_stats)
    end
end

function create_unified_evaluator(solver::InitialValueSolver)
    """Create unified evaluator supporting both HDF5 and NetCDF"""
    return UnifiedEvaluator(solver)
end

function add_file_handler(evaluator::UnifiedEvaluator, filename::String, format::Symbol=:auto; kwargs...)
    """Add file handler with automatic format detection or explicit format specification
    
    Supported formats:
    - :hdf5 or :h5 - HDF5 format
    - :netcdf or :nc - NetCDF format  
    - :auto - Auto-detect from file extension
    """
    
    # Determine format
    if format == :auto
        _, ext = splitext(lowercase(filename))
        if ext in [".nc", ".netcdf"]
            format = :netcdf
        elseif ext in [".h5", ".hdf5"]
            format = :hdf5
        else
            # Default to NetCDF for new files
            format = :netcdf
            if !endswith(lowercase(filename), ".nc")
                filename = filename * ".nc"
            end
        end
    end
    
    if format in [:netcdf, :nc]
        handler = NetCDFFileHandler(filename; kwargs...)
        push!(evaluator.netcdf_handlers, handler)
        @info "Added NetCDF file handler: $filename"
        return handler
    elseif format in [:hdf5, :h5]
        handler = FileHandler(filename; kwargs...)
        push!(evaluator.hdf5_handlers, handler)
        @info "Added HDF5 file handler: $filename"
        return handler
    else
        throw(ArgumentError("Unsupported format: $format. Use :hdf5, :netcdf, or :auto"))
    end
end

function evaluate_unified_handlers!(evaluator::UnifiedEvaluator, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Evaluate both HDF5 and NetCDF handlers"""
    
    # Evaluate HDF5 handlers (existing functionality)
    for handler in evaluator.hdf5_handlers
        if should_write_hdf5(handler, wall_time, sim_time, iteration)
            write_hdf5_data!(handler, sim_time, iteration)
            handler.last_write_time = wall_time
        end
    end
    
    # Evaluate NetCDF handlers
    for handler in evaluator.netcdf_handlers
        if should_write(handler, wall_time, sim_time, iteration)
            write_netcdf_data!(handler, sim_time, iteration)
            handler.last_write_time = wall_time
        end
    end
    
    return evaluator
end

# Convenience functions for NetCDF-specific features
function add_netcdf_handler(evaluator::UnifiedEvaluator, filename::String; 
                           precision::NetCDFPrecision=Float64,
                           per_processor_files::Bool=true,
                           compression_level::Int=4,
                           kwargs...)
    """Add NetCDF handler with full control over NetCDF-specific options"""
    
    handler = NetCDFFileHandler(filename; 
                               precision=precision,
                               per_processor_files=per_processor_files,
                               compression_level=compression_level,
                               kwargs...)
    push!(evaluator.netcdf_handlers, handler)
    
    @info "Added NetCDF handler: $filename (precision: $precision, per-processor: $per_processor_files)"
    return handler
end

function set_coordinates!(evaluator::UnifiedEvaluator, coords::Dict{String, Any})
    """Set coordinate information for all NetCDF handlers"""
    
    for handler in evaluator.netcdf_handlers
        set_coordinates!(handler, coords)
    end
    
    return evaluator
end

function finalize_evaluator!(evaluator::UnifiedEvaluator)
    """Finalize and close all file handlers"""
    
    # Close NetCDF handlers
    for handler in evaluator.netcdf_handlers
        close_netcdf_handler!(handler)
    end
    
    # Close HDF5 handlers (if any cleanup needed)
    for handler in evaluator.hdf5_handlers
        # HDF5 cleanup if needed
    end
    
    return evaluator
end

# Helper functions for HDF5 compatibility
function should_write_hdf5(handler::FileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Determine if HDF5 handler should write (similar to NetCDF logic)"""
    
    if handler.max_writes !== nothing && handler.write_count >= handler.max_writes
        return false
    end
    
    if handler.cadence !== nothing
        return (iteration % handler.cadence == 0)
    end
    
    if handler.sim_dt !== nothing
        time_since_write = sim_time - handler.last_write_sim_time
        return time_since_write >= handler.sim_dt
    end
    
    if handler.wall_dt !== nothing
        wall_since_write = wall_time - handler.last_write_time
        return wall_since_write >= handler.wall_dt
    end
    
    return true
end

function write_hdf5_data!(handler::FileHandler, sim_time::Float64, iteration::Int)
    """
    Write HDF5 data following Dedalus structure.
    Based on dedalus/core/evaluator.py:580-595
    """
    
    @debug "Writing HDF5 data: $(handler.filename), t=$sim_time, iter=$iteration"
    
    # Increment write counter
    handler.write_count += 1
    handler.last_write_sim_time = sim_time
    
    # Open/create HDF5 file
    h5open(handler.filename, "cw") do file
        # Write file metadata (following Dedalus write_file_metadata)
        write_file_metadata!(file, handler, iteration, sim_time)
        
        # Write task data (following Dedalus write_task pattern)
        for (task_name, task_data) in handler.datasets
            write_task_data!(file, task_name, task_data, handler.write_count)
        end
    end
    
    @debug "HDF5 write completed: $(handler.filename), write #$(handler.write_count)"
    
    return handler
end

function write_file_metadata!(file::HDF5.File, handler::FileHandler, iteration::Int, sim_time::Float64)
    """
    Write file metadata and time scales.
    Following Dedalus write_file_metadata pattern
    """
    
    # Create/update metadata group
    if !haskey(file, "metadata")
        metadata = create_group(file, "metadata")
    else
        metadata = file["metadata"]
    end
    
    # Write time information (following Dedalus pattern)
    if !haskey(file, "scales")
        scales_group = create_group(file, "scales")
        
        # Initialize time scales
        sim_time_dset = create_dataset(scales_group, "sim_time", Float64, (1,), maxdims=(nothing,))
        iteration_dset = create_dataset(scales_group, "iteration", Int, (1,), maxdims=(nothing,))
        write_number_dset = create_dataset(scales_group, "write_number", Int, (1,), maxdims=(nothing,))
        
    else
        scales_group = file["scales"]
        sim_time_dset = scales_group["sim_time"]
        iteration_dset = scales_group["iteration"]
        write_number_dset = scales_group["write_number"]
        
        # Resize datasets for new write
        HDF5.set_extent_dims(sim_time_dset, (handler.write_count,))
        HDF5.set_extent_dims(iteration_dset, (handler.write_count,))
        HDF5.set_extent_dims(write_number_dset, (handler.write_count,))
    end
    
    # Write current time step data
    sim_time_dset[handler.write_count] = sim_time
    iteration_dset[handler.write_count] = iteration
    write_number_dset[handler.write_count] = handler.write_count
    
    # Write additional metadata
    attrs(metadata)["write_count"] = handler.write_count
    attrs(metadata)["last_sim_time"] = sim_time
    attrs(metadata)["last_iteration"] = iteration
end

function write_task_data!(file::HDF5.File, task_name::String, task_data::Any, write_number::Int)
    """
    Write task data to HDF5 file.
    Following Dedalus write_task pattern from evaluator.py:641-650 and 693-702
    """
    
    # Create tasks group if it doesn't exist
    if !haskey(file, "tasks")
        tasks_group = create_group(file, "tasks")
    else
        tasks_group = file["tasks"]
    end
    
    # Determine data shape and type
    data_array = get_task_data_array(task_data)
    data_shape = size(data_array)
    data_type = eltype(data_array)
    
    # Create or access dataset
    if !haskey(tasks_group, task_name)
        # Create new dataset with time dimension
        # Shape: (time, spatial_dimensions...)
        full_shape = (1, data_shape...)
        max_shape = (nothing, data_shape...)  # Unlimited time dimension
        
        dataset = create_dataset(tasks_group, task_name, data_type, full_shape, maxdims=max_shape)
        
        # Write first data point
        dataset[1, :] = data_array
    else
        # Resize existing dataset and write new data
        dataset = tasks_group[task_name]
        current_shape = size(dataset)
        new_shape = (write_number, current_shape[2:end]...)
        
        HDF5.set_extent_dims(dataset, new_shape)
        dataset[write_number, :] = data_array
    end
end

function get_task_data_array(task_data::Any)
    """
    Extract array data from task data.
    Handles different field types (ScalarField, VectorField, etc.)
    """
    
    if isa(task_data, ScalarField)
        # Get grid-space data
        return task_data["g"]
    elseif isa(task_data, VectorField)
        # Stack component data
        components_data = [comp["g"] for comp in task_data.components]
        return cat(components_data..., dims=1)  # Stack along first dimension
    elseif isa(task_data, AbstractArray)
        # Direct array data
        return task_data
    else
        error("Unsupported task data type: $(typeof(task_data))")
    end
end

# Utility function to add tasks to file handler
function add_task!(handler::FileHandler, name::String, field::Union{ScalarField, VectorField, AbstractArray})
    """Add a field/array to be written by this file handler."""
    handler.datasets[name] = field
end

# GPU-specific operator evaluation functions
function evaluate_gradient_gpu(grad_op::Gradient, device_config::DeviceConfig)
    """Evaluate gradient operator with GPU acceleration"""
    
    # Ensure operand data is on correct device
    operand = grad_op.operand
    operand.data_g = ensure_device!(operand.data_g, device_config)
    operand.data_c = ensure_device!(operand.data_c, device_config)
    
    # Use existing gradient evaluation (works with GPU arrays)
    result = evaluate_gradient(grad_op)
    
    # Ensure result components are on correct device
    if isa(result, VectorField)
        for comp in result.components
            comp.data_g = ensure_device!(comp.data_g, device_config)
            comp.data_c = ensure_device!(comp.data_c, device_config)
        end
    end
    
    # Synchronize GPU operations
    gpu_synchronize(device_config)
    
    return result
end

function evaluate_divergence_gpu(div_op::Divergence, device_config::DeviceConfig)
    """Evaluate divergence operator with GPU acceleration"""
    
    # Ensure operand components are on correct device
    operand = div_op.operand
    if isa(operand, VectorField)
        for comp in operand.components
            comp.data_g = ensure_device!(comp.data_g, device_config)
            comp.data_c = ensure_device!(comp.data_c, device_config)
        end
    end
    
    # Use existing divergence evaluation (works with GPU arrays)
    result = evaluate_divergence(div_op)
    
    # Ensure result is on correct device
    result.data_g = ensure_device!(result.data_g, device_config)
    result.data_c = ensure_device!(result.data_c, device_config)
    
    # Synchronize GPU operations
    gpu_synchronize(device_config)
    
    return result
end

function evaluate_curl_gpu(curl_op::Curl, device_config::DeviceConfig)
    """Evaluate curl operator with GPU acceleration"""
    
    # Ensure operand components are on correct device
    operand = curl_op.operand
    if isa(operand, VectorField)
        for comp in operand.components
            comp.data_g = ensure_device!(comp.data_g, device_config)
            comp.data_c = ensure_device!(comp.data_c, device_config)
        end
    end
    
    # Use existing curl evaluation (works with GPU arrays) 
    result = evaluate_curl(curl_op)
    
    # Ensure result is on correct device
    if isa(result, ScalarField)
        result.data_g = ensure_device!(result.data_g, device_config)
        result.data_c = ensure_device!(result.data_c, device_config)
    elseif isa(result, VectorField)
        for comp in result.components
            comp.data_g = ensure_device!(comp.data_g, device_config)
            comp.data_c = ensure_device!(comp.data_c, device_config)
        end
    end
    
    # Synchronize GPU operations
    gpu_synchronize(device_config)
    
    return result
end

function evaluate_laplacian_gpu(lap_op::Laplacian, device_config::DeviceConfig)
    """Evaluate Laplacian operator with GPU acceleration"""
    
    # Ensure operand data is on correct device
    operand = lap_op.operand
    operand.data_g = ensure_device!(operand.data_g, device_config)
    operand.data_c = ensure_device!(operand.data_c, device_config)
    
    # Use existing Laplacian evaluation (works with GPU arrays)
    result = evaluate_laplacian(lap_op)
    
    # Ensure result is on correct device
    result.data_g = ensure_device!(result.data_g, device_config)
    result.data_c = ensure_device!(result.data_c, device_config)
    
    # Synchronize GPU operations
    gpu_synchronize(device_config)
    
    return result
end

function evaluate_differentiate_gpu(diff_op::Differentiate, device_config::DeviceConfig)
    """Evaluate differentiation operator with GPU acceleration"""
    
    # Ensure operand data is on correct device
    operand = diff_op.operand
    operand.data_g = ensure_device!(operand.data_g, device_config)
    operand.data_c = ensure_device!(operand.data_c, device_config)
    
    # Use existing differentiation evaluation (works with GPU arrays)
    result = evaluate_differentiate(diff_op)
    
    # Ensure result is on correct device
    result.data_g = ensure_device!(result.data_g, device_config)
    result.data_c = ensure_device!(result.data_c, device_config)
    
    # Synchronize GPU operations
    gpu_synchronize(device_config)
    
    return result
end

# GPU-aware evaluation functions
function evaluate_handlers_gpu!(evaluator::Evaluator, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Evaluate all handlers with GPU support and write output if conditions are met"""
    
    start_time = time()
    
    for handler in evaluator.handlers
        if should_write(handler, wall_time, sim_time, iteration)
            write_handler_gpu!(handler, evaluator.solver, wall_time, sim_time, iteration, evaluator.device_config)
        end
    end
    
    # Update performance statistics
    evaluator.performance_stats.total_time += time() - start_time
    evaluator.performance_stats.total_evaluations += 1
end

function write_handler_gpu!(handler::FileHandler, solver::InitialValueSolver, wall_time::Float64, sim_time::Float64, iteration::Int, device_config::DeviceConfig)
    """Write handler output to file with GPU-aware data transfer"""
    
    # Create filename with iteration number
    base_name = splitext(handler.filename)[1]
    extension = splitext(handler.filename)[2]
    if isempty(extension)
        extension = ".h5"
    end
    
    filename = "$(base_name)_$(iteration)$(extension)"
    
    # Only rank 0 creates the file initially
    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
    else
        rank = 0
    end
    
    # Evaluate all tasks and gather data (GPU-aware)
    task_data = Dict{String, Any}()
    
    for (name, task) in handler.datasets
        data = evaluate_task_gpu(task, solver, device_config)
        task_data[name] = data
    end
    
    # Write to HDF5 file
    write_hdf5_output(filename, task_data, sim_time, iteration, rank)
    
    # Update handler state
    handler.write_count += 1
    handler.last_write_time = wall_time
    handler.last_write_sim_time = sim_time
    
    if rank == 0
        @info "Written GPU output to $filename at t=$(sim_time), iteration=$iteration"
    end
end

function evaluate_task_gpu(task, solver::InitialValueSolver, device_config::DeviceConfig)
    """GPU-aware task evaluation"""
    
    gpu_transfer_start = time()
    
    if isa(task, ScalarField)
        # Return field data in grid layout (GPU-aware)
        ensure_layout!(task, :g)
        task.data_g = ensure_device!(task.data_g, device_config)
        result = Array(task.data_g)  # Always return CPU array for file I/O
        
    elseif isa(task, VectorField)
        # Return vector components (GPU-aware)
        data = []
        for component in task.components
            ensure_layout!(component, :g)
            component.data_g = ensure_device!(component.data_g, device_config)
            push!(data, Array(component.data_g))
        end
        result = data
        
    elseif isa(task, TensorField)
        # Return tensor components (GPU-aware)
        data = []
        for i in 1:size(task.components, 1), j in 1:size(task.components, 2)
            component = task.components[i, j]
            ensure_layout!(component, :g)
            component.data_g = ensure_device!(component.data_g, device_config)
            push!(data, Array(component.data_g))
        end
        result = data
        
    elseif isa(task, Operator)
        # Evaluate operator (GPU-aware)
        operator_result = evaluate_operator_gpu(task, device_config)
        result = evaluate_task_gpu(operator_result, solver, device_config)
        
    else
        throw(ArgumentError("Unknown task type: $(typeof(task))"))
    end
    
    return result
end

# Performance tracking structure for evaluator
mutable struct EvaluatorPerformanceStats
    total_time::Float64
    gpu_transfer_time::Float64
    total_evaluations::Int
    total_writes::Int
    avg_evaluation_time::Float64
    
    function EvaluatorPerformanceStats()
        new(0.0, 0.0, 0, 0, 0.0)
    end
end

# GPU utility functions for evaluator
function move_evaluator_to_device!(evaluator::Evaluator, device_config::DeviceConfig)
    """Move evaluator to specified device"""
    
    old_device = evaluator.device_config
    evaluator.device_config = device_config
    
    # Clear GPU workspace if changing device types
    if old_device.device_type != device_config.device_type
        empty!(evaluator.gpu_workspace)
    end
    
    @info "Moved evaluator from $(old_device.device_type) to $(device_config.device_type)"
    
    return evaluator
end

function move_evaluator_to_device!(evaluator::UnifiedEvaluator, device_config::DeviceConfig)
    """Move unified evaluator to specified device"""
    
    old_device = evaluator.device_config
    evaluator.device_config = device_config
    
    # Clear GPU workspace if changing device types
    if old_device.device_type != device_config.device_type
        empty!(evaluator.gpu_workspace)
    end
    
    @info "Moved unified evaluator from $(old_device.device_type) to $(device_config.device_type)"
    
    return evaluator
end

function move_evaluator_to_device!(flow::GlobalFlowProperty, device_config::DeviceConfig)
    """Move global flow property evaluator to specified device"""
    
    old_device = flow.device_config
    flow.device_config = device_config
    
    # Clear GPU workspace if changing device types
    if old_device.device_type != device_config.device_type
        empty!(flow.gpu_workspace)
    end
    
    @info "Moved flow property evaluator from $(old_device.device_type) to $(device_config.device_type)"
    
    return flow
end

function get_evaluator_memory_info(evaluator::Union{Evaluator, UnifiedEvaluator, GlobalFlowProperty})
    """Get GPU memory usage information for evaluator"""
    
    if evaluator.device_config.device_type != CPU_DEVICE
        memory_info = gpu_memory_info(evaluator.device_config)
        
        # Estimate memory used by evaluator workspace
        evaluator_memory = 0
        for (key, arr) in evaluator.gpu_workspace
            evaluator_memory += sizeof(arr)
        end
        
        return (
            total_memory = memory_info.total,
            available_memory = memory_info.available,
            used_memory = memory_info.used,
            evaluator_memory = evaluator_memory,
            memory_utilization = evaluator_memory / memory_info.total * 100
        )
    else
        return (
            total_memory = typemax(Int64),
            available_memory = typemax(Int64),
            used_memory = 0,
            evaluator_memory = 0,
            memory_utilization = 0.0
        )
    end
end

function log_evaluator_performance(evaluator::Union{Evaluator, UnifiedEvaluator, GlobalFlowProperty})
    """Log evaluator performance statistics"""
    
    stats = evaluator.performance_stats
    
    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if rank == 0
            @info "Evaluator performance ($(evaluator.device_config.device_type)):"
            @info "  Total evaluations: $(stats.total_evaluations)"
            @info "  Total writes: $(stats.total_writes)"
            @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
            if stats.total_evaluations > 0
                @info "  Average evaluation time: $(round(stats.total_time/stats.total_evaluations*1000, digits=3)) ms"
            end
            @info "  GPU transfer time: $(round(stats.gpu_transfer_time, digits=3)) seconds ($(round(100*stats.gpu_transfer_time/max(stats.total_time, 1e-10), digits=1))%)"
        end
    else
        @info "Evaluator performance ($(evaluator.device_config.device_type)):"
        @info "  Total evaluations: $(stats.total_evaluations)"
        @info "  Total writes: $(stats.total_writes)"
        @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
        if stats.total_evaluations > 0
            @info "  Average evaluation time: $(round(stats.total_time/stats.total_evaluations*1000, digits=3)) ms"
        end
        @info "  GPU transfer time: $(round(stats.gpu_transfer_time, digits=3)) seconds ($(round(100*stats.gpu_transfer_time/max(stats.total_time, 1e-10), digits=1))%)"
    end
end

# Enhanced GPU-aware reduction operations
function global_min_gpu(reducer::GlobalArrayReducer, data::AbstractArray, device_config::DeviceConfig; empty::Float64=Inf)
    """Compute global min with GPU support"""
    
    # Move data to GPU for computation if needed
    data_device = ensure_device!(data, device_config)
    
    local_min = isempty(data_device) ? empty : minimum(Array(data_device))
    return reduce_scalar(reducer, local_min, MPI.MIN)
end

function global_max_gpu(reducer::GlobalArrayReducer, data::AbstractArray, device_config::DeviceConfig; empty::Float64=-Inf)
    """Compute global max with GPU support"""
    
    # Move data to GPU for computation if needed
    data_device = ensure_device!(data, device_config)
    
    local_max = isempty(data_device) ? empty : maximum(Array(data_device))
    return reduce_scalar(reducer, local_max, MPI.MAX)
end

function global_mean_gpu(reducer::GlobalArrayReducer, data::AbstractArray, device_config::DeviceConfig)
    """Compute global mean with GPU support"""
    
    # Move data to GPU for computation if needed
    data_device = ensure_device!(data, device_config)
    
    local_sum = sum(Array(data_device))
    local_size = Float64(length(data_device))
    global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)
    global_size = reduce_scalar(reducer, local_size, MPI.SUM)
    return global_size > 0 ? global_sum / global_size : 0.0
end

# Enhanced flow property methods with GPU support
function max_gpu(flow::GlobalFlowProperty, name::String)
    """Compute global max of a property on GPU"""
    gdata = evaluate_property(flow, name)
    return global_max_gpu(flow.reducer, gdata, flow.device_config)
end

function min_gpu(flow::GlobalFlowProperty, name::String)
    """Compute global min of a property on GPU"""
    gdata = evaluate_property(flow, name)
    return global_min_gpu(flow.reducer, gdata, flow.device_config)
end

function grid_average_gpu(flow::GlobalFlowProperty, name::String)
    """Compute global mean of a property on GPU"""
    gdata = evaluate_property(flow, name)
    return global_mean_gpu(flow.reducer, gdata, flow.device_config)
end