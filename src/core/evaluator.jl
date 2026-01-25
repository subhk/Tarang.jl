"""
Evaluator for analysis and output

Provides functionality for:
- File output (HDF5 and NetCDF formats)
- Global property evaluation and reduction
- Volume integrals and averages

## GPU Compatibility

This module is GPU-aware and handles data transfer automatically:

- `evaluate_task`: Returns CPU arrays for file I/O (GPU→CPU via gather_array or on_architecture)
- `evaluate_property`: Returns CPU arrays (GPU→CPU via on_architecture)
- `get_task_data_array`: Ensures arrays are on CPU for file I/O (GPU→CPU via on_architecture)
- `global_min/max/mean/sum`: Work with both CPU and GPU arrays (reduction returns scalar)
- `gather_array`: Automatically transfers GPU arrays to CPU before MPI operations

All file I/O operations (HDF5, NetCDF) receive CPU arrays, regardless of whether
the underlying field data is on CPU or GPU.
"""

using HDF5
using MPI
using Dates: now

# Performance tracking structure for evaluator
mutable struct EvaluatorPerformanceStats
    total_time::Float64
    total_evaluations::Int
    total_writes::Int
    avg_evaluation_time::Float64

    function EvaluatorPerformanceStats()
        new(0.0, 0, 0, 0.0)
    end
end

# NetCDF support is included in main Tarang.jl module

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
    workspace::Dict{String, AbstractArray}
    performance_stats::EvaluatorPerformanceStats

    function Evaluator(solver::InitialValueSolver)
        workspace = Dict{String, AbstractArray}()
        perf_stats = EvaluatorPerformanceStats()
        new(solver, FileHandler[], workspace, perf_stats)
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
        rank = MPI.Comm_rank(solver.dist.comm)
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
    """
    Evaluate a task (field or operator) and return data.

    GPU-aware: All results are returned on CPU for file I/O compatibility.
    """

    if isa(task, ScalarField)
        # gather_array handles GPU→CPU conversion
        ensure_layout!(task, :g)
        result = gather_array(task.dist, get_grid_data(task))

    elseif isa(task, VectorField)
        # _component_grid_array uses gather_array which handles GPU→CPU
        component_arrays = [_component_grid_array(component) for component in task.components]
        result = _stack_component_arrays(component_arrays)

    elseif isa(task, TensorField)
        # _stack_tensor_components uses _component_grid_array which handles GPU→CPU
        result = _stack_tensor_components(task.components)

    elseif isa(task, Operator)
        operator_result = evaluate_operator(task)
        result = evaluate_task(operator_result, solver)

    elseif isa(task, AbstractArray)
        # Ensure array is on CPU for file I/O
        result = on_architecture(CPU(), task)

    else
        throw(ArgumentError("Unknown task type: $(typeof(task))"))
    end

    return result
end

function _component_grid_array(component::ScalarField)
    ensure_layout!(component, :g)
    return gather_array(component.dist, get_grid_data(component))
end

function _stack_component_arrays(component_arrays::Vector{<:AbstractArray})
    if isempty(component_arrays)
        return Float64[]  # Return properly typed empty array
    end

    first_array = component_arrays[1]
    comp_shape = size(first_array)
    result = Array{eltype(first_array)}(undef, length(component_arrays), comp_shape...)
    index_tail = ntuple(_ -> Colon(), ndims(first_array))

    for (i, arr) in enumerate(component_arrays)
        if size(arr) != comp_shape
            throw(ArgumentError("VectorField component shapes must match; got $(size(arr)) and $comp_shape"))
        end
        result[i, index_tail...] = arr
    end

    return result
end

function _stack_tensor_components(components::AbstractMatrix{<:ScalarField})
    n1, n2 = size(components)
    first_array = _component_grid_array(components[1, 1])
    comp_shape = size(first_array)
    result = Array{eltype(first_array)}(undef, n1, n2, comp_shape...)
    index_tail = ntuple(_ -> Colon(), ndims(first_array))

    for i in 1:n1, j in 1:n2
        arr = _component_grid_array(components[i, j])
        if size(arr) != comp_shape
            throw(ArgumentError("TensorField component shapes must match; got $(size(arr)) and $comp_shape"))
        end
        result[i, j, index_tail...] = arr
    end

    return result
end
function evaluate_operator(op::Operator, layout::Symbol=:g)
    """Evaluate operator and return result field."""
    return evaluate(op, layout)
end


# Note: evaluate_curl and evaluate_laplacian are defined in operators.jl
# Do not redefine here to avoid method overwrite warnings

function write_hdf5_output(filename::String, data::Dict{String, Any}, sim_time::Float64, iteration::Int, rank::Int)
    """Write data to HDF5 file"""

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

struct GlobalArrayReducer
    comm::MPI.Comm
    scalar_buffer::Vector{Float64}

    function GlobalArrayReducer(comm::MPI.Comm)
        new(comm, zeros(Float64, 1))
    end

    function GlobalArrayReducer()
        if !MPI.Initialized()
            throw(ErrorException("MPI must be initialized before creating GlobalArrayReducer without explicit communicator"))
        end
        new(MPI.COMM_WORLD, zeros(Float64, 1))
    end
end

# Analysis utilities
mutable struct GlobalFlowProperty
    solver::InitialValueSolver
    cadence::Int
    properties::Dict{String, Any}
    reducer::GlobalArrayReducer
    evaluator_handler::Union{Nothing, FileHandler}
    workspace::Dict{String, AbstractArray}
    performance_stats::EvaluatorPerformanceStats

    function GlobalFlowProperty(solver::InitialValueSolver; cadence::Int=1)
        reducer = GlobalArrayReducer(solver.dist.comm)
        workspace = Dict{String, AbstractArray}()
        perf_stats = EvaluatorPerformanceStats()
        new(solver, cadence, Dict{String, Any}(), reducer, nothing, workspace, perf_stats)
    end
end

function reduce_scalar(reducer::GlobalArrayReducer, local_scalar::Real, mpi_op)
    """Compute global reduction of a scalar from each process."""
    reducer.scalar_buffer[1] = Float64(local_scalar)
    if MPI.Initialized()
        MPI.Allreduce!(reducer.scalar_buffer, mpi_op, reducer.comm)
    end
    return reducer.scalar_buffer[1]
end

function global_min(reducer::GlobalArrayReducer, data::AbstractArray; empty::Float64=Inf)
    """
    Compute global min of all array data.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local reduction is performed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
    if isempty(data)
        local_min = empty
    else
        # minimum() works on both CPU and GPU arrays
        # For GPU arrays, CUDA.jl performs the reduction on GPU and returns a scalar
        local_min = Float64(minimum(data))
    end
    return reduce_scalar(reducer, local_min, MPI.MIN)
end

function global_max(reducer::GlobalArrayReducer, data::AbstractArray; empty::Float64=-Inf)
    """
    Compute global max of all array data.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local reduction is performed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
    if isempty(data)
        local_max = empty
    else
        # maximum() works on both CPU and GPU arrays
        # For GPU arrays, CUDA.jl performs the reduction on GPU and returns a scalar
        local_max = Float64(maximum(data))
    end
    return reduce_scalar(reducer, local_max, MPI.MAX)
end

# Scalar versions for convenience (used by CFL in flow_tools.jl)
global_min(reducer::GlobalArrayReducer, value::Real) = reduce_scalar(reducer, Float64(value), MPI.MIN)
global_max(reducer::GlobalArrayReducer, value::Real) = reduce_scalar(reducer, Float64(value), MPI.MAX)

function global_mean(reducer::GlobalArrayReducer, data::AbstractArray)
    """
    Compute global mean of all array data.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local sum is computed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
    # sum() and length() work on both CPU and GPU arrays
    local_sum = Float64(sum(data))
    local_size = Float64(length(data))
    global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)
    global_size = reduce_scalar(reducer, local_size, MPI.SUM)
    return global_size > 0 ? global_sum / global_size : 0.0
end

function add_property!(flow::GlobalFlowProperty, field::Union{ScalarField, VectorField}, name::String)
    """
    Add property to track.
    Following pattern: properties.add_task(property, layout='g', name=name)
    """
    # Store field reference for evaluation
    # In Tarang, this gets added to the dictionary handler as a task
    flow.properties[name] = field
end

function evaluate_property(flow::GlobalFlowProperty, name::String)
    """
    Get grid data for property evaluation.
    Following pattern: gdata = self.properties[name]['g']
    Returns the grid data array for the named property.

    GPU-aware: Returns CPU arrays (GPU arrays are converted via on_architecture).
    """
    if !haskey(flow.properties, name)
        throw(KeyError("Property '$name' not found"))
    end

    field = flow.properties[name]

    if isa(field, ScalarField)
        # Ensure grid layout and get CPU array
        ensure_layout!(field, :g)
        result = on_architecture(CPU(), get_grid_data(field))
    elseif isa(field, VectorField)
        # Return all components stacked, consistent with evaluate_task
        component_arrays = Vector{AbstractArray}(undef, length(field.components))
        for (i, comp) in enumerate(field.components)
            ensure_layout!(comp, :g)
            component_arrays[i] = on_architecture(CPU(), get_grid_data(comp))
        end
        result = _stack_component_arrays(component_arrays)
    elseif isa(field, AbstractArray)
        # Ensure array is on CPU
        result = on_architecture(CPU(), field)
    else
        throw(ArgumentError("Unsupported property type: $(typeof(field))"))
    end

    return result
end

function property_max(flow::GlobalFlowProperty, name::String)
    """
    Compute global max of a property on the grid.
    Following implementation in flow_tools:107-110
    """
    gdata = evaluate_property(flow, name)
    return global_max(flow.reducer, gdata)
end

function property_min(flow::GlobalFlowProperty, name::String)
    """
    Compute global min of a property on the grid.
    Following implementation in flow_tools:102-105
    """
    gdata = evaluate_property(flow, name)
    return global_min(flow.reducer, gdata)
end

function grid_average(flow::GlobalFlowProperty, name::String)
    """
    Compute global mean of a property on the grid.
    Following implementation in flow_tools:112-115
    """
    gdata = evaluate_property(flow, name)
    return global_mean(flow.reducer, gdata)
end

function volume_integral(flow::GlobalFlowProperty, name::String)
    """
    Compute volume integral of a property.
    Following implementation in flow_tools:117-130

    Uses proper quadrature weights for each basis type:
    - Fourier: uniform weights (trapezoidal rule)
    - Chebyshev: Clenshaw-Curtis weights
    - Legendre: Gauss-Legendre weights

    The integral is computed as:
    ∫ f(x) dx ≈ Σᵢ wᵢ f(xᵢ)

    For multi-dimensional domains, the weights are the outer product
    of 1D weights along each axis.
    """
    # Check for precomputed integral
    integral_name = "_$(name)_integral"
    if haskey(flow.properties, integral_name)
        integral_data = evaluate_property(flow, integral_name)
        integral_value = global_sum(flow.reducer, integral_data)
        return integral_value
    end

    # Get field data in grid space
    if !haskey(flow.properties, name)
        throw(KeyError("Property '$name' not found"))
    end

    gdata = evaluate_property(flow, name)

    # Get domain and integration weights
    domain = get_solver_domain(flow.solver)
    if domain === nothing
        # Fallback: simple sum approximation (assumes uniform grid with unit spacing)
        @warn "No domain information available, using simple sum for volume integral"
        local_integral = sum(gdata)
        return global_sum(flow.reducer, [local_integral])
    end

    # Get integration weights for each dimension
    # Request CPU weights since gdata is on CPU (from evaluate_property)
    weights = integration_weights(domain, on_device=false)

    if isempty(weights)
        # Fallback: simple sum
        local_integral = sum(gdata)
        return global_sum(flow.reducer, [local_integral])
    end

    # Slice weights to local ranges for distributed data
    dist = domain.dist
    local_weights = Vector{Any}(undef, length(weights))
    for (i, basis) in enumerate(domain.bases)
        axis = get_basis_axis(dist, basis) + 1
        local_range = local_indices(dist, axis, length(weights[i]))
        local_weights[i] = weights[i][local_range]
    end

    # Compute weighted integral
    # For multi-dimensional data, we need to apply weights along each axis
    local_integral = compute_weighted_integral(gdata, local_weights)

    # Global reduction across MPI processes
    return global_sum(flow.reducer, [local_integral])
end

function compute_weighted_integral(data::AbstractArray, weights::Vector)
    """
    Compute weighted integral of multi-dimensional data.

    For N-dimensional data with weights w₁, w₂, ..., wₙ along each axis,
    the integral is:
    ∫∫...∫ f(x₁,x₂,...,xₙ) dx₁dx₂...dxₙ ≈ Σᵢ₁Σᵢ₂...Σᵢₙ w₁[i₁]w₂[i₂]...wₙ[iₙ] f[i₁,i₂,...,iₙ]
    """
    ndims_data = ndims(data)
    nweights = length(weights)

    if ndims_data == 0
        return data[]
    end

    if nweights == 0
        return sum(data)
    end

    # Handle dimension mismatch
    if ndims_data != nweights
        @warn "Data dimensions ($ndims_data) != weight dimensions ($nweights), using available weights"
    end

    # 1D case
    if ndims_data == 1 && nweights >= 1
        w = weights[1]
        if length(w) == length(data)
            return sum(w .* data)
        else
            # Size mismatch, interpolate weights or use uniform
            return sum(data) * (sum(w) / length(w))
        end
    end

    # Multi-dimensional case: apply weights successively along each axis
    result = data
    for (axis, w) in enumerate(weights)
        if axis > ndims_data
            break
        end

        # Create weight array with proper shape for broadcasting
        weight_shape = ones(Int, ndims_data)
        weight_shape[axis] = length(w)

        # Check if weight length matches data size along this axis
        if length(w) == size(data, axis)
            w_reshaped = reshape(w, weight_shape...)
            result = result .* w_reshaped
        else
            # Size mismatch along this axis, use mean weight
            mean_weight = sum(w) / length(w)
            result = result .* mean_weight
        end
    end

    return sum(result)
end

function global_sum(reducer::GlobalArrayReducer, data::AbstractArray)
    """
    Compute global sum of all array data across MPI processes.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local sum is computed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
    # sum() works on both CPU and GPU arrays
    local_sum = Float64(sum(data))
    return reduce_scalar(reducer, local_sum, MPI.SUM)
end

function get_solver_domain(solver::InitialValueSolver)
    """Get domain from solver, handling various solver configurations."""
    # Try direct domain access
    if hasfield(typeof(solver), :domain) && solver.domain !== nothing
        return solver.domain
    end

    # Try through problem
    if hasfield(typeof(solver), :problem) && solver.problem !== nothing
        problem = solver.problem
        if hasfield(typeof(problem), :domain) && problem.domain !== nothing
            return problem.domain
        end
        # Try through variables
        if hasfield(typeof(problem), :variables) && !isempty(problem.variables)
            var = problem.variables[1]
            if hasfield(typeof(var), :domain) && var.domain !== nothing
                return var.domain
            end
        end
    end

    # Try through state
    if hasfield(typeof(solver), :state) && solver.state !== nothing && !isempty(solver.state)
        field = solver.state[1]
        if hasfield(typeof(field), :domain) && field.domain !== nothing
            return field.domain
        end
    end

    return nothing
end

function volume_average(flow::GlobalFlowProperty, name::String)
    """
    Compute volume average of a property.
    Following implementation in flow_tools:132-137

    Volume average = (∫ f dV) / (∫ dV) = volume_integral(f) / hypervolume
    """
    # Get domain to compute hypervolume
    domain = get_solver_domain(flow.solver)

    if domain === nothing
        # Fallback to grid average
        return grid_average(flow, name)
    end

    # Compute hypervolume (total volume of domain)
    hypervolume = compute_hypervolume(domain)

    if hypervolume <= 0.0
        # Fallback to grid average
        @warn "Invalid hypervolume ($hypervolume), using grid average"
        return grid_average(flow, name)
    end

    # Compute volume integral and divide by hypervolume
    integral_value = volume_integral(flow, name)
    return integral_value / hypervolume
end

function compute_hypervolume(domain::Domain)
    """
    Compute the total volume (hypervolume) of a domain.

    For a domain with bases along coordinates x₁, x₂, ..., xₙ,
    the hypervolume is the product of the interval lengths:
    V = (b₁ - a₁) × (b₂ - a₂) × ... × (bₙ - aₙ)
    """
    hypervolume = 1.0

    for basis in domain.bases
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :bounds)
            bounds = basis.meta.bounds
            if length(bounds) >= 2
                L = bounds[2] - bounds[1]
                # Validate that interval length is positive
                if L <= 0
                    @warn "Invalid bounds for basis: [$( bounds[1]), $(bounds[2])], using absolute value"
                    L = abs(L)
                    if L == 0
                        L = 1.0  # Fallback to unit interval for degenerate case
                    end
                end
                hypervolume *= L
            else
                # Assume unit interval
                hypervolume *= 1.0
            end
        else
            # Default to unit interval
            hypervolume *= 1.0
        end
    end

    return hypervolume
end

# Enhanced evaluator functions with NetCDF support
function create_evaluator(solver::InitialValueSolver)
    """Create evaluator for solver"""
    return Evaluator(solver)
end

function create_netcdf_evaluator(solver::InitialValueSolver)
    """Create NetCDF evaluator for solver"""
    return UnifiedEvaluator(solver)
end

# Unified evaluator interface that supports both HDF5 and NetCDF
mutable struct UnifiedEvaluator
    solver::InitialValueSolver
    hdf5_handlers::Vector{FileHandler}
    netcdf_handlers::Vector{NetCDFFileHandler}
    workspace::Dict{String, AbstractArray}
    performance_stats::EvaluatorPerformanceStats

    function UnifiedEvaluator(solver::InitialValueSolver)
        workspace = Dict{String, AbstractArray}()
        perf_stats = EvaluatorPerformanceStats()
        new(solver, FileHandler[], NetCDFFileHandler[], workspace, perf_stats)
    end
end

# Alias for backward compatibility
const NetCDFEvaluator = UnifiedEvaluator

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
        _, ext = splitext(filename)
        ext_lower = lowercase(ext)
        if ext_lower in [".nc", ".netcdf"]
            format = :netcdf
        elseif ext_lower in [".h5", ".hdf5"]
            format = :hdf5
        else
            # Default to NetCDF for new files
            format = :netcdf
            if ext_lower != ".nc"
                filename = filename * ".nc"
            end
        end
    end

    if format in [:netcdf, :nc]
        # NetCDFFileHandler requires dist and vars arguments
        dist = evaluator.solver.dist
        vars = hasfield(typeof(evaluator.solver), :namespace) ? evaluator.solver.namespace : Dict{String, Any}()
        handler = NetCDFFileHandler(filename, dist, vars; kwargs...)
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

# Wrapper functions for NetCDFFileHandler compatibility
function should_write(handler::NetCDFFileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Check if NetCDF handler should write - delegates to check_schedule"""
    return check_schedule(handler; iteration=iteration, wall_time=wall_time, sim_time=sim_time)
end

function write_netcdf_data!(handler::NetCDFFileHandler, sim_time::Float64, iteration::Int)
    """Write NetCDF data - delegates to process!"""
    return process!(handler; iteration=iteration, sim_time=sim_time)
end

function close_netcdf_handler!(handler::NetCDFFileHandler)
    """Close NetCDF handler - delegates to close!"""
    return close!(handler)
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
    # Note: NetCDFFileHandler tracks writes internally via total_write_num/file_write_num
    # which are updated by process!() called from write_netcdf_data!()
    for handler in evaluator.netcdf_handlers
        if should_write(handler, wall_time, sim_time, iteration)
            write_netcdf_data!(handler, sim_time, iteration)
            # NetCDFFileHandler doesn't have last_write_time field;
            # write tracking is handled internally by process!()
        end
    end

    return evaluator
end

# Convenience functions for NetCDF-specific features
function add_netcdf_handler(evaluator::UnifiedEvaluator, filename::String;
                           precision::Type{<:AbstractFloat}=Float64,
                           kwargs...)
    """Add NetCDF handler with full control over NetCDF-specific options"""

    # NetCDFFileHandler requires dist and vars arguments
    dist = evaluator.solver.dist
    vars = hasfield(typeof(evaluator.solver), :namespace) ? evaluator.solver.namespace : Dict{String, Any}()

    handler = NetCDFFileHandler(filename, dist, vars;
                               precision=precision,
                               kwargs...)
    push!(evaluator.netcdf_handlers, handler)

    @info "Added NetCDF handler: $filename (precision: $precision)"
    return handler
end

function set_coordinates!(evaluator::UnifiedEvaluator, coords::Dict{String, Any})
    """Set coordinate information for all NetCDF handlers

    Note: NetCDFFileHandler stores coordinate information in its vars field
    and handles coordinate writing internally during process!() calls.
    This function updates the vars dictionary for all handlers.
    """

    for handler in evaluator.netcdf_handlers
        # Update the vars dictionary with coordinate information
        merge!(handler.vars, coords)
    end

    return evaluator
end

function finalize_evaluator!(evaluator::UnifiedEvaluator)
    """Finalize and close all file handlers"""
    
    # Close NetCDF handlers
    for handler in evaluator.netcdf_handlers
        close_netcdf_handler!(handler)
    end
    
    # Close HDF5 handlers
    # Note: HDF5 files are automatically closed when h5open block exits,
    # so no explicit cleanup is needed. This loop is kept for potential
    # future cleanup operations (e.g., flushing buffers, updating metadata).
    for handler in evaluator.hdf5_handlers
        # Reset write tracking state if needed for potential reuse
        handler.write_count = 0
        handler.last_write_time = 0.0
        handler.last_write_sim_time = 0.0
    end
    
    return evaluator
end

# Helper functions for HDF5 compatibility
function should_write_hdf5(handler::FileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    """Determine if HDF5 handler should write (consistent with should_write logic)"""

    # Check max writes limit
    if handler.max_writes !== nothing && handler.write_count >= handler.max_writes
        return false
    end

    # Check cadence - must be at iteration cadence if specified
    if handler.cadence !== nothing
        if iteration % handler.cadence != 0
            return false
        end
    end

    # Check simulation time interval - must have elapsed if specified
    if handler.sim_dt !== nothing
        if sim_time - handler.last_write_sim_time < handler.sim_dt
            return false
        end
    end

    # Check wall time interval - must have elapsed if specified
    if handler.wall_dt !== nothing
        if wall_time - handler.last_write_time < handler.wall_dt
            return false
        end
    end

    return true
end

function write_hdf5_data!(handler::FileHandler, sim_time::Float64, iteration::Int)
    """
    Write HDF5 data following Tarang structure.
    Based on Tarang/core/evaluator:580-595
    """
    
    @debug "Writing HDF5 data: $(handler.filename), t=$sim_time, iter=$iteration"
    
    # Increment write counter
    handler.write_count += 1
    handler.last_write_sim_time = sim_time
    
    # Evaluate tasks on all ranks (MPI gather may be required)
    task_arrays = Dict{String, Any}()
    for (task_name, task_data) in handler.datasets
        task_arrays[task_name] = get_task_data_array(task_data)
    end

    rank = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0

    # Only rank 0 writes the file
    if rank == 0
        h5open(handler.filename, "cw") do file
            # Write file metadata (following Tarang write_file_metadata)
            write_file_metadata!(file, handler, iteration, sim_time)
            
            # Write task data (following Tarang write_task pattern)
            for (task_name, task_data) in task_arrays
                write_task_data!(file, task_name, task_data, handler.write_count)
            end
        end
    end
    
    @debug "HDF5 write completed: $(handler.filename), write #$(handler.write_count)"
    
    return handler
end

function write_file_metadata!(file::HDF5.File, handler::FileHandler, iteration::Int, sim_time::Float64)
    """
    Write file metadata and time scales.
    Following write_file_metadata pattern
    """
    
    # Create/update metadata group
    if !haskey(file, "metadata")
        metadata = create_group(file, "metadata")
    else
        metadata = file["metadata"]
    end
    
    # Write time information (following Tarang pattern)
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

function write_task_data!(file::HDF5.File, task_name::String, data_array::AbstractArray, write_number::Int)
    """
    Write task data to HDF5 file.
    Following write_task pattern from evaluator:641-650 and 693-702

    Accepts AbstractArray directly (preferred) or field objects via the Any method below.
    """

    # Create tasks group if it doesn't exist
    if !haskey(file, "tasks")
        tasks_group = create_group(file, "tasks")
    else
        tasks_group = file["tasks"]
    end

    # Use data_array directly - no conversion needed
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
        dataset_index = (1, ntuple(_ -> Colon(), ndims(data_array))...)
        dataset[dataset_index...] = data_array
    else
        # Resize existing dataset and write new data
        dataset = tasks_group[task_name]
        current_shape = size(dataset)
        new_shape = (write_number, current_shape[2:end]...)
        
        HDF5.set_extent_dims(dataset, new_shape)
        dataset_index = (write_number, ntuple(_ -> Colon(), ndims(data_array))...)
        dataset[dataset_index...] = data_array
    end
end

# Fallback method for field objects - converts to array first
function write_task_data!(file::HDF5.File, task_name::String, task_data, write_number::Int)
    """Write task data (field objects) to HDF5 file - converts to array first."""
    data_array = get_task_data_array(task_data)
    return write_task_data!(file, task_name, data_array, write_number)
end

function get_task_data_array(task_data::Any)
    """
    Extract array data from task data.
    Handles different field types (ScalarField, VectorField, etc.)

    GPU-aware: Automatically converts GPU arrays to CPU for file I/O.
    """

    if isa(task_data, ScalarField)
        # Get grid-space data (gather_array handles GPU→CPU conversion)
        ensure_layout!(task_data, :g)
        return gather_array(task_data.dist, get_grid_data(task_data))
    elseif isa(task_data, VectorField)
        # _component_grid_array uses gather_array which handles GPU→CPU
        component_arrays = [_component_grid_array(comp) for comp in task_data.components]
        return _stack_component_arrays(component_arrays)
    elseif isa(task_data, AbstractArray)
        # Direct array data - ensure it's on CPU for file I/O
        # GPU arrays are converted to CPU; CPU arrays pass through unchanged
        return on_architecture(CPU(), task_data)
    else
        throw(ArgumentError("Unsupported task data type: $(typeof(task_data))"))
    end
end

# Utility function to add tasks to file handler
# Note: Parameter order (handler, field, name) matches earlier definitions at lines 67-83
function add_task!(handler::FileHandler, field::Union{ScalarField, VectorField, AbstractArray}, name::String)
    """Add a field/array to be written by this file handler."""
    handler.datasets[name] = field
end

function log_evaluator_performance(evaluator::Union{Evaluator, UnifiedEvaluator, GlobalFlowProperty})
    """Log evaluator performance statistics"""

    stats = evaluator.performance_stats

    # Only log on rank 0 (or if MPI not initialized)
    rank = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
    rank != 0 && return nothing

    @info "Evaluator performance:"
    @info "  Total evaluations: $(stats.total_evaluations)"
    @info "  Total writes: $(stats.total_writes)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    if stats.total_evaluations > 0
        @info "  Average evaluation time: $(round(stats.total_time/stats.total_evaluations*1000, digits=3)) ms"
    end

    return nothing
end

# ============================================================================
# Convenience functions for global reductions on fields
# ============================================================================
# These functions automatically handle MPI reductions across all ranks,
# similar to Dedalus's GlobalArrayReducer pattern.
#
# GPU + Multi-GPU Support:
# - Works transparently with CPU arrays and GPU arrays (CuArray)
# - For multi-GPU (one GPU per MPI rank): local GPU reduction → scalar → MPI reduction
# - No explicit data transfer needed; CUDA.jl handles scalar transfer automatically

"""
    global_max(field::ScalarField)

Compute the global maximum of a scalar field across all MPI ranks.
This is the correct way to compute maximum values in parallel simulations.

GPU-compatible: Works with fields on CPU or GPU. For GPU fields, the local
reduction is performed on GPU, then the scalar is transferred to CPU for
MPI reduction across ranks.

# Example
```julia
ensure_layout!(q, :g)
max_q = global_max(q)  # Returns same value on all ranks
```

# Multi-GPU Example
```julia
# Each MPI rank has its own GPU
# global_max automatically:
# 1. Computes local max on each GPU
# 2. Transfers scalar to CPU
# 3. MPI.Allreduce across all ranks
max_q = global_max(q)
```
"""
function global_max(field::ScalarField)
    ensure_layout!(field, :g)
    reducer = GlobalArrayReducer(field.dist.comm)
    return global_max(reducer, get_grid_data(field))
end

"""
    global_min(field::ScalarField)

Compute the global minimum of a scalar field across all MPI ranks.

# Example
```julia
ensure_layout!(T, :g)
min_T = global_min(T)  # Returns same value on all ranks
```
"""
function global_min(field::ScalarField)
    ensure_layout!(field, :g)
    reducer = GlobalArrayReducer(field.dist.comm)
    return global_min(reducer, get_grid_data(field))
end

"""
    global_mean(field::ScalarField)

Compute the global mean of a scalar field across all MPI ranks.

# Example
```julia
ensure_layout!(rho, :g)
mean_rho = global_mean(rho)  # Returns same value on all ranks
```
"""
function global_mean(field::ScalarField)
    ensure_layout!(field, :g)
    reducer = GlobalArrayReducer(field.dist.comm)
    return global_mean(reducer, get_grid_data(field))
end

"""
    global_sum(field::ScalarField)

Compute the global sum of a scalar field across all MPI ranks.

# Example
```julia
ensure_layout!(mass, :g)
total_mass = global_sum(mass)  # Returns same value on all ranks
```
"""
function global_sum(field::ScalarField)
    ensure_layout!(field, :g)
    reducer = GlobalArrayReducer(field.dist.comm)
    return global_sum(reducer, get_grid_data(field))
end

"""
    global_max(dist::Distributor, data::AbstractArray)

Compute the global maximum of array data across all MPI ranks using the
communicator from the given distributor.

# Example
```julia
local_data = abs.(q.data_g)
max_val = global_max(q.dist, local_data)
```
"""
function global_max(dist::Distributor, data::AbstractArray)
    reducer = GlobalArrayReducer(dist.comm)
    return global_max(reducer, data)
end

"""
    global_min(dist::Distributor, data::AbstractArray)

Compute the global minimum of array data across all MPI ranks using the
communicator from the given distributor.
"""
function global_min(dist::Distributor, data::AbstractArray)
    reducer = GlobalArrayReducer(dist.comm)
    return global_min(reducer, data)
end

"""
    global_mean(dist::Distributor, data::AbstractArray)

Compute the global mean of array data across all MPI ranks using the
communicator from the given distributor.
"""
function global_mean(dist::Distributor, data::AbstractArray)
    reducer = GlobalArrayReducer(dist.comm)
    return global_mean(reducer, data)
end

"""
    global_sum(dist::Distributor, data::AbstractArray)

Compute the global sum of array data across all MPI ranks using the
communicator from the given distributor.
"""
function global_sum(dist::Distributor, data::AbstractArray)
    reducer = GlobalArrayReducer(dist.comm)
    return global_sum(reducer, data)
end

# ============================================================================
# Exports
# ============================================================================

# Export types
export EvaluatorPerformanceStats, FileHandler, Evaluator,
       GlobalArrayReducer, GlobalFlowProperty,
       UnifiedEvaluator, NetCDFEvaluator

# Export file handler functions
export add_file_handler, add_task!, evaluate_handlers!, should_write,
       write_handler!, evaluate_task, evaluate_operator, write_hdf5_output

# Export global reduction functions
export reduce_scalar, global_min, global_max, global_mean, global_sum

# Export property functions
export add_property!, evaluate_property, property_max, property_min,
       grid_average, volume_integral, compute_weighted_integral,
       volume_average, compute_hypervolume, get_solver_domain

# Export evaluator creation functions
export create_evaluator, create_netcdf_evaluator, create_unified_evaluator

# Export unified evaluator functions
export evaluate_unified_handlers!, add_netcdf_handler, set_coordinates!,
       finalize_evaluator!, should_write_hdf5, write_hdf5_data!,
       write_file_metadata!, write_task_data!, get_task_data_array

# Export NetCDF wrapper functions
export write_netcdf_data!, close_netcdf_handler!

# Export performance logging
export log_evaluator_performance
