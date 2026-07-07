"""
Evaluator for analysis and output

Provides functionality for:
- File output (NetCDF format)
- Global property evaluation and reduction
- Volume integrals and averages

## GPU Compatibility

This module is GPU-aware and handles data transfer automatically:

- `evaluate_task`: Returns CPU arrays for file I/O (GPU→CPU via gather_array or on_architecture)
- `evaluate_property`: Returns CPU arrays (GPU→CPU via on_architecture)
- `get_task_data_array`: Ensures arrays are on CPU for file I/O (GPU→CPU via on_architecture)
- `global_min/max/mean/sum`: Work with both CPU and GPU arrays (reduction returns scalar)
- `gather_array`: Automatically transfers GPU arrays to CPU before MPI operations

All file I/O operations (NetCDF) receive CPU arrays, regardless of whether
the underlying field data is on CPU or GPU.
"""

# MPI already in Tarang.jl
using NetCDF
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

"""
    get_solver_dist(solver)

Get the distributor from a solver by looking at problem variables or state fields.
"""
function get_solver_dist(solver)
    if hasfield(typeof(solver), :problem) && !isempty(solver.problem.variables)
        return solver.problem.variables[1].dist
    elseif hasfield(typeof(solver), :state) && !isempty(solver.state)
        return solver.state[1].dist
    else
        error("Cannot get distributor from solver: no variables or state fields available")
    end
end

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

"""
    DictionaryHandler

In-memory analysis handler that stores evaluated field data in a dictionary
instead of writing to disk. Useful for on-the-fly analysis, coupling to
other solvers, or unit testing.

Scheduling follows the same cadence/sim_dt/wall_dt/max_writes logic as FileHandler.
"""
mutable struct DictionaryHandler
    datasets::Dict{String, Any}        # Tasks to evaluate (field/operator references)
    fields::Dict{String, Any}          # Evaluated results (in-memory arrays)
    cadence::Union{Int, Nothing}
    sim_dt::Union{Float64, Nothing}
    wall_dt::Union{Float64, Nothing}
    max_writes::Union{Int, Nothing}
    write_count::Int
    last_write_time::Float64
    last_write_sim_time::Float64

    function DictionaryHandler(; cadence::Union{Int, Nothing}=nothing,
                                sim_dt::Union{Float64, Nothing}=nothing,
                                wall_dt::Union{Float64, Nothing}=nothing,
                                max_writes::Union{Int, Nothing}=nothing)
        new(Dict{String, Any}(), Dict{String, Any}(),
            cadence, sim_dt, wall_dt, max_writes, 0, 0.0, 0.0)
    end
end

mutable struct Evaluator <: AbstractEvaluator
    solver::InitialValueSolver
    file_handlers::Vector{FileHandler}
    dictionary_handlers::Vector{DictionaryHandler}
    # Output handlers registered against the solver (e.g. NetCDFFileHandler from
    # `add_file_handler(path, solver, vars)`). `run!` auto-processes these every
    # step so a manual `process!(h)` in the loop is not required. Typed `Any`
    # because NetCDFFileHandler is defined in a later-loaded module.
    output_handlers::Vector{Any}
    performance_stats::EvaluatorPerformanceStats

    function Evaluator(solver::InitialValueSolver)
        perf_stats = EvaluatorPerformanceStats()
        new(solver, FileHandler[], DictionaryHandler[], Any[], perf_stats)
    end
end

# File handler management
"""Add file handler for output"""
function add_file_handler(evaluator::Evaluator, filename::String; kwargs...)
    handler = FileHandler(filename; kwargs...)
    push!(evaluator.file_handlers, handler)
    return handler
end

"""Add field or operator to file handler"""
function add_task!(handler::FileHandler, field::Union{ScalarField, VectorField, TensorField, Operator}; name::String="")
    if isempty(name)
        if hasfield(typeof(field), :name)
            name = field.name
        else
            name = "field_$(length(handler.datasets)+1)"
        end
    end
    
    handler.datasets[name] = field
end

"""Add field or operator to file handler with explicit name"""
function add_task!(handler::FileHandler, field, name::String)
    handler.datasets[name] = field
end

# Evaluation and output
"""Evaluate all handlers and write output if conditions are met"""
function evaluate_handlers!(evaluator::Evaluator, wall_time::Float64, sim_time::Float64, iteration::Int)

    for handler in evaluator.file_handlers
        # Determine write decision on rank 0 and broadcast to all ranks,
        # so that MPI collective operations in evaluate_task are entered
        # consistently by all ranks (avoiding deadlock).
        do_write = should_write(handler, wall_time, sim_time, iteration)
        # Only the wall_dt cadence is rank-divergent (per-rank wall_time); iter/sim_dt
        # decisions are identical on every rank, so the deadlock-guard Bcast (and the
        # size>1 guard) are needed only when wall_dt is set (C1 efficiency fix).
        if handler.wall_dt !== nothing && MPI.Initialized()
            dist = get_solver_dist(evaluator.solver)
            if dist.size > 1
                do_write_arr = Ref(do_write)
                MPI.Bcast!(do_write_arr, dist.comm; root=0)
                do_write = do_write_arr[]
            end
        end
        if do_write
            write_handler!(handler, evaluator.solver, wall_time, sim_time, iteration)
        end
    end

    for handler in evaluator.dictionary_handlers
        process!(handler, evaluator.solver, wall_time, sim_time, iteration)
    end
end

"""Check if handler should write output.

    Uses OR semantics: a write is triggered if ANY specified cadence condition
    is met. This ensures output is never silently skipped when multiple cadences
    are configured (e.g., cadence=10 OR sim_dt=0.1 — whichever fires first).
    """
function should_write(handler::FileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)

    # Check max writes
    if handler.max_writes !== nothing && handler.write_count >= handler.max_writes
        return false
    end

    # At least one cadence must be specified and satisfied
    has_any_cadence = false
    triggered = false

    # Check iteration cadence
    if handler.cadence !== nothing
        has_any_cadence = true
        if iteration % handler.cadence == 0
            triggered = true
        end
    end

    # Check simulation time interval
    if handler.sim_dt !== nothing
        has_any_cadence = true
        if sim_time - handler.last_write_sim_time >= handler.sim_dt
            triggered = true
        end
    end

    # Check wall time interval
    if handler.wall_dt !== nothing
        has_any_cadence = true
        if wall_time - handler.last_write_time >= handler.wall_dt
            triggered = true
        end
    end

    # If no cadence is specified, always write; otherwise require at least one trigger
    return !has_any_cadence || triggered
end

"""Write handler output to file"""
function write_handler!(handler::FileHandler, solver::InitialValueSolver, wall_time::Float64, sim_time::Float64, iteration::Int)

    # Create filename with iteration number
    base_name = splitext(handler.filename)[1]
    extension = splitext(handler.filename)[2]
    if isempty(extension)
        extension = ".nc"
    end

    filename = "$(base_name)_$(iteration)$(extension)"

    # Only rank 0 creates the file initially
    if MPI.Initialized()
        dist = get_solver_dist(solver)
        rank = MPI.Comm_rank(dist.comm)
    else
        rank = 0
    end

    # Evaluate all tasks and gather data
    task_data = Dict{String, Any}()

    for (name, task) in handler.datasets
        data = evaluate_task(task, solver)
        task_data[name] = data
    end

    # Write to NetCDF file
    write_netcdf_simple_output(filename, task_data, sim_time, iteration, rank)

    # Update handler state
    handler.write_count += 1
    handler.last_write_time = wall_time
    handler.last_write_sim_time = sim_time

    if rank == 0
        @info "Written output to $filename at t=$(sim_time), iteration=$iteration"
    end
end

"""
    Evaluate a task (field or operator) and return data.

    GPU-aware: All results are returned on CPU for file I/O compatibility.
    """
function evaluate_task(task::ScalarField, solver::InitialValueSolver)
    ensure_layout!(task, :g)
    return gather_array(task.dist, get_grid_data(task))
end

function evaluate_task(task::VectorField, solver::InitialValueSolver)
    component_arrays = [_component_grid_array(component) for component in task.components]
    return _stack_component_arrays(component_arrays)
end

function evaluate_task(task::TensorField, solver::InitialValueSolver)
    return _stack_tensor_components(task.components)
end

function evaluate_task(task::Operator, solver::InitialValueSolver)
    operator_result = evaluate_operator(task)
    return evaluate_task(operator_result, solver)
end

function evaluate_task(task::AbstractArray, solver::InitialValueSolver)
    return on_architecture(CPU(), task)
end

function evaluate_task(task, solver::InitialValueSolver)
    throw(ArgumentError("Unknown task type: $(typeof(task))"))
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
"""Evaluate operator and return result field."""
function evaluate_operator(op::Operator, layout::Symbol=:g)
    return evaluate(op, layout)
end


# Note: evaluate_curl and evaluate_laplacian are defined in operators.jl
# Do not redefine here to avoid method overwrite warnings

"""Write data to NetCDF file (simple per-snapshot output).

Uses `_nc_put` (nccreate + ncwrite) — plain `ncwrite` cannot create the file
or its variables. Each snapshot gets a fresh per-iteration file (see
`write_handler!`), so variable creation never collides.
"""
function write_netcdf_simple_output(filename::String, data::Dict{String, Any}, sim_time::Float64, iteration::Int, rank::Int)

    # Only rank 0 writes to file (task data is already gathered + replicated)
    if rank == 0
        # Write metadata as 1-element arrays
        _nc_put(filename, "sim_time", [sim_time])
        _nc_put(filename, "iteration", [Float64(iteration)])

        # Write datasets
        for (name, dataset) in data
            if isa(dataset, AbstractArray)
                _nc_put(filename, name, dataset)
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
    performance_stats::EvaluatorPerformanceStats

    function GlobalFlowProperty(solver::InitialValueSolver; cadence::Int=1)
        dist = get_solver_dist(solver)
        reducer = GlobalArrayReducer(dist.comm)
        perf_stats = EvaluatorPerformanceStats()
        new(solver, cadence, Dict{String, Any}(), reducer, perf_stats)
    end
end

"""Compute global reduction of a scalar from each process."""
function reduce_scalar(reducer::GlobalArrayReducer, local_scalar::Real, mpi_op)
    reducer.scalar_buffer[1] = Float64(local_scalar)
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Allreduce!(reducer.scalar_buffer, mpi_op, reducer.comm)
    end
    return reducer.scalar_buffer[1]
end

"""
    reduce_vector!(reducer, locals, mpi_op) -> locals

Batched in-place reduction: one `Allreduce!` over `locals` (per-item LOCAL
reductions) instead of one collective per item. Lets callers fold K scalar
reductions (e.g. one per velocity field in CFL) into a single latency-bound
collective. No-op (returns the local values) when MPI is not active.
"""
function reduce_vector!(reducer::GlobalArrayReducer, locals::AbstractVector{Float64}, mpi_op)
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Allreduce!(locals, mpi_op, reducer.comm)
    end
    return locals
end

"""
    Compute global min of all array data.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local reduction is performed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
function global_min(reducer::GlobalArrayReducer, data::AbstractArray; empty::Float64=Inf)
    if isempty(data)
        local_min = empty
    else
        # minimum() works on both CPU and GPU arrays. `parent` keeps the reduction
        # LOCAL — minimum(::PencilArray) is collective (re-reduced below; value is
        # correct under idempotent MIN but it errors on non-Intel MPI). No-op serially.
        local_min = Float64(minimum(parent(data)))
    end
    return reduce_scalar(reducer, local_min, MPI.MIN)
end

"""
    Compute global max of all array data.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local reduction is performed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
function global_max(reducer::GlobalArrayReducer, data::AbstractArray; empty::Float64=-Inf)
    if isempty(data)
        local_max = empty
    else
        # maximum() works on both CPU and GPU arrays. `parent` keeps the reduction
        # LOCAL — maximum(::PencilArray) is collective (re-reduced below; correct under
        # idempotent MAX but errors on non-Intel MPI). No-op serially.
        local_max = Float64(maximum(parent(data)))
    end
    return reduce_scalar(reducer, local_max, MPI.MAX)
end

# Scalar versions for convenience (used by CFL in flow_tools.jl)
global_min(reducer::GlobalArrayReducer, value::Real) = reduce_scalar(reducer, Float64(value), MPI.MIN)
global_max(reducer::GlobalArrayReducer, value::Real) = reduce_scalar(reducer, Float64(value), MPI.MAX)

"""
    Compute global mean of all array data.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local sum is computed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
function global_mean(reducer::GlobalArrayReducer, data::AbstractArray)
    # Reduce the LOCAL slab via `parent` (see global_sum): summing a PencilArray is
    # collective, so sum(data) + reduce_scalar would double-reduce → nprocs× mean.
    ld = parent(data)
    local_sum = Float64(real(sum(ld)))
    local_size = Float64(length(ld))
    if eltype(data) <: Complex
        local_imag = Float64(imag(sum(ld)))
        global_real = reduce_scalar(reducer, local_sum, MPI.SUM)
        global_imag = reduce_scalar(reducer, local_imag, MPI.SUM)
        global_size = reduce_scalar(reducer, local_size, MPI.SUM)
        return global_size > 0 ? complex(global_real, global_imag) / global_size : 0.0
    end
    global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)
    global_size = reduce_scalar(reducer, local_size, MPI.SUM)
    return global_size > 0 ? global_sum / global_size : 0.0
end

"""
    Add property to track.
    Following pattern: properties.add_task(property, layout='g', name=name)
    """
function add_property!(flow::GlobalFlowProperty, field::Union{ScalarField, VectorField}, name::String)
    # Store field reference for evaluation
    # In Tarang, this gets added to the dictionary handler as a task
    flow.properties[name] = field
end

"""
    Get grid data for property evaluation.
    Following pattern: gdata = self.properties[name]['g']
    Returns the grid data array for the named property.

    GPU-aware: Returns CPU arrays (GPU arrays are converted via on_architecture).
    """
function evaluate_property(flow::GlobalFlowProperty, name::String)
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

"""
    Compute global max of a property on the grid.
    Following implementation in flow_tools:107-110
    """
function property_max(flow::GlobalFlowProperty, name::String)
    gdata = evaluate_property(flow, name)
    return global_max(flow.reducer, gdata)
end

"""
    Compute global min of a property on the grid.
    Following implementation in flow_tools:102-105
    """
function property_min(flow::GlobalFlowProperty, name::String)
    gdata = evaluate_property(flow, name)
    return global_min(flow.reducer, gdata)
end

"""
    Compute global mean of a property on the grid.
    Following implementation in flow_tools:112-115
    """
function grid_average(flow::GlobalFlowProperty, name::String)
    gdata = evaluate_property(flow, name)
    return global_mean(flow.reducer, gdata)
end

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
function volume_integral(flow::GlobalFlowProperty, name::String)
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

    # evaluate_property returns `Array(get_grid_data(field))` — the local slab in PARENT
    # (storage) order. The quadrature weights below are sliced by LOGICAL axis
    # (`local_indices` on each basis axis) and broadcast against data dimension `axis`,
    # so for a PERMUTED grid pencil the two mis-align: a non-uniform Chebyshev/Legendre
    # weight would multiply the wrong axis (silently wrong integral; uniform Fourier
    # weights are unaffected). Reorder gdata into LOGICAL axis order so weights align.
    # No-op for the usual NoPermutation grid pencil and for serial plain Arrays.
    let f = haskey(flow.properties, name) ? flow.properties[name] : nothing
        if f isa ScalarField
            gd = get_grid_data(f)
            if gd isa PencilArrays.PencilArray
                perm = Tuple(PencilArrays.permutation(gd))
                if perm !== nothing                       # non-identity ⇒ storage ≠ logical
                    gdata = permutedims(gdata, invperm(collect(perm)))
                end
            end
        end
    end

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

"""
    Compute weighted integral of multi-dimensional data.

    For N-dimensional data with weights w₁, w₂, ..., wₙ along each axis,
    the integral is:
    ∫∫...∫ f(x₁,x₂,...,xₙ) dx₁dx₂...dxₙ ≈ Σᵢ₁Σᵢ₂...Σᵢₙ w₁[i₁]w₂[i₂]...wₙ[iₙ] f[i₁,i₂,...,iₙ]
    """
function compute_weighted_integral(data::AbstractArray, weights::Vector)
    ndims_data = ndims(data)
    nweights = length(weights)

    if ndims_data == 0
        return data[]
    end

    if nweights == 0
        return sum(data)
    end

    # Dimension mismatch is a correctness hazard: weights[k] is the quadrature weight for
    # SPATIAL axis k and must line up 1:1 with the data axes. A VectorField property carries
    # an extra component axis (ndims_data = nweights+1), which would shift every weight onto
    # the wrong axis and leave one spatial axis unweighted — a silently wrong integral.
    # Refuse rather than guess which axis is the component.
    if ndims_data != nweights
        throw(ArgumentError(
            "compute_weighted_integral: data has $ndims_data dims but $nweights spatial " *
            "quadrature weights. volume_integral is defined for scalar fields; for a " *
            "VectorField integrate a single component or its magnitude (a scalar)."))
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
            # Size mismatch along this axis — using mean weight as approximation.
            # For non-uniform quadrature (Chebyshev, Legendre), this can introduce
            # significant errors. This typically happens after dealiasing padding.
            @warn "Integration weight size mismatch along axis $axis: " *
                  "weight length $(length(w)) ≠ data size $(size(data, axis)). " *
                  "Using mean weight approximation." maxlog=1
            mean_weight = sum(w) / length(w)
            result = result .* mean_weight
        end
    end

    return sum(result)
end

"""
    Compute global sum of all array data across MPI processes.

    GPU-compatible: Works with both CPU arrays and GPU arrays (CuArray).
    For GPU arrays, the local sum is computed on GPU, then the scalar
    is transferred to CPU for MPI reduction across ranks.
    """
function global_sum(reducer::GlobalArrayReducer, data::AbstractArray)
    # `parent` extracts the LOCAL slab: summing a PencilArray directly is a COLLECTIVE
    # op (already global), which the reduce_scalar below would reduce a SECOND time →
    # nprocs× the true sum (and errors on non-Intel MPI). parent is a no-op for plain
    # serial/local arrays. Works on both CPU and GPU arrays.
    local_val = sum(parent(data))
    if local_val isa Complex
        global_real = reduce_scalar(reducer, Float64(real(local_val)), MPI.SUM)
        global_imag = reduce_scalar(reducer, Float64(imag(local_val)), MPI.SUM)
        return complex(global_real, global_imag)
    end
    return reduce_scalar(reducer, Float64(local_val), MPI.SUM)
end

"""Get domain from solver, handling various solver configurations."""
function get_solver_domain(solver::InitialValueSolver)
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

"""
    Compute volume average of a property.
    Following implementation in flow_tools:132-137

    Volume average = (∫ f dV) / (∫ dV) = volume_integral(f) / hypervolume
    """
function volume_average(flow::GlobalFlowProperty, name::String)
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

"""
    Compute the total volume (hypervolume) of a domain.

    For a domain with bases along coordinates x₁, x₂, ..., xₙ,
    the hypervolume is the product of the interval lengths:
    V = (b₁ - a₁) × (b₂ - a₂) × ... × (bₙ - aₙ)
    """
function compute_hypervolume(domain::Domain)
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

# ============================================================================
# DictionaryHandler — In-Memory Analysis Handler
# ============================================================================

Base.getindex(dh::DictionaryHandler, key::String) = dh.fields[key]
Base.haskey(dh::DictionaryHandler, key::String) = haskey(dh.fields, key)
Base.keys(dh::DictionaryHandler) = keys(dh.fields)

"""Add a field/operator task to the dictionary handler."""
function add_task!(handler::DictionaryHandler, field, name::String)
    handler.datasets[name] = field
end

function add_task!(handler::DictionaryHandler, field::Union{ScalarField, VectorField, TensorField, Operator}; name::String="")
    if isempty(name)
        if hasfield(typeof(field), :name)
            name = field.name
        else
            name = "field_$(length(handler.datasets)+1)"
        end
    end
    handler.datasets[name] = field
end

function should_write(handler::DictionaryHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    if handler.max_writes !== nothing && handler.write_count >= handler.max_writes
        return false
    end
    if handler.cadence !== nothing && iteration % handler.cadence != 0
        return false
    end
    if handler.sim_dt !== nothing && sim_time - handler.last_write_sim_time < handler.sim_dt
        return false
    end
    if handler.wall_dt !== nothing && wall_time - handler.last_write_time < handler.wall_dt
        return false
    end
    return true
end

function process!(handler::DictionaryHandler, solver::InitialValueSolver,
                  wall_time::Float64, sim_time::Float64, iteration::Int)
    """Evaluate all tasks and store results in-memory."""
    # Decide on rank 0 and broadcast: evaluate_task gathers collectively, so a
    # per-rank should_write (which depends on wall_time) could deadlock when only
    # some ranks enter. All ranks must agree.
    do_write = should_write(handler, wall_time, sim_time, iteration)
    # Only the wall_dt cadence is rank-divergent; gate the deadlock-guard Bcast
    # (and add the missing size>1 guard) on it (C1 efficiency fix).
    if handler.wall_dt !== nothing && MPI.Initialized()
        dist = get_solver_dist(solver)
        if dist.size > 1
            dw = Ref(do_write)
            MPI.Bcast!(dw, dist.comm; root=0)
            do_write = dw[]
        end
    end
    do_write || return

    for (name, task) in handler.datasets
        data = evaluate_task(task, solver)
        handler.fields[name] = data
    end

    handler.write_count += 1
    handler.last_write_time = wall_time
    handler.last_write_sim_time = sim_time
end

"""
    add_dictionary_handler(evaluator; kwargs...)

Add a DictionaryHandler to an Evaluator or UnifiedEvaluator.
Returns the handler so tasks can be added to it.
"""
function add_dictionary_handler(evaluator::Evaluator; kwargs...)
    handler = DictionaryHandler(; kwargs...)
    push!(evaluator.dictionary_handlers, handler)
    return handler
end

# ============================================================================
# VirtualFileHandler — Parallel NetCDF Output with Per-Rank Files
# ============================================================================

"""
    VirtualFileHandler

Parallel file handler where each MPI rank writes to its own NetCDF file,
and rank 0 creates a manifest/virtual file that maps global array regions
to per-rank files.

Output structure:
- `{name}_s{set}_p{rank}.nc` — per-rank data files
- `{name}_s{set}.nc` — manifest file with layout metadata

A post-processing `merge_virtual!` function can combine per-rank files
into a single consolidated file.
"""
mutable struct VirtualFileHandler
    base_path::String                    # Directory for output
    name::String                         # Handler name
    datasets::Dict{String, Any}          # Tasks to evaluate
    # Scheduling (same fields as FileHandler)
    cadence::Union{Int, Nothing}
    sim_dt::Union{Float64, Nothing}
    wall_dt::Union{Float64, Nothing}
    max_writes::Union{Int, Nothing}
    write_count::Int
    last_write_time::Float64
    last_write_sim_time::Float64
    # File management
    set_num::Int
    file_write_num::Int
    # MPI
    comm::MPI.Comm
    rank::Int
    nprocs::Int

    function VirtualFileHandler(base_path::String, name::String;
                                comm::MPI.Comm=MPI.COMM_WORLD,
                                cadence::Union{Int, Nothing}=nothing,
                                sim_dt::Union{Float64, Nothing}=nothing,
                                wall_dt::Union{Float64, Nothing}=nothing,
                                max_writes::Union{Int, Nothing}=nothing)
        rank = MPI.Initialized() ? MPI.Comm_rank(comm) : 0
        nprocs = MPI.Initialized() ? MPI.Comm_size(comm) : 1
        mkpath(base_path)
        new(base_path, name, Dict{String, Any}(),
            cadence, sim_dt, wall_dt, max_writes,
            0, 0.0, 0.0, 1, 0, comm, rank, nprocs)
    end
end

function add_task!(handler::VirtualFileHandler, field, name::String)
    handler.datasets[name] = field
end

function add_task!(handler::VirtualFileHandler, field::Union{ScalarField, VectorField, TensorField, Operator}; name::String="")
    if isempty(name)
        if hasfield(typeof(field), :name)
            name = field.name
        else
            name = "field_$(length(handler.datasets)+1)"
        end
    end
    handler.datasets[name] = field
end

function should_write(handler::VirtualFileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    if handler.max_writes !== nothing && handler.write_count >= handler.max_writes
        return false
    end
    if handler.cadence !== nothing && iteration % handler.cadence != 0
        return false
    end
    if handler.sim_dt !== nothing && sim_time - handler.last_write_sim_time < handler.sim_dt
        return false
    end
    if handler.wall_dt !== nothing && wall_time - handler.last_write_time < handler.wall_dt
        return false
    end
    return true
end

"""
    _nc_put(filename, name, data)

Create variable `name` (with per-variable dimension names) in `filename` —
creating the file on first variable — then write `data`. NetCDF.jl's plain
`ncwrite` requires the file and variable to already exist.
"""
function _nc_put(filename::String, name::String, data::AbstractArray)
    dimargs = Any[]
    for (i, s) in enumerate(size(data))
        push!(dimargs, "$(name)_d$(i)")
        push!(dimargs, s)
    end
    NetCDF.nccreate(filename, name, dimargs...; t=eltype(data))
    ncwrite(data, filename, name)
    return nothing
end

"""
    _local_task_data(task) -> (block, starts, gshape)

Per-rank slab of a task plus its placement: `block` is this rank's LOCAL data
as a plain CPU `Array` (logical axis order), `starts` are the 1-based GLOBAL
indices of the block's first element, `gshape` is the global array shape.
No communication — this is the whole point of VirtualFileHandler.
"""
function _local_task_data(task::ScalarField)
    ensure_layout!(task, :g)
    d = get_grid_data(task)
    if isa(d, PencilArrays.PencilArray)
        gv = PencilArrays.global_view(d)            # permutation-aware
        block = collect(gv)                          # plain Array, logical order
        starts = Int[first(r) for r in axes(gv)]
        gshape = collect(Int, PencilArrays.size_global(d))
    else
        block = Array(on_architecture(CPU(), d))     # GPU → host copy
        starts = ones(Int, ndims(block))
        gshape = collect(Int, size(block))
    end
    if eltype(block) <: Complex
        # NetCDF has no complex type. Encode as a real array with a trailing size-2
        # axis [real, imag]; every rank owns the full re/im axis, so the per-rank
        # slab placement (starts/gshape) just gains a leading 1 / trailing 2 and the
        # downstream write + offset merge stay entirely real. Readers reconstruct via
        # `complex.(A[..., 1], A[..., 2])`.
        block = cat(real(block), imag(block); dims=ndims(block) + 1)
        push!(starts, 1)
        push!(gshape, 2)
    end
    return block, starts, gshape
end

function _local_task_data(task::VectorField)
    parts = [_local_task_data(c) for c in task.components]
    blocks = first.(parts)
    block = _stack_component_arrays(blocks)
    starts = vcat(1, parts[1][2])
    gshape = vcat(length(parts), parts[1][3])
    return block, starts, gshape
end

function _local_task_data(task::Operator)
    result = evaluate_operator(task)
    return _local_task_data(result)
end

# Replicated plain arrays (e.g. integrate/average results): every rank holds
# the full array, so each writes it at offset 1 and merge overwrites in place.
function _local_task_data(task::AbstractArray)
    block = Array(on_architecture(CPU(), task))
    return block, ones(Int, ndims(block)), collect(Int, size(block))
end

"""
    process!(handler::VirtualFileHandler, solver, wall_time, sim_time, iteration)

Write per-rank data files (LOCAL slabs only — no gather) and rank-0 manifest.
Each dataset `name` is stored with `name_start` (global offsets of the slab)
and the manifest stores `name_gshape` (global shape) for reconstruction.
"""
function process!(handler::VirtualFileHandler, solver::InitialValueSolver,
                  wall_time::Float64, sim_time::Float64, iteration::Int)
    # Decide on rank 0 and broadcast: should_write depends on wall_time, which
    # differs per rank, so a per-rank check could return on some ranks and fall
    # through to the collective per-rank write + MPI.Barrier below on others →
    # deadlock. All ranks must agree to enter or skip together.
    do_write = should_write(handler, wall_time, sim_time, iteration)
    # Only the wall_dt cadence is rank-divergent; gate the deadlock-guard Bcast on it
    # (handler carries its own comm/nprocs) (C1 efficiency fix).
    if handler.wall_dt !== nothing && MPI.Initialized() && handler.nprocs > 1
        dw = Ref(do_write)
        MPI.Bcast!(dw, handler.comm; root=0)
        do_write = dw[]
    end
    do_write || return

    handler.file_write_num += 1
    set_num = handler.set_num

    # Evaluate all tasks to (local block, global starts, global shape)
    task_data = Dict{String, Tuple{Array, Vector{Int}, Vector{Int}}}()
    for (name, task) in handler.datasets
        task_data[name] = _local_task_data(task)
    end

    # Each rank writes its own NetCDF file with only its local slab
    rank_filename = joinpath(handler.base_path,
                             "$(handler.name)_s$(set_num)_p$(handler.rank).nc")

    _nc_put(rank_filename, "rank", [Float64(handler.rank)])
    _nc_put(rank_filename, "nprocs", [Float64(handler.nprocs)])
    _nc_put(rank_filename, "sim_time", [sim_time])
    _nc_put(rank_filename, "iteration", [Float64(iteration)])
    _nc_put(rank_filename, "set_num", [Float64(set_num)])
    _nc_put(rank_filename, "write_num", [Float64(handler.file_write_num)])

    for (name, (block, starts, _)) in task_data
        _nc_put(rank_filename, name, block)
        _nc_put(rank_filename, "$(name)_start", Float64.(starts))
    end

    # Rank 0 writes the manifest with global layout metadata
    if handler.rank == 0
        manifest_filename = joinpath(handler.base_path,
                                     "$(handler.name)_s$(set_num).nc")

        _nc_put(manifest_filename, "nprocs", [Float64(handler.nprocs)])
        _nc_put(manifest_filename, "sim_time", [sim_time])
        _nc_put(manifest_filename, "iteration", [Float64(iteration)])
        _nc_put(manifest_filename, "set_num", [Float64(set_num)])

        for (name, (_, _, gshape)) in task_data
            _nc_put(manifest_filename, "$(name)_gshape", Float64.(gshape))
        end

        @debug "VirtualFileHandler wrote set $set_num: $manifest_filename"
    end

    # Synchronize
    if MPI.Initialized()
        MPI.Barrier(handler.comm)
    end

    handler.write_count += 1
    handler.set_num += 1  # Advance to next set so each snapshot gets its own file
    handler.last_write_time = wall_time
    handler.last_write_sim_time = sim_time
end

"""
    merge_virtual!(handler::VirtualFileHandler, set_num::Int=1)

Post-processing: merge per-rank NetCDF files into a single consolidated file.
Reads the manifest to determine the file layout, then concatenates data
along the decomposed dimension.
"""
function merge_virtual!(handler::VirtualFileHandler; set_num::Int=max(1, handler.set_num - 1))
    # Default to the LAST WRITTEN set: process! increments handler.set_num after
    # each write, so handler.set_num points one past the last set actually on disk.
    manifest_filename = joinpath(handler.base_path,
                                 "$(handler.name)_s$(set_num).nc")
    if !isfile(manifest_filename)
        error("Manifest file not found: $manifest_filename")
    end

    output_filename = joinpath(handler.base_path,
                               "$(handler.name)_s$(set_num)_merged.nc")

    # Read manifest
    nprocs_arr = ncread(manifest_filename, "nprocs")
    nprocs = Int(nprocs_arr[1])

    # Metadata variable names (not field data)
    metadata_vars = Set(["rank", "nprocs", "sim_time", "iteration", "set_num", "write_num"])
    _is_layout_var(name) = endswith(name, "_start") || endswith(name, "_gshape")

    # Place each rank's local block at its recorded global offsets
    merged_data = Dict{String, Array{Float64}}()
    for p in 0:(nprocs - 1)
        rank_file = joinpath(handler.base_path,
                             "$(handler.name)_s$(set_num)_p$(p).nc")
        if !isfile(rank_file)
            @warn "Rank file not found: $rank_file"
            continue
        end

        var_names = String[]
        NetCDF.open(rank_file) do nc
            for (name, _) in nc.vars
                (name in metadata_vars || _is_layout_var(name)) && continue
                push!(var_names, name)
            end
        end

        for name in var_names
            block = ncread(rank_file, name)
            starts = Int.(ncread(rank_file, "$(name)_start"))
            merged = get!(merged_data, name) do
                gshape = Int.(ncread(manifest_filename, "$(name)_gshape"))
                zeros(Float64, gshape...)
            end
            rng = ntuple(d -> starts[d]:(starts[d] + size(block, d) - 1), ndims(block))
            merged[rng...] = block
        end
    end

    # Write merged file — copy manifest metadata then reconstructed data
    _nc_put(output_filename, "sim_time", ncread(manifest_filename, "sim_time"))
    _nc_put(output_filename, "iteration", ncread(manifest_filename, "iteration"))
    _nc_put(output_filename, "nprocs", nprocs_arr)

    for (name, merged) in merged_data
        _nc_put(output_filename, name, merged)
    end

    @info "Merged virtual files into $output_filename"
    return output_filename
end

# Unified evaluator interface for NetCDF output
mutable struct UnifiedEvaluator
    solver::InitialValueSolver
    netcdf_handlers::Vector{NetCDFFileHandler}
    dictionary_handlers::Vector{DictionaryHandler}
    virtual_handlers::Vector{VirtualFileHandler}
    performance_stats::EvaluatorPerformanceStats

    function UnifiedEvaluator(solver::InitialValueSolver)
        perf_stats = EvaluatorPerformanceStats()
        new(solver, NetCDFFileHandler[], DictionaryHandler[], VirtualFileHandler[], perf_stats)
    end
end

# Alias for backward compatibility
const NetCDFEvaluator = UnifiedEvaluator

"""Add a VirtualFileHandler to the UnifiedEvaluator."""
function add_virtual_file_handler(evaluator::UnifiedEvaluator, base_path::String, name::String; kwargs...)
    dist = get_solver_dist(evaluator.solver)
    handler = VirtualFileHandler(base_path, name; comm=dist.comm, kwargs...)
    push!(evaluator.virtual_handlers, handler)
    return handler
end

# Enhanced evaluator functions with NetCDF support
"""Create evaluator for solver"""
function create_evaluator(solver::InitialValueSolver)
    return Evaluator(solver)
end

"""Create NetCDF evaluator for solver"""
function create_netcdf_evaluator(solver::InitialValueSolver)
    return UnifiedEvaluator(solver)
end

"""Create unified evaluator for NetCDF output"""
function create_unified_evaluator(solver::InitialValueSolver)
    return UnifiedEvaluator(solver)
end

"""Add file handler with automatic format detection

    Supported formats:
    - :netcdf or :nc - NetCDF format
    - :auto - Auto-detect from file extension (defaults to NetCDF)
    """
function add_file_handler(evaluator::UnifiedEvaluator, filename::String, format::Symbol=:auto; kwargs...)

    # Ensure NetCDF extension
    _, ext = splitext(filename)
    ext_lower = lowercase(ext)
    if ext_lower ∉ [".nc", ".netcdf"]
        # Strip legacy HDF5 extensions if present
        filename = replace(filename, r"\.(h5|hdf5)$"i => "")
        if !endswith(filename, ".nc")
            filename = filename * ".nc"
        end
    end

    # NetCDFFileHandler requires dist and vars arguments
    dist = get_solver_dist(evaluator.solver)
    vars = hasfield(typeof(evaluator.solver), :namespace) ? evaluator.solver.namespace : Dict{String, Any}()
    handler = NetCDFFileHandler(filename, dist, vars; kwargs...)
    push!(evaluator.netcdf_handlers, handler)
    @info "Added NetCDF file handler: $filename"
    return handler
end

# Wrapper functions for NetCDFFileHandler compatibility
"""Check if NetCDF handler should write - delegates to check_schedule"""
function should_write(handler::NetCDFFileHandler, wall_time::Float64, sim_time::Float64, iteration::Int)
    return check_schedule(handler; iteration=iteration, wall_time=wall_time, sim_time=sim_time)
end

"""Write NetCDF data - delegates to process!"""
function write_netcdf_data!(handler::NetCDFFileHandler, sim_time::Float64, iteration::Int)
    return process!(handler; iteration=iteration, sim_time=sim_time)
end

"""Close NetCDF handler - delegates to close!"""
function close_netcdf_handler!(handler::NetCDFFileHandler)
    return close!(handler)
end

"""Add a DictionaryHandler to the UnifiedEvaluator."""
function add_dictionary_handler(evaluator::UnifiedEvaluator; kwargs...)
    handler = DictionaryHandler(; kwargs...)
    push!(evaluator.dictionary_handlers, handler)
    return handler
end

"""Evaluate NetCDF and dictionary handlers."""
function evaluate_unified_handlers!(evaluator::UnifiedEvaluator, wall_time::Float64, sim_time::Float64, iteration::Int)

    # Evaluate NetCDF handlers
    for handler in evaluator.netcdf_handlers
        # Broadcast write decision from rank 0 to prevent MPI deadlock
        # (write_netcdf_data! may invoke collective MPI operations)
        do_write = should_write(handler, wall_time, sim_time, iteration)
        # Only the wall_dt cadence is rank-divergent; gate the deadlock-guard Bcast
        # (and add the missing size>1 guard) on it (C1 efficiency fix).
        if handler.wall_dt !== nothing && MPI.Initialized()
            dist = get_solver_dist(evaluator.solver)
            if dist.size > 1
                do_write_arr = Ref(do_write)
                MPI.Bcast!(do_write_arr, dist.comm; root=0)
                do_write = do_write_arr[]
            end
        end
        if do_write
            write_netcdf_data!(handler, sim_time, iteration)
        end
    end

    # Evaluate dictionary handlers (in-memory). process! broadcasts its own
    # write decision (see DictionaryHandler process!), so no gating here.
    for handler in evaluator.dictionary_handlers
        process!(handler, evaluator.solver, wall_time, sim_time, iteration)
    end

    # Evaluate virtual (per-rank NetCDF) handlers. Previously omitted entirely,
    # so add_virtual_file_handler output was silently never written via the
    # unified path. process! broadcasts its own write decision.
    for handler in evaluator.virtual_handlers
        process!(handler, evaluator.solver, wall_time, sim_time, iteration)
    end

    return evaluator
end

# Convenience functions for NetCDF-specific features
function add_netcdf_handler(evaluator::UnifiedEvaluator, filename::String;
                           precision::Type{<:AbstractFloat}=Float64,
                           kwargs...)
    """Add NetCDF handler with full control over NetCDF-specific options"""

    # NetCDFFileHandler requires dist and vars arguments
    dist = get_solver_dist(evaluator.solver)
    vars = hasfield(typeof(evaluator.solver), :namespace) ? evaluator.solver.namespace : Dict{String, Any}()

    handler = NetCDFFileHandler(filename, dist, vars;
                               precision=precision,
                               kwargs...)
    push!(evaluator.netcdf_handlers, handler)

    @info "Added NetCDF handler: $filename (precision: $precision)"
    return handler
end

"""Set coordinate information for all NetCDF handlers

    Note: NetCDFFileHandler stores coordinate information in its vars field
    and handles coordinate writing internally during process!() calls.
    This """
function set_coordinates!(evaluator::UnifiedEvaluator, coords::Dict{String, Any})

    for handler in evaluator.netcdf_handlers
        # Update the vars dictionary with coordinate information
        merge!(handler.vars, coords)
    end

    return evaluator
end

"""Finalize and close all file handlers."""
function finalize_evaluator!(evaluator::UnifiedEvaluator)

    # Close NetCDF handlers
    for handler in evaluator.netcdf_handlers
        close_netcdf_handler!(handler)
    end

    return evaluator
end


"""
    Extract array data from task data.
    Handles different field types (ScalarField, VectorField, etc.)

    GPU-aware: Automatically converts GPU arrays to CPU for file I/O.
    """
function get_task_data_array(task_data::Any)

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
"""Add a field/array to be written by this file handler."""
function add_task!(handler::FileHandler, field::Union{ScalarField, VectorField, AbstractArray}, name::String)
    handler.datasets[name] = field
end

"""Log evaluator performance statistics"""
function log_evaluator_performance(evaluator::Union{Evaluator, UnifiedEvaluator, GlobalFlowProperty})

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
# similar to GlobalArrayReducer pattern.
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
local_data = abs.(get_grid_data(q))
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
       UnifiedEvaluator, NetCDFEvaluator,
       DictionaryHandler, VirtualFileHandler

# Export file handler functions
export add_file_handler, add_task!, evaluate_handlers!, should_write,
       write_handler!, evaluate_task, evaluate_operator, write_netcdf_simple_output

# Export global reduction functions
export reduce_scalar, global_min, global_max, global_mean, global_sum

# Export property functions
export add_property!, evaluate_property, property_max, property_min,
       grid_average, volume_integral, compute_weighted_integral,
       volume_average, compute_hypervolume, get_solver_domain

# Export evaluator creation functions
export create_evaluator, create_netcdf_evaluator, create_unified_evaluator,
       add_dictionary_handler, add_virtual_file_handler, merge_virtual!

# Export unified evaluator functions
export evaluate_unified_handlers!, add_netcdf_handler, set_coordinates!,
       finalize_evaluator!, get_task_data_array

# Export NetCDF wrapper functions
export write_netcdf_data!, close_netcdf_handler!

# Export performance logging
export log_evaluator_performance
