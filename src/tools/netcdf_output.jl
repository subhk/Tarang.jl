"""
NetCDF Output Support for Tarang.jl

This module provides NetCDF output capabilities:
- Per-processor NetCDF files: handler_s1/handler_s1_p0.nc
- User-selectable precision (Float32/Float64)
- Rich metadata and coordinate information
- CF conventions compliance
- Set-based file organization with automatic file management
- Task-based field output system
"""

using NetCDF
using MPI
using Dates
using Printf

# Precision types for user selection
const NetCDFPrecision = Union{Type{Float32}, Type{Float64}}

"""
NetCDF File Handler matching Tarang H5FileHandler structure

Follows Tarang pattern:
- base_path/handler_name_s1/handler_name_s1_p0.nc for processor files
- base_path/handler_name_s1.nc for gathered files
- /scales/ group with time coordinates
- /tasks/ group with field data
"""
mutable struct NetCDFFileHandler
    # Base attributes (matching Tarang)
    base_path::String
    name::String
    dist::Any  # Domain distributor
    vars::Dict{String, Any}  # Variables for parsing
    
    # Scheduling (matching Tarang Handler)
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
        
        # Base path handling (matching Tarang)
        if endswith(base_path, ".nc")
            base_path = base_path[1:end-3]  # Remove .nc extension
        end
        
        # Extract handler name from path
        name = splitpath(base_path)[end]
        
        # Initialize file numbering 
        set_num = 1
        total_write_num = 0
        file_write_num = 0
        
        # Mode handling (matching Tarang logic)
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
Get current set path following Tarang naming: handler_name_s1/
"""
function current_path(handler::NetCDFFileHandler)
    set_name = "$(handler.name)_s$(handler.set_num)"
    return joinpath(dirname(handler.base_path), set_name)
end

"""
Get current file path following Tarang naming: handler_name_s1/handler_name_s1_p0.nc
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
Add task to handler (matching Tarang API)

# Example usage (matching Tarang style):
snapshots = evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
"""
function add_task!(handler::NetCDFFileHandler, task; layout="g", name=nothing, scales=nothing, postprocess=nothing)
    # Default name
    if name === nothing
        name = string(task)
    end
    
    # Create operator following Tarang patterns
    operator = create_operator(task, handler.vars, handler.dist)
    
    # Check and remedy scales following Tarang logic
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
    
    # Create task dictionary (matching Tarang structure)
    task_dict = Dict{String, Any}(
        "operator" => operator,
        "layout" => layout_obj,
        "scales" => scales,
        "name" => name,
        "dtype" => get_operator_dtype(operator, handler.precision),
        "postprocess" => postprocess
    )
    
    # Add data distribution information following Tarang get_data_distribution
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
Create operator from different input types following Tarang patterns
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
    parse_field_expression(expr_str::String, vars::Dict, dist) -> Union{Operator, ScalarField, Dict}

Parse a field expression string into an operator tree or field.

Supports expressions like:
- Variable references: "u", "velocity"
- Arithmetic: "u + v", "2*u", "u - v"
- Derivatives: "dx(u)", "d(u,x)", "lap(u)"
- Products: "u*v", "dot(u, v)"

# Arguments
- `expr_str`: Expression string to parse
- `vars`: Dictionary mapping variable names to fields/values
- `dist`: Distributor for creating new fields if needed

# Returns
- Parsed operator, field, or Dict representation

# Examples
```julia
vars = Dict("u" => u_field, "v" => v_field, "nu" => 0.01)
op = parse_field_expression("nu * lap(u)", vars, dist)
```
"""
function parse_field_expression(expr_str::String, vars::Dict, dist)
    expr_str = strip(expr_str)

    if isempty(expr_str)
        @warn "Empty expression string"
        return Dict("type" => "zero", "value" => 0.0)
    end

    # Convert vars Dict to String keys for namespace compatibility
    namespace = Dict{String, Any}()
    for (k, v) in vars
        namespace[string(k)] = v
    end

    # Handle simple variable lookup first
    if haskey(namespace, expr_str)
        value = namespace[expr_str]
        if isa(value, ScalarField) || isa(value, VectorField) || isa(value, TensorField)
            return value
        elseif isa(value, Number)
            return ConstantOperator(Float64(value))
        end
    end

    # Handle numeric literals
    try
        num_val = parse(Float64, expr_str)
        return ConstantOperator(num_val)
    catch
        # Not a number, continue parsing
    end

    # Handle zero
    if expr_str == "0" || lowercase(expr_str) == "zero"
        return ZeroOperator()
    end

    # Try to parse using the expression parser from problems.jl
    try
        # Use Meta.parse for proper Julia expression parsing
        parsed_ast = Meta.parse(expr_str)

        # Evaluate the parsed AST with the namespace
        result = evaluate_field_ast(parsed_ast, namespace, dist)

        if result !== nothing
            return result
        end
    catch e
        @debug "Expression parsing failed: $e, trying alternative methods"
    end

    # Fallback: try pattern-based parsing for common expressions
    result = parse_field_expression_patterns(expr_str, namespace, dist)
    if result !== nothing
        return result
    end

    # Last resort: return a deferred evaluation Dict
    @debug "Returning deferred expression for: $expr_str"
    return Dict(
        "type" => "deferred_expression",
        "expr" => expr_str,
        "namespace" => namespace
    )
end

"""
    evaluate_field_ast(ast, namespace::Dict{String,Any}, dist) -> Union{Operator, ScalarField, Nothing}

Evaluate a parsed Julia AST in the context of field operations.
"""
function evaluate_field_ast(ast, namespace::Dict{String,Any}, dist)
    # Handle symbols (variable references)
    if isa(ast, Symbol)
        name = string(ast)
        if haskey(namespace, name)
            value = namespace[name]
            if isa(value, Number)
                return ConstantOperator(Float64(value))
            end
            return value
        end
        return nothing
    end

    # Handle numeric literals
    if isa(ast, Number)
        return ConstantOperator(Float64(ast))
    end

    # Handle expressions
    if isa(ast, Expr)
        if ast.head == :call
            return evaluate_field_call(ast, namespace, dist)
        elseif ast.head == :block
            # Handle begin...end blocks - evaluate last expression
            for arg in ast.args
                if !isa(arg, LineNumberNode)
                    result = evaluate_field_ast(arg, namespace, dist)
                    if result !== nothing
                        return result
                    end
                end
            end
        end
    end

    return nothing
end

"""
    evaluate_field_call(ast::Expr, namespace::Dict{String,Any}, dist) -> Union{Operator, Nothing}

Evaluate a function call expression for field operations.
"""
function evaluate_field_call(ast::Expr, namespace::Dict{String,Any}, dist)
    func = ast.args[1]
    args = ast.args[2:end]

    func_name = isa(func, Symbol) ? string(func) : string(func)

    # Evaluate arguments recursively
    eval_args = []
    for arg in args
        eval_arg = evaluate_field_ast(arg, namespace, dist)
        if eval_arg === nothing
            return nothing
        end
        push!(eval_args, eval_arg)
    end

    # Handle arithmetic operators
    if func_name == "+" && length(eval_args) == 2
        return create_add_operator(eval_args[1], eval_args[2])
    elseif func_name == "-" && length(eval_args) == 2
        return create_subtract_operator(eval_args[1], eval_args[2])
    elseif func_name == "-" && length(eval_args) == 1
        # Unary minus
        return create_multiply_operator(ConstantOperator(-1.0), eval_args[1])
    elseif func_name == "*" && length(eval_args) == 2
        return create_multiply_operator(eval_args[1], eval_args[2])
    elseif func_name == "/" && length(eval_args) == 2
        if isa(eval_args[2], ConstantOperator)
            return create_multiply_operator(eval_args[1], ConstantOperator(1.0 / eval_args[2].value))
        end
        return nothing

    # Handle differential operators
    elseif func_name in ["dx", "dy", "dz"] && length(eval_args) >= 1
        coord_name = func_name[2:end]  # "x", "y", or "z"
        operand = eval_args[1]
        order = length(eval_args) >= 2 && isa(eval_args[2], ConstantOperator) ? Int(eval_args[2].value) : 1
        return create_differentiate_operator(operand, coord_name, order, namespace)

    elseif func_name == "d" && length(eval_args) >= 2
        # d(field, coord) or d(field, coord, order)
        operand = eval_args[1]
        coord = eval_args[2]
        coord_name = isa(coord, Symbol) ? string(coord) : (isa(coord, String) ? coord : "x")
        order = length(eval_args) >= 3 && isa(eval_args[3], ConstantOperator) ? Int(eval_args[3].value) : 1
        return create_differentiate_operator(operand, coord_name, order, namespace)

    elseif func_name in ["lap", "laplacian"] && length(eval_args) == 1
        return create_laplacian_operator(eval_args[1], namespace)

    elseif func_name in ["grad", "gradient"] && length(eval_args) == 1
        return create_gradient_operator(eval_args[1], namespace)

    elseif func_name in ["div", "divergence"] && length(eval_args) == 1
        return create_divergence_operator(eval_args[1], namespace)

    elseif func_name == "curl" && length(eval_args) == 1
        return create_curl_operator(eval_args[1], namespace)

    elseif func_name == "dot" && length(eval_args) == 2
        return create_dot_product(eval_args[1], eval_args[2])
    end

    return nothing
end

"""
    parse_field_expression_patterns(expr_str::String, namespace::Dict, dist) -> Union{Operator, Nothing}

Parse common field expression patterns using regex matching.
Fallback for when AST parsing fails.
"""
function parse_field_expression_patterns(expr_str::String, namespace::Dict, dist)
    # Pattern: dx(var), dy(var), dz(var)
    m = match(r"^d([xyz])\((\w+)\)$", expr_str)
    if m !== nothing
        coord_name = m.captures[1]
        var_name = m.captures[2]
        if haskey(namespace, var_name)
            return create_differentiate_operator(namespace[var_name], coord_name, 1, namespace)
        end
    end

    # Pattern: lap(var) or laplacian(var)
    m = match(r"^(?:lap|laplacian)\((\w+)\)$", expr_str)
    if m !== nothing
        var_name = m.captures[1]
        if haskey(namespace, var_name)
            return create_laplacian_operator(namespace[var_name], namespace)
        end
    end

    # Pattern: scalar * var
    m = match(r"^([\d.]+)\s*\*\s*(\w+)$", expr_str)
    if m !== nothing
        scalar = parse(Float64, m.captures[1])
        var_name = m.captures[2]
        if haskey(namespace, var_name)
            return create_multiply_operator(ConstantOperator(scalar), namespace[var_name])
        end
    end

    # Pattern: var * scalar
    m = match(r"^(\w+)\s*\*\s*([\d.]+)$", expr_str)
    if m !== nothing
        var_name = m.captures[1]
        scalar = parse(Float64, m.captures[2])
        if haskey(namespace, var_name)
            return create_multiply_operator(namespace[var_name], ConstantOperator(scalar))
        end
    end

    return nothing
end

# Helper functions for creating operators

function create_add_operator(left, right)
    if isa(left, Operator) && isa(right, Operator)
        return AddOperator(left, right)
    elseif isa(left, ScalarField) && isa(right, ScalarField)
        # Return a deferred addition
        return Dict("type" => "add", "left" => left, "right" => right)
    end
    return Dict("type" => "add", "left" => left, "right" => right)
end

function create_subtract_operator(left, right)
    if isa(left, Operator) && isa(right, Operator)
        return SubtractOperator(left, right)
    end
    return Dict("type" => "subtract", "left" => left, "right" => right)
end

function create_multiply_operator(left, right)
    if isa(left, Operator) || isa(right, Operator)
        left_op = isa(left, Operator) ? left : ConstantOperator(Float64(left))
        right_val = isa(right, Number) ? Float64(right) : right
        return MultiplyOperator(left_op, right_val)
    end
    return Dict("type" => "multiply", "left" => left, "right" => right)
end

function create_differentiate_operator(operand, coord_name::String, order::Int, namespace::Dict)
    # Try to find coordinate in namespace
    coord = get(namespace, coord_name, nothing)
    if coord === nothing
        # Create a placeholder coordinate reference
        coord = Symbol(coord_name)
    end

    if isa(operand, ScalarField)
        return Differentiate(operand, coord, order)
    end
    return Dict("type" => "differentiate", "operand" => operand, "coord" => coord_name, "order" => order)
end

function create_laplacian_operator(operand, namespace::Dict)
    if isa(operand, ScalarField) || isa(operand, VectorField)
        coordsys = get(namespace, "coordsys", nothing)
        if coordsys !== nothing
            return CartesianLaplacian(operand, coordsys)
        end
    end
    return Dict("type" => "laplacian", "operand" => operand)
end

function create_gradient_operator(operand, namespace::Dict)
    if isa(operand, ScalarField)
        coordsys = get(namespace, "coordsys", nothing)
        if coordsys !== nothing
            return CartesianGradient(operand, coordsys)
        end
    end
    return Dict("type" => "gradient", "operand" => operand)
end

function create_divergence_operator(operand, namespace::Dict)
    if isa(operand, VectorField)
        coordsys = get(namespace, "coordsys", nothing)
        if coordsys !== nothing
            return CartesianDivergence(operand, coordsys)
        end
    end
    return Dict("type" => "divergence", "operand" => operand)
end

function create_curl_operator(operand, namespace::Dict)
    if isa(operand, VectorField)
        coordsys = get(namespace, "coordsys", nothing)
        if coordsys !== nothing
            return CartesianCurl(operand, coordsys)
        end
    end
    return Dict("type" => "curl", "operand" => operand)
end

function create_dot_product(left, right)
    return Dict("type" => "dot", "left" => left, "right" => right)
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
    get_domain_dealias(operator) -> Union{Tuple, Vector, Nothing}

Extract dealias scales from an operator's domain.

Returns the dealias scales defined in the domain's bases, which control
the padding used for dealiasing in nonlinear computations (typically 3/2 rule).

# Arguments
- `operator`: An operator, field, or Dict that may contain domain information

# Returns
- Tuple or Vector of dealias scales for each dimension, or `nothing` if not available
"""
function get_domain_dealias(operator)
    # Try to extract domain from different operator types
    domain = nothing

    if isa(operator, ScalarField)
        domain = operator.domain
    elseif isa(operator, VectorField)
        # Get domain from first component
        if !isempty(operator.components)
            domain = operator.components[1].domain
        end
    elseif isa(operator, TensorField)
        # Get domain from first component
        if !isempty(operator.components)
            domain = operator.components[1, 1].domain
        end
    elseif isa(operator, Dict)
        # Check for domain in Dict-based operators
        if haskey(operator, "domain")
            domain = operator["domain"]
        elseif haskey(operator, "operand")
            # Recursively get from operand
            return get_domain_dealias(operator["operand"])
        end
    elseif hasfield(typeof(operator), :operand)
        # Recursive case for wrapped operators
        return get_domain_dealias(operator.operand)
    elseif hasfield(typeof(operator), :domain)
        domain = operator.domain
    end

    if domain === nothing
        return nothing
    end

    # Extract dealias scales from domain's bases
    if hasfield(typeof(domain), :bases) && domain.bases !== nothing
        dealias_scales = Float64[]
        for basis in domain.bases
            if basis === nothing
                push!(dealias_scales, 1.0)
            elseif hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :dealias)
                push!(dealias_scales, Float64(basis.meta.dealias))
            elseif hasfield(typeof(basis), :dealias)
                push!(dealias_scales, Float64(basis.dealias))
            else
                push!(dealias_scales, 1.0)  # Default: no dealiasing
            end
        end
        return Tuple(dealias_scales)
    end

    return nothing
end

"""
    remedy_scales(dist, scales) -> Tuple

Remedy and validate scales parameter for a distributor.

Converts various scale input formats to a standardized tuple of Float64 values,
ensuring consistency and validity.

# Arguments
- `dist`: Distributor object (or nothing for default behavior)
- `scales`: Scale input - can be Nothing, Number, Tuple, or Vector

# Returns
- Tuple of Float64 scales matching the distributor's dimensionality

# Example
```julia
scales = remedy_scales(dist, 1.5)      # -> (1.5, 1.5, 1.5) for 3D
scales = remedy_scales(dist, nothing)  # -> (1.0, 1.0, 1.0) for 3D
scales = remedy_scales(dist, [1.0, 2.0, 1.5])  # -> (1.0, 2.0, 1.5)
```
"""
function remedy_scales(dist, scales)
    # Determine dimensionality
    ndim = 3  # Default
    if dist !== nothing
        if hasfield(typeof(dist), :dim)
            ndim = dist.dim
        elseif hasfield(typeof(dist), :coords)
            ndim = length(dist.coords)
        end
    end

    # Handle nothing -> default scales of 1.0
    if scales === nothing
        return Tuple(ones(Float64, ndim))
    end

    # Handle single number -> broadcast to all dimensions
    if isa(scales, Number)
        return Tuple(fill(Float64(scales), ndim))
    end

    # Handle tuple or array -> convert to Float64 tuple
    if isa(scales, Tuple) || isa(scales, AbstractVector)
        scale_vec = Float64.(collect(scales))

        # Pad or truncate to match dimensionality
        if length(scale_vec) < ndim
            # Extend with last value or 1.0
            last_val = isempty(scale_vec) ? 1.0 : scale_vec[end]
            scale_vec = vcat(scale_vec, fill(last_val, ndim - length(scale_vec)))
        elseif length(scale_vec) > ndim
            scale_vec = scale_vec[1:ndim]
        end

        # Validate positive scales
        if any(s -> s <= 0, scale_vec)
            @warn "Scales must be positive; using absolute values"
            scale_vec = abs.(scale_vec)
            scale_vec[scale_vec .== 0] .= 1.0
        end

        return Tuple(scale_vec)
    end

    # Fallback: return default
    @warn "Unrecognized scales type $(typeof(scales)), using default"
    return Tuple(ones(Float64, ndim))
end

"""
    get_layout_object(dist, layout) -> Any

Get a layout object from a distributor for a given layout specification.

# Arguments
- `dist`: Distributor object
- `layout`: Layout specification (Symbol like :g or :c, or layout object)

# Returns
- Layout object suitable for data transformation
"""
function get_layout_object(dist, layout)
    if dist === nothing
        return layout
    end

    # If layout is already an object, return it
    if !isa(layout, Symbol) && !isa(layout, String)
        return layout
    end

    # Convert string to symbol
    layout_sym = isa(layout, String) ? Symbol(layout) : layout

    # Check if distributor has layouts cache
    if hasfield(typeof(dist), :layouts) && dist.layouts !== nothing
        layout_key = layout_sym == :g ? "grid" : (layout_sym == :c ? "coeff" : string(layout_sym))

        if haskey(dist.layouts, layout_key)
            return dist.layouts[layout_key]
        end

        # Try alternative keys
        if haskey(dist.layouts, layout_sym)
            return dist.layouts[layout_sym]
        end
    end

    # Return the symbol as-is if no layout object found
    return layout_sym
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
Get data distribution information following Tarang patterns
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
    get_operator_domain(operator)

Extract domain information from an operator or field.

Returns a Dict with:
- `dims`: Number of dimensions
- `shape`: Global shape tuple
- `domain`: The actual Domain object if available (for accessing bases, distributor)
- `dist`: The Distributor object if available

Handles ScalarField, VectorField, TensorField, and Dict-based operators.
"""
function get_operator_domain(operator)
    if isa(operator, Dict) && haskey(operator, "shape")
        return Dict(
            "dims" => length(operator["shape"]),
            "shape" => operator["shape"],
            "domain" => get(operator, "domain", nothing),
            "dist" => get(operator, "dist", nothing)
        )
    elseif isa(operator, ScalarField)
        # Get domain from field if available
        domain_obj = operator.domain
        dist_obj = domain_obj !== nothing ? domain_obj.dist : nothing

        # Get global shape from domain's bases
        if domain_obj !== nothing
            global_shape = tuple([basis.meta.size for basis in domain_obj.bases]...)
        else
            # Fallback to grid data shape
            if operator.data_g === nothing
                ensure_layout!(operator, :g)
            end
            global_shape = size(operator.data_g)
        end

        return Dict(
            "dims" => length(global_shape),
            "shape" => global_shape,
            "domain" => domain_obj,
            "dist" => dist_obj
        )
    elseif isa(operator, VectorField)
        first_comp = operator.components[1]
        domain_obj = first_comp.domain
        dist_obj = domain_obj !== nothing ? domain_obj.dist : nothing

        if domain_obj !== nothing
            global_shape = tuple([basis.meta.size for basis in domain_obj.bases]...)
        else
            if first_comp.data_g === nothing
                ensure_layout!(first_comp, :g)
            end
            global_shape = size(first_comp.data_g)
        end

        return Dict(
            "dims" => length(global_shape),
            "shape" => global_shape,
            "domain" => domain_obj,
            "dist" => dist_obj
        )
    elseif isa(operator, TensorField)
        first_comp = operator.components[1, 1]
        domain_obj = first_comp.domain
        dist_obj = domain_obj !== nothing ? domain_obj.dist : nothing

        if domain_obj !== nothing
            global_shape = tuple([basis.meta.size for basis in domain_obj.bases]...)
        else
            if first_comp.data_g === nothing
                ensure_layout!(first_comp, :g)
            end
            global_shape = size(first_comp.data_g)
        end

        return Dict(
            "dims" => length(global_shape),
            "shape" => global_shape,
            "domain" => domain_obj,
            "dist" => dist_obj
        )
    end
    return Dict("dims" => 0, "shape" => (), "domain" => nothing, "dist" => nothing)
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
    get_global_shape(layout, domain_info, scales)

Compute the global shape for data output, applying dealiasing scales.

# Arguments
- `layout`: Layout specification (symbol like :g/:c, or Layout object)
- `domain_info`: Dict containing domain information from get_operator_domain
- `scales`: Tuple of scale factors (e.g., (1.0, 1.0, 1.0) or (2/3, 2/3, 2/3))

# Returns
- Tuple of global dimensions after applying scales

The scales parameter allows for dealiased output - scale < 1.0 reduces
the resolution for each dimension (typically 2/3 for standard dealiasing).
"""
function get_global_shape(layout, domain_info, scales)
    base_shape = domain_info["shape"]

    if isempty(base_shape)
        return ()
    end

    # Apply scales to get output shape
    # Scales can be:
    # - A tuple of floats (e.g., (1.0, 1.0, 1.0) for full resolution)
    # - A single float to apply uniformly
    # - An integer (treated as scale of 1.0)

    if scales === nothing || scales == 1 || scales == 1.0
        return base_shape
    end

    if isa(scales, Number)
        # Single scale value applied to all dimensions
        return tuple([max(1, round(Int, s * scales)) for s in base_shape]...)
    elseif isa(scales, Tuple) || isa(scales, AbstractVector)
        # Per-dimension scales
        n_scales = length(scales)
        n_dims = length(base_shape)

        scaled_shape = Vector{Int}(undef, n_dims)
        for i in 1:n_dims
            scale_i = i <= n_scales ? scales[i] : 1.0
            scaled_shape[i] = max(1, round(Int, base_shape[i] * scale_i))
        end
        return tuple(scaled_shape...)
    end

    return base_shape
end

"""
    get_local_shape(layout, domain_info, scales, rank)

Compute the local shape owned by a specific MPI rank after mesh decomposition.

# Arguments
- `layout`: Layout specification (symbol like :g/:c, or Layout object)
- `domain_info`: Dict containing domain information from get_operator_domain
- `scales`: Tuple of scale factors for dealiasing
- `rank`: MPI rank (0-indexed)

# Returns
- Tuple of local dimensions for this rank

The local shape depends on:
1. The global shape (after scaling)
2. The MPI mesh decomposition
3. This rank's position in the mesh

For dimensions that are decomposed, the local size is computed by
dividing the global size among processes with remainder distribution.
"""
function get_local_shape(layout, domain_info, scales, rank)
    # First get the scaled global shape
    global_shape = get_global_shape(layout, domain_info, scales)

    if isempty(global_shape)
        return ()
    end

    # Get distributor for mesh decomposition info
    dist = domain_info["dist"]

    if dist === nothing || dist.size == 1 || dist.mesh === nothing
        # Serial execution - local shape equals global shape
        return global_shape
    end

    # Parallel execution - compute local shape based on mesh decomposition
    mesh = dist.mesh
    n_dims = length(global_shape)
    n_mesh_dims = length(mesh)

    local_dims = Vector{Int}(undef, n_dims)

    # Decomposition applies to the last n_mesh_dims dimensions
    # (matching PencilArrays convention)
    for i in 1:n_dims
        mesh_dim_idx = i - (n_dims - n_mesh_dims)

        if mesh_dim_idx >= 1 && mesh_dim_idx <= n_mesh_dims
            # This dimension is decomposed
            n_procs = mesh[mesh_dim_idx]
            proc_coord = get_process_coordinate_for_rank(dist, mesh_dim_idx, rank)
            global_size = global_shape[i]

            # Standard load-balanced decomposition
            base_size = div(global_size, n_procs)
            remainder = global_size % n_procs

            if proc_coord < remainder
                local_dims[i] = base_size + 1
            else
                local_dims[i] = base_size
            end
        else
            # This dimension is not decomposed
            local_dims[i] = global_shape[i]
        end
    end

    return tuple(local_dims...)
end

"""
    get_process_coordinate_for_rank(dist, mesh_dim, rank)

Get the process coordinate in a specific mesh dimension for a given rank.
Uses row-major ordering consistent with PencilArrays.
"""
function get_process_coordinate_for_rank(dist, mesh_dim::Int, rank::Int)
    if dist.mesh === nothing
        return 0
    end

    mesh = dist.mesh
    n_mesh_dims = length(mesh)

    # Compute coordinate using row-major ordering
    remaining_rank = rank
    for dim in 1:n_mesh_dims
        stride = prod(mesh[dim+1:end]; init=1)
        coord = div(remaining_rank, stride)
        if dim == mesh_dim
            return coord
        end
        remaining_rank = remaining_rank % stride
    end

    return 0
end

"""
    get_local_start(layout, domain_info, scales, rank)

Compute the global starting indices for a specific MPI rank's local data.

# Arguments
- `layout`: Layout specification (symbol like :g/:c, or Layout object)
- `domain_info`: Dict containing domain information from get_operator_domain
- `scales`: Tuple of scale factors for dealiasing
- `rank`: MPI rank (0-indexed)

# Returns
- Tuple of 0-indexed starting indices for each dimension

These indices represent where this rank's local data begins in the global
array, which is essential for parallel NetCDF writes where each process
writes to a different region of the global dataset.
"""
function get_local_start(layout, domain_info, scales, rank)
    # First get the scaled global shape
    global_shape = get_global_shape(layout, domain_info, scales)

    if isempty(global_shape)
        return ()
    end

    n_dims = length(global_shape)

    # Get distributor for mesh decomposition info
    dist = domain_info["dist"]

    if dist === nothing || dist.size == 1 || dist.mesh === nothing
        # Serial execution - starts at origin
        return ntuple(_ -> 0, n_dims)
    end

    # Parallel execution - compute start indices based on mesh decomposition
    mesh = dist.mesh
    n_mesh_dims = length(mesh)

    start_indices = Vector{Int}(undef, n_dims)

    # Decomposition applies to the last n_mesh_dims dimensions
    for i in 1:n_dims
        mesh_dim_idx = i - (n_dims - n_mesh_dims)

        if mesh_dim_idx >= 1 && mesh_dim_idx <= n_mesh_dims
            # This dimension is decomposed - compute start index
            n_procs = mesh[mesh_dim_idx]
            proc_coord = get_process_coordinate_for_rank(dist, mesh_dim_idx, rank)
            global_size = global_shape[i]

            # Standard load-balanced decomposition
            base_size = div(global_size, n_procs)
            remainder = global_size % n_procs

            # Compute start index (0-indexed for NetCDF)
            if proc_coord < remainder
                # Processes with coord < remainder get (base_size + 1) elements each
                start_indices[i] = proc_coord * (base_size + 1)
            else
                # Processes with coord >= remainder get base_size elements each
                # First 'remainder' processes take (base_size + 1) * remainder elements
                start_indices[i] = remainder * (base_size + 1) + (proc_coord - remainder) * base_size
            end
        else
            # This dimension is not decomposed - starts at 0
            start_indices[i] = 0
        end
    end

    return tuple(start_indices...)
end

"""
Check if handler should process based on schedule (matching Tarang logic)
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
        # Simplified logic - in real Tarang this is more complex
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
Create NetCDF file with Tarang-style structure
"""
function create_current_file!(handler::NetCDFFileHandler)
    filename = current_file(handler)

    # Create file directory if needed
    dir = dirname(filename)
    if !isdir(dir)
        mkpath(dir)
    end

    # Remove existing file to avoid dimension conflicts
    if isfile(filename)
        rm(filename)
    end

    # Create basic structure matching Tarang HDF5 layout
    # Use a single nccreate call pattern that works with NetCDF.jl
    # All variables share the same unlimited "sim_time" dimension

    # Create sim_time variable (defines the dimension)
    nccreate(filename, "sim_time", "sim_time", Inf,
            atts=Dict("long_name" => "simulation time",
                     "units" => "dimensionless time",
                     "axis" => "T"))

    # For other variables, we need to specify the dimension size as Inf again
    # NetCDF.jl will recognize it's the same unlimited dimension
    nccreate(filename, "wall_time", "sim_time", Inf,
            atts=Dict("long_name" => "wall clock time",
                     "units" => "seconds"))

    nccreate(filename, "timestep", "sim_time", Inf,
            atts=Dict("long_name" => "timestep",
                     "units" => "dimensionless time"))

    nccreate(filename, "iteration", "sim_time", Inf, t=Int64,
            atts=Dict("long_name" => "iteration number"))

    nccreate(filename, "write_number", "sim_time", Inf, t=Int64,
            atts=Dict("long_name" => "write number"))
    
    # Add global attributes (matching Tarang metadata)
    # Note: NetCDF.jl doesn't support Bool attributes, so we use Int (0/1)
    global_attrs = Dict(
        "set_number" => handler.set_num,
        "handler_name" => handler.name,
        "writes" => handler.file_write_num,
        "title" => "Tarang.jl simulation output",
        "institution" => "Generated by Tarang.jl",
        "source" => "Tarang.jl - Julia implementation of Tarang",
        "history" => "Created on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))",
        "Conventions" => "CF-1.8",
        "tarang_version" => "0.1.0",
        "software" => "Tarang",
        "software_repository" => "https://github.com/subhk/Tarang.jl",
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
Process handler: write all tasks to NetCDF (matching Tarang process method)
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
    # Ensure plain array for NetCDF - handle scalars by wrapping in 1-element array
    if isa(data, Number)
        data = [data]
    elseif isa(data, AbstractArray)
        data = Array(data)
    else
        data = [data]  # Fallback for other types
    end
    nc_type = eltype(data) <: Float32 ? NC_FLOAT : NC_DOUBLE
    
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
                # Handle single-element case where range() fails with different endpoints
                coord_data = coord_length == 1 ? [0.0] : collect(range(0.0, 1.0, length=coord_length))
                nccreate(filename, dim_name, dim_name, coord_length,
                        atts=Dict("long_name" => dim_name,
                                 "axis" => "XYZ"[min(i,3):min(i,3)]))
                ncwrite(coord_data, filename, dim_name)
            end
        end
        
        # Create main data variable using NetCDF.jl syntax
        # Build dimension arguments for nccreate: dim1, size1, dim2, size2, ...
        # Use Inf for unlimited time dimension, actual sizes for spatial dims

        # Note: NetCDF.jl doesn't support Bool or Symbol attributes, so we convert
        var_atts = Dict(
            "long_name" => task_name,
            "standard_name" => task_name,
            "_FillValue" => handler.precision(NaN),
            "grid_space" => 1,  # 1 = true (NetCDF doesn't support Bool)
            "layout" => string(task["layout"])  # Convert Symbol to String
        )
        if task["scales"] !== nothing
            var_atts["scales"] = string(task["scales"])
        end

        # Build the nccreate call with alternating dim names and sizes
        # nccreate(filename, varname, dim1, size1, dim2, size2, ...; kwargs...)
        dim_args = Any[]
        push!(dim_args, "sim_time")
        push!(dim_args, Inf)  # Unlimited time dimension
        for (i, dim_name) in enumerate(dim_names[2:end])
            push!(dim_args, dim_name)
            push!(dim_args, data_shape[i])
        end

        nccreate(filename, task_name, dim_args..., t=nc_type, atts=var_atts)
    end
    
    # Write data with time index
    # Reshape data to include time dimension: (x,) -> (1, x) for writing at time index
    data_with_time = reshape(data, 1, size(data)...)
    start_indices = [write_index; ones(Int, length(size(data)))]
    ncwrite(data_with_time, filename, task_name, start=start_indices)

    return true
end

"""
Convenience function to create NetCDF file handler (matching Tarang API)
Usage: handler = add_file_handler("snapshots", dist, vars, sim_dt=0.25, max_writes=50)
"""
function add_netcdf_handler(base_path::String, dist, vars; kwargs...)
    return NetCDFFileHandler(base_path, dist, vars; kwargs...)
end

# Export main functionality matching Tarang interface
export NetCDFFileHandler, add_netcdf_handler

"""
Tarang-style helper to create a NetCDF file handler.
Matches evaluator.add_file_handler(...) usage in Tarang.
"""
function add_file_handler(base_path::String, dist, vars; kwargs...)
    return NetCDFFileHandler(base_path, dist, vars; kwargs...)
end

"""
Alias without bang for Tarang-style task addition.
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
function add_slice_task!(handler::NetCDFFileHandler, field; slices=Dict(), dim=nothing, idx=nothing, name=nothing)
    # Support both slices dict and dim/idx syntax
    if dim !== nothing && idx !== nothing
        # Use dim/idx syntax - create a slice at index idx along dimension dim
        if name === nothing
            field_name = get_field_name(field)
            name = "$(field_name)_slice_dim$(dim)_idx$(idx)"
        end
        # Create postprocess function that extracts slice at given index
        postprocess = data -> begin
            indices = ntuple(i -> i == dim ? idx : Colon(), ndims(data))
            data[indices...]
        end
    else
        # Use slices dict syntax
        if name === nothing
            field_name = get_field_name(field)
            slice_parts = ["$(k)=$(v)" for (k, v) in pairs(slices)]
            slice_str = join(slice_parts, "_")
            name = isempty(slice_str) ? field_name : "$(field_name)_$(slice_str)"
        end
        # Create postprocess function for slicing
        postprocess = data -> apply_field_slices(data, field, slices)
    end

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
