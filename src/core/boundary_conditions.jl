"""
Boundary condition entrypoint and runtime refresh for Tarang.jl.

The parsed BC representations, manager state, and cache storage live in
`boundary_conditions/types.jl`. Constructor and manager setup helpers live in
`boundary_conditions/construction.jl`. This file keeps cache helpers, equation
conversion, dynamic value evaluation, and refresh of `problem.equation_data`.

Read this file alongside `solver_stepping.jl` and `step_subproblem_rk.jl`:
the solver refreshes dynamic BCs once per step, while the subproblem RK path
refreshes them again at each stage time to preserve stage-order accuracy.
"""

# LinearAlgebra, SparseArrays already in Tarang.jl

# GPU support: This module works with both CPU and GPU arrays.
# - Expression evaluation uses broadcast() which handles GPU arrays automatically
# - Cached BC values may be AbstractArray on either CPU or GPU
# - When coordinate arrays (x, y, z grids) are on GPU, results are on GPU

include("boundary_conditions/types.jl")
include("boundary_conditions/construction.jl")

@inline _bc_time_key(bc_index::Int, current_time::Real) = (bc_index, Float64(current_time))
@inline _bc_spatial_key(bc_index::Int, current_time::Real, coordinates::Dict) =
    (bc_index, Float64(current_time), hash((coordinates, Float64(current_time))))

function _clear_time_bc_cache!(cache::BCCacheStore)
    empty!(cache.time_values)
    empty!(cache.time_robin_values)
    return cache
end

function _clear_spatial_bc_cache!(cache::BCCacheStore)
    empty!(cache.spatial_values)
    empty!(cache.spatial_robin_values)
    return cache
end

function _store_time_bc_value!(cache::BCCacheStore, bc_index::Int, current_time::Real, value)
    cache.time_values[_bc_time_key(bc_index, current_time)] = value
    return value
end

function _store_time_bc_value!(cache::BCCacheStore, bc_index::Int, current_time::Real,
                               alpha, beta, value)
    robin = BCRobinCacheValue(alpha, beta, value)
    cache.time_robin_values[_bc_time_key(bc_index, current_time)] = robin
    return robin
end

function _store_spatial_bc_value!(cache::BCCacheStore, bc_index::Int, current_time::Real,
                                  coordinates::Dict, value)
    cache.spatial_values[_bc_spatial_key(bc_index, current_time, coordinates)] = value
    return value
end

function _store_spatial_bc_value!(cache::BCCacheStore, bc_index::Int, current_time::Real,
                                  coordinates::Dict, alpha, beta, value)
    robin = BCRobinCacheValue(alpha, beta, value)
    cache.spatial_robin_values[_bc_spatial_key(bc_index, current_time, coordinates)] = robin
    return robin
end

function _get_time_bc_value(cache::BCCacheStore, bc::AbstractBoundaryCondition,
                            bc_index::Int, current_time::Real)
    key = _bc_time_key(bc_index, current_time)
    if isa(bc, RobinBC)
        robin = get(cache.time_robin_values, key, nothing)
        robin === nothing && return nothing
        return (robin.alpha, robin.beta, robin.value)
    end
    return get(cache.time_values, key, nothing)
end

function _get_spatial_bc_value(cache::BCCacheStore, bc::AbstractBoundaryCondition,
                               bc_index::Int, current_time::Real, coordinates::Dict)
    key = _bc_spatial_key(bc_index, current_time, coordinates)
    if isa(bc, RobinBC)
        robin = get(cache.spatial_robin_values, key, nothing)
        robin === nothing && return nothing
        return (robin.alpha, robin.beta, robin.value)
    end
    return get(cache.spatial_values, key, nothing)
end

# BC to equation conversion
function _bc_value_to_string(value)
    if isa(value, String)
        return value
    elseif isa(value, FieldReference)
        if isa(value.expression, String)
            return value.expression
        elseif isa(value.expression, Function)
            return _bc_value_to_string(value.expression)
        else
            return value.name
        end
    elseif isa(value, Function)
        return string(Base.nameof(value))
    else
        return string(value)
    end
end

function _bc_derivative_str(field::String, coordinate::String, order::Int)
    if order == 1
        return "d($field, $coordinate)"
    end
    return "d($field, $coordinate, $order)"
end

"""Convert Dirichlet BC to equation string"""
function bc_to_equation(manager::BoundaryConditionManager, bc::DirichletBC)
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    val_str = _bc_value_to_string(bc.value)
    
    equation = "$(bc.field)($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        # Include tau term in the equation
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

"""Convert Neumann BC to equation string"""
function bc_to_equation(manager::BoundaryConditionManager, bc::NeumannBC)
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    val_str = _bc_value_to_string(bc.value)
    
    # Create derivative notation
    deriv = _bc_derivative_str(bc.field, bc.coordinate, bc.derivative_order)
    
    equation = "$deriv($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

"""Convert Robin BC to equation string"""
function bc_to_equation(manager::BoundaryConditionManager, bc::RobinBC)
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    alpha_str = _bc_value_to_string(bc.alpha)
    beta_str = _bc_value_to_string(bc.beta)
    val_str = _bc_value_to_string(bc.value)
    
    deriv = _bc_derivative_str(bc.field, bc.coordinate, 1)
    equation = "($(alpha_str))*$(bc.field)($(bc.coordinate)=$pos_str) + " *
               "($(beta_str))*$(deriv)($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

"""Convert stress-free BC to equations"""
function bc_to_equation(manager::BoundaryConditionManager, bc::StressFreeBC)
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    
    # Stress-free: u = 0 and du/dz = 0 at boundary (vanishing tangential stress)
    equations = [
        "$(bc.velocity_field)($(bc.coordinate)=$pos_str) = 0",
        "$( _bc_derivative_str(bc.velocity_field, bc.coordinate, 1) )($(bc.coordinate)=$pos_str) = 0"
    ]
    
    return equations
end

"""Convert custom BC to equation string"""
function bc_to_equation(manager::BoundaryConditionManager, bc::CustomBC)
    return bc.expression
end

# Problem integration
"""Apply all boundary conditions to problem"""
function apply_boundary_conditions!(manager::BoundaryConditionManager, problem)
    
    equations_added = String[]
    
    for bc in manager.conditions
        if isa(bc, PeriodicBC)
            # Periodic BCs are enforced by the Fourier basis (inherently periodic
            # spectral representation) — no equation needed.
            continue
        elseif isa(bc, StressFreeBC)
            # Stress-free BCs generate multiple equations
            eqs = bc_to_equation(manager, bc)
            for eq in eqs
                add_equation!(problem, eq)
                push!(equations_added, eq)
            end
        else
            # Other BCs generate single equation
            eq = bc_to_equation(manager, bc)
            add_equation!(problem, eq)
            push!(equations_added, eq)
        end
    end
    
    return equations_added
end

"""Validate boundary conditions for consistency and completeness"""
function validate_boundary_conditions(manager::BoundaryConditionManager, problem)
    
    warnings = String[]
    errors = String[]
    
    # Check for required tau fields
    for bc in manager.conditions
        tau_needed = false
        tau_field = nothing
        
        if isa(bc, DirichletBC)
            tau_needed = true
            tau_field = bc.tau_field
        elseif isa(bc, NeumannBC)
            tau_needed = true
            tau_field = bc.tau_field
        elseif isa(bc, RobinBC)
            tau_needed = true
            tau_field = bc.tau_field
        end
        
        if tau_needed && tau_field === nothing
            # This is informational - tau fields are optional for simple problems
            @debug "Boundary condition for $(bc.field) does not have tau field (optional for simple problems)"
        elseif tau_needed && tau_field !== nothing && get_tau_field(manager, tau_field) === nothing
            push!(errors, "Tau field $tau_field not registered")
        end
    end

    # Check for coordinate consistency
    field_coords = Dict{String, Set{String}}()
    for bc in manager.conditions
        field_name = ""
        coord = ""
        
        if isa(bc, DirichletBC)
            field_name = bc.field
            coord = bc.coordinate
        elseif isa(bc, NeumannBC)
            field_name = bc.field
            coord = bc.coordinate
        elseif isa(bc, RobinBC)
            field_name = bc.field
            coord = bc.coordinate
        elseif isa(bc, StressFreeBC)
            field_name = bc.velocity_field
            coord = bc.coordinate
        end
        
        if field_name != "" && coord != ""
            if !haskey(field_coords, field_name)
                field_coords[field_name] = Set{String}()
            end
            push!(field_coords[field_name], coord)
        end
    end
    
    # Report validation results
    if !isempty(warnings)
        @warn "Boundary condition warnings:\n" * join(warnings, "\n")
    end
    
    if !isempty(errors)
        throw(ArgumentError("Boundary condition errors:\n" * join(errors, "\n")))
    end
    
    return true
end

# Utility functions
"""Get count of boundary conditions by type"""
function get_bc_count_by_type(manager::BoundaryConditionManager)
    counts = Dict{String, Int}()
    
    for bc in manager.conditions
        bc_type = string(typeof(bc))
        counts[bc_type] = get(counts, bc_type, 0) + 1
    end
    
    return counts
end

"""Get list of required tau field names"""
function get_required_tau_fields(manager::BoundaryConditionManager)
    tau_fields = String[]
    
    for bc in manager.conditions
        if isa(bc, DirichletBC) && bc.tau_field !== nothing
            push!(tau_fields, bc.tau_field)
        elseif isa(bc, NeumannBC) && bc.tau_field !== nothing
            push!(tau_fields, bc.tau_field)
        elseif isa(bc, RobinBC) && bc.tau_field !== nothing
            push!(tau_fields, bc.tau_field)
        elseif isa(bc, StressFreeBC)
            append!(tau_fields, bc.tau_fields)
        elseif isa(bc, CustomBC)
            append!(tau_fields, bc.tau_fields)
        end
    end
    
    return unique(tau_fields)
end

# Time and space dependency management
"""Set the time variable for time-dependent boundary conditions"""
function set_time_variable!(manager::BoundaryConditionManager, time_var::String, time_field=nothing)
    manager.time_variable = time_var
    if time_field !== nothing
        manager.coordinate_fields[time_var] = time_field
    end
    return manager
end

"""Add spatial coordinate field for space-dependent boundary conditions"""
function add_coordinate_field!(manager::BoundaryConditionManager, coord_name::String, field)
    manager.coordinate_fields[coord_name] = field
    return manager
end

"""Evaluate boundary condition value at current time and spatial coordinates"""
function evaluate_bc_value(manager::BoundaryConditionManager, bc, current_time=0.0, coords=Dict())

    if isa(bc, DirichletBC)
        value = bc.value
    elseif isa(bc, NeumannBC)
        value = bc.value
    elseif isa(bc, RobinBC)
        # For Robin BCs, evaluate all components
        alpha = evaluate_expression(bc.alpha, current_time, coords)
        beta = evaluate_expression(bc.beta, current_time, coords)
        val = evaluate_expression(bc.value, current_time, coords)
        
        # Update performance statistics
        manager.performance_stats.total_evaluations += 3
        
        return (alpha, beta, val)
    else
        return nothing
    end
    
    result = evaluate_expression(value, current_time, coords)
    
    # Update performance statistics
    manager.performance_stats.total_evaluations += 1
    
    return result
end

"""
    evaluate_expression(expr, current_time=0.0, coords=Dict())

Evaluate a boundary condition expression with current time and coordinates.

Supports:
- Numeric values (returned as-is)
- String expressions with mathematical functions and variables
- Function objects
- TimeDependentValue and TimeSpaceDependentValue wrappers

String expressions can contain:
- Variables: t (time), x, y, z, r, θ, φ (coordinates)
- Constants: pi, π, e
- Functions: sin, cos, tan, exp, log, sqrt, abs, sinh, cosh, tanh
- Operators: +, -, *, /, ^, ()

Examples:
- "sin(2*pi*t)" - time-dependent sinusoid
- "exp(-t)*cos(x)" - exponentially decaying spatial cosine
- "x^2 + y^2" - spatial quadratic
- "sin(t)*sin(pi*x)*sin(pi*y)" - product of temporal and spatial modes
"""
function evaluate_expression(expr, current_time=0.0, coords=Dict())
    if isa(expr, Real)
        return expr
    elseif isa(expr, String)
        return _evaluate_string_expression(expr, current_time, coords)
    elseif isa(expr, Function)
        return _evaluate_function_expression(expr, current_time, coords)
    elseif isa(expr, FieldReference)
        return evaluate_expression(expr.expression, current_time, coords)
    elseif isa(expr, TimeDependentValue) || isa(expr, TimeSpaceDependentValue)
        if expr.function_obj !== nothing
            return _evaluate_function_expression(expr.function_obj, current_time, coords)
        else
            return evaluate_expression(expr.expression, current_time, coords)
        end
    elseif isa(expr, SpaceDependentValue)
        if expr.function_obj !== nothing
            return _evaluate_space_function_expression(expr.function_obj, coords)
        else
            return evaluate_expression(expr.expression, current_time, coords)
        end
    end

    return expr
end

"""
Evaluate a string expression by substituting variables and parsing.
Uses Julia's Meta.parse for safe expression evaluation.
"""
function _evaluate_string_expression(expr::String, current_time, coords)
    # Handle simple constant cases first
    stripped = strip(expr)
    if stripped == "0" || stripped == "0.0"
        return 0.0
    elseif stripped == "1" || stripped == "1.0"
        return 1.0
    elseif stripped == "-1" || stripped == "-1.0"
        return -1.0
    end

    # Try to parse as a simple number
    try
        return parse(Float64, stripped)
    catch
        # Continue with expression parsing
    end

    # Build variable substitution dictionary
    vars = Dict{String, Any}()
    vars["t"] = Float64(current_time)
    vars["pi"] = Float64(π)
    vars["π"] = Float64(π)
    vars["e"] = Float64(ℯ)

    # Add coordinate values
    for (key, val) in coords
        if isa(val, Real)
            vars[string(key)] = Float64(val)
        elseif isa(val, AbstractArray)
            # For array-valued coordinates, we'll need special handling
            vars[string(key)] = val
        end
    end

    # Check if all required variables are present
    # Build the expression with variable substitution
    try
        result = _safe_eval_math_expr(expr, vars)
        return result
    catch e
        @warn "BC expression evaluation failed for '$expr': $e" maxlog=5
        return expr
    end
end

"""
Safely evaluate a mathematical expression string with variable substitutions.
Only allows mathematical operations - no arbitrary code execution.
"""
function _safe_eval_math_expr(expr_str::String, vars::Dict{String, T}) where T
    # Allowed mathematical functions
    allowed_funcs = Dict{String, Function}(
        "sin" => sin,
        "cos" => cos,
        "tan" => tan,
        "sinh" => sinh,
        "cosh" => cosh,
        "tanh" => tanh,
        "asin" => asin,
        "acos" => acos,
        "atan" => atan,
        "exp" => exp,
        "log" => log,
        "log10" => log10,
        "log2" => log2,
        "sqrt" => sqrt,
        "abs" => abs,
        "sign" => sign,
        "floor" => floor,
        "ceil" => ceil,
        "round" => round,
        "rem" => rem,  # Remainder function (same as % operator)
        "min" => min,
        "max" => max,
    )

    # Parse the expression
    parsed = Meta.parse(expr_str)

    # Evaluate with restrictions
    return _eval_safe_ast(parsed, vars, allowed_funcs)
end

"""
Recursively evaluate an AST node with safety restrictions.
Only allows arithmetic operations and whitelisted functions.
"""
function _eval_safe_ast(node, vars::Dict{String, T}, allowed_funcs::Dict{String, Function}) where T
    if isa(node, Number)
        return Float64(node)
    elseif isa(node, Symbol)
        name = string(node)
        if haskey(vars, name)
            return vars[name]
        elseif name == "pi" || name == "π"
            return Float64(π)
        elseif name == "e" || name == "ℯ"
            return Float64(ℯ)
        else
            throw(ArgumentError("Unknown variable: $name"))
        end
    elseif isa(node, Expr)
        if node.head == :call
            func_name = string(node.args[1])
            args = [_eval_safe_ast(arg, vars, allowed_funcs) for arg in node.args[2:end]]

            # Check if it's an allowed function
            if haskey(allowed_funcs, func_name)
                func = allowed_funcs[func_name]
                return _apply_safe_function(func, args)
            # Handle binary operators
            elseif func_name == "+"
                return _apply_safe_operator(+, args)
            elseif func_name == "-"
                if length(args) == 1
                    return _apply_safe_operator(-, args)
                else
                    return _apply_safe_operator(-, args)
                end
            elseif func_name == "*"
                return _apply_safe_operator(*, args)
            elseif func_name == "/"
                return _apply_safe_operator(/, args)
            elseif func_name == "^"
                return _apply_safe_operator(^, args)
            elseif func_name == "%"
                # Julia's % is rem (remainder), not mod
                return _apply_safe_operator(rem, args)
            elseif func_name == "mod"
                return _apply_safe_operator(mod, args)
            else
                throw(ArgumentError("Function not allowed: $func_name"))
            end
        elseif node.head == :block
            # Handle block expressions (multiple statements)
            result = nothing
            for child in node.args
                if !isa(child, LineNumberNode)
                    result = _eval_safe_ast(child, vars, allowed_funcs)
                end
            end
            return result
        elseif node.head == :(=)
            throw(ArgumentError("Assignment not allowed in expressions"))
        else
            throw(ArgumentError("Expression type not allowed: $(node.head)"))
        end
    elseif isa(node, LineNumberNode)
        return nothing
    else
        throw(ArgumentError("Unsupported node type: $(typeof(node))"))
    end
end

function _apply_safe_function(func::Function, args::AbstractVector)
    if any(arg -> arg isa AbstractArray, args)
        return broadcast(func, args...)
    end
    return func(args...)
end

function _apply_safe_operator(op::Function, args::AbstractVector)
    if any(arg -> arg isa AbstractArray, args)
        return broadcast(op, args...)
    end
    return op(args...)
end

"""
Evaluate a function expression with appropriate arguments.
"""
function _coord_value(coords, name::String)
    if haskey(coords, name)
        return coords[name]
    end
    sym = Symbol(name)
    return haskey(coords, sym) ? coords[sym] : nothing
end

function _evaluate_function_expression(func::Function, current_time, coords)
    # Try different argument combinations based on function arity
    try
        return func(current_time, coords)
    catch
        # Continue with component-wise arguments
    end

    x = _coord_value(coords, "x")
    y = _coord_value(coords, "y")
    z = _coord_value(coords, "z")
    r = _coord_value(coords, "r")
    theta = _coord_value(coords, "θ")
    if theta === nothing
        theta = _coord_value(coords, "theta")
    end
    phi = _coord_value(coords, "φ")
    if phi === nothing
        phi = _coord_value(coords, "phi")
    end

    try
        # Try with all available arguments
        if x !== nothing && y !== nothing && z !== nothing
            return func(current_time, x, y, z)
        elseif x !== nothing && y !== nothing
            return func(current_time, x, y)
        elseif r !== nothing && theta !== nothing && phi !== nothing
            return func(current_time, r, theta, phi)
        elseif r !== nothing && theta !== nothing
            return func(current_time, r, theta)
        elseif x !== nothing
            return func(current_time, x)
        else
            return func(current_time)
        end
    catch
        # If that fails, try just time
        try
            return func(current_time)
        catch
            # Last resort: try no arguments
            return func()
        end
    end
end

function _evaluate_space_function_expression(func::Function, coords)
    # Try different argument combinations based on function arity
    try
        return func(coords)
    catch
        # Continue with component-wise arguments
    end

    x = _coord_value(coords, "x")
    y = _coord_value(coords, "y")
    z = _coord_value(coords, "z")
    r = _coord_value(coords, "r")
    theta = _coord_value(coords, "θ")
    if theta === nothing
        theta = _coord_value(coords, "theta")
    end
    phi = _coord_value(coords, "φ")
    if phi === nothing
        phi = _coord_value(coords, "phi")
    end

    try
        if x !== nothing && y !== nothing && z !== nothing
            return func(x, y, z)
        elseif x !== nothing && y !== nothing
            return func(x, y)
        elseif r !== nothing && theta !== nothing && phi !== nothing
            return func(r, theta, phi)
        elseif r !== nothing && theta !== nothing
            return func(r, theta)
        elseif x !== nothing
            return func(x)
        elseif r !== nothing
            return func(r)
        elseif theta !== nothing
            return func(theta)
        elseif phi !== nothing
            return func(phi)
        else
            return func()
        end
    catch
        try
            return func()
        catch
            return func
        end
    end
end

"""Update all time-dependent boundary conditions for current time"""
function update_time_dependent_bcs!(manager::BoundaryConditionManager, current_time)
    
    if isempty(manager.time_dependent_bcs)
        return manager
    end
    
    start_time = time()
    @debug "Updating $(length(manager.time_dependent_bcs)) time-dependent BCs at t=$current_time"
    
    # Use coordinate fields for BCs that are both time and space dependent
    coords = manager.coordinate_fields

    # Clear stale cache entries from previous timesteps to prevent unbounded growth.
    # Each BC only needs its current-time value cached (for substep reuse within a step).
    if !isempty(manager.bc_cache.time_values) || !isempty(manager.bc_cache.time_robin_values)
        _clear_time_bc_cache!(manager.bc_cache)
        manager.performance_stats.cache_misses += 1
    end

    for bc_index in manager.time_dependent_bcs
        bc = manager.conditions[bc_index]

        # Tuple key avoids string allocation every timestep
        if isa(bc, DirichletBC) && bc.is_time_dependent
            if _get_time_bc_value(manager.bc_cache, bc, bc_index, current_time) === nothing
                new_value = evaluate_bc_value(manager, bc, current_time, coords)
                _store_time_bc_value!(manager.bc_cache, bc_index, current_time, new_value)
            end
            @debug "Updated Dirichlet BC for $(bc.field)"

        elseif isa(bc, NeumannBC) && bc.is_time_dependent
            if _get_time_bc_value(manager.bc_cache, bc, bc_index, current_time) === nothing
                new_value = evaluate_bc_value(manager, bc, current_time, coords)
                _store_time_bc_value!(manager.bc_cache, bc_index, current_time, new_value)
            end
            @debug "Updated Neumann BC for $(bc.field)"

        elseif isa(bc, RobinBC) && bc.is_time_dependent
            if _get_time_bc_value(manager.bc_cache, bc, bc_index, current_time) === nothing
                alpha, beta, value = evaluate_bc_value(manager, bc, current_time, coords)
                _store_time_bc_value!(manager.bc_cache, bc_index, current_time, alpha, beta, value)
            end
            @debug "Updated Robin BC for $(bc.field)"
        end
    end
    
    # Update performance statistics
    update_time = time() - start_time
    manager.performance_stats.total_time += update_time
    manager.performance_stats.bc_updates += 1
    
    manager.bc_update_required = false
    return manager
end

"""Retrieve the most recently cached value for a time-dependent BC."""
function get_current_bc_value(manager::BoundaryConditionManager, bc_index::Int, current_time)
    bc = manager.conditions[bc_index]
    return _get_time_bc_value(manager.bc_cache, bc, bc_index, current_time)
end

"""Check if boundary conditions need updating"""
function requires_bc_update(manager::BoundaryConditionManager)
    return manager.bc_update_required || !isempty(manager.time_dependent_bcs)
end

"""Check if manager has any time-dependent boundary conditions"""
function has_time_dependent_bcs(manager::BoundaryConditionManager)
    return !isempty(manager.time_dependent_bcs)
end

"""Check if manager has any space-dependent boundary conditions"""
function has_space_dependent_bcs(manager::BoundaryConditionManager)
    return !isempty(manager.space_dependent_bcs)
end

"""Clear all boundary conditions and associated caches"""
function clear_boundary_conditions!(manager::BoundaryConditionManager)
    empty!(manager.conditions)
    empty!(manager.tau_fields)
    empty!(manager.lift_operators)
    empty!(manager.time_dependent_bcs)
    empty!(manager.space_dependent_bcs)
    empty!(manager.bc_equation_indices)
    empty!(manager.bc_cache)
    empty!(manager.workspace)
    manager.bc_update_required = false
    return manager
end

"""Log boundary condition performance statistics"""
function log_bc_performance(manager::BoundaryConditionManager)

    stats = manager.performance_stats

    @info "BC performance:"
    @info "  Total evaluations: $(stats.total_evaluations)"
    @info "  BC updates: $(stats.bc_updates)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    if stats.total_evaluations > 0
        @info "  Average evaluation time: $(round(stats.total_time/stats.total_evaluations*1000, digits=3)) ms"
    end
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses ($(round(100*stats.cache_hits/max(stats.cache_hits+stats.cache_misses, 1), digits=1))% hit rate)"
end

function evaluate_space_dependent_bcs!(manager::BoundaryConditionManager, coordinates::Dict,
                                      current_time::Real=0.0)
    """Evaluate space-dependent boundary conditions at the given time and coordinates"""

    if isempty(manager.space_dependent_bcs)
        return manager
    end

    start_time = time()

    @debug "Evaluating $(length(manager.space_dependent_bcs)) space-dependent BCs at t=$current_time"

    for bc_index in manager.space_dependent_bcs
        bc = manager.conditions[bc_index]

        if _get_spatial_bc_value(manager.bc_cache, bc, bc_index, current_time, coordinates) === nothing
            if isa(bc, DirichletBC) && bc.is_space_dependent
                new_value = evaluate_bc_value(manager, bc, current_time, coordinates)
                _store_spatial_bc_value!(manager.bc_cache, bc_index, current_time, coordinates, new_value)
                manager.performance_stats.cache_misses += 1
            elseif isa(bc, NeumannBC) && bc.is_space_dependent
                new_value = evaluate_bc_value(manager, bc, current_time, coordinates)
                _store_spatial_bc_value!(manager.bc_cache, bc_index, current_time, coordinates, new_value)
                manager.performance_stats.cache_misses += 1
            elseif isa(bc, RobinBC) && bc.is_space_dependent
                alpha, beta, value = evaluate_bc_value(manager, bc, current_time, coordinates)
                _store_spatial_bc_value!(manager.bc_cache, bc_index, current_time, coordinates, alpha, beta, value)
                manager.performance_stats.cache_misses += 1
            end
        else
            manager.performance_stats.cache_hits += 1
        end
    end

    manager.performance_stats.total_time += time() - start_time

    return manager
end

"""Clear cache for boundary conditions"""
function clear_bc_cache!(manager::BoundaryConditionManager)
    empty!(manager.bc_cache)
    return manager
end

# Export all public functions
export AbstractBoundaryCondition, DirichletBC, NeumannBC, RobinBC, PeriodicBC,
       StressFreeBC, CustomBC, BoundaryConditionManager,
       dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
       add_bc!, add_dirichlet!, add_neumann!, add_robin!, add_periodic!,
       add_stress_free!, add_custom!,
       register_tau_field!, get_tau_field, auto_generate_tau_fields!, validate_tau_fields!,
       create_lift_operator, apply_lift, bc_to_equation, get_boundary_basis,
       apply_boundary_conditions!, validate_boundary_conditions,
       get_bc_count_by_type, get_required_tau_fields, clear_boundary_conditions!,
       TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
       set_time_variable!, add_coordinate_field!, evaluate_bc_value, evaluate_expression,
       update_time_dependent_bcs!, requires_bc_update, has_time_dependent_bcs, has_space_dependent_bcs,
       is_time_dependent, is_space_dependent, get_current_bc_value,
       register_coordinate_info!, register_domain_info!,
       log_bc_performance, evaluate_space_dependent_bcs!, clear_bc_cache!
