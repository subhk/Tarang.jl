
"""
    add_bc!(problem, bc::AbstractBoundaryCondition)

Add a structured boundary condition object to the problem.

See also: [`no_slip!`](@ref), [`fixed_value!`](@ref), [`free_slip!`](@ref),
[`insulating!`](@ref), [`dirichlet_bc`](@ref), [`neumann_bc`](@ref)
"""
function add_bc!(problem::Problem, bc::AbstractBoundaryCondition)
    add_bc!(problem.bc_manager, bc)

    # Also add to legacy string list for compatibility
    eq = bc_to_equation(problem.bc_manager, bc)
    if isa(eq, Vector)
        append!(problem.boundary_conditions, eq)
    else
        push!(problem.boundary_conditions, eq)
    end

    return bc
end

# Boundary conditions can also be passed to add_equation!() following convention:
#   add_equation!(problem, "u(z=0) = 0")  # Dirichlet BC
#   add_equation!(problem, "dz(u)(z=1) = 0")  # Neumann BC

"""Register tau field for boundary condition enforcement"""
function register_tau_field!(problem::Problem, name::String, field)
    register_tau_field!(problem.bc_manager, name, field)
    
    # Add to problem variables if not already present
    if !any(var -> (isa(var, ScalarField) && var.name == name), problem.variables)
        push!(problem.variables, field)
    end
    
    return problem
end

"""Set parameter value"""
function set_parameter!(problem::Problem, name::String, value)
    problem.parameters[name] = value
end

"""Get parameter value"""
function get_parameter(problem::Problem, name::String, default=nothing)
    return get(problem.parameters, name, default)
end

# Equation parsing following structure
"""
    Parse equation string into operator expressions following formulation requirements.
    
    Requires:
    - LHS: Linear terms only (first-order in time derivatives, linear in variables)
    - RHS: Nonlinear terms, time-dependent terms, non-constant coefficients
    - Form: M·∂tX + L·X = F(X,t)
    
    Following problems:add_equation pattern (problems:65-80).
    """
function parse_equation(equation::String, namespace::Dict{String, Any})
    
    try
        # Split equation into LHS and RHS expressions
        if isa(equation, String)
            # Parse string-valued equations following pattern
            LHS_str, RHS_str = split_equation(equation)
            
            # Parse LHS (should contain only linear terms)
            LHS = parse_linear_expression(LHS_str, namespace)
            
            # Parse RHS (can contain nonlinear terms)
            RHS = parse_expression(RHS_str, namespace)
            
            # Validate equation structure following requirements
            validate_equation_structure(LHS, RHS, equation)
            
            return LHS, RHS
        else
            throw(ArgumentError("Equation must be a string"))
        end
        
    catch e
        @error "Failed to parse equation: $equation" exception=e
        # Return fallback expressions as proper operator types
        return UnknownOperator(equation), ZeroOperator()
    end
end

"""
    Parse LHS expression ensuring it contains only linear terms.
    
    Linear terms allowed on LHS:
    - Time derivatives (first-order): dt(u)
    - Linear spatial derivatives: d(u, x), lap(u)
    - Constant coefficients: 2*dt(u), -viscosity*lap(u)
    - Linear combinations: dt(u) + diffusion*lap(u)
    
    Non-constant coefficients should be moved to RHS.
    """
function parse_linear_expression(expr_str::AbstractString, namespace::Dict{String, Any})

    expr = parse_expression(expr_str, namespace)
    
    return expr
end

# Note: has_variables() is now replaced by the proper has() method from field.jl/operators.jl
# following the spectral pattern for variable linearity checking

"""
    is_linear_expression(expr, namespace::Dict{String, Any}) -> Bool

Check if expression is linear in the problem variables.
Follows the spectral approach:

- Multiplication is nonlinear ONLY if BOTH factors contain problem variables
- Addition: all addends must be linear
- Linear operators (grad, div, lap, skew, etc.): delegate to operand
- Nonlinear operators (power, sqrt applied to variables): nonlinear

Key insight: `c * u` is linear (c is coefficient),
but `u * v` is nonlinear (both are variables).

Uses the proper has() and require_linearity() methods from field.jl/operators.jl
which follow the spectral pattern exactly.
"""
function is_linear_expression(expr, namespace::Dict{String, Any})
    # Extract problem variables from namespace
    variables = collect_problem_variables(namespace)

    # Use require_linearity() following spectral pattern
    # allow_affine=true because constant terms are allowed on LHS
    try
        require_linearity(expr, variables...; allow_affine=true)
        return true
    catch e
        @debug "Nonlinearity detected" exception=e expr=expr
        return false
    end
end

"""Collect all problem variables (fields) from namespace."""
function collect_problem_variables(namespace::Dict{String, Any})
    variables = []
    for (name, val) in namespace
        if isa(val, ScalarField) || isa(val, VectorField) || isa(val, TensorField)
            push!(variables, val)
        end
    end
    return variables
end

"""Check if expression is a direct reference to a problem variable."""
function is_variable_reference(expr, namespace::Dict{String, Any})
    if hasfield(typeof(expr), :name) && isa(getfield(expr, :name), String)
        var_name = getfield(expr, :name)
        if haskey(namespace, var_name)
            val = namespace[var_name]
            return isa(val, ScalarField) || isa(val, VectorField) || isa(val, TensorField)
        end
    end
    # Also check if it's directly a field
    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        return true
    end
    return false
end

"""
    Check if expression represents a constant coefficient.
    Non-constant coefficients should be moved to RHS per requirements.
    """
function is_constant_coefficient(expr, namespace::Dict{String, Any})
    
    if isa(expr, ConstantOperator)
        return true
    elseif isa(expr, UnknownOperator)
        # Check if it's a known constant parameter
        expr_str = expr.expression
        if haskey(namespace, expr_str)
            param = namespace[expr_str]
            # This would need proper type checking for constant parameters
            return isa(param, Number) || isa(param, AbstractFloat) || isa(param, Complex)
        end
    end
    
    return false
end

"""
    Validate that equation follows structure requirements.
    
    Requirements:
    1. LHS must be linear in dependent variables
    2. LHS must be first-order in time derivatives  
    3. RHS can contain nonlinear terms
    4. Non-constant coefficients should be on RHS
    """
function validate_equation_structure(LHS, RHS, original_equation::String)

    # Check for temporal derivatives on RHS (not allowed)
    if contains_time_derivatives(RHS)
        throw(ArgumentError("Time derivatives found on RHS of equation: \"$original_equation\". " *
                           "All time derivatives must be on LHS."))
    end

    # IMEX placement only matters for evolution equations (those with ∂t).
    # Diagnostic/constraint equations (e.g. "u - skew(grad(ψ)) = 0", "integ(ψ) = 0")
    # have no time splitting, so skip the check.
    if !contains_time_derivatives(LHS)
        return true
    end

    # Collect misplaced terms and suggest correction
    lhs_terms = _collect_addends(LHS)
    rhs_terms = _collect_addends(RHS)

    nonlinear_on_lhs = filter(t -> !_is_linear_term(t), lhs_terms)
    linear_on_rhs = filter(t -> _is_linear_term(t) && !_is_constant_term(t), rhs_terms)

    if !isempty(nonlinear_on_lhs) || !isempty(linear_on_rhs)
        # Build suggested equation
        new_lhs_terms = filter(t -> _is_linear_term(t), lhs_terms)
        # Move linear RHS terms to LHS (flip sign)
        for t in linear_on_rhs
            push!(new_lhs_terms, NegateOperator(t))
        end
        new_rhs_terms = filter(t -> !_is_linear_term(t) || _is_constant_term(t), rhs_terms)
        # Move nonlinear LHS terms to RHS (flip sign)
        for t in nonlinear_on_lhs
            push!(new_rhs_terms, NegateOperator(t))
        end

        suggested_lhs = _format_sum(new_lhs_terms)
        suggested_rhs = isempty(new_rhs_terms) ? "0" : _format_sum(new_rhs_terms)

        @warn "Equation has misplaced terms: \"$original_equation\"\n" *
              "  Suggested form: $suggested_lhs = $suggested_rhs"
    end

    return true
end

"""Collect top-level addends from an expression tree (split by +/-)."""
function _collect_addends(expr)
    if isa(expr, AddOperator)
        return vcat(_collect_addends(expr.left), _collect_addends(expr.right))
    elseif isa(expr, SubtractOperator)
        return vcat(_collect_addends(expr.left), Any[NegateOperator(t) for t in _collect_addends(expr.right)])
    else
        return Any[expr]
    end
end

"""Check if a term is linear (contains a differential operator or ∂t, but no variable products)."""
function _is_linear_term(expr)
    if isa(expr, TimeDerivative) || isa(expr, Laplacian) || isa(expr, Gradient) ||
       isa(expr, Divergence) || isa(expr, Differentiate) || isa(expr, Lift) ||
       isa(expr, FractionalLaplacian) || isa(expr, Curl) || isa(expr, Skew) ||
       isa(expr, Trace) || isa(expr, Integrate) || isa(expr, Average) ||
       isa(expr, Interpolate) || isa(expr, Convert) || isa(expr, HilbertTransform) ||
       isa(expr, Component) || isa(expr, RadialComponent) ||
       isa(expr, AngularComponent) || isa(expr, AzimuthalComponent)
        return true
    elseif isa(expr, NegateOperator)
        return _is_linear_term(expr.operand)
    elseif isa(expr, MultiplyOperator)
        # c * L(u) is linear if one side is a constant/parameter
        left_is_const = _is_constant_term(expr.left)
        right_is_const = _is_constant_term(expr.right)
        if left_is_const
            return _is_linear_term(expr.right)
        elseif right_is_const
            return _is_linear_term(expr.left)
        else
            return false  # variable * variable = nonlinear
        end
    elseif isa(expr, ScalarField) || isa(expr, VectorField)
        return true  # bare field reference is linear
    else
        return _is_constant_term(expr)
    end
end

"""Check if a term is a constant (number, parameter, ZeroOperator)."""
function _is_constant_term(expr)
    isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator) ||
    (isa(expr, NegateOperator) && _is_constant_term(expr.operand))
end

"""Format a list of addends as a readable sum string, handling signs cleanly."""
function _format_sum(terms)
    isempty(terms) && return "0"
    parts = String[]
    for (i, t) in enumerate(terms)
        s = if isa(t, NegateOperator)
            "- " * expression_to_string(t.operand)
        else
            (i == 1 ? "" : "+ ") * expression_to_string(t)
        end
        push!(parts, s)
    end
    return join(parts, " ")
end

"""Check if expression tree contains any time derivatives."""
function contains_time_derivatives(expr)
    if expr === nothing
        return false
    elseif isa(expr, TimeDerivative)
        return true
    elseif isa(expr, AddOperator) || isa(expr, SubtractOperator) || isa(expr, MultiplyOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, DivideOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, PowerOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, NegateOperator)
        return contains_time_derivatives(expr.operand)
    elseif isa(expr, IndexOperator)
        return contains_time_derivatives(expr.array)
    elseif hasfield(typeof(expr), :operand)
        # Handle operators like Laplacian, Differentiate, Gradient, etc.
        return contains_time_derivatives(expr.operand)
    else
        return false
    end
end


"""
    Check if LHS has proper structure for matrix formulation.
    Should be of the form: M·∂tX + L·X where M and L are linear operators.

    The LHS of a equation must satisfy:
    1. All terms must be linear in the state variables
    2. Time derivatives must be first-order only (∂t, not ∂t²)
    3. Spatial operators must be linear (derivatives, Laplacian, etc.)
    4. Coefficients must be constant (not space/time-dependent)
    5. No products of state variables (those go to RHS as nonlinear terms)

    Returns a tuple (is_valid::Bool, info::Dict) where info contains:
    - :has_time_derivative => whether the expression contains ∂t terms
    - :has_spatial_operators => whether spatial operators are present
    - :is_linear => whether all terms are linear
    - :error_message => description of any structural issues
    """
function is_proper_lhs_structure(expr)

    info = Dict{Symbol, Any}(
        :has_time_derivative => false,
        :has_spatial_operators => false,
        :is_linear => true,
        :error_message => nothing,
        :time_derivative_order => 0,
        :dependent_variables => Set{String}()
    )

    # Recursively analyze the expression structure
    is_valid = _analyze_lhs_structure!(expr, info, Dict{String, Any}())

    return (is_valid, info)
end

"""
    Recursively analyze expression structure for LHS validity.
    """
function _analyze_lhs_structure!(expr, info::Dict{Symbol, Any}, namespace::Dict{String, Any})

    # Handle nothing/empty case
    if expr === nothing
        return true
    end

    # Zero operator is always valid
    if isa(expr, ZeroOperator)
        return true
    end

    # Constant operator is valid
    if isa(expr, ConstantOperator)
        return true
    end

    # Numeric literals are valid coefficients
    if isa(expr, Number)
        return true
    end

    # Time derivative: check order and operand
    if isa(expr, TimeDerivative)
        info[:has_time_derivative] = true

        # Only first-order time derivatives allowed on LHS
        if expr.order > 1
            info[:is_linear] = false
            info[:error_message] = "Higher-order time derivatives (order=$(expr.order)) not allowed on LHS"
            return false
        end

        info[:time_derivative_order] = max(info[:time_derivative_order], expr.order)

        # The operand of time derivative should be a state variable or linear operator on it
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    # Spatial differential operators: linear by nature
    if isa(expr, Differentiate)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Laplacian)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Gradient)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Divergence)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Curl)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    # Addition/subtraction: both sides must be valid
    if isa(expr, AddOperator)
        left_valid = _analyze_lhs_structure!(expr.left, info, namespace)
        right_valid = _analyze_lhs_structure!(expr.right, info, namespace)
        return left_valid && right_valid
    end

    if isa(expr, SubtractOperator)
        left_valid = _analyze_lhs_structure!(expr.left, info, namespace)
        right_valid = _analyze_lhs_structure!(expr.right, info, namespace)
        return left_valid && right_valid
    end

    # Negation: operand must be valid
    if isa(expr, NegateOperator)
        return _analyze_lhs_structure!(expr.operand, info, namespace)
    end

    # Multiplication: exactly one factor must be a constant coefficient
    if isa(expr, MultiplyOperator)
        left_is_const = _is_constant_coefficient_strict(expr.left, namespace)
        right_is_const = _is_constant_coefficient_strict(expr.right, namespace)

        if left_is_const && right_is_const
            # Both constant - this is fine (just a constant)
            return true
        elseif left_is_const
            # Left is constant, right must be linear in variables
            return _analyze_lhs_structure!(expr.right, info, namespace)
        elseif right_is_const
            # Right is constant, left must be linear in variables
            return _analyze_lhs_structure!(expr.left, info, namespace)
        else
            # Neither is constant - this is a nonlinear term (product of variables)
            info[:is_linear] = false
            info[:error_message] = "Product of non-constant terms on LHS (nonlinear)"
            return false
        end
    end

    # Division: numerator must be linear, denominator must be constant
    if isa(expr, DivideOperator)
        if !_is_constant_coefficient_strict(expr.right, namespace)
            info[:is_linear] = false
            info[:error_message] = "Division by non-constant on LHS"
            return false
        end
        return _analyze_lhs_structure!(expr.left, info, namespace)
    end

    # Power operator: only valid if base is constant OR exponent is 1
    if isa(expr, PowerOperator)
        exp_val = _get_constant_value(expr.right)
        if exp_val !== nothing && exp_val == 1
            return _analyze_lhs_structure!(expr.left, info, namespace)
        elseif _is_constant_coefficient_strict(expr.left, namespace)
            return true
        else
            info[:is_linear] = false
            info[:error_message] = "Power of non-constant with exponent ≠ 1 on LHS"
            return false
        end
    end

    # ScalarField: this is a state variable - valid as linear term
    if isa(expr, ScalarField)
        push!(info[:dependent_variables], expr.name)
        return true
    end

    # VectorField: state variable
    if isa(expr, VectorField)
        push!(info[:dependent_variables], expr.name)
        return true
    end

    # Index operator on a valid structure
    if isa(expr, IndexOperator)
        return _analyze_lhs_structure!(expr.array, info, namespace)
    end

    # Unknown operator: check if it's in the namespace as a constant
    if isa(expr, UnknownOperator)
        if haskey(namespace, expr.expression)
            val = namespace[expr.expression]
            if isa(val, Number) || isa(val, ConstantOperator)
                return true
            elseif isa(val, ScalarField) || isa(val, VectorField)
                push!(info[:dependent_variables], expr.expression)
                return true
            end
        end
        # Unknown expression - be conservative and mark as potentially invalid
        @debug "Unknown expression in LHS: $(expr.expression)"
        return true  # Allow it but could be stricter
    end

    # Nonlinear operators are not allowed on LHS
    if isa(expr, NonlinearOperator)
        info[:is_linear] = false
        info[:error_message] = "Nonlinear operator on LHS"
        return false
    end

    # Default: if we can't classify it, check if it looks linear
    return is_linear_expression(expr, namespace)
end

"""
    Analyze the operand of a differential operator.
    """
function _analyze_lhs_operand!(operand, info::Dict{Symbol, Any}, namespace::Dict{String, Any})
    if isa(operand, ScalarField)
        push!(info[:dependent_variables], operand.name)
        return true
    elseif isa(operand, VectorField)
        push!(info[:dependent_variables], operand.name)
        return true
    else
        return _analyze_lhs_structure!(operand, info, namespace)
    end
end

"""
    Strictly check if expression is a constant coefficient.
    Constants are: numbers, ConstantOperator, or namespace entries that are constant.
    """
function _is_constant_coefficient_strict(expr, namespace::Dict{String, Any})
    if isa(expr, Number)
        return true
    end

    if isa(expr, ConstantOperator)
        return true
    end

    if isa(expr, ZeroOperator)
        return true
    end

    if isa(expr, UnknownOperator)
        if haskey(namespace, expr.expression)
            val = namespace[expr.expression]
            return isa(val, Number) || isa(val, ConstantOperator)
        end
        return false
    end

    # Arithmetic on constants
    if isa(expr, AddOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, SubtractOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, MultiplyOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, DivideOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, NegateOperator)
        return _is_constant_coefficient_strict(expr.operand, namespace)
    end

    if isa(expr, PowerOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    # Fields are not constants
    if isa(expr, ScalarField) || isa(expr, VectorField)
        return false
    end

    return false
end

"""
    Extract numeric value from a constant expression.
    Returns nothing if not a simple constant.
    """
function _get_constant_value(expr)
    if isa(expr, Number)
        return expr
    end

    if isa(expr, ConstantOperator)
        return expr.value
    end

    return nothing
end

"""
    Validate LHS structure and return a detailed report.
    Throws an error if the structure is invalid.
    """
function validate_lhs_structure(expr)
    is_valid, info = is_proper_lhs_structure(expr)

    if !is_valid
        error_msg = info[:error_message]
        if error_msg === nothing
            error_msg = "LHS structure is invalid for matrix formulation"
        end
        throw(ArgumentError("Invalid LHS structure: $error_msg"))
    end

    return info
end

"""
    Parse expression string into operator tree following patterns.
    Uses Julia's Meta.parse for proper AST parsing and operator precedence handling.
    
    This function evaluates mathematical expressions similar to how one would use
    eval(string, namespace) in problems:73-74.
    """
function parse_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    
    expr_str = strip(expr_str)
    
    # Handle empty expressions
    if isempty(expr_str)
        throw(ArgumentError("Empty expression string"))
    end
    
    # Handle simple zero cases
    if expr_str == "0" || expr_str == "zero"
        return ZeroOperator()
    end
    
    try
        # Use Julia's Meta.parse for proper expression parsing
        parsed = Meta.parse(expr_str)
        
        # Recursively evaluate the parsed expression with namespace
        result = evaluate_parsed_expression(parsed, namespace)
        
        return result
        
    catch e
        if isa(e, Meta.ParseError)
            @warn "Parse error in expression '$expr_str': $(e.msg)"
            return UnknownOperator(expr_str)
        else
            # For evaluation errors, try fallback parsing patterns
            @warn "Error evaluating expression '$expr_str': $e" maxlog=1
            return fallback_parse_expression(expr_str, namespace)
        end
    end
end

"""
    Recursively evaluate parsed expression with namespace substitution.
    Similar to eval(string, namespace) but with proper Julia AST handling.
    """
function evaluate_parsed_expression(expr, namespace::Dict{String, Any})
    
    if isa(expr, Symbol)
        # Variable/function lookup
        var_name = string(expr)
        if haskey(namespace, var_name)
            return namespace[var_name]
        else
            # Try built-in Julia functions/constants
            try
                return eval(expr)
            catch
                @warn "Unknown variable: $var_name"
                return UnknownOperator(var_name)
            end
        end
        
    elseif isa(expr, Number)
        # Numeric constants -> wrap in ConstantOperator for consistency
        return ConstantOperator(Float64(expr))
        
    elseif isa(expr, String)
        # String literals
        return expr
        
    elseif isa(expr, Expr)
        if expr.head == :call
            func_expr = expr.args[1]
            arg_exprs = expr.args[2:end]

            if func_expr isa Symbol && func_expr in (:+, :-, :*, :/, :^)
                op_args = [evaluate_parsed_expression(arg, namespace) for arg in arg_exprs]
                if isempty(op_args)
                    throw(ArgumentError("Operator $(func_expr) requires at least one operand"))
                end

                if func_expr == :+
                    result = op_args[1]
                    for i in 2:length(op_args)
                        result = AddOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :-
                    if length(op_args) == 1
                        return NegateOperator(op_args[1])
                    end
                    result = SubtractOperator(op_args[1], op_args[2])
                    for i in 3:length(op_args)
                        result = SubtractOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :*
                    result = op_args[1]
                    for i in 2:length(op_args)
                        result = MultiplyOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :/
                    if length(op_args) == 1
                        throw(ArgumentError("Division requires at least two operands"))
                    end
                    result = DivideOperator(op_args[1], op_args[2])
                    for i in 3:length(op_args)
                        result = DivideOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :^
                    if length(op_args) == 1
                        return op_args[1]
                    end
                    result = PowerOperator(op_args[end - 1], op_args[end])
                    for i in (length(op_args) - 2):-1:1
                        result = PowerOperator(op_args[i], result)
                    end
                    return result
                end
            end

            # ── Advection operator: u⋅∇(f) or dot(u, grad(f)) ──
            # style: u⋅∇(f) expands to Σᵢ uᵢ ∂ᵢf, handling
            # both scalar f (→ scalar result) and vector f (→ vector result).
            if func_expr === :⋅ || func_expr === :dot
                left = evaluate_parsed_expression(arg_exprs[1], namespace)
                right_expr = length(arg_exprs) >= 2 ? arg_exprs[2] : nothing
                # Detect ∇(f) or grad(f) on the right
                if right_expr isa Expr && right_expr.head == :call &&
                   right_expr.args[1] in (:∇, :grad, :gradient)
                    f = evaluate_parsed_expression(right_expr.args[2], namespace)
                    return _expand_advection(left, f)
                else
                    # Regular dot product
                    right = evaluate_parsed_expression(right_expr, namespace)
                    return MultiplyOperator(left, right)
                end
            end

            # Check for BC syntax: field(coord=value) → Interpolate
            # This handles expressions like u(y=0), T(z=1.0), dy(u)(y=0), etc.
            # In Julia's AST, keyword arguments appear as Expr(:kw, coord, value)
            # Following spectral methods pattern)
            if length(arg_exprs) == 1 && isa(arg_exprs[1], Expr) && arg_exprs[1].head == :kw
                kw_expr = arg_exprs[1]
                coord_name = string(kw_expr.args[1])
                position_expr = kw_expr.args[2]

                # Evaluate the function expression - this could be:
                # 1. A Symbol (field name like :u)
                # 2. An Expr (derivative expression like dy(u))
                operand = nothing
                if func_expr isa Symbol
                    field_name = string(func_expr)
                    if haskey(namespace, field_name)
                        operand = namespace[field_name]
                    end
                elseif func_expr isa Expr
                    # Recursively evaluate - handles dy(u)(y=0), lap(u)(y=0), etc.
                    operand = evaluate_parsed_expression(func_expr, namespace)
                end

                # Check if this is a valid operand for interpolation
                if operand !== nothing && isa(operand, Operand)
                    # Try to find the coordinate
                    coord = _find_coordinate_for_field(operand, coord_name, namespace)

                    # If not found in field, check namespace directly
                    if coord === nothing && haskey(namespace, coord_name)
                        coord_obj = namespace[coord_name]
                        if isa(coord_obj, Coordinate)
                            coord = coord_obj
                        end
                    end

                    if coord !== nothing
                        # Evaluate position value
                        position = coerce_constant_value(evaluate_parsed_expression(position_expr, namespace))

                        @debug "Detected BC syntax: $func_expr($coord_name=$position) → Interpolate"
                        return interpolate(operand, coord, Float64(position))
                    end
                end
            end

            if func_expr isa Symbol
                if func_expr == :dt || func_expr == :∂t
                    if isempty(arg_exprs)
                        throw(ArgumentError("dt/∂t requires at least one argument"))
                    end
                    field = evaluate_parsed_expression(arg_exprs[1], namespace)
                    order = if length(arg_exprs) >= 2
                        val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                        Int(round(val))
                    else
                        1
                    end
                    return dt(field, order)
                elseif func_expr == :d || func_expr == :diff
                    if length(arg_exprs) < 2
                        throw(ArgumentError("d requires at least an operand and a coordinate"))
                    end
                    field = evaluate_parsed_expression(arg_exprs[1], namespace)
                    coord = evaluate_parsed_expression(arg_exprs[2], namespace)
                    order = if length(arg_exprs) >= 3
                        val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                        Int(round(val))
                    else
                        1
                    end
                    return d(field, coord, order)
                elseif func_expr == :lift
                    if length(arg_exprs) == 2
                        # Short form: lift(tau, -1) — auto-detect basis
                        operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                        n_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                        n_int = Int(round(n_val))
                        try
                            return lift(operand, n_int)
                        catch
                            # Auto-detection failed; return operand directly.
                            # For matrix sizing the Lift is just a shape-preserving
                            # wrapper, so the operand alone is sufficient.
                            return operand
                        end
                    elseif length(arg_exprs) >= 3
                        # Full form: lift(tau, basis, -1)
                        operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                        basis = evaluate_parsed_expression(arg_exprs[2], namespace)
                        n_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                        return lift(operand, basis, Int(round(n_val)))
                    else
                        throw(ArgumentError("lift requires (operand, n) or (operand, basis, n)"))
                    end
                elseif func_expr == :fraclap || func_expr == :Δᵅ
                    # Fractional Laplacian: fraclap(f, α) or Δᵅ(f, α)
                    if length(arg_exprs) < 2
                        throw(ArgumentError("fraclap requires operand and exponent α"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    α_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                    return fraclap(operand, Float64(α_val))
                elseif func_expr == :sqrtlap || func_expr == :√Δ || func_expr == :Δ½
                    # Square root Laplacian: (-Δ)^(1/2)
                    if isempty(arg_exprs)
                        throw(ArgumentError("sqrtlap requires an operand"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    return sqrtlap(operand)
                elseif func_expr == :invsqrtlap || func_expr == :Δ⁻½
                    # Inverse square root Laplacian: (-Δ)^(-1/2)
                    if isempty(arg_exprs)
                        throw(ArgumentError("invsqrtlap requires an operand"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    return invsqrtlap(operand)
                elseif func_expr in (:integ, :integrate)
                    # Deferred integration — must NOT eagerly evaluate to a number.
                    # integ(f) → integrate over all coords; integ(f, coord) → one coord.
                    if isempty(arg_exprs)
                        throw(ArgumentError("integ requires at least one argument"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    if length(arg_exprs) >= 2
                        coord = evaluate_parsed_expression(arg_exprs[2], namespace)
                        return integrate(operand, coord)
                    elseif isa(operand, Operand) && hasfield(typeof(operand), :dist) &&
                           operand.dist !== nothing && hasfield(typeof(operand.dist), :coordsys)
                        # Construct Integrate node directly to avoid dispatch
                        # ambiguity with integrate(::ScalarField, axes=:)
                        all_coords = tuple(coords(operand.dist.coordsys)...)
                        return Integrate(operand, all_coords)
                    else
                        return Integrate(operand, ())
                    end
                elseif func_expr in (:avg, :average)
                    # Deferred average — same pattern as integ.
                    if length(arg_exprs) < 2
                        throw(ArgumentError("average requires (operand, coord)"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    coord = evaluate_parsed_expression(arg_exprs[2], namespace)
                    return average(operand, coord)
                end
            end
            
            func = evaluate_parsed_expression(func_expr, namespace)
            evaluated_args = [coerce_constant_value(evaluate_parsed_expression(arg, namespace)) for arg in arg_exprs]

            if isa(func, Function)
                # Short-circuit ONLY when an argument contains an
                # `UnknownOperator` placeholder (typically a coordinate or
                # time variable like `x` / `t` in a space- or time-dependent
                # BC RHS that can't be bound at parse time). In that case
                # the function call would fail at the Julia level (e.g.
                # `sin(::UnknownOperator)` → MethodError). The spatial /
                # temporal BC evaluator in `_apply_bc_values_to_equations!`
                # re-evaluates the BC's original string at runtime against
                # actual coordinate grid arrays / current time, and
                # overwrites `equation_data[eq_idx]["F"]` with the result —
                # so the parser's symbolic placeholder is never consulted.
                #
                # For normal field-operator expressions like `trace(grad_u)`
                # where `grad_u` is a stored substitution (an `AddOperator`
                # / `Future` tree of concrete fields), no `UnknownOperator`
                # is present and the function call proceeds as normal,
                # letting operators like `trace` algebraically simplify to
                # `divergence(u) + lift(...)`.
                if any(_expression_contains_unknown, evaluated_args)
                    return UnknownOperator(string(expr))
                end
                return func(evaluated_args...)
            else
                @warn "Unknown function in expression: $(string(func_expr))"
                return UnknownOperator(string(expr))
            end
            
        elseif expr.head in [:+, :-, :*, :/, :^]
            # Arithmetic operations with proper precedence from Meta.parse
            if length(expr.args) == 2
                left = evaluate_parsed_expression(expr.args[1], namespace)
                right = evaluate_parsed_expression(expr.args[2], namespace)
                
                if expr.head == :+
                    return AddOperator(left, right)
                elseif expr.head == :-
                    return SubtractOperator(left, right)
                elseif expr.head == :*
                    return MultiplyOperator(left, right)
                elseif expr.head == :/
                    return DivideOperator(left, right)
                elseif expr.head == :^
                    return PowerOperator(left, right)
                end
            elseif length(expr.args) == 1 && expr.head == :-
                # Unary minus
                operand = evaluate_parsed_expression(expr.args[1], namespace)
                return NegateOperator(operand)
            end
            
        elseif expr.head == :ref
            # Array/field indexing: field[i, j]
            array_expr = expr.args[1]
            indices = expr.args[2:end]
            
            array = evaluate_parsed_expression(array_expr, namespace)
            evaluated_indices = [evaluate_parsed_expression(idx, namespace) for idx in indices]
            
            return IndexOperator(array, evaluated_indices)
            
        elseif expr.head == :block
            # Block expressions - evaluate meaningful statements
            meaningful_args = filter(arg -> !isa(arg, LineNumberNode), expr.args)
            if !isempty(meaningful_args)
                return evaluate_parsed_expression(meaningful_args[end], namespace)
            end
        end
        
        # Fallback for unsupported expression types
        @debug "Unsupported expression type: $(expr.head)"
        return UnknownOperator(string(expr))
        
    else
        # Return other types as-is
        return expr
    end
end

"""
    Fallback parsing for expressions that fail Meta.parse evaluation.
    Uses simple string pattern matching as backup.
    """
function fallback_parse_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    
    # Handle simple field references
    if haskey(namespace, expr_str)
        return namespace[expr_str]
    end
    
    # Handle simple numeric constants
    try
        val = parse(Float64, expr_str)
        return ConstantOperator(val)
    catch
        # Continue with pattern matching
    end
    
    # Pattern-based parsing for common PDE operators (as fallback)
    # Unicode time derivative: ∂t(field)
    if startswith(expr_str, "∂t(") && endswith(expr_str, ")")
        field_name = expr_str[nextind(expr_str, 0, 3):end-1]  # Skip "∂t(" which is 3 chars
        if haskey(namespace, field_name)
            return TimeDerivative(namespace[field_name], 1)
        end
    end
    
    if startswith(expr_str, "d(") && endswith(expr_str, ")")
        args_str = expr_str[3:end-1]
        args = [strip(arg) for arg in split(args_str, ',')]
        if length(args) >= 2 && haskey(namespace, args[1]) && haskey(namespace, args[2])
            return Differentiate(namespace[args[1]], namespace[args[2]], 1)
        end
    end
    
    # Return as unknown operator if all parsing fails
    return UnknownOperator(expr_str)
end

"""
    _expand_advection(u, f) → expression tree for u⋅∇(f) = Σᵢ uᵢ ∂ᵢf

Expands the advection operator into scalar derivative terms.
Works for both scalar f (returns scalar) and vector f (returns vector via
component-wise differentiation).
"""
function _expand_advection(u, f)
    # Handle -u⋅∇(f) → -(u⋅∇(f)):  Julia parses -u⋅∇(f) as (-u)⋅∇(f)
    if isa(u, NegateOperator) && isa(u.operand, VectorField)
        return NegateOperator(_expand_advection(u.operand, f))
    end
    # Handle c*u⋅∇(f) → c*(u⋅∇(f))
    if isa(u, MultiplyOperator)
        if isa(u.left, VectorField)
            return MultiplyOperator(u.right, _expand_advection(u.left, f))
        elseif isa(u.right, VectorField)
            return MultiplyOperator(u.left, _expand_advection(u.right, f))
        end
    end
    if !isa(u, VectorField)
        throw(ArgumentError("Advection u⋅∇(f) requires u to be a VectorField, got $(typeof(u))"))
    end
    coordsys = u.coordsys
    coord_list = coords(coordsys)
    ndim = min(length(coord_list), length(u.components))

    result = nothing
    for i in 1:ndim
        ui = u.components[i]
        ci = coord_list[i]
        # uᵢ * ∂ᵢf — Differentiate handles both ScalarField and VectorField
        term = MultiplyOperator(ui, Differentiate(f, ci, 1))
        result = result === nothing ? term : AddOperator(result, term)
    end
    return result
end

"""
    _expression_contains_unknown(expr) -> Bool

Return `true` if an expression tree (either an `Operator` subtype or a
`Future` node) contains an `UnknownOperator` placeholder anywhere in its
subtree. Used by the equation parser's function-call path to decide
whether to short-circuit `func(args...)` into a symbolic
`UnknownOperator(string(expr))` — which it does when an argument carries
an unbound coordinate / time placeholder — versus proceeding with the
call, which is correct for normal operator expressions built from
concrete fields.

The walk handles:
- `UnknownOperator` directly
- Operator subtypes via `operand` / `left` / `right` fields
- `Future` subtypes via `future_args`
- `Number` and literal values (short-circuit false)
"""
function _expression_contains_unknown(expr)
    isa(expr, UnknownOperator) && return true
    (isa(expr, Number) || isa(expr, Symbol) || isa(expr, AbstractString)) && return false
    # Operator subtypes commonly store children in :operand / :left / :right.
    for f in (:operand, :left, :right)
        if hasfield(typeof(expr), f)
            child = getfield(expr, f)
            _expression_contains_unknown(child) && return true
        end
    end
    # Future subtypes store children via `future_args`.
    if isa(expr, Future)
        for arg in future_args(expr)
            _expression_contains_unknown(arg) && return true
        end
    end
    return false
end

# Helper operator types for parsing (following operator structure)
struct ZeroOperator <: Operator end
struct ConstantOperator <: Operator
    value::Float64
end
struct ArrayOperator <: Operator
    value::AbstractArray
end
struct UnknownOperator <: Operator
    expression::String
end

# has() definitions for ZeroOperator, ConstantOperator, ArrayOperator, UnknownOperator
# These types never contain problem variables (following spectral pattern)
has(::ZeroOperator, vars...) = false
has(::ConstantOperator, vars...) = false
has(::ArrayOperator, vars...) = false
has(::UnknownOperator, vars...) = false  # Conservative: unknowns don't contain tracked vars

@inline coerce_constant_value(x) = x isa ConstantOperator ? x.value : x

# Note: AddOperator, SubtractOperator, MultiplyOperator, DivideOperator, PowerOperator,
# NegateOperator, IndexOperator are defined in operators.jl

"""
    _find_coordinate_for_field(field, coord_name, namespace)

Find a Coordinate object for the given coordinate name that is associated with the field.
Used for automatic BC syntax parsing: field(coord=value) → Interpolate(field, coord, value)

Following spectral methods pattern):
- If coord is already a Coordinate, use it directly
- If coord is a string, look it up in the field's domain or namespace
"""
function _find_coordinate_for_field(field::Operand, coord_name::String, namespace::Dict{String, Any})
    # First check namespace for coordinate objects
    if haskey(namespace, coord_name)
        coord = namespace[coord_name]
        if isa(coord, Coordinate)
            return coord
        end
    end

    # Try to find coordinate from field's domain/bases
    if isa(field, ScalarField) && hasfield(typeof(field), :dist)
        dist = field.dist
        if dist !== nothing && hasfield(typeof(dist), :coords)
            for c in dist.coords
                if c.name == coord_name
                    return c
                end
            end
            # Try coordinate-system indexing if available
            if hasfield(typeof(dist), :coordsys)
                try
                    return dist.coordsys[coord_name]
                catch
                    # Coordinate not found in coordsys
                end
            end
        end
    end

    # Try to construct coordinate from field's bases
    if hasfield(typeof(field), :bases)
        for basis in field.bases
            if basis !== nothing && hasfield(typeof(basis), :meta)
                if basis.meta.element_label == coord_name
                    # Create coordinate from basis metadata
                    if hasfield(typeof(basis.meta), :coordsys) && basis.meta.coordsys !== nothing
                        coordsys = basis.meta.coordsys
                        if hasfield(typeof(coordsys), :coords)
                            for c in coordsys.coords
                                if c.name == coord_name
                                    return c
                                end
                            end
                        end
                        # Try subscript access
                        try
                            return coordsys[coord_name]
                        catch
                            # Not found
                        end
                    end
                end
            end
        end
    end

    @debug "Could not find coordinate '$coord_name' for field"
    return nothing
end

