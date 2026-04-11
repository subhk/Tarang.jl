# Domain setup
"""Setup domain for problem based on variables"""
function setup_domain!(problem::Problem)
    
    if length(problem.variables) == 0
        throw(ArgumentError("No variables specified"))
    end
    
    # Get distributor and bases from first variable
    first_var = problem.variables[1]
    if isa(first_var, ScalarField)
        problem.domain = first_var.domain
    elseif isa(first_var, VectorField)
        problem.domain = first_var.domain
    elseif isa(first_var, TensorField)
        problem.domain = first_var.domain
    else
        throw(ArgumentError("Unknown variable type: $(typeof(first_var))"))
    end
    
    # Verify all variables have compatible domains
    for var in problem.variables
        var_domain = nothing
        if isa(var, ScalarField)
            var_domain = var.domain
        elseif isa(var, VectorField)
            var_domain = var.domain
        elseif isa(var, TensorField)
            var_domain = var.domain
        end
        
        if var_domain !== nothing && var_domain != problem.domain
            @debug "Variable $(var.name) has incompatible domain"
        end
    end
end

# Problem validation
"""Validate problem formulation"""
function validate_problem(problem::Problem)

    errors = String[]

    if length(problem.variables) == 0
        push!(errors, "No variables specified")
    end

    if length(problem.equations) == 0
        push!(errors, "No equations specified")
    end

    # For IVPs/EVPs: equations should match variables exactly
    # For BVPs: equations should be >= variables (includes BCs)
    if isa(problem, LBVP) || isa(problem, NLBVP)
        # BVPs can have extra equations for boundary conditions
        if length(problem.equations) < length(problem.variables)
            push!(errors, "Number of equations ($(length(problem.equations))) is less than number of variables ($(length(problem.variables)))")
        end
    else
        # IVP/EVP: strict match
        if length(problem.equations) != length(problem.variables)
            push!(errors, "Number of equations ($(length(problem.equations))) does not match number of variables ($(length(problem.variables)))")
        end
    end

    # Check for required boundary conditions in boundary value problems
    # Note: BCs can be embedded in equations via field(coord=value) syntax
    if isa(problem, LBVP) || isa(problem, NLBVP)
        # For BVPs, we either need explicit BCs or equations > variables (implicit BCs)
        has_explicit_bcs = length(problem.boundary_conditions) > 0 || length(problem.bc_manager.conditions) > 0
        has_implicit_bcs = length(problem.equations) > length(problem.variables)
        if !has_explicit_bcs && !has_implicit_bcs
            push!(errors, "Boundary value problem requires boundary conditions")
        end
    end
    
    # Validate boundary conditions if using advanced BC system
    if length(problem.bc_manager.conditions) > 0
        try
            validate_boundary_conditions(problem.bc_manager, problem)
        catch e
            push!(errors, "Boundary condition validation failed: $e")
        end
    end
    
    if length(errors) > 0
        throw(ArgumentError("Problem validation failed:\n" * join(errors, "\n")))
    end
    
    return true
end

# Substitutions and namespace management

"""
    add_substitution!(problem, name, expression)

Add a named parameter to the problem namespace. Prefer `add_parameters!` for
setting multiple parameters at once:

```julia
add_parameters!(problem, nu=1e-3, kappa=1e-4)
```
"""
function add_substitution!(problem::Problem, name::String, expression;
                           _internal::Bool=false)
    if !_internal
        Base.depwarn(
            "`add_substitution!(problem, name, value)` is deprecated, " *
            "use `add_parameters!(problem, $name=$expression)` instead.",
            :add_substitution!)
    end
    problem.namespace[name] = expression
end

"""
    Expand substitutions in equations following pattern.
    Following expand(*vars) methods (arithmetic:319-329, operators:704-739).
    """
function expand_substitutions!(problem::Problem)
    
    # Get all variables for expansion
    variables = problem.variables
    
    # Expand equation expressions if they exist
    if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
        # Expand parsed equation data
        for eq_data in problem.equation_data
            try
                # Expand each matrix expression
                for matrix_name in ["M", "L", "F"]
                    if haskey(eq_data, matrix_name) && eq_data[matrix_name] !== nothing
                        expr = eq_data[matrix_name]
                        expanded_expr = expand_expression(expr, variables)
                        eq_data[matrix_name] = expanded_expr
                        @debug "Expanded $matrix_name expression" original=expr expanded=expanded_expr
                    end
                end
                
                # Expand LHS and RHS if they exist
                if haskey(eq_data, "lhs") && eq_data["lhs"] !== nothing
                    eq_data["lhs"] = expand_expression(eq_data["lhs"], variables)
                end
                if haskey(eq_data, "rhs") && eq_data["rhs"] !== nothing
                    eq_data["rhs"] = expand_expression(eq_data["rhs"], variables)
                end
                
            catch e
                @warn "Failed to expand substitutions in equation" exception=e
            end
        end
    else
        # Expand string equations if no equation data exists
        for (i, equation_str) in enumerate(problem.equations)
            try
                # Parse and expand the equation
                lhs, rhs = parse_equation(equation_str, problem.namespace)
                expanded_lhs = expand_expression(lhs, variables)
                expanded_rhs = expand_expression(rhs, variables)
                
                # Reconstruct expanded equation string
                expanded_equation = reconstruct_equation_string(expanded_lhs, expanded_rhs)
                problem.equations[i] = expanded_equation
                
                @debug "Expanded equation $i" original=equation_str expanded=expanded_equation
                
            catch e
                @warn "Failed to expand equation $i: $equation_str" exception=e
            end
        end
    end
    
    # Expand namespace substitutions
    expand_namespace_substitutions!(problem)
    
    @info "Substitution expansion completed for $(length(problem.equations)) equations"
    return problem
end

"""
    Expand expression over specified variables following pattern.
    Following operators:expand and arithmetic:expand methods.
    """
function expand_expression(expr, variables::Vector)
    
    if expr === nothing || isa(expr, String)
        return expr
    end
    
    # Check if expression contains any of the specified variables
    if !has_variables(expr, variables)
        return expr
    end
    
    # Expand based on expression type
    if isa(expr, AddOperator)
        # Expand addition: sum(expand(arg) for arg in args)
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return combine_add_expressions(left_expanded, right_expanded)
        
    elseif isa(expr, MultiplyOperator)
        # Expand multiplication with distribution over addition
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return expand_multiply_expressions(left_expanded, right_expanded, variables)
        
    elseif isa(expr, SubtractOperator)
        # Expand subtraction
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return SubtractOperator(left_expanded, right_expanded)

    elseif isa(expr, DivideOperator)
        # Expand division (only expand numerator, denominator stays as-is)
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return DivideOperator(left_expanded, right_expanded)

    elseif isa(expr, PowerOperator)
        # Expand power (expand base, exponent stays as-is typically)
        base_expanded = expand_expression(expr.left, variables)
        exp_expanded = expand_expression(expr.right, variables)
        return PowerOperator(base_expanded, exp_expanded)

    elseif isa(expr, NegateOperator)
        # Expand negation
        expanded_operand = expand_expression(expr.operand, variables)
        return NegateOperator(expanded_operand)

    elseif isa(expr, IndexOperator)
        # Expand indexed expression
        expanded_array = expand_expression(expr.array, variables)
        return IndexOperator(expanded_array, expr.indices)

    elseif isa(expr, Union{TimeDerivative, Laplacian, Gradient, Divergence, Differentiate})
        # Expand operators: distribute over operand
        expanded_operand = expand_expression(expr.operand, variables)
        return distribute_operator_over_operand(expr, expanded_operand, variables)

    elseif isa(expr, ConstantOperator) || isa(expr, ZeroOperator)
        # Constants don't expand
        return expr

    else
        # Unknown expression types
        @debug "Unknown expression type in expansion: $(typeof(expr))"
        return expr
    end
end

"""
    Check if expression contains any of the specified variables.

    Wrapper around the proper has() method from field.jl/operators.jl following
    the spectral pattern for variable linearity checking.
    """
function has_variables(expr, variables::Vector)

    if expr === nothing || isa(expr, String)
        return false
    end

    # Use the proper has() method with splatted variables
    return has(expr, variables...)
end

"""Combine two expressions with addition, flattening nested additions"""
function combine_add_expressions(left, right)
    
    if isa(left, ZeroOperator)
        return right
    elseif isa(right, ZeroOperator)
        return left
    else
        return AddOperator(left, right)
    end
end

"""
    Expand multiplication with distribution over addition.
    Following arithmetic:expand multiplication pattern.
    """
function expand_multiply_expressions(left, right, variables::Vector)
    
    # If either operand is addition involving variables, distribute
    if isa(left, AddOperator) && has_variables(left, variables)
        # (a + b) * c = a*c + b*c
        term1 = expand_multiply_expressions(left.left, right, variables)
        term2 = expand_multiply_expressions(left.right, right, variables)
        return combine_add_expressions(term1, term2)
        
    elseif isa(right, AddOperator) && has_variables(right, variables)
        # a * (b + c) = a*b + a*c
        term1 = expand_multiply_expressions(left, right.left, variables)
        term2 = expand_multiply_expressions(left, right.right, variables)
        return combine_add_expressions(term1, term2)
        
    else
        # Simple multiplication
        return MultiplyOperator(left, right)
    end
end

"""
    Distribute operator over expanded operand.
    Following operators:_expand_add pattern.
    """
function distribute_operator_over_operand(operator, expanded_operand, variables::Vector)
    
    if isa(expanded_operand, AddOperator) && has_variables(expanded_operand, variables)
        # Op(a + b) = Op(a) + Op(b) for linear operators
        left_term = distribute_operator_over_operand(operator, expanded_operand.left, variables)
        right_term = distribute_operator_over_operand(operator, expanded_operand.right, variables)
        return combine_add_expressions(left_term, right_term)
        
    else
        # Apply operator to single operand
        return create_similar_operator(operator, expanded_operand)
    end
end

"""Create new operator of same type with different operand"""
function create_similar_operator(operator, new_operand)
    
    if isa(operator, TimeDerivative)
        return TimeDerivative(new_operand, operator.order)
    elseif isa(operator, Laplacian)
        return Laplacian(new_operand)
    elseif isa(operator, Gradient)
        return Gradient(new_operand, operator.coordsys)
    elseif isa(operator, Divergence)
        return Divergence(new_operand)
    elseif isa(operator, Differentiate)
        return Differentiate(new_operand, operator.coord, operator.order)
    else
        @debug "Unknown operator type for recreation: $(typeof(operator))"
        return operator  # Return original if can't recreate
    end
end

"""Reconstruct equation string from expanded expressions"""
function reconstruct_equation_string(lhs, rhs)
    
    lhs_str = expression_to_string(lhs)
    rhs_str = expression_to_string(rhs)
    
    return "$lhs_str = $rhs_str"
end

"""Convert expression back to string representation"""
# Convert expression tree to string — uses dispatch instead of if/elseif chain
expression_to_string(expr::String) = expr
expression_to_string(expr::Number) = string(expr)
expression_to_string(::ZeroOperator) = "0"
expression_to_string(expr::ConstantOperator) = string(expr.value)
expression_to_string(expr::AddOperator) = "($(expression_to_string(expr.left)) + $(expression_to_string(expr.right)))"
expression_to_string(expr::SubtractOperator) = "($(expression_to_string(expr.left)) - $(expression_to_string(expr.right)))"
expression_to_string(expr::MultiplyOperator) = "($(expression_to_string(expr.left)) * $(expression_to_string(expr.right)))"
expression_to_string(expr::DivideOperator) = "($(expression_to_string(expr.left)) / $(expression_to_string(expr.right)))"
expression_to_string(expr::PowerOperator) = "($(expression_to_string(expr.left)) ^ $(expression_to_string(expr.right)))"
expression_to_string(expr::NegateOperator) = "(-$(expression_to_string(expr.operand)))"
expression_to_string(expr::TimeDerivative) = "∂t($(expression_to_string(expr.operand)))"
expression_to_string(expr::Laplacian) = "Δ($(expression_to_string(expr.operand)))"
expression_to_string(expr::Gradient) = "∇($(expression_to_string(expr.operand)))"
expression_to_string(expr::Divergence) = "∇·($(expression_to_string(expr.operand)))"
function expression_to_string(expr::Differentiate)
    coord_name = hasfield(typeof(expr.coord), :name) ? expr.coord.name : string(expr.coord)
    return "d($(expression_to_string(expr.operand)), $coord_name)"
end
expression_to_string(expr) = hasfield(typeof(expr), :name) ? expr.name : string(expr)

"""
    expand_namespace_substitutions!(problem::Problem)

Expand any substitution definitions in problem namespace.

Substitutions are defined in the namespace as strings of the form "pattern = replacement".
They are applied to all equations using word-boundary-aware replacement to avoid
partial matches within variable names.

# Example
```julia
problem.namespace["sub1"] = "nu = 1e-3"
problem.namespace["sub2"] = "f = sin(x)"
# "nu*lap(u)" becomes "1e-3*lap(u)"
# "f + g" becomes "sin(x) + g" but "freq" stays unchanged
```
"""
function expand_namespace_substitutions!(problem::Problem)
    # Look for substitution patterns in namespace
    substitutions = Dict{String, String}()

    for (key, value) in problem.namespace
        if isa(value, String) && contains(value, "=")
            # Potential substitution definition
            try
                lhs, rhs = split_equation(value)
                lhs_clean = strip(lhs)
                rhs_clean = strip(rhs)
                if !isempty(lhs_clean) && !isempty(rhs_clean)
                    substitutions[lhs_clean] = rhs_clean
                end
            catch
                # Not a valid substitution, skip
            end
        end
    end

    if isempty(substitutions)
        return
    end

    @debug "Found substitutions in namespace" substitutions

    # Sort substitutions by pattern length (longest first) to avoid partial replacements
    sorted_patterns = sort(collect(keys(substitutions)), by=length, rev=true)

    # Apply substitutions to equations
    for (i, equation) in enumerate(problem.equations)
        modified_equation = apply_substitutions(equation, substitutions, sorted_patterns)
        if modified_equation != equation
            problem.equations[i] = modified_equation
            @debug "Applied substitution to equation $i" original=equation modified=modified_equation
        end
    end

    # Also apply to boundary conditions if they are strings
    for (i, bc) in enumerate(problem.boundary_conditions)
        if isa(bc, String)
            modified_bc = apply_substitutions(bc, substitutions, sorted_patterns)
            if modified_bc != bc
                problem.boundary_conditions[i] = modified_bc
                @debug "Applied substitution to boundary condition $i"
            end
        end
    end
end

"""
    apply_substitutions(text::String, substitutions::Dict{String,String},
                       sorted_patterns::Vector{String}) -> String

Apply substitutions to text using word-boundary-aware replacement.

Only replaces patterns that appear as whole words (not as substrings of longer
identifiers). Uses regex word boundaries to ensure correct matching.

# Arguments
- `text`: The string to perform substitutions on
- `substitutions`: Dict mapping patterns to their replacements
- `sorted_patterns`: Patterns sorted by length (longest first)
"""
function apply_substitutions(text::String, substitutions::Dict{String,String},
                            sorted_patterns::Vector{String})
    result = text

    for pattern in sorted_patterns
        replacement = substitutions[pattern]

        # Build word-boundary-aware regex pattern
        # Match pattern only when surrounded by non-identifier characters
        # This prevents "nu" from matching inside "enumerate"
        escaped_pattern = escape_regex_chars(pattern)

        # Use word boundaries: pattern must be preceded and followed by
        # non-word characters (or start/end of string)
        regex_pattern = Regex("(?<![a-zA-Z0-9_])$(escaped_pattern)(?![a-zA-Z0-9_])")

        result = replace(result, regex_pattern => replacement)
    end

    return result
end

"""
    escape_regex_chars(s::String) -> String

Escape special regex characters in a string for literal matching.
"""
function escape_regex_chars(s::String)
    # Characters that have special meaning in regex
    special_chars = raw"\.^$*+?{}[]|()"

    result = IOBuffer()
    for c in s
        if c in special_chars
            write(result, '\\')
        end
        write(result, c)
    end

    return String(take!(result))
end

"""
    apply_substitution_recursive(expr, substitutions::Dict)

Apply substitutions to an operator expression tree recursively.
Used when equations are already parsed into operator form.
Returns a new expression tree with substitutions applied (immutable-safe).
"""
function apply_substitution_recursive(expr, substitutions::Dict)
    if expr === nothing
        return expr
    end

    # Handle different expression types
    if isa(expr, ScalarField) && haskey(substitutions, expr.name)
        # Can't directly replace a field, but we can note it
        @debug "Found field $(expr.name) in substitutions but cannot replace parsed field"
        return expr

    elseif isa(expr, AddOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = apply_substitution_recursive(expr.right, substitutions)
        return AddOperator(new_left, new_right)

    elseif isa(expr, SubtractOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = apply_substitution_recursive(expr.right, substitutions)
        return SubtractOperator(new_left, new_right)

    elseif isa(expr, MultiplyOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = isa(expr.right, Number) ? expr.right : apply_substitution_recursive(expr.right, substitutions)
        return MultiplyOperator(new_left, new_right)

    elseif isa(expr, DivideOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = apply_substitution_recursive(expr.right, substitutions)
        return DivideOperator(new_left, new_right)

    elseif isa(expr, NegateOperator)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return NegateOperator(new_operand)

    elseif isa(expr, PowerOperator)
        new_base = apply_substitution_recursive(expr.left, substitutions)
        new_exp = apply_substitution_recursive(expr.right, substitutions)
        return PowerOperator(new_base, new_exp)

    elseif isa(expr, TimeDerivative)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return TimeDerivative(new_operand, expr.order)

    elseif isa(expr, Laplacian)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return Laplacian(new_operand)

    elseif isa(expr, Differentiate)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return Differentiate(new_operand, expr.coord, expr.order)

    elseif hasfield(typeof(expr), :operand)
        # Generic operand-based operator
        new_operand = apply_substitution_recursive(getfield(expr, :operand), substitutions)
        # Try to construct a new instance with the same type
        return expr  # Fallback: return original if can't reconstruct

    elseif hasfield(typeof(expr), :left) && hasfield(typeof(expr), :right)
        # Generic binary operator - return original since we can't reconstruct unknown types
        return expr
    end

    return expr
end

# Problem metadata
"""Get names of all variables"""
function get_variable_names(problem::Problem)
    names = String[]
    for var in problem.variables
        if isa(var, ScalarField)
            push!(names, var.name)
        elseif isa(var, VectorField)
            for comp in var.components
                push!(names, comp.name)
            end
        elseif isa(var, TensorField)
            for i in 1:size(var.components, 1), j in 1:size(var.components, 2)
                push!(names, var.components[i,j].name)
            end
        end
    end
    return names
end

"""Get total number of equations including boundary conditions"""
function get_equation_count(problem::Problem)
    return length(problem.equations) + length(problem.boundary_conditions)
end

"""Get total number of scalar variables"""
function get_variable_count(problem::Problem)
    count = 0
    for var in problem.variables
        if isa(var, ScalarField)
            count += 1
        elseif isa(var, VectorField)
            count += length(var.components)
        elseif isa(var, TensorField)
            count += length(var.components)
        end
    end
    return count
end

# ============================================================================
# Convenience Macros
# ============================================================================

"""
    @equations problem begin
        "equation1"
        "equation2"
        ...
    end

Add multiple equations to a problem with cleaner syntax.

# Example
```julia
problem = IVP([u, v, p])

@equations problem begin
    "∂t(u) - ν*Δ(u) + ∇(p) = -u⋅∇(u)"
    "∂t(v) - ν*Δ(v) + ∇(p) = -u⋅∇(v)"
    "div(u) = 0"
end
```

Equivalent to:
```julia
add_equation!(problem, "∂t(u) - ν*Δ(u) + ∇(p) = -u⋅∇(u)")
add_equation!(problem, "∂t(v) - ν*Δ(v) + ∇(p) = -u⋅∇(v)")
add_equation!(problem, "div(u) = 0")
```
"""
macro equations(problem, block)
    if block.head != :block
        error("@equations requires a begin...end block")
    end

    calls = Expr[]
    for arg in block.args
        # Skip LineNumberNodes
        if arg isa LineNumberNode
            continue
        end
        # Each argument should be a string (equation)
        push!(calls, :(add_equation!($(esc(problem)), $(esc(arg)))))
    end

    return Expr(:block, calls...)
end

"""
    @bcs problem begin
        "bc1"
        "bc2"
        ...
    end

Add multiple boundary conditions to a problem with cleaner syntax.

# Example
```julia
@bcs problem begin
    "u(z=0) = 0"
    "u(z=Lz) = 0"
    "integ(p) = 0"
end
```

Equivalent to:
```julia
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=Lz) = 0")
add_bc!(problem, "integ(p) = 0")
```
"""
macro bcs(problem, block)
    if block.head != :block
        error("@bcs requires a begin...end block")
    end

    calls = Expr[]
    for arg in block.args
        # Skip LineNumberNodes
        if arg isa LineNumberNode
            continue
        end
        push!(calls, :(add_bc!($(esc(problem)), $(esc(arg)))))
    end

    return Expr(:block, calls...)
end

"""
    @substitutions problem begin
        "name1" => value1
        "name2" => value2
        ...
    end

Add multiple substitutions to a problem with cleaner syntax.

# Example
```julia
@substitutions problem begin
    "nu" => 1e-3
    "kappa" => 1e-4
    "Ra" => 1e6
end
```

Equivalent to:
```julia
add_substitution!(problem, "nu", 1e-3)
add_substitution!(problem, "kappa", 1e-4)
add_substitution!(problem, "Ra", 1e6)
```
"""
macro substitutions(problem, block)
    if block.head != :block
        error("@substitutions requires a begin...end block")
    end

    calls = Expr[]
    for arg in block.args
        # Skip LineNumberNodes
        if arg isa LineNumberNode
            continue
        end
        # Expect pair syntax: "name" => value
        if arg isa Expr && arg.head == :call && arg.args[1] == :(=>)
            name = arg.args[2]
            value = arg.args[3]
            push!(calls, :(add_substitution!($(esc(problem)), $(esc(name)), $(esc(value)))))
        else
            error("@substitutions expects \"name\" => value pairs")
        end
    end

    return Expr(:block, calls...)
end

# ============================================================================
# Exports
# ============================================================================

# Export problem types
export Problem, IVP, LBVP, NLBVP, EVP

# Export core API functions
export add_equation!, add_bc!, add_substitution!
export @equations, @bcs, @substitutions
export set_parameter!, get_parameter, register_tau_field!

# Export stochastic forcing integration
export add_stochastic_forcing!, has_stochastic_forcing, get_stochastic_forcing

# Export temporal filter integration
export add_temporal_filter!, has_temporal_filters, get_temporal_filter, get_all_temporal_filters

# Export problem query functions
export get_variable_names, get_equation_count, get_variable_count

# Export problem building/validation functions
export build_problem_namespace, build_matrices, build_matrix_expressions!
export validate_problem, parse_equation, parse_expression

# Export BC string parsing functions
export parse_bc_string, parse_neumann_bc_string, parse_robin_bc_string, parse_stress_free_bc_string

# Export equation data functions
export set_equation_condition!, set_valid_modes!, get_matrix_expression
export is_equation_valid, check_equation_condition
