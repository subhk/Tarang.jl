"""
Problem formulation classes

Translated from dedalus/core/problems.py
"""

using LinearAlgebra
using SparseArrays

abstract type Problem end

mutable struct IVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    time::Union{Nothing, ScalarField}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

mutable struct LBVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

mutable struct NLBVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

mutable struct EVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    eigenvalue::Union{Nothing, Symbol}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

function build_problem_namespace(variables::Vector{Operand}, user_ns::Union{Nothing, AbstractDict})
    ns = Dict{String, Any}()
    
    # Lowest priority: operator registries (parseables first, then aliases)
    for (name, value) in OPERATOR_PARSEABLES
        ns[name] = value
    end
    for (name, value) in OPERATOR_ALIASES
        ns[name] = value
    end
    
    # User-supplied namespace overrides registry entries
    if user_ns !== nothing
        for (name, value) in user_ns
            ns[string(name)] = value
        end
    end
    
    # Highest priority: problem variables by name
    for var in variables
        if hasproperty(var, :name)
            name = getfield(var, :name)
            if name !== nothing && !isempty(name)
                ns[name] = var
            end
        end
    end
    
    return ns
end

function _build_ivp(variables::Vector{Operand}; namespace::Union{Nothing, Dict{String, Any}}=Dict{String, Any}())
    vars = copy(variables)
    ns = build_problem_namespace(vars, namespace)
    return IVP(vars, String[], String[], Dict{String, Any}(), ns,
               nothing, nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

function _build_lbvp(variables::Vector{Operand}; namespace::Union{Nothing, Dict{String, Any}}=Dict{String, Any}())
    vars = copy(variables)
    ns = build_problem_namespace(vars, namespace)
    return LBVP(vars, String[], String[], Dict{String, Any}(), ns,
                nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

function _build_nlbvp(variables::Vector{Operand}; namespace::Union{Nothing, Dict{String, Any}}=Dict{String, Any}())
    vars = copy(variables)
    ns = build_problem_namespace(vars, namespace)
    return NLBVP(vars, String[], String[], Dict{String, Any}(), ns,
                 nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

function _build_evp(variables::Vector{Operand}; eigenvalue::Union{Nothing, Symbol}=nothing, namespace::Union{Nothing, Dict{String, Any}}=Dict{String, Any}())
    vars = copy(variables)
    ns = build_problem_namespace(vars, namespace)
    return EVP(vars, String[], String[], Dict{String, Any}(), ns,
               eigenvalue, nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

const _IVP_constructor = _build_ivp
const _LBVP_constructor = _build_lbvp
const _NLBVP_constructor = _build_nlbvp
const _EVP_constructor = _build_evp

function IVP(variables::Vector{Operand}; kwargs...)
    return multiclass_new(IVP, variables; kwargs...)
end

function LBVP(variables::Vector{Operand}; kwargs...)
    return multiclass_new(LBVP, variables; kwargs...)
end

function NLBVP(variables::Vector{Operand}; kwargs...)
    return multiclass_new(NLBVP, variables; kwargs...)
end

function EVP(variables::Vector{Operand}; kwargs...)
    return multiclass_new(EVP, variables; kwargs...)
end

_problem_builder(::Type{IVP}) = _IVP_constructor
_problem_builder(::Type{LBVP}) = _LBVP_constructor
_problem_builder(::Type{NLBVP}) = _NLBVP_constructor
_problem_builder(::Type{EVP}) = _EVP_constructor
_problem_builder(::Type{T}) where {T<:Problem} = error("No problem builder registered for type $(T)")

function _validate_problem_kwargs(kwargs::NamedTuple)
    if haskey(kwargs, :namespace)
        ns = kwargs[:namespace]
        if !(ns isa AbstractDict || ns === nothing)
            throw(ArgumentError("namespace keyword must be a dictionary or nothing"))
        end
    end
end

function _validate_problem_variables(args::Tuple, kwargs::NamedTuple)
    if isempty(args)
        throw(ArgumentError("Problem constructors require a variables vector"))
    end
    variables = args[1]
    if !(variables isa Vector)
        throw(ArgumentError("Problem variables must be provided as a Vector"))
    end
    if !all(var -> var isa Operand, variables)
        throw(ArgumentError("All problem variables must be Operands"))
    end
    _validate_problem_kwargs(kwargs)
end

function dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Problem}
    if length(args) != 1
        throw(ArgumentError("$(T) expects a single Vector{Operand} argument"))
    end
    return (args, kwargs)
end

function dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Problem}
    _validate_problem_variables(args, kwargs)
    return true
end

function dispatch_check(::Type{EVP}, args::Tuple, kwargs::NamedTuple)
    _validate_problem_variables(args, kwargs)
    if haskey(kwargs, :eigenvalue)
        eigen = kwargs[:eigenvalue]
        if !(eigen === nothing || eigen isa Symbol)
            throw(ArgumentError("Eigenvalue keyword must be a Symbol or nothing"))
        end
    end
    return true
end

function invoke_constructor(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Problem}
    builder = _problem_builder(T)
    variables = copy(args[1])
    return builder(variables; kwargs...)
end

# Problem building and manipulation
function add_equation!(problem::Problem, equation::String)
    """Add equation to problem"""
    push!(problem.equations, equation)
end

function add_bc!(problem::Problem, bc::String)
    """Add boundary condition to problem (legacy string interface)"""
    push!(problem.boundary_conditions, bc)
end

# Enhanced boundary condition functions
function add_bc!(problem::Problem, bc::AbstractBoundaryCondition)
    """Add structured boundary condition to problem"""
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

function add_dirichlet_bc!(problem::Problem, field::String, coordinate::String, 
                          position, value; kwargs...)
    """Add Dirichlet boundary condition: u(coord=pos) = value"""
    bc = add_dirichlet!(problem.bc_manager, field, coordinate, position, value; kwargs...)
    
    # Add to legacy string list
    eq = bc_to_equation(problem.bc_manager, bc)
    push!(problem.boundary_conditions, eq)
    
    return bc
end

function add_neumann_bc!(problem::Problem, field::String, coordinate::String,
                        position, value; kwargs...)
    """Add Neumann boundary condition: du/dcoord(coord=pos) = value"""
    bc = add_neumann!(problem.bc_manager, field, coordinate, position, value; kwargs...)
    
    # Add to legacy string list
    eq = bc_to_equation(problem.bc_manager, bc)
    push!(problem.boundary_conditions, eq)
    
    return bc
end

function add_robin_bc!(problem::Problem, field::String, coordinate::String,
                      position, alpha, beta, value; kwargs...)
    """Add Robin boundary condition: alpha*u + beta*du/dcoord = value"""
    bc = add_robin!(problem.bc_manager, field, coordinate, position, alpha, beta, value; kwargs...)
    
    # Add to legacy string list  
    eq = bc_to_equation(problem.bc_manager, bc)
    push!(problem.boundary_conditions, eq)
    
    return bc
end

function add_stress_free_bc!(problem::Problem, velocity_field::String, 
                            coordinate::String, position; kwargs...)
    """Add stress-free boundary condition for velocity field"""
    bc = add_stress_free!(problem.bc_manager, velocity_field, coordinate, position; kwargs...)
    
    # Add to legacy string list
    eqs = bc_to_equation(problem.bc_manager, bc)
    append!(problem.boundary_conditions, eqs)
    
    return bc
end

function register_tau_field!(problem::Problem, name::String, field)
    """Register tau field for boundary condition enforcement"""
    register_tau_field!(problem.bc_manager, name, field)
    
    # Add to problem variables if not already present
    if !any(var -> (isa(var, ScalarField) && var.name == name), problem.variables)
        push!(problem.variables, field)
    end
    
    return problem
end

function set_parameter!(problem::Problem, name::String, value)
    """Set parameter value"""
    problem.parameters[name] = value
end

function get_parameter(problem::Problem, name::String, default=nothing)
    """Get parameter value"""
    return get(problem.parameters, name, default)
end

# Equation parsing following Dedalus structure
function parse_equation(equation::String, namespace::Dict{String, Any})
    """
    Parse equation string into operator expressions following Dedalus formulation requirements.
    
    Dedalus requires:
    - LHS: Linear terms only (first-order in time derivatives, linear in variables)
    - RHS: Nonlinear terms, time-dependent terms, non-constant coefficients
    - Form: M·∂ₜX + L·X = F(X,t)
    
    Following Dedalus problems.py:add_equation pattern (problems.py:65-80).
    """
    
    try
        # Split equation into LHS and RHS expressions
        if isa(equation, String)
            # Parse string-valued equations following Dedalus pattern
            LHS_str, RHS_str = split_equation(equation)
            
            # Parse LHS (should contain only linear terms)
            LHS = parse_linear_expression(LHS_str, namespace)
            
            # Parse RHS (can contain nonlinear terms)
            RHS = parse_expression(RHS_str, namespace)
            
            # Validate equation structure following Dedalus requirements
            validate_equation_structure(LHS, RHS, equation)
            
            return LHS, RHS
        else
            throw(ArgumentError("Equation must be a string"))
        end
        
    catch e
        @error "Failed to parse equation: $equation" exception=e
        # Return fallback expressions
        return equation, "0"
    end
end

function parse_linear_expression(expr_str::String, namespace::Dict{String, Any})
    """
    Parse LHS expression ensuring it contains only linear terms.
    
    Linear terms allowed on LHS:
    - Time derivatives (first-order): dt(u)
    - Linear spatial derivatives: d(u, x), lap(u)
    - Constant coefficients: 2*dt(u), -viscosity*lap(u)
    - Linear combinations: dt(u) + diffusion*lap(u)
    
    Non-constant coefficients should be moved to RHS.
    """
    
    expr = parse_expression(expr_str, namespace)
    
    # Validate linearity (this would need proper implementation)
    if !is_linear_expression(expr, namespace)
        @warn "Non-linear terms detected on LHS of equation. Consider moving to RHS: $expr_str"
        @warn "Dedalus requires: Linear terms (LHS) = Nonlinear terms (RHS)"
    end
    
    return expr
end

function is_linear_expression(expr, namespace::Dict{String, Any})
    """
    Check if expression contains only linear terms suitable for LHS.
    
    This is a simplified check - full implementation would need to:
    1. Verify all terms are linear in the dependent variables
    2. Check that time derivatives are first-order only
    3. Ensure coefficients are constant (non-constant coefficients go to RHS)
    """
    
    # Simple heuristic checks
    if isa(expr, ZeroOperator) || isa(expr, ConstantOperator)
        return true
    elseif isa(expr, TimeDerivative)
        # Time derivatives are linear if the field is linear
        return expr.order == 1  # First-order time derivatives only
    elseif isa(expr, Differentiate) || isa(expr, Laplacian) || isa(expr, Gradient)
        # Spatial derivatives are linear
        return true
    elseif isa(expr, AddOperator) || isa(expr, SubtractOperator)
        # Linear combinations are okay
        return is_linear_expression(expr.left, namespace) && is_linear_expression(expr.right, namespace)
    elseif isa(expr, MultiplyOperator)
        # Multiplication is linear if one factor is constant
        left_constant = is_constant_coefficient(expr.left, namespace)
        right_constant = is_constant_coefficient(expr.right, namespace)
        return left_constant || right_constant
    else
        # Other operations (powers, products of variables, etc.) are nonlinear
        return false
    end
end

function is_constant_coefficient(expr, namespace::Dict{String, Any})
    """
    Check if expression represents a constant coefficient.
    Non-constant coefficients should be moved to RHS per Dedalus requirements.
    """
    
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

function validate_equation_structure(LHS, RHS, original_equation::String)
    """
    Validate that equation follows Dedalus structure requirements.
    
    Dedalus requirements:
    1. LHS must be linear in dependent variables
    2. LHS must be first-order in time derivatives  
    3. RHS can contain nonlinear terms
    4. Non-constant coefficients should be on RHS
    """
    
    # Check for temporal derivatives on RHS (not allowed)
    if contains_time_derivatives(RHS)
        throw(ArgumentError("Time derivatives found on RHS of equation: $original_equation. " *
                           "Dedalus requires all time derivatives on LHS."))
    end
    
    # Check for proper linear structure on LHS
    if !is_proper_lhs_structure(LHS)
        @warn "LHS may not follow Dedalus linear structure: $original_equation"
        @warn "Ensure LHS contains only: dt(vars), linear spatial derivatives, constant coefficients"
    end
    
    return true
end

function contains_time_derivatives(expr)
    """Check if expression tree contains any time derivatives."""
    if isa(expr, TimeDerivative)
        return true
    elseif isa(expr, AddOperator) || isa(expr, SubtractOperator) || isa(expr, MultiplyOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, NegateOperator)
        return contains_time_derivatives(expr.operand)
    else
        return false
    end
end

function is_proper_lhs_structure(expr)
    """
    Check if LHS has proper structure for Dedalus matrix formulation.
    Should be of the form: M·∂ₜX + L·X where M and L are linear operators.
    """
    
    # This is a simplified check - full implementation would verify
    # the mathematical structure more rigorously
    return is_linear_expression(expr, Dict{String, Any}())
end

function parse_expression(expr_str::String, namespace::Dict{String, Any})
    """
    Parse expression string into operator tree following Dedalus patterns.
    Uses Julia's Meta.parse for proper AST parsing and operator precedence handling.
    
    This function evaluates mathematical expressions similar to how Dedalus uses
    eval(string, namespace) in problems.py:73-74.
    """
    
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
        if isa(e, ParseError)
            @warn "Parse error in expression '$expr_str': $(e.msg)"
            return UnknownOperator(expr_str)
        else
            # For evaluation errors, try fallback parsing patterns
            @debug "Error evaluating expression '$expr_str': $e, trying fallback patterns"
            return fallback_parse_expression(expr_str, namespace)
        end
    end
end

function evaluate_parsed_expression(expr, namespace::Dict{String, Any})
    """
    Recursively evaluate parsed expression with namespace substitution.
    Similar to Dedalus eval(string, namespace) but with proper Julia AST handling.
    """
    
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
            if func_expr isa Symbol
                if func_expr == :dt
                    if isempty(arg_exprs)
                        throw(ArgumentError("dt requires at least one argument"))
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
                    if length(arg_exprs) < 3
                        throw(ArgumentError("lift requires operand, basis, and polynomial index"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    basis = evaluate_parsed_expression(arg_exprs[2], namespace)
                    n_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                    return lift(operand, basis, Int(round(n_val)))
                end
            end
            
            func = evaluate_parsed_expression(func_expr, namespace)
            evaluated_args = [coerce_constant_value(evaluate_parsed_expression(arg, namespace)) for arg in arg_exprs]
            
            if isa(func, Function)
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

function fallback_parse_expression(expr_str::String, namespace::Dict{String, Any})
    """
    Fallback parsing for expressions that fail Meta.parse evaluation.
    Uses simple string pattern matching as backup.
    """
    
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
    if startswith(expr_str, "dt(") && endswith(expr_str, ")")
        field_name = expr_str[4:end-1]
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

# Helper operator types for parsing (following Dedalus operator structure)
struct ZeroOperator <: Operator end
struct ConstantOperator <: Operator
    value::Float64
end
struct UnknownOperator <: Operator
    expression::String
end

@inline coerce_constant_value(x) = x isa ConstantOperator ? x.value : x

# Arithmetic operators (these would be replaced by proper operator implementations)
struct AddOperator <: Operator
    left::Any
    right::Any
end

struct SubtractOperator <: Operator
    left::Any
    right::Any
end

struct MultiplyOperator <: Operator
    left::Any
    right::Any
end

struct DivideOperator <: Operator
    left::Any
    right::Any
end

struct PowerOperator <: Operator
    base::Any
    exponent::Any
end

struct NegateOperator <: Operator
    operand::Any
end

struct IndexOperator <: Operator
    array::Any
    indices::Vector{Any}
end

function build_matrices(problem::Problem)
    """
    Build system matrices for problem following Dedalus structure.
    Following subsystems.py:build_subproblem_matrices (subsystems.py:72-81) and
    Subproblem.build_matrices (subsystems.py:497-576).
    """
    
    if length(problem.equations) == 0
        throw(ArgumentError("No equations specified"))
    end
    
    # Build matrix expressions from equations (following problems.py:_build_matrix_expressions)
    build_matrix_expressions!(problem)
    
    # Compute field sizes for each equation and variable
    eqn_sizes = [compute_field_size(eq_data) for eq_data in problem.equation_data]
    var_sizes = [compute_field_size(var) for var in problem.variables]
    
    I = sum(eqn_sizes)  # Total rows
    J = sum(var_sizes)  # Total columns
    dtype = ComplexF64   # Problem dtype
    
    @debug "Building matrices: equations=$I, variables=$J"
    
    # Matrix names to build (following Dedalus convention)
    matrix_names = ["M", "L"]  # M = mass matrix, L = stiffness matrix
    
    # Build sparse matrices following Dedalus subsystems.py:513-537 pattern
    matrices = Dict{String, Any}()
    for name in matrix_names
        # Collect sparse matrix entries
        data, rows, cols = Float64[], Int[], Int[]
        
        i0 = 0  # Row offset
        for (eq_idx, eq_data) in enumerate(problem.equation_data)
            eqn_size = eqn_sizes[eq_idx]
            if eqn_size > 0 && check_equation_condition(eq_data)
                # Get expression matrix blocks for this equation
                expr = get_matrix_expression(eq_data, name)
                if expr !== nothing && !is_zero_expression(expr)
                    # Build expression matrices for each variable
                    j0 = 0  # Column offset
                    for (var_idx, var) in enumerate(problem.variables)
                        var_size = var_sizes[var_idx]
                        if var_size > 0
                            # Get matrix block for this variable
                            block = build_expression_matrix_block(expr, var, eqn_size, var_size)
                            if !isempty(block.nzval)
                                # Add to sparse matrix data
                                append!(data, block.nzval)
                                append!(rows, i0 .+ block.rowval)
                                append!(cols, j0 .+ block.colval)
                            end
                        end
                        j0 += var_size
                    end
                end
            end
            i0 += eqn_size
        end
        
        # Create sparse matrix
        if !isempty(data)
            # Filter small entries (following Dedalus entry_cutoff pattern)
            entry_cutoff = 1e-14
            significant = abs.(data) .>= entry_cutoff
            data = data[significant]
            rows = rows[significant]
            cols = cols[significant]
            
            matrices[name] = sparse(rows, cols, data, I, J)
        else
            # Empty matrix
            matrices[name] = spzeros(ComplexF64, I, J)
        end
        
        @debug "Built matrix $name: size=($I, $J), nnz=$(nnz(matrices[name]))"
    end
    
    # Build forcing vector (RHS terms)
    F_vector = build_forcing_vector(problem, eqn_sizes, I)
    
    # Return matrices in standard format
    L_matrix = matrices["L"]
    M_matrix = matrices["M"] 
    
    @info "Matrix building completed: L=$(size(L_matrix)), M=$(size(M_matrix)), F=$(length(F_vector))"
    
    return L_matrix, M_matrix, F_vector
end

function build_matrix_expressions!(problem::Problem)
    """
    Build matrix expressions from parsed equations.
    Following Dedalus problems.py:_build_matrix_expressions patterns.
    """
    
    problem.equation_data = []
    
    for (i, equation_str) in enumerate(problem.equations)
        try
            # Parse equation
            lhs, rhs = parse_equation(equation_str, problem.namespace)
            
            # Build matrix expressions following problem type
            eq_data = build_equation_expressions(lhs, rhs, problem.variables)
            eq_data["equation_index"] = i
            eq_data["equation_string"] = equation_str
            if !haskey(eq_data, "equation_size")
                vars = get(eq_data, "variables", problem.variables)
                if isa(vars, Vector)
                    eq_data["equation_size"] = sum(field_dofs(var) for var in vars)
                end
            end

            push!(problem.equation_data, eq_data)
            
        catch e
            @error "Failed to build matrix expressions for equation $i: $equation_str" exception=e
            # Create fallback equation data
            fallback_data = Dict(
                "M" => nothing,
                "L" => UnknownOperator(equation_str),
                "F" => ZeroOperator(),
                "equation_index" => i,
                "equation_string" => equation_str,
                "equation_size" => 0
            )
            push!(problem.equation_data, fallback_data)
        end
    end
end

function build_equation_expressions(lhs, rhs, variables::Vector)
    """
    Build matrix expressions from LHS and RHS operators.
    Following Dedalus _build_matrix_expressions patterns.
    """
    
    eq_data = Dict{String, Any}()
    
    # Split LHS into mass matrix (time derivatives) and stiffness matrix (spatial) terms
    # Following IVP pattern: M.dt(X) + L.X = F (problems.py:328)
    M_terms, L_terms = split_time_spatial_operators(lhs)
    
    # Store matrix expressions
    eq_data["M"] = combine_operators(M_terms)      # Mass matrix terms
    eq_data["L"] = combine_operators(L_terms)      # Stiffness matrix terms  
    eq_data["F"] = rhs                             # Forcing terms
    
    # Metadata
    eq_data["variables"] = variables
    eq_data["lhs"] = lhs
    eq_data["rhs"] = rhs
    
    return eq_data
end

function split_time_spatial_operators(operator)
    """
    Split operator into time derivative (mass matrix) and spatial (stiffness) terms.
    Following Dedalus operators split pattern.
    """
    
    M_terms = []  # Time derivative terms
    L_terms = []  # Spatial terms
    
    if isa(operator, TimeDerivative)
        # Pure time derivative
        push!(M_terms, operator)
        
    elseif isa(operator, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Pure spatial operator
        push!(L_terms, operator)
        
    elseif isa(operator, AddOperator)
        # Split addition terms recursively
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)
        append!(M_terms, left_M)
        append!(L_terms, left_L)
        append!(M_terms, right_M)
        append!(L_terms, right_L)
        
    elseif isa(operator, SubtractOperator)
        # Split subtraction terms recursively (with sign handling)
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)
        append!(M_terms, left_M)
        append!(L_terms, left_L)
        # Right terms get negative sign (simplified)
        append!(M_terms, right_M)
        append!(L_terms, right_L)
        
    elseif hasfield(typeof(operator), :name)
        # Direct variable reference -> identity in L
        push!(L_terms, operator)
        
    else
        # Other operators go to L by default
        push!(L_terms, operator)
    end
    
    return M_terms, L_terms
end

function combine_operators(terms::Vector)
    """Combine operator terms into single expression"""
    if isempty(terms)
        return ZeroOperator()
    elseif length(terms) == 1
        return terms[1]
    else
        # Combine with addition
        result = terms[1]
        for i in 2:length(terms)
            result = AddOperator(result, terms[i])
        end
        return result
    end
end

# Supporting functions for matrix building

function field_dofs(field::ScalarField)
    if field.data_c !== nothing
        return length(field.data_c)
    elseif field.data_g !== nothing
        return length(field.data_g)
    else
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    end
end

field_dofs(field::VectorField) = sum(field_dofs(comp) for comp in field.components)
field_dofs(field::TensorField) = sum(field_dofs(comp) for comp in vec(field.components))

function compute_field_size(field_or_data)
    """Compute size (degrees of freedom) of field or equation data"""
    if isa(field_or_data, Dict)
        if haskey(field_or_data, "equation_size")
            return field_or_data["equation_size"]
        elseif haskey(field_or_data, "variables")
            vars = field_or_data["variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        end
        return 0
    elseif isa(field_or_data, ScalarField)
        return field_dofs(field_or_data)
    elseif isa(field_or_data, VectorField) || isa(field_or_data, TensorField)
        return field_dofs(field_or_data)
    elseif hasfield(typeof(field_or_data), :data_c) && field_or_data.data_c !== nothing
        return length(field_or_data.data_c)
    elseif hasfield(typeof(field_or_data), :data_g) && field_or_data.data_g !== nothing
        return length(field_or_data.data_g)
    else
        return 0
    end
end

function check_equation_condition(eq_data::Dict)
    """Check if equation should be included in matrix"""
    # For now, include all equations
    return true
end

function get_matrix_expression(eq_data::Dict, matrix_name::String)
    """Get matrix expression from equation data"""
    return get(eq_data, matrix_name, nothing)
end

function is_zero_expression(expr)
    """Check if expression is effectively zero"""
    return isa(expr, ZeroOperator) || expr === nothing
end

function build_expression_matrix_block(expr, var, eqn_size::Int, var_size::Int)
    """
    Build matrix block for expression acting on variable.
    Following Dedalus expression_matrices pattern.
    """
    
    if isa(expr, TimeDerivative) && expr.operand === var
        # Time derivative of this variable -> identity block
        return sparse(I, var_size)
        
    elseif isa(expr, Laplacian) && expr.operand === var
        # Laplacian of this variable -> Laplacian matrix (simplified)
        return sparse(-I, var_size)  # Negative Laplacian
        
    elseif isa(expr, Union{Gradient, Divergence, Differentiate}) && expr.operand === var
        # Other spatial operators -> identity (simplified)
        return sparse(I, var_size)
        
    elseif expr === var
        # Direct variable reference -> identity
        return sparse(I, var_size)
        
    elseif isa(expr, AddOperator)
        # Sum of operators
        left_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        right_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        return left_block + right_block
        
    elseif isa(expr, SubtractOperator)
        # Difference of operators
        left_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        right_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        return left_block - right_block
        
    elseif isa(expr, MultiplyOperator) && isa(expr.right, ConstantOperator)
        # Constant multiplication
        coeff = expr.right.value
        base_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        return coeff * base_block
        
    elseif isa(expr, ConstantOperator)
        # Constant expression -> zero block (constants don't depend on variables)
        return spzeros(ComplexF64, eqn_size, var_size)
        
    elseif isa(expr, ZeroOperator)
        # Zero expression -> zero block
        return spzeros(ComplexF64, eqn_size, var_size)
        
    else
        # Unknown expression -> zero block
        @debug "Unknown expression type for matrix block: $(typeof(expr))"
        return spzeros(ComplexF64, eqn_size, var_size)
    end
end

function build_forcing_vector(problem::Problem, eqn_sizes::Vector{Int}, total_size::Int)
    """Build forcing vector from RHS terms"""
    
    F_vector = zeros(ComplexF64, total_size)
    
    i0 = 0
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        eqn_size = eqn_sizes[eq_idx]
        if eqn_size > 0
            rhs_expr = get(eq_data, "F", ZeroOperator())
            
            # Evaluate RHS expression to get forcing values
            if isa(rhs_expr, ConstantOperator)
                F_vector[i0+1:i0+eqn_size] .= rhs_expr.value
            elseif isa(rhs_expr, ZeroOperator)
                F_vector[i0+1:i0+eqn_size] .= 0.0
            else
                # Complex RHS expressions would need proper evaluation
                @debug "Complex RHS expression not fully supported: $(typeof(rhs_expr))"
                F_vector[i0+1:i0+eqn_size] .= 0.0
            end
        end
        i0 += eqn_size
    end
    
    return F_vector
end

# Legacy functions (kept for compatibility)

function process_lhs_operator!(L_matrix::Matrix, M_matrix::Matrix, lhs_op, eq_idx::Int, variables::Vector)
    """
    Process LHS operator and extract contributions to system matrices.
    Following Dedalus pattern where time derivatives go to M_matrix,
    spatial operators go to L_matrix.
    """
    
    if isa(lhs_op, TimeDerivative)
        # Time derivative terms go to mass matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            M_matrix[eq_idx, var_idx] = 1.0
        else
            @debug "Unknown variable in time derivative"
        end
        
    elseif isa(lhs_op, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Spatial operators go to linear operator matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            # Coefficient would depend on operator type and discretization
            # For now, use placeholder values
            if isa(lhs_op, Laplacian)
                L_matrix[eq_idx, var_idx] = -1.0  # Typical Laplacian sign
            else
                L_matrix[eq_idx, var_idx] = 1.0
            end
        else
            @debug "Unknown variable in spatial operator"
        end
        
    elseif isa(lhs_op, AddOperator)
        # Recursively process addition terms
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.right, eq_idx, variables)
        
    elseif isa(lhs_op, SubtractOperator)
        # Process left term normally, right term with negative sign
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        # Would need to negate contributions from right side
        # This requires more sophisticated matrix coefficient tracking
        @debug "Subtraction operator needs more sophisticated handling"
        
    elseif isa(lhs_op, MultiplyOperator)
        # Handle coefficient multiplication
        if isa(lhs_op.right, ConstantOperator)
            coeff = lhs_op.right.value
            # Apply coefficient to left operand contributions
            # This would require modifying matrix entries by coefficient
            @debug "Coefficient multiplication needs coefficient tracking: $coeff"
            process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        else
            @debug "General multiplication not yet supported"
        end
        
    elseif hasfield(typeof(lhs_op), :name) && haskey(variables, lhs_op.name)
        # Direct variable reference
        var_idx = find_variable_index(lhs_op, variables)
        if var_idx !== nothing
            L_matrix[eq_idx, var_idx] = 1.0
        end
        
    elseif isa(lhs_op, ZeroOperator)
        # Zero contribution
        nothing
        
    elseif isa(lhs_op, ConstantOperator)
        # Constant terms shouldn't appear in LHS typically
        @debug "Constant term in LHS: $(lhs_op.value)"
        
    else
        @debug "Unhandled LHS operator type: $(typeof(lhs_op))"
    end
end

function process_rhs_operator!(F_vector::Vector, rhs_op, eq_idx::Int, variables::Vector)
    """
    Process RHS operator and extract contributions to forcing vector.
    Following Dedalus pattern where RHS represents known terms/forcing.
    """
    
    if isa(rhs_op, ConstantOperator)
        # Constant forcing term
        F_vector[eq_idx] = rhs_op.value
        
    elseif isa(rhs_op, ZeroOperator)
        # Zero RHS (homogeneous equation)
        F_vector[eq_idx] = 0.0
        
    elseif isa(rhs_op, AddOperator)
        # Sum of RHS terms
        # For now, simplified handling
        @debug "Addition in RHS needs proper evaluation"
        
    elseif isa(rhs_op, String) && (rhs_op == "0" || rhs_op == "zero")
        # String representation of zero
        F_vector[eq_idx] = 0.0
        
    else
        @debug "Unhandled RHS operator type: $(typeof(rhs_op)), using zero"
        F_vector[eq_idx] = 0.0
    end
end

function find_variable_index(operand, variables::Vector)
    """Find index of variable in problem variable list"""
    
    # Handle direct variable reference
    for (i, var) in enumerate(variables)
        if operand === var
            return i
        end
    end
    
    # Handle by name if operand has name field
    if hasfield(typeof(operand), :name)
        for (i, var) in enumerate(variables)
            if hasfield(typeof(var), :name) && operand.name == var.name
                return i
            end
        end
    end
    
    return nothing
end

# Domain setup
function setup_domain!(problem::Problem)
    """Setup domain for problem based on variables"""
    
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
            @warn "Variable $(var.name) has incompatible domain"
        end
    end
end

# Problem validation
function validate_problem(problem::Problem)
    """Validate problem formulation"""
    
    errors = String[]
    
    if length(problem.variables) == 0
        push!(errors, "No variables specified")
    end
    
    if length(problem.equations) == 0
        push!(errors, "No equations specified")
    end
    
    if length(problem.equations) != length(problem.variables)
        push!(errors, "Number of equations ($(length(problem.equations))) does not match number of variables ($(length(problem.variables)))")
    end
    
    # Check for required boundary conditions in boundary value problems
    if isa(problem, LBVP) || isa(problem, NLBVP)
        if length(problem.boundary_conditions) == 0 && length(problem.bc_manager.conditions) == 0
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
function add_substitution!(problem::Problem, name::String, expression)
    """Add substitution to problem namespace"""
    problem.namespace[name] = expression
end

function expand_substitutions!(problem::Problem)
    """
    Expand substitutions in equations following Dedalus pattern.
    Following Dedalus expand(*vars) methods (arithmetic.py:319-329, operators.py:704-739).
    """
    
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

function expand_expression(expr, variables::Vector)
    """
    Expand expression over specified variables following Dedalus pattern.
    Following operators.py:expand and arithmetic.py:expand methods.
    """
    
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

function has_variables(expr, variables::Vector)
    """Check if expression contains any of the specified variables"""
    
    if expr === nothing || isa(expr, String)
        return false
    end
    
    # Check direct variable reference
    for var in variables
        if expr === var
            return true
        end
    end
    
    # Check by name if possible
    if hasfield(typeof(expr), :name)
        for var in variables
            if hasfield(typeof(var), :name) && expr.name == var.name
                return true
            end
        end
    end
    
    # Recursively check compound expressions
    if isa(expr, Union{AddOperator, SubtractOperator, MultiplyOperator})
        return has_variables(expr.left, variables) || has_variables(expr.right, variables)
    elseif isa(expr, Union{TimeDerivative, Laplacian, Gradient, Divergence, Differentiate})
        return has_variables(expr.operand, variables)
    end
    
    return false
end

function combine_add_expressions(left, right)
    """Combine two expressions with addition, flattening nested additions"""
    
    if isa(left, ZeroOperator)
        return right
    elseif isa(right, ZeroOperator)
        return left
    else
        return AddOperator(left, right)
    end
end

function expand_multiply_expressions(left, right, variables::Vector)
    """
    Expand multiplication with distribution over addition.
    Following arithmetic.py:expand multiplication pattern.
    """
    
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

function distribute_operator_over_operand(operator, expanded_operand, variables::Vector)
    """
    Distribute operator over expanded operand.
    Following operators.py:_expand_add pattern.
    """
    
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

function create_similar_operator(operator, new_operand)
    """Create new operator of same type with different operand"""
    
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

function reconstruct_equation_string(lhs, rhs)
    """Reconstruct equation string from expanded expressions"""
    
    lhs_str = expression_to_string(lhs)
    rhs_str = expression_to_string(rhs)
    
    return "$lhs_str = $rhs_str"
end

function expression_to_string(expr)
    """Convert expression back to string representation"""
    
    if isa(expr, String)
        return expr
    elseif isa(expr, ZeroOperator)
        return "0"
    elseif isa(expr, ConstantOperator)
        return string(expr.value)
    elseif isa(expr, AddOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str + $right_str)"
    elseif isa(expr, SubtractOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str - $right_str)"
    elseif isa(expr, MultiplyOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str * $right_str)"
    elseif isa(expr, TimeDerivative)
        operand_str = expression_to_string(expr.operand)
        return "dt($operand_str)"
    elseif isa(expr, Laplacian)
        operand_str = expression_to_string(expr.operand)
        return "lap($operand_str)"
    elseif hasfield(typeof(expr), :name)
        return expr.name
    else
        return string(expr)
    end
end

function expand_namespace_substitutions!(problem::Problem)
    """Expand any substitution definitions in problem namespace"""
    
    # Look for substitution patterns in namespace
    substitutions = Dict{String, Any}()
    
    for (key, value) in problem.namespace
        if isa(value, String) && contains(value, "=")
            # Potential substitution definition
            try
                lhs, rhs = split_equation(value)
                substitutions[strip(lhs)] = strip(rhs)
            catch
                # Not a valid substitution, skip
            end
        end
    end
    
    if !isempty(substitutions)
        @debug "Found substitutions in namespace" substitutions
        
        # Apply substitutions to equations (simplified)
        # This would need more sophisticated pattern matching in practice
        for (i, equation) in enumerate(problem.equations)
            modified_equation = equation
            for (old_pattern, new_pattern) in substitutions
                modified_equation = replace(modified_equation, old_pattern => new_pattern)
            end
            if modified_equation != equation
                problem.equations[i] = modified_equation
                @debug "Applied substitution to equation $i" original=equation modified=modified_equation
            end
        end
    end
end

# Problem metadata
function get_variable_names(problem::Problem)
    """Get names of all variables"""
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

function get_equation_count(problem::Problem)
    """Get total number of equations including boundary conditions"""
    return length(problem.equations) + length(problem.boundary_conditions)
end

function get_variable_count(problem::Problem)
    """Get total number of scalar variables"""
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
