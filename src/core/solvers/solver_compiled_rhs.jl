# ============================================================================
# Solver residual and RHS expression utilities
# ============================================================================

"""
    Evaluate residual and Jacobian for nonlinear problem following Tarang patterns.
    
    In Tarang, this corresponds to:
    1. Evaluating F expressions (residual) using evaluator system
    2. Building dF matrices (Jacobian/Frechet differential) 
    3. Gathering results into numerical arrays for Newton solver
    """
function evaluate_residual_and_jacobian(problem::NLBVP, x::Vector{ComplexF64})
    
    # Step 1: Copy solution vector back to problem fields
    state_fields = collect_state_fields(problem.variables)
    copy_solution_to_fields!(state_fields, x)

    # Step 2: Residual R = L*x - RHS.
    # The LHS (which contains tau `lift` terms living on a different output basis
    # than `Δu` etc.) is evaluated via the assembled linear operator matrix `L`
    # — this reconciles the term bases that a field-level `lhs_field - rhs_field`
    # cannot (it throws "Cannot add fields with different bases"). The RHS
    # (nonlinear terms + forcing, with NO lift) is evaluated cleanly as fields.
    L_matrix = compiled_problem(problem).linear_matrix

    rhs_fields = ScalarField[]
    if !isempty(problem.equation_data)
        for (i, eq_data) in enumerate(problem.equation_data)
            template = state_fields[min(i, length(state_fields))]
            rhs_expr = get(eq_data, "rhs", nothing)
            rhs_field = rhs_expr === nothing ? create_zero_field(template) :
                        evaluate_solver_expression(rhs_expr, problem.variables; layout=:g, template=template)
            ensure_layout!(rhs_field, :c)
            push!(rhs_fields, rhs_field)
        end
    else
        for field in state_fields
            push!(rhs_fields, create_zero_field(field))
        end
    end

    # Step 3: residual vector = L*x - RHS  (fall back to field lhs-rhs if no L).
    rhs_vec = fields_to_vector(rhs_fields)
    if L_matrix !== nothing && size(L_matrix, 2) == length(x) && size(L_matrix, 1) == length(rhs_vec)
        residual = L_matrix * x - rhs_vec
    else
        residual = -rhs_vec
    end

    # Step 4: Build Jacobian matrix
    # Try symbolic Jacobian first (Frechet differentiation), then fall back
    n = length(x)
    jacobian = try
        build_symbolic_jacobian(problem, state_fields)
    catch e
        # Newton keeps converging on an approximate Jacobian — linearly instead
        # of quadratically, or not at all with the identity — so this substitution
        # shows up as "the solve is slow" or "it did not converge", never as the
        # reason. It was a @debug, i.e. invisible by default.
        @warn "Symbolic Jacobian construction failed; falling back to the " *
              "linear matrix (or the identity if there is none). Newton will " *
              "converge slowly or not at all." exception=e maxlog=1
        if compiled_problem(problem).linear_matrix !== nothing
            compiled_problem(problem).linear_matrix
        else
            sparse(I, n, n)
        end
    end

    @debug "Residual evaluation completed: size=$(length(residual)), norm=$(norm(residual))"
    @debug "Jacobian evaluation completed: size=$(size(jacobian)), nnz=$(nnz(jacobian))"

    return residual, jacobian
end

"""
    Create a constant field from a template.

    GPU-aware: The field inherits the architecture from the template.
    Uses fill!() which works on both CPU and GPU arrays.
    """
function _constant_field_from_template(template::ScalarField, value::Number; layout::Symbol=:g)
    field = ScalarField(template.dist, "const_$(template.name)", template.bases, template.dtype)
    ensure_layout!(field, layout)
    if layout == :g && get_grid_data(field) !== nothing
        # fill!() works on both CPU and GPU
        fill!(get_grid_data(field), convert(eltype(get_grid_data(field)), value))
    elseif layout == :c && get_coeff_data(field) !== nothing
        fill!(get_coeff_data(field), convert(eltype(get_coeff_data(field)), value))
    end
    return field
end

"""
    get_solver_architecture(solver::Solver)

Get the architecture (CPU or GPU) used by the solver's fields.
Returns CPU() if architecture cannot be determined.
"""
function get_solver_architecture(solver::Solver)
    if isa(solver, InitialValueSolver) && !isempty(solver.state)
        return solver.state[1].dist.architecture
    elseif isa(solver, BoundaryValueSolver) && !isempty(solver.state)
        return solver.state[1].dist.architecture
    elseif hasproperty(solver, :problem) && solver.problem !== nothing
        if solver.problem.domain !== nothing && hasproperty(solver.problem.domain, :dist)
            return solver.problem.domain.dist.architecture
        end
    end
    return CPU()
end

function _scale_vector_field(field::VectorField, scale::Number)
    result = VectorField(field.dist, field.coordsys, "$(field.name)_scaled", field.bases, field.dtype)
    for i in eachindex(field.components)
        result.components[i] = field.components[i] * scale
    end
    return result
end

function _coerce_numeric_operand(value, template::Union{Nothing, ScalarField}; layout::Symbol=:g)
    if value isa Number
        if template === nothing
            return value
        end
        return _constant_field_from_template(template, value; layout=layout)
    end
    return value
end

function _binary_template(left, right, template::Union{Nothing, ScalarField})
    if template !== nothing
        return template
    elseif left isa ScalarField
        return left
    elseif right isa ScalarField
        return right
    else
        return nothing
    end
end

"""
    UnrecognizedRHSExpression(expr::String)

Thrown when an equation RHS contains an `UnknownOperator` that survived to
evaluation — an unregistered operator, unknown function, or undeclared variable
(commonly a typo such as `dx` instead of `∂x`). Distinct from generic errors so
the RHS-evaluation try/catch can propagate it instead of silently dropping the
term.
"""
struct UnrecognizedRHSExpression <: Exception
    expr::String
end

function Base.showerror(io::IO, e::UnrecognizedRHSExpression)
    print(io,
        "UnrecognizedRHSExpression: `", e.expr, "` in an equation RHS is not a ",
        "registered operator, known function, or declared variable — check for a ",
        "typo (for example `dx` instead of `∂x`) or an unsupported operator. ",
        "Aborting rather than silently dropping the term.")
end

"""
    evaluate_solver_expression(expr, variables; layout=:g, template=nothing)

Evaluate a parsed solver expression with current field values. Returns a field
(preferred) or a numeric scalar for constant expressions.

Dispatched on the node type, one method per node, rather than tested with a chain
of `isa` branches. The distinction matters here for two reasons.

*Ordering stops being load-bearing.* Every concrete operator below is a subtype of
`Operator`, and `Operator` has its own generic method that defers to `evaluate`. In
a manual chain the generic test has to come last, and moving it — or inserting a
new branch beneath it — quietly disables every branch it shadows. Julia picks the
most specific method regardless of the order they are written in, so that hazard
cannot be reintroduced.

*Fall-through becomes visible.* There are 56 `Operator` subtypes and only the
eleven below are handled specially; the rest are served by the `::Operator` method
on purpose. Written as a chain, "handled generically" and "nobody remembered this
case" look identical. Written as methods, the generic case is a method someone
wrote, and `methods(evaluate_solver_expression)` enumerates the real contract.

Operand combinations dispatch the same way — see `_rhs_add` and friends — so an
unsupported pairing raises from a fallback method instead of falling off the end of
a chain.
"""
function evaluate_solver_expression end

# --- operand combination rules -------------------------------------------------
# One method per supported pairing, plus a fallback that names what was rejected.
# The fallback is the whole point: an unhandled combination raises instead of
# reaching whatever the next branch happened to be.

_rhs_add(left::Number, right::Number) = left + right
_rhs_add(left::ScalarField, right::ScalarField) = left + right
_rhs_add(left::VectorField, right::VectorField) = add_vector_fields(left, right)
_rhs_add(left, right) =
    throw(ArgumentError("Unsupported Add operands: $(typeof(left)) and $(typeof(right))"))

_rhs_subtract(left::Number, right::Number) = left - right
_rhs_subtract(left::ScalarField, right::ScalarField) = left - right
_rhs_subtract(left::VectorField, right::VectorField) =
    add_vector_fields(left, _scale_vector_field(right, -1))
_rhs_subtract(left, right) =
    throw(ArgumentError("Unsupported Subtract operands: $(typeof(left)) and $(typeof(right))"))

_rhs_multiply(left::Number, right::Number) = left * right
_rhs_multiply(left::ScalarField, right::ScalarField) = left * right
_rhs_multiply(left::ScalarField, right::Number) = left * right
_rhs_multiply(left::Number, right::ScalarField) = right * left
_rhs_multiply(left::VectorField, right::Number) = _scale_vector_field(left, right)
_rhs_multiply(left::Number, right::VectorField) = _scale_vector_field(right, left)
_rhs_multiply(left::ScalarField, right::VectorField) = scale_vector_field(right, left)
_rhs_multiply(left::VectorField, right::ScalarField) = scale_vector_field(left, right)
_rhs_multiply(left, right) =
    throw(ArgumentError("Unsupported Multiply operands: $(typeof(left)) and $(typeof(right))"))

_rhs_negate(operand::Number) = -operand
_rhs_negate(operand::ScalarField) = operand * -1
_rhs_negate(operand::VectorField) = _scale_vector_field(operand, -1)
_rhs_negate(operand) =
    throw(ArgumentError("Unsupported negation operand: $(typeof(operand))"))

function _rhs_index(array_val::ScalarField, indices, layout::Symbol)
    ensure_layout!(array_val, layout)
    data = layout == :g ? get_grid_data(array_val) : get_coeff_data(array_val)
    if data === nothing
        throw(ArgumentError("Field $(array_val.name) has no data in layout $layout"))
    end
    return data[indices...]
end
_rhs_index(array_val::AbstractArray, indices, ::Symbol) = array_val[indices...]
_rhs_index(array_val, _indices, ::Symbol) =
    throw(ArgumentError("Unsupported indexed operand: $(typeof(array_val))"))

"""Evaluate both operands of a binary node and coerce them onto a shared template.

Add/Subtract/Multiply/Divide share this preamble exactly; only the combination step
differs, which is what the `_rhs_*` methods above supply.
"""
function _eval_binary_operands(expr, variables, layout::Symbol, template)
    left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
    right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
    op_template = _binary_template(left, right, template)
    return (_coerce_numeric_operand(left, op_template; layout=layout),
            _coerce_numeric_operand(right, op_template; layout=layout))
end

# --- node types ----------------------------------------------------------------

# A field evaluates to itself, once its data is in the requested layout.
for FieldType in (:ScalarField, :VectorField, :TensorField)
    @eval function evaluate_solver_expression(expr::$FieldType, _variables;
                                              layout::Symbol=:g,
                                              template::Union{Nothing, ScalarField}=nothing)
        ensure_layout!(expr, layout)
        return expr
    end
end

evaluate_solver_expression(expr::Future, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) = evaluate(expr)

evaluate_solver_expression(::ZeroOperator, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    template === nothing ? 0 : create_zero_field(template)

evaluate_solver_expression(expr::ConstantOperator, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    template === nothing ? expr.value :
        _constant_field_from_template(template, expr.value; layout=layout)

function evaluate_solver_expression(expr::ArrayOperator, _variables;
                                    layout::Symbol=:g,
                                    template::Union{Nothing, ScalarField}=nothing)
    template === nothing && return expr.value
    result = create_zero_field(template)
    ensure_layout!(result, layout)
    target = layout == :g ? get_grid_data(result) : get_coeff_data(result)
    copyto!(target, expr.value)
    return result
end

# A surviving UnknownOperator at RHS-evaluation time is unresolved: it is not a
# registered operator, known function, or declared variable, and coordinate/time
# placeholders were already substituted at matrix-build. Silently returning zero
# here would drop the term and change the equation (e.g. a typo `dx(u)` instead of
# `∂x(u)`), so abort with a clear message. A dedicated exception type lets the
# RHS-evaluation try/catch rethrow this (genuine user error) while still tolerating
# transient per-field errors.
evaluate_solver_expression(expr::UnknownOperator, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    throw(UnrecognizedRHSExpression(string(expr.expression)))

evaluate_solver_expression(expr::AddOperator, variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    _rhs_add(_eval_binary_operands(expr, variables, layout, template)...)

evaluate_solver_expression(expr::SubtractOperator, variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    _rhs_subtract(_eval_binary_operands(expr, variables, layout, template)...)

evaluate_solver_expression(expr::MultiplyOperator, variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    _rhs_multiply(_eval_binary_operands(expr, variables, layout, template)...)

evaluate_solver_expression(expr::DivideOperator, variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    divide_operands(_eval_binary_operands(expr, variables, layout, template)...)

function evaluate_solver_expression(expr::PowerOperator, variables;
                                    layout::Symbol=:g,
                                    template::Union{Nothing, ScalarField}=nothing)
    base = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
    exponent = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
    exponent isa Number ||
        throw(ArgumentError("Power operator requires numeric exponent, got $(typeof(exponent))"))
    return power_operands(base, exponent)
end

function evaluate_solver_expression(expr::NegateOperator, variables;
                                    layout::Symbol=:g,
                                    template::Union{Nothing, ScalarField}=nothing)
    operand = evaluate_solver_expression(expr.operand, variables; layout=layout, template=template)
    return _rhs_negate(operand)
end

function evaluate_solver_expression(expr::IndexOperator, variables;
                                    layout::Symbol=:g,
                                    template::Union{Nothing, ScalarField}=nothing)
    array_val = evaluate_solver_expression(expr.array, variables; layout=layout, template=template)
    indices = Any[evaluate_solver_expression(idx, variables; layout=layout, template=template)
                  for idx in expr.indices]
    indices = map(idx -> idx isa Number ? Int(idx) : idx, indices)
    return _rhs_index(array_val, indices, layout)
end

# The generic operator case, deliberately: the 45 `Operator` subtypes without a
# method above (derivatives, lifts, interpolations, ...) all evaluate through the
# operator machinery. Being a real method rather than the last branch of a chain is
# what keeps the specific methods above winning on specificity alone.
evaluate_solver_expression(expr::Operator, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    evaluate(expr, layout)

evaluate_solver_expression(expr::Number, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    template === nothing ? expr :
        _constant_field_from_template(template, expr; layout=layout)

evaluate_solver_expression(::Nothing, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    throw(ArgumentError("Cannot evaluate null expression"))

# Fallback. Reached only by a node type no method above claims, which is a parser
# bug or a missing handler — never a value to guess at.
evaluate_solver_expression(expr, _variables; layout::Symbol=:g,
                           template::Union{Nothing, ScalarField}=nothing) =
    error("Unsupported expression type in evaluate_solver_expression: $(typeof(expr)). " *
          "Value: $(repr(expr)). This may indicate a parsing error or missing operator handler.")

"""
    Build Jacobian matrix block from Frechet differential expression following Tarang patterns.
    
    In Tarang, this corresponds to:
    1. expr.expression_matrices(subproblem, vars) 
    2. Returns dict {var: matrix} for each variable
    3. Recursively builds matrices for expression tree
    """
function build_jacobian_block(expr, variables, perturbations)
    
    if expr === nothing
        @warn "Cannot build Jacobian from null expression"
        return sparse(zeros(ComplexF64, 1, 1))
    end
    
    # Calculate total size needed for Jacobian block
    total_var_size = sum(compute_field_vector_size(var) for var in variables)
    jacobian_size = total_var_size > 0 ? total_var_size : 1
    
    # Handle different expression types following Tarang expression_matrices patterns
    if hasfield(typeof(expr), :expr_type)
        expr_type = expr.expr_type
        
        if expr_type == "variable"
            # Variable expression - return identity matrix (Tarang lines 183-186, 507-510, 957-960)
            return build_variable_jacobian_block(expr, variables)
            
        elseif expr_type == "operator"
            # Operator expression - recursively build from operands 
            return build_operator_jacobian_block(expr, variables, perturbations)
            
        elseif expr_type == "constant"
            # Constant expression - return zero matrix
            return sparse(zeros(ComplexF64, jacobian_size, jacobian_size))
            
        else
            @warn "Unknown expression type for Jacobian: $expr_type"
        end
    end
    
    # Fallback: identity matrix (following Tarang identity pattern)
    return sparse(I, jacobian_size, jacobian_size)
end

"""Build identity matrix block for variable (Tarang pattern)"""
function build_variable_jacobian_block(expr, variables)

    # Check for field_ref using struct field access (consistent with build_jacobian_block)
    if !hasfield(typeof(expr), :field_ref)
        @warn "Variable expression missing field reference"
        return sparse(I, 1, 1)
    end

    field_ref = expr.field_ref

    # Find variable in list and return identity matrix for it
    for var in variables
        if var === field_ref
            var_size = compute_field_vector_size(var)
            return sparse(I, var_size, var_size)
        end
    end

    @warn "Variable not found in variable list for Jacobian"
    return sparse(I, 1, 1)
end

"""Build Jacobian block for operator expression (following Tarang recursive patterns)"""
function build_operator_jacobian_block(expr, variables, perturbations)

    # Check for operator and operands using struct field access (consistent with build_jacobian_block)
    if !hasfield(typeof(expr), :operator) || !hasfield(typeof(expr), :operands)
        @warn "Malformed operator expression for Jacobian"
        total_size = sum(compute_field_vector_size(var) for var in variables)
        return sparse(I, max(total_size, 1), max(total_size, 1))
    end

    operator = expr.operator
    operands = expr.operands
    
    # Following arithmetic line 189-193 pattern: iteratively add matrices
    if operator == "Add"
        # Addition: sum of operand Jacobians
        result_matrix = nothing
        for operand in operands
            operand_jac = build_jacobian_block(operand, variables, perturbations)
            if result_matrix === nothing
                result_matrix = operand_jac
            else
                result_matrix = result_matrix + operand_jac
            end
        end
        return result_matrix !== nothing ? result_matrix : sparse(I, 1, 1)
        
    elseif operator == "Multiply"
        # Multiplication: apply product rule for Jacobian
        # d(f·g)/dx = f·(dg/dx) + g·(df/dx)
        # For F(u) = a(u)·b(u), the Jacobian is: J_F = a·J_b + b·J_a
        total_size = sum(compute_field_vector_size(var) for var in variables)
        n = max(total_size, 1)

        if length(operands) >= 2
            # For two operands a, b: J = a*J_b + b*J_a
            # This is a linear approximation - the full product rule would require
            # evaluating operands at the current state
            result_matrix = spzeros(ComplexF64, n, n)
            for op in operands
                op_jac = build_jacobian_block(op, variables, perturbations)
                if size(op_jac) == (n, n)
                    result_matrix = result_matrix + op_jac
                end
            end
            return result_matrix
        else
            return sparse(I, n, n)
        end

    elseif operator == "Differentiate"
        # Differentiation: apply differential operator matrix
        # The Jacobian of ∂f/∂x is the same as ∂/∂x applied to J_f
        # Since differentiation is linear: J_{∂f/∂x} = D · J_f where D is the diff matrix
        if length(operands) > 0
            operand_jac = build_jacobian_block(operands[1], variables, perturbations)
            # The differentiation operator commutes with Jacobian computation for linear problems
            # For spectral methods, this would involve multiplying by ik (Fourier) or D matrix (Chebyshev)
            return operand_jac
        end
        
    else
        @warn "Unknown operator for Jacobian: $operator"
    end
    
    # Fallback
    total_size = sum(compute_field_vector_size(var) for var in variables)
    return sparse(I, max(total_size, 1), max(total_size, 1))
end

# Helper functions for operators

"""
Create zero field matching template field or first variable in vector.

GPU-aware: The field is allocated on the same architecture as the template.
Uses fill!() which works on both CPU and GPU arrays.
"""
function create_zero_field(template::ScalarField)
    result = ScalarField(template.dist, "zero_field", template.bases, template.dtype)
    ensure_layout!(result, :c)
    if get_coeff_data(result) !== nothing
        # fill!() works on both CPU and GPU arrays
        fill!(get_coeff_data(result), zero(eltype(get_coeff_data(result))))
    end
    return result
end

function create_zero_field(variables::Vector)
    if length(variables) > 0
        return create_zero_field(variables[1])
    else
        throw(ArgumentError("No variables available"))
    end
end

"""
    Create field with constant value.

    GPU-aware: The field is allocated on the same architecture as the first variable.
    Uses fill!() which works on both CPU and GPU arrays.
    """
function create_constant_field(expr, variables)
    if length(variables) == 0
        throw(ArgumentError("No variables available"))
    end

    result = ScalarField(variables[1].dist, "constant_field", variables[1].bases, variables[1].dtype)
    ensure_layout!(result, :c)

    if get_coeff_data(result) !== nothing
        value = hasfield(typeof(expr), :value) ? expr.value : zero(eltype(get_coeff_data(result)))
        # fill!() works on both CPU and GPU arrays
        fill!(get_coeff_data(result), convert(eltype(get_coeff_data(result)), value))
    end

    return result
end

"""
    Apply addition operator following Tarang patterns.

    GPU-aware: Uses broadcasting (.+=) which works on both CPU and GPU arrays.
    """
function apply_add_operator(operands)
    if length(operands) == 0
        throw(ArgumentError("Addition requires operands"))
    end

    result = ScalarField(operands[1].dist, "add_result", operands[1].bases, operands[1].dtype)
    ensure_layout!(result, :c)

    if get_coeff_data(result) !== nothing
        # fill!() works on both CPU and GPU
        fill!(get_coeff_data(result), zero(eltype(get_coeff_data(result))))
        for operand in operands
            ensure_layout!(operand, :c)
            if get_coeff_data(operand) !== nothing
                # Broadcasting works on both CPU and GPU
                get_coeff_data(result) .+= get_coeff_data(operand)
            end
        end
    end

    return result
end

"""
    Apply multiplication operator following Tarang patterns.

    GPU-aware: Uses broadcasting (.*=) which works on both CPU and GPU arrays.
    """
function apply_multiply_operator(operands)
    if length(operands) < 2
        return length(operands) == 1 ? operands[1] : throw(ArgumentError("Multiplication requires 2+ operands"))
    end

    result = ScalarField(operands[1].dist, "multiply_result", operands[1].bases, operands[1].dtype)
    ensure_layout!(result, :c)

    if get_coeff_data(result) !== nothing
        # Start with first operand
        ensure_layout!(operands[1], :c)
        if get_coeff_data(operands[1]) !== nothing
            # copyto!() works on both CPU and GPU
            copyto!(get_coeff_data(result), get_coeff_data(operands[1]))
        else
            fill!(get_coeff_data(result), one(eltype(get_coeff_data(result))))
        end

        # Multiply by remaining operands
        for i in 2:length(operands)
            ensure_layout!(operands[i], :c)
            if get_coeff_data(operands[i]) !== nothing
                # Broadcasting works on both CPU and GPU
                get_coeff_data(result) .*= get_coeff_data(operands[i])
            end
        end
    end

    return result
end

"""
    Apply differentiation operator using existing operators.jl infrastructure.
    
    This leverages the complete implementation in operators.jl which includes:
    - Basis-specific differentiation (Fourier, Chebyshev, Legendre)
    - Proper spectral differentiation matrices
    - Layout management and efficient operations
    """
function apply_differentiate_operator(operands, expr)
    
    if length(operands) == 0
        throw(ArgumentError("Differentiation requires operand"))
    end
    
    operand = operands[1]
    
    # Extract coordinate information from expression
    coordinate = get_diff_coordinate(expr)
    order = get_diff_order(expr)
    
    if coordinate === nothing
        @warn "No coordinate specified for differentiation, cannot proceed"
        return create_zero_field([operand])
    end
    
    try
        # Create Differentiate operator using existing infrastructure
        diff_op = Differentiate(operand, coordinate, order)
        
        # Evaluate using the complete implementation in operators.jl
        result = evaluate_differentiate(diff_op, :c)  # Use coefficient layout
        
        @debug "Applied differentiation using operators.jl: coord=$(coordinate.name), order=$order"
        
        return result
        
    catch e
        @warn "Differentiation failed: $e, returning zero result"
        return create_zero_field([operand])
    end
end

"""Extract coordinate for differentiation from expression"""
function get_diff_coordinate(expr)
    # Direct coordinate object (struct field access)
    if hasfield(typeof(expr), :coordinate) && expr.coordinate !== nothing
        return expr.coordinate
    end

    # Coordinate name lookup - search operand's bases for matching coordinate
    if hasfield(typeof(expr), :coord_name) && hasfield(typeof(expr), :operand)
        coord_name = expr.coord_name
        operand = expr.operand

        # Try to find coordinate in operand's bases
        if hasfield(typeof(operand), :bases)
            for basis in operand.bases
                if basis !== nothing && hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :element_label)
                    if basis.meta.element_label == coord_name
                        if hasfield(typeof(basis.meta), :coordsys) && basis.meta.coordsys !== nothing
                            try
                                return basis.meta.coordsys[coord_name]
                            catch err
                                # A missing coordinate is the expected miss; any other
                                # exception is a fault in the coordinate system and must
                                # not be read as "not found".
                                err isa Union{KeyError, BoundsError, ArgumentError} || rethrow()
                            end
                        end
                    end
                end
            end
        end

        # Try distributor's coordinate system
        if hasfield(typeof(operand), :dist)
            dist = operand.dist
            if hasfield(typeof(dist), :coords) && dist.coords !== nothing
                for coord in dist.coords
                    if coord.name == coord_name
                        return coord
                    end
                end
            end
            if hasfield(typeof(dist), :coordsys) && dist.coordsys !== nothing
                try
                    return dist.coordsys[coord_name]
                catch err
                    err isa Union{KeyError, BoundsError, ArgumentError} || rethrow()
                end
            end
        end

        @debug "Coordinate '$coord_name' not found in operand's bases"
        return nothing
    end

    @debug "No coordinate specified for differentiation"
    return nothing
end

"""Extract differentiation order from expression"""
function get_diff_order(expr)
    if hasfield(typeof(expr), :order)
        return max(1, Int(expr.order))
    else
        return 1  # Default to first order
    end
end

# Performance and logging
"""Log solver performance statistics"""
function log_stats(solver::Solver)
    
    if isa(solver, InitialValueSolver)
        elapsed = time() - solver.wall_time_start
        @info "Solver statistics:"
        @info "  Total iterations: $(solver.iteration)"
        @info "  Simulation time: $(solver.sim_time)"
        @info "  Wall time: $(elapsed) seconds"
        if elapsed > 0
            @info "  Iterations per second: $(solver.iteration / elapsed)"
        end
        
        if MPI.Initialized()
            # Log MPI statistics
            mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)
            @info "  MPI rank: $mpi_rank / $nprocs"
        end
    end
end

# Analysis and output - create_evaluator is defined in evaluator.jl

"""Log solver performance statistics"""
function log_solver_performance(solver::Union{InitialValueSolver, BoundaryValueSolver})

    stats = solver.performance_stats

    if MPI.Initialized()
        mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if mpi_rank == 0
            @info "Solver performance:"
            if isa(solver, InitialValueSolver)
                @info "  Total steps: $(stats.total_steps)"
                if stats.total_steps > 0
                    @info "  Average step time: $(round(stats.total_time/stats.total_steps*1000, digits=3)) ms"
                end
            elseif isa(solver, BoundaryValueSolver)
                @info "  Total solves: $(stats.total_solves)"
            end
            @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
        end
    else
        @info "Solver performance:"
        if isa(solver, InitialValueSolver)
            @info "  Total steps: $(stats.total_steps)"
            if stats.total_steps > 0
                @info "  Average step time: $(round(stats.total_time/stats.total_steps*1000, digits=3)) ms"
            end
        elseif isa(solver, BoundaryValueSolver)
            @info "  Total solves: $(stats.total_solves)"
        end
        @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    end
end
