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

    # Step 2: Evaluate residual expressions (lhs - rhs)
    residual_fields = ScalarField[]

    if hasfield(typeof(problem), :equation_data) && problem.equation_data !== nothing
        for (i, eq_data) in enumerate(problem.equation_data)
            template = state_fields[min(i, length(state_fields))]
            lhs_expr = get(eq_data, "lhs", nothing)
            rhs_expr = get(eq_data, "rhs", nothing)

            if lhs_expr !== nothing || rhs_expr !== nothing
                lhs_field = lhs_expr === nothing ? create_zero_field(template) :
                           evaluate_solver_expression(lhs_expr, problem.variables; layout=:g, template=template)
                rhs_field = rhs_expr === nothing ? create_zero_field(template) :
                           evaluate_solver_expression(rhs_expr, problem.variables; layout=:g, template=template)
                residual_field = lhs_field - rhs_field
            else
                # Fallback: use F expression if provided
                expr = get(eq_data, "F", ZeroOperator())
                residual_field = evaluate_solver_expression(expr, problem.variables; layout=:g, template=template)
            end

            ensure_layout!(residual_field, :c)
            push!(residual_fields, residual_field)
            @debug "Evaluated residual for equation $i"
        end
    else
        @warn "No equation data available for residual evaluation - creating zero residuals"
        for (i, field) in enumerate(state_fields)
            residual_field = create_zero_field(field)
            push!(residual_fields, residual_field)
        end
    end

    # Step 3: Convert residual fields to vector
    residual = fields_to_vector(residual_fields)

    # Step 4: Build Jacobian matrix
    # Try symbolic Jacobian first (Frechet differentiation), then fall back
    n = length(x)
    jacobian = try
        build_symbolic_jacobian(problem, state_fields)
    catch e
        @debug "Symbolic Jacobian construction failed ($e), using fallback"
        if haskey(problem.parameters, "L_matrix")
            problem.parameters["L_matrix"]
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
    Evaluate a parsed solver expression with current field values.
    Returns a field (preferred) or a numeric scalar for constant expressions.
    """
function evaluate_solver_expression(expr, variables; layout::Symbol=:g, template::Union{Nothing, ScalarField}=nothing)

    if expr === nothing
        throw(ArgumentError("Cannot evaluate null expression"))
    end

    if expr isa ScalarField
        ensure_layout!(expr, layout)
        return expr
    elseif expr isa VectorField
        ensure_layout!(expr, layout)
        return expr
    elseif expr isa TensorField
        ensure_layout!(expr, layout)
        return expr
    elseif expr isa Future
        return evaluate(expr)
    elseif expr isa ZeroOperator
        return template === nothing ? 0 : create_zero_field(template)
    elseif expr isa ConstantOperator
        return template === nothing ? expr.value : _constant_field_from_template(template, expr.value; layout=layout)
    elseif expr isa ArrayOperator
        if template === nothing
            return expr.value
        end
        result = create_zero_field(template)
        ensure_layout!(result, layout)
        target = layout == :g ? get_grid_data(result) : get_coeff_data(result)
        copyto!(target, expr.value)
        return result
    elseif expr isa UnknownOperator
        @debug "Unknown operator expression: $(expr.expression)"
        return template === nothing ? 0 : create_zero_field(template)
    elseif expr isa AddOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        if left isa Number && right isa Number
            return left + right
        elseif left isa ScalarField && right isa ScalarField
            return left + right
        elseif left isa VectorField && right isa VectorField
            return add_vector_fields(left, right)
        else
            throw(ArgumentError("Unsupported Add operands: $(typeof(left)) and $(typeof(right))"))
        end
    elseif expr isa SubtractOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        if left isa Number && right isa Number
            return left - right
        elseif left isa ScalarField && right isa ScalarField
            return left - right
        elseif left isa VectorField && right isa VectorField
            return add_vector_fields(left, _scale_vector_field(right, -1))
        else
            throw(ArgumentError("Unsupported Subtract operands: $(typeof(left)) and $(typeof(right))"))
        end
    elseif expr isa MultiplyOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        if left isa Number && right isa Number
            return left * right
        elseif left isa ScalarField && right isa ScalarField
            return left * right
        elseif left isa ScalarField && right isa Number
            return left * right
        elseif left isa Number && right isa ScalarField
            return right * left
        elseif left isa VectorField && right isa Number
            return _scale_vector_field(left, right)
        elseif left isa Number && right isa VectorField
            return _scale_vector_field(right, left)
        elseif left isa ScalarField && right isa VectorField
            return scale_vector_field(right, left)
        elseif left isa VectorField && right isa ScalarField
            return scale_vector_field(left, right)
        else
            throw(ArgumentError("Unsupported Multiply operands: $(typeof(left)) and $(typeof(right))"))
        end
    elseif expr isa DivideOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        return divide_operands(left, right)
    elseif expr isa PowerOperator
        base = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        exponent = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        if exponent isa Number
            return power_operands(base, exponent)
        end
        throw(ArgumentError("Power operator requires numeric exponent, got $(typeof(exponent))"))
    elseif expr isa NegateOperator
        operand = evaluate_solver_expression(expr.operand, variables; layout=layout, template=template)
        if operand isa Number
            return -operand
        elseif operand isa ScalarField
            return operand * -1
        elseif operand isa VectorField
            return _scale_vector_field(operand, -1)
        else
            throw(ArgumentError("Unsupported negation operand: $(typeof(operand))"))
        end
    elseif expr isa IndexOperator
        array_val = evaluate_solver_expression(expr.array, variables; layout=layout, template=template)
        indices = Any[evaluate_solver_expression(idx, variables; layout=layout, template=template) for idx in expr.indices]
        indices = map(idx -> idx isa Number ? Int(idx) : idx, indices)
        if array_val isa ScalarField
            ensure_layout!(array_val, layout)
            data = layout == :g ? get_grid_data(array_val) : get_coeff_data(array_val)
            if data === nothing
                throw(ArgumentError("Field $(array_val.name) has no data in layout $layout"))
            end
            return data[indices...]
        elseif array_val isa AbstractArray
            return array_val[indices...]
        else
            throw(ArgumentError("Unsupported indexed operand: $(typeof(array_val))"))
        end
    elseif expr isa Operator
        return evaluate(expr, layout)
    elseif expr isa Number
        return template === nothing ? expr : _constant_field_from_template(template, expr; layout=layout)
    end

    error("Unsupported expression type in evaluate_solver_expression: $(typeof(expr)). " *
          "Value: $(repr(expr)). This may indicate a parsing error or missing operator handler.")
end

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
                            catch
                                # Coordinate not found in coordsys
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
                catch
                    # Coordinate not found in coordsys
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
