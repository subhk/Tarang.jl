"""
    Problem matrix legacy helpers

This file contains forcing-vector construction plus the older compatibility
helpers that operate on dense row/column views of the matrix system.
"""

function build_forcing_vector(problem::Problem, eqn_sizes::Vector{Int}, total_size::Int)
    
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

"""
    Process LHS operator and extract contributions to system matrices.
    Following pattern where time derivatives go to M_matrix,
    spatial operators go to L_matrix.
    """
function process_lhs_operator!(L_matrix::Matrix, M_matrix::Matrix, lhs_op, eq_idx::Int, variables::Vector)
    
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
            # Store operator type marker - actual spectral matrix coefficients
            # are computed during subproblem matrix assembly based on basis type
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
        
    elseif hasfield(typeof(lhs_op), :name)
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

"""
    Process RHS operator and extract contributions to forcing vector.
    Following pattern where RHS represents known terms/forcing.

    Recursively evaluates composite operators (Add, Subtract, Multiply) to
    compute the scalar forcing value for each equation.
    """
function process_rhs_operator!(F_vector::Vector, rhs_op, eq_idx::Int, variables::Vector)

    if isa(rhs_op, ConstantOperator)
        # Constant forcing term
        F_vector[eq_idx] = rhs_op.value

    elseif isa(rhs_op, ZeroOperator)
        # Zero RHS (homogeneous equation)
        F_vector[eq_idx] = 0.0

    elseif isa(rhs_op, AddOperator)
        # Sum of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value + right_value

    elseif isa(rhs_op, SubtractOperator)
        # Difference of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value - right_value

    elseif isa(rhs_op, MultiplyOperator)
        # Product of RHS terms
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        if isa(rhs_op.right, Number)
            F_vector[eq_idx] = left_value * rhs_op.right
        else
            right_value = evaluate_rhs_scalar(rhs_op.right, variables)
            F_vector[eq_idx] = left_value * right_value
        end

    elseif isa(rhs_op, Number)
        # Direct numeric value
        F_vector[eq_idx] = Float64(real(rhs_op))

    elseif isa(rhs_op, String) && (rhs_op == "0" || rhs_op == "zero")
        # String representation of zero
        F_vector[eq_idx] = 0.0

    else
        @debug "Unhandled RHS operator type: $(typeof(rhs_op)), using zero"
        F_vector[eq_idx] = 0.0
    end
end

"""
    evaluate_rhs_scalar(op, variables::Vector) -> Float64

Recursively evaluate an operator expression to obtain a scalar value.
Used for extracting forcing terms from composite RHS expressions.

Returns the scalar value of the expression, or 0.0 for unhandled types.
"""
function evaluate_rhs_scalar(op, variables::Vector)
    if isa(op, ConstantOperator)
        return Float64(op.value)

    elseif isa(op, ZeroOperator)
        return 0.0

    elseif isa(op, Number)
        return Float64(real(op))

    elseif isa(op, AddOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val + right_val

    elseif isa(op, SubtractOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val - right_val

    elseif isa(op, MultiplyOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        if isa(op.right, Number)
            return left_val * op.right
        else
            right_val = evaluate_rhs_scalar(op.right, variables)
            return left_val * right_val
        end

    elseif isa(op, ScalarField)
        # For field-valued RHS, we need to evaluate at specific points
        # For now, return the mean value if available
        if get_grid_data(op) !== nothing && length(get_grid_data(op)) > 0
            return real(sum(get_grid_data(op)) / length(get_grid_data(op)))
        elseif get_coeff_data(op) !== nothing && length(get_coeff_data(op)) > 0
            # First coefficient is often the mean for spectral methods
            # Use GPU-safe indexing to avoid scalar indexing on GPU arrays
            if is_gpu_array(get_coeff_data(op))
                # Copy first element to CPU to avoid GPU scalar indexing
                first_coef = Array(@view get_coeff_data(op)[1:1])[1]
            else
                first_coef = get_coeff_data(op)[1]
            end
            return real(first_coef)
        else
            return 0.0
        end

    elseif isa(op, String)
        # Try to parse as number
        if op == "0" || op == "zero"
            return 0.0
        end
        try
            return parse(Float64, op)
        catch
            return 0.0
        end

    else
        @debug "evaluate_rhs_scalar: unhandled type $(typeof(op)), returning 0.0"
        return 0.0
    end
end

"""Find index of variable in problem variable list"""
function find_variable_index(operand, variables::Vector)
    
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
