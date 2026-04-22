"""
    Basic tensor operators

This file contains evaluation functions for trace, skew, and
transpose-components.
"""

# ============================================================================
# Trace Evaluation
# ============================================================================

"""
    evaluate_trace(trace_op::Trace, layout::Symbol=:g)

Evaluate trace of a tensor field.
trace(T) = Sigma_i T_ii

For DirectProduct coordinates, dispatches to DirectProductTrace which sums
traces of diagonal blocks for each coordinate subsystem.
Following spectral methods pattern CartesianTrace and DirectProductTrace.
"""
function evaluate_trace(trace_op::Trace, layout::Symbol=:g)
    operand = trace_op.operand

    # Evaluate expression tree to concrete field if needed
    if !isa(operand, TensorField) && isa(operand, Operator)
        operand = evaluate(operand, layout)
    end

    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField, got $(typeof(operand))"))
    end

    # Check if coordinate system is DirectProduct - if so, delegate to DirectProductTrace
    coordsys = operand.coordsys
    if isa(coordsys, DirectProduct)
        dp_trace = DirectProductTrace(operand; index=0)
        return evaluate(dp_trace, layout)
    end

    # Standard Cartesian trace: sum diagonal elements
    # Create result scalar field
    result = ScalarField(operand.dist, "trace_$(operand.name)", operand.bases, operand.dtype)

    # Ensure diagonal components are in correct layout
    dim = size(operand.components, 1)

    for i in 1:dim
        ensure_layout!(operand.components[i,i], layout)
    end

    ensure_layout!(result, layout)

    # Sum diagonal components
    if layout == :g
        fill!(get_grid_data(result), 0.0)
        for i in 1:dim
            get_grid_data(result) .+= get_grid_data(operand.components[i,i])
        end
    else
        fill!(get_coeff_data(result), 0.0)
        for i in 1:dim
            get_coeff_data(result) .+= get_coeff_data(operand.components[i,i])
        end
    end

    return result
end

# ============================================================================
# Skew Evaluation
# ============================================================================

"""
    evaluate_skew(skew_op::Skew, layout::Symbol=:g)

Evaluate skew operator. Behavior depends on operand type:
- TensorField: Returns skew-symmetric part, skew(T) = (T - T^T) / 2
- VectorField (2D): Returns 90 degree rotation, skew(u_x, u_y) = (-u_y, u_x)
  This is used for 2D QG: u = skew(grad(psi)) gives divergence-free velocity.
"""
function evaluate_skew(skew_op::Skew, layout::Symbol=:g)
    operand = skew_op.operand

    # If operand is an operator, evaluate it first
    if isa(operand, Operator)
        operand = evaluate(operand, layout)
    end

    # Dispatch based on evaluated operand type
    if isa(operand, VectorField)
        # 2D vector rotation: skew(u_x, u_y) = (-u_y, u_x)
        return _evaluate_skew_vector(operand, layout)
    elseif isa(operand, TensorField)
        # Tensor skew-symmetric part: skew(T) = (T - T^T) / 2
        return _evaluate_tensor_skew(operand, layout)
    else
        throw(ArgumentError("Skew requires a TensorField or VectorField, got $(typeof(operand))"))
    end
end

# Forward declaration - implementation in cartesian_operators.jl
function _evaluate_skew_vector end

"""
    _evaluate_tensor_skew(operand::TensorField, layout::Symbol)

Internal: Evaluate skew-symmetric part of a tensor field.
"""
function _evaluate_tensor_skew(operand::TensorField, layout::Symbol)
    coordsys = operand.coordsys
    result = TensorField(operand.dist, coordsys, "skew_$(operand.name)", operand.bases, operand.dtype)

    dim = size(operand.components, 1)

    for i in 1:dim
        for j in 1:dim
            ensure_layout!(operand.components[i,j], layout)
            ensure_layout!(operand.components[j,i], layout)
            ensure_layout!(result.components[i,j], layout)

            if layout == :g
                get_grid_data(result.components[i,j]) .= 0.5 .* (get_grid_data(operand.components[i,j]) .- get_grid_data(operand.components[j,i]))
            else
                get_coeff_data(result.components[i,j]) .= 0.5 .* (get_coeff_data(operand.components[i,j]) .- get_coeff_data(operand.components[j,i]))
            end
        end
    end

    return result
end

# ============================================================================
# TransposeComponents Evaluation
# ============================================================================

"""
    evaluate_transpose_components(trans_op::TransposeComponents, layout::Symbol=:g)

Evaluate transpose of tensor field components.
"""
function evaluate_transpose_components(trans_op::TransposeComponents, layout::Symbol=:g)
    operand = trans_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("TransposeComponents requires a TensorField"))
    end

    coordsys = operand.coordsys
    result = TensorField(operand.dist, coordsys, "trans_$(operand.name)", operand.bases, operand.dtype)

    dim = size(operand.components, 1)

    for i in 1:dim
        for j in 1:dim
            ensure_layout!(operand.components[j,i], layout)
            ensure_layout!(result.components[i,j], layout)

            if layout == :g
                copyto!(get_grid_data(result.components[i,j]), get_grid_data(operand.components[j,i]))
            else
                copyto!(get_coeff_data(result.components[i,j]), get_coeff_data(operand.components[j,i]))
            end
        end
    end

    return result
end

# ============================================================================
