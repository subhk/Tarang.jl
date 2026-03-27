"""
    Main evaluate dispatch

This file contains the unified evaluate() function that dispatches to
specific operator evaluators, plus arithmetic operator evaluation functions.
"""

# ============================================================================
# Unified Operator Evaluation Dispatcher
# ============================================================================

"""
    evaluate(op::Operator, layout::Symbol=:g)

Unified evaluation function that dispatches to specific operator evaluators.
"""
function evaluate(op::Operator, layout::Symbol=:g)
    # Cartesian-specific operators (defined in cartesian_operators.jl)
    if isa(op, CartesianGradient)
        return evaluate_cartesian_gradient(op, layout)
    elseif isa(op, CartesianDivergence)
        return evaluate_cartesian_divergence(op, layout)
    elseif isa(op, CartesianCurl)
        return evaluate_cartesian_curl(op, layout)
    elseif isa(op, CartesianLaplacian)
        return evaluate_cartesian_laplacian(op, layout)
    elseif isa(op, CartesianTrace)
        return evaluate_cartesian_trace(op, layout)
    elseif isa(op, CartesianSkew)
        return evaluate_cartesian_skew(op, layout)
    elseif isa(op, CartesianComponent)
        return evaluate_cartesian_component(op, layout)
    # Generic operators
    elseif isa(op, Gradient)
        return evaluate_gradient(op, layout)
    elseif isa(op, Divergence)
        return evaluate_divergence(op, layout)
    elseif isa(op, Curl)
        return evaluate_curl(op, layout)
    elseif isa(op, Laplacian)
        return evaluate_laplacian(op, layout)
    elseif isa(op, FractionalLaplacian)
        return evaluate_fractional_laplacian(op, layout)
    elseif isa(op, Differentiate)
        return evaluate_differentiate(op, layout)
    elseif isa(op, Interpolate)
        return evaluate_interpolate(op, layout)
    elseif isa(op, Integrate)
        return evaluate_integrate(op, layout)
    elseif isa(op, Average)
        return evaluate_average(op, layout)
    elseif isa(op, Lift)
        return evaluate_lift(op, layout)
    elseif isa(op, Convert)
        return evaluate_convert(op, layout)
    elseif isa(op, GeneralFunction)
        return evaluate_general_function(op, layout)
    elseif isa(op, UnaryGridFunction)
        return evaluate_unary_grid_function(op, layout)
    elseif isa(op, Grid)
        return evaluate_grid(op)
    elseif isa(op, Coeff)
        return evaluate_coeff(op)
    elseif isa(op, Component)
        return evaluate_component(op)
    elseif isa(op, RadialComponent)
        return evaluate_radial_component(op)
    elseif isa(op, AngularComponent)
        return evaluate_angular_component(op)
    elseif isa(op, AzimuthalComponent)
        return evaluate_azimuthal_component(op)
    elseif isa(op, Trace)
        return evaluate_trace(op, layout)
    elseif isa(op, Skew)
        return evaluate_skew(op, layout)
    elseif isa(op, TransposeComponents)
        return evaluate_transpose_components(op, layout)
    elseif isa(op, Outer)
        return evaluate_outer(op, layout)
    elseif isa(op, AdvectiveCFL)
        return evaluate_advective_cfl(op, layout)
    elseif isa(op, Copy)
        return evaluate_copy(op, layout)
    elseif isa(op, HilbertTransform)
        return evaluate_hilbert_transform(op, layout)
    elseif isa(op, TimeDerivative)
        # TimeDerivative is handled by solvers, not direct evaluation
        throw(ArgumentError(
            "TimeDerivative (∂t) cannot be directly evaluated outside a solver. " *
            "Use `InitialValueSolver(problem, timestepper; dt=...)` to solve time-dependent problems. " *
            "The solver handles ∂t terms automatically via the timestepping algorithm."))
    elseif isa(op, NegateOperator)
        return evaluate_negate(op, layout)
    elseif isa(op, MultiplyOperator)
        return evaluate_multiply(op, layout)
    elseif isa(op, AddOperator)
        return evaluate_add(op, layout)
    elseif isa(op, SubtractOperator)
        return evaluate_subtract(op, layout)
    elseif isa(op, DivideOperator)
        return evaluate_divide(op, layout)
    elseif isa(op, PowerOperator)
        return evaluate_power(op, layout)
    elseif isa(op, IndexOperator)
        return evaluate_index(op, layout)
    else
        throw(ArgumentError(
            "Evaluation not implemented for operator type $(typeof(op)). " *
            "This operator may need to be wrapped in a solver (for ∂t terms) or " *
            "may require a custom evaluate() method. " *
            "Built-in operators: Add, Multiply, Differentiate, Gradient, Divergence, Curl, Laplacian."))
    end
end

# ============================================================================
# Arithmetic Operator Evaluation
# ============================================================================

"""
    evaluate_negate(op::NegateOperator, layout::Symbol=:g)

Evaluate negation operator: returns -operand.
"""
function evaluate_negate(op::NegateOperator, layout::Symbol=:g)
    # Evaluate the operand first
    operand = op.operand
    if isa(operand, Operator)
        result = evaluate(operand, layout)
    elseif isa(operand, Future)
        result = evaluate(operand; force=true)
    else
        result = operand
    end

    # Negate the result
    if isa(result, ScalarField)
        # Use copy to preserve array structure (important for MPI/PencilArrays)
        negated = copy(result)
        negated.name = "neg_$(result.name)"
        ensure_layout!(negated, layout)
        if layout == :g
            data = get_grid_data(negated)
            if data !== nothing
                data .= .-data
            end
        else
            data = get_coeff_data(negated)
            if data !== nothing
                data .= .-data
            end
        end
        return negated
    elseif isa(result, VectorField)
        # Copy each component
        negated = VectorField(result.dist, result.coordsys, "neg_$(result.name)", result.bases, result.dtype)
        for (i, comp) in enumerate(result.components)
            negated.components[i] = copy(comp)
            negated.components[i].name = "neg_$(comp.name)"
            ensure_layout!(negated.components[i], layout)
            if layout == :g
                data = get_grid_data(negated.components[i])
                if data !== nothing
                    data .= .-data
                end
            else
                data = get_coeff_data(negated.components[i])
                if data !== nothing
                    data .= .-data
                end
            end
        end
        return negated
    elseif isa(result, Number)
        return -result
    elseif isa(result, AbstractArray)
        return .-result
    else
        throw(ArgumentError("Cannot negate result of type $(typeof(result))"))
    end
end

"""
Helper to evaluate any operand (Operator, Future, or Field).
"""
function _eval_operand(arg, layout::Symbol)
    if isa(arg, Operator)
        return evaluate(arg, layout)
    elseif isa(arg, Future)
        return evaluate(arg; force=true)
    else
        return arg
    end
end

"""
    evaluate_multiply(op::MultiplyOperator, layout::Symbol=:g)

Evaluate multiplication operator: scalar * field or field * scalar.
"""
function evaluate_multiply(op::MultiplyOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)

    # Scalar * Field
    if isa(left, Number) && isa(right, ScalarField)
        result = copy(right)
        result.name = "mul_$(right.name)"
        ensure_layout!(result, layout)
        if layout == :g
            data = get_grid_data(result)
            if data !== nothing
                data .= left .* data
            end
        else
            data = get_coeff_data(result)
            if data !== nothing
                data .= left .* data
            end
        end
        return result
    # Field * Scalar
    elseif isa(left, ScalarField) && isa(right, Number)
        result = copy(left)
        result.name = "mul_$(left.name)"
        ensure_layout!(result, layout)
        if layout == :g
            data = get_grid_data(result)
            if data !== nothing
                data .= data .* right
            end
        else
            data = get_coeff_data(result)
            if data !== nothing
                data .= data .* right
            end
        end
        return result
    # Field * Field (element-wise in grid space)
    elseif isa(left, ScalarField) && isa(right, ScalarField)
        result = copy(left)
        result.name = "mul_$(left.name)_$(right.name)"
        ensure_layout!(left, :g)
        ensure_layout!(right, :g)
        ensure_layout!(result, :g)
        left_data = get_grid_data(left)
        right_data = get_grid_data(right)
        res_data = get_grid_data(result)
        if left_data !== nothing && right_data !== nothing && res_data !== nothing
            res_data .= left_data .* right_data
        end
        if layout == :c
            ensure_layout!(result, :c)
        end
        return result
    # Number * VectorField (component-wise scaling)
    elseif isa(left, Number) && isa(right, VectorField)
        result = copy(right)
        result.name = "mul_$(right.name)"
        for comp in result.components
            ensure_layout!(comp, layout)
            data = layout == :g ? get_grid_data(comp) : get_coeff_data(comp)
            data !== nothing && (data .= left .* data)
        end
        return result
    # VectorField * Number (component-wise scaling)
    elseif isa(left, VectorField) && isa(right, Number)
        result = copy(left)
        result.name = "mul_$(left.name)"
        for comp in result.components
            ensure_layout!(comp, layout)
            data = layout == :g ? get_grid_data(comp) : get_coeff_data(comp)
            data !== nothing && (data .= data .* right)
        end
        return result
    # Number * Number
    elseif isa(left, Number) && isa(right, Number)
        return left * right
    else
        throw(ArgumentError(
            "Cannot multiply $(typeof(left)) and $(typeof(right)). " *
            "Supported: ScalarField*ScalarField, Number*ScalarField, ScalarField*Number, " *
            "Number*VectorField, VectorField*Number. " *
            "For dot product use `dot(u, v)` or `u ⋅ v`. " *
            "For cross product use `cross(u, v)` or `u × v`."))
    end
end

"""
    evaluate_add(op::AddOperator, layout::Symbol=:g)

Evaluate addition operator: field + field or field + scalar.
"""
function evaluate_add(op::AddOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)

    if isa(left, ScalarField) && isa(right, ScalarField)
        result = copy(left)
        result.name = "add_$(left.name)_$(right.name)"
        ensure_layout!(left, layout)
        ensure_layout!(right, layout)
        ensure_layout!(result, layout)
        if layout == :g
            left_data = get_grid_data(left)
            right_data = get_grid_data(right)
            res_data = get_grid_data(result)
            if left_data !== nothing && right_data !== nothing && res_data !== nothing
                res_data .= left_data .+ right_data
            end
        else
            left_data = get_coeff_data(left)
            right_data = get_coeff_data(right)
            res_data = get_coeff_data(result)
            if left_data !== nothing && right_data !== nothing && res_data !== nothing
                res_data .= left_data .+ right_data
            end
        end
        return result
    elseif isa(left, Number) && isa(right, Number)
        return left + right
    else
        throw(ArgumentError(
            "Cannot add $(typeof(left)) and $(typeof(right)). " *
            "Supported: ScalarField+ScalarField, ScalarField+Number, Number+Number. " *
            "VectorField addition: use `add_vector_fields(a, b)`."))
    end
end

"""
    evaluate_subtract(op::SubtractOperator, layout::Symbol=:g)

Evaluate subtraction operator: field - field.
"""
function evaluate_subtract(op::SubtractOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)

    if isa(left, ScalarField) && isa(right, ScalarField)
        result = copy(left)
        result.name = "sub_$(left.name)_$(right.name)"
        ensure_layout!(left, layout)
        ensure_layout!(right, layout)
        ensure_layout!(result, layout)
        if layout == :g
            left_data = get_grid_data(left)
            right_data = get_grid_data(right)
            res_data = get_grid_data(result)
            if left_data !== nothing && right_data !== nothing && res_data !== nothing
                res_data .= left_data .- right_data
            end
        else
            left_data = get_coeff_data(left)
            right_data = get_coeff_data(right)
            res_data = get_coeff_data(result)
            if left_data !== nothing && right_data !== nothing && res_data !== nothing
                res_data .= left_data .- right_data
            end
        end
        return result
    elseif isa(left, Number) && isa(right, Number)
        return left - right
    else
        throw(ArgumentError(
            "Cannot subtract $(typeof(left)) and $(typeof(right)). " *
            "Supported: ScalarField-ScalarField, Number-Number."))
    end
end

"""
    evaluate_divide(op::DivideOperator, layout::Symbol=:g)

Evaluate division operator: field / scalar.
"""
function evaluate_divide(op::DivideOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)

    if isa(left, ScalarField) && isa(right, Number)
        result = copy(left)
        result.name = "div_$(left.name)"
        ensure_layout!(result, layout)
        if layout == :g
            data = get_grid_data(result)
            if data !== nothing
                data .= data ./ right
            end
        else
            data = get_coeff_data(result)
            if data !== nothing
                data .= data ./ right
            end
        end
        return result
    elseif isa(left, Number) && isa(right, Number)
        return left / right
    else
        throw(ArgumentError(
            "Cannot divide $(typeof(left)) by $(typeof(right)). " *
            "Division is supported for ScalarField/Number and Number/Number. " *
            "Field-by-field division is not directly supported; use pointwise: " *
            "`get_grid_data(result) .= get_grid_data(a) ./ get_grid_data(b)`."))
    end
end

"""
    evaluate_power(op::PowerOperator, layout::Symbol=:g)

Evaluate power operator: field ^ exponent (element-wise in grid space).
Only numeric exponents are supported.
"""
function evaluate_power(op::PowerOperator, layout::Symbol=:g)
    base = _eval_operand(op.left, layout)
    exponent = _eval_operand(op.right, layout)

    if !(exponent isa Number)
        throw(ArgumentError("Power operator requires numeric exponent, got $(typeof(exponent))"))
    end

    return power_operands(base, exponent)
end

"""
    evaluate_index(op::IndexOperator, layout::Symbol=:g)

Evaluate indexing operator: array[indices...].
"""
function evaluate_index(op::IndexOperator, layout::Symbol=:g)
    array_val = _eval_operand(op.array, layout)
    indices = [_eval_operand(idx, layout) for idx in op.indices]
    indices = map(idx -> idx isa Number ? Int(idx) : idx, indices)

    if array_val isa ScalarField
        ensure_layout!(array_val, layout)
        data = layout == :g ? get_grid_data(array_val) : get_coeff_data(array_val)
        if data === nothing
            throw(ArgumentError(
                "Field '$(array_val.name)' has no data in layout :$layout. " *
                "Call `ensure_layout!(field, :$layout)` or `allocate_data!(field)` first."))
        end
        return data[indices...]
    elseif array_val isa AbstractArray
        return array_val[indices...]
    else
        throw(ArgumentError("Cannot index into $(typeof(array_val))"))
    end
end

# ============================================================================
# Exports
# ============================================================================

# Export abstract type
export Operator

# Export differential operator types
export Gradient, Divergence, Curl, Laplacian, FractionalLaplacian
export Trace, Skew, TransposeComponents, TimeDerivative

# Export interpolation/integration operator types
export Interpolate, Integrate, Average

# Export conversion operator types
export Convert, Grid, Coeff, Lift

# Export component extraction operator types
export Component, RadialComponent, AngularComponent, AzimuthalComponent

# Export differentiation and product operator types
export Differentiate, Outer, AdvectiveCFL

# Export arithmetic operator types
export AddOperator, SubtractOperator, MultiplyOperator, DivideOperator, NegateOperator, PowerOperator

# Export function operator types
export GeneralFunction, UnaryGridFunction

# Export copy and Hilbert transform operator types
export Copy, HilbertTransform

# Export constructor functions
export grad, divergence, div_op, curl, lap, trace, skew, transpose_components
export interpolate, integrate, average, convert_basis, lift, d, dt

# Export fractional/hyperviscosity constructors
export fraclap, sqrtlap, invsqrtlap, hyperlap

# Export other operator constructors
export outer, advective_cfl, cfl
export component, radial, angular, azimuthal
export grid, coeff, apply_function, copy_field, hilbert
export sin_field, cos_field, tan_field, exp_field, log_field, sqrt_field, abs_field, tanh_field

# Export Unicode operator aliases (defined in constructors.jl)
export ∇, Δ, ∇², ∂t, Δᵅ, Δ², Δ⁴, Δ⁶, Δ⁸

# Export evaluation functions
export evaluate
export evaluate_gradient, evaluate_divergence, evaluate_curl, evaluate_laplacian
export evaluate_fractional_laplacian, evaluate_differentiate
export evaluate_interpolate, evaluate_integrate, evaluate_average
export evaluate_lift, evaluate_convert
export evaluate_grid, evaluate_coeff, evaluate_component
export evaluate_radial_component, evaluate_angular_component, evaluate_azimuthal_component
export evaluate_trace, evaluate_skew, evaluate_transpose_components
export evaluate_outer, evaluate_advective_cfl
export evaluate_general_function, evaluate_unary_grid_function
export evaluate_copy, evaluate_hilbert_transform
export evaluate_power, evaluate_index
export evaluate_cartesian_gradient, evaluate_cartesian_divergence
export evaluate_cartesian_curl, evaluate_cartesian_laplacian
export evaluate_cartesian_trace, evaluate_cartesian_skew, evaluate_cartesian_component

# Export matrix interface functions
export matrix_dependence, matrix_coupling, subproblem_matrix
export check_conditions, enforce_conditions, is_linear, operator_order

# Export operator registration functions
export register_operator_alias!, register_operator_parseable!, register_operator_prefix!
export OPERATOR_ALIASES, OPERATOR_PARSEABLES, OPERATOR_PREFIXES
