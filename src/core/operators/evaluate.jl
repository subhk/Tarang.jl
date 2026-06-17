"""
    Main evaluate dispatch

This file contains evaluate() method definitions that use Julia's multiple dispatch
instead of runtime isa checks. Each operator type gets its own method, enabling
the compiler to inline and optimize at compile time.

Note: Cartesian operator evaluate() methods are defined in cartesian_operators.jl.
"""

# ============================================================================
# Operator Evaluation via Multiple Dispatch
# ============================================================================

# Fallback for unknown operator types
function evaluate(op::Operator, layout::Symbol=:g)
    throw(ArgumentError(
        "Evaluation not implemented for operator type $(typeof(op)). " *
        "This operator may need to be wrapped in a solver (for ∂t terms) or " *
        "may require a custom evaluate() method. " *
        "Built-in operators: Add, Multiply, Differentiate, Gradient, Divergence, Curl, Laplacian."))
end

# Generic operators
evaluate(op::Gradient, layout::Symbol=:g) = evaluate_gradient(op, layout)
evaluate(op::Divergence, layout::Symbol=:g) = evaluate_divergence(op, layout)
evaluate(op::Curl, layout::Symbol=:g) = evaluate_curl(op, layout)
evaluate(op::Laplacian, layout::Symbol=:g) = evaluate_laplacian(op, layout)
evaluate(op::FractionalLaplacian, layout::Symbol=:g) = evaluate_fractional_laplacian(op, layout)
evaluate(op::Differentiate, layout::Symbol=:g) = evaluate_differentiate(op, layout)
evaluate(op::Interpolate, layout::Symbol=:g) = evaluate_interpolate(op, layout)
evaluate(op::Integrate, layout::Symbol=:g) = evaluate_integrate(op, layout)
evaluate(op::Average, layout::Symbol=:g) = evaluate_average(op, layout)
evaluate(op::Lift, layout::Symbol=:g) = evaluate_lift(op, layout)
evaluate(op::Convert, layout::Symbol=:g) = evaluate_convert(op, layout)
evaluate(op::GeneralFunction, layout::Symbol=:g) = evaluate_general_function(op, layout)
evaluate(op::UnaryGridFunction, layout::Symbol=:g) = evaluate_unary_grid_function(op, layout)
evaluate(op::Trace, layout::Symbol=:g) = evaluate_trace(op, layout)
evaluate(op::Skew, layout::Symbol=:g) = evaluate_skew(op, layout)
evaluate(op::TransposeComponents, layout::Symbol=:g) = evaluate_transpose_components(op, layout)
evaluate(op::Outer, layout::Symbol=:g) = evaluate_outer(op, layout)
evaluate(op::AdvectiveCFL, layout::Symbol=:g) = evaluate_advective_cfl(op, layout)
evaluate(op::Copy, layout::Symbol=:g) = evaluate_copy(op, layout)
evaluate(op::HilbertTransform, layout::Symbol=:g) = evaluate_hilbert_transform(op, layout)

# Component extraction (no layout needed, but accept it for uniform API)
evaluate(op::Grid, ::Symbol=:g) = evaluate_grid(op)
evaluate(op::Coeff, ::Symbol=:g) = evaluate_coeff(op)
evaluate(op::Component, ::Symbol=:g) = evaluate_component(op)
evaluate(op::RadialComponent, ::Symbol=:g) = evaluate_radial_component(op)
evaluate(op::AngularComponent, ::Symbol=:g) = evaluate_angular_component(op)
evaluate(op::AzimuthalComponent, ::Symbol=:g) = evaluate_azimuthal_component(op)

# Arithmetic operators
evaluate(op::NegateOperator, layout::Symbol=:g) = evaluate_negate(op, layout)
evaluate(op::MultiplyOperator, layout::Symbol=:g) = evaluate_multiply(op, layout)
evaluate(op::AddOperator, layout::Symbol=:g) = evaluate_add(op, layout)
evaluate(op::SubtractOperator, layout::Symbol=:g) = evaluate_subtract(op, layout)
evaluate(op::DivideOperator, layout::Symbol=:g) = evaluate_divide(op, layout)
evaluate(op::PowerOperator, layout::Symbol=:g) = evaluate_power(op, layout)
evaluate(op::IndexOperator, layout::Symbol=:g) = evaluate_index(op, layout)

# TimeDerivative cannot be directly evaluated
function evaluate(op::TimeDerivative, ::Symbol=:g)
    throw(ArgumentError(
        "TimeDerivative (∂t) cannot be directly evaluated outside a solver. " *
        "Use `InitialValueSolver(problem, timestepper; dt=...)` to solve time-dependent problems. " *
        "The solver handles ∂t terms automatically via the timestepping algorithm."))
end

# ============================================================================
# Arithmetic Operator Evaluation
# ============================================================================

"""
    evaluate_negate(op::NegateOperator, layout::Symbol=:g)

Evaluate negation operator: returns -operand.
"""
function evaluate_negate(op::NegateOperator, layout::Symbol=:g)
    result = _eval_operand(op.operand, layout)
    return _negate_result(result, layout)
end

# Dispatch methods for negation result handling
function _negate_result(result::ScalarField, layout::Symbol)
    negated = checkout_or_alloc(result.bases, result.dtype, result.dist)
    ensure_layout!(result, layout)
    ensure_layout!(negated, layout)
    src = layout == :g ? get_grid_data(result) : get_coeff_data(result)
    dst = layout == :g ? get_grid_data(negated) : get_coeff_data(negated)
    if src !== nothing && dst !== nothing
        dst .= .-src
    end
    return negated
end

function _negate_result(result::VectorField, layout::Symbol)
    negated = VectorField(result.dist, result.coordsys, "_neg", result.bases, result.dtype)
    for (i, comp) in enumerate(result.components)
        ensure_layout!(comp, layout)
        ensure_layout!(negated.components[i], layout)
        src = layout == :g ? get_grid_data(comp) : get_coeff_data(comp)
        dst = layout == :g ? get_grid_data(negated.components[i]) : get_coeff_data(negated.components[i])
        if src !== nothing && dst !== nothing
            dst .= .-src
        end
    end
    return negated
end

_negate_result(result::Number, ::Symbol) = -result
_negate_result(result::AbstractArray, ::Symbol) = .-result
_negate_result(result, ::Symbol) = throw(ArgumentError("Cannot negate result of type $(typeof(result))"))

"""
Helper to evaluate any operand (Operator, Future, or Field).
Uses dispatch instead of isa() chain.
"""
_eval_operand(arg::Operator, layout::Symbol) = evaluate(arg, layout)
_eval_operand(arg::Future, layout::Symbol) = evaluate(arg; force=true)
_eval_operand(arg, ::Symbol) = arg  # Fields, Numbers, etc. are already evaluated

"""
    evaluate_multiply(op::MultiplyOperator, layout::Symbol=:g)

Evaluate multiplication operator: scalar * field or field * scalar.
"""
function evaluate_multiply(op::MultiplyOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)
    return _multiply_result(left, right, layout)
end

# Dispatch methods for multiply result handling — avoids isa() chain
function _multiply_result(left::Number, right::ScalarField, layout::Symbol)
    result = checkout_or_alloc(right.bases, right.dtype, right.dist)
    ensure_layout!(right, layout)
    ensure_layout!(result, layout)
    src = layout == :g ? get_grid_data(right) : get_coeff_data(right)
    dst = layout == :g ? get_grid_data(result) : get_coeff_data(result)
    if src !== nothing && dst !== nothing
        @. dst = left * src
    end
    return result
end

function _multiply_result(left::ScalarField, right::Number, layout::Symbol)
    return _multiply_result(right, left, layout)
end

function _multiply_result(left::ScalarField, right::ScalarField, layout::Symbol)
    result = checkout_or_alloc(left.bases, left.dtype, left.dist)
    ensure_layout!(left, :g)
    ensure_layout!(right, :g)
    ensure_layout!(result, :g)
    left_data = get_grid_data(left)
    right_data = get_grid_data(right)
    res_data = get_grid_data(result)
    if left_data !== nothing && right_data !== nothing && res_data !== nothing
        @. res_data = left_data * right_data
    end
    if layout == :c
        ensure_layout!(result, :c)
    end
    return result
end

function _multiply_result(left::Number, right::VectorField, layout::Symbol)
    result = VectorField(right.dist, right.coordsys, "_mul", right.bases, right.dtype)
    for (i, comp) in enumerate(right.components)
        ensure_layout!(comp, layout)
        ensure_layout!(result.components[i], layout)
        src = layout == :g ? get_grid_data(comp) : get_coeff_data(comp)
        dst = layout == :g ? get_grid_data(result.components[i]) : get_coeff_data(result.components[i])
        if src !== nothing && dst !== nothing
            @. dst = left * src
        end
    end
    return result
end

function _multiply_result(left::VectorField, right::Number, layout::Symbol)
    return _multiply_result(right, left, layout)
end

_multiply_result(left::Number, right::Number, ::Symbol) = left * right

function _multiply_result(left, right, ::Symbol)
    throw(ArgumentError(
        "Cannot multiply $(typeof(left)) and $(typeof(right)). " *
        "Supported: ScalarField*ScalarField, Number*ScalarField, ScalarField*Number, " *
        "Number*VectorField, VectorField*Number. " *
        "For dot product use `dot(u, v)` or `u ⋅ v`. " *
        "For cross product use `cross(u, v)` or `u × v`."))
end

"""
    evaluate_add(op::AddOperator, layout::Symbol=:g)

Evaluate addition operator: field + field or field + scalar.
"""
function evaluate_add(op::AddOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)
    return _add_result(left, right, layout)
end

function _add_result(left::ScalarField, right::ScalarField, layout::Symbol)
    result = checkout_or_alloc(left.bases, left.dtype, left.dist)
    ensure_layout!(left, layout)
    ensure_layout!(right, layout)
    ensure_layout!(result, layout)
    left_data = layout == :g ? get_grid_data(left) : get_coeff_data(left)
    right_data = layout == :g ? get_grid_data(right) : get_coeff_data(right)
    res_data = layout == :g ? get_grid_data(result) : get_coeff_data(result)
    if left_data !== nothing && right_data !== nothing && res_data !== nothing
        @. res_data = left_data + right_data
    end
    return result
end

_add_result(left::Number, right::Number, ::Symbol) = left + right

function _add_result(left::VectorField, right::VectorField, layout::Symbol)
    result = VectorField(left.dist, left.coordsys, "_add", left.bases, left.dtype)
    for (i, (lcomp, rcomp)) in enumerate(zip(left.components, right.components))
        ensure_layout!(lcomp, layout)
        ensure_layout!(rcomp, layout)
        ensure_layout!(result.components[i], layout)
        l_data = layout == :g ? get_grid_data(lcomp) : get_coeff_data(lcomp)
        r_data = layout == :g ? get_grid_data(rcomp) : get_coeff_data(rcomp)
        res_data = layout == :g ? get_grid_data(result.components[i]) : get_coeff_data(result.components[i])
        if l_data !== nothing && r_data !== nothing && res_data !== nothing
            @. res_data = l_data + r_data
        end
    end
    return result
end

function _add_result(left, right, ::Symbol)
    throw(ArgumentError(
        "Cannot add $(typeof(left)) and $(typeof(right)). " *
        "Supported: ScalarField+ScalarField, VectorField+VectorField, Number+Number."))
end

"""
    evaluate_subtract(op::SubtractOperator, layout::Symbol=:g)

Evaluate subtraction operator: field - field.
"""
function evaluate_subtract(op::SubtractOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)
    return _subtract_result(left, right, layout)
end

function _subtract_result(left::ScalarField, right::ScalarField, layout::Symbol)
    result = checkout_or_alloc(left.bases, left.dtype, left.dist)
    ensure_layout!(left, layout)
    ensure_layout!(right, layout)
    ensure_layout!(result, layout)
    left_data = layout == :g ? get_grid_data(left) : get_coeff_data(left)
    right_data = layout == :g ? get_grid_data(right) : get_coeff_data(right)
    res_data = layout == :g ? get_grid_data(result) : get_coeff_data(result)
    if left_data !== nothing && right_data !== nothing && res_data !== nothing
        @. res_data = left_data - right_data
    end
    return result
end

_subtract_result(left::Number, right::Number, ::Symbol) = left - right

function _subtract_result(left::VectorField, right::VectorField, layout::Symbol)
    result = VectorField(left.dist, left.coordsys, "_sub", left.bases, left.dtype)
    for (i, (lcomp, rcomp)) in enumerate(zip(left.components, right.components))
        ensure_layout!(lcomp, layout)
        ensure_layout!(rcomp, layout)
        ensure_layout!(result.components[i], layout)
        l_data = layout == :g ? get_grid_data(lcomp) : get_coeff_data(lcomp)
        r_data = layout == :g ? get_grid_data(rcomp) : get_coeff_data(rcomp)
        res_data = layout == :g ? get_grid_data(result.components[i]) : get_coeff_data(result.components[i])
        if l_data !== nothing && r_data !== nothing && res_data !== nothing
            @. res_data = l_data - r_data
        end
    end
    return result
end

function _subtract_result(left, right, ::Symbol)
    throw(ArgumentError(
        "Cannot subtract $(typeof(left)) and $(typeof(right)). " *
        "Supported: ScalarField-ScalarField, VectorField-VectorField, Number-Number."))
end

"""
    evaluate_divide(op::DivideOperator, layout::Symbol=:g)

Evaluate division operator: field / scalar.
"""
function evaluate_divide(op::DivideOperator, layout::Symbol=:g)
    left = _eval_operand(op.left, layout)
    right = _eval_operand(op.right, layout)
    return _divide_result(left, right, layout)
end

function _divide_result(left::ScalarField, right::Number, layout::Symbol)
    result = checkout_or_alloc(left.bases, left.dtype, left.dist)
    ensure_layout!(left, layout)
    ensure_layout!(result, layout)
    src = layout == :g ? get_grid_data(left) : get_coeff_data(left)
    dst = layout == :g ? get_grid_data(result) : get_coeff_data(result)
    if src !== nothing && dst !== nothing
        @. dst = src / right
    end
    return result
end

_divide_result(left::Number, right::Number, ::Symbol) = left / right

# Delegate everything else (Number/Field, Vector/Number, Vector/Field, Field/Field)
# to the Future-path `divide_operands`, which implements all of them and is what the
# solver RHS interpreter uses. The operator/evaluate path (analysis/output of parsed
# expressions like `1/u` or `B/rho0`) previously threw for these, so the SAME term
# that works inside the solver errored when evaluated for output. `divide_operands`
# raises its own clear error for genuinely-unsupported combinations.
_divide_result(left, right, ::Symbol) = divide_operands(left, right)

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

    result = power_operands(base, exponent)
    # `power_operands` is a pointwise (nonlinear) op done in grid space, so it returns a
    # :g field. Honor the requested layout — otherwise a top-level `evaluate(u^2, :c)`
    # returns a field flagged :g with stale/empty coefficients (siblings add/multiply/negate
    # all transform to the requested layout).
    return _ensure_result_layout!(result, layout)
end

# Transform a field result to `layout`; pass Numbers/Arrays through unchanged.
_ensure_result_layout!(x, ::Symbol) = x
_ensure_result_layout!(x::ScalarField, layout::Symbol) = (ensure_layout!(x, layout); x)
function _ensure_result_layout!(x::VectorField, layout::Symbol)
    for c in x.components; ensure_layout!(c, layout); end
    return x
end
function _ensure_result_layout!(x::TensorField, layout::Symbol)
    for c in x.components; ensure_layout!(c, layout); end
    return x
end

"""
    evaluate_index(op::IndexOperator, layout::Symbol=:g)

Evaluate indexing operator: array[indices...].
"""
function evaluate_index(op::IndexOperator, layout::Symbol=:g)
    array_val = _eval_operand(op.array, layout)
    # Evaluate and coerce to Int in one pass (one allocation instead of two).
    indices = map(idx -> (v = _eval_operand(idx, layout); v isa Number ? Int(v) : v), op.indices)

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
