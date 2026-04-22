"""
    Problem matrix expression analysis

This file contains equation-variable detection, DOF inference, and operator
splitting helpers used before any sparse matrix blocks are built.
"""

function _detect_equation_variables(expr, variables::Vector{<:Operand})
    found = Operand[]
    _collect_equation_variables!(found, expr, variables)
    return found
end

function _collect_equation_variables!(found::Vector{Operand}, expr, variables::Vector{<:Operand})
    expr === nothing && return

    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        for var in variables
            if _operand_matches_variable(expr, var)
                _maybe_add_variable!(found, var)
                break
            end
        end
    end

    if hasfield(typeof(expr), :left)
        _collect_equation_variables!(found, getfield(expr, :left), variables)
    end
    if hasfield(typeof(expr), :right)
        _collect_equation_variables!(found, getfield(expr, :right), variables)
    end
    if hasfield(typeof(expr), :operand)
        _collect_equation_variables!(found, getfield(expr, :operand), variables)
    end
    if hasfield(typeof(expr), :operands)
        ops = getfield(expr, :operands)
        if ops !== nothing
            for op in ops
                _collect_equation_variables!(found, op, variables)
            end
        end
    end
    if hasfield(typeof(expr), :array)
        _collect_equation_variables!(found, getfield(expr, :array), variables)
    end
    if hasfield(typeof(expr), :indices)
        idxs = getfield(expr, :indices)
        if idxs !== nothing
            for idx in idxs
                _collect_equation_variables!(found, idx, variables)
            end
        end
    end
    if hasfield(typeof(expr), :base)
        _collect_equation_variables!(found, getfield(expr, :base), variables)
    end
    if hasfield(typeof(expr), :exponent)
        _collect_equation_variables!(found, getfield(expr, :exponent), variables)
    end
end

function _maybe_add_variable!(found::Vector{Operand}, var::Operand)
    for existing in found
        if existing === var
            return
        end
        existing_name = _operand_name(existing)
        new_name = _operand_name(var)
        if existing_name !== nothing && existing_name == new_name
            return
        end
    end
    push!(found, var)
end

@inline function _operand_matches_variable(expr, var::Operand)
    (expr === var) && return true
    expr_name = _operand_name(expr)
    var_name = _operand_name(var)
    return expr_name !== nothing && expr_name == var_name
end

@inline function _operand_name(var)
    return hasfield(typeof(var), :name) ? getfield(var, :name) : nothing
end

"""
    _equation_output_dofs(expr)

Compute the number of DOFs (rows) an equation's LHS expression produces,
by propagating output shapes through the operator tree — the same approach
uses.  This makes equation ordering irrelevant; the matrix builder
determines each equation's row count from its mathematical structure.

Returns 0 for unknown/constant expressions (overridden by `max` in Add/Sub).
"""
function _equation_output_dofs(expr)
    # ── Direct field types ──────────────────────────────────────
    isa(expr, ScalarField)  && return _coeff_space_dofs(expr)
    isa(expr, VectorField)  && return _coeff_space_dofs(expr)
    isa(expr, TensorField)  && return _coeff_space_dofs(expr)

    # ── Arithmetic (both Operator and Future types) ──────────────
    # All addends of a well-formed equation share the same shape,
    # so `max` skips zeros/constants and picks the informative term.
    #
    # Handles both hierarchies:
    #   Operator: AddOperator, SubtractOperator, MultiplyOperator, NegateOperator
    #   Future:   Add, Multiply, Subtract, Negate
    if isa(expr, AddOperator) || isa(expr, SubtractOperator)
        return max(_equation_output_dofs(expr.left),
                   _equation_output_dofs(expr.right))
    end
    if isa(expr, MultiplyOperator) || isa(expr, DivideOperator)
        return max(_equation_output_dofs(expr.left),
                   _equation_output_dofs(expr.right))
    end
    isa(expr, NegateOperator) && return _equation_output_dofs(expr.operand)

    # Future arithmetic types (from Julia-level expression building)
    if isa(expr, Future)
        args = future_args(expr)
        if !isempty(args)
            return maximum(_equation_output_dofs(a) for a in args)
        end
        return 0
    end

    # ── Shape-preserving operators ──────────────────────────────
    if isa(expr, TimeDerivative) || isa(expr, Laplacian) ||
       isa(expr, FractionalLaplacian) || isa(expr, Differentiate) ||
       isa(expr, Lift) || isa(expr, Skew) || isa(expr, Curl) ||
       isa(expr, Trace) || isa(expr, Convert) || isa(expr, HilbertTransform) ||
       isa(expr, Component) || isa(expr, RadialComponent) ||
       isa(expr, AngularComponent) || isa(expr, AzimuthalComponent)
        return _equation_output_dofs(expr.operand)
    end

    # ── Rank-changing operators ─────────────────────────────────
    if isa(expr, Gradient)
        # scalar(D) → vector(ndim*D)
        inner = _equation_output_dofs(expr.operand)
        ndim  = hasfield(typeof(expr), :coordsys) && expr.coordsys !== nothing ?
                length(expr.coordsys.coords) : 2
        return inner * ndim
    end
    if isa(expr, Divergence)
        # vector(ndim*D) → scalar(D).  Recover D = total / ndim.
        inner = _equation_output_dofs(expr.operand)
        ndim  = _infer_ndim(expr.operand)
        return ndim > 0 ? inner ÷ ndim : inner
    end

    # ── Reducing operators ──────────────────────────────────────
    if isa(expr, Integrate) || isa(expr, Average)
        return 1  # Reduces all spatial dims → scalar
    end
    if isa(expr, Interpolate)
        # Interpolate removes one spatial dim.
        # Output DOFs = operand DOFs / interpolated-dim coeff size.
        inner = _equation_output_dofs(expr.operand)
        cdim = _coord_coeff_size(expr.coord, expr.operand)
        return cdim > 0 ? inner ÷ cdim : max(inner, 1)
    end

    # ── Constants / zeros ───────────────────────────────────────
    (isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator)) && return 0

    # ── Fallback: find any field anywhere in the tree ───────────
    # This prevents unknown operator types from silently zeroing
    # the equation size and making the matrix non-square.
    dofs = _find_any_field_dofs(expr)
    if dofs > 0
        @debug "equation_output_dofs fallback for $(typeof(expr)): using field DOFs=$dofs"
        return dofs
    end
    return 0
end

"""Find coeff-space size of the basis corresponding to a coordinate."""
function _coord_coeff_size(coord::Coordinate, operand)
    field = _find_any_scalar_field(operand)
    field === nothing && return 0
    coord_name = hasfield(typeof(coord), :name) ? string(coord.name) : nothing

    # Match basis by element_label (stored in basis.meta.element_label)
    first_rf = true
    for basis in field.bases
        basis === nothing && continue
        label = hasfield(typeof(basis), :meta) ? string(basis.meta.element_label) : nothing
        if label !== nothing && coord_name !== nothing && label == coord_name
            # First RealFourier dimension is halved by rfft; others keep full size
            if isa(basis, RealFourier) && first_rf
                return div(basis.meta.size, 2) + 1
            else
                return basis.meta.size
            end
        end
        if isa(basis, RealFourier)
            first_rf = false
        end
    end
    return 0
end

"""Find any ScalarField in an expression tree (handles Operator and Future types)."""
function _find_any_scalar_field(expr)
    isa(expr, ScalarField) && return expr
    isa(expr, VectorField) && return expr.components[1]
    isa(expr, TensorField) && return expr.components[1,1]
    # Operator fields: :operand, :left, :right
    for f in (:operand, :left, :right)
        hasfield(typeof(expr), f) || continue
        result = _find_any_scalar_field(getfield(expr, f))
        result !== nothing && return result
    end
    # Future types: search args
    if isa(expr, Future)
        for arg in future_args(expr)
            result = _find_any_scalar_field(arg)
            result !== nothing && return result
        end
    end
    return nothing
end

"""Find any field DOFs in an expression tree (handles Operator and Future types)."""
function _find_any_field_dofs(expr)
    isa(expr, ScalarField)  && return _coeff_space_dofs(expr)
    isa(expr, VectorField)  && return _coeff_space_dofs(expr)
    isa(expr, TensorField)  && return _coeff_space_dofs(expr)
    for f in (:operand, :left, :right)
        hasfield(typeof(expr), f) || continue
        child = getfield(expr, f)
        child === nothing && continue
        dofs = _find_any_field_dofs(child)
        dofs > 0 && return dofs
    end
    if isa(expr, Future)
        for arg in future_args(expr)
            dofs = _find_any_field_dofs(arg)
            dofs > 0 && return dofs
        end
    end
    return 0
end

"""Infer spatial dimension count from an expression (for Divergence)."""
function _infer_ndim(expr)
    isa(expr, VectorField) && return length(expr.components)
    isa(expr, Gradient) && hasfield(typeof(expr), :coordsys) &&
        expr.coordsys !== nothing && return length(expr.coordsys.coords)
    # Recurse through Operator wrappers
    hasfield(typeof(expr), :operand) && return _infer_ndim(expr.operand)
    # Future types: search args for any that reveal ndim
    if isa(expr, Future)
        for arg in future_args(expr)
            n = _infer_ndim(arg)
            n > 0 && return n
        end
    end
    return 0
end

"""
    Split operator into time derivative (mass matrix) and spatial (stiffness) terms.
"""
function split_time_spatial_operators(operator)
    
    M_terms = []  # Time derivative terms
    L_terms = []  # Spatial terms
    empty_namespace = Dict{String, Any}()
    
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
        # Split subtraction terms recursively
        # For A - B, we split both and negate the right side terms
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)

        # Add left terms directly
        append!(M_terms, left_M)
        append!(L_terms, left_L)

        # Right terms need negation - wrap in NegateOperator or multiply by -1
        for term in right_M
            push!(M_terms, NegateOperator(term))
        end
        for term in right_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, NegateOperator)
        inner_M, inner_L = split_time_spatial_operators(operator.operand)
        for term in inner_M
            push!(M_terms, NegateOperator(term))
        end
        for term in inner_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, MultiplyOperator)
        coeff = nothing
        inner = nothing

        if _is_constant_coefficient_strict(operator.left, empty_namespace) &&
           !_is_constant_coefficient_strict(operator.right, empty_namespace)
            coeff = operator.left
            inner = operator.right
        elseif _is_constant_coefficient_strict(operator.right, empty_namespace) &&
               !_is_constant_coefficient_strict(operator.left, empty_namespace)
            coeff = operator.right
            inner = operator.left
        end

        if inner !== nothing
            scaled = MultiplyOperator(coeff, inner)
            if isa(inner, TimeDerivative)
                push!(M_terms, scaled)
            elseif isa(inner, Union{Laplacian, Gradient, Divergence, Differentiate}) || hasfield(typeof(inner), :name)
                push!(L_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end

    elseif isa(operator, DivideOperator)
        if _is_constant_coefficient_strict(operator.right, empty_namespace)
            scaled = DivideOperator(operator.left, operator.right)
            if isa(operator.left, TimeDerivative)
                push!(M_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end
        
    elseif hasfield(typeof(operator), :name)
        # Direct variable reference -> identity in L
        push!(L_terms, operator)
        
    else
        # Other operators go to L by default
        push!(L_terms, operator)
    end
    
    return M_terms, L_terms
end

"""Combine operator terms into a single expression."""
function combine_operators(terms::Vector)
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
