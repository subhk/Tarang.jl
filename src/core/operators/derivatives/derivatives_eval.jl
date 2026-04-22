# Derivative evaluator entry points for gradient, divergence, and Differentiate.

# ============================================================================
# Gradient and Divergence Evaluation
# ============================================================================

"""Evaluate gradient operator."""
function evaluate_gradient(grad_op::Gradient, layout::Symbol=:g)
    operand = grad_op.operand
    coordsys = grad_op.coordsys

    if isa(operand, ScalarField)
        # Scalar → VectorField (∂f/∂xᵢ for each i)
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end
        return result

    elseif isa(operand, VectorField)
        # Vector → TensorField (Jacobian: Tᵢⱼ = ∂uⱼ/∂xᵢ)
        ndim = length(coordsys.names)
        result = TensorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            for j in 1:length(operand.components)
                result.components[i, j] = evaluate_differentiate(
                    Differentiate(operand.components[j], coord, 1), layout)
            end
        end
        return result

    else
        throw(ArgumentError("Gradient not implemented for operand type $(typeof(operand))"))
    end
end

"""Evaluate divergence operator."""
function evaluate_divergence(div_op::Divergence, layout::Symbol=:g)
    operand = div_op.operand

    if isa(operand, VectorField)
        # Sum partial derivatives of components
        coordsys = operand.coordsys

        # Create result field from pool, then copy data to preserve PencilArray structure
        result = checkout_or_alloc(operand.components[1].bases, operand.components[1].dtype, operand.components[1].dist)
        copy_field_data!(result, operand.components[1])
        result.current_layout = operand.components[1].current_layout
        result.name = "div_$(operand.name)"

        # Initialize result to zero — ensure data is allocated even if copy didn't provide it
        ensure_layout!(result, layout)
        if layout == :g
            grid_data = get_grid_data(result)
            if grid_data === nothing
                # Allocate grid data if not present (copy may not have provided it)
                set_grid_data!(result, zeros(eltype(get_grid_data(operand.components[1])),
                                             size(get_grid_data(operand.components[1]))))
            elseif isa(grid_data, PencilArrays.PencilArray)
                fill!(parent(grid_data), zero(eltype(grid_data)))
            else
                fill!(grid_data, zero(eltype(grid_data)))
            end
        else
            coeff_data = get_coeff_data(result)
            if coeff_data === nothing
                set_coeff_data!(result, zeros(eltype(get_coeff_data(operand.components[1])),
                                              size(get_coeff_data(operand.components[1]))))
            elseif isa(coeff_data, PencilArrays.PencilArray)
                fill!(parent(coeff_data), zero(eltype(coeff_data)))
            else
                fill!(coeff_data, zero(eltype(coeff_data)))
            end
        end

        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Add d(u_i)/d(x_i) — accumulate into field data directly, not via symbolic +
            component_deriv = evaluate_differentiate(Differentiate(operand.components[i], coord, 1), layout)
            if layout == :g
                get_grid_data(result) .+= get_grid_data(component_deriv)
            else
                get_coeff_data(result) .+= get_coeff_data(component_deriv)
            end
        end

        return result
    else
        throw(ArgumentError("Divergence not implemented for operand type $(typeof(operand))"))
    end
end

# ============================================================================
# Differentiate Evaluation
# ============================================================================

"""Evaluate differentiation operator."""
function evaluate_differentiate(diff_op::Differentiate, layout::Symbol=:g)
    operand = diff_op.operand
    coord = diff_op.coord
    order = diff_op.order

    # VectorField: differentiate each component, return VectorField
    if isa(operand, VectorField)
        diff_comps = [evaluate_differentiate(Differentiate(c, coord, order), layout)
                      for c in operand.components]
        # Create result with differentiated component bases (may differ from
        # original for Chebyshev: ChebyshevT → ChebyshevU after differentiation)
        result = VectorField(operand.dist, operand.coordsys,
                             "d$(order)_$(operand.name)_d$(coord.name)",
                             diff_comps[1].bases, operand.dtype)
        for (i, dc) in enumerate(diff_comps)
            copy_field_data!(result.components[i], dc)
            result.components[i].current_layout = dc.current_layout
        end
        return result
    end

    if !isa(operand, ScalarField)
        throw(ArgumentError(
            "Differentiation requires ScalarField or VectorField, got $(typeof(operand))"))
    end

    # Short-circuit for zero-order derivative (identity operation)
    if order == 0
        result = checkout_or_alloc(operand.bases, operand.dtype, operand.dist)
        copy_field_data!(result, operand)
        result.current_layout = operand.current_layout
        result.name = "d0_$(operand.name)"
        ensure_layout!(result, layout)
        return result
    end

    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        # Coordinate not present in bases (constant dimension): derivative is zero
        result = checkout_or_alloc(operand.bases, operand.dtype, operand.dist)
        copy_field_data!(result, operand)
        result.current_layout = operand.current_layout
        result.name = "d$(order)_$(operand.name)_d$(coord.name)$(order)"
        ensure_layout!(result, layout)

        # Zero out the data
        if layout == :g
            grid_data = get_grid_data(result)
            if grid_data !== nothing
                if isa(grid_data, PencilArrays.PencilArray)
                    fill!(parent(grid_data), zero(eltype(grid_data)))
                else
                    fill!(grid_data, zero(eltype(grid_data)))
                end
            end
        else
            coeff_data = get_coeff_data(result)
            if coeff_data !== nothing
                if isa(coeff_data, PencilArrays.PencilArray)
                    fill!(parent(coeff_data), zero(eltype(coeff_data)))
                else
                    fill!(coeff_data, zero(eltype(coeff_data)))
                end
            end
        end

        return result
    end

    basis = operand.bases[basis_index]
    result = checkout_or_alloc(operand.bases, operand.dtype, operand.dist)
    copy_field_data!(result, operand)
    result.current_layout = operand.current_layout
    result.name = "d$(order)_$(operand.name)_d$(coord.name)$(order)"

    # Apply differentiation based on basis type
    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        evaluate_fourier_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, ChebyshevT)
        evaluate_chebyshev_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, Legendre)
        evaluate_legendre_derivative!(result, operand, basis_index, order, layout)
    else
        throw(ArgumentError(
            "Differentiation not implemented for basis type $(typeof(basis)). " *
            "Supported basis types: RealFourier, ComplexFourier, ChebyshevT, Legendre. " *
            "Check that the coordinate '$(coord.name)' has a valid basis assigned."))
    end

    return result
end
