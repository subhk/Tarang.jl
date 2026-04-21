# Evaluation helpers for concrete Cartesian operators.
#
# This file owns the realized field computations after the operator structs and
# matrix/layout rules are defined.

# ============================================================================
# Evaluation functions for Cartesian operators
# ============================================================================

"""
    evaluate_cartesian_gradient(op::CartesianGradient, layout::Symbol=:g)

Evaluate Cartesian gradient operator.
"""
function evaluate_cartesian_gradient(op::CartesianGradient, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # Scalar → Vector: (∇f)_i = ∂f/∂x_i
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)

        for (i, coord) in enumerate(coordsys.coords)
            deriv_op = Differentiate(operand, coord, 1)
            result.components[i] = evaluate_differentiate(deriv_op, layout)
        end

        return result

    elseif isa(operand, VectorField)
        # Vector → Tensor: (∇u)_ij = ∂u_j/∂x_i (first index = derivative direction)
        result = TensorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)

        for (i, coord) in enumerate(coordsys.coords)
            for (j, comp) in enumerate(operand.components)
                deriv_op = Differentiate(comp, coord, 1)
                result.components[i, j] = evaluate_differentiate(deriv_op, layout)
            end
        end

        return result

    else
        throw(ArgumentError("CartesianGradient requires ScalarField or VectorField operand, got $(typeof(operand))"))
    end
end

"""
    evaluate_cartesian_divergence(op::CartesianDivergence, layout::Symbol=:g)

Evaluate Cartesian divergence operator.
"""
function evaluate_cartesian_divergence(op::CartesianDivergence, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("CartesianDivergence requires VectorField"))
    end

    # Create scalar field for result
    result = ScalarField(operand.dist, "div_$(operand.name)", operand.components[1].bases, operand.dtype)
    ensure_layout!(result, layout)

    # Initialize to zero
    if layout == :g
        fill!(get_grid_data(result), 0.0)
    else
        fill!(get_coeff_data(result), 0.0)
    end

    # Sum partial derivatives of components
    for (i, coord) in enumerate(coordsys.coords)
        comp = operand.components[i]
        deriv_op = Differentiate(comp, coord, 1)
        comp_deriv = evaluate_differentiate(deriv_op, layout)

        if layout == :g
            get_grid_data(result) .+= get_grid_data(comp_deriv)
        else
            get_coeff_data(result) .+= get_coeff_data(comp_deriv)
        end
    end

    return result
end

"""
    evaluate_cartesian_curl(op::CartesianCurl, layout::Symbol=:g)

Evaluate Cartesian curl operator (3D only).
"""
function evaluate_cartesian_curl(op::CartesianCurl, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("CartesianCurl requires VectorField"))
    end

    if coordsys.dim != 3
        throw(ArgumentError("CartesianCurl is only for 3D. For 2D, use skew(grad(f))."))
    end

    # Create result vector field
    result = VectorField(operand.dist, coordsys, "curl_$(operand.name)", operand.components[1].bases, operand.dtype)

    cx, cy, cz = coordsys.coords
    ux, uy, uz = operand.components

    # curl_x = ∂uz/∂y - ∂uy/∂z
    duz_dy = evaluate_differentiate(Differentiate(uz, cy, 1), layout)
    duy_dz = evaluate_differentiate(Differentiate(uy, cz, 1), layout)
    result.components[1] = field_subtract(duz_dy, duy_dz, layout)

    # curl_y = ∂ux/∂z - ∂uz/∂x
    dux_dz = evaluate_differentiate(Differentiate(ux, cz, 1), layout)
    duz_dx = evaluate_differentiate(Differentiate(uz, cx, 1), layout)
    result.components[2] = field_subtract(dux_dz, duz_dx, layout)

    # curl_z = ∂uy/∂x - ∂ux/∂y
    duy_dx = evaluate_differentiate(Differentiate(uy, cx, 1), layout)
    dux_dy = evaluate_differentiate(Differentiate(ux, cy, 1), layout)
    result.components[3] = field_subtract(duy_dx, dux_dy, layout)

    # Handle handedness
    if coordsys.right_handed === false
        for comp in result.components
            if layout == :g
                get_grid_data(comp) .*= -1
            else
                get_coeff_data(comp) .*= -1
            end
        end
    end

    return result
end

"""
    field_subtract(a, b, layout)

Subtract two scalar fields.
"""
function field_subtract(a::ScalarField, b::ScalarField, layout::Symbol)
    result = ScalarField(a.dist, "sub", a.bases, a.dtype)
    ensure_layout!(result, layout)
    ensure_layout!(a, layout)
    ensure_layout!(b, layout)

    if layout == :g
        get_grid_data(result) .= get_grid_data(a) .- get_grid_data(b)
    else
        get_coeff_data(result) .= get_coeff_data(a) .- get_coeff_data(b)
    end

    return result
end

"""
    evaluate_cartesian_laplacian(op::CartesianLaplacian, layout::Symbol=:g)

Evaluate Cartesian Laplacian operator.
"""
function evaluate_cartesian_laplacian(op::CartesianLaplacian, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # Create result field
        result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(result, layout)

        # Initialize to zero
        if layout == :g
            fill!(get_grid_data(result), 0.0)
        else
            fill!(get_coeff_data(result), 0.0)
        end

        # Sum second derivatives
        for coord in coordsys.coords
            d2_op = Differentiate(operand, coord, 2)
            d2 = evaluate_differentiate(d2_op, layout)

            if layout == :g
                get_grid_data(result) .+= get_grid_data(d2)
            else
                get_coeff_data(result) .+= get_coeff_data(d2)
            end
        end

        return result
    elseif isa(operand, VectorField)
        # Apply Laplacian to each component
        result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                            operand.components[1].bases, operand.dtype)

        for (i, comp) in enumerate(operand.components)
            lap_op = CartesianLaplacian(comp, coordsys)
            result.components[i] = evaluate_cartesian_laplacian(lap_op, layout)
        end

        return result
    else
        throw(ArgumentError("CartesianLaplacian not implemented for $(typeof(operand))"))
    end
end

"""
    evaluate_cartesian_trace(op::CartesianTrace, layout::Symbol=:g)

Evaluate Cartesian trace operator.
"""
function evaluate_cartesian_trace(op::CartesianTrace, layout::Symbol=:g)
    operand = op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("CartesianTrace requires TensorField"))
    end

    ensure_layout!(operand, layout)

    # Sum diagonal elements
    dim = size(operand.components, 1)  # Get dimension from matrix size

    # Check for empty tensor
    if dim == 0 || isempty(operand.components)
        throw(ArgumentError("CartesianTrace requires non-empty TensorField"))
    end

    # Get first component to create result
    first_comp = operand.components[1, 1]
    result = ScalarField(first_comp.dist, "trace_$(operand.name)", first_comp.bases, first_comp.dtype)
    ensure_layout!(result, layout)

    # Initialize to zero
    if layout == :g
        fill!(get_grid_data(result), 0.0)
        for i in 1:dim
            get_grid_data(result) .+= get_grid_data(operand.components[i, i])
        end
    else
        fill!(get_coeff_data(result), 0.0)
        for i in 1:dim
            get_coeff_data(result) .+= get_coeff_data(operand.components[i, i])
        end
    end

    return result
end

"""
    evaluate_cartesian_skew(op::CartesianSkew, layout::Symbol=:g)

Evaluate Cartesian skew operator (2D vector rotation by 90°).
"""
function evaluate_cartesian_skew(op::CartesianSkew, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("CartesianSkew requires VectorField"))
    end

    ensure_layout!(operand, layout)

    # skew(u_x, u_y) = (-u_y, u_x)
    result = VectorField(operand.dist, coordsys, "skew_$(operand.name)",
                        operand.components[1].bases, operand.dtype)

    for comp in result.components
        ensure_layout!(comp, layout)
    end

    # skew(u_x, u_y) = (-u_y, u_x): first component is negated second, second is first
    if layout == :g
        get_grid_data(result.components[1]) .= .-get_grid_data(operand.components[2])
        copyto!(get_grid_data(result.components[2]), get_grid_data(operand.components[1]))
    else
        get_coeff_data(result.components[1]) .= .-get_coeff_data(operand.components[2])
        copyto!(get_coeff_data(result.components[2]), get_coeff_data(operand.components[1]))
    end

    return result
end

"""
    _evaluate_skew_vector(operand::VectorField, layout::Symbol)

Implementation of skew for VectorField, called from operators.jl's evaluate_skew.
Computes 2D vector rotation: skew(u_x, u_y) = (-u_y, u_x).
Used for 2D QG turbulence: u = skew(grad(ψ)) gives divergence-free velocity.
"""
function _evaluate_skew_vector(operand::VectorField, layout::Symbol)
    return evaluate_cartesian_skew(CartesianSkew(operand), layout)
end

# ============================================================================
