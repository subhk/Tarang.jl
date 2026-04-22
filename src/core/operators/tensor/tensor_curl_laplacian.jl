"""
    Curl and Laplacian operators

This file contains curl and standard Laplacian evaluation for scalar and
vector operands.
"""

# ============================================================================
# Curl Evaluation
# ============================================================================

"""
    evaluate_curl(curl_op::Curl, layout::Symbol=:g)

Evaluate curl of a vector field.
2D: curl(v) = dv_y/dx - dv_x/dy (scalar)
3D: curl(v) = (dv_z/dy - dv_y/dz, dv_x/dz - dv_z/dx, dv_y/dx - dv_x/dy)
"""
function evaluate_curl(curl_op::Curl, layout::Symbol=:g)
    operand = curl_op.operand
    coordsys = curl_op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires a VectorField"))
    end

    dim = length(operand.components)

    if dim == 2
        # 2D curl: returns scalar
        return evaluate_curl_2d(operand, coordsys, layout)
    elseif dim == 3
        # 3D curl: returns vector
        return evaluate_curl_3d(operand, coordsys, layout)
    else
        throw(ArgumentError("Curl only implemented for 2D and 3D"))
    end
end

function evaluate_curl_2d(operand::VectorField, coordsys::CoordinateSystem, layout::Symbol)
    # curl(v) = dv_y/dx - dv_x/dy
    vx = operand.components[1]
    vy = operand.components[2]

    coord_x = coordsys.coords[1]
    coord_y = coordsys.coords[2]

    # dv_y/dx
    dvy_dx = evaluate_differentiate(Differentiate(vy, coord_x, 1), layout)

    # dv_x/dy
    dvx_dy = evaluate_differentiate(Differentiate(vx, coord_y, 1), layout)

    # Result = dvy_dx - dvx_dy
    result = ScalarField(operand.dist, "curl_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, layout)

    if layout == :g
        get_grid_data(result) .= get_grid_data(dvy_dx) .- get_grid_data(dvx_dy)
    else
        get_coeff_data(result) .= get_coeff_data(dvy_dx) .- get_coeff_data(dvx_dy)
    end

    return result
end

function evaluate_curl_3d(operand::VectorField, coordsys::CoordinateSystem, layout::Symbol)
    vx = operand.components[1]
    vy = operand.components[2]
    vz = operand.components[3]

    coord_x = coordsys.coords[1]
    coord_y = coordsys.coords[2]
    coord_z = coordsys.coords[3]

    # Component 1: dv_z/dy - dv_y/dz
    dvz_dy = evaluate_differentiate(Differentiate(vz, coord_y, 1), layout)
    dvy_dz = evaluate_differentiate(Differentiate(vy, coord_z, 1), layout)

    # Component 2: dv_x/dz - dv_z/dx
    dvx_dz = evaluate_differentiate(Differentiate(vx, coord_z, 1), layout)
    dvz_dx = evaluate_differentiate(Differentiate(vz, coord_x, 1), layout)

    # Component 3: dv_y/dx - dv_x/dy
    dvy_dx = evaluate_differentiate(Differentiate(vy, coord_x, 1), layout)
    dvx_dy = evaluate_differentiate(Differentiate(vx, coord_y, 1), layout)

    result = VectorField(operand.dist, coordsys, "curl_$(operand.name)", operand.bases, operand.dtype)

    for comp in result.components
        ensure_layout!(comp, layout)
    end

    # Set component data
    if layout == :g
        get_grid_data(result.components[1]) .= get_grid_data(dvz_dy) .- get_grid_data(dvy_dz)
        get_grid_data(result.components[2]) .= get_grid_data(dvx_dz) .- get_grid_data(dvz_dx)
        get_grid_data(result.components[3]) .= get_grid_data(dvy_dx) .- get_grid_data(dvx_dy)
    else
        get_coeff_data(result.components[1]) .= get_coeff_data(dvz_dy) .- get_coeff_data(dvy_dz)
        get_coeff_data(result.components[2]) .= get_coeff_data(dvx_dz) .- get_coeff_data(dvz_dx)
        get_coeff_data(result.components[3]) .= get_coeff_data(dvy_dx) .- get_coeff_data(dvx_dy)
    end

    return result
end

# ============================================================================
# Laplacian Evaluation
# ============================================================================

"""
    evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)

Evaluate Laplacian operator.
nabla^2 f = Sigma_i d^2f/dx_i^2
"""
function evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)
    operand = lap_op.operand

    if isa(operand, ScalarField)
        return evaluate_scalar_laplacian(operand, layout)
    elseif isa(operand, VectorField)
        return evaluate_vector_laplacian(operand, layout)
    else
        throw(ArgumentError("Laplacian not implemented for $(typeof(operand))"))
    end
end

function evaluate_scalar_laplacian(operand::ScalarField, layout::Symbol)
    result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, layout)

    if layout == :g
        fill!(get_grid_data(result), 0.0)
    else
        fill!(get_coeff_data(result), 0.0)
    end

    for (i, basis) in enumerate(operand.bases)
        # Find coordinate for this basis via CoordinateSystem indexing
        coord = basis.meta.coordsys[basis.meta.element_label]

        # Second derivative
        d2f = evaluate_differentiate(Differentiate(operand, coord, 2), layout)

        if layout == :g
            get_grid_data(result) .+= get_grid_data(d2f)
        else
            get_coeff_data(result) .+= get_coeff_data(d2f)
        end
    end

    return result
end

function evaluate_vector_laplacian(operand::VectorField, layout::Symbol)
    result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                        operand.bases, operand.dtype)

    for (i, comp) in enumerate(operand.components)
        lap_comp = evaluate_scalar_laplacian(comp, layout)

        ensure_layout!(result.components[i], layout)
        if layout == :g
            copyto!(get_grid_data(result.components[i]), get_grid_data(lap_comp))
        else
            copyto!(get_coeff_data(result.components[i]), get_coeff_data(lap_comp))
        end
    end

    return result
end
