# Dispatch glue, fallback matrix semantics, and evaluate integration.

# ============================================================================
# Multiclass dispatch integration
# Following MultiClass metaclass pattern
# ============================================================================

"""
    dispatch_cartesian_operator(OpType, operand, args...; kwargs...)

Dispatch to Cartesian-specific or DirectProduct-specific operator variant
based on coordinate system type.
"""
function dispatch_cartesian_operator(::Type{Gradient}, operand, coordsys; kwargs...)
    # Cartesian coordinates use specialized fast paths; DirectProduct keeps the
    # same public operator API for composed coordinate systems.
    if isa(coordsys, CartesianCoordinates) || isa(coordsys, Coordinate)
        return CartesianGradient(operand, coordsys)
    elseif isa(coordsys, DirectProduct)
        return DirectProductGradient(operand, coordsys)
    else
        throw(ArgumentError("Gradient not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

function dispatch_cartesian_operator(::Type{Divergence}, operand; index::Int=0, kwargs...)
    tensorsig = get_tensorsig(operand)
    if isempty(tensorsig)
        throw(ArgumentError("Divergence requires a tensor operand"))
    end

    # `index` selects which tensor axis the divergence consumes.
    coordsys = tensorsig[index + 1]

    if isa(coordsys, CartesianCoordinates) || isa(coordsys, Coordinate)
        return CartesianDivergence(operand; index=index)
    elseif isa(coordsys, DirectProduct)
        return DirectProductDivergence(operand; index=index)
    else
        throw(ArgumentError("Divergence not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

function dispatch_cartesian_operator(::Type{Curl}, operand; index::Int=0, kwargs...)
    tensorsig = get_tensorsig(operand)
    if isempty(tensorsig)
        throw(ArgumentError("Curl requires a tensor operand"))
    end

    coordsys = tensorsig[index + 1]

    if isa(coordsys, CartesianCoordinates)
        return CartesianCurl(operand; index=index)
    elseif isa(coordsys, DirectProduct)
        return DirectProductCurl(operand; index=index)
    else
        throw(ArgumentError("Curl not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

function dispatch_cartesian_operator(::Type{Laplacian}, operand, coordsys; kwargs...)
    if isa(coordsys, CartesianCoordinates) || isa(coordsys, Coordinate)
        return CartesianLaplacian(operand, coordsys)
    elseif isa(coordsys, DirectProduct)
        return DirectProductLaplacian(operand, coordsys)
    else
        throw(ArgumentError("Laplacian not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

# ============================================================================
# Generic matrix operation fallbacks
# ============================================================================

"""
    matrix_dependence(operand, vars...)

Default matrix dependence for operands (not operators).
"""
function matrix_dependence(operand::Operand, vars...)
    # Check if operand is one of the variables
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if operand === var ||
           (hasfield(typeof(operand), :name) && hasfield(typeof(var), :name) && operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(op::Differentiate, vars...)

Matrix coupling for Differentiate operator.
A linear operator that couples its operand variable into the equation.
"""
function matrix_coupling(op::Differentiate, vars...)
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if op.operand === var ||
           (hasfield(typeof(op.operand), :name) && hasfield(typeof(var), :name) &&
            op.operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(operand, vars...)

Default matrix coupling for operands (not operators).
"""
function matrix_coupling(operand::Operand, vars...)
    # Single variable doesn't couple to others by default
    return falses(length(vars))
end

# ============================================================================
# Integration with evaluate dispatcher
# ============================================================================

# Extend the main evaluate function to handle Cartesian operators
function evaluate(op::CartesianComponent, layout::Symbol=:g)
    return evaluate_cartesian_component(op, layout)
end

function evaluate(op::CartesianGradient, layout::Symbol=:g)
    return evaluate_cartesian_gradient(op, layout)
end

function evaluate(op::CartesianDivergence, layout::Symbol=:g)
    return evaluate_cartesian_divergence(op, layout)
end

function evaluate(op::CartesianCurl, layout::Symbol=:g)
    return evaluate_cartesian_curl(op, layout)
end

function evaluate(op::CartesianLaplacian, layout::Symbol=:g)
    return evaluate_cartesian_laplacian(op, layout)
end

function evaluate(op::CartesianTrace, layout::Symbol=:g)
    return evaluate_cartesian_trace(op, layout)
end

function evaluate(op::CartesianSkew, layout::Symbol=:g)
    return evaluate_cartesian_skew(op, layout)
end

# DirectProduct evaluation methods
function evaluate(op::DirectProductGradient, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # A DirectProduct gradient is assembled component-by-component by
        # differentiating along each coordinate of the product space.
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        for (i, coord) in enumerate(coordsys.coords)
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end
        return result
    end

    throw(ArgumentError("DirectProductGradient evaluation only implemented for scalar fields"))
end

function evaluate(op::DirectProductDivergence, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("DirectProductDivergence requires VectorField"))
    end

    result = ScalarField(operand.dist, "div_$(operand.name)", operand.components[1].bases, operand.dtype)
    ensure_layout!(result, layout)

    if layout == :g
        fill!(get_grid_data(result), 0.0)
        for (i, coord) in enumerate(coordsys.coords)
            # Sum directional derivatives into the scalar divergence field.
            comp = operand.components[i]
            deriv = evaluate_differentiate(Differentiate(comp, coord, 1), layout)
            get_grid_data(result) .+= get_grid_data(deriv)
        end
    else
        fill!(get_coeff_data(result), 0.0)
        for (i, coord) in enumerate(coordsys.coords)
            # Coefficient-space divergence follows the same accumulation, but
            # reads/writes spectral buffers.
            comp = operand.components[i]
            deriv = evaluate_differentiate(Differentiate(comp, coord, 1), layout)
            get_coeff_data(result) .+= get_coeff_data(deriv)
        end
    end

    return result
end

function evaluate(op::DirectProductLaplacian, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(result, layout)

        if layout == :g
            fill!(get_grid_data(result), 0.0)
            for coord in coordsys.coords
                d2 = evaluate_differentiate(Differentiate(operand, coord, 2), layout)
                get_grid_data(result) .+= get_grid_data(d2)
            end
        else
            fill!(get_coeff_data(result), 0.0)
            for coord in coordsys.coords
                d2 = evaluate_differentiate(Differentiate(operand, coord, 2), layout)
                get_coeff_data(result) .+= get_coeff_data(d2)
            end
        end
        return result
    elseif isa(operand, VectorField)
        result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                            operand.components[1].bases, operand.dtype)
        for (i, comp) in enumerate(operand.components)
            lap_comp = evaluate(DirectProductLaplacian(comp, coordsys), layout)
            result.components[i] = lap_comp
        end
        return result
    end

    throw(ArgumentError("DirectProductLaplacian not implemented for $(typeof(operand))"))
end

"""
    evaluate(op::DirectProductTrace, layout::Symbol=:g)

Evaluate trace of a tensor over DirectProduct coordinates.
Sums traces of diagonal blocks for each coordinate subsystem.
Following spectral methods pattern.
"""
function evaluate(op::DirectProductTrace, layout::Symbol=:g)
    # Sum traces of all diagonal blocks
    result = nothing
    for trace_op in op.args
        trace_result = evaluate(trace_op, layout)
        if result === nothing
            result = trace_result
        else
            # Add trace results
            if layout == :g
                get_grid_data(result) .+= get_grid_data(trace_result)
            else
                get_coeff_data(result) .+= get_coeff_data(trace_result)
            end
        end
    end
    return result
end

"""
    evaluate(op::DirectProductCurl, layout::Symbol=:g)

Evaluate DirectProduct curl operator (3D only).
"""
function evaluate(op::DirectProductCurl, layout::Symbol=:g)
    return evaluate_direct_product_curl(op, layout)
end

"""
    evaluate_direct_product_curl(op::DirectProductCurl, layout::Symbol=:g)

Evaluate curl for a 3D vector field on DirectProduct coordinates.

Uses the standard Levi-Civita formula: curl(u)_i = ε_ijk ∂u_k/∂x_j
applied to the DirectProduct's coordinate ordering.
"""
function evaluate_direct_product_curl(op::DirectProductCurl, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("DirectProductCurl requires VectorField"))
    end

    # Create result vector field
    result = VectorField(operand.dist, coordsys, "curl_$(operand.name)",
                         operand.components[1].bases, operand.dtype)

    c1, c2, c3 = coordsys.coords
    u1, u2, u3 = operand.components

    # curl_1 = ∂u_3/∂c_2 - ∂u_2/∂c_3
    du3_dc2 = evaluate_differentiate(Differentiate(u3, c2, 1), layout)
    du2_dc3 = evaluate_differentiate(Differentiate(u2, c3, 1), layout)
    result.components[1] = field_subtract(du3_dc2, du2_dc3, layout)

    # curl_2 = ∂u_1/∂c_3 - ∂u_3/∂c_1
    du1_dc3 = evaluate_differentiate(Differentiate(u1, c3, 1), layout)
    du3_dc1 = evaluate_differentiate(Differentiate(u3, c1, 1), layout)
    result.components[2] = field_subtract(du1_dc3, du3_dc1, layout)

    # curl_3 = ∂u_2/∂c_1 - ∂u_1/∂c_2
    du2_dc1 = evaluate_differentiate(Differentiate(u2, c1, 1), layout)
    du1_dc2 = evaluate_differentiate(Differentiate(u1, c2, 1), layout)
    result.components[3] = field_subtract(du2_dc1, du1_dc2, layout)

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
    evaluate(op::DirectProductComponent, layout::Symbol=:g)

Extract a sub-vector from a DirectProduct VectorField.

!!! note "Return Value Semantics"
    The returned VectorField contains **references** to the original operand's
    components (not copies). Modifications to the returned components will
    affect the original VectorField.
"""
function evaluate(op::DirectProductComponent, layout::Symbol=:g)
    operand = op.operand
    if !isa(operand, VectorField)
        throw(ArgumentError("DirectProductComponent requires VectorField"))
    end

    if !isa(operand.coordsys, DirectProduct)
        throw(ArgumentError("DirectProductComponent requires DirectProduct coordinates"))
    end

    coordsys = operand.coordsys
    subaxis = get(coordsys.subaxis_by_cs, op.comp, nothing)
    if subaxis === nothing
        idx = findfirst(cs -> cs.names == op.comp.names, coordsys.coordsystems)
        if idx === nothing
            throw(ArgumentError("Component coordinate system not found in DirectProduct"))
        end
        subaxis = coordsys.subaxis_by_cs[coordsys.coordsystems[idx]]
    end

    dim = op.comp.dim
    start_idx = subaxis + 1
    end_idx = start_idx + dim - 1

    comp_tag = isempty(op.comp.names) ? "component" : join(op.comp.names, "")
    result = VectorField(operand.dist, op.comp, "$(operand.name)_$(comp_tag)", operand.bases, operand.dtype)
    for i in 1:dim
        result.components[i] = operand.components[start_idx + i - 1]
        ensure_layout!(result.components[i], layout)
    end

    return result
end
