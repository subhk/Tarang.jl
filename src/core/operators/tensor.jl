"""
    Tensor operations and special operators

This file contains evaluation functions for:
- Trace, skew, transpose_components
- Curl (2D and 3D)
- Laplacian (scalar and vector)
- Fractional Laplacian
- Outer product
- AdvectiveCFL
"""

using LinearAlgebra
using SparseArrays

# ============================================================================
# Trace Evaluation
# ============================================================================

"""
    evaluate_trace(trace_op::Trace, layout::Symbol=:g)

Evaluate trace of a tensor field.
trace(T) = Sigma_i T_ii

For DirectProduct coordinates, dispatches to DirectProductTrace which sums
traces of diagonal blocks for each coordinate subsystem.
Following Dedalus operators.py:1852-1875 CartesianTrace and DirectProductTrace.
"""
function evaluate_trace(trace_op::Trace, layout::Symbol=:g)
    operand = trace_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField"))
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

# ============================================================================
# Fractional Laplacian Evaluation
# ============================================================================

"""
    evaluate_fractional_laplacian(frac_lap::FractionalLaplacian, layout::Symbol=:g)

Evaluate fractional Laplacian operator: (-Delta)^alpha

In spectral space, this multiplies each Fourier coefficient by |k|^(2*alpha),
where k = sqrt(k1^2 + k2^2 + ...) is the wavenumber magnitude.

For alpha > 0: High-order dissipation (smoothing)
For alpha < 0: Inverse operation (integration/smoothing)
For alpha = 1: Negative Laplacian, (-Δ)^1 = -Δ (note: opposite sign from Laplacian())
For alpha = 1/2: Square root Laplacian (SQG dissipation)
For alpha = -1/2: Inverse square root (SQG streamfunction from buoyancy)

Note: For alpha < 0, the k=0 mode is handled specially to avoid division by zero.
The k=0 mode is set to zero (removes mean).
"""
function evaluate_fractional_laplacian(frac_lap::FractionalLaplacian, layout::Symbol=:g)
    operand = frac_lap.operand
    alpha = frac_lap.α

    if isa(operand, ScalarField)
        return evaluate_scalar_fractional_laplacian(operand, alpha, layout)
    elseif isa(operand, VectorField)
        return evaluate_vector_fractional_laplacian(operand, alpha, layout)
    else
        throw(ArgumentError("Fractional Laplacian not implemented for $(typeof(operand))"))
    end
end

function evaluate_scalar_fractional_laplacian(operand::ScalarField, alpha::Float64, layout::Symbol)
    # Fractional Laplacian is only defined for Fourier bases (wavenumber-based).
    # For Chebyshev/Legendre bases, the wavenumber grid would be all zeros,
    # producing a silently wrong zero result.
    has_fourier = any(isa(b, Union{RealFourier, ComplexFourier}) for b in operand.bases)
    if !has_fourier
        throw(ArgumentError("Fractional Laplacian requires at least one Fourier basis. " *
            "For Chebyshev/Legendre fields, use a matrix-based approach instead."))
    end

    # Create result field
    alpha_str = alpha >= 0 ? "$(alpha)" : "m$(abs(alpha))"
    result = ScalarField(operand.dist, "fraclap$(alpha_str)_$(operand.name)", operand.bases, operand.dtype)

    # Work on a copy to avoid mutating the caller's field layout
    work = copy(operand)
    ensure_layout!(work, :c)
    ensure_layout!(result, :c)

    # Get wavenumber grids for each Fourier basis
    k_squared_total = compute_wavenumber_squared_grid(work)

    # Compute |k|^(2*alpha) factor
    if alpha >= 0
        k_factor = k_squared_total .^ alpha
    else
        # For inverse operations, set k=0 mode to zero
        k_factor = similar(k_squared_total)
        threshold = 1e-14
        k_factor .= ifelse.(k_squared_total .> threshold, k_squared_total .^ alpha, zero(eltype(k_squared_total)))
    end

    # Apply the fractional Laplacian in spectral space
    get_coeff_data(result) .= get_coeff_data(work) .* k_factor

    # Transform to requested layout if needed
    if layout == :g
        ensure_layout!(result, :g)
    end

    return result
end

function evaluate_vector_fractional_laplacian(operand::VectorField, alpha::Float64, layout::Symbol)
    alpha_str = alpha >= 0 ? "$(alpha)" : "m$(abs(alpha))"
    result = VectorField(operand.dist, operand.coordsys, "fraclap$(alpha_str)_$(operand.name)",
                        operand.bases, operand.dtype)

    for (i, comp) in enumerate(operand.components)
        frac_lap_comp = evaluate_scalar_fractional_laplacian(comp, alpha, layout)

        ensure_layout!(result.components[i], layout)
        if layout == :g
            copyto!(get_grid_data(result.components[i]), get_grid_data(frac_lap_comp))
        else
            copyto!(get_coeff_data(result.components[i]), get_coeff_data(frac_lap_comp))
        end
    end

    return result
end

"""
    compute_wavenumber_squared_grid(field::ScalarField)

Compute |k|^2 = k1^2 + k2^2 + ... for each point in spectral space.
Supports both CPU and GPU arrays.

Returns an array with the same shape as get_coeff_data(field) containing
the squared wavenumber magnitude at each spectral coefficient location.
"""
function compute_wavenumber_squared_grid(field::ScalarField)
    bases = field.bases
    data_shape = size(get_coeff_data(field))

    # Initialize with zeros on the same device as get_coeff_data(field)
    k_squared = similar_zeros(get_coeff_data(field), Float64, data_shape...)

    # Add contribution from each basis
    for (axis, basis) in enumerate(bases)
        if isa(basis, RealFourier)
            # For RealFourier, check if we're in RFFT layout (N/2+1) or native layout (N)
            N = basis.meta.size
            actual_size = data_shape[axis]
            rfft_size = N ÷ 2 + 1

            if actual_size == rfft_size
                # RFFT layout: [k=0, k=1, ..., k=N/2]
                k_axis_cpu = wavenumbers_rfft(basis)
            else
                # Native cos/sin layout: [cos0, cos1, msin1, cos2, msin2, ...]
                # Both cos(k) and msin(k) have the same wavenumber k
                k_axis_cpu = wavenumbers(basis)
            end

            add_wavenumber_squared_contribution!(k_squared, k_axis_cpu, axis, length(bases))

        elseif isa(basis, ComplexFourier)
            # Get wavenumbers for ComplexFourier basis
            k_axis_cpu = wavenumbers(basis)
            add_wavenumber_squared_contribution!(k_squared, k_axis_cpu, axis, length(bases))
        end
        # For Chebyshev/Legendre bases, no wavenumber contribution
    end

    return k_squared
end

"""
Add k^2 contribution from one axis to the total wavenumber grid.
Works with both CPU and GPU arrays.
"""
function add_wavenumber_squared_contribution!(k_squared::AbstractArray, k_axis_cpu::Vector{Float64}, axis::Int, ndims::Int)
    # Create shape for broadcasting
    shape = ones(Int, ndims)
    shape[axis] = length(k_axis_cpu)

    # Move k_axis to the same device as k_squared if needed
    if is_gpu_array(k_squared)
        k_axis = copy_to_device(k_axis_cpu, k_squared)
    else
        k_axis = k_axis_cpu
    end

    # Reshape k_axis for broadcasting
    k_reshaped = reshape(k_axis, Tuple(shape))

    # Add k^2 contribution
    k_squared .+= k_reshaped.^2
end

# ============================================================================
# Fractional Laplacian - Matrix methods
# ============================================================================

"""
    matrix_dependence(op::FractionalLaplacian, vars...)

Determine which variables this fractional Laplacian operator depends on.
"""
function matrix_dependence(op::FractionalLaplacian, vars...)
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if op.operand === var || (hasfield(typeof(op.operand), :name) &&
                                   hasfield(typeof(var), :name) &&
                                   op.operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(op::FractionalLaplacian, vars...)

Fractional Laplacian only couples a variable to itself.
"""
function matrix_coupling(op::FractionalLaplacian, vars...)
    return matrix_dependence(op, vars...)
end

"""
    subproblem_matrix(op::FractionalLaplacian, subproblem)

Build the sparse matrix representation of fractional Laplacian for implicit solvers.
"""
function subproblem_matrix(op::FractionalLaplacian, subproblem)
    operand = op.operand
    alpha = op.α

    if isa(operand, ScalarField)
        n = prod(size(get_coeff_data(operand)))
    else
        throw(ArgumentError("subproblem_matrix for FractionalLaplacian only implemented for ScalarField"))
    end

    # Compute wavenumber squared grid
    k_squared = compute_wavenumber_squared_grid(operand)

    # Convert to CPU if on GPU
    if is_gpu_array(k_squared)
        k_squared = Array(k_squared)
    end

    k_squared_flat = vec(k_squared)

    # Compute |k|^(2*alpha) diagonal entries
    if alpha >= 0
        diag_entries = k_squared_flat .^ alpha
    else
        threshold = 1e-14
        diag_entries = ifelse.(k_squared_flat .> threshold, k_squared_flat .^ alpha, 0.0)
    end

    return spdiagm(0 => diag_entries)
end

"""Check conditions for fractional Laplacian."""
function check_conditions(op::FractionalLaplacian)
    operand = op.operand

    if isa(operand, ScalarField)
        if hasfield(typeof(operand), :current_layout)
            layout = operand.current_layout
            if layout == :c
                return get_coeff_data(operand) !== nothing
            elseif layout == :g
                return get_grid_data(operand) !== nothing
            end
        end
    elseif isa(operand, VectorField)
        for comp in operand.components
            if !check_conditions(FractionalLaplacian(comp, op.α))
                return false
            end
        end
    end

    return true
end

"""Enforce conditions for fractional Laplacian."""
function enforce_conditions(op::FractionalLaplacian)
    operand = op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
    elseif isa(operand, VectorField)
        for comp in operand.components
            ensure_layout!(comp, :c)
        end
    end
end

"""Fractional Laplacian is linear."""
is_linear(op::FractionalLaplacian) = true

"""Return effective derivative order: 2*alpha."""
operator_order(op::FractionalLaplacian) = 2 * op.α

# ============================================================================
# TransposeComponents Matrix Methods
# ============================================================================

"""
    _transpose_matrix(op::TransposeComponents)

Build permutation matrix for transposing tensor components.
"""
function _transpose_matrix(op::TransposeComponents)
    operand = op.operand
    i0, i1 = op.indices

    if !isa(operand, TensorField)
        throw(ArgumentError("_transpose_matrix requires TensorField"))
    end

    dim = size(operand.components, 1)
    n = dim * dim

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # For a rank-2 tensor T_{ij}, transposing indices (i0, i1) maps T_{ij} -> T_{ji}.
    # In flat storage: input at i*dim+j goes to output at j*dim+i.
    # This correctly swaps the index POSITIONS (not values independently).
    for i in 0:(dim-1)
        for j in 0:(dim-1)
            input_idx = i * dim + j + 1
            output_idx = j * dim + i + 1
            push!(rows, output_idx)
            push!(cols, input_idx)
            push!(vals, 1.0)
        end
    end

    return sparse(rows, cols, vals, n, n)
end

"""Build operator matrix for TransposeComponents."""
function subproblem_matrix(op::TransposeComponents, subproblem)
    transpose_mat = _transpose_matrix(op)

    operand = op.operand
    coeff_size = 1
    if isa(operand, TensorField) && length(operand.components) > 0
        first_comp = operand.components[1,1]
        if first_comp !== nothing && hasfield(typeof(first_comp), :bases)
            for basis in first_comp.bases
                coeff_size *= basis.meta.size
            end
        end
    end

    eye = sparse(I, coeff_size, coeff_size)

    return kron(transpose_mat, eye)
end

check_conditions(op::TransposeComponents) = true
enforce_conditions(op::TransposeComponents) = nothing

function matrix_dependence(op::TransposeComponents, vars...)
    if hasmethod(matrix_dependence, Tuple{typeof(op.operand), typeof.(vars)...})
        return matrix_dependence(op.operand, vars...)
    end
    return falses(length(vars))
end

function matrix_coupling(op::TransposeComponents, vars...)
    if hasmethod(matrix_coupling, Tuple{typeof(op.operand), typeof.(vars)...})
        return matrix_coupling(op.operand, vars...)
    end
    return falses(length(vars))
end

is_linear(op::TransposeComponents) = true

# ============================================================================
# Outer Product Evaluation
# ============================================================================

"""
    evaluate_outer(outer_op::Outer, layout::Symbol=:g)

Evaluate outer product (tensor product) of two vector fields.

For vectors u and v, returns tensor T where T_ij = u_i * v_j.
"""
function evaluate_outer(outer_op::Outer, layout::Symbol=:g)
    left = outer_op.left
    right = outer_op.right

    if !isa(left, VectorField) || !isa(right, VectorField)
        throw(ArgumentError("Outer product requires two VectorFields"))
    end

    dim_left = length(left.components)
    dim_right = length(right.components)

    dist = left.dist
    coordsys = left.coordsys
    bases = left.bases
    dtype = promote_type(left.dtype, right.dtype)

    result = TensorField(dist, coordsys, "outer_$(left.name)_$(right.name)", bases, dtype)

    # Ensure proper dimensions
    if size(result.components) != (dim_left, dim_right)
        result.components = Matrix{ScalarField}(undef, dim_left, dim_right)
        for i in 1:dim_left
            for j in 1:dim_right
                result.components[i,j] = ScalarField(dist, "T_$(i)$(j)", bases, dtype)
            end
        end
    end

    # Compute T_ij = u_i * v_j
    # Outer product is nonlinear (pointwise multiply) — must be done in grid space.
    for i in 1:dim_left
        for j in 1:dim_right
            ensure_layout!(left.components[i], :g)
            ensure_layout!(right.components[j], :g)
            ensure_layout!(result.components[i,j], :g)
            get_grid_data(result.components[i,j]) .= get_grid_data(left.components[i]) .* get_grid_data(right.components[j])
        end
    end

    return result
end

# ============================================================================
# AdvectiveCFL Evaluation
# ============================================================================

"""
    evaluate_advective_cfl(cfl_op::AdvectiveCFL, layout::Symbol=:g)

Evaluate advective CFL grid-crossing frequency field.

Computes the local grid-crossing frequency:
    f = |u|/dx + |v|/dy + |w|/dz

This field can be used for adaptive timestepping: dt < 1/max(f).
"""
function evaluate_advective_cfl(cfl_op::AdvectiveCFL, layout::Symbol=:g)
    velocity = cfl_op.operand
    coords = cfl_op.coords

    if layout != :g
        @warn "AdvectiveCFL requires grid space; converting to :g"
        layout = :g
    end

    if !isa(velocity, VectorField)
        throw(ArgumentError("AdvectiveCFL requires a VectorField (velocity)"))
    end

    dim = length(velocity.components)
    dist = velocity.dist
    bases = velocity.bases
    dtype = velocity.dtype

    result = ScalarField(dist, "cfl_freq", bases, dtype)
    ensure_layout!(result, :g)

    get_grid_data(result) .= 0

    ndims_field = length(bases)
    for i in 1:dim
        ensure_layout!(velocity.components[i], :g)
        vel_data = get_grid_data(velocity.components[i])

        basis = bases[i]
        grid_spacing = compute_grid_spacing(basis, dist, i)

        if isa(grid_spacing, AbstractArray) && ndims_field > 1
            # Reshape spacing vector to broadcast along the correct axis.
            # A 1D vector broadcasts along axis 1 by default; we need axis i.
            shape = ones(Int, ndims_field)
            shape[i] = length(grid_spacing)
            spacing_shaped = reshape(grid_spacing, Tuple(shape))
            get_grid_data(result) .+= abs.(vel_data) ./ spacing_shaped
        else
            get_grid_data(result) .+= abs.(vel_data) ./ grid_spacing
        end
    end

    return result
end

"""
    compute_grid_spacing(basis::Basis, dist, axis::Int)

Compute local grid spacing for a basis.

For Fourier bases: uniform spacing dx = L/N
For Chebyshev bases: variable spacing based on Chebyshev nodes
"""
function compute_grid_spacing(basis::Basis, dist, axis::Int)
    if basis === nothing
        return Inf
    end

    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]

    # Get stretch factor from COV if available
    stretch = if hasfield(typeof(basis.meta), :COV) && basis.meta.COV !== nothing
        basis.meta.COV.stretch
    else
        L / 2.0
    end

    if isa(basis, FourierBasis)
        # Fourier grid spacing is simply L/N (uniform grid on [a, b))
        # Note: stretch factor L/2 is for Chebyshev's [-1,1] native domain,
        # not Fourier's [0, 2π) native domain.
        return L / N
    elseif isa(basis, ChebyshevT)
        i = collect(0:(N-1))
        theta = pi .* (i .+ 0.5) ./ N
        spacing = stretch .* sin.(theta) .* pi ./ N
        return spacing
    elseif isa(basis, ChebyshevU) || isa(basis, ChebyshevV)
        i = collect(0:(N-1))
        theta = pi .* (i .+ 0.5) ./ N
        spacing = stretch .* sin.(theta) .* pi ./ N
        return spacing
    elseif isa(basis, Legendre)
        return L / N
    elseif isa(basis, Jacobi)
        if isapprox(basis.a, -0.5) && isapprox(basis.b, -0.5)
            i = collect(0:(N-1))
            theta = pi .* (i .+ 0.5) ./ N
            spacing = stretch .* sin.(theta) .* pi ./ N
            return spacing
        else
            return L / N
        end
    else
        return L / N
    end
end
