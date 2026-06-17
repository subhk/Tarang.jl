"""
    Tensor miscellany

This file contains outer-product evaluation and AdvectiveCFL utilities.
"""

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

    # Outer product is computed in grid space (nonlinear); honor the requested layout
    # so `evaluate(outer, :c)` returns coefficients rather than a :g-flagged tensor.
    return _ensure_result_layout!(result, layout)
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
