# Integration and averaging evaluation along spectral coordinates.

# ============================================================================
# Integration Evaluation
# ============================================================================

"""
    evaluate_integrate(int_op::Integrate, layout::Symbol=:g)

Evaluate integration operator over specified coordinate(s).
Following operators Integrate implementation.

Uses appropriate quadrature weights for each basis type:
- Fourier: trapezoidal rule (uniform weights)
- Chebyshev: Clenshaw-Curtis quadrature
- Legendre: Gauss-Legendre quadrature
"""
function evaluate_integrate(int_op::Integrate, layout::Symbol=:g)
    operand = int_op.operand
    coord = int_op.coord

    if !isa(operand, ScalarField)
        throw(ArgumentError("Integrate currently only supports scalar fields"))
    end

    # Handle single coordinate or tuple of coordinates
    coords = isa(coord, Coordinate) ? (coord,) : coord

    # Start with the operand
    result_field = copy(operand)

    # Integrate over each coordinate in sequence
    for c in coords
        result_field = integrate_along_coord(result_field, c)
    end

    # If all dimensions integrated, return scalar
    if length(coords) == length(operand.bases)
        # integrate_along_coord may already return a scalar for 1D fields
        if result_field isa Number
            return result_field
        end
        ensure_layout!(result_field, :g)
        return sum(get_grid_data(result_field))
    end

    if layout == :g
        ensure_layout!(result_field, :g)
    else
        ensure_layout!(result_field, :c)
    end

    return result_field
end

"""
    integrate_along_coord(field, coord)

Integrate field along a single coordinate using appropriate quadrature.
"""
function integrate_along_coord(field::ScalarField, coord::Coordinate)
    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(field.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = field.bases[basis_index]

    # Work in grid space for integration
    ensure_layout!(field, :g)

    # Get quadrature weights for this basis
    weights = get_integration_weights(basis)

    # Apply weighted sum along the axis
    data = get_grid_data(field)

    # Sum along the specified axis with weights
    result_data = integrate_weighted_sum(data, weights, basis_index)

    # Create result field with reduced dimensionality
    if ndims(data) == 1
        return sum(data .* weights)
    else
        # Create new field without the integrated dimension
        new_bases = [b for (i, b) in enumerate(field.bases) if i != basis_index]
        if isempty(new_bases)
            return sum(data .* reshape(weights, size_for_axis(length(weights), basis_index, ndims(data))))
        end

        result = ScalarField(field.dist, "int_$(field.name)", tuple(new_bases...), field.dtype)
        set_grid_data!(result, result_data)
        result.current_layout = :g
        return result
    end
end

"""
    get_integration_weights(basis)

Get quadrature weights for integration over a basis.
"""
function get_integration_weights(basis::Basis)
    N = basis.meta.size
    a, b = basis.meta.bounds
    L = b - a

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        # Uniform weights for periodic Fourier
        return fill(L / N, N)

    elseif isa(basis, ChebyshevT)
        # Clenshaw-Curtis quadrature weights
        return clenshaw_curtis_weights(N, L)

    elseif isa(basis, Legendre)
        # Gauss-Legendre quadrature weights
        _, weights = gauss_legendre_quadrature(N)
        return weights .* (L / 2)  # Scale from [-1,1] to [a,b]

    else
        # Default: uniform weights
        return fill(L / N, N)
    end
end

"""
    clenshaw_curtis_weights(N, L)

Compute Clenshaw-Curtis quadrature weights for Chebyshev integration.
"""
function clenshaw_curtis_weights(N::Int, L::Float64)
    weights = zeros(N)

    if N == 1
        return [L]
    end

    # Clenshaw-Curtis weights on [-1, 1] via DCT-I approach
    # Sum goes to floor((N-1)/2) — the max harmonic of the DCT-I on N points
    M = (N - 1) ÷ 2  # maximum harmonic index
    for j in 1:N
        theta_j = pi * (j - 1) / (N - 1)
        w = 0.0
        for k in 0:M
            # DCT-I boundary factors: half-weight at k=0 and k=M (when N-1 is even)
            if k == 0 || (k == M && iseven(N - 1))
                factor = 1.0
            else
                factor = 2.0
            end
            w += factor * cos(2 * k * theta_j) / (1 - 4 * k^2)
        end
        weights[j] = 2 * w / (N - 1)
    end

    # Endpoint halving for Gauss-Lobatto grid (endpoints count half)
    weights[1] *= 0.5
    weights[N] *= 0.5

    # Scale to interval [a, b]
    return weights .* (L / 2)
end

"""
    gauss_legendre_quadrature(N)

Compute Gauss-Legendre quadrature points and weights on [-1, 1].
"""
function gauss_legendre_quadrature(N::Int)
    points = zeros(N)
    weights = zeros(N)

    # Initial guesses for roots using Chebyshev nodes
    for i in 1:N
        points[i] = -cos(pi * (i - 0.25) / (N + 0.5))
    end

    # Newton-Raphson iteration for roots
    for i in 1:N
        x = points[i]
        for _ in 1:100  # Max iterations
            # Evaluate P_N and P_{N-1} using recurrence
            p_km2 = 1.0
            p_km1 = x

            for k in 2:N
                p_k = ((2*k - 1) * x * p_km1 - (k - 1) * p_km2) / k
                p_km2 = p_km1
                p_km1 = p_k
            end

            # Derivative: P'_N = N(x P_N - P_{N-1}) / (x^2 - 1)
            dp = N * (x * p_km1 - p_km2) / (x^2 - 1)

            # Newton update
            dx = -p_km1 / dp
            x += dx

            if abs(dx) < 1e-14
                break
            end
        end

        points[i] = x

        # Compute weight
        p_km2 = 1.0
        p_km1 = x
        for k in 2:N
            p_k = ((2*k - 1) * x * p_km1 - (k - 1) * p_km2) / k
            p_km2 = p_km1
            p_km1 = p_k
        end
        dp = N * (x * p_km1 - p_km2) / (x^2 - 1)
        weights[i] = 2 / ((1 - x^2) * dp^2)
    end

    return points, weights
end

"""
    integrate_weighted_sum(data, weights, axis)

Apply weighted sum along specified axis.
"""
function integrate_weighted_sum(data::AbstractArray, weights::AbstractVector, axis::Int)
    nd = ndims(data)

    # Move weights to same device as data if needed
    if is_gpu_array(data)
        weights_device = copy_to_device(weights, data)
    else
        weights_device = weights
    end

    if nd == 1
        return sum(data .* weights_device)
    end

    # Reshape weights to broadcast along the correct axis
    shape = ones(Int, nd)
    shape[axis] = length(weights_device)
    w_shaped = reshape(weights_device, shape...)

    # Weighted sum along axis
    return dropdims(sum(data .* w_shaped, dims=axis), dims=axis)
end

"""
    size_for_axis(n, axis, ndims)

Create a size tuple with n in the specified axis position and 1 elsewhere.
"""
function size_for_axis(n::Int, axis::Int, nd::Int)
    shape = ones(Int, nd)
    shape[axis] = n
    return tuple(shape...)
end

# ============================================================================
# Average Evaluation
# ============================================================================

"""
    evaluate_average(avg_op::Average, layout::Symbol=:g)

Evaluate averaging operator along a coordinate.
Average = Integrate / (interval length)
"""
function evaluate_average(avg_op::Average, layout::Symbol=:g)
    operand = avg_op.operand
    coord = avg_op.coord

    if !isa(operand, ScalarField)
        throw(ArgumentError("Average currently only supports scalar fields"))
    end

    # Find the basis for this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = operand.bases[basis_index]
    L = basis.meta.bounds[2] - basis.meta.bounds[1]

    # Integrate and divide by interval length
    int_result = integrate_along_coord(operand, coord)

    if isa(int_result, Real)
        return int_result / L
    else
        # Scale only the active layout's data to avoid corrupting stale buffers
        if int_result.current_layout == :g
            data = get_grid_data(int_result)
            if data !== nothing
                data ./= L
            end
        else
            data = get_coeff_data(int_result)
            if data !== nothing
                data ./= L
            end
        end
        return int_result
    end
end
