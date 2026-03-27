"""
    Various operator operations

This file contains evaluation functions for:
- Interpolate, integrate, average, lift, convert
- GeneralFunction and UnaryGridFunction
- Grid and coeff conversion
- Component extraction (component, radial, angular, azimuthal)
"""

# ============================================================================
# Interpolation Evaluation
# ============================================================================

"""
    evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)

Evaluate interpolation operator at a specific position along a coordinate.
Following operators Interpolate implementation.

For Fourier bases: uses spectral interpolation (sum of modes)
For Jacobi bases: uses barycentric interpolation or Clenshaw algorithm
"""
function evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)
    operand = interp_op.operand
    coord = interp_op.coord
    position = interp_op.position

    if !isa(operand, ScalarField)
        throw(ArgumentError("Interpolate currently only supports scalar fields"))
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
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = operand.bases[basis_index]

    # Work in coefficient space for spectral interpolation
    ensure_layout!(operand, :c)

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        return interpolate_fourier(operand, basis, basis_index, position, layout)
    elseif isa(basis, JacobiBasis)
        return interpolate_jacobi(operand, basis, basis_index, position, layout)
    else
        throw(ArgumentError("Interpolation not implemented for basis type $(typeof(basis))"))
    end
end

"""
    interpolate_fourier(field, basis, axis, position, layout)

Interpolate Fourier field at a specific position using spectral reconstruction.
f(x) = Sigma c_k exp(i k x) for ComplexFourier
f(x) = a_0 + Sigma (a_k cos(kx) + b_k sin(kx)) for RealFourier
"""
function interpolate_fourier(field::ScalarField, basis::FourierBasis, axis::Int, position::Real, layout::Symbol)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    x0 = basis.meta.bounds[1]

    # Normalize position to [0, L)
    x = mod(position - x0, L)

    # Get coefficient data (copy to CPU if on GPU - interpolation uses scalar indexing)
    if is_gpu_array(get_coeff_data(field))
        coeffs = Array(get_coeff_data(field))
    else
        coeffs = get_coeff_data(field)
    end

    # For 1D fields, evaluate directly
    if ndims(coeffs) == 1
        return _interpolate_fourier_1d(coeffs, basis, N, L, x)
    end

    # For multi-D fields, interpolate along the specified axis using spectral weights
    interp_weights = _fourier_interp_weights(basis, N, L, x)

    # Weighted sum along axis: result[...] = Σ_i weights[i] * coeffs[..., i, ...]
    shape = ones(Int, ndims(coeffs))
    shape[axis] = N
    w_shaped = reshape(interp_weights, shape...)
    result_data = dropdims(sum(coeffs .* w_shaped, dims=axis), dims=axis)

    # Create reduced-dimension field
    new_bases = [b for (i, b) in enumerate(field.bases) if i != axis]
    if isempty(new_bases)
        return real(result_data isa AbstractArray ? sum(result_data) : result_data)
    end

    result_field = ScalarField(field.dist, "interp_$(field.name)", tuple(new_bases...), field.dtype)
    ensure_layout!(result_field, :c)
    set_coeff_data!(result_field, real.(result_data))
    result_field.current_layout = :c

    if layout == :g
        backward_transform!(result_field)
    end

    return result_field
end

"""
    _interpolate_fourier_1d(coeffs, basis, N, L, x)

Evaluate 1D Fourier spectral reconstruction at position x.
"""
function _interpolate_fourier_1d(coeffs::AbstractVector, basis::FourierBasis, N::Int, L::Real, x::Real)
    if isa(basis, RealFourier)
        # RealFourier uses msin convention: [a_0, a_1, b_1, a_2, b_2, ..., a_nyq]
        # where b_k is the coefficient of -sin(kx), not +sin(kx)
        result = coeffs[1]  # DC component

        k_max = N ÷ 2
        is_even = (N % 2 == 0)

        for k in 1:(k_max - (is_even ? 1 : 0))
            k_phys = 2π * k / L
            cos_idx = 2*k
            sin_idx = 2*k + 1

            if cos_idx <= length(coeffs) && sin_idx <= length(coeffs)
                result += coeffs[cos_idx] * cos(k_phys * x)
                result += coeffs[sin_idx] * (-sin(k_phys * x))
            end
        end

        # Nyquist component for even N
        if is_even && N <= length(coeffs)
            k_nyq = 2π * k_max / L
            result += coeffs[N] * cos(k_nyq * x)
        end

        return result
    else  # ComplexFourier
        result = complex(0.0, 0.0)

        for i in 1:N
            if i <= N÷2 + 1
                k = i - 1
            else
                k = i - N - 1
            end
            k_phys = 2π * k / L
            result += coeffs[i] * exp(im * k_phys * x)
        end

        return real(result)
    end
end

"""
    _fourier_interp_weights(basis, N, L, x)

Build spectral interpolation weight vector for Fourier basis at position x.
"""
function _fourier_interp_weights(basis::FourierBasis, N::Int, L::Real, x::Real)
    if isa(basis, RealFourier)
        weights = zeros(N)
        weights[1] = 1.0  # DC
        k_max = N ÷ 2
        is_even = (N % 2 == 0)
        for k in 1:(k_max - (is_even ? 1 : 0))
            k_phys = 2π * k / L
            if 2*k <= N
                weights[2*k] = cos(k_phys * x)
            end
            if 2*k + 1 <= N
                weights[2*k + 1] = -sin(k_phys * x)  # msin convention
            end
        end
        if is_even && N >= 2
            weights[N] = cos(2π * k_max / L * x)
        end
        return weights
    else  # ComplexFourier
        weights = zeros(ComplexF64, N)
        for i in 1:N
            k = i <= N÷2 + 1 ? i - 1 : i - N - 1
            weights[i] = exp(im * 2π * k / L * x)
        end
        return weights
    end
end

"""
    interpolate_jacobi(field, basis, axis, position, layout)

Interpolate Jacobi-type field (Chebyshev, Legendre) using Clenshaw algorithm.
"""
function interpolate_jacobi(field::ScalarField, basis::JacobiBasis, axis::Int, position::Real, layout::Symbol)
    N = basis.meta.size
    a, b = basis.meta.bounds

    # Map position to native [-1, 1] interval
    x_native = 2.0 * (position - a) / (b - a) - 1.0

    # Clamp to valid range
    x_native = clamp(x_native, -1.0, 1.0)

    # Get coefficient data (copy to CPU if on GPU - Clenshaw uses scalar indexing)
    if is_gpu_array(get_coeff_data(field))
        coeffs = Array(get_coeff_data(field))
    else
        coeffs = get_coeff_data(field)
    end

    # Select the appropriate Clenshaw function
    clenshaw_fn = if isa(basis, ChebyshevT)
        clenshaw_chebyshev_t
    elseif isa(basis, ChebyshevU)
        clenshaw_chebyshev_u
    elseif isa(basis, Legendre)
        clenshaw_legendre
    else
        (c, x) -> clenshaw_jacobi(c, x, basis.a, basis.b)
    end

    # For 1D fields, evaluate directly
    if ndims(coeffs) == 1
        return clenshaw_fn(coeffs, x_native)
    end

    # For multi-D fields, apply Clenshaw along each slice of the specified axis
    nd = ndims(coeffs)
    result_shape = ntuple(d -> d == axis ? 1 : size(coeffs, d), nd)
    result_size = ntuple(d -> size(coeffs, d), nd)

    # Build result by iterating over all non-axis indices
    other_dims = [d for d in 1:nd if d != axis]
    other_sizes = [size(coeffs, d) for d in other_dims]

    if isempty(other_sizes)
        # All dimensions are this axis — 1D case handled above
        return clenshaw_fn(vec(coeffs), x_native)
    end

    result_data = zeros(eltype(coeffs), other_sizes...)

    for idx in CartesianIndices(tuple(other_sizes...))
        # Build slice ranges for the full array
        full_idx = Vector{Any}(undef, nd)
        other_pos = 1
        for d in 1:nd
            if d == axis
                full_idx[d] = 1:size(coeffs, d)
            else
                full_idx[d] = idx[other_pos]
                other_pos += 1
            end
        end
        slice = vec(coeffs[full_idx...])
        result_data[idx] = clenshaw_fn(slice, x_native)
    end

    # Create reduced-dimension field
    new_bases = [bs for (i, bs) in enumerate(field.bases) if i != axis]
    if isempty(new_bases)
        return result_data isa AbstractArray ? result_data[] : result_data
    end

    result_field = ScalarField(field.dist, "interp_$(field.name)", tuple(new_bases...), field.dtype)
    ensure_layout!(result_field, :c)
    set_coeff_data!(result_field, result_data)
    result_field.current_layout = :c

    if layout == :g
        backward_transform!(result_field)
    end

    return result_field
end

# ============================================================================
# Clenshaw Algorithm Implementations
# ============================================================================

"""
    clenshaw_chebyshev_t(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev T polynomial sum.
T_n(x) satisfies: T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
"""
function clenshaw_chebyshev_t(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence: b_k = c_k + 2x b_{k+1} - b_{k+2}
    b_k2 = 0.0  # b_{n+1}
    b_k1 = 0.0  # b_{n}

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step: f(x) = c_0 + x*b_1 - b_2
    return coeffs[1] + x * b_k1 - b_k2
end

"""
    clenshaw_chebyshev_u(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev U polynomial sum.
U_n(x) satisfies: U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
"""
function clenshaw_chebyshev_u(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Same recurrence as T_n
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # For U_n: f(x) = c_0 * U_0(x) + b_1 * U_1(x) - b_2 * U_0(x)
    # where U_0(x) = 1, U_1(x) = 2x
    return coeffs[1] + 2x * b_k1 - b_k2
end

"""
    clenshaw_legendre(coeffs, x)

Clenshaw algorithm for evaluating Legendre polynomial sum.
P_n(x) satisfies: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
"""
function clenshaw_legendre(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence for Legendre
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        # Clenshaw backward recurrence: b_k = c_k + α_k * b_{k+1} - C_{k+1} * b_{k+2}
        # where α_k = (2k+1)/(k+1) * x and C_k = k/(k+1) for Legendre
        deg = k - 1  # 0-indexed polynomial degree
        alpha = (2*deg + 1) / (deg + 1) * x
        beta = (deg + 1) / (deg + 2)  # C_{deg+1}, NOT C_{deg}
        b_k = coeffs[k] + alpha * b_k1 - beta * b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step
    return coeffs[1] + x * b_k1 - 0.5 * b_k2
end

"""
    clenshaw_jacobi(coeffs, x, a, b)

Clenshaw algorithm for evaluating general Jacobi polynomial sum.
"""
function clenshaw_jacobi(coeffs::AbstractVector, x::Real, a::Float64, b::Float64)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Use direct evaluation for general Jacobi (less efficient but correct)
    result = 0.0
    for k in 1:n
        result += coeffs[k] * jacobi_polynomial(k-1, a, b, x)
    end
    return result
end

"""
    jacobi_polynomial(n, a, b, x)

Evaluate Jacobi polynomial P_n^{(a,b)}(x) using three-term recurrence.
"""
function jacobi_polynomial(n::Int, a::Float64, b::Float64, x::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return 0.5 * (a - b + (a + b + 2) * x)
    end

    p_km2 = 1.0
    p_km1 = 0.5 * (a - b + (a + b + 2) * x)

    for k in 2:n
        # Three-term recurrence for Jacobi polynomials
        k_f = Float64(k)
        a1 = 2 * k_f * (k_f + a + b) * (2*k_f + a + b - 2)
        a2 = (2*k_f + a + b - 1) * (a^2 - b^2)
        a3 = (2*k_f + a + b - 2) * (2*k_f + a + b - 1) * (2*k_f + a + b)
        a4 = 2 * (k_f + a - 1) * (k_f + b - 1) * (2*k_f + a + b)

        p_k = ((a2 + a3 * x) * p_km1 - a4 * p_km2) / a1
        p_km2 = p_km1
        p_km1 = p_k
    end

    return p_km1
end

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

# ============================================================================
# Lift Evaluation
# ============================================================================

"""
    evaluate_lift(lift_op::Lift, layout::Symbol=:g)

Evaluate lifting operator for tau method boundary conditions.
Following the Dedalus LiftJacobi implementation (basis.py:790-814).

The Lift operator creates a polynomial field P on the output basis with coefficient
at mode n set to 1, then returns P * operand. This "lifts" the operand (typically
a tau variable) into spectral space at the specified mode.

Convention (following Dedalus):
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
"""
function evaluate_lift(lift_op::Lift, layout::Symbol=:g)
    operand = lift_op.operand
    output_basis = lift_op.basis  # The output basis
    n = lift_op.n

    if !isa(operand, ScalarField)
        throw(ArgumentError("Lift currently only supports scalar fields"))
    end

    # Get basis size
    N = output_basis.meta.size

    # Handle negative index wrap-around (Dedalus convention)
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode
    end
    lift_mode += 1  # Convert from 0-indexed to 1-indexed Julia convention

    # Validate mode index
    if lift_mode < 1 || lift_mode > N
        throw(ArgumentError("Lift mode index $n (resolved to $lift_mode) out of bounds for basis size $N"))
    end

    # Find or create the output bases tuple
    output_bases = _get_lift_output_bases(operand, output_basis)

    # Step 1: Build polynomial P with coefficient 1 at mode n
    P = ScalarField(operand.dist, "lift_poly", output_bases, operand.dtype)
    ensure_layout!(P, :c)

    # Find which axis corresponds to the output basis
    basis_axis = _find_basis_axis(output_bases, output_basis)

    # Build P coefficients on CPU, then transfer to GPU if needed
    p_data = get_coeff_data(P)
    arch = operand.dist.architecture
    if is_gpu_array(p_data)
        # Build on CPU first, then copy to GPU
        cpu_p = zeros(eltype(p_data), size(p_data))
        if ndims(cpu_p) == 1
            cpu_p[lift_mode] = one(eltype(cpu_p))
        else
            selectdim(cpu_p, basis_axis, lift_mode) .= one(eltype(cpu_p))
        end
        copyto!(p_data, on_architecture(arch, cpu_p))
    else
        fill!(p_data, zero(eltype(p_data)))
        if ndims(p_data) == 1
            p_data[lift_mode] = one(eltype(p_data))
        else
            selectdim(p_data, basis_axis, lift_mode) .= one(eltype(p_data))
        end
    end

    # Step 2: Compute result = P * operand
    ensure_layout!(operand, :c)

    # Create result field
    result = ScalarField(operand.dist, "lift_$(operand.name)", output_bases, operand.dtype)
    ensure_layout!(result, :c)

    # Multiply P * operand: place operand's data at mode lift_mode
    _multiply_lift_polynomial!(get_coeff_data(result), get_coeff_data(P),
                               get_coeff_data(operand), basis_axis, lift_mode, arch)

    if layout == :g
        backward_transform!(result)
    end

    return result
end

"""
    _get_lift_output_bases(operand, output_basis)

Get output bases for lift operation, substituting input basis with output basis.
"""
function _get_lift_output_bases(operand::ScalarField, output_basis::Basis)
    output_coord = output_basis.meta.element_label

    new_bases = Vector{Any}(undef, length(operand.bases))
    found = false

    for (i, b) in enumerate(operand.bases)
        if b === nothing
            new_bases[i] = nothing
        elseif b.meta.element_label == output_coord
            new_bases[i] = output_basis
            found = true
        else
            new_bases[i] = b
        end
    end

    # If no matching basis found, this is a lift from no-basis to output_basis
    if !found
        push!(new_bases, output_basis)
    end

    return tuple(new_bases...)
end

"""
    _find_basis_axis(bases, target_basis)

Find which axis (1-indexed) corresponds to the target basis.
"""
function _find_basis_axis(bases::Tuple, target_basis::Basis)
    for (i, b) in enumerate(bases)
        if b === target_basis ||
           (b !== nothing && b.meta.element_label == target_basis.meta.element_label)
            return i
        end
    end
    return 1  # Default to first axis
end

"""
    _set_lift_coefficient!(data, axis, mode, value)

Set coefficient at specified mode along axis to given value.
"""
function _set_lift_coefficient!(data::AbstractArray, axis::Int, mode::Int, value::Real)
    view = selectdim(data, axis, mode)
    fill!(view, value)
end

"""
    _multiply_lift_polynomial!(result, P_data, operand_data, basis_axis, lift_mode, arch)

Multiply lift polynomial P by operand.
P has a single non-zero coefficient at lift_mode.
Result = P * operand places operand's values at mode lift_mode.

GPU-compatible: avoids scalar indexing by building on CPU and copying,
or using broadcasting operations that work on GPU arrays.
"""
function _multiply_lift_polynomial!(result::AbstractArray, P_data::AbstractArray,
                                    operand_data::AbstractArray, basis_axis::Int,
                                    lift_mode::Int, arch=nothing)
    if is_gpu_array(result)
        # GPU path: build result on CPU, then copy to GPU
        cpu_result = zeros(eltype(result), size(result))
        cpu_operand = is_gpu_array(operand_data) ? Array(operand_data) : operand_data

        if ndims(cpu_result) == 1
            if length(cpu_operand) >= 1
                cpu_result[lift_mode] = cpu_operand[1]
            end
        else
            result_slice = selectdim(cpu_result, basis_axis, lift_mode)
            if ndims(cpu_operand) == ndims(cpu_result)
                operand_slice = selectdim(cpu_operand, basis_axis, 1)
                result_slice .= operand_slice
            elseif ndims(cpu_operand) < ndims(cpu_result)
                result_slice .= cpu_operand
            else
                result_slice .= selectdim(cpu_operand, basis_axis, 1)
            end
        end

        # Transfer to GPU
        if arch !== nothing
            copyto!(result, on_architecture(arch, cpu_result))
        else
            copyto!(result, cpu_result)
        end
    else
        # CPU path: direct operations
        fill!(result, zero(eltype(result)))

        if ndims(result) == 1
            if length(operand_data) >= 1
                result[lift_mode] = operand_data[1]
            end
        else
            result_slice = selectdim(result, basis_axis, lift_mode)
            if ndims(operand_data) == ndims(result)
                operand_slice = selectdim(operand_data, basis_axis, 1)
                result_slice .= operand_slice
            elseif ndims(operand_data) < ndims(result)
                result_slice .= operand_data
            else
                result_slice .= selectdim(operand_data, basis_axis, 1)
            end
        end
    end
end

"""
    apply_lift_nd!(result, operand, axis, lift_mode)

Apply lift operation along specified axis for multi-dimensional arrays.
(Legacy helper - kept for compatibility)
"""
function apply_lift_nd!(result::AbstractArray, operand::AbstractArray, axis::Int, lift_mode::Int)
    selectdim(result, axis, lift_mode) .= selectdim(operand, axis, 1)
end

# ============================================================================
# Convert Evaluation
# ============================================================================

"""
    evaluate_convert(conv_op::Convert, layout::Symbol=:g)

Evaluate basis conversion operator.
Following operators Convert implementation.

Converts field from one basis representation to another using
spectral conversion matrices.
"""
function evaluate_convert(conv_op::Convert, layout::Symbol=:g)
    operand = conv_op.operand
    out_basis = conv_op.basis

    if !isa(operand, ScalarField)
        throw(ArgumentError("Convert currently only supports scalar fields"))
    end

    # Find the input basis to convert
    in_basis_index = nothing
    in_basis = nothing

    for (i, b) in enumerate(operand.bases)
        if b !== nothing && isa(b, JacobiBasis) && isa(out_basis, JacobiBasis)
            # Check if bases are on same coordinate
            if b.meta.element_label == out_basis.meta.element_label
                in_basis_index = i
                in_basis = b
                break
            end
        end
    end

    if in_basis === nothing
        # No conversion needed or not applicable
        return copy(operand)
    end

    # Work in coefficient space
    ensure_layout!(operand, :c)

    # Build or retrieve conversion matrix
    conv_mat = conversion_matrix(in_basis, out_basis)

    # Create result field
    new_bases = collect(operand.bases)
    new_bases[in_basis_index] = out_basis
    result = ScalarField(operand.dist, "conv_$(operand.name)", tuple(new_bases...), operand.dtype)
    ensure_layout!(result, :c)

    # Apply conversion matrix
    if ndims(get_coeff_data(operand)) == 1
        get_coeff_data(result) .= conv_mat * get_coeff_data(operand)
    else
        get_coeff_data(result) .= apply_matrix_along_axis(conv_mat, get_coeff_data(operand), in_basis_index)
    end

    if layout == :g
        backward_transform!(result)
    end

    return result
end

# ============================================================================
# General Function Evaluation
# ============================================================================

"""
    evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)

Evaluate general function operator in grid space.
"""
function evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)
    operand = gf_op.operand
    f = gf_op.func
    name = gf_op.name

    if !isa(operand, ScalarField)
        throw(ArgumentError("GeneralFunction currently only supports scalar fields"))
    end

    # Work in grid space
    ensure_layout!(operand, :g)

    # Create result field
    result = ScalarField(operand.dist, "$(name)_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, :g)

    # Apply function element-wise
    get_grid_data(result) .= f.(get_grid_data(operand))

    if layout == :c
        forward_transform!(result)
    end

    return result
end

"""
    evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)

Evaluate unary grid function operator.
"""
function evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)
    return evaluate_general_function(
        GeneralFunction(ugf_op.operand, ugf_op.func, ugf_op.name),
        layout
    )
end

# ============================================================================
# Grid and Coeff Conversion Evaluation
# ============================================================================

"""
    evaluate_grid(grid_op::Grid)

Convert operand to grid space.
"""
function evaluate_grid(grid_op::Grid)
    operand = grid_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :g)
        return operand
    else
        throw(ArgumentError("Grid conversion not implemented for $(typeof(operand))"))
    end
end

"""
    evaluate_coeff(coeff_op::Coeff)

Convert operand to coefficient space.
"""
function evaluate_coeff(coeff_op::Coeff)
    operand = coeff_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
        return operand
    else
        throw(ArgumentError("Coeff conversion not implemented for $(typeof(operand))"))
    end
end

# ============================================================================
# Component Extraction Evaluation
# ============================================================================

"""
    evaluate_component(comp_op::Component)

Extract specific component from vector/tensor field.
"""
function evaluate_component(comp_op::Component)
    operand = comp_op.operand
    index = comp_op.index

    if isa(operand, VectorField)
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    elseif isa(operand, TensorField)
        # For tensors, index could be linear or we need (i,j)
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    else
        throw(ArgumentError("Component extraction requires VectorField or TensorField"))
    end
end

"""
    evaluate_radial_component(rc_op::RadialComponent)

Extract radial component from vector field.
For Cartesian coordinates, this is the x-component.
"""
function evaluate_radial_component(rc_op::RadialComponent)
    operand = rc_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end

    # In Cartesian, "radial" is typically the first component
    return operand.components[1]
end

"""
    evaluate_angular_component(ac_op::AngularComponent)

Extract angular component from vector field.
For Cartesian 2D, this is the y-component.
"""
function evaluate_angular_component(ac_op::AngularComponent)
    operand = ac_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end

    if length(operand.components) < 2
        throw(ArgumentError("VectorField must have at least 2 components"))
    end

    return operand.components[2]
end

"""
    evaluate_azimuthal_component(az_op::AzimuthalComponent)

Extract azimuthal component from vector field.
For Cartesian 3D, this is the z-component.
"""
function evaluate_azimuthal_component(az_op::AzimuthalComponent)
    operand = az_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end

    if length(operand.components) < 3
        throw(ArgumentError("VectorField must have at least 3 components"))
    end

    return operand.components[3]
end

# ============================================================================
# Copy Evaluation
# ============================================================================

"""
    evaluate_copy(op::Copy, layout::Symbol=:g)

Evaluate copy operator: produces an independent deep copy of the operand.

GPU-compatible: Uses ScalarField's custom deepcopy which properly handles
CuArray data (deepcopy creates an independent copy on the same device).
"""
function evaluate_copy(op::Copy, layout::Symbol=:g)
    operand = op.operand
    if isa(operand, Operator)
        operand = evaluate(operand, layout)
    end

    result = deepcopy(operand)
    if isa(result, ScalarField)
        ensure_layout!(result, layout)
    end
    return result
end

# ============================================================================
# Hilbert Transform Evaluation
# ============================================================================

"""
    evaluate_hilbert_transform(op::HilbertTransform, layout::Symbol=:g)

Evaluate Hilbert transform in spectral space.

For ComplexFourier: multiply mode k by -i*sign(k), k=0 → 0.
For RealFourier (interleaved [a0, a1, b1, a2, b2, ...]): swap cos↔sin
with sign change: H[cos(nx)] = sin(nx), H[sin(nx)] = -cos(nx).

GPU-compatible: Coefficient manipulation uses scalar indexing, so GPU arrays
are transferred to CPU, transformed, and copied back (same pattern as
interpolation and lift operations).
"""
function evaluate_hilbert_transform(op::HilbertTransform, layout::Symbol=:g)
    operand = op.operand
    if isa(operand, Operator)
        operand = evaluate(operand, :c)
    end

    if !isa(operand, ScalarField)
        throw(ArgumentError("HilbertTransform currently only supports scalar fields"))
    end

    result = deepcopy(operand)
    ensure_layout!(result, :c)
    coeff = get_coeff_data(result)

    # GPU path: transfer to CPU, apply, copy back (scalar indexing required)
    if is_gpu_array(coeff)
        cpu_coeff = Array(coeff)
        _apply_hilbert_spectral!(cpu_coeff, result.bases)
        arch = result.dist.architecture
        copyto!(coeff, on_architecture(arch, cpu_coeff))
    else
        _apply_hilbert_spectral!(coeff, result.bases)
    end

    if layout == :g
        backward_transform!(result)
    end
    return result
end

"""
    _apply_hilbert_spectral!(coeff, bases)

Apply -i*sign(k) multiplier in spectral space for each Fourier basis.
Non-Fourier bases are left unchanged (Hilbert transform only acts on
periodic dimensions).

Note: This function assumes `coeff` is a CPU array. GPU arrays must be
transferred to CPU before calling this (handled by evaluate_hilbert_transform).
"""
function _apply_hilbert_spectral!(coeff::AbstractArray, bases::Tuple)
    # For 1D fields, apply directly
    if ndims(coeff) == 1 && length(bases) >= 1
        basis = bases[1]
        _apply_hilbert_1d!(coeff, basis)
        return
    end

    # For multi-D, apply along each Fourier axis
    for (axis, basis) in enumerate(bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            _apply_hilbert_along_axis!(coeff, basis, axis)
        end
    end
end

"""
Apply Hilbert transform to 1D coefficient array for a single basis.
Operates on CPU arrays only (scalar indexing).
"""
function _apply_hilbert_1d!(coeff::AbstractVector, basis::Basis)
    if isa(basis, ComplexFourier)
        N = length(coeff)
        for i in 1:N
            if i <= N ÷ 2 + 1
                k = i - 1           # DC and positive frequencies (including Nyquist)
            else
                k = i - N - 1       # negative frequencies
            end
            if k == 0
                coeff[i] = zero(eltype(coeff))
            else
                # Multiply by -i*sign(k)
                coeff[i] = -im * sign(k) * coeff[i]
            end
        end
    elseif isa(basis, RealFourier)
        # RealFourier coefficients from rfft: complex vector [c0, c1, ..., c_{N/2}]
        # where c_k is the complex coefficient for wavenumber k (all k >= 0).
        # Hilbert transform: multiply by -i*sign(k).
        # Since rfft only stores k >= 0, sign(k) = +1 for k > 0.
        N = length(coeff)
        coeff[1] = zero(eltype(coeff))  # DC (k=0) → 0
        for i in 2:N
            coeff[i] *= -im  # -i*sign(k) = -i for k > 0
        end
        # Nyquist mode (k=N/2): Hilbert of cos(N/2*x) is sin(N/2*x)
        # but sin at Nyquist is unrepresentable in rfft → zero it out
        grid_size = basis.meta.size
        if grid_size % 2 == 0
            coeff[N] = zero(eltype(coeff))
        end
    end
    # Non-Fourier bases: no-op
end

"""
Apply Hilbert transform along a specific axis of a multi-dimensional array.
Operates on CPU arrays only (scalar indexing via CartesianIndices and view).
"""
function _apply_hilbert_along_axis!(coeff::AbstractArray, basis::Basis, axis::Int)
    for idx in CartesianIndices(ntuple(d -> d == axis ? (1:1) : (1:size(coeff, d)), ndims(coeff)))
        ranges = ntuple(d -> d == axis ? (1:size(coeff, d)) : (idx[d]:idx[d]), ndims(coeff))
        slice = view(coeff, ranges...)
        _apply_hilbert_1d!(vec(slice), basis)
    end
end
