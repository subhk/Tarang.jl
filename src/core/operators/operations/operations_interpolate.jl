# Interpolation evaluation plus Clenshaw-based polynomial reconstruction helpers.

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
    nb = length(field.bases)
    new_bases = ntuple(i -> field.bases[i < axis ? i : i + 1], nb - 1)
    if nb == 1
        return real(result_data isa AbstractArray ? sum(result_data) : result_data)
    end

    result_field = ScalarField(field.dist, "interp_$(field.name)", new_bases, field.dtype)
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

    # Allocate index buffer once; axis slot is a constant UnitRange, other slots are Int.
    full_idx = Vector{Union{UnitRange{Int}, Int}}(undef, nd)
    full_idx[axis] = 1:size(coeffs, axis)

    for idx in CartesianIndices(tuple(other_sizes...))
        other_pos = 1
        for d in 1:nd
            if d != axis
                full_idx[d] = idx[other_pos]
                other_pos += 1
            end
        end
        slice = vec(coeffs[full_idx...])
        result_data[idx] = clenshaw_fn(slice, x_native)
    end

    # Create reduced-dimension field
    nb = length(field.bases)
    new_bases = ntuple(i -> field.bases[i < axis ? i : i + 1], nb - 1)
    if nb == 1
        return result_data isa AbstractArray ? result_data[] : result_data
    end

    result_field = ScalarField(field.dist, "interp_$(field.name)", new_bases, field.dtype)
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
