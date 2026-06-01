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

    # Accept composite expressions: reduce the operand tree to a scalar field first.
    if !isa(operand, ScalarField)
        operand = evaluate(operand)
    end
    if !isa(operand, ScalarField)
        throw(ArgumentError("interpolate: operand must reduce to a scalar field, got $(typeof(operand))"))
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
    # The weight vector length matches the coefficient extent along this axis
    # (a half-spectrum N÷2+1 for the rfft'd RealFourier axis, N otherwise).
    shape = ones(Int, ndims(coeffs))
    shape[axis] = length(interp_weights)
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
    k0 = 2π / L
    if isa(basis, RealFourier)
        # RealFourier coefficients are stored as an UNNORMALIZED complex
        # half-spectrum (rfft): coeffs[i] holds mode k = i-1 for i = 1..N÷2+1.
        # Reconstruction: s(x) = (1/N) [ Re(c_0)
        #                                + Σ_{k=1}^{K} w_k Re(c_k e^{i k k0 x}) ]
        # with weight w_k = 2 for interior modes and w_k = 1 for the (even-N)
        # Nyquist mode. The DC term carries weight 1.
        half = length(coeffs)
        result = real(coeffs[1])  # DC component
        nyquist = iseven(N)
        for i in 2:half
            k = i - 1
            factor = (nyquist && i == half) ? 1.0 : 2.0
            result += factor * real(coeffs[i] * cis(k0 * k * x))
        end
        return result / N
    else  # ComplexFourier: full unnormalized spectrum, FFT-ordered wavenumbers
        result = complex(0.0, 0.0)
        for i in 1:N
            k = i <= N ÷ 2 + 1 ? i - 1 : i - N - 1
            result += coeffs[i] * cis(k0 * k * x)
        end
        return real(result) / N
    end
end

"""
    _fourier_interp_weights(basis, N, L, x)

Build spectral interpolation weight vector for Fourier basis at position x.
"""
function _fourier_interp_weights(basis::FourierBasis, N::Int, L::Real, x::Real)
    k0 = 2π / L
    if isa(basis, RealFourier)
        # Complex weights for the unnormalized rfft half-spectrum (length N÷2+1).
        # Applied to the complex coefficient array; real() is taken afterwards.
        # Encodes the 1/N normalization and the factor-2 on interior modes.
        half = N ÷ 2 + 1
        weights = zeros(ComplexF64, half)
        weights[1] = 1.0 / N  # DC
        nyquist = iseven(N)
        for i in 2:half
            k = i - 1
            factor = (nyquist && i == half) ? 1.0 : 2.0
            weights[i] = (factor / N) * cis(k0 * k * x)
        end
        return weights
    else  # ComplexFourier
        weights = zeros(ComplexF64, N)
        for i in 1:N
            k = i <= N ÷ 2 + 1 ? i - 1 : i - N - 1
            weights[i] = cis(k0 * k * x) / N
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
    other_dims = ntuple(i -> i < axis ? i : i + 1, nd - 1)
    other_sizes = ntuple(i -> size(coeffs, other_dims[i]), nd - 1)

    if nd == 1
        # All dimensions are this axis — 1D case handled above
        return clenshaw_fn(vec(coeffs), x_native)
    end

    result_data = zeros(eltype(coeffs), other_sizes...)

    axis_range = 1:size(coeffs, axis)
    for idx in CartesianIndices(other_sizes)
        full_idx = ntuple(d -> d == axis ? axis_range : idx[d < axis ? d : d - 1], nd)
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
