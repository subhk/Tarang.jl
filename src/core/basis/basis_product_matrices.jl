# NCC product matrices plus valid-element filtering for basis layouts.

# ============================================================================
# Product matrices for NCC support
# ============================================================================

"""
    ncc_matrix(ncc_basis, arg_basis, out_basis, coeffs; cutoff=1e-6)

Build full NCC matrix via direct summation of product matrices.

The NCC matrix represents multiplication by a spatially-varying coefficient field:
    (ncc * operand)_coeffs = NCC_matrix @ operand_coeffs

where ncc is expanded in spectral coefficients and each mode contributes
via its product_matrix.

# Arguments
- `ncc_basis`: Basis for the NCC field
- `arg_basis`: Basis for the argument/operand field
- `out_basis`: Basis for the output field
- `coeffs`: Spectral coefficients of the NCC field
- `cutoff`: Coefficient cutoff for sparsity (default 1e-6)

# Returns
Sparse matrix representing the NCC multiplication operation.
"""
function ncc_matrix(ncc_basis::Basis, arg_basis, out_basis, coeffs::AbstractVector; cutoff::Float64=1e-6)
    N = length(coeffs)
    total = nothing

    for i in 1:N
        coeff = coeffs[i]

        # Skip small coefficients
        if abs(coeff) <= cutoff
            continue
        end

        # Get product matrix for mode i-1 (0-based mode index)
        matrix = product_matrix(ncc_basis, arg_basis, out_basis, i - 1)

        # Scale by coefficient and accumulate
        if total === nothing
            total = coeff * matrix
        else
            total = total + coeff * matrix
        end
    end

    if total === nothing
        N_out = out_basis === nothing ? ncc_basis.meta.size : out_basis.meta.size
        N_arg = arg_basis === nothing ? 1 : arg_basis.meta.size
        return spzeros(Float64, N_out, N_arg)
    end

    # Eliminate small entries
    droptol!(total, cutoff)
    return total
end

"""
    product_matrix(basis::JacobiBasis, arg_basis, out_basis, ncc_mode::Int)

Build multiplication matrix for Non-Constant Coefficient (NCC) terms.

This computes the matrix M such that:
    (f * g)_coeffs = M @ g_coeffs
where f is the NCC field (expanded to ncc_mode) and g is the argument field.
"""
function product_matrix(basis::JacobiBasis, arg_basis, out_basis, ncc_mode::Int)
    cache_key = (arg_basis, out_basis, ncc_mode)

    # Check cache
    if haskey(basis._product_matrix_cache, cache_key)
        return basis._product_matrix_cache[cache_key]
    end

    N = basis.meta.size
    a, b = basis.a, basis.b

    # Build Jacobi product matrix using linearization coefficients
    # P_m * P_n = sum_k c_{m,n,k} P_k
    matrix = _jacobi_product_matrix(N, a, b, ncc_mode, arg_basis, out_basis)

    basis._product_matrix_cache[cache_key] = matrix
    return matrix
end

"""
    product_matrix(basis::RealFourier, arg_basis, out_basis, ncc_mode::Int)

Build multiplication matrix for RealFourier NCC.

For Fourier: cos(m*x) * cos(n*x) = 0.5*(cos((m-n)*x) + cos((m+n)*x))
"""
function product_matrix(basis::RealFourier, arg_basis, out_basis, ncc_mode::Int)
    cache_key = (arg_basis, out_basis, ncc_mode)

    if haskey(basis._product_matrix_cache, cache_key)
        return basis._product_matrix_cache[cache_key]
    end

    N_out = out_basis === nothing ? basis.meta.size : out_basis.meta.size
    N_arg = arg_basis === nothing ? 1 : arg_basis.meta.size

    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # Get physical wavenumber directly from storage index.
    # RealFourier storage: [cos_0, cos_1, msin_1, cos_2, msin_2, ...]
    # 1-based index j: j=1→cos_0(k=0), j=2→cos_1(k=1), j=3→msin_1(k=1), j=4→cos_2(k=2), ...
    # Do NOT use wavenumbers(basis) here — that returns FFT-ordered wavenumbers
    # which don't correspond to the interleaved cos/sin storage layout.
    j = ncc_mode + 1  # convert to 1-based storage index
    if j == 1
        m = 0
    elseif iseven(j)
        m = j ÷ 2
    else
        m = (j - 1) ÷ 2
    end
    is_sin_mode = ncc_mode != 0 && isodd(j)

    # Build sparse product matrix
    if m == 0
        # Constant NCC: identity or truncation
        if !is_sin_mode  # cos mode
            matrix = sparse(I, N_out, N_arg)
        else  # sin mode (which is zero for k=0)
            matrix = spzeros(Float64, N_out, N_arg)
        end
    else
        matrix = _build_real_fourier_product_matrix(N_arg, N_out, m, is_sin_mode)
    end

    basis._product_matrix_cache[cache_key] = matrix
    return matrix
end

"""
Build RealFourier product matrix for specific wavenumber.

The RealFourier product matrix handles multiplication of trig functions:
- 2 cos(mx) cos(nx) = cos((m+n)x) + cos((m-n)x)
- 2 cos(mx) sin(nx) = sin((m+n)x) - sin((m-n)x)  (msin = -sin notation)
- 2 sin(mx) cos(nx) = sin((m+n)x) + sin((m-n)x)
- 2 sin(mx) sin(nx) = -cos((m+n)x) + cos((m-n)x)
"""
function _build_real_fourier_product_matrix(N_arg::Int, N_out::Int, m::Int, is_sin::Bool)
    # Use wavenumber intersection approach
    # Indexing: 1 -> cos0, even indices -> cos(k>=1), odd indices >1 -> -sin(k) (msin)

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # Generate wavenumber arrays
    # RealFourier stores: [cos_0, cos_1, msin_1, cos_2, msin_2, ...]
    # where msin = -sin

    # Process the three coupling cases:
    # 1. k_out = k_ncc + k_arg (rows_p, cols_p)
    # 2. k_out = k_ncc - k_arg (rows_m, cols_m)
    # 3. k_out = k_arg - k_ncc for negative result (rows_mn, cols_mn)

    # In Julia 1-based:
    # cos_0 at 1, cos_1 at 2, msin_1 at 3, cos_2 at 4, msin_2 at 5...
    for j_arg in 1:N_arg
        if j_arg == 1
            n = 0
            is_sin_arg = false
        elseif iseven(j_arg)
            n = j_arg ÷ 2
            is_sin_arg = false
        else
            n = (j_arg - 1) ÷ 2
            is_sin_arg = true
        end

        if !is_sin_arg
            # cos input modes
            k_plus = m + n
            if k_plus >= 0
                out_cos_idx = k_plus == 0 ? 1 : 2 * k_plus
                out_sin_idx = 2 * k_plus + 1

                if is_sin
                    # sin(mx) * cos(nx) -> sin((m+n)x)
                    if out_sin_idx <= N_out
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * cos(nx) -> cos((m+n)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end

            k_minus = m - n
            if k_minus >= 0
                out_cos_idx = k_minus == 0 ? 1 : 2 * k_minus
                out_sin_idx = 2 * k_minus + 1

                if is_sin
                    # sin(mx) * cos(nx) -> sin((m-n)x)
                    if out_sin_idx <= N_out && k_minus > 0
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * cos(nx) -> cos((m-n)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end

            k_nm = n - m
            if k_nm > 0
                out_cos_idx = 2 * k_nm
                out_sin_idx = 2 * k_nm + 1

                if is_sin
                    # sin(mx) * cos(nx) -> -sin((n-m)x) = msin((n-m)x)
                    if out_sin_idx <= N_out
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, -0.5)
                    end
                else
                    # cos(mx) * cos(nx) -> cos((n-m)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end
        else
            if n == 0
                continue
            end

            # msin input modes
            k_plus = m + n
            out_cos_idx = k_plus == 0 ? 1 : 2 * k_plus
            out_sin_idx = 2 * k_plus + 1

            if is_sin
                # sin(mx) * sin(nx) = -0.5*cos((m+n)x) + 0.5*cos((m-n)x)
                # -> -cos((m+n)x) contribution
                if out_cos_idx <= N_out
                    push!(I_list, out_cos_idx)
                    push!(J_list, j_arg)
                    push!(V_list, -0.5)
                end
            else
                # cos(mx) * sin(nx) -> sin((m+n)x)
                if out_sin_idx <= N_out
                    push!(I_list, out_sin_idx)
                    push!(J_list, j_arg)
                    push!(V_list, 0.5)
                end
            end

            k_minus = m - n
            if k_minus >= 0
                out_cos_idx = k_minus == 0 ? 1 : 2 * k_minus
                out_sin_idx = 2 * k_minus + 1

                if is_sin
                    # sin(mx) * sin(nx) -> cos((m-n)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * sin(nx) -> -sin((m-n)x)
                    if out_sin_idx <= N_out && k_minus > 0
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, -0.5)
                    end
                end
            end

            k_nm = n - m
            if k_nm > 0
                out_cos_idx = 2 * k_nm
                out_sin_idx = 2 * k_nm + 1

                if is_sin
                    # sin(mx) * sin(nx) -> cos((n-m)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * sin(nx) -> sin((n-m)x)
                    if out_sin_idx <= N_out
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N_out, N_arg)
    end

    return sparse(I_list, J_list, V_list, N_out, N_arg)
end

"""Build Jacobi product matrix using linearization coefficients."""
function _jacobi_product_matrix(N::Int, a::Float64, b::Float64,
                                 ncc_mode::Int, arg_basis, out_basis)
    N_out = out_basis === nothing ? N : out_basis.meta.size
    N_arg = arg_basis === nothing ? 1 : arg_basis.meta.size

    # For Jacobi polynomials: P_m^{(a,b)} * P_n^{(a,b)} = sum_k c_{m,n,k} P_k^{(a,b)}
    # The linearization coefficients c_{m,n,k} are computed using Clebsch-Gordan-like formulas

    m = ncc_mode

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N_arg-1)
        # Compute linearization coefficients for P_m * P_n
        coeffs = _jacobi_linearization_coefficients(m, n, a, b, N_out)

        for (k, c) in enumerate(coeffs)
            if abs(c) > 1e-14 && k <= N_out
                push!(I_list, k)
                push!(J_list, n + 1)
                push!(V_list, c)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N_out, N_arg)
    end

    return sparse(I_list, J_list, V_list, N_out, N_arg)
end

"""
Compute Jacobi polynomial linearization coefficients.
P_m^{(a,b)}(x) * P_n^{(a,b)}(x) = sum_{k=|m-n|}^{m+n} c_k P_k^{(a,b)}(x)

Uses Clenshaw algorithm with Jacobi matrices.
"""
function _jacobi_linearization_coefficients(m::Int, n::Int, a::Float64, b::Float64, N_max::Int)
    coeffs = zeros(Float64, N_max)

    # Note: Linearization coefficients are only non-zero for |m-n| <= k <= m+n
    # This constraint is implicitly handled by the quadrature-based computation

    # Special case: m=0 or n=0 (multiplication by P_0 = 1)
    if m == 0
        if n < N_max
            coeffs[n + 1] = 1.0
        end
        return coeffs
    end
    if n == 0
        if m < N_max
            coeffs[m + 1] = 1.0
        end
        return coeffs
    end

    # General Jacobi case (includes all special cases: Chebyshev, Legendre, etc.):
    # Note: The Chebyshev T (a=b=-1/2) and U (a=b=1/2) formulas for product
    # linearization use the normalized Chebyshev polynomials T_n and U_n,
    # not the standard Jacobi polynomials P_n^{(a,b)}. Since this function
    # computes coefficients for standard Jacobi polynomials, we use numerical
    # quadrature for all cases to ensure correctness.
    # P_m^{(a,b)} * P_n^{(a,b)} = sum_k A_{m,n,k}^{(a,b)} P_k^{(a,b)}

    # Build Jacobi matrix J (tridiagonal) for the recurrence relation
    # x * P_n^{(a,b)} = A_n * P_{n-1} + B_n * P_n + C_n * P_{n+1}

    N_work = max(m + n + 2, N_max + 1)

    # Compute using matrix Clenshaw algorithm
    # This evaluates P_m(J) where J is the Jacobi matrix
    coeffs = _jacobi_linearization_clenshaw(m, n, a, b, N_max, N_work)

    return coeffs
end

"""
Compute Jacobi linearization using Gauss-Jacobi quadrature.
This provides accurate linearization coefficients for general Jacobi polynomials.

The linearization coefficients are computed via projection:
c_k = ∫_{-1}^{1} P_m^{(a,b)}(x) P_n^{(a,b)}(x) P_k^{(a,b)}(x) w(x) dx / h_k

where w(x) = (1-x)^a (1+x)^b is the weight function and h_k is the normalization.
"""
function _jacobi_linearization_clenshaw(m::Int, n::Int, a::Float64, b::Float64, N_max::Int, N_work::Int)
    coeffs = zeros(Float64, N_max)

    # Use Gauss-Jacobi quadrature with enough points for exactness
    # For product of P_m * P_n * P_k (degrees m, n, k), we need N_quad >= (m+n+k+1)/2
    N_quad = max(m + n + N_max + 2, 2 * N_work)

    # Get Gauss-Jacobi quadrature nodes and weights
    nodes, weights = _gauss_jacobi_quadrature(N_quad, a, b)

    # Evaluate P_m and P_n at quadrature nodes
    P_m = _jacobi_polynomial_values(m, a, b, nodes)
    P_n = _jacobi_polynomial_values(n, a, b, nodes)

    # Product values
    product = P_m .* P_n

    # Compute coefficients by projection
    for k in 0:(N_max-1)
        P_k = _jacobi_polynomial_values(k, a, b, nodes)

        # Normalization factor h_k for Jacobi polynomials
        h_k = _jacobi_norm_squared(k, a, b)

        # Inner product: ∫ P_m * P_n * P_k * w dx ≈ sum of weights * values
        inner_prod = sum(weights .* product .* P_k)

        coeffs[k + 1] = inner_prod / h_k
    end

    # Clean up small values
    cutoff = 1e-12
    for k in 1:N_max
        if abs(coeffs[k]) < cutoff
            coeffs[k] = 0.0
        end
    end

    return coeffs
end

"""
Compute Gauss-Jacobi quadrature nodes and weights.
Uses the Golub-Welsch algorithm via eigenvalue decomposition of the Jacobi matrix.
Special handling for Chebyshev T (a=b=-0.5) and Chebyshev U (a=b=0.5) cases.
"""
function _gauss_jacobi_quadrature(N::Int, a::Float64, b::Float64)
    if N < 1
        return Float64[], Float64[]
    end

    # Special case: Chebyshev T (a=b=-0.5)
    # The standard Golub-Welsch algorithm fails for a+b=-1 because beta[1]=0.
    # Use the known Chebyshev-Gauss quadrature formula instead.
    if abs(a + 0.5) < 1e-10 && abs(b + 0.5) < 1e-10
        nodes = zeros(Float64, N)
        weights = zeros(Float64, N)
        for k in 1:N
            nodes[k] = cos((2*k - 1) * π / (2*N))
            weights[k] = π / N
        end
        perm = sortperm(nodes)
        return nodes[perm], weights[perm]
    end

    # Build the symmetric tridiagonal Jacobi matrix
    alpha = zeros(Float64, N)   # Main diagonal
    beta = zeros(Float64, N-1)  # Sub/super diagonal

    for k in 0:(N-1)
        # Diagonal element
        denom = (2*k + a + b) * (2*k + a + b + 2)
        if abs(denom) > 1e-14
            alpha[k+1] = (b^2 - a^2) / denom
        else
            alpha[k+1] = 0.0
        end

        # Off-diagonal element
        if k < N - 1
            num = 4 * (k + 1) * (k + 1 + a) * (k + 1 + b) * (k + 1 + a + b)
            denom = (2*k + a + b + 1) * (2*k + a + b + 2)^2 * (2*k + a + b + 3)
            if abs(denom) > 1e-14 && num >= 0
                beta[k+1] = sqrt(num / denom)
            else
                beta[k+1] = 0.0
            end
        end
    end

    # Build symmetric tridiagonal matrix and compute eigendecomposition
    J = SymTridiagonal(alpha, beta)
    eigen_decomp = eigen(J)
    nodes = eigen_decomp.values
    V = eigen_decomp.vectors

    # Weights: w_i = μ_0 * v_{1,i}^2
    mu_0 = 2^(a + b + 1) * exp(lgamma(a + 1) + lgamma(b + 1) - lgamma(a + b + 2))
    weights = mu_0 * (V[1, :].^2)

    return nodes, weights
end

"""
Evaluate Jacobi polynomial P_n^{(a,b)}(x) at given points using stable recurrence.
"""
function _jacobi_polynomial_values(n::Int, a::Float64, b::Float64, x::AbstractVector{Float64})
    N_pts = length(x)

    if n == 0
        return ones(Float64, N_pts)
    end

    P_prev = ones(Float64, N_pts)
    P_curr = @. (a - b) / 2 + (a + b + 2) / 2 * x

    if n == 1
        return P_curr
    end

    for k in 1:(n-1)
        k_ab = 2*k + a + b
        A_k = (k_ab + 1) * (k_ab + 2) / (2 * (k + 1) * (k + a + b + 1))
        B_k = (a^2 - b^2) * (k_ab + 1) / (2 * (k + 1) * (k + a + b + 1) * k_ab)
        C_k = (k + a) * (k + b) * (k_ab + 2) / ((k + 1) * (k + a + b + 1) * k_ab)

        P_next = @. (A_k * x + B_k) * P_curr - C_k * P_prev
        P_prev = P_curr
        P_curr = P_next
    end

    return P_curr
end

"""
Compute the squared norm (h_n) of Jacobi polynomial P_n^{(a,b)}.
"""
function _jacobi_norm_squared(n::Int, a::Float64, b::Float64)
    if n == 0
        return 2^(a + b + 1) * exp(lgamma(a + 1) + lgamma(b + 1) - lgamma(a + b + 2))
    end

    log_h = (a + b + 1) * log(2) - log(2*n + a + b + 1)
    log_h += lgamma(n + a + 1) + lgamma(n + b + 1)
    log_h -= lgamma(n + 1) + lgamma(n + a + b + 1)

    return exp(log_h)
end

"""Compute Legendre linearization coefficient using Clebsch-Gordan formula."""
function _legendre_linearization_coeff(m::Int, n::Int, k::Int)
    # P_m * P_n = sum_k c_{m,n,k} P_k
    # c_{m,n,k} = (2k+1) * C(m,n,k)^2 where C is Clebsch-Gordan

    # Selection rules: |m-n| <= k <= m+n, and m+n+k is even
    if k < abs(m - n) || k > m + n || (m + n + k) % 2 != 0
        return 0.0
    end

    # Use 3j symbol formula (Wigner 3j symbols via Clebsch-Gordan coefficients)
    # Reference: Varshalovich et al., "Quantum Theory of Angular Momentum"
    s = (m + n + k) ÷ 2

    # Compute in log space using lgamma for numerical stability at all indices.
    # c_{m,n,k} = (2k+1) * [s-m)!(s-n)!(s-k)! / (s+1)!]^2 * (2s)! / [(2s-2m)!(2s-2n)!(2s-2k)!]
    # Note: (2s+1)!/(2s)! = 2s+1, so the original formula with (2s+1)! simplifies.
    log_num = lgamma(s - m + 1) + lgamma(s - n + 1) + lgamma(s - k + 1)
    log_den = lgamma(s + 2)  # (s+1)!

    log_f1 = lgamma(2*s - 2*m + 1) + lgamma(2*s - 2*n + 1) + lgamma(2*s - 2*k + 1)
    log_f2 = lgamma(2*s + 2)  # (2s+1)!

    log_coeff = log(2*k + 1) + 2*(log_num - log_den) + (log_f2 - log_f1)

    return exp(log_coeff)
end

# ============================================================================
# Valid elements / mode filtering
# ============================================================================

"""
    valid_elements(basis::Basis, tensorsig, grid_space, elements)

Determine which elements are valid for the given tensor signature.
"""
function valid_elements(basis::RealFourier, tensorsig, grid_space, elements)
    vshape = (length(tensorsig) > 0 ? prod(cs.dim for cs in tensorsig) : 1,) .* size(elements[1])
    valid = trues(vshape)

    if !grid_space[1]
        # Drop msin part of k=0 for all Cartesian components
        groups = elements_to_groups(basis, grid_space, elements)
        for i in eachindex(valid)
            if groups[1][i] == 0 && elements[1][i] % 2 == 1
                valid[i] = false
            end
        end
    end

    return valid
end

function valid_elements(basis::JacobiBasis, tensorsig, grid_space, elements)
    # Jacobi bases have all elements valid
    vshape = (length(tensorsig) > 0 ? prod(cs.dim for cs in tensorsig) : 1,) .* size(elements[1])
    return trues(vshape)
end

"""Convert elements to groups."""
function elements_to_groups(basis::RealFourier, grid_space, elements)
    # RealFourier has group_shape = (2,), groups are element ÷ 2
    return (elements[1] .÷ 2,)
end

function elements_to_groups(basis::JacobiBasis, grid_space, elements)
    # Jacobi has group_shape = (1,), groups equal elements
    return elements
end
