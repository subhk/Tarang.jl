# Basis derivatives, conversions, differentiation matrices, and dispatch helpers.

# ============================================================================
# Derivative basis (D maps T to U)
# ============================================================================

"""
    derivative_basis(basis::JacobiBasis, order::Int=1)

Return the basis for the derivative of fields in this basis.

Derivative basis chain:
- ∂/∂x(ChebyshevT) → ChebyshevU (Jacobi a,b: -1/2,-1/2 → 1/2,1/2)
- ∂/∂x(ChebyshevU) → ChebyshevV (Jacobi a,b: 1/2,1/2 → 3/2,3/2)
- ∂/∂x(ChebyshevV) → Jacobi(5/2, 5/2)
- General: ∂/∂x P_n^{(a,b)} is proportional to P_{n-1}^{(a+1,b+1)}
"""
function derivative_basis(basis::ChebyshevT, order::Int=1)
    # ∂/∂x(T_n) is proportional to U_{n-1}
    # After differentiation, output is in ChebyshevU
    # Create new ChebyshevU basis with same domain parameters
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = ChebyshevU(coord;
                        size=basis.meta.size,
                        bounds=basis.meta.bounds,
                        dealias=basis.meta.dealias,
                        dtype=basis.meta.dtype)

    # Recursively apply for higher orders
    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::ChebyshevU, order::Int=1)
    # ∂/∂x(U_n) is proportional to polynomials in ChebyshevV family
    # ChebyshevU = Jacobi(1/2, 1/2), derivative -> Jacobi(3/2, 3/2) = ChebyshevV
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = ChebyshevV(coord;
                        size=basis.meta.size,
                        bounds=basis.meta.bounds,
                        dealias=basis.meta.dealias,
                        dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::ChebyshevV, order::Int=1)
    # ∂/∂x(V_n) → Jacobi(5/2, 5/2)
    # Continue the Jacobi parameter increment chain
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    # Create Jacobi basis with incremented parameters
    output = Jacobi(coord;
                    a=basis.a + 1.0,
                    b=basis.b + 1.0,
                    size=basis.meta.size,
                    bounds=basis.meta.bounds,
                    dealias=basis.meta.dealias,
                    dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::Legendre, order::Int=1)
    # ∂/∂x(P_n) is proportional to Jacobi with a=1, b=1
    # Legendre = Jacobi(0,0), derivative -> Jacobi(1,1)
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = Jacobi(coord;
                    a=1.0,
                    b=1.0,
                    size=basis.meta.size,
                    bounds=basis.meta.bounds,
                    dealias=basis.meta.dealias,
                    dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::Jacobi, order::Int=1)
    # ∂/∂x P_n^{(a,b)} is proportional to P_{n-1}^{(a+1,b+1)}
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = Jacobi(coord;
                    a=basis.a + 1.0,
                    b=basis.b + 1.0,
                    size=basis.meta.size,
                    bounds=basis.meta.bounds,
                    dealias=basis.meta.dealias,
                    dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::FourierBasis, order::Int=1)
    # Fourier derivative stays in same basis
    # ∂/∂x(exp(ikx)) = ik·exp(ikx), still in Fourier space
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    return basis
end

# ============================================================================
# Conversion matrices between bases
# ============================================================================

"""
    conversion_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)

Build conversion matrix from input basis to output basis.
"""
function conversion_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)
    # Key on output TYPE as well as (a,b): two bases can share Jacobi parameters
    # but differ in normalization (e.g. ChebyshevU vs raw Jacobi(1/2,1/2)).
    cache_key = (typeof(output_basis), input_basis.a, input_basis.b,
                 output_basis.a, output_basis.b)

    if haskey(input_basis._conversion_matrix_cache, cache_key)
        return input_basis._conversion_matrix_cache[cache_key]
    end

    N = input_basis.meta.size
    a0, b0 = input_basis.a, input_basis.b
    a1, b1 = output_basis.a, output_basis.b

    if abs(a0 - a1) < 1e-10 && abs(b0 - b1) < 1e-10
        # Identity (same parameters): exact, preserves solver sparsity.
        matrix = sparse(I, N, N)
    elseif isa(input_basis, ChebyshevT) && isa(output_basis, ChebyshevU)
        # Exact bidiagonal T_n=(U_n-U_{n-2})/2. Kept as a clean sparse special
        # case because it is on the implicit-solver hot path (expression_matrices
        # for the Convert operator) and must stay banded.
        matrix = _chebyshev_t_to_u_matrix(N)
    else
        # General case: collocation from the ACTUAL basis functions. Correct
        # regardless of Chebyshev-vs-Jacobi normalization and shift direction —
        # replaces the recurrence path (which mishandled both). See below.
        matrix = _collocation_conversion_matrix(input_basis, output_basis, N)
    end

    input_basis._conversion_matrix_cache[cache_key] = matrix
    return matrix
end

"""
    _collocation_conversion_matrix(input_basis, output_basis, N) -> sparse matrix

Build the basis-conversion matrix `M` (with `c_out = M * c_in`) by collocation:
sample both bases' functions on N nodes and solve `B_out * c_out = B_in * c_in`,
i.e. `M = B_out \\ B_in` where `B[i,n] = φ_n(x_i)` from `evaluate_basis`.

Because it uses the actual basis functions, it is correct for ANY pair of
Jacobi-family bases independent of their normalization (ChebyshevT/U/V, Legendre,
generic Jacobi) and shift direction — unlike the parameter-only recurrence path,
which mishandled the Chebyshev-vs-Jacobi normalization (e.g. T->V) and had a
broken down-shift recurrence. The collocation matrix is exact for polynomials of
degree < N (both bases span that space), so it preserves the represented function.
"""
function _collocation_conversion_matrix(input_basis::JacobiBasis,
                                        output_basis::JacobiBasis, N::Int)
    a, b = input_basis.meta.bounds
    native = _native_grid(input_basis, 1.0)                  # N nodes in [-1, 1]
    nodes = [a + (x + 1) * (b - a) / 2 for x in native]      # map to physical [a, b]
    B_in  = Matrix{Float64}(evaluate_basis(input_basis, nodes, 0:(N - 1)))
    B_out = Matrix{Float64}(evaluate_basis(output_basis, nodes, 0:(N - 1)))
    M = sparse(B_out \ B_in)                                 # c_out = M c_in
    droptol!(M, 1e-12)                                       # drop solver-noise fill
    return M
end

"""Build Jacobi conversion matrix."""
function _jacobi_conversion_matrix(N::Int, a0::Float64, b0::Float64, a1::Float64, b1::Float64)
    # Convert from P_n^{(a0,b0)} to P_n^{(a1,b1)}
    # This uses recurrence relations from Jacobi theory

    if abs(a0 - a1) < 1e-10 && abs(b0 - b1) < 1e-10
        return sparse(I, N, N)
    end

    # For ChebyshevT -> ChebyshevU conversion (a=-1/2 -> a=1/2)
    if abs(a0 + 0.5) < 1e-10 && abs(b0 + 0.5) < 1e-10 &&
       abs(a1 - 0.5) < 1e-10 && abs(b1 - 0.5) < 1e-10
        return _chebyshev_t_to_u_matrix(N)
    end

    # General case: use recursion
    return _general_jacobi_conversion(N, a0, b0, a1, b1)
end

"""Build ChebyshevT to ChebyshevU conversion matrix."""
function _chebyshev_t_to_u_matrix(N::Int)
    # T_n = (U_n - U_{n-2}) / 2 for n >= 2
    # T_0 = U_0, T_1 = U_1 / 2

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # T_0 -> U_0
    push!(I_list, 1); push!(J_list, 1); push!(V_list, 1.0)

    # T_1 -> U_1 / 2
    if N > 1
        push!(I_list, 2); push!(J_list, 2); push!(V_list, 0.5)
    end

    # T_n -> (U_n - U_{n-2}) / 2 for n >= 2
    for n in 2:(N-1)
        push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, 0.5)
        push!(I_list, n - 1); push!(J_list, n + 1); push!(V_list, -0.5)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
    _general_jacobi_conversion(N, a0, b0, a1, b1)

General Jacobi polynomial conversion matrix from P_n^{(a0,b0)} to P_n^{(a1,b1)}.

Uses the connection coefficient formula. For integer parameter shifts, the conversion
matrix is sparse (banded). For general shifts, we use quadrature-based projection.

The conversion is computed via:
    P_n^{(a0,b0)}(x) = Σ_m C_{nm} P_m^{(a1,b1)}(x)

where C_{nm} are the connection coefficients.

Reference:
- NIST DLMF 18.9 (Connection formulas)
- Project jacobi.py
"""
function _general_jacobi_conversion(N::Int, a0::Float64, b0::Float64, a1::Float64, b1::Float64)
    # Check if parameters are the same (identity conversion)
    if abs(a0 - a1) < 1e-12 && abs(b0 - b1) < 1e-12
        return sparse(I, N, N)
    end

    # Check for integer shifts - these have sparse representations
    da = a1 - a0
    db = b1 - b0

    # Special case: shift by integer in both parameters
    if abs(da - round(da)) < 1e-12 && abs(db - round(db)) < 1e-12
        da_int = Int(round(da))
        db_int = Int(round(db))

        # Build conversion through successive single-step shifts
        if da_int >= 0 && db_int >= 0
            return _jacobi_conversion_positive_shift(N, a0, b0, da_int, db_int)
        elseif da_int <= 0 && db_int <= 0
            return _jacobi_conversion_negative_shift(N, a0, b0, -da_int, -db_int)
        end
    end

    # General case: use quadrature-based projection
    # Evaluate input basis at output basis quadrature points, then project
    return _jacobi_conversion_quadrature(N, a0, b0, a1, b1)
end

"""
Build conversion for positive integer shifts in (a,b) parameters.
Uses the recurrence: P_n^{(a,b)} can be written in terms of P_m^{(a+1,b)} or P_m^{(a,b+1)}.
"""
function _jacobi_conversion_positive_shift(N::Int, a0::Float64, b0::Float64, da::Int, db::Int)
    result = sparse(I, N, N)

    # Shift a first, then b
    a_curr, b_curr = a0, b0

    for _ in 1:da
        step = _jacobi_a_shift_up_matrix(N, a_curr, b_curr)
        result = step * result
        a_curr += 1.0
    end

    for _ in 1:db
        step = _jacobi_b_shift_up_matrix(N, a_curr, b_curr)
        result = step * result
        b_curr += 1.0
    end

    return result
end

"""
Build conversion for negative integer shifts in (a,b) parameters.
"""
function _jacobi_conversion_negative_shift(N::Int, a0::Float64, b0::Float64, da::Int, db::Int)
    result = sparse(I, N, N)

    a_curr, b_curr = a0, b0

    for _ in 1:da
        step = _jacobi_a_shift_down_matrix(N, a_curr, b_curr)
        result = step * result
        a_curr -= 1.0
    end

    for _ in 1:db
        step = _jacobi_b_shift_down_matrix(N, a_curr, b_curr)
        result = step * result
        b_curr -= 1.0
    end

    return result
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a+1,b)}

Uses the recurrence relation (DLMF 18.9.5):
P_n^{(a,b)}(x) = c1 * P_n^{(a+1,b)}(x) + c2 * P_{n-1}^{(a+1,b)}(x)

where:
c1 = (n + a + b + 1) / (2n + a + b + 1)
c2 = (n + b) / (2n + a + b + 1)
"""
function _jacobi_a_shift_up_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        denom = 2*n + a + b + 1

        if abs(denom) > 1e-14
            c1 = (n + a + b + 1) / denom
            # DLMF 18.9.5: P_n^{(a,b)} = c1 P_n^{(a+1,b)} - (n+b)/denom P_{n-1}^{(a+1,b)}.
            # The subdiagonal term is NEGATIVE (the b-shift counterpart below is
            # positive — the asymmetry follows from P_n^{(a,b)}(-x)=(-1)^n P_n^{(b,a)}(x)).
            c2 = -(n + b) / denom

            # P_n^{(a,b)} contributes to P_n^{(a+1,b)} with coefficient c1
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, c1)

            # P_n^{(a,b)} contributes to P_{n-1}^{(a+1,b)} with coefficient c2
            if n > 0
                push!(I_list, n); push!(J_list, n + 1); push!(V_list, c2)
            end
        else
            # Degenerate case: identity
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, 1.0)
        end
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a,b+1)}

Similar recurrence with swapped roles of a and b.
"""
function _jacobi_b_shift_up_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        denom = 2*n + a + b + 1

        if abs(denom) > 1e-14
            c1 = (n + a + b + 1) / denom
            c2 = (n + a) / denom

            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, c1)

            if n > 0
                push!(I_list, n); push!(J_list, n + 1); push!(V_list, c2)
            end
        else
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, 1.0)
        end
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a-1,b)}

Uses the inverse/adjoint relationship.
"""
function _jacobi_a_shift_down_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        # Use recurrence: P_n^{(a-1,b)} = d1 * P_n^{(a,b)} + d2 * P_{n+1}^{(a,b)}
        # Derived from inverting the shift-up relation

        denom_n = 2*n + a + b
        denom_np1 = 2*(n+1) + a + b

        if abs(denom_n) > 1e-14
            d1 = (2*n + a + b) / (n + a + b)
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, d1)
        end

        if n + 1 < N && abs(denom_np1) > 1e-14
            d2 = -(n + 1) / (n + a + b + 1)
            push!(I_list, n + 1); push!(J_list, n + 2); push!(V_list, d2)
        end
    end

    if isempty(I_list)
        return sparse(I, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a,b-1)}
"""
function _jacobi_b_shift_down_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        denom_n = 2*n + a + b
        denom_np1 = 2*(n+1) + a + b

        if abs(denom_n) > 1e-14
            d1 = (2*n + a + b) / (n + a + b)
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, d1)
        end

        if n + 1 < N && abs(denom_np1) > 1e-14
            d2 = -(n + 1) / (n + a + b + 1)
            push!(I_list, n + 1); push!(J_list, n + 2); push!(V_list, d2)
        end
    end

    if isempty(I_list)
        return sparse(I, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Quadrature-based conversion for non-integer parameter shifts.

Evaluates input polynomials at output basis quadrature points,
then computes weighted projection coefficients.
"""
function _jacobi_conversion_quadrature(N::Int, a0::Float64, b0::Float64, a1::Float64, b1::Float64)
    # Get Gauss-Jacobi quadrature for output basis (need 2N points for exactness)
    N_quad = 2 * N
    nodes, weights = gauss_jacobi_quadrature(N_quad, a1, b1)

    # Evaluate input basis polynomials at quadrature points
    P_input = zeros(Float64, N_quad, N)
    for j in 1:N
        P_input[:, j] = jacobi_polynomial.(nodes, j - 1, a0, b0)
    end

    # Evaluate output basis polynomials at quadrature points
    P_output = zeros(Float64, N_quad, N)
    for j in 1:N
        P_output[:, j] = jacobi_polynomial.(nodes, j - 1, a1, b1)
    end

    # Compute conversion matrix via weighted inner products
    # C_ij = <P_i^{out}, P_j^{in}>_w / <P_i^{out}, P_i^{out}>_w
    # where w is the Jacobi weight for output basis

    W = Diagonal(weights)

    # Normalization factors for output basis
    norms = zeros(Float64, N)
    for i in 1:N
        norms[i] = dot(P_output[:, i], W * P_output[:, i])
    end

    # Projection matrix
    C = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            if abs(norms[i]) > 1e-14
                C[i, j] = dot(P_output[:, i], W * P_input[:, j]) / norms[i]
            end
        end
    end

    # Sparsify small entries
    threshold = 1e-14 * maximum(abs.(C))
    C[abs.(C) .< threshold] .= 0.0

    return sparse(C)
end

"""
Evaluate Jacobi polynomial P_n^{(a,b)}(x) using three-term recurrence.
"""
function jacobi_polynomial(x::Float64, n::Int, a::Float64, b::Float64)
    if n == 0
        return 1.0
    elseif n == 1
        return 0.5 * (a - b + (a + b + 2) * x)
    end

    P_prev2 = 1.0
    P_prev1 = 0.5 * (a - b + (a + b + 2) * x)

    for k in 2:n
        # Three-term recurrence coefficients
        k_f = Float64(k)
        c1 = 2 * k_f * (k_f + a + b) * (2*k_f + a + b - 2)
        c2 = (2*k_f + a + b - 1) * (a^2 - b^2)
        c3 = (2*k_f + a + b - 2) * (2*k_f + a + b - 1) * (2*k_f + a + b)
        c4 = 2 * (k_f + a - 1) * (k_f + b - 1) * (2*k_f + a + b)

        if abs(c1) > 1e-14
            P_curr = ((c2 + c3 * x) * P_prev1 - c4 * P_prev2) / c1
        else
            P_curr = P_prev1
        end

        P_prev2 = P_prev1
        P_prev1 = P_curr
    end

    return P_prev1
end

"""
Compute Gauss-Jacobi quadrature nodes and weights.

Uses the Golub-Welsch algorithm based on the eigenvalues of the Jacobi matrix.
Special handling for Chebyshev T (a=b=-0.5) where the standard algorithm fails.
"""
function gauss_jacobi_quadrature(N::Int, a::Float64, b::Float64)
    if N < 1
        return Float64[], Float64[]
    end

    # Special case: Chebyshev T (a=b=-0.5)
    # The standard Golub-Welsch algorithm fails for a+b=-1 because denominators vanish.
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

    # Build tridiagonal Jacobi matrix
    # The eigenvalues give the nodes, eigenvectors give the weights

    # Diagonal elements
    d = zeros(Float64, N)
    for n in 0:(N-1)
        num = b^2 - a^2
        denom = (2*n + a + b) * (2*n + a + b + 2)
        if abs(denom) > 1e-14
            d[n + 1] = num / denom
        end
    end

    # Sub/super-diagonal elements
    e = zeros(Float64, N - 1)
    for n in 1:(N-1)
        num = 2 * sqrt(n * (n + a) * (n + b) * (n + a + b))
        denom = (2*n + a + b) * sqrt((2*n + a + b - 1) * (2*n + a + b + 1))
        if abs(denom) > 1e-14
            e[n] = num / denom
        end
    end

    # Build tridiagonal matrix and compute eigendecomposition
    J = SymTridiagonal(d, e)
    eigenvalues, eigenvectors = eigen(J)

    # Nodes are eigenvalues
    nodes = eigenvalues

    # Weights from first component of eigenvectors
    # w_i = μ_0 * v_i[1]^2 where μ_0 = ∫_{-1}^{1} (1-x)^a (1+x)^b dx
    mu0 = 2^(a + b + 1) * gamma(a + 1) * gamma(b + 1) / gamma(a + b + 2)
    weights = mu0 .* eigenvectors[1, :].^2

    return nodes, weights
end

# ============================================================================
# Differentiation matrices
# ============================================================================

"""
    differentiation_matrix(basis::JacobiBasis, order::Int=1)

Build spectral differentiation matrix.
"""
function differentiation_matrix(basis::JacobiBasis, order::Int=1)
    if haskey(basis._differentiation_matrix_cache, order)
        return basis._differentiation_matrix_cache[order]
    end

    N = basis.meta.size
    a, b = basis.a, basis.b

    # Get domain scaling factor
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("differentiation_matrix: domain length is zero"))
    end
    scale = 2.0 / L

    matrix = _jacobi_differentiation_matrix(N, a, b, order) * scale^order

    basis._differentiation_matrix_cache[order] = matrix
    return matrix
end

"""Build Jacobi differentiation matrix."""
function _jacobi_differentiation_matrix(N::Int, a::Float64, b::Float64, order::Int)
    # d/dx P_n^{(a,b)} = (n + a + b + 1) / 2 * P_{n-1}^{(a+1,b+1)}

    if order == 0
        return sparse(I, N, N)
    end

    # Build single derivative matrix
    D = spzeros(Float64, N, N)

    # Chebyshev T case (a = b = -1/2)
    if abs(a + 0.5) < 1e-10 && abs(b + 0.5) < 1e-10
        D = _chebyshev_t_differentiation_matrix(N)
    # Legendre case (a = b = 0)
    elseif abs(a) < 1e-10 && abs(b) < 1e-10
        D = _legendre_differentiation_matrix(N)
    else
        # General Jacobi
        D = _general_jacobi_differentiation_matrix(N, a, b)
    end

    # Apply multiple times for higher orders
    result = D
    for _ in 2:order
        result = D * result
    end

    return result
end

"""Build Chebyshev T differentiation matrix."""
function _chebyshev_t_differentiation_matrix(N::Int)
    # Standard Chebyshev differentiation recurrence:
    # c'_0 = sum_{j odd} j*c_j  (factor of 1/2 relative to other rows)
    # c'_k = sum_{j=k+1, j-k odd} 2*j*c_j for k >= 1

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for k in 0:(N-2)
        for j in (k+1):(N-1)
            if (j - k) % 2 == 1
                push!(I_list, k + 1)
                push!(J_list, j + 1)
                # Factor of 1/2 for k=0 row due to Chebyshev normalization
                coeff = k == 0 ? Float64(j) : 2.0 * j
                push!(V_list, coeff)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""Build Legendre differentiation matrix."""
function _legendre_differentiation_matrix(N::Int)
    # Legendre differentiation formula:
    # d(P_n)/dx = sum_{k<n, (n-k) odd} (2k+1) * P_k
    #
    # In matrix form: D[k+1, j+1] = (2k+1) when (j-k) is odd and j > k
    # This represents: (df/dx)_k = sum_{j>k, (j-k) odd} (2k+1) * f_j

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for k in 0:(N-2)
        for j in (k+1):(N-1)
            if (j - k) % 2 == 1
                push!(I_list, k + 1)
                push!(J_list, j + 1)
                push!(V_list, 2.0 * k + 1.0)  # Note: 2*k+1, not 2*j+1
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Build general Jacobi differentiation matrix that maps P^{(a,b)} coefficients
to P^{(a,b)} coefficients of the derivative.

Uses: d/dx P_n^{(a,b)} = (n+a+b+1)/2 · P_{n-1}^{(a+1,b+1)}
then converts from P^{(a+1,b+1)} back to P^{(a,b)}.
"""
function _general_jacobi_differentiation_matrix(N::Int, a::Float64, b::Float64)
    # Step 1: Build the raw derivative super-diagonal in P^{(a+1,b+1)} basis
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 1:(N-1)
        coeff = (n + a + b + 1) / 2
        push!(I_list, n)
        push!(J_list, n + 1)
        push!(V_list, coeff)
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    D_raw = sparse(I_list, J_list, V_list, N, N)

    # Step 2: Convert from P^{(a+1,b+1)} back to P^{(a,b)}
    # This uses the negative shift: P^{(a+1,b+1)} → P^{(a,b)}
    C = _jacobi_conversion_matrix(N, a + 1.0, b + 1.0, a, b)

    return C * D_raw
end

# ============================================================================
# Basis dispatcher helpers
# ============================================================================

_basis_builder(::Type{RealFourier}) = _RealFourier_constructor
_basis_builder(::Type{ComplexFourier}) = _ComplexFourier_constructor
_basis_builder(::Type{ChebyshevT}) = _ChebyshevT_constructor
_basis_builder(::Type{ChebyshevU}) = _ChebyshevU_constructor
_basis_builder(::Type{ChebyshevV}) = _ChebyshevV_constructor
_basis_builder(::Type{Legendre}) = _Legendre_constructor
_basis_builder(::Type{Ultraspherical}) = _Ultraspherical_constructor
_basis_builder(::Type{Jacobi}) = _Jacobi_constructor
_basis_builder(::Type{T}) where {T<:Basis} = error("No basis builder registered for type $(T)")

function dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Basis}
    if length(args) != 1
        throw(ArgumentError("$(T) expects exactly one Coordinate argument"))
    end
    return (args, kwargs)
end

function dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Basis}
    coord = args[1]
    if !isa(coord, Coordinate)
        throw(ArgumentError("$(T) requires a Coordinate argument"))
    end
    return true
end

function invoke_constructor(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Basis}
    builder = _basis_builder(T)
    coord = args[1]
    return builder(coord; kwargs...)
end
