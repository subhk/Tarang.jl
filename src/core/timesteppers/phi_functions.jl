# Exponential Time Differencing (ETD) utility functions
"""
Compute φ functions for exponential time differencing methods.

φ₀(z) = exp(z)
φ₁(z) = (exp(z) - 1) / z
φ₂(z) = (exp(z) - 1 - z) / z²
φ₃(z) = (exp(z) - 1 - z - z²/2) / z³

These functions handle the z ≈ 0 case using Taylor expansions.
"""
function phi_functions(z::Number)
    if abs(z) < 1e-8
        # Use Taylor expansions for small z to avoid numerical issues
        φ₀ = 1 + z + z^2/2 + z^3/6 + z^4/24
        φ₁ = 1 + z/2 + z^2/6 + z^3/24 + z^4/120
        φ₂ = 1/2 + z/6 + z^2/24 + z^3/120 + z^4/720
        φ₃ = 1/6 + z/24 + z^2/120 + z^3/720 + z^4/5040
    else
        exp_z = exp(z)
        φ₀ = exp_z
        φ₁ = (exp_z - 1) / z
        φ₂ = (exp_z - 1 - z) / z^2
        φ₃ = (exp_z - 1 - z - z^2/2) / z^3
    end
    return φ₀, φ₁, φ₂, φ₃
end

function phi_functions_matrix(A::AbstractMatrix, dt::Float64)
    """Compute matrix φ functions for exponential integrators"""

    z = dt * A

    # Check matrix norm for stability
    z_norm = norm(z)

    if z_norm < 1e-8
        # Use Taylor expansions for small matrices (numerically stable)
        I = Matrix{eltype(A)}(LinearAlgebra.I, size(A, 1), size(A, 1))

        # Taylor series: φ₀(z) = I + z + z²/2 + z³/6 + z⁴/24
        exp_z = I + z + (z^2)/2 + (z^3)/6 + (z^4)/24

        # Taylor series: φ₁(z) = I + z/2 + z²/6 + z³/24 + z⁴/120
        φ₁ = I + z/2 + (z^2)/6 + (z^3)/24 + (z^4)/120

        # Taylor series: φ₂(z) = I/2 + z/6 + z²/24 + z³/120 + z⁴/720
        φ₂ = I/2 + z/6 + (z^2)/24 + (z^3)/120 + (z^4)/720

        return exp_z, φ₁, φ₂

    elseif z_norm < 50.0
        # Use matrix exponential for moderate matrices
        I = Matrix{eltype(A)}(LinearAlgebra.I, size(A, 1), size(A, 1))

        try
            exp_z = exp(z)

            # Use stable computation
            φ₁ = _compute_phi1_stable(z, exp_z, I)
            φ₂ = _compute_phi2_stable(z, exp_z, I, φ₁)

            return exp_z, φ₁, φ₂

        catch e
            @warn "Matrix exponential failed: $e, using Padé approximation"
            return _phi_functions_pade(z)
        end
    else
        # Use Krylov subspace methods for large/stiff matrices
        @warn "Matrix is large or stiff (norm=$z_norm), using Krylov approximation"
        return _phi_functions_krylov(z)
    end
end

function _compute_phi1_stable(z, exp_z, I)
    """Stable computation of φ₁"""
    z_norm = norm(z)
    if z_norm < 1e-2
        # Use series expansion for better accuracy
        return I + z/2 + z^2/6 + z^3/24 + z^4/120
    else
        # Use left-division z \ (exp_z - I) for robustness with near-singular z,
        # rather than right-division (exp_z - I) / z which assumes z is invertible.
        return z \ (exp_z - I)
    end
end

function _compute_phi2_stable(z, exp_z, I, φ₁)
    """
    Stable computation of φ₂(z) = (exp(z) - 1 - z) / z² = (φ₁(z) - I) / z

    Reference: Cox & Matthews (2002), Hochbruck & Ostermann (2010)
    """
    z_norm = norm(z)
    if z_norm < 1e-2
        # Use series expansion for better accuracy near z=0
        # φ₂(z) = 1/2 + z/6 + z²/24 + z³/120 + z⁴/720 + O(z⁵)
        return I/2 + z/6 + z^2/24 + z^3/120 + z^4/720
    else
        # Use left-division for robustness with near-singular z
        return z \ (φ₁ - I)
    end
end

function _phi_functions_pade(z)
    """Padé approximation fallback for φ functions.
    Uses scaling-and-squaring with Padé [3/3] for better accuracy than [1/1].
    """
    n = size(z, 1)
    I_mat = Matrix{eltype(z)}(LinearAlgebra.I, n, n)

    # Scaling: reduce norm by dividing by 2^s
    z_norm = norm(z)
    s = max(0, ceil(Int, log2(z_norm / 1.0)))
    z_scaled = z / 2^s

    # Padé [3/3] approximation for exp(z_scaled)
    # R_{3,3}(z) = N(z)/D(z) where:
    # N(z) = I + z/2 + z²/10 + z³/120
    # D(z) = I - z/2 + z²/10 - z³/120
    z2 = z_scaled^2
    z3 = z2 * z_scaled
    N_pade = I_mat + z_scaled/2 + z2/10 + z3/120
    D_pade = I_mat - z_scaled/2 + z2/10 - z3/120
    exp_scaled = D_pade \ N_pade

    # Squaring: exp(z) = exp(z_scaled)^(2^s)
    exp_z = exp_scaled
    for _ in 1:s
        exp_z = exp_z * exp_z
    end

    # Compute φ functions from exp_z using left-division (z \ ...) for numerical stability.
    # Right-division (/ z) inverts z which is ill-conditioned for large matrices.
    exp_z_minus_I = exp_z - I_mat
    φ₁ = z \ exp_z_minus_I
    φ₂ = z \ (φ₁ - I_mat)

    return exp_z, φ₁, φ₂
end

function _phi_functions_krylov(A::AbstractMatrix, krylov_dim::Int=30)
    """
    Krylov subspace approximation for φ functions using ExponentialUtilities.jl.

    Uses the phiv function which computes [φ₀(A)b, φ₁(A)b, ..., φₖ(A)b] efficiently
    via Krylov subspace methods (Arnoldi iteration).

    For matrix φ functions, we compute φₖ(A) by applying to identity vectors.
    """
    n = size(A, 1)
    I_mat = Matrix{eltype(A)}(LinearAlgebra.I, n, n)

    # Allocate result matrices
    exp_A = similar(I_mat)
    φ₁ = similar(I_mat)
    φ₂ = similar(I_mat)

    # Use ExponentialUtilities.phiv to compute φ functions column by column
    # phiv(t, A, b, k) returns [φ₀(tA)b, φ₁(tA)b, ..., φₖ(tA)b]
    # We use t=1 since A already contains the timestep scaling

    try
        for j in 1:n
            # Unit vector e_j
            e_j = zeros(eltype(A), n)
            e_j[j] = one(eltype(A))

            # Compute φ functions applied to e_j using Krylov methods
            # phiv returns a matrix where columns are φ₀(A)e_j, φ₁(A)e_j, φ₂(A)e_j
            phi_result = phiv(1.0, A, e_j, 2; m=min(krylov_dim, n))

            # Extract columns for each φ function
            exp_A[:, j] = phi_result[:, 1]  # φ₀(A)e_j = exp(A)e_j
            φ₁[:, j] = phi_result[:, 2]     # φ₁(A)e_j
            φ₂[:, j] = phi_result[:, 3]     # φ₂(A)e_j
        end

        return exp_A, φ₁, φ₂

    catch e
        @warn "Krylov φ computation failed: $e, falling back to direct method"
        # Fallback to direct computation
        try
            exp_A = exp(A)
            φ₁ = (exp_A - I_mat) * inv(A)
            φ₂ = (exp_A - I_mat - A) * inv(A^2)
            return exp_A, φ₁, φ₂
        catch e2
            error("All φ function computation methods failed for matrix of size $(size(A)) " *
                  "with norm $(norm(A)). Krylov error: $e, direct error: $e2. " *
                  "Consider reducing the timestep or using a different timestepper.")
        end
    end
end

function phiv_vector(t::Real, A::AbstractMatrix, b::AbstractVector, k::Int; m::Int=30)
    """
    Compute [φ₀(tA)b, φ₁(tA)b, ..., φₖ(tA)b] using Krylov subspace methods.

    This is a convenience wrapper around ExponentialUtilities.phiv for
    computing φ-function vector products efficiently.

    Arguments:
    - t: Time scaling factor
    - A: Matrix (typically the linear operator L)
    - b: Vector to apply φ functions to
    - k: Maximum φ index to compute (computes φ₀ through φₖ)
    - m: Krylov subspace dimension (default 30)

    Returns:
    - Matrix of size (n, k+1) where column j+1 contains φⱼ(tA)b
    """
    return phiv(t, A, b, k; m=min(m, length(b)))
end

function expv_krylov(t::Real, A::AbstractMatrix, b::AbstractVector; m::Int=30)
    """
    Compute exp(tA)b using Krylov subspace methods.

    More efficient than computing exp(tA) and then multiplying by b,
    especially for large sparse matrices.

    Arguments:
    - t: Time scaling factor
    - A: Matrix (typically the linear operator L)
    - b: Vector to apply exponential to
    - m: Krylov subspace dimension (default 30)

    Returns:
    - Vector exp(tA)b
    """
    return expv(t, A, b; m=min(m, length(b)))
end
