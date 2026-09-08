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
    exp_z = exp(z)
    φ₀ = exp_z
    # The direct formulas cancel catastrophically well above roundoff: with a
    # 1e-8 cutoff, φ₂'s relative error reaches ~1% at |z|=1e-7 and φ₃'s exceeds
    # O(1) up to |z|~1e-5. At the 1e-2 crossover the series truncation (≲1e-13
    # relative) meets the direct formulas' cancellation error (≲1e-10).
    if abs(z) < 1e-2
        φ₁ = 1 + z/2 + z^2/6 + z^3/24 + z^4/120 + z^5/720
        φ₂ = 1/2 + z/6 + z^2/24 + z^3/120 + z^4/720 + z^5/5040
        φ₃ = 1/6 + z/24 + z^2/120 + z^3/720 + z^4/5040 + z^5/40320
    else
        φ₁ = (exp_z - 1) / z
        φ₂ = (exp_z - 1 - z) / z^2
        φ₃ = (exp_z - 1 - z - z^2/2) / z^3
    end
    return φ₀, φ₁, φ₂, φ₃
end

# Cached structured identities for phi_functions_matrix. `Diagonal` keeps the
# retained storage O(n) rather than pinning an additional dense O(n²) matrix for
# every ETD problem size; arithmetic with the dense z/exp(z) operands still
# materializes the same dense results where required.
const _phi_identity_cache = Dict{Tuple{Int, DataType}, Diagonal}()
const _phi_identity_lock = ReentrantLock()
const _PHI_IDENTITY_CACHE_MAX_SIZE = 16

@inline function _evict_phi_identity_cache_entry!()
    isempty(_phi_identity_cache) && return nothing
    delete!(_phi_identity_cache, first(keys(_phi_identity_cache)))
    return nothing
end

@inline function _get_identity_matrix(n::Int, T::Type)
    lock(_phi_identity_lock) do
        get!(_phi_identity_cache, (n, T)) do
            length(_phi_identity_cache) >= _PHI_IDENTITY_CACHE_MAX_SIZE &&
                _evict_phi_identity_cache_entry!()
            Diagonal(fill(one(T), n))
        end
    end
end

"""Copy `A` to one dense matrix and scale it in place."""
@inline function _scaled_dense_operator(A::AbstractMatrix, dt::Float64)
    T = promote_type(eltype(A), typeof(dt))
    z = Matrix{T}(A)
    rmul!(z, dt)
    return z
end

"""Compute matrix φ functions for exponential integrators.

The full matrix functions use inverse-free scaling and squaring. In particular,
zero eigenvalues and Jordan blocks require no special eigendecomposition: every
φ function is entire, even when the operator is singular or not diagonalizable.
"""
function phi_functions_matrix(A::AbstractMatrix, dt::Float64)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("ETD matrix operator must be square"))

    # Full matrix functions retain dense O(n²) storage and cost O(n³).
    # Check the original dimension before allocating any dense work buffers.
    if n > 4096
        throw(ArgumentError(
            "ETD matrix exponential requires dense O(n²) storage but n=$n is too large. " *
            "Use RK222/SBDF2 for this problem size, or reduce resolution."))
    end
    return _phi_functions_scaling_squaring(_scaled_dense_operator(A, dt))
end

"""
    _phi_functions_scaling_squaring(z) -> (exp_z, φ₁, φ₂)

Evaluate the convergent Taylor series for φ₂ on `B = z/2^s`, with
`opnorm(B, Inf) <= 1/2`. Recover φ₁(B) = I + B*φ₂(B) and exp(B) = I + B*φ₁(B),
then undo the scaling using the exact doubling identities

    φ₂(2B) = (exp(B)*φ₂(B) + φ₂(B) + φ₁(B))/4
    φ₁(2B) = (exp(B)*φ₁(B) + φ₁(B))/2
    exp(2B) = exp(B)*exp(B).

These are polynomial/entire-function identities and remain valid for singular,
non-diagonalizable matrices. No division by z or eigenvector inverse is used.
The work buffers remain n×n, avoiding the ninefold storage increase of a 3n×3n
block exponential and the columnwise matrix exponentials used by dense `phi`.
"""
function _phi_functions_scaling_squaring(z::AbstractMatrix)
    n = size(z, 1)
    n == size(z, 2) || throw(DimensionMismatch("ETD matrix operator must be square"))
    z_norm = opnorm(z, Inf)
    isfinite(z_norm) || throw(ArgumentError("ETD matrix operator must have finite norm"))
    s = z_norm <= 0.5 ? 0 : max(0, ceil(Int, log2(z_norm)) + 1)
    real_type = typeof(real(one(eltype(z))))
    # Form the small scale directly: computing 2^s can overflow for finite z.
    scaled = z .* (real_type(2)^(-s))
    identity = _get_identity_matrix(n, eltype(z))

    φ₂ = Matrix(identity / 2)
    term = copy(φ₂)
    next_term = similar(φ₂)
    tolerance = eps(real(one(eltype(z))))
    converged = false
    for k in 1:256
        mul!(next_term, scaled, term)
        rmul!(next_term, inv(real_type(k + 2)))
        φ₂ .+= next_term
        term, next_term = next_term, term
        if opnorm(term, Inf) <= tolerance * opnorm(φ₂, Inf)
            converged = true
            break
        end
    end
    converged || error("ETD matrix φ₂ Taylor series did not converge after scaling")

    # Reuse the Taylor term buffers for the other two retained functions.
    φ₁ = next_term
    mul!(φ₁, scaled, φ₂)
    exp_z = term
    for i in 1:n
        φ₁[i, i] += one(eltype(z))
    end
    mul!(exp_z, scaled, φ₁)
    for i in 1:n
        exp_z[i, i] += one(eltype(z))
    end

    if s > 0
        workspace = similar(exp_z)
        for _ in 1:s
            # φ₂ must be updated before φ₁, and both before exp, so each
            # identity reads the three functions at the same scale.
            mul!(workspace, exp_z, φ₂)
            @. workspace = (workspace + φ₂ + φ₁) / 4
            φ₂, workspace = workspace, φ₂
            mul!(workspace, exp_z, φ₁)
            @. workspace = (workspace + φ₁) / 2
            φ₁, workspace = workspace, φ₁
            mul!(workspace, exp_z, exp_z)
            exp_z, workspace = workspace, exp_z
        end
    end
    return exp_z, φ₁, φ₂
end

# Retain the existing internal entry points for callers/tests, but use the
# inverse-free evaluator throughout. The former eigen fallback discarded Jordan
# couplings; the division-based phi and Padé helpers failed on singular z.
_compute_phi1_stable(z, exp_z, identity) = _phi_functions_scaling_squaring(z)[2]
_compute_phi2_stable(z, exp_z, identity, φ₁) = _phi_functions_scaling_squaring(z)[3]
_phi_via_eigen(z::AbstractMatrix, identity) = _phi_functions_scaling_squaring(z)
_phi_functions_pade(z) = _phi_functions_scaling_squaring(z)

"""
    Krylov subspace approximation for φ functions using ExponentialUtilities.jl.

    Uses the phiv function which computes [φ₀(A)b, φ₁(A)b, ..., φₖ(A)b] efficiently
    via Krylov subspace methods (Arnoldi iteration).

    For matrix φ functions, we compute φₖ(A) by applying to identity vectors.
    """
function _phi_functions_krylov(A::AbstractMatrix, krylov_dim::Int=30)
    n = size(A, 1)
    T = eltype(A)

    # Allocate result matrices
    exp_A = Matrix{T}(undef, n, n)
    φ₁ = Matrix{T}(undef, n, n)
    φ₂ = Matrix{T}(undef, n, n)

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
        # The direct fallback must also handle singular/Jordan operators.
        try
            return _phi_functions_scaling_squaring(A)
        catch e2
            error("All φ function computation methods failed for matrix of size $(size(A)) " *
                  "with norm $(norm(A)). Krylov error: $e, direct error: $e2. " *
                  "Consider reducing the timestep or using a different timestepper.")
        end
    end
end

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
function phiv_vector(t::Real, A::AbstractMatrix, b::AbstractVector, k::Int; m::Int=30)
    return phiv(t, A, b, k; m=min(m, length(b)))
end

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
function expv_krylov(t::Real, A::AbstractMatrix, b::AbstractVector; m::Int=30)
    return expv(t, A, b; m=min(m, length(b)))
end
