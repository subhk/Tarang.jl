# ============================================================================
# IMEX / SBDF Integration Support
# ============================================================================

"""
    IMEXFilterCoefficients{T}

Precomputed coefficients for IMEX time integration of temporal filters.

The filter equations can be written as:
    dy/dt = L·y + f(t)

where L is the linear operator (implicit part) and f is the forcing (explicit part).

For SBDF methods, the implicit solve becomes:
    (c₀·I - dt·L)·yⁿ⁺¹ = RHS

This struct stores the precomputed matrix (c₀·I - dt·L)⁻¹ for efficient updates.
"""
struct IMEXFilterCoefficients{T<:AbstractFloat}
    # For ExponentialMean: scalar coefficient
    # For Butterworth: 2×2 matrix coefficients
    exp_coeff::T                    # 1/(c₀ + α·dt) for exponential
    bw_M_inv::SMatrix{2, 2, T, 4}   # (c₀·I + α·dt·A)⁻¹ for Butterworth
    α::T
    dt::T
    scheme::Symbol                  # :SBDF1, :SBDF2, :SBDF3
end

"""
    precompute_imex_coefficients(filter::ExponentialMean, dt::Real; scheme::Symbol=:SBDF2)

Precompute IMEX coefficients for the exponential mean filter.

# SBDF Schemes
- `:SBDF1` (Backward Euler): c₀ = 1
- `:SBDF2`: c₀ = 3/2
- `:SBDF3`: c₀ = 11/6

# Returns
IMEXFilterCoefficients struct with precomputed solve coefficients.

# Example
```julia
filter = ExponentialMean((64, 64); α=0.5)
coeffs = precompute_imex_coefficients(filter, dt; scheme=:SBDF2)

# In time loop:
update_imex!(filter, h_current, h_prev, coeffs)
```
"""
function precompute_imex_coefficients(
    filter::ExponentialMean{T, N},
    dt::Real;
    scheme::Symbol = :SBDF2
) where {T, N}

    α = filter.α
    dt_T = T(dt)

    # SBDF coefficients for implicit term
    c0 = if scheme == :SBDF1
        one(T)
    elseif scheme == :SBDF2
        T(3) / T(2)
    elseif scheme == :SBDF3
        T(11) / T(6)
    else
        throw(ArgumentError("Unknown scheme: $scheme. Use :SBDF1, :SBDF2, or :SBDF3"))
    end

    # For exponential: dy/dt = -α·y + α·h
    # Implicit part: -α·y
    # (c₀ + α·dt)·yⁿ⁺¹ = RHS
    exp_coeff = one(T) / (c0 + α * dt_T)

    # Dummy Butterworth matrix (not used for ExponentialMean)
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    bw_M_inv = inv(c0 * I + α * dt_T * A)

    IMEXFilterCoefficients{T}(exp_coeff, bw_M_inv, α, dt_T, scheme)
end

"""
    precompute_imex_coefficients(filter::ButterworthFilter, dt::Real; scheme::Symbol=:SBDF2)

Precompute IMEX coefficients for the Butterworth filter.

The Butterworth filter has a 2×2 linear operator that is treated implicitly.
"""
function precompute_imex_coefficients(
    filter::ButterworthFilter{T, N},
    dt::Real;
    scheme::Symbol = :SBDF2
) where {T, N}

    α = filter.α
    dt_T = T(dt)

    c0 = if scheme == :SBDF1
        one(T)
    elseif scheme == :SBDF2
        T(3) / T(2)
    elseif scheme == :SBDF3
        T(11) / T(6)
    else
        throw(ArgumentError("Unknown scheme: $scheme. Use :SBDF1, :SBDF2, or :SBDF3"))
    end

    # Butterworth matrix A
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))

    # Implicit solve matrix: (c₀·I + α·dt·A)
    M = c0 * SMatrix{2, 2, T}(1, 0, 0, 1) + α * dt_T * A
    bw_M_inv = inv(M)

    exp_coeff = one(T) / (c0 + α * dt_T)  # Not used for Butterworth

    IMEXFilterCoefficients{T}(exp_coeff, bw_M_inv, α, dt_T, scheme)
end

"""
    update_imex!(filter::ExponentialMean, h_history::NTuple, coeffs::IMEXFilterCoefficients)

Update exponential mean filter using IMEX/SBDF time integration.

# Arguments
- `filter`: ExponentialMean filter to update
- `h_history`: Tuple of field histories (hⁿ,) for SBDF1, (hⁿ, hⁿ⁻¹) for SBDF2, etc.
- `coeffs`: Precomputed IMEX coefficients

# SBDF2 Formula
```
(3/2)h̄ⁿ⁺¹ + α·dt·h̄ⁿ⁺¹ = 2h̄ⁿ - (1/2)h̄ⁿ⁻¹ + α·dt·(2hⁿ - hⁿ⁻¹)
```

This is **unconditionally stable** - no timestep restriction from the filter!
Uses broadcasting for GPU compatibility.
"""
function update_imex!(
    filter::ExponentialMean{T, N, A},
    h_history::NTuple{1, AbstractArray{T, N}},  # SBDF1
    coeffs::IMEXFilterCoefficients{T}
) where {T, N, A}

    h = h_history[1]
    α = coeffs.α
    dt = coeffs.dt
    c = coeffs.exp_coeff  # 1/(1 + α·dt)
    h̄ = filter.h̄
    h̄_prev = filter.h̄_prev
    αdt = α * dt

    # Save current h̄ to history
    h̄_prev .= h̄

    # SBDF1: h̄ⁿ⁺¹ = (h̄ⁿ + α·dt·hⁿ) / (1 + α·dt) = c * (h̄_prev + αdt * h)
    @. h̄ = c * (h̄_prev + αdt * h)

    return h̄
end

function update_imex!(
    filter::ExponentialMean{T, N, A},
    h_history::NTuple{2, AbstractArray{T, N}},  # SBDF2
    coeffs::IMEXFilterCoefficients{T}
) where {T, N, A}

    h_n, h_nm1 = h_history
    α = coeffs.α
    dt = coeffs.dt
    c = coeffs.exp_coeff  # 1/(3/2 + α·dt)
    h̄ = filter.h̄
    h̄_prev = filter.h̄_prev

    two = T(2)
    half = T(0.5)
    αdt = α * dt

    # Save current h̄ for next step (need a copy since we'll overwrite h̄)
    h̄_curr = copy(h̄)

    # SBDF2: h̄ⁿ⁺¹ = c * (2h̄ⁿ - 0.5h̄ⁿ⁻¹ + αdt*(2hⁿ - hⁿ⁻¹))
    @. h̄ = c * (two * h̄_curr - half * h̄_prev + αdt * (two * h_n - h_nm1))

    # Update history
    h̄_prev .= h̄_curr

    return h̄
end

"""
    update_imex!(filter::ButterworthFilter, h_history::NTuple, coeffs::IMEXFilterCoefficients)

Update Butterworth filter using IMEX/SBDF time integration.

The 2×2 coupled system is solved implicitly, making this **unconditionally stable**.
Uses broadcasting for GPU compatibility.
"""
function update_imex!(
    filter::ButterworthFilter{T, N, Arr},
    h_history::NTuple{1, AbstractArray{T, N}},  # SBDF1
    coeffs::IMEXFilterCoefficients{T}
) where {T, N, Arr}

    h = h_history[1]
    α = coeffs.α
    dt = coeffs.dt
    M_inv = coeffs.bw_M_inv
    h̃ = filter.h̃
    h̄ = filter.h̄
    h̃_prev = filter.h̃_prev
    h̄_prev = filter.h̄_prev

    αdt = α * dt

    # Extract matrix elements for broadcasting
    M11, M12, M21, M22 = M_inv[1,1], M_inv[1,2], M_inv[2,1], M_inv[2,2]

    # Save current values for history
    h̃_prev .= h̃
    h̄_prev .= h̄

    # Compute RHS: rhs1 = h̃_prev + αdt * h, rhs2 = h̄_prev
    # Solve: h̃_new = M11*rhs1 + M12*rhs2, h̄_new = M21*rhs1 + M22*rhs2
    # Need to compute both before overwriting
    @. h̃ = M11 * (h̃_prev + αdt * h) + M12 * h̄_prev
    @. h̄ = M21 * (h̃_prev + αdt * h) + M22 * h̄_prev

    return h̄
end

function update_imex!(
    filter::ButterworthFilter{T, N, Arr},
    h_history::NTuple{2, AbstractArray{T, N}},  # SBDF2
    coeffs::IMEXFilterCoefficients{T}
) where {T, N, Arr}

    h_n, h_nm1 = h_history
    α = coeffs.α
    dt = coeffs.dt
    M_inv = coeffs.bw_M_inv
    h̃ = filter.h̃
    h̄ = filter.h̄
    h̃_prev = filter.h̃_prev
    h̄_prev = filter.h̄_prev

    two = T(2)
    half = T(0.5)
    αdt = α * dt

    # Extract matrix elements for broadcasting
    M11, M12, M21, M22 = M_inv[1,1], M_inv[1,2], M_inv[2,1], M_inv[2,2]

    # Save current values for history (need copy since we'll compute rhs from them)
    h̃_curr = copy(h̃)
    h̄_curr = copy(h̄)

    # Compute RHS vectors using broadcasting
    # rhs1 = 2*h̃_curr - 0.5*h̃_prev + αdt*(2*h_n - h_nm1)
    # rhs2 = 2*h̄_curr - 0.5*h̄_prev

    # For efficiency, compute in single fused broadcasts
    @. h̃ = M11 * (two * h̃_curr - half * h̃_prev + αdt * (two * h_n - h_nm1)) + M12 * (two * h̄_curr - half * h̄_prev)
    @. h̄ = M21 * (two * h̃_curr - half * h̃_prev + αdt * (two * h_n - h_nm1)) + M22 * (two * h̄_curr - half * h̄_prev)

    # Update history
    h̃_prev .= h̃_curr
    h̄_prev .= h̄_curr

    return h̄
end

"""
    linear_operator_coefficients(filter::ExponentialMean)

Return the linear operator coefficient for the filter equation.

For ExponentialMean: dy/dt = -α·y + α·h
Returns: -α (the coefficient of y in the implicit term)

This allows integration with general IMEX timestepping frameworks.
"""
linear_operator_coefficients(filter::ExponentialMean) = -filter.α

"""
    linear_operator_coefficients(filter::ButterworthFilter)

Return the linear operator matrix for the Butterworth filter.

For Butterworth: d/dt [h̃; h̄] = -α·A·[h̃; h̄] + α·[h; 0]
Returns: -α·A (the 2×2 matrix for the implicit term)
"""
function linear_operator_coefficients(filter::ButterworthFilter{T, N}) where {T, N}
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    return -filter.α * A
end


# ============================================================================
# Exponential Time Differencing (ETD) Support
# ============================================================================

"""
    ETDFilterCoefficients{T}

Precomputed coefficients for Exponential Time Differencing (ETD) integration.

For the ODE: dy/dt = L·y + f(t)
The exact solution is: y(t+dt) = exp(L·dt)·y(t) + ∫₀^dt exp(L·(dt-τ))·f(t+τ) dτ

ETD methods approximate the integral while treating exp(L·dt) exactly.
This provides **unconditional stability** for any timestep size.

# Fields
- `exp_scalar::T`: exp(-α·dt) for ExponentialMean
- `phi1_scalar::T`: φ₁(-α·dt)·dt = (1 - exp(-α·dt))/α for ExponentialMean
- `exp_matrix::SMatrix{2,2,T}`: exp(L·dt) for Butterworth
- `phi1_matrix::SMatrix{2,2,T}`: φ₁(L·dt)·dt for Butterworth
"""
struct ETDFilterCoefficients{T<:AbstractFloat}
    exp_scalar::T                       # exp(-α·dt)
    phi1_scalar::T                      # (1 - exp(-α·dt))/α = φ₁(-α·dt)·dt/α
    exp_matrix::SMatrix{2, 2, T, 4}     # exp(L·dt) for Butterworth
    phi1_matrix::SMatrix{2, 2, T, 4}    # φ₁(L·dt)·dt for Butterworth
    α::T
    dt::T
end

"""
    precompute_etd_coefficients(filter::ExponentialMean, dt::Real)

Precompute ETD coefficients for the exponential mean filter.

# ETD1 (Exponential Euler) Formula
For dh̄/dt = -α·h̄ + α·h:

    h̄ⁿ⁺¹ = exp(-α·dt)·h̄ⁿ + (1 - exp(-α·dt))·hⁿ

This is **exact** if h is constant over the timestep, and unconditionally stable!

# Example
```julia
filter = ExponentialMean((64, 64); α=0.5)
coeffs = precompute_etd_coefficients(filter, dt)

# In time loop - unconditionally stable for ANY dt!
update_etd!(filter, h, coeffs)
```
"""
function precompute_etd_coefficients(
    filter::ExponentialMean{T, N},
    dt::Real
) where {T, N}

    α = filter.α
    dt_T = T(dt)
    z = -α * dt_T  # z = L·dt for scalar case

    # exp(-α·dt)
    exp_scalar = exp(z)

    # φ₁(z)·dt = (exp(z) - 1)/z · dt = (exp(-α·dt) - 1)/(-α) = (1 - exp(-α·dt))/α
    # For numerical stability when z → 0, use: φ₁(z) = (exp(z)-1)/z ≈ 1 + z/2 + z²/6 + ...
    if abs(z) < 1e-4
        phi1_scalar = dt_T * (one(T) + z/2 + z^2/6 + z^3/24)
    else
        phi1_scalar = (exp_scalar - one(T)) / (-α)
    end

    # Dummy Butterworth matrices
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    L = -α * A
    Ldt = L * dt_T

    # Matrix exponential and φ₁ for Butterworth (computed even for ExponentialMean for type stability)
    exp_matrix = _matrix_exp_2x2(Ldt)
    phi1_matrix = _matrix_phi1_2x2(Ldt) * dt_T

    ETDFilterCoefficients{T}(exp_scalar, phi1_scalar, exp_matrix, phi1_matrix, α, dt_T)
end

"""
    precompute_etd_coefficients(filter::ButterworthFilter, dt::Real)

Precompute ETD coefficients for the Butterworth filter.

The 2×2 matrix exponential and φ₁ functions are computed exactly.
"""
function precompute_etd_coefficients(
    filter::ButterworthFilter{T, N},
    dt::Real
) where {T, N}

    α = filter.α
    dt_T = T(dt)

    # Scalar coefficients (for completeness)
    z = -α * dt_T
    exp_scalar = exp(z)
    phi1_scalar = abs(z) < 1e-4 ? dt_T * (one(T) + z/2) : (exp_scalar - one(T)) / (-α)

    # Butterworth linear operator L = -α·A
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    L = -α * A
    Ldt = L * dt_T

    # Matrix exponential and φ₁
    exp_matrix = _matrix_exp_2x2(Ldt)
    phi1_matrix = _matrix_phi1_2x2(Ldt) * dt_T

    ETDFilterCoefficients{T}(exp_scalar, phi1_scalar, exp_matrix, phi1_matrix, α, dt_T)
end

# Helper: 2×2 matrix exponential using eigendecomposition
function _matrix_exp_2x2(M::SMatrix{2, 2, T, 4}) where T
    # For a 2×2 matrix, use the formula based on trace and determinant
    # exp(M) = exp(tr(M)/2) * [cosh(Δ)·I + sinh(Δ)/Δ · (M - tr(M)/2·I)]
    # where Δ = sqrt((tr(M)/2)² - det(M))

    tr_M = M[1,1] + M[2,2]
    det_M = M[1,1]*M[2,2] - M[1,2]*M[2,1]

    half_tr = tr_M / 2
    discriminant = half_tr^2 - det_M

    exp_half_tr = exp(half_tr)

    if discriminant >= 0
        # Real eigenvalues
        Δ = sqrt(discriminant)
        if abs(Δ) < 1e-10
            # Repeated eigenvalue
            return exp_half_tr * (SMatrix{2,2,T}(1,0,0,1) + (M - half_tr * SMatrix{2,2,T}(1,0,0,1)))
        else
            cosh_Δ = cosh(Δ)
            sinh_Δ_over_Δ = sinh(Δ) / Δ
            M_shifted = M - half_tr * SMatrix{2,2,T}(1,0,0,1)
            return exp_half_tr * (cosh_Δ * SMatrix{2,2,T}(1,0,0,1) + sinh_Δ_over_Δ * M_shifted)
        end
    else
        # Complex eigenvalues (this is the Butterworth case!)
        ω = sqrt(-discriminant)
        cos_ω = cos(ω)
        sin_ω_over_ω = sin(ω) / ω
        M_shifted = M - half_tr * SMatrix{2,2,T}(1,0,0,1)
        return exp_half_tr * (cos_ω * SMatrix{2,2,T}(1,0,0,1) + sin_ω_over_ω * M_shifted)
    end
end

# Helper: 2×2 matrix φ₁ function: φ₁(M) = (exp(M) - I) * M⁻¹
function _matrix_phi1_2x2(M::SMatrix{2, 2, T, 4}) where T
    exp_M = _matrix_exp_2x2(M)
    I2 = SMatrix{2,2,T}(1,0,0,1)

    # φ₁(M) = (exp(M) - I) * M⁻¹
    # For numerical stability, check if M is nearly singular
    det_M = M[1,1]*M[2,2] - M[1,2]*M[2,1]

    if abs(det_M) < 1e-10
        # M nearly singular, use Taylor series: φ₁(M) ≈ I + M/2 + M²/6 + ...
        M2 = M * M
        return I2 + M/2 + M2/6
    else
        M_inv = inv(M)
        return (exp_M - I2) * M_inv
    end
end

"""
    update_etd!(filter::ExponentialMean, h::AbstractArray, coeffs::ETDFilterCoefficients)

Update exponential mean filter using ETD1 (Exponential Euler).

# Formula
    h̄ⁿ⁺¹ = exp(-α·dt)·h̄ⁿ + (1 - exp(-α·dt))·hⁿ

This is **unconditionally stable** for any timestep size!
Uses broadcasting for GPU compatibility.
"""
function update_etd!(
    filter::ExponentialMean{T, N, A},
    h::AbstractArray{T, N},
    coeffs::ETDFilterCoefficients{T}
) where {T, N, A}

    exp_factor = coeffs.exp_scalar
    phi1_factor = coeffs.phi1_scalar * coeffs.α  # (1 - exp(-α·dt))
    h̄ = filter.h̄
    h̄_prev = filter.h̄_prev

    # Save current h̄ to history
    h̄_prev .= h̄

    # ETD1: h̄ⁿ⁺¹ = exp_factor * h̄_prev + phi1_factor * h
    @. h̄ = exp_factor * h̄_prev + phi1_factor * h

    return h̄
end

"""
    update_etd!(filter::ButterworthFilter, h::AbstractArray, coeffs::ETDFilterCoefficients)

Update Butterworth filter using ETD1 (Exponential Euler).

# Formula
    [h̃; h̄]ⁿ⁺¹ = exp(L·dt)·[h̃; h̄]ⁿ + φ₁(L·dt)·dt·α·[hⁿ; 0]

The matrix exponential handles the complex eigenvalues exactly,
making this **unconditionally stable** for any timestep!
Uses broadcasting for GPU compatibility.
"""
function update_etd!(
    filter::ButterworthFilter{T, N, Arr},
    h::AbstractArray{T, N},
    coeffs::ETDFilterCoefficients{T}
) where {T, N, Arr}

    exp_M = coeffs.exp_matrix
    phi1_M = coeffs.phi1_matrix
    α = coeffs.α
    h̃ = filter.h̃
    h̄ = filter.h̄
    h̃_prev = filter.h̃_prev
    h̄_prev = filter.h̄_prev

    # Extract matrix elements for broadcasting
    E11, E12, E21, E22 = exp_M[1,1], exp_M[1,2], exp_M[2,1], exp_M[2,2]
    P11, P21 = phi1_M[1,1], phi1_M[2,1]  # Only need first column since f2 = 0

    # Save current values for history
    h̃_prev .= h̃
    h̄_prev .= h̄

    # yⁿ⁺¹ = exp(L·dt)·yⁿ + φ₁(L·dt)·dt·α·[h; 0]
    # h̃_new = E11*h̃ + E12*h̄ + P11*α*h
    # h̄_new = E21*h̃ + E22*h̄ + P21*α*h
    @. h̃ = E11 * h̃_prev + E12 * h̄_prev + P11 * α * h
    @. h̄ = E21 * h̃_prev + E22 * h̄_prev + P21 * α * h

    return h̄
end


