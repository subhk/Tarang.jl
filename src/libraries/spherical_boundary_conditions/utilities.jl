"""
Utility functions for spherical boundary conditions.

Contains Zernike polynomial evaluation, spherical harmonics computation,
and other mathematical utilities following dedalus approach.
"""

using LinearAlgebra

export evaluate_zernike_at_center, evaluate_zernike_at_boundary
export evaluate_zernike_derivative_at_boundary, compute_zernike_radial, compute_zernike_radial_derivative
export evaluate_spin_weighted_harmonic_at_pole

"""
    evaluate_zernike_at_center(n, l, T) -> T

Evaluate Zernike polynomial Z_n^l at center r=0.

Following dedalus approach, only l=0 modes with even n contribute at center:
- l > 0: returns 0 (due to r^l factor)
- l = 0, odd n: returns 0 
- l = 0, even n: returns (-1)^(n/2) from Jacobi polynomial P_s^(0,0)(-1) where s = n/2

# Mathematical Background
At r=0, Z_n^l(r=0) = δ_{l,0} * P_{n/2}^{(0,0)}(-1) * δ_{n even}
where P_s^{(0,0)}(-1) = (-1)^s gives the alternating pattern.
"""
function evaluate_zernike_at_center(n::Int, l::Int, ::Type{T}) where T<:Real
    if l > 0
        return T(0)  # r^l factor makes this vanish at r=0
    elseif l == 0
        if n % 2 != 0
            return T(0)  # Only even n contribute for l=0
        else
            s = n ÷ 2  # n = 2s for even n
            # Jacobi polynomial P_s^{(0,0)}(-1) = (-1)^s
            return T((-1)^s)
        end
    else
        return T(0)  # Invalid case
    end
end

"""
    evaluate_zernike_at_boundary(n, l, T) -> T

Evaluate Zernike polynomial Z_n^l at boundary r=1.

At the boundary r=1, Z_n^l(1) = 1 for all valid (n,l) pairs
since the radial part evaluates to unity.
"""
function evaluate_zernike_at_boundary(n::Int, l::Int, ::Type{T}) where T<:Real
    # At r=1, all Zernike polynomials evaluate to 1
    # This comes from the normalization Z_n^l(r=1) = 1
    return T(1)
end

"""
    evaluate_zernike_derivative_at_boundary(n, deriv_order, T) -> T

Evaluate derivatives of Zernike polynomials at boundary r=1.

For the radial derivative ∂Z_n^l/∂r|_{r=1}, we use the relationship:
∂Z_n^l/∂r = (n + l + 1) * Z_{n-1}^{l+1} for appropriate n,l

Higher order derivatives follow from recursive application.

# Arguments  
- `n::Int`: Radial mode number
- `deriv_order::Int`: Order of derivative (1 for first derivative, etc.)
- `T::Type`: Numeric type

# Returns
Value of the derivative at r=1
"""
function evaluate_zernike_derivative_at_boundary(n::Int, deriv_order::Int, ::Type{T}) where T<:Real
    # Exact Dedalus formulation for ∂^k Z_n^l/∂r^k|_{r=1}
    # Based on Jacobi polynomial derivative theory with scaling factor (2/radius)
    
    if deriv_order < 0 || n < 0
        return T(0)
    end
    
    if deriv_order == 0
        # Function value at boundary: Z_n^l(1) = 1 for all valid (n,l)
        return T(1)
    elseif deriv_order == 1
        # First derivative: ∂Z_n^l/∂r|_{r=1}
        # Special case: constant polynomial (n=0) has zero derivative
        if n == 0
            return T(0)
        else
            # General case: ∂Z_n^l/∂r|_{r=1} = 2n + 1 for n≥1
            # Exact Dedalus formula from D operator scaling
            return T(2*n + 1)
        end
    elseif deriv_order == 2
        # Second derivative: ∂²Z_n^l/∂r²|_{r=1}
        # Special cases: constant (n=0) and linear (n=1) polynomials
        if n <= 1
            return T(0)  # Constants and linears have zero second derivative
        else
            # General case: ∂²Z_n^l/∂r²|_{r=1} = 2n(n+1) for n≥2
            # From progressive application of D operators
            return T(2*n*(n+1))
        end
    else
        # Higher-order derivatives: ∂^k Z_n^l/∂r^k|_{r=1} = ∏_{i=0}^{k-1} 2(n-i)
        # Following Dedalus Jacobi polynomial derivative chain rule
        
        if deriv_order > n
            return T(0)  # Derivative order exceeds polynomial degree
        end
        
        result = T(1)
        for i in 0:(deriv_order-1)
            result *= T(2*(n - i))
        end
        return result
    end
end

"""
    compute_zernike_radial(n, l, r, T) -> T

Compute Zernike radial polynomial Z_n^l(r) at arbitrary radius r.

Uses the Jacobi polynomial representation:
Z_n^l(r) = r^l * P_{(n-l)/2}^{(0,l)}(2r²-1)

where P_k^{(α,β)} are Jacobi polynomials.

# Mathematical Details
The Zernike polynomials are orthogonal on the unit disk and satisfy:
∫₀¹ Z_n^l(r) Z_m^l(r) r dr = δ_{nm} / (2(n+1))

# Arguments
- `n::Int`: Radial degree (n ≥ l, n-l even)
- `l::Int`: Angular frequency (l ≥ 0)  
- `r::T`: Radial coordinate (0 ≤ r ≤ 1)
- `T::Type`: Numeric precision type

# Returns
Value of Z_n^l(r)
"""
function compute_zernike_radial(n::Int, l::Int, r::T, ::Type{T}) where T<:Real
    # Input validation
    if n < l || (n - l) % 2 != 0
        return T(0)  # Invalid (n,l) combination
    end
    
    if abs(r) > 1 + 1e-12  # Allow for small numerical errors
        return T(0)  # Outside unit ball
    end
    
    # Handle boundary cases
    if r == 0
        return evaluate_zernike_at_center(n, l, T)
    end
    
    # Compute s = (n-l)/2
    s = (n - l) ÷ 2
    
    # Z_n^l(r) = r^l * P_s^{(0,l)}(2r²-1)
    r_power = r^l
    xi = 2 * r^2 - 1  # Transform to Jacobi polynomial domain [-1,1]
    
    # Compute Jacobi polynomial P_s^{(0,l)}(xi)
    # Using recurrence relation for efficiency
    jacobi_val = compute_jacobi_polynomial(s, 0, l, xi, T)
    
    return r_power * jacobi_val
end

"""
    compute_zernike_radial_derivative(n, l, r, T) -> T

Compute radial derivative ∂Z_n^l/∂r at radius r.

Using the chain rule:
∂Z_n^l/∂r = l*r^{l-1}*P_s^{(0,l)}(ξ) + r^l * ∂P_s^{(0,l)}/∂ξ * ∂ξ/∂r

where ξ = 2r²-1 and ∂ξ/∂r = 4r.
"""
function compute_zernike_radial_derivative(n::Int, l::Int, r::T, ::Type{T}) where T<:Real
    if n < l || (n - l) % 2 != 0
        return T(0)
    end
    
    if abs(r) > 1 + 1e-12
        return T(0)
    end
    
    # Handle r=0 case
    if r == 0
        if l == 1
            # Only l=1 terms have non-zero derivative at r=0
            s = (n - 1) ÷ 2
            return compute_jacobi_polynomial(s, 0, 1, -1, T)
        else
            return T(0)
        end
    end
    
    s = (n - l) ÷ 2
    xi = 2 * r^2 - 1
    
    # Compute P_s^{(0,l)}(ξ) and its derivative
    jacobi_val = compute_jacobi_polynomial(s, 0, l, xi, T)
    jacobi_deriv = compute_jacobi_polynomial_derivative(s, 0, l, xi, T)
    
    # Apply chain rule
    term1 = l * r^(l-1) * jacobi_val
    term2 = r^l * jacobi_deriv * 4 * r
    
    return term1 + term2
end

"""
    compute_jacobi_polynomial(n, alpha, beta, x, T) -> T

Compute Jacobi polynomial P_n^{(α,β)}(x) using three-term recurrence.

The Jacobi polynomials satisfy:
P_0^{(α,β)}(x) = 1
P_1^{(α,β)}(x) = (α - β + (α + β + 2)x)/2
And the recurrence relation for n ≥ 2.
"""
function compute_jacobi_polynomial(n::Int, alpha::Int, beta::Int, x::T, ::Type{T}) where T<:Real
    if n == 0
        return T(1)
    elseif n == 1
        return T((alpha - beta + (alpha + beta + 2) * x) / 2)
    end
    
    # Use three-term recurrence relation
    p0 = T(1)
    p1 = T((alpha - beta + (alpha + beta + 2) * x) / 2)
    
    for k in 2:n
        a1 = 2 * k * (k + alpha + beta) * (2 * k + alpha + beta - 2)
        a2 = (2 * k + alpha + beta - 1) * (alpha^2 - beta^2)
        a3 = (2 * k + alpha + beta - 1) * (2 * k + alpha + beta) * (2 * k + alpha + beta - 2)
        a4 = 2 * (k + alpha - 1) * (k + beta - 1) * (2 * k + alpha + beta)
        
        pk = ((a2 + a3 * x) * p1 - a4 * p0) / a1
        p0, p1 = p1, pk
    end
    
    return p1
end

"""
    compute_jacobi_polynomial_derivative(n, alpha, beta, x, T) -> T

Compute derivative of Jacobi polynomial dP_n^{(α,β)}/dx.

Uses the identity:
dP_n^{(α,β)}/dx = (n + α + β + 1)/2 * P_{n-1}^{(α+1,β+1)}(x)
"""
function compute_jacobi_polynomial_derivative(n::Int, alpha::Int, beta::Int, x::T, ::Type{T}) where T<:Real
    if n == 0
        return T(0)
    end
    
    coeff = T((n + alpha + beta + 1) / 2)
    return coeff * compute_jacobi_polynomial(n - 1, alpha + 1, beta + 1, x, T)
end

"""
    evaluate_spin_weighted_harmonic_at_pole(l, m, s, is_north_pole, T) -> Complex{T}

Evaluate spin-weighted spherical harmonic _{s}Y_l^m at pole following dedalus approach.

At the poles θ=0,π, only certain modes contribute:
- North pole (θ=0): only m=±s modes are non-zero
- South pole (θ=π): phase factor (-1)^{l-s} applies

# Mathematical Background
The spin-weighted spherical harmonics have the form:
_{s}Y_l^m(θ,φ) = √[(2l+1)/(4π)] d^l_{m,s}(θ) e^{imφ}

where d^l_{m,s}(θ) are Wigner d-functions.

At poles:
- θ=0: d^l_{m,s}(0) ≠ 0 only if m = ±s
- θ=π: d^l_{m,s}(π) = (-1)^{l-s} d^l_{m,s}(0)

# Arguments
- `l::Int`: Harmonic degree (l ≥ max(|m|,|s|))
- `m::Int`: Azimuthal order (-l ≤ m ≤ l)
- `s::Int`: Spin weight
- `is_north_pole::Bool`: true for θ=0, false for θ=π
- `T::Type`: Real number type

# Returns
Complex{T}: Value of _{s}Y_l^m at the pole
"""
function evaluate_spin_weighted_harmonic_at_pole(l::Int, m::Int, s::Int, is_north_pole::Bool, ::Type{T}) where T<:Real
    # Validity check
    if l < max(abs(m), abs(s))
        return Complex{T}(0)
    end
    
    # At poles, only m = ±s modes contribute
    if m != s && m != -s
        return Complex{T}(0)
    end
    
    # Normalization factor: √[(2l+1)/(4π)]
    norm_factor = sqrt(T(2*l + 1) / (4 * π))
    
    # Wigner d-function at poles
    if m == s
        # d^l_{s,s}(θ) at poles
        d_value = if is_north_pole
            # At θ=0: d^l_{s,s}(0) = δ_{s,0} for standard convention
            s == 0 ? T(1) : T(0)
        else
            # At θ=π: d^l_{s,s}(π) = (-1)^{l-s} d^l_{s,s}(0)
            s == 0 ? T((-1)^l) : T(0)
        end
    elseif m == -s
        # d^l_{-s,s}(θ) at poles  
        d_value = if is_north_pole
            s == 0 ? T(1) : T(0)  # Only s=0 contributes at north pole for m=-s
        else
            s == 0 ? T((-1)^l) : T(0)  # Phase factor at south pole
        end
    else
        d_value = T(0)
    end
    
    return Complex{T}(norm_factor * d_value)
end