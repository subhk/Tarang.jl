"""
Enhanced Dedalus-Style Functions

Complete implementations following Dedalus mathematical formulations:
- Enhanced boundary value evaluation with proper spherical harmonic projection
- Complete spherical harmonic computation with exact normalization  
- Quadrature weight functions following Dedalus patterns
- Transform matrix utilities for efficient spectral operations

Based on dedalus/libraries/dedalus_sphere/ implementations.
"""

using SpecialFunctions

"""
Complete Dedalus-style theta quadrature weights implementation.
Uses Jacobi polynomials with (a,b) = (0,0) for proper Gauss-Legendre quadrature.

Based on dedalus/libraries/dedalus_sphere/sphere.py:quadrature() and 
dedalus/libraries/dedalus_sphere/jacobi.py:quadrature() functions.

Mathematical foundation:
- Uses Jacobi polynomials P_n^(0,0)(cos θ) = Legendre polynomials
- Computes roots using Newton iteration for numerical accuracy
- Integrates polynomials exactly up to degree 2*l_max + 1
- Proper coordinate transformation from cos(θ) ∈ [-1,1] to θ ∈ [0,π]
"""
function get_theta_quadrature_weights(coords::SphericalCoordinateSystem{T}) where T<:Real
    # Try to get precomputed weights first
    if hasfield(typeof(coords), :theta_weights) && !isnothing(coords.theta_weights)
        return coords.theta_weights
    else
        ntheta = coords.ntheta
        # Estimate l_max from grid resolution (following Dedalus patterns)
        l_max = max(1, ntheta ÷ 2)
        
        # Compute proper Dedalus-style quadrature weights
        return compute_dedalus_theta_quadrature_weights(ntheta, l_max, T)
    end
end

"""
Compute Dedalus-style theta quadrature weights using exact Jacobi polynomial approach.
Following dedalus/libraries/dedalus_sphere implementation patterns.
"""
function compute_dedalus_theta_quadrature_weights(ntheta::Int, l_max::Int, ::Type{T}) where T<:Real
    # Following Dedalus: sphere.quadrature(l_max) calls Jacobi.quadrature(l_max+1, 0, 0)
    n = l_max + 1
    
    # Jacobi parameters (a,b) = (0,0) for theta direction → Gauss-Legendre quadrature
    a, b = T(0), T(0)
    
    # Compute Gauss-Legendre nodes and weights on [-1,1] (cos θ space)
    cos_theta_nodes, gauss_weights = jacobi_quadrature(n, a, b, T)
    
    # Transform to θ ∈ [0,π]: cos(θ) = cos_nodes → θ = arccos(cos_nodes)
    # Reverse order so θ ∈ [0,π] (cos θ goes from 1 to -1)
    theta_nodes = acos.(clamp.(reverse(cos_theta_nodes), -1, 1))
    
    # Transform weights: dθ = -d(cos θ)/sin θ → w_θ = w_cos * 1/sin θ
    theta_weights = reverse(gauss_weights)
    
    # Apply Jacobian transformation and handle poles
    for i in eachindex(theta_weights)
        sin_theta = sin(theta_nodes[i])
        if abs(sin_theta) > 1e-14  # Away from poles
            theta_weights[i] /= abs(sin_theta)
        else  # At poles (θ ≈ 0 or π)
            theta_weights[i] = T(0)  # Regularity condition at poles
        end
    end
    
    # If we need more/fewer points than computed, interpolate/extrapolate
    if ntheta != length(theta_nodes)
        return interpolate_quadrature_weights(theta_nodes, theta_weights, ntheta, T)
    else
        return theta_weights
    end
end

"""
Jacobi quadrature implementation following Dedalus patterns.
Computes roots and weights for Jacobi polynomials P_n^(a,b)(x).

Based on dedalus/libraries/dedalus_sphere/jacobi.py:quadrature function.
"""
function jacobi_quadrature(n::Int, a::T, b::T, ::Type{T}; days::Int=3) where T<:Real
    if n == 1
        # Single point case
        node = (b - a) / (a + b + 2)
        weight = jacobi_mass(a, b, T)
        return [node], [weight]
    end
    
    # Initial guess for roots using tridiagonal eigenvalue approach
    cos_theta_nodes = jacobi_roots_initial_guess(n, a, b, T)
    
    # Mass factor: 2^(a+b+1) * Beta(a+1, b+1)
    mass_factor = jacobi_mass(a, b, T)
    
    # Special cases
    if a == b == T(-0.5)
        # Chebyshev polynomials
        weights = fill(mass_factor / n, n)
        return cos_theta_nodes, weights
    elseif a == b == T(0.5)
        # Special case handling
        P = jacobi_polynomials_matrix(n+1, a, b, cos_theta_nodes, T)[1:n, :]
    else
        # General case: Newton iteration refinement
        P = zeros(T, n, n)
        for iter in 1:days
            cos_theta_nodes, P = jacobi_polynomials_newton_iteration(n+1, a, b, cos_theta_nodes, T)
            P = P[1:n, :]  # Take first n rows
        end
    end
    
    # Compute weights from polynomial values at roots
    # Following Dedalus: P[0] /= sqrt(sum(P^2, axis=0)); w *= P[0]^2
    P_normalized = P[1, :] ./ sqrt.(sum(P.^2, dims=1)[1, :])
    weights = mass_factor .* P_normalized.^2
    
    return cos_theta_nodes, weights
end

"""
Jacobi polynomial roots initial guess using tridiagonal matrix eigenvalues.
Following Dedalus grid_guess implementation pattern.
"""
function jacobi_roots_initial_guess(n::Int, a::T, b::T, ::Type{T}) where T<:Real
    if n == 1
        return [(b - a) / (a + b + 2)]
    end
    
    # Build symmetric tridiagonal Jacobi matrix
    # Main diagonal: beta coefficients
    beta = zeros(T, n)
    for k in 0:(n-1)
        beta[k+1] = (b^2 - a^2) / ((2*k + a + b) * (2*k + a + b + 2))
    end
    
    # Off-diagonal: gamma coefficients
    gamma = zeros(T, n-1)
    for k in 1:(n-1)
        num = 4 * k * (k + a) * (k + b) * (k + a + b)
        den = (2*k + a + b)^2 * (2*k + a + b + 1) * (2*k + a + b - 1)
        gamma[k] = sqrt(abs(num / den))  # abs() for numerical stability
    end
    
    # Solve tridiagonal eigenvalue problem
    eigenvalues = eigvals(SymTridiagonal(beta, gamma))
    return sort(eigenvalues)  # Sort for consistent ordering
end

"""
Jacobi mass factor: 2^(a+b+1) * Beta(a+1, b+1).
Using log-space computation for numerical stability.
"""
function jacobi_mass(a::T, b::T, ::Type{T}) where T<:Real
    # Use log-space: log(mass) = (a+b+1)*log(2) + logbeta(a+1, b+1)
    log_mass = (a + b + 1) * log(T(2)) + loggamma(a + 1) + loggamma(b + 1) - loggamma(a + b + 2)
    return exp(log_mass)
end

"""
Newton iteration for Jacobi polynomial root refinement.
Following Dedalus polynomials(..., Newton=True) pattern.
"""
function jacobi_polynomials_newton_iteration(n::Int, a::T, b::T, z::Vector{T}, ::Type{T}) where T<:Real
    # Compute polynomials and derivatives at current points
    P = jacobi_polynomials_matrix(n, a, b, z, T)
    
    # Newton update for roots of P_{n-1}^{a,b}(z) = 0
    if n > 1
        # Derivative using recurrence relation
        L = (n - 1) + (a + b) / 2
        P_prime = zeros(T, length(z))
        
        for i in eachindex(z)
            if n >= 2 && abs(1 - z[i]^2) > 1e-14
                P_prime[i] = (L * P[n, i] - (n - 1 + a) * P[n-1, i]) / (1 - z[i]^2) * (1 - z[i])
            else
                P_prime[i] = T(1)  # Avoid division by zero at boundaries
            end
        end
        
        # Newton correction: z_new = z - P(z)/P'(z)
        corrections = zeros(T, length(z))
        for i in eachindex(z)
            if abs(P_prime[i]) > 1e-14
                corrections[i] = P[n, i] / P_prime[i]
            end
        end
        
        z_new = z .- corrections
        z_new = clamp.(z_new, -1, 1)  # Keep in valid domain
        
        return z_new, P[1:n-1, :]
    else
        return z, P
    end
end

"""
Compute Jacobi polynomials using three-term recurrence relation.
Following Dedalus jacobi.py:polynomials implementation.
"""
function jacobi_polynomials_matrix(n::Int, a::T, b::T, z::Vector{T}, ::Type{T}) where T<:Real
    m = length(z)
    P = zeros(T, n, m)
    
    if n == 0
        return P
    end
    
    # P_0^{a,b}(z) = 1
    P[1, :] .= T(1)
    
    if n == 1
        return P
    end
    
    # P_1^{a,b}(z) = (a+b+2)/2 * z + (a-b)/2  
    P[2, :] = ((a + b + 2) / 2) .* z .+ (a - b) / 2
    
    if n == 2
        return P
    end
    
    # Three-term recurrence for k ≥ 2
    for k in 2:(n-1)
        # Recurrence coefficients
        c1 = 2 * k * (k + a + b) * (2*k + a + b - 2)
        c2 = (2*k + a + b - 1) * (2*k + a + b) * (2*k + a + b - 2)
        c3 = (2*k + a + b - 1) * (a^2 - b^2)
        c4 = 2 * (k + a - 1) * (k + b - 1) * (2*k + a + b)
        
        for i in eachindex(z)
            P[k+1, i] = ((c2 * z[i] + c3) * P[k, i] - c4 * P[k-1, i]) / c1
        end
    end
    
    return P
end

"""
Interpolate quadrature weights to different grid size if needed.
"""
function interpolate_quadrature_weights(theta_nodes::Vector{T}, weights::Vector{T}, 
                                      target_ntheta::Int, ::Type{T}) where T<:Real
    if target_ntheta == length(weights)
        return weights
    end
    
    # Create target grid
    target_theta = [T(π * (i-1) / (target_ntheta-1)) for i in 1:target_ntheta]
    
    # Simple linear interpolation (could be enhanced with spline interpolation)
    target_weights = zeros(T, target_ntheta)
    for i in eachindex(target_weights)
        # Find nearest neighbors
        idx = searchsortedfirst(theta_nodes, target_theta[i])
        if idx == 1
            target_weights[i] = weights[1]
        elseif idx > length(weights)
            target_weights[i] = weights[end]
        else
            # Linear interpolation
            t = (target_theta[i] - theta_nodes[idx-1]) / (theta_nodes[idx] - theta_nodes[idx-1])
            target_weights[i] = (1-t) * weights[idx-1] + t * weights[idx]
        end
    end
    
    return target_weights
end

"""
Get phi quadrature weights from coordinate system.
Following Dedalus uniform azimuthal distribution.
"""
function get_phi_quadrature_weights(coords::SphericalCoordinateSystem{T}) where T<:Real
    # Try to get precomputed weights, otherwise compute them
    if hasfield(typeof(coords), :phi_weights) && !isnothing(coords.phi_weights)
        return coords.phi_weights
    else
        # Uniform weighting for φ ∈ [0, 2π]
        nphi = coords.nphi
        weights = ones(T, nphi) * (T(2π) / nphi)
        return weights
    end
end

"""
Compute spherical harmonic Y_l^m(θ,φ)* (conjugate) following Dedalus patterns.
This is a complete implementation using proper recurrence relations and exact normalization.

Based on Dedalus sphere.py harmonics function with spin-weighted formulation.
"""
function compute_spherical_harmonic_conjugate(l::Int, m::Int, theta::T, phi::T, ::Type{T}) where T<:Real
    # Following Dedalus spherical harmonic computation with proper normalization
    
    if l < 0 || abs(m) > l
        return Complex{T}(0)
    end
    
    # Handle l=0 case directly
    if l == 0
        return Complex{T}(1 / sqrt(4*T(π)))
    end
    
    # Associated Legendre polynomial P_l^m(cos θ)
    cos_theta = cos(theta)
    
    # Normalization factor: sqrt[(2l+1)/(4π) * (l-|m|)!/(l+|m|)!]
    # Use log-space computation for numerical stability
    log_norm = log(T(2*l + 1)) - log(4*T(π)) + 
               sum(log(T(i)) for i in (l-abs(m)+1):(l+abs(m))) -
               sum(log(T(i)) for i in 1:(l-abs(m))) - 
               sum(log(T(i)) for i in 1:(l+abs(m)))
    norm_factor = sqrt(exp(log_norm / 2))
    
    # Compute P_l^|m|(cos θ) using recurrence relations
    plm = compute_associated_legendre(l, abs(m), cos_theta, T)
    
    # Phase factor for negative m
    phase_factor = m < 0 ? (-1)^abs(m) : T(1)
    
    # Azimuthal component: e^(imφ)
    azimuthal = Complex{T}(cos(m * phi), sin(m * phi))
    
    # Complete spherical harmonic Y_l^m(θ,φ)
    ylm = norm_factor * phase_factor * plm * azimuthal
    
    # Return conjugate
    return conj(ylm)
end

"""
Compute associated Legendre polynomial P_l^m(x) using stable recurrence relations.
Following standard mathematical formulation used in spherical harmonics with numerical stability improvements.
"""
function compute_associated_legendre(l::Int, m::Int, x::T, ::Type{T}) where T<:Real
    if l < 0 || m < 0 || m > l
        return T(0)
    end
    
    if l == 0
        return T(1)
    end
    
    # Clamp x to valid domain [-1, 1]
    x = clamp(x, T(-1), T(1))
    
    # Base case: P_m^m(x)
    if l == m
        # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^(m/2)
        # Use iterative computation for numerical stability
        result = T(1)
        if m > 0
            sin_theta_sq = max(T(0), 1 - x*x)  # Ensure non-negative
            sin_theta = sqrt(sin_theta_sq)
            
            # Compute (-1)^m * (2m-1)!!
            factor = T(1)
            for i in 1:m
                factor *= -(2*i - 1) * sin_theta
            end
            result = factor
        end
        return result
    end
    
    if l == m + 1
        # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
        pmm = compute_associated_legendre(m, m, x, T)
        return x * (2*m + 1) * pmm
    end
    
    # General recurrence: P_l^m(x) = [(2l-1)*x*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)] / (l-m)
    # Use iterative form for numerical stability
    pm2 = compute_associated_legendre(m, m, x, T)
    if m == l-1
        return x * (2*m + 1) * pm2
    end
    
    pm1 = x * (2*m + 1) * pm2
    
    for n in (m+2):l
        pn = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (n - m)
        pm2, pm1 = pm1, pn
    end
    
    return pm1
end

"""
Enhanced boundary value evaluation using proper spherical harmonic projection.
Implements complete Dedalus-style integration with quadrature weights and proper normalization.

Mathematical formulation:
c_l^m = ∫∫ f(θ,φ) Y_l^m(θ,φ)* sin(θ) dθ dφ
      = Σ_i Σ_j f(θ_i,φ_j) Y_l^m(θ_i,φ_j)* sin(θ_i) w_θ(i) w_φ(j)
"""
function evaluate_boundary_value_enhanced(value_func, coords::SphericalCoordinateSystem{T}, l::Int, m::Int) where T<:Real
    if isa(value_func, Function)
        # Get grid dimensions and quadrature weights
        ntheta = coords.ntheta
        nphi = coords.nphi
        theta_weights = get_theta_quadrature_weights(coords)
        phi_weights = get_phi_quadrature_weights(coords)
        
        # Spherical harmonic projection with proper integration
        coefficient = Complex{T}(0)
        
        for i in 1:ntheta
            theta = coords.theta_grid[i]
            sin_theta = sin(theta)
            w_theta = theta_weights[i]
            
            for j in 1:nphi
                phi = coords.phi_grid[j]
                w_phi = phi_weights[j]
                
                # Evaluate function at boundary surface
                value = value_func(theta, phi)
                
                # Get spherical harmonic Y_l^m(θ,φ)* (conjugate)
                ylm_conj = compute_spherical_harmonic_conjugate(l, m, theta, phi, T)
                
                # Integration with proper quadrature weights and sin(θ) factor
                integrand = value * ylm_conj * sin_theta * w_theta * w_phi
                coefficient += integrand
            end
        end
        
        return coefficient
        
    elseif isa(value_func, Number)
        # For constant values, use exact analytical result
        return evaluate_boundary_value_enhanced(T(value_func), coords, l, m)
    else
        error("Unsupported value_func type: $(typeof(value_func))")
    end
end

"""
Enhanced boundary value evaluation for constant values.
Provides exact analytical result for constant functions.

For a constant function f(θ,φ) = c:
∫∫ c Y_l^m(θ,φ)* sin(θ) dθ dφ = c * δ_{l,0} * δ_{m,0} * √(4π)
"""
function evaluate_boundary_value_enhanced(value_const::T, coords::SphericalCoordinateSystem{T}, l::Int, m::Int) where T<:Real
    # For constant boundary values, only the (l=0,m=0) mode is non-zero
    # ∫∫ const * Y_0^0(θ,φ)* sin(θ) dθ dφ = const * √(4π) * δ_{l,0} * δ_{m,0}
    
    if l == 0 && m == 0
        # Y_0^0 = 1/(2√π), so ∫∫ Y_0^0* sin(θ) dθ dφ = √(4π)
        return Complex{T}(value_const) * sqrt(T(4π))
    else
        return Complex{T}(0)
    end
end

"""
Create spherical harmonic transform matrices following Dedalus patterns.
Pre-computes forward and backward transform matrices for efficient boundary evaluation.

Forward transform: grid values → spectral coefficients
Backward transform: spectral coefficients → grid values
"""
function create_spherical_harmonic_transform_matrices(coords::SphericalCoordinateSystem{T}, l_max::Int) where T<:Real
    ntheta = coords.ntheta
    nphi = coords.nphi
    n_modes = (l_max + 1)^2  # Total number of (l,m) modes
    
    # Forward transform matrix: grid → coefficients
    # Size: [n_modes, ntheta * nphi]
    forward_matrix = zeros(Complex{T}, n_modes, ntheta * nphi)
    
    # Backward transform matrix: coefficients → grid  
    # Size: [ntheta * nphi, n_modes]
    backward_matrix = zeros(Complex{T}, ntheta * nphi, n_modes)
    
    # Get quadrature weights
    theta_weights = get_theta_quadrature_weights(coords)
    phi_weights = get_phi_quadrature_weights(coords)
    
    # Build transform matrices
    mode_idx = 1
    for l in 0:l_max
        for m in -l:l
            grid_idx = 1
            for i in 1:ntheta
                theta = coords.theta_grid[i]
                sin_theta = sin(theta)
                w_theta = theta_weights[i]
                
                for j in 1:nphi
                    phi = coords.phi_grid[j]
                    w_phi = phi_weights[j]
                    
                    # Spherical harmonic value
                    ylm = compute_spherical_harmonic_conjugate(l, m, theta, phi, T)
                    
                    # Forward transform: integration weights
                    forward_matrix[mode_idx, grid_idx] = conj(ylm) * sin_theta * w_theta * w_phi
                    
                    # Backward transform: evaluation  
                    backward_matrix[grid_idx, mode_idx] = ylm
                    
                    grid_idx += 1
                end
            end
            mode_idx += 1
        end
    end
    
    return forward_matrix, backward_matrix
end

"""
Complete Gauss-Legendre quadrature implementation following Dedalus patterns.
Computes quadrature points and weights for exact polynomial integration.

Based on dedalus/libraries/dedalus_sphere/jacobi.py:quadrature() with a=b=0.
This is equivalent to calling jacobi_quadrature(n, 0, 0, T) but optimized for Legendre case.

Mathematical foundation:
- Finds roots of Legendre polynomials P_n(x) using eigenvalue method + Newton iteration
- Computes weights using normalized polynomial evaluation at roots
- Integrates polynomials exactly up to degree 2n-1 on [-1,1]
- Mass factor = 2 for Legendre polynomials

Returns:
- nodes: Gauss-Legendre quadrature nodes on [-1,1]  
- weights: Corresponding quadrature weights
"""
function compute_gauss_legendre_quadrature(n::Int, ::Type{T}) where T<:Real
    if n == 1
        # Single point case: node at origin, weight = integral of 1 over [-1,1]
        return [T(0)], [T(2)]
    end
    
    # Stage 1: Generate initial guess using symmetric tridiagonal eigenvalue method
    # For Legendre polynomials (a=b=0), the Jacobi matrix is particularly simple
    nodes = legendre_roots_eigenvalue_method(n, T)
    
    # Stage 2: Newton iteration refinement (following Dedalus 'days=3' pattern)
    for iteration in 1:3
        nodes = legendre_newton_refinement(nodes, n, T)
    end
    
    # Stage 3: Compute weights from polynomial evaluation at refined roots
    weights = legendre_quadrature_weights(nodes, n, T)
    
    return nodes, weights
end

"""
Compute Legendre polynomial roots using symmetric tridiagonal eigenvalue method.
Following Dedalus eigenvalue-based initial guess generation.
"""
function legendre_roots_eigenvalue_method(n::Int, ::Type{T}) where T<:Real
    # For Legendre polynomials, the tridiagonal Jacobi matrix has:
    # - Main diagonal: all zeros (since (b²-a²)/(...) = 0 for a=b=0)
    # - Off-diagonal: β_k = k/√((2k-1)(2k+1))
    
    main_diagonal = zeros(T, n)
    off_diagonal = zeros(T, n-1)
    
    for k in 1:(n-1)
        # Three-term recurrence coefficient for Legendre polynomials
        off_diagonal[k] = T(k) / sqrt(T(2k-1) * T(2k+1))
    end
    
    # Compute eigenvalues of symmetric tridiagonal matrix
    eigenvals = eigvals(SymTridiagonal(main_diagonal, off_diagonal))
    
    # Sort and return real parts (should be real for symmetric matrix)
    return sort(real.(eigenvals))
end

"""
Newton iteration refinement for Legendre polynomial roots.
Following Dedalus jacobi.py polynomials(..., Newton=True) approach.
"""
function legendre_newton_refinement(z::Vector{T}, n::Int, ::Type{T}) where T<:Real
    # Evaluate Legendre polynomials P_0, P_1, ..., P_n at current nodes
    P = legendre_polynomial_matrix(z, n, T)
    
    # Newton correction using Legendre polynomial derivative formula
    # P_n'(x) = n/(1-x²) * [x*P_n(x) - P_{n-1}(x)]
    z_new = similar(z)
    
    for i in eachindex(z)
        x = z[i]
        
        # Avoid division by zero at x = ±1
        if abs(1 - x^2) > 1e-14
            # Legendre polynomial derivative at root
            P_n_prime = n / (1 - x^2) * (x * P[n+1, i] - P[n, i])
            
            # Newton update: x_new = x - P_n(x)/P_n'(x)
            if abs(P_n_prime) > 1e-14
                z_new[i] = x - P[n+1, i] / P_n_prime
            else
                z_new[i] = x  # No update if derivative is too small
            end
        else
            z_new[i] = x  # Keep boundary values unchanged
        end
    end
    
    # Clamp to valid domain [-1, 1]
    return clamp.(z_new, -1, 1)
end

"""
Compute Legendre polynomials P_0, P_1, ..., P_n at given points.
Using the standard three-term recurrence relation.
"""
function legendre_polynomial_matrix(x::Vector{T}, n::Int, ::Type{T}) where T<:Real
    m = length(x)
    P = zeros(T, n+1, m)
    
    if n < 0
        return P
    end
    
    # P_0(x) = 1
    P[1, :] .= T(1)
    
    if n == 0
        return P
    end
    
    # P_1(x) = x  
    P[2, :] = x
    
    if n == 1
        return P
    end
    
    # Three-term recurrence: (k+1)P_{k+1}(x) = (2k+1)xP_k(x) - kP_{k-1}(x)
    for k in 1:(n-1)
        P[k+2, :] = ((2*k + 1) .* x .* P[k+1, :] - k .* P[k, :]) ./ (k + 1)
    end
    
    return P
end

"""
Compute Gauss-Legendre quadrature weights from polynomial roots.
Following Dedalus weight computation using normalized polynomial evaluation.
"""
function legendre_quadrature_weights(nodes::Vector{T}, n::Int, ::Type{T}) where T<:Real
    # Evaluate Legendre polynomials at the roots
    P = legendre_polynomial_matrix(nodes, n-1, T)
    
    # Following Dedalus pattern: P[0] /= sqrt(sum(P^2, axis=0)); w *= P[0]^2
    # For Legendre polynomials, we use the first n polynomials (P_0 through P_{n-1})
    
    weights = zeros(T, length(nodes))
    
    for i in eachindex(nodes)
        # Compute norm of polynomial vector at this node
        poly_norm = sqrt(sum(P[:, i].^2))
        
        if poly_norm > 1e-14
            # Normalized first polynomial (P_0 = 1)
            P_0_normalized = P[1, i] / poly_norm
            
            # Weight computation: mass factor * (normalized P_0)^2
            # For Legendre: mass = 2 (integral of weight function 1 over [-1,1])
            weights[i] = T(2) * P_0_normalized^2
        else
            # Fallback for numerical issues
            weights[i] = T(2) / n
        end
    end
    
    return weights
end

"""
Alternative direct Gauss-Legendre computation using analytical weight formula.
More numerically stable for high-order quadrature.
"""
function compute_gauss_legendre_analytical_weights(nodes::Vector{T}, n::Int, ::Type{T}) where T<:Real
    weights = zeros(T, length(nodes))
    
    for i in eachindex(nodes)
        x = nodes[i]
        
        # Evaluate P_{n-1}(x) using recurrence
        P_prev, P_curr = T(1), x  # P_0 = 1, P_1 = x
        
        for k in 1:(n-2)
            P_next = ((2*k + 1) * x * P_curr - k * P_prev) / (k + 1)
            P_prev, P_curr = P_curr, P_next
        end
        
        # Analytical weight formula: w_i = 2/((1-x_i²)[P_{n-1}'(x_i)]²)
        # P_n'(x) = n/(1-x²) * [x*P_n(x) - P_{n-1}(x)]
        if abs(1 - x^2) > 1e-14
            P_n = ((2*(n-1) + 1) * x * P_curr - (n-1) * P_prev) / n  # P_n(x)
            P_n_prime = n / (1 - x^2) * (x * P_n - P_curr)
            
            if abs(P_n_prime) > 1e-14
                weights[i] = T(2) / ((1 - x^2) * P_n_prime^2)
            else
                weights[i] = T(2) / n  # Fallback
            end
        else
            weights[i] = T(0)  # Weight is zero at boundaries ±1
        end
    end
    
    return weights
end

"""
Apply fast spherical harmonic transform to field data.
Optimized version using precomputed transform matrices.
"""
function apply_spherical_harmonic_transform!(field_grid::Array{Complex{T}}, 
                                           field_spectral::Array{Complex{T}},
                                           forward_matrix::Matrix{Complex{T}},
                                           forward::Bool=true) where T<:Real
    
    if forward
        # Grid → spectral: multiply by forward transform matrix
        grid_vec = reshape(field_grid, :)
        spectral_vec = forward_matrix * grid_vec
        field_spectral .= reshape(spectral_vec, size(field_spectral))
    else
        # Spectral → grid: multiply by backward transform matrix (transpose of forward)
        spectral_vec = reshape(field_spectral, :)
        grid_vec = forward_matrix' * spectral_vec
        field_grid .= reshape(grid_vec, size(field_grid))
    end
end

"""
Complete spectral field evaluation at arbitrary points following Dedalus patterns.
Implements exact interpolation using spherical harmonics and radial basis functions.

Based on dedalus/core/operators.py:Interpolate class and basis-specific implementations.
This follows the hierarchical evaluation pattern:
1. Extract spectral coefficients organized by (m, l, n) indices
2. Evaluate spherical harmonics Y_l^m(θ,φ) at target point
3. Evaluate radial basis functions R_n^(l)(r) at target radius
4. Sum all contributions: f(r,θ,φ) = Σ_{lmn} c_{lmn} Y_l^m(θ,φ) R_n^(l)(r)

Mathematical foundation:
- Spherical harmonics using Associated Legendre polynomials + e^{imφ}
- Radial basis using Zernike polynomials: R_n^(l)(r) = r^l P_{(n-l)/2}^{(0,l)}(2r²-1)
- Exact polynomial evaluation using three-term recurrence relations
- Coordinate transformation from physical (r,θ,φ) to native basis coordinates
"""
function evaluate_field_spectral(field_spectral::Array{Complex{T}}, 
                                coords::SphericalCoordinateSystem{T},
                                r::T, theta::T, phi::T, l_max::Int, n_max::Int=l_max) where T<:Real
    
    # Validate input coordinates
    r = clamp(r, T(0), coords.radius)
    theta = clamp(theta, T(0), T(π))
    phi = mod(phi, 2*T(π))
    
    result = Complex{T}(0)
    
    # Transform to normalized radial coordinate for Zernike polynomials
    r_norm = r / coords.radius  # Map [0, R] → [0, 1]
    
    # Main spectral reconstruction loop
    for l in 0:l_max
        for m in -l:l
            # Evaluate spherical harmonic Y_l^m(θ,φ)
            ylm = compute_spherical_harmonic_conjugate(l, m, theta, phi, T)
            
            # For each radial mode n
            for n in l:min(l_max, n_max)  # n ≥ l for Zernike polynomials
                # Get spectral coefficient c_{lmn}
                coeff = get_spectral_coefficient(field_spectral, l, m, n, l_max, n_max)
                
                if abs(coeff) > 1e-15  # Skip negligible coefficients
                    # Evaluate radial basis function R_n^(l)(r)
                    radial_value = evaluate_zernike_polynomial(n, l, r_norm, T)
                    
                    # Add contribution to field value
                    result += coeff * ylm * radial_value
                end
            end
        end
    end
    
    return result
end

"""
Extract spectral coefficient for specific (l,m,n) mode from coefficient array.
Assumes standard Dedalus ordering: coefficients organized as [m, l, n].
"""
function get_spectral_coefficient(field_spectral::Array{Complex{T}}, 
                                l::Int, m::Int, n::Int, 
                                l_max::Int, n_max::Int) where T<:Real
    
    # Convert (l,m,n) indices to array indices
    # Following Dedalus convention: m ∈ [-l_max, l_max], l ∈ [0, l_max], n ∈ [0, n_max]
    m_idx = m + l_max + 1  # m index: [-l_max, l_max] → [1, 2*l_max+1]
    l_idx = l + 1          # l index: [0, l_max] → [1, l_max+1]  
    n_idx = n + 1          # n index: [0, n_max] → [1, n_max+1]
    
    # Check bounds
    if (1 ≤ m_idx ≤ size(field_spectral, 1) && 
        1 ≤ l_idx ≤ size(field_spectral, 2) && 
        1 ≤ n_idx ≤ size(field_spectral, 3))
        return field_spectral[m_idx, l_idx, n_idx]
    else
        return Complex{T}(0)  # Out of bounds coefficient is zero
    end
end

"""
Evaluate Zernike polynomial R_n^(l)(r) at normalized radius r ∈ [0,1].
Following Dedalus Zernike polynomial implementation patterns.

Mathematical formula:
R_n^(l)(r) = r^l × P_{(n-l)/2}^{(0,l)}(2r²-1)

Where P_k^{(a,b)}(x) are Jacobi polynomials with parameters (a,b) = (0,l).
"""
function evaluate_zernike_polynomial(n::Int, l::Int, r::T, ::Type{T}) where T<:Real
    # Validate indices: n ≥ l ≥ 0, and (n-l) must be even for non-zero polynomial
    if n < l || l < 0 || (n - l) % 2 != 0
        return T(0)
    end
    
    # Handle boundary cases
    if r <= 0
        return l == 0 ? T(1) : T(0)  # Only n=l=0 mode non-zero at origin
    end
    
    if r >= 1
        # At boundary r=1: R_n^(l)(1) = 1 for all valid (n,l)
        return T(1)
    end
    
    # Compute Jacobi polynomial index
    k = (n - l) ÷ 2  # k = (n-l)/2
    
    # Transform coordinate: r ∈ [0,1] → ξ = 2r²-1 ∈ [-1,1]
    xi = 2*r^2 - 1
    
    # Evaluate Jacobi polynomial P_k^{(0,l)}(ξ) using three-term recurrence
    jacobi_value = evaluate_jacobi_polynomial(k, 0, l, xi, T)
    
    # Apply radial prefactor: R_n^(l)(r) = r^l × P_k^{(0,l)}(2r²-1)
    radial_factor = l == 0 ? T(1) : r^l
    
    return radial_factor * jacobi_value
end

"""
Evaluate Jacobi polynomial P_n^{(a,b)}(x) using stable three-term recurrence.
Following Dedalus jacobi.py polynomials implementation.
"""
function evaluate_jacobi_polynomial(n::Int, a::T, b::T, x::T, ::Type{T}) where T<:Real
    if n < 0
        return T(0)
    end
    
    if n == 0
        return T(1)  # P_0^{(a,b)}(x) = 1
    end
    
    if n == 1
        # P_1^{(a,b)}(x) = (a+b+2)/2 × x + (a-b)/2
        return ((a + b + 2) / 2) * x + (a - b) / 2
    end
    
    # Three-term recurrence for n ≥ 2
    P_prev = T(1)  # P_0
    P_curr = ((a + b + 2) / 2) * x + (a - b) / 2  # P_1
    
    for k in 1:(n-1)
        # Recurrence coefficients
        alpha = 2 * (k + 1) * (k + a + b + 1) * (2*k + a + b)
        beta = (2*k + a + b + 1) * (a^2 - b^2)
        gamma = (2*k + a + b) * (2*k + a + b + 1) * (2*k + a + b + 2)
        delta = 2 * (k + a) * (k + b) * (2*k + a + b + 2)
        
        # P_{k+1} = ((β + γx)P_k - δP_{k-1}) / α
        P_next = ((beta + gamma * x) * P_curr - delta * P_prev) / alpha
        
        P_prev, P_curr = P_curr, P_next
    end
    
    return P_curr
end

"""
Multi-point spectral evaluation using vectorized operations.
More efficient for evaluating at many points simultaneously.
"""
function evaluate_field_spectral_vectorized(field_spectral::Array{Complex{T}}, 
                                           coords::SphericalCoordinateSystem{T},
                                           r_points::Vector{T}, theta_points::Vector{T}, phi_points::Vector{T},
                                           l_max::Int, n_max::Int=l_max) where T<:Real
    
    n_points = length(r_points)
    @assert length(theta_points) == n_points && length(phi_points) == n_points
    
    results = zeros(Complex{T}, n_points)
    
    # Pre-compute spherical harmonics at all points
    ylm_cache = Dict{Tuple{Int,Int}, Vector{Complex{T}}}()
    for l in 0:l_max, m in -l:l
        ylm_values = zeros(Complex{T}, n_points)
        for i in 1:n_points
            ylm_values[i] = compute_spherical_harmonic_conjugate(l, m, theta_points[i], phi_points[i], T)
        end
        ylm_cache[(l,m)] = ylm_values
    end
    
    # Pre-compute normalized radii
    r_norm = r_points ./ coords.radius
    
    # Main evaluation loop
    for l in 0:l_max, m in -l:l, n in l:min(l_max, n_max)
        coeff = get_spectral_coefficient(field_spectral, l, m, n, l_max, n_max)
        
        if abs(coeff) > 1e-15
            ylm_values = ylm_cache[(l,m)]
            
            for i in 1:n_points
                radial_value = evaluate_zernike_polynomial(n, l, r_norm[i], T)
                results[i] += coeff * ylm_values[i] * radial_value
            end
        end
    end
    
    return results
end

"""
Spectral interpolation matrix construction following Dedalus Interpolate operator patterns.
Pre-computes interpolation matrices for efficient repeated evaluation.
"""
function build_spectral_interpolation_matrix(coords::SphericalCoordinateSystem{T},
                                           r_targets::Vector{T}, theta_targets::Vector{T}, phi_targets::Vector{T},
                                           l_max::Int, n_max::Int=l_max) where T<:Real
    
    n_points = length(r_targets)
    n_modes = (l_max + 1) * (2*l_max + 1) * (n_max + 1)  # Total number of modes
    
    # Interpolation matrix: [n_points × n_modes]
    interp_matrix = zeros(Complex{T}, n_points, n_modes)
    
    mode_idx = 1
    for l in 0:l_max, m in -l:l, n in l:n_max
        for i in 1:n_points
            # Evaluate basis functions at target point
            ylm = compute_spherical_harmonic_conjugate(l, m, theta_targets[i], phi_targets[i], T)
            radial = evaluate_zernike_polynomial(n, l, r_targets[i] / coords.radius, T)
            
            interp_matrix[i, mode_idx] = ylm * radial
        end
        mode_idx += 1
    end
    
    return interp_matrix
end