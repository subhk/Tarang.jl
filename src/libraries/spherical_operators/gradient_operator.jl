"""
Spherical Gradient Operator

Implements gradient operator ∇ in spherical coordinates for ball domains.
Combines radial and angular derivative operators with proper geometric factors.
"""

using LinearAlgebra
using SparseArrays

"""
Spherical Gradient Operator ∇

Implements gradient in spherical coordinates:
∇f = (∂f/∂r)êᵣ + (1/r)(∂f/∂θ)êθ + (1/(r sin θ))(∂f/∂φ)êφ
"""
struct SphericalGradient{T<:Real}
    coords::SphericalCoordinates{T}
    radial_deriv::RadialDerivativeOperator{T}
    angular_ops::Dict{Symbol, AngularMomentumOperator{T}}
    
    function SphericalGradient{T}(coords::SphericalCoordinates{T}) where T<:Real
        # Create component operators
        radial_deriv = RadialDerivativeOperator{T}(
            ZernikePolynomials{T}(coords.nr-1, coords.ntheta÷2), 1
        )
        
        harmonics = SphericalHarmonics{T}(coords.ntheta÷2, coords.ntheta, coords.nphi)
        angular_ops = Dict{Symbol, AngularMomentumOperator{T}}(
            :theta => AngularMomentumOperator{T}(harmonics, :L_y),
            :phi => AngularMomentumOperator{T}(harmonics, :L_z)
        )
        
        new{T}(coords, radial_deriv, angular_ops)
    end
end

"""
Apply gradient operator to scalar field.
"""
function apply_gradient!(grad_op::SphericalGradient{T}, input_scalar::Array{Complex{T},3}, 
                        output_vector::Array{Complex{T},4}) where T<:Real
    
    coords = grad_op.coords
    r_grid = coords.r_grid
    theta_grid = coords.theta_grid
    
    # ∇f = (∂f/∂r)êᵣ + (1/r)(∂f/∂θ)êθ + (1/(r sin θ))(∂f/∂φ)êφ
    
    # Radial component: ∂f/∂r
    apply_radial_derivative!(grad_op.radial_deriv, input_scalar, view(output_vector, 1, :, :, :))
    
    # Theta component: (1/r)(∂f/∂θ)
    apply_angular_derivative!(grad_op.angular_ops[:theta], input_scalar, view(output_vector, 2, :, :, :))
    for idx in CartesianIndices(size(input_scalar))
        r = r_grid[idx]
        if r > 1e-12
            output_vector[2, idx] /= r
        else
            output_vector[2, idx] = 0  # Regularity at center
        end
    end
    
    # Phi component: (1/(r sin θ))(∂f/∂φ)
    apply_azimuthal_derivative!(input_scalar, view(output_vector, 3, :, :, :))
    for idx in CartesianIndices(size(input_scalar))
        r = r_grid[idx]
        theta = theta_grid[idx]
        sin_theta = sin(theta)
        
        if r > 1e-12 && sin_theta > 1e-12
            output_vector[3, idx] /= (r * sin_theta)
        else
            output_vector[3, idx] = 0  # Regularity at center and poles
        end
    end
end

"""
Apply spherical gradient using spin-weighted approach.
Following Dedalus ball_wrapper.grad and SphericalGradient implementation.

Based on Dedalus patterns:
- Component 0 (minus): E_{ell-1} * out[0] = xi(-1, ell) * D- * input
- Component 1 (plus): E_{ell+1} * out[1] = xi(+1, ell) * D+ * input  
- Component 2 (radial): Standard radial derivative
"""
function apply_spherical_gradient_spin!(input::Array{Complex{T},3}, output::Array{Complex{T},4}, 
                                      coords::SphericalCoordinates{T}) where T<:Real
    
    n_phi, n_theta, n_r = size(input)
    l_max = n_theta ÷ 2
    
    # Initialize output
    fill!(output, zero(Complex{T}))
    
    # Apply gradient for each ell mode following Dedalus pattern
    for ell in 0:l_max
        apply_gradient_ell_mode_complete!(input, output, ell, coords, n_r, T)
    end
    
    # Apply regularity conditions at coordinate singularities
    apply_gradient_regularity_conditions!(output, coords)
end

"""
Apply gradient for single ell mode following Dedalus ball_wrapper.grad implementation.

Following the exact Dedalus pattern from ball_wrapper.py:
- Extract coefficient data for rank-0 (scalar) field
- Apply two gradient operations: D- and D+ with xi scaling
- Store results in proper regularity/spin components
"""
function apply_gradient_ell_mode_complete!(input::Array{Complex{T},3}, output::Array{Complex{T},4},
                                         ell::Int, coords::SphericalCoordinates{T}, n_r::Int, ::Type{T}) where T<:Real
    
    n_phi, n_theta, _ = size(input)
    
    # Extract input data for this ell mode (simplified indexing)
    # In practice, this would use proper spherical harmonic transforms
    mode_idx = min(ell + 1, n_theta)
    input_coeffs = zeros(Complex{T}, n_r)
    
    # For simplified demonstration, use first phi mode
    for r_idx in 1:n_r
        input_coeffs[r_idx] = input[1, mode_idx, r_idx]
    end
    
    # Apply Dedalus gradient pattern: two components with xi scaling
    
    # Component 0 (minus): if ell+tau_bar >= 1, apply D- with xi(-1) scaling
    if ell >= 1  # tau_bar = 0 for rank-0 scalar
        E_ell_minus1 = create_E_matrix(n_r, ell-1, T)
        D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
        xi_minus = xi_factor(-1, ell, T)
        
        rhs = xi_minus * (D_minus * input_coeffs)
        gradient_coeffs_minus = E_ell_minus1 \ rhs
        
        # Store result (component index 0 = minus)
        for r_idx in 1:n_r
            output[1, 1, mode_idx, r_idx] += gradient_coeffs_minus[r_idx]  # Simplified indexing
        end
    end
    
    # Component 1 (plus): if ell+tau_bar >= 0, apply D+ with xi(+1) scaling  
    if ell >= 0
        E_ell_plus1 = create_E_matrix(n_r, ell+1, T)
        D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
        xi_plus = xi_factor(1, ell, T)
        
        rhs = xi_plus * (D_plus * input_coeffs)
        gradient_coeffs_plus = E_ell_plus1 \ rhs
        
        # Store result (component index 1 = plus)
        for r_idx in 1:n_r
            output[2, 1, mode_idx, r_idx] += gradient_coeffs_plus[r_idx]  # Simplified indexing
        end
    end
    
    # Note: In Dedalus ball_wrapper.grad, only two operations are performed for rank-0 scalars
    # The radial component comes from the combination of these spin-weighted components
end

"""
Create gradient matrix transformation for unified operator application.
Following Dedalus approach for precomputed operator matrices.
"""
function create_gradient_matrix(n_r::Int, ell::Int, ::Type{T}) where T<:Real
    
    # Create unified gradient transformation matrix
    # Input: scalar field coefficients (n_r)
    # Output: vector field coefficients (3 × n_r for r, θ, φ components)
    
    total_size = 3 * n_r
    G_matrix = spzeros(Complex{T}, total_size, n_r)
    
    # Get E matrix and derivative operators
    E_matrix = create_E_matrix(n_r, ell, T)
    E_inv = inv(Matrix(E_matrix))  # For simplicity, use dense inverse
    
    # Radial component block (rows 1:n_r)
    D_radial = create_radial_operator_matrix(n_r, :radial_deriv, ell, T)
    G_matrix[1:n_r, 1:n_r] = E_inv * D_radial
    
    # Theta component block (rows n_r+1:2*n_r)
    if ell >= 1
        L_theta_matrix = spdiagm(0 => fill(sqrt(T(ell * (ell + 1))) / ell, n_r))
        G_matrix[(n_r+1):(2*n_r), 1:n_r] = L_theta_matrix
    end
    
    # Phi component block (rows 2*n_r+1:3*n_r)
    if ell >= 1
        L_phi_matrix = spdiagm(0 => fill(im * T(ell), n_r))
        G_matrix[(2*n_r+1):(3*n_r), 1:n_r] = L_phi_matrix
    end
    
    return G_matrix
end

"""
Apply regularity conditions for gradient operator at coordinate singularities.
"""
function apply_gradient_regularity_conditions!(output_vector::Array{Complex{T},4}, 
                                             coords::SphericalCoordinates{T}) where T<:Real
    
    n_comp, n_phi, n_theta, n_r = size(output_vector)
    r_grid = coords.r_grid
    theta_grid = coords.theta_grid
    
    # Regularity at center (r = 0)
    for j in 1:n_theta, i in 1:n_phi
        if r_grid[i, j, 1] < 1e-12
            for comp in 1:n_comp
                output_vector[comp, i, j, 1] = zero(Complex{T})
            end
        end
    end
    
    # Regularity at poles (θ = 0, π) 
    for k in 1:n_r, i in 1:n_phi
        # North pole (θ = 0)
        if abs(theta_grid[i, 1, k]) < 1e-12
            for comp in 1:n_comp
                output_vector[comp, i, 1, k] = zero(Complex{T})
            end
        end
        # South pole (θ = π)
        if abs(theta_grid[i, end, k] - π) < 1e-12
            for comp in 1:n_comp
                output_vector[comp, i, end, k] = zero(Complex{T})
            end
        end
    end
end

"""
Apply gradient with proper geometric scaling factors.
Includes 1/r and 1/(r sin θ) factors for angular components.
"""
function apply_gradient_with_geometry!(grad_op::SphericalGradient{T}, input_scalar::Array{Complex{T},3}, 
                                     output_vector::Array{Complex{T},4}) where T<:Real
    
    # First apply basic gradient
    apply_gradient!(grad_op, input_scalar, output_vector)
    
    # Then apply geometric scaling factors
    coords = grad_op.coords
    r_grid = coords.r_grid
    theta_grid = coords.theta_grid
    
    # Scale theta component by 1/r
    for idx in CartesianIndices(size(input_scalar))
        r = r_grid[idx]
        if r > 1e-12
            output_vector[2, idx] /= r
        else
            output_vector[2, idx] = zero(Complex{T})
        end
    end
    
    # Scale phi component by 1/(r sin θ)
    for idx in CartesianIndices(size(input_scalar))
        r = r_grid[idx]
        theta = theta_grid[idx]
        sin_theta = sin(theta)
        
        if r > 1e-12 && sin_theta > 1e-12
            output_vector[3, idx] /= (r * sin_theta)
        else
            output_vector[3, idx] = zero(Complex{T})
        end
    end
end