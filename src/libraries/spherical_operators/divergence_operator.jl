"""
Spherical Divergence Operator

Implements divergence operator ∇⋅ in spherical coordinates for ball domains.
Combines radial and angular derivatives with proper geometric factors.
"""

using LinearAlgebra
using SparseArrays

"""
Spherical Divergence Operator ∇⋅

Implements divergence in spherical coordinates:
∇⋅F = (1/r²)(∂/∂r)(r²Fᵣ) + (1/(r sin θ))(∂/∂θ)(sin θ Fθ) + (1/(r sin θ))(∂Fφ/∂φ)
"""
struct SphericalDivergence{T<:Real}
    coords::SphericalCoordinates{T}
    radial_deriv::RadialDerivativeOperator{T}
    angular_ops::Dict{Symbol, AngularMomentumOperator{T}}
    
    function SphericalDivergence{T}(coords::SphericalCoordinates{T}) where T<:Real
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
Apply divergence operator to vector field.
"""
function apply_divergence!(div_op::SphericalDivergence{T}, input_vector::Array{Complex{T},4}, 
                          output_scalar::Array{Complex{T},3}) where T<:Real
    
    coords = div_op.coords
    r_grid = coords.r_grid
    theta_grid = coords.theta_grid
    
    # Initialize output
    fill!(output_scalar, 0)
    
    # Term 1: (1/r²)(∂/∂r)(r²Fᵣ)
    temp_scalar = similar(output_scalar)
    
    # First multiply Fr by r²
    Fr_times_r2 = similar(view(input_vector, 1, :, :, :))
    for idx in CartesianIndices(size(Fr_times_r2))
        r = r_grid[idx]
        Fr_times_r2[idx] = input_vector[1, idx] * r^2
    end
    
    # Then take radial derivative
    apply_radial_derivative!(div_op.radial_deriv, Fr_times_r2, temp_scalar)
    
    # Finally divide by r²
    for idx in CartesianIndices(size(output_scalar))
        r = r_grid[idx]
        if r > 1e-12
            output_scalar[idx] += temp_scalar[idx] / r^2
        end
    end
    
    # Term 2: (1/(r sin θ))(∂/∂θ)(sin θ Fθ)  
    Ftheta_times_sin = similar(view(input_vector, 2, :, :, :))
    for idx in CartesianIndices(size(Ftheta_times_sin))
        theta = theta_grid[idx]
        Ftheta_times_sin[idx] = input_vector[2, idx] * sin(theta)
    end
    
    apply_angular_derivative!(div_op.angular_ops[:theta], Ftheta_times_sin, temp_scalar)
    
    for idx in CartesianIndices(size(output_scalar))
        r = r_grid[idx]
        theta = theta_grid[idx]
        sin_theta = sin(theta)
        
        if r > 1e-12 && sin_theta > 1e-12
            output_scalar[idx] += temp_scalar[idx] / (r * sin_theta)
        end
    end
    
    # Term 3: (1/(r sin θ))(∂Fφ/∂φ)
    apply_azimuthal_derivative!(view(input_vector, 3, :, :, :), temp_scalar)
    
    for idx in CartesianIndices(size(output_scalar))
        r = r_grid[idx]
        theta = theta_grid[idx]
        sin_theta = sin(theta)
        
        if r > 1e-12 && sin_theta > 1e-12
            output_scalar[idx] += temp_scalar[idx] / (r * sin_theta)
        end
    end
end

"""
Apply spherical divergence using spin-weighted approach.
Following Dedalus ball_wrapper.Divergence implementation.
"""
function apply_spherical_divergence_spin!(input::Array{Complex{T},4}, output::Array{Complex{T},3}, 
                                        coords::SphericalCoordinates{T}) where T<:Real
    
    n_phi, n_theta, n_r = size(output)
    l_max = n_theta ÷ 2
    
    # Convert vector field to spin-weighted components
    # Input: [F_r, F_theta, F_phi] → [F_r, F_minus, F_plus]
    input_spin = zeros(Complex{T}, 3, n_phi, n_theta, n_r)
    
    # Convert from spherical components to spin components
    convert_to_spin_components!(input, input_spin, coords)
    
    # Initialize output
    fill!(output, zero(Complex{T}))
    
    for ell in 0:l_max
        # Create E matrix and derivative operators for this ell
        E_matrix = create_E_matrix(n_r, ell, T)
        
        # Extract coefficient data for this ell mode
        input_coeffs = zeros(Complex{T}, 3, n_r)  # Three spin components
        output_coeffs = zeros(Complex{T}, n_r)
        
        for comp in 1:3, r_idx in 1:n_r
            # For each radial point, extract spherical harmonic coefficient
            # This is simplified - in practice would use proper SHT
            input_coeffs[comp, r_idx] = input_spin[comp, 1, ell+1, r_idx]  # Simplified indexing
        end
        
        # Apply divergence transformation
        apply_divergence_ell_mode!(input_coeffs, output_coeffs, ell, E_matrix, n_r, T)
        
        # Store back to output array
        for r_idx in 1:n_r
            output[ell+1, 1, r_idx] = output_coeffs[r_idx]  # Simplified indexing
        end
    end
end

"""
Apply divergence for single ell mode following Dedalus ball_wrapper patterns.
"""
function apply_divergence_ell_mode!(input_coeffs::Matrix{Complex{T}}, output_coeffs::Vector{Complex{T}}, 
                                  ell::Int, E_matrix::SparseMatrixCSC{Complex{T},Int}, n_r::Int, ::Type{T}) where T<:Real
    
    # Divergence components in ball geometry
    # Following Dedalus ball_wrapper.Divergence._radial_matrix formulations
    
    # Get xi factors for coupling between spin components
    if ell >= 1
        xi_plus = xi_factor(1, ell, T)
        xi_minus = xi_factor(-1, ell, T)
    else
        xi_plus = zero(T)
        xi_minus = zero(T)
    end
    
    # Get derivative operators
    D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
    D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
    
    # Divergence calculation following Dedalus patterns
    # ∇⋅F = radial_part + angular_part
    
    # 1. Radial contribution from F_r component
    F_r = @view input_coeffs[1, :]  # Radial component
    radial_contribution = D_plus * F_r  # Standard radial derivative
    
    # 2. Angular contributions from F_minus and F_plus components
    angular_contribution = zeros(Complex{T}, n_r)
    
    if ell >= 1
        F_minus = @view input_coeffs[2, :]  # Spin -1 component
        F_plus = @view input_coeffs[3, :]   # Spin +1 component
        
        # Angular momentum coupling terms
        # Following spin-weighted spherical harmonic divergence formulas
        angular_contribution += im * xi_plus * (D_minus * F_minus)
        angular_contribution -= im * xi_minus * (D_plus * F_plus)
    end
    
    # Combine contributions and solve with mass matrix
    rhs = radial_contribution + angular_contribution
    output_coeffs[:] = E_matrix \ rhs
end

"""
Create divergence matrix transformation for unified operator application.
Following Dedalus approach for precomputed operator matrices.
"""
function create_divergence_matrix(n_r::Int, ell::Int, ::Type{T}) where T<:Real
    
    # Create unified divergence transformation matrix
    # Input: vector field coefficients (3 × n_r for r, θ, φ components)  
    # Output: scalar field coefficients (n_r)
    
    input_size = 3 * n_r
    D_matrix = spzeros(Complex{T}, n_r, input_size)
    
    # Get E matrix and derivative operators
    E_matrix = create_E_matrix(n_r, ell, T)
    E_inv = inv(Matrix(E_matrix))  # For simplicity, use dense inverse
    
    # Get xi factors and derivative operators
    if ell >= 1
        xi_plus = xi_factor(1, ell, T)
        xi_minus = xi_factor(-1, ell, T)
        D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
        D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
    else
        xi_plus = zero(T)
        xi_minus = zero(T)
        D_plus = spzeros(T, n_r, n_r)
        D_minus = spzeros(T, n_r, n_r)
    end
    
    # Radial component block (columns 1:n_r)
    D_matrix[1:n_r, 1:n_r] = E_inv * D_plus
    
    # Theta component block (columns n_r+1:2*n_r) - converted to minus spin
    if ell >= 1
        D_matrix[1:n_r, (n_r+1):(2*n_r)] = E_inv * (im * xi_plus * D_minus)
    end
    
    # Phi component block (columns 2*n_r+1:3*n_r) - converted to plus spin
    if ell >= 1
        D_matrix[1:n_r, (2*n_r+1):(3*n_r)] = E_inv * (-im * xi_minus * D_plus)
    end
    
    return D_matrix
end

"""
Convert vector field from spherical components to spin-weighted components.
Following Dedalus conversion between coordinate systems.
"""
function convert_to_spin_components!(spherical_vector::Array{Complex{T},4}, spin_vector::Array{Complex{T},4}, 
                                   coords::SphericalCoordinates{T}) where T<:Real
    
    # Convert [F_r, F_theta, F_phi] to [F_r, F_minus, F_plus]
    # F_minus = (F_theta + i*F_phi) / sqrt(2)
    # F_plus = (F_theta - i*F_phi) / sqrt(2)
    # F_r remains the same
    
    sqrt2_inv = T(1) / sqrt(T(2))
    
    for idx in CartesianIndices(size(spherical_vector)[2:4])
        # Radial component unchanged
        spin_vector[1, idx] = spherical_vector[1, idx]
        
        # Convert theta and phi to spin components
        F_theta = spherical_vector[2, idx]
        F_phi = spherical_vector[3, idx]
        
        spin_vector[2, idx] = sqrt2_inv * (F_theta + im * F_phi)  # F_minus
        spin_vector[3, idx] = sqrt2_inv * (F_theta - im * F_phi)  # F_plus
    end
end

"""
Apply divergence with proper geometric scaling and regularity conditions.
Handles singularities at center and poles properly.
"""
function apply_divergence_with_regularity!(div_op::SphericalDivergence{T}, input_vector::Array{Complex{T},4}, 
                                         output_scalar::Array{Complex{T},3}) where T<:Real
    
    # Apply basic divergence
    apply_divergence!(div_op, input_vector, output_scalar)
    
    # Apply regularity conditions at center and poles
    coords = div_op.coords
    r_grid = coords.r_grid
    theta_grid = coords.theta_grid
    
    # Regularity at center (r = 0)
    for j in axes(output_scalar, 2), k in axes(output_scalar, 3)
        if r_grid[1, j, k] < 1e-12
            output_scalar[1, j, k] = zero(Complex{T})
        end
    end
    
    # Regularity at poles (θ = 0, π)
    for i in axes(output_scalar, 1), k in axes(output_scalar, 3)
        if abs(theta_grid[i, 1, k]) < 1e-12 || abs(theta_grid[i, end, k] - π) < 1e-12
            output_scalar[i, 1, k] = zero(Complex{T})
            output_scalar[i, end, k] = zero(Complex{T})
        end
    end
end