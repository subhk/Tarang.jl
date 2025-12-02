"""
Spherical Curl Operator

Implements curl operator ∇× in spherical coordinates using spin-weighted spherical harmonics.
Based on Dedalus ball_wrapper curl implementation with proper spin transformations.
"""

using LinearAlgebra
using SparseArrays

"""
Spherical Curl Operator ∇×

Implements curl in spherical coordinates using spin-weighted spherical harmonics.
"""
struct SphericalCurl{T<:Real}
    coords::SphericalCoordinates{T}
    radial_deriv::RadialDerivativeOperator{T}
    angular_ops::Dict{Symbol, AngularMomentumOperator{T}}
    
    function SphericalCurl{T}(coords::SphericalCoordinates{T}) where T<:Real
        radial_deriv = RadialDerivativeOperator{T}(
            ZernikePolynomials{T}(coords.nr-1, coords.ntheta÷2), 1
        )
        
        harmonics = SphericalHarmonics{T}(coords.ntheta÷2, coords.ntheta, coords.nphi, 1)  # Spin weight 1
        angular_ops = Dict{Symbol, AngularMomentumOperator{T}}(
            :L_plus => AngularMomentumOperator{T}(harmonics, :L_plus),
            :L_minus => AngularMomentumOperator{T}(harmonics, :L_minus)
        )
        
        new{T}(coords, radial_deriv, angular_ops)
    end
end

"""
Apply curl operator to vector field following Dedalus ball_wrapper implementation.

Based on Dedalus curl implementation with spin-weighted spherical harmonics and
proper radial operator matrices. Uses E matrices and D+/D- operators with xi scaling.
"""
function apply_curl!(curl_op::SphericalCurl{T}, input_vector::Array{Complex{T},4}, 
                    output_vector::Array{Complex{T},4}) where T<:Real
    
    coords = curl_op.coords
    n_comp, n_phi, n_theta, n_r = size(input_vector)
    l_max = n_theta ÷ 2
    
    # Initialize output
    fill!(output_vector, zero(Complex{T}))
    
    # Apply curl for each ell mode following Dedalus pattern
    for ell in 0:l_max
        # Apply curl transformation for this ell mode
        apply_curl_ell_mode!(input_vector, output_vector, ell, coords, curl_op, T)
    end
    
    # Apply regularity conditions at coordinate singularities
    apply_curl_regularity_conditions!(output_vector, coords)
end

"""
Apply curl for single ell mode following Dedalus ball_wrapper.curl implementation.

The Dedalus curl implementation uses three separate operations:
1. Component 0 (minus): E_{ell-1} * out[0] = -i * xi(+1,ell) * D- * in[1] 
2. Component 1 (zero): E_{ell} * out[1] = i * xi(-1,ell) * D+ * in[2] - i * xi(+1,ell) * D- * in[0]
3. Component 2 (plus): E_{ell+1} * out[2] = i * xi(-1,ell) * D+ * in[1]
"""
function apply_curl_ell_mode!(input_vector::Array{Complex{T},4}, output_vector::Array{Complex{T},4},
                             ell::Int, coords::SphericalCoordinates{T}, curl_op::SphericalCurl{T}, ::Type{T}) where T<:Real
    
    n_comp, n_phi, n_theta, n_r = size(input_vector)
    
    # Get xi scaling factors
    xi_minus = xi_factor(-1, ell, T)  # xi(-1, ell)
    xi_plus = xi_factor(1, ell, T)    # xi(+1, ell)
    
    # Create radial operator matrices and E matrices
    E_ell = create_E_matrix(n_r, ell, T)
    
    # Extract input data for this ell mode (simplified indexing)
    # In practice, this would use proper spherical harmonic transforms
    input_data = zeros(Complex{T}, 3, n_r)
    output_data = zeros(Complex{T}, 3, n_r)
    
    # For simplified demonstration, use first mode  
    mode_idx = min(ell + 1, n_theta)
    for comp in 1:3, r_idx in 1:n_r
        input_data[comp, r_idx] = input_vector[comp, 1, mode_idx, r_idx]
    end
    
    # Component 1 (minus): Apply curl from plus to minus component
    if ell >= 1
        E_ell_minus1 = create_E_matrix(n_r, ell-1, T)
        D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
        
        # E_{ell-1} * output[1] = -i * xi(+1,ell) * D- * input[2]
        rhs = -im * xi_plus * (D_minus * input_data[2, :])
        output_data[1, :] = E_ell_minus1 \ rhs
    else
        output_data[1, :] .= zero(Complex{T})
    end
    
    # Component 2 (zero): Apply curl from minus and plus to zero component  
    E_ell_zero = create_E_matrix(n_r, ell, T)
    D_minus_ell1 = create_radial_operator_matrix(n_r, :D_minus, ell+1, T)
    
    rhs_zero = im * xi_minus * (D_minus_ell1 * input_data[3, :])  # from plus component
    
    if ell >= 1
        D_plus_ellm1 = create_radial_operator_matrix(n_r, :D_plus, ell-1, T)  
        rhs_zero -= im * xi_plus * (D_plus_ellm1 * input_data[1, :])  # from minus component
    end
    
    output_data[2, :] = E_ell_zero \ rhs_zero
    
    # Component 3 (plus): Apply curl from zero to plus component
    E_ell_plus1 = create_E_matrix(n_r, ell+1, T)
    D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
    
    # E_{ell+1} * output[3] = i * xi(-1,ell) * D+ * input[2]
    rhs = im * xi_minus * (D_plus * input_data[2, :])
    output_data[3, :] = E_ell_plus1 \ rhs
    
    # Store output data back (simplified)
    for comp in 1:3, r_idx in 1:n_r
        output_vector[comp, 1, mode_idx, r_idx] += output_data[comp, r_idx]
    end
end

"""
Apply curl for minus spin component (ω₋).
In dedalus: regindex (0,) → (2,) mapping
"""
function apply_spin_minus_curl!(curl_op::SphericalCurl{T}, input_spin::Array{Complex{T},4}, 
                               output_spin::Array{Complex{T},4}) where T<:Real
    
    n_spin, n_phi, n_theta, n_r = size(input_spin)
    
    # Following dedalus regindex_out logic: (-, +) → 0
    # Apply radial curl operator for minus component
    for i_phi in 1:n_phi, i_theta in 1:n_theta
        # Get ell value for this (phi, theta) mode
        ell = get_ell_value(curl_op, i_phi, i_theta)
        
        # Get radial matrix for minus component curl
        radial_matrix = get_radial_curl_matrix(curl_op, :minus, ell)
        
        # Apply matrix operation: output[3] = D₋ * input[1]  
        input_radial = view(input_spin, 1, i_phi, i_theta, :)    # minus component
        output_radial = view(output_spin, 3, i_phi, i_theta, :)   # to zero component
        
        # Matrix-vector multiplication
        mul!(output_radial, radial_matrix, input_radial)
    end
end

"""
Apply curl for plus spin component (ω₊).
In dedalus: regindex (1,) → (2,) mapping
"""
function apply_spin_plus_curl!(curl_op::SphericalCurl{T}, input_spin::Array{Complex{T},4}, 
                              output_spin::Array{Complex{T},4}) where T<:Real
    
    n_spin, n_phi, n_theta, n_r = size(input_spin)
    
    # Following dedalus regindex_out logic: (-, +) → 0
    # Apply radial curl operator for plus component
    for i_phi in 1:n_phi, i_theta in 1:n_theta
        # Get ell value for this (phi, theta) mode
        ell = get_ell_value(curl_op, i_phi, i_theta)
        
        # Get radial matrix for plus component curl
        radial_matrix = get_radial_curl_matrix(curl_op, :plus, ell)
        
        # Apply matrix operation: output[3] += D₊ * input[2]
        input_radial = view(input_spin, 2, i_phi, i_theta, :)    # plus component
        output_radial = view(output_spin, 3, i_phi, i_theta, :)   # to zero component
        
        # Matrix-vector multiplication (accumulate with minus result)
        output_temp = radial_matrix * input_radial
        output_radial .+= output_temp
    end
end

"""
Apply curl for zero spin component (ω₀).
In dedalus: regindex (2,) → (0,), (1,) mapping
"""
function apply_spin_zero_curl!(curl_op::SphericalCurl{T}, input_spin::Array{Complex{T},4}, 
                              output_spin::Array{Complex{T},4}) where T<:Real
    
    n_spin, n_phi, n_theta, n_r = size(input_spin)
    
    # Following dedalus regindex_out logic: 0 → (-, +)
    # Apply radial curl operator for zero component
    for i_phi in 1:n_phi, i_theta in 1:n_theta
        # Get ell value for this (phi, theta) mode
        ell = get_ell_value(curl_op, i_phi, i_theta)
        
        # Get radial matrices for zero component curl  
        radial_matrix_minus = get_radial_curl_matrix(curl_op, :zero_to_minus, ell)
        radial_matrix_plus = get_radial_curl_matrix(curl_op, :zero_to_plus, ell)
        
        input_radial = view(input_spin, 3, i_phi, i_theta, :)     # zero component
        
        # Apply to minus output: output[1] = D₀₋ * input[3]
        output_minus = view(output_spin, 1, i_phi, i_theta, :)    # to minus component
        mul!(output_minus, radial_matrix_minus, input_radial)
        
        # Apply to plus output: output[2] = D₀₊ * input[3]  
        output_plus = view(output_spin, 2, i_phi, i_theta, :)     # to plus component
        mul!(output_plus, radial_matrix_plus, input_radial)
    end
end

"""
Get ell value for spherical harmonic mode at given grid indices.
"""
function get_ell_value(curl_op::SphericalCurl{T}, i_phi::Int, i_theta::Int) where T<:Real
    # Simplified mapping - actual implementation would use proper SHT indexing
    l_max = size(curl_op.coords.theta_grid, 2) ÷ 2
    return min(i_theta - 1, l_max)
end

"""
Get radial curl matrix for specific spin component and ell mode.
Follows dedalus _radial_matrix patterns with D+ and D- operators.
"""
function get_radial_curl_matrix(curl_op::SphericalCurl{T}, component::Symbol, ell::Int) where T<:Real
    n_r = size(curl_op.coords.r_grid, 1)
    
    # Following dedalus _radial_matrix patterns with proper xi scaling
    if component == :minus
        # - component to 0 component: -i * xi(+1, ell+1) * D+ matrix
        xi_plus = xi_factor(1, ell+1, T)
        D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
        return -im * xi_plus * D_plus
        
    elseif component == :plus
        # + component to 0 component: i * xi(-1, ell-1) * D- matrix
        if ell >= 1
            xi_minus = xi_factor(-1, ell-1, T)
            D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
            return im * xi_minus * D_minus
        else
            return spzeros(Complex{T}, n_r, n_r)
        end
        
    elseif component == :zero_to_minus
        # 0 component to - component: -i * xi(+1, ell) * D- matrix
        xi_plus = xi_factor(1, ell, T)
        D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
        return -im * xi_plus * D_minus
        
    elseif component == :zero_to_plus
        # 0 component to + component: i * xi(-1, ell) * D+ matrix
        xi_minus = xi_factor(-1, ell, T)
        D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
        return im * xi_minus * D_plus
        
    else
        error("Unknown radial matrix component: $component")
    end
end

"""
Transform vector field from coordinate components to spin-weighted components.
"""
function transform_to_spin_components!(coord_vector::Array{Complex{T},4}, spin_vector::Array{Complex{T},4}, 
                                     coords::SphericalCoordinates{T}) where T<:Real
    
    # Transform [F_r, F_theta, F_phi] to [F_r, F_minus, F_plus]
    # Following standard spherical → spin transformation
    
    sqrt2_inv = T(1) / sqrt(T(2))
    
    for idx in CartesianIndices(size(coord_vector)[2:4])
        # Radial component unchanged
        spin_vector[1, idx] = coord_vector[1, idx]
        
        # Convert theta and phi to spin components
        F_theta = coord_vector[2, idx]
        F_phi = coord_vector[3, idx]
        
        spin_vector[2, idx] = sqrt2_inv * (F_theta + im * F_phi)  # F_minus
        spin_vector[3, idx] = sqrt2_inv * (F_theta - im * F_phi)  # F_plus
    end
end

"""
Transform vector field from spin-weighted components to coordinate components.
"""
function transform_from_spin_components!(spin_vector::Array{Complex{T},4}, coord_vector::Array{Complex{T},4}, 
                                       coords::SphericalCoordinates{T}) where T<:Real
    
    # Transform [F_r, F_minus, F_plus] to [F_r, F_theta, F_phi]
    # Inverse of spin → coordinate transformation
    
    sqrt2 = sqrt(T(2))
    
    for idx in CartesianIndices(size(spin_vector)[2:4])
        # Radial component unchanged
        coord_vector[1, idx] = spin_vector[1, idx]
        
        # Convert spin components back to theta and phi
        F_minus = spin_vector[2, idx]
        F_plus = spin_vector[3, idx]
        
        coord_vector[2, idx] = sqrt2 * real(F_minus + F_plus)      # F_theta
        coord_vector[3, idx] = sqrt2 * imag(F_plus - F_minus)      # F_phi
    end
end

"""
Create spin curl transformation matrix following Dedalus ball_wrapper patterns.
"""
function create_spin_curl_transformation_matrix(curl_op::SphericalCurl{T}, ell::Int) where T<:Real
    n_r = size(curl_op.coords.r_grid, 1)
    n_spin = 3  # Three spin components: (-, +, 0) → (1, 2, 3)
    
    # Create block transformation matrix: [3×n_r, 3×n_r] for all spin components
    matrix_size = n_spin * n_r
    transform_matrix = spzeros(Complex{T}, matrix_size, matrix_size)
    
    # Get xi scaling factors following Dedalus xi(mu, ell) pattern
    xi_minus = xi_factor(-1, ell, T)  # xi(-1, ell)
    xi_plus = xi_factor(1, ell, T)    # xi(+1, ell)
    
    # Create radial operator matrices
    E_matrix = create_E_matrix(n_r, ell, T)
    D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
    D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
    
    # Block structure:
    # Input:  [u_minus; u_plus; u_zero]  (spin components -, +, 0)
    # Output: [curl_minus; curl_plus; curl_zero]
    
    # Following Dedalus curl transformation formulas:
    
    # 1. curl_minus component (output block 1) from u_plus (input block 2)
    #    E * curl_minus = -i * xi(+1, ell+1) * D- * u_plus
    if ell >= 0
        xi_plus_next = xi_factor(1, ell+1, T)
        block_row = 1:n_r
        block_col = (n_r+1):(2*n_r)  # u_plus block
        transform_matrix[block_row, block_col] = E_matrix \ (-im * xi_plus_next * D_minus)
    end
    
    # 2. curl_plus component (output block 2) from u_zero (input block 3)  
    #    E * curl_plus = i * xi(-1, ell) * D+ * u_zero
    block_row = (n_r+1):(2*n_r)
    block_col = (2*n_r+1):(3*n_r)  # u_zero block
    transform_matrix[block_row, block_col] = E_matrix \ (im * xi_minus * D_plus)
    
    # 3. curl_zero component (output block 3) from u_minus and u_plus
    #    E * curl_zero = -i * xi(+1, ell) * D- * u_minus + i * xi(-1, ell+1) * D+ * u_plus
    block_row = (2*n_r+1):(3*n_r)
    
    # From u_minus (input block 1)
    if ell >= 1
        block_col = 1:n_r  # u_minus block
        transform_matrix[block_row, block_col] = E_matrix \ (-im * xi_plus * D_minus)
    end
    
    # From u_plus (input block 2)
    if ell >= 0
        xi_minus_next = xi_factor(-1, ell+1, T)
        block_col = (n_r+1):(2*n_r)  # u_plus block
        transform_matrix[block_row, block_col] += E_matrix \ (im * xi_minus_next * D_plus)
    end
    
    return transform_matrix
end

"""
Apply enhanced spin curl with complete Dedalus ball_wrapper patterns.
"""
function enhanced_apply_spin_curl!(curl_op::SphericalCurl{T}, input_spin::Array{Complex{T},4}, 
                                 output_spin::Array{Complex{T},4}, ell::Int) where T<:Real
    
    n_spin, n_phi, n_theta, n_r = size(input_spin)
    
    # Get scaling factors
    xi_minus = xi_factor(-1, ell, T)  
    xi_plus = xi_factor(1, ell, T)
    
    # Create operator matrices
    E_matrix = create_E_matrix(n_r, ell, T)
    D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
    D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
    
    # Apply curl transformation following dedalus ball_wrapper.curl pattern
    
    # First component (minus): from plus component using D- and xi factors
    if ell >= 1
        # Solve E_matrix * output[1] = -i * xi_plus * D_minus * input[2]
        rhs = -im * xi_plus * (D_minus * input_spin[2, :, :, :])
        output_spin[1, :, :, :] = E_matrix \ rhs
    else
        output_spin[1, :, :, :] .= 0
    end
    
    # Second component (zero): from both minus and plus components
    # Solve E_matrix * output[3] = i * xi_minus * D_plus * input[3] - i * xi_plus * D_minus * input[1]
    rhs = im * xi_minus * (D_plus * input_spin[3, :, :, :])
    if ell >= 1
        rhs -= im * xi_plus * (D_minus * input_spin[1, :, :, :])
    end
    output_spin[3, :, :, :] = E_matrix \ rhs
    
    # Third component (plus): from zero component using D+ and xi factors  
    # Solve E_matrix * output[2] = i * xi_minus * D_plus * input[3]
    rhs = im * xi_minus * (D_plus * input_spin[3, :, :, :])
    output_spin[2, :, :, :] = E_matrix \ rhs
end

"""
Apply regularity conditions for curl operator at coordinate singularities.
"""
function apply_curl_regularity_conditions!(output_vector::Array{Complex{T},4}, 
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
Create unitary transformation matrix for coordinate ↔ spin conversions.
"""
function create_unitary_spin_matrix(::Type{T}; adjoint::Bool=false) where T<:Real
    
    # Unitary matrix for [r, θ, φ] ↔ [r, -, +] transformation
    sqrt2_inv = T(1) / sqrt(T(2))
    
    if adjoint
        # Spin → Coordinate transformation
        return Complex{T}[
            1    0              0
            0    sqrt2_inv      sqrt2_inv
            0    im*sqrt2_inv  -im*sqrt2_inv
        ]
    else
        # Coordinate → Spin transformation  
        return Complex{T}[
            1    0              0
            0    sqrt2_inv     -im*sqrt2_inv
            0    sqrt2_inv      im*sqrt2_inv
        ]
    end
end