"""
Radial Operators for Spherical Coordinates

Radial derivative operators and Jacobi polynomial matrices for ball spectral methods.
Includes differentiation matrices, connection coefficients, and operator matrix creation.
"""

using LinearAlgebra
using SparseArrays

"""
Radial Derivative Operator

Implements radial derivatives ∂/∂r using Zernike polynomial differentiation.
Based on dedalus derivative operator construction.
"""
struct RadialDerivativeOperator{T<:Real}
    basis::ZernikePolynomials{T}
    derivative_order::Int
    
    # Differentiation matrices for each (l,m) mode
    derivative_matrices::Dict{Tuple{Int,Int}, SparseMatrixCSC{T,Int}}
    
    function RadialDerivativeOperator{T}(basis::ZernikePolynomials{T}, derivative_order::Int=1) where T<:Real
        derivative_matrices = build_radial_derivative_matrices(basis, derivative_order, T)
        new{T}(basis, derivative_order, derivative_matrices)
    end
end

"""
Build radial derivative matrices for all (l,m) combinations.
"""
function build_radial_derivative_matrices(basis::ZernikePolynomials{T}, order::Int, ::Type{T}) where T<:Real
    matrices = Dict{Tuple{Int,Int}, SparseMatrixCSC{T,Int}}()
    
    for l in 0:basis.l_max
        for m in (-l):l
            D_matrix = build_single_radial_derivative_matrix(basis, l, order, T)
            matrices[(l,m)] = D_matrix
        end
    end
    
    return matrices
end

"""
Build single radial derivative matrix for given l quantum number.

Based on dedalus Zernike differentiation: D_n^l = connection matrices between
different Jacobi parameter sets.
"""
function build_single_radial_derivative_matrix(basis::ZernikePolynomials{T}, l::Int, order::Int, ::Type{T}) where T<:Real
    n_max = basis.n_max
    alpha = basis.alpha
    beta = basis.beta
    
    # Build differentiation matrix using Jacobi polynomial connection relations
    D = spzeros(T, n_max + 1, n_max + 1)
    
    # First-order derivative matrix
    if order == 1
        for n in 0:n_max
            for m in 0:n_max
                # Connection coefficients for ∂/∂r Zernike polynomials
                # Z_n^l → sum_m c_{nm}^l Z_m^{l+1}
                coeff = zernike_derivative_coefficient(n, m, l, alpha, beta, T)
                if abs(coeff) > 1e-14
                    D[m+1, n+1] = coeff
                end
            end
        end
    else
        # Higher-order derivatives through matrix powers
        D1 = build_single_radial_derivative_matrix(basis, l, 1, T)
        D = D1
        for i in 2:order
            D = D * D1
        end
    end
    
    return D
end

"""
Apply radial derivative to field data.
"""
function apply_radial_derivative!(radial_op::RadialDerivativeOperator{T}, 
                                 input::Array{Complex{T},3}, output::Array{Complex{T},3}) where T<:Real
    
    nphi, ntheta, nr = size(input)
    
    # Apply derivative to each (φ,θ) mode separately
    for j in 1:ntheta, i in 1:nphi
        # Extract radial profile
        radial_profile = @view input[i, j, :]
        output_profile = @view output[i, j, :]
        
        # Apply differentiation matrix (using l=0 for scalar field)
        D_matrix = radial_op.derivative_matrices[(0, 0)]
        mul!(output_profile, D_matrix, radial_profile)
    end
end

"""
Create radial operator matrices D+, D- following dedalus patterns.
Based on Jacobi polynomial differentiation for Zernike basis.
"""
function create_radial_operator_matrix(n_r::Int, type::Symbol, ell::Int, ::Type{T}) where T<:Real
    # Create radial differential operator matrices for Jacobi polynomial basis
    # Following Dedalus approach for ball spectral methods with proper connection coefficients
    
    if type == :D_plus
        # D+ operator: radial derivative that raises spin weight
        # Maps from Q_n^{(0,ell+1/2)} to Q_{n-1}^{(0,ell+3/2)} basis
        # Following dedalus ball_wrapper pattern for D+ matrices
        
        return create_jacobi_derivative_matrix(n_r, ell, :raising, T)
        
    elseif type == :D_minus
        # D- operator: radial derivative that lowers spin weight  
        # Maps from Q_n^{(0,ell+1/2)} to Q_{n-1}^{(0,ell-1/2)} basis
        # Following dedalus ball_wrapper pattern for D- matrices
        
        return create_jacobi_derivative_matrix(n_r, ell, :lowering, T)
        
    elseif type == :radial_deriv
        # Standard radial derivative in same basis
        # ∂/∂r acting on Q_n^{(0,ell+1/2)} → Q_{n-1}^{(0,ell+1/2)}
        
        return create_jacobi_derivative_matrix(n_r, ell, :standard, T)
        
    else
        error("Unknown radial operator type: $type. Valid types: :D_plus, :D_minus, :radial_deriv")
    end
end

"""
    create_jacobi_derivative_matrix(n_r, ell, derivative_type, T) -> SparseMatrixCSC

Create Jacobi polynomial derivative matrices with proper connection coefficients.
Used for ball domain radial derivatives following Dedalus approach.

# Arguments
- `n_r::Int`: Number of radial modes
- `ell::Int`: Spherical harmonic degree
- `derivative_type::Symbol`: Type of derivative (:raising, :lowering, :standard)
- `T::Type`: Floating point type

# Mathematical Background
For ball domain with Jacobi polynomials Q_n^{(α,β)}, derivative matrices connect:
- :raising: Q_n^{(α,β)} → Q_{n-1}^{(α,β+1)} (D+ operator)
- :lowering: Q_n^{(α,β)} → Q_{n-1}^{(α,β-1)} (D- operator)  
- :standard: Q_n^{(α,β)} → Q_{n-1}^{(α,β)} (standard ∂/∂r)
"""
function create_jacobi_derivative_matrix(n_r::Int, ell::Int, derivative_type::Symbol, ::Type{T}) where T<:Real
    # Create derivative matrix for Jacobi polynomials in ball domain
    # Following Dedalus connection coefficient approach
    
    D = spzeros(T, n_r, n_r)
    
    # Jacobi polynomial parameters for ball domain
    alpha = T(0)                    # Ball domain parameter
    beta_current = ell + T(0.5)     # Current ell-dependent parameter
    
    # Target beta parameter after derivative
    if derivative_type == :raising
        beta_target = beta_current + 1
    elseif derivative_type == :lowering
        beta_target = max(beta_current - 1, T(-0.5))  # Don't go below -1/2
    else  # :standard
        beta_target = beta_current
    end
    
    # Connection coefficients for Jacobi polynomial derivatives
    # Following standard formulas for differentiation of orthogonal polynomials
    
    for n in 0:(n_r-1)
        for m in 0:min(n+1, n_r-1)  # Upper triangular structure for derivatives
            
            # Connection coefficient between Jacobi polynomials
            # ∂/∂r P_n^{(α,β)} = sum_m c_{nm} P_m^{(α,β')}
            
            if derivative_type == :standard && m == n-1 && n >= 1
                # Standard derivative: main sub-diagonal
                coeff = (n + alpha + beta_current) / 2
                D[m+1, n+1] = coeff
                
            elseif derivative_type == :raising && m == n-1 && n >= 1
                # Raising derivative: increases beta parameter
                coeff = (n + alpha + beta_current) / 2
                D[m+1, n+1] = coeff
                
            elseif derivative_type == :lowering && m == n-1 && n >= 1 && beta_target >= T(-0.5)
                # Lowering derivative: decreases beta parameter
                coeff = (n + alpha + beta_current) / 2
                # Additional factor for lowering transformation
                if beta_current > T(-0.5)
                    beta_factor = sqrt(beta_current / (beta_current + 1))
                    coeff *= beta_factor
                end
                D[m+1, n+1] = coeff
            end
        end
    end
    
    return D
end

"""
    radial_curl_matrix_component(n_r, ell, component, T) -> SparseMatrixCSC

Create radial matrix components for curl operator following Dedalus patterns.
Used in spin-weighted curl calculations for vector fields in ball geometry.
"""
function radial_curl_matrix_component(n_r::Int, ell::Int, component::Symbol, ::Type{T}) where T<:Real
    # Create radial matrix components for curl operator
    # Following dedalus ball_wrapper._radial_matrix patterns
    
    if component == :minus_from_plus
        # - component from + component: -i * xi(+1, ell+1) * D- matrix
        if ell >= 0
            xi_plus = xi_factor(1, ell+1, T)
            D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
            return -im * xi_plus * D_minus
        else
            return spzeros(Complex{T}, n_r, n_r)
        end
        
    elseif component == :plus_from_zero
        # + component from 0 component: i * xi(-1, ell) * D+ matrix
        xi_minus = xi_factor(-1, ell, T)
        D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
        return im * xi_minus * D_plus
        
    elseif component == :zero_from_minus
        # 0 component from - component: -i * xi(+1, ell) * D- matrix
        xi_plus = xi_factor(1, ell, T)
        D_minus = create_radial_operator_matrix(n_r, :D_minus, ell, T)
        return -im * xi_plus * D_minus
        
    elseif component == :zero_from_plus
        # 0 component from + component: i * xi(-1, ell+1) * D+ matrix
        if ell >= 0
            xi_minus = xi_factor(-1, ell+1, T)
            D_plus = create_radial_operator_matrix(n_r, :D_plus, ell, T)
            return im * xi_minus * D_plus
        else
            return spzeros(Complex{T}, n_r, n_r)
        end
        
    else
        error("Unknown radial curl component: $component")
    end
end