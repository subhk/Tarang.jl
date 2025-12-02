"""
Angular Operators for Spherical Coordinates

Angular momentum and angular derivative operators using spin-weighted spherical harmonics.
Includes L± operators, angular derivatives, and matrix element calculations.
"""

using LinearAlgebra
using SparseArrays

"""
Angular Momentum Operator L̂

Implements angular momentum operators L̂₊, L̂₋, L̂ₓ, L̂ᵧ, L̂ᵧ using spin-weighted
spherical harmonics. Based on dedalus spin operators.
"""
struct AngularMomentumOperator{T<:Real}
    harmonics::SphericalHarmonics{T}
    component::Symbol  # :L_plus, :L_minus, :L_x, :L_y, :L_z, :L_squared
    
    # Operator matrices for each radial mode
    operator_matrices::Dict{Int, SparseMatrixCSC{Complex{T},Int}}
    
    function AngularMomentumOperator{T}(harmonics::SphericalHarmonics{T}, component::Symbol) where T<:Real
        operator_matrices = build_angular_momentum_matrices(harmonics, component, T)
        new{T}(harmonics, component, operator_matrices)
    end
end

"""
Build angular momentum operator matrices.
"""
function build_angular_momentum_matrices(harmonics::SphericalHarmonics{T}, component::Symbol, ::Type{T}) where T<:Real
    matrices = Dict{Int, SparseMatrixCSC{Complex{T},Int}}()
    
    l_max = harmonics.l_max
    matrix_size = (l_max + 1) * (2 * l_max + 1)  # Total (l,m) modes
    
    # Single matrix for all radial modes (angular operators don't mix radial modes)
    L_matrix = spzeros(Complex{T}, matrix_size, matrix_size)
    
    # Fill matrix based on angular momentum operator action on Y_l^m
    idx = 1
    for l1 in 0:l_max, m1 in (-l1):l1
        jdx = 1
        for l2 in 0:l_max, m2 in (-l2):l2
            matrix_element = angular_momentum_matrix_element(l1, m1, l2, m2, component, T)
            if abs(matrix_element) > 1e-14
                L_matrix[idx, jdx] = matrix_element
            end
            jdx += 1
        end
        idx += 1
    end
    
    # Same matrix for all radial modes
    for n in 0:10  # Sufficient for most applications
        matrices[n] = L_matrix
    end
    
    return matrices
end

"""
Apply angular momentum operator to field data.
"""
function apply_angular_momentum_operator!(angular_op::AngularMomentumOperator{T}, 
                                        input::Array{Complex{T},3}, output::Array{Complex{T},3}) where T<:Real
    
    nphi, ntheta, nr = size(input)
    harmonics = angular_op.harmonics
    component = angular_op.component
    
    # Transform to spectral space
    spectral_coeffs = zeros(Complex{T}, nphi÷2+1, harmonics.l_max+1, nr)
    forward_spherical_harmonic_transform!(input, spectral_coeffs, harmonics)
    
    # Apply angular momentum operator
    spectral_output = zeros(Complex{T}, size(spectral_coeffs))
    
    for k in 1:nr
        for l in 0:harmonics.l_max
            for m_idx in 1:(nphi÷2+1)
                m = m_idx - 1 - nphi÷4
                if abs(m) <= l
                    coeff_value = spectral_coeffs[m_idx, l+1, k]
                    
                    if component == :L_plus && abs(m+1) <= l
                        # L₊ |l,m⟩ = √(l(l+1) - m(m+1)) |l,m+1⟩
                        factor = sqrt(T(l * (l + 1) - m * (m + 1)))
                        target_m_idx = (m+1) + nphi÷4 + 1
                        if target_m_idx >= 1 && target_m_idx <= nphi÷2+1
                            spectral_output[target_m_idx, l+1, k] += factor * coeff_value
                        end
                        
                    elseif component == :L_minus && abs(m-1) <= l
                        # L₋ |l,m⟩ = √(l(l+1) - m(m-1)) |l,m-1⟩
                        factor = sqrt(T(l * (l + 1) - m * (m - 1)))
                        target_m_idx = (m-1) + nphi÷4 + 1
                        if target_m_idx >= 1 && target_m_idx <= nphi÷2+1
                            spectral_output[target_m_idx, l+1, k] += factor * coeff_value
                        end
                        
                    elseif component == :L_z
                        # L_z |l,m⟩ = m |l,m⟩
                        spectral_output[m_idx, l+1, k] = T(m) * coeff_value
                        
                    elseif component == :L_squared
                        # L² |l,m⟩ = l(l+1) |l,m⟩
                        spectral_output[m_idx, l+1, k] = T(l * (l + 1)) * coeff_value
                    end
                end
            end
        end
    end
    
    # Transform back to physical space
    backward_spherical_harmonic_transform!(spectral_output, output, harmonics)
end

"""
Apply angular derivative (θ or φ) to field data.
"""
function apply_angular_derivative!(angular_op::AngularMomentumOperator{T}, 
                                  input::Array{Complex{T},3}, output::Array{Complex{T},3}) where T<:Real
    
    nphi, ntheta, nr = size(input)
    harmonics = angular_op.harmonics
    
    # Get angular momentum component type
    component = angular_op.component
    
    if component == :theta
        # Apply θ derivative using combination of L+ and L- operators
        apply_theta_derivative_proper!(input, output, harmonics, T)
    elseif component == :phi
        # Apply φ derivative using FFT (already implemented in apply_azimuthal_derivative!)
        apply_azimuthal_derivative!(input, output)
    else
        # Apply specific angular momentum operator
        apply_angular_momentum_operator!(angular_op, input, output)
    end
end

"""
Apply proper θ derivative using spin-weighted spherical harmonic approach.
Based on Dedalus SphericalGradient θ-component implementation.
"""
function apply_theta_derivative_proper!(input::Array{Complex{T},3}, output::Array{Complex{T},3}, 
                                      harmonics::SphericalHarmonics{T}, ::Type{T}) where T<:Real
    
    nphi, ntheta, nr = size(input)
    
    # Transform to spectral space
    spectral_coeffs = zeros(Complex{T}, nphi÷2+1, harmonics.l_max+1, nr)
    forward_spherical_harmonic_transform!(input, spectral_coeffs, harmonics)
    
    # Apply θ derivative in spectral space
    # ∂/∂θ = (1/2)[(L+ * e^{-iφ}) + (L- * e^{iφ})] / i
    # Simplified as combination of L+ and L- actions on Y_l^m
    
    spectral_output = zeros(Complex{T}, size(spectral_coeffs))
    
    for k in 1:nr
        for l in 1:harmonics.l_max  # Start from l=1 (no θ derivative for l=0)
            for m_idx in 1:(nphi÷2+1)
                m = m_idx - 1 - nphi÷4
                if abs(m) <= l
                    coeff_value = spectral_coeffs[m_idx, l+1, k]
                    
                    # θ derivative combines raising/lowering operators
                    # ∂Y_l^m/∂θ involves Y_{l}^{m±1} terms
                    
                    if abs(m+1) <= l
                        # Contribution from m+1 term
                        factor_plus = sqrt(T((l - m) * (l + m + 1))) / 2
                        spectral_output[m_idx, l+1, k] += factor_plus * coeff_value
                    end
                    
                    if abs(m-1) <= l
                        # Contribution from m-1 term  
                        factor_minus = -sqrt(T((l + m) * (l - m + 1))) / 2
                        spectral_output[m_idx, l+1, k] += factor_minus * coeff_value
                    end
                end
            end
        end
    end
    
    # Transform back to physical space
    backward_spherical_harmonic_transform!(spectral_output, output, harmonics)
end

"""
Apply azimuthal (φ) derivative using FFT.
∂f/∂φ = im * FFT^{-1}[m * FFT[f]]
"""
function apply_azimuthal_derivative!(input::Array{Complex{T},3}, output::Array{Complex{T},3}) where T<:Real
    
    nphi, ntheta, nr = size(input)
    
    # Apply derivative to each (θ, r) slice
    for j in 1:ntheta, k in 1:nr
        # Extract φ profile
        phi_profile = @view input[:, j, k]
        output_profile = @view output[:, j, k]
        
        # FFT approach: ∂f/∂φ = im * FFT^{-1}[m * FFT[f]]
        fft_coeffs = fft(phi_profile)
        
        # Apply derivative in Fourier space
        for i in 1:nphi
            m = (i <= nphi÷2 + 1) ? (i - 1) : (i - nphi - 1)  # Frequency index
            fft_coeffs[i] *= im * T(m)
        end
        
        # Transform back
        output_profile[:] = ifft(fft_coeffs)
    end
end

"""
Compute angular momentum matrix elements ⟨Y_l₁^m₁|L̂|Y_l₂^m₂⟩.
"""
function angular_momentum_matrix_element(l1::Int, m1::Int, l2::Int, m2::Int, component::Symbol, ::Type{T}) where T<:Real
    if component == :L_plus
        # L₊ = Lₓ + iL_y raises m by 1
        if l1 == l2 && m1 == m2 + 1
            return sqrt(T(l2 * (l2 + 1) - m2 * (m2 + 1)))
        end
    elseif component == :L_minus
        # L₋ = Lₓ - iL_y lowers m by 1  
        if l1 == l2 && m1 == m2 - 1
            return sqrt(T(l2 * (l2 + 1) - m2 * (m2 - 1)))
        end
    elseif component == :L_x
        # Lₓ = (L₊ + L₋)/2
        if l1 == l2 && abs(m1 - m2) == 1
            if m1 == m2 + 1
                return sqrt(T(l2 * (l2 + 1) - m2 * (m2 + 1))) / 2
            elseif m1 == m2 - 1
                return sqrt(T(l2 * (l2 + 1) - m2 * (m2 - 1))) / 2
            end
        end
    elseif component == :L_y
        # L_y = (L₊ - L₋)/(2i)
        if l1 == l2 && abs(m1 - m2) == 1
            if m1 == m2 + 1
                return -im * sqrt(T(l2 * (l2 + 1) - m2 * (m2 + 1))) / 2
            elseif m1 == m2 - 1
                return im * sqrt(T(l2 * (l2 + 1) - m2 * (m2 - 1))) / 2
            end
        end
    elseif component == :L_z
        # L_z = m (diagonal operator)
        if l1 == l2 && m1 == m2
            return T(m2)
        end
    elseif component == :L_squared
        # L² = l(l+1) (diagonal operator)
        if l1 == l2 && m1 == m2
            return T(l2 * (l2 + 1))
        end
    end
    
    return Complex{T}(0)
end

"""
Create spin-weighted angular operator matrices for curl operations.
Following Dedalus spin-weighted spherical harmonic approach.
"""
function create_spin_angular_matrices(l_max::Int, spin_weight::Int, ::Type{T}) where T<:Real
    
    # Create matrices for spin-weighted spherical harmonics _{s}Y_l^m
    matrix_size = (l_max + 1) * (2 * l_max + 1)
    
    # L+ matrix for spin-weighted harmonics
    L_plus = spzeros(Complex{T}, matrix_size, matrix_size)
    L_minus = spzeros(Complex{T}, matrix_size, matrix_size)
    
    idx = 1
    for l in 0:l_max, m in (-l):l
        if l >= abs(spin_weight)  # Spin-weighted harmonics only exist for l ≥ |s|
            
            jdx = 1  
            for l2 in 0:l_max, m2 in (-l2):l2
                if l2 >= abs(spin_weight)
                    
                    # Spin-weighted L+ action: raises m by 1
                    if l == l2 && m == m2 + 1
                        factor = sqrt(T((l - m2) * (l + m2 + 1)))
                        L_plus[idx, jdx] = factor
                    end
                    
                    # Spin-weighted L- action: lowers m by 1
                    if l == l2 && m == m2 - 1
                        factor = sqrt(T((l + m2) * (l - m2 + 1)))
                        L_minus[idx, jdx] = factor
                    end
                    
                end
                jdx += 1
            end
        end
        idx += 1
    end
    
    return L_plus, L_minus
end