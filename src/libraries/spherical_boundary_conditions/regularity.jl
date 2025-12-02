"""
Regularity enforcement for spherical boundary conditions.

Implements center and pole regularity following dedalus approach:
- Center regularity: only l=0 modes contribute at r=0
- Pole regularity: spherical harmonic evaluation at θ=0,π
- Vector field regularity: spin-weighted harmonic approach
"""

using LinearAlgebra
using ..SphericalBoundaryConditions: SphericalScalarField, SphericalVectorField, SphericalCoordinates
include("utilities.jl")

export enforce_regularity!, enforce_scalar_regularity!, enforce_vector_regularity!
export apply_center_regularity_scalar!, apply_pole_regularity_scalar!
export compute_regular_value_at_center, compute_regular_value_at_pole
export compute_radial_regular_value, compute_angular_regular_value_at_pole

"""
    enforce_regularity!(field, coords) 

Enforce regularity conditions on spherical field following dedalus approach.

Automatically detects field type and applies appropriate regularity:
- Scalar fields: center and pole regularity
- Vector fields: component-wise regularity with spin-weight considerations
"""
function enforce_regularity!(field::Union{SphericalScalarField{T}, SphericalVectorField{T}},
                            coords::SphericalCoordinates{T}) where T<:Real
    if isa(field, SphericalScalarField)
        enforce_scalar_regularity!(field)
    else
        enforce_vector_regularity!(field) 
    end
end

"""
    enforce_scalar_regularity!(field)

Enforce regularity conditions for scalar fields in spherical coordinates.

Applies both center regularity (r=0) and pole regularity (θ=0,π) by replacing
irregular grid values with spectrally computed regular values.
"""
function enforce_scalar_regularity!(field::SphericalScalarField{T}) where T<:Real
    # Apply center regularity at r=0
    apply_center_regularity_scalar!(field, field.coords)
    
    # Apply pole regularity at θ=0,π  
    apply_pole_regularity_scalar!(field, field.coords)
    
    return field
end

"""
    enforce_vector_regularity!(field)

Enforce regularity conditions for vector fields in spherical coordinates.

Applies component-specific regularity:
- Radial component: scalar-like regularity
- Angular components: spin-weighted harmonic regularity
"""
function enforce_vector_regularity!(field::SphericalVectorField{T}) where T<:Real
    # Apply radial component regularity (component 1)
    apply_vector_component_regularity!(field, 1, field.coords)
    
    # Apply angular component regularity (components 2,3)
    for component in 2:3
        apply_vector_component_regularity!(field, component, field.coords) 
    end
    
    return field
end

"""
    apply_center_regularity_scalar!(field, coords)

Apply center regularity for scalar field at r=0.

Replaces grid values at center with spectrally computed regular values
using only l=0 modes that remain finite at r=0.
"""
function apply_center_regularity_scalar!(field::SphericalScalarField{T}, 
                                       coords::SphericalCoordinates{T}) where T<:Real
    grid_shape = size(field.data_grid)
    
    # Find center indices (r=0 corresponds to r_index=1 typically)
    center_r_idx = 1
    
    for phi_idx in 1:grid_shape[1]
        for theta_idx in 1:grid_shape[2]
            center_idx = CartesianIndex(phi_idx, theta_idx, center_r_idx)
            
            # Compute regular value at center using spectral expansion
            regular_value = compute_regular_value_at_center(field, center_idx, coords)
            field.data_grid[center_idx] = regular_value
        end
    end
end

"""
    apply_pole_regularity_scalar!(field, coords)

Apply pole regularity for scalar field at θ=0,π.

Replaces grid values at poles with spectrally computed regular values
using proper spherical harmonic evaluation.
"""
function apply_pole_regularity_scalar!(field::SphericalScalarField{T}, 
                                     coords::SphericalCoordinates{T}) where T<:Real
    grid_shape = size(field.data_grid)
    n_phi, n_theta, n_r = grid_shape
    
    # Apply at north pole (theta_idx = 1)
    theta_north = 1
    for phi_idx in 1:n_phi
        for r_idx in 1:n_r
            pole_idx = CartesianIndex(phi_idx, theta_north, r_idx)
            regular_value = compute_regular_value_at_pole(field, pole_idx, coords)
            field.data_grid[pole_idx] = regular_value
        end
    end
    
    # Apply at south pole (theta_idx = n_theta)  
    theta_south = n_theta
    for phi_idx in 1:n_phi
        for r_idx in 1:n_r
            pole_idx = CartesianIndex(phi_idx, theta_south, r_idx)
            regular_value = compute_regular_value_at_pole(field, pole_idx, coords)
            field.data_grid[pole_idx] = regular_value
        end
    end
end

"""
    apply_vector_component_regularity!(field, component, coords)

Apply regularity conditions to specific component of vector field.

Different components require different regularity treatments:
- Component 1 (radial): scalar-like regularity
- Components 2,3 (angular): spin-weighted treatment
"""
function apply_vector_component_regularity!(field::SphericalVectorField{T}, component::Int,
                                          coords::SphericalCoordinates{T}) where T<:Real
    if component == 1
        # Radial component: apply scalar-like regularity
        apply_radial_component_regularity!(field, coords)
    else
        # Angular components: apply spin-weighted regularity  
        apply_angular_component_regularity!(field, component, coords)
    end
end

"""
    compute_regular_value_at_center(field, idx, coords) -> Complex{T}

Compute regular value at center r=0 using spectral expansion.

Only l=0, m=0 modes contribute at center following dedalus approach:
- Sum over spectral modes with l=0, m=0
- Weight by Zernike polynomial values at center
- Apply Y_0^0 normalization factor
"""
function compute_regular_value_at_center(field::SphericalScalarField{T}, idx::CartesianIndex,
                                       coords::SphericalCoordinates{T}) where T<:Real
    # Compute regular value at center following dedalus approach
    spectral_coeffs = field.data_spectral
    n_coeffs = length(spectral_coeffs)
    
    center_value = Complex{T}(0)
    mode_idx = 0
    
    # Iterate over modes using typical ball domain parameters
    max_modes = Int(floor(n_coeffs^(1/3)))
    l_max = min(max_modes, 10)
    n_max = min(max_modes, 10)
    
    for l in 0:l_max
        for m in -l:l
            for n in 0:n_max
                mode_idx += 1
                
                if mode_idx <= n_coeffs
                    # Only l=0, m=0 modes contribute at r=0
                    if l == 0 && m == 0
                        zernike_at_center = evaluate_zernike_at_center(n, 0, T)
                        center_value += spectral_coeffs[mode_idx] * zernike_at_center
                    end
                else
                    break
                end
            end
            if mode_idx > n_coeffs
                break
            end
        end
        if mode_idx > n_coeffs
            break
        end
    end
    
    # Apply spherical harmonic normalization: Y_0^0(θ,φ) = 1/√(4π)
    center_value *= T(1) / sqrt(4 * π)
    
    return center_value
end

"""
    compute_regular_value_at_pole(field, idx, coords) -> Complex{T}

Compute regular value at poles θ=0,π using spherical harmonic expansion.

At poles, only m=0 modes contribute due to azimuthal symmetry:
- Sum over l modes with m=0
- Evaluate Y_l^0 at pole using Legendre polynomial values
- Account for alternating signs at south pole
"""
function compute_regular_value_at_pole(field::SphericalScalarField{T}, idx::CartesianIndex,
                                     coords::SphericalCoordinates{T}) where T<:Real
    # Compute regular value at poles following dedalus approach
    spectral_coeffs = field.data_spectral
    n_coeffs = length(spectral_coeffs)
    
    # Determine if we're at north pole (θ=0) or south pole (θ=π)
    theta_idx = length(idx.I) >= 2 ? idx.I[2] : 1
    
    # For typical spherical grids: θ=0 is at index 1, θ=π is at the last index
    n_theta = size(field.data_grid, 2)
    is_north_pole = (theta_idx == 1)
    is_south_pole = (theta_idx == n_theta)
    
    if !is_north_pole && !is_south_pole
        # Not at a pole, return current value
        return field.data_grid[idx]
    end
    
    pole_value = Complex{T}(0)
    
    # Sum over spectral modes, but only m=0 modes contribute at poles
    mode_idx = 0
    max_modes = Int(floor(n_coeffs^(1/3)))
    l_max = min(max_modes, 20)
    
    for l in 0:l_max
        for m in -l:l
            mode_idx += 1
            
            if mode_idx <= n_coeffs && m == 0
                # Only m=0 modes contribute at poles due to azimuthal symmetry
                coeff = spectral_coeffs[mode_idx]
                
                # Evaluate spherical harmonic Y_l^0 at the pole
                if is_north_pole
                    # P_l(1) = 1 for all l
                    legendre_value = T(1)
                else  # is_south_pole
                    # P_l(-1) = (-1)^l
                    legendre_value = T((-1)^l)
                end
                
                # Spherical harmonic normalization: √[(2l+1)/(4π)]
                norm_factor = sqrt((2*l + 1) / (4 * π))
                
                # Add contribution from this mode
                sph_harm_value = norm_factor * legendre_value
                pole_value += coeff * sph_harm_value
                
            elseif mode_idx > n_coeffs
                break
            end
        end
        
        if mode_idx > n_coeffs
            break
        end
    end
    
    return pole_value
end

"""
    apply_radial_component_regularity!(field, coords)

Apply regularity to radial component of vector field.

Radial component behaves like a scalar field for regularity purposes.
"""
function apply_radial_component_regularity!(field::SphericalVectorField{T}, 
                                          coords::SphericalCoordinates{T}) where T<:Real
    grid_shape = size(field.data_grid)
    
    # Apply center regularity for radial component
    center_r_idx = 1
    for phi_idx in 1:grid_shape[1]
        for theta_idx in 1:grid_shape[2]
            center_idx = CartesianIndex(phi_idx, theta_idx, center_r_idx)
            regular_value = compute_radial_regular_value(field, center_idx, coords)
            field.data_grid[center_idx, 1] = regular_value  # Component 1 is radial
        end
    end
    
    # Apply pole regularity for radial component
    n_phi, n_theta, n_r = grid_shape
    
    # North pole
    for phi_idx in 1:n_phi, r_idx in 1:n_r
        pole_idx = CartesianIndex(phi_idx, 1, r_idx)
        regular_value = compute_radial_regular_value(field, pole_idx, coords)
        field.data_grid[pole_idx, 1] = regular_value
    end
    
    # South pole
    for phi_idx in 1:n_phi, r_idx in 1:n_r
        pole_idx = CartesianIndex(phi_idx, n_theta, r_idx)
        regular_value = compute_radial_regular_value(field, pole_idx, coords)
        field.data_grid[pole_idx, 1] = regular_value
    end
end

"""
    apply_angular_component_regularity!(field, component, coords)

Apply regularity to angular components of vector field.

Angular components require spin-weighted spherical harmonic treatment.
"""
function apply_angular_component_regularity!(field::SphericalVectorField{T}, component::Int,
                                           coords::SphericalCoordinates{T}) where T<:Real
    grid_shape = size(field.data_grid)  
    n_phi, n_theta, n_r = grid_shape
    
    # Apply pole regularity for angular components
    # North pole
    for phi_idx in 1:n_phi, r_idx in 1:n_r
        pole_idx = CartesianIndex(phi_idx, 1, r_idx)
        regular_value = compute_angular_regular_value_at_pole(field, component, pole_idx, coords)
        field.data_grid[pole_idx, component] = regular_value
    end
    
    # South pole  
    for phi_idx in 1:n_phi, r_idx in 1:n_r
        pole_idx = CartesianIndex(phi_idx, n_theta, r_idx)
        regular_value = compute_angular_regular_value_at_pole(field, component, pole_idx, coords)
        field.data_grid[pole_idx, component] = regular_value
    end
end

"""
    compute_radial_regular_value(field, idx, coords) -> Complex{T}

Compute regular value for radial component of vector field.

Radial component follows scalar regularity: only l=0 modes at center.
"""
function compute_radial_regular_value(field::SphericalVectorField{T}, idx::CartesianIndex,
                                    coords::SphericalCoordinates{T}) where T<:Real
    # Extract radial component spectral data
    # Assuming spectral coefficients are organized as [mode][component]
    spectral_coeffs_radial = field.data_spectral[:, 1]  # Component 1 = radial
    
    # Create temporary scalar field for regularity computation
    temp_scalar = SphericalScalarField{T}(spectral_coeffs_radial, field.data_grid[:,:,:,1], coords)
    
    # Use scalar regularity computation
    return compute_regular_value_at_center(temp_scalar, idx, coords)
end

"""
    compute_angular_regular_value_at_pole(field, component, idx, coords) -> Complex{T}

Compute regular value for angular component at poles using spin-weighted harmonics.

Angular components (θ,φ) have spin weights s=±1 and require special treatment
at poles where only specific (l,m,s) combinations contribute.
"""
function compute_angular_regular_value_at_pole(field::SphericalVectorField{T}, component::Int,
                                             idx::CartesianIndex, coords::SphericalCoordinates{T}) where T<:Real
    # Extract angular component spectral data  
    spectral_coeffs_angular = field.data_spectral[:, component]
    
    n_coeffs = length(spectral_coeffs_angular)
    
    # Determine pole location
    theta_idx = idx.I[2]
    n_theta = size(field.data_grid, 2)
    is_north_pole = (theta_idx == 1)
    
    if theta_idx != 1 && theta_idx != n_theta
        # Not at pole, return current value
        return field.data_grid[idx, component]
    end
    
    # Spin weight for angular components
    spin_weight = component == 2 ? 1 : -1  # θ component: s=1, φ component: s=-1
    
    pole_value = Complex{T}(0)
    mode_idx = 0
    max_modes = Int(floor(n_coeffs^(1/3)))
    l_max = min(max_modes, 20)
    
    for l in 0:l_max
        for m in -l:l
            mode_idx += 1
            
            if mode_idx <= n_coeffs
                coeff = spectral_coeffs_angular[mode_idx]
                
                # Evaluate spin-weighted harmonic at pole
                sph_harm_value = evaluate_spin_weighted_harmonic_at_pole(l, m, spin_weight, is_north_pole, T)
                pole_value += coeff * sph_harm_value
                
            elseif mode_idx > n_coeffs
                break
            end
        end
        
        if mode_idx > n_coeffs
            break
        end
    end
    
    return pole_value
end