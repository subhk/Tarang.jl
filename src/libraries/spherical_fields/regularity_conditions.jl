"""
Regularity Conditions for Spherical Fields

Handles regularity conditions at singular points in spherical coordinates:
- Pole regularity: Proper behavior at θ = 0, π
- Center regularity: Proper behavior at r = 0  
- Component regularity: Vector/tensor component transformations

Based on dedalus sphere handling of coordinate singularities.
"""

"""
Apply regularity conditions to field.
"""
function apply_regularity_conditions!(field::SphericalScalarField{T}) where T<:Real
    if field.regularity_conditions["pole_regularity"]
        apply_pole_regularity!(field)
    end
    
    if field.regularity_conditions["center_regularity"]
        apply_center_regularity!(field)
    end
    
    return field
end

function apply_regularity_conditions!(field::SphericalVectorField{T}) where T<:Real
    if field.regularity_conditions["pole_regularity"]
        apply_pole_regularity!(field)
    end
    
    if field.regularity_conditions["center_regularity"]
        apply_center_regularity!(field)
    end
    
    # Apply component-specific regularity
    if haskey(field.regularity_conditions, "component_regularity")
        apply_component_regularity!(field)
    end
    
    return field
end

"""
Apply pole regularity conditions at θ = 0, π.
For scalar fields: ensure continuity across poles.
"""
function apply_pole_regularity!(field::SphericalScalarField{T}) where T<:Real
    coords = field.domain.coords
    
    # At θ = 0 (north pole): field should be independent of φ
    if coords.ntheta > 1
        theta_idx_north = 1
        for r_idx in 1:coords.nr
            # Average over all φ values at north pole
            avg_value = mean(field.data_grid[:, theta_idx_north, r_idx])
            field.data_grid[:, theta_idx_north, r_idx] .= avg_value
        end
        
        # At θ = π (south pole): similar treatment
        theta_idx_south = coords.ntheta
        for r_idx in 1:coords.nr
            avg_value = mean(field.data_grid[:, theta_idx_south, r_idx])
            field.data_grid[:, theta_idx_south, r_idx] .= avg_value
        end
    end
end

"""
Apply pole regularity for vector fields.
Vector components transform differently at poles.
"""
function apply_pole_regularity!(field::SphericalVectorField{T}) where T<:Real
    coords = field.domain.coords
    
    if coords.ntheta > 1
        # At poles, apply proper vector component transformations
        apply_vector_pole_conditions!(field, 1)  # North pole
        apply_vector_pole_conditions!(field, coords.ntheta)  # South pole
    end
end

"""
Apply center regularity conditions at r = 0.
Fields must be finite and well-behaved at the center.
"""
function apply_center_regularity!(field::SphericalScalarField{T}) where T<:Real
    coords = field.domain.coords
    
    if coords.nr > 1
        r_idx_center = 1  # Assuming r = 0 is at index 1
        
        # At r = 0, scalar field should be constant in θ,φ
        for theta_idx in 1:coords.ntheta, phi_idx in 1:coords.nphi
            # Use spectral projection to determine center value
            center_value = compute_center_value(field, theta_idx, phi_idx, T)
            field.data_grid[phi_idx, theta_idx, r_idx_center] = center_value
        end
    end
end

function apply_center_regularity!(field::SphericalVectorField{T}) where T<:Real
    coords = field.domain.coords
    
    if coords.nr > 1
        r_idx_center = 1
        
        # Vector components have different regularity requirements at r = 0
        for comp in 1:3
            apply_component_center_regularity!(field, comp, r_idx_center, T)
        end
    end
end

"""
Apply component-specific regularity for vector fields.
"""
function apply_component_regularity!(field::SphericalVectorField{T}) where T<:Real
    # Transform to regularity representation if needed
    if field.current_representation != :regularity
        transform_to_regularity_representation!(field)
    end
    
    # Apply regularity conditions in regularity space
    coords = field.domain.coords
    
    for comp in 1:3
        if field.regularity_conditions["component_regularity"][comp]
            apply_single_component_regularity!(field, comp)
        end
    end
    
    # Transform back to coordinate representation
    if field.current_representation == :regularity
        transform_from_regularity_representation!(field)
    end
end

"""
Transform vector field to regularity representation.
"""
function transform_to_regularity_representation!(field::SphericalVectorField{T}) where T<:Real
    # Apply Q_forward transformation: spin → regularity
    for idx in CartesianIndices(size(field.data_grid)[2:end])
        spin_vector = [field.data_spin[comp, idx] for comp in 1:3]
        reg_vector = field.Q_forward * spin_vector
        
        for comp in 1:3
            field.data_regularity[comp, idx] = reg_vector[comp]
        end
    end
    
    field.current_representation = :regularity
end

"""
Transform vector field from regularity representation.
"""
function transform_from_regularity_representation!(field::SphericalVectorField{T}) where T<:Real
    # Apply Q_backward transformation: regularity → spin
    for idx in CartesianIndices(size(field.data_grid)[2:end])
        reg_vector = [field.data_regularity[comp, idx] for comp in 1:3]
        spin_vector = field.Q_backward * reg_vector
        
        for comp in 1:3
            field.data_spin[comp, idx] = spin_vector[comp]
        end
    end
    
    # Then transform to coordinate representation if needed
    transform_spin_to_coordinate!(field)
end

"""
Transform from spin to coordinate representation.
"""
function transform_spin_to_coordinate!(field::SphericalVectorField{T}) where T<:Real
    for idx in CartesianIndices(size(field.data_grid)[2:end])
        spin_vector = [field.data_spin[comp, idx] for comp in 1:3]
        coord_vector = field.U_backward * spin_vector
        
        for comp in 1:3
            field.data_grid[comp, idx] = coord_vector[comp]
        end
    end
    
    field.current_representation = :coordinate
end

"""
Apply vector component conditions at poles.
"""
function apply_vector_pole_conditions!(field::SphericalVectorField{T}, theta_idx::Int) where T<:Real
    coords = field.domain.coords
    
    for r_idx in 1:coords.nr
        # At poles, vector components must satisfy specific symmetry conditions
        # This is a simplified implementation
        
        # θ-component should be continuous
        avg_theta = mean(field.data_grid[2, :, theta_idx, r_idx])
        field.data_grid[2, :, theta_idx, r_idx] .= avg_theta
        
        # φ-component typically vanishes at poles
        field.data_grid[3, :, theta_idx, r_idx] .= 0
        
        # r-component should be continuous
        avg_r = mean(field.data_grid[1, :, theta_idx, r_idx])
        field.data_grid[1, :, theta_idx, r_idx] .= avg_r
    end
end

"""
Compute appropriate center value for scalar field.
"""
function compute_center_value(field::SphericalScalarField{T}, theta_idx::Int, phi_idx::Int, ::Type{T}) where T<:Real
    # Use nearby grid points to estimate center value
    coords = field.domain.coords
    
    if coords.nr > 2
        # Linear extrapolation from nearby points
        r1_val = field.data_grid[phi_idx, theta_idx, 2]
        r2_val = field.data_grid[phi_idx, theta_idx, 3]
        r1 = coords.r_grid[2]
        r2 = coords.r_grid[3]
        
        # Extrapolate to r = 0
        center_val = r1_val - (r2_val - r1_val) * r1 / (r2 - r1)
        return center_val
    else
        # Use existing value
        return field.data_grid[phi_idx, theta_idx, 1]
    end
end

"""
Apply center regularity for specific vector component.
"""
function apply_component_center_regularity!(field::SphericalVectorField{T}, comp::Int, r_idx_center::Int, ::Type{T}) where T<:Real
    coords = field.domain.coords
    
    if comp == 1  # Radial component
        # Radial component can be non-zero at center
        for theta_idx in 1:coords.ntheta, phi_idx in 1:coords.nphi
            center_value = compute_vector_center_value(field, comp, theta_idx, phi_idx, T)
            field.data_grid[comp, phi_idx, theta_idx, r_idx_center] = center_value
        end
    else  # Angular components (θ, φ)
        # Angular components typically vanish at r = 0
        field.data_grid[comp, :, :, r_idx_center] .= 0
    end
end

"""
Compute center value for vector component.
"""
function compute_vector_center_value(field::SphericalVectorField{T}, comp::Int, theta_idx::Int, phi_idx::Int, ::Type{T}) where T<:Real
    coords = field.domain.coords
    
    if coords.nr > 2 && comp == 1  # Only radial component
        # Extrapolate from nearby points
        r1_val = field.data_grid[comp, phi_idx, theta_idx, 2]
        r2_val = field.data_grid[comp, phi_idx, theta_idx, 3]
        r1 = coords.r_grid[2]
        r2 = coords.r_grid[3]
        
        center_val = r1_val - (r2_val - r1_val) * r1 / (r2 - r1)
        return center_val
    else
        return Complex{T}(0)  # Angular components vanish at center
    end
end

"""
Apply regularity to single vector component.
"""
function apply_single_component_regularity!(field::SphericalVectorField{T}, comp::Int) where T<:Real
    coords = field.domain.coords
    
    # Apply component-specific regularity conditions based on Dedalus implementation
    # Each component (r=1, θ=2, φ=3) has different regularity requirements
    
    if comp == 1  # Radial component (v_r)
        # Radial component regularity at poles: v_r should be continuous across poles
        # At θ = 0, π: enforce phi-independence 
        for r_idx in 1:coords.nr
            # North pole (θ = 0)
            if coords.ntheta > 1
                avg_north = mean(view(field.data_regularity, comp, :, 1, r_idx))
                field.data_regularity[comp, :, 1, r_idx] .= avg_north
                
                # South pole (θ = π) 
                avg_south = mean(view(field.data_regularity, comp, :, coords.ntheta, r_idx))
                field.data_regularity[comp, :, coords.ntheta, r_idx] .= avg_south
            end
        end
        
        # Center regularity at r = 0: v_r should vanish at center for regularity
        if coords.nr > 1
            field.data_regularity[comp, :, :, 1] .= zero(T)
        end
        
    elseif comp == 2  # Theta component (v_θ)
        # Theta component regularity at poles: v_θ should vanish at poles
        for r_idx in 1:coords.nr
            if coords.ntheta > 1
                # Must vanish at both poles due to coordinate singularity
                field.data_regularity[comp, :, 1, r_idx] .= zero(T)
                field.data_regularity[comp, :, coords.ntheta, r_idx] .= zero(T)
            end
        end
        
        # Center regularity: finite at r = 0 but may need special handling
        # depending on the specific harmonic content
        
    elseif comp == 3  # Phi component (v_φ) 
        # Phi component regularity at poles: v_φ should vanish at poles
        for r_idx in 1:coords.nr
            if coords.ntheta > 1
                # Must vanish at both poles due to coordinate singularity
                field.data_regularity[comp, :, 1, r_idx] .= zero(T)
                field.data_regularity[comp, :, coords.ntheta, r_idx] .= zero(T)
            end
        end
        
        # Center regularity: finite at r = 0
    end
    
    # Additional spectral regularity enforcement in coefficient space
    # This ensures the spectral representation respects regularity constraints
    if field.current_representation == :regularity
        # Apply spectral filtering to remove modes that violate regularity
        for n_idx in 1:coords.nr
            for l_idx in 1:coords.ntheta
                for m_idx in 1:coords.nphi
                    # Component-specific spectral constraints based on symmetry
                    if comp == 1  # v_r: even parity across equator for l-m even modes
                        if (l_idx + m_idx) % 2 == 1 && abs(m_idx - coords.nphi÷2) > l_idx
                            field.data_spectral[comp, m_idx, l_idx, n_idx] = zero(Complex{T})
                        end
                    elseif comp == 2 || comp == 3  # v_θ, v_φ: odd parity constraints
                        if (l_idx + m_idx) % 2 == 0 && abs(m_idx - coords.nphi÷2) > l_idx
                            field.data_spectral[comp, m_idx, l_idx, n_idx] = zero(Complex{T})
                        end
                    end
                end
            end
        end
    end
end