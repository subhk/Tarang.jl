"""
Field Operations and Utilities

Core operations for spherical fields including:
- Field creation and initialization
- Spectral coefficient access and manipulation
- Field system management
- Basic field utilities

This module contains functions that operate on field objects but don't involve
complex transformations, boundary conditions, or regularity conditions.
"""

"""
Create and register field in field system.
"""
function create_field(system::SphericalFieldSystem{T}, field_type::Symbol, name::String, 
                     args...; kwargs...) where T<:Real
    
    if field_type == :scalar
        field = SphericalScalarField{T}(name, system.domain; kwargs...)
    elseif field_type == :vector
        field = SphericalVectorField{T}(name, system.domain; kwargs...)
    elseif field_type == :tensor
        field = SphericalTensorField{T}(name, system.domain, args...; kwargs...)
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end
    
    system.fields[name] = field
    return field
end

"""
Get spectral mode coefficients for a specific (l,m) mode.
"""
function get_spectral_mode_coefficients(field::SphericalScalarField{T}, l::Int, m::Int) where T<:Real
    # Extract coefficients for the (l,m) mode from the spectral data
    # This assumes the spectral data is organized as (m, l, n)
    m_idx = m + field.layout_manager.ball_basis.l_max + 1  # Convert to 1-based indexing
    l_idx = l + 1
    
    return view(field.data_spectral, m_idx, l_idx, :)
end

"""
Set spectral coefficient for a specific (l,m,n) mode.
"""
function set_spectral_coefficient!(field::SphericalScalarField{T}, 
                                l::Int, m::Int, n::Int, value::Complex{T}) where T<:Real
    m_idx = m + field.layout_manager.ball_basis.l_max + 1
    l_idx = l + 1
    n_idx = n + 1
    
    field.data_spectral[m_idx, l_idx, n_idx] = value
end

"""
Initialize field with data or function.
"""
function initialize_field!(field::SphericalScalarField{T}, data_or_func) where T<:Real
    coords = field.domain.coords
    
    if isa(data_or_func, Function)
        # Initialize from function f(r,θ,φ)
        for idx in CartesianIndices(size(field.data_grid))
            r = coords.r_grid[idx]
            theta = coords.theta_grid[idx] 
            phi = coords.phi_grid[idx]
            
            field.data_grid[idx] = data_or_func(r, theta, phi)
        end
    else
        # Initialize from array data
        field.data_grid .= data_or_func
    end
    
    # Apply regularity conditions
    apply_regularity_conditions!(field)
end

function initialize_field!(field::SphericalVectorField{T}, data_or_func) where T<:Real
    coords = field.domain.coords
    
    if isa(data_or_func, Function)
        # Initialize from function returning (Fr, Fθ, Fφ)
        for idx in CartesianIndices(size(field.data_grid)[2:end])
            r = coords.r_grid[idx]
            theta = coords.theta_grid[idx]
            phi = coords.phi_grid[idx]
            
            components = data_or_func(r, theta, phi)
            
            for comp in 1:3
                field.data_grid[comp, idx] = components[comp]
            end
        end
    else
        # Initialize from array data
        field.data_grid .= data_or_func
    end
    
    # Apply regularity conditions
    apply_regularity_conditions!(field)
end

"""
Ensure field is in specific layout, transforming if necessary.
"""
function ensure_layout!(field::Union{SphericalScalarField{T}, SphericalVectorField{T}}, 
                       target_layout::SphericalLayout) where T<:Real
    current = field.layout_manager.current_layout[]
    
    if current != target_layout
        transform_layout!(field, current, target_layout)
        field.layout_manager.current_layout[] = target_layout
    end
end

"""
Get field values at specific coordinates.
"""
function get_field_value(field::SphericalScalarField{T}, r::T, theta::T, phi::T) where T<:Real
    # Ensure field is in grid layout for interpolation
    ensure_layout!(field, GRID_LAYOUT)
    
    # Simple trilinear interpolation (can be enhanced with spectral interpolation)
    coords = field.domain.coords
    
    # Find grid indices
    r_idx = searchsortedfirst(coords.r_grid, r)
    theta_idx = searchsortedfirst(coords.theta_grid, theta)  
    phi_idx = searchsortedfirst(coords.phi_grid, phi)
    
    # Clamp to valid indices
    r_idx = clamp(r_idx, 1, length(coords.r_grid))
    theta_idx = clamp(theta_idx, 1, length(coords.theta_grid))
    phi_idx = clamp(phi_idx, 1, length(coords.phi_grid))
    
    return field.data_grid[phi_idx, theta_idx, r_idx]
end

function get_field_value(field::SphericalVectorField{T}, r::T, theta::T, phi::T) where T<:Real
    # Ensure field is in grid layout for interpolation
    ensure_layout!(field, GRID_LAYOUT)
    
    coords = field.domain.coords
    
    # Find grid indices
    r_idx = searchsortedfirst(coords.r_grid, r)
    theta_idx = searchsortedfirst(coords.theta_grid, theta)
    phi_idx = searchsortedfirst(coords.phi_grid, phi)
    
    # Clamp to valid indices
    r_idx = clamp(r_idx, 1, length(coords.r_grid))
    theta_idx = clamp(theta_idx, 1, length(coords.theta_grid))
    phi_idx = clamp(phi_idx, 1, length(coords.phi_grid))
    
    return [field.data_grid[comp, phi_idx, theta_idx, r_idx] for comp in 1:3]
end

"""
Set field constant flag and update metadata.
"""
function set_constant!(field::Union{SphericalScalarField{T}, SphericalVectorField{T}}, 
                      is_constant::Bool) where T<:Real
    field.constant = is_constant
    
    if is_constant
        # For constant fields, we can optimize storage and operations
        field.regularity_conditions["pole_regularity"] = false
        field.regularity_conditions["center_regularity"] = false
    end
end

"""
Get field system summary information.
"""
function get_system_info(system::SphericalFieldSystem{T}) where T<:Real
    info = Dict{String, Any}(
        "domain_radius" => system.domain.coords.radius,
        "grid_size" => (system.domain.coords.nphi, system.domain.coords.ntheta, system.domain.coords.nr),
        "field_count" => length(system.fields),
        "current_layout" => system.global_layout[],
        "time" => system.time[],
        "iteration" => system.iteration[]
    )
    
    field_info = Dict{String, String}()
    for (name, field) in system.fields
        if isa(field, SphericalScalarField)
            field_info[name] = "scalar"
        elseif isa(field, SphericalVectorField)
            field_info[name] = "vector"
        elseif isa(field, SphericalTensorField)
            field_info[name] = "tensor($(field.tensor_rank))"
        end
    end
    info["fields"] = field_info
    
    return info
end

"""
Copy field data from one field to another.
"""
function copy_field_data!(dest::SphericalScalarField{T}, src::SphericalScalarField{T}) where T<:Real
    # Ensure both fields are in the same layout
    src_layout = src.layout_manager.current_layout[]
    ensure_layout!(dest, src_layout)
    
    if src_layout == GRID_LAYOUT
        dest.data_grid .= src.data_grid
    else
        dest.data_spectral .= src.data_spectral
    end
    
    # Copy metadata
    dest.constant = src.constant
    dest.scales .= src.scales
end

function copy_field_data!(dest::SphericalVectorField{T}, src::SphericalVectorField{T}) where T<:Real
    # Ensure both fields are in the same layout
    src_layout = src.layout_manager.current_layout[]
    ensure_layout!(dest, src_layout)
    
    if src_layout == GRID_LAYOUT
        dest.data_grid .= src.data_grid
    else
        dest.data_spectral .= src.data_spectral
    end
    
    # Copy representation and metadata
    dest.current_representation = src.current_representation
    dest.constant = src.constant
    dest.scales .= src.scales
end