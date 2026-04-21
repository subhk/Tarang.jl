"""Shared domain and shape extraction helpers for flow diagnostics and spectra."""

"""
    get_domain_size(domain) -> Tuple

Extract physical domain size (extent) from domain object.

Returns a tuple of physical lengths for each dimension, computed from
the bounds of each basis in the domain.

# Arguments
- `domain`: Domain object containing bases with bounds information

# Returns
- Tuple of physical extents `(L1, L2, ...)` for each dimension

# Example
```julia
# For a 3D periodic box [0, 2π] × [0, 2π] × [0, 2π]
Lx, Ly, Lz = get_domain_size(domain)  # Returns (2π, 2π, 2π)
```
"""
function get_domain_size(domain)
    if domain === nothing
        @warn "No domain provided, returning default size"
        return (2π, 2π, 2π)
    end

    # Check if domain has bases
    if !hasfield(typeof(domain), :bases) || domain.bases === nothing
        @warn "Domain has no bases, returning default size"
        return (2π, 2π, 2π)
    end

    sizes = Float64[]

    for basis in domain.bases
        if basis === nothing
            push!(sizes, 2π)  # Default for missing basis
            continue
        end

        # Extract bounds from basis metadata
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :bounds)
            bounds = basis.meta.bounds
            if bounds !== nothing && length(bounds) >= 2
                extent = Float64(bounds[2] - bounds[1])
                push!(sizes, extent)
            else
                push!(sizes, 2π)  # Default
            end
        elseif hasfield(typeof(basis), :bounds)
            # Direct bounds field
            bounds = basis.bounds
            if bounds !== nothing && length(bounds) >= 2
                extent = Float64(bounds[2] - bounds[1])
                push!(sizes, extent)
            else
                push!(sizes, 2π)
            end
        else
            push!(sizes, 2π)  # Default for unknown basis type
        end
    end

    if isempty(sizes)
        return (2π, 2π, 2π)
    end

    return Tuple(sizes)
end

"""
    get_domain_bounds(domain) -> Vector{Tuple{Float64, Float64}}

Extract physical domain bounds from domain object.

Returns a vector of (min, max) tuples for each dimension.

# Arguments
- `domain`: Domain object containing bases with bounds information

# Returns
- Vector of `(min, max)` tuples for each dimension
"""
function get_domain_bounds(domain)
    if domain === nothing || !hasfield(typeof(domain), :bases) || domain.bases === nothing
        return [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    bounds_list = Tuple{Float64, Float64}[]

    for basis in domain.bases
        if basis === nothing
            push!(bounds_list, (0.0, 2π))
            continue
        end

        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :bounds)
            bounds = basis.meta.bounds
            if bounds !== nothing && length(bounds) >= 2
                push!(bounds_list, (Float64(bounds[1]), Float64(bounds[2])))
            else
                push!(bounds_list, (0.0, 2π))
            end
        elseif hasfield(typeof(basis), :bounds)
            bounds = basis.bounds
            if bounds !== nothing && length(bounds) >= 2
                push!(bounds_list, (Float64(bounds[1]), Float64(bounds[2])))
            else
                push!(bounds_list, (0.0, 2π))
            end
        else
            push!(bounds_list, (0.0, 2π))
        end
    end

    if isempty(bounds_list)
        return [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    return bounds_list
end

"""Extract local Fourier shape from velocity field"""
function get_fourier_shape(velocity::VectorField, fourier_axes::Vector{Int})
    first_component = velocity.components[1]
    ensure_layout!(first_component, :c)
    return size(get_coeff_data(first_component))
end
