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
function get_domain_size(domain::Union{Nothing, Domain})
    if domain === nothing
        @warn "No domain provided, returning default size" maxlog=1
        return (2π, 2π, 2π)
    end

    # `Domain.bases` is `Tuple{Vararg{Basis}}` and `BasisMeta.bounds` is
    # `Tuple{Float64, Float64}`, so there is nothing here to be defensive about.
    # This used to be a `hasfield` chain — `:bases` on the domain, then
    # `.meta.bounds`, then a direct `.bounds`, then a bare `else` — each arm with
    # its own 2π default, and every arm but the first unreachable.
    isempty(domain.bases) && return (2π, 2π, 2π)
    return Tuple(Float64(b.meta.bounds[2] - b.meta.bounds[1]) for b in domain.bases)
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
function get_domain_bounds(domain::Union{Nothing, Domain})
    (domain === nothing || isempty(domain.bases)) &&
        return [(0.0, 2π), (0.0, 2π), (0.0, 2π)]

    # See `get_domain_size` for why none of this needs guarding.
    return [(Float64(b.meta.bounds[1]), Float64(b.meta.bounds[2])) for b in domain.bases]
end

"""Extract local Fourier shape from velocity field"""
function get_fourier_shape(velocity::VectorField, fourier_axes::Vector{Int})
    first_component = velocity.components[1]
    return size(coeff_data!(first_component))
end
