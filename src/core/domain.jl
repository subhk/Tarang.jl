"""
Domain class definition
"""

using LinearAlgebra
using OrderedCollections: OrderedDict

mutable struct DomainPerformanceStats
    total_time::Float64
    coordinate_generations::Int
    weight_computations::Int
    cache_hits::Int
    cache_misses::Int
    
    function DomainPerformanceStats()
        new(0.0, 0, 0, 0, 0)
    end
end

mutable struct Domain
    dist::Distributor
    bases::Tuple{Vararg{Basis}}
    dim::Int
    device_type::Symbol
    grid_coordinates::Dict{String, AbstractArray}  # Cached grid coordinates on device
    integration_weights_cache::Dict{String, AbstractArray}  # Cached integration weights on device
    performance_stats::DomainPerformanceStats
    attribute_cache::Dict{Symbol, Any}

    function Domain(dist::Distributor, bases::Tuple{Vararg{Basis}}; device_type::Symbol=CPU_DEVICE, device=nothing)
        # Filter out nothing bases and remove duplicates
        filtered_bases = filter(b -> b !== nothing, bases)
        unique_bases = unique(filtered_bases)

        # Check for overlapping bases (same coordinate name within same coordinate system)
        # Two bases overlap if they operate on the same coordinate
        coord_keys = [(basis.meta.coordsys, basis.meta.element_label) for basis in unique_bases]
        if length(Set(coord_keys)) < length(coord_keys)
            throw(ArgumentError("Overlapping bases specified"))
        end

        # Sort bases by axis index
        sorted_bases = sort(unique_bases, by=b -> get_basis_axis(dist, b))

        # Calculate total dimension
        total_dim = sum(basis.meta.dim for basis in sorted_bases)

        grid_coords = Dict{String, AbstractArray}()
        weights_cache = Dict{String, AbstractArray}()
        perf_stats = DomainPerformanceStats()
        attribute_cache = Dict{Symbol, Any}()

        new(dist, tuple(sorted_bases...), total_dim, device_type, grid_coords, weights_cache, perf_stats, attribute_cache)
    end
end

# Convenience constructors
function Domain(dist::Distributor, bases::Vararg{Basis}; device_type::Symbol=CPU_DEVICE, device=nothing)
    return Domain(dist, tuple(bases...); device_type=device_type, device=device)
end

@inline function _domain_cached_get!(builder::Function, domain::Domain, key::Symbol)
    cache = domain.attribute_cache
    return get!(cache, key) do
        builder()
    end
end

function bases_by_axis(domain::Domain)
    """Ordered mapping from global axis index to basis."""
    _domain_cached_get!(domain, :bases_by_axis) do
        axes = OrderedDict{Int, Basis}()
        for basis in domain.bases
            first_axis = get_basis_axis(domain.dist, basis)
            for subaxis in 0:(basis.meta.dim - 1)
                axes[first_axis + subaxis] = basis
            end
        end
        axes
    end
end

function full_bases(domain::Domain)
    """Tuple mapping each distributor axis to its active basis (or nothing)."""
    _domain_cached_get!(domain, :full_bases) do
        full = Vector{Union{Basis, Nothing}}(undef, domain.dist.dim)
        fill!(full, nothing)
        for basis in domain.bases
            first_axis = get_basis_axis(domain.dist, basis)
            for subaxis in 0:(basis.meta.dim - 1)
                full[first_axis + subaxis + 1] = basis
            end
        end
        tuple(full...)
    end
end

function bases_by_coord(domain::Domain)
    """Ordered mapping from coordinates/coordinate systems to bases."""
    _domain_cached_get!(domain, :bases_by_coord) do
        mapping = OrderedDict{Any, Union{Basis, Nothing}}()
        for coord in domain.dist.coords
            if coord.coordsys isa CartesianCoordinates
                mapping[coord] = nothing
            else
                mapping[coord.coordsys] = nothing
            end
        end
        for basis in domain.bases
            mapping[basis.meta.coordsys] = basis
            for coord in coords(basis.meta.coordsys)
                mapping[coord] = basis
            end
        end
        mapping
    end
end

function dealias(domain::Domain)
    """Tuple of dealiasing factors per axis."""
    _domain_cached_get!(domain, :dealias) do
        factors = ones(Float64, domain.dist.dim)
        for basis in domain.bases
            first_axis = get_basis_axis(domain.dist, basis)
            dealias_meta = basis.meta.dealias
            for subaxis in 0:(basis.meta.dim - 1)
                value = dealias_meta isa AbstractVector ? dealias_meta[subaxis + 1] :
                        dealias_meta isa Tuple ? dealias_meta[subaxis + 1] :
                        dealias_meta
                factors[first_axis + subaxis + 1] = value
            end
        end
        tuple(factors...)
    end
end

function constant(domain::Domain)
    """Tuple indicating which axes are constant."""
    _domain_cached_get!(domain, :constant) do
        const_flags = falses(domain.dist.dim)
        for basis in domain.bases
            first_axis = get_basis_axis(domain.dist, basis)
            for subaxis in 0:(basis.meta.dim - 1)
                meta_const = basis.meta.constant
                value = meta_const[subaxis + 1]
                const_flags[first_axis + subaxis + 1] = value
            end
        end
        tuple(const_flags...)
    end
end

function nonconstant(domain::Domain)
    """Tuple inverse of constant axes."""
    _domain_cached_get!(domain, :nonconstant) do
        tuple((!flag for flag in constant(domain))...)
    end
end

function mode_dependence(domain::Domain)
    """Tuple of mode-dependence flags per axis."""
    _domain_cached_get!(domain, :mode_dependence) do
        dep_flags = trues(domain.dist.dim)
        for basis in domain.bases
            first_axis = get_basis_axis(domain.dist, basis)
            for subaxis in 0:(basis.meta.dim - 1)
                meta_dep = basis.meta.subaxis_dependence
                dep_flags[first_axis + subaxis + 1] = meta_dep[subaxis + 1]
            end
        end
        tuple(dep_flags...)
    end
end

function substitute_basis(domain::Domain, old_basis::Basis, new_basis::Basis)
    """Return new domain with one basis substituted."""
    bases_vec = collect(domain.bases)
    idx = findfirst(==(old_basis), bases_vec)
    if idx !== nothing
        deleteat!(bases_vec, idx)
    end
    push!(bases_vec, new_basis)
    return Domain(domain.dist, tuple(bases_vec...); device_type=domain.device_type)
end

function get_basis(domain::Domain, coords)
    """Retrieve basis associated with coordinate or axis index."""
    axis = coords isa Int ? coords : get_axis(domain.dist, coords)
    full = full_bases(domain)
    idx = axis + 1
    if idx < 1 || idx > length(full)
        throw(ArgumentError("Axis $axis out of bounds for domain of dimension $(length(full))"))
    end
    return full[idx]
end

function get_basis_subaxis(domain::Domain, coord::Coordinate)
    """Return subaxis index for coordinate within its basis."""
    axis = get_axis(domain.dist, coord)
    for basis in domain.bases
        first_axis = get_basis_axis(domain.dist, basis)
        if first_axis <= axis < first_axis + basis.meta.dim
            return axis - first_axis
        end
    end
    throw(ArgumentError("Coordinate $(coord.name) not found in any basis"))
end

function get_coord(domain::Domain, name::AbstractString)
    """Retrieve coordinate by name across all bases."""
    for basis in domain.bases
        for coord in coords(basis.meta.coordsys)
            if coord.name == name
                return coord
            end
        end
    end
    throw(ArgumentError("Coordinate name $name not found in domain"))
end

function enumerate_unique_bases(domain::Domain)
    """Iterator of (axis, basis) pairs with unique bases."""
    _domain_cached_get!(domain, :enumerate_unique_bases) do
        seen = OrderedSet()
        pairs = Vector{Tuple{Int, Union{Basis, Nothing}}}()
        for (axis, basis) in enumerate(full_bases(domain))
            actual_axis = axis - 1  # 0-based axes
            if basis === nothing || !(basis in seen)
                push!(pairs, (actual_axis, basis))
                if basis !== nothing
                    push!(seen, basis)
                end
            end
        end
        pairs
    end
end

function dim(domain::Domain)
    """Effective dimension counting non-constant axes."""
    _domain_cached_get!(domain, :dim_cached) do
        sum(flag ? 1 : 0 for flag in nonconstant(domain))
    end
end

function clear_domain_cache!(domain::Domain)
    empty!(domain.attribute_cache)
    empty!(domain.grid_coordinates)
    empty!(domain.integration_weights_cache)
    return domain
end

function volume(domain::Domain)
    """Calculate domain volume"""
    vol = 1.0
    for basis in domain.bases
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            # For Fourier basis, volume is the period
            vol *= basis.meta.bounds[2] - basis.meta.bounds[1]
        elseif isa(basis, ChebyshevT) || isa(basis, ChebyshevU) || isa(basis, Legendre)
            # For polynomial bases, volume is the interval length
            vol *= basis.meta.bounds[2] - basis.meta.bounds[1]
        else
            # Default case
            vol *= basis.meta.bounds[2] - basis.meta.bounds[1]
        end
    end
    return vol
end

function global_shape(domain::Domain, layout_name::Symbol=:g)
    """Get global shape for domain in specified layout"""
    if layout_name == :g  # Grid layout
        return tuple([basis.meta.size for basis in domain.bases]...)
    elseif layout_name == :c  # Coefficient layout  
        return tuple([basis.meta.size for basis in domain.bases]...)
    else
        throw(ArgumentError("Unknown layout: $layout_name"))
    end
end

function local_shape(domain::Domain, layout_name::Symbol=:g)
    """Get local shape for domain in specified layout"""
    layout = get_layout(domain.dist, domain.bases)
    return layout.local_shape
end

function get_pencil(domain::Domain, decomp_index::Int=1)
    """Get pencil array for domain"""
    gshape = global_shape(domain)
    return create_pencil(domain.dist, gshape, decomp_index)
end

# Grid and coefficient utilities
function grid_spacing(domain::Domain)
    """Calculate grid spacing for each dimension"""
    spacings = Float64[]
    for basis in domain.bases
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            dx = L / basis.meta.size
            push!(spacings, dx)
        elseif isa(basis, ChebyshevT)
            # Non-uniform spacing for Chebyshev
            push!(spacings, 2.0 / (basis.meta.size - 1))  # Approximate
        else
            # Uniform spacing
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            dx = L / basis.meta.size
            push!(spacings, dx)
        end
    end
    return spacings
end

function integration_weights(domain::Domain)
    """Get integration weights for each basis"""

    start_time = time()
    weights = []

    for (i, basis) in enumerate(domain.bases)
        basis_name = basis.meta.element_label
        cache_key = "weights_$(basis_name)_$(basis.meta.size)"

        # Check cache first
        if haskey(domain.integration_weights_cache, cache_key)
            push!(weights, Array(domain.integration_weights_cache[cache_key]))
            continue
        end

        # Compute weights
        w = if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            # Uniform weights for Fourier
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            dx = L / basis.meta.size
            device_fill(dx, (basis.meta.size,), domain)
        elseif isa(basis, ChebyshevT)
            # Clenshaw-Curtis weights for Chebyshev
            N = basis.meta.size
            w_cpu = ones(N)
            w_cpu[1] *= 0.5
            w_cpu[end] *= 0.5
            # Scale by interval length
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            w_cpu .*= L / (N - 1)
            w_cpu
        elseif isa(basis, Legendre)
            # Gauss-Legendre weights
            N = basis.meta.size
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            device_fill(L/N, (N,), domain)  # Placeholder - uniform weights
        else
            # Default uniform weights
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            dx = L / basis.meta.size
            device_fill(dx, (basis.meta.size,), domain)
        end

        # Cache the weights
        domain.integration_weights_cache[cache_key] = w
        push!(weights, Array(w))
    end

    # Update performance statistics
    domain.performance_stats.total_time += time() - start_time
    domain.performance_stats.weight_computations += 1
    
    return weights
end

# Domain queries
function is_compound(domain::Domain)
    """Check if domain is compound (multiple bases)"""
    return length(domain.bases) > 1
end

function has_basis(domain::Domain, basis::Basis)
    """Check if domain contains a specific basis"""
    return basis in domain.bases
end

function basis_names(domain::Domain)
    """Get names of all bases in domain"""
    return [basis.meta.element_label for basis in domain.bases]
end

function get_grid_coordinates(domain::Domain)
    """Get grid coordinates for all bases (CPU only)."""

    start_time = time()
    coordinates = Dict{String, Vector{Float64}}()

    for basis in domain.bases
        coord_name = basis.meta.element_label
        cache_key = "coords_$(coord_name)_$(basis.meta.size)"

        if haskey(domain.grid_coordinates, cache_key)
            coordinates[coord_name] = domain.grid_coordinates[cache_key]
            continue
        end

        coords = if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            dx = L / basis.meta.size
            collect(range(basis.meta.bounds[1], length=basis.meta.size, step=dx))
        elseif isa(basis, ChebyshevT)
            N = basis.meta.size
            a, b = basis.meta.bounds
            cheb_points = [cos(π * (2*k - 1) / (2*N)) for k in N:-1:1]
            [(b - a) * (p + 1) / 2 + a for p in cheb_points]
        elseif isa(basis, Legendre)
            collect(range(basis.meta.bounds[1], basis.meta.bounds[2], length=basis.meta.size))
        else
            collect(range(basis.meta.bounds[1], basis.meta.bounds[2], length=basis.meta.size))
        end

        domain.grid_coordinates[cache_key] = coords
        coordinates[coord_name] = coords
    end

    domain.performance_stats.total_time += time() - start_time
    domain.performance_stats.coordinate_generations += 1

    return coordinates
end

function create_meshgrid(domain::Domain)
    """Create meshgrid arrays for multi-dimensional domains (CPU)."""

    coords = get_grid_coordinates(domain)

    if length(domain.bases) == 1
        coord_name = domain.bases[1].meta.element_label
        return Dict(coord_name => coords[coord_name])
    elseif length(domain.bases) == 2
        x_name = domain.bases[1].meta.element_label
        y_name = domain.bases[2].meta.element_label

        x_coords = coords[x_name]
        y_coords = coords[y_name]

        X = repeat(reshape(x_coords, 1, :), length(y_coords), 1)'
        Y = repeat(reshape(y_coords, :, 1), 1, length(x_coords))

        return Dict(x_name => X, y_name => Y)
    elseif length(domain.bases) == 3
        x_name = domain.bases[1].meta.element_label
        y_name = domain.bases[2].meta.element_label
        z_name = domain.bases[3].meta.element_label

        x_coords = coords[x_name]
        y_coords = coords[y_name]
        z_coords = coords[z_name]

        nx, ny, nz = length(x_coords), length(y_coords), length(z_coords)

        X = repeat(reshape(x_coords, nx, 1, 1), 1, ny, nz)
        Y = repeat(reshape(y_coords, 1, ny, 1), nx, 1, nz)
        Z = repeat(reshape(z_coords, 1, 1, nz), nx, ny, 1)

        return Dict(x_name => X, y_name => Y, z_name => Z)
    else
        throw(ArgumentError("Meshgrid not implemented for $(length(domain.bases))D domains"))
    end
end

function domain_volume(domain::Domain)
    """Calculate domain volume (CPU)."""
    vol = 1.0
    for basis in domain.bases
        interval_length = basis.meta.bounds[2] - basis.meta.bounds[1]
        vol *= interval_length
    end
    return vol
end

function log_domain_performance(domain::Domain)
    """Log domain performance statistics"""

    stats = domain.performance_stats

    @info "Domain performance:"
    @info "  Coordinate generations: $(stats.coordinate_generations)"
    @info "  Weight computations: $(stats.weight_computations)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"
end

function get_domain_memory_info(domain::Domain)
    """Return placeholder memory info (CPU-only)."""
    mem = default_memory_info()
    return (
        total_memory = mem.total,
        available_memory = mem.available,
        used_memory = mem.used,
        domain_memory = 0,
        memory_utilization = 0.0
    )
end

# Iteration interface
Base.iterate(domain::Domain) = iterate(domain.bases)
Base.iterate(domain::Domain, state) = iterate(domain.bases, state)
Base.length(domain::Domain) = length(domain.bases)
Base.getindex(domain::Domain, i::Int) = domain.bases[i]
