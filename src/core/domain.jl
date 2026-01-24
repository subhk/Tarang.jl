"""
Domain class definition
"""

using LinearAlgebra
using MPI
using OrderedCollections: OrderedDict

"""
    gauss_legendre_weights(N)

Compute Gauss-Legendre quadrature weights for N points on [-1, 1].
Uses the Newton-Raphson method to find roots of the Legendre polynomial,
then computes weights from the derivative formula.
"""
function gauss_legendre_weights(N::Int)
    if N < 1
        throw(ArgumentError("Number of Gauss-Legendre points must be at least 1, got $N"))
    end

    # Special case: N=1 has single point at z=0 with weight 2
    if N == 1
        return [2.0]
    end

    # Compute Gauss-Legendre weights on [-1, 1]
    # Using Newton-Raphson iteration to find roots, then compute weights
    w = zeros(N)

    # Use symmetry: only need to compute first half
    m = div(N + 1, 2)

    for i in 1:m
        # Initial guess using Chebyshev approximation
        z = cos(π * (i - 0.25) / (N + 0.5))

        # Newton-Raphson iteration to find root z
        for _ in 1:100
            p1 = 1.0
            p2 = 0.0

            # Recurrence for Legendre polynomial P_N(z)
            for j in 1:N
                p3 = p2
                p2 = p1
                p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
            end

            # Derivative: P'_N(z) = N * (z*P_N - P_{N-1}) / (z² - 1)
            # Guard against z ≈ ±1 (should not happen for interior GL points)
            denom = z^2 - 1
            if abs(denom) < 1e-14
                denom = copysign(1e-14, denom)
            end
            pp = N * (z * p1 - p2) / denom

            z_old = z
            z = z_old - p1 / pp

            if abs(z - z_old) < 1e-15
                break
            end
        end

        # Compute weight: w_i = 2 / ((1 - x_i²) * [P'_N(x_i)]²)
        p1 = 1.0
        p2 = 0.0
        for j in 1:N
            p3 = p2
            p2 = p1
            p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
        end
        # Guard against 1 - z² ≈ 0 (should not happen for interior GL points)
        one_minus_z2 = 1 - z^2
        if abs(one_minus_z2) < 1e-14
            one_minus_z2 = 1e-14
        end
        denom = z^2 - 1
        if abs(denom) < 1e-14
            denom = copysign(1e-14, denom)
        end
        pp = N * (z * p1 - p2) / denom
        w[i] = 2.0 / (one_minus_z2 * pp^2)
        w[N + 1 - i] = w[i]
    end

    return w
end

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
    grid_coordinates::Dict{String, AbstractArray}  # Cached grid coordinates on device (CPU or GPU)
    integration_weights_cache::Dict{String, AbstractArray}  # Cached integration weights on device
    performance_stats::DomainPerformanceStats
    attribute_cache::Dict{Symbol, Any}

    # GPU support: Domain uses the architecture from its Distributor
    # Grid coordinates and integration weights are stored on the appropriate device
    function Domain(dist::Distributor, bases::Tuple{Vararg{Basis}})
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

        domain = new(dist, tuple(sorted_bases...), total_dim, grid_coords, weights_cache, perf_stats, attribute_cache)

        plan_transforms!(dist, domain)

        return domain
    end
end

# Convenience constructor for vararg bases
function Domain(dist::Distributor, bases::Vararg{Basis})
    return Domain(dist, tuple(bases...))
end

# Architecture helper functions for Domain
"""
    architecture(domain::Domain)

Get the architecture (CPU or GPU) from the domain's distributor.
"""
architecture(domain::Domain) = domain.dist.architecture

"""
    is_gpu(domain::Domain)

Check if the domain is configured for GPU.
"""
is_gpu(domain::Domain) = is_gpu(domain.dist.architecture)

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
                # Handle both indexable (Vector, Tuple) and scalar constant values
                value = meta_const isa AbstractVector ? meta_const[subaxis + 1] :
                        meta_const isa Tuple ? meta_const[subaxis + 1] :
                        meta_const
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
                # Handle both indexable (Vector, Tuple) and scalar dependence values
                value = meta_dep isa AbstractVector ? meta_dep[subaxis + 1] :
                        meta_dep isa Tuple ? meta_dep[subaxis + 1] :
                        meta_dep
                dep_flags[first_axis + subaxis + 1] = value
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
    return Domain(domain.dist, tuple(bases_vec...))
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
    """Calculate domain volume as product of interval lengths for all bases."""
    vol = 1.0
    for basis in domain.bases
        interval_length = basis.meta.bounds[2] - basis.meta.bounds[1]
        if interval_length <= 0
            @warn "Non-positive interval length $(interval_length) for basis $(basis.meta.element_label), using absolute value"
            interval_length = abs(interval_length)
            if interval_length == 0
                interval_length = 1.0  # Fallback for degenerate case
            end
        end
        vol *= interval_length
    end
    return vol
end

function global_shape(domain::Domain, layout_name::Symbol=:g)
    """Get global shape for domain in specified layout"""
    if layout_name == :g  # Grid layout
        return tuple([basis.meta.size for basis in domain.bases]...)
    elseif layout_name == :c  # Coefficient layout
        return coefficient_shape(domain)
    else
        throw(ArgumentError("Unknown layout: $layout_name"))
    end
end

function coefficient_shape(domain::Domain)
    """
    Get coefficient space shape for domain.

    For RealFourier bases, the coefficient array has size div(N, 2) + 1 (complex).
    For other bases (ComplexFourier, Chebyshev, Legendre), size is the same as grid space.
    """
    shape = Int[]
    for basis in domain.bases
        if isa(basis, RealFourier)
            # RealFourier: rfft output has size N/2 + 1
            push!(shape, div(basis.meta.size, 2) + 1)
        else
            # Other bases: same size in coefficient space
            push!(shape, basis.meta.size)
        end
    end
    return tuple(shape...)
end

function local_shape(domain::Domain, layout_name::Symbol=:g)
    """Get local shape for domain in specified layout"""
    if layout_name == :g
        layout = get_layout(domain.dist, domain.bases)
        return layout.local_shape
    elseif layout_name == :c
        cshape = coefficient_shape(domain)
        if domain.dist.size == 1 || domain.dist.mesh === nothing
            return cshape
        end
        return compute_local_shape(domain.dist, cshape)
    else
        throw(ArgumentError("Unknown layout: $layout_name"))
    end
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
        size = basis.meta.size
        L = basis.meta.bounds[2] - basis.meta.bounds[1]

        if size < 1
            @warn "Invalid basis size $size for $(basis.meta.element_label), using spacing 1.0"
            push!(spacings, 1.0)
        elseif isa(basis, RealFourier) || isa(basis, ComplexFourier)
            # Uniform spacing for Fourier: dx = L / N
            push!(spacings, L / size)
        elseif isa(basis, ChebyshevT)
            # Non-uniform spacing for Chebyshev (approximate average)
            # For size == 1, spacing is the full interval
            dx = size > 1 ? L / (size - 1) : L
            push!(spacings, dx)
        else
            # Uniform spacing: dx = L / N
            push!(spacings, L / size)
        end
    end
    return spacings
end

function integration_weights(domain::Domain; on_device::Bool=true)
    """
    Get integration weights for each basis.

    Arguments:
    - domain: The Domain object
    - on_device: If true (default), return arrays on the domain's architecture (CPU or GPU).
                 If false, always return CPU arrays.

    Returns:
    - Vector{AbstractArray} of integration weights for each basis
    """

    start_time = time()
    arch = architecture(domain)
    weights = Vector{AbstractArray}()

    for basis in domain.bases
        basis_name = basis.meta.element_label
        # Include architecture in cache key when on_device=true
        arch_suffix = on_device ? "_$(typeof(arch))" : "_CPU"
        cache_key = "weights_$(typeof(basis))_$(basis_name)_$(basis.meta.size)_$(basis.meta.bounds)$(arch_suffix)"

        # Check cache first
        if haskey(domain.integration_weights_cache, cache_key)
            domain.performance_stats.cache_hits += 1
            push!(weights, domain.integration_weights_cache[cache_key])
            continue
        end
        domain.performance_stats.cache_misses += 1

        # Compute weights on CPU first
        w = get_integration_weights(basis)

        # Move to device if requested and domain is on GPU
        if on_device && is_gpu(domain)
            w = on_architecture(arch, w)
        end

        # Cache the weights
        domain.integration_weights_cache[cache_key] = w
        push!(weights, w)
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

function get_grid_coordinates(domain::Domain; on_device::Bool=true)
    """
    Get grid coordinates for all bases.

    Arguments:
    - domain: The Domain object
    - on_device: If true (default), return arrays on the domain's architecture (CPU or GPU).
                 If false, always return CPU arrays.

    Returns:
    - Dict{String, AbstractArray} mapping coordinate names to coordinate arrays
    """

    start_time = time()
    arch = architecture(domain)
    coordinates = Dict{String, AbstractArray}()

    for basis in domain.bases
        coord_name = basis.meta.element_label
        # Include architecture in cache key when on_device=true
        arch_suffix = on_device ? "_$(typeof(arch))" : "_CPU"
        cache_key = "coords_$(typeof(basis))_$(coord_name)_$(basis.meta.size)_$(basis.meta.bounds)$(arch_suffix)"

        if haskey(domain.grid_coordinates, cache_key)
            domain.performance_stats.cache_hits += 1
            coordinates[coord_name] = domain.grid_coordinates[cache_key]
            continue
        end
        domain.performance_stats.cache_misses += 1

        # Compute coordinates on CPU first
        native_grid = _native_grid(basis, 1.0)
        coord_values = if isa(basis, FourierBasis)
            native_grid
        elseif basis.meta.COV !== nothing
            problem_coord(basis.meta.COV, native_grid)
        else
            _problem_coord_fallback(basis, native_grid)
        end

        # Move to device if requested and domain is on GPU
        if on_device && is_gpu(domain)
            coord_values = on_architecture(arch, coord_values)
        end

        domain.grid_coordinates[cache_key] = coord_values
        coordinates[coord_name] = coord_values
    end

    domain.performance_stats.total_time += time() - start_time
    domain.performance_stats.coordinate_generations += 1

    return coordinates
end

function create_meshgrid(domain::Domain; on_device::Bool=true)
    """
    Create meshgrid arrays for multi-dimensional domains.

    Supports both CPU and GPU: when domain is on GPU, returns GPU arrays.
    The `repeat` and `reshape` operations work on both CPU and GPU arrays.

    Arguments:
    - domain: The Domain object
    - on_device: If true (default), return arrays on the domain's architecture (CPU or GPU).
                 If false, always return CPU arrays.

    Returns:
    - Dict{String, AbstractArray} mapping coordinate names to meshgrid arrays
    """

    grid_coords = get_grid_coordinates(domain; on_device=on_device)

    if length(domain.bases) == 1
        coord_name = domain.bases[1].meta.element_label
        return Dict(coord_name => grid_coords[coord_name])
    elseif length(domain.bases) == 2
        x_name = domain.bases[1].meta.element_label
        y_name = domain.bases[2].meta.element_label

        x_coords = grid_coords[x_name]
        y_coords = grid_coords[y_name]

        nx, ny = length(x_coords), length(y_coords)

        # Both arrays should have shape (nx, ny) for consistency
        # reshape and repeat work on both CPU and GPU arrays
        X = repeat(reshape(x_coords, nx, 1), 1, ny)
        Y = repeat(reshape(y_coords, 1, ny), nx, 1)

        return Dict(x_name => X, y_name => Y)
    elseif length(domain.bases) == 3
        x_name = domain.bases[1].meta.element_label
        y_name = domain.bases[2].meta.element_label
        z_name = domain.bases[3].meta.element_label

        x_coords = grid_coords[x_name]
        y_coords = grid_coords[y_name]
        z_coords = grid_coords[z_name]

        nx, ny, nz = length(x_coords), length(y_coords), length(z_coords)

        # reshape and repeat work on both CPU and GPU arrays
        X = repeat(reshape(x_coords, nx, 1, 1), 1, ny, nz)
        Y = repeat(reshape(y_coords, 1, ny, 1), nx, 1, nz)
        Z = repeat(reshape(z_coords, 1, 1, nz), nx, ny, 1)

        return Dict(x_name => X, y_name => Y, z_name => Z)
    else
        throw(ArgumentError("Meshgrid not implemented for $(length(domain.bases))D domains"))
    end
end

# Alias for backward compatibility
domain_volume(domain::Domain) = volume(domain)

function log_domain_performance(domain::Domain)
    """Log domain performance statistics"""

    # Only log on rank 0 to avoid spamming output
    rank = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
    rank != 0 && return nothing

    stats = domain.performance_stats

    @info "Domain performance:"
    @info "  Coordinate generations: $(stats.coordinate_generations)"
    @info "  Weight computations: $(stats.weight_computations)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"

    return nothing
end

function get_domain_memory_info(domain::Domain)
    """
    Return memory information for domains.

    For CPU domains, returns system memory info.
    For GPU domains, returns GPU memory info if CUDA is available,
    otherwise returns default placeholder values.

    Returns a NamedTuple with:
    - architecture: The domain architecture type (CPU or GPU)
    - total_memory: Total memory available
    - available_memory: Memory currently available
    - used_memory: Memory currently in use
    - domain_memory: Memory used by domain cached arrays
    - memory_utilization: Fraction of total memory in use
    """
    arch = architecture(domain)

    # Estimate domain memory usage from cached arrays
    domain_mem = 0
    for (_, arr) in domain.grid_coordinates
        domain_mem += sizeof(arr)
    end
    for (_, arr) in domain.integration_weights_cache
        domain_mem += sizeof(arr)
    end

    # Get memory info based on architecture
    # For GPU, the CUDA extension will override default_memory_info
    mem = default_memory_info()
    total_mem = mem.total
    used_mem = mem.used

    return (
        architecture = typeof(arch),
        total_memory = total_mem,
        available_memory = mem.available,
        used_memory = used_mem,
        domain_memory = domain_mem,
        memory_utilization = total_mem > 0 ? used_mem / total_mem : 0.0
    )
end

# Iteration interface
Base.iterate(domain::Domain) = iterate(domain.bases)
Base.iterate(domain::Domain, state) = iterate(domain.bases, state)
Base.length(domain::Domain) = length(domain.bases)
Base.getindex(domain::Domain, i::Int) = domain.bases[i]

# ============================================================================
# Exports
# ============================================================================

# Export types
export DomainPerformanceStats, Domain

# Export domain property functions
export bases_by_axis, full_bases, bases_by_coord, dealias, constant, nonconstant,
       mode_dependence, substitute_basis, get_basis, get_basis_subaxis, get_coord,
       enumerate_unique_bases, dim, clear_domain_cache!

# Export domain geometry functions
export volume, global_shape, local_shape, coefficient_shape, get_pencil, grid_spacing,
       integration_weights, domain_volume, get_grid_coordinates, create_meshgrid

# Export domain query functions
export is_compound, has_basis, basis_names

# Export performance and diagnostics
export log_domain_performance, get_domain_memory_info

# Export helper functions
export gauss_legendre_weights
