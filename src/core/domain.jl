"""
Domain class definition
"""

# LinearAlgebra and MPI already imported in Tarang.jl
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

        # IMPORTANT: MPI parallelization only supports Fourier domains
        # RealFourier is only supported with PencilArrays (which handles RFFT)
        validate_mpi_fourier_only(unique_bases, dist.size; use_pencil_arrays=dist.use_pencil_arrays)

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

# ── Domain cache ─────────────────────────────────────────────────────────────
# Building a Domain is expensive: the inner constructor runs `plan_transforms!`
# (FFT planning) and allocates several caches. A Domain is fully determined by
# its `(dist, bases)`, yet `ScalarField` previously built a fresh one on EVERY
# construction — so direct field arithmetic (`u + 2v`) and the interpreted RHS
# path re-planned transforms for every temporary, ~215 KB per result field.
#
# Cache Domains by the identity of their `(dist, bases)`. The cached Domain holds
# `dist` and `bases` alive, so their `objectid`s stay valid while cached (no
# stale-key reuse). Sharing is safe: a Domain's mutable state is idempotent
# attribute/grid caches plus transform plans that depend only on `(dist, bases)`
# — nothing field-specific (per-field scales live on the field, not the Domain).
# Mirrors the FFT/DCT plan caches. Use `clear_domain_cache!()` to reset.
const _DOMAIN_CACHE = Dict{Tuple{UInt, Tuple{Vararg{UInt}}}, Domain}()
const _DOMAIN_CACHE_LOCK = ReentrantLock()

"""Get a cached Domain for `(dist, bases)`, building and caching on first use."""
function get_or_build_domain(dist::Distributor, bases::Tuple{Vararg{Basis}})
    key = (objectid(dist), map(objectid, bases))
    lock(_DOMAIN_CACHE_LOCK) do
        get!(() -> Domain(dist, bases), _DOMAIN_CACHE, key)
    end
end

"""Clear the cached Domains (thread-safe)."""
clear_domain_cache!() = lock(() -> empty!(_DOMAIN_CACHE), _DOMAIN_CACHE_LOCK)

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

"""Ordered mapping from global axis index to basis."""
function bases_by_axis(domain::Domain)
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

"""Tuple mapping each distributor axis to its active basis (or nothing)."""
function full_bases(domain::Domain)
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

"""Ordered mapping from coordinates/coordinate systems to bases."""
function bases_by_coord(domain::Domain)
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

"""
    dealias(domain::Domain) -> Tuple

Tuple of per-axis dealiasing factors read from `basis.meta.dealias`.

⚠ CURRENTLY UNUSED: this function aggregates per-basis dealias metadata
but `NonlinearEvaluator` uses its own global `dealiasing_factor`
(default 3/2) for the padded multiply path, not this per-basis value.
The per-basis `dealias=...` argument on basis constructors is inert —
it's stored and displayed but does not drive the padding factor.
See docs/src/pages/dealiasing.md for details.

Kept for metadata/display purposes; delete if no future use emerges.
"""
function dealias(domain::Domain)
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

"""Tuple indicating which axes are constant."""
function constant(domain::Domain)
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

"""Tuple inverse of constant axes."""
function nonconstant(domain::Domain)
    _domain_cached_get!(domain, :nonconstant) do
        map(!, constant(domain))
    end
end

"""Tuple of mode-dependence flags per axis."""
function mode_dependence(domain::Domain)
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

"""Return new domain with one basis substituted."""
function substitute_basis(domain::Domain, old_basis::Basis, new_basis::Basis)
    bases_vec = collect(domain.bases)
    idx = findfirst(==(old_basis), bases_vec)
    if idx !== nothing
        deleteat!(bases_vec, idx)
    end
    push!(bases_vec, new_basis)
    return Domain(domain.dist, tuple(bases_vec...))
end

"""Retrieve basis associated with coordinate or axis index."""
function get_basis(domain::Domain, coords)
    axis = coords isa Int ? coords : get_axis(domain.dist, coords)
    full = full_bases(domain)
    idx = axis + 1
    if idx < 1 || idx > length(full)
        throw(ArgumentError("Axis $axis out of bounds for domain of dimension $(length(full))"))
    end
    return full[idx]
end

"""Return subaxis index for coordinate within its basis."""
function get_basis_subaxis(domain::Domain, coord::Coordinate)
    axis = get_axis(domain.dist, coord)
    for basis in domain.bases
        first_axis = get_basis_axis(domain.dist, basis)
        if first_axis <= axis < first_axis + basis.meta.dim
            return axis - first_axis
        end
    end
    throw(ArgumentError("Coordinate $(coord.name) not found in any basis"))
end

"""Retrieve coordinate by name across all bases."""
function get_coord(domain::Domain, name::AbstractString)
    for basis in domain.bases
        for coord in coords(basis.meta.coordsys)
            if coord.name == name
                return coord
            end
        end
    end
    throw(ArgumentError("Coordinate name $name not found in domain"))
end

"""Iterator of (axis, basis) pairs with unique bases."""
function enumerate_unique_bases(domain::Domain)
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

"""Effective dimension counting non-constant axes."""
function dim(domain::Domain)
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

"""Calculate domain volume as product of interval lengths for all bases."""
function volume(domain::Domain)
    vol = 1.0
    for basis in domain.bases
        interval_length = basis.meta.bounds[2] - basis.meta.bounds[1]
        if interval_length == 0
            throw(ArgumentError(
                "Zero-length interval for basis '$(basis.meta.element_label)': " *
                "bounds = $(basis.meta.bounds). Cannot compute domain volume."))
        elseif interval_length < 0
            throw(ArgumentError(
                "Reversed interval for basis '$(basis.meta.element_label)': " *
                "bounds = $(basis.meta.bounds) (lower > upper). Check basis construction."))
        end
        vol *= interval_length
    end
    return vol
end

"""Get global shape for domain in specified layout"""
function global_shape(domain::Domain, layout_name::Symbol=:g)
    if layout_name == :g  # Grid layout
        nb = length(domain.bases); return ntuple(i -> domain.bases[i].meta.size, nb)
    elseif layout_name == :c  # Coefficient layout
        # Use context-aware shape: in MPI+PencilFFTs mode, only the first
        # RealFourier axis uses RFFT (N/2+1), others use FFT (full N).
        return get_coefficient_shape_for_context(domain, domain.dist)
    else
        throw(ArgumentError("Unknown layout: $layout_name"))
    end
end

# ---------------------------------------------------------------------------
# Coefficient shape — shared rule for serial and MPI
# ---------------------------------------------------------------------------
#
# The rule is the same in both modes, for subtly different reasons:
#
#   • MPI+PencilFFTs: `PencilFFTPlan` can only apply RFFT to the FIRST
#     Fourier axis. Subsequent RealFourier axes must use FFT (full size N).
#
#   • Serial FFTW: `_fourier_forward` dispatches on input element type.
#     The first transform produces complex output (rfft: real → complex);
#     after that, any subsequent RealFourier transform sees complex input
#     and falls through to `FFTW.fft` (full size N), not `FFTW.rfft`.
#
# So in both modes: halve the first Fourier axis only if it's RealFourier,
# and leave every subsequent Fourier axis at full size. Non-Fourier bases
# (Chebyshev, Legendre) don't change shape under their transforms and are
# always at full size.
#
# This was previously wrong in serial mode — `coefficient_shape` halved
# EVERY RealFourier axis, which produced a pre-allocated `coeff_data`
# buffer of the wrong shape for multi-Fourier fields. The old allocating
# forward_transform! path masked the issue by always replacing the
# buffer; the in-place refactor made it visible (first call had to
# reallocate, losing the pre-allocation). Unifying the logic closes this
# gap and lets the in-place fast path fire from the very first call.

"""
    _fourier_output_size(basis, is_first_fourier)

Return the size along this axis in coefficient space, given whether
this is the first Fourier axis in the transform chain. See the comment
block above for the rule.
"""
@inline function _fourier_output_size(basis::Basis, is_first_fourier::Bool)
    if isa(basis, RealFourier) && is_first_fourier
        return div(basis.meta.size, 2) + 1  # rfft halves the first Fourier axis
    else
        return basis.meta.size               # fft (or non-Fourier): full size
    end
end

"""
    _coefficient_shape_impl(domain::Domain) -> Tuple

Compute the coefficient-space shape for a domain using the shared rule
(see the comment block above). Callable from both serial and MPI wrappers.
"""
function _coefficient_shape_impl(domain::Domain)
    # Find the first Fourier axis (RealFourier or ComplexFourier) — this
    # is the only one that can use RFFT (or the only one whose shape can
    # change under the transform).
    first_fourier_idx = nothing
    for (i, basis) in enumerate(domain.bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            first_fourier_idx = i
            break
        end
    end

    shape = Int[]
    for (i, basis) in enumerate(domain.bases)
        is_first = (first_fourier_idx !== nothing && i == first_fourier_idx)
        push!(shape, _fourier_output_size(basis, is_first))
    end
    return tuple(shape...)
end

"""
    coefficient_shape(domain::Domain) -> Tuple

Get the coefficient-space shape for a domain. In both serial and MPI
modes, only the first Fourier axis is halved (and only if it's
RealFourier). See the `_coefficient_shape_impl` comment block for the
rationale.
"""
coefficient_shape(domain::Domain) = _coefficient_shape_impl(domain)

"""
    coefficient_shape_mpi(domain::Domain) -> Tuple

Retained as a distinct public name for callers that want to express
"I specifically mean the MPI-compatible shape." Returns the same result
as `coefficient_shape` — the rule is identical in both modes.
"""
coefficient_shape_mpi(domain::Domain) = _coefficient_shape_impl(domain)

"""
    get_coefficient_shape_for_context(domain, dist) -> Tuple

Return the coefficient-space shape for a domain. Both the serial and
MPI wrappers compute the same result now (see `_coefficient_shape_impl`
comment block), so this just dispatches to either — retained for
call-site clarity.
"""
function get_coefficient_shape_for_context(domain::Domain, dist::Distributor)
    if dist.size > 1 && dist.use_pencil_arrays
        return coefficient_shape_mpi(domain)
    else
        return coefficient_shape(domain)
    end
end

"""Get local shape for domain in specified layout"""
function local_shape(domain::Domain, layout_name::Symbol=:g)
    if layout_name == :g
        layout = get_layout(domain.dist, domain.bases)
        return layout.local_shape
    elseif layout_name == :c
        # IMPORTANT: Use get_coefficient_shape_for_context to handle MPI/PencilFFTs RFFT rules
        # In MPI mode with PencilFFTs, only the first RealFourier axis uses RFFT (N/2+1),
        # subsequent RealFourier axes use FFT (full size N).
        cshape = get_coefficient_shape_for_context(domain, domain.dist)
        if domain.dist.size == 1 || domain.dist.mesh === nothing
            return cshape
        end
        return compute_local_shape(domain.dist, cshape)
    else
        throw(ArgumentError("Unknown layout: $layout_name"))
    end
end

"""Get pencil array for domain"""
function get_pencil(domain::Domain, decomp_index::Int=1)
    gshape = global_shape(domain)
    return create_pencil(domain.dist, gshape, decomp_index)
end

# Grid and coefficient utilities
"""Calculate grid spacing for each dimension"""
function grid_spacing(domain::Domain)
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
            # Chebyshev-Gauss-Lobatto grid: x_k = -cos(πk/(N-1))
            # Minimum spacing occurs near endpoints: Δx_min ≈ L·(1 - cos(π/(N-1)))/2
            # Using minimum spacing is critical for CFL stability estimates.
            if size <= 1
                dx = L
            elseif size == 2
                dx = L
            else
                dx = L * (1 - cos(π / (size - 1))) / 2
            end
            push!(spacings, dx)
        else
            # Uniform spacing: dx = L / N
            push!(spacings, L / size)
        end
    end
    return spacings
end

"""
    Get integration weights for each basis.

    Arguments:
    - domain: The Domain object
    - on_device: If true (default), return arrays on the domain's architecture (CPU or GPU).
                 If false, always return CPU arrays.

    Returns:
    - Vector{AbstractArray} of integration weights for each basis
    """
function integration_weights(domain::Domain; on_device::Bool=true)

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
"""Check if domain is compound (multiple bases)"""
function is_compound(domain::Domain)
    return length(domain.bases) > 1
end

"""Check if domain contains a specific basis"""
function has_basis(domain::Domain, basis::Basis)
    return basis in domain.bases
end

"""Get names of all bases in domain"""
function basis_names(domain::Domain)
    return [basis.meta.element_label for basis in domain.bases]
end

"""
    Get grid coordinates for all bases.

    Arguments:
    - domain: The Domain object
    - on_device: If true (default), return arrays on the domain's architecture (CPU or GPU).
                 If false, always return CPU arrays.

    Returns:
    - Dict{String, AbstractArray} mapping coordinate names to coordinate arrays
    """
function get_grid_coordinates(domain::Domain; on_device::Bool=true)

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
function create_meshgrid(domain::Domain; on_device::Bool=true)

    grid_coords = get_grid_coordinates(domain; on_device=on_device)
    ndim = length(domain.bases)

    if ndim == 1
        coord_name = domain.bases[1].meta.element_label
        return Dict(coord_name => grid_coords[coord_name])
    end

    # General N-dimensional meshgrid using reshape + repeat
    names = [b.meta.element_label for b in domain.bases]
    coords = [grid_coords[n] for n in names]
    sizes = [length(c) for c in coords]

    result = Dict{String, AbstractArray}()
    for (d, name) in enumerate(names)
        # Shape: 1 in all dims except d where it's sizes[d]
        shape = ntuple(i -> i == d ? sizes[i] : 1, ndim)
        # Repeat: sizes[i] in all dims except d where it's 1
        reps = ntuple(i -> i == d ? 1 : sizes[i], ndim)
        result[name] = repeat(reshape(coords[d], shape...), reps...)
    end
    return result
end

# Alias for backward compatibility
domain_volume(domain::Domain) = volume(domain)

"""Log domain performance statistics"""
function log_domain_performance(domain::Domain)

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
function get_domain_memory_info(domain::Domain)
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
