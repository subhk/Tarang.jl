"""
    Field distributor utilities

This file contains local/global index helpers, decomposition validation, and
domain-size queries used by field storage management.
"""

"""
    validate_decomposition_convention(dist::Distributor, expected_convention::Symbol)

Validate that the distributor's decomposition convention matches the expected one.
Throws an error if there's a mismatch in MPI mode.

Arguments:
- dist: Distributor to validate
- expected_convention: :pencil_arrays (LAST dims) or :transposable_field (FIRST dims)

This function helps catch convention mismatches early, preventing silent data corruption.
"""
function validate_decomposition_convention(dist::Distributor, expected_convention::Symbol)
    if dist.size == 1
        return  # Serial mode - convention doesn't matter
    end

    actual_convention = dist.use_pencil_arrays ? :pencil_arrays : :transposable_field

    if expected_convention == :pencil_arrays && !dist.use_pencil_arrays
        error("Convention mismatch: Expected PencilArrays convention " *
              "(decompose LAST dims, per decomposed_axes) but Distributor has " *
              "use_pencil_arrays=false (TransposableField/FIRST dims). This would cause " *
              "data layout corruption in MPI mode.")
    elseif expected_convention == :transposable_field && dist.use_pencil_arrays
        error("Convention mismatch: Expected TransposableField convention " *
              "(decompose FIRST dims, per decomposed_axes) but Distributor has " *
              "use_pencil_arrays=true (PencilArrays/LAST dims). This would cause data " *
              "layout corruption in MPI mode.")
    end
end

"""
    Get the coordinate of a process in the specified mesh dimension.

    For a mesh (P₁, P₂, ..., Pₖ), the process with rank r has coordinates:
    (r % P₁, (r ÷ P₁) % P₂, ..., (r ÷ (P₁×P₂×...×Pₖ₋₁)) % Pₖ)

    `rank` defaults to this process's own rank; pass an explicit rank to ask
    the same question about a DIFFERENT process (e.g. a root rank computing
    the range it must send to each destination during a scatter).

    `dist` is untyped (reads only `.mesh`, and `.rank` for the default `rank`)
    so this can drive the TransposableField coordinate math from a duck-typed
    fake distributor in serial tests, matching `decomposed_axes`/`mesh_axis_for`.
    """
function get_process_coordinate(dist, dim::Int, rank::Int=dist.rank)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return 0
    end

    mesh = dist.mesh

    # Compute coordinate using column-major ordering (Fortran-style)
    # rank = coord[1] + mesh[1]*(coord[2] + mesh[2]*(coord[3] + ...))
    # So: coord[i] = (rank ÷ prod(mesh[1:i-1])) % mesh[i]
    stride = 1
    for i in 1:(dim-1)
        stride *= mesh[i]
    end

    coord = div(rank, stride) % mesh[dim]
    return coord
end

"""
    Get the local range [start, end] for a process in a given global axis.

    Arguments:
    - dist: Distributor with MPI decomposition info
    - global_size: Size of the global array in this axis
    - axis: Global axis index (1-based)
    - rank: process to compute the range for (defaults to this process's own
      rank). Pass an explicit rank to ask what ANOTHER rank owns — e.g. rank 0
      slicing out the piece it must send to each destination during a scatter.

    Returns:
    - (start_idx, end_idx) tuple with 1-based indices

    Note: which axes are decomposed comes from `decomposed_axes` — see its
    docstring for both conventions.
    """
function get_local_range(dist::Distributor, global_size::Int, axis::Int, rank::Int=dist.rank)
    if dist.size == 1 || dist.mesh === nothing || axis < 1 || axis > dist.dim
        return (1, global_size)
    end

    # `dist.dim` is this distributor's field dimensionality; get_local_range is
    # called with a global axis index into a field of that rank.
    mesh_axis = mesh_axis_for(dist, dist.dim, axis)
    if mesh_axis === nothing
        return (1, global_size)
    end

    n_procs = dist.mesh[mesh_axis]

    # Match PencilArrays' decomposition exactly on the PencilArrays path so the
    # returned range addresses the same slab PencilArrays actually owns (MPI-Cart
    # row-major coords + remainder-on-LAST). The legacy column-major /
    # remainder-on-FIRST formula below diverges on non-divisible or >=2D-mesh
    # decompositions; keep it only off the PencilArrays path (GPU /
    # TransposableField). Mirrors compute_local_shape / local_indices.
    # `pencil_local_range` reads the CALLING rank's own live MPI topology
    # coordinates, so it can only answer for `dist.rank` itself; an explicit
    # other rank falls through to the generic formula below.
    pr = (rank == dist.rank) ? pencil_local_range(dist, mesh_axis, n_procs, global_size) : nothing
    if pr !== nothing
        return (first(pr), last(pr))
    end

    proc_coord = get_process_coordinate(dist, mesh_axis, rank)

    base_size = div(global_size, n_procs)
    remainder = global_size % n_procs

    if proc_coord < remainder
        local_size = base_size + 1
        start_idx = proc_coord * (base_size + 1) + 1
    else
        local_size = base_size
        start_idx = remainder * (base_size + 1) + (proc_coord - remainder) * base_size + 1
    end

    end_idx = start_idx + local_size - 1
    return (start_idx, end_idx)
end

"""
    Convert a global index to a local index for this process.

    Returns nothing if the global index is not owned by this process.
    """
function global_to_local_index(dist::Distributor, global_idx::Int, axis::Int)
    start_idx, end_idx = get_local_range(dist, get_global_size(dist, axis), axis)

    if global_idx >= start_idx && global_idx <= end_idx
        return global_idx - start_idx + 1
    else
        return nothing
    end
end

"""
    Convert a local index to a global index.
    """
function local_to_global_index(dist::Distributor, local_idx::Int, global_size::Int, axis::Int)
    start_idx, _ = get_local_range(dist, global_size, axis)
    return start_idx + local_idx - 1
end

"""
    get_global_size(dist::Distributor, dim::Int)

Get the global size in a dimension. This method requires domain/basis information
to determine actual sizes. Without that context, it returns a conservative default.

For accurate global sizes, use one of the following methods instead:
- `get_global_size(dist, basis, dim)` - for a specific basis
- `get_global_size(dist, domain, dim)` - for a specific domain
- `get_global_grid_shape(dist, domain; scales=...)` - for full grid shape

# Arguments
- `dist`: The Distributor
- `dim`: Dimension index (1-based)

# Returns
- The global size in the specified dimension, or a default value if unknown
"""
function get_global_size(dist::Distributor, dim::Int)
    # Without domain/basis context, we cannot determine the actual global size.
    # Check if distributor has cached layout information that might help.
    if !isempty(dist.layouts)
        # Try to get size from cached layouts
        for (key, layout) in dist.layouts
            if hasfield(typeof(layout), :global_shape) && layout.global_shape !== nothing
                if dim <= length(layout.global_shape)
                    return layout.global_shape[dim]
                end
            end
        end
    end

    # Fallback: Check if pencil_cache has any entries with shape info
    if !isempty(dist.pencil_cache)
        for (cache_key, _) in dist.pencil_cache
            # Pencil cache keys are (global_shape, decomp_dims, dtype).
            global_shape = first(cache_key)
            if isa(global_shape, Tuple) && dim <= length(global_shape)
                return global_shape[dim]
            end
        end
    end

    @warn "get_global_size called without domain context; returning default. " *
          "Use get_global_size(dist, basis, dim) or get_global_size(dist, domain, dim) for accurate sizes." maxlog=1
    return 64
end

"""
    get_global_size(dist::Distributor, basis::Basis, dim::Int=1)

Get the global size for a specific basis dimension.

# Arguments
- `dist`: The Distributor (unused but kept for API consistency)
- `basis`: The Basis to query
- `dim`: Dimension within the basis (default 1, as most bases are 1D)

# Returns
- The global size (number of grid/coefficient points) for this basis
"""
function get_global_size(dist::Distributor, basis::Basis, dim::Int=1)
    if dim != 1
        @warn "Most bases are 1D; dim=$dim requested but using basis size"
    end
    return basis.meta.size
end

"""
    get_global_size(dist::Distributor, domain::Domain, dim::Int)

Get the global size in a specific dimension of the domain.

# Arguments
- `dist`: The Distributor (unused but kept for API consistency)
- `domain`: The Domain containing the bases
- `dim`: Dimension index (1-based)

# Returns
- The global size in the specified dimension
"""
function get_global_size(dist::Distributor, domain::Domain, dim::Int)
    if dim < 1 || dim > length(domain.bases)
        throw(BoundsError("Dimension $dim out of range for domain with $(length(domain.bases)) dimensions"))
    end
    return domain.bases[dim].meta.size
end

"""
    get_global_sizes(dist::Distributor, domain::Domain)

Get all global sizes for a domain as a tuple.

# Arguments
- `dist`: The Distributor
- `domain`: The Domain containing the bases

# Returns
- Tuple of global sizes for each dimension
"""
function get_global_sizes(dist::Distributor, domain::Domain)
    nb = length(domain.bases); return ntuple(i -> domain.bases[i].meta.size, nb)
end
