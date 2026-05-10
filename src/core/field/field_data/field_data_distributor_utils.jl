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
        error("Convention mismatch: Expected PencilArrays convention (decompose LAST dims) " *
              "but Distributor has use_pencil_arrays=false (TransposableField/FIRST dims). " *
              "This would cause data layout corruption in MPI mode.")
    elseif expected_convention == :transposable_field && dist.use_pencil_arrays
        error("Convention mismatch: Expected TransposableField convention (decompose FIRST dims) " *
              "but Distributor has use_pencil_arrays=true (PencilArrays/LAST dims). " *
              "This would cause data layout corruption in MPI mode.")
    end
end

"""
    Get the coordinate of this process in the specified mesh dimension.

    For a mesh (P₁, P₂, ..., Pₖ), the process with rank r has coordinates:
    (r % P₁, (r ÷ P₁) % P₂, ..., (r ÷ (P₁×P₂×...×Pₖ₋₁)) % Pₖ)
    """
function get_process_coordinate(dist::Distributor, dim::Int)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return 0
    end

    mesh = dist.mesh
    rank = dist.rank

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
    Get the local range [start, end] for this process in a given global axis.

    Arguments:
    - dist: Distributor with MPI decomposition info
    - global_size: Size of the global array in this axis
    - axis: Global axis index (1-based)

    Returns:
    - (start_idx, end_idx) tuple with 1-based indices

    Note: Respects dist.use_pencil_arrays:
    - PencilArrays convention: decompose LAST ndims_mesh dimensions
    - TransposableField convention: decompose FIRST ndims_mesh dimensions
    """
function get_local_range(dist::Distributor, global_size::Int, axis::Int)
    if dist.size == 1 || dist.mesh === nothing || axis < 1 || axis > dist.dim
        return (1, global_size)
    end

    mesh_dim = length(dist.mesh)

    # Determine which mesh dimension (if any) corresponds to this axis
    mesh_axis = nothing
    if dist.use_pencil_arrays
        # PencilArrays convention: decompose LAST mesh_dim dimensions
        # Axis dist.dim is mesh[mesh_dim], axis dist.dim-1 is mesh[mesh_dim-1], etc.
        decomp_start = max(1, dist.dim - mesh_dim + 1)
        if axis >= decomp_start
            mesh_axis = axis - decomp_start + 1
        end
    else
        # TransposableField convention: decompose FIRST mesh_dim dimensions
        # Axis 1 is mesh[1], axis 2 is mesh[2], etc.
        if axis <= mesh_dim
            mesh_axis = axis
        end
    end

    # If axis is not decomposed, return full range
    if mesh_axis === nothing || mesh_axis < 1 || mesh_axis > mesh_dim
        return (1, global_size)
    end

    n_procs = dist.mesh[mesh_axis]
    proc_coord = get_process_coordinate(dist, mesh_axis)

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
        for (shape_key, pencil) in dist.pencil_cache
            if isa(shape_key, Tuple) && dim <= length(shape_key)
                return shape_key[dim]
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
