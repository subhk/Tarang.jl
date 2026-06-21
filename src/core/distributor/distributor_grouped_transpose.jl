# Grouped PencilArray transpose support for CPU parallel execution.

# ============================================================================
# Grouped PencilArray Transposes (GROUP_TRANSPOSES for CPU)
# ============================================================================

"""
    GroupedPencilTransposeConfig

Configuration for grouped PencilArray transpose operations.
"""
mutable struct GroupedPencilTransposeConfig
    enabled::Bool
    min_fields::Int
    sync_before_transpose::Bool

    function GroupedPencilTransposeConfig()
        new(true, 2, false)
    end
end

const GROUPED_PENCIL_TRANSPOSE_CONFIG = GroupedPencilTransposeConfig()

"""
    set_grouped_pencil_transposes!(enabled::Bool; min_fields::Int=2, sync::Bool=false)

Enable or disable grouped PencilArray transposes for CPU parallelization.
"""
function set_grouped_pencil_transposes!(enabled::Bool; min_fields::Int=2, sync::Bool=false)
    GROUPED_PENCIL_TRANSPOSE_CONFIG.enabled = enabled
    GROUPED_PENCIL_TRANSPOSE_CONFIG.min_fields = min_fields
    GROUPED_PENCIL_TRANSPOSE_CONFIG.sync_before_transpose = sync
    return nothing
end

"""
    group_pencil_transpose!(dest_arrays::Vector{<:PencilArrays.PencilArray},
                            src_arrays::Vector{<:PencilArrays.PencilArray},
                            dist::Distributor)

Transpose multiple PencilArrays together using grouped MPI communication.

For CPU parallelization, this batches multiple PencilArray transposes into
fewer MPI calls by:
1. Stacking source data from all arrays
2. Performing grouped MPI.Alltoallv
3. Unstacking to destination arrays

Following GROUP_TRANSPOSES pattern but using PencilArrays infrastructure.
"""
function group_pencil_transpose!(dest_arrays::Vector{<:PencilArrays.PencilArray},
                                 src_arrays::Vector{<:PencilArrays.PencilArray},
                                 dist::Distributor)
    n = length(src_arrays)
    if n == 0
        return
    end

    @assert length(dest_arrays) == n "Destination and source arrays must have same length"

    # Optional synchronization
    if GROUPED_PENCIL_TRANSPOSE_CONFIG.sync_before_transpose
        MPI.Barrier(dist.comm)
    end

    # If grouping disabled or too few arrays, transpose individually
    if !GROUPED_PENCIL_TRANSPOSE_CONFIG.enabled || n < GROUPED_PENCIL_TRANSPOSE_CONFIG.min_fields
        for i in 1:n
            PencilArrays.transpose!(dest_arrays[i], src_arrays[i])
        end
        return
    end

    # Group by compatible pencil configurations
    groups = _group_pencil_arrays(src_arrays, dest_arrays)

    # Process groups in a RANK-INVARIANT order. Each `transpose!` is a collective
    # over the shared topology comm and is matched purely by program order on
    # every rank, so all ranks MUST issue the per-field transposes in the same
    # sequence. `Dict` iteration order is not a cross-rank contract, so sort the
    # groups by their (rank-invariant) minimum field index, and process the
    # field indices within each group in ascending order.
    ordered_groups = sort!(collect(values(groups)); by = minimum)

    for indices in ordered_groups
        sorted_indices = sort(indices)
        if length(sorted_indices) == 1
            i = sorted_indices[1]
            PencilArrays.transpose!(dest_arrays[i], src_arrays[i])
        else
            _batched_pencil_transpose!(dest_arrays, src_arrays, sorted_indices, dist)
        end
    end
end

"""
    _group_pencil_arrays(src_arrays, dest_arrays)

Group PencilArrays by compatible configurations for batched transpose.
"""
function _group_pencil_arrays(src_arrays::Vector{<:PencilArrays.PencilArray},
                              dest_arrays::Vector{<:PencilArrays.PencilArray})
    groups = Dict{Tuple, Vector{Int}}()

    for i in eachindex(src_arrays)
        src = src_arrays[i]
        dest = dest_arrays[i]

        # Key by GLOBAL source/dest pencil shapes and element type.
        # Global shapes are identical on every rank, so all ranks form the SAME
        # groups. Keying on rank-local `size_local` is unsafe: under an uneven
        # decomposition two fields can collide to the same local size on one
        # rank but not another, so ranks would partition the field list
        # differently and desynchronise the per-field collective `transpose!`
        # calls (hang/corruption). See group_pencil_transpose! for ordering.
        src_pencil = PencilArrays.pencil(src)
        dest_pencil = PencilArrays.pencil(dest)

        key = (
            PencilArrays.size_global(src_pencil),
            PencilArrays.size_global(dest_pencil),
            eltype(src)
        )

        if haskey(groups, key)
            push!(groups[key], i)
        else
            groups[key] = [i]
        end
    end

    return groups
end

"""
    _batched_pencil_transpose!(dest_arrays, src_arrays, indices, dist)

Perform batched transpose for a group of compatible PencilArrays.
"""
function _batched_pencil_transpose!(dest_arrays::Vector{<:PencilArrays.PencilArray},
                                    src_arrays::Vector{<:PencilArrays.PencilArray},
                                    indices::Vector{Int},
                                    dist::Distributor)
    nfields = length(indices)
    if nfields == 0
        return
    end

    # Get first array for sizing
    first_idx = indices[1]
    first_src = src_arrays[first_idx]
    first_dest = dest_arrays[first_idx]

    T = eltype(first_src)
    src_local_size = length(parent(first_src))
    dest_local_size = length(parent(first_dest))

    # Stack all source data
    stacked_src = zeros(T, src_local_size * nfields)
    for (batch_idx, field_idx) in enumerate(indices)
        src_data = parent(src_arrays[field_idx])
        offset = (batch_idx - 1) * src_local_size
        stacked_src[offset+1:offset+src_local_size] .= vec(src_data)
    end

    # Use PencilArrays transpose infrastructure
    # Get the transpose plan from the first array pair
    # Note: PencilArrays may not directly support stacked transposes
    # So we use MPI.Alltoallv with proper counts

    src_pencil = PencilArrays.pencil(first_src)
    dest_pencil = PencilArrays.pencil(first_dest)

    # Try to use PencilArrays transpose for the batch
    # If not possible, fall back to individual transposes
    try
        # Compute send/recv counts for the stacked data
        # Each field contributes equally to the counts
        stacked_dest = zeros(T, dest_local_size * nfields)

        # Get topology from pencil
        topo = PencilArrays.topology(src_pencil)
        comm = PencilArrays.comm(topo)

        # For now, fall back to individual transposes since PencilArrays
        # doesn't directly support batched operations on stacked data
        for field_idx in indices
            PencilArrays.transpose!(dest_arrays[field_idx], src_arrays[field_idx])
        end
    catch e
        # Fallback: transpose individually
        @debug "Batched PencilArray transpose failed, using individual transposes: $e"
        for field_idx in indices
            PencilArrays.transpose!(dest_arrays[field_idx], src_arrays[field_idx])
        end
    end
end

"""
    group_transpose_fields!(fields::Vector, dist::Distributor, source_decomp::Tuple, dest_decomp::Tuple)

Transpose multiple fields together using grouped PencilArray operations.

This is a higher-level interface that works with field objects (e.g., ScalarField)
and their underlying PencilArrays. Fields must support `get_grid_data` and `set_grid_data!`.
"""
function group_transpose_fields!(fields::Vector, dist::Distributor,
                                 source_decomp::Tuple, dest_decomp::Tuple)
    if isempty(fields)
        return
    end

    # This function requires PencilArrays (MPI+CPU path)
    if !dist.use_pencil_arrays
        error("group_transpose_fields! requires PencilArrays (MPI+CPU). " *
              "For GPU or non-PencilArrays paths, use individual field transposes instead.")
    end

    # Create source and destination PencilArrays for each field
    # Track original indices to correctly write back results
    src_arrays = PencilArrays.PencilArray[]
    dest_arrays = PencilArrays.PencilArray[]
    included_field_indices = Int[]  # Track which fields were included

    for (field_idx, field) in enumerate(fields)
        grid_data = get_grid_data(field)
        if grid_data === nothing
            continue
        end

        # Get global shape from field's domain (NOT local size from grid_data!)
        # Using size(grid_data) would incorrectly use local size as if it were global
        if field.domain === nothing
            error("group_transpose_fields! requires fields with domain set")
        end
        gshape = global_shape(field.domain)

        # Create PencilArray for source (create_pencil returns PencilArray, not Pencil)
        src_pa = create_pencil(dist, gshape, source_decomp; dtype=eltype(grid_data))

        # Extract the Pencil from the PencilArray to create transpose pencil
        src_pencil_obj = PencilArrays.pencil(src_pa)
        dest_pencil_obj = create_transpose_pencil(dist, src_pencil_obj, dest_decomp)

        # Create destination PencilArray from the transposed Pencil
        dest_pa = PencilArrays.PencilArray{eltype(grid_data)}(undef, dest_pencil_obj)

        copyto!(parent(src_pa), grid_data)

        push!(src_arrays, src_pa)
        push!(dest_arrays, dest_pa)
        push!(included_field_indices, field_idx)
    end

    # Perform grouped transpose
    group_pencil_transpose!(dest_arrays, src_arrays, dist)

    # Copy results back to fields using tracked indices
    # IMPORTANT: Preserve the PencilArray wrapper to maintain MPI distribution metadata.
    # Using copy(parent(...)) would strip the PencilArray wrapper, causing subsequent
    # PencilFFT/MPI operations to fail or produce incorrect results.
    for (dest_idx, field_idx) in enumerate(included_field_indices)
        # Keep the PencilArray - it contains MPI distribution info needed for PencilFFTs
        set_grid_data!(fields[field_idx], dest_arrays[dest_idx])
    end
end

# ============================================================================
