"""
    Transpose Grouped - Grouped transpose operations for TransposableField

This file contains grouped transpose operations that process multiple fields
at once for better MPI efficiency (following Dedalus GROUP_TRANSPOSES pattern):
- GroupedTransposeConfig
- group_transpose_z_to_y!
- group_transpose_y_to_z!
- group_transpose_y_to_x!
- group_transpose_x_to_y!
- _batched_transpose_* helpers
"""

# ============================================================================
# Grouped Transposes (Following Dedalus GROUP_TRANSPOSES)
# ============================================================================
#
# NOTE: TransposableField is designed for GPU+MPI distributed computing where
# PencilArrays cannot be used directly (PencilArrays is CPU-only). Therefore,
# these grouped transposes use explicit MPI.Alltoallv calls with custom pack/unpack.
#
# For CPU-only parallelization with ScalarFields, use the PencilArrays-based
# grouped transposes in distributor.jl:
#   - group_pencil_transpose!(dest_arrays, src_arrays, dist)
#   - group_transpose_fields!(fields, dist, source_decomp, dest_decomp)
#
# Design rationale:
# - TransposableField: GPU+MPI path with manual MPI for CUDA-aware communication
# - Distributor: CPU path using PencilArrays.transpose! for optimized communication
# ============================================================================

"""
    GroupedTransposeConfig

Configuration for grouped transpose operations (for TransposableField GPU+MPI path).
"""
mutable struct GroupedTransposeConfig
    enabled::Bool                    # Whether to use grouped transposes
    min_fields::Int                  # Minimum number of fields to trigger grouping
    sync_before_transpose::Bool      # Whether to add barrier before transposes
    max_batch_size::Int             # Maximum number of fields per batch

    function GroupedTransposeConfig()
        new(true, 2, false, 32)
    end
end

const GROUPED_TRANSPOSE_CONFIG = GroupedTransposeConfig()

"""
    set_group_transposes!(enabled::Bool; min_fields::Int=2, sync::Bool=false, max_batch::Int=32)

Enable or disable grouped transposes. When enabled, multiple TransposableFields
are transposed together in a single MPI communication for improved efficiency.

Following Dedalus GROUP_TRANSPOSES and SYNC_TRANSPOSES configuration.
"""
function set_group_transposes!(enabled::Bool; min_fields::Int=2, sync::Bool=false, max_batch::Int=32)
    GROUPED_TRANSPOSE_CONFIG.enabled = enabled
    GROUPED_TRANSPOSE_CONFIG.min_fields = min_fields
    GROUPED_TRANSPOSE_CONFIG.sync_before_transpose = sync
    GROUPED_TRANSPOSE_CONFIG.max_batch_size = max_batch
    return nothing
end

"""
    group_transpose_z_to_y!(fields::Vector{<:TransposableField})

Transpose multiple TransposableFields from ZLocal to YLocal in a single
batched MPI communication.

Following Dedalus Transpose.increment_group pattern.

# Algorithm
1. Group fields by (local_shape, topology)
2. For each group:
   - Stack all field data into one contiguous buffer
   - Perform single MPI.Alltoallv for the entire stack
   - Unstack results back to individual fields

# Performance Benefits
- Reduced MPI call overhead (one call instead of N)
- Better message aggregation for network efficiency
- Improved latency hiding
"""
function group_transpose_z_to_y!(fields::Vector{<:TransposableField})
    if isempty(fields)
        return
    end

    # Optional synchronization before transpose (Dedalus SYNC_TRANSPOSES)
    if GROUPED_TRANSPOSE_CONFIG.sync_before_transpose && !isempty(fields)
        topo = fields[1].topology
        if topo.row_comm !== nothing
            MPI.Barrier(topo.row_comm)
        end
    end

    # If grouping disabled, too few fields, or on GPU (scalar indexing not allowed), transpose individually
    # GPU uses batched transposes with scalar indexing which fails with allowscalar(false)
    if !GROUPED_TRANSPOSE_CONFIG.enabled || length(fields) < GROUPED_TRANSPOSE_CONFIG.min_fields || is_gpu(fields[1].buffers.architecture)
        for field in fields
            transpose_z_to_y!(field)
        end
        return
    end

    # Group fields by shape and topology
    field_groups = _group_transposable_fields(fields, ZLocal)

    for (key, group_fields) in field_groups
        if length(group_fields) == 1
            transpose_z_to_y!(group_fields[1])
        else
            _batched_transpose_z_to_y!(group_fields)
        end
    end
end

"""
    group_transpose_y_to_z!(fields::Vector{<:TransposableField})

Transpose multiple TransposableFields from YLocal to ZLocal in a single batch.
"""
function group_transpose_y_to_z!(fields::Vector{<:TransposableField})
    if isempty(fields)
        return
    end

    if GROUPED_TRANSPOSE_CONFIG.sync_before_transpose && !isempty(fields)
        topo = fields[1].topology
        if topo.row_comm !== nothing
            MPI.Barrier(topo.row_comm)
        end
    end

    # GPU uses batched transposes with scalar indexing which fails with allowscalar(false)
    if !GROUPED_TRANSPOSE_CONFIG.enabled || length(fields) < GROUPED_TRANSPOSE_CONFIG.min_fields || is_gpu(fields[1].buffers.architecture)
        for field in fields
            transpose_y_to_z!(field)
        end
        return
    end

    field_groups = _group_transposable_fields(fields, YLocal)

    for (key, group_fields) in field_groups
        if length(group_fields) == 1
            transpose_y_to_z!(group_fields[1])
        else
            _batched_transpose_y_to_z!(group_fields)
        end
    end
end

"""
    group_transpose_y_to_x!(fields::Vector{<:TransposableField})

Transpose multiple TransposableFields from YLocal to XLocal in a single batch.
"""
function group_transpose_y_to_x!(fields::Vector{<:TransposableField})
    if isempty(fields)
        return
    end

    if GROUPED_TRANSPOSE_CONFIG.sync_before_transpose && !isempty(fields)
        topo = fields[1].topology
        if topo.col_comm !== nothing
            MPI.Barrier(topo.col_comm)
        end
    end

    # GPU uses batched transposes with scalar indexing which fails with allowscalar(false)
    if !GROUPED_TRANSPOSE_CONFIG.enabled || length(fields) < GROUPED_TRANSPOSE_CONFIG.min_fields || is_gpu(fields[1].buffers.architecture)
        for field in fields
            transpose_y_to_x!(field)
        end
        return
    end

    field_groups = _group_transposable_fields(fields, YLocal)

    for (key, group_fields) in field_groups
        if length(group_fields) == 1
            transpose_y_to_x!(group_fields[1])
        else
            _batched_transpose_y_to_x!(group_fields)
        end
    end
end

"""
    group_transpose_x_to_y!(fields::Vector{<:TransposableField})

Transpose multiple TransposableFields from XLocal to YLocal in a single batch.
"""
function group_transpose_x_to_y!(fields::Vector{<:TransposableField})
    if isempty(fields)
        return
    end

    if GROUPED_TRANSPOSE_CONFIG.sync_before_transpose && !isempty(fields)
        topo = fields[1].topology
        if topo.col_comm !== nothing
            MPI.Barrier(topo.col_comm)
        end
    end

    # GPU uses batched transposes with scalar indexing which fails with allowscalar(false)
    if !GROUPED_TRANSPOSE_CONFIG.enabled || length(fields) < GROUPED_TRANSPOSE_CONFIG.min_fields || is_gpu(fields[1].buffers.architecture)
        for field in fields
            transpose_x_to_y!(field)
        end
        return
    end

    field_groups = _group_transposable_fields(fields, XLocal)

    for (key, group_fields) in field_groups
        if length(group_fields) == 1
            transpose_x_to_y!(group_fields[1])
        else
            _batched_transpose_x_to_y!(group_fields)
        end
    end
end

"""
    _group_transposable_fields(fields, expected_layout)

Group TransposableFields by their shape and topology for batched processing.
"""
function _group_transposable_fields(fields::Vector{<:TransposableField}, expected_layout::TransposeLayout)
    groups = Dict{Tuple, Vector{TransposableField}}()

    for field in fields
        # CRITICAL: Use error() instead of @assert for production safety
        if field.buffers.active_layout[] != expected_layout
            error("_group_transposable_fields: Field must be in $expected_layout layout, " *
                  "currently in $(field.buffers.active_layout[])")
        end

        # Key by global shape, topology, and element type
        key = (field.global_shape, field.topology.Rx, field.topology.Ry,
               eltype(field.buffers.z_local_data))

        if haskey(groups, key)
            push!(groups[key], field)
        else
            groups[key] = [field]
        end
    end

    return groups
end

"""
    _batched_transpose_z_to_y!(fields::Vector{<:TransposableField})

Perform Z→Y transpose for multiple fields in a single MPI operation.
"""
function _batched_transpose_z_to_y!(fields::Vector{<:TransposableField{F,T,N}}) where {F,T,N}
    nfields = length(fields)
    if nfields == 0
        return
    end

    first_field = fields[1]
    topo = first_field.topology

    # Check for true 2D mesh on 2D domain - this case requires Allgatherv, not Alltoallv
    # Fall back to individual transposes which handle this correctly
    if N == 2 && topo.Rx > 1 && topo.Ry > 1
        for field in fields
            transpose_z_to_y!(field)
        end
        return
    end

    # Determine communicator
    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    else
        if topo.row_size > 1
            comm = topo.row_comm
            comm_size = topo.row_size
        elseif topo.col_size > 1
            comm = topo.col_comm
            comm_size = topo.col_size
        else
            comm = nothing
            comm_size = 1
        end
    end

    if comm_size == 1
        for field in fields
            copyto!(vec(field.buffers.y_local_data), vec(field.buffers.z_local_data))
            field.buffers.active_layout[] = YLocal
        end
        return
    end

    arch = first_field.buffers.architecture

    # Stack all field data
    z_shape = size(first_field.buffers.z_local_data)
    y_shape = size(first_field.buffers.y_local_data)

    stacked_z = zeros(arch, T, (nfields, z_shape...))
    stacked_y = zeros(arch, T, (nfields, y_shape...))

    for (i, field) in enumerate(fields)
        selectdim(stacked_z, 1, i) .= field.buffers.z_local_data
    end

    # Compute stacked counts (multiply original counts by nfields)
    stacked_send_counts = first_field.counts.zy_send_counts .* nfields
    stacked_recv_counts = first_field.counts.zy_recv_counts .* nfields

    # Pack stacked data
    stacked_send_buf = zeros(arch, T, sum(stacked_send_counts))
    stacked_recv_buf = zeros(arch, T, sum(stacked_recv_counts))

    # Use same dims as non-batched (must match transpose_z_to_y! logic):
    # 3D: pack dim=3 (Z redistributed)
    # 2D with Ry > 1: pack dim=1 (x split)
    # 2D with Rx > 1: pack dim=2 (y split)
    # _pack_stacked_data! expects field dimension (not including stacked dimension)
    if N >= 3
        pack_dim = 3
    elseif topo.Ry > 1
        pack_dim = 1
    else
        # Rx > 1: ZLocal has partial x, full y. Split y.
        pack_dim = 2
    end
    _pack_stacked_data!(stacked_send_buf, stacked_z, first_field.counts.zy_send_counts,
                        first_field.counts.zy_send_displs, pack_dim, comm_size, nfields, arch)

    # Recompute displacements for stacked counts
    stacked_send_displs = zeros(Int, comm_size)
    stacked_recv_displs = zeros(Int, comm_size)
    for i in 2:comm_size
        stacked_send_displs[i] = stacked_send_displs[i-1] + stacked_send_counts[i-1]
        stacked_recv_displs[i] = stacked_recv_displs[i-1] + stacked_recv_counts[i-1]
    end

    # Single MPI communication for all fields (handles CUDA staging if needed)
    _do_alltoallv!(stacked_send_buf, stacked_recv_buf, stacked_send_counts, stacked_recv_counts,
                   comm, arch, first_field.buffers)

    # Unpack stacked data - use same dim as non-batched (must match transpose_z_to_y! logic):
    # 3D: unpack dim=2 (receiving Y chunks)
    # 2D with Ry > 1: unpack dim=2 (y received)
    # 2D with Rx > 1: unpack dim=1 (x received)
    if N >= 3
        unpack_dim = 2
    elseif topo.Ry > 1
        unpack_dim = 2
    else
        # Rx > 1: receive x dimension
        unpack_dim = 1
    end
    _unpack_stacked_data!(stacked_y, stacked_recv_buf, first_field.counts.zy_recv_counts,
                          first_field.counts.zy_recv_displs, unpack_dim, comm_size, nfields, arch)

    # Copy results back to individual fields
    for (i, field) in enumerate(fields)
        field.buffers.y_local_data .= selectdim(stacked_y, 1, i)
        field.buffers.active_layout[] = YLocal
        field.num_transposes += 1
    end
end

"""
    _batched_transpose_y_to_z!(fields::Vector{<:TransposableField})

Perform Y→Z transpose for multiple fields in a single MPI operation.
"""
function _batched_transpose_y_to_z!(fields::Vector{<:TransposableField{F,T,N}}) where {F,T,N}
    nfields = length(fields)
    if nfields == 0
        return
    end

    first_field = fields[1]
    topo = first_field.topology

    # Check for true 2D mesh on 2D domain - this case requires local scatter, not Alltoallv
    # Fall back to individual transposes which handle this correctly
    if N == 2 && topo.Rx > 1 && topo.Ry > 1
        for field in fields
            transpose_y_to_z!(field)
        end
        return
    end

    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    else
        if topo.row_size > 1
            comm = topo.row_comm
            comm_size = topo.row_size
        elseif topo.col_size > 1
            comm = topo.col_comm
            comm_size = topo.col_size
        else
            comm = nothing
            comm_size = 1
        end
    end

    if comm_size == 1
        for field in fields
            copyto!(vec(field.buffers.z_local_data), vec(field.buffers.y_local_data))
            field.buffers.active_layout[] = ZLocal
        end
        return
    end

    arch = first_field.buffers.architecture

    z_shape = size(first_field.buffers.z_local_data)
    y_shape = size(first_field.buffers.y_local_data)

    stacked_y = zeros(arch, T, (nfields, y_shape...))
    stacked_z = zeros(arch, T, (nfields, z_shape...))

    for (i, field) in enumerate(fields)
        selectdim(stacked_y, 1, i) .= field.buffers.y_local_data
    end

    stacked_send_counts = first_field.counts.zy_recv_counts .* nfields
    stacked_recv_counts = first_field.counts.zy_send_counts .* nfields

    stacked_send_buf = zeros(arch, T, sum(stacked_send_counts))
    stacked_recv_buf = zeros(arch, T, sum(stacked_recv_counts))

    # Use same dims as non-batched transpose_y_to_z!
    if N >= 3
        pack_dim = 2
    elseif topo.Ry > 1
        pack_dim = 2    # Ry>1: split y, receive x
    else
        pack_dim = 1    # Rx>1: split x, receive y
    end
    _pack_stacked_data!(stacked_send_buf, stacked_y, first_field.counts.zy_recv_counts,
                        first_field.counts.zy_recv_displs, pack_dim, comm_size, nfields, arch)

    stacked_send_displs = zeros(Int, comm_size)
    stacked_recv_displs = zeros(Int, comm_size)
    for i in 2:comm_size
        stacked_send_displs[i] = stacked_send_displs[i-1] + stacked_send_counts[i-1]
        stacked_recv_displs[i] = stacked_recv_displs[i-1] + stacked_recv_counts[i-1]
    end

    # Handle CUDA staging if needed
    _do_alltoallv!(stacked_send_buf, stacked_recv_buf, stacked_send_counts, stacked_recv_counts,
                   comm, arch, first_field.buffers)

    # Unpack dimension matching non-batched version
    if N >= 3
        unpack_dim = 3
    elseif topo.Ry > 1
        unpack_dim = 1  # Ry>1: receive x chunks
    else
        unpack_dim = 2  # Rx>1: receive y chunks
    end
    _unpack_stacked_data!(stacked_z, stacked_recv_buf, first_field.counts.zy_send_counts,
                          first_field.counts.zy_send_displs, unpack_dim, comm_size, nfields, arch)

    for (i, field) in enumerate(fields)
        field.buffers.z_local_data .= selectdim(stacked_z, 1, i)
        field.buffers.active_layout[] = ZLocal
        field.num_transposes += 1
    end
end

"""
    _batched_transpose_y_to_x!(fields::Vector{<:TransposableField})

Perform Y→X transpose for multiple fields in a single MPI operation.
"""
function _batched_transpose_y_to_x!(fields::Vector{<:TransposableField{F,T,N}}) where {F,T,N}
    nfields = length(fields)
    if nfields == 0
        return
    end

    first_field = fields[1]
    topo = first_field.topology

    if topo.col_size == 1
        for field in fields
            copyto!(vec(field.buffers.x_local_data), vec(field.buffers.y_local_data))
            field.buffers.active_layout[] = XLocal
        end
        return
    end

    arch = first_field.buffers.architecture

    y_shape = size(first_field.buffers.y_local_data)
    x_shape = size(first_field.buffers.x_local_data)

    stacked_y = zeros(arch, T, (nfields, y_shape...))
    stacked_x = zeros(arch, T, (nfields, x_shape...))

    for (i, field) in enumerate(fields)
        selectdim(stacked_y, 1, i) .= field.buffers.y_local_data
    end

    stacked_send_counts = first_field.counts.yx_send_counts .* nfields
    stacked_recv_counts = first_field.counts.yx_recv_counts .* nfields

    stacked_send_buf = zeros(arch, T, sum(stacked_send_counts))
    stacked_recv_buf = zeros(arch, T, sum(stacked_recv_counts))

    # Use same dims as non-batched transpose_y_to_x!: pack by Y (dim=2), unpack by X (dim=1)
    _pack_stacked_data!(stacked_send_buf, stacked_y, first_field.counts.yx_send_counts,
                        first_field.counts.yx_send_displs, 2, topo.col_size, nfields, arch)

    stacked_send_displs = zeros(Int, topo.col_size)
    stacked_recv_displs = zeros(Int, topo.col_size)
    for i in 2:topo.col_size
        stacked_send_displs[i] = stacked_send_displs[i-1] + stacked_send_counts[i-1]
        stacked_recv_displs[i] = stacked_recv_displs[i-1] + stacked_recv_counts[i-1]
    end

    # Handle CUDA staging if needed
    _do_alltoallv!(stacked_send_buf, stacked_recv_buf, stacked_send_counts, stacked_recv_counts,
                   topo.col_comm, arch, first_field.buffers)

    _unpack_stacked_data!(stacked_x, stacked_recv_buf, first_field.counts.yx_recv_counts,
                          first_field.counts.yx_recv_displs, 1, topo.col_size, nfields, arch)

    for (i, field) in enumerate(fields)
        field.buffers.x_local_data .= selectdim(stacked_x, 1, i)
        field.buffers.active_layout[] = XLocal
        field.num_transposes += 1
    end
end

"""
    _batched_transpose_x_to_y!(fields::Vector{<:TransposableField})

Perform X→Y transpose for multiple fields in a single MPI operation.
"""
function _batched_transpose_x_to_y!(fields::Vector{<:TransposableField{F,T,N}}) where {F,T,N}
    nfields = length(fields)
    if nfields == 0
        return
    end

    first_field = fields[1]
    topo = first_field.topology

    if topo.col_size == 1
        for field in fields
            copyto!(vec(field.buffers.y_local_data), vec(field.buffers.x_local_data))
            field.buffers.active_layout[] = YLocal
        end
        return
    end

    arch = first_field.buffers.architecture

    x_shape = size(first_field.buffers.x_local_data)
    y_shape = size(first_field.buffers.y_local_data)

    stacked_x = zeros(arch, T, (nfields, x_shape...))
    stacked_y = zeros(arch, T, (nfields, y_shape...))

    for (i, field) in enumerate(fields)
        selectdim(stacked_x, 1, i) .= field.buffers.x_local_data
    end

    stacked_send_counts = first_field.counts.yx_recv_counts .* nfields
    stacked_recv_counts = first_field.counts.yx_send_counts .* nfields

    stacked_send_buf = zeros(arch, T, sum(stacked_send_counts))
    stacked_recv_buf = zeros(arch, T, sum(stacked_recv_counts))

    # Use same dims as non-batched transpose_x_to_y!: pack by X (dim=1), unpack by Y (dim=2)
    _pack_stacked_data!(stacked_send_buf, stacked_x, first_field.counts.yx_recv_counts,
                        first_field.counts.yx_recv_displs, 1, topo.col_size, nfields, arch)

    stacked_send_displs = zeros(Int, topo.col_size)
    stacked_recv_displs = zeros(Int, topo.col_size)
    for i in 2:topo.col_size
        stacked_send_displs[i] = stacked_send_displs[i-1] + stacked_send_counts[i-1]
        stacked_recv_displs[i] = stacked_recv_displs[i-1] + stacked_recv_counts[i-1]
    end

    # Handle CUDA staging if needed
    _do_alltoallv!(stacked_send_buf, stacked_recv_buf, stacked_send_counts, stacked_recv_counts,
                   topo.col_comm, arch, first_field.buffers)

    _unpack_stacked_data!(stacked_y, stacked_recv_buf, first_field.counts.yx_send_counts,
                          first_field.counts.yx_send_displs, 2, topo.col_size, nfields, arch)

    for (i, field) in enumerate(fields)
        field.buffers.y_local_data .= selectdim(stacked_y, 1, i)
        field.buffers.active_layout[] = YLocal
        field.num_transposes += 1
    end
end

"""
    _pack_stacked_data!(buffer, stacked_data, counts, displs, dim, nranks, nfields, arch)

Pack stacked field data into contiguous buffer for batched MPI transpose.
Uses proper dimension-aware packing for each field in the stack.

The stacked_data has shape (nfields, field_shape...) where field_shape is 2D or 3D.
The dim parameter refers to the dimension in the FIELD (not including the stacked dimension).
"""
function _pack_stacked_data!(buffer, stacked_data, counts, displs, dim::Int,
                             nranks::Int, nfields::Int, arch::AbstractArchitecture)
    # Pack each field using proper dimension-aware packing
    # Buffer layout: for each rank, all fields' chunks contiguously

    ndims_field = ndims(stacked_data) - 1  # Exclude stacked dimension

    if ndims_field == 3
        # stacked_data shape: (nfields, Nx, Ny, Nz)
        _, Nx, Ny, Nz = size(stacked_data)
        _pack_stacked_3d!(buffer, stacked_data, counts, displs, dim, nranks, nfields, Nx, Ny, Nz)
    elseif ndims_field == 2
        # stacked_data shape: (nfields, Nx, Ny)
        _, Nx, Ny = size(stacked_data)
        _pack_stacked_2d!(buffer, stacked_data, counts, displs, dim, nranks, nfields, Nx, Ny)
    else
        # Fallback for 1D - simple interleaved copy
        buf_idx = 1
        for rank in 1:nranks
            count = counts[rank]
            for field_idx in 1:nfields
                field_data = selectdim(stacked_data, 1, field_idx)
                src_start = displs[rank] + 1
                src_end = src_start + count - 1
                if src_end <= length(field_data)
                    buffer[buf_idx:buf_idx+count-1] .= vec(field_data)[src_start:src_end]
                    buf_idx += count
                end
            end
        end
    end
end

function _pack_stacked_3d!(buffer, stacked_data, counts, displs, dim::Int,
                           nranks::Int, nfields::Int, Nx::Int, Ny::Int, Nz::Int)
    # Compute chunk sizes from counts (same logic as _pack_3d_cpu!)
    chunk_sizes = zeros(Int, nranks)
    if dim == 3  # Z being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Ny)
        end
    elseif dim == 2  # Y being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Nz)
        end
    else  # dim == 1, X being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Ny * Nz)
        end
    end

    # Pack: buffer layout is [rank1_field1, rank1_field2, ..., rank2_field1, ...]
    buf_offset = 0
    for rank in 1:nranks
        for field_idx in 1:nfields
            field_data = @view stacked_data[field_idx, :, :, :]

            # Pack this field's chunk for this rank
            for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
                # Determine if this element belongs to current rank
                if dim == 3
                    z_offset = sum(chunk_sizes[1:rank-1])
                    if iz > z_offset && iz <= z_offset + chunk_sizes[rank]
                        local_iz = iz - z_offset
                        local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
                        buffer[buf_offset + local_idx] = field_data[ix, iy, iz]
                    end
                elseif dim == 2
                    y_offset = sum(chunk_sizes[1:rank-1])
                    if iy > y_offset && iy <= y_offset + chunk_sizes[rank]
                        local_iy = iy - y_offset
                        local_idx = (iz - 1) * Nx * chunk_sizes[rank] + (local_iy - 1) * Nx + ix
                        buffer[buf_offset + local_idx] = field_data[ix, iy, iz]
                    end
                else  # dim == 1
                    x_offset = sum(chunk_sizes[1:rank-1])
                    if ix > x_offset && ix <= x_offset + chunk_sizes[rank]
                        local_ix = ix - x_offset
                        local_idx = (iz - 1) * chunk_sizes[rank] * Ny + (iy - 1) * chunk_sizes[rank] + local_ix
                        buffer[buf_offset + local_idx] = field_data[ix, iy, iz]
                    end
                end
            end
            buf_offset += counts[rank]
        end
    end
end

function _pack_stacked_2d!(buffer, stacked_data, counts, displs, dim::Int,
                           nranks::Int, nfields::Int, Nx::Int, Ny::Int)
    # Compute chunk sizes from counts
    chunk_sizes = zeros(Int, nranks)
    if dim == 2  # Y being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Nx
        end
    else  # dim == 1, X being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Ny
        end
    end

    # Pack: buffer layout is [rank1_field1, rank1_field2, ..., rank2_field1, ...]
    buf_offset = 0
    for rank in 1:nranks
        for field_idx in 1:nfields
            field_data = @view stacked_data[field_idx, :, :]

            for iy in 1:Ny, ix in 1:Nx
                if dim == 2
                    y_offset = sum(chunk_sizes[1:rank-1])
                    if iy > y_offset && iy <= y_offset + chunk_sizes[rank]
                        local_iy = iy - y_offset
                        local_idx = (local_iy - 1) * Nx + ix
                        buffer[buf_offset + local_idx] = field_data[ix, iy]
                    end
                else  # dim == 1
                    x_offset = sum(chunk_sizes[1:rank-1])
                    if ix > x_offset && ix <= x_offset + chunk_sizes[rank]
                        local_ix = ix - x_offset
                        local_idx = (iy - 1) * chunk_sizes[rank] + local_ix
                        buffer[buf_offset + local_idx] = field_data[ix, iy]
                    end
                end
            end
            buf_offset += counts[rank]
        end
    end
end

"""
    _unpack_stacked_data!(stacked_data, buffer, counts, displs, dim, nranks, nfields, arch)

Unpack buffer into stacked field data after batched MPI transpose.
Uses proper dimension-aware unpacking for each field in the stack.

The stacked_data has shape (nfields, field_shape...) where field_shape is 2D or 3D.
The dim parameter refers to the dimension in the FIELD that received chunks (not including stacked dimension).
"""
function _unpack_stacked_data!(stacked_data, buffer, counts, displs, dim::Int,
                               nranks::Int, nfields::Int, arch::AbstractArchitecture)
    ndims_field = ndims(stacked_data) - 1  # Exclude stacked dimension

    if ndims_field == 3
        # stacked_data shape: (nfields, Nx, Ny, Nz)
        _, Nx, Ny, Nz = size(stacked_data)
        _unpack_stacked_3d!(stacked_data, buffer, counts, displs, dim, nranks, nfields, Nx, Ny, Nz)
    elseif ndims_field == 2
        # stacked_data shape: (nfields, Nx, Ny)
        _, Nx, Ny = size(stacked_data)
        _unpack_stacked_2d!(stacked_data, buffer, counts, displs, dim, nranks, nfields, Nx, Ny)
    else
        # Fallback for 1D - simple interleaved copy
        buf_idx = 1
        for rank in 1:nranks
            count = counts[rank]
            for field_idx in 1:nfields
                field_data = selectdim(stacked_data, 1, field_idx)
                dest_start = displs[rank] + 1
                dest_end = dest_start + count - 1
                if dest_end <= length(field_data)
                    vec(field_data)[dest_start:dest_end] .= buffer[buf_idx:buf_idx+count-1]
                    buf_idx += count
                end
            end
        end
    end
end

function _unpack_stacked_3d!(stacked_data, buffer, counts, displs, dim::Int,
                             nranks::Int, nfields::Int, Nx::Int, Ny::Int, Nz::Int)
    # Compute chunk sizes from counts
    chunk_sizes = zeros(Int, nranks)
    if dim == 2  # After Z→Y: receiving y-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Nz)
        end
    elseif dim == 1  # After Y→X: receiving x-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Ny * Nz)
        end
    else  # dim == 3: receiving z-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Ny)
        end
    end

    # Unpack: buffer layout is [rank1_field1, rank1_field2, ..., rank2_field1, ...]
    buf_offset = 0
    for rank in 1:nranks
        for field_idx in 1:nfields
            field_data = @view stacked_data[field_idx, :, :, :]

            for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
                if dim == 2  # Receiving y-chunks
                    y_offset = sum(chunk_sizes[1:rank-1])
                    if iy > y_offset && iy <= y_offset + chunk_sizes[rank]
                        local_iy = iy - y_offset
                        local_idx = (iz - 1) * Nx * chunk_sizes[rank] + (local_iy - 1) * Nx + ix
                        field_data[ix, iy, iz] = buffer[buf_offset + local_idx]
                    end
                elseif dim == 1  # Receiving x-chunks
                    x_offset = sum(chunk_sizes[1:rank-1])
                    if ix > x_offset && ix <= x_offset + chunk_sizes[rank]
                        local_ix = ix - x_offset
                        local_idx = (iz - 1) * chunk_sizes[rank] * Ny + (iy - 1) * chunk_sizes[rank] + local_ix
                        field_data[ix, iy, iz] = buffer[buf_offset + local_idx]
                    end
                else  # dim == 3: receiving z-chunks
                    z_offset = sum(chunk_sizes[1:rank-1])
                    if iz > z_offset && iz <= z_offset + chunk_sizes[rank]
                        local_iz = iz - z_offset
                        local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
                        field_data[ix, iy, iz] = buffer[buf_offset + local_idx]
                    end
                end
            end
            buf_offset += counts[rank]
        end
    end
end

function _unpack_stacked_2d!(stacked_data, buffer, counts, displs, dim::Int,
                             nranks::Int, nfields::Int, Nx::Int, Ny::Int)
    # Compute chunk sizes from counts
    chunk_sizes = zeros(Int, nranks)
    if dim == 2  # Receiving y-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Nx
        end
    else  # dim == 1: receiving x-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Ny
        end
    end

    # Unpack: buffer layout is [rank1_field1, rank1_field2, ..., rank2_field1, ...]
    buf_offset = 0
    for rank in 1:nranks
        for field_idx in 1:nfields
            field_data = @view stacked_data[field_idx, :, :]

            for iy in 1:Ny, ix in 1:Nx
                if dim == 2  # Receiving y-chunks
                    y_offset = sum(chunk_sizes[1:rank-1])
                    if iy > y_offset && iy <= y_offset + chunk_sizes[rank]
                        local_iy = iy - y_offset
                        local_idx = (local_iy - 1) * Nx + ix
                        field_data[ix, iy] = buffer[buf_offset + local_idx]
                    end
                else  # dim == 1: receiving x-chunks
                    x_offset = sum(chunk_sizes[1:rank-1])
                    if ix > x_offset && ix <= x_offset + chunk_sizes[rank]
                        local_ix = ix - x_offset
                        local_idx = (iy - 1) * chunk_sizes[rank] + local_ix
                        field_data[ix, iy] = buffer[buf_offset + local_idx]
                    end
                end
            end
            buf_offset += counts[rank]
        end
    end
end

"""
    group_distributed_forward_transform!(fields::Vector{<:TransposableField}; overlap=false)

Perform forward transforms on multiple TransposableFields using grouped transposes.
"""
function group_distributed_forward_transform!(fields::Vector{<:TransposableField};
                                              overlap::Bool=false)
    if isempty(fields)
        return
    end

    # Initialize all fields to ZLocal layout
    for field in fields
        field.buffers.active_layout[] = ZLocal
        copyto!(vec(field.buffers.z_local_data), vec(field.field["g"]))
    end

    # Get dimensions from first field
    N = length(fields[1].global_shape)

    if N >= 3
        # Step 1: FFT in z (all fields, local operation)
        for field in fields
            zbasis = get_basis_for_dim(field, 3)
            transform_in_dim!(field.buffers.z_local_data, 3, :forward, zbasis,
                            field.buffers.architecture)
        end

        # Step 2: Grouped Z→Y transpose
        group_transpose_z_to_y!(fields)

        # Step 3: FFT in y (all fields, local operation)
        for field in fields
            ybasis = get_basis_for_dim(field, 2)
            transform_in_dim!(field.buffers.y_local_data, 2, :forward, ybasis,
                            field.buffers.architecture)
        end

        # Step 4: Grouped Y→X transpose
        group_transpose_y_to_x!(fields)

        # Step 5: FFT in x (all fields, local operation)
        for field in fields
            xbasis = get_basis_for_dim(field, 1)
            transform_in_dim!(field.buffers.x_local_data, 1, :forward, xbasis,
                            field.buffers.architecture)
        end

        # Step 6: Transpose back to ZLocal layout to match field["c"] allocation
        # (field["c"] is allocated with ZLocal decomposition via get_local_array_size)
        group_transpose_x_to_y!(fields)
        group_transpose_y_to_z!(fields)

        # Copy results to field coefficient data — now both are in ZLocal layout
        for field in fields
            copyto!(vec(field.field["c"]), vec(field.buffers.z_local_data))
        end

    elseif N == 2
        # Check for true 2D mesh - requires different transform sequence
        topo = fields[1].topology
        if topo.Rx > 1 && topo.Ry > 1
            # True 2D mesh on 2D domain
            # ZLocal=(Nx/Rx, Ny/Ry) → YLocal=(Nx/Rx, Ny) → XLocal=(Nx, Ny/Rx)

            # Step 1: Z→Y transpose to get y local (grouped, falls back to Allgatherv)
            group_transpose_z_to_y!(fields)

            # Step 2: Transform in y (dim 2, now local in YLocal)
            for field in fields
                ybasis = get_basis_for_dim(field, 2)
                transform_in_dim!(field.buffers.y_local_data, 2, :forward, ybasis,
                                field.buffers.architecture)
            end

            # Step 3: Y→X transpose to get x local
            group_transpose_y_to_x!(fields)

            # Step 4: Transform in x (dim 1, now local in XLocal)
            for field in fields
                xbasis = get_basis_for_dim(field, 1)
                transform_in_dim!(field.buffers.x_local_data, 1, :forward, xbasis,
                                field.buffers.architecture)
            end

            # Step 5: Transpose back to ZLocal layout to match field["c"] allocation
            group_transpose_x_to_y!(fields)
            group_transpose_y_to_z!(fields)

            for field in fields
                copyto!(vec(field.field["c"]), vec(field.buffers.z_local_data))
            end
        else
            # 1D decomposition: ZLocal=(Nx, Ny/P) → YLocal=(Nx/P, Ny)
            # x is local in ZLocal

            # Step 1: Transform in x (dim 1, local in ZLocal)
            for field in fields
                xbasis = get_basis_for_dim(field, 1)
                transform_in_dim!(field.buffers.z_local_data, 1, :forward, xbasis,
                                field.buffers.architecture)
            end

            # Step 2: Z→Y transpose
            group_transpose_z_to_y!(fields)

            # Step 3: Transform in y (dim 2, now local in YLocal)
            for field in fields
                ybasis = get_basis_for_dim(field, 2)
                transform_in_dim!(field.buffers.y_local_data, 2, :forward, ybasis,
                                field.buffers.architecture)
            end

            # Step 4: Transpose back to ZLocal layout to match field["c"] allocation
            group_transpose_y_to_z!(fields)

            for field in fields
                copyto!(vec(field.field["c"]), vec(field.buffers.z_local_data))
            end
        end
    end
end

"""
    group_distributed_backward_transform!(fields::Vector{<:TransposableField}; overlap=false)

Perform backward transforms on multiple TransposableFields using grouped transposes.
"""
function group_distributed_backward_transform!(fields::Vector{<:TransposableField};
                                               overlap::Bool=false)
    if isempty(fields)
        return
    end

    N = length(fields[1].global_shape)

    if N >= 3
        # field["c"] is stored in ZLocal layout — load into z_local_data first,
        # then transpose to XLocal before starting inverse FFTs
        for field in fields
            field.buffers.active_layout[] = ZLocal
            copyto!(vec(field.buffers.z_local_data), vec(field.field["c"]))
        end

        # Transpose ZLocal → YLocal → XLocal to reach the starting layout for inverse FFTs
        group_transpose_z_to_y!(fields)
        group_transpose_y_to_x!(fields)

        # Step 1: Inverse FFT in x (XLocal, x is local)
        for field in fields
            xbasis = get_basis_for_dim(field, 1)
            transform_in_dim!(field.buffers.x_local_data, 1, :backward, xbasis,
                            field.buffers.architecture)
        end

        # Step 2: X→Y transpose
        group_transpose_x_to_y!(fields)

        # Step 3: Inverse FFT in y (YLocal, y is local)
        for field in fields
            ybasis = get_basis_for_dim(field, 2)
            transform_in_dim!(field.buffers.y_local_data, 2, :backward, ybasis,
                            field.buffers.architecture)
        end

        # Step 4: Y→Z transpose
        group_transpose_y_to_z!(fields)

        # Step 5: Inverse FFT in z (ZLocal, z is local)
        for field in fields
            zbasis = get_basis_for_dim(field, 3)
            transform_in_dim!(field.buffers.z_local_data, 3, :backward, zbasis,
                            field.buffers.architecture)
            # Preserve complex values if field dtype is complex
            if field.field.dtype <: Complex
                copyto!(vec(field.field["g"]), vec(field.buffers.z_local_data))
            else
                copyto!(vec(field.field["g"]), real.(vec(field.buffers.z_local_data)))
            end
        end

    elseif N == 2
        # Check for true 2D mesh - requires different transform sequence
        topo = fields[1].topology
        if topo.Rx > 1 && topo.Ry > 1
            # True 2D mesh on 2D domain
            # field["c"] is in ZLocal layout — transpose to XLocal first
            for field in fields
                field.buffers.active_layout[] = ZLocal
                copyto!(vec(field.buffers.z_local_data), vec(field.field["c"]))
            end

            # Transpose ZLocal → YLocal → XLocal
            group_transpose_z_to_y!(fields)
            group_transpose_y_to_x!(fields)

            # Step 1: Inverse transform in x (dim 1, local in XLocal)
            for field in fields
                xbasis = get_basis_for_dim(field, 1)
                transform_in_dim!(field.buffers.x_local_data, 1, :backward, xbasis,
                                field.buffers.architecture)
            end

            # Step 2: X→Y transpose
            group_transpose_x_to_y!(fields)

            # Step 3: Inverse transform in y (dim 2, local in YLocal)
            for field in fields
                ybasis = get_basis_for_dim(field, 2)
                transform_in_dim!(field.buffers.y_local_data, 2, :backward, ybasis,
                                field.buffers.architecture)
            end

            # Step 4: Y→Z transpose
            group_transpose_y_to_z!(fields)

            # Copy results
            for field in fields
                if field.field.dtype <: Complex
                    copyto!(vec(field.field["g"]), vec(field.buffers.z_local_data))
                else
                    copyto!(vec(field.field["g"]), real.(vec(field.buffers.z_local_data)))
                end
            end
        else
            # 1D decomposition: field["c"] is in ZLocal layout
            # ZLocal has x local, so start inverse FFTs from ZLocal
            for field in fields
                field.buffers.active_layout[] = ZLocal
                copyto!(vec(field.buffers.z_local_data), vec(field.field["c"]))
            end

            # Step 1: Inverse transform in x (dim 1, local in ZLocal)
            for field in fields
                xbasis = get_basis_for_dim(field, 1)
                transform_in_dim!(field.buffers.z_local_data, 1, :backward, xbasis,
                                field.buffers.architecture)
            end

            # Step 2: Z→Y transpose
            group_transpose_z_to_y!(fields)

            # Step 3: Inverse transform in y (dim 2, local in YLocal)
            for field in fields
                ybasis = get_basis_for_dim(field, 2)
                transform_in_dim!(field.buffers.y_local_data, 2, :backward, ybasis,
                                field.buffers.architecture)
            end

            # Step 4: Y→Z transpose back to ZLocal for field["g"]
            group_transpose_y_to_z!(fields)

            # Copy results
            for field in fields
                if field.field.dtype <: Complex
                    copyto!(vec(field.field["g"]), vec(field.buffers.z_local_data))
                else
                    copyto!(vec(field.field["g"]), real.(vec(field.buffers.z_local_data)))
                end
            end
        end
    end
end

