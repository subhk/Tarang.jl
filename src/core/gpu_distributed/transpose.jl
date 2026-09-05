# ============================================================================
# MPI Transpose Operations
# ============================================================================

"""
    _compute_alltoall_counts_reverse(dims, dim, config)

Send/recv counts + displacements for the REVERSE all-to-all (undoing the
forward transpose): split along `dim` (now full `global_n`), assemble along
`other_dim`. This is the orientation `mpi_alltoall_transpose_reverse` uses; the
forward `_compute_alltoall_counts` splits along `other_dim` instead, so the
CUDA-aware reverse path MUST use these (its pack splits along `dim`). For
even nprocs=2 splits the two count sets coincide, which previously masked the
mismatch on nprocs>=4 / non-divisible sizes.
"""
function _compute_alltoall_counts_reverse(dims::Tuple, dim::Int, config::DistributedGPUConfig)
    nprocs = config.size
    rank = config.rank
    ndims_data = length(dims)
    global_n = config.global_shape[dim]
    other_dim = dim == ndims_data ? 1 : ndims_data
    other_n = config.global_shape[other_dim]

    chunk_other = Vector{Int}(undef, nprocs)
    chunk_dim = Vector{Int}(undef, nprocs)
    for p in 1:nprocs
        chunk_other[p] = div(other_n, nprocs) + (p - 1 < mod(other_n, nprocs) ? 1 : 0)
        chunk_dim[p]   = div(global_n, nprocs) + (p - 1 < mod(global_n, nprocs) ? 1 : 0)
    end
    chunk_dim_me = chunk_dim[rank + 1]
    remaining = div(prod(dims), dims[dim] * dims[other_dim])

    send_counts = Vector{Int}(undef, nprocs)
    recv_counts = Vector{Int}(undef, nprocs)
    send_displs = Vector{Int}(undef, nprocs)
    recv_displs = Vector{Int}(undef, nprocs)
    send_offset = 0
    recv_offset = 0
    for p in 1:nprocs
        send_counts[p] = chunk_dim[p] * dims[other_dim] * remaining
        recv_counts[p] = chunk_other[p] * chunk_dim_me * remaining
        send_displs[p] = send_offset
        recv_displs[p] = recv_offset
        send_offset += send_counts[p]
        recv_offset += recv_counts[p]
    end
    return send_counts, recv_counts, send_displs, recv_displs
end

"""
    mpi_alltoall_transpose(data, dim, config)

Perform MPI all-to-all transpose to redistribute data.

Before: data distributed along `dim` (each rank has local slice)
After: data transposed so `dim` is fully local on each rank
       (but another dimension is now distributed)
"""
function mpi_alltoall_transpose(data::Array{T}, dim::Int, config::DistributedGPUConfig) where T
    comm = config.comm
    nprocs = config.size
    rank = config.rank

    if nprocs == 1
        return data
    end

    dims = size(data)
    ndims_data = length(dims)
    global_n = config.global_shape[dim]
    local_n = dims[dim]

    # Determine split dimension (will become distributed after transpose)
    other_dim = dim == ndims_data ? 1 : ndims_data
    other_n = dims[other_dim]

    # Compute per-rank chunk sizes along other_dim and dim
    chunk_other = Vector{Int}(undef, nprocs)
    chunk_dim = Vector{Int}(undef, nprocs)
    for p in 1:nprocs
        chunk_other[p] = div(other_n, nprocs) + (p - 1 < mod(other_n, nprocs) ? 1 : 0)
        chunk_dim[p] = div(global_n, nprocs) + (p - 1 < mod(global_n, nprocs) ? 1 : 0)
    end
    chunk_other_me = chunk_other[rank + 1]

    # Compute send/recv counts
    remaining = div(prod(dims), other_n * local_n)
    send_counts = Vector{Int}(undef, nprocs)
    recv_counts = Vector{Int}(undef, nprocs)
    send_displs = Vector{Int}(undef, nprocs)
    recv_displs = Vector{Int}(undef, nprocs)
    send_offset = 0
    recv_offset = 0
    for p in 1:nprocs
        send_counts[p] = chunk_other[p] * local_n * remaining
        recv_counts[p] = chunk_dim[p] * chunk_other_me * remaining
        send_displs[p] = send_offset
        recv_displs[p] = recv_offset
        send_offset += send_counts[p]
        recv_offset += recv_counts[p]
    end

    # Pack data: split along other_dim, each rank's portion contiguous
    send_buf = Vector{T}(undef, send_offset)
    _cpu_pack_for_transpose!(send_buf, data, other_dim, chunk_other, send_displs)

    # All-to-all communication
    recv_buf = Vector{T}(undef, recv_offset)
    sendbuf = MPI.VBuffer(send_buf, send_counts, send_displs)
    recvbuf = MPI.VBuffer(recv_buf, recv_counts, recv_displs)
    MPI.Alltoallv!(sendbuf, recvbuf, comm)

    # Unpack: assemble along dim from per-rank contributions
    transposed_dims = _compute_transposed_dims(dims, dim, config)
    output = Array{T}(undef, transposed_dims...)
    _cpu_unpack_from_transpose!(output, recv_buf, dim, chunk_dim, recv_displs)

    return output
end

"""
    mpi_alltoall_transpose_reverse(data, dim, config)

Reverse the transpose operation: makes other_dim fully local and redistributes dim.
"""
function mpi_alltoall_transpose_reverse(data::Array{T}, dim::Int, config::DistributedGPUConfig) where T
    comm = config.comm
    nprocs = config.size
    rank = config.rank

    if nprocs == 1
        return data
    end

    dims = size(data)
    ndims_data = length(dims)
    global_n = config.global_shape[dim]

    other_dim = dim == ndims_data ? 1 : ndims_data
    other_n = config.global_shape[other_dim]  # Full other_dim size

    # After transpose, data has: dim=global_n, other_dim=chunk_other_me
    # Reverse: split along dim, assemble along other_dim
    chunk_other = Vector{Int}(undef, nprocs)
    chunk_dim = Vector{Int}(undef, nprocs)
    for p in 1:nprocs
        chunk_other[p] = div(other_n, nprocs) + (p - 1 < mod(other_n, nprocs) ? 1 : 0)
        chunk_dim[p] = div(global_n, nprocs) + (p - 1 < mod(global_n, nprocs) ? 1 : 0)
    end
    chunk_dim_me = chunk_dim[rank + 1]

    # For reverse: we split along dim (now global_n) and assemble along other_dim
    remaining = div(prod(dims), dims[dim] * dims[other_dim])

    send_counts = Vector{Int}(undef, nprocs)
    recv_counts = Vector{Int}(undef, nprocs)
    send_displs = Vector{Int}(undef, nprocs)
    recv_displs = Vector{Int}(undef, nprocs)
    send_offset = 0
    recv_offset = 0
    for p in 1:nprocs
        # Send dim-chunks to each rank
        send_counts[p] = chunk_dim[p] * dims[other_dim] * remaining
        # Receive other_dim-chunks from each rank
        recv_counts[p] = chunk_other[p] * chunk_dim_me * remaining
        send_displs[p] = send_offset
        recv_displs[p] = recv_offset
        send_offset += send_counts[p]
        recv_offset += recv_counts[p]
    end

    # Pack: split along dim
    send_buf = Vector{T}(undef, send_offset)
    _cpu_pack_for_transpose!(send_buf, data, dim, chunk_dim, send_displs)

    # All-to-all
    recv_buf = Vector{T}(undef, recv_offset)
    sendbuf = MPI.VBuffer(send_buf, send_counts, send_displs)
    recvbuf = MPI.VBuffer(recv_buf, recv_counts, recv_displs)
    MPI.Alltoallv!(sendbuf, recvbuf, comm)

    # Unpack: assemble along other_dim
    # Output shape: original dims before forward transpose
    out_dims = collect(size(data))
    out_dims[dim] = chunk_dim_me
    out_dims[other_dim] = other_n
    output = Array{T}(undef, Tuple(out_dims)...)
    _cpu_unpack_from_transpose!(output, recv_buf, other_dim, chunk_other, recv_displs)

    return output
end

"""
    _cpu_pack_for_transpose!(send_buf, data, split_dim, chunk_sizes, displs)

CPU pack: rearrange data so each rank's portion along `split_dim` is contiguous.
"""
function _cpu_pack_for_transpose!(send_buf::Vector, data::Array, split_dim::Int,
                                   chunk_sizes::Vector{Int}, displs::Vector{Int})
    dims = size(data)
    nranks = length(chunk_sizes)
    total_split = dims[split_dim]

    # Precompute rank lookup table: rank_for_idx[i] = rank that owns index i
    # This replaces the O(nranks) linear search per element with O(1) lookup.
    rank_for_idx = Vector{Int}(undef, total_split)
    cumulative = 0
    for r in 1:nranks
        for i in 1:chunk_sizes[r]
            cumulative += 1
            rank_for_idx[cumulative] = r
        end
    end

    # Track write positions for each rank
    write_pos = copy(displs) .+ 1  # 1-indexed write positions

    # Iterate over all elements and place them in the correct rank's buffer section
    for idx in CartesianIndices(data)
        rank = rank_for_idx[idx[split_dim]]
        send_buf[write_pos[rank]] = data[idx]
        write_pos[rank] += 1
    end
end

"""
    _cpu_unpack_from_transpose!(output, recv_buf, assemble_dim, chunk_sizes, displs)

CPU unpack: reassemble data from per-rank chunks along `assemble_dim`.
"""
function _cpu_unpack_from_transpose!(output::Array, recv_buf::Vector, assemble_dim::Int,
                                      chunk_sizes::Vector{Int}, displs::Vector{Int})
    out_dims = size(output)
    nranks = length(chunk_sizes)
    total_assemble = out_dims[assemble_dim]

    # Precompute rank lookup table: O(1) per element instead of O(nranks)
    rank_for_idx = Vector{Int}(undef, total_assemble)
    cumulative = 0
    for r in 1:nranks
        for i in 1:chunk_sizes[r]
            cumulative += 1
            rank_for_idx[cumulative] = r
        end
    end

    # Track read positions for each rank
    read_pos = copy(displs) .+ 1  # 1-indexed read positions

    # Iterate over all output elements and read from correct rank's buffer section
    for idx in CartesianIndices(output)
        rank = rank_for_idx[idx[assemble_dim]]
        output[idx] = recv_buf[read_pos[rank]]
        read_pos[rank] += 1
    end
end

