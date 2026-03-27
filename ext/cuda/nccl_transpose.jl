# ============================================================================
# NCCL-based Transpose for Pencil Decomposition
# ============================================================================

"""
    NCCLTransposeBuffer{T}

Pre-allocated buffers for NCCL all-to-all transpose operations.

This struct provides efficient GPU-to-GPU communication for pencil decomposition
transposes using NCCL grouped send/recv operations. Since NCCL does not have
native Alltoallv, we implement it using grouped point-to-point operations.

# Fields
- `send_buffer`: Flattened GPU buffer for outgoing data
- `recv_buffer`: Flattened GPU buffer for incoming data
- `send_counts`: Number of elements to send to each peer
- `recv_counts`: Number of elements to receive from each peer
- `send_displs`: Starting offsets in send buffer for each peer
- `recv_displs`: Starting offsets in recv buffer for each peer
- `nccl_subcomms`: NCCL sub-communicators for row/col communication
- `pencil`: Reference to the pencil decomposition
"""
struct NCCLTransposeBuffer{T}
    # Send/recv buffers (flattened for all-to-all)
    send_buffer::CuArray{T, 1}
    recv_buffer::CuArray{T, 1}

    # Counts and displacements for each peer
    send_counts::Vector{Int}
    recv_counts::Vector{Int}
    send_displs::Vector{Int}
    recv_displs::Vector{Int}

    # NCCL sub-communicators (from Task 4)
    nccl_subcomms::Tarang.NCCLSubComms

    # Pencil reference
    pencil::PencilDecomposition
end

"""
    NCCLTransposeBuffer(pencil::PencilDecomposition, T::Type)

Create transpose buffers for the given pencil decomposition.

Allocates GPU buffers sized for the largest local pencil shape to accommodate
any transpose direction. Initializes NCCL sub-communicators for row and column
communication groups.

# Arguments
- `pencil`: The pencil decomposition describing the domain layout
- `T`: Element type for the buffers (e.g., Float64, Float32)

# Returns
- `NCCLTransposeBuffer{T}`: Pre-allocated buffers ready for transpose operations
"""
function NCCLTransposeBuffer(pencil::PencilDecomposition, T::Type)
    # Calculate buffer size based on largest local pencil
    max_local = max(
        prod(pencil.x_pencil_shape),
        prod(pencil.y_pencil_shape),
        prod(pencil.z_pencil_shape)
    )

    P1, P2 = pencil.proc_grid
    max_peers = max(P1, P2)

    # Allocate flattened buffers
    send_buffer = CUDA.zeros(T, max_local)
    recv_buffer = CUDA.zeros(T, max_local)

    # Initialize NCCL sub-communicators
    nccl_subcomms = Tarang.init_nccl_subcomms!(pencil.row_comm, pencil.col_comm)

    return NCCLTransposeBuffer{T}(
        send_buffer, recv_buffer,
        zeros(Int, max_peers),
        zeros(Int, max_peers),
        zeros(Int, max_peers),
        zeros(Int, max_peers),
        nccl_subcomms,
        pencil
    )
end

# ============================================================================
# Pack/Unpack Kernels for Pencil Transposes
# ============================================================================

"""
Pack kernel for Z->Y transpose.
Reorganizes data from Z-pencil layout to prepare for all-to-all communication.
Each element is placed contiguously for its destination rank.
"""
@kernel function pack_z_to_y_kernel!(packed, @Const(data), Nx, Ny, Nz,
                                      @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz
    if idx > total
        return
    end

    # Calculate 3D indices from linear index (column-major)
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this Z-slice goes to (split Z, the fully-local dimension)
    rank = 1
    z_offset = 0
    for r in 1:nranks
        if k <= z_offset + chunk_sizes[r]
            rank = r
            break
        end
        z_offset += chunk_sizes[r]
    end

    # Compute position within this rank's chunk
    local_k = k - z_offset
    local_idx = (local_k - 1) * Nx * Ny + (j - 1) * Nx + i
    buf_idx = displs[rank] + local_idx

    @inbounds packed[buf_idx] = data[i, j, k]
end

"""
Unpack kernel for Z->Y transpose.
Reorganizes received data into Y-pencil layout after all-to-all communication.
"""
@kernel function unpack_z_to_y_kernel!(data, @Const(packed), Nx, Ny, Nz,
                                        @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz
    if idx > total
        return
    end

    # Calculate 3D indices for Y-pencil layout (Nx, Ny_global, Nz_local)
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank contributed this Y-slice (each peer sent its Ny_local chunk)
    rank = 1
    y_offset = 0
    for r in 1:nranks
        if j <= y_offset + chunk_sizes[r]
            rank = r
            break
        end
        y_offset += chunk_sizes[r]
    end

    # Compute position within this rank's received chunk: (Nx, Ny_r, Nz_local)
    local_j = j - y_offset
    local_idx = (k - 1) * Nx * chunk_sizes[rank] + (local_j - 1) * Nx + i
    buf_idx = displs[rank] + local_idx

    @inbounds data[i, j, k] = packed[buf_idx]
end

"""
Pack kernel for Y->Z transpose (reverse of Z->Y).
Partitions the Y dimension and packs data for all-to-all communication.
Each element is placed contiguously for its destination rank.

In Y-pencil: (Nx_local, Ny, Nz_local) where Ny is full, Nz is partitioned
We send Y-chunks to each rank, keeping our Nz_local portion.
"""
@kernel function pack_y_to_z_kernel!(packed, @Const(data), Nx, Ny, Nz_local,
                                      @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz_local
    if idx > total
        return
    end

    # Calculate 3D indices from linear index (column-major)
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this Y-slice goes to (partition by Y)
    rank = 1
    y_offset = 0
    for r in 1:nranks
        if j <= y_offset + chunk_sizes[r]
            rank = r
            break
        end
        y_offset += chunk_sizes[r]
    end

    # Compute position within this rank's chunk
    # Layout within chunk: (Nx, chunk_size_y, Nz_local)
    local_j = j - y_offset
    local_idx = (k - 1) * Nx * chunk_sizes[rank] + (local_j - 1) * Nx + i
    buf_idx = displs[rank] + local_idx

    @inbounds packed[buf_idx] = data[i, j, k]
end

"""
Unpack kernel for Y->Z transpose.
Reassembles received Z-chunks into Z-pencil layout.

After communication, we receive Nz_chunk from each rank.
Output is Z-pencil: (Nx_local, Ny_local, Nz) where Nz is now full.
"""
@kernel function unpack_y_to_z_kernel!(data, @Const(packed), Nx, Ny_local, Nz,
                                        @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny_local * Nz
    if idx > total
        return
    end

    # Calculate 3D indices for Z-pencil layout
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny_local) + 1
    k = ((idx - 1) ÷ (Nx * Ny_local)) + 1

    # Determine which rank contributed this Z-slice
    rank = 1
    z_offset = 0
    for r in 1:nranks
        if k <= z_offset + chunk_sizes[r]
            rank = r
            break
        end
        z_offset += chunk_sizes[r]
    end

    # Compute position within this rank's received chunk
    # Received layout: (Nx, Ny_local, chunk_size_z)
    local_k = k - z_offset
    local_idx = (local_k - 1) * Nx * Ny_local + (j - 1) * Nx + i
    buf_idx = displs[rank] + local_idx

    @inbounds data[i, j, k] = packed[buf_idx]
end

"""
Pack kernel for Y->X transpose.
Reorganizes data from Y-pencil layout for column communicator all-to-all.
"""
@kernel function pack_y_to_x_kernel!(packed, @Const(data), Nx, Ny, Nz,
                                      @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz
    if idx > total
        return
    end

    # Calculate 3D indices from linear index
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this Y-slice goes to (split Y, the fully-local dimension)
    rank = 1
    y_offset = 0
    for r in 1:nranks
        if j <= y_offset + chunk_sizes[r]
            rank = r
            break
        end
        y_offset += chunk_sizes[r]
    end

    # Compute position within this rank's chunk: (Nx, Ny_r, Nz)
    local_j = j - y_offset
    local_idx = (k - 1) * Nx * chunk_sizes[rank] + (local_j - 1) * Nx + i
    buf_idx = displs[rank] + local_idx

    @inbounds packed[buf_idx] = data[i, j, k]
end

"""
Unpack kernel for Y->X transpose.
Reorganizes received data into X-pencil layout.
"""
@kernel function unpack_y_to_x_kernel!(data, @Const(packed), Nx, Ny, Nz,
                                        @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz
    if idx > total
        return
    end

    # Calculate 3D indices for X-pencil layout (Nx_global, Ny_local, Nz_local)
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank contributed this X-slice (each peer sent its Nx_local chunk)
    rank = 1
    x_offset = 0
    for r in 1:nranks
        if i <= x_offset + chunk_sizes[r]
            rank = r
            break
        end
        x_offset += chunk_sizes[r]
    end

    # Compute position within this rank's received chunk: (Nx_r, Ny, Nz)
    local_i = i - x_offset
    local_idx = (k - 1) * chunk_sizes[rank] * Ny + (j - 1) * chunk_sizes[rank] + local_i
    buf_idx = displs[rank] + local_idx

    @inbounds data[i, j, k] = packed[buf_idx]
end

"""
Pack kernel for X->Y transpose.
Reorganizes data from X-pencil layout for column communicator all-to-all.
Splits the X dimension among ranks (the reverse of the Y->X unpack).
"""
@kernel function pack_x_to_y_kernel!(packed, @Const(data), Nx, Ny, Nz,
                                      @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz
    if idx > total
        return
    end

    # Calculate 3D indices from linear index (X-pencil: Nx, Ny_local, Nz_local)
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this X-slice goes to (split X among ranks)
    rank = 1
    x_offset = 0
    for r in 1:nranks
        if i <= x_offset + chunk_sizes[r]
            rank = r
            break
        end
        x_offset += chunk_sizes[r]
    end

    # Compute position within this rank's chunk: (Nx_r, Ny, Nz)
    local_i = i - x_offset
    local_idx = (k - 1) * chunk_sizes[rank] * Ny + (j - 1) * chunk_sizes[rank] + local_i
    buf_idx = displs[rank] + local_idx

    @inbounds packed[buf_idx] = data[i, j, k]
end

"""
Unpack kernel for X->Y transpose.
Reorganizes received data into Y-pencil layout.
Gathers Y-slices from each rank (the reverse of the Y->X pack).
"""
@kernel function unpack_x_to_y_kernel!(data, @Const(packed), Nx, Ny, Nz,
                                        @Const(chunk_sizes), @Const(displs), nranks)
    idx = @index(Global)

    total = Nx * Ny * Nz
    if idx > total
        return
    end

    # Calculate 3D indices for Y-pencil layout (Nx_local, Ny_global, Nz_local)
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank contributed this Y-slice
    rank = 1
    y_offset = 0
    for r in 1:nranks
        if j <= y_offset + chunk_sizes[r]
            rank = r
            break
        end
        y_offset += chunk_sizes[r]
    end

    # Compute position within this rank's received chunk: (Nx, Ny_r, Nz)
    local_j = j - y_offset
    local_idx = (k - 1) * Nx * chunk_sizes[rank] + (local_j - 1) * Nx + i
    buf_idx = displs[rank] + local_idx

    @inbounds data[i, j, k] = packed[buf_idx]
end

# ============================================================================
# Simplified Pack/Unpack Interface (for testing)
# ============================================================================

"""
    nccl_pack_for_transpose!(packed::CuArray{T}, data::CuArray{T,3}, dim::Int) where T

Pack 3D array for transpose along specified dimension.

This is a simplified interface for testing. For actual transposes, use the
full transpose_z_to_y!, transpose_y_to_x!, etc. functions.

# Arguments
- `packed`: Output buffer (flattened)
- `data`: Input 3D array
- `dim`: Dimension along which to prepare for transpose (1=X, 2=Y, 3=Z)
"""
function nccl_pack_for_transpose!(packed::CuArray{T}, data::CuArray{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(data)
    total = Nx * Ny * Nz

    # For testing: simple copy (actual pack logic is in transpose functions)
    if dim == 3 || dim == 2 || dim == 1
        copyto!(view(packed, 1:total), reshape(data, :))
    end

    return packed
end

"""
    nccl_unpack_from_transpose!(data::CuArray{T,3}, packed::CuArray{T}, dim::Int) where T

Unpack 3D array after transpose along specified dimension.

This is a simplified interface for testing. For actual transposes, use the
full transpose_z_to_y!, transpose_y_to_x!, etc. functions.

# Arguments
- `data`: Output 3D array
- `packed`: Input buffer (flattened)
- `dim`: Dimension along which transpose was performed (1=X, 2=Y, 3=Z)
"""
function nccl_unpack_from_transpose!(data::CuArray{T,3}, packed::CuArray{T}, dim::Int) where T
    Nx, Ny, Nz = size(data)
    total = Nx * Ny * Nz

    # For testing: simple copy (actual unpack logic is in transpose functions)
    if dim == 3 || dim == 2 || dim == 1
        copyto!(reshape(data, :), view(packed, 1:total))
    end

    return data
end

# ============================================================================
# NCCL All-to-All Operations
# ============================================================================

"""
    nccl_alltoall!(send_buf, recv_buf, send_counts, recv_counts,
                   send_displs, recv_displs, nccl_comm)

Perform all-to-all using NCCL grouped send/recv operations.

NCCL does not have a native Alltoallv operation, so we implement it using
grouped point-to-point send!/recv! operations wrapped in group_start/group_end.

# Arguments
- `send_buf`: GPU buffer containing data to send
- `recv_buf`: GPU buffer to receive data into
- `send_counts`: Number of elements to send to each rank
- `recv_counts`: Number of elements to receive from each rank
- `send_displs`: Starting offset in send_buf for each rank (0-indexed)
- `recv_displs`: Starting offset in recv_buf for each rank (0-indexed)
- `nccl_comm`: NCCL communicator (or nothing for single-rank case)
"""
function nccl_alltoall!(send_buf::CuArray, recv_buf::CuArray,
                         send_counts::Vector{Int}, recv_counts::Vector{Int},
                         send_displs::Vector{Int}, recv_displs::Vector{Int},
                         nccl_comm; my_rank::Int=-1)
    # CRITICAL: Use error() instead of @assert for production safety
    total_send = sum(send_counts)
    total_recv = sum(recv_counts)
    if total_send > length(send_buf)
        error("nccl_alltoall!: Send counts ($total_send) exceed buffer size ($(length(send_buf)))")
    end
    if total_recv > length(recv_buf)
        error("nccl_alltoall!: Recv counts ($total_recv) exceed buffer size ($(length(recv_buf)))")
    end

    if nccl_comm === nothing
        # Single rank - just copy
        total = sum(send_counts)
        if total > 0
            copyto!(view(recv_buf, 1:total), view(send_buf, 1:total))
        end
        return
    end

    nranks = length(send_counts)

    # Ensure NCCL is loaded (should be loaded by init_nccl_subcomms!, but verify)
    if !isdefined(@__MODULE__, :NCCL)
        @eval using NCCL
    end

    # Handle self-communication with direct copyto! (NCCL P2P does not support self-send/recv)
    if my_rank >= 0 && send_counts[my_rank+1] > 0 && recv_counts[my_rank+1] > 0
        self_send_start = send_displs[my_rank+1] + 1
        self_send_end = self_send_start + send_counts[my_rank+1] - 1
        self_recv_start = recv_displs[my_rank+1] + 1
        self_recv_end = self_recv_start + recv_counts[my_rank+1] - 1
        copyto!(view(recv_buf, self_recv_start:self_recv_end),
                view(send_buf, self_send_start:self_send_end))
    end

    # NCCL grouped P2P for all remote peers
    NCCL.group_start()

    for peer in 0:(nranks-1)
        # Skip self — already handled above
        if peer == my_rank
            continue
        end

        if send_counts[peer+1] > 0
            send_start = send_displs[peer+1] + 1
            send_end = send_start + send_counts[peer+1] - 1
            send_slice = view(send_buf, send_start:send_end)
            NCCL.send!(send_slice, peer, nccl_comm)
        end

        if recv_counts[peer+1] > 0
            recv_start = recv_displs[peer+1] + 1
            recv_end = recv_start + recv_counts[peer+1] - 1
            recv_slice = view(recv_buf, recv_start:recv_end)
            NCCL.recv!(recv_slice, peer, nccl_comm)
        end
    end

    NCCL.group_end()

    CUDA.synchronize()
end

# ============================================================================
# Transpose Operations
# ============================================================================

"""
    transpose_z_to_y!(buffer::NCCLTransposeBuffer, data::CuArray, pencil::PencilDecomposition)

Transpose from Z-pencil to Y-pencil layout using NCCL all-to-all.

In Z-pencil layout, the Z dimension is fully local on each rank.
After transpose, the Y dimension becomes fully local.

NOTE: This implementation is optimized for uniform decompositions where the grid
divides evenly. Non-uniform decompositions (where some ranks have different chunk
sizes) may require more sophisticated handling in future versions.

# Arguments
- `buffer`: Pre-allocated transpose buffers
- `data`: Input data in Z-pencil layout
- `pencil`: Pencil decomposition describing the domain layout

# Returns
- Output data in Y-pencil layout (newly allocated)
"""
function transpose_z_to_y!(buffer::NCCLTransposeBuffer{T},
                            data::CuArray{T, 3},
                            pencil::PencilDecomposition) where T
    # CRITICAL: Use error() instead of @assert for production safety
    if current_orientation(pencil) != :z_pencil
        error("transpose_z_to_y!: Must be in Z-pencil orientation, currently in $(current_orientation(pencil))")
    end

    P1, P2 = pencil.proc_grid
    row_size = MPI.Comm_size(pencil.row_comm)

    Nx_local, Ny_local, Nz = size(data)
    total = Nx_local * Ny_local * Nz

    # For single-rank row comm, just reshape
    if row_size == 1
        set_orientation!(pencil, :y_pencil)
        output = CUDA.zeros(T, pencil.y_pencil_shape...)
        copyto!(reshape(output, :), reshape(data, :))
        return output
    end

    # Compute counts: split Z (fully local) among row_size peers, gather Y
    Ny_global = pencil.global_shape[2]
    row_rank = MPI.Comm_rank(pencil.row_comm)
    Nz_me = div(Nz, row_size) + (row_rank < mod(Nz, row_size) ? 1 : 0)
    for i in 1:row_size
        Nz_i = div(Nz, row_size) + ((i-1) < mod(Nz, row_size) ? 1 : 0)
        Ny_i = div(Ny_global, row_size) + ((i-1) < mod(Ny_global, row_size) ? 1 : 0)
        # Send: our (Nx_local, Ny_local) face × Nz_i z-slices for rank i
        buffer.send_counts[i] = Nx_local * Ny_local * Nz_i
        # Recv: rank i's Ny_i y-chunk × our Nz_me z-slices
        buffer.recv_counts[i] = Nx_local * Ny_i * Nz_me
    end

    # Compute displacements
    buffer.send_displs[1] = 0
    buffer.recv_displs[1] = 0
    for i in 2:row_size
        buffer.send_displs[i] = buffer.send_displs[i-1] + buffer.send_counts[i-1]
        buffer.recv_displs[i] = buffer.recv_displs[i-1] + buffer.recv_counts[i-1]
    end

    # Pack data: split by Z dimension (z_chunk_sizes for each rank)
    chunk_sizes_gpu = CuArray(Int[div(Nz, row_size) + ((i-1) < mod(Nz, row_size) ? 1 : 0) for i in 1:row_size])
    displs_gpu = CuArray(buffer.send_displs[1:row_size])

    kernel = pack_z_to_y_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx_local, Ny_local, Nz,
           chunk_sizes_gpu, displs_gpu, row_size; ndrange=total)
    CUDA.synchronize()

    # Perform NCCL all-to-all (pass my_rank to avoid self-send deadlock)
    row_rank = MPI.Comm_rank(pencil.row_comm)
    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:row_size], buffer.recv_counts[1:row_size],
        buffer.send_displs[1:row_size], buffer.recv_displs[1:row_size],
        buffer.nccl_subcomms.row_comm; my_rank=row_rank
    )

    # Create output in Y-pencil shape and unpack
    output = CUDA.zeros(T, pencil.y_pencil_shape...)
    Nx_y, Ny_y, Nz_y = pencil.y_pencil_shape

    # Unpack: each peer contributed its Ny_local chunk (Y-chunks)
    recv_chunk_sizes_gpu = CuArray(Int[div(Ny_global, row_size) + ((i-1) < mod(Ny_global, row_size) ? 1 : 0) for i in 1:row_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:row_size])

    kernel_unpack = unpack_z_to_y_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_y, Ny_y, Nz_y,
                  recv_chunk_sizes_gpu, recv_displs_gpu, row_size; ndrange=prod(pencil.y_pencil_shape))
    CUDA.synchronize()

    set_orientation!(pencil, :y_pencil)
    return output
end

"""
    transpose_y_to_z!(buffer::NCCLTransposeBuffer, data::CuArray, pencil::PencilDecomposition)

Transpose from Y-pencil to Z-pencil layout (inverse of Z->Y).

NOTE: This implementation is optimized for uniform decompositions where the grid
divides evenly. Non-uniform decompositions may require more sophisticated handling
in future versions.

# Arguments
- `buffer`: Pre-allocated transpose buffers
- `data`: Input data in Y-pencil layout
- `pencil`: Pencil decomposition describing the domain layout

# Returns
- Output data in Z-pencil layout (newly allocated)
"""
function transpose_y_to_z!(buffer::NCCLTransposeBuffer{T},
                            data::CuArray{T, 3},
                            pencil::PencilDecomposition) where T
    # CRITICAL: Use error() instead of @assert for production safety
    if current_orientation(pencil) != :y_pencil
        error("transpose_y_to_z!: Must be in Y-pencil orientation, currently in $(current_orientation(pencil))")
    end

    P1, P2 = pencil.proc_grid
    row_size = MPI.Comm_size(pencil.row_comm)

    Nx_local, Ny, Nz_local = size(data)
    total = Nx_local * Ny * Nz_local

    if row_size == 1
        set_orientation!(pencil, :z_pencil)
        output = CUDA.zeros(T, pencil.z_pencil_shape...)
        copyto!(reshape(output, :), reshape(data, :))
        return output
    end

    # For Y->Z transpose: partition Y (send Y-chunks), receive Z-chunks
    # Y-pencil: (Nx_local, Ny, Nz_local) where Ny is full, Nz is partitioned
    # Z-pencil: (Nx_local, Ny_local, Nz) where Ny is partitioned, Nz is full
    Nz_global = pencil.global_shape[3]
    Ny_local_after = pencil.z_pencil_shape[2]  # Ny/row_size after transpose

    for i in 1:row_size
        # CRITICAL FIX: Correct counts for Y→Z transpose
        # Send: partition Ny dimension for each destination rank
        Ny_chunk_for_i = div(Ny, row_size) + ((i-1) < mod(Ny, row_size) ? 1 : 0)
        buffer.send_counts[i] = Nx_local * Ny_chunk_for_i * Nz_local

        # Recv: receive Nz_chunk from each source rank
        Nz_chunk_from_i = div(Nz_global, row_size) + ((i-1) < mod(Nz_global, row_size) ? 1 : 0)
        buffer.recv_counts[i] = Nx_local * Ny_local_after * Nz_chunk_from_i
    end

    buffer.send_displs[1] = 0
    buffer.recv_displs[1] = 0
    for i in 2:row_size
        buffer.send_displs[i] = buffer.send_displs[i-1] + buffer.send_counts[i-1]
        buffer.recv_displs[i] = buffer.recv_displs[i-1] + buffer.recv_counts[i-1]
    end

    # CRITICAL FIX: Use proper pack kernel for uneven decomposition
    # Partition Y dimension and pack data for each destination rank
    chunk_sizes_gpu = CuArray(Int[div(Ny, row_size) + ((i-1) < mod(Ny, row_size) ? 1 : 0) for i in 1:row_size])
    displs_gpu = CuArray(buffer.send_displs[1:row_size])

    kernel = pack_y_to_z_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx_local, Ny, Nz_local,
           chunk_sizes_gpu, displs_gpu, row_size; ndrange=total)
    CUDA.synchronize()

    row_rank = MPI.Comm_rank(pencil.row_comm)
    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:row_size], buffer.recv_counts[1:row_size],
        buffer.send_displs[1:row_size], buffer.recv_displs[1:row_size],
        buffer.nccl_subcomms.row_comm; my_rank=row_rank
    )

    # CRITICAL FIX: Use proper unpack kernel
    # Reassemble Z-chunks from each rank into contiguous Z dimension
    output = CUDA.zeros(T, pencil.z_pencil_shape...)
    Nx_z, Ny_z, Nz_z = pencil.z_pencil_shape

    recv_chunk_sizes_gpu = CuArray(Int[div(Nz_global, row_size) + ((i-1) < mod(Nz_global, row_size) ? 1 : 0) for i in 1:row_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:row_size])

    kernel_unpack = unpack_y_to_z_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_z, Ny_z, Nz_z,
                  recv_chunk_sizes_gpu, recv_displs_gpu, row_size; ndrange=prod(pencil.z_pencil_shape))
    CUDA.synchronize()

    set_orientation!(pencil, :z_pencil)
    return output
end

"""
    transpose_y_to_x!(buffer::NCCLTransposeBuffer, data::CuArray, pencil::PencilDecomposition)

Transpose from Y-pencil to X-pencil layout using column communicator.

In Y-pencil layout, the Y dimension is fully local on each rank.
After transpose, the X dimension becomes fully local.

NOTE: This implementation is optimized for uniform decompositions where the grid
divides evenly. Non-uniform decompositions may require more sophisticated handling
in future versions.

# Arguments
- `buffer`: Pre-allocated transpose buffers
- `data`: Input data in Y-pencil layout
- `pencil`: Pencil decomposition describing the domain layout

# Returns
- Output data in X-pencil layout (newly allocated)
"""
function transpose_y_to_x!(buffer::NCCLTransposeBuffer{T},
                            data::CuArray{T, 3},
                            pencil::PencilDecomposition) where T
    # CRITICAL: Use error() instead of @assert for production safety
    if current_orientation(pencil) != :y_pencil
        error("transpose_y_to_x!: Must be in Y-pencil orientation, currently in $(current_orientation(pencil))")
    end

    P1, P2 = pencil.proc_grid
    col_size = MPI.Comm_size(pencil.col_comm)

    Nx_local, Ny, Nz_local = size(data)
    total = Nx_local * Ny * Nz_local

    if col_size == 1
        set_orientation!(pencil, :x_pencil)
        output = CUDA.zeros(T, pencil.x_pencil_shape...)
        copyto!(reshape(output, :), reshape(data, :))
        return output
    end

    # Compute counts: split Y (fully local) among col_size peers, gather X
    Nx_global = pencil.global_shape[1]
    col_rank = MPI.Comm_rank(pencil.col_comm)
    Ny_me = div(Ny, col_size) + (col_rank < mod(Ny, col_size) ? 1 : 0)
    for i in 1:col_size
        Ny_i = div(Ny, col_size) + ((i-1) < mod(Ny, col_size) ? 1 : 0)
        Nx_i = div(Nx_global, col_size) + ((i-1) < mod(Nx_global, col_size) ? 1 : 0)
        # Send: our (Nx_local, Nz_local) face × Ny_i y-slices for rank i
        buffer.send_counts[i] = Nx_local * Ny_i * Nz_local
        # Recv: rank i's Nx_i x-chunk × our Ny_me y-slices
        buffer.recv_counts[i] = Nx_i * Ny_me * Nz_local
    end

    buffer.send_displs[1] = 0
    buffer.recv_displs[1] = 0
    for i in 2:col_size
        buffer.send_displs[i] = buffer.send_displs[i-1] + buffer.send_counts[i-1]
        buffer.recv_displs[i] = buffer.recv_displs[i-1] + buffer.recv_counts[i-1]
    end

    # Pack data: split by Y dimension (y_chunk_sizes for each rank)
    chunk_sizes_gpu = CuArray(Int[div(Ny, col_size) + ((i-1) < mod(Ny, col_size) ? 1 : 0) for i in 1:col_size])
    displs_gpu = CuArray(buffer.send_displs[1:col_size])

    kernel = pack_y_to_x_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx_local, Ny, Nz_local,
           chunk_sizes_gpu, displs_gpu, col_size; ndrange=total)
    CUDA.synchronize()

    # Perform NCCL all-to-all on column communicator (pass my_rank to avoid self-send deadlock)
    col_rank = MPI.Comm_rank(pencil.col_comm)
    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:col_size], buffer.recv_counts[1:col_size],
        buffer.send_displs[1:col_size], buffer.recv_displs[1:col_size],
        buffer.nccl_subcomms.col_comm; my_rank=col_rank
    )

    # Create output in X-pencil shape and unpack
    output = CUDA.zeros(T, pencil.x_pencil_shape...)
    Nx_x, Ny_x, Nz_x = pencil.x_pencil_shape

    # Unpack: each peer contributed its Nx_local chunk (X-chunks)
    recv_chunk_sizes_gpu = CuArray(Int[div(Nx_global, col_size) + ((i-1) < mod(Nx_global, col_size) ? 1 : 0) for i in 1:col_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:col_size])

    kernel_unpack = unpack_y_to_x_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_x, Ny_x, Nz_x,
                  recv_chunk_sizes_gpu, recv_displs_gpu, col_size; ndrange=prod(pencil.x_pencil_shape))
    CUDA.synchronize()

    set_orientation!(pencil, :x_pencil)
    return output
end

"""
    transpose_x_to_y!(buffer::NCCLTransposeBuffer, data::CuArray, pencil::PencilDecomposition)

Transpose from X-pencil to Y-pencil layout (inverse of Y->X).

NOTE: This implementation is optimized for uniform decompositions where the grid
divides evenly. Non-uniform decompositions may require more sophisticated handling
in future versions.

# Arguments
- `buffer`: Pre-allocated transpose buffers
- `data`: Input data in X-pencil layout
- `pencil`: Pencil decomposition describing the domain layout

# Returns
- Output data in Y-pencil layout (newly allocated)
"""
function transpose_x_to_y!(buffer::NCCLTransposeBuffer{T},
                            data::CuArray{T, 3},
                            pencil::PencilDecomposition) where T
    # CRITICAL: Use error() instead of @assert for production safety
    if current_orientation(pencil) != :x_pencil
        error("transpose_x_to_y!: Must be in X-pencil orientation, currently in $(current_orientation(pencil))")
    end

    P1, P2 = pencil.proc_grid
    col_size = MPI.Comm_size(pencil.col_comm)

    Nx, Ny_local, Nz_local = size(data)
    total = Nx * Ny_local * Nz_local

    if col_size == 1
        set_orientation!(pencil, :y_pencil)
        output = CUDA.zeros(T, pencil.y_pencil_shape...)
        copyto!(reshape(output, :), reshape(data, :))
        return output
    end

    # For X->Y transpose, we split X among ranks and gather Y
    Nx_global = Nx  # In X-pencil, Nx is the full global X dimension
    Ny_global = pencil.global_shape[2]
    col_rank = MPI.Comm_rank(pencil.col_comm)

    for i in 1:col_size
        Nx_i = div(Nx_global, col_size) + ((i-1) < mod(Nx_global, col_size) ? 1 : 0)
        Ny_i = div(Ny_global, col_size) + ((i-1) < mod(Ny_global, col_size) ? 1 : 0)
        # Send: Nx_i x-slices × our (Ny_local, Nz_local)
        buffer.send_counts[i] = Nx_i * Ny_local * Nz_local
        # Recv: rank i's Ny_i y-slices × our Nx_local
        Nx_me = div(Nx_global, col_size) + (col_rank < mod(Nx_global, col_size) ? 1 : 0)
        buffer.recv_counts[i] = Nx_me * Ny_i * Nz_local
    end

    buffer.send_displs[1] = 0
    buffer.recv_displs[1] = 0
    for i in 2:col_size
        buffer.send_displs[i] = buffer.send_displs[i-1] + buffer.send_counts[i-1]
        buffer.recv_displs[i] = buffer.recv_displs[i-1] + buffer.recv_counts[i-1]
    end

    # Pack data: split by X dimension using GPU kernel
    chunk_sizes_gpu = CuArray(Int[div(Nx_global, col_size) + ((i-1) < mod(Nx_global, col_size) ? 1 : 0) for i in 1:col_size])
    displs_gpu = CuArray(buffer.send_displs[1:col_size])

    kernel = pack_x_to_y_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx, Ny_local, Nz_local,
           chunk_sizes_gpu, displs_gpu, col_size; ndrange=total)
    CUDA.synchronize()

    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:col_size], buffer.recv_counts[1:col_size],
        buffer.send_displs[1:col_size], buffer.recv_displs[1:col_size],
        buffer.nccl_subcomms.col_comm; my_rank=col_rank
    )

    # Unpack into Y-pencil layout using GPU kernel
    output = CUDA.zeros(T, pencil.y_pencil_shape...)
    Nx_y, Ny_y, Nz_y = pencil.y_pencil_shape

    recv_chunk_sizes_gpu = CuArray(Int[div(Ny_global, col_size) + ((i-1) < mod(Ny_global, col_size) ? 1 : 0) for i in 1:col_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:col_size])

    kernel_unpack = unpack_x_to_y_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_y, Ny_y, Nz_y,
                  recv_chunk_sizes_gpu, recv_displs_gpu, col_size; ndrange=prod(pencil.y_pencil_shape))
    CUDA.synchronize()

    set_orientation!(pencil, :y_pencil)
    return output
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    compute_transpose_counts!(buffer::NCCLTransposeBuffer, direction::Symbol)

Compute send/recv counts for the specified transpose direction.

# Arguments
- `buffer`: Transpose buffer to update counts in
- `direction`: One of :z_to_y, :y_to_z, :y_to_x, :x_to_y
"""
function compute_transpose_counts!(buffer::NCCLTransposeBuffer, direction::Symbol)
    pencil = buffer.pencil

    if direction == :z_to_y || direction == :y_to_z
        comm_size = MPI.Comm_size(pencil.row_comm)
    else
        comm_size = MPI.Comm_size(pencil.col_comm)
    end

    if direction == :z_to_y
        # Z->Y: distribute Y, gather Z
        Ny = pencil.global_shape[2]
        Nz = pencil.global_shape[3]
        local_shape = pencil.z_pencil_shape

        for i in 1:comm_size
            chunk_y = div(Ny, comm_size) + ((i-1) < mod(Ny, comm_size) ? 1 : 0)
            chunk_z = div(Nz, comm_size) + ((i-1) < mod(Nz, comm_size) ? 1 : 0)
            buffer.send_counts[i] = local_shape[1] * chunk_y * local_shape[3]
            buffer.recv_counts[i] = local_shape[1] * local_shape[2] * chunk_z
        end
    elseif direction == :y_to_x
        # Y->X: distribute X, gather Y
        Nx = pencil.global_shape[1]
        Ny = pencil.global_shape[2]
        local_shape = pencil.y_pencil_shape

        for i in 1:comm_size
            chunk_x = div(Nx, comm_size) + ((i-1) < mod(Nx, comm_size) ? 1 : 0)
            chunk_y = div(Ny, comm_size) + ((i-1) < mod(Ny, comm_size) ? 1 : 0)
            buffer.send_counts[i] = chunk_x * local_shape[2] * local_shape[3]
            buffer.recv_counts[i] = Nx * chunk_y * local_shape[3]
        end
    end

    # Compute displacements
    buffer.send_displs[1] = 0
    buffer.recv_displs[1] = 0
    for i in 2:comm_size
        buffer.send_displs[i] = buffer.send_displs[i-1] + buffer.send_counts[i-1]
        buffer.recv_displs[i] = buffer.recv_displs[i-1] + buffer.recv_counts[i-1]
    end
end

"""
    finalize_nccl_transpose!(buffer::NCCLTransposeBuffer)

Clean up NCCL transpose buffer resources.
"""
function finalize_nccl_transpose!(buffer::NCCLTransposeBuffer)
    Tarang.finalize_nccl_subcomms!(buffer.nccl_subcomms)
end
