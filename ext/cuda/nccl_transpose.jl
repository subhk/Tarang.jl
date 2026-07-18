# ============================================================================
# NCCL-based Transpose for Pencil Decomposition
# ============================================================================

# Binary search helpers for GPU kernels (shared with transpose_kernels.jl)
# Find which rank owns index `idx` given cumulative prefix sums.
@inline function _gpu_find_rank(idx::Int, prefix_sums, nranks::Int)
    lo = 1
    hi = nranks
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if idx <= prefix_sums[mid]
            hi = mid
        else
            lo = mid + 1
        end
    end
    @inbounds offset = lo > 1 ? prefix_sums[lo - 1] : 0
    return lo - 1, offset  # 0-based rank
end

@inline function _gpu_find_rank_1based(idx::Int, prefix_sums, nranks::Int)
    # Kernel pack/unpack code needs a 1-based rank for indexing Julia arrays
    # such as `chunk_sizes` and `displs`.
    lo = 1
    hi = nranks
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if idx <= prefix_sums[mid]
            hi = mid
        else
            lo = mid + 1
        end
    end
    @inbounds offset = lo > 1 ? prefix_sums[lo - 1] : 0
    return lo, offset  # 1-based rank
end

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
    # Allocate for the largest orientation so one buffer can serve every
    # transpose direction without reallocating between stages.
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

    # A multi-rank pencil without working NCCL sub-communicators must fail
    # loudly here: silently continuing would let the transposes degrade into
    # local self-copies and produce completely wrong results.
    row_size = MPI.Comm_size(pencil.row_comm)
    col_size = MPI.Comm_size(pencil.col_comm)
    if (row_size > 1 || col_size > 1) && !nccl_subcomms.initialized
        error("NCCLTransposeBuffer: NCCL sub-communicator initialization failed for a " *
              "multi-rank pencil (row_size=$row_size, col_size=$col_size). " *
              "Load NCCL.jl with `using NCCL` before multi-GPU transposes; " *
              "see the init_nccl_subcomms! warning for the underlying failure.")
    end

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
                                      @Const(chunk_sizes), @Const(displs), nranks,
                                      @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Z is the distributed dimension in this direction; use the prefix sums to
    # map each global z-index to its destination rank and local offset.
    rank, z_offset = _gpu_find_rank_1based(k, prefix_sums, nranks)
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
                                        @Const(chunk_sizes), @Const(displs), nranks,
                                        @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    # Y is the distributed dimension after the transpose, so received chunks
    # are unpacked according to the y-index owner.
    rank, y_offset = _gpu_find_rank_1based(j, prefix_sums, nranks)
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
                                      @Const(chunk_sizes), @Const(displs), nranks,
                                      @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz_local
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    rank, y_offset = _gpu_find_rank_1based(j, prefix_sums, nranks)
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
                                        @Const(chunk_sizes), @Const(displs), nranks,
                                        @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny_local * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny_local) + 1
    k = ((idx - 1) ÷ (Nx * Ny_local)) + 1

    rank, z_offset = _gpu_find_rank_1based(k, prefix_sums, nranks)
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
                                      @Const(chunk_sizes), @Const(displs), nranks,
                                      @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    rank, y_offset = _gpu_find_rank_1based(j, prefix_sums, nranks)
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
                                        @Const(chunk_sizes), @Const(displs), nranks,
                                        @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    rank, x_offset = _gpu_find_rank_1based(i, prefix_sums, nranks)
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
                                      @Const(chunk_sizes), @Const(displs), nranks,
                                      @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    rank, x_offset = _gpu_find_rank_1based(i, prefix_sums, nranks)
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
                                        @Const(chunk_sizes), @Const(displs), nranks,
                                        @Const(prefix_sums))
    idx = @index(Global)

    total = Nx * Ny * Nz
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    k = ((idx - 1) ÷ (Nx * Ny)) + 1

    rank, y_offset = _gpu_find_rank_1based(j, prefix_sums, nranks)
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
    @warn "nccl_pack_for_transpose! is a testing stub (simple copy). For actual transposes, use transpose_z_to_y! etc." maxlog=1
    Nx, Ny, Nz = size(data)
    total = Nx * Ny * Nz

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
    @warn "nccl_unpack_from_transpose! is a testing stub (simple copy). For actual transposes, use transpose_z_to_y! etc." maxlog=1
    Nx, Ny, Nz = size(data)
    total = Nx * Ny * Nz

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
grouped point-to-point `NCCL.Send`/`NCCL.Recv!` operations wrapped in
`NCCL.groupStart()`/`NCCL.groupEnd()`.

Complex element types are handled by reinterpreting the buffers to the
underlying real type at the wire (NCCL has no complex `ncclDataType_t`), with
all counts and displacements doubled accordingly.

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
                         nccl_comm; my_rank::Int)
    # NCCL has no complex ncclDataType_t: reinterpret complex buffers to the
    # underlying real type at the wire and DOUBLE (never halve) every count and
    # displacement. `reinterpret` on a CuArray returns a memory-sharing CuArray,
    # so the recursion below runs the identical real-typed path.
    if eltype(send_buf) <: Complex
        RT = real(eltype(send_buf))
        return nccl_alltoall!(reinterpret(RT, send_buf), reinterpret(RT, recv_buf),
                              2 .* send_counts, 2 .* recv_counts,
                              2 .* send_displs, 2 .* recv_displs,
                              nccl_comm; my_rank=my_rank)
    end

    # CRITICAL: Use error() instead of @assert for production safety
    total_send = sum(send_counts)
    total_recv = sum(recv_counts)
    if total_send > length(send_buf)
        error("nccl_alltoall!: Send counts ($total_send) exceed buffer size ($(length(send_buf)))")
    end
    if total_recv > length(recv_buf)
        error("nccl_alltoall!: Recv counts ($total_recv) exceed buffer size ($(length(recv_buf)))")
    end

    nranks = length(send_counts)

    if nccl_comm === nothing
        # The self-copy shortcut is only legal when the communicator genuinely
        # spans a single rank. With nranks > 1 a missing NCCL communicator means
        # initialization failed — self-copying would silently produce a wrong
        # transpose, so throw instead.
        if nranks != 1
            error("nccl_alltoall!: NCCL communicator is missing but the transpose spans " *
                  "$nranks ranks. Refusing the single-rank self-copy fallback, which would " *
                  "silently produce wrong results. Load NCCL.jl with `using NCCL` before " *
                  "multi-GPU transposes.")
        end
        total = sum(send_counts)
        if total > 0
            copyto!(view(recv_buf, 1:total), view(send_buf, 1:total))
        end
        return
    end

    # Resolve the user-loaded NCCL.jl module (Tarang/ext have no NCCL dependency)
    nccl = Tarang._require_nccl_module()

    # Task-local CUDA stream: pack kernels and the copyto! below are ordered on
    # this stream, so passing it to NCCL keeps pack -> NCCL -> unpack sound.
    comm_stream = CUDA.stream()

    # Handle self-communication with direct copyto! (NCCL P2P does not support self-send/recv)
    if my_rank >= 0 && send_counts[my_rank+1] > 0
        self_send_start = send_displs[my_rank+1] + 1
        self_send_end = self_send_start + send_counts[my_rank+1] - 1
        self_recv_start = recv_displs[my_rank+1] + 1
        self_recv_end = self_recv_start + recv_counts[my_rank+1] - 1
        copyto!(view(recv_buf, self_recv_start:self_recv_end),
                view(send_buf, self_send_start:self_send_end))
    end

    # NCCL grouped P2P for all remote peers
    # (real API: NCCL.groupStart()/groupEnd(), NCCL.Send(buf, comm; dest, stream),
    #  NCCL.Recv!(buf, comm; source, stream))
    nccl.groupStart()
    try
        for peer in 0:(nranks-1)
            # Skip self — already handled above
            if peer == my_rank
                continue
            end

            if send_counts[peer+1] > 0
                send_start = send_displs[peer+1] + 1
                send_end = send_start + send_counts[peer+1] - 1
                send_slice = view(send_buf, send_start:send_end)
                nccl.Send(send_slice, nccl_comm; dest=peer, stream=comm_stream)
            end

            if recv_counts[peer+1] > 0
                recv_start = recv_displs[peer+1] + 1
                recv_end = recv_start + recv_counts[peer+1] - 1
                recv_slice = view(recv_buf, recv_start:recv_end)
                nccl.Recv!(recv_slice, nccl_comm; source=peer, stream=comm_stream)
            end
        end
    finally
        nccl.groupEnd()
    end

    # Block until the grouped NCCL ops on `comm_stream` complete, so the recv
    # buffer is safe for the unpack kernels that follow.
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
function Tarang.transpose_z_to_y!(buffer::NCCLTransposeBuffer{T},
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

    # Multi-rank transpose requires a live NCCL row communicator; a missing one
    # would silently degrade into a wrong self-copy inside nccl_alltoall!.
    if !buffer.nccl_subcomms.initialized || buffer.nccl_subcomms.row_comm === nothing
        error("transpose_z_to_y!: row communicator spans $row_size ranks but the NCCL row " *
              "sub-communicator is not initialized. Load NCCL.jl with `using NCCL` before " *
              "multi-GPU transposes (see the init_nccl_subcomms! warning for the root cause).")
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

    prefix_sums_gpu = cumsum(chunk_sizes_gpu)
    kernel = pack_z_to_y_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx_local, Ny_local, Nz,
           chunk_sizes_gpu, displs_gpu, row_size, prefix_sums_gpu; ndrange=total)
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

    recv_prefix_sums_gpu = cumsum(recv_chunk_sizes_gpu)
    kernel_unpack = unpack_z_to_y_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_y, Ny_y, Nz_y,
                  recv_chunk_sizes_gpu, recv_displs_gpu, row_size, recv_prefix_sums_gpu; ndrange=prod(pencil.y_pencil_shape))
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
function Tarang.transpose_y_to_z!(buffer::NCCLTransposeBuffer{T},
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

    # Multi-rank transpose requires a live NCCL row communicator; a missing one
    # would silently degrade into a wrong self-copy inside nccl_alltoall!.
    if !buffer.nccl_subcomms.initialized || buffer.nccl_subcomms.row_comm === nothing
        error("transpose_y_to_z!: row communicator spans $row_size ranks but the NCCL row " *
              "sub-communicator is not initialized. Load NCCL.jl with `using NCCL` before " *
              "multi-GPU transposes (see the init_nccl_subcomms! warning for the root cause).")
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

    prefix_sums_gpu = cumsum(chunk_sizes_gpu)
    kernel = pack_y_to_z_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx_local, Ny, Nz_local,
           chunk_sizes_gpu, displs_gpu, row_size, prefix_sums_gpu; ndrange=total)
    CUDA.synchronize()

    row_rank = MPI.Comm_rank(pencil.row_comm)
    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:row_size], buffer.recv_counts[1:row_size],
        buffer.send_displs[1:row_size], buffer.recv_displs[1:row_size],
        buffer.nccl_subcomms.row_comm; my_rank=row_rank
    )

    output = CUDA.zeros(T, pencil.z_pencil_shape...)
    Nx_z, Ny_z, Nz_z = pencil.z_pencil_shape

    recv_chunk_sizes_gpu = CuArray(Int[div(Nz_global, row_size) + ((i-1) < mod(Nz_global, row_size) ? 1 : 0) for i in 1:row_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:row_size])
    recv_prefix_sums_gpu = cumsum(recv_chunk_sizes_gpu)

    kernel_unpack = unpack_y_to_z_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_z, Ny_z, Nz_z,
                  recv_chunk_sizes_gpu, recv_displs_gpu, row_size, recv_prefix_sums_gpu; ndrange=prod(pencil.z_pencil_shape))
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
function Tarang.transpose_y_to_x!(buffer::NCCLTransposeBuffer{T},
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

    # Multi-rank transpose requires a live NCCL col communicator; a missing one
    # would silently degrade into a wrong self-copy inside nccl_alltoall!.
    if !buffer.nccl_subcomms.initialized || buffer.nccl_subcomms.col_comm === nothing
        error("transpose_y_to_x!: col communicator spans $col_size ranks but the NCCL col " *
              "sub-communicator is not initialized. Load NCCL.jl with `using NCCL` before " *
              "multi-GPU transposes (see the init_nccl_subcomms! warning for the root cause).")
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

    prefix_sums_gpu = cumsum(chunk_sizes_gpu)
    kernel = pack_y_to_x_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx_local, Ny, Nz_local,
           chunk_sizes_gpu, displs_gpu, col_size, prefix_sums_gpu; ndrange=total)
    CUDA.synchronize()

    col_rank = MPI.Comm_rank(pencil.col_comm)
    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:col_size], buffer.recv_counts[1:col_size],
        buffer.send_displs[1:col_size], buffer.recv_displs[1:col_size],
        buffer.nccl_subcomms.col_comm; my_rank=col_rank
    )

    output = CUDA.zeros(T, pencil.x_pencil_shape...)
    Nx_x, Ny_x, Nz_x = pencil.x_pencil_shape

    recv_chunk_sizes_gpu = CuArray(Int[div(Nx_global, col_size) + ((i-1) < mod(Nx_global, col_size) ? 1 : 0) for i in 1:col_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:col_size])
    recv_prefix_sums_gpu = cumsum(recv_chunk_sizes_gpu)

    kernel_unpack = unpack_y_to_x_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_x, Ny_x, Nz_x,
                  recv_chunk_sizes_gpu, recv_displs_gpu, col_size, recv_prefix_sums_gpu; ndrange=prod(pencil.x_pencil_shape))
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
function Tarang.transpose_x_to_y!(buffer::NCCLTransposeBuffer{T},
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

    # Multi-rank transpose requires a live NCCL col communicator; a missing one
    # would silently degrade into a wrong self-copy inside nccl_alltoall!.
    if !buffer.nccl_subcomms.initialized || buffer.nccl_subcomms.col_comm === nothing
        error("transpose_x_to_y!: col communicator spans $col_size ranks but the NCCL col " *
              "sub-communicator is not initialized. Load NCCL.jl with `using NCCL` before " *
              "multi-GPU transposes (see the init_nccl_subcomms! warning for the root cause).")
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

    prefix_sums_gpu = cumsum(chunk_sizes_gpu)
    kernel = pack_x_to_y_kernel!(CUDABackend())
    kernel(buffer.send_buffer, data, Nx, Ny_local, Nz_local,
           chunk_sizes_gpu, displs_gpu, col_size, prefix_sums_gpu; ndrange=total)
    CUDA.synchronize()

    nccl_alltoall!(
        buffer.send_buffer, buffer.recv_buffer,
        buffer.send_counts[1:col_size], buffer.recv_counts[1:col_size],
        buffer.send_displs[1:col_size], buffer.recv_displs[1:col_size],
        buffer.nccl_subcomms.col_comm; my_rank=col_rank
    )

    output = CUDA.zeros(T, pencil.y_pencil_shape...)
    Nx_y, Ny_y, Nz_y = pencil.y_pencil_shape

    recv_chunk_sizes_gpu = CuArray(Int[div(Ny_global, col_size) + ((i-1) < mod(Ny_global, col_size) ? 1 : 0) for i in 1:col_size])
    recv_displs_gpu = CuArray(buffer.recv_displs[1:col_size])
    recv_prefix_sums_gpu = cumsum(recv_chunk_sizes_gpu)

    kernel_unpack = unpack_x_to_y_kernel!(CUDABackend())
    kernel_unpack(output, buffer.recv_buffer, Nx_y, Ny_y, Nz_y,
                  recv_chunk_sizes_gpu, recv_displs_gpu, col_size, recv_prefix_sums_gpu; ndrange=prod(pencil.y_pencil_shape))
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

The formulas mirror EXACTLY the inline count computations inside the production
`transpose_z_to_y!`/`transpose_y_to_z!`/`transpose_y_to_x!`/`transpose_x_to_y!`
(pairwise symmetric, uneven-decomposition-aware, per-rank chunk sizes — never
full pencil dims). Invariant: `sum(send_counts) == prod(<source pencil shape>)`.

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
        # Z->Y (row comm): split our fully-local Z among peers, gather Y.
        # Source layout is the Z-pencil (Nx_local, Ny_local, Nz_global).
        Ny_g = pencil.global_shape[2]
        Nz_g = pencil.global_shape[3]
        Nx_local = pencil.z_pencil_shape[1]
        Ny_local = pencil.z_pencil_shape[2]
        row_rank = MPI.Comm_rank(pencil.row_comm)
        Nz_me = div(Nz_g, comm_size) + (row_rank < mod(Nz_g, comm_size) ? 1 : 0)

        for i in 1:comm_size
            Nz_i = div(Nz_g, comm_size) + ((i-1) < mod(Nz_g, comm_size) ? 1 : 0)
            Ny_i = div(Ny_g, comm_size) + ((i-1) < mod(Ny_g, comm_size) ? 1 : 0)
            # Send: our (Nx_local, Ny_local) face × rank i's Nz_i z-slices
            buffer.send_counts[i] = Nx_local * Ny_local * Nz_i
            # Recv: rank i's Ny_i y-chunk × our Nz_me z-slices
            buffer.recv_counts[i] = Nx_local * Ny_i * Nz_me
        end
    elseif direction == :y_to_z
        # Y->Z (row comm): split our fully-local Y among peers, gather Z.
        # Source layout is the Y-pencil (Nx_local, Ny_global, Nz_local).
        Ny_g = pencil.global_shape[2]
        Nz_g = pencil.global_shape[3]
        Nx_local = pencil.y_pencil_shape[1]
        Nz_local = pencil.y_pencil_shape[3]
        Ny_local_after = pencil.z_pencil_shape[2]  # our Y chunk after transpose

        for i in 1:comm_size
            Ny_i = div(Ny_g, comm_size) + ((i-1) < mod(Ny_g, comm_size) ? 1 : 0)
            Nz_i = div(Nz_g, comm_size) + ((i-1) < mod(Nz_g, comm_size) ? 1 : 0)
            # Send: rank i's Ny_i y-chunk × our Nz_local z-slices
            buffer.send_counts[i] = Nx_local * Ny_i * Nz_local
            # Recv: our Ny_local_after y-chunk × rank i's Nz_i z-slices
            buffer.recv_counts[i] = Nx_local * Ny_local_after * Nz_i
        end
    elseif direction == :y_to_x
        # Y->X (col comm): split our fully-local Y among peers, gather X.
        # Source layout is the Y-pencil (Nx_local, Ny_global, Nz_local).
        Nx_g = pencil.global_shape[1]
        Ny_g = pencil.global_shape[2]
        Nx_local = pencil.y_pencil_shape[1]
        Nz_local = pencil.y_pencil_shape[3]
        col_rank = MPI.Comm_rank(pencil.col_comm)
        Ny_me = div(Ny_g, comm_size) + (col_rank < mod(Ny_g, comm_size) ? 1 : 0)

        for i in 1:comm_size
            Ny_i = div(Ny_g, comm_size) + ((i-1) < mod(Ny_g, comm_size) ? 1 : 0)
            Nx_i = div(Nx_g, comm_size) + ((i-1) < mod(Nx_g, comm_size) ? 1 : 0)
            # Send: our (Nx_local, Nz_local) face × rank i's Ny_i y-slices
            buffer.send_counts[i] = Nx_local * Ny_i * Nz_local
            # Recv: rank i's Nx_i x-chunk × our Ny_me y-slices
            buffer.recv_counts[i] = Nx_i * Ny_me * Nz_local
        end
    elseif direction == :x_to_y
        # X->Y (col comm): split our fully-local X among peers, gather Y.
        # Source layout is the X-pencil (Nx_global, Ny_local, Nz_local).
        Nx_g = pencil.global_shape[1]
        Ny_g = pencil.global_shape[2]
        Ny_local = pencil.x_pencil_shape[2]
        Nz_local = pencil.x_pencil_shape[3]
        col_rank = MPI.Comm_rank(pencil.col_comm)
        Nx_me = div(Nx_g, comm_size) + (col_rank < mod(Nx_g, comm_size) ? 1 : 0)

        for i in 1:comm_size
            Nx_i = div(Nx_g, comm_size) + ((i-1) < mod(Nx_g, comm_size) ? 1 : 0)
            Ny_i = div(Ny_g, comm_size) + ((i-1) < mod(Ny_g, comm_size) ? 1 : 0)
            # Send: rank i's Nx_i x-slices × our (Ny_local, Nz_local)
            buffer.send_counts[i] = Nx_i * Ny_local * Nz_local
            # Recv: our Nx_me x-chunk × rank i's Ny_i y-slices
            buffer.recv_counts[i] = Nx_me * Ny_i * Nz_local
        end
    else
        error("compute_transpose_counts!: unsupported direction $direction " *
              "(expected :z_to_y, :y_to_z, :y_to_x, or :x_to_y)")
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
