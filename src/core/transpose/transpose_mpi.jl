"""
    Transpose MPI - MPI communication helpers for TransposableField

This file contains MPI communication functions used by transpose operations:
- _do_alltoallv!
- _do_ialltoallv!
- _do_allgatherv!
- _alltoallv_cpu!
- _allgatherv_cpu!
"""

# ============================================================================
# GPU Synchronization Helper
# ============================================================================

"""
    _sync_gpu_if_needed(arch)

Synchronize GPU stream before MPI operations to ensure all GPU kernels have completed.
CRITICAL: Must be called before:
1. CUDA-aware MPI operations (to ensure send buffer data is ready)
2. GPU-to-CPU copies (to ensure GPU data is complete before staging)

This prevents race conditions where MPI reads from GPU buffers before
prior GPU operations (FFTs, pack kernels) have completed.
"""
function _sync_gpu_if_needed(arch::AbstractArchitecture)
    # GPU case: synchronize() is defined on the architecture abstraction layer.
    # For GPU architectures, this calls the appropriate device synchronize
    # (overridden by CUDA extension via Tarang.synchronize dispatch).
    if is_gpu(arch)
        synchronize(arch)
    end
end

# CPU case - no sync needed
_sync_gpu_if_needed(::CPU) = nothing

# ============================================================================
# MPI Communication Helpers
# ============================================================================

"""
    _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)

Perform blocking MPI.Alltoallv with appropriate handling for CPU and GPU arrays.
"""
function _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                        comm::MPI.Comm, arch::CPU, buffers=nothing)
    # Validate counts are non-negative and match communicator size
    comm_size = MPI.Comm_size(comm)
    if length(send_counts) != comm_size
        throw(ArgumentError(
            "Alltoallv send_counts length ($(length(send_counts))) does not match " *
            "communicator size ($comm_size)."
        ))
    end
    if length(recv_counts) != comm_size
        throw(ArgumentError(
            "Alltoallv recv_counts length ($(length(recv_counts))) does not match " *
            "communicator size ($comm_size)."
        ))
    end
    for i in 1:comm_size
        if send_counts[i] < 0
            throw(ArgumentError("Alltoallv send_counts[$i]=$(send_counts[i]) is negative."))
        end
        if recv_counts[i] < 0
            throw(ArgumentError("Alltoallv recv_counts[$i]=$(recv_counts[i]) is negative."))
        end
    end

    # Validate buffer sizes match counts to prevent overflow/underflow
    total_send = sum(send_counts)
    total_recv = sum(recv_counts)
    if total_send > length(send_buf)
        throw(ArgumentError(
            "Alltoallv send buffer too small: sum(send_counts)=$total_send > length(send_buf)=$(length(send_buf)). " *
            "This would cause buffer overread."
        ))
    end
    if total_recv > length(recv_buf)
        throw(ArgumentError(
            "Alltoallv recv buffer too small: sum(recv_counts)=$total_recv > length(recv_buf)=$(length(recv_buf)). " *
            "This would cause buffer overflow."
        ))
    end

    sendbuf = MPI.VBuffer(send_buf, send_counts)
    recvbuf = MPI.VBuffer(recv_buf, recv_counts)
    MPI.Alltoallv!(sendbuf, recvbuf, comm)
    return recv_buf
end

function _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                        comm::MPI.Comm, arch::AbstractArchitecture, buffers=nothing)
    # Check for CUDA-aware MPI
    if is_gpu(arch) && check_cuda_aware_mpi()
        # CRITICAL: Sync GPU before MPI to ensure pack kernels completed
        _sync_gpu_if_needed(arch)
        # Direct GPU buffer transfer
        sendbuf = MPI.VBuffer(send_buf, send_counts)
        recvbuf = MPI.VBuffer(recv_buf, recv_counts)
        MPI.Alltoallv!(sendbuf, recvbuf, comm)
    else
        # Stage through CPU using pre-allocated staging buffers
        # CRITICAL: Check if pre-allocated staging buffers are large enough
        # Batched transposes may need larger buffers than single-field operations
        send_size = length(send_buf)
        recv_size = length(recv_buf)
        use_preallocated = (buffers !== nothing &&
                           buffers.send_staging !== nothing &&
                           !buffers.staging_locked[] &&
                           length(buffers.send_staging) >= send_size &&
                           length(buffers.recv_staging) >= recv_size)

        if use_preallocated
            # Use pre-allocated staging buffers (sized appropriately)
            send_cpu = view(buffers.send_staging, 1:send_size)
            recv_cpu = view(buffers.recv_staging, 1:recv_size)
            # CRITICAL: Sync GPU before copy to ensure pack kernels completed
            _sync_gpu_if_needed(arch)
            copyto!(send_cpu, on_architecture(CPU(), send_buf))
        else
            # Staging buffers too small or not available - allocate fresh
            # WARNING: This can cause memory allocation in hot path and potential leaks
            # in batched/grouped operations. Consider increasing staging buffer sizes.
            if buffers !== nothing && buffers.send_staging !== nothing
                staging_size = length(buffers.send_staging)
                @warn "Alltoallv staging buffer too small: have $staging_size, need send=$send_size recv=$recv_size. " *
                      "Allocating fresh buffers. Consider increasing staging buffer size for grouped transposes." maxlog=1
            end
            # CRITICAL: Sync GPU before copy to ensure pack kernels completed
            _sync_gpu_if_needed(arch)
            send_cpu = on_architecture(CPU(), send_buf)
            recv_cpu = similar(send_cpu, eltype(send_cpu), recv_size)
        end

        sendbuf = MPI.VBuffer(send_cpu, send_counts)
        recvbuf = MPI.VBuffer(recv_cpu, recv_counts)
        MPI.Alltoallv!(sendbuf, recvbuf, comm)
        copyto!(recv_buf, on_architecture(arch, recv_cpu))
    end
    return recv_buf
end

"""
    _do_ialltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)

Perform non-blocking MPI.Ialltoallv (async) if available.
Returns MPI.Request for later waiting, or nothing if using blocking fallback.
"""
function _do_ialltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                         comm::MPI.Comm, arch::CPU, buffers=nothing)
    # Check if non-blocking Ialltoallv! is available (not in all MPI.jl versions)
    if isdefined(MPI, :Ialltoallv!)
        sendbuf = MPI.VBuffer(send_buf, send_counts)
        recvbuf = MPI.VBuffer(recv_buf, recv_counts)
        request = MPI.Ialltoallv!(sendbuf, recvbuf, comm)
        return request
    else
        # Fallback to blocking version
        _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)
        return nothing
    end
end

function _do_ialltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                         comm::MPI.Comm, arch::AbstractArchitecture, buffers=nothing)
    # Check if non-blocking Ialltoallv! is available (not in all MPI.jl versions)
    if !isdefined(MPI, :Ialltoallv!)
        # Fallback to blocking version
        _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)
        return nothing
    end

    if is_gpu(arch) && check_cuda_aware_mpi()
        # CRITICAL: Sync GPU before MPI to ensure pack kernels completed
        _sync_gpu_if_needed(arch)
        # Direct GPU buffer transfer (non-blocking)
        sendbuf = MPI.VBuffer(send_buf, send_counts)
        recvbuf = MPI.VBuffer(recv_buf, recv_counts)
        request = MPI.Ialltoallv!(sendbuf, recvbuf, comm)
        return request
    else
        # For non-CUDA-aware MPI, we need to use staging buffers
        # Copy to CPU first, then do async MPI
        # CRITICAL: Check if pre-allocated staging buffers are large enough
        # Batched transposes may need larger buffers than single-field operations
        send_size = length(send_buf)
        recv_size = length(recv_buf)
        use_preallocated = (buffers !== nothing &&
                           buffers.send_staging !== nothing &&
                           length(buffers.send_staging) >= send_size &&
                           length(buffers.recv_staging) >= recv_size)

        if use_preallocated && !buffers.staging_locked[]
            # Lock staging buffers to prevent sync operations from reusing them
            # while this async operation is in flight
            buffers.staging_locked[] = true

            send_cpu = view(buffers.send_staging, 1:send_size)
            recv_cpu = view(buffers.recv_staging, 1:recv_size)
            # CRITICAL: Sync GPU before copy to ensure pack kernels completed
            _sync_gpu_if_needed(arch)
            copyto!(send_cpu, on_architecture(CPU(), send_buf))

            sendbuf = MPI.VBuffer(send_cpu, send_counts)
            recvbuf = MPI.VBuffer(recv_cpu, recv_counts)
            request = MPI.Ialltoallv!(sendbuf, recvbuf, comm)

            # Note: We'll need to copy recv_cpu back to GPU in wait_transpose!
            return request
        else
            # Staging buffers too small or not available - fall back to blocking version
            # (async with CPU allocation would be complex and potentially leak memory)
            if buffers !== nothing && buffers.send_staging !== nothing
                staging_size = length(buffers.send_staging)
                @warn "Async transpose falling back to blocking: staging buffers too small " *
                      "(have $staging_size, need send=$send_size, recv=$recv_size). " *
                      "Consider increasing buffer sizes for grouped transposes." maxlog=1
            end
            _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)
            return nothing  # Signals to wait_transpose! that communication already completed
        end
    end
end

# ============================================================================
# Allgatherv Operations (for 2D mesh on 2D domain)
# ============================================================================

"""
    _do_allgatherv!(send_buf, recv_buf, recv_counts, comm, arch, buffers=nothing)

Perform MPI.Allgatherv for gathering operations.
Used for Z→Y transpose on 2D domains with 2D mesh where y needs to be gathered.
"""
function _do_allgatherv!(send_buf, recv_buf, send_count::Int, recv_counts::Vector{Int},
                         comm::MPI.Comm, arch::CPU, buffers=nothing)
    # Validate send_count is non-negative
    if send_count < 0
        throw(ArgumentError(
            "Allgatherv send_count ($send_count) is negative. " *
            "This indicates corrupted count computation."
        ))
    end

    # Validate recv_counts array length matches communicator size
    comm_size = MPI.Comm_size(comm)
    if length(recv_counts) != comm_size
        throw(ArgumentError(
            "Allgatherv recv_counts length ($(length(recv_counts))) does not match " *
            "communicator size ($comm_size). This indicates incorrect count array setup."
        ))
    end

    # Validate no negative values in recv_counts
    for i in 1:length(recv_counts)
        if recv_counts[i] < 0
            throw(ArgumentError(
                "Allgatherv recv_counts[$i]=$(recv_counts[i]) is negative. " *
                "This indicates corrupted count computation."
            ))
        end
    end

    # Validate this rank's contribution is consistent with recv_counts
    my_rank = MPI.Comm_rank(comm)
    if recv_counts[my_rank + 1] != send_count
        throw(ArgumentError(
            "Allgatherv consistency error: this rank's send_count ($send_count) does not match " *
            "recv_counts[$(my_rank + 1)]=$(recv_counts[my_rank + 1]). " *
            "All ranks must agree on each rank's contribution size."
        ))
    end

    # Compute recv displacements
    recv_displs = zeros(Int, length(recv_counts))
    offset = 0
    for i in 1:length(recv_counts)
        recv_displs[i] = offset
        offset += recv_counts[i]
    end

    # Validate buffer size matches recv_counts total
    total_recv = sum(recv_counts)
    if length(recv_buf) < total_recv
        throw(ArgumentError(
            "Allgatherv recv_buf too small: buffer has $(length(recv_buf)) elements, " *
            "but sum(recv_counts)=$total_recv. This would cause buffer overflow."
        ))
    end

    MPI.Allgatherv!(view(send_buf, 1:send_count), MPI.VBuffer(recv_buf, recv_counts, recv_displs), comm)
    return recv_buf
end

function _do_allgatherv!(send_buf, recv_buf, send_count::Int, recv_counts::Vector{Int},
                         comm::MPI.Comm, arch::AbstractArchitecture, buffers=nothing)
    # Validate send_count is non-negative
    if send_count < 0
        throw(ArgumentError(
            "Allgatherv send_count ($send_count) is negative. " *
            "This indicates corrupted count computation."
        ))
    end

    # Validate recv_counts array length matches communicator size
    comm_size = MPI.Comm_size(comm)
    if length(recv_counts) != comm_size
        throw(ArgumentError(
            "Allgatherv recv_counts length ($(length(recv_counts))) does not match " *
            "communicator size ($comm_size). This indicates incorrect count array setup."
        ))
    end

    # Validate no negative values in recv_counts
    for i in 1:length(recv_counts)
        if recv_counts[i] < 0
            throw(ArgumentError(
                "Allgatherv recv_counts[$i]=$(recv_counts[i]) is negative. " *
                "This indicates corrupted count computation."
            ))
        end
    end

    # Validate this rank's contribution is consistent with recv_counts
    my_rank = MPI.Comm_rank(comm)
    if recv_counts[my_rank + 1] != send_count
        throw(ArgumentError(
            "Allgatherv consistency error: this rank's send_count ($send_count) does not match " *
            "recv_counts[$(my_rank + 1)]=$(recv_counts[my_rank + 1]). " *
            "All ranks must agree on each rank's contribution size."
        ))
    end

    # Compute recv displacements
    recv_displs = zeros(Int, length(recv_counts))
    offset = 0
    for i in 1:length(recv_counts)
        recv_displs[i] = offset
        offset += recv_counts[i]
    end

    # Validate buffer size matches recv_counts total
    total_recv = sum(recv_counts)
    if length(recv_buf) < total_recv
        throw(ArgumentError(
            "Allgatherv recv_buf too small: buffer has $(length(recv_buf)) elements, " *
            "but sum(recv_counts)=$total_recv. This would cause buffer overflow."
        ))
    end

    if is_gpu(arch) && check_cuda_aware_mpi()
        # CRITICAL: Sync GPU before MPI to ensure data is ready
        _sync_gpu_if_needed(arch)
        # Direct GPU buffer transfer
        MPI.Allgatherv!(view(send_buf, 1:send_count), MPI.VBuffer(recv_buf, recv_counts, recv_displs), comm)
    else
        # Stage through CPU
        # CRITICAL: Sync GPU before copy to ensure data is ready
        _sync_gpu_if_needed(arch)
        send_cpu = on_architecture(CPU(), view(send_buf, 1:send_count))
        total_recv = sum(recv_counts)
        recv_cpu = similar(send_cpu, total_recv)

        MPI.Allgatherv!(send_cpu, MPI.VBuffer(recv_cpu, recv_counts, recv_displs), comm)

        copyto!(view(recv_buf, 1:total_recv), on_architecture(arch, recv_cpu))
    end
    return recv_buf
end

"""
    _unpack_gathered_y!(dest, buffer, z_shape, y_shape, nranks, arch)

Unpack data gathered via Allgatherv into YLocal layout.
For 2D domains with 2D mesh: rearranges data so y dimension is contiguous.

Each rank in row_comm contributed (Nx/Rx, Ny_r) where Ny_r = divide_evenly(Ny, Ry, r).
Result should be (Nx/Rx, Ny) with y contiguous.

Note: Handles uneven decomposition when Ny % Ry ≠ 0.

CRITICAL: nranks must match the actual communicator size used in Allgatherv.
"""
function _unpack_gathered_y!(dest::AbstractArray{T,2}, buffer::AbstractVector{T},
                             z_shape::NTuple{2,Int}, y_shape::NTuple{2,Int},
                             nranks::Int, arch::CPU) where T
    # Validate nranks is positive
    if nranks < 1
        error("_unpack_gathered_y! requires nranks >= 1, got nranks=$nranks")
    end

    local_nx = z_shape[1]  # Nx/Rx (same for Z and Y)
    total_ny = y_shape[2]  # Ny

    # CRITICAL: Validate that total contribution from all ranks equals expected buffer size
    # This catches mismatches between nranks and actual comm_size
    expected_total = local_nx * total_ny
    actual_total = 0
    for rank in 0:(nranks-1)
        y_range = local_range(total_ny, nranks, rank)
        actual_total += local_nx * length(y_range)
    end
    if actual_total != expected_total
        error("_unpack_gathered_y! nranks mismatch: with nranks=$nranks, total contribution " *
              "is $actual_total elements, but expected $expected_total (y_shape=$y_shape). " *
              "Ensure nranks matches the actual communicator size (e.g., row_size or Ry).")
    end

    # Validate buffer has enough data
    if length(buffer) < expected_total
        error("_unpack_gathered_y! buffer too small: have $(length(buffer)) elements, " *
              "need $expected_total for y_shape=$y_shape")
    end

    # Buffer layout from Allgatherv: contiguous blocks from each rank
    # Each block is (local_nx × Ny_r) where Ny_r varies per rank
    # Dest layout: (local_nx × total_ny) in column-major order

    buf_offset = 0
    for rank in 0:(nranks-1)
        # Get the y-range for this rank using proper uneven decomposition
        y_range = local_range(total_ny, nranks, rank)
        rank_ny = length(y_range)
        y_start = first(y_range)

        for iy_local in 1:rank_ny
            iy_global = y_start + iy_local - 1
            for ix in 1:local_nx
                buf_idx = buf_offset + (iy_local - 1) * local_nx + ix
                @inbounds dest[ix, iy_global] = buffer[buf_idx]
            end
        end

        # Advance buffer offset by this rank's contribution
        buf_offset += local_nx * rank_ny
    end

    return dest
end

function _unpack_gathered_y!(dest::AbstractArray{T,2}, buffer::AbstractVector{T},
                             z_shape::NTuple{2,Int}, y_shape::NTuple{2,Int},
                             nranks::Int, arch::AbstractArchitecture) where T
    # For GPU: transfer to CPU, unpack, transfer back
    dest_cpu = on_architecture(CPU(), dest)
    buffer_cpu = on_architecture(CPU(), buffer)
    _unpack_gathered_y!(dest_cpu, buffer_cpu, z_shape, y_shape, nranks, CPU())
    copyto!(dest, on_architecture(arch, dest_cpu))
    return dest
end

"""
    _pack_scatter_y!(buffer, src, z_shape, y_shape, nranks, my_rank, arch)

Pack YLocal data for scattering y back to ZLocal (reverse of Allgatherv).
Each rank extracts its own y-portion from the full y data.

Note: Handles uneven decomposition when Ny % Ry ≠ 0.
"""
function _pack_scatter_y!(buffer::AbstractVector{T}, src::AbstractArray{T,2},
                          z_shape::NTuple{2,Int}, y_shape::NTuple{2,Int},
                          nranks::Int, my_rank::Int, arch::CPU) where T
    local_nx = z_shape[1]  # Nx/Rx
    total_ny = y_shape[2]  # Ny

    # Extract only our y-portion using proper uneven decomposition
    y_range = local_range(total_ny, nranks, my_rank)
    y_start = first(y_range)
    rank_ny = length(y_range)

    idx = 1
    for iy_local in 1:rank_ny
        iy_global = y_start + iy_local - 1
        for ix in 1:local_nx
            @inbounds buffer[idx] = src[ix, iy_global]
            idx += 1
        end
    end

    return buffer
end

function _pack_scatter_y!(buffer::AbstractVector{T}, src::AbstractArray{T,2},
                          z_shape::NTuple{2,Int}, y_shape::NTuple{2,Int},
                          nranks::Int, my_rank::Int, arch::AbstractArchitecture) where T
    # For GPU: transfer to CPU, pack, transfer back
    src_cpu = on_architecture(CPU(), src)
    buffer_cpu = on_architecture(CPU(), buffer)
    _pack_scatter_y!(buffer_cpu, src_cpu, z_shape, y_shape, nranks, my_rank, CPU())
    copyto!(buffer, on_architecture(arch, buffer_cpu))
    return buffer
end

