"""
    Transpose Async - Asynchronous transpose operations for TransposableField

This file contains the asynchronous (non-blocking) transpose operations that
enable communication/computation overlap:
- async_transpose_z_to_y!
- async_transpose_y_to_x!
- wait_transpose!
- is_transpose_complete
"""

# ============================================================================
# Async Transpose Operations
# ============================================================================

"""
    async_transpose_z_to_y!(tf::TransposableField)

Start asynchronous transpose from ZLocal to YLocal.
Returns immediately after initiating communication.
Use `wait_transpose!(tf)` to complete the operation.
"""
function async_transpose_z_to_y!(tf::TransposableField{F,T,N}) where {F,T,N}
    # CRITICAL: Use error() instead of @assert for production safety
    # @assert can be disabled with --check-bounds=no, leaving no protection
    if tf.buffers.active_layout[] != ZLocal
        error("async_transpose_z_to_y!: Must be in ZLocal layout, currently in $(tf.buffers.active_layout[])")
    end
    if tf.async_state.in_progress
        error("async_transpose_z_to_y!: Another async operation is already in progress. " *
              "Call wait_transpose!() before starting a new async operation.")
    end

    topo = tf.topology

    # Error on 2D true mesh: async uses Alltoallv but 2D mesh needs Allgatherv
    if N == 2 && topo.Rx > 1 && topo.Ry > 1
        error("async_transpose_z_to_y! is not supported for 2D true mesh (Rx=$(topo.Rx), Ry=$(topo.Ry)). " *
              "Use blocking transpose_z_to_y! instead, which uses Allgatherv for correct data layout.")
    end

    # Use same communicator selection logic as transpose_z_to_y!
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
        copyto!(vec(tf.buffers.y_local_data), vec(tf.buffers.z_local_data))
        tf.buffers.active_layout[] = YLocal
        return tf
    end

    # CRITICAL: Validate comm_size matches counts array lengths
    # (mirroring the check in sync transpose_z_to_y!)
    if length(tf.counts.zy_send_counts) != comm_size
        error("async_transpose_z_to_y!: comm_size ($comm_size) does not match " *
              "zy_send_counts length ($(length(tf.counts.zy_send_counts))). " *
              "This indicates TransposeCounts were computed for different topology.")
    end

    arch = tf.buffers.architecture

    # Get buffers (use second set for async to avoid conflicts)
    send_buf = tf.buffers.send_buffer_2
    recv_buf = tf.buffers.recv_buffer_2

    # Determine pack dimension (same logic as transpose_z_to_y! blocking version)
    # 3D: pack dim=3 (Z redistributed)
    # 2D with Ry > 1 (y decomposed in ZLocal): pack dim=1 (x split)
    # 2D with Rx > 1 (x decomposed in ZLocal): pack dim=2 (y split)
    if N >= 3
        pack_dim = 3
    elseif topo.Ry > 1
        pack_dim = 1
    else
        # Rx > 1: ZLocal has partial x, full y. Split y.
        pack_dim = 2
    end

    # Pack data
    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.z_local_data,
                        tf.counts.zy_send_counts, tf.counts.zy_send_displs,
                        pack_dim, comm_size, arch)
    tf.async_state.pack_time = time() - pack_start

    # Start non-blocking alltoallv
    request = _do_ialltoallv!(send_buf, recv_buf,
                              tf.counts.zy_send_counts, tf.counts.zy_recv_counts,
                              comm, arch, tf.buffers)

    tf.async_state.request = request
    tf.async_state.in_progress = true
    tf.async_state.from_layout = ZLocal
    tf.async_state.to_layout = YLocal
    # CRITICAL: Store recv_size for correct staging buffer copy in wait_transpose!
    tf.async_state.recv_size = sum(tf.counts.zy_recv_counts)

    return tf
end

"""
    async_transpose_y_to_x!(tf::TransposableField)

Start asynchronous transpose from YLocal to XLocal.
"""
function async_transpose_y_to_x!(tf::TransposableField{F,T,N}) where {F,T,N}
    # CRITICAL: Use error() instead of @assert for production safety
    if tf.buffers.active_layout[] != YLocal
        error("async_transpose_y_to_x!: Must be in YLocal layout, currently in $(tf.buffers.active_layout[])")
    end
    if tf.async_state.in_progress
        error("async_transpose_y_to_x!: Another async operation is already in progress. " *
              "Call wait_transpose!() before starting a new async operation.")
    end

    topo = tf.topology

    if topo.col_size == 1
        copyto!(vec(tf.buffers.x_local_data), vec(tf.buffers.y_local_data))
        tf.buffers.active_layout[] = XLocal
        return tf
    end

    # CRITICAL: Check that yx_counts were computed (same guard as blocking version)
    # For 2D with 1D decomposition (Rx>1, Ry==1 or vice versa), yx counts may not be computed
    # because Y↔X transposes are not needed in that configuration (ZLocal==XLocal)
    if sum(tf.counts.yx_send_counts) == 0 && sum(tf.counts.yx_recv_counts) == 0
        if N == 2 && (topo.Rx == 1 || topo.Ry == 1)
            error("Async Y→X transpose not supported for 2D domain with 1D decomposition (Rx=$(topo.Rx), Ry=$(topo.Ry)). " *
                  "In this configuration, ZLocal and XLocal are equivalent - use transpose_z_to_y! and " *
                  "transpose_y_to_z! instead.")
        else
            error("Y→X transpose counts not computed. This may indicate incorrect TransposableField setup.")
        end
    end

    # CRITICAL: Validate col_size matches counts array lengths
    # (mirroring the check in sync transpose_y_to_x!)
    if length(tf.counts.yx_send_counts) != topo.col_size
        error("async_transpose_y_to_x!: col_size ($(topo.col_size)) does not match " *
              "yx_send_counts length ($(length(tf.counts.yx_send_counts))). " *
              "This indicates TransposeCounts were computed for different topology.")
    end

    arch = tf.buffers.architecture

    send_buf = tf.buffers.send_buffer_2
    recv_buf = tf.buffers.recv_buffer_2

    # Y→X: pack by Y dimension (dim=2)
    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.y_local_data,
                        tf.counts.yx_send_counts, tf.counts.yx_send_displs,
                        2, topo.col_size, arch)
    tf.async_state.pack_time = time() - pack_start

    request = _do_ialltoallv!(send_buf, recv_buf,
                              tf.counts.yx_send_counts, tf.counts.yx_recv_counts,
                              topo.col_comm, arch, tf.buffers)

    tf.async_state.request = request
    tf.async_state.in_progress = true
    tf.async_state.from_layout = YLocal
    tf.async_state.to_layout = XLocal
    # CRITICAL: Store recv_size for correct staging buffer copy in wait_transpose!
    tf.async_state.recv_size = sum(tf.counts.yx_recv_counts)

    return tf
end

"""
    wait_transpose!(tf::TransposableField)

Wait for asynchronous transpose to complete and finalize the operation.
"""
function wait_transpose!(tf::TransposableField{F,T,N}) where {F,T,N}
    if !tf.async_state.in_progress
        return tf
    end

    arch = tf.buffers.architecture
    topo = tf.topology

    # Wait for MPI communication to complete
    wait_start = time()
    used_async = tf.async_state.request !== nothing
    if used_async
        MPI.Wait(tf.async_state.request)
    end
    tf.async_state.wait_time = time() - wait_start

    # Unlock staging buffers now that the async MPI operation is complete
    tf.buffers.staging_locked[] = false

    # Get the receive buffer - for non-CUDA-aware MPI on GPU, data is in staging buffer
    recv_buf = tf.buffers.recv_buffer_2

    # If we used staging buffers (non-CUDA-aware MPI with GPU), copy back to GPU
    # CRITICAL: Only do this when async was actually used (used_async=true).
    # When blocking fallback was used (request=nothing), _do_alltoallv! already
    # copied the result directly to recv_buf, so copying from stale staging would corrupt data.
    # CRITICAL: Only copy recv_size elements, not the entire staging buffer!
    # The staging buffer may be larger than the actual received data, and copying
    # uninitialized memory would corrupt the result.
    if used_async && is_gpu(arch) && !check_cuda_aware_mpi() && tf.buffers.recv_staging !== nothing
        recv_size = tf.async_state.recv_size

        # Validate recv_size before using it
        if recv_size < 0
            error("wait_transpose!: Invalid recv_size=$recv_size (negative). " *
                  "This indicates async_state was not properly initialized.")
        end
        if recv_size > length(tf.buffers.recv_staging)
            error("wait_transpose!: recv_size=$recv_size exceeds staging buffer size " *
                  "$(length(tf.buffers.recv_staging)). This indicates a buffer allocation or " *
                  "recv_size tracking bug.")
        end
        if recv_size > length(recv_buf)
            error("wait_transpose!: recv_size=$recv_size exceeds recv_buffer size " *
                  "$(length(recv_buf)). This indicates mismatched buffer sizes.")
        end

        if recv_size > 0
            staging_view = view(tf.buffers.recv_staging, 1:recv_size)
            recv_view = view(recv_buf, 1:recv_size)
            copyto!(recv_view, on_architecture(arch, staging_view))
        end
    end

    # Unpack based on destination layout
    unpack_start = time()
    if tf.async_state.to_layout == YLocal
        # Determine correct comm_size for 2D vs 3D (same logic as transpose_z_to_y!)
        if N >= 3
            comm_size = topo.row_size
        else
            comm_size = topo.row_size > 1 ? topo.row_size : topo.col_size
        end
        # Determine unpack_dim (same logic as blocking version)
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
        unpack_from_transpose!(tf.buffers.y_local_data, recv_buf,
                              tf.counts.zy_recv_counts, tf.counts.zy_recv_displs,
                              unpack_dim, comm_size, arch)
    elseif tf.async_state.to_layout == XLocal
        unpack_from_transpose!(tf.buffers.x_local_data, recv_buf,
                              tf.counts.yx_recv_counts, tf.counts.yx_recv_displs,
                              1, topo.col_size, arch)
    end
    tf.async_state.unpack_time = time() - unpack_start

    # Update state
    tf.buffers.active_layout[] = tf.async_state.to_layout
    tf.async_state.in_progress = false
    tf.async_state.request = nothing

    # Update statistics
    tf.total_pack_time += tf.async_state.pack_time
    tf.total_unpack_time += tf.async_state.unpack_time
    tf.num_transposes += 1

    return tf
end

"""
    is_transpose_complete(tf::TransposableField)

Check if an async transpose has completed without blocking.

If the MPI communication has completed, automatically calls `wait_transpose!`
to unpack the data and update the layout, ensuring the field is in a
consistent state when this function returns `true`.
"""
function is_transpose_complete(tf::TransposableField)
    if !tf.async_state.in_progress
        return true
    end

    if tf.async_state.request === nothing
        # Communication done (e.g., blocking fallback was used), finalize
        wait_transpose!(tf)
        return true
    end

    flag, _ = MPI.Test(tf.async_state.request)
    if flag
        # MPI communication complete — finalize: unpack data, update layout.
        # MPI.Test has already internally freed the MPI handle (set to MPI_REQUEST_NULL).
        # We leave tf.async_state.request as-is (not nothing) so that wait_transpose!
        # correctly identifies this as an async operation and performs the GPU staging
        # buffer copy for non-CUDA-aware MPI. MPI.Wait on MPI_REQUEST_NULL is a no-op.
        wait_transpose!(tf)
    end
    return flag
end

