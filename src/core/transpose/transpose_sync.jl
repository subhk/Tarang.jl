"""
    Transpose Sync - Synchronous transpose operations for TransposableField

This file contains the synchronous (blocking) transpose operations:
- transpose_z_to_y!
- transpose_y_to_z!
- transpose_y_to_x!
- transpose_x_to_y!
"""

# ============================================================================
# Synchronous Transpose Operations
# ============================================================================

"""
    transpose_z_to_y!(tf::TransposableField)

Transpose from ZLocal layout to YLocal layout (synchronous).
Uses MPI.Alltoallv for communication.
"""
function transpose_z_to_y!(tf::TransposableField{F,T,N}) where {F,T,N}
    # CRITICAL: Use error() instead of @assert for production safety
    if tf.buffers.active_layout[] != ZLocal
        error("transpose_z_to_y!: Must be in ZLocal layout, currently in $(tf.buffers.active_layout[])")
    end
    # Guard against calling synchronous transpose while async is in flight —
    # they share buffer_2 and would corrupt data.
    if tf.async_state.in_progress
        error("transpose_z_to_y!: Cannot call synchronous transpose while async operation is in progress. " *
              "Call wait_transpose!() first.")
    end

    topo = tf.topology
    Rx, Ry = topo.Rx, topo.Ry

    # Check for true 2D mesh on 2D domain (special case using Allgatherv)
    use_allgather = (N == 2) && (Rx > 1) && (Ry > 1)

    # Determine communicator
    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    elseif use_allgather
        # True 2D mesh on 2D domain: use row_comm for gathering y
        comm = topo.row_comm
        comm_size = topo.row_size
    else
        # 2D case with 1D decomposition: use whichever communicator has multiple processes
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
        # No transpose needed - just copy data
        # CRITICAL: Validate buffers are allocated before use
        if tf.buffers.z_local_data === nothing || tf.buffers.y_local_data === nothing
            error("TransposableField buffers not allocated. " *
                  "Call allocate_transpose_buffers!(tf) before transpose operations.")
        end
        copyto!(vec(tf.buffers.y_local_data), vec(tf.buffers.z_local_data))
        tf.buffers.active_layout[] = YLocal
        return tf
    end

    arch = tf.buffers.architecture
    start_time = time()

    # Validate comm_size matches counts array lengths
    if length(tf.counts.zy_send_counts) != comm_size
        error("transpose_z_to_y!: comm_size ($comm_size) does not match " *
              "zy_send_counts length ($(length(tf.counts.zy_send_counts))). " *
              "This indicates TransposeCounts were computed for different topology.")
    end

    # CRITICAL: Validate communicator is not null (can happen after improper cleanup)
    if comm === nothing || comm == MPI.COMM_NULL
        error("transpose_z_to_y!: communicator is null but comm_size=$comm_size > 1. " *
              "This indicates topology was not properly initialized or was already freed.")
    end

    # Get active buffers
    send_buf, recv_buf = get_active_buffers(tf)

    if use_allgather
        # True 2D mesh on 2D domain: use Allgatherv to gather y
        # ZLocal: (Nx/Rx, Ny/Ry) → YLocal: (Nx/Rx, Ny)
        z_shape = tf.local_shapes[ZLocal]
        y_shape = tf.local_shapes[YLocal]
        send_count = prod(z_shape)
        local_nx = z_shape[1]  # Nx/Rx
        Ny = y_shape[2]  # Full y dimension

        # CRITICAL: Validate that sum(recv_counts) matches expected YLocal size
        # This catches decomposition mismatches that could cause partial overwrites
        expected_y_total = prod(y_shape)
        actual_recv_total = sum(tf.counts.zy_recv_counts)
        if actual_recv_total != expected_y_total
            error("Allgatherv Z→Y count mismatch: sum(recv_counts)=$actual_recv_total " *
                  "but expected YLocal size=$expected_y_total (shape=$y_shape). " *
                  "This indicates a decomposition or count computation error.")
        end

        # CRITICAL: Validate individual recv_counts match divide_evenly distribution
        # Sum check alone can pass with wrong individual values (e.g., [5,3,2] vs [4,3,3])
        for p in 0:(comm_size-1)
            Ny_p = divide_evenly(Ny, comm_size, p)
            expected_count = local_nx * Ny_p
            actual_count = tf.counts.zy_recv_counts[p+1]
            if actual_count != expected_count
                error("Allgatherv Z→Y recv_counts[$p] mismatch: expected $expected_count " *
                      "(local_nx=$local_nx × Ny_p=$Ny_p), got $actual_count. " *
                      "This indicates wrong divide_evenly distribution.")
            end
        end

        # Pack Z data contiguously into send buffer
        pack_start = time()
        copyto!(view(send_buf, 1:send_count), vec(tf.buffers.z_local_data))
        tf.total_pack_time += time() - pack_start

        # Perform Allgatherv
        _do_allgatherv!(send_buf, recv_buf, send_count, tf.counts.zy_recv_counts,
                        comm, arch, tf.buffers)

        # Unpack received data into y_local_data
        # Data comes from Ry processes, each contributing (Nx/Rx, Ny/Ry)
        # Need to reorder so y is contiguous
        unpack_start = time()
        _unpack_gathered_y!(tf.buffers.y_local_data, recv_buf, tf.local_shapes[ZLocal],
                            tf.local_shapes[YLocal], comm_size, arch)
        tf.total_unpack_time += time() - unpack_start
    else
        # Standard Alltoallv approach
        # Determine pack/unpack dimensions based on ndims and decomposition
        # 3D: pack dim=3 (Z redistributed), unpack dim=2 (receiving Y chunks)
        # 2D with Ry > 1 (y decomposed in ZLocal): split x among processes, receive y
        #   ZLocal=(Nx, Ny/Ry) → YLocal=(Nx/Ry, Ny): pack dim=1 (x split), unpack dim=2 (y received)
        # 2D with Rx > 1 (x decomposed in ZLocal): split y among processes, receive x
        #   ZLocal=(Nx/Rx, Ny) → YLocal=(Nx, Ny/Rx): pack dim=2 (y split), unpack dim=1 (x received)
        if N >= 3
            pack_dim = 3
            unpack_dim = 2
        elseif Ry > 1
            # Ry > 1: ZLocal has full x, partial y. Split x, receive y.
            pack_dim = 1
            unpack_dim = 2
        else
            # Rx > 1: ZLocal has partial x, full y. Split y, receive x.
            pack_dim = 2
            unpack_dim = 1
        end

        # Pack data into send buffer
        pack_start = time()
        pack_for_transpose!(send_buf, tf.buffers.z_local_data,
                            tf.counts.zy_send_counts, tf.counts.zy_send_displs,
                            pack_dim, comm_size, arch)
        tf.total_pack_time += time() - pack_start

        # Perform MPI communication
        _do_alltoallv!(send_buf, recv_buf,
                       tf.counts.zy_send_counts, tf.counts.zy_recv_counts,
                       comm, arch, tf.buffers)

        # Unpack received data
        unpack_start = time()
        unpack_from_transpose!(tf.buffers.y_local_data, recv_buf,
                              tf.counts.zy_recv_counts, tf.counts.zy_recv_displs,
                              unpack_dim, comm_size, arch)
        tf.total_unpack_time += time() - unpack_start
    end

    tf.buffers.active_layout[] = YLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

"""
    transpose_y_to_z!(tf::TransposableField)

Transpose from YLocal layout to ZLocal layout (reverse of Z→Y).
"""
function transpose_y_to_z!(tf::TransposableField{F,T,N}) where {F,T,N}
    # CRITICAL: Use error() instead of @assert for production safety
    if tf.buffers.active_layout[] != YLocal
        error("transpose_y_to_z!: Must be in YLocal layout, currently in $(tf.buffers.active_layout[])")
    end
    if tf.async_state.in_progress
        error("transpose_y_to_z!: Cannot call synchronous transpose while async operation is in progress. " *
              "Call wait_transpose!() first.")
    end

    topo = tf.topology
    Rx, Ry = topo.Rx, topo.Ry

    # Check for true 2D mesh on 2D domain (special case)
    use_local_extract = (N == 2) && (Rx > 1) && (Ry > 1)

    # Use same communicator logic as transpose_z_to_y! (reverse direction)
    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    elseif use_local_extract
        # True 2D mesh on 2D domain: use row_comm info for local extraction
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
        # CRITICAL: Validate buffers are allocated before use
        if tf.buffers.z_local_data === nothing || tf.buffers.y_local_data === nothing
            error("TransposableField buffers not allocated. " *
                  "Call allocate_transpose_buffers!(tf) before transpose operations.")
        end
        copyto!(vec(tf.buffers.z_local_data), vec(tf.buffers.y_local_data))
        tf.buffers.active_layout[] = ZLocal
        return tf
    end

    arch = tf.buffers.architecture
    start_time = time()

    # Validate comm_size matches counts array lengths (for Alltoallv path)
    if !use_local_extract && length(tf.counts.zy_recv_counts) != comm_size
        error("transpose_y_to_z!: comm_size ($comm_size) does not match " *
              "zy_recv_counts length ($(length(tf.counts.zy_recv_counts))). " *
              "This indicates TransposeCounts were computed for different topology.")
    end

    # CRITICAL: Validate communicator is not null for Alltoallv path
    if !use_local_extract && (comm === nothing || comm == MPI.COMM_NULL)
        error("transpose_y_to_z!: communicator is null but comm_size=$comm_size > 1. " *
              "This indicates topology was not properly initialized or was already freed.")
    end

    if use_local_extract
        # True 2D mesh on 2D domain: Y→Z is just local extraction (no MPI needed)
        # Extract our y-portion from the gathered YLocal data
        send_buf, _ = get_active_buffers(tf)

        pack_start = time()
        _pack_scatter_y!(send_buf, tf.buffers.y_local_data,
                         tf.local_shapes[ZLocal], tf.local_shapes[YLocal],
                         comm_size, topo.row_rank, arch)
        tf.total_pack_time += time() - pack_start

        # Copy to z_local_data
        z_size = prod(tf.local_shapes[ZLocal])
        copyto!(vec(tf.buffers.z_local_data), view(send_buf, 1:z_size))
    else
        # Standard Alltoallv approach
        send_buf, recv_buf = get_active_buffers(tf)

        # Determine pack/unpack dimensions based on ndims (reverse of Z→Y)
        # 3D: pack dim=2 (Y redistributed), unpack dim=3 (receiving Z chunks)
        # 2D with Ry > 1: YLocal=(Nx/Ry, Ny) → ZLocal=(Nx, Ny/Ry)
        #   Split y among processes, receive x: pack dim=2 (y split), unpack dim=1 (x received)
        # 2D with Rx > 1: YLocal=(Nx, Ny/Rx) → ZLocal=(Nx/Rx, Ny)
        #   Split x among processes, receive y: pack dim=1 (x split), unpack dim=2 (y received)
        if N >= 3
            pack_dim = 2
            unpack_dim = 3
        elseif Ry > 1
            # Reverse of Z→Y for Ry > 1: split y, receive x
            pack_dim = 2
            unpack_dim = 1
        else
            # Reverse of Z→Y for Rx > 1: split x, receive y
            pack_dim = 1
            unpack_dim = 2
        end

        pack_start = time()
        pack_for_transpose!(send_buf, tf.buffers.y_local_data,
                            tf.counts.zy_recv_counts, tf.counts.zy_recv_displs,
                            pack_dim, comm_size, arch)
        tf.total_pack_time += time() - pack_start

        # Note: swap send/recv counts for reverse direction
        _do_alltoallv!(send_buf, recv_buf,
                       tf.counts.zy_recv_counts, tf.counts.zy_send_counts,
                       comm, arch, tf.buffers)

        unpack_start = time()
        unpack_from_transpose!(tf.buffers.z_local_data, recv_buf,
                              tf.counts.zy_send_counts, tf.counts.zy_send_displs,
                              unpack_dim, comm_size, arch)
        tf.total_unpack_time += time() - unpack_start
    end

    tf.buffers.active_layout[] = ZLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

"""
    transpose_y_to_x!(tf::TransposableField)

Transpose from YLocal layout to XLocal layout.
"""
function transpose_y_to_x!(tf::TransposableField{F,T,N}) where {F,T,N}
    # CRITICAL: Use error() instead of @assert for production safety
    if tf.buffers.active_layout[] != YLocal
        error("transpose_y_to_x!: Must be in YLocal layout, currently in $(tf.buffers.active_layout[])")
    end
    if tf.async_state.in_progress
        error("transpose_y_to_x!: Cannot call synchronous transpose while async operation is in progress. " *
              "Call wait_transpose!() first.")
    end

    topo = tf.topology

    if topo.col_size == 1
        # CRITICAL: Validate buffers are allocated before use
        if tf.buffers.x_local_data === nothing || tf.buffers.y_local_data === nothing
            error("TransposableField buffers not allocated. " *
                  "Call allocate_transpose_buffers!(tf) before transpose operations.")
        end
        copyto!(vec(tf.buffers.x_local_data), vec(tf.buffers.y_local_data))
        tf.buffers.active_layout[] = XLocal
        return tf
    end

    # CRITICAL: Check that yx_counts were computed
    # For 2D with 1D decomposition (Rx>1, Ry==1 or vice versa), yx counts may not be computed
    # because Y↔X transposes are not needed in that configuration (ZLocal==XLocal)
    if sum(tf.counts.yx_send_counts) == 0 && sum(tf.counts.yx_recv_counts) == 0
        if N == 2 && (topo.Rx == 1 || topo.Ry == 1)
            error("Y→X transpose not supported for 2D domain with 1D decomposition (Rx=$(topo.Rx), Ry=$(topo.Ry)). " *
                  "In this configuration, ZLocal and XLocal are equivalent - use transpose_z_to_y! and " *
                  "transpose_y_to_z! instead.")
        else
            error("Y→X transpose counts not computed. This may indicate incorrect TransposableField setup.")
        end
    end

    arch = tf.buffers.architecture
    start_time = time()

    # Validate col_size matches counts array lengths
    if length(tf.counts.yx_send_counts) != topo.col_size
        error("transpose_y_to_x!: col_size ($(topo.col_size)) does not match " *
              "yx_send_counts length ($(length(tf.counts.yx_send_counts))). " *
              "This indicates TransposeCounts were computed for different topology.")
    end

    # CRITICAL: Validate communicator is not null
    if topo.col_comm === nothing || topo.col_comm == MPI.COMM_NULL
        error("transpose_y_to_x!: col_comm is null but col_size=$(topo.col_size) > 1. " *
              "This indicates topology was not properly initialized or was already freed.")
    end

    send_buf, recv_buf = get_active_buffers(tf)

    # Y→X: pack by Y dimension (dim=2), unpack by X dimension (dim=1)
    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.y_local_data,
                        tf.counts.yx_send_counts, tf.counts.yx_send_displs,
                        2, topo.col_size, arch)
    tf.total_pack_time += time() - pack_start

    _do_alltoallv!(send_buf, recv_buf,
                   tf.counts.yx_send_counts, tf.counts.yx_recv_counts,
                   topo.col_comm, arch, tf.buffers)

    unpack_start = time()
    unpack_from_transpose!(tf.buffers.x_local_data, recv_buf,
                          tf.counts.yx_recv_counts, tf.counts.yx_recv_displs,
                          1, topo.col_size, arch)
    tf.total_unpack_time += time() - unpack_start

    tf.buffers.active_layout[] = XLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

"""
    transpose_x_to_y!(tf::TransposableField)

Transpose from XLocal layout to YLocal layout (reverse of Y→X).
"""
function transpose_x_to_y!(tf::TransposableField{F,T,N}) where {F,T,N}
    # CRITICAL: Use error() instead of @assert for production safety
    if tf.buffers.active_layout[] != XLocal
        error("transpose_x_to_y!: Must be in XLocal layout, currently in $(tf.buffers.active_layout[])")
    end
    if tf.async_state.in_progress
        error("transpose_x_to_y!: Cannot call synchronous transpose while async operation is in progress. " *
              "Call wait_transpose!() first.")
    end

    topo = tf.topology

    if topo.col_size == 1
        # CRITICAL: Validate buffers are allocated before use
        if tf.buffers.x_local_data === nothing || tf.buffers.y_local_data === nothing
            error("TransposableField buffers not allocated. " *
                  "Call allocate_transpose_buffers!(tf) before transpose operations.")
        end
        copyto!(vec(tf.buffers.y_local_data), vec(tf.buffers.x_local_data))
        tf.buffers.active_layout[] = YLocal
        return tf
    end

    # CRITICAL: Check that yx_counts were computed
    # For 2D with 1D decomposition, yx counts may not be computed
    if sum(tf.counts.yx_send_counts) == 0 && sum(tf.counts.yx_recv_counts) == 0
        if N == 2 && (topo.Rx == 1 || topo.Ry == 1)
            error("X→Y transpose not supported for 2D domain with 1D decomposition (Rx=$(topo.Rx), Ry=$(topo.Ry)). " *
                  "In this configuration, ZLocal and XLocal are equivalent - use transpose_z_to_y! and " *
                  "transpose_y_to_z! instead.")
        else
            error("X→Y transpose counts not computed. This may indicate incorrect TransposableField setup.")
        end
    end

    arch = tf.buffers.architecture
    start_time = time()

    # Validate col_size matches counts array lengths
    if length(tf.counts.yx_recv_counts) != topo.col_size
        error("transpose_x_to_y!: col_size ($(topo.col_size)) does not match " *
              "yx_recv_counts length ($(length(tf.counts.yx_recv_counts))). " *
              "This indicates TransposeCounts were computed for different topology.")
    end

    # CRITICAL: Validate communicator is not null
    if topo.col_comm === nothing || topo.col_comm == MPI.COMM_NULL
        error("transpose_x_to_y!: col_comm is null but col_size=$(topo.col_size) > 1. " *
              "This indicates topology was not properly initialized or was already freed.")
    end

    send_buf, recv_buf = get_active_buffers(tf)

    # X→Y: pack by X dimension (dim=1), unpack by Y dimension (dim=2)
    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.x_local_data,
                        tf.counts.yx_recv_counts, tf.counts.yx_recv_displs,
                        1, topo.col_size, arch)
    tf.total_pack_time += time() - pack_start

    _do_alltoallv!(send_buf, recv_buf,
                   tf.counts.yx_recv_counts, tf.counts.yx_send_counts,
                   topo.col_comm, arch, tf.buffers)

    unpack_start = time()
    unpack_from_transpose!(tf.buffers.y_local_data, recv_buf,
                          tf.counts.yx_send_counts, tf.counts.yx_send_displs,
                          2, topo.col_size, arch)
    tf.total_unpack_time += time() - unpack_start

    tf.buffers.active_layout[] = YLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

