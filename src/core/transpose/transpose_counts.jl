"""
    Transpose Counts - MPI count computation for TransposableField

This file contains functions for computing send/receive counts and
displacements for MPI.Alltoallv operations.
"""

# ============================================================================
# Transpose Count Computation
# ============================================================================

"""
    compute_transpose_counts!(tf::TransposableField)

Compute send/receive counts and displacements for MPI.Alltoallv.
"""
function compute_transpose_counts!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology

    if topo.Rx * topo.Ry == 1
        return
    end

    # Z↔Y transpose counts (along row communicator)
    compute_zy_counts_2d!(tf)

    # Y↔X transpose counts (along column communicator)
    compute_yx_counts_2d!(tf)

    # Validate count consistency
    _validate_transpose_counts!(tf)
end

"""
    _validate_transpose_counts!(tf::TransposableField)

Validate that computed counts are consistent and non-negative.
Catches issues with zero-sized partitions and count mismatches.
"""
function _validate_transpose_counts!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology

    # Validate Z↔Y counts
    if topo.row_size > 1
        # Check for negative counts (should never happen, but defensive)
        for i in 1:length(tf.counts.zy_send_counts)
            if tf.counts.zy_send_counts[i] < 0 || tf.counts.zy_recv_counts[i] < 0
                error("Negative count detected in Z↔Y transpose: " *
                      "send_counts[$i]=$(tf.counts.zy_send_counts[i]), " *
                      "recv_counts[$i]=$(tf.counts.zy_recv_counts[i]). " *
                      "This indicates a bug in count computation.")
            end
        end

        # Validate that sum of send_counts equals total local data size
        z_shape = tf.local_shapes[ZLocal]
        expected_send_total = prod(z_shape)
        actual_send_total = sum(tf.counts.zy_send_counts)
        if actual_send_total != expected_send_total
            if actual_send_total == 0
                error("Z↔Y send counts are all zeros but row_size > 1. " *
                      "This indicates counts were not computed properly. " *
                      "Expected sum=$expected_send_total for ZLocal shape=$z_shape.")
            else
                @debug "Z↔Y send count mismatch: sum(send_counts)=$actual_send_total, " *
                       "expected=$expected_send_total. This may indicate a decomposition issue." maxlog=1
            end
        end
    end

    # Validate Y↔X counts
    if topo.col_size > 1
        for i in 1:length(tf.counts.yx_send_counts)
            if tf.counts.yx_send_counts[i] < 0 || tf.counts.yx_recv_counts[i] < 0
                error("Negative count detected in Y↔X transpose: " *
                      "send_counts[$i]=$(tf.counts.yx_send_counts[i]), " *
                      "recv_counts[$i]=$(tf.counts.yx_recv_counts[i]). " *
                      "This indicates a bug in count computation.")
            end
        end
    end

    # CRITICAL: Validate displs are correct cumulative sums of counts
    # Incorrect displs cause pack/unpack to read/write wrong buffer positions
    # Skip zy_send validation for 2D/2D-mesh Allgatherv case (send_displs are all 0)
    is_allgatherv_case = topo.Rx > 1 && topo.Ry > 1 && length(tf.global_shape) == 2
    if !is_allgatherv_case
        _validate_displs_cumsum(tf.counts.zy_send_counts, tf.counts.zy_send_displs, "zy_send")
    end
    _validate_displs_cumsum(tf.counts.zy_recv_counts, tf.counts.zy_recv_displs, "zy_recv")
    _validate_displs_cumsum(tf.counts.yx_send_counts, tf.counts.yx_send_displs, "yx_send")
    _validate_displs_cumsum(tf.counts.yx_recv_counts, tf.counts.yx_recv_displs, "yx_recv")
end

"""
    _validate_displs_cumsum(counts, displs, name)

Validate that displs array is the cumulative sum of counts (starting at 0).
"""
function _validate_displs_cumsum(counts::Vector{Int}, displs::Vector{Int}, name::String)
    if length(counts) != length(displs)
        error("$name: counts length ($(length(counts))) != displs length ($(length(displs)))")
    end

    expected_displ = 0
    for i in 1:length(counts)
        if displs[i] != expected_displ
            error("$name displs[$i]=$(displs[i]) does not match expected cumulative sum $expected_displ. " *
                  "Previous counts: $(counts[1:i-1]), displs: $(displs[1:i])")
        end
        expected_displ += counts[i]
    end
end

function compute_zy_counts_2d!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology
    gshape = tf.global_shape

    if N == 3
        # For 3D, Z→Y uses row_comm
        if topo.row_size == 1
            return
        end
        nprocs = topo.row_size
        Nx, Ny, Nz = gshape

        z_shape = tf.local_shapes[ZLocal]
        y_shape = tf.local_shapes[YLocal]

        # Z→Y: redistribute z-dimension among row communicator
        # Local x-size stays the same (Nx/Rx)
        local_nx = z_shape[1]

        send_offset = 0
        recv_offset = 0

        for p in 0:(nprocs-1)
            # Chunks of z we send to process p
            Nz_p = divide_evenly(Nz, nprocs, p)
            local_ny_send = z_shape[2]
            send_count = local_nx * local_ny_send * Nz_p

            # Chunks of y we receive from process p
            Ny_p = divide_evenly(Ny, nprocs, p)
            local_nz_recv = y_shape[3]
            recv_count = local_nx * Ny_p * local_nz_recv

            tf.counts.zy_send_counts[p+1] = send_count
            tf.counts.zy_recv_counts[p+1] = recv_count
            tf.counts.zy_send_displs[p+1] = send_offset
            tf.counts.zy_recv_displs[p+1] = recv_offset

            send_offset += send_count
            recv_offset += recv_count
        end

    elseif N == 2
        Nx, Ny = gshape
        Rx, Ry = topo.Rx, topo.Ry

        z_shape = tf.local_shapes[ZLocal]
        y_shape = tf.local_shapes[YLocal]

        # Check for true 2D mesh decomposition
        if Rx > 1 && Ry > 1
            # True 2D mesh: Z→Y uses Allgatherv on row_comm (size = Ry) to gather y.
            # ZLocal: (Nx/Rx, Ny/Ry) → YLocal: (Nx/Rx, Ny)
            # Each process contributes its local y-slice; all receive the full y.
            # Only recv_counts/recv_displs are used by Allgatherv.
            # send_counts are set to the local data size for consistency/validation.
            nprocs = topo.row_size  # = Ry

            if nprocs <= 1
                return
            end

            local_nx = z_shape[1]  # Nx/Rx, stays the same
            local_ny = z_shape[2]  # Ny/Ry, our y-portion
            local_send_total = local_nx * local_ny  # Total local data sent once

            recv_offset = 0

            for p in 0:(nprocs-1)
                # Send: our full local data (Allgatherv sends once, not per-process)
                # Store local_send_total in slot 0 only; others are zero.
                # This avoids the misleading sum(send_counts) = nprocs * local_data.
                tf.counts.zy_send_counts[p+1] = (p == 0) ? local_send_total : 0
                tf.counts.zy_send_displs[p+1] = 0

                # Receive: y-portion from process p
                Ny_p = divide_evenly(Ny, nprocs, p)
                recv_count = local_nx * Ny_p

                tf.counts.zy_recv_counts[p+1] = recv_count
                tf.counts.zy_recv_displs[p+1] = recv_offset

                recv_offset += recv_count
            end
        else
            # 1D decomposition: use whichever communicator has multiple processes
            actual_nprocs = topo.row_size > 1 ? topo.row_size : topo.col_size

            if actual_nprocs <= 1
                return  # No transpose needed
            end

            send_offset = 0
            recv_offset = 0

            if Ry > 1
                # Ry > 1: y decomposed in ZLocal, x will be decomposed in YLocal
                # ZLocal = (Nx, Ny/Ry) → YLocal = (Nx/Ry, Ny)
                # We have full x, partial y. We send x-portions to each process, keeping our y.
                # We receive our x-portion from each process with their y-portions.
                local_ny_send = z_shape[2]   # Our local y portion (Ny/Ry)
                local_nx_recv = y_shape[1]   # Our local x portion in YLocal (Nx/Ry)

                for p in 0:(actual_nprocs-1)
                    # Send: x-portion for process p with our y-portion
                    Nx_p = divide_evenly(Nx, actual_nprocs, p)
                    send_count = Nx_p * local_ny_send

                    # Recv: our x-portion with y-portion from process p
                    Ny_p = divide_evenly(Ny, actual_nprocs, p)
                    recv_count = local_nx_recv * Ny_p

                    tf.counts.zy_send_counts[p+1] = send_count
                    tf.counts.zy_recv_counts[p+1] = recv_count
                    tf.counts.zy_send_displs[p+1] = send_offset
                    tf.counts.zy_recv_displs[p+1] = recv_offset

                    send_offset += send_count
                    recv_offset += recv_count
                end
            else
                # Rx > 1: x decomposed in ZLocal, y will be decomposed in YLocal
                # ZLocal = (Nx/Rx, Ny) → YLocal = (Nx, Ny/Rx)
                # We have partial x, full y. We send y-portions to each process, keeping our x.
                # We receive our y-portion from each process with their x-portions.
                local_nx_send = z_shape[1]   # Our local x portion (Nx/Rx)
                local_ny_recv = y_shape[2]   # Our local y portion in YLocal (Ny/Rx)

                for p in 0:(actual_nprocs-1)
                    # Send: our x-portion with y-portion for process p
                    Ny_p = divide_evenly(Ny, actual_nprocs, p)
                    send_count = local_nx_send * Ny_p

                    # Recv: x-portion from process p with our y-portion
                    Nx_p = divide_evenly(Nx, actual_nprocs, p)
                    recv_count = Nx_p * local_ny_recv

                    tf.counts.zy_send_counts[p+1] = send_count
                    tf.counts.zy_recv_counts[p+1] = recv_count
                    tf.counts.zy_send_displs[p+1] = send_offset
                    tf.counts.zy_recv_displs[p+1] = recv_offset

                    send_offset += send_count
                    recv_offset += recv_count
                end
            end
        end
    end
end

function compute_yx_counts_2d!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology

    if topo.col_size == 1
        return
    end

    nprocs = topo.col_size
    gshape = tf.global_shape
    Rx, Ry = topo.Rx, topo.Ry

    if N == 3
        Nx, Ny, Nz = gshape

        y_shape = tf.local_shapes[YLocal]
        x_shape = tf.local_shapes[XLocal]

        # Y→X transpose: redistribute y→partial, x→full
        # - We have YLocal: (Nx/Rx, Ny, Nz/Ry) with full y
        # - We want XLocal: (Nx, Ny/Rx, Nz/Ry) with full x
        # - Communication is along col_comm (size = Rx)
        # - Send: partition our y dimension among col_comm processes
        # - Recv: gather x dimension from col_comm processes
        local_nx = y_shape[1]   # Our x portion (Nx/Rx)
        local_ny = x_shape[2]   # Our y portion after transpose (Ny/Rx)
        local_nz = y_shape[3]   # z portion (same in both layouts)

        send_offset = 0
        recv_offset = 0

        for p in 0:(nprocs-1)
            # Send: chunks of our y dimension to process p
            # Process p will own y indices: local_range(Ny, nprocs, p)
            Ny_p = divide_evenly(Ny, nprocs, p)
            send_count = local_nx * Ny_p * local_nz

            # Recv: chunks of x dimension from process p
            # Process p owns x indices: local_range(Nx, nprocs, p)
            Nx_p = divide_evenly(Nx, nprocs, p)
            recv_count = Nx_p * local_ny * local_nz

            tf.counts.yx_send_counts[p+1] = send_count
            tf.counts.yx_recv_counts[p+1] = recv_count
            tf.counts.yx_send_displs[p+1] = send_offset
            tf.counts.yx_recv_displs[p+1] = recv_offset

            send_offset += send_count
            recv_offset += recv_count
        end

    elseif N == 2 && Rx > 1 && Ry > 1
        # True 2D mesh on 2D domain
        # Y→X uses col_comm (size = Rx) for Alltoallv exchange
        # YLocal: (Nx/Rx, Ny) → XLocal: (Nx, Ny/Rx)
        Nx, Ny = gshape

        y_shape = tf.local_shapes[YLocal]  # (Nx/Rx, Ny)
        x_shape = tf.local_shapes[XLocal]  # (Nx, Ny/Rx)

        local_nx = y_shape[1]   # Our x portion (Nx/Rx)
        local_ny = x_shape[2]   # Our y portion after transpose (Ny/Rx)

        send_offset = 0
        recv_offset = 0

        for p in 0:(nprocs-1)
            # Send: chunks of our y dimension to process p
            # Process p will own y indices: local_range(Ny, nprocs, p)
            Ny_p = divide_evenly(Ny, nprocs, p)
            send_count = local_nx * Ny_p

            # Recv: chunks of x dimension from process p
            # Process p owns x indices: local_range(Nx, nprocs, p)
            Nx_p = divide_evenly(Nx, nprocs, p)
            recv_count = Nx_p * local_ny

            tf.counts.yx_send_counts[p+1] = send_count
            tf.counts.yx_recv_counts[p+1] = recv_count
            tf.counts.yx_send_displs[p+1] = send_offset
            tf.counts.yx_recv_displs[p+1] = recv_offset

            send_offset += send_count
            recv_offset += recv_count
        end
    end
    # Note: For 2D with 1D decomposition, Y↔X is handled in Z↔Y (no separate Y↔X needed)
end

