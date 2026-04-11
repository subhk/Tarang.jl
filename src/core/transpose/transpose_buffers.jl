"""
    Transpose Buffers - Buffer allocation for TransposableField

This file contains functions for allocating data and communication buffers
for transpose operations.
"""

"""
    allocate_transpose_buffers!(tf::TransposableField)

Allocate data and communication buffers for transpose operations.
Includes double-buffered communication buffers for async operations.
"""
function allocate_transpose_buffers!(tf::TransposableField{F,T,N}) where {F,T,N}
    arch = tf.buffers.architecture

    # Allocate data arrays for each layout
    z_shape = tf.local_shapes[ZLocal]
    y_shape = tf.local_shapes[YLocal]
    x_shape = tf.local_shapes[XLocal]

    # CRITICAL: Check for integer overflow in buffer size calculation
    # Very large grids with small process counts can overflow Int64
    function checked_prod(shape::NTuple)
        result = Int64(1)
        for dim in shape
            # Check if multiplication would overflow
            if dim > 0 && result > typemax(Int64) ÷ dim
                error("Buffer size overflow: shape $shape would require more than " *
                      "$(typemax(Int64)) elements. Reduce grid size or increase process count.")
            end
            result *= dim
        end
        return result
    end

    z_size = checked_prod(z_shape)
    y_size = checked_prod(y_shape)
    x_size = checked_prod(x_shape)

    # Create arrays on appropriate architecture
    z_data = zeros(arch, T, z_shape...)
    y_data = zeros(arch, T, y_shape...)
    x_data = zeros(arch, T, x_shape...)

    # Compute max buffer size needed for communication
    max_size = max(z_size, y_size, x_size)

    # Double-buffered communication buffers
    send_buf_1 = zeros(arch, T, max_size)
    recv_buf_1 = zeros(arch, T, max_size)
    send_buf_2 = zeros(arch, T, max_size)
    recv_buf_2 = zeros(arch, T, max_size)

    # CPU staging buffers for non-CUDA-aware MPI
    send_staging = is_gpu(arch) ? zeros(T, max_size) : nothing
    recv_staging = is_gpu(arch) ? zeros(T, max_size) : nothing

    # Update buffers struct
    tf.buffers = TransposeBuffers{T,N}(
        x_data, y_data, z_data,
        send_buf_1, recv_buf_1, send_buf_2, recv_buf_2,
        send_staging, recv_staging,
        tf.buffers.active_layout,
        tf.buffers.active_buffer,
        tf.buffers.staging_locked,
        arch
    )

    return tf
end

# ============================================================================
# Buffer Helpers
# ============================================================================

"""
    get_active_buffers(tf::TransposableField)

Get the currently active send/recv buffer pair.
"""
function get_active_buffers(tf::TransposableField)
    if tf.buffers.active_buffer[] == 1
        return (tf.buffers.send_buffer, tf.buffers.recv_buffer)
    else
        return (tf.buffers.send_buffer_2, tf.buffers.recv_buffer_2)
    end
end

"""
    swap_buffers!(tf::TransposableField)

Swap to the other buffer set for double buffering.
"""
function swap_buffers!(tf::TransposableField)
    tf.buffers.active_buffer[] = tf.buffers.active_buffer[] == 1 ? 2 : 1
    return tf
end
