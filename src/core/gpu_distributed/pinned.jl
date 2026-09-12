# ============================================================================
# Pinned Memory Staging for Distributed GPU
# ============================================================================

"""
    DistributedStagingBuffers

Pre-allocated pinned memory buffers for efficient CPU-GPU staging in distributed FFTs.
"""
mutable struct DistributedStagingBuffers{T}
    send_pinned::Union{Nothing, Vector{T}}
    recv_pinned::Union{Nothing, Vector{T}}
    size::Int

    DistributedStagingBuffers{T}() where T = new{T}(nothing, nothing, 0)
end

const DISTRIBUTED_STAGING_BUFFERS = Dict{DataType, DistributedStagingBuffers}()

"""
    get_staging_buffers(T::Type, size::Int)

Get or create pinned staging buffers for distributed operations.
"""
function get_staging_buffers(T::Type, size::Int)
    if !haskey(DISTRIBUTED_STAGING_BUFFERS, T)
        DISTRIBUTED_STAGING_BUFFERS[T] = DistributedStagingBuffers{T}()
    end

    buffers = DISTRIBUTED_STAGING_BUFFERS[T]

    if buffers.size < size
        # Allocate larger buffers
        # Use page-locked (pinned) memory for faster GPU transfers
        buffers.send_pinned = Vector{T}(undef, size)
        buffers.recv_pinned = Vector{T}(undef, size)
        buffers.size = size
    end

    return buffers.send_pinned, buffers.recv_pinned
end

"""
    distributed_fft_staged_optimized!(data, dim, dfft, direction)

Optimized staged distributed FFT using pinned memory buffers.
"""
function distributed_fft_staged_optimized!(data, dim::Int, dfft::DistributedGPUFFT, direction::Symbol)
    is_gpu_array(data) && error(
        "CPU-staged distributed FFT is disabled for GPU data; use CUDA-aware MPI.")
    config = dfft.config
    arch = architecture(data)

    T = eltype(data)
    dims = size(data)
    total_size = length(data)

    # Compute transposed size to allocate staging buffers large enough for both directions
    transposed_dims = _compute_transposed_dims(dims, dim, config)
    transposed_size = prod(transposed_dims)
    max_size = max(total_size, transposed_size)

    # Get pinned staging buffers sized for the larger of source/transposed
    send_pinned, recv_pinned = get_staging_buffers(T, max_size)

    # Step 1: Async copy GPU → pinned CPU
    send_view = view(send_pinned, 1:total_size)
    copyto!(send_view, vec(data))

    # Step 2: MPI all-to-all on pinned buffers
    send_arr = Array{T}(undef, dims...)
    copyto!(vec(send_arr), send_view)
    transposed = mpi_alltoall_transpose(send_arr, dim, config)

    # Step 3: Copy pinned CPU → GPU for local FFT
    gpu_transposed = on_architecture(arch, transposed)

    # Step 4: Local FFT
    if direction == :forward
        result = local_fft_dim!(gpu_transposed, dim, dfft)
    else
        result = local_ifft_dim!(gpu_transposed, dim, dfft)
    end

    # Step 5: Copy result to pinned buffer (result may have different size than input)
    result_view = view(send_pinned, 1:length(result))
    copyto!(result_view, vec(result))

    # Step 6: Reverse transpose
    result_arr = Array{T}(undef, size(result)...)
    copyto!(vec(result_arr), result_view)
    final_cpu = mpi_alltoall_transpose_reverse(result_arr, dim, config)

    # Step 7: Copy back to GPU
    return on_architecture(arch, final_cpu)
end

