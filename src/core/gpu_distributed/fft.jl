# ============================================================================
# Distributed GPU FFT
# ============================================================================

"""
    DistributedGPUFFT

Distributed FFT using GPU computation and MPI communication.

For a domain decomposed along dimension `decomp_dim`:
- FFTs along other dimensions are local (use CUFFT)
- FFT along decomp_dim requires distributed transpose
"""
mutable struct DistributedGPUFFT
    config::DistributedGPUConfig

    # Local FFT plans (for non-distributed dimensions)
    local_plans::Dict{Int, Any}

    # Transpose buffers
    transpose_send::Union{Nothing, Any}  # GPU array for send
    transpose_recv::Union{Nothing, Any}  # GPU array for recv

    function DistributedGPUFFT(config::DistributedGPUConfig)
        new(config, Dict{Int, Any}(), nothing, nothing)
    end
end

"""
    distributed_fft_forward!(output, input, dfft::DistributedGPUFFT)

Perform forward distributed FFT.

Strategy:
1. Local FFTs on non-distributed dimensions (CUFFT)
2. Distributed transpose to make decomp_dim local
3. Local FFT on decomp_dim (CUFFT)
4. Distributed transpose back to original layout
"""
function distributed_fft_forward!(output, input, dfft::DistributedGPUFFT)
    config = dfft.config
    ndims_data = ndims(input)
    decomp_dim = config.decomp_dim

    # Step 1: Local FFTs on non-distributed dimensions
    current = input
    for dim in 1:ndims_data
        if dim != decomp_dim
            current = local_fft_dim!(current, dim, dfft)
        end
    end

    # Step 2: Distributed FFT on decomp_dim
    if config.size > 1
        # Need distributed transpose + local FFT + transpose back
        current = distributed_fft_dim!(current, decomp_dim, dfft, :forward)
    else
        # Serial: just do local FFT
        current = local_fft_dim!(current, decomp_dim, dfft)
    end

    output .= current
    return output
end

"""
    distributed_fft_backward!(output, input, dfft::DistributedGPUFFT)

Perform backward (inverse) distributed FFT.
"""
function distributed_fft_backward!(output, input, dfft::DistributedGPUFFT)
    config = dfft.config
    ndims_data = ndims(input)
    decomp_dim = config.decomp_dim

    current = input

    # Step 1: Distributed inverse FFT on decomp_dim
    if config.size > 1
        current = distributed_fft_dim!(current, decomp_dim, dfft, :backward)
    else
        current = local_ifft_dim!(current, decomp_dim, dfft)
    end

    # Step 2: Local inverse FFTs on non-distributed dimensions
    for dim in ndims_data:-1:1
        if dim != decomp_dim
            current = local_ifft_dim!(current, dim, dfft)
        end
    end

    output .= current
    return output
end

"""
    local_fft_dim!(data, dim, dfft)

Perform local FFT along dimension `dim` using GPU.
This is a placeholder - actual implementation requires CUDA extension.
"""
function local_fft_dim!(data, dim::Int, dfft::DistributedGPUFFT)
    arch = architecture(data)
    if is_gpu(arch)
        # Delegate to fft_in_dim! which the CUDA extension overrides for CuArray
        return fft_in_dim!(data, dim, :forward, arch)
    end
    data .= FFTW.fft(data, dim)
    return data
end

"""
    local_ifft_dim!(data, dim, dfft)

Perform local inverse FFT along dimension `dim` using GPU.
"""
function local_ifft_dim!(data, dim::Int, dfft::DistributedGPUFFT)
    arch = architecture(data)
    if is_gpu(arch)
        return fft_in_dim!(data, dim, :backward, arch)
    end
    data .= FFTW.ifft(data, dim)
    return data
end

"""
    distributed_fft_dim!(data, dim, dfft, direction)

Perform distributed FFT along decomposed dimension.

Steps:
1. Transpose to make dimension local (all-to-all)
2. Local FFT
3. Transpose back to original layout (all-to-all)
"""
function distributed_fft_dim!(data, dim::Int, dfft::DistributedGPUFFT, direction::Symbol)
    config = dfft.config

    # This requires MPI all-to-all communication
    # For CUDA-aware MPI: direct GPU buffer transfer
    # Non-CUDA-aware MPI is unsupported in strict GPU mode.

    if config.cuda_aware_mpi
        return distributed_fft_cuda_aware!(data, dim, dfft, direction)
    else
        error("Distributed GPU FFT requires CUDA-aware MPI; CPU staging fallback is disabled.")
    end
end

"""
    distributed_fft_staged!(data, dim, dfft, direction)

Distributed FFT with CPU staging for MPI communication.
"""
function distributed_fft_staged!(data, dim::Int, dfft::DistributedGPUFFT, direction::Symbol)
    is_gpu_array(data) && error(
        "CPU-staged distributed FFT is disabled for GPU data; use CUDA-aware MPI.")
    config = dfft.config
    arch = architecture(data)

    # Step 1: Copy data to CPU
    cpu_data = on_architecture(CPU(), data)

    # Step 2: MPI all-to-all transpose
    transposed = mpi_alltoall_transpose(cpu_data, dim, config)

    # Step 3: Copy back to GPU for local FFT
    gpu_transposed = arch isa CPU ? transposed : on_architecture(arch, transposed)

    # Step 4: Local FFT on now-local dimension
    if direction == :forward
        result = local_fft_dim!(gpu_transposed, dim, dfft)
    else
        result = local_ifft_dim!(gpu_transposed, dim, dfft)
    end

    # Step 5: Copy to CPU for reverse transpose
    cpu_result = on_architecture(CPU(), result)

    # Step 6: MPI all-to-all reverse transpose
    final_cpu = mpi_alltoall_transpose_reverse(cpu_result, dim, config)

    # Step 7: Copy back to GPU
    return arch isa CPU ? final_cpu : on_architecture(arch, final_cpu)
end

"""
    distributed_fft_cuda_aware!(data, dim, dfft, direction)

Distributed FFT with CUDA-aware MPI (direct GPU transfers).

This implementation uses direct GPU buffer MPI operations when CUDA-aware MPI is available,
avoiding costly CPU staging. The algorithm:
1. Pack GPU data into contiguous send buffer
2. Direct MPI.Alltoallv! on GPU buffers
3. Unpack into transposed layout
4. Local FFT on GPU
5. Reverse transpose via MPI
"""
function distributed_fft_cuda_aware!(data, dim::Int, dfft::DistributedGPUFFT, direction::Symbol)
    config = dfft.config
    arch = architecture(data)

    if !is_gpu_array(data)
        @warn "CUDA-aware MPI called with non-GPU data, falling back to staged" maxlog=1
        return distributed_fft_staged!(data, dim, dfft, direction)
    end

    # Verify CUDA-aware MPI is actually available
    if !_verify_cuda_aware_mpi()
        config.cuda_aware_mpi = false
        _CUDA_AWARE_MPI_VERIFIED[] = false  # Update global cache to match
        error("Distributed GPU FFT requires verified CUDA-aware MPI; CPU staging fallback is disabled.")
    end

    comm = config.comm
    nprocs = config.size
    rank = config.rank

    # Get data dimensions
    dims = size(data)
    global_n = config.global_shape[dim]

    # Compute transposed dimensions and buffer sizes
    transposed_dims = _compute_transposed_dims(dims, dim, config)
    send_elements = prod(dims)
    recv_elements = prod(transposed_dims)
    buf_size = max(send_elements, recv_elements)

    # Ensure transpose buffers are large enough for both directions
    if dfft.transpose_send === nothing || length(dfft.transpose_send) < buf_size
        dfft.transpose_send = similar(data, buf_size)
        dfft.transpose_recv = similar(data, buf_size)
    end

    # Step 1: Pack data into send buffer
    _gpu_pack_for_transpose!(dfft.transpose_send, data, dim, config)

    # CRITICAL: Synchronize GPU before MPI to ensure pack kernels have completed
    synchronize_device!(arch)

    # Step 2: Compute send/recv counts for all-to-all
    send_counts, recv_counts, send_displs, recv_displs = _compute_alltoall_counts(dims, dim, config)

    # Step 3: Direct GPU all-to-all via CUDA-aware MPI
    sendbuf = MPI.VBuffer(dfft.transpose_send, send_counts, send_displs)
    recvbuf = MPI.VBuffer(dfft.transpose_recv, recv_counts, recv_displs)
    MPI.Alltoallv!(sendbuf, recvbuf, comm)

    # Step 4: Unpack into transposed layout using GPU kernels
    gpu_transposed = similar(data, transposed_dims...)
    _gpu_unpack_from_transpose!(gpu_transposed, dfft.transpose_recv, dim, config)

    # Step 5: Local FFT on now-local dimension
    if direction == :forward
        result = local_fft_dim!(gpu_transposed, dim, dfft)
    else
        result = local_ifft_dim!(gpu_transposed, dim, dfft)
    end

    # Step 6: Pack for reverse transpose (result has transposed_dims shape)
    _gpu_pack_for_transpose!(dfft.transpose_send, result, dim, config)

    # CRITICAL: Synchronize GPU before MPI to ensure pack kernels have completed
    synchronize_device!(arch)

    # Step 7: Reverse all-to-all — REVERSE-orientation counts (split along `dim`,
    # assemble along `other_dim`), matching the reverse pack and
    # mpi_alltoall_transpose_reverse. Using the forward `_compute_alltoall_counts`
    # here mis-segments the Alltoallv on nprocs>=4 / non-divisible splits.
    rev_send_counts, rev_recv_counts, rev_send_displs, rev_recv_displs =
        _compute_alltoall_counts_reverse(size(result), dim, config)
    sendbuf_r = MPI.VBuffer(dfft.transpose_send, rev_send_counts, rev_send_displs)
    recvbuf_r = MPI.VBuffer(dfft.transpose_recv, rev_recv_counts, rev_recv_displs)
    MPI.Alltoallv!(sendbuf_r, recvbuf_r, comm)

    # Step 8: Unpack back to original layout using GPU kernels
    output = similar(data, dims...)
    _gpu_unpack_from_transpose!(output, dfft.transpose_recv, dim, config)

    return output
end

"""
    _verify_cuda_aware_mpi()

Runtime verification that CUDA-aware MPI actually works.
Performs a small test transfer to verify functionality.
"""
const _CUDA_AWARE_MPI_VERIFIED = Ref{Union{Nothing, Bool}}(nothing)

function _verify_cuda_aware_mpi()
    _CUDA_AWARE_MPI_VERIFIED[] !== nothing && return _CUDA_AWARE_MPI_VERIFIED[]

    try
        if !has_cuda()
            _CUDA_AWARE_MPI_VERIFIED[] = false
            return false
        end
        result = check_cuda_aware_mpi()
        _CUDA_AWARE_MPI_VERIFIED[] = result
        return result
    catch e
        @debug "CUDA-aware MPI verification failed: $e"
        _CUDA_AWARE_MPI_VERIFIED[] = false
        return false
    end
end

"""
    _gpu_pack_for_transpose!(send_buf, data, dim, config)

Pack data into contiguous buffer for MPI all-to-all transpose.
Default implementation - overridden by CUDA extension for GPU data.
"""
function _gpu_pack_for_transpose!(send_buf, data, dim::Int, config::DistributedGPUConfig)
    # Default: simple copy for CPU data
    copyto!(send_buf, vec(data))
end

"""
    _gpu_unpack_from_transpose!(output, recv_buf, dim, config)

Unpack received buffer into correctly-shaped output array after MPI all-to-all.
Default implementation - overridden by CUDA extension for GPU data.

For forward transpose: `dim` is the dimension being assembled (now fully local).
For reverse transpose: `dim` is passed as-is from the caller context.
The function determines direction from the output shape vs config.
"""
function _gpu_unpack_from_transpose!(output, recv_buf, dim::Int, config::DistributedGPUConfig)
    # Default CPU implementation: copy from flat buffer
    copyto!(vec(output), recv_buf)
end

"""
    _compute_alltoall_counts(dims, dim, config)

Compute send/recv counts and displacements for all-to-all transpose.

The transpose makes `dim` fully local (assembling from all ranks) and splits
`other_dim` among ranks. Send counts reflect how much of our local data goes
to each rank (split along other_dim). Recv counts reflect how much each rank
contributes to our transposed array (their portion along dim).
"""
function _compute_alltoall_counts(dims::Tuple, dim::Int, config::DistributedGPUConfig)
    nprocs = config.size
    rank = config.rank
    ndims_data = length(dims)
    global_n = config.global_shape[dim]
    local_n = dims[dim]

    # Determine the dimension that will become distributed after transpose
    other_dim = dim == ndims_data ? 1 : ndims_data
    other_n = dims[other_dim]

    # Product of all dimensions except dim and other_dim
    remaining = div(prod(dims), other_n * local_n)

    # Our chunk of other_dim after transpose
    chunk_other_me = div(other_n, nprocs) + (rank < mod(other_n, nprocs) ? 1 : 0)

    send_counts = Vector{Int}(undef, nprocs)
    recv_counts = Vector{Int}(undef, nprocs)
    send_displs = Vector{Int}(undef, nprocs)
    recv_displs = Vector{Int}(undef, nprocs)

    send_offset = 0
    recv_offset = 0

    for p in 0:(nprocs-1)
        # Chunk of other_dim that rank p will own after transpose
        chunk_other_p = div(other_n, nprocs) + (p < mod(other_n, nprocs) ? 1 : 0)

        # Chunk of dim that rank p currently owns
        local_n_p = div(global_n, nprocs) + (p < mod(global_n, nprocs) ? 1 : 0)

        # We send chunk_other_p indices of other_dim (with all our local_n and remaining)
        send_counts[p+1] = chunk_other_p * local_n * remaining

        # We receive local_n_p indices of dim from rank p (with our chunk_other_me and remaining)
        recv_counts[p+1] = local_n_p * chunk_other_me * remaining

        send_displs[p+1] = send_offset
        recv_displs[p+1] = recv_offset

        send_offset += send_counts[p+1]
        recv_offset += recv_counts[p+1]
    end

    return send_counts, recv_counts, send_displs, recv_displs
end

"""
    _compute_transposed_dims(dims, dim, config)

Compute dimensions after transpose operation.
"""
function _compute_transposed_dims(dims::Tuple, dim::Int, config::DistributedGPUConfig)
    nprocs = config.size
    rank = config.rank
    global_n = config.global_shape[dim]

    new_dims = collect(dims)
    new_dims[dim] = global_n  # Now fully local

    # Another dimension becomes distributed — use this rank's actual chunk size
    other_dim = dim == length(dims) ? 1 : length(dims)
    other_n = dims[other_dim]
    new_dims[other_dim] = div(other_n, nprocs) + (rank < mod(other_n, nprocs) ? 1 : 0)

    return Tuple(new_dims)
end

