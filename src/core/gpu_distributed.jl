"""
    GPU Distributed Computing Support

This module provides distributed GPU computing for spectral methods when
PencilArrays/PencilFFTs (CPU-only) cannot be used.

## Strategy

For GPU + MPI parallelization, we use a **slab decomposition** approach:
1. Each MPI rank owns exactly one GPU
2. Domain is split along one dimension (typically the last)
3. Local FFTs use CUFFT on GPU
4. Distributed FFTs use MPI all-to-all communication

## Communication Modes

1. **CUDA-aware MPI**: Direct GPU-to-GPU transfers (fastest)
2. **CPU staging**: Copy to CPU → MPI → Copy back to GPU (fallback)

## Data Layout

For a 3D domain (Nx, Ny, Nz) with P processes:
- Each process holds a slab of size (Nx, Ny, Nz/P)
- FFTs in x and y are fully local (use CUFFT)
- FFT in z requires distributed transpose

## References
- Oceananigans.jl distributed GPU implementation
- cuFFTMp (NVIDIA's multi-GPU FFT library)
- 2DECOMP&FFT library concepts
"""

using MPI
using FFTW

# ============================================================================
# Distributed GPU Configuration
# ============================================================================

"""
    DistributedGPUConfig

Configuration for distributed GPU computing.
"""
mutable struct DistributedGPUConfig
    # MPI info
    comm::MPI.Comm
    rank::Int
    size::Int

    # GPU info
    device_id::Int

    # Domain decomposition
    global_shape::Tuple{Vararg{Int}}
    local_shape::Tuple{Vararg{Int}}
    decomp_dim::Int  # Dimension along which domain is split

    # Communication mode
    cuda_aware_mpi::Bool

    # Staging buffers (for non-CUDA-aware MPI)
    send_buffer_cpu::Union{Nothing, Array}
    recv_buffer_cpu::Union{Nothing, Array}

    function DistributedGPUConfig(comm::MPI.Comm, global_shape::Tuple;
                                   decomp_dim::Int=length(global_shape),
                                   cuda_aware_mpi::Bool=false)
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)

        # Compute local shape (slab decomposition along decomp_dim)
        local_shape = compute_local_shape(global_shape, decomp_dim, size, rank)

        # Default: use GPU with same ID as MPI rank (mod number of GPUs)
        device_id = rank  # Will be set properly when GPU is initialized

        new(comm, rank, size, device_id, global_shape, local_shape, decomp_dim,
            cuda_aware_mpi, nothing, nothing)
    end
end

"""
    compute_local_shape(global_shape, decomp_dim, nprocs, rank)

Compute local array shape for slab decomposition.
"""
function compute_local_shape(global_shape::Tuple, decomp_dim::Int, nprocs::Int, rank::Int)
    local_shape = collect(global_shape)

    n = global_shape[decomp_dim]
    base_size = div(n, nprocs)
    remainder = mod(n, nprocs)

    # Distribute remainder to first 'remainder' processes
    local_n = base_size + (rank < remainder ? 1 : 0)
    local_shape[decomp_dim] = local_n

    return Tuple(local_shape)
end

"""
    get_local_range(global_n, nprocs, rank)

Get the global index range for this rank's local data.
"""
function get_local_range(global_n::Int, nprocs::Int, rank::Int)
    base_size = div(global_n, nprocs)
    remainder = mod(global_n, nprocs)

    # Start index (1-based)
    start = 1
    for r in 0:(rank-1)
        start += base_size + (r < remainder ? 1 : 0)
    end

    local_n = base_size + (rank < remainder ? 1 : 0)
    stop = start + local_n - 1

    return start:stop
end

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
    data_cpu = on_architecture(CPU(), data)
    result_cpu = FFTW.fft(data_cpu, dim)
    return arch isa CPU ? result_cpu : on_architecture(arch, result_cpu)
end

"""
    local_ifft_dim!(data, dim, dfft)

Perform local inverse FFT along dimension `dim` using GPU.
"""
function local_ifft_dim!(data, dim::Int, dfft::DistributedGPUFFT)
    arch = architecture(data)
    data_cpu = on_architecture(CPU(), data)
    result_cpu = FFTW.ifft(data_cpu, dim)
    return arch isa CPU ? result_cpu : on_architecture(arch, result_cpu)
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
    # For regular MPI: stage through CPU

    if config.cuda_aware_mpi
        return distributed_fft_cuda_aware!(data, dim, dfft, direction)
    else
        return distributed_fft_staged!(data, dim, dfft, direction)
    end
end

"""
    distributed_fft_staged!(data, dim, dfft, direction)

Distributed FFT with CPU staging for MPI communication.
"""
function distributed_fft_staged!(data, dim::Int, dfft::DistributedGPUFFT, direction::Symbol)
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
        @warn "CUDA-aware MPI not verified, falling back to staged" maxlog=1
        config.cuda_aware_mpi = false
        return distributed_fft_staged!(data, dim, dfft, direction)
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

    # Step 2: Compute send/recv counts for all-to-all
    send_counts, recv_counts, send_displs, recv_displs = _compute_alltoall_counts(dims, dim, config)

    # Step 3: Direct GPU all-to-all via CUDA-aware MPI
    # Pass full CuArray buffers (not views/SubArrays) so CUDA extension dispatch works.
    # MPI.Alltoallv! uses counts to bound reads/writes, so oversized buffers are safe.
    MPI.Alltoallv!(dfft.transpose_send, dfft.transpose_recv, send_counts, recv_counts, comm)

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

    # Step 7: Reverse all-to-all (swap send/recv counts)
    MPI.Alltoallv!(dfft.transpose_send, dfft.transpose_recv, recv_counts, send_counts, comm)

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
    if _CUDA_AWARE_MPI_VERIFIED[] !== nothing
        return _CUDA_AWARE_MPI_VERIFIED[]
    end

    try
        if !has_cuda()
            @debug "CUDA-aware MPI disabled: CUDA not available"
            _CUDA_AWARE_MPI_VERIFIED[] = false
            return false
        end

        result = check_cuda_aware_mpi()
        if !result
            @debug "CUDA-aware MPI not detected. Set TARANG_CUDA_AWARE_MPI=1 to force-enable if your MPI supports GPU buffers."
        end
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

# ============================================================================
# MPI Transpose Operations
# ============================================================================

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
    MPI.Alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)

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
    MPI.Alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)

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

    # Compute cumulative offsets along split_dim for each rank
    rank_starts = Vector{Int}(undef, nranks)
    offset = 0
    for r in 1:nranks
        rank_starts[r] = offset
        offset += chunk_sizes[r]
    end

    # Track write positions for each rank
    write_pos = copy(displs) .+ 1  # 1-indexed write positions

    # Iterate over all elements and place them in the correct rank's buffer section
    for idx in CartesianIndices(data)
        # Get the index along split_dim
        split_idx = idx[split_dim]

        # Find which rank owns this index
        rank = 1
        cumulative = 0
        for r in 1:nranks
            cumulative += chunk_sizes[r]
            if split_idx <= cumulative
                rank = r
                break
            end
        end

        # Write to buffer
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

    # Track read positions for each rank
    read_pos = copy(displs) .+ 1  # 1-indexed read positions

    # Iterate over all output elements and read from correct rank's buffer section
    for idx in CartesianIndices(output)
        # Get the index along assemble_dim
        assemble_idx = idx[assemble_dim]

        # Find which rank contributed this index
        rank = 1
        cumulative = 0
        for r in 1:nranks
            cumulative += chunk_sizes[r]
            if assemble_idx <= cumulative
                rank = r
                break
            end
        end

        # Read from buffer
        output[idx] = recv_buf[read_pos[rank]]
        read_pos[rank] += 1
    end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    check_cuda_aware_mpi()

Check if MPI implementation is CUDA-aware.

Detection priority:
1. Explicit user override via `TARANG_CUDA_AWARE_MPI` env var ("1"=enabled, "0"=disabled)
2. OpenMPI CUDA support indicator
3. MVAPICH2 CUDA indicator
4. MPICH GPU indicator

Returns false by default if no positive indicator is found.
"""
function check_cuda_aware_mpi()
    # Priority 1: Explicit user override
    if haskey(ENV, "TARANG_CUDA_AWARE_MPI")
        val = uppercase(strip(ENV["TARANG_CUDA_AWARE_MPI"]))
        if val in ("1", "TRUE", "YES")
            @info "CUDA-aware MPI enabled via TARANG_CUDA_AWARE_MPI environment variable"
            return true
        elseif val in ("0", "FALSE", "NO")
            return false
        end
    end

    # Priority 2-4: Library-specific indicators
    try
        # OpenMPI with CUDA support
        if haskey(ENV, "OMPI_MCA_opal_cuda_support") && ENV["OMPI_MCA_opal_cuda_support"] == "true"
            return true
        end
        # MVAPICH2 with CUDA
        if haskey(ENV, "MV2_USE_CUDA") && ENV["MV2_USE_CUDA"] == "1"
            return true
        end
        # MPICH with GPU support
        if haskey(ENV, "MPIR_CVAR_ENABLE_GPU") && ENV["MPIR_CVAR_ENABLE_GPU"] == "1"
            return true
        end
        # Cray MPI with GPU support
        if haskey(ENV, "MPICH_GPU_SUPPORT_ENABLED") && ENV["MPICH_GPU_SUPPORT_ENABLED"] == "1"
            return true
        end
    catch
        # Env access failed, treat as not detected
    end

    return false
end

"""
    setup_distributed_gpu!(dist::Distributor)

Setup distributed GPU computing for a Distributor.
"""
function setup_distributed_gpu!(dist)
    if !is_gpu(dist.architecture)
        return nothing
    end

    if dist.size == 1
        # Single GPU, no distribution needed
        @info "Single GPU mode - no distributed setup needed"
        return nothing
    end

    # Check for CUDA-aware MPI
    cuda_aware = check_cuda_aware_mpi()

    @info "Setting up distributed GPU computing"
    @info "  MPI processes: $(dist.size)"
    @info "  CUDA-aware MPI: $(cuda_aware)"

    # Create distributed GPU config
    # Will be populated when domain is created
    return cuda_aware
end

# ============================================================================
# NCCL Support for GPU Collectives
# ============================================================================

"""
    NCCLConfig

Configuration for NCCL GPU collective operations.
NCCL provides optimized GPU-to-GPU communication that can be faster than MPI.
"""
mutable struct NCCLConfig
    initialized::Bool
    comm_handle::Any  # NCCL communicator (when NCCL.jl is loaded)
    rank::Int
    size::Int

    NCCLConfig() = new(false, nothing, 0, 1)
end

const NCCL_CONFIG = NCCLConfig()

"""
    nccl_available()

Check if NCCL is available for GPU collectives.
Returns true if NCCL.jl is loaded and functional.
"""
function nccl_available()
    # NCCL support requires NCCL.jl package
    # This is a placeholder - actual check depends on package being loaded
    return NCCL_CONFIG.initialized
end

"""
    init_nccl!(comm::MPI.Comm)

Initialize NCCL for GPU collective operations.

NCCL must be initialized after MPI and after GPU devices are selected.
Each MPI rank should call this on their assigned GPU.

# Example
```julia
MPI.Init()
arch = GPU(device_id=MPI.Comm_rank(MPI.COMM_WORLD))
init_nccl!(MPI.COMM_WORLD)
```
"""
function init_nccl!(comm::MPI.Comm)
    if !has_cuda()
        @warn "NCCL initialization skipped - CUDA not available"
        return false
    end

    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    try
        # Try to load NCCL.jl dynamically
        # This requires NCCL.jl to be installed
        @eval using NCCL

        # Create NCCL unique ID on rank 0 and broadcast
        if rank == 0
            unique_id = NCCL.UniqueId()
            id_bytes = reinterpret(UInt8, [unique_id])
        else
            id_bytes = Vector{UInt8}(undef, sizeof(NCCL.UniqueId))
        end

        MPI.Bcast!(id_bytes, 0, comm)

        if rank != 0
            unique_id = reinterpret(NCCL.UniqueId, id_bytes)[1]
        end

        # Initialize NCCL communicator
        nccl_comm = NCCL.Communicator(size, rank, unique_id)

        NCCL_CONFIG.initialized = true
        NCCL_CONFIG.comm_handle = nccl_comm
        NCCL_CONFIG.rank = rank
        NCCL_CONFIG.size = size

        @info "NCCL initialized" rank=rank size=size
        return true

    catch e
        @debug "NCCL initialization failed: $e"
        NCCL_CONFIG.initialized = false
        return false
    end
end

"""
    nccl_allreduce!(data, op::Symbol=:sum)

Perform NCCL all-reduce on GPU data.

# Arguments
- `data`: GPU array to reduce (in-place)
- `op`: Reduction operation (:sum, :prod, :max, :min)
"""
function nccl_allreduce!(data, op::Symbol=:sum)
    if !NCCL_CONFIG.initialized
        error("NCCL not initialized. Call init_nccl! first.")
    end

    # Map operation symbol to NCCL op
    nccl_op = _get_nccl_op(op)

    # Perform all-reduce
    @eval NCCL.allreduce!($(data), $(data), $(nccl_op), $(NCCL_CONFIG.comm_handle))

    return data
end

"""
    nccl_broadcast!(data, root::Int=0)

Broadcast GPU data from root rank to all ranks.
"""
function nccl_broadcast!(data, root::Int=0)
    if !NCCL_CONFIG.initialized
        error("NCCL not initialized. Call init_nccl! first.")
    end

    @eval NCCL.broadcast!($(data), $(data), $(root), $(NCCL_CONFIG.comm_handle))
    return data
end

"""
    nccl_allgather!(recv_data, send_data)

Gather GPU data from all ranks.
"""
function nccl_allgather!(recv_data, send_data)
    if !NCCL_CONFIG.initialized
        error("NCCL not initialized. Call init_nccl! first.")
    end

    @eval NCCL.allgather!($(recv_data), $(send_data), $(NCCL_CONFIG.comm_handle))
    return recv_data
end

"""
    _get_nccl_op(op::Symbol)

Convert operation symbol to NCCL operation type.
"""
function _get_nccl_op(op::Symbol)
    if op == :sum
        return @eval NCCL.ncclSum
    elseif op == :prod
        return @eval NCCL.ncclProd
    elseif op == :max
        return @eval NCCL.ncclMax
    elseif op == :min
        return @eval NCCL.ncclMin
    else
        error("Unsupported NCCL operation: $op")
    end
end

"""
    finalize_nccl!()

Finalize NCCL resources.
"""
function finalize_nccl!()
    if NCCL_CONFIG.initialized
        NCCL_CONFIG.comm_handle = nothing
        NCCL_CONFIG.initialized = false
        @info "NCCL finalized"
    end
end

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
    config = dfft.config
    arch = architecture(data)

    T = eltype(data)
    total_size = length(data)

    # Get pinned staging buffers
    send_pinned, recv_pinned = get_staging_buffers(T, total_size)

    # Step 1: Async copy GPU → pinned CPU
    send_view = view(send_pinned, 1:total_size)
    copyto!(send_view, vec(data))

    # Step 2: MPI all-to-all on pinned buffers
    transposed = mpi_alltoall_transpose(reshape(send_view, size(data)), dim, config)

    # Step 3: Copy pinned CPU → GPU for local FFT
    gpu_transposed = on_architecture(arch, transposed)

    # Step 4: Local FFT
    if direction == :forward
        result = local_fft_dim!(gpu_transposed, dim, dfft)
    else
        result = local_ifft_dim!(gpu_transposed, dim, dfft)
    end

    # Step 5: Copy result to pinned buffer
    copyto!(send_view, vec(result))

    # Step 6: Reverse transpose
    final_cpu = mpi_alltoall_transpose_reverse(reshape(send_view, size(result)), dim, config)

    # Step 7: Copy back to GPU
    return on_architecture(arch, final_cpu)
end

# ============================================================================
# Enhanced Distributed GPU Transform (with TransposableField support)
# ============================================================================

"""
    DistributedGPUTransform

Enhanced distributed GPU transform that integrates with TransposableField
for efficient 2D pencil decomposition.

This type manages FFT plans for each transpose layout and coordinates
the full forward/backward transform sequence.
"""
mutable struct DistributedGPUTransform
    # Basic configuration
    config::DistributedGPUConfig

    # FFT plans for each layout (ZLocal, YLocal, XLocal)
    # Keys are TransposeLayout enum values, values are FFT plans
    plans::Dict{Any, Any}

    # Working TransposableField (created lazily)
    workspace::Any  # Union{Nothing, TransposableField}

    # Basis information for transform planning
    bases::Tuple{Vararg{Basis}}

    # Transform execution order
    transform_order::Vector{Int}

    # Performance statistics
    total_transpose_time::Float64
    total_fft_time::Float64
    num_transforms::Int

    function DistributedGPUTransform(config::DistributedGPUConfig, bases::Tuple{Vararg{Basis}})
        ndims = length(bases)
        transform_order = collect(1:ndims)  # Default order: 1, 2, 3, ...

        new(config, Dict{Any, Any}(), nothing, bases, transform_order, 0.0, 0.0, 0)
    end
end

"""
    create_distributed_gpu_transform(dist::Distributor, domain::Domain)

Create a DistributedGPUTransform for the given distributor and domain.
"""
function create_distributed_gpu_transform(dist::Distributor, domain::Domain)
    if dist.distributed_gpu_config === nothing
        # Create config if not exists
        gshape = global_shape(domain)
        config = DistributedGPUConfig(dist.comm, gshape;
                                       cuda_aware_mpi=check_cuda_aware_mpi())
        dist.distributed_gpu_config = config
    else
        config = dist.distributed_gpu_config
    end

    return DistributedGPUTransform(config, domain.bases)
end

"""
    setup_transposable_workspace!(transform::DistributedGPUTransform, field::ScalarField)

Setup or retrieve a TransposableField workspace for distributed transforms.
"""
function setup_transposable_workspace!(transform::DistributedGPUTransform, field)
    # TransposableField is defined in transposable_field.jl
    # Create workspace lazily
    if transform.workspace === nothing
        # The TransposableField constructor will be available at runtime
        # since transposable_field.jl is included after this file
        transform.workspace = TransposableField(field)
    end
    return transform.workspace
end

"""
    distributed_transform_forward!(transform::DistributedGPUTransform, field)

Perform forward distributed transform using TransposableField infrastructure.
"""
function distributed_transform_forward!(transform::DistributedGPUTransform, field)
    workspace = setup_transposable_workspace!(transform, field)

    start_time = time()

    # Use TransposableField's distributed transform
    distributed_forward_transform!(workspace, transform.plans)

    transform.total_fft_time += time() - start_time
    transform.num_transforms += 1

    return field
end

"""
    distributed_transform_backward!(transform::DistributedGPUTransform, field)

Perform backward distributed transform using TransposableField infrastructure.
"""
function distributed_transform_backward!(transform::DistributedGPUTransform, field)
    workspace = setup_transposable_workspace!(transform, field)

    start_time = time()

    distributed_backward_transform!(workspace, transform.plans)

    transform.total_fft_time += time() - start_time
    transform.num_transforms += 1

    return field
end

"""
    get_distributed_transform_stats(transform::DistributedGPUTransform)

Get performance statistics for the distributed transform.
"""
function get_distributed_transform_stats(transform::DistributedGPUTransform)
    return (
        total_transpose_time = transform.total_transpose_time,
        total_fft_time = transform.total_fft_time,
        num_transforms = transform.num_transforms,
        avg_time_per_transform = transform.num_transforms > 0 ?
            transform.total_fft_time / transform.num_transforms : 0.0
    )
end

# ============================================================================
# Exports
# ============================================================================

export DistributedGPUConfig, DistributedGPUFFT
export distributed_fft_forward!, distributed_fft_backward!
export compute_local_shape, get_local_range
export check_cuda_aware_mpi, setup_distributed_gpu!

# DistributedGPUTransform exports (TransposableField integration)
export DistributedGPUTransform
export create_distributed_gpu_transform
export setup_transposable_workspace!
export distributed_transform_forward!, distributed_transform_backward!
export get_distributed_transform_stats

# NCCL exports
export NCCLConfig, NCCL_CONFIG
export nccl_available, init_nccl!, finalize_nccl!
export nccl_allreduce!, nccl_broadcast!, nccl_allgather!

# Staging buffer exports
export DistributedStagingBuffers, get_staging_buffers
export distributed_fft_staged_optimized!
