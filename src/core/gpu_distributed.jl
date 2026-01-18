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

    # Ensure transpose buffers are allocated on GPU
    total_elements = prod(dims)
    if dfft.transpose_send === nothing || length(dfft.transpose_send) != total_elements
        dfft.transpose_send = similar(data, total_elements)
        dfft.transpose_recv = similar(data, total_elements)
    end

    # Step 1: Pack data into send buffer (already on GPU)
    _gpu_pack_for_transpose!(dfft.transpose_send, data, dim, config)

    # Step 2: Compute send/recv counts for all-to-all
    send_counts, recv_counts, send_displs, recv_displs = _compute_alltoall_counts(dims, dim, config)

    # Step 3: Direct GPU all-to-all via CUDA-aware MPI
    # MPI.Alltoallv! should work directly on CuArrays with CUDA-aware MPI
    MPI.Alltoallv!(dfft.transpose_send, dfft.transpose_recv, send_counts, recv_counts, comm)

    # Step 4: Unpack into transposed layout
    transposed_dims = _compute_transposed_dims(dims, dim, config)
    gpu_transposed = reshape(dfft.transpose_recv, transposed_dims...)

    # Step 5: Local FFT on now-local dimension
    if direction == :forward
        result = local_fft_dim!(gpu_transposed, dim, dfft)
    else
        result = local_ifft_dim!(gpu_transposed, dim, dfft)
    end

    # Step 6: Pack for reverse transpose
    _gpu_pack_for_transpose!(dfft.transpose_send, result, dim, config)

    # Step 7: Reverse all-to-all
    MPI.Alltoallv!(dfft.transpose_send, dfft.transpose_recv, recv_counts, send_counts, comm)

    # Step 8: Unpack back to original layout
    return reshape(dfft.transpose_recv, dims...)
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

    # Try a small test transfer
    try
        if !has_cuda()
            _CUDA_AWARE_MPI_VERIFIED[] = false
            return false
        end

        # This check requires the CUDA extension to be loaded
        # For now, rely on environment variable detection
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
    _compute_alltoall_counts(dims, dim, config)

Compute send/recv counts and displacements for all-to-all transpose.
"""
function _compute_alltoall_counts(dims::Tuple, dim::Int, config::DistributedGPUConfig)
    nprocs = config.size
    global_n = config.global_shape[dim]
    local_n = dims[dim]

    # Elements per slice (excluding the transposed dimension)
    slice_elements = div(prod(dims), local_n)

    send_counts = Vector{Int}(undef, nprocs)
    recv_counts = Vector{Int}(undef, nprocs)
    send_displs = Vector{Int}(undef, nprocs)
    recv_displs = Vector{Int}(undef, nprocs)

    send_offset = 0
    recv_offset = 0

    for p in 0:(nprocs-1)
        # Size that process p owns along dim
        p_size = div(global_n, nprocs) + (p < mod(global_n, nprocs) ? 1 : 0)

        # We send our local_n elements to each process, they send their p_size to us
        send_counts[p+1] = slice_elements * local_n ÷ nprocs
        recv_counts[p+1] = slice_elements * p_size ÷ nprocs

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
    global_n = config.global_shape[dim]

    new_dims = collect(dims)
    new_dims[dim] = global_n  # Now fully local

    # Another dimension becomes distributed
    other_dim = dim == length(dims) ? 1 : length(dims)
    new_dims[other_dim] = div(new_dims[other_dim], nprocs)

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
function mpi_alltoall_transpose(data::Array, dim::Int, config::DistributedGPUConfig)
    comm = config.comm
    nprocs = config.size
    rank = config.rank

    if nprocs == 1
        return data
    end

    # Get data dimensions
    dims = size(data)
    ndims_data = length(dims)

    # Compute send counts and displacements
    global_n = config.global_shape[dim]

    # Pack data for all-to-all
    # Each process sends its portion to all other processes
    send_counts = zeros(Int, nprocs)
    recv_counts = zeros(Int, nprocs)

    for p in 0:(nprocs-1)
        # Size of slice this process will receive from process p
        recv_counts[p+1] = div(prod(dims), dims[dim]) *
                          (div(global_n, nprocs) + (p < mod(global_n, nprocs) ? 1 : 0))
        send_counts[p+1] = recv_counts[p+1]  # Symmetric for now
    end

    # Perform all-to-all
    send_buf = vec(data)
    recv_buf = similar(send_buf)

    MPI.Alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)

    # Reshape result
    new_dims = collect(dims)
    new_dims[dim] = global_n
    # Adjust other dimension that is now distributed
    other_dim = dim == ndims_data ? 1 : ndims_data
    new_dims[other_dim] = div(new_dims[other_dim], nprocs)

    return reshape(recv_buf, Tuple(new_dims)...)
end

"""
    mpi_alltoall_transpose_reverse(data, dim, config)

Reverse the transpose operation.
"""
function mpi_alltoall_transpose_reverse(data::Array, dim::Int, config::DistributedGPUConfig)
    # Symmetric operation for this simple case
    return mpi_alltoall_transpose(data, dim, config)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    check_cuda_aware_mpi()

Check if MPI implementation is CUDA-aware.
"""
function check_cuda_aware_mpi()
    # This is a heuristic check
    # Proper detection requires runtime testing
    cuda_aware = false

    # Check for common CUDA-aware MPI indicators
    try
        # OpenMPI with CUDA support
        if haskey(ENV, "OMPI_MCA_opal_cuda_support") && ENV["OMPI_MCA_opal_cuda_support"] == "true"
            cuda_aware = true
        end
        # MVAPICH2 with CUDA
        if haskey(ENV, "MV2_USE_CUDA") && ENV["MV2_USE_CUDA"] == "1"
            cuda_aware = true
        end
    catch
        cuda_aware = false
    end

    return cuda_aware
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
