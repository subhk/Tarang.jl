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
2. Non-CUDA-aware MPI is rejected; host staging fallback is disabled.

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

# MPI, FFTW already in Tarang.jl

# GPU synchronization helper — overridden by CUDA extension
synchronize_device!(::CPU) = nothing
synchronize_device!(arch::AbstractArchitecture) = nothing  # no-op fallback; CUDA ext overrides

# ============================================================================
# Distributed GPU Configuration
# ============================================================================

"""
    DistributedGPUConfig

Configuration for distributed GPU computing.
"""
mutable struct DistributedGPUConfig <: AbstractDistributedGPUConfig
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

