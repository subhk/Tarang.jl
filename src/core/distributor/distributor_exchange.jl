# Async MPI exchange, communication buffer management, and diagnostics.

# ============================================================================
# Optimized MPI Communication Patterns
# ============================================================================

"""
    async_allreduce!(dest::AbstractArray, src::AbstractArray, op, dist::Distributor)

Non-blocking allreduce operation for overlapping communication and computation.
Returns an MPI request that can be waited on later.

GPU arrays require CUDA-aware MPI; host staging is disabled.
"""
function async_allreduce!(dest::AbstractArray, src::AbstractArray, op, dist::Distributor)
    start_time = time()

    # Handle GPU arrays with non-CUDA-aware MPI
    if is_gpu_array(src) && !check_cuda_aware_mpi()
        error("GPU all-reduce requires CUDA-aware MPI; host staging fallback is disabled.")
    end

    request = MPI.Iallreduce!(src, dest, op, dist.comm)
    dist.performance_stats.mpi_operations += 1

    return (request=request, staged=false)
end

"""
    wait_async!(async_result, dist::Distributor)

Wait for an asynchronous MPI operation to complete.
Handles both raw MPI.Request and staged GPU operations.
"""
function wait_async!(async_result::NamedTuple, dist::Distributor)
    start_time = time()

    MPI.Wait(async_result.request)

    async_result.staged && error(
        "Async GPU communication returned a staged result; host staging is disabled.")

    dist.performance_stats.communication_time += time() - start_time
end

# Legacy support for raw MPI.Request
function wait_async!(request::MPI.Request, dist::Distributor)
    start_time = time()
    MPI.Wait(request)
    dist.performance_stats.communication_time += time() - start_time
end

"""
    neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                       recv_left::AbstractArray, recv_right::AbstractArray,
                       dim::Int, dist::Distributor; periodic::Bool=true)

Optimized nearest-neighbor exchange for stencil operations.
Uses non-blocking MPI sends/receives for efficient bidirectional communication.

# Arguments
- `send_left/right`: Data to send to left/right neighbor
- `recv_left/right`: Buffers for received data
- `dim`: Mesh dimension for neighbor identification
- `dist`: Distributor with MPI info
- `periodic`: If true (default), assumes periodic domain with wrap-around neighbors.
              If false, processes at domain boundaries have no neighbors on that side
              and the corresponding recv buffers are left unchanged.

# Note
For periodic domains (the default), every process has both left and right neighbors,
even at domain boundaries (they wrap around). For non-periodic domains, boundary
processes will skip communication on the boundary side.
"""
function neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                           recv_left::AbstractArray, recv_right::AbstractArray,
                           dim::Int, dist::Distributor; periodic::Bool=true)
    if dist.size == 1
        return  # No communication needed for serial
    end

    # CRITICAL: Check for GPU arrays with non-CUDA-aware MPI
    # MPI operations on GPU arrays require CUDA-aware MPI.
    if (is_gpu_array(send_left) || is_gpu_array(send_right) ||
        is_gpu_array(recv_left) || is_gpu_array(recv_right)) && !check_cuda_aware_mpi()
        error("neighbor_exchange! with GPU arrays requires CUDA-aware MPI. " *
              "Set TARANG_CUDA_AWARE_MPI=1 if your MPI supports it, or copy data " *
              "to CPU before calling. TransposableField GPU operations also require " *
              "CUDA-aware MPI; implicit staging is disabled.")
    end

    start_time = time()

    # Get neighbor ranks in the specified dimension
    left_rank, right_rank = get_neighbor_ranks(dist, dim; periodic=periodic)

    # Use distinct tags for left and right to avoid message matching issues
    # when left_rank == right_rank (e.g., 2-process periodic mesh)
    # Tag encoding: dim * 10 + direction (0=left, 1=right)
    tag_left = dim * 10 + 0
    tag_right = dim * 10 + 1

    # Use non-blocking sends for overlap
    reqs = MPI.Request[]

    if left_rank >= 0
        # Send to left, receive from left (they send right to us)
        push!(reqs, MPI.Isend(send_left, dist.comm; dest=left_rank, tag=tag_left))
        push!(reqs, MPI.Irecv!(recv_left, dist.comm; source=left_rank, tag=tag_right))
    end

    if right_rank >= 0
        # Send to right, receive from right (they send left to us)
        push!(reqs, MPI.Isend(send_right, dist.comm; dest=right_rank, tag=tag_right))
        push!(reqs, MPI.Irecv!(recv_right, dist.comm; source=right_rank, tag=tag_left))
    end

    # Wait for all operations to complete
    MPI.Waitall(reqs)

    dist.performance_stats.communication_time += time() - start_time
    dist.performance_stats.mpi_operations += length(reqs)
end

"""
    async_neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                             recv_left::AbstractArray, recv_right::AbstractArray,
                             dim::Int, dist::Distributor; periodic::Bool=true)

OPTIMIZED: Non-blocking neighbor exchange for computation/communication overlap.
Returns MPI requests that can be waited on later with wait_neighbor_exchange!.

# Arguments
- `send_left/right`: Data to send to left/right neighbor
- `recv_left/right`: Buffers for received data
- `dim`: Mesh dimension for neighbor identification
- `dist`: Distributor with MPI info
- `periodic`: If true (default), assumes periodic domain with wrap-around neighbors.
              If false, processes at domain boundaries skip communication on that side.

Usage pattern for overlap:
    reqs = async_neighbor_exchange!(...)  # Start communication
    compute_interior!(data)                # Compute while communicating
    wait_neighbor_exchange!(reqs, dist)    # Wait for boundary data
    compute_boundary!(data)                # Process boundary using received data
"""
function async_neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                                  recv_left::AbstractArray, recv_right::AbstractArray,
                                  dim::Int, dist::Distributor; periodic::Bool=true)
    if dist.size == 1
        return MPI.Request[]  # No communication needed for serial
    end

    # CRITICAL: Check for GPU arrays with non-CUDA-aware MPI
    # Async MPI operations on GPU arrays are especially problematic - the GPU pointers
    # may become invalid before the async operation completes
    if (is_gpu_array(send_left) || is_gpu_array(send_right) ||
        is_gpu_array(recv_left) || is_gpu_array(recv_right)) && !check_cuda_aware_mpi()
        error("async_neighbor_exchange! with GPU arrays requires CUDA-aware MPI. " *
              "Async operations on GPU data without CUDA-aware MPI can leave invalid " *
              "device pointers in flight. Set TARANG_CUDA_AWARE_MPI=1 if your MPI " *
              "supports it, or use synchronous neighbor_exchange! with CPU staging.")
    end

    # Get neighbor ranks in the specified dimension
    left_rank, right_rank = get_neighbor_ranks(dist, dim; periodic=periodic)

    # Use distinct tags for left and right to avoid message matching issues
    # when left_rank == right_rank (e.g., 2-process periodic mesh)
    # Tag encoding: dim * 10 + direction (0=left, 1=right)
    tag_left = dim * 10 + 0
    tag_right = dim * 10 + 1

    # Use non-blocking sends/receives
    reqs = MPI.Request[]

    if left_rank >= 0
        # Send to left, receive from left (they send right to us)
        push!(reqs, MPI.Isend(send_left, dist.comm; dest=left_rank, tag=tag_left))
        push!(reqs, MPI.Irecv!(recv_left, dist.comm; source=left_rank, tag=tag_right))
    end

    if right_rank >= 0
        # Send to right, receive from right (they send left to us)
        push!(reqs, MPI.Isend(send_right, dist.comm; dest=right_rank, tag=tag_right))
        push!(reqs, MPI.Irecv!(recv_right, dist.comm; source=right_rank, tag=tag_left))
    end

    dist.performance_stats.mpi_operations += length(reqs)
    return reqs
end

"""
    wait_neighbor_exchange!(reqs::Vector{MPI.Request}, dist::Distributor)

Wait for async neighbor exchange to complete.
"""
function wait_neighbor_exchange!(reqs::Vector{MPI.Request}, dist::Distributor)
    if isempty(reqs)
        return
    end

    start_time = time()
    MPI.Waitall(reqs)
    dist.performance_stats.communication_time += time() - start_time
end

"""
    test_neighbor_exchange(reqs::Vector{MPI.Request})

Test if any async neighbor exchange operations have completed.
Returns (completed, pending) tuple of request indices.
Useful for checking progress during computation overlap.
"""
function test_neighbor_exchange(reqs::Vector{MPI.Request})
    if isempty(reqs)
        return (Int[], Int[])
    end

    completed = Int[]
    pending = Int[]

    for (i, req) in enumerate(reqs)
        flag, _ = MPI.Test(req)
        if flag
            push!(completed, i)
        else
            push!(pending, i)
        end
    end

    return (completed, pending)
end

"""
    get_neighbor_ranks(dist::Distributor, dim::Int)

Get the MPI ranks of left and right neighbors in the specified mesh dimension.
OPTIMIZED: Uses precomputed neighbor_ranks cache for O(1) lookup.
Returns (-1, -1) if no neighbors exist (boundary processes).
"""
function get_neighbor_ranks(dist::Distributor, dim::Int; periodic::Bool=true)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return (-1, -1)
    end

    # OPTIMIZATION: Use precomputed neighbor ranks (only for periodic=true, which is the common case)
    if periodic && haskey(dist.neighbor_ranks, dim)
        return dist.neighbor_ranks[dim]
    end

    # Compute (and cache for periodic case)
    left_rank, right_rank = compute_neighbor_ranks_for_dim(dist, dim; periodic=periodic)

    # Only cache periodic neighbors (non-periodic may be requested per-call)
    if periodic
        dist.neighbor_ranks[dim] = (left_rank, right_rank)
    end

    return (left_rank, right_rank)
end

"""
    coord_to_rank(dist::Distributor, dim::Int, coord::Int)

Convert a coordinate in the given dimension to an MPI rank.
"""
function coord_to_rank(dist::Distributor, dim::Int, coord::Int)
    if dist.mesh === nothing
        return -1
    end

    # Current process coordinates in all dimensions
    current_coords = [get_process_coordinate_in_mesh(dist, i) for i in 1:length(dist.mesh)]

    # Replace the specified dimension with new coordinate
    current_coords[dim] = coord

    # Convert to rank using row-major ordering
    rank = 0
    stride = 1
    for i in 1:length(dist.mesh)
        rank += current_coords[i] * stride
        stride *= dist.mesh[i]
    end

    return rank
end

# ============================================================================
# Memory Management for Parallel Operations
# ============================================================================

# Module-level cache for communication buffers, keyed by (distributor_id, shape, dim)
const _COMM_BUFFER_CACHE = Dict{UInt64, Dict{Tuple, NamedTuple}}()

"""
    _get_distributor_id(dist::Distributor)

Get a unique identifier for a distributor instance for buffer caching.
"""
function _get_distributor_id(dist::Distributor)
    return objectid(dist)
end

"""
    preallocate_communication_buffers!(dist::Distributor, shapes::Vector{Tuple})

Preallocate communication buffers for common operations to avoid runtime allocation.

For each shape, allocates send/receive buffers for neighbor exchange operations
in all mesh dimensions. Buffers are cached and reused across multiple calls.

# Arguments
- `dist`: The Distributor managing parallel decomposition
- `shapes`: Vector of array shapes that will be communicated

# Example
```julia
# Preallocate buffers for 3D field data
shapes = [(64, 64, 1), (64, 1, 64), (1, 64, 64)]  # Boundary slices
preallocate_communication_buffers!(dist, shapes)
```
"""
function preallocate_communication_buffers!(dist::Distributor, shapes::Vector{Tuple})
    if dist.size == 1
        @debug "Serial mode: no communication buffers needed"
        return
    end

    dist_id = _get_distributor_id(dist)

    # Initialize cache for this distributor if needed
    if !haskey(_COMM_BUFFER_CACHE, dist_id)
        _COMM_BUFFER_CACHE[dist_id] = Dict{Tuple, NamedTuple}()
    end

    buffer_cache = _COMM_BUFFER_CACHE[dist_id]
    n_dims = dist.mesh !== nothing ? length(dist.mesh) : dist.dim

    for shape in shapes
        for dim in 1:n_dims
            cache_key = (shape, dim)

            if haskey(buffer_cache, cache_key)
                continue  # Already allocated
            end

            # Allocate send/receive buffers for this shape and dimension
            # Buffer type matches distributor dtype (complex for spectral methods)
            T = dist.dtype <: Real ? Complex{dist.dtype} : dist.dtype

            send_left = zeros(T, shape...)
            send_right = zeros(T, shape...)
            recv_left = zeros(T, shape...)
            recv_right = zeros(T, shape...)

            buffer_cache[cache_key] = (
                send_left = send_left,
                send_right = send_right,
                recv_left = recv_left,
                recv_right = recv_right
            )
        end
    end

    @debug "Preallocated communication buffers" n_shapes=length(shapes) n_dims=n_dims total_buffers=length(buffer_cache)
end

"""
    get_communication_buffers(dist::Distributor, shape::Tuple, dim::Int)

Retrieve preallocated communication buffers for a given shape and dimension.
Returns a NamedTuple with (send_left, send_right, recv_left, recv_right).

If buffers haven't been preallocated, allocates them on demand.
"""
function get_communication_buffers(dist::Distributor, shape::Tuple, dim::Int)
    if dist.size == 1
        # Return empty buffers for serial mode
        T = dist.dtype <: Real ? Complex{dist.dtype} : dist.dtype
        empty = zeros(T, 0)
        return (send_left=empty, send_right=empty, recv_left=empty, recv_right=empty)
    end

    dist_id = _get_distributor_id(dist)
    cache_key = (shape, dim)

    # Initialize cache if needed
    if !haskey(_COMM_BUFFER_CACHE, dist_id)
        _COMM_BUFFER_CACHE[dist_id] = Dict{Tuple, NamedTuple}()
    end

    buffer_cache = _COMM_BUFFER_CACHE[dist_id]

    # Allocate on demand if not preallocated
    if !haskey(buffer_cache, cache_key)
        preallocate_communication_buffers!(dist, [shape])
    end

    return buffer_cache[cache_key]
end

"""
    clear_communication_buffers!(dist::Distributor)

Clear all preallocated communication buffers for a distributor.
Useful for freeing memory when buffers are no longer needed.
"""
function clear_communication_buffers!(dist::Distributor)
    dist_id = _get_distributor_id(dist)
    if haskey(_COMM_BUFFER_CACHE, dist_id)
        delete!(_COMM_BUFFER_CACHE, dist_id)
        @debug "Cleared communication buffers for distributor"
    end
end

"""
    get_optimal_chunk_size(total_size::Int, num_procs::Int)

Compute optimal chunk size for load-balanced distribution.
Handles remainders by giving extra work to first few processes.
"""
function get_optimal_chunk_size(total_size::Int, num_procs::Int)
    base_size = div(total_size, num_procs)
    remainder = total_size % num_procs
    return (base_size, remainder)
end

# ============================================================================
# Performance Diagnostics
# ============================================================================

"""
    diagnose_parallel_performance(dist::Distributor)

Print detailed performance diagnostics for parallel operations.
"""
function diagnose_parallel_performance(dist::Distributor)
    stats = dist.performance_stats

    if dist.rank == 0
        println("\n" * "="^60)
        println("Parallel Performance Diagnostics")
        println("="^60)
        println("MPI Configuration:")
        println("  Processes: $(dist.size)")
        println("  Mesh: $(dist.mesh)")
        println("  Rank 0 coordinates: $(ntuple(i -> get_process_coordinate_in_mesh(dist, i), length(dist.mesh)))")
        println()
        println("Timing Statistics:")
        println("  Total distributor time: $(round(stats.total_time, digits=4))s")
        println("  Transpose time: $(round(stats.transpose_time, digits=4))s")
        println("  Communication time: $(round(stats.communication_time, digits=4))s")
        println()
        println("Operation Counts:")
        println("  Pencil creations: $(stats.pencil_creations)")
        println("  Layout creations: $(stats.layout_creations)")
        println("  MPI operations: $(stats.mpi_operations)")
        println()
        println("Cache Performance:")
        total_cache = stats.cache_hits + stats.cache_misses
        hit_rate = total_cache > 0 ? 100.0 * stats.cache_hits / total_cache : 0.0
        println("  Cache hits: $(stats.cache_hits)")
        println("  Cache misses: $(stats.cache_misses)")
        println("  Hit rate: $(round(hit_rate, digits=1))%")
        println()
        println("Memory:")
        println("  Cached layouts: $(length(dist.layouts))")
        println("  Cached pencils: $(length(dist.pencil_cache))")
        println("="^60)
    end

    # Synchronize to ensure clean output
    MPI.Barrier(dist.comm)
end

"""
    reset_performance_stats!(dist::Distributor)

Reset all performance statistics counters.
"""
function reset_performance_stats!(dist::Distributor)
    stats = dist.performance_stats
    stats.total_time = 0.0
    stats.pencil_creations = 0
    stats.layout_creations = 0
    stats.mpi_operations = 0
    stats.cache_hits = 0
    stats.cache_misses = 0
    stats.transpose_time = 0.0
    stats.communication_time = 0.0
end

# ============================================================================
