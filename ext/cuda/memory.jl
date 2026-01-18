# ============================================================================
# Memory Pool Management
# ============================================================================

"""
    GPUMemoryPool

Memory pool for efficient GPU array allocation and reuse.
Reduces allocation overhead in long-running simulations.

**Multi-GPU support:** Pool keys include device ID to prevent cross-device reuse.
"""
mutable struct GPUMemoryPool
    # Pools organized by (device_id, element type, total size)
    pools::Dict{Tuple{Int, DataType, Int}, Vector{CuArray}}
    max_pool_size::Int  # Maximum arrays per pool
    total_pooled_bytes::Int
    hits::Int
    misses::Int

    function GPUMemoryPool(; max_pool_size::Int=10)
        new(Dict{Tuple{Int, DataType, Int}, Vector{CuArray}}(),
            max_pool_size, 0, 0, 0)
    end
end

const GPU_MEMORY_POOL = GPUMemoryPool()

"""
    _pool_key(device_id::Int, T::Type, total_size::Int)

Generate memory pool key including device ID for multi-GPU safety.
"""
_pool_key(device_id::Int, T::Type, total_size::Int) = (device_id, T, total_size)

"""
    pool_allocate(T::Type, dims...; device_id::Int=CUDA.deviceid())

Allocate from memory pool or create new array if pool is empty.
Uses current device by default; specify device_id for multi-GPU.
"""
function pool_allocate(T::Type, dims...; device_id::Int=CUDA.deviceid())
    if !GPU_CONFIG.use_memory_pool
        return CUDA.zeros(T, dims...)
    end

    total_size = prod(dims)
    key = _pool_key(device_id, T, total_size)

    if haskey(GPU_MEMORY_POOL.pools, key) && !isempty(GPU_MEMORY_POOL.pools[key])
        # Reuse from pool - verify device matches
        arr = pop!(GPU_MEMORY_POOL.pools[key])

        # Safety check: ensure array is on expected device
        if CUDA.deviceid(CUDA.device(arr)) != device_id
            # Wrong device - don't use, allocate new
            push!(GPU_MEMORY_POOL.pools[key], arr)  # Return to pool
            GPU_MEMORY_POOL.misses += 1
            return CUDA.zeros(T, dims...)
        end

        GPU_MEMORY_POOL.hits += 1
        GPU_MEMORY_POOL.total_pooled_bytes -= sizeof(T) * total_size

        # Reshape if needed and zero out
        if size(arr) != dims
            arr = reshape(arr, dims...)
        end
        fill!(arr, zero(T))
        return arr
    else
        # Allocate new
        GPU_MEMORY_POOL.misses += 1
        return CUDA.zeros(T, dims...)
    end
end

"""
    pool_allocate(arch::GPU{CuDevice}, T::Type, dims...)

Allocate from memory pool on the specific device.
"""
function pool_allocate(arch::GPU{CuDevice}, T::Type, dims...)
    ensure_device!(arch)
    pool_allocate(T, dims...; device_id=CUDA.deviceid(arch.device))
end

"""
    pool_release!(arr::CuArray)

Return array to memory pool for reuse.
Device ID is automatically determined from the array.
"""
function pool_release!(arr::CuArray{T}) where T
    if !GPU_CONFIG.use_memory_pool
        return
    end

    total_size = length(arr)
    device_id = CUDA.deviceid(CUDA.device(arr))
    key = _pool_key(device_id, T, total_size)

    if !haskey(GPU_MEMORY_POOL.pools, key)
        GPU_MEMORY_POOL.pools[key] = Vector{CuArray}()
    end

    if length(GPU_MEMORY_POOL.pools[key]) < GPU_MEMORY_POOL.max_pool_size
        push!(GPU_MEMORY_POOL.pools[key], arr)
        GPU_MEMORY_POOL.total_pooled_bytes += sizeof(T) * total_size
    end
    # If pool is full, let GC handle the array
end

"""
    clear_memory_pool!()

Clear all pooled GPU memory.
"""
function clear_memory_pool!()
    for (key, arrays) in GPU_MEMORY_POOL.pools
        empty!(arrays)
    end
    empty!(GPU_MEMORY_POOL.pools)
    GPU_MEMORY_POOL.total_pooled_bytes = 0
    CUDA.reclaim()
    @info "GPU memory pool cleared"
end

"""
    memory_pool_stats()

Get memory pool statistics.
"""
function memory_pool_stats()
    total_hit_rate = GPU_MEMORY_POOL.hits + GPU_MEMORY_POOL.misses > 0 ?
        GPU_MEMORY_POOL.hits / (GPU_MEMORY_POOL.hits + GPU_MEMORY_POOL.misses) : 0.0

    return (
        hits = GPU_MEMORY_POOL.hits,
        misses = GPU_MEMORY_POOL.misses,
        hit_rate = total_hit_rate,
        pooled_bytes = GPU_MEMORY_POOL.total_pooled_bytes,
        pooled_mb = GPU_MEMORY_POOL.total_pooled_bytes / 1e6,
        num_pools = length(GPU_MEMORY_POOL.pools)
    )
end

# ============================================================================
# Pinned Memory for Fast CPU-GPU Transfers
# ============================================================================

"""
    PinnedBufferPool

Pool of pinned (page-locked) CPU memory for fast GPU transfers.
"""
mutable struct PinnedBufferPool
    buffers::Dict{Tuple{DataType, Tuple}, Vector{UInt8}}  # Pinned byte arrays
    max_buffers_per_size::Int

    function PinnedBufferPool(; max_buffers::Int=5)
        new(Dict{Tuple{DataType, Tuple}, Vector{UInt8}}(), max_buffers)
    end
end

const PINNED_BUFFER_POOL = PinnedBufferPool()

"""
    get_pinned_buffer(T::Type, shape::Tuple)

Get a pinned memory buffer for fast GPU transfers.
"""
function get_pinned_buffer(T::Type, shape::Tuple)
    key = (T, shape)

    if haskey(PINNED_BUFFER_POOL.buffers, key)
        # Return existing pinned buffer wrapped as array
        buf = PINNED_BUFFER_POOL.buffers[key]
        return unsafe_wrap(Array{T}, Ptr{T}(pointer(buf)), shape)
    else
        # Create new pinned buffer
        total_bytes = sizeof(T) * prod(shape)
        pinned_ptr = CUDA.Mem.alloc(CUDA.Mem.Host, total_bytes)
        buf = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(pinned_ptr), total_bytes)
        PINNED_BUFFER_POOL.buffers[key] = buf
        return unsafe_wrap(Array{T}, Ptr{T}(pointer(buf)), shape)
    end
end

"""
    async_copy_to_gpu!(dst::CuArray, src::Array; stream=nothing, synchronize=false)

Asynchronously copy data from CPU to GPU on the specified stream.
The copy is truly asynchronous - call CUDA.synchronize(stream) to wait for completion.

For best performance, use pinned (page-locked) memory for `src` via `get_pinned_buffer`.
"""
function async_copy_to_gpu!(dst::CuArray{T}, src::Array{T}; stream=nothing, synchronize::Bool=false) where T
    s = stream !== nothing ? stream : get_transfer_stream()

    if s !== nothing
        # Execute copy on the specified stream using stream context
        CUDA.stream!(s) do
            copyto!(dst, src)
        end
        # Optionally wait for completion
        if synchronize
            CUDA.synchronize(s)
        end
    else
        # No stream specified, use default (synchronous)
        copyto!(dst, src)
    end
    return dst
end

"""
    async_copy_to_cpu!(dst::Array, src::CuArray; stream=nothing, synchronize=false)

Asynchronously copy data from GPU to CPU on the specified stream.
The copy is truly asynchronous - call CUDA.synchronize(stream) to wait for completion.

For best performance, use pinned (page-locked) memory for `dst` via `get_pinned_buffer`.
"""
function async_copy_to_cpu!(dst::Array{T}, src::CuArray{T}; stream=nothing, synchronize::Bool=false) where T
    s = stream !== nothing ? stream : get_transfer_stream()

    if s !== nothing
        # Execute copy on the specified stream using stream context
        CUDA.stream!(s) do
            copyto!(dst, src)
        end
        # Optionally wait for completion
        if synchronize
            CUDA.synchronize(s)
        end
    else
        # No stream specified, use default (synchronous)
        copyto!(dst, src)
    end
    return dst
end
