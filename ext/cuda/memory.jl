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
        # Ensure allocation on correct device
        prev_device = CUDA.device()
        CUDA.device!(CuDevice(device_id))
        arr = CUDA.zeros(T, dims...)
        CUDA.device!(prev_device)
        return arr
    end

    total_size = prod(dims)
    key = _pool_key(device_id, T, total_size)

    if haskey(GPU_MEMORY_POOL.pools, key) && !isempty(GPU_MEMORY_POOL.pools[key])
        # Reuse from pool - verify device matches
        arr = pop!(GPU_MEMORY_POOL.pools[key])

        # Safety check: ensure array is on expected device
        if CUDA.deviceid(CUDA.device(arr)) != device_id
            # Wrong device - don't use, allocate new on correct device
            push!(GPU_MEMORY_POOL.pools[key], arr)  # Return to pool
            GPU_MEMORY_POOL.misses += 1
            prev_device = CUDA.device()
            CUDA.device!(CuDevice(device_id))
            new_arr = CUDA.zeros(T, dims...)
            CUDA.device!(prev_device)
            return new_arr
        end

        GPU_MEMORY_POOL.hits += 1
        GPU_MEMORY_POOL.total_pooled_bytes -= sizeof(T) * total_size

        # Switch to the array's device for reshape/fill! in multi-GPU setups
        prev_device = CUDA.device()
        CUDA.device!(CuDevice(device_id))

        # Reshape if needed and zero out
        if size(arr) != dims
            arr = reshape(arr, dims...)
        end
        fill!(arr, zero(T))

        CUDA.device!(prev_device)
        return arr
    else
        # Allocate new on correct device
        GPU_MEMORY_POOL.misses += 1
        prev_device = CUDA.device()
        CUDA.device!(CuDevice(device_id))
        arr = CUDA.zeros(T, dims...)
        CUDA.device!(prev_device)
        return arr
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
Uses checkout/return semantics to prevent concurrent access to the same buffer.
Tracks total pooled bytes and caps growth to prevent pinned memory leaks.
"""
mutable struct PinnedBufferPool
    # Available (not in-use) buffers per (type, shape) key
    available::Dict{Tuple{DataType, Tuple}, Vector{Vector{UInt8}}}
    max_buffers_per_key::Int
    # Track total pinned bytes to cap memory growth
    total_pooled_bytes::Int
    max_total_bytes::Int  # Cap total pinned memory (default 1 GB)
    # Track checked-out buffers by pointer address → (backing buffer, byte size)
    # This prevents permanent leaks if callers forget release_pinned_buffer!
    in_use::Dict{UInt, Tuple{Vector{UInt8}, Int}}

    function PinnedBufferPool(; max_buffers::Int=5, max_total_mb::Int=1024)
        new(Dict{Tuple{DataType, Tuple}, Vector{Vector{UInt8}}}(),
            max_buffers, 0, max_total_mb * 1_000_000,
            Dict{UInt, Tuple{Vector{UInt8}, Int}}())
    end
end

const PINNED_BUFFER_POOL = PinnedBufferPool()

"""
    get_pinned_buffer(T::Type, shape::Tuple)

Check out a pinned memory buffer for fast GPU transfers.
Each call returns a unique buffer (not shared with other callers).
Call `release_pinned_buffer!` when done to return it to the pool.

If no buffer is available in the pool, a new one is allocated.
"""
function get_pinned_buffer(T::Type, shape::Tuple)
    key = (T, shape)
    total_bytes = sizeof(T) * prod(shape)

    if haskey(PINNED_BUFFER_POOL.available, key) && !isempty(PINNED_BUFFER_POOL.available[key])
        # Check out an available buffer from the pool
        buf = pop!(PINNED_BUFFER_POOL.available[key])
        PINNED_BUFFER_POOL.total_pooled_bytes -= total_bytes
    else
        # Allocate a new pinned buffer
        pinned_ptr = CUDA.Mem.alloc(CUDA.Mem.Host, total_bytes)
        buf = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(pinned_ptr), total_bytes)
    end

    # Track the checked-out buffer so it can be reclaimed if caller forgets release
    ptr_addr = UInt(pointer(buf))
    PINNED_BUFFER_POOL.in_use[ptr_addr] = (buf, total_bytes)

    return unsafe_wrap(Array{T}, Ptr{T}(pointer(buf)), shape)
end

"""
    release_pinned_buffer!(arr::Array{T}, shape::Tuple) where T

Return a pinned buffer to the pool for reuse.
The caller must ensure all async operations using this buffer have completed
(e.g., by calling `CUDA.synchronize(stream)`) before releasing.

If the pool is full or at its memory cap, the pinned memory is explicitly freed.
"""
function release_pinned_buffer!(arr::Array{T}, shape::Tuple) where T
    key = (T, shape)
    total_bytes = sizeof(T) * prod(shape)
    ptr_addr = UInt(pointer(arr))

    # Remove from in-use tracking; retrieve the backing buffer
    tracked = pop!(PINNED_BUFFER_POOL.in_use, ptr_addr, nothing)

    if tracked === nothing
        # Not allocated by get_pinned_buffer, or already released (double-release).
        # Do NOT free — the pointer may be a regular Array or already-freed memory.
        @warn "release_pinned_buffer! called on untracked pointer (not from get_pinned_buffer, or double-release)" maxlog=1
        return
    end

    if !haskey(PINNED_BUFFER_POOL.available, key)
        PINNED_BUFFER_POOL.available[key] = Vector{Vector{UInt8}}()
    end

    # Only pool if under both per-key limit and total memory cap
    can_pool = length(PINNED_BUFFER_POOL.available[key]) < PINNED_BUFFER_POOL.max_buffers_per_key &&
               PINNED_BUFFER_POOL.total_pooled_bytes + total_bytes <= PINNED_BUFFER_POOL.max_total_bytes

    if can_pool
        push!(PINNED_BUFFER_POOL.available[key], tracked[1])
        PINNED_BUFFER_POOL.total_pooled_bytes += total_bytes
    else
        # Explicitly free the pinned memory to prevent leaks
        CUDA.Mem.free(CUDA.Mem.Host, pointer(tracked[1]), total_bytes)
    end
end

"""
    clear_pinned_buffer_pool!()

Free all pinned memory in the pool. Call this to reclaim host memory.
"""
function clear_pinned_buffer_pool!()
    # Free available (pooled) buffers
    for (key, buffers) in PINNED_BUFFER_POOL.available
        for buf in buffers
            total_bytes = length(buf)
            CUDA.Mem.free(CUDA.Mem.Host, pointer(buf), total_bytes)
        end
        empty!(buffers)
    end
    empty!(PINNED_BUFFER_POOL.available)
    PINNED_BUFFER_POOL.total_pooled_bytes = 0

    # Also free any in-use (leaked) buffers
    reclaim_leaked_pinned_buffers!()
end

"""
    reclaim_leaked_pinned_buffers!()

Free all pinned buffers that were checked out but never released.
Call this to reclaim host memory leaked by callers who forgot `release_pinned_buffer!`.

Returns the number of leaked buffers that were freed.
"""
function reclaim_leaked_pinned_buffers!()
    n_leaked = length(PINNED_BUFFER_POOL.in_use)
    if n_leaked > 0
        for (ptr_addr, (buf, total_bytes)) in PINNED_BUFFER_POOL.in_use
            CUDA.Mem.free(CUDA.Mem.Host, pointer(buf), total_bytes)
        end
        @warn "Freed $n_leaked leaked pinned buffer(s)" maxlog=1
    end
    empty!(PINNED_BUFFER_POOL.in_use)
    return n_leaked
end

"""
    async_copy_to_gpu!(dst::CuArray, src::Array; stream=nothing)

Copy data from CPU to GPU using the specified stream, with automatic synchronization.

This function always synchronizes after the copy to ensure data integrity.
The copy uses a dedicated transfer stream (separate from compute) to enable
overlap with GPU computation when properly orchestrated.

For best performance, use pinned (page-locked) memory for `src` via `get_pinned_buffer`.

# Arguments
- `dst`: Destination GPU array
- `src`: Source CPU array
- `stream`: Optional CUDA stream (defaults to the transfer stream for dst's device)

# Example
```julia
# Basic usage - always safe
async_copy_to_gpu!(gpu_array, cpu_array)

# With pinned memory for better performance
pinned_src = get_pinned_buffer(Float64, size(cpu_array))
copyto!(pinned_src, cpu_array)
async_copy_to_gpu!(gpu_array, pinned_src)
release_pinned_buffer!(pinned_src, size(cpu_array))
```
"""
function async_copy_to_gpu!(dst::CuArray{T}, src::Array{T}; stream=nothing) where T
    # Use the device of the destination array for stream selection
    dst_device = CUDA.device(dst)
    device_id = CUDA.deviceid(dst_device)
    s = stream !== nothing ? stream : get_transfer_stream(; device_id=device_id)

    if s !== nothing
        prev_device = CUDA.device()
        CUDA.device!(dst_device)
        CUDA.stream!(s) do
            copyto!(dst, src)
        end
        # Always synchronize for safety - prevents race conditions
        CUDA.synchronize(s)
        CUDA.device!(prev_device)
    else
        copyto!(dst, src)
    end
    return dst
end

"""
    async_copy_to_cpu!(dst::Array, src::CuArray; stream=nothing)

Copy data from GPU to CPU using the specified stream, with automatic synchronization.

This function always synchronizes after the copy to ensure data integrity.
The copy uses a dedicated transfer stream (separate from compute) to enable
overlap with GPU computation when properly orchestrated.

For best performance, use pinned (page-locked) memory for `dst` via `get_pinned_buffer`.

# Arguments
- `dst`: Destination CPU array
- `src`: Source GPU array
- `stream`: Optional CUDA stream (defaults to the transfer stream for src's device)

# Example
```julia
# Basic usage - always safe
async_copy_to_cpu!(cpu_array, gpu_array)

# With pinned memory for better performance
pinned_dst = get_pinned_buffer(Float64, size(gpu_array))
async_copy_to_cpu!(pinned_dst, gpu_array)
copyto!(cpu_array, pinned_dst)
release_pinned_buffer!(pinned_dst, size(gpu_array))
```
"""
function async_copy_to_cpu!(dst::Array{T}, src::CuArray{T}; stream=nothing) where T
    # Use the device of the source array for stream selection
    src_device = CUDA.device(src)
    device_id = CUDA.deviceid(src_device)
    s = stream !== nothing ? stream : get_transfer_stream(; device_id=device_id)

    if s !== nothing
        prev_device = CUDA.device()
        CUDA.device!(src_device)
        CUDA.stream!(s) do
            copyto!(dst, src)
        end
        # Always synchronize for safety - prevents race conditions
        CUDA.synchronize(s)
        CUDA.device!(prev_device)
    else
        copyto!(dst, src)
    end
    return dst
end
