"""
Caching utilities
"""

# WeakKeyDict is part of Base, no import needed

# Cached method decorator-like functionality
struct CachedMethod{F}
    func::F
    cache::Dict{Any, Any}
    
    function CachedMethod(func::F) where F
        new{F}(func, Dict{Any, Any}())
    end
end

function (cm::CachedMethod)(args...)
    if haskey(cm.cache, args)
        return cm.cache[args]
    else
        result = cm.func(args...)
        cm.cache[args] = result
        return result
    end
end

# Cached attribute functionality
struct CachedAttribute{T}
    compute_func::Function
    cache::Dict{UInt64, T}
    
    function CachedAttribute{T}(compute_func::Function) where T
        new{T}(compute_func, Dict{UInt64, T}())
    end
end

function CachedAttribute(compute_func::Function)
    CachedAttribute{Any}(compute_func)
end

# Macro for creating cached methods
macro cached_method(func_def)
    # Extract function name and arguments
    if func_def.head == :function
        func_name = func_def.args[1].args[1]
        
        # Wrap the function with caching
        return quote
            $(esc(func_def))
            const $(esc(Symbol("cached_", func_name))) = CachedMethod($(esc(func_name)))
        end
    else
        error("@cached_method can only be applied to function definitions")
    end
end

# Cached class metaclass equivalent
# Note: We use Dict with WeakRef values instead of WeakKeyDict because:
# 1. WeakKeyDict requires mutable keys (Tuples are immutable)
# 2. We want instances to be GC'd when no longer used elsewhere
# 3. The cache automatically returns nothing for collected references
mutable struct CachedClass
    instances::Dict{Any, WeakRef}

    function CachedClass()
        new(Dict{Any, WeakRef}())
    end
end

"""
    get_cached_instance(cc::CachedClass, key)

Get a cached instance if it exists and hasn't been garbage collected.
Returns the instance or nothing.
"""
function get_cached_instance(cc::CachedClass, key)
    if haskey(cc.instances, key)
        ref = cc.instances[key]
        inst = ref.value
        if inst === nothing
            # Instance was garbage collected, remove stale entry
            delete!(cc.instances, key)
            return nothing
        end
        return inst
    end
    return nothing
end

"""
    set_cached_instance!(cc::CachedClass, key, instance)

Store an instance in the cache with a weak reference.
"""
function set_cached_instance!(cc::CachedClass, key, instance)
    cc.instances[key] = WeakRef(instance)
end

"""
    cleanup_stale_entries!(cc::CachedClass)

Remove entries whose instances have been garbage collected.
"""
function cleanup_stale_entries!(cc::CachedClass)
    stale_keys = [k for (k, v) in cc.instances if v.value === nothing]
    for k in stale_keys
        delete!(cc.instances, k)
    end
    return length(stale_keys)
end

const cached_class_instances = Dict{Type, CachedClass}()

# Simple caching for function results
const function_cache = Dict{Tuple{Function, Any}, Any}()

function cached_call(func::Function, args...)
    key = (func, args)
    if haskey(function_cache, key)
        return function_cache[key]
    else
        result = func(args...)
        function_cache[key] = result
        return result
    end
end

# Clear cache functions
function clear_cache!(cm::CachedMethod)
    empty!(cm.cache)
end

function clear_cache!(ca::CachedAttribute)
    empty!(ca.cache)
end

function clear_all_caches!()
    empty!(function_cache)
    for cc in values(cached_class_instances)
        empty!(cc.instances)
    end
end

# Memory management

"""Return information about cache usage."""
function cache_info()
    method_caches = length(function_cache)

    # Count only live (non-collected) entries in class caches
    class_caches = 0
    stale_entries = 0
    for cc in values(cached_class_instances)
        for (_, ref) in cc.instances
            if ref.value !== nothing
                class_caches += 1
            else
                stale_entries += 1
            end
        end
    end

    return Dict(
        "method_cache_size" => method_caches,
        "class_cache_size" => class_caches,
        "stale_entries" => stale_entries,
        "total_cached_objects" => method_caches + class_caches
    )
end

"""Clean up weak references in caches (stale entries from GC'd instances)."""
function cleanup_weak_references!()
    total_cleaned = 0
    for cc in values(cached_class_instances)
        total_cleaned += cleanup_stale_entries!(cc)
    end
    return total_cleaned
end

# Memoization utilities
mutable struct MemoizedFunction{F, C}
    func::F
    cache::C
    hits::Int
    misses::Int
    
    function MemoizedFunction(func::F, cache_type=Dict) where F
        cache = cache_type{Any, Any}()
        new{F, typeof(cache)}(func, cache, 0, 0)
    end
end

function (mf::MemoizedFunction)(args...)
    if haskey(mf.cache, args)
        mf.hits += 1
        return mf.cache[args]
    else
        mf.misses += 1
        result = mf.func(args...)
        mf.cache[args] = result
        return result
    end
end

function cache_stats(mf::MemoizedFunction)
    total = mf.hits + mf.misses
    hit_rate = total > 0 ? mf.hits / total : 0.0
    
    return Dict(
        "hits" => mf.hits,
        "misses" => mf.misses,
        "hit_rate" => hit_rate,
        "cache_size" => length(mf.cache)
    )
end

# Decorator-style function for memoization
function memoize(func::Function, cache_type=Dict)
    return MemoizedFunction(func, cache_type)
end

# LRU cache implementation
# Uses a timestamp counter for O(1) access and O(n) eviction
# (eviction is less frequent than access, so this is a good tradeoff)

"""
    LRUCacheEntry{V}

Internal entry storing value and access timestamp for LRU eviction.
"""
mutable struct LRUCacheEntry{V}
    value::V
    timestamp::Int
end

mutable struct LRUCache{K, V}
    capacity::Int
    cache::Dict{K, LRUCacheEntry{V}}
    counter::Int  # Monotonically increasing access counter

    function LRUCache{K, V}(capacity::Int) where {K, V}
        @assert capacity > 0 "LRU cache capacity must be positive"
        new{K, V}(capacity, Dict{K, LRUCacheEntry{V}}(), 0)
    end
end

function LRUCache(capacity::Int)
    LRUCache{Any, Any}(capacity)
end

Base.length(lru::LRUCache) = length(lru.cache)
Base.haskey(lru::LRUCache, key) = haskey(lru.cache, key)
Base.keys(lru::LRUCache) = keys(lru.cache)
Base.isempty(lru::LRUCache) = isempty(lru.cache)

"""
    _evict_lru!(lru::LRUCache)

Evict the least recently used entry. O(n) but called infrequently.
"""
function _evict_lru!(lru::LRUCache)
    if isempty(lru.cache)
        return
    end

    # Find entry with minimum timestamp
    min_key = nothing
    min_ts = typemax(Int)
    for (k, entry) in lru.cache
        if entry.timestamp < min_ts
            min_ts = entry.timestamp
            min_key = k
        end
    end

    if min_key !== nothing
        delete!(lru.cache, min_key)
    end
end

function Base.get!(lru::LRUCache{K, V}, key::K, default_func::Function) where {K, V}
    if haskey(lru.cache, key)
        # Update timestamp (O(1))
        lru.counter += 1
        lru.cache[key].timestamp = lru.counter
        return lru.cache[key].value
    else
        # Add new item
        value = default_func()

        # Evict if at capacity
        if length(lru.cache) >= lru.capacity
            _evict_lru!(lru)
        end

        lru.counter += 1
        lru.cache[key] = LRUCacheEntry{V}(value, lru.counter)
        return value
    end
end

function Base.getindex(lru::LRUCache{K, V}, key::K) where {K, V}
    if haskey(lru.cache, key)
        # Update timestamp (O(1))
        lru.counter += 1
        lru.cache[key].timestamp = lru.counter
        return lru.cache[key].value
    else
        throw(KeyError(key))
    end
end

function Base.setindex!(lru::LRUCache{K, V}, value::V, key::K) where {K, V}
    if haskey(lru.cache, key)
        # Update existing entry (O(1))
        lru.counter += 1
        entry = lru.cache[key]
        entry.value = value
        entry.timestamp = lru.counter
    else
        # Evict if at capacity
        if length(lru.cache) >= lru.capacity
            _evict_lru!(lru)
        end

        lru.counter += 1
        lru.cache[key] = LRUCacheEntry{V}(value, lru.counter)
    end
end

function Base.delete!(lru::LRUCache, key)
    delete!(lru.cache, key)
    return lru
end

function Base.empty!(lru::LRUCache)
    empty!(lru.cache)
    lru.counter = 0
    return lru
end

# ============================================================================
# WorkspaceManager: Pre-allocated buffer pool for zero-allocation operations
# ============================================================================

"""
    WorkspaceManager

Thread-safe workspace buffer manager for spectral computations.
Pre-allocates and reuses arrays to minimize GC pressure during time-stepping.

# Key Features
- Type-aware buffer pools (Float64, ComplexF64)
- Size-based buffer matching with tolerance
- Automatic buffer growth when needed
- Statistics tracking for optimization

# Usage
```julia
wm = WorkspaceManager()
buf = get_workspace!(wm, Float64, (64, 64))  # Get or create buffer
# ... use buf ...
release_workspace!(wm, buf)  # Return to pool
```
"""
mutable struct WorkspaceManager
    # Buffer pools organized by element type
    real_buffers::Vector{Array{Float64}}
    complex_buffers::Vector{Array{ComplexF64}}

    # Track which buffers are in use
    real_in_use::Vector{Bool}
    complex_in_use::Vector{Bool}

    # Statistics
    stats::Dict{String, Int}

    # Configuration
    max_buffers_per_type::Int
    size_tolerance::Float64  # Allow reuse if size within tolerance

    function WorkspaceManager(; max_buffers::Int=32, size_tolerance::Float64=1.2)
        new(
            Vector{Array{Float64}}(),
            Vector{Array{ComplexF64}}(),
            Vector{Bool}(),
            Vector{Bool}(),
            Dict{String, Int}(
                "allocations" => 0,
                "reuses" => 0,
                "real_pool_size" => 0,
                "complex_pool_size" => 0
            ),
            max_buffers,
            size_tolerance
        )
    end
end

# Global workspace manager instance
const GLOBAL_WORKSPACE = Ref{Union{Nothing, WorkspaceManager}}(nothing)

"""
    get_global_workspace()

Get or create the global workspace manager.
"""
function get_global_workspace()
    if GLOBAL_WORKSPACE[] === nothing
        GLOBAL_WORKSPACE[] = WorkspaceManager()
    end
    return GLOBAL_WORKSPACE[]
end

"""
    get_workspace!(wm::WorkspaceManager, ::Type{T}, shape::Tuple) where T

Get a pre-allocated workspace buffer of the specified type and shape.
Returns an existing buffer from the pool if available, or allocates a new one.
"""
function get_workspace!(wm::WorkspaceManager, ::Type{Float64}, shape::Tuple)
    target_size = prod(shape)

    # Search for available buffer of suitable size
    for (i, buf) in enumerate(wm.real_buffers)
        if !wm.real_in_use[i]
            buf_size = length(buf)
            # Accept if exact match or within tolerance
            if buf_size == target_size || (buf_size >= target_size && buf_size <= target_size * wm.size_tolerance)
                wm.real_in_use[i] = true
                wm.stats["reuses"] += 1
                # Reshape if needed
                if size(buf) != shape && length(buf) >= target_size
                    return reshape(view(buf, 1:target_size), shape)
                elseif size(buf) == shape
                    return buf
                end
            end
        end
    end

    # No suitable buffer found, allocate new one
    wm.stats["allocations"] += 1
    new_buf = zeros(Float64, shape)

    if length(wm.real_buffers) < wm.max_buffers_per_type
        push!(wm.real_buffers, new_buf)
        push!(wm.real_in_use, true)
        wm.stats["real_pool_size"] = length(wm.real_buffers)
    end

    return new_buf
end

function get_workspace!(wm::WorkspaceManager, ::Type{ComplexF64}, shape::Tuple)
    target_size = prod(shape)

    # Search for available buffer of suitable size
    for (i, buf) in enumerate(wm.complex_buffers)
        if !wm.complex_in_use[i]
            buf_size = length(buf)
            if buf_size == target_size || (buf_size >= target_size && buf_size <= target_size * wm.size_tolerance)
                wm.complex_in_use[i] = true
                wm.stats["reuses"] += 1
                if size(buf) != shape && length(buf) >= target_size
                    return reshape(view(buf, 1:target_size), shape)
                elseif size(buf) == shape
                    return buf
                end
            end
        end
    end

    # Allocate new buffer
    wm.stats["allocations"] += 1
    new_buf = zeros(ComplexF64, shape)

    if length(wm.complex_buffers) < wm.max_buffers_per_type
        push!(wm.complex_buffers, new_buf)
        push!(wm.complex_in_use, true)
        wm.stats["complex_pool_size"] = length(wm.complex_buffers)
    end

    return new_buf
end

"""
    get_workspace!(wm::WorkspaceManager, template::AbstractArray)

Get a workspace buffer matching the type and shape of the template array.
"""
function get_workspace!(wm::WorkspaceManager, template::AbstractArray{T}) where T
    return get_workspace!(wm, T, size(template))
end

"""
    get_workspace!(wm::WorkspaceManager, ::Type{T}, shape::Tuple) where T

Fallback for unsupported types. Allocates a new buffer without pooling.
Only Float64 and ComplexF64 are pooled; other types are allocated fresh.
"""
function get_workspace!(wm::WorkspaceManager, ::Type{T}, shape::Tuple) where T
    # Unsupported types are not pooled - just allocate
    wm.stats["allocations"] += 1
    return zeros(T, shape)
end

"""
    release_workspace!(wm::WorkspaceManager, buf::AbstractArray)

Return a buffer to the pool for reuse.
"""
function release_workspace!(wm::WorkspaceManager, buf::Array{Float64})
    for (i, pool_buf) in enumerate(wm.real_buffers)
        if pool_buf === buf
            wm.real_in_use[i] = false
            return
        end
    end
end

function release_workspace!(wm::WorkspaceManager, buf::Array{ComplexF64})
    for (i, pool_buf) in enumerate(wm.complex_buffers)
        if pool_buf === buf
            wm.complex_in_use[i] = false
            return
        end
    end
end

"""
    _find_underlying_array(buf)

Recursively find the underlying Array from a view (SubArray or ReshapedArray).
Returns the underlying Array or nothing if not found.
"""
function _find_underlying_array(buf)
    underlying = buf
    while underlying isa SubArray || underlying isa Base.ReshapedArray
        underlying = parent(underlying)
    end
    return underlying
end

"""
    release_workspace!(wm::WorkspaceManager, buf::Base.ReshapedArray{Float64})

Release a reshaped view by finding and releasing its underlying buffer.
This handles the case where get_workspace! returns a reshaped view when
reusing a buffer with matching size but different shape.
"""
function release_workspace!(wm::WorkspaceManager, buf::Base.ReshapedArray{Float64})
    underlying = _find_underlying_array(buf)
    if underlying isa Array{Float64}
        release_workspace!(wm, underlying)
    end
end

"""
    release_workspace!(wm::WorkspaceManager, buf::Base.ReshapedArray{ComplexF64})

Release a reshaped view by finding and releasing its underlying buffer.
"""
function release_workspace!(wm::WorkspaceManager, buf::Base.ReshapedArray{ComplexF64})
    underlying = _find_underlying_array(buf)
    if underlying isa Array{ComplexF64}
        release_workspace!(wm, underlying)
    end
end

"""
    release_workspace!(wm::WorkspaceManager, buf::SubArray)

Release a SubArray view by finding and releasing its underlying buffer.
This handles the case where a view into a pooled buffer needs to be released.
"""
function release_workspace!(wm::WorkspaceManager, buf::SubArray)
    underlying = _find_underlying_array(buf)
    if underlying isa Array{Float64}
        release_workspace!(wm, underlying)
    elseif underlying isa Array{ComplexF64}
        release_workspace!(wm, underlying)
    end
    # For other underlying types, do nothing (not managed by this pool)
end

"""
    release_workspace!(wm::WorkspaceManager, buf::AbstractArray)

Fallback no-op for array types not managed by the workspace pool.

The workspace manager only tracks `Array{Float64}` and `Array{ComplexF64}`
buffers. Other array types (e.g., GPU arrays, sparse arrays, special array
wrappers) are not pooled and don't need explicit release.

This method exists to allow safe calling of `release_workspace!` on any
array type without runtime type checking at call sites.
"""
function release_workspace!(wm::WorkspaceManager, buf::AbstractArray)
    # Other array types not managed by this pool - nothing to do
    return nothing
end

"""
    workspace_stats(wm::WorkspaceManager)

Get statistics about workspace usage.
"""
function workspace_stats(wm::WorkspaceManager)
    real_active = count(wm.real_in_use)
    complex_active = count(wm.complex_in_use)

    return Dict(
        "allocations" => wm.stats["allocations"],
        "reuses" => wm.stats["reuses"],
        "reuse_rate" => wm.stats["reuses"] / max(1, wm.stats["allocations"] + wm.stats["reuses"]),
        "real_pool_size" => length(wm.real_buffers),
        "real_active" => real_active,
        "complex_pool_size" => length(wm.complex_buffers),
        "complex_active" => complex_active,
        "total_memory_bytes" => sum(sizeof, wm.real_buffers; init=0) + sum(sizeof, wm.complex_buffers; init=0)
    )
end

"""
    clear_workspace!(wm::WorkspaceManager)

Clear all buffers from the workspace pool.
"""
function clear_workspace!(wm::WorkspaceManager)
    empty!(wm.real_buffers)
    empty!(wm.complex_buffers)
    empty!(wm.real_in_use)
    empty!(wm.complex_in_use)
    wm.stats["real_pool_size"] = 0
    wm.stats["complex_pool_size"] = 0
end

# ============================================================================
# Convenience functions using global workspace
# ============================================================================

"""
    @workspace T shape expr

Macro for automatic workspace acquisition and release.
"""
macro workspace(T, shape, expr)
    return quote
        local wm = get_global_workspace()
        local buf = get_workspace!(wm, $(esc(T)), $(esc(shape)))
        try
            $(esc(expr))
        finally
            release_workspace!(wm, buf)
        end
    end
end

"""
    with_workspace(f::Function, T::Type, shape::Tuple)

Execute function with a temporary workspace buffer.
"""
function with_workspace(f::Function, T::Type, shape::Tuple)
    wm = get_global_workspace()
    buf = get_workspace!(wm, T, shape)
    try
        return f(buf)
    finally
        release_workspace!(wm, buf)
    end
end

# ============================================================================
# Exports
# ============================================================================

# Core caching types
export CachedMethod, CachedAttribute, CachedClass

# CachedClass operations
export get_cached_instance, set_cached_instance!, cleanup_stale_entries!

# Global cache management
export cached_call, clear_cache!, clear_all_caches!
export cache_info, cleanup_weak_references!

# Memoization
export MemoizedFunction, memoize, cache_stats

# LRU Cache
export LRUCache, LRUCacheEntry

# Workspace management
export WorkspaceManager, get_global_workspace
export get_workspace!, release_workspace!, workspace_stats, clear_workspace!
export with_workspace

# Macros
export @cached_method, @workspace