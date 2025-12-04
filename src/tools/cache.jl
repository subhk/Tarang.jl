"""
Caching utilities

Simplified version of dedalus caching system
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
struct CachedClass
    instances::WeakKeyDict{Tuple, Any}
    
    function CachedClass()
        new(WeakKeyDict{Tuple, Any}())
    end
end

const cached_class_instances = WeakKeyDict{Type, CachedClass}()

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
function cache_info()
    """Return information about cache usage"""
    method_caches = length(function_cache)
    class_caches = sum(length(cc.instances) for cc in values(cached_class_instances))
    
    return Dict(
        "method_cache_size" => method_caches,
        "class_cache_size" => class_caches,
        "total_cached_objects" => method_caches + class_caches
    )
end

function cleanup_weak_references!()
    """Clean up weak references in caches"""
    # This would be called periodically to clean up dead references
    # Julia's GC handles most of this automatically
    GC.gc()
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
mutable struct LRUCache{K, V}
    capacity::Int
    cache::Dict{K, V}
    order::Vector{K}
    
    function LRUCache{K, V}(capacity::Int) where {K, V}
        new{K, V}(capacity, Dict{K, V}(), K[])
    end
end

function LRUCache(capacity::Int)
    LRUCache{Any, Any}(capacity)
end

function Base.get!(lru::LRUCache{K, V}, key::K, default_func::Function) where {K, V}
    if haskey(lru.cache, key)
        # Move to end (most recently used)
        filter!(x -> x != key, lru.order)
        push!(lru.order, key)
        return lru.cache[key]
    else
        # Add new item
        value = default_func()
        
        # Check capacity
        if length(lru.order) >= lru.capacity
            # Remove least recently used
            old_key = popfirst!(lru.order)
            delete!(lru.cache, old_key)
        end
        
        lru.cache[key] = value
        push!(lru.order, key)
        return value
    end
end

function Base.getindex(lru::LRUCache{K, V}, key::K) where {K, V}
    if haskey(lru.cache, key)
        # Move to end
        filter!(x -> x != key, lru.order)
        push!(lru.order, key)
        return lru.cache[key]
    else
        throw(KeyError(key))
    end
end

function Base.setindex!(lru::LRUCache{K, V}, value::V, key::K) where {K, V}
    if haskey(lru.cache, key)
        lru.cache[key] = value
        # Move to end
        filter!(x -> x != key, lru.order)
        push!(lru.order, key)
    else
        # Check capacity
        if length(lru.order) >= lru.capacity
            # Remove least recently used
            old_key = popfirst!(lru.order)
            delete!(lru.cache, old_key)
        end
        
        lru.cache[key] = value
        push!(lru.order, key)
    end
end