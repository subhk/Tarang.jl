# Pencil transpose helpers and transpose-buffer cache.

# ============================================================================
# Pencil Transpose Support for Multi-dimensional FFTs
# Following PencilArrays transpose API for efficient parallel transforms
# ============================================================================

"""
    create_transpose_pencil(dist::Distributor, source_pencil::PencilArrays.Pencil, new_decomp_dims::Tuple)

Create a new Pencil with different decomposition dimensions for transpose operations.
This is essential for multi-dimensional FFTs where we need to change which dimensions
are local vs distributed.

# Arguments
- `dist`: Distributor with MPI configuration
- `source_pencil`: Original Pencil configuration
- `new_decomp_dims`: New dimensions to decompose (e.g., (1, 3) instead of (2, 3))

# Returns
- New Pencil object with the specified decomposition
"""
function create_transpose_pencil(dist::Distributor, source_pencil::PencilArrays.Pencil,
                                 new_decomp_dims::Tuple)
    if dist.size == 1
        return source_pencil  # No transpose needed for serial
    end

    try
        # Create new Pencil with different decomposition using PencilArrays API
        # Pencil(pen; decomp_dims=new_dims) shares memory buffers with original
        new_pencil = PencilArrays.Pencil(source_pencil; decomp_dims=new_decomp_dims)
        return new_pencil
    catch e
        # CRITICAL: Transpose pencils are required for correct distributed FFT operation.
        # Returning the original pencil would silently produce incorrect results.
        @error "Failed to create transpose pencil" exception=e new_decomp_dims=new_decomp_dims
        error("Failed to create transpose pencil with decomposition $new_decomp_dims. " *
              "This is required for correct distributed FFT operation with $(dist.size) processes. " *
              "Please check your PencilArrays installation or use serial execution.")
    end
end

"""
    transpose_pencil_data!(dest::PencilArrays.PencilArray, src::PencilArrays.PencilArray)

Perform MPI transpose operation between two PencilArrays with different decompositions.
Uses PencilArrays' optimized transpose! function.

# Arguments
- `dest`: Destination PencilArray (different decomposition than src)
- `src`: Source PencilArray

# Note
This is a key operation for multi-dimensional FFTs:
1. FFT along local dimension
2. Transpose to make another dimension local
3. FFT along new local dimension
4. Transpose back
"""
function transpose_pencil_data!(dest::PencilArrays.PencilArray, src::PencilArrays.PencilArray,
                                dist::Distributor)
    start_time = time()

    try
        # Use PencilArrays transpose! for optimized MPI communication
        PencilArrays.transpose!(dest, src)

        dist.performance_stats.transpose_time += time() - start_time
        dist.performance_stats.mpi_operations += 1
    catch e
        # CRITICAL: MPI transpose is essential for correct parallel FFT operation
        # A simple copy would produce incorrect results
        @error "PencilArrays transpose failed" exception=e
        error("MPI transpose operation failed with $(dist.size) processes. " *
              "This operation is essential for correct parallel FFT computation. " *
              "Please check your PencilArrays/MPI installation or use serial execution.")
    end

    return dest
end

# ============================================================================
# Transpose Buffer Cache for Zero-Allocation Transforms
# ============================================================================

"""
    TransposeBufferCache

Pre-allocated buffer cache for transpose operations during spectral transforms.
Avoids repeated allocations during time-stepping loop.
"""
mutable struct TransposeBufferCache
    # Cached PencilArrays for different configurations
    pencil_buffers::Dict{Tuple, PencilArrays.PencilArray}
    # Statistics
    hits::Int
    misses::Int

    function TransposeBufferCache()
        new(Dict{Tuple, PencilArrays.PencilArray}(), 0, 0)
    end
end

# Global transpose buffer cache (lazy initialization)
const TRANSPOSE_CACHE = Ref{Union{Nothing, TransposeBufferCache}}(nothing)

# Hard cap on distinct cached transpose buffers, mirroring `_SP_MEMO_CAP` (=256) in
# subsystems/subproblem_types.jl. The `:solve`/`:coupled_dct` keys are objectid- or
# shape-based; on the interpreted-RHS fallback (churning field objectids) the Dict would
# otherwise grow without bound — each miss allocs a full solve-pencil PencilArray and
# keeps it forever. On overflow we evict wholesale; subsequent steps repopulate the few
# buffers that are actually live.
const _TRANSPOSE_CACHE_CAP = 256

function get_transpose_cache()
    if TRANSPOSE_CACHE[] === nothing
        TRANSPOSE_CACHE[] = TransposeBufferCache()
    end
    return TRANSPOSE_CACHE[]
end

"""
    get_transpose_buffer!(cache::TransposeBufferCache, pencil::PencilArrays.Pencil,
                          dtype::Type, key::Tuple)

Get or create a transpose buffer for the given pencil configuration.
"""
function get_transpose_buffer!(cache::TransposeBufferCache, pencil::PencilArrays.Pencil,
                               dtype::Type, key::Tuple)
    full_key = (key..., dtype)

    if haskey(cache.pencil_buffers, full_key)
        cache.hits += 1
        return cache.pencil_buffers[full_key]
    end

    cache.misses += 1

    # Bound the cache (see `_TRANSPOSE_CACHE_CAP`): evict wholesale on overflow. Any
    # buffer still in use stays alive via the field/stash that references it — `empty!`
    # only drops the Dict's reference, exactly the `_SP_MEMO_CAP` memo-cache pattern.
    length(cache.pencil_buffers) >= _TRANSPOSE_CACHE_CAP && empty!(cache.pencil_buffers)

    # Create new PencilArray buffer
    buf = PencilArrays.PencilArray{dtype}(undef, pencil)
    fill!(buf, zero(dtype))

    cache.pencil_buffers[full_key] = buf
    return buf
end

"""
    transpose_pencil_cached!(dest, src, dist::Distributor; cache=nothing)

Cached version of transpose_pencil_data! that reuses buffers.
"""
function transpose_pencil_cached!(dest::PencilArrays.PencilArray, src::PencilArrays.PencilArray,
                                  dist::Distributor)
    start_time = time()

    try
        # Use PencilArrays transpose! (already optimized)
        PencilArrays.transpose!(dest, src)

        dist.performance_stats.transpose_time += time() - start_time
        dist.performance_stats.mpi_operations += 1
    catch e
        # CRITICAL: MPI transpose is essential for correct parallel FFT operation
        # A simple copy would produce incorrect results (copyto! is NOT a transpose!)
        @error "PencilArrays cached transpose failed" exception=e
        error("MPI transpose operation failed with $(dist.size) processes. " *
              "This operation is essential for correct parallel FFT computation. " *
              "Please check your PencilArrays/MPI installation or use serial execution.")
    end

    return dest
end

"""
    clear_transpose_cache!()

Clear the transpose buffer cache (useful for memory reclamation).
"""
function clear_transpose_cache!()
    cache = get_transpose_cache()
    empty!(cache.pencil_buffers)
    cache.hits = 0
    cache.misses = 0
end

"""
    transpose_cache_stats()

Get statistics about the transpose buffer cache.
"""
function transpose_cache_stats()
    cache = get_transpose_cache()
    total = cache.hits + cache.misses
    hit_rate = total > 0 ? cache.hits / total : 0.0

    return (
        hits = cache.hits,
        misses = cache.misses,
        hit_rate = hit_rate,
        num_buffers = length(cache.pencil_buffers),
        memory_bytes = sum(sizeof ∘ parent, values(cache.pencil_buffers); init=0)
    )
end
