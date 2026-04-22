"""
Distributor class for parallel distribution and transformations

Key parallelization features:
- PencilArrays for efficient MPI domain decomposition
- PencilFFTs for parallel spectral transforms
- Automatic mesh optimization for 2D/3D decomposition
- Layout caching for performance
"""


# Runtime map:
#   distributor_core.jl              — distributor types, topology setup, layouts, and local indexing
#   distributor_mpi.jl               — gather/scatter/allreduce helpers, cache limits, and mesh utilities
#   distributor_transpose.jl         — transpose-pencil helpers and transpose-buffer cache
#   distributor_exchange.jl          — async MPI exchange, communication buffers, and diagnostics
#   distributor_grouped_transpose.jl — grouped PencilArray transpose support

include("distributor/distributor_core.jl")
include("distributor/distributor_mpi.jl")
include("distributor/distributor_transpose.jl")
include("distributor/distributor_exchange.jl")
include("distributor/distributor_grouped_transpose.jl")

# ============================================================================
# Exports
# ============================================================================

# Export types
export Layout, DistributorPerformanceStats, Distributor, TransposeBufferCache

# Export core functions
export setup_pencil_arrays, create_pencil, compute_local_shape,
       get_process_coordinate_in_mesh, get_layout, local_indices,
       local_grids, remedy_scales, get_axis, get_basis_axis,
       first_axis, last_axis

# Export MPI communication functions
export gather_array, scatter_array, allreduce_array, mpi_alltoall,
       async_allreduce!, wait_async!,
       neighbor_exchange!, async_neighbor_exchange!, wait_neighbor_exchange!,
       test_neighbor_exchange, get_neighbor_ranks, coord_to_rank

# Export pencil transpose functions
export create_transpose_pencil, transpose_pencil_data!, transpose_pencil_cached!,
       get_transpose_cache, get_transpose_buffer!, clear_transpose_cache!,
       transpose_cache_stats

# Export grouped PencilArray transposes (GROUP_TRANSPOSES for CPU)
export GroupedPencilTransposeConfig, set_grouped_pencil_transposes!
export group_pencil_transpose!, group_transpose_fields!

# Export cache and memory management
export clear_distributor_cache!, enforce_cache_limits!, maybe_cleanup_caches!,
       get_distributor_memory_info, preallocate_communication_buffers!,
       get_communication_buffers, clear_communication_buffers!,
       get_optimal_chunk_size

# Export performance and diagnostics
export log_distributor_performance, diagnose_parallel_performance,
       reset_performance_stats!

# Export mesh creation utilities
export create_2d_process_mesh, create_3d_process_mesh
