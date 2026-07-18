# ============================================================================
# GPU Memory Helpers
# ============================================================================
# Follows Oceananigans' approach: use CUDA.jl's built-in memory pool
# instead of reimplementing pooling. See JULIA_CUDA_MEMORY_POOL for config.

"""
    async_copy_to_gpu!(dst::CuArray, src::Array)

Copy data from CPU to GPU.

NOTE: despite the name, this is a plain `copyto!` — for pageable host memory
the transfer is synchronous. True async copies need pinned host memory and an
explicit stream; callers must not rely on overlap.
"""
async_copy_to_gpu!(dst::CuArray{T}, src::Array{T}) where T = copyto!(dst, src)

"""
    async_copy_to_cpu!(dst::Array, src::CuArray)

Copy data from GPU to CPU. Same caveat as [`async_copy_to_gpu!`](@ref): this
is a synchronous `copyto!` for pageable host memory.
"""
async_copy_to_cpu!(dst::Array{T}, src::CuArray{T}) where T = copyto!(dst, src)
