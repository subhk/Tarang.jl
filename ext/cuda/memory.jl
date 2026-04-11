# ============================================================================
# GPU Memory Helpers
# ============================================================================
# Follows Oceananigans' approach: use CUDA.jl's built-in memory pool
# instead of reimplementing pooling. See JULIA_CUDA_MEMORY_POOL for config.

"""
    async_copy_to_gpu!(dst::CuArray, src::Array)

Copy data from CPU to GPU.
"""
async_copy_to_gpu!(dst::CuArray{T}, src::Array{T}) where T = copyto!(dst, src)

"""
    async_copy_to_cpu!(dst::Array, src::CuArray)

Copy data from GPU to CPU.
"""
async_copy_to_cpu!(dst::Array{T}, src::CuArray{T}) where T = copyto!(dst, src)
