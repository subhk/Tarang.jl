# Distributed GPU support, split by section; see gpu_distributed/*.jl.

include("gpu_distributed/config.jl")
include("gpu_distributed/fft.jl")
include("gpu_distributed/transpose.jl")
include("gpu_distributed/utils.jl")
include("gpu_distributed/nccl.jl")
include("gpu_distributed/pinned.jl")
include("gpu_distributed/transform.jl")

# ============================================================================
# Exports
# ============================================================================

export DistributedGPUConfig, DistributedGPUFFT
export distributed_fft_forward!, distributed_fft_backward!
export compute_local_shape, get_local_range
export check_cuda_aware_mpi, setup_distributed_gpu!

# DistributedGPUTransform exports (TransposableField integration)
export DistributedGPUTransform
export create_distributed_gpu_transform
export setup_transposable_workspace!
export distributed_transform_forward!, distributed_transform_backward!
export get_distributed_transform_stats

# NCCL exports
export NCCLConfig, NCCL_CONFIG
export nccl_available, init_nccl!, finalize_nccl!
export nccl_allreduce!, nccl_broadcast!, nccl_allgather!

# NCCL sub-communicator exports (for pencil decomposition)
export NCCLSubComms
export init_nccl_subcomms!, finalize_nccl_subcomms!
export create_nccl_comm_from_mpi

# Staging buffer exports
export DistributedStagingBuffers, get_staging_buffers
export distributed_fft_staged_optimized!
