"""
    TarangCUDAExt - CUDA GPU support extension for Tarang.jl

This extension is automatically loaded when CUDA.jl is available.
It provides GPU implementations of:
- Array allocation and data movement
- FFT transforms using CUFFT
- DCT transforms for Chebyshev basis
- Mixed Fourier-Chebyshev transforms
- Element-wise kernels using KernelAbstractions
- Nonlinear term evaluation on GPU

The extension is organized into the following files:
- config.jl: GPU configuration and tensor core support
- memory.jl: Memory pool and pinned memory management
- architecture.jl: GPU architecture implementation and data movement
- transforms.jl: Field transforms and FFT plans
- dct.jl: DCT transforms for Chebyshev basis (2D and 3D)
- mixed_transforms.jl: Mixed Fourier-Chebyshev transform plans
- kernels.jl: Element-wise and fused GPU kernels
- batched_fft.jl: Batched FFT support
- utils.jl: Utility functions, dealiasing, memory management
"""
module TarangCUDAExt

using Tarang
using Tarang: AbstractArchitecture, AbstractSerialArchitecture, GPU, CPU
using Tarang: device, array_type, architecture, on_architecture, workgroup_size, launch!, KernelOperation
using Tarang: synchronize, unsafe_free!, has_cuda
using Tarang: ScalarField, VectorField, Distributor, Domain, Basis
using Tarang: get_local_data, set_local_data!
using Tarang: get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!
using Tarang: is_gpu_array, gpu_forward_transform!, gpu_backward_transform!, should_use_gpu_fft
using Tarang: RealFourier, ComplexFourier, ChebyshevT, Legendre
using Tarang: DistributedGPUFFT
# Note: global_shape and coefficient_shape are NOT imported - transform functions
# use local array sizes (size(data_g), size(data_c)) for MPI correctness
# Note: coefficient_eltype, gpu_multiply_fields!, local_fft_dim!, local_ifft_dim!
# are not exported from Tarang - access them via Tarang.function_name prefix

using CUDA
using CUDA: CuArray, CuDevice, device!, synchronize as cuda_sync
using CUDA: CuStream, default_stream, @sync
using CUDA.CUFFT
using CUDA.CUBLAS
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const

# Include sub-modules in dependency order
include("cuda/config.jl")
include("cuda/memory.jl")
include("cuda/architecture.jl")
include("cuda/transforms.jl")
include("cuda/dct.jl")
include("cuda/mixed_transforms.jl")
include("cuda/kernels.jl")
include("cuda/batched_fft.jl")
include("cuda/utils.jl")
include("cuda/transpose_kernels.jl")  # TransposableField GPU support

# ============================================================================
# Exports
# ============================================================================

# GPU Configuration and Device Management
export GPUConfig, GPU_CONFIG
export init_gpu_config!, get_compute_stream, get_transfer_stream, sync_streams!
export ensure_device!  # Multi-GPU device context management

# Tensor Core support
export enable_tensor_cores!, disable_tensor_cores!

# Memory pool
export GPUMemoryPool, GPU_MEMORY_POOL
export pool_allocate, pool_release!, clear_memory_pool!, memory_pool_stats

# Pinned memory
export PinnedBufferPool, PINNED_BUFFER_POOL
export get_pinned_buffer, release_pinned_buffer!, clear_pinned_buffer_pool!
export async_copy_to_gpu!, async_copy_to_cpu!

# FFT plans and transforms
export GPUFFTPlan, GPUTransformCache
export plan_gpu_fft, gpu_forward_fft!, gpu_backward_fft!
export get_gpu_fft_plan, clear_gpu_transform_cache!
export gpu_fft_async!, gpu_ifft_async!

# Batched FFT
export BatchedGPUFFTPlan, BATCHED_FFT_CACHE
export plan_batched_gpu_fft, get_batched_fft_plan
export batched_fft!, batched_ifft!, clear_batched_fft_cache!

# GPU DCT (Discrete Cosine Transform) for Chebyshev basis
export GPUDCTPlan, GPUDCTPlanDim
export plan_gpu_dct, plan_gpu_dct_dim
export gpu_forward_dct_1d!, gpu_backward_dct_1d!
export gpu_dct_dim!

# Dimension-by-dimension FFT plans
export GPUFFTPlanDim
export plan_gpu_fft_dim
export gpu_fft_dim!, gpu_ifft_dim!

# Mixed Fourier-Chebyshev transforms
export GPUMixedTransformPlan, GPU_MIXED_TRANSFORM_CACHE
export plan_gpu_mixed_transform, get_gpu_mixed_transform_plan
export clear_gpu_mixed_transform_cache!
export gpu_mixed_forward_transform!, gpu_mixed_backward_transform!

# Basic kernels
export gpu_add!, gpu_sub!, gpu_mul!, gpu_scale!, gpu_axpy!, gpu_linear_combination!
export GPU_ADD_OP, GPU_SUB_OP, GPU_MUL_OP, GPU_SCALE_OP, GPU_AXPY_OP, GPU_LINEAR_COMB_OP

# Fused kernels
export gpu_rk_stage!, gpu_axpby!, gpu_fma!
export gpu_scale_multiply!, gpu_dealias_multiply!, gpu_triple_product!
export gpu_conj_multiply!, gpu_squared_magnitude!
export gpu_kinetic_energy_2d!, gpu_kinetic_energy_3d!
export gpu_grad_mag_sq_2d!, gpu_viscous_damping!

# Utility functions
# NOTE: For array allocation and data transfers, use:
#   - Base.zeros(arch, T, dims...) / Base.ones(arch, T, dims...)
#   - Tarang.on_architecture(arch, array) for data transfers
# The to_gpu/to_cpu functions are internal to this extension - use on_architecture instead.
export allocate_gpu_data
export gpu_fft_2d_forward!, gpu_fft_2d_backward!, gpu_fft_3d_forward!, gpu_fft_3d_backward!
export create_dealiasing_mask_gpu, apply_dealiasing_gpu!
export gpu_memory_info, check_gpu_memory

# Internal allocation helpers are NOT exported - use Base.zeros/ones/similar with GPU arch instead
# These functions are internal: _gpu_zeros, _gpu_ones, _gpu_similar, _gpu_fill

# Transpose kernels for TransposableField
export pack_for_transpose_kernel_3d!, pack_for_transpose_kernel_2d!
export unpack_from_transpose_kernel_3d!, unpack_from_transpose_kernel_2d!
export gpu_pack_for_transpose!, gpu_unpack_from_transpose!

end # module TarangCUDAExt
