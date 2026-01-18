# ============================================================================
# GPU Configuration and Global State
# ============================================================================

"""
    GPUConfig

Global GPU configuration for performance tuning.
"""
mutable struct GPUConfig
    # Stream management
    default_stream::Union{Nothing, CuStream}
    compute_stream::Union{Nothing, CuStream}
    transfer_stream::Union{Nothing, CuStream}

    # Memory pool settings
    use_memory_pool::Bool

    # Tensor core settings
    use_tensor_cores::Bool
    tensor_math_mode::Any  # CUBLAS math mode

    # Workgroup sizes
    default_workgroup_1d::Int
    default_workgroup_2d::Tuple{Int, Int}
    default_workgroup_3d::Tuple{Int, Int, Int}

    function GPUConfig()
        new(nothing, nothing, nothing,
            true, false, nothing,
            256, (16, 16), (8, 8, 4))
    end
end

const GPU_CONFIG = GPUConfig()

"""
    init_gpu_config!(; use_streams::Bool=true, use_tensor_cores::Bool=false,
                      use_memory_pool::Bool=true)

Initialize GPU configuration with performance options.

# Arguments
- `use_streams`: Enable CUDA streams for async operations (default: true)
- `use_tensor_cores`: Enable Tensor Cores for compatible operations (default: false)
- `use_memory_pool`: Enable memory pooling for reduced allocation overhead (default: true)

# Example
```julia
init_gpu_config!(use_streams=true, use_tensor_cores=true)
```
"""
function init_gpu_config!(; use_streams::Bool=true,
                           use_tensor_cores::Bool=false,
                           use_memory_pool::Bool=true)
    if use_streams
        GPU_CONFIG.default_stream = CuStream()
        GPU_CONFIG.compute_stream = CuStream()
        GPU_CONFIG.transfer_stream = CuStream()
        @info "GPU streams initialized"
    end

    GPU_CONFIG.use_memory_pool = use_memory_pool

    if use_tensor_cores
        enable_tensor_cores!()
    end

    return GPU_CONFIG
end

"""
    get_compute_stream()

Get the compute stream for kernel execution.
"""
function get_compute_stream()
    return GPU_CONFIG.compute_stream !== nothing ? GPU_CONFIG.compute_stream : CUDA.stream()
end

"""
    get_transfer_stream()

Get the transfer stream for CPU-GPU data movement.
"""
function get_transfer_stream()
    return GPU_CONFIG.transfer_stream !== nothing ? GPU_CONFIG.transfer_stream : CUDA.stream()
end

"""
    sync_streams!()

Synchronize all GPU streams.
"""
function sync_streams!()
    if GPU_CONFIG.compute_stream !== nothing
        CUDA.synchronize(GPU_CONFIG.compute_stream)
    end
    if GPU_CONFIG.transfer_stream !== nothing
        CUDA.synchronize(GPU_CONFIG.transfer_stream)
    end
end

# ============================================================================
# Tensor Core Support
# ============================================================================

"""
    enable_tensor_cores!()

Enable Tensor Core operations for compatible CUDA operations.
Provides significant speedup on Volta+ GPUs for matrix operations.
"""
function enable_tensor_cores!()
    try
        # Enable TF32 mode for single precision (Ampere+)
        CUDA.math_mode!(CUDA.FAST_MATH)
        GPU_CONFIG.use_tensor_cores = true
        GPU_CONFIG.tensor_math_mode = CUDA.FAST_MATH
        @info "Tensor Cores enabled (FAST_MATH mode)"
    catch e
        @warn "Could not enable Tensor Cores: $e"
        GPU_CONFIG.use_tensor_cores = false
    end
end

"""
    disable_tensor_cores!()

Disable Tensor Core operations for strict IEEE compliance.
"""
function disable_tensor_cores!()
    try
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
        GPU_CONFIG.use_tensor_cores = false
        GPU_CONFIG.tensor_math_mode = CUDA.DEFAULT_MATH
        @info "Tensor Cores disabled (DEFAULT_MATH mode)"
    catch e
        @warn "Could not disable Tensor Cores: $e"
    end
end
