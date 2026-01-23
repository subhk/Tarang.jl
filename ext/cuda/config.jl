# ============================================================================
# GPU Configuration and Global State
# ============================================================================

"""
    GPUConfig

Global GPU configuration for performance tuning.
Streams are stored per-device to ensure multi-GPU correctness.
"""
mutable struct GPUConfig
    # Stream management - per device ID for multi-GPU correctness
    compute_streams::Dict{Int, CuStream}
    transfer_streams::Dict{Int, CuStream}
    streams_enabled::Bool

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
        new(Dict{Int, CuStream}(), Dict{Int, CuStream}(), false,
            true, false, nothing,
            256, (16, 16), (8, 8, 4))
    end
end

const GPU_CONFIG = GPUConfig()

"""
    init_gpu_config!(; use_streams::Bool=true, use_tensor_cores::Bool=false,
                      use_memory_pool::Bool=true)

Initialize GPU configuration with performance options.
Creates streams on the current device. Call once per device in multi-GPU setups.

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
    GPU_CONFIG.streams_enabled = use_streams
    if use_streams
        # Create streams on the current device
        device_id = CUDA.deviceid()
        GPU_CONFIG.compute_streams[device_id] = CuStream()
        GPU_CONFIG.transfer_streams[device_id] = CuStream()
        @info "GPU streams initialized for device $device_id"
    end

    GPU_CONFIG.use_memory_pool = use_memory_pool

    if use_tensor_cores
        enable_tensor_cores!()
    end

    return GPU_CONFIG
end

"""
    get_compute_stream(; device_id::Int=CUDA.deviceid())

Get the compute stream for kernel execution on the specified device.
Returns `nothing` when streams are disabled; callers should use the plain
(non-stream) code path in that case. Creates a new stream on-demand if one
doesn't exist for this device.
"""
function get_compute_stream(; device_id::Int=CUDA.deviceid())
    if !GPU_CONFIG.streams_enabled
        return nothing
    end
    if !haskey(GPU_CONFIG.compute_streams, device_id)
        # Create stream on the target device (streams are device-specific in CUDA)
        prev_device = CUDA.device()
        CUDA.device!(CuDevice(device_id))
        GPU_CONFIG.compute_streams[device_id] = CuStream()
        CUDA.device!(prev_device)
    end
    return GPU_CONFIG.compute_streams[device_id]
end

"""
    get_transfer_stream(; device_id::Int=CUDA.deviceid())

Get the transfer stream for CPU-GPU data movement on the specified device.
Returns `nothing` when streams are disabled; callers should use the plain
(non-stream) code path in that case. Creates a new stream on-demand if one
doesn't exist for this device.
"""
function get_transfer_stream(; device_id::Int=CUDA.deviceid())
    if !GPU_CONFIG.streams_enabled
        return nothing
    end
    if !haskey(GPU_CONFIG.transfer_streams, device_id)
        # Create stream on the target device (streams are device-specific in CUDA)
        prev_device = CUDA.device()
        CUDA.device!(CuDevice(device_id))
        GPU_CONFIG.transfer_streams[device_id] = CuStream()
        CUDA.device!(prev_device)
    end
    return GPU_CONFIG.transfer_streams[device_id]
end

"""
    sync_streams!(; device_id::Union{Nothing, Int}=nothing)

Synchronize GPU streams. If `device_id` is specified, only sync that device's streams.
Otherwise, sync all devices' streams.
"""
function sync_streams!(; device_id::Union{Nothing, Int}=nothing)
    if device_id !== nothing
        # Sync specific device
        if haskey(GPU_CONFIG.compute_streams, device_id)
            CUDA.synchronize(GPU_CONFIG.compute_streams[device_id])
        end
        if haskey(GPU_CONFIG.transfer_streams, device_id)
            CUDA.synchronize(GPU_CONFIG.transfer_streams[device_id])
        end
    else
        # Sync all devices
        for (_, stream) in GPU_CONFIG.compute_streams
            CUDA.synchronize(stream)
        end
        for (_, stream) in GPU_CONFIG.transfer_streams
            CUDA.synchronize(stream)
        end
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
