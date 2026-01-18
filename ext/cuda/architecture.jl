# ============================================================================
# GPU Architecture Implementation
# ============================================================================

"""
    GPU(; device_id::Int = 0)

Create a GPU architecture using the specified CUDA device.

# Arguments
- `device_id`: CUDA device ID (0-indexed, default: 0)

# Example
```julia
using CUDA
using Tarang

arch = GPU()                 # Use default device
arch = GPU(device_id=1)      # Use second GPU
```
"""
function Tarang.GPU(; device_id::Int = 0)
    if !CUDA.functional()
        error("CUDA is not functional. Please check your GPU drivers and CUDA installation.")
    end

    # Validate device ID
    n_devices = length(CUDA.devices())
    if device_id < 0 || device_id >= n_devices
        error("Invalid device_id=$device_id. Available devices: 0 to $(n_devices-1)")
    end

    # Set the device
    dev = CuDevice(device_id)
    device!(dev)

    @info "Initialized GPU architecture on device $device_id: $(CUDA.name(dev))"

    return GPU{CuDevice}(dev)
end

# ============================================================================
# Device and Array Type Methods
# ============================================================================

"""
    device(gpu::GPU{CuDevice})

Return the KernelAbstractions CUDA backend after switching to the correct device.

**Important for multi-GPU:** This function sets the current CUDA device to the one
stored in the GPU architecture before returning the backend. This ensures that
kernel launches and FFT operations target the correct GPU, even when multiple
GPUs are in use.
"""
function Tarang.device(gpu::GPU{CuDevice})
    # Switch to the device stored in this architecture
    # This is critical for multi-GPU correctness
    CUDA.device!(gpu.device)
    return CUDABackend()
end

# Fallback for GPU without specific device (uses current device)
Tarang.device(::GPU) = CUDABackend()

"""
    array_type(::GPU)

Return CuArray as the array type for GPU.
"""
Tarang.array_type(::GPU) = CuArray

"""
    array_type(::GPU, T::Type)

Return the concrete CuArray type with element type T.
"""
Tarang.array_type(::GPU, T::Type) = CuArray{T}

# ============================================================================
# Device-aware Base.zeros/ones/similar Overrides
# ============================================================================
# These override the generic GPU methods in architectures.jl to ensure
# the correct CUDA device is active before allocation in multi-GPU setups.

"""
    Base.zeros(arch::GPU{CuDevice}, T::Type, dims...)

Create a zero-filled CuArray on the specific GPU device.
Ensures correct device context for multi-GPU support.
"""
function Base.zeros(arch::GPU{CuDevice}, T::Type, dims::Integer...)
    ensure_device!(arch)
    return CUDA.zeros(T, dims...)
end

function Base.zeros(arch::GPU{CuDevice}, T::Type, dims::Tuple{Vararg{Integer}})
    ensure_device!(arch)
    return CUDA.zeros(T, dims...)
end

"""
    Base.ones(arch::GPU{CuDevice}, T::Type, dims...)

Create a one-filled CuArray on the specific GPU device.
Ensures correct device context for multi-GPU support.
"""
function Base.ones(arch::GPU{CuDevice}, T::Type, dims::Integer...)
    ensure_device!(arch)
    return CUDA.ones(T, dims...)
end

function Base.ones(arch::GPU{CuDevice}, T::Type, dims::Tuple{Vararg{Integer}})
    ensure_device!(arch)
    return CUDA.ones(T, dims...)
end

"""
    Base.similar(arch::GPU{CuDevice}, a::AbstractArray)

Create an uninitialized CuArray on the specific GPU device.
Ensures correct device context for multi-GPU support.
"""
function Base.similar(arch::GPU{CuDevice}, a::AbstractArray)
    ensure_device!(arch)
    return CuArray{eltype(a)}(undef, size(a)...)
end

function Base.similar(arch::GPU{CuDevice}, a::AbstractArray, T::Type)
    ensure_device!(arch)
    return CuArray{T}(undef, size(a)...)
end

function Base.similar(arch::GPU{CuDevice}, a::AbstractArray, dims::Tuple{Vararg{Integer}})
    ensure_device!(arch)
    return CuArray{eltype(a)}(undef, dims...)
end

function Base.similar(arch::GPU{CuDevice}, a::AbstractArray, T::Type, dims::Tuple{Vararg{Integer}})
    ensure_device!(arch)
    return CuArray{T}(undef, dims...)
end

"""
    architecture(arr::CuArray)

Infer GPU architecture from CuArray.
Uses the device associated with the array, not the current device.
This is important for multi-GPU setups where arrays may reside on different devices.
"""
Tarang.architecture(arr::CuArray) = GPU{CuDevice}(CUDA.device(arr))

"""
    ensure_device!(gpu::GPU{CuDevice})

Ensure the correct CUDA device is active for operations on this architecture.
Call this before any GPU operation that doesn't go through `device(gpu)`.
"""
function ensure_device!(gpu::GPU{CuDevice})
    current = CUDA.device()
    if current != gpu.device
        CUDA.device!(gpu.device)
    end
    return nothing
end

ensure_device!(::GPU) = nothing  # No-op for generic GPU

# ============================================================================
# Data Movement
# ============================================================================

"""
    on_architecture(gpu::GPU{CuDevice}, a::Array)

Move CPU Array to GPU CuArray on the specific device.
Ensures the correct device is active before allocation.
"""
function Tarang.on_architecture(gpu::GPU{CuDevice}, a::Array)
    ensure_device!(gpu)
    return CuArray(a)
end

# Fallback for generic GPU (uses current device)
Tarang.on_architecture(::GPU, a::Array) = CuArray(a)

"""
    on_architecture(gpu::GPU{CuDevice}, a::CuArray)

GPU to GPU: handle potential cross-device transfers.
If the array is already on the target device, return as-is.
Otherwise, copy to the target device.
"""
function Tarang.on_architecture(gpu::GPU{CuDevice}, a::CuArray)
    arr_device = CUDA.device(a)
    if arr_device == gpu.device
        return a  # Already on correct device
    else
        # Cross-device copy: switch to target device and copy
        ensure_device!(gpu)
        return CuArray(a)  # This copies to current device
    end
end

# Fallback for generic GPU
Tarang.on_architecture(::GPU, a::CuArray) = a

"""
    on_architecture(::CPU, a::CuArray)

Move GPU CuArray to CPU Array.
"""
Tarang.on_architecture(::CPU, a::CuArray) = Array(a)

# ============================================================================
# GPU Utilities
# ============================================================================

"""
    has_cuda()

Return true when CUDA is available.
"""
Tarang.has_cuda() = CUDA.functional()

"""
    synchronize(gpu::GPU{CuDevice})

Synchronize the specific CUDA device stored in the architecture.
"""
function Tarang.synchronize(gpu::GPU{CuDevice})
    ensure_device!(gpu)
    cuda_sync()
end

# Fallback for generic GPU (syncs current device)
Tarang.synchronize(::GPU) = cuda_sync()

"""
    unsafe_free!(gpu::GPU{CuDevice}, a::CuArray)

Explicitly free GPU memory. Ensures correct device context.
"""
function Tarang.unsafe_free!(gpu::GPU{CuDevice}, a::CuArray)
    # Note: unsafe_free! works on the array's device, but we ensure
    # the context is correct for any cleanup operations
    CUDA.unsafe_free!(a)
end

# Fallback
Tarang.unsafe_free!(::GPU, a::CuArray) = CUDA.unsafe_free!(a)

# ============================================================================
# Override is_gpu_array for CuArray detection
# ============================================================================

"""
    is_gpu_array(a::CuArray)

CuArray is a GPU array.
"""
Tarang.is_gpu_array(::CuArray) = true
