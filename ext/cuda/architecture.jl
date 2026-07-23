# ============================================================================
# GPU Architecture Implementation
# ============================================================================

"""
    Tarang._gpu_device(device_id::Int)

Select and return the specified CUDA device for Tarang's `GPU` constructor.

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
function Tarang._gpu_device(device_id::Int)
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

    return dev
end

# ============================================================================
# Device and Array Type Methods
# ============================================================================

"""
    device(gpu::GPU{CuDevice})

Return the KernelAbstractions CUDA backend for GPU execution.

Note: device selection is handled by `ensure_device!`, which `launch!` calls
before invoking `device()`. This keeps `device()` free of side effects.
"""
function Tarang.device(gpu::GPU{CuDevice})
    return CUDABackend()
end

"""
    array_type(::GPU{CuDevice})

Return CuArray as the array type for GPU.

All methods in this extension dispatch on `GPU{CuDevice}` (an ext-owned type
parameter) — generic `::GPU` signatures would overwrite the same-signature
error fallbacks in src/core/architectures.jl.
"""
Tarang.array_type(::GPU{CuDevice}) = CuArray

"""
    array_type(::GPU{CuDevice}, T::Type)

Return the concrete CuArray type with element type T.
"""
Tarang.array_type(::GPU{CuDevice}, T::Type) = CuArray{T}

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

Must be defined as `Tarang.ensure_device!` — the name is imported via
`using Tarang: ensure_device!`, and an unqualified definition of an imported
function is an error at extension load time.
"""
function Tarang.ensure_device!(gpu::GPU{CuDevice})
    current = CUDA.device()
    if current != gpu.device
        CUDA.device!(gpu.device)
    end
    return nothing
end

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
Otherwise, copy to the target device via host memory (does not require P2P access).
"""
function Tarang.on_architecture(gpu::GPU{CuDevice}, a::CuArray)
    arr_device = CUDA.device(a)
    if arr_device == gpu.device
        return a  # Already on correct device
    else
        # Cross-device copy: go through host memory to avoid requiring
        # P2P access between devices (not all GPU pairs support it).
        prev_device = CUDA.device()
        try
            CUDA.device!(arr_device)
            host_data = Array(a)
            CUDA.device!(gpu.device)
            result = CuArray(host_data)
            return result
        finally
            CUDA.device!(prev_device)
        end
    end
end

# Fallback for generic GPU
Tarang.on_architecture(::GPU, a::CuArray) = a

"""
    on_architecture(::CPU, a::CuArray)

Move GPU CuArray to CPU Array.
"""
Tarang.on_architecture(::CPU, a::CuArray) = Array(a)

"""
    on_architecture(::CPU, a::AnyCuArray)

Move a wrapped CuArray (SubArray/ReshapedArray/... view of device memory) to a
CPU Array. Materialize on the device first — `Array(::SubArray{...,CuArray})`
would fall back to element-wise scalar indexing.
"""
Tarang.on_architecture(::CPU, a::CUDA.AnyCuArray) = Array(CuArray(a))

"""
    on_architecture(gpu::GPU{CuDevice}, a::AnyCuArray)

Materialize a wrapped CuArray on its own device, then move to `gpu`'s device
if different.
"""
function Tarang.on_architecture(gpu::GPU{CuDevice}, a::CUDA.AnyCuArray)
    root = _root_cuarray(a)
    prev_device = CUDA.device()
    dense = try
        CUDA.device!(CUDA.device(root))
        CuArray(a)
    finally
        CUDA.device!(prev_device)
    end
    return Tarang.on_architecture(gpu, dense)
end

"""
    on_architecture(gpu::GPU{CuDevice}, a::AbstractArray)

Generic host-array fallback (ReshapedArray/SubArray/... of host memory):
materialize to a dense Array, then upload. Without this, `scatter!`'s
`reshape(slice, shape)` views hit a MethodError on GPU fields.
"""
Tarang.on_architecture(gpu::GPU{CuDevice}, a::AbstractArray) = Tarang.on_architecture(gpu, Array(a))

# ============================================================================
# GPU Utilities
# ============================================================================

"""
    has_cuda()

Return true when CUDA is available.
"""
Tarang._cuda_functional(::Val{:cuda}) = CUDA.functional()

"""
    synchronize(gpu::GPU{CuDevice})

Synchronize the specific CUDA device stored in the architecture.
"""
function Tarang.synchronize(gpu::GPU{CuDevice})
    ensure_device!(gpu)
    cuda_sync()
end

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
    is_gpu_array(a)

Any CuArray — including wrapped forms (SubArray, ReshapedArray,
PermutedDimsArray, … over a CuArray, i.e. `CUDA.AnyCuArray`) — is a GPU array.
Covering only bare `CuArray` made views of device memory classify as CPU and
sent them down FFTW/scalar-indexing paths.
"""
Tarang.is_gpu_array(::CUDA.AnyCuArray) = true

# Walk wrapper parents down to the root CuArray (AnyCuArray guarantees one).
_root_cuarray(a::CuArray) = a
function _root_cuarray(a::AbstractArray)
    p = parent(a)
    p === a && error("no CuArray root found for $(typeof(a))")
    return _root_cuarray(p)
end

"""
    architecture(arr::AnyCuArray)

Infer GPU architecture from a (possibly wrapped) CuArray, using the device of
the underlying storage.
"""
Tarang.architecture(arr::CUDA.AnyCuArray) = GPU{CuDevice}(CUDA.device(_root_cuarray(arr)))
