"""
    Architectures - Backend abstraction for CPU and GPU execution

Architecture abstraction following Oceananigans.jl pattern for backend-agnostic computation.
This enables running the same code on CPU or GPU with minimal changes.

Key concepts:
- Architecture types (CPU, GPU) define the compute backend
- array_type() maps architecture to array constructors
- on_architecture() moves data between backends
- device() returns the KernelAbstractions backend

Usage:
    arch = CPU()           # Use CPU backend (default)
    arch = GPU()           # Use CUDA GPU backend (requires CUDA.jl)

    # Create arrays on architecture
    a = zeros(arch, Float64, 100, 100)

    # Move data between architectures
    a_gpu = on_architecture(GPU(), a_cpu)
    a_cpu = on_architecture(CPU(), a_gpu)
"""

using KernelAbstractions

# ============================================================================
# Abstract Architecture Types
# ============================================================================

"""
    AbstractArchitecture

Abstract supertype for all compute architectures.
"""
abstract type AbstractArchitecture end

"""
    AbstractSerialArchitecture <: AbstractArchitecture

Architecture for single-device (non-distributed) computation.
"""
abstract type AbstractSerialArchitecture <: AbstractArchitecture end

# ============================================================================
# CPU Architecture
# ============================================================================

"""
    CPU <: AbstractSerialArchitecture

CPU architecture using Julia's native arrays and threading.
This is the default architecture.

# Example
```julia
arch = CPU()
dist = Distributor(coordsys; architecture=arch)
```
"""
struct CPU <: AbstractSerialArchitecture end

# Note: The struct definition above provides the default CPU() constructor.
# No additional constructor needed for CPU since it has no fields.

"""
    device(::CPU)

Return the KernelAbstractions backend for CPU.
"""
device(::CPU) = KernelAbstractions.CPU()

"""
    array_type(::CPU)

Return the array type for CPU architecture.
"""
array_type(::CPU) = Array

"""
    array_type(::CPU, T::Type)

Return the concrete array type for CPU with element type T.
"""
array_type(::CPU, T::Type) = Array{T}

# ============================================================================
# GPU Architecture (placeholder - actual implementation via extension)
# ============================================================================

"""
    GPU <: AbstractSerialArchitecture

GPU architecture using CUDA.jl for NVIDIA GPUs.
Requires CUDA.jl to be loaded for full functionality.

# Example
```julia
using CUDA  # Must load CUDA first
arch = GPU()
dist = Distributor(coordsys; architecture=arch)
```

# Notes
- GPU support is optional and loaded via package extension
- If CUDA.jl is not available, GPU() will throw an error
- For multi-GPU, use with MPI (one GPU per MPI rank)
"""
struct GPU{D} <: AbstractSerialArchitecture
    device::D
end

# Default GPU constructor - will be overridden by extension
function GPU(; device_id::Int = 0)
    error("""
        GPU architecture requires CUDA.jl to be loaded.

        Please add CUDA.jl to your project and load it:
            using Pkg
            Pkg.add("CUDA")
            using CUDA
            using Tarang

            arch = GPU()
        """)
end

# Placeholder methods - overridden by CUDA extension
device(gpu::GPU) = gpu.device
array_type(::GPU) = error("GPU array_type requires CUDA.jl to be loaded")
array_type(::GPU, T::Type) = error("GPU array_type requires CUDA.jl to be loaded")

# ============================================================================
# Architecture Detection and Inference
# ============================================================================

"""
    architecture(a::Array)

Infer the architecture from an array type.
Returns CPU() for standard Julia arrays.
"""
architecture(::Array) = CPU()
architecture(::AbstractArray) = CPU()  # Default fallback

"""
    architecture(arch::AbstractArchitecture)

Return the architecture itself (identity).
"""
architecture(arch::AbstractArchitecture) = arch

# ============================================================================
# Data Movement Between Architectures
# ============================================================================

"""
    on_architecture(arch::AbstractArchitecture, data)

Move data to the specified architecture.
Returns the data on the target architecture.

# Arguments
- `arch`: Target architecture (CPU() or GPU())
- `data`: Data to move (Array, Tuple, NamedTuple, etc.)

# Returns
- Data converted to the target architecture's array type

# Example
```julia
a_cpu = rand(100, 100)
a_gpu = on_architecture(GPU(), a_cpu)
a_back = on_architecture(CPU(), a_gpu)
```
"""
function on_architecture end

# CPU to CPU: no-op
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a

# Convert any array to CPU
function on_architecture(::CPU, a::AbstractArray)
    Array(a)
end

# Handle tuples recursively
function on_architecture(arch::AbstractArchitecture, t::Tuple)
    Tuple(on_architecture(arch, elem) for elem in t)
end

# Handle NamedTuples recursively
function on_architecture(arch::AbstractArchitecture, nt::NamedTuple)
    names = keys(nt)
    values = Tuple(on_architecture(arch, nt[k]) for k in names)
    NamedTuple{names}(values)
end

# Handle Nothing
on_architecture(::AbstractArchitecture, ::Nothing) = nothing

# Handle scalars (pass through)
on_architecture(::AbstractArchitecture, x::Number) = x
on_architecture(::AbstractArchitecture, x::String) = x
on_architecture(::AbstractArchitecture, x::Symbol) = x

# ============================================================================
# Array Allocation on Architecture
# ============================================================================

"""
    zeros(arch::AbstractArchitecture, T::Type, dims...)

Create a zero-filled array on the specified architecture.

# Example
```julia
a = zeros(CPU(), Float64, 100, 100)
a = zeros(GPU(), Float64, 100, 100)
```
"""
function Base.zeros(arch::CPU, T::Type, dims::Integer...)
    # For CPU, just use standard Julia zeros which creates Array{T}
    return zeros(T, dims...)
end

function Base.zeros(arch::GPU, T::Type, dims::Integer...)
    # For GPU, use the array_type which returns CuArray{T}
    # CUDA.jl provides zeros(CuArray{T}, dims...) method
    AT = array_type(arch, T)
    return zeros(AT, dims...)
end

function Base.zeros(arch::AbstractArchitecture, T::Type, dims::Tuple{Vararg{Integer}})
    zeros(arch, T, dims...)
end

"""
    ones(arch::AbstractArchitecture, T::Type, dims...)

Create an array of ones on the specified architecture.
"""
function Base.ones(arch::CPU, T::Type, dims::Integer...)
    # For CPU, just use standard Julia ones which creates Array{T}
    return ones(T, dims...)
end

function Base.ones(arch::GPU, T::Type, dims::Integer...)
    # For GPU, use the array_type which returns CuArray{T}
    # CUDA.jl provides ones(CuArray{T}, dims...) method
    AT = array_type(arch, T)
    return ones(AT, dims...)
end

function Base.ones(arch::AbstractArchitecture, T::Type, dims::Tuple{Vararg{Integer}})
    ones(arch, T, dims...)
end

"""
    similar(arch::AbstractArchitecture, a::AbstractArray)

Create an uninitialized array with the same size on the specified architecture.
"""
function Base.similar(arch::CPU, a::AbstractArray)
    # For CPU, create a standard Array with same element type and size
    return similar(Array{eltype(a)}, size(a))
end

function Base.similar(arch::GPU, a::AbstractArray)
    # For GPU, use array_type to get CuArray type
    AT = array_type(arch, eltype(a))
    return similar(AT, size(a))
end

function Base.similar(arch::CPU, a::AbstractArray, T::Type)
    # For CPU, create a standard Array with specified element type
    return similar(Array{T}, size(a))
end

function Base.similar(arch::GPU, a::AbstractArray, T::Type)
    # For GPU, use array_type to get CuArray type
    AT = array_type(arch, T)
    return similar(AT, size(a))
end

# ============================================================================
# Architecture Utilities
# ============================================================================

"""
    is_gpu(arch::AbstractArchitecture)

Check if the architecture is a GPU.
"""
is_gpu(::CPU) = false
is_gpu(::GPU) = true

"""
    has_cuda()

Check if CUDA is available (placeholder - set by extension).
"""
has_cuda() = false  # Overridden by CUDA extension

"""
    is_gpu_array(a::AbstractArray)

Check if an array is on GPU (CUDA array).
Returns false for standard Julia arrays, true for CuArrays.
This is a default implementation; CUDA extension overrides for CuArray.
"""
is_gpu_array(::Array) = false
is_gpu_array(::AbstractArray) = false  # Default: not a GPU array

# ============================================================================
# GPU Compatibility Helpers
# ============================================================================

"""
    allocate_like(a::AbstractArray, T::Type, dims...)

Allocate a zeros array with the same backend (CPU/GPU) as the input array.
This ensures GPU arrays stay on GPU and CPU arrays stay on CPU.
"""
function allocate_like(a::AbstractArray, T::Type, dims...)
    # Default implementation for CPU arrays
    return zeros(T, dims...)
end

function allocate_like(a::AbstractArray, dims...)
    return allocate_like(a, eltype(a), dims...)
end

"""
    similar_zeros(a::AbstractArray)

Create a zero-filled array with same type, element type, and size as input.
Works for both CPU and GPU arrays.
"""
function similar_zeros(a::AbstractArray)
    result = similar(a)
    fill!(result, zero(eltype(a)))
    return result
end

"""
    similar_zeros(a::AbstractArray, dims...)

Create a zero-filled array with same type and element type as input, but different size.
Works for both CPU and GPU arrays.
"""
function similar_zeros(a::AbstractArray, dims...)
    result = similar(a, dims...)
    fill!(result, zero(eltype(a)))
    return result
end

"""
    similar_zeros(a::AbstractArray, T::Type, dims...)

Create a zero-filled array with same array type as input, but different element type and size.
Works for both CPU and GPU arrays.
"""
function similar_zeros(a::AbstractArray, T::Type, dims...)
    result = similar(a, T, dims...)
    fill!(result, zero(T))
    return result
end

"""
    copy_to_device(a::AbstractArray, target::AbstractArray)

Copy array `a` to the same device as `target`.
Returns `a` if already on same device, otherwise creates a copy on target device.
"""
function copy_to_device(a::AbstractArray, target::AbstractArray)
    # Default: just copy (both assumed CPU)
    return copy(a)
end

# ============================================================================
# Architecture Array Creation
# ============================================================================

"""
    create_array(arch::AbstractArchitecture, T::Type, dims...)

Create a zero-filled array of type T with given dimensions on the specified architecture.
This is a convenience function that works like zeros(arch, T, dims...).

# Example
```julia
a = create_array(CPU(), Float64, 100, 100)
a = create_array(GPU(), ComplexF64, 64, 64)
```
"""
function create_array(arch::AbstractArchitecture, T::Type, dims::Integer...)
    return zeros(arch, T, dims...)
end

function create_array(arch::AbstractArchitecture, T::Type, dims::Tuple{Vararg{Integer}})
    return zeros(arch, T, dims...)
end

"""
    synchronize(arch::AbstractArchitecture)

Synchronize the device (wait for all operations to complete).
No-op for CPU, calls CUDA.synchronize() for GPU.
"""
synchronize(::CPU) = nothing
synchronize(::GPU) = error("GPU synchronize requires CUDA.jl to be loaded")

"""
    unsafe_free!(arch::AbstractArchitecture, a)

Explicitly free GPU memory. No-op for CPU.
"""
unsafe_free!(::CPU, a) = nothing
unsafe_free!(::GPU, a) = error("GPU unsafe_free! requires CUDA.jl to be loaded")

# ============================================================================
# Kernel Launch Utilities
# ============================================================================

"""
    launch_config(arch::AbstractArchitecture, n::Int)

Get optimal kernel launch configuration for n elements.
Returns (threads, blocks) for GPU, (n,) for CPU.
"""
function launch_config(::CPU, n::Int)
    return (n,)  # CPU uses all elements
end

function launch_config(::GPU, n::Int)
    threads = 256  # Standard CUDA threads per block
    blocks = cld(n, threads)
    return (threads, blocks)
end

"""
    workgroup_size(arch::AbstractArchitecture, N)

Get the workgroup size for KernelAbstractions kernels.
Supports integer `N` or tuple/Cartesian nd-ranges (following Oceananigans.jl).
"""
workgroup_size(::CPU, N::Integer) = min(N, 64)
workgroup_size(::GPU, N::Integer) = 256

_ndrange_length(N::Integer) = N
_ndrange_length(N::Tuple) = prod(N)
_ndrange_length(N::CartesianIndices) = length(N)
_ndrange_length(N::AbstractArray) = length(N)
_ndrange_length(N::AbstractRange) = length(N)

function workgroup_size(arch::AbstractArchitecture, ndrange)
    return workgroup_size(arch, _ndrange_length(ndrange))
end

"""
    launch!(arch_or_data, kernel, args...; ndrange, dependencies=nothing, wait=true, kwargs...)

Oceananigans-style helper to launch KernelAbstractions kernels on CPU or GPU.
- `arch_or_data`: Architecture (`CPU()`, `GPU()`) or an array (architecture inferred)
- `kernel`: KernelAbstractions `@kernel` function
- `ndrange`: Number of elements/work items (integer or tuple)
- `dependencies`: Optional kernel dependencies
- `wait`: If true (default), wait for the returned event before returning

Returns the KernelAbstractions event, even when `wait=true`, so callers can
chain launches using `dependencies` similar to Oceananigans.
"""
function launch!(arch::AbstractArchitecture, kernel, args...;
                 ndrange,
                 dependencies=nothing,
                 wait::Bool=true,
                 kwargs...)
    backend = device(arch)
    workgroup = workgroup_size(arch, ndrange)
    ka_kernel = kernel(backend, workgroup)
    event = ka_kernel(args...; ndrange=ndrange, dependencies=dependencies, kwargs...)
    if wait
        Base.wait(event)
    end
    return event
end

function launch!(data::AbstractArray, kernel, args...; kwargs...)
    arch = architecture(data)
    return launch!(arch, kernel, args...; kwargs...)
end

# ============================================================================
# Printing
# ============================================================================

Base.summary(::CPU) = "CPU"
Base.summary(gpu::GPU) = "GPU"

Base.show(io::IO, ::CPU) = print(io, "CPU()")
Base.show(io::IO, gpu::GPU) = print(io, "GPU(device=$(gpu.device))")

# ============================================================================
# Exports
# ============================================================================

export AbstractArchitecture, AbstractSerialArchitecture
export CPU, GPU
export device, array_type, architecture
export on_architecture
export is_gpu, has_cuda, synchronize, unsafe_free!
export is_gpu_array, create_array
export allocate_like, similar_zeros, copy_to_device
export launch_config, workgroup_size, launch!, KernelOperation

"""
    KernelOperation(kernel; ndrange_fn=args -> length(args[1]))

User-facing wrapper that couples a KernelAbstractions `@kernel` with a default
`ndrange` computation. Calling the object launches via `launch!`, so backend
logic always flows through the architecture abstraction.
"""
struct KernelOperation{K,F}
    kernel::K
    ndrange_fn::F  # Computes default ndrange from kernel arguments
end

default_ndrange_fn(args...) = length(args[1])

KernelOperation(kernel::K; ndrange_fn=default_ndrange_fn) where {K} = KernelOperation{K, typeof(ndrange_fn)}(kernel, ndrange_fn)

function (op::KernelOperation)(arch_or_data, args...; ndrange=nothing, dependencies=nothing, wait::Bool=true, kwargs...)
    effective_ndrange = isnothing(ndrange) ? op.ndrange_fn(args...) : ndrange
    return launch!(arch_or_data, op.kernel, args...; ndrange=effective_ndrange,
                   dependencies=dependencies, wait=wait, kwargs...)
end
