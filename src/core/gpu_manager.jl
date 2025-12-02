"""
GPU Management System for Tarang.jl
===================================

Comprehensive GPU compatibility layer supporting:
- CUDA (NVIDIA)
- AMD ROCm (AMDGPU) 
- Metal (Apple Silicon)
- CPU fallback

This module provides unified GPU operations for all Tarang components.
"""

using LinearAlgebra
using Random
using Statistics

# GPU support packages
using GPUArraysCore
using GPUArrays
using KernelAbstractions
using Adapt

# Optional GPU backends (loaded conditionally)
import Preferences
has_cuda = false
has_amdgpu = false
has_metal = false

try
    import CUDA
    if CUDA.functional()
        global has_cuda = true
        using CUDA
        import CUDA: cu
    end
catch
    @debug "CUDA not available"
end

try
    import AMDGPU
    if AMDGPU.functional()
        global has_amdgpu = true
        using AMDGPU
        import AMDGPU: roc
    end
catch
    @debug "AMDGPU not available"
end

try
    import Metal
    if Metal.functional()
        global has_metal = true
        using Metal
        import Metal: mtl
    end
catch
    @debug "Metal not available"
end

# Export device types and functions
export DeviceType, DeviceConfig
export CPU_DEVICE, GPU_CUDA, GPU_AMDGPU, GPU_METAL
export select_device, get_device_config, set_global_device!
export device_array, device_zeros, device_ones, device_rand, device_randn
export device_identity, device_similar, device_fill
export gpu_synchronize, available_gpu_memory, gpu_memory_info
export @device_kernel, device_reduce, device_sum, device_maximum, device_minimum
export ensure_device!, check_device_compatibility

# Device management
@enum DeviceType CPU_DEVICE GPU_CUDA GPU_AMDGPU GPU_METAL

struct DeviceConfig
    device_type::DeviceType
    device_id::Int
    memory_limit::Int64  # Memory limit in bytes
    
    function DeviceConfig(device_type::DeviceType=CPU_DEVICE, device_id::Int=0, memory_limit::Int64=0)
        if memory_limit == 0
            memory_limit = get_default_memory_limit(device_type, device_id)
        end
        new(device_type, device_id, memory_limit)
    end
end

# Global device configuration
global_device_config = Ref{DeviceConfig}(DeviceConfig())

function set_global_device!(device::String, device_id::Int=0)
    """Set global device configuration"""
    config = select_device(device, device_id)
    global_device_config[] = config
    @info "Global device set to: $(config.device_type) (ID: $(config.device_id))"
    return config
end

function get_device_config()
    """Get current global device configuration"""
    return global_device_config[]
end

# Device selection functions
function select_device(device::String, device_id::Int=0)
    """Select computational device based on user input"""
    device_lower = lowercase(device)
    
    if device_lower == "cpu"
        return DeviceConfig(CPU_DEVICE, device_id)
    elseif device_lower == "gpu" || device_lower == "cuda"
        if has_cuda
            CUDA.device!(device_id)
            return DeviceConfig(GPU_CUDA, device_id)
        elseif has_amdgpu
            AMDGPU.device!(device_id)
            return DeviceConfig(GPU_AMDGPU, device_id)
        elseif has_metal
            return DeviceConfig(GPU_METAL, device_id)
        else
            @warn "No GPU backend available, falling back to CPU"
            return DeviceConfig(CPU_DEVICE, 0)
        end
    elseif device_lower == "amdgpu" || device_lower == "rocm"
        if has_amdgpu
            AMDGPU.device!(device_id)
            return DeviceConfig(GPU_AMDGPU, device_id)
        else
            @warn "AMDGPU not available, falling back to CPU"
            return DeviceConfig(CPU_DEVICE, 0)
        end
    elseif device_lower == "metal"
        if has_metal
            return DeviceConfig(GPU_METAL, device_id)
        else
            @warn "Metal not available, falling back to CPU"
            return DeviceConfig(CPU_DEVICE, 0)
        end
    else
        @warn "Unknown device '$device', falling back to CPU"
        return DeviceConfig(CPU_DEVICE, 0)
    end
end

function get_default_memory_limit(device_type::DeviceType, device_id::Int)
    """Get default memory limit for device"""
    if device_type == CPU_DEVICE
        return typemax(Int64)  # Unlimited for CPU
    elseif device_type == GPU_CUDA && has_cuda
        try
            CUDA.device!(device_id)
            return Int64(CUDA.total_memory())
        catch
            return 8 * 1024^3  # Default 8GB
        end
    elseif device_type == GPU_AMDGPU && has_amdgpu
        return 8 * 1024^3  # Default 8GB (AMDGPU.jl may not have memory query)
    elseif device_type == GPU_METAL && has_metal
        return 8 * 1024^3  # Default 8GB (Metal.jl may not have memory query)
    else
        return 8 * 1024^3
    end
end

# Device-agnostic array operations
function device_array(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Move array to specified device"""
    if config.device_type == CPU_DEVICE
        return Array(A)
    elseif config.device_type == GPU_CUDA && has_cuda
        return cu(A)
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        return roc(A)
    elseif config.device_type == GPU_METAL && has_metal
        return mtl(A)
    else
        @warn "Device not available, using CPU"
        return Array(A)
    end
end

function device_zeros(T::Type, dims, config::DeviceConfig=get_device_config())
    """Create zero array on specified device"""
    if config.device_type == CPU_DEVICE
        return zeros(T, dims)
    elseif config.device_type == GPU_CUDA && has_cuda
        return CUDA.zeros(T, dims)
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        return AMDGPU.zeros(T, dims)
    elseif config.device_type == GPU_METAL && has_metal
        return Metal.zeros(T, dims)
    else
        @warn "Device not available, using CPU"
        return zeros(T, dims)
    end
end

function device_ones(T::Type, dims, config::DeviceConfig=get_device_config())
    """Create ones array on specified device"""
    if config.device_type == CPU_DEVICE
        return ones(T, dims)
    elseif config.device_type == GPU_CUDA && has_cuda
        return CUDA.ones(T, dims)
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        return AMDGPU.ones(T, dims)
    elseif config.device_type == GPU_METAL && has_metal
        return Metal.ones(T, dims)
    else
        @warn "Device not available, using CPU"
        return ones(T, dims)
    end
end

function device_rand(T::Type, dims, config::DeviceConfig=get_device_config())
    """Create random array on specified device"""
    if config.device_type == CPU_DEVICE
        return rand(T, dims)
    elseif config.device_type == GPU_CUDA && has_cuda
        return CUDA.rand(T, dims)
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        return AMDGPU.rand(T, dims)
    elseif config.device_type == GPU_METAL && has_metal
        return Metal.rand(T, dims)
    else
        @warn "Device not available, using CPU"
        return rand(T, dims)
    end
end

function device_randn(T::Type, dims, config::DeviceConfig=get_device_config())
    """Create random normal array on specified device"""
    if config.device_type == CPU_DEVICE
        return randn(T, dims)
    elseif config.device_type == GPU_CUDA && has_cuda
        return CUDA.randn(T, dims)
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        # AMDGPU may not have randn, use rand and transform
        return sqrt(-2 * log.(AMDGPU.rand(T, dims))) .* cos.(2π * AMDGPU.rand(T, dims))
    elseif config.device_type == GPU_METAL && has_metal
        # Metal may not have randn, use rand and transform
        return sqrt(-2 * log.(Metal.rand(T, dims))) .* cos.(2π * Metal.rand(T, dims))
    else
        @warn "Device not available, using CPU"
        return randn(T, dims)
    end
end

function device_identity(T::Type, n::Int, config::DeviceConfig=get_device_config())
    """Create identity matrix on specified device"""
    I_cpu = Matrix{T}(LinearAlgebra.I, n, n)
    return device_array(I_cpu, config)
end

function device_similar(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Create similar array on specified device"""
    return device_zeros(eltype(A), size(A), config)
end

function device_fill(x, dims, config::DeviceConfig=get_device_config())
    """Create filled array on specified device"""
    result = device_zeros(typeof(x), dims, config)
    result .= x
    return result
end

# Device operations and synchronization
function gpu_synchronize(config::DeviceConfig=get_device_config())
    """Synchronize GPU operations"""
    if config.device_type == GPU_CUDA && has_cuda
        CUDA.synchronize()
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        AMDGPU.synchronize()
    elseif config.device_type == GPU_METAL && has_metal
        Metal.synchronize()
    end
    # CPU operations are always synchronous
end

function available_gpu_memory(config::DeviceConfig=get_device_config())
    """Get available GPU memory in bytes"""
    if config.device_type == GPU_CUDA && has_cuda
        return CUDA.available_memory()
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        # AMDGPU memory query (simplified)
        return 8 * 1024^3  # Default to 8GB if unable to query
    elseif config.device_type == GPU_METAL && has_metal
        # Metal memory query (simplified) 
        return 8 * 1024^3  # Default to 8GB if unable to query
    else
        return typemax(Int64)  # Unlimited for CPU
    end
end

function gpu_memory_info(config::DeviceConfig=get_device_config())
    """Get detailed GPU memory information"""
    if config.device_type == GPU_CUDA && has_cuda
        return (
            available = CUDA.available_memory(),
            total = CUDA.total_memory(),
            used = CUDA.total_memory() - CUDA.available_memory()
        )
    else
        return (
            available = available_gpu_memory(config),
            total = config.memory_limit,
            used = 0
        )
    end
end

# Device-specific reductions and operations
function device_reduce(op, A::AbstractArray, config::DeviceConfig=get_device_config())
    """Device-optimized reduction operation"""
    if config.device_type == CPU_DEVICE
        return reduce(op, A)
    else
        # Use GPUArrays.jl for GPU reductions
        return reduce(op, A)
    end
end

function device_sum(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Device-optimized sum"""
    return sum(A)
end

function device_maximum(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Device-optimized maximum"""
    return maximum(A)
end

function device_minimum(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Device-optimized minimum"""
    return minimum(A)
end

# Utility functions
function ensure_device!(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Ensure array is on the correct device"""
    if config.device_type == CPU_DEVICE && isa(A, Array)
        return A
    elseif config.device_type == GPU_CUDA && has_cuda && isa(A, CUDA.CuArray)
        return A
    elseif config.device_type == GPU_AMDGPU && has_amdgpu && isa(A, AMDGPU.ROCArray)
        return A  
    elseif config.device_type == GPU_METAL && has_metal && isa(A, Metal.MtlArray)
        return A
    else
        return device_array(A, config)
    end
end

function check_device_compatibility(A::AbstractArray, config::DeviceConfig=get_device_config())
    """Check if array is compatible with device configuration"""
    if config.device_type == CPU_DEVICE
        return isa(A, Array)
    elseif config.device_type == GPU_CUDA && has_cuda
        return isa(A, CUDA.CuArray)
    elseif config.device_type == GPU_AMDGPU && has_amdgpu
        return isa(A, AMDGPU.ROCArray)
    elseif config.device_type == GPU_METAL && has_metal
        return isa(A, Metal.MtlArray)
    else
        return false
    end
end

# Kernel abstractions for custom kernels
macro device_kernel(expr)
    """Macro for device-agnostic kernel definitions"""
    quote
        @kernel function $(expr.args[1])($(expr.args[2:end]...))
            $(expr.args[end])
        end
    end
end

# High-level device management
function with_device(f, config::DeviceConfig)
    """Execute function with specific device configuration"""
    old_config = get_device_config()
    try
        global_device_config[] = config
        return f()
    finally
        global_device_config[] = old_config
    end
end

function benchmark_device_operations(config::DeviceConfig=get_device_config(), n::Int=1000)
    """Benchmark basic operations on device"""
    println("Benchmarking device: $(config.device_type)")
    
    # Array creation
    @time A = device_rand(Float64, (n, n), config)
    @time B = device_rand(Float64, (n, n), config)
    
    # Basic operations
    @time C = A .+ B
    @time D = A * B
    @time s = device_sum(A, config)
    
    gpu_synchronize(config)
    
    return (sum=s, norm=norm(D))
end

# Display device information
function show_device_info()
    """Display available devices and their capabilities"""
    println("=== Tarang.jl GPU Device Information ===")
    println()
    
    println("CPU: Always available")
    
    if has_cuda
        println("✓ CUDA: Available")
        try
            println("  - CUDA version: $(CUDA.version())")
            println("  - Devices: $(length(CUDA.devices()))")
            for (i, dev) in enumerate(CUDA.devices())
                println("    Device $i: $(CUDA.name(dev))")
                println("      Memory: $(CUDA.total_memory(dev) ÷ 1024^3) GB")
            end
        catch e
            println("  - Error getting CUDA info: $e")
        end
    else
        println("✗ CUDA: Not available")
    end
    
    if has_amdgpu
        println("✓ AMDGPU: Available") 
        try
            println("  - Devices: $(AMDGPU.device_count())")
        catch e
            println("  - Error getting AMDGPU info: $e")
        end
    else
        println("✗ AMDGPU: Not available")
    end
    
    if has_metal
        println("✓ Metal: Available")
        try
            println("  - Metal GPU available")
        catch e
            println("  - Error getting Metal info: $e")
        end
    else
        println("✗ Metal: Not available")
    end
    
    println()
    println("Current global device: $(get_device_config().device_type)")
end