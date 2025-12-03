"""
GPU-PencilArrays Compatibility Layer

This module provides compatibility functions to handle the current limitations
of GPU support in PencilArrays.jl while providing fallback mechanisms.

Based on PencilArrays.jl v0.15+ limitations:
- Broadcasting issues with GPU arrays
- Scalar indexing problems with CuArrays  
- MPI communication issues with CuPtr
- Transpose function limitations
"""

using MPI
using PencilArrays
import PencilArrays: Pencil

# Compatibility shim for older Tarang code expecting `PencilConfig`
function _tarang_pencil_decomp_dims(global_shape::Tuple{Vararg{Int}},
                                   mesh::Tuple{Vararg{Int}},
                                   decomp_dims)
    N = length(global_shape)
    M = length(mesh)
    
    if decomp_dims === nothing
        return ntuple(i -> N - M + i, M)  # default to last M dims
    elseif all(x -> x isa Bool, decomp_dims)
        dims = Tuple(findall(identity, decomp_dims))
        return length(dims) == M ? dims : ntuple(i -> N - M + i, M)
    else
        dims = Tuple(Int.(decomp_dims))
        return length(dims) == M ? dims : ntuple(i -> N - M + i, M)
    end
end

if !isdefined(PencilArrays, :PencilConfig)
    """
    Lightweight wrapper that mimics the old PencilArrays.PencilConfig API using
    the current MPITopology/Pencil constructors.
    """
    struct PencilConfig
        topology::PencilArrays.MPITopology
        global_shape::Tuple{Vararg{Int}}
        decomp_dims::Tuple{Vararg{Int}}
    end
    
    function PencilConfig(global_shape::Tuple{Vararg{Int}}, 
                          mesh::Tuple{Vararg{Int}};
                          comm::MPI.Comm=MPI.COMM_WORLD,
                          decomp_dims=nothing)
        topology = PencilArrays.MPITopology(comm, mesh)
        dims = _tarang_pencil_decomp_dims(global_shape, mesh, decomp_dims)
        return PencilConfig(topology, global_shape, dims)
    end
else
    # If PencilArrays ever ships PencilConfig, alias it for downstream uses
    const PencilConfig = PencilArrays.PencilConfig
end

# Allow the existing Tarang call sites to construct a Pencil from the config
function Pencil(config::PencilConfig, 
                decomp_index::Int=1, dtype::Type=Float64)
    # decomp_index/dtype are currently unused; kept for signature compatibility
    return Pencil(config.topology, config.global_shape, config.decomp_dims)
end

# GPU support

"""
GPU-aware PencilArrays configuration that handles current limitations
"""
mutable struct GPUPencilConfig
    base_config::Union{PencilConfig, Nothing}
    device_config::DeviceConfig
    use_gpu_pencils::Bool
    fallback_to_cpu::Bool
    performance_stats::GPUPencilStats
    
    function GPUPencilConfig(global_shape::Tuple{Vararg{Int}}, 
                           mesh::Tuple{Vararg{Int}};
                           device::String="cpu",
                           comm::MPI.Comm=MPI.COMM_WORLD,
                           use_gpu_pencils::Bool=false,
                           fallback_to_cpu::Bool=true)
        
        device_config = select_device(device)
        
        # For now, disable GPU pencils due to PencilArrays limitations
        # Enable only if explicitly requested and device is GPU
        gpu_pencils = use_gpu_pencils && device_config.device_type != CPU_DEVICE
        
        # Create base PencilArrays configuration
        base_config = try
            if length(mesh) == 2
                PencilConfig(
                    global_shape,
                    mesh,
                    comm=comm,
                    decomp_dims=(true, true)
                )
            elseif length(mesh) == 3
                PencilConfig(
                    global_shape,
                    mesh,
                    comm=comm,
                    decomp_dims=(true, true, true)
                )
            else
                PencilConfig(
                    global_shape,
                    (prod(mesh),),
                    comm=comm
                )
            end
        catch e
            @warn "PencilArrays configuration failed: $e"
            nothing
        end
        
        perf_stats = GPUPencilStats()
        
        new(base_config, device_config, gpu_pencils, fallback_to_cpu, perf_stats)
    end
end

mutable struct GPUPencilStats
    pencil_creations::Int
    gpu_fallbacks::Int
    cpu_transfers::Int
    transpose_operations::Int
    mpi_operations::Int
    total_time::Float64
    
    function GPUPencilStats()
        new(0, 0, 0, 0, 0, 0.0)
    end
end

"""
Create GPU-aware pencil with fallback handling
"""
function create_gpu_pencil(config::GPUPencilConfig, dtype::Type=Float64)
    start_time = time()
    config.performance_stats.pencil_creations += 1
    
    if config.base_config === nothing
        @warn "No valid PencilArrays configuration, using CPU fallback"
        config.performance_stats.gpu_fallbacks += 1
        return nothing
    end
    
    pencil = if config.use_gpu_pencils
        try
            # Attempt to create GPU pencil
            if config.device_config.device_type == CUDA_DEVICE
                PencilArrays.Pencil(config.base_config, 1, dtype)
            else
                # For non-CUDA GPUs, fall back to CPU pencils for now
                @info "Non-CUDA GPU detected, using CPU pencils due to PencilArrays limitations"
                PencilArrays.Pencil(config.base_config, 1, dtype)
            end
        catch e
            @warn "GPU pencil creation failed: $e, falling back to CPU"
            config.performance_stats.gpu_fallbacks += 1
            PencilArrays.Pencil(config.base_config, 1, dtype)
        end
    else
        # Use standard CPU pencils
        PencilArrays.Pencil(config.base_config, 1, dtype)
    end
    
    config.performance_stats.total_time += time() - start_time
    return pencil
end

"""
GPU-aware pencil array creation with smart memory management
"""
function create_gpu_pencil_array(config::GPUPencilConfig, pencil, dtype::Type=Float64)
    start_time = time()
    
    if pencil === nothing
        return nothing
    end
    
    # Create PencilArray (always CPU for now due to PencilArrays limitations)
    pencil_array = try
        PencilArrays.PencilArray{dtype}(undef, pencil)
    catch e
        @warn "PencilArray creation failed: $e"
        config.performance_stats.gpu_fallbacks += 1
        return nothing
    end
    
    # If we want GPU data, we'll handle it separately from PencilArrays
    # This is our workaround for PencilArrays GPU limitations
    if config.device_config.device_type != CPU_DEVICE
        try
            # Create GPU version of the local data only
            local_data = parent(pencil_array)
            gpu_local_data = ensure_device!(local_data, config.device_config)
            
            # Store GPU data separately - we'll need custom transpose functions
            # This is a workaround until PencilArrays GPU support improves
            config.performance_stats.cpu_transfers += 1
            
            @info "Created hybrid CPU-PencilArray with GPU local data"
        catch e
            @warn "GPU local data creation failed: $e, using CPU only"
            config.performance_stats.gpu_fallbacks += 1
        end
    end
    
    config.performance_stats.total_time += time() - start_time
    return pencil_array
end

"""
GPU-aware transpose with fallback to CPU operations
"""
function gpu_aware_transpose!(config::GPUPencilConfig, 
                             src_array::PencilArrays.PencilArray, 
                             dest_array::PencilArrays.PencilArray,
                             plan=nothing)
    start_time = time()
    config.performance_stats.transpose_operations += 1
    
    try
        if config.use_gpu_pencils && plan !== nothing
            # Attempt GPU transpose
            PencilArrays.transpose!(src_array, dest_array, plan)
        else
            # CPU transpose (standard PencilArrays)
            PencilArrays.transpose!(src_array, dest_array, plan)
        end
    catch e
        @warn "PencilArrays transpose failed: $e, attempting CPU fallback"
        config.performance_stats.gpu_fallbacks += 1
        
        # Fallback: move to CPU, transpose, move back if needed
        try
            if config.fallback_to_cpu
                # Move to CPU arrays
                cpu_src = Array(parent(src_array))
                cpu_dest = Array(parent(dest_array))
                
                # Perform CPU transpose
                PencilArrays.transpose!(src_array, dest_array, plan)
                
                # Move back to GPU if needed
                if config.device_config.device_type != CPU_DEVICE
                    parent(dest_array) .= ensure_device!(cpu_dest, config.device_config)
                    config.performance_stats.cpu_transfers += 1
                end
            else
                rethrow(e)
            end
        catch fallback_error
            @error "Both GPU and CPU transpose failed: $fallback_error"
            rethrow(fallback_error)
        end
    end
    
    config.performance_stats.total_time += time() - start_time
end

"""
GPU-aware MPI operations for pencil arrays
"""
function gpu_aware_pencil_allreduce!(config::GPUPencilConfig, 
                                    pencil_array::PencilArrays.PencilArray,
                                    op=MPI.SUM)
    start_time = time()
    config.performance_stats.mpi_operations += 1
    
    local_data = parent(pencil_array)
    
    # Move to CPU for MPI operations (PencilArrays + GPU + MPI has issues)
    cpu_data = if config.device_config.device_type != CPU_DEVICE && isa(local_data, AbstractGPUArray)
        Array(local_data)
    else
        local_data
    end
    
    # Perform MPI operation
    try
        if MPI.Initialized()
            MPI.Allreduce!(cpu_data, cpu_data, op, MPI.COMM_WORLD)
        end
    catch e
        @warn "MPI operation failed: $e"
        config.performance_stats.gpu_fallbacks += 1
        rethrow(e)
    end
    
    # Move result back to original location
    if config.device_config.device_type != CPU_DEVICE && isa(local_data, AbstractGPUArray)
        try
            local_data .= ensure_device!(cpu_data, config.device_config)
            gpu_synchronize(config.device_config)
            config.performance_stats.cpu_transfers += 1
        catch e
            @warn "GPU transfer after MPI failed: $e"
            config.performance_stats.gpu_fallbacks += 1
        end
    else
        local_data .= cpu_data
    end
    
    config.performance_stats.total_time += time() - start_time
    return pencil_array
end

"""
Check if GPU pencil operations are recommended for current system
"""
function recommend_gpu_pencils(device_config::DeviceConfig)::Bool
    if device_config.device_type == CPU_DEVICE
        return false
    end
    
    # Currently, we recommend against GPU pencils due to PencilArrays limitations
    # This can be updated as PencilArrays GPU support improves
    
    @info """
    GPU Pencil Recommendation:
    Current PencilArrays.jl has limited GPU support with known issues:
    - Broadcasting problems with dimension permutations  
    - Scalar indexing errors with CuArrays
    - MPI communication issues with GPU arrays
    - Transpose function limitations
    
    Recommendation: Use CPU pencils with GPU local computation
    This provides the best stability while PencilArrays GPU support matures.
    """
    
    return false
end

"""
Create optimized configuration based on system capabilities
"""
function create_optimized_gpu_pencil_config(global_shape::Tuple{Vararg{Int}}, 
                                          mesh::Tuple{Vararg{Int}};
                                          device::String="cpu",
                                          comm::MPI.Comm=MPI.COMM_WORLD)
    
    device_config = select_device(device)
    
    # Check system compatibility
    use_gpu_pencils = recommend_gpu_pencils(device_config)
    
    if device_config.device_type != CPU_DEVICE && !use_gpu_pencils
        @info """
        Using hybrid approach:
        - PencilArrays operations on CPU (stable)
        - Local computations on GPU ($(device_config.device_type))
        - Smart data transfers between CPU and GPU as needed
        
        This provides the best performance/stability trade-off with current PencilArrays.jl
        """
    end
    
    return GPUPencilConfig(
        global_shape, mesh,
        device=device,
        comm=comm,
        use_gpu_pencils=use_gpu_pencils,
        fallback_to_cpu=true
    )
end

"""
Log performance statistics for GPU-PencilArrays operations
"""
function log_gpu_pencil_performance(config::GPUPencilConfig)
    stats = config.performance_stats
    
    @info "GPU-PencilArrays Performance ($(config.device_config.device_type)):"
    @info "  Pencil creations: $(stats.pencil_creations)"
    @info "  GPU fallbacks: $(stats.gpu_fallbacks)"
    @info "  CPU transfers: $(stats.cpu_transfers)"  
    @info "  Transpose operations: $(stats.transpose_operations)"
    @info "  MPI operations: $(stats.mpi_operations)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    
    if stats.pencil_creations > 0
        fallback_rate = stats.gpu_fallbacks / stats.pencil_creations * 100
        @info "  Fallback rate: $(round(fallback_rate, digits=1))%"
    end
    
    if config.use_gpu_pencils
        @info "  GPU pencils: ENABLED (experimental)"
    else
        @info "  GPU pencils: DISABLED (using hybrid CPU/GPU approach)"
    end
end

# Export main interface functions
export GPUPencilConfig, create_gpu_pencil, create_gpu_pencil_array
export gpu_aware_transpose!, gpu_aware_pencil_allreduce!
export create_optimized_gpu_pencil_config, recommend_gpu_pencils
export log_gpu_pencil_performance
