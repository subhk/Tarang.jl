"""
Distributor class for parallel distribution and transformations

Translated from dedalus/core/distributor.py with MPI and PencilArrays integration
"""

using MPI
using PencilArrays
using LinearAlgebra

struct Layout
    dist::Any
    local_shape::Tuple{Vararg{Int}}
    global_shape::Tuple{Vararg{Int}}
end

# Performance tracking structure for distributor
mutable struct DistributorPerformanceStats
    total_time::Float64
    pencil_creations::Int
    layout_creations::Int
    mpi_operations::Int
    gpu_transfers::Int
    cache_hits::Int
    cache_misses::Int
    
    function DistributorPerformanceStats()
        new(0.0, 0, 0, 0, 0, 0, 0)
    end
end

mutable struct Distributor
    comm::MPI.Comm
    size::Int
    rank::Int
    mesh::Union{Nothing, Tuple{Vararg{Int}}}
    coordsys::CoordinateSystem  # Primary coordinate system (for backward compatibility)
    coordsystems::Tuple{Vararg{CoordinateSystem}}  # All coordinate systems
    coords::Tuple{Vararg{Coordinate}}  # All coordinates from all systems
    dim::Int  # Total dimension
    dtype::Type

    # PencilArrays integration
    use_pencil_arrays::Bool  # Flag to enable/disable PencilArrays for MPI parallelization
    pencil_config::Union{Nothing, PencilConfig}
    transforms::Vector{Any}

    # GPU-PencilArrays compatibility
    gpu_pencil_config::Union{Nothing, GPUPencilConfig}

    # Layout cache
    layouts::Dict{Any, Layout}

    # GPU support
    device_config::DeviceConfig
    gpu_memory_pool::Vector{AbstractArray}
    performance_stats::DistributorPerformanceStats

    # Multi-GPU support (optional)
    multi_gpu_config::Union{Nothing, MultiGPUConfig}
    
    function Distributor(coordsys::CoordinateSystem; 
                        comm::MPI.Comm=MPI.COMM_WORLD, 
                        mesh::Union{Nothing, Tuple{Vararg{Int}}}=nothing,
                        dtype::Type=Float64,
                        device::String="cpu")
        
        size = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        
        if mesh === nothing
            # Auto-generate optimal mesh based on coordinate system dimension
            if isa(coordsys, CartesianCoordinates)
                if coordsys.dim == 1
                    mesh = (size,)
                elseif coordsys.dim == 2
                    mesh = create_2d_process_mesh(size)
                elseif coordsys.dim == 3
                    mesh = create_3d_process_mesh(size)
                else
                    mesh = (size,)  # Fallback for higher dimensions
                end
            else
                mesh = (size,)  # Default for other coordinate systems
            end
        end
        
        # Validate mesh
        if prod(mesh) != size
            throw(ArgumentError("Mesh size $(prod(mesh)) does not match number of processes $size"))
        end
        
        # Build coordinate information following Dedalus pattern
        coordsystems = (coordsys,)  # Single coordinate system for now
        coords_tuple = coords(coordsys)  # Get coordinates from the coordinate system
        total_dim = coordsys.dim  # Total dimension
        
        # Initialize empty structures
        # Enable PencilArrays for MPI parallelization (always true for distributed runs)
        use_pencil_arrays = (size > 1)  # Use PencilArrays for MPI, not for serial runs
        pencil_config = nothing
        transforms = Any[]
        layouts = Dict{Any, Layout}()

        # Initialize GPU support
        device_config = select_device(device)
        gpu_memory_pool = AbstractArray[]
        perf_stats = DistributorPerformanceStats()

        # Initialize GPU-PencilArrays compatibility (but don't create config yet)
        gpu_pencil_config = nothing

        # Initialize multi-GPU config (optional - created when needed)
        multi_gpu_config = nothing

        new(comm, size, rank, mesh, coordsys, coordsystems, coords_tuple, total_dim, dtype,
            use_pencil_arrays, pencil_config, transforms, gpu_pencil_config, layouts,
            device_config, gpu_memory_pool, perf_stats, multi_gpu_config)
    end
end


function setup_pencil_arrays(dist::Distributor, global_shape::Tuple{Vararg{Int}})
    """Setup PencilArrays configuration for given global shape with GPU compatibility"""
    
    if length(global_shape) != length(dist.mesh)
        throw(ArgumentError("Global shape dimensions must match mesh dimensions"))
    end
    
    # Create standard PencilArrays configuration
    if length(dist.mesh) == 2
        # Both dimensions can be distributed
        decomp_dims = (true, true)
    else
        decomp_dims = ntuple(i -> true, length(dist.mesh))
    end
    
    dist.pencil_config = PencilConfig(
        global_shape,
        dist.mesh,
        comm=dist.comm,
        decomp_dims=decomp_dims
    )
    
    # Create GPU-PencilArrays compatibility configuration
    dist.gpu_pencil_config = create_optimized_gpu_pencil_config(
        global_shape, 
        dist.mesh,
        device=dist.device_config.device_type == CPU_DEVICE ? "cpu" : "cuda",  # Simplified for now
        comm=dist.comm
    )
    
    # Log recommendations
    if dist.device_config.device_type != CPU_DEVICE
        @info """GPU-PencilArrays Setup Complete:
        PencilArrays.jl currently has limitations with GPU support.
        Using hybrid approach:
        - Distributed operations: CPU (stable)
        - Local computations: GPU ($(dist.device_config.device_type))
        - Smart data transfers as needed
        
        This provides optimal performance/stability until PencilArrays GPU support matures.
        """
    end
    
    return dist.pencil_config
end

function create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}}, 
                      decomp_index::Int=1; dtype::Type=dist.dtype)
    """Create a pencil array with specified decomposition with GPU support"""
    
    start_time = time()
    
    if dist.pencil_config === nothing
        setup_pencil_arrays(dist, global_shape)
    end
    
    pencil = PencilArrays.Pencil(dist.pencil_config, decomp_index, dtype)
    
    # Move pencil data to GPU if needed
    if dist.device_config.device_type != CPU_DEVICE
        try
            # Convert pencil data to GPU arrays if possible
            # Note: This depends on PencilArrays GPU support implementation
            pencil_data = ensure_device!(pencil.data, dist.device_config)
            
            # Store in memory pool for management
            push!(dist.gpu_memory_pool, pencil_data)
            
            # Update performance stats
            dist.performance_stats.total_time += time() - start_time
            dist.performance_stats.pencil_creations += 1
            
            gpu_synchronize(dist.device_config)
        catch e
            @warn "GPU pencil creation failed, falling back to CPU: $e"
        end
    end
    
    return pencil
end

function get_layout(dist::Distributor, bases::Tuple{Vararg{Basis}}, dtype::Type=dist.dtype)
    """Get layout for given bases with GPU support"""
    
    start_time = time()
    
    key = (bases, dtype)
    if haskey(dist.layouts, key)
        # Cache hit
        dist.performance_stats.cache_hits += 1
        return dist.layouts[key]
    end
    
    # Cache miss
    dist.performance_stats.cache_misses += 1
    
    # Calculate global shape from bases
    global_shape = tuple([basis.meta.size for basis in bases]...)
    
    # Create pencil for this layout
    pencil = create_pencil(dist, global_shape, 1, dtype=dtype)
    local_shape = size(pencil)
    
    layout = Layout(pencil, local_shape, global_shape)
    dist.layouts[key] = layout
    
    # Update performance stats
    dist.performance_stats.total_time += time() - start_time
    dist.performance_stats.layout_creations += 1
    
    return layout
end

function Field(dist::Distributor; name::String="field", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    """Create a scalar field with GPU support"""
    field = ScalarField(dist, name, bases, dtype)
    
    # Move field data to GPU if configured
    if dist.device_config.device_type != CPU_DEVICE
        try
            field.data = ensure_device!(field.data, dist.device_config)
            push!(dist.gpu_memory_pool, field.data)
        catch e
            @warn "GPU field creation failed, using CPU: $e"
        end
    end
    
    return field
end

function VectorField(dist::Distributor, coordsys::CoordinateSystem; name::String="vector", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    """Create a vector field with GPU support"""
    field = VectorField(dist, coordsys, name, bases, dtype)
    
    # Move field components to GPU if configured
    if dist.device_config.device_type != CPU_DEVICE
        try
            for i in 1:length(field.components)
                field.components[i].data = ensure_device!(field.components[i].data, dist.device_config)
                push!(dist.gpu_memory_pool, field.components[i].data)
            end
        catch e
            @warn "GPU vector field creation failed, using CPU: $e"
        end
    end
    
    return field
end

function TensorField(dist::Distributor, coordsys::CoordinateSystem; name::String="tensor", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    """Create a tensor field with GPU support"""
    field = TensorField(dist, coordsys, name, bases, dtype)
    
    # Move tensor components to GPU if configured
    if dist.device_config.device_type != CPU_DEVICE
        try
            for i in 1:size(field.components, 1)
                for j in 1:size(field.components, 2)
                    field.components[i,j].data = ensure_device!(field.components[i,j].data, dist.device_config)
                    push!(dist.gpu_memory_pool, field.components[i,j].data)
                end
            end
        catch e
            @warn "GPU tensor field creation failed, using CPU: $e"
        end
    end
    
    return field
end

function local_grids(dist::Distributor, bases::Vararg{Basis}; scales=nothing)
    """
    Return local coordinate grids for the given bases.
    Following Dedalus implementation in distributor.py:294
    """
    scales = remedy_scales(dist, scales, length(bases))
    grids = []
    
    for (i, basis) in enumerate(bases)
        # Get scales for this basis
        axis_start = first_axis(dist, basis)
        axis_end = last_axis(dist, basis)
        basis_scales = scales[axis_start+1:axis_end+1]  # Julia 1-indexed
        
        # Get local grids from the basis
        basis_grids = local_grids(basis, dist, basis_scales)
        append!(grids, basis_grids)
    end
    
    return grids
end

function remedy_scales(dist::Distributor, scales, num_bases)
    """Process and validate scales parameter."""
    if scales === nothing
        return ones(Float64, num_bases)
    elseif isa(scales, Number)
        return fill(Float64(scales), num_bases)
    elseif isa(scales, AbstractArray)
        return Float64.(scales)
    else
        error("Invalid scales type: $(typeof(scales))")
    end
end

function remedy_scales(dist::Distributor, scales)
    """
    Remedy different scale inputs.
    Following Dedalus implementation in distributor.py:188-197
    """
    if scales === nothing
        scales = 1.0
    end
    
    if isa(scales, Number)
        scales = tuple(fill(Float64(scales), dist.dim)...)
    end
    
    scales = tuple(Float64.(scales)...)
    
    if any(s -> s <= 0, scales)
        throw(ArgumentError("Scales must be positive nonzero."))
    end
    
    return scales
end

function get_axis(dist::Distributor, coord::Coordinate)
    """
    Get axis index for a coordinate.
    Following Dedalus implementation in distributor.py:202
    """
    for (i, c) in enumerate(dist.coords)
        if c.coordsys == coord.coordsys && c.name == coord.name
            return i - 1  # 0-indexed like Python
        end
    end
    throw(ArgumentError("Coordinate $(coord.name) not found in distributor"))
end

function get_axis(dist::Distributor, coordsys::CoordinateSystem)
    """Get axis for coordinate system (uses first coordinate)."""
    return get_axis(dist, coords(coordsys)[1])
end

function get_basis_axis(dist::Distributor, basis::Basis)
    """
    Get axis index for a basis.
    Following Dedalus implementation in distributor.py:207
    """
    # In Dedalus: return self.get_axis(basis.coordsys.coords[0])
    basis_coords = coords(basis.meta.coordsys)
    return get_axis(dist, basis_coords[1])  # First coordinate of the basis's coordinate system
end

function first_axis(dist::Distributor, basis::Basis)
    """
    Get first axis index for a basis.
    Following Dedalus implementation in distributor.py:210
    """
    return get_basis_axis(dist, basis)
end

function last_axis(dist::Distributor, basis::Basis)
    """
    Get last axis index for a basis.
    Following Dedalus implementation in distributor.py:213
    """
    return first_axis(dist, basis) + basis.meta.dim - 1
end

# GPU+MPI communication helpers
function gather_array(dist::Distributor, local_array::AbstractArray)
    """Gather array from all processes with GPU support"""
    
    start_time = time()
    
    # Move to CPU for MPI operations if needed
    cpu_array = if dist.device_config.device_type != CPU_DEVICE && isa(local_array, AbstractGPUArray)
        Array(local_array)
    else
        local_array
    end
    
    result = MPI.Allgather(cpu_array, dist.comm)
    
    # Move result back to GPU if needed
    if dist.device_config.device_type != CPU_DEVICE
        try
            result = ensure_device!(result, dist.device_config)
            gpu_synchronize(dist.device_config)
        catch e
            @warn "GPU gather result transfer failed: $e"
        end
    end
    
    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time
    
    return result
end

function scatter_array(dist::Distributor, global_array::AbstractArray)
    """Scatter array to all processes with GPU support"""
    
    start_time = time()
    
    # Move to CPU for MPI operations if needed
    cpu_array = if dist.device_config.device_type != CPU_DEVICE && isa(global_array, AbstractGPUArray)
        Array(global_array)
    else
        global_array
    end
    
    local_size = div(length(cpu_array), dist.size)
    local_array = zeros(eltype(cpu_array), local_size)
    MPI.Scatter!(cpu_array, local_array, 0, dist.comm)
    
    # Move result to GPU if needed
    if dist.device_config.device_type != CPU_DEVICE
        try
            local_array = ensure_device!(local_array, dist.device_config)
            gpu_synchronize(dist.device_config)
        catch e
            @warn "GPU scatter result transfer failed: $e"
        end
    end
    
    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time
    
    return local_array
end

function allreduce_array(dist::Distributor, local_array::AbstractArray, op=MPI.SUM)
    """All-reduce operation on array with GPU support"""
    
    start_time = time()
    
    # Move to CPU for MPI operations if needed
    cpu_array = if dist.device_config.device_type != CPU_DEVICE && isa(local_array, AbstractGPUArray)
        Array(local_array)
    else
        local_array
    end
    
    result = similar(cpu_array)
    MPI.Allreduce!(cpu_array, result, op, dist.comm)
    
    # Move result back to GPU if needed
    if dist.device_config.device_type != CPU_DEVICE
        try
            result = ensure_device!(result, dist.device_config)
            gpu_synchronize(dist.device_config)
        catch e
            @warn "GPU allreduce result transfer failed: $e"
        end
    end
    
    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time
    
    return result
end

# GPU-specific distributor functions
function move_distributor_to_device!(dist::Distributor, device_config::DeviceConfig)
    """Move distributor data to specified device"""
    
    start_time = time()
    old_device = dist.device_config
    dist.device_config = device_config
    
    # Move memory pool arrays to new device
    new_pool = AbstractArray[]
    for array in dist.gpu_memory_pool
        try
            new_array = ensure_device!(Array(array), device_config)
            push!(new_pool, new_array)
        catch e
            @warn "Failed to move array to device: $e"
        end
    end
    dist.gpu_memory_pool = new_pool
    
    # Update cached layouts if they contain device-specific data
    for (key, layout) in dist.layouts
        if hasfield(typeof(layout.dist), :data)
            try
                layout.dist.data = ensure_device!(Array(layout.dist.data), device_config)
            catch e
                @warn "Failed to move layout data to device: $e"
            end
        end
    end
    
    # Update performance stats
    dist.performance_stats.gpu_transfers += 1
    dist.performance_stats.total_time += time() - start_time
    
    @info "Moved distributor from $(old_device.device_type) to $(device_config.device_type)"
    
    return dist
end

function clear_distributor_cache!(dist::Distributor)
    """Clear GPU caches and memory pool for distributor"""
    
    # Clear memory pool
    empty!(dist.gpu_memory_pool)
    
    # Clear layout cache
    empty!(dist.layouts)
    
    # Reset PencilArrays configuration if needed
    dist.pencil_config = nothing
    
    @info "Cleared distributor caches ($(dist.device_config.device_type))"
    
    return dist
end

function get_distributor_memory_info(dist::Distributor)
    """Get GPU memory usage information for distributor"""
    
    if dist.device_config.device_type != CPU_DEVICE
        memory_info = gpu_memory_info(dist.device_config)
        
        # Estimate memory used by distributor
        distributor_memory = 0
        
        for array in dist.gpu_memory_pool
            distributor_memory += sizeof(array)
        end
        
        # Add layout memory
        for (key, layout) in dist.layouts
            if hasfield(typeof(layout.dist), :data) && isa(layout.dist.data, AbstractArray)
                distributor_memory += sizeof(layout.dist.data)
            end
        end
        
        return (
            total_memory = memory_info.total,
            available_memory = memory_info.available,
            used_memory = memory_info.used,
            distributor_memory = distributor_memory,
            memory_utilization = distributor_memory / memory_info.total * 100,
            pool_arrays = length(dist.gpu_memory_pool),
            cached_layouts = length(dist.layouts)
        )
    else
        return (
            total_memory = typemax(Int64),
            available_memory = typemax(Int64),
            used_memory = 0,
            distributor_memory = 0,
            memory_utilization = 0.0,
            pool_arrays = length(dist.gpu_memory_pool),
            cached_layouts = length(dist.layouts)
        )
    end
end

function log_distributor_performance(dist::Distributor)
    """Log distributor performance statistics"""
    
    stats = dist.performance_stats
    
    @info "Distributor performance ($(dist.device_config.device_type)):"
    @info "  Pencil creations: $(stats.pencil_creations)"
    @info "  Layout creations: $(stats.layout_creations)"
    @info "  MPI operations: $(stats.mpi_operations)"
    @info "  GPU transfers: $(stats.gpu_transfers)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"
    
    # Memory usage
    mem_info = get_distributor_memory_info(dist)
    @info "  GPU memory usage: $(round(mem_info.distributor_memory / 1024^2, digits=2)) MB ($(round(mem_info.memory_utilization, digits=1))%)"
    @info "  Memory pool: $(mem_info.pool_arrays) arrays, Cached layouts: $(mem_info.cached_layouts)"
end

# Enhanced MPI+GPU communication functions
function gpu_aware_alltoall(dist::Distributor, send_data::AbstractArray, recv_data::AbstractArray)
    """GPU-aware all-to-all communication"""
    
    start_time = time()
    
    # Move to CPU for MPI if needed
    send_cpu = if dist.device_config.device_type != CPU_DEVICE && isa(send_data, AbstractGPUArray)
        Array(send_data)
    else
        send_data
    end
    
    recv_cpu = if dist.device_config.device_type != CPU_DEVICE && isa(recv_data, AbstractGPUArray)
        Array(recv_data)
    else
        recv_data
    end
    
    # Perform MPI all-to-all
    MPI.Alltoall!(send_cpu, recv_cpu, dist.comm)
    
    # Move back to GPU if needed
    if dist.device_config.device_type != CPU_DEVICE
        try
            recv_data .= ensure_device!(recv_cpu, dist.device_config)
            gpu_synchronize(dist.device_config)
        catch e
            @warn "GPU alltoall result transfer failed: $e"
            recv_data .= recv_cpu
        end
    else
        recv_data .= recv_cpu
    end
    
    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time
    
    return recv_data
end

function create_2d_process_mesh(nproc::Int)
    """Create optimal 2D process mesh for given number of processes"""
    # Find factors closest to square
    factors = []
    for i in 1:floor(Int, sqrt(nproc))
        if nproc % i == 0
            push!(factors, (i, nproc ÷ i))
        end
    end
    
    if isempty(factors)
        return (1, nproc)
    end
    
    # Choose factors closest to square
    best_factor = factors[end]
    return best_factor
end

function create_3d_process_mesh(nproc::Int)
    """Create optimal 3D process mesh for given number of processes"""
    # Simple heuristic for 3D decomposition
    if nproc <= 8
        # Small cases
        if nproc == 1
            return (1, 1, 1)
        elseif nproc == 2
            return (2, 1, 1)
        elseif nproc == 4
            return (2, 2, 1)
        elseif nproc == 8
            return (2, 2, 2)
        else
            return (nproc, 1, 1)
        end
    else
        # For larger cases, try to find good 3D factorization
        cube_root = round(Int, nproc^(1/3))
        
        # Search around cube root
        for i in max(1, cube_root-2):cube_root+2
            if nproc % i == 0
                remaining = nproc ÷ i
                sqrt_remaining = round(Int, sqrt(remaining))
                
                for j in max(1, sqrt_remaining-1):sqrt_remaining+1
                    if remaining % j == 0
                        k = remaining ÷ j
                        if i * j * k == nproc
                            return (i, j, k)
                        end
                    end
                end
            end
        end
        
        # Fallback to 2D decomposition in z-plane
        mesh_2d = create_2d_process_mesh(nproc)
        return (mesh_2d[1], mesh_2d[2], 1)
    end
end
