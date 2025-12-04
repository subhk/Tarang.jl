"""
Parallel computing utilities

MPI and parallel processing support
"""

using MPI
using PencilArrays

# MPI utilities
function ensure_mpi_initialized()
    """Ensure MPI is initialized"""
    if !MPI.Initialized()
        MPI.Init()
    end
end

function get_mpi_info()
    """Get MPI communicator information"""
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        return (comm=comm, rank=rank, size=size)
    else
        return (comm=nothing, rank=0, size=1)
    end
end

function parallel_print(msg::String, root::Int=0)
    """Print message from specified rank only"""
    info = get_mpi_info()
    if info.rank == root
        println(msg)
    end
end

function barrier()
    """Synchronize all MPI processes"""
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

# Directory creation
function parallel_mkdir(path::String; mode::Int=0o755)
    """Create directory in parallel-safe way"""
    info = get_mpi_info()
    
    if info.rank == 0
        if !isdir(path)
            mkpath(path)
        end
    end
    
    # Synchronize to ensure directory is created before other ranks proceed
    barrier()
    
    return path
end

# Load balancing utilities
function distribute_work(total_work::Int, num_workers::Int, worker_id::Int)
    """Distribute work among workers"""
    if worker_id < 0 || worker_id >= num_workers
        throw(ArgumentError("Worker ID must be in range [0, $(num_workers-1)]"))
    end
    
    work_per_worker = div(total_work, num_workers)
    remainder = total_work % num_workers
    
    if worker_id < remainder
        start_idx = worker_id * (work_per_worker + 1) + 1
        end_idx = start_idx + work_per_worker
    else
        start_idx = worker_id * work_per_worker + remainder + 1
        end_idx = start_idx + work_per_worker - 1
    end
    
    return (start_idx, end_idx)
end

function get_local_work(total_work::Int)
    """Get work range for current MPI rank"""
    info = get_mpi_info()
    return distribute_work(total_work, info.size, info.rank)
end

# Collective operations
function parallel_sum(value::Number)
    """Sum value across all MPI processes"""
    if MPI.Initialized()
        return MPI.Allreduce(value, +, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_max(value::Number)
    """Find maximum value across all MPI processes"""
    if MPI.Initialized()
        return MPI.Allreduce(value, max, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_min(value::Number)
    """Find minimum value across all MPI processes"""
    if MPI.Initialized()
        return MPI.Allreduce(value, min, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_all(value::Bool)
    """Logical AND across all MPI processes"""
    if MPI.Initialized()
        return MPI.Allreduce(value, &, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_any(value::Bool)
    """Logical OR across all MPI processes"""
    if MPI.Initialized()
        return MPI.Allreduce(value, |, MPI.COMM_WORLD)
    else
        return value
    end
end

# GPU-aware array operations
function gather_array(local_array::AbstractArray, root::Int=0)
    """Gather arrays from all processes to root with GPU support"""
    if MPI.Initialized()
        # Move to CPU for MPI operations if needed
        cpu_array = if device_config.device_type != CPU_DEVICE 
            Array(local_array)
        else
            local_array
        end
        
        result = MPI.Gather(cpu_array, root, MPI.COMM_WORLD)
        
        # Move result back to GPU if needed
        if device_config.device_type != CPU_DEVICE && result !== nothing
            try
                result = [arr for arr in result]
            catch e
                @warn "GPU gather result transfer failed: $e"
            end
        end
        
        return result
    else
        return [local_array]
    end
end

function allgather_array(local_array::AbstractArray)
    """Gather arrays from all processes to all processes with GPU support"""
    if MPI.Initialized()
        # Move to CPU for MPI operations if needed
        cpu_array = if device_config.device_type != CPU_DEVICE 
            Array(local_array)
        else
            local_array
        end
        
        result = MPI.Allgather(cpu_array, MPI.COMM_WORLD)
        
        # Move result back to GPU if needed
        if device_config.device_type != CPU_DEVICE
            try
                result = [arr for arr in result]
            catch e
                @warn "GPU allgather result transfer failed: $e"
            end
        end
        
        return result
    else
        return [local_array]
    end
end

function broadcast_array(array::AbstractArray, root::Int=0)
    """Broadcast array from root to all processes with GPU support"""
    if MPI.Initialized()
        # Move to CPU for MPI operations if needed
        cpu_array = if device_config.device_type != CPU_DEVICE 
            Array(array)
        else
            array
        end
        
        result = MPI.Bcast(cpu_array, root, MPI.COMM_WORLD)
        
        # Move result back to GPU if needed
        if device_config.device_type != CPU_DEVICE
            try
                result = result
            catch e
                @warn "GPU broadcast result transfer failed: $e"
            end
        end
        
        return result
    else
        return array
    end
end

function scatter_array(global_array::AbstractArray, root::Int=0)
    """Scatter array from root to all processes with GPU support"""
    if MPI.Initialized()
        info = get_mpi_info()
        
        # Move to CPU for MPI operations if needed
        cpu_array = if device_config.device_type != CPU_DEVICE 
            Array(global_array)
        else
            global_array
        end
        
        local_size = div(length(cpu_array), info.size)
        local_array = similar(cpu_array, local_size)
        MPI.Scatter!(cpu_array, local_array, root, MPI.COMM_WORLD)
        
        # Move result to GPU if needed
        if device_config.device_type != CPU_DEVICE
            try
                local_array = local_array
            catch e
                @warn "GPU scatter result transfer failed: $e"
            end
        end
        
        return local_array
    else
        return global_array
    end
end

# GPU-aware performance monitoring
mutable struct PerformanceTimer
    name::String
    start_time::Float64
    total_time::Float64
    call_count::Int
    
    
    function PerformanceTimer(name::String)
        new(name, 0.0, 0.0, 0, device_config, Float64[])
    end
end

function start_timer!(timer::PerformanceTimer)
    """Start performance timer with GPU synchronization"""
    if timer !== nothing && timerfalse
    end
    timer.start_time = time()
end

function stop_timer!(timer::PerformanceTimer)
    """Stop performance timer and accumulate time with GPU synchronization"""
    if timer !== nothing && timerfalse
    end
    
    elapsed = time() - timer.start_time
    timer.total_time += elapsed
    timer.call_count += 1
    
    # Store GPU timing if applicable
    if timer !== nothing && timerfalse
    end
    
    return elapsed
end

function reset_timer!(timer::PerformanceTimer)
    """Reset performance timer"""
    timer.total_time = 0.0
    timer.call_count = 0
    timer.start_time = 0.0
end

function average_time(timer::PerformanceTimer)
    """Get average time per call"""
    return timer.call_count > 0 ? timer.total_time / timer.call_count : 0.0
end

function timer_stats(timer::PerformanceTimer)
    """Get timer statistics with GPU metrics"""
    stats = Dict(
        "name" => timer.name,
        "total_time" => timer.total_time,
        "call_count" => timer.call_count,
        "average_time" => average_time(timer),
    )
    
        stats["device_type"] = timer.device_type
    end
    
    return stats
end

# GPU-aware parallel timing context
struct TimedRegion
    timer::PerformanceTimer
    
    function TimedRegion(name::String)
        timer = PerformanceTimer(name, device_config=device_config)
        start_timer!(timer)
        new(timer)
    end
end

function Base.close(region::TimedRegion)
    """Close timed region and record elapsed time"""
    return stop_timer!(region.timer)
end

# Macro for timing code blocks
macro timed(name, expr)
    quote
        region = TimedRegion($(esc(name)))
        try
            $(esc(expr))
        finally
            elapsed = close(region)
            parallel_print("Timed region '$(region.timer.name)': $(elapsed) seconds")
        end
    end
end

# Profiling wrapper
struct ProfileWrapper
    enabled::Bool
    profile_data::Dict{String, Any}
    
    function ProfileWrapper(enabled::Bool=false)
        new(enabled, Dict{String, Any}())
    end
end

function start_profiling!(wrapper::ProfileWrapper, name::String)
    """Start profiling section"""
    if wrapper.enabled
        wrapper.profile_data[name] = time()
    end
end

function stop_profiling!(wrapper::ProfileWrapper, name::String)
    """Stop profiling section and record time"""
    if wrapper.enabled && haskey(wrapper.profile_data, name)
        elapsed = time() - wrapper.profile_data[name]
        wrapper.profile_data[name] = elapsed
        return elapsed
    end
    return 0.0
end

function profile_report(wrapper::ProfileWrapper)
    """Generate profiling report"""
    if !wrapper.enabled
        return "Profiling disabled"
    end
    
    report = "Profiling Report:\n"
    total_time = sum(values(wrapper.profile_data))
    
    for (name, time) in sort(collect(wrapper.profile_data), by=x->x[2], rev=true)
        percentage = (time / total_time) * 100
        report *= "  $(name): $(round(time, digits=4))s ($(round(percentage, digits=1))%)\n"
    end
    
    return report
end

# Process mesh utilities
function create_process_mesh(total_procs::Int, dimensions::Int=2)
    """Create optimal process mesh for given number of processes"""
    
    if dimensions == 1
        return (total_procs,)
    elseif dimensions == 2
        return create_2d_process_mesh(total_procs)
    elseif dimensions == 3
        return create_3d_process_mesh(total_procs)
    else
        throw(ArgumentError("Process mesh creation for $dimensions dimensions not implemented"))
    end
end

function create_2d_process_mesh(total_procs::Int)
    """Create optimal 2D process mesh"""
    # Find factors close to sqrt(total_procs)
    sqrt_procs = Int(round(sqrt(total_procs)))
    
    for i in sqrt_procs:-1:1
        if total_procs % i == 0
            j = div(total_procs, i)
            return (i, j)
        end
    end
    
    # Fallback: use all processes in one dimension
    return (1, total_procs)
end

function create_3d_process_mesh(total_procs::Int)
    """Create optimal 3D process mesh"""
    
    # Handle special cases
    if total_procs == 1
        return (1, 1, 1)
    elseif total_procs == 2
        return (1, 1, 2)
    elseif total_procs == 4
        return (1, 2, 2)
    end
    
    # Try to find a near-cubic factorization
    cube_root = round(Int, total_procs^(1/3))
    
    # Search for factors starting from cube root
    best_mesh = (1, 1, total_procs)
    min_surface_area = typemax(Int)
    
    for nz in 1:total_procs
        if total_procs % nz != 0
            continue
        end
        
        remaining = div(total_procs, nz)
        
        for ny in 1:remaining
            if remaining % ny != 0
                continue
            end
            
            nx = div(remaining, ny)
            
            # Prefer more cubic arrangements (minimize surface area)
            surface_area = 2 * (nx * ny + ny * nz + nz * nx)
            
            if surface_area < min_surface_area
                min_surface_area = surface_area
                best_mesh = (nx, ny, nz)
            end
        end
    end
    
    return best_mesh
end

function optimize_3d_process_mesh(total_procs::Int, global_shape::Tuple{Int,Int,Int})
    """Create 3D process mesh optimized for given global shape"""
    
    nx_global, ny_global, nz_global = global_shape
    
    # Find all possible factorizations
    factorizations = []
    
    for nz_proc in 1:total_procs
        if total_procs % nz_proc != 0
            continue
        end
        
        remaining = div(total_procs, nz_proc)
        
        for ny_proc in 1:remaining
            if remaining % ny_proc != 0
                continue
            end
            
            nx_proc = div(remaining, ny_proc)
            
            # Calculate local sizes
            local_nx = div(nx_global, nx_proc)
            local_ny = div(ny_global, ny_proc)
            local_nz = div(nz_global, nz_proc)
            
            # Skip if any dimension doesn't divide evenly
            if nx_global % nx_proc != 0 || ny_global % ny_proc != 0 || nz_global % nz_proc != 0
                continue
            end
            
            # Calculate communication cost (surface area of local domains)
            comm_cost = 2 * (local_nx * local_ny + local_ny * local_nz + local_nz * local_nx)
            
            push!(factorizations, ((nx_proc, ny_proc, nz_proc), comm_cost))
        end
    end
    
    if isempty(factorizations)
        # Fallback to basic factorization if no even division possible
        return create_3d_process_mesh(total_procs)
    end
    
    # Choose factorization with minimum communication cost
    best_mesh, _ = minimum(factorizations, by=x -> x[2])
    return best_mesh
end

function validate_process_mesh(mesh::Tuple{Vararg{Int}}, total_procs::Int)
    """Validate that process mesh is compatible with total processes"""
    if prod(mesh) != total_procs
        throw(ArgumentError("Process mesh $(mesh) incompatible with $total_procs processes"))
    end
    return true
end

# Load balancing using PencilArrays for distributed computation
function balance_load(global_shape::Tuple{Vararg{Int}}, pencil_config::PencilConfig)
    """
    Balance load across processes using PencilArrays decomposition.
    
    Integrates with PencilArrays.jl for distributed spectral methods, following dedalus patterns
    but using Julia's PencilArrays infrastructure for parallelization.
    
    Args:
        global_shape: Global dimensions of the data
        pencil_config: PencilArrays configuration for domain decomposition
    
    Returns:
        Tuple of (local_starts, local_ends, local_sizes) for current process
    """
    info = get_mpi_info()
    
    if !MPI.Initialized() || info.size == 1
        # Serial case - process owns all data
        local_starts = ones(Int, length(global_shape))
        local_ends = collect(global_shape)
        local_sizes = collect(global_shape)
        return (local_starts, local_ends, local_sizes)
    end
    
    # Get PencilArrays decomposition information
    pencil = PencilArrays.pencil_from_shape(pencil_config, global_shape)
    local_range = PencilArrays.range_local(pencil)
    
    # Extract local domain information from PencilArrays
    local_starts = [r.start for r in local_range]
    local_ends = [r.stop for r in local_range]
    local_sizes = [length(r) for r in local_range]
    
    return (tuple(local_starts...), tuple(local_ends...), tuple(local_sizes...))
end

function balance_load(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}}, axis::Int=1)
    """
    Balance load across processes using PencilArrays with specified mesh.
    
    Creates optimal PencilArrays configuration based on dedalus-style mesh and applies
    load balancing through PencilArrays decomposition.
    
    Args:
        global_shape: Global dimensions of the data  
        mesh: Process mesh dimensions
        axis: Axis along which to distribute (1-based indexing)
    
    Returns:
        Tuple of (local_starts, local_ends, local_sizes) for current process
    """
    info = get_mpi_info()
    
    if !MPI.Initialized() || info.size == 1
        # Serial case - process owns all data
        local_starts = ones(Int, length(global_shape))
        local_ends = collect(global_shape)
        local_sizes = collect(global_shape)
        return (local_starts, local_ends, local_sizes)
    end
    
    # Validate mesh compatibility
    if prod(mesh) != info.size
        throw(ArgumentError("Process mesh $mesh incompatible with $(info.size) processes"))
    end
    
    # Create PencilArrays configuration from mesh specification
    pencil_config = create_pencilarray_config(global_shape, mesh, info.comm)
    
    # Use PencilArrays for load balancing
    return balance_load(global_shape, pencil_config)
end

function create_pencilarray_config(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}}, comm::MPI.Comm)
    """
    Create PencilArrays configuration from process mesh specification.
    
    Converts dedalus-style mesh specification to PencilArrays.jl configuration,
    following the project requirements to use PencilArrays for parallelization.
    """
    decomp_dims = ntuple(_ -> true, length(mesh))
    return PencilConfig(global_shape, mesh; comm=comm, decomp_dims=decomp_dims)
end

function calculate_block_distribution(global_size::Int, mesh_size::Int, coord::Int)
    """
    Calculate block distribution using dedalus ceil-based algorithm.
    
    Based on dedalus transposes.pyx:
    B = ceil(global_shape[axis] / pycomm.size)
    starts = minimum(B*ranks, global_shape[axis])
    ends = minimum(B*(ranks+1), global_shape[axis])
    """
    if mesh_size <= 0 || coord < 0 || coord >= mesh_size
        throw(ArgumentError("Invalid mesh_size=$mesh_size or coord=$coord"))
    end
    
    # Ceil-based block size (key dedalus algorithm)
    block_size = cld(global_size, mesh_size)  # ceiling division
    
    # Calculate start and end indices (convert to 1-based)
    start_idx = min(block_size * coord + 1, global_size)
    end_idx = min(block_size * (coord + 1), global_size)
    
    # Handle edge case where coord >= required processes
    if start_idx > global_size
        start_idx = global_size + 1
        end_idx = global_size
    end
    
    return (start_idx, end_idx)
end

function get_mesh_coordinates(rank::Int, mesh::Tuple{Vararg{Int}})
    """
    Convert linear MPI rank to multidimensional mesh coordinates.
    Uses row-major (C-style) ordering like dedalus.
    """
    coords = zeros(Int, length(mesh))
    remaining_rank = rank
    
    # Convert to mesh coordinates (row-major order)
    for i in length(mesh):-1:1
        coords[i] = remaining_rank % mesh[i]
        remaining_rank = div(remaining_rank, mesh[i])
    end
    
    return tuple(coords...)
end

function balance_load_dynamic(local_work_measurements::Vector{Float64})
    """
    Dynamic load balancing based on actual work measurements.
    
    Based on dedalus performance monitoring and adaptive distribution.
    Redistributes work based on measured computation times.
    """
    info = get_mpi_info()
    
    if !MPI.Initialized() || info.size == 1
        return local_work_measurements
    end
    
    # Gather work measurements from all processes
    all_measurements = MPI.Allgather(local_work_measurements, MPI.COMM_WORLD)
    
    # Calculate statistics
    total_work = sum(sum(measurements) for measurements in all_measurements)
    mean_work = total_work / info.size
    work_per_process = [sum(measurements) for measurements in all_measurements]
    
    # Identify load imbalance
    max_work = maximum(work_per_process)
    min_work = minimum(work_per_process)
    imbalance_ratio = max_work / max(min_work, 1e-10)  # Avoid division by zero
    
    parallel_print("Load balance analysis: max/min ratio = $(round(imbalance_ratio, digits=2))")
    
    # Rebalance if significant imbalance detected
    imbalance_threshold = 1.2  # 20% imbalance threshold
    
    if imbalance_ratio > imbalance_threshold
        parallel_print("Load imbalance detected, redistributing work...")
        
        # Simple redistribution algorithm - move work from heavy to light processes
        target_work = mean_work
        redistribution_plan = balance_work_redistribution(work_per_process, target_work, info.rank)
        
        return redistribution_plan
    else
        parallel_print("Load is well balanced")
        return local_work_measurements
    end
end

function balance_work_redistribution(work_per_process::Vector{Float64}, target_work::Float64, current_rank::Int)
    """
    Create redistribution plan to balance work across processes.
    Based on dedalus-style work migration algorithms.
    """
    num_processes = length(work_per_process)
    current_work = work_per_process[current_rank + 1]  # Convert to 1-based indexing
    
    # Calculate adjustment needed for current process
    work_adjustment = target_work - current_work
    
    # For this implementation, return adjusted target work
    # In practice, this would coordinate actual data movement
    adjusted_work = max(0.0, current_work + 0.5 * work_adjustment)  # Gradual adjustment
    
    return [adjusted_work]
end

function optimize_spectral_load_balance(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}})
    """
    Optimize load balancing specifically for spectral methods using PencilArrays.
    
    Based on dedalus distributor.py layout optimization but adapted for PencilArrays.jl
    Considers FFT communication patterns and PencilArrays optimization.
    """
    info = get_mpi_info()
    
    if !MPI.Initialized() || info.size == 1
        return global_shape
    end
    
    # Create PencilArrays configuration and analyze efficiency
    pencil_config = create_pencilarray_config(global_shape, mesh, info.comm)
    
    # Validate mesh for spectral method efficiency with PencilArrays
    validation_result = validate_pencilarray_mesh(global_shape, mesh, pencil_config)
    
    if !validation_result.is_optimal
        parallel_print("Warning: Process mesh may not be optimal for PencilArrays spectral methods")
        parallel_print("  $(validation_result.message)")
        
        if validation_result.suggested_mesh !== nothing
            parallel_print("  Suggested mesh: $(validation_result.suggested_mesh)")
        end
    end
    
    # Calculate PencilArrays-specific metrics
    pencil_metrics = analyze_pencilarray_performance(global_shape, pencil_config)
    
    parallel_print("PencilArrays load balance metrics:")
    parallel_print("  Decomposition efficiency: $(round(pencil_metrics.efficiency * 100, digits=1))%")
    parallel_print("  Memory per process: $(round(pencil_metrics.memory_per_process/1024/1024, digits=2)) MB")
    parallel_print("  Communication overhead: $(round(pencil_metrics.comm_overhead * 100, digits=1))%")
    
    return global_shape
end

function validate_pencilarray_mesh(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}}, pencil_config::PencilConfig)
    """
    Validate process mesh for PencilArrays spectral method efficiency.
    Extends dedalus validation with PencilArrays-specific considerations.
    """
    issues = String[]
    
    # Check 1: Avoid empty cores
    for (dim, (global_size, mesh_size)) in enumerate(zip(global_shape, mesh))
        if mesh_size > global_size
            push!(issues, "Mesh dimension $dim ($mesh_size) > global size ($global_size) - empty cores")
        end
    end
    
    # Check 2: PencilArrays decomposition efficiency
    try
        # Test PencilArrays decomposition
        pencil = PencilArrays.pencil_from_shape(pencil_config, global_shape)
        local_size = PencilArrays.size_local(pencil)
        
        # Check if local sizes are reasonable
        if any(s -> s < 4, local_size)
            push!(issues, "Local pencil dimensions too small (< 4) - inefficient for FFTs")
        end
        
        # Check load balance across dimensions
        global_elements = prod(global_shape)
        local_elements = prod(local_size)
        expected_elements = global_elements ÷ prod(mesh)
        
        efficiency = local_elements / expected_elements
        if efficiency < 0.8
            push!(issues, "Poor load balance - process utilization < 80%")
        end
        
    catch e
        push!(issues, "PencilArrays decomposition failed: $e")
    end
    
    # Check 3: Spectral method compatibility
    if length(mesh) >= 2
        # For 2D simulations, prefer decomposition in both horizontal directions
        # as specified in project requirements
        mesh_ratios = [mesh[i]/mesh[1] for i in 2:length(mesh)]
        if any(r -> r > 8 || r < 0.125, mesh_ratios)
            push!(issues, "Highly anisotropic mesh may not be optimal for 2D spectral methods")
        end
    end
    
    is_optimal = isempty(issues)
    message = isempty(issues) ? "Process mesh is optimal for PencilArrays spectral methods" : join(issues, "; ")
    
    # Suggest improvements if needed
    suggested_mesh = nothing
    if !is_optimal && length(global_shape) <= 3
        suggested_mesh = create_process_mesh(prod(mesh), length(mesh))
    end
    
    return MeshValidationResult(is_optimal, message, suggested_mesh)
end

struct PencilArrayMetrics
    efficiency::Float64
    memory_per_process::Float64
    comm_overhead::Float64
    local_shape::Tuple{Vararg{Int}}
end

function analyze_pencilarray_performance(global_shape::Tuple{Vararg{Int}}, pencil_config::PencilConfig)
    """
    Analyze PencilArrays performance metrics for the given configuration.
    """
    try
        # Create pencil decomposition
        pencil = PencilArrays.pencil_from_shape(pencil_config, global_shape)
        local_shape = PencilArrays.size_local(pencil)
        
        # Calculate efficiency metrics
        global_elements = prod(global_shape)
        local_elements = prod(local_shape)
        info = get_mpi_info()
        expected_elements = global_elements ÷ info.size
        
        efficiency = local_elements / expected_elements
        
        # Estimate memory usage (complex Float64)
        memory_per_process = local_elements * 16  # 16 bytes per complex number
        
        # Estimate communication overhead for PencilArrays transposes
        # Based on surface area of local domain
        surface_elements = sum(prod(local_shape) ÷ local_shape[i] for i in 1:length(local_shape))
        comm_overhead = surface_elements / local_elements
        
        return PencilArrayMetrics(efficiency, memory_per_process, comm_overhead, local_shape)
        
    catch e
        # Return default metrics if analysis fails
        return PencilArrayMetrics(0.0, 0.0, 1.0, ())
    end
end

struct MeshValidationResult
    is_optimal::Bool
    message::String
    suggested_mesh::Union{Tuple{Vararg{Int}}, Nothing}
end

function validate_spectral_mesh(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}})
    """
    Validate process mesh for spectral method efficiency.
    Based on dedalus performance guidelines.
    """
    issues = String[]
    
    # Check 1: Avoid empty cores
    for (dim, (global_size, mesh_size)) in enumerate(zip(global_shape, mesh))
        if mesh_size > global_size
            push!(issues, "Mesh dimension $dim ($mesh_size) > global size ($global_size) - empty cores")
        end
    end
    
    # Check 2: Prefer power-of-2 mesh dimensions
    for (dim, mesh_size) in enumerate(mesh)
        if mesh_size > 1 && !ispowerof2(mesh_size) && !is_small_prime_factor(mesh_size)
            push!(issues, "Mesh dimension $dim ($mesh_size) not power-of-2 or small prime - may reduce FFT efficiency")
        end
    end
    
    # Check 3: Isotropic mesh preference for multi-dimensional problems
    if length(mesh) > 1
        mesh_ratios = [mesh[i]/mesh[1] for i in 2:length(mesh)]
        if any(r -> r > 4 || r < 0.25, mesh_ratios)
            push!(issues, "Highly anisotropic mesh - consider more balanced dimensions")
        end
    end
    
    is_optimal = isempty(issues)
    message = isempty(issues) ? "Process mesh is optimal for spectral methods" : join(issues, "; ")
    
    # Suggest improvements if needed
    suggested_mesh = nothing
    if !is_optimal && length(global_shape) <= 3
        suggested_mesh = create_process_mesh(prod(mesh), length(mesh))
    end
    
    return MeshValidationResult(is_optimal, message, suggested_mesh)
end

function is_small_prime_factor(n::Int)
    """Check if number has only small prime factors (2, 3, 5)"""
    while n % 2 == 0; n = div(n, 2); end
    while n % 3 == 0; n = div(n, 3); end
    while n % 5 == 0; n = div(n, 5); end
    return n == 1
end

function estimate_communication_cost(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}})
    """
    Estimate communication cost for spectral method transposes.
    Based on dedalus transpose operation analysis.
    """
    total_elements = prod(global_shape)
    elements_per_process = total_elements / prod(mesh)
    
    # Estimate based on all-to-all communication pattern
    # Cost proportional to data per process and number of processes
    comm_cost = elements_per_process * prod(mesh)
    
    return comm_cost
end

function estimate_memory_per_process(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}})
    """
    Estimate memory usage per process.
    Assumes complex double precision data (16 bytes per element).
    """
    total_elements = prod(global_shape)
    elements_per_process = total_elements / prod(mesh)
    bytes_per_element = 16  # Complex Float64
    
    return elements_per_process * bytes_per_element
end

# Legacy function for backwards compatibility
function balance_load(local_sizes::Vector{Int})
    """
    Legacy balance_load function for backwards compatibility.
    Redirects to improved load balancing based on dedalus patterns.
    """
    info = get_mpi_info()
    
    if !MPI.Initialized() || info.size == 1
        return local_sizes
    end
    
    # Gather all local sizes for analysis
    all_sizes = MPI.Allgather(local_sizes, MPI.COMM_WORLD)
    total_work = sum(sum(sizes) for sizes in all_sizes)
    
    # Use improved block distribution algorithm
    # Assumes 1D distribution for legacy compatibility
    global_size = total_work
    mesh_size = info.size
    coord = info.rank
    
    start_idx, end_idx = calculate_block_distribution(global_size, mesh_size, coord)
    balanced_size = end_idx - start_idx + 1
    
    return [balanced_size]
end

# GPU-aware parallel computing utilities

    """GPU-aware parallel manager for MPI+GPU computing"""
    
    mpi_info::NamedTuple
    memory_pool::Vector{AbstractArray}
    performance_stats::GPUParallelStats
    
        
        mpi_info = get_mpi_info()
        memory_pool = AbstractArray[]
        perf_stats = GPUParallelStats()
        
        new(device_config, mpi_info, memory_pool, perf_stats)
    end
end

mutable struct GPUParallelStats
    total_time::Float64
    gpu_transfers::Int
    mpi_operations::Int
    memory_allocations::Int
    cache_hits::Int
    cache_misses::Int
    
    function GPUParallelStats()
        new(0.0, 0, 0, 0, 0, 0)
    end
end

    """GPU-aware MPI allreduce operation"""
    start_time = time()
    
    # Move to CPU for MPI if needed
    cpu_data = if managerfalse 
        Array(data)
    else
        data
    end
    
    # Perform MPI allreduce
    result = if MPI.Initialized()
        MPI.Allreduce(cpu_data, op, MPI.COMM_WORLD)
    else
        cpu_data
    end
    
    # Move back to GPU if needed
    if managerfalse
        try
            result = result
        catch e
            @warn "GPU allreduce result transfer failed: $e"
        end
    end
    
    # Update statistics
    manager.performance_stats.mpi_operations += 1
    manager.performance_stats.total_time += time() - start_time
    
    return result
end

    """GPU-aware distributed transpose using PencilArrays"""
    start_time = time()
    
    # Ensure data is on correct device
    gpu_data = data
    
    # Perform transpose operation
    result = if plan !== nothing && MPI.Initialized()
        try
            # Use PencilArrays for distributed transpose
            PencilArrays.transpose!(gpu_data, plan)
        catch e
            @warn "PencilArrays transpose failed, using fallback: $e"
            gpu_data  # Fallback to no-op
        end
    else
        gpu_data
    end
    
    # Synchronize GPU operations
    if managerfalse
    end
    
    # Update statistics
    manager.performance_stats.total_time += time() - start_time
    
    return result
end

    """Create PencilArrays configuration optimized for GPU+MPI"""
    
    # Create pencil configuration following project requirements
    pencil_config = if length(mesh) == 2
        # 2D case with both horizontal and vertical distribution
        PencilConfig(
            global_shape,
            mesh,
            comm=manager.mpi_info.comm,
            decomp_dims=(true, true)  # Enable both dimensions
        )
    elseif length(mesh) == 3
        # 3D case with full decomposition
        PencilConfig(
            global_shape,
            mesh,
            comm=manager.mpi_info.comm,
            decomp_dims=(true, true, true)
        )
    else
        # 1D fallback
        PencilConfig(
            global_shape,
            (prod(mesh),),
            comm=manager.mpi_info.comm
        )
    end
    
    return pencil_config
end

    """Benchmark GPU-MPI communication bandwidth"""
    
    if !MPI.Initialized() || manager.mpi_info.size < 2
        @info "MPI not initialized or single process - skipping bandwidth test"
        return 0.0
    end
    
    # Create test data on GPU
    test_data = if managerfalse
        device_fill(Float64(manager.mpi_info.rank), (data_size,), manager)
    else
        fill(Float64(manager.mpi_info.rank), data_size)
    end
    
    # Warm-up run
    
    # Benchmark run
    start_time = time()
    num_runs = 10
    
    for _ in 1:num_runs
    end
    
    elapsed_time = time() - start_time
    
    # Calculate bandwidth (bytes per second)
    bytes_per_operation = data_size * sizeof(Float64) * 2  # Send + receive
    total_bytes = bytes_per_operation * num_runs
    bandwidth = total_bytes / elapsed_time
    
    @info "GPU-MPI bandwidth: $(round(bandwidth / 1024^3, digits=2)) GB/s"
    
    return bandwidth
end

    """Optimize GPU memory layout for distributed spectral methods"""
    
    
    # Estimate memory requirements
    elements_per_process = prod(global_shape) ÷ prod(mesh)
    bytes_per_element = 16  # Complex Float64
    required_memory = elements_per_process * bytes_per_element
    
    # Add overhead for transforms and temporary arrays (factor of 3)
    estimated_memory = required_memory * 3
    
    if managerfalse && estimated_memory > memory_info.available * 0.8
        @warn "Estimated GPU memory usage ($(round(estimated_memory/1024^3, digits=2)) GB) exceeds 80% of available memory ($(round(memory_info.available/1024^3, digits=2)) GB)"
        
        # Suggest optimizations
        @info "Consider:"
        @info "  - Using more processes to reduce memory per GPU"
        @info "  - Reducing problem size"
        @info "  - Using CPU fallback for large arrays"
    else
        @info "GPU memory layout is optimal"
        @info "  Estimated usage: $(round(estimated_memory/1024^2, digits=2)) MB"
        @info "  Available: $(round(memory_info.available/1024^2, digits=2)) MB"
    end
    
    return estimated_memory
end

    """Move parallel manager to different device"""
    
    old_device = manager
    manager = device_config
    
    # Move memory pool to new device
    new_pool = AbstractArray[]
    for array in manager.memory_pool
        try
            new_array = Array(array)
            push!(new_pool, new_array)
        catch e
            @warn "Failed to move array to device: $e"
        end
    end
    manager.memory_pool = new_pool
    
    @info "Moved GPU parallel manager from $(old_device.device_type) to $(device_config.device_type)"
    
    return manager
end

    """Clear GPU memory pool"""
    empty!(manager.memory_pool)
    
    if managerfalse
        try
            # Force GPU garbage collection if available
        catch e
            @warn "GPU cleanup failed: $e"
        end
    end
    
    @info "Cleared GPU memory pool ($(manager.device_type))"
    
    return manager
end

    """Log GPU parallel performance statistics"""
    
    stats = manager.performance_stats
    
    @info "GPU Parallel performance ($(manager.device_type)):"
    @info "  MPI operations: $(stats.mpi_operations)"
    @info "  Memory allocations: $(stats.memory_allocations)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"
    
    # GPU memory info
    if managerfalse
        pool_memory = sum(sizeof(arr) for arr in manager.memory_pool)
        
        @info "  GPU memory: $(round(pool_memory/1024^2, digits=2)) MB in pool"
        @info "  GPU utilization: $(round(pool_memory/memory_info.total*100, digits=1))%"
    end
end

# Convenience functions for GPU-aware parallel operations
    """Convenience function for GPU-aware parallel sum"""
end

    """Convenience function for GPU-aware parallel max"""
end

    """Convenience function for GPU-aware parallel min"""
end

# Enhanced load balancing for GPU clusters
    """Balance load across heterogeneous GPU cluster"""
    
    info = get_mpi_info()
    
    if !MPI.Initialized() || info.size == 1
        return global_shape
    end
    
    # Query GPU capabilities across all processes
    local_device = length(available_devices) > 0 ? available_devices[1] : "cpu"
    
    
    local_capability = if device_config.device_type != CPU_DEVICE
        memory_info.total
    else
        0  # CPU fallback
    end
    
    # Gather capabilities from all processes
    all_capabilities = if MPI.Initialized()
        MPI.Allgather([local_capability], MPI.COMM_WORLD)
    else
        [local_capability]
    end
    
    # Calculate optimal distribution based on relative capabilities
    total_capability = sum(all_capabilities)
    
    if total_capability > 0
        capability_fraction = local_capability / total_capability
        
        @info "Process $(info.rank): GPU capability fraction = $(round(capability_fraction*100, digits=1))%"
        
        # Adjust local work based on capability
        # This is a simplified approach - real implementation would adjust mesh accordingly
        return global_shape
    else
        @info "No GPUs available, using CPU load balancing"
        return global_shape
    end
end
