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
    cache_hits::Int
    cache_misses::Int

    function DistributorPerformanceStats()
        new(0.0, 0, 0, 0, 0, 0)
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

    # Layout cache
    layouts::Dict{Any, Layout}

    # Performance tracking
    performance_stats::DistributorPerformanceStats
    
    function Distributor(coordsys::CoordinateSystem;
                        comm::MPI.Comm=MPI.COMM_WORLD,
                        mesh::Union{Nothing, Tuple{Vararg{Int}}}=nothing,
                        dtype::Type=Float64)
        
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
        perf_stats = DistributorPerformanceStats()

        new(comm, size, rank, mesh, coordsys, coordsystems, coords_tuple, total_dim, dtype,
            use_pencil_arrays, pencil_config, transforms, layouts, perf_stats)
    end
end


function setup_pencil_arrays(dist::Distributor, global_shape::Tuple{Vararg{Int}})
    """Setup PencilArrays configuration for given global shape"""

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

    return dist.pencil_config
end

function create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}},
                      decomp_index::Int=1; dtype::Type=dist.dtype)
    """Create a pencil array with specified decomposition"""

    start_time = time()

    if dist.pencil_config === nothing
        setup_pencil_arrays(dist, global_shape)
    end

    # PencilArrays.Pencil requires (global_dims, decomp_dims, comm) or similar
    # For serial execution or when PencilArrays isn't properly configured,
    # return a simple array wrapper
    config = dist.pencil_config

    if dist.size == 1
        # Serial execution - just create a regular array
        pencil = zeros(dtype, global_shape...)
    else
        # Parallel execution - use PencilArrays properly
        try
            # Create a Pencil using the proper PencilArrays API
            # PencilArrays.Pencil constructor: Pencil(topology, local_range, global_dims)
            # For simplicity, we create a PencilArray directly
            pencil = PencilArrays.PencilArray{dtype}(undef, global_shape, config.comm)
        catch e
            # Fallback to simple array if PencilArrays fails
            @warn "PencilArrays creation failed, using regular array" exception=e
            pencil = zeros(dtype, global_shape...)
        end
    end

    # Update performance stats
    dist.performance_stats.total_time += time() - start_time
    dist.performance_stats.pencil_creations += 1

    return pencil
end

function get_layout(dist::Distributor, bases::Tuple{Vararg{Basis}}, dtype::Type=dist.dtype)
    """Get layout for given bases"""
    
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
    """Create a scalar field"""
    field = ScalarField(dist, name, bases, dtype)
    return field
end

function VectorField(dist::Distributor, coordsys::CoordinateSystem; name::String="vector", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    """Create a vector field"""
    field = VectorField(dist, coordsys, name, bases, dtype)
    return field
end

function TensorField(dist::Distributor, coordsys::CoordinateSystem; name::String="tensor", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    """Create a tensor field"""
    field = TensorField(dist, coordsys, name, bases, dtype)
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
    # Find the coordinate that matches this basis's element_label
    coord_name = basis.meta.element_label
    for (i, c) in enumerate(dist.coords)
        if c.name == coord_name
            return i - 1  # 0-indexed like Python
        end
    end
    # Fallback: use the first coordinate of the basis's coordinate system
    basis_coords = coords(basis.meta.coordsys)
    return get_axis(dist, basis_coords[1])
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

# MPI communication helpers
function gather_array(dist::Distributor, local_array::AbstractArray)
    """Gather array from all processes"""

    start_time = time()

    result = MPI.Allgather(local_array, dist.comm)

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return result
end

function scatter_array(dist::Distributor, global_array::AbstractArray)
    """Scatter array to all processes"""

    start_time = time()

    local_size = div(length(global_array), dist.size)
    local_array = zeros(eltype(global_array), local_size)
    MPI.Scatter!(global_array, local_array, 0, dist.comm)

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return local_array
end

function allreduce_array(dist::Distributor, local_array::AbstractArray, op=MPI.SUM)
    """All-reduce operation on array"""

    start_time = time()

    result = similar(local_array)
    MPI.Allreduce!(local_array, result, op, dist.comm)

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return result
end

function clear_distributor_cache!(dist::Distributor)
    """Clear caches for distributor"""

    # Clear layout cache
    empty!(dist.layouts)

    # Reset PencilArrays configuration if needed
    dist.pencil_config = nothing

    @info "Cleared distributor caches"

    return dist
end

function get_distributor_memory_info(dist::Distributor)
    """Get memory usage information for distributor"""

    return (
        cached_layouts = length(dist.layouts)
    )
end

function log_distributor_performance(dist::Distributor)
    """Log distributor performance statistics"""

    stats = dist.performance_stats

    @info "Distributor performance:"
    @info "  Pencil creations: $(stats.pencil_creations)"
    @info "  Layout creations: $(stats.layout_creations)"
    @info "  MPI operations: $(stats.mpi_operations)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"

    # Memory usage
    mem_info = get_distributor_memory_info(dist)
    @info "  Cached layouts: $(mem_info.cached_layouts)"
end

# MPI communication functions
function mpi_alltoall(dist::Distributor, send_data::AbstractArray, recv_data::AbstractArray)
    """All-to-all communication"""

    start_time = time()

    # Perform MPI all-to-all
    MPI.Alltoall!(send_data, recv_data, dist.comm)

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
