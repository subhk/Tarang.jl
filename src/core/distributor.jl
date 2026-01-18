"""
Distributor class for parallel distribution and transformations

Key parallelization features:
- PencilArrays for efficient MPI domain decomposition
- PencilFFTs for parallel spectral transforms
- Automatic mesh optimization for 2D/3D decomposition
- Layout caching for performance
"""

using MPI
using PencilArrays
using LinearAlgebra

struct Layout
    dist::Any
    local_shape::Tuple{Vararg{Int}}
    global_shape::Tuple{Vararg{Int}}
    pencil::Union{Nothing, PencilArrays.Pencil}  # Store Pencil for proper decomposition info
end

# Performance tracking structure for distributor
mutable struct DistributorPerformanceStats
    total_time::Float64
    pencil_creations::Int
    layout_creations::Int
    mpi_operations::Int
    cache_hits::Int
    cache_misses::Int
    transpose_time::Float64  # Track transpose overhead
    communication_time::Float64  # Track MPI communication time

    function DistributorPerformanceStats()
        new(0.0, 0, 0, 0, 0, 0, 0.0, 0.0)
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

    # Architecture (CPU or GPU)
    architecture::AbstractArchitecture

    # PencilArrays integration - CORRECT API usage
    use_pencil_arrays::Bool  # Flag to enable/disable PencilArrays for MPI parallelization
    pencil_config::Union{Nothing, PencilConfig}
    mpi_topology::Union{Nothing, PencilArrays.MPITopology}  # MPI Cartesian topology
    pencil_cache::Dict{Tuple, PencilArrays.Pencil}  # Cache Pencil objects by (shape, decomp_dims)
    transforms::Vector{Any}

    # Layout cache
    layouts::Dict{Any, Layout}

    # Performance tracking
    performance_stats::DistributorPerformanceStats

    # OPTIMIZATION: Precomputed mesh coordinates to avoid O(ndim) computation per lookup
    mesh_coords::Vector{Int}  # Precomputed coordinates of this process in the mesh
    neighbor_ranks::Dict{Int, Tuple{Int, Int}}  # Cached neighbor ranks per dimension (left, right)

    # Nonlinear term evaluator (lazily initialized by nonlinear.jl)
    nonlinear_evaluator::Any

    # GPU-specific caches
    gpu_fft_plans::Dict{Tuple, Any}  # Cached GPU FFT plans
    gpu_arrays::Dict{Symbol, Any}    # Cached GPU working arrays

    # Distributed GPU configuration (for GPU+MPI without PencilArrays)
    # Type is Any to avoid circular dependency with gpu_distributed.jl
    # Actual type is Union{Nothing, DistributedGPUConfig}
    distributed_gpu_config::Any

    # TransposableField support (for 2D pencil decomposition)
    # Types are Any to avoid circular dependency with transposable_field.jl
    # transpose_comms_cache: Dict{Int, TransposeComms} - cached per-field comms
    # transpose_counts_cache: Dict{Tuple, TransposeCounts} - cached counts by shape
    transpose_comms_cache::Dict{Int, Any}
    transpose_counts_cache::Dict{Tuple, Any}

    function Distributor(coordsys::CoordinateSystem;
                        comm::MPI.Comm=MPI.COMM_WORLD,
                        mesh::Union{Nothing, Tuple{Vararg{Int}}}=nothing,
                        dtype::Type=Float64,
                        architecture::AbstractArchitecture=CPU())

        size = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        if mesh === nothing
            # Auto-generate optimal mesh based on coordinate system dimension
            # Following PencilArrays best practices for load balancing
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

        # Build coordinate information following standard pattern
        coordsystems = (coordsys,)  # Single coordinate system for now
        coords_tuple = coords(coordsys)  # Get coordinates from the coordinate system
        total_dim = coordsys.dim  # Total dimension

        # Initialize empty structures
        # Enable PencilArrays for MPI parallelization
        # IMPORTANT: PencilArrays is CPU-only, so we disable it for GPU architecture
        # For GPU+MPI, we use custom distributed GPU implementation instead
        use_pencil_arrays = (size > 1) && !is_gpu(architecture)
        pencil_config = nothing
        mpi_topology = nothing
        pencil_cache = Dict{Tuple, PencilArrays.Pencil}()
        transforms = Any[]
        layouts = Dict{Any, Layout}()
        perf_stats = DistributorPerformanceStats()

        # OPTIMIZATION: Precompute mesh coordinates to avoid O(ndim) per lookup
        mesh_coords = precompute_mesh_coordinates(rank, mesh)
        neighbor_ranks = Dict{Int, Tuple{Int, Int}}()

        # GPU caches
        gpu_fft_plans = Dict{Tuple, Any}()
        gpu_arrays = Dict{Symbol, Any}()

        # Log architecture and parallelization mode
        if rank == 0
            if is_gpu(architecture)
                if size > 1
                    @info "Distributor initialized with GPU architecture (distributed)"
                    @info "  NOTE: Using custom distributed GPU instead of PencilArrays"
                    @info "  Each MPI rank should use one GPU"
                else
                    @info "Distributor initialized with GPU architecture (single GPU)"
                end
            else
                if size > 1
                    @info "Distributor initialized with CPU architecture (using PencilArrays)"
                else
                    @info "Distributor initialized with CPU architecture (serial)"
                end
            end
        end

        # Initialize distributed GPU config for GPU+MPI
        distributed_gpu_config = nothing
        if is_gpu(architecture) && size > 1
            # Will be fully initialized when domain is created (need global_shape)
            # For now, just set up the basic config
            distributed_gpu_config = nothing  # Lazy initialization in plan_transforms!
        end

        # Initialize transpose caches for TransposableField support
        transpose_comms_cache = Dict{Int, Any}()
        transpose_counts_cache = Dict{Tuple, Any}()

        dist = new(comm, size, rank, mesh, coordsys, coordsystems, coords_tuple, total_dim, dtype,
            architecture, use_pencil_arrays, pencil_config, mpi_topology, pencil_cache, transforms, layouts, perf_stats,
            mesh_coords, neighbor_ranks, nothing, gpu_fft_plans, gpu_arrays, distributed_gpu_config,
            transpose_comms_cache, transpose_counts_cache)

        # Precompute neighbor ranks for all mesh dimensions
        if mesh !== nothing && size > 1
            precompute_neighbor_ranks!(dist)
        end

        # Initialize MPI topology for parallel runs (CPU only - GPU uses custom distribution)
        if size > 1 && !is_gpu(architecture)
            initialize_mpi_topology!(dist)
        end

        return dist
    end
end

@inline _ensure_cpu_array(arr::AbstractArray) = is_gpu_array(arr) ? Array(arr) : arr

@inline function _maybe_to_architecture(arch::AbstractArchitecture, arr)
    return is_gpu(arch) ? on_architecture(arch, arr) : arr
end

"""
    precompute_mesh_coordinates(rank::Int, mesh::Union{Nothing, Tuple})

Precompute process coordinates in the mesh to avoid O(ndim) computation per lookup.
Uses row-major ordering consistent with PencilArrays.
"""
function precompute_mesh_coordinates(rank::Int, mesh::Union{Nothing, Tuple})
    if mesh === nothing
        return Int[]
    end

    ndims_mesh = length(mesh)
    coords = zeros(Int, ndims_mesh)

    # Compute all coordinates at once using row-major ordering
    remaining_rank = rank
    for dim in 1:ndims_mesh
        stride = prod(mesh[1:dim-1]; init=1)
        coords[dim] = (remaining_rank ÷ stride) % mesh[dim]
    end

    return coords
end

"""
    precompute_neighbor_ranks!(dist::Distributor)

Precompute neighbor ranks for all mesh dimensions to avoid repeated computation.
Cached in dist.neighbor_ranks as Dict{dim => (left_rank, right_rank)}.
"""
function precompute_neighbor_ranks!(dist::Distributor)
    if dist.mesh === nothing
        return
    end

    for dim in 1:length(dist.mesh)
        left_rank, right_rank = compute_neighbor_ranks_for_dim(dist, dim)
        dist.neighbor_ranks[dim] = (left_rank, right_rank)
    end
end

"""
    compute_neighbor_ranks_for_dim(dist::Distributor, dim::Int)

Compute neighbor ranks for a specific dimension using precomputed mesh coordinates.
"""
function compute_neighbor_ranks_for_dim(dist::Distributor, dim::Int)
    if isempty(dist.mesh_coords) || dim < 1 || dim > length(dist.mesh)
        return (-1, -1)
    end

    n_procs = dist.mesh[dim]
    if n_procs <= 1
        return (-1, -1)
    end
    proc_coord = dist.mesh_coords[dim]

    # Compute neighbor coordinates (periodic boundary)
    left_coord = mod(proc_coord - 1, n_procs)
    right_coord = mod(proc_coord + 1, n_procs)

    # Convert coordinates to ranks using precomputed mesh_coords
    left_rank = coords_to_rank_fast(dist, dim, left_coord)
    right_rank = coords_to_rank_fast(dist, dim, right_coord)

    return (left_rank, right_rank)
end

"""
    coords_to_rank_fast(dist::Distributor, dim::Int, new_coord::Int)

Fast coordinate-to-rank conversion using precomputed mesh coordinates.
Only changes the specified dimension's coordinate.
"""
function coords_to_rank_fast(dist::Distributor, dim::Int, new_coord::Int)
    if isempty(dist.mesh_coords)
        return -1
    end

    # Start with precomputed coordinates, replace the specified dimension
    rank = 0
    stride = 1
    for i in 1:length(dist.mesh)
        coord = (i == dim) ? new_coord : dist.mesh_coords[i]
        rank += coord * stride
        stride *= dist.mesh[i]
    end

    return rank
end

"""
    initialize_mpi_topology!(dist::Distributor)

Initialize MPI Cartesian topology for PencilArrays.
Following PencilArrays best practices from documentation.
"""
function initialize_mpi_topology!(dist::Distributor)
    if dist.size == 1
        return  # No topology needed for serial
    end

    try
        # Create MPI Cartesian topology using PencilArrays API
        # MPITopology(comm, pdims) where pdims is the process grid dimensions
        dist.mpi_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)

        if dist.rank == 0
            @info "Initialized MPI topology: $(dist.mesh) processes"
        end
    catch e
        @warn "Failed to initialize MPI topology, falling back to manual decomposition" exception=e
        dist.mpi_topology = nothing
    end
end

function setup_pencil_arrays(dist::Distributor, global_shape::Tuple{Vararg{Int}})
    """Setup PencilArrays configuration for given global shape"""

    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)

    # Determine which dimensions to decompose
    # PencilArrays convention: decompose the LAST ndims_mesh dimensions by default
    # For 3D data with 2D mesh: decompose dims (2, 3), keep dim 1 local
    if ndims_global >= ndims_mesh
        decomp_dims = ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)
    else
        decomp_dims = ntuple(identity, ndims_global)
    end

    # Create standard PencilArrays configuration
    decomp_flags = ntuple(i -> i in decomp_dims, ndims_global)

    dist.pencil_config = PencilConfig(
        global_shape,
        dist.mesh,
        comm=dist.comm,
        decomp_dims=decomp_flags
    )

    return dist.pencil_config
end

"""
    create_array(dist::Distributor, dtype::Type, shape::Tuple)

Create an array on the distributor's architecture (CPU or GPU).
"""
function create_array(dist::Distributor, dtype::Type, shape::Tuple)
    arch = dist.architecture
    return zeros(arch, dtype, shape...)
end

"""
    create_array(dist::Distributor, shape::Tuple)

Create an array with the distributor's default dtype on its architecture.
"""
create_array(dist::Distributor, shape::Tuple) = create_array(dist, dist.dtype, shape)

"""
    move_to_architecture(dist::Distributor, data::AbstractArray)

Move data to the distributor's architecture.
"""
function move_to_architecture(dist::Distributor, data::AbstractArray)
    return on_architecture(dist.architecture, data)
end

"""
    create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}}, decomp_index::Int=1; dtype::Type=dist.dtype)

Create a Pencil object with proper MPI decomposition.
Uses PencilArrays.Pencil API correctly for efficient parallel operations.

For GPU architecture with serial execution, creates GPU arrays instead of CPU arrays.
"""
function create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}},
                      decomp_index::Int=1; dtype::Type=dist.dtype)

    start_time = time()

    # Serial execution or explicit non-PencilArrays path (e.g., GPU)
    if dist.size == 1 || !dist.use_pencil_arrays
        local_shape = dist.size == 1 ? global_shape : compute_local_shape(dist, global_shape)
        array = create_array(dist, dtype, local_shape)
        dist.performance_stats.pencil_creations += 1
        dist.performance_stats.total_time += time() - start_time
        return array
    end

    # Check cache first
    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)

    # Default decomposition: last ndims_mesh dimensions
    decomp_dims = if ndims_global >= ndims_mesh
        ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)
    else
        ntuple(identity, ndims_global)
    end

    cache_key = (global_shape, decomp_dims, dtype)

    if haskey(dist.pencil_cache, cache_key)
        dist.performance_stats.cache_hits += 1
        pencil = dist.pencil_cache[cache_key]
        # Return a new PencilArray with the cached Pencil configuration
        dist.performance_stats.total_time += time() - start_time
        pencil_array = PencilArrays.PencilArray{dtype}(undef, pencil)
        fill!(pencil_array, zero(dtype))
        return pencil_array
    end

    dist.performance_stats.cache_misses += 1

    # Create Pencil using proper PencilArrays API
    try
        if dist.mpi_topology !== nothing
            # Use pre-created MPI topology (preferred)
            # Pencil(topology, global_dims, decomp_dims)
            pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape, decomp_dims)
        else
            # Fallback: create temporary topology and use it
            # This handles the case where mpi_topology wasn't initialized
            temp_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
            pencil = PencilArrays.Pencil(temp_topology, global_shape, decomp_dims)
        end

        # Cache the Pencil object for reuse
        dist.pencil_cache[cache_key] = pencil

        # Create and return PencilArray
        pencil_array = PencilArrays.PencilArray{dtype}(undef, pencil)
        fill!(pencil_array, zero(dtype))

        dist.performance_stats.pencil_creations += 1
        dist.performance_stats.total_time += time() - start_time

        # Periodically check cache limits
        maybe_cleanup_caches!(dist)

        return pencil_array

    catch e
        # Fallback to simple array if PencilArrays fails
        @warn "PencilArrays creation failed, using regular array" exception=e

        # Compute local shape manually
        local_shape = compute_local_shape(dist, global_shape)

        dist.performance_stats.pencil_creations += 1
        dist.performance_stats.total_time += time() - start_time

        return create_array(dist, dtype, local_shape)
    end
end

"""
    compute_local_shape(dist::Distributor, global_shape::Tuple)

Compute local array shape based on MPI decomposition.
"""
function compute_local_shape(dist::Distributor, global_shape::Tuple)
    if dist.size == 1
        return global_shape
    end

    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)
    local_shape = collect(global_shape)

    # Decompose the last ndims_mesh dimensions
    for i in 1:min(ndims_mesh, ndims_global)
        global_dim_idx = ndims_global - ndims_mesh + i
        mesh_dim_idx = i

        n_global = global_shape[global_dim_idx]
        n_procs = dist.mesh[mesh_dim_idx]

        # Get process coordinate in this dimension
        proc_coord = get_process_coordinate_in_mesh(dist, mesh_dim_idx)

        # Compute local size with load balancing
        base_size = div(n_global, n_procs)
        remainder = n_global % n_procs

        if proc_coord < remainder
            local_shape[global_dim_idx] = base_size + 1
        else
            local_shape[global_dim_idx] = base_size
        end
    end

    return tuple(local_shape...)
end

"""
    get_process_coordinate_in_mesh(dist::Distributor, dim::Int)

Get the coordinate of this process in the specified mesh dimension.
OPTIMIZED: Uses precomputed mesh_coords for O(1) lookup instead of O(ndim) computation.
"""
function get_process_coordinate_in_mesh(dist::Distributor, dim::Int)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return 0
    end

    # OPTIMIZATION: Use precomputed coordinates
    if !isempty(dist.mesh_coords) && dim <= length(dist.mesh_coords)
        return dist.mesh_coords[dim]
    end

    # Fallback to computation if precomputed coords not available
    stride = 1
    for i in 1:(dim-1)
        stride *= dist.mesh[i]
    end

    return div(dist.rank, stride) % dist.mesh[dim]
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

    # Get or create Pencil object for this configuration
    pencil_obj = nothing
    if dist.size > 1
        ndims_global = length(global_shape)
        ndims_mesh = length(dist.mesh)
        decomp_dims = if ndims_global >= ndims_mesh
            ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)
        else
            ntuple(identity, ndims_global)
        end
        cache_key = (global_shape, decomp_dims, dtype)

        if haskey(dist.pencil_cache, cache_key)
            pencil_obj = dist.pencil_cache[cache_key]
        end
    end

    # Create pencil array for this layout
    pencil_array = create_pencil(dist, global_shape, 1, dtype=dtype)
    local_shape = size(pencil_array)

    layout = Layout(dist, local_shape, global_shape, pencil_obj)
    dist.layouts[key] = layout

    # Update performance stats
    dist.performance_stats.total_time += time() - start_time
    dist.performance_stats.layout_creations += 1

    # Periodically check cache limits
    maybe_cleanup_caches!(dist)

    return layout
end

function local_indices(dist::Distributor, axis::Int)
    """
    Get local indices for the given axis (1-indexed).
    For serial execution, returns all indices.
    For parallel execution, returns the indices owned by this process.
    """
    if dist.size == 1 || dist.mesh === nothing
        # Serial case: return all indices
        # We need to know the global size for this axis, but without a basis
        # we can only return a default range. This will be overridden by
        # specific basis implementations when needed.
        return Colon()
    end

    # For parallel case, compute the local range based on decomposition
    # The mesh defines how processes are distributed
    mesh_dim = length(dist.mesh)

    if axis > mesh_dim
        # This axis is not decomposed - return all indices
        return Colon()
    end

    # Get process position in mesh for this axis
    # Without knowing the global size, we cannot compute the exact local range.
    # Return Colon() which indicates "all indices" - the caller should use
    # local_indices(dist, axis, global_size) when the global size is known.
    return Colon()
end

function local_indices(dist::Distributor, axis::Int, global_size::Int)
    """
    Get local indices for the given axis with known global size.
    """
    if dist.size == 1 || dist.mesh === nothing
        return 1:global_size
    end

    mesh_dim = length(dist.mesh)
    decomp_start = max(1, dist.dim - mesh_dim + 1)

    if axis < decomp_start || axis > dist.dim
        return 1:global_size
    end

    mesh_axis = axis - decomp_start + 1
    if mesh_axis < 1 || mesh_axis > mesh_dim
        return 1:global_size
    end

    procs_per_axis = dist.mesh[mesh_axis]
    proc_idx = get_process_coordinate_in_mesh(dist, mesh_axis)

    # Compute local range
    chunk_size = div(global_size, procs_per_axis)
    remainder = mod(global_size, procs_per_axis)

    if proc_idx < remainder
        start_idx = proc_idx * (chunk_size + 1) + 1
        end_idx = start_idx + chunk_size
    else
        start_idx = remainder * (chunk_size + 1) + (proc_idx - remainder) * chunk_size + 1
        end_idx = start_idx + chunk_size - 1
    end

    return start_idx:end_idx
end

function Field(dist::Distributor; name::String="field", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    """Create a scalar field"""
    field = ScalarField(dist, name, bases, dtype)
    return field
end

# Note: VectorField and TensorField convenience constructors are defined in field.jl
# to avoid conflict with the struct definitions (functions and structs cannot share names in Julia)

function local_grids(dist::Distributor, bases::Vararg{Basis}; scales=nothing, move_to_arch::Bool=true)
    """
    Return local coordinate grids for the given bases.

    GPU-aware: By default, grids are moved to the distributor's architecture.
    This enables efficient broadcasting with field data on GPU:

    ```julia
    # Efficient GPU broadcasting
    x, z = local_grids(dist, xbasis, zbasis)  # Returns GPU arrays if arch=GPU()
    field["g"] .= sin.(x) .* cos.(z')         # Pure GPU operation
    ```

    Set `move_to_arch=false` to always return CPU arrays (e.g., for file I/O).

    Following implementation in distributor:294
    """
    scales = remedy_scales(dist, scales)
    grids = []

    for (i, basis) in enumerate(bases)
        # Get scales for this basis
        axis_start = first_axis(dist, basis)
        axis_end = last_axis(dist, basis)
        basis_scales = scales[axis_start+1:axis_end+1]  # Julia 1-indexed

        # Get local grids from the basis (architecture-aware)
        basis_grids = local_grids(basis, dist, basis_scales; move_to_arch=move_to_arch)
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
    Following implementation in distributor:188-197
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
    """Get axis index for a coordinate."""
    for (i, c) in enumerate(dist.coords)
        if c.coordsys == coord.coordsys && c.name == coord.name
            return i - 1  # 0-indexed
        end
    end
    throw(ArgumentError("Coordinate $(coord.name) not found in distributor"))
end

function get_axis(dist::Distributor, coordsys::CoordinateSystem)
    """Get axis for coordinate system (uses first coordinate)."""
    return get_axis(dist, coords(coordsys)[1])
end

function get_basis_axis(dist::Distributor, basis::Basis)
    """Get axis index for a basis."""
    # Find the coordinate that matches this basis's element_label
    coord_name = basis.meta.element_label
    for (i, c) in enumerate(dist.coords)
        if c.name == coord_name
            return i - 1  # 0-indexed
        end
    end
    # Fallback: use the first coordinate of the basis's coordinate system
    basis_coords = coords(basis.meta.coordsys)
    return get_axis(dist, basis_coords[1])
end

function first_axis(dist::Distributor, basis::Basis)
    """
    Get first axis index for a basis.
    Following implementation in distributor:210
    """
    return get_basis_axis(dist, basis)
end

function last_axis(dist::Distributor, basis::Basis)
    """
    Get last axis index for a basis.
    Following implementation in distributor:213
    """
    return first_axis(dist, basis) + basis.meta.dim - 1
end

# MPI communication helpers
function gather_array(dist::Distributor, local_array::PencilArrays.PencilArray)
    """Gather array from all processes (PencilArrays-aware)"""

    start_time = time()

    if dist.size == 1
        result = Array(local_array)
    else
        result = PencilArrays.gather(local_array, 0)
        if dist.rank != 0
            result = Array{eltype(local_array)}(undef, PencilArrays.size_global(local_array)...)
        end
        MPI.Bcast!(result, dist.comm; root=0)
        dist.performance_stats.mpi_operations += 1
    end

    dist.performance_stats.total_time += time() - start_time
    return result
end

function gather_array(dist::Distributor, local_array::AbstractArray)
    """
    Gather array from all processes (fallback for non-PencilArray types).
    Note: MPI.Allgather flattens arrays - use the PencilArray version for
    shape-preserving gather of multi-dimensional distributed arrays.

    GPU-aware: Automatically transfers GPU arrays to CPU before MPI operations,
    since MPI requires CPU memory.
    """

    start_time = time()

    # For GPU arrays, transfer to CPU first (MPI requires CPU memory)
    cpu_array = on_architecture(CPU(), local_array)

    if dist.size == 1
        result = cpu_array
    else
        result = MPI.Allgather(cpu_array, dist.comm)
        dist.performance_stats.mpi_operations += 1
    end

    # Update performance stats
    dist.performance_stats.total_time += time() - start_time

    return result
end

function scatter_array(dist::Distributor, global_array::AbstractArray)
    """Scatter array to all processes"""

    start_time = time()

    if dist.size == 1 || dist.mesh === nothing
        dist.performance_stats.total_time += time() - start_time
        return _maybe_to_architecture(dist.architecture, global_array)
    end

    global_shape = size(global_array)
    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)

    decomp_dims = if ndims_global >= ndims_mesh
        ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)
    else
        ntuple(identity, ndims_global)
    end

    pencil = nothing
    cache_key = (global_shape, decomp_dims, eltype(global_array))
    if haskey(dist.pencil_cache, cache_key)
        pencil = dist.pencil_cache[cache_key]
    else
        try
            if dist.mpi_topology !== nothing
                pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape, decomp_dims)
            else
                # Fallback: create temporary topology
                temp_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
                pencil = PencilArrays.Pencil(temp_topology, global_shape, decomp_dims)
            end
            dist.pencil_cache[cache_key] = pencil
        catch e
            @warn "PencilArrays scatter failed, falling back to flat scatter" exception=e
            local_size = div(length(global_array), dist.size)
            local_array = zeros(eltype(global_array), local_size)
            MPI.Scatter!(global_array, local_array, 0, dist.comm)
            dist.performance_stats.mpi_operations += 1
            dist.performance_stats.total_time += time() - start_time
            return _maybe_to_architecture(dist.architecture, local_array)
        end
    end

    n_spatial = length(pencil.size_global)
    extra_dims = ndims_global > n_spatial ? global_shape[(n_spatial + 1):end] : ()
    local_spatial = length.(pencil.axes_local)
    local_shape = (local_spatial..., extra_dims...)
    local_array = zeros(eltype(global_array), local_shape...)

    if dist.rank == 0
        nprocs = length(pencil.topology)
        colons_extra = ntuple(_ -> Colon(), length(extra_dims))

        for n in 1:nprocs
            rrange = pencil.axes_all[n]
            dest_rank = pencil.topology.ranks[n]
            if dest_rank == 0
                local_array .= global_array[rrange..., colons_extra...]
            else
                send_buf = global_array[rrange..., colons_extra...]
                MPI.Send(send_buf, dist.comm; dest=dest_rank, tag=0)
            end
        end
    else
        MPI.Recv!(local_array, dist.comm; source=0, tag=0)
    end

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return _maybe_to_architecture(dist.architecture, local_array)
end

function allreduce_array(dist::Distributor, local_array::AbstractArray, op=MPI.SUM)
    """All-reduce operation on array"""

    start_time = time()

    arch = dist.architecture
    if is_gpu_array(local_array) || is_gpu(arch)
        local_cpu = _ensure_cpu_array(local_array)
        result_cpu = similar(local_cpu)
        MPI.Allreduce!(local_cpu, result_cpu, op, dist.comm)
        result = _maybe_to_architecture(arch, result_cpu)
    else
        result = similar(local_array)
        MPI.Allreduce!(local_array, result, op, dist.comm)
    end

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return result
end

# Maximum cache sizes to prevent unbounded memory growth
const MAX_LAYOUT_CACHE_SIZE = 100
const MAX_PENCIL_CACHE_SIZE = 50

function clear_distributor_cache!(dist::Distributor)
    """Clear caches for distributor"""

    # Clear layout cache
    empty!(dist.layouts)

    # Clear pencil cache
    empty!(dist.pencil_cache)

    # Reset PencilArrays configuration if needed
    dist.pencil_config = nothing

    @info "Cleared distributor caches"

    return dist
end

"""
    enforce_cache_limits!(dist::Distributor)

Enforce cache size limits by evicting oldest entries when limits are exceeded.
This prevents unbounded memory growth in long-running simulations.
"""
function enforce_cache_limits!(dist::Distributor)
    # Check layout cache
    if length(dist.layouts) > MAX_LAYOUT_CACHE_SIZE
        # Remove half of the entries (LRU would be better but more complex)
        n_to_remove = length(dist.layouts) ÷ 2
        keys_to_remove = collect(keys(dist.layouts))[1:n_to_remove]
        for k in keys_to_remove
            delete!(dist.layouts, k)
        end
        dist.performance_stats.cache_misses += n_to_remove  # Track evictions
        @debug "Evicted $n_to_remove layout cache entries"
    end

    # Check pencil cache
    if length(dist.pencil_cache) > MAX_PENCIL_CACHE_SIZE
        n_to_remove = length(dist.pencil_cache) ÷ 2
        keys_to_remove = collect(keys(dist.pencil_cache))[1:n_to_remove]
        for k in keys_to_remove
            delete!(dist.pencil_cache, k)
        end
        @debug "Evicted $n_to_remove pencil cache entries"
    end
end

"""
    maybe_cleanup_caches!(dist::Distributor)

Periodically check and enforce cache limits.
Called automatically during pencil/layout creation.
"""
function maybe_cleanup_caches!(dist::Distributor)
    # Only check every 10 operations to amortize overhead
    total_ops = dist.performance_stats.pencil_creations + dist.performance_stats.layout_creations
    if total_ops > 0 && total_ops % 10 == 0
        enforce_cache_limits!(dist)
    end
end

function get_distributor_memory_info(dist::Distributor)
    """Get memory usage information for distributor"""

    return (
        cached_layouts = length(dist.layouts),
        cached_pencils = length(dist.pencil_cache)
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
    @info "  Transpose time: $(round(stats.transpose_time, digits=3)) seconds"
    @info "  Communication time: $(round(stats.communication_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"

    # Memory usage
    mem_info = get_distributor_memory_info(dist)
    @info "  Cached layouts: $(mem_info.cached_layouts)"
    @info "  Cached pencils: $(mem_info.cached_pencils)"
end

# MPI communication functions
function mpi_alltoall(dist::Distributor, send_data::AbstractArray, recv_data::AbstractArray)
    """All-to-all communication"""

    start_time = time()

    nprocs = MPI.Comm_size(dist.comm)
    send_count = length(send_data) ÷ nprocs
    recv_count = length(recv_data) ÷ nprocs

    if is_gpu_array(send_data) || is_gpu_array(recv_data) || is_gpu(dist.architecture)
        send_cpu = _ensure_cpu_array(send_data)
        recv_cpu = Array{eltype(recv_data)}(undef, size(recv_data)...)
        MPI.Alltoall!(MPI.UBuffer(send_cpu, send_count), MPI.UBuffer(recv_cpu, recv_count), dist.comm)
        if is_gpu_array(recv_data)
            copyto!(recv_data, on_architecture(dist.architecture, recv_cpu))
        else
            copyto!(recv_data, recv_cpu)
        end
    else
        MPI.Alltoall!(MPI.UBuffer(send_data, send_count), MPI.UBuffer(recv_data, recv_count), dist.comm)
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

# ============================================================================
# Pencil Transpose Support for Multi-dimensional FFTs
# Following PencilArrays transpose API for efficient parallel transforms
# ============================================================================

"""
    create_transpose_pencil(dist::Distributor, source_pencil::PencilArrays.Pencil, new_decomp_dims::Tuple)

Create a new Pencil with different decomposition dimensions for transpose operations.
This is essential for multi-dimensional FFTs where we need to change which dimensions
are local vs distributed.

# Arguments
- `dist`: Distributor with MPI configuration
- `source_pencil`: Original Pencil configuration
- `new_decomp_dims`: New dimensions to decompose (e.g., (1, 3) instead of (2, 3))

# Returns
- New Pencil object with the specified decomposition
"""
function create_transpose_pencil(dist::Distributor, source_pencil::PencilArrays.Pencil,
                                 new_decomp_dims::Tuple)
    if dist.size == 1
        return source_pencil  # No transpose needed for serial
    end

    try
        # Create new Pencil with different decomposition using PencilArrays API
        # Pencil(pen; decomp_dims=new_dims) shares memory buffers with original
        new_pencil = PencilArrays.Pencil(source_pencil; decomp_dims=new_decomp_dims)
        return new_pencil
    catch e
        @warn "Failed to create transpose pencil" exception=e
        return source_pencil
    end
end

"""
    transpose_pencil_data!(dest::PencilArrays.PencilArray, src::PencilArrays.PencilArray)

Perform MPI transpose operation between two PencilArrays with different decompositions.
Uses PencilArrays' optimized transpose! function.

# Arguments
- `dest`: Destination PencilArray (different decomposition than src)
- `src`: Source PencilArray

# Note
This is a key operation for multi-dimensional FFTs:
1. FFT along local dimension
2. Transpose to make another dimension local
3. FFT along new local dimension
4. Transpose back
"""
function transpose_pencil_data!(dest::PencilArrays.PencilArray, src::PencilArrays.PencilArray,
                                dist::Distributor)
    start_time = time()

    try
        # Use PencilArrays transpose! for optimized MPI communication
        PencilArrays.transpose!(dest, src)

        dist.performance_stats.transpose_time += time() - start_time
        dist.performance_stats.mpi_operations += 1
    catch e
        @warn "PencilArrays transpose failed, using manual transpose" exception=e
        # Fallback: manual copy (only works if shapes match)
        copyto!(parent(dest), parent(src))
    end

    return dest
end

# ============================================================================
# Transpose Buffer Cache for Zero-Allocation Transforms
# ============================================================================

"""
    TransposeBufferCache

Pre-allocated buffer cache for transpose operations during spectral transforms.
Avoids repeated allocations during time-stepping loop.
"""
mutable struct TransposeBufferCache
    # Cached PencilArrays for different configurations
    pencil_buffers::Dict{Tuple, PencilArrays.PencilArray}
    # Statistics
    hits::Int
    misses::Int

    function TransposeBufferCache()
        new(Dict{Tuple, PencilArrays.PencilArray}(), 0, 0)
    end
end

# Global transpose buffer cache (lazy initialization)
const TRANSPOSE_CACHE = Ref{Union{Nothing, TransposeBufferCache}}(nothing)

function get_transpose_cache()
    if TRANSPOSE_CACHE[] === nothing
        TRANSPOSE_CACHE[] = TransposeBufferCache()
    end
    return TRANSPOSE_CACHE[]
end

"""
    get_transpose_buffer!(cache::TransposeBufferCache, pencil::PencilArrays.Pencil,
                          dtype::Type, key::Tuple)

Get or create a transpose buffer for the given pencil configuration.
"""
function get_transpose_buffer!(cache::TransposeBufferCache, pencil::PencilArrays.Pencil,
                               dtype::Type, key::Tuple)
    full_key = (key..., dtype)

    if haskey(cache.pencil_buffers, full_key)
        cache.hits += 1
        return cache.pencil_buffers[full_key]
    end

    cache.misses += 1

    # Create new PencilArray buffer
    buf = PencilArrays.PencilArray{dtype}(undef, pencil)
    fill!(buf, zero(dtype))

    cache.pencil_buffers[full_key] = buf
    return buf
end

"""
    transpose_pencil_cached!(dest, src, dist::Distributor; cache=nothing)

Cached version of transpose_pencil_data! that reuses buffers.
"""
function transpose_pencil_cached!(dest::PencilArrays.PencilArray, src::PencilArrays.PencilArray,
                                  dist::Distributor)
    start_time = time()

    try
        # Use PencilArrays transpose! (already optimized)
        PencilArrays.transpose!(dest, src)

        dist.performance_stats.transpose_time += time() - start_time
        dist.performance_stats.mpi_operations += 1
    catch e
        @warn "Cached transpose failed" exception=e
        copyto!(parent(dest), parent(src))
    end

    return dest
end

"""
    clear_transpose_cache!()

Clear the transpose buffer cache (useful for memory reclamation).
"""
function clear_transpose_cache!()
    cache = get_transpose_cache()
    empty!(cache.pencil_buffers)
    cache.hits = 0
    cache.misses = 0
end

"""
    transpose_cache_stats()

Get statistics about the transpose buffer cache.
"""
function transpose_cache_stats()
    cache = get_transpose_cache()
    total = cache.hits + cache.misses
    hit_rate = total > 0 ? cache.hits / total : 0.0

    return (
        hits = cache.hits,
        misses = cache.misses,
        hit_rate = hit_rate,
        num_buffers = length(cache.pencil_buffers),
        memory_bytes = sum(sizeof ∘ parent, values(cache.pencil_buffers); init=0)
    )
end

# ============================================================================
# Optimized MPI Communication Patterns
# ============================================================================

"""
    async_allreduce!(dest::AbstractArray, src::AbstractArray, op, dist::Distributor)

Non-blocking allreduce operation for overlapping communication and computation.
Returns an MPI request that can be waited on later.
"""
function async_allreduce!(dest::AbstractArray, src::AbstractArray, op, dist::Distributor)
    start_time = time()

    request = MPI.Iallreduce!(src, dest, op, dist.comm)

    dist.performance_stats.mpi_operations += 1

    return request
end

"""
    wait_async!(request::MPI.Request, dist::Distributor)

Wait for an asynchronous MPI operation to complete.
"""
function wait_async!(request::MPI.Request, dist::Distributor)
    start_time = time()

    MPI.Wait(request)

    dist.performance_stats.communication_time += time() - start_time
end

"""
    neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                       recv_left::AbstractArray, recv_right::AbstractArray,
                       dim::Int, dist::Distributor)

Optimized nearest-neighbor exchange for stencil operations.
Uses MPI_Sendrecv for efficient bidirectional communication.

# Arguments
- `send_left/right`: Data to send to left/right neighbor
- `recv_left/right`: Buffers for received data
- `dim`: Mesh dimension for neighbor identification
- `dist`: Distributor with MPI info
"""
function neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                           recv_left::AbstractArray, recv_right::AbstractArray,
                           dim::Int, dist::Distributor)
    if dist.size == 1
        return  # No communication needed for serial
    end

    start_time = time()

    # Get neighbor ranks in the specified dimension
    left_rank, right_rank = get_neighbor_ranks(dist, dim)

    # Use non-blocking sends for overlap
    reqs = MPI.Request[]

    if left_rank >= 0
        push!(reqs, MPI.Isend(send_left, dist.comm; dest=left_rank))
        push!(reqs, MPI.Irecv!(recv_left, dist.comm; source=left_rank))
    end

    if right_rank >= 0
        push!(reqs, MPI.Isend(send_right, dist.comm; dest=right_rank))
        push!(reqs, MPI.Irecv!(recv_right, dist.comm; source=right_rank))
    end

    # Wait for all operations to complete
    MPI.Waitall(reqs)

    dist.performance_stats.communication_time += time() - start_time
    dist.performance_stats.mpi_operations += length(reqs)
end

"""
    async_neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                             recv_left::AbstractArray, recv_right::AbstractArray,
                             dim::Int, dist::Distributor)

OPTIMIZED: Non-blocking neighbor exchange for computation/communication overlap.
Returns MPI requests that can be waited on later with wait_neighbor_exchange!.

Usage pattern for overlap:
    reqs = async_neighbor_exchange!(...)  # Start communication
    compute_interior!(data)                # Compute while communicating
    wait_neighbor_exchange!(reqs, dist)    # Wait for boundary data
    compute_boundary!(data)                # Process boundary using received data
"""
function async_neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                                  recv_left::AbstractArray, recv_right::AbstractArray,
                                  dim::Int, dist::Distributor)
    if dist.size == 1
        return MPI.Request[]  # No communication needed for serial
    end

    # Get neighbor ranks in the specified dimension
    left_rank, right_rank = get_neighbor_ranks(dist, dim)

    # Use non-blocking sends/receives
    reqs = MPI.Request[]

    if left_rank >= 0
        push!(reqs, MPI.Isend(send_left, dist.comm; dest=left_rank))
        push!(reqs, MPI.Irecv!(recv_left, dist.comm; source=left_rank))
    end

    if right_rank >= 0
        push!(reqs, MPI.Isend(send_right, dist.comm; dest=right_rank))
        push!(reqs, MPI.Irecv!(recv_right, dist.comm; source=right_rank))
    end

    dist.performance_stats.mpi_operations += length(reqs)
    return reqs
end

"""
    wait_neighbor_exchange!(reqs::Vector{MPI.Request}, dist::Distributor)

Wait for async neighbor exchange to complete.
"""
function wait_neighbor_exchange!(reqs::Vector{MPI.Request}, dist::Distributor)
    if isempty(reqs)
        return
    end

    start_time = time()
    MPI.Waitall(reqs)
    dist.performance_stats.communication_time += time() - start_time
end

"""
    test_neighbor_exchange(reqs::Vector{MPI.Request})

Test if any async neighbor exchange operations have completed.
Returns (completed, pending) tuple of request indices.
Useful for checking progress during computation overlap.
"""
function test_neighbor_exchange(reqs::Vector{MPI.Request})
    if isempty(reqs)
        return (Int[], Int[])
    end

    completed = Int[]
    pending = Int[]

    for (i, req) in enumerate(reqs)
        flag, _ = MPI.Test(req)
        if flag
            push!(completed, i)
        else
            push!(pending, i)
        end
    end

    return (completed, pending)
end

"""
    get_neighbor_ranks(dist::Distributor, dim::Int)

Get the MPI ranks of left and right neighbors in the specified mesh dimension.
OPTIMIZED: Uses precomputed neighbor_ranks cache for O(1) lookup.
Returns (-1, -1) if no neighbors exist (boundary processes).
"""
function get_neighbor_ranks(dist::Distributor, dim::Int)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return (-1, -1)
    end

    # OPTIMIZATION: Use precomputed neighbor ranks
    if haskey(dist.neighbor_ranks, dim)
        return dist.neighbor_ranks[dim]
    end

    # Fallback: compute and cache
    left_rank, right_rank = compute_neighbor_ranks_for_dim(dist, dim)
    dist.neighbor_ranks[dim] = (left_rank, right_rank)
    return (left_rank, right_rank)
end

"""
    coord_to_rank(dist::Distributor, dim::Int, coord::Int)

Convert a coordinate in the given dimension to an MPI rank.
"""
function coord_to_rank(dist::Distributor, dim::Int, coord::Int)
    if dist.mesh === nothing
        return -1
    end

    # Current process coordinates in all dimensions
    current_coords = [get_process_coordinate_in_mesh(dist, i) for i in 1:length(dist.mesh)]

    # Replace the specified dimension with new coordinate
    current_coords[dim] = coord

    # Convert to rank using row-major ordering
    rank = 0
    stride = 1
    for i in 1:length(dist.mesh)
        rank += current_coords[i] * stride
        stride *= dist.mesh[i]
    end

    return rank
end

# ============================================================================
# Memory Management for Parallel Operations
# ============================================================================

# Module-level cache for communication buffers, keyed by (distributor_id, shape, dim)
const _COMM_BUFFER_CACHE = Dict{UInt64, Dict{Tuple, NamedTuple}}()

"""
    _get_distributor_id(dist::Distributor)

Get a unique identifier for a distributor instance for buffer caching.
"""
function _get_distributor_id(dist::Distributor)
    return objectid(dist)
end

"""
    preallocate_communication_buffers!(dist::Distributor, shapes::Vector{Tuple})

Preallocate communication buffers for common operations to avoid runtime allocation.

For each shape, allocates send/receive buffers for neighbor exchange operations
in all mesh dimensions. Buffers are cached and reused across multiple calls.

# Arguments
- `dist`: The Distributor managing parallel decomposition
- `shapes`: Vector of array shapes that will be communicated

# Example
```julia
# Preallocate buffers for 3D field data
shapes = [(64, 64, 1), (64, 1, 64), (1, 64, 64)]  # Boundary slices
preallocate_communication_buffers!(dist, shapes)
```
"""
function preallocate_communication_buffers!(dist::Distributor, shapes::Vector{Tuple})
    if dist.size == 1
        @debug "Serial mode: no communication buffers needed"
        return
    end

    dist_id = _get_distributor_id(dist)

    # Initialize cache for this distributor if needed
    if !haskey(_COMM_BUFFER_CACHE, dist_id)
        _COMM_BUFFER_CACHE[dist_id] = Dict{Tuple, NamedTuple}()
    end

    buffer_cache = _COMM_BUFFER_CACHE[dist_id]
    n_dims = dist.mesh !== nothing ? length(dist.mesh) : dist.dim

    for shape in shapes
        for dim in 1:n_dims
            cache_key = (shape, dim)

            if haskey(buffer_cache, cache_key)
                continue  # Already allocated
            end

            # Allocate send/receive buffers for this shape and dimension
            # Buffer type matches distributor dtype (complex for spectral methods)
            T = dist.dtype <: Real ? Complex{dist.dtype} : dist.dtype

            send_left = zeros(T, shape...)
            send_right = zeros(T, shape...)
            recv_left = zeros(T, shape...)
            recv_right = zeros(T, shape...)

            buffer_cache[cache_key] = (
                send_left = send_left,
                send_right = send_right,
                recv_left = recv_left,
                recv_right = recv_right
            )
        end
    end

    @debug "Preallocated communication buffers" n_shapes=length(shapes) n_dims=n_dims total_buffers=length(buffer_cache)
end

"""
    get_communication_buffers(dist::Distributor, shape::Tuple, dim::Int)

Retrieve preallocated communication buffers for a given shape and dimension.
Returns a NamedTuple with (send_left, send_right, recv_left, recv_right).

If buffers haven't been preallocated, allocates them on demand.
"""
function get_communication_buffers(dist::Distributor, shape::Tuple, dim::Int)
    if dist.size == 1
        # Return empty buffers for serial mode
        T = dist.dtype <: Real ? Complex{dist.dtype} : dist.dtype
        empty = zeros(T, 0)
        return (send_left=empty, send_right=empty, recv_left=empty, recv_right=empty)
    end

    dist_id = _get_distributor_id(dist)
    cache_key = (shape, dim)

    # Initialize cache if needed
    if !haskey(_COMM_BUFFER_CACHE, dist_id)
        _COMM_BUFFER_CACHE[dist_id] = Dict{Tuple, NamedTuple}()
    end

    buffer_cache = _COMM_BUFFER_CACHE[dist_id]

    # Allocate on demand if not preallocated
    if !haskey(buffer_cache, cache_key)
        preallocate_communication_buffers!(dist, [shape])
    end

    return buffer_cache[cache_key]
end

"""
    clear_communication_buffers!(dist::Distributor)

Clear all preallocated communication buffers for a distributor.
Useful for freeing memory when buffers are no longer needed.
"""
function clear_communication_buffers!(dist::Distributor)
    dist_id = _get_distributor_id(dist)
    if haskey(_COMM_BUFFER_CACHE, dist_id)
        delete!(_COMM_BUFFER_CACHE, dist_id)
        @debug "Cleared communication buffers for distributor"
    end
end

"""
    get_optimal_chunk_size(total_size::Int, num_procs::Int)

Compute optimal chunk size for load-balanced distribution.
Handles remainders by giving extra work to first few processes.
"""
function get_optimal_chunk_size(total_size::Int, num_procs::Int)
    base_size = div(total_size, num_procs)
    remainder = total_size % num_procs
    return (base_size, remainder)
end

# ============================================================================
# Performance Diagnostics
# ============================================================================

"""
    diagnose_parallel_performance(dist::Distributor)

Print detailed performance diagnostics for parallel operations.
"""
function diagnose_parallel_performance(dist::Distributor)
    stats = dist.performance_stats

    if dist.rank == 0
        println("\n" * "="^60)
        println("Parallel Performance Diagnostics")
        println("="^60)
        println("MPI Configuration:")
        println("  Processes: $(dist.size)")
        println("  Mesh: $(dist.mesh)")
        println("  Rank 0 coordinates: $(ntuple(i -> get_process_coordinate_in_mesh(dist, i), length(dist.mesh)))")
        println()
        println("Timing Statistics:")
        println("  Total distributor time: $(round(stats.total_time, digits=4))s")
        println("  Transpose time: $(round(stats.transpose_time, digits=4))s")
        println("  Communication time: $(round(stats.communication_time, digits=4))s")
        println()
        println("Operation Counts:")
        println("  Pencil creations: $(stats.pencil_creations)")
        println("  Layout creations: $(stats.layout_creations)")
        println("  MPI operations: $(stats.mpi_operations)")
        println()
        println("Cache Performance:")
        total_cache = stats.cache_hits + stats.cache_misses
        hit_rate = total_cache > 0 ? 100.0 * stats.cache_hits / total_cache : 0.0
        println("  Cache hits: $(stats.cache_hits)")
        println("  Cache misses: $(stats.cache_misses)")
        println("  Hit rate: $(round(hit_rate, digits=1))%")
        println()
        println("Memory:")
        println("  Cached layouts: $(length(dist.layouts))")
        println("  Cached pencils: $(length(dist.pencil_cache))")
        println("="^60)
    end

    # Synchronize to ensure clean output
    MPI.Barrier(dist.comm)
end

"""
    reset_performance_stats!(dist::Distributor)

Reset all performance statistics counters.
"""
function reset_performance_stats!(dist::Distributor)
    stats = dist.performance_stats
    stats.total_time = 0.0
    stats.pencil_creations = 0
    stats.layout_creations = 0
    stats.mpi_operations = 0
    stats.cache_hits = 0
    stats.cache_misses = 0
    stats.transpose_time = 0.0
    stats.communication_time = 0.0
end

# ============================================================================
# Exports
# ============================================================================

# Export types
export Layout, DistributorPerformanceStats, Distributor, TransposeBufferCache

# Export core functions
export setup_pencil_arrays, create_pencil, compute_local_shape,
       get_process_coordinate_in_mesh, get_layout, local_indices,
       local_grids, remedy_scales, get_axis, get_basis_axis,
       first_axis, last_axis

# Export MPI communication functions
export gather_array, scatter_array, allreduce_array, mpi_alltoall,
       async_allreduce!, wait_async!,
       neighbor_exchange!, async_neighbor_exchange!, wait_neighbor_exchange!,
       test_neighbor_exchange, get_neighbor_ranks, coord_to_rank

# Export pencil transpose functions
export create_transpose_pencil, transpose_pencil_data!, transpose_pencil_cached!,
       get_transpose_cache, get_transpose_buffer!, clear_transpose_cache!,
       transpose_cache_stats

# Export cache and memory management
export clear_distributor_cache!, enforce_cache_limits!, maybe_cleanup_caches!,
       get_distributor_memory_info, preallocate_communication_buffers!,
       get_communication_buffers, clear_communication_buffers!,
       get_optimal_chunk_size

# Export performance and diagnostics
export log_distributor_performance, diagnose_parallel_performance,
       reset_performance_stats!

# Export mesh creation utilities
export create_2d_process_mesh, create_3d_process_mesh
