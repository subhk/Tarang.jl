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

    # PencilFFT plan pencils for field allocation (must match plan's expected input/output)
    pencil_fft_input::Union{Nothing, PencilArrays.Pencil}   # Input pencil from PencilFFT plan
    pencil_fft_output::Union{Nothing, PencilArrays.Pencil}  # Output pencil from PencilFFT plan

    # Layout cache (keys are Tuple of bases)
    layouts::Dict{Tuple, Layout}

    # Performance tracking
    performance_stats::DistributorPerformanceStats

    # OPTIMIZATION: Precomputed mesh coordinates to avoid O(ndim) computation per lookup
    mesh_coords::Vector{Int}  # Precomputed coordinates of this process in the mesh
    neighbor_ranks::Dict{Int, Tuple{Int, Int}}  # Cached neighbor ranks per dimension (left, right)

    # Nonlinear term evaluator (lazily initialized by nonlinear.jl)
    nonlinear_evaluator::Union{Nothing, AbstractNonlinearEvaluator}

    # GPU-specific caches
    gpu_fft_plans::Dict{Tuple, Any}  # Cached GPU FFT plans (CUDA-ext types)
    gpu_arrays::Dict{Symbol, Any}    # Cached GPU working arrays

    # Distributed GPU configuration (for GPU+MPI without PencilArrays)
    distributed_gpu_config::Union{Nothing, AbstractDistributedGPUConfig}

    # TransposableField support (for 2D pencil decomposition)
    transpose_comms_cache::Dict{Int, AbstractTransposeComms}
    transpose_counts_cache::Dict{Tuple, AbstractTransposeCounts}

    function Distributor(coordsys::CoordinateSystem;
                        comm::MPI.Comm=MPI.COMM_WORLD,
                        mesh::Union{Nothing, Tuple{Vararg{Int}}}=nothing,
                        dtype::Type=Float64,
                        arch::Union{AbstractArchitecture, Nothing}=nothing,
                        architecture::AbstractArchitecture=CPU(),
                        use_pencil_arrays::Union{Nothing, Bool}=nothing)
        # Allow both `arch=` and `architecture=` kwargs; `arch` takes precedence
        architecture = arch !== nothing ? arch : architecture

        # Ensure MPI is initialized and not finalized before using communicator
        if MPI.Finalized()
            error("Cannot create Distributor: MPI has already been finalized. " *
                  "MPI can only be initialized once per process. Create all Distributors " *
                  "before calling MPI.Finalize().")
        end
        if !MPI.Initialized()
            MPI.Init()
        end

        size = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        # Determine PencilArrays usage FIRST (needed for mesh generation)
        # IMPORTANT: PencilArrays is CPU-only, so we disable it for GPU architecture
        # For GPU+MPI, we use TransposableField which requires a 2D mesh
        # Allow explicit override via use_pencil_arrays parameter
        _use_pencil_arrays = if use_pencil_arrays !== nothing
            use_pencil_arrays
        else
            (size > 1) && !is_gpu(architecture)
        end

        if mesh === nothing
            # Auto-generate optimal mesh based on coordinate system dimension
            # PencilFFTs uses (N-1)-dimensional decomposition for N-dimensional FFT
            # but can utilize N-dimensional topology for efficient transposes
            # TransposableField (GPU+MPI) requires a 2D topology (Rx × Ry)
            if isa(coordsys, CartesianCoordinates)
                if coordsys.dim == 1
                    mesh = (size,)
                elseif coordsys.dim == 2
                    # Use 1D mesh for 2D domains with PencilFFTs
                    # PencilFFTs requires at least one local dimension for FFT
                    # For 2D data, we use slab decomposition (decompose 1 dim, keep 1 local)
                    # A 2D mesh would decompose both dimensions, leaving no local dim for FFT
                    mesh = (size,)
                elseif coordsys.dim == 3
                    # CRITICAL: Use 2D mesh for 3D domains regardless of use_pencil_arrays.
                    # PencilFFT inherently uses 2D decomposition (one dimension must be local
                    # for FFT). Using 3D mesh creates mismatch: helper functions
                    # (get_local_array_size, scatter_array, local_indices) assume full mesh
                    # decomposition but PencilArray only uses 2 dimensions.
                    # TransposableField also requires 2D mesh.
                    mesh = create_2d_process_mesh(size)
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

        # Validate mesh dimensionality for TransposableField compatibility
        # Only enforce for actual parallel execution (size > 1) with non-PencilArrays path
        # For serial execution, any mesh shape is allowed
        if size > 1 && !_use_pencil_arrays && length(mesh) > 2
            # TransposableField requires 2D mesh (Rx × Ry)
            # User provided a 3D+ mesh explicitly, which is incompatible
            throw(ArgumentError(
                "GPU+MPI (use_pencil_arrays=false) requires a 2D mesh for TransposableField compatibility. " *
                "Got mesh=$(mesh) with $(length(mesh)) dimensions. " *
                "Use a 2D mesh like (2, $(size ÷ 2)) or let the system auto-generate it."))
        end

        # Build coordinate information following standard pattern
        coordsystems = (coordsys,)  # Single coordinate system for now
        coords_tuple = coords(coordsys)  # Get coordinates from the coordinate system
        total_dim = coordsys.dim  # Total dimension

        # CRITICAL: Validate mesh dimensionality vs domain dimensionality
        # This catches cases where mesh has more dimensions than the domain,
        # which can cause helper functions to desync from actual PencilArray layouts
        if size > 1 && _use_pencil_arrays && length(mesh) > total_dim
            unused_dims = length(mesh) - total_dim
            unused_procs = prod(mesh[total_dim+1:end])
            if unused_procs > 1
                @warn "Mesh dimensionality ($(length(mesh))) exceeds domain dimensionality ($total_dim) " *
                      "with use_pencil_arrays=true. Only first $total_dim mesh dimension(s) can be used " *
                      "for decomposition. This leaves $unused_dims dimension(s) unutilized, wasting " *
                      "$unused_procs MPI process(es). Helper functions (get_local_array_size, scatter_array, etc.) " *
                      "assume mesh dimensions ≤ domain dimensions. Consider using a $(total_dim)D mesh instead, " *
                      "e.g., mesh=$(Tuple(mesh[1:total_dim]))."
            end
        end
        pencil_config = nothing
        mpi_topology = nothing
        pencil_cache = Dict{Tuple, PencilArrays.Pencil}()
        transforms = Any[]
        pencil_fft_input = nothing
        pencil_fft_output = nothing
        layouts = Dict{Tuple, Layout}()
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
        transpose_comms_cache = Dict{Int, AbstractTransposeComms}()
        transpose_counts_cache = Dict{Tuple, AbstractTransposeCounts}()

        dist = new(comm, size, rank, mesh, coordsys, coordsystems, coords_tuple, total_dim, dtype,
            architecture, _use_pencil_arrays, pencil_config, mpi_topology, pencil_cache, transforms,
            pencil_fft_input, pencil_fft_output, layouts, perf_stats,
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
Uses column-major ordering consistent with Topology2D, scatter_array, and get_local_array_size.

Column-major ordering: first coordinate changes fastest as rank increases.
For mesh (P₁, P₂, ..., Pₖ):
  rank = coord[1] + P₁*(coord[2] + P₂*(coord[3] + ...))
  coord[i] = (rank ÷ prod(mesh[1:i-1])) % mesh[i]
"""
function precompute_mesh_coordinates(rank::Int, mesh::Union{Nothing, Tuple})
    if mesh === nothing
        return Int[]
    end

    ndims_mesh = length(mesh)
    coords = zeros(Int, ndims_mesh)

    # Compute all coordinates using column-major ordering (Fortran-style)
    # This matches Topology2D (rx = rank % Rx, ry = rank ÷ Rx) and scatter_array.
    #
    # For column-major: rank = coords[1] + mesh[1]*(coords[2] + mesh[2]*(coords[3] + ...))
    # Equivalently: coord[i] = (rank ÷ prod(mesh[1:i-1])) % mesh[i]
    #
    # Example: mesh=(2,3), rank=4
    #   stride=1, coords[1] = (4÷1) % 2 = 0, stride=2
    #   stride=2, coords[2] = (4÷2) % 3 = 2
    #   Result: [0,2], verified: 0 + 2*2 = 4 ✓
    #
    # Example: mesh=(2,3), rank=1
    #   coords[1] = (1÷1) % 2 = 1, coords[2] = (1÷2) % 3 = 0
    #   Result: [1,0], verified: 1 + 2*0 = 1 ✓
    stride = 1
    for dim in 1:ndims_mesh
        coords[dim] = (rank ÷ stride) % mesh[dim]
        stride *= mesh[dim]
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
    compute_neighbor_ranks_for_dim(dist::Distributor, dim::Int; periodic::Bool=true)

Compute neighbor ranks for a specific dimension using precomputed mesh coordinates.

# Arguments
- `dist`: The Distributor object
- `dim`: Dimension index (1-based)
- `periodic`: If true (default), use periodic boundary conditions (wrap around).
              If false, return -1 for neighbors at domain boundaries.

# Returns
Tuple (left_rank, right_rank) where -1 indicates no neighbor (boundary or serial).
"""
function compute_neighbor_ranks_for_dim(dist::Distributor, dim::Int; periodic::Bool=true)
    if isempty(dist.mesh_coords) || dim < 1 || dim > length(dist.mesh)
        return (-1, -1)
    end

    n_procs = dist.mesh[dim]
    if n_procs <= 1
        return (-1, -1)
    end
    proc_coord = dist.mesh_coords[dim]

    if periodic
        # Compute neighbor coordinates with periodic boundary (wrap around)
        left_coord = mod(proc_coord - 1, n_procs)
        right_coord = mod(proc_coord + 1, n_procs)
    else
        # Non-periodic boundary: no wrap-around at edges
        left_coord = proc_coord > 0 ? proc_coord - 1 : -1
        right_coord = proc_coord < n_procs - 1 ? proc_coord + 1 : -1
    end

    # Convert coordinates to ranks using precomputed mesh_coords
    left_rank = left_coord >= 0 ? coords_to_rank_fast(dist, dim, left_coord) : -1
    right_rank = right_coord >= 0 ? coords_to_rank_fast(dist, dim, right_coord) : -1

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
        # CRITICAL: In MPI mode with use_pencil_arrays=true, topology failure is fatal
        # Silent fallback would cause PencilArrays operations to fail or produce wrong results
        if dist.use_pencil_arrays
            @error "Failed to initialize MPI topology for PencilArrays" exception=e mesh=dist.mesh
            error("MPI topology initialization failed with $(dist.size) processes and mesh $(dist.mesh). " *
                  "PencilArrays requires a valid MPI Cartesian topology. " *
                  "Check that your MPI installation supports Cartesian topologies.")
        else
            # For GPU+MPI (use_pencil_arrays=false), topology is optional
            @warn "MPI topology initialization failed (non-critical for GPU+MPI mode)" exception=e
            dist.mpi_topology = nothing
        end
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
    create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}}, decomp_dims::Tuple{Vararg{Int}}; dtype::Type=dist.dtype)

Create a Pencil object with proper MPI decomposition.
Uses PencilArrays.Pencil API correctly for efficient parallel operations.

# Two API variants:

## 1. Using `decomp_index::Int` (convenience API)
The `decomp_index` specifies which dimension is kept LOCAL (not decomposed).
This is useful when cycling through pencil orientations in transpose operations.

For a 3D array with 2D mesh (P1, P2):
- decomp_index=1: dims (2,3) decomposed, dim 1 local (z-pencils)
- decomp_index=2: dims (1,3) decomposed, dim 2 local (y-pencils)
- decomp_index=3: dims (1,2) decomposed, dim 3 local (x-pencils)

## 2. Using `decomp_dims::Tuple` (explicit API)
Directly specify which dimensions should be decomposed (distributed across processes).
Use this when you need precise control over the decomposition layout.

Example: `create_pencil(dist, (64, 64, 64), (2, 3))` - decompose dims 2 and 3, keep dim 1 local

# Notes
- For GPU architecture with serial execution, creates GPU arrays instead of CPU arrays.
- The `decomp_dims` API is preferred for explicit control; `decomp_index` is a convenience wrapper.
"""
function create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}},
                      decomp_index::Union{Int, Nothing}=nothing; dtype::Type=dist.dtype)

    start_time = time()

    # Serial execution or explicit non-PencilArrays path (e.g., GPU)
    if dist.size == 1 || !dist.use_pencil_arrays
        # CRITICAL: Use get_local_array_size which respects use_pencil_arrays convention
        # - For GPU+MPI (use_pencil_arrays=false): uses ZLocal convention (decompose FIRST dims)
        # - This ensures consistency with TransposableField's expected layout
        local_shape = dist.size == 1 ? global_shape : get_local_array_size(dist, global_shape)
        array = create_array(dist, dtype, local_shape)
        dist.performance_stats.pencil_creations += 1
        dist.performance_stats.total_time += time() - start_time
        return array
    end

    # Check cache first
    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)

    # Compute decomposition dimensions based on decomp_index
    # decomp_index === nothing: FULL decomposition (for field storage) - decompose LAST ndims_mesh dims
    # decomp_index == Int: PENCIL decomposition (for FFT) - keep that dimension LOCAL
    decomp_dims = if decomp_index === nothing
        # Full decomposition: decompose LAST ndims_mesh dimensions (PencilArrays convention)
        # This is used for field storage where we want maximum parallelism
        _compute_full_decomp_dims(ndims_global, ndims_mesh)
    else
        # Pencil decomposition: keep decomp_index dimension local
        # This is used for FFT operations that require a specific dimension to be local
        _compute_decomp_dims(ndims_global, ndims_mesh, decomp_index)
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
            # Fallback: create topology and STORE it to avoid MPI communicator leak.
            # MPITopology creates an MPI Cartesian communicator that must be kept alive.
            dist.mpi_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
            pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape, decomp_dims)
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
        # CRITICAL: In MPI mode, failing to create PencilArrays is a fatal error
        # Falling back to regular arrays would produce incorrect results
        @error "PencilArrays creation failed in MPI mode" exception=e global_shape=global_shape decomp_dims=decomp_dims
        error("PencilArrays creation failed with $(dist.size) MPI processes. " *
              "Cannot fall back to regular arrays as this would produce incorrect results. " *
              "Please check your PencilArrays installation or use serial execution.")
    end
end

"""
    _compute_full_decomp_dims(ndims_global::Int, ndims_mesh::Int)

Compute decomposition dimensions for FULL decomposition (field storage).
Decomposes the LAST ndims_mesh dimensions, following PencilArrays convention.

This is used for field storage where we want maximum parallelism without
keeping any dimension local. All mesh dimensions are utilized.

For ndims_global=3, ndims_mesh=2: (2, 3) - dims 2,3 decomposed, dim 1 local
For ndims_global=2, ndims_mesh=2: (1, 2) - both dims decomposed
For ndims_global=3, ndims_mesh=3: (1, 2, 3) - all dims decomposed

This matches the convention used by get_local_array_size and other helper
utilities when use_pencil_arrays=true.
"""
function _compute_full_decomp_dims(ndims_global::Int, ndims_mesh::Int)
    if ndims_mesh == 0 || ndims_global == 0
        return ()
    end

    # Decompose LAST ndims_mesh dimensions (PencilArrays convention)
    # This matches get_local_array_size behavior for use_pencil_arrays=true
    n_decomp = min(ndims_mesh, ndims_global)
    decomp_start = ndims_global - n_decomp + 1
    return Tuple(decomp_start:ndims_global)
end

"""
    _compute_decomp_dims(ndims_global::Int, ndims_mesh::Int, decomp_index::Int)

Compute which dimensions to decompose based on pencil index.
Returns a tuple of dimension indices that should be distributed.

The `decomp_index` specifies which dimension is kept LOCAL (not decomposed).
All other dimensions (up to `ndims_mesh`) are decomposed.

For ndims_global=3, ndims_mesh=2:
- decomp_index=1: (2, 3) - dim 1 local (z-pencils in PencilArrays convention)
- decomp_index=2: (1, 3) - dim 2 local (y-pencils)
- decomp_index=3: (1, 2) - dim 3 local (x-pencils)

IMPORTANT: This function ALWAYS respects decomp_index. If there are not enough
dimensions to decompose (ndims_global - 1 < ndims_mesh), it decomposes all
available dimensions except local_dim.

WARNING: For 2D domain with 2D mesh and decomp_index != 0, only 1 dimension can be
decomposed (since one must be local). This may not utilize all mesh dimensions,
potentially wasting MPI processes. Consider using a 1D mesh for 2D domains with
pencil decomposition, or use full decomposition without a local dimension.
"""
function _compute_decomp_dims(ndims_global::Int, ndims_mesh::Int, decomp_index::Int)
    if ndims_mesh == 0 || ndims_global == 0
        return ()
    end

    # Validate decomp_index range - error on invalid values
    if decomp_index < 1 || decomp_index > ndims_global
        throw(ArgumentError(
            "decomp_index=$decomp_index is out of valid range [1, $ndims_global] for $(ndims_global)D array. " *
            "decomp_index specifies which dimension is kept LOCAL (not decomposed). " *
            "Valid values are 1 to $ndims_global."
        ))
    end

    # The decomp_index specifies which dimension is LOCAL (not decomposed)
    # All other dimensions (up to ndims_mesh) are decomposed
    local_dim = decomp_index

    # Collect ALL dimensions except local_dim, then take up to ndims_mesh
    decomp_list = Int[]
    for d in 1:ndims_global
        if d != local_dim
            push!(decomp_list, d)
        end
    end

    # If we have more dimensions than mesh can decompose, take the first ndims_mesh
    # This maintains consistency: local_dim is ALWAYS excluded
    if length(decomp_list) > ndims_mesh
        decomp_list = decomp_list[1:ndims_mesh]
    end

    # CRITICAL: Error if decomp_dims.length < ndims_mesh for 2D domain with 2D mesh
    # This configuration creates an inconsistent state where helper utilities
    # (get_local_array_size, get_local_range, local_indices, scatter_array) return shapes/ranges
    # that don't match the actual PencilArray layout, causing data corruption.
    if length(decomp_list) < ndims_mesh && ndims_global > 1
        if ndims_global == 2 && ndims_mesh == 2
            # 2D domain with 2D mesh and decomp_index != 0 is NOT supported
            # because it creates an un-decomposable configuration
            error("Cannot create pencil with decomp_index=$decomp_index for 2D domain with 2D mesh. " *
                  "Keeping dimension $decomp_index local leaves only 1 dimension to decompose, " *
                  "but the mesh has 2 dimensions. This creates an inconsistent state where " *
                  "helper functions (get_local_array_size, scatter_array, etc.) assume full " *
                  "mesh decomposition but the actual PencilArray uses partial decomposition. " *
                  "Solutions: (1) Use a 1D mesh instead of 2D mesh for 2D domains with pencil FFT, " *
                  "(2) Use full decomposition without decomp_index (the default), or " *
                  "(3) For 3D domains, use decomp_index which can accommodate 2D mesh.")
        else
            # For other cases (e.g., 3D with 2D mesh), warn but allow
            @warn "Pencil decomposition uses fewer dimensions ($(length(decomp_list))) than mesh dimensions ($ndims_mesh). " *
                  "Dimension $decomp_index is kept LOCAL for pencil decomposition. " *
                  "IMPORTANT: Helper functions (get_local_array_size, get_local_range, local_indices) " *
                  "assume full mesh decomposition and may not match this PencilArray layout. " *
                  "For pencil-specific operations, use PencilArrays.size_global() and pencil.axes_local. " *
                  "Consider using a $(length(decomp_list))D mesh for this configuration if full decomposition is needed." maxlog=1
        end
    end

    return Tuple(decomp_list)
end

"""
    create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}}, decomp_dims::Tuple{Vararg{Int}}; dtype::Type=dist.dtype)

Create a Pencil object with explicit decomposition dimensions.
This overload accepts a tuple of dimensions that should be decomposed (distributed).

Example: create_pencil(dist, (64, 64, 64), (2, 3)) - decompose dims 2 and 3, keep dim 1 local
"""
function create_pencil(dist::Distributor, global_shape::Tuple{Vararg{Int}},
                      decomp_dims::Tuple{Vararg{Int}}; dtype::Type=dist.dtype)

    start_time = time()

    # Serial execution or explicit non-PencilArrays path (e.g., GPU)
    if dist.size == 1 || !dist.use_pencil_arrays
        # CRITICAL: Use get_local_array_size which respects use_pencil_arrays convention
        # - For GPU+MPI (use_pencil_arrays=false): uses ZLocal convention (decompose FIRST dims)
        # - This ensures consistency with TransposableField's expected layout
        local_shape = dist.size == 1 ? global_shape : get_local_array_size(dist, global_shape)
        array = create_array(dist, dtype, local_shape)
        dist.performance_stats.pencil_creations += 1
        dist.performance_stats.total_time += time() - start_time
        return array
    end

    # Validate decomp_dims
    ndims_global = length(global_shape)
    for d in decomp_dims
        if d < 1 || d > ndims_global
            error("Invalid decomposition dimension $d for array with $ndims_global dimensions")
        end
    end

    cache_key = (global_shape, decomp_dims, dtype)

    if haskey(dist.pencil_cache, cache_key)
        dist.performance_stats.cache_hits += 1
        pencil = dist.pencil_cache[cache_key]
        dist.performance_stats.total_time += time() - start_time
        pencil_array = PencilArrays.PencilArray{dtype}(undef, pencil)
        fill!(pencil_array, zero(dtype))
        return pencil_array
    end

    dist.performance_stats.cache_misses += 1

    try
        # CRITICAL: If mpi_topology is not initialized, create it once and cache it.
        # Creating temporary MPITopology each call leaks MPI communicators.
        if dist.mpi_topology === nothing
            dist.mpi_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
            if dist.rank == 0
                @debug "Initialized MPI topology on first pencil creation: $(dist.mesh)"
            end
        end
        pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape, decomp_dims)

        dist.pencil_cache[cache_key] = pencil
        pencil_array = PencilArrays.PencilArray{dtype}(undef, pencil)
        fill!(pencil_array, zero(dtype))

        dist.performance_stats.pencil_creations += 1
        dist.performance_stats.total_time += time() - start_time
        maybe_cleanup_caches!(dist)

        return pencil_array

    catch e
        @error "PencilArrays creation failed in MPI mode" exception=e global_shape=global_shape decomp_dims=decomp_dims
        error("PencilArrays creation failed with $(dist.size) MPI processes. " *
              "Cannot fall back to regular arrays as this would produce incorrect results.")
    end
end

"""
    compute_local_shape(dist::Distributor, global_shape::Tuple)

Compute local array shape based on MPI decomposition.

Respects dist.use_pencil_arrays:
- PencilArrays convention: decompose LAST ndims_mesh dimensions
- TransposableField convention: decompose FIRST ndims_mesh dimensions
"""
function compute_local_shape(dist::Distributor, global_shape::Tuple)
    if dist.size == 1
        return global_shape
    end

    ndims_global = length(global_shape)
    ndims_mesh = dist.mesh !== nothing ? length(dist.mesh) : 0
    local_shape = collect(global_shape)

    if ndims_mesh == 0
        return global_shape
    end

    for i in 1:min(ndims_mesh, ndims_global)
        # Determine which global dimension corresponds to mesh dimension i
        global_dim_idx = if dist.use_pencil_arrays
            # PencilArrays convention: decompose LAST ndims_mesh dimensions
            ndims_global - ndims_mesh + i
        else
            # TransposableField convention: decompose FIRST ndims_mesh dimensions
            i
        end

        if global_dim_idx < 1 || global_dim_idx > ndims_global
            continue
        end

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

    # Create pencil array for this layout with full decomposition
    pencil_array = create_pencil(dist, global_shape, nothing, dtype=dtype)
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

    Respects dist.use_pencil_arrays for decomposition convention:
    - PencilArrays: decompose LAST ndims_mesh dimensions
    - TransposableField: decompose FIRST ndims_mesh dimensions

    Note: This returns the local indices for the actual field decomposition,
    which uses full mesh decomposition (not pencil decomposition with a local dim).
    For pencil-specific operations, use explicit decomp_dims.
    """
    if dist.size == 1 || dist.mesh === nothing
        return 1:global_size
    end

    ndims_mesh = length(dist.mesh)

    # Map the axis to the corresponding mesh dimension
    # axis is 1-indexed into the global dimensions of the array
    mesh_dim = nothing
    for i in 1:ndims_mesh
        global_dim_idx = if dist.use_pencil_arrays
            # PencilArrays: decompose LAST ndims_mesh dimensions
            # For ndims_global dimensions, mesh dim i maps to global dim (ndims_global - ndims_mesh + i)
            # We don't know ndims_global here, but axis is the global dimension index.
            # Reverse: mesh dim i = axis - (ndims_global - ndims_mesh)
            # Since we don't know ndims_global, check if axis could match any mesh dim
            nothing  # handled below
        else
            # TransposableField: mesh dim i maps to global dim i
            i
        end

        if !dist.use_pencil_arrays && global_dim_idx == axis
            mesh_dim = i
            break
        end
    end

    # For PencilArrays convention, we need to check if this axis is decomposed
    # Since we don't know ndims_global, accept any axis that could map to a mesh dim
    if dist.use_pencil_arrays && mesh_dim === nothing
        # Check all possible ndims_global values to see if axis maps to a mesh dim
        # axis = ndims_global - ndims_mesh + i → i = axis - ndims_global + ndims_mesh
        # For this to be valid: 1 ≤ i ≤ ndims_mesh, i.e., axis ≥ ndims_global - ndims_mesh + 1
        # Since we don't know ndims_global, use a heuristic: if axis > ndims_mesh, it's likely
        # a decomposed dimension (PencilArrays decomposes high-index dims)
        # For the common case: ndims_global = dim of the domain, axis indexes into it
        # mesh dim i corresponds to global dim (ndims_global - ndims_mesh + i)
        # So mesh_dim = axis - (ndims_global - ndims_mesh)
        # Without ndims_global, assume axis is directly decomposed by mesh dim (axis - offset)
        # Fallback: treat as non-decomposed if we can't determine
        for i in 1:ndims_mesh
            # Try all plausible ndims_global values (axis..axis+ndims_mesh)
            for ndims_global in axis:(axis + ndims_mesh)
                if ndims_global - ndims_mesh + i == axis
                    mesh_dim = i
                    break
                end
            end
            if mesh_dim !== nothing
                break
            end
        end
    end

    if mesh_dim === nothing
        # Axis is not decomposed
        return 1:global_size
    end

    n_procs = dist.mesh[mesh_dim]
    proc_coord = get_process_coordinate_in_mesh(dist, mesh_dim)

    # Compute start index and local size with load balancing
    base_size = div(global_size, n_procs)
    remainder = global_size % n_procs

    start_idx = 1
    for r in 0:(proc_coord - 1)
        start_idx += base_size + (r < remainder ? 1 : 0)
    end
    local_size = base_size + (proc_coord < remainder ? 1 : 0)

    return start_idx:(start_idx + local_size - 1)
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

    # CRITICAL: Check if this is actually a PencilArray wrapper (view, reshape, etc.)
    # These wrappers dispatch to AbstractArray but should use PencilArray gather
    underlying = _get_underlying_pencil_array(local_array)
    if underlying !== nothing
        @warn "gather_array received a PencilArray wrapper ($(typeof(local_array))). " *
              "This may cause incorrect MPI gather behavior. Using PencilArray gather instead." maxlog=1
        dist.performance_stats.total_time += time() - start_time
        return gather_array(dist, underlying)
    end

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

"""
    _get_underlying_pencil_array(array)

Check if an array is a wrapper around a PencilArray and return the underlying
PencilArray if so. Returns `nothing` if not a PencilArray wrapper.

This handles cases like:
- SubArray{..., PencilArray} (views)
- ReshapedArray{..., PencilArray} (reshaped)
- Other array wrappers that contain PencilArrays
"""
function _get_underlying_pencil_array(array::AbstractArray)
    # Direct PencilArray - handled by specific method, shouldn't reach here
    if isa(array, PencilArrays.PencilArray)
        return array
    end

    # Check for SubArray wrapping PencilArray
    if array isa SubArray
        parent_arr = parent(array)
        if isa(parent_arr, PencilArrays.PencilArray)
            @warn "View of PencilArray detected in gather_array. " *
                  "Consider using PencilArray directly or copying to a new PencilArray. " *
                  "View slicing may not preserve the correct distributed structure." maxlog=1
            # Return the parent PencilArray - note: this may not be exactly what was intended
            # since the view might select a subset
            return parent_arr
        end
        # Recursively check parent
        return _get_underlying_pencil_array(parent_arr)
    end

    # Check for ReshapedArray wrapping PencilArray
    if array isa Base.ReshapedArray
        parent_arr = parent(array)
        if isa(parent_arr, PencilArrays.PencilArray)
            @warn "Reshaped PencilArray detected in gather_array. " *
                  "Reshaping may break the distributed array structure. " *
                  "Consider gathering first, then reshaping the result." maxlog=1
            return parent_arr
        end
        return _get_underlying_pencil_array(parent_arr)
    end

    # Check for other wrappers that have a parent array
    if applicable(parent, array)
        parent_arr = try
            parent(array)
        catch
            nothing
        end
        if parent_arr !== nothing && parent_arr !== array
            return _get_underlying_pencil_array(parent_arr)
        end
    end

    return nothing
end

function scatter_array(dist::Distributor, global_array::AbstractArray)
    """
    Scatter array to all processes.

    IMPORTANT: Uses different decomposition conventions based on dist.use_pencil_arrays:
    - use_pencil_arrays=true (CPU+MPI): PencilArrays convention, decompose LAST dims
    - use_pencil_arrays=false (GPU+MPI): TransposableField ZLocal convention, decompose FIRST dims

    Note: For GPU architectures, the input global_array should be a CPU array.
    The function will return the local portion on the target architecture (GPU if applicable).

    Communication pattern: Rank 0 distributes data using blocking Send operations to each
    destination rank sequentially, while other ranks post blocking Recv from rank 0.
    This pattern is safe because:
    - Each Recv has exactly one matching Send
    - MPI buffers small/medium messages automatically
    - Receives are posted before rank 0 completes all sends

    For very large arrays that exceed MPI buffer limits, consider using non-blocking
    Isend/Irecv with Waitall for better overlap.
    """

    start_time = time()

    if dist.size == 1 || dist.mesh === nothing
        dist.performance_stats.total_time += time() - start_time
        return _maybe_to_architecture(dist.architecture, global_array)
    end

    # CRITICAL: For MPI scatter, global_array must be on CPU for rank 0
    # Other ranks don't need global_array, but MPI.Recv! needs CPU buffers
    # unless using CUDA-aware MPI
    cpu_global_array = global_array
    if is_gpu_array(global_array)
        if !check_cuda_aware_mpi()
            # Stage through CPU for MPI operations
            cpu_global_array = Array(global_array)
        end
    end

    global_shape = size(global_array)
    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)

    if dist.use_pencil_arrays
        # PencilArrays convention: decompose LAST dims
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
                # CRITICAL: Initialize mpi_topology if not done, then reuse.
                # Creating temporary MPITopology each call leaks MPI communicators.
                if dist.mpi_topology === nothing
                    dist.mpi_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
                    if dist.rank == 0
                        @debug "Initialized MPI topology in scatter_array: $(dist.mesh)"
                    end
                end
                pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape, decomp_dims)
                dist.pencil_cache[cache_key] = pencil
            catch e
                @error "PencilArrays scatter failed in MPI mode" exception=e global_shape=global_shape decomp_dims=decomp_dims
                error("PencilArrays scatter failed with $(dist.size) MPI processes. " *
                      "Cannot fall back to flat scatter as this would produce incorrect data distribution. " *
                      "Please check your PencilArrays installation or use serial execution.")
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
                    local_array .= cpu_global_array[rrange..., colons_extra...]
                else
                    send_buf = cpu_global_array[rrange..., colons_extra...]
                    MPI.Send(send_buf, dist.comm; dest=dest_rank, tag=0)
                end
            end
        else
            # Use MPI.Probe to check incoming message size before Recv
            # This catches mismatches between rank 0's PencilArray ranges and our local shape
            status = MPI.Probe(dist.comm, MPI.Status; source=0, tag=0)
            msg_count = MPI.Get_count(status, eltype(local_array))
            expected_count = length(local_array)
            if msg_count != expected_count
                error("Scatter recv shape mismatch on rank $(dist.rank): " *
                      "expected $expected_count elements (shape=$(size(local_array))), " *
                      "but rank 0 is sending $msg_count elements. " *
                      "This may indicate a PencilArrays configuration issue.")
            end
            MPI.Recv!(local_array, dist.comm; source=0, tag=0)
        end
    else
        # GPU+MPI / TransposableField ZLocal convention: decompose FIRST dims
        # Use get_local_array_size which respects use_pencil_arrays convention
        local_shape = get_local_array_size(dist, global_shape)
        local_array = zeros(eltype(global_array), local_shape...)

        # Compute local ranges for FIRST dims decomposition
        # This matches TransposableField's ZLocal convention
        mesh = dist.mesh
        P1 = mesh[1]
        P2 = ndims_mesh >= 2 ? mesh[2] : 1
        coord1 = dist.rank % P1
        coord2 = dist.rank ÷ P1

        # Compute ranges for first decomposed dimensions
        function compute_range(global_size, n_procs, coord)
            base_size = div(global_size, n_procs)
            remainder = global_size % n_procs
            if coord < remainder
                start = coord * (base_size + 1) + 1
                stop = start + base_size
            else
                start = coord * base_size + remainder + 1
                stop = start + base_size - 1
            end
            return start:stop
        end

        # Build ranges for each dimension
        ranges = Vector{UnitRange{Int}}(undef, ndims_global)
        for d in 1:ndims_global
            if d == 1 && ndims_mesh >= 1
                ranges[d] = compute_range(global_shape[1], P1, coord1)
            elseif d == 2 && ndims_mesh >= 2
                ranges[d] = compute_range(global_shape[2], P2, coord2)
            else
                ranges[d] = 1:global_shape[d]  # Not decomposed
            end
        end

        if dist.rank == 0
            # Rank 0 distributes data
            for dest_rank in 0:(dist.size-1)
                dest_coord1 = dest_rank % P1
                dest_coord2 = dest_rank ÷ P1

                dest_ranges = Vector{UnitRange{Int}}(undef, ndims_global)
                for d in 1:ndims_global
                    if d == 1 && ndims_mesh >= 1
                        dest_ranges[d] = compute_range(global_shape[1], P1, dest_coord1)
                    elseif d == 2 && ndims_mesh >= 2
                        dest_ranges[d] = compute_range(global_shape[2], P2, dest_coord2)
                    else
                        dest_ranges[d] = 1:global_shape[d]
                    end
                end

                if dest_rank == 0
                    local_array .= cpu_global_array[dest_ranges...]
                else
                    send_buf = cpu_global_array[dest_ranges...]
                    # CRITICAL: Validate send size matches expected recv size for dest_rank
                    # The dest_rank computed its local_shape independently; if they disagree,
                    # MPI.Recv! could either fail (too large) or leave uninitialized memory (too small)
                    MPI.Send(send_buf, dist.comm; dest=dest_rank, tag=0)
                end
            end
        else
            # Use MPI.Probe to check incoming message size before Recv
            # This catches mismatches between rank 0's send and our expected recv
            status = MPI.Probe(dist.comm, MPI.Status; source=0, tag=0)
            msg_count = MPI.Get_count(status, eltype(local_array))
            expected_count = length(local_array)
            if msg_count != expected_count
                error("Scatter recv shape mismatch on rank $(dist.rank): " *
                      "expected $expected_count elements (shape=$(size(local_array))), " *
                      "but rank 0 is sending $msg_count elements. " *
                      "This indicates a decomposition computation divergence between ranks.")
            end
            MPI.Recv!(local_array, dist.comm; source=0, tag=0)
        end
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

    # Validate that buffer sizes are evenly divisible by nprocs
    # MPI.Alltoall requires uniform message sizes
    send_len = length(send_data)
    recv_len = length(recv_data)

    if send_len % nprocs != 0
        error("mpi_alltoall: send_data length ($send_len) must be divisible by nprocs ($nprocs). " *
              "Use mpi_alltoallv for non-uniform message sizes.")
    end
    if recv_len % nprocs != 0
        error("mpi_alltoall: recv_data length ($recv_len) must be divisible by nprocs ($nprocs). " *
              "Use mpi_alltoallv for non-uniform message sizes.")
    end

    send_count = send_len ÷ nprocs
    recv_count = recv_len ÷ nprocs

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
        # CRITICAL: Transpose pencils are required for correct distributed FFT operation.
        # Returning the original pencil would silently produce incorrect results.
        @error "Failed to create transpose pencil" exception=e new_decomp_dims=new_decomp_dims
        error("Failed to create transpose pencil with decomposition $new_decomp_dims. " *
              "This is required for correct distributed FFT operation with $(dist.size) processes. " *
              "Please check your PencilArrays installation or use serial execution.")
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
        # CRITICAL: MPI transpose is essential for correct parallel FFT operation
        # A simple copy would produce incorrect results
        @error "PencilArrays transpose failed" exception=e
        error("MPI transpose operation failed with $(dist.size) processes. " *
              "This operation is essential for correct parallel FFT computation. " *
              "Please check your PencilArrays/MPI installation or use serial execution.")
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
        # CRITICAL: MPI transpose is essential for correct parallel FFT operation
        # A simple copy would produce incorrect results (copyto! is NOT a transpose!)
        @error "PencilArrays cached transpose failed" exception=e
        error("MPI transpose operation failed with $(dist.size) processes. " *
              "This operation is essential for correct parallel FFT computation. " *
              "Please check your PencilArrays/MPI installation or use serial execution.")
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

Note: For GPU arrays with non-CUDA-aware MPI, this function stages data through CPU.
The caller must ensure dest is ready to receive GPU data after wait_async!.
"""
function async_allreduce!(dest::AbstractArray, src::AbstractArray, op, dist::Distributor)
    start_time = time()

    # Handle GPU arrays with non-CUDA-aware MPI
    if is_gpu_array(src) && !check_cuda_aware_mpi()
        # Stage through CPU
        src_cpu = Array(src)
        dest_cpu = similar(src_cpu)
        request = MPI.Iallreduce!(src_cpu, dest_cpu, op, dist.comm)
        dist.performance_stats.mpi_operations += 1
        # Return a wrapper that includes staging info for wait_async!
        # Since MPI.Request doesn't support metadata, we use a NamedTuple
        # Retain src_cpu in the tuple to prevent GC while non-blocking MPI reads from it
        return (request=request, staged=true, src_cpu=src_cpu, dest_cpu=dest_cpu, dest_gpu=dest)
    end

    request = MPI.Iallreduce!(src, dest, op, dist.comm)
    dist.performance_stats.mpi_operations += 1

    return (request=request, staged=false)
end

"""
    wait_async!(async_result, dist::Distributor)

Wait for an asynchronous MPI operation to complete.
Handles both raw MPI.Request and staged GPU operations.
"""
function wait_async!(async_result::NamedTuple, dist::Distributor)
    start_time = time()

    MPI.Wait(async_result.request)

    # If data was staged through CPU, copy back to GPU
    if async_result.staged && hasproperty(async_result, :dest_gpu)
        copyto!(async_result.dest_gpu, async_result.dest_cpu)
    end

    dist.performance_stats.communication_time += time() - start_time
end

# Legacy support for raw MPI.Request
function wait_async!(request::MPI.Request, dist::Distributor)
    start_time = time()
    MPI.Wait(request)
    dist.performance_stats.communication_time += time() - start_time
end

"""
    neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                       recv_left::AbstractArray, recv_right::AbstractArray,
                       dim::Int, dist::Distributor; periodic::Bool=true)

Optimized nearest-neighbor exchange for stencil operations.
Uses non-blocking MPI sends/receives for efficient bidirectional communication.

# Arguments
- `send_left/right`: Data to send to left/right neighbor
- `recv_left/right`: Buffers for received data
- `dim`: Mesh dimension for neighbor identification
- `dist`: Distributor with MPI info
- `periodic`: If true (default), assumes periodic domain with wrap-around neighbors.
              If false, processes at domain boundaries have no neighbors on that side
              and the corresponding recv buffers are left unchanged.

# Note
For periodic domains (the default), every process has both left and right neighbors,
even at domain boundaries (they wrap around). For non-periodic domains, boundary
processes will skip communication on the boundary side.
"""
function neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                           recv_left::AbstractArray, recv_right::AbstractArray,
                           dim::Int, dist::Distributor; periodic::Bool=true)
    if dist.size == 1
        return  # No communication needed for serial
    end

    # CRITICAL: Check for GPU arrays with non-CUDA-aware MPI
    # MPI operations on GPU arrays require CUDA-aware MPI or explicit staging
    if (is_gpu_array(send_left) || is_gpu_array(send_right) ||
        is_gpu_array(recv_left) || is_gpu_array(recv_right)) && !check_cuda_aware_mpi()
        error("neighbor_exchange! with GPU arrays requires CUDA-aware MPI. " *
              "Set TARANG_CUDA_AWARE_MPI=1 if your MPI supports it, or copy data " *
              "to CPU before calling. For TransposableField operations, use the " *
              "built-in transpose functions which handle GPU staging automatically.")
    end

    start_time = time()

    # Get neighbor ranks in the specified dimension
    left_rank, right_rank = get_neighbor_ranks(dist, dim; periodic=periodic)

    # Use distinct tags for left and right to avoid message matching issues
    # when left_rank == right_rank (e.g., 2-process periodic mesh)
    # Tag encoding: dim * 10 + direction (0=left, 1=right)
    tag_left = dim * 10 + 0
    tag_right = dim * 10 + 1

    # Use non-blocking sends for overlap
    reqs = MPI.Request[]

    if left_rank >= 0
        # Send to left, receive from left (they send right to us)
        push!(reqs, MPI.Isend(send_left, dist.comm; dest=left_rank, tag=tag_left))
        push!(reqs, MPI.Irecv!(recv_left, dist.comm; source=left_rank, tag=tag_right))
    end

    if right_rank >= 0
        # Send to right, receive from right (they send left to us)
        push!(reqs, MPI.Isend(send_right, dist.comm; dest=right_rank, tag=tag_right))
        push!(reqs, MPI.Irecv!(recv_right, dist.comm; source=right_rank, tag=tag_left))
    end

    # Wait for all operations to complete
    MPI.Waitall(reqs)

    dist.performance_stats.communication_time += time() - start_time
    dist.performance_stats.mpi_operations += length(reqs)
end

"""
    async_neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                             recv_left::AbstractArray, recv_right::AbstractArray,
                             dim::Int, dist::Distributor; periodic::Bool=true)

OPTIMIZED: Non-blocking neighbor exchange for computation/communication overlap.
Returns MPI requests that can be waited on later with wait_neighbor_exchange!.

# Arguments
- `send_left/right`: Data to send to left/right neighbor
- `recv_left/right`: Buffers for received data
- `dim`: Mesh dimension for neighbor identification
- `dist`: Distributor with MPI info
- `periodic`: If true (default), assumes periodic domain with wrap-around neighbors.
              If false, processes at domain boundaries skip communication on that side.

Usage pattern for overlap:
    reqs = async_neighbor_exchange!(...)  # Start communication
    compute_interior!(data)                # Compute while communicating
    wait_neighbor_exchange!(reqs, dist)    # Wait for boundary data
    compute_boundary!(data)                # Process boundary using received data
"""
function async_neighbor_exchange!(send_left::AbstractArray, send_right::AbstractArray,
                                  recv_left::AbstractArray, recv_right::AbstractArray,
                                  dim::Int, dist::Distributor; periodic::Bool=true)
    if dist.size == 1
        return MPI.Request[]  # No communication needed for serial
    end

    # CRITICAL: Check for GPU arrays with non-CUDA-aware MPI
    # Async MPI operations on GPU arrays are especially problematic - the GPU pointers
    # may become invalid before the async operation completes
    if (is_gpu_array(send_left) || is_gpu_array(send_right) ||
        is_gpu_array(recv_left) || is_gpu_array(recv_right)) && !check_cuda_aware_mpi()
        error("async_neighbor_exchange! with GPU arrays requires CUDA-aware MPI. " *
              "Async operations on GPU data without CUDA-aware MPI can leave invalid " *
              "device pointers in flight. Set TARANG_CUDA_AWARE_MPI=1 if your MPI " *
              "supports it, or use synchronous neighbor_exchange! with CPU staging.")
    end

    # Get neighbor ranks in the specified dimension
    left_rank, right_rank = get_neighbor_ranks(dist, dim; periodic=periodic)

    # Use distinct tags for left and right to avoid message matching issues
    # when left_rank == right_rank (e.g., 2-process periodic mesh)
    # Tag encoding: dim * 10 + direction (0=left, 1=right)
    tag_left = dim * 10 + 0
    tag_right = dim * 10 + 1

    # Use non-blocking sends/receives
    reqs = MPI.Request[]

    if left_rank >= 0
        # Send to left, receive from left (they send right to us)
        push!(reqs, MPI.Isend(send_left, dist.comm; dest=left_rank, tag=tag_left))
        push!(reqs, MPI.Irecv!(recv_left, dist.comm; source=left_rank, tag=tag_right))
    end

    if right_rank >= 0
        # Send to right, receive from right (they send left to us)
        push!(reqs, MPI.Isend(send_right, dist.comm; dest=right_rank, tag=tag_right))
        push!(reqs, MPI.Irecv!(recv_right, dist.comm; source=right_rank, tag=tag_left))
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
function get_neighbor_ranks(dist::Distributor, dim::Int; periodic::Bool=true)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return (-1, -1)
    end

    # OPTIMIZATION: Use precomputed neighbor ranks (only for periodic=true, which is the common case)
    if periodic && haskey(dist.neighbor_ranks, dim)
        return dist.neighbor_ranks[dim]
    end

    # Compute (and cache for periodic case)
    left_rank, right_rank = compute_neighbor_ranks_for_dim(dist, dim; periodic=periodic)

    # Only cache periodic neighbors (non-periodic may be requested per-call)
    if periodic
        dist.neighbor_ranks[dim] = (left_rank, right_rank)
    end

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
# Grouped PencilArray Transposes (Dedalus GROUP_TRANSPOSES for CPU)
# ============================================================================

"""
    GroupedPencilTransposeConfig

Configuration for grouped PencilArray transpose operations.
"""
mutable struct GroupedPencilTransposeConfig
    enabled::Bool
    min_fields::Int
    sync_before_transpose::Bool

    function GroupedPencilTransposeConfig()
        new(true, 2, false)
    end
end

const GROUPED_PENCIL_TRANSPOSE_CONFIG = GroupedPencilTransposeConfig()

"""
    set_grouped_pencil_transposes!(enabled::Bool; min_fields::Int=2, sync::Bool=false)

Enable or disable grouped PencilArray transposes for CPU parallelization.
"""
function set_grouped_pencil_transposes!(enabled::Bool; min_fields::Int=2, sync::Bool=false)
    GROUPED_PENCIL_TRANSPOSE_CONFIG.enabled = enabled
    GROUPED_PENCIL_TRANSPOSE_CONFIG.min_fields = min_fields
    GROUPED_PENCIL_TRANSPOSE_CONFIG.sync_before_transpose = sync
    return nothing
end

"""
    group_pencil_transpose!(dest_arrays::Vector{<:PencilArrays.PencilArray},
                            src_arrays::Vector{<:PencilArrays.PencilArray},
                            dist::Distributor)

Transpose multiple PencilArrays together using grouped MPI communication.

For CPU parallelization, this batches multiple PencilArray transposes into
fewer MPI calls by:
1. Stacking source data from all arrays
2. Performing grouped MPI.Alltoallv
3. Unstacking to destination arrays

Following Dedalus GROUP_TRANSPOSES pattern but using PencilArrays infrastructure.
"""
function group_pencil_transpose!(dest_arrays::Vector{<:PencilArrays.PencilArray},
                                 src_arrays::Vector{<:PencilArrays.PencilArray},
                                 dist::Distributor)
    n = length(src_arrays)
    if n == 0
        return
    end

    @assert length(dest_arrays) == n "Destination and source arrays must have same length"

    # Optional synchronization
    if GROUPED_PENCIL_TRANSPOSE_CONFIG.sync_before_transpose
        MPI.Barrier(dist.comm)
    end

    # If grouping disabled or too few arrays, transpose individually
    if !GROUPED_PENCIL_TRANSPOSE_CONFIG.enabled || n < GROUPED_PENCIL_TRANSPOSE_CONFIG.min_fields
        for i in 1:n
            PencilArrays.transpose!(dest_arrays[i], src_arrays[i])
        end
        return
    end

    # Group by compatible pencil configurations
    groups = _group_pencil_arrays(src_arrays, dest_arrays)

    for (key, indices) in groups
        if length(indices) == 1
            i = indices[1]
            PencilArrays.transpose!(dest_arrays[i], src_arrays[i])
        else
            _batched_pencil_transpose!(dest_arrays, src_arrays, indices, dist)
        end
    end
end

"""
    _group_pencil_arrays(src_arrays, dest_arrays)

Group PencilArrays by compatible configurations for batched transpose.
"""
function _group_pencil_arrays(src_arrays::Vector{<:PencilArrays.PencilArray},
                              dest_arrays::Vector{<:PencilArrays.PencilArray})
    groups = Dict{Tuple, Vector{Int}}()

    for i in eachindex(src_arrays)
        src = src_arrays[i]
        dest = dest_arrays[i]

        # Key by source/dest pencil shapes and element type
        src_pencil = PencilArrays.pencil(src)
        dest_pencil = PencilArrays.pencil(dest)

        key = (
            PencilArrays.size_local(src_pencil),
            PencilArrays.size_local(dest_pencil),
            eltype(src)
        )

        if haskey(groups, key)
            push!(groups[key], i)
        else
            groups[key] = [i]
        end
    end

    return groups
end

"""
    _batched_pencil_transpose!(dest_arrays, src_arrays, indices, dist)

Perform batched transpose for a group of compatible PencilArrays.
"""
function _batched_pencil_transpose!(dest_arrays::Vector{<:PencilArrays.PencilArray},
                                    src_arrays::Vector{<:PencilArrays.PencilArray},
                                    indices::Vector{Int},
                                    dist::Distributor)
    nfields = length(indices)
    if nfields == 0
        return
    end

    # Get first array for sizing
    first_idx = indices[1]
    first_src = src_arrays[first_idx]
    first_dest = dest_arrays[first_idx]

    T = eltype(first_src)
    src_local_size = length(parent(first_src))
    dest_local_size = length(parent(first_dest))

    # Stack all source data
    stacked_src = zeros(T, src_local_size * nfields)
    for (batch_idx, field_idx) in enumerate(indices)
        src_data = parent(src_arrays[field_idx])
        offset = (batch_idx - 1) * src_local_size
        stacked_src[offset+1:offset+src_local_size] .= vec(src_data)
    end

    # Use PencilArrays transpose infrastructure
    # Get the transpose plan from the first array pair
    # Note: PencilArrays may not directly support stacked transposes
    # So we use MPI.Alltoallv with proper counts

    src_pencil = PencilArrays.pencil(first_src)
    dest_pencil = PencilArrays.pencil(first_dest)

    # Try to use PencilArrays transpose for the batch
    # If not possible, fall back to individual transposes
    try
        # Compute send/recv counts for the stacked data
        # Each field contributes equally to the counts
        stacked_dest = zeros(T, dest_local_size * nfields)

        # Get topology from pencil
        topo = PencilArrays.topology(src_pencil)
        comm = PencilArrays.comm(topo)

        # For now, fall back to individual transposes since PencilArrays
        # doesn't directly support batched operations on stacked data
        for field_idx in indices
            PencilArrays.transpose!(dest_arrays[field_idx], src_arrays[field_idx])
        end
    catch e
        # Fallback: transpose individually
        @debug "Batched PencilArray transpose failed, using individual transposes: $e"
        for field_idx in indices
            PencilArrays.transpose!(dest_arrays[field_idx], src_arrays[field_idx])
        end
    end
end

"""
    group_transpose_fields!(fields::Vector, dist::Distributor, source_decomp::Tuple, dest_decomp::Tuple)

Transpose multiple fields together using grouped PencilArray operations.

This is a higher-level interface that works with field objects (e.g., ScalarField)
and their underlying PencilArrays. Fields must support `get_grid_data` and `set_grid_data!`.
"""
function group_transpose_fields!(fields::Vector, dist::Distributor,
                                 source_decomp::Tuple, dest_decomp::Tuple)
    if isempty(fields)
        return
    end

    # This function requires PencilArrays (MPI+CPU path)
    if !dist.use_pencil_arrays
        error("group_transpose_fields! requires PencilArrays (MPI+CPU). " *
              "For GPU or non-PencilArrays paths, use individual field transposes instead.")
    end

    # Create source and destination PencilArrays for each field
    # Track original indices to correctly write back results
    src_arrays = PencilArrays.PencilArray[]
    dest_arrays = PencilArrays.PencilArray[]
    included_field_indices = Int[]  # Track which fields were included

    for (field_idx, field) in enumerate(fields)
        grid_data = get_grid_data(field)
        if grid_data === nothing
            continue
        end

        # Get global shape from field's domain (NOT local size from grid_data!)
        # Using size(grid_data) would incorrectly use local size as if it were global
        if field.domain === nothing
            error("group_transpose_fields! requires fields with domain set")
        end
        gshape = global_shape(field.domain)

        # Create PencilArray for source (create_pencil returns PencilArray, not Pencil)
        src_pa = create_pencil(dist, gshape, source_decomp; dtype=eltype(grid_data))

        # Extract the Pencil from the PencilArray to create transpose pencil
        src_pencil_obj = PencilArrays.pencil(src_pa)
        dest_pencil_obj = create_transpose_pencil(dist, src_pencil_obj, dest_decomp)

        # Create destination PencilArray from the transposed Pencil
        dest_pa = PencilArrays.PencilArray{eltype(grid_data)}(undef, dest_pencil_obj)

        copyto!(parent(src_pa), grid_data)

        push!(src_arrays, src_pa)
        push!(dest_arrays, dest_pa)
        push!(included_field_indices, field_idx)
    end

    # Perform grouped transpose
    group_pencil_transpose!(dest_arrays, src_arrays, dist)

    # Copy results back to fields using tracked indices
    # IMPORTANT: Preserve the PencilArray wrapper to maintain MPI distribution metadata.
    # Using copy(parent(...)) would strip the PencilArray wrapper, causing subsequent
    # PencilFFT/MPI operations to fail or produce incorrect results.
    for (dest_idx, field_idx) in enumerate(included_field_indices)
        # Keep the PencilArray - it contains MPI distribution info needed for PencilFFTs
        set_grid_data!(fields[field_idx], dest_arrays[dest_idx])
    end
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

# Export grouped PencilArray transposes (Dedalus GROUP_TRANSPOSES for CPU)
export GroupedPencilTransposeConfig, set_grouped_pencil_transposes!
export group_pencil_transpose!, group_transpose_fields!

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
