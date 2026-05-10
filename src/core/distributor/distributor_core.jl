# Core distributor types, topology setup, layout creation, and local indexing.
#
# This file owns the constructor path and the shape/layout logic that the rest
# of the MPI and transpose helpers build on.

# MPI, PencilArrays, LinearAlgebra already in Tarang.jl

struct Layout{N}
    dist::Any
    local_shape::NTuple{N, Int}
    global_shape::NTuple{N, Int}
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
    pencil_fft_plan::Any  # Cached PencilFFTs.PencilFFTPlan reference (avoids Vector{Any} scan)

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
                        device::Union{AbstractArchitecture, Nothing}=nothing,
                        arch::Union{AbstractArchitecture, Nothing}=nothing,
                        architecture::AbstractArchitecture=CPU(),
                        use_pencil_arrays::Union{Nothing, Bool}=nothing)
        # Allow `device=`, `arch=`, and `architecture=` kwargs; device > arch > architecture
        architecture = device !== nothing ? device : (arch !== nothing ? arch : architecture)

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

        # Route based on device:
        #   device=CPU() + MPI → PencilArrays/PencilFFTs for distributed FFTs
        #   device=GPU()       → CUDA/cuFFT, TransposableField for MPI communication
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
        pencil_fft_plan = nothing
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

        # Log device and parallelization mode
        if rank == 0
            if is_gpu(architecture)
                if size > 1
                    @info "device=GPU(): using CUDA/cuFFT with TransposableField for MPI ($size processes)"
                else
                    @info "device=GPU(): using CUDA/cuFFT (single GPU)"
                end
            else
                if size > 1
                    @info "device=CPU(): using PencilArrays/PencilFFTs for MPI ($size processes)"
                else
                    @info "device=CPU(): serial execution"
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
            pencil_fft_plan, pencil_fft_input, pencil_fft_output, layouts, perf_stats,
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

"""Setup PencilArrays configuration for given global shape"""
function setup_pencil_arrays(dist::Distributor, global_shape::Tuple{Vararg{Int}})

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
    # decomp_dims for PencilConfig is NTuple{M, Bool} matching the mesh dimensions
    decomp_flags = ntuple(_ -> true, ndims_mesh)

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

"""Get layout for given bases"""
function get_layout(dist::Distributor, bases::Tuple{Vararg{Basis}}, dtype::Type=dist.dtype)

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
    global_shape = ntuple(i -> bases[i].meta.size, length(bases))

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

"""
    Get local indices for the given axis (1-indexed).
    For serial execution, returns all indices.
    For parallel execution, returns the indices owned by this process.
    """
function local_indices(dist::Distributor, axis::Int)
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

"""
    Get local indices for the given axis with known global size.

    Respects dist.use_pencil_arrays for decomposition convention:
    - PencilArrays: decompose LAST ndims_mesh dimensions
    - TransposableField: decompose FIRST ndims_mesh dimensions

    Note: This returns the local indices for the actual field decomposition,
    which uses full mesh decomposition (not pencil decomposition with a local dim).
    For pencil-specific operations, use explicit decomp_dims.
    """
function local_indices(dist::Distributor, axis::Int, global_size::Int)
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

"""Create a scalar field"""
function Field(dist::Distributor; name::String="field", bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
    field = ScalarField(dist, name, bases, dtype)
    return field
end

# Note: VectorField and TensorField convenience constructors are defined in field.jl
# to avoid conflict with the struct definitions (functions and structs cannot share names in Julia)

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
function local_grids(dist::Distributor, bases::Vararg{Basis}; scales=nothing, move_to_arch::Bool=true)
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

"""Process and validate scales parameter."""
function remedy_scales(dist::Distributor, scales, num_bases)
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

"""
    Remedy different scale inputs.
    Following implementation in distributor:188-197
    """
function remedy_scales(dist::Distributor, scales)
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

"""Get axis index for a coordinate."""
function get_axis(dist::Distributor, coord::Coordinate)
    for (i, c) in enumerate(dist.coords)
        if c.coordsys == coord.coordsys && c.name == coord.name
            return i - 1  # 0-indexed
        end
    end
    throw(ArgumentError("Coordinate $(coord.name) not found in distributor"))
end

"""Get axis for coordinate system (uses first coordinate)."""
function get_axis(dist::Distributor, coordsys::CoordinateSystem)
    return get_axis(dist, coords(coordsys)[1])
end

"""Get axis index for a basis."""
function get_basis_axis(dist::Distributor, basis::Basis)
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

"""
    Get first axis index for a basis.
    Following implementation in distributor:210
    """
function first_axis(dist::Distributor, basis::Basis)
    return get_basis_axis(dist, basis)
end

"""
    Get last axis index for a basis.
    Following implementation in distributor:213
    """
function last_axis(dist::Distributor, basis::Basis)
    return first_axis(dist, basis) + basis.meta.dim - 1
end
