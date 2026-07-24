# Utility functions for PencilArray compatibility
"""
    Convert field data to PencilArray format compatible with the given PencilConfig.

    This function ensures that the field's data is:
    1. In grid space layout (for nonlinear operations)
    2. Compatible with the PencilConfig's global shape
    3. Uses the correct data type
    4. Properly distributed according to the mesh configuration

    Returns the field's grid space data as a PencilArray or compatible array.
    """
function get_pencil_compatible_data(field::ScalarField, config::PencilConfig)

    # Ensure field is in grid space layout for nonlinear operations
    ensure_layout!(field, :g)

    # Verify field has allocated data
    if get_grid_data(field) === nothing
        throw(ArgumentError("Field $(field.name) has no grid space data allocated"))
    end

    field_data = get_grid_data(field)
    field_shape = size(field_data)

    # PencilArrays is CPU-only. GPU nonlinear paths use TransposableField.
    if is_gpu_array(field_data)
        error("PencilArray conversion is unavailable for GPU fields; CPU fallback is disabled.")
    else
        field_pencil = field_data
    end

    # Verify shape compatibility
    # The field's local shape should be consistent with the global shape and mesh decomposition
    if !is_shape_compatible(field_shape, config.global_shape, config.mesh, config.comm;
                            use_pencil_arrays=field.dist.use_pencil_arrays)
        @warn "Shape mismatch: field local shape $(field_shape) may not be compatible with " *
              "global shape $(config.global_shape) and mesh $(config.mesh)"
    end

    # Verify data type compatibility and convert if needed
    if eltype(field_pencil) != config.dtype
        @debug "Converting field data type from $(eltype(field_pencil)) to $(config.dtype)"
        # Create converted copy if types don't match
        # Preserve PencilArray wrapper if present (for MPI parallelization)
        if isa(field_pencil, PencilArrays.PencilArray)
            # Create a new PencilArray with the same structure but new dtype
            # Use similar() to preserve the pencil configuration, then copyto! converted data
            pencil = PencilArrays.pencil(field_pencil)
            converted_pencil = PencilArrays.PencilArray{config.dtype}(undef, pencil)
            # Convert and copy data
            src_data = parent(field_pencil)
            dest_data = parent(converted_pencil)
            copyto!(dest_data, convert.(config.dtype, src_data))
            return converted_pencil
        else
            converted_data = convert.(config.dtype, field_pencil)
            return converted_data
        end
    end

    # Verify MPI communicator compatibility
    if field.dist.use_pencil_arrays && field.dist.pencil_config !== nothing
        if field.dist.pencil_config.comm != config.comm
            @warn "MPI communicator mismatch between field and config"
        end
    end

    @debug "Retrieved pencil compatible data for field $(field.name)" size=field_shape eltype=eltype(field_pencil)

    return field_pencil
end

function is_shape_compatible(local_shape::Tuple, global_shape::Tuple, mesh::Tuple, comm::MPI.Comm;
                             use_pencil_arrays::Bool=true)
    """
    Check if the local shape is compatible with the global shape given the mesh decomposition.

    For a valid pencil decomposition:
    - The product of local shapes across all ranks should equal the global shape
    - The local shape should be approximately global_shape / mesh for distributed dimensions

    The `use_pencil_arrays` flag controls which dimensions are expected to be decomposed:
    - true (default): PencilArrays convention - decompose LAST dimensions
    - false: TransposableField ZLocal convention - decompose FIRST dimensions
    """

    if length(local_shape) != length(global_shape)
        return false
    end

    # For serial execution (single rank), local shape should match global shape
    if MPI.Comm_size(comm) == 1
        return local_shape == global_shape
    end

    # For parallel execution, check that local shape is reasonable
    # (within expected range given the mesh decomposition)
    num_dims = length(global_shape)
    mesh_dims = length(mesh)

    for i in 1:num_dims
        # Determine if this dimension is decomposed based on convention
        is_decomposed = if use_pencil_arrays
            # PencilArrays: decompose LAST mesh_dims dimensions
            # For 3D with 2D mesh: dims 2,3 decomposed; dim 1 local
            decomp_start = num_dims - mesh_dims + 1
            mesh_idx = i - decomp_start + 1
            i >= decomp_start && mesh_idx >= 1 && mesh_idx <= mesh_dims && mesh[mesh_idx] > 1
        else
            # TransposableField ZLocal: decompose FIRST mesh_dims dimensions
            # mesh[1] (Rx) decomposes dim 1, mesh[2] (Ry) decomposes dim 2, etc.
            i <= mesh_dims && mesh[i] > 1
        end

        if is_decomposed
            # This dimension is distributed - get the correct mesh divisor
            mesh_idx = if use_pencil_arrays
                decomp_start = num_dims - mesh_dims + 1
                i - decomp_start + 1
            else
                i
            end

            expected_local = ceil(Int, global_shape[i] / mesh[mesh_idx])
            min_local = floor(Int, global_shape[i] / mesh[mesh_idx])

            if local_shape[i] < min_local || local_shape[i] > expected_local
                return false
            end
        else
            # This dimension is not distributed, should match global
            if local_shape[i] != global_shape[i]
                return false
            end
        end
    end

    return true
end

"""
    Extract a PencilConfig from a ScalarField's distributor configuration.
    """
function get_pencil_config_from_field(field::ScalarField)
    dist = field.dist

    if dist.pencil_config !== nothing
        return dist.pencil_config
    end

    # Build a config from field properties
    if field.domain === nothing
        throw(ArgumentError("Field $(field.name) has no domain set - cannot create PencilConfig"))
    end
    gshape = global_shape(field.domain)
    mesh_config = dist.mesh !== nothing ? dist.mesh : (1,)

    return PencilConfig(
        gshape,
        mesh_config;
        comm=dist.comm,
        dtype=field.dtype
    )
end

"""
    compute_local_shape(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)

Compute the local shape for this rank given the global shape and mesh decomposition.

The mesh defines how processes are arranged in a Cartesian grid. For example,
mesh=(2,4) means 2 processes in dimension 1 and 4 in dimension 2, for 8 total.

The global array is decomposed such that each dimension is split among the
processes in that mesh dimension. Load balancing distributes remainders
to the first ranks in each dimension.

# Arguments
- `global_shape`: Total size of the array in each dimension
- `mesh`: Number of processes in each decomposition dimension
- `comm`: MPI communicator

# Returns
- Tuple of local sizes for each dimension on this rank

# Example
```julia
# 4 processes with mesh (2, 2), global shape (100, 100)
# Each process gets local shape (50, 50)
local_shape = compute_local_shape((100, 100), (2, 2), comm)
```
"""
function compute_local_shape(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)
    mpi_rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    num_dims = length(global_shape)
    mesh_dims = length(mesh)

    # Validate mesh configuration
    mesh_size = prod(mesh)
    if mesh_size != nprocs
        @warn "Mesh size ($mesh_size) doesn't match number of processes ($nprocs)"
    end

    local_shape = collect(global_shape)

    # Compute mesh coordinates for this rank using row-major (C-style) ordering
    # mpi_rank = coord[1] + coord[2]*mesh[1] + coord[3]*mesh[1]*mesh[2] + ...
    mesh_coords = zeros(Int, mesh_dims)
    remaining_rank = mpi_rank
    for i in 1:mesh_dims
        mesh_coords[i] = remaining_rank % mesh[i]
        remaining_rank = remaining_rank ÷ mesh[i]
    end

    # CRITICAL: Decompose LAST dimensions to match Distributor convention
    # Distributor uses: decomp_dims = ntuple(i -> ndim - ndims_mesh + i, ndims_mesh)
    # This means for 3D data with 2D mesh: decompose dims (2, 3), keep dim 1 local
    # For 2D data with 1D mesh: decompose dim 2, keep dim 1 local

    # Compute local sizes for each decomposed dimension
    for i in 1:min(num_dims, mesh_dims)
        # Map mesh dimension i to global dimension (last dimensions)
        global_dim_idx = num_dims - mesh_dims + i

        if mesh[i] > 1 && global_dim_idx >= 1
            n = global_shape[global_dim_idx]  # Global size in this dimension
            p = mesh[i]                        # Number of processes in this dimension
            coord = mesh_coords[i]             # This rank's position in mesh dimension i

            # Compute local size with load balancing
            # First (remainder) ranks get one extra element
            base_size = n ÷ p
            remainder = n % p

            if coord < remainder
                # First 'remainder' ranks get base_size + 1
                local_shape[global_dim_idx] = base_size + 1
            else
                local_shape[global_dim_idx] = base_size
            end
        end
        # Dimensions before decomp_dims keep their global size (local)
    end

    return tuple(local_shape...)
end

"""
    compute_local_range(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)

Compute the global index ranges owned by this rank for each dimension.

# Returns
- Vector of (start, stop) tuples for each dimension (1-based indices)
"""
function compute_local_range(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)
    mpi_rank = MPI.Comm_rank(comm)
    num_dims = length(global_shape)
    mesh_dims = length(mesh)

    # Compute mesh coordinates
    mesh_coords = zeros(Int, mesh_dims)
    remaining_rank = mpi_rank
    for i in 1:mesh_dims
        mesh_coords[i] = remaining_rank % mesh[i]
        remaining_rank = remaining_rank ÷ mesh[i]
    end

    # Determine which dimensions are decomposed
    # Convention: mesh decomposes the LAST dimensions
    # For 3D data with 2D mesh: mesh[1] -> dim 2, mesh[2] -> dim 3
    # For 3D data with 1D mesh: mesh[1] -> dim 3
    decomp_start = num_dims - mesh_dims + 1

    ranges = Vector{Tuple{Int,Int}}(undef, num_dims)

    for i in 1:num_dims
        # Map dimension i to mesh dimension (if decomposed)
        mesh_idx = i - decomp_start + 1

        if mesh_idx >= 1 && mesh_idx <= mesh_dims && mesh[mesh_idx] > 1
            n = global_shape[i]
            p = mesh[mesh_idx]
            coord = mesh_coords[mesh_idx]

            base_size = n ÷ p
            remainder = n % p

            if coord < remainder
                local_size = base_size + 1
                start_idx = coord * (base_size + 1) + 1
            else
                local_size = base_size
                start_idx = remainder * (base_size + 1) + (coord - remainder) * base_size + 1
            end

            ranges[i] = (start_idx, start_idx + local_size - 1)
        else
            # Not decomposed in this dimension
            ranges[i] = (1, global_shape[i])
        end
    end

    return ranges
end

"""
    Interpolate source data into destination array.
    Uses nearest-neighbor or linear interpolation depending on relative sizes.
    """
function interpolate_field_data!(dest::AbstractArray, src::AbstractArray)
    src_shape = size(src)
    dest_shape = size(dest)

    if src_shape == dest_shape
        copyto!(dest, src)
        return
    end

    # Use nearest-neighbor interpolation for simplicity
    num_dims = length(dest_shape)

    for I in CartesianIndices(dest)
        src_indices = ntuple(num_dims) do d
            # Map destination index to source index
            src_idx = round(Int, (I[d] - 1) * (src_shape[d] - 1) / max(dest_shape[d] - 1, 1)) + 1
            clamp(src_idx, 1, src_shape[d])
        end
        dest[I] = src[src_indices...]
    end
end

"""
    Set field data from PencilArray format.
    Since ScalarField stores data as PencilArrays, this mainly ensures
    proper layout and copies the data.
    """
function set_pencil_compatible_data!(field::ScalarField, data, config::PencilConfig)

    # Ensure field is in grid space layout
    ensure_layout!(field, :g)

    # Verify that field has allocated data
    if get_grid_data(field) === nothing
        throw(ArgumentError("Field $(field.name) has no grid space data allocated"))
    end

    # Verify data compatibility
    if size(data) != size(get_grid_data(field))
        throw(DimensionMismatch("Data size $(size(data)) does not match field size $(size(get_grid_data(field)))"))
    end

    if eltype(data) != eltype(get_grid_data(field))
        @warn "Data type mismatch during set: incoming $(eltype(data)), field $(eltype(get_grid_data(field)))"
    end

    # Copy data into the field's PencilArray, respecting architecture
    arr = data
    if eltype(get_grid_data(field)) != eltype(arr)
        arr = convert.(eltype(get_grid_data(field)), arr)
    end

    arch = field.dist.architecture
    if is_gpu_array(get_grid_data(field))
        arr = on_architecture(arch, arr)
    end

    copyto!(get_grid_data(field), arr)

    # Mark field as having valid grid space data
    field.current_layout = :g

    @debug "Set pencil compatible data for field $(field.name)" size(data) eltype(data)
end

# Memory management
"""Get temporary field for intermediate calculations """
function get_temp_field(evaluator::NonlinearEvaluator, template::ScalarField, name::String)

    key = "$(name)_$(hash(template.bases))"

    if !haskey(evaluator.temp_fields, key)
        temp_field = ScalarField(template.dist, name, template.bases, template.dtype)

        # Ensure temporary field has allocated data
        ensure_layout!(temp_field, :g)

        evaluator.temp_fields[key] = temp_field
    end

    return evaluator.temp_fields[key]
end

"""Clear temporary fields to free memory"""
function clear_temp_fields!(evaluator::NonlinearEvaluator)
    empty!(evaluator.temp_fields)
    empty!(evaluator.nl_result_pool)
    GC.gc()
end

"""
    get_temp_array(evaluator::NonlinearEvaluator, shape::Tuple, dtype::Type)

Get temporary array for intermediate calculations.
Architecture-aware: allocates on GPU if evaluator uses GPU architecture.
"""
function get_temp_array(evaluator::NonlinearEvaluator, shape::Tuple, dtype::Type)
    arch = architecture(evaluator)
    if is_gpu(arch)
        # GPU: use architecture-aware allocation
        return create_array(arch, dtype, shape...)
    else
        # CPU: standard zeros allocation
        return zeros(dtype, shape...)
    end
end

"""
    return_temp_array!(evaluator::NonlinearEvaluator, arr::AbstractArray)

Return temporary array to pool or free GPU memory.
For CPU, this is a no-op (GC handles memory).
For GPU, this can free memory explicitly if needed.
"""
function return_temp_array!(evaluator::NonlinearEvaluator, arr::AbstractArray)
    if is_gpu(evaluator) && is_gpu_array(arr)
        # For GPU arrays, we can explicitly free if needed
        # unsafe_free!(arr)  # Uncomment if explicit memory management is needed
    end
    return nothing
end
