"""
    Field copying and allocation

This file contains ScalarField copy/deepcopy behavior, storage accessors,
architecture synchronization, and initial data allocation.
"""

# Copy methods for ScalarField
"""Create a shallow copy of ScalarField with copied data arrays.
Uses empty bases to construct a skeleton field without allocating data,
then copies only the source field's arrays — avoids double allocation."""
function Base.copy(field::ScalarField)
    new_field = ScalarField(field.dist, field.name, (), field.dtype)
    # Restore metadata from source field
    new_field.bases = field.bases
    new_field.domain = field.domain
    new_field.layout = field.layout
    new_field.current_layout = field.current_layout
    new_field.scales = field.scales
    new_field.fft_mode = field.fft_mode
    new_field.buffers.architecture = field.buffers.architecture
    # Only copy the live data array — the other is stale and will be
    # recomputed on next layout change via ensure_layout!
    if field.current_layout == :c
        if get_coeff_data(field) !== nothing
            set_coeff_data!(new_field, copy(get_coeff_data(field)))
        end
    else
        if get_grid_data(field) !== nothing
            set_grid_data!(new_field, copy(get_grid_data(field)))
        end
    end
    return new_field
end

function Base.deepcopy_internal(field::ScalarField, stackdict::IdDict)
    # Return existing copy if already visited (cycle detection)
    haskey(stackdict, field) && return stackdict[field]::ScalarField

    # Construct skeleton field without data allocation
    new_field = ScalarField(field.dist, field.name, (), field.dtype)
    stackdict[field] = new_field  # Register before recursing to break cycles

    # Deep-copy mutable metadata with cycle tracking
    new_field.bases = Base.deepcopy_internal(field.bases, stackdict)
    new_field.domain = field.domain === nothing ? nothing : Base.deepcopy_internal(field.domain, stackdict)
    # Layout is a struct with immutable fields — shallow copy is fine
    new_field.layout = field.layout
    new_field.current_layout = field.current_layout
    new_field.scales = field.scales
    new_field.fft_mode = field.fft_mode
    new_field.buffers.architecture = field.buffers.architecture
    # Deep-copy only the data arrays that exist
    if get_grid_data(field) !== nothing
        set_grid_data!(new_field, Base.deepcopy_internal(get_grid_data(field), stackdict))
    end
    if get_coeff_data(field) !== nothing
        set_coeff_data!(new_field, Base.deepcopy_internal(get_coeff_data(field), stackdict))
    end
    return new_field
end

# Data allocation and management
coefficient_eltype(dtype::Type) = dtype <: Complex ? dtype : Complex{dtype}

# Typed length-0 placeholder so storage is never `nothing`. Grid uses the field
# element type; coeff uses the complex coefficient type. The 1-D length-0 arrays
# are never indexed (every 0-D-field consumer guards with isempty(field.bases)).
_empty_grid(::Type{T}) where {T} = Array{T,1}(undef, 0)
_empty_coeff(::Type{T}) where {T} = Array{coefficient_eltype(T),1}(undef, 0)

"""
    field_architecture(field::ScalarField)

Return the architecture where the field's storage currently lives.
"""
field_architecture(field::ScalarField) = field.buffers.architecture

"""
    synchronize_field_architecture!(field::ScalarField; arch=field.dist.architecture, move_grid::Bool=true, move_coefficients::Bool=true)

Ensure a field's stored arrays live on the requested architecture.
Moves grid (`data_g`) and coefficient (`data_c`) arrays via `on_architecture` when requested.
"""
function synchronize_field_architecture!(field::ScalarField; arch::AbstractArchitecture=field.dist.architecture,
                                          move_grid::Bool=true, move_coefficients::Bool=true)
    if move_grid && get_grid_data(field) !== nothing && !(field.dist.use_pencil_arrays)
        if architecture(get_grid_data(field)) != arch
            set_grid_data!(field, on_architecture(arch, get_grid_data(field)))
        end
    end
    if move_coefficients && get_coeff_data(field) !== nothing && !(field.dist.use_pencil_arrays)
        if architecture(get_coeff_data(field)) != arch
            set_coeff_data!(field, on_architecture(arch, get_coeff_data(field)))
        end
    end
    field.buffers.architecture = arch
    return field
end

"""
    Allocate data for field following proper PencilArrays pattern.

    Key principles:
    1. For MPI (use_pencil_arrays=true): Store PencilArray objects to maintain distribution
    2. For serial: Use regular arrays on the appropriate architecture (CPU/GPU)
    3. NEVER convert Pencil to Array - work with pencil.data for local access
    4. For RealFourier bases, coefficient array has different size (N/2 + 1 complex values)
    5. For GPU architecture, allocate on GPU using CuArray (via architecture abstraction)
    """
function allocate_data!(field::ScalarField)
    if field.domain === nothing
        return
    end

    # IMPORTANT: Stored field buffers use `basis.meta.size` (the
    # non-dealiased grid), NOT the 1.5× padded size suggested by a
    # basis's `dealias` argument. The padded shape is a transient scratch
    # used inside `NonlinearEvaluator.evaluate_padded_multiply` for
    # quadratic product dealiasing (Orszag 1971 3/2-rule); storing every
    # state field at the padded size would be 3-3.4× wasteful in memory
    # and time for linear operators that don't need dealiasing.
    # See docs/src/pages/dealiasing.md for the full story.
    gshape = global_shape(field.domain)
    # Coefficient shape: halve the FIRST Fourier axis via rfft, leave
    # subsequent Fourier axes at full size (subsequent transforms see
    # complex input and use fft, not rfft). See `_coefficient_shape_impl`
    # in domain.jl for the unified serial+MPI rule.
    cshape = get_coefficient_shape_for_context(field.domain, field.dist)
    arch = field.dist.architecture

    if field.dist.use_pencil_arrays
        # CORRECT: Store PencilArray objects directly for MPI parallelization
        # The Pencil object maintains decomposition information needed for:
        # - Transpose operations between decompositions
        # - PencilFFT transforms
        # - MPI communication patterns
        # Note: PencilArrays currently only supports CPU. For GPU+MPI, use serial GPU per rank.

        coeff_dtype = coefficient_eltype(field.dtype)

        # CRITICAL: Use PencilFFTs.allocate_input/allocate_output for compatibility
        # These functions are GUARANTEED to create arrays that work with the plan's mul!/ldiv!
        # Using PencilArray{T}(undef, pencil) can fail if the pencil doesn't match exactly
        pencil_plan = _find_pencil_plan(field.dist)

        if pencil_plan !== nothing
            # Use PencilFFTs' official allocators - guaranteed to be compatible
            set_grid_data!(field, PencilFFTs.allocate_input(pencil_plan))
            set_coeff_data!(field, PencilFFTs.allocate_output(pencil_plan))
        elseif field.dist.pencil_fft_input !== nothing && field.dist.pencil_fft_output !== nothing
            # Fallback to stored pencils (less safe but should work)
            set_grid_data!(field, PencilArrays.PencilArray{field.dtype}(undef, field.dist.pencil_fft_input))
            set_coeff_data!(field, PencilArrays.PencilArray{coeff_dtype}(undef, field.dist.pencil_fft_output))
        else
            # Last resort: create new pencils (may not be compatible with PencilFFTs)
            set_grid_data!(field, create_pencil(field.dist, gshape, nothing, dtype=field.dtype))
            set_coeff_data!(field, create_pencil(field.dist, cshape, nothing, dtype=coeff_dtype))
        end

        fill!(get_grid_data(field), zero(field.dtype))
        fill!(get_coeff_data(field), zero(coeff_dtype))

        # For local computations on pencil.data
    else
        # Serial computation - use architecture-aware array creation
        local_gsize = get_local_array_size(field.dist, gshape)
        local_csize = get_local_array_size(field.dist, cshape)
        coeff_dtype = coefficient_eltype(field.dtype)

        # Create arrays on the appropriate architecture (CPU or GPU)
        set_grid_data!(field, zeros(arch, field.dtype, local_gsize...))
        set_coeff_data!(field, zeros(arch, coeff_dtype, local_csize...))
    end

    field.buffers.architecture = arch
end

const GPU_FFT_MODES = (:auto, :cpu, :gpu)

"""
    gpu_fft_mode(field::ScalarField)

Return the GPU FFT preference (:auto, :cpu, or :gpu).
"""
gpu_fft_mode(field::ScalarField) = field.fft_mode

"""
    set_gpu_fft_mode!(field::ScalarField, mode::Symbol)

Set the GPU FFT preference for a field. Use `:gpu` to force GPU transforms,
`:cpu` to keep them on CPU, or `:auto` to rely on heuristics.
"""
function set_gpu_fft_mode!(field::ScalarField, mode::Symbol)
    mode in GPU_FFT_MODES || throw(ArgumentError("Invalid GPU FFT mode $mode (expected :auto, :cpu, or :gpu)"))
    field.fft_mode = mode
    return field
end

"""
    get_grid_data(field::ScalarField)

Return the raw grid-space data array **without** transforming.
Does not check or change the current layout — use only when you know
the field is already in grid space.

For auto-transforming access, use `grid_data(field)` or `field["g"]` instead.
"""
@inline get_grid_data(field::ScalarField) = getfield(getfield(field, :storage), :grid)

"""
    get_coeff_data(field::ScalarField)

Return the raw coefficient-space data array **without** transforming.
Does not check or change the current layout — use only when you know
the field is already in coefficient space.

For auto-transforming access, use `coeff_data(field)` or `field["c"]` instead.
"""
@inline get_coeff_data(field::ScalarField) = getfield(getfield(field, :storage), :coeff)

"""
    set_grid_data!(field::ScalarField, data)

Assign the grid data array while keeping buffer metadata consistent.
"""
@inline function set_grid_data!(field::ScalarField, data)
    storage = getfield(field, :storage)
    setfield!(storage, :grid, data)
    _update_field_buffer_architecture!(storage, data)
    return field
end

"""
    set_coeff_data!(field::ScalarField, data)

Assign the coefficient data array while keeping buffer metadata consistent.
"""
@inline function set_coeff_data!(field::ScalarField, data)
    storage = getfield(field, :storage)
    setfield!(storage, :coeff, data)
    _update_field_buffer_architecture!(storage, data)
    return field
end

"""
    Get local array size for this process based on MPI decomposition.

    IMPORTANT: This function assumes FULL MESH decomposition (no pencil/local dimension).
    It uses different conventions depending on use_pencil_arrays:

    When use_pencil_arrays=true (CPU+MPI with PencilFFTs):
    - PencilArrays convention: decompose LAST ndims_mesh dimensions
    - 3D with 3D mesh: all dims decomposed
    - 3D with 2D mesh: dims 2,3 decomposed (x LOCAL not reflected here!)
    - 2D with 2D mesh: x and y decomposed

    When use_pencil_arrays=false (GPU+MPI or TransposableField):
    - TransposableField ZLocal convention: decompose FIRST ndims_mesh dimensions
    - 3D with 2D mesh: x decomposed by Rx, y decomposed by Ry, z LOCAL
    - 2D with 2D mesh: x decomposed by Rx, y decomposed by Ry

    WARNING: For pencil decomposition (e.g., FFT with decomp_index=1 keeping dim 1 local),
    this function's result may NOT match the actual PencilArray layout. For pencil-specific
    operations, use size(pencil_array) for local size or PencilArrays.size_global() for global.

    Arguments:
    - dist: Distributor with MPI decomposition info
    - global_shape: Tuple of global array dimensions

    Returns:
    - Tuple of local array dimensions for this process
    """
function get_local_array_size(dist::Distributor, global_shape::Tuple)
    # Serial case: local = global
    if dist.size == 1 || dist.mesh === nothing
        return global_shape
    end

    mesh = dist.mesh
    ndims_global = length(global_shape)
    ndims_mesh = length(mesh)

    # CRITICAL: Validate mesh dimensionality vs domain dimensionality
    # When mesh has more dimensions than domain, we can only use min(ndims_global, ndims_mesh) dimensions
    # This can lead to underutilized mesh dimensions and desync with PencilArrays
    if ndims_mesh > ndims_global
        effective_mesh_dims = ndims_global
        unused_mesh_dims = ndims_mesh - ndims_global
        unused_procs = prod(mesh[effective_mesh_dims+1:end])
        if unused_procs > 1
            @warn "Mesh dimensionality ($ndims_mesh) exceeds domain dimensionality ($ndims_global). " *
                  "Only first $effective_mesh_dims mesh dimensions will be used for decomposition. " *
                  "This leaves $(unused_mesh_dims) mesh dimension(s) unutilized, potentially wasting " *
                  "$unused_procs MPI process(es). Consider using a mesh with $ndims_global dimensions, " *
                  "e.g., mesh=$(Tuple(mesh[1:effective_mesh_dims]))." maxlog=1
        end
    end

    local_shape = collect(global_shape)

    # Get process coordinates for all mesh dimensions using general formula
    # For a mesh (P₁, P₂, ..., Pₖ), process with rank r has coordinates:
    # coord[i] = (r ÷ (P₁×P₂×...×Pᵢ₋₁)) % Pᵢ
    coords = Vector{Int}(undef, ndims_mesh)
    stride = 1
    for i in 1:ndims_mesh
        coords[i] = (dist.rank ÷ stride) % mesh[i]
        stride *= mesh[i]
    end

    if dist.use_pencil_arrays
        # PencilArrays convention: decompose LAST ndims_mesh dimensions
        # For 3D with 3D mesh: all dims decomposed
        # For 3D with 2D mesh: dims 2,3 decomposed; dim 1 local
        # For 2D with 2D mesh: dims 1,2 decomposed

        if ndims_global >= ndims_mesh
            # Decompose last dimensions
            decomp_start = ndims_global - ndims_mesh + 1

            for mesh_idx in 1:ndims_mesh
                dim = decomp_start + mesh_idx - 1
                if dim <= ndims_global
                    n_global = global_shape[dim]
                    n_procs = mesh[mesh_idx]
                    proc_coord = coords[mesh_idx]

                    base_size = div(n_global, n_procs)
                    remainder = n_global % n_procs
                    local_shape[dim] = base_size + (proc_coord < remainder ? 1 : 0)
                end
            end
        end
    else
        # TransposableField ZLocal convention: decompose FIRST ndims_mesh dimensions
        # mesh[1] (Rx) decomposes dimension 1 (x)
        # mesh[2] (Ry) decomposes dimension 2 (y)
        # Higher dimensions (z, etc.) remain LOCAL
        # NOTE: TransposableField only supports 2D mesh, so we limit to 2 dims

        for mesh_idx in 1:min(ndims_mesh, 2)  # TransposableField max 2D
            dim = mesh_idx
            if dim <= ndims_global
                n_global = global_shape[dim]
                n_procs = mesh[mesh_idx]
                proc_coord = coords[mesh_idx]

                base_size = div(n_global, n_procs)
                remainder = n_global % n_procs
                local_shape[dim] = base_size + (proc_coord < remainder ? 1 : 0)
            end
        end

        # Dimensions >= 3 (z, etc.) remain LOCAL (not decomposed)
    end

    return tuple(local_shape...)
end
