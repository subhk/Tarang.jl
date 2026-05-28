"""
    Field copying and allocation

This file contains ScalarField copy/deepcopy behavior, storage accessors,
architecture synchronization, and initial data allocation.
"""

# Copy methods for ScalarField
"""Create a shallow copy of ScalarField with copied data arrays.
Constructs the skeleton with the source field's REAL bases so the parametric
`SerialFieldStorage{G,C}` is allocated with matching concrete array types, then
copies the live-layout array's values in place (the other array stays zeroed and
is recomputed on the next layout change via `ensure_layout!`)."""
function Base.copy(field::ScalarField)
    # Build with the real bases → correctly-typed (zeroed) storage arrays.
    new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
    # Restore state not derived from bases. Scales MUST be restored before
    # adopting the source arrays: a non-default-scaled source has differently
    # sized arrays, so the default-scale skeleton arrays won't match.
    new_field.layout = field.layout
    new_field.scales = field.scales
    new_field.current_layout = field.current_layout
    new_field.fft_mode = field.fft_mode
    new_field.buffers.architecture = field.buffers.architecture
    # Adopt a copy of the live-layout array (sized to the source, incl. scales);
    # shrink the off-layout array to a 0-sized buffer of the same concrete type so
    # the copy stays lazy (the stale layout is recomputed on the next
    # ensure_layout!). PencilArrays can't be resized to 0, so they stay allocated.
    if field.current_layout == :c
        set_coeff_data!(new_field, copy(get_coeff_data(field)))
        set_grid_data!(new_field, _shrink_off_layout(get_grid_data(new_field)))
    else
        set_grid_data!(new_field, copy(get_grid_data(field)))
        set_coeff_data!(new_field, _shrink_off_layout(get_coeff_data(new_field)))
    end
    return new_field
end

# Shrink an off-layout buffer to a 0-sized array of the SAME concrete type so the
# parametric storage type is preserved while holding no data. Re-materialized on
# the next ensure_layout!. PencilArrays are left as-is (can't be cleanly 0-sized).
_shrink_off_layout(a::Array) = similar(a, ntuple(_ -> 0, ndims(a)))
_shrink_off_layout(a) = a

function Base.deepcopy_internal(field::ScalarField, stackdict::IdDict)
    # Return existing copy if already visited (cycle detection)
    haskey(stackdict, field) && return stackdict[field]::ScalarField

    # Build with the real bases → correctly-typed (zeroed) storage arrays so the
    # parametric SerialFieldStorage{G,C} matches the source's concrete types.
    new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
    stackdict[field] = new_field  # Register before recursing to break cycles

    # Deep-copy mutable metadata (bases/domain) with cycle tracking — overwriting
    # these struct fields does not change the already-locked storage array types.
    new_field.bases = Base.deepcopy_internal(field.bases, stackdict)
    new_field.domain = field.domain === nothing ? nothing : Base.deepcopy_internal(field.domain, stackdict)
    # Restore remaining state not derived from bases (scales before adopting the
    # source arrays — see Base.copy for why).
    new_field.layout = field.layout
    new_field.scales = field.scales
    new_field.current_layout = field.current_layout
    new_field.fft_mode = field.fft_mode
    new_field.buffers.architecture = field.buffers.architecture
    # Deep-copy the live-layout array; keep the off-layout array lazy (0-sized,
    # same concrete type) — recomputed on the next ensure_layout!.
    if field.current_layout == :c
        set_coeff_data!(new_field, Base.deepcopy_internal(get_coeff_data(field), stackdict))
        set_grid_data!(new_field, _shrink_off_layout(get_grid_data(new_field)))
    else
        set_grid_data!(new_field, Base.deepcopy_internal(get_grid_data(field), stackdict))
        set_coeff_data!(new_field, _shrink_off_layout(get_coeff_data(new_field)))
    end
    return new_field
end

# Data allocation and management
coefficient_eltype(dtype::Type) = dtype <: Complex ? dtype : Complex{dtype}

"""
    coefficient_eltype(domain::Domain, ::Type{T})

Basis-aware coefficient element type. Fourier transforms (RealFourier /
ComplexFourier) map real grid data to COMPLEX coefficients, while Jacobi-family
transforms (Chebyshev*, Legendre, Jacobi, Ultraspherical) preserve the element
type (real grid data → real coefficients — see each transform's
`_forward_output_spec`). So the coefficient array is complex iff the domain has
at least one Fourier axis; otherwise it stays at the grid element type `T`.

This mirrors the basis-aware coefficient *shape* rule in `domain.jl`
(`_coefficient_shape_impl`): both are needed so the pre-allocated `coeff` buffer
matches exactly what the transform chain produces — which the parametric
`SerialFieldStorage{G,C}` now requires, since the coeff array type is frozen at
construction (the old mutable-slot path silently swapped a complex buffer for a
real one on the first transform of a pure-Jacobi field)."""
function coefficient_eltype(domain::Domain, ::Type{T}) where {T}
    has_fourier = any(b -> isa(b, FourierBasis), domain.bases)
    return has_fourier ? coefficient_eltype(T) : T
end

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
    # Fields are architecture-fixed: their array types are stable for life. A
    # same-architecture call is a no-op; a cross-architecture request is a bug
    # (build the field on the target architecture instead).
    field.buffers.architecture == arch ||
        throw(ArgumentError("synchronize_field_architecture!: in-place architecture moves are no longer supported " *
                            "(field on $(field.buffers.architecture), requested $arch). Construct the field on the target architecture."))
    return field
end

"""
    _build_field_arrays(dist, domain, dtype)

Value-returning allocator: builds and returns `(grid_array, coeff_array)` for
a field with the given distributor, domain, and element type, without touching
any field struct. Mirrors `allocate_data!` logic exactly so that Task 7 can
parametrize `SerialFieldStorage{G,C}` by calling this helper before constructing
the storage struct.

Behavior:
- MPI / pencil path: delegates to PencilFFTs.allocate_input/allocate_output
  (preferred), then to stored pencil objects, then to `create_pencil` as last resort.
  Arrays are zero-filled via `fill!` after allocation.
- Serial path: uses `zeros(arch, T, ...)` which already returns zeroed arrays.
"""
function _build_field_arrays(dist::Distributor, domain::Domain, ::Type{T}) where {T}
    # See allocate_data! for the rationale on shape choices (non-dealiased sizes).
    gshape = global_shape(domain)
    cshape = get_coefficient_shape_for_context(domain, dist)
    arch = dist.architecture
    # Basis-aware coeff eltype: complex for Fourier domains, real for pure-Jacobi
    # (Chebyshev/Legendre) domains — matches what the transform chain produces so
    # the parametric SerialFieldStorage{G,C} type is frozen correctly. The pencil
    # path always has a Fourier axis (pure-Jacobi MPI is rejected in
    # plan_transforms!), so this stays complex there as before.
    coeff_dtype = coefficient_eltype(domain, T)

    if dist.use_pencil_arrays
        pencil_plan = _find_pencil_plan(dist)

        if pencil_plan !== nothing
            # PencilFFTs' official allocators — guaranteed compatible with mul!/ldiv!
            g = PencilFFTs.allocate_input(pencil_plan)
            c = PencilFFTs.allocate_output(pencil_plan)
        elseif dist.pencil_fft_input !== nothing && dist.pencil_fft_output !== nothing
            # Fallback to stored pencils (less safe but should work)
            g = PencilArrays.PencilArray{T}(undef, dist.pencil_fft_input)
            c = PencilArrays.PencilArray{coeff_dtype}(undef, dist.pencil_fft_output)
        else
            # Last resort: create new pencils (may not be compatible with PencilFFTs)
            g = create_pencil(dist, gshape, nothing, dtype=T)
            c = create_pencil(dist, cshape, nothing, dtype=coeff_dtype)
        end

        fill!(g, zero(T))
        fill!(c, zero(coeff_dtype))
        return (g, c)
    else
        local_gsize = get_local_array_size(dist, gshape)
        local_csize = get_local_array_size(dist, cshape)
        return (zeros(arch, T, local_gsize...), zeros(arch, coeff_dtype, local_csize...))
    end
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
    field.domain === nothing && return
    g, c = _build_field_arrays(field.dist, field.domain, field.dtype)
    set_grid_data!(field, g)
    set_coeff_data!(field, c)
    field.buffers.architecture = field.dist.architecture
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
