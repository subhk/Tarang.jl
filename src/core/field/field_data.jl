# ============================================================================
# has() method - spectral pattern for checking if expression contains variables
# Field data access patterns
# ============================================================================

"""
    has(operand, vars...) -> Bool

Determine if expression tree contains any of the specified operands/operators.
This is the spectral pattern for linearity checking.

For fields: returns true if `operand in vars`
For operators: recursively checks operands/arguments
"""
function has(field::ScalarField, vars...)
    # Field.has: returns (not vars) or (self in vars)
    return isempty(vars) || field in vars
end

function has(field::VectorField, vars...)
    return isempty(vars) || field in vars
end

function has(field::TensorField, vars...)
    return isempty(vars) || field in vars
end

# Generic fallback for numbers/constants - they never contain variables
has(::Number, vars...) = false

"""
    require_linearity(expr, vars...; allow_affine=false)

Require expression to be linear in specified variables.
Following spectral pattern from arithmetic.py.

Raises error if expression is nonlinear in vars.
"""
function require_linearity end  # Forward declaration, implemented in operators.jl

"""
    LockedField <: Operand

A field wrapper that restricts layout and scale changes to specific allowed values.
Following field:LockedField pattern.

This is useful for:
- Output handlers that need fields in specific layouts
- Preventing accidental layout changes during evaluation
- Enforcing coefficient-space or grid-space operations

# Example
```julia
locked = LockedField(u, :g)          # Lock to grid space
locked = LockedField(u, :c)          # Lock to coefficient space
locked = LockedField(u, :g, (1,))    # Lock layout and scales
```
"""
struct LockedField <: Operand
    field::ScalarField
    layout::Symbol
    scales::Union{Nothing, Tuple}

    function LockedField(field::ScalarField, layout::Symbol, scales::Union{Nothing, Tuple}=nothing)
        # Ensure field is in the locked layout
        ensure_layout!(field, layout)
        if scales !== nothing
            normalized_scales = remedy_scales(field.dist, scales)
            set_scales!(field, normalized_scales)
            return new(field, layout, normalized_scales)
        end
        return new(field, layout, scales)
    end
end

function Base.getproperty(field::ScalarField, s::Symbol)
    if s === :buffers
        return getfield(field, :storage)  # backward-compatible :buffers → :storage
    elseif s === :data_g
        throw(ArgumentError("Direct access to field.data_g is deprecated. Use get_grid_data(field) instead."))
    elseif s === :data_c
        throw(ArgumentError("Direct access to field.data_c is deprecated. Use get_coeff_data(field) instead."))
    else
        return getfield(field, s)
    end
end

function Base.setproperty!(field::ScalarField, s::Symbol, value)
    if s === :buffers
        setfield!(field, :storage, value)  # backward-compatible :buffers → :storage
    elseif s === :data_g
        throw(ArgumentError("Assign to field.data_g via set_grid_data!(field, value) instead of direct property access."))
    elseif s === :data_c
        throw(ArgumentError("Assign to field.data_c via set_coeff_data!(field, value) instead of direct property access."))
    else
        setfield!(field, s, value)
    end
    return value
end

function Base.propertynames(::ScalarField, private::Bool=false)
    base_names = fieldnames(ScalarField)
    filtered = filter(n -> n ∉ (:data_g, :data_c), base_names)
    return (filtered..., :buffers)  # virtual property for backward compat
end

# LockedField convenience constructors
LockedField(field::ScalarField) = LockedField(field, field.current_layout, field.scales)

# Forward field access methods to underlying field
function Base.getindex(lf::LockedField, layout::String)
    layout_sym = Symbol(layout)
    if layout_sym != lf.layout
        throw(ArgumentError("Cannot access LockedField layout $layout (locked to $(lf.layout))"))
    end
    return getindex(lf.field, layout)
end

function Base.getindex(lf::LockedField, layout::Symbol)
    if layout != lf.layout
        throw(ArgumentError("Cannot access LockedField layout $layout (locked to $(lf.layout))"))
    end
    return layout == :g ? get_local_data(get_grid_data(lf.field)) : get_local_data(get_coeff_data(lf.field))
end

Base.getindex(lf::LockedField, key) = getindex(lf.field, key)
Base.size(lf::LockedField) = size(lf.layout == :g ? get_grid_data(lf.field) : get_coeff_data(lf.field))

# Property forwarding
function Base.getproperty(lf::LockedField, s::Symbol)
    if s in (:field, :layout, :scales)
        return getfield(lf, s)
    elseif s == :name
        return lf.field.name
    elseif s == :dist
        return lf.field.dist
    elseif s == :bases
        return lf.field.bases
    elseif s == :dtype
        return lf.field.dtype
    elseif s == :domain
        return lf.field.domain
    elseif s == :data_g
        return get_grid_data(lf.field)
    elseif s == :data_c
        return get_coeff_data(lf.field)
    elseif s == :current_layout
        return lf.layout  # Return locked layout, not field's current
    else
        return getfield(lf, s)
    end
end

# ---------------------------------------------------------------------------
# Vector field component buffers (structure-of-arrays helpers)
# ---------------------------------------------------------------------------

"""
    stack_components(vf::VectorField; layout::Symbol=:g, arch::AbstractArchitecture=vf.dist.architecture, force::Bool=false)

Build (or reuse) a contiguous buffer containing all vector components stacked along the
first dimension. This provides an easy structure-of-arrays view that is convenient for
GPU kernels expecting component-major memory layout. For PencilArray storage, the buffer
contains the local slab for each component and can be created on CPU or GPU (data is
copied from the host to the requested architecture).
"""
function stack_components(vf::VectorField; layout::Symbol=:g,
                           arch::AbstractArchitecture=vf.dist.architecture,
                           force::Bool=false)
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for stack_components"))
    isempty(vf.components) && throw(ArgumentError("VectorField has no components"))

    using_pencils = is_pencil_storage(vf)
    for component in vf.components
        ensure_layout!(component, layout)
        if !using_pencils
            synchronize_field_architecture!(component; arch=arch,
                                            move_grid = layout == :g,
                                            move_coefficients = layout == :c)
        end
    end

    sample = layout == :g ? get_grid_data(vf.components[1]) : get_coeff_data(vf.components[1])
    sample isa AbstractArray || throw(ArgumentError("stack_components requires array-backed components, got $(typeof(sample))"))
    local_sample = using_pencils ? get_local_data(sample) : sample
    local_sample isa AbstractArray || throw(ArgumentError("Unable to obtain local data for stacking"))

    buffer_shape = (length(vf.components), size(local_sample)...)
    buffer_arch = arch
    needs_new = force || vf.component_buffer === nothing ||
                size(vf.component_buffer) != buffer_shape ||
                architecture(vf.component_buffer) != buffer_arch

    if needs_new
        vf.component_buffer = zeros(buffer_arch, vf.dtype, buffer_shape...)
    end

    for (i, component) in enumerate(vf.components)
        src = layout == :g ? get_grid_data(component) : get_coeff_data(component)
        src_local = using_pencils ? get_local_data(src) : src
        slice_view = selectdim(vf.component_buffer, 1, i)
        copyto!(slice_view, src_local)
    end

    vf.buffer_layout = layout
    vf.buffer_architecture = buffer_arch
    return vf.component_buffer
end

"""
    unstack_components!(vf::VectorField, buffer; layout::Union{Symbol,Nothing}=vf.buffer_layout)

Scatter a stacked buffer back into the vector field components.
"""
function unstack_components!(vf::VectorField, buffer::AbstractArray; layout::Union{Symbol,Nothing}=vf.buffer_layout)
    layout === nothing && throw(ArgumentError("Cannot unstack components without a known layout"))
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for unstack_components!"))
    size(buffer, 1) == length(vf.components) || throw(ArgumentError("Component count mismatch: expected $(length(vf.components)), got $(size(buffer, 1))"))

    using_pencils = is_pencil_storage(vf)
    buffer_arch = architecture(buffer)

    for (i, component) in enumerate(vf.components)
        ensure_layout!(component, layout)
        slice_view = selectdim(buffer, 1, i)
        if using_pencils
            dest = layout == :g ? get_local_data(get_grid_data(component)) : get_local_data(get_coeff_data(component))
            if buffer_arch != CPU()
                copyto!(dest, on_architecture(CPU(), slice_view))
            else
                dest .= slice_view
            end
        else
            dest = layout == :g ? get_grid_data(component) : get_coeff_data(component)
            dest .= slice_view
        end
    end

    if !using_pencils
        for component in vf.components
            synchronize_field_architecture!(component; arch=buffer_arch,
                                            move_grid = layout == :g,
                                            move_coefficients = layout == :c)
        end
    end

    vf.component_buffer = buffer
    vf.buffer_layout = layout
    vf.buffer_architecture = buffer_arch
    return vf
end

"""
    stack_tensor_components(tf::TensorField; layout::Symbol=:g, arch::AbstractArchitecture=tf.dist.architecture, force::Bool=false)

Stack tensor components (matrix of `ScalarField`s) into a structure-of-arrays buffer of shape
`(dim, dim, ...)` where additional dimensions correspond to the underlying scalar data. For
PencilArray storage, the buffer contains only the local slab and can be created on CPU or GPU
(data is copied from the host when needed).
"""
function stack_tensor_components(tf::TensorField; layout::Symbol=:g,
                                  arch::AbstractArchitecture=tf.dist.architecture,
                                  force::Bool=false)
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for stack_tensor_components"))

    dim = tf.coordsys.dim
    dim == size(tf.components, 1) == size(tf.components, 2) || throw(ArgumentError("TensorField component matrix mismatch"))

    using_pencils = is_pencil_storage(tf)
    for i in 1:dim, j in 1:dim
        component = tf.components[i, j]
        ensure_layout!(component, layout)
        if !using_pencils
            synchronize_field_architecture!(component; arch=arch,
                                            move_grid = layout == :g,
                                            move_coefficients = layout == :c)
        end
    end

    sample = layout == :g ? get_grid_data(tf.components[1,1]) : get_coeff_data(tf.components[1,1])
    sample isa AbstractArray || throw(ArgumentError("Tensor components must be array-backed"))
    local_sample = using_pencils ? get_local_data(sample) : sample
    local_sample isa AbstractArray || throw(ArgumentError("Unable to obtain local tensor data for stacking"))
    buffer_shape = (dim, dim, size(local_sample)...)

    buffer_arch = arch
    needs_new = force || tf.component_buffer === nothing ||
                size(tf.component_buffer) != buffer_shape ||
                architecture(tf.component_buffer) != buffer_arch

    if needs_new
        tf.component_buffer = zeros(buffer_arch, tf.dtype, buffer_shape...)
    end

    for i in 1:dim, j in 1:dim
        src = layout == :g ? get_grid_data(tf.components[i,j]) : get_coeff_data(tf.components[i,j])
        src_local = using_pencils ? get_local_data(src) : src
        slice = selectdim(selectdim(tf.component_buffer, 1, i), 2, j)
        copyto!(slice, src_local)
    end

    tf.buffer_layout = layout
    tf.buffer_architecture = buffer_arch
    return tf.component_buffer
end

"""
    unstack_tensor_components!(tf::TensorField, buffer; layout::Union{Symbol,Nothing}=tf.buffer_layout)

Scatter stacked tensor buffer back into component fields.
"""
function unstack_tensor_components!(tf::TensorField, buffer::AbstractArray; layout::Union{Symbol,Nothing}=tf.buffer_layout)
    layout === nothing && throw(ArgumentError("Cannot unstack tensor components without layout information"))
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for unstack_tensor_components!"))

    dim = tf.coordsys.dim
    size(buffer, 1) == dim && size(buffer, 2) == dim || throw(ArgumentError("Tensor buffer must have leading dimensions ($dim, $dim)"))

    using_pencils = is_pencil_storage(tf)
    buffer_arch = architecture(buffer)

    for i in 1:dim, j in 1:dim
        component = tf.components[i, j]
        ensure_layout!(component, layout)
        slice = selectdim(selectdim(buffer, 1, i), 2, j)
        if using_pencils
            dest = layout == :g ? get_local_data(get_grid_data(component)) : get_local_data(get_coeff_data(component))
            if buffer_arch != CPU()
                copyto!(dest, on_architecture(CPU(), slice))
            else
                dest .= slice
            end
        else
            dest = layout == :g ? get_grid_data(component) : get_coeff_data(component)
            dest .= slice
        end
    end

    if !using_pencils
        for i in 1:dim, j in 1:dim
            synchronize_field_architecture!(tf.components[i,j]; arch=buffer_arch,
                                            move_grid = layout == :g,
                                            move_coefficients = layout == :c)
        end
    end

    tf.component_buffer = buffer
    tf.buffer_layout = layout
    tf.buffer_architecture = buffer_arch
    return tf
end

"""
    change_scales!(lf::LockedField, new_scales)

Attempt to change scales on a locked field.
Only succeeds if new_scales matches locked scales or locked scales is nothing.
"""
function change_scales!(lf::LockedField, new_scales)
    if lf.scales !== nothing
        normalized_scales = remedy_scales(lf.field.dist, new_scales)
        locked_scales = remedy_scales(lf.field.dist, lf.scales)
        if normalized_scales != locked_scales
            throw(ArgumentError("Cannot change scales on LockedField from $(lf.scales) to $new_scales"))
        end
    end
    change_scales!(lf.field, new_scales)
    return lf
end

"""
    ensure_layout!(lf::LockedField, layout::Symbol)

Attempt to change layout on a locked field.
Only succeeds if layout matches the locked layout.
"""
function ensure_layout!(lf::LockedField, layout::Symbol)
    if layout != lf.layout
        throw(ArgumentError("Cannot change layout on LockedField from $(lf.layout) to $layout"))
    end
    ensure_layout!(lf.field, layout)
    return lf
end

"""
    unlock(lf::LockedField)

Get the underlying unlocked field.
"""
unlock(lf::LockedField) = lf.field

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

"""
    validate_decomposition_convention(dist::Distributor, expected_convention::Symbol)

Validate that the distributor's decomposition convention matches the expected one.
Throws an error if there's a mismatch in MPI mode.

Arguments:
- dist: Distributor to validate
- expected_convention: :pencil_arrays (LAST dims) or :transposable_field (FIRST dims)

This function helps catch convention mismatches early, preventing silent data corruption.
"""
function validate_decomposition_convention(dist::Distributor, expected_convention::Symbol)
    if dist.size == 1
        return  # Serial mode - convention doesn't matter
    end

    actual_convention = dist.use_pencil_arrays ? :pencil_arrays : :transposable_field

    if expected_convention == :pencil_arrays && !dist.use_pencil_arrays
        error("Convention mismatch: Expected PencilArrays convention (decompose LAST dims) " *
              "but Distributor has use_pencil_arrays=false (TransposableField/FIRST dims). " *
              "This would cause data layout corruption in MPI mode.")
    elseif expected_convention == :transposable_field && dist.use_pencil_arrays
        error("Convention mismatch: Expected TransposableField convention (decompose FIRST dims) " *
              "but Distributor has use_pencil_arrays=true (PencilArrays/LAST dims). " *
              "This would cause data layout corruption in MPI mode.")
    end
end

"""
    Get the coordinate of this process in the specified mesh dimension.

    For a mesh (P₁, P₂, ..., Pₖ), the process with rank r has coordinates:
    (r % P₁, (r ÷ P₁) % P₂, ..., (r ÷ (P₁×P₂×...×Pₖ₋₁)) % Pₖ)
    """
function get_process_coordinate(dist::Distributor, dim::Int)
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return 0
    end

    mesh = dist.mesh
    rank = dist.rank

    # Compute coordinate using column-major ordering (Fortran-style)
    # rank = coord[1] + mesh[1]*(coord[2] + mesh[2]*(coord[3] + ...))
    # So: coord[i] = (rank ÷ prod(mesh[1:i-1])) % mesh[i]
    stride = 1
    for i in 1:(dim-1)
        stride *= mesh[i]
    end

    coord = div(rank, stride) % mesh[dim]
    return coord
end

"""
    Get the local range [start, end] for this process in a given global axis.

    Arguments:
    - dist: Distributor with MPI decomposition info
    - global_size: Size of the global array in this axis
    - axis: Global axis index (1-based)

    Returns:
    - (start_idx, end_idx) tuple with 1-based indices

    Note: Respects dist.use_pencil_arrays:
    - PencilArrays convention: decompose LAST ndims_mesh dimensions
    - TransposableField convention: decompose FIRST ndims_mesh dimensions
    """
function get_local_range(dist::Distributor, global_size::Int, axis::Int)
    if dist.size == 1 || dist.mesh === nothing || axis < 1 || axis > dist.dim
        return (1, global_size)
    end

    mesh_dim = length(dist.mesh)

    # Determine which mesh dimension (if any) corresponds to this axis
    mesh_axis = nothing
    if dist.use_pencil_arrays
        # PencilArrays convention: decompose LAST mesh_dim dimensions
        # Axis dist.dim is mesh[mesh_dim], axis dist.dim-1 is mesh[mesh_dim-1], etc.
        decomp_start = max(1, dist.dim - mesh_dim + 1)
        if axis >= decomp_start
            mesh_axis = axis - decomp_start + 1
        end
    else
        # TransposableField convention: decompose FIRST mesh_dim dimensions
        # Axis 1 is mesh[1], axis 2 is mesh[2], etc.
        if axis <= mesh_dim
            mesh_axis = axis
        end
    end

    # If axis is not decomposed, return full range
    if mesh_axis === nothing || mesh_axis < 1 || mesh_axis > mesh_dim
        return (1, global_size)
    end

    n_procs = dist.mesh[mesh_axis]
    proc_coord = get_process_coordinate(dist, mesh_axis)

    base_size = div(global_size, n_procs)
    remainder = global_size % n_procs

    if proc_coord < remainder
        local_size = base_size + 1
        start_idx = proc_coord * (base_size + 1) + 1
    else
        local_size = base_size
        start_idx = remainder * (base_size + 1) + (proc_coord - remainder) * base_size + 1
    end

    end_idx = start_idx + local_size - 1
    return (start_idx, end_idx)
end

"""
    Convert a global index to a local index for this process.

    Returns nothing if the global index is not owned by this process.
    """
function global_to_local_index(dist::Distributor, global_idx::Int, axis::Int)
    start_idx, end_idx = get_local_range(dist, get_global_size(dist, axis), axis)

    if global_idx >= start_idx && global_idx <= end_idx
        return global_idx - start_idx + 1
    else
        return nothing
    end
end

"""
    Convert a local index to a global index.
    """
function local_to_global_index(dist::Distributor, local_idx::Int, global_size::Int, axis::Int)
    start_idx, _ = get_local_range(dist, global_size, axis)
    return start_idx + local_idx - 1
end

"""
    get_global_size(dist::Distributor, dim::Int)

Get the global size in a dimension. This method requires domain/basis information
to determine actual sizes. Without that context, it returns a conservative default.

For accurate global sizes, use one of the following methods instead:
- `get_global_size(dist, basis, dim)` - for a specific basis
- `get_global_size(dist, domain, dim)` - for a specific domain
- `get_global_grid_shape(dist, domain; scales=...)` - for full grid shape

# Arguments
- `dist`: The Distributor
- `dim`: Dimension index (1-based)

# Returns
- The global size in the specified dimension, or a default value if unknown
"""
function get_global_size(dist::Distributor, dim::Int)
    # Without domain/basis context, we cannot determine the actual global size.
    # Check if distributor has cached layout information that might help.
    if !isempty(dist.layouts)
        # Try to get size from cached layouts
        for (key, layout) in dist.layouts
            if hasfield(typeof(layout), :global_shape) && layout.global_shape !== nothing
                if dim <= length(layout.global_shape)
                    return layout.global_shape[dim]
                end
            end
        end
    end

    # Fallback: Check if pencil_cache has any entries with shape info
    if !isempty(dist.pencil_cache)
        for (shape_key, pencil) in dist.pencil_cache
            if isa(shape_key, Tuple) && dim <= length(shape_key)
                return shape_key[dim]
            end
        end
    end

    @warn "get_global_size called without domain context; returning default. " *
          "Use get_global_size(dist, basis, dim) or get_global_size(dist, domain, dim) for accurate sizes." maxlog=1
    return 64
end

"""
    get_global_size(dist::Distributor, basis::Basis, dim::Int=1)

Get the global size for a specific basis dimension.

# Arguments
- `dist`: The Distributor (unused but kept for API consistency)
- `basis`: The Basis to query
- `dim`: Dimension within the basis (default 1, as most bases are 1D)

# Returns
- The global size (number of grid/coefficient points) for this basis
"""
function get_global_size(dist::Distributor, basis::Basis, dim::Int=1)
    if dim != 1
        @warn "Most bases are 1D; dim=$dim requested but using basis size"
    end
    return basis.meta.size
end

"""
    get_global_size(dist::Distributor, domain::Domain, dim::Int)

Get the global size in a specific dimension of the domain.

# Arguments
- `dist`: The Distributor (unused but kept for API consistency)
- `domain`: The Domain containing the bases
- `dim`: Dimension index (1-based)

# Returns
- The global size in the specified dimension
"""
function get_global_size(dist::Distributor, domain::Domain, dim::Int)
    if dim < 1 || dim > length(domain.bases)
        throw(BoundsError("Dimension $dim out of range for domain with $(length(domain.bases)) dimensions"))
    end
    return domain.bases[dim].meta.size
end

"""
    get_global_sizes(dist::Distributor, domain::Domain)

Get all global sizes for a domain as a tuple.

# Arguments
- `dist`: The Distributor
- `domain`: The Domain containing the bases

# Returns
- Tuple of global sizes for each dimension
"""
function get_global_sizes(dist::Distributor, domain::Domain)
    return tuple([basis.meta.size for basis in domain.bases]...)
end

"""
    Set new transform scales without data transformation.

    Scales control the grid resolution relative to the coefficient resolution.
    A scale of 1.0 means grid size equals coefficient size.
    A scale of 1.5 (3/2 rule) is used for dealiasing in nonlinear computations.

    This function:
    1. Updates the field's scale parameters
    2. Reallocates data arrays if the new scaled size differs
    3. Does NOT preserve or transform existing data (use set_scales! for that)

    Arguments:
    - field: ScalarField to modify
    - scales: New scale values (scalar applied to all dims, or per-dimension)

    Returns:
    - The modified field
    """
function preset_scales!(field::ScalarField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    new_scales = remedy_scales(field.dist, scales)
    old_scales = field.scales

    # Return if scales are unchanged
    if new_scales == old_scales
        return field
    end

    # Compute old and new grid sizes
    old_grid_shape = get_scaled_shape(field, old_scales)
    new_grid_shape = get_scaled_shape(field, new_scales)

    # Update scales
    field.scales = new_scales

    # Reallocate grid data if shape changed
    if old_grid_shape != new_grid_shape && field.domain !== nothing
        @debug "Reallocating field data for new scales" old_grid_shape new_grid_shape

        # Reallocate grid-space data with new size
        if field.dist.use_pencil_arrays
            # For MPI: create new pencil with full decomposition for storage
            # NOTE: Rescaled data has a different shape than the PencilFFT plan expects.
            # This pencil is NOT compatible with the plan's mul!/ldiv! - it's meant for
            # intermediate computations (like dealiasing) that don't directly use PencilFFT.
            # If you need to transform rescaled data, you'd need a separate plan.
            set_grid_data!(field, create_pencil(field.dist, new_grid_shape, nothing, dtype=field.dtype))
        else
            # For serial: create new array
            local_shape = get_local_array_size(field.dist, new_grid_shape)
            arch = field.buffers.architecture
            set_grid_data!(field, zeros(arch, field.dtype, local_shape...))
        end

        # Coefficient data size typically doesn't change with scales
        # (scales affect grid resolution, not spectral resolution)
    end

    @debug "Updated field scales" old_scales=old_scales new_scales=new_scales
    return field
end

"""
    Compute the grid shape with given scales applied.

    For each basis, the scaled size is: ceil(Int, basis_size * scale)
    """
function get_scaled_shape(field::ScalarField, scales::Union{Tuple, Nothing})
    if field.domain === nothing || isempty(field.bases)
        return ()
    end

    # Handle nothing scales (use 1.0 for all dimensions)
    if scales === nothing
        scales = tuple(ones(Float64, length(field.bases))...)
    end

    scaled_shape = Int[]
    for (i, basis) in enumerate(field.bases)
        base_size = basis.meta.size
        scale = i <= length(scales) ? scales[i] : 1.0
        scaled_size = ceil(Int, base_size * scale)
        push!(scaled_shape, scaled_size)
    end

    return tuple(scaled_shape...)
end

"""Get the current scaled grid shape."""
function get_scaled_shape(field::ScalarField)
    if field.scales === nothing
        scales = tuple(ones(Float64, length(field.bases))...)
    else
        scales = field.scales
    end
    return get_scaled_shape(field, scales)
end

"""
    Get the coefficient (unscaled) shape, accounting for MPI execution context.

    IMPORTANT: In MPI mode with PencilFFTs, only the FIRST RealFourier axis uses RFFT
    (size N/2+1). Subsequent RealFourier axes use FFT (full size N) because PencilFFTs
    can only apply RFFT to the first transform dimension.

    In serial mode or MPI without PencilFFTs, all RealFourier axes use RFFT (N/2+1).
    """
function get_coefficient_shape(field::ScalarField)
    if field.domain === nothing || isempty(field.bases)
        return ()
    end

    # Use context-aware coefficient shape computation
    return get_coefficient_shape_for_context(field.domain, field.dist)
end

"""
    Ensure field has the specified scales, reallocating if necessary.
    Similar to require_scales pattern.
    """
function require_scales!(field::ScalarField, scales::Union{Real, Tuple, Nothing})
    new_scales = remedy_scales(field.dist, scales)

    if field.scales != new_scales
        preset_scales!(field, new_scales)
    end

    return field
end

"""
    Get the standard 3/2 dealiasing scales for this field.
    Used for computing nonlinear terms without aliasing errors.
    """
function dealias_scales(field::ScalarField)
    ndims = length(field.bases)
    return tuple(fill(1.5, ndims)...)
end

"""Apply 3/2 dealiasing scales to the field."""
function apply_dealiasing_scales!(field::ScalarField)
    return preset_scales!(field, dealias_scales(field))
end

"""
    Change data to specified scales, properly handling data transformation.
    Following implementation in field:631-649

    When changing scales:
    - If in grid space: interpolate/resample data to new grid size
    - If in coefficient space: pad/truncate spectral coefficients

    For upscaling (scale increases):
    - Grid space: interpolate to finer grid
    - Coefficient space: zero-pad high frequencies

    For downscaling (scale decreases):
    - Grid space: subsample or average to coarser grid
    - Coefficient space: truncate high frequencies

    Arguments:
    - field: ScalarField to modify
    - scales: New scale values

    Returns:
    - The modified field with transformed data
    """
function set_scales!(field::ScalarField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    # Remedy scales
    new_scales = remedy_scales(field.dist, scales)
    old_scales = field.scales

    # Quit if new scales aren't new
    if new_scales == old_scales
        return field
    end

    # Get old and new shapes
    old_grid_shape = get_scaled_shape(field, old_scales)
    new_grid_shape = get_scaled_shape(field, new_scales)
    coeff_shape = get_coefficient_shape(field)

    # Determine current layout
    is_grid_space = (field.current_layout == :g)

    if is_grid_space && get_grid_data(field) !== nothing
        # Transform grid-space data to new resolution
        old_data = get_local_data(get_grid_data(field))

        if old_data !== nothing && !isempty(old_data)
            # Store coefficient data if it exists
            old_coeff_data = get_coeff_data(field) !== nothing ? copy(get_local_data(get_coeff_data(field))) : nothing

            # Reallocate with new scales
            preset_scales!(field, new_scales)

            # Resample grid data to new resolution
            new_data = get_local_data(get_grid_data(field))
            if new_data !== nothing
                resample_grid_data!(new_data, old_data, old_grid_shape, new_grid_shape)
            end

            # Restore coefficient data (unchanged by grid scaling)
            if old_coeff_data !== nothing && get_coeff_data(field) !== nothing
                coeff_data = get_local_data(get_coeff_data(field))
                if coeff_data !== nothing && size(coeff_data) == size(old_coeff_data)
                    copyto!(coeff_data, old_coeff_data)
                end
            end
        else
            preset_scales!(field, new_scales)
        end

    elseif !is_grid_space && get_coeff_data(field) !== nothing
        # In coefficient space: scales don't affect coefficient storage
        # but we still need to update the field's scale parameter
        # The grid data will be recomputed on next backward transform
        preset_scales!(field, new_scales)

    else
        # No data to transform
        preset_scales!(field, new_scales)
    end

    @debug "Changed field scales with data transformation" old_scales=old_scales new_scales=new_scales
    return field
end

"""
    gpu_resample_grid_data!(new_data, old_data, old_shape, new_shape)

GPU-native spectral resampling using cuFFT.
Returns true if GPU resample was applied, false otherwise.
Override provided by TarangCUDAExt when CUDA is loaded.
"""
function gpu_resample_grid_data!(new_data::AbstractArray, old_data::AbstractArray,
                                 old_shape::Tuple, new_shape::Tuple)
    return false  # No GPU support without CUDA extension
end

function resample_grid_data!(new_data::AbstractArray, old_data::AbstractArray,
                             old_shape::Tuple, new_shape::Tuple)
    """
    Resample grid data from old resolution to new resolution.

    Uses spectral interpolation for accurate resampling:
    1. FFT old data to spectral space
    2. Pad/truncate spectral coefficients
    3. IFFT back to new grid

    For simple cases, uses linear interpolation as fallback.
    """
    if is_gpu_array(new_data) || is_gpu_array(old_data)
        if gpu_resample_grid_data!(new_data, old_data, old_shape, new_shape)
            return
        end
        # Fallback: CPU round-trip if GPU-native resample unavailable
        old_cpu = on_architecture(CPU(), old_data)
        new_cpu = similar(old_cpu, eltype(old_cpu), size(new_data)...)
        resample_grid_data!(new_cpu, old_cpu, old_shape, new_shape)
        copyto!(new_data, on_architecture(architecture(new_data), new_cpu))
        return
    end

    old_size = size(old_data)
    new_size = size(new_data)

    # Handle dimension mismatch
    if length(old_size) != length(new_size)
        @warn "Dimension mismatch in resample_grid_data!"
        fill!(new_data, 0)
        return
    end

    # If sizes match, just copy
    if old_size == new_size
        copyto!(new_data, old_data)
        return
    end

    ndims_data = length(old_size)

    if ndims_data == 1
        resample_1d!(new_data, old_data)
    elseif ndims_data == 2
        resample_2d!(new_data, old_data)
    elseif ndims_data == 3
        resample_3d!(new_data, old_data)
    else
        # Fallback: simple nearest-neighbor for higher dimensions
        resample_nearest!(new_data, old_data)
    end
end

"""Resample 1D data using spectral interpolation."""
function resample_1d!(new_data::AbstractVector, old_data::AbstractVector)
    n_old = length(old_data)
    n_new = length(new_data)

    if n_old == 1 || n_new == 1
        fill!(new_data, old_data[1])
        return
    end

    if n_old == n_new
        copyto!(new_data, old_data)
        return
    end

    # Use FFT-based resampling for spectral accuracy
    try
        # Forward FFT
        old_fft = FFTW.fft(old_data)

        # Create padded/truncated spectrum
        new_fft = zeros(eltype(old_fft), n_new)

        # FFT layout for N points (even N):
        # Index 1: DC (f=0)
        # Indices 2 to N/2: positive frequencies (f=1 to f=N/2-1)
        # Index N/2+1: Nyquist (f=N/2)
        # Indices N/2+2 to N: negative frequencies (f=-(N/2-1) to f=-1)

        if n_new > n_old
            # Upsampling: zero-pad high frequencies
            n_pos_old = div(n_old, 2)
            if iseven(n_old)
                # Even N: copy positive freqs excluding Nyquist (index n_pos_old+1).
                # Zeroing the Nyquist prevents aliased half-period oscillation.
                new_fft[1:n_pos_old] = old_fft[1:n_pos_old]
            else
                # Odd N: no Nyquist bin exists. Copy all positive frequencies
                # including the highest (index n_pos_old+1).
                new_fft[1:n_pos_old+1] = old_fft[1:n_pos_old+1]
            end
            # Copy negative frequencies
            n_neg = n_old - n_pos_old - 1  # Number of negative frequencies
            if n_neg > 0
                new_fft[end-n_neg+1:end] = old_fft[end-n_neg+1:end]
            end
            # Scale for energy conservation
            new_fft .*= n_new / n_old
        else
            # Downsampling: truncate high frequencies
            # Copy positive frequencies including new Nyquist
            n_pos_new = div(n_new, 2)
            new_fft[1:n_pos_new+1] = old_fft[1:n_pos_new+1]
            # Copy negative frequencies (excluding Nyquist)
            n_neg_new = n_new - n_pos_new - 1  # Number of negative frequencies in new array
            if n_neg_new > 0
                new_fft[n_pos_new+2:n_new] = old_fft[n_old-n_neg_new+1:n_old]
            end
            # Scale for energy conservation
            new_fft .*= n_new / n_old
        end

        # Inverse FFT
        result = FFTW.ifft(new_fft)
        if eltype(new_data) <: Real
            copyto!(new_data, real(result))
        else
            copyto!(new_data, result)
        end

    catch e
        @warn "FFT resampling failed, using linear interpolation: $e"
        resample_linear_1d!(new_data, old_data)
    end
end

"""Fallback linear interpolation for 1D resampling."""
function resample_linear_1d!(new_data::AbstractVector, old_data::AbstractVector)
    n_old = length(old_data)
    n_new = length(new_data)

    if n_old == 1 || n_new == 1
        fill!(new_data, old_data[1])
        return
    end

    for i in 1:n_new
        # Map new index to old index (0-based for interpolation)
        t = (i - 1) / (n_new - 1) * (n_old - 1)
        i_old = floor(Int, t) + 1
        frac = t - (i_old - 1)

        if i_old >= n_old
            new_data[i] = old_data[n_old]
        elseif i_old < 1
            new_data[i] = old_data[1]
        else
            new_data[i] = (1 - frac) * old_data[i_old] + frac * old_data[i_old + 1]
        end
    end
end

"""
    Resample 2D data using separable spectral interpolation.

    Uses 1D spectral resampling along each dimension sequentially,
    which is equivalent to tensor-product interpolation. This approach
    is more robust than direct 2D FFT padding and handles arbitrary
    grid size changes correctly.
    """
function resample_2d!(new_data::AbstractMatrix, old_data::AbstractMatrix)
    n_old = size(old_data)
    n_new = size(new_data)

    if n_old == n_new
        copyto!(new_data, old_data)
        return
    end

    try
        # Separable resampling: apply 1D spectral interpolation along each axis
        # This is mathematically equivalent to 2D spectral interpolation for
        # tensor-product grids and is numerically more stable

        # Temporary array for intermediate result
        temp = zeros(eltype(new_data), n_new[1], n_old[2])

        # Resample along first dimension (rows)
        for j in 1:n_old[2]
            resample_1d!(view(temp, :, j), view(old_data, :, j))
        end

        # Resample along second dimension (columns)
        for i in 1:n_new[1]
            resample_1d!(view(new_data, i, :), view(temp, i, :))
        end

    catch e
        @warn "2D spectral resampling failed: $e"
        resample_nearest!(new_data, old_data)
    end
end

"""Resample 3D data using separable 1D spectral interpolation."""
function resample_3d!(new_data::AbstractArray{T,3}, old_data::AbstractArray{T,3}) where T
    n_old = size(old_data)
    n_new = size(new_data)

    if n_old == n_new
        copyto!(new_data, old_data)
        return
    end

    try
        # Separable resampling: resample along each dimension sequentially
        temp1 = zeros(T, n_new[1], n_old[2], n_old[3])
        temp2 = zeros(T, n_new[1], n_new[2], n_old[3])

        # Resample along first dimension
        for k in 1:n_old[3], j in 1:n_old[2]
            resample_1d!(view(temp1, :, j, k), view(old_data, :, j, k))
        end

        # Resample along second dimension
        for k in 1:n_old[3], i in 1:n_new[1]
            resample_1d!(view(temp2, i, :, k), view(temp1, i, :, k))
        end

        # Resample along third dimension
        for j in 1:n_new[2], i in 1:n_new[1]
            resample_1d!(view(new_data, i, j, :), view(temp2, i, j, :))
        end

    catch e
        @warn "3D spectral resampling failed: $e"
        resample_nearest!(new_data, old_data)
    end
end

"""Nearest-neighbor resampling for arbitrary dimensions."""
function resample_nearest!(new_data::AbstractArray, old_data::AbstractArray)
    old_size = size(old_data)
    new_size = size(new_data)
    ndims_data = length(old_size)

    for I in CartesianIndices(new_data)
        # Map new indices to old indices
        old_indices = ntuple(ndims_data) do d
            # Scale index from new to old grid
            new_idx = I[d]
            if new_size[d] <= 1 || old_size[d] <= 1
                return 1
            end
            old_idx = round(Int, (new_idx - 1) / (new_size[d] - 1) * (old_size[d] - 1)) + 1
            clamp(old_idx, 1, old_size[d])
        end

        new_data[I] = old_data[CartesianIndex(old_indices)]
    end
end

# Alias for compatibility  
change_scales!(field::ScalarField, scales) = set_scales!(field, scales)

# VectorField scaling methods
"""Set scales for all vector field components."""
function preset_scales!(field::VectorField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    for component in field.components
        preset_scales!(component, scales)
    end
    return field
end

"""Change scales for all vector field components."""
function set_scales!(field::VectorField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    for component in field.components
        set_scales!(component, scales)
    end
    return field
end

change_scales!(field::VectorField, scales) = set_scales!(field, scales)

# Helper functions for safe data access from Pencil or Array
"""
    get_local_data(field_data)

Get local data array from either a PencilArray or regular Array.
For PencilArray: returns the local buffer (this MPI rank)
For Array: returns the array itself
"""
function get_local_data(field_data::PencilArrays.PencilArray)
    return parent(field_data)  # Get underlying local array
end

function get_local_data(field_data::AbstractArray)
    return field_data
end

function get_local_data(field_data::Nothing)
    return nothing
end

