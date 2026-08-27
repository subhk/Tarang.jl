"""
    Field layout access

This file contains CPU/local-data access helpers, scalar/vector/tensor layout
transitions, and axis-wise transform stepping utilities.
"""

# ============================================================================
# GPU-Aware Data Access (for File I/O)
# ============================================================================

"""
    get_cpu_data(field_data)

Get data as a CPU Array, automatically transferring from GPU if needed.
This is essential for file I/O since the NetCDF library requires CPU arrays.

For GPU arrays (CuArray): Transfers data to CPU
For CPU arrays: Returns as-is
For PencilArray: Gets local data then transfers if on GPU
"""
function get_cpu_data(field_data::AbstractArray)
    # Check if data is on GPU and transfer to CPU
    # The on_architecture function handles the conversion
    return on_architecture(CPU(), field_data)
end

function get_cpu_data(field_data::PencilArrays.PencilArray)
    # Get local data first, then transfer to CPU if needed
    local_data = parent(field_data)
    return on_architecture(CPU(), local_data)
end

function get_cpu_data(field_data::Nothing)
    return nothing
end

"""
    get_cpu_local_data(field::ScalarField, layout::Symbol)

Get field data in the specified layout as a CPU Array.
Automatically handles GPU to CPU transfer for file I/O operations.

# Arguments
- `field`: The ScalarField to extract data from
- `layout`: :g for grid space, :c for coefficient space

# Returns
CPU Array containing the field data
"""
function get_cpu_local_data(field::ScalarField, layout::Symbol)
    ensure_layout!(field, layout)
    if layout == :g
        return get_cpu_data(get_grid_data(field))
    else
        return get_cpu_data(get_coeff_data(field))
    end
end

"""
    get_cpu_local_data(field::VectorField, layout::Symbol)

Get vector field data as CPU Arrays.
"""
function get_cpu_local_data(field::VectorField, layout::Symbol)
    return [get_cpu_local_data(comp, layout) for comp in field.components]
end

"""
    get_cpu_local_data(field::TensorField, layout::Symbol)

Get tensor field data as CPU Arrays.
"""
function get_cpu_local_data(field::TensorField, layout::Symbol)
    return [get_cpu_local_data(field.components[i,j], layout)
            for i in 1:size(field.components, 1), j in 1:size(field.components, 2)]
end

"""
    is_gpu_field(field::ScalarField)

Check if a field's data is on GPU.
"""
function is_gpu_field(field::ScalarField)
    # Check actual array storage rather than architecture metadata
    gd = field.buffers.grid
    if gd isa AbstractArray
        return is_gpu_array(gd)
    end
    cd = field.buffers.coeff
    if cd isa AbstractArray
        return is_gpu_array(cd)
    end
    # Fallback to architecture if no data allocated yet (or Pencil arrays = CPU)
    return is_gpu(field.buffers.architecture)
end

is_gpu_field(field::VectorField) = is_gpu_field(field.components[1])
is_gpu_field(field::TensorField) = is_gpu_field(field.components[1,1])

"""True when either a field's declared distributor or its current storage uses GPU."""
_field_uses_gpu(field::ScalarField) =
    is_gpu(field.dist.architecture) || is_gpu(field_architecture(field)) || is_gpu_field(field)

"""
    set_local_data!(field_data, values)

Set local data in either a PencilArray or regular Array.
"""
function set_local_data!(field_data::PencilArrays.PencilArray, values)
    parent(field_data) .= values
    return field_data
end

function set_local_data!(field_data::AbstractArray, values)
    field_data .= values
    return field_data
end

# Data access and manipulation
"""
    Get data in specified layout.

    Returns local data if using PencilArrays (MPI), otherwise returns full array.
    For user code operating on local data, this is the correct access pattern.
    """
function Base.getindex(field::ScalarField, layout::String)
    if layout == "g"
        ensure_layout!(field, :g)
        return get_local_data(get_grid_data(field))
    elseif layout == "c"
        ensure_layout!(field, :c)
        return get_local_data(get_coeff_data(field))
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

"""
    Set data in specified layout.

    Properly handles both PencilArray data (MPI) and regular arrays.
    """
function Base.setindex!(field::ScalarField, values, layout::String)
    if layout == "g"
        ensure_layout!(field, :g)
        set_local_data!(get_grid_data(field), values)
        field.current_layout = :g
    elseif layout == "c"
        ensure_layout!(field, :c)
        set_local_data!(get_coeff_data(field), values)
        field.current_layout = :c
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

"""Ensure field is in the target layout, transforming if necessary"""
function ensure_layout!(field::ScalarField, target_layout::Symbol)
    # Skip 0D fields (tau variables) which have no spatial data
    if isempty(field.bases)
        return
    end

    # Check if field has any data allocated - if not, try to allocate
    if get_grid_data(field) === nothing && get_coeff_data(field) === nothing
        if field.domain !== nothing
            allocate_data!(field)
        else
            # No domain, no data - nothing to transform
            return
        end
    end

    if field.current_layout == target_layout
        return
    end

    if target_layout == :g && field.current_layout == :c
        # Transform from coefficient to grid space
        backward_transform!(field)
        # Note: backward_transform! sets current_layout = :g when successful
    elseif target_layout == :c && field.current_layout == :g
        # Transform from grid to coefficient space
        forward_transform!(field)
        # Note: forward_transform! sets current_layout = :c when successful
    end
    # Don't set current_layout here - the transform functions handle it
    # Setting it unconditionally would be incorrect if the transform failed or returned early
end

"""Ensure all components of VectorField are in the target layout"""
function ensure_layout!(field::VectorField, target_layout::Symbol)
    for comp in field.components
        ensure_layout!(comp, target_layout)
    end
end

"""Ensure all components of TensorField are in the target layout"""
function ensure_layout!(field::TensorField, target_layout::Symbol)
    for comp in field.components  # Matrix iteration goes element-by-element
        ensure_layout!(comp, target_layout)
    end
end

# Note: forward_transform! and backward_transform! for ScalarField are defined in transforms.jl
# to avoid duplicate method definitions. The transforms.jl versions have more complete
# implementations with optional target_layout parameters.

"""
    grid_data!(field)  -> grid-space data buffer
    coeff_data!(field) -> coefficient-space data buffer

Transform `field` into the required layout if it is not already there, then return
that layout's buffer. The trailing `!` is the transform, not a write to the buffer.

Prefer these over the `ensure_layout!` + `get_*_data` pair they replace:

    ensure_layout!(u, :g)          #  grid_data!(u)
    data = get_grid_data(u)        #

`get_grid_data` and `get_coeff_data` hand back whatever buffer the field is holding
without checking `current_layout`, so reading the wrong one is not an error — it is
stale or untransformed numbers, which is the silent-wrong-value failure this project
keeps rediscovering. Writing the pair correctly is a per-call-site obligation
repeated a few hundred times over `src/`; the layout is a mutable `Symbol` on the
field, so nothing in the type system can help. Folding the two calls into one at
least makes the guarantee part of the name and removes the chance to write the
second line without the first.

This does NOT make the raw accessors unsafe to use, and it does not help the harder
case — code that sets a layout for something to read later, somewhere else. Those
sites are counted by the ratchet in `test/test_layout_discipline_ratchet.jl`.
"""
function grid_data!(field)
    ensure_layout!(field, :g)
    return get_grid_data(field)
end

function coeff_data!(field)
    ensure_layout!(field, :c)
    return get_coeff_data(field)
end

"""
    Require one axis (default: all axes) to be in grid space.
    Following implementation in field:674-681
    """
function require_grid_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    if field.domain === nothing
        return
    end
    
    if axis === nothing
        # Require all axes to be in grid space
        while field.current_layout != :g
            towards_grid_space!(field)
        end
    else
        # For specific axis: ensure field is in grid space
        # Tarang uses a two-state layout model (:c for coefficient, :g for grid)
        # rather than per-axis tracking. For single-axis requirements,
        # we transform the entire field to grid space, which ensures the
        # requested axis is in grid space along with all others.
        towards_grid_space!(field)
    end
end

"""
    Require one axis (default: all axes) to be in coefficient space.
    Following implementation in field:683-690
    """
function require_coeff_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    if field.domain === nothing
        return
    end

    if axis === nothing
        # Require all axes to be in coefficient space
        while field.current_layout != :c
            towards_coeff_space!(field)
        end
    else
        # For specific axis: ensure field is in coefficient space
        # Tarang uses a two-state layout model (:c for coefficient, :g for grid)
        # rather than per-axis tracking. For single-axis requirements,
        # we transform the entire field to coefficient space, which ensures
        # the requested axis is in coefficient space along with all others.
        towards_coeff_space!(field)
    end
end

"""
    Change to next layout towards grid space.
    Following implementation in field:664-667
    """
function towards_grid_space!(field::ScalarField)
    if field.current_layout == :c
        # Transform from coefficient to grid space
        # Note: backward_transform_axis! sets current_layout = :g when successful
        # It may return early without transforming if field.domain is nothing
        backward_transform_axis!(field)
    end
end

"""
    Change to next layout towards coefficient space.
    Following implementation in field:669-672
    """
function towards_coeff_space!(field::ScalarField)
    if field.current_layout == :g
        # Transform from grid to coefficient space
        # Note: forward_transform_axis! sets current_layout = :c when successful
        # It may return early without transforming if field.domain is nothing
        forward_transform_axis!(field)
    end
end

"""
    forward_transform_axis!(field)

Move `field` to coefficient space using its canonical full-field transform.

Tarang tracks only a two-state field layout (`:g` or `:c`), not a separate
layout for each axis.  This function is therefore a compatibility wrapper for
`forward_transform!`; keeping a second PencilFFT execution path here would
bypass the canonical mixed-basis and real-to-complex handling.
"""
function forward_transform_axis!(field::ScalarField)
    if field.domain === nothing || field.bases === ()
        return
    end
    forward_transform!(field)
end

"""
    backward_transform_axis!(field)

Move `field` to grid space using its canonical full-field transform.  See
[`forward_transform_axis!`](@ref) for why this is intentionally a wrapper.
"""
function backward_transform_axis!(field::ScalarField)
    if field.domain === nothing || field.bases === ()
        return
    end
    backward_transform!(field)
end

# VectorField transform methods
"""Require vector field components to be in grid space."""
function require_grid_space!(field::VectorField, axis::Union{Int, Nothing}=nothing)
    for component in field.components
        require_grid_space!(component, axis)
    end
end

"""Require vector field components to be in coefficient space."""
function require_coeff_space!(field::VectorField, axis::Union{Int, Nothing}=nothing)
    for component in field.components
        require_coeff_space!(component, axis)
    end
end

"""Transform vector field from grid to coefficient space."""
function forward_transform!(field::VectorField)
    for component in field.components
        forward_transform!(component)
    end
end

"""Transform vector field from coefficient to grid space."""
function backward_transform!(field::VectorField)
    for component in field.components
        backward_transform!(component)
    end
end

"""Transform tensor field components from grid to coefficient space."""
function forward_transform!(field::TensorField)
    for component in field.components
        forward_transform!(component)
    end
end

"""Transform tensor field components from coefficient to grid space."""
function backward_transform!(field::TensorField)
    for component in field.components
        backward_transform!(component)
    end
end
