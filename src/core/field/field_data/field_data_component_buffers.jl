"""
    Field component buffers

This file contains structure-of-arrays helpers for stacking and unstacking
vector and tensor field component data.
"""

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
