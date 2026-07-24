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
    Forward transform field using PencilFFTs for parallel transforms.

    CORRECT PencilFFTs usage pattern:
    1. Input/output are PencilArray objects (NOT plain arrays)
    2. PencilFFT automatically handles:
       - Required transpose operations between decompositions
       - Multi-dimensional FFT across decomposed axes
       - Scaling and normalization
    3. For 2D: Enables BOTH vertical and horizontal parallelization

    Following distributor pattern in distributor:636-649
    """
function forward_transform_axis!(field::ScalarField)
    if field.domain === nothing || field.bases === ()
        return
    end

    # Use PencilFFTs-based transforms from the distributor's transform plans
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # CORRECT: Apply PencilFFT to PencilArray objects
        # PencilFFT handles transposes internally

        if field.dist.use_pencil_arrays && isa(get_grid_data(field), PencilArrays.PencilArray)
            # Apply forward transform: grid space (physical) → coefficient space (spectral)
            # Note: mul! is the in-place version
            # Result goes into data_c pencil
            if get_coeff_data(field) === nothing || !isa(get_coeff_data(field), PencilArrays.PencilArray)
                # CRITICAL: Use PencilFFTs.allocate_output for compatible coeff-space array
                set_coeff_data!(field, PencilFFTs.allocate_output(pencil_plan))
            end

            # Apply PencilFFT: transforms AND transposes as needed
            mul!(get_coeff_data(field), pencil_plan, get_grid_data(field))

            @debug "Applied PencilFFT forward transform" typeof(pencil_plan) size(get_grid_data(field))
            field.current_layout = :c
            return  # Successfully applied transform
        else
            # CRITICAL: PencilFFT found but cannot be applied - this indicates a configuration error
            if field.dist.size > 1
                error("PencilFFT transform found but field data is not a PencilArray. " *
                      "In MPI mode, field data must be stored as PencilArrays for correct parallel transforms. " *
                      "use_pencil_arrays=$(field.dist.use_pencil_arrays), " *
                      "grid_data type=$(typeof(get_grid_data(field)))")
            else
                # Serial mode: fall through to standard transform
                @debug "PencilFFT found but using serial transform (serial execution)"
            end
        end
    end

    # No PencilFFT or serial mode: use standard transform
    forward_transform!(field)
end

"""
    Backward transform field using PencilFFTs for parallel transforms.

    CORRECT PencilFFTs usage pattern:
    1. Inverse FFT: coefficient space (spectral) → grid space (physical)
    2. Uses ldiv! (\\) for backward transform
    3. Maintains PencilArray objects throughout

    Following distributor pattern in distributor:621-634
    """
function backward_transform_axis!(field::ScalarField)
    if field.domain === nothing || field.bases === ()
        return
    end

    # Use PencilFFTs-based transforms from the distributor's transform plans
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # CORRECT: Apply inverse PencilFFT to PencilArray objects

        if field.dist.use_pencil_arrays && isa(get_coeff_data(field), PencilArrays.PencilArray)
            # Apply backward transform: coefficient space → grid space
            if get_grid_data(field) === nothing || !isa(get_grid_data(field), PencilArrays.PencilArray)
                # CRITICAL: Use PencilFFTs.allocate_input for compatible grid-space array
                # allocate_input creates the input array for forward transform,
                # which is the output of backward transform
                set_grid_data!(field, PencilFFTs.allocate_input(pencil_plan))
            end

            # Apply inverse PencilFFT: transforms AND transposes as needed
            # ldiv! is in-place inverse (like \ but in-place)
            ldiv!(get_grid_data(field), pencil_plan, get_coeff_data(field))

            @debug "Applied PencilFFT backward transform" typeof(pencil_plan) size(get_coeff_data(field))
            field.current_layout = :g
            return  # Successfully applied transform
        else
            # CRITICAL: PencilFFT found but cannot be applied - this indicates a configuration error
            if field.dist.size > 1
                error("PencilFFT transform found but field data is not a PencilArray. " *
                      "In MPI mode, field data must be stored as PencilArrays for correct parallel transforms. " *
                      "use_pencil_arrays=$(field.dist.use_pencil_arrays), " *
                      "coeff_data type=$(typeof(get_coeff_data(field)))")
            else
                # Serial mode: fall through to standard transform
                @debug "PencilFFT found but using serial transform (serial execution)"
            end
        end
    end

    # No PencilFFT or serial mode: use standard transform
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
