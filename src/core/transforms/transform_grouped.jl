"""
    Transform Grouped - Grouped transform operations

This file contains grouped transform operations that process multiple fields
at once for better efficiency (following Dedalus GROUP_TRANSFORMS pattern).
"""

# ============================================================================
# Grouped Transforms (Following Dedalus GROUP_TRANSFORMS)
# ============================================================================

"""
    GroupedTransformConfig

Configuration for grouped transform operations.
"""
mutable struct GroupedTransformConfig
    enabled::Bool                    # Whether to use grouped transforms
    min_fields::Int                  # Minimum number of fields to trigger grouping
    max_batch_size::Int             # Maximum number of fields per batch
    batch_buffer::Union{Nothing, AbstractArray}  # Reusable buffer for batching

    function GroupedTransformConfig()
        new(true, 2, 32, nothing)
    end
end

const GROUPED_TRANSFORM_CONFIG = GroupedTransformConfig()

"""
    set_group_transforms!(enabled::Bool; min_fields::Int=2, max_batch_size::Int=32)

Enable or disable grouped transforms. When enabled, multiple fields are transformed
together in batches for improved efficiency.

Following Dedalus GROUP_TRANSFORMS configuration.
"""
function set_group_transforms!(enabled::Bool; min_fields::Int=2, max_batch_size::Int=32)
    GROUPED_TRANSFORM_CONFIG.enabled = enabled
    GROUPED_TRANSFORM_CONFIG.min_fields = min_fields
    GROUPED_TRANSFORM_CONFIG.max_batch_size = max_batch_size
    return nothing
end

"""
    group_forward_transform!(fields::Vector{<:ScalarField})

Apply forward transforms to multiple fields simultaneously.
Batches fields with compatible shapes and transforms them together.

Following Dedalus Transform.decrement_group pattern.

# Performance Benefits
- Reduced function call overhead
- Better cache utilization
- Potential for SIMD optimization in FFT libraries
"""
function group_forward_transform!(fields::Vector{<:ScalarField})
    if isempty(fields)
        return
    end

    # If grouping disabled or too few fields, transform individually
    if !GROUPED_TRANSFORM_CONFIG.enabled || length(fields) < GROUPED_TRANSFORM_CONFIG.min_fields
        for field in fields
            forward_transform!(field)
        end
        return
    end

    # Group fields by shape and transform type
    field_groups = _group_fields_by_shape(fields, :g)

    for (shape_key, group_fields) in field_groups
        if length(group_fields) == 1
            forward_transform!(group_fields[1])
        else
            _batch_forward_transform!(group_fields)
        end
    end
end

"""
    group_backward_transform!(fields::Vector{<:ScalarField})

Apply backward transforms to multiple fields simultaneously.
Batches fields with compatible shapes and transforms them together.

Following Dedalus Transform.increment_group pattern.
"""
function group_backward_transform!(fields::Vector{<:ScalarField})
    if isempty(fields)
        return
    end

    # If grouping disabled or too few fields, transform individually
    if !GROUPED_TRANSFORM_CONFIG.enabled || length(fields) < GROUPED_TRANSFORM_CONFIG.min_fields
        for field in fields
            backward_transform!(field)
        end
        return
    end

    # Group fields by shape and transform type
    field_groups = _group_fields_by_shape(fields, :c)

    for (shape_key, group_fields) in field_groups
        if length(group_fields) == 1
            backward_transform!(group_fields[1])
        else
            _batch_backward_transform!(group_fields)
        end
    end
end

"""
    _group_fields_by_shape(fields, layout_sym)

Group fields by their data shape and domain for batched processing.
Returns Dict{shape_key => Vector{ScalarField}}.
"""
function _group_fields_by_shape(fields::Vector{<:ScalarField}, layout_sym::Symbol)
    groups = Dict{Tuple, Vector{ScalarField}}()

    for field in fields
        # Ensure field is in correct layout
        ensure_layout!(field, layout_sym)

        # Get shape key based on layout
        data = layout_sym == :g ? get_grid_data(field) : get_coeff_data(field)
        if data === nothing
            continue
        end

        shape_key = (size(data)..., eltype(data), objectid(field.dist))

        if haskey(groups, shape_key)
            push!(groups[shape_key], field)
        else
            groups[shape_key] = [field]
        end
    end

    return groups
end

"""
    _batch_forward_transform!(fields::Vector{<:ScalarField})

Batch forward transform for fields with the same shape.

Uses PencilFFTs when available:
- For PencilFFTs plans: transforms are applied per-field but with shared buffers
- For FFTW transforms: stacks field data for batched FFT operations

Note: PencilArrays.transpose! is used internally by PencilFFTs for the
multi-dimensional parallel transforms.
"""
function _batch_forward_transform!(fields::Vector{<:ScalarField})
    if isempty(fields)
        return
    end

    first_field = fields[1]
    dist = first_field.dist

    # Check for PencilFFTs plan
    pencil_plan = nothing
    for t in dist.transforms
        if isa(t, PencilFFTs.PencilFFTPlan)
            pencil_plan = t
            break
        end
    end

    if pencil_plan !== nothing
        # PencilFFTs path: use the optimized PencilFFTs plan for each field
        # PencilFFTs internally uses PencilArrays.transpose! for dimension swaps
        # The benefit here is reusing the same plan and reducing setup overhead
        _pencil_batch_forward_transform!(fields, pencil_plan)
    else
        # FFTW path: stack field data for batched FFT operations
        _stacked_forward_transform!(fields)
    end
end

"""
    _batch_backward_transform!(fields::Vector{<:ScalarField})

Batch backward transform for fields with the same shape.
"""
function _batch_backward_transform!(fields::Vector{<:ScalarField})
    if isempty(fields)
        return
    end

    first_field = fields[1]
    dist = first_field.dist

    pencil_plan = nothing
    for t in dist.transforms
        if isa(t, PencilFFTs.PencilFFTPlan)
            pencil_plan = t
            break
        end
    end

    if pencil_plan !== nothing
        _pencil_batch_backward_transform!(fields, pencil_plan)
    else
        _stacked_backward_transform!(fields)
    end
end

"""
    _pencil_batch_forward_transform!(fields::Vector{<:ScalarField}, plan::PencilFFTs.PencilFFTPlan)

Forward transform using PencilFFTs for a batch of fields.
PencilFFTs handles multi-dimensional transforms with automatic pencil transposes.

This reuses the same PencilFFT plan for all fields, avoiding repeated plan creation.
The PencilFFTs library internally uses PencilArrays.transpose! for efficient
MPI communication during dimension swaps.
"""
function _pencil_batch_forward_transform!(fields::Vector{<:ScalarField},
                                          plan::PencilFFTs.PencilFFTPlan)
    # Get pencil configuration from plan
    # PencilFFTs allocates internal buffers, we reuse them across fields

    for field in fields
        grid_data = get_grid_data(field)

        if is_gpu_array(grid_data)
            # PencilFFTs is CPU-only; transfer to CPU first
            host_data = Array(grid_data)
            host_result = plan * host_data  # Uses PencilArrays.transpose! internally
            set_coeff_data!(field, copy_to_device(host_result, grid_data))
        else
            # Direct PencilFFT transform (uses PencilArrays for transposes)
            set_coeff_data!(field, plan * grid_data)
        end

        field.current_layout = :c
    end
end

"""
    _pencil_batch_backward_transform!(fields::Vector{<:ScalarField}, plan::PencilFFTs.PencilFFTPlan)

Backward transform using PencilFFTs for a batch of fields.
"""
function _pencil_batch_backward_transform!(fields::Vector{<:ScalarField},
                                           plan::PencilFFTs.PencilFFTPlan)
    for field in fields
        coeff_data = get_coeff_data(field)

        if is_gpu_array(coeff_data)
            host_data = Array(coeff_data)
            host_result = plan \ host_data  # Uses PencilArrays.transpose! internally
            set_grid_data!(field, copy_to_device(host_result, coeff_data))
        else
            set_grid_data!(field, plan \ coeff_data)
        end

        field.current_layout = :g
    end
end

"""
    _stacked_forward_transform!(fields::Vector{<:ScalarField})

Forward transform using stacked array approach.
Stacks fields along a new first dimension, transforms, unstacks.
"""
function _stacked_forward_transform!(fields::Vector{<:ScalarField})
    nfields = length(fields)
    if nfields == 0
        return
    end

    # Check for Legendre transforms — batched Legendre is not implemented,
    # so fall back to per-field transforms to avoid silent incorrect results
    dist = fields[1].dist
    has_legendre = any(isa(t, LegendreTransform) for t in dist.transforms)
    if has_legendre
        for field in fields
            forward_transform!(field)
        end
        return
    end

    # Get grid data and stack
    first_data = get_grid_data(fields[1])
    field_shape = size(first_data)
    dtype = eltype(first_data)
    stacked_shape = (nfields, field_shape...)

    # Allocate stacked array (reuse if possible)
    stacked = zeros(dtype, stacked_shape)

    # Copy field data into stacked array
    for (i, field) in enumerate(fields)
        data = get_grid_data(field)
        selectdim(stacked, 1, i) .= data
    end

    # Apply transforms to each field index
    # Note: We transform along dimensions 2:ndims, not 1
    for transform in dist.transforms
        if isa(transform, FourierTransform)
            # Shift axis by 1 for stacked array
            shifted_axis = transform.axis + 1
            dims = (shifted_axis,)

            if isa(transform.basis, RealFourier)
                if dtype <: Complex
                    stacked = FFTW.fft(stacked, dims)
                else
                    stacked = FFTW.rfft(stacked, dims)
                end
            else
                stacked = FFTW.fft(stacked, dims)
            end

        elseif isa(transform, ChebyshevTransform)
            # Apply Chebyshev to each field in the stack
            shifted_axis = transform.axis + 1
            stacked = _stacked_chebyshev_forward(stacked, transform, shifted_axis)

        elseif isa(transform, LegendreTransform)
            # Should not reach here due to early return above
            shifted_axis = transform.axis + 1
            stacked = _stacked_legendre_forward(stacked, transform, shifted_axis)
        end
    end

    # Unstack results back to fields
    for (i, field) in enumerate(fields)
        coeff_data = selectdim(stacked, 1, i)
        set_coeff_data!(field, collect(coeff_data))
        field.current_layout = :c
    end
end

"""
    _stacked_backward_transform!(fields::Vector{<:ScalarField})

Backward transform using stacked array approach.
"""
function _stacked_backward_transform!(fields::Vector{<:ScalarField})
    nfields = length(fields)
    if nfields == 0
        return
    end

    # Check for Legendre transforms — batched Legendre is not implemented,
    # so fall back to per-field transforms to avoid silent incorrect results
    dist = fields[1].dist
    has_legendre = any(isa(t, LegendreTransform) for t in dist.transforms)
    if has_legendre
        for field in fields
            backward_transform!(field)
        end
        return
    end

    # Get coeff data and stack
    first_data = get_coeff_data(fields[1])
    field_shape = size(first_data)
    dtype = eltype(first_data)
    stacked_shape = (nfields, field_shape...)

    # Allocate stacked array
    stacked = zeros(dtype, stacked_shape)

    # Copy field data into stacked array
    for (i, field) in enumerate(fields)
        data = get_coeff_data(field)
        selectdim(stacked, 1, i) .= data
    end

    # Apply transforms in reverse order
    for transform in reverse(dist.transforms)
        if isa(transform, FourierTransform)
            shifted_axis = transform.axis + 1
            dims = (shifted_axis,)

            if isa(transform.basis, RealFourier)
                actual_size = size(stacked, shifted_axis)
                expected_rfft_size = div(transform.basis.meta.size, 2) + 1

                if actual_size == expected_rfft_size
                    stacked = FFTW.irfft(stacked, transform.basis.meta.size, dims)
                else
                    stacked = FFTW.ifft(stacked, dims)
                end
            else
                stacked = FFTW.ifft(stacked, dims)
            end

        elseif isa(transform, ChebyshevTransform)
            shifted_axis = transform.axis + 1
            stacked = _stacked_chebyshev_backward(stacked, transform, shifted_axis)

        elseif isa(transform, LegendreTransform)
            shifted_axis = transform.axis + 1
            stacked = _stacked_legendre_backward(stacked, transform, shifted_axis)
        end
    end

    # Unstack results back to fields
    for (i, field) in enumerate(fields)
        grid_data = selectdim(stacked, 1, i)
        collected = collect(grid_data)
        # Only take real part for real-valued fields; preserve complex data for ComplexFourier
        if field.dtype <: Complex
            set_grid_data!(field, collected)
        else
            set_grid_data!(field, real.(collected))
        end
        field.current_layout = :g
    end
end

"""
    _stacked_chebyshev_forward(data, transform, axis)

Chebyshev forward transform for stacked array.
"""
function _stacked_chebyshev_forward(data::AbstractArray, transform::ChebyshevTransform, axis::Int)
    # Use DCT-I (REDFT00) to match the Gauss-Lobatto grid used by the per-field path
    n = size(data, axis)
    coeff_size = transform.coeff_size
    real_type = real(eltype(data))

    # DCT-I normalization: divide by (N-1), half-weight at endpoints
    norm_factor = real_type(n > 1 ? 1.0 / (n - 1) : 1.0)

    if eltype(data) <: Complex
        # Transform real and imaginary parts separately
        real_part = FFTW.r2r(real.(data), FFTW.REDFT00, (axis,))
        imag_part = FFTW.r2r(imag.(data), FFTW.REDFT00, (axis,))

        real_part .*= norm_factor
        imag_part .*= norm_factor
        _scale_first_along_axis!(real_part, axis, real_type(0.5))
        _scale_last_along_axis!(real_part, axis, real_type(0.5))
        _scale_first_along_axis!(imag_part, axis, real_type(0.5))
        _scale_last_along_axis!(imag_part, axis, real_type(0.5))

        full_result = complex.(real_part, imag_part)
    else
        full_result = FFTW.r2r(data, FFTW.REDFT00, (axis,))
        full_result .*= norm_factor
        _scale_first_along_axis!(full_result, axis, real_type(0.5))
        _scale_last_along_axis!(full_result, axis, real_type(0.5))
    end

    # Truncate to coeff_size for dealiasing (matching per-field path)
    if coeff_size > 0 && coeff_size < n
        ncopy = min(n, coeff_size)
        out_shape = ntuple(i -> i == axis ? coeff_size : size(full_result, i), ndims(full_result))
        out = zeros(eltype(full_result), out_shape)
        idx = ntuple(i -> i == axis ? (1:ncopy) : Colon(), ndims(full_result))
        out[idx...] .= full_result[idx...]
        return out
    end

    return full_result
end

"""
    _stacked_chebyshev_backward(data, transform, axis)

Chebyshev backward transform for stacked array.
"""
function _stacked_chebyshev_backward(data::AbstractArray, transform::ChebyshevTransform, axis::Int)
    coeff_size = size(data, axis)
    grid_size = transform.grid_size
    real_type = real(eltype(data))

    # DCT-I backward: undo endpoint halving (double DC and last coeff)
    # This matches the per-field _chebyshev_backward path
    function _prescale_for_dct1_backward!(arr, ax, cs, gs)
        _scale_first_along_axis!(arr, ax, real_type(2.0))
        if cs > 1 && cs == gs
            _scale_last_along_axis!(arr, ax, real_type(2.0))
        end
    end

    # Zero-pad from coeff_size to grid_size for dealiasing (matching per-field path)
    if grid_size > 0 && grid_size != coeff_size
        padded_shape = ntuple(i -> i == axis ? grid_size : size(data, i), ndims(data))
        ncopy = min(coeff_size, grid_size)
        idx = ntuple(i -> i == axis ? (1:ncopy) : Colon(), ndims(data))

        if eltype(data) <: Complex
            scaled_real = copy(real.(data))
            scaled_imag = copy(imag.(data))
            _prescale_for_dct1_backward!(scaled_real, axis, coeff_size, grid_size)
            _prescale_for_dct1_backward!(scaled_imag, axis, coeff_size, grid_size)

            padded_real = zeros(real_type, padded_shape)
            padded_imag = zeros(real_type, padded_shape)
            padded_real[idx...] .= scaled_real[idx...]
            padded_imag[idx...] .= scaled_imag[idx...]

            real_part = FFTW.r2r(padded_real, FFTW.REDFT00, (axis,))
            imag_part = FFTW.r2r(padded_imag, FFTW.REDFT00, (axis,))
            # DCT-I(DCT-I(x)) = 2(N-1)*x, forward already divided by (N-1), so divide by 2
            real_part ./= real_type(2.0)
            imag_part ./= real_type(2.0)
            return complex.(real_part, imag_part)
        else
            scaled_data = copy(real_type.(data))
            _prescale_for_dct1_backward!(scaled_data, axis, coeff_size, grid_size)

            padded_data = zeros(real_type, padded_shape)
            padded_data[idx...] .= scaled_data[idx...]
            result = FFTW.r2r(padded_data, FFTW.REDFT00, (axis,))
            result ./= real_type(2.0)
            return result
        end
    end

    # No padding needed: coeff_size == grid_size
    if eltype(data) <: Complex
        scaled_real = copy(real.(data))
        scaled_imag = copy(imag.(data))
        _prescale_for_dct1_backward!(scaled_real, axis, coeff_size, grid_size)
        _prescale_for_dct1_backward!(scaled_imag, axis, coeff_size, grid_size)

        real_part = FFTW.r2r(scaled_real, FFTW.REDFT00, (axis,))
        imag_part = FFTW.r2r(scaled_imag, FFTW.REDFT00, (axis,))
        real_part ./= real_type(2.0)
        imag_part ./= real_type(2.0)
        return complex.(real_part, imag_part)
    else
        scaled_data = copy(real_type.(data))
        _prescale_for_dct1_backward!(scaled_data, axis, coeff_size, grid_size)
        result = FFTW.r2r(scaled_data, FFTW.REDFT00, (axis,))
        result ./= real_type(2.0)
        return result
    end
end

"""
    _stacked_legendre_forward(data, transform, axis)

Legendre forward transform for stacked array.
"""
function _stacked_legendre_forward(data::AbstractArray, transform::LegendreTransform, axis::Int)
    # Batched Legendre not yet implemented — signal to caller to use per-field path
    error("Batched Legendre forward transform not implemented. " *
          "Disable grouped transforms for Legendre bases or use per-field transforms.")
end

"""
    _stacked_legendre_backward(data, transform, axis)

Legendre backward transform for stacked array.
"""
function _stacked_legendre_backward(data::AbstractArray, transform::LegendreTransform, axis::Int)
    # Batched Legendre not yet implemented — signal to caller to use per-field path
    error("Batched Legendre backward transform not implemented. " *
          "Disable grouped transforms for Legendre bases or use per-field transforms.")
end

