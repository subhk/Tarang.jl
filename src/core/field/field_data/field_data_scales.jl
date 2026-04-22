"""
    Field scaling and resampling

This file contains scale-management, grid resampling, and local-data helper
functions for scalar and vector fields.
"""

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
