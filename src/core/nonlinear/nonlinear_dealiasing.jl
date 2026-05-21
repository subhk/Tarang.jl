# Dealiasing functions
"""Apply 2D dealiasing using PencilFFT transforms"""
function apply_2d_dealiasing(data::AbstractArray, transform_info::PencilTransformConfig, dealiasing_factor::Float64)

    # Transform to spectral space
    fft_plan = transform_info.fft_plan_1
    spectral_data = fft_plan * data

    # Zero out high-frequency modes (2/3 rule: keep |k| <= N/(2*factor))
    shape = transform_info.shape
    cutoff_x = Int(floor(shape[1] / (2 * dealiasing_factor)))
    cutoff_y = Int(floor(shape[2] / (2 * dealiasing_factor)))

    # Apply dealiasing cutoff
    apply_spectral_cutoff!(spectral_data, (cutoff_x, cutoff_y))

    # Transform back to grid space using backward FFT with normalization
    # FFTW's ifft = bfft / N, so we use bfft and divide by the total size
    dealiased_data = FFTW.bfft(spectral_data) / length(spectral_data)

    return dealiased_data
end

"""
    apply_3d_dealiasing!(field::ScalarField, dealiasing_factor::Float64)

Apply 3D dealiasing.
GPU-compatible: uses appropriate implementation based on field's architecture.
"""
function apply_3d_dealiasing!(field::ScalarField, dealiasing_factor::Float64)
    nb = length(field.bases)
    # Compute cutoffs: keep modes with |k| <= N/(2*dealiasing_factor)
    cutoffs_check = ntuple(nb) do i
        basis = field.bases[i]
        isa(basis, Union{RealFourier, ComplexFourier}) ?
            Int(floor(basis.meta.size / (2 * dealiasing_factor))) : typemax(Int) >> 1
    end

    # Skip if grid too small for meaningful dealiasing
    any_zero_cutoff = any(i -> isa(field.bases[i], Union{RealFourier, ComplexFourier}) &&
                               cutoffs_check[i] == 0, 1:nb)
    if any_zero_cutoff
        return
    end

    # Transform to coefficient space
    ensure_layout!(field, :c)

    coeff_data = get_coeff_data(field)

    if isa(coeff_data, PencilArrays.PencilArray)
        # MPI-distributed: use per-rank global wavenumbers (local-index cutoff is wrong here)
        _apply_spectral_cutoff_distributed!(coeff_data, field.bases, dealiasing_factor)
    else
        cutoffs = ntuple(nb) do i
            basis = field.bases[i]
            isa(basis, Union{RealFourier, ComplexFourier}) ?
                Int(floor(basis.meta.size / (2 * dealiasing_factor))) : size(coeff_data, i)
        end

        rfft_dims = ntuple(nb) do i
            basis = field.bases[i]
            isa(basis, RealFourier) && size(coeff_data, i) == div(basis.meta.size, 2) + 1
        end

        # Apply spectral cutoff - this function handles GPU arrays automatically
        apply_spectral_cutoff!(coeff_data, cutoffs, rfft_dims)
    end

    # Transform back to grid space
    backward_transform!(field)
end

"""
    apply_basic_dealiasing!(field::ScalarField, dealiasing_factor::Float64)

Basic dealiasing for fields without PencilFFT support.
GPU-compatible: uses appropriate implementation based on field's architecture.

Applies the 2/3 rule: after pointwise multiplication in grid space, the product
contains aliased high-frequency modes. This function removes them by:
1. Forward FFT to spectral space
2. Zero modes with |k| > N/(2*dealiasing_factor) (= N/3 for standard 3/2 rule)
3. Inverse FFT back to grid space

Note: This is "truncation-after-multiply" which removes aliased energy in
high modes but cannot undo aliasing contamination of low modes. For exact
dealiasing, use the padding approach (pad to 3N/2, multiply, truncate).
"""
function apply_basic_dealiasing!(field::ScalarField, dealiasing_factor::Float64)
    nb = length(field.bases)
    # Compute cutoff wavenumbers for each Fourier basis dimension.
    # For the 2/3 rule (dealiasing_factor=1.5): keep modes with |k| <= N/3.
    cutoffs_check = ntuple(nb) do i
        basis = field.bases[i]
        isa(basis, Union{RealFourier, ComplexFourier}) ?
            Int(floor(basis.meta.size / (2 * dealiasing_factor))) : typemax(Int) >> 1
    end

    # Skip dealiasing if any Fourier cutoff is 0 (grid too small for meaningful dealiasing).
    any_zero_cutoff = any(i -> isa(field.bases[i], Union{RealFourier, ComplexFourier}) &&
                               cutoffs_check[i] == 0, 1:nb)
    if any_zero_cutoff
        return
    end

    # Transform to spectral space
    forward_transform!(field)

    coeff_data = get_coeff_data(field)

    if isa(coeff_data, PencilArrays.PencilArray)
        # MPI-distributed: use per-rank global wavenumbers (local-index cutoff is wrong here)
        _apply_spectral_cutoff_distributed!(coeff_data, field.bases, dealiasing_factor)
    else
        # Recompute cutoffs using actual coeff array sizes for non-Fourier bases
        cutoffs = ntuple(nb) do i
            basis = field.bases[i]
            isa(basis, Union{RealFourier, ComplexFourier}) ?
                Int(floor(basis.meta.size / (2 * dealiasing_factor))) : size(coeff_data, i)
        end

        rfft_dims = ntuple(nb) do i
            basis = field.bases[i]
            isa(basis, RealFourier) && size(coeff_data, i) == div(basis.meta.size, 2) + 1
        end

        # Apply spectral cutoff - this function handles GPU arrays automatically
        apply_spectral_cutoff!(coeff_data, cutoffs, rfft_dims)
    end

    # Transform back to grid space
    backward_transform!(field)
end


"""
    apply_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=())

Apply spectral cutoff to remove high-frequency modes (dealiasing).

For spectral data stored in standard FFT layout:
- Positive frequencies: indices 1 to N/2+1
- Negative frequencies: indices N/2+2 to N (for complex FFT)

For rfft output dimensions (indicated by rfft_dims):
- All indices are positive frequencies: k = 0, 1, ..., N/2
- No negative frequency region

This function zeros out modes beyond the cutoff wavenumber in each dimension.
Used for dealiasing in nonlinear term evaluation.

GPU-compatible: Uses broadcasting-based implementation for GPU arrays.

Arguments:
- data: Complex spectral coefficient array
- cutoffs: Tuple of cutoff wavenumbers for each dimension
- rfft_dims: Tuple of Bool indicating which dimensions are rfft output
             (all positive frequencies, no negative frequency region)

The cutoff is applied symmetrically: modes with |k| > cutoff are zeroed.
"""
function apply_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    ndims_data = ndims(data)

    # Check if data is on GPU - use broadcasting-based implementation
    if is_gpu_array(data)
        apply_spectral_cutoff_gpu!(data, cutoffs, rfft_dims)
        return
    end

    # CPU implementation with loops
    if ndims_data == 1
        if length(rfft_dims) >= 1 && rfft_dims[1]
            apply_rfft_spectral_cutoff!(data, cutoffs[1])
        else
            apply_1d_spectral_cutoff!(data, 1, cutoffs[1])
        end
    elseif ndims_data == 2
        apply_2d_spectral_cutoff!(data, cutoffs, rfft_dims)
    elseif ndims_data == 3
        apply_3d_spectral_cutoff!(data, cutoffs, rfft_dims)
    else
        # General N-dimensional case
        apply_nd_spectral_cutoff!(data, cutoffs, rfft_dims)
    end
end

"""
    apply_spectral_cutoff_gpu!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=())

GPU-compatible spectral cutoff using broadcasting with a mask.
Creates a dealiasing mask and applies it element-wise via broadcasting.
"""
function apply_spectral_cutoff_gpu!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    shape = size(data)

    # Build the dealiasing mask using broadcasting
    # The mask is 1.0 where modes should be kept, 0.0 where they should be zeroed
    mask = create_dealiasing_mask(shape, cutoffs, eltype(data), rfft_dims)

    # Move mask to same device as data
    # Use architecture(data) to infer the correct architecture from the array
    arch = architecture(data)
    mask_device = on_architecture(arch, mask)

    # Apply mask using broadcasting (GPU-compatible)
    data .*= mask_device
end

"""
    _axis_dealias_factor(basis, fallback) -> Float64

Per-axis dealiasing factor, read literally from the basis's `dealias` setting.
The Fourier bases default to 3/2 (the standard 2/3-rule), so a basis left unset
still dealiases. Setting `dealias=1` (or any value ≤ 1) disables dealiasing on
that axis (all modes kept, aliasing accepted); larger values mean stronger
dealiasing. Non-Fourier bases return the fallback unchanged.
"""
function _axis_dealias_factor(basis, fallback::Float64)
    isa(basis, Union{RealFourier, ComplexFourier}) || return fallback
    d = basis.meta.dealias
    return d isa Number ? Float64(d) : Float64(first(d))
end

"""
    _any_axis_dealias(bases, fallback) -> Bool

True if any Fourier axis requests dealiasing (factor > 1). Used to gate the
nonlinear-product dealiasing: when every Fourier axis has `dealias ≤ 1`, the
product is computed without any dealiasing.
"""
function _any_axis_dealias(bases, fallback::Float64)
    for basis in bases
        isa(basis, Union{RealFourier, ComplexFourier}) || continue
        _axis_dealias_factor(basis, fallback) > 1 && return true
    end
    return false
end

"""
    _apply_spectral_cutoff_distributed!(coeff_data::PencilArray, bases, dealiasing_factor)

Apply a 2/3-rule dealiasing cutoff to MPI-distributed coefficient data.

Unlike `apply_spectral_cutoff!`, which derives wavenumbers from the LOCAL array
size, this version uses each rank's GLOBAL wavenumber indices (via the pencil's
`axes_local` range) so the correct modes are zeroed regardless of how the
spectrum is decomposed across ranks. The mode-number layout per axis matches the
PencilFFT coefficient layout:
  - first RealFourier axis (RFFT): modes 0, 1, …, N/2
  - later RealFourier / ComplexFourier axes (FFT): 0, 1, …, N/2-1, -N/2, …, -1

Modes with |mode| > floor(N / (2·dealiasing_factor)) are zeroed on each Fourier axis.
Mirrors the per-rank wavenumber handling in `_apply_spectral_derivative_distributed!`.
"""
function _apply_spectral_cutoff_distributed!(coeff_data::PencilArrays.PencilArray,
                                             bases, dealiasing_factor::Float64)
    local_data = parent(coeff_data)
    pencil = PencilArrays.pencil(coeff_data)
    local_axes = pencil.axes_local
    perm = PencilArrays.permutation(coeff_data)
    perm_tuple = Tuple(perm)

    for axis in 1:length(bases)
        basis = bases[axis]
        (isa(basis, RealFourier) || isa(basis, ComplexFourier)) || continue

        N = basis.meta.size
        factor = _axis_dealias_factor(basis, dealiasing_factor)
        factor <= 1 && continue   # dealias ≤ 1 → no truncation on this axis
        # Quadratic-alias-safe cutoff: keep |mode| ≤ cutoff with 3·cutoff < N so the
        # highest product mode (2·cutoff) cannot alias back into the retained band.
        # Truncation can keep at most (N-1)÷3 modes alias-free regardless of factor, so
        # a larger factor (stronger dealiasing) lowers the cutoff; a smaller one is
        # capped at the alias-free limit.
        cutoff = min(floor(Int, N / (2 * factor)), (N - 1) ÷ 3)

        # Global integer mode numbers matching the coefficient layout on this axis.
        if isa(basis, RealFourier) && _is_first_real_fourier_axis(bases, axis)
            modes_global = collect(0:div(N, 2))                 # RFFT: 0 … N/2
        else
            modes_global = round.(Int, _fftfreq(N) .* N)        # FFT: 0…N/2-1, -N/2…-1
        end

        # Validate the coefficient global size matches the expected layout.
        global_size_axis = PencilArrays.size_global(coeff_data)[axis]
        if global_size_axis != length(modes_global)
            error("Distributed spectral cutoff size mismatch on axis $axis: " *
                  "global coefficient size is $global_size_axis but expected " *
                  "$(length(modes_global)) (basis=$(typeof(basis)), N=$N).")
        end

        # Slice the global mode numbers to this rank's owned range on this axis.
        if axis <= length(local_axes)
            local_range = local_axes[axis]
            modes_local = modes_global[local_range]
        else
            modes_local = modes_global
        end

        keep = abs.(modes_local) .<= cutoff
        all(keep) && continue  # nothing to zero on this rank for this axis

        # Broadcast the keep-mask along the PHYSICAL axis (accounting for permutation).
        physical_axis = findfirst(==(axis), perm_tuple)
        physical_axis === nothing && (physical_axis = axis)
        mask_shape = ntuple(i -> i == physical_axis ? length(keep) : 1, ndims(local_data))
        local_data .*= reshape(keep, mask_shape...)
    end

    return coeff_data
end

"""
    create_dealiasing_mask(shape::Tuple, cutoffs::Tuple, T::Type, rfft_dims::Tuple=())

Create a dealiasing mask array for the given shape and cutoffs.
Returns a CPU array; caller should move to GPU if needed.

The mask is 1 where |k_i| <= cutoff_i for all dimensions, 0 otherwise.

For rfft dimensions (rfft_dims[d] == true), all indices represent positive
frequencies (k = 0, 1, ..., N/2) and the mask zeros k > cutoff.
For standard FFT dimensions, the standard layout with negative frequencies is used.
"""
function create_dealiasing_mask(shape::Tuple, cutoffs::Tuple, T::Type, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    ndims_data = length(shape)

    # Build 1D masks for each dimension, then combine with outer product
    masks_1d = Vector{Vector{real(T)}}(undef, ndims_data)

    for d in 1:ndims_data
        n = shape[d]
        cutoff = d <= length(cutoffs) ? cutoffs[d] : div(n, 2)
        is_rfft = d <= length(rfft_dims) && rfft_dims[d]

        mask_d = zeros(real(T), n)
        for i in 1:n
            if is_rfft
                # rfft output: all indices are positive frequencies k = i-1
                k = i - 1
                mask_d[i] = k <= cutoff ? one(real(T)) : zero(real(T))
            else
                # Standard complex FFT layout:
                # indices 1 to N/2+1: k = 0, 1, ..., N/2
                # indices N/2+2 to N: k = -(N/2-1), ..., -1
                half_n = div(n, 2)
                k = i <= half_n + 1 ? i - 1 : i - n - 1
                mask_d[i] = abs(k) <= cutoff ? one(real(T)) : zero(real(T))
            end
        end
        masks_1d[d] = mask_d
    end

    # Combine 1D masks into N-D mask using broadcasting
    if ndims_data == 1
        return masks_1d[1]
    elseif ndims_data == 2
        # Outer product of two 1D masks
        return masks_1d[1] .* masks_1d[2]'
    elseif ndims_data == 3
        # 3D: reshape and broadcast
        mask_x = reshape(masks_1d[1], :, 1, 1)
        mask_y = reshape(masks_1d[2], 1, :, 1)
        mask_z = reshape(masks_1d[3], 1, 1, :)
        return mask_x .* mask_y .* mask_z
    else
        # General N-D case: use recursive broadcasting
        mask = ones(real(T), shape...)
        for d in 1:ndims_data
            # Create shape for broadcasting: all 1s except dimension d
            broadcast_shape = ntuple(i -> i == d ? shape[d] : 1, ndims_data)
            mask_d = reshape(masks_1d[d], broadcast_shape...)
            mask .*= mask_d
        end
        return mask
    end
end

"""
    apply_rfft_spectral_cutoff!(data::AbstractVector, cutoff::Int)

Apply spectral cutoff to rfft output (all positive frequencies).

For rfft output with N/2+1 points:
- Index 1: k=0 (DC component)
- Index i: k=i-1 (all positive frequencies)

Modes with k > cutoff are set to zero.
"""
function apply_rfft_spectral_cutoff!(data::AbstractVector, cutoff::Int)
    n = length(data)
    # rfft: all indices are positive frequencies k = i-1
    # Zero indices where k > cutoff, i.e., i > cutoff+1
    for i in (cutoff + 2):n
        data[i] = zero(eltype(data))
    end
end

"""
    Apply 1D spectral cutoff along a vector.

    For FFT layout with N points:
    - Index 1: k=0 (DC component)
    - Indices 2 to N/2+1: positive frequencies k=1 to N/2
    - Indices N/2+2 to N: negative frequencies k=-(N/2-1) to -1

    Modes with |k| > cutoff are set to zero.
    """
function apply_1d_spectral_cutoff!(data::AbstractVector, axis::Int, cutoff::Int)
    n = length(data)
    if cutoff >= div(n, 2)
        return  # No cutoff needed
    end

    # Zero positive high frequencies: indices cutoff+2 to N/2+1
    # (index 1 is k=0, index 2 is k=1, ..., index cutoff+1 is k=cutoff)
    half_n = div(n, 2)
    for i in (cutoff + 2):(half_n + 1)
        if i <= n
            data[i] = zero(eltype(data))
        end
    end

    # Zero negative high frequencies: indices N/2+2 to N-cutoff
    # Negative frequencies are stored in reverse order at the end
    for i in (half_n + 2):(n - cutoff)
        if i <= n
            data[i] = zero(eltype(data))
        end
    end
end

"""
    Apply 1D spectral cutoff along specified axis of multi-dimensional array.
    """
function apply_1d_spectral_cutoff!(data::AbstractArray, axis::Int, cutoff::Int)
    shape = size(data)
    n = shape[axis]

    if cutoff >= div(n, 2)
        return  # No cutoff needed
    end

    half_n = div(n, 2)

    # Create index ranges for slicing
    # Zero out positive high frequencies
    for k in (cutoff + 2):(half_n + 1)
        if k <= n
            indices = ntuple(ndims(data)) do d
                d == axis ? k : Colon()
            end
            data[indices...] .= zero(eltype(data))
        end
    end

    # Zero out negative high frequencies
    for k in (half_n + 2):(n - cutoff)
        if k <= n
            indices = ntuple(ndims(data)) do d
                d == axis ? k : Colon()
            end
            data[indices...] .= zero(eltype(data))
        end
    end
end

"""
    Apply 2D spectral cutoff for dealiasing.

    Zeros out modes where |kx| > cutoffs[1] or |ky| > cutoffs[2].
    For rfft dimensions, all indices are positive frequencies (k=i-1).
    """
function apply_2d_spectral_cutoff!(data::AbstractMatrix, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, 2))
    nx, ny = size(data)
    kx_cut = cutoffs[1]
    ky_cut = length(cutoffs) >= 2 ? cutoffs[2] : div(ny, 2)

    x_is_rfft = length(rfft_dims) >= 1 && rfft_dims[1]
    y_is_rfft = length(rfft_dims) >= 2 && rfft_dims[2]

    half_nx = div(nx, 2)
    half_ny = div(ny, 2)

    for j in 1:ny
        # Determine wavenumber for y-dimension
        if y_is_rfft
            ky = j - 1  # rfft: all positive
            y_in_range = ky <= ky_cut
        else
            ky = j <= half_ny + 1 ? j - 1 : j - ny - 1
            y_in_range = abs(ky) <= ky_cut
        end

        for i in 1:nx
            # Determine wavenumber for x-dimension
            if x_is_rfft
                kx = i - 1  # rfft: all positive
                x_in_range = kx <= kx_cut
            else
                kx = i <= half_nx + 1 ? i - 1 : i - nx - 1
                x_in_range = abs(kx) <= kx_cut
            end

            # Zero out if either frequency is outside cutoff
            if !x_in_range || !y_in_range
                data[i, j] = zero(eltype(data))
            end
        end
    end
end

"""
    Apply 3D spectral cutoff for dealiasing.

    Zeros out modes where |kx| > cutoffs[1], |ky| > cutoffs[2], or |kz| > cutoffs[3].
    For rfft dimensions, all indices are positive frequencies (k=i-1).
    """
function apply_3d_spectral_cutoff!(data::AbstractArray{T, 3}, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, 3)) where T
    nx, ny, nz = size(data)
    kx_cut = cutoffs[1]
    ky_cut = length(cutoffs) >= 2 ? cutoffs[2] : div(ny, 2)
    kz_cut = length(cutoffs) >= 3 ? cutoffs[3] : div(nz, 2)

    x_is_rfft = length(rfft_dims) >= 1 && rfft_dims[1]
    y_is_rfft = length(rfft_dims) >= 2 && rfft_dims[2]
    z_is_rfft = length(rfft_dims) >= 3 && rfft_dims[3]

    half_nx = div(nx, 2)
    half_ny = div(ny, 2)
    half_nz = div(nz, 2)

    for k in 1:nz
        if z_is_rfft
            kz_val = k - 1
            z_in_range = kz_val <= kz_cut
        else
            kz_val = k <= half_nz + 1 ? k - 1 : k - nz - 1
            z_in_range = abs(kz_val) <= kz_cut
        end

        for j in 1:ny
            if y_is_rfft
                ky_val = j - 1
                y_in_range = ky_val <= ky_cut
            else
                ky_val = j <= half_ny + 1 ? j - 1 : j - ny - 1
                y_in_range = abs(ky_val) <= ky_cut
            end

            for i in 1:nx
                if x_is_rfft
                    kx_val = i - 1
                    x_in_range = kx_val <= kx_cut
                else
                    kx_val = i <= half_nx + 1 ? i - 1 : i - nx - 1
                    x_in_range = abs(kx_val) <= kx_cut
                end

                if !x_in_range || !y_in_range || !z_in_range
                    data[i, j, k] = zero(T)
                end
            end
        end
    end
end

"""
    Apply N-dimensional spectral cutoff (general case).
    """
function apply_nd_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple, rfft_dims::Tuple=ntuple(i->false, length(cutoffs)))
    shape = size(data)
    ndims_data = ndims(data)

    # Extend cutoffs to match dimensions
    actual_cutoffs = ntuple(ndims_data) do d
        d <= length(cutoffs) ? cutoffs[d] : div(shape[d], 2)
    end

    half_shape = div.(shape, 2)

    for I in CartesianIndices(data)
        # Check if any frequency is outside cutoff
        outside_cutoff = false

        for d in 1:ndims_data
            idx = I[d]
            n = shape[d]
            is_rfft = d <= length(rfft_dims) && rfft_dims[d]

            if is_rfft
                # rfft: all indices are positive frequencies
                k = idx - 1
                if k > actual_cutoffs[d]
                    outside_cutoff = true
                    break
                end
            else
                # Standard FFT layout
                half_n = half_shape[d]
                k = idx <= half_n + 1 ? idx - 1 : idx - n - 1
                if abs(k) > actual_cutoffs[d]
                    outside_cutoff = true
                    break
                end
            end
        end

        if outside_cutoff
            data[I] = zero(eltype(data))
        end
    end
end

"""
    apply_spherical_spectral_cutoff!(data::AbstractArray, k_max::Int)

Apply spherical spectral cutoff: zero modes with |k| > k_max.

This is useful for isotropic dealiasing where the cutoff is based
on the magnitude of the wavevector rather than individual components.

|k|² = kx² + ky² + kz² (for 3D)

GPU-compatible: Uses broadcasting-based implementation for GPU arrays.
"""
function apply_spherical_spectral_cutoff!(data::AbstractArray, k_max::Int)
    # Check if data is on GPU - use broadcasting-based implementation
    if is_gpu_array(data)
        apply_spherical_spectral_cutoff_gpu!(data, k_max)
        return
    end

    # CPU implementation with loops
    shape = size(data)
    ndims_data = ndims(data)
    half_shape = div.(shape, 2)
    k_max_sq = k_max^2

    for I in CartesianIndices(data)
        # Compute |k|²
        k_sq = 0
        for d in 1:ndims_data
            idx = I[d]
            n = shape[d]
            half_n = half_shape[d]
            k = idx <= half_n + 1 ? idx - 1 : idx - n - 1
            k_sq += k^2
        end

        if k_sq > k_max_sq
            data[I] = zero(eltype(data))
        end
    end
end

"""
    apply_spherical_spectral_cutoff_gpu!(data::AbstractArray, k_max::Int)

GPU-compatible spherical spectral cutoff using broadcasting with a mask.
"""
function apply_spherical_spectral_cutoff_gpu!(data::AbstractArray, k_max::Int)
    shape = size(data)
    ndims_data = ndims(data)

    # Create spherical mask on CPU, then move to device
    mask = create_spherical_mask(shape, k_max, eltype(data))

    # Use architecture(data) to infer the correct architecture from the array
    arch = architecture(data)
    mask_device = on_architecture(arch, mask)

    # Apply mask using broadcasting
    data .*= mask_device
end

"""
    create_spherical_mask(shape::Tuple, k_max::Int, T::Type)

Create a spherical dealiasing mask for the given shape and k_max.
Mask is 1 where |k| <= k_max, 0 otherwise.
"""
function create_spherical_mask(shape::Tuple, k_max::Int, T::Type)
    ndims_data = length(shape)
    k_max_sq = k_max^2

    # Create wavenumber arrays for each dimension
    k_arrays = Vector{Vector{Int}}(undef, ndims_data)
    for d in 1:ndims_data
        n = shape[d]
        half_n = div(n, 2)
        k_arrays[d] = [i <= half_n + 1 ? i - 1 : i - n - 1 for i in 1:n]
    end

    # Build mask based on |k|² <= k_max²
    if ndims_data == 1
        return real(T).([abs(k) <= k_max ? 1.0 : 0.0 for k in k_arrays[1]])
    elseif ndims_data == 2
        return real(T).([k_arrays[1][i]^2 + k_arrays[2][j]^2 <= k_max_sq ? 1.0 : 0.0
                         for i in 1:shape[1], j in 1:shape[2]])
    elseif ndims_data == 3
        return real(T).([k_arrays[1][i]^2 + k_arrays[2][j]^2 + k_arrays[3][k]^2 <= k_max_sq ? 1.0 : 0.0
                         for i in 1:shape[1], j in 1:shape[2], k in 1:shape[3]])
    else
        # General N-D case
        mask = zeros(real(T), shape...)
        for I in CartesianIndices(mask)
            k_sq = sum(k_arrays[d][I[d]]^2 for d in 1:ndims_data)
            mask[I] = k_sq <= k_max_sq ? one(real(T)) : zero(real(T))
        end
        return mask
    end
end

"""
    Compute spectral cutoffs for dealiasing.

    For the 2/3 rule (dealiasing_factor=1.5):
    cutoff = N / (2 * dealiasing_factor) = N/3

    This keeps modes with |k| <= N/3. When combined with proper padding
    (or used as a post-multiply filter), modes above this cutoff are zeroed
    to suppress aliasing from quadratic nonlinear interactions.
    """
function get_dealiasing_cutoffs(shape::Tuple, dealiasing_factor::Float64=1.5)
    return ntuple(i -> floor(Int, shape[i] / (2 * dealiasing_factor)), length(shape))
end
