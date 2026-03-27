"""
    Transform Chebyshev - Chebyshev transform execution

This file contains the forward and backward Chebyshev transform implementations
using DCT-I (Fast Cosine Transform).
"""

"""Scale the first slice along `axis` by `factor`."""
function _scale_first_along_axis!(data::AbstractArray, axis::Int, factor)
    idx = ntuple(i -> i == axis ? (1:1) : Colon(), ndims(data))
    data[idx...] .*= factor
end

"""Scale the last slice along `axis` by `factor`."""
function _scale_last_along_axis!(data::AbstractArray, axis::Int, factor)
    n = size(data, axis)
    idx = ntuple(i -> i == axis ? (n:n) : Colon(), ndims(data))
    data[idx...] .*= factor
end

function _chebyshev_forward(data::AbstractArray, transform::ChebyshevTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        grid_size = size(host_data, axis)
        coeff_size = transform.coeff_size

        real_type = real(eltype(host_data))

        # Use DCT-I (REDFT00) to match the Gauss-Lobatto grid: x_k = -cos(πk/(N-1))
        real_data = real.(host_data)
        temp_real = FFTW.r2r(real_data, FFTW.REDFT00, (axis,))

        # DCT-I normalization: divide by (N-1), half-weight at endpoints
        norm_factor = real_type(grid_size > 1 ? 1.0 / (grid_size - 1) : 1.0)
        temp_real .*= norm_factor
        # Half the first and last coefficients (DCT-I endpoint convention)
        _scale_first_along_axis!(temp_real, axis, real_type(0.5))
        _scale_last_along_axis!(temp_real, axis, real_type(0.5))

        out_shape = ntuple(i -> i == axis ? coeff_size : size(temp_real, i), ndims(temp_real))
        out_real = zeros(real_type, out_shape)
        ncopy = min(grid_size, coeff_size)
        idx = ntuple(i -> i == axis ? (1:ncopy) : Colon(), ndims(temp_real))
        out_real[idx...] .= temp_real[idx...]

        if eltype(host_data) <: Complex
            imag_data = imag.(host_data)
            temp_imag = FFTW.r2r(imag_data, FFTW.REDFT00, (axis,))
            temp_imag .*= norm_factor
            _scale_first_along_axis!(temp_imag, axis, real_type(0.5))
            _scale_last_along_axis!(temp_imag, axis, real_type(0.5))

            out_imag = zeros(real_type, out_shape)
            out_imag[idx...] .= temp_imag[idx...]
            return complex.(out_real, out_imag)
        end

        return out_real
    end
end

function _chebyshev_backward(data::AbstractArray, transform::ChebyshevTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        coeff_size = size(host_data, axis)
        grid_size = transform.grid_size

        real_type = real(eltype(host_data))

        # Prepare coefficients for DCT-I backward (undo the endpoint halving)
        # The forward transform halves the DC (first) and the physical last DCT-I mode
        # (at index grid_size). Only undo the last-endpoint doubling if the stored last
        # coefficient IS the physical last DCT-I mode (coeff_size == grid_size).
        scaled_real = copy(real.(host_data))
        _scale_first_along_axis!(scaled_real, axis, real_type(2.0))
        if coeff_size > 1 && coeff_size == grid_size
            _scale_last_along_axis!(scaled_real, axis, real_type(2.0))
        end

        # Zero-pad or truncate to grid_size
        padded_shape = ntuple(i -> i == axis ? grid_size : size(host_data, i), ndims(host_data))
        padded_real = zeros(real_type, padded_shape)
        ncopy = min(coeff_size, grid_size)
        idx = ntuple(i -> i == axis ? (1:ncopy) : Colon(), ndims(host_data))
        padded_real[idx...] .= scaled_real[idx...]

        # DCT-I backward (REDFT00 is its own inverse up to normalization)
        # The DCT-I is symmetric: applying it again with normalization gives the inverse
        temp_real = FFTW.r2r(padded_real, FFTW.REDFT00, (axis,))
        # No additional normalization needed — the forward already divided by (N-1)
        # and DCT-I(DCT-I(x)) = 2(N-1)*x, so we divide by 2
        temp_real ./= real_type(2.0)

        if eltype(host_data) <: Complex
            scaled_imag = copy(imag.(host_data))
            _scale_first_along_axis!(scaled_imag, axis, real_type(2.0))
            if coeff_size > 1 && coeff_size == grid_size
                _scale_last_along_axis!(scaled_imag, axis, real_type(2.0))
            end
            padded_imag = zeros(real_type, padded_shape)
            padded_imag[idx...] .= scaled_imag[idx...]
            temp_imag = FFTW.r2r(padded_imag, FFTW.REDFT00, (axis,))
            temp_imag ./= real_type(2.0)
            return complex.(temp_real, temp_imag)
        end

        return temp_real
    end
end

# Chebyshev transform application functions following Tarang patterns
function apply_chebyshev_forward!(field::ScalarField, transform::ChebyshevTransform)
    """
    Apply forward Chebyshev transform (grid to coefficients) with in-place operations.

    Based on Tarang ScipyDCT.forward and FFTWDCT.forward methods:
    - Uses DCT-II with proper scaling for unit-amplitude normalization
    - Handles padding/truncation for different grid/coefficient sizes
    - Follows resize_rescale_forward pattern
    - OPTIMIZED: Uses workspace buffers to minimize allocations
    """

    if ndims(get_grid_data(field)) != 1 || eltype(get_grid_data(field)) <: Complex || is_gpu_array(get_grid_data(field))
        set_coeff_data!(field, _chebyshev_forward(get_grid_data(field), transform))
        return
    end

    if transform.forward_plan !== nothing
        # Use FFTW DCT-I plan for CPU with workspace buffer
        N = transform.grid_size
        wm = get_global_workspace()
        temp_data = get_workspace!(wm, Float64, (N,))
        try
            # DCT-I forward via FFTW plan
            mul!(temp_data, transform.forward_plan, get_grid_data(field))

            # Ensure output array exists (use Float64 for real input fields)
            if get_coeff_data(field) === nothing
                set_coeff_data!(field, zeros(Float64, transform.coeff_size))
            end

            # DCT-I normalization: divide by (N-1), endpoint half-weighting
            Nm1 = max(N - 1, 1)
            @inbounds get_coeff_data(field)[1] = temp_data[1] / (2.0 * Nm1)  # DC: extra /2

            if transform.Kmax > 0
                inv_Nm1 = 1.0 / Nm1
                @inbounds @simd for k in 1:min(transform.Kmax, transform.coeff_size-1)
                    get_coeff_data(field)[k+1] = temp_data[k+1] * inv_Nm1
                end
                # Last coefficient gets /2 only if the physical DCT-I endpoint
                # (at index N) is actually stored. When coeff_size < N, the
                # endpoint is truncated away and the last retained coefficient
                # should NOT be halved.
                if transform.coeff_size >= N && N > 1
                    @inbounds get_coeff_data(field)[N] *= 0.5
                end
            end

            # Zero padding if coeff_size > grid_size
            if transform.coeff_size > N
                @inbounds @simd for k in (N + 1):transform.coeff_size
                    get_coeff_data(field)[k] = 0.0
                end
            end
        catch e
            @warn "DCT forward transform failed: $e, falling back to matrix method"
            apply_chebyshev_matrix_forward!(field, transform)
        finally
            release_workspace!(wm, temp_data)
        end

    else
        apply_chebyshev_matrix_forward!(field, transform)
    end
end

function apply_chebyshev_backward!(field::ScalarField, transform::ChebyshevTransform)
    """
    Apply backward Chebyshev transform (coefficients to grid) with in-place operations.

    Based on Tarang ScipyDCT.backward and FFTWDCT.backward methods:
    - Uses DCT-III with proper scaling for unit-amplitude normalization
    - Handles padding/truncation for different coefficient/grid sizes
    - OPTIMIZED: Uses workspace buffers and in-place operations
    """

    if ndims(get_coeff_data(field)) != 1 || eltype(get_coeff_data(field)) <: Complex || is_gpu_array(get_coeff_data(field))
        set_grid_data!(field, _chebyshev_backward(get_coeff_data(field), transform))
        return
    end

    if transform.backward_plan !== nothing
        # Use FFTW DCT-I plan for CPU with workspace
        # DCT-I backward: undo endpoint halving, apply DCT-I, divide by 2
        N = transform.grid_size
        Nm1 = max(N - 1, 1)
        wm = get_global_workspace()
        temp_data = get_workspace!(wm, Float64, (N,))
        try
            fill!(temp_data, 0.0)

            # Copy coefficients and undo the endpoint halving from forward
            ncopy = min(length(get_coeff_data(field)), N)
            if ncopy > 0
                @inbounds temp_data[1] = real(get_coeff_data(field)[1]) * 2.0  # undo DC /2
            end
            if ncopy > 1
                @inbounds @simd for k in 2:ncopy-1
                    temp_data[k] = real(get_coeff_data(field)[k])
                end
                # Only undo the last-endpoint halving if the stored last coeff
                # is the physical last DCT-I mode (ncopy == N)
                if ncopy == N
                    @inbounds temp_data[ncopy] = real(get_coeff_data(field)[ncopy]) * 2.0  # undo last /2
                else
                    @inbounds temp_data[ncopy] = real(get_coeff_data(field)[ncopy])
                end
            end

            # Ensure output array exists
            if get_grid_data(field) === nothing
                set_grid_data!(field, zeros(Float64, N))
            end

            # DCT-I is its own inverse: DCT-I(DCT-I(x)) = 2(N-1)*x
            # Forward divided by (N-1), so backward DCT-I then divide by 2 recovers original
            mul!(get_grid_data(field), transform.backward_plan, temp_data)
            get_grid_data(field) ./= 2.0
        catch e
            @warn "DCT backward transform failed: $e, falling back to matrix method"
            apply_chebyshev_matrix_backward!(field, transform)
        finally
            release_workspace!(wm, temp_data)
        end

    else
        apply_chebyshev_matrix_backward!(field, transform)
    end
end

function apply_chebyshev_matrix_forward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply forward Chebyshev transform using in-place matrix multiplication"""

    if haskey(transform.matrices, "forward")
        mat = transform.matrices["forward"]
        # Ensure output array exists with correct size
        out_size = size(mat, 1)
        coeff_dtype = coefficient_eltype(field.dtype)
        if get_coeff_data(field) === nothing || length(get_coeff_data(field)) != out_size || eltype(get_coeff_data(field)) != coeff_dtype
            set_coeff_data!(field, zeros(coeff_dtype, out_size))
        end
        # Ensure input grid data exists
        if get_grid_data(field) === nothing
            throw(ArgumentError("Chebyshev forward transform requires grid data for field $(field.name)"))
        end
        # In-place matrix-vector multiply
        mul!(get_coeff_data(field), mat, get_grid_data(field))
    else
        @warn "No forward matrix available for Chebyshev transform"
        if get_coeff_data(field) === nothing
            set_coeff_data!(field, copy(get_grid_data(field)))
        else
            copyto!(get_coeff_data(field), get_grid_data(field))
        end
    end
end

function apply_chebyshev_matrix_backward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply backward Chebyshev transform using in-place matrix multiplication"""

    if haskey(transform.matrices, "backward")
        mat = transform.matrices["backward"]
        out_size = size(mat, 1)

        if get_coeff_data(field) === nothing
            throw(ArgumentError("Chebyshev backward transform requires coefficient data for field $(field.name)"))
        end

        # Get workspace for real part extraction
        wm = get_global_workspace()
        coeff_len = length(get_coeff_data(field))
        real_type = real(eltype(get_coeff_data(field)))
        real_coeffs = get_workspace!(wm, real_type, (coeff_len,))
        imag_coeffs = eltype(get_coeff_data(field)) <: Complex ? get_workspace!(wm, real_type, (coeff_len,)) : nothing

        # Extract real part in-place
        @inbounds @simd for i in eachindex(real_coeffs, get_coeff_data(field))
            real_coeffs[i] = real(get_coeff_data(field)[i])
            if imag_coeffs !== nothing
                imag_coeffs[i] = imag(get_coeff_data(field)[i])
            end
        end

        # Ensure output array exists
        if get_grid_data(field) === nothing || length(get_grid_data(field)) != out_size || eltype(get_grid_data(field)) != field.dtype
            set_grid_data!(field, zeros(field.dtype, out_size))
        end

        if eltype(get_grid_data(field)) <: Real
            if imag_coeffs !== nothing && any(x -> !iszero(x), imag_coeffs)
                @warn "Discarding imaginary coefficients for real Chebyshev backward transform of $(field.name)"
            end
            mul!(get_grid_data(field), mat, real_coeffs)
        else
            out_real = get_workspace!(wm, real_type, (out_size,))
            out_imag = get_workspace!(wm, real_type, (out_size,))
            mul!(out_real, mat, real_coeffs)
            if imag_coeffs === nothing
                fill!(out_imag, zero(real_type))
            else
                mul!(out_imag, mat, imag_coeffs)
            end
            get_grid_data(field) .= complex.(out_real, out_imag)
            release_workspace!(wm, out_real)
            release_workspace!(wm, out_imag)
        end

        release_workspace!(wm, real_coeffs)
        if imag_coeffs !== nothing
            release_workspace!(wm, imag_coeffs)
        end
    else
        @warn "No backward matrix available for Chebyshev transform"
        if get_grid_data(field) === nothing
            set_grid_data!(field, real.(get_coeff_data(field)))
        else
            @inbounds @simd for i in eachindex(get_grid_data(field))
                get_grid_data(field)[i] = real(get_coeff_data(field)[i])
            end
        end
    end
end

# Legendre transform application functions following Tarang JacobiMMT patterns
