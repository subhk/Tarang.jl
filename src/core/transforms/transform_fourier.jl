"""
    Transform Fourier - Fourier transform execution

This file contains the forward and backward Fourier transform implementations.
"""

function _fourier_forward(data::AbstractArray, transform::FourierTransform)
    return _execute_on_cpu(data) do host_data
        # Use precomputed plan only if data size and element type match the plan
        # (FFTW plans are type-specific: a Float64 plan cannot be applied to Float32 data)
        if ndims(host_data) == 1 && transform.plan_forward !== nothing &&
           length(host_data) == transform.basis.meta.size &&
           eltype(host_data) === transform.plan_dtype
            return transform.plan_forward * host_data
        end

        dims = (transform.axis,)
        if isa(transform.basis, RealFourier)
            # rfft requires real input; if data is already complex (from a prior
            # rfft or fft along another axis in a multi-dimensional transform),
            # use fft instead. This is standard for multi-dimensional real FFTs:
            # rfft is applied along one axis, fft along the remaining axes.
            if eltype(host_data) <: Complex
                return FFTW.fft(host_data, dims)
            end
            return FFTW.rfft(host_data, dims)
        end

        return FFTW.fft(host_data, dims)
    end
end

"""Apply forward Fourier transform"""
function apply_fourier_forward!(field::ScalarField, transform::FourierTransform)
    set_coeff_data!(field, _fourier_forward(get_grid_data(field), transform))
end

"""Apply backward transform to field """
function backward_transform!(field::ScalarField, target_layout::Symbol=:g)

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :c)  # Start in coefficient space

    # Try GPU transform first if on GPU architecture
    if gpu_backward_transform!(field)
        field.current_layout = :g
        return
    end

    # Find appropriate transform
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # PencilFFTs is CPU-only; if data is on GPU, move to CPU first
        coeff_data = get_coeff_data(field)
        if is_gpu_array(coeff_data)
            host_data = Array(coeff_data)
            host_result = pencil_plan \ host_data
            set_grid_data!(field, copy_to_device(host_result, coeff_data))
        else
            set_grid_data!(field, pencil_plan \ coeff_data)
        end
        field.current_layout = :g
        return
    end

    # CRITICAL: Guard against running local transforms on distributed data
    # If we reach here with dist.size > 1, no PencilFFTPlan was found above,
    # so local transforms would produce incorrect results on distributed data.
    if field.dist.size > 1
        if is_gpu_array(get_coeff_data(field))
            error("Cannot run local GPU transforms on distributed data without TransposableField. " *
                  "GPU+MPI requires explicit transposes for correct distributed FFT. " *
                  "Use TransposableField with distributed_forward_transform!/distributed_backward_transform! " *
                  "for correct GPU+MPI spectral transforms.")
        else
            error("Cannot run local FFTW transforms on distributed CPU data without PencilFFTs. " *
                  "No PencilFFTPlan found for this domain. " *
                  "For MPI+CPU Fourier, set use_pencil_arrays=true in Distributor. " *
                  "For MPI+GPU, use TransposableField with distributed transforms.")
        end
    end

    current = get_coeff_data(field)
    for transform in reverse(field.dist.transforms)
        current = _apply_backward(current, transform)
    end

    # Fallback for other transforms
    if current === get_coeff_data(field)
        set_grid_data!(field, copy(get_coeff_data(field)))
    else
        set_grid_data!(field, current)
    end
    field.current_layout = :g
end

function _fourier_backward(data::AbstractArray, transform::FourierTransform)
    return _execute_on_cpu(data) do host_data
        # Use precomputed plan only if data size and element type match the plan
        # (FFTW plans are type-specific; backward plans expect Complex{plan_dtype} input)
        expected_rfft_size = isa(transform.basis, RealFourier) ? div(transform.basis.meta.size, 2) + 1 : transform.basis.meta.size
        if ndims(host_data) == 1 && transform.plan_backward !== nothing &&
           length(host_data) == expected_rfft_size &&
           eltype(host_data) === Complex{transform.plan_dtype}
            return transform.plan_backward * host_data
        end

        dims = (transform.axis,)
        if isa(transform.basis, RealFourier)
            # Check actual size along the transform axis to determine if rfft or fft was used
            actual_size = size(host_data, transform.axis)
            expected_rfft_coeff_size = div(transform.basis.meta.size, 2) + 1

            # If size matches rfft output, use irfft; otherwise use ifft (fft was used for complex input)
            if actual_size == expected_rfft_coeff_size
                return FFTW.irfft(host_data, transform.basis.meta.size, dims)
            else
                # fft was used (complex input case), use ifft
                return FFTW.ifft(host_data, dims)
            end
        end

        return FFTW.ifft(host_data, dims)
    end
end

"""Apply backward Fourier transform"""
function apply_fourier_backward!(field::ScalarField, transform::FourierTransform)
    set_grid_data!(field, _fourier_backward(get_coeff_data(field), transform))
end

# Dispatch methods for transform loop (replaces isa() chains)
_apply_forward(current, t::FourierTransform) = _fourier_forward(current, t)
_apply_backward(current, t::FourierTransform) = _fourier_backward(current, t)

# Axis-aware Chebyshev helpers
function _scale_along_axis!(data::AbstractArray, axis::Int, scale::AbstractVector{<:Real})
    if axis > ndims(data)
        return
    end
    shape = ntuple(i -> i == axis ? length(scale) : 1, ndims(data))
    data .*= reshape(scale, shape...)
end

