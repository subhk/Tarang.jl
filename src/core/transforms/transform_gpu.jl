"""
    Transform GPU - GPU transform support and heuristics

This file contains GPU-specific transform support including
FFT heuristics and CPU fallback execution.
"""

# ============================================================================
# GPU Transform Support
# ============================================================================

# Note: is_gpu_array is defined in architectures.jl

# ---------------------------------------------------------------------------
# GPU transform heuristics
# ---------------------------------------------------------------------------

const GPU_FFT_MIN_ELEMENTS = Ref(32_768)

"""
    set_gpu_fft_min_elements!(n::Integer)

Set the minimum number of elements required before GPU FFTs are attempted.
"""
function set_gpu_fft_min_elements!(n::Integer)
    GPU_FFT_MIN_ELEMENTS[] = max(1, Int(n))
    return GPU_FFT_MIN_ELEMENTS[]
end

gpu_fft_min_elements() = GPU_FFT_MIN_ELEMENTS[]

function should_use_gpu_fft(field::ScalarField, data_shape::Tuple)
    mode = gpu_fft_mode(field)
    if mode === :gpu
        return true
    elseif mode === :cpu
        return false
    end
    return prod(data_shape) >= GPU_FFT_MIN_ELEMENTS[]
end

should_use_gpu_fft(field::ScalarField) = (get_grid_data(field) !== nothing) && should_use_gpu_fft(field, size(get_grid_data(field)))

"""
    gpu_forward_transform!(field::ScalarField)

GPU-specific forward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_forward_transform!(field::ScalarField)
    # Check if we're on GPU architecture
    arch = field.dist.architecture
    if !is_gpu(arch)
        return false
    end

    # Check if data is on GPU
    data_g = get_grid_data(field)
    if !is_gpu_array(data_g)
        return false
    end

    # GPU transform will be dispatched via extension
    # The extension overrides this function when CUDA is loaded
    @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
    return false
end

"""
    gpu_backward_transform!(field::ScalarField)

GPU-specific backward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_backward_transform!(field::ScalarField)
    # Check if we're on GPU architecture
    arch = field.dist.architecture
    if !is_gpu(arch)
        return false
    end

    # Check if data is on GPU
    data_c = get_coeff_data(field)
    if !is_gpu_array(data_c)
        return false
    end

    # GPU transform will be dispatched via extension
    @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
    return false
end

# -----------------------------------------------------------------------------
# Helper utilities for GPU fallbacks
# -----------------------------------------------------------------------------

"""
    _execute_on_cpu(f, data)

Ensure `f` runs on CPU memory even when `data` lives on a GPU.
Returns the result on the original device (GPU or CPU).

Note: f is the first argument to support do-block syntax:
    _execute_on_cpu(data) do host_data
        ...
    end
"""
function _execute_on_cpu(f::Function, data::AbstractArray)
    if is_gpu_array(data)
        host_data = Array(data)
        host_result = f(host_data)
        return copy_to_device(host_result, data)
    end
    return f(data)
end

# Transform execution functions
function forward_transform!(field::ScalarField, target_layout::Symbol=:c)
    """Apply forward transform to field"""

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :g)  # Start in grid space

    # Try GPU transform first if on GPU architecture
    if gpu_forward_transform!(field)
        field.current_layout = :c
        return
    end

    # Find appropriate transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # PencilFFTs is CPU-only; if data is on GPU, move to CPU first
            grid_data = get_grid_data(field)
            if is_gpu_array(grid_data)
                host_data = Array(grid_data)
                host_result = transform * host_data
                set_coeff_data!(field, copy_to_device(host_result, grid_data))
            else
                # Use in-place mul! if coeff data is already allocated
                coeff_data = get_coeff_data(field)
                if coeff_data !== nothing && isa(coeff_data, PencilArrays.PencilArray)
                    try
                        mul!(coeff_data, transform, grid_data)
                    catch
                        set_coeff_data!(field, transform * grid_data)
                    end
                else
                    set_coeff_data!(field, transform * grid_data)
                end
            end
            field.current_layout = :c
            return
        end
    end

    # CRITICAL: Guard against running local transforms on distributed data
    # If we reach here with dist.size > 1, no PencilFFTPlan was found above,
    # so local transforms would produce incorrect results on distributed data.
    if field.dist.size > 1
        if is_gpu_array(get_grid_data(field))
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

    current = get_grid_data(field)
    for transform in field.dist.transforms
        if isa(transform, FourierTransform)
            current = _fourier_forward(current, transform)
        elseif isa(transform, ChebyshevTransform)
            current = _chebyshev_forward(current, transform)
        elseif isa(transform, LegendreTransform)
            current = _legendre_forward(current, transform)
        end
    end

    # Fallback for other transforms or missing plans
    if current === get_grid_data(field)
        set_coeff_data!(field, copy(get_grid_data(field)))
    else
        set_coeff_data!(field, current)
    end
    field.current_layout = :c
end

