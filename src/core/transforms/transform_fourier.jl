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
            # In-place backward via `ldiv!` when the grid buffer is a
            # pre-allocated PencilArray (the common case — `allocate_data!`
            # uses PencilFFTs.allocate_input which returns one). Mirrors
            # the forward branch's `mul!(coeff_data, plan, grid_data)`
            # fast path. Falls back to the allocating `\` if ldiv! raises
            # (e.g. on PencilFFTs versions that don't support it on this
            # plan type). The fallback is logged once so an MPI run will
            # make it obvious if the fast path isn't firing.
            grid_data = get_grid_data(field)
            if grid_data !== nothing && isa(grid_data, PencilArrays.PencilArray)
                try
                    ldiv!(grid_data, pencil_plan, coeff_data)
                catch err
                    @warn "backward_transform!: PencilFFTs ldiv! fast path failed, falling back to allocating `\\`. This costs one PencilArray per transform — consider upgrading PencilFFTs or checking the field buffer layout." exception=(err, catch_backtrace()) maxlog=1
                    set_grid_data!(field, pencil_plan \ coeff_data)
                end
            else
                @warn "backward_transform!: grid buffer is not a PencilArray; using allocating `\\` fallback. Expected `allocate_data!` to pre-allocate via PencilFFTs.allocate_input." maxlog=1
                set_grid_data!(field, pencil_plan \ coeff_data)
            end
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

    # ── Zero-allocation in-place backward transform chain ─────────────────
    # Mirror of the forward path in transform_gpu.jl. Walk transforms in
    # reverse order. For intermediate stages, write into a cached scratch;
    # for the FINAL stage, write directly into the field's pre-allocated
    # grid buffer.
    #
    # We iterate via index arithmetic (`n_transforms:-1:1`) rather than
    # `collect(reverse(...))` to avoid allocating a fresh reversed vector
    # each call.
    current = get_coeff_data(field)
    transforms = field.dist.transforms
    n_transforms = length(transforms)
    if n_transforms == 0
        grid = get_grid_data(field)
        if grid !== nothing && size(grid) == size(current) && eltype(grid) == eltype(current)
            copyto!(grid, current)
        else
            set_grid_data!(field, copy(current))
        end
        field.current_layout = :g
        return
    end

    # Walk transforms in reverse order. `step` counts 1..n_transforms so
    # we can detect the final step (step == n_transforms) to write directly
    # into the field's pre-allocated grid buffer.
    for step in 1:n_transforms
        transform = transforms[n_transforms - step + 1]
        out_shape, out_eltype = _backward_output_spec(current, transform)
        if step == n_transforms
            grid = get_grid_data(field)
            if grid === nothing || size(grid) != out_shape || eltype(grid) != out_eltype
                grid = zeros(out_eltype, out_shape...)
                set_grid_data!(field, grid)
            end
            _apply_backward!(grid, current, transform)
            current = grid
        else
            out = _get_scratch_for_transform!(transform, :bwd_inter, out_shape, out_eltype)
            _apply_backward!(out, current, transform)
            current = out
        end
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

# Dispatch methods for transform loop (replaces isa() chains)
_apply_forward(current, t::FourierTransform) = _fourier_forward(current, t)
_apply_backward(current, t::FourierTransform) = _fourier_backward(current, t)

# ============================================================================
# In-place Fourier transforms — zero-alloc once plan+scratch caches warm
# ============================================================================
#
# The original out-of-place paths above (`_fourier_forward` /
# `_fourier_backward`) are kept for rare callers and as a fallback. The
# in-place entry points below are used by `forward_transform!` /
# `backward_transform!` on the hot path.
#
# Per-call work for a warm cache:
#   1. Dict lookup on `transform.fwd_plan_cache[(size, eltype)]` → plan
#   2. `mul!(out, plan, in)` — pure FFTW call, no allocations
#
# Separate plan caches are needed because FFTW plans are parameterized
# by input size, element type, AND strides. A plan built for a (192, 48)
# Float64 array cannot be applied to a (192, 32) array or to a Float32
# array. The `(size, eltype)` key captures the relevant dimensions.
#
# For RealFourier, the forward transform uses `plan_rfft` when the input
# is real, but a full complex `plan_fft` when the input is already
# complex (which happens when Fourier is NOT the first transform in the
# chain and a previous transform produced complex output). These two
# plan types have different output shapes and are cached under separate
# keys (`(size, Float64)` vs `(size, ComplexF64)`).

function _forward_output_spec(in::AbstractArray, transform::FourierTransform)
    in_shape = size(in)
    in_eltype = eltype(in)
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    complex_T = Complex{real_T}
    ax = transform.axis

    if isa(transform.basis, RealFourier)
        if in_eltype <: Complex
            # Already-complex input: full fft, shape unchanged
            return in_shape, complex_T
        end
        # Real input: rfft halves the transform axis from N to div(N, 2) + 1
        out_shape = ntuple(i -> i == ax ? div(in_shape[i], 2) + 1 : in_shape[i],
                           length(in_shape))
        return out_shape, complex_T
    else  # ComplexFourier
        return in_shape, complex_T
    end
end

function _backward_output_spec(in::AbstractArray, transform::FourierTransform)
    in_shape = size(in)
    in_eltype = eltype(in)
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    ax = transform.axis

    if isa(transform.basis, RealFourier)
        expected_rfft_size = div(transform.basis.meta.size, 2) + 1
        axis_len = in_shape[ax]
        if axis_len == expected_rfft_size
            # irfft: axis expands back from N/2+1 to basis.meta.size
            out_shape = ntuple(i -> i == ax ? transform.basis.meta.size : in_shape[i],
                               length(in_shape))
            return out_shape, real_T
        else
            # ifft fallback (complex input from a non-first-axis fft): same shape, still complex
            return in_shape, Complex{real_T}
        end
    else  # ComplexFourier
        return in_shape, Complex{real_T}
    end
end

"""Get or create a cached forward plan for this (input_size, input_eltype)."""
function _get_or_plan_forward!(transform::FourierTransform, in::AbstractArray)
    in_shape = size(in)
    in_eltype = eltype(in)
    key = (in_shape, in_eltype)
    cached = get(transform.fwd_plan_cache, key, nothing)
    if cached !== nothing
        return cached
    end

    dims = (transform.axis,)
    plan = if isa(transform.basis, RealFourier) && !(in_eltype <: Complex)
        FFTW.plan_rfft(in, dims)
    else
        FFTW.plan_fft(in, dims)
    end
    transform.fwd_plan_cache[key] = plan
    return plan
end

"""Get or create a cached backward plan for this (input_size, input_eltype)."""
function _get_or_plan_backward!(transform::FourierTransform, in::AbstractArray)
    in_shape = size(in)
    in_eltype = eltype(in)
    key = (in_shape, in_eltype)
    cached = get(transform.bwd_plan_cache, key, nothing)
    if cached !== nothing
        return cached
    end

    dims = (transform.axis,)
    plan = if isa(transform.basis, RealFourier)
        expected_rfft_size = div(transform.basis.meta.size, 2) + 1
        axis_len = in_shape[transform.axis]
        if axis_len == expected_rfft_size
            # irfft path: plan_irfft needs a dummy input of the same shape
            FFTW.plan_irfft(in, transform.basis.meta.size, dims)
        else
            FFTW.plan_ifft(in, dims)
        end
    else  # ComplexFourier
        FFTW.plan_ifft(in, dims)
    end
    transform.bwd_plan_cache[key] = plan
    return plan
end

"""
    _apply_forward!(out, in, transform::FourierTransform) → out

Zero-allocation forward Fourier transform into pre-allocated `out`.
Caller must ensure `out` has the shape/eltype returned by
`_forward_output_spec(in, transform)`.
"""
function _apply_forward!(out::AbstractArray, in::AbstractArray, transform::FourierTransform)
    # Only FFTW (CPU) arrays are handled here; GPU data is caught earlier.
    if is_gpu_array(in) || is_gpu_array(out)
        # Shouldn't happen on the new hot path — forward_transform! has
        # already branched to the GPU path by now — but handle defensively.
        result = _fourier_forward(in, transform)
        if size(result) == size(out) && eltype(result) == eltype(out)
            copyto!(out, result)
        end
        return out
    end

    plan = _get_or_plan_forward!(transform, in)
    mul!(out, plan, in)
    return out
end

"""
    _apply_backward!(out, in, transform::FourierTransform) → out

Zero-allocation backward Fourier transform into pre-allocated `out`.
"""
function _apply_backward!(out::AbstractArray, in::AbstractArray, transform::FourierTransform)
    if is_gpu_array(in) || is_gpu_array(out)
        result = _fourier_backward(in, transform)
        if size(result) == size(out) && eltype(result) == eltype(out)
            copyto!(out, result)
        end
        return out
    end

    plan = _get_or_plan_backward!(transform, in)

    # FFTW irfft plans are destructive on the input array. The high-level
    # `FFTW.irfft(x, n, dims)` protects callers by internally copying `x`
    # into a workspace — exactly the allocation we're trying to avoid.
    # Use a cached scratch that's reused across calls. Only irfft needs
    # this; ifft (complex → complex) is non-destructive.
    is_irfft_path = false
    if isa(transform.basis, RealFourier)
        expected_rfft_size = div(transform.basis.meta.size, 2) + 1
        if size(in, transform.axis) == expected_rfft_size
            is_irfft_path = true
        end
    end

    if is_irfft_path
        scratch_key = (size(in), eltype(in), :irfft_scratch)
        scratch = _get_or_alloc_scratch!(transform.bwd_scratch, scratch_key,
                                         size(in), eltype(in))
        copyto!(scratch, in)
        mul!(out, plan, scratch)
    else
        mul!(out, plan, in)
    end
    return out
end

# Axis-aware Chebyshev helpers
function _scale_along_axis!(data::AbstractArray, axis::Int, scale::AbstractVector{<:Real})
    if axis > ndims(data)
        return
    end
    shape = ntuple(i -> i == axis ? length(scale) : 1, ndims(data))
    data .*= reshape(scale, shape...)
end

