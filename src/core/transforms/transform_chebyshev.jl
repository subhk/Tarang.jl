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
        scaled_real = real.(host_data)  # already a new array, no copy needed
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
            scaled_imag = imag.(host_data)  # already a new array, no copy needed
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

_apply_forward(current, t::ChebyshevTransform) = _chebyshev_forward(current, t)
_apply_backward(current, t::ChebyshevTransform) = _chebyshev_backward(current, t)

# ============================================================================
# In-place Chebyshev transforms — zero-alloc once scratch cache is warm
# ============================================================================
#
# Chebyshev is trickier than Fourier because:
#
#   1. The DCT-I (REDFT00) only accepts real input. Complex data must be
#      split into real/imag halves, each transformed independently, then
#      recombined into a complex output array.
#
#   2. The forward transform may truncate along the transform axis
#      (coeff_size != grid_size), which means the DCT output and the
#      final output have different shapes. We need an intermediate
#      "full-size" scratch for the DCT result before copying/truncating
#      into the caller's `out` buffer.
#
#   3. The backward transform zero-pads along the transform axis, which
#      also requires a "full-size" scratch before the DCT-I call.
#
#   4. Endpoint half-weighting (the DCT-I-on-Gauss-Lobatto convention)
#      is applied in the frequency domain and needs separate handling
#      for real and imaginary parts.
#
# The ChebScratch struct (defined in transform_types.jl) holds all four
# buffers (real_in, imag_in, tmp_real, tmp_imag) plus the r2r plan. The
# scratch Dict on the transform is keyed by
# `(input_shape, input_eltype)` for forward, and
# `(input_shape, input_eltype)` for backward. Forward and backward have
# separate dicts because they use different-shaped intermediate buffers.

function _forward_output_spec(in::AbstractArray, transform::ChebyshevTransform)
    in_shape = size(in)
    ax = transform.axis
    if ax > ndims(in)
        return in_shape, eltype(in)
    end
    coeff_size = transform.coeff_size
    out_shape = ntuple(i -> i == ax ? coeff_size : in_shape[i], length(in_shape))
    return out_shape, eltype(in)
end

function _backward_output_spec(in::AbstractArray, transform::ChebyshevTransform)
    in_shape = size(in)
    ax = transform.axis
    if ax > ndims(in)
        return in_shape, eltype(in)
    end
    grid_size = transform.grid_size
    out_shape = ntuple(i -> i == ax ? grid_size : in_shape[i], length(in_shape))
    return out_shape, eltype(in)
end

"""
Get-or-allocate the forward scratch entry for this (input_shape, input_eltype).
The scratch includes real/imag split buffers sized to the INPUT shape (so the
DCT-I plan matches), and an r2r plan bound to `tmp_real`.
"""
function _get_or_alloc_cheb_forward_scratch!(transform::ChebyshevTransform,
                                             in_shape::Tuple, in_eltype::Type)
    key = (in_shape, in_eltype)
    cached = get(transform.fwd_scratch, key, nothing)
    if cached !== nothing
        return cached
    end

    ax = transform.axis
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    # DCT-I plan needs an array of the INPUT shape (the r2r happens
    # along `ax`, keeping shape unchanged).
    tmp_real = zeros(real_T, in_shape...)
    plan = FFTW.plan_r2r(tmp_real, FFTW.REDFT00, (ax,))
    real_in = zeros(real_T, in_shape...)
    imag_in = in_eltype <: Complex ? zeros(real_T, in_shape...) : nothing
    tmp_imag = in_eltype <: Complex ? zeros(real_T, in_shape...) : nothing
    scratch = ChebScratch(real_in, imag_in, tmp_real, tmp_imag, plan)
    transform.fwd_scratch[key] = scratch
    return scratch
end

"""
Get-or-allocate the backward scratch entry. The buffers must be sized
to the OUTPUT (grid) shape because the DCT-I is applied AFTER zero-padding
the coefficient input up to grid size.
"""
function _get_or_alloc_cheb_backward_scratch!(transform::ChebyshevTransform,
                                              in_shape::Tuple, in_eltype::Type)
    key = (in_shape, in_eltype)
    cached = get(transform.bwd_scratch, key, nothing)
    if cached !== nothing
        return cached
    end

    ax = transform.axis
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    grid_size = transform.grid_size
    padded_shape = ntuple(i -> i == ax ? grid_size : in_shape[i], length(in_shape))

    # Scratch sized to the padded (grid-size) shape for the DCT-I.
    tmp_real = zeros(real_T, padded_shape...)
    plan = FFTW.plan_r2r(tmp_real, FFTW.REDFT00, (ax,))
    # real_in / imag_in hold the zero-padded-and-endpoint-scaled
    # coefficients, sized to the grid shape.
    real_in = zeros(real_T, padded_shape...)
    imag_in = in_eltype <: Complex ? zeros(real_T, padded_shape...) : nothing
    tmp_imag = in_eltype <: Complex ? zeros(real_T, padded_shape...) : nothing
    scratch = ChebScratch(real_in, imag_in, tmp_real, tmp_imag, plan)
    transform.bwd_scratch[key] = scratch
    return scratch
end

"""
    _apply_forward!(out, in, transform::ChebyshevTransform) → out

Zero-allocation forward Chebyshev (DCT-I) transform.

Algorithm (matches `_chebyshev_forward` exactly):
  1. Copy `real.(in)` into `scratch.real_in` (and `imag.(in)` into
     `scratch.imag_in` if input is complex).
  2. Apply the r2r REDFT00 plan: `mul!(tmp_real, plan, real_in)`
     (and likewise for imag). Both scratches have the INPUT shape.
  3. Scale by `1/(N-1)` and halve the first and last coefficients
     along the transform axis (endpoint convention).
  4. Copy the leading `coeff_size` slab along the axis into `out`,
     recombining real + imag into complex if needed.

Step 4 is where the shape mismatch between tmp_real (input-shape) and
out (coeff_size-along-axis) is resolved.
"""
function _apply_forward!(out::AbstractArray, in::AbstractArray, transform::ChebyshevTransform)
    if is_gpu_array(in) || is_gpu_array(out)
        result = _chebyshev_forward(in, transform)
        if size(result) == size(out) && eltype(result) == eltype(out)
            copyto!(out, result)
        end
        return out
    end

    ax = transform.axis
    if ax > ndims(in)
        # No transform axis — just pass-through copy.
        if out !== in
            copyto!(out, in)
        end
        return out
    end

    in_shape = size(in)
    in_eltype = eltype(in)
    grid_size = in_shape[ax]
    coeff_size = transform.coeff_size

    scratch = _get_or_alloc_cheb_forward_scratch!(transform, in_shape, in_eltype)

    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    norm_factor = real_T(grid_size > 1 ? 1.0 / (grid_size - 1) : 1.0)
    half = real_T(0.5)

    # ── Real part ─────────────────────────────────────────────────
    if in_eltype <: Complex
        @inbounds @. scratch.real_in = real(in)
    else
        copyto!(scratch.real_in, in)
    end
    mul!(scratch.tmp_real, scratch.plan, scratch.real_in)
    scratch.tmp_real .*= norm_factor
    _scale_first_along_axis!(scratch.tmp_real, ax, half)
    _scale_last_along_axis!(scratch.tmp_real, ax, half)

    # ── Imaginary part (if complex) ───────────────────────────────
    if in_eltype <: Complex
        @inbounds @. scratch.imag_in = imag(in)
        mul!(scratch.tmp_imag, scratch.plan, scratch.imag_in)
        scratch.tmp_imag .*= norm_factor
        _scale_first_along_axis!(scratch.tmp_imag, ax, half)
        _scale_last_along_axis!(scratch.tmp_imag, ax, half)
    end

    # ── Copy leading `coeff_size` slab into `out` (truncation) ────
    ncopy = min(grid_size, coeff_size)
    src_idx = ntuple(i -> i == ax ? (1:ncopy) : Colon(), ndims(in))
    dst_idx = ntuple(i -> i == ax ? (1:ncopy) : Colon(), ndims(out))

    if in_eltype <: Complex
        @inbounds @views @. out[dst_idx...] = complex(scratch.tmp_real[src_idx...],
                                                      scratch.tmp_imag[src_idx...])
    else
        # Real-to-real path: `out` has the same real type.
        @inbounds @views copyto!(out[dst_idx...], scratch.tmp_real[src_idx...])
    end

    # Zero the trailing slab along the axis if coeff_size > grid_size
    # (zero-padding on the coefficient side — rare but legal).
    if coeff_size > grid_size
        pad_idx = ntuple(i -> i == ax ? ((grid_size + 1):coeff_size) : Colon(), ndims(out))
        @inbounds @views fill!(out[pad_idx...], zero(eltype(out)))
    end

    return out
end

"""
    _apply_backward!(out, in, transform::ChebyshevTransform) → out

Zero-allocation backward Chebyshev (DCT-I) transform.

Algorithm (matches `_chebyshev_backward` exactly):
  1. Copy `real.(in)` into `scratch.real_in`, zero-padded to grid size
     along the transform axis. Undo the endpoint halving from the
     forward pass (multiply first coefficient by 2; likewise the last
     coefficient IF it's the physical DCT-I endpoint, i.e.,
     `coeff_size == grid_size`).
  2. DCT-I is its own inverse up to normalization: apply the cached
     r2r REDFT00 plan, then divide by 2.
  3. Copy `scratch.tmp_real` (and imag) into `out`, recombining into
     complex if needed.
"""
function _apply_backward!(out::AbstractArray, in::AbstractArray, transform::ChebyshevTransform)
    if is_gpu_array(in) || is_gpu_array(out)
        result = _chebyshev_backward(in, transform)
        if size(result) == size(out) && eltype(result) == eltype(out)
            copyto!(out, result)
        end
        return out
    end

    ax = transform.axis
    if ax > ndims(in)
        if out !== in
            copyto!(out, in)
        end
        return out
    end

    in_shape = size(in)
    in_eltype = eltype(in)
    coeff_size = in_shape[ax]
    grid_size = transform.grid_size

    scratch = _get_or_alloc_cheb_backward_scratch!(transform, in_shape, in_eltype)
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    two = real_T(2.0)
    half = real_T(0.5)

    # ── Real part: copy, zero-pad, undo endpoint halving ──────────
    fill!(scratch.real_in, zero(real_T))
    ncopy = min(coeff_size, grid_size)
    src_idx = ntuple(i -> i == ax ? (1:ncopy) : Colon(), ndims(in))
    dst_idx = ntuple(i -> i == ax ? (1:ncopy) : Colon(), length(size(scratch.real_in)))

    if in_eltype <: Complex
        @inbounds @views @. scratch.real_in[dst_idx...] = real(in[src_idx...])
    else
        @inbounds @views copyto!(scratch.real_in[dst_idx...], in[src_idx...])
    end
    _scale_first_along_axis!(scratch.real_in, ax, two)
    # Only undo the last-endpoint halving if the stored last coeff IS the
    # physical last DCT-I mode (coeff_size == grid_size). When the
    # coefficient axis is truncated (dealiased Chebyshev), the last
    # retained coefficient was never halved in the forward pass, so we
    # must not double it here.
    if coeff_size > 1 && coeff_size == grid_size
        _scale_last_along_axis!(scratch.real_in, ax, two)
    end

    mul!(scratch.tmp_real, scratch.plan, scratch.real_in)
    scratch.tmp_real .*= half  # DCT-I is its own inverse up to factor 2

    # ── Imag part if complex ──────────────────────────────────────
    if in_eltype <: Complex
        fill!(scratch.imag_in, zero(real_T))
        @inbounds @views @. scratch.imag_in[dst_idx...] = imag(in[src_idx...])
        _scale_first_along_axis!(scratch.imag_in, ax, two)
        if coeff_size > 1 && coeff_size == grid_size
            _scale_last_along_axis!(scratch.imag_in, ax, two)
        end
        mul!(scratch.tmp_imag, scratch.plan, scratch.imag_in)
        scratch.tmp_imag .*= half
    end

    # ── Copy scratch (grid-shape) into out (grid-shape) ───────────
    # The scratch and the output have the SAME shape because backward
    # always produces the full grid_size along the transform axis.
    if in_eltype <: Complex
        @inbounds @. out = complex(scratch.tmp_real, scratch.tmp_imag)
    else
        copyto!(out, scratch.tmp_real)
    end

    return out
end
