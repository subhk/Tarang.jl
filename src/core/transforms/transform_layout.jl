"""
    Transform layout — the single source of truth for spectral array layout.

Every transform backend needs the same four answers for each axis: which
operation runs, how long the axis becomes, whether the data turns complex, and
(backward) whether the stored spectrum must be zero-padded first. Those rules
used to be written out separately in `domain.jl` (`coefficient_shape`), in the
CPU stage specs (`transform_fourier.jl` / `transform_chebyshev.jl`), and again in
the CUDA extension (`plan_gpu_mixed_transform` plus the branch chains in
`_gpu_{forward,backward}_transform_impl!`). Three hand-maintained copies of one
rule set is where the layout bugs came from:

  * the GPU mixed plan derived its coefficient shape from the GRID shape, so a
    scaled Chebyshev axis silently kept every grid mode while the CPU chain
    truncated to the basis size;
  * a `ComplexFourier` first axis with a real dtype was handled by the GPU
    (promote to complex) and by the out-of-place CPU helper (`FFTW.fft`), but not
    by the in-place CPU stage, which had no rule for it at all.

The functions here state each rule once. Callers ask for an `AxisOp` instead of
re-deriving `div(N, 2) + 1` — that expression appears at 60+ sites across the
package, and every one of them is a place the copies can drift apart.

The per-axis layout rules are pure, allocation-free, and return isbits values,
so they are safe on the zero-allocation transform hot path. Plan-construction
helpers below may allocate metadata outside that hot path.
"""

"""
    AxisOp

What one transform stage does to one axis.

- `op` — `:rfft`, `:fft`, `:irfft`, `:irfft_upsampled`, `:ifft`, `:dct1_forward`,
  `:dct1_backward`, or `:none`
- `out_len` — the axis length AFTER the stage
- `out_complex` — whether the stage's output is complex
- `pad_len` — `:irfft_upsampled` only: the half-spectrum length to zero-pad to
  before the inverse (`0` otherwise)
"""
struct AxisOp
    op::Symbol
    out_len::Int
    out_complex::Bool
    pad_len::Int
end

AxisOp(op::Symbol, out_len::Int, out_complex::Bool) = AxisOp(op, out_len, out_complex, 0)

"""
    rfft_len(n) -> Int

Length of the non-redundant half-spectrum of a real signal of length `n`. THE
definition of the rfft layout rule; prefer it over writing `div(n, 2) + 1`.
"""
@inline rfft_len(n::Integer) = div(n, 2) + 1

# `is_fourier_basis` is defined once, in basis/basis_core.jl. A second definition
# lived here and disagreed with it about an absent axis; see the note there.
#
# `first_fourier_axis(bases)` also lived here — "axis index of the first Fourier
# basis, or 0". It had no callers: transform_planning.jl computes a LOCAL variable
# of the same name from its own `fourier_axes` list, which reads like a call and is
# not one. The rule it documented is still true and still enforced there — only
# this axis can be halved, because after any FFT the data is complex, so every
# later Fourier axis runs a full-size C2C transform (serial FFTW dispatches on
# element type; the MPI PencilFFTs plan does RFFT only on the first Fourier axis),
# which is why coefficient shapes agree in serial and distributed runs.

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

"""
    forward_axis_op(basis, in_len, in_complex) -> AxisOp

The forward stage for one axis, given the CURRENT axis length and whether the
data arriving at this stage is already complex.

`RealFourier` halves the axis only when it still sees real data — that is what
makes "only the first Fourier axis is halved" fall out rather than being a
special case. `ChebyshevT` maps the (possibly scaled) grid length to the basis
size: the DCT-I runs on the full grid and the result is truncated to
`basis.meta.size` coefficients, which is exactly the step the GPU mixed driver
was missing for scaled fields.
"""
@inline function forward_axis_op(basis, in_len::Int, in_complex::Bool)
    if isa(basis, RealFourier)
        return in_complex ? AxisOp(:fft, in_len, true) :
                            AxisOp(:rfft, rfft_len(in_len), true)
    elseif isa(basis, ComplexFourier)
        # Real input is promoted to complex first (FFTW's in-place `mul!` cannot
        # do that itself; the allocating `FFTW.fft` and cuFFT both can).
        return AxisOp(:fft, in_len, true)
    elseif isa(basis, ChebyshevT)
        return AxisOp(:dct1_forward, basis.meta.size, in_complex)
    else
        return AxisOp(:none, in_len, in_complex)
    end
end

"""
    forward_layout(bases, grid_shape, dtype) -> (ops, coeff_shape, coeff_complex)

Walk `bases` in axis order and return the per-axis `AxisOp`s, the resulting
coefficient shape, and whether the coefficients are complex. `grid_shape` is the
LOCAL, possibly scaled grid — pass `basis.meta.size` per axis for the canonical
(unscaled) layout.
"""
function forward_layout(bases, grid_shape::Tuple, dtype::Type)
    n = length(bases)
    ops = Vector{AxisOp}(undef, n)
    shape = collect(Int, grid_shape)
    complex_now = dtype <: Complex
    for ax in 1:n
        op = forward_axis_op(bases[ax], shape[ax], complex_now)
        ops[ax] = op
        shape[ax] = op.out_len
        complex_now = op.out_complex
    end
    return ops, tuple(shape...), complex_now
end

"""
    transform_stage_shapes(ops, input_shape, order) -> Vector{Tuple}

Return the input shape followed by the shape after each axis operation in
`order`. This lets a backend execute the shared `AxisOp` rules in a different
order (for example, Fourier axes before Chebyshev axes on CUDA) without
re-deriving intermediate buffer sizes.

`order` must be a permutation of every axis exactly once. The helper runs only
while constructing a transform plan, so clarity is preferred over hot-path
allocation constraints.
"""
function transform_stage_shapes(ops, input_shape::Tuple, order)
    n = length(ops)
    length(input_shape) == n || throw(DimensionMismatch(
        "received $n axis operations for an input shape with " *
        "$(length(input_shape)) dimensions"))

    axes_order = collect(Int, order)
    if length(axes_order) != n || sort(axes_order) != collect(1:n)
        throw(ArgumentError(
            "transform order must contain every axis 1:$n exactly once, got " *
            "$(Tuple(axes_order))"))
    end

    shape = collect(Int, input_shape)
    stages = Tuple[Tuple(shape)]
    for axis in axes_order
        shape[axis] = ops[axis].out_len
        push!(stages, Tuple(shape))
    end
    return stages
end

function _validate_axis_prefix_shapes(dst::AbstractArray, src::AbstractArray,
                                      axis::Int, operation::Symbol)
    ndims(dst) == ndims(src) || throw(DimensionMismatch(
        "$operation requires arrays with the same dimensionality, got " *
        "$(ndims(dst))D and $(ndims(src))D"))
    1 <= axis <= ndims(src) || throw(ArgumentError(
        "resize axis must be between 1 and $(ndims(src)), got $axis"))
    for dim in 1:ndims(src)
        dim == axis && continue
        size(dst, dim) == size(src, dim) || throw(DimensionMismatch(
            "$operation may change only axis $axis, but axis $dim has sizes " *
            "$(size(dst, dim)) and $(size(src, dim))"))
    end
    return nothing
end

"""Copy the leading part of `src` along `axis` into the shorter `dst`."""
function _copy_axis_prefix!(dst::AbstractArray, src::AbstractArray, axis::Int)
    _validate_axis_prefix_shapes(dst, src, axis, :truncate)
    size(dst, axis) <= size(src, axis) || throw(ArgumentError(
        "truncate destination axis $axis has length $(size(dst, axis)), larger than " *
        "source length $(size(src, axis))"))
    indices = ntuple(dim -> dim == axis ? (1:size(dst, axis)) : Colon(), ndims(src))
    @views dst .= src[indices...]
    return dst
end

"""Zero `dst`, then copy `src` into its leading entries along `axis`."""
function _zero_pad_axis_prefix!(dst::AbstractArray, src::AbstractArray, axis::Int)
    _validate_axis_prefix_shapes(dst, src, axis, :zero_pad)
    size(dst, axis) >= size(src, axis) || throw(ArgumentError(
        "zero-pad destination axis $axis has length $(size(dst, axis)), smaller than " *
        "source length $(size(src, axis))"))
    fill!(dst, zero(eltype(dst)))
    indices = ntuple(dim -> dim == axis ? (1:size(src, axis)) : Colon(), ndims(dst))
    @views dst[indices...] .= src
    return dst
end

"""
    layout_coefficient_shape(bases, dtype) -> Tuple

Canonical (unscaled) coefficient shape implied by `bases` and `dtype`. This is
the rule `Domain`'s `coefficient_shape` reports and every buffer is sized from.
"""
function layout_coefficient_shape(bases, dtype::Type)
    grid = ntuple(i -> bases[i].meta.size, length(bases))
    _, coeff_shape, _ = forward_layout(bases, grid, dtype)
    return coeff_shape
end

# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

"""
    backward_axis_op(basis, in_len, grid_len, in_complex) -> AxisOp

The inverse stage for one axis: `in_len` is the stored spectral length and
`grid_len` the target (possibly scaled) grid length.

The `RealFourier` classification order matters and is shared by both backends:
test the direct half-spectrum FIRST (a pure shape heuristic misreads `N = 1` and
`N = 2`, where `rfft_len(N) == N`), then the UPSAMPLED half-spectrum of a scaled
field, and only then fall through to a same-shape complex inverse. Guessing wrong
here stores a wrong-length or wrong-typed array instead of raising.
"""
@inline function backward_axis_op(basis, in_len::Int, grid_len::Int, in_complex::Bool)
    if isa(basis, RealFourier)
        base_len = basis.meta.size
        if in_len == rfft_len(grid_len)
            return AxisOp(:irfft, grid_len, false)
        elseif grid_len > base_len && in_len == rfft_len(base_len)
            # Scaled field: pad the base half-spectrum out to the finer grid.
            return AxisOp(:irfft_upsampled, grid_len, false, rfft_len(grid_len))
        else
            # Complex spectrum at full length (e.g. the axis was transformed C2C
            # because it was not the first Fourier axis).
            return AxisOp(:ifft, in_len, true)
        end
    elseif isa(basis, ComplexFourier)
        return AxisOp(:ifft, in_len, true)
    elseif isa(basis, ChebyshevT)
        # Zero-pads (or truncates) the coefficient axis to the grid length.
        return AxisOp(:dct1_backward, grid_len, in_complex)
    else
        return AxisOp(:none, in_len, in_complex)
    end
end

"""
    scaled_chebyshev_axis(bases, grid_shape) -> Union{Nothing, Tuple{Int,Int,Int}}

First Chebyshev axis whose grid length differs from its basis size, as
`(axis, grid_len, basis_len)`, or `nothing`. A backend without a Chebyshev
truncation/zero-pad stage must refuse such a field rather than return the
untruncated spectrum — see `forward_axis_op`.
"""
function scaled_chebyshev_axis(bases, grid_shape::Tuple)
    for (ax, b) in enumerate(bases)
        if isa(b, ChebyshevT) && ax <= length(grid_shape) && grid_shape[ax] != b.meta.size
            return (ax, grid_shape[ax], b.meta.size)
        end
    end
    return nothing
end
