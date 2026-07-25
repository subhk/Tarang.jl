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

All functions are pure, allocation-free, and return isbits values, so they are
safe on the zero-allocation transform hot path.
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

"""
    is_fourier_basis(basis) -> Bool
"""
@inline is_fourier_basis(basis) = isa(basis, RealFourier) || isa(basis, ComplexFourier)

"""
    first_fourier_axis(bases) -> Int

Axis index of the first Fourier basis, or `0` if there is none. Only this axis
can be halved: after any FFT the data is complex, so every later Fourier axis
sees complex input and runs a full-size C2C transform. Both the serial FFTW chain
(`_fourier_forward` dispatches on element type) and the MPI PencilFFTs plan
(RFFT only on the first Fourier axis) follow this rule, which is why coefficient
shapes agree in serial and distributed runs.
"""
@inline function first_fourier_axis(bases)
    for (i, b) in enumerate(bases)
        is_fourier_basis(b) && return i
    end
    return 0
end

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
