"""
    Transform Types - Core type definitions for spectral transforms

This file contains the abstract Transform type and concrete transform structs.
"""

abstract type Transform end

# PencilFFTs-based transforms for parallel 2D FFTs
struct PencilFFTTransform <: Transform
    plan::Union{Nothing, PencilFFTs.PencilFFTPlan}
    basis::Basis

    function PencilFFTTransform(basis::Basis)
        new(nothing, basis)
    end
end

mutable struct FourierTransform <: Transform
    plan_forward::Union{Nothing, AbstractFFTPlan}
    plan_backward::Union{Nothing, AbstractFFTPlan}
    basis::Basis
    axis::Int
    plan_dtype::Type{<:AbstractFloat}  # Real element type used for plan creation (e.g., Float64, Float32)

    # ── Per-shape plan / scratch caches for zero-alloc in-place transforms ──
    #
    # The original `plan_forward`/`plan_backward` fields above hold 1D plans
    # created at solver setup time. They only fire for `ndims(data) == 1`
    # inputs (tau fields), so 2D state fields used to fall through to
    # `FFTW.rfft(data, dims)` which allocates a fresh array per call.
    #
    # `fwd_plan_cache` and `bwd_plan_cache` cache multi-dim plans keyed by
    # `(size, eltype)` of the input array, so subsequent calls at the same
    # shape reuse the cached plan via `mul!`. Separate caches for forward
    # and backward because they have different plan types and dimensions.
    #
    # `fwd_scratch` / `bwd_scratch` cache intermediate output buffers keyed
    # by `(size, eltype)` of the output array — used when a transform is
    # in the middle of the chain and its output feeds the next transform.
    # The final transform writes directly to the field's pre-allocated
    # coeff/grid buffer, not into this scratch.
    fwd_plan_cache::Dict{Tuple, Any}
    bwd_plan_cache::Dict{Tuple, Any}
    fwd_scratch::Dict{Tuple, AbstractArray}
    bwd_scratch::Dict{Tuple, AbstractArray}

    function FourierTransform(basis::Basis, axis::Int)
        new(nothing, nothing, basis, axis, Float64,
            Dict{Tuple, Any}(), Dict{Tuple, Any}(),
            Dict{Tuple, AbstractArray}(), Dict{Tuple, AbstractArray}())
    end
end

"""
    ChebScratch

Pre-allocated scratch buffers for a single in-place Chebyshev transform
at a given input shape / eltype. Cached on `ChebyshevTransform.fwd_scratch`
/ `bwd_scratch` keyed by `(input_shape, input_eltype, coeff_size)` so that
each distinct call shape pays the allocation cost exactly once.

Fields:
- `real_in` / `imag_in` — shape equal to the transform INPUT (real part and,
  for complex input, imaginary part split out for the DCT-I which requires
  real data).
- `tmp_real` / `tmp_imag` — shape equal to the DCT-I OUTPUT before truncation.
  For forward, same shape as input; for backward, equal to the zero-padded
  grid shape along the transform axis.
- `plan` — FFTW r2r REDFT00 plan. Created with `plan_r2r(tmp_real, REDFT00,
  (axis,))` and reused via `mul!`.
"""
mutable struct ChebScratch{T<:AbstractFloat, N}
    real_in::Array{T, N}
    imag_in::Union{Nothing, Array{T, N}}
    tmp_real::Array{T, N}
    tmp_imag::Union{Nothing, Array{T, N}}
    plan::Any
end

mutable struct ChebyshevTransform <: Transform
    forward_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    backward_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    basis::ChebyshevT

    # FFTW DCT plans
    forward_plan::Union{Nothing, AbstractFFTPlan}
    backward_plan::Union{Nothing, AbstractFFTPlan}

    # Scaling factors for FastCosineTransform
    forward_rescale_zero::Float64
    forward_rescale_pos::Float64
    backward_rescale_zero::Float64
    backward_rescale_pos::Float64

    # Size information for padding/truncation
    grid_size::Int
    coeff_size::Int
    Kmax::Int
    axis::Int

    # ── Per-shape scratch caches for zero-alloc in-place transforms ────────
    # Keyed by `(input_shape, input_eltype)`. Each entry holds a ChebScratch
    # with pre-allocated real/imag split buffers, r2r scratch, and plan.
    # See ChebScratch above for details.
    fwd_scratch::Dict{Tuple, ChebScratch}
    bwd_scratch::Dict{Tuple, ChebScratch}

    function ChebyshevTransform(basis::ChebyshevT)
        new(
            nothing, nothing,      # forward/backward matrices
            basis,
            nothing, nothing,      # FFTW plans
            0.0, 0.0, 0.0, 0.0,    # Scaling factors
            0, 0, 0, 0,            # Sizes and axis
            Dict{Tuple, ChebScratch}(),
            Dict{Tuple, ChebScratch}(),
        )
    end
end

mutable struct LegendreTransform <: Transform
    forward_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    backward_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    basis::Legendre

    # Quadrature information
    grid_points::Union{Nothing, Vector{Float64}}
    quad_weights::Union{Nothing, Vector{Float64}}

    # Size information for dealiasing
    grid_size::Int
    coeff_size::Int
    axis::Int

    function LegendreTransform(basis::Legendre)
        new(
            nothing, nothing,      # forward/backward matrices
            basis,
            nothing, nothing,      # Quadrature points and weights
            0, 0, 0                # Sizes and axis
        )
    end
end

# ---------------------------------------------------------------------------
# Dispatch helpers — replace isa() chains in transform loops
# Concrete methods for _apply_forward/_apply_backward are defined in
# transform_fourier.jl, transform_chebyshev.jl, transform_legendre.jl
# after the _fourier_forward etc. functions exist.
# ---------------------------------------------------------------------------

"""
    _apply_forward(current, transform) → transformed data

Apply a single forward transform to `current` data. Dispatch on transform type
replaces isa() chains in transform loops.
"""
function _apply_forward end

"""
    _apply_backward(current, transform) → transformed data

Apply a single backward (inverse) transform to `current` data.
"""
function _apply_backward end

# Fallbacks: skip unknown transform types (e.g., PencilFFTTransform handled separately)
_apply_forward(current, ::Transform) = current
_apply_backward(current, ::Transform) = current

# ---------------------------------------------------------------------------
# In-place transform protocol
# ---------------------------------------------------------------------------
#
# The in-place variants (`_apply_forward!` / `_apply_backward!`) write the
# transformed result into a caller-provided output buffer of the correct
# shape and eltype. They use cached plans and scratch buffers on the
# transform object so steady-state calls allocate zero bytes.
#
# Shape-spec helpers (`_forward_output_spec` / `_backward_output_spec`)
# tell `forward_transform!` / `backward_transform!` the expected output
# shape and eltype for a given input, so the caller can pick (or allocate)
# the correct buffer BEFORE the transform runs. For the final transform
# in a chain, the caller passes `field.coeff_data` / `field.grid_data`
# directly; for intermediate transforms, the caller uses a cached scratch
# buffer on the transform itself.

"""
    _apply_forward!(out, in, transform) → out

Apply a single forward transform writing into pre-allocated `out`.
Concrete methods must not allocate after their plan/scratch cache is warm.
"""
function _apply_forward! end

"""
    _apply_backward!(out, in, transform) → out

Apply a single backward transform writing into pre-allocated `out`.
"""
function _apply_backward! end

"""
    _forward_output_spec(in, transform) → (out_shape::Tuple, out_eltype::DataType)

Compute the output shape and eltype of a forward transform given an input
array. Used by the top-level transform loop to size the output buffer
before calling `_apply_forward!`.
"""
function _forward_output_spec end

"""
    _backward_output_spec(in, transform) → (out_shape::Tuple, out_eltype::DataType)

Backward-transform counterpart of `_forward_output_spec`.
"""
function _backward_output_spec end

# Fallbacks: transforms that don't implement the in-place protocol
# (Legendre, stacked) pass data through unchanged.
_apply_forward!(out, in, ::Transform) = (out === in ? out : copyto!(out, in))
_apply_backward!(out, in, ::Transform) = (out === in ? out : copyto!(out, in))
_forward_output_spec(in, ::Transform) = (size(in), eltype(in))
_backward_output_spec(in, ::Transform) = (size(in), eltype(in))

# ---------------------------------------------------------------------------
# Scratch buffer helper — reused across transform types
# ---------------------------------------------------------------------------

"""
    _get_or_alloc_scratch!(cache::Dict, key::Tuple, shape::Tuple, T::Type) → AbstractArray

Look up a pre-allocated array in `cache[key]`. If the entry is missing or
has the wrong shape/eltype, allocate a new `zeros(T, shape...)` and store
it. Subsequent calls with the same key return the cached array without
allocation.
"""
@inline function _get_or_alloc_scratch!(cache::Dict, key::Tuple, shape::Tuple, ::Type{T}) where {T}
    buf = get(cache, key, nothing)
    if buf !== nothing && size(buf) == shape && eltype(buf) === T
        return buf::AbstractArray{T}
    end
    new_buf = zeros(T, shape...)
    cache[key] = new_buf
    return new_buf
end

"""
    _find_pencil_plan(dist) → Union{Nothing, PencilFFTs.PencilFFTPlan}

Find cached PencilFFT plan from distributor transforms. Avoids repeated
linear scans of dist.transforms in hot paths.
"""
_find_pencil_plan(dist) = dist.pencil_fft_plan
