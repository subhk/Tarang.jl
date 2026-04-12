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
mutable struct ChebScratch
    real_in::AbstractArray
    imag_in::Union{Nothing, AbstractArray}
    tmp_real::AbstractArray
    tmp_imag::Union{Nothing, AbstractArray}
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

"""
    _find_pencil_plan(dist) → Union{Nothing, PencilFFTs.PencilFFTPlan}

Find cached PencilFFT plan from distributor transforms. Avoids repeated
linear scans of dist.transforms in hot paths.
"""
_find_pencil_plan(dist) = dist.pencil_fft_plan
