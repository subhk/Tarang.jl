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

# Parameterized on the concrete basis type `B` (RealFourier / ComplexFourier).
# A concrete `basis::B` field is what makes `_forward_output_spec` /
# `_backward_output_spec` type-stable: their `isa(transform.basis, RealFourier)`
# branch resolves at compile time, so the returned output shape/eltype are
# statically typed and the in-place transform path runs allocation-free. With an
# abstract `basis::Basis` field those specs inferred `Tuple`/`DataType` and the
# downstream `size()` checks heap-allocated a `Tuple{Int,Int}` on every call.
mutable struct FourierTransform{B<:Basis} <: Transform
    plan_forward::Union{Nothing, AbstractFFTPlan}
    plan_backward::Union{Nothing, AbstractFFTPlan}
    basis::B
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

    function FourierTransform(basis::B, axis::Int) where {B<:Basis}
        new{B}(nothing, nothing, basis, axis, Float64,
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
mutable struct ChebScratch{T<:AbstractFloat, N, P}
    real_in::Array{T, N}
    imag_in::Union{Nothing, Array{T, N}}
    tmp_real::Array{T, N}
    tmp_imag::Union{Nothing, Array{T, N}}
    plan::P
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
    # Keyed by `(input_shape, input_eltype)`. Each entry holds a ChebScratch{T,N,P}
    # stored as Any to allow heterogeneous plan types; function barriers in
    # _apply_forward! and _apply_backward! recover the concrete type.
    fwd_scratch::Dict{Tuple, Any}
    bwd_scratch::Dict{Tuple, Any}

    function ChebyshevTransform(basis::ChebyshevT)
        new(
            nothing, nothing,      # forward/backward matrices
            basis,
            nothing, nothing,      # FFTW plans
            0.0, 0.0, 0.0, 0.0,    # Scaling factors
            0, 0, 0, 0,            # Sizes and axis
            Dict{Tuple, Any}(),
            Dict{Tuple, Any}(),
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

"""
    JacobiTransform <: Transform

Collocation transform for Jacobi-family bases that have no dedicated fast
transform (ChebyshevU, ChebyshevV, Ultraspherical, generic Jacobi). The backward
matrix is the Vandermonde-like matrix `B[i, n+1] = φ_n(x_i)` of the basis
functions evaluated on the basis's own grid (via `evaluate_basis`); the forward
matrix is its inverse. Both are applied through the same dense-matrix machinery
as `LegendreTransform` (see `_legendre_forward`/`_legendre_backward`).

This exists because the transform planner previously had no branch for these
bases, so a field built on one silently used an identity (no-op) transform —
`get_grid_data` returned the raw coefficients. The collocation matrix is well
conditioned on these grids (cond ≈ 0.65·N), so the inverse is numerically safe.
"""
mutable struct JacobiTransform <: Transform
    forward_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    backward_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    basis::JacobiBasis

    # Grid information (no exact quadrature exists for these grids; the forward
    # matrix is the inverse of the backward collocation matrix, not a quadrature)
    grid_points::Union{Nothing, Vector{Float64}}
    quad_weights::Union{Nothing, Vector{Float64}}

    # Size information for dealiasing
    grid_size::Int
    coeff_size::Int
    axis::Int

    function JacobiTransform(basis::JacobiBasis)
        new(nothing, nothing, basis, nothing, nothing, 0, 0, 0)
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
    _get_or_alloc_scratch!(cache::Dict, key::Tuple, shape::NTuple{N,Int}, T::Type) → Array{T,N}

Look up a pre-allocated array in `cache[key]`. If the entry is missing or
has the wrong shape/eltype, allocate a new `zeros(T, shape...)` and store
it. Subsequent calls with the same key return the cached array without
allocation.
"""
# ---------------------------------------------------------------------------
# Allocation-free cache keys.
#
# Plan / scratch caches were keyed by tuples containing a `Type` (the element
# type) and a `Symbol` (the scratch slot). Both are pointers, so the key tuple
# is NOT isbits and Julia heap-allocates it on EVERY hot-path lookup (~tens of
# bytes/call, and markedly worse on some platforms — e.g. x64 Linux CI). Encode
# the element type and slot as `UInt8` tags instead: `(shape, eltype_tag,
# slot_tag)` is fully isbits, so building and hashing it allocates nothing.
#
# FFTs only support the four element types below; an unsupported eltype throws
# (it could not be transformed anyway), so the mapping is total and collision-free.
# ---------------------------------------------------------------------------
@inline _fft_eltype_tag(::Type{Float64})    = 0x00
@inline _fft_eltype_tag(::Type{ComplexF64}) = 0x01
@inline _fft_eltype_tag(::Type{Float32})    = 0x02
@inline _fft_eltype_tag(::Type{ComplexF32}) = 0x03
@inline _fft_eltype_tag(::Type{T}) where {T} =
    throw(ArgumentError("Tarang transform cache: unsupported FFT element type $T"))

# Scratch-slot discriminators (replace the previous `Symbol` slot component).
# Plain `UInt8` constants — passed as literals at call sites so they stay
# compile-time constants (a runtime `Val(symbol)` would itself allocate).
const SLOT_FWD_INTER = 0x00
const SLOT_BWD_INTER = 0x01
const SLOT_IRFFT     = 0x02

@inline function _get_or_alloc_scratch!(cache::Dict, key::Tuple, shape::NTuple{N,Int}, ::Type{T}) where {N,T}
    buf = get(cache, key, nothing)
    # `buf isa Array{T,N}` narrows the abstract `Dict`-value type to a concrete
    # array, so `size(buf)` and the returned value are statically typed. Without
    # the narrowing, `size(buf)`/`eltype(buf)` on an abstract `AbstractArray`
    # dynamic-dispatch and box on every hot-path call. This function only ever
    # stores `zeros(T, shape...)` (an `Array`), so the narrowing always hits.
    if buf isa Array{T,N} && size(buf) == shape
        return buf
    end
    new_buf = zeros(T, shape...)
    cache[key] = new_buf
    return new_buf
end

"""
    _buffer_matches(buf, shape, T) -> Bool

True when `buf` is already a concrete `Array{T,N}` of exactly `shape`. The
`buf isa Array{T,N}` test narrows `buf` (typically a `Union{Nothing,AbstractArray}`
field value) to a concrete array BEFORE `size(buf)` runs, so the size tuple is
not heap-allocated. Calling `size()` directly on the abstract field value boxes
the returned `Tuple` on every transform call.
"""
@inline _buffer_matches(buf, shape::NTuple{N,Int}, ::Type{T}) where {N,T} =
    buf isa Array{T,N} && size(buf) == shape

"""
    _find_pencil_plan(dist) → Union{Nothing, PencilFFTs.PencilFFTPlan}

Find cached PencilFFT plan from distributor transforms. Avoids repeated
linear scans of dist.transforms in hot paths.
"""
_find_pencil_plan(dist) = dist.pencil_fft_plan
