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

    function FourierTransform(basis::Basis, axis::Int)
        new(nothing, nothing, basis, axis, Float64)
    end
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

    function ChebyshevTransform(basis::ChebyshevT)
        new(
            nothing, nothing,      # forward/backward matrices
            basis,
            nothing, nothing,      # FFTW plans
            0.0, 0.0, 0.0, 0.0,    # Scaling factors
            0, 0, 0, 0             # Sizes and axis
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
