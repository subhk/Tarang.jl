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
    plan_forward::Union{Nothing, Any}
    plan_backward::Union{Nothing, Any}
    basis::Basis
    axis::Int
    plan_dtype::Type  # Real element type used for plan creation (e.g., Float64, Float32)

    function FourierTransform(basis::Basis, axis::Int)
        new(nothing, nothing, basis, axis, Float64)
    end
end

mutable struct ChebyshevTransform <: Transform
    matrices::Dict{String, AbstractMatrix}
    basis::ChebyshevT

    # FFTW DCT plans
    forward_plan::Union{Nothing, Any}
    backward_plan::Union{Nothing, Any}

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
            Dict{String, AbstractMatrix}(),
            basis,
            nothing, nothing,      # FFTW plans
            0.0, 0.0, 0.0, 0.0,    # Scaling factors
            0, 0, 0, 0             # Sizes and axis
        )
    end
end

mutable struct LegendreTransform <: Transform
    matrices::Dict{String, AbstractMatrix}
    basis::Legendre

    # Quadrature information
    grid_points::Union{Nothing, AbstractVector{Float64}}
    quad_weights::Union{Nothing, AbstractVector{Float64}}

    # Size information for dealiasing
    grid_size::Int
    coeff_size::Int
    axis::Int

    function LegendreTransform(basis::Legendre)
        new(
            Dict{String, AbstractMatrix}(),
            basis,
            nothing, nothing,  # Quadrature points and weights
            0, 0, 0            # Sizes and axis
        )
    end
end
