"""
Spectral transform classes with PencilFFTs integration

This module provides spectral transforms for various bases:
- Fourier (FFT/RFFT via FFTW and PencilFFTs)
- Chebyshev (DCT-I via FFTW)
- Legendre (Gauss-Legendre quadrature)

## File Organization

This module is split into multiple files for maintainability:
- transform_types.jl: Core type definitions + in-place dispatch protocol
- transform_planning.jl: Transform planning (builds 1D FFTW plans)
- transform_gpu.jl: Serial CPU + GPU dispatch (`forward_transform!`)
- transform_fourier.jl: Fourier transform execution (in-place and legacy)
- transform_chebyshev.jl: Chebyshev transform execution (in-place and legacy)
- transform_legendre.jl: Legendre transform execution
- transform_transposable.jl: TransposableField transform planning
"""

# PencilFFTs, FFTW, LinearAlgebra, SparseArrays already in Tarang.jl

# Include all the split files
include("transforms/transform_types.jl")
include("transforms/transform_planning.jl")
include("transforms/transform_gpu.jl")
include("transforms/transform_fourier.jl")
include("transforms/transform_chebyshev.jl")
include("transforms/transform_legendre.jl")
include("transforms/transform_transposable.jl")

# ============================================================================
# Exports
# ============================================================================

# Export abstract type
export Transform

# Export transform types
export PencilFFTTransform, FourierTransform, ChebyshevTransform, LegendreTransform

# Export main transform planning function
export plan_transforms!

# TransposableField transform planning
export plan_transposable_transforms!, setup_distributed_transforms!

# Export forward/backward transform functions
export forward_transform!, backward_transform!

# Export setup functions
export setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!
export setup_fftw_transform!
export setup_chebyshev_transform!, setup_legendre_transform!

# Export Legendre quadrature functions
export compute_legendre_quadrature
export evaluate_legendre_and_derivative
export build_legendre_polynomials

