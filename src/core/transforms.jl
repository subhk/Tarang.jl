"""
Spectral transform classes with PencilFFTs integration

This module provides spectral transforms for various bases:
- Fourier (FFT/RFFT via FFTW and PencilFFTs)
- Chebyshev (DCT-I via FFTW)
- Legendre (Gauss-Legendre quadrature)

## File Organization

This module is split into multiple files for maintainability:
- transform_types.jl: Core type definitions
- transform_planning.jl: Transform planning functions
- transform_gpu.jl: GPU transform support
- transform_fourier.jl: Fourier transform execution
- transform_chebyshev.jl: Chebyshev transform execution
- transform_legendre.jl: Legendre transform execution
- transform_transposable.jl: TransposableField transform planning
- transform_grouped.jl: Grouped transform operations
"""

using PencilFFTs
using FFTW
using LinearAlgebra
using SparseArrays

# Include all the split files
include("transforms/transform_types.jl")
include("transforms/transform_planning.jl")
include("transforms/transform_gpu.jl")
include("transforms/transform_fourier.jl")
include("transforms/transform_chebyshev.jl")
include("transforms/transform_legendre.jl")
include("transforms/transform_transposable.jl")
include("transforms/transform_grouped.jl")

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
export forward_transform_3d!, backward_transform_3d!

# Export grouped transform functions (Dedalus GROUP_TRANSFORMS equivalent)
export GroupedTransformConfig, set_group_transforms!
export group_forward_transform!, group_backward_transform!

# Export setup functions
export setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!
export setup_fftw_transform!, setup_fftw_transforms_nd!
export setup_chebyshev_transform!, setup_legendre_transform!
export setup_pencil_arrays_3d, create_pencil_3d

# Export dealiasing functions
export dealias!, dealias_3d!, dealias_field!
export apply_basis_dealiasing!

# Export Legendre quadrature functions
export compute_legendre_quadrature
export evaluate_legendre_and_derivative
export build_legendre_polynomials

# Export utility functions
export get_transform_for_basis
export is_pencil_compatible, is_3d_pencil_optimal
export synchronize_transforms!

