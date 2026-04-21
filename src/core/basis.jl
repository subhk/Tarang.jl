"""
Spectral basis classes

This module implements the spectral basis hierarchy:
- Basis: Abstract base type
- IntervalBasis: 1D bases on intervals
- Jacobi: General Jacobi polynomial basis (base for Chebyshev, Legendre, etc.)
- Ultraspherical/ChebyshevT/ChebyshevU: Jacobi with specific parameters
- RealFourier/ComplexFourier: Periodic Fourier bases

Key features implemented:
- Proper Jacobi parameter inheritance (a, b, a0, b0)
- product_matrix for NCC (Non-Constant Coefficient) support
- valid_elements for mode filtering
- derivative_basis returning correct output basis type
- Conversion matrices between bases

## GPU Compatibility

This module is designed for CPU-only setup operations. The basis module computes:
- Grid coordinates (returned as CPU arrays)
- Spectral matrices (returned as sparse CPU matrices)
- Wavenumbers (returned as CPU vectors)

For GPU execution, the domain.jl and field.jl modules handle data movement:
- Grid coordinates are computed on CPU, then moved to GPU via `on_architecture()`
- Field data is allocated on the appropriate architecture
- Transforms use GPU-accelerated FFT (CUFFT) when available

The `evaluate_basis` functions accept GPU coordinates and will automatically
convert them to CPU for evaluation, with a warning message. For GPU-native
operations, use transform-based methods instead of direct basis evaluation.
"""

# LinearAlgebra, SparseArrays, FFTW already in Tarang.jl
using SpecialFunctions: gamma, lgamma

export Basis, IntervalBasis, JacobiBasis, FourierBasis,
       Jacobi, Ultraspherical, ChebyshevT, ChebyshevU, ChebyshevV, Chebyshev, Legendre,
       RealFourier, ComplexFourier, Fourier,
       AffineCOV, BasisMeta,
       wavenumbers, derivative_basis, conversion_matrix, differentiation_matrix,
       product_matrix, ncc_matrix, valid_elements,
       grid_shape, coeff_shape, element_label, coordsys, pencil_compatible_size,
       local_grid, local_grids, evaluate_basis,
       is_fourier_basis, is_complex_fourier_basis, is_pure_fourier_domain,
       is_pure_complex_fourier_domain, validate_mpi_fourier_only


# Runtime map:
#   basis_core.jl             — basis types, metadata, constructors, and MPI validation
#   basis_wavenumbers.jl      — Fourier wavenumber storage/layout helpers
#   basis_product_matrices.jl — NCC product matrices and valid-element filtering
#   basis_operators.jl        — derivative bases, conversions, differentiation, dispatch
#   basis_interface.jl        — grid/interface methods and basis evaluation helpers

include("basis/basis_core.jl")
include("basis/basis_wavenumbers.jl")
include("basis/basis_product_matrices.jl")
include("basis/basis_operators.jl")
include("basis/basis_interface.jl")
