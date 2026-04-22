"""
    Derivative evaluation functions

This file contains all differentiation implementations including:
- Gradient, divergence, and differentiate evaluators
- Fourier derivative functions (distributed and local, CPU and GPU)
- Chebyshev derivative functions
- Legendre derivative functions
- Matrix application helpers
"""

# LinearAlgebra, SparseArrays, FFTW already in Tarang.jl


# Runtime map:
#   derivatives_eval.jl         — gradient/divergence evaluators and Differentiate dispatch
#   derivatives_fourier.jl      — Fourier derivative implementations for local and distributed layouts
#   derivatives_polynomial.jl   — Chebyshev and Legendre derivative implementations
#   derivatives_matrix_apply.jl — dense/sparse matrix application helpers along arbitrary axes

include("derivatives/derivatives_eval.jl")
include("derivatives/derivatives_fourier.jl")
include("derivatives/derivatives_polynomial.jl")
include("derivatives/derivatives_matrix_apply.jl")
