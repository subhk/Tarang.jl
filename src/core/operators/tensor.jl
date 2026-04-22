"""
    Tensor operations and special operators

This file contains evaluation and matrix helpers for tensor-style operators.
"""


# Runtime map:
#   tensor_basic.jl                — trace, skew, and transpose-components evaluation
#   tensor_curl_laplacian.jl       — curl and standard Laplacian evaluation
#   tensor_fractional_laplacian.jl — fractional Laplacian evaluation and matrix methods
#   tensor_misc.jl                 — outer-product and AdvectiveCFL utilities

include("tensor/tensor_basic.jl")
include("tensor/tensor_curl_laplacian.jl")
include("tensor/tensor_fractional_laplacian.jl")
include("tensor/tensor_misc.jl")
