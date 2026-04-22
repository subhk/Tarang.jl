"""
    Problem matrix building

This file contains solver-facing matrix construction for parsed problems.
"""


# Runtime map:
#   problem_matrices_build.jl         — top-level matrix assembly and equation-expression construction
#   problem_matrices_expr_analysis.jl — equation-variable detection, DOF inference, and operator splitting
#   problem_matrices_support.jl       — size helpers, equation conditions, and shared small utilities
#   problem_matrices_spectral.jl      — spectral operator blocks and expression-matrix assembly
#   problem_matrices_legacy.jl        — forcing-vector construction and legacy compatibility helpers

include("problem_matrices/problem_matrices_build.jl")
include("problem_matrices/problem_matrices_expr_analysis.jl")
include("problem_matrices/problem_matrices_support.jl")
include("problem_matrices/problem_matrices_spectral.jl")
include("problem_matrices/problem_matrices_legacy.jl")
