"""
    Expression matrices

This file contains sparse operator-matrix construction for implicit solvers.
"""


# Runtime map:
#   matrices_expression.jl          — compositional expression_matrices entry points and linearity helpers
#   matrices_subproblem_helpers.jl  — subproblem-specific basis, operand, and mode helpers
#   matrices_subproblem_operators.jl — subproblem_matrix implementations and remaining operator expression fallbacks
#   matrices_builders.jl            — low-level differentiation and lift matrix builders

include("matrices/matrices_expression.jl")
include("matrices/matrices_subproblem_helpers.jl")
include("matrices/matrices_subproblem_operators.jl")
include("matrices/matrices_builders.jl")
