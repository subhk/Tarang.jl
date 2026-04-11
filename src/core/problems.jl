"""
Problem definitions and equation parsing for Tarang.jl

Split into focused sub-files:
- problem_types.jl: IVP, LBVP, NLBVP, EVP definitions and constructors
- problem_parsing.jl: Expression parsing and evaluation
- problem_matrices.jl: Matrix building for solvers
- problem_utils.jl: Validation, substitution, introspection, exports
"""

include("problems/problem_types.jl")
include("problems/problem_parsing.jl")
include("problems/problem_matrices.jl")
include("problems/problem_utils.jl")
