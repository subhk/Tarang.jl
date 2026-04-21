"""
Subproblem build helpers split into focused sub-files:
- subproblem_build_orchestration.jl: subproblem construction and NCC setup
- subproblem_expr_helpers.jl: expression DOF helpers and valid-mode selectors
- subproblem_matrix_build.jl: sparse matrix assembly and Woodbury rank helpers
"""

include("subproblem_build_orchestration.jl")
include("subproblem_expr_helpers.jl")
include("subproblem_matrix_build.jl")
