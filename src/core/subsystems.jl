"""
Subsystem and Subproblem classes for pencil-based matrix assembly.

Split into focused sub-files:
- subsystem_types.jl: subsystem configuration, construction, and coupling analysis
- subsystem_methods.jl: subsystem field access and gather/scatter helpers
- subproblem_types.jl: Subproblem definitions and sizing metadata
- subproblem_runtime.jl: subproblem gather/scatter, caches, BC projection, equation-space F gather
- subproblem_build.jl: subproblem assembly, matrix construction, and expression matrices
- subproblem_permutations.jl: Dedalus-style row/column permutations
- subproblem_matrix_utils.jl: sparse matrix utility helpers
- subproblem_ncc.jl: non-constant coefficient matrix builders and compatibility shims
- subsystem_exports.jl: exports for the subsystem API
"""

include("subsystems/subsystem_types.jl")
include("subsystems/subsystem_methods.jl")
include("subsystems/subproblem_types.jl")
include("subsystems/subproblem_runtime.jl")
include("subsystems/subproblem_build.jl")
include("subsystems/subproblem_permutations.jl")
include("subsystems/subproblem_matrix_utils.jl")
include("subsystems/subproblem_ncc.jl")
include("subsystems/subsystem_exports.jl")
