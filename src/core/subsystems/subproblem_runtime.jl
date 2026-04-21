"""
Subproblem runtime helpers split into focused sub-files:
- subproblem_io.jl: gather/scatter helpers, cached vectors, and space compression
- subproblem_rhs.jl: equation-space F gather, BC projections, and BC-array caches
- subproblem_modes.jl: condition parsing and valid-mode selection helpers
"""

include("subproblem_io.jl")
include("subproblem_rhs.jl")
include("subproblem_modes.jl")
