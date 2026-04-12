"""
Solver implementations for Tarang.jl

Split into focused sub-files:
- solver_types.jl: Solver definitions and constructors
- solver_stepping.jl: Time stepping, BVP/EVP solve
- lazy_rhs.jl: Type-specialized lazy RHS evaluation with broadcasting fusion
- solver_utils.jl: Diagnostics and exports
"""

include("solvers/solver_types.jl")
include("solvers/solver_stepping.jl")
include("solvers/lazy_rhs.jl")
include("solvers/solver_utils.jl")
