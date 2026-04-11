"""
Solver implementations for Tarang.jl

Split into focused sub-files:
- solver_types.jl: Solver definitions and constructors
- solver_stepping.jl: Time stepping, BVP/EVP solve
- solver_compiled_rhs.jl: RHS compilation and execution
- solver_utils.jl: Diagnostics and exports
"""

include("solvers/solver_types.jl")
include("solvers/solver_stepping.jl")
include("solvers/solver_compiled_rhs.jl")
include("solvers/solver_utils.jl")
