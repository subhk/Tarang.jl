"""
Solver implementations for Tarang.jl

Split into focused sub-files:
- solver_execution_plan.jl: runtime path facts resolved once at construction
- solver_types.jl: Solver definitions and constructors
- solver_state_vectors.jl: Field/vector transport for matrix solver paths
- solver_stepping.jl: Time stepping, BVP/EVP solve
- lazy_rhs.jl: Type-specialized lazy RHS evaluation with broadcasting fusion
- rhs_runtime.jl: RHS evaluation strategy selection
- solver_utils.jl: Diagnostics and exports

Checkpoint save/load moved to src/tools/solver_checkpoint.jl: it reads and writes
NetCDF, whose slab layer loads after the whole core stack.
"""

include("solvers/solver_execution_plan.jl")
include("solvers/solver_types.jl")
include("solvers/solver_state_vectors.jl")
include("solvers/solver_compiled_rhs.jl")
include("solvers/lazy_rhs.jl")
include("solvers/rhs_runtime.jl")
include("solvers/solver_runtime.jl")
include("solvers/solver_stepping.jl")
include("solvers/solver_utils.jl")
