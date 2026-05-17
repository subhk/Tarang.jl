"""
Timestepping schemes for initial value problems.

This module provides IMEX (Implicit-Explicit) time integration methods
where linear terms are treated implicitly and nonlinear terms are treated explicitly.

Available methods:
- RK111, RK222, RK443: IMEX Runge-Kutta methods
- CNAB1, CNAB2: Crank-Nicolson Adams-Bashforth
- SBDF1-4: Semi-implicit Backward Differentiation Formulas
- ETD_RK222, ETD_CNAB2, ETD_SBDF2: Exponential Time Differencing
- DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2: GPU-native diagonal IMEX

All IMEX RK and multistep methods dispatch through the per-Fourier-mode
subproblem path (`step_subproblem_rk!` / `step_subproblem_multistep!`) when
subproblems are available, which is the recommended path for Chebyshev-tau
problems. Pure-Fourier domains can optionally use `SpectralLinearOperator`
for a diagonal-IMEX optimization.

**Note**: earlier versions shipped a `PencilLinearOperator` + `step_pencil_*!`
family that implemented per-wavenumber 1D solves directly via PencilFFTs
layouts. Those files were removed after the subproblem refactor made them
redundant — `step_subproblem_rk!` and `step_subproblem_multistep!` now
handle Chebyshev-Fourier MPI runs through the unified subproblem path with
correct DAE-style BC handling.
"""

# LinearAlgebra, SparseArrays, LoopVectorization, ExponentialUtilities already in Tarang.jl
using SparseArrays: SparseMatrixCSC, nnz

# TimeStepper abstract type is forward-declared in Tarang.jl

# Include submodules in dependency order
include("phi_functions.jl")      # ETD phi function utilities
include("types.jl")              # Timestepper struct definitions
include("spectral_operators.jl") # SpectralLinearOperator for diagonal IMEX
include("state.jl")              # TimestepperState and state management
include("state_utils.jl")        # State manipulation utilities
include("step_selection.jl")     # Runtime path and compatibility decisions
include("step_rk.jl")            # RK step functions
include("step_multistep.jl")     # CNAB, SBDF step functions
include("step_etd.jl")           # ETD step functions
include("step_diagonal_imex.jl") # GPU-native diagonal IMEX step functions
include("step_advanced.jl")      # Additional methods (MCNAB2, CNLF2, etc.)
include("dispatch.jl")           # Main step! dispatch function
