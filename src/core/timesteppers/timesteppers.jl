"""
Timestepping schemes for initial value problems

This module provides IMEX (Implicit-Explicit) time integration methods
following the Dedalus convention where linear terms are treated implicitly
and nonlinear terms are treated explicitly.

Available methods:
- RK111, RK222, RK443: IMEX Runge-Kutta methods
- CNAB1, CNAB2: Crank-Nicolson Adams-Bashforth
- SBDF1-4: Semi-implicit Backward Differentiation Formulas
- ETD_RK222, ETD_CNAB2, ETD_SBDF2: Exponential Time Differencing
- DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2: GPU-native diagonal IMEX
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays: SparseMatrixCSC, nnz
using LoopVectorization  # For SIMD loops
using ExponentialUtilities  # For Krylov-based φ functions

# Abstract type for all timesteppers
abstract type TimeStepper end

# Include submodules in dependency order
include("phi_functions.jl")      # ETD phi function utilities
include("types.jl")              # Timestepper struct definitions
include("spectral_operators.jl") # SpectralLinearOperator for diagonal IMEX
include("state.jl")              # TimestepperState and state management
include("state_utils.jl")        # State manipulation utilities
include("step_rk.jl")            # RK step functions
include("step_multistep.jl")     # CNAB, SBDF step functions
include("step_etd.jl")           # ETD step functions
include("step_diagonal_imex.jl") # GPU-native diagonal IMEX step functions
include("step_advanced.jl")      # Advanced methods (MCNAB2, CNLF2, etc.)
include("dispatch.jl")           # Main step! dispatch function
