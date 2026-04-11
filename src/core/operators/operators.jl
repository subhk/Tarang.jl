"""
    Operators Module for Tarang.jl

This file includes all operator sub-modules in the correct dependency order.
The operators system provides spectral differential operators for PDEs including:

- Differential operators: gradient, divergence, curl, laplacian, fractional laplacian
- Tensor operators: trace, skew, transpose
- Interpolation and integration operators
- Basis conversion operators
- Component extraction operators
- Arithmetic operator composition

The include order matters due to type and function dependencies:
1. types.jl - Abstract types and struct definitions (must be first)
2. registration.jl - Operator registration tables
3. arithmetic.jl - Arithmetic operator types (depends on types.jl)
4. utils.jl - Utility functions
5. constructors.jl - Constructor functions (depends on types, registration)
6. dispatch.jl - Multiclass dispatch (depends on types, constructors)
7. derivatives.jl - Differentiation implementations
8. matrices.jl - Expression matrices for solvers
9. operations.jl - Various operator evaluations
10. tensor.jl - Tensor and special operator evaluations
11. evaluate.jl - Main evaluate dispatch and exports (must be last)
"""

# ============================================================================
# Include sub-modules in dependency order
# ============================================================================

# 1. Type definitions - abstract types and structs
include("types.jl")

# 2. Registration tables for operator parsing
include("registration.jl")

# 3. Arithmetic operator types (AddOperator, MultiplyOperator, etc.)
include("arithmetic.jl")

# 4. Utility functions (fftfreq, has, require_linearity)
include("utils.jl")

# 5. Constructor functions (grad, div, curl, lap, etc.)
include("constructors.jl")

# 6. Multiclass dispatch functions
include("dispatch.jl")

# 7. Derivative evaluation functions
include("derivatives.jl")

# 8. Expression matrices for implicit solvers
include("matrices.jl")

# 9. Various operator operations (interpolate, integrate, lift, etc.)
include("operations.jl")

# 10. Tensor operations and special operators
include("tensor.jl")

# 11. Symbolic differentiation for NLBVP Jacobians
include("symbolic_diff.jl")

# 12. Main evaluate dispatch and exports
include("evaluate.jl")
