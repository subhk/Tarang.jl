"""
Custom exception definitions for Tarang.jl.

Providing named exception types helps upstream code express intent clearly
and lets callers catch specific error conditions.

All Tarang exceptions inherit from `TarangException`, which itself inherits
from Julia's `Exception` type. This allows catching all Tarang-specific
exceptions with a single `catch err::TarangException` clause.

# Exception Types
- `NonlinearOperatorError`: Raised when an operator is not linear in specified variables
- `DependentOperatorError`: Raised when an operator has invalid dependencies
- `SymbolicParsingError`: Raised when equation/expression parsing fails
- `UnsupportedEquationError`: Raised when an equation type is not supported
- `UndefinedParityError`: Raised when parity cannot be determined
- `SkipDispatchException`: Control flow exception for early dispatch returns
"""

export TarangException,
       NonlinearOperatorError,
       DependentOperatorError,
       SymbolicParsingError,
       UnsupportedEquationError,
       UndefinedParityError,
       SkipDispatchException

"""
Abstract base type for all Tarang-specific exceptions.

Allows catching all Tarang exceptions with `catch err::TarangException`.
"""
abstract type TarangException <: Exception end

"""
    NonlinearOperatorError(message::String)

Raised when an operator is not linear in the specified variables.

This typically occurs during matrix assembly when a nonlinear term
appears where a linear operator is expected.
"""
struct NonlinearOperatorError <: TarangException
    message::String
end
Base.showerror(io::IO, err::NonlinearOperatorError) = print(io, "NonlinearOperatorError: ", err.message)

"""
    DependentOperatorError(message::String)

Raised when an operator has invalid or circular dependencies.
"""
struct DependentOperatorError <: TarangException
    message::String
end
Base.showerror(io::IO, err::DependentOperatorError) = print(io, "DependentOperatorError: ", err.message)

"""
    SymbolicParsingError(message::String)

Raised when parsing of symbolic equations or expressions fails.

Common causes include:
- Unmatched brackets
- Invalid syntax
- Unknown operators or functions
"""
struct SymbolicParsingError <: TarangException
    message::String
end
Base.showerror(io::IO, err::SymbolicParsingError) = print(io, "SymbolicParsingError: ", err.message)

"""
    UnsupportedEquationError(message::String)

Raised when an equation type or configuration is not supported.

Examples include:
- Domain size mismatches
- Incompatible equation types for a problem class
- Invalid boundary condition specifications
"""
struct UnsupportedEquationError <: TarangException
    message::String
end
Base.showerror(io::IO, err::UnsupportedEquationError) = print(io, "UnsupportedEquationError: ", err.message)

"""
    UndefinedParityError(message::String)

Raised when the parity of an expression cannot be determined.

Parity (even/odd symmetry) is required for certain optimizations
in spectral methods.
"""
struct UndefinedParityError <: TarangException
    message::String
end
Base.showerror(io::IO, err::UndefinedParityError) = print(io, "UndefinedParityError: ", err.message)

"""
    SkipDispatchException(output)

Control flow exception used internally by the dispatch system.

When thrown during preprocessing, the dispatch system catches this
exception and returns `output` directly instead of invoking the
target constructor. This is used for short-circuit evaluation
(e.g., returning 0 for `Add(0, 0)` without creating an Add operator).

Not intended for use outside the dispatch machinery.
"""
struct SkipDispatchException <: TarangException
    output::Any
end
Base.showerror(io::IO, err::SkipDispatchException) = print(io, "SkipDispatchException (output: ", typeof(err.output), ")")

