"""
Custom exception definitions.

Providing named exception types helps upstream code express intent clearly
and lets callers catch specific error conditions.
"""

abstract type TarangException <: Exception end

struct NonlinearOperatorError <: TarangException
    message::String
end
Base.showerror(io::IO, err::NonlinearOperatorError) = print(io, err.message)

struct DependentOperatorError <: TarangException
    message::String
end
Base.showerror(io::IO, err::DependentOperatorError) = print(io, err.message)

struct SymbolicParsingError <: TarangException
    message::String
end
Base.showerror(io::IO, err::SymbolicParsingError) = print(io, err.message)

struct UnsupportedEquationError <: TarangException
    message::String
end
Base.showerror(io::IO, err::UnsupportedEquationError) = print(io, err.message)

struct UndefinedParityError <: TarangException
    message::String
end
Base.showerror(io::IO, err::UndefinedParityError) = print(io, err.message)

struct SkipDispatchException <: TarangException
    output::Any
end
Base.showerror(io::IO, ::SkipDispatchException) = print(io, "SkipDispatchException")

