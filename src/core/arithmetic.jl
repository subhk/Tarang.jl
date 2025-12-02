"""
Arithmetic futures translated from ``dedalus/core/arithmetic.py``.

These types create deferred-operation nodes that integrate with the
``Future`` infrastructure provided in ``future.jl``.  The goal is feature
parity with Dedalus while remaining idiomatic to Julia.
"""

# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

mutable struct Add <: Future
    state::FutureState
    function Add(state::FutureState)
        new(state)
    end
end

function Add(args...)
    return multiclass_new(Add, args...)
end

function preprocess_add(args)
    collected = Any[]
    for arg in args
        if arg === nothing
            continue
        elseif arg isa Add
            append!(collected, future_args(arg))
        elseif isa(arg, Number) && arg == 0
            continue
        else
            push!(collected, arg)
        end
    end
    return collected
end

function operate(::Add, evaluated_args::Vector{Any})
    result = evaluated_args[1]
    for idx in 2:length(evaluated_args)
        result = combine_add(result, evaluated_args[idx])
    end
    return result
end

function combine_add(a, b)
    if isa(a, ScalarField) && isa(b, ScalarField)
        return a + b
    elseif isa(a, ScalarField) && isa(b, Number)
        return add_scalar_to_field(a, b)
    elseif isa(a, Number) && isa(b, ScalarField)
        return add_scalar_to_field(b, a)
    elseif isa(a, Number) && isa(b, Number)
        return a + b
    elseif isa(a, VectorField) && isa(b, VectorField)
        return add_vector_fields(a, b)
    else
        return a + b
    end
end

# ---------------------------------------------------------------------------
# Multiply
# ---------------------------------------------------------------------------

mutable struct Multiply <: Future
    state::FutureState
    function Multiply(state::FutureState)
        new(state)
    end
end

function Multiply(args...)
    return multiclass_new(Multiply, args...)
end

function preprocess_multiply(args)
    collected = Any[]
    for arg in args
        if arg === nothing
            continue
        elseif isa(arg, Multiply)
            append!(collected, future_args(arg))
        elseif isa(arg, Number)
            if arg == 0
                return [0]
            elseif arg == 1
                continue
            else
                push!(collected, arg)
            end
        else
            push!(collected, arg)
        end
    end
    return collected
end

function operate(::Multiply, evaluated_args::Vector{Any})
    result = evaluated_args[1]
    for idx in 2:length(evaluated_args)
        result = combine_multiply(result, evaluated_args[idx])
    end
    return result
end

function combine_multiply(a, b)
    if isa(a, ScalarField) && isa(b, ScalarField)
        return a * b
    elseif isa(a, ScalarField) && isa(b, Number)
        return a * b
    elseif isa(a, Number) && isa(b, ScalarField)
        return b * a
    elseif isa(a, Number) && isa(b, Number)
        return a * b
    else
        return a * b
    end
end

# ---------------------------------------------------------------------------
# Dot product
# ---------------------------------------------------------------------------

mutable struct DotProduct <: Future
    state::FutureState
    function DotProduct(state::FutureState)
        new(state)
    end
end

function DotProduct(a, b)
    return multiclass_new(DotProduct, a, b)
end

function operate(::DotProduct, evaluated_args::Vector{Any})
    if length(evaluated_args) != 2
        throw(ArgumentError("DotProduct expects exactly two operands"))
    end
    return dot_operands(evaluated_args[1], evaluated_args[2])
end

function dot_operands(a::VectorField, b::VectorField)
    if length(a.components) != length(b.components)
        throw(ArgumentError("VectorField components mismatch for dot product"))
    end
    result = a.components[1] * b.components[1]
    for i in 2:length(a.components)
        result = result + a.components[i] * b.components[i]
    end
    return result
end

function dot_operands(a::Number, b::Number)
    return a * b
end

function dot_operands(a::AbstractArray, b::AbstractArray)
    return LinearAlgebra.dot(a, b)
end

function dot_operands(a, b)
    throw(ArgumentError("Dot product not implemented for operands of type $(typeof(a)) and $(typeof(b))"))
end

# ---------------------------------------------------------------------------
# Cross product
# ---------------------------------------------------------------------------

mutable struct CrossProduct <: Future
    state::FutureState
    function CrossProduct(state::FutureState)
        new(state)
    end
end

function CrossProduct(a, b)
    return multiclass_new(CrossProduct, a, b)
end

function operate(::CrossProduct, evaluated_args::Vector{Any})
    if length(evaluated_args) != 2
        throw(ArgumentError("CrossProduct expects exactly two operands"))
    end
    return cross_operands(evaluated_args[1], evaluated_args[2])
end

function cross_operands(a::VectorField, b::VectorField)
    if length(a.components) != 3 || length(b.components) != 3
        throw(ArgumentError("Cross product requires 3-component vector fields"))
    end
    dist = a.dist
    coordsys = a.coordsys
    result = VectorField(dist, coordsys, "$(a.name)_cross_$(b.name)", a.bases, a.dtype, a.device_config)

    result[1] = a.components[2] * b.components[3] - a.components[3] * b.components[2]
    result[2] = a.components[3] * b.components[1] - a.components[1] * b.components[3]
    result[3] = a.components[1] * b.components[2] - a.components[2] * b.components[1]

    return result
end

function cross_operands(a::AbstractVector, b::AbstractVector)
    return LinearAlgebra.cross(a, b)
end

function cross_operands(a, b)
    throw(ArgumentError("Cross product not implemented for operands of type $(typeof(a)) and $(typeof(b))"))
end

# ---------------------------------------------------------------------------
# Support utilities
# ---------------------------------------------------------------------------

function add_scalar_to_field(field::ScalarField, value::Number)
    const_field = constant_field_like(field, value)
    return field + const_field
end

function constant_field_like(field::ScalarField, value::Number)
    const_field = ScalarField(field.dist, "$(field.name)_const", field.bases, field.dtype, field.device_config)
    ensure_layout!(const_field, :g)
    if const_field.data_g !== nothing
        fill!(const_field.data_g, convert(field.dtype, value))
    end
    return const_field
end

function add_vector_fields(a::VectorField, b::VectorField)
    if length(a.components) != length(b.components)
        throw(ArgumentError("VectorField component mismatch for addition"))
    end
    result = VectorField(a.dist, a.coordsys, "$(a.name)_plus_$(b.name)", a.bases, a.dtype, a.device_config)
    for i in 1:length(a.components)
        result[i] = a.components[i] + b.components[i]
    end
    return result
end

# ---------------------------------------------------------------------------
# Fallback arithmetic overloads (deferred by default)
# ---------------------------------------------------------------------------

Base.:+(a::Operand, b::Operand) = multiclass_new(Add, a, b)
Base.:+(a::Operand, b::Number) = multiclass_new(Add, a, b)
Base.:+(a::Number, b::Operand) = multiclass_new(Add, a, b)

Base.:*(a::Operand, b::Operand) = multiclass_new(Multiply, a, b)
Base.:*(a::Operand, b::Number) = multiclass_new(Multiply, a, b)
Base.:*(a::Number, b::Operand) = multiclass_new(Multiply, a, b)

Base.:*(a::Future, b) = multiclass_new(Multiply, a, b)
Base.:*(a, b::Future) = multiclass_new(Multiply, a, b)
Base.:+(a::Future, b) = multiclass_new(Add, a, b)
Base.:+(a, b::Future) = multiclass_new(Add, a, b)

import LinearAlgebra: dot, cross
dot(a::Operand, b::Operand) = DotProduct(a, b)
cross(a::Operand, b::Operand) = CrossProduct(a, b)

Base.:⋅(a::Operand, b::Operand) = DotProduct(a, b)
Base.:×(a::Operand, b::Operand) = CrossProduct(a, b)

# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

function dispatch_preprocess(::Type{Add}, args::Tuple, kwargs::NamedTuple)
    processed = preprocess_add(args)
    if isempty(processed)
        throw(SkipDispatchException(0))
    elseif length(processed) == 1
        throw(SkipDispatchException(processed[1]))
    end
    return (Tuple(processed), kwargs)
end

dispatch_check(::Type{Add}, args::Tuple, kwargs::NamedTuple) = true

function invoke_constructor(::Type{Add}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(Vector{Any}(args); name=:Add)
    return Add(state)
end

function dispatch_preprocess(::Type{Multiply}, args::Tuple, kwargs::NamedTuple)
    processed = preprocess_multiply(args)
    if isempty(processed)
        throw(SkipDispatchException(1))
    elseif length(processed) == 1
        throw(SkipDispatchException(processed[1]))
    end
    return (Tuple(processed), kwargs)
end

dispatch_check(::Type{Multiply}, args::Tuple, kwargs::NamedTuple) = true

function invoke_constructor(::Type{Multiply}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(Vector{Any}(args); name=:Multiply)
    return Multiply(state)
end

function dispatch_check(::Type{DotProduct}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2
end

function invoke_constructor(::Type{DotProduct}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(Vector{Any}(args); name=:DotProduct)
    return DotProduct(state)
end

function dispatch_check(::Type{CrossProduct}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2
end

function invoke_constructor(::Type{CrossProduct}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(Vector{Any}(args); name=:CrossProduct)
    return CrossProduct(state)
end
