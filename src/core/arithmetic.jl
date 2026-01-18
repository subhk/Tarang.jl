"""
Arithmetic futures for deferred operations.

These types create deferred-operation nodes that integrate with the
``Future`` infrastructure provided in ``future.jl``.
"""

export Add, Multiply, DotProduct, CrossProduct, Power, Negate, Subtract, Divide
export dot, cross, ⋅, ×

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
    elseif isa(a, VectorField) && isa(b, Number)
        return add_scalar_to_vector_field(a, b)
    elseif isa(a, Number) && isa(b, VectorField)
        return add_scalar_to_vector_field(b, a)
    elseif isa(a, VectorField) && isa(b, ScalarField)
        throw(ArgumentError("Cannot add VectorField and ScalarField directly - incompatible types"))
    elseif isa(a, ScalarField) && isa(b, VectorField)
        throw(ArgumentError("Cannot add ScalarField and VectorField directly - incompatible types"))
    elseif isa(a, AbstractArray) && isa(b, AbstractArray)
        return _binary_array_op((x, y) -> x + y, a, b)
    else
        # Generic fallback - try direct addition (may fail for incompatible types)
        return a + b
    end
end

function add_scalar_to_vector_field(field::VectorField, value::Number)
    """Add a scalar to each component of a VectorField"""
    result = VectorField(field.dist, field.coordsys, "$(field.name)_plus_$(value)", field.bases, field.dtype)
    for i in 1:length(field.components)
        result[i] = add_scalar_to_field(field.components[i], value)
    end
    return result
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
    elseif isa(a, VectorField) && (isa(b, ScalarField) || isa(b, Number))
        return scale_vector_field(a, b)
    elseif (isa(a, ScalarField) || isa(a, Number)) && isa(b, VectorField)
        return scale_vector_field(b, a)
    elseif isa(a, VectorField) && isa(b, VectorField)
        # VectorField * VectorField is ambiguous - use dot() or cross() explicitly
        throw(ArgumentError("VectorField * VectorField is ambiguous. Use dot(a, b) for dot product or cross(a, b) for cross product"))
    elseif isa(a, AbstractArray) && isa(b, AbstractArray)
        return _binary_array_op((x, y) -> x * y, a, b)
    else
        # Generic fallback - try direct multiplication (may fail for incompatible types)
        return a * b
    end
end

@inline function _align_arrays(a::AbstractArray, b::AbstractArray)
    if is_gpu_array(a) && !is_gpu_array(b)
        return a, copy_to_device(b, a)
    elseif !is_gpu_array(a) && is_gpu_array(b)
        return copy_to_device(a, b), b
    else
        return a, b
    end
end

@inline function _binary_array_op(op::Function, a::AbstractArray, b::AbstractArray)
    a_aligned, b_aligned = _align_arrays(a, b)
    return op(a_aligned, b_aligned)
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
    if a.bases != b.bases
        throw(ArgumentError("Cannot compute dot product of VectorFields with different bases"))
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
    a_aligned, b_aligned = _align_arrays(a, b)
    return LinearAlgebra.dot(a_aligned, b_aligned)
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
    if a.bases != b.bases
        throw(ArgumentError("Cannot compute cross product of VectorFields with different bases"))
    end
    dist = a.dist
    coordsys = a.coordsys
    result = VectorField(dist, coordsys, "$(a.name)_cross_$(b.name)", a.bases, a.dtype)

    result[1] = a.components[2] * b.components[3] - a.components[3] * b.components[2]
    result[2] = a.components[3] * b.components[1] - a.components[1] * b.components[3]
    result[3] = a.components[1] * b.components[2] - a.components[2] * b.components[1]

    return result
end

function cross_operands(a::AbstractVector, b::AbstractVector)
    if length(a) != 3 || length(b) != 3
        throw(ArgumentError("Cross product requires 3-element vectors"))
    end
    # GPU-safe: LinearAlgebra.cross uses scalar indexing which fails on GPU
    # For small 3-element vectors, copying to CPU has negligible overhead
    if is_gpu_array(a) || is_gpu_array(b)
        a_cpu = Array(a)
        b_cpu = Array(b)
        result_cpu = LinearAlgebra.cross(a_cpu, b_cpu)
        # Return result on same device as first input
        return is_gpu_array(a) ? copy_to_device(result_cpu, a) : result_cpu
    else
        return LinearAlgebra.cross(a, b)
    end
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
    const_field = ScalarField(field.dist, "$(field.name)_const", field.bases, field.dtype)
    if field.scales !== nothing
        preset_scales!(const_field, field.scales)
    end
    ensure_layout!(const_field, :g)
    if get_grid_data(const_field) !== nothing
        fill!(get_grid_data(const_field), convert(field.dtype, value))
    end
    return const_field
end

function add_vector_fields(a::VectorField, b::VectorField)
    if length(a.components) != length(b.components)
        throw(ArgumentError("VectorField component mismatch for addition"))
    end
    if a.bases != b.bases
        throw(ArgumentError("Cannot add VectorFields with different bases"))
    end
    result = VectorField(a.dist, a.coordsys, "$(a.name)_plus_$(b.name)", a.bases, a.dtype)
    for i in 1:length(a.components)
        result[i] = a.components[i] + b.components[i]
    end
    return result
end

function scale_vector_field(field::VectorField, scalar)
    scalar_label = scalar isa ScalarField ? scalar.name : scalar
    result = VectorField(field.dist, field.coordsys, "$(field.name)_times_$(scalar_label)", field.bases, field.dtype)
    for i in 1:length(field.components)
        result[i] = field.components[i] * scalar
    end
    return result
end

# ---------------------------------------------------------------------------
# Power (deferred exponentiation)
# ---------------------------------------------------------------------------

"""
    Power <: Future

Deferred power operation for field exponentiation.
Following Dedalus arithmetic:Power pattern.

# Example
```julia
u_squared = Power(u, 2)      # u^2
u_sqrt = Power(u, 0.5)       # sqrt(u)
u_inv = Power(u, -1)         # 1/u
```
"""
mutable struct Power <: Future
    state::FutureState
    function Power(state::FutureState)
        new(state)
    end
end

function Power(operand, exponent::Real)
    return multiclass_new(Power, operand, exponent)
end

function operate(::Power, evaluated_args::Vector{Any})
    if length(evaluated_args) != 2
        throw(ArgumentError("Power expects exactly two arguments (operand, exponent)"))
    end
    return power_operands(evaluated_args[1], evaluated_args[2])
end

function power_operands(a::ScalarField, p::Real)
    # Work in grid space for nonlinear operation
    ensure_layout!(a, :g)
    result = ScalarField(a.dist, "$(a.name)_pow_$p", a.bases, a.dtype)
    ensure_layout!(result, :g)
    get_grid_data(result) .= get_grid_data(a) .^ p
    return result
end

function power_operands(a::Number, p::Real)
    return a^p
end

function power_operands(a::VectorField, p::Real)
    # Apply power to each component
    result = VectorField(a.dist, a.coordsys, "$(a.name)_pow_$p", a.bases, a.dtype)
    for i in 1:length(a.components)
        result.components[i] = power_operands(a.components[i], p)
    end
    return result
end

function power_operands(a, p)
    throw(ArgumentError("Power not implemented for operand of type $(typeof(a))"))
end

# ---------------------------------------------------------------------------
# Negate (unary minus)
# ---------------------------------------------------------------------------

"""
    Negate <: Future

Deferred negation operation.
Following Dedalus arithmetic pattern for unary minus.

# Example
```julia
neg_u = Negate(u)    # -u
neg_u = -u           # Same via operator overload
```
"""
mutable struct Negate <: Future
    state::FutureState
    function Negate(state::FutureState)
        new(state)
    end
end

function Negate(operand)
    return multiclass_new(Negate, operand)
end

function operate(::Negate, evaluated_args::Vector{Any})
    if length(evaluated_args) != 1
        throw(ArgumentError("Negate expects exactly one argument"))
    end
    return negate_operand(evaluated_args[1])
end

function negate_operand(a::ScalarField)
    result = ScalarField(a.dist, "neg_$(a.name)", a.bases, a.dtype)
    # Use the field's current layout to determine which data to negate
    if a.current_layout == :c
        ensure_layout!(result, :c)
        get_coeff_data(result) .= .-get_coeff_data(a)
    else
        # Default to grid space (covers :g and any uninitialized state)
        ensure_layout!(result, :g)
        get_grid_data(result) .= .-get_grid_data(a)
    end
    return result
end

function negate_operand(a::VectorField)
    result = VectorField(a.dist, a.coordsys, "neg_$(a.name)", a.bases, a.dtype)
    for i in 1:length(a.components)
        result.components[i] = negate_operand(a.components[i])
    end
    return result
end

function negate_operand(a::Number)
    return -a
end

function negate_operand(a)
    throw(ArgumentError("Negate not implemented for operand of type $(typeof(a))"))
end

# ---------------------------------------------------------------------------
# Subtract (a - b = a + (-b))
# ---------------------------------------------------------------------------

"""
    Subtract <: Future

Deferred subtraction operation.
Implemented as Add(a, Negate(b)) for consistency.

# Example
```julia
diff = Subtract(u, v)    # u - v
diff = u - v             # Same via operator overload
```
"""
mutable struct Subtract <: Future
    state::FutureState
    function Subtract(state::FutureState)
        new(state)
    end
end

function Subtract(a, b)
    return multiclass_new(Subtract, a, b)
end

function operate(::Subtract, evaluated_args::Vector{Any})
    if length(evaluated_args) != 2
        throw(ArgumentError("Subtract expects exactly two arguments"))
    end
    return subtract_operands(evaluated_args[1], evaluated_args[2])
end

function subtract_operands(a::ScalarField, b::ScalarField)
    if a.bases != b.bases
        throw(ArgumentError("Cannot subtract fields with different bases"))
    end
    result = ScalarField(a.dist, "$(a.name)_minus_$(b.name)", a.bases, a.dtype)
    if a.current_layout == :g && b.current_layout == :g
        ensure_layout!(result, :g)
        get_grid_data(result) .= get_grid_data(a) .- get_grid_data(b)
    else
        ensure_layout!(a, :c)
        ensure_layout!(b, :c)
        ensure_layout!(result, :c)
        get_coeff_data(result) .= get_coeff_data(a) .- get_coeff_data(b)
    end
    return result
end

function subtract_operands(a::VectorField, b::VectorField)
    if length(a.components) != length(b.components)
        throw(ArgumentError("VectorField component mismatch for subtraction"))
    end
    if a.bases != b.bases
        throw(ArgumentError("Cannot subtract VectorFields with different bases"))
    end
    result = VectorField(a.dist, a.coordsys, "$(a.name)_minus_$(b.name)", a.bases, a.dtype)
    for i in 1:length(a.components)
        result.components[i] = subtract_operands(a.components[i], b.components[i])
    end
    return result
end

function subtract_operands(a::Number, b::Number)
    return a - b
end

function subtract_operands(a::ScalarField, b::Number)
    return combine_add(a, -b)
end

function subtract_operands(a::Number, b::ScalarField)
    return combine_add(a, negate_operand(b))
end

function subtract_operands(a::VectorField, b::Number)
    # VectorField - Number: subtract from each component
    return add_scalar_to_vector_field(a, -b)
end

function subtract_operands(a::Number, b::VectorField)
    # Number - VectorField: negate vector then add scalar
    return add_scalar_to_vector_field(negate_operand(b), a)
end

function subtract_operands(a::VectorField, b::ScalarField)
    throw(ArgumentError("Cannot subtract ScalarField from VectorField directly - incompatible types"))
end

function subtract_operands(a::ScalarField, b::VectorField)
    throw(ArgumentError("Cannot subtract VectorField from ScalarField directly - incompatible types"))
end

function subtract_operands(a, b)
    throw(ArgumentError("Subtract not implemented for $(typeof(a)) and $(typeof(b))"))
end

# ---------------------------------------------------------------------------
# Divide (a / b)
# ---------------------------------------------------------------------------

"""
    Divide <: Future

Deferred division operation.
For field / scalar, this is efficient multiplication by 1/scalar.
For field / field, requires grid-space evaluation.

# Example
```julia
half_u = Divide(u, 2)    # u / 2
ratio = Divide(u, v)     # u / v (pointwise)
```
"""
mutable struct Divide <: Future
    state::FutureState
    function Divide(state::FutureState)
        new(state)
    end
end

function Divide(a, b)
    return multiclass_new(Divide, a, b)
end

function operate(::Divide, evaluated_args::Vector{Any})
    if length(evaluated_args) != 2
        throw(ArgumentError("Divide expects exactly two arguments"))
    end
    return divide_operands(evaluated_args[1], evaluated_args[2])
end

function divide_operands(a::ScalarField, b::Number)
    # a / b = a * (1/b)
    return combine_multiply(a, 1/b)
end

function divide_operands(a::ScalarField, b::ScalarField)
    if a.bases != b.bases
        throw(ArgumentError("Cannot divide fields with different bases"))
    end
    # Pointwise division in grid space
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    result = ScalarField(a.dist, "$(a.name)_div_$(b.name)", a.bases, a.dtype)
    ensure_layout!(result, :g)
    get_grid_data(result) .= get_grid_data(a) ./ get_grid_data(b)
    return result
end

function divide_operands(a::VectorField, b::Number)
    return scale_vector_field(a, 1/b)
end

function divide_operands(a::VectorField, b::ScalarField)
    result = VectorField(a.dist, a.coordsys, "$(a.name)_div_$(b.name)", a.bases, a.dtype)
    for i in 1:length(a.components)
        result[i] = divide_operands(a.components[i], b)
    end
    return result
end

function divide_operands(a::Number, b::ScalarField)
    # a / field = a * field^(-1)
    return combine_multiply(a, power_operands(b, -1))
end

function divide_operands(a::Number, b::Number)
    return a / b
end

function divide_operands(a, b)
    throw(ArgumentError("Divide not implemented for $(typeof(a)) and $(typeof(b))"))
end

# ---------------------------------------------------------------------------
# Fallback arithmetic overloads (deferred by default)
# ---------------------------------------------------------------------------

Base.:+(a::Operand, b::Operand) = multiclass_new(Add, a, b)
Base.:+(a::Operand, b::Number) = multiclass_new(Add, a, b)
Base.:+(a::Number, b::Operand) = multiclass_new(Add, a, b)

Base.:-(a::Operand) = Negate(a)
Base.:-(a::Operand, b::Operand) = Subtract(a, b)
Base.:-(a::Operand, b::Number) = Subtract(a, b)
Base.:-(a::Number, b::Operand) = Subtract(a, b)

Base.:*(a::Operand, b::Operand) = multiclass_new(Multiply, a, b)
Base.:*(a::Operand, b::Number) = multiclass_new(Multiply, a, b)
Base.:*(a::Number, b::Operand) = multiclass_new(Multiply, a, b)

Base.:/(a::Operand, b::Operand) = Divide(a, b)
Base.:/(a::Operand, b::Number) = Divide(a, b)
Base.:/(a::Number, b::Operand) = Divide(a, b)

Base.:^(a::Operand, p::Real) = Power(a, p)

# Disambiguating methods for Future-Operand combinations
# These are needed because Future <: Operand, so (Future, Operand) could match
# either (Operand, Operand) or (Future, Any), causing ambiguity.
Base.:+(a::Future, b::Operand) = multiclass_new(Add, a, b)
Base.:+(a::Operand, b::Future) = multiclass_new(Add, a, b)
Base.:+(a::Future, b::Future) = multiclass_new(Add, a, b)
Base.:+(a::Future, b::Number) = multiclass_new(Add, a, b)
Base.:+(a::Number, b::Future) = multiclass_new(Add, a, b)

Base.:-(a::Future, b::Operand) = Subtract(a, b)
Base.:-(a::Operand, b::Future) = Subtract(a, b)
Base.:-(a::Future, b::Future) = Subtract(a, b)
Base.:-(a::Future, b::Number) = Subtract(a, b)
Base.:-(a::Number, b::Future) = Subtract(a, b)
Base.:-(a::Future) = Negate(a)

Base.:*(a::Future, b::Operand) = multiclass_new(Multiply, a, b)
Base.:*(a::Operand, b::Future) = multiclass_new(Multiply, a, b)
Base.:*(a::Future, b::Future) = multiclass_new(Multiply, a, b)
Base.:*(a::Future, b::Number) = multiclass_new(Multiply, a, b)
Base.:*(a::Number, b::Future) = multiclass_new(Multiply, a, b)

Base.:/(a::Future, b::Operand) = Divide(a, b)
Base.:/(a::Operand, b::Future) = Divide(a, b)
Base.:/(a::Future, b::Future) = Divide(a, b)
Base.:/(a::Future, b::Number) = Divide(a, b)
Base.:/(a::Number, b::Future) = Divide(a, b)

Base.:^(a::Future, p::Real) = Power(a, p)

import LinearAlgebra: dot, cross, ⋅, ×

# Unicode operators for dot and cross products
# ⋅ (\cdot) for dot product - allows u⋅∇(u) syntax
⋅(a::Operand, b::Operand) = DotProduct(a, b)
⋅(a::Operand, b::Future) = DotProduct(a, b)
⋅(a::Future, b::Operand) = DotProduct(a, b)
⋅(a::Future, b::Future) = DotProduct(a, b)

# × (\times) for cross product - allows u×v syntax
×(a::Operand, b::Operand) = CrossProduct(a, b)
×(a::Operand, b::Future) = CrossProduct(a, b)
×(a::Future, b::Operand) = CrossProduct(a, b)
×(a::Future, b::Future) = CrossProduct(a, b)

# Note: In Julia's LinearAlgebra, `⋅` is an alias for `dot` and `×` is an alias for `cross`.
# The Unicode operator definitions above (⋅ and ×) automatically provide the ASCII `dot` and `cross`
# functions for Operand/Future types, so no additional definitions are needed here.

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
    state = build_future_state(collect(Any, args); name=:Add)
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
    state = build_future_state(collect(Any, args); name=:Multiply)
    return Multiply(state)
end

function dispatch_check(::Type{DotProduct}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2
end

function invoke_constructor(::Type{DotProduct}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(collect(Any, args); name=:DotProduct)
    return DotProduct(state)
end

function dispatch_check(::Type{CrossProduct}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2
end

function invoke_constructor(::Type{CrossProduct}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(collect(Any, args); name=:CrossProduct)
    return CrossProduct(state)
end

# Power dispatch
function dispatch_check(::Type{Power}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2 && isa(args[2], Real)
end

function invoke_constructor(::Type{Power}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(collect(Any, args); name=:Power)
    return Power(state)
end

# Negate dispatch
function dispatch_check(::Type{Negate}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 1
end

function invoke_constructor(::Type{Negate}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(collect(Any, args); name=:Negate)
    return Negate(state)
end

# Subtract dispatch
function dispatch_check(::Type{Subtract}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2
end

function invoke_constructor(::Type{Subtract}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(collect(Any, args); name=:Subtract)
    return Subtract(state)
end

# Divide dispatch
function dispatch_check(::Type{Divide}, args::Tuple, kwargs::NamedTuple)
    return length(args) == 2
end

function invoke_constructor(::Type{Divide}, args::Tuple, kwargs::NamedTuple)
    state = build_future_state(collect(Any, args); name=:Divide)
    return Divide(state)
end
