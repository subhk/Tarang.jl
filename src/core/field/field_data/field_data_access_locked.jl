"""
    Field data access and locking

This file contains the `has` helpers, `LockedField`, backward-compatible
ScalarField property shims, and LockedField-specific layout/scale guards.
"""

"""
    has(operand, vars...) -> Bool

Determine if expression tree contains any of the specified operands/operators.
This is the spectral pattern for linearity checking.

For fields: returns true if `operand in vars`
For operators: recursively checks operands/arguments
"""
function has(field::ScalarField, vars...)
    # Field.has: returns (not vars) or (self in vars)
    return isempty(vars) || field in vars
end

function has(field::VectorField, vars...)
    return isempty(vars) || field in vars
end

function has(field::TensorField, vars...)
    return isempty(vars) || field in vars
end

# Generic fallback for numbers/constants - they never contain variables
has(::Number, vars...) = false

"""
    require_linearity(expr, vars...; allow_affine=false)

Require expression to be linear in specified variables.
Following spectral pattern from arithmetic.py.

Raises error if expression is nonlinear in vars.
"""
function require_linearity end  # Forward declaration, implemented in operators.jl

"""
    LockedField <: Operand

A field wrapper that restricts layout and scale changes to specific allowed values.
Following field:LockedField pattern.

This is useful for:
- Output handlers that need fields in specific layouts
- Preventing accidental layout changes during evaluation
- Enforcing coefficient-space or grid-space operations

# Example
```julia
locked = LockedField(u, :g)          # Lock to grid space
locked = LockedField(u, :c)          # Lock to coefficient space
locked = LockedField(u, :g, (1,))    # Lock layout and scales
```
"""
struct LockedField <: Operand
    field::ScalarField
    layout::Symbol
    scales::Union{Nothing, Tuple}

    function LockedField(field::ScalarField, layout::Symbol, scales::Union{Nothing, Tuple}=nothing)
        # Ensure field is in the locked layout
        ensure_layout!(field, layout)
        if scales !== nothing
            normalized_scales = remedy_scales(field.dist, scales)
            set_scales!(field, normalized_scales)
            return new(field, layout, normalized_scales)
        end
        return new(field, layout, scales)
    end
end

function Base.getproperty(field::ScalarField, s::Symbol)
    if s === :buffers
        return getfield(field, :storage)  # backward-compatible :buffers → :storage
    elseif s === :data_g
        throw(ArgumentError("Direct access to field.data_g is deprecated. Use get_grid_data(field) instead."))
    elseif s === :data_c
        throw(ArgumentError("Direct access to field.data_c is deprecated. Use get_coeff_data(field) instead."))
    else
        return getfield(field, s)
    end
end

function Base.setproperty!(field::ScalarField, s::Symbol, value)
    if s === :buffers
        setfield!(field, :storage, value)  # backward-compatible :buffers → :storage
    elseif s === :data_g
        throw(ArgumentError("Assign to field.data_g via set_grid_data!(field, value) instead of direct property access."))
    elseif s === :data_c
        throw(ArgumentError("Assign to field.data_c via set_coeff_data!(field, value) instead of direct property access."))
    else
        setfield!(field, s, value)
    end
    return value
end

function Base.propertynames(::ScalarField, private::Bool=false)
    base_names = fieldnames(ScalarField)
    filtered = filter(n -> n ∉ (:data_g, :data_c), base_names)
    return (filtered..., :buffers)  # virtual property for backward compat
end

# LockedField convenience constructors
LockedField(field::ScalarField) = LockedField(field, field.current_layout, field.scales)

# Forward field access methods to underlying field
function Base.getindex(lf::LockedField, layout::String)
    layout_sym = Symbol(layout)
    if layout_sym != lf.layout
        throw(ArgumentError("Cannot access LockedField layout $layout (locked to $(lf.layout))"))
    end
    return getindex(lf.field, layout)
end

function Base.getindex(lf::LockedField, layout::Symbol)
    if layout != lf.layout
        throw(ArgumentError("Cannot access LockedField layout $layout (locked to $(lf.layout))"))
    end
    return layout == :g ? get_local_data(get_grid_data(lf.field)) : get_local_data(get_coeff_data(lf.field))
end

Base.getindex(lf::LockedField, key) = getindex(lf.field, key)
Base.size(lf::LockedField) = size(lf.layout == :g ? get_grid_data(lf.field) : get_coeff_data(lf.field))

# Property forwarding
function Base.getproperty(lf::LockedField, s::Symbol)
    if s in (:field, :layout, :scales)
        return getfield(lf, s)
    elseif s == :name
        return lf.field.name
    elseif s == :dist
        return lf.field.dist
    elseif s == :bases
        return lf.field.bases
    elseif s == :dtype
        return lf.field.dtype
    elseif s == :domain
        return lf.field.domain
    elseif s == :data_g
        return get_grid_data(lf.field)
    elseif s == :data_c
        return get_coeff_data(lf.field)
    elseif s == :current_layout
        return lf.layout  # Return locked layout, not field's current
    else
        return getfield(lf, s)
    end
end


function change_scales!(lf::LockedField, new_scales)
    if lf.scales !== nothing
        normalized_scales = remedy_scales(lf.field.dist, new_scales)
        locked_scales = remedy_scales(lf.field.dist, lf.scales)
        if normalized_scales != locked_scales
            throw(ArgumentError("Cannot change scales on LockedField from $(lf.scales) to $new_scales"))
        end
    end
    change_scales!(lf.field, new_scales)
    return lf
end

"""
    ensure_layout!(lf::LockedField, layout::Symbol)

Attempt to change layout on a locked field.
Only succeeds if layout matches the locked layout.
"""
function ensure_layout!(lf::LockedField, layout::Symbol)
    if layout != lf.layout
        throw(ArgumentError("Cannot change layout on LockedField from $(lf.layout) to $layout"))
    end
    ensure_layout!(lf.field, layout)
    return lf
end

"""
    unlock(lf::LockedField)

Get the underlying unlocked field.
"""
unlock(lf::LockedField) = lf.field
