"""
Multiple-dispatch helpers for class-style type hierarchies.

Julia already provides multiple dispatch, but this module builds additional logic
around class-style hierarchies for `MultiClass`/`CachedMultiClass` behaviour,
similar to Dedalus's Python implementation.

# Overview

The dispatch system allows defining type hierarchies where the concrete type
to instantiate is determined at runtime based on argument types and values.
This is useful for operator construction where the specific operator type
depends on the operand types.

# Dispatch Hooks

The following hooks can be overridden for specific types:

- `dispatch_preprocess(::Type{T}, args, kwargs)` - Transform arguments before dispatch
- `dispatch_check(::Type{T}, args, kwargs)` - Return `true` if type T matches the arguments
- `dispatch_postprocess(::Type{T}, args, kwargs)` - Transform arguments after dispatch
- `stop_dispatch(::Type{T})` - Return `true` to exclude T and its subtypes from dispatch

# Example

```julia
abstract type Shape end
struct Circle <: Shape
    radius::Float64
end
struct Rectangle <: Shape
    width::Float64
    height::Float64
end

# Define dispatch checks
dispatch_check(::Type{Circle}, args::Tuple, kwargs::NamedTuple) = length(args) == 1
dispatch_check(::Type{Rectangle}, args::Tuple, kwargs::NamedTuple) = length(args) == 2

# Use multiclass dispatch
shape1 = multiclass_new(Shape, 5.0)        # Returns Circle(5.0)
shape2 = multiclass_new(Shape, 3.0, 4.0)   # Returns Rectangle(3.0, 4.0)
```
"""

using InteractiveUtils: subtypes

export multiclass_new, cached_multiclass_new, dispatch_preprocess,
       dispatch_check, dispatch_postprocess, stop_dispatch, invoke_constructor

# ---------------------------------------------------------------------------
# Default hooks
# ---------------------------------------------------------------------------

"""
    dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) -> (args, kwargs)

Transform arguments before dispatch matching.

Called once on the base class before searching for matching subtypes.
Override this to normalize or validate arguments, or to add default values.

Default implementation returns arguments unchanged.
"""
dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where T = (args, kwargs)

"""
    dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) -> Bool

Check if type T matches the given arguments.

Return `true` if T can be constructed with these arguments, `false` otherwise.
Can also throw an `ArgumentError` if the arguments are clearly invalid.

Default implementation returns `true` (all types match by default).
"""
dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) where T = true

"""
    dispatch_postprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) -> (args, kwargs)

Transform arguments after dispatch matching but before construction.

Called on the matched target type after dispatch resolution.
Override this to perform type-specific argument transformations.

Default implementation returns arguments unchanged.
"""
dispatch_postprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where T = (args, kwargs)

"""
    stop_dispatch(::Type{T}) -> Bool

Check if T should be excluded from dispatch traversal.

Return `true` to prevent T and all its subtypes from being considered
as dispatch candidates. Useful for abstract intermediate types.

Default implementation returns `false` (all types are included).
"""
stop_dispatch(::Type{T}) where T = false

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
Collect subtypes recursively, respecting `stop_dispatch` as a barrier.
Types where `stop_dispatch` returns `true` are excluded along with their subtypes.
"""
function _dispatch_descendants(cls::Type)
    descendants = Type[]
    for sub in subtypes(cls)
        if stop_dispatch(sub)
            continue
        end
        push!(descendants, sub)
        append!(descendants, _dispatch_descendants(sub))
    end
    return descendants
end

"""
Filter a list of types to keep only the most specific ones.
A type is removed if any other type in the list is a subtype of it.
"""
function _most_specific_types(types::Vector{Type})
    keep = trues(length(types))
    for i in 1:length(types)
        for j in 1:length(types)
            if i != j && types[j] <: types[i]
                keep[i] = false
                break
            end
        end
    end
    return types[keep]
end

# ---------------------------------------------------------------------------
# Core dispatch routine
# ---------------------------------------------------------------------------

"""
    multiclass_new(cls::Type{T}, args...; kwargs...) where T

Create an instance of `cls` or one of its subtypes based on argument matching.

# Algorithm

1. Call `dispatch_preprocess(cls, args, kwargs)` to transform arguments
2. Collect `cls` and all its subtypes (respecting `stop_dispatch` barriers)
3. Filter candidates by calling `dispatch_check(candidate, args, kwargs)`
4. Select the most specific matching type(s)
5. If exactly one match, call `dispatch_postprocess(target, args, kwargs)`
6. Construct and return the instance via `invoke_constructor`

# Errors

- `ArgumentError` if no types match the arguments
- `ArgumentError` if multiple equally-specific types match (ambiguous dispatch)

# Special Handling

If `dispatch_preprocess` throws a `SkipDispatchException`, its `output` field
is returned directly without constructing an instance. This is used for
short-circuit evaluation (e.g., `Add(0, 0)` returns `0` directly).

# Example

```julia
# Direct call with known type
grad = multiclass_new(Gradient, scalar_field)

# Dispatch to subtype based on arguments
basis = multiclass_new(Basis, coordinate)  # Returns Fourier, Chebyshev, etc.
```
"""
function multiclass_new(cls::Type{T}, args...; kwargs...) where T
    arg_tuple = Tuple(args)
    kw_nt = (; kwargs...)

    try
        arg_tuple, kw_nt = dispatch_preprocess(cls, arg_tuple, kw_nt)
        candidates = [cls; _dispatch_descendants(cls)]
        unique!(candidates)

        matches = Type[]
        for candidate in candidates
            if dispatch_check(candidate, arg_tuple, kw_nt)
                push!(matches, candidate)
            end
        end

        if isempty(matches)
            if length(candidates) == 1
                throw(ArgumentError("Provided arguments do not satisfy dispatch check for $(cls)."))
            else
                throw(ArgumentError("No subtypes of $(cls) match the supplied arguments."))
            end
        end

        matches = _most_specific_types(matches)
        if length(matches) > 1
            throw(ArgumentError("Ambiguous dispatch: multiple subtypes of $(cls) match the arguments: $(matches)."))
        end

        target = only(matches)
        arg_tuple, kw_nt = dispatch_postprocess(target, arg_tuple, kw_nt)
        return invoke_constructor(target, arg_tuple, kw_nt)

    catch err
        if err isa SkipDispatchException
            return err.output
        else
            rethrow()
        end
    end
end

"""
    invoke_constructor(cls::Type, args::Tuple, kwargs::NamedTuple)

Invoke the constructor for `cls` with the given arguments.

This is the final step of dispatch after all preprocessing and matching.
Override this for types that need special construction logic.
"""
function invoke_constructor(cls::Type, args::Tuple, kwargs::NamedTuple)
    return cls(args...; kwargs...)
end

# ---------------------------------------------------------------------------
# Cached variant
# ---------------------------------------------------------------------------

"""
    cached_multiclass_new(cls::Type{T}, args...; kwargs...) where T

Create an instance with caching to avoid duplicate constructions.

Works like `multiclass_new`, but caches instances by their construction
arguments. If an instance with the same arguments already exists (and
hasn't been garbage collected), it is returned instead of creating a new one.

# Caching Behavior

- Cache key is `(args_tuple, kwargs_namedtuple)`
- Uses weak references to allow garbage collection of unused instances
- Thread-safe via the underlying `CachedClass` implementation

# When to Use

Use `cached_multiclass_new` for:
- Immutable objects where identity doesn't matter
- Expensive-to-construct objects that may be requested multiple times
- Operators and fields that should be deduplicated

Use regular `multiclass_new` for:
- Mutable objects where each instance should be unique
- Objects with side effects in construction

# Example

```julia
# These return the same instance
field1 = cached_multiclass_new(ScalarField, domain, "u")
field2 = cached_multiclass_new(ScalarField, domain, "u")
field1 === field2  # true
```
"""
function cached_multiclass_new(cls::Type{T}, args...; kwargs...) where T
    cache = get!(cached_class_instances, cls) do
        CachedClass()
    end
    key = (Tuple(args), (; kwargs...))

    # Check for cached instance (handles WeakRef and GC'd entries)
    inst = get_cached_instance(cache, key)
    if inst !== nothing
        return inst
    end

    # Create new instance and cache it
    inst = multiclass_new(cls, args...; kwargs...)
    set_cached_instance!(cache, key, inst)
    return inst
end
