"""
Deferred evaluation infrastructure for symbolic/deferred operator trees.

This module provides:

- ``FutureState`` stores common bookkeeping (arguments, cached outputs, etc.).
- Concrete ``Future`` subtypes wrap a ``FutureState`` and implement ``operate``.
- ``evaluate`` recursively resolves subtrees and applies the concrete operator.

The implementation focuses on argument tracking, caching, and substitution
helpers. Additional hooks (e.g. NCC preparation) can be expanded later.
"""

# ---------------------------------------------------------------------------
# Core definitions
# ---------------------------------------------------------------------------

abstract type Future <: Operand end

mutable struct FutureState
    args::Vector{Any}
    original_args::Vector{Any}
    out::Any
    dist::Any
    dtype::Union{Nothing, DataType}
    store_last::Bool
    last_id::Union{Nothing, Int}
    last_out::Any
    name::Symbol
    _eval_buffer::Vector{Any}  # pre-allocated buffer reused by evaluate()
end

function FutureState(args::Vector{Any}; out=nothing, store_last::Bool=true,
                     metadata=Dict{Symbol, Any}(), name::Symbol=:unknown)
    dist = unify_attributes(args, "dist")
    dtype = unify_attributes(args, "dtype")
    # Support legacy metadata dict: extract :name if present
    effective_name = haskey(metadata, :name) ? metadata[:name]::Symbol : name
    n = length(args)
    return FutureState(copy(args), copy(args), out, dist, dtype, store_last,
                       nothing, nothing, effective_name, Vector{Any}(undef, n))
end

function build_future_state(args; name::Symbol, out=nothing, store_last::Bool=true)
    return FutureState(collect(Any, args); out=out, store_last=store_last, name=name)
end

# ---------------------------------------------------------------------------
# Helpers for working with futures
# ---------------------------------------------------------------------------

function future_state(f::Future)
    if !hasfield(typeof(f), :state)
        throw(ArgumentError("Future subtype $(typeof(f)) must have a :state field of type FutureState"))
    end
    return getfield(f, :state)
end

future_args(f::Future) = future_state(f).args
original_args(f::Future) = future_state(f).original_args

"""
    future_metadata(f::Future)

Return metadata as a Dict for backward compatibility.
Internally, only the :name field is stored.
"""
function future_metadata(f::Future)
    state = future_state(f)
    return Dict{Symbol, Any}(:name => state.name)
end

function set_future_args!(f::Future, args::Vector{Any})
    state = future_state(f)
    empty!(state.args)
    append!(state.args, args)
    # Resize eval buffer to match new arg count
    resize!(state._eval_buffer, length(args))
    state.last_id = nothing
    state.last_out = nothing
    return f
end

function reset!(f::Future)
    state = future_state(f)
    empty!(state.args)
    append!(state.args, state.original_args)
    # Resize eval buffer to match restored arg count
    resize!(state._eval_buffer, length(state.original_args))
    state.last_id = nothing
    state.last_out = nothing
    return f
end

"""
    atoms(f::Future, types...)

Collect leaf operands matching the provided types.
"""
function atoms(f::Future, types...)
    state = future_state(f)
    leaves = OrderedSet{Any}()
    for arg in state.args
        if arg isa Future
            for leaf in atoms(arg, types...)
                push!(leaves, leaf)
            end
        elseif isempty(types) || any(arg isa T for T in types)
            push!(leaves, arg)
        end
    end
    return leaves
end

"""
    has_operand(f::Future, vars...)

Return true if any subtree matches the given operands/operators.
"""
function has_operand(f::Future, vars...)
    # Helper to check if an object matches any of the vars
    function _matches_any(obj, vars)
        for var in vars
            if isa(var, Type)
                obj isa var && return true
            else
                obj === var && return true
            end
        end
        return false
    end

    # Check if f itself matches
    _matches_any(f, vars) && return true

    # Check all arguments recursively
    state = future_state(f)
    for arg in state.args
        if arg isa Future
            has_operand(arg, vars...) && return true
        else
            _matches_any(arg, vars) && return true
        end
    end
    return false
end

function replace_operand(arg, old, new)
    if arg === old
        return new
    elseif isa(old, Type) && arg isa old
        return new
    elseif arg isa Future
        return substitute_future(arg, old, new)
    else
        return arg
    end
end

"""
    substitute_future(f::Future, old, new)

Substitute `old` with `new` in a Future expression tree.
Named to avoid shadowing Base.replace.

Note: This function mutates `f` in-place (following the same semantics as
`substitute_future!`). If `f` itself matches `old`, returns `new` directly.

This is intentionally mutating despite not having `!` in the name, because
the recursive tree traversal via `replace_operand` requires returning the
(potentially different) replacement value, which is incompatible with the
Bool-returning `substitute_future!` interface.
"""
function substitute_future(f::Future, old, new)
    if f === old || (isa(old, Type) && f isa old)
        return new
    end
    state = future_state(f)
    new_args = Any[replace_operand(arg, old, new) for arg in state.args]
    set_future_args!(f, new_args)
    return f
end

"""
    substitute_future!(f::Future, old, new)

In-place substitution of `old` with `new` in a Future expression tree.
Returns `true` if substitution was performed, `false` if `f` itself matched `old`.

Use this when you explicitly want to mutate the original Future.
"""
function substitute_future!(f::Future, old, new)
    if f === old || (isa(old, Type) && f isa old)
        return false  # Cannot mutate f to become new in-place
    end
    state = future_state(f)
    new_args = Any[replace_operand(arg, old, new) for arg in state.args]
    set_future_args!(f, new_args)
    return true
end

function evaluate_operand(arg; id=nothing, force=true)
    if arg isa Future
        return evaluate(arg; id=id, force=force)
    else
        return arg
    end
end

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

function operate(f::Future, ::Vector{Any})
    throw(ArgumentError("operate not implemented for $(typeof(f))"))
end

function evaluate(f::Future; id=nothing, force::Bool=true)
    state = future_state(f)

    if state.store_last && !force && id !== nothing && state.last_id == id
        return state.last_out
    end

    # Evaluate args into pre-allocated buffer (avoids allocating a new Any[] per call)
    buf = state._eval_buffer
    args = state.args
    n = length(args)
    if length(buf) != n
        resize!(buf, n)
    end
    @inbounds for i in 1:n
        buf[i] = evaluate_operand(args[i]; id=id, force=force)
    end
    result = operate(f, buf)

    if state.store_last && id !== nothing
        state.last_id = id
        state.last_out = result
    end

    return result
end

# ---------------------------------------------------------------------------
# NCC hooks - prepare and gather non-constant coefficient data
# ---------------------------------------------------------------------------

function prep_nccs(f::Future, vars)
    for arg in future_args(f)
        if arg isa Future
            prep_nccs(arg, vars)
        elseif applicable(prep_nccs, arg, vars)
            prep_nccs(arg, vars)
        end
    end
    return nothing
end

function gather_ncc_coeffs(f::Future)
    for arg in future_args(f)
        if arg isa Future
            gather_ncc_coeffs(arg)
        elseif applicable(gather_ncc_coeffs, arg)
            gather_ncc_coeffs(arg)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

function Base.show(io::IO, f::Future)
    args = future_args(f)
    print(io, "$(typeof(f))(", join(string.(args), ", "), ")")
end

# ============================================================================
# Exports
# ============================================================================

# Export types
export Future, FutureState

# Export state management functions
export future_state, future_args, original_args, future_metadata,
       set_future_args!, reset!, build_future_state

# Export tree traversal and substitution functions
export atoms, has_operand, replace_operand, substitute_future, substitute_future!

# Export evaluation functions
export evaluate_operand, operate, evaluate

# Export NCC hooks
export prep_nccs, gather_ncc_coeffs
