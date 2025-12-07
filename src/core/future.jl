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
    metadata::Dict{Symbol, Any}
end

function FutureState(args::Vector{Any}; out=nothing, store_last::Bool=true, metadata=Dict{Symbol, Any}())
    dist = unify_attributes(args, "dist")
    dtype = unify_attributes(args, "dtype")
    return FutureState(copy(args), copy(args), out, dist, dtype, store_last, nothing, nothing, metadata)
end

function build_future_state(args; name::Symbol, out=nothing, store_last::Bool=true)
    metadata = Dict{Symbol, Any}(:name => name)
    return FutureState(Vector{Any}(args); out=out, store_last=store_last, metadata=metadata)
end

# ---------------------------------------------------------------------------
# Helpers for working with futures
# ---------------------------------------------------------------------------

future_state(f::Future) = getfield(f, :state)

future_args(f::Future) = future_state(f).args
original_args(f::Future) = future_state(f).original_args
future_metadata(f::Future) = future_state(f).metadata

function set_future_args!(f::Future, args::Vector{Any})
    state = future_state(f)
    empty!(state.args)
    append!(state.args, args)
    return f
end

function reset!(f::Future)
    state = future_state(f)
    empty!(state.args)
    append!(state.args, state.original_args)
    return f
end

function atoms(f::Future, types...)
    """Collect leaf operands matching the provided types."""
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

function has_operand(f::Future, vars...)
    """Return true if any subtree matches the given operands/operators."""
    state = future_state(f)
    for arg in state.args
        if arg isa Future
            if has_operand(arg, vars...)
                return true
            end
        else
            for var in vars
                if isa(var, Type)
                    if arg isa var
                        return true
                    end
                elseif arg === var
                    return true
                end
            end
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
        return replace(arg, old, new)
    else
        return arg
    end
end

function replace(f::Future, old, new)
    if f === old || (isa(old, Type) && f isa old)
        return new
    end
    state = future_state(f)
    new_args = Any[replace_operand(arg, old, new) for arg in state.args]
    set_future_args!(f, new_args)
    return f
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

    evaluated_args = Any[evaluate_operand(arg; id=id, force=force) for arg in state.args]
    result = operate(f, evaluated_args)

    if state.store_last && id !== nothing
        state.last_id = id
        state.last_out = result
    end

    return result
end

# ---------------------------------------------------------------------------
# NCC hooks (placeholders for future expansion)
# ---------------------------------------------------------------------------

function prep_nccs(f::Future, vars)
    for arg in future_args(f)
        if arg isa Future
            prep_nccs(arg, vars)
        elseif hasfield(typeof(arg), :prep_nccs)
            arg.prep_nccs(vars)
        end
    end
end

function gather_ncc_coeffs(f::Future)
    for arg in future_args(f)
        if arg isa Future
            gather_ncc_coeffs(arg)
        elseif hasfield(typeof(arg), :gather_ncc_coeffs)
            arg.gather_ncc_coeffs()
        end
    end
end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

function Base.show(io::IO, f::Future)
    args = future_args(f)
    print(io, "$(typeof(f))(", join(string.(args), ", "), ")")
end
