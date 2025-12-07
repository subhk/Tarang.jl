"""
Multiple-dispatch helpers.

Julia already provides multiple dispatch, but this module builds additional logic
around class-style hierarchies for ``MultiClass``/``CachedMultiClass`` behaviour.
"""

using InteractiveUtils: subtypes

export multiclass_new, cached_multiclass_new, dispatch_preprocess,
       dispatch_check, dispatch_postprocess, stop_dispatch

# ---------------------------------------------------------------------------
# Default hooks
# ---------------------------------------------------------------------------

dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where T = (args, kwargs)
dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) where T = true
dispatch_postprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where T = (args, kwargs)
stop_dispatch(::Type{T}) where T = false

# ---------------------------------------------------------------------------
# Core dispatch routine
# ---------------------------------------------------------------------------

function multiclass_new(cls::Type{T}, args...; kwargs...) where T
    arg_tuple = Tuple(args)
    kw_nt = (; kwargs...)

    try
        arg_tuple, kw_nt = dispatch_preprocess(cls, arg_tuple, kw_nt)
        subclasses = [sub for sub in subtypes(cls) if !stop_dispatch(sub)]

        if isempty(subclasses)
            if dispatch_check(cls, arg_tuple, kw_nt)
                return invoke_constructor(cls, arg_tuple, kw_nt)
            else
                throw(ArgumentError("Provided arguments do not satisfy dispatch check for $(cls)."))
            end
        end

        passlist = Type[]
        for subclass in subclasses
            if dispatch_check(subclass, arg_tuple, kw_nt)
                push!(passlist, subclass)
            end
        end

        if isempty(passlist)
            throw(ErrorException("No subclasses of $(cls) match the supplied arguments."))
        elseif length(passlist) > 1
            throw(ArgumentError("Degenerate subclasses of $(cls) match the supplied arguments: $(passlist)."))
        end

        subclass = first(passlist)
        arg_tuple, kw_nt = dispatch_postprocess(subclass, arg_tuple, kw_nt)
        return invoke_constructor(subclass, arg_tuple, kw_nt)

    catch err
        if err isa SkipDispatchException
            return err.output
        else
            rethrow()
        end
    end
end

function invoke_constructor(cls::Type, args::Tuple, kwargs::NamedTuple)
    return cls(args...; kwargs...)
end

# ---------------------------------------------------------------------------
# Cached variant
# ---------------------------------------------------------------------------

function cached_multiclass_new(cls::Type{T}, args...; kwargs...) where T
    cache = get!(cached_class_instances, cls) do
        CachedClass()
    end
    key = (Tuple(args), (; kwargs...))
    if haskey(cache.instances, key)
        inst = cache.instances[key]
        return inst
    else
        inst = multiclass_new(cls, args...; kwargs...)
        cache.instances[key] = inst
        return inst
    end
end
