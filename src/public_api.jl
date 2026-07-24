# Public interface. Keep this file declarative: it defines the supported
# root-module surface, while implementation load order stays in Tarang.jl.

const _PUBLIC_API_NAMES = Set{Symbol}()

function _register_public_api!(names::Tuple)
    union!(_PUBLIC_API_NAMES, names)
    return nothing
end

"""
    @public_api name, ...

Export names and register them in Tarang's checked public-API manifest. Core
implementation files may retain legacy exports during the compatibility
window, but new supported API must be declared under `src/api/public/` with
this macro.
"""
macro public_api(names...)
    values = length(names) == 1 && names[1] isa Expr && names[1].head === :tuple ?
             names[1].args : collect(names)
    all(value -> value isa Symbol, values) ||
        error("@public_api accepts only bare symbol names")
    exported = Expr(:export, map(esc, values)...)
    registered = :(_register_public_api!($(Expr(:tuple, map(QuoteNode, values)...))))
    return Expr(:block, exported, registered)
end

"""Return the names covered by Tarang's supported public-API contract."""
public_api_names() = sort!(collect(_PUBLIC_API_NAMES); by=string)

"""Return whether `name` belongs to Tarang's supported public API."""
is_public_api(name::Union{Symbol, AbstractString}) = Symbol(name) in _PUBLIC_API_NAMES

include("api/public/quick_start.jl")
include("api/public/architecture.jl")
include("api/public/distributed_gpu.jl")
include("api/public/fields.jl")
include("api/public/operators.jl")
include("api/public/timesteppers.jl")
include("api/public/problems.jl")
include("api/public/diagnostics.jl")
include("api/public/output.jl")
include("api/public/forcing.jl")
include("api/public/filters_models.jl")
include("api/public/physics.jl")

@public_api public_api_names, is_public_api
