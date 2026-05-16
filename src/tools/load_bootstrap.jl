# Bootstrap utilities required by early core definitions.

include("general.jl")
include("exceptions.jl")
include("cache.jl")  # Must be before dispatch.jl.
include("dispatch.jl")
include("parsing.jl")
