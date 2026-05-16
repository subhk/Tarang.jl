# Public interface. Keep this file declarative: it defines the root-module
# compatibility surface, while implementation load order stays in Tarang.jl.

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
