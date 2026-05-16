"""
Tarang.jl - Spectral PDE framework for Julia

"""

module Tarang

include("dependencies.jl")
include("load_order.jl")
include("public_api.jl")
include("api/namespaces.jl")
include("runtime_init.jl")

end # module
