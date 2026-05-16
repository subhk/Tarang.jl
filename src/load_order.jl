# Package implementation load order.
#
# Keep this file declarative and ordered. Each loader owns a coherent domain
# slice while preserving the concrete include order used by the package.

include("core/load_contracts.jl")
include("tools/load_bootstrap.jl")
include("core/load_fields.jl")
include("core/load_problem_stack.jl")
include("tools/load_matsolvers.jl")
include("core/load_solver_stack.jl")
include("tools/load_output.jl")
include("core/load_evaluation.jl")
include("tools/load_runtime.jl")
include("core/load_models.jl")
include("extras/load_extras.jl")
include("tools/load_pretty_printing.jl")
