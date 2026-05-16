# Package implementation load order.
#
# Keep this file declarative and ordered. Files are included into the root
# Tarang module; grouping here documents dependency order without mixing it
# with package imports, public exports, or runtime initialization.

# Architecture abstraction for CPU/GPU support.
include("core/architectures.jl")

# Shared contracts used by early-loaded files to reference later subsystems.
include("core/module_contracts.jl")

# Core utilities needed across modules.
include("tools/general.jl")
include("tools/exceptions.jl")
include("tools/cache.jl")  # Must be before dispatch.jl.
include("tools/dispatch.jl")
include("tools/parsing.jl")

# Coordinate systems.
include("core/coords.jl")

# Core modules.
include("core/basis.jl")
include("core/distributor.jl")
include("core/domain.jl")
include("core/field.jl")
include("core/field_pool.jl")
include("core/future.jl")
include("core/arithmetic.jl")
include("core/operators/operators.jl")
include("core/cartesian_operators.jl")
include("core/transforms.jl")
include("core/boundary_conditions.jl")
include("core/problems.jl")
include("core/subsystems.jl")
include("core/system.jl")
include("core/linalg.jl")
include("tools/matsolvers.jl")
include("tools/gpu_matsolvers.jl")
include("core/solvers.jl")
include("core/stochastic_forcing.jl")
include("core/timesteppers/timesteppers.jl")
include("core/timesteppers/step_subproblem_rk.jl")
include("core/timesteppers/step_subproblem_multistep.jl")
include("core/gpu_distributed.jl")
include("core/transposable_field.jl")

# NetCDF output must be included before evaluator.
include("tools/netcdf_output.jl")
include("core/evaluator.jl")
include("core/nonlinear.jl")

# Tools.
include("tools/config.jl")
include("tools/array.jl")
include("tools/parallel.jl")
include("tools/logging.jl")
include("tools/progress.jl")
include("tools/random_arrays.jl")
include("tools/netcdf_merge.jl")
include("tools/temporal_filters.jl")
include("core/les_models.jl")

# Extras.
include("extras/flow_tools.jl")
include("extras/plot_tools.jl")
include("extras/quick_domains.jl")
include("extras/analysis_tasks.jl")

# Pretty printing after all types are defined.
include("tools/pretty_printing.jl")
