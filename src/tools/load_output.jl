# Output handlers that must be loaded before evaluator implementations.

# The group C-API layer first: netcdf_output.jl calls into it.
include("netcdf_group_api.jl")
include("netcdf_output.jl")
