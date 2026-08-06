# Runtime utilities, configuration, logging, and post-processing tools.

include("config.jl")
include("array.jl")
include("parallel.jl")
include("logging.jl")
include("progress.jl")
include("random_arrays.jl")
include("netcdf_merge.jl")
# Layout/spectral reconstruction used by the merge paths above.
include("netcdf_merge_layout.jl")
include("netcdf_slab_io.jl")
# Field-level and solver-level NetCDF persistence. These sit ABOVE the slab layer
# and the merge readers included just above, which is why they cannot live under
# `core/` — see field_netcdf_io.jl's header and test/test_layering.jl.
include("field_netcdf_io.jl")
include("solver_checkpoint.jl")
include("temporal_filters.jl")
