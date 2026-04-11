"""
Analysis task helpers for Tarang.

Note: The helper functions add_mean_task!, add_slice_task!, and add_profile_task!
are defined in src/tools/netcdf_output.jl to avoid method redefinition errors.
They support layout and scales kwargs.

This file is kept for backwards compatibility with any code that may reference it.
"""

# Functions are defined in netcdf_output.jl:
# - add_mean_task!(handler, field; dims, name, layout, scales)
# - add_slice_task!(handler, field; slices, dim, idx, name, layout, scales)
# - add_profile_task!(handler, field; dim, name, layout, scales)
