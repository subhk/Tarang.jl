"""
    Field data operations

This file contains data access, allocation, scaling, and storage helpers for
field types.
"""


# Runtime map:
#   field_data_access_locked.jl     — `has`, LockedField, and access/property shims
#   field_data_component_buffers.jl — stacked component-buffer helpers
#   field_data_copy_alloc.jl        — copy/deepcopy, raw storage accessors, and allocation
#   field_data_distributor_utils.jl — local/global size and index helpers
#   field_data_scales.jl            — scale changes, resampling, and local-data helpers

include("field_data/field_data_access_locked.jl")
include("field_data/field_data_component_buffers.jl")
include("field_data/field_data_copy_alloc.jl")
include("field_data/field_data_distributor_utils.jl")
include("field_data/field_data_scales.jl")
