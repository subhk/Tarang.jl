"""
    Field layout operations

This file contains layout transitions, field-level access helpers, arithmetic,
and shape/filter utilities for field types.
"""


# Runtime map:
#   field_layout_access.jl           — CPU/local data access and layout transitions
#   field_layout_operations.jl       — random fill, integration, and Vector/Tensor field API helpers
#   field_layout_arithmetic_io.jl    — scalar-field arithmetic and simple field I/O
#   field_layout_filters_shapes.jl   — spectral filtering and global/local shape helpers
#   field_layout_vectorized.jl       — vectorized array kernels, fast_axpy!, and unit-vector helpers

include("field_layout/field_layout_access.jl")
include("field_layout/field_layout_operations.jl")
include("field_layout/field_layout_arithmetic_io.jl")
include("field_layout/field_layout_filters_shapes.jl")
include("field_layout/field_layout_vectorized.jl")
