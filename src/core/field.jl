"""
Field types and operations for Tarang.jl

Split into focused sub-files:
- field_types.jl: Type definitions (ScalarField, VectorField, TensorField)
- field_data.jl: Data access, allocation, component operations
- field_layout.jl: Layout transitions, transforms, field operations
- field_exports.jl: Export declarations
"""

include("field/field_types.jl")
include("field/field_data.jl")
include("field/field_layout.jl")
include("field/field_exports.jl")
