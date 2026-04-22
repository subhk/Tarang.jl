"""
    Various operator operations

This file contains evaluation functions for:
- Interpolate, integrate, average, lift, convert
- GeneralFunction and UnaryGridFunction
- Grid and coeff conversion
- Component extraction (component, radial, angular, azimuthal)
"""


# Runtime map:
#   operations_interpolate.jl  — interpolation evaluation and Clenshaw helpers
#   operations_integrate.jl    — integration and averaging evaluation
#   operations_lift_convert.jl — lift and basis-conversion evaluation
#   operations_misc.jl         — general functions, layout conversion, components, copy, and Hilbert transform

include("operations/operations_interpolate.jl")
include("operations/operations_integrate.jl")
include("operations/operations_lift_convert.jl")
include("operations/operations_misc.jl")
