"""
    Operator constructor functions

This file contains the public constructor functions for all operators.
These are the primary user-facing API for creating operators.
"""

# ============================================================================
# Differential Operator Constructors
# ============================================================================

"""Gradient operator."""
function grad(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    return multiclass_new(Gradient, operand, coordsys)
end

"""Divergence operator."""
function divergence(operand::Operand)
    return multiclass_new(Divergence, operand)
end

# Alias for convenience (but users should prefer `divergence` to avoid confusion with Base.div)
const div_op = divergence

"""Curl operator."""
function curl(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    return multiclass_new(Curl, operand, coordsys)
end

"""Laplacian operator."""
function lap(operand::Operand)
    return multiclass_new(Laplacian, operand)
end

# ============================================================================
# Unicode Aliases for Differential Operators
# ============================================================================

# nabla for gradient
const ∇ = grad

# Delta for Laplacian
const Δ = lap

# nabla squared as alternate notation for Laplacian
const ∇² = lap

# ============================================================================
# Fractional Laplacian Constructors
# ============================================================================

"""
    fraclap(operand, alpha)

Fractional Laplacian operator: (-Delta)^alpha

In spectral space, this multiplies by |k|^(2*alpha) where k is the wavenumber magnitude.

# Common use cases:
- `fraclap(theta, 0.5)` or `(-Delta)^(1/2)`: SQG dissipation
- `fraclap(theta, -0.5)` or `(-Delta)^(-1/2)`: SQG streamfunction inversion (psi = (-Delta)^(-1/2) theta)
- `fraclap(f, 1.0)`: Standard Laplacian (equivalent to `lap(f)`)

# Example - SQG equation:
```julia
# d(theta)/dt + u*grad(theta) = kappa*(-Delta)^(1/2)*theta
add_equation!(problem, "dt(theta) = -u*grad(theta) + kappa*fraclap(theta, 0.5)")

# Streamfunction from buoyancy: psi = (-Delta)^(-1/2)*theta
psi = fraclap(theta, -0.5)
```
"""
function fraclap(operand::Operand, α::Real)
    return FractionalLaplacian(operand, α)
end

# Unicode alias for fractional Laplacian with specific exponents
# Note: For general alpha, use fraclap(f, alpha) or Delta_alpha(f, alpha)
"""
    Δᵅ(operand, alpha)

Unicode alias for fractional Laplacian. Equivalent to `fraclap(operand, alpha)`.
Type: \\Delta Tab \\_\\alpha Tab
"""
const Δᵅ = fraclap

# Square root Laplacian: (-Delta)^(1/2), common in SQG
"""
    sqrtlap(operand)

Square root Laplacian: (-Delta)^(1/2)

Equivalent to `fraclap(operand, 0.5)`.
Common in Surface Quasi-Geostrophic (SQG) equations for dissipation.
"""
sqrtlap(operand::Operand) = fraclap(operand, 0.5)

# Inverse square root Laplacian: (-Delta)^(-1/2), for SQG streamfunction
"""
    invsqrtlap(operand)

Inverse square root Laplacian: (-Delta)^(-1/2)

Equivalent to `fraclap(operand, -0.5)`.
Used in SQG to compute streamfunction from buoyancy: psi = (-Delta)^(-1/2)*theta
"""
invsqrtlap(operand::Operand) = fraclap(operand, -0.5)

# ============================================================================
# Hyperviscosity / Higher-Order Laplacian Operators
# ============================================================================

"""
    hyperlap(operand, n::Integer)

Higher-order Laplacian (hyperviscosity) operator: (-Delta)^n

In Fourier space: (-Delta)^n f_hat(k) = |k|^(2n) f_hat(k)

Common use cases:
- `hyperlap(u, 2)` or `Delta2(u)`: Biharmonic operator for 4th-order hyperviscosity
- `hyperlap(u, 4)` or `Delta4(u)`: 8th-order hyperviscosity
- `hyperlap(u, 8)` or `Delta8(u)`: 16th-order hyperviscosity

For turbulence simulations with hyperviscosity dissipation:
```julia
# 4th-order hyperviscosity (biharmonic)
add_equation!(problem, "dt(u) = -u*grad(u) - nu4*Delta2(u)")

# 8th-order hyperviscosity
add_equation!(problem, "dt(u) = -u*grad(u) - nu8*Delta4(u)")
```

Note: For Fourier-based spectral methods, hyperviscosity is efficient because
it's just multiplication by |k|^(2n) in spectral space. For non-periodic
bases (Chebyshev), higher-order operators require more tau corrections.
"""
function hyperlap(operand::Operand, n::Integer)
    n >= 1 || throw(ArgumentError("hyperlap order n must be >= 1, got $n"))
    return FractionalLaplacian(operand, Float64(n))
end

# Unicode shortcuts for common hyperviscosity orders
# In spectral space: Delta^n computes |k|^(2n) (same as (-Delta)^n for integer n>=1)

"""
    Δ²(operand)

Biharmonic operator: (-Delta)^2 = |k|^4 in Fourier space.

Commonly used for 4th-order hyperviscosity in turbulence simulations:
```julia
add_equation!(problem, "dt(omega) = -u*grad(omega) - nu4*Delta2(omega)")
```

Type: \\Delta Tab \\^2 Tab
"""
Δ²(operand::Operand) = hyperlap(operand, 2)

"""
    Δ⁴(operand)

4th power Laplacian: (-Delta)^4 = |k|^8 in Fourier space.

Used for 8th-order hyperviscosity:
```julia
add_equation!(problem, "dt(omega) = -u*grad(omega) - nu8*Delta4(omega)")
```

Type: \\Delta Tab \\^4 Tab
"""
Δ⁴(operand::Operand) = hyperlap(operand, 4)

"""
    Δ⁶(operand)

6th power Laplacian: (-Delta)^6 = |k|^12 in Fourier space.

Used for 12th-order hyperviscosity.

Type: \\Delta Tab \\^6 Tab
"""
Δ⁶(operand::Operand) = hyperlap(operand, 6)

"""
    Δ⁸(operand)

8th power Laplacian: (-Delta)^8 = |k|^16 in Fourier space.

Used for 16th-order hyperviscosity for very high Reynolds number simulations.

Type: \\Delta Tab \\^8 Tab
"""
Δ⁸(operand::Operand) = hyperlap(operand, 8)

# ============================================================================
# Tensor Operator Constructors
# ============================================================================

"""Trace operator."""
function trace(operand::Operand)
    return multiclass_new(Trace, operand)
end

"""Skew-symmetric part of tensor."""
function skew(operand::Operand)
    return multiclass_new(Skew, operand)
end

"""
    transpose_components(operand; indices=(1,2))

Transpose tensor components, swapping specified indices.
Following Dedalus operators.py:1878-1981.

Arguments:
- operand: TensorField to transpose
- indices: Tuple of two tensor indices to swap (default: (1,2) for rank-2 transpose)

Example:
```julia
# Transpose a rank-2 tensor: T_ij -> T_ji
T_transposed = transpose_components(T)

# Or explicitly specify indices
T_transposed = transpose_components(T, indices=(1,2))
```
"""
function transpose_components(operand::Operand; indices::Tuple{Int,Int}=(1, 2))
    return multiclass_new(TransposeComponents, operand, indices)
end

# ============================================================================
# Interpolation and Integration Constructors
# ============================================================================

"""Interpolate operand along a coordinate at a given position."""
function interpolate(operand::Operand, coord::Coordinate, position::Real)
    return multiclass_new(Interpolate, operand, coord, position)
end

"""Integrate operand over the specified coordinate(s)."""
function integrate(operand::Operand, coord::Union{Coordinate, Tuple{Vararg{Coordinate}}})
    return multiclass_new(Integrate, operand, coord)
end

"""
    integrate(operand, coordsys::CartesianCoordinates)

Integrate operand over all coordinates in a Cartesian coordinate system.
Following Dedalus operators.py:1167-1168 - splits CartesianCoordinates into individual coords.
"""
function integrate(operand::Operand, coordsys::CartesianCoordinates)
    # Split Cartesian coordinates into tuple of individual coordinates
    coords_tuple = tuple(coordsys.coords...)
    return integrate(operand, coords_tuple)
end

"""
    integrate(operand, coordsys::DirectProduct)

Integrate operand over all coordinates in a DirectProduct coordinate system.
Following Dedalus operators.py:1170-1171 - splits DirectProduct into coordinate systems.
"""
function integrate(operand::Operand, coordsys::DirectProduct)
    # Split DirectProduct into tuple of all individual coordinates
    all_coords = tuple(coordsys.coords...)
    return integrate(operand, all_coords)
end

"""Average operand along a coordinate."""
function average(operand::Operand, coord::Coordinate)
    return multiclass_new(Average, operand, coord)
end

"""
    average(operand, coordsys::CartesianCoordinates)

Average operand over all coordinates in a Cartesian coordinate system.
Following Dedalus operators.py:1240-1241 - splits CartesianCoordinates and recurses.
"""
function average(operand::Operand, coordsys::CartesianCoordinates)
    # Recursively average over each coordinate
    result = operand
    for coord in coordsys.coords
        result = average(result, coord)
    end
    return result
end

"""
    average(operand, coordsys::DirectProduct)

Average operand over all coordinates in a DirectProduct coordinate system.
Following Dedalus operators.py:1243-1244 - splits DirectProduct and recurses.
"""
function average(operand::Operand, coordsys::DirectProduct)
    # Recursively average over each coordinate
    result = operand
    for coord in coordsys.coords
        result = average(result, coord)
    end
    return result
end

"""
    average(operand, coords::Tuple{Vararg{Coordinate}})

Average operand over multiple coordinates.
"""
function average(operand::Operand, coords::Tuple{Vararg{Coordinate}})
    result = operand
    for coord in coords
        result = average(result, coord)
    end
    return result
end

# ============================================================================
# Conversion Constructors
# ============================================================================

"""Convert operand to a different basis."""
function convert_basis(operand::Operand, basis::Basis)
    return multiclass_new(Convert, operand, basis)
end

"""Time derivative."""
function dt(operand::Operand, order::Int=1)
    return TimeDerivative(operand, order)
end

# partial_t for time derivative (Unicode alias)
const ∂t = dt

# ============================================================================
# Differentiation Constructors
# ============================================================================

"""Differentiate with respect to coordinate."""
function d(operand::Operand, coord::Coordinate, order::Int=1)
    return multiclass_new(Differentiate, operand, coord, order)
end

# ============================================================================
# Lift Constructor
# ============================================================================

"""
    lift(operand, basis, n)

Apply lifting operator for tau method boundary conditions.
Following Dedalus Lift operator (operators.py:4264-4309, basis.py:790-814).

Creates a polynomial P on the output basis with coefficient 1 at mode n,
then returns P * operand. This "lifts" the operand (typically a tau variable)
into spectral space at the specified mode.

Arguments:
- operand: The field to lift (typically a tau variable)
- basis: The output basis (where to place the lifted field)
- n: Mode index. Convention:
  - n < 0: Negative indexing from end (recommended for tau method)
    - n = -1: last mode (N)
    - n = -2: second-to-last mode (N-1)
  - n >= 0: Direct 0-indexed mode (less common)

Note: Dedalus only allows n < 0. We allow positive n for flexibility,
but negative indices are recommended for tau method boundary conditions.

Example:
```julia
# Typical tau method usage: lift tau to highest modes
tau = ScalarField(dist, "tau", ())  # Tau variable (no spectral basis)
lift_term = lift(tau, chebyshev_basis, -1)  # Lift to last Chebyshev mode
lift_term2 = lift(tau, chebyshev_basis, -2)  # Lift to second-to-last mode
```
"""
function lift(operand::Operand, basis::Basis, n::Int)
    return multiclass_new(Lift, operand, basis, n)
end

# ============================================================================
# Component Extraction Constructors
# ============================================================================

"""Extract component from vector/tensor field."""
function component(operand::Operand, index::Int)
    return multiclass_new(Component, operand, index)
end

"""Extract radial component from vector field."""
function radial(operand::Operand)
    return multiclass_new(RadialComponent, operand)
end

"""Extract angular component from vector field."""
function angular(operand::Operand)
    return multiclass_new(AngularComponent, operand)
end

"""Extract azimuthal component from vector field."""
function azimuthal(operand::Operand)
    return multiclass_new(AzimuthalComponent, operand)
end

# ============================================================================
# Grid/Coeff Conversion Constructors
# ============================================================================

"""Convert operand to grid space."""
function grid(operand::Operand)
    return multiclass_new(Grid, operand)
end

"""Convert operand to coefficient space."""
function coeff(operand::Operand)
    return multiclass_new(Coeff, operand)
end

# ============================================================================
# Outer Product Constructor
# ============================================================================

"""
    outer(left, right)

Outer product (tensor product) of two vector fields.

Returns a rank-2 tensor field T where T_ij = u_i * v_j.

# Arguments
- `left`: First vector field (u)
- `right`: Second vector field (v)

# Returns
- TensorField with components T_ij = u_i * v_j
"""
function outer(left::Operand, right::Operand)
    return multiclass_new(Outer, left, right)
end

# ============================================================================
# AdvectiveCFL Constructor
# ============================================================================

"""
    advective_cfl(velocity, coords)

Compute advective CFL grid-crossing frequency field.

Returns a scalar field representing the local grid-crossing frequency:
f = |u|/dx + |v|/dy + |w|/dz

This is useful for adaptive timestepping where dt < 1/max(f).

# Arguments
- `velocity`: Vector field representing fluid velocity
- `coords`: Coordinate system for the domain

# Returns
- ScalarField with CFL frequency at each grid point
"""
function advective_cfl(velocity::Operand, coords::CoordinateSystem)
    return multiclass_new(AdvectiveCFL, velocity, coords)
end

# Alias for convenience
const cfl = advective_cfl

# ============================================================================
# General Function Constructor
# ============================================================================

"""
    apply_function(operand::Operand, f::Function, name::String="func")

Apply arbitrary function to operand in grid space.
"""
function apply_function(operand::Operand, f::Function, name::String="func")
    return multiclass_new(GeneralFunction, operand, f, name)
end

# ============================================================================
# Unary Grid Functions
# ============================================================================

# Common unary functions
function sin_field(operand::Operand)
    return UnaryGridFunction(operand, sin, "sin")
end

function cos_field(operand::Operand)
    return UnaryGridFunction(operand, cos, "cos")
end

function tan_field(operand::Operand)
    return UnaryGridFunction(operand, tan, "tan")
end

function exp_field(operand::Operand)
    return UnaryGridFunction(operand, exp, "exp")
end

function log_field(operand::Operand)
    return UnaryGridFunction(operand, log, "log")
end

function sqrt_field(operand::Operand)
    return UnaryGridFunction(operand, sqrt, "sqrt")
end

function abs_field(operand::Operand)
    return UnaryGridFunction(operand, abs, "abs")
end

function tanh_field(operand::Operand)
    return UnaryGridFunction(operand, tanh, "tanh")
end

# ============================================================================
# Copy Operator Constructor
# ============================================================================

"""
    copy_field(operand::Operand)

Create an independent deep copy of a field. Modifications to the copy
do not affect the original.
"""
function copy_field(operand::Operand)
    return Copy(operand)
end

# ============================================================================
# Hilbert Transform Constructor
# ============================================================================

"""
    hilbert(operand::Operand)

Apply the Hilbert transform to a field.

For ComplexFourier: multiplies mode k by -i*sign(k) (k=0 maps to 0).
For RealFourier (interleaved cos/sin): H[cos(nx)] = sin(nx), H[sin(nx)] = -cos(nx).

Properties: H[H[f]] = -f for zero-mean f.
"""
function hilbert(operand::Operand)
    return HilbertTransform(operand)
end

# ============================================================================
# Operator Registration Calls
# ============================================================================

# Register core operators for parsing namespace consistency
register_operator_alias!(grad, "grad", "gradient")
register_operator_parseable!(grad, "grad", "gradient")

register_operator_alias!(divergence, "div", "divergence")
register_operator_parseable!(divergence, "div", "divergence")

register_operator_alias!(curl, "curl")
register_operator_parseable!(curl, "curl")

register_operator_alias!(lap, "lap", "laplacian")
register_operator_parseable!(lap, "lap", "laplacian")

register_operator_alias!(trace, "trace")
register_operator_parseable!(trace, "trace")

register_operator_alias!(skew, "skew")
register_operator_parseable!(skew, "skew")

register_operator_alias!(transpose_components, "transpose_components", "transpose", "trans")
register_operator_parseable!(transpose_components, "transpose_components", "transpose", "trans")

register_operator_alias!(interpolate, "interpolate")
register_operator_parseable!(interpolate, "interpolate")

register_operator_alias!(integrate, "integrate", "integ")
register_operator_parseable!(integrate, "integrate", "integ")

register_operator_alias!(average, "average", "avg", "ave")
register_operator_parseable!(average, "average", "avg", "ave")

register_operator_alias!(convert_basis, "convert")
register_operator_parseable!(convert_basis, "convert")

register_operator_alias!(dt, "dt")
register_operator_parseable!(dt, "dt")

register_operator_alias!(d, "d", "differentiate")
register_operator_parseable!(d, "d", "differentiate")

register_operator_alias!(lift, "lift")
register_operator_parseable!(lift, "lift")

register_operator_alias!(component, "component", "comp")
register_operator_parseable!(component, "component", "comp")

register_operator_alias!(radial, "radial")
register_operator_parseable!(radial, "radial")

register_operator_alias!(angular, "angular")
register_operator_parseable!(angular, "angular")

register_operator_alias!(azimuthal, "azimuthal")
register_operator_parseable!(azimuthal, "azimuthal")

register_operator_alias!(grid, "grid")
register_operator_parseable!(grid, "grid")

register_operator_alias!(coeff, "coeff")
register_operator_parseable!(coeff, "coeff")

register_operator_alias!(outer, "outer", "tensor_product")
register_operator_parseable!(outer, "outer", "tensor_product")

register_operator_alias!(advective_cfl, "advective_cfl", "cfl")
register_operator_parseable!(advective_cfl, "advective_cfl", "cfl")

# Register fractional/hyperviscosity operators for equation parsing
register_operator_alias!(fraclap, "fraclap", "fractional_laplacian")
register_operator_parseable!(fraclap, "fraclap", "fractional_laplacian")
register_operator_parseable!(sqrtlap, "sqrtlap")
register_operator_parseable!(invsqrtlap, "invsqrtlap")
register_operator_alias!(hyperlap, "hyperlap")
register_operator_parseable!(hyperlap, "hyperlap")
register_operator_parseable!(Δ², "Delta2")
register_operator_parseable!(Δ⁴, "Delta4")
register_operator_parseable!(Δ⁶, "Delta6")
register_operator_parseable!(Δ⁸, "Delta8")

# Register general function
register_operator_alias!(apply_function, "apply_function", "apply_func")
register_operator_parseable!(apply_function, "apply_function", "apply_func")

# Register unary functions
register_operator_parseable!(sin_field, "sin")
register_operator_parseable!(cos_field, "cos")
register_operator_parseable!(tan_field, "tan")
register_operator_parseable!(exp_field, "exp")
register_operator_parseable!(log_field, "log")
register_operator_parseable!(sqrt_field, "sqrt")
register_operator_parseable!(abs_field, "abs")
register_operator_parseable!(tanh_field, "tanh")

# Register copy and Hilbert transform operators
register_operator_alias!(copy_field, "copy", "copy_field")
register_operator_parseable!(copy_field, "copy", "copy_field")

register_operator_alias!(hilbert, "hilbert", "HilbertTransform")
register_operator_parseable!(hilbert, "hilbert", "HilbertTransform")
