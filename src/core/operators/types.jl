"""
    Operator type definitions

This file contains all abstract types and struct definitions for spectral operators.
These types form the foundation of the operator system in Tarang.jl.
"""

# ============================================================================
# Abstract Type
# ============================================================================

abstract type Operator <: Operand end

# ============================================================================
# Basic Differential Operators
# ============================================================================

struct Gradient <: Operator
    operand::Operand
    coordsys::CoordinateSystem
end
const _Gradient_constructor = Gradient

struct Divergence <: Operator
    operand::Operand
end
const _Divergence_constructor = Divergence

struct Curl <: Operator
    operand::Operand
    coordsys::CoordinateSystem
end
const _Curl_constructor = Curl

struct Laplacian <: Operator
    operand::Operand
end
const _Laplacian_constructor = Laplacian

# Fractional Laplacian: (-Delta)^alpha where alpha can be any real number
# In spectral space: (-Delta)^alpha f_hat(k) = |k|^(2*alpha) f_hat(k)
# Common values: alpha=1/2 for SQG dissipation, alpha=-1/2 for SQG streamfunction inversion
struct FractionalLaplacian <: Operator
    operand::Operand
    α::Float64  # Fractional exponent

    function FractionalLaplacian(operand::Operand, α::Real)
        new(operand, Float64(α))
    end
end
const _FractionalLaplacian_constructor = FractionalLaplacian

struct Trace <: Operator
    operand::Operand
end
const _Trace_constructor = Trace

struct Skew <: Operator
    operand::Operand
end
const _Skew_constructor = Skew

"""
    TransposeComponents <: Operator

Transpose tensor component indices.
Following Dedalus operators.py:1878-1981 StandardTransposeComponents.

Fields:
- operand: TensorField to transpose
- indices: Tuple of two indices to swap (default (1,2) for 0-indexed compatibility becomes (1,2))
- coordsys: Coordinate system of the tensor indices being swapped

Note: Dedalus uses 0-indexed (0,1) default; we use 1-indexed (1,2) for Julia convention.
"""
struct TransposeComponents <: Operator
    operand::Operand
    indices::Tuple{Int, Int}  # 1-indexed tensor indices to swap
    coordsys::Union{CoordinateSystem, Nothing}  # Coordinate system of swapped indices

    function TransposeComponents(operand::Operand, indices::Tuple{Int, Int}=(1, 2))
        # Validate operand is a tensor
        if !isa(operand, TensorField)
            throw(ArgumentError("TransposeComponents requires a TensorField"))
        end

        i0, i1 = indices

        # Validate indices are different
        if i0 == i1
            throw(ArgumentError("Cannot transpose same index with itself"))
        end

        # Validate indices are within tensor dimensions
        dim = size(operand.components, 1)
        if i0 < 1 || i0 > dim || i1 < 1 || i1 > dim
            throw(ArgumentError("TransposeComponents indices ($i0, $i1) out of range for $(dim)×$(dim) tensor"))
        end

        # Get coordinate system from tensor (assumes square tensor with same coordsys)
        coordsys = operand.coordsys

        new(operand, indices, coordsys)
    end
end
const _TransposeComponents_constructor = TransposeComponents

# ============================================================================
# Time Derivative
# ============================================================================

struct TimeDerivative <: Operator
    operand::Operand
    order::Int

    function TimeDerivative(operand::Operand, order::Int=1)
        new(operand, order)
    end
end

# ============================================================================
# Interpolation and Integration
# ============================================================================

struct Interpolate <: Operator
    operand::Operand
    coord::Coordinate
    position::Real
end
const _Interpolate_constructor = Interpolate

struct Integrate <: Operator
    operand::Operand
    coord::Union{Coordinate, Tuple{Vararg{Coordinate}}}
end
const _Integrate_constructor = Integrate

struct Average <: Operator
    operand::Operand
    coord::Coordinate
end
const _Average_constructor = Average

# ============================================================================
# Conversion Operators
# ============================================================================

struct Convert <: Operator
    operand::Operand
    basis::Basis
end
const _Convert_constructor = Convert

struct Grid <: Operator
    operand::Operand
end
const _Grid_constructor = Grid

struct Coeff <: Operator
    operand::Operand
end
const _Coeff_constructor = Coeff

# Lifting operator for tau method boundary conditions
# Following Dedalus Lift operator (operators.py:4264-4309, basis.py:790-814)
# Creates polynomial P with coefficient 1 at mode n, returns P * operand
struct Lift <: Operator
    operand::Operand  # Field to lift (typically tau variable)
    basis::Basis       # Output basis (where to place lifted field)
    n::Int            # Mode index (negative recommended: -1=last, -2=second-last)
end
const _Lift_constructor = Lift

# ============================================================================
# Component Extraction
# ============================================================================

struct Component <: Operator
    operand::Operand
    index::Int
end
const _Component_constructor = Component

struct RadialComponent <: Operator
    operand::Operand
end
const _RadialComponent_constructor = RadialComponent

struct AngularComponent <: Operator
    operand::Operand
end
const _AngularComponent_constructor = AngularComponent

struct AzimuthalComponent <: Operator
    operand::Operand
end
const _AzimuthalComponent_constructor = AzimuthalComponent

# ============================================================================
# Spectral Differentiation
# ============================================================================

struct Differentiate <: Operator
    operand::Operand
    coord::Coordinate
    order::Int
end
const _Differentiate_constructor = Differentiate

# ============================================================================
# Outer Product (Tensor Product)
# ============================================================================

struct Outer <: Operator
    left::Operand
    right::Operand
end
const _Outer_constructor = Outer

# ============================================================================
# AdvectiveCFL Operator
# ============================================================================

# AdvectiveCFL operator - computes grid-crossing frequency field
struct AdvectiveCFL <: Operator
    operand::Operand  # Velocity vector field
    coords::CoordinateSystem
end
const _AdvectiveCFL_constructor = AdvectiveCFL

# ============================================================================
# General Function Operators
# ============================================================================

"""
    GeneralFunction <: Operator

Apply general function to field in grid space.
"""
struct GeneralFunction <: Operator
    operand::Operand
    func::Function
    name::String
end
const _GeneralFunction_constructor = GeneralFunction

"""
    UnaryGridFunction <: Operator

Apply unary numpy-style function (sin, cos, exp, etc.) to field.
"""
struct UnaryGridFunction <: Operator
    operand::Operand
    func::Function
    name::String
end
const _UnaryGridFunction_constructor = UnaryGridFunction

# ============================================================================
# Copy Operator
# ============================================================================

"""
    Copy <: Operator

Deep-copy a field, producing an independent clone whose data can be
modified without affecting the original.
"""
struct Copy <: Operator
    operand::Operand
end
const _Copy_constructor = Copy

# ============================================================================
# Hilbert Transform Operator
# ============================================================================

"""
    HilbertTransform <: Operator

Apply the Hilbert transform in spectral space.
For Fourier modes: multiplies mode k by -i*sign(k) (k=0 maps to 0).
"""
struct HilbertTransform <: Operator
    operand::Operand
end
const _HilbertTransform_constructor = HilbertTransform
