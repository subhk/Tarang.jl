"""
    Operator type definitions

This file contains all abstract types and struct definitions for spectral operators.
These types form the foundation of the operator system in Tarang.jl.
"""

abstract type Operator <: Operand end

# ============================================================================
# Basic Differential Operators
# ============================================================================

struct Gradient{O<:Operand} <: Operator
    operand::O
    coordsys::CoordinateSystem
end
const _Gradient_constructor = Gradient

struct Divergence{O<:Operand} <: Operator
    operand::O
end
const _Divergence_constructor = Divergence

struct Curl{O<:Operand} <: Operator
    operand::O
    coordsys::CoordinateSystem
end
const _Curl_constructor = Curl

struct Laplacian{O<:Operand} <: Operator
    operand::O
end
const _Laplacian_constructor = Laplacian

# Fractional Laplacian: (-Delta)^alpha where alpha can be any real number
# In spectral space: (-Delta)^alpha f_hat(k) = |k|^(2*alpha) f_hat(k)
# Common values: alpha=1/2 for SQG dissipation, alpha=-1/2 for SQG streamfunction inversion
struct FractionalLaplacian{O<:Operand} <: Operator
    operand::O
    α::Float64  # Fractional exponent

    function FractionalLaplacian(operand::O, α::Real) where {O<:Operand}
        new{O}(operand, Float64(α))
    end
end
const _FractionalLaplacian_constructor = FractionalLaplacian

struct Trace{O<:Operand} <: Operator
    operand::O
end
const _Trace_constructor = Trace

struct Skew{O<:Operand} <: Operator
    operand::O
end
const _Skew_constructor = Skew

"""
    TransposeComponents <: Operator

Transpose tensor component indices.
Following spectral methods pattern StandardTransposeComponents.

Fields:
- operand: TensorField to transpose
- indices: Tuple of two indices to swap (default (1,2) for 0-indexed compatibility becomes (1,2))
- coordsys: Coordinate system of the tensor indices being swapped

Note: uses 0-indexed (0,1) default; we use 1-indexed (1,2) for Julia convention.
"""
struct TransposeComponents{O<:Operand} <: Operator
    operand::O
    indices::Tuple{Int, Int}  # 1-indexed tensor indices to swap
    coordsys::Union{CoordinateSystem, Nothing}  # Coordinate system of swapped indices

    function TransposeComponents(operand::O, indices::Tuple{Int, Int}=(1, 2)) where {O<:Operand}
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

        new{O}(operand, indices, coordsys)
    end
end
const _TransposeComponents_constructor = TransposeComponents

# ============================================================================
# Time Derivative
# ============================================================================

struct TimeDerivative{O<:Operand} <: Operator
    operand::O
    order::Int

    function TimeDerivative(operand::O, order::Int=1) where {O<:Operand}
        new{O}(operand, order)
    end
end

# ============================================================================
# Interpolation and Integration
# ============================================================================

struct Interpolate{O<:Operand} <: Operator
    operand::O
    coord::Coordinate
    position::Float64

    function Interpolate(operand::O, coord::Coordinate, position::Real) where {O<:Operand}
        new{O}(operand, coord, Float64(position))
    end
end
const _Interpolate_constructor = Interpolate

struct Integrate{O<:Operand} <: Operator
    operand::O
    coord::Union{Coordinate, Tuple{Vararg{Coordinate}}}
end
const _Integrate_constructor = Integrate

struct Average{O<:Operand} <: Operator
    operand::O
    coord::Coordinate
end
const _Average_constructor = Average

# ============================================================================
# Conversion Operators
# ============================================================================

struct Convert{O<:Operand} <: Operator
    operand::O
    basis::Basis
end
const _Convert_constructor = Convert

struct Grid{O<:Operand} <: Operator
    operand::O
end
const _Grid_constructor = Grid

struct Coeff{O<:Operand} <: Operator
    operand::O
end
const _Coeff_constructor = Coeff

# Lifting operator for tau method boundary conditions
# Following spectral methods pattern, basis.py:790-814)
# Creates polynomial P with coefficient 1 at mode n, returns P * operand
struct Lift{O<:Operand} <: Operator
    operand::O   # Field to lift (typically tau variable)
    basis::Basis  # Output basis (where to place lifted field)
    n::Int        # Mode index (negative recommended: -1=last, -2=second-last)
end
const _Lift_constructor = Lift

# ============================================================================
# Component Extraction
# ============================================================================

struct Component{O<:Operand} <: Operator
    operand::O
    index::Int
end
const _Component_constructor = Component

struct RadialComponent{O<:Operand} <: Operator
    operand::O
end
const _RadialComponent_constructor = RadialComponent

struct AngularComponent{O<:Operand} <: Operator
    operand::O
end
const _AngularComponent_constructor = AngularComponent

struct AzimuthalComponent{O<:Operand} <: Operator
    operand::O
end
const _AzimuthalComponent_constructor = AzimuthalComponent

# ============================================================================
# Spectral Differentiation
# ============================================================================

struct Differentiate{O<:Operand} <: Operator
    operand::O
    coord::Coordinate
    order::Int
end
const _Differentiate_constructor = Differentiate

# ============================================================================
# Outer Product (Tensor Product)
# ============================================================================

struct Outer{L<:Operand, R<:Operand} <: Operator
    left::L
    right::R
end
const _Outer_constructor = Outer

# ============================================================================
# AdvectiveCFL Operator
# ============================================================================

# AdvectiveCFL operator - computes grid-crossing frequency field
struct AdvectiveCFL{O<:Operand} <: Operator
    operand::O  # Velocity vector field
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
struct GeneralFunction{F, O<:Operand} <: Operator
    operand::O
    func::F
    name::String
end
const _GeneralFunction_constructor = GeneralFunction

"""
    UnaryGridFunction <: Operator

Apply unary numpy-style function (sin, cos, exp, etc.) to field.
"""
struct UnaryGridFunction{F, O<:Operand} <: Operator
    operand::O
    func::F
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
struct Copy{O<:Operand} <: Operator
    operand::O
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
struct HilbertTransform{O<:Operand} <: Operator
    operand::O
end
const _HilbertTransform_constructor = HilbertTransform
