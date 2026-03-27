"""
    Multiclass dispatch functions

This file contains the dispatch_preprocess, dispatch_check, and invoke_constructor
functions for each operator type. These enable the multiclass_new pattern for
operator construction.
"""

# ============================================================================
# Helper: Extract coordinate system from any Operand
# ============================================================================

"""
    _extract_coordsys(operand)

Recursively extract the coordinate system from an operand.
Fields (ScalarField, VectorField) have `dist.coordsys`.
Operators with `coordsys` field use it directly.
Other operators recurse into their `operand` field.
"""
function _extract_coordsys(op)
    # Fields have dist.coordsys
    if hasfield(typeof(op), :dist) && op.dist !== nothing
        return op.dist.coordsys
    end
    # Some operators store coordsys directly
    if hasfield(typeof(op), :coordsys) && getfield(op, :coordsys) !== nothing
        return getfield(op, :coordsys)
    end
    # Recurse into operator's operand
    if hasfield(typeof(op), :operand)
        return _extract_coordsys(getfield(op, :operand))
    end
    # Binary operators: try left operand
    if hasfield(typeof(op), :left)
        return _extract_coordsys(getfield(op, :left))
    end
    throw(ArgumentError("Cannot determine coordinate system from $(typeof(op))"))
end

# ============================================================================
# Operator Arithmetic (Basic operators for composition)
# ============================================================================

Base.:+(op1::Operator, op2::Operator) = AddOperator(op1, op2)
Base.:-(op1::Operator, op2::Operator) = SubtractOperator(op1, op2)
Base.:*(op1::Operator, op2::Operator) = MultiplyOperator(op1, op2)

# ============================================================================
# Gradient Dispatch
# ============================================================================

function dispatch_preprocess(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 1
        operand = args[1]
        coordsys = _extract_coordsys(operand)
        return ((operand, coordsys), kwargs)
    elseif length(args) == 2
        return (args, kwargs)
    else
        throw(ArgumentError("Gradient expects 1 or 2 arguments"))
    end
end

function dispatch_check(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, Operand)
        throw(ArgumentError("Gradient requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    operand, coordsys = args
    return _Gradient_constructor(operand, coordsys)
end

# ============================================================================
# Divergence Dispatch
# ============================================================================

function dispatch_check(::Type{Divergence}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !(isa(operand, VectorField) || isa(operand, Operator) || isa(operand, Future))
        throw(ArgumentError("Divergence requires a VectorField or an operator that produces one"))
    end
    return true
end

function invoke_constructor(::Type{Divergence}, args::Tuple, kwargs::NamedTuple)
    return _Divergence_constructor(args[1])
end

# ============================================================================
# Curl Dispatch
# ============================================================================

function dispatch_preprocess(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 1
        operand = args[1]
        coordsys = _extract_coordsys(operand)
        return ((operand, coordsys), kwargs)
    elseif length(args) == 2
        return (args, kwargs)
    else
        throw(ArgumentError("Curl expects 1 or 2 arguments"))
    end
end

function dispatch_check(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    operand, coordsys = args
    return _Curl_constructor(operand, coordsys)
end

# ============================================================================
# Laplacian Dispatch
# ============================================================================

function dispatch_check(::Type{Laplacian}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, Operand)
        throw(ArgumentError("Laplacian requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Laplacian}, args::Tuple, kwargs::NamedTuple)
    return _Laplacian_constructor(args[1])
end

# ============================================================================
# Trace Dispatch
# ============================================================================

function dispatch_check(::Type{Trace}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{Trace}, args::Tuple, kwargs::NamedTuple)
    return _Trace_constructor(args[1])
end

# ============================================================================
# Interpolate Dispatch
# ============================================================================

function dispatch_check(::Type{Interpolate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, position = args
    if !isa(operand, Operand)
        throw(ArgumentError("Interpolate requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Interpolate requires a Coordinate"))
    end
    if !isa(position, Real)
        throw(ArgumentError("Interpolate position must be real"))
    end
    return true
end

function invoke_constructor(::Type{Interpolate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, position = args
    return _Interpolate_constructor(operand, coord, position)
end

# ============================================================================
# Integrate Dispatch
# ============================================================================

function dispatch_check(::Type{Integrate}, args::Tuple, kwargs::NamedTuple)
    operand, coord = args
    if !isa(operand, Operand)
        throw(ArgumentError("Integrate requires an Operand"))
    end
    if !isa(coord, Coordinate) && !isa(coord, Tuple{Vararg{Coordinate}})
        throw(ArgumentError("Integrate requires a Coordinate or Tuple of Coordinates"))
    end
    return true
end

function invoke_constructor(::Type{Integrate}, args::Tuple, kwargs::NamedTuple)
    return _Integrate_constructor(args[1], args[2])
end

# ============================================================================
# Average Dispatch
# ============================================================================

function dispatch_check(::Type{Average}, args::Tuple, kwargs::NamedTuple)
    operand, coord = args
    if !isa(operand, Operand)
        throw(ArgumentError("Average requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Average requires a Coordinate"))
    end
    return true
end

function invoke_constructor(::Type{Average}, args::Tuple, kwargs::NamedTuple)
    return _Average_constructor(args[1], args[2])
end

# ============================================================================
# Differentiate Dispatch
# ============================================================================

function dispatch_preprocess(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 2
        operand, coord = args
        return ((operand, coord, 1), kwargs)
    elseif length(args) == 3
        return (args, kwargs)
    else
        throw(ArgumentError("Differentiate expects 2 or 3 arguments"))
    end
end

function dispatch_check(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, order = args
    if !isa(operand, Operand)
        throw(ArgumentError("Differentiate requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Differentiate requires a Coordinate"))
    end
    if !(order isa Integer) || order < 0
        throw(ArgumentError("Differentiate order must be a non-negative integer"))
    end
    return true
end

function invoke_constructor(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, order = args
    return _Differentiate_constructor(operand, coord, order)
end

# ============================================================================
# Convert Dispatch
# ============================================================================

function dispatch_check(::Type{Convert}, args::Tuple, kwargs::NamedTuple)
    operand, basis = args
    if !isa(operand, Operand)
        throw(ArgumentError("Convert requires an Operand"))
    end
    if !isa(basis, Basis)
        throw(ArgumentError("Convert requires a Basis"))
    end
    return true
end

function invoke_constructor(::Type{Convert}, args::Tuple, kwargs::NamedTuple)
    return _Convert_constructor(args[1], args[2])
end

# ============================================================================
# Grid and Coeff Dispatch
# ============================================================================

function dispatch_check(::Type{Grid}, args::Tuple, kwargs::NamedTuple)
    if !isa(args[1], Operand)
        throw(ArgumentError("Grid conversion requires an Operand"))
    end
    return true
end

function dispatch_check(::Type{Coeff}, args::Tuple, kwargs::NamedTuple)
    if !isa(args[1], Operand)
        throw(ArgumentError("Coeff conversion requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Grid}, args::Tuple, kwargs::NamedTuple)
    return _Grid_constructor(args[1])
end

function invoke_constructor(::Type{Coeff}, args::Tuple, kwargs::NamedTuple)
    return _Coeff_constructor(args[1])
end

# ============================================================================
# Skew Dispatch
# ============================================================================

function dispatch_check(::Type{Skew}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    # Accept any Operand - type checking happens at evaluation time
    if !isa(operand, Operand)
        throw(ArgumentError("Skew requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Skew}, args::Tuple, kwargs::NamedTuple)
    return _Skew_constructor(args[1])
end

# ============================================================================
# TransposeComponents Dispatch
# ============================================================================

function dispatch_preprocess(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    # Handle indices from kwargs
    indices = get(kwargs, :indices, (1, 2))
    if length(args) >= 2
        indices = args[2]
    end
    return ((args[1], indices), NamedTuple())
end

function dispatch_check(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("TransposeComponents requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    operand, indices = args
    return _TransposeComponents_constructor(operand, indices)
end

# ============================================================================
# Lift Dispatch
# ============================================================================

function dispatch_check(::Type{Lift}, args::Tuple, kwargs::NamedTuple)
    operand, basis, n = args
    if !isa(operand, Operand)
        throw(ArgumentError("Lift requires an Operand"))
    end
    if !isa(basis, Basis)
        throw(ArgumentError("Lift requires a Basis"))
    end
    if !(n isa Integer)
        throw(ArgumentError("Lift mode index must be an integer"))
    end
    return true
end

function invoke_constructor(::Type{Lift}, args::Tuple, kwargs::NamedTuple)
    operand, basis, n = args
    return _Lift_constructor(operand, basis, n)
end

# ============================================================================
# Component Dispatch
# ============================================================================

function dispatch_check(::Type{Component}, args::Tuple, kwargs::NamedTuple)
    operand, index = args
    if !isa(operand, Operand)
        throw(ArgumentError("Component extraction requires an Operand"))
    end
    if !(index isa Integer) || index < 1
        throw(ArgumentError("Component index must be a positive integer"))
    end
    return true
end

function invoke_constructor(::Type{Component}, args::Tuple, kwargs::NamedTuple)
    operand, index = args
    return _Component_constructor(operand, index)
end

# ============================================================================
# RadialComponent Dispatch
# ============================================================================

function dispatch_check(::Type{RadialComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{RadialComponent}, args::Tuple, kwargs::NamedTuple)
    return _RadialComponent_constructor(args[1])
end

# ============================================================================
# AngularComponent Dispatch
# ============================================================================

function dispatch_check(::Type{AngularComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{AngularComponent}, args::Tuple, kwargs::NamedTuple)
    return _AngularComponent_constructor(args[1])
end

# ============================================================================
# AzimuthalComponent Dispatch
# ============================================================================

function dispatch_check(::Type{AzimuthalComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{AzimuthalComponent}, args::Tuple, kwargs::NamedTuple)
    return _AzimuthalComponent_constructor(args[1])
end

# ============================================================================
# GeneralFunction Dispatch
# ============================================================================

function dispatch_check(::Type{GeneralFunction}, args::Tuple, kwargs::NamedTuple)
    operand, f, name = args
    if !isa(operand, Operand)
        throw(ArgumentError("GeneralFunction requires an Operand"))
    end
    if !isa(f, Function)
        throw(ArgumentError("GeneralFunction requires a Function"))
    end
    return true
end

function invoke_constructor(::Type{GeneralFunction}, args::Tuple, kwargs::NamedTuple)
    operand, f, name = args
    return _GeneralFunction_constructor(operand, f, name)
end
