"""
    Arithmetic operators

This file contains the arithmetic operator types and their associated
has() and require_linearity() methods.
"""

# ============================================================================
# Arithmetic Operator Types
# ============================================================================

struct AddOperator <: Operator
    left::Any
    right::Any
end

struct SubtractOperator <: Operator
    left::Any
    right::Any
end

struct MultiplyOperator <: Operator
    left::Any
    right::Any
end

struct DivideOperator <: Operator
    left::Any
    right::Any
end

struct PowerOperator <: Operator
    left::Any  # base
    right::Any # exponent
end

struct NegateOperator <: Operator
    operand::Any
end

struct IndexOperator <: Operator
    array::Any
    indices::Vector{Any}
end

# ============================================================================
# has() for arithmetic operators
# Following Dedalus arithmetic.py:147-150 (Add) and 331-352 (Multiply)
# ============================================================================

function has(op::AddOperator, vars...)
    return has(op.left, vars...) || has(op.right, vars...)
end

function has(op::SubtractOperator, vars...)
    return has(op.left, vars...) || has(op.right, vars...)
end

function has(op::MultiplyOperator, vars...)
    return has(op.left, vars...) || has(op.right, vars...)
end

function has(op::DivideOperator, vars...)
    return has(op.left, vars...) || has(op.right, vars...)
end

function has(op::PowerOperator, vars...)
    return has(op.left, vars...) || has(op.right, vars...)
end

function has(op::NegateOperator, vars...)
    return has(op.operand, vars...)
end

function has(op::IndexOperator, vars...)
    return has(op.array, vars...)
end

# ============================================================================
# require_linearity() for arithmetic operators
# ============================================================================

"""
    require_linearity(op::AddOperator, vars...; allow_affine=false)

For addition: all arguments must be linear.
Following Dedalus arithmetic.py:147-150.
"""
function require_linearity(op::AddOperator, vars...; allow_affine=false)
    require_linearity(op.left, vars...; allow_affine=allow_affine)
    require_linearity(op.right, vars...; allow_affine=allow_affine)
end

function require_linearity(op::SubtractOperator, vars...; allow_affine=false)
    require_linearity(op.left, vars...; allow_affine=allow_affine)
    require_linearity(op.right, vars...; allow_affine=allow_affine)
end

"""
    require_linearity(op::MultiplyOperator, vars...; allow_affine=false)

For multiplication: nonlinear ONLY if BOTH factors contain variables.
This is the key insight from Dedalus arithmetic.py:331-352.

- c * u is LINEAR (c is constant, u is variable)
- u * v is NONLINEAR (both are variables)
"""
function require_linearity(op::MultiplyOperator, vars...; allow_affine=false)
    left_has_vars = has(op.left, vars...)
    right_has_vars = has(op.right, vars...)

    if left_has_vars && right_has_vars
        # Both factors contain variables -> NONLINEAR
        error("Expression $(op) is nonlinear in variables: both factors contain problem variables")
    elseif left_has_vars
        # Only left has variables -> recurse on left
        require_linearity(op.left, vars...; allow_affine=allow_affine)
    elseif right_has_vars
        # Only right has variables -> recurse on right
        require_linearity(op.right, vars...; allow_affine=allow_affine)
    elseif !allow_affine
        # Neither has variables -> affine/constant
        error("Expression must be strictly linear, not just affine (constant)")
    end
end

"""
    require_linearity(op::DivideOperator, vars...; allow_affine=false)

Division: linear only if denominator doesn't contain variables.
"""
function require_linearity(op::DivideOperator, vars...; allow_affine=false)
    if has(op.right, vars...)
        error("Division by expression containing variables is nonlinear")
    end
    require_linearity(op.left, vars...; allow_affine=allow_affine)
end

"""
    require_linearity(op::PowerOperator, vars...; allow_affine=false)

Power: nonlinear if base contains variables (u^2, u^n are nonlinear).
"""
function require_linearity(op::PowerOperator, vars...; allow_affine=false)
    if has(op.left, vars...)
        error("Power of expression containing variables is nonlinear")
    end
end

"""
    require_linearity(op::NegateOperator, vars...; allow_affine=false)

Negation preserves linearity - delegate to operand.
"""
function require_linearity(op::NegateOperator, vars...; allow_affine=false)
    require_linearity(op.operand, vars...; allow_affine=allow_affine)
end

"""
    require_linearity(op::IndexOperator, vars...; allow_affine=false)

Indexing preserves linearity - delegate to array.
"""
function require_linearity(op::IndexOperator, vars...; allow_affine=false)
    require_linearity(op.array, vars...; allow_affine=allow_affine)
end
