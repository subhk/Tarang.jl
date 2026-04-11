"""
    Operator utility functions

This file contains utility functions used by operators, including:
- fftfreq for computing FFT sample frequencies
- has() methods for checking if expressions contain variables
- require_linearity() methods for linearity checking
"""

# LinearAlgebra, SparseArrays already in Tarang.jl

# ============================================================================
# FFT Frequency Utility
# ============================================================================

"""
    _fftfreq(n, d=1.0)

Return the Discrete Fourier Transform sample frequencies.

The returned array contains the frequency bin centers in cycles per unit of the sample spacing.
For n even: [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)
For n odd:  [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)

Equivalent to numpy.fft.fftfreq / AbstractFFTs.fftfreq.
Named with underscore prefix to avoid shadowing AbstractFFTs.fftfreq.
"""
function _fftfreq(n::Int, d::Real=1.0)
    results = zeros(Float64, n)
    N = (n - 1) ÷ 2 + 1
    for i in 0:(N-1)
        results[i+1] = i / (d * n)
    end
    for i in N:(n-1)
        results[i+1] = (i - n) / (d * n)
    end
    return results
end

# ============================================================================
# has() methods - Check if expression contains variables
# See future.jl for reference
# ============================================================================

"""
    has(op::Operator, vars...) -> Bool

Determine if operator tree contains any of the specified operands/operators.
Following spectral pattern: recursively checks operands.
"""
# Generic implementation for operators with single operand
function has(op::Operator, vars...)
    # Check if this operator type itself is in vars
    if any(v -> op isa typeof(v), vars)
        return true
    end
    # Check operand if present
    if hasfield(typeof(op), :operand)
        return has(op.operand, vars...)
    end
    # Check left/right for binary operators
    if hasfield(typeof(op), :left) && hasfield(typeof(op), :right)
        return has(op.left, vars...) || has(op.right, vars...)
    end
    return false
end

# Specific implementations for operators with different field structures
function has(op::Outer, vars...)
    return has(op.left, vars...) || has(op.right, vars...)
end

# Note: has() definitions for ZeroOperator, ConstantOperator, UnknownOperator
# are in problems.jl after those types are defined

# ============================================================================
# require_linearity() - Linearity checking
# See arithmetic.jl for reference
# ============================================================================

"""
    require_linearity(op::Operator, vars...; allow_affine=false)

Require expression to be linear in specified variables.
Linear operators just delegate to their operand.
"""
function require_linearity(op::Operator, vars...; allow_affine=false)
    # Linear operators preserve linearity - just check operand
    if hasfield(typeof(op), :operand)
        require_linearity(op.operand, vars...; allow_affine=allow_affine)
    end
end

# Fields are trivially linear in themselves
function require_linearity(field::ScalarField, vars...; allow_affine=false)
    # A field by itself is linear
    return nothing
end

function require_linearity(field::VectorField, vars...; allow_affine=false)
    return nothing
end

function require_linearity(field::TensorField, vars...; allow_affine=false)
    return nothing
end

# Numbers are affine (constant)
function require_linearity(::Number, vars...; allow_affine=false)
    if !allow_affine
        error("Expression must be strictly linear, not just affine (constant)")
    end
end
