"""
    Operator registration system

This file contains the registration tables and functions for operator parsing.
These allow operators to be referenced by name in equation parsing.
"""

# ============================================================================
# Operator Registration Tables
# ============================================================================

const OPERATOR_ALIASES = Dict{String, Any}()
const OPERATOR_PARSEABLES = Dict{String, Any}()
const OPERATOR_PREFIXES = Dict{String, Any}()

# ============================================================================
# Registration Functions
# ============================================================================

function register_operator_alias!(op, names::AbstractString...)
    for name in names
        OPERATOR_ALIASES[name] = op
    end
    return op
end

function register_operator_parseable!(op, names::AbstractString...)
    for name in names
        OPERATOR_PARSEABLES[name] = op
    end
    return op
end

function register_operator_prefix!(op, names::AbstractString...)
    for name in names
        OPERATOR_PREFIXES[name] = op
    end
    return op
end

# ============================================================================
# Register Core Operators
# ============================================================================

# These registrations are called after constructors are defined.
# The actual registration calls are at the end of constructors.jl
