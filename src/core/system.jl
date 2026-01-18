"""
Coefficient/field system helpers.

This module manages large contiguous coefficient buffers to
streamline pencil manipulations by storing contiguous views
for each subproblem/subsystem pair.
"""

struct CoeffSystem{T}
    data::Vector{T}
    views::Dict{Any, UnitRange{Int}}
end

function CoeffSystem(subproblems::Tuple{Vararg{Subproblem}}, dtype::DataType=ComplexF64)
    total = 0
    views = Dict{Any, UnitRange{Int}}()
    for sp in subproblems
        for ss in sp.subsystems
            coeff_size = subsystem_coeff_size(ss, sp)
            range = (total + 1):(total + coeff_size)
            views[(sp, ss)] = range
            total += coeff_size
        end
    end
    return CoeffSystem{dtype}(zeros(dtype, total), views)
end

subsystem_coeff_size(ss::Subsystem, sp::Subproblem) = ss.total_variable_size

function flatten_scalar_fields(vars)
    result = ScalarField[]
    for var in vars
        if isa(var, ScalarField)
            push!(result, var)
        elseif isa(var, VectorField)
            append!(result, var.components)
        elseif isa(var, TensorField)
            append!(result, vec(var.components))
        end
    end
    return result
end

function get_subdata(system::CoeffSystem, sp::Subproblem, ss::Subsystem)
    range = system.views[(sp, ss)]
    return view(system.data, range)
end

struct FieldSystem
    coeffs::CoeffSystem
    fields::Vector{ScalarField}
end

function FieldSystem(fields::Vector{ScalarField}, subproblems::Tuple{Vararg{Subproblem}})
    coeff_system = CoeffSystem(subproblems)
    return FieldSystem(coeff_system, fields)
end

function _resolve_subsystem(subproblem::Subproblem, subsystem::Union{Nothing, Subsystem})
    if subsystem === nothing
        if length(subproblem.subsystems) != 1
            throw(ArgumentError("Subproblem has $(length(subproblem.subsystems)) subsystems; pass a specific subsystem"))
        end
        return subproblem.subsystems[1]
    end
    if !(subsystem in subproblem.subsystems)
        throw(ArgumentError("Subsystem does not belong to the provided subproblem"))
    end
    return subsystem
end

function gather!(system::FieldSystem, subproblem::Subproblem; subsystem::Union{Nothing, Subsystem}=nothing)
    ss = _resolve_subsystem(subproblem, subsystem)
    view = get_subdata(system.coeffs, subproblem, ss)
    expected = length(view)
    needed = sum(length(_coeff_data(field)) for field in system.fields)
    if expected < needed
        throw(ArgumentError("CoeffSystem view length $expected is smaller than field data length $needed"))
    end
    offset = 0
    for field in system.fields
        data_vec = _field_coeff_vector(field)
        n = length(data_vec)
        copyto!(view, offset + 1, data_vec, 1, n)
        offset += n
    end
    return view
end

function scatter!(system::FieldSystem, subproblem::Subproblem, data::AbstractVector;
                  subsystem::Union{Nothing, Subsystem}=nothing)
    ss = _resolve_subsystem(subproblem, subsystem)
    view = get_subdata(system.coeffs, subproblem, ss)
    if length(data) != length(view)
        throw(ArgumentError("Scatter data length $(length(data)) does not match CoeffSystem view length $(length(view))"))
    end
    # Verify view is large enough for all field data
    needed = sum(length(_coeff_data(field)) for field in system.fields)
    if length(view) < needed
        throw(ArgumentError("CoeffSystem view length $(length(view)) is smaller than field data length $needed"))
    end
    copyto!(view, data)
    offset = 0
    for field in system.fields
        coeffs = _coeff_data(field)
        n = length(coeffs)
        slice = view[offset+1:offset+n]
        _assign_coefficients_from_slice!(field, coeffs, slice)
        offset += n
    end
    return nothing
end

# ============================================================================
# Exports
# ============================================================================

# Export types
export CoeffSystem, FieldSystem

# Export CoeffSystem functions
export subsystem_coeff_size, get_subdata

# Export FieldSystem functions
export flatten_scalar_fields, gather!, scatter!
