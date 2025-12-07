"""
Coefficient/field system helpers.

This module manages large contiguous coefficient buffers to
streamline pencil manipulations with a simplified version that stores
contiguous views for each subproblem/subsystem pair.
"""

struct CoeffSystem
    data::Vector{ComplexF64}
    views::Dict{Any, UnitRange{Int}}
end

function CoeffSystem(subproblems::Tuple{Vararg{Subproblem}}, dtype::DataType=ComplexF64)
    total = 0
    views = Dict{Any, UnitRange{Int}}()
    for sp in subproblems
        for ss in sp.subsystems
            size = subsystem_coeff_size(ss, sp)
            range = (total + 1):(total + size)
            views[(sp, ss)] = range
            total += size
        end
    end
    return CoeffSystem(zeros(dtype, total), views)
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

function gather!(system::FieldSystem, subproblem::Subproblem)
    view = get_subdata(system.coeffs, subproblem, first(subproblem.subsystems))
    offset = 0
    for field in system.fields
        data = _coeff_data(field)
        n = length(data)
        view[offset+1:offset+n] = vec(data)
        offset += n
    end
    return view
end

function scatter!(system::FieldSystem, subproblem::Subproblem, data::AbstractVector)
    view = get_subdata(system.coeffs, subproblem, first(subproblem.subsystems))
    copyto!(view, data)
    offset = 0
    for field in system.fields
        coeffs = _coeff_data(field)
        n = length(coeffs)
        coeffs .= reshape(view[offset+1:offset+n], size(coeffs))
        offset += n
    end
    return nothing
end
