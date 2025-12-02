"""
Lightweight subsystem helpers inspired by ``dedalus/core/subsystems.py``.

The full Dedalus implementation performs sophisticated pencil management,
preconditioning, and MPI-aware matrix assembly.  Tarang currently lacks that
machinery, so this module provides a minimal structure that matches the
expected API shape while operating on the complete domain as a single
subsystem.  This allows us to incrementally port solver features without
blocking on the full subsystem rewrite.
"""

using SparseArrays

# ---------------------------------------------------------------------------
# Subsystem construction
# ---------------------------------------------------------------------------

const SUBSYSTEM_GROUP = (:global,)

struct Subsystem
    solver::Any
    problem::Problem
    dist::Distributor
    dtype::DataType
    group::Tuple
    scalar_ranges::Dict{ScalarField, UnitRange{Int}}
    variable_ranges::Dict{Any, UnitRange{Int}}
    equation_ranges::Dict{Int, UnitRange{Int}}
    total_variable_size::Int
    total_equation_size::Int
end

scalar_components(field::ScalarField) = [field]
scalar_components(vec::VectorField) = vec.components
scalar_components(tensor::TensorField) = collect(vec(tensor.components))

function scalar_field_dofs(field::ScalarField)
    if field.data_c !== nothing
        return length(field.data_c)
    elseif field.data_g !== nothing
        return length(field.data_g)
    else
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    end
end

function infer_problem_dtype(problem::Problem)
    for var in problem.variables
        for comp in scalar_components(var)
            return comp.dtype
        end
    end
    return ComplexF64
end

function infer_problem_dist(problem::Problem)
    if problem.domain !== nothing
        return problem.domain.dist
    end
    for var in problem.variables
        for comp in scalar_components(var)
            return comp.dist
        end
    end
    throw(ArgumentError("Problem has no associated distributor"))
end

function compute_variable_ranges(problem::Problem)
    scalar_ranges = Dict{ScalarField, UnitRange{Int}}()
    variable_ranges = Dict{Any, UnitRange{Int}}()
    offset = 0
    for var in problem.variables
        components = scalar_components(var)
        var_start = offset + 1
        for comp in components
            size = scalar_field_dofs(comp)
            if size == 0
                continue
            end
            range = (offset + 1):(offset + size)
            scalar_ranges[comp] = range
            offset += size
        end
        if offset >= var_start
            variable_ranges[var] = var_start:offset
        else
            variable_ranges[var] = var_start:(var_start - 1)
        end
    end
    return scalar_ranges, variable_ranges, offset
end

function compute_equation_ranges(problem::Problem)
    eq_ranges = Dict{Int, UnitRange{Int}}()
    offset = 0
    if isempty(problem.equation_data)
        equations = problem.equations
        for (i, _) in enumerate(equations)
            size = compute_field_size(Dict{String,Any}())
            range = (offset + 1):(offset + size)
            eq_ranges[i] = range
            offset += size
        end
    else
        for (i, eq_data) in enumerate(problem.equation_data)
            size = compute_field_size(eq_data)
            range = (offset + 1):(offset + size)
            eq_ranges[i] = range
            offset += size
        end
    end
    return eq_ranges, offset
end

function Subsystem(solver, group::Tuple=SUBSYSTEM_GROUP)
    problem = solver.problem
    dtype = infer_problem_dtype(problem)
    dist = infer_problem_dist(problem)
    scalar_ranges, variable_ranges, total_var = compute_variable_ranges(problem)
    equation_ranges, total_eq = compute_equation_ranges(problem)
    return Subsystem(solver, problem, dist, dtype, group,
                     scalar_ranges, variable_ranges, equation_ranges,
                     total_var, total_eq)
end

function build_subsystems(solver)
    """
    Build subsystem list for solver. Currently creates a single subsystem that
    spans the full problem but carries detailed range information for future
    block assembly.
    """
    return (Subsystem(solver),)
end

# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

function _coeff_data(field)
    ensure_layout!(field, :c)
    if field.data_c === nothing
        throw(ArgumentError("Field $(field.name) has no coefficient data available."))
    end
    return field.data_c
end

function coeff_size(subsystem::Subsystem, field::ScalarField)
    range = get(subsystem.scalar_ranges, field, 1:0)
    if isempty(range)
        return scalar_field_dofs(field)
    else
        return length(range)
    end
end
coeff_size(subsystem::Subsystem, field::VectorField) = sum(coeff_size(subsystem, comp) for comp in field.components)
coeff_size(subsystem::Subsystem, field::TensorField) = sum(coeff_size(subsystem, comp) for comp in vec(field.components))

# ---------------------------------------------------------------------------
# Gather / scatter
# ---------------------------------------------------------------------------

function gather(subsystem::Subsystem, fields::Vector{<:ScalarField})
    """
    Gather coefficient data from the provided fields into a single vector.
    """
    buffers = ComplexF64[]
    for field in fields
        data = _coeff_data(field)
        append!(buffers, vec(data))
    end
    return buffers
end

function scatter(subsystem::Subsystem, data::AbstractVector, fields::Vector{<:ScalarField})
    """
    Scatter vector entries back into field coefficient arrays.
    """
    offset = 0
    for field in fields
        coeffs = _coeff_data(field)
        n = length(coeffs)
        if offset + n > length(data)
            throw(ArgumentError("Insufficient data provided for scatter."))
        end
        coeffs .= reshape(data[offset+1:offset+n], size(coeffs))
        offset += n
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Subproblem containers
# ---------------------------------------------------------------------------

mutable struct Subproblem
    solver::Any
    subsystems::Tuple{Vararg{Subsystem}}
    group::Tuple
    variable_range::UnitRange{Int}
    equation_range::UnitRange{Int}
    matrices::Dict{String, Any}
end

function combined_range(ranges)
    starts = Int[]
    ends = Int[]
    for range in ranges
        if isempty(range)
            continue
        end
        push!(starts, first(range))
        push!(ends, last(range))
    end
    if isempty(starts)
        return 1:0
    else
        return minimum(starts):maximum(ends)
    end
end

function Subproblem(solver, subsystems::Tuple{Vararg{Subsystem}}, group::Tuple=SUBSYSTEM_GROUP)
    scalar_ranges = [collect(values(ss.scalar_ranges)) for ss in subsystems]
    eq_ranges = [collect(values(ss.equation_ranges)) for ss in subsystems]
    flat_scalar = reduce(vcat, scalar_ranges; init=UnitRange{Int}[])
    flat_eq = reduce(vcat, eq_ranges; init=UnitRange{Int}[])
    var_range = combined_range(flat_scalar)
    eq_range = combined_range(flat_eq)
    return Subproblem(solver, subsystems, group, var_range, eq_range, Dict{String, Any}())
end

function build_subproblems(solver, subsystems; build_matrices=nothing)
    """
    Construct subproblems from the supplied subsystems.  This simplified version
    maps each subsystem to one subproblem and runs a fallback matrix builder
    when requested.
    """
    subproblems = Tuple(Subproblem(solver, (ss,), ss.group) for ss in subsystems)
    if build_matrices !== nothing
        build_subproblem_matrices(solver, subproblems, build_matrices)
    end
    return subproblems
end

function build_subproblem_matrices(solver, subproblems, matrices)
    """
    Populate subproblem-local matrices by slicing the global system matrices
    stored on the problem.
    """
    if isempty(subproblems)
        return nothing
    end

    problem = solver.problem
    global_mats = Dict{String, Any}()

    if haskey(problem.parameters, "L_matrix")
        global_mats["L"] = problem.parameters["L_matrix"]
    end
    if haskey(problem.parameters, "M_matrix")
        global_mats["M"] = problem.parameters["M_matrix"]
    end
    if haskey(problem.parameters, "F_vector")
        global_mats["F"] = problem.parameters["F_vector"]
    end

    for sp in log_progress(subproblems; desc="Building subproblem matrices", frac=1.0, iter=1)
        eq_range = sp.equation_range
        var_range = sp.variable_range
        local = Dict{String, Any}()
        for name in matrices
            matrix_name = String(name)
            if matrix_name == "F" && haskey(global_mats, "F")
                local["F"] = copy(global_mats["F"][eq_range])
            elseif haskey(global_mats, matrix_name)
                global_mat = global_mats[matrix_name]
                local[matrix_name] = sparse(global_mat[eq_range, var_range])
            end
        end
        sp.matrices = local
    end

    return nothing
end
