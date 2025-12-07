"""
Subsystem and Subproblem classes for pencil-based matrix assembly.

This module implements the subsystem architecture including:
- Subsystem: Represents a subset of the global coefficient space (pencils)
- Subproblem: Represents coupled subsystems with matrix assembly
- Preconditioning: Left/right permutation and valid mode filtering
- NCC support: Non-constant coefficient handling

The key concepts are:
- Each subsystem is described by a "group" tuple containing a group index
  (for separable axes) or nothing (for coupled axes)
- Subproblems collect subsystems with the same matrix group
- Preconditioners handle permutation and valid mode filtering
"""

using SparseArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

const SUBSYSTEM_GROUP = (:global,)

# Solver configuration options (following solvers.py:39-65)
Base.@kwdef mutable struct SolverConfig
    ncc_cutoff::Float64 = 1e-6
    max_ncc_terms::Union{Nothing, Int} = nothing
    entry_cutoff::Float64 = 1e-12
    bc_top::Bool = true
    tau_left::Bool = true
    interleave_components::Bool = true
    store_expanded_matrices::Bool = false
end

# ---------------------------------------------------------------------------
# Subsystem construction
# ---------------------------------------------------------------------------

"""
    Subsystem

Represents a subset of the global coefficient space (pencil).
Following subsystems.py:107-150.

Each subsystem is described by a "group" tuple containing a
group index (for each separable axis) or nothing (for each coupled axis).
"""
struct Subsystem
    solver::Any
    problem::Problem
    dist::Distributor
    dtype::DataType
    group::Tuple
    matrix_group::Tuple
    scalar_ranges::Dict{ScalarField, UnitRange{Int}}
    variable_ranges::Dict{Any, UnitRange{Int}}
    equation_ranges::Dict{Int, UnitRange{Int}}
    total_variable_size::Int
    total_equation_size::Int
    # Reference to parent subproblem (set after construction)
    subproblem::Base.RefValue{Any}
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

"""
    compute_matrix_group(group, matrix_dependence, matrix_coupling, default_nonconst_groups)

Compute matrix group from subsystem group.
Following subsystems.py:124-129.
"""
function compute_matrix_group(group::Tuple, matrix_dependence::Vector{Bool},
                              matrix_coupling::Vector{Bool}, default_nonconst_groups::Tuple)
    matrix_group = collect(group)
    for i in eachindex(matrix_group)
        if i <= length(matrix_dependence) && i <= length(matrix_coupling)
            # Map non-dependent groups to default, since group 0 may have different truncation
            dep = matrix_dependence[i] || matrix_coupling[i]
            if !dep && matrix_group[i] != 0
                if i <= length(default_nonconst_groups)
                    matrix_group[i] = default_nonconst_groups[i]
                end
            end
        end
    end
    return Tuple(matrix_group)
end

function Subsystem(solver, group::Tuple=SUBSYSTEM_GROUP)
    problem = solver.problem
    dtype = infer_problem_dtype(problem)
    dist = infer_problem_dist(problem)
    scalar_ranges, variable_ranges, total_var = compute_variable_ranges(problem)
    equation_ranges, total_eq = compute_equation_ranges(problem)

    # Compute matrix group (following subsystems.py:124-129)
    matrix_coupling = hasfield(typeof(solver), :base) ? solver.base.matrix_coupling : fill(true, dist.dim)
    matrix_dependence = copy(matrix_coupling)  # Default: same as coupling
    default_nonconst_groups = hasfield(typeof(dist), :default_nonconst_groups) ?
                              dist.default_nonconst_groups : ntuple(_ -> 1, dist.dim)
    matrix_group = compute_matrix_group(group, matrix_dependence, matrix_coupling, default_nonconst_groups)

    return Subsystem(solver, problem, dist, dtype, group, matrix_group,
                     scalar_ranges, variable_ranges, equation_ranges,
                     total_var, total_eq, Ref{Any}(nothing))
end

"""
    build_subsystems(solver)

Build local subsystem objects.
Following subsystems.py:34-53.
"""
function build_subsystems(solver)
    # For now, create a single subsystem spanning the full problem
    # TODO: Implement full pencil decomposition based on matrix_coupling
    return (Subsystem(solver),)
end

# ---------------------------------------------------------------------------
# Subsystem methods
# ---------------------------------------------------------------------------

function coeff_slices(subsystem::Subsystem, domain)
    # Return slices for coefficient data in this subsystem
    # For global subsystem, return full slices
    return ntuple(_ -> Colon(), length(subsystem.group))
end

function coeff_shape(subsystem::Subsystem, domain)
    # Return shape of coefficient data in this subsystem
    if domain === nothing
        return ()
    end
    if hasfield(typeof(domain), :coeff_shape)
        return domain.coeff_shape
    end
    # Fallback: compute from bases
    return tuple([b.meta.size for b in domain.bases]...)
end

function coeff_size(subsystem::Subsystem, domain)
    return prod(coeff_shape(subsystem, domain))
end

function field_slices(subsystem::Subsystem, field::ScalarField)
    # Component slices (none for scalar) + coefficient slices
    return coeff_slices(subsystem, field_domain(field))
end

function field_shape(subsystem::Subsystem, field::ScalarField)
    return coeff_shape(subsystem, field_domain(field))
end

function field_size(subsystem::Subsystem, field::ScalarField)
    range = get(subsystem.scalar_ranges, field, 1:0)
    if isempty(range)
        return scalar_field_dofs(field)
    else
        return length(range)
    end
end

field_size(subsystem::Subsystem, field::VectorField) = sum(field_size(subsystem, comp) for comp in field.components)
field_size(subsystem::Subsystem, field::TensorField) = sum(field_size(subsystem, comp) for comp in vec(field.components))

function field_domain(field::ScalarField)
    if hasfield(typeof(field), :domain) && field.domain !== nothing
        return field.domain
    end
    return nothing
end

function _coeff_data(field)
    ensure_layout!(field, :c)
    if field.data_c === nothing
        throw(ArgumentError("Field $(field.name) has no coefficient data available."))
    end
    return field.data_c
end

"""
    gather(subsystem, fields)

Gather coefficient data from fields into a single vector.
Following subsystems.py:213-220.
"""
function gather(subsystem::Subsystem, fields::Vector{<:ScalarField})
    buffers = ComplexF64[]
    for field in fields
        data = _coeff_data(field)
        append!(buffers, vec(data))
    end
    return buffers
end

"""
    scatter(subsystem, data, fields)

Scatter vector entries back into field coefficient arrays.
Following subsystems.py:222-231.
"""
function scatter(subsystem::Subsystem, data::AbstractVector, fields::Vector{<:ScalarField})
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

"""
    Subproblem

Object representing one coupled subsystem of a problem.
Following subsystems.py:234-296.

Subproblems are identified by their group multi-index, which identifies
the corresponding group of each separable dimension of the problem.
"""
mutable struct Subproblem
    solver::Any
    problem::Problem
    subsystems::Tuple{Vararg{Subsystem}}
    group::Tuple
    dist::Any
    domain::Any
    dtype::DataType
    group_dict::Dict{String, Any}

    # Ranges
    variable_range::UnitRange{Int}
    equation_range::UnitRange{Int}

    # Matrices (built by build_matrices!)
    matrices::Dict{String, Any}

    # Preconditioners (following subsystems.py:560-563)
    pre_left::Union{Nothing, SparseMatrixCSC}
    pre_left_pinv::Union{Nothing, SparseMatrixCSC}
    pre_right::Union{Nothing, SparseMatrixCSC}
    pre_right_pinv::Union{Nothing, SparseMatrixCSC}

    # Minimal matrices for fast operations
    L_min::Union{Nothing, SparseMatrixCSC}
    M_min::Union{Nothing, SparseMatrixCSC}

    # Expanded matrices for IMEX recombination
    L_exp::Union{Nothing, SparseMatrixCSC}
    M_exp::Union{Nothing, SparseMatrixCSC}
    LHS::Union{Nothing, SparseMatrixCSC}

    # Update rank for Woodbury formula
    update_rank::Int

    # Input/output buffers
    _input_buffer::Union{Nothing, Matrix{ComplexF64}}
    _output_buffer::Union{Nothing, Matrix{ComplexF64}}
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
    problem = solver.problem

    # Compute ranges
    scalar_ranges = [collect(values(ss.scalar_ranges)) for ss in subsystems]
    eq_ranges = [collect(values(ss.equation_ranges)) for ss in subsystems]
    flat_scalar = reduce(vcat, scalar_ranges; init=UnitRange{Int}[])
    flat_eq = reduce(vcat, eq_ranges; init=UnitRange{Int}[])
    var_range = combined_range(flat_scalar)
    eq_range = combined_range(flat_eq)

    # Get distributor and domain
    dist = length(subsystems) > 0 ? subsystems[1].dist : infer_problem_dist(problem)
    domain = length(problem.variables) > 0 ? field_domain(problem.variables[1]) : nothing
    dtype = infer_problem_dtype(problem)

    # Cross-reference subsystems to this subproblem
    for subsystem in subsystems
        subsystem.subproblem[] = nothing  # Will be set after construction
    end

    # Build group dictionary (following subsystems.py:257-261)
    group_dict = Dict{String, Any}()
    for (axis, ax_group) in enumerate(group)
        if ax_group !== nothing && hasfield(typeof(dist), :coords)
            if axis <= length(dist.coords.names)
                coord_name = dist.coords.names[axis]
                group_dict["n" * coord_name] = ax_group
            end
        end
    end

    sp = Subproblem(
        solver, problem, subsystems, group, dist, domain, dtype, group_dict,
        var_range, eq_range,
        Dict{String, Any}(),
        nothing, nothing, nothing, nothing,  # Preconditioners
        nothing, nothing,  # Minimal matrices
        nothing, nothing, nothing,  # Expanded matrices
        0,  # update_rank
        nothing, nothing  # Buffers
    )

    # Set back-reference
    for subsystem in subsystems
        subsystem.subproblem[] = sp
    end

    return sp
end

# Subproblem accessor methods (delegating to first subsystem)
coeff_slices(sp::Subproblem, domain) = coeff_slices(sp.subsystems[1], domain)
coeff_shape(sp::Subproblem, domain) = coeff_shape(sp.subsystems[1], domain)
coeff_size(sp::Subproblem, domain) = coeff_size(sp.subsystems[1], domain)
field_slices(sp::Subproblem, field) = field_slices(sp.subsystems[1], field)
field_shape(sp::Subproblem, field) = field_shape(sp.subsystems[1], field)
field_size(sp::Subproblem, field) = field_size(sp.subsystems[1], field)

"""
    check_condition(sp::Subproblem, eq_data)

Check if equation condition is satisfied for this subproblem.
Following subsystems.py:494-495.
"""
function check_condition(sp::Subproblem, eq_data::Dict)
    condition = get(eq_data, "condition", "true")
    if condition == "true" || condition === nothing
        return true
    end
    # Evaluate condition with group dictionary
    # TODO: Implement proper condition evaluation
    return true
end

"""
    valid_modes(sp::Subproblem, field, valid_modes_array)

Get valid modes for field in this subproblem.
Following subsystems.py:476-478.
"""
function valid_modes(sp::Subproblem, field, valid_modes_array)
    if valid_modes_array === nothing
        # All modes valid by default
        return ones(Bool, field_size(sp, field))
    end
    slices = field_slices(sp, field)
    return valid_modes_array[slices...]
end

# ---------------------------------------------------------------------------
# Build subproblems
# ---------------------------------------------------------------------------

"""
    build_subproblems(solver, subsystems; build_matrices=nothing)

Construct subproblems from the supplied subsystems.
Following subsystems.py:55-70.
"""
function build_subproblems(solver, subsystems; build_matrices=nothing)
    # Arrange subsystems by matrix groups
    subsystems_by_group = Dict{Tuple, Vector{Subsystem}}()
    for subsystem in subsystems
        group = subsystem.matrix_group
        if !haskey(subsystems_by_group, group)
            subsystems_by_group[group] = Subsystem[]
        end
        push!(subsystems_by_group[group], subsystem)
    end

    # Build subproblem objects for each matrix group
    subproblems = Subproblem[]
    for (matrix_group, group_subsystems) in subsystems_by_group
        subproblem = Subproblem(solver, Tuple(group_subsystems), matrix_group)
        push!(subproblems, subproblem)
    end
    subproblems = Tuple(subproblems)

    # Build matrices if requested
    if build_matrices !== nothing
        build_subproblem_matrices(solver, subproblems, build_matrices)
    end

    return subproblems
end

"""
    build_subproblem_matrices(solver, subproblems, matrices)

Build matrices for all subproblems.
Following subsystems.py:72-81.
"""
function build_subproblem_matrices(solver, subproblems, matrices)
    # Setup NCCs (gather coefficients)
    problem = solver.problem
    for eq_data in problem.equation_data
        for matrix_name in matrices
            expr = get(eq_data, String(matrix_name), nothing)
            if expr !== nothing
                gather_ncc_coeffs!(expr)
            end
        end
    end

    # Build matrices for each subproblem
    for sp in log_progress(subproblems; desc="Building subproblem matrices", frac=1.0, iter=1)
        build_matrices!(sp, matrices, solver)
    end

    return nothing
end

"""
    gather_ncc_coeffs!(expr)

Gather NCC coefficients for expression.
Following arithmetic.py:370-373.
"""
function gather_ncc_coeffs!(expr)
    # Recursively gather NCC coefficients
    if hasfield(typeof(expr), :operand)
        gather_ncc_coeffs!(expr.operand)
    end
    if hasfield(typeof(expr), :operands)
        for op in expr.operands
            gather_ncc_coeffs!(op)
        end
    end
    # Store NCC data if this is an NCC multiply
    if hasfield(typeof(expr), :ncc_data)
        # NCC coefficient gathering logic
        # TODO: Implement full NCC coefficient storage
    end
end

gather_ncc_coeffs!(::Nothing) = nothing
gather_ncc_coeffs!(::Number) = nothing

# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------

"""
    build_matrices!(sp::Subproblem, names, solver)

Build problem matrices for subproblem.
Following subsystems.py:497-596.
"""
function build_matrices!(sp::Subproblem, names, solver)
    problem = sp.problem
    eqns = problem.equation_data
    vars = problem.variables
    dtype = sp.dtype

    # Get solver config
    config = hasfield(typeof(solver), :config) ? solver.config : SolverConfig()
    ncc_cutoff = config.ncc_cutoff
    max_ncc_terms = config.max_ncc_terms
    entry_cutoff = config.entry_cutoff
    bc_top = config.bc_top
    tau_left = config.tau_left
    interleave_components = config.interleave_components
    store_expanded = config.store_expanded_matrices

    # Compute sizes
    eqn_conditions = [check_condition(sp, eq) for eq in eqns]
    eqn_sizes = [get(eq, "equation_size", 0) for eq in eqns]
    var_sizes = [field_dofs(var) for var in vars]
    I = sum(eqn_sizes)
    J = sum(var_sizes)

    # Construct matrices (following subsystems.py:512-537)
    matrices = Dict{String, SparseMatrixCSC}()
    for name in names
        name_str = String(name)
        data, rows, cols = Float64[], Int[], Int[]

        i0 = 0
        for (eq_idx, (eq_data, eqn_size, eqn_cond)) in enumerate(zip(eqns, eqn_sizes, eqn_conditions))
            if eqn_size > 0 && eqn_cond
                expr = get(eq_data, name_str, nothing)
                if expr !== nothing && !is_zero_expression(expr)
                    # Build expression matrices for each variable
                    eqn_blocks = expression_matrices(expr, sp, vars;
                                                     ncc_cutoff=ncc_cutoff,
                                                     max_ncc_terms=max_ncc_terms)
                    j0 = 0
                    for (var_idx, (var, var_size)) in enumerate(zip(vars, var_sizes))
                        if var_size > 0 && haskey(eqn_blocks, var)
                            block = eqn_blocks[var]
                            if isa(block, SparseMatrixCSC) && nnz(block) > 0
                                block_rows, block_cols, block_vals = findnz(block)
                                append!(data, block_vals)
                                append!(rows, i0 .+ block_rows)
                                append!(cols, j0 .+ block_cols)
                            end
                        end
                        j0 += var_size
                    end
                end
            end
            i0 += eqn_size
        end

        # Build sparse matrix with entry cutoff filter
        if !isempty(data)
            significant = abs.(data) .>= entry_cutoff
            data = data[significant]
            rows = rows[significant]
            cols = cols[significant]
        end
        matrices[name_str] = sparse(rows, cols, ComplexF64.(data), I, J)
    end

    # Valid modes (following subsystems.py:539-548)
    valid_eqn = vcat([get_valid_modes(eq, sp, eqn_sizes[i]) .* eqn_conditions[i]
                      for (i, eq) in enumerate(eqns)]...)
    valid_var = vcat([get_valid_modes_var(var, sp) for var in vars]...)

    # Convert to diagonal filter matrices
    valid_eqn_mat = spdiagm(0 => Float64.(valid_eqn))
    valid_var_mat = spdiagm(0 => Float64.(valid_var))

    # Check squareness (following subsystems.py:551-552)
    n_valid_eqn = sum(valid_eqn)
    n_valid_var = sum(valid_var)
    if n_valid_eqn != n_valid_var
        @warn "Non-square system: group=$(sp.group), valid_eqn=$n_valid_eqn, valid_var=$n_valid_var"
    end

    # Build permutation matrices (following subsystems.py:554-556)
    left_perm = left_permutation(sp, eqns, bc_top, interleave_components)
    right_perm = right_permutation(sp, vars, tau_left, interleave_components)

    # Build preconditioners (following subsystems.py:559-563)
    sp.pre_left = drop_empty_rows(left_perm * valid_eqn_mat)
    sp.pre_left_pinv = sparse(sp.pre_left')
    sp.pre_right_pinv = drop_empty_rows(right_perm * valid_var_mat)
    sp.pre_right = sparse(sp.pre_right_pinv')

    # Precondition matrices (following subsystems.py:569-571)
    for (name, matrix) in matrices
        matrices[name] = sp.pre_left * matrix * sp.pre_right
    end

    # Store minimal CSR matrices (following subsystems.py:573-575)
    sp.matrices = matrices
    if haskey(matrices, "L")
        sp.L_min = matrices["L"]
    end
    if haskey(matrices, "M")
        sp.M_min = matrices["M"]
    end

    # Store expanded matrices for IMEX if requested (following subsystems.py:577-588)
    if length(matrices) > 1 && store_expanded
        sp.LHS = zeros_with_pattern(values(matrices)...)
        for (name, matrix) in matrices
            expanded = expand_pattern(matrix, sp.LHS)
            if name == "L"
                sp.L_exp = expanded
            elseif name == "M"
                sp.M_exp = expanded
            end
        end
    else
        # Placeholder for shape access
        sp.LHS = spzeros(dtype, n_valid_eqn, n_valid_var)
    end

    # Compute update rank for Woodbury formula
    sp.update_rank = compute_update_rank(sp, eqns, eqn_conditions, eqn_sizes)

    return nothing
end

"""
    get_valid_modes(eq_data, sp, size)

Get valid modes array for equation.
"""
function get_valid_modes(eq_data::Dict, sp::Subproblem, size::Int)
    valid = get(eq_data, "valid_modes", nothing)
    if valid === nothing
        return ones(Bool, size)
    end
    return vec(valid)
end

"""
    get_valid_modes_var(var, sp)

Get valid modes array for variable.
"""
function get_valid_modes_var(var, sp::Subproblem)
    if hasfield(typeof(var), :valid_modes) && var.valid_modes !== nothing
        return vec(var.valid_modes)
    end
    return ones(Bool, field_dofs(var))
end

"""
    expression_matrices(expr, sp, vars; kwargs...)

Build expression matrices for each variable.
"""
function expression_matrices(expr, sp::Subproblem, vars; kwargs...)
    # Default: return empty dict
    # This should be overridden by specific expression types
    return Dict{Any, SparseMatrixCSC}()
end

expression_matrices(::Nothing, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(::Number, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()

"""
    is_zero_expression(expr)

Check if expression is zero.
"""
is_zero_expression(expr) = false
is_zero_expression(::Nothing) = true
is_zero_expression(x::Number) = x == 0

"""
    compute_update_rank(sp, eqns, eqn_conditions, eqn_sizes)

Compute update rank for Woodbury formula.
Following subsystems.py:591-595.
"""
function compute_update_rank(sp::Subproblem, eqns, eqn_conditions, eqn_sizes)
    # Group equation DOFs by dimension
    eqn_dofs_by_dim = Dict{Int, Int}()
    for (eq, cond, size) in zip(eqns, eqn_conditions, eqn_sizes)
        if cond
            dim = get(eq, "domain_dim", 0)
            eqn_dofs_by_dim[dim] = get(eqn_dofs_by_dim, dim, 0) + size
        end
    end

    if isempty(eqn_dofs_by_dim)
        return 0
    end

    max_dim = maximum(keys(eqn_dofs_by_dim))
    total_dofs = sum(values(eqn_dofs_by_dim))
    return total_dofs - get(eqn_dofs_by_dim, max_dim, 0)
end

# ---------------------------------------------------------------------------
# Permutation matrices
# ---------------------------------------------------------------------------

"""
    left_permutation(sp, equations, bc_top, interleave_components)

Left permutation acting on equations.
Following subsystems.py:614-675.

bc_top determines if lower-dimensional equations are placed at the top or bottom.

Input ordering: Equations > Components > Modes
Output ordering with interleave_components=true: Modes > Components > Equations
Output ordering with interleave_components=false: Modes > Equations > Components
"""
function left_permutation(sp::Subproblem, equations, bc_top::Bool, interleave_components::Bool)
    # Compute hierarchy of input equation indices
    i = 0
    L0 = Vector{Vector{Vector{Int}}}()

    for eq_data in equations
        L1 = Vector{Vector{Int}}()
        eq_size = get(eq_data, "equation_size", 0)

        if eq_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end

        # Get shape info
        tensorsig = get(eq_data, "tensorsig", ())
        rank = length(tensorsig)
        n_comps = max(1, prod(cs isa CoordinateSystem ? cs.dim : 1 for cs in tensorsig))
        n_coeffs = eq_size ÷ n_comps

        for comp in 1:n_comps
            L2 = Int[]
            for coeff in 1:n_coeffs
                push!(L2, i)
                i += 1
            end
            push!(L1, L2)
        end
        push!(L0, L1)
    end

    # Reverse list hierarchy, grouping by dimension
    indices_by_dim = Dict{Int, Vector{Int}}()
    n1max = length(L0)
    n2max = maximum(length(L1) for L1 in L0; init=1)
    n3max = maximum(length(L2) for L1 in L0 for L2 in L1; init=1)

    if interleave_components
        for n3 in 1:n3max
            for n2 in 1:n2max
                for n1 in 1:n1max
                    dim = get(equations[n1], "domain_dim", 0)
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    else
        for n3 in 1:n3max
            for n1 in 1:n1max
                dim = get(equations[n1], "domain_dim", 0)
                for n2 in 1:n2max
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    end

    # Combine indices by dimension
    dims = sort(collect(keys(indices_by_dim)))
    if !bc_top
        dims = reverse(dims)
    end

    indices = Int[]
    for dim in dims
        append!(indices, indices_by_dim[dim])
    end

    return perm_matrix(indices .+ 1, i)  # +1 for 1-based indexing
end

"""
    right_permutation(sp, variables, tau_left, interleave_components)

Right permutation acting on variables.
Following subsystems.py:678-739.

tau_left determines if lower-dimensional variables are placed at the left or right.

Input ordering: Variables > Components > Modes
Output ordering with interleave_components=true: Modes > Components > Variables
Output ordering with interleave_components=false: Modes > Variables > Components
"""
function right_permutation(sp::Subproblem, variables, tau_left::Bool, interleave_components::Bool)
    # Compute hierarchy of input variable indices
    i = 0
    L0 = Vector{Vector{Vector{Int}}}()

    for var in variables
        L1 = Vector{Vector{Int}}()
        var_size = field_dofs(var)

        if var_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end

        # Get shape info
        components = scalar_components(var)
        n_comps = length(components)
        n_coeffs = var_size ÷ n_comps

        for comp in 1:n_comps
            L2 = Int[]
            for coeff in 1:n_coeffs
                push!(L2, i)
                i += 1
            end
            push!(L1, L2)
        end
        push!(L0, L1)
    end

    # Reverse list hierarchy, grouping by dimension
    indices_by_dim = Dict{Int, Vector{Int}}()
    n1max = length(L0)
    n2max = maximum(length(L1) for L1 in L0; init=1)
    n3max = maximum(length(L2) for L1 in L0 for L2 in L1; init=1)

    if interleave_components
        for n3 in 1:n3max
            for n2 in 1:n2max
                for n1 in 1:n1max
                    var = variables[n1]
                    dim = get_var_dim(var)
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    else
        for n3 in 1:n3max
            for n1 in 1:n1max
                var = variables[n1]
                dim = get_var_dim(var)
                for n2 in 1:n2max
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    end

    # Combine indices by dimension
    dims = sort(collect(keys(indices_by_dim)))
    if !tau_left
        dims = reverse(dims)
    end

    indices = Int[]
    for dim in dims
        append!(indices, indices_by_dim[dim])
    end

    return perm_matrix(indices .+ 1, i)  # +1 for 1-based indexing
end

"""
    get_var_dim(var)

Get dimension of variable's domain.
"""
function get_var_dim(var)
    if hasfield(typeof(var), :domain) && var.domain !== nothing
        return var.domain.dim
    end
    if hasfield(typeof(var), :bases)
        return length(var.bases)
    end
    return 0
end

"""
    perm_matrix(indices, n)

Create permutation matrix from index array.
Following tools/array.py perm_matrix.
"""
function perm_matrix(indices::Vector{Int}, n::Int)
    m = length(indices)
    if m == 0 || n == 0
        return spzeros(Float64, m, n)
    end

    rows = collect(1:m)
    cols = indices
    vals = ones(Float64, m)

    # Filter valid indices
    valid = (cols .>= 1) .& (cols .<= n)
    rows = rows[valid]
    cols = cols[valid]
    vals = vals[valid]

    return sparse(rows, cols, vals, m, n)
end

# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

"""
    drop_empty_rows(A)

Remove empty rows from sparse matrix.
Following tools/array.py drop_empty_rows.
"""
function drop_empty_rows(A::SparseMatrixCSC)
    row_sums = vec(sum(abs.(A), dims=2))
    non_empty = findall(row_sums .> 0)
    if isempty(non_empty)
        return spzeros(eltype(A), 0, size(A, 2))
    end
    return A[non_empty, :]
end

"""
    zeros_with_pattern(matrices...)

Create zero matrix with combined sparsity pattern.
Following tools/array.py zeros_with_pattern.
"""
function zeros_with_pattern(matrices...)
    if isempty(matrices)
        return spzeros(ComplexF64, 0, 0)
    end

    # Get combined shape
    m = size(first(matrices), 1)
    n = size(first(matrices), 2)

    # Collect all non-zero positions
    rows = Int[]
    cols = Int[]

    for mat in matrices
        r, c, _ = findnz(mat)
        append!(rows, r)
        append!(cols, c)
    end

    if isempty(rows)
        return spzeros(ComplexF64, m, n)
    end

    # Create matrix with zeros at all pattern positions
    vals = zeros(ComplexF64, length(rows))
    return sparse(rows, cols, vals, m, n)
end

"""
    expand_pattern(A, B)

Expand A to have same sparsity pattern as B.
Following tools/array.py expand_pattern.
"""
function expand_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    # Get pattern of B
    rows_b, cols_b, _ = findnz(B)

    # Get values of A at pattern positions
    vals = zeros(eltype(A), length(rows_b))
    for (i, (r, c)) in enumerate(zip(rows_b, cols_b))
        if r <= size(A, 1) && c <= size(A, 2)
            vals[i] = A[r, c]
        end
    end

    return sparse(rows_b, cols_b, vals, size(B)...)
end

"""
    apply_sparse(A, x; axis=0, out=nothing)

Apply sparse matrix along specified axis.
Following tools/array.py apply_sparse.
"""
function apply_sparse(A::SparseMatrixCSC, x::AbstractArray; axis::Int=1, out::Union{Nothing, AbstractArray}=nothing)
    if axis == 1
        result = A * x
    elseif axis == 2
        result = (A * x')'
    else
        throw(ArgumentError("axis must be 1 or 2"))
    end

    if out !== nothing
        copyto!(out, result)
        return out
    end
    return result
end

# ---------------------------------------------------------------------------
# Gather/scatter for subproblems
# ---------------------------------------------------------------------------

"""
    gather_inputs(sp::Subproblem, fields)

Gather and precondition subproblem data from input-like field list.
Following subsystems.py:340-350.
"""
function gather_inputs(sp::Subproblem, fields::Vector{<:ScalarField})
    # Gather from subsystems
    data = gather(sp.subsystems[1], fields)

    # Apply right preconditioner inverse to compress inputs
    if sp.pre_right_pinv !== nothing
        data = Vector(sp.pre_right_pinv * data)
    end

    return data
end

"""
    scatter_inputs(sp::Subproblem, data, fields)

Precondition and scatter subproblem data out to input-like field list.
Following subsystems.py:364-371.
"""
function scatter_inputs(sp::Subproblem, data::AbstractVector, fields::Vector{<:ScalarField})
    # Undo right preconditioner inverse to expand inputs
    if sp.pre_right !== nothing
        data = Vector(sp.pre_right * data)
    end

    # Scatter to fields
    scatter(sp.subsystems[1], data, fields)
end

# ---------------------------------------------------------------------------
# NCC (Non-Constant Coefficient) handling
# ---------------------------------------------------------------------------

"""
    NCCData

Storage for non-constant coefficient data.
"""
mutable struct NCCData
    coeffs::Union{Nothing, Array}
    cutoff::Float64
    max_terms::Union{Nothing, Int}
end

NCCData() = NCCData(nothing, 1e-6, nothing)

"""
    build_ncc_matrix(ncc_data, sp, arg_domain, out_domain; ncc_cutoff=1e-6, max_ncc_terms=nothing)

Build NCC matrix for Cartesian coordinates.
Following arithmetic.py:418-444.
"""
function build_ncc_matrix(ncc_data::NCCData, sp::Subproblem, arg_domain, out_domain;
                          ncc_cutoff::Float64=1e-6, max_ncc_terms::Union{Nothing, Int}=nothing)
    if ncc_data.coeffs === nothing
        return nothing
    end

    coeffs = ncc_data.coeffs
    shape = (field_size(sp, out_domain), field_size(sp, arg_domain))
    matrix = spzeros(ComplexF64, shape...)

    # Get subproblem shape
    sp_shape = coeff_shape(sp, out_domain)

    # Loop over NCC modes
    ncc_shape = size(coeffs)
    n_terms = 0

    for ncc_mode in CartesianIndices(ncc_shape)
        ncc_coeff = coeffs[ncc_mode]

        # Apply cutoff
        if abs(ncc_coeff) > ncc_cutoff
            # Build mode matrix
            mode_mat = cartesian_mode_matrix(sp_shape, arg_domain, out_domain, Tuple(ncc_mode))
            matrix = matrix + ncc_coeff * mode_mat

            n_terms += 1
            if max_ncc_terms !== nothing && n_terms >= max_ncc_terms
                break
            end
        end
    end

    return matrix
end

"""
    cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode)

Build mode matrix for Cartesian NCC.
Following arithmetic.py:446-460.
"""
function cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode::Tuple)
    # Build Kronecker product of 1D mode matrices
    matrix = sparse([1.0])

    for axis in eachindex(sp_shape)
        n = sp_shape[axis]
        mode = axis <= length(ncc_mode) ? ncc_mode[axis] : 1

        # Product matrix for this axis
        # For identity: sparse identity
        # For mode k: shift matrix
        if mode == 1
            axis_mat = sparse(I, n, n)
        else
            # Circulant shift for periodic, or convolution for non-periodic
            axis_mat = sparse(I, n, n)  # Simplified: identity
        end

        matrix = kron(matrix, axis_mat)
    end

    return matrix
end

# ---------------------------------------------------------------------------
# Legacy compatibility functions
# ---------------------------------------------------------------------------

# Keep old function signatures for backwards compatibility
coeff_size(subsystem::Subsystem, field::ScalarField) = field_size(subsystem, field)
coeff_size(subsystem::Subsystem, field::VectorField) = sum(field_size(subsystem, comp) for comp in field.components)
coeff_size(subsystem::Subsystem, field::TensorField) = sum(field_size(subsystem, comp) for comp in vec(field.components))
