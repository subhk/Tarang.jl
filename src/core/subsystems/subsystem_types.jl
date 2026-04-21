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

# SparseArrays, LinearAlgebra already in Tarang.jl

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

const SUBSYSTEM_GROUP = (:global,)

# Solver configuration options (following solvers:39-65)
Base.@kwdef mutable struct SolverConfig
    ncc_cutoff::Float64 = 1e-6
    max_ncc_terms::Union{Nothing, Int} = nothing
    entry_cutoff::Float64 = 1e-12
    bc_top::Bool = true
    tau_left::Bool = true
    interleave_components::Bool = true
    store_expanded_matrices::Bool = true  # Enable in-place LHS pattern updates
    # Reorder the local subproblem iteration order so heavier (higher-nnz)
    # subproblems are processed first. Under the subproblem stepper this
    # gives the JIT and the sparse LU solver a chance to warm up expensive
    # kernels while less-contended subproblems (the DC mode, typically)
    # still have work for the multi-threaded LU backend. Set to `false` to
    # preserve the natural kx-ordering for cache-locality reasons. Only
    # affects local ordering on a single rank — does NOT redistribute
    # subproblems across ranks (that's a Dedalus-style MPI rebalancing
    # project — out of scope for now).
    balance_local_cost::Bool = false
end

# ---------------------------------------------------------------------------
# Subsystem construction
# ---------------------------------------------------------------------------

"""
    Subsystem

Represents a subset of the global coefficient space (pencil).
Following subsystems:107-150.

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
scalar_components(vfield::VectorField) = vfield.components
scalar_components(tensor::TensorField) = collect(vec(tensor.components))

function scalar_field_dofs(field::ScalarField)
    if get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing
        return length(get_grid_data(field))
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
            dof_count = scalar_field_dofs(comp)
            if dof_count == 0
                continue
            end
            range = (offset + 1):(offset + dof_count)
            scalar_ranges[comp] = range
            offset += dof_count
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
            eq_size = compute_field_size(Dict{String,Any}())
            range = (offset + 1):(offset + eq_size)
            eq_ranges[i] = range
            offset += eq_size
        end
    else
        for (i, eq_data) in enumerate(problem.equation_data)
            eq_size = compute_field_size(eq_data)
            range = (offset + 1):(offset + eq_size)
            eq_ranges[i] = range
            offset += eq_size
        end
    end
    return eq_ranges, offset
end

"""
    compute_matrix_group(group, matrix_dependence, matrix_coupling, default_nonconst_groups)

Compute matrix group from subsystem group.
Following subsystems:124-129.
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

function _normalize_subsystem_group(dist::Distributor, group::Tuple)
    if group === SUBSYSTEM_GROUP || (length(group) == 1 && group[1] === :global)
        return ntuple(_ -> nothing, dist.dim)
    end
    if length(group) != dist.dim
        throw(ArgumentError("Subsystem group length $(length(group)) does not match dist.dim=$(dist.dim)"))
    end
    return group
end

function Subsystem(solver, group::Tuple=SUBSYSTEM_GROUP)
    problem = solver.problem
    dtype = infer_problem_dtype(problem)
    dist = infer_problem_dist(problem)
    group = _normalize_subsystem_group(dist, group)
    scalar_ranges, variable_ranges, total_var = compute_variable_ranges(problem)
    equation_ranges, total_eq = compute_equation_ranges(problem)

    # Compute matrix group (following subsystems:124-129)
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

Build local subsystem objects based on matrix coupling analysis.

For each dimension, the solver determines whether modes couple across that dimension:
- Coupled dimensions (matrix_coupling[i] = true): All modes in that dimension must be
  solved together, so they belong to the same subsystem group
- Separable dimensions (matrix_coupling[i] = false): Each mode can be solved independently,
  so they form separate subsystem groups

This implements pencil-based matrix assembly where:
- The coupled dimensions form the "local" part of each pencil
- The separable dimensions are iterated over to form multiple subsystems

Following subsystems:34-53.
"""
function build_subsystems(solver)
    problem = solver.problem
    dist = infer_problem_dist(problem)

    if dist === nothing
        # No distributor available, use global subsystem
        return (Subsystem(solver),)
    end

    # Get matrix coupling information from solver or compute from equations
    matrix_coupling = get_matrix_coupling(solver, problem, dist)

    # Determine which dimensions are separable (can be parallelized)
    separable_dims = findall(.!matrix_coupling)

    if isempty(separable_dims)
        # All dimensions are coupled - single global subsystem
        return (Subsystem(solver),)
    end

    # Build subsystems for each mode group in separable dimensions
    subsystems = Subsystem[]

    # Get the number of modes in each separable dimension
    separable_sizes = Int[]
    for dim in separable_dims
        if dim <= length(dist.coords)
            coord = dist.coords[dim]
            if hasfield(typeof(coord), :size)
                push!(separable_sizes, coord.size)
            else
                # Fallback: get size from problem domain
                dim_size = get_separable_dim_size(problem, dim)
                push!(separable_sizes, dim_size)
            end
        else
            push!(separable_sizes, 1)
        end
    end

    if isempty(separable_sizes) || all(s -> s <= 1, separable_sizes)
        # No meaningful separation possible
        return (Subsystem(solver),)
    end

    # Generate mode group combinations for separable dimensions.
    # For MPI: only build groups for locally-owned modes.
    mode_groups = generate_mode_groups(separable_dims, separable_sizes, dist.dim)

    # Filter to local modes when using MPI
    if dist.size > 1
        local_mode_groups = Tuple[]
        for group in mode_groups
            is_local = true
            for (i, dim) in enumerate(separable_dims)
                if dim <= dist.dim && group[dim] !== nothing
                    local_range = local_indices(dist, dim, separable_sizes[i])
                    # group[dim] is 0-based; local_range is 1-based
                    if (group[dim] + 1) ∉ local_range
                        is_local = false
                        break
                    end
                end
            end
            if is_local
                push!(local_mode_groups, group)
            end
        end
        mode_groups = local_mode_groups
    end

    for group in mode_groups
        subsystem = Subsystem(solver, group)
        push!(subsystems, subsystem)
    end

    if isempty(subsystems)
        # Fallback to global subsystem
        return (Subsystem(solver),)
    end

    return Tuple(subsystems)
end

"""
    get_matrix_coupling(solver, problem, dist) -> Vector{Bool}

Determine matrix coupling for each dimension based on operator analysis.

Returns a boolean vector where `true` indicates that dimension couples
modes together (requires simultaneous solution), and `false` indicates
modes are separable (can be solved independently).
"""
function get_matrix_coupling(solver, problem::Problem, dist::Distributor)
    ndims = dist.dim

    # Check if solver has explicit matrix coupling
    if hasfield(typeof(solver), :base) && hasfield(typeof(solver.base), :matrix_coupling)
        return collect(solver.base.matrix_coupling)
    end

    # Analyze operators in equations to determine coupling
    coupling = fill(false, ndims)

    for eq_data in problem.equation_data
        if haskey(eq_data, "lhs")
            lhs_op = eq_data["lhs"]
            op_coupling = analyze_operator_coupling(lhs_op, ndims)
            coupling .|= op_coupling
        end
        if haskey(eq_data, "rhs")
            rhs_op = eq_data["rhs"]
            op_coupling = analyze_operator_coupling(rhs_op, ndims)
            coupling .|= op_coupling
        end
    end

    # If no coupling detected, assume all coupled for safety
    if !any(coupling)
        return fill(true, ndims)
    end

    return coupling
end

"""
    analyze_operator_coupling(op, ndims) -> Vector{Bool}

Analyze an operator to determine which dimensions it couples.

Differential operators couple modes in their differentiation direction,
while multiplication operators may couple all dimensions for NCCs.
"""
function analyze_operator_coupling(op, ndims::Int)
    coupling = fill(false, ndims)

    if op === nothing
        return coupling
    end

    # Differential operators couple along their direction
    if isa(op, Differentiate)
        coord = op.coord
        if coord !== nothing && hasfield(typeof(coord), :coordsys) && coord.coordsys !== nothing
            coordsys = coord.coordsys
            if hasfield(typeof(coordsys), :coords)
                idx = findfirst(c -> c.name == coord.name, coordsys.coords)
                if idx !== nothing && idx <= ndims
                    coupling[idx] = true
                end
            end
        end

    elseif isa(op, CartesianGradient) || isa(op, CartesianDivergence) ||
           isa(op, CartesianCurl) || isa(op, CartesianLaplacian)
        # Vector calculus operators couple all spatial dimensions
        fill!(coupling, true)

    elseif isa(op, AddOperator) || isa(op, SubtractOperator)
        # Binary operators: combine coupling from both operands
        left_coupling = analyze_operator_coupling(op.left, ndims)
        right_coupling = analyze_operator_coupling(op.right, ndims)
        coupling .|= left_coupling
        coupling .|= right_coupling

    elseif isa(op, MultiplyOperator)
        left_coupling = analyze_operator_coupling(op.left, ndims)
        coupling .|= left_coupling
        if !(op.right isa Number || op.right isa ConstantOperator || op.right isa ZeroOperator)
            right_coupling = analyze_operator_coupling(op.right, ndims)
            coupling .|= right_coupling
        end

    elseif hasfield(typeof(op), :operand)
        # Single-operand operators
        coupling .|= analyze_operator_coupling(op.operand, ndims)
    end

    return coupling
end

"""
    get_separable_dim_size(problem, dim) -> Int

Get the number of modes in a separable dimension from problem structure.
"""
function get_separable_dim_size(problem::Problem, dim::Int)
    if problem.domain !== nothing
        coeff_shape = coefficient_shape(problem.domain)
        if dim <= length(coeff_shape)
            return coeff_shape[dim]
        end
    end

    # Fallback: reconstruct the per-axis size from variables.
    # Applies the same rule as `_coefficient_shape_impl` — only the FIRST
    # Fourier axis gets halved by rfft; subsequent Fourier axes see
    # complex input and use fft (full size). Previously this branch
    # hardcoded `div(N, 2) + 1` for every RealFourier, which was wrong
    # for multi-Fourier layouts and only worked because the primary path
    # through `coefficient_shape` was also wrong (both bugs cancelled).
    for var in problem.variables
        for comp in scalar_components(var)
            if dim <= length(comp.bases) && comp.bases[dim] !== nothing
                basis = comp.bases[dim]
                # Find the first Fourier axis in this component's basis list.
                first_fourier_idx = nothing
                for (i, b) in enumerate(comp.bases)
                    if isa(b, RealFourier) || isa(b, ComplexFourier)
                        first_fourier_idx = i
                        break
                    end
                end
                is_first = (first_fourier_idx !== nothing && dim == first_fourier_idx)
                if isa(basis, RealFourier) && is_first
                    return div(basis.meta.size, 2) + 1
                end
                return basis.meta.size
            end
        end
    end

    return 1
end

"""
    generate_mode_groups(separable_dims, separable_sizes, ndims) -> Vector{Tuple}

Generate all mode group combinations for separable dimensions.

Each group tuple has:
- An integer index for each separable dimension (0 to size-1)
- `nothing` for each coupled dimension
"""
function generate_mode_groups(separable_dims::Vector{Int}, separable_sizes::Vector{Int}, ndims::Int)
    groups = Tuple[]

    # Create ranges for each separable dimension
    ranges = [0:(sz-1) for sz in separable_sizes]

    # Generate Cartesian product of mode indices
    for indices in Iterators.product(ranges...)
        # Build full group tuple
        group = Vector{Union{Int, Nothing}}(nothing, ndims)
        for (i, dim) in enumerate(separable_dims)
            if dim <= ndims
                group[dim] = indices[i]
            end
        end
        push!(groups, Tuple(group))
    end

    return groups
end

