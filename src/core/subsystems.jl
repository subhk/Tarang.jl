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

    # Fallback: try to get from variables
    for var in problem.variables
        for comp in scalar_components(var)
            if dim <= length(comp.bases) && comp.bases[dim] !== nothing
                basis = comp.bases[dim]
                if isa(basis, RealFourier)
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

# ---------------------------------------------------------------------------
# Subsystem methods
# ---------------------------------------------------------------------------

function coeff_slices(subsystem::Subsystem, domain)
    if domain === nothing
        return ntuple(_ -> Colon(), length(subsystem.group))
    end

    coeff_shape = coefficient_shape(domain)
    return ntuple(_ -> Colon(), length(coeff_shape))
end

function coeff_shape(subsystem::Subsystem, domain)
    if domain === nothing
        return ()
    end

    return coefficient_shape(domain)
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

function field_domain(field::VectorField)
    # Get domain from first component
    if !isempty(field.components)
        return field_domain(field.components[1])
    end
    return nothing
end

function field_domain(field::TensorField)
    # Get domain from first component
    if !isempty(field.components)
        return field_domain(field.components[1])
    end
    return nothing
end

# Fallback for other Operand types
field_domain(::Operand) = nothing

function _coeff_data(field)
    ensure_layout!(field, :c)
    if get_coeff_data(field) === nothing
        throw(ArgumentError("Field $(field.name) has no coefficient data available."))
    end
    return get_coeff_data(field)
end

function _field_coeff_vector(field::ScalarField)
    data = _coeff_data(field)
    cpu_data = is_gpu_array(data) ? get_cpu_data(data) : data
    return vec(cpu_data)
end

function _assign_coefficients_from_slice!(field::ScalarField, coeffs::AbstractArray, slice::AbstractVector{<:Number})
    target_shape = size(coeffs)
    expected = prod(target_shape)

    if length(slice) != expected
        throw(ArgumentError("Slice length $(length(slice)) does not match coefficient size $expected for field $(field.name)"))
    end

    target_eltype = eltype(coeffs)
    data_array = if target_eltype <: Real
        if any(!iszero(imag(val)) for val in slice)
            @warn "Discarding imaginary part when assigning real coefficients for field $(field.name)"
        end
        reshape(real.(slice), target_shape)
    elseif target_eltype == eltype(slice)
        reshape(slice, target_shape)
    else
        reshape(convert.(target_eltype, slice), target_shape)
    end

    arch = field.dist.architecture
    if is_gpu(arch)
        copyto!(coeffs, on_architecture(arch, data_array))
    else
        copyto!(coeffs, data_array)
    end

    field.current_layout = :c
end

"""
    gather(subsystem, fields)

Gather coefficient data from fields into a single vector.
Following subsystems:213-220.
"""
function gather(subsystem::Subsystem, fields::Vector{<:ScalarField})
    buffers = ComplexF64[]
    for field in fields
        append!(buffers, _field_coeff_vector(field))
    end
    return buffers
end

"""
    scatter(subsystem, data, fields)

Scatter vector entries back into field coefficient arrays.
Following subsystems:222-231.
"""
function scatter(subsystem::Subsystem, data::AbstractVector, fields::Vector{<:ScalarField})
    offset = 0
    for field in fields
        coeffs = _coeff_data(field)
        n = length(coeffs)
        if offset + n > length(data)
            throw(ArgumentError("Insufficient data provided for scatter."))
        end
        _assign_coefficients_from_slice!(field, coeffs, view(data, offset+1:offset+n))
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
Following subsystems:234-296.

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

    # Preconditioners (following subsystems:560-563)
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

    # Per-(a_ii) LHS factorizations for IMEX RK.
    #
    # Keyed by the implicit diagonal coefficient `a_ii` so that ESDIRK
    # methods (RK222, RK443) — where every implicit stage shares the
    # same γ on the diagonal — cache and reuse a SINGLE factorization
    # across all stages. Previously this was a Vector{Any} indexed by
    # stage, which rebuilt the same LHS matrix N times per step for
    # N-stage ESDIRK methods.
    #
    # The key is `Float64` (the `a_ii` value). For RK222 there's only
    # one key (γ); for RK443 also one key (γ for stages 2-4; stage 1
    # has a_ii = 0 which hits the mass-only path and isn't cached
    # here). For non-ESDIRK DIRKs with different a_ii per stage, each
    # distinct a_ii gets its own cache entry.
    LHS_solvers::Dict{Float64, Any}

    # Woodbury block decomposition: bulk vs BC row/col indices (post-filtering,
    # into L_min/M_min). Used for efficient block-LU solving.
    bulk_rows::Vector{Int}
    bc_rows::Vector{Int}
    bulk_cols::Vector{Int}
    bc_cols::Vector{Int}
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
    if group === SUBSYSTEM_GROUP || (length(group) == 1 && group[1] === :global)
        group = length(subsystems) > 0 ? subsystems[1].group : ntuple(_ -> nothing, dist.dim)
    end

    # Cross-reference subsystems to this subproblem
    for subsystem in subsystems
        subsystem.subproblem[] = nothing  # Will be set after construction
    end

    # Build group dictionary (following subsystems:257-261)
    group_dict = Dict{String, Any}()
    for (axis, ax_group) in enumerate(group)
        if ax_group !== nothing && hasfield(typeof(dist), :coords)
            if axis <= length(dist.coords)
                coord_name = dist.coords[axis].name
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
        nothing, nothing,  # Buffers
        Dict{Float64, Any}(),  # LHS_solvers (keyed by a_ii for ESDIRK reuse)
        Int[], Int[], Int[], Int[]  # bulk_rows, bc_rows, bulk_cols, bc_cols
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

# ---------------------------------------------------------------------------
# Per-subproblem field sizing
# ---------------------------------------------------------------------------

"""
    _basis_coeff_size(basis::Basis) -> Int

Return the number of coefficient-space modes for a single basis.
RealFourier uses N/2+1 (rfft convention); all others use basis.meta.size.
"""
function _basis_coeff_size(basis::Basis)
    if isa(basis, RealFourier)
        return div(basis.meta.size, 2) + 1
    else
        return basis.meta.size
    end
end

"""
    subproblem_field_size(sp::Subproblem, field::ScalarField) -> Int

Return the number of DOFs that `field` contributes to this subproblem's
local matrix system.

For each basis dimension of the field:
- If the corresponding group entry is an `Int` (separable / single Fourier
  mode), that dimension contributes **1** DOF.
- If the corresponding group entry is `nothing` (coupled), that dimension
  contributes the full basis coefficient size.

Fields with no bases (0-D taus) return 1.
"""
function subproblem_field_size(sp::Subproblem, field::ScalarField)
    bases = field.bases
    # 0-D tau fields (no bases) contribute 1 DOF
    if isempty(bases)
        return 1
    end

    group = sp.group
    dofs = 1
    for (i, basis) in enumerate(bases)
        if basis === nothing
            # No basis in this dimension — contributes nothing extra
            continue
        end
        if i <= length(group) && group[i] isa Int
            # Separable dimension: single Fourier mode => 1 DOF
            dofs *= 1
        else
            # Coupled dimension: full basis coefficient size
            dofs *= _basis_coeff_size(basis)
        end
    end
    return dofs
end

"""
    subproblem_field_size(sp::Subproblem, field::VectorField) -> Int

Sum of per-component subproblem sizes (n_components * scalar size when all
components share the same bases).
"""
subproblem_field_size(sp::Subproblem, field::VectorField) =
    sum(subproblem_field_size(sp, comp) for comp in field.components)

"""
    subproblem_field_size(sp::Subproblem, field::TensorField) -> Int

Sum of per-component subproblem sizes over all tensor entries.
"""
subproblem_field_size(sp::Subproblem, field::TensorField) =
    sum(subproblem_field_size(sp, comp) for comp in vec(field.components))

# ---------------------------------------------------------------------------
# Per-subproblem equation sizing (following Dedalus subsystems.py:504)
# ---------------------------------------------------------------------------

"""Get Chebyshev basis from subproblem problem variables."""
function _subproblem_cheb_basis_from_sp(sp::Subproblem)
    for var in sp.problem.variables
        for comp in scalar_components(var)
            for basis in comp.bases
                if basis !== nothing && isa(basis, JacobiBasis)
                    return basis
                end
            end
        end
    end
    return nothing
end

"""Get coordinate system from an expression tree (traverses Future args too)."""
function _get_expr_coordsys(expr)
    hasfield(typeof(expr), :coordsys) && return expr.coordsys
    if hasfield(typeof(expr), :operand)
        cs = _get_expr_coordsys(expr.operand)
        cs !== nothing && return cs
    end
    # Future types (Add, Subtract, etc.) have args via future_args
    if isa(expr, Future)
        for arg in future_args(expr)
            cs = _get_expr_coordsys(arg)
            cs !== nothing && return cs
        end
    end
    if hasfield(typeof(expr), :left)
        cs = _get_expr_coordsys(expr.left)
        cs !== nothing && return cs
    end
    hasfield(typeof(expr), :right) && return _get_expr_coordsys(expr.right)
    return nothing
end

"""
    _expression_subproblem_dofs(sp, expr) -> Int

Compute the per-subproblem output DOFs of an expression by walking the tree.
Each operator type transforms the dimensionality:
- Interpolate/Integrate: removes the Chebyshev dimension (÷ Nz)
- Gradient: adds a vector dimension (× ndim)
- Divergence: removes a vector dimension (÷ ndim)
- Trace: removes two tensor dimensions (÷ ndim²)
- TimeDerivative, Laplacian, Lift, Negate: unchanged
- Add, Subtract, Multiply: max of children
"""
function _expression_subproblem_dofs(sp::Subproblem, expr)
    isa(expr, ScalarField) && return subproblem_field_size(sp, expr)
    isa(expr, VectorField) && return subproblem_field_size(sp, expr)
    isa(expr, TensorField) && return subproblem_field_size(sp, expr)

    if isa(expr, Interpolate)
        inner = _expression_subproblem_dofs(sp, expr.operand)
        cheb = _subproblem_cheb_basis_from_sp(sp)
        cheb !== nothing && return max(1, div(inner, cheb.meta.size))
        return 1
    end
    if isa(expr, Integrate)
        inner = _expression_subproblem_dofs(sp, expr.operand)
        cheb = _subproblem_cheb_basis_from_sp(sp)
        cheb !== nothing && return max(1, div(inner, cheb.meta.size))
        return 1
    end
    if isa(expr, Gradient)
        return _expression_subproblem_dofs(sp, expr.operand) * expr.coordsys.dim
    end
    if isa(expr, Divergence)
        inner = _expression_subproblem_dofs(sp, expr.operand)
        cs = _get_expr_coordsys(expr)
        return cs !== nothing ? max(1, div(inner, cs.dim)) : inner
    end
    if isa(expr, Trace)
        inner = _expression_subproblem_dofs(sp, expr.operand)
        cs = _get_expr_coordsys(expr)
        return cs !== nothing ? max(1, div(inner, cs.dim^2)) : inner
    end
    # Future hierarchy types (Add, Subtract, Multiply, Negate)
    # These have args accessed via future_args(f), not left/right fields.
    if isa(expr, Future)
        args = future_args(expr)
        if isempty(args)
            return 0
        end
        return maximum(_expression_subproblem_dofs(sp, a) for a in args)
    end
    # Single-operand operators: same size as operand
    if hasfield(typeof(expr), :operand)
        return _expression_subproblem_dofs(sp, expr.operand)
    end
    # Binary operators: max of children
    if hasfield(typeof(expr), :left) && hasfield(typeof(expr), :right)
        return max(_expression_subproblem_dofs(sp, expr.left),
                   _expression_subproblem_dofs(sp, expr.right))
    end
    # Constants, ZeroOperator, numbers
    return 0
end

"""
    _subproblem_eqn_size(sp::Subproblem, eq_data::Dict) -> Int

Compute per-subproblem output DOFs for an equation from its expression tree.
Tries lhs first, then L, then M expressions.
"""
function _subproblem_eqn_size(sp::Subproblem, eq_data::Dict)
    for key in ("lhs", "L", "M")
        expr = get(eq_data, key, nothing)
        if expr !== nothing
            sz = _expression_subproblem_dofs(sp, expr)
            sz > 0 && return sz
        end
    end
    return 0
end

function _subproblem_eqn_sizes(sp::Subproblem)
    cached = get(sp.matrices, "_eqn_sizes", nothing)
    cached !== nothing && return cached

    eqns = sp.problem.equation_data
    eqn_sizes = Int[_subproblem_eqn_size(sp, eq) for eq in eqns]
    sp.matrices["_eqn_sizes"] = eqn_sizes
    sp.matrices["_eqn_raw_size"] = sum(eqn_sizes; init=0)
    return eqn_sizes
end

function _subproblem_raw_eqn_size(sp::Subproblem)
    cached = get(sp.matrices, "_eqn_raw_size", nothing)
    cached !== nothing && return cached
    _subproblem_eqn_sizes(sp)
    return sp.matrices["_eqn_raw_size"]
end

function _subproblem_eqn_targets(sp::Subproblem, state_fields::Vector)
    cached = get(sp.matrices, "_eqn_targets", nothing)
    cached !== nothing && return cached

    problem = sp.problem
    eqns = problem.equation_data
    targets = Vector{Vector{Int}}(undef, length(eqns))
    for (eq_idx, eq_data) in enumerate(eqns)
        M_expr = get(eq_data, "M", nothing)
        targets[eq_idx] = if M_expr !== nothing && !_is_zero_m_term(M_expr)
            _find_time_derivative_targets(M_expr, state_fields, problem.variables)
        else
            Int[]
        end
    end
    sp.matrices["_eqn_targets"] = targets
    return targets
end

# ---------------------------------------------------------------------------
# Per-subproblem gather/scatter
# ---------------------------------------------------------------------------

"""Return the global 1-based Fourier mode index for this subproblem."""
function _kx_index_global(sp::Subproblem)
    for g in sp.group
        if g isa Integer
            return g + 1  # 0-based → 1-based
        end
    end
    return 1
end

"""
Convert a global 1-based Fourier mode index to a local index within the
coefficient data array. For serial runs, local == global. For MPI with
PencilArrays, the local buffer only holds this rank's modes.
"""
function _global_to_local_kx(kx_global::Int, field::ScalarField, sp::Subproblem)
    dist = sp.dist
    if dist === nothing || dist.size <= 1
        return kx_global
    end
    # Find the separable (Fourier) axis and its global size
    for (axis, g) in enumerate(sp.group)
        if g isa Integer && axis <= length(field.bases) && field.bases[axis] !== nothing
            basis = field.bases[axis]
            if isa(basis, FourierBasis)
                global_size = isa(basis, RealFourier) ? div(basis.meta.size, 2) + 1 : basis.meta.size
                local_range = local_indices(dist, axis, global_size)
                return kx_global - first(local_range) + 1
            end
        end
    end
    return kx_global
end

"""
Get the local coefficient data array. For PencilArrays (MPI), returns the
underlying local buffer via `parent()`. For regular arrays, returns as-is.
"""
_local_coeff_data(cd::AbstractArray) = get_local_data(cd)
_local_coeff_data(::Nothing) = nothing

function _subproblem_backend_matrix!(sp::Subproblem, matrix, cache_key::String, data::AbstractVector)
    matrix === nothing && return nothing
    if !is_gpu_array(data)
        return matrix
    end
    cached = get(sp.matrices, cache_key, nothing)
    cached !== nothing && return cached
    backend = _gpu_sparse_csr(matrix, eltype(matrix))
    sp.matrices[cache_key] = backend
    return backend
end

"""
    _bc_rows_device(sp, reference)

Return `sp.bc_rows` as an index vector on the same device as `reference`.

For CPU `reference`, returns the CPU `Vector{Int}` as-is. For a GPU
`reference`, returns (and caches) a device-side `Int` array so that vectorized
`rhs[bc_rows] .= coeff .* alg_f[bc_rows]` stays on-device and does not trigger
scalar indexing when `CUDA.allowscalar(false)` is set.
"""
function _bc_rows_device(sp::Subproblem, reference::AbstractVector)
    if isempty(sp.bc_rows) || !is_gpu_array(reference)
        return sp.bc_rows
    end
    cached = get(sp.matrices, "_bc_rows_device", nothing)
    if cached !== nothing && length(cached) == length(sp.bc_rows)
        return cached
    end
    device_rows = similar(reference, Int, length(sp.bc_rows))
    copyto!(device_rows, sp.bc_rows)
    sp.matrices["_bc_rows_device"] = device_rows
    return device_rows
end

"""
    apply_bc_override!(rhs, alg_f, sp, coeff)

Override the algebraic-constraint rows of `rhs` with `coeff * alg_f[bc_rows]`.

Pipeline:

    tmp .= alg_f[bc_rows]     # gather into cached scratch buffer
    tmp .*= coeff             # in-place scale
    rhs[bc_rows] = tmp        # scatter

All three operations are vectorized and stay on the same device as `rhs`, so
this is safe under `CUDA.allowscalar(false)`. The `bc_rows` index vector is
materialized on the correct device by `_bc_rows_device` and cached on the
subproblem to avoid per-step host→device transfers. The gather target
`tmp` is also a cached scratch buffer (`_bc_override_scratch!`), so this
function performs zero per-call allocation after the first call.
"""
function apply_bc_override!(rhs::AbstractVector, alg_f::AbstractVector,
                            sp::Subproblem, coeff)
    isempty(sp.bc_rows) && return rhs
    c = ComplexF64(coeff)
    if is_gpu_array(rhs)
        # GPU path — vectorized scatter via on-device index array. The
        # `alg_f[bc_rows]` gather allocates a small CuArray (length =
        # length(sp.bc_rows), typically ≤ 16), but the allocation is cheap
        # on modern device runtimes and cannot be avoided without a custom
        # kernel. The cached `_bc_rows_device` avoids per-call H2D.
        bc_rows = _bc_rows_device(sp, rhs)
        rhs[bc_rows] = c .* alg_f[bc_rows]
    else
        # CPU path — scalar loop, zero allocation. For small bc_rows counts
        # (typical: 5–20) this is faster than broadcast-gather anyway.
        bc_rows = sp.bc_rows  # CPU Vector{Int}
        @inbounds @simd for i in eachindex(bc_rows)
            r = bc_rows[i]
            rhs[r] = c * alg_f[r]
        end
    end
    return rhs
end

function _subproblem_selection_indices!(sp::Subproblem, matrix::Union{Nothing, SparseMatrixCSC},
                                        cache_key::String)
    matrix === nothing && return nothing
    cached = get(sp.matrices, cache_key, nothing)
    cached !== nothing && return cached

    m, n = size(matrix)
    if nnz(matrix) == 0
        indices = Int[]
        sp.matrices[cache_key] = indices
        return indices
    end

    rows, cols, vals = findnz(matrix)
    length(rows) == m || return nothing

    indices = Vector{Int}(undef, m)
    seen = falses(m)
    oneval = one(eltype(matrix))
    @inbounds for k in eachindex(rows)
        r = rows[k]
        c = cols[k]
        if r < 1 || r > m || c < 1 || c > n || seen[r] || vals[k] != oneval
            return nothing
        end
        seen[r] = true
        indices[r] = c
    end
    all(seen) || return nothing

    sp.matrices[cache_key] = indices
    return indices
end

function _assign_to_buffer!(dest::AbstractVector{ComplexF64}, src)
    if is_gpu_array(dest)
        src_dev = is_gpu_array(src) ? src : on_architecture(architecture(dest), Array(src))
        if eltype(src_dev) <: Complex
            dest .= src_dev
        else
            dest .= ComplexF64.(src_dev)
        end
    else
        src_cpu = is_gpu_array(src) ? Array(src) : src
        if eltype(src_cpu) <: Complex
            copyto!(dest, src_cpu)
        else
            copyto!(dest, ComplexF64.(src_cpu))
        end
    end
    return dest
end

function _assign_from_buffer!(dest, src)
    if is_gpu_array(dest)
        src_dev = is_gpu_array(src) ? src : on_architecture(architecture(dest), Array(src))
        if eltype(dest) <: Real
            dest .= real.(src_dev)
        else
            dest .= src_dev
        end
    else
        src_cpu = is_gpu_array(src) ? Array(src) : src
        if eltype(dest) <: Real
            dest .= real.(src_cpu)
        else
            dest .= src_cpu
        end
    end
    return dest
end

function _subproblem_cached_vector!(sp::Subproblem, key::String, n::Int;
                                    like::Union{Nothing, AbstractVector}=nothing)
    cached = get(sp.matrices, key, nothing)
    if cached !== nothing && length(cached) == n
        if like === nothing || is_gpu_array(cached) == is_gpu_array(like)
            return cached
        end
    end

    buffer = if like === nothing
        zeros(sp.dist.architecture, ComplexF64, n)
    else
        similar_zeros(like, ComplexF64, n)
    end
    sp.matrices[key] = buffer
    return buffer
end

"""Extract per-mode coefficients from all fields into a flat vector (no permutation)."""
function _gather_subproblem_raw!(buffer::AbstractVector{ComplexF64}, sp::Subproblem, fields::Vector)
    kx_global = _kx_index_global(sp)
    offset = 0
    for field in fields
        offset = _gather_field_raw!(buffer, offset, field, kx_global, sp)
    end
    return buffer
end

function _gather_subproblem_raw(sp::Subproblem, fields::Vector)
    total = sum(subproblem_field_size(sp, field) for field in fields)
    buffer = zeros(sp.dist.architecture, ComplexF64, total)
    return _gather_subproblem_raw!(buffer, sp, fields)
end

function _gather_field_raw!(buffer::AbstractVector{ComplexF64}, offset::Int, field::ScalarField, kx_global::Int, sp::Subproblem)
    ensure_layout!(field, :c)
    cd_raw = get_coeff_data(field)
    if cd_raw === nothing
        n = subproblem_field_size(sp, field)
        # Vectorized zero-fill: works on CPU and GPU without scalar indexing.
        if n > 0
            fill!(view(buffer, offset + 1 : offset + n), ComplexF64(0))
        end
        return offset + n
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        dest = view(buffer, offset + 1:offset + 1)
        _assign_to_buffer!(dest, view(cd, 1:1))
        return offset + 1
    elseif ndims(cd) == 1
        # 1D field (tau with Fourier basis): convert to local index
        kx_local = _global_to_local_kx(kx_global, field, sp)
        dest = view(buffer, offset + 1:offset + 1)
        if kx_local >= 1 && kx_local <= length(cd)
            _assign_to_buffer!(dest, view(cd, kx_local:kx_local))
        else
            fill!(dest, ComplexF64(0))
        end
        return offset + 1
    else
        # 2D field: extract local row across all Chebyshev modes
        kx_local = _global_to_local_kx(kx_global, field, sp)
        Nz = size(cd, 2)
        dest = view(buffer, offset + 1:offset + Nz)
        if kx_local >= 1 && kx_local <= size(cd, 1)
            _assign_to_buffer!(dest, selectdim(cd, 1, kx_local))
        else
            fill!(dest, ComplexF64(0))
        end
        return offset + Nz
    end
end

function _gather_field_raw!(buffer::AbstractVector{ComplexF64}, offset::Int, field::VectorField, kx_global::Int, sp::Subproblem)
    for comp in field.components
        offset = _gather_field_raw!(buffer, offset, comp, kx_global, sp)
    end
    return offset
end

"""Write per-mode coefficients back to fields from a flat vector (no permutation)."""
function _scatter_subproblem_raw(sp::Subproblem, data::AbstractVector, fields::Vector)
    kx_global = _kx_index_global(sp)
    offset = 0
    for field in fields
        offset = _scatter_field_raw!(field, data, offset, kx_global, sp)
    end
end

function _scatter_field_raw!(field::ScalarField, data::AbstractVector, offset::Int, kx_global::Int, sp::Subproblem)
    ensure_layout!(field, :c)
    cd_raw = get_coeff_data(field)
    if cd_raw === nothing
        return offset + subproblem_field_size(sp, field)
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        _assign_from_buffer!(view(cd, 1:1), view(data, offset + 1:offset + 1))
        return offset + 1
    elseif ndims(cd) == 1
        kx_local = _global_to_local_kx(kx_global, field, sp)
        if kx_local >= 1 && kx_local <= length(cd)
            _assign_from_buffer!(view(cd, kx_local:kx_local), view(data, offset + 1:offset + 1))
        end
        return offset + 1
    else
        kx_local = _global_to_local_kx(kx_global, field, sp)
        Nz = size(cd, 2)
        if kx_local >= 1 && kx_local <= size(cd, 1)
            _assign_from_buffer!(selectdim(cd, 1, kx_local), view(data, offset + 1:offset + Nz))
        end
        return offset + Nz
    end
end

function _scatter_field_raw!(field::VectorField, data::AbstractVector, offset::Int, kx_global::Int, sp::Subproblem)
    for comp in field.components
        offset = _scatter_field_raw!(comp, data, offset, kx_global, sp)
    end
    return offset
end

function compress_variable_space!(dest::AbstractVector, sp::Subproblem, raw::AbstractVector)
    if sp.pre_right_pinv !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(raw)) ?
                  _subproblem_selection_indices!(sp, sp.pre_right_pinv, "_pre_right_pinv_indices") :
                  nothing
        if indices !== nothing
            @inbounds for i in eachindex(indices)
                dest[i] = raw[indices[i]]
            end
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_right_pinv, "_pre_right_pinv_backend", raw)
            if !is_gpu_array(dest) && !is_gpu_array(raw) && pre isa AbstractMatrix
                mul!(dest, pre, raw)
            else
                _assign_to_buffer!(dest, pre * raw)
            end
        end
    else
        _assign_to_buffer!(dest, raw)
    end
    return dest
end

function compress_variable_space(sp::Subproblem, raw::AbstractVector)
    n = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 1) : length(raw)
    dest = _subproblem_cached_vector!(sp, "_compress_variable_space", n; like=raw)
    return compress_variable_space!(dest, sp, raw)
end

function expand_variable_space!(dest::AbstractVector, sp::Subproblem, data::AbstractVector)
    if sp.pre_right !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(data)) ?
                  _subproblem_selection_indices!(sp, sp.pre_right_pinv, "_pre_right_pinv_indices") :
                  nothing
        if indices !== nothing
            fill!(dest, zero(eltype(dest)))
            @inbounds for i in eachindex(indices)
                dest[indices[i]] = data[i]
            end
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_right, "_pre_right_backend", data)
            if !is_gpu_array(dest) && !is_gpu_array(data) && pre isa AbstractMatrix
                mul!(dest, pre, data)
            else
                _assign_to_buffer!(dest, pre * data)
            end
        end
    else
        _assign_to_buffer!(dest, data)
    end
    return dest
end

function expand_variable_space(sp::Subproblem, data::AbstractVector)
    n = sp.pre_right !== nothing ? size(sp.pre_right, 1) : length(data)
    dest = _subproblem_cached_vector!(sp, "_expand_variable_space", n; like=data)
    return expand_variable_space!(dest, sp, data)
end

function compress_equation_space!(dest::AbstractVector, sp::Subproblem, raw::AbstractVector)
    if sp.pre_left !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(raw)) ?
                  _subproblem_selection_indices!(sp, sp.pre_left, "_pre_left_indices") :
                  nothing
        if indices !== nothing
            @inbounds for i in eachindex(indices)
                dest[i] = raw[indices[i]]
            end
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_left, "_pre_left_backend", raw)
            if !is_gpu_array(dest) && !is_gpu_array(raw) && pre isa AbstractMatrix
                mul!(dest, pre, raw)
            else
                _assign_to_buffer!(dest, pre * raw)
            end
        end
    else
        _assign_to_buffer!(dest, raw)
    end
    return dest
end

function compress_equation_space(sp::Subproblem, raw::AbstractVector)
    n = sp.pre_left !== nothing ? size(sp.pre_left, 1) : length(raw)
    dest = _subproblem_cached_vector!(sp, "_compress_equation_space", n; like=raw)
    return compress_equation_space!(dest, sp, raw)
end

"""Gather per-mode coefficients and compress them into variable space."""
function gather_inputs(sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, "_gather_inputs_raw", raw_len)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space(sp, raw)
end

function gather_inputs!(dest::AbstractVector, sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, "_gather_inputs_raw", raw_len; like=dest)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space!(dest, sp, raw)
end

"""Expand from variable space and scatter back to fields."""
function scatter_inputs(sp::Subproblem, data::AbstractVector, fields::Vector)
    expanded_len = sp.pre_right !== nothing ? size(sp.pre_right, 1) : length(data)
    expanded = _subproblem_cached_vector!(sp, "_scatter_inputs_expanded", expanded_len; like=data)
    expand_variable_space!(expanded, sp, data)
    _scatter_subproblem_raw(sp, expanded, fields)
end

"""
Gather per-mode RHS coefficients in state/variable ordering.

`evaluate_rhs` returns one field per state variable, so the explicit RHS must be
compressed with the variable-side preconditioner, not the equation-side one.
"""
function gather_outputs(sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, "_gather_outputs_raw", raw_len)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space(sp, raw)
end

function gather_outputs!(dest::AbstractVector, sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, "_gather_outputs_raw", raw_len; like=dest)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space!(dest, sp, raw)
end

# ---------------------------------------------------------------------------
# Per-equation F gather (equation space, matches L_min rows)
#
# The subproblem stepper needs F in equation space — the same row ordering
# as `L_min`. Earlier `gather_outputs!` packed F in *variable* space, which
# (a) misaligned PDE rows and (b) silently dropped BC F values because BC
# equations have no time derivative.
#
# `gather_eqn_F!` walks the equations in the original order and packs:
#   - PDE rows: from `pde_F_fields[tidx]` (one entry per state field target)
#   - BC rows:  from the equation's own F expression (constants projected
#               onto the current Fourier mode)
# then applies `pre_left` to match L_min's filtered row space.
# ---------------------------------------------------------------------------

_is_zero_F_expr(::Nothing) = true
_is_zero_F_expr(::ZeroOperator) = true
_is_zero_F_expr(x::Number) = x == 0
_is_zero_F_expr(c::ConstantOperator) = c.value == 0
_is_zero_F_expr(::Any) = false

_extract_F_constant(c::ConstantOperator) = Float64(c.value)
_extract_F_constant(x::Number) = Float64(x)
_extract_F_constant(::Any) = nothing

"""
    _evaluate_alg_F(F_expr, sp) -> ComplexF64

Dispatch-based evaluation of an algebraic-equation F expression for a given
subproblem. Returns the complex value to write into the BC row of the raw
equation-space vector.

Currently supports:
- `ZeroOperator` / `nothing` → 0
- `ConstantOperator` / `Number` → `v * Nx` at DC, 0 elsewhere
- `ArrayOperator` → unnormalized RFFT of the grid-space array, picked at the
  subproblem's Fourier mode (for space-dependent BCs)
"""
_evaluate_alg_F(::Nothing, ::Subproblem) = ComplexF64(0)
_evaluate_alg_F(::ZeroOperator, ::Subproblem) = ComplexF64(0)
_evaluate_alg_F(c::ConstantOperator, sp::Subproblem) = _bc_constant_projection(Float64(c.value), sp)
_evaluate_alg_F(x::Number, sp::Subproblem) = _bc_constant_projection(Float64(x), sp)
_evaluate_alg_F(a::ArrayOperator, sp::Subproblem) = _bc_array_projection(a.value, sp)
function _evaluate_alg_F(expr, sp::Subproblem)
    @debug "gather_alg_F!: unsupported F expression type" expr_type=typeof(expr)
    return ComplexF64(0)
end

"""
Project a constant value onto the current subproblem's Fourier mode.

Tarang uses unnormalized `FFTW.rfft` / `FFTW.fft` along each separable
axis. For a constant `v` in grid space, the only nonzero Fourier
coefficient is at the full-DC mode `(kx=0, ky=0, ...)`, with value
`v * Nx * Ny * ...` (product of all Fourier-axis grid sizes). All
non-DC Fourier modes project to zero.

For problems without any Fourier axis (e.g. a pure-coupled BVP), the
DC-mode value is `v` itself — no scaling applied.
"""
function _bc_constant_projection(v::Float64, sp::Subproblem)
    v == 0 && return ComplexF64(0)
    # Every separable (Fourier) axis of this subproblem must be at its DC
    # mode (global index 1) for a constant to have any contribution.
    fourier_idx = _subproblem_fourier_group_indices(sp)
    for k in fourier_idx
        if k != 1
            return ComplexF64(0)
        end
    end
    # DC on every Fourier axis → `v * ∏ N_k` via unnormalized FFTs.
    scale = 1.0
    for N in _bc_fourier_axis_sizes(sp)
        scale *= Float64(N)
    end
    return ComplexF64(v * scale)
end

function _find_bc_fourier_basis(sp::Subproblem)
    for var in sp.problem.variables
        for comp in scalar_components(var)
            for basis in comp.bases
                if basis !== nothing && isa(basis, FourierBasis)
                    return basis
                end
            end
        end
    end
    return nothing
end

"""
    _bc_fourier_axis_sizes(sp) -> Vector{Int}

Ordered list of grid sizes for the problem's separable (Fourier) axes. Used
to determine the expected output shape of a BC array so that lower-rank
user inputs (e.g. `sin(x)` in a 3D problem) can be broadcast to the full
output before being transformed.
"""
function _bc_fourier_axis_sizes(sp::Subproblem)
    sizes = Int[]
    seen = Set{String}()
    for var in sp.problem.variables
        for comp in scalar_components(var)
            for basis in comp.bases
                basis === nothing && continue
                isa(basis, FourierBasis) || continue
                label = String(basis.meta.element_label)
                label in seen && continue
                push!(seen, label)
                push!(sizes, basis.meta.size)
            end
        end
    end
    return sizes
end

"""
    _subproblem_fourier_group_indices(sp) -> Vector{Int}

Return the 1-based Fourier mode indices for every separable axis of this
subproblem. For a 2D problem with one Fourier axis this is `[kx_global]`;
for a 3D problem it's `[kx_global, ky_global]`.
"""
function _subproblem_fourier_group_indices(sp::Subproblem)
    idx = Int[]
    for g in sp.group
        g isa Integer || continue
        push!(idx, g + 1)
    end
    return idx
end

"""
    _bc_array_projection(arr, sp)

Project a grid-space array `arr` (from a space-dependent BC) onto the current
subproblem's Fourier mode. Returns a `ComplexF64` value suitable for writing
into the BC row of the raw equation-space vector.

Handles several shape conventions for `arr`:
- `arr` is already the full BC output shape (1D for 2D problems, 2D for
  3D-with-two-periodic-axes problems, etc.) — FFT it directly.
- `arr` is lower-dimensional than the output (e.g., a 1D `sin(x)` in a 3D
  `(x, y, z)` problem) — broadcast it to the full Fourier output shape
  under the assumption that axes beyond `ndims(arr)` are constant.
- `arr` is a scalar-like 1-element array — treat as constant (DC only).

The broadcast/FFT result is cached by identity on `sp.problem.parameters`
via an `IdDict`, so all subproblems sharing the same `ArrayOperator` reuse
a single FFT per refresh.
"""
function _bc_array_projection(arr::AbstractArray, sp::Subproblem)
    (arr === nothing || length(arr) == 0) && return ComplexF64(0)

    fourier_sizes = _bc_fourier_axis_sizes(sp)
    if isempty(fourier_sizes)
        # No Fourier axes at all (pure-coupled / BVP-like). Use arr[1] as
        # the DC-mode value.
        return ComplexF64(first(arr))
    end

    coeffs = _get_or_compute_bc_array_coeffs!(arr, sp, fourier_sizes)
    coeffs === nothing && return ComplexF64(0)

    fourier_idx = _subproblem_fourier_group_indices(sp)
    if ndims(coeffs) == 1
        kx = isempty(fourier_idx) ? 1 : first(fourier_idx)
        return (kx >= 1 && kx <= length(coeffs)) ?
               ComplexF64(coeffs[kx]) : ComplexF64(0)
    elseif ndims(coeffs) == 2
        length(fourier_idx) >= 2 || return ComplexF64(0)
        kx, ky = fourier_idx[1], fourier_idx[2]
        return (1 <= kx <= size(coeffs, 1) && 1 <= ky <= size(coeffs, 2)) ?
               ComplexF64(coeffs[kx, ky]) : ComplexF64(0)
    elseif ndims(coeffs) == 3
        length(fourier_idx) >= 3 || return ComplexF64(0)
        kx, ky, kz = fourier_idx[1], fourier_idx[2], fourier_idx[3]
        return (1 <= kx <= size(coeffs, 1) &&
                1 <= ky <= size(coeffs, 2) &&
                1 <= kz <= size(coeffs, 3)) ?
               ComplexF64(coeffs[kx, ky, kz]) : ComplexF64(0)
    else
        @warn "BC array projection: unsupported coefficient rank $(ndims(coeffs))" maxlog=3
        return ComplexF64(0)
    end
end

"""
    _get_or_compute_bc_array_coeffs!(arr, sp, fourier_sizes) -> coefficients

Cache-backed helper that:
1. Reshapes/broadcasts `arr` to the full Fourier-output shape (derived from
   `fourier_sizes`) so lower-rank inputs work in higher-dim problems.
2. Takes an unnormalized forward FFT along the first axis (via `rfft` for
   real input) and complex FFT along remaining axes.
3. Returns the complex coefficient array for downstream indexing.

The result is cached on `sp.problem.parameters["_bc_rfft_cache"]` by array
object identity (`IdDict`) so all subproblems reusing the same `arr` share
a single FFT per refresh.
"""
function _get_or_compute_bc_array_coeffs!(arr::AbstractArray,
                                          sp::Subproblem,
                                          fourier_sizes::Vector{Int})
    cache = get(sp.problem.parameters, "_bc_rfft_cache", nothing)
    if cache === nothing
        cache = IdDict{Any, Any}()
        sp.problem.parameters["_bc_rfft_cache"] = cache
    end
    cached = get(cache, arr, nothing)
    cached !== nothing && return cached

    coeffs = try
        broadcast_arr = _broadcast_bc_array_to_output(arr, fourier_sizes)
        _forward_fft_bc(broadcast_arr)
    catch err
        @warn "BC array FFT failed: $err" maxlog=1
        return nothing
    end

    cache[arr] = coeffs
    return coeffs
end

"""
    _broadcast_bc_array_to_output(arr, fourier_sizes) -> Array

Broadcast a grid-space BC array to the full Fourier-output shape.

- If `arr` already matches `fourier_sizes`, returns it (as a concrete array).
- If `arr` has fewer dimensions, reshape it into a singleton-padded shape
  and broadcast to the full shape. We assume the user supplies the array
  in axis order matching the problem's Fourier axes, so a 1D `arr` of
  length `fourier_sizes[k]` is broadcast along the matching dimension and
  replicated along the rest.
- A single-element `arr` becomes a uniform constant over the full shape.
"""
function _broadcast_bc_array_to_output(arr::AbstractArray, fourier_sizes::Vector{Int})
    output_shape = Tuple(fourier_sizes)
    ndout = length(fourier_sizes)

    # Scalar-ish input (length 1) → constant over the output.
    if length(arr) == 1
        result = Array{Float64}(undef, output_shape...)
        fill!(result, Float64(first(arr)))
        return result
    end

    # Exact match (shape and rank) → collect to a concrete array to keep
    # downstream FFT predictable.
    if size(arr) == output_shape
        return collect(Float64.(arr))
    end

    # Same rank, same total length but different dim ordering (unusual).
    if length(arr) == prod(output_shape)
        return reshape(collect(Float64.(arr)), output_shape...)
    end

    # Lower-dimensional input: pad its shape with trailing singletons so
    # that Julia's `broadcast` can expand it along the missing axes.
    nd_in = ndims(arr)
    if nd_in < ndout
        # Try to match each input dimension to an output dimension of the
        # same size; fall back to leading-dim match.
        dims_padded = ntuple(i -> i <= nd_in ? size(arr, i) : 1, ndout)
        if dims_padded[1] == output_shape[1] || any(i -> dims_padded[i] == output_shape[i], 1:nd_in)
            reshaped = reshape(collect(Float64.(arr)), dims_padded)
            target = Array{Float64}(undef, output_shape...)
            target .= reshaped
            return target
        end
    end

    # Length matches a single Fourier axis but rank is 1 — broadcast along
    # the first axis with that size. (Covers 1D `sin(x)` in 3D problems.)
    if nd_in == 1
        for (axis, sz) in enumerate(output_shape)
            if length(arr) == sz
                # Reshape to have the Fourier length on `axis`, singletons elsewhere.
                rshape = ntuple(i -> i == axis ? sz : 1, ndout)
                reshaped = reshape(collect(Float64.(arr)), rshape)
                target = Array{Float64}(undef, output_shape...)
                target .= reshaped
                return target
            end
        end
    end

    throw(ArgumentError(
        "BC array shape $(size(arr)) incompatible with Fourier output " *
        "shape $(output_shape); provide the array in either the full " *
        "output shape or a 1-D/1-element form that can be broadcast."
    ))
end

"""
    _forward_fft_bc(arr) -> complex coefficient array

Unnormalized forward Fourier transform matching Tarang's `RealFourier`
convention: `FFTW.rfft` along the first dimension for real input, full
`FFTW.fft` for complex input. Multi-dim real input transforms the first
dim with `rfft` and remaining dims with `fft`, matching the shape that
`_bc_fourier_axis_sizes` + `_subproblem_fourier_group_indices` expect
when looking up per-mode coefficients.
"""
function _forward_fft_bc(arr::AbstractArray)
    if eltype(arr) <: Complex
        return FFTW.fft(ComplexF64.(arr))
    else
        return FFTW.rfft(Float64.(arr))
    end
end

"""
    invalidate_bc_array_cache!(problem)

Clear the per-problem BC-array FFT cache. Call when BC arrays change (e.g.
at the start of each step, or after evaluating new time-dependent array
values via `_apply_bc_values_to_equations!`).
"""
function invalidate_bc_array_cache!(problem)
    cache = get(problem.parameters, "_bc_rfft_cache", nothing)
    cache === nothing && return
    if cache isa IdDict
        empty!(cache)
    end
    return
end

"""
    gather_eqn_F!(dest, sp, solver, pde_F_fields, state_fields)

Pack PDE-equation F values into equation-space. For each equation with a time
derivative (`M` term), pull F from `pde_F_fields` at the equation's target
state-field indices. Algebraic/BC equations (no `M` term) contribute ZERO to
this vector — they are handled separately via `gather_alg_F!` + a direct RHS
override in the stepper, because the IMEX-RK accumulated formula gives the
wrong `1/γ` scaling for inhomogeneous algebraic constraints.
"""
function gather_eqn_F!(dest::AbstractVector{ComplexF64}, sp::Subproblem, solver,
                       pde_F_fields::Vector, state_fields::Vector)
    problem = sp.problem
    eqns = problem.equation_data

    eqn_sizes = _subproblem_eqn_sizes(sp)
    eqn_targets = _subproblem_eqn_targets(sp, state_fields)
    I_raw = _subproblem_raw_eqn_size(sp)

    raw = _subproblem_cached_vector!(sp, "_gather_eqn_F_raw", I_raw; like=dest)
    fill!(raw, zero(eltype(raw)))

    kx_global = _kx_index_global(sp)

    i0 = 0
    for (eq_idx, eq_data) in enumerate(eqns)
        eq_size = eqn_sizes[eq_idx]
        if eq_size == 0
            continue
        end

        target_indices = eqn_targets[eq_idx]
        if !isempty(target_indices)
            offset = i0
            for tidx in target_indices
                if tidx >= 1 && tidx <= length(pde_F_fields)
                    fld = pde_F_fields[tidx]
                    if fld !== nothing
                        offset = _gather_field_raw!(raw, offset, fld, kx_global, sp)
                        continue
                    end
                end
                if tidx >= 1 && tidx <= length(state_fields)
                    offset += subproblem_field_size(sp, state_fields[tidx])
                end
            end
        end
        # Algebraic rows intentionally left zero — see gather_alg_F!.

        i0 += eq_size
    end

    compress_equation_space!(dest, sp, raw)
    return dest
end

"""
    gather_alg_F!(dest, sp)

Pack algebraic-constraint F values (from BC / constraint equations that have
no time derivative) into equation-space, with zeros at PDE rows.

For each equation without an `M` term, evaluate its stored `F` expression
(typically a `ConstantOperator`) and project onto the current Fourier mode.
The result is used by the stepper to OVERRIDE the BC rows of the RHS with
`dt * a_ii * F_alg`, yielding `L_row * X = F_alg` at each stage — the correct
enforcement of the algebraic constraint.

This override is necessary because the standard IMEX-RK accumulated RHS
formula gives `L_row * X = (A^E[i,j]/a_ii) * F_BC` which is wrong by a factor
of `1/γ` for inhomogeneous algebraic constraints.
"""
function gather_alg_F!(dest::AbstractVector{ComplexF64}, sp::Subproblem)
    problem = sp.problem
    eqns = problem.equation_data

    eqn_sizes = _subproblem_eqn_sizes(sp)
    I_raw = _subproblem_raw_eqn_size(sp)

    # Build the sparse BC F vector on the HOST via scalar writes (a few nonzero
    # entries at BC row offsets, the rest zero). Then upload once into the
    # device-resident `raw` buffer via `_assign_to_buffer!` — this keeps
    # scalar indexing off of GPU arrays so the helper is safe under
    # `CUDA.allowscalar(false)`.
    raw_cpu_key = "_gather_alg_F_raw_cpu"
    raw_cpu = get(sp.matrices, raw_cpu_key, nothing)
    if raw_cpu === nothing || length(raw_cpu) != I_raw
        raw_cpu = zeros(ComplexF64, I_raw)
        sp.matrices[raw_cpu_key] = raw_cpu
    else
        fill!(raw_cpu, zero(ComplexF64))
    end

    i0 = 0
    for (eq_idx, eq_data) in enumerate(eqns)
        eq_size = eqn_sizes[eq_idx]
        if eq_size == 0
            continue
        end

        M_expr = get(eq_data, "M", nothing)
        is_alg = M_expr === nothing || _is_zero_m_term(M_expr)

        if is_alg
            F_expr = get(eq_data, "F_expr", nothing)
            if F_expr === nothing
                F_expr = get(eq_data, "F", nothing)
            end
            if !_is_zero_F_expr(F_expr)
                coeff = _evaluate_alg_F(F_expr, sp)
                if coeff != 0
                    # Replicate the value across all rows of the BC
                    # equation's block. For scalar BCs `eq_size == 1` and
                    # this writes a single entry; for vector BCs (e.g.
                    # `u(z=0) = c` with `u` a 2-component vector, `eq_size
                    # == 2`) the same coefficient is written to every
                    # component row. The Interpolate LHS for a vector
                    # operand is `kron(I_ncomp, row)`, so replicating the
                    # scalar F across rows enforces the same value on each
                    # component — which is what "u = c" means.
                    @inbounds for r in 1:eq_size
                        raw_cpu[i0 + r] = coeff
                    end
                end
            end
        end

        i0 += eq_size
    end

    # Upload the CPU-built raw vector into the device-resident raw buffer.
    raw = _subproblem_cached_vector!(sp, "_gather_alg_F_raw", I_raw; like=dest)
    _assign_to_buffer!(raw, raw_cpu)

    compress_equation_space!(dest, sp, raw)
    return dest
end

"""
    check_condition(sp::Subproblem, eq_data)

Check if equation condition is satisfied for this subproblem.
Following subsystems:494-495.
"""
function check_condition(sp::Subproblem, eq_data::Dict)
    condition = get(eq_data, "condition", "true")
    if condition == "true" || condition === nothing || condition == true
        return true
    end
    if condition == "false" || condition == false
        return false
    end

    # Evaluate condition expression with subproblem group dictionary
    # The condition is typically a string expression involving group indices
    # For example: "nx != 0" or "kx == 0 && ky == 0"
    group_dict = sp.group_dict

    # Simple boolean conditions
    if isa(condition, Bool)
        return condition
    end

    # String conditions - evaluate with group variables
    if isa(condition, String)
        condition_str = strip(condition)

        # Handle compound conditions with && and ||
        if occursin("&&", condition_str)
            parts = split(condition_str, "&&")
            return all(check_condition(strip(p), group_dict) for p in parts)
        elseif occursin("||", condition_str)
            parts = split(condition_str, "||")
            return any(check_condition(strip(p), group_dict) for p in parts)
        end

        # Parse simple comparisons like "nx != 0", "kx == 0"
        if occursin("!=", condition_str)
            parts = split(condition_str, "!=")
            if length(parts) == 2
                var_name = strip(parts[1])
                value = tryparse(Int, strip(parts[2]))
                if value !== nothing && haskey(group_dict, var_name)
                    return group_dict[var_name] != value
                elseif value !== nothing && haskey(group_dict, Symbol(var_name))
                    return group_dict[Symbol(var_name)] != value
                end
            end
        elseif occursin("==", condition_str)
            parts = split(condition_str, "==")
            if length(parts) == 2
                var_name = strip(parts[1])
                value = tryparse(Int, strip(parts[2]))
                if value !== nothing && haskey(group_dict, var_name)
                    return group_dict[var_name] == value
                elseif value !== nothing && haskey(group_dict, Symbol(var_name))
                    return group_dict[Symbol(var_name)] == value
                end
            end
        end

        # Unparseable condition — warn loudly rather than silently assuming true
        @warn "Could not parse condition expression: '$condition_str'. Assuming true (equation included)." maxlog=3
    end

    return true
end

"""
    valid_modes(sp::Subproblem, field, valid_modes_array)

Get valid modes for field in this subproblem.
Following subsystems:476-478.
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
Following subsystems:55-70.
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

    # Optionally reorder local subproblems by estimated cost (heaviest
    # first). Enables better pipelining with multi-threaded sparse LU
    # backends because the slowest factorizations start first and run in
    # parallel with the cheaper ones. Opt-in via
    # `SolverConfig.balance_local_cost = true` — default keeps the
    # natural kx-ordering for cache locality.
    config = hasfield(typeof(solver), :config) ? solver.config : SolverConfig()
    if config.balance_local_cost && length(subproblems) > 1
        # `_sp_local_cost` uses `nnz(L) + nnz(M) + n²/√n + length(bc_rows)`
        # as a simple proxy for per-subproblem cost. Only valid AFTER
        # matrices are built; if `build_matrices` was nothing we skip.
        if build_matrices !== nothing
            costs = [_sp_local_cost(sp) for sp in subproblems]
            order = sortperm(costs; rev=true)
            subproblems = Tuple(subproblems[i] for i in order)
        end
    end

    return subproblems
end

"""Estimate per-subproblem cost from matrix structure. Must be called
after `build_matrices!` has populated `sp.L_min` / `sp.M_min`."""
function _sp_local_cost(sp::Subproblem)
    sp.M_min === nothing && return 0.0
    n = size(sp.M_min, 1)
    nnz_L = sp.L_min === nothing ? 0 : nnz(sp.L_min)
    nnz_M = nnz(sp.M_min)
    bc_cost = length(sp.bc_rows)
    return Float64(n)^2 / max(sqrt(Float64(n)), 1.0) +
           Float64(nnz_L + nnz_M) +
           Float64(bc_cost)
end

"""
    build_subproblem_matrices(solver, subproblems, matrices)

Build matrices for all subproblems.
Following subsystems:72-81.
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
Following arithmetic:370-373.
"""
function gather_ncc_coeffs!(expr)
    # Recursively gather NCC coefficients (Non-Constant Coefficients)
    # NCCs are spatially-varying coefficients that require special matrix treatment
    if hasfield(typeof(expr), :operand)
        gather_ncc_coeffs!(expr.operand)
    end
    if hasfield(typeof(expr), :operands)
        for op in expr.operands
            gather_ncc_coeffs!(op)
        end
    end

    # Store NCC data if this is an NCC multiply
    # NCCs arise when a spatially-varying field multiplies another field
    # They require expansion in the basis and convolution in coefficient space
    if hasfield(typeof(expr), :ncc_data) && expr.ncc_data !== nothing
        ncc_data = expr.ncc_data

        # Get the NCC field coefficients
        if hasfield(typeof(ncc_data), :coeffs) && ncc_data.coeffs !== nothing
            # Coefficients already gathered
            return
        end

        # Gather coefficients from the NCC field
        if hasfield(typeof(ncc_data), :field)
            ncc_field = ncc_data.field
            if hasproperty(ncc_field, :data_c)
                # Transform to coefficient space if needed
                if hasproperty(ncc_field, :current_layout) && ncc_field.current_layout != :c
                    forward_transform!(ncc_field)
                end
                # Store the coefficient data
                if get_coeff_data(ncc_field) !== nothing
                    ncc_data.coeffs = copy(get_coeff_data(ncc_field))
                end
            end
        end
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
Following subsystems:497-596.
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

    # Compute per-subproblem sizes.
    # Following Dedalus (subsystems.py:504): eqn_sizes = [sp.field_size(eqn['eqn']) for eqn in eqns]
    # Each equation's output size is determined by the LHS expression's structure,
    # not by the corresponding variable's size.
    eqn_conditions = [check_condition(sp, eq) for eq in eqns]
    var_sizes = [subproblem_field_size(sp, var) for var in vars]
    eqn_sizes = [_subproblem_eqn_size(sp, eq) for eq in eqns]
    @debug "build_matrices! group=$(sp.group): eqn_sizes=$eqn_sizes, var_sizes=$var_sizes, I=$(sum(eqn_sizes)), J=$(sum(var_sizes))"
    I = sum(eqn_sizes)
    J = sum(var_sizes)

    # Construct matrices (following subsystems:512-537)
    matrices = Dict{String, SparseMatrixCSC}()
    for name in names
        name_str = String(name)
        data, rows, cols = ComplexF64[], Int[], Int[]

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
                                # Validate block dimensions
                                if !isempty(block_rows) && (maximum(block_rows) > eqn_size || maximum(block_cols) > var_size)
                                    var_name = hasfield(typeof(var), :name) ? var.name : "?"
                                    eq_str = get(eq_data, "equation_string", "?")
                                    @error "Block size mismatch in eq$eq_idx($name_str) × var$var_idx($var_name): " *
                                           "block=$(size(block)), expected rows≤$eqn_size cols≤$var_size, " *
                                           "actual max_row=$(maximum(block_rows)) max_col=$(maximum(block_cols)). " *
                                           "Equation: $eq_str"
                                    continue  # skip this block to avoid crash
                                end
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

    # ── Dedalus-style valid mode filtering ──────────────────────────────
    # For non-DC modes, gauge constraints like integ(p)=0 produce zero rows
    # (the integral of a non-DC Fourier mode over x is zero), leaving the
    # paired tau variable (tau_p) as a free parameter — a genuine gauge
    # degree of freedom that Dedalus excludes via valid_modes.
    #
    # Approach (following Dedalus subsystems.py:539-563):
    # 1. Detect equation rows that are all-zero in both L and M (trivially
    #    satisfied, e.g., integ(p)=0 for non-DC).
    # 2. For each zero row, find the paired 0D tau variable: the 1-DOF
    #    variable whose total column norm is SMALLEST (the least-used tau
    #    is the gauge Lagrange multiplier).
    # 3. Mark both the row and column invalid.
    # 4. Apply permutation matrices via drop_empty_rows to filter them out,
    #    giving a smaller square sparse system that can use fast sparse LU.
    valid_eqn = ones(Bool, I)
    valid_var = ones(Bool, J)

    if haskey(matrices, "L")
        L_raw = matrices["L"]
        M_raw = get(matrices, "M", nothing)

        # Compute column norms for all 0D/1D tau variables (candidates for removal)
        tau_cols = Int[]
        tau_norms = Float64[]
        j0 = 0
        for (vi, (var, vsz)) in enumerate(zip(vars, var_sizes))
            if vsz == 1
                col = j0 + 1
                n = sum(abs.(L_raw[:, col]))
                if M_raw !== nothing
                    n += sum(abs.(M_raw[:, col]))
                end
                push!(tau_cols, col)
                push!(tau_norms, n)
            end
            j0 += vsz
        end

        # For each zero row, pair it with the least-used tau column
        used_cols = Set{Int}()
        for row in 1:I
            L_nnz = count(!iszero, L_raw[row, :])
            M_nnz = M_raw !== nothing ? count(!iszero, M_raw[row, :]) : 0
            if L_nnz == 0 && M_nnz == 0
                # Find the smallest-norm unused tau column
                best_col = 0
                best_norm = Inf
                for (ci, col) in enumerate(tau_cols)
                    if col in used_cols
                        continue
                    end
                    if tau_norms[ci] < best_norm
                        best_norm = tau_norms[ci]
                        best_col = col
                    end
                end
                if best_col > 0
                    valid_eqn[row] = false
                    valid_var[best_col] = false
                    push!(used_cols, best_col)
                end
            end
        end
    end

    # Build filter matrices
    valid_eqn_mat = spdiagm(0 => ComplexF64.(valid_eqn))
    valid_var_mat = spdiagm(0 => ComplexF64.(valid_var))

    # Dedalus-style permutations before dropping invalid rows/columns.
    # This preserves the grouped equation/variable ordering needed by bordered
    # formulations and keeps gather/scatter consistent with the compressed space.
    left_perm = left_permutation(sp, eqns, eqn_sizes, bc_top, interleave_components)
    right_perm = right_permutation(sp, vars, tau_left, interleave_components)

    # Preconditioners: permutation + valid-mode filtering (Dedalus subsystems.py:560-563)
    sp.pre_left = drop_empty_rows(left_perm * valid_eqn_mat)
    sp.pre_left_pinv = sparse(sp.pre_left')
    sp.pre_right_pinv = drop_empty_rows(right_perm * valid_var_mat)
    sp.pre_right = sparse(sp.pre_right_pinv')

    # Apply permutations: L_min = pre_left * L * pre_right (Dedalus subsystems.py:569-571)
    for (name, matrix) in matrices
        matrices[name] = sp.pre_left * matrix * sp.pre_right
    end

    n_valid_eqn = sum(valid_eqn)
    n_valid_var = sum(valid_var)
    if n_valid_eqn != n_valid_var
        @warn "Non-square filtered system: group=$(sp.group), valid_eqn=$n_valid_eqn, valid_var=$n_valid_var" maxlog=3
    end

    # Store minimal CSR matrices (following subsystems:573-575)
    sp.matrices = matrices
    if haskey(matrices, "L")
        sp.L_min = matrices["L"]
    end
    if haskey(matrices, "M")
        sp.M_min = matrices["M"]
    end

    # ── Woodbury block classification ───────────────────────────────────
    # Classify rows/columns after permutation + valid-mode filtering.
    #
    # Rows: all DOFs belonging to highest-dimensional equations are "bulk";
    # lower-dimensional equations are the BC block.
    #
    # Columns: highest-dimensional variables are bulk by construction. For
    # lower-dimensional variables (taus), keep any column that couples into a
    # bulk row in the bulk block as well. Only "pure BC" taus, which have no
    # support on bulk rows, stay in the BC block.
    #
    # This matches the Dedalus requirement that coupling taus stay in the bulk;
    # a size-only split is incorrect for first-order/tau formulations.
    if haskey(matrices, "L") && !isempty(eqn_sizes) && !isempty(var_sizes)
        row_order = [idx for idx in _left_permutation_indices(sp, eqns, eqn_sizes, bc_top, interleave_components) if valid_eqn[idx]]
        col_order = [idx for idx in _right_permutation_indices(sp, vars, tau_left, interleave_components) if valid_var[idx]]

        cheb_basis = _subproblem_cheb_basis_from_sp(sp)
        Nz = cheb_basis !== nothing ? cheb_basis.meta.size : 1
        is_bulk_size(sz) = Nz > 1 ? (sz >= Nz && sz % Nz == 0) : sz > 1

        row_is_bulk = Vector{Bool}(undef, I)
        offset = 0
        for sz in eqn_sizes
            if sz > 0
                row_is_bulk[offset + 1:offset + sz] .= is_bulk_size(sz)
            end
            offset += sz
        end

        col_is_bulk = Vector{Bool}(undef, J)
        offset = 0
        for sz in var_sizes
            if sz > 0
                col_is_bulk[offset + 1:offset + sz] .= is_bulk_size(sz)
            end
            offset += sz
        end

        if !isempty(row_order) && !isempty(col_order)
            post_row_is_bulk = row_is_bulk[row_order]
            post_col_is_bulk = col_is_bulk[col_order]

            sp.bulk_rows = findall(identity, post_row_is_bulk)
            sp.bc_rows = findall(!, post_row_is_bulk)

            bulk_row_mask = falses(length(post_row_is_bulk))
            bulk_row_mask[sp.bulk_rows] .= true

            bulk_cols = Int[]
            bc_cols = Int[]
            L_min = matrices["L"]
            M_min = get(matrices, "M", nothing)
            for col in eachindex(post_col_is_bulk)
                if post_col_is_bulk[col]
                    push!(bulk_cols, col)
                    continue
                end

                touches_bulk = _column_touches_rows(L_min, col, bulk_row_mask)
                if !touches_bulk && M_min !== nothing
                    touches_bulk = _column_touches_rows(M_min, col, bulk_row_mask)
                end

                if touches_bulk
                    push!(bulk_cols, col)
                else
                    push!(bc_cols, col)
                end
            end

            sp.bulk_cols = bulk_cols
            sp.bc_cols = bc_cols
        else
            empty!(sp.bulk_rows)
            empty!(sp.bc_rows)
            empty!(sp.bulk_cols)
            empty!(sp.bc_cols)
        end
    end

    # Store expanded matrices for IMEX in-place LHS updates (Dedalus pattern).
    # Pre-allocate sp.LHS with the union of M and L sparsity patterns, then
    # store M_exp and L_exp that have this same pattern. At timestep time,
    # we can update sp.LHS in-place via:
    #   sp.LHS.nzval .= a0 * M_exp.nzval + b0 * L_exp.nzval
    # avoiding matrix allocation and sparsity re-analysis.
    if length(matrices) > 1 && store_expanded && haskey(matrices, "L") && haskey(matrices, "M")
        L_min = matrices["L"]
        M_min = matrices["M"]
        sp.LHS = zeros_with_pattern(L_min, M_min)
        sp.L_exp = expand_pattern(L_min, sp.LHS)
        sp.M_exp = expand_pattern(M_min, sp.LHS)
    else
        # Minimal init for shape access / solver compatibility
        rows = haskey(matrices, "L") ? size(matrices["L"], 1) : 0
        cols = haskey(matrices, "L") ? size(matrices["L"], 2) : 0
        sp.LHS = spzeros(dtype, rows, cols)
    end

    # Compute update rank for Woodbury formula
    sp.update_rank = compute_update_rank(sp, eqns, eqn_conditions, eqn_sizes)

    return nothing
end

function _coord_name(coord)
    isa(coord, Tuple) && !isempty(coord) && return _coord_name(coord[1])
    return isa(coord.name, Symbol) ? String(coord.name) : String(coord.name)
end

function _subproblem_reduce_dofs(sp::Subproblem, inner::Int, operand, coord; interpolate::Bool=false)
    coord_name = _coord_name(coord)
    basis = _operand_basis_for_coord(operand, coord_name)
    basis === nothing && return inner

    if basis isa FourierBasis
        if interpolate
            return inner
        end
        group_entry = _subproblem_group_index(sp, coord_name)
        return (group_entry isa Integer && group_entry == 0) ? inner : 0
    end

    bsize = _basis_coeff_size(basis)
    if bsize <= 0 || inner == 0
        return inner
    end
    return inner % bsize == 0 ? div(inner, bsize) : inner
end

function _subproblem_expr_dofs(sp::Subproblem, expr)
    expr === nothing && return 0
    (isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator)) && return 0
    isa(expr, ScalarField) && return subproblem_field_size(sp, expr)
    isa(expr, VectorField) && return subproblem_field_size(sp, expr)
    isa(expr, TensorField) && return subproblem_field_size(sp, expr)

    if isa(expr, AddOperator) || isa(expr, SubtractOperator)
        return max(_subproblem_expr_dofs(sp, expr.left), _subproblem_expr_dofs(sp, expr.right))
    end
    if isa(expr, MultiplyOperator) || isa(expr, DivideOperator)
        return max(_subproblem_expr_dofs(sp, expr.left), _subproblem_expr_dofs(sp, expr.right))
    end
    isa(expr, NegateOperator) && return _subproblem_expr_dofs(sp, expr.operand)

    if isa(expr, Future)
        args = future_args(expr)
        return isempty(args) ? 0 : maximum(_subproblem_expr_dofs(sp, arg) for arg in args)
    end

    if isa(expr, Integrate) || isa(expr, Average)
        inner = _subproblem_expr_dofs(sp, expr.operand)
        coords = isa(expr.coord, Tuple) ? expr.coord : (expr.coord,)
        for coord in coords
            inner = _subproblem_reduce_dofs(sp, inner, expr.operand, coord; interpolate=false)
        end
        return inner
    end

    if isa(expr, Interpolate)
        return _subproblem_reduce_dofs(sp, _subproblem_expr_dofs(sp, expr.operand),
                                       expr.operand, expr.coord; interpolate=true)
    end

    if isa(expr, Trace)
        field = _resolve_operand_field(expr.operand)
        if isa(field, VectorField) && !isempty(field.components)
            return subproblem_field_size(sp, field.components[1])
        elseif isa(field, ScalarField)
            return subproblem_field_size(sp, field)
        end

        inner = _subproblem_expr_dofs(sp, expr.operand)
        ndim = _infer_ndim(expr.operand)
        if ndim > 0 && inner % (ndim * ndim) == 0
            return div(inner, ndim * ndim)
        elseif ndim > 0 && inner % ndim == 0
            return div(inner, ndim)
        end
        return inner
    end

    if hasfield(typeof(expr), :operand)
        mat = subproblem_matrix(expr, sp)
        if mat !== nothing
            return size(mat, 1)
        end
        return _subproblem_expr_dofs(sp, getfield(expr, :operand))
    end

    field = _resolve_operand_field(expr)
    field !== nothing && return subproblem_field_size(sp, field)
    return 0
end

function _has_only_zero_dim_bases(field::ScalarField)
    isempty(field.bases) && return true
    return all(b -> b === nothing, field.bases)
end

function _has_only_zero_dim_bases(field::VectorField)
    return all(_has_only_zero_dim_bases, field.components)
end

function _has_only_zero_dim_bases(field::TensorField)
    return all(_has_only_zero_dim_bases, vec(field.components))
end

function _is_zero_separable_group(sp::Subproblem)
    for group_entry in sp.group
        if group_entry isa Integer && group_entry != 0
            return false
        end
    end
    return true
end

"""
    get_valid_modes(eq_data, sp, num_modes)

Get valid modes array for equation.
"""
function get_valid_modes(eq_data::Dict, sp::Subproblem, num_modes::Int)
    valid = get(eq_data, "valid_modes", nothing)
    if valid === nothing
        return ones(Bool, num_modes)
    end
    return vec(valid)
end

"""
    get_valid_modes_var(var, sp)

Get valid modes array for variable.
"""
function get_valid_modes_var(var, sp::Subproblem)
    local_size = subproblem_field_size(sp, var)
    if local_size > 0 && _has_only_zero_dim_bases(var) && !_is_zero_separable_group(sp)
        return zeros(Bool, local_size)
    end
    if hasfield(typeof(var), :valid_modes) && var.valid_modes !== nothing
        valid = vec(var.valid_modes)
        if length(valid) == local_size
            return valid
        end
    end
    return ones(Bool, local_size)
end

"""
    expression_matrices(expr, sp, vars; kwargs...)

Build expression matrices for each variable.
"""
# Fallbacks for non-Operator expression types passed to expression_matrices.
# NOTE: Do NOT define a catch-all `expression_matrices(expr, sp::Subproblem, ...)` here —
# it would create a method ambiguity with `expression_matrices(op::Operator, sp, ...)`
# in matrices.jl, causing Julia MethodError for Operator+Subproblem calls.
expression_matrices(::Nothing, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(::Number, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(expr::Future, sp::Subproblem, vars; kwargs...) =
    _expression_matrices_future(expr, sp, vars; kwargs...)
function expression_matrices(field::ScalarField, sp::Subproblem, vars; kwargs...)
    if _field_in_vars(field, vars)
        n = subproblem_field_size(sp, field)
        return Dict{Any, SparseMatrixCSC}(field => sparse(ComplexF64(1)*I, n, n))
    end

    # Check if this ScalarField is a component of a VectorField in vars
    parent_info = _find_parent_vector(field, vars)
    if parent_info !== nothing
        parent, comp_idx = parent_info
        n_parent = subproblem_field_size(sp, parent)
        n_comp = length(parent.components)
        comp_size = div(n_parent, n_comp)
        # Build selector: (comp_size × n_parent) with identity block at comp_idx
        rows = collect(1:comp_size)
        cols = collect((comp_idx-1)*comp_size+1 : comp_idx*comp_size)
        vals = ones(ComplexF64, comp_size)
        selector = sparse(rows, cols, vals, comp_size, n_parent)
        return Dict{Any, SparseMatrixCSC}(parent => selector)
    end

    return Dict{Any, SparseMatrixCSC}()
end

function expression_matrices(field::VectorField, sp::Subproblem, vars; kwargs...)
    if _field_in_vars(field, vars)
        n = subproblem_field_size(sp, field)
        return Dict{Any, SparseMatrixCSC}(field => sparse(ComplexF64(1)*I, n, n))
    end
    return Dict{Any, SparseMatrixCSC}()
end

"""
    _field_in_vars(field, vars) -> Bool

Check if `field` is in `vars` by object identity or name matching.
"""
function _field_in_vars(field, vars)
    for v in vars
        v === field && return true
        if hasfield(typeof(v), :name) && hasfield(typeof(field), :name) && v.name == field.name
            return true
        end
    end
    return false
end

"""
    _find_parent_vector(field::ScalarField, vars) -> (parent::VectorField, comp_idx::Int) or nothing

Find the parent VectorField in `vars` that contains this ScalarField as a component.
Returns (parent, component_index) or nothing if no parent found.
"""
function _find_parent_vector(field::ScalarField, vars)
    for v in vars
        if isa(v, VectorField)
            for (ci, comp) in enumerate(v.components)
                if comp === field || (hasfield(typeof(comp), :name) && comp.name == field.name)
                    return (v, ci)
                end
            end
        end
        if isa(v, TensorField)
            for (ci, comp) in enumerate(v.components)
                if comp === field || (hasfield(typeof(comp), :name) && comp.name == field.name)
                    return (v, ci)
                end
            end
        end
    end
    return nothing
end
expression_matrices(::String, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(::Symbol, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()

# Note: is_zero_expression is defined in problems.jl
# Add additional methods here for type dispatch
is_zero_expression(::Nothing) = true
is_zero_expression(x::Number) = x == 0

"""
    compute_update_rank(sp, eqns, eqn_conditions, eqn_sizes)

Compute update rank for Woodbury formula.
Following subsystems:591-595.
"""
function compute_update_rank(sp::Subproblem, eqns, eqn_conditions, eqn_sizes)
    # Group equation DOFs by dimension
    eqn_dofs_by_dim = Dict{Int, Int}()
    for (eq, cond, eq_sz) in zip(eqns, eqn_conditions, eqn_sizes)
        if cond
            dim = get(eq, "domain_dim", 0)
            eqn_dofs_by_dim[dim] = get(eqn_dofs_by_dim, dim, 0) + eq_sz
        end
    end

    if isempty(eqn_dofs_by_dim)
        return 0
    end

    max_dim = maximum(keys(eqn_dofs_by_dim))
    total_dofs = sum(values(eqn_dofs_by_dim))
    return total_dofs - get(eqn_dofs_by_dim, max_dim, 0)
end

function _column_touches_rows(A::SparseMatrixCSC, col::Int, row_mask::AbstractVector{Bool})
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for ptr in nzrange(A, col)
        row = rows[ptr]
        if row <= length(row_mask) && row_mask[row] && vals[ptr] != zero(eltype(A))
            return true
        end
    end
    return false
end

# ---------------------------------------------------------------------------
# Permutation matrices
# ---------------------------------------------------------------------------

"""
    left_permutation(sp, equations, bc_top, interleave_components)

Left permutation acting on equations.
Following subsystems:614-675.

bc_top determines if lower-dimensional equations are placed at the top or bottom.

Input ordering: Equations > Components > Modes
Output ordering with interleave_components=true: Modes > Components > Equations
Output ordering with interleave_components=false: Modes > Equations > Components
"""
function _left_permutation_indices(sp::Subproblem, equations, eqn_sizes::AbstractVector{<:Integer}, bc_top::Bool, interleave_components::Bool)
    # Compute hierarchy of input equation indices
    i = 0
    L0 = Vector{Vector{Vector{Int}}}()

    for (eq_data, eq_size) in zip(equations, eqn_sizes)
        L1 = Vector{Vector{Int}}()

        if eq_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end

        # Get shape info
        tensorsig = get(eq_data, "tensorsig", ())
        n_comps = max(1, prod((cs isa CoordinateSystem ? cs.dim : 1 for cs in tensorsig); init=1))
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

    return indices .+ 1  # 1-based indexing
end

function left_permutation(sp::Subproblem, equations, eqn_sizes::AbstractVector{<:Integer}, bc_top::Bool, interleave_components::Bool)
    indices = _left_permutation_indices(sp, equations, eqn_sizes, bc_top, interleave_components)
    return perm_matrix(indices, length(indices))
end

function left_permutation(sp::Subproblem, equations, bc_top::Bool, interleave_components::Bool)
    eqn_sizes = [_subproblem_expr_dofs(sp, get(eq, "lhs", nothing)) for eq in equations]
    return left_permutation(sp, equations, eqn_sizes, bc_top, interleave_components)
end

"""
    right_permutation(sp, variables, tau_left, interleave_components)

Right permutation acting on variables.
Following subsystems:678-739.

tau_left determines if lower-dimensional variables are placed at the left or right.

Input ordering: Variables > Components > Modes
Output ordering with interleave_components=true: Modes > Components > Variables
Output ordering with interleave_components=false: Modes > Variables > Components
"""
function _right_permutation_indices(sp::Subproblem, variables, tau_left::Bool, interleave_components::Bool)
    # Compute hierarchy of input variable indices
    i = 0
    L0 = Vector{Vector{Vector{Int}}}()

    for var in variables
        L1 = Vector{Vector{Int}}()
        var_size = subproblem_field_size(sp, var)

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

    return indices .+ 1  # 1-based indexing
end

function right_permutation(sp::Subproblem, variables, tau_left::Bool, interleave_components::Bool)
    indices = _right_permutation_indices(sp, variables, tau_left, interleave_components)
    return perm_matrix(indices, length(indices))
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
Following tools/array perm_matrix.
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
Following tools/array drop_empty_rows.
"""
function drop_empty_rows(A::SparseMatrixCSC)
    m, n = size(A)
    if nnz(A) == 0
        return spzeros(eltype(A), 0, n)
    end
    # Use sparse structure directly instead of dense sum(abs.(A), dims=2)
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    has_nonzero = falses(m)
    @inbounds for i in eachindex(rows)
        if vals[i] != zero(eltype(A))
            has_nonzero[rows[i]] = true
        end
    end
    non_empty = findall(has_nonzero)
    if isempty(non_empty)
        return spzeros(eltype(A), 0, n)
    end
    if length(non_empty) == m
        return A  # All rows non-empty, no slicing needed
    end
    return A[non_empty, :]
end

"""
    zeros_with_pattern(matrices...)

Create zero matrix with combined sparsity pattern.
Following tools/array zeros_with_pattern.

Note: Julia's sparse format doesn't store explicit zeros, so we use
a tiny value (eps) to preserve the structural pattern. This is needed
for IMEX recombination where the sparsity pattern must be preserved.
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

    # Use tiny non-zero values to preserve structural pattern
    # (Julia's sparse format optimizes away explicit zeros)
    # The sparse() call also handles duplicate indices by summing values
    placeholder_vals = fill(eps(Float64) * (1.0 + 0.0im), length(rows))
    return sparse(rows, cols, placeholder_vals, m, n)
end

"""
    expand_pattern(A, B)

Expand A to have same sparsity pattern as B.
Following tools/array expand_pattern.
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
    apply_sparse(A, x; axis=1, out=nothing)

Apply sparse matrix along specified axis.
Following tools/array apply_sparse.

For axis=1: Applies A to x normally (A * x)
For axis=2: Applies A along the second axis of x (for 2D arrays: (A * x')')
            For 1D vectors, axis=2 is treated same as axis=1.
"""
function apply_sparse(A::SparseMatrixCSC, x::AbstractArray; axis::Int=1, out::Union{Nothing, AbstractArray}=nothing)
    if axis == 1
        result = A * x
    elseif axis == 2
        if ndims(x) == 1
            # For 1D vectors, axis=2 is equivalent to axis=1
            result = A * x
        else
            # For 2D+ arrays, multiply along second axis
            # This applies A to each row of x
            result = (A * x')'
        end
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

Build the NCC (non-constant coefficient) multiplication matrix for a single
subproblem. An NCC is a spatially-varying coefficient appearing as a
multiplicative factor in an equation, e.g. `ν(z) * ∇²u` where `ν` varies
with the coupled coordinate.

The matrix is built by summing mode-shift matrices (one per significant
spectral coefficient of the NCC field), weighted by that coefficient's
value:

    M = Σ_k ν_k · S_k

where `ν_k` is the k-th spectral coefficient of the NCC field and `S_k` is
the shift matrix that implements spectral-space convolution by mode k.

## Two-stage truncation

To keep the resulting sparse matrix as small as possible (which directly
reduces sparse LU factorization cost), we apply two levels of truncation:

1. **Per-mode cutoff** (`ncc_cutoff`): skip contributions from NCC modes
   whose absolute coefficient is below `ncc_cutoff`. Dedalus's default is
   `1e-6`; tighter cutoffs give sparser matrices at some accuracy cost.
2. **Max-terms cap** (`max_ncc_terms`): after sorting modes by coefficient
   magnitude, retain at most `max_ncc_terms` dominant modes. Useful for
   rapidly-decaying NCCs where a small number of modes capture most of the
   information.

After accumulation, the matrix is passed through `droptol!` to remove any
entries whose absolute value is below `ncc_cutoff` — this catches residual
entries that appear from cross-mode accumulation (e.g. two large opposite
contributions that partially cancel) and shrinks the nnz count for the
downstream sparse LU.

Follows the spectral methods pattern from the Dedalus arithmetic module.
"""
function build_ncc_matrix(ncc_data::NCCData, sp::Subproblem, arg_domain, out_domain;
                          ncc_cutoff::Float64=1e-6, max_ncc_terms::Union{Nothing, Int}=nothing)
    if ncc_data.coeffs === nothing
        return nothing
    end

    coeffs = ncc_data.coeffs
    shape = (coeff_size(sp, out_domain), coeff_size(sp, arg_domain))
    matrix = spzeros(ComplexF64, shape...)

    sp_shape = coeff_shape(sp, out_domain)
    ncc_shape = size(coeffs)

    # ─── Relative cutoff ─────────────────────────────────────────────
    # Dedalus convention: interpret `ncc_cutoff` as a RELATIVE threshold
    # against the L-infinity norm of the coefficient array, not as an
    # absolute magnitude. This scales automatically with the problem's
    # natural coefficient magnitude — a viscosity `ν ~ 1e-3` and a
    # temperature `T ~ 1.0` both get truncated at the same relative
    # precision without the user having to tune `ncc_cutoff` per field.
    #
    # Matches `dedalus.core.arithmetic.Multiply._ncc_matrices` where
    # `cutoff` is divided by the coefficient L∞ norm before comparison.
    coeff_max = 0.0
    @inbounds for i in eachindex(coeffs)
        ai = abs(coeffs[i])
        ai > coeff_max && (coeff_max = ai)
    end
    # Absolute threshold: `ncc_cutoff × max|coefficient|`. For
    # coeff_max == 0 (all-zero NCC), treat as zero — build_ncc_matrix
    # returns an empty sparse matrix.
    abs_cutoff = ncc_cutoff * coeff_max

    # ─── Energy-weighted mode selection ──────────────────────────────
    # Collect modes above the relative cutoff, along with their "energy"
    # |coeff|² which is what determines the truncation error in an
    # L²-sense approximation. Sort by energy descending so the
    # `max_ncc_terms` cap keeps the modes that carry the most spectral
    # power, not just the modes that are pointwise-largest.
    #
    # For smoothly-decaying coefficient fields (e.g. exponentially
    # decaying Chebyshev coefficients) energy-sorted and magnitude-sorted
    # orderings agree. For oscillatory fields or fields with fat tails,
    # energy sorting gives a better approximation at the same
    # `max_ncc_terms` budget.
    significant_modes = Tuple{Float64, CartesianIndex}[]
    @inbounds for ncc_mode in CartesianIndices(ncc_shape)
        mag = abs(coeffs[ncc_mode])
        if mag >= abs_cutoff && mag > 0
            push!(significant_modes, (mag * mag, ncc_mode))  # energy weight
        end
    end
    sort!(significant_modes; by = first, rev = true)

    # Optional energy-retention cap: if max_ncc_terms is nothing, we can
    # additionally cap by CUMULATIVE energy fraction — keep enough modes
    # to capture 1 - ncc_cutoff² of the total power. This is another
    # Dedalus-style truncation that's independent of the count cap.
    total_energy = 0.0
    @inbounds for k in 1:length(significant_modes)
        total_energy += significant_modes[k][1]
    end

    n_to_use = length(significant_modes)
    if max_ncc_terms !== nothing
        n_to_use = min(n_to_use, max_ncc_terms)
    else
        # Cap by cumulative energy: retain enough modes to capture
        # (1 - ncc_cutoff²) of the total L² energy. For ncc_cutoff = 1e-6
        # this keeps essentially all significant modes; for ncc_cutoff =
        # 0.01 it drops trailing modes that contribute < 1e-4 of energy.
        target_energy = (1.0 - ncc_cutoff * ncc_cutoff) * total_energy
        cumulative = 0.0
        cap = 0
        @inbounds for k in 1:length(significant_modes)
            cumulative += significant_modes[k][1]
            cap = k
            cumulative >= target_energy && break
        end
        n_to_use = max(cap, 1)
    end

    # ─── Accumulate mode-shift contributions ─────────────────────────
    @inbounds for k in 1:n_to_use
        ncc_mode = significant_modes[k][2]
        ncc_coeff = coeffs[ncc_mode]
        mode_mat = cartesian_mode_matrix(sp_shape, arg_domain, out_domain, Tuple(ncc_mode))
        # In-place accumulation into `matrix.nzval` would avoid the
        # temporary sparse matrix on the RHS, but sparse-structure merging
        # is awkward when new mode matrices introduce additional nonzero
        # patterns. The current add-and-reassign is O(nnz) per iteration
        # and dominated by the subsequent LU, so it's not the bottleneck.
        matrix = matrix + ncc_coeff * mode_mat
    end

    # ─── Post-accumulation sparsification ────────────────────────────
    # After summing all contributions, drop residual small entries. Use
    # the absolute cutoff (scaled by max coefficient) to match the
    # per-mode threshold, and then dropzeros! to clean up structural
    # cancellations.
    if abs_cutoff > 0 && nnz(matrix) > 0
        droptol!(matrix, abs_cutoff)
    end
    if nnz(matrix) > 0
        dropzeros!(matrix)
    end

    return matrix
end

"""
    cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode)

Build mode matrix for Cartesian NCC.
Following arithmetic:446-460.
"""
function cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode::Tuple)
    # Build Kronecker product of 1D mode matrices for NCC convolution
    # This implements the mode matrix for Non-Constant Coefficient multiplication
    # in spectral space, which corresponds to convolution in coefficient space
    matrix = sparse([1.0])

    for axis in eachindex(sp_shape)
        n = sp_shape[axis]
        mode = axis <= length(ncc_mode) ? ncc_mode[axis] : 0

        # Product matrix for this axis
        # mode == 0: identity (no shift)
        # mode != 0: circulant shift matrix for periodic bases
        if mode == 0 || mode == 1
            # Identity matrix - no mode shift
            axis_mat = sparse(I, n, n)
        else
            # Build circulant shift matrix for mode k
            # This shifts coefficients by the NCC mode index
            # For periodic bases: c_j -> c_{j-k} (circular)
            rows = Int[]
            cols = Int[]
            vals = Float64[]

            for j in 1:n
                # Target index after shift (circular)
                target = mod1(j + mode - 1, n)
                push!(rows, target)
                push!(cols, j)
                push!(vals, 1.0)
            end

            axis_mat = sparse(rows, cols, vals, n, n)
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

# ============================================================================
# Exports
# ============================================================================

# Export configuration
export SolverConfig, SUBSYSTEM_GROUP

# Export main types
export Subsystem, Subproblem, NCCData

# Export subsystem construction
export build_subsystems, build_subproblems
export scalar_components, scalar_field_dofs
export infer_problem_dtype, infer_problem_dist
export compute_variable_ranges, compute_equation_ranges, compute_matrix_group

# Export subsystem methods
export coeff_slices, coeff_shape, coeff_size
export field_slices, field_shape, field_size, field_domain, subproblem_field_size
export gather, scatter

# Export subproblem methods
export check_condition, valid_modes
export gather_inputs, scatter_inputs, gather_eqn_F!, gather_alg_F!, apply_bc_override!

# Export matrix building
export build_matrices!, build_subproblem_matrices
export expression_matrices, gather_ncc_coeffs!
export get_valid_modes, get_valid_modes_var
export compute_update_rank

# Export permutation functions
export left_permutation, right_permutation, perm_matrix
export get_var_dim

# Export matrix utilities
export drop_empty_rows, zeros_with_pattern, expand_pattern, apply_sparse

# Export NCC functions
export build_ncc_matrix, cartesian_mode_matrix

# Export coupling analysis
export get_matrix_coupling, analyze_operator_coupling
export get_separable_dim_size, generate_mode_groups
