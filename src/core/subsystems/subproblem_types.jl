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
