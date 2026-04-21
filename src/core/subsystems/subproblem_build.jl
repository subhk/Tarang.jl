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

