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
