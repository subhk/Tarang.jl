"""
    Problem matrix assembly

This file contains the top-level `build_matrices(problem)` path and the
construction of equation-level matrix expressions from parsed equations.
"""
function build_matrices(problem::Problem)
    
    if length(problem.equations) == 0
        throw(ArgumentError("No equations specified"))
    end
    
    # Build matrix expressions from equations (following problems:_build_matrix_expressions)
    build_matrix_expressions!(problem)

    # Compute field sizes in COEFFICIENT SPACE (the timestepper operates there).
    # Must NOT call ensure_layout! — that would clobber user data set before
    # the solver is created.  Instead compute sizes from basis metadata.
    eqn_sizes = [compute_field_size(eq_data) for eq_data in problem.equation_data]
    var_sizes = [_coeff_space_dofs(var) for var in problem.variables]

    total_rows = sum(eqn_sizes)  # Total rows
    total_cols = sum(var_sizes)  # Total columns

    if total_rows != total_cols
        @warn "Matrix is not square: rows=$total_rows, cols=$total_cols" *
              "\n  eqn_sizes=$eqn_sizes\n  var_sizes=$var_sizes" maxlog=1
    end
    @debug "Building matrices: equations=$total_rows, variables=$total_cols"

    # Matrix names to build (following convention)
    matrix_names = ["M", "L"]  # M = mass matrix, L = stiffness matrix

    # Build sparse matrices following subsystems:513-537 pattern
    matrices = Dict{String, Any}()
    for name in matrix_names
        # Collect sparse matrix entries (ComplexF64 for spectral methods)
        data, rows, cols = ComplexF64[], Int[], Int[]
        
        i0 = 0  # Row offset
        for (eq_idx, eq_data) in enumerate(problem.equation_data)
            eqn_size = eqn_sizes[eq_idx]
            if eqn_size > 0 && check_equation_condition(eq_data)
                # Get expression matrix blocks for this equation
                expr = get_matrix_expression(eq_data, name)
                if expr !== nothing && !is_zero_expression(expr)
                    # Build expression matrices for each variable
                    j0 = 0  # Column offset
                    for (var_idx, var) in enumerate(problem.variables)
                        var_size = var_sizes[var_idx]
                        if var_size > 0
                            # Get matrix block for this variable
                            block = build_expression_matrix_block(expr, var, eqn_size, var_size)
                            if !isempty(block.nzval)
                                # Add to sparse matrix data
                                # SparseMatrixCSC stores: rowval (row indices), colptr (column pointers), nzval (values)
                                # We need to expand colptr to get column indices for each non-zero
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
        
        # Create sparse matrix
        if !isempty(data)
            # Filter small entries (following entry_cutoff pattern)
            entry_cutoff = 1e-14
            significant = abs.(data) .>= entry_cutoff
            data = data[significant]
            rows = rows[significant]
            cols = cols[significant]
            
            matrices[name] = sparse(rows, cols, data, total_rows, total_cols)
        else
            # Empty matrix
            matrices[name] = spzeros(ComplexF64, total_rows, total_cols)
        end

        @debug "Built matrix $name: size=($total_rows, $total_cols), nnz=$(nnz(matrices[name]))"
    end
    
    # Build forcing vector (RHS terms)
    F_vector = build_forcing_vector(problem, eqn_sizes, total_rows)
    
    # Return matrices in standard format
    L_matrix = matrices["L"]
    M_matrix = matrices["M"] 
    
    # Only log on rank 0 to avoid repeated messages
    if length(problem.variables) > 0 && problem.variables[1].dist.rank == 0
        @info "Matrix building completed: L=$(size(L_matrix)), M=$(size(M_matrix)), F=$(length(F_vector))"
    end

    return L_matrix, M_matrix, F_vector
end

"""
    Build matrix expressions from parsed equations.
"""
function build_matrix_expressions!(problem::Problem)
    
    empty!(problem.equation_data)
    
    for (i, equation_str) in enumerate(problem.equations)
        # Parse both sides into operator trees. Nonlinear RHS terms do not need to
        # be representable as matrices, but they still must parse successfully so
        # the explicit runtime evaluator receives the original expression.
        lhs_str, rhs_str = split_equation(equation_str)
        lhs = nothing
        rhs = nothing
        try
            lhs = parse_expression(strip(lhs_str), problem.namespace)
        catch e
            @error "Failed to parse LHS of equation $i: $equation_str" exception=e
        end
        try
            rhs = parse_expression(strip(rhs_str), problem.namespace)
        catch e
            throw(ArgumentError(
                "Failed to parse RHS of equation $i (`$(strip(rhs_str))`): " *
                sprint(showerror, e),
            ))
        end
        if lhs === nothing
            lhs = UnknownOperator(equation_str)
        end

        try
            if problem isa InitialValueProblem
                _validate_ivp_equation_format(lhs, rhs, problem.variables)
            end
            eq_data = EquationIR(build_equation_expressions(lhs, rhs, problem.variables))
            eq_data["equation_index"] = i
            eq_data["equation_string"] = equation_str
            eq_size = _equation_output_dofs(lhs)
            if eq_size == 0
                @warn "equation_output_dofs=0 for eq $i: $(equation_str)" maxlog=1
            end
            eq_data["equation_size"] = eq_size
            push!(problem.equation_data, eq_data)
        catch e
            throw(ArgumentError(
                "Failed to build matrix expressions for equation $i " *
                "(`$equation_str`): " * sprint(showerror, e),
            ))
        end
        _check_duplicate_tau_lifts(problem.equation_data[end], equation_str)
    end
end

# Collect the `Lift` terms that enter an expression ADDITIVELY at top level —
# through +, −, negation and scalar multiplication only. A lift nested inside a
# derivative operator (`div(grad(b) + ez*lift(tau1, b, -1))`) lands on different
# rows than a top-level `lift(tau2, b, -1)` and is the canonical well-posed
# channel formulation, so the walk deliberately stops at any other operator.
function _collect_top_level_lifts!(acc::Vector{Any}, expr)
    if expr isa Lift
        push!(acc, expr)
    elseif expr isa AddOperator || expr isa SubtractOperator
        _collect_top_level_lifts!(acc, expr.left)
        _collect_top_level_lifts!(acc, expr.right)
    elseif expr isa NegateOperator
        _collect_top_level_lifts!(acc, expr.operand)
    elseif expr isa MultiplyOperator
        # scalar · lift keeps the lift at top level; field · lift does not.
        if expr.left isa Number || expr.left isa ConstantOperator
            _collect_top_level_lifts!(acc, expr.right)
        elseif expr.right isa Number || expr.right isa ConstantOperator
            _collect_top_level_lifts!(acc, expr.left)
        end
    end
    return acc
end

"""
    _check_duplicate_tau_lifts(eq_data, equation_str)

Refuse an equation that lifts two DIFFERENT tau variables onto the same mode of
the same basis at top level, e.g. `lift(tau1, b, -1) + lift(tau2, b, -1)`.

Those two columns of the per-mode matrix are identical, so every stage system is
singular. The stepper's sparse-QR least-squares fallback would then "solve" it,
and with a moving boundary condition that recurrence is exponentially unstable
(2026-09-05 audit: max|u| 1 → 5e8 within 300 steps while the boundary values
stayed exact, so a short boundary check could not see it). One tau per mode —
`-1` and `-2` — is the well-posed form. Checked here, at parse time, because at
factorization time it is indistinguishable from the legitimately singular
pressure-gauge mode of an incompressible problem.
"""
function _check_duplicate_tau_lifts(eq_data, equation_str::AbstractString)
    lifts = Any[]
    for slot in ("L", "M")
        expr = get(eq_data, slot, nothing)
        expr === nothing && continue
        _collect_top_level_lifts!(lifts, expr)
    end
    length(lifts) < 2 && return nothing
    for i in eachindex(lifts), j in (i + 1):lastindex(lifts)
        a, b = lifts[i], lifts[j]
        (_lift_basis_signature(a.basis) == _lift_basis_signature(b.basis) &&
         a.n == b.n && a.operand !== b.operand) || continue
        throw(ArgumentError(
            "Equation `$equation_str` lifts two different tau variables onto the SAME mode " *
            "($(a.n)) of the same basis: `lift($(_lift_operand_name(a)), ..., $(a.n))` and " *
            "`lift($(_lift_operand_name(b)), ..., $(b.n))`. Those matrix columns are identical, " *
            "so every per-mode stage system is singular; the least-squares treatment that would " *
            "result is exponentially unstable under a moving boundary condition. Give each tau " *
            "its own mode, e.g. `lift(tau1, basis, -1) + lift(tau2, basis, -2)`."))
    end
    return nothing
end

"""Structural identity of a lift's output basis.

`derivative_basis` constructs a FRESH basis object on every call, so the common
`lift(tau1, derivative_basis(zb, 1), -1) + lift(tau2, derivative_basis(zb, 1), -1)`
spelling holds two distinct objects that describe one and the same basis. An
`===` test waves that singular system straight through to the least-squares
fallback, which is exactly what the check above exists to refuse.
"""
function _lift_basis_signature(basis::Basis)
    meta = basis.meta
    # Jacobi-family bases differ by their (a, b) parameters at equal size/bounds.
    jacobi = (hasproperty(basis, :a) ? Float64(getproperty(basis, :a)) : nothing,
              hasproperty(basis, :b) ? Float64(getproperty(basis, :b)) : nothing)
    return (nameof(typeof(basis)), String(meta.element_label), meta.size,
            meta.bounds, jacobi)
end

_lift_operand_name(l::Lift) = hasproperty(l.operand, :name) ? String(l.operand.name) : repr(l.operand)

"""
    Build matrix expressions from LHS and RHS operators.
"""
function build_equation_expressions(lhs, rhs, variables::Vector)
    
    eq_data = Dict{String, Any}()
    
    # Split LHS into mass matrix (time derivatives) and stiffness matrix (spatial) terms
    # Following InitialValueProblem pattern: M.dt(X) + L.X = F (problems:328)
    M_terms, L_terms = split_time_spatial_operators(lhs)
    
    # Store matrix expressions
    eq_data["M"] = combine_operators(M_terms)      # Mass matrix terms
    eq_data["L"] = combine_operators(L_terms)      # Stiffness matrix terms  
    eq_data["F"] = rhs                             # Forcing terms

    # Determine which variables participate in this equation
    eq_vars = _detect_equation_variables(lhs, variables)
    if isempty(eq_vars)
        # Some constraint equations (e.g., BCs) only reference variables on RHS
        eq_vars = _detect_equation_variables(rhs, variables)
    end
    if isempty(eq_vars)
        # Fall back to all variables to keep matrix sizes consistent
        eq_vars = copy(variables)
    end

    eq_data["equation_variables"] = eq_vars
    # NOTE: equation_size is set by the caller (build_matrix_expressions!)
    # based on the 1:1 equation-variable mapping, not here.
    
    # Metadata
    eq_data["variables"] = variables
    eq_data["lhs"] = lhs
    eq_data["rhs"] = rhs
    
    return eq_data
end
