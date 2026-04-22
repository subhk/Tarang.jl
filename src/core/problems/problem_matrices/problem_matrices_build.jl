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
    
    problem.equation_data = Dict{String, Any}[]
    
    for (i, equation_str) in enumerate(problem.equations)
        # Parse LHS first (for sizing) — it must succeed.
        # RHS parsing may fail for nonlinear terms (e.g. u⋅∇(b)) that aren't
        # representable in the matrix; that's fine — the explicit RHS evaluator
        # handles them at runtime.
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
            @debug "RHS parse failed for equation $i (will use runtime evaluation): $e"
            rhs = ZeroOperator()
        end
        if lhs === nothing
            lhs = UnknownOperator(equation_str)
        end

        try
            eq_data = build_equation_expressions(lhs, rhs, problem.variables)
            eq_data["equation_index"] = i
            eq_data["equation_string"] = equation_str
            eq_size = _equation_output_dofs(lhs)
            if eq_size == 0
                @warn "equation_output_dofs=0 for eq $i: $(equation_str)" maxlog=1
            end
            eq_data["equation_size"] = eq_size
            push!(problem.equation_data, eq_data)
        catch e
            @error "Failed to build matrix expressions for equation $i: $equation_str" exception=e
            eq_size = lhs !== nothing ? _equation_output_dofs(lhs) : 0
            fallback_data = Dict(
                "M" => nothing,
                "L" => lhs isa UnknownOperator ? lhs : UnknownOperator(equation_str),
                "F" => ZeroOperator(),
                "equation_index" => i,
                "equation_string" => equation_str,
                "equation_size" => eq_size
            )
            push!(problem.equation_data, fallback_data)
        end
    end
end

"""
    Build matrix expressions from LHS and RHS operators.
"""
function build_equation_expressions(lhs, rhs, variables::Vector)
    
    eq_data = Dict{String, Any}()
    
    # Split LHS into mass matrix (time derivatives) and stiffness matrix (spatial) terms
    # Following IVP pattern: M.dt(X) + L.X = F (problems:328)
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
