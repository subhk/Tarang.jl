"""
    Build system matrices for problem following structure.
    Following subsystems:build_subproblem_matrices (subsystems:72-81) and
    Subproblem.build_matrices (subsystems:497-576).
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
    Following problems:_build_matrix_expressions patterns.
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
    Following _build_matrix_expressions patterns.
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

function _detect_equation_variables(expr, variables::Vector{<:Operand})
    found = Operand[]
    _collect_equation_variables!(found, expr, variables)
    return found
end

function _collect_equation_variables!(found::Vector{Operand}, expr, variables::Vector{<:Operand})
    expr === nothing && return

    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        for var in variables
            if _operand_matches_variable(expr, var)
                _maybe_add_variable!(found, var)
                break
            end
        end
    end

    if hasfield(typeof(expr), :left)
        _collect_equation_variables!(found, getfield(expr, :left), variables)
    end
    if hasfield(typeof(expr), :right)
        _collect_equation_variables!(found, getfield(expr, :right), variables)
    end
    if hasfield(typeof(expr), :operand)
        _collect_equation_variables!(found, getfield(expr, :operand), variables)
    end
    if hasfield(typeof(expr), :operands)
        ops = getfield(expr, :operands)
        if ops !== nothing
            for op in ops
                _collect_equation_variables!(found, op, variables)
            end
        end
    end
    if hasfield(typeof(expr), :array)
        _collect_equation_variables!(found, getfield(expr, :array), variables)
    end
    if hasfield(typeof(expr), :indices)
        idxs = getfield(expr, :indices)
        if idxs !== nothing
            for idx in idxs
                _collect_equation_variables!(found, idx, variables)
            end
        end
    end
    if hasfield(typeof(expr), :base)
        _collect_equation_variables!(found, getfield(expr, :base), variables)
    end
    if hasfield(typeof(expr), :exponent)
        _collect_equation_variables!(found, getfield(expr, :exponent), variables)
    end
end

function _maybe_add_variable!(found::Vector{Operand}, var::Operand)
    for existing in found
        if existing === var
            return
        end
        existing_name = _operand_name(existing)
        new_name = _operand_name(var)
        if existing_name !== nothing && existing_name == new_name
            return
        end
    end
    push!(found, var)
end

@inline function _operand_matches_variable(expr, var::Operand)
    (expr === var) && return true
    expr_name = _operand_name(expr)
    var_name = _operand_name(var)
    return expr_name !== nothing && expr_name == var_name
end

@inline function _operand_name(var)
    return hasfield(typeof(var), :name) ? getfield(var, :name) : nothing
end

"""
    _equation_output_dofs(expr)

Compute the number of DOFs (rows) an equation's LHS expression produces,
by propagating output shapes through the operator tree — the same approach
uses.  This makes equation ordering irrelevant; the matrix builder
determines each equation's row count from its mathematical structure.

Returns 0 for unknown/constant expressions (overridden by `max` in Add/Sub).
"""
function _equation_output_dofs(expr)
    # ── Direct field types ──────────────────────────────────────
    isa(expr, ScalarField)  && return _coeff_space_dofs(expr)
    isa(expr, VectorField)  && return _coeff_space_dofs(expr)
    isa(expr, TensorField)  && return _coeff_space_dofs(expr)

    # ── Arithmetic (both Operator and Future types) ──────────────
    # All addends of a well-formed equation share the same shape,
    # so `max` skips zeros/constants and picks the informative term.
    #
    # Handles both hierarchies:
    #   Operator: AddOperator, SubtractOperator, MultiplyOperator, NegateOperator
    #   Future:   Add, Multiply, Subtract, Negate
    if isa(expr, AddOperator) || isa(expr, SubtractOperator)
        return max(_equation_output_dofs(expr.left),
                   _equation_output_dofs(expr.right))
    end
    if isa(expr, MultiplyOperator) || isa(expr, DivideOperator)
        return max(_equation_output_dofs(expr.left),
                   _equation_output_dofs(expr.right))
    end
    isa(expr, NegateOperator) && return _equation_output_dofs(expr.operand)

    # Future arithmetic types (from Julia-level expression building)
    if isa(expr, Future)
        args = future_args(expr)
        if !isempty(args)
            return maximum(_equation_output_dofs(a) for a in args)
        end
        return 0
    end

    # ── Shape-preserving operators ──────────────────────────────
    if isa(expr, TimeDerivative) || isa(expr, Laplacian) ||
       isa(expr, FractionalLaplacian) || isa(expr, Differentiate) ||
       isa(expr, Lift) || isa(expr, Skew) || isa(expr, Curl) ||
       isa(expr, Trace) || isa(expr, Convert) || isa(expr, HilbertTransform) ||
       isa(expr, Component) || isa(expr, RadialComponent) ||
       isa(expr, AngularComponent) || isa(expr, AzimuthalComponent)
        return _equation_output_dofs(expr.operand)
    end

    # ── Rank-changing operators ─────────────────────────────────
    if isa(expr, Gradient)
        # scalar(D) → vector(ndim*D)
        inner = _equation_output_dofs(expr.operand)
        ndim  = hasfield(typeof(expr), :coordsys) && expr.coordsys !== nothing ?
                length(expr.coordsys.coords) : 2
        return inner * ndim
    end
    if isa(expr, Divergence)
        # vector(ndim*D) → scalar(D).  Recover D = total / ndim.
        inner = _equation_output_dofs(expr.operand)
        ndim  = _infer_ndim(expr.operand)
        return ndim > 0 ? inner ÷ ndim : inner
    end

    # ── Reducing operators ──────────────────────────────────────
    if isa(expr, Integrate) || isa(expr, Average)
        return 1  # Reduces all spatial dims → scalar
    end
    if isa(expr, Interpolate)
        # Interpolate removes one spatial dim.
        # Output DOFs = operand DOFs / interpolated-dim coeff size.
        inner = _equation_output_dofs(expr.operand)
        cdim = _coord_coeff_size(expr.coord, expr.operand)
        return cdim > 0 ? inner ÷ cdim : max(inner, 1)
    end

    # ── Constants / zeros ───────────────────────────────────────
    (isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator)) && return 0

    # ── Fallback: find any field anywhere in the tree ───────────
    # This prevents unknown operator types from silently zeroing
    # the equation size and making the matrix non-square.
    dofs = _find_any_field_dofs(expr)
    if dofs > 0
        @debug "equation_output_dofs fallback for $(typeof(expr)): using field DOFs=$dofs"
        return dofs
    end
    return 0
end

"""Find coeff-space size of the basis corresponding to a coordinate."""
function _coord_coeff_size(coord::Coordinate, operand)
    field = _find_any_scalar_field(operand)
    field === nothing && return 0
    coord_name = hasfield(typeof(coord), :name) ? string(coord.name) : nothing

    # Match basis by element_label (stored in basis.meta.element_label)
    first_rf = true
    for basis in field.bases
        basis === nothing && continue
        label = hasfield(typeof(basis), :meta) ? string(basis.meta.element_label) : nothing
        if label !== nothing && coord_name !== nothing && label == coord_name
            # First RealFourier dimension is halved by rfft; others keep full size
            if isa(basis, RealFourier) && first_rf
                return div(basis.meta.size, 2) + 1
            else
                return basis.meta.size
            end
        end
        if isa(basis, RealFourier)
            first_rf = false
        end
    end
    return 0
end

"""Find any ScalarField in an expression tree (handles Operator and Future types)."""
function _find_any_scalar_field(expr)
    isa(expr, ScalarField) && return expr
    isa(expr, VectorField) && return expr.components[1]
    isa(expr, TensorField) && return expr.components[1,1]
    # Operator fields: :operand, :left, :right
    for f in (:operand, :left, :right)
        hasfield(typeof(expr), f) || continue
        result = _find_any_scalar_field(getfield(expr, f))
        result !== nothing && return result
    end
    # Future types: search args
    if isa(expr, Future)
        for arg in future_args(expr)
            result = _find_any_scalar_field(arg)
            result !== nothing && return result
        end
    end
    return nothing
end

"""Find any field DOFs in an expression tree (handles Operator and Future types)."""
function _find_any_field_dofs(expr)
    isa(expr, ScalarField)  && return _coeff_space_dofs(expr)
    isa(expr, VectorField)  && return _coeff_space_dofs(expr)
    isa(expr, TensorField)  && return _coeff_space_dofs(expr)
    for f in (:operand, :left, :right)
        hasfield(typeof(expr), f) || continue
        child = getfield(expr, f)
        child === nothing && continue
        dofs = _find_any_field_dofs(child)
        dofs > 0 && return dofs
    end
    if isa(expr, Future)
        for arg in future_args(expr)
            dofs = _find_any_field_dofs(arg)
            dofs > 0 && return dofs
        end
    end
    return 0
end

"""Infer spatial dimension count from an expression (for Divergence)."""
function _infer_ndim(expr)
    isa(expr, VectorField) && return length(expr.components)
    isa(expr, Gradient) && hasfield(typeof(expr), :coordsys) &&
        expr.coordsys !== nothing && return length(expr.coordsys.coords)
    # Recurse through Operator wrappers
    hasfield(typeof(expr), :operand) && return _infer_ndim(expr.operand)
    # Future types: search args for any that reveal ndim
    if isa(expr, Future)
        for arg in future_args(expr)
            n = _infer_ndim(arg)
            n > 0 && return n
        end
    end
    return 0
end

"""
    Split operator into time derivative (mass matrix) and spatial (stiffness) terms.
    Following operators split pattern.
    """
function split_time_spatial_operators(operator)
    
    M_terms = []  # Time derivative terms
    L_terms = []  # Spatial terms
    empty_namespace = Dict{String, Any}()
    
    if isa(operator, TimeDerivative)
        # Pure time derivative
        push!(M_terms, operator)
        
    elseif isa(operator, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Pure spatial operator
        push!(L_terms, operator)
        
    elseif isa(operator, AddOperator)
        # Split addition terms recursively
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)
        append!(M_terms, left_M)
        append!(L_terms, left_L)
        append!(M_terms, right_M)
        append!(L_terms, right_L)
        
    elseif isa(operator, SubtractOperator)
        # Split subtraction terms recursively
        # For A - B, we split both and negate the right side terms
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)

        # Add left terms directly
        append!(M_terms, left_M)
        append!(L_terms, left_L)

        # Right terms need negation - wrap in NegateOperator or multiply by -1
        for term in right_M
            push!(M_terms, NegateOperator(term))
        end
        for term in right_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, NegateOperator)
        inner_M, inner_L = split_time_spatial_operators(operator.operand)
        for term in inner_M
            push!(M_terms, NegateOperator(term))
        end
        for term in inner_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, MultiplyOperator)
        coeff = nothing
        inner = nothing

        if _is_constant_coefficient_strict(operator.left, empty_namespace) &&
           !_is_constant_coefficient_strict(operator.right, empty_namespace)
            coeff = operator.left
            inner = operator.right
        elseif _is_constant_coefficient_strict(operator.right, empty_namespace) &&
               !_is_constant_coefficient_strict(operator.left, empty_namespace)
            coeff = operator.right
            inner = operator.left
        end

        if inner !== nothing
            scaled = MultiplyOperator(coeff, inner)
            if isa(inner, TimeDerivative)
                push!(M_terms, scaled)
            elseif isa(inner, Union{Laplacian, Gradient, Divergence, Differentiate}) || hasfield(typeof(inner), :name)
                push!(L_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end

    elseif isa(operator, DivideOperator)
        if _is_constant_coefficient_strict(operator.right, empty_namespace)
            scaled = DivideOperator(operator.left, operator.right)
            if isa(operator.left, TimeDerivative)
                push!(M_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end
        
    elseif hasfield(typeof(operator), :name)
        # Direct variable reference -> identity in L
        push!(L_terms, operator)
        
    else
        # Other operators go to L by default
        push!(L_terms, operator)
    end
    
    return M_terms, L_terms
end

"""Combine operator terms into single expression"""
function combine_operators(terms::Vector)
    if isempty(terms)
        return ZeroOperator()
    elseif length(terms) == 1
        return terms[1]
    else
        # Combine with addition
        result = terms[1]
        for i in 2:length(terms)
            result = AddOperator(result, terms[i])
        end
        return result
    end
end

# Supporting functions for matrix building

function field_dofs(field::ScalarField)
    # Tau variables (empty bases): always 1 DOF, consistent with
    # _coeff_space_dofs, compute_field_vector_size, extract_field_data_for_vector
    if isempty(field.bases)
        return 1
    end
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

field_dofs(field::VectorField) = sum(field_dofs(comp) for comp in field.components)
field_dofs(field::TensorField) = sum(field_dofs(comp) for comp in vec(field.components))

"""
    _coeff_space_dofs(var)

Compute GLOBAL coefficient-space DOF count from basis metadata, WITHOUT
transforming the field or reading field data (which may be a local MPI slice).

Julia's `rfft` only halves the FIRST RealFourier dimension.  Subsequent
RealFourier axes use regular FFT (full size N).  This matches
`coefficient_shape_mpi` logic and is correct for both serial and MPI modes.
"""
function _coeff_space_dofs(var::ScalarField)
    # Compute from basis metadata — always global, works for serial + MPI
    if !isempty(var.bases)
        total = 1
        first_rf = true
        for basis in var.bases
            if basis !== nothing
                if isa(basis, RealFourier) && first_rf
                    total *= div(basis.meta.size, 2) + 1
                    first_rf = false
                else
                    total *= basis.meta.size
                end
            end
        end
        return total
    end

    # Tau variables (empty bases): always 1 DOF (0D scalar gauge variable).
    # Don't check data — its allocation state varies between matrix-build
    # and stepping, causing off-by-1 mismatches. The fallback in
    # compute_field_vector_size also returns 1 for empty-bases fields.
    return 1
end
_coeff_space_dofs(var::VectorField) = sum(_coeff_space_dofs(c) for c in var.components)
_coeff_space_dofs(var::TensorField) = sum(_coeff_space_dofs(c) for c in vec(var.components))

"""Compute size (degrees of freedom) of field or equation data"""
function compute_field_size(field_or_data)
    if isa(field_or_data, Dict)
        if haskey(field_or_data, "equation_size")
            return field_or_data["equation_size"]
        elseif haskey(field_or_data, "equation_variables")
            vars = field_or_data["equation_variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        elseif haskey(field_or_data, "variables")
            vars = field_or_data["variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        end
        return 0
    elseif isa(field_or_data, ScalarField)
        return field_dofs(field_or_data)
    elseif isa(field_or_data, VectorField) || isa(field_or_data, TensorField)
        return field_dofs(field_or_data)
    elseif hasfield(typeof(field_or_data), :buffers) && get_coeff_data(field_or_data) !== nothing
        return length(get_coeff_data(field_or_data))
    elseif hasfield(typeof(field_or_data), :buffers) && get_grid_data(field_or_data) !== nothing
        return length(get_grid_data(field_or_data))
    else
        return 0
    end
end

"""
    Check if equation should be included in matrix assembly.

    An equation is included if:
    1. It has valid matrix expressions (M, L, or F)
    2. It is marked as enabled (if "enabled" key exists)
    3. It has a valid condition (if "condition" key exists)
    4. It references at least one problem variable
    5. The equation is well-formed (not flagged as invalid)

    Following patterns where equations can be conditionally
    included/excluded based on wavenumber, problem parameters, etc.
    """
function check_equation_condition(eq_data::Dict)

    # Check if equation is explicitly disabled
    if haskey(eq_data, "enabled") && !eq_data["enabled"]
        @debug "Equation excluded: explicitly disabled" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has a condition function that evaluates to false
    if haskey(eq_data, "condition")
        condition = eq_data["condition"]
        if isa(condition, Bool)
            if !condition
                @debug "Equation excluded: condition is false" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        elseif isa(condition, Function)
            # Condition is a function - evaluate it
            try
                result = condition(eq_data)
                if !result
                    @debug "Equation excluded: condition function returned false" eq_index=get(eq_data, "equation_index", 0)
                    return false
                end
            catch e
                @warn "Equation condition evaluation failed, including equation" exception=e
            end
        end
    end

    # Check if equation is flagged as invalid
    if get(eq_data, "is_invalid", false)
        @debug "Equation excluded: flagged as invalid" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has any matrix content
    has_M = haskey(eq_data, "M") && !is_zero_expression(eq_data["M"])
    has_L = haskey(eq_data, "L") && !is_zero_expression(eq_data["L"])
    has_F = haskey(eq_data, "F") && !is_zero_expression(eq_data["F"])

    if !has_M && !has_L && !has_F
        @debug "Equation excluded: no matrix content (M, L, F all zero/missing)" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check equation size
    eq_size = get(eq_data, "equation_size", 0)
    if eq_size <= 0
        @debug "Equation excluded: equation_size <= 0" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check wavenumber conditions (for spectral problems)
    if haskey(eq_data, "valid_modes")
        valid_modes = eq_data["valid_modes"]
        current_mode = get(eq_data, "current_mode", nothing)
        if current_mode !== nothing && !in(current_mode, valid_modes)
            @debug "Equation excluded: mode not in valid_modes" current_mode valid_modes
            return false
        end
    end

    # Check for wavenumber-based conditions (k=0 special handling, etc.)
    if haskey(eq_data, "exclude_k_zero") && eq_data["exclude_k_zero"]
        wavenumber = get(eq_data, "wavenumber", nothing)
        if wavenumber !== nothing
            # Check if all wavenumber components are zero
            if isa(wavenumber, Number) && wavenumber == 0
                @debug "Equation excluded: k=0 mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            elseif isa(wavenumber, Tuple) && all(k -> k == 0, wavenumber)
                @debug "Equation excluded: k=(0,...,0) mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        end
    end

    # Check for gauge conditions (pressure gauge, etc.)
    if haskey(eq_data, "is_gauge_condition") && eq_data["is_gauge_condition"]
        # Gauge conditions may have special handling
        gauge_mode = get(eq_data, "gauge_mode", nothing)
        current_mode = get(eq_data, "current_mode", nothing)

        if gauge_mode !== nothing && current_mode !== nothing
            if gauge_mode != current_mode
                # Only include gauge condition for specific mode
                return false
            end
        end
    end

    # Check if this is a boundary condition equation
    if haskey(eq_data, "is_boundary_condition") && eq_data["is_boundary_condition"]
        # Boundary conditions are always included if they're valid
        bc_valid = get(eq_data, "bc_valid", true)
        if !bc_valid
            @debug "Equation excluded: boundary condition marked invalid"
            return false
        end
    end

    # All checks passed
    return true
end

"""
    Check if equation data is structurally valid.
    Returns (is_valid::Bool, error_message::Union{String,Nothing})
    """
function is_equation_valid(eq_data::Dict)

    # Must have equation string
    if !haskey(eq_data, "equation_string")
        return (false, "Missing equation_string")
    end

    # Must have LHS
    if !haskey(eq_data, "lhs")
        return (false, "Missing LHS expression")
    end

    # Check for parse errors
    if haskey(eq_data, "parse_error")
        return (false, "Parse error: $(eq_data["parse_error"])")
    end

    # Check LHS structure if we have the expression
    lhs = eq_data["lhs"]
    if lhs !== nothing
        is_valid_lhs, lhs_info = is_proper_lhs_structure(lhs)
        if !is_valid_lhs
            return (false, "Invalid LHS structure: $(lhs_info[:error_message])")
        end
    end

    return (true, nothing)
end

"""
    Set a condition for equation inclusion in matrix assembly.
    """
function set_equation_condition!(eq_data::Dict, condition::Union{Bool, Function})
    eq_data["condition"] = condition
end

"""Enable an equation for matrix assembly."""
function enable_equation!(eq_data::Dict)
    eq_data["enabled"] = true
end

"""Disable an equation from matrix assembly."""
function disable_equation!(eq_data::Dict)
    eq_data["enabled"] = false
end

"""
    Set the valid wavenumber modes for this equation.
    The equation will only be included for these modes.
    """
function set_valid_modes!(eq_data::Dict, modes::Union{Vector, Set, AbstractRange})
    eq_data["valid_modes"] = Set(modes)
end

"""
    Exclude this equation from k=0 (homogeneous) mode.
    Useful for gauge conditions in incompressible flow problems.
    """
function exclude_k_zero!(eq_data::Dict, exclude::Bool=true)
    eq_data["exclude_k_zero"] = exclude
end

"""Get matrix expression from equation data"""
function get_matrix_expression(eq_data::Dict, matrix_name::String)
    return get(eq_data, matrix_name, nothing)
end

"""Check if expression is effectively zero"""
function is_zero_expression(expr)
    return isa(expr, ZeroOperator) || expr === nothing
end

@inline _zero_block(eqn_size::Int, var_size::Int) = spzeros(ComplexF64, eqn_size, var_size)

function _identity_block(eqn_size::Int, var_size::Int; scale::Number=1.0)
    if eqn_size == 0 || var_size == 0
        return _zero_block(eqn_size, var_size)
    end
    diag_len = min(eqn_size, var_size)
    vals = fill(ComplexF64(scale), diag_len)
    return spdiagm(eqn_size, var_size, 0 => vals)
end

# ============================================================================
# Spectral operator matrices (replaces identity markers for ETD accuracy)
# ============================================================================

"""
    _spectral_operator_matrix(expr, var, eqn_size, var_size)

Try to build the actual spectral operator matrix for `expr` acting on `var`.
Returns the matrix if successful, or `nothing` to fall back to identity markers.

Uses existing infrastructure:
- `build_operator_differentiation_matrix` for ∂/∂x (Fourier diagonal, Chebyshev dense)
- Kronecker products for multi-dimensional operators
"""
function _spectral_operator_matrix(expr, var, eqn_size::Int, var_size::Int)
    field = _get_matching_scalar_field(expr, var)
    field === nothing && return nothing

    if isa(expr, Laplacian)
        return _spectral_laplacian(field, eqn_size, var_size)
    elseif isa(expr, FractionalLaplacian)
        return _spectral_fractional_laplacian(field, expr.α, eqn_size, var_size)
    elseif isa(expr, Differentiate)
        return _spectral_differentiate(field, expr.coord, expr.order, eqn_size, var_size)
    end
    return nothing
end

"""Get the ScalarField that the operator acts on (matching var)."""
function _get_matching_scalar_field(expr, var)
    op = expr
    while hasfield(typeof(op), :operand)
        op = op.operand
    end
    if isa(op, ScalarField) && _operand_matches_variable(op, var)
        return op
    end
    return nothing
end

"""
    _coeff_shape(field)

Coefficient-space array shape, matching the actual state vector layout.
First RealFourier dimension is halved (rFFT); Chebyshev/Legendre keep N.
"""
function _coeff_shape(field::ScalarField)
    shape = Int[]
    first_rf = true
    for basis in field.bases
        basis === nothing && continue
        if isa(basis, RealFourier) && first_rf
            push!(shape, div(basis.meta.size, 2) + 1)
            first_rf = false
        else
            push!(shape, basis.meta.size)
        end
    end
    return tuple(shape...)
end

"""
    _spectral_laplacian(field, eqn_size, var_size)

Spectral Laplacian Δ = Σᵢ ∂²/∂xᵢ² in the complex coefficient representation.

- Fourier dimensions: diagonal -(2πk/L)²
- Chebyshev dimensions: dense D² matrix (from differentiation_matrix)
- Combined: Kronecker product matching the state vector layout
"""
function _spectral_laplacian(field::ScalarField, eqn_size::Int, var_size::Int)
    cshape = _coeff_shape(field)
    prod(cshape) != var_size && return nothing

    # Build 1D second-derivative operators for each dimension
    ops_1d = AbstractMatrix[]
    for (dim, basis) in enumerate(field.bases)
        basis === nothing && continue
        Nk = cshape[dim]

        if isa(basis, FourierBasis)
            # Fourier: diagonal with -k²
            N = basis.meta.size
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            k0 = 2π / L
            is_rfft = isa(basis, RealFourier) && dim == findfirst(b -> isa(b, RealFourier), field.bases)

            vals = zeros(ComplexF64, Nk)
            for j in 1:Nk
                if is_rfft
                    k = (j - 1) * k0
                else
                    k = j <= N÷2+1 ? (j-1)*k0 : (j-N-1)*k0
                end
                vals[j] = -k^2
            end
            push!(ops_1d, spdiagm(0 => vals))
        else
            # Chebyshev/Legendre: dense D² matrix
            D2 = try
                ComplexF64.(differentiation_matrix(basis, 2))
            catch
                nothing
            end
            D2 === nothing && return nothing
            push!(ops_1d, sparse(D2))
        end
    end

    isempty(ops_1d) && return nothing

    # Sum of Kronecker products: Σ_d (I ⊗ ... ⊗ D²_d ⊗ ... ⊗ I)
    ndims = length(ops_1d)
    result = spzeros(ComplexF64, var_size, var_size)
    for d in 1:ndims
        # Build Kronecker factors: identity for all dims except d
        mats = [i == d ? ops_1d[d] : sparse(ComplexF64(1)*I, size(ops_1d[i],1), size(ops_1d[i],1))
                for i in 1:ndims]
        # Kronecker in reverse order for column-major layout
        term = mats[end]
        for i in (ndims-1):-1:1
            term = kron(term, mats[i])
        end
        result += term
    end

    return size(result) == (eqn_size, var_size) ? result : nothing
end

"""Spectral fractional Laplacian: (-Δ)^α — uses |k|^{2α} for fully Fourier domains."""
function _spectral_fractional_laplacian(field::ScalarField, α::Float64, eqn_size::Int, var_size::Int)
    all_fourier = all(b -> b !== nothing && isa(b, FourierBasis), field.bases)
    !all_fourier && return nothing

    lap = _spectral_laplacian(field, eqn_size, var_size)
    lap === nothing && return nothing

    # Laplacian is diagonal for Fourier: entries are -|k|²
    # (-Δ)^α = |k|^{2α}
    diag_vals = diag(lap)
    frac_vals = [abs(v)^α for v in diag_vals]
    n = min(eqn_size, var_size, length(frac_vals))
    return spdiagm(eqn_size, var_size, 0 => ComplexF64.(frac_vals[1:n]))
end

"""
    _try_div_grad_spectral(div_expr, var, eqn_size, var_size)

Recognize div(grad(f)) or div(grad(f) + ...) as a Laplacian acting on `f`.
The first-order substitution writes div(grad_f) where
grad_f = grad(f) + ez*lift(tau). The Gradient(f) term produces the
spectral Laplacian; the lift terms are handled separately.
"""
function _try_div_grad_spectral(div_expr::Divergence, var, eqn_size::Int, var_size::Int)
    inner = div_expr.operand

    # Search all addends for Gradient(var) — handles both
    # div(grad(f)) and div(grad(f) + ez*lift(tau))
    addends = isa(inner, Gradient) ? [inner] : _collect_all_addends(inner)
    for addend in addends
        actual = isa(addend, NegateOperator) ? addend.operand : addend
        if isa(actual, Gradient) && _operand_matches_variable(actual.operand, var)
            return _spectral_laplacian_for_var(var, eqn_size, var_size)
        end
    end

    return nothing
end

"""Build spectral Laplacian sized to match a variable (scalar or block-diagonal for vector)."""
function _spectral_laplacian_for_var(var, eqn_size::Int, var_size::Int)
    if isa(var, ScalarField)
        return _spectral_laplacian(var, eqn_size, var_size)
    elseif isa(var, VectorField)
        # Block-diagonal: Δ ⊕ Δ ⊕ ... ⊕ Δ (one block per component)
        n_comp = length(var.components)
        comp_size = var_size ÷ n_comp
        lap = _spectral_laplacian(var.components[1], comp_size, comp_size)
        lap === nothing && return nothing
        # Build block-diagonal via kron(I_ncomp, lap)
        result = kron(sparse(ComplexF64(1)*I, n_comp, n_comp), lap)
        return size(result) == (eqn_size, var_size) ? result : nothing
    end
    return nothing
end

"""Collect all addends from nested Add/AddOperator/Subtract/Future chains."""
function _collect_all_addends(expr)
    result = Any[]
    if isa(expr, AddOperator)
        append!(result, _collect_all_addends(expr.left))
        append!(result, _collect_all_addends(expr.right))
    elseif isa(expr, SubtractOperator)
        append!(result, _collect_all_addends(expr.left))
        for a in _collect_all_addends(expr.right)
            push!(result, NegateOperator(a))
        end
    elseif isa(expr, Add) || isa(expr, Subtract)
        args = future_args(expr)
        if isa(expr, Add)
            for arg in args
                append!(result, _collect_all_addends(arg))
            end
        elseif length(args) >= 2
            append!(result, _collect_all_addends(args[1]))
            for a in _collect_all_addends(args[2])
                push!(result, NegateOperator(a))
            end
        end
    else
        push!(result, expr)
    end
    return result
end

"""Spectral differentiation: d^n/dx^n in complex coefficient representation."""
function _spectral_differentiate(field::ScalarField, coord::Coordinate, order::Int,
                                  eqn_size::Int, var_size::Int)
    cshape = _coeff_shape(field)
    prod(cshape) != var_size && return nothing

    # Find which dimension matches this coordinate
    dim_idx = nothing
    for (i, basis) in enumerate(field.bases)
        basis === nothing && continue
        if basis.meta.element_label == coord.name
            dim_idx = i
            break
        end
    end
    dim_idx === nothing && return nothing

    basis = field.bases[dim_idx]
    Nk = cshape[dim_idx]

    # Build 1D differentiation operator
    D1d = nothing
    if isa(basis, FourierBasis)
        N = basis.meta.size
        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        k0 = 2π / L
        is_rfft = isa(basis, RealFourier) && dim_idx == findfirst(b -> isa(b, RealFourier), field.bases)

        vals = zeros(ComplexF64, Nk)
        for j in 1:Nk
            k = is_rfft ? (j-1)*k0 : (j <= N÷2+1 ? (j-1)*k0 : (j-N-1)*k0)
            vals[j] = (im * k)^order
        end
        D1d = spdiagm(0 => vals)
    else
        D_dense = try ComplexF64.(differentiation_matrix(basis, order)) catch; nothing end
        D_dense === nothing && return nothing
        D1d = sparse(D_dense)
    end

    # Kronecker product for multi-dimensional
    ndims = length(cshape)
    if ndims == 1
        return size(D1d) == (eqn_size, var_size) ? D1d : nothing
    end

    mats = [i == dim_idx ? D1d : sparse(ComplexF64(1)*I, cshape[i], cshape[i])
            for i in 1:ndims]
    result = mats[end]
    for i in (ndims-1):-1:1
        result = kron(result, mats[i])
    end
    return size(result) == (eqn_size, var_size) ? result : nothing
end

"""
    Build matrix block for expression acting on variable.
    Following expression_matrices pattern.
    """
function build_expression_matrix_block(expr, var, eqn_size::Int, var_size::Int)

    # ── 1. Direct variable reference ────────────────────────────
    if _operand_matches_variable(expr, var)
        return _identity_block(eqn_size, var_size)
    end

    # ── 2. Arithmetic combinators ───────────────────────────────
    if isa(expr, AddOperator)
        return build_expression_matrix_block(expr.left, var, eqn_size, var_size) +
               build_expression_matrix_block(expr.right, var, eqn_size, var_size)
    end
    if isa(expr, SubtractOperator)
        return build_expression_matrix_block(expr.left, var, eqn_size, var_size) -
               build_expression_matrix_block(expr.right, var, eqn_size, var_size)
    end
    if isa(expr, NegateOperator)
        return -build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
    end
    if isa(expr, MultiplyOperator)
        # const * expr  or  expr * const  → scale the block
        if _is_const_or_param(expr.left)
            coeff = _extract_scalar(expr.left)
            return ComplexF64(coeff) * build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        elseif _is_const_or_param(expr.right)
            coeff = _extract_scalar(expr.right)
            return ComplexF64(coeff) * build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        else
            # Nonlinear product (field * field) → belongs on RHS, zero in L
            return _zero_block(eqn_size, var_size)
        end
    end
    if isa(expr, DivideOperator)
        if _is_const_or_param(expr.right)
            coeff = _extract_scalar(expr.right)
            return (ComplexF64(1) / ComplexF64(coeff)) *
                   build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        else
            return _zero_block(eqn_size, var_size)
        end
    end
    if isa(expr, PowerOperator)
        # x^const is nonlinear unless x is itself const → zero
        return _zero_block(eqn_size, var_size)
    end

    # ── 2b. Future arithmetic types (from Julia-level expression building) ──
    # The first-order substitutions (grad_u = grad(u) + ez*lift(tau_u1))
    # produce Add/Multiply/Negate/Subtract Future types, not AddOperator etc.
    if isa(expr, Future)
        args = future_args(expr)
        if isa(expr, Add)
            # Add flattens to N args: sum all blocks
            result = _zero_block(eqn_size, var_size)
            for arg in args
                result = result + build_expression_matrix_block(arg, var, eqn_size, var_size)
            end
            return result
        elseif isa(expr, Subtract) && length(args) >= 2
            return build_expression_matrix_block(args[1], var, eqn_size, var_size) -
                   build_expression_matrix_block(args[2], var, eqn_size, var_size)
        elseif isa(expr, Negate) && length(args) >= 1
            return -build_expression_matrix_block(args[1], var, eqn_size, var_size)
        elseif isa(expr, Multiply) && length(args) >= 2
            # Separate scalar constants from field/operator factors
            scalars = filter(a -> _is_const_or_param(a) || isa(a, Number), args)
            fields  = filter(a -> !(_is_const_or_param(a) || isa(a, Number)), args)
            if length(fields) > 1
                # Multiple field factors → nonlinear product → zero in L
                return _zero_block(eqn_size, var_size)
            end
            scalar_val = isempty(scalars) ? ComplexF64(1) :
                         prod(ComplexF64(_extract_scalar(s)) for s in scalars)
            if isempty(fields)
                return _zero_block(eqn_size, var_size)  # pure scalar product
            end
            return scalar_val * build_expression_matrix_block(fields[1], var, eqn_size, var_size)
        else
            # Unknown Future type (DotProduct, CrossProduct, Power, etc.)
            # These are generally nonlinear → zero in L
            return _zero_block(eqn_size, var_size)
        end
    end

    # ── 3. Constants & zeros ────────────────────────────────────
    if isa(expr, ZeroOperator) || isa(expr, ConstantOperator) ||
       isa(expr, Number) || isa(expr, AbstractArray)
        return _zero_block(eqn_size, var_size)
    end
    if isa(expr, UnknownOperator) || isa(expr, ArrayOperator)
        return _zero_block(eqn_size, var_size)
    end

    # ── 4. Spectral operator matrices ─────────────────────────────
    #
    # For Laplacian, FractionalLaplacian, Differentiate: try to build
    # the actual spectral matrix (Fourier diagonal, Chebyshev dense).
    # Falls back to identity markers if spectral matrix can't be built.
    # Accurate spectral matrices are essential for ETD time integration.

    if isa(expr, Laplacian)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size; scale=-1.0)
    end

    if isa(expr, FractionalLaplacian)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    if isa(expr, Differentiate)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    # ── 4b. div(grad(f)) → spectral Laplacian (first-order form) ──
    # The first-order substitution turns Δ(f) into div(grad_f).
    # When the Divergence operand contains a Gradient of the target variable,
    # build the spectral Laplacian matrix (sum of second-derivative operators).
    if isa(expr, Divergence)
        lap_matrix = _try_div_grad_spectral(expr, var, eqn_size, var_size)
        if lap_matrix !== nothing
            return lap_matrix
        end
    end

    if isa(expr, Union{
            # Time derivative
            TimeDerivative,
            # Differential operators (spectral matrices attempted in section 4 above;
            # this fallback handles cases where spectral matrix couldn't be built)
            FractionalLaplacian, Gradient, Divergence, Curl, Differentiate,
            # Algebraic operators
            Skew, Trace, TransposeComponents,
            # Projection / extraction
            Component, RadialComponent, AngularComponent, AzimuthalComponent,
            # Reduction operators
            Integrate, Average, Interpolate,
            # Conversion / representation
            Convert, Grid, Coeff, Lift, Copy,
            # Spectral operators
            HilbertTransform,
            # Function operators (sin(u), exp(u), etc. — nonlinear in general,
            # but we still mark participation so the matrix isn't structurally
            # singular; the implicit solve is an approximation for these)
            GeneralFunction, UnaryGridFunction,
            # CFL (diagnostic only, but handle gracefully)
            AdvectiveCFL})
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    # ── 5. Multi-operand operators ──────────────────────────────
    if isa(expr, Outer)
        # Outer product: left ⊗ right — nonlinear if both depend on variables.
        # Treat like Multiply: if one side is const, recurse the other.
        if _is_const_or_param(expr.left)
            return build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        elseif _is_const_or_param(expr.right)
            return build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        else
            return _zero_block(eqn_size, var_size)
        end
    end
    if isa(expr, IndexOperator)
        return hasfield(typeof(expr), :array) ?
               build_expression_matrix_block(expr.array, var, eqn_size, var_size) :
               _zero_block(eqn_size, var_size)
    end

    # ── 6. Fallback ─────────────────────────────────────────────
    @debug "Unhandled expression type in matrix block: $(typeof(expr))"
    return _zero_block(eqn_size, var_size)
end

# ── Helpers ─────────────────────────────────────────────────────

"""Recurse into a single-operand operator to build its matrix block."""
function _recurse_operand(expr, var, eqn_size::Int, var_size::Int; scale::Number=1.0)
    inner = build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
    return scale == 1.0 ? inner : ComplexF64(scale) * inner
end

"""
Check if an expression is a constant (number, parameter, ConstantOperator,
or a ScalarField with fixed data — e.g. unit vector components).

A ScalarField is "constant" if it has no spatial bases (0D tau variable)
or has uniform single-element data (unit vector component created by
`unit_vector_fields`).
"""
function _is_const_or_param(expr)
    isa(expr, Number) && return true
    isa(expr, ConstantOperator) && return true
    isa(expr, ZeroOperator) && return true
    isa(expr, NegateOperator) && return _is_const_or_param(expr.operand)
    # Constant ScalarField: 0D or single-element data (unit vectors, tau vars)
    if isa(expr, ScalarField)
        isempty(expr.bases) && return true
        gdata = get_grid_data(expr)
        gdata !== nothing && length(gdata) == 1 && return true
        cdata = get_coeff_data(expr)
        cdata !== nothing && length(cdata) == 1 && return true
    end
    return false
end

"""Extract scalar value from a constant expression."""
function _extract_scalar(expr)
    isa(expr, Number) && return expr
    isa(expr, ConstantOperator) && return expr.value
    isa(expr, ZeroOperator) && return 0.0
    isa(expr, NegateOperator) && return -_extract_scalar(expr.operand)
    if isa(expr, ScalarField)
        gdata = get_grid_data(expr)
        gdata !== nothing && length(gdata) >= 1 && return real(gdata[1])
        cdata = get_coeff_data(expr)
        cdata !== nothing && length(cdata) >= 1 && return real(cdata[1])
        return 0.0  # uninitialized constant → zero
    end
    return 1.0  # fallback
end

"""Build forcing vector from RHS terms"""
function build_forcing_vector(problem::Problem, eqn_sizes::Vector{Int}, total_size::Int)
    
    F_vector = zeros(ComplexF64, total_size)
    
    i0 = 0
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        eqn_size = eqn_sizes[eq_idx]
        if eqn_size > 0
            rhs_expr = get(eq_data, "F", ZeroOperator())
            
            # Evaluate RHS expression to get forcing values
            if isa(rhs_expr, ConstantOperator)
                F_vector[i0+1:i0+eqn_size] .= rhs_expr.value
            elseif isa(rhs_expr, ZeroOperator)
                F_vector[i0+1:i0+eqn_size] .= 0.0
            else
                # Complex RHS expressions would need proper evaluation
                @debug "Complex RHS expression not fully supported: $(typeof(rhs_expr))"
                F_vector[i0+1:i0+eqn_size] .= 0.0
            end
        end
        i0 += eqn_size
    end
    
    return F_vector
end

# Legacy functions (kept for compatibility)

"""
    Process LHS operator and extract contributions to system matrices.
    Following pattern where time derivatives go to M_matrix,
    spatial operators go to L_matrix.
    """
function process_lhs_operator!(L_matrix::Matrix, M_matrix::Matrix, lhs_op, eq_idx::Int, variables::Vector)
    
    if isa(lhs_op, TimeDerivative)
        # Time derivative terms go to mass matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            M_matrix[eq_idx, var_idx] = 1.0
        else
            @debug "Unknown variable in time derivative"
        end
        
    elseif isa(lhs_op, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Spatial operators go to linear operator matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            # Store operator type marker - actual spectral matrix coefficients
            # are computed during subproblem matrix assembly based on basis type
            if isa(lhs_op, Laplacian)
                L_matrix[eq_idx, var_idx] = -1.0  # Typical Laplacian sign
            else
                L_matrix[eq_idx, var_idx] = 1.0
            end
        else
            @debug "Unknown variable in spatial operator"
        end
        
    elseif isa(lhs_op, AddOperator)
        # Recursively process addition terms
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.right, eq_idx, variables)
        
    elseif isa(lhs_op, SubtractOperator)
        # Process left term normally, right term with negative sign
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        # Would need to negate contributions from right side
        # This requires more sophisticated matrix coefficient tracking
        @debug "Subtraction operator needs more sophisticated handling"
        
    elseif isa(lhs_op, MultiplyOperator)
        # Handle coefficient multiplication
        if isa(lhs_op.right, ConstantOperator)
            coeff = lhs_op.right.value
            # Apply coefficient to left operand contributions
            # This would require modifying matrix entries by coefficient
            @debug "Coefficient multiplication needs coefficient tracking: $coeff"
            process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        else
            @debug "General multiplication not yet supported"
        end
        
    elseif hasfield(typeof(lhs_op), :name)
        # Direct variable reference
        var_idx = find_variable_index(lhs_op, variables)
        if var_idx !== nothing
            L_matrix[eq_idx, var_idx] = 1.0
        end
        
    elseif isa(lhs_op, ZeroOperator)
        # Zero contribution
        nothing
        
    elseif isa(lhs_op, ConstantOperator)
        # Constant terms shouldn't appear in LHS typically
        @debug "Constant term in LHS: $(lhs_op.value)"
        
    else
        @debug "Unhandled LHS operator type: $(typeof(lhs_op))"
    end
end

"""
    Process RHS operator and extract contributions to forcing vector.
    Following pattern where RHS represents known terms/forcing.

    Recursively evaluates composite operators (Add, Subtract, Multiply) to
    compute the scalar forcing value for each equation.
    """
function process_rhs_operator!(F_vector::Vector, rhs_op, eq_idx::Int, variables::Vector)

    if isa(rhs_op, ConstantOperator)
        # Constant forcing term
        F_vector[eq_idx] = rhs_op.value

    elseif isa(rhs_op, ZeroOperator)
        # Zero RHS (homogeneous equation)
        F_vector[eq_idx] = 0.0

    elseif isa(rhs_op, AddOperator)
        # Sum of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value + right_value

    elseif isa(rhs_op, SubtractOperator)
        # Difference of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value - right_value

    elseif isa(rhs_op, MultiplyOperator)
        # Product of RHS terms
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        if isa(rhs_op.right, Number)
            F_vector[eq_idx] = left_value * rhs_op.right
        else
            right_value = evaluate_rhs_scalar(rhs_op.right, variables)
            F_vector[eq_idx] = left_value * right_value
        end

    elseif isa(rhs_op, Number)
        # Direct numeric value
        F_vector[eq_idx] = Float64(real(rhs_op))

    elseif isa(rhs_op, String) && (rhs_op == "0" || rhs_op == "zero")
        # String representation of zero
        F_vector[eq_idx] = 0.0

    else
        @debug "Unhandled RHS operator type: $(typeof(rhs_op)), using zero"
        F_vector[eq_idx] = 0.0
    end
end

"""
    evaluate_rhs_scalar(op, variables::Vector) -> Float64

Recursively evaluate an operator expression to obtain a scalar value.
Used for extracting forcing terms from composite RHS expressions.

Returns the scalar value of the expression, or 0.0 for unhandled types.
"""
function evaluate_rhs_scalar(op, variables::Vector)
    if isa(op, ConstantOperator)
        return Float64(op.value)

    elseif isa(op, ZeroOperator)
        return 0.0

    elseif isa(op, Number)
        return Float64(real(op))

    elseif isa(op, AddOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val + right_val

    elseif isa(op, SubtractOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val - right_val

    elseif isa(op, MultiplyOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        if isa(op.right, Number)
            return left_val * op.right
        else
            right_val = evaluate_rhs_scalar(op.right, variables)
            return left_val * right_val
        end

    elseif isa(op, ScalarField)
        # For field-valued RHS, we need to evaluate at specific points
        # For now, return the mean value if available
        if get_grid_data(op) !== nothing && length(get_grid_data(op)) > 0
            return real(sum(get_grid_data(op)) / length(get_grid_data(op)))
        elseif get_coeff_data(op) !== nothing && length(get_coeff_data(op)) > 0
            # First coefficient is often the mean for spectral methods
            # Use GPU-safe indexing to avoid scalar indexing on GPU arrays
            if is_gpu_array(get_coeff_data(op))
                # Copy first element to CPU to avoid GPU scalar indexing
                first_coef = Array(@view get_coeff_data(op)[1:1])[1]
            else
                first_coef = get_coeff_data(op)[1]
            end
            return real(first_coef)
        else
            return 0.0
        end

    elseif isa(op, String)
        # Try to parse as number
        if op == "0" || op == "zero"
            return 0.0
        end
        try
            return parse(Float64, op)
        catch
            return 0.0
        end

    else
        @debug "evaluate_rhs_scalar: unhandled type $(typeof(op)), returning 0.0"
        return 0.0
    end
end

"""Find index of variable in problem variable list"""
function find_variable_index(operand, variables::Vector)
    
    # Handle direct variable reference
    for (i, var) in enumerate(variables)
        if operand === var
            return i
        end
    end
    
    # Handle by name if operand has name field
    if hasfield(typeof(operand), :name)
        for (i, var) in enumerate(variables)
            if hasfield(typeof(var), :name) && operand.name == var.name
                return i
            end
        end
    end
    
    return nothing
end

