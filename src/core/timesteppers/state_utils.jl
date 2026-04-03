# ============================================================================
# State Manipulation Utilities (GPU-aware)
# ============================================================================

"""
    evaluate_rhs(solver::InitialValueSolver, state::Vector{<:ScalarField}, time::Float64)

Evaluate right-hand side of differential equations following Tarang pattern.

IMPORTANT: This function returns RHS fields matching the state vector structure,
not the equation structure. Each state field gets a corresponding RHS field.
Equations with time derivatives contribute their F_expr to the appropriate state field;
constraint equations (no dt) contribute zero.
"""
function evaluate_rhs(solver::InitialValueSolver, state::Vector{<:ScalarField}, time::Float64)
    problem = solver.problem

    # Use compiled RHS plan if available (zero-dispatch, pre-allocated)
    if solver.compiled_rhs !== nothing && solver.compiled_rhs.is_compiled
        return execute_compiled_rhs!(solver.compiled_rhs, state, solver)
    end

    # Fallback: interpreted expression evaluation (original path)
    # Initialize RHS with zero fields for ALL state fields
    rhs = ScalarField[]
    for field in state
        push!(rhs, create_rhs_zero_field(field))
    end

    try
        # Sync current state into problem variables for expression evaluation
        sync_state_to_problem!(problem, state)

        # CRITICAL: Solve algebraic constraints before evaluating RHS
        # This is needed for MPI mode where we use explicit-only time stepping
        # and algebraic constraints (like Poisson equations) aren't handled by matrix solve
        _solve_algebraic_constraints!(problem, state)

        # Update time parameter if it exists
        if hasfield(typeof(problem), :time) && problem.time !== nothing
            if problem.time isa ScalarField
                if get_grid_data(problem.time) !== nothing
                    ensure_layout!(problem.time, :g)
                    fill!(get_grid_data(problem.time), time)
                elseif get_coeff_data(problem.time) !== nothing
                    ensure_layout!(problem.time, :c)
                    fill!(get_coeff_data(problem.time), time)
                elseif hasfield(typeof(problem), :parameters)
                    problem.parameters["t"] = time
                end
            elseif hasfield(typeof(problem.time), :data)
                problem.time.data = time
            elseif hasfield(typeof(problem.time), :value)
                problem.time.value = time
            end
        end

        # Build mapping from state field index to equation F_expr
        # Only equations with time derivatives contribute to F
        if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
            for (eq_idx, eq_data) in enumerate(problem.equation_data)
                # Check if this equation has a time derivative (M term)
                M_expr = get(eq_data, "M", nothing)
                if M_expr === nothing || _is_zero_m_term(M_expr)
                    # Constraint equation - no contribution to explicit F
                    continue
                end

                # Find which state field(s) this equation's dt term targets
                target_indices = _find_time_derivative_targets(M_expr, state, problem.variables)
                if isempty(target_indices)
                    @debug "Equation $eq_idx has M term but no matching state field target"
                    continue
                end

                # Get F expression
                expr = if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                    eq_data["F_expr"]
                else
                    get(eq_data, "F", ZeroOperator())
                end

                # Evaluate and assign to target state field(s)
                for state_idx in target_indices
                    if state_idx <= length(rhs)
                        template = state[state_idx]
                        try
                            rhs_field = evaluate_solver_expression(expr, problem.variables; layout=:g, template=template)
                            if isa(rhs_field, ScalarField)
                                ensure_layout!(rhs_field, :c)
                                rhs[state_idx] = rhs_field
                            else
                                @warn "RHS expression for state field $state_idx did not evaluate to ScalarField"
                            end
                        catch e
                            bt = catch_backtrace()
                            @warn "Failed to evaluate RHS expression for state field $state_idx: $e" backtrace=sprint(showerror, e, bt)
                        end
                    end
                end
            end
        end

        @debug "Evaluated RHS for $(length(rhs)) state fields at time $time"

        # Add stochastic forcing if registered
        if hasfield(typeof(problem), :stochastic_forcings) && !isempty(problem.stochastic_forcings)
            for (var_idx, forcing) in problem.stochastic_forcings
                if var_idx <= length(rhs)
                    rhs_field = rhs[var_idx]
                    coeff_data = get_coeff_data(rhs_field)
                    if coeff_data !== nothing
                        F_view = _matched_forcing_view(forcing, size(coeff_data))
                        ensure_layout!(rhs_field, :c)
                        if F_view !== nothing
                            coeff_data .+= F_view
                            @debug "Added stochastic forcing to state field $var_idx"
                        else
                            @warn "Forcing size doesn't match RHS size for state field $var_idx"
                        end
                    end
                end
            end
        end

    catch e
        @error "RHS evaluation failed: $e"
        rethrow()
    end

    return rhs
end

"""
    _is_zero_m_term(M_expr)

Check if M term is effectively zero (no time derivative contribution).
"""
function _is_zero_m_term(M_expr)
    if M_expr === nothing
        return true
    elseif isa(M_expr, ZeroOperator)
        return true
    elseif isa(M_expr, Number) && M_expr == 0
        return true
    elseif isa(M_expr, Vector) && isempty(M_expr)
        return true
    end
    return false
end

"""
    _find_time_derivative_targets(M_expr, state, variables)

Find which state field indices are targeted by the time derivative in M_expr.
Returns a vector of state indices.
"""
function _find_time_derivative_targets(M_expr, state::Vector{<:ScalarField}, variables::Vector)
    targets = Int[]
    _collect_dt_targets!(targets, M_expr, state, variables)
    return unique(targets)
end

function _collect_dt_targets!(targets::Vector{Int}, expr, state::Vector{<:ScalarField}, variables::Vector)
    if expr === nothing
        return
    end

    if isa(expr, TimeDerivative)
        # Found a time derivative - extract the operand variable
        operand = expr.operand
        state_idx = _find_state_index_for_operand(operand, state, variables)
        if state_idx !== nothing
            push!(targets, state_idx)
        end
        return
    end

    # Recurse into composite expressions
    if isa(expr, Vector)
        for item in expr
            _collect_dt_targets!(targets, item, state, variables)
        end
    elseif hasfield(typeof(expr), :operand)
        _collect_dt_targets!(targets, getfield(expr, :operand), state, variables)
    elseif hasfield(typeof(expr), :left)
        _collect_dt_targets!(targets, getfield(expr, :left), state, variables)
    end
    if hasfield(typeof(expr), :right)
        _collect_dt_targets!(targets, getfield(expr, :right), state, variables)
    end
end

"""
    _find_state_index_for_operand(operand, state, variables)

Find the state field index corresponding to an operand (variable or field).
Returns nothing if not found.
"""
function _find_state_index_for_operand(operand, state::Vector{<:ScalarField}, variables::Vector)
    # Direct ScalarField match
    if isa(operand, ScalarField)
        for (i, s) in enumerate(state)
            if s === operand || s.name == operand.name
                return i
            end
        end
    end

    # VectorField component - need to find which component and map to state
    if isa(operand, VectorField)
        # VectorField components are expanded in state, return first component index
        state_idx = 1
        for var in variables
            if var === operand || (isa(var, VectorField) && var.name == operand.name)
                return state_idx
            end
            if isa(var, ScalarField)
                state_idx += 1
            elseif isa(var, VectorField)
                state_idx += length(var.components)
            elseif isa(var, TensorField)
                state_idx += length(vec(var.components))
            end
        end
    end

    # String name lookup
    if isa(operand, AbstractString)
        for (i, s) in enumerate(state)
            if s.name == operand
                return i
            end
        end
    end

    return nothing
end

"""
    _zero_array!(a)

Fill an array with zeros, handling PencilArray wrappers.
"""
@inline function _zero_array!(a::AbstractArray)
    if isa(a, PencilArrays.PencilArray)
        fill!(parent(a), zero(eltype(a)))
    else
        fill!(a, zero(eltype(a)))
    end
end

"""
    _zero_like(a)

Create a zero-filled array with the same type, size, and structure as `a`.
Uses `similar` + `fill!` instead of `copy` + `fill!` to avoid copying data
that will be immediately overwritten with zeros.
"""
@inline function _zero_like(a::AbstractArray)
    z = similar(a)
    _zero_array!(z)
    return z
end

function create_rhs_zero_field(template_field::ScalarField)
    """Create a zero RHS field matching the template field properties.

    IMPORTANT: In MPI mode, we use similar() to preserve the PencilArray structure
    (same pencil decomposition) without copying data that would be immediately zeroed.
    """

    # Skip 0D fields (tau variables) which have no spatial data
    if isempty(template_field.bases)
        return ScalarField(template_field.dist, "rhs_$(template_field.name)", template_field.bases, template_field.dtype)
    end

    template_grid = get_grid_data(template_field)
    template_coeff = get_coeff_data(template_field)
    has_data = template_grid !== nothing || template_coeff !== nothing

    # Build skeleton field without allocating data (empty bases → no allocate_data!)
    rhs_field = ScalarField(template_field.dist, "rhs_$(template_field.name)", (), template_field.dtype)
    rhs_field.bases = template_field.bases
    rhs_field.domain = template_field.domain
    rhs_field.layout = template_field.layout
    rhs_field.scales = template_field.scales
    rhs_field.fft_mode = template_field.fft_mode
    rhs_field.buffers.architecture = template_field.buffers.architecture

    if has_data
        # Allocate zero-filled arrays matching template structure (preserves PencilArray decomposition)
        if template_coeff !== nothing
            set_coeff_data!(rhs_field, _zero_like(template_coeff))
            rhs_field.current_layout = :c
        end
        if template_grid !== nothing
            set_grid_data!(rhs_field, _zero_like(template_grid))
            if template_coeff === nothing
                rhs_field.current_layout = :g
            end
        end
    else
        # Template has no data — allocate from scratch (zeros by default)
        if rhs_field.domain !== nothing
            allocate_data!(rhs_field)
        end
        rhs_field.current_layout = :c
    end

    return rhs_field
end

"""
    add_scaled_state(state1::Vector{<:ScalarField}, state2::Vector{<:ScalarField}, scale::Float64)

Compute state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state(state1::Vector{<:ScalarField}, state2::Vector{<:ScalarField}, scale::Float64)
    result = ScalarField[]

    for (i, field1) in enumerate(state1)
        field2 = state2[i]

        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(field1.bases)
            new_field = ScalarField(field1.dist, field1.name, field1.bases, field1.dtype)
            push!(result, new_field)
            continue
        end

        # Use copy() to preserve PencilArray structure in MPI mode
        new_field = copy(field1)

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(new_field, :g)

        # Check for nil data
        data1 = get_grid_data(field1)
        data2 = get_grid_data(field2)
        new_data = get_grid_data(new_field)

        if data1 === nothing || data2 === nothing || new_data === nothing
            push!(result, new_field)
            continue
        end

        if is_gpu_array(data1)
            new_data .= data1 .+ scale .* data2
        else
            n = length(data1)
            use_blas = n > 2000 &&
                       data1 isa StridedArray &&
                       data2 isa StridedArray &&
                       new_data isa StridedArray
            if use_blas
                copyto!(new_data, data1)
                BLAS.axpy!(scale, data2, new_data)
            elseif n > 100
                scale_local = scale
                @turbo for j in eachindex(new_data, data1, data2)
                    new_data[j] = data1[j] + scale_local * data2[j]
                end
            else
                new_data .= data1 .+ scale .* data2
            end
        end
        push!(result, new_field)
    end

    return result
end

"""
    add_scaled_state!(dest::Vector{<:ScalarField}, state1::Vector{<:ScalarField},
                      state2::Vector{<:ScalarField}, scale::Float64)

In-place version: dest = state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state!(dest::Vector{<:ScalarField}, state1::Vector{<:ScalarField},
                           state2::Vector{<:ScalarField}, scale::Float64)
    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        dest_field = dest[i]

        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(field1.bases)
            continue
        end

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(dest_field, :g)

        # Check for nil data
        data1 = get_grid_data(field1)
        data2 = get_grid_data(field2)
        dest_data = get_grid_data(dest_field)

        if data1 === nothing || data2 === nothing || dest_data === nothing
            continue
        end

        if is_gpu_array(data1)
            dest_data .= data1 .+ scale .* data2
        else
            n = length(data1)
            use_blas = n > 2000 &&
                       data1 isa StridedArray &&
                       data2 isa StridedArray &&
                       dest_data isa StridedArray
            if use_blas
                copyto!(dest_data, data1)
                BLAS.axpy!(scale, data2, dest_data)
            elseif n > 100
                scale_local = scale
                @turbo for j in eachindex(dest_data, data1, data2)
                    dest_data[j] = data1[j] + scale_local * data2[j]
                end
            else
                dest_data .= data1 .+ scale .* data2
            end
        end
    end
end

"""
    axpy_state!(scale::Float64, x::Vector{<:ScalarField}, y::Vector{<:ScalarField})

In-place AXPY: y = y + scale * x (GPU-aware)
"""
function axpy_state!(scale::Float64, x::Vector{<:ScalarField}, y::Vector{<:ScalarField})
    for i in eachindex(x, y)
        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(x[i].bases)
            continue
        end

        ensure_layout!(x[i], :g)
        ensure_layout!(y[i], :g)

        # Check for nil data
        x_data = get_grid_data(x[i])
        y_data = get_grid_data(y[i])

        if x_data === nothing || y_data === nothing
            continue
        end

        if is_gpu_array(x_data)
            y_data .+= scale .* x_data
        else
            n = length(x_data)
            use_blas = n > 2000 &&
                       x_data isa StridedArray &&
                       y_data isa StridedArray
            if use_blas
                BLAS.axpy!(scale, x_data, y_data)
            elseif n > 100
                scale_local = scale
                @turbo for j in eachindex(y_data, x_data)
                    y_data[j] += scale_local * x_data[j]
                end
            else
                y_data .+= scale .* x_data
            end
        end
    end
end

"""
    linear_combination_state!(dest::Vector{<:ScalarField}, α::Float64, a::Vector{<:ScalarField},
                              β::Float64, b::Vector{<:ScalarField})

In-place linear combination: dest = α*a + β*b (GPU-aware)
"""
function linear_combination_state!(dest::Vector{<:ScalarField}, α::Float64, a::Vector{<:ScalarField},
                                   β::Float64, b::Vector{<:ScalarField})
    for i in eachindex(dest, a, b)
        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(a[i].bases)
            continue
        end

        ensure_layout!(a[i], :g)
        ensure_layout!(b[i], :g)
        ensure_layout!(dest[i], :g)

        # Check for nil data
        a_data = get_grid_data(a[i])
        b_data = get_grid_data(b[i])
        dest_data = get_grid_data(dest[i])

        if a_data === nothing || b_data === nothing || dest_data === nothing
            continue
        end

        if is_gpu_array(a_data)
            dest_data .= α .* a_data .+ β .* b_data
        else
            n = length(a_data)
            if n > 100
                α_local, β_local = α, β
                @turbo for j in eachindex(dest_data, a_data, b_data)
                    dest_data[j] = α_local * a_data[j] + β_local * b_data[j]
                end
            else
                dest_data .= α .* a_data .+ β .* b_data
            end
        end
    end
end

"""
    copy_state(state::Vector{<:ScalarField})

Create a deep copy of state
"""
function copy_state(state::Vector{<:ScalarField})
    new_state = ScalarField[]

    for field in state
        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(field.bases)
            new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
            push!(new_state, new_field)
            continue
        end

        # Use copy() to preserve PencilArray structure in MPI mode
        new_field = copy(field)

        push!(new_state, new_field)
    end

    return new_state
end

# Helper for stochastic forcing matching
function _matched_forcing_view(forcing, target_size)
    if forcing === nothing || forcing.cached_forcing === nothing
        return nothing
    end

    F = forcing.cached_forcing
    if size(F) == target_size
        return F
    end

    # Try to create a matching view
    try
        if ndims(F) == ndims(target_size)
            # Truncate or pad as needed
            slices = ntuple(d -> 1:min(size(F, d), target_size[d]), ndims(F))
            return view(F, slices...)
        end
    catch
        return nothing
    end

    return nothing
end

"""
    _solve_algebraic_constraints!(problem, state)

Solve algebraic constraint equations before evaluating RHS.
This is critical for MPI mode where explicit time-stepping is used
and algebraic constraints (like Poisson equations) aren't handled by matrix solve.

For QG-type problems, this handles:
1. Poisson equations: Δ(ψ) + tau - q = 0 → ψ = -Δ⁻¹(q - tau)
2. Algebraic substitutions: u - expr = 0 → u = expr
"""
function _solve_algebraic_constraints!(problem::Problem, state::Vector{<:ScalarField})
    # Only needed if we have equation_data
    if !hasfield(typeof(problem), :equation_data) || isempty(problem.equation_data)
        return
    end

    # Track which variables have been solved to ensure correct ordering
    solved_vars = Set{String}()

    # First pass: solve Poisson-type equations (need to be solved first for velocity)
    # Second pass: solve algebraic substitutions (simple var = expr)
    for pass in 1:2
        for (eq_idx, eq_data) in enumerate(problem.equation_data)
            # Skip equations with time derivatives
            M_expr = get(eq_data, "M", nothing)
            if M_expr !== nothing && !_is_zero_m_term(M_expr)
                continue
            end

            # Get LHS and RHS of constraint
            L_expr = get(eq_data, "L", nothing)
            F_expr = get(eq_data, "F", nothing)

            if L_expr === nothing
                continue
            end

            if pass == 1
                # First pass: Poisson-type equations like "Δ(ψ) + tau - q = 0"
                _try_solve_poisson_constraint!(problem, L_expr, F_expr, state, solved_vars)
            else
                # Second pass: simple substitutions like "u - skew(grad(ψ)) = 0"
                _try_solve_simple_constraint!(problem, L_expr, F_expr, state, solved_vars)
            end
        end
    end
end

"""
Try to solve a simple algebraic constraint of the form: var - expr = 0
"""
function _try_solve_simple_constraint!(problem, L_expr, F_expr, state::Vector{<:ScalarField}, solved_vars::Set{String})
    @debug "Simple constraint check: L_expr type=$(typeof(L_expr))"

    # For equations like "u - skew(grad(ψ)) = 0", the L_expr is AddOperator(VectorField, NegateOperator)
    # We need to find the variable and evaluate the expression

    target_var = nothing
    target_name = nothing
    eval_expr = nothing

    # Check if L_expr is a direct variable
    if isa(L_expr, ScalarField)
        target_var = L_expr
        target_name = L_expr.name
        eval_expr = F_expr
    elseif isa(L_expr, VectorField)
        target_var = L_expr
        target_name = L_expr.name
        eval_expr = F_expr
    elseif isa(L_expr, AddOperator)
        # Look for pattern: var - expr = 0 (represented as Add(var, Negate(expr)))
        terms = _flatten_add_terms(L_expr)
        for term in terms
            if isa(term, VectorField)
                target_var = term
                target_name = term.name
                # Find the negated expression
                for other in terms
                    if isa(other, NegateOperator)
                        eval_expr = other.operand
                        break
                    end
                end
                break
            elseif isa(term, ScalarField) && !isempty(term.bases)
                target_var = term
                target_name = term.name
                for other in terms
                    if isa(other, NegateOperator)
                        eval_expr = other.operand
                        break
                    end
                end
                break
            end
        end
    end

    if target_var === nothing || eval_expr === nothing
        @debug "Simple constraint: no valid target found"
        return
    end

    @debug "Simple constraint: target=$(target_name), eval_expr type=$(typeof(eval_expr))"

    # Don't re-solve already solved variables
    if target_name in solved_vars
        return
    end

    try
        # Evaluate the expression to get the value
        result = evaluate_solver_expression(eval_expr, problem.variables; layout=:g)

        if isa(result, ScalarField) && isa(target_var, ScalarField)
            # Update the target variable
            copy_field_data!(target_var, result)
            push!(solved_vars, target_name)
            @debug "Solved simple constraint for $target_name"
        elseif isa(result, VectorField) && isa(target_var, VectorField)
            # Update each component
            for (i, comp) in enumerate(target_var.components)
                copy_field_data!(comp, result.components[i])
            end
            push!(solved_vars, target_name)
            @debug "Solved simple constraint for $target_name (vector)"
        end
    catch e
        @debug "Could not solve simple constraint for $target_name: $e"
    end
end

"""
Try to solve a Poisson-type constraint: Δ(var) + ... = rhs
Uses spectral inversion: var = -Δ⁻¹(rhs - ...)
"""
function _try_solve_poisson_constraint!(problem, L_expr, F_expr, state::Vector{<:ScalarField}, solved_vars::Set{String})
    # Look for Laplacian operator on LHS
    laplacian_var, other_lhs_terms = _extract_laplacian_term(L_expr)

    @debug "Poisson constraint check: L_expr type=$(typeof(L_expr)), laplacian_var=$(laplacian_var !== nothing ? laplacian_var.name : "nothing")"

    if laplacian_var === nothing
        return
    end

    target_name = laplacian_var.name

    # Don't re-solve already solved variables
    if target_name in solved_vars
        return
    end

    try
        # Build RHS for Poisson equation
        # For "Δ(ψ) + tau - q = 0", we need Δ(ψ) = q - tau
        # L = Δ(ψ) + tau - q, F = 0
        # So we need: rhs = -(other_lhs_terms) = -(tau - q) = q - tau

        # First, find source variable in other_lhs_terms
        # For QG: Δ(ψ) + tau_ψ - q = 0, the source is q

        rhs_value = _evaluate_poisson_rhs(problem, other_lhs_terms, F_expr)

        if rhs_value === nothing || !isa(rhs_value, ScalarField)
            @debug "Poisson RHS is nothing or not ScalarField"
            return
        end

        @debug "About to solve Poisson for $(laplacian_var.name)"

        # Solve Poisson equation: Δ(var) = rhs → var = -rhs/k²
        result = _spectral_poisson_solve(rhs_value, laplacian_var.dist)
        @debug "Poisson solve returned: $(result !== nothing ? "success" : "nothing")"

        if result !== nothing
            copy_field_data!(laplacian_var, result)
            push!(solved_vars, target_name)
            @debug "Solved Poisson constraint for $target_name"
        end
    catch e
        @debug "Could not solve Poisson constraint for $target_name: $e"
    end
end

"""
Evaluate the RHS for a Poisson equation by negating other LHS terms.
For "Δ(ψ) + tau - q = 0", the RHS is "q - tau" = -(tau - q) = -other_lhs_terms

Special handling: 0D tau variables (gauge constants) only affect k=0 mode,
which is set to 0 by the gauge condition anyway, so we can ignore them.
"""
function _evaluate_poisson_rhs(problem, other_lhs_terms, F_expr)
    if other_lhs_terms === nothing && F_expr === nothing
        return nothing
    end

    # For Poisson equations like "Δ(ψ) + tau - q = 0", the other terms are "tau - q"
    # The RHS we need is -(tau - q) = q - tau
    # Since tau is a 0D constant affecting only k=0, and we set ψ[k=0]=0 anyway,
    # we can focus on finding the main source field (q) in the terms

    rhs_value = nothing

    if other_lhs_terms !== nothing
        # Try to extract the main source field from the other terms
        # Look for ScalarField or NegateOperator(ScalarField)
        source_field = _find_source_field_in_terms(other_lhs_terms)

        if source_field !== nothing
            @debug "Found source field: $(source_field.name)"
            rhs_value = copy(source_field)
            ensure_layout!(rhs_value, :c)
            @debug "Created RHS for Poisson solve from $(source_field.name)"
        else
            # Fallback: try to evaluate directly
            try
                other_value = evaluate_solver_expression(other_lhs_terms, problem.variables; layout=:c)
                if isa(other_value, ScalarField)
                    rhs_value = copy(other_value)
                    ensure_layout!(rhs_value, :c)
                    coeff_data = get_coeff_data(rhs_value)
                    if isa(coeff_data, PencilArrays.PencilArray)
                        parent(coeff_data) .*= -1
                    else
                        coeff_data .*= -1
                    end
                end
            catch e
                @debug "Could not evaluate other LHS terms: $e"
            end
        end
    end

    # Add F_expr contribution
    if F_expr !== nothing
        try
            f_value = evaluate_solver_expression(F_expr, problem.variables; layout=:c)
            if isa(f_value, ScalarField)
                ensure_layout!(f_value, :c)
                if rhs_value === nothing
                    rhs_value = copy(f_value)
                else
                    ensure_layout!(rhs_value, :c)
                    rhs_coeff = get_coeff_data(rhs_value)
                    f_coeff = get_coeff_data(f_value)
                    if isa(rhs_coeff, PencilArrays.PencilArray)
                        parent(rhs_coeff) .+= parent(f_coeff)
                    else
                        rhs_coeff .+= f_coeff
                    end
                end
            end
        catch e
            @debug "Could not evaluate F_expr: $e"
        end
    end

    return rhs_value
end

"""
Find the main spatial source field in an expression, ignoring 0D fields (tau variables).
For "tau - q", this returns q.
For "-q", this returns q.
"""
function _find_source_field_in_terms(expr)
    # Flatten to get all terms
    terms = _flatten_add_terms(expr)

    for term in terms
        # Direct ScalarField with spatial dimensions
        if isa(term, ScalarField) && !isempty(term.bases)
            return term
        end

        # NegateOperator around ScalarField
        if isa(term, NegateOperator)
            inner = term.operand
            if isa(inner, ScalarField) && !isempty(inner.bases)
                return inner
            end
        end
    end

    # Try to handle single term that's not an AddOperator
    if isa(expr, ScalarField) && !isempty(expr.bases)
        return expr
    end

    if isa(expr, NegateOperator)
        inner = expr.operand
        if isa(inner, ScalarField) && !isempty(inner.bases)
            return inner
        end
    end

    return nothing
end

"""
Extract Laplacian term from an expression.
Returns (variable being Laplacian'd, other terms) or (nothing, nothing).
Recursively searches nested AddOperator expressions.
"""
function _extract_laplacian_term(expr)
    if expr === nothing
        return nothing, nothing
    end

    # Direct Laplacian
    if isa(expr, Laplacian)
        if isa(expr.operand, ScalarField)
            @debug "Found direct Laplacian of $(expr.operand.name)"
            return expr.operand, nothing
        end
        return nothing, nothing
    end

    # Collect all terms from nested AddOperator structure
    terms = _flatten_add_terms(expr)
    @debug "Extracted $(length(terms)) terms from L_expr, types: $([typeof(t) for t in terms])"

    if isempty(terms)
        return nothing, nothing
    end

    # Find Laplacian term
    for (i, term) in enumerate(terms)
        if isa(term, Laplacian) && isa(term.operand, ScalarField)
            # Found Laplacian - collect other terms
            other_terms = [t for (j, t) in enumerate(terms) if j != i]
            if isempty(other_terms)
                return term.operand, nothing
            elseif length(other_terms) == 1
                return term.operand, other_terms[1]
            else
                # Reconstruct as AddOperator chain
                result = other_terms[1]
                for j in 2:length(other_terms)
                    result = AddOperator(result, other_terms[j])
                end
                return term.operand, result
            end
        end
    end

    return nothing, nothing
end

"""
Flatten nested AddOperator into a list of terms.
"""
function _flatten_add_terms(expr)
    terms = Any[]
    _collect_add_terms!(terms, expr)
    return terms
end

function _collect_add_terms!(terms::Vector{Any}, expr)
    if expr === nothing
        return
    end

    if isa(expr, AddOperator)
        _collect_add_terms!(terms, expr.left)
        _collect_add_terms!(terms, expr.right)
    else
        push!(terms, expr)
    end
end

"""
Spectral Poisson solve: given rhs, compute var = -rhs/k²
Works with MPI/PencilArrays.
"""
function _spectral_poisson_solve(rhs::ScalarField, dist::Distributor)
    ensure_layout!(rhs, :c)

    # Create result field
    result = copy(rhs)
    result.name = "poisson_solution"
    ensure_layout!(result, :c)

    coeff_data = get_coeff_data(result)

    if isempty(rhs.bases)
        return result
    end

    # Build k² for spectral Poisson inversion
    k_squared = _build_k_squared(rhs)

    if k_squared === nothing
        @warn "Could not build k² for Poisson solve"
        return nothing
    end

    # Apply Poisson inversion: result = -rhs/k² (avoiding k=0)
    if isa(coeff_data, PencilArrays.PencilArray)
        local_data = parent(coeff_data)
        for idx in CartesianIndices(local_data)
            k2 = k_squared[idx]
            if k2 > 1e-12
                local_data[idx] = -local_data[idx] / k2
            else
                local_data[idx] = 0.0  # k=0: gauge condition
            end
        end
    else
        for idx in CartesianIndices(coeff_data)
            k2 = k_squared[idx]
            if k2 > 1e-12
                coeff_data[idx] = -coeff_data[idx] / k2
            else
                coeff_data[idx] = 0.0
            end
        end
    end

    return result
end

"""
Build k² array for spectral Poisson inversion.
"""
function _build_k_squared(field::ScalarField)
    ensure_layout!(field, :c)
    coeff_data = get_coeff_data(field)

    if coeff_data === nothing
        return nothing
    end

    # Get local shape
    local_shape = if isa(coeff_data, PencilArrays.PencilArray)
        size(parent(coeff_data))
    else
        size(coeff_data)
    end

    k_squared = zeros(real(eltype(coeff_data)), local_shape)

    # Get wavenumber contributions from each basis
    for (dim, basis) in enumerate(field.bases)
        k_array = _get_wavenumber_array_for_poisson(field, dim, local_shape)
        if k_array !== nothing
            for idx in CartesianIndices(k_squared)
                k_squared[idx] += k_array[idx]^2
            end
        end
    end

    return k_squared
end

"""
Get wavenumber array for a specific dimension, handling MPI offsets.
"""
function _get_wavenumber_array_for_poisson(field::ScalarField, dim::Int, local_shape)
    if dim > length(field.bases)
        return nothing
    end

    basis = field.bases[dim]
    N = basis.meta.size

    # Get domain size from basis bounds
    bounds = basis.meta.bounds
    L = bounds[2] - bounds[1]

    k_scale = 2π / L

    coeff_data = get_coeff_data(field)
    is_mpi = isa(coeff_data, PencilArrays.PencilArray)

    # Initialize wavenumber array
    k_array = zeros(local_shape)

    if is_mpi
        pencil = PencilArrays.pencil(coeff_data)
        local_axes = pencil.axes_local
        perm = PencilArrays.permutation(coeff_data)
        perm_tuple = Tuple(perm)
        physical_dim = findfirst(==(dim), perm_tuple)
        if physical_dim === nothing
            physical_dim = dim
        end

        if physical_dim > length(local_axes)
            return zeros(local_shape)
        end

        global_start = first(local_axes[physical_dim])

        is_rfft = isa(basis, RealFourier)

        for idx in CartesianIndices(k_array)
            local_idx = Tuple(idx)[physical_dim]
            global_idx = global_start + local_idx - 1

            if is_rfft
                k = (global_idx - 1) * k_scale
            else
                if global_idx <= N÷2 + 1
                    k = (global_idx - 1) * k_scale
                else
                    k = (global_idx - N - 1) * k_scale
                end
            end
            k_array[idx] = k
        end
    else
        is_rfft = isa(basis, RealFourier)

        for idx in CartesianIndices(k_array)
            global_idx = Tuple(idx)[dim]

            if is_rfft
                k = (global_idx - 1) * k_scale
            else
                if global_idx <= N÷2 + 1
                    k = (global_idx - 1) * k_scale
                else
                    k = (global_idx - N - 1) * k_scale
                end
            end
            k_array[idx] = k
        end
    end

    return k_array
end
