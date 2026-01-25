# ============================================================================
# State Manipulation Utilities (GPU-aware)
# ============================================================================

"""
    evaluate_rhs(solver::InitialValueSolver, state::Vector{ScalarField}, time::Float64)

Evaluate right-hand side of differential equations following Tarang pattern.

This evaluates the F expressions from problem.equation_data for each equation.
"""
function evaluate_rhs(solver::InitialValueSolver, state::Vector{ScalarField}, time::Float64)
    problem = solver.problem
    rhs = ScalarField[]

    try
        # Sync current state into problem variables for expression evaluation
        sync_state_to_problem!(problem, state)

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

        # Evaluate each equation's RHS (F expression)
        if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
            for (eq_idx, eq_data) in enumerate(problem.equation_data)
                template = state[min(eq_idx, length(state))]
                expr = if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                    eq_data["F_expr"]
                else
                    get(eq_data, "F", ZeroOperator())
                end

                try
                    rhs_field = evaluate_solver_expression(expr, problem.variables; layout=:g, template=template)
                    if isa(rhs_field, ScalarField)
                        ensure_layout!(rhs_field, :c)
                        push!(rhs, rhs_field)
                    else
                        @warn "RHS expression $eq_idx did not evaluate to ScalarField, using zero field"
                        push!(rhs, create_rhs_zero_field(template))
                    end
                catch e
                    @warn "Failed to evaluate RHS expression for equation $eq_idx: $e"
                    push!(rhs, create_rhs_zero_field(template))
                end
            end
        else
            @warn "No equation_data found in problem, creating zero fields"
            for field in state
                push!(rhs, create_rhs_zero_field(field))
            end
        end

        @debug "Evaluated RHS for $(length(rhs)) equations at time $time"

        # Add stochastic forcing if registered
        if hasfield(typeof(problem), :stochastic_forcings) && !isempty(problem.stochastic_forcings)
            for (var_idx, forcing) in problem.stochastic_forcings
                if var_idx <= length(rhs)
                    rhs_field = rhs[var_idx]
                    F_view = _matched_forcing_view(forcing, size(get_coeff_data(rhs_field)))

                    ensure_layout!(rhs_field, :c)
                    if F_view !== nothing
                        get_coeff_data(rhs_field) .+= F_view
                        @debug "Added stochastic forcing to equation $var_idx"
                    else
                        @warn "Forcing size doesn't match RHS size for equation $var_idx"
                    end
                end
            end
        end

    catch e
        @error "RHS evaluation failed: $e"
        for field in state
            rhs_field = create_rhs_zero_field(field)
            push!(rhs, rhs_field)
        end
    end

    return rhs
end

function create_rhs_zero_field(template_field::ScalarField)
    """Create a zero RHS field matching the template field properties"""
    rhs_field = ScalarField(template_field.dist, "rhs_$(template_field.name)", template_field.bases, template_field.dtype)
    ensure_layout!(rhs_field, :c)
    fill!(get_coeff_data(rhs_field), zero(eltype(get_coeff_data(rhs_field))))
    return rhs_field
end

"""
    add_scaled_state(state1::Vector{ScalarField}, state2::Vector{ScalarField}, scale::Float64)

Compute state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state(state1::Vector{ScalarField}, state2::Vector{ScalarField}, scale::Float64)
    result = ScalarField[]

    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        new_field = ScalarField(field1.dist, field1.name, field1.bases, field1.dtype)

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(new_field, :g)

        if is_gpu_array(get_grid_data(field1))
            get_grid_data(new_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
        else
            n = length(get_grid_data(field1))
            use_blas = n > 2000 &&
                       get_grid_data(field1) isa StridedArray &&
                       get_grid_data(field2) isa StridedArray &&
                       get_grid_data(new_field) isa StridedArray
            if use_blas
                copyto!(get_grid_data(new_field), get_grid_data(field1))
                BLAS.axpy!(scale, get_grid_data(field2), get_grid_data(new_field))
            elseif n > 100
                scale_local = scale
                new_data = get_grid_data(new_field)
                data1 = get_grid_data(field1)
                data2 = get_grid_data(field2)
                @turbo for j in eachindex(new_data, data1, data2)
                    new_data[j] = data1[j] + scale_local * data2[j]
                end
            else
                get_grid_data(new_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
            end
        end
        push!(result, new_field)
    end

    return result
end

"""
    add_scaled_state!(dest::Vector{ScalarField}, state1::Vector{ScalarField},
                      state2::Vector{ScalarField}, scale::Float64)

In-place version: dest = state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state!(dest::Vector{ScalarField}, state1::Vector{ScalarField},
                           state2::Vector{ScalarField}, scale::Float64)
    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        dest_field = dest[i]

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(dest_field, :g)

        if is_gpu_array(get_grid_data(field1))
            get_grid_data(dest_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
        else
            n = length(get_grid_data(field1))
            use_blas = n > 2000 &&
                       get_grid_data(field1) isa StridedArray &&
                       get_grid_data(field2) isa StridedArray &&
                       get_grid_data(dest_field) isa StridedArray
            if use_blas
                copyto!(get_grid_data(dest_field), get_grid_data(field1))
                BLAS.axpy!(scale, get_grid_data(field2), get_grid_data(dest_field))
            elseif n > 100
                scale_local = scale
                dest_data = get_grid_data(dest_field)
                data1 = get_grid_data(field1)
                data2 = get_grid_data(field2)
                @turbo for j in eachindex(dest_data, data1, data2)
                    dest_data[j] = data1[j] + scale_local * data2[j]
                end
            else
                get_grid_data(dest_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
            end
        end
    end
end

"""
    axpy_state!(scale::Float64, x::Vector{ScalarField}, y::Vector{ScalarField})

In-place AXPY: y = y + scale * x (GPU-aware)
"""
function axpy_state!(scale::Float64, x::Vector{ScalarField}, y::Vector{ScalarField})
    for i in eachindex(x, y)
        ensure_layout!(x[i], :g)
        ensure_layout!(y[i], :g)

        if is_gpu_array(get_grid_data(x[i]))
            get_grid_data(y[i]) .+= scale .* get_grid_data(x[i])
        else
            n = length(get_grid_data(x[i]))
            use_blas = n > 2000 &&
                       get_grid_data(x[i]) isa StridedArray &&
                       get_grid_data(y[i]) isa StridedArray
            if use_blas
                BLAS.axpy!(scale, get_grid_data(x[i]), get_grid_data(y[i]))
            elseif n > 100
                scale_local = scale
                y_data = get_grid_data(y[i])
                x_data = get_grid_data(x[i])
                @turbo for j in eachindex(y_data, x_data)
                    y_data[j] += scale_local * x_data[j]
                end
            else
                get_grid_data(y[i]) .+= scale .* get_grid_data(x[i])
            end
        end
    end
end

"""
    linear_combination_state!(dest::Vector{ScalarField}, α::Float64, a::Vector{ScalarField},
                              β::Float64, b::Vector{ScalarField})

In-place linear combination: dest = α*a + β*b (GPU-aware)
"""
function linear_combination_state!(dest::Vector{ScalarField}, α::Float64, a::Vector{ScalarField},
                                   β::Float64, b::Vector{ScalarField})
    for i in eachindex(dest, a, b)
        ensure_layout!(a[i], :g)
        ensure_layout!(b[i], :g)
        ensure_layout!(dest[i], :g)

        if is_gpu_array(get_grid_data(a[i]))
            get_grid_data(dest[i]) .= α .* get_grid_data(a[i]) .+ β .* get_grid_data(b[i])
        else
            n = length(get_grid_data(a[i]))
            if n > 100
                α_local, β_local = α, β
                dest_data = get_grid_data(dest[i])
                a_data = get_grid_data(a[i])
                b_data = get_grid_data(b[i])
                @turbo for j in eachindex(dest_data, a_data, b_data)
                    dest_data[j] = α_local * a_data[j] + β_local * b_data[j]
                end
            else
                get_grid_data(dest[i]) .= α .* get_grid_data(a[i]) .+ β .* get_grid_data(b[i])
            end
        end
    end
end

"""
    copy_state(state::Vector{ScalarField})

Create a deep copy of state
"""
function copy_state(state::Vector{ScalarField})
    new_state = ScalarField[]

    for field in state
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(new_field, :g)
        get_grid_data(new_field) .= get_grid_data(field)
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
