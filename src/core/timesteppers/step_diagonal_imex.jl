# =============================================================================
# Diagonal IMEX Step Functions (GPU-native)
# =============================================================================

"""
    step_diagonal_imex_rk222!(state::TimestepperState, solver::InitialValueSolver)

2nd-order diagonal IMEX RK step with GPU-native implicit treatment.

Uses the standard IMEX-RK formulation where:
- Explicit tableau handles nonlinear terms F(u)
- Implicit diagonal operator L handles linear terms (viscosity/hyperviscosity)

For each stage s, we solve:
    (1 + dt*γ*L̂) * Ŷ_s = X̂_n + dt * Σ_{j<s} a_j * F̂_j

where L̂ is the diagonal spectral operator.

This avoids sparse matrix solves and stays 100% on GPU.
"""
function step_diagonal_imex_rk222!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    # Get spectral linear operator from solver/problem
    L_spectral = _get_spectral_linear_operator(solver)

    if L_spectral === nothing
        # No spectral operator: fall back to explicit RK
        @debug "DiagonalIMEX_RK222: No spectral operator, using explicit RK"
        _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
        return
    end

    γ = ts.γ
    A = ts.A_explicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit
    stages = ts.stages

    # Stage storage for RHS evaluations
    F_stages = Vector{Vector{ScalarField}}(undef, stages)
    # Stage values
    Y_stages = Vector{Vector{ScalarField}}(undef, stages)

    # IMEX-RK with per-stage implicit solve, all in coefficient space:
    # For each stage s, solve (I + dt*γ*L̂) Ŷ_s = X̂_n + dt * Σ_{j<s} a_{sj} F̂_j
    # then evaluate the explicit RHS at the implicitly-solved stage value.
    #
    # WARNING: This implementation omits the off-diagonal implicit contributions
    # (- dt * Σ_{j<s} A_imp[s,j] * L̂ * Ŷ_j) from the stage RHS. For stiffly
    # accurate ESDIRK methods this means the method is only first-order accurate
    # in the stiff limit (dt*|λ| >> 1). The full IMEX-RK formulation is in step_rk.jl.

    # Ensure initial state is in coefficient space
    for field in current_state
        ensure_layout!(field, :c)
    end

    for s in 1:stages
        state.current_substep = s

        # Build explicit contribution in coefficient space:
        # Ŷ_s = X̂_n + dt * Σ_{j<s} A[s,j] * F̂_j
        Y_s = copy_state(current_state)
        for (k, field) in enumerate(Y_s)
            ensure_layout!(field, :c)
            coeff_data = get_coeff_data(field)
            for j in 1:(s-1)
                if abs(A[s, j]) > 1e-14
                    ensure_layout!(F_stages[j][k], :c)
                    coeff_data .+= dt .* A[s, j] .* get_coeff_data(F_stages[j][k])
                end
            end

            # Apply per-stage implicit solve: Ŷ_s = RHS / (1 + dt*γ*L̂)
            coeff_data ./= (1 .+ dt .* γ .* L_spectral.coefficients)
        end

        Y_stages[s] = Y_s

        # Evaluate nonlinear RHS at implicitly-solved stage value
        F_stages[s] = evaluate_rhs(solver, Y_s, t + c[s] * dt)
    end

    # Final update in coefficient space using separate explicit and implicit weights:
    # X̂_{n+1} = X̂_n + dt * Σ b_exp[s] * F̂_s - dt * Σ b_imp[s] * L̂ * Ŷ_s
    new_state = copy_state(current_state)
    for (k, field) in enumerate(new_state)
        ensure_layout!(field, :c)
        coeff_data = get_coeff_data(field)
        for s in 1:stages
            # Explicit contribution: +dt * b_exp[s] * F̂_s
            if abs(b_exp[s]) > 1e-14
                ensure_layout!(F_stages[s][k], :c)
                coeff_data .+= dt .* b_exp[s] .* get_coeff_data(F_stages[s][k])
            end
            # Implicit contribution: -dt * b_imp[s] * L̂ * Ŷ_s
            if abs(b_imp[s]) > 1e-14
                ensure_layout!(Y_stages[s][k], :c)
                coeff_data .-= dt .* b_imp[s] .* L_spectral.coefficients .* get_coeff_data(Y_stages[s][k])
            end
        end
    end

    push!(state.history, new_state)

    if length(state.history) > 1
        popfirst!(state.history)
    end
end

"""
    step_diagonal_imex_rk443!(state::TimestepperState, solver::InitialValueSolver)

3rd-order diagonal IMEX RK step with GPU-native implicit treatment.

Uses classical RK4 explicit tableau for nonlinear terms, with implicit
treatment of linear operator at each stage and final update.
"""
function step_diagonal_imex_rk443!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    # Get spectral linear operator from solver/problem
    L_spectral = _get_spectral_linear_operator(solver)

    if L_spectral === nothing
        @debug "DiagonalIMEX_RK443: No spectral operator, using explicit RK"
        _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
        return
    end

    A = ts.A_explicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit
    γ_diag = ts.A_implicit_diag
    stages = ts.stages

    # IMEX-RK with per-stage implicit solve (SDIRK structure), all in coefficient space:
    # For each stage s, solve (I + dt*γ_s*L̂) Ŷ_s = X̂_n + dt * Σ_{j<s} a_{sj} F̂_j
    F_stages = Vector{Vector{ScalarField}}(undef, stages)
    Y_stages = Vector{Vector{ScalarField}}(undef, stages)

    # Ensure initial state is in coefficient space
    for field in current_state
        ensure_layout!(field, :c)
    end

    for s in 1:stages
        state.current_substep = s

        # Build explicit contribution in coefficient space:
        # Ŷ_s = X̂_n + dt * Σ_{j<s} A[s,j] * F̂_j
        Y_s = copy_state(current_state)
        γ_s = γ_diag[s]
        for (k, field) in enumerate(Y_s)
            ensure_layout!(field, :c)
            coeff_data = get_coeff_data(field)
            for j in 1:(s-1)
                if abs(A[s, j]) > 1e-14
                    ensure_layout!(F_stages[j][k], :c)
                    coeff_data .+= dt .* A[s, j] .* get_coeff_data(F_stages[j][k])
                end
            end

            # Apply per-stage implicit solve: Ŷ_s = RHS / (1 + dt*γ_s*L̂)
            coeff_data ./= (1 .+ dt .* γ_s .* L_spectral.coefficients)
        end

        Y_stages[s] = Y_s

        # Evaluate nonlinear RHS at implicitly-solved stage value
        F_stages[s] = evaluate_rhs(solver, Y_s, t + c[s] * dt)
    end

    # Final update in coefficient space using separate explicit and implicit weights:
    # X̂_{n+1} = X̂_n + dt * Σ b_exp[s] * F̂_s - dt * Σ b_imp[s] * L̂ * Ŷ_s
    new_state = copy_state(current_state)
    for (k, field) in enumerate(new_state)
        ensure_layout!(field, :c)
        coeff_data = get_coeff_data(field)
        for s in 1:stages
            # Explicit contribution: +dt * b_exp[s] * F̂_s
            if abs(b_exp[s]) > 1e-14
                ensure_layout!(F_stages[s][k], :c)
                coeff_data .+= dt .* b_exp[s] .* get_coeff_data(F_stages[s][k])
            end
            # Implicit contribution: -dt * b_imp[s] * L̂ * Ŷ_s
            if abs(b_imp[s]) > 1e-14
                ensure_layout!(Y_stages[s][k], :c)
                coeff_data .-= dt .* b_imp[s] .* L_spectral.coefficients .* get_coeff_data(Y_stages[s][k])
            end
        end
    end

    push!(state.history, new_state)

    if length(state.history) > 1
        popfirst!(state.history)
    end
end

"""
    step_diagonal_imex_sbdf2!(state::TimestepperState, solver::InitialValueSolver)

2nd-order SBDF with diagonal spectral implicit treatment.

SBDF2 scheme:
    (3/2)X_{n+1} - 2X_n + (1/2)X_{n-1} = dt * (2F_n - F_{n-1}) - dt*L*X_{n+1}

Rearranged:
    (3/2 + dt*L) X_{n+1} = 2X_n - (1/2)X_{n-1} + dt*(2F_n - F_{n-1})

With diagonal spectral operator:
    X̂_{n+1} = RHS / (3/2 + dt*L̂)
"""
function step_diagonal_imex_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    L_spectral = _get_spectral_linear_operator(solver)

    # Initialize history if needed
    if !haskey(state.timestepper_data, "F_history")
        state.timestepper_data["F_history"] = Vector{ScalarField}[]
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]
    F_history = state.timestepper_data["F_history"]

    # Evaluate current RHS
    F_n = evaluate_rhs(solver, current_state, t)

    if iteration == 0 || length(state.history) < 2
        # First step: use backward Euler (SBDF1)
        # (1 + dt*L) X_{n+1} = X_n + dt*F_n
        new_state = copy_state(current_state)
        axpy_state!(dt, F_n, new_state)

        if L_spectral !== nothing
            for field in new_state
                # X̂ = RHS / (1 + dt*L̂)
                ensure_layout!(field, :c)
                get_coeff_data(field) ./= (1 .+ dt .* L_spectral.coefficients)
            end
        end

        push!(state.history, new_state)
        if length(state.history) > 2
            popfirst!(state.history)
        end

        # Store F history
        push!(F_history, F_n)
        if length(F_history) > 2
            popfirst!(F_history)
        end
    else
        # SBDF2 step
        X_n = current_state
        X_nm1 = state.history[end-1]
        F_nm1 = F_history[end]

        # BDF2 + Adams-Bashforth extrapolation with constant-dt coefficients.
        # For variable timestep, these should use w = dt/dt_prev ratios.
        dt_prev = length(state.dt_history) >= 2 ? state.dt_history[end-1] : dt
        if !isapprox(dt, dt_prev, rtol=0.01)
            @warn "DiagonalIMEX_SBDF2 uses constant-dt coefficients but dt/dt_prev = $(dt/dt_prev). Use SBDF2 (step_multistep.jl) for variable timesteps." maxlog=1
        end
        # RHS = 2*X_n - (1/2)*X_{n-1} + dt*(2*F_n - F_{n-1})
        new_state = ScalarField[]
        for (i, field_n) in enumerate(X_n)
            field_nm1 = X_nm1[i]
            f_n = F_n[i]
            f_nm1 = F_nm1[i]

            result = ScalarField(field_n.dist, field_n.name, field_n.bases, field_n.dtype)

            ensure_layout!(field_n, :c)
            ensure_layout!(field_nm1, :c)
            ensure_layout!(f_n, :c)
            ensure_layout!(f_nm1, :c)
            ensure_layout!(result, :c)

            # RHS = 2*X_n - 0.5*X_{n-1} + dt*(2*F_n - F_{n-1})
            get_coeff_data(result) .= 2 .* get_coeff_data(field_n) .-
                                       0.5 .* get_coeff_data(field_nm1) .+
                                       dt .* (2 .* get_coeff_data(f_n) .- get_coeff_data(f_nm1))

            # Implicit step: X̂ = RHS / (1.5 + dt*L̂)
            if L_spectral !== nothing
                get_coeff_data(result) ./= (1.5 .+ dt .* L_spectral.coefficients)
            else
                get_coeff_data(result) ./= 1.5
            end

            push!(new_state, result)
        end

        push!(state.history, new_state)
        if length(state.history) > 2
            popfirst!(state.history)
        end

        # Update F history
        push!(F_history, F_n)
        if length(F_history) > 2
            popfirst!(F_history)
        end
    end

    state.timestepper_data["iteration"] = iteration + 1
end

# Note: _get_spectral_linear_operator and set_spectral_linear_operator! are
# defined in spectral_operators.jl which is included before this file.
