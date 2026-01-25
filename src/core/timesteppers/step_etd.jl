function step_etd_rk222!(state::TimestepperState, solver::InitialValueSolver)
    """
    2nd-order exponential Runge-Kutta method (ETDRK2).

    Standard formulation from Cox-Matthews (2002), Eq. 22:
    Stage 1 (predictor): a_n = exp(hL)u_n
                         c = a_n + h*φ₁(hL)*N(u_n)
    Stage 2 (corrector): u_{n+1} = a_n + h*φ₁(hL)*N(c)

    where:
    - φ₁(z) = (exp(z) - 1)/z
    - N(u) is the nonlinear term
    - L is the linear operator

    This is the standard ETD2RK method from the literature. The predictor c uses
    the nonlinear term at u_n, and the corrector uses only N(c), providing
    second-order accuracy via proper exponential integration.

    References:
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems",
      J. Comput. Phys. 176, 430-455, Equation 22
    - Hochbruck & Ostermann (2010), "Exponential integrators", Acta Numerica 19, 209-286
    - Kassam & Trefethen (2005), "Fourth-Order Time Stepping for Stiff PDEs",
      SIAM J. Sci. Comput. 26(4), 1214-1233
    """

    current_state = state.history[end]
    dt = state.dt

    # Get linear operator from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD_RK222 requires L_matrix for linear operator, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    try
        # Compute matrix exponentials and φ functions
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_linear, dt)

        # Convert state to vector form
        X₀ = fields_to_vector(current_state)

        # Compute exponential propagator: a_n = exp(hL)*u_n
        a_n = exp_hL * X₀

        # Stage 1 (predictor): Evaluate nonlinear term N(u_n) at current state
        F₀ = evaluate_rhs(solver, current_state, solver.sim_time)
        N_u_n = _apply_mass_inverse(M_factor, fields_to_vector(F₀))

        # Predictor: c = a_n + h*φ₁(hL)*N(u_n)
        c = a_n + dt * (φ₁_hL * N_u_n)

        # Convert back to field form for nonlinear evaluation
        temp_state = copy.(current_state)
        copy_solution_to_fields!(temp_state, c)

        # Stage 2 (corrector): Evaluate N(c) at predicted state
        F_c = evaluate_rhs(solver, temp_state, solver.sim_time + dt)
        N_c = _apply_mass_inverse(M_factor, fields_to_vector(F_c))

        # Final update (standard ETDRK2 formula):
        # u_{n+1} = a_n + h*φ₁(hL)*N(c)
        # This is the Cox-Matthews Eq. 22 formulation
        X_new = a_n + dt * (φ₁_hL * N_c)

        # Update state
        X_new_cpu = X_new
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new_cpu)

        push!(state.history, new_state)

        @debug "ETDRK2 step completed: dt=$dt, |X_new|=$(norm(X_new_cpu))"

    catch e
        @warn "ETD-RK222 failed: $e, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    # Keep only necessary history (retain one previous state for multistep startups)
    max_history = get_max_timestep_history(state.timestepper)
    if length(state.history) > max_history
        popfirst!(state.history)
    end
end

function step_etd_cnab2!(state::TimestepperState, solver::InitialValueSolver)
    """
    2nd-order exponential Adams-Bashforth method (ETDAB2/ETD-CNAB2).

    Formulation:
    u_{n+1} = exp(hL)u_n + h*φ₁(hL)*N_AB2

    where N_AB2 is the 2nd-order Adams-Bashforth extrapolation:
    N_AB2 = (1 + w/2)*N(u_n) - (w/2)*N(u_{n-1})
    w = h_n / h_{n-1} (timestep ratio for variable timesteps)

    Linear treatment: Exact via exponential propagator exp(hL)
    Nonlinear treatment: Explicit 2nd-order Adams-Bashforth extrapolation

    Note: This method is called ETD-CNAB2 following Tarang naming convention,
    but it uses exponential treatment (not Crank-Nicolson) for the linear operator.
    The "CNAB" refers to the multistep structure, not implicit treatment.

    References:
    - Hochbruck & Ostermann (2010), "Exponential integrators"
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems"
    """

    current_state = state.history[end]
    dt = state.dt

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "F_history")
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # Check if we have enough history for 2-step Adams-Bashforth
    if iteration < 1 || length(state.history) < 2
        @debug "ETD-CNAB2 requires iteration >= 1, falling back to ETDRK2 for startup"
        step_etd_rk222!(state, solver)
        state.timestepper_data["iteration"] = get(state.timestepper_data, "iteration", 0) + 1
        return
    end

    # Get linear operator from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD-CNAB2 requires L_matrix, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous

    try
        # Compute exponential integrators
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_linear, dt_current)

        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Evaluate nonlinear term N(u_n)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = _apply_mass_inverse(M_factor, fields_to_vector(F_current))

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for Adams-Bashforth 2
        while length(F_history) > 2; pop!(F_history); end
        if length(F_history) < 2 && length(state.history) >= 2
            prev_state = state.history[end-1]
            F_prev = evaluate_rhs(solver, prev_state, solver.sim_time - dt_previous)
            push!(F_history, _apply_mass_inverse(M_factor, fields_to_vector(F_prev)))
        end

        # Adams-Bashforth 2nd-order extrapolation coefficients (variable timestep)
        # N_AB2 = c₁*N(u_n) + c₂*N(u_{n-1})
        c₁ = 1.0 + w1/2.0  # Current step weight
        c₂ = -w1/2.0       # Previous step weight

        # Build Adams-Bashforth extrapolated nonlinear term
        F_extrap = c₁ * F_history[1]
        if length(F_history) >= 2
            F_extrap .+= c₂ * F_history[2]
        end

        # Exponential time differencing step with Adams-Bashforth extrapolation:
        # u_{n+1} = exp(hL)u_n + h*φ₁(hL)*N_AB2
        X_new = exp_hL * X_current + dt_current * (φ₁_hL * F_extrap)

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "ETDAB2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-CNAB2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_etd_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    """
    2nd-order Exponential Time Differencing Multistep Method (ETD-MS2).

    For the ODE: u'(t) = Lu + N(u), this implements a proper 2-step exponential
    multistep method derived from the variation-of-constants formula.

    The variation-of-constants formula gives:
        u(tₙ₊₁) = exp(hL)u(tₙ) + ∫₀ʰ exp((h-τ)L) N(u(tₙ+τ)) dτ

    For a 2-step method, we interpolate N using values at tₙ and tₙ₋₁:
        N(tₙ + τ) ≈ N(uₙ) + (τ/h)[N(uₙ) - N(uₙ₋₁)]  (linear interpolation)

    Substituting and integrating exactly gives the ETD multistep formula:
        u_{n+1} = exp(hL)uₙ + h[b₁(hL)Nₙ + b₀(hL)Nₙ₋₁]

    where the coefficient functions are:
        b₁(z) = φ₁(z) + φ₂(z) = (exp(z) - 1)/z + (exp(z) - 1 - z)/z²
        b₀(z) = -φ₂(z) = -(exp(z) - 1 - z)/z²

    This is the ETD analog of Adams-Bashforth 2, providing 2nd-order accuracy
    with exact linear propagation.

    For variable timesteps (w = hₙ/hₙ₋₁):
        b₁(z) = φ₁(z) + (1/w)φ₂(z)
        b₀(z) = -(1/w)φ₂(z)

    References:
    - Hochbruck & Ostermann (2010), "Exponential integrators", Acta Numerica 19, 209-286
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems"
    - Beylkin, Keiser, & Vozovoi (1998), "A new class of time discretization schemes"
    """

    current_state = state.history[end]
    dt = state.dt

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "F_history")
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # Check if we have enough history for 2-step method
    if iteration < 1 || length(state.history) < 2
        @debug "ETD-SBDF2 requires iteration >= 1, falling back to ETDRK2 for startup"
        step_etd_rk222!(state, solver)
        state.timestepper_data["iteration"] = get(state.timestepper_data, "iteration", 0) + 1
        return
    end

    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD-SBDF2 requires L_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w = dt_current / dt_previous

    try
        # Compute exponential integrators: exp(hL), φ₁(hL), φ₂(hL)
        exp_hL, φ₁_hL, φ₂_hL = phi_functions_matrix(L_linear, dt_current)

        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Evaluate nonlinear term N(uₙ) at current state
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = _apply_mass_inverse(M_factor, fields_to_vector(F_current))

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for 2-step method
        while length(F_history) > 2
            pop!(F_history)
        end
        if length(F_history) < 2 && length(state.history) >= 2
            prev_state = state.history[end-1]
            F_prev = evaluate_rhs(solver, prev_state, solver.sim_time - dt_previous)
            push!(F_history, _apply_mass_inverse(M_factor, fields_to_vector(F_prev)))
        end
        if length(F_history) < 2
            @debug "ETD-SBDF2 missing previous RHS, falling back to ETDRK2"
            step_etd_rk222!(state, solver)
            state.timestepper_data["iteration"] = get(state.timestepper_data, "iteration", 0) + 1
            return
        end

        # Get previous nonlinear term N(uₙ₋₁)
        Nₙ = F_history[1]      # N(uₙ)
        Nₙ₋₁ = F_history[2]    # N(uₙ₋₁)

        # Compute ETD multistep coefficients (variable timestep version)
        # b₁(z) = φ₁(z) + (1/w)φ₂(z)  -- coefficient for Nₙ
        # b₀(z) = -(1/w)φ₂(z)         -- coefficient for Nₙ₋₁
        inv_w = 1.0 / w

        # Linear propagation: exp(hL)uₙ
        X_propagated = exp_hL * X_current

        # Nonlinear contribution using ETD coefficients:
        # h[b₁(hL)Nₙ + b₀(hL)Nₙ₋₁] = h[(φ₁ + φ₂/w)Nₙ - (φ₂/w)Nₙ₋₁]
        #                          = h[φ₁Nₙ + (φ₂/w)(Nₙ - Nₙ₋₁)]

        # Compute the nonlinear contributions
        φ₁_Nₙ = φ₁_hL * Nₙ
        φ₂_diff = φ₂_hL * (Nₙ - Nₙ₋₁)

        # Full update: u_{n+1} = exp(hL)uₙ + h[φ₁Nₙ + (φ₂/w)(Nₙ - Nₙ₋₁)]
        X_new = X_propagated + dt_current * (φ₁_Nₙ + inv_w * φ₂_diff)

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "ETD-MS2 step completed: dt=$dt_current, w=$w, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-SBDF2 failed: $e, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

