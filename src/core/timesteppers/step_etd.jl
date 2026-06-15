"""
    2nd-order exponential Runge-Kutta method (ETDRK2).

    Standard formulation from Cox-Matthews (2002), Eq. 22:
    Stage 1 (predictor): a_n = exp(hL)u_n
                         c = a_n + h*φ₁(hL)*N(u_n)
    Stage 2 (corrector): u_{n+1} = c + h*φ₂(hL)*(N(c) - N(u_n))

    Equivalently: u_{n+1} = a_n + h*(φ₁-φ₂)*N(u_n) + h*φ₂*N(c)

    where:
    - φ₁(z) = (exp(z) - 1)/z
    - φ₂(z) = (exp(z) - 1 - z)/z²
    - N(u) is the nonlinear term
    - L is the linear operator

    Second-order accuracy requires both N(u_n) and N(c) in the corrector
    with weights (φ₁-φ₂) and φ₂ respectively.

    References:
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems",
      J. Comput. Phys. 176, 430-455, Equation 22
    - Hochbruck & Ostermann (2010), "Exponential integrators", Acta Numerica 19, 209-286
    - Kassam & Trefethen (2005), "Fourth-Order Time Stepping for Stiff PDEs",
      SIAM J. Sci. Comput. 26(4), 1214-1233
    """
function step_etd_rk222!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # MPI pure-Fourier: the global dense matrix exponential cannot run distributed
    # (and `fields_to_vector!` has no MPI gather). The linear operator is diagonal
    # in Fourier space, so use the per-mode distributed ETD instead.
    if _distributed_diagonal_imex_applicable(solver)
        step_distributed_diagonal_etd_rk222!(state, solver)
        return
    end

    # Get linear operator from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD_RK222 requires L_matrix for linear operator, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    # L_linear = -M^{-1}*L (RHS form): converts from M*dX/dt + L*X = F
    # to the ETD form dX/dt = L_linear*X + N(X)
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    try
        # Compute matrix exponentials and φ functions (cached when dt is unchanged)
        cache = state.timestepper_data
        cache_dt = get(cache, :etd_phi_dt, nothing)
        if cache_dt === nothing || cache_dt != dt
            cache[:etd_phi] = phi_functions_matrix(L_linear, dt)
            cache[:etd_phi_dt] = dt
        end
        exp_hL, φ₁_hL, φ₂_hL = cache[:etd_phi]::NTuple{3, Matrix{ComplexF64}}

        X₀ = _timestep_fields_vector!(state, :etd_rk2_X0, current_state)

        # Diagnostic: check size match before multiply
        if size(exp_hL, 2) != length(X₀)
            field_sizes = [(f.name, isempty(f.bases) ? "tau" : "field",
                           compute_field_vector_size(f)) for f in current_state]
            @warn "ETD size mismatch: L=$(size(exp_hL)), X=$(length(X₀))" *
                  "\n  per-field: $field_sizes"
        end

        n = length(X₀)
        a_n   = _timestep_vector_buffer!(state, :etd_rk2_an,   n)
        N_buf = _timestep_vector_buffer!(state, :etd_rk2_Nbuf, n)
        c_vec = _timestep_vector_buffer!(state, :etd_rk2_c,    n)
        diff  = _timestep_vector_buffer!(state, :etd_rk2_diff, n)
        X_new = _timestep_vector_buffer!(state, :etd_rk2_Xnew, n)

        # Compute exponential propagator: a_n = exp(hL)*u_n
        mul!(a_n, exp_hL, X₀)

        # Stage 1 (predictor): Evaluate nonlinear term N(u_n) at current state
        F₀ = evaluate_rhs(solver, current_state, solver.sim_time)
        F₀_vec = _timestep_fields_vector!(state, :etd_rk2_Fraw, F₀)
        _apply_mass_inverse!(N_buf, M_factor, F₀_vec)

        # Predictor: c = a_n + h*φ₁(hL)*N(u_n)  — reuse diff as φ₁*N_u_n scratch
        mul!(diff, φ₁_hL, N_buf)
        @. c_vec = a_n + dt * diff

        temp_state = _timestep_field_state!(state, :etd_rk2_temp_state, current_state)
        vector_to_fields!(temp_state, c_vec, current_state)

        # Stage 2 (corrector): Evaluate N(c) at predicted state
        F_c = evaluate_rhs(solver, temp_state, solver.sim_time + dt)
        F_c_vec = _timestep_fields_vector!(state, :etd_rk2_Fraw, F_c)
        # Reuse diff as N_c (mass-inverse applied in place); save N_u_n → N_buf
        _apply_mass_inverse!(diff, M_factor, F_c_vec)

        # X_new = c + dt*φ₂*(N_c - N_u_n)
        # = c + dt*φ₂*diff - dt*φ₂*N_buf
        # Use X_new as scratch: X_new = N_c - N_u_n
        @. X_new = diff - N_buf
        mul!(diff, φ₂_hL, X_new)
        @. X_new = c_vec + dt * diff

        max_history = get_max_timestep_history(state.timestepper)
        _push_vector_state!(state.history, X_new, current_state, max_history)

        @debug "ETDRK2 step completed: dt=$dt, |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-RK222 failed: $e, falling back to RK222"
        step_rk222!(state, solver)
        return
    end
end

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
function step_etd_cnab2!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # MPI pure-Fourier: use the per-mode distributed ETD (the global matrix path
    # cannot run distributed). Uses the ETDRK2 scheme under MPI.
    if _distributed_diagonal_imex_applicable(solver)
        step_distributed_diagonal_etd_rk222!(state, solver)
        return
    end

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, :etd_cnab2_F_history)
        state.timestepper_data[:etd_cnab2_F_history] = Vector{Vector{ComplexF64}}()
        state.timestepper_data[:etd_cnab2_iteration] = 0
    end

    iteration = state.timestepper_data[:etd_cnab2_iteration]

    # Check if we have enough history for 2-step Adams-Bashforth
    if iteration < 1 || length(state.history) < 2
        @debug "ETD-CNAB2 requires iteration >= 1, falling back to ETDRK2 for startup"
        step_etd_rk222!(state, solver)
        state.timestepper_data[:etd_cnab2_iteration] = get(state.timestepper_data, :etd_cnab2_iteration, 0) + 1
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
    # L_linear = -M^{-1}*L (RHS form): converts from M*dX/dt + L*X = F
    # to the ETD form dX/dt = L_linear*X + N(X)
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous

    try
        # Compute exponential integrators (cached when dt is unchanged)
        cache = state.timestepper_data
        cache_dt = get(cache, :etd_cnab_phi_dt, nothing)
        if cache_dt === nothing || cache_dt != dt_current
            cache[:etd_cnab_phi] = phi_functions_matrix(L_linear, dt_current)
            cache[:etd_cnab_phi_dt] = dt_current
        end
        exp_hL, φ₁_hL, φ₂_hL = cache[:etd_cnab_phi]::NTuple{3, Matrix{ComplexF64}}

        X_current = _timestep_fields_vector!(state, :etd_cnab2_X_current, current_state)

        # Evaluate nonlinear term N(u_n) — in-place mass inverse
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_raw = _timestep_fields_vector!(state, :etd_cnab2_Fraw, F_current)
        n = length(F_raw)
        F_current_vec = _timestep_vector_buffer!(state, :etd_cnab2_Fcur, n)
        _apply_mass_inverse!(F_current_vec, M_factor, F_raw)

        # Rotate and store history
        F_history = cache[:etd_cnab2_F_history]::Vector{Vector{ComplexF64}}
        _prepend_history_buffer!(F_history, F_current_vec, 2)
        if length(F_history) < 2 && length(state.history) >= 2
            prev_state = state.history[end-1]
            F_prev_raw = _timestep_fields_vector!(state, :etd_cnab2_Fprev_raw,
                                             evaluate_rhs(solver, prev_state, solver.sim_time - dt_previous))
            F_prev_vec = _timestep_vector_buffer!(state, :etd_cnab2_Fprev, n)
            _apply_mass_inverse!(F_prev_vec, M_factor, F_prev_raw)
            push!(F_history, copy(F_prev_vec))
        end

        # 2nd-order ETD multistep (ETD-AB2) update — zero-alloc via mul!/axpy!:
        #   u_{n+1} = exp(hL)uₙ + h[φ₁·Nₙ + w·φ₂·(Nₙ − Nₙ₋₁)]
        # This is the canonical 2nd-order form (identical to step_etd_sbdf2!). The
        # previous code applied a SINGLE φ₁ to the AB2 extrapolation
        # (1+w/2)Nₙ − (w/2)Nₙ₋₁, which is 2nd order only as hL→0 and loses accuracy
        # for stiff L (the ETD regime, ~10× larger local error at |hL|~10): the
        # history term must be weighted by φ₂, not φ₁.
        Nₙ = F_history[1]
        X_new = _timestep_vector_buffer!(state, :etd_cnab2_Xnew, n)
        buf1  = _timestep_vector_buffer!(state, :etd_cnab2_buf1, n)
        mul!(X_new, exp_hL, X_current)          # exp(hL)·uₙ
        mul!(buf1, φ₁_hL, Nₙ)                   # φ₁·Nₙ
        axpy!(dt_current, buf1, X_new)           # X_new += h·φ₁·Nₙ
        if length(F_history) >= 2
            buf2 = _timestep_vector_buffer!(state, :etd_cnab2_buf2, n)
            @. buf2 = Nₙ - F_history[2]          # Nₙ − Nₙ₋₁
            mul!(buf1, φ₂_hL, buf2)              # φ₂·(Nₙ − Nₙ₋₁)
            axpy!(dt_current * w1, buf1, X_new)  # X_new += h·w·φ₂·(Nₙ − Nₙ₋₁)
        end

        _push_vector_state!(state.history, X_new, current_state, 4)
        cache[:etd_cnab2_iteration] += 1

        @debug "ETDAB2 step completed: dt=$dt_current, w1=$w1, iteration=$(cache[:etd_cnab2_iteration]), |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-CNAB2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end
end

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
        b₁(z) = φ₁(z) + w·φ₂(z)
        b₀(z) = -w·φ₂(z)

    Derivation: the interpolation slope is (Nₙ - Nₙ₋₁)/hₙ₋₁, so
    ∫₀^{hₙ} τ·exp((hₙ-τ)L) dτ · (Nₙ-Nₙ₋₁)/hₙ₋₁ = hₙ²·φ₂·(Nₙ-Nₙ₋₁)/hₙ₋₁
    = hₙ·(hₙ/hₙ₋₁)·φ₂·(Nₙ-Nₙ₋₁) = hₙ·w·φ₂·(Nₙ-Nₙ₋₁)

    References:
    - Hochbruck & Ostermann (2010), "Exponential integrators", Acta Numerica 19, 209-286
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems"
    - Beylkin, Keiser, & Vozovoi (1998), "A new class of time discretization schemes"
    """
function step_etd_sbdf2!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # MPI pure-Fourier: use the per-mode distributed ETD (global matrix path
    # cannot run distributed). Uses the ETDRK2 scheme under MPI.
    if _distributed_diagonal_imex_applicable(solver)
        step_distributed_diagonal_etd_rk222!(state, solver)
        return
    end

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, :etd_sbdf2_F_history)
        state.timestepper_data[:etd_sbdf2_F_history] = Vector{Vector{ComplexF64}}()
        state.timestepper_data[:etd_sbdf2_iteration] = 0
    end

    iteration = state.timestepper_data[:etd_sbdf2_iteration]

    # Check if we have enough history for 2-step method
    if iteration < 1 || length(state.history) < 2
        @debug "ETD-SBDF2 requires iteration >= 1, falling back to ETDRK2 for startup"
        step_etd_rk222!(state, solver)
        state.timestepper_data[:etd_sbdf2_iteration] = get(state.timestepper_data, :etd_sbdf2_iteration, 0) + 1
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
    # L_linear = -M^{-1}*L (RHS form): converts from M*dX/dt + L*X = F
    # to the ETD form dX/dt = L_linear*X + N(X)
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w = dt_current / dt_previous

    try
        # Compute exponential integrators (cached when dt is unchanged)
        cache = state.timestepper_data
        cache_dt = get(cache, :etd_sbdf_phi_dt, nothing)
        if cache_dt === nothing || cache_dt != dt_current
            cache[:etd_sbdf_phi] = phi_functions_matrix(L_linear, dt_current)
            cache[:etd_sbdf_phi_dt] = dt_current
        end
        exp_hL, φ₁_hL, φ₂_hL = cache[:etd_sbdf_phi]::NTuple{3, Matrix{ComplexF64}}

        X_current = _timestep_fields_vector!(state, :etd_sbdf2_X_current, current_state)
        n = length(X_current)

        # Evaluate nonlinear term N(uₙ) at current state — in-place mass inverse
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_raw = _timestep_fields_vector!(state, :etd_sbdf2_Fraw, F_current)
        F_current_vec = _timestep_vector_buffer!(state, :etd_sbdf2_Fcur, n)
        _apply_mass_inverse!(F_current_vec, M_factor, F_raw)

        # Rotate and store history
        F_history = cache[:etd_sbdf2_F_history]::Vector{Vector{ComplexF64}}
        _prepend_history_buffer!(F_history, F_current_vec, 2)
        if length(F_history) < 2 && length(state.history) >= 2
            prev_state = state.history[end-1]
            F_prev_raw = _timestep_fields_vector!(state, :etd_sbdf2_Fprev_raw,
                                             evaluate_rhs(solver, prev_state, solver.sim_time - dt_previous))
            F_prev_vec = _timestep_vector_buffer!(state, :etd_sbdf2_Fprev, n)
            _apply_mass_inverse!(F_prev_vec, M_factor, F_prev_raw)
            push!(F_history, copy(F_prev_vec))
        end
        if length(F_history) < 2
            @debug "ETD-SBDF2 missing previous RHS, falling back to ETDRK2"
            step_etd_rk222!(state, solver)
            cache[:etd_sbdf2_iteration] = get(cache, :etd_sbdf2_iteration, 0) + 1
            return
        end

        # N(uₙ) and N(uₙ₋₁) from typed history
        Nₙ   = F_history[1]
        Nₙ₋₁ = F_history[2]

        # Zero-alloc ETD update: u_{n+1} = exp(hL)uₙ + h[φ₁·Nₙ + w·φ₂·(Nₙ - Nₙ₋₁)]
        X_new  = _timestep_vector_buffer!(state, :etd_sbdf2_Xnew, n)
        buf1   = _timestep_vector_buffer!(state, :etd_sbdf2_buf1, n)
        buf2   = _timestep_vector_buffer!(state, :etd_sbdf2_buf2, n)

        mul!(X_new, exp_hL, X_current)      # X_new = exp(hL)*uₙ
        mul!(buf1, φ₁_hL, Nₙ)              # buf1  = φ₁·Nₙ
        axpy!(dt_current, buf1, X_new)      # X_new += h·φ₁·Nₙ
        @. buf2 = Nₙ - Nₙ₋₁               # buf2  = Nₙ - Nₙ₋₁
        mul!(buf1, φ₂_hL, buf2)             # buf1  = φ₂·(Nₙ - Nₙ₋₁)
        axpy!(dt_current * w, buf1, X_new)  # X_new += h·w·φ₂·(Nₙ - Nₙ₋₁)

        _push_vector_state!(state.history, X_new, current_state, 4)
        cache[:etd_sbdf2_iteration] += 1

        @debug "ETD-MS2 step completed: dt=$dt_current, w=$w, iteration=$(cache[:etd_sbdf2_iteration]), |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-SBDF2 failed: $e, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
end
