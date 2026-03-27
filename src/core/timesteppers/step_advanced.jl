# ============================================================================
# Additional Timestepper Step Functions
# ============================================================================

function step_mcnab2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Modified Crank-Nicolson Adams-Bashforth 2nd order step.

    Uses modified θ parameter for the implicit Crank-Nicolson treatment.

    The modification uses θ slightly different from 0.5 to improve stability
    for certain stiff problems while maintaining 2nd order accuracy.

    Formula:
    (M + θ*dt*L) X^{n+1} = (M - (1-θ)*dt*L) X^n + dt*(c₁*F^n + c₂*F^{n-1})

    where c₁ = 1.5, c₂ = -0.5 are Adams-Bashforth 2 coefficients.
    """

    current_state = state.history[end]
    dt = state.dt
    θ = state.timestepper.implicit_coefficient

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # Check if we have enough history for MCNAB2
    if iteration < 1 || length(state.history) < 2
        @debug "MCNAB2 requires iteration >= 1, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end

    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "MCNAB2 requires L_matrix and M_matrix, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous

    # MCNAB2 coefficients with modified θ
    # a coefficients for time derivative (same as CNAB2)
    a = [1.0/dt_current, -1.0/dt_current]
    # b coefficients for implicit treatment with modified θ
    b = [θ, 1.0 - θ]
    # c coefficients for Adams-Bashforth 2 extrapolation (variable timestep)
    c = [0.0, 1.0 + w1/2.0, -w1/2.0]

    try
        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Compute M.X[0] and L.X[0]
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current

        # Evaluate F(X[0]) at current time step
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Rotate and store history
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]

        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history
        while length(MX_history) > 2; pop!(MX_history); end
        while length(LX_history) > 2; pop!(LX_history); end
        while length(F_history) > 2; pop!(F_history); end

        # Build RHS: c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - b[1]*LX[0]
        rhs = c[2] * F_history[1]
        if length(F_history) >= 2
            rhs .+= c[3] * F_history[2]
        end
        rhs .-= a[2] * MX_history[1]
        rhs .-= b[2] * LX_history[1]

        # Build and solve LHS: (a[0]*M + b[0]*L) X = RHS
        # Cache the LU factorization keyed on (a[1], b[1]) to avoid recomputing
        # when dt is constant.
        cache_key = (a[1], b[1])
        lhs_cache = get!(state.timestepper_data, "mcnab2_lhs_cache") do
            Dict{Tuple{Float64,Float64}, Any}()
        end
        lhs_factor = get!(lhs_cache, cache_key) do
            factorize(a[1] * M_matrix + b[1] * L_matrix)
        end
        X_new = lhs_factor \ rhs

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "MCNAB2 step completed: dt=$dt_current, θ=$θ, iteration=$(state.timestepper_data["iteration"])"

    catch e
        @warn "MCNAB2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_cnlf2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Leapfrog 2nd order step with variable-timestep coefficients.

    Uses the Dedalus CNLF2 formulation (Wang 2008, Eqn 2.11) which correctly
    handles variable timesteps via the ratio w1 = dt_current / dt_previous.

    This is a 3-level method: X^{n-1}, X^n → X^{n+1}.

    Coefficients (w1 = dt_n / dt_{n-1}):
      a = [1/(1+w1)/dt, (w1-1)/dt, -w1^2/(1+w1)/dt]   # M stencil (3 levels)
      b = [1/(2*w1), (1-1/w1)/2, 1/2]                    # Wide CN stencil
      c = [0, 1, 0]                                       # Leapfrog: only F^n

    Solve: (a[1]*M + b[1]*L) X^{n+1} = RHS
    RHS = c[2]*F^n - a[2]*M*X^n - a[3]*M*X^{n-1} - b[2]*L*X^n - b[3]*L*X^{n-1}

    For constant dt (w1=1): reduces to standard CNLF with θ=0.5.
    """

    current_state = state.history[end]
    dt_current = state.dt

    # Initialize history tracking
    if !haskey(state.timestepper_data, "iteration")
        state.timestepper_data["iteration"] = 0
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
    end

    iteration = state.timestepper_data["iteration"]

    # CNLF requires X^{n-1}, so need at least 2 history states and 1 prior step
    if iteration < 1 || length(state.history) < 2
        @debug "CNLF2 requires 2 history states, falling back to CNAB1"
        step_cnab1!(state, solver)
        state.timestepper_data["iteration"] += 1
        return
    end

    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNLF2 requires L_matrix and M_matrix, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    try
        # Compute variable-timestep ratio using the main dt_history (updated by solvers.jl)
        dt_previous = get_previous_timestep(state)
        w1 = dt_current / dt_previous

        # Dedalus CNLF2 coefficients (Wang 2008 eqn 2.11)
        # a: coefficients for M stencil [X^{n+1}, X^n, X^{n-1}]
        a1 = 1.0 / ((1.0 + w1) * dt_current)
        a2 = (w1 - 1.0) / dt_current
        a3 = -w1^2 / ((1.0 + w1) * dt_current)

        # b: coefficients for L (wide Crank-Nicolson) [X^{n+1}, X^n, X^{n-1}]
        b1 = 1.0 / (2.0 * w1)
        b2 = (1.0 - 1.0 / w1) / 2.0
        b3 = 1.0 / 2.0

        # c: leapfrog only uses F^n
        # c = [0, 1, 0]

        # Get X^n and X^{n-1}
        X_current = fields_to_vector(current_state)
        X_previous = fields_to_vector(state.history[end-1])

        # Compute matrix-vector products
        MX_current = M_matrix * X_current
        MX_previous = M_matrix * X_previous
        LX_current = L_matrix * X_current
        LX_previous = L_matrix * X_previous

        # Evaluate F^n at current state
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Build RHS:
        # RHS = c[2]*F^n - a[2]*M*X^n - a[3]*M*X^{n-1} - b[2]*L*X^n - b[3]*L*X^{n-1}
        rhs = F_current_vec                       # c[2] = 1
        rhs = rhs .- a2 .* MX_current
        rhs = rhs .- a3 .* MX_previous
        rhs = rhs .- b2 .* LX_current
        rhs = rhs .- b3 .* LX_previous

        # Build and solve LHS: (a[1]*M + b[1]*L) X^{n+1} = RHS
        # Cache factorization keyed on (a1, b1) for constant-dt reuse.
        cache_key = (a1, b1)
        lhs_cache = get!(state.timestepper_data, "cnlf2_lhs_cache") do
            Dict{Tuple{Float64,Float64}, Any}()
        end
        lhs_factor = get!(lhs_cache, cache_key) do
            factorize(a1 * M_matrix + b1 * L_matrix)
        end
        X_new = lhs_factor \ rhs

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)

        state.timestepper_data["iteration"] += 1

        @debug "CNLF2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"])"

    catch e
        @warn "CNLF2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Keep reasonable history length (CNLF needs 2 previous states + current)
    if length(state.history) > 4
        popfirst!(state.history)
    end

end

function step_rksmr!(state::TimestepperState, solver::InitialValueSolver)
    """
    Strong Stability Preserving Runge-Kutta 3rd order step (SSP-RK3).

    This is the Shu-Osher form of SSP-RK3, optimal for hyperbolic PDEs.

    Shu-Osher form:
    Stage 1: u^(1) = u^n + dt*F(u^n)
    Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1) + 1/4*dt*F(u^(1))
    Stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*F(u^(2))

    Properties:
    - 3rd order accurate
    - SSP with CFL coefficient C = 1
    - TVD (Total Variation Diminishing) for scalar conservation laws
    """

    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    # RKSMR is fully explicit and assumes M = I (identity mass matrix).
    # Only warn if the problem defines a non-trivial (non-identity) M_matrix.
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if M_matrix !== nothing && size(M_matrix, 1) > 0 && norm(M_matrix - I, Inf) > 1e-12
        @warn "RKSMR (SSP-RK3) is a fully explicit method that assumes M = I. " *
              "The problem defines a non-identity M_matrix which will be ignored. " *
              "Use an IMEX method (SBDF, CNAB) for problems with non-identity mass matrix." maxlog=1
    end

    # RKSMR is purely explicit — it does NOT treat L_matrix implicitly.
    # If a non-zero L_matrix is present, warn that stiff linear terms are untreated.
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix !== nothing && size(L_matrix, 1) > 0 && norm(L_matrix, Inf) > 1e-12
        @warn "RKSMR (SSP-RK3) is a fully explicit method. The problem defines a " *
              "non-zero L_matrix (stiff linear operator) which will NOT be treated " *
              "implicitly. This may cause instability for diffusion-dominated problems. " *
              "Use an IMEX method (RK222, RK443, SBDF) for implicit linear treatment." maxlog=1
    end

    alpha = state.timestepper.alpha
    beta = state.timestepper.beta

    try
        # Stage 1: u^(1) = u^n + dt*F(u^n)
        F0 = evaluate_rhs(solver, current_state, t)
        u1 = add_scaled_state(current_state, F0, dt * beta[1])

        # Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1) + 1/4*dt*F(u^(1))
        F1 = evaluate_rhs(solver, u1, t + dt)
        # u^(2) = alpha[2,1]*u^n + alpha[2,2]*u^(1) + beta[2]*dt*F(u^(1))
        u2 = ScalarField[]
        for (i, field0) in enumerate(current_state)
            field1 = u1[i]
            new_field = ScalarField(field0.dist, field0.name, field0.bases, field0.dtype)
            ensure_layout!(field0, :g)
            ensure_layout!(field1, :g)
            ensure_layout!(F1[i], :g)
            ensure_layout!(new_field, :g)

            get_grid_data(new_field) .= alpha[2,1] .* get_grid_data(field0) .+
                                alpha[2,2] .* get_grid_data(field1) .+
                                dt * beta[2] .* get_grid_data(F1[i])
            push!(u2, new_field)
        end

        # Stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*F(u^(2))
        F2 = evaluate_rhs(solver, u2, t + 0.5*dt)  # SSP-RK3 Shu-Osher c₃ = 1/2
        new_state = ScalarField[]
        for (i, field0) in enumerate(current_state)
            field2 = u2[i]
            new_field = ScalarField(field0.dist, field0.name, field0.bases, field0.dtype)
            ensure_layout!(field0, :g)
            ensure_layout!(field2, :g)
            ensure_layout!(F2[i], :g)
            ensure_layout!(new_field, :g)

            get_grid_data(new_field) .= alpha[3,1] .* get_grid_data(field0) .+
                                alpha[3,3] .* get_grid_data(field2) .+
                                dt * beta[3] .* get_grid_data(F2[i])
            push!(new_state, new_field)
        end

        push!(state.history, new_state)

        @debug "RKSMR (SSP-RK3) step completed: dt=$dt"

    catch e
        @warn "RKSMR failed: $e, falling back to RK443"
        step_rk443!(state, solver)
        return
    end

    # Keep only necessary history
    if length(state.history) > 2
        popfirst!(state.history)
    end
end

function step_rkgfy!(state::TimestepperState, solver::InitialValueSolver)
    # Reuse the generic IMEX RK implementation for consistent M/L handling
    try
        step_rk_imex!(state, solver)
    catch e
        @warn "RKGFY failed: $e, falling back to RK443"
        step_rk443!(state, solver)
    end
end

function step_rk443_imex!(state::TimestepperState, solver::InitialValueSolver)
    # Reuse the generic IMEX RK implementation for consistent M/L handling
    try
        step_rk_imex!(state, solver)
    catch e
        @warn "RK443_IMEX failed: $e, falling back to RK443"
        step_rk443!(state, solver)
    end
end

