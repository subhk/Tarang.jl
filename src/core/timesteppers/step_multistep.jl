function step_cnab1!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Adams-Bashforth 1st order following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:95-188 MultistepIMEX.step method:
    - Proper coefficient computation: a[0] = 1/dt, a[1] = -1/dt, b[0] = 1/2, b[1] = 1/2, c[1] = 1
    - RHS construction: c[1]*F[0] - a[1]*MX[0] - b[1]*LX[0] (following lines 156-166)
    - LHS solution: (a[0]*M + b[0]*L).X = RHS (following lines 174-184)
    - Proper state rotation and history management
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed (following Tarang MultistepIMEX.__init__)
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNAB1 requires L_matrix and M_matrix, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end

    # Get CNAB1 coefficients following Tarang (timesteppers:206-220)
    a = [1.0/dt, -1.0/dt]  # a[0], a[1]
    b = [0.5, 0.5]         # b[0], b[1] 
    c = [0.0, 1.0]         # c[0], c[1]
    
    try
        # Step 1: Convert current state to vector (following Tarang gather_inputs)
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Tarang lines 142-147)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step (following Tarang lines 149-153)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history (following Tarang lines 124-126)
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for CNAB1 (amax=1, bmax=1, cmax=1)
        while length(MX_history) > 1; pop!(MX_history); end
        while length(LX_history) > 1; pop!(LX_history); end
        while length(F_history) > 1; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang exactly (timesteppers:156-166)
        # RHS = c[1] * F[0] - a[1] * MX[0] - b[1] * LX[0]
        rhs = c[2] * F_history[1]  # c[1] * F[0] (using 1-based indexing)
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(LX_history) >= 1  # b[1] term
            rhs .-= b[2] * LX_history[1]  # -b[1] * LX[0]
        end
        
        # Step 6: Build and solve LHS system (following Tarang lines 174-184)
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state (following Tarang scatter_inputs)
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "CNAB1 step completed: dt=$dt, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "CNAB1 failed: $e, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 3
        popfirst!(state.history)
    end
end

function step_cnab2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Adams-Bashforth 2nd order following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:95-188 MultistepIMEX.step method:
    - Variable timestep coefficients: w1 = k1/k0, c[1] = 1 + w1/2, c[2] = -w1/2 (lines 276-290)
    - Full RHS construction: c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - b[1]*LX[0] (lines 156-166)
    - Proper history management with rotation for MX, LX, F arrays (lines 124-126)
    - Falls back to CNAB1 for iteration < 1 (line 274)
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    iteration = state.timestepper_data["iteration"]
    
    # Check if we have enough history for CNAB2 (following Tarang line 274)
    if iteration < 1 || length(state.history) < 2
        @debug "CNAB2 requires iteration >= 1, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNAB2 requires L_matrix and M_matrix, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
    # Get timestep history for variable timestep (following Tarang lines 280-281)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get CNAB2 coefficients following Tarang exactly (timesteppers:283-288)
    a = [1.0/dt_current, -1.0/dt_current]  # a[0], a[1]
    b = [0.5, 0.5]                         # b[0], b[1]
    c = [0.0, 1.0 + w1/2.0, -w1/2.0]      # c[0], c[1], c[2]
    
    @debug "CNAB2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Tarang lines 142-147)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step (following Tarang lines 149-153)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history (following Tarang lines 124-126)
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for CNAB2 (amax=2, bmax=2, cmax=2)
        while length(MX_history) > 2; pop!(MX_history); end
        while length(LX_history) > 2; pop!(LX_history); end
        while length(F_history) > 2; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang exactly (timesteppers:156-166)
        # RHS = c[1] * F[0] + c[2] * F[1] - a[1] * MX[0] - b[1] * LX[0]
        rhs = c[2] * F_history[1]  # c[1] * F[0]
        if length(F_history) >= 2  # c[2] term
            rhs .+= c[3] * F_history[2]  # c[2] * F[1]
        end
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(LX_history) >= 1  # b[1] term
            rhs .-= b[2] * LX_history[1]  # -b[1] * LX[0]
        end
        
        # Step 6: Build and solve LHS system (following Tarang lines 174-184)
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "CNAB2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "CNAB2 failed: $e, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

# BDF methods
function step_sbdf1!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF1 (backward Euler) following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:224-252 SBDF1 coefficients:
    - a[0] = 1/k0, a[1] = -1/k0 (BDF1 time derivative)
    - b[0] = 1 (fully implicit, not Crank-Nicolson 1/2)
    - c[1] = 1 (forward Euler explicit)
    
    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF1 requires L_matrix and M_matrix, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    # Get SBDF1 coefficients following Tarang exactly (timesteppers:247-250)
    a = [1.0/dt, -1.0/dt]  # a[0], a[1] - BDF1 time derivative
    b = [1.0]              # b[0] - fully implicit (not 1/2 like CNAB)
    c = [0.0, 1.0]         # c[0], c[1] - forward Euler explicit
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Tarang MultistepIMEX pattern)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for SBDF1 (amax=1, bmax=1, cmax=1)
        while length(MX_history) > 1; pop!(MX_history); end
        while length(LX_history) > 1; pop!(LX_history); end
        while length(F_history) > 1; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang MultistepIMEX pattern
        # RHS = c[1] * F[0] - a[1] * MX[0] - 0 * LX[0] (since bmax=1, no b[1] term)
        rhs = c[2] * F_history[1]  # c[1] * F[0]
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        # No b[1] term for SBDF1 since bmax=1
        
        # Step 6: Build and solve LHS system
        # (a[0] * M + b[0] * L).X = RHS  ->  (1/dt * M + 1 * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "SBDF1 step completed: dt=$dt, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF1 failed: $e, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 3
        popfirst!(state.history)
    end
end

function step_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF2 following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:333-367 SBDF2 coefficients:
    - Variable timestep with w1 = k1/k0
    - a[0] = (1 + 2*w1) / (1 + w1) / k1
    - a[1] = -(1 + w1) / k1  
    - a[2] = w1^2 / (1 + w1) / k1
    - b[0] = 1, c[1] = 1 + w1, c[2] = -w1
    - Falls back to SBDF1 for iteration < 1
    
    Implicit: 2nd-order BDF
    Explicit: 2nd-order extrapolation
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    iteration = state.timestepper_data["iteration"]
    
    # Check if we have enough history for SBDF2 (following Tarang line 350)
    if iteration < 1 || length(state.history) < 2
        @debug "SBDF2 requires iteration >= 1, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF2 requires L_matrix and M_matrix, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
    # Get timestep history for variable timestep (following Tarang lines 357-358)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get SBDF2 coefficients following Tarang exactly (timesteppers:360-365)
    a = [(1.0 + 2.0*w1) / (1.0 + w1) / dt_current,  # a[0]
         -(1.0 + w1) / dt_current,                    # a[1]
         w1^2 / (1.0 + w1) / dt_current]              # a[2]
    b = [1.0]                                         # b[0] - fully implicit
    c = [0.0, 1.0 + w1, -w1]                        # c[0], c[1], c[2]
    
    @debug "SBDF2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0]
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for SBDF2 (amax=2, bmax=2, cmax=2)
        while length(MX_history) > 2; pop!(MX_history); end
        while length(LX_history) > 2; pop!(LX_history); end
        while length(F_history) > 2; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang MultistepIMEX pattern
        # RHS = c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - a[2]*MX[1] - 0*LX terms (bmax=1)
        rhs = c[2] * F_history[1]  # c[1] * F[0]
        if length(F_history) >= 2  # c[2] term
            rhs .+= c[3] * F_history[2]  # c[2] * F[1]
        end
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(MX_history) >= 2  # a[2] term
            rhs .-= a[3] * MX_history[2]  # -a[2] * MX[1]
        end
        # No b[1], b[2] terms since bmax=1 for SBDF2
        
        # Step 6: Build and solve LHS system
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "SBDF2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF2 failed: $e, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_sbdf3!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF3 following Tarang implementation.
    
    Tarang coefficients (timesteppers:425-447):
    For iteration >= 2: uses complex 3rd-order BDF coefficients
    For iteration < 2: falls back to SBDF2
    
    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation
    """
    
    # Check if we have enough history for SBDF3
    if length(state.history) < 3
        @debug "SBDF3 requires 3 history states, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    current_state = state.history[end]
    dt = state.dt
    
    # Get timestep history for variable timestep ratios (Tarang pattern)
    if length(state.dt_history) < 3
        @warn "SBDF3 requires 3 timestep history, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    k2 = state.dt_history[end]     # current timestep
    k1 = state.dt_history[end-1]   # previous timestep  
    k0 = state.dt_history[end-2]   # timestep before that
    
    # Compute timestep ratios following Tarang (timesteppers:435-436)
    w2 = k2 / k1
    w1 = k1 / k0
    
    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF3 requires L_matrix and M_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    try
        # Get SBDF3 coefficients following Tarang exactly (timesteppers:438-445)
        a = zeros(4)
        b = zeros(4)
        c = zeros(4)
        
        a[1] = (1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2
        a[2] = (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2
        a[3] = w2^2 * (w1 + 1/(1 + w2)) / k2
        a[4] = -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2
        b[1] = 1
        c[2] = (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1)
        c[3] = -w2*(1 + w1*(1 + w2))
        c[4] = w1*w1*w2*(1 + w2) / (1 + w1)
        
        # Evaluate RHS terms at current and previous times
        # k1 = dt used to step from t_{n-1} to t_n, k0 = dt from t_{n-2} to t_{n-1}
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_prev1 = evaluate_rhs(solver, state.history[end-1], solver.sim_time - k1)
        F_prev2 = evaluate_rhs(solver, state.history[end-2], solver.sim_time - k1 - k0)

        # Convert states to vectors
        X_current = fields_to_vector(current_state)
        X_prev1 = fields_to_vector(state.history[end-1])
        X_prev2 = fields_to_vector(state.history[end-2])
        F_current_vec = fields_to_vector(F_current)
        F_prev1_vec = fields_to_vector(F_prev1)
        F_prev2_vec = fields_to_vector(F_prev2)
        
        # Build RHS following Tarang multistep pattern
        # RHS = sum(cj * F(n-j)) - sum(aj * M.X(n-j)) (j >= 2 for a)
        rhs = (c[2] * F_current_vec + c[3] * F_prev1_vec + c[4] * F_prev2_vec - 
               a[2] * (M_matrix * X_current) - a[3] * (M_matrix * X_prev1) - a[4] * (M_matrix * X_prev2))
        
        # Build and solve LHS system: (a0 M + b0 L).X(n+1) = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix
        
        # Solve linear system
        X_new = LHS_matrix \ rhs
        
        # Convert back to fields and update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        
        @debug "SBDF3 step completed: dt=$k2, w2=$w2, w1=$w1, |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF3 failed: $e, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    # Keep only necessary history for SBDF3
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_sbdf4!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF4 following Tarang implementation.
    
    Tarang coefficients (timesteppers:466-495):
    For iteration >= 3: uses complex 4th-order BDF coefficients  
    For iteration < 3: falls back to SBDF3
    
    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation
    """
    
    # Check if we have enough history for SBDF4
    if length(state.history) < 4
        @debug "SBDF4 requires 4 history states, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    current_state = state.history[end]
    dt = state.dt
    
    # Get timestep history for variable timestep ratios
    if length(state.dt_history) < 4
        @warn "SBDF4 requires 4 timestep history, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    k3 = state.dt_history[end]     # current timestep
    k2 = state.dt_history[end-1]   # previous timestep
    k1 = state.dt_history[end-2]   # timestep before that
    k0 = state.dt_history[end-3]   # timestep 3 back
    
    # Compute timestep ratios following Tarang (timesteppers:476-478)
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0
    
    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF4 requires L_matrix and M_matrix, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    try
        # Get SBDF4 coefficients following Tarang exactly (timesteppers:480-494)
        A1 = 1 + w1*(1 + w2)
        A2 = 1 + w2*(1 + w3) 
        A3 = 1 + w1*A2
        
        a = zeros(5)
        b = zeros(5)
        c = zeros(5)
        
        a[1] = (1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3
        a[2] = (-1 - w3*(1 + w2*(1 + w3)/(1 + w2)*(1 + w1*A2/A1))) / k3
        a[3] = w3 * (w3/(1 + w3) + w2*w3*(A3 + w1)/(1 + w1)) / k3
        a[4] = -w2^3 * w3^2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
        a[5] = (1 + w3) / (1 + w1) * A2 / A1 * w1^4 * w2^3 * w3^2 / A3 / k3
        b[1] = 1
        c[2] = w2 * (1 + w3) / (1 + w2) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2) / A1
        c[3] = -A2 * A3 * w3 / (1 + w1)
        c[4] = w2^2 * w3 * (1 + w3) / (1 + w2) * A3
        c[5] = -w1^3 * w2^2 * w3 * (1 + w3) / (1 + w1) * A2 / A1
        
        # Evaluate RHS terms at current and previous times
        # k2 = dt from t_{n-1} to t_n, k1 = dt from t_{n-2} to t_{n-1}, k0 = dt from t_{n-3} to t_{n-2}
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_prev1 = evaluate_rhs(solver, state.history[end-1], solver.sim_time - k2)
        F_prev2 = evaluate_rhs(solver, state.history[end-2], solver.sim_time - k2 - k1)
        F_prev3 = evaluate_rhs(solver, state.history[end-3], solver.sim_time - k2 - k1 - k0)
        
        # Convert states to vectors
        X_current = fields_to_vector(current_state)
        X_prev1 = fields_to_vector(state.history[end-1])
        X_prev2 = fields_to_vector(state.history[end-2])
        X_prev3 = fields_to_vector(state.history[end-3])
        F_current_vec = fields_to_vector(F_current)
        F_prev1_vec = fields_to_vector(F_prev1)
        F_prev2_vec = fields_to_vector(F_prev2)
        F_prev3_vec = fields_to_vector(F_prev3)
        
        # Build RHS following Tarang multistep pattern
        rhs = (c[2] * F_current_vec + c[3] * F_prev1_vec + c[4] * F_prev2_vec + c[5] * F_prev3_vec - 
               a[2] * (M_matrix * X_current) - a[3] * (M_matrix * X_prev1) - 
               a[4] * (M_matrix * X_prev2) - a[5] * (M_matrix * X_prev3))
        
        # Build and solve LHS system
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix
        
        # Solve linear system
        X_new = LHS_matrix \ rhs
        
        # Convert back to fields and update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        
        @debug "SBDF4 step completed: dt=$k3, w3=$w3, w2=$w2, w1=$w1, |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF4 failed: $e, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    # Keep only necessary history for SBDF4
    if length(state.history) > 5
        popfirst!(state.history)
    end
end

# Exponential Time Differencing methods
