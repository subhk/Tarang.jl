"""
    Crank-Nicolson Adams-Bashforth 1st order following Tarang MultistepIMEX implementation.

    Based on Tarang timesteppers:95-188 MultistepIMEX.step method:
    - Proper coefficient computation: a[0] = 1/dt, a[1] = -1/dt, b[0] = 1/2, b[1] = 1/2, c[1] = 1
    - RHS construction: c[1]*F[0] - a[1]*MX[0] - b[1]*LX[0] (following lines 156-166)
    - LHS solution: (a[0]*M + b[0]*L).X = RHS (following lines 174-184)
    - Proper state rotation and history management
    """

function _init_global_multistep_history!(state::TimestepperState, iteration_key::Symbol)
    if !haskey(state.timestepper_data, :MX_history)
        state.timestepper_data[:MX_history] = Vector{ComplexF64}[]
        state.timestepper_data[:LX_history] = Vector{ComplexF64}[]
        state.timestepper_data[:F_history] = Vector{ComplexF64}[]
        state.timestepper_data[iteration_key] = 0
    elseif !haskey(state.timestepper_data, iteration_key)
        state.timestepper_data[iteration_key] = 0
    end
end

function _global_multistep_state_vector!(state::TimestepperState, fields::Vector{<:ScalarField})
    _ensure_coeff_layout!(fields)
    vector = _timestep_vector_buffer!(state, :multistep_X_current_vec, _fields_vector_size(fields))
    return fields_to_vector!(vector, fields)
end

function _global_multistep_fields_vector!(state::TimestepperState, key::Symbol,
                                          fields::Vector{<:ScalarField})
    _ensure_coeff_layout!(fields)
    vector = _timestep_vector_buffer!(state, key, _fields_vector_size(fields))
    return fields_to_vector!(vector, fields)
end

function _global_multistep_matvec!(state::TimestepperState, key::Symbol,
                                   matrix::AbstractMatrix, vector::AbstractVector{ComplexF64})
    dest = _timestep_vector_buffer!(state, key, size(matrix, 1))
    mul!(dest, matrix, vector)
    return dest
end

function _prepend_history_buffer!(history::Vector{Vector{ComplexF64}},
                                  scratch::Vector{ComplexF64}, max_len::Int)
    max_len <= 0 && return history

    if length(history) >= max_len
        slot = pop!(history)
        if length(slot) != length(scratch)
            slot = Vector{ComplexF64}(undef, length(scratch))
        end
    else
        slot = Vector{ComplexF64}(undef, length(scratch))
    end

    copyto!(slot, scratch)
    pushfirst!(history, slot)
    return history
end

function _global_multistep_zero_rhs!(state::TimestepperState, n::Int)
    rhs = _timestep_vector_buffer!(state, :multistep_rhs_vec, n)
    fill!(rhs, zero(ComplexF64))
    return rhs
end

function _global_multistep_solve!(state::TimestepperState, cache_key,
                                  M_matrix::AbstractMatrix, L_matrix::AbstractMatrix,
                                  a0, b0, rhs::Vector{ComplexF64})
    if !haskey(state.timestepper_data, :lhs_cache) ||
       get(state.timestepper_data, :lhs_cache_key, nothing) != cache_key
        LHS = a0 * M_matrix + b0 * L_matrix
        state.timestepper_data[:lhs_cache] = factorize(LHS)
        state.timestepper_data[:lhs_cache_key] = cache_key
    end

    lhs_solver = state.timestepper_data[:lhs_cache]
    solution = _timestep_vector_buffer!(state, :multistep_X_new_vec, length(rhs))
    try
        ldiv!(solution, lhs_solver, rhs)
    catch e
        isa(e, MethodError) || rethrow(e)
        copyto!(solution, lhs_solver \ rhs)
    end
    return solution
end

function step_cnab1!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path: use the per-Fourier-mode multistep stepper when available.
    # This is the only path that correctly handles inhomogeneous algebraic
    # constraints (BCs like `T(z=0) = 1`), because the global-matrix path below
    # packs F in variable space and silently drops BC F values.
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            a, b, c = _cnab1_coefs(dt)
            step_subproblem_multistep!(state, solver, sps, a, b, c)
            return
        end
    end

    # Check for MPI mode - multistep IMEX methods don't support distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            # Multistep IMEX requires global matrix solves which don't work with distributed PencilArrays
            step_rk222!(state, solver)
            return
        end
    end

    # Initialize history arrays if needed (following Tarang MultistepIMEX.__init__)
    _init_global_multistep_history!(state, :cnab1_iteration)

    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNAB1 requires L_matrix and M_matrix, falling back to forward Euler" maxlog=1
        step_rk111!(state, solver)
        return
    end

    # Get CNAB1 coefficients following Tarang (timesteppers:206-220)
    # Using tuples to avoid heap allocation every step
    a = (1.0/dt, -1.0/dt)  # a[0], a[1]
    b = (0.5, 0.5)         # b[0], b[1]
    c = (0.0, 1.0)         # c[0], c[1]
    
    # Step 1: Convert current state to vector (following Tarang gather_inputs)
    X_current = _global_multistep_state_vector!(state, current_state)

    # Step 2: Compute M.X[0] and L.X[0] (following Tarang lines 142-147)
    MX_current = _global_multistep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)
    LX_current = _global_multistep_matvec!(state, :multistep_LX_current_vec, L_matrix, X_current)

    # Step 3: Evaluate F(X[0]) at current time step (following Tarang lines 149-153)
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _global_multistep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: Rotate and store history (following Tarang lines 124-126)
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    LX_history = state.timestepper_data[:LX_history]::Vector{Vector{ComplexF64}}
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

    _prepend_history_buffer!(MX_history, MX_current, 1)
    _prepend_history_buffer!(LX_history, LX_current, 1)
    _prepend_history_buffer!(F_history, F_current_vec, 2)

    # Step 5: Build RHS following Tarang exactly (timesteppers:156-166)
    # RHS = c[1] * F[0] - a[1] * MX[0] - b[1] * LX[0]
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    @. rhs += c[2] * F_history[1]  # c[1] * F[0] (using 1-based indexing)
    if length(MX_history) >= 1  # a[1] term
        @. rhs -= a[2] * MX_history[1]  # -a[1] * MX[0]
    end
    if length(LX_history) >= 1  # b[1] term
        @. rhs -= b[2] * LX_history[1]  # -b[1] * LX[0]
    end

    # Step 6: Build and solve LHS system (following Tarang lines 174-184)
    # (a[0] * M + b[0] * L).X = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: Update state
    _push_vector_state!(state.history, X_new, current_state, 3)
    state.timestepper_data[:cnab1_iteration] += 1

    @debug "CNAB1 step completed: dt=$dt, iteration=$(state.timestepper_data[:cnab1_iteration]), |X_new|=$(norm(X_new))"
end

"""
    Crank-Nicolson Adams-Bashforth 2nd order following Tarang MultistepIMEX implementation.

    Based on Tarang timesteppers:95-188 MultistepIMEX.step method:
    - Variable timestep coefficients: w1 = k1/k0, c[1] = 1 + w1/2, c[2] = -w1/2 (lines 276-290)
    - Full RHS construction: c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - b[1]*LX[0] (lines 156-166)
    - Proper history management with rotation for MX, LX, F arrays (lines 124-126)
    - Falls back to CNAB1 for iteration < 1 (line 274)
    """
function step_cnab2!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path handles inhomogeneous BCs correctly (see step_cnab1!).
    # CNAB2 needs 1 prior F-history entry to start; fall back to CNAB1 when
    # the history is empty (first call after solver build).
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            if _sp_multistep_history_depth(state) < 1
                step_cnab1!(state, solver)
                return
            end
            dt_prev = get_previous_timestep(state)
            a, b, c = _cnab2_coefs(dt, dt_prev)
            step_subproblem_multistep!(state, solver, sps, a, b, c)
            return
        end
    end

    # Check for MPI mode - multistep IMEX methods don't support distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            step_rk222!(state, solver)
            return
        end
    end

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :cnab2_iteration)

    iteration = state.timestepper_data[:cnab2_iteration]

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
        @warn "CNAB2 requires L_matrix and M_matrix, falling back to CNAB1" maxlog=1
        step_cnab1!(state, solver)
        return
    end
    
    # Get timestep history for variable timestep (following Tarang lines 280-281)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get CNAB2 coefficients following Tarang exactly (timesteppers:283-288)
    a = (1.0/dt_current, -1.0/dt_current)  # a[0], a[1]
    b = (0.5, 0.5)                         # b[0], b[1]
    c = (0.0, 1.0 + w1/2.0, -w1/2.0)      # c[0], c[1], c[2]
    
    @debug "CNAB2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    # Step 1: Convert current state to vector
    X_current = _global_multistep_state_vector!(state, current_state)

    # Step 2: Compute M.X[0] and L.X[0] (following Tarang lines 142-147)
    MX_current = _global_multistep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)
    LX_current = _global_multistep_matvec!(state, :multistep_LX_current_vec, L_matrix, X_current)

    # Step 3: Evaluate F(X[0]) at current time step (following Tarang lines 149-153)
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _global_multistep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: Rotate and store history (following Tarang lines 124-126)
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    LX_history = state.timestepper_data[:LX_history]::Vector{Vector{ComplexF64}}
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

    _prepend_history_buffer!(MX_history, MX_current, 1)
    _prepend_history_buffer!(LX_history, LX_current, 1)
    _prepend_history_buffer!(F_history, F_current_vec, 2)

    # Step 5: Build RHS following Tarang exactly (timesteppers:156-166)
    # RHS = c[1] * F[0] + c[2] * F[1] - a[1] * MX[0] - b[1] * LX[0]
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    @. rhs += c[2] * F_history[1]  # c[1] * F[0]
    if length(F_history) >= 2  # c[2] term (Adams-Bashforth 2 extrapolation)
        @. rhs += c[3] * F_history[2]  # c[2] * F[1]
    else
        @warn "CNAB2: insufficient F_history ($(length(F_history)) < 2), falling back to first-order extrapolation" maxlog=1
    end
    if length(MX_history) >= 1  # a[1] term
        @. rhs -= a[2] * MX_history[1]  # -a[1] * MX[0]
    end
    if length(LX_history) >= 1  # b[1] term
        @. rhs -= b[2] * LX_history[1]  # -b[1] * LX[0]
    end

    # Step 6: Build and solve LHS system (following Tarang lines 174-184)
    # (a[0] * M + b[0] * L).X = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: Update state
    _push_vector_state!(state.history, X_new, current_state, 4)
    state.timestepper_data[:cnab2_iteration] += 1

    @debug "CNAB2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data[:cnab2_iteration]), |X_new|=$(norm(X_new))"
end

# BDF methods
"""
    Semi-implicit BDF1 (backward Euler) following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:224-252 SBDF1 coefficients:
    - a[0] = 1/k0, a[1] = -1/k0 (BDF1 time derivative)
    - b[0] = 1 (fully implicit, not Crank-Nicolson 1/2)
    - c[1] = 1 (forward Euler explicit)
    
    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)
    """
function step_sbdf1!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path handles inhomogeneous BCs correctly (see step_cnab1!).
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            a, b, c = _sbdf1_coefs(dt)
            step_subproblem_multistep!(state, solver, sps, a, b, c)
            return
        end
    end

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf1_iteration)

    # Check for MPI mode - SBDF methods don't support distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            # SBDF requires global matrix solves which don't work with distributed PencilArrays
            # Fall back to explicit RK which works with distributed data
            step_rk222!(state, solver)
            return
        end
    end

    # Get matrices from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF1 requires L_matrix and M_matrix, falling back to forward Euler" maxlog=1
        step_rk111!(state, solver)
        return
    end
    
    # Get SBDF1 coefficients following Tarang exactly (timesteppers:247-250)
    a = (1.0/dt, -1.0/dt)  # a[0], a[1] - BDF1 time derivative
    b = (1.0,)             # b[0] - fully implicit (not 1/2 like CNAB)
    c = (0.0, 1.0)         # c[0], c[1] - forward Euler explicit
    
    # Step 1: Convert current state to vector
    X_current = _global_multistep_state_vector!(state, current_state)

    # Step 2: Compute M.X[0] and L.X[0] (following Tarang MultistepIMEX pattern)
    MX_current = _global_multistep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)
    LX_current = _global_multistep_matvec!(state, :multistep_LX_current_vec, L_matrix, X_current)

    # Step 3: Evaluate F(X[0]) at current time step
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _global_multistep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: Rotate and store history
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    LX_history = state.timestepper_data[:LX_history]::Vector{Vector{ComplexF64}}
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

    _prepend_history_buffer!(MX_history, MX_current, 1)
    _prepend_history_buffer!(LX_history, LX_current, 1)
    _prepend_history_buffer!(F_history, F_current_vec, 1)

    # Step 5: Build RHS following Tarang MultistepIMEX pattern
    # RHS = c[1] * F[0] - a[1] * MX[0] - 0 * LX[0] (since bmax=1, no b[1] term)
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    @. rhs += c[2] * F_history[1]  # c[1] * F[0]
    if length(MX_history) >= 1  # a[1] term
        @. rhs -= a[2] * MX_history[1]  # -a[1] * MX[0]
    end
    # No b[1] term for SBDF1 since bmax=1

    # Step 6: Build and solve LHS system
    # (a[0] * M + b[0] * L).X = RHS  ->  (1/dt * M + 1 * L).X = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: Update state
    _push_vector_state!(state.history, X_new, current_state, 3)
    state.timestepper_data[:sbdf1_iteration] += 1

    @debug "SBDF1 step completed: dt=$dt, iteration=$(state.timestepper_data[:sbdf1_iteration]), |X_new|=$(norm(X_new))"
end

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
function step_sbdf2!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path handles inhomogeneous BCs correctly (see step_cnab1!).
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            if _sp_multistep_history_depth(state) < 1
                step_sbdf1!(state, solver)
                return
            end
            a, b, c = _sbdf2_coefs(dt)
            step_subproblem_multistep!(state, solver, sps, a, b, c)
            return
        end
    end

    # Check for MPI mode - SBDF methods don't support distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            # SBDF requires global matrix solves which don't work with distributed PencilArrays
            step_rk222!(state, solver)
            return
        end
    end

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf2_iteration)

    iteration = state.timestepper_data[:sbdf2_iteration]

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
        @warn "SBDF2 requires L_matrix and M_matrix, falling back to SBDF1" maxlog=1
        step_sbdf1!(state, solver)
        return
    end
    
    # Get timestep history for variable timestep (following Tarang lines 357-358)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get SBDF2 coefficients following Tarang exactly (timesteppers:360-365)
    a = ((1.0 + 2.0*w1) / (1.0 + w1) / dt_current,  # a[0]
         -(1.0 + w1) / dt_current,                    # a[1]
         w1^2 / (1.0 + w1) / dt_current)              # a[2]
    b = (1.0,)                                        # b[0] - fully implicit
    c = (0.0, 1.0 + w1, -w1)                         # c[0], c[1], c[2]
    
    @debug "SBDF2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    # Step 1: Convert current state to vector
    X_current = _global_multistep_state_vector!(state, current_state)

    # Step 2: Compute M.X[0] and L.X[0]
    MX_current = _global_multistep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)
    LX_current = _global_multistep_matvec!(state, :multistep_LX_current_vec, L_matrix, X_current)

    # Step 3: Evaluate F(X[0]) at current time step
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _global_multistep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: Rotate and store history
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    LX_history = state.timestepper_data[:LX_history]::Vector{Vector{ComplexF64}}
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

    _prepend_history_buffer!(MX_history, MX_current, 2)
    _prepend_history_buffer!(LX_history, LX_current, 2)
    _prepend_history_buffer!(F_history, F_current_vec, 2)

    # Step 5: Build RHS following Tarang MultistepIMEX pattern
    # RHS = c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - a[2]*MX[1] - 0*LX terms (bmax=1)
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    @. rhs += c[2] * F_history[1]  # c[1] * F[0]
    if length(F_history) >= 2  # c[2] term
        @. rhs += c[3] * F_history[2]  # c[2] * F[1]
    end
    if length(MX_history) >= 1  # a[1] term
        @. rhs -= a[2] * MX_history[1]  # -a[1] * MX[0]
    end
    if length(MX_history) >= 2  # a[2] term
        @. rhs -= a[3] * MX_history[2]  # -a[2] * MX[1]
    end
    # No b[1], b[2] terms since bmax=1 for SBDF2

    # Step 6: Build and solve LHS system
    # (a[0] * M + b[0] * L).X = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: Update state
    _push_vector_state!(state.history, X_new, current_state, 4)
    state.timestepper_data[:sbdf2_iteration] += 1

    @debug "SBDF2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data[:sbdf2_iteration]), |X_new|=$(norm(X_new))"
end

"""
    Semi-implicit BDF3 following Tarang implementation.

    Tarang coefficients (timesteppers:425-447):
    For iteration >= 2: uses complex 3rd-order BDF coefficients
    For iteration < 2: falls back to SBDF2

    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation
    """
function step_sbdf3!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path handles inhomogeneous BCs correctly (see step_cnab1!).
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            if _sp_multistep_history_depth(state) < 2
                step_sbdf2!(state, solver)
                return
            end
            a, b, c = _sbdf3_coefs(dt)
            step_subproblem_multistep!(state, solver, sps, a, b, c)
            return
        end
    end

    # Check for MPI mode - SBDF methods don't support distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            step_rk222!(state, solver)
            return
        end
    end

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf3_iteration)

    iteration = state.timestepper_data[:sbdf3_iteration]

    # Check if we have enough history for SBDF3
    if iteration < 2 || length(state.history) < 3
        @debug "SBDF3 requires iteration >= 2, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

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

    # Get SBDF3 coefficients following Tarang exactly (timesteppers:438-445)
    a = ((1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2,
         (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2,
         w2^2 * (w1 + 1/(1 + w2)) / k2,
         -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2)
    b = (1.0, 0.0, 0.0, 0.0)
    c = (0.0,
         (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1),
         -w2*(1 + w1*(1 + w2)),
         w1*w1*w2*(1 + w2) / (1 + w1))

    # Step 1: Convert current state to vector
    X_current = _global_multistep_state_vector!(state, current_state)

    # Step 2: Compute M.X[0]
    MX_current = _global_multistep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)

    # Step 3: Evaluate F(X[0]) at current time step (only 1 RHS eval, reuse history)
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _global_multistep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: Rotate and store history (following SBDF1/SBDF2 pattern)
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

    _prepend_history_buffer!(MX_history, MX_current, 3)
    _prepend_history_buffer!(F_history, F_current_vec, 3)

    # Step 5: Build RHS following Tarang multistep pattern
    # RHS = c[1]*F[0] + c[2]*F[1] + c[3]*F[2] - a[1]*MX[0] - a[2]*MX[1] - a[3]*MX[2]
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    @. rhs += c[2] * F_history[1]  # c[1] * F[0]
    if length(F_history) >= 2
        @. rhs += c[3] * F_history[2]  # c[2] * F[1]
    end
    if length(F_history) >= 3
        @. rhs += c[4] * F_history[3]  # c[3] * F[2]
    end
    if length(MX_history) >= 1  # a[1] term
        @. rhs -= a[2] * MX_history[1]  # -a[1] * MX[0]
    end
    if length(MX_history) >= 2  # a[2] term
        @. rhs -= a[3] * MX_history[2]  # -a[2] * MX[1]
    end
    if length(MX_history) >= 3  # a[3] term
        @. rhs -= a[4] * MX_history[3]  # -a[3] * MX[2]
    end

    # Step 6: Build and solve LHS system: (a[0]*M + b[0]*L).X(n+1) = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: Update state
    _push_vector_state!(state.history, X_new, current_state, 4)
    state.timestepper_data[:sbdf3_iteration] += 1

    @debug "SBDF3 step completed: dt=$k2, w2=$w2, w1=$w1, |X_new|=$(norm(X_new))"
end

"""
    Semi-implicit BDF4 following Tarang implementation.

    Tarang coefficients (timesteppers:466-495):
    For iteration >= 3: uses complex 4th-order BDF coefficients
    For iteration < 3: falls back to SBDF3

    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation
    """
function step_sbdf4!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path handles inhomogeneous BCs correctly (see step_cnab1!).
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            if _sp_multistep_history_depth(state) < 3
                step_sbdf3!(state, solver)
                return
            end
            a, b, c = _sbdf4_coefs(dt)
            step_subproblem_multistep!(state, solver, sps, a, b, c)
            return
        end
    end

    # Check for MPI mode - SBDF methods don't support distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            step_rk222!(state, solver)
            return
        end
    end

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf4_iteration)

    iteration = state.timestepper_data[:sbdf4_iteration]

    # Check if we have enough history for SBDF4
    if iteration < 3 || length(state.history) < 4
        @debug "SBDF4 requires iteration >= 3, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end

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

    # Get SBDF4 coefficients following Tarang exactly (timesteppers:480-494)
    A1 = 1 + w1*(1 + w2)
    A2 = 1 + w2*(1 + w3)
    A3 = 1 + w1*A2

    a = ((1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3,
         (-1 - w3*(1 + (w2*(1 + w3)/(1 + w2)) * (1 + w1*A2/A1))) / k3,
         w3 * (w3/(1 + w3) + (w2*w3*(A3 + w1))/(1 + w1)) / k3,
         -(w2^3 * w3^2 * (1 + w3) * A3) / ((1 + w2) * A2 * k3),
         ((1 + w3) * A2 * w1^4 * w2^3 * w3^2) / ((1 + w1) * A1 * A3 * k3))
    b = (1.0, 0.0, 0.0, 0.0, 0.0)
    c = (0.0,
         (w2 * (1 + w3) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2)) / ((1 + w2) * A1),
         -(A2 * A3 * w3) / (1 + w1),
         (w2^2 * w3 * (1 + w3) * A3) / (1 + w2),
         -(w1^3 * w2^2 * w3 * (1 + w3) * A2) / ((1 + w1) * A1))

    # Step 1: Convert current state to vector
    X_current = _global_multistep_state_vector!(state, current_state)

    # Step 2: Compute M.X[0]
    MX_current = _global_multistep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)

    # Step 3: Evaluate F(X[0]) at current time step (only 1 RHS eval, reuse history)
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _global_multistep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: Rotate and store history (following SBDF1/SBDF2/SBDF3 pattern)
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

    _prepend_history_buffer!(MX_history, MX_current, 4)
    _prepend_history_buffer!(F_history, F_current_vec, 4)

    # Step 5: Build RHS following Tarang multistep pattern
    # RHS = c[1]*F[0] + c[2]*F[1] + c[3]*F[2] + c[4]*F[3]
    #     - a[1]*MX[0] - a[2]*MX[1] - a[3]*MX[2] - a[4]*MX[3]
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    @. rhs += c[2] * F_history[1]  # c[1] * F[0]
    if length(F_history) >= 2
        @. rhs += c[3] * F_history[2]  # c[2] * F[1]
    end
    if length(F_history) >= 3
        @. rhs += c[4] * F_history[3]  # c[3] * F[2]
    end
    if length(F_history) >= 4
        @. rhs += c[5] * F_history[4]  # c[4] * F[3]
    end
    if length(MX_history) >= 1  # a[1] term
        @. rhs -= a[2] * MX_history[1]  # -a[1] * MX[0]
    end
    if length(MX_history) >= 2  # a[2] term
        @. rhs -= a[3] * MX_history[2]  # -a[2] * MX[1]
    end
    if length(MX_history) >= 3  # a[3] term
        @. rhs -= a[4] * MX_history[3]  # -a[3] * MX[2]
    end
    if length(MX_history) >= 4  # a[4] term
        @. rhs -= a[5] * MX_history[4]  # -a[4] * MX[3]
    end

    # Step 6: Build and solve LHS system: (a[0]*M + b[0]*L).X(n+1) = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: Update state
    _push_vector_state!(state.history, X_new, current_state, 5)
    state.timestepper_data[:sbdf4_iteration] += 1

    @debug "SBDF4 step completed: dt=$k3, w3=$w3, w2=$w2, w1=$w1, |X_new|=$(norm(X_new))"
end

# Exponential Time Differencing methods
