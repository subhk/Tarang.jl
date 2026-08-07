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

    solution = _timestep_vector_buffer!(state, :multistep_X_new_vec, length(rhs))
    _timestep_ldiv!(solution, state.timestepper_data[:lhs_cache], rhs)
    return solution
end

function _global_multistep_distributed_fallback!(state::TimestepperState,
                                                 solver::InitialValueSolver,
                                                 current_state::Vector{<:ScalarField},
                                                 method_name::String)
    reason = _global_matrix_implicit_distributed_fallback_reason(current_state)
    reason === nothing && return false

    throw(ArgumentError(
        "$method_name has no distributed diagonal-IMEX implementation for MPI " *
        "pure-Fourier problems. Refusing to drop the implicit linear operator " *
        "and take an explicit step, which can be unstable for stiff systems. " *
        "On CPU-MPI use SBDF2 or a distributed diagonal-IMEX Runge-Kutta method. " *
        "On GPU the distributed diagonal path also declines — run single-GPU " *
        "(DiagonalIMEX_RK222/RK443/SBDF2) or CPU-MPI instead."))
end

function _prepare_global_multistep_matrices!(state::TimestepperState,
                                             solver::InitialValueSolver,
                                             method_name::String,
                                             fallback_name::String,
                                             fallback_step!::F) where {F}
    L_matrix, M_matrix = _global_matrix_implicit_matrices(solver)
    reason = _global_matrix_implicit_missing_matrix_reason(L_matrix, M_matrix)
    if reason !== nothing
        _log_global_matrix_implicit_matrix_fallback(method_name, reason, fallback_name)
        fallback_step!(state, solver)
        return L_matrix, M_matrix, true
    end

    return L_matrix, M_matrix, false
end

"""Accumulate `sign * coefs[i] * history[i-1]` for every term the history can supply.

`coefs` is 1-based, so `coefs[1]` is the implicit `a[0]`/`b[0]` handled by the LHS
and only `coefs[2:end]` contribute here. A scheme running below its nominal order
(startup, or after a history reset) simply has fewer entries and drops the tail —
the same per-term `length(history) >= k` guards the six schemes each wrote out by
hand, now written once."""
@inline function _accumulate_history_terms!(rhs::Vector{ComplexF64}, coefs::Tuple,
                                            history::Vector{Vector{ComplexF64}}, sign::Float64)
    @inbounds for i in 2:length(coefs)
        (i - 1) <= length(history) || continue
        coef = sign * coefs[i]
        h = history[i - 1]
        @. rhs += coef * h
    end
    return rhs
end

"""
    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c; ...)

The global-matrix MultistepIMEX update shared by CNAB1/2 and SBDF1-4:

    (a[0] M + b[0] L) X^{n+1} = Σ c[i] F[i-1] − Σ a[i] MX[i-1] − Σ b[i] LX[i-1]

(0-based in the formula, 1-based in the tuples.) The schemes differ ONLY in the
coefficients, the history depths and the state-history cap — the seven numbered
steps below were six near-identical copies.

That duplication was not free. It is how `step_sbdf2!` came to be the only scheme
carrying the distributed diagonal-IMEX branch, and how `step_sbdf1!` came to be
the only one initialising its history before rather than after the explicit-field
probe: differences that mattered and differences that did not looked identical in
the source. Both remaining asymmetries now live in the prologues, which are short
enough to compare at a glance.

`use_lx=false` (SBDF3/4) skips the `L·X` product AND the `LX_history` rotation
entirely, as those schemes already did — their `b` is `(1, 0, 0, …)`, so the terms
would be zero, but the matvec would not be.
"""
function _global_multistep_core!(state::TimestepperState, solver::InitialValueSolver,
                                 current_state::Vector{<:ScalarField},
                                 L_matrix::AbstractMatrix, M_matrix::AbstractMatrix,
                                 a::Tuple, b::Tuple, c::Tuple;
                                 mx_depth::Int, f_depth::Int, state_cap::Int,
                                 iter_key::Symbol, name::String,
                                 use_lx::Bool=true, lx_depth::Int=mx_depth,
                                 warn_short_f::Bool=false)
    # Step 1: current state as a vector
    X_current = _timestep_fields_vector!(state, :multistep_X_current_vec, current_state)

    # Step 2: M.X[0] and (when the scheme uses it) L.X[0]
    MX_current = _timestep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)

    # Step 3: F(X[0]) at the current time
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    F_current_vec = _timestep_fields_vector!(state, :multistep_F_current_vec, F_current)

    # Step 4: rotate and store history
    MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
    F_history  = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}
    _prepend_history_buffer!(MX_history, MX_current, mx_depth)
    _prepend_history_buffer!(F_history, F_current_vec, f_depth)

    LX_history = state.timestepper_data[:LX_history]::Vector{Vector{ComplexF64}}
    if use_lx
        LX_current = _timestep_matvec!(state, :multistep_LX_current_vec, L_matrix, X_current)
        _prepend_history_buffer!(LX_history, LX_current, lx_depth)
    end

    if warn_short_f && length(F_history) < length(c) - 1
        @warn "$name: insufficient F_history ($(length(F_history)) < $(length(c) - 1)), " *
              "falling back to lower-order extrapolation" maxlog=1
    end

    # Step 5: RHS
    rhs = _global_multistep_zero_rhs!(state, length(X_current))
    _accumulate_history_terms!(rhs, c, F_history, 1.0)
    _accumulate_history_terms!(rhs, a, MX_history, -1.0)
    use_lx && _accumulate_history_terms!(rhs, b, LX_history, -1.0)

    # Step 6: solve (a[0] M + b[0] L) X = RHS
    cache_key = (a[1], b[1])
    X_new = _global_multistep_solve!(state, cache_key, M_matrix, L_matrix, a[1], b[1], rhs)

    # Step 7: update state
    _push_vector_state!(state.history, X_new, current_state, state_cap)
    state.timestepper_data[iter_key] += 1

    @debug "$name step completed: dt=$(state.dt), iteration=$(state.timestepper_data[iter_key]), |X_new|=$(norm(X_new))"
    return X_new
end

function step_cnab1!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path: use the per-Fourier-mode multistep stepper when available.
    # This is the only path that correctly handles inhomogeneous algebraic
    # constraints (BCs like `T(z=0) = 1`), because the global-matrix path below
    # packs F in variable space and silently drops BC F values.
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        a, b, c = _cnab1_coefs(dt)
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    # GPU / MPI without subproblems: no global matrix exists, so an explicit
    # problem takes the matrix-free field combination (see step_multistep_field.jl).
    _try_step_explicit_multistep_field!(state, solver, :cnab1) && return

    _global_multistep_distributed_fallback!(state, solver, current_state, "CNAB1") && return

    # Initialize history arrays if needed (following Tarang MultistepIMEX.__init__)
    _init_global_multistep_history!(state, :cnab1_iteration)

    L_matrix, M_matrix, fell_back =
        _prepare_global_multistep_matrices!(state, solver, "CNAB1", "forward Euler", step_rk111!)
    fell_back && return

    # Get CNAB1 coefficients following Tarang (timesteppers:206-220)
    # Using tuples to avoid heap allocation every step
    a = (1.0/dt, -1.0/dt)  # a[0], a[1]
    b = (0.5, 0.5)         # b[0], b[1]
    c = (0.0, 1.0)         # c[0], c[1]

    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c;
                            mx_depth=1, lx_depth=1, f_depth=2, state_cap=3,
                            iter_key=:cnab1_iteration, name="CNAB1")
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
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        if _sp_multistep_history_depth(state) < 1
            step_cnab1!(state, solver)
            return
        end
        dt_prev = get_previous_timestep(state)
        a, b, c = _cnab2_coefs(dt, dt_prev)
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    _try_step_explicit_multistep_field!(state, solver, :cnab2) && return

    _global_multistep_distributed_fallback!(state, solver, current_state, "CNAB2") && return

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :cnab2_iteration)

    iteration = state.timestepper_data[:cnab2_iteration]

    # Check if we have enough history for CNAB2 (following Tarang line 274)
    if iteration < 1 || length(state.history) < 2
        @debug "CNAB2 requires iteration >= 1, falling back to CNAB1"
        step_cnab1!(state, solver)
        state.timestepper_data[:cnab2_iteration] += 1
        return
    end

    L_matrix, M_matrix, fell_back =
        _prepare_global_multistep_matrices!(state, solver, "CNAB2", "CNAB1", step_cnab1!)
    fell_back && return
    
    # Get timestep history for variable timestep (following Tarang lines 280-281)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get CNAB2 coefficients following Tarang exactly (timesteppers:283-288)
    a = (1.0/dt_current, -1.0/dt_current)  # a[0], a[1]
    b = (0.5, 0.5)                         # b[0], b[1]
    c = (0.0, 1.0 + w1/2.0, -w1/2.0)      # c[0], c[1], c[2]
    
    @debug "CNAB2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"

    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c;
                            mx_depth=1, lx_depth=1, f_depth=2, state_cap=4,
                            iter_key=:cnab2_iteration, name="CNAB2", warn_short_f=true)
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
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        a, b, c = _sbdf1_coefs(dt)
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    _try_step_explicit_multistep_field!(state, solver, :sbdf1) && return

    _global_multistep_distributed_fallback!(state, solver, current_state, "SBDF1") && return

    # Initialize history arrays if needed. This used to sit ABOVE the explicit-field
    # probe — SBDF1 was the only one of the six that did — which allocated the
    # global-matrix history deques on a path that never reads them. Nothing outside
    # this file reads `:sbdf1_iteration`.
    _init_global_multistep_history!(state, :sbdf1_iteration)

    L_matrix, M_matrix, fell_back =
        _prepare_global_multistep_matrices!(state, solver, "SBDF1", "forward Euler", step_rk111!)
    fell_back && return
    
    # Get SBDF1 coefficients following Tarang exactly (timesteppers:247-250)
    a = (1.0/dt, -1.0/dt)  # a[0], a[1] - BDF1 time derivative
    b = (1.0,)             # b[0] - fully implicit (not 1/2 like CNAB); bmax=1, so no b[1] term
    c = (0.0, 1.0)         # c[0], c[1] - forward Euler explicit

    # `use_lx` stays true even though `b` contributes no history term: SBDF1 has
    # always kept `LX_history` rotating, and CNAB1/2 — which SBDF-family fallbacks
    # can reach — read it.
    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c;
                            mx_depth=1, lx_depth=1, f_depth=1, state_cap=3,
                            iter_key=:sbdf1_iteration, name="SBDF1")
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
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        if _sp_multistep_history_depth(state) < 1
            step_sbdf1!(state, solver)
            return
        end
        dt_prev = get_previous_timestep(state)
        a, b, c = _sbdf2_coefs(dt, dt_prev)
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    # MPI pure-Fourier: no subproblems are built (those are Fourier+Chebyshev
    # only), and the global-matrix implicit solve can't run distributed. Use the
    # distributed diagonal IMEX path instead of an explicit fallback that would
    # mishandle stiff implicit linear terms (e.g. hyperviscosity).
    if _distributed_diagonal_imex_applicable(solver)
        step_distributed_diagonal_imex_sbdf2!(state, solver)
        return
    end

    # After the distributed diagonal-IMEX branch above: that path already serves
    # MPI pure-Fourier (implicit L included), so it keeps precedence there. This
    # reaches GPU, where it declines.
    _try_step_explicit_multistep_field!(state, solver, :sbdf2) && return

    _global_multistep_distributed_fallback!(state, solver, current_state, "SBDF2") && return

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf2_iteration)

    iteration = state.timestepper_data[:sbdf2_iteration]

    # Check if we have enough history for SBDF2 (following Tarang line 350)
    if iteration < 1 || length(state.history) < 2
        @debug "SBDF2 requires iteration >= 1, falling back to SBDF1"
        step_sbdf1!(state, solver)
        state.timestepper_data[:sbdf2_iteration] += 1
        return
    end

    L_matrix, M_matrix, fell_back =
        _prepare_global_multistep_matrices!(state, solver, "SBDF2", "SBDF1", step_sbdf1!)
    fell_back && return

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

    # bmax=1, so `b` contributes no history term; LX_history keeps rotating as it
    # always has (see step_sbdf1!).
    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c;
                            mx_depth=2, lx_depth=2, f_depth=2, state_cap=4,
                            iter_key=:sbdf2_iteration, name="SBDF2")
end

"""
RK443-seeded startup for the global (non-subproblem) multistep methods: record
the M·X and F history at the current state, then advance the state with an
order-3 IMEX RK step. A high-order self-start (instead of SBDF1/SBDF2) keeps the
one-time startup error from capping the multistep's global convergence order.
"""
function _multistep_rk443_startup!(state::TimestepperState, solver::InitialValueSolver,
                                   depth::Int, iter_key::Symbol)
    current_state = state.history[end]
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix !== nothing && M_matrix !== nothing
        X_current = _timestep_fields_vector!(state, :multistep_X_current_vec, current_state)
        MX_current = _timestep_matvec!(state, :multistep_MX_current_vec, M_matrix, X_current)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = _timestep_fields_vector!(state, :multistep_F_current_vec, F_current)
        MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
        F_history = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}
        _prepend_history_buffer!(MX_history, MX_current, depth)
        _prepend_history_buffer!(F_history, F_current_vec, depth)
    end
    # Advance the state with an order-3 IMEX RK step (RK443 tableau passed
    # explicitly; `state.timestepper` is parametric on the multistep type and
    # cannot be reassigned).
    step_rk_imex!(state, solver; ts=_RK443_SINGLETON)
    state.timestepper_data[iter_key] += 1
    return
end

"""Semi-implicit BDF3: 3rd-order BDF (implicit) + 3rd-order extrapolation (explicit).
RK443-seeded startup for the first 2 steps; full SBDF3 thereafter."""
function step_sbdf3!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    # Subproblem path handles inhomogeneous BCs correctly (see step_cnab1!).
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        if _sp_multistep_history_depth(state) < 2 || length(state.dt_history) < 3
            _seed_subproblem_multistep_history!(state, solver, sps, 3)
            step_rk_imex!(state, solver; ts=_RK443_SINGLETON)
            return
        end
        k2 = state.dt_history[end]
        k1 = state.dt_history[end-1]
        k0 = state.dt_history[end-2]
        a, b, c = _sbdf3_coefs(k2, k1, k0)
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    _try_step_explicit_multistep_field!(state, solver, :sbdf3) && return

    _global_multistep_distributed_fallback!(state, solver, current_state, "SBDF3") && return

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf3_iteration)

    iteration = state.timestepper_data[:sbdf3_iteration]

    # Startup: seed the early steps with order-3 IMEX RK so the multistep reaches
    # its nominal 3rd order (SBDF1/SBDF2 startup would cap it at order 2). The
    # global path needs the MX/F deques (populated by the startup), NOT a deep
    # state.history, so the switch is gated on iteration/timestep history only.
    if iteration < 2 || length(state.dt_history) < 3
        _multistep_rk443_startup!(state, solver, 3, :sbdf3_iteration)
        return
    end

    dt = state.dt

    k2 = state.dt_history[end]     # current timestep
    k1 = state.dt_history[end-1]   # previous timestep
    k0 = state.dt_history[end-2]   # timestep before that

    # Compute timestep ratios following Tarang (timesteppers:435-436)
    w2 = k2 / k1
    w1 = k1 / k0

    L_matrix, M_matrix, fell_back =
        _prepare_global_multistep_matrices!(state, solver, "SBDF3", "SBDF2", step_sbdf2!)
    fell_back && return

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

    @debug "SBDF3 variable timestep: k2=$k2, k1=$k1, k0=$k0, w2=$w2, w1=$w1"

    # `use_lx=false`: b = (1, 0, 0, 0), so the L·X history terms are all zero — and
    # skipping them also skips the matvec, which is not.
    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c;
                            mx_depth=3, f_depth=3, state_cap=4, use_lx=false,
                            iter_key=:sbdf3_iteration, name="SBDF3")
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
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        if _sp_multistep_history_depth(state) < 3 || length(state.dt_history) < 4
            _seed_subproblem_multistep_history!(state, solver, sps, 4)
            step_rk_imex!(state, solver; ts=_RK443_SINGLETON)
            return
        end
        k3 = state.dt_history[end]
        k2 = state.dt_history[end-1]
        k1 = state.dt_history[end-2]
        k0 = state.dt_history[end-3]
        a, b, c = _sbdf4_coefs(k3, k2, k1, k0)
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    _try_step_explicit_multistep_field!(state, solver, :sbdf4) && return

    _global_multistep_distributed_fallback!(state, solver, current_state, "SBDF4") && return

    # Initialize history arrays if needed
    _init_global_multistep_history!(state, :sbdf4_iteration)

    iteration = state.timestepper_data[:sbdf4_iteration]

    # Startup: seed the early steps with order-3 IMEX RK (gated on iteration /
    # timestep history; the global path needs the MX/F deques, not a deep
    # state.history). SBDF1/SBDF2 startup would cap the order at 2.
    if iteration < 3 || length(state.dt_history) < 4
        _multistep_rk443_startup!(state, solver, 4, :sbdf4_iteration)
        return
    end

    dt = state.dt

    k3 = state.dt_history[end]     # current timestep
    k2 = state.dt_history[end-1]   # previous timestep
    k1 = state.dt_history[end-2]   # timestep before that
    k0 = state.dt_history[end-3]   # timestep 3 back

    # Compute timestep ratios following Tarang (timesteppers:476-478)
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0

    L_matrix, M_matrix, fell_back =
        _prepare_global_multistep_matrices!(state, solver, "SBDF4", "SBDF3", step_sbdf3!)
    fell_back && return

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

    @debug "SBDF4 variable timestep: k3=$k3, k2=$k2, k1=$k1, k0=$k0, w3=$w3, w2=$w2, w1=$w1"

    # `use_lx=false` for the same reason as SBDF3.
    _global_multistep_core!(state, solver, current_state, L_matrix, M_matrix, a, b, c;
                            mx_depth=4, f_depth=4, state_cap=5, use_lx=false,
                            iter_key=:sbdf4_iteration, name="SBDF4")
end

# Exponential Time Differencing methods
