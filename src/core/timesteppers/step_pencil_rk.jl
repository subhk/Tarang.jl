# ── Per-pencil IMEX Runge-Kutta step ─────────────────────────────────────────

"""
Per-pencil IMEX RK step using the variable-centric PencilSystem.
"""
function step_pencil_system_rk!(state::TimestepperState, solver::InitialValueSolver,
                                 ps::PencilSystem)
    ts = state.timestepper
    dt = state.dt
    t = solver.sim_time
    stages = ts.stages
    A_exp = ts.A_explicit
    A_imp = ts.A_implicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit
    problem = solver.problem

    # Scatter problem variables to pencils
    X_n = vars_to_pencils(problem.variables, ps)

    # M * X_n per pencil
    MX_n = [ps.M_pencils[k] * X_n[k] for k in 1:ps.n_pencils]

    # Stage storage
    F_exp = Vector{Vector{Vector{ComplexF64}}}(undef, stages)
    F_imp = Vector{Vector{Vector{ComplexF64}}}(undef, stages)
    Xs = X_n

    for s in 1:stages
        state.current_substep = s

        # RHS = M*X_n + dt*Σ(a_exp*F_exp - a_imp*F_imp)
        rhs = [copy(MX_n[k]) for k in 1:ps.n_pencils]
        for j in 1:(s-1)
            a_ej = dt * A_exp[s, j]
            a_ij = dt * A_imp[s, j]
            for k in 1:ps.n_pencils
                abs(a_ej) > 1e-14 && (rhs[k] .+= a_ej .* F_exp[j][k])
                abs(a_ij) > 1e-14 && (rhs[k] .-= a_ij .* F_imp[j][k])
            end
        end

        # Implicit solve per pencil
        a_ii = A_imp[s, s]
        Xs = solve_pencil_system!(ps, rhs, dt, a_ii)

        # Scatter pencil solution back to problem variables
        pencils_to_vars!(problem.variables, Xs, ps)

        # Evaluate explicit RHS — returns Vector{ScalarField} (flat scalar list)
        current_state = collect_state_fields(problem.variables)
        F_fields = evaluate_rhs(solver, current_state, t + c[s] * dt)
        # Scatter RHS fields to pencils using the flat-state helper
        F_exp[s] = scalar_state_to_pencils(F_fields, problem.variables, ps)

        # Implicit contribution: L * Xs
        F_imp[s] = [ps.L_pencils[k] * Xs[k] for k in 1:ps.n_pencils]
    end

    # Final update
    is_stiffly_accurate = b_imp ≈ A_imp[end, :]
    X_new = if is_stiffly_accurate
        Xs
    else
        # Full weighted update: M*X_new = M*X_n + dt*Σ(b_exp*F - b_imp*L*X)
        # Then solve M*X_new = result per pencil
        result = [copy(MX_n[k]) for k in 1:ps.n_pencils]
        for s in 1:stages
            be = dt * b_exp[s]
            bi = dt * b_imp[s]
            for k in 1:ps.n_pencils
                abs(be) > 1e-14 && (result[k] .+= be .* F_exp[s][k])
                abs(bi) > 1e-14 && (result[k] .-= bi .* F_imp[s][k])
            end
        end
        # Solve M * X_new = result (apply M^{-1})
        for k in 1:ps.n_pencils
            M_lu = lu(ps.M_pencils[k]; check=false)
            if issuccess(M_lu)
                result[k] = M_lu \ result[k]
            end
            # If M is singular (DAE), keep result as-is (approximation)
        end
        result
    end

    # Write solution to problem variables
    pencils_to_vars!(problem.variables, X_new, ps)

    # Update solver state from problem variables
    new_state = collect_state_fields(problem.variables)
    _push_trim!(state.history, new_state, 1)

    # Evict stale LHS cache entries
    filter!(kv -> kv[1][2] == dt, ps.lhs_cache)
end
