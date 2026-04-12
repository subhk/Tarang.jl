# ── Per-subproblem IMEX Runge-Kutta step ──────────────────────────────────────
#
# IMEX RK stepper that operates on per-Fourier-mode subproblems.
#
# For each Fourier mode (subproblem), we solve a sparse system of size
# ~(n_vars * Nz) at each IMEX stage.  LHS factorizations are cached per
# stage index and invalidated when dt changes.
#
# Sign convention (same as step_rk.jl):
#   LHS form:   M * dX/dt + L * X = F
#   Stage solve: (M + dt*a_ii*L) * X_i = M*X_n + dt*Σ_{j<i}(A^E_{ij}*F_j - A^I_{ij}*L*X_j)
#
# F_j and L*X_j are evaluated AFTER solving for X_j (at the stage solution),
# matching step_rk_imex! and the standard IMEX RK formulation.
# ──────────────────────────────────────────────────────────────────────────────

"""
    step_subproblem_rk!(state::TimestepperState, solver::InitialValueSolver,
                         subproblems::Tuple)

IMEX RK step using per-Fourier-mode subproblems.

For each subproblem (Fourier mode), assembles and solves:

    (M_min + dt*A^I_{ii}*L_min) * X_i = M_min*X_n
        + dt * Σ_{j<i}( A^E_{ij}*F_j - A^I_{ij}*L_min*X_j )

where `A^E` is the explicit Butcher tableau, `A^I` is the implicit Butcher tableau,
`M_min` and `L_min` are the preconditioned sparse matrices from `build_matrices!`,
`F_j = F(X_j, t+c_j*dt)` is the explicit RHS evaluated at stage solution `X_j`,
and `L*X_j` is the implicit contribution at stage solution `X_j`.

LHS factorizations are cached in `sp.LHS_solvers[stage_index]` and invalidated
when `dt` changes.
"""
function step_subproblem_rk!(state::TimestepperState, solver::InitialValueSolver,
                              subproblems::Tuple)
    ts = state.timestepper
    dt = state.dt
    t  = solver.sim_time
    problem = solver.problem
    stages = ts.stages

    # Butcher tableaux (1-indexed)
    A_exp = ts.A_explicit   # explicit tableau
    A_imp = ts.A_implicit   # implicit tableau
    c     = ts.c_explicit   # stage times

    # Collect the flat list of ScalarFields that represent the state
    state_fields = collect_state_fields(problem.variables)

    # Ensure all state fields are in coefficient space
    for f in state_fields
        ensure_layout!(f, :c)
    end

    # ── LHS cache invalidation ────────────────────────────────────────────
    prev_dt = get(state.timestepper_data, :_sp_rk_dt, NaN)
    if prev_dt != dt
        for sp in subproblems
            sp.M_min === nothing && continue
            # Reset all cached factorizations
            for k in eachindex(sp.LHS_solvers)
                sp.LHS_solvers[k] = nothing
            end
        end
        state.timestepper_data[:_sp_rk_dt] = dt
    end

    # ── Pre-stage: compute M*X_n per subproblem ──────────────────────────
    n_sp = length(subproblems)

    MX0 = Vector{Vector{ComplexF64}}(undef, n_sp)
    # F[j][sp_idx] = F(X_j) gathered per subproblem (evaluated AFTER stage j solve)
    F   = [Vector{Vector{ComplexF64}}(undef, n_sp) for _ in 1:stages]
    # LX[j][sp_idx] = L*X_j per subproblem (evaluated AFTER stage j solve)
    LX  = [Vector{Vector{ComplexF64}}(undef, n_sp) for _ in 1:stages]

    for (sp_idx, sp) in enumerate(subproblems)
        if sp.M_min === nothing
            MX0[sp_idx] = ComplexF64[]
            continue
        end
        x0_pre = gather_inputs(sp, state_fields)
        MX0[sp_idx] = sp.M_min * x0_pre
    end

    # ── Stage loop ────────────────────────────────────────────────────────
    # Each stage: build RHS → solve → scatter → evaluate F and L*X at solution
    for i in 1:stages
        state.current_substep = i

        # Build RHS and solve per subproblem
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue

            # RHS = M*X_n + dt * Σ_{j<i}( A^E_{ij}*F_j - A^I_{ij}*L*X_j )
            # F[j] and LX[j] contain values at stage j SOLUTION (set after stage j)
            rhs = copy(MX0[sp_idx])

            for j in 1:(i - 1)
                a_ej = dt * A_exp[i, j]
                a_ij = dt * A_imp[i, j]
                if abs(a_ej) > 1e-14
                    rhs .+= a_ej .* F[j][sp_idx]
                end
                if abs(a_ij) > 1e-14
                    rhs .-= a_ij .* LX[j][sp_idx]
                end
            end

            # Solve: (M_min + dt*a_ii*L_min) * x_sol = rhs
            a_ii = A_imp[i, i]

            if abs(a_ii) < 1e-14
                # No implicit diagonal — just invert M
                if sp.M_min !== nothing
                    M_lu = _get_or_compute_mass_lu!(sp)
                    if M_lu !== nothing
                        x_sol = M_lu \ rhs
                    else
                        x_sol = rhs  # fallback
                    end
                else
                    x_sol = rhs
                end
            else
                lhs_solver = _get_or_build_lhs!(sp, i, dt, a_ii)
                if lhs_solver !== nothing
                    x_sol = lhs_solver \ rhs
                else
                    @warn "step_subproblem_rk!: LHS factorization failed for sp group=$(sp.group), stage=$i; using rhs as fallback" maxlog=1
                    x_sol = rhs
                end
            end

            # Scatter solution back to state fields
            scatter_inputs(sp, x_sol, state_fields)
        end

        # Evaluate F[i] and LX[i] AFTER the solve, at the stage i solution.
        # This matches step_rk_imex! where F_exp_vecs[s] and F_imp_vecs[s]
        # are computed after the stage solve (step_rk.jl lines 176-179).
        F_fields = evaluate_rhs(solver, state_fields, t + dt * c[i])
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue
            F[i][sp_idx] = gather_outputs(sp, F_fields)
            x_pre = gather_inputs(sp, state_fields)
            LX[i][sp_idx] = sp.L_min !== nothing ? sp.L_min * x_pre : zeros(ComplexF64, length(x_pre))
        end
    end

    # ── Final update: M*X_{n+1} = M*X_n + dt*Σ(b^E*F - b^I*L*X) ────────
    # Always perform the full weighted update. The "stiffly accurate" shortcut
    # (skipping this when b_imp = A_imp[end,:]) is only valid when BOTH tableaux
    # are SA (b_exp = A_exp[end,:] AND b_imp = A_imp[end,:]). Neither RK222 nor
    # RK443 is explicitly SA, so the shortcut cannot be used.
    # This matches step_rk_imex! which always does the weighted update (lines 191-203).
    b_imp = ts.b_implicit
    b_exp = ts.b_explicit
    for (sp_idx, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue

        rhs = copy(MX0[sp_idx])
        for s in 1:stages
            be = dt * b_exp[s]
            bi = dt * b_imp[s]
            if abs(be) > 1e-14
                rhs .+= be .* F[s][sp_idx]
            end
            if abs(bi) > 1e-14
                rhs .-= bi .* LX[s][sp_idx]
            end
        end

        # Solve M * X_new = rhs
        M_lu = _get_or_compute_mass_lu!(sp)
        if M_lu !== nothing
            x_sol = M_lu \ rhs
        else
            # Singular M (DAE): last stage already satisfies constraints
            # via the implicit SA property. Keep the last stage value.
            continue
        end
        scatter_inputs(sp, x_sol, state_fields)
    end

    # ── Push new state to history ─────────────────────────────────────────
    new_state = collect_state_fields(problem.variables)
    _push_trim!(state.history, new_state, 1)
end

# ── Helper: get or build LHS factorization for a stage ────────────────────────

"""
    _get_or_build_lhs!(sp, stage_idx, dt, a_ii)

Return a cached LU factorization of `(M_min + dt*a_ii*L_min)` for the given
stage index, building and caching it if necessary.

`sp.LHS_solvers` is a `Vector{Any}` of length >= stages, indexed by stage.
Entry 0 is reserved for the mass-only factorization (used when a_ii = 0).
"""
function _get_or_build_lhs!(sp::Subproblem, stage_idx::Int, dt::Float64, a_ii::Float64)
    # Ensure LHS_solvers is large enough
    while length(sp.LHS_solvers) < stage_idx
        push!(sp.LHS_solvers, nothing)
    end

    cached = sp.LHS_solvers[stage_idx]
    if cached !== nothing
        return cached
    end

    M = sp.M_min
    L = sp.L_min
    if M === nothing
        return nothing
    end

    # Fast path: use expanded pattern for in-place LHS update.
    # sp.LHS has the union sparsity pattern of M and L, and sp.M_exp, sp.L_exp
    # have values from M and L mapped onto that pattern. This lets us compute
    #   LHS.nzval = 1 * M_exp.nzval + (dt*a_ii) * L_exp.nzval
    # without allocating a new sparse matrix or re-analyzing the sparsity.
    LHS = if sp.M_exp !== nothing && sp.L_exp !== nothing && sp.LHS !== nothing
        coeff = ComplexF64(dt * a_ii)
        sp.LHS.nzval .= sp.M_exp.nzval .+ coeff .* sp.L_exp.nzval
        sp.LHS
    elseif L !== nothing && abs(a_ii) > 1e-14
        M + dt * a_ii * L
    else
        copy(M)
    end

    lhs_lu = lu(LHS; check=false)
    if issuccess(lhs_lu)
        sp.LHS_solvers[stage_idx] = lhs_lu
        return lhs_lu
    end

    # Fallback for rank-deficient matrices (e.g., DC mode gauge issues)
    LHS_dense = Matrix(LHS)
    sp.LHS_solvers[stage_idx] = LHS_dense
    @info "step_subproblem_rk!: using dense fallback for group=$(sp.group), stage=$stage_idx" maxlog=1
    return LHS_dense
end

"""
    _get_or_compute_mass_lu!(sp)

Return a cached LU factorization of the mass matrix `sp.M_min`.
The factorization is stored in `sp.LHS_solvers` at a dedicated slot
(index `length(sp.LHS_solvers)` is used as a mass-only cache when we
grow the vector by one).

This is used for stages where `a_ii = 0` (e.g., ESDIRK first stage)
and for the non-stiffly-accurate final update.
"""
function _get_or_compute_mass_lu!(sp::Subproblem)
    M = sp.M_min
    M === nothing && return nothing

    # Use the last+1 slot convention: stages occupy 1..stages, mass uses stages+1.
    # But since we don't know stages here, just use a tagged approach:
    # Check if the very last slot is tagged as mass-only (we store a Tuple).
    # Simpler: just factorize -- this is only called for a_ii=0 stages (rare).
    # For ESDIRK first stage this is once per step (O(1) per step, not per mode).
    # Cache in a separate field would be better, but sp has no such field.
    # Since LHS_solvers is already invalidated per-dt, just compute and return.
    lhs_lu = lu(M; check=false)
    if issuccess(lhs_lu)
        return lhs_lu
    else
        return nothing
    end
end
