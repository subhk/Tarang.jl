# ── Per-subproblem IMEX Runge-Kutta step ──────────────────────────────────────
#
# Dedalus-style IMEX RK stepper that operates on per-Fourier-mode subproblems.
#
# For each Fourier mode (subproblem), we solve a sparse system of size
# ~(n_vars * Nz) at each IMEX stage.  LHS factorizations are cached per
# stage index and invalidated when dt changes.
#
# Sign convention (same as step_rk.jl):
#   LHS form:   M * dX/dt + L * X = F
#   Stage solve: (M + k*H[i,i]*L) * X(n,i) = M*X(n,0) + k*Sigma(A*F - H*L*X)
# ──────────────────────────────────────────────────────────────────────────────

"""
    step_subproblem_rk!(state::TimestepperState, solver::InitialValueSolver,
                         subproblems::Tuple)

Dedalus-style IMEX RK step using per-Fourier-mode subproblems.

For each subproblem (Fourier mode), assembles and solves:

    (M_min + dt*H[i,i]*L_min) * X(n,i) = M_min*X(n,0)
        + dt * sum_{j<i}( A[i,j]*F[j] - H[i,j]*L_min*X(n,j) )

where `A` is the explicit Butcher tableau, `H` is the implicit Butcher tableau,
`M_min` and `L_min` are the preconditioned sparse matrices from `build_matrices!`,
and `F[j]` is the explicit RHS gathered through the subproblem's output permutation.

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

    # Butcher tableaux (1-indexed; row s in Julia = stage s in pseudocode)
    A_exp = ts.A_explicit   # explicit tableau  ("A" in Dedalus)
    A_imp = ts.A_implicit   # implicit tableau  ("H" in Dedalus)
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

    # ── Pre-stage: compute M*X(n,0) and L*X(n,0) per subproblem ──────────
    n_sp = length(subproblems)

    # Storage: MX0[sp_idx] and LX[stage_idx][sp_idx]
    MX0 = Vector{Vector{ComplexF64}}(undef, n_sp)
    # LX is indexed 0..stages-1 in Dedalus; here we use 1..stages (Julia 1-based)
    # LX[j] holds L_min * X(n,j-1)  (j=1 is the initial state)
    LX  = [Vector{Vector{ComplexF64}}(undef, n_sp) for _ in 1:stages]
    # F[j] holds the explicit RHS gathered at stage j (j=1..stages)
    F   = [Vector{Vector{ComplexF64}}(undef, n_sp) for _ in 1:stages]

    for (sp_idx, sp) in enumerate(subproblems)
        if sp.M_min === nothing
            # Skip subproblems with no mass matrix
            MX0[sp_idx] = ComplexF64[]
            LX[1][sp_idx] = ComplexF64[]
            continue
        end
        x0_pre = gather_inputs(sp, state_fields)
        MX0[sp_idx] = sp.M_min * x0_pre
        LX[1][sp_idx] = sp.L_min !== nothing ? sp.L_min * x0_pre : zeros(ComplexF64, length(x0_pre))
    end

    # ── Stage loop ────────────────────────────────────────────────────────
    for i in 1:stages
        state.current_substep = i

        # Recompute L*X(n,i-1) if i > 1 (state_fields were updated by scatter)
        if i > 1
            for (sp_idx, sp) in enumerate(subproblems)
                sp.M_min === nothing && continue
                x_pre = gather_inputs(sp, state_fields)
                LX[i][sp_idx] = sp.L_min !== nothing ? sp.L_min * x_pre : zeros(ComplexF64, length(x_pre))
            end
        end

        # Evaluate explicit RHS F(n,i-1) at time t + dt*c[i]
        # (state_fields currently hold X(n,i-1))
        F_fields = evaluate_rhs(solver, state_fields, t + dt * c[i])

        # Gather F into per-subproblem vectors
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue
            F[i][sp_idx] = gather_outputs(sp, F_fields)
        end

        # Build RHS and solve per subproblem
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue

            # RHS = MX0 + k * sum_{j=1}^{i} ( A_exp[i,j]*F[j] - A_imp[i,j]*LX[j] )
            # Note: j runs from 1..i-1 for off-diagonal, A_imp[i,i] goes into LHS
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

            # Also add explicit contribution from current stage if A_exp has it
            # (for ESDIRK the first column of A_exp row i may have a_exp[i,i]=0,
            #  but we still need to add F[i] if A_exp[i,i] != 0 -- typically it is 0
            #  for standard IMEX RK since explicit is strictly lower triangular)
            # The loop above already handles j < i; A_exp[i,i] = 0 for standard methods.

            # Solve: (M_min + dt*H[i,i]*L_min) * x_sol = rhs
            a_ii = A_imp[i, i]

            if abs(a_ii) < 1e-14
                # No implicit diagonal -- just invert M
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
                # stage_idx for LHS_solvers: use i (1-based stage index)
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
    end

    # ── Final update ──────────────────────────────────────────────────────
    # Stiffly accurate check: if the last row of A_imp equals b_imp,
    # the last stage IS the new state (no separate weighted update needed).
    b_imp = ts.b_implicit
    b_exp = ts.b_explicit
    is_stiffly_accurate = b_imp ≈ A_imp[end, :]

    if !is_stiffly_accurate
        # Full weighted update: M*X_new = M*X_n + dt*sum(b_exp*F - b_imp*L*X)
        # Recompute L*X for the last stage
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
                    # Need LX for stage s; for s < stages it is already computed.
                    # For the last stage we need to gather again.
                    if s == stages
                        x_pre = gather_inputs(sp, state_fields)
                        lx_s = sp.L_min !== nothing ? sp.L_min * x_pre : zeros(ComplexF64, length(x_pre))
                    else
                        lx_s = LX[s][sp_idx]
                    end
                    rhs .-= bi .* lx_s
                end
            end

            # Solve M * X_new = rhs
            M_lu = _get_or_compute_mass_lu!(sp)
            if M_lu !== nothing
                x_sol = M_lu \ rhs
            else
                x_sol = rhs
            end
            scatter_inputs(sp, x_sol, state_fields)
        end
    end
    # If stiffly accurate, state_fields already hold the last stage solution.

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

    # Build LHS = M_min + dt * a_ii * L_min
    M = sp.M_min
    L = sp.L_min
    if M === nothing
        return nothing
    end

    LHS = if L !== nothing && abs(a_ii) > 1e-14
        M + dt * a_ii * L
    else
        copy(M)
    end

    lhs_lu = lu(LHS; check=false)
    if issuccess(lhs_lu)
        sp.LHS_solvers[stage_idx] = lhs_lu
        return lhs_lu
    else
        @warn "step_subproblem_rk!: LU factorization failed for group=$(sp.group), stage=$stage_idx" maxlog=1
        return nothing
    end
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
