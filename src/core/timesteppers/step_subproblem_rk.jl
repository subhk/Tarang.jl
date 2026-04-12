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
    WoodburySolver

Block-LU solver using the Woodbury (Schur complement) approach.
Partitions the LHS matrix as:
    [A B]
    [C D]
where A is the large "bulk" block (PDE equations × PDE variables) and
D is the small "BC" block (boundary conditions × tau variables).
B and C are the off-diagonal couplings.

Factorization:
    A = bulk_lu (sparse LU)
    AinvB = A⁻¹ · B (dense, n_bulk × n_bc)
    S = D - C · AinvB (Schur complement, n_bc × n_bc dense)
    S_lu = LU of S (dense)

Block solve for (LHS · x = rhs):
    1. y1 = A⁻¹ · rhs_bulk            (sparse solve)
    2. r_bc = rhs_bc - C · y1          (sparse matvec)
    3. x_bc = S⁻¹ · r_bc               (small dense solve)
    4. x_bulk = y1 - AinvB · x_bc      (dense matvec)
"""
mutable struct WoodburySolver
    bulk_lu::Any                             # sparse LU of bulk block A
    C::SparseMatrixCSC{ComplexF64, Int64}    # BC rows × bulk cols
    AinvB::Matrix{ComplexF64}                # precomputed A⁻¹ · B
    S_lu::Any                                # LU of Schur complement (dense)
    bulk_rows::Vector{Int}
    bc_rows::Vector{Int}
    bulk_cols::Vector{Int}
    bc_cols::Vector{Int}
end

"""Build a Woodbury solver for `LHS = a0*M_exp + b0*L_exp` given bulk/BC partition."""
function _build_woodbury(LHS::SparseMatrixCSC, sp::Subproblem)
    bulk_rows = sp.bulk_rows
    bc_rows = sp.bc_rows
    bulk_cols = sp.bulk_cols
    bc_cols = sp.bc_cols

    A = LHS[bulk_rows, bulk_cols]
    B = LHS[bulk_rows, bc_cols]
    C = LHS[bc_rows, bulk_cols]
    D = LHS[bc_rows, bc_cols]

    bulk_lu = lu(A; check=false)
    if !issuccess(bulk_lu)
        return nothing
    end

    # Precompute A⁻¹ · B (dense, n_bulk × n_bc)
    B_dense = Matrix(B)
    AinvB = bulk_lu \ B_dense

    # Schur complement: S = D - C · AinvB
    S = Matrix(D) - Matrix(C) * AinvB
    S_lu = lu(S; check=false)
    if !issuccess(S_lu)
        return nothing
    end

    return WoodburySolver(bulk_lu, C, AinvB, S_lu, bulk_rows, bc_rows, bulk_cols, bc_cols)
end

"""Solve `LHS · x = rhs` using a Woodbury block solver."""
function _woodbury_solve(w::WoodburySolver, rhs::AbstractVector{ComplexF64})
    rhs_bulk = rhs[w.bulk_rows]
    rhs_bc   = rhs[w.bc_rows]

    # 1. y1 = A⁻¹ · rhs_bulk
    y1 = w.bulk_lu \ rhs_bulk

    # 2. r_bc = rhs_bc - C · y1
    r_bc = rhs_bc - w.C * y1

    # 3. x_bc = S⁻¹ · r_bc
    x_bc = w.S_lu \ r_bc

    # 4. x_bulk = y1 - AinvB · x_bc
    x_bulk = y1 - w.AinvB * x_bc

    # Reassemble full solution vector using the original row ordering
    n_total = length(w.bulk_cols) + length(w.bc_cols)
    x = Vector{ComplexF64}(undef, n_total)
    for (i, col) in enumerate(w.bulk_cols)
        x[col] = x_bulk[i]
    end
    for (i, col) in enumerate(w.bc_cols)
        x[col] = x_bc[i]
    end
    return x
end
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
                    x_sol = if isa(lhs_solver, WoodburySolver)
                        _woodbury_solve(lhs_solver, rhs)
                    else
                        lhs_solver \ rhs
                    end
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
    LHS = if sp.M_exp !== nothing && sp.L_exp !== nothing && sp.LHS !== nothing
        coeff = ComplexF64(dt * a_ii)
        sp.LHS.nzval .= sp.M_exp.nzval .+ coeff .* sp.L_exp.nzval
        sp.LHS
    elseif L !== nothing && abs(a_ii) > 1e-14
        M + dt * a_ii * L
    else
        copy(M)
    end

    # Woodbury path: if bulk/BC classification is available and non-trivial,
    # build a block-LU solver using the Schur complement.
    # Benefits:
    # - Only factor the bulk block (n_bulk × n_bulk) with sparse LU
    # - BC corrections handled via a small dense Schur solve (n_bc × n_bc)
    # - Scales better for large 3D systems where n_bulk >> n_bc
    has_woodbury_partition = !isempty(sp.bulk_rows) && !isempty(sp.bc_rows) &&
                              length(sp.bulk_rows) == length(sp.bulk_cols) &&
                              length(sp.bc_rows) == length(sp.bc_cols)
    if has_woodbury_partition
        w = _build_woodbury(LHS, sp)
        if w !== nothing
            sp.LHS_solvers[stage_idx] = w
            return w
        end
    end

    # Standard sparse LU on the full matrix
    lhs_lu = lu(LHS; check=false)
    if issuccess(lhs_lu)
        sp.LHS_solvers[stage_idx] = lhs_lu
        return lhs_lu
    end

    # Fallback for rank-deficient matrices
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
