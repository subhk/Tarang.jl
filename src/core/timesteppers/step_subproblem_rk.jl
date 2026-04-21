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
    C::SparseMatrixCSC                       # BC rows × bulk cols
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

    # Require non-empty bulk AND bc, and square bulk/bc sub-blocks
    if isempty(bulk_rows) || isempty(bc_rows)
        return nothing
    end
    if length(bulk_rows) != length(bulk_cols) || length(bc_rows) != length(bc_cols)
        return nothing
    end

    A = LHS[bulk_rows, bulk_cols]
    B = LHS[bulk_rows, bc_cols]
    C = LHS[bc_rows, bulk_cols]
    D = LHS[bc_rows, bc_cols]

    bulk_lu = lu(A; check=false)
    if !issuccess(bulk_lu)
        return nothing
    end

    # Precompute A⁻¹ · B as a dense matrix (n_bulk × n_bc)
    B_dense = Matrix{ComplexF64}(B)
    AinvB = Matrix{ComplexF64}(bulk_lu \ B_dense)

    # Schur complement: S = D - C · AinvB (dense, n_bc × n_bc)
    S = Matrix{ComplexF64}(D) - Matrix{ComplexF64}(C) * AinvB
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
    _refresh_bcs_for_stage!(solver, stage_time) -> Bool

Re-evaluate time-dependent BCs at `stage_time` and rewrite the
`equation_data["F"]` / `["F_expr"]` slots. Returns `true` if any BC was
actually refreshed (so the caller can re-gather the algebraic F vector),
`false` otherwise.

Only runs for time-dependent (including space+time) BCs. Pure
space-dependent BCs are already populated at solver-build time and don't
change between stages, so we skip them to avoid redundant FFT work.

For time-varying BCs this is called at every RK stage with `t + c[i]*dt`
so multi-stage methods keep their formal order of accuracy on rapidly
varying BCs.
"""
function _refresh_bcs_for_stage!(solver::InitialValueSolver, stage_time::Real)
    bcm = solver.problem.bc_manager
    has_time_dependent_bcs(bcm) || return false
    update_time_dependent_bcs!(bcm, stage_time)
    _apply_bc_values_to_equations!(solver, stage_time)
    return true
end

_subproblem_solver_type(choice) = choice isa Tuple ? choice[1] : choice

function _subproblem_solver_kwargs(choice)
    choice isa Tuple || return Pair{Symbol, Any}[]
    kwargs = Pair{Symbol, Any}[]
    for item in choice[2:end]
        item isa Pair || continue
        key = item.first
        key isa Symbol || continue
        push!(kwargs, key => item.second)
    end
    return kwargs
end

function _solve_cached_system(lhs_solver, rhs::AbstractVector{ComplexF64})
    return if isa(lhs_solver, WoodburySolver)
        _woodbury_solve(lhs_solver, rhs)
    elseif lhs_solver isa MatSolvers.AbstractMatSolver
        MatSolvers.solve(lhs_solver, rhs)
    else
        lhs_solver \ rhs
    end
end

function _solve_cached_system!(dest::AbstractVector{ComplexF64}, lhs_solver,
                               rhs::AbstractVector{ComplexF64})
    if isa(lhs_solver, WoodburySolver)
        _assign_to_buffer!(dest, _woodbury_solve(lhs_solver, rhs))
    elseif lhs_solver isa MatSolvers.AbstractMatSolver
        MatSolvers.solve!(dest, lhs_solver, rhs)
    else
        _assign_to_buffer!(dest, lhs_solver \ rhs)
    end
    return dest
end

function _subproblem_operator(sp::Subproblem, which::Symbol, data::AbstractVector)
    matrix = which === :M ? sp.M_min : sp.L_min
    matrix === nothing && return nothing
    cache_key = which === :M ? "_M_min_backend" : "_L_min_backend"
    return _subproblem_backend_matrix!(sp, matrix, cache_key, data)
end

function _apply_subproblem_operator!(dest::AbstractVector, op, x::AbstractVector)
    if op === nothing
        fill!(dest, zero(eltype(dest)))
    elseif !is_gpu_array(dest) && !is_gpu_array(x) && op isa AbstractMatrix
        mul!(dest, op, x)
    else
        _assign_to_buffer!(dest, op * x)
    end
    return dest
end

function _sp_stage_vector!(sp::Subproblem, key::String, n::Int, like::AbstractVector)
    return _subproblem_cached_vector!(sp, key, n; like=like)
end

function _sp_stage_vector!(sp::Subproblem, key::String, n::Int)
    return _subproblem_cached_vector!(sp, key, n)
end

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

LHS factorizations are cached in `sp.LHS_solvers` keyed by `a_ii`
(ESDIRK methods share a single factorization across all stages) and
invalidated lazily when `dt` changes via the dirty-flag path in
`_get_or_build_lhs!`.
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

    # ── LHS cache invalidation (symbolic-reuse aware) ─────────────────────
    # When `dt` changes, the LHS matrix `(M + dt*a_ii*L)` has new numeric
    # values but the same sparsity pattern (assuming the expanded-pattern
    # fast path is active — which it is by default). Rather than dumping
    # the cached LU and triggering a full symbolic + numeric rebuild, we
    # mark each cached `a_ii` entry as "needs refactor" and
    # `_get_or_build_lhs!` calls `MatSolvers.refactor!` which reuses the
    # cached symbolic factor and runs only the numeric phase. Saves
    # 30–60% of the LU cost under adaptive CFL.
    prev_dt = get(state.timestepper_data, :_sp_rk_dt, NaN)
    if prev_dt != dt
        for sp in subproblems
            sp.M_min === nothing && continue
            # Mark all cached entries dirty; _get_or_build_lhs! will
            # refactor in place on the next call.
            for a_ii_key in keys(sp.LHS_solvers)
                sp.matrices["_lhs_dirty_$(a_ii_key)"] = true
            end
        end
        state.timestepper_data[:_sp_rk_dt] = dt
    end

    # ── Pre-stage: compute M*X_n per subproblem ──────────────────────────
    n_sp = length(subproblems)

    MX0 = Vector{AbstractVector{ComplexF64}}(undef, n_sp)
    # F[j][sp_idx] = F(X_j) gathered per subproblem (evaluated AFTER stage j solve)
    F   = [Vector{AbstractVector{ComplexF64}}(undef, n_sp) for _ in 1:stages]
    # LX[j][sp_idx] = L*X_j per subproblem (evaluated AFTER stage j solve)
    LX  = [Vector{AbstractVector{ComplexF64}}(undef, n_sp) for _ in 1:stages]
    RHS = Vector{AbstractVector{ComplexF64}}(undef, n_sp)
    # Algebraic-constraint F (from BC / no-M equations), packed in equation
    # space with zeros at PDE rows. Computed ONCE per step because BC values
    # are stage-independent. Used to override BC rows in the stage RHS with
    # `dt*a_ii*F_alg`, yielding `L_row*X = F_alg` at each stage (correct DAE
    # enforcement). Without this override, the accumulated IMEX-RK formula
    # scales BC F by `A^E[i,j]/a_ii` (= 1/γ for RK222 stage 2 = 2+√2 ≈ 3.414)
    # and inhomogeneous BCs would be enforced at the wrong value.
    ALG_F = Vector{AbstractVector{ComplexF64}}(undef, n_sp)

    for (sp_idx, sp) in enumerate(subproblems)
        if sp.M_min === nothing
            MX0[sp_idx] = ComplexF64[]
            RHS[sp_idx] = ComplexF64[]
            ALG_F[sp_idx] = ComplexF64[]
            continue
        end
        n_eq = size(sp.M_min, 1)
        n_var = size(sp.M_min, 2)
        mx0 = _sp_stage_vector!(sp, "_sp_rk_mx0", n_eq)
        rhs = _sp_stage_vector!(sp, "_sp_rk_rhs", n_var)
        x0_pre = gather_inputs!(rhs, sp, state_fields)
        _apply_subproblem_operator!(mx0, _subproblem_operator(sp, :M, x0_pre), x0_pre)
        MX0[sp_idx] = mx0
        RHS[sp_idx] = rhs

        alg_f = _sp_stage_vector!(sp, "_sp_rk_alg_f", n_eq, mx0)
        gather_alg_F!(alg_f, sp)
        ALG_F[sp_idx] = alg_f
    end

    # Whether this problem has any time-dependent BCs. If so, the stage
    # loop below refreshes `ALG_F` at each stage time `t + c[i]*dt` to
    # retain full stage-order accuracy on rapidly-varying BCs. Pure
    # space-dependent (non-time) BCs are already populated at solver build
    # time — they don't need per-stage refreshes.
    bc_dynamic = has_time_dependent_bcs(problem.bc_manager)

    # ── Stage loop ────────────────────────────────────────────────────────
    # Each stage: build RHS → solve → scatter → evaluate F and L*X at solution
    for i in 1:stages
        state.current_substep = i

        # Per-stage BC refresh + ALG_F re-gather (only for dynamic BCs).
        # `update_time_dependent_bcs!` / `_apply_bc_values_to_equations!`
        # rewrite `equation_data[eq_idx]["F"]`, which `gather_alg_F!` reads
        # on each call, so we simply re-gather after refreshing.
        if bc_dynamic
            stage_time = t + dt * c[i]
            if _refresh_bcs_for_stage!(solver, stage_time)
                for (sp_idx, sp) in enumerate(subproblems)
                    sp.M_min === nothing && continue
                    alg_f = ALG_F[sp_idx]
                    gather_alg_F!(alg_f, sp)
                end
            end
        end

        # Build RHS and solve per subproblem
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue

            # RHS = M*X_n + dt * Σ_{j<i}( A^E_{ij}*F_j - A^I_{ij}*L*X_j )
            # F[j] and LX[j] contain values at stage j SOLUTION (set after stage j)
            rhs = RHS[sp_idx]
            _assign_to_buffer!(rhs, MX0[sp_idx])

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

            # Override BC rows with `dt*a_ii*F_alg`. For algebraic rows (M=0),
            # the stage LHS reduces to `dt*a_ii*L_row*X = dt*a_ii*F_alg`, i.e.,
            # `L_row*X = F_alg`, enforcing the BC at every stage regardless of
            # accumulated history. `sp.bc_rows` lists the L_min row indices that
            # correspond to algebraic (BC-like) equations, computed in
            # build_matrices! from the Woodbury block classification.
            # Vectorized (CUDA.allowscalar(false)-safe) override via
            # `apply_bc_override!`.
            if abs(a_ii) > 1e-14
                apply_bc_override!(rhs, ALG_F[sp_idx], sp, dt * a_ii)
            end

            if abs(a_ii) < 1e-14
                # No implicit diagonal — just invert M
                x_sol = _sp_stage_vector!(sp, "_sp_rk_sol_stage_$i", size(sp.M_min, 2), RHS[sp_idx])
                if sp.M_min !== nothing
                    M_lu = _get_or_compute_mass_lu!(sp)
                    if M_lu !== nothing
                        _solve_cached_system!(x_sol, M_lu, rhs)
                    else
                        _assign_to_buffer!(x_sol, rhs)  # fallback
                    end
                else
                    _assign_to_buffer!(x_sol, rhs)
                end
            else
                x_sol = _sp_stage_vector!(sp, "_sp_rk_sol_stage_$i", size(sp.M_min, 2), RHS[sp_idx])
                lhs_solver = _get_or_build_lhs!(sp, i, dt, a_ii)
                if lhs_solver !== nothing
                    _solve_cached_system!(x_sol, lhs_solver, rhs)
                else
                    @warn "step_subproblem_rk!: LHS factorization failed for sp group=$(sp.group), stage=$i; using rhs as fallback" maxlog=1
                    _assign_to_buffer!(x_sol, rhs)
                end
            end

            # Scatter solution back to state fields
            scatter_inputs(sp, x_sol, state_fields)
        end

        # Evaluate F[i] and LX[i] AFTER the solve, at the stage i solution.
        # This matches step_rk_imex! where F_exp_vecs[s] and F_imp_vecs[s]
        # are computed after the stage solve (step_rk.jl lines 176-179).
        #
        # `evaluate_rhs` returns per-STATE-FIELD F values for PDE equations.
        # `gather_eqn_F!` packs them into equation-space rows, leaving BC rows
        # at zero — those are handled by the `ALG_F` override above because
        # the IMEX-RK accumulation formula gives the wrong `1/γ` scaling for
        # inhomogeneous algebraic constraints like `T(z=0) = 1`.
        F_fields = evaluate_rhs_buffered(solver, state_fields, t + dt * c[i])
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue
            f_stage = _sp_stage_vector!(sp, "_sp_rk_F_stage_$i", size(sp.M_min, 1), MX0[sp_idx])
            x_pre = RHS[sp_idx]
            gather_eqn_F!(f_stage, sp, solver, F_fields, state_fields)
            gather_inputs!(x_pre, sp, state_fields)
            L_op = _subproblem_operator(sp, :L, x_pre)
            lx_stage = _sp_stage_vector!(sp, "_sp_rk_LX_stage_$i", length(x_pre), x_pre)
            _apply_subproblem_operator!(lx_stage, L_op, x_pre)
            F[i][sp_idx] = f_stage
            LX[i][sp_idx] = lx_stage
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

        # Cached scratch buffer for the final-update rhs (size n_eq). Reusing
        # a persistent vector instead of `copy(MX0[sp_idx])` eliminates one
        # ComplexF64 allocation per subproblem per step.
        rhs = _sp_stage_vector!(sp, "_sp_rk_final_rhs", size(sp.M_min, 1), MX0[sp_idx])
        copyto!(rhs, MX0[sp_idx])
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
            x_sol = _sp_stage_vector!(sp, "_sp_rk_sol_final", size(sp.M_min, 2), RHS[sp_idx])
            _solve_cached_system!(x_sol, M_lu, rhs)
        else
            # Singular M (DAE): last stage already satisfies constraints
            # via the implicit SA property. Keep the last stage value.
            continue
        end
        scatter_inputs(sp, x_sol, state_fields)
    end

    # ── Push new state to history ─────────────────────────────────────────
    _push_trim!(state.history, state_fields, 1)
end

# ── Helper: get or build LHS factorization for a stage ────────────────────────

"""
    _get_or_build_lhs!(sp, stage_idx, dt, a_ii)

Return a cached solver for `(M_min + dt*a_ii*L_min)`, building it lazily.

### Cache structure

`sp.LHS_solvers` is a `Dict{Float64, Any}` keyed by `a_ii` — NOT by
stage index. For ESDIRK methods where every implicit stage shares the
same `γ` on the diagonal (RK222, RK443), a single factorization is
built once and reused across all stages, saving `(stages - 1)` LU
factorizations per step. For non-ESDIRK DIRKs with distinct per-stage
`a_ii` values, each distinct `a_ii` gets its own cache entry.

### Dt changes

When `dt` changes, the dt-change handler in `step_subproblem_rk!` marks
each cached entry as dirty via `sp.matrices["_lhs_dirty_<a_ii>"]`. The
next call to `_get_or_build_lhs!` detects the dirty flag and refactors
in place via `MatSolvers.refactor!` — reusing the cached symbolic
factorization and running only the numeric phase.

The `stage_idx` parameter is kept for logging/diagnostic purposes only;
the cache lookup uses `a_ii`.
"""
function _get_or_build_lhs!(sp::Subproblem, stage_idx::Int, dt::Float64, a_ii::Float64)
    M = sp.M_min
    L = sp.L_min
    if M === nothing
        return nothing
    end
    solver_type = _subproblem_solver_type(sp.solver.base.matsolver)
    solver_kwargs = _subproblem_solver_kwargs(sp.solver.base.matsolver)

    # Rebuild the LHS matrix. The fast path updates `sp.LHS.nzval` in
    # place — same sparsity pattern as the previously-cached factor, so
    # we can refactor via symbolic reuse below.
    LHS = if sp.M_exp !== nothing && sp.L_exp !== nothing && sp.LHS !== nothing
        coeff = ComplexF64(dt * a_ii)
        sp.LHS.nzval .= sp.M_exp.nzval .+ coeff .* sp.L_exp.nzval
        sp.LHS
    elseif L !== nothing && abs(a_ii) > 1e-14
        M + dt * a_ii * L
    else
        copy(M)
    end

    # Look up cached entry by a_ii (shared across ESDIRK stages).
    cached = get(sp.LHS_solvers, a_ii, nothing)
    dirty_key = "_lhs_dirty_$(a_ii)"
    is_dirty = get(sp.matrices, dirty_key, false)::Bool

    if cached !== nothing && !is_dirty
        # Matrix unchanged since last dt — cache hit, zero work.
        return cached
    end

    # Symbolic-reuse fast path: cached solver exists but dt has changed.
    # Call `refactor!` to reuse the symbolic factorization and run only
    # the numeric phase. Works for any `AbstractMatSolver` subtype that
    # defines `refactor!` — CPU `SparseLUSolver`, GPU `CuSparseLU`
    # (Float64 via cusolverRF). Detected via `hasmethod` so the check is
    # independent of whether CUDA.jl is loaded.
    if cached !== nothing && is_dirty &&
       hasmethod(MatSolvers.refactor!, Tuple{typeof(cached), typeof(LHS)})
        try
            MatSolvers.refactor!(cached, LHS)
            sp.matrices[dirty_key] = false
            return cached
        catch err
            @debug "refactor! failed, falling back to full rebuild" exception=(err, catch_backtrace())
            # fall through to build-from-scratch
        end
    end
    # Fresh build (first call at this a_ii, or refactor failed).
    sp.matrices[dirty_key] = false

    # Woodbury block-LU is only safe when the bulk/BC metadata keeps coupling
    # taus in the bulk block. The partition is built during matrix assembly
    # from the permuted/filterd sparse structure rather than from DOF size
    # alone, so first-order/tau formulations can still use the Schur solve.
    has_woodbury_partition = !isempty(sp.bulk_rows) && !isempty(sp.bc_rows) &&
                             length(sp.bulk_rows) == length(sp.bulk_cols) &&
                             length(sp.bc_rows) == length(sp.bc_cols)
    if has_woodbury_partition && solver_type == MatSolvers.SparseLUSolver
        w = _build_woodbury(LHS, sp)
        if w !== nothing
            sp.LHS_solvers[a_ii] = w
            return w
        end
    end

    try
        lhs_solver = MatSolvers.solver_instance(solver_type, LHS; solver_kwargs...)
        sp.LHS_solvers[a_ii] = lhs_solver
        return lhs_solver
    catch err
        if solver_type != MatSolvers.SPQRSolver
            try
                qr_solver = MatSolvers.solver_instance(MatSolvers.SPQRSolver, LHS)
                sp.LHS_solvers[a_ii] = qr_solver
                @info "step_subproblem_rk!: using sparse QR fallback for group=$(sp.group), stage=$stage_idx" maxlog=1
                return qr_solver
            catch
            end
        end
        @debug "step_subproblem_rk!: solver build failed for group=$(sp.group), stage=$stage_idx" exception=(err, catch_backtrace())
    end

    # Final fallback for rank-deficient or unsupported matrices
    LHS_dense = Matrix(LHS)
    sp.LHS_solvers[a_ii] = LHS_dense
    @info "step_subproblem_rk!: using dense fallback for group=$(sp.group), stage=$stage_idx" maxlog=1
    return LHS_dense
end

"""
    _get_or_compute_mass_lu!(sp)

Return a cached solver for the mass matrix `sp.M_min`.
The solver is cached in `sp.matrices["_mass_solver"]`.

This is used for stages where `a_ii = 0` (e.g., ESDIRK first stage)
and for the non-stiffly-accurate final update.
"""
function _get_or_compute_mass_lu!(sp::Subproblem)
    M = sp.M_min
    M === nothing && return nothing
    cached = get(sp.matrices, "_mass_solver", nothing)
    cached !== nothing && return cached

    solver_type = _subproblem_solver_type(sp.solver.base.matsolver)
    solver_kwargs = _subproblem_solver_kwargs(sp.solver.base.matsolver)
    try
        mass_solver = MatSolvers.solver_instance(solver_type, M; solver_kwargs...)
        sp.matrices["_mass_solver"] = mass_solver
        return mass_solver
    catch err
        if solver_type != MatSolvers.SPQRSolver
            try
                mass_solver = MatSolvers.solver_instance(MatSolvers.SPQRSolver, M)
                sp.matrices["_mass_solver"] = mass_solver
                return mass_solver
            catch
            end
        end
        @debug "step_subproblem_rk!: mass solver build failed for group=$(sp.group)" exception=(err, catch_backtrace())
        return nothing
    end
end
