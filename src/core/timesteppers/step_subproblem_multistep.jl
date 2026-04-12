# ── Per-subproblem IMEX multistep step ────────────────────────────────────────
#
# Generic IMEX multistep stepper that operates on per-Fourier-mode subproblems.
#
# Handles any multistep method parameterized by (a, b, c) coefficient tuples:
#
#   Σ_k a[k] M X^{n+1-k} + Σ_k b[k] L X^{n+1-k} = Σ_k c[k] F^{n+1-k}
#
# where k=0 is the unknown new step and k≥1 are stored history values.
# Rearranging to solve form:
#
#   (a[0]*M + b[0]*L) X^{n+1} = Σ_{k≥1} [ c[k]*F_k − a[k]*M*X_k − b[k]*L*X_k ]
#
# For inhomogeneous algebraic constraints (BC rows where M_row = 0), the RHS
# rows are OVERRIDDEN with `b[0] * F_BC`. This mirrors the dt*a_ii override in
# `step_subproblem_rk!` — without it, the accumulation formula gives wrong
# scaling for F_BC (the previous-step L*X cancellation only works if prior
# stages satisfied the constraint, which fails on the first call with a
# history-free state).
# ──────────────────────────────────────────────────────────────────────────────

"""
    step_subproblem_multistep!(state, solver, subproblems, a, b, c)

Advance the state using an IMEX multistep method with coefficients `(a, b, c)`:

    Σ_k a[k] M X^{n+1-k} + Σ_k b[k] L X^{n+1-k} = Σ_k c[k] F^{n+1-k}

History is stored per subproblem and automatically trimmed to `length(a) - 1`
entries (the minimum needed for the next step). The caller is responsible for
falling back to a lower-order startup scheme when insufficient history exists.
"""
function step_subproblem_multistep!(
    state::TimestepperState, solver::InitialValueSolver,
    subproblems::Tuple, a::Tuple, b::Tuple, c::Tuple,
)
    problem = solver.problem

    # History depth — how many back steps we need to store for the NEXT call.
    # `a`, `b`, `c` contain the current-step value at index 1 (Julia 1-based),
    # and history at indices 2..end. We only need to store up to max(len-1).
    a_hist_depth = length(a) - 1
    b_hist_depth = length(b) - 1
    c_hist_depth = length(c) - 1
    max_depth = max(a_hist_depth, b_hist_depth, c_hist_depth, 0)

    state_fields = collect_state_fields(problem.variables)
    for f in state_fields
        ensure_layout!(f, :c)
    end

    n_sp = length(subproblems)

    # ── Per-subproblem history storage (cached in timestepper_data) ──────────
    # History entries are `AbstractVector{ComplexF64}` so they can hold either
    # regular `Vector` (CPU, CPU+MPI) or `CuArray`/`ROCArray` (GPU) as produced
    # by `similar_zeros` below. The outer shape is always `sp_idx → history
    # vector` and each rank only touches its own local subproblems under MPI.
    MX_hist = get!(state.timestepper_data, :sp_multistep_MX_hist) do
        [Vector{AbstractVector{ComplexF64}}() for _ in 1:n_sp]
    end::Vector{Vector{AbstractVector{ComplexF64}}}
    LX_hist = get!(state.timestepper_data, :sp_multistep_LX_hist) do
        [Vector{AbstractVector{ComplexF64}}() for _ in 1:n_sp]
    end::Vector{Vector{AbstractVector{ComplexF64}}}
    F_hist = get!(state.timestepper_data, :sp_multistep_F_hist) do
        [Vector{AbstractVector{ComplexF64}}() for _ in 1:n_sp]
    end::Vector{Vector{AbstractVector{ComplexF64}}}

    # ── Step 1: compute M*X_current, L*X_current, F_current per subproblem ──
    F_fields = evaluate_rhs(solver, state_fields, solver.sim_time)

    for (sp_idx, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue
        n = size(sp.M_min, 1)

        # Gather X in variable space — returns CPU or GPU vector matching the
        # state-field storage (handled by `_gather_field_raw!`).
        x_cur = gather_inputs(sp, state_fields)

        # Allocate fresh history-bound vectors matching `x_cur`'s backend
        # (CPU / GPU). We cannot reuse the step-scoped cached vectors here
        # because the history keeps references across steps.
        mx_cur = similar_zeros(x_cur, ComplexF64, n)
        lx_cur = similar_zeros(x_cur, ComplexF64, n)
        f_cur  = similar_zeros(x_cur, ComplexF64, n)

        # M*X_current and L*X_current in equation space
        M_op = _subproblem_operator(sp, :M, x_cur)
        L_op = _subproblem_operator(sp, :L, x_cur)
        _apply_subproblem_operator!(mx_cur, M_op, x_cur)
        _apply_subproblem_operator!(lx_cur, L_op, x_cur)

        # F_current in equation space (PDE rows only — BC rows handled below)
        gather_eqn_F!(f_cur, sp, solver, F_fields, state_fields)

        # Prepend into history (newest at index 1)
        pushfirst!(MX_hist[sp_idx], mx_cur)
        pushfirst!(LX_hist[sp_idx], lx_cur)
        pushfirst!(F_hist[sp_idx], f_cur)

        # Trim to needed depth
        while length(MX_hist[sp_idx]) > max(a_hist_depth, 1)
            pop!(MX_hist[sp_idx])
        end
        while length(LX_hist[sp_idx]) > max(b_hist_depth, 1)
            pop!(LX_hist[sp_idx])
        end
        while length(F_hist[sp_idx]) > max(c_hist_depth, 1)
            pop!(F_hist[sp_idx])
        end
    end

    # ── Step 2: check if we have enough history to advance ──────────────────
    # Need at least one of each history type (we just prepended the current
    # step, so depth 1 means "only the current step" which is enough for
    # backward-Euler / CNAB1-type methods that only use k=0 and k=1).
    have_enough = true
    for sp_idx in 1:n_sp
        subproblems[sp_idx].M_min === nothing && continue
        if length(F_hist[sp_idx]) < max(c_hist_depth, 1) ||
           length(MX_hist[sp_idx]) < max(a_hist_depth, 1) ||
           length(LX_hist[sp_idx]) < max(b_hist_depth, 1)
            have_enough = false
            break
        end
    end

    if !have_enough
        # Caller should have fallen back to startup scheme already; this is
        # a safety net that silently uses zero contributions for missing
        # history entries (giving a lower-order first step).
        @debug "step_subproblem_multistep!: insufficient history, using zero padding"
    end

    # ── Step 3: compute F_alg once per step for BC row override ─────────────
    # Allocated to match the F_hist backend (CPU or GPU) so downstream
    # broadcasts stay on a single device.
    ALG_F = Vector{Any}(undef, n_sp)
    for (sp_idx, sp) in enumerate(subproblems)
        if sp.M_min === nothing
            ALG_F[sp_idx] = ComplexF64[]
            continue
        end
        n = size(sp.M_min, 1)
        # Use an existing history entry as the backend reference.
        ref = isempty(F_hist[sp_idx]) ? nothing : F_hist[sp_idx][1]
        alg_f = ref === nothing ? zeros(ComplexF64, n) :
                                  similar_zeros(ref, ComplexF64, n)
        gather_alg_F!(alg_f, sp)
        ALG_F[sp_idx] = alg_f
    end

    # ── Step 4: build RHS and solve per subproblem ──────────────────────────
    for (sp_idx, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue
        n = size(sp.M_min, 1)

        # rhs matches the history backend (CPU or GPU).
        ref = isempty(F_hist[sp_idx]) ? nothing : F_hist[sp_idx][1]
        rhs = ref === nothing ? zeros(ComplexF64, n) :
                                similar_zeros(ref, ComplexF64, n)

        # Accumulate RHS = Σ_{k≥1} [c[k]*F_k - a[k]*MX_k - b[k]*LX_k]
        # (k=1 in math maps to index 2 in the Julia 1-based tuple; our history
        # vectors are 1-indexed with the newest entry at position 1, which
        # corresponds to the "k=1 step back" — one call cycle behind us.)
        for k in 2:length(c)
            ck = c[k]
            abs(ck) < 1e-14 && continue
            hist_idx = k - 1  # newest history entry (step n) is at index 1
            if hist_idx <= length(F_hist[sp_idx])
                @. rhs += ck * F_hist[sp_idx][hist_idx]
            end
        end
        for k in 2:length(a)
            ak = a[k]
            abs(ak) < 1e-14 && continue
            hist_idx = k - 1
            if hist_idx <= length(MX_hist[sp_idx])
                @. rhs -= ak * MX_hist[sp_idx][hist_idx]
            end
        end
        for k in 2:length(b)
            bk = b[k]
            abs(bk) < 1e-14 && continue
            hist_idx = k - 1
            if hist_idx <= length(LX_hist[sp_idx])
                @. rhs -= bk * LX_hist[sp_idx][hist_idx]
            end
        end

        # Override BC rows: rhs[bc] = b[0] * F_alg[bc] so that
        # b[0]*L_row*X_new = b[0]*F_alg → L_row*X_new = F_alg.
        # Vectorized (CUDA.allowscalar(false)-safe) via `apply_bc_override!`.
        b0 = b[1]
        if abs(b0) > 1e-14
            apply_bc_override!(rhs, ALG_F[sp_idx], sp, b0)
        end

        # Solve (a[0]*M + b[0]*L) * X_new = rhs
        a0 = a[1]
        lhs_solver = _get_or_build_multistep_lhs!(sp, a0, b0)
        if lhs_solver === nothing
            @warn "step_subproblem_multistep!: LHS factorization failed for group=$(sp.group); skipping step" maxlog=1
            continue
        end
        x_new = _solve_cached_system(lhs_solver, rhs)
        scatter_inputs(sp, x_new, state_fields)
    end

    # ── Step 5: push new state to history ───────────────────────────────────
    new_state = collect_state_fields(problem.variables)
    _push_trim!(state.history, new_state, 1)
end

"""
    _get_or_build_multistep_lhs!(sp, a0, b0)

Return a cached solver for `(a0*M_min + b0*L_min)`, building it lazily and
invalidating when `(a0, b0)` changes (e.g., under variable-dt CNAB2).
"""
function _get_or_build_multistep_lhs!(sp::Subproblem, a0::Float64, b0::Float64)
    M = sp.M_min
    L = sp.L_min
    M === nothing && return nothing

    cache_key = "_sp_multistep_lhs"
    cache_key_k = "_sp_multistep_lhs_key"
    cached = get(sp.matrices, cache_key, nothing)
    cached_k = get(sp.matrices, cache_key_k, nothing)

    current_key = (a0, b0)
    if cached !== nothing && cached_k === current_key
        return cached
    end

    LHS = if L !== nothing && abs(b0) > 1e-14
        ComplexF64(a0) * M + ComplexF64(b0) * L
    else
        ComplexF64(a0) * M
    end

    solver_type = _subproblem_solver_type(sp.solver.base.matsolver)
    solver_kwargs = _subproblem_solver_kwargs(sp.solver.base.matsolver)

    try
        lhs_solver = MatSolvers.solver_instance(solver_type, LHS; solver_kwargs...)
        sp.matrices[cache_key] = lhs_solver
        sp.matrices[cache_key_k] = current_key
        return lhs_solver
    catch err
        if solver_type != MatSolvers.SPQRSolver
            try
                qr_solver = MatSolvers.solver_instance(MatSolvers.SPQRSolver, LHS)
                sp.matrices[cache_key] = qr_solver
                sp.matrices[cache_key_k] = current_key
                return qr_solver
            catch
            end
        end
        @debug "multistep LHS build failed for group=$(sp.group)" exception=(err, catch_backtrace())
    end

    # Dense fallback
    LHS_dense = Matrix(LHS)
    sp.matrices[cache_key] = LHS_dense
    sp.matrices[cache_key_k] = current_key
    return LHS_dense
end

# ──────────────────────────────────────────────────────────────────────────────
# Coefficient builders for concrete multistep schemes
# ──────────────────────────────────────────────────────────────────────────────

"""
    _sp_multistep_history_depth(state)

Return the minimum F-history depth across all subproblems stored in
`timestepper_data`. Callers use this to decide whether a higher-order
multistep method has enough history to advance, or whether to fall back to a
lower-order startup scheme.
"""
function _sp_multistep_history_depth(state::TimestepperState)
    F_hist = get(state.timestepper_data, :sp_multistep_F_hist, nothing)
    F_hist === nothing && return 0
    depth = typemax(Int)
    any_counted = false
    for h in F_hist
        isempty(h) && continue
        depth = min(depth, length(h))
        any_counted = true
    end
    return any_counted ? depth : 0
end

"""CNAB1 coefficients for fixed timestep `dt`."""
@inline _cnab1_coefs(dt::Float64) =
    ((1.0/dt, -1.0/dt),    # a
     (0.5, 0.5),           # b
     (0.0, 1.0))           # c  (c[1] = current, c[2] = F_{n-0} → 1*F_n)

"""CNAB2 coefficients for current/previous timestep (variable step)."""
@inline function _cnab2_coefs(dt::Float64, dt_prev::Float64)
    w1 = dt / dt_prev
    a = ((1.0 + 2.0*w1) / ((1.0 + w1) * dt),
         -(1.0 + w1) / dt,
         w1^2 / ((1.0 + w1) * dt))
    b = (0.5, 0.5, 0.0)
    c = (0.0, 1.0 + w1/2.0, -w1/2.0)
    return a, b, c
end

"""SBDF1 coefficients (backward Euler, 1st-order explicit)."""
@inline _sbdf1_coefs(dt::Float64) =
    ((1.0/dt, -1.0/dt),
     (1.0,),
     (0.0, 1.0))

"""SBDF2 coefficients."""
@inline _sbdf2_coefs(dt::Float64) =
    ((1.5/dt, -2.0/dt, 0.5/dt),
     (1.0,),
     (0.0, 2.0, -1.0))

"""SBDF3 coefficients."""
@inline _sbdf3_coefs(dt::Float64) =
    ((11.0/(6.0*dt), -3.0/dt, 1.5/dt, -1.0/(3.0*dt)),
     (1.0,),
     (0.0, 3.0, -3.0, 1.0))

"""SBDF4 coefficients."""
@inline _sbdf4_coefs(dt::Float64) =
    ((25.0/(12.0*dt), -4.0/dt, 3.0/dt, -4.0/(3.0*dt), 0.25/dt),
     (1.0,),
     (0.0, 4.0, -6.0, 4.0, -1.0))
