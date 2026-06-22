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
#
# **Per-step allocation**: history is stored in fixed-capacity `SPHistoryRing`
# circular buffers, pre-allocated on first use and reused thereafter. The only
# allocations per step are the small cached vectors managed by
# `_subproblem_cached_vector!`, which are also reused across steps. Expected
# per-step allocation for steady-state runs: **zero**.
# ──────────────────────────────────────────────────────────────────────────────

"""
    SPHistoryRing

Fixed-capacity circular buffer for per-subproblem multistep history. Stores
`capacity` pre-allocated `AbstractVector{ComplexF64}` slots and maintains a
`head` index pointing at the newest entry. Allows O(1) replacement
(`ring_push_newest!`) and O(1) age-indexed access (`ring_get`) without any
per-step allocation after the initial slot allocation.

Used by `step_subproblem_multistep!` to store `M*X`, `L*X`, and `F` history
per subproblem across timesteps. Replaces the earlier `Vector{Vector}` +
`pushfirst!` / `pop!` implementation which allocated three fresh vectors per
subproblem per step.
"""
mutable struct SPHistoryRing
    buffers::Vector{AbstractVector{ComplexF64}}  # Pre-allocated slots (length = capacity)
    head::Int                                     # 1-based slot of newest entry, or 0 if empty
    count::Int                                    # Number of valid entries, 0 ≤ count ≤ capacity
    capacity::Int
end

SPHistoryRing(capacity::Int) = SPHistoryRing(
    Vector{AbstractVector{ComplexF64}}(undef, 0),
    0, 0, max(capacity, 1),
)

"""Return the k-th newest entry (k=1 is newest), or `nothing` if `k > count`.

Mapping from age `k` to slot: with `head` pointing at the newest slot, age
`k` lives at `mod1(head - k + 1, capacity)`.
"""
@inline function ring_get(ring::SPHistoryRing, k::Int)
    (k < 1 || k > ring.count) && return nothing
    slot = mod1(ring.head - k + 1, ring.capacity)
    return ring.buffers[slot]
end

"""
    ring_push_newest!(ring, reference) -> AbstractVector

Advance the ring's head and return the underlying slot for the newest
entry so the caller can **write into it directly** (zero-copy). The caller
must not retain the returned reference after the next `ring_push_newest!`
call — it may be overwritten when the ring wraps.

On first use, allocates `capacity` slots of length `length(reference)`
using `similar_zeros(reference, ComplexF64, length(reference))`, so the
buffer backend (CPU / CuArray / ROCArray) matches the reference's backend.
"""
@inline function ring_push_newest!(ring::SPHistoryRing, reference::AbstractVector)
    if isempty(ring.buffers)
        n = length(reference)
        ring.buffers = [similar_zeros(reference, ComplexF64, n) for _ in 1:ring.capacity]
    end
    ring.head = mod1(ring.head + 1, ring.capacity)
    ring.count = min(ring.count + 1, ring.capacity)
    return ring.buffers[ring.head]
end

"""Grow a ring's capacity (e.g. when the user switches from SBDF2 to SBDF4
mid-run). The existing entries are preserved, with new empty slots added
at the back. A no-op if the requested capacity is not larger than the
current capacity."""
function ring_resize!(ring::SPHistoryRing, new_capacity::Int)
    new_capacity <= ring.capacity && return ring
    if isempty(ring.buffers)
        ring.capacity = new_capacity
        return ring
    end
    # Copy entries age-sorted (newest first) into a new slot array.
    old_cap = ring.capacity
    old_count = ring.count
    new_bufs = similar(ring.buffers, new_capacity)
    for k in 1:old_count
        old_slot = mod1(ring.head - k + 1, old_cap)
        new_slot = mod1(new_capacity - k + 1, new_capacity)
        new_bufs[new_slot] = ring.buffers[old_slot]
    end
    # Any remaining new slots point at scratch buffers we'll create on demand;
    # mark them by reusing the first allocated buffer (shape reference).
    ref = ring.buffers[1]
    n = length(ref)
    for i in 1:new_capacity
        if !isassigned(new_bufs, i)
            new_bufs[i] = similar_zeros(ref, ComplexF64, n)
        end
    end
    ring.buffers = new_bufs
    ring.capacity = new_capacity
    ring.head = new_capacity  # newest entry ended up at slot `new_capacity`
    return ring
end

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

    # Distributor handle for the mixed Fourier–Chebyshev solve-layout transpose
    # (to_solve_layout!/from_solve_layout! are no-ops outside distributed
    # mixed-basis runs). Same bracketing pattern as step_subproblem_rk!.
    dist = length(subproblems) >= 1 ? subproblems[1].dist : nothing

    n_sp = length(subproblems)

    # ── Per-subproblem ring-buffer history (cached in timestepper_data) ──────
    # We pre-allocate `max_capacity` slots per ring on first use. Subsequent
    # steps reuse those slots via `ring_push_newest!` (zero per-step
    # allocation). If the user switches to a higher-order method mid-run
    # (e.g. SBDF2 → SBDF4), we `ring_resize!` to grow capacity while
    # preserving existing history.
    max_capacity = max(a_hist_depth, b_hist_depth, c_hist_depth, 1)
    MX_rings = get!(state.timestepper_data, :sp_multistep_MX_rings) do
        [SPHistoryRing(max_capacity) for _ in 1:n_sp]
    end::Vector{SPHistoryRing}
    LX_rings = get!(state.timestepper_data, :sp_multistep_LX_rings) do
        [SPHistoryRing(max_capacity) for _ in 1:n_sp]
    end::Vector{SPHistoryRing}
    F_rings = get!(state.timestepper_data, :sp_multistep_F_rings) do
        [SPHistoryRing(max_capacity) for _ in 1:n_sp]
    end::Vector{SPHistoryRing}

    # Grow any ring whose capacity is now smaller than the current method's
    # requirement (only happens on a timestepper-type switch).
    @inbounds for i in 1:n_sp
        MX_rings[i].capacity < max_capacity && ring_resize!(MX_rings[i], max_capacity)
        LX_rings[i].capacity < max_capacity && ring_resize!(LX_rings[i], max_capacity)
        F_rings[i].capacity  < max_capacity && ring_resize!(F_rings[i],  max_capacity)
    end

    # ── Step 1: compute M*X_current, L*X_current, F_current per subproblem ──
    # Write directly into ring slots — no per-step allocation.
    F_fields = evaluate_rhs_buffered(solver, state_fields, solver.sim_time)

    # Solve layout for the per-mode gather (state + F coeffs). One collective
    # transpose per field list, OUTSIDE the subproblem loop. F_fields are
    # read-only here → restored by pointer swap only (no transpose back).
    _ms_g_state = to_solve_layout!(state_fields, dist)
    _ms_g_F     = to_solve_layout!(F_fields, dist)

    for (sp_idx, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue
        n = size(sp.M_min, 1)

        # Gather X in variable space — CPU or GPU matching state-field storage.
        x_cur = gather_inputs(sp, state_fields)

        # Advance each ring and obtain the newest-slot buffer to write into.
        mx_cur = ring_push_newest!(MX_rings[sp_idx], x_cur)
        lx_cur = ring_push_newest!(LX_rings[sp_idx], x_cur)
        f_cur  = ring_push_newest!(F_rings[sp_idx],  x_cur)

        # M*X_current and L*X_current in equation space
        M_op = _subproblem_operator(sp, :M, x_cur)
        L_op = _subproblem_operator(sp, :L, x_cur)
        _apply_subproblem_operator!(mx_cur, M_op, x_cur)
        _apply_subproblem_operator!(lx_cur, L_op, x_cur)

        # F_current in equation space (PDE rows only — BC rows handled below)
        gather_eqn_F!(f_cur, sp, solver, F_fields, state_fields)
    end

    # Restore FFT layout for state (transpose back) and F (pointer swap only —
    # its solved values are discarded; the pencil must be the FFT pencil for the
    # next step's buffered RHS transform).
    from_solve_layout!(_ms_g_state, dist)
    for (f, fft_pa) in _ms_g_F
        set_coeff_data!(f, fft_pa)
    end

    # ── Step 2: check if we have enough history to advance ──────────────────
    # Need at least one of each history type (we just pushed current, so
    # count ≥ 1). The `ring_get` calls in the RHS loop return `nothing` for
    # missing entries, which the accumulator silently skips — giving a
    # lower-order first step if the caller's startup dispatch didn't fall
    # back. This is the same behavior as the old `Vector{Vector}` path.
    have_enough = true
    for sp_idx in 1:n_sp
        subproblems[sp_idx].M_min === nothing && continue
        if F_rings[sp_idx].count  < max(c_hist_depth, 1) ||
           MX_rings[sp_idx].count < max(a_hist_depth, 1) ||
           LX_rings[sp_idx].count < max(b_hist_depth, 1)
            have_enough = false
            break
        end
    end
    if !have_enough
        @debug "step_subproblem_multistep!: insufficient history, using zero padding"
    end

    # ── Step 3: compute F_alg once per step for BC row override ─────────────
    # Uses cached per-subproblem stage vectors — no per-step allocation.
    ALG_F = _subproblem_vector_slots(n_sp)
    for (sp_idx, sp) in enumerate(subproblems)
        if sp.M_min === nothing
            ALG_F[sp_idx] = ComplexF64[]
            continue
        end
        n = size(sp.M_min, 1)
        ref = ring_get(F_rings[sp_idx], 1)  # always valid after step 1
        alg_f = ref === nothing ? zeros(ComplexF64, n) :
                                  _sp_stage_vector!(sp, :multistep_alg_f, n, ref)
        gather_alg_F!(alg_f, sp)
        ALG_F[sp_idx] = alg_f
    end

    # ── Step 4: build RHS and solve per subproblem ──────────────────────────
    # Solve layout for the per-mode scatter (writes X_new back). One collective
    # transpose OUTSIDE the loop; restored before _push_trim! reads grid space.
    _ms_s_state = to_solve_layout!(state_fields, dist)
    for (sp_idx, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue
        n = size(sp.M_min, 1)

        # Use a cached rhs buffer; reused across steps, matches ring backend.
        ref = ring_get(F_rings[sp_idx], 1)
        rhs = ref === nothing ? zeros(ComplexF64, n) :
                                _sp_stage_vector!(sp, :multistep_rhs, n, ref)
        fill!(rhs, zero(ComplexF64))

        # Accumulate RHS = Σ_{k≥1} [c[k]*F_k - a[k]*MX_k - b[k]*LX_k]
        # (k=1 in math maps to Julia tuple index 2; our rings are age-indexed
        # with age 1 = current step, age 2 = one step back, etc.)
        for k in 2:length(c)
            ck = c[k]
            abs(ck) < 1e-14 && continue
            entry = ring_get(F_rings[sp_idx], k - 1)
            entry === nothing && continue
            @. rhs += ck * entry
        end
        for k in 2:length(a)
            ak = a[k]
            abs(ak) < 1e-14 && continue
            entry = ring_get(MX_rings[sp_idx], k - 1)
            entry === nothing && continue
            @. rhs -= ak * entry
        end
        for k in 2:length(b)
            bk = b[k]
            abs(bk) < 1e-14 && continue
            entry = ring_get(LX_rings[sp_idx], k - 1)
            entry === nothing && continue
            @. rhs -= bk * entry
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
        x_new = _sp_stage_vector!(sp, :multistep_sol, size(sp.M_min, 2), ref)
        _solve_cached_system!(x_new, lhs_solver, rhs)
        scatter_inputs(sp, x_new, state_fields)
    end

    # Restore FFT layout before the grid-space history push below.
    from_solve_layout!(_ms_s_state, dist)

    # ── Step 5: push new state to history ───────────────────────────────────
    _push_trim!(state.history, state_fields, 1)
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

    # Build the new LHS matrix. Under variable-dt multistep the sparsity
    # pattern is unchanged — only `nzval` differs — so we can refactor the
    # cached `SparseLUSolver` in place and reuse its symbolic analysis.
    LHS = if L !== nothing && abs(b0) > 1e-14
        ComplexF64(a0) * M + ComplexF64(b0) * L
    else
        ComplexF64(a0) * M
    end

    # Symbolic-reuse fast path: if the cached solver supports
    # `refactor!` (CPU SparseLUSolver, GPU CuSparseLU :rf backend, or
    # any future solver that opts in), call it to reuse the symbolic
    # factorization. Detected via `hasmethod` so the GPU path doesn't
    # need a hard dependency on CUDA.jl being loaded.
    if cached !== nothing && cached_k !== nothing &&
       hasmethod(MatSolvers.refactor!, Tuple{typeof(cached), typeof(LHS)})
        try
            MatSolvers.refactor!(cached, LHS)
            sp.matrices[cache_key_k] = current_key
            return cached
        catch err
            @debug "multistep refactor! failed, rebuilding from scratch" exception=(err, catch_backtrace())
            # fall through to the full-rebuild path below.
        end
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
    # Prefer the ring-buffer storage (new path). Fall back to the old
    # Vector{Vector} storage if a legacy cache is still present (e.g.,
    # mid-session after a code reload).
    F_rings = get(state.timestepper_data, :sp_multistep_F_rings, nothing)
    if F_rings !== nothing
        depth = typemax(Int)
        any_counted = false
        for ring in F_rings
            ring.count == 0 && continue
            depth = min(depth, ring.count)
            any_counted = true
        end
        return any_counted ? depth : 0
    end
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
    a = (1.0/dt, -1.0/dt)
    b = (0.5, 0.5)
    c = (0.0, 1.0 + w1/2.0, -w1/2.0)
    return a, b, c
end

"""SBDF1 coefficients (backward Euler, 1st-order explicit)."""
@inline _sbdf1_coefs(dt::Float64) =
    ((1.0/dt, -1.0/dt),
     (1.0,),
     (0.0, 1.0))

"""SBDF2 coefficients."""
@inline _sbdf2_coefs(dt::Float64) = _sbdf2_coefs(dt, dt)
@inline function _sbdf2_coefs(dt::Float64, dt_prev::Float64)
    w1 = dt / dt_prev
    a = ((1.0 + 2.0*w1) / ((1.0 + w1) * dt),
         -(1.0 + w1) / dt,
         w1^2 / ((1.0 + w1) * dt))
    b = (1.0,)
    c = (0.0, 1.0 + w1, -w1)
    return a, b, c
end

"""SBDF3 coefficients."""
@inline _sbdf3_coefs(dt::Float64) = _sbdf3_coefs(dt, dt, dt)
@inline function _sbdf3_coefs(k2::Float64, k1::Float64, k0::Float64)
    w2 = k2 / k1
    w1 = k1 / k0
    a = ((1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2,
         (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2,
         w2^2 * (w1 + 1/(1 + w2)) / k2,
         -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2)
    b = (1.0, 0.0, 0.0, 0.0)
    c = (0.0,
         (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1),
         -w2*(1 + w1*(1 + w2)),
         w1*w1*w2*(1 + w2) / (1 + w1))
    return a, b, c
end

"""SBDF4 coefficients."""
@inline _sbdf4_coefs(dt::Float64) = _sbdf4_coefs(dt, dt, dt, dt)
@inline function _sbdf4_coefs(k3::Float64, k2::Float64, k1::Float64, k0::Float64)
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0
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
    return a, b, c
end
