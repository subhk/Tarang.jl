# ============================================================================
# Explicit-multistep FIELD path.
#
# The global-matrix multistep solve is
#     (a[1]·M + b[1]·L)·X_new = Σ_k c[k+1]·F[k] − Σ_k a[k+1]·M·X[k] − Σ_k b[k+1]·L·X[k]
# When the problem carries no implicit linear operator the mass operator is the
# identity and L vanishes identically, so every L term drops and the M terms
# become the states themselves:
#     X_new = (Σ_k c[k+1]·F[k] − Σ_k a[k+1]·X[k]) / a[1]
# a linear combination of stored states and right-hand sides. No global matrix,
# no factorization, no host staging — it runs wherever the field data lives.
#
# WHY THIS EXISTS. A pure-Fourier GPU solver (and an MPI pure-Fourier solver)
# assembles no global matrices and builds no subproblems. Every multistep method
# therefore reached `_prepare_global_matrices!`, found `L_matrix === nothing`,
# emitted one `maxlog=1` warning and fell back: CNAB2 → CNAB1 → forward Euler,
# SBDF4 → SBDF3 → … → forward Euler, on every step for the rest of the run. The
# nominal order 2/3/4 silently became order 1 (measured: ~285× more error than
# the CPU path at dt=0.02). The matrices were never the missing ingredient —
# they encoded `L = 0` and `M = I`, which this path knows structurally.
#
# The same reduction is what `_step_explicit_rk_gpu!` does for Runge-Kutta; this
# is its multistep counterpart, and it shares that path's contract: the mass
# operator is taken to be the identity (true for the Fourier bases that reach
# here), and algebraic variables are restored by `_refresh_algebraic_state!`.
#
# GATED, never a silent substitution. A problem WITH an implicit linear operator
# must not come here — dropping L would integrate the equations without their
# stiff terms. `_explicit_multistep_field_eligible` refuses those, leaving the
# existing loud refusals (`_check_gpu_implicit_compatibility!`, the distributed
# fallback throw) to name the working alternative.
# ============================================================================

"""Deepest state/RHS history a method's own coefficient formula reads."""
function _explicit_multistep_depth(method::Symbol)
    method === :cnab1 && return 1
    method === :sbdf1 && return 1
    method === :cnab2 && return 2
    method === :sbdf2 && return 2
    method === :sbdf3 && return 3
    method === :sbdf4 && return 4
    throw(ArgumentError("unknown explicit-multistep method $method"))
end

"""
    _explicit_multistep_field_eligible(solver) -> Bool

True when the multistep update can be taken as a matrix-free field combination:
no per-mode subproblems (those already solve the implicit part) and no implicit
linear operator to drop.
"""
function _explicit_multistep_field_eligible(solver::InitialValueSolver)
    _timestepper_subproblems(solver) === nothing || return false
    return !_problem_has_implicit_linear_term(solver)
end

"""
    _try_step_explicit_multistep_field!(state, solver, method) -> Bool

Take the field path when the architecture needs it (GPU or MPI PencilArrays,
where no global matrix exists) and the problem is genuinely explicit. Returns
`false` without touching `state` when either condition fails, so callers fall
through to their existing behaviour.
"""
function _try_step_explicit_multistep_field!(state::TimestepperState,
                                             solver::InitialValueSolver,
                                             method::Symbol)
    _distributed_field_path_required(state.history[end]) || return false
    _explicit_multistep_field_eligible(solver) || return false
    _step_explicit_multistep_field!(state, solver, method)
    return true
end

"""Return (and lazily create) one of the field-path history deques, newest first."""
function _explicit_multistep_history!(state::TimestepperState, key::Symbol,
                                      template::V) where {V<:Vector{<:ScalarField}}
    cached = get(state.timestepper_data, key, nothing)
    cached isa Vector{V} && return cached
    fresh = V[]
    state.timestepper_data[key] = fresh
    return fresh
end

"""Push `value` onto the front of `hist`, recycling the dropped tail buffer."""
function _explicit_multistep_prepend!(hist::Vector{V}, template::V, value::V,
                                      depth::Int) where {V<:Vector{<:ScalarField}}
    buffer = length(hist) >= depth ? pop!(hist) : copy_state(template)
    _copy_field_state!(buffer, value)
    pushfirst!(hist, buffer)
    return hist
end

"""
True while `method` still needs a higher-order self-start. Only SBDF3/SBDF4 do:
CNAB2 and SBDF2 bridge with their own first-order member instead, exactly as the
global-matrix path does, so the two agree step for step. A first-order start would
cap SBDF3/SBDF4's global order at 1.
"""
function _explicit_multistep_needs_seeding(state::TimestepperState, method::Symbol,
                                           depth::Int)
    method === :sbdf3 && return depth < 3 || length(state.dt_history) < 3
    method === :sbdf4 && return depth < 4 || length(state.dt_history) < 4
    return false
end

"""
Coefficients for one field-path step. Only called once `_explicit_multistep_needs_seeding`
is false, so every branch returns a concrete tuple pair.
"""
function _explicit_multistep_coefficients(state::TimestepperState, method::Symbol,
                                          depth::Int)
    dt = state.dt
    if method === :cnab1 || (method === :cnab2 && depth < 2)
        a, _, c = _cnab1_coefs(dt)
        return a, c
    elseif method === :sbdf1 || (method === :sbdf2 && depth < 2)
        a, _, c = _sbdf1_coefs(dt)
        return a, c
    elseif method === :cnab2
        a, _, c = _cnab2_coefs(dt, get_previous_timestep(state))
        return a, c
    elseif method === :sbdf2
        a, _, c = _sbdf2_coefs(dt, get_previous_timestep(state))
        return a, c
    elseif method === :sbdf3
        k2 = state.dt_history[end]
        k1 = state.dt_history[end-1]
        k0 = state.dt_history[end-2]
        a, _, c = _sbdf3_coefs(k2, k1, k0)
        return a, c
    elseif method === :sbdf4
        k3 = state.dt_history[end]
        k2 = state.dt_history[end-1]
        k1 = state.dt_history[end-2]
        k0 = state.dt_history[end-3]
        a, _, c = _sbdf4_coefs(k3, k2, k1, k0)
        return a, c
    end
    throw(ArgumentError("unknown explicit-multistep method $method"))
end

"""
Apply one field-space multistep update. A function barrier: `a` and `c` arrive
from a branch that can yield several tuple lengths, so specializing here keeps the
arithmetic and the history indexing concretely typed.
"""
function _explicit_multistep_apply!(state::TimestepperState, solver::InitialValueSolver,
                                    current_state::V, X_history::Vector{V},
                                    F_history::Vector{V}, a::NTuple{NA,Float64},
                                    c::NTuple{NC,Float64}) where {V<:Vector{<:ScalarField},NA,NC}
    inv_a0 = 1.0 / a[1]
    new_state = _acquire_recycled_history_state!(state, :explicit_field_ms_recycle,
                                                 current_state)
    # Seed with the two terms every method has, then accumulate the deeper levels.
    linear_combination_state!(new_state, c[2] * inv_a0, F_history[1],
                              -a[2] * inv_a0, X_history[1])
    @inbounds for k in 2:min(NC - 1, length(F_history))
        c[k+1] == 0 && continue
        axpy_state!(c[k+1] * inv_a0, F_history[k], new_state)
    end
    @inbounds for k in 2:min(NA - 1, length(X_history))
        a[k+1] == 0 && continue
        axpy_state!(-a[k+1] * inv_a0, X_history[k], new_state)
    end

    _refresh_algebraic_state!(solver.problem, new_state)
    _push_recycled_history_state!(state, :explicit_field_ms_recycle, new_state)
    return nothing
end

"""
    _step_explicit_multistep_field!(state, solver, method)

Advance one step of `method` as a field-space linear combination. Ungated: the
caller is responsible for checking `_explicit_multistep_field_eligible`. Tests
drive this directly on CPU, where the global-matrix path is the reference.
"""
function _step_explicit_multistep_field!(state::TimestepperState,
                                         solver::InitialValueSolver,
                                         method::Symbol)
    current_state = state.history[end]
    depth = _explicit_multistep_depth(method)

    X_history = _explicit_multistep_history!(state, :explicit_field_ms_X, current_state)
    F_history = _explicit_multistep_history!(state, :explicit_field_ms_F, current_state)

    # Record the current state and its right-hand side at the head of both deques
    # BEFORE choosing coefficients, so a seeding step still contributes history
    # (this is what `_multistep_rk443_startup!` does on the global path).
    F_current = evaluate_rhs(solver, current_state, solver.sim_time)
    _explicit_multistep_prepend!(F_history, current_state, F_current, depth)
    _release_rhs_buffer!(F_current, solver)
    _explicit_multistep_prepend!(X_history, current_state, current_state, depth)

    if _explicit_multistep_needs_seeding(state, method, length(F_history))
        # Order-3 self-start. `step_rk_imex!` routes to the field-native explicit
        # RK on exactly the architectures that reach here.
        step_rk_imex!(state, solver; ts=_RK443_SINGLETON)
        return nothing
    end

    a, c = _explicit_multistep_coefficients(state, method, length(F_history))
    _explicit_multistep_apply!(state, solver, current_state, X_history, F_history, a, c)
    return nothing
end
