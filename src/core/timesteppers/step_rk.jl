# ============================================================================
# Top-level IMEX Runge-Kutta stepping.
#
# This file decides which RK runtime path is taken:
# - preferred path: per-subproblem sparse solves in `step_subproblem_rk.jl`
# - fallback path: explicit treatment when GPU/MPI lacks subproblems
# - legacy path: global-matrix factorization when no subproblems are present
#
# Sign convention: `L_matrix` uses the LHS form `M*dX/dt + L*X = F`, so the
# stage solve is `(M + dt*a*L) * X_s = RHS` and the implicit contribution
# `L*X` is subtracted on the RHS accumulation.
#
# This matches the multistep IMEX methods. ETD methods instead work with the
# RHS-form operator `-M^{-1}L`.
# ============================================================================

# Function barrier: recover concrete factorization type F from Dict{...,Any} so
# ldiv! inside is statically dispatched. Eliminates allocation from lhs \ rhs.
@inline function _rk_ldiv!(dest::AbstractVector, lhs::F, rhs::AbstractVector) where {F}
    ldiv!(dest, lhs, rhs)
    return dest
end

@inline function _axpy_complex_vector!(y::AbstractVector{ComplexF64},
                                       scale,
                                       x::AbstractVector{ComplexF64})
    @inbounds for i in eachindex(y, x)
        y[i] += scale * x[i]
    end
    return y
end

"""
    step_rk_imex!(state, solver; ts=state.timestepper) -> nothing

Advance the solution one step with an IMEX Runge-Kutta scheme: the explicit
tableau `(A_explicit, b_explicit)` treats `F(X)` (advection, forcing) while the
implicit tableau `(A_implicit, b_implicit)` treats `L*X` (stiff diffusion).

Picks a runtime path in priority order (see the file header):
per-subproblem sparse solves → explicit fallback (GPU/MPI without subproblems,
or no `L_matrix`) → global-matrix factorization. Each stage `s` solves
`(M + dt*a_ss*L) X_s = M*X_n + dt Σ_{j<s}(a^E_{sj} F_j - a^I_{sj} L X_j)`; the
new state is then recovered from a final mass-matrix solve. Handles the three
mass-matrix regimes — absent (`I + dt a L`), regular, and singular/DAE (where
`L` fills the constraint rows). Mutates `state` in place (pushes the new state
onto `state.history`).
"""
function step_rk_imex!(state::TimestepperState, solver::InitialValueSolver; ts::TimeStepper=state.timestepper)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = ts.stages

    # Extract Butcher tableaux
    A_exp = ts.A_explicit
    A_imp = ts.A_implicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit  # Stage times (same for both tableaux)

    # Check if L_matrix is available for implicit treatment
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")

    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        step_subproblem_rk!(state, solver, sps)
        return nothing
    end

    fallback_reason = _imex_rk_explicit_fallback_reason(state, solver, current_state, L_matrix)
    if fallback_reason !== nothing
        # MPI pure-Fourier has no subproblems and no global L_matrix, so the IMEX
        # solve can't run as a matrix solve — but the linear operator is diagonal
        # in Fourier space, so treat it per-mode instead of degrading to a fully
        # explicit step (which blows up on stiff νΔ⁴/μ → high-wavenumber noise).
        if _distributed_diagonal_imex_applicable(solver)
            step_distributed_diagonal_imex_rk!(state, solver, ts)
            return nothing
        end
        _log_imex_rk_explicit_fallback(fallback_reason)
        _step_rk_imex_explicit_fallback!(state, solver, ts)
        return nothing
    end

    # Mass matrix handling: distinguish "no M" from "singular M (DAE)".
    # For DAE systems (constraint equations with zero M rows), we must still
    # use M in the stage factorizations so that (M + dt*a*L) correctly enforces
    # algebraic constraints (e.g., div(u) = 0, BCs, gauge conditions).
    # _get_mass_factor! returns nothing for singular M (cached after first call).
    has_mass = M_matrix !== nothing
    M_factor = has_mass ? _get_mass_factor!(state, M_matrix) : nothing
    M_is_singular = has_mass && M_factor === nothing

    _ensure_coeff_layout!(current_state)
    vector_size = _fields_vector_size(current_state)
    X_n_vec = _timestep_vector_buffer!(state, :imex_rk_X_n_vec, vector_size)
    fields_to_vector!(X_n_vec, current_state)

    if has_mass
        MX_n_vec = _timestep_vector_buffer!(state, :imex_rk_MX_n_vec, vector_size)
        mul!(MX_n_vec, M_matrix, X_n_vec)
    else
        MX_n_vec = X_n_vec
    end

    F_exp_vecs = _timestep_stage_vectors!(state, :imex_rk_F_exp_vecs, stages, vector_size)
    F_imp_vecs = _timestep_stage_vectors!(state, :imex_rk_F_imp_vecs, stages, vector_size)
    rhs_vec = _timestep_vector_buffer!(state, :imex_rk_rhs_vec, vector_size)
    Xs_vec = _timestep_vector_buffer!(state, :imex_rk_Xs_vec, vector_size)
    # Cache LHS factorizations across timesteps. Key is (dt, a_ii) so the cache
    # automatically invalidates when dt changes (adaptive stepping).
    lhs_cache = get!(state.timestepper_data, :imex_rk_lhs_cache) do
        Dict{Tuple{Float64, Float64}, Any}()
    end::Dict{Tuple{Float64, Float64}, Any}

    # Loop over stages
    # Each stage solves: (M + dt*a_ss*L) * X_s = M*X_n + dt*Σ_{j<s}(a^E*F - a^I*L*X)
    @inbounds for s in 1:stages
        state.current_substep = s

        copyto!(rhs_vec, MX_n_vec)

        # Accumulate explicit and implicit contributions from previous stages
        for j in 1:(s-1)
            a_exp_sj = dt * A_exp[s, j]
            a_imp_sj = dt * A_imp[s, j]
            if abs(a_exp_sj) > 1e-14
                @. rhs_vec += a_exp_sj * F_exp_vecs[j]
            end
            # Subtract implicit contributions (L on LHS)
            if abs(a_imp_sj) > 1e-14
                @. rhs_vec -= a_imp_sj * F_imp_vecs[j]
            end
        end

        # Implicit solve: (M + dt*a*L) * X = rhs
        # When M is absent, uses (I + dt*a*L). When M is singular (DAE),
        # (M + dt*a*L) is still non-singular because L fills constraint rows.
        a_ii = A_imp[s, s]
        if !has_mass
            if abs(a_ii) < 1e-14
                copyto!(Xs_vec, rhs_vec)
            else
                lhs = get!(lhs_cache, (dt, a_ii)) do
                    try
                        factorize(I + dt * a_ii * L_matrix)
                    catch e
                        isa(e, SingularException) || rethrow(e)
                        @warn "IMEX RK: system matrix is singular, falling back to explicit" maxlog=1
                        return nothing
                    end
                end
                if lhs === nothing
                    _step_rk_imex_explicit_fallback!(state, solver, ts)
                    return nothing
                end
                _rk_ldiv!(Xs_vec, lhs, rhs_vec)
            end
        else
            if abs(a_ii) < 1e-14
                if M_factor !== nothing
                    _apply_mass_inverse!(Xs_vec, M_factor, rhs_vec)
                else
                    # Singular M with a_ii=0 (ESDIRK first stage):
                    # M*X_s = M*X_n is under-determined for constraint rows (zero M rows).
                    # Use X_n directly so constraint variable values are preserved.
                    copyto!(Xs_vec, X_n_vec)
                end
            else
                # (M + dt*a*L) * X = rhs — works for both regular and singular M
                lhs = get!(lhs_cache, (dt, a_ii)) do
                    try
                        factorize(M_matrix + dt * a_ii * L_matrix)
                    catch e
                        isa(e, SingularException) || rethrow(e)
                        @warn "IMEX RK: system matrix is singular, falling back to explicit" maxlog=1
                        return nothing
                    end
                end
                if lhs === nothing
                    _step_rk_imex_explicit_fallback!(state, solver, ts)
                    return nothing
                end
                _rk_ldiv!(Xs_vec, lhs, rhs_vec)
            end
        end

        Xs_fields = _timestep_field_state!(state, :imex_rk_stage_state, current_state)
        vector_to_fields!(Xs_fields, Xs_vec, current_state)
        F_exp_fields = evaluate_rhs(solver, Xs_fields, t + c[s] * dt)
        fields_to_vector!(F_exp_vecs[s], F_exp_fields)
        mul!(F_imp_vecs[s], L_matrix, Xs_vec)
    end

    # Final update: M*X_{n+1} = M*X_n + dt*Σ(b_exp*F - b_imp*L*X)
    copyto!(rhs_vec, MX_n_vec)
    @inbounds for s in 1:stages
        be = dt * b_exp[s]
        bi = dt * b_imp[s]
        if abs(be) > 1e-14
            @. rhs_vec += be * F_exp_vecs[s]
        end
        if abs(bi) > 1e-14
            @. rhs_vec -= bi * F_imp_vecs[s]
        end
    end

    X_new_vec = _timestep_vector_buffer!(state, :imex_rk_X_new_vec, vector_size)
    if M_is_singular
        _solve_constrained_mass_update!(X_new_vec, state, solver, M_matrix, L_matrix, rhs_vec)
    else
        _apply_mass_inverse!(X_new_vec, M_factor, rhs_vec)
    end

    _push_vector_state!(state.history, X_new_vec, current_state, 1)
    return nothing
end

# ----------------------------------------------------------------------------
# DAE / singular-mass final update.
#
# When M has all-zero rows, those rows are algebraic constraints (div(u)=0,
# boundary conditions, gauge), not evolution equations, so `M X_{n+1} = rhs`
# is singular and cannot be solved directly. The fix: replace each zero M row
# with the corresponding L row, turning the constraint into an enforceable
# linear equation, then solve the patched system. The patched matrix and its
# factorization are cached and rebuilt only when M or L change identity.
# ----------------------------------------------------------------------------

"""
    _solve_constrained_mass_update!(dest, state, solver, M_matrix, L_matrix, rhs) -> dest

Final-stage update for DAE systems (singular `M`). Solves the row-patched system
(zero `M` rows swapped for `L` rows) and writes the algebraic constraint targets
into the corresponding RHS entries before solving. See the block comment above.
"""
function _solve_constrained_mass_update!(dest::AbstractVector{ComplexF64},
                                         state::TimestepperState,
                                         solver::InitialValueSolver,
                                         M_matrix::AbstractMatrix,
                                         L_matrix::AbstractMatrix,
                                         rhs::AbstractVector{ComplexF64})
    lhs_solver, zero_rows = _get_constrained_mass_solver!(state, M_matrix, L_matrix)
    constrained_rhs = _timestep_vector_buffer!(state, :imex_rk_constrained_rhs, length(rhs))
    copyto!(constrained_rhs, rhs)
    _apply_global_algebraic_rhs!(constrained_rhs, solver.problem, zero_rows)
    _rk_ldiv!(dest, lhs_solver, constrained_rhs)
    return dest
end

function _get_constrained_mass_solver!(state::TimestepperState,
                                       M_matrix::AbstractMatrix,
                                       L_matrix::AbstractMatrix)
    cache = state.timestepper_data
    if get(cache, :imex_rk_constrained_M_source, nothing) !== M_matrix ||
       get(cache, :imex_rk_constrained_L_source, nothing) !== L_matrix
        zero_rows = _zero_mass_rows(M_matrix)
        constrained_mass = _constrained_mass_matrix(M_matrix, L_matrix, zero_rows)
        cache[:imex_rk_constrained_mass_solver] = factorize(constrained_mass)
        cache[:imex_rk_constrained_mass_rows] = zero_rows
        cache[:imex_rk_constrained_M_source] = M_matrix
        cache[:imex_rk_constrained_L_source] = L_matrix
    end
    return cache[:imex_rk_constrained_mass_solver], cache[:imex_rk_constrained_mass_rows]::Vector{Int}
end

function _zero_mass_rows(M::AbstractMatrix)
    rows = Int[]
    for i in axes(M, 1)
        row_max = zero(real(eltype(M)))
        for j in axes(M, 2)
            row_max = max(row_max, abs(M[i, j]))
            row_max > 1e-14 && break
        end
        row_max <= 1e-14 && push!(rows, i)
    end
    return rows
end

function _constrained_mass_matrix(M::AbstractMatrix, L::AbstractMatrix, zero_rows::Vector{Int})
    constrained = sparse(M)
    for row in zero_rows
        constrained[row, :] = L[row, :]
    end
    return constrained
end

# Write the target value of each algebraic constraint into its RHS slot. After
# the row-swap, a zero-M row enforces `L_row * X = value`; this fills in `value`
# from each equation's F term (constraints default to 0, e.g. div(u)=0).
function _apply_global_algebraic_rhs!(rhs::AbstractVector{ComplexF64},
                                      problem::Problem,
                                      zero_rows::Vector{Int})
    isempty(zero_rows) && return rhs
    fill!(view(rhs, zero_rows), zero(ComplexF64))

    hasfield(typeof(problem), :equation_data) || return rhs
    isempty(problem.equation_data) && return rhs

    zero_row_set = Set(zero_rows)
    offset = 0
    for eq_data in problem.equation_data
        eq_size = compute_field_size(eq_data)
        if eq_size <= 0
            continue
        end

        M_expr = get(eq_data, "M", nothing)
        is_alg = M_expr === nothing || _is_zero_m_term(M_expr)
        if is_alg
            value = _global_algebraic_rhs_value(eq_data)
            if value != 0
                @inbounds for local_row in 1:eq_size
                    row = offset + local_row
                    if row in zero_row_set
                        rhs[row] = value
                    end
                end
            end
        end
        offset += eq_size
    end
    return rhs
end

function _global_algebraic_rhs_value(eq_data)
    F_expr = get(eq_data, "F_expr", nothing)
    F_expr === nothing && (F_expr = get(eq_data, "F", nothing))
    if F_expr isa ConstantOperator
        return ComplexF64(F_expr.value)
    elseif F_expr isa Number
        return ComplexF64(F_expr)
    else
        return ComplexF64(0)
    end
end

"""
    _apply_linear_operator(L_matrix, state)

Apply the linear operator L to a state (collection of fields).
Returns F_imp = L * X as a vector of fields.
"""
function _apply_linear_operator(L_matrix::AbstractMatrix, state::Vector{<:ScalarField})
    X_vector = fields_to_vector(state)
    LX_vector = L_matrix * X_vector
    return vector_to_fields(LX_vector, state)
end

"""
    _step_explicit_rk!(state, solver, A, b, c)

Generic explicit Runge-Kutta step for X' = M^-1 F(X).

GPU-aware: When no mass matrix is present (M_matrix === nothing), uses a
GPU-optimized path that keeps all field data on the GPU and avoids CPU transfers.
When a mass matrix is present, falls back to CPU-based linear solve.
"""
function _step_explicit_rk!(state::TimestepperState, solver::InitialValueSolver,
                            A::AbstractMatrix, b::AbstractVector, c::AbstractVector)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = length(b)

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")

    if _distributed_field_path_required(current_state)
        # Note: M_matrix is ignored in GPU/MPI mode. For Fourier bases M=I anyway.
        # Non-trivial M (e.g., Chebyshev) would require distributed sparse solvers.
        _step_explicit_rk_gpu!(state, solver, A, b, c)  # Works for both GPU and MPI
    else
        _step_explicit_rk_cpu!(state, solver, A, b, c, M_matrix)
    end
end

"""
    _step_explicit_rk_gpu!(state, solver, A, b, c)

GPU-optimized explicit RK step. Keeps all data on GPU, no CPU transfers.
Only used when M_matrix === nothing (no mass matrix inversion needed).
"""
function _step_explicit_rk_gpu!(state::TimestepperState, solver::InitialValueSolver,
                                A::AbstractMatrix, b::AbstractVector, c::AbstractVector)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = length(b)

    # Store stage derivatives (k values) as field vectors
    k_stages = Vector{Vector{ScalarField}}(undef, stages)
    n_fields = length(current_state)

    # Pre-build a reusable stage_state from workspace fields (avoids copy_state per stage)
    stage_state = Vector{ScalarField}(undef, n_fields)
    for (k, src_field) in enumerate(current_state)
        stage_state[k] = get_workspace_field!(state, src_field, k)
    end

    @inbounds for s in 1:stages
        state.current_substep = s

        # Compute stage value: Y_s = X_n + dt * sum_{j<s} A[s,j] * k_j
        # Copy current_state into workspace (in-place, no allocation)
        for (k, src_field) in enumerate(current_state)
            copy_field_data!(stage_state[k], src_field)
            stage_state[k].current_layout = src_field.current_layout
        end
        # Add contributions from previous stages
        for j in 1:(s-1)
            if abs(A[s, j]) > 1e-14
                axpy_state!(dt * A[s, j], k_stages[j], stage_state)
            end
        end

        # Evaluate RHS: k_s = F(t + c[s]*dt, Y_s)
        F_stage = evaluate_rhs(solver, stage_state, t + c[s] * dt)
        k_stages[s] = F_stage
    end

    # Compute final update: X_{n+1} = X_n + dt * sum_s b[s] * k_s
    new_state = copy_state(current_state)
    @inbounds for s in 1:stages
        if abs(b[s]) > 1e-14
            axpy_state!(dt * b[s], k_stages[s], new_state)
        end
    end

    _refresh_algebraic_state!(solver.problem, new_state)
    _push_trim!(state.history, new_state, 1)
end

"""
    _step_explicit_rk_cpu!(state, solver, A, b, c, M_matrix)

CPU-based explicit RK step. Used when mass matrix inversion is needed,
which requires sparse linear solves on CPU.

Uses cached LU factorization of M_matrix for efficiency.
"""
function _step_explicit_rk_cpu!(state::TimestepperState, solver::InitialValueSolver,
                                A::AbstractMatrix, b::AbstractVector, c::AbstractVector,
                                M_matrix)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = length(b)

    # Get cached mass matrix factorization (returns nothing for singular M)
    M_factor = M_matrix === nothing ? nothing : _get_mass_factor!(state, M_matrix)

    _ensure_coeff_layout!(current_state)
    vector_size = _fields_vector_size(current_state)
    X_n_vec = _timestep_vector_buffer!(state, :explicit_rk_X_n_vec, vector_size)
    fields_to_vector!(X_n_vec, current_state)
    k_vecs = _timestep_stage_vectors!(state, :explicit_rk_k_vecs, stages, vector_size)
    Y_vec = _timestep_vector_buffer!(state, :explicit_rk_Y_vec, vector_size)
    F_vec = _timestep_vector_buffer!(state, :explicit_rk_F_vec, vector_size)
    stage_state = _timestep_field_state!(state, :explicit_rk_stage_state, current_state)

    @inbounds for s in 1:stages
        state.current_substep = s

        copyto!(Y_vec, X_n_vec)
        for j in 1:(s-1)
            if abs(A[s, j]) > 1e-14
                _axpy_complex_vector!(Y_vec, dt * A[s, j], k_vecs[j])
            end
        end

        vector_to_fields!(stage_state, Y_vec, current_state)
        F_stage = evaluate_rhs(solver, stage_state, t + c[s] * dt)
        fields_to_vector!(F_vec, F_stage)
        if M_factor === nothing
            copyto!(k_vecs[s], F_vec)
        else
            _apply_mass_inverse!(k_vecs[s], M_factor, F_vec)
        end
    end

    # Compute final update (reuse Y_vec buffer)
    copyto!(Y_vec, X_n_vec)
    @inbounds for s in 1:stages
        if abs(b[s]) > 1e-14
            _axpy_complex_vector!(Y_vec, dt * b[s], k_vecs[s])
        end
    end

    _push_vector_state!(state.history, Y_vec, current_state, 1)
end

"""
    _step_rk_imex_explicit_fallback!(state, solver, ts=state.timestepper)

Fallback to fully explicit RK when no L_matrix is provided. Uses only `ts`'s
explicit tableau coefficients.

`ts` MUST be the RK tableau actually in use, which is NOT always
`state.timestepper`: during multistep startup (e.g. SBDF3) the bootstrap calls
`step_rk_imex!(state, solver; ts=RK443())` while `state.timestepper` is the
multistep method, which has no `A_explicit` field. Reading `state.timestepper`
here instead of the passed `ts` therefore errors with `type SBDF3 has no field
A_explicit`. The caller passes its `ts` through.
"""
function _step_rk_imex_explicit_fallback!(state::TimestepperState, solver::InitialValueSolver,
                                          ts::TimeStepper=state.timestepper)
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end

# Singleton instances for explicit RK fallbacks (avoid allocation per call)
const _RK111_SINGLETON = RK111()
const _RK222_SINGLETON = RK222()
const _RK443_SINGLETON = RK443()

# Explicit RK fallbacks used by other timesteppers
function step_rk111!(state::TimestepperState, solver::InitialValueSolver)
    ts = _RK111_SINGLETON
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end

function step_rk222!(state::TimestepperState, solver::InitialValueSolver)
    ts = _RK222_SINGLETON
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end

function step_rk443!(state::TimestepperState, solver::InitialValueSolver)
    ts = _RK443_SINGLETON
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end
