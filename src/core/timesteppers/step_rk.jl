# =======================================
# IMEX Runge-Kutta Step Functions
#
# Sign convention: L_matrix uses the LHS form  M*dX/dt + L*X = F
# so L appears with a POSITIVE sign on the LHS of the implicit solve:
#   (M + dt*a*L) * X_s = RHS
# and the implicit RHS contribution (L*X) is SUBTRACTED.
#
# This is consistent with the multistep IMEX methods (CNAB, SBDF) which
# also use L_matrix directly in LHS form.  ETD methods instead use
# _get_linear_operator_eff!() which returns -M^{-1}*L (RHS form).
# =======================================

function step_rk_imex!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
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

    # Subproblem-based IMEX: use sparse subproblems if available
    if haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        if sps isa Tuple
            step_subproblem_rk!(state, solver, sps)
            return
        end
    end

    # If no linear operator provided, fall back to explicit-only treatment
    if L_matrix === nothing
        @debug "IMEX RK: No L_matrix found, treating all terms explicitly"
        _step_rk_imex_explicit_fallback!(state, solver)
        return
    end

    # Fall back to explicit treatment when matrix-vector operations are incompatible:
    # - GPU: SparseMatrixCSC is CPU-only, cannot multiply with CuArray
    # - MPI+PencilArrays: matrix-vector doesn't work with distributed data
    if !isempty(current_state)
        dist = current_state[1].dist
        if is_gpu(dist.architecture)
            @debug "IMEX RK: GPU detected, falling back to explicit treatment (CPU sparse matrices incompatible with CuArray)"
            _step_rk_imex_explicit_fallback!(state, solver)
            return
        end
        if dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
            @debug "IMEX RK: MPI mode detected, falling back to explicit treatment"
            _step_rk_imex_explicit_fallback!(state, solver)
            return
        end
    end

    # Check if L_matrix is effectively zero (no linear terms)
    L_is_zero = (L_matrix isa SparseMatrixCSC && nnz(L_matrix) == 0) ||
                (norm(L_matrix, Inf) < 1e-14)

    if L_is_zero
        @debug "IMEX RK: L_matrix is zero, falling back to explicit treatment"
        _step_rk_imex_explicit_fallback!(state, solver)
        return
    end

    # Mass matrix handling: distinguish "no M" from "singular M (DAE)".
    # For DAE systems (constraint equations with zero M rows), we must still
    # use M in the stage factorizations so that (M + dt*a*L) correctly enforces
    # algebraic constraints (e.g., div(u) = 0, BCs, gauge conditions).
    # _get_mass_factor! returns nothing for singular M (cached after first call).
    has_mass = M_matrix !== nothing
    M_factor = has_mass ? _get_mass_factor!(state, M_matrix) : nothing
    M_is_singular = has_mass && M_factor === nothing

    X_n_vec = copy(fields_to_vector(current_state))  # copy: cache is shared across calls
    MX_n_vec = has_mass ? (M_matrix * X_n_vec) : X_n_vec

    T_vec = eltype(X_n_vec)
    F_exp_vecs = Vector{Vector{T_vec}}(undef, stages)
    F_imp_vecs = Vector{Vector{T_vec}}(undef, stages)
    rhs_vec = similar(MX_n_vec)  # Pre-allocate once, reuse across stages
    Xs_vec = similar(X_n_vec)    # Will hold last stage value after loop
    # Cache LHS factorizations across timesteps. Key is (dt, a_ii) so the cache
    # automatically invalidates when dt changes (adaptive stepping).
    lhs_cache = get!(state.timestepper_data, :imex_rk_lhs_cache) do
        Dict{Tuple{Float64, Float64}, Any}()
    end

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
                Xs_vec = copy(rhs_vec)
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
                    _step_rk_imex_explicit_fallback!(state, solver)
                    return
                end
                Xs_vec = lhs \ rhs_vec
            end
        else
            if abs(a_ii) < 1e-14
                if M_factor !== nothing
                    Xs_vec = M_factor \ rhs_vec
                else
                    # Singular M with a_ii=0 (ESDIRK first stage):
                    # M*X_s = M*X_n is under-determined for constraint rows (zero M rows).
                    # Use X_n directly so constraint variable values are preserved.
                    Xs_vec = copy(X_n_vec)
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
                    _step_rk_imex_explicit_fallback!(state, solver)
                    return
                end
                Xs_vec = lhs \ rhs_vec
            end
        end

        Xs_fields = vector_to_fields(Xs_vec, current_state)
        F_exp_fields = evaluate_rhs(solver, Xs_fields, t + c[s] * dt)
        F_exp_vecs[s] = fields_to_vector(F_exp_fields)
        F_imp_vecs[s] = L_matrix * Xs_vec
    end

    # Final update
    if M_is_singular
        # DAE system: the last stage value from the (M + dt*a*L) solve already
        # satisfies the algebraic constraints (div-free, BCs, gauge).
        # For stiffly accurate SDIRK methods (b_imp = A_imp[end,:]),
        # X_last is the correct implicit solution. Using it directly avoids
        # the need for M^{-1} which doesn't exist for singular M.
        X_new_vec = Xs_vec
    else
        # Standard update: M*X_{n+1} = M*X_n + dt*Σ(b_exp*F - b_imp*L*X)
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
        X_new_vec = _apply_mass_inverse(M_factor, rhs_vec)
    end

    new_state = copy_state(current_state)
    vector_to_fields!(new_state, X_new_vec, current_state)

    _push_trim!(state.history, new_state, 1)
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

    # Check if we should use field-based path
    # Field-based path is required for: GPU, or MPI with PencilArrays (distributed data)
    # because the vectorized matrix approach doesn't work with distributed data
    use_field_path = false
    is_mpi_mode = false
    if !isempty(current_state)
        arch = field_architecture(current_state[1])
        dist = current_state[1].dist
        is_mpi_mode = dist.use_pencil_arrays && MPI.Comm_size(MPI.COMM_WORLD) > 1
        # Use field-based path for GPU or MPI mode
        use_field_path = is_gpu(arch) || is_mpi_mode
    end

    if use_field_path
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

    X_n_vec = copy(fields_to_vector(current_state))  # copy: cache is shared across calls
    k_vecs = Vector{Vector{eltype(X_n_vec)}}(undef, stages)
    Y_vec = similar(X_n_vec)  # Pre-allocate once, reuse across stages

    @inbounds for s in 1:stages
        state.current_substep = s

        copyto!(Y_vec, X_n_vec)
        for j in 1:(s-1)
            if abs(A[s, j]) > 1e-14
                @. Y_vec += dt * A[s, j] * k_vecs[j]
            end
        end

        stage_state = vector_to_fields(Y_vec, current_state)
        F_stage = evaluate_rhs(solver, stage_state, t + c[s] * dt)
        F_vec = fields_to_vector(F_stage)
        k_vecs[s] = _apply_mass_inverse(M_factor, F_vec)
    end

    # Compute final update (reuse Y_vec buffer)
    copyto!(Y_vec, X_n_vec)
    @inbounds for s in 1:stages
        if abs(b[s]) > 1e-14
            @. Y_vec += dt * b[s] * k_vecs[s]
        end
    end

    new_state = copy_state(current_state)
    vector_to_fields!(new_state, Y_vec, current_state)
    _push_trim!(state.history, new_state, 1)
end

"""
    _step_rk_imex_explicit_fallback!(state, solver)

Fallback to fully explicit RK when no L_matrix is provided.
Uses only the explicit tableau coefficients.
"""
function _step_rk_imex_explicit_fallback!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
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


