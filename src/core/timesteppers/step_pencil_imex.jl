# ============================================================================
# Pencil-based IMEX Step Functions for Chebyshev-Fourier Domains
# ============================================================================
#
# These step functions implement SBDF and CNAB methods using pencil-based solves
# for Chebyshev-Fourier domains. Unlike global matrix methods, these work with
# MPI-distributed data because each Fourier mode is solved independently.
#
# The key insight is that for mixed Fourier-Chebyshev domains:
# - Fourier modes are decoupled (diagonal in spectral space)
# - Only Chebyshev couples modes (banded differentiation matrices)
# - Each (kx, ky) pencil can be solved independently
#
# ============================================================================

# SparseArrays, LinearAlgebra already in Tarang.jl

# Pre-computed Symbol keys — avoids String→Symbol allocation on every call
const _CACHE_SBDF1_LHS     = :sbdf1_lhs
const _CACHE_SBDF1_LHS_INV = :sbdf1_lhs_inv
const _CACHE_SBDF2_LHS     = :sbdf2_lhs
const _CACHE_SBDF2_LHS_INV = :sbdf2_lhs_inv
const _CACHE_CNAB_LHS      = :cnab_lhs
const _CACHE_CNAB_LHS_INV  = :cnab_lhs_inv
const _CACHE_CNAB_RHS      = :cnab_rhs
const _CACHE_CNAB_RHS_INV  = :cnab_rhs_inv

"""
    _get_or_factorize!(cache, key, build_fn)

Get a cached LU factorization or compute and cache it.
Falls back to direct computation when cache is nothing.
"""
function _get_or_factorize!(build_fn::Function, cache::Nothing, key)
    return build_fn()
end

function _get_or_factorize!(build_fn::Function, cache::Dict, key)
    return get!(build_fn, cache, key)
end

"""
    _get_pencil_lhs_cache!(state::TimestepperState, key::Symbol, inv_key::Symbol, invalidation_key)

Get or create a pencil LHS factorization cache in timestepper_data.
Invalidates the cache when `invalidation_key` changes (typically dt, or
(dt, a0) for methods with variable BDF coefficients).
"""
function _get_pencil_lhs_cache!(state::TimestepperState, key::Symbol, inv_key::Symbol, invalidation_key)
    data = state.timestepper_data
    if !haskey(data, key) || get(data, inv_key, nothing) != invalidation_key
        data[key] = Dict{Tuple{Int,Int}, Any}()
        data[inv_key] = invalidation_key
    end
    return data[key]
end

"""
    step_pencil_sbdf2!(state::TimestepperState, solver::InitialValueSolver)

SBDF2 using pencil-based solves for Chebyshev-Fourier domains.

This method works with MPI-distributed data by solving each Fourier mode independently.

Formula for SBDF2 (variable timestep, w = dt/dt_prev):
    (a0 + dt*L_k) X̂^{n+1} = -a1*X̂^n - a2*X̂^{n-1} + dt*((1+w)*F̂^n - w*F̂^{n-1})

BDF2 coefficients: a0=(1+2w)/(1+w), a1=-(1+w), a2=w²/(1+w)
For constant dt (w=1): a0=3/2, a1=-2, a2=1/2, ext=(2,-1).
"""
function step_pencil_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt

    # Get pencil linear operator
    L = _get_pencil_linear_operator(solver)
    if L === nothing
        @warn "step_pencil_sbdf2!: No PencilLinearOperator configured, falling back to RK222" maxlog=1
        step_rk222!(state, solver)
        return
    end

    # Initialize history if needed
    if !haskey(state.timestepper_data, :F_history_pencil)
        state.timestepper_data[:F_history_pencil] = []
        state.timestepper_data[:iteration_pencil] = 0
    end

    iteration = state.timestepper_data[:iteration_pencil]

    # Need 2 steps of history for SBDF2
    if iteration < 1 || length(state.history) < 2
        @debug "SBDF2 pencil: insufficient history, using SBDF1"
        step_pencil_sbdf1!(state, solver)
        return
    end

    # SBDF2 coefficients (variable timestep via BDF2 formula)
    dt_prev = get_previous_timestep(state)
    w = dt / dt_prev  # timestep ratio
    a0 = (1 + 2w) / (1 + w)
    a1 = -(1 + w)
    a2 = w^2 / (1 + w)
    # Variable-timestep extrapolation: N* = (1+w)*N^n - w*N^{n-1}
    ext_c1 = 1.0 + w   # coefficient for F^n (2 for constant dt)
    ext_c2 = w          # coefficient for F^{n-1} (1 for constant dt)

    try
        # Get history states
        X_n = current_state
        X_nm1 = state.history[end-1]

        # Evaluate F(X_n)
        F_n = evaluate_rhs(solver, X_n, solver.sim_time)

        # Get F history
        F_history = state.timestepper_data[:F_history_pencil]
        _prepend_trim!(F_history, F_n, 2)

        # For each field, apply pencil-based SBDF2
        new_state = Vector{ScalarField}(undef, length(X_n))

        for (i, field) in enumerate(X_n)
            ensure_layout!(field, :c)
            new_field = ScalarField(field.dist, field.name, (), field.dtype)
            new_field.bases = field.bases
            new_field.domain = field.domain
            new_field.current_layout = :c
            # Allocate coeff storage matching source (no data copy — solve overwrites it)
            set_coeff_data!(new_field, similar(get_coeff_data(field)))
            ensure_layout!(X_nm1[i], :c)
            ensure_layout!(F_history[1][i], :c)
            if length(F_history) >= 2
                ensure_layout!(F_history[2][i], :c)
            end

            # Transpose to solve layout (Chebyshev-local) if MPI with PencilFFTs
            needs_transpose = _needs_solve_transpose(field.dist, get_coeff_data(field), L)
            if needs_transpose
                data_n = _to_solve_layout(get_coeff_data(field), field.dist, L; cache=state.timestepper_data)
                data_nm1 = _to_solve_layout(get_coeff_data(X_nm1[i]), field.dist, L; cache=state.timestepper_data)
                data_F_n = _to_solve_layout(get_coeff_data(F_history[1][i]), field.dist, L; cache=state.timestepper_data)
                data_F_nm1 = length(F_history) >= 2 ? _to_solve_layout(get_coeff_data(F_history[2][i]), field.dist, L; cache=state.timestepper_data) : nothing
                data_new = similar(data_n)
            else
                data_n = get_coeff_data(field)
                data_nm1 = get_coeff_data(X_nm1[i])
                data_F_n = get_coeff_data(F_history[1][i])
                data_F_nm1 = length(F_history) >= 2 ? get_coeff_data(F_history[2][i]) : nothing
                data_new = get_coeff_data(new_field)
            end

            lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_SBDF2_LHS, _CACHE_SBDF2_LHS_INV, (dt, a0))
            _pencil_sbdf2_field!(
                data_new, data_n, data_nm1, data_F_n, data_F_nm1,
                L, dt, a0, a1, a2;
                ext_c1=ext_c1, ext_c2=ext_c2, lhs_cache=lhs_cache
            )

            if needs_transpose
                _from_solve_layout!(get_coeff_data(new_field), data_new, field.dist, L; cache=state.timestepper_data)
            end

            new_state[i] = new_field
        end

        # Update state
        _push_trim!(state.history, new_state, 3)
        state.timestepper_data[:iteration_pencil] += 1

    catch e
        @warn "step_pencil_sbdf2! failed: $e, falling back to RK222"
        step_rk222!(state, solver)
        return
    end
end

"""
    step_pencil_sbdf1!(state::TimestepperState, solver::InitialValueSolver)

SBDF1 (Backward Euler/Forward Euler) using pencil-based solves.

Formula:
    (M + dt*L_k) X̂^{n+1} = M X̂^n + dt*F̂^n
"""
function step_pencil_sbdf1!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt

    # Get pencil linear operator
    L = _get_pencil_linear_operator(solver)
    if L === nothing
        @warn "step_pencil_sbdf1!: No PencilLinearOperator configured, falling back to RK111" maxlog=1
        step_rk111!(state, solver)
        return
    end

    # Initialize history if needed
    if !haskey(state.timestepper_data, :F_history_pencil)
        state.timestepper_data[:F_history_pencil] = []
        state.timestepper_data[:iteration_pencil] = 0
    end

    try
        # Evaluate F(X_n)
        F_n = evaluate_rhs(solver, current_state, solver.sim_time)

        # Update F history
        F_history = state.timestepper_data[:F_history_pencil]
        _prepend_trim!(F_history, F_n, 2)

        # For each field, apply pencil-based SBDF1
        new_state = Vector{ScalarField}(undef, length(current_state))

        for (i, field) in enumerate(current_state)
            ensure_layout!(field, :c)
            new_field = ScalarField(field.dist, field.name, (), field.dtype)
            new_field.bases = field.bases
            new_field.domain = field.domain
            new_field.current_layout = :c
            # Allocate coeff storage matching source (no data copy — solve overwrites it)
            set_coeff_data!(new_field, similar(get_coeff_data(field)))
            ensure_layout!(F_n[i], :c)

            # Transpose to solve layout (Chebyshev-local) if MPI with PencilFFTs
            needs_transpose = _needs_solve_transpose(field.dist, get_coeff_data(field), L)
            if needs_transpose
                data_n = _to_solve_layout(get_coeff_data(field), field.dist, L; cache=state.timestepper_data)
                data_F_n = _to_solve_layout(get_coeff_data(F_n[i]), field.dist, L; cache=state.timestepper_data)
                data_new = similar(data_n)
            else
                data_n = get_coeff_data(field)
                data_F_n = get_coeff_data(F_n[i])
                data_new = get_coeff_data(new_field)
            end

            lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_SBDF1_LHS, _CACHE_SBDF1_LHS_INV, dt)
            _pencil_sbdf1_field!(data_new, data_n, data_F_n, L, dt; lhs_cache=lhs_cache)

            if needs_transpose
                _from_solve_layout!(get_coeff_data(new_field), data_new, field.dist, L; cache=state.timestepper_data)
            end

            new_state[i] = new_field
        end

        # Update state
        _push_trim!(state.history, new_state, 3)
        state.timestepper_data[:iteration_pencil] += 1

    catch e
        @warn "step_pencil_sbdf1! failed: $e, falling back to RK111"
        step_rk111!(state, solver)
        return
    end
end

"""
    _pencil_sbdf1_field!(data_new, data_n, data_F_n, L, dt; lhs_cache=nothing)

Apply pencil-based SBDF1 to a single field's coefficient data.

For each (kx, ky) pencil:
    (1 + dt*L_k) X̂^{n+1} = X̂^n + dt*F̂^n

where L_k is the per-wavenumber operator.

If `lhs_cache` is provided (Dict keyed by (ikx,iky)), LU factorizations are
cached and reused across timesteps when dt is constant.
"""
function _pencil_sbdf1_field!(
    data_new::AbstractArray,
    data_n::AbstractArray,
    data_F_n::AbstractArray,
    L::PencilLinearOperator{T},
    dt::Real;
    lhs_cache::Union{Nothing, Dict}=nothing
) where T

    ndims = length(L.fourier_basis_indices)
    Nz = L.Nz
    ν = L.parameters[:ν]

    rhs_buf = L._rhs_buf
    sol_buf = L._sol_buf

    if ndims == 1
        # 2D: (Nkx, Nz)
        nkx = size(data_n, 1)

        for ikx in 1:nkx
            if ikx <= length(L.local_kx_range)
                k2 = L.k2_values[ikx, 1]

                factor = _get_or_factorize!(lhs_cache, (ikx, 1)) do
                    LHS = (1 + dt * ν * k2) * sparse(I, Nz, Nz) - dt * ν * L.chebyshev_D2
                    lu(LHS)
                end

                # Build RHS in-place: rhs = X̂^n + dt*F̂^n
                pencil_n = @view data_n[ikx, :]
                pencil_F = @view data_F_n[ikx, :]
                @. rhs_buf = pencil_n + dt * pencil_F

                # Solve and store
                ldiv!(sol_buf, factor, rhs_buf)
                data_new[ikx, :] .= sol_buf
            end
        end

    elseif ndims == 2
        # 3D: (Nkx, Nky, Nz)
        nkx = size(data_n, 1)
        nky = size(data_n, 2)

        for iky in 1:nky
            for ikx in 1:nkx
                if ikx <= length(L.local_kx_range) && iky <= length(L.local_ky_range)
                    k2 = L.k2_values[ikx, iky]

                    factor = _get_or_factorize!(lhs_cache, (ikx, iky)) do
                        LHS = (1 + dt * ν * k2) * sparse(I, Nz, Nz) - dt * ν * L.chebyshev_D2
                        lu(LHS)
                    end

                    pencil_n = @view data_n[ikx, iky, :]
                    pencil_F = @view data_F_n[ikx, iky, :]
                    @. rhs_buf = pencil_n + dt * pencil_F

                    ldiv!(sol_buf, factor, rhs_buf)
                    data_new[ikx, iky, :] .= sol_buf
                end
            end
        end
    end
end

"""
    _pencil_sbdf2_field!(data_new, data_n, data_nm1, data_F_n, data_F_nm1, L, dt, a0, a1, a2; ext_c1, ext_c2, lhs_cache)

Apply pencil-based SBDF2 to a single field's coefficient data.

For each (kx, ky) pencil:
    (a0 + dt*L_k) X̂^{n+1} = -a1*X̂^n - a2*X̂^{n-1} + dt*(ext_c1*F̂^n - ext_c2*F̂^{n-1})

ext_c1 and ext_c2 are the variable-timestep extrapolation coefficients:
    ext_c1 = 1+w, ext_c2 = w  where w = dt/dt_prev
"""
function _pencil_sbdf2_field!(
    data_new::AbstractArray,
    data_n::AbstractArray,
    data_nm1::AbstractArray,
    data_F_n::AbstractArray,
    data_F_nm1::Union{Nothing, AbstractArray},
    L::PencilLinearOperator{T},
    dt::Real,
    a0::Real, a1::Real, a2::Real;
    ext_c1::Real=2.0, ext_c2::Real=1.0,
    lhs_cache::Union{Nothing, Dict}=nothing
) where T

    ndims = length(L.fourier_basis_indices)
    Nz = L.Nz
    ν = L.parameters[:ν]

    rhs_buf = L._rhs_buf
    sol_buf = L._sol_buf

    if ndims == 1
        # 2D: (Nkx, Nz)
        nkx = size(data_n, 1)

        for ikx in 1:nkx
            if ikx <= length(L.local_kx_range)
                k2 = L.k2_values[ikx, 1]

                # LHS = a0*I + dt*L_k = a0*I + dt*ν*k²*I - dt*ν*D²
                factor = _get_or_factorize!(lhs_cache, (ikx, 1)) do
                    LHS = (a0 + dt * ν * k2) * sparse(I, Nz, Nz) - dt * ν * L.chebyshev_D2
                    lu(LHS)
                end

                # RHS = -a1*X̂^n - a2*X̂^{n-1} + dt*(ext_c1*F̂^n - ext_c2*F̂^{n-1})
                pencil_n = @view data_n[ikx, :]
                pencil_nm1 = @view data_nm1[ikx, :]
                pencil_F_n = @view data_F_n[ikx, :]

                @. rhs_buf = -a1 * pencil_n - a2 * pencil_nm1 + dt * ext_c1 * pencil_F_n
                if data_F_nm1 !== nothing
                    pencil_F_nm1 = @view data_F_nm1[ikx, :]
                    @. rhs_buf -= dt * ext_c2 * pencil_F_nm1
                end

                ldiv!(sol_buf, factor, rhs_buf)
                data_new[ikx, :] .= sol_buf
            end
        end

    elseif ndims == 2
        # 3D: (Nkx, Nky, Nz)
        nkx = size(data_n, 1)
        nky = size(data_n, 2)

        for iky in 1:nky
            for ikx in 1:nkx
                if ikx <= length(L.local_kx_range) && iky <= length(L.local_ky_range)
                    k2 = L.k2_values[ikx, iky]

                    factor = _get_or_factorize!(lhs_cache, (ikx, iky)) do
                        LHS = (a0 + dt * ν * k2) * sparse(I, Nz, Nz) - dt * ν * L.chebyshev_D2
                        lu(LHS)
                    end

                    pencil_n = @view data_n[ikx, iky, :]
                    pencil_nm1 = @view data_nm1[ikx, iky, :]
                    pencil_F_n = @view data_F_n[ikx, iky, :]

                    @. rhs_buf = -a1 * pencil_n - a2 * pencil_nm1 + dt * ext_c1 * pencil_F_n
                    if data_F_nm1 !== nothing
                        pencil_F_nm1 = @view data_F_nm1[ikx, iky, :]
                        @. rhs_buf -= dt * ext_c2 * pencil_F_nm1
                    end

                    ldiv!(sol_buf, factor, rhs_buf)
                    data_new[ikx, iky, :] .= sol_buf
                end
            end
        end
    end
end

"""
    step_pencil_cnab2!(state::TimestepperState, solver::InitialValueSolver)

CNAB2 using pencil-based solves for Chebyshev-Fourier domains.

Formula:
    (M + 0.5*dt*L_k) X̂^{n+1} = (M - 0.5*dt*L_k) X̂^n + dt*(3/2*F̂^n - 1/2*F̂^{n-1})
"""
function step_pencil_cnab2!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt

    # Get pencil linear operator
    L = _get_pencil_linear_operator(solver)
    if L === nothing
        @warn "step_pencil_cnab2!: No PencilLinearOperator configured, falling back to RK222" maxlog=1
        step_rk222!(state, solver)
        return
    end

    # Initialize history if needed
    if !haskey(state.timestepper_data, :F_history_pencil)
        state.timestepper_data[:F_history_pencil] = []
        state.timestepper_data[:iteration_pencil] = 0
    end

    iteration = state.timestepper_data[:iteration_pencil]

    # Need 1 step of F history for CNAB2
    if iteration < 1
        @debug "CNAB2 pencil: insufficient history, using CNAB1"
        step_pencil_cnab1!(state, solver)
        return
    end

    try
        # Evaluate F(X_n)
        F_n = evaluate_rhs(solver, current_state, solver.sim_time)

        # Get F history
        F_history = state.timestepper_data[:F_history_pencil]
        _prepend_trim!(F_history, F_n, 2)

        # For each field, apply pencil-based CNAB2
        new_state = Vector{ScalarField}(undef, length(current_state))

        for (i, field) in enumerate(current_state)
            ensure_layout!(field, :c)
            new_field = ScalarField(field.dist, field.name, (), field.dtype)
            new_field.bases = field.bases
            new_field.domain = field.domain
            new_field.current_layout = :c
            # Allocate coeff storage matching source (no data copy — solve overwrites it)
            set_coeff_data!(new_field, similar(get_coeff_data(field)))
            ensure_layout!(F_history[1][i], :c)
            if length(F_history) >= 2
                ensure_layout!(F_history[2][i], :c)
            end

            needs_transpose = _needs_solve_transpose(field.dist, get_coeff_data(field), L)
            if needs_transpose
                data_n = _to_solve_layout(get_coeff_data(field), field.dist, L; cache=state.timestepper_data)
                data_F_n = _to_solve_layout(get_coeff_data(F_history[1][i]), field.dist, L; cache=state.timestepper_data)
                data_F_nm1 = length(F_history) >= 2 ? _to_solve_layout(get_coeff_data(F_history[2][i]), field.dist, L; cache=state.timestepper_data) : nothing
                data_new = similar(data_n)
            else
                data_n = get_coeff_data(field)
                data_F_n = get_coeff_data(F_history[1][i])
                data_F_nm1 = length(F_history) >= 2 ? get_coeff_data(F_history[2][i]) : nothing
                data_new = get_coeff_data(new_field)
            end

            dt_prev = get_previous_timestep(state)
            lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_CNAB_LHS, _CACHE_CNAB_LHS_INV, dt)
            rhs_cache = _get_pencil_lhs_cache!(state, _CACHE_CNAB_RHS, _CACHE_CNAB_RHS_INV, dt)
            _pencil_cnab2_field!(data_new, data_n, data_F_n, data_F_nm1, L, dt; dt_prev=dt_prev, lhs_cache=lhs_cache, rhs_cache=rhs_cache)

            if needs_transpose
                _from_solve_layout!(get_coeff_data(new_field), data_new, field.dist, L; cache=state.timestepper_data)
            end

            new_state[i] = new_field
        end

        # Update state
        _push_trim!(state.history, new_state, 3)
        state.timestepper_data[:iteration_pencil] += 1

    catch e
        @warn "step_pencil_cnab2! failed: $e, falling back to RK222"
        step_rk222!(state, solver)
        return
    end
end

"""
    step_pencil_cnab1!(state::TimestepperState, solver::InitialValueSolver)

CNAB1 (Crank-Nicolson/Forward Euler) using pencil-based solves.

Formula:
    (M + 0.5*dt*L_k) X̂^{n+1} = (M - 0.5*dt*L_k) X̂^n + dt*F̂^n
"""
function step_pencil_cnab1!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt

    # Get pencil linear operator
    L = _get_pencil_linear_operator(solver)
    if L === nothing
        @warn "step_pencil_cnab1!: No PencilLinearOperator configured, falling back to RK111" maxlog=1
        step_rk111!(state, solver)
        return
    end

    # Initialize history if needed
    if !haskey(state.timestepper_data, :F_history_pencil)
        state.timestepper_data[:F_history_pencil] = []
        state.timestepper_data[:iteration_pencil] = 0
    end

    try
        # Evaluate F(X_n)
        F_n = evaluate_rhs(solver, current_state, solver.sim_time)

        # Update F history
        F_history = state.timestepper_data[:F_history_pencil]
        _prepend_trim!(F_history, F_n, 2)

        # For each field, apply pencil-based CNAB1
        new_state = Vector{ScalarField}(undef, length(current_state))

        for (i, field) in enumerate(current_state)
            ensure_layout!(field, :c)
            new_field = ScalarField(field.dist, field.name, (), field.dtype)
            new_field.bases = field.bases
            new_field.domain = field.domain
            new_field.current_layout = :c
            # Allocate coeff storage matching source (no data copy — solve overwrites it)
            set_coeff_data!(new_field, similar(get_coeff_data(field)))
            ensure_layout!(F_n[i], :c)

            needs_transpose = _needs_solve_transpose(field.dist, get_coeff_data(field), L)
            if needs_transpose
                data_n = _to_solve_layout(get_coeff_data(field), field.dist, L; cache=state.timestepper_data)
                data_F_n = _to_solve_layout(get_coeff_data(F_n[i]), field.dist, L; cache=state.timestepper_data)
                data_new = similar(data_n)
            else
                data_n = get_coeff_data(field)
                data_F_n = get_coeff_data(F_n[i])
                data_new = get_coeff_data(new_field)
            end

            lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_CNAB_LHS, _CACHE_CNAB_LHS_INV, dt)
            rhs_cache = _get_pencil_lhs_cache!(state, _CACHE_CNAB_RHS, _CACHE_CNAB_RHS_INV, dt)
            _pencil_cnab1_field!(data_new, data_n, data_F_n, L, dt; lhs_cache=lhs_cache, rhs_cache=rhs_cache)

            if needs_transpose
                _from_solve_layout!(get_coeff_data(new_field), data_new, field.dist, L; cache=state.timestepper_data)
            end

            new_state[i] = new_field
        end

        # Update state
        _push_trim!(state.history, new_state, 3)
        state.timestepper_data[:iteration_pencil] += 1

    catch e
        @warn "step_pencil_cnab1! failed: $e, falling back to RK111"
        step_rk111!(state, solver)
        return
    end
end

"""
    _pencil_cnab1_field!(data_new, data_n, data_F_n, L, dt; lhs_cache=nothing)

Apply pencil-based CNAB1 to a single field's coefficient data.

For each (kx, ky) pencil:
    (1 + 0.5*dt*L_k) X̂^{n+1} = (1 - 0.5*dt*L_k) X̂^n + dt*F̂^n
"""
function _pencil_cnab1_field!(
    data_new::AbstractArray,
    data_n::AbstractArray,
    data_F_n::AbstractArray,
    L::PencilLinearOperator{T},
    dt::Real;
    lhs_cache::Union{Nothing, Dict}=nothing,
    rhs_cache::Union{Nothing, Dict}=nothing
) where T

    ndims = length(L.fourier_basis_indices)
    Nz = L.Nz
    ν = L.parameters[:ν]
    θ = 0.5  # Crank-Nicolson parameter

    rhs_buf = L._rhs_buf
    sol_buf = L._sol_buf

    if ndims == 1
        # 2D: (Nkx, Nz)
        nkx = size(data_n, 1)

        for ikx in 1:nkx
            if ikx <= length(L.local_kx_range)
                k2 = L.k2_values[ikx, 1]

                # L_k = ν*k² - ν*D²
                # LHS = I + θ*dt*L_k = (1 + θ*dt*ν*k²)*I - θ*dt*ν*D²
                # RHS_mat = I - (1-θ)*dt*L_k = (1 - (1-θ)*dt*ν*k²)*I + (1-θ)*dt*ν*D²
                factor = _get_or_factorize!(lhs_cache, (ikx, 1)) do
                    LHS = (1 + θ * dt * ν * k2) * sparse(I, Nz, Nz) - θ * dt * ν * L.chebyshev_D2
                    lu(LHS)
                end
                RHS_mat = _get_or_factorize!(rhs_cache, (ikx, 1)) do
                    (1 - (1-θ) * dt * ν * k2) * sparse(I, Nz, Nz) + (1-θ) * dt * ν * L.chebyshev_D2
                end

                pencil_n = @view data_n[ikx, :]
                pencil_F = @view data_F_n[ikx, :]
                mul!(rhs_buf, RHS_mat, pencil_n)
                @. rhs_buf += dt * pencil_F

                ldiv!(sol_buf, factor, rhs_buf)
                data_new[ikx, :] .= sol_buf
            end
        end

    elseif ndims == 2
        # 3D: (Nkx, Nky, Nz)
        nkx = size(data_n, 1)
        nky = size(data_n, 2)

        for iky in 1:nky
            for ikx in 1:nkx
                if ikx <= length(L.local_kx_range) && iky <= length(L.local_ky_range)
                    k2 = L.k2_values[ikx, iky]

                    factor = _get_or_factorize!(lhs_cache, (ikx, iky)) do
                        LHS = (1 + θ * dt * ν * k2) * sparse(I, Nz, Nz) - θ * dt * ν * L.chebyshev_D2
                        lu(LHS)
                    end
                    RHS_mat = _get_or_factorize!(rhs_cache, (ikx, iky)) do
                        (1 - (1-θ) * dt * ν * k2) * sparse(I, Nz, Nz) + (1-θ) * dt * ν * L.chebyshev_D2
                    end

                    pencil_n = @view data_n[ikx, iky, :]
                    pencil_F = @view data_F_n[ikx, iky, :]
                    mul!(rhs_buf, RHS_mat, pencil_n)
                    @. rhs_buf += dt * pencil_F

                    ldiv!(sol_buf, factor, rhs_buf)
                    data_new[ikx, iky, :] .= sol_buf
                end
            end
        end
    end
end

"""
    _pencil_cnab2_field!(data_new, data_n, data_F_n, data_F_nm1, L, dt; dt_prev, lhs_cache=nothing)

Apply pencil-based CNAB2 to a single field's coefficient data.

For each (kx, ky) pencil:
    (1 + 0.5*dt*L_k) X̂^{n+1} = (1 - 0.5*dt*L_k) X̂^n + dt*(3/2*F̂^n - 1/2*F̂^{n-1})
"""
function _pencil_cnab2_field!(
    data_new::AbstractArray,
    data_n::AbstractArray,
    data_F_n::AbstractArray,
    data_F_nm1::Union{Nothing, AbstractArray},
    L::PencilLinearOperator{T},
    dt::Real;
    dt_prev::Real=dt,
    lhs_cache::Union{Nothing, Dict}=nothing,
    rhs_cache::Union{Nothing, Dict}=nothing
) where T

    ndims = length(L.fourier_basis_indices)
    Nz = L.Nz
    ν = L.parameters[:ν]
    θ = 0.5  # Crank-Nicolson parameter

    # Variable-timestep Adams-Bashforth 2 coefficients
    w = dt / dt_prev  # timestep ratio (w=1 for constant dt gives c1=3/2, c2=1/2)
    ab2_c1 = 1 + w / 2
    ab2_c2 = w / 2

    rhs_buf = L._rhs_buf
    sol_buf = L._sol_buf

    if ndims == 1
        # 2D: (Nkx, Nz)
        nkx = size(data_n, 1)

        for ikx in 1:nkx
            if ikx <= length(L.local_kx_range)
                k2 = L.k2_values[ikx, 1]

                factor = _get_or_factorize!(lhs_cache, (ikx, 1)) do
                    LHS = (1 + θ * dt * ν * k2) * sparse(I, Nz, Nz) - θ * dt * ν * L.chebyshev_D2
                    lu(LHS)
                end
                RHS_mat = _get_or_factorize!(rhs_cache, (ikx, 1)) do
                    (1 - (1-θ) * dt * ν * k2) * sparse(I, Nz, Nz) + (1-θ) * dt * ν * L.chebyshev_D2
                end

                pencil_n = @view data_n[ikx, :]
                pencil_F_n = @view data_F_n[ikx, :]

                # Adams-Bashforth 2 extrapolation with variable-timestep coefficients
                mul!(rhs_buf, RHS_mat, pencil_n)
                @. rhs_buf += dt * ab2_c1 * pencil_F_n
                if data_F_nm1 !== nothing
                    pencil_F_nm1 = @view data_F_nm1[ikx, :]
                    @. rhs_buf -= dt * ab2_c2 * pencil_F_nm1
                end

                ldiv!(sol_buf, factor, rhs_buf)
                data_new[ikx, :] .= sol_buf
            end
        end

    elseif ndims == 2
        # 3D: (Nkx, Nky, Nz)
        nkx = size(data_n, 1)
        nky = size(data_n, 2)

        for iky in 1:nky
            for ikx in 1:nkx
                if ikx <= length(L.local_kx_range) && iky <= length(L.local_ky_range)
                    k2 = L.k2_values[ikx, iky]

                    factor = _get_or_factorize!(lhs_cache, (ikx, iky)) do
                        LHS = (1 + θ * dt * ν * k2) * sparse(I, Nz, Nz) - θ * dt * ν * L.chebyshev_D2
                        lu(LHS)
                    end
                    RHS_mat = _get_or_factorize!(rhs_cache, (ikx, iky)) do
                        (1 - (1-θ) * dt * ν * k2) * sparse(I, Nz, Nz) + (1-θ) * dt * ν * L.chebyshev_D2
                    end

                    pencil_n = @view data_n[ikx, iky, :]
                    pencil_F_n = @view data_F_n[ikx, iky, :]

                    mul!(rhs_buf, RHS_mat, pencil_n)
                    @. rhs_buf += dt * ab2_c1 * pencil_F_n
                    if data_F_nm1 !== nothing
                        pencil_F_nm1 = @view data_F_nm1[ikx, iky, :]
                        @. rhs_buf -= dt * ab2_c2 * pencil_F_nm1
                    end

                    ldiv!(sol_buf, factor, rhs_buf)
                    data_new[ikx, iky, :] .= sol_buf
                end
            end
        end
    end
end
