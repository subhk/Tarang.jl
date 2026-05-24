# =============================================================================
# Diagonal IMEX Step Functions (GPU-native)
# =============================================================================

"""
    step_diagonal_imex_rk222!(state::TimestepperState, solver::InitialValueSolver)

Diagonal IMEX RK step with GPU-native implicit treatment.

Uses a simplified IMEX-RK formulation where:
- Explicit tableau handles nonlinear terms F(u)
- Implicit diagonal operator L̂ handles linear terms (viscosity/hyperviscosity)

For each stage s, we solve:
    (1 + dt*γ*L̂) * Ŷ_s = X̂_n + dt * Σ_{j<s} a_j * F̂_j

where L̂ is the diagonal spectral operator.

This avoids sparse matrix solves and stays 100% on GPU.

!!! warning "Order reduction in stiff limit"
    This implementation omits the off-diagonal implicit contributions
    (- dt * Σ_{j<s} A_imp[s,j] * L̂ * Ŷ_j) from the stage RHS. This means
    the method is only **first-order accurate in the stiff limit** (dt*|λ| >> 1).
    For full 2nd-order IMEX-RK accuracy, use the dense IMEX methods in `step_rk.jl`.
"""
function step_diagonal_imex_rk222!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    # Get spectral linear operator from solver/problem
    L_spectral = _get_spectral_linear_operator(solver)

    if L_spectral === nothing
        # No spectral operator: fall back to explicit RK
        @debug "DiagonalIMEX_RK222: No spectral operator, using explicit RK"
        _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
        return
    end

    γ = ts.γ
    A = ts.A_explicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit
    stages = ts.stages

    # Stage storage for RHS evaluations
    F_stages = Vector{Vector{ScalarField}}(undef, stages)
    # Stage values
    Y_stages = Vector{Vector{ScalarField}}(undef, stages)

    # IMEX-RK with per-stage implicit solve, all in coefficient space:
    # For each stage s, solve (I + dt*γ*L̂) Ŷ_s = X̂_n + dt * Σ_{j<s} a_{sj} F̂_j
    # then evaluate the explicit RHS at the implicitly-solved stage value.
    #
    # WARNING: This implementation omits the off-diagonal implicit contributions
    # (- dt * Σ_{j<s} A_imp[s,j] * L̂ * Ŷ_j) from the stage RHS. For stiffly
    # accurate ESDIRK methods this means the method is only first-order accurate
    # in the stiff limit (dt*|λ| >> 1). The full IMEX-RK formulation is in step_rk.jl.

    # Ensure initial state is in coefficient space
    for field in current_state
        ensure_layout!(field, :c)
    end

    n_fields = length(current_state)

    for s in 1:stages
        state.current_substep = s

        # Build explicit contribution in coefficient space:
        # Ŷ_s = X̂_n + dt * Σ_{j<s} A[s,j] * F̂_j
        # Use workspace fields instead of copy_state to avoid allocation
        Y_s = Vector{ScalarField}(undef, n_fields)
        for (k, src_field) in enumerate(current_state)
            ws_idx = (s - 1) * n_fields + k
            ws_field = get_workspace_field!(state, src_field, ws_idx)
            copy_field_data!(ws_field, src_field)
            ws_field.current_layout = src_field.current_layout
            Y_s[k] = ws_field

            coeff_data = get_coeff_data(ws_field)
            for j in 1:(s-1)
                if abs(A[s, j]) > 1e-14
                    coeff_data .+= dt .* A[s, j] .* get_coeff_data(F_stages[j][k])
                end
            end

            # Apply per-stage implicit solve: Ŷ_s = RHS / (1 + dt*γ*L̂)
            coeff_data ./= (1 .+ dt .* γ .* L_spectral.coefficients)
        end

        Y_stages[s] = Y_s

        # Evaluate nonlinear RHS at implicitly-solved stage value
        F_stages[s] = evaluate_rhs(solver, Y_s, t + c[s] * dt)
    end

    # Final update in coefficient space using separate explicit and implicit weights:
    # X̂_{n+1} = X̂_n + dt * Σ b_exp[s] * F̂_s - dt * Σ b_imp[s] * L̂ * Ŷ_s
    new_state = copy_state(current_state)
    for (k, field) in enumerate(new_state)
        coeff_data = get_coeff_data(field)
        for s in 1:stages
            # Explicit contribution: +dt * b_exp[s] * F̂_s
            if abs(b_exp[s]) > 1e-14
                ensure_layout!(F_stages[s][k], :c)
                coeff_data .+= dt .* b_exp[s] .* get_coeff_data(F_stages[s][k])
            end
            # Implicit contribution: -dt * b_imp[s] * L̂ * Ŷ_s
            if abs(b_imp[s]) > 1e-14
                ensure_layout!(Y_stages[s][k], :c)
                coeff_data .-= dt .* b_imp[s] .* L_spectral.coefficients .* get_coeff_data(Y_stages[s][k])
            end
        end
    end

    _push_trim!(state.history, new_state, 1)
end

"""
    step_diagonal_imex_rk443!(state::TimestepperState, solver::InitialValueSolver)

Diagonal IMEX RK step with GPU-native implicit treatment (4 stages).

Uses classical RK4 explicit tableau for nonlinear terms, with implicit
treatment of linear operator at each stage and final update.

!!! warning "Order reduction in stiff limit"
    Like `step_diagonal_imex_rk222!`, this omits off-diagonal implicit
    contributions, reducing to first-order accuracy in the stiff limit.
    For full 3rd-order IMEX accuracy, use the dense IMEX methods in `step_rk.jl`.
"""
function step_diagonal_imex_rk443!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    # Get spectral linear operator from solver/problem
    L_spectral = _get_spectral_linear_operator(solver)

    if L_spectral === nothing
        @debug "DiagonalIMEX_RK443: No spectral operator, using explicit RK"
        _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
        return
    end

    A = ts.A_explicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit
    γ_diag = ts.A_implicit_diag
    stages = ts.stages

    # IMEX-RK with per-stage implicit solve (SDIRK structure), all in coefficient space:
    # For each stage s, solve (I + dt*γ_s*L̂) Ŷ_s = X̂_n + dt * Σ_{j<s} a_{sj} F̂_j
    F_stages = Vector{Vector{ScalarField}}(undef, stages)
    Y_stages = Vector{Vector{ScalarField}}(undef, stages)

    # Ensure initial state is in coefficient space
    for field in current_state
        ensure_layout!(field, :c)
    end

    n_fields = length(current_state)

    for s in 1:stages
        state.current_substep = s

        # Build explicit contribution in coefficient space:
        # Ŷ_s = X̂_n + dt * Σ_{j<s} A[s,j] * F̂_j
        # Use workspace fields instead of copy_state to avoid allocation
        Y_s = Vector{ScalarField}(undef, n_fields)
        γ_s = γ_diag[s]
        for (k, src_field) in enumerate(current_state)
            ws_idx = (s - 1) * n_fields + k
            ws_field = get_workspace_field!(state, src_field, ws_idx)
            copy_field_data!(ws_field, src_field)
            ws_field.current_layout = src_field.current_layout
            Y_s[k] = ws_field

            coeff_data = get_coeff_data(ws_field)
            for j in 1:(s-1)
                if abs(A[s, j]) > 1e-14
                    coeff_data .+= dt .* A[s, j] .* get_coeff_data(F_stages[j][k])
                end
            end

            # Apply per-stage implicit solve: Ŷ_s = RHS / (1 + dt*γ_s*L̂)
            coeff_data ./= (1 .+ dt .* γ_s .* L_spectral.coefficients)
        end

        Y_stages[s] = Y_s

        # Evaluate nonlinear RHS at implicitly-solved stage value
        F_stages[s] = evaluate_rhs(solver, Y_s, t + c[s] * dt)
    end

    # Final update in coefficient space using separate explicit and implicit weights:
    # X̂_{n+1} = X̂_n + dt * Σ b_exp[s] * F̂_s - dt * Σ b_imp[s] * L̂ * Ŷ_s
    new_state = copy_state(current_state)
    for (k, field) in enumerate(new_state)
        coeff_data = get_coeff_data(field)
        for s in 1:stages
            # Explicit contribution: +dt * b_exp[s] * F̂_s
            if abs(b_exp[s]) > 1e-14
                ensure_layout!(F_stages[s][k], :c)
                coeff_data .+= dt .* b_exp[s] .* get_coeff_data(F_stages[s][k])
            end
            # Implicit contribution: -dt * b_imp[s] * L̂ * Ŷ_s
            if abs(b_imp[s]) > 1e-14
                ensure_layout!(Y_stages[s][k], :c)
                coeff_data .-= dt .* b_imp[s] .* L_spectral.coefficients .* get_coeff_data(Y_stages[s][k])
            end
        end
    end

    _push_trim!(state.history, new_state, 1)
end

"""
    step_diagonal_imex_sbdf2!(state::TimestepperState, solver::InitialValueSolver)

2nd-order SBDF with diagonal spectral implicit treatment.

SBDF2 scheme:
    (3/2)X_{n+1} - 2X_n + (1/2)X_{n-1} = dt * (2F_n - F_{n-1}) - dt*L*X_{n+1}

Rearranged:
    (3/2 + dt*L) X_{n+1} = 2X_n - (1/2)X_{n-1} + dt*(2F_n - F_{n-1})

With diagonal spectral operator:
    X̂_{n+1} = RHS / (3/2 + dt*L̂)
"""
function step_diagonal_imex_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    L_spectral = _get_spectral_linear_operator(solver)

    # Initialize history if needed
    if !haskey(state.timestepper_data, :F_history)
        state.timestepper_data[:F_history] = Vector{ScalarField}[]
        state.timestepper_data[:iteration] = 0
    end

    iteration = state.timestepper_data[:iteration]::Int
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ScalarField}}

    # Evaluate current RHS
    F_n = evaluate_rhs(solver, current_state, t)

    if iteration == 0 || length(state.history) < 2
        # First step: use backward Euler (SBDF1)
        # (1 + dt*L) X_{n+1} = X_n + dt*F_n
        new_state = copy_state(current_state)
        axpy_state!(dt, F_n, new_state)

        if L_spectral !== nothing
            for field in new_state
                # X̂ = RHS / (1 + dt*L̂)
                ensure_layout!(field, :c)
                get_coeff_data(field) ./= (1 .+ dt .* L_spectral.coefficients)
            end
        end

        _push_trim!(state.history, new_state, 2)

        # Store F history
        _push_trim!(F_history, F_n, 2)
    else
        # SBDF2 step
        X_n = current_state
        X_nm1 = state.history[end-1]
        F_nm1 = F_history[end]

        # BDF2 + Adams-Bashforth extrapolation with constant-dt coefficients.
        # For variable timestep, these should use w = dt/dt_prev ratios.
        dt_prev = length(state.dt_history) >= 2 ? state.dt_history[end-1] : dt
        if !isapprox(dt, dt_prev, rtol=0.01)
            @warn "DiagonalIMEX_SBDF2 uses constant-dt coefficients but dt/dt_prev = $(dt/dt_prev). Use SBDF2 (step_multistep.jl) for variable timesteps." maxlog=1
        end
        # RHS = 2*X_n - 0.5*X_{n-1} + dt*(2*F_n - F_{n-1}); then divide by (1.5 + dt*L̂)
        new_state = copy_state(X_n)
        for (i, result) in enumerate(new_state)
            field_n   = X_n[i];   field_nm1 = X_nm1[i]
            f_n       = F_n[i];   f_nm1     = F_nm1[i]

            ensure_layout!(field_n, :c);  ensure_layout!(field_nm1, :c)
            ensure_layout!(f_n, :c);      ensure_layout!(f_nm1, :c)
            ensure_layout!(result, :c)

            d = get_coeff_data(result)
            dn = get_coeff_data(field_n);  dnm1 = get_coeff_data(field_nm1)
            fn = get_coeff_data(f_n);      fnm1 = get_coeff_data(f_nm1)

            @. d = 2*dn - 0.5*dnm1 + dt*(2*fn - fnm1)

            if L_spectral !== nothing
                @. d /= (1.5 + dt * L_spectral.coefficients)
            else
                d ./= 1.5
            end
        end

        _push_trim!(state.history, new_state, 2)

        # Update F history
        _push_trim!(F_history, F_n, 2)
    end

    state.timestepper_data[:iteration] = iteration + 1
end

# Note: _get_spectral_linear_operator and set_spectral_linear_operator! are
# defined in spectral_operators.jl which is included before this file.

# ============================================================================
# Distributed diagonal IMEX SBDF2 for MPI pure-Fourier problems
#
# Pure-Fourier problems never build subproblems (only mixed Fourier+Chebyshev
# do — see `_try_build_subproblems!`), so under MPI the multistep path falls
# back to an explicit RK that cannot stably integrate stiff implicit linear
# terms (e.g. hyperviscosity νΔ⁴), producing high-wavenumber noise. This path
# treats each time-stepped field's implicit linear operator DIAGONALLY in
# coefficient space using correct per-rank FFT-layout wavenumbers, and refreshes
# the algebraic constraints (Poisson, velocity, …) via the existing solver.
# ============================================================================

"""True when the distributed diagonal IMEX path should handle this solver:
MPI PencilArrays distribution and a fully-Fourier (separable) spatial domain."""
function _distributed_diagonal_imex_applicable(solver::InitialValueSolver)
    field = nothing
    for f in solver.state
        if !isempty(f.bases); field = f; break; end
    end
    field === nothing && return false
    (field.dist.use_pencil_arrays && field.dist.size > 1) || return false
    return all(b -> b === nothing || isa(b, FourierBasis), field.bases)
end

_local_coeff(cd) = cd isa PencilArrays.PencilArray ? parent(cd) : cd

"""
Diagonal implicit operator L̂(k) for a time-stepped field, built by parsing the
equation's linear operator `L_expr` into diagonal Fourier-space terms (constant
damping + fractional-Laplacian / Laplacian powers) over the per-rank |k|² grid.
Returns `nothing` if the operator contains a term that is not diagonal in a
pure-Fourier basis (caller then leaves the field to the explicit fallback).
"""
function _diagonal_Lhat_from_expr(L_expr, field::ScalarField)
    ensure_layout!(field, :c)
    cd = get_coeff_data(field)
    cd === nothing && return nothing
    (L_expr === nothing || is_zero_expression(L_expr)) && return zeros(Float64, size(_local_coeff(cd)))

    k2grid = compute_wavenumber_squared_grid(field)
    k2 = _local_coeff(k2grid)
    Lhat = zeros(Float64, size(k2))
    _accumulate_diagonal_L!(Lhat, k2, L_expr, 1.0) || return nothing
    return Lhat
end

"""Accumulate the diagonal Fourier-space contribution of a linear operator
expression into `Lhat` over the |k|² grid `k2`. Returns false on a term that is
not diagonal in a pure-Fourier basis."""
function _accumulate_diagonal_L!(Lhat, k2, expr, sgn::Float64)
    if isa(expr, AddOperator)
        return _accumulate_diagonal_L!(Lhat, k2, expr.left, sgn) &&
               _accumulate_diagonal_L!(Lhat, k2, expr.right, sgn)
    elseif isa(expr, SubtractOperator)
        return _accumulate_diagonal_L!(Lhat, k2, expr.left, sgn) &&
               _accumulate_diagonal_L!(Lhat, k2, expr.right, -sgn)
    elseif isa(expr, NegateOperator)
        return _accumulate_diagonal_L!(Lhat, k2, expr.operand, -sgn)
    elseif isa(expr, MultiplyOperator)
        if expr.left isa Number
            return _accumulate_diagonal_term!(Lhat, k2, sgn * Float64(expr.left), expr.right)
        elseif expr.right isa Number
            return _accumulate_diagonal_term!(Lhat, k2, sgn * Float64(expr.right), expr.left)
        end
        return false
    else
        return _accumulate_diagonal_term!(Lhat, k2, sgn, expr)
    end
end

function _accumulate_diagonal_term!(Lhat, k2, coeff::Float64, op)
    if isa(op, FractionalLaplacian)
        α = op.α
        @. Lhat += coeff * k2 ^ α
        return true
    elseif isa(op, Laplacian)
        # ∇² → −k²
        @. Lhat += coeff * (-k2)
        return true
    elseif isa(op, ScalarField)
        @. Lhat += coeff   # constant (e.g. linear damping μ·ζ)
        return true
    end
    return false
end

"""Build (and cache) the per-time-stepped-field diagonal L̂ operators."""
function _get_distributed_diagonal_Lhats!(state::TimestepperState, solver::InitialValueSolver)
    cached = get(state.timestepper_data, :dd_imex_Lhats, nothing)
    cached !== nothing && return cached::Dict{Int, Array{Float64}}

    problem = solver.problem
    sfields = solver.state
    Lhats = Dict{Int, Array{Float64}}()
    for eq_data in problem.equation_data
        M_expr = get(eq_data, "M", nothing)
        (M_expr === nothing || _is_zero_m_term(M_expr)) && continue
        targets = _find_time_derivative_targets(M_expr, sfields, problem.variables)
        L_expr = get(eq_data, "L", nothing)
        for idx in targets
            (idx isa Integer && 1 <= idx <= length(sfields)) || continue
            Lh = _diagonal_Lhat_from_expr(L_expr, sfields[idx])
            Lh === nothing && continue
            Lhats[Int(idx)] = Float64.(Lh)   # keep local-coeff shape for broadcasting
        end
    end
    state.timestepper_data[:dd_imex_Lhats] = Lhats
    return Lhats
end

# Function barriers: called with the concrete coefficient/operator array types
# (resolved via dynamic dispatch at the call), so the broadcast bodies are fully
# type-stable and allocation-free regardless of the abstract `_local_coeff`
# return type or the `Dict` value type at the call site.
@inline function _ddi_sbdf1_update!(d::AbstractArray, Lhat::AbstractArray, dt::Float64)
    @inbounds @. d /= (1.0 + dt * Lhat)
    return d
end

@inline function _ddi_sbdf2_update!(d::AbstractArray, dn::AbstractArray, dnm1::AbstractArray,
                                    fn::AbstractArray, fnm1::AbstractArray,
                                    Lhat::AbstractArray, dt::Float64, w::Float64)
    # Variable-dt SBDF2 (w = dtₙ/dtₙ₋₁): BDF2 implicit on the linear operator,
    # AB2 extrapolation of the nonlinear term. Reduces to the constant-dt scheme
    # (3/2, 2, -1/2, 2, -1) at w=1. Correct under CFL-adaptive timestepping.
    a0 = (1.0 + 2.0 * w) / (1.0 + w)
    a2 = w * w / (1.0 + w)
    @inbounds @. d = ((1.0 + w) * dn - a2 * dnm1 + dt * ((1.0 + w) * fn - w * fnm1)) / (a0 + dt * Lhat)
    return d
end

"""
    step_distributed_diagonal_imex_sbdf2!(state, solver)

SBDF2 with diagonal implicit treatment of each time-stepped field's linear
operator, for MPI pure-Fourier problems. Algebraic variables are refreshed via
`_refresh_algebraic_state!`. Uses variable-dt SBDF2 coefficients (w = dtₙ/dtₙ₋₁),
so CFL-adaptive timestepping is handled correctly.
"""
function step_distributed_diagonal_imex_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    Lhats = _get_distributed_diagonal_Lhats!(state, solver)

    if !haskey(state.timestepper_data, :dd_imex_Fhist)
        state.timestepper_data[:dd_imex_Fhist] = Vector{Vector{ScalarField}}()
        state.timestepper_data[:dd_imex_iter] = 0
    end
    iteration = state.timestepper_data[:dd_imex_iter]::Int
    Fhist = state.timestepper_data[:dd_imex_Fhist]::Vector{Vector{ScalarField}}

    F_n = evaluate_rhs(solver, current_state, t)

    if iteration == 0 || length(state.history) < 2
        # SBDF1 startup: (1 + dt·L̂) X_new = X_n + dt·F_n
        new_state = copy_state(current_state)
        axpy_state!(dt, F_n, new_state)
        for (i, field) in enumerate(new_state)
            haskey(Lhats, i) || continue
            ensure_layout!(field, :c)
            _ddi_sbdf1_update!(_local_coeff(get_coeff_data(field)), Lhats[i], dt)
        end
        _refresh_algebraic_state!(solver.problem, new_state)
        _push_trim!(state.history, new_state, 2)
        # evaluate_rhs may return reused buffer fields; store a copy so the next
        # step's F evaluation cannot overwrite this history entry (else F_{n-1}≡F_n).
        _push_trim!(Fhist, copy_state(F_n), 2)
    else
        dt_prev = length(state.dt_history) >= 2 ? state.dt_history[end-1] : dt
        w = dt / dt_prev   # variable-dt SBDF2 ratio (handles CFL-adaptive dt)
        X_n = current_state
        X_nm1 = state.history[end-1]
        F_nm1 = Fhist[end]
        new_state = copy_state(X_n)
        for (i, field) in enumerate(new_state)
            haskey(Lhats, i) || continue
            ensure_layout!(field, :c); ensure_layout!(X_n[i], :c); ensure_layout!(X_nm1[i], :c)
            ensure_layout!(F_n[i], :c); ensure_layout!(F_nm1[i], :c)
            _ddi_sbdf2_update!(_local_coeff(get_coeff_data(field)),
                               _local_coeff(get_coeff_data(X_n[i])),
                               _local_coeff(get_coeff_data(X_nm1[i])),
                               _local_coeff(get_coeff_data(F_n[i])),
                               _local_coeff(get_coeff_data(F_nm1[i])),
                               Lhats[i], dt, w)
        end
        _refresh_algebraic_state!(solver.problem, new_state)
        _push_trim!(state.history, new_state, 2)
        _push_trim!(Fhist, copy_state(F_n), 2)
    end

    state.timestepper_data[:dd_imex_iter] = iteration + 1
end

# ============================================================================
# Distributed diagonal ETD (exponential time differencing) for MPI pure-Fourier.
#
# In a pure-Fourier basis the linear operator is diagonal in coefficient space,
# so the matrix exponential and φ functions reduce to per-mode scalars. This
# avoids the global dense matrix exponential + serial gather that the standard
# ETD path uses (which cannot run distributed), letting ETD run under MPI.
# ============================================================================

"""Build (and cache, per dt) the per-field diagonal ETD operators
(exp(z), φ₁(z), φ₂(z)) with z = -dt·L̂, evaluated element-wise over each rank's
local wavenumber grid. The removable singularity at z=0 (e.g. the k=0 mode) is
handled by the Taylor branch."""
function _get_distributed_diagonal_phi!(state::TimestepperState, dt::Float64,
                                        Lhats::Dict{Int, Array{Float64}})
    cached = get(state.timestepper_data, :dd_etd_phi, nothing)
    cached_dt = get(state.timestepper_data, :dd_etd_phi_dt, nothing)
    if cached !== nothing && cached_dt == dt
        return cached::Dict{Int, NTuple{3, Array{Float64}}}
    end
    phis = Dict{Int, NTuple{3, Array{Float64}}}()
    for (i, Lhat) in Lhats
        expz = similar(Lhat); ph1 = similar(Lhat); ph2 = similar(Lhat)
        @inbounds for j in eachindex(Lhat)
            z = -dt * Lhat[j]
            ez = exp(z)
            expz[j] = ez
            if abs(z) < 1e-8
                ph1[j] = 1.0 + z/2 + z*z/6
                ph2[j] = 0.5 + z/6 + z*z/24
            else
                ph1[j] = (ez - 1.0) / z
                ph2[j] = (ez - 1.0 - z) / (z*z)
            end
        end
        phis[i] = (expz, ph1, ph2)
    end
    state.timestepper_data[:dd_etd_phi] = phis
    state.timestepper_data[:dd_etd_phi_dt] = dt
    return phis
end

# Function barriers (concrete array types resolved at the call → type-stable, alloc-free).
@inline function _ddetd_predictor!(d::AbstractArray, xn::AbstractArray, expz::AbstractArray,
                                   ph1::AbstractArray, Nn::AbstractArray, dt::Float64)
    @inbounds @. d = expz * xn + dt * ph1 * Nn
    return d
end

@inline function _ddetd_corrector!(d::AbstractArray, ph2::AbstractArray,
                                   Nc::AbstractArray, Nn::AbstractArray, dt::Float64)
    @inbounds @. d = d + dt * ph2 * (Nc - Nn)
    return d
end

"""
    step_distributed_diagonal_etd_rk222!(state, solver)

ETDRK2 (Cox-Matthews 2002) with the linear operator treated diagonally per
Fourier mode, for MPI pure-Fourier problems. Each time-stepped field is
propagated as `Xₙ₊₁ = c + dt·φ₂⊙(N(c) − N(Xₙ))`, `c = exp(z)⊙Xₙ + dt·φ₁⊙N(Xₙ)`,
with `z = -dt·L̂` element-wise. Algebraic variables are refreshed between stages.
"""
function step_distributed_diagonal_etd_rk222!(state::TimestepperState, solver::InitialValueSolver)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    Lhats = _get_distributed_diagonal_Lhats!(state, solver)
    phis = _get_distributed_diagonal_phi!(state, dt, Lhats)

    # N(Xₙ). evaluate_rhs may return reused buffers, so copy before re-evaluating.
    N_n = copy_state(evaluate_rhs(solver, current_state, t))

    # Predictor: c = exp(z)⊙Xₙ + dt·φ₁⊙N(Xₙ)
    pred = copy_state(current_state)
    for (i, field) in enumerate(pred)
        haskey(phis, i) || continue
        ensure_layout!(field, :c); ensure_layout!(current_state[i], :c); ensure_layout!(N_n[i], :c)
        expz, ph1, _ = phis[i]
        _ddetd_predictor!(_local_coeff(get_coeff_data(field)),
                          _local_coeff(get_coeff_data(current_state[i])),
                          expz, ph1, _local_coeff(get_coeff_data(N_n[i])), dt)
    end

    # Corrector: Xₙ₊₁ = c + dt·φ₂⊙(N(c) − N(Xₙ)), written in place into `pred`.
    # evaluate_rhs refreshes pred's algebraic state internally (no separate refresh).
    N_c = evaluate_rhs(solver, pred, t + dt)
    for (i, field) in enumerate(pred)
        haskey(phis, i) || continue
        ensure_layout!(field, :c); ensure_layout!(N_c[i], :c); ensure_layout!(N_n[i], :c)
        _, _, ph2 = phis[i]
        _ddetd_corrector!(_local_coeff(get_coeff_data(field)), ph2,
                          _local_coeff(get_coeff_data(N_c[i])),
                          _local_coeff(get_coeff_data(N_n[i])), dt)
    end
    _refresh_algebraic_state!(solver.problem, pred)
    _push_trim!(state.history, pred, 2)
end

# ============================================================================
# Distributed diagonal IMEX Runge-Kutta for MPI pure-Fourier problems.
#
# The global IMEX-RK path solves (M + dt·aᴵ_ss·L)·Yₛ = RHS per stage with a
# global matrix; under MPI (no subproblems) it instead falls back to a fully
# EXPLICIT RK, which cannot integrate stiff implicit linear terms (νΔ⁴, μ) and
# produces high-wavenumber blowup. Here each stage's implicit solve is diagonal
# in Fourier space: Yₛ = RHS / (1 + dt·aᴵ_ss·L̂(k)) per rank-local mode.
# ============================================================================

"""
    step_distributed_diagonal_imex_rk!(state, solver, ts)

IMEX Runge-Kutta (ARS-type tableau `ts`) with per-mode diagonal implicit
treatment of each time-stepped field's linear operator, for MPI pure-Fourier
problems. Stage solve: `(1 + dt·aᴵ_ss·L̂)·Yₛ = Xₙ + dt·Σ_{j<s}(aᴱ_sj·F_j − aᴵ_sj·L̂·Y_j)`;
update `Xₙ₊₁ = Xₙ + dt·Σ_s(bᴱ_s·F_s − bᴵ_s·L̂·Y_s)`. Algebraic variables are
refreshed per stage via `_refresh_algebraic_state!`.
"""
function step_distributed_diagonal_imex_rk!(state::TimestepperState, solver::InitialValueSolver, ts)
    X_n = state.history[end]
    dt = state.dt
    t = solver.sim_time
    Lhats = _get_distributed_diagonal_Lhats!(state, solver)
    S = ts.stages
    AE = ts.A_explicit; AI = ts.A_implicit
    bE = ts.b_explicit; bI = ts.b_implicit; cc = ts.c_explicit

    Ys = Vector{Vector{ScalarField}}(undef, S)   # stage states
    Fs = Vector{Vector{ScalarField}}(undef, S)    # F(Y_s), copied (evaluate_rhs reuses buffers)

    for s in 1:S
        Y = copy_state(X_n)
        for (i, field) in enumerate(Y)
            haskey(Lhats, i) || continue
            ensure_layout!(field, :c)
            d = _local_coeff(get_coeff_data(field))           # starts as X_n[i]
            Lhat = Lhats[i]
            for j in 1:s-1
                if AE[s, j] != 0.0
                    ensure_layout!(Fs[j][i], :c)
                    fj = _local_coeff(get_coeff_data(Fs[j][i]))
                    _ddirk_axpy!(d, dt * AE[s, j], fj)
                end
                if AI[s, j] != 0.0
                    ensure_layout!(Ys[j][i], :c)
                    yj = _local_coeff(get_coeff_data(Ys[j][i]))
                    _ddirk_axpy_lhat!(d, -dt * AI[s, j], Lhat, yj)
                end
            end
            if AI[s, s] != 0.0
                _ddirk_implicit_divide!(d, Lhat, dt * AI[s, s])
            end
        end
        Ys[s] = Y
        # evaluate_rhs refreshes the algebraic state of Y internally, so no
        # separate _refresh_algebraic_state! is needed here (avoids a redundant
        # constraint solve + its transforms per stage).
        Fs[s] = copy_state(evaluate_rhs(solver, Y, t + cc[s] * dt))
    end

    X_new = copy_state(X_n)
    for (i, field) in enumerate(X_new)
        haskey(Lhats, i) || continue
        ensure_layout!(field, :c)
        d = _local_coeff(get_coeff_data(field))
        Lhat = Lhats[i]
        for s in 1:S
            if bE[s] != 0.0
                ensure_layout!(Fs[s][i], :c)
                fs = _local_coeff(get_coeff_data(Fs[s][i]))
                _ddirk_axpy!(d, dt * bE[s], fs)
            end
            if bI[s] != 0.0
                ensure_layout!(Ys[s][i], :c)
                ys = _local_coeff(get_coeff_data(Ys[s][i]))
                _ddirk_axpy_lhat!(d, -dt * bI[s], Lhat, ys)
            end
        end
    end
    _refresh_algebraic_state!(solver.problem, X_new)
    _push_trim!(state.history, X_new, 2)
end

# Function barriers (concrete array types resolved at the call → type-stable).
@inline function _ddirk_axpy!(d::AbstractArray, a::Float64, x::AbstractArray)
    @inbounds @. d += a * x
    return d
end
@inline function _ddirk_axpy_lhat!(d::AbstractArray, a::Float64, Lhat::AbstractArray, x::AbstractArray)
    @inbounds @. d += a * Lhat * x
    return d
end
@inline function _ddirk_implicit_divide!(d::AbstractArray, Lhat::AbstractArray, c::Float64)
    @inbounds @. d /= (1.0 + c * Lhat)
    return d
end
