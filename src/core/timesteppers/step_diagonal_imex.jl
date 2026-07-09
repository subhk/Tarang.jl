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

Uses the full ARS(2,2,2) ESDIRK tableau (including off-diagonal implicit terms),
so it is L-stable and 2nd-order — identical math to `RK222`, but with the
implicit solve done diagonally per Fourier mode instead of via a global matrix.
"""
function step_diagonal_imex_rk222!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    L_spectral = _get_spectral_linear_operator(solver)
    if L_spectral === nothing
        @debug "DiagonalIMEX_RK222: No spectral operator, using explicit RK"
        _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
        return
    end
    _step_diagonal_imex_rk_impl!(state, solver, ts, L_spectral)
    return nothing
end

# Function barrier: L is now a concrete SpectralLinearOperator{T,N,A}, so
# L.coefficients has concrete type A and all broadcasts below are type-stable.
#
# Unified IMEX-RK step for the GPU-native diagonal steppers (DiagonalIMEX_RK222 /
# RK443). The implicit operator L̂ is diagonal in coefficient space, so each stage
# solve is a per-mode division — but we use the FULL ESDIRK tableau `ts.A_implicit`
# including the off-diagonal terms (−dt·Σ_{j<s} AI[s,j]·L̂·Y_j). Dropping those (as
# the previous implementation did) makes the method unstable in the stiff limit
# (R(z)→1−1/γ, |R|>1 for dt·λ≳5) rather than L-stable. This mirrors the math of
# the distributed sibling `step_distributed_diagonal_imex_rk!`.
function _step_diagonal_imex_rk_impl!(state::TimestepperState, solver::InitialValueSolver,
                                       ts::TimeStepper, L::SpectralLinearOperator)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    AE = ts.A_explicit
    AI = ts.A_implicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit
    stages = ts.stages

    F_stages = Vector{Vector{ScalarField}}(undef, stages)
    n_fields = length(current_state)
    # Pre-allocate Y_s containers once; reuse across stages (each inner slot gets
    # replaced by a workspace field reference, so no per-stage Vector allocation).
    Y_stages = [Vector{ScalarField}(undef, n_fields) for _ in 1:stages]

    for field in current_state
        ensure_layout!(field, :c)
    end

    for s in 1:stages
        state.current_substep = s
        Y_s = Y_stages[s]
        for (k, src_field) in enumerate(current_state)
            ws_idx = (s - 1) * n_fields + k
            ws_field = get_workspace_field!(state, src_field, ws_idx)
            copy_field_data!(ws_field, src_field)
            # Sync the coefficient buffer: copy_field_data! copies GRID data and
            # leaves the field in :g, so merely flagging current_layout=:c would
            # leave a stale (zero) coeff buffer — every in-place stage edit below
            # would then be discarded when the field is re-read in :c (silently
            # degrading the whole method to explicit forward Euler). ensure_layout!
            # transforms grid→coeff so the implicit solve acts on the real X_n[k].
            ensure_layout!(ws_field, :c)
            Y_s[k] = ws_field

            coeff_data = get_coeff_data(ws_field)   # live coeff, = X_n[k]
            for j in 1:(s-1)
                if abs(AE[s, j]) > 1e-14
                    coeff_data .+= dt .* AE[s, j] .* get_coeff_data(F_stages[j][k])
                end
                if abs(AI[s, j]) > 1e-14
                    # Off-diagonal implicit contribution −dt·AI[s,j]·L̂·Y_j (the
                    # term whose omission caused the stiff-limit instability).
                    ensure_layout!(Y_stages[j][k], :c)
                    coeff_data .-= dt .* AI[s, j] .* L.coefficients .* get_coeff_data(Y_stages[j][k])
                end
            end
            # Diagonal implicit solve (1 + dt·AI[s,s]·L̂)·Y_s = RHS. For the ESDIRK
            # explicit first stage AI[1,1]=0, so this is a no-op there.
            γ_s = AI[s, s]
            if abs(γ_s) > 1e-14
                coeff_data ./= (1 .+ dt .* γ_s .* L.coefficients)
            end
        end
        # evaluate_rhs may return reused buffer fields; copy so a later stage's RHS
        # evaluation cannot overwrite an earlier stage's stored F (matches the
        # distributed sibling).
        F_stages[s] = copy_state(evaluate_rhs(solver, Y_s, t + c[s] * dt))
    end

    new_state = copy_state(current_state)
    for (k, field) in enumerate(new_state)
        # copy_state may return the field in grid layout with a stale coefficient
        # buffer; normalize to :c so the implicit update below writes the
        # authoritative data (otherwise the edits are discarded when the field is
        # next read in :c from its grid, and the state never evolves).
        ensure_layout!(field, :c)
        coeff_data = get_coeff_data(field)
        for s in 1:stages
            if abs(b_exp[s]) > 1e-14
                ensure_layout!(F_stages[s][k], :c)
                coeff_data .+= dt .* b_exp[s] .* get_coeff_data(F_stages[s][k])
            end
            if abs(b_imp[s]) > 1e-14
                ensure_layout!(Y_stages[s][k], :c)
                coeff_data .-= dt .* b_imp[s] .* L.coefficients .* get_coeff_data(Y_stages[s][k])
            end
        end
    end

    _push_trim!(state.history, new_state, 1)
end

"""
    step_diagonal_imex_rk443!(state::TimestepperState, solver::InitialValueSolver)

Diagonal IMEX RK step with GPU-native implicit treatment (4 stages).

Uses the Kennedy-Carpenter ARK3(2)4L[2]SA tableau (explicit ERK + full ESDIRK
implicit, including off-diagonal terms), so it is L-stable and 3rd-order —
identical math to `RK443`, with the implicit solve done diagonally per Fourier
mode instead of via a global matrix.
"""
function step_diagonal_imex_rk443!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    L_spectral = _get_spectral_linear_operator(solver)
    if L_spectral === nothing
        @debug "DiagonalIMEX_RK443: No spectral operator, using explicit RK"
        _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
        return
    end
    _step_diagonal_imex_rk_impl!(state, solver, ts, L_spectral)
    return nothing
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

    if !haskey(state.timestepper_data, :F_history)
        state.timestepper_data[:F_history] = Vector{ScalarField}[]
        state.timestepper_data[:iteration] = 0
    end

    iteration = state.timestepper_data[:iteration]::Int
    F_history = state.timestepper_data[:F_history]::Vector{Vector{ScalarField}}

    F_n = evaluate_rhs(solver, current_state, t)

    if iteration == 0 || length(state.history) < 2
        new_state = copy_state(current_state)
        axpy_state!(dt, F_n, new_state)
        if L_spectral !== nothing
            _sbdf2_apply_be_L!(new_state, L_spectral, dt)
        end
        _push_trim!(state.history, new_state, 2)
        # evaluate_rhs returns reused buffer fields; copy before storing so the next
        # step's RHS evaluation cannot overwrite this history entry (else F_{n-1}≡F_n
        # and the AB2 extrapolation 2F_n−F_{n-1} collapses to first order).
        _push_trim!(F_history, copy_state(F_n), 2)
    else
        X_n = current_state
        X_nm1 = state.history[end-1]
        F_nm1 = F_history[end]

        dt_prev = length(state.dt_history) >= 2 ? state.dt_history[end-1] : dt

        new_state = copy_state(X_n)
        if L_spectral !== nothing
            _sbdf2_apply_bdf2_L!(new_state, X_n, X_nm1, F_n, F_nm1, dt, dt_prev, L_spectral)
        else
            # Variable-dt SBDF2 (L̂ = 0, no spectral linear operator). w = dtₙ/dtₙ₋₁;
            # reduces to the constant-dt (2, −½, 2, −1)/1.5 weights at w = 1. The previous
            # hardcoded constant-dt coefficients were inconsistent under adaptive dt.
            w = dt / dt_prev
            a0 = (1.0 + 2.0w) / (1.0 + w)
            a2 = w * w / (1.0 + w)
            for (i, result) in enumerate(new_state)
                field_n = X_n[i];  field_nm1 = X_nm1[i]
                f_n = F_n[i];      f_nm1 = F_nm1[i]
                ensure_layout!(field_n, :c);  ensure_layout!(field_nm1, :c)
                ensure_layout!(f_n, :c);      ensure_layout!(f_nm1, :c)
                ensure_layout!(result, :c)
                d = get_coeff_data(result)
                @. d = ((1.0 + w) * get_coeff_data(field_n) - a2 * get_coeff_data(field_nm1) +
                        dt * ((1.0 + w) * get_coeff_data(f_n) - w * get_coeff_data(f_nm1))) / a0
            end
        end

        _push_trim!(state.history, new_state, 2)
        _push_trim!(F_history, copy_state(F_n), 2)  # copy: see startup branch above
    end

    state.timestepper_data[:iteration] = iteration + 1
end

# Function barriers: L has concrete SpectralLinearOperator{T,N,A} type here,
# so L.coefficients is concrete type A and broadcasts are type-stable.
function _sbdf2_apply_be_L!(fields::Vector{<:ScalarField}, L::SpectralLinearOperator, dt::Float64)
    for field in fields
        ensure_layout!(field, :c)
        _ddi_sbdf1_update!(get_coeff_data(field), L.coefficients, dt)
    end
end

function _sbdf2_apply_bdf2_L!(new_state::Vector{<:ScalarField},
                               X_n::Vector{<:ScalarField}, X_nm1::Vector{<:ScalarField},
                               F_n::Vector{<:ScalarField}, F_nm1::Vector{<:ScalarField},
                               dt::Float64, dt_prev::Float64, L::SpectralLinearOperator)
    w = dt / dt_prev
    for (i, result) in enumerate(new_state)
        field_n = X_n[i];  field_nm1 = X_nm1[i]
        f_n = F_n[i];      f_nm1 = F_nm1[i]
        ensure_layout!(field_n, :c);  ensure_layout!(field_nm1, :c)
        ensure_layout!(f_n, :c);      ensure_layout!(f_nm1, :c)
        ensure_layout!(result, :c)
        _ddi_sbdf2_update!(get_coeff_data(result),
                           get_coeff_data(field_n), get_coeff_data(field_nm1),
                           get_coeff_data(f_n), get_coeff_data(f_nm1),
                           L.coefficients, dt, w)
    end
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
    # L̂ is ComplexF64: Laplacian/fractional/damping terms are real (−k², k²^α, const),
    # but first-derivative terms contribute imaginary (ik)^order multipliers.
    (L_expr === nothing || is_zero_expression(L_expr)) && return zeros(ComplexF64, size(_local_coeff(cd)))

    k2grid = compute_wavenumber_squared_grid(field)
    k2 = _local_coeff(k2grid)
    Lhat = zeros(ComplexF64, size(k2))
    _accumulate_diagonal_L!(Lhat, k2, L_expr, 1.0, field) || return nothing
    return Lhat
end

"""Accumulate the diagonal Fourier-space contribution of a linear operator
expression into `Lhat` over the |k|² grid `k2`. Returns false on a term that is
not diagonal in a pure-Fourier basis."""
function _accumulate_diagonal_L!(Lhat, k2, expr, sgn::Float64, field::ScalarField)
    if isa(expr, AddOperator)
        return _accumulate_diagonal_L!(Lhat, k2, expr.left, sgn, field) &&
               _accumulate_diagonal_L!(Lhat, k2, expr.right, sgn, field)
    elseif isa(expr, SubtractOperator)
        return _accumulate_diagonal_L!(Lhat, k2, expr.left, sgn, field) &&
               _accumulate_diagonal_L!(Lhat, k2, expr.right, -sgn, field)
    elseif isa(expr, NegateOperator)
        return _accumulate_diagonal_L!(Lhat, k2, expr.operand, -sgn, field)
    elseif isa(expr, MultiplyOperator)
        # The scalar coefficient may be a raw Number (inlined parameter) or a
        # ConstantOperator (parsed numeric literal, e.g. `0.3*d(q,x)`).
        cl = _as_diagonal_scalar(expr.left)
        if cl !== nothing
            return _accumulate_diagonal_term!(Lhat, k2, sgn * cl, expr.right, field)
        end
        cr = _as_diagonal_scalar(expr.right)
        if cr !== nothing
            return _accumulate_diagonal_term!(Lhat, k2, sgn * cr, expr.left, field)
        end
        return false
    else
        return _accumulate_diagonal_term!(Lhat, k2, sgn, expr, field)
    end
end

"""Extract a real scalar from a `Number` or `ConstantOperator`; `nothing` otherwise."""
_as_diagonal_scalar(x) = x isa Number ? Float64(x) :
                         (isa(x, ConstantOperator) ? Float64(x.value) : nothing)

function _accumulate_diagonal_term!(Lhat, k2, coeff::Float64, op, field::ScalarField)
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
    elseif isa(op, Differentiate)
        # First/higher derivative of the time-stepped field itself is diagonal in
        # Fourier space: multiplier (i·k_axis)^order. Only diagonal when the
        # derivative acts on THIS field (a derivative of another variable is a
        # coupling term, not representable as a scalar diagonal L̂).
        operand = op.operand
        is_self = operand === field ||
                  (isa(operand, ScalarField) && operand.name == field.name)
        is_self || return false
        ikgrid = _diagonal_deriv_grid(field, op.coord, op.order)
        ikgrid === nothing && return false
        @. Lhat += coeff * ikgrid
        return true
    end
    return false
end

"""
Diagonal Fourier multiplier `(i·k)^order` for a single-axis derivative, evaluated
over `field`'s LOCAL coefficient grid (per-rank under MPI, rfft/fft layout, and
PencilArrays permutation aware — mirrors `compute_wavenumber_squared_grid`).
Returns `nothing` if `coord` maps to no Fourier axis of `field`.
"""
function _diagonal_deriv_grid(field::ScalarField, coord::Coordinate, order::Int)
    bases = field.bases
    cd = get_coeff_data(field)
    cd === nothing && return nothing
    axis = _resolve_diff_axis(coord, bases)
    axis == 0 && return nothing
    basis = bases[axis]
    isa(basis, Union{RealFourier, ComplexFourier}) || return nothing

    k_axis = if isa(basis, RealFourier)
        _is_first_real_fourier_axis(bases, axis) ? wavenumbers_rfft(basis) : wavenumbers_fft(basis)
    else
        wavenumbers(basis)
    end

    lc = _local_coeff(cd)
    out = zeros(ComplexF64, size(lc))
    if isa(cd, PencilArrays.PencilArray)
        local_axes = PencilArrays.pencil(cd).axes_local
        # Tuple(NoPermutation()) is `nothing`, NOT identity — guard before findfirst.
        _perm_raw = Tuple(PencilArrays.permutation(cd))
        perm_tuple = _perm_raw === nothing ? ntuple(identity, ndims(out)) : _perm_raw
        lr = axis <= length(local_axes) ? local_axes[axis] : (1:length(k_axis))
        ik = ComplexF64.((im .* Float64.(k_axis[lr])) .^ order)
        paxis = findfirst(==(axis), perm_tuple)
        paxis === nothing && (paxis = axis)
        shp = ntuple(i -> i == paxis ? length(ik) : 1, ndims(out))
        out .+= reshape(ik, shp...)
    else
        ik = ComplexF64.((im .* Float64.(k_axis)) .^ order)
        shp = ntuple(i -> i == axis ? length(ik) : 1, ndims(out))
        out .+= reshape(ik, shp...)
    end
    return out
end

"""Build (and cache) the per-time-stepped-field diagonal L̂ operators."""
function _get_distributed_diagonal_Lhats!(state::TimestepperState, solver::InitialValueSolver)
    cached = get(state.timestepper_data, :dd_imex_Lhats, nothing)
    cached !== nothing && return cached::Dict{Int, Array{ComplexF64}}

    problem = solver.problem
    sfields = solver.state
    Lhats = Dict{Int, Array{ComplexF64}}()
    for eq_data in problem.equation_data
        M_expr = get(eq_data, "M", nothing)
        (M_expr === nothing || _is_zero_m_term(M_expr)) && continue
        targets = _find_time_derivative_targets(M_expr, sfields, problem.variables)
        L_expr = get(eq_data, "L", nothing)
        for idx in targets
            (idx isa Integer && 1 <= idx <= length(sfields)) || continue
            Lh = _diagonal_Lhat_from_expr(L_expr, sfields[idx])
            if Lh === nothing
                # A non-zero implicit linear operator that is NOT diagonal in a
                # pure-Fourier basis cannot be applied per-mode. Refuse loudly:
                # the old behavior silently skipped (froze) the field — a correctness bug.
                error("Distributed diagonal-IMEX: the implicit linear operator for field " *
                      "'$(sfields[idx].name)' has a term that is not diagonal in a pure-Fourier " *
                      "basis. Supported implicit terms: Laplacian, hyper/fractional Laplacian, " *
                      "constant damping, and derivatives of the field itself. Move the offending " *
                      "term to the explicit RHS, or run serially / via the Chebyshev subproblem path.")
            end
            Lhats[Int(idx)] = ComplexF64.(Lh)   # keep local-coeff shape for broadcasting
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
        # Route pushes through the recycle-aware helpers so the state/F rings are
        # seeded: from the next step on, the dropped field-sets are reused in place
        # of copy_state (see _ddirk_acquire_xnew! / _ddi_sbdf2_push_fhist!). The F
        # copy still guards F_{n-1}≢F_n (evaluate_rhs reuses buffers).
        _ddirk_push_recycle!(state, new_state)
        _ddi_sbdf2_push_fhist!(state, Fhist, F_n, length(F_n))
    else
        dt_prev = length(state.dt_history) >= 2 ? state.dt_history[end-1] : dt
        w = dt / dt_prev   # variable-dt SBDF2 ratio (handles CFL-adaptive dt)
        X_n = current_state
        X_nm1 = state.history[end-1]
        F_nm1 = Fhist[end]
        # new_state reuses the history-dropped field-set (X_{n-2}, provably out of
        # the live window: SBDF2 reads back only to history[end-1]=X_{n-1}) instead
        # of a fresh copy_state(X_n). Seeded with X_n's coeffs; _ddi_sbdf2_update!
        # overwrites every stepped field and _refresh_algebraic_state! the rest.
        new_state = _ddirk_acquire_xnew!(state, X_n, length(X_n))
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
        _ddirk_push_recycle!(state, new_state)
        _ddi_sbdf2_push_fhist!(state, Fhist, F_n, length(F_n))
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
                                        Lhats::Dict{Int, Array{ComplexF64}})
    cached = get(state.timestepper_data, :dd_etd_phi, nothing)
    cached_dt = get(state.timestepper_data, :dd_etd_phi_dt, nothing)
    if cached !== nothing && cached_dt == dt
        return cached::Dict{Int, NTuple{3, Array{ComplexF64}}}
    end
    phis = Dict{Int, NTuple{3, Array{ComplexF64}}}()
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

    # N(Xₙ) must survive the corrector's second evaluate_rhs (which reuses buffers).
    # Reuse a persistent 1-slot cache + coeff-copy instead of copy_state every step;
    # only φ-fields (those read by the predictor/corrector) are written.
    n_fields = length(current_state)
    _Nn_src = evaluate_rhs(solver, current_state, t)
    N_n = _ddetd_nn_cache!(state, current_state, n_fields)
    @inbounds for i in 1:n_fields
        haskey(phis, i) || continue
        fr = _Nn_src[i]
        ensure_layout!(fr, :c)
        _ddirk_copy!(_local_coeff(get_coeff_data(N_n[i])), _local_coeff(get_coeff_data(fr)))
        N_n[i].current_layout = :c
    end

    # Predictor: c = exp(z)⊙Xₙ + dt·φ₁⊙N(Xₙ). `pred` reuses the history-dropped
    # field-set (ETD is one-step — reads only history[end]=current_state — so the
    # recycled set is provably not live) instead of a fresh copy_state.
    pred = _ddirk_acquire_xnew!(state, current_state, n_fields)
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
    _ddirk_push_recycle!(state, pred)
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

    n_fields = length(X_n)
    Ys = Vector{Vector{ScalarField}}(undef, S)   # stage states (workspace-backed)
    Fs = Vector{Vector{ScalarField}}(undef, S)    # F(Y_s), copied (evaluate_rhs reuses buffers)

    # Persistent per-stage F storage, reused across steps (replaces the per-stage
    # `Fs[s] = copy_state(evaluate_rhs(...))` full-state allocation). evaluate_rhs
    # reuses its output buffers across stages, so each stage's F must be RETAINED
    # until the final update — we copy it (coeff space) into this dedicated cache
    # instead of allocating S fresh field-sets every step. The workspace POOL is
    # unusable for this: Y already consumes up to S·n_fields of the
    # `_workspace_count(ts)` (=4 for the DiagonalIMEX tableaux) sets.
    Fs_cache = _ddirk_fs_cache!(state, X_n, S, n_fields)

    for s in 1:S
        # Stage state Yₛ reuses the timestepper's pre-allocated workspace fields
        # (one distinct slot per (stage,field): ws_idx = (s-1)*n_fields + i) instead
        # of a fresh full-state copy_state allocation per stage. Earlier stages' Yⱼ
        # stay live for the off-diagonal implicit term and the final update because
        # each (s,i) maps to its own slot. Budget: S·n_fields ≤
        # _workspace_count(ts)·n_fields — this path runs under RK111/RK222/RK443
        # (3 / 6 / 12 sets), so S (1 / 2 / 4) always fits. The copy is done in
        # COEFFICIENT space (no grid↔coeff transforms) — exact here since the
        # problem is pure-Fourier and X_n is forced to :c first. Mirrors the serial
        # sibling _step_diagonal_imex_rk_impl! (get_workspace_field! + per-stage idx).
        Y = Vector{ScalarField}(undef, n_fields)
        for i in 1:n_fields
            src = X_n[i]
            ws = get_workspace_field!(state, src, (s - 1) * n_fields + i)
            if !isempty(src.bases)
                ensure_layout!(src, :c)
                _ddirk_copy!(_local_coeff(get_coeff_data(ws)), _local_coeff(get_coeff_data(src)))
                ws.current_layout = :c
            end
            Y[i] = ws
            haskey(Lhats, i) || continue   # algebraic vars: refreshed inside evaluate_rhs
            d = _local_coeff(get_coeff_data(ws))              # starts as X_n[i]
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
        # constraint solve + its transforms per stage). Copy the stage RHS into
        # the persistent cache in COEFFICIENT space — only the fields actually
        # read downstream (those with a diagonal L̂) are copied; algebraic/0D
        # fields' F is never used by the explicit accumulation.
        F_result = evaluate_rhs(solver, Y, t + cc[s] * dt)
        dst = Fs_cache[s]
        @inbounds for i in 1:n_fields
            haskey(Lhats, i) || continue
            fr = F_result[i]
            # A field with a diagonal L̂ always has spatial bases (its RHS is
            # non-0D). The downstream reads are gated by `haskey(Lhats,i)` ALONE,
            # so this entry MUST be written — fail loud rather than skip (which
            # would leave the read seeing a stale previous-step F).
            isempty(fr.bases) && error("diagonal-IMEX RK: field $i has a diagonal L̂ but an empty-bases RHS")
            ensure_layout!(fr, :c)
            _ddirk_copy!(_local_coeff(get_coeff_data(dst[i])), _local_coeff(get_coeff_data(fr)))
            dst[i].current_layout = :c
        end
        Fs[s] = dst
    end

    # X_new reuses the field-set that `_push_trim!`-to-2 dropped from history last
    # step (out of history, no longer `solver.state`, never read by this ONE-step
    # RK method) instead of a fresh `copy_state(X_n)`. Falls back to copy_state on
    # the first steps (history not yet full). The coeff-space seed-copy of X_n
    # below is exact; algebraic/0D fields are refreshed by `_refresh_algebraic_state!`.
    X_new = _ddirk_acquire_xnew!(state, X_n, n_fields)
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
    _ddirk_push_recycle!(state, X_new)
end

# ── Persistent-storage helpers for step_distributed_diagonal_imex_rk! ────────
# All reuse field-sets across steps to drop the S+1 per-step `copy_state` allocs.

"""Cached per-stage F storage (`S` field-sets shaped like `X_n`), reused every
step. (Re)built only when missing, too few sets (timestepper switch grows `S`),
or shape-mismatched. `X_n` is forced to `:c` first so the `copy_state` templates
carry allocated coefficient arrays for the in-place coeff copies."""
function _ddirk_fs_cache!(state::TimestepperState, X_n, S::Int, n_fields::Int)
    cache = get!(() -> Vector{Vector{ScalarField}}(),
                 state.timestepper_data, :_ddirk_Fs)::Vector{Vector{ScalarField}}
    if length(cache) < S || (!isempty(cache) && length(cache[1]) != n_fields)
        for f in X_n
            isempty(f.bases) || ensure_layout!(f, :c)
        end
        if !isempty(cache) && length(cache[1]) != n_fields
            empty!(cache)
        end
        while length(cache) < S
            push!(cache, copy_state(X_n))
        end
    end
    return cache
end

"""Acquire X_new storage: reuse the history-dropped field-set stashed by
`_ddirk_push_recycle!` (seeding it with X_n's coefficients in place), else
`copy_state(X_n)`. The recycled set is provably out of `history` and not
`solver.state`, so reusing it cannot alias a live state."""
function _ddirk_acquire_xnew!(state::TimestepperState, X_n, n_fields::Int)
    rec = get(state.timestepper_data, :_ddirk_recycle, nothing)
    if rec isa Vector{ScalarField} && length(rec) == n_fields
        state.timestepper_data[:_ddirk_recycle] = nothing
        @inbounds for i in 1:n_fields
            isempty(X_n[i].bases) && continue
            ensure_layout!(X_n[i], :c)
            _ddirk_copy!(_local_coeff(get_coeff_data(rec[i])), _local_coeff(get_coeff_data(X_n[i])))
            rec[i].current_layout = :c
        end
        return rec
    end
    # Guard failed (first steps, or a shape mismatch after a problem change):
    # drop any stale stashed set so it can't linger, and allocate fresh.
    state.timestepper_data[:_ddirk_recycle] = nothing
    return copy_state(X_n)
end

"""Push X_new and trim history to 2, stashing the dropped field-set for reuse as
the next step's X_new (see `_ddirk_acquire_xnew!`)."""
function _ddirk_push_recycle!(state::TimestepperState, X_new)
    push!(state.history, X_new)
    while length(state.history) > 2
        state.timestepper_data[:_ddirk_recycle] = popfirst!(state.history)
    end
    return state.history
end

"""Persistent single field-set holding N(Xₙ) across the ETD corrector's second
`evaluate_rhs` (which reuses buffers), reused every step in place of
`copy_state(evaluate_rhs(...))`. Mirrors `_ddirk_fs_cache!` with one set; the
caller writes only φ-fields, non-φ slots are never read. (Re)built on missing set
or field-count change."""
function _ddetd_nn_cache!(state::TimestepperState, template, n_fields::Int)
    cache = get!(() -> Vector{ScalarField}[],
                 state.timestepper_data, :_ddetd_Nn)::Vector{Vector{ScalarField}}
    if isempty(cache) || length(cache[1]) != n_fields
        for f in template
            isempty(f.bases) || ensure_layout!(f, :c)
        end
        empty!(cache)
        push!(cache, copy_state(template))
    end
    return cache[1]
end

"""Push a copy of `F_n` onto the rolling-2 SBDF2 `Fhist`, reusing the field-set
dropped from `Fhist` on the previous step (stashed under `:_ddi_sbdf2_frec`) in
place of a fresh `copy_state(F_n)`. The dropped set is `F_{n-2}`; SBDF2 reads only
`Fhist[end]=F_{n-1}`, so the recycled set is provably out of the live window and
reuse cannot alias a read F. Falls back to `copy_state` until the ring fills
(steady state: 3 distinct sets rotate — 2 in `Fhist` + 1 stashed)."""
function _ddi_sbdf2_push_fhist!(state::TimestepperState, Fhist, F_n, n_fields::Int)
    rec = get(state.timestepper_data, :_ddi_sbdf2_frec, nothing)
    if rec isa Vector{ScalarField} && length(rec) == n_fields
        state.timestepper_data[:_ddi_sbdf2_frec] = nothing
        @inbounds for i in 1:n_fields
            isempty(F_n[i].bases) && continue
            ensure_layout!(F_n[i], :c)
            _ddirk_copy!(_local_coeff(get_coeff_data(rec[i])), _local_coeff(get_coeff_data(F_n[i])))
            rec[i].current_layout = :c
        end
        push!(Fhist, rec)
    else
        # Guard failed (first steps, or a field-count change): drop any stale stash
        # so it can't linger, and store a fresh copy.
        state.timestepper_data[:_ddi_sbdf2_frec] = nothing
        push!(Fhist, copy_state(F_n))
    end
    while length(Fhist) > 2
        state.timestepper_data[:_ddi_sbdf2_frec] = popfirst!(Fhist)
    end
    return Fhist
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
@inline function _ddirk_copy!(d::AbstractArray, s::AbstractArray)
    @inbounds copyto!(d, s)
    return d
end
