# ============================================================================
# Timestepper State Management
# ============================================================================

# ---------------------------------------------------------------------------
# History rotation helpers — avoid while-loop + popfirst!/pop! overhead
# ---------------------------------------------------------------------------

"""
    _push_trim!(vec, item, max_len)

Append `item` to `vec` and remove oldest (first) entries to keep length ≤ `max_len`.
Used for state.history where newest is last.
"""
@inline function _push_trim!(vec::Vector, item, max_len::Int)
    push!(vec, item)
    while length(vec) > max_len
        popfirst!(vec)
    end
    return vec
end

"""
    _prepend_trim!(vec, item, max_len)

Prepend `item` to `vec` and remove oldest (last) entries to keep length ≤ `max_len`.
Used for MX/LX/F histories in multistep methods where newest is first.
"""
@inline function _prepend_trim!(vec::Vector, item, max_len::Int)
    pushfirst!(vec, item)
    while length(vec) > max_len
        pop!(vec)
    end
    return vec
end

"""
Timestepper state management with workspace optimization.

Holds the current state, history, and workspace fields for time-stepping.
"""
@inline _empty_workspace_fields(initial_state::Vector{F}) where {F<:ScalarField{<:Any, SerialFieldStorage}} =
    similar(initial_state, 0)

@inline _empty_workspace_fields(::Vector{<:ScalarField}) = ScalarField[]

mutable struct TimestepperState{TS<:TimeStepper, V<:Vector{<:ScalarField}, W<:Vector{<:ScalarField}} <: AbstractTimestepperState
    timestepper::TS
    dt::Float64
    history::Vector{V}
    dt_history::Vector{Float64}  # Track timestep history for variable timesteps
    stage::Int
    timestepper_data::Dict{Symbol, Any}  # Additional data for specific timesteppers

    # Pre-allocated workspace fields for zero-allocation time-stepping
    workspace_fields::W  # Reusable scratch fields
    workspace_allocated::Bool

    # Stochastic forcing support (following GeophysicalFlows.jl pattern)
    # Forcing is computed ONCE at the beginning of each timestep and stays constant
    # across all substeps (important for Stratonovich calculus correctness)
    forcing::Union{Nothing, Forcing}  # StochasticForcing or DeterministicForcing
    current_substep::Int  # Track which substep we're in (1-indexed)
    forcing_generated::Bool  # Flag to track if forcing was generated this timestep

    function TimestepperState(timestepper::TS, dt::Float64, initial_state::V;
                              forcing=nothing) where {TS<:TimeStepper, V<:Vector{<:ScalarField}}
        dt_history = [dt]
        timestepper_data = Dict{Symbol, Any}()

        # Pre-allocate workspace fields FIRST — workspace allocation can
        # invalidate shared buffers from the distributor's layout cache,
        # so it must happen before copying the initial state.
        n_fields = length(initial_state)
        n_workspace_sets = _workspace_count(timestepper)
        workspace_fields = _empty_workspace_fields(initial_state)

        for _ in 1:n_workspace_sets
            for field in initial_state
                ws_field = ScalarField(field.dist, "workspace", field.bases, field.dtype)
                push!(workspace_fields, ws_field)
            end
        end

        # Copy AFTER workspace allocation to avoid buffer aliasing
        initial_history = copy_state(initial_state)
        history = Vector{typeof(initial_history)}(undef, 1)
        history[1] = initial_history

        new{TS, typeof(initial_history), typeof(workspace_fields)}(
            timestepper, dt, history, dt_history, 0, timestepper_data,
            workspace_fields, true, forcing, 1, false)
    end
end

"""
    _timestep_vector_buffer!(state, key, n)

Return a reusable `Vector{ComplexF64}` scratch buffer of length `n` from
`state.timestepper_data`, replacing stale or incorrectly-sized entries.
"""
function _timestep_vector_buffer!(state::TimestepperState, key::Symbol, n::Integer)
    cache = state.timestepper_data
    cached = get(cache, key, nothing)
    if cached isa Vector{ComplexF64} && length(cached) == n
        return cached
    end

    buffer = Vector{ComplexF64}(undef, n)
    cache[key] = buffer
    return buffer
end

"""
    _timestep_stage_vectors!(state, key, stages, n)

Return `stages` reusable `Vector{ComplexF64}` buffers, each of length `n`.
Used for RK stage data that must remain valid until the final update.
"""
function _timestep_stage_vectors!(state::TimestepperState, key::Symbol,
                                  stages::Integer, n::Integer)
    cache = state.timestepper_data
    cached = get(cache, key, nothing)
    if cached isa Vector{Vector{ComplexF64}} && length(cached) == stages
        reusable = true
        for buffer in cached
            if length(buffer) != n
                reusable = false
                break
            end
        end
        reusable && return cached
    end

    buffers = [Vector{ComplexF64}(undef, n) for _ in 1:stages]
    cache[key] = buffers
    return buffers
end

"""
    set_forcing!(state::TimestepperState, forcing)

Set the stochastic forcing configuration for the timestepper.
The forcing will be generated once per timestep and held constant across substeps.
"""
function set_forcing!(state::TimestepperState, forcing)
    state.forcing = forcing
    state.forcing_generated = false
end

"""
    update_forcing!(state::TimestepperState, sim_time::Float64)

Generate new forcing realization at the beginning of a timestep.
This should be called ONCE at the start of step!, not at each substep.

Following GeophysicalFlows.jl pattern:
- Forcing is white in time but spatially correlated
- For Stratonovich calculus, forcing must be constant within each timestep
"""
function update_forcing!(state::TimestepperState, sim_time::Float64)
    if state.forcing !== nothing && !state.forcing_generated
        # Update dt in forcing if it changed
        if hasfield(typeof(state.forcing), :dt) && state.forcing.dt != state.dt
            set_dt!(state.forcing, state.dt)
        end
        # Generate new forcing realization
        generate_forcing!(state.forcing, sim_time)
        state.forcing_generated = true
    end
end

"""
    reset_forcing_flag!(state::TimestepperState)

Reset the forcing generation flag. Call this at the END of each timestep
to prepare for the next forcing generation.
"""
function reset_forcing_flag!(state::TimestepperState)
    state.forcing_generated = false
    state.current_substep = 1
end

# ============================================================================
# Mass-matrix and linear-operator helpers
#
# Sign convention for L_matrix throughout Tarang.jl
# -------------------------------------------------
# The problem is formulated in LHS form (LHS convention):
#
#     M * dX/dt + L * X = F(X, t)          ... (LHS form)
#
# where M is the mass matrix, L is the linear (stiffness) operator, and F is
# the explicit/nonlinear right-hand side returned by evaluate_rhs().
#
# IMEX multistep methods (CNAB, SBDF) and IMEX-RK solve the LHS form directly:
#   - They build  (a*M + b*L) * X_new = RHS  with L appearing with a POSITIVE
#     sign, matching the LHS convention.
#
# ETD (exponential time differencing) methods need the ODE in RHS form:
#
#     dX/dt = -M^{-1} * L * X + M^{-1} * F  = L_rhs * X + N(X)
#
# where L_rhs = -M^{-1} * L.  The function _get_linear_operator_eff!() returns
# this NEGATED, mass-inverse-applied operator for use by ETD methods.
#
# Summary:
#   _get_linear_operator_eff!  -->  returns  -M^{-1}*L   (RHS form, for ETD)
#   _get_linear_operator_lhs!  -->  returns  +M^{-1}*L   (LHS form, for IMEX-RK)
#   CNAB/SBDF use L_matrix directly (no mass inversion, they build M+L system)
# ============================================================================

"""
Return a cached factorization of the mass matrix, or `nothing` if M is
singular (DAE system with non-evolution equations that have zero M rows).
All callers and `_apply_mass_inverse` already treat `nothing` as identity.
"""
function _get_mass_factor!(state::TimestepperState, M_matrix::AbstractMatrix)
    cache = state.timestepper_data
    if !haskey(cache, :M_factor) || get(cache, :M_factor_source, nothing) !== M_matrix
        if _is_singular_mass(M_matrix)
            cache[:M_factor] = nothing
        else
            cache[:M_factor] = factorize(M_matrix)
        end
        cache[:M_factor_source] = M_matrix
        cache[:L_eff] = nothing
        cache[:L_eff_source] = nothing
    end
    return cache[:M_factor]
end

"""
Quick singularity check for mass matrices.  M is singular when any row
is entirely zero (non-evolution equation with no ∂t term).
"""
function _is_singular_mass(M::AbstractMatrix)
    n = size(M, 1)
    m = size(M, 2)
    for i in 1:n
        row_max = zero(real(eltype(M)))
        for j in 1:m
            row_max = max(row_max, abs(M[i, j]))
            row_max > 1e-14 && break  # early exit: row is non-zero
        end
        if row_max <= 1e-14
            return true
        end
    end
    return false
end

"""
    _get_linear_operator_eff!(state, L_matrix, M_matrix) -> (L_rhs, M_factor)

Compute the **RHS-form** linear operator for ETD methods:

    L_rhs = -M^{-1} * L_matrix

This negates L_matrix to convert from the LHS convention (`M*dX/dt + L*X = F`)
to the RHS convention (`dX/dt = L_rhs*X + N(X)`) required by exponential
integrators (ETD-RK, ETD-CNAB, ETD-SBDF).

When `M_matrix === nothing`, the mass matrix is treated as identity and
the result is simply `-L_matrix`.

Returns `(L_rhs, M_factor)` where `M_factor` is the factorized mass matrix
(or `nothing` if no mass matrix).

See also: [`_get_linear_operator_lhs!`](@ref) for the non-negated version.
"""
function _get_linear_operator_eff!(state::TimestepperState, L_matrix::AbstractMatrix,
                                   M_matrix::Union{Nothing, AbstractMatrix})
    if M_matrix === nothing
        # No mass matrix: L_rhs = -L_matrix  (negate for RHS form, cached)
        cache = state.timestepper_data
        if !haskey(cache, :L_neg) || get(cache, :L_neg_source, nothing) !== L_matrix
            cache[:L_neg] = -L_matrix
            cache[:L_neg_source] = L_matrix
        end
        return cache[:L_neg]::AbstractMatrix, nothing
    end

    M_factor = _get_mass_factor!(state, M_matrix)
    if M_factor === nothing
        # Singular mass (DAE system) — treat M as identity
        cache = state.timestepper_data
        if !haskey(cache, :L_neg) || get(cache, :L_neg_source, nothing) !== L_matrix
            cache[:L_neg] = -L_matrix
            cache[:L_neg_source] = L_matrix
        end
        return cache[:L_neg]::AbstractMatrix, nothing
    end

    cache = state.timestepper_data
    if !haskey(cache, :L_eff) || get(cache, :L_eff_source, nothing) !== L_matrix
        cache[:L_eff] = M_factor \ L_matrix   # M^{-1} * L (cached, positive)
        cache[:L_eff_neg] = -cache[:L_eff]    # Negated version (cached)
        cache[:L_eff_source] = L_matrix
    end

    # Negate to convert from LHS form (M*dX/dt + L*X = F)
    # to RHS form (dX/dt = -M^{-1}*L*X + M^{-1}*F)
    return cache[:L_eff_neg]::AbstractMatrix, M_factor
end

"""
    _get_linear_operator_lhs!(state, L_matrix, M_matrix) -> (L_lhs, M_factor)

Compute the **LHS-form** mass-inverse-applied linear operator for IMEX-RK methods:

    L_lhs = +M^{-1} * L_matrix

This preserves the sign of L_matrix (LHS convention: `M*dX/dt + L*X = F`),
applying only the mass-matrix inverse.  IMEX-RK methods can then form
`(I + dt*a*L_lhs)` for the implicit solve, consistent with the standard
IMEX-RK formulation where L appears with a positive sign on the LHS.

When `M_matrix === nothing`, the mass matrix is treated as identity and
the result is simply `+L_matrix`.

Returns `(L_lhs, M_factor)` where `M_factor` is the factorized mass matrix
(or `nothing` if no mass matrix).

See also: [`_get_linear_operator_eff!`](@ref) for the negated (RHS-form) version.
"""
function _get_linear_operator_lhs!(state::TimestepperState, L_matrix::AbstractMatrix,
                                    M_matrix::Union{Nothing, AbstractMatrix})
    if M_matrix === nothing
        # No mass matrix: L_lhs = +L_matrix  (preserve LHS sign)
        return L_matrix, nothing
    end

    M_factor = _get_mass_factor!(state, M_matrix)
    if M_factor === nothing
        # Singular mass (DAE system) — treat M as identity
        return L_matrix, nothing
    end

    cache = state.timestepper_data
    if !haskey(cache, :L_eff) || get(cache, :L_eff_source, nothing) !== L_matrix
        cache[:L_eff] = M_factor \ L_matrix   # M^{-1} * L (cached, positive)
        cache[:L_eff_source] = L_matrix
    end

    # Return positive M^{-1}*L (LHS convention)
    return cache[:L_eff]::AbstractMatrix, M_factor
end

function _apply_mass_inverse!(dest::AbstractVector, ::Nothing, vec::AbstractVector)
    dest === vec || copyto!(dest, vec)
    return dest
end

function _apply_mass_inverse!(dest::AbstractVector, M_factor::F, vec::AbstractVector) where {F}
    ldiv!(dest, M_factor, vec)
    return dest
end

"""
    _get_problem_matrix(problem, key)

Fetch a matrix from `problem.parameters` ensuring it resides on CPU memory.
If the stored matrix is a GPU array, it is copied back to CPU once and the
problem parameter is updated in place so subsequent calls reuse the CPU copy.
"""
function _get_problem_matrix(problem::Problem, key::AbstractString)::Union{Nothing, AbstractMatrix}
    params = problem.parameters
    key_str = key isa String ? key : String(key)
    if !haskey(params, key_str)
        return nothing
    end
    matrix = params[key_str]
    result = _ensure_cpu_matrix!(params, key_str, matrix)
    return result isa AbstractMatrix ? result : nothing
end

function _ensure_cpu_matrix!(params::Dict{String, Any}, key::String, matrix)
    if matrix isa AbstractArray && is_gpu_array(matrix)
        cpu_matrix = Array(matrix)
        params[key] = cpu_matrix
        return cpu_matrix
    end
    return matrix
end

"""
    get_cached_forcing(state::TimestepperState)

Get the cached forcing array. Returns nothing if no forcing is configured.
"""
function get_cached_forcing(state::TimestepperState)
    if state.forcing !== nothing
        return state.forcing.cached_forcing
    end
    return nothing
end

"""
    _update_registered_forcings!(solver::InitialValueSolver, sim_time::Float64, dt::Float64)

Generate new forcing realizations for all forcings registered via `add_stochastic_forcing!`.
Called ONCE at the beginning of each timestep to ensure Stratonovich calculus correctness.
"""
function _update_registered_forcings!(solver::InitialValueSolver, sim_time::Float64, dt::Float64)
    problem = solver.problem

    # Check if problem has stochastic_forcings field (only IVP does)
    if !hasfield(typeof(problem), :stochastic_forcings)
        return
    end

    # Generate forcing for each registered forcing
    for (var_idx, forcing) in problem.stochastic_forcings
        # Update dt if it changed
        if hasfield(typeof(forcing), :dt) && forcing.dt != dt
            set_dt!(forcing, dt)
        end

        # Generate new forcing realization
        # Substep=1 ensures forcing is actually regenerated (not just cached)
        generate_forcing!(forcing, sim_time, 1)
    end
end

"""
    _update_temporal_filters!(solver::InitialValueSolver, dt::Float64)

Update all temporal filters registered via `add_temporal_filter!`.
Called at the END of each timestep after the solution has been advanced.
"""
function _update_temporal_filters!(solver::InitialValueSolver, dt::Float64)
    problem = solver.problem

    # Check if problem has temporal_filters field (only IVP does)
    if !hasfield(typeof(problem), :temporal_filters)
        return
    end

    # Skip if no filters registered
    if isempty(problem.temporal_filters)
        return
    end

    # Get variable name to field mapping
    var_map = Dict{String, Any}()
    for var in problem.variables
        if hasproperty(var, :name)
            var_map[getfield(var, :name)] = var
        end
    end

    # Update each registered filter
    for (filter_name, filter_info) in problem.temporal_filters
        filter = filter_info.filter
        source_sym = filter_info.source
        source_name = String(source_sym)

        if haskey(var_map, source_name)
            source_var = var_map[source_name]
            # Get the physical space data from the variable
            if source_var isa ScalarField
                ensure_layout!(source_var, :g)
                data = get_grid_data(source_var)
            elseif source_var isa VectorField
                # For vector fields, use first component's grid data
                ensure_layout!(source_var.components[1], :g)
                data = get_grid_data(source_var.components[1])
            elseif hasproperty(source_var, :data)
                data = getfield(source_var, :data)
            else
                @warn "Cannot find data for variable $source_name"
                continue
            end

            # Update the filter with current data
            # Use try-catch in case filter types differ
            try
                update!(filter, data, dt)
            catch e
                @warn "Failed to update temporal filter :$filter_name: $e"
            end
        end
    end
end

"""
    _workspace_count(timestepper)

Return the number of workspace field sets needed for a timestepper.
"""
# IMEX RK methods (stages * 3 for X_stages, F_exp, F_imp)
function _workspace_count(::RK111)
    return 3  # 1 stage: X_stages, F_exp, F_imp
end

function _workspace_count(::RK222)
    return 6  # 2 stages: 2 * (X_stages, F_exp, F_imp)
end

function _workspace_count(::RK443)
    return 12  # 4 stages: 4 * (X_stages, F_exp, F_imp)
end

function _workspace_count(::Union{CNAB1, CNAB2})
    return 2
end

function _workspace_count(::Union{SBDF1, SBDF2, SBDF3, SBDF4})
    return 2
end

function _workspace_count(::Union{ETD_RK222, ETD_CNAB2, ETD_SBDF2})
    return 3
end

function _workspace_count(::Union{DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2})
    return 4  # Stage storage for diagonal IMEX
end

function _workspace_count(::TimeStepper)
    return 2  # Default fallback
end

"""
    get_workspace_field!(state::TimestepperState, template::ScalarField, idx::Int)

Get a pre-allocated workspace field, or allocate one if needed.
"""
function get_workspace_field!(state::TimestepperState, template::ScalarField, idx::Int)
    if idx <= length(state.workspace_fields)
        ws = state.workspace_fields[idx]
        # Reset to grid layout
        ws.current_layout = :g
        return ws
    else
        # Fallback: allocate new field (should rarely happen)
        return ScalarField(template.dist, "workspace", template.bases, template.dtype)
    end
end

"""
    copy_field_data!(dest::ScalarField, src::ScalarField)

Copy field data in-place without allocation.

GPU-aware: Uses copyto!() which works on both CPU and GPU arrays.
Both source and destination should be on the same architecture.
"""
function copy_field_data!(dest::ScalarField, src::ScalarField)
    # Skip 0D fields (tau variables) which have no spatial data
    if isempty(src.bases) || isempty(dest.bases)
        return
    end

    # Ensure source has data before trying to copy
    if get_grid_data(src) === nothing && get_coeff_data(src) === nothing
        # Source has no data - nothing to copy
        @debug "copy_field_data!: source $(src.name) has no data, skipping"
        return
    end

    # Ensure destination has data allocated
    if get_grid_data(dest) === nothing && get_coeff_data(dest) === nothing
        # Try to allocate data for destination using same pattern as source
        if dest.domain !== nothing
            allocate_data!(dest)
        end
    end

    # Now do the actual copy
    ensure_layout!(src, :g)
    ensure_layout!(dest, :g)

    src_data = get_grid_data(src)
    dest_data = get_grid_data(dest)

    if src_data === nothing || dest_data === nothing
        @debug "copy_field_data!: cannot copy - src data=$(src_data !== nothing), dest data=$(dest_data !== nothing)"
        return
    end

    # copyto! works on both CPU and GPU arrays
    copyto!(dest_data, src_data)
    dest.current_layout = src.current_layout
end

"""
    get_previous_timestep(state::TimestepperState)

Get the previous timestep from the history (for variable timestep methods).
Returns current dt if history is empty.
"""
function get_previous_timestep(state::TimestepperState)
    if length(state.dt_history) >= 2
        return state.dt_history[end-1]
    end
    return state.dt  # Fallback to current timestep
end

"""
    update_timestep_history!(state::TimestepperState, dt::Float64)

Update the timestep history with the new timestep.
"""
function update_timestep_history!(state::TimestepperState, dt::Float64)
    # Update the current timestep so all step functions see the correct dt
    state.dt = dt

    max_history = get_max_timestep_history(state.timestepper)
    history = state.dt_history
    if length(history) < max_history
        push!(history, dt)
    else
        @inbounds for i in 1:(length(history) - 1)
            history[i] = history[i + 1]
        end
        history[end] = dt
    end
    return nothing
end

"""
    get_max_timestep_history(timestepper)

Return the maximum timestep history length needed for a timestepper.
"""
get_max_timestep_history(::Union{SBDF3, SBDF4}) = 4
get_max_timestep_history(::Union{SBDF2, CNAB2, ETD_CNAB2, ETD_SBDF2, DiagonalIMEX_SBDF2}) = 3
get_max_timestep_history(::TimeStepper) = 2
