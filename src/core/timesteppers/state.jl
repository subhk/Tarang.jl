# ============================================================================
# Timestepper State Management
# ============================================================================

"""
Timestepper state management with workspace optimization.

Holds the current state, history, and workspace fields for time-stepping.
"""
mutable struct TimestepperState
    timestepper::TimeStepper
    dt::Float64
    history::Vector{Vector{ScalarField}}
    dt_history::Vector{Float64}  # Track timestep history for variable timesteps
    stage::Int
    timestepper_data::Dict{String, Any}  # Additional data for specific timesteppers

    # Pre-allocated workspace fields for zero-allocation time-stepping
    workspace_fields::Vector{ScalarField}  # Reusable scratch fields
    workspace_allocated::Bool

    # Stochastic forcing support (following GeophysicalFlows.jl pattern)
    # Forcing is computed ONCE at the beginning of each timestep and stays constant
    # across all substeps (important for Stratonovich calculus correctness)
    forcing::Union{Nothing, Any}  # StochasticForcing or nothing
    current_substep::Int  # Track which substep we're in (1-indexed)
    forcing_generated::Bool  # Flag to track if forcing was generated this timestep

    function TimestepperState(timestepper::TimeStepper, dt::Float64, initial_state::Vector{ScalarField};
                              forcing=nothing)
        history = [copy.(initial_state)]
        dt_history = [dt]  # Initialize with current timestep
        timestepper_data = Dict{String, Any}()

        # Pre-allocate workspace fields based on timestepper requirements
        n_fields = length(initial_state)
        n_workspace_sets = _workspace_count(timestepper)
        workspace_fields = ScalarField[]

        # Pre-allocate workspace fields matching the initial state structure
        for _ in 1:n_workspace_sets
            for field in initial_state
                ws_field = ScalarField(field.dist, "workspace", field.bases, field.dtype)
                push!(workspace_fields, ws_field)
            end
        end

        new(timestepper, dt, history, dt_history, 0, timestepper_data, workspace_fields, true,
            forcing, 1, false)
    end
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
            state.forcing.dt = state.dt
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

# Mass-matrix helpers for exponential integrators
function _get_mass_factor!(state::TimestepperState, M_matrix::AbstractMatrix)
    cache = state.timestepper_data
    if !haskey(cache, "M_factor") || get(cache, "M_factor_source", nothing) !== M_matrix
        cache["M_factor"] = factorize(M_matrix)
        cache["M_factor_source"] = M_matrix
        cache["L_eff"] = nothing
        cache["L_eff_source"] = nothing
    end
    return cache["M_factor"]
end

function _get_linear_operator_eff!(state::TimestepperState, L_matrix::AbstractMatrix,
                                   M_matrix::Union{Nothing, AbstractMatrix})
    if M_matrix === nothing
        return -L_matrix, nothing
    end

    M_factor = _get_mass_factor!(state, M_matrix)
    cache = state.timestepper_data
    if !haskey(cache, "L_eff") || get(cache, "L_eff_source", nothing) !== L_matrix
        cache["L_eff"] = M_factor \ L_matrix
        cache["L_eff_source"] = L_matrix
    end

    return -cache["L_eff"], M_factor
end

function _apply_mass_inverse(M_factor, vec::AbstractVector)
    return M_factor === nothing ? vec : (M_factor \ vec)
end

"""
    _get_problem_matrix(problem, key)

Fetch a matrix from `problem.parameters` ensuring it resides on CPU memory.
If the stored matrix is a GPU array, it is copied back to CPU once and the
problem parameter is updated in place so subsequent calls reuse the CPU copy.
"""
function _get_problem_matrix(problem::Problem, key::AbstractString)
    params = problem.parameters
    key_str = key isa String ? key : String(key)
    if !haskey(params, key_str)
        return nothing
    end
    matrix = params[key_str]
    return _ensure_cpu_matrix!(params, key_str, matrix)
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
            forcing.dt = dt
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
            elseif hasproperty(source_var, :data_g)
                data = getproperty(source_var, :data_g)
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
    ensure_layout!(src, :g)
    ensure_layout!(dest, :g)
    # copyto! works on both CPU and GPU arrays
    copyto!(get_grid_data(dest), get_grid_data(src))
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
    push!(state.dt_history, dt)
    # Keep history bounded
    max_history = get_max_timestep_history(state.timestepper)
    if length(state.dt_history) > max_history
        popfirst!(state.dt_history)
    end
end

"""
    get_max_timestep_history(timestepper)

Return the maximum timestep history length needed for a timestepper.
"""
function get_max_timestep_history(timestepper::TimeStepper)
    if timestepper isa Union{SBDF3, SBDF4}
        return 4
    elseif timestepper isa Union{SBDF2, CNAB2, ETD_CNAB2, ETD_SBDF2, DiagonalIMEX_SBDF2}
        return 3
    else
        return 2
    end
end
