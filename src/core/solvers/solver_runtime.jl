# -----------------------------------------------------------------------------
# InitialValueSolver runtime orchestration helpers.
#
# These helpers keep `step!(solver)` as a short lifecycle coordinator:
# refresh mutable problem inputs, ensure timestepper state, run the timestepper,
# sync public variables, then advance solver clocks/statistics.
# -----------------------------------------------------------------------------

"""
    _refresh_step_boundary_conditions!(solver, dt)

Refresh time-dependent boundary conditions for the target time of the next
step. Pure space-only boundary conditions are populated at solver-build time
and are intentionally skipped here.
"""
function _refresh_step_boundary_conditions!(solver::InitialValueSolver, dt::Real)
    bcm = solver.problem.bc_manager
    if has_time_dependent_bcs(bcm)
        target_time = solver.sim_time + Float64(dt)
        update_time_dependent_bcs!(bcm, target_time)
        _apply_bc_values_to_equations!(solver, target_time)
        @debug "Refreshed BCs at t=$target_time"
    end
    return nothing
end

"""
    _ensure_timestepper_state!(solver, dt) -> TimestepperState

Create or update the timestepper state used by scheme-specific stepping code.
The solver's `dt` is updated here so all later runtime helpers observe one
authoritative timestep value.
"""
function _ensure_timestepper_state!(solver::InitialValueSolver, dt::Real)
    dt64 = Float64(dt)
    solver.dt = dt64
    _invalidate_deterministic_forcing_memo!(solver)
    if solver.timestepper_state === nothing
        solver.timestepper_state = TimestepperState(solver.timestepper, dt64, solver.state)
    else
        update_timestep_history!(solver.timestepper_state, dt64)
    end
    return solver.timestepper_state
end

"""
    _sync_solver_from_timestepper!(solver)

Point `solver.state` at the newest timestepper history buffer, then bind the
user's `problem.variables` handles to the same storage. Storage includes the
authoritative layout, so changing either handle's grid or coefficient data is
immediately visible through the other handle without a per-step array copy.
"""
function _sync_solver_from_timestepper!(solver::InitialValueSolver)
    ts_state = solver.timestepper_state
    if ts_state !== nothing && !isempty(ts_state.history)
        solver.state = ts_state.history[end]
    end
    _alias_state_to_problem!(solver.problem, solver.state)
    return solver.state
end

"""Bind problem variables to live state or stage storage in flattened field order.

Share the complete storage object, including its layout flag, so transformations
and buffer replacements through either handle cannot leave the other stale.
Rebinding also leaves the previous storage independent for retained history.
"""
function _alias_state_to_problem!(problem::Problem, state::Vector{<:ScalarField})
    idx = 1
    @inline function bind!(comp)
        idx > length(state) && return
        s = state[idx]
        if comp !== s
            comp.storage = s.storage
            comp.layout = s.layout
            comp.scales = s.scales
            comp.transform_bundle = s.transform_bundle
            comp.fft_mode = s.fft_mode
        end
        idx += 1
        return
    end

    for var in problem.variables
        if isa(var, ScalarField)
            bind!(var)
        elseif isa(var, VectorField)
            for comp in var.components
                bind!(comp)
            end
        elseif isa(var, TensorField)
            for comp in vec(var.components)
                bind!(comp)
            end
        end
    end
    return problem
end

"""
    _advance_solver_clock!(solver, dt, step_time)

Advance simulation time, iteration count, and lightweight performance stats
after a successful step.
"""
function _advance_solver_clock!(solver::InitialValueSolver, dt::Real, step_time::Real)
    solver.sim_time += Float64(dt)
    solver.iteration += 1
    solver.performance_stats.total_time += Float64(step_time)
    solver.performance_stats.total_steps += 1
    return solver
end
