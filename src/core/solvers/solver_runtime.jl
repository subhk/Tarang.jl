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
    if solver.timestepper_state === nothing
        solver.timestepper_state = TimestepperState(solver.timestepper, dt64, solver.state)
    else
        update_timestep_history!(solver.timestepper_state, dt64)
    end
    return solver.timestepper_state
end

"""
    _sync_solver_from_timestepper!(solver)

Copy the newest timestepper history state into `solver.state` and then sync it
back to `problem.variables`, which are the fields users usually inspect.
"""
function _sync_solver_from_timestepper!(solver::InitialValueSolver)
    ts_state = solver.timestepper_state
    if ts_state !== nothing && !isempty(ts_state.history)
        solver.state = ts_state.history[end]
    end
    sync_state_to_problem!(solver.problem, solver.state)
    return solver.state
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
