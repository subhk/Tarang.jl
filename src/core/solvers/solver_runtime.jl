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

Point `solver.state` at the newest timestepper history buffer, then re-bind
`problem.variables` — the field handles the user built and still holds — onto that same
storage.

This used to `sync_state_to_problem!`, i.e. COPY the state into the user's handles. Two
problems with that. The steppers recycle their field-sets, so after one step
`solver.state[1] !== T` and the user's `T` was a stale duplicate: anything written through it
(a nudging term, a mid-run reset, a hand-set IC) was **silently discarded** on the next sync,
with no error. And the copy itself was a full state-sized memcpy every step, which the rest of
this stepper works hard to avoid.

Aliasing fixes both: the handle and the live state share one array, so a user write IS the
state, reads are always current, and the per-step copy disappears.
"""
function _sync_solver_from_timestepper!(solver::InitialValueSolver)
    ts_state = solver.timestepper_state
    if ts_state !== nothing && !isempty(ts_state.history)
        solver.state = ts_state.history[end]
    end
    _alias_state_to_problem!(solver.problem, solver.state)
    return solver.state
end

"""Re-bind each problem variable's storage to the corresponding live state field.

A pointer swap per field, not a copy. Walks the variables in the same order as
`sync_state_to_problem!` so the state index lines up. A variable that already IS the state
field (the common case before the first recycle) is left alone."""
function _alias_state_to_problem!(problem::Problem, state::Vector{<:ScalarField})
    idx = 1
    @inline function bind!(comp)
        idx > length(state) && return
        s = state[idx]
        if comp !== s
            set_grid_data!(comp, get_grid_data(s))
            set_coeff_data!(comp, get_coeff_data(s))
            comp.current_layout = s.current_layout
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
