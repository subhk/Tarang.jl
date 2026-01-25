# ============================================================================
# Main Step Function Dispatch
# ============================================================================

"""
    step!(state::TimestepperState, solver::InitialValueSolver)

Advance solution by one timestep.

IMPORTANT for stochastic forcing (following GeophysicalFlows.jl pattern):
- Forcing is generated ONCE at the beginning of the timestep
- Forcing stays CONSTANT across all substeps (RK stages)
- This is essential for correct Stratonovich calculus interpretation
"""
function step!(state::TimestepperState, solver::InitialValueSolver)
    # Generate stochastic forcing ONCE at the beginning of the timestep
    update_forcing!(state, solver.sim_time)

    # Also generate forcing for any forcings registered via add_stochastic_forcing!
    _update_registered_forcings!(solver, solver.sim_time, state.dt)

    state.current_substep = 1  # Reset substep counter

    # IMEX Runge-Kutta methods (Dedalus-compatible)
    if isa(state.timestepper, RK111)
        step_rk_imex!(state, solver)
    elseif isa(state.timestepper, RK222)
        step_rk_imex!(state, solver)
    elseif isa(state.timestepper, RK443)
        step_rk_imex!(state, solver)
    # Multistep IMEX methods
    elseif isa(state.timestepper, CNAB1)
        step_cnab1!(state, solver)
    elseif isa(state.timestepper, CNAB2)
        step_cnab2!(state, solver)
    elseif isa(state.timestepper, SBDF1)
        step_sbdf1!(state, solver)
    elseif isa(state.timestepper, SBDF2)
        step_sbdf2!(state, solver)
    elseif isa(state.timestepper, SBDF3)
        step_sbdf3!(state, solver)
    elseif isa(state.timestepper, SBDF4)
        step_sbdf4!(state, solver)
    elseif isa(state.timestepper, ETD_RK222)
        step_etd_rk222!(state, solver)
    elseif isa(state.timestepper, ETD_CNAB2)
        step_etd_cnab2!(state, solver)
    elseif isa(state.timestepper, ETD_SBDF2)
        step_etd_sbdf2!(state, solver)
    elseif isa(state.timestepper, MCNAB2)
        step_mcnab2!(state, solver)
    elseif isa(state.timestepper, CNLF2)
        step_cnlf2!(state, solver)
    elseif isa(state.timestepper, RKSMR)
        step_rksmr!(state, solver)
    elseif isa(state.timestepper, RKGFY)
        step_rkgfy!(state, solver)
    elseif isa(state.timestepper, RK443_IMEX)
        step_rk443_imex!(state, solver)
    # Diagonal IMEX methods (GPU-native)
    elseif isa(state.timestepper, DiagonalIMEX_RK222)
        step_diagonal_imex_rk222!(state, solver)
    elseif isa(state.timestepper, DiagonalIMEX_RK443)
        step_diagonal_imex_rk443!(state, solver)
    elseif isa(state.timestepper, DiagonalIMEX_SBDF2)
        step_diagonal_imex_sbdf2!(state, solver)
    else
        throw(ArgumentError("Unknown timestepper type: $(typeof(state.timestepper))"))
    end

    # Update temporal filters with the new solution
    _update_temporal_filters!(solver, state.dt)

    # Reset forcing flag at the end of timestep (prepare for next forcing generation)
    reset_forcing_flag!(state)
end
