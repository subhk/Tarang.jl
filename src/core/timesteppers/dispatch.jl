# ============================================================================
# Timestepper dispatch table.
#
# This file answers one question only:
# "given `state.timestepper`, which implementation should run?"
#
# Keep the heavy math out of this file. Scheme-specific logic belongs in the
# corresponding `step_*.jl` implementation file so contributors can find the
# real runtime path quickly.
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
    _check_stochastic_timestepper_compatibility!(state, solver)

    # Generate stochastic forcing ONCE at the beginning of the timestep
    update_forcing!(state, solver.sim_time)

    # Also generate forcing for any forcings registered via add_stochastic_forcing!
    _update_registered_forcings!(solver, solver.sim_time, state.dt)

    state.current_substep = 1  # Reset substep counter

    # Dispatch to the appropriate stepping method based on timestepper type
    _dispatch_step!(state.timestepper, state, solver)

    # Update temporal filters with the new solution
    _update_temporal_filters!(solver, state.dt)

    # Reset forcing flag at the end of timestep (prepare for next forcing generation)
    reset_forcing_flag!(state)
    return nothing
end

const _RHS_EXTRAPOLATING_STOCHASTIC_TIMESTEPPERS = Union{
    CNAB2, SBDF2, SBDF3, SBDF4, ETD_CNAB2, ETD_SBDF2,
}

function _has_registered_stochastic_forcing(solver::InitialValueSolver)
    problem = solver.problem
    hasfield(typeof(problem), :stochastic_forcings) || return false
    return any(forcing -> forcing isa StochasticForcingType,
               values(problem.stochastic_forcings))
end

function _check_stochastic_timestepper_compatibility!(state::TimestepperState,
                                                       solver::InitialValueSolver)
    timestepper = state.timestepper
    timestepper isa _RHS_EXTRAPOLATING_STOCHASTIC_TIMESTEPPERS || return nothing
    has_stochastic_forcing = state.forcing isa StochasticForcingType ||
                             _has_registered_stochastic_forcing(solver)
    has_stochastic_forcing || return nothing

    scheme = nameof(typeof(timestepper))
    throw(ArgumentError(
        "StochasticForcing is incompatible with $scheme because multistep RHS " *
        "extrapolation colors white noise in time and changes its variance. " *
        "Use a supported one-step method such as RK222, RK443, or ETD_RK222 " *
        "(CNAB1 and SBDF1 are also supported first-order methods)."
    ))
end

# ============================================================================
# One dispatch method per timestepper type
# ============================================================================

# --- IMEX Runge-Kutta methods  ---
_dispatch_step!(::RK111, state, solver) = step_rk_imex!(state, solver)
_dispatch_step!(::RK222, state, solver) = step_rk_imex!(state, solver)
_dispatch_step!(::RK443, state, solver) = step_rk_imex!(state, solver)

# --- Multistep IMEX methods ---
# All multistep methods dispatch through `step_subproblem_multistep!` via
# the subproblem-path branch inside each `step_<method>!` function (see
# step_multistep.jl). The legacy `PencilLinearOperator` + `step_pencil_*!`
# path has been removed — the subproblem path handles Chebyshev-Fourier
# MPI correctly and with DAE-aware BC enforcement.
_dispatch_step!(::CNAB1, state, solver) = step_cnab1!(state, solver)
_dispatch_step!(::CNAB2, state, solver) = step_cnab2!(state, solver)
_dispatch_step!(::SBDF1, state, solver) = step_sbdf1!(state, solver)
_dispatch_step!(::SBDF2, state, solver) = step_sbdf2!(state, solver)

function _dispatch_step!(::SBDF3, state, solver)
    _check_mpi_implicit_compat!(solver, "SBDF3")
    step_sbdf3!(state, solver)
end

function _dispatch_step!(::SBDF4, state, solver)
    _check_mpi_implicit_compat!(solver, "SBDF4")
    step_sbdf4!(state, solver)
end

# --- Exponential Time Differencing (ETD) methods ---
_dispatch_step!(::ETD_RK222, state, solver) = step_etd_rk222!(state, solver)
_dispatch_step!(::ETD_CNAB2, state, solver) = step_etd_cnab2!(state, solver)
_dispatch_step!(::ETD_SBDF2, state, solver) = step_etd_sbdf2!(state, solver)

# --- Additional timesteppers ---
function _dispatch_step!(::MCNAB2, state, solver)
    _check_mpi_implicit_compat!(solver, "MCNAB2")
    step_mcnab2!(state, solver)
end

function _dispatch_step!(::CNLF2, state, solver)
    _check_mpi_implicit_compat!(solver, "CNLF2")
    step_cnlf2!(state, solver)
end
_dispatch_step!(::RKSMR, state, solver) = step_rksmr!(state, solver)   # IMEX ARK → step_rk_imex!
_dispatch_step!(::RKGFY, state, solver) = step_rkgfy!(state, solver)
_dispatch_step!(::RK443_IMEX, state, solver) = step_rk443_imex!(state, solver)

# --- Diagonal IMEX methods (GPU-native) ---
_dispatch_step!(::DiagonalIMEX_RK222, state, solver) = step_diagonal_imex_rk222!(state, solver)
_dispatch_step!(::DiagonalIMEX_RK443, state, solver) = step_diagonal_imex_rk443!(state, solver)
_dispatch_step!(::DiagonalIMEX_SBDF2, state, solver) = step_diagonal_imex_sbdf2!(state, solver)

# --- Fallback for unknown types ---
function _dispatch_step!(ts::TimeStepper, state, solver)
    throw(ArgumentError(
        "No stepping method defined for timestepper type $(typeof(ts)). " *
        "Please add a `_dispatch_step!` method for this type in dispatch.jl."
    ))
end
