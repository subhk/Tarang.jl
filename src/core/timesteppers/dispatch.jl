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

    # Dispatch to the appropriate stepping method based on timestepper type
    _dispatch_step!(state.timestepper, state, solver)

    # Update temporal filters with the new solution
    _update_temporal_filters!(solver, state.dt)

    # Reset forcing flag at the end of timestep (prepare for next forcing generation)
    reset_forcing_flag!(state)
end

# ============================================================================
# Type-stable dispatch methods (one per timestepper type)
# ============================================================================

# --- IMEX Runge-Kutta methods (Dedalus-compatible) ---
_dispatch_step!(::RK111, state, solver) = step_rk_imex!(state, solver)
_dispatch_step!(::RK222, state, solver) = step_rk_imex!(state, solver)
_dispatch_step!(::RK443, state, solver) = step_rk_imex!(state, solver)

# --- Multistep IMEX methods ---
# For Chebyshev-Fourier domains with PencilLinearOperator, use pencil-based methods
function _dispatch_step!(::CNAB1, state, solver)
    if _get_pencil_linear_operator(solver) !== nothing
        step_pencil_cnab1!(state, solver)
    else
        step_cnab1!(state, solver)
    end
end

function _dispatch_step!(::CNAB2, state, solver)
    if _get_pencil_linear_operator(solver) !== nothing
        step_pencil_cnab2!(state, solver)
    else
        step_cnab2!(state, solver)
    end
end

function _dispatch_step!(::SBDF1, state, solver)
    if _get_pencil_linear_operator(solver) !== nothing
        step_pencil_sbdf1!(state, solver)
    else
        step_sbdf1!(state, solver)
    end
end

function _dispatch_step!(::SBDF2, state, solver)
    if _get_pencil_linear_operator(solver) !== nothing
        step_pencil_sbdf2!(state, solver)
    else
        step_sbdf2!(state, solver)
    end
end

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
_dispatch_step!(::RKSMR, state, solver) = step_rksmr!(state, solver)
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

# ============================================================================
# MPI Compatibility Check for Global-Matrix Implicit Solvers
# ============================================================================

"""
    _check_mpi_implicit_compat!(solver, method_name)

Check if a global-matrix implicit timestepper can run with MPI fields.
These methods (SBDF3/4, MCNAB2, CNLF2) use global matrix solves that require
all field data on a single rank.

For MPI runs, the check either:
1. Allows it if fields are small enough for gather/scatter (< 1M DOF)
2. Errors with guidance to use a pencil-compatible method instead
"""
function _check_mpi_implicit_compat!(solver::InitialValueSolver, method_name::String)
    dist = solver.state[1].dist
    if dist.size <= 1
        return  # Serial — no issue
    end

    # Check if a PencilLinearOperator is available (pencil IMEX path)
    if _get_pencil_linear_operator(solver) !== nothing
        @warn "$method_name does not have a pencil-based variant. " *
              "A PencilLinearOperator is configured but will NOT be used. " *
              "Consider using CNAB1, CNAB2, SBDF1, or SBDF2 which have MPI-compatible pencil variants. " *
              "Falling back to global matrix gather/scatter (all data collected to each rank)." maxlog=1
    end

    # Check if we can do gather/scatter (small enough problem)
    total_dof = sum(length(get_coeff_data(f)) for f in solver.state
                    if get_coeff_data(f) !== nothing; init=0)

    if total_dof > 1_000_000
        throw(ArgumentError(
            "$method_name with MPI requires global matrix solve (gather/scatter). " *
            "Total DOF=$total_dof exceeds 1M limit for gather/scatter approach. " *
            "Use a pencil-compatible method instead: CNAB1, CNAB2, SBDF1, SBDF2 " *
            "(with `set_pencil_linear_operator!`), or use DiagonalIMEX_RK222/RK443 " *
            "for purely Fourier domains."))
    end

    @debug "$method_name: using global gather/scatter for MPI with $total_dof DOF"
end
