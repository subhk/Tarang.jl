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
    _check_gpu_implicit_compatibility!(state, solver)

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

const _UNSAFE_STOCHASTIC_MULTISTEP_TIMESTEPPERS = Union{
    CNAB2, SBDF2, SBDF3, SBDF4, ETD_CNAB2, ETD_SBDF2,
    MCNAB2, DiagonalIMEX_SBDF2, CNLF2,
}

# The only schemes with an on-device per-mode implicit solve (diagonal Fourier
# operator). Every other scheme needs a global-matrix or subproblem solve that a
# pure-Fourier GPU IVP does not build.
const _DIAGONAL_IMEX_TIMESTEPPERS = Union{
    DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2,
}

"""Does the problem carry a nonzero implicit (LHS) linear operator on an evolution
equation? Architecture-independent — reads the parsed L/M expressions, not the
(possibly skipped) assembled global matrix."""
function _problem_has_implicit_linear_term(solver::InitialValueSolver)
    problem = solver.problem
    hasfield(typeof(problem), :equation_data) || return false
    for eq_data in problem.equation_data
        M_expr = get(eq_data, "M", nothing)
        (M_expr === nothing || _is_zero_m_term(M_expr)) && continue   # not an evolution equation
        L_expr = get(eq_data, "L", nothing)
        (L_expr === nothing || is_zero_expression(L_expr)) && continue
        return true
    end
    return false
end

"""
    _check_gpu_implicit_compatibility!(state, solver)

Refuse — loudly — to silently drop an implicit linear operator on a single-GPU IVP
that has no per-mode implicit path.

A pure-Fourier GPU IVP skips global-matrix and subproblem assembly, so
`L_matrix === nothing` no longer means "no implicit term". Every standard IMEX RK /
multistep / ETD scheme then falls through to a fully-explicit step and drops the
implicit `L` with no error — a heat equation runs inviscid. Only the diagonal-IMEX
schemes solve the diagonal Fourier operator per mode on-device. Turn the silent
wrong answer into a clear error naming the working alternative.

Scoped to GPU: the coupled (Fourier×Chebyshev) GPU path DOES build subproblems and
is exempt; the CPU global-matrix path is unaffected.
"""
function _check_gpu_implicit_compatibility!(state::TimestepperState, solver::InitialValueSolver)
    state.timestepper isa _DIAGONAL_IMEX_TIMESTEPPERS && return nothing

    _distributed_field_path_reason(solver.state) === :gpu || return nothing

    # Coupled Fourier×Chebyshev GPU builds subproblems that solve the implicit part
    # per mode — not affected by the pure-Fourier matrix skip.
    if haskey(solver.problem.parameters, "subproblems") &&
       solver.problem.parameters["subproblems"] !== nothing
        return nothing
    end

    _problem_has_implicit_linear_term(solver) || return nothing   # genuinely explicit — fine

    scheme = nameof(typeof(state.timestepper))
    # `error` (ErrorException) mirrors the original loud-refusal guard at
    # step_rk.jl:87 that this reaches before, and the existing GPU test asserts
    # `@test_throws ErrorException`.
    error(
        "$scheme cannot treat an implicit (left-hand-side) linear operator on a single " *
        "GPU: a pure-Fourier GPU solver builds no global matrix or subproblem, so the " *
        "term has no per-mode implicit solve and would be silently dropped — the equation " *
        "would integrate without it. Use a diagonal-IMEX scheme (DiagonalIMEX_RK222, " *
        "DiagonalIMEX_RK443, or DiagonalIMEX_SBDF2), which solves the diagonal Fourier " *
        "operator per mode on-device, or move the linear term to the explicit right-hand " *
        "side (e.g. write `dt(u) = nu*lap(u) + ...` instead of `dt(u) - nu*lap(u) = ...`)."
    )
end

function _has_registered_stochastic_forcing(solver::InitialValueSolver)
    problem = solver.problem
    hasfield(typeof(problem), :stochastic_forcings) || return false
    return any(forcing -> forcing isa StochasticForcingType,
               values(problem.stochastic_forcings))
end

function _check_stochastic_timestepper_compatibility!(state::TimestepperState,
                                                       solver::InitialValueSolver)
    timestepper = state.timestepper
    timestepper isa _UNSAFE_STOCHASTIC_MULTISTEP_TIMESTEPPERS || return nothing
    has_stochastic_forcing = state.forcing isa StochasticForcingType ||
                             _has_registered_stochastic_forcing(solver)
    has_stochastic_forcing || return nothing

    scheme = nameof(typeof(timestepper))
    throw(ArgumentError(
        "StochasticForcing is incompatible with $scheme because multistep reuse or " *
        "combination of stochastic RHS/state across time levels colors white noise " *
        "in time or gives it the wrong variance. " *
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
