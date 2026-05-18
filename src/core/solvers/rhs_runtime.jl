# -----------------------------------------------------------------------------
# RHS runtime strategy selection.
#
# The lazy RHS machinery is an optimization layer. This file owns the runtime
# decision for whether a solver can use that compiled path or must fall back to
# interpreted expression evaluation in `timesteppers/state_utils.jl`.
# -----------------------------------------------------------------------------

@inline _exec_lazy_rhs!(plan::P, state, solver) where {P} =
    execute_lazy_rhs!(plan, state, solver)

@inline _exec_lazy_rhs_buffered!(plan::P, state, solver) where {P} =
    execute_lazy_rhs_buffered!(plan, state, solver)

function _compiled_lazy_rhs_available(solver::InitialValueSolver)
    plan = solver.rhs_plan
    plan === nothing && return false
    return (plan::LazyRHSPlan).is_compiled
end

"""
    _rhs_evaluation_strategy(solver; buffered=false) -> Symbol

Return the RHS runtime path for this solver.

Possible values:
- `:lazy`: use the compiled lazy RHS plan
- `:lazy_buffered`: use the compiled lazy RHS plan's reusable output buffers
- `:interpreted`: evaluate equation expressions directly
"""
function _rhs_evaluation_strategy(solver::InitialValueSolver; buffered::Bool=false)
    _compiled_lazy_rhs_available(solver) || return :interpreted
    return buffered ? :lazy_buffered : :lazy
end

function _evaluate_rhs_with_strategy(strategy::Symbol,
                                     solver::InitialValueSolver,
                                     state::Vector{<:ScalarField})
    if strategy === :lazy
        return _exec_lazy_rhs!(solver.rhs_plan::LazyRHSPlan, state, solver)
    elseif strategy === :lazy_buffered
        return _exec_lazy_rhs_buffered!(solver.rhs_plan::LazyRHSPlan, state, solver)
    end
    return nothing
end
