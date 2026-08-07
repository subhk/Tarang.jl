# ============================================================================
# Execution plan — one resolved answer to "which runtime path does this solver
# use?", recorded at construction instead of re-derived at every call site.
#
# Historically each `step_*!` implementation re-asked the same questions on its
# own: is this GPU, is it MPI-distributed, were subproblems built, were global
# matrices assembled, does the problem carry an implicit linear operator. Two
# failure modes followed from that, and both are in this project's bug history.
#
# 1. A question answered by INSPECTING state that a skipped step never produced.
#    `_problem_has_implicit_linear_term` read `problem.equation_data`, which is
#    filled by global-matrix assembly — precisely the step a pure-Fourier GPU IVP
#    skips. The guard that existed to catch a silently-dropped implicit operator
#    was therefore blind exactly when it mattered. The fix is not a better
#    inspection: it is to RECORD what construction actually did, so a consumer
#    reads a fact rather than reconstructing an inference.
#
# 2. The same cascade copy-pasted into sibling schemes, then diverging. Only
#    `step_sbdf2!` ever gained the distributed diagonal-IMEX branch; CNAB1/2 and
#    SBDF1/3/4 still refuse the configuration their sibling serves. Nothing
#    detected that, because there was no single place where the set of paths was
#    written down.
#
# So: structural facts are resolved once, eagerly, here; per-scheme capability is
# a declared trait (see `step_selection.jl`) rather than an implicit consequence
# of which branches somebody remembered to paste.
# ============================================================================

"""
    ExecutionPlan

The runtime path facts for one solver, resolved once at construction.

Every field except `implicit_linear` is a record of what construction observed or
did, not a re-derivable inference — that is the point. `assembled_subproblems`
and `assembled_global_matrices` in particular say whether the artifacts exist,
so a consumer never has to guess from an empty container whether assembly was
skipped or merely produced nothing.

# Fields
- `architecture`: `:cpu` or `:gpu`, from the state fields' distributor.
- `distribution`: `:serial` or `:mpi_pencil`.
- `spectral_structure`: `:fourier` when every basis is Fourier — the case whose
  implicit operator is diagonal per mode — `:coupled` when any non-Fourier
  (Chebyshev/Jacobi) direction is present, `:none` for a state with no bases.
- `assembled_global_matrices`: the global `L`/`M` assembly ran.
- `assembled_subproblems`: per-mode coefficient-space subproblems were built.
- `implicit_linear`: memo for "does an evolution equation carry a nonzero `L`?".
  Resolved lazily by [`has_implicit_linear_term`](@ref) rather than eagerly,
  because answering it can require building the matrix expression IR, and doing
  that during construction would move a failure that today surfaces on the first
  step — and only on the paths that ask — into every solver build.
"""
struct ExecutionPlan
    architecture::Symbol
    distribution::Symbol
    spectral_structure::Symbol
    assembled_global_matrices::Bool
    assembled_subproblems::Bool
    implicit_linear::Base.RefValue{Union{Nothing, Bool}}
end

function ExecutionPlan(; architecture::Symbol, distribution::Symbol,
                       spectral_structure::Symbol,
                       assembled_global_matrices::Bool,
                       assembled_subproblems::Bool)
    return ExecutionPlan(architecture, distribution, spectral_structure,
                         assembled_global_matrices, assembled_subproblems,
                         Base.RefValue{Union{Nothing, Bool}}(nothing))
end

"""
    _plan_architecture(state) -> Symbol

`:gpu` when the state is device-resident, `:cpu` otherwise. Reads the fields
rather than the requested `device` keyword so the plan describes where the data
actually lives.
"""
function _plan_architecture(state::Vector{<:ScalarField})
    for field in state
        isempty(field.bases) && continue
        _field_uses_gpu(field) && return :gpu
    end
    return :cpu
end

function _plan_distribution(state::Vector{<:ScalarField})
    isempty(state) && return :serial
    dist = state[1].dist
    return (dist.use_pencil_arrays && dist.size > 1) ? :mpi_pencil : :serial
end

function _plan_spectral_structure(state::Vector{<:ScalarField})
    found_spatial = false
    for field in state
        isempty(field.bases) && continue
        found_spatial = true
        all(is_fourier_axis, field.bases) || return :coupled
    end
    return found_spatial ? :fourier : :none
end

"""
    _resolve_execution_plan(state, problem; assembled_global_matrices) -> ExecutionPlan

Build the plan at the end of solver construction, once every artifact it records
is final. Nothing here may throw: these are observations of already-built objects,
and a solver that constructed successfully must get a plan.
"""
function _resolve_execution_plan(state::Vector{<:ScalarField}, problem;
                                 assembled_global_matrices::Bool)
    return ExecutionPlan(;
        architecture = _plan_architecture(state),
        distribution = _plan_distribution(state),
        spectral_structure = _plan_spectral_structure(state),
        assembled_global_matrices = assembled_global_matrices,
        assembled_subproblems = compiled_subproblems(problem) isa Tuple,
    )
end

# Convenience predicates. These read the plan; they never re-inspect the solver,
# which is what keeps every consumer on the same answer.
plan_is_gpu(plan::ExecutionPlan) = plan.architecture === :gpu
plan_is_distributed(plan::ExecutionPlan) = plan.distribution === :mpi_pencil
plan_is_diagonal_fourier(plan::ExecutionPlan) = plan.spectral_structure === :fourier
