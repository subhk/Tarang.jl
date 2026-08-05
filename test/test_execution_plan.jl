"""
Execution plan: the solver's runtime path facts are resolved ONCE and every
consumer reads that one answer.

WHY THIS FILE EXISTS. Timestepper implementations used to each re-derive "is this
GPU / is this MPI / were subproblems built / is there an implicit L". Two bug
shapes came out of that, and both are in this project's history:

  * A consumer INFERRED a fact from state that a skipped step never produced.
    `_problem_has_implicit_linear_term` read `problem.equation_data`, which only
    global-matrix assembly fills — the exact step a pure-Fourier GPU IVP skips. So
    the guard against a silently-dropped implicit operator was blind precisely in
    the case it existed for, and a heat equation ran inviscid with no error.

  * The same cascade was copy-pasted into sibling schemes and then diverged.
    `step_sbdf2!` gained the distributed diagonal-IMEX branch; CNAB1/2 and
    SBDF1/3/4 never did. SBDF2 therefore solves an MPI pure-Fourier problem with a
    stiff implicit operator while its siblings refuse the identical setup. That is
    not a bug in any one function — it is invisible because the set of paths was
    never written down in one place.

So this file pins two things. First, that the plan RECORDS what construction did
rather than re-deriving it, and that the record stays true for the solver's life.
Second, the per-scheme capability table, so a scheme cannot quietly acquire or
lose a path, and so the multistep gap above is a visible, asserted fact rather
than something you find by diffing six near-identical functions.
"""

using Test
using Tarang
using InteractiveUtils: subtypes

# ---------------------------------------------------------------------------
# Builders — one per configuration whose plan should differ.
# ---------------------------------------------------------------------------

"""Serial CPU, pure Fourier: global matrices assembled, no subproblems (a
diagonal per-mode implicit operator does not need tau rows)."""
function _plan_solver_fourier_1d(stepper = RK222(); N = 16)
    domain = PeriodicDomain(N)
    u = ScalarField(domain, "u"); set!(u, (x,) -> sin(x))
    prob = IVP([u]); add_parameters!(prob, kappa = 0.1)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    return InitialValueSolver(prob, stepper; dt = 0.01)
end

"""Serial CPU, Fourier×Chebyshev: the coupled direction forces per-mode
subproblem assembly, which is the fact `_check_gpu_implicit_compatibility!`
consults to decide it is exempt."""
function _plan_solver_fourier_cheb(stepper = RK222(); Nx = 8, Nz = 16)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = Nx, bounds = (0.0, 2π), dealias = 1.0)
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, 1.0), dealias = 1.0)
    domain = Domain(dist, (xb, zb))
    u = ScalarField(domain, "u"); set!(u, (x, z) -> sin(x) * z * (1 - z))
    prob = IVP([u]); add_parameters!(prob, kappa = 0.1)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    return InitialValueSolver(prob, stepper; dt = 0.01)
end

@testset "execution plan records construction" begin
    solver = _plan_solver_fourier_1d()
    plan = solver.execution_plan

    @test plan isa Tarang.ExecutionPlan
    @test plan.architecture === :cpu
    @test plan.distribution === :serial
    @test plan.spectral_structure === :fourier

    # The load-bearing pair: assembly RAN, so consumers may read the artifacts.
    # A pure-Fourier GPU IVP is the case where this is false, and reading
    # `equation_data` instead of this flag is what made the old guard blind.
    @test plan.assembled_global_matrices
    @test !plan.assembled_subproblems

    # The plan must agree with the field-level predicates that pre-date it. If
    # these ever disagree, two consumers are answering the same question two ways
    # — the condition this whole design exists to remove.
    @test Tarang.plan_is_gpu(plan) == (Tarang._distributed_field_path_reason(solver.state) === :gpu)
    @test Tarang.plan_is_distributed(plan) ==
          (Tarang._distributed_field_path_reason(solver.state) === :mpi_pencil)
    @test plan.assembled_subproblems == (Tarang._timestepper_subproblems(solver) !== nothing)
end

@testset "execution plan sees the coupled direction" begin
    solver = _plan_solver_fourier_cheb()
    plan = solver.execution_plan

    @test plan.spectral_structure === :coupled
    @test plan.assembled_global_matrices
    # A Chebyshev direction means per-mode tau subproblems were built.
    @test plan.assembled_subproblems
    @test Tarang._timestepper_subproblems(solver) !== nothing
end

@testset "execution plan is stable over the solver's life" begin
    solver = _plan_solver_fourier_1d()
    plan = solver.execution_plan

    before = (plan.architecture, plan.distribution, plan.spectral_structure,
              plan.assembled_global_matrices, plan.assembled_subproblems)

    for _ in 1:5
        step!(solver, 0.01)
    end
    solver.dt = 0.005
    step!(solver, 0.005)

    after_plan = solver.execution_plan
    after = (after_plan.architecture, after_plan.distribution, after_plan.spectral_structure,
             after_plan.assembled_global_matrices, after_plan.assembled_subproblems)

    # Stepping and changing dt must not invalidate the plan — that is the premise
    # that makes resolving it once legitimate.
    @test after === before
    @test after_plan === plan
end

@testset "implicit-linear query is memoized, not re-derived" begin
    solver = _plan_solver_fourier_1d()
    plan = solver.execution_plan

    # Unresolved until first asked: resolving it can require building the matrix
    # expression IR, so it must not happen during construction.
    @test plan.implicit_linear[] === nothing

    answer = Tarang._problem_has_implicit_linear_term(solver)
    @test answer isa Bool
    @test plan.implicit_linear[] === answer

    # Second call returns the memo, and the memo agrees with recomputing from
    # scratch — a memo that drifts from its source is worse than no memo.
    @test Tarang._problem_has_implicit_linear_term(solver) === answer
    @test Tarang._compute_problem_has_implicit_linear_term(solver.problem) === answer
end

# ---------------------------------------------------------------------------
# Per-scheme capability table.
# ---------------------------------------------------------------------------

# Every scheme that has a distributed diagonal-IMEX implementation today. This is
# an audit of the actual call sites, not a wish list:
#   step_rk.jl:84        — step_rk_imex!, serving the whole IMEX RK family
#   step_etd.jl:35,152,296
#   step_multistep.jl:381 — SBDF2 ONLY
#   step_diagonal_imex.jl — the DiagonalIMEX_* family's own path
const EXPECTED_DISTRIBUTED_DIAGONAL = Set([
    :RK111, :RK222, :RK443, :RKSMR, :RKGFY, :RK443_IMEX,
    :ETD_RK222, :ETD_CNAB2, :ETD_SBDF2,
    :DiagonalIMEX_RK222, :DiagonalIMEX_RK443, :DiagonalIMEX_SBDF2,
    :SBDF2,
])

@testset "distributed diagonal-IMEX capability table" begin
    all_schemes = subtypes(Tarang.TimeStepper)
    @test !isempty(all_schemes)

    declared = Set(nameof(T) for T in all_schemes
                   if Tarang.supports_distributed_diagonal_imex(T()))

    # A new scheme defaults to `false`, so it can only enter this set by an
    # explicit declaration — and it cannot leave silently either.
    @test declared == EXPECTED_DISTRIBUTED_DIAGONAL

    # The asymmetry, asserted rather than merely described. SBDF2 has the path;
    # its five multistep siblings do not, so on MPI pure-Fourier with a stiff
    # implicit operator SBDF2 solves and they refuse. Making them work is a
    # numerical change, not a refactor — when someone implements it, this block
    # is what they come here and update deliberately.
    @test Tarang.supports_distributed_diagonal_imex(SBDF2())
    for stepper in (CNAB1(), CNAB2(), SBDF1(), SBDF3(), SBDF4())
        @test !Tarang.supports_distributed_diagonal_imex(stepper)
    end

    # Global-matrix-only schemes likewise: they refuse via
    # `_check_mpi_implicit_compat!` rather than degrading.
    @test !Tarang.supports_distributed_diagonal_imex(Tarang.MCNAB2())
    @test !Tarang.supports_distributed_diagonal_imex(Tarang.CNLF2())
end

@testset "solver plan fields name a contract, not Any" begin
    # `rhs_plan` was `::Any`. Its siblings on the same struct — `evaluator`,
    # `timestepper_state` — have always been declared through an abstract
    # contract from module_contracts.jl; `rhs_plan` was the odd one out only
    # because `LazyRHSPlan` loads long after `InitialValueSolver`.
    #
    # This does NOT make the field concrete, and it is not a performance fix:
    # the JET report count is unchanged, and every consumer still narrows to
    # `::LazyRHSPlan` to reach concrete fields. What it buys is that assigning an
    # unrelated object now fails at the assignment instead of surviving until
    # some later read asserts a type it never had.
    @test fieldtype(InitialValueSolver, :rhs_plan) == Union{Nothing, Tarang.AbstractRHSPlan}
    @test fieldtype(InitialValueSolver, :rhs_plan) != Any
    @test Tarang.LazyRHSPlan <: Tarang.AbstractRHSPlan

    # `execution_plan` is the stricter case: a concrete type, resolved once.
    @test fieldtype(InitialValueSolver, :execution_plan) == Tarang.ExecutionPlan

    solver = _plan_solver_fourier_1d()
    @test solver.rhs_plan === nothing || solver.rhs_plan isa Tarang.AbstractRHSPlan

    # `nothing` stays assignable — compilation may be skipped or declined.
    saved = solver.rhs_plan
    solver.rhs_plan = nothing
    @test solver.rhs_plan === nothing
    solver.rhs_plan = saved

    @test_throws MethodError solver.rhs_plan = "not a plan"
end

@testset "capability gates the path, and every scheme is dispatchable" begin
    # Serial solver: the distributed path must decline for every scheme, capable
    # or not, because distribution — not capability — is the first gate.
    for stepper in (RK222(), SBDF2(), SBDF3())
        solver = _plan_solver_fourier_1d(stepper)
        @test !Tarang._distributed_diagonal_imex_applicable(solver)
    end

    # `_dispatch_step!` has a catch-all that throws "no stepping method defined".
    # Every concrete scheme must have its own method, so that fallback stays
    # unreachable for shipped types.
    for T in subtypes(Tarang.TimeStepper)
        @test hasmethod(Tarang._dispatch_step!, Tuple{T, Any, Any})
        ms = methods(Tarang._dispatch_step!, Tuple{T, Any, Any})
        # The matched method must be specific to this type, not the
        # `::TimeStepper` catch-all.
        @test only(ms).sig.parameters[2] !== Tarang.TimeStepper
    end
end
