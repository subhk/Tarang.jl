# Serial-CPU coverage tests for src/core/solvers/solver_utils.jl
#
# Targets the helper utilities:
#   - estimate_subproblem_cost(sp)        : LU/nnz/bc cost heuristic
#   - subproblem_locality_report(solver)  : per-rank locality NamedTuple (serial path)
#   - diagnose(solver)                     : tree-style configuration summary
#   - Base.show(io, ::LazyRHSPlan)
#
# All tests run on the serial CPU path (dist.size == 1). MPI Allgather branches
# are intentionally not exercised here.

using Test, Tarang, LinearAlgebra
using SparseArrays: nnz

# -----------------------------------------------------------------------------
# Helpers: build small serial solvers.
# -----------------------------------------------------------------------------

# Pure-Chebyshev diffusion IVP. This is a *coupled* (non-separable) direction,
# so InitialValueSolver builds real per-mode subproblem matrices and stores them
# in problem.parameters["subproblems"]. Its lazy RHS does NOT compile (Laplacian
# on RHS), so solver.rhs_plan is a non-compiled LazyRHSPlan -> exercises the
# "interpreted (lazy translation skipped)" diagnose branch.
function build_cheb_solver(; N=16)
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
    field = ScalarField(dist, "u", (zb,), Float64)
    problem = IVP([field])
    Tarang.add_equation!(problem, "dt(u) = lap(u)")
    solver = InitialValueSolver(problem, RK111(); dt=1e-3, device="cpu")
    return solver
end

# Pure-Fourier trivial IVP. The implicit operator is diagonal per-mode, so NO
# subproblems are built (parameters has no "subproblems" key). Its lazy RHS DOES
# compile -> exercises the "lazy (type-specialized)" diagnose branch.
function build_fourier_solver(; N=8)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    field = ScalarField(dist, "u", (basis,), Float64)
    problem = IVP([field])
    Tarang.add_equation!(problem, "dt(u) = 0")
    solver = InitialValueSolver(problem, RK111(); dt=1e-3, device="cpu")
    return solver
end

# Capture stdout of a zero-arg function (diagnose prints via println).
function capture_stdout(f)
    buf = IOBuffer()
    out = stdout
    rd, wr = redirect_stdout()
    local result
    try
        result = f()
    finally
        redirect_stdout(out)
        close(wr)
    end
    return result, read(rd, String)
end

@testset "solver_utils serial CPU coverage" begin

    # -------------------------------------------------------------------------
    @testset "estimate_subproblem_cost" begin
        solver = build_cheb_solver(; N=16)
        @test haskey(solver.problem.parameters, "subproblems")
        sps = solver.problem.parameters["subproblems"]
        @test sps isa Tuple
        @test length(sps) >= 1
        sp = sps[1]

        # The subproblem has real M/L sparse matrices on the serial Cheb path.
        @test sp.M_min !== nothing
        @test sp.L_min !== nothing

        n = size(sp.M_min, 1)
        nnz_L = sp.L_min === nothing ? 0 : nnz(sp.L_min)
        nnz_M = sp.M_min === nothing ? 0 : nnz(sp.M_min)
        bc = length(sp.bc_rows)

        # Reproduce the documented heuristic exactly:
        #   n^2 / sqrt(n) + (nnz_L + nnz_M) + length(bc_rows)
        expected = Float64(n)^2 / max(sqrt(Float64(n)), 1.0) +
                   Float64(nnz_L + nnz_M) +
                   Float64(bc)
        got = Tarang.estimate_subproblem_cost(sp)
        @test got isa Float64
        @test got ≈ expected
        # Cost must be strictly positive when matrices exist.
        @test got > 0.0

        # M_min === nothing short-circuit returns 0.0 (no matrices => no cost).
        saved_M = sp.M_min
        sp.M_min = nothing
        @test Tarang.estimate_subproblem_cost(sp) == 0.0
        sp.M_min = saved_M  # restore

        # Monotonicity sanity: a larger system costs more than a smaller one.
        solver2 = build_cheb_solver(; N=32)
        sp2 = solver2.problem.parameters["subproblems"][1]
        @test Tarang.estimate_subproblem_cost(sp2) > Tarang.estimate_subproblem_cost(sp)
    end

    # -------------------------------------------------------------------------
    @testset "subproblem_locality_report (serial)" begin
        solver = build_cheb_solver(; N=16)
        sps = solver.problem.parameters["subproblems"]
        rpt = Tarang.subproblem_locality_report(solver)

        # Serial run: everything lives on rank 0 and per_rank has length 1.
        @test rpt.rank == 0
        @test rpt.per_rank == [length(sps)]
        @test rpt.total == length(sps)
        @test rpt.local_count == length(sps)
        @test rpt.with_matrices == count(sp -> sp.M_min !== nothing, sps)
        @test rpt.min_per_rank == length(sps)
        @test rpt.max_per_rank == length(sps)
        @test rpt.mean_per_rank ≈ float(length(sps))

        # A single rank is perfectly balanced by definition.
        @test rpt.imbalance_pct == 0.0
        @test rpt.cost_imbalance_pct == 0.0

        # Cost bookkeeping: local cost equals the sum over subproblems and equals
        # the single per_rank_cost entry.
        cost_sum = sum(Tarang.estimate_subproblem_cost(sp) for sp in sps; init=0.0)
        @test rpt.local_cost ≈ cost_sum
        @test length(rpt.per_rank_cost) == 1
        @test rpt.per_rank_cost[1] ≈ cost_sum
        @test rpt.local_cost > 0.0
    end

    # -------------------------------------------------------------------------
    @testset "subproblem_locality_report with no subproblems" begin
        # Pure-Fourier problem builds no subproblems. The report must degrade
        # gracefully: zero counts, zero cost, no imbalance.
        solver = build_fourier_solver(; N=8)
        @test !haskey(solver.problem.parameters, "subproblems")
        rpt = Tarang.subproblem_locality_report(solver)
        @test rpt.rank == 0
        @test rpt.total == 0
        @test rpt.local_count == 0
        @test rpt.with_matrices == 0
        @test rpt.local_cost == 0.0
        @test rpt.per_rank == [0]
        @test rpt.per_rank_cost == [0.0]
        @test rpt.imbalance_pct == 0.0
        @test rpt.cost_imbalance_pct == 0.0
    end

    # -------------------------------------------------------------------------
    @testset "Base.show(::LazyRHSPlan)" begin
        # Compiled plan (pure Fourier, dt(u)=0).
        fsolver = build_fourier_solver(; N=8)
        @test fsolver.rhs_plan !== nothing
        @test fsolver.rhs_plan.is_compiled
        s_ok = repr(fsolver.rhs_plan)
        @test occursin("LazyRHSPlan", s_ok)
        @test occursin("compiled", s_ok)
        @test occursin("equations", s_ok)

        # Non-compiled plan (Chebyshev Laplacian on RHS fails to lazy-compile).
        csolver = build_cheb_solver(; N=16)
        @test csolver.rhs_plan !== nothing
        @test !csolver.rhs_plan.is_compiled
        s_fail = repr(csolver.rhs_plan)
        @test occursin("LazyRHSPlan", s_fail)
        @test occursin("failed", s_fail)
    end

    # -------------------------------------------------------------------------
    @testset "diagnose: Chebyshev (interpreted RHS + subproblems)" begin
        solver = build_cheb_solver(; N=16)
        ret, out = capture_stdout(() -> Tarang.diagnose(solver))
        # diagnose returns the value of its last println (nothing-ish); we only
        # care about the printed tree.
        @test occursin("Simulation of 1-equation IVP", out)
        @test occursin("timestepper: RK111", out)
        @test occursin("architecture: CPU", out)
        @test occursin("MPI ranks: 1", out)
        @test occursin("ChebyshevT(N=16", out)
        @test occursin("state fields: 1", out)
        @test occursin("Total DOF:", out)
        @test occursin("equations: 1", out)
        # Lazy compile fails for this RHS -> interpreted branch.
        @test occursin("RHS: interpreted (lazy translation skipped)", out)
        @test occursin("boundary conditions: 0", out)
        # Serial subproblem summary branch (dist.size == 1).
        @test occursin("subproblems: 1 (1 with matrices)", out)
        @test occursin("└── stop:", out)
    end

    # -------------------------------------------------------------------------
    @testset "diagnose: Fourier (compiled RHS + performance stats)" begin
        solver = build_fourier_solver(; N=8)
        # Take a step so the performance-stats block (total_steps > 0) prints.
        Tarang.step!(solver, 1e-3)
        @test solver.performance_stats.total_steps >= 1

        ret, out = capture_stdout(() -> Tarang.diagnose(solver))
        @test occursin("RealFourier(N=8", out)
        # Compiled lazy RHS branch.
        @test occursin("RHS: lazy (type-specialized", out)
        # Performance block.
        @test occursin("performance:", out)
        @test occursin("total steps:", out)
        @test occursin("wall time:", out)
        @test occursin("avg step:", out)
        # No subproblems for pure-Fourier -> no "subproblems:" line.
        @test !occursin("subproblems:", out)
    end

end
