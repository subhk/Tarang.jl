"""
Build-pass memo for `_implicit_ncc_matrix` (collective-safety under MPI).

The implicit-NCC builder runs MPI-collective operations (PencilArray reductions,
`ensure_layout!` :c↔:g transposes) and is invoked from inside per-LOCAL-subproblem
matrix-build loops; local subproblem counts differ across ranks whenever
#Fourier modes % nprocs != 0, so per-subproblem invocation issues unmatched
collectives — deadlock (2026-08-20 MPI review, finding R1/S1/V1; reachable via
the NLBVP per-mode Jacobian rebuild, which calls `build_matrices!` directly —
the IVP route rejects implicit NCCs earlier, in global matrix assembly).
The memo makes every rank compute each coefficient exactly ONCE per build pass.

Pinned here (serially observable):
  1. Repeat calls within a pass return the identical cached matrix (no rebuild).
  2. Invalidation is what exposes new coefficient DATA — after
     `_invalidate_implicit_ncc_memo!()` the rebuilt matrix reflects the change
     (a stale memo would silently freeze the old coefficient).
  3. `build_subproblem_matrices` starts a fresh pass (clears the memo).
"""

using Test
using Tarang
using SparseArrays

@testset "implicit NCC build-pass memo" begin
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64)
    zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    # A standalone coefficient field; the memo keys on the field object itself.
    q = ScalarField(dist, "q", (zb,), Float64)
    zg = vec(collect(Tarang.local_grid(zb, dist, 1.0)))
    q["g"] = 1.0 .+ 0.5 .* zg

    @testset "repeat calls hit the cache" begin
        Tarang._invalidate_implicit_ncc_memo!()
        m1 = Tarang._implicit_ncc_matrix_memoized(q)
        m2 = Tarang._implicit_ncc_matrix_memoized(q)
        @test m1 isa SparseMatrixCSC
        @test m1 === m2                     # identical object — builder ran once
        @test length(Tarang._IMPLICIT_NCC_MEMO) == 1
    end

    @testset "invalidation exposes changed coefficient data" begin
        m1 = Tarang._implicit_ncc_matrix_memoized(q)
        q["g"] = 2.0 .+ 0.5 .* zg           # q -> q + 1
        @test Tarang._implicit_ncc_matrix_memoized(q) === m1   # memo semantics
        Tarang._invalidate_implicit_ncc_memo!()
        m3 = Tarang._implicit_ncc_matrix_memoized(q)
        @test m3 !== m1
        @test m3 != m1                      # stale memo would keep them equal
    end

    @testset "build_subproblem_matrices starts a fresh pass" begin
        # Any solver works — the invalidation happens before the subproblem loop.
        u = ScalarField(dist, "u", (zb,), Float64)
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lb = derivative_basis(zb, 2)
        problem = IVP([u, tau1, tau2])
        add_parameters!(problem; lb=lb)
        add_equation!(problem, "∂t(u) - ∂z(∂z(u)) + lift(tau1, lb, -1) + lift(tau2, lb, -2) = 0")
        add_bc!(problem, "u(z=0) = 0")
        add_bc!(problem, "u(z=1) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        sps = Tarang._timestepper_subproblems(solver)
        @test sps !== nothing

        Tarang._implicit_ncc_matrix_memoized(q)   # populate
        @test !isempty(Tarang._IMPLICIT_NCC_MEMO)
        Tarang.build_subproblem_matrices(solver, collect(sps), ["L"])
        # The build pass invalidates on entry; no NCC in these equations, so the
        # memo stays empty afterwards.
        @test isempty(Tarang._IMPLICIT_NCC_MEMO)
    end
end
