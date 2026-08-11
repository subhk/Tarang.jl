"""
The batched mass solve.

`M_min` is rank-deficient in every tau/BC formulation, so the per-mode path
solves `M x = b` with a sparse least-squares (`SPQRSolver`). Measured, though,
`M_min` is a 0/1 PARTIAL PERMUTATION — at most one nonzero per row and per
column, every value exactly 1, the tau columns empty. For such a matrix the
minimum-norm least-squares solution is `x = M⁺b` (the pseudo-inverse — for a
scaled partial permutation, the reciprocal-scaled transpose) for ANY `b`,
which is one kernel rather than `nmodes` sparse solves.

This file pins the structural check that decides whether that shortcut is
legal, because applying it to a genuine mass matrix would be silently wrong
rather than an error.
"""

using Test
using Tarang
using SparseArrays
using LinearAlgebra

# Local copy of the channel-problem builder (matches
# `test_mode_batch_signature.jl`'s `_channel_solver`, renamed so the two don't
# collide when the suite includes both files into the same namespace). Not
# `include`d from that file: the test-inventory guard treats every
# `test_*.jl` as an independent entry point.
function _mass_channel_solver(; nx=16, nz=8, dt=1e-3)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
    xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π), dealias=3 / 2)
    zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xbasis, zbasis))
    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
    tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    tau_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * tau_lift(tau1)
    problem = IVP([b, tau1, tau2])
    add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
    add_equation!(problem,
                  "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 1")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt)
    step!(solver)
    return solver
end

@testset "mass_selection_plan" begin
    @testset "accepts a 0/1 partial permutation" begin
        # cols 1,2 empty; col 3 -> row 1, col 4 -> row 4, col 5 -> row 5
        M = sparse([1, 4, 5], [3, 4, 5], ComplexF64[1, 1, 1], 5, 5)
        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing
        src, scale = plan
        @test src == [0, 0, 1, 4, 5]
        @test scale == ComplexF64[1, 1, 1, 1, 1]
    end

    @testset "accepts a SCALED partial permutation" begin
        M = sparse([1, 4], [3, 4], ComplexF64[2.0, 0.5], 4, 4)
        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing
        src, scale = plan
        @test src == [0, 0, 1, 4]
        @test scale[3] == 2.0
        @test scale[4] == 0.5
    end

    @testset "rejects two nonzeros in one column" begin
        M = sparse([1, 2], [1, 1], ComplexF64[1, 1], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects two nonzeros in one row" begin
        M = sparse([1, 1], [1, 2], ComplexF64[1, 1], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects a genuine mass matrix" begin
        # tridiagonal — the shape a non-identity basis normalisation produces
        M = spdiagm(-1 => ComplexF64[1, 1], 0 => ComplexF64[4, 4, 4],
                    1 => ComplexF64[1, 1])
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "an explicitly stored zero is not a nonzero" begin
        # A stored zero must not be treated as a mapping, or the plan would
        # divide by it.
        M = SparseMatrixCSC(3, 3, [1, 2, 2, 2], [1], ComplexF64[0.0])
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects a stored NaN" begin
        M = sparse([1], [1], ComplexF64[NaN], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects a stored Inf" begin
        # The most dangerous case: `b[src[j]] / Inf` divides down to a
        # plausible 0 with no error raised — the exact silent-wrong-answer
        # failure mode this function exists to prevent.
        M = sparse([1], [1], ComplexF64[Inf], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects NaN hiding in the imaginary part only" begin
        # isfinite on a Complex checks both parts; pin that a NaN in only the
        # imaginary part is not missed by a check that only looks at the real
        # part (the real part alone here looks perfectly finite).
        M = sparse([1], [1], [ComplexF64(1.0, NaN)], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end
end

@testset "batched_mass_apply! equals the per-mode least-squares solve" begin
    using Tarang: MatSolvers

    @testset "against SPQR on the real M_min" begin
        solver = _mass_channel_solver()
        sps = collect(solver.problem.compiled.subproblems)
        live = [sp for sp in sps if sp.M_min !== nothing]
        M = live[1].M_min
        n = size(M, 1)
        nmodes = length(live)

        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing          # the premise of this whole task
        src, scale = plan

        B = ComplexF64[(0.37i + 0.11j) + (0.5i - 0.2j) * im
                       for i in 1:n, j in 1:nmodes]
        X = zeros(ComplexF64, n, nmodes)
        Tarang.batched_mass_apply!(X, B, src, scale)

        # The reference: exactly what the per-mode path does today.
        ref_solver = MatSolvers.solver_instance(MatSolvers.SPQRSolver, M)
        for m in 1:nmodes
            expected = zeros(ComplexF64, n)
            MatSolvers.solve!(expected, ref_solver, B[:, m])
            @test isapprox(X[:, m], expected; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "a scaled permutation divides, not just permutes" begin
        # Pins that `scale` is actually applied: with all-ones data a kernel
        # that ignored `scale` would agree by accident.
        M = sparse([1, 3], [2, 3], ComplexF64[4.0, 0.25], 3, 3)
        src, scale = Tarang.mass_selection_plan(M)
        B = reshape(ComplexF64[8, 12, 16], 3, 1)
        X = zeros(ComplexF64, 3, 1)
        Tarang.batched_mass_apply!(X, B, src, scale)
        @test X[1, 1] == 0            # null column
        @test X[2, 1] == 8 / 4.0      # draws row 1, divided by 4
        @test X[3, 1] == 16 / 0.25    # draws row 3, divided by 0.25
    end

    @testset "null columns are written, not left stale" begin
        M = sparse([2], [2], ComplexF64[1.0], 2, 2)
        src, scale = Tarang.mass_selection_plan(M)
        X = fill(ComplexF64(9999), 2, 1)     # reused buffer
        B = reshape(ComplexF64[5, 7], 2, 1)
        Tarang.batched_mass_apply!(X, B, src, scale)
        @test X[1, 1] == 0                   # must be cleared, not stale
        @test X[2, 1] == 7
    end
end
