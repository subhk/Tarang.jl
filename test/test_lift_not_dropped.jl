# Guard: the equation parser used to SILENTLY DROP a `lift` operator.
#
# `problem_parsing.jl`'s short-form branch called `lift(operand, n)` inside a bare `try` and,
# when basis auto-detection failed, returned the BARE OPERAND with the comment "for matrix
# sizing the Lift is just a shape-preserving wrapper, so the operand alone is sufficient".
#
# That comment is false, measured on the manufactured Poisson LBVP below
# (u(x,z) = sin(x)·z(Lz−z), Δu + lift(τ₁,−1) + lift(τ₂,−2) = f, u(z=0)=u(z=Lz)=0):
#
#   Lift present : max abs err 1.4e-16
#   Lift dropped : max abs err 2.5e-01  on a solution of scale 0.247  (~100% relative)
#
# `subproblem_matrix(::Lift)` places each tau DOF at Chebyshev mode Nz+n+1; a bare operand
# instead goes through `_promote_expression_rows`, which deliberately targets the FIRST row
# block. Dropping the wrapper made the DC subproblem rank-deficient (rank 16 of 18,
# cond 2.5e21) and shrank the others from 18×18 to 16×16 — the tau DOFs left the system.
#
# Auto-detection failure is a diagnosable configuration error and `lift` already raises a
# good ArgumentError naming the 3-argument fix. It must reach the user.
#
# Uniquely-prefixed names (lnd_*) — the full suite shares the Main namespace.

using Test
using Tarang

@testset "a lift operator is never silently dropped" begin

    @testset "parser: undetectable lift basis raises instead of returning the operand" begin
        # A full-shape 1-D Chebyshev field: there is no larger basis set for `lift(u, -1)`
        # to infer a missing non-periodic basis from, so auto-detection cannot succeed.
        lnd_coords = CartesianCoordinates("x")
        lnd_dist = Distributor(lnd_coords; mesh=(1,), dtype=Float64)
        lnd_zb = ChebyshevT(lnd_coords["x"]; size=16, bounds=(-1.0, 1.0))
        lnd_dom = Domain(lnd_dist, (lnd_zb,))
        lnd_u = ScalarField(lnd_dom, "u")
        ns = Dict{String, Any}("u" => lnd_u, "x" => lnd_coords["x"])

        @test_throws ArgumentError parse_expression("lift(u, -1)", ns)
        # The direct constructor already behaved correctly; pin that too.
        @test_throws ArgumentError lift(lnd_u, -1)
    end

    @testset "value oracle: tau-lifted Poisson BVP with x-structure is exact" begin
        Lz, Nx, Nz = 1.0, 8, 16
        u_exact(x, z) = sin(x) * z * (Lz - z)
        f_rhs(x, z) = -sin(x) * (z * (Lz - z) + 2)

        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
        dom = Domain(dist, (xb, zb))
        u = ScalarField(dom, "u")
        f = ScalarField(dom, "f")
        tau1 = ScalarField(dist, "tau1", (xb,), Float64)
        tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        lb2 = derivative_basis(zb, 2)

        xg = vec(Array(Tarang.local_grid(xb, dist, 1)))
        zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
        ensure_layout!(f, :g)
        fd = Tarang.get_grid_data(f)
        for (i, xv) in enumerate(xg), (j, zv) in enumerate(zg)
            fd[i, j] = f_rhs(xv, zv)
        end

        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), f=f)
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = f")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")

        solver = Tarang.BoundaryValueSolver(prob)
        Tarang.solve!(solver)
        ensure_layout!(u, :g)

        got = Array(Tarang.get_grid_data(u))
        expected = [u_exact(xv, zv) for xv in xg, zv in zg]
        # Dropping either Lift lands at ~2.5e-1 here, so this tolerance is the real guard.
        @test maximum(abs.(got .- expected)) < 1e-10
    end
end
