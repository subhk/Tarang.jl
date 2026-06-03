"""
End-to-end boundary-value-problem (BVP) solve tests with a manufactured
(analytic) oracle: pick u_exact, derive forcing + BCs, solve, compare. Expected
values come ONLY from the manufactured solution, never the solver's own output.

The steady BVP solver was rehabilitated 2026-06-03 to solve PER-FOURIER-MODE
(one square tau subproblem per separable mode), mirroring the IVP timestepper and
Dedalus. Fixes: `_solver_type` defined; `add_bc!` BCs merged in the BVP build;
matrix-coupling configured (Fourier separable / Chebyshev coupled) so
build_subsystems creates per-mode subproblems; `solve_linear!` rewritten to
assemble each subproblem RHS (PDE forcing + BC rows) and solve `L_sp x = F_sp`.

Problem (Dedalus second-order tau formulation):
    Δu + lift(tau1,-1) + lift(tau2,-2) = -2   on z in [0, Lz]
    u(z=0) = 0,  u(z=Lz) = 0
    u_exact(z) = z (Lz - z)   (x-independent; peak Lz^2/4)

Uniquely-prefixed names (bvp_*) — the full suite shares the Main namespace.
"""

using Test
using Tarang

const bvp_Lz = 1.0
const bvp_Nx = 4
const bvp_Nz = 16
bvp_u_exact(z) = z * (bvp_Lz - z)

# Build the manufactured Poisson LBVP (square per-mode tau system).
function bvp_build_problem()
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=bvp_Nx, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=bvp_Nz, bounds=(0.0, bvp_Lz))
    dom = Domain(dist, (xb, zb))
    u    = ScalarField(dom, "u")
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    lb2  = derivative_basis(zb, 2)
    prob = Tarang.LBVP([u, tau1, tau2])
    add_parameters!(prob; Lz=bvp_Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
    Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
    Tarang.add_bc!(prob, "u(z=0) = 0")
    Tarang.add_bc!(prob, "u(z=Lz) = 0")
    return prob, u, zb, dist, dom
end

@testset "BVP solver" begin
    @testset "Fixes A/B present: _solver_type defined, BCs merge" begin
        @test isdefined(Tarang, :_solver_type)
        @test Tarang._solver_type(:sparse) === :sparse
        @test Tarang._solver_type((:sparse, (;))) === :sparse

        prob, _, _, _, _ = bvp_build_problem()
        Tarang.setup_domain!(prob)
        Tarang._merge_boundary_conditions!(prob)
        merged_ok = try
            Tarang.validate_problem(prob); true
        catch err
            !occursin("less than number of variables", sprint(showerror, err))
        end
        @test merged_ok
    end

    @testset "per-Fourier-mode build: square tau subproblems" begin
        prob, _, _, _, _ = bvp_build_problem()
        solver = Tarang.BoundaryValueSolver(prob)
        @test !isempty(solver.subproblems)
        # Each subproblem is square (Nz + n_tau) for its Fourier mode.
        for sp in solver.subproblems
            sp.L_min === nothing && continue
            @test size(sp.L_min, 1) == size(sp.L_min, 2)
        end
    end

    @testset "Linear BVP solves manufactured Poisson (analytic oracle)" begin
        prob, u, zb, dist, dom = bvp_build_problem()
        solver = Tarang.BoundaryValueSolver(prob)
        Tarang.solve!(solver)
        ensure_layout!(u, :g)

        zc = vec(Array(Tarang.local_grid(zb, dist, 1)))     # 1D z nodes
        g  = Array(Tarang.get_grid_data(u))                  # (Nx, Nz)
        expected = bvp_u_exact.(zc)
        # x-independent solution: every x-row equals u_exact(z) at the grid nodes.
        # (This is the definitive oracle — the analytic solution sampled on the
        # actual Chebyshev nodes; the grid does not sample z=Lz/2, so a separate
        # peak-value check would be a grid artifact, not a correctness check.)
        for ix in 1:size(g, 1)
            @test isapprox(g[ix, :], expected; rtol=1e-8, atol=1e-9)
        end
    end
end
