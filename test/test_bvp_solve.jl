"""
End-to-end boundary-value-problem (BVP) solve tests with a manufactured
(analytic) oracle: pick u_exact, derive forcing + BCs, solve, compare. Expected
values come ONLY from the manufactured solution, never the solver's own output.

STATUS (2026-06-03):
  The steady BVP/NLBVP solver had three independent bugs. Two are FIXED:
    A. `_solver_type` was undefined -> every BoundaryValueSolver/EigenvalueSolver
       construction threw UndefVarError. FIXED (defined in solver_types.jl).
    B. The BVP build never merged `add_bc!` boundary conditions, so the system
       was under-determined and validation failed. FIXED (_merge_boundary_conditions!
       now called in _build_boundary_value_solver, mirroring the IVP build).
  One remains (documented @test_broken):
    C. The GLOBAL tau-method matrix assembly (`build_matrices`) produces a
       STRUCTURALLY SINGULAR L for a multi-variable tau system: for the Poisson
       problem below it is 54x54 with rank 45 — zero rows (43,46) and zero columns
       (4,49,50,51), i.e. tau/BC degrees of freedom that no equation constrains.
       Construction therefore throws SingularException when the (LU) global solver
       factorizes L, and no correct solution can be produced. The per-subproblem
       assembly used by the IVP timestepper works (test_pencil_imex), so the fix
       is either to repair the global multi-variable tau assembly or to solve the
       BVP per-subproblem (reusing the timestepper machinery). UNFIXED.

Uniquely-prefixed names (bvp_*) — the full suite shares the Main namespace.
"""

using Test
using LinearAlgebra
using Tarang

const bvp_Lz = 1.0
const bvp_Nx = 4
const bvp_Nz = 16
bvp_u_exact(z) = z * (bvp_Lz - z)          # Poisson: Δu=-2, u(0)=u(Lz)=0

# Build the manufactured Poisson LBVP (square tau-method system) and run the
# build steps up to (but not throwing on) the global solve.
function bvp_build_problem()
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=bvp_Nx, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=bvp_Nz, bounds=(0.0, bvp_Lz))
    dom = Domain(dist, (xb, zb))
    u    = ScalarField(dom, "u")
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    τl(A) = lift(A, derivative_basis(zb, 1), -1)
    gu = grad(u) + ez * τl(tau1)
    prob = Tarang.LBVP([u, tau1, tau2])
    add_parameters!(prob; Lz=bvp_Lz, ez=ez, τ_lift=τl, gu=gu)
    Tarang.add_equation!(prob, "div(gu) + τ_lift(tau2) = -2")
    Tarang.add_bc!(prob, "u(z=0) = 0")
    Tarang.add_bc!(prob, "u(z=Lz) = 0")
    return prob, u, dom
end

@testset "BVP solver" begin
    @testset "Bug A FIXED: _solver_type is defined" begin
        @test isdefined(Tarang, :_solver_type)
        # behaviour: passes through a plain choice, unwraps a (type, kwargs) tuple
        @test Tarang._solver_type(:sparse) === :sparse
        @test Tarang._solver_type((:sparse, (;))) === :sparse
    end

    @testset "Bug B FIXED: add_bc! BCs merge into the BVP system" begin
        prob, _, _ = bvp_build_problem()
        Tarang.setup_domain!(prob)
        Tarang._merge_boundary_conditions!(prob)
        # After merging, validation no longer fails with "equations < variables".
        merged_ok = try
            Tarang.validate_problem(prob); true
        catch err
            !occursin("less than number of variables", sprint(showerror, err))
        end
        @test merged_ok
        # The global matrices assemble (square system) once BCs are merged.
        L, M, F = Tarang.build_matrices(prob)
        @test size(L, 1) == size(L, 2)          # square
        @test length(F) == size(L, 1)
    end

    @testset "Bug C: global tau assembly is singular (BVP solve broken)" begin
        prob, u, dom = bvp_build_problem()
        Tarang.setup_domain!(prob)
        Tarang._merge_boundary_conditions!(prob)
        L, _, _ = Tarang.build_matrices(prob)
        Lm = Matrix(L)
        # DOCUMENTED current (buggy) behaviour: L is rank-deficient.
        @test rank(Lm) < size(Lm, 1)            # singular today (rank 45 < 54)

        # CORRECT behaviour (manufactured oracle): the BVP solves to u_exact.
        # Blocked by the singular global L -> construction/solve fails. @test_broken.
        solved_ok = false
        try
            solver = Tarang.BoundaryValueSolver(prob)
            Tarang.solve!(solver)
            zc = vec(Array(Tarang.create_meshgrid(dom; on_device=false)["z"]))
            grid = Array(Tarang.get_grid_data(u))
            # x-independent solution: compare any x-column to u_exact(z)
            col = ndims(grid) == 2 ? grid[1, :] : grid
            solved_ok = isapprox(col, bvp_u_exact.(zc); rtol=1e-6, atol=1e-8)
        catch
            solved_ok = false
        end
        @test_broken solved_ok
    end
end
