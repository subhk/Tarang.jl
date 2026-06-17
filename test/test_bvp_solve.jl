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

    # Nonlinear BVP: same manufactured oracle, but with a quadratic nonlinearity.
    #     Δu + lift(tau1,-1) + lift(tau2,-2) = u² + g,   u(0)=u(Lz)=0
    # Choose g = -2 - u_exact² so the equation reduces to Δu = -2, giving the
    # SAME analytic solution u_exact(z) = z(Lz - z). Exercises the per-mode Newton
    # (Frechet Jacobian dF = Δ + lift - 2u rebuilt each iteration, per Fourier mode).
    @testset "Nonlinear BVP solves manufactured Poisson (Newton, analytic oracle)" begin
        coords = CartesianCoordinates("x", "z")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=bvp_Nx, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=bvp_Nz, bounds=(0.0, bvp_Lz))
        dom = Domain(dist, (xb, zb))
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (xb,), Float64)
        tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        lb2  = derivative_basis(zb, 2)
        g    = ScalarField(dom, "g"); ensure_layout!(g, :g)
        zg   = Tarang.create_meshgrid(dom; on_device=false)["z"]
        Tarang.get_grid_data(g) .= -2 .- (zg .* (bvp_Lz .- zg)).^2   # g = -2 - u_exact²

        prob = Tarang.NLBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=bvp_Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), g=g)
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = u*u + g")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")

        solver = Tarang.BoundaryValueSolver(prob)
        solver.tolerance = 1e-10
        ensure_layout!(u, :g); Tarang.get_grid_data(u) .= 0.0   # zero initial guess
        Tarang.solve!(solver)
        ensure_layout!(u, :g)

        zc = vec(Array(Tarang.local_grid(zb, dist, 1)))
        gd = Array(Tarang.get_grid_data(u))
        expected = bvp_u_exact.(zc)
        for ix in 1:size(gd, 1)
            @test isapprox(gd[ix, :], expected; rtol=1e-7, atol=1e-9)
        end
    end

    # 1D PURE-CHEBYSHEV BVP (no separable Fourier axis). Regression guard for the
    # gather/scatter bug fixed 2026-06-04: `_gather_field_raw!`/`_scatter_field_raw!`
    # used to treat ANY 1D coefficient array as a single-entry Fourier tau, so a
    # pure-Chebyshev field (1D coeff vector of length Nz) had only its first
    # coefficient gathered — the solve returned a constant. The fix gathers the
    # full coupled spectrum when the 1D field's basis is non-Fourier.
    @testset "1D pure-Chebyshev LBVP solves manufactured Poisson" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        zb = ChebyshevT(coords["z"]; size=bvp_Nz, bounds=(0.0, bvp_Lz))
        dom = Domain(dist, (zb,))
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lb2  = derivative_basis(zb, 2)
        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=bvp_Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")

        solver = Tarang.BoundaryValueSolver(prob)
        Tarang.solve!(solver)
        ensure_layout!(u, :g)

        zc = vec(Array(Tarang.local_grid(zb, dist, 1)))
        g  = vec(Array(Tarang.get_grid_data(u)))
        @test isapprox(g, bvp_u_exact.(zc); rtol=1e-8, atol=1e-9)
    end

    # REGRESSION GUARD: high Fourier modes in steady solvers. The subsystem build
    # used to collapse every nonzero mode onto a single {0,1} representative
    # (compute_matrix_group saw matrix_dependence=false on separable axes), so any
    # forcing at |k|≥2 was silently never solved. Manufactured Poisson with
    # u_exact = sin(πz) cos(2x) (kx=2) pins the per-mode solve.
    @testset "Linear BVP solves a HIGH Fourier mode (kx=2)" begin
        Nx, Nz, Lz = 8, 24, 1.0
        coords = CartesianCoordinates("x", "z"); dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=1.0)
        dom = Domain(dist, (xb, zb))
        u = ScalarField(dom, "u"); tau1 = ScalarField(dist, "tau1", (xb,), Float64); tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        fld = ScalarField(dom, "f"); lb2 = derivative_basis(zb, 2)
        xg = vec(Array(Tarang.local_grid(xb, dist, 1))); zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
        uex(x, z) = sin(π * z / Lz) * cos(2x); λ = (π / Lz)^2 + 4
        ensure_layout!(fld, :g); fd = Tarang.get_grid_data(fld)
        for i in 1:Nx, k in 1:Nz; fd[i, k] = -λ * uex(xg[i], zg[k]); end
        prob = Tarang.LBVP([u, tau1, tau2]); prob.namespace["f"] = fld
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = f")
        Tarang.add_bc!(prob, "u(z=0) = 0"); Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob); Tarang.solve!(solver); ensure_layout!(u, :g)
        g = Array(Tarang.get_grid_data(u))
        for i in 1:Nx, k in 1:Nz
            @test isapprox(g[i, k], uex(xg[i], zg[k]); atol=1e-9)
        end
    end

    # REGRESSION GUARD: 3D mixed basis (2 Fourier + 1 Chebyshev). Exercises
    # (a) multi-Fourier-axis subproblem gather/scatter, (b) per-mode subproblem
    # generation, and (c) the fft-layout (negative) wavenumber on the 2nd Fourier
    # axis. u_exact = sin(πz) cos(x) cos(2y), with kx=1 ≠ ky=2.
    @testset "Linear BVP solves a 3D Fourier×Fourier×Chebyshev Poisson" begin
        Nx, Ny, Nz, Lz = 8, 8, 24, 1.0
        coords = CartesianCoordinates("x", "y", "z"); dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2π), dealias=1.0)
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=1.0)
        dom = Domain(dist, (xb, yb, zb))
        u = ScalarField(dom, "u"); tau1 = ScalarField(dist, "tau1", (xb, yb), Float64); tau2 = ScalarField(dist, "tau2", (xb, yb), Float64)
        fld = ScalarField(dom, "f"); lb2 = derivative_basis(zb, 2)
        xg = vec(Array(Tarang.local_grid(xb, dist, 1))); yg = vec(Array(Tarang.local_grid(yb, dist, 1))); zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
        uex(x, y, z) = sin(π * z / Lz) * cos(x) * cos(2y); λ = (π / Lz)^2 + 1 + 4
        ensure_layout!(fld, :g); fd = Tarang.get_grid_data(fld)
        for i in 1:Nx, j in 1:Ny, k in 1:Nz; fd[i, j, k] = -λ * uex(xg[i], yg[j], zg[k]); end
        prob = Tarang.LBVP([u, tau1, tau2]); prob.namespace["f"] = fld
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = f")
        Tarang.add_bc!(prob, "u(z=0) = 0"); Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob); Tarang.solve!(solver); ensure_layout!(u, :g)
        g = Array(Tarang.get_grid_data(u))
        err = maximum(abs(g[i, j, k] - uex(xg[i], yg[j], zg[k])) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        @test err < 1e-8
    end

    # REGRESSION GUARD: coupled MULTI-FIELD steady solve. Every bulk PDE equation
    # used to target the SAME first coupled variable `udix`, so the forcing slot
    # was overwritten and all bulk equations received the LAST equation's RHS.
    # Two co-solved Poissons at the same x-mode, different amplitudes (v=3u): with
    # the bug u would pick up v's forcing (→ u = 3·u_exact).
    @testset "Linear BVP solves a coupled multi-field system" begin
        Nx, Nz, Lz = 8, 24, 1.0
        coords = CartesianCoordinates("x", "z"); dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=1.0)
        dom = Domain(dist, (xb, zb))
        u = ScalarField(dom, "u"); v = ScalarField(dom, "v")
        tu1 = ScalarField(dist, "tu1", (xb,), Float64); tu2 = ScalarField(dist, "tu2", (xb,), Float64)
        tv1 = ScalarField(dist, "tv1", (xb,), Float64); tv2 = ScalarField(dist, "tv2", (xb,), Float64)
        fu = ScalarField(dom, "fu"); fv = ScalarField(dom, "fv"); lb2 = derivative_basis(zb, 2)
        xg = vec(Array(Tarang.local_grid(xb, dist, 1))); zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
        uex(x, z) = sin(π * z / Lz) * cos(2x); vex(x, z) = 3 * uex(x, z); λ = (π / Lz)^2 + 4
        ensure_layout!(fu, :g); ensure_layout!(fv, :g)
        fud = Tarang.get_grid_data(fu); fvd = Tarang.get_grid_data(fv)
        for i in 1:Nx, k in 1:Nz; fud[i, k] = -λ * uex(xg[i], zg[k]); fvd[i, k] = -λ * vex(xg[i], zg[k]); end
        prob = Tarang.LBVP([u, v, tu1, tu2, tv1, tv2]); prob.namespace["fu"] = fu; prob.namespace["fv"] = fv
        add_parameters!(prob; Lz=Lz, lu1=lift(tu1, lb2, -1), lu2=lift(tu2, lb2, -2),
                        lv1=lift(tv1, lb2, -1), lv2=lift(tv2, lb2, -2))
        Tarang.add_equation!(prob, "Δ(u) + lu1 + lu2 = fu")
        Tarang.add_equation!(prob, "Δ(v) + lv1 + lv2 = fv")
        Tarang.add_bc!(prob, "u(z=0) = 0"); Tarang.add_bc!(prob, "u(z=Lz) = 0")
        Tarang.add_bc!(prob, "v(z=0) = 0"); Tarang.add_bc!(prob, "v(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob); Tarang.solve!(solver)
        ensure_layout!(u, :g); ensure_layout!(v, :g)
        gu = Array(Tarang.get_grid_data(u)); gv = Array(Tarang.get_grid_data(v))
        @test maximum(abs(gu[i, k] - uex(xg[i], zg[k])) for i in 1:Nx, k in 1:Nz) < 1e-8
        @test maximum(abs(gv[i, k] - vex(xg[i], zg[k])) for i in 1:Nx, k in 1:Nz) < 1e-8
    end

    @testset "1D Chebyshev variable-coefficient LBVP (implicit NCC)" begin
        # REGRESSION GUARD: a non-constant FIELD coefficient on the implicit side,
        # `q(z)*u`, used to be SILENTLY DROPPED (MultiplyOperator matrix Case 3 returned
        # the variable's block, ignoring the coefficient). It now builds a pseudospectral
        # multiply-by-q matrix. Manufactured: Δu + q·u = f, u_exact=sin(πz/Lz), q=1+z.
        Nz = 48; Lz = 1.0
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
        dom = Domain(dist, (zb,))
        zc = vec(Array(Tarang.create_meshgrid(dom)["z"]))
        uex(z) = sin(π*z/Lz)
        d2uex(z) = -(π/Lz)^2 * sin(π*z/Lz)

        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lb2  = derivative_basis(zb, 2)
        q = ScalarField(dom, "q"); ensure_layout!(q, :g); Tarang.get_grid_data(q) .= (1.0 .+ zc)
        f = ScalarField(dom, "f"); ensure_layout!(f, :g)
        Tarang.get_grid_data(f) .= d2uex.(zc) .+ (1.0 .+ zc) .* uex.(zc)
        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), q=q, f=f)
        Tarang.add_equation!(prob, "Δ(u) + q*u + l1 + l2 = f")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob); Tarang.solve!(solver)
        ensure_layout!(u, :g)
        @test maximum(abs.(vec(Array(Tarang.get_grid_data(u))) .- uex.(zc))) < 1e-8
    end

    @testset "2D Fourier×Cheb z-dependent-coefficient LBVP (implicit NCC)" begin
        # REGRESSION GUARD: a coefficient varying along the Chebyshev direction (constant
        # along Fourier) in a mixed Fourier-x × Cheb-z domain — the channel-flow case. The
        # multiply-by-q(z) matrix is built per Fourier-mode subproblem. Manufactured:
        # Δu + q(z)·u = f, u_exact = sin(πz/Lz)·cos(2x), q = 1+z (constant in x).
        Nx = 8; Nz = 32; Lz = 1.0
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
        dom = Domain(dist, (xb, zb))
        mesh = Tarang.create_meshgrid(dom); X = Array(mesh["x"]); Z = Array(mesh["z"])
        uex(x, z) = sin(π*z/Lz) * cos(2x)
        lap_uex(x, z) = -(4 + (π/Lz)^2) * uex(x, z)

        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (xb,), Float64)
        tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        lb2  = derivative_basis(zb, 2)
        q = ScalarField(dom, "q"); ensure_layout!(q, :g); Tarang.get_grid_data(q) .= (1.0 .+ Z)
        f = ScalarField(dom, "f"); ensure_layout!(f, :g)
        Tarang.get_grid_data(f) .= lap_uex.(X, Z) .+ (1.0 .+ Z) .* uex.(X, Z)
        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), q=q, f=f)
        Tarang.add_equation!(prob, "Δ(u) + q*u + l1 + l2 = f")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob); Tarang.solve!(solver)
        ensure_layout!(u, :g)
        @test maximum(abs.(Array(Tarang.get_grid_data(u)) .- uex.(X, Z))) < 1e-8
    end
end
