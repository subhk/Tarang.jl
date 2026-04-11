"""
Integration test: Subproblem RK — 2D Rayleigh-Benard Convection smoke test.

Verifies the full subproblem architecture (Tasks 1-9) works end-to-end:
  1. Equation parsing → equation_data with M and L expressions
  2. Subsystem/subproblem building with M_min/L_min via expression_matrices
  3. IMEX RK dispatch to step_subproblem_rk!
  4. Per-subproblem sparse LU solves with caching

Uses a small-resolution RBC problem (Nx=16, Nz=8, Ra=1e4) and runs
a few timesteps to confirm no NaN or singular systems.
"""

using Test
using Tarang
using SparseArrays
using Printf

@testset "Subproblem RK — RBC 2D smoke test" begin
    Lx, Lz = 4.0, 1.0
    Nx, Nz = 16, 8
    Ra, Pr = 1e4, 1.0
    kappa = (Ra * Pr)^(-1/2)
    nu = (Ra / Pr)^(-1/2)

    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx))
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
    domain = Domain(dist, (xbasis, zbasis))

    p = ScalarField(domain, "p")
    b = ScalarField(domain, "b")
    u = VectorField(domain, "u")
    tau_p = ScalarField(dist, "tau_p", (), Float64)
    tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), Float64)
    tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), Float64)
    tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
    tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

    ex, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_u = grad(u) + ez * τ_lift(tau_u1)
    grad_b = grad(b) + ez * τ_lift(tau_b1)

    problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])
    add_parameters!(problem, kappa=kappa, nu=nu, Lz=Lz, ez=ez,
        grad_u=grad_u, grad_b=grad_b, τ_lift=τ_lift)

    add_equation!(problem, "trace(grad_u) + tau_p = 0")
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")
    add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - b*ez + τ_lift(tau_u2) = -u⋅∇(u)")
    add_bc!(problem, "b(z=0) = Lz")
    add_bc!(problem, "u(z=0) = 0")
    add_bc!(problem, "b(z=Lz) = 0")
    add_bc!(problem, "u(z=Lz) = 0")
    add_bc!(problem, "integ(p) = 0")

    solver = InitialValueSolver(problem, RK222(); dt=0.01)

    # ── Check subproblems were built ──────────────────────────────────────
    @test haskey(solver.problem.parameters, "subproblems")
    subproblems = solver.problem.parameters["subproblems"]
    @test subproblems isa Tuple
    @test length(subproblems) > 0

    # ── Check subproblems have non-trivial matrices ───────────────────────
    n_with_matrices = 0
    for sp in subproblems
        if sp.M_min !== nothing && sp.L_min !== nothing
            @test nnz(sp.M_min) > 0 || nnz(sp.L_min) > 0
            n_with_matrices += 1
        end
    end
    @test n_with_matrices > 0

    # ── Set initial conditions ────────────────────────────────────────────
    x, z = local_grids(dist, xbasis, zbasis)
    fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)
    get_grid_data(b) .*= z' .* (Lz .- z')       # Damp noise at walls
    get_grid_data(b) .+= Lz .- z'               # Add linear background
    ensure_layout!(b, :c)                        # Pre-compute coefficients

    # ── Run a few timesteps ───────────────────────────────────────────────
    run!(solver; stop_time=0.05, log_interval=1000)

    @test solver.sim_time > 0.0
    @test solver.iteration > 0

    # ── Check no NaN in fields ────────────────────────────────────────────
    ensure_layout!(b, :g)
    @test all(isfinite, get_grid_data(b))
    for comp in u.components
        ensure_layout!(comp, :g)
        @test all(isfinite, get_grid_data(comp))
    end
end
