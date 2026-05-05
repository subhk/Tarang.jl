"""
Regression tests for the active Chebyshev-Fourier IMEX path.

The old PencilLinearOperator implementation was removed after the unified
subproblem steppers took over the same per-Fourier-mode solve role. Keep this
file as an optional smoke test for that replacement path so stale pencil
symbols do not silently return to the test suite.
"""

using Test
using SparseArrays
using Tarang

function _cheb_fourier_rbc_solver(timestepper; Nx=8, Nz=6, dt=0.01)
    Lx, Lz = 4.0, 1.0
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

    solver = InitialValueSolver(problem, timestepper; dt=dt)

    _, z = local_grids(dist, xbasis, zbasis)
    fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)
    get_grid_data(b) .*= z' .* (Lz .- z')
    get_grid_data(b) .+= Lz .- z'
    ensure_layout!(b, :c)

    return solver, b, u
end

function _assert_subproblem_system(solver)
    @test haskey(solver.problem.parameters, "subproblems")
    subproblems = solver.problem.parameters["subproblems"]
    @test subproblems isa Tuple
    @test length(subproblems) > 0
    @test any(sp -> sp.M_min !== nothing && sp.L_min !== nothing &&
                    (nnz(sp.M_min) > 0 || nnz(sp.L_min) > 0),
              subproblems)
    return subproblems
end

function _assert_finite_fields(b, u)
    ensure_layout!(b, :g)
    @test all(isfinite, get_grid_data(b))
    for comp in u.components
        ensure_layout!(comp, :g)
        @test all(isfinite, get_grid_data(comp))
    end
end

@testset "Legacy pencil symbols remain removed" begin
    @test !isdefined(Tarang, :PencilLinearOperator)
    @test isdefined(Tarang, :step_subproblem_rk!)
    @test isdefined(Tarang, :step_subproblem_multistep!)
end

@testset "Chebyshev-Fourier RK subproblem IMEX" begin
    solver, b, u = _cheb_fourier_rbc_solver(RK222())
    _assert_subproblem_system(solver)

    run!(solver; stop_time=0.02, log_interval=1000)

    @test solver.iteration > 0
    @test solver.sim_time > 0.0
    _assert_finite_fields(b, u)
end

@testset "Chebyshev-Fourier multistep subproblem IMEX" begin
    solver, b, u = _cheb_fourier_rbc_solver(CNAB1())
    _assert_subproblem_system(solver)

    run!(solver; stop_time=0.02, log_interval=1000)

    @test solver.iteration > 0
    @test haskey(solver.timestepper_state.timestepper_data, :sp_multistep_F_rings)
    _assert_finite_fields(b, u)
end
