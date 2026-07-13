# Guard: two paths that used to SILENTLY REPLACE A TERM WITH ZERO.
#
# Both were found by making the documentation's own examples actually run. Neither raised an
# error, neither allocated differently, and in both cases the solver reported success — the
# answer was simply wrong. That is the worst failure mode there is, so both get a guard.
#
#  A. A COMPOUND-CONSTANT boundary condition. `T(z=0) = 250` worked, but `T(z=0) = 10*25`
#     (or `h*T_amb`, or `1/Re`) arrives as a Multiply/Add operator tree rather than a
#     ConstantOperator, hit a generic fallback in `_evaluate_alg_F`, and was enforced as ZERO.
#     The constant folder it needed (`_is_const_or_param` / `_extract_scalar`) already existed
#     and was already used for the L/M matrices — the BC path just never called it.
#
#  B. An EXPLICIT-RHS term multiplying a UNIT VECTOR, e.g. `∂t(u) - nu*lap(u) = dpdx*ex`.
#     `unit_vector_fields` builds its components as 0-D fields (length-1 data), so
#     `LazyParamField`'s evaluator saw size (1,) vs (Nx,Ny), fell into its shape-mismatch
#     branch and did `fill!(out, 0)`. A pressure-gradient-driven flow was never forced: the
#     fluid stayed exactly at rest while `solver.rhs_plan.is_compiled` reported `true`.
using Test
using Tarang

@testset "terms that must never be silently zeroed" begin

    @testset "compound-constant BC right-hand side is enforced, not zeroed" begin
        Nx, Nz = 8, 12
        coords = CartesianCoordinates("x", "z")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
        domain = Domain(dist, (xb, zb))

        # `250` and `10*25` are the same boundary condition and must be enforced identically.
        function bc_value(rhs::String)
            T    = ScalarField(domain, "T")
            tau1 = ScalarField(dist, "tau_T1", (xb,), Float64)
            tau2 = ScalarField(dist, "tau_T2", (xb,), Float64)
            _, ez = unit_vector_fields(coords, dist)
            lb = derivative_basis(zb, 1)
            τ(A) = lift(A, lb, -1)
            grad_T = grad(T) + ez * τ(tau1)
            pr = IVP([T, tau1, tau2])
            add_parameters!(pr, kappa=1.0, ez=ez, grad_T=grad_T, τ_lift=τ)
            add_equation!(pr, "∂t(T) - kappa*div(grad_T) + τ_lift(tau_T2) = 0")
            add_bc!(pr, "T(z=0) = $rhs")
            add_bc!(pr, "T(z=1) = 0")
            s = InitialValueSolver(pr, RK222(); dt=1e-3)
            for _ in 1:40; step!(s, 1e-3); end
            ensure_layout!(T, :g)
            get_grid_data(T)[1, 1]
        end

        @test isapprox(bc_value("250"), 250.0; rtol=1e-2)
        @test isapprox(bc_value("10*25"), 250.0; rtol=1e-2)   # was 0.0
    end

    @testset "explicit RHS forcing by a unit vector is applied, not zeroed" begin
        # From rest, with no viscosity and no advection, ∂t(u) = F ⇒ u(t) = F·t exactly.
        N, dpdx, nsteps, dt = 8, 1.0, 50, 1e-3
        coords = CartesianCoordinates("x", "y")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb))
        ex, _ = unit_vector_fields(coords, dist)

        u = VectorField(domain, "u")
        for c in u.components
            ensure_layout!(c, :g); get_grid_data(c) .= 0.0; ensure_layout!(c, :c)
        end
        pr = IVP([u])
        add_parameters!(pr, nu=0.0, dpdx=dpdx, ex=ex)
        add_equation!(pr, "∂t(u) - nu*lap(u) = dpdx*ex")
        s = InitialValueSolver(pr, RK222(); dt=dt)
        @test s.rhs_plan.is_compiled          # it always claimed this, even while zeroing the term
        for _ in 1:nsteps; step!(s, dt); end

        ensure_layout!(u.components[1], :g)
        got = maximum(abs.(get_grid_data(u.components[1])))
        @test isapprox(got, dpdx * nsteps * dt; rtol=1e-3)   # was exactly 0.0
    end
end
