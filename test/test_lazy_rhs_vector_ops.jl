# Guard: lap / div(grad(·)) / div(·) on the EXPLICIT RHS must compile to the lazy
# (type-specialized) RHS plan, and must agree with the interpreted evaluator.
#
# translate_to_lazy handled Differentiate, arithmetic, Future and ufuncs, but had no
# case for the vector/tensor operators. Any RHS containing lap/div/grad therefore
# returned `nothing`, and build_lazy_rhs_plan! then bailed for the WHOLE solver —
# dropping every equation onto the "~100x-slower interpreted RHS evaluator" (its own
# warning). `lap(q)` on the IMPLICIT side is fine (it goes into the L matrix and never
# reaches the RHS), which is why the existing benchmarks and guards never saw this.
#
# The distributed cost is worse than the serial one: the interpreted Fourier derivative
# copies the whole field and does a full N-D distributed FFT round-trip per derivative
# (derivatives_fourier.jl), versus the lazy path's single-axis in-place coefficient
# scaling.
using Test
using Tarang

const N = 16
const NU = 0.02

function _setup()
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    coords, dist, domain
end

function _init!(q)
    ensure_layout!(q, :g)
    xs = [2π*(i-1)/N for i in 1:N]
    get_grid_data(q) .= [sin(2xs[i] - xs[j]) + 0.5cos(xs[i] + 3xs[j]) for i in 1:N, j in 1:N]
    ensure_layout!(q, :c)
    q
end

# Build a solver whose explicit RHS is `rhs_str`, with `params` bound.
function _solver(rhs_str, params::NamedTuple=NamedTuple())
    _, _, domain = _setup()
    q = ScalarField(domain, "q"); _init!(q)
    problem = IVP([q])
    add_parameters!(problem; nu=NU, params...)
    add_equation!(problem, "dt(q) = $rhs_str")
    InitialValueSolver(problem, RK222(); dt=1e-3), q
end

_grid(f) = (ensure_layout!(f, :g); copy(get_grid_data(f)))

@testset "lazy RHS compiles vector/tensor operators" begin
    @testset "lap(q) on the RHS compiles" begin
        solver, _ = _solver("nu*lap(q)")
        @test solver.rhs_plan.is_compiled
    end

    @testset "lap(q) compiled result == interpreted evaluator" begin
        solver, q = _solver("nu*lap(q)")
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        expected = NU .* _grid(evaluate(lap(q)))
        @test isapprox(_grid(F[1]), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "div(grad(q)) on the RHS compiles and matches lap(q)" begin
        solver, q = _solver("nu*div(grad(q))")
        @test solver.rhs_plan.is_compiled
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        expected = NU .* _grid(evaluate(lap(q)))
        @test isapprox(_grid(F[1]), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "div(vector field) on the RHS compiles and matches interpreted" begin
        coords, dist, domain = _setup()
        q = ScalarField(domain, "q"); _init!(q)
        u = VectorField(dist, coords, "u", (domain.bases[1], domain.bases[2]), Float64)
        for (k, c) in enumerate(u.components)
            ensure_layout!(c, :g)
            xs = [2π*(i-1)/N for i in 1:N]
            get_grid_data(c) .= [sin(k*xs[i]) * cos(xs[j]) for i in 1:N, j in 1:N]
            ensure_layout!(c, :c)
        end
        problem = IVP([q])
        add_parameters!(problem, nu=NU, u=u)
        add_equation!(problem, "dt(q) = nu*div(u)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        @test solver.rhs_plan.is_compiled
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        expected = NU .* _grid(evaluate(Tarang.Divergence(u)))
        @test isapprox(_grid(F[1]), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "div(a*u) conservative flux compiles and matches interpreted" begin
        coords, dist, domain = _setup()
        q = ScalarField(domain, "q"); _init!(q)
        a = ScalarField(domain, "a"); ensure_layout!(a, :g)
        xs = [2π*(i-1)/N for i in 1:N]
        get_grid_data(a) .= [2.0 + sin(xs[i]) * cos(xs[j]) for i in 1:N, j in 1:N]
        ensure_layout!(a, :c)
        u = VectorField(dist, coords, "u", (domain.bases[1], domain.bases[2]), Float64)
        for (k, c) in enumerate(u.components)
            ensure_layout!(c, :g)
            get_grid_data(c) .= [sin(k*xs[i]) * cos(xs[j]) for i in 1:N, j in 1:N]
            ensure_layout!(c, :c)
        end
        problem = IVP([q])
        add_parameters!(problem, a=a, u=u)
        add_equation!(problem, "dt(q) = -div(a*u)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        @test solver.rhs_plan.is_compiled
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        expected = .-_grid(evaluate(Tarang.Divergence(a * u)))
        @test isapprox(_grid(F[1]), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "div(a*grad(q)) variable-coefficient diffusion compiles and matches interpreted" begin
        coords, dist, domain = _setup()
        q = ScalarField(domain, "q"); _init!(q)
        a = ScalarField(domain, "a"); ensure_layout!(a, :g)
        xs = [2π*(i-1)/N for i in 1:N]
        get_grid_data(a) .= [1.5 + 0.4*cos(xs[i]) * sin(xs[j]) for i in 1:N, j in 1:N]
        ensure_layout!(a, :c)
        problem = IVP([q])
        add_parameters!(problem, a=a)
        add_equation!(problem, "dt(q) = div(a*grad(q))")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        @test solver.rhs_plan.is_compiled
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        expected = _grid(evaluate(Tarang.Divergence(a * grad(q))))
        @test isapprox(_grid(F[1]), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "constant coefficient div(c*u) still compiles and matches c*div(u)" begin
        coords, dist, domain = _setup()
        q = ScalarField(domain, "q"); _init!(q)
        u = VectorField(dist, coords, "u", (domain.bases[1], domain.bases[2]), Float64)
        xs = [2π*(i-1)/N for i in 1:N]
        for (k, c) in enumerate(u.components)
            ensure_layout!(c, :g)
            get_grid_data(c) .= [sin(k*xs[i]) * cos(xs[j]) for i in 1:N, j in 1:N]
            ensure_layout!(c, :c)
        end
        problem = IVP([q])
        add_parameters!(problem, u=u)
        add_equation!(problem, "dt(q) = -2.5*div(u)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        @test solver.rhs_plan.is_compiled
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        expected = -2.5 .* _grid(evaluate(Tarang.Divergence(u)))
        @test isapprox(_grid(F[1]), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "Legendre lap() compiles AND is correct" begin
        # `differentiation_matrix` for Legendre is the classical recurrence for UNNORMALIZED
        # Pₙ, but the transform stores ORTHONORMAL P̃ₙ coefficients. Applying it straight to a
        # field's coefficients was silently wrong — measured err 5.335 on an amplitude-11.8
        # answer, wrong sign near 0 — so the translator used to DECLINE here.
        #
        # That mismatch is now bridged once, at `spectral_derivative_matrix`, so the lazy path
        # is correct and compiles. The value assertion is the point; `is_compiled` is asserted
        # too so a silent regression to the ~100x-slower interpreted path is caught.
        N = 10
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, architecture=CPU())
        zb = Legendre(coords["z"]; size=N, bounds=(0.0, 2.0))
        q = ScalarField(dist, "q", (zb,), Float64)
        zg = collect(Tarang.local_grid(zb, dist, 1.0))
        ensure_layout!(q, :g); get_grid_data(q) .= zg .^ 3 .- 0.3 .* zg; ensure_layout!(q, :c)
        problem = IVP([q]); add_parameters!(problem, nu=1.0)
        add_equation!(problem, "dt(q) = nu*lap(q)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-4)
        @test solver.rhs_plan.is_compiled
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        @test isapprox(_grid(F[1]), 6 .* zg; rtol=1e-6, atol=1e-8)   # d²/dz²(z³-0.3z) = 6z
    end

    @testset "bare Legendre derivatives compile and are correct" begin
        N = 10
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, architecture=CPU())
        zb = Legendre(coords["z"]; size=N, bounds=(0.0, 2.0))
        q = ScalarField(dist, "q_bare", (zb,), Float64)
        zg = collect(Tarang.local_grid(zb, dist, 1.0))
        ensure_layout!(q, :g)
        get_grid_data(q) .= zg .^ 3 .- 0.3 .* zg
        ensure_layout!(q, :c)

        problem = IVP([q])
        add_equation!(problem, "dt(q_bare) = d(q_bare,z)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-4)
        @test solver.rhs_plan.is_compiled

        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        @test isapprox(_grid(F[1]), 3 .* zg .^ 2 .- 0.3; rtol=1e-6, atol=1e-8)
    end

    @testset "unsupported operator still bails to the interpreted path" begin
        # curl is deliberately NOT translated: the solver must fall back rather than
        # miscompute. (The interpreted curl path has its own unrelated defect, so this
        # asserts only the fallback decision, not an evaluated value.)
        coords, dist, domain = _setup()
        q = ScalarField(domain, "q"); _init!(q)
        u = VectorField(dist, coords, "u", (domain.bases[1], domain.bases[2]), Float64)
        for c in u.components; ensure_layout!(c, :g); get_grid_data(c) .= 1.0; ensure_layout!(c, :c); end
        problem = IVP([q])
        add_parameters!(problem, nu=NU, u=u)
        add_equation!(problem, "dt(q) = nu*div(curl(u))")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        @test !solver.rhs_plan.is_compiled
    end

    @testset "RHS fallback policy is per solver" begin
        N = 8
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, architecture=CPU())
        zb = Legendre(coords["z"]; size=N, bounds=(0.0, 1.0))

        function legendre_problem(name)
            q = ScalarField(dist, name, (zb,), Float64)
            problem = IVP([q])
            add_equation!(problem, "dt($name) = lap($name)")
            return problem
        end

        # Legendre now COMPILES (the normalization bridge), so it is no longer a
        # vehicle for exercising the decline. The policy plumbing is what this
        # testset is about, so assert that instead: the policy is recorded, and an
        # unknown policy is still rejected at construction.
        compatible = InitialValueSolver(
            legendre_problem("q_compatible"), RK222();
            dt=1e-4, rhs_fallback=:interpreted,
        )
        @test compatible.rhs_fallback_policy === :interpreted
        @test compatible.rhs_plan.is_compiled

        strict = InitialValueSolver(
            legendre_problem("q_strict"), RK222();
            dt=1e-4, rhs_fallback=:strict,
        )
        @test strict.rhs_fallback_policy === :strict
        @test strict.rhs_plan.is_compiled          # :strict is satisfied by compiling

        @test_throws ArgumentError InitialValueSolver(
            legendre_problem("q_invalid"), RK222();
            dt=1e-4, rhs_fallback=:unknown,
        )
    end
end
