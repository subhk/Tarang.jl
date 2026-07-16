using Test
using Tarang

@testset "Lazy RHS Fourier differentiation" begin
    @testset "second RealFourier axis uses FFT wavenumber ordering" begin
        N = 32
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2pi), dealias=3/2)
        ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2pi), dealias=3/2)
        domain = Domain(dist, (xbasis, ybasis))

        q = ScalarField(domain, "q")
        problem = IVP([q])
        add_equation!(problem, "∂t(q) = d(q,y)")
        solver = InitialValueSolver(problem, SBDF1(); dt=1e-3)

        @test solver.rhs_plan !== nothing
        @test solver.rhs_plan.is_compiled

        mesh = Tarang.create_meshgrid(domain)
        x, y = mesh["x"], mesh["y"]
        q["g"] = @. sin(3 * x - 5 * y)

        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :g)

        expected = @. -5 * cos(3 * x - 5 * y)
        @test get_grid_data(rhs) ≈ expected atol=1e-10 rtol=1e-10
    end

    @testset "fractional hyperlaplacian compiles on an explicit RHS" begin
        N = 16
        ν = 2e-5
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        domain = Domain(dist, (xbasis, ybasis))

        q = ScalarField(domain, "q")
        mesh = Tarang.create_meshgrid(domain)
        x, y = mesh["x"], mesh["y"]
        q["g"] = @. sin(2x - 3y)

        problem = IVP([q])
        add_parameters!(problem; ν)
        add_equation!(problem, "∂t(q) = -ν*Δ⁴(q)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)

        @test solver.rhs_plan !== nothing
        @test solver.rhs_plan.is_compiled

        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :g)
        expected = @. -ν * (2^2 + 3^2)^4 * sin(2x - 3y)
        @test get_grid_data(rhs) ≈ expected atol=1e-9 rtol=1e-10
    end

    @testset "fractional wavenumber cache distinguishes domain bounds" begin
        N = 16
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=CPU())

        # Populate the cache with the same array shape/backend but a different
        # physical period before constructing the actual regression case.
        reference_basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        reference_domain = Domain(dist, (reference_basis,))
        reference_q = ScalarField(reference_domain, "reference_q")
        reference_x = Tarang.create_meshgrid(reference_domain)["x"]
        reference_q["g"] = sin.(reference_x)
        reference_problem = IVP([reference_q])
        add_equation!(reference_problem, "∂t(reference_q) = -Δ⁴(reference_q)")
        reference_solver = InitialValueSolver(reference_problem, RK222(); dt=1e-3)
        Tarang.evaluate_rhs(reference_solver, reference_solver.state, 0.0)

        basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 4π))
        domain = Domain(dist, (basis,))
        q = ScalarField(domain, "q")
        x = Tarang.create_meshgrid(domain)["x"]
        q["g"] = sin.(x)

        problem = IVP([q])
        add_equation!(problem, "∂t(q) = -Δ⁴(q)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        @test solver.rhs_plan.is_compiled

        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :g)
        @test get_grid_data(rhs) ≈ .-sin.(x) atol=1e-9 rtol=1e-10
    end
end
