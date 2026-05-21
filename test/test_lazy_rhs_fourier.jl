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
end
