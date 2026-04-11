using Test
using Tarang

@testset "CFL Fourier domain" begin
    for N in (16, 32), L in (1.0, 2.0), safety in (0.2, 0.4), velocity_mag in (0.5, 3.0)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)

        x_coord = coords["x"]
        xb = RealFourier(x_coord; size=N, bounds=(0.0, L), dtype=Float64)

        u = VectorField(dist, coords, "u", (xb,), Float64)
        Tarang.ensure_layout!(u.components[1], :g)
        fill!(Tarang.get_grid_data(u.components[1]), velocity_mag)

        problem = IVP([u]; namespace=Dict("u" => u))
        Tarang.add_equation!(problem, "∂t(u) = 0")

        solver = InitialValueSolver(problem, RK111(); device="cpu")
        cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1)
        Tarang.add_velocity!(cfl, u)

        dt = Tarang.compute_timestep(cfl)

        spacing = Tarang.grid_spacing(u.domain)[1]
        expected_dt = safety * spacing / abs(velocity_mag)

        @test isapprox(dt, expected_dt; rtol=1e-6)
    end
end
