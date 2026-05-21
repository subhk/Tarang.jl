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

@testset "CFL multi-dimensional advective frequency" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)

    x_coord = coords["x"]
    y_coord = coords["y"]
    xb = RealFourier(x_coord; size=16, bounds=(0.0, 1.0), dtype=Float64)
    yb = RealFourier(y_coord; size=16, bounds=(0.0, 1.0), dtype=Float64)

    u = VectorField(dist, coords, "u", (xb, yb), Float64)
    Tarang.ensure_layout!(u.components[1], :g)
    Tarang.ensure_layout!(u.components[2], :g)
    fill!(Tarang.get_grid_data(u.components[1]), 1.0)
    fill!(Tarang.get_grid_data(u.components[2]), 1.0)

    problem = IVP([u]; namespace=Dict("u" => u))
    Tarang.add_equation!(problem, "∂t(u) = 0")

    safety = 0.5
    solver = InitialValueSolver(problem, RK111(); device="cpu")
    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1)
    Tarang.add_velocity!(cfl, u)

    dt = Tarang.compute_timestep(cfl)

    dx, dy = Tarang.grid_spacing(u.domain)
    expected_dt = safety / (1.0 / dx + 1.0 / dy)

    @test isapprox(dt, expected_dt; rtol=1e-6)
end
