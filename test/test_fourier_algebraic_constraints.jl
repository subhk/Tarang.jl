using Test
using Tarang

@testset "2D Fourier streamfunction velocity algebraic solve" begin
    Nx, Ny = 16, 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2pi), dealias=3/2)
    ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2pi), dealias=3/2)
    domain = Domain(dist, (xbasis, ybasis))

    psi = ScalarField(domain, "psi")
    u = VectorField(domain, "u")

    problem = IVP([psi, u])
    add_equation!(problem, "∂t(psi) = 0")
    add_equation!(problem, "u - skew(grad(psi)) = 0")

    solver = InitialValueSolver(problem, SBDF1(); dt=1e-3)

    x = Tarang.get_grid_coordinates(domain; on_device=false)["x"]
    y = Tarang.get_grid_coordinates(domain; on_device=false)["y"]
    psi["g"] = sin.(x) .* cos.(y')

    step!(solver, 1e-3)

    ensure_layout!(u.components[1], :g)
    ensure_layout!(u.components[2], :g)

    expected_ux = sin.(x) .* sin.(y')
    expected_uy = cos.(x) .* cos.(y')

    @test get_grid_data(u.components[1]) ≈ expected_ux atol=1e-10
    @test get_grid_data(u.components[2]) ≈ expected_uy atol=1e-10
end

@testset "Serial lazy RHS refreshes algebraic state before nonlinear RHS" begin
    Nx, Ny = 16, 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2pi), dealias=3/2)
    ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2pi), dealias=3/2)
    domain = Domain(dist, (xbasis, ybasis))

    q = ScalarField(domain, "q")
    psi = ScalarField(domain, "psi")
    u = VectorField(domain, "u")
    tau_psi = ScalarField(dist, "tau_psi", (), Float64)

    problem = IVP([q, psi, u, tau_psi])
    add_equation!(problem, "∂t(q) = -u⋅∇(q)")
    add_equation!(problem, "Δ(psi) + tau_psi - q = 0")
    add_equation!(problem, "u - skew(grad(psi)) = 0")
    add_bc!(problem, "integ(psi) = 0")

    solver = InitialValueSolver(problem, SBDF1(); dt=1e-3)
    @test solver.rhs_plan !== nothing
    @test solver.rhs_plan.is_compiled

    x = Tarang.get_grid_coordinates(domain; on_device=false)["x"]
    y = Tarang.get_grid_coordinates(domain; on_device=false)["y"]
    q["g"] = sin.(x) .* cos.(y') .+ 0.25 .* cos.(2 .* x) .* sin.(y')

    rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
    ensure_layout!(rhs, :g)
    ensure_layout!(u.components[1], :g)
    ensure_layout!(u.components[2], :g)

    @test maximum(abs.(get_grid_data(u.components[1]))) > 1e-6
    @test maximum(abs.(get_grid_data(u.components[2]))) > 1e-6
    @test maximum(abs.(get_grid_data(rhs))) > 1e-6
end
