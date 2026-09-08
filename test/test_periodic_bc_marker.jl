using Test, Tarang

@testset "Periodic boundary markers do not add equations" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    u = ScalarField(Domain(dist, (basis,)), "u")
    set!(u, x -> sin(x) + 0.25cos(3x))
    initial = copy(get_grid_data(u))
    problem = InitialValueProblem([u])
    add_equation!(problem, "dt(u) = 0")
    marker = periodic_bc("u", "x")

    added = try
        add_bc!(problem, marker)
    catch err
        err
    end
    @test added === marker
    @test problem.bc_manager.conditions == [marker]
    @test isempty(problem.boundary_conditions)
    @test length(problem.equations) == 1
    @test isempty(apply_boundary_conditions!(problem.bc_manager, problem))

    solver = InitialValueSolver(problem, RK222(); dt=0.01)
    for _ in 1:3
        step!(solver)
    end
    ensure_layout!(u, :g)
    @test get_grid_data(u) ≈ initial atol=1e-12
    @test length(problem.equations) == 1
    @test isempty(problem.bc_manager.bc_equation_indices)
end
