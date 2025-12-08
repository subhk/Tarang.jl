using Test
using Tarang

@testset "InitialValueSolver minimal workflow" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    field = ScalarField(dist, "u", (basis,), Float64)

    problem = IVP([field])
    Tarang.add_equation!(problem, "∂t(u) = 0")

    solver = InitialValueSolver(problem, RK111(); device="cpu")

    @test solver.problem === problem
    @test length(solver.state) == 1
    @test solver.dt ≈ 1e-3
    @test haskey(problem.parameters, "L_matrix")
    @test solver.base.entry_cutoff ≈ 1e-12
    @test solver.evaluator !== nothing

    Tarang.step!(solver, 1e-3)
    @test solver.iteration == 1
    @test isapprox(solver.sim_time, 1e-3; atol=1e-6)
end
