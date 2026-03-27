using Test
using Tarang

@testset "Subsystems" begin
    @testset "Group Normalization and Coefficient Shapes" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=10, bounds=(0.0, 2π))

        field = ScalarField(dist, "u", (xb, yb), Float64)
        problem = IVP([field])

        struct DummyBase
            matrix_coupling::Vector{Bool}
        end

        struct DummySolver
            problem::Problem
            base::DummyBase
        end

        solver = DummySolver(problem, DummyBase(fill(true, dist.dim)))
        subsystem = Subsystem(solver)

        @test subsystem.group == (nothing, nothing)

        expected_shape = (div(xb.meta.size, 2) + 1, div(yb.meta.size, 2) + 1)
        @test coeff_shape(subsystem, field.domain) == expected_shape
        @test coeff_size(subsystem, field.domain) == prod(expected_shape)
    end
end
