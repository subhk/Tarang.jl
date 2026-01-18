using Test
using Tarang

@testset "Transforms" begin
    @testset "Legendre 2D forward/backward" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = Legendre(coords["x"]; size=8, bounds=(-1.0, 1.0))
        yb = Legendre(coords["y"]; size=6, bounds=(-1.0, 1.0))

        field = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(field.domain)
        x = mesh["x"]
        y = mesh["y"]

        Tarang.get_grid_data(field) .= @. x^2 + 0.5 * y - 0.3 * x * y + 0.1
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-10)
    end
end
