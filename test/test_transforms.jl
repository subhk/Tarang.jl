using Test
using Tarang

@testset "Transforms" begin
    @testset "Fourier 2D roundtrip normalization" begin
        # Test that forward → backward transform preserves data (verifies normalization)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))

        field = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(field.domain)
        x = mesh["x"]
        y = mesh["y"]

        # Set smooth test function
        Tarang.get_grid_data(field) .= @. sin(2x) * cos(3y) + 0.5
        original = copy(Tarang.get_grid_data(field))

        # Forward then backward should recover original
        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-12)
    end

    @testset "ComplexFourier 2D roundtrip normalization" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=ComplexF64)
        xb = ComplexFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=32, bounds=(0.0, 2π))

        field = ScalarField(dist, "u", (xb, yb), ComplexF64)
        mesh = Tarang.create_meshgrid(field.domain)
        x = mesh["x"]
        y = mesh["y"]

        # Set smooth test function
        Tarang.get_grid_data(field) .= @. exp(im * (2x + 3y)) + 0.5
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-12)
    end

    @testset "Fourier 3D roundtrip normalization" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        zb = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "u", (xb, yb, zb), Float64)
        mesh = Tarang.create_meshgrid(field.domain)
        x = mesh["x"]
        y = mesh["y"]
        z = mesh["z"]

        Tarang.get_grid_data(field) .= @. sin(x) * cos(y) * sin(2z) + 0.1
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-12)
    end

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
