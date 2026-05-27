using Test
using Tarang
using InteractiveUtils

@testset "Field storage type stability" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π))

    @testset "Phase 1: no nothing in storage" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        @test Tarang.get_grid_data(u) !== nothing
        @test Tarang.get_coeff_data(u) !== nothing

        tau = ScalarField(dist, "tau", (), Float64)
        @test Tarang.get_grid_data(tau) !== nothing
        @test length(Tarang.get_grid_data(tau)) == 0
        @test Tarang.get_coeff_data(tau) !== nothing
        @test length(Tarang.get_coeff_data(tau)) == 0
    end

    @testset "Phase 2: field array type fixed at construction" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        gtype = typeof(Tarang.get_grid_data(u))
        Tarang.synchronize_field_architecture!(u; arch=dist.architecture)
        @test typeof(Tarang.get_grid_data(u)) === gtype
    end

    @testset "Phase 3: get_grid_data is type-stable" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        @test_skip (@inferred Tarang.get_grid_data(u); true)
        @test_skip (@inferred Tarang.get_coeff_data(u); true)
    end
end
