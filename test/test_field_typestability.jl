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
        # @inferred throws unless the inferred return type is concrete — asserts
        # the SerialFieldStorage{G,C} parametrization payoff.
        @test (@inferred Tarang.get_grid_data(u); true)
        @test (@inferred Tarang.get_coeff_data(u); true)
    end

    @testset "copy preserves data, scales, and lazy off-layout buffer" begin
        # Regression guard (review C2/C3): copy must (a) not crash on scaled
        # fields and (b) keep the off-layout array un-materialized (empty) — lazy.
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :c)
        c = copy(u)
        @test Tarang.get_coeff_data(c) == Tarang.get_coeff_data(u)  # live layout copied
        @test isempty(Tarang.get_grid_data(c))                      # off-layout lazy

        us = ScalarField(dist, "us", (xb, yb), Float64)
        set_scales!(us, (1.5, 1.5))
        ensure_layout!(us, :g)
        cs = copy(us)                                               # must not BoundsError
        @test size(Tarang.get_grid_data(cs)) == size(Tarang.get_grid_data(us))
    end
end
