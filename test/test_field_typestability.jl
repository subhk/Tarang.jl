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

    @testset "copy preserves live data and scales" begin
        # Regression guard (review C3): copy must not crash on a scaled field, and
        # must duplicate (not alias) the live-layout data. The off-layout array is
        # kept full-size (not a 0-sized placeholder) so a later ensure_layout!/
        # transform of the copy can plan its FFT.
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :c)
        Tarang.get_coeff_data(u) .= rand(ComplexF64, size(Tarang.get_coeff_data(u)))
        c = copy(u)
        @test Tarang.get_coeff_data(c) == Tarang.get_coeff_data(u)   # live layout copied
        @test Tarang.get_coeff_data(c) !== Tarang.get_coeff_data(u)  # not aliased
        ensure_layout!(c, :g)  # must be able to plan the backward transform
        @test c.current_layout == :g

        us = ScalarField(dist, "us", (xb, yb), Float64)
        set_scales!(us, (1.5, 1.5))
        ensure_layout!(us, :g)
        cs = copy(us)                                                # must not BoundsError (C3)
        @test size(Tarang.get_grid_data(cs)) == size(Tarang.get_grid_data(us))
    end
end
