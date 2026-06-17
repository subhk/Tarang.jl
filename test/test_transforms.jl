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

    @testset "Scaled RealFourier roundtrip (non-integer N*scale)" begin
        # The scaled grid is allocated with ceil(N*scale); the backward irfft target
        # must use the SAME ceil (it used round, which disagreed for N=7, scale=1.5 →
        # 11 vs 10, breaking the round-trip). Use odd N where ceil != round.
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        for (N, scale) in ((7, 1.5), (11, 1.5), (11, 1.3))
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            field = ScalarField(Domain(dist, (xb,)), "u")
            Tarang.require_scales!(field, scale)
            ensure_layout!(field, :g)
            g = Tarang.get_grid_data(field)
            Ng = length(g)
            @test Ng == ceil(Int, N * scale)
            xs = collect(0:Ng-1) .* (2π / Ng)
            g .= @. 1.0 + cos(xs) + 0.5 * sin(2 * xs)   # band-limited (modes 0,1,2)
            original = copy(g)
            forward_transform!(field)
            backward_transform!(field)
            ensure_layout!(field, :g)
            gout = Tarang.get_grid_data(field)
            @test length(gout) == Ng        # grid length preserved (was collapsing to round)
            @test isapprox(gout, original; rtol=1e-10, atol=1e-10)
        end
    end

    @testset "Scaled ChebyshevT roundtrip (dealiased grid)" begin
        # A scaled Chebyshev field lives on M = ceil(N*scale) Gauss-Lobatto points
        # but stores N coefficients. The backward used the BASE grid_size, so the
        # round-trip collapsed the grid from M back to N (→ NaN). The backward now
        # zero-pads to M and inverse-DCT-I's at size M. Use a band-limited (low-degree)
        # polynomial so truncation to N modes is lossless.
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        for (N, scale) in ((8, 1.5), (12, 1.5), (10, 1.3))
            zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
            field = ScalarField(Domain(dist, (zb,)), "u")
            Tarang.require_scales!(field, scale)
            ensure_layout!(field, :g)
            g = Tarang.get_grid_data(field)
            M = length(g)
            @test M == ceil(Int, N * scale)
            zc = Float64[-cos(pi * (j - 1) / (M - 1)) for j in 1:M]   # scaled GL nodes
            # degree-3 polynomial: representable in N (>=4) Chebyshev modes
            g .= @. 1.0 + 2zc - 0.5 * (2zc^2 - 1) + 0.25 * zc^3
            original = copy(g)
            forward_transform!(field)
            backward_transform!(field)
            ensure_layout!(field, :g)
            gout = Tarang.get_grid_data(field)
            @test length(gout) == M          # grid preserved (was collapsing to base N)
            @test isapprox(gout, original; rtol=1e-10, atol=1e-10)
        end
    end
end

@testset "Transform dispatch helpers" begin
    using Tarang
    import Tarang: _apply_forward, _apply_backward, _find_pencil_plan, Transform

    @testset "Fallback dispatch returns data unchanged" begin
        struct _TestUnknownTransform <: Transform end
        data = randn(ComplexF64, 8)
        @test _apply_forward(data, _TestUnknownTransform()) === data
        @test _apply_backward(data, _TestUnknownTransform()) === data
    end

    @testset "_find_pencil_plan with no plans" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        result = _find_pencil_plan(dist)
        @test result === nothing
    end
end
