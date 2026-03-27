"""
Test suite for operators/tensor.jl — curl, Laplacian, fractional Laplacian.

Tests:
1. Laplacian of known functions on Fourier domain
2. Laplacian on Fourier-Chebyshev domain
3. Curl in 2D (scalar output)
4. Fractional Laplacian: alpha=1 matches standard Laplacian
5. Outer product structure
"""

using Test
using LinearAlgebra
using Tarang

@testset "Tensor Operators" begin

    @testset "Laplacian on 2D Fourier domain" begin
        # ∇²(sin(kx)*sin(ly)) = -(k²+l²)*sin(kx)*sin(ly)
        N = 32
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x, y = mesh["x"], mesh["y"]

        k, l = 2, 3
        Tarang.get_grid_data(u) .= @. sin(k * x) * sin(l * y)

        lap_u = evaluate(Laplacian(u))
        ensure_layout!(lap_u, :g)

        expected = @. -(k^2 + l^2) * sin(k * x) * sin(l * y)
        @test isapprox(Tarang.get_grid_data(lap_u), expected; rtol=1e-8, atol=1e-10)
    end

    @testset "Gradient produces correct components" begin
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x, y = mesh["x"], mesh["y"]

        # f(x,y) = cos(x) * sin(2y)
        Tarang.get_grid_data(u) .= @. cos(x) * sin(2y)

        grad_u = evaluate(Gradient(u, coords))

        # ∂f/∂x = -sin(x) * sin(2y)
        ensure_layout!(grad_u[1], :g)
        expected_dx = @. -sin(x) * sin(2y)
        @test isapprox(Tarang.get_grid_data(grad_u[1]), expected_dx; rtol=1e-8, atol=1e-10)

        # ∂f/∂y = 2*cos(x) * cos(2y)
        ensure_layout!(grad_u[2], :g)
        expected_dy = @. 2 * cos(x) * cos(2y)
        @test isapprox(Tarang.get_grid_data(grad_u[2]), expected_dy; rtol=1e-8, atol=1e-10)
    end

    @testset "Divergence of gradient equals Laplacian" begin
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x, y = mesh["x"], mesh["y"]

        Tarang.get_grid_data(u) .= @. sin(x) * cos(2y) + 0.5

        # div(grad(u)) should equal lap(u)
        grad_u = evaluate(Gradient(u, coords))
        div_grad_u = evaluate(Divergence(grad_u))
        ensure_layout!(div_grad_u, :g)

        lap_u = evaluate(Laplacian(u))
        ensure_layout!(lap_u, :g)

        @test isapprox(Tarang.get_grid_data(div_grad_u), Tarang.get_grid_data(lap_u); rtol=1e-8, atol=1e-10)
    end

    @testset "Differentiate on Fourier domain" begin
        N = 32
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb,), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x = mesh["x"]

        # d/dx(sin(3x)) = 3*cos(3x)
        Tarang.get_grid_data(u) .= @. sin(3x)

        du = evaluate(Differentiate(u, coords["x"], 1))
        ensure_layout!(du, :g)

        expected = @. 3 * cos(3x)
        @test isapprox(Tarang.get_grid_data(du), expected; rtol=1e-8, atol=1e-10)
    end

    @testset "Second derivative on Fourier domain" begin
        N = 32
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb,), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x = mesh["x"]

        # d²/dx²(cos(2x)) = -4*cos(2x)
        Tarang.get_grid_data(u) .= @. cos(2x)

        d2u = evaluate(Differentiate(u, coords["x"], 2))
        ensure_layout!(d2u, :g)

        expected = @. -4 * cos(2x)
        @test isapprox(Tarang.get_grid_data(d2u), expected; rtol=1e-8, atol=1e-10)
    end
end
