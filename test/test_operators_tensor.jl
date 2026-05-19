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

# ============================================================================
# Fractional Laplacian Operators
# ============================================================================

@testset "Fractional Laplacian Operators" begin

    # ------------------------------------------------------------------
    # Construction tests
    # ------------------------------------------------------------------

    @testset "FractionalLaplacian construction with various alpha" begin
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)

        for α in [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 3.0]
            op = FractionalLaplacian(u, α)
            @test op isa FractionalLaplacian
            @test op.α == Float64(α)
            @test op.operand === u
        end
    end

    @testset "fraclap convenience constructor" begin
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)

        op = fraclap(u, 0.5)
        @test op isa FractionalLaplacian
        @test op.α == 0.5

        # Unicode alias
        op2 = Δᵅ(u, 0.5)
        @test op2 isa FractionalLaplacian
        @test op2.α == 0.5
    end

    @testset "sqrtlap and invsqrtlap constructors" begin
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)

        sq = sqrtlap(u)
        @test sq isa FractionalLaplacian
        @test sq.α == 0.5

        isq = invsqrtlap(u)
        @test isq isa FractionalLaplacian
        @test isq.α == -0.5
    end

    @testset "hyperlap and Δ² Δ⁴ Δ⁶ Δ⁸ constructors" begin
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)

        # hyperlap with explicit order
        for n in [1, 2, 3, 4]
            op = hyperlap(u, n)
            @test op isa FractionalLaplacian
            @test op.α == Float64(n)
        end

        # hyperlap rejects n < 1
        @test_throws ArgumentError hyperlap(u, 0)

        # Unicode aliases
        @test Δ²(u) isa FractionalLaplacian
        @test Δ²(u).α == 2.0

        @test Δ⁴(u) isa FractionalLaplacian
        @test Δ⁴(u).α == 4.0

        @test Δ⁶(u) isa FractionalLaplacian
        @test Δ⁶(u).α == 6.0

        @test Δ⁸(u) isa FractionalLaplacian
        @test Δ⁸(u).α == 8.0
    end

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    @testset "Fractional Laplacian alpha=1 matches standard Laplacian" begin
        # (-Δ)^1 sin(kx)*sin(ly) = (k²+l²) sin(kx)*sin(ly)
        # Note: FractionalLaplacian computes (-Δ)^α, which has opposite sign from Laplacian().
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

        # Evaluate (-Δ)^1 via fractional Laplacian
        frac_result = evaluate(FractionalLaplacian(u, 1.0))
        ensure_layout!(frac_result, :g)

        # (-Δ)^1 f = (k²+l²) f  =>  sign is positive
        expected = @. (k^2 + l^2) * sin(k * x) * sin(l * y)
        @test isapprox(Tarang.get_grid_data(frac_result), expected; rtol=1e-8, atol=1e-10)

        # Compare with standard Laplacian (which gives -k² f):
        # (-Δ)^1 f should equal -Laplacian(f)
        lap_result = evaluate(Laplacian(u))
        ensure_layout!(lap_result, :g)
        @test isapprox(Tarang.get_grid_data(frac_result), .-Tarang.get_grid_data(lap_result); rtol=1e-8, atol=1e-10)
    end

    @testset "Fractional Laplacian alpha=0.5 (sqrtlap) on single mode" begin
        # (-Δ)^(1/2) sin(kx) = |k| sin(kx) = k sin(kx)   for k > 0
        N = 32
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb,), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x = mesh["x"]

        k = 3
        Tarang.get_grid_data(u) .= @. sin(k * x)

        result = evaluate(sqrtlap(u))
        ensure_layout!(result, :g)

        expected = @. k * sin(k * x)
        @test isapprox(Tarang.get_grid_data(result), expected; rtol=1e-8, atol=1e-10)
    end

    @testset "Fractional Laplacian alpha=2 (Δ²) on single mode" begin
        # (-Δ)^2 sin(kx)*sin(ly) = (k²+l²)² sin(kx)*sin(ly)
        N = 32
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x, y = mesh["x"], mesh["y"]

        k, l = 1, 2
        Tarang.get_grid_data(u) .= @. sin(k * x) * sin(l * y)

        result = evaluate(Δ²(u))
        ensure_layout!(result, :g)

        expected = @. (k^2 + l^2)^2 * sin(k * x) * sin(l * y)
        @test isapprox(Tarang.get_grid_data(result), expected; rtol=1e-8, atol=1e-10)
    end

    @testset "invsqrtlap inverts sqrtlap for nonzero modes" begin
        # (-Δ)^(-1/2) * (-Δ)^(1/2) f = f  for modes with k != 0
        N = 32
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb,), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x = mesh["x"]

        # Use a mode with k > 0 so the inverse is well-defined
        k = 4
        Tarang.get_grid_data(u) .= @. cos(k * x)

        # Apply sqrtlap then invsqrtlap
        forward = evaluate(sqrtlap(u))
        roundtrip = evaluate(invsqrtlap(forward))
        ensure_layout!(roundtrip, :g)

        expected = @. cos(k * x)
        @test isapprox(Tarang.get_grid_data(roundtrip), expected; rtol=1e-8, atol=1e-10)
    end

    @testset "invsqrtlap spectral matrix zeros mean mode" begin
        N = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :c)
        n = length(vec(Tarang.get_coeff_data(u)))

        mat = Tarang._spectral_fractional_laplacian(u, -0.5, n, n)
        d = diag(mat)

        @test d[1] == 0
        @test all(isfinite, d)
    end

    @testset "operator_order returns 2*alpha" begin
        N = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)

        @test Tarang.operator_order(FractionalLaplacian(u, 0.5)) == 1.0
        @test Tarang.operator_order(FractionalLaplacian(u, 1.0)) == 2.0
        @test Tarang.operator_order(FractionalLaplacian(u, 2.0)) == 4.0
        @test Tarang.operator_order(FractionalLaplacian(u, -0.5)) == -1.0
    end

    @testset "is_linear returns true" begin
        N = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)

        @test Tarang.is_linear(FractionalLaplacian(u, 0.5)) == true
        @test Tarang.is_linear(FractionalLaplacian(u, 2.0)) == true
    end
end
