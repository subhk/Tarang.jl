# Tests for the interpolation operator and its Clenshaw / Jacobi polynomial
# reconstruction helpers (src/core/operators/operations/operations_interpolate.jl).
#
# The polynomial evaluators are checked against closed-form polynomial values so
# the test is an INDEPENDENT oracle: a mismatch means a real bug, not a tautology.
# The field-level interpolation paths set a band-limited function on the basis's
# own nodes and require spectral interpolation to reproduce it exactly.

using Test
using Tarang

# --- Closed-form reference polynomials (independent of the implementation) ---
cheb_T(n, x) = n == 0 ? 1.0 :
               n == 1 ? x :
               n == 2 ? 2x^2 - 1 :
               n == 3 ? 4x^3 - 3x :
               n == 4 ? 8x^4 - 8x^2 + 1 : error("extend cheb_T")

cheb_U(n, x) = n == 0 ? 1.0 :
               n == 1 ? 2x :
               n == 2 ? 4x^2 - 1 :
               n == 3 ? 8x^3 - 4x :
               n == 4 ? 16x^4 - 12x^2 + 1 : error("extend cheb_U")

legendre_P(n, x) = n == 0 ? 1.0 :
                   n == 1 ? x :
                   n == 2 ? (3x^2 - 1) / 2 :
                   n == 3 ? (5x^3 - 3x) / 2 :
                   n == 4 ? (35x^4 - 30x^2 + 3) / 8 : error("extend legendre_P")

const XS = (-0.9, -0.41, 0.0, 0.27, 0.83)

@testset "Interpolation polynomial reconstruction" begin
    @testset "clenshaw_chebyshev_t" begin
        # empty / single-coefficient edge cases
        @test Tarang.clenshaw_chebyshev_t(Float64[], 0.5) == 0.0
        @test Tarang.clenshaw_chebyshev_t([3.0], 0.5) == 3.0
        for x in XS
            for n in 0:4
                c = zeros(n + 1); c[n+1] = 1.0
                @test Tarang.clenshaw_chebyshev_t(c, x) ≈ cheb_T(n, x) atol = 1e-12
            end
            # linear combination: 1*T0 + 2*T1 + 3*T2 + (-1)*T3
            c = [1.0, 2.0, 3.0, -1.0]
            ref = sum(c[k+1] * cheb_T(k, x) for k in 0:3)
            @test Tarang.clenshaw_chebyshev_t(c, x) ≈ ref atol = 1e-12
        end
    end

    @testset "clenshaw_chebyshev_u" begin
        @test Tarang.clenshaw_chebyshev_u(Float64[], 0.5) == 0.0
        @test Tarang.clenshaw_chebyshev_u([2.5], 0.5) == 2.5
        for x in XS
            for n in 0:4
                c = zeros(n + 1); c[n+1] = 1.0
                @test Tarang.clenshaw_chebyshev_u(c, x) ≈ cheb_U(n, x) atol = 1e-12
            end
            c = [0.5, -1.0, 2.0, 1.5]
            ref = sum(c[k+1] * cheb_U(k, x) for k in 0:3)
            @test Tarang.clenshaw_chebyshev_u(c, x) ≈ ref atol = 1e-12
        end
    end

    @testset "clenshaw_legendre" begin
        @test Tarang.clenshaw_legendre(Float64[], 0.5) == 0.0
        @test Tarang.clenshaw_legendre([1.25], 0.5) == 1.25
        for x in XS
            for n in 0:4
                c = zeros(n + 1); c[n+1] = 1.0
                @test Tarang.clenshaw_legendre(c, x) ≈ legendre_P(n, x) atol = 1e-12
            end
            c = [2.0, 1.0, -0.5, 0.75]
            ref = sum(c[k+1] * legendre_P(k, x) for k in 0:3)
            @test Tarang.clenshaw_legendre(c, x) ≈ ref atol = 1e-12
        end
    end

    @testset "jacobi_polynomial reduces to Legendre at a=b=0" begin
        for x in XS, n in 0:4
            @test Tarang.jacobi_polynomial(n, 0.0, 0.0, x) ≈ legendre_P(n, x) atol = 1e-12
        end
    end

    @testset "clenshaw_jacobi matches clenshaw_legendre at a=b=0" begin
        c = [2.0, 1.0, -0.5, 0.75, 0.1]
        for x in XS
            @test Tarang.clenshaw_jacobi(c, x, 0.0, 0.0) ≈ Tarang.clenshaw_legendre(c, x) atol = 1e-10
        end
    end
end

@testset "Interpolation operator (field level)" begin
    @testset "RealFourier spectral interpolation" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh = (1,), dtype = Float64)
        N, L = 16, 2π
        basis = RealFourier(coords["x"]; size = N, bounds = (0.0, L))
        f = ScalarField(dist, "f", (basis,), Float64)
        ensure_layout!(f, :g)

        # band-limited function exactly representable at this resolution
        fx(x) = 2.0 + 3.0 * cos(x) - 1.5 * sin(2x) + 0.7 * cos(3x)
        xj = Array(Tarang.local_grid(basis, dist, 1))
        Tarang.get_grid_data(f) .= fx.(vec(xj))

        for p in (0.0, 0.37, 1.9, 4.2, 6.0)
            op = Tarang.Interpolate(f, coords["x"], p)
            val = Tarang.evaluate_interpolate(op, :g)
            @test val ≈ fx(p) atol = 1e-8
        end
    end

    @testset "ComplexFourier spectral interpolation" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh = (1,), dtype = ComplexF64)
        N, L = 16, 2π
        basis = ComplexFourier(coords["x"]; size = N, bounds = (0.0, L))
        g = ScalarField(dist, "g", (basis,), ComplexF64)
        ensure_layout!(g, :g)

        gx(x) = 2.0 + 3.0 * cos(x) - 1.5 * sin(2x) + 0.7 * cos(3x)
        xj = Array(Tarang.local_grid(basis, dist, 1))
        Tarang.get_grid_data(g) .= ComplexF64.(gx.(vec(xj)))

        for p in (0.0, 0.37, 1.9, 4.2)
            op = Tarang.Interpolate(g, coords["x"], p)
            val = Tarang.evaluate_interpolate(op, :g)
            @test real(val) ≈ gx(p) atol = 1e-8
        end
    end

    @testset "multi-D Fourier interpolation along one axis" begin
        # Interpolating along one Fourier axis of a 2-D field reduces it to a 1-D
        # grid function on the remaining axis. f(x,y)=cos(x)·sin(y): sin(y) has a
        # purely IMAGINARY y-spectrum, so a result that drops the imaginary part
        # (the previous `real.()` bug) collapses to ~0 — this catches that.
        N = 8
        c2 = CartesianCoordinates("x", "y")
        d2 = Distributor(c2; mesh = (1, 1), dtype = Float64)
        bx = RealFourier(c2["x"]; size = N, bounds = (0.0, 2π))
        by = RealFourier(c2["y"]; size = N, bounds = (0.0, 2π))
        f2 = ScalarField(d2, "f2", (bx, by), Float64)
        ensure_layout!(f2, :g)
        pts = collect(0:N-1) .* (2π / N)
        g = Tarang.get_grid_data(f2)
        for i in 1:N, j in 1:N
            g[i, j] = cos(pts[i]) * sin(pts[j])
        end

        # Interpolate along x (the rfft axis) → remaining y axis is full-spectrum.
        x0 = π / 3
        gy = Tarang.evaluate_interpolate(interpolate(f2, c2["x"], x0), :g)
        gy = gy isa Tarang.ScalarField ? Tarang.get_grid_data(gy) : gy
        @test gy ≈ cos(x0) .* sin.(pts) atol = 1e-12

        # Interpolate along y → remaining x axis is the rfft half-spectrum (irfft path).
        ensure_layout!(f2, :g)
        for i in 1:N, j in 1:N
            g[i, j] = cos(pts[i]) * sin(pts[j])
        end
        y0 = π / 6
        hx = Tarang.evaluate_interpolate(interpolate(f2, c2["y"], y0), :g)
        hx = hx isa Tarang.ScalarField ? Tarang.get_grid_data(hx) : hx
        @test hx ≈ sin(y0) .* cos.(pts) atol = 1e-12
    end

    @testset "ChebyshevT polynomial interpolation" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh = (1,), dtype = Float64)
        N = 12
        a, b = -1.0, 1.0
        basis = ChebyshevT(coords["z"]; size = N, bounds = (a, b))
        f = ScalarField(dist, "f", (basis,), Float64)
        ensure_layout!(f, :g)

        # polynomial of degree < N is represented exactly
        fz(z) = 3z^4 - 2z^2 + z - 0.5
        zj = Array(Tarang.local_grid(basis, dist, 1))
        Tarang.get_grid_data(f) .= fz.(vec(zj))

        for p in (-0.8, -0.2, 0.0, 0.5, 0.9)
            op = Tarang.Interpolate(f, coords["z"], p)
            val = Tarang.evaluate_interpolate(op, :g)
            @test val ≈ fz(p) atol = 1e-8
        end
    end
end
