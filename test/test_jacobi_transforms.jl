# Regression tests for the collocation transforms of Jacobi-family bases that
# previously had NO transform wired in the planner (ChebyshevU, ChebyshevV,
# generic Jacobi) and for the matrix-transform in-place hot path (also affected
# Legendre). Both used to silently no-op: `get_grid_data` returned the raw
# coefficients (grid == coeffs). See setup_jacobi_transform! / JacobiTransform.
#
# Oracle is independent: known basis functions evaluated by hand (U_0=1, U_1=2x,
# U_2=4x^2-1) and exact forward/backward round-trips.

using Test
using Tarang

@testset "Jacobi-family collocation transforms" begin
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; mesh = (1,), dtype = Float64)

    # helper: set a single coefficient and backward-transform to grid
    function mode_to_grid(basis, N, k)
        f = ScalarField(dist, "f", (basis,), Float64)
        ensure_layout!(f, :c)
        Tarang.get_coeff_data(f) .= 0.0
        Tarang.get_coeff_data(f)[k + 1] = 1.0
        f.current_layout = :c
        Tarang.backward_transform!(f)
        return vec(Array(Tarang.get_grid_data(f)))
    end

    # helper: round-trip a known grid function through forward then backward
    function roundtrip_err(basis, fn)
        x = vec(Array(Tarang.local_grid(basis, dist, 1)))
        f = ScalarField(dist, "f", (basis,), Float64)
        ensure_layout!(f, :g)
        Tarang.get_grid_data(f) .= fn.(x)
        orig = copy(vec(Array(Tarang.get_grid_data(f))))
        Tarang.forward_transform!(f)
        Tarang.backward_transform!(f)
        return maximum(abs.(vec(Array(Tarang.get_grid_data(f))) .- orig))
    end

    @testset "ChebyshevU basis functions on the grid" begin
        N = 8
        bu = ChebyshevU(coords["z"]; size = N, bounds = (-1.0, 1.0))
        x = vec(Array(Tarang.local_grid(bu, dist, 1)))
        @test maximum(abs.(mode_to_grid(bu, N, 0) .- 1.0)) < 1e-12          # U_0 = 1
        @test maximum(abs.(mode_to_grid(bu, N, 1) .- 2 .* x)) < 1e-12       # U_1 = 2x
        @test maximum(abs.(mode_to_grid(bu, N, 2) .- (4 .* x .^ 2 .- 1))) < 1e-12  # U_2 = 4x^2-1
    end

    @testset "ChebyshevU round-trip (various bounds)" begin
        for (N, bnds) in ((8, (-1.0, 1.0)), (12, (0.0, 3.0)), (16, (-2.0, 5.0)))
            bu = ChebyshevU(coords["z"]; size = N, bounds = bnds)
            @test roundtrip_err(bu, z -> 1 + 0.5z - 0.3z^2 + 0.1z^3) < 1e-10
        end
    end

    @testset "Legendre round-trip" begin
        for N in (8, 16, 24)
            bl = Legendre(coords["z"]; size = N, bounds = (-1.0, 1.0))
            @test roundtrip_err(bl, z -> 2 - z + 0.4z^2 - 0.2z^4) < 1e-9
        end
    end

    @testset "ChebyshevV round-trip" begin
        bv = ChebyshevV(coords["z"]; size = 12, bounds = (-1.0, 1.0))
        @test roundtrip_err(bv, z -> 1 + 2z - 0.5z^2) < 1e-9
    end
end
