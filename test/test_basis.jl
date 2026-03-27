"""
Test suite for basis.jl

Tests:
1. RealFourier creation with size, bounds, dealias
2. ComplexFourier creation
3. ChebyshevT creation with size, bounds
4. Legendre creation
5. Basis metadata (size, bounds, element_label)
6. Jacobi parameters for each basis type
7. derivative_basis for all basis families
8. Domain creation from bases
"""

using Test
using LinearAlgebra

@testset "Basis Module" begin
    using Tarang

    # ========================================================================
    # RealFourier Basis
    # ========================================================================

    @testset "RealFourier" begin
        coords = CartesianCoordinates("x")

        @testset "Default construction" begin
            xb = RealFourier(coords["x"]; size=32)
            @test xb isa RealFourier
            @test xb.meta.size == 32
            @test xb.meta.bounds == (0.0, 2π)
            @test xb.meta.dealias == 1.0
            @test xb.meta.dtype == Float64
            @test xb.meta.dim == 1
        end

        @testset "Custom bounds and dealias" begin
            xb = RealFourier(coords["x"]; size=64, bounds=(0.0, 4π), dealias=1.5)
            @test xb.meta.size == 64
            @test xb.meta.bounds == (0.0, 4π)
            @test xb.meta.dealias == 1.5
        end

        @testset "Metadata accessors" begin
            xb = RealFourier(coords["x"]; size=16)
            @test element_label(xb) == "x"
            @test coordsys(xb) === coords
            @test grid_shape(xb) == (16,)
            @test coeff_shape(xb) == (16,)
        end

        @testset "Fourier alias" begin
            xb = Fourier(coords["x"]; size=16)
            @test xb isa RealFourier
        end

        @testset "Convenience constructor (CoordinateSystem, name, size)" begin
            xb = RealFourier(coords, "x", 16)
            @test xb isa RealFourier
            @test xb.meta.size == 16
            @test element_label(xb) == "x"
        end
    end

    # ========================================================================
    # ComplexFourier Basis
    # ========================================================================

    @testset "ComplexFourier" begin
        coords = CartesianCoordinates("x")

        @testset "Default construction" begin
            xb = ComplexFourier(coords["x"]; size=32)
            @test xb isa ComplexFourier
            @test xb.meta.size == 32
            @test xb.meta.bounds == (0.0, 2π)
            @test xb.meta.dtype == ComplexF64
        end

        @testset "Custom parameters" begin
            xb = ComplexFourier(coords["x"]; size=64, bounds=(0.0, 4π), dealias=1.5)
            @test xb.meta.size == 64
            @test xb.meta.bounds == (0.0, 4π)
            @test xb.meta.dealias == 1.5
        end

        @testset "Convenience constructor" begin
            xb = ComplexFourier(coords, "x", 16)
            @test xb isa ComplexFourier
            @test xb.meta.size == 16
        end
    end

    # ========================================================================
    # ChebyshevT Basis
    # ========================================================================

    @testset "ChebyshevT" begin
        coords = CartesianCoordinates("z")

        @testset "Default construction" begin
            zb = ChebyshevT(coords["z"]; size=32)
            @test zb isa ChebyshevT
            @test zb.meta.size == 32
            @test zb.meta.bounds == (-1.0, 1.0)
            @test zb.meta.dealias == 1.0
            @test zb.meta.dtype == Float64
        end

        @testset "Custom bounds" begin
            zb = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
            @test zb.meta.size == 64
            @test zb.meta.bounds == (0.0, 1.0)
        end

        @testset "Jacobi parameters" begin
            zb = ChebyshevT(coords["z"]; size=16)
            @test zb.a == -0.5
            @test zb.b == -0.5
            @test zb.a0 == -0.5
            @test zb.b0 == -0.5
        end

        @testset "Metadata accessors" begin
            zb = ChebyshevT(coords["z"]; size=16)
            @test element_label(zb) == "z"
            @test coordsys(zb) === coords
            @test grid_shape(zb) == (16,)
        end

        @testset "Chebyshev alias" begin
            zb = Chebyshev(coords["z"]; size=16)
            @test zb isa ChebyshevT
        end

        @testset "Affine change of variables" begin
            zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
            cov = zb.meta.COV
            @test cov !== nothing
            @test cov.native_bounds == (-1.0, 1.0)
            @test cov.problem_bounds == (0.0, 1.0)
        end
    end

    # ========================================================================
    # Legendre Basis
    # ========================================================================

    @testset "Legendre" begin
        coords = CartesianCoordinates("z")

        @testset "Default construction" begin
            zb = Legendre(coords["z"]; size=32)
            @test zb isa Legendre
            @test zb.meta.size == 32
            @test zb.meta.bounds == (-1.0, 1.0)
        end

        @testset "Jacobi parameters" begin
            zb = Legendre(coords["z"]; size=16)
            @test zb.a == 0.0
            @test zb.b == 0.0
            @test zb.a0 == 0.0
            @test zb.b0 == 0.0
        end

        @testset "Metadata accessors" begin
            zb = Legendre(coords["z"]; size=16)
            @test element_label(zb) == "z"
            @test coordsys(zb) === coords
        end
    end

    # ========================================================================
    # Fourier Basis Type Checks
    # ========================================================================

    @testset "Fourier Type Helpers" begin
        coords = CartesianCoordinates("x")
        rfb = RealFourier(coords["x"]; size=16)
        cfb = ComplexFourier(coords["x"]; size=16)
        cheb = ChebyshevT(coords["x"]; size=16)

        @test is_fourier_basis(rfb) == true
        @test is_fourier_basis(cfb) == true
        @test is_fourier_basis(cheb) == false

        @test is_complex_fourier_basis(cfb) == true
        @test is_complex_fourier_basis(rfb) == false
    end

    # ========================================================================
    # derivative_basis
    # ========================================================================

    @testset "derivative_basis" begin
        coords = CartesianCoordinates("z")

        @testset "ChebyshevT derivative -> ChebyshevU" begin
            zb = ChebyshevT(coords["z"]; size=32, bounds=(-1.0, 1.0))
            db = derivative_basis(zb)
            @test db isa ChebyshevU
            @test db.meta.size == 32
            @test db.meta.bounds == (-1.0, 1.0)
        end

        @testset "ChebyshevT order=0 returns self" begin
            zb = ChebyshevT(coords["z"]; size=16)
            @test derivative_basis(zb, 0) === zb
        end

        @testset "ChebyshevT order=2 -> ChebyshevV" begin
            zb = ChebyshevT(coords["z"]; size=16)
            db2 = derivative_basis(zb, 2)
            @test db2 isa ChebyshevV
        end

        @testset "Legendre derivative -> Jacobi(1,1)" begin
            zb = Legendre(coords["z"]; size=32)
            db = derivative_basis(zb)
            @test db isa Jacobi
            @test db.a == 1.0
            @test db.b == 1.0
        end

        @testset "Fourier derivative returns self" begin
            coords_x = CartesianCoordinates("x")
            xb = RealFourier(coords_x["x"]; size=32)
            db = derivative_basis(xb)
            @test db === xb

            xbc = ComplexFourier(coords_x["x"]; size=32)
            dbc = derivative_basis(xbc)
            @test dbc === xbc
        end

        @testset "Negative order throws error" begin
            zb = ChebyshevT(coords["z"]; size=16)
            @test_throws ArgumentError derivative_basis(zb, -1)
        end
    end

    # ========================================================================
    # Domain Creation from Bases
    # ========================================================================

    @testset "Domain from Bases" begin
        @testset "1D Fourier Domain" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            domain = Domain(dist, (xb,))
            @test domain.dim == 1
            @test length(domain.bases) == 1
            @test domain.bases[1] === xb
        end

        @testset "2D Fourier-Chebyshev Domain" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))
            domain = Domain(dist, (xb, zb))
            @test domain.dim == 2
            @test length(domain.bases) == 2
        end

        @testset "3D Fourier-Fourier-Chebyshev Domain" begin
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
            domain = Domain(dist, (xb, yb, zb))
            @test domain.dim == 3
            @test length(domain.bases) == 3
        end
    end
end
