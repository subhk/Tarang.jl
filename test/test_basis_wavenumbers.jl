# Tests for Fourier wavenumber layout helpers (src/core/basis/basis_wavenumbers.jl).
#
# Every expected array is hand-computed from the documented storage layout, so
# the test is an independent oracle, not a mirror of the implementation.
#   RealFourier native:  [cos_0, cos_1, sin_1, cos_2, sin_2, ..., (even-N) cos_Nyq]
#   ComplexFourier / fft: [0, 1, ..., N/2-1, -N/2, ..., -1]  (even N)
#   rfft:                 [0, 1, ..., N/2]                    (length N/2+1)

using Test
using Tarang

const W = Tarang

@testset "basis_wavenumbers.jl" begin
    coords = CartesianCoordinates("x")

    @testset "wavenumbers(RealFourier) native cos/sin layout" begin
        # N even, L=2π  => k0 = 1
        b = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        @test W.wavenumbers(b) ≈ Float64[0, 1, 1, 2, 2, 3, 3, 4]
        # N odd
        bo = RealFourier(coords["x"]; size = 7, bounds = (0.0, 2π))
        @test W.wavenumbers(bo) ≈ Float64[0, 1, 1, 2, 2, 3, 3]
        # domain length scaling: L = 4π => k0 = 0.5
        bL = RealFourier(coords["x"]; size = 8, bounds = (0.0, 4π))
        @test W.wavenumbers(bL) ≈ 0.5 .* Float64[0, 1, 1, 2, 2, 3, 3, 4]
    end

    @testset "wavenumbers(ComplexFourier) fft order" begin
        b = ComplexFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        @test W.wavenumbers(b) ≈ Float64[0, 1, 2, 3, -4, -3, -2, -1]
        bo = ComplexFourier(coords["x"]; size = 7, bounds = (0.0, 2π))
        @test W.wavenumbers(bo) ≈ Float64[0, 1, 2, 3, -3, -2, -1]
    end

    @testset "wavenumbers_rfft length N/2+1, non-negative" begin
        b = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        @test W.wavenumbers_rfft(b) ≈ Float64[0, 1, 2, 3, 4]
        bo = RealFourier(coords["x"]; size = 7, bounds = (0.0, 2π))
        @test W.wavenumbers_rfft(bo) ≈ Float64[0, 1, 2, 3]   # 7÷2 = 3
    end

    @testset "wavenumbers_fft(RealFourier) full fft order" begin
        b = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        @test W.wavenumbers_fft(b) ≈ Float64[0, 1, 2, 3, -4, -3, -2, -1]
        bo = RealFourier(coords["x"]; size = 7, bounds = (0.0, 2π))
        @test W.wavenumbers_fft(bo) ≈ Float64[0, 1, 2, 3, -3, -2, -1]
    end

    @testset "wavenumbers_for_coefficients dispatch" begin
        b = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        @test W.wavenumbers_for_coefficients(b; use_rfft = true) ≈ W.wavenumbers_rfft(b)
        @test W.wavenumbers_for_coefficients(b; use_rfft = false) ≈ W.wavenumbers(b)
        c = ComplexFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        @test W.wavenumbers_for_coefficients(c) ≈ W.wavenumbers(c)
        # Non-Fourier basis: no wavenumber concept
        z = CartesianCoordinates("z")
        cheb = ChebyshevT(z["z"]; size = 8, bounds = (-1.0, 1.0))
        @test W.wavenumbers_for_coefficients(cheb) === nothing
    end

    @testset "internal consistency" begin
        # rfft layout is the non-negative prefix of the fft layout
        b = RealFourier(coords["x"]; size = 16, bounds = (0.0, 2π))
        kfft = W.wavenumbers_fft(b)
        krfft = W.wavenumbers_rfft(b)
        @test krfft[1:(8)] ≈ kfft[1:8]            # 0..N/2-1 match
        @test krfft[end] ≈ 8.0                    # Nyquist N/2
        @test length(krfft) == 9 && length(kfft) == 16
    end
end
