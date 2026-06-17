"""
Tests for spectral-analysis helpers in extras/flow_tools (scalar `power_spectrum`).

These functions had no direct test coverage. The defining property: a single
Fourier mode must place (almost) all spectral power in the wavenumber bin that
contains that mode, and power must scale with amplitude².
"""

using Test
using Tarang

# Build a 2D periodic scalar field and fill its grid via f(x, y).
function _mode_field(f)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)
    ensure_layout!(u, :g)
    xs = collect(range(0, 2π, length=17))[1:16]
    ys = collect(range(0, 2π, length=17))[1:16]
    gd = Tarang.get_grid_data(u)
    for i in 1:16, j in 1:16
        gd[i, j] = f(xs[i], ys[j])
    end
    return u
end

# Index of the bin whose half-open edge interval [lo, hi) contains k.
function _bin_containing(ps, k)
    for b in 1:length(ps.power)
        if ps.bin_edges[b] <= k < ps.bin_edges[b + 1]
            return b
        end
    end
    return 0
end

@testset "Spectra (power_spectrum)" begin
    @testset "returns a well-formed binned spectrum" begin
        ps = Tarang.power_spectrum(_mode_field((x, y) -> cos(3x)))
        @test ps.k isa AbstractVector
        @test length(ps.power) == length(ps.k)
        @test length(ps.bin_edges) == length(ps.power) + 1
        @test all(ps.power .>= 0)
    end

    @testset "single x-mode concentrates power at its wavenumber" begin
        ps = Tarang.power_spectrum(_mode_field((x, y) -> cos(3x)))
        peak = argmax(ps.power)
        @test _bin_containing(ps, 3) == peak
        @test ps.power[peak] / sum(ps.power) > 0.99
    end

    @testset "single y-mode concentrates power at its wavenumber" begin
        ps = Tarang.power_spectrum(_mode_field((x, y) -> cos(2y)))
        peak = argmax(ps.power)
        @test _bin_containing(ps, 2) == peak
        @test ps.power[peak] / sum(ps.power) > 0.99
    end

    @testset "high x-mode above N/4 is retained, not truncated" begin
        # REGRESSION GUARD: N=16 ⇒ Nyquist N/2 = 8. The rfft x-axis stores N/2+1=9
        # coefficients; the old kmax used (N/2+1)÷2 = 4 ≈ N/4, silently discarding
        # every mode in 5..8. Mode k=6 lies in that dropped band.
        ps = Tarang.power_spectrum(_mode_field((x, y) -> cos(6x)))
        @test maximum(ps.bin_edges) >= 6          # bin range must reach Nyquist, not N/4
        peak = argmax(ps.power)
        @test _bin_containing(ps, 6) == peak       # power lands in the k=6 bin
        @test ps.power[peak] / sum(ps.power) > 0.99
    end

    @testset "power scales with amplitude squared" begin
        ps1 = Tarang.power_spectrum(_mode_field((x, y) -> cos(3x)))
        ps2 = Tarang.power_spectrum(_mode_field((x, y) -> 2.0 * cos(3x)))
        p1 = maximum(ps1.power)
        p2 = maximum(ps2.power)
        @test isapprox(p2, 4 * p1; rtol=1e-6)
    end

    @testset "x-mode and y-mode of equal |k| give identical spectra (isotropy)" begin
        # REGRESSION GUARD: in a RealFourier×RealFourier field the FIRST axis is the
        # rfft half-spectrum (k=0..N/2) but every later axis is a full complex FFT
        # whose wavenumbers run 0..N/2,-N/2+1..-1. The wavenumber builder used to label
        # that 2nd axis as 0..N-1, so each negative-frequency y-mode got a spurious
        # large |k| and was dropped beyond kmax. A cos(2y) field then lost its ky=-2
        # half, making its spectrum HALF that of the physically-identical cos(2x).
        psx = Tarang.power_spectrum(_mode_field((x, y) -> cos(2x)))
        psy = Tarang.power_spectrum(_mode_field((x, y) -> cos(2y)))
        @test _bin_containing(psx, 2) == argmax(psx.power)
        @test _bin_containing(psy, 2) == argmax(psy.power)
        # Isotropy: identical wavenumber in x or y must yield identical peak power.
        @test isapprox(maximum(psy.power), maximum(psx.power); rtol=1e-6)
    end

    @testset "vector energy_spectrum on a non-2π domain keeps all energy" begin
        # REGRESSION GUARD: energy_spectrum bins PHYSICAL wavenumbers (k=2π·n/L) but
        # used a MODE-COUNT ceiling (kmax=N/2). On L<2π the physical k exceeds N/2, so
        # every such mode was binned past kmax and silently dropped → lost energy. The
        # ceiling is now the physical radial Nyquist. Here L=π ⇒ k0=2, so the n=6 mode
        # sits at physical kx=12 > N/2=8 and used to vanish entirely.
        N = 16; L = Float64(π)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        u = VectorField(dist, "u", (xb, yb), Float64)
        xs = collect(range(0, L, length=N + 1))[1:N]
        ensure_layout!(u.components[1], :g)
        g1 = Tarang.get_grid_data(u.components[1])
        for i in 1:N, j in 1:N
            g1[i, j] = cos(2 * π * 6 * xs[i] / L)   # physical kx = 12
        end
        ensure_layout!(u.components[2], :g)
        fill!(Tarang.get_grid_data(u.components[2]), 0.0)

        ps = Tarang.energy_spectrum(u)
        @test maximum(ps.k) >= 12          # physical ceiling reaches the mode (was 8)
        @test sum(ps.bin_counts) > 0
        @test sum(ps.power) > 0            # energy retained (was ≈ 0 — dropped)
        # peak power sits in the bin containing the physical wavenumber 12
        peak_k = ps.k[argmax(ps.power)]
        @test isapprox(peak_k, 12.0; atol = 1.0)
    end
end
