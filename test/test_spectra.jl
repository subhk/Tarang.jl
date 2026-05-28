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

    @testset "power scales with amplitude squared" begin
        ps1 = Tarang.power_spectrum(_mode_field((x, y) -> cos(3x)))
        ps2 = Tarang.power_spectrum(_mode_field((x, y) -> 2.0 * cos(3x)))
        p1 = maximum(ps1.power)
        p2 = maximum(ps2.power)
        @test isapprox(p2, 4 * p1; rtol=1e-6)
    end
end
