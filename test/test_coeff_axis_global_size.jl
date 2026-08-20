"""
`_coeff_axis_global_size`: only the FIRST Fourier axis is rfft-halved.

The transform plan gives the first Fourier axis RFFT (RealFourier → N/2+1
complex modes) and every subsequent Fourier axis a full-length C2C fft — the
rule `get_separable_dim_size`/`_coefficient_shape_impl` also follow. The
per-mode gather/scatter index math (`_global_to_local_kx`,
`_subproblem_coeff_index`) used to halve EVERY RealFourier axis, so under MPI
`local_indices` described a split of N/2+1 on an axis the pencil decomposes
over N — wrong local mode ranges and silent zero-fill on the upper ranks
(2026-08-20 MPI review, finding T4/V4).

The end-to-end 3D Cheb×RF×RF solve cannot currently be exercised: 3D lift-tau
matrix assembly fails at solver construction (`expression_matrices(::Lift)`
DimensionMismatch — pre-existing, serial too), so this pins the shared sizing
helper the index math now delegates to.
"""

using Test
using Tarang

@testset "_coeff_axis_global_size: first-Fourier-only rfft halving" begin
    coords = CartesianCoordinates("z", "x", "y")
    dist = Distributor(coords; dtype=Float64)
    zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    cb = ComplexFourier(coords["x"]; size=8, bounds=(0.0, 2π))

    u = ScalarField(dist, "u", (zb, xb, yb), Float64)
    @test Tarang._coeff_axis_global_size(u, 1) == 16       # Chebyshev: full
    @test Tarang._coeff_axis_global_size(u, 2) == 5        # first Fourier: N/2+1
    @test Tarang._coeff_axis_global_size(u, 3) == 8        # second RealFourier: FULL

    v = ScalarField(dist, "v", (zb, cb, yb), Float64)
    @test Tarang._coeff_axis_global_size(v, 2) == 8        # ComplexFourier: full
    @test Tarang._coeff_axis_global_size(v, 3) == 8        # RF after CF: FULL (C2C)

    w = ScalarField(dist, "w", (xb, zb), Float64)
    @test Tarang._coeff_axis_global_size(w, 1) == 5        # RF first axis: halved
    @test Tarang._coeff_axis_global_size(w, 2) == 16
end
