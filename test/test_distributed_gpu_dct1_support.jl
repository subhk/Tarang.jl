"""
Tests for axis_kinds and distributed_gpu_supported predicate.

axis_kinds classifies each basis in a tuple as :real_fourier, :complex_fourier,
or :chebyshev. distributed_gpu_supported decides whether a 3D bases tuple is
eligible for the distributed GPU Chebyshev transform path:
  - must be 3D
  - must have at least one ChebyshevT axis
  - every RealFourier axis must be on dim 1 (the framework convention)
  - RealFourier on dim 1 must not be combined with a Fourier transverse axis
    (the backward Hermitian expansion cannot place conjugate partners at the
    flipped transverse wavenumber — such layouts fall back to CPU)
"""

using Test
using Tarang
using Tarang: axis_kinds, distributed_gpu_supported

@testset "axis_kinds + distributed_gpu_supported" begin
    # Build three real basis objects using the canonical constructor API.
    # Each basis needs a Coordinate from CartesianCoordinates.
    coords = CartesianCoordinates("x", "y", "z")

    rf = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    cf = ComplexFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    cb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))

    # axis_kinds returns a tuple of symbols
    @test axis_kinds((rf, cf, cb)) == (:real_fourier, :complex_fourier, :chebyshev)

    # Not supported: RealFourier on dim 1 with a Fourier transverse axis. The
    # forward pipeline completes, but the backward Hermitian expansion needs
    # conjugate partners at the FLIPPED transverse wavenumber and hard-errors
    # (guard in distributed_backward_dct!) — the predicate rejects the layout
    # up front so it falls back to CPU instead of dying on the first backward.
    @test distributed_gpu_supported((rf, cf, cb)) == false

    # Supported: RealFourier on dim 1 with only Chebyshev transverse axes
    @test distributed_gpu_supported((rf, cb, cb)) == true

    # Supported: no RealFourier at all, has ChebyshevT, 3D
    @test distributed_gpu_supported((cf, cf, cb)) == true

    # Supported: all Chebyshev, 3D
    @test distributed_gpu_supported((cb, cb, cb)) == true

    # Not supported: RealFourier on dim 2 (must be dim 1)
    @test distributed_gpu_supported((cf, rf, cb)) == false

    # Not supported: RealFourier on dim 3 (must be dim 1)
    @test distributed_gpu_supported((cb, cb, rf)) == false

    # Not supported: no ChebyshevT axis
    @test distributed_gpu_supported((rf, cf, cf)) == false

    # Not supported: only 2D
    @test distributed_gpu_supported((rf, cb)) == false
end

# ---------------------------------------------------------------------------
# Helper: build a length-N vector that is a valid Hermitian (real-signal) spectrum.
# DC (and Nyquist for even N) are real; positive-freq entries are arbitrary complex;
# negative-freq entries are their conjugates.  The index convention matches the
# expansion: X[N-k+2] = conj(X[k]) for k = 2 … (N - div(N,2)).
# ---------------------------------------------------------------------------
function _make_hermitian(N)
    full = zeros(ComplexF64, N)
    full[1] = real(randn())                         # DC real
    kmax = iseven(N) ? div(N,2)-1 : div(N,2)        # last non-Nyquist positive freq
    for k in 2:(kmax+1)
        full[k] = randn(ComplexF64)
        full[N-k+2] = conj(full[k])
    end
    if iseven(N)
        full[div(N,2)+1] = real(randn())            # Nyquist real
    end
    return full
end

@testset "hermitian half->full expansion" begin
    using Tarang: _hermitian_full_from_half
    for N in (8, 7, 16, 9, 4, 5)
        full = _make_hermitian(N)
        half = full[1:div(N,2)+1]
        @test _hermitian_full_from_half(half, N) ≈ full
    end
end
