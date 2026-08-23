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
    flipped transverse wavenumber — such layouts are rejected explicitly)
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
    # up front so it fails explicitly before the first backward transform.
    @test distributed_gpu_supported((rf, cf, cb)) == false

    # Supported: RealFourier on dim 1 with only Chebyshev transverse axes
    @test distributed_gpu_supported((rf, cb, cb)) == true

    # Supported: no RealFourier at all, has ChebyshevT, 3D
    @test distributed_gpu_supported((cf, cf, cb)) == true

    # Not yet supported: all-Chebyshev Float64 fields have real coefficient
    # storage, while the distributed DCT driver currently emits complex buffers.
    @test distributed_gpu_supported((cb, cb, cb)) == false

    # Not supported: RealFourier on dim 2 (must be dim 1)
    @test distributed_gpu_supported((cf, rf, cb)) == false

    # Not supported: RealFourier on dim 3 (must be dim 1)
    @test distributed_gpu_supported((cb, cb, rf)) == false

    # Not supported: no ChebyshevT axis
    @test distributed_gpu_supported((rf, cf, cf)) == false

    # Not supported: only 2D
    @test distributed_gpu_supported((rf, cb)) == false

    @testset "GPU+MPI domain validation matches the transform predicate" begin
        @test Tarang.validate_mpi_fourier_only(
            (rf, cb, cb), 2; use_pencil_arrays=false,
        )
        @test_throws ErrorException Tarang.validate_mpi_fourier_only(
            (rf, cf, cb), 2; use_pencil_arrays=false,
        )
        @test_throws ErrorException Tarang.validate_mpi_fourier_only(
            (cb, cb, cb), 2; use_pencil_arrays=false,
        )
    end
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

# ── The refusal REASON must be derived from the predicate, not restated ──────
#
# `distributed_gpu_supported` is now defined AS
# `_distributed_gpu_dct_unsupported_reason(bases) === nothing`, because the two
# had drifted: the hand-written refusal text said "every RealFourier axis on dim
# 1", which `(RealFourier, ComplexFourier, ChebyshevT)` satisfies — yet that
# layout is refused (the backward Hermitian expansion cannot place the dim-1
# half-spectrum's conjugate partners at the flipped transverse wavenumber). A
# reader who followed the advice hit the identical message again.
@testset "refusal reason agrees with the predicate and is actionable" begin
    coords = CartesianCoordinates("x", "y", "z")
    names = ("x", "y", "z")
    mk(i, kind) = kind === :RF ? RealFourier(coords[names[i]]; size=8, bounds=(0.0, 2π)) :
                  kind === :CF ? ComplexFourier(coords[names[i]]; size=8, bounds=(0.0, 2π)) :
                                 ChebyshevT(coords[names[i]]; size=8, bounds=(-1.0, 1.0))

    # An INDEPENDENT restatement of the documented rule, so a refactor of the
    # reason function cannot quietly change which layouts are accepted.
    function expected_supported(kinds)
        length(kinds) == 3 || return false
        any(==(:CHEB), kinds) || return false
        any(k -> k in (:RF, :CF), kinds) || return false
        any(i -> kinds[i] === :RF, 2:3) && return false
        kinds[1] === :RF && any(k -> k in (:RF, :CF), kinds[2:3]) && return false
        return true
    end

    for k1 in (:RF, :CF, :CHEB), k2 in (:RF, :CF, :CHEB), k3 in (:RF, :CF, :CHEB)
        kinds = (k1, k2, k3)
        bases = ntuple(i -> mk(i, kinds[i]), 3)
        reason = Tarang.distributed_gpu_unsupported_reason(bases)
        want = expected_supported(kinds)
        @test distributed_gpu_supported(bases) == want
        @test (reason === nothing) == want
        want || @test !isempty(reason)
    end

    # The layout that exposed the drift: it obeys "every RealFourier axis on dim
    # 1" and is still refused, so the reason must NOT tell the reader to move a
    # RealFourier axis to dim 1.
    drift = (mk(1, :RF), mk(2, :CF), mk(3, :CHEB))
    @test !distributed_gpu_supported(drift)
    drift_reason = Tarang.distributed_gpu_unsupported_reason(drift)
    @test occursin("dim 1 is RealFourier", drift_reason)
    @test !occursin("move it to dim 1", drift_reason)

    # "move it to dim 1" is only ever offered when doing so actually helps, i.e.
    # when the RealFourier axis is the only Fourier axis.
    @test occursin("move it to dim 1",
                   Tarang.distributed_gpu_unsupported_reason((mk(1, :CHEB), mk(2, :RF), mk(3, :CHEB))))
    @test !occursin("move it to dim 1",
                    Tarang.distributed_gpu_unsupported_reason((mk(1, :CF), mk(2, :RF), mk(3, :CHEB))))

    # Following the advice must reach a SUPPORTED layout, not another refusal.
    @test distributed_gpu_supported((mk(1, :CF), mk(2, :CF), mk(3, :CHEB)))
    @test distributed_gpu_supported((mk(1, :RF), mk(2, :CHEB), mk(3, :CHEB)))

    # Non-3D and unsupported basis families report their own specific reason.
    @test occursin("3D", Tarang.distributed_gpu_unsupported_reason((mk(1, :RF), mk(2, :CHEB))))
    @test occursin("Chebyshev", Tarang.distributed_gpu_unsupported_reason((mk(1, :CF), mk(2, :CF), mk(3, :CF))))
end
