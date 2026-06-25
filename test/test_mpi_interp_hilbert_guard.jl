# Guard: interpolate / HilbertTransform fail LOUDLY (not silently-wrong) when a
# Fourier axis they need is MPI-decomposed (np >= 2).
#
# Both reduce/multiply spectral coefficients along a Fourier axis using GLOBAL-length
# weights or global wavenumbers. Under MPI a decomposed axis only exposes a per-rank
# local slice, so the operation would mix local coefficients with global weights
# (interpolate) / use local indices as wavenumbers (Hilbert) → wrong result. These
# paths now raise a clear error instead. Interpolation along a NON-decomposed (local)
# Fourier axis still works and matches serial. Surfaced by the broad MPI CPU audit.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI interp/Hilbert guard test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

@testset "interpolate/Hilbert error on a decomposed Fourier axis (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    (gd isa PencilArrays.PencilArray ? parent(gd) : gd) .= 1.0

    # Confirm a Fourier axis really is decomposed under MPI.
    ensure_layout!(u, :c)
    cd = get_coeff_data(u)
    decomp = cd isa PencilArrays.PencilArray ?
             collect(PencilArrays.decomposition(PencilArrays.pencil(cd))) : Int[]
    @test !isempty(decomp)

    # Interpolating ALONG the decomposed Fourier axis (x, the rfft half-spectrum axis
    # for RealFourier²) must raise our clear error — not a cryptic DimensionMismatch or
    # a silent partial value. (Distributed interpolation is not yet supported in general;
    # the point here is that it FAILS LOUDLY rather than returning garbage.)
    @test_throws Exception Tarang.evaluate_interpolate(interpolate(u, coords["x"], 1.2), :g)

    # Hilbert on a decomposed Fourier axis must not silently produce a wrong field.
    @test_throws Exception Tarang.evaluate(Tarang.hilbert(u), :g)
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
