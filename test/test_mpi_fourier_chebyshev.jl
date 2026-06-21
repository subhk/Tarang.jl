# Guard: MPI mixed Fourier-Chebyshev layout support.
#
# PencilArrays decomposes the TRAILING dimensions; a Chebyshev axis cannot be
# decomposed (its DCT needs the full axis local). So FFC works in MPI only when
# the Chebyshev axis is NOT in the decomposed (trailing) position:
#   * Chebyshev-FIRST (z:Cheb, x:F, y:F) -> supported, round-trips to machine eps.
#   * Chebyshev-LAST  (x:F, y:F, z:Cheb) -> unsupported; must fail with a clear,
#     actionable error (not PencilFFTs' cryptic "decomposed dimensions" message).
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "FFC MPI layout test requires >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

_loc(f) = get_grid_data(f) isa PencilArrays.PencilArray ? parent(get_grid_data(f)) : get_grid_data(f)
function _assign_local!(field, gdata)
    data = get_grid_data(field)
    if data isa PencilArrays.PencilArray
        ax = PencilArrays.pencil(data).axes_local
        parent(data) .= gdata[ax...]
    else
        data .= gdata
    end
end

@testset "FFC MPI: Chebyshev-last errors clearly, Chebyshev-first works (rank=$rank)" begin
    # --- Chebyshev-LAST: unsupported decomposition -> clear, actionable error ---
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=12, bounds=(-1.0, 1.0))
    err = nothing
    try
        Domain(dist, (xb, yb, zb))
    catch e
        err = e
    end
    @test err !== nothing
    @test occursin("Chebyshev", sprint(showerror, err))
    @test occursin(r"reorder|before the Fourier"i, sprint(showerror, err))

    # --- Chebyshev-FIRST: supported -> builds + round-trips a real Fourier mode ---
    coords2 = CartesianCoordinates("z", "x", "y")
    dist2 = Distributor(coords2; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    zb2 = ChebyshevT(coords2["z"]; size=12, bounds=(-1.0, 1.0))
    xb2 = RealFourier(coords2["x"]; size=16, bounds=(0.0, 2π))
    yb2 = ComplexFourier(coords2["y"]; size=16, bounds=(0.0, 2π))
    w = ScalarField(dist2, "w", (zb2, xb2, yb2), Float64)
    ensure_layout!(w, :g)
    xx = [2π*(i-1)/16 for i in 1:16]; yy = [2π*(j-1)/16 for j in 1:16]
    g = [cos(2xx[ix]) + cos(3yy[iy]) for iz in 1:12, ix in 1:16, iy in 1:16]
    _assign_local!(w, g); orig = copy(_loc(w))
    forward_transform!(w); backward_transform!(w); ensure_layout!(w, :g)
    e = MPI.Allreduce(maximum(abs.(_loc(w) .- orig)), MPI.MAX, comm)
    rank == 0 && println("  FFC Cheb-first round-trip maxerr = ", e)
    @test e < 1e-10
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
