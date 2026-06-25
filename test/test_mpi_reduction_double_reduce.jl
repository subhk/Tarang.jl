# Guard: MPI reductions over field data must NOT double-reduce (np >= 2).
#
# sum()/maximum()/minimum() on a PencilArray are COLLECTIVE in PencilArrays (they
# Allreduce internally and return the GLOBAL value). Code that reduced the
# PencilArray directly and THEN did its own MPI.Allreduce double-reduced → nprocs×
# (sum/mean) or sqrt(nprocs)× (rms) inflation on Intel, and an outright crash on
# non-Intel MPI ("User-defined reduction operators ... not supported"). The fix sums
# the LOCAL slab via parent(). Surfaced by the broad MPI CPU adversarial audit.
#
# Field: u(x,y) = 2 + sin(x)cos(y) on 16² → Σ = 512, mean = 2 (the sin·cos part
# integrates to 0). Velocity (sin·cos, cos·sin) → Σ|v|² = 128, N = 512, rms = 0.5.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI reduction double-reduce test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const N = 16
_setglobal!(f, fn) = begin
    ensure_layout!(f, :g); gd = get_grid_data(f)
    gv = gd isa PencilArrays.PencilArray ? PencilArrays.global_view(gd) : gd
    xg = [(i-1)*2π/N for i in 1:N]
    for I in CartesianIndices(gv); i, j = Tuple(I); gv[I] = fn(xg[i], xg[j]); end
end

@testset "MPI reductions do not double-reduce (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))

    u = ScalarField(dist, "u", (xb, yb), Float64)
    _setglobal!(u, (x, y) -> 2.0 + sin(x)*cos(y))
    @test isapprox(global_sum(u), 512.0; atol=1e-8)     # was nprocs×512 on Intel / crash on ARM
    @test isapprox(global_mean(u), 2.0; atol=1e-10)     # was nprocs×2

    v = VectorField(dist, coords, "v", (xb, yb), Float64)
    _setglobal!(v.components[1], (x, y) -> sin(x)*cos(y))
    _setglobal!(v.components[2], (x, y) -> cos(x)*sin(y))
    stats = turbulence_statistics(v)
    @test isapprox(stats["velocity_rms"], 0.5; atol=1e-10)  # was sqrt(nprocs)×0.5
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
