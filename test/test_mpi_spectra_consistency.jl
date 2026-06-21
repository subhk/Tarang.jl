using Test
using MPI
using Tarang

if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "MPI spectra consistency test requires >= 2 processes"
    MPI.Finalize()
    exit(0)
end

# 2D RealFourier x RealFourier VectorField, decomposed across ranks (mesh=(nprocs,)).
N = 16
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(nprocs,), dtype=Float64)
xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
u = VectorField(dist, "u", (xb, yb), Float64)

# Fill each rank's LOCAL grid slab. A constant field puts energy in the DC bin;
# the exact content is irrelevant to the bug under test (a crash + a per-rank
# num_bins mismatch), only that the distributed spectrum path actually runs.
ensure_layout!(u.components[1], :g)
fill!(Tarang.get_grid_data(u.components[1]), 1.0)
ensure_layout!(u.components[2], :g)
fill!(Tarang.get_grid_data(u.components[2]), 0.0)

# PRE-FIX: this throws on every rank — UndefVarError(`axes_local`) inside
# _get_pencil_array_offsets (bug A), and even with A patched the per-rank
# num_bins mismatch crashes the internal MPI.Allreduce (bug B).
es = Tarang.energy_spectrum(u)
# Scalar path (power_spectrum) shares the same two bugs.
ps = Tarang.power_spectrum(u.components[1])

@testset "MPI spectra run + global consistency (np=$nprocs)" begin
    for (name, sp) in (("energy", es), ("power", ps))
        # (1) num_bins identical on every rank (was per-rank-local pre-fix).
        nb = length(sp.power)
        nb_all = MPI.Allgather(nb, comm)
        @test all(==(nb), nb_all)
        # (2) the returned spectrum is globally MPI.Allreduce'd, hence byte-identical
        #     on all ranks — verify by reducing it again with MAX and MIN.
        pmax = MPI.Allreduce(collect(sp.power), MPI.MAX, comm)
        pmin = MPI.Allreduce(collect(sp.power), MPI.MIN, comm)
        @test pmax == pmin
        @test sp.power == pmax
        # (3) energy was actually retained and binned.
        @test sum(sp.bin_counts) > 0
        @test sum(sp.power) > 0
    end
end

MPI.Finalize()
