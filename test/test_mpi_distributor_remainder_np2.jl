using Tarang
using MPI
using PencilArrays
using Test

if !MPI.Initialized()
    MPI.Init()
end
const comm   = MPI.COMM_WORLD
const rank   = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs != 2
    rank == 0 && @warn "C3 remainder guard needs exactly 2 ranks; got $nprocs"
    MPI.Finalized() || MPI.Finalize()
    exit(0)
end

# 1D mesh decomposes the LAST axis (axis 2). N=5 is non-divisible over 2 ranks:
# PencilArrays gives the remainder to the LAST rank -> rank0=1:2, rank1=3:5.
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; mesh=(2,), dtype=Float64, architecture=CPU())
gshape = (4, 5)

pa  = Tarang.create_pencil(dist, gshape, nothing)
pen = PencilArrays.pencil(pa)
rl  = range_local(pen, LogicalOrder())

li2   = Tarang.local_indices(dist, 2, gshape[2])
shape = Tarang.compute_local_shape(dist, gshape)
local_array_size = Tarang.get_local_array_size(dist, gshape)

ok_local = (li2 == rl[2]) && (shape == size(pa)) && (local_array_size == size(pa))

for r in 0:nprocs-1
    r == rank && println("rank $rank li2=$li2 rl2=$(rl[2]) shape=$shape local_array_size=$local_array_size size(pa)=$(size(pa))")
    MPI.Barrier(comm)
end

ok_global = MPI.Allreduce(ok_local ? 1 : 0, MPI.MIN, comm) == 1

@testset "C3 non-divisible decomposed axis matches PencilArrays (np=2, rank=$rank)" begin
    @test li2 == rl[2]
    @test shape == size(pa)
    @test local_array_size == size(pa)
    @test ok_global
    # Pin the remainder-on-LAST-rank convention explicitly.
    @test li2 == (rank == 0 ? (1:2) : (3:5))
end

MPI.Finalized() || MPI.Finalize()
