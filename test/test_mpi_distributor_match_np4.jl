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

if nprocs != 4
    rank == 0 && @warn "C1/C3 2D-mesh guard needs exactly 4 ranks; got $nprocs"
    MPI.Finalized() || MPI.Finalize()
    exit(0)
end

# Non-degenerate 2x2 mesh (exercises the coord-ordering bug C1 on off-diagonal
# ranks) with a NON-divisible decomposed axis N=5 (exercises remainder bug C3).
coords = CartesianCoordinates("x", "y", "z")
dist   = Distributor(coords; mesh=(2, 2), dtype=Float64, architecture=CPU())
gshape = (8, 5, 6)   # axis1 local; axes 2,3 decomposed over the 2x2 mesh

# Authoritative owned slab straight from PencilArrays (the SAME Pencil the field
# data uses for storage): decompose the LAST ndims_mesh dims, NoPermutation.
pa  = Tarang.create_pencil(dist, gshape, nothing)
pen = PencilArrays.pencil(pa)
rl  = range_local(pen, LogicalOrder())   # one UnitRange per global (logical) axis

# Tarang's notion of the owned ranges / shape, which MUST agree with `rl`.
li    = ntuple(ax -> Tarang.local_indices(dist, ax, gshape[ax]), length(gshape))
shape = Tarang.compute_local_shape(dist, gshape)
local_array_size = Tarang.get_local_array_size(dist, gshape)

ranges_match = all(ax -> li[ax] == rl[ax], 1:length(gshape))
shape_match  = shape == size(pa)
local_size_match = local_array_size == size(pa)
ok_local     = ranges_match && shape_match && local_size_match

for r in 0:nprocs-1
    if r == rank
        println("rank $rank coords_local=$(dist.mpi_topology.coords_local) ",
                "li=$(li) rl=$(Tuple(rl)) shape=$(shape) local_array_size=$(local_array_size) ",
                "size(pa)=$(size(pa)) ok=$ok_local")
    end
    MPI.Barrier(comm)
end

ok_global = MPI.Allreduce(ok_local ? 1 : 0, MPI.MIN, comm) == 1

@testset "C1/C3 Distributor matches PencilArrays slab (2x2 mesh, np=4, rank=$rank)" begin
    @test ranges_match
    @test shape_match
    @test local_size_match
    @test ok_global
end

MPI.Finalized() || MPI.Finalize()
