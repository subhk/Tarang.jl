# Guard for GROUP N1: group_pencil_transpose! must group/order fields in a
# RANK-INVARIANT way. Pre-fix _group_pencil_arrays keys on rank-LOCAL size, so
# under an uneven decomposition two fields can collide to the same local size on
# one rank but not another -> ranks partition the field list differently ->
# desynchronised collective transpose! (hang/corruption). This test exercises
# ONLY the (collective-free) grouping helper so it can never hang pre-fix; it
# detects the divergent partition deterministically via an Allgather.

using Tarang
using MPI
using PencilArrays
using Test

if !MPI.Initialized()
    MPI.Init()
end
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

@testset "grouped transpose rank-invariant grouping (rank=$rank)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

    # Decompose the LAST (2nd) axis across the 1-D mesh.
    decomp = (2,)

    # Two fields whose decomposed-axis global sizes (17 vs 16) make their
    # rank-LOCAL sizes COLLIDE on rank 0 (both 8) but DIFFER on rank 1 (9 vs 8):
    #   local_data_range(p,P,N) = ((N*(p-1))/P+1):(N*p/P)
    #   N=17,P=2 -> rank0 1:8 (len 8), rank1 9:17 (len 9)
    #   N=16,P=2 -> rank0 1:8 (len 8), rank1 9:16 (len 8)
    paA = Tarang.create_pencil(dist, (8, 17), decomp; dtype=Float64)
    paB = Tarang.create_pencil(dist, (8, 16), decomp; dtype=Float64)

    @test paA isa PencilArrays.PencilArray
    @test paB isa PencilArrays.PencilArray

    src_arrays = PencilArrays.PencilArray[paA, paB]
    dest_arrays = PencilArrays.PencilArray[paA, paB]

    groups = Tarang._group_pencil_arrays(src_arrays, dest_arrays)

    # Canonical per-rank grouping signature: each field index -> min index of its
    # group. Same partition on all ranks <=> identical signatures.
    n = length(src_arrays)
    rep = zeros(Int, n)
    for inds in values(groups)
        m = minimum(inds)
        for i in inds
            rep[i] = m
        end
    end

    gathered = MPI.Allgather(rep, comm)          # length n*nprocs
    mat = reshape(gathered, n, nprocs)
    # Every rank must agree on the grouping (the core invariant).
    for p in 2:nprocs
        @test mat[:, p] == mat[:, 1]
    end

    # With global-shape keys, the two DISTINCT global shapes form two separate
    # groups on EVERY rank.
    @test rep == [1, 2]
end

MPI.Barrier(comm)
if MPI.Initialized() && !MPI.Finalized()
    MPI.Finalize()
end
