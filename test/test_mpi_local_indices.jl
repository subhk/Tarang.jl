using Test
using MPI
MPI.Init()
using Tarang

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)

# Regression guard for the `local_indices` PencilArrays bug: the decomposition
# heuristic always selected mesh_dim=1, so the LOCAL (non-decomposed) leading
# axis was wrongly sliced to global_size/nprocs. Correct behaviour: PencilArrays
# decomposes the LAST ndims_mesh dimensions, so the first axis is full on every
# rank and the trailing axes' local slabs sum to the global size.
@testset "MPI local_indices decomposition (np=$nprocs)" begin
    N = 16

    @testset "2D: leading axis local, trailing axis decomposed" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        xg = local_grid(xb, dist, 1.0; move_to_arch=false)
        yg = local_grid(yb, dist, 1.0; move_to_arch=false)

        # x (axis 1) is never decomposed → full N on every rank.
        @test length(xg) == N
        # y (axis 2) is decomposed → local slabs sum to N across ranks.
        @test MPI.Allreduce(length(yg), MPI.SUM, comm) == N
    end

    @testset "3D: leading axis local on every rank" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        zb = RealFourier(coords["z"]; size=N, bounds=(0.0, 2π))
        xg = local_grid(xb, dist, 1.0; move_to_arch=false)
        yg = local_grid(yb, dist, 1.0; move_to_arch=false)
        zg = local_grid(zb, dist, 1.0; move_to_arch=false)

        # x (axis 1) is local for a 3D domain (mesh decomposes axes 2,3).
        @test length(xg) == N
        # decomposed axes never exceed the global size.
        @test length(yg) <= N
        @test length(zg) <= N
    end
end
