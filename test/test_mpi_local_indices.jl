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

    @testset "global size from pencil cache" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        global_shape = (17, 19)

        @test isempty(dist.layouts)
        create_pencil(dist, global_shape, 1; dtype=Float64)
        @test get_global_size(dist, 1) == global_shape[1]
        @test get_global_size(dist, 2) == global_shape[2]
    end

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

    @testset "three-way agreement: local_indices / get_local_array_size / compute_local_shape" begin
        # test_decomposition_convention.jl has a serial version of this same
        # comparison; a reviewer correctly found it vacuous, since at mesh=(1,)
        # with size==1 every function below takes its identity early-return and
        # nothing is exercised. THIS is the load-bearing version: it runs under
        # LIVE decomposition (np=2, np=4), so the decomposed branches actually
        # run and can actually disagree if a future edit breaks one of them.
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords)
        gshape = (16, 12, 8)

        local_shape_alloc = Tarang.get_local_array_size(dist, gshape)
        local_shape_compute = Tarang.compute_local_shape(dist, gshape)
        @test collect(local_shape_compute) == collect(local_shape_alloc)

        for axis in 1:3
            n_local = length(Tarang.local_indices(dist, axis, gshape[axis]))
            @test n_local == local_shape_alloc[axis]
            @test n_local == local_shape_compute[axis]
        end
    end
end
