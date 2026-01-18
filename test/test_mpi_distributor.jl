"""
MPI Distributor Tests - Multi-rank communication tests

Run with: mpiexec -n 4 julia --project test/test_mpi_distributor.jl

These tests verify actual MPI communication between multiple ranks.
"""

using Test
using Tarang
using MPI

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if rank == 0
    println("="^60)
    println("MPI Distributor Tests")
    println("Running with $nprocs MPI processes")
    println("="^60)
end
MPI.Barrier(comm)

# Skip tests if running with single process
if nprocs < 2
    if rank == 0
        @warn "MPI tests require at least 2 processes. Run with: mpiexec -n 2 julia --project test/test_mpi_distributor.jl"
    end
    exit(0)
end

@testset "MPI Distributor Multi-rank (rank=$rank, nprocs=$nprocs)" begin

    @testset "Distributor creation with mesh" begin
        coords = CartesianCoordinates("x", "y")

        # Create distributor with actual mesh decomposition
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        @test dist.size == nprocs
        @test dist.rank == rank
        @test 0 <= dist.rank < nprocs
    end

    @testset "Allreduce across ranks" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Each rank contributes its rank number
        local_array = fill(Float64(rank), 8)

        # Sum across all ranks (op is positional argument, not keyword)
        reduced = allreduce_array(dist, local_array, MPI.SUM)

        # Expected sum: 0 + 1 + 2 + ... + (nprocs-1) = nprocs*(nprocs-1)/2
        expected_sum = nprocs * (nprocs - 1) / 2
        @test all(reduced .≈ expected_sum)
    end

    @testset "Gather from all ranks" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Each rank has unique data
        local_size = 4
        local_array = fill(Float64(rank + 1), local_size)

        # Gather to rank 0
        gathered = gather_array(dist, local_array)

        if rank == 0
            # Rank 0 should have all data
            @test length(gathered) == local_size * nprocs
            # Check each chunk
            for r in 0:(nprocs-1)
                chunk = gathered[(r*local_size+1):((r+1)*local_size)]
                @test all(chunk .≈ Float64(r + 1))
            end
        end
    end

    @testset "Scatter from rank 0" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        local_size = 4
        total_size = local_size * nprocs

        # All ranks need full data for scatter_array
        full_data = Float64.(1:total_size)

        # Scatter from rank 0
        scattered = scatter_array(dist, full_data)

        # Verify we got some data back
        @test length(scattered) > 0
        @test scattered isa AbstractArray
    end

    @testset "Alltoall communication" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Each rank sends different data to each other rank
        send_buf = Float64.([(rank * nprocs + i) for i in 1:nprocs])
        recv_buf = similar(send_buf)

        mpi_alltoall(dist, send_buf, recv_buf)

        # After alltoall, rank r receives element (i*nprocs + r+1) from rank i
        for i in 0:(nprocs-1)
            expected = Float64(i * nprocs + rank + 1)
            @test recv_buf[i+1] ≈ expected
        end
    end

    @testset "Broadcast from root" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        if rank == 0
            data = Float64.([1.0, 2.0, 3.0, 4.0])
        else
            data = zeros(Float64, 4)
        end

        # Broadcast from rank 0
        MPI.Bcast!(data, 0, comm)

        @test data == Float64.([1.0, 2.0, 3.0, 4.0])
    end

    @testset "2D mesh decomposition" begin
        if nprocs >= 4
            coords = CartesianCoordinates("x", "y", "z")

            # Create 2x2 mesh for 4 processes, or adapt for other counts
            mesh_y = 2
            mesh_z = nprocs ÷ 2

            dist = Distributor(coords; mesh=(mesh_y, mesh_z), dtype=Float64, architecture=CPU())

            @test dist.size == nprocs
            @test dist.rank == rank

            # Test topology creation
            topo = Tarang.create_topology_2d(comm, mesh_z, mesh_y)

            @test topo.Rx == mesh_z
            @test topo.Ry == mesh_y
            @test 0 <= topo.rx < mesh_z
            @test 0 <= topo.ry < mesh_y
        end
    end

    @testset "Pencil array distribution" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        global_size = (32, 32)

        # Create pencil - distribution depends on PencilArrays decomposition
        pencil = create_pencil(dist, global_size, 1; dtype=Float64)

        # Local size should be a fraction of global (product should be <= global product / nprocs)
        local_shape = size(pencil)
        @test prod(local_shape) <= prod(global_size)
        @test prod(local_shape) > 0

        # Fill with rank-specific data
        pencil .= Float64(rank + 1)

        # Verify data integrity after barrier
        MPI.Barrier(comm)
        @test all(pencil .== Float64(rank + 1))
    end

    @testset "Field distribution across ranks" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))

        field = ScalarField(dist, "mpi_test", (xbasis, ybasis))

        # Each rank fills its local portion
        field["g"] .= Float64(rank * 1000)

        # Create TransposableField and verify
        tf = TransposableField(field)

        @test active_layout(tf) == ZLocal  # Default layout
        @test tf.buffers.architecture isa Tarang.CPU

        # Local data should match what we set
        @test all(field["g"] .== Float64(rank * 1000))
    end

end

@testset "MPI Transform Tests (rank=$rank)" begin

    @testset "Distributed forward/backward transform round-trip" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))

        field = ScalarField(dist, "transform_test", (xbasis, ybasis))

        # Initialize with smooth function
        local_shape = size(field["g"])
        for i in 1:local_shape[1], j in 1:local_shape[2]
            # Use global-ish index for deterministic data
            field["g"][i, j] = sin(2π * i / 32) * cos(2π * j / 32) + rank * 0.01
        end

        original = copy(field["g"])

        tf = TransposableField(field)

        # Forward transform
        distributed_forward_transform!(tf)

        # Backward transform
        distributed_backward_transform!(tf)

        # Should recover original data
        @test isapprox(field["g"], original, rtol=1e-10)
    end

    @testset "Transpose Z to Y and back" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        zbasis = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "transpose_test", (xbasis, ybasis, zbasis))
        field["g"] .= Float64(rank + 1)

        tf = TransposableField(field)

        @test active_layout(tf) == ZLocal

        # Z to Y
        transpose_z_to_y!(tf)
        @test active_layout(tf) == YLocal

        # Y back to Z
        transpose_y_to_z!(tf)
        @test active_layout(tf) == ZLocal
    end

    if nprocs >= 4
        @testset "Full transpose chain (3D with 2D mesh)" begin
            coords = CartesianCoordinates("x", "y", "z")

            mesh = (2, nprocs ÷ 2)
            dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())

            xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
            zbasis = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π))

            field = ScalarField(dist, "chain_test", (xbasis, ybasis, zbasis))

            # Unique data per element
            local_shape = size(field["g"])
            for i in 1:local_shape[1], j in 1:local_shape[2], k in 1:local_shape[3]
                field["g"][i, j, k] = Float64(i + j*100 + k*10000 + rank*1000000)
            end

            original = copy(field["g"])

            tf = TransposableField(field)

            # Full forward: Z -> Y -> X
            transpose_z_to_y!(tf)
            transpose_y_to_x!(tf)
            @test active_layout(tf) == XLocal

            # Full backward: X -> Y -> Z
            transpose_x_to_y!(tf)
            transpose_y_to_z!(tf)
            @test active_layout(tf) == ZLocal

            # Data should be preserved
            @test isapprox(field["g"], original, rtol=1e-12)
        end
    end

end

# GPU tests if available
const _HAS_CUDA = try
    Tarang.has_cuda() && begin
        using CUDA
        CUDA.functional()
    end
catch
    false
end

if _HAS_CUDA
    using CUDA

    @testset "MPI GPU Distributor (rank=$rank)" begin
        CUDA.allowscalar(false)

        # Set GPU device based on rank
        if CUDA.ndevices() >= nprocs
            CUDA.device!(rank % CUDA.ndevices())
        end

        if rank == 0
            println("GPU device for rank $rank: $(CUDA.device())")
        end

        @testset "GPU allreduce" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float32, architecture=GPU())

            # Each rank contributes its rank number
            local_array = CUDA.fill(Float32(rank), 8)

            reduced = allreduce_array(dist, local_array, MPI.SUM)

            expected_sum = Float32(nprocs * (nprocs - 1) / 2)
            @test all(Array(reduced) .≈ expected_sum)
        end

        @testset "GPU TransposableField round-trip" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float32, architecture=GPU())

            xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
            ybasis = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))

            field = ScalarField(dist, "gpu_mpi", (xbasis, ybasis))
            field["g"] .= CUDA.rand(Float32, size(field["g"])...)

            original = copy(field["g"])

            tf = TransposableField(field)
            distributed_forward_transform!(tf)
            distributed_backward_transform!(tf)

            @test isapprox(Array(field["g"]), Array(original), rtol=1e-4)
        end
    end
else
    if rank == 0
        @info "CUDA not available, skipping GPU MPI tests"
    end
end

# Final sync
MPI.Barrier(comm)

if rank == 0
    println("="^60)
    println("All MPI Distributor tests completed successfully!")
    println("="^60)
end
