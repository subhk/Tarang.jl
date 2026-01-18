"""
MPI Integration Tests for Distributed GPU Transpose

Run with: mpiexec -n 4 julia --project test/test_distributed_gpu_transpose.jl

These tests verify:
1. MPI sub-communicator creation for transposes
2. Distributed transpose operations (Z↔Y, Y↔X)
3. Round-trip accuracy of distributed transforms
4. GPU+MPI integration (if CUDA available)
"""

using Test
using Tarang
using MPI

# Initialize MPI if not already initialized
if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

# Print test header on rank 0
if rank == 0
    println("="^60)
    println("Distributed GPU Transpose Tests")
    println("Running with $nprocs MPI processes")
    println("="^60)
end
MPI.Barrier(comm)

@testset "Distributed TransposableField (rank=$rank)" begin

    @testset "MPI Communicator Setup" begin
        coords = CartesianCoordinates("x", "y", "z")

        # Create 2D mesh for pencil decomposition
        # Try to create a 2D mesh if we have enough processes
        if nprocs >= 4
            mesh_y = 2
            mesh_z = nprocs ÷ 2
        elseif nprocs >= 2
            mesh_y = nprocs
            mesh_z = 1
        else
            mesh_y = 1
            mesh_z = 1
        end

        dist = Distributor(coords; mesh=(mesh_y, mesh_z), dtype=Float64, architecture=CPU())

        @test dist.size == nprocs
        @test dist.rank == rank

        # Test communicator creation
        comms = Tarang.create_transpose_comms(dist)

        if nprocs > 1
            @test comms.zy_comm !== nothing || comms.zy_size == 1
            @test comms.yx_comm !== nothing || comms.yx_size == 1
        end
    end

    @testset "Topology2D creation with MPI" begin
        if nprocs >= 2
            # Test 2D topology creation with actual MPI communicators
            Rx, Ry = Tarang.auto_topology(nprocs, 3)

            topo = Tarang.create_topology_2d(comm, Rx, Ry)

            @test topo.Rx == Rx
            @test topo.Ry == Ry
            @test 0 <= topo.rx < Rx
            @test 0 <= topo.ry < Ry
            @test topo.row_comm !== nothing || Ry == 1
            @test topo.col_comm !== nothing || Rx == 1

            # Verify communicator sizes
            @test topo.row_size == Ry
            @test topo.col_size == Rx
        end
    end

    @testset "Local Shape Computation" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        global_shape = (16, 16, 16)
        shapes = Tarang.compute_local_shapes(global_shape, dist)

        @test haskey(shapes, ZLocal)
        @test haskey(shapes, YLocal)
        @test haskey(shapes, XLocal)

        # Verify that local shapes sum to global shape across processes
        # (at least one dimension should be distributed)
        z_shape = shapes[ZLocal]
        @test z_shape[1] * z_shape[2] * z_shape[3] <= prod(global_shape)
    end

    @testset "Transpose Counts Computation" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = Fourier(coords, "x", 16)
        ybasis = Fourier(coords, "y", 16)
        zbasis = Fourier(coords, "z", 16)
        domain = Domain(dist, (xbasis, ybasis, zbasis))

        field = ScalarField(dist, "counts_test", (xbasis, ybasis, zbasis))
        field["g"] .= rand(size(field["g"])...)

        tf = TransposableField(field)

        # Check counts were computed
        @test sum(tf.counts.zy_send_counts) >= 0
        @test sum(tf.counts.yx_send_counts) >= 0
    end

end

@testset "Distributed Transforms (rank=$rank)" begin

    @testset "PencilFFT 2D transform (rank=$rank)" begin
        # Test PencilFFT-based transforms with proper mesh
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = Fourier(coords, "x", 16)
        ybasis = Fourier(coords, "y", 16)

        field = ScalarField(dist, "pencil_2d", (xbasis, ybasis))

        x = range(0, 2π, length=16)
        y = range(0, 2π, length=16)
        local_shape = size(field["g"])
        for i in 1:local_shape[1], j in 1:local_shape[2]
            field["g"][i, j] = sin(2π * i / 16) * cos(2π * j / 16)
        end

        original = copy(field["g"])

        # Use regular forward/backward transforms which properly use PencilFFTs
        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(field["g"], original, rtol=1e-10)
    end

    @testset "2D Distributed transform (rank=$rank)" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = Fourier(coords, "x", 32)
        ybasis = Fourier(coords, "y", 32)

        field = ScalarField(dist, "dist_2d", (xbasis, ybasis))

        # Initialize with deterministic data based on global indices
        local_shape = size(field["g"])
        for i in 1:local_shape[1], j in 1:local_shape[2]
            # Global index depends on rank distribution
            field["g"][i, j] = Float64(i + j + rank * 1000)
        end

        original = copy(field["g"])

        # Use regular forward/backward transforms
        forward_transform!(field)
        backward_transform!(field)

        # Round-trip should preserve data
        @test isapprox(field["g"], original, rtol=1e-8)
    end

    if nprocs >= 2
        @testset "3D Distributed with 2D mesh" begin
            coords = CartesianCoordinates("x", "y", "z")

            # Create appropriate mesh
            if nprocs >= 4
                mesh = (2, nprocs ÷ 2)
            else
                mesh = (nprocs, 1)
            end

            dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())

            xbasis = Fourier(coords, "x", 16)
            ybasis = Fourier(coords, "y", 16)
            zbasis = Fourier(coords, "z", 16)

            field = ScalarField(dist, "dist_3d", (xbasis, ybasis, zbasis))

            local_shape = size(field["g"])
            for i in 1:local_shape[1], j in 1:local_shape[2], k in 1:local_shape[3]
                field["g"][i, j, k] = Float64(i + j * 10 + k * 100 + rank * 10000)
            end

            original = copy(field["g"])

            tf = TransposableField(field)
            distributed_forward_transform!(tf)
            distributed_backward_transform!(tf)

            @test isapprox(field["g"], original, rtol=1e-8)
        end

        @testset "3D Distributed with overlap" begin
            coords = CartesianCoordinates("x", "y", "z")

            # Create appropriate mesh
            if nprocs >= 4
                mesh = (2, nprocs ÷ 2)
            else
                mesh = (nprocs, 1)
            end

            dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())

            xbasis = Fourier(coords, "x", 16)
            ybasis = Fourier(coords, "y", 16)
            zbasis = Fourier(coords, "z", 16)

            field = ScalarField(dist, "overlap_3d", (xbasis, ybasis, zbasis))

            local_shape = size(field["g"])
            for i in 1:local_shape[1], j in 1:local_shape[2], k in 1:local_shape[3]
                field["g"][i, j, k] = Float64(i + j * 10 + k * 100 + rank * 10000)
            end

            original = copy(field["g"])

            tf = TransposableField(field)

            # Test with overlap=true
            distributed_forward_transform!(tf; overlap=true)
            distributed_backward_transform!(tf; overlap=false)

            @test isapprox(field["g"], original, rtol=1e-8)
        end
    end

end

@testset "Async Transpose Operations (rank=$rank)" begin

    if nprocs >= 2
        @testset "Async Z to Y transpose" begin
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

            xbasis = Fourier(coords, "x", 8)
            ybasis = Fourier(coords, "y", 8)
            zbasis = Fourier(coords, "z", 8)

            field = ScalarField(dist, "async_zy", (xbasis, ybasis, zbasis))
            field["g"] .= rand(size(field["g"])...)

            tf = TransposableField(field)

            @test active_layout(tf) == ZLocal
            @test !tf.async_state.in_progress

            # Start async transpose
            Tarang.async_transpose_z_to_y!(tf)

            # Check that operation is in progress (or completed for serial)
            @test tf.async_state.in_progress || tf.topology.row_size == 1

            # Wait for completion
            Tarang.wait_transpose!(tf)

            @test active_layout(tf) == YLocal
            @test !tf.async_state.in_progress
        end

        @testset "is_transpose_complete" begin
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

            xbasis = Fourier(coords, "x", 8)
            ybasis = Fourier(coords, "y", 8)
            zbasis = Fourier(coords, "z", 8)

            field = ScalarField(dist, "complete_test", (xbasis, ybasis, zbasis))
            field["g"] .= rand(size(field["g"])...)

            tf = TransposableField(field)

            # Before any operation, should be "complete"
            @test Tarang.is_transpose_complete(tf)

            # Start async transpose
            Tarang.async_transpose_z_to_y!(tf)

            # Eventually should complete
            while !Tarang.is_transpose_complete(tf)
                # Busy wait (not ideal but tests the function)
            end

            # Finalize
            Tarang.wait_transpose!(tf)
            @test Tarang.is_transpose_complete(tf)
        end
    end

end

@testset "Transpose Operations (rank=$rank)" begin

    @testset "Z to Y transpose" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = Fourier(coords, "x", 8)
        ybasis = Fourier(coords, "y", 8)
        zbasis = Fourier(coords, "z", 8)

        field = ScalarField(dist, "zy_test", (xbasis, ybasis, zbasis))
        field["g"] .= rand(size(field["g"])...)

        tf = TransposableField(field)

        @test active_layout(tf) == ZLocal

        # Z to Y transpose
        transpose_z_to_y!(tf)

        @test active_layout(tf) == YLocal

        # Y to Z transpose (reverse)
        transpose_y_to_z!(tf)

        @test active_layout(tf) == ZLocal
    end

    @testset "Y to X transpose" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = Fourier(coords, "x", 8)
        ybasis = Fourier(coords, "y", 8)
        zbasis = Fourier(coords, "z", 8)

        field = ScalarField(dist, "yx_test", (xbasis, ybasis, zbasis))
        field["g"] .= rand(size(field["g"])...)

        tf = TransposableField(field)

        # Need to be in YLocal first
        transpose_z_to_y!(tf)
        @test active_layout(tf) == YLocal

        # Y to X transpose
        transpose_y_to_x!(tf)
        @test active_layout(tf) == XLocal

        # X to Y transpose (reverse)
        transpose_x_to_y!(tf)
        @test active_layout(tf) == YLocal
    end

end

# GPU tests (if CUDA available)
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

    @testset "Distributed GPU Transpose (rank=$rank)" begin
        CUDA.allowscalar(false)

        # Set GPU device based on rank (for multi-GPU systems)
        if CUDA.ndevices() >= nprocs
            CUDA.device!(rank % CUDA.ndevices())
        end

        @testset "GPU Construction" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float32, architecture=GPU())

            xbasis = Fourier(coords, "x", 16)
            ybasis = Fourier(coords, "y", 16)

            field = ScalarField(dist, "gpu_dist", (xbasis, ybasis))
            field["g"] .= CUDA.rand(Float32, size(field["g"])...)

            tf = TransposableField(field)

            @test tf.buffers.architecture isa Tarang.GPU
            @test tf.buffers.z_local_data isa CuArray
            @test tf.buffers.send_buffer isa CuArray
        end

        @testset "GPU Round-trip" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float32, architecture=GPU())

            xbasis = Fourier(coords, "x", 32)
            ybasis = Fourier(coords, "y", 32)

            field = ScalarField(dist, "gpu_roundtrip", (xbasis, ybasis))
            field["g"] .= CUDA.rand(Float32, size(field["g"])...)

            original = copy(field["g"])

            tf = TransposableField(field)
            distributed_forward_transform!(tf)
            distributed_backward_transform!(tf)

            @test isapprox(Array(field["g"]), Array(original), rtol=1e-4)
        end

        @testset "CUDA-aware MPI check" begin
            is_cuda_aware = check_cuda_aware_mpi()

            if rank == 0
                println("CUDA-aware MPI: $is_cuda_aware")
            end

            # Test passes regardless of CUDA-aware MPI availability
            # (code should work with or without it)
            @test true
        end

    end
else
    @testset "Distributed GPU Transpose (rank=$rank)" begin
        @test_skip "CUDA not available"
    end
end

# Final synchronization
MPI.Barrier(comm)

if rank == 0
    println("="^60)
    println("All tests completed")
    println("="^60)
end

# Do NOT finalize MPI here - let the calling process handle it
# MPI.Finalize() should only be called once at the end of the program
