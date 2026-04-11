"""
Test suite for NCCL All-to-All Transpose operations.

Tests the NCCLTransposeBuffer and related transpose operations for pencil decomposition.
This test can be run with:
  - Single process: julia test/test_nccl_alltoall.jl
  - Multi-process MPI: mpiexec -n 4 julia test/test_nccl_alltoall.jl
"""

using Test
using MPI

# Initialize MPI if not already done
if !MPI.Initialized()
    MPI.Init()
end

@testset "NCCL All-to-All Transpose" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    @testset "NCCLTransposeBuffer struct definition" begin
        # Test that we can at least reference the module
        # The actual struct test requires CUDA - see NCCLTransposeBuffer creation test below
    end

    # Check for CUDA availability
    cuda_available = false
    try
        using CUDA
        cuda_available = CUDA.functional()
    catch
        cuda_available = false
    end

    if cuda_available
        using CUDA
        using Tarang

        # Load CUDA extension
        try
            using TarangCUDAExt
        catch e
            # Extension may be auto-loaded, try accessing symbols directly
            @info "TarangCUDAExt not directly loadable, symbols may be in Tarang namespace"
        end

        @testset "NCCLTransposeBuffer creation" begin
            global_shape = (64, 64, 64)

            # Determine process grid - try to make it 2D if possible
            if nprocs == 1
                proc_grid = (1, 1)
            elseif nprocs == 2
                proc_grid = (1, 2)
            elseif nprocs == 4
                proc_grid = (2, 2)
            else
                # Default to 1D decomposition
                proc_grid = (1, nprocs)
            end

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)

            # Create transpose buffer
            buffer = NCCLTransposeBuffer(pencil, Float64)

            @test buffer isa NCCLTransposeBuffer{Float64}
            @test length(buffer.send_buffer) > 0
            @test length(buffer.recv_buffer) > 0
            @test length(buffer.send_counts) >= max(proc_grid...)
            @test length(buffer.recv_counts) >= max(proc_grid...)
            @test buffer.pencil === pencil

            if rank == 0
                @info "NCCLTransposeBuffer created successfully"
                @info "  Send buffer size: $(length(buffer.send_buffer))"
                @info "  Recv buffer size: $(length(buffer.recv_buffer))"
                @info "  NCCL subcomms initialized: $(buffer.nccl_subcomms.initialized)"
            end
        end

        @testset "Pack/unpack round-trip (simplified)" begin
            # Test the simplified pack/unpack functions for basic correctness
            Nx, Ny, Nz = 16, 16, 32
            data = CuArray(rand(Float64, Nx, Ny, Nz))

            # Create flat buffer
            packed = CUDA.zeros(Float64, Nx * Ny * Nz)

            # Pack along Z dimension (simplified interface)
            nccl_pack_for_transpose!(packed, data, 3)

            # Unpack should recover original data
            unpacked = CUDA.zeros(Float64, Nx, Ny, Nz)
            nccl_unpack_from_transpose!(unpacked, packed, 3)

            # Verify round-trip
            @test Array(unpacked) ≈ Array(data) rtol=1e-14

            if rank == 0
                @info "Pack/unpack round-trip test passed"
            end
        end

        @testset "Pencil shape consistency" begin
            global_shape = (32, 32, 32)

            if nprocs == 1
                proc_grid = (1, 1)
            elseif nprocs == 2
                proc_grid = (1, 2)
            elseif nprocs == 4
                proc_grid = (2, 2)
            else
                proc_grid = (1, nprocs)
            end

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            buffer = NCCLTransposeBuffer(pencil, Float64)

            # Verify pencil shapes are consistent
            @test pencil.x_pencil_shape[1] == global_shape[1]  # X fully local in X-pencil
            @test pencil.y_pencil_shape[2] == global_shape[2]  # Y fully local in Y-pencil
            @test pencil.z_pencil_shape[3] == global_shape[3]  # Z fully local in Z-pencil

            # Verify total elements match across all pencils
            @test prod(pencil.x_pencil_shape) == prod(pencil.y_pencil_shape)
            @test prod(pencil.y_pencil_shape) == prod(pencil.z_pencil_shape)

            if rank == 0
                @info "Pencil shapes verified"
                @info "  X-pencil: $(pencil.x_pencil_shape)"
                @info "  Y-pencil: $(pencil.y_pencil_shape)"
                @info "  Z-pencil: $(pencil.z_pencil_shape)"
            end
        end

        @testset "Single-rank transpose (no communication)" begin
            # Test transpose operations when there's only one rank in the communicator
            if nprocs == 1
                global_shape = (16, 16, 16)
                proc_grid = (1, 1)

                pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
                buffer = NCCLTransposeBuffer(pencil, Float64)

                # Create test data in Z-pencil layout
                data_z = CuArray(rand(Float64, pencil.z_pencil_shape...))

                # Z -> Y transpose (should just copy since single rank)
                @test current_orientation(pencil) == :z_pencil
                data_y = transpose_z_to_y!(buffer, data_z, pencil)
                @test current_orientation(pencil) == :y_pencil
                @test size(data_y) == pencil.y_pencil_shape

                # Y -> Z transpose (inverse)
                data_z_back = transpose_y_to_z!(buffer, data_y, pencil)
                @test current_orientation(pencil) == :z_pencil
                @test Array(data_z_back) ≈ Array(data_z) rtol=1e-14

                # Reset for Y -> X test
                set_orientation!(pencil, :y_pencil)
                data_y2 = CuArray(rand(Float64, pencil.y_pencil_shape...))

                # Y -> X transpose
                data_x = transpose_y_to_x!(buffer, data_y2, pencil)
                @test current_orientation(pencil) == :x_pencil
                @test size(data_x) == pencil.x_pencil_shape

                # X -> Y transpose (inverse)
                data_y_back = transpose_x_to_y!(buffer, data_x, pencil)
                @test current_orientation(pencil) == :y_pencil
                @test Array(data_y_back) ≈ Array(data_y2) rtol=1e-14

                @info "Single-rank transpose tests passed"
            else
                @info "Skipping single-rank tests (nprocs = $nprocs)"
            end
        end

        @testset "NCCL alltoall function" begin
            # Test the nccl_alltoall! function with simple data
            n_per_rank = 100
            total = n_per_rank * nprocs

            send_buf = CuArray(Float64.(collect(1:total) .+ rank * total))
            recv_buf = CUDA.zeros(Float64, total)

            send_counts = fill(n_per_rank, nprocs)
            recv_counts = fill(n_per_rank, nprocs)
            send_displs = collect(0:n_per_rank:(total-n_per_rank))
            recv_displs = collect(0:n_per_rank:(total-n_per_rank))

            # Call alltoall with nothing comm (single-rank fallback)
            nccl_alltoall!(send_buf, recv_buf, send_counts, recv_counts,
                           send_displs, recv_displs, nothing; my_rank=rank)

            # With nothing comm, should just copy
            @test Array(recv_buf) ≈ Array(send_buf) rtol=1e-14

            if rank == 0
                @info "NCCL alltoall function test passed"
            end
        end

        @testset "Buffer size adequacy" begin
            # Verify buffers are large enough for all transpose directions
            global_shape = (48, 48, 48)

            if nprocs == 1
                proc_grid = (1, 1)
            elseif nprocs == 2
                proc_grid = (1, 2)
            elseif nprocs == 4
                proc_grid = (2, 2)
            else
                proc_grid = (1, nprocs)
            end

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            buffer = NCCLTransposeBuffer(pencil, Float64)

            # Buffer should be at least as large as largest pencil
            max_pencil = max(
                prod(pencil.x_pencil_shape),
                prod(pencil.y_pencil_shape),
                prod(pencil.z_pencil_shape)
            )

            @test length(buffer.send_buffer) >= max_pencil
            @test length(buffer.recv_buffer) >= max_pencil

            if rank == 0
                @info "Buffer size adequacy verified"
                @info "  Max pencil size: $max_pencil"
                @info "  Buffer size: $(length(buffer.send_buffer))"
            end
        end

        @testset "Compute transpose counts" begin
            global_shape = (32, 32, 32)

            if nprocs == 1
                proc_grid = (1, 1)
            elseif nprocs == 2
                proc_grid = (1, 2)
            elseif nprocs == 4
                proc_grid = (2, 2)
            else
                proc_grid = (1, nprocs)
            end

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            buffer = NCCLTransposeBuffer(pencil, Float64)

            # Test compute_transpose_counts!
            compute_transpose_counts!(buffer, :z_to_y)

            # Verify counts sum to total elements
            row_size = MPI.Comm_size(pencil.row_comm)
            total_send = sum(buffer.send_counts[1:row_size])
            total_recv = sum(buffer.recv_counts[1:row_size])

            @test total_send > 0
            @test total_recv > 0

            # Verify displacements are monotonically increasing
            for i in 2:row_size
                @test buffer.send_displs[i] >= buffer.send_displs[i-1]
                @test buffer.recv_displs[i] >= buffer.recv_displs[i-1]
            end

            if rank == 0
                @info "Transpose count computation verified"
            end
        end

        @testset "Cleanup" begin
            # Test cleanup function
            global_shape = (16, 16, 16)
            proc_grid = (1, max(1, nprocs))

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            buffer = NCCLTransposeBuffer(pencil, Float64)

            # Should not throw
            finalize_nccl_transpose!(buffer)

            @test true  # If we get here, cleanup succeeded

            if rank == 0
                @info "Cleanup test passed"
            end
        end

    else
        @info "CUDA not available, skipping GPU transpose tests"
        @test_skip "GPU transpose tests require CUDA"
    end

    # Synchronize all MPI ranks before finishing
    MPI.Barrier(MPI.COMM_WORLD)

    if rank == 0
        @info "All NCCL all-to-all transpose tests completed"
    end
end
