"""
NCCL Sub-communicator Tests - Unit tests for NCCL sub-communicators

Run with: julia --project test/test_nccl_subcomm.jl

These tests verify the NCCLSubComms struct and initialization functions
for creating NCCL communicators that mirror MPI row/col communicators
from PencilDecomposition.

Note: These tests skip gracefully when CUDA/NCCL are not available.
"""

using Test
using MPI

# Initialize MPI for testing
if !MPI.Initialized()
    MPI.Init()
end

using Tarang

@testset "NCCL Sub-communicators" begin
    @testset "NCCLSubComms struct exists and defaults" begin
        # Test that we can create the struct with default values
        subcomms = Tarang.NCCLSubComms()
        @test subcomms.initialized == false
        @test subcomms.row_comm === nothing
        @test subcomms.col_comm === nothing
        @test subcomms.row_rank == 0
        @test subcomms.row_size == 1
        @test subcomms.col_rank == 0
        @test subcomms.col_size == 1
    end

    @testset "NCCLSubComms is mutable" begin
        subcomms = Tarang.NCCLSubComms()
        # Should be able to modify fields
        subcomms.row_rank = 5
        @test subcomms.row_rank == 5
        subcomms.initialized = true
        @test subcomms.initialized == true
    end

    @testset "init_nccl_subcomms! returns NCCLSubComms" begin
        # Create simple MPI sub-communicators for testing
        row_comm = MPI.COMM_SELF
        col_comm = MPI.COMM_SELF

        # Call init function - should return NCCLSubComms
        # Without CUDA/NCCL, it should return uninitialized subcomms
        subcomms = Tarang.init_nccl_subcomms!(row_comm, col_comm)

        @test subcomms isa Tarang.NCCLSubComms

        # Without CUDA, initialized should be false
        if !Tarang.has_cuda()
            @test subcomms.initialized == false
            @info "NCCL sub-comm initialization skipped (no CUDA)"
        end
    end

    @testset "init_nccl_subcomms! handles single-rank communicators" begin
        # COMM_SELF has size 1
        row_comm = MPI.COMM_SELF
        col_comm = MPI.COMM_SELF

        subcomms = Tarang.init_nccl_subcomms!(row_comm, col_comm)

        # With size-1 communicators, NCCL comms should be nothing
        # (no point in NCCL for single rank)
        if subcomms.initialized
            @test subcomms.row_comm === nothing  # Single rank, no NCCL needed
            @test subcomms.col_comm === nothing
        end
    end

    @testset "create_nccl_comm_from_mpi function exists" begin
        # Just verify the function exists and is callable
        @test hasmethod(Tarang.create_nccl_comm_from_mpi, Tuple{MPI.Comm})
    end

    @testset "Row/Column communicator sizes with PencilDecomposition" begin
        # Test with simulated 2x2 grid
        proc_grid = (2, 2)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nprocs = MPI.Comm_size(MPI.COMM_WORLD)

        if nprocs >= 4
            # Only run with 4+ processes
            global_shape = (64, 64, 64)

            # Try to load PencilDecomposition
            _HAS_CUDA = try
                using CUDA
                CUDA.functional()
            catch
                false
            end

            if _HAS_CUDA
                using TarangCUDAExt
                pencil = TarangCUDAExt.PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)

                # Row comm should have P2 ranks (2 in 2x2 grid)
                @test MPI.Comm_size(pencil.row_comm) == proc_grid[2]

                # Col comm should have P1 ranks (2 in 2x2 grid)
                @test MPI.Comm_size(pencil.col_comm) == proc_grid[1]

                # Test NCCL subcomm initialization with pencil's MPI comms
                subcomms = Tarang.init_nccl_subcomms!(pencil.row_comm, pencil.col_comm)

                if subcomms.initialized
                    @test subcomms.row_size == proc_grid[2]
                    @test subcomms.col_size == proc_grid[1]
                end
            else
                @info "Skipping PencilDecomposition NCCL test (CUDA not available)"
            end
        else
            @info "Skipping multi-process NCCL test (need 4+ processes, have $nprocs)"
        end
    end

    @testset "NCCLSubComms exported from Tarang" begin
        # Verify the type is properly exported
        @test isdefined(Tarang, :NCCLSubComms)
        @test isdefined(Tarang, :init_nccl_subcomms!)
        @test isdefined(Tarang, :create_nccl_comm_from_mpi)
        @test isdefined(Tarang, :finalize_nccl_subcomms!)
    end

    @testset "finalize_nccl_subcomms! resets struct" begin
        # Create and manually set up a subcomms struct
        subcomms = Tarang.NCCLSubComms()
        subcomms.initialized = true
        subcomms.row_comm = :mock_row_comm  # Use placeholder
        subcomms.col_comm = :mock_col_comm
        subcomms.row_rank = 3
        subcomms.row_size = 4
        subcomms.col_rank = 2
        subcomms.col_size = 2

        # Finalize should reset the struct
        Tarang.finalize_nccl_subcomms!(subcomms)

        @test subcomms.initialized == false
        @test subcomms.row_comm === nothing
        @test subcomms.col_comm === nothing
        # Note: row_rank, row_size, col_rank, col_size are not reset by finalize
    end
end

println("NCCL Sub-communicator tests completed!")
