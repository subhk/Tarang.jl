# ============================================================================
# Tests for Distributed GPU Transform Dispatch
# ============================================================================
#
# This test verifies that gpu_forward_transform! and gpu_backward_transform!
# correctly dispatch to distributed GPU transforms when:
# 1. Running on GPU architecture
# 2. Multiple MPI processes are available
# 3. NCCL is available
# 4. The field has Chebyshev bases (needs DCT)

using Test
using MPI

if !MPI.Initialized()
    MPI.Init()
end

@testset "Distributed Transform Dispatch" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    # Check if CUDA is available
    cuda_available = try
        using CUDA
        CUDA.functional()
    catch
        false
    end

    if cuda_available
        using Tarang
        using CUDA

        # Load the CUDA extension
        # The extension is loaded automatically when CUDA is available
        # We need to access functions through the module
        TarangCUDAExt = Base.get_extension(Tarang, :TarangCUDAExt)

        if TarangCUDAExt !== nothing
            @testset "is_distributed_gpu detection" begin
                arch = GPU()

                if nprocs > 1
                    # When multiple processes and NCCL is available, should return true
                    result = TarangCUDAExt.is_distributed_gpu(arch, nprocs)
                    @test result == Tarang.nccl_available()
                else
                    # Single process should always return false
                    @test TarangCUDAExt.is_distributed_gpu(arch, 1) == false
                end

                # CPU architecture should always return false
                @test TarangCUDAExt.is_distributed_gpu(CPU(), nprocs) == false
                @test TarangCUDAExt.is_distributed_gpu(CPU(), 1) == false
            end

            @testset "needs_distributed_dct detection" begin
                # Verify the function exists
                @test isdefined(TarangCUDAExt, :needs_distributed_dct)

                # Create a mock-like test using the actual type system
                # We test that the function properly checks for Chebyshev bases

                # For a complete test, we would need to create actual fields
                # which requires full domain/distributor setup
                # For now, test that the function exists and is callable
                @test hasmethod(TarangCUDAExt.needs_distributed_dct, Tuple{Any})
            end

            @testset "get_or_create_pencil" begin
                # Test pencil creation helper exists
                @test isdefined(TarangCUDAExt, :get_or_create_pencil)
            end

            @testset "distributed_gpu_forward_transform! exists" begin
                @test isdefined(TarangCUDAExt, :distributed_gpu_forward_transform!)
            end

            @testset "distributed_gpu_backward_transform! exists" begin
                @test isdefined(TarangCUDAExt, :distributed_gpu_backward_transform!)
            end

            @testset "get_or_create_distributed_dct_plan exists" begin
                @test isdefined(TarangCUDAExt, :get_or_create_distributed_dct_plan)
            end

            @testset "clear_distributed_dct_plan_cache! exists" begin
                @test isdefined(TarangCUDAExt, :clear_distributed_dct_plan_cache!)
                # Test that it's callable without error
                @test_nowarn TarangCUDAExt.clear_distributed_dct_plan_cache!()
            end

            @testset "_compute_proc_grid" begin
                # Test process grid computation
                @test TarangCUDAExt._compute_proc_grid(1) == (1, 1)
                @test TarangCUDAExt._compute_proc_grid(4) == (2, 2)
                @test TarangCUDAExt._compute_proc_grid(6) == (2, 3)
                @test TarangCUDAExt._compute_proc_grid(8) == (2, 4)
                @test TarangCUDAExt._compute_proc_grid(9) == (3, 3)
                @test TarangCUDAExt._compute_proc_grid(16) == (4, 4)

                # Verify all process grid products equal input
                for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32]
                    pg = TarangCUDAExt._compute_proc_grid(n)
                    @test pg[1] * pg[2] == n
                end
            end

            if nprocs > 1 && Tarang.nccl_available()
                @testset "Distributed GPU dispatch integration" begin
                    # This test requires actual multi-GPU setup with NCCL
                    # Only run if we have multiple processes and NCCL

                    # Create a 3D Chebyshev domain for testing
                    coords = CartesianCoordinates("x", "y", "z")
                    dist = Distributor(coords; dtype=Float64, architecture=GPU())

                    # Create Chebyshev bases for all dimensions
                    Nx, Ny, Nz = 32, 32, 32
                    xb = ChebyshevT(coords["x"]; size=Nx, bounds=(0.0, 1.0))
                    yb = ChebyshevT(coords["y"]; size=Ny, bounds=(0.0, 1.0))
                    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))

                    # Create field
                    field = ScalarField(dist, "test", (xb, yb, zb), Float64)

                    # Verify needs_distributed_dct returns true for this field
                    @test TarangCUDAExt.needs_distributed_dct(field) == true

                    # Verify is_distributed_gpu returns true
                    @test TarangCUDAExt.is_distributed_gpu(GPU(), nprocs) == true

                    # Initialize with some data
                    data = get_grid_data(field)
                    if data !== nothing
                        fill!(data, Float64(rank + 1))
                    end

                    # Test forward transform
                    @test_nowarn Tarang.gpu_forward_transform!(field)

                    # Test backward transform
                    @test_nowarn Tarang.gpu_backward_transform!(field)

                    # Clean up
                    TarangCUDAExt.clear_distributed_dct_plan_cache!()
                end
            else
                @info "Skipping distributed GPU dispatch integration test" nprocs=nprocs nccl_available=Tarang.nccl_available()
            end
        else
            @info "TarangCUDAExt not loaded, skipping dispatch tests"
        end
    else
        @info "CUDA not available, skipping distributed dispatch tests"
    end
end

# MPI barrier to ensure all processes complete
MPI.Barrier(MPI.COMM_WORLD)
