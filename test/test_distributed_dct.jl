using Test
using MPI
using CUDA

if !MPI.Initialized()
    MPI.Init()
end

@testset "Distributed DCT" begin
    if CUDA.functional()
        using Tarang
        using Tarang: GPU

        # Import from TarangCUDAExt
        # Note: These are exported from the extension when CUDA is available
        try
            # Try to get exports from TarangCUDAExt
            @eval using TarangCUDAExt
        catch
            # Extension may be auto-loaded, try direct access
        end

        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nprocs = MPI.Comm_size(MPI.COMM_WORLD)

        @testset "DistributedDCTPlan creation" begin
            global_shape = (64, 64, 64)
            proc_grid = (1, max(1, nprocs))

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            plan = DistributedDCTPlan(pencil, Float64)

            @test plan.pencil === pencil
            @test length(plan.local_dct_plans) == 3
            @test length(plan.work_arrays) == 3

            # Verify work array shapes match pencil shapes
            @test size(plan.work_arrays[1]) == pencil.x_pencil_shape
            @test size(plan.work_arrays[2]) == pencil.y_pencil_shape
            @test size(plan.work_arrays[3]) == pencil.z_pencil_shape

            # Cleanup
            finalize_distributed_dct_plan!(plan)
        end

        @testset "Local DCT along dimension" begin
            # Test local_dct_along_dim! for single GPU
            N = 32
            data = CuArray(rand(Float64, N, N, N))
            output = similar(data)
            recovered = similar(data)

            arch = GPU()
            dct_plan = plan_optimized_gpu_dct(arch, N, Float64)

            # Test dimension 3 (Z)
            local_dct_along_dim!(output, data, dct_plan, 3, :forward)
            local_dct_along_dim!(recovered, output, dct_plan, 3, :backward)
            @test Array(recovered) ≈ Array(data) rtol=1e-10

            # Test dimension 2 (Y)
            local_dct_along_dim!(output, data, dct_plan, 2, :forward)
            local_dct_along_dim!(recovered, output, dct_plan, 2, :backward)
            @test Array(recovered) ≈ Array(data) rtol=1e-10

            # Test dimension 1 (X)
            local_dct_along_dim!(output, data, dct_plan, 1, :forward)
            local_dct_along_dim!(recovered, output, dct_plan, 1, :backward)
            @test Array(recovered) ≈ Array(data) rtol=1e-10
        end

        @testset "Single-rank distributed DCT round-trip" begin
            if nprocs == 1
                global_shape = (32, 32, 32)
                proc_grid = (1, 1)

                pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
                plan = DistributedDCTPlan(pencil, Float64)

                # Create test data in Z-pencil
                data = CuArray(rand(Float64, pencil.z_pencil_shape...))
                original = copy(data)

                # Forward DCT
                coeffs = CUDA.zeros(Float64, pencil.x_pencil_shape...)
                distributed_forward_dct!(coeffs, data, plan)

                # For single rank, reset orientation for backward
                set_orientation!(pencil, :x_pencil)

                # Backward DCT
                recovered = CUDA.zeros(Float64, pencil.z_pencil_shape...)
                distributed_backward_dct!(recovered, coeffs, plan)

                # Verify round-trip accuracy
                @test Array(recovered) ≈ Array(original) rtol=1e-10

                # Cleanup
                finalize_distributed_dct_plan!(plan)
            else
                @info "Skipping single-rank test (running with $nprocs processes)"
            end
        end

        @testset "Non-square domain" begin
            if nprocs == 1
                global_shape = (16, 32, 48)
                proc_grid = (1, 1)

                pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
                plan = DistributedDCTPlan(pencil, Float64)

                # Create test data
                data = CuArray(rand(Float64, pencil.z_pencil_shape...))
                original = copy(data)

                # Forward DCT
                coeffs = CUDA.zeros(Float64, pencil.x_pencil_shape...)
                distributed_forward_dct!(coeffs, data, plan)

                # Reset orientation for backward
                set_orientation!(pencil, :x_pencil)

                # Backward DCT
                recovered = CUDA.zeros(Float64, pencil.z_pencil_shape...)
                distributed_backward_dct!(recovered, coeffs, plan)

                # Verify round-trip accuracy
                @test Array(recovered) ≈ Array(original) rtol=1e-10

                # Cleanup
                finalize_distributed_dct_plan!(plan)
            else
                @info "Skipping non-square domain test (running with $nprocs processes)"
            end
        end

        @testset "Float32 precision" begin
            if nprocs == 1
                global_shape = (32, 32, 32)
                proc_grid = (1, 1)

                pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
                plan = DistributedDCTPlan(pencil, Float32)

                # Create test data
                data = CuArray(rand(Float32, pencil.z_pencil_shape...))
                original = copy(data)

                # Forward DCT
                coeffs = CUDA.zeros(Float32, pencil.x_pencil_shape...)
                distributed_forward_dct!(coeffs, data, plan)

                # Reset orientation for backward
                set_orientation!(pencil, :x_pencil)

                # Backward DCT
                recovered = CUDA.zeros(Float32, pencil.z_pencil_shape...)
                distributed_backward_dct!(recovered, coeffs, plan)

                # Float32 has less precision
                @test Array(recovered) ≈ Array(original) rtol=1e-5

                # Cleanup
                finalize_distributed_dct_plan!(plan)
            else
                @info "Skipping Float32 test (running with $nprocs processes)"
            end
        end

        @testset "Plan cleanup" begin
            global_shape = (32, 32, 32)
            proc_grid = (1, max(1, nprocs))

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            plan = DistributedDCTPlan(pencil, Float64)

            # Cleanup should not error
            finalize_distributed_dct_plan!(plan)
            @test true  # If we get here, cleanup worked
        end

        @testset "Multi-rank basic test" begin
            if nprocs > 1
                # Use a domain size that divides evenly by nprocs
                Nz = nprocs * 16
                global_shape = (32, 32, Nz)
                proc_grid = (1, nprocs)

                pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
                plan = DistributedDCTPlan(pencil, Float64)

                # Verify pencil shapes are computed correctly
                @test pencil.z_pencil_shape[3] == Nz  # Z is full in Z-pencil
                @test pencil.y_pencil_shape[2] == global_shape[2]  # Y is full in Y-pencil
                @test pencil.x_pencil_shape[1] == global_shape[1]  # X is full in X-pencil

                # Verify work arrays match pencil shapes
                @test size(plan.work_arrays[1]) == pencil.x_pencil_shape
                @test size(plan.work_arrays[2]) == pencil.y_pencil_shape
                @test size(plan.work_arrays[3]) == pencil.z_pencil_shape

                # Cleanup
                finalize_distributed_dct_plan!(plan)

                @info "Multi-rank test passed on rank $rank"
            else
                @info "Skipping multi-rank test (running with single process)"
            end
        end

    else
        @info "CUDA not available, skipping distributed DCT tests"
    end
end
