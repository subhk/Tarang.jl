using Test
using CUDA

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "Optimized DCT" begin
        @testset "1D optimized DCT vs reference" begin
            N = 64
            x = CuArray(rand(Float64, N))

            # Reference: existing FFT-based DCT (2N symmetric extension)
            arch = GPU()
            ref_plan = plan_gpu_dct(arch, N, Float64, 1)
            ref_output = similar(x)
            gpu_forward_dct_1d!(ref_output, x, ref_plan)

            # Optimized: R2C FFT with reordering
            opt_plan = plan_optimized_gpu_dct(arch, N, Float64)
            opt_output = similar(x)
            optimized_forward_dct_1d!(opt_output, x, opt_plan)

            @test Array(opt_output) ≈ Array(ref_output) rtol=1e-12
        end

        @testset "1D optimized DCT round-trip" begin
            N = 128
            x = CuArray(rand(Float64, N))

            arch = GPU()
            plan = plan_optimized_gpu_dct(arch, N, Float64)

            # Forward then backward should recover original
            coeffs = similar(x)
            recovered = similar(x)

            optimized_forward_dct_1d!(coeffs, x, plan)
            optimized_backward_dct_1d!(recovered, coeffs, plan)

            @test Array(recovered) ≈ Array(x) rtol=1e-11
        end

        @testset "Memory usage is smaller" begin
            N = 512
            arch = GPU()

            # Reference plan uses 2N complex work array
            ref_plan = plan_gpu_dct(arch, N, Float64, 1)
            ref_work_size = length(ref_plan.work_complex) * sizeof(ComplexF64)

            # Optimized plan uses N real + N/2+1 complex
            opt_plan = plan_optimized_gpu_dct(arch, N, Float64)
            opt_work_size = length(opt_plan.work_real) * sizeof(Float64) +
                           length(opt_plan.work_complex) * sizeof(ComplexF64)

            # Optimized should use less memory
            @test opt_work_size < ref_work_size
            @test opt_work_size / ref_work_size < 0.7
        end

        @testset "Various sizes" begin
            arch = GPU()
            for N in [32, 64, 128, 256]
                x = CuArray(rand(Float64, N))
                plan = plan_optimized_gpu_dct(arch, N, Float64)

                coeffs = similar(x)
                recovered = similar(x)

                optimized_forward_dct_1d!(coeffs, x, plan)
                optimized_backward_dct_1d!(recovered, coeffs, plan)

                @test Array(recovered) ≈ Array(x) rtol=1e-10
            end
        end

        @testset "Float32 support" begin
            N = 64
            x = CuArray(rand(Float32, N))

            arch = GPU()
            plan = plan_optimized_gpu_dct(arch, N, Float32)

            coeffs = similar(x)
            recovered = similar(x)

            optimized_forward_dct_1d!(coeffs, x, plan)
            optimized_backward_dct_1d!(recovered, coeffs, plan)

            @test Array(recovered) ≈ Array(x) rtol=1e-5
        end
    end
else
    @warn "CUDA not available, skipping optimized DCT tests"
end
