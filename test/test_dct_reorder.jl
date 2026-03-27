#!/usr/bin/env julia
"""
    DCT Reorder Kernels Tests for Tarang.jl

Tests for the even-odd interleaving reorder operations used in memory-efficient DCT.

The reorder pattern:
- work[k] = x[2k] for k=0..N/2-1 (even indices to first half)
- work[N-1-k] = x[2k+1] for k=0..N/2-1 (odd indices reversed to second half)

Example: [1,2,3,4,5,6,7,8] -> [1,3,5,7,8,6,4,2]

Run with:
    julia test/test_dct_reorder.jl
"""

using Test
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Tolerance levels for numerical comparison
const RTOL = 1e-12
const ATOL = 1e-12

println("="^70)
println("DCT REORDER KERNELS TESTS FOR TARANG.jl")
println("="^70)
println()

# ============================================================================
# Check CUDA availability
# ============================================================================

println("[0] Checking CUDA availability...")
try
    using CUDA
    if !CUDA.functional()
        println("    CUDA not functional - skipping GPU tests")
        println("    Exiting.")
        exit(0)
    end
    println("    CUDA is functional")
    println("    Device: $(CUDA.name(CUDA.device()))")
catch e
    println("    CUDA not available: $e")
    exit(0)
end

# Load Tarang after CUDA
using Tarang

println()

# ============================================================================
# Reference implementations for testing
# ============================================================================

"""
    reorder_for_dct_reference(x::Vector{T}) where T

CPU reference implementation of the even-odd reorder for DCT.
- work[k+1] = x[2k+1] for k=0..N/2-1 (even indices to first half, 1-indexed)
- work[N-k] = x[2k+2] for k=0..N/2-1 (odd indices reversed to second half, 1-indexed)
"""
function reorder_for_dct_reference(x::Vector{T}) where T
    N = length(x)
    @assert iseven(N) "N must be even"
    work = similar(x)

    for k in 0:(N÷2 - 1)
        # Even indices (0, 2, 4, ...) go to first half
        work[k + 1] = x[2k + 1]  # 1-indexed: work[1..N/2] = x[1,3,5,...]
        # Odd indices (1, 3, 5, ...) go to second half reversed
        work[N - k] = x[2k + 2]  # 1-indexed: work[N..N/2+1] = x[2,4,6,...]
    end

    return work
end

"""
    inverse_reorder_for_dct_reference(work::Vector{T}) where T

CPU reference implementation of the inverse even-odd reorder for DCT.
"""
function inverse_reorder_for_dct_reference(work::Vector{T}) where T
    N = length(work)
    @assert iseven(N) "N must be even"
    x = similar(work)

    for k in 0:(N÷2 - 1)
        # Reconstruct even indices from first half
        x[2k + 1] = work[k + 1]  # x[1,3,5,...] = work[1..N/2]
        # Reconstruct odd indices from second half (reversed)
        x[2k + 2] = work[N - k]  # x[2,4,6,...] = work[N..N/2+1]
    end

    return x
end

"""
    reorder_for_dct_3d_reference(x::Array{T,3}, dim::Int) where T

CPU reference implementation of 3D reorder along specified dimension.
"""
function reorder_for_dct_3d_reference(x::Array{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(x)
    work = similar(x)

    if dim == 1
        @assert iseven(Nx) "Nx must be even for dim=1"
        for j in 1:Ny, k in 1:Nz
            fiber = x[:, j, k]
            reordered = reorder_for_dct_reference(fiber)
            work[:, j, k] = reordered
        end
    elseif dim == 2
        @assert iseven(Ny) "Ny must be even for dim=2"
        for i in 1:Nx, k in 1:Nz
            fiber = x[i, :, k]
            reordered = reorder_for_dct_reference(fiber)
            work[i, :, k] = reordered
        end
    else  # dim == 3
        @assert iseven(Nz) "Nz must be even for dim=3"
        for i in 1:Nx, j in 1:Ny
            fiber = x[i, j, :]
            reordered = reorder_for_dct_reference(fiber)
            work[i, j, :] = reordered
        end
    end

    return work
end

"""
    inverse_reorder_for_dct_3d_reference(work::Array{T,3}, dim::Int) where T

CPU reference implementation of 3D inverse reorder along specified dimension.
"""
function inverse_reorder_for_dct_3d_reference(work::Array{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(work)
    x = similar(work)

    if dim == 1
        @assert iseven(Nx) "Nx must be even for dim=1"
        for j in 1:Ny, k in 1:Nz
            fiber = work[:, j, k]
            original = inverse_reorder_for_dct_reference(fiber)
            x[:, j, k] = original
        end
    elseif dim == 2
        @assert iseven(Ny) "Ny must be even for dim=2"
        for i in 1:Nx, k in 1:Nz
            fiber = work[i, :, k]
            original = inverse_reorder_for_dct_reference(fiber)
            x[i, :, k] = original
        end
    else  # dim == 3
        @assert iseven(Nz) "Nz must be even for dim=3"
        for i in 1:Nx, j in 1:Ny
            fiber = work[i, j, :]
            original = inverse_reorder_for_dct_reference(fiber)
            x[i, j, :] = original
        end
    end

    return x
end

# ============================================================================
# TEST 1: 1D Reorder Pattern Test
# ============================================================================

@testset "1D DCT Reorder Pattern" begin
    println("\n[1] Testing 1D DCT Reorder Pattern...")

    # Test the specific pattern: [1,2,3,4,5,6,7,8] -> [1,3,5,7,8,6,4,2]
    x = Float64[1, 2, 3, 4, 5, 6, 7, 8]
    expected = Float64[1, 3, 5, 7, 8, 6, 4, 2]

    # CPU reference test
    result_cpu = reorder_for_dct_reference(x)
    @test result_cpu == expected
    println("    CPU reference pattern: PASS")

    # GPU test
    x_gpu = CuArray(x)
    work_gpu = CUDA.zeros(Float64, length(x))

    TarangCUDAExt.reorder_for_dct!(work_gpu, x_gpu)
    result_gpu = Array(work_gpu)

    @test result_gpu == expected
    println("    GPU kernel pattern: PASS")

    # Test inverse reorder recovers original
    inverse_expected = x
    inverse_cpu = inverse_reorder_for_dct_reference(expected)
    @test inverse_cpu == inverse_expected
    println("    CPU inverse pattern: PASS")

    # GPU inverse test
    recovered_gpu = CUDA.zeros(Float64, length(x))
    TarangCUDAExt.inverse_reorder_for_dct!(recovered_gpu, work_gpu)
    result_inverse_gpu = Array(recovered_gpu)

    @test result_inverse_gpu == inverse_expected
    println("    GPU inverse kernel pattern: PASS")
end

# ============================================================================
# TEST 2: 1D Reorder Round-trip Test
# ============================================================================

@testset "1D DCT Reorder Round-trip" begin
    println("\n[2] Testing 1D DCT Reorder Round-trip...")

    test_sizes = [8, 16, 32, 64, 128, 256]

    for N in test_sizes
        x_cpu = rand(Float64, N)
        x_gpu = CuArray(x_cpu)
        work_gpu = CUDA.zeros(Float64, N)
        recovered_gpu = CUDA.zeros(Float64, N)

        # Forward reorder
        TarangCUDAExt.reorder_for_dct!(work_gpu, x_gpu)

        # Inverse reorder
        TarangCUDAExt.inverse_reorder_for_dct!(recovered_gpu, work_gpu)

        result = Array(recovered_gpu)

        passed = isapprox(result, x_cpu; rtol=RTOL, atol=ATOL)
        @test passed

        if passed
            println("    N=$N: PASS")
        else
            max_diff = maximum(abs.(result .- x_cpu))
            println("    N=$N: FAIL (max diff: $max_diff)")
        end
    end
end

# ============================================================================
# TEST 3: 1D Reorder GPU vs CPU Reference
# ============================================================================

@testset "1D DCT Reorder GPU vs CPU" begin
    println("\n[3] Testing 1D DCT Reorder GPU vs CPU Reference...")

    test_sizes = [8, 16, 32, 64, 128]

    for N in test_sizes
        x_cpu = rand(Float64, N)
        x_gpu = CuArray(x_cpu)
        work_gpu = CUDA.zeros(Float64, N)

        # GPU forward reorder
        TarangCUDAExt.reorder_for_dct!(work_gpu, x_gpu)
        result_gpu = Array(work_gpu)

        # CPU reference forward reorder
        expected = reorder_for_dct_reference(x_cpu)

        passed = isapprox(result_gpu, expected; rtol=RTOL, atol=ATOL)
        @test passed

        if passed
            println("    Forward N=$N: PASS")
        else
            max_diff = maximum(abs.(result_gpu .- expected))
            println("    Forward N=$N: FAIL (max diff: $max_diff)")
        end

        # GPU inverse reorder
        recovered_gpu = CUDA.zeros(Float64, N)
        TarangCUDAExt.inverse_reorder_for_dct!(recovered_gpu, work_gpu)
        result_inverse_gpu = Array(recovered_gpu)

        # CPU reference inverse reorder
        expected_inverse = inverse_reorder_for_dct_reference(expected)

        passed_inverse = isapprox(result_inverse_gpu, expected_inverse; rtol=RTOL, atol=ATOL)
        @test passed_inverse

        if passed_inverse
            println("    Inverse N=$N: PASS")
        else
            max_diff = maximum(abs.(result_inverse_gpu .- expected_inverse))
            println("    Inverse N=$N: FAIL (max diff: $max_diff)")
        end
    end
end

# ============================================================================
# TEST 4: 3D Reorder Along Dimension Tests
# ============================================================================

@testset "3D DCT Reorder Along Dimensions" begin
    println("\n[4] Testing 3D DCT Reorder Along Dimensions...")

    test_sizes = [(8, 6, 4), (16, 12, 8), (8, 8, 8)]

    for (Nx, Ny, Nz) in test_sizes
        x_cpu = rand(Float64, Nx, Ny, Nz)
        x_gpu = CuArray(x_cpu)

        for dim in 1:3
            # Skip if dimension size is odd
            dim_size = (Nx, Ny, Nz)[dim]
            if isodd(dim_size)
                println("    Size ($Nx,$Ny,$Nz) dim=$dim: SKIPPED (odd size)")
                continue
            end

            work_gpu = CUDA.zeros(Float64, Nx, Ny, Nz)
            recovered_gpu = CUDA.zeros(Float64, Nx, Ny, Nz)

            # GPU forward reorder
            TarangCUDAExt.reorder_for_dct_dim!(work_gpu, x_gpu, dim)
            result_forward_gpu = Array(work_gpu)

            # CPU reference forward reorder
            expected_forward = reorder_for_dct_3d_reference(x_cpu, dim)

            passed_forward = isapprox(result_forward_gpu, expected_forward; rtol=RTOL, atol=ATOL)
            @test passed_forward

            if passed_forward
                println("    Size ($Nx,$Ny,$Nz) dim=$dim forward: PASS")
            else
                max_diff = maximum(abs.(result_forward_gpu .- expected_forward))
                println("    Size ($Nx,$Ny,$Nz) dim=$dim forward: FAIL (max diff: $max_diff)")
            end

            # GPU inverse reorder
            TarangCUDAExt.inverse_reorder_for_dct_dim!(recovered_gpu, work_gpu, dim)
            result_inverse_gpu = Array(recovered_gpu)

            # CPU reference inverse reorder
            expected_inverse = inverse_reorder_for_dct_3d_reference(expected_forward, dim)

            passed_inverse = isapprox(result_inverse_gpu, expected_inverse; rtol=RTOL, atol=ATOL)
            @test passed_inverse

            if passed_inverse
                println("    Size ($Nx,$Ny,$Nz) dim=$dim inverse: PASS")
            else
                max_diff = maximum(abs.(result_inverse_gpu .- expected_inverse))
                println("    Size ($Nx,$Ny,$Nz) dim=$dim inverse: FAIL (max diff: $max_diff)")
            end

            # Round-trip test
            passed_roundtrip = isapprox(result_inverse_gpu, x_cpu; rtol=RTOL, atol=ATOL)
            @test passed_roundtrip

            if passed_roundtrip
                println("    Size ($Nx,$Ny,$Nz) dim=$dim round-trip: PASS")
            else
                max_diff = maximum(abs.(result_inverse_gpu .- x_cpu))
                println("    Size ($Nx,$Ny,$Nz) dim=$dim round-trip: FAIL (max diff: $max_diff)")
            end
        end
    end
end

# ============================================================================
# TEST 5: Float32 Support
# ============================================================================

@testset "Float32 Support" begin
    println("\n[5] Testing Float32 Support...")

    # 1D Float32 test
    N = 64
    x_cpu = rand(Float32, N)
    x_gpu = CuArray(x_cpu)
    work_gpu = CUDA.zeros(Float32, N)
    recovered_gpu = CUDA.zeros(Float32, N)

    TarangCUDAExt.reorder_for_dct!(work_gpu, x_gpu)
    TarangCUDAExt.inverse_reorder_for_dct!(recovered_gpu, work_gpu)

    result = Array(recovered_gpu)
    passed_1d = isapprox(result, x_cpu; rtol=1e-5, atol=1e-5)
    @test passed_1d
    println("    1D Float32 round-trip: $(passed_1d ? "PASS" : "FAIL")")

    # 3D Float32 test
    Nx, Ny, Nz = 8, 8, 8
    x3d_cpu = rand(Float32, Nx, Ny, Nz)
    x3d_gpu = CuArray(x3d_cpu)

    for dim in 1:3
        work3d_gpu = CUDA.zeros(Float32, Nx, Ny, Nz)
        recovered3d_gpu = CUDA.zeros(Float32, Nx, Ny, Nz)

        TarangCUDAExt.reorder_for_dct_dim!(work3d_gpu, x3d_gpu, dim)
        TarangCUDAExt.inverse_reorder_for_dct_dim!(recovered3d_gpu, work3d_gpu, dim)

        result3d = Array(recovered3d_gpu)
        passed_3d = isapprox(result3d, x3d_cpu; rtol=1e-5, atol=1e-5)
        @test passed_3d
        println("    3D Float32 dim=$dim round-trip: $(passed_3d ? "PASS" : "FAIL")")
    end
end

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^70)
println("DCT REORDER TESTS COMPLETED")
println("="^70)
