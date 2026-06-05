"""
Tests for the O(N log N) FFT-based DCT (`transform_fft_dct.jl`) that replaces the
O(N²) cos-sum used by the GPU multi-dimensional DCT. Device-agnostic, so the
ALGORITHM is fully validated here on the CPU; the GPU runs the identical code via
CUFFT (and `test_gpu_transform_correctness.jl` checks GPU==CPU end-to-end).

Oracle = Tarang's exact DCT convention computed directly (the cos-sum the GPU
kernel implements): forward X[k] = scale_k Σ x_n cos(πk(2n+1)/2N), scale_0=1/2N,
scale_{k≥1}=1/N; backward x_n = 2 Σ_k X_k cos(πk(2n+1)/2N).
"""

using Test
using Tarang
using LinearAlgebra

# O(N²) reference along a dimension (Tarang convention).
function _cossum_forward(x::AbstractArray, dim::Int)
    mapslices(x; dims=dim) do v
        N = length(v)
        [(m == 1 ? 1/(2N) : 1/N) * sum(v[i] * cos(pi * (m-1) * (2*(i-1)+1) / (2N)) for i in 1:N) for m in 1:N]
    end
end
function _cossum_backward(X::AbstractArray, dim::Int)
    mapslices(X; dims=dim) do c
        N = length(c)
        [2 * sum(c[k] * cos(pi * (k-1) * (2*(n-1)+1) / (2N)) for k in 1:N) for n in 1:N]
    end
end

@testset "FFT-based DCT (O(N log N), device-agnostic)" begin
    fwd = Tarang.fft_dct_forward_dim
    bwd = Tarang.fft_dct_backward_dim

    @testset "1D forward matches cos-sum + round-trip" begin
        for N in (4, 8, 16, 31, 64)
            x = randn(N)
            @test isapprox(fwd(x, 1), vec(_cossum_forward(x, 1)); rtol=1e-10, atol=1e-12)
            @test isapprox(bwd(fwd(x, 1), 1), x; rtol=1e-10, atol=1e-12)              # round-trip
            @test isapprox(bwd(x, 1), vec(_cossum_backward(x, 1)); rtol=1e-10, atol=1e-12)
        end
    end

    @testset "2D along each dim" begin
        x = randn(8, 6)
        for d in 1:2
            @test isapprox(fwd(x, d), _cossum_forward(x, d); rtol=1e-10, atol=1e-12)
            @test isapprox(bwd(fwd(x, d), d), x; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "3D along each dim" begin
        x = randn(8, 6, 4)
        for d in 1:3
            @test isapprox(fwd(x, d), _cossum_forward(x, d); rtol=1e-10, atol=1e-12)
            @test isapprox(bwd(fwd(x, d), d), x; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "known small case (constant → DC only; inverse recovers)" begin
        c = fill(2.0, 8)
        X = fwd(c, 1)
        @test isapprox(X[1], 1.0; atol=1e-12)                 # X_0 = Σc/(2N) = c/2 = 1.0
        @test all(isapprox.(X[2:end], 0.0; atol=1e-12))       # no other modes
        @test isapprox(bwd(X, 1), c; rtol=1e-12)              # x_n = 2·X_0 = 2.0 = c
    end
end
