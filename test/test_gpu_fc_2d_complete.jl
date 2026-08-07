using Test
using Tarang

const _FC_REQUIRE_CUDA = lowercase(get(ENV, "TARANG_REQUIRE_CUDA", "false")) in
                         ("1", "true", "yes")
const _FC_HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !_FC_HAS_CUDA
    _FC_REQUIRE_CUDA && error(
        "2D Fourier--Chebyshev validation requires CUDA.jl and a functional CUDA device")
    @testset "Complete 2D Fourier--Chebyshev GPU path" begin
        @test_skip "CUDA not functional on this host"
    end
else
    CUDA.allowscalar(false)

    @testset "2D FC device axis resizing" begin
        source_host = reshape(collect(Float64, 1:54), 6, 9)
        source = CuArray(source_host)
        truncated = CUDA.zeros(Float64, 6, 5)

        @test Tarang._copy_axis_prefix!(truncated, source, 2) === truncated
        @test Array(truncated) == source_host[:, 1:5]

        restored = CUDA.fill(-1.0, 6, 9)
        @test Tarang._zero_pad_axis_prefix!(restored, truncated, 2) === restored
        @test Array(restored[:, 1:5]) == source_host[:, 1:5]
        @test iszero(Array(restored[:, 6:9]))

        complex_host = complex.(reshape(collect(Float64, 1:24), 6, 4),
                                reshape(collect(Float64, 25:48), 6, 4))
        complex_source = CuArray(complex_host)
        complex_truncated = CUDA.zeros(ComplexF64, 3, 4)
        Tarang._copy_axis_prefix!(complex_truncated, complex_source, 1)
        @test Array(complex_truncated) == complex_host[1:3, :]

        complex_restored = CUDA.fill(ComplexF64(-1, -1), 6, 4)
        Tarang._zero_pad_axis_prefix!(complex_restored, complex_truncated, 1)
        @test Array(complex_restored[1:3, :]) == complex_host[1:3, :]
        @test iszero(Array(complex_restored[4:6, :]))
    end
end
