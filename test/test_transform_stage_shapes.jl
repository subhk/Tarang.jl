using Test
using Tarang

@testset "Mixed transform stage shapes" begin
    @testset "Chebyshev first with both axes scaled" begin
        coords = CartesianCoordinates("z", "x")
        bases = (
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
        )

        ops, coeff_shape, _ = Tarang.forward_layout(bases, (14, 12), Float64)
        stages = Tarang.transform_stage_shapes(ops, (14, 12), (2, 1))

        @test coeff_shape == (9, 7)
        @test stages == [(14, 12), (14, 7), (9, 7)]
        @test reverse(stages) == [(9, 7), (14, 7), (14, 12)]
    end

    @testset "Fourier first and complex Fourier preserve the declared order" begin
        coords = CartesianCoordinates("x", "z")
        real_bases = (
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
        )
        real_ops, _, _ = Tarang.forward_layout(real_bases, (12, 14), Float64)
        @test Tarang.transform_stage_shapes(real_ops, (12, 14), (1, 2)) ==
              [(12, 14), (7, 14), (7, 9)]

        complex_bases = (
            ComplexFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
        )
        complex_ops, _, _ = Tarang.forward_layout(complex_bases, (12, 14), ComplexF64)
        @test Tarang.transform_stage_shapes(complex_ops, (12, 14), (1, 2)) ==
              [(12, 14), (12, 14), (12, 9)]

        reverse_coords = CartesianCoordinates("z", "x")
        reverse_complex_bases = (
            ChebyshevT(reverse_coords["z"]; size=9, bounds=(0.0, 1.0)),
            ComplexFourier(reverse_coords["x"]; size=8, bounds=(0.0, 2π)),
        )
        reverse_ops, _, _ = Tarang.forward_layout(
            reverse_complex_bases, (14, 12), ComplexF64)
        @test Tarang.transform_stage_shapes(reverse_ops, (14, 12), (2, 1)) ==
              [(14, 12), (14, 12), (9, 12)]
    end

    @testset "unscaled and malformed orders" begin
        coords = CartesianCoordinates("x", "z")
        bases = (
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
        )
        ops, _, _ = Tarang.forward_layout(bases, (8, 9), Float64)
        @test Tarang.transform_stage_shapes(ops, (8, 9), (1, 2)) ==
              [(8, 9), (5, 9), (5, 9)]

        @test_throws ArgumentError Tarang.transform_stage_shapes(ops, (8, 9), (1, 1))
        @test_throws ArgumentError Tarang.transform_stage_shapes(ops, (8, 9), (1,))
        @test_throws DimensionMismatch Tarang.transform_stage_shapes(ops, (8,), (1, 2))
    end
end

@testset "Transform axis prefix resizing" begin
    @testset "truncate and restore axis 2" begin
        source = reshape(collect(1:54), 6, 9)
        truncated = zeros(Int, 6, 5)

        @test Tarang._copy_axis_prefix!(truncated, source, 2) === truncated
        @test truncated == source[:, 1:5]

        restored = fill(-1, 6, 9)
        @test Tarang._zero_pad_axis_prefix!(restored, truncated, 2) === restored
        @test restored[:, 1:5] == truncated
        @test iszero(restored[:, 6:9])
    end

    @testset "complex data along axis 1" begin
        source = complex.(reshape(collect(1:24), 6, 4),
                          reshape(collect(25:48), 6, 4))
        truncated = zeros(Complex{Int}, 3, 4)
        Tarang._copy_axis_prefix!(truncated, source, 1)
        @test truncated == source[1:3, :]

        restored = fill(Complex{Int}(-1, -1), 6, 4)
        Tarang._zero_pad_axis_prefix!(restored, truncated, 1)
        @test restored[1:3, :] == truncated
        @test iszero(restored[4:6, :])
    end

    @testset "invalid resize shapes are rejected" begin
        @test_throws ArgumentError Tarang._copy_axis_prefix!(zeros(5, 4), zeros(3, 4), 1)
        @test_throws ArgumentError Tarang._zero_pad_axis_prefix!(zeros(3, 4), zeros(5, 4), 1)
        @test_throws DimensionMismatch Tarang._copy_axis_prefix!(zeros(3, 5), zeros(6, 4), 1)
        @test_throws DimensionMismatch Tarang._zero_pad_axis_prefix!(zeros(6, 5), zeros(3, 4), 1)
        @test_throws ArgumentError Tarang._copy_axis_prefix!(zeros(3, 4), zeros(6, 4), 3)
    end
end
