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
