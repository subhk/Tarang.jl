using Test
using Tarang

@testset "Architecture namespace facades" begin
    @test isdefined(Tarang, :Fields)
    @test isdefined(Tarang, :Problems)
    @test isdefined(Tarang, :Solvers)
    @test isdefined(Tarang, :Timesteppers)
    @test isdefined(Tarang, :Transforms)
    @test isdefined(Tarang, :TransformOps)
    @test isdefined(Tarang, :Output)

    @test Tarang.Fields.ScalarField === Tarang.ScalarField
    @test Tarang.Fields.VectorField === Tarang.VectorField
    @test Tarang.Fields.PeriodicDomain === Tarang.PeriodicDomain

    @test Tarang.Problems.IVP === Tarang.IVP
    @test Tarang.Problems.add_equation! === Tarang.add_equation!
    @test Tarang.Problems.add_bc! === Tarang.add_bc!

    @test Tarang.Solvers.InitialValueSolver === Tarang.InitialValueSolver
    @test Tarang.Solvers.diagnose === Tarang.diagnose

    @test Tarang.Timesteppers.RK111 === Tarang.RK111
    @test Tarang.Timesteppers.CNAB2 === Tarang.CNAB2
    @test Tarang.Timesteppers.SpectralLinearOperator === Tarang.SpectralLinearOperator

    @test Tarang.TransformOps.forward_transform! === Tarang.forward_transform!
    @test Tarang.TransformOps.backward_transform! === Tarang.backward_transform!
    @test Tarang.Transforms !== Tarang.TransformOps

    @test Tarang.Output.NetCDFFileHandler === Tarang.NetCDFFileHandler
    @test Tarang.Output.NetCDFMerger === Tarang.NetCDFMerger
end
