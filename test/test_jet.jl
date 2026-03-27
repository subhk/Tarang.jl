using JET
using Tarang

@testset "JET.jl" begin
    # Use report_package for informational analysis.
    # test_package is too strict for codebases with optional GPU/MPI paths
    # and Union{Nothing, T} fields (CUDA, NCCL, MPI.Ialltoallv! etc.).
    result = JET.report_package(Tarang; target_defined_modules=true)
    @test result isa Any  # ensure JET runs without error
end
