using Test

@testset "Test file inventory" begin
    all_test_files = Set(filter(name -> startswith(name, "test_") && endswith(name, ".jl"),
                                readdir(@__DIR__)))

    known_test_files = Set(vcat(TEST_FILES,
                                OPTIONAL_TEST_FILES,
                                GPU_TEST_FILES,
                                MPI_TEST_FILES,
                                DISTRIBUTED_GPU_TEST_FILES))

    @test isempty(setdiff(all_test_files, known_test_files))
    @test isempty(setdiff(known_test_files, all_test_files))

    # The documented shell entry point must use the same registry-backed driver
    # as CI; a hand-maintained list here previously ran only 7/57 MPI files.
    mpi_wrapper = read(joinpath(@__DIR__, "run_mpi_tests.sh"), String)
    @test occursin("run_mpi_ci.jl", mpi_wrapper)
    @test !occursin("MPI_TESTS=(", mpi_wrapper)
end
