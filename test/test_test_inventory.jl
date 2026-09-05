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

    # A registered file that exists on disk but is not tracked by git passes
    # locally and fails on every clean clone. It has happened three times.
    tracked = try
        Set(basename.(filter(!isempty, split(read(
            setenv(`git ls-files -- test/`; dir=joinpath(@__DIR__, "..")),
            String), '\n'))))
    catch err
        @info "git unavailable; skipping tracked-file check" exception = err
        nothing
    end
    if tracked !== nothing
        untracked = sort!(collect(setdiff(known_test_files, tracked)))
        isempty(untracked) || @error "registered test files not tracked by git" untracked
        @test isempty(untracked)
    end
end
