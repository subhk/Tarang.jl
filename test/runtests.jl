using Test
using Tarang

# Test-file registry (TEST_FILES, OPTIONAL_TEST_FILES, GPU_TEST_FILES,
# MPI_TEST_FILES, DISTRIBUTED_GPU_TEST_FILES). Single source of truth, shared
# with test/run_mpi_ci.jl and guarded by test_test_inventory.jl.
include("file_lists.jl")

const RUN_OPTIONAL_TESTS = get(ENV, "TARANG_RUN_OPTIONAL_TESTS", "false") == "true"
const ONLY_OPTIONAL_TESTS = get(ENV, "TARANG_ONLY_OPTIONAL_TESTS", "false") == "true"
const RUN_GPU_TESTS = get(ENV, "TARANG_RUN_GPU_TESTS", "false") == "true"

if !ONLY_OPTIONAL_TESTS
    for file in TEST_FILES
        @testset "$file" begin
            include(file)
        end
    end
end

# Run optional tests if explicitly requested
if RUN_OPTIONAL_TESTS || ONLY_OPTIONAL_TESTS
    for file in OPTIONAL_TEST_FILES
        @testset "$file" begin
            include(file)
        end
    end
end

if RUN_GPU_TESTS
    for file in GPU_TEST_FILES
        @testset "$file" begin
            include(file)
        end
    end
end

# Print reminder about MPI tests
if !ONLY_OPTIONAL_TESTS
    @info """
    MPI tests are not included in the standard test suite.
    To run MPI tests with multiple processes, use:

        mpiexec -n 4 julia --project test/test_mpi_distributor.jl
        mpiexec -n 4 julia --project test/test_mpi_algebraic_constraints.jl
        mpiexec -n 4 julia --project test/test_mpi_lazy_rhs_fourier.jl
        mpiexec -n 4 julia --project test/test_distributed_gpu_transpose.jl

    Or use the test runner script:

        ./test/run_mpi_tests.sh 4
    """
end
