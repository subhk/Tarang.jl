using Test
using Tarang

const TEST_FILES = [
    "test_aqua.jl",
    "test_jet.jl",
    "test_coords.jl",
    "test_basis.jl",
    "test_architectures.jl",
    "test_config.jl",
    "test_fields.jl",
    "test_problems.jl",
    "test_operators_basic.jl",
    "test_transforms_extended.jl",
    "test_cfl.jl",
    "test_domain_metadata.jl",
    "test_subsystems.jl",
    "test_transforms.jl",
    "test_solvers.jl",
    "test_flow_tools.jl",
    "test_quick_domains.jl",
    "test_plot_tools.jl",
    "test_compatibility.jl",
    "test_cartesian_operators.jl",
    "test_stochastic_forcing.jl",
    "test_temporal_filters.jl",
    "test_arithmetic.jl",
    "test_distributor.jl",
    "test_les_models.jl",
    "test_chebyshev.jl",
    "test_domain.jl",
    "test_boundary_conditions.jl",
    "test_nonlinear.jl",
    "test_general.jl",
    "test_dedalus_features.jl",
    "test_analysis_tasks.jl",
    "test_phi_functions.jl",
    "test_evaluator.jl",
    "test_parsing.jl",
    "test_operators_tensor.jl",
    "test_diagonal_imex.jl",
    "test_rksmr_convergence.jl",
    "test_kernel_operations.jl",
    "test_cpu_architecture.jl",
    "test_dealiasing_math.jl",
    "test_gpu_solver_cpu.jl",
]

# Tests that may require special setup or longer runtime
const OPTIONAL_TEST_FILES = [
    "test_etdrk2_convergence.jl",  # Convergence test - may be slow
]

# MPI tests that must be run separately with mpiexec
# Run with: mpiexec -n 4 julia --project test/test_mpi_distributor.jl
#           mpiexec -n 4 julia --project test/test_distributed_gpu_transpose.jl
# Or use:   ./test/run_mpi_tests.sh 4
const MPI_TEST_FILES = [
    "test_mpi_distributor.jl",
    "test_distributed_gpu_transpose.jl",
    "test_transposable_field.jl",
]

for file in TEST_FILES
    @testset "$file" begin
        include(file)
    end
end

# Run optional tests if explicitly requested
if get(ENV, "TARANG_RUN_OPTIONAL_TESTS", "false") == "true"
    for file in OPTIONAL_TEST_FILES
        @testset "$file" begin
            include(file)
        end
    end
end

# Print reminder about MPI tests
@info """
MPI tests are not included in the standard test suite.
To run MPI tests with multiple processes, use:

    mpiexec -n 4 julia --project test/test_mpi_distributor.jl
    mpiexec -n 4 julia --project test/test_distributed_gpu_transpose.jl

Or use the test runner script:

    ./test/run_mpi_tests.sh 4
"""
