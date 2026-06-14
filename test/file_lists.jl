# Central registry of test files, grouped by how they must be executed.
#
# This is the single source of truth for the test inventory. It is included by:
#   * test/runtests.jl     — the package test entry point (CPU + optional + GPU)
#   * test/run_mpi_ci.jl   — the multi-rank MPI driver used in CI
#   * test/test_test_inventory.jl (transitively) — guards that every test_*.jl
#     file on disk is registered in exactly one of the lists below.
#
# Keep new test files registered here so they cannot silently sit outside the
# runner. The inventory test fails if a file is unlisted or a listed file is
# missing.

# Default CPU suite — runs on every CI job and `Pkg.test()`.
const TEST_FILES = [
    "test_test_inventory.jl",
    "test_aqua.jl",
    "test_jet.jl",
    "test_coords.jl",
    "test_basis.jl",
    "test_basis_wavenumbers.jl",
    "test_architectures.jl",
    "test_config.jl",
    "test_fields.jl",
    "test_problems.jl",
    "test_operators_basic.jl",
    "test_interpolation.jl",
    "test_matrix_apply.jl",
    "test_field_layout_operations.jl",
    "test_cartesian_operator_core.jl",
    "test_transforms_extended.jl",
    "test_transform_inplace.jl",
    "test_bc_regression.jl",
    "test_cfl.jl",
    "test_domain_metadata.jl",
    "test_subsystems.jl",
    "test_expression_matrices.jl",
    "test_ncc_product_matrices.jl",
    "test_lift_convert.jl",
    "test_pencil_system.jl",
    "test_pencil_matrices.jl",
    "test_transforms.jl",
    "test_jacobi_transforms.jl",
    "test_fft_dct.jl",
    "test_fftw_threads.jl",
    "test_solvers.jl",
    "test_bvp_solve.jl",
    "test_evp_solve.jl",
    "test_subproblem_modes.jl",
    "test_problem_matrices_support.jl",
    "test_flow_tools.jl",
    "test_spectra.jl",
    "test_quick_domains.jl",
    "test_plot_tools.jl",
    "test_compatibility.jl",
    "test_namespaces.jl",
    "test_root_module_structure.jl",
    "test_cartesian_operators.jl",
    "test_stochastic_forcing.jl",
    "test_temporal_filters.jl",
    "test_temporal_filters_extended.jl",
    "test_arithmetic.jl",
    "test_distributor.jl",
    "test_les_models.jl",
    "test_chebyshev.jl",
    "test_domain.jl",
    "test_boundary_conditions.jl",
    "test_nonlinear.jl",
    "test_general.jl",
    "test_dedalus_features.jl",
    "test_symbolic_diff.jl",
    "test_derivatives_polynomial.jl",
    "test_streamfunction.jl",
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
    "test_type_stability.jl",
    "test_field_typestability.jl",
    "test_filter_forcing_typestability.jl",
    "test_fourier_algebraic_constraints.jl",
    "test_lazy_rhs_fourier.jl",
    "test_timestepper_boundaries.jl",
    "test_field_pool.jl",
    "test_linalg.jl",
    "test_tools_array.jl",
    "test_tools_parallel.jl",
    "test_random_arrays.jl",
    "test_matrices_builders.jl",
    "test_component_buffers.jl",
    "test_tensor_misc.jl",
    "test_subproblem_ncc.jl",
    "test_pretty_printing.jl",
    "test_progress.jl",
    "test_convenience_api.jl",
]

# CPU tests that are valid in ordinary CI but are kept out of the default
# package test path because they are slower convergence/end-to-end checks.
# Run with TARANG_RUN_OPTIONAL_TESTS=true (alongside the default suite) or
# TARANG_ONLY_OPTIONAL_TESTS=true (these only). The `optional-cpu-tests` CI job
# uses the latter.
const OPTIONAL_TEST_FILES = [
    "test_etdrk2_convergence.jl",  # Convergence test - may be slow
    "test_end_to_end_pde.jl",      # Full PDE solve test
    "test_pencil_imex.jl",
    "test_subproblem_rk.jl",       # Subproblem RK integration test (RBC 2D)
]

# Single-process CUDA tests. Run with TARANG_RUN_GPU_TESTS=true on a CUDA host
# (the JuliaGPU Buildkite pipeline sets this).
const GPU_TEST_FILES = [
    "test_dct_reorder.jl",
    "test_optimized_dct.jl",
    "test_gpu_transform_correctness.jl",
    "test_ilu0_preconditioner.jl",
]

# Multi-rank MPI tests. Each file calls MPI.Init itself and must be launched in
# its own `mpiexec -n <ranks>` world (never include()-ed into one process).
# Every file here is CPU-capable: the two CUDA-named entries fall back to a CPU
# path (round-trip on CPU arrays / @test_skip when CUDA is absent), so the whole
# list runs on GitHub-hosted runners. Driven by test/run_mpi_ci.jl.
# Run one manually with, e.g.:
#   mpiexec -n 4 julia --project test/test_mpi_distributor.jl
# Or the whole list:
#   ./test/run_mpi_tests.sh 4
const MPI_TEST_FILES = [
    "test_mpi_distributor.jl",
    "test_mpi_local_indices.jl",
    "test_mpi_field_initialization.jl",
    "test_mpi_algebraic_constraints.jl",
    "test_mpi_lazy_rhs_fourier.jl",
    "test_mpi_dealiasing_product.jl",
    "test_mpi_advection_term.jl",
    "test_mpi_implicit_advection.jl",
    "test_mpi_batched_transform.jl",
    "test_mpi_dotproduct_term.jl",
    "test_stochastic_forcing_mpi.jl",
    "test_mpi_integrate.jl",
    "test_mpi_spectral_filter.jl",
    "test_mpi_virtual_output.jl",
    "test_distributed_gpu_transpose.jl",
    "test_transposable_field.jl",
]

# Distributed CUDA/NCCL tests. These require CUDA extension symbols, NCCL, or
# multi-process GPU setup, so they cannot run on GitHub-hosted runners — they
# run on the JuliaGPU Buildkite pipeline. Keep this list explicit so new CPU
# test files do not silently sit outside the runner.
const DISTRIBUTED_GPU_TEST_FILES = [
    "test_distributed_dct.jl",
    "test_distributed_dispatch.jl",
    "test_distributed_parseval.jl",
    "test_nccl_alltoall.jl",
    "test_nccl_subcomm.jl",
    "test_pencil_decomposition.jl",
]
