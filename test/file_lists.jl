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
    "test_cpu_audit_2026_06_19.jl",
    "test_cpu_audit_2026_06_20.jl",
    "test_cpu_audit_batch3.jl",
    "test_cpu_audit_2026_06_21.jl",
    # CPU coverage-raising tests (2026-06-19): behavior-driven, target previously
    # under-covered serial-CPU modules.
    "test_cov_step_diagonal_imex.jl",
    "test_cov_problem_parsing.jl",
    "test_cov_basis_operators.jl",
    "test_cov_solver_compiled_rhs.jl",
    "test_cov_solver_state_vectors.jl",
    "test_cov_solver_utils.jl",
    "test_cov_subsystem_types.jl",
    "test_cov_subproblem_expr_helpers.jl",
    "test_cov_matrices_subproblem_helpers.jl",
    "test_cov_field_data_scales.jl",
    "test_cov_field_data_copy_alloc.jl",
    "test_cov_tensor_fractional_laplacian.jl",
    "test_cov_flow_tools_domain_utils.jl",
    "test_analysis_tasks.jl",
    "test_phi_functions.jl",
    "test_evaluator.jl",
    "test_parsing.jl",
    "test_operators_tensor.jl",
    "test_diagonal_imex.jl",
    "test_etd_multistep.jl",
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
    "test_distributed_gpu_dct1_support.jl",
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
    # GPU-CI-only: distributed RealFourier×Chebyshev DCT-I local primitives PLUS an
    # end-to-end field-level testset (Task 9) that round-trips a 3D RealFourier ×
    # ComplexFourier × Chebyshev GPU field through forward/backward_transform! and
    # checks the unsupported-layout (RealFourier on dim 2) CPU fallback.
    # Self-guards with CUDA.functional() (skips cleanly when no GPU is present).
    # NOTE: at nprocs==1 (single-process include here) the field test is a wiring +
    # round-trip smoke test (distributed dispatch is gated off; CPU DCT-I fallback).
    # The .buildkite GPU pipeline MUST also run this file at nprocs ∈ {2,4} under
    # mpiexec WITH NCCL to exercise the live distributed multi-GPU transform path.
    "test_gpu_distributed_dct1.jl",
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
    "test_mpi_reductions.jl",
    "test_mpi_reduction_double_reduce.jl",   # global_sum/mean/turbulence_rms must not double-reduce PencilArray (np>=2)
    "test_mpi_fill_random_walltime.jl",      # fill_random reproducible decomp-independent + proceed() collective wall-time stop (np>=2)
    "test_mpi_interp_hilbert_guard.jl",      # interpolate/Hilbert error loudly on a decomposed Fourier axis; local-axis interp works (np>=2)
    "test_mpi_spectral_filter.jl",
    "test_mpi_virtual_output.jl",
    # MPI correctness fixes 2026-06-21 (see memory/project_mpi_audit_2026_06_21.md).
    "test_mpi_spectra_consistency.jl",       # spectra axes_local + global num_bins (np>=2)
    "test_mpi_forcing_diag.jl",              # C4 _forcing_reduce_partial (np>=2)
    "test_mpi_unit_factor_mesh.jl",          # (1,N) mesh normalization (np>=2)
    "test_mpi_grouped_transpose_rankinv.jl", # N1 rank-invariant grouping (np>=2)
    "test_mpi_distributor_match_np4.jl",     # C1 coord ordering, 2x2 mesh (np==4)
    "test_mpi_distributor_remainder_np2.jl", # C3 remainder-on-last-rank (np==2)
    "test_mpi_fourier_chebyshev.jl",         # FFC: Cheb-last clear error, Cheb-first round-trip (np>=2)
    "test_mpi_cheb_fourier_ivp.jl",          # distributed Cheb-Fourier IMEX IVP == serial (np>=2/4)
    "test_mpi_cheb_fourier_ivp_nonlinear.jl", # distributed NONLINEAR Cheb-Fourier channel IVP (advection+dealias+tau-BC+IMEX) == serial (np>=2)
    # MPI correctness fixes 2026-06-23 (see memory/project_mpi_audit_2026_06_21.md).
    "test_mpi_decomp_forcing_audit.jl",      # #1/#4 get_local_range slab; #2 forcing wavenumber placement (np>=2)
    "test_mpi_bvp_cheb_fourier.jl",          # steady LBVP/NLBVP solve-layout transpose + Newton resnorm Allreduce (np>=2)
    "test_mpi_output_audit.jl",              # wall_dt schedule Bcast (no deadlock) + complex checkpoint metadata (np>=2)
    "test_mpi_forcing_work.jl",              # work_stratonovich/ito/instantaneous_power distributed == serial (np>=2)
    "test_mpi_padded_dealiasing.jl",         # distributed 3/2 padded dealiasing == serial (transpose-pad) (np>=2)
    "test_mpi_padded_dealiasing_3d.jl",      # distributed 3D padded dealiasing == serial (N-D, 2D-mesh at np=4) (np>=2)
    "test_mpi_padded_dealiasing_chebfourier.jl", # distributed mixed Cheb-Fourier dealiasing == serial (Fourier-only pad) (np>=2)
    "test_mpi_dealiasing_ivp_3d.jl",         # 3D Burgers IVP solve distributed == serial (e2e dealias-in-timestepper) (np>=2)
    "test_mpi_padded_dealiasing_3d_mixed.jl", # 3D Cheb-Fourier-Fourier dealiasing == serial (decomp-order alignment fix) (np>=2)
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
