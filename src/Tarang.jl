"""
Tarang.jl - Spectral PDE framework for Julia

"""

module Tarang

const __version__ = "1.0.0"

using MPI
using PencilArrays
using PencilFFTs
using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using FFTW
using StaticArrays
using Parameters
using ChainRulesCore
using ForwardDiff
using LoopVectorization
using ExponentialUtilities  # For Krylov-based exponential integrators
using FastGaussQuadrature   # For Gauss-Legendre quadrature in Legendre transforms
using KernelAbstractions    # For backend-agnostic GPU/CPU kernels

# Architecture abstraction for CPU/GPU support
include("core/architectures.jl")

# Abstract types for breaking circular dependencies between modules.
# Concrete types are defined in their respective files but the Distributor
# struct (loaded early) needs to reference them by abstract type.
#
# TARGET SUBMODULE ARCHITECTURE (for future refactoring):
#   Tarang.Core       — AbstractArchitecture, Coordinate, Basis, Domain, Distributor, Field
#   Tarang.Operators  — Differentiate, Gradient, Divergence, Curl, operator evaluation
#   Tarang.Transforms — FourierTransform, ChebyshevTransform, LegendreTransform, distributed
#   Tarang.Solvers    — IVP, EVP, LBVP, timesteppers, BCs
#   Tarang.IO         — NetCDF, logging, progress, config
#   Tarang.Extras     — flow_tools, plot_tools, quick_domains
#
# The abstract types below bridge these would-be submodules:
abstract type AbstractNonlinearEvaluator end   # Solvers → Operators bridge
abstract type AbstractDistributedGPUConfig end  # Core → Transforms bridge
abstract type AbstractTransposeComms end        # Core → Transforms bridge
abstract type AbstractTransposeCounts end       # Core → Transforms bridge

# Custom PencilConfig struct for pencil array configuration
struct PencilConfig
    global_shape::Tuple{Vararg{Int}}
    mesh::Tuple{Vararg{Int}}
    comm::MPI.Comm
    decomp_dims::Tuple{Vararg{Bool}}
    dtype::Type  # Data type for the pencil arrays

    function PencilConfig(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}};
                         comm::MPI.Comm=MPI.COMM_WORLD,
                         decomp_dims::Tuple{Vararg{Bool}}=ntuple(i -> true, length(mesh)),
                         dtype::Type=Float64)
        new(global_shape, mesh, comm, decomp_dims, dtype)
    end
end

# Core utilities needed across modules
include("tools/general.jl")
include("tools/exceptions.jl")
include("tools/cache.jl")  # Must be before dispatch.jl (cached_class_instances, CachedClass)
include("tools/dispatch.jl")
include("tools/parsing.jl")

# Coordinate systems
include("core/coords.jl")

# Core modules
include("core/basis.jl")  
include("core/distributor.jl")
include("core/domain.jl")
include("core/field.jl")
include("core/future.jl")
include("core/arithmetic.jl")
include("core/operators/operators.jl")
include("core/cartesian_operators.jl")
include("core/transforms.jl")
include("core/boundary_conditions.jl")
include("core/problems.jl")
include("core/subsystems.jl")
include("core/system.jl")
include("tools/matsolvers.jl")
include("tools/gpu_matsolvers.jl")
include("core/solvers.jl")
include("core/stochastic_forcing.jl")
include("core/timesteppers/timesteppers.jl")
include("core/gpu_distributed.jl")  # Distributed GPU computing (GPU + MPI)
include("core/transposable_field.jl")  # TransposableField for 2D pencil decomposition

# NetCDF output must be included before evaluator (evaluator uses NetCDFFileHandler)
include("tools/netcdf_output.jl")
include("core/evaluator.jl")
include("core/nonlinear.jl")

# Tools
include("tools/config.jl")
# cache.jl moved to top (before dispatch.jl)
include("tools/array.jl")
include("tools/parallel.jl")
include("tools/logging.jl")
include("tools/progress.jl")
include("tools/random_arrays.jl")
include("tools/netcdf_merge.jl")
include("tools/temporal_filters.jl")
include("core/les_models.jl")

# Extras
include("extras/flow_tools.jl")
include("extras/plot_tools.jl")
include("extras/quick_domains.jl")
include("extras/analysis_tasks.jl")

# Pretty printing (after all types are defined)
include("tools/pretty_printing.jl")

# Libraries submodule to avoid name collisions with core types
# (Currently empty - spherical geometry support removed)

# Public interface
export
    # ═══════════════════════════════════════════════════════════════
    # Quick Start — most users only need these
    # ═══════════════════════════════════════════════════════════════
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    ScalarField, VectorField, TensorField,
    IVP, EVP, LBVP, NLBVP,
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    diagnose,
    add_parameters!, add_equation!, add_bc!,
    no_slip!, fixed_value!, free_slip!, insulating!,
    grid_data, coeff_data, set!,
    on_interval, on_sim_time,
    @root_only,

    # ═══════════════════════════════════════════════════════════════
    # Architecture (CPU / GPU)
    # ═══════════════════════════════════════════════════════════════
    AbstractArchitecture, AbstractSerialArchitecture,
    CPU, GPU,
    device, array_type, architecture,
    on_architecture, is_gpu, has_cuda,
    synchronize, unsafe_free!,
    launch_config, workgroup_size, launch!, KernelOperation,
    create_array, move_to_architecture,
    set_gpu_fft_min_elements!, gpu_fft_min_elements, should_use_gpu_fft,
    allocate_like, similar_zeros, copy_to_device, is_gpu_array,

    # ═══════════════════════════════════════════════════════════════
    # Distributed GPU (GPU + MPI)
    # ═══════════════════════════════════════════════════════════════
    DistributedGPUConfig, DistributedGPUFFT,
    distributed_fft_forward!, distributed_fft_backward!,
    check_cuda_aware_mpi, setup_distributed_gpu!,
    TransposableField, TransposeLayout, XLocal, YLocal, ZLocal,
    TransposeBuffers, TransposeCounts, TransposeComms,
    Topology2D, create_topology_2d, auto_topology, AsyncTransposeState,
    make_transposable,
    transpose_z_to_y!, transpose_y_to_z!, transpose_y_to_x!, transpose_x_to_y!,
    async_transpose_z_to_y!, async_transpose_y_to_x!, wait_transpose!, is_transpose_complete,
    distributed_forward_transform!, distributed_backward_transform!,
    active_layout, current_data, local_shape,

    # ═══════════════════════════════════════════════════════════════
    # Coordinates, Bases, Domains, Fields
    # ═══════════════════════════════════════════════════════════════
    CartesianCoordinates,
    coords, unit_vector_fields,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi,
    derivative_basis, product_matrix, ncc_matrix,
    Domain, Distributor, Field,

    # ═══════════════════════════════════════════════════════════════
    # Field operations & data access
    # ═══════════════════════════════════════════════════════════════
    field_architecture, synchronize_field_architecture!,
    gpu_fft_mode, set_gpu_fft_mode!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    FieldStorageMode, SerialStorage, PencilStorage,
    storage_mode, is_pencil_storage, is_serial_storage,
    stack_components, unstack_components!,
    ensure_layout!, forward_transform!, backward_transform!,
    get_cpu_data, get_cpu_local_data, get_local_data, is_gpu_field,
    local_grid, local_grids,

    # ═══════════════════════════════════════════════════════════════
    # Operators — differential
    # ═══════════════════════════════════════════════════════════════
    grad, div, curl, lap, trace, skew, transpose_components,
    Gradient, Divergence, Curl, Laplacian, Trace, Skew, TransposeComponents,
    ∇, Δ, ∇², ∂t,                       # Unicode: ∇=grad, Δ=∇²=lap, ∂t=dt
    d, dt, lift, Lift,
    interpolate, integrate, average, convert, evaluate,

    # ═══════════════════════════════════════════════════════════════
    # Operators — fractional & hyperviscosity
    # ═══════════════════════════════════════════════════════════════
    fraclap, sqrtlap, invsqrtlap, Δᵅ,   # fraclap(f,α), Δᵅ=fraclap
    FractionalLaplacian,
    hyperlap, Δ², Δ⁴, Δ⁶, Δ⁸,           # hyperlap(f,n), Δⁿ shortcuts

    # ═══════════════════════════════════════════════════════════════
    # Operators — vector / tensor arithmetic
    # ═══════════════════════════════════════════════════════════════
    dot, cross, ⋅, ×,                    # ⋅=dot, ×=cross
    DotProduct, CrossProduct,
    outer, advective_cfl, cfl,
    Copy, HilbertTransform, copy_field, hilbert,
    sym_diff, simplify, UFUNC_DERIVATIVES,
    frechet_differential, build_symbolic_jacobian,

    # ═══════════════════════════════════════════════════════════════
    # Operators — Cartesian-specific (multiclass dispatch)
    # ═══════════════════════════════════════════════════════════════
    CartesianComponent, CartesianGradient, CartesianDivergence, CartesianCurl,
    CartesianLaplacian, CartesianTrace, CartesianSkew,
    DirectProductGradient, DirectProductDivergence, DirectProductLaplacian,
    cartesian_component, dispatch_cartesian_operator,
    matrix_dependence, matrix_coupling, subproblem_matrix,
    check_conditions, enforce_conditions, is_linear, operator_order,

    # ═══════════════════════════════════════════════════════════════
    # Nonlinear operators
    # ═══════════════════════════════════════════════════════════════
    advection, nonlinear_momentum, convection,
    AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator,
    NonlinearEvaluator, evaluate_nonlinear_term, evaluate_transform_multiply,

    # ═══════════════════════════════════════════════════════════════
    # Time steppers
    # ═══════════════════════════════════════════════════════════════
    # IMEX Runge-Kutta (Dedalus-compatible)
    RK111, RK222, RK443, RKSMR,
    # Multistep IMEX
    CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    # Exponential Time Differencing
    ETD_RK222, ETD_CNAB2, ETD_SBDF2,
    # Diagonal IMEX (GPU-native)
    DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2,
    SpectralLinearOperator, set_spectral_linear_operator!,
    # Pencil IMEX (Chebyshev-Fourier MPI)
    PencilLinearOperator, set_pencil_linear_operator!, PencilLHSCache,
    pencil_implicit_solve!, pencil_implicit_solve_inplace!,
    build_pencil_lhs_matrix, get_pencil_lhs_factor!,
    is_pencil_imex_compatible, has_chebyshev_basis,

    # ═══════════════════════════════════════════════════════════════
    # Boundary conditions
    # ═══════════════════════════════════════════════════════════════
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
    set_time_variable!, add_coordinate_field!, update_time_dependent_bcs!,
    has_time_dependent_bcs, has_space_dependent_bcs, requires_bc_update,

    # ═══════════════════════════════════════════════════════════════
    # Analysis & diagnostics
    # ═══════════════════════════════════════════════════════════════
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    global_max, global_min, global_mean, global_sum, reduce_scalar,
    MatSolvers,

    # ═══════════════════════════════════════════════════════════════
    # I/O — NetCDF output & merging
    # ═══════════════════════════════════════════════════════════════
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler,
    merge_processor_files, get_netcdf_info,
    DictionaryHandler, VirtualFileHandler,
    add_dictionary_handler, add_virtual_file_handler, merge_virtual!,
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP,

    # ═══════════════════════════════════════════════════════════════
    # Stochastic forcing
    # ═══════════════════════════════════════════════════════════════
    Forcing, StochasticForcingType, DeterministicForcingType,
    StochasticForcing, DeterministicForcing,
    generate_forcing!, reset_forcing!, set_dt!,
    apply_forcing!, get_forcing_real,
    energy_injection_rate, instantaneous_power,
    compute_forcing_spectrum,
    add_stochastic_forcing!, has_stochastic_forcing, get_stochastic_forcing,

    # ═══════════════════════════════════════════════════════════════
    # Temporal filters (Lagrangian averaging)
    # ═══════════════════════════════════════════════════════════════
    TemporalFilter, ExponentialMean, ButterworthFilter, LagrangianFilter,
    get_mean, get_auxiliary, set_α!,
    update_displacement!, lagrangian_mean!, get_mean_velocity, get_displacement,
    filter_response, effective_averaging_time,
    add_temporal_filter!, has_temporal_filters, get_temporal_filter, get_all_temporal_filters,

    # ═══════════════════════════════════════════════════════════════
    # LES models (Large Eddy Simulation)
    # ═══════════════════════════════════════════════════════════════
    SGSModel, EddyViscosityModel,
    SmagorinskyModel, AMDModel,
    compute_eddy_viscosity!, compute_eddy_diffusivity!,
    compute_sgs_stress,
    get_eddy_viscosity, get_eddy_diffusivity, get_filter_width,
    mean_eddy_viscosity, max_eddy_viscosity,
    sgs_dissipation, mean_sgs_dissipation,
    set_constant!,

    # ═══════════════════════════════════════════════════════════════
    # Physics modules — SQG, QG, Boundary Advection-Diffusion
    # ═══════════════════════════════════════════════════════════════
    # Surface Quasi-Geostrophic
    perp_grad, ∇⊥,
    sqg_streamfunction, sqg_velocity, sqg_problem_setup,
    # Quasi-Geostrophic (coupled surface-interior)
    QGSystem, qg_system_setup, qg_invert!, qg_step!,
    qg_surface_velocity!, qg_advection_rhs, qg_energy, extract_surface,
    # Boundary Advection-Diffusion
    BoundaryAdvectionDiffusion, BoundarySpec, DiffusionSpec,
    VelocitySource, PrescribedVelocity, InteriorDerivedVelocity, SelfDerivedVelocity,
    boundary_advection_diffusion_setup,
    bad_step!, bad_compute_velocity!, bad_compute_rhs!, bad_add_source!,
    bad_energy, bad_enstrophy, bad_max_velocity, bad_cfl_dt

# Initialize MPI, configuration, and logging at runtime
function __init__()
    if !MPI.Initialized()
        MPI.Init()
    end

    # Initialize configuration system (load config files, apply env overrides)
    init_config!()

    # Initialize logging system (respects TARANG_LOG_LEVEL, TARANG_LOG_FILE env vars)
    init_logging!()

    # Initialize GPU solvers if CUDA is available
    _init_gpu_solvers!()

    # Note: OMP_NUM_THREADS controls OpenMP threads in BLAS/LAPACK libraries (OpenBLAS, MKL).
    # When using MPI parallelism, it's recommended to set OMP_NUM_THREADS=1 to avoid
    # oversubscription. Julia's native threading (Threads.@threads) is controlled by
    # JULIA_NUM_THREADS and works independently.
end

end # module
