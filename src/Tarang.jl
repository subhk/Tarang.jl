"""
Tarang.jl - Spectral PDE framework for Julia

"""

module Tarang

__version__ = "1.0.0"
const dtype = Float64  # Default scalar type for CPU-only builds
const shape = ()  # Placeholder tuple to satisfy legacy references during CPU-only load
const evaluator = nothing  # Placeholder to satisfy legacy references during CPU-only load

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
using HDF5
using LoopVectorization
using ExponentialUtilities  # For Krylov-based exponential integrators
using FastGaussQuadrature   # For Gauss-Legendre quadrature in Legendre transforms
using KernelAbstractions    # For backend-agnostic GPU/CPU kernels

# Architecture abstraction for CPU/GPU support
include("core/architectures.jl")

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
include("core/operators.jl")
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
include("core/timesteppers.jl")
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

# Libraries submodule to avoid name collisions with core types
# (Currently empty - spherical geometry support removed)

# Public interface
export
    # Architecture (CPU/GPU support)
    AbstractArchitecture, AbstractSerialArchitecture,
    CPU, GPU,
    device, array_type, architecture,
    on_architecture, is_gpu, has_cuda,
    synchronize, unsafe_free!,
    launch_config, workgroup_size, launch!, KernelOperation,
    create_array, move_to_architecture,
    set_gpu_fft_min_elements!, gpu_fft_min_elements, should_use_gpu_fft,
    # Distributed GPU (GPU + MPI)
    DistributedGPUConfig, DistributedGPUFFT,
    distributed_fft_forward!, distributed_fft_backward!,
    check_cuda_aware_mpi, setup_distributed_gpu!,
    # TransposableField for 2D pencil decomposition
    TransposableField, TransposeLayout, XLocal, YLocal, ZLocal,
    TransposeBuffers, TransposeCounts, TransposeComms,
    Topology2D, create_topology_2d, auto_topology, AsyncTransposeState,
    make_transposable,
    transpose_z_to_y!, transpose_y_to_z!, transpose_y_to_x!, transpose_x_to_y!,
    async_transpose_z_to_y!, async_transpose_y_to_x!, wait_transpose!, is_transpose_complete,
    distributed_forward_transform!, distributed_backward_transform!,
    active_layout, current_data, local_shape,
    pack_for_transpose!, unpack_from_transpose!,
    compute_local_shapes, compute_local_shapes_2d, divide_evenly, local_range,
    create_transpose_comms, get_transpose_stats, reset_transpose_stats!,

    # Coordinate systems
    CartesianCoordinates,
    coords, unit_vector_fields,

    # Bases
    RealFourier, ComplexFourier, Fourier, ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi,
    derivative_basis, product_matrix, ncc_matrix,

    # Core classes
    Domain, Distributor, Field, ScalarField, VectorField, TensorField,

    # Operators
    grad, div, curl, lap, trace, skew, transpose_components,
    Gradient, Divergence, Curl, Laplacian, Trace, Skew, TransposeComponents,  # Operator types
    ∇, Δ, ∇², ∂t,  # Unicode aliases: ∇=grad, Δ=lap, ∇²=lap, ∂t=dt
    # Fractional Laplacian for SQG and other applications
    fraclap, sqrtlap, invsqrtlap,  # fraclap(f,α), sqrtlap=(-Δ)^(1/2), invsqrtlap=(-Δ)^(-1/2)
    Δᵅ,  # Unicode alias: Δᵅ(f, α) = fraclap(f, α)
    FractionalLaplacian,  # Type export
    # Hyperviscosity operators (higher-order Laplacian)
    hyperlap,  # hyperlap(f, n) = (-Δ)^n = |k|^(2n) in Fourier space
    Δ², Δ⁴, Δ⁶, Δ⁸,  # Unicode shortcuts: Δ²=biharmonic, Δ⁴, Δ⁶, Δ⁸
    dot, cross, ⋅, ×,  # Vector operations with Unicode: ⋅=dot, ×=cross
    DotProduct, CrossProduct,  # Arithmetic types
    outer, advective_cfl, cfl,
    interpolate, integrate, average, convert, lift, d, dt,
    # Evaluation function
    evaluate,

    # Cartesian-specific operators (multiclass dispatch)
    CartesianComponent, CartesianGradient, CartesianDivergence, CartesianCurl,
    CartesianLaplacian, CartesianTrace, CartesianSkew,
    DirectProductGradient, DirectProductDivergence, DirectProductLaplacian,
    cartesian_component, dispatch_cartesian_operator,

    # Matrix operation methods for implicit solvers
    matrix_dependence, matrix_coupling, subproblem_matrix,
    check_conditions, enforce_conditions, is_linear, operator_order,
    
    # Nonlinear operators
    advection, nonlinear_momentum, convection, AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator,
    NonlinearEvaluator, evaluate_nonlinear_term, evaluate_transform_multiply,
    
    # Problem types
    IVP, EVP, LBVP, NLBVP,
    
    # Solvers
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    
    # Field operations
    has_spectral_bases, apply_dealiasing_to_product!, fast_axpy!,
    vectorized_add!, vectorized_sub!, vectorized_mul!, vectorized_scale!,
    vectorized_axpy!, vectorized_linear_combination!,
    field_architecture, synchronize_field_architecture!,
    gpu_fft_mode, set_gpu_fft_mode!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    FieldStorageMode, SerialStorage, PencilStorage,
    storage_mode, is_pencil_storage, is_serial_storage,
    stack_components, unstack_components!,
    stack_tensor_components, unstack_tensor_components!,
    ensure_layout!, forward_transform!, backward_transform!,
    # GPU-aware data access (for file I/O)
    get_cpu_data, get_cpu_local_data, get_local_data, is_gpu_field,
    
    # Timesteppers (IMEX RK - Dedalus-compatible)
    RK111, RK222, RK443,
    # Timesteppers (Multistep IMEX)
    CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    # Timesteppers (Exponential Time Differencing)
    ETD_RK222, ETD_CNAB2, ETD_SBDF2,
    # Timesteppers (Diagonal IMEX - GPU-native)
    DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2,
    SpectralLinearOperator, set_spectral_linear_operator!,

    # Stochastic Forcing
    Forcing, StochasticForcingType, DeterministicForcingType,
    StochasticForcing, DeterministicForcing,
    generate_forcing!, reset_forcing!, set_dt!,
    apply_forcing!, get_forcing_real,
    energy_injection_rate, instantaneous_power,
    compute_forcing_spectrum,
    # Automatic stochastic forcing integration
    add_stochastic_forcing!, has_stochastic_forcing, get_stochastic_forcing,
    # Automatic temporal filter integration
    add_temporal_filter!, has_temporal_filters, get_temporal_filter, get_all_temporal_filters,

    # Analysis
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    # Global reduction functions (for parallel simulations)
    global_max, global_min, global_mean, global_sum, reduce_scalar,

    # SQG (Surface Quasi-Geostrophic) tools
    perp_grad, ∇⊥,  # Perpendicular gradient: ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)
    sqg_streamfunction, sqg_velocity, sqg_problem_setup,

    # Full QG (Quasi-Geostrophic) with coupled surface-interior dynamics
    QGSystem, qg_system_setup, qg_invert!, qg_step!,
    qg_surface_velocity!, qg_advection_rhs, qg_energy, extract_surface,

    # General Boundary Advection-Diffusion Framework
    BoundaryAdvectionDiffusion, BoundarySpec, DiffusionSpec,
    VelocitySource, PrescribedVelocity, InteriorDerivedVelocity, SelfDerivedVelocity,
    boundary_advection_diffusion_setup,
    bad_step!, bad_compute_velocity!, bad_compute_rhs!, bad_add_source!,
    bad_energy, bad_enstrophy, bad_max_velocity, bad_cfl_dt,

    # NetCDF Output
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler, 
    merge_processor_files, get_netcdf_info,
    
    # NetCDF Merging
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP,
    
    # Boundary Conditions (types for BoundaryConditionManager)
    BoundaryConditionManager, DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    # BC helper functions (use add_equation! for BCs - Dedalus style)
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    # Time/Space Dependent BCs
    TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
    set_time_variable!, add_coordinate_field!, update_time_dependent_bcs!,
    has_time_dependent_bcs, has_space_dependent_bcs, requires_bc_update,
    
    # Grid functions
    local_grid, local_grids,

    # Utility
    Lift,
    chunked_rng,
    rng_element,
    rng_elements,
    IndexArray,
    ChunkedRandomArray,
    MatSolvers,

    # GPU/CPU array helpers (for architecture-agnostic code)
    allocate_like, similar_zeros, copy_to_device, is_gpu_array,

    # Temporal Filters for Lagrangian Averaging
    TemporalFilter, ExponentialMean, ButterworthFilter, LagrangianFilter,
    get_mean, get_auxiliary, set_α!,
    update_displacement!, lagrangian_mean!, get_mean_velocity, get_displacement,
    filter_response, effective_averaging_time,

    # LES Models (Large Eddy Simulation)
    SGSModel, EddyViscosityModel,
    SmagorinskyModel, AMDModel,
    compute_eddy_viscosity!, compute_eddy_diffusivity!,
    compute_sgs_stress,
    get_eddy_viscosity, get_eddy_diffusivity, get_filter_width,
    mean_eddy_viscosity, max_eddy_viscosity,
    sgs_dissipation, mean_sgs_dissipation,
    set_constant!

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
