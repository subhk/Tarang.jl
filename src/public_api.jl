# Public interface. Keep this file declarative: it defines the root-module
# compatibility surface, while implementation load order stays in Tarang.jl.
export
    # Quick Start - most users only need these
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

    # Architecture (CPU / GPU)
    AbstractArchitecture, AbstractSerialArchitecture,
    CPU, GPU,
    device, array_type, architecture,
    on_architecture, is_gpu, has_cuda,
    synchronize, unsafe_free!,
    launch_config, workgroup_size, launch!, KernelOperation,
    create_array, move_to_architecture,
    set_gpu_fft_min_elements!, gpu_fft_min_elements, should_use_gpu_fft,
    allocate_like, similar_zeros, copy_to_device, is_gpu_array,
    _gpu_chebyshev_deriv!,

    # Distributed GPU (GPU + MPI)
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

    # Coordinates, Bases, Domains, Fields
    CartesianCoordinates,
    coords, unit_vector_fields,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi,
    derivative_basis, product_matrix, ncc_matrix,
    Domain, Distributor, Field,

    # Field operations and data access
    field_architecture, synchronize_field_architecture!,
    gpu_fft_mode, set_gpu_fft_mode!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    FieldStorageMode, SerialStorage, PencilStorage,
    storage_mode, is_pencil_storage, is_serial_storage,
    stack_components, unstack_components!,
    ensure_layout!, forward_transform!, backward_transform!,
    get_cpu_data, get_cpu_local_data, get_local_data, is_gpu_field,
    local_grid, local_grids,

    # Differential operators
    grad, div, curl, lap, trace, skew, transpose_components,
    Gradient, Divergence, Curl, Laplacian, Trace, Skew, TransposeComponents,
    ∇, Δ, ∇², ∂t,
    d, dt, lift, Lift,
    interpolate, integrate, average, convert, evaluate,

    # Fractional operators and hyperviscosity
    fraclap, sqrtlap, invsqrtlap, Δᵅ,
    FractionalLaplacian,
    hyperlap, Δ², Δ⁴, Δ⁶, Δ⁸,

    # Vector and tensor arithmetic
    dot, cross, ⋅, ×,
    DotProduct, CrossProduct,
    outer, advective_cfl, cfl,
    Copy, HilbertTransform, copy_field, hilbert,
    sym_diff, simplify,
    frechet_differential, build_symbolic_jacobian,

    # Cartesian-specific operators
    CartesianComponent, CartesianGradient, CartesianDivergence, CartesianCurl,
    CartesianLaplacian, CartesianTrace, CartesianSkew,
    DirectProductGradient, DirectProductDivergence, DirectProductLaplacian,
    cartesian_component,
    is_linear, operator_order,

    # Nonlinear operators
    advection, nonlinear_momentum, convection,
    AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator,

    # Time steppers
    RK111, RK222, RK443, RKSMR,
    CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    ETD_RK222, ETD_CNAB2, ETD_SBDF2,
    DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2,
    SpectralLinearOperator, set_spectral_linear_operator!,

    # Boundary conditions
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
    set_time_variable!, add_coordinate_field!, update_time_dependent_bcs!,
    has_time_dependent_bcs, has_space_dependent_bcs, requires_bc_update,

    # Analysis and diagnostics
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    global_max, global_min, global_mean, global_sum, reduce_scalar,
    MatSolvers,

    # I/O - NetCDF output and merging
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler,
    DictionaryHandler, VirtualFileHandler,
    add_dictionary_handler, add_virtual_file_handler, merge_virtual!,
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP,

    # Stochastic forcing
    Forcing, StochasticForcingType, DeterministicForcingType,
    StochasticForcing, DeterministicForcing,
    generate_forcing!, reset_forcing!, set_dt!,
    apply_forcing!, get_forcing_real,
    energy_injection_rate, instantaneous_power,
    compute_forcing_spectrum,
    add_stochastic_forcing!, has_stochastic_forcing, get_stochastic_forcing,

    # Temporal filters
    TemporalFilter, ExponentialMean, ButterworthFilter, LagrangianFilter,
    get_mean, get_auxiliary, set_α!,
    update_displacement!, lagrangian_mean!, get_mean_velocity, get_displacement,
    filter_response, effective_averaging_time,
    add_temporal_filter!, has_temporal_filters, get_temporal_filter, get_all_temporal_filters,

    # LES models
    SGSModel, EddyViscosityModel,
    SmagorinskyModel, AMDModel,
    compute_eddy_viscosity!, compute_eddy_diffusivity!,
    compute_sgs_stress,
    get_eddy_viscosity, get_eddy_diffusivity, get_filter_width,
    mean_eddy_viscosity, max_eddy_viscosity,
    sgs_dissipation, mean_sgs_dissipation,
    set_constant!,

    # Physics modules - SQG, QG, Boundary Advection-Diffusion
    perp_grad, ∇⊥,
    sqg_streamfunction, sqg_velocity, sqg_problem_setup,
    QGSystem, qg_system_setup, qg_invert!, qg_step!,
    qg_surface_velocity!, qg_advection_rhs, qg_energy, extract_surface,
    BoundaryAdvectionDiffusion, BoundarySpec, DiffusionSpec,
    VelocitySource, PrescribedVelocity, InteriorDerivedVelocity, SelfDerivedVelocity,
    boundary_advection_diffusion_setup,
    bad_step!, bad_compute_velocity!, bad_compute_rhs!, bad_add_source!,
    bad_energy, bad_enstrophy, bad_max_velocity, bad_cfl_dt
