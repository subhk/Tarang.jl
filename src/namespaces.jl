"""
    Tarang.Fields

Facade for field, domain, and data-layout operations.

This module intentionally re-exports existing Tarang bindings instead of moving
implementations. It gives users and internal code a stable architecture boundary
while preserving the long-standing `Tarang.X` API.
"""
module Fields
import ..Tarang:
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    Domain, Distributor, Field,
    ScalarField, VectorField, TensorField,
    grid_data, coeff_data, set!,
    ensure_layout!, forward_transform!, backward_transform!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    field_architecture, synchronize_field_architecture!,
    get_cpu_data, get_cpu_local_data, get_local_data,
    local_grid, local_grids

export
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    Domain, Distributor, Field,
    ScalarField, VectorField, TensorField,
    grid_data, coeff_data, set!,
    ensure_layout!, forward_transform!, backward_transform!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    field_architecture, synchronize_field_architecture!,
    get_cpu_data, get_cpu_local_data, get_local_data,
    local_grid, local_grids
end

"""
    Tarang.Problems

Facade for problem definitions, equations, and boundary conditions.
"""
module Problems
import ..Tarang:
    IVP, EVP, LBVP, NLBVP,
    add_parameters!, add_equation!, add_bc!,
    no_slip!, fixed_value!, free_slip!, insulating!,
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    set_time_variable!, add_coordinate_field!,
    update_time_dependent_bcs!, has_time_dependent_bcs,
    has_space_dependent_bcs, requires_bc_update

export
    IVP, EVP, LBVP, NLBVP,
    add_parameters!, add_equation!, add_bc!,
    no_slip!, fixed_value!, free_slip!, insulating!,
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    set_time_variable!, add_coordinate_field!,
    update_time_dependent_bcs!, has_time_dependent_bcs,
    has_space_dependent_bcs, requires_bc_update
end

"""
    Tarang.Solvers

Facade for solver types, diagnostics, reducers, and linear algebra backends.
"""
module Solvers
import ..Tarang:
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    diagnose, MatSolvers,
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    global_max, global_min, global_mean, global_sum, reduce_scalar

export
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    diagnose, MatSolvers,
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    global_max, global_min, global_mean, global_sum, reduce_scalar
end

"""
    Tarang.Timesteppers

Facade for timestepper schemes and timestepper-local operator helpers.
"""
module Timesteppers
import ..Tarang:
    TimeStepper,
    RK111, RK222, RK443, RKSMR,
    CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    ETD_RK222, ETD_CNAB2, ETD_SBDF2,
    DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2,
    SpectralLinearOperator, set_spectral_linear_operator!

export
    TimeStepper,
    RK111, RK222, RK443, RKSMR,
    CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    ETD_RK222, ETD_CNAB2, ETD_SBDF2,
    DiagonalIMEX_RK222, DiagonalIMEX_RK443, DiagonalIMEX_SBDF2,
    SpectralLinearOperator, set_spectral_linear_operator!
end

"""
    Tarang.TransformOps

Facade for Tarang transform operations.

`Tarang.Transforms` is already a binding imported from PencilFFTs, so this
namespace uses `TransformOps` to avoid shadowing third-party internals.
"""
module TransformOps
import ..Tarang:
    forward_transform!, backward_transform!,
    distributed_forward_transform!, distributed_backward_transform!,
    setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi

export
    forward_transform!, backward_transform!,
    distributed_forward_transform!, distributed_backward_transform!,
    setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi
end

"""
    Tarang.Output

Facade for output handlers and NetCDF merge utilities.
"""
module Output
import ..Tarang:
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler,
    DictionaryHandler, VirtualFileHandler,
    add_dictionary_handler, add_virtual_file_handler, merge_virtual!,
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP

export
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler,
    DictionaryHandler, VirtualFileHandler,
    add_dictionary_handler, add_virtual_file_handler, merge_virtual!,
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP
end
