"""
Tarang.jl - Spectral PDE framework for Julia
Copyright (c) 2024, Subhajit Kar

This file is part of Tarang, which is free software distributed
under the terms of the GPLv3 license.
"""

module Tarang

__version__ = "0.1.0"
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
include("core/solvers.jl")
include("core/stochastic_forcing.jl")
include("core/timesteppers.jl")

# NetCDF output must be included before evaluator (evaluator uses NetCDFFileHandler)
include("tools/netcdf_output.jl")
include("core/evaluator.jl")
include("core/nonlinear_terms.jl")

# Tools
include("tools/config.jl")
include("tools/cache.jl")
include("tools/array.jl")
include("tools/parallel.jl")
include("tools/logging.jl")
include("tools/progress.jl")
include("tools/random_arrays.jl")
include("tools/netcdf_merge.jl")

# Extras
include("extras/flow_tools.jl")
include("extras/plot_tools.jl")
include("extras/quick_domains.jl")
include("extras/analysis_tasks.jl")

# Libraries submodule to avoid name collisions with core types
# (Currently empty - spherical geometry support removed)

# Public interface
export
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
    ∇, Δ, ∇², ∂t,  # Unicode aliases: ∇=grad, Δ=lap, ∇²=lap, ∂t=dt
    # Fractional Laplacian for SQG and other applications
    fraclap, sqrtlap, invsqrtlap,  # fraclap(f,α), sqrtlap=(-Δ)^(1/2), invsqrtlap=(-Δ)^(-1/2)
    Δᵅ,  # Unicode alias: Δᵅ(f, α) = fraclap(f, α)
    FractionalLaplacian,  # Type export
    # Hyperviscosity operators (higher-order Laplacian)
    hyperlap,  # hyperlap(f, n) = (-Δ)^n = |k|^(2n) in Fourier space
    Δ², Δ⁴, Δ⁶, Δ⁸,  # Unicode shortcuts: Δ²=biharmonic, Δ⁴, Δ⁶, Δ⁸
    dot, cross, ⋅, ×,  # Vector operations with Unicode: ⋅=dot, ×=cross
    outer, advective_cfl, cfl,
    interpolate, integrate, average, convert, lift, d, dt,

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
    
    # Timesteppers (IMEX RK - Dedalus-compatible)
    RK111, RK222, RK443,
    # Timesteppers (Multistep IMEX)
    CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    # Timesteppers (Exponential Time Differencing)
    ETD_RK222, ETD_CNAB2, ETD_SBDF2,

    # Stochastic Forcing
    Forcing, StochasticForcingType, DeterministicForcingType,
    StochasticForcing, DeterministicForcing,
    generate_forcing!, reset_forcing!, set_dt!,
    apply_forcing!, get_forcing_real,
    energy_injection_rate, instantaneous_power,
    compute_forcing_spectrum,
    
    # Analysis
    GlobalFlowProperty, CFL,

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
    
    # Boundary Conditions
    BoundaryConditionManager, DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    add_dirichlet_bc!, add_neumann_bc!, add_robin_bc!, add_stress_free_bc!, add_no_slip_bc!,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    # Time/Space Dependent BCs
    TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
    set_time_variable!, add_coordinate_field!, update_time_dependent_bcs!,
    has_time_dependent_bcs, has_space_dependent_bcs, requires_bc_update,
    
    # Utility
    Lift,
    chunked_rng,
    rng_element,
    rng_elements,
    IndexArray,
    ChunkedRandomArray,
    MatSolvers

# Initialize MPI if not already initialized
function __init__()
    if !MPI.Initialized()
        MPI.Init()
    end

    # Note: OMP_NUM_THREADS controls OpenMP threads in BLAS/LAPACK libraries (OpenBLAS, MKL).
    # When using MPI parallelism, it's recommended to set OMP_NUM_THREADS=1 to avoid
    # oversubscription. Julia's native threading (Threads.@threads) is controlled by
    # JULIA_NUM_THREADS and works independently.
end

end # module
