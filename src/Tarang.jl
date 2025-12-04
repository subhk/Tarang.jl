"""
Tarang.jl - Julia implementation of Dedalus spectral PDE framework
Copyright (c) 2024, Subhajit Kar

This file is part of Tarang, which is free software distributed
under the terms of the GPLv3 license.
"""

module Tarang

__version__ = "0.1.0"
const dtype = Float64  # Default scalar type for CPU-only builds

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

# Custom PencilConfig struct for pencil array configuration
struct PencilConfig
    global_shape::Tuple{Vararg{Int}}
    mesh::Tuple{Vararg{Int}}
    comm::MPI.Comm
    decomp_dims::Tuple{Vararg{Bool}}

    function PencilConfig(global_shape::Tuple{Vararg{Int}}, mesh::Tuple{Vararg{Int}};
                         comm::MPI.Comm=MPI.COMM_WORLD,
                         decomp_dims::Tuple{Vararg{Bool}}=ntuple(i -> true, length(mesh)))
        new(global_shape, mesh, comm, decomp_dims)
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
include("core/transforms.jl")
include("core/boundary_conditions.jl")
include("core/problems.jl")
include("core/subsystems.jl")
include("core/system.jl")
include("tools/matsolvers.jl")
include("core/solvers.jl")
include("core/timesteppers.jl")
include("core/evaluator.jl")
include("core/nonlinear_terms.jl")
# Note: Optimizations are integrated directly into the core modules above

# Tools
include("tools/config.jl")
include("tools/cache.jl")
include("tools/array.jl")
include("tools/parallel.jl")
include("tools/logging.jl")
include("tools/progress.jl")
include("tools/random_arrays.jl")
include("tools/memory_3d.jl")
include("tools/netcdf_output.jl")
include("tools/netcdf_merge.jl")

# Extras
include("extras/flow_tools.jl")
include("extras/plot_tools.jl")
include("extras/quick_domains.jl")
include("extras/analysis_tasks.jl")

# Libraries submodule to avoid name collisions with core types
# (Currently empty - spherical geometry support removed)

# Public interface - mirroring Dedalus structure
export
    # Coordinate systems
    CartesianCoordinates,
    coords, unit_vector_fields,

    # Bases
    RealFourier, ComplexFourier, Fourier, ChebyshevT, ChebyshevU, Legendre,

    # Core classes
    Domain, Distributor, Field, ScalarField, VectorField, TensorField,
    
    # Operators
    grad, div, curl, lap, trace, skew, transpose_components,
    
    # Nonlinear operators
    advection, nonlinear_momentum, convection, AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator,
    NonlinearEvaluator, evaluate_nonlinear_term, evaluate_transform_multiply,
    
    # Problem types
    IVP, EVP, LBVP, NLBVP,
    
    # Solvers
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    
    # Optimized operations (integrated into core modules)
    has_spectral_bases, apply_dealiasing_to_product!, optimized_axpy!,
    vectorized_add!, vectorized_sub!, vectorized_mul!, vectorized_scale!,
    vectorized_axpy!, vectorized_linear_combination!,
    
    # Timesteppers
    RK111, RK222, RK443, CNAB1, CNAB2, SBDF1, SBDF2, SBDF3, SBDF4,
    
    # Analysis
    GlobalFlowProperty, CFL,
    
    # NetCDF Output
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler, 
    merge_processor_files, get_netcdf_info,
    
    # NetCDF Merging
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP,
    
    # Boundary Conditions
    BoundaryConditionManager, DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    add_dirichlet_bc!, add_neumann_bc!, add_robin_bc!, add_stress_free_bc!,
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
    
    # Warn about threading like Dedalus
    omp_num_threads = get(ENV, "OMP_NUM_THREADS", nothing)
    if omp_num_threads != "1"
        @warn "Threading has not been disabled. This may massively degrade Tarang performance."
        @warn "We strongly suggest setting the \"OMP_NUM_THREADS\" environment variable to \"1\"."
    end
end

end # module
