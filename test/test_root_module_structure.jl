using Test

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const TARANG_SOURCE = read(joinpath(ROOT_DIR, "src", "Tarang.jl"), String)
const LOAD_ORDER_SOURCE = read(joinpath(ROOT_DIR, "src", "load_order.jl"), String)
const PUBLIC_API_SOURCE = read(joinpath(ROOT_DIR, "src", "public_api.jl"), String)
const BOUNDARY_CONDITIONS_SOURCE = read(joinpath(ROOT_DIR, "src", "core", "boundary_conditions.jl"), String)
const API_DIR = joinpath(ROOT_DIR, "src", "api")
const PUBLIC_API_DIR = joinpath(API_DIR, "public")
const BOUNDARY_CONDITIONS_DIR = joinpath(ROOT_DIR, "src", "core", "boundary_conditions")

@testset "Root module structure" begin
    @test occursin("include(\"dependencies.jl\")", TARANG_SOURCE)
    @test occursin("include(\"load_order.jl\")", TARANG_SOURCE)
    @test occursin("include(\"public_api.jl\")", TARANG_SOURCE)
    @test occursin("include(\"api/namespaces.jl\")", TARANG_SOURCE)
    @test occursin("include(\"runtime_init.jl\")", TARANG_SOURCE)

    @test !occursin("using MPI", TARANG_SOURCE)
    @test !occursin("using PencilArrays", TARANG_SOURCE)
    @test !occursin("using KernelAbstractions", TARANG_SOURCE)
    @test !occursin("abstract type AbstractNonlinearEvaluator", TARANG_SOURCE)
    @test !occursin("struct PencilConfig", TARANG_SOURCE)
    @test !occursin("include(\"core/coords.jl\")", TARANG_SOURCE)
    @test !occursin("include(\"tools/netcdf_output.jl\")", TARANG_SOURCE)
    @test !occursin("include(\"namespaces.jl\")", TARANG_SOURCE)
    @test !occursin("function __init__", TARANG_SOURCE)
    @test !occursin("# Public interface\nexport", TARANG_SOURCE)

    @test isfile(joinpath(ROOT_DIR, "src", "dependencies.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "load_order.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "module_contracts.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "public_api.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "runtime_init.jl"))
    @test isfile(joinpath(API_DIR, "namespaces.jl"))
    @test isfile(joinpath(API_DIR, "fields.jl"))
    @test isfile(joinpath(API_DIR, "problems.jl"))
    @test isfile(joinpath(API_DIR, "solvers.jl"))
    @test isfile(joinpath(API_DIR, "timesteppers.jl"))
    @test isfile(joinpath(API_DIR, "transform_ops.jl"))
    @test isfile(joinpath(API_DIR, "output.jl"))
end

@testset "Public API export structure" begin
    export_files = [
        "quick_start.jl",
        "architecture.jl",
        "distributed_gpu.jl",
        "fields.jl",
        "operators.jl",
        "timesteppers.jl",
        "problems.jl",
        "diagnostics.jl",
        "output.jl",
        "forcing.jl",
        "filters_models.jl",
        "physics.jl",
    ]

    for file in export_files
        @test occursin("include(\"api/public/$file\")", PUBLIC_API_SOURCE)
        @test isfile(joinpath(PUBLIC_API_DIR, file))
    end

    @test !occursin("\nexport\n", PUBLIC_API_SOURCE)
    @test !occursin("PeriodicDomain, ChebyshevDomain", PUBLIC_API_SOURCE)
    @test !occursin("NetCDFFileHandler, NetCDFEvaluator", PUBLIC_API_SOURCE)
end

@testset "Boundary condition module structure" begin
    @test occursin("include(\"boundary_conditions/types.jl\")", BOUNDARY_CONDITIONS_SOURCE)
    @test occursin("include(\"boundary_conditions/construction.jl\")", BOUNDARY_CONDITIONS_SOURCE)
    @test isfile(joinpath(BOUNDARY_CONDITIONS_DIR, "types.jl"))
    @test isfile(joinpath(BOUNDARY_CONDITIONS_DIR, "construction.jl"))

    @test !occursin("mutable struct BoundaryConditionManager", BOUNDARY_CONDITIONS_SOURCE)
    @test !occursin("mutable struct BCCacheStore", BOUNDARY_CONDITIONS_SOURCE)
    @test !occursin("struct DirichletBC", BOUNDARY_CONDITIONS_SOURCE)
    @test !occursin("function dirichlet_bc", BOUNDARY_CONDITIONS_SOURCE)
    @test !occursin("function add_bc!", BOUNDARY_CONDITIONS_SOURCE)
    @test !occursin("function register_tau_field!", BOUNDARY_CONDITIONS_SOURCE)
end

@testset "Implementation load order structure" begin
    @test occursin("include(\"core/load_contracts.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"tools/load_bootstrap.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"core/load_fields.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"core/load_problem_stack.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"tools/load_matsolvers.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"core/load_solver_stack.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"tools/load_output.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"core/load_evaluation.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"tools/load_runtime.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"core/load_models.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"extras/load_extras.jl\")", LOAD_ORDER_SOURCE)
    @test occursin("include(\"tools/load_pretty_printing.jl\")", LOAD_ORDER_SOURCE)

    @test !occursin("include(\"core/architectures.jl\")", LOAD_ORDER_SOURCE)
    @test !occursin("include(\"tools/general.jl\")", LOAD_ORDER_SOURCE)
    @test !occursin("include(\"core/field.jl\")", LOAD_ORDER_SOURCE)
    @test !occursin("include(\"tools/netcdf_output.jl\")", LOAD_ORDER_SOURCE)
    @test !occursin("include(\"extras/flow_tools.jl\")", LOAD_ORDER_SOURCE)

    @test isfile(joinpath(ROOT_DIR, "src", "core", "load_contracts.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "tools", "load_bootstrap.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "load_fields.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "load_problem_stack.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "tools", "load_matsolvers.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "load_solver_stack.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "tools", "load_output.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "load_evaluation.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "tools", "load_runtime.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "load_models.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "extras", "load_extras.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "tools", "load_pretty_printing.jl"))
end
