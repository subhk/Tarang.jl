using Test

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const TARANG_SOURCE = read(joinpath(ROOT_DIR, "src", "Tarang.jl"), String)

@testset "Root module structure" begin
    @test occursin("include(\"load_order.jl\")", TARANG_SOURCE)
    @test occursin("include(\"public_api.jl\")", TARANG_SOURCE)
    @test occursin("include(\"runtime_init.jl\")", TARANG_SOURCE)

    @test !occursin("abstract type AbstractNonlinearEvaluator", TARANG_SOURCE)
    @test !occursin("struct PencilConfig", TARANG_SOURCE)
    @test !occursin("include(\"core/coords.jl\")", TARANG_SOURCE)
    @test !occursin("include(\"tools/netcdf_output.jl\")", TARANG_SOURCE)
    @test !occursin("function __init__", TARANG_SOURCE)
    @test !occursin("# Public interface\nexport", TARANG_SOURCE)

    @test isfile(joinpath(ROOT_DIR, "src", "load_order.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "core", "module_contracts.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "public_api.jl"))
    @test isfile(joinpath(ROOT_DIR, "src", "runtime_init.jl"))
end
