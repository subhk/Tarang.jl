# ONE statement of "which axes are decomposed".
#
# This convention used to be re-derived by hand at nine call sites. Nine copies
# of one rule is how the PencilArrays convention (decompose LAST mesh dims) and
# the TransposableField convention (decompose FIRST mesh dims) drifted apart.
# This file pins the single source of truth and ratchets against new copies.
using Test
using Tarang

# A Distributor stand-in: decomposed_axes reads only these three fields, so the
# convention can be tested for mesh/ndim combinations that need no live MPI
# world (the duck-typed-fake-dist trick from test_netcdf_slab_geometry.jl).
struct FakeDist
    size::Int
    mesh::Union{Nothing, Tuple{Vararg{Int}}}
    use_pencil_arrays::Bool
end

@testset "decomposed_axes" begin

    @testset "serial and unmeshed decompose nothing" begin
        @test Tarang.decomposed_axes(FakeDist(1, (4,), true), 3) == ()
        @test Tarang.decomposed_axes(FakeDist(4, nothing, true), 3) == ()
        @test Tarang.decomposed_axes(FakeDist(1, (2, 2), false), 2) == ()
    end

    @testset "PencilArrays decomposes the LAST mesh dims" begin
        @test Tarang.decomposed_axes(FakeDist(4, (4,), true), 2) == (2,)
        @test Tarang.decomposed_axes(FakeDist(4, (4,), true), 3) == (3,)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 3) == (2, 3)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 2) == (1, 2)
    end

    @testset "TransposableField decomposes the FIRST mesh dims, at most two" begin
        @test Tarang.decomposed_axes(FakeDist(4, (4, 1), false), 2) == (1, 2)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), false), 3) == (1, 2)
        @test Tarang.decomposed_axes(FakeDist(2, (2,), false), 2) == (1,)
    end

    @testset "pencil path cannot decompose more dims than the field has" begin
        # get_local_array_size leaves the shape untouched when ndim < length(mesh);
        # decomposed_axes must agree or the allocator and the index math diverge.
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 1) == ()
    end

    @testset "mesh_axis_for inverts decomposed_axes" begin
        d = FakeDist(4, (2, 2), true)
        @test Tarang.mesh_axis_for(d, 3, 1) === nothing
        @test Tarang.mesh_axis_for(d, 3, 2) == 1
        @test Tarang.mesh_axis_for(d, 3, 3) == 2
        @test Tarang.is_decomposed_axis(d, 3, 3)
        @test !Tarang.is_decomposed_axis(d, 3, 1)

        t = FakeDist(4, (2, 2), false)
        @test Tarang.mesh_axis_for(t, 3, 1) == 1
        @test Tarang.mesh_axis_for(t, 3, 2) == 2
        @test Tarang.mesh_axis_for(t, 3, 3) === nothing
    end

    @testset "out-of-range axes are not decomposed" begin
        d = FakeDist(4, (2, 2), true)
        @test Tarang.mesh_axis_for(d, 3, 0) === nothing
        @test Tarang.mesh_axis_for(d, 3, 4) === nothing
    end
end

@testset "get_local_range agrees with the convention" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    # Serial: every axis is whole.
    for axis in 1:3
        @test Tarang.get_local_range(dist, 12, axis) == (1, 12)
    end
end

@testset "allocator and index math agree on every axis" begin
    # get_local_array_size decides the ALLOCATED shape; local_indices decides
    # which global indices those slots mean. If they disagree the field is
    # silently mis-addressed — no error, wrong values.
    #
    # SERIAL SMOKE CHECK ONLY: at mesh=(1,) with size==1, every function below
    # takes its identity early-return, so this exercises none of the
    # decomposed branches. The load-bearing three-way agreement assertion —
    # under LIVE decomposition, np=2 and np=4 — lives in
    # test/test_mpi_local_indices.jl.
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    gshape = (8, 6, 4)
    local_shape = Tarang.get_local_array_size(dist, gshape)
    for axis in 1:3
        @test length(Tarang.local_indices(dist, axis, gshape[axis])) == local_shape[axis]
    end
    @test collect(Tarang.compute_local_shape(dist, gshape)) == collect(local_shape)
end

@testset "the convention is stated in exactly one place" begin
    # Nine independently-maintained copies of this rule are how the PencilArrays
    # and TransposableField conventions drifted apart, and how two of them ended
    # up disagreeing about a field with fewer dims than the mesh. A tenth copy
    # must fail the build, not wait for the next audit.
    srcdir = joinpath(@__DIR__, "..", "src")
    allowed = joinpath("core", "distributor", "distributor_core.jl")

    # The tell is a use_pencil_arrays branch that decides axis indices, restated
    # in prose as "decompose(s) [the] LAST/FIRST ..." — in EITHER word order.
    # A single verb-first alternative with no fixed suffix after LAST/FIRST
    # already subsumes the subject-first case as a substring match (regex
    # `occursin` doesn't require the whole sentence, just some contiguous
    # span), but both orders are spelled out explicitly so the check does not
    # depend on that being true forever:
    #   - verb-first:    "decompose(s)/(d) [the] LAST/FIRST ..."
    #   - subject-first: "mesh decompose(s)/(d) the LAST/FIRST ..."
    # This replaces an earlier regex (`decompose\s+(LAST|FIRST)\s+\w*mesh`)
    # that additionally required a mesh-suffixed word immediately after
    # LAST/FIRST, e.g. "decompose LAST ndims_mesh dimensions". That missed
    # plain prose restatements with no such word — e.g. "mesh decomposes the
    # LAST dimensions" — which is real text nonlinear_pencil_utils.jl used to
    # carry before being migrated onto decomposed_axes (see git history).
    convention_re = r"decompose[sd]?\s+(the\s+)?(LAST|FIRST)|mesh\s+decompose[sd]?\s+the\s+(LAST|FIRST)"i

    offenders = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        occursin(allowed, path) && continue
        text = read(path, String)
        if occursin(r"use_pencil_arrays"i, text) && occursin(convention_re, text)
            push!(offenders, relpath(path, srcdir))
        end
    end

    @test isempty(offenders)
    isempty(offenders) || @info "convention re-derived outside decomposed_axes" offenders
end
