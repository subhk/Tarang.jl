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
    # silently mis-addressed — no error, wrong values. Nothing forced them to
    # agree before this test existed.
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    gshape = (8, 6, 4)
    local_shape = Tarang.get_local_array_size(dist, gshape)
    for axis in 1:3
        @test length(Tarang.local_indices(dist, axis, gshape[axis])) == local_shape[axis]
    end
    @test collect(Tarang.compute_local_shape(dist, gshape)) == collect(local_shape)
end
