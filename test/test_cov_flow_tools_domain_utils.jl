using Test, Tarang, LinearAlgebra

# Coverage tests for src/extras/flow_tools/flow_tools_domain_utils.jl
#
# Targets get_domain_size, get_domain_bounds, get_fourier_shape.
# The first two are intentionally duck-typed (untyped `domain` argument,
# guarded by `hasfield`), so we exercise both the real-Domain happy path
# and the many fallback branches with lightweight mock structs.

# ---------------------------------------------------------------------------
# Mock domain/basis structs to reach the duck-typed fallback branches.
# The real `Domain` filters out `nothing` bases and always has `:bases`, so
# the only way to reach those branches is via mocks matching the field shapes
# the functions probe with `hasfield`.
# ---------------------------------------------------------------------------

# A mock with NO `:bases` field at all.
struct NoBasesDomain
    something_else::Int
end

# A mock whose `bases` is `nothing`.
struct NilBasesDomain
    bases::Any
end

# A mock domain wrapping an arbitrary iterable of "bases".
struct MockDomain
    bases::Any
end

# A mock basis-meta carrying bounds (matches `basis.meta.bounds`).
struct MockMeta
    bounds::Any
end

# A mock basis that exposes `.meta.bounds`.
struct MockMetaBasis
    meta::MockMeta
end

# A mock basis that exposes a direct `.bounds` field (no `.meta`).
struct MockDirectBasis
    bounds::Any
end

# A mock basis with neither `.meta` nor `.bounds` (unknown type).
struct MockUnknownBasis
    label::String
end

@testset "flow_tools_domain_utils coverage" begin

    # -----------------------------------------------------------------------
    # get_domain_size: real domains (happy path, lines 35-48,64,66,70)
    # -----------------------------------------------------------------------
    @testset "get_domain_size real 1D Chebyshev" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["x"]; size=16, bounds=(-1.0, 1.0))
        dom = Domain(dist, (zb,))
        sz = get_domain_size(dom)
        @test sz isa Tuple
        @test length(sz) == 1
        @test sz[1] ≈ 2.0           # 1.0 - (-1.0)
    end

    @testset "get_domain_size real 1D Fourier custom extent" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        dom = Domain(dist, (xb,))
        sz = get_domain_size(dom)
        @test length(sz) == 1
        @test sz[1] ≈ 2π
    end

    @testset "get_domain_size real 2D mixed" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 4.0))
        yb = ChebyshevT(coords["y"]; size=12, bounds=(2.0, 5.0))
        dom = Domain(dist, (xb, yb))
        sz = get_domain_size(dom)
        @test length(sz) == 2
        @test all(s -> s > 0, sz)
        # extents are 4.0 and 3.0 in some axis order; assert as a set
        @test sort(collect(sz)) ≈ [3.0, 4.0]
    end

    # -----------------------------------------------------------------------
    # get_domain_size: fallback branches via mocks
    # -----------------------------------------------------------------------
    @testset "get_domain_size nothing domain -> default" begin
        sz = get_domain_size(nothing)
        @test sz == (2π, 2π, 2π)
    end

    @testset "get_domain_size no :bases field -> default" begin
        sz = get_domain_size(NoBasesDomain(7))
        @test sz == (2π, 2π, 2π)
    end

    @testset "get_domain_size :bases === nothing -> default" begin
        sz = get_domain_size(NilBasesDomain(nothing))
        @test sz == (2π, 2π, 2π)
    end

    @testset "get_domain_size nothing-basis element -> push default 2π" begin
        # A basis entry that is `nothing` pushes a 2π default and continues.
        good = MockMetaBasis(MockMeta((0.0, 3.0)))
        dom = MockDomain([nothing, good])
        sz = get_domain_size(dom)
        @test length(sz) == 2
        @test sz[1] ≈ 2π            # default for the nothing entry
        @test sz[2] ≈ 3.0          # extent of the good basis
    end

    @testset "get_domain_size meta.bounds too short -> default" begin
        # bounds present but length < 2 hits the else branch (line 50).
        dom = MockDomain([MockMetaBasis(MockMeta((1.0,)))])
        sz = get_domain_size(dom)
        @test sz == (2π,)
    end

    @testset "get_domain_size meta.bounds === nothing -> default" begin
        dom = MockDomain([MockMetaBasis(MockMeta(nothing))])
        sz = get_domain_size(dom)
        @test sz == (2π,)
    end

    @testset "get_domain_size direct :bounds field" begin
        # No `.meta`, but a direct `.bounds` field (lines 52-57).
        dom = MockDomain([MockDirectBasis((1.0, 4.5))])
        sz = get_domain_size(dom)
        @test length(sz) == 1
        @test sz[1] ≈ 3.5
    end

    @testset "get_domain_size direct :bounds too short -> default" begin
        # Direct bounds present but length < 2 (line 59).
        dom = MockDomain([MockDirectBasis((2.0,))])
        sz = get_domain_size(dom)
        @test sz == (2π,)
    end

    @testset "get_domain_size unknown basis type -> default" begin
        # Neither `.meta` nor `.bounds` (line 62).
        dom = MockDomain([MockUnknownBasis("mystery")])
        sz = get_domain_size(dom)
        @test sz == (2π,)
    end

    @testset "get_domain_size empty bases -> 3D default" begin
        # Iterable but empty -> isempty(sizes) (line 67).
        dom = MockDomain(Any[])
        sz = get_domain_size(dom)
        @test sz == (2π, 2π, 2π)
    end

    # -----------------------------------------------------------------------
    # get_domain_bounds: real domains (happy path, lines 91-102,116,118,122)
    # -----------------------------------------------------------------------
    @testset "get_domain_bounds real 1D Chebyshev" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["x"]; size=16, bounds=(-1.0, 1.0))
        dom = Domain(dist, (zb,))
        b = get_domain_bounds(dom)
        @test b isa Vector
        @test length(b) == 1
        @test b[1] == (-1.0, 1.0)
        @test eltype(b) == Tuple{Float64, Float64}
    end

    @testset "get_domain_bounds real 2D mixed" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 4.0))
        yb = ChebyshevT(coords["y"]; size=12, bounds=(2.0, 5.0))
        dom = Domain(dist, (xb, yb))
        b = get_domain_bounds(dom)
        @test length(b) == 2
        # bounds appear in axis order; assert as a set of (min,max) tuples
        @test Set(b) == Set([(0.0, 4.0), (2.0, 5.0)])
        # extents derived from bounds must match get_domain_size
        sz = get_domain_size(dom)
        @test Set(hi - lo for (lo, hi) in b) == Set(sz)
    end

    # -----------------------------------------------------------------------
    # get_domain_bounds: fallback branches via mocks
    # -----------------------------------------------------------------------
    @testset "get_domain_bounds nothing domain -> default" begin
        b = get_domain_bounds(nothing)
        @test b == [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    @testset "get_domain_bounds no :bases field -> default" begin
        b = get_domain_bounds(NoBasesDomain(3))
        @test b == [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    @testset "get_domain_bounds :bases === nothing -> default" begin
        b = get_domain_bounds(NilBasesDomain(nothing))
        @test b == [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    @testset "get_domain_bounds nothing-basis element -> push default" begin
        good = MockMetaBasis(MockMeta((1.0, 6.0)))
        dom = MockDomain([nothing, good])
        b = get_domain_bounds(dom)
        @test length(b) == 2
        @test b[1] == (0.0, 2π)
        @test b[2] == (1.0, 6.0)
    end

    @testset "get_domain_bounds meta.bounds too short -> default" begin
        dom = MockDomain([MockMetaBasis(MockMeta((1.0,)))])
        b = get_domain_bounds(dom)
        @test b == [(0.0, 2π)]
    end

    @testset "get_domain_bounds meta.bounds === nothing -> default" begin
        dom = MockDomain([MockMetaBasis(MockMeta(nothing))])
        b = get_domain_bounds(dom)
        @test b == [(0.0, 2π)]
    end

    @testset "get_domain_bounds direct :bounds field" begin
        dom = MockDomain([MockDirectBasis((-2.0, 3.0))])
        b = get_domain_bounds(dom)
        @test b == [(-2.0, 3.0)]
    end

    @testset "get_domain_bounds direct :bounds too short -> default" begin
        dom = MockDomain([MockDirectBasis((2.0,))])
        b = get_domain_bounds(dom)
        @test b == [(0.0, 2π)]
    end

    @testset "get_domain_bounds direct :bounds === nothing -> default" begin
        dom = MockDomain([MockDirectBasis(nothing)])
        b = get_domain_bounds(dom)
        @test b == [(0.0, 2π)]
    end

    @testset "get_domain_bounds unknown basis type -> default" begin
        dom = MockDomain([MockUnknownBasis("mystery")])
        b = get_domain_bounds(dom)
        @test b == [(0.0, 2π)]
    end

    @testset "get_domain_bounds empty bases -> 3D default" begin
        dom = MockDomain(Any[])
        b = get_domain_bounds(dom)
        @test b == [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    # -----------------------------------------------------------------------
    # get_fourier_shape: real VectorField (lines 126-129)
    # -----------------------------------------------------------------------
    @testset "get_fourier_shape real 2D vector field" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        u = VectorField(dist, "u", (xb, yb), Float64)

        # Establish a known grid state, then query the coeff-layout shape.
        for c in u.components
            ensure_layout!(c, :g)
            fill!(get_grid_data(c), 0.0)
        end

        shp = get_fourier_shape(u, [1, 2])
        # Must equal the coeff-data size of the first component.
        first_comp = u.components[1]
        ensure_layout!(first_comp, :c)
        @test shp == size(get_coeff_data(first_comp))
        @test shp isa Tuple
        @test length(shp) == 2
        @test all(d -> d > 0, shp)
    end
end
