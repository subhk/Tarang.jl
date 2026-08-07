using Test, Tarang, LinearAlgebra

# Coverage tests for src/extras/flow_tools/flow_tools_domain_utils.jl
#
# Targets get_domain_size, get_domain_bounds, get_fourier_shape.
# The first two are intentionally duck-typed (untyped `domain` argument,
# guarded by `hasfield`), so we exercise both the real-Domain happy path
# and the many fallback branches with lightweight mock structs.

# ---------------------------------------------------------------------------
# NO MOCKS. This file used to define six stand-in structs (NoBasesDomain,
# NilBasesDomain, MockDomain, MockMeta, MockMetaBasis, MockDirectBasis,
# MockUnknownBasis) whose sole purpose was to reach `hasfield` fallback branches
# in get_domain_size/get_domain_bounds. Those branches could not be reached from
# the package: `Domain.bases` is `Tuple{Vararg{Basis}}` (so it is never `nothing`
# and holds no `nothing` entries) and `BasisMeta.bounds` is
# `Tuple{Float64, Float64}` (so it is never `nothing` and never shorter than 2).
# The mocks were the only callers those branches ever had, and the branches are
# gone. Both functions now take `::Union{Nothing, Domain}`, so a stand-in is a
# MethodError — asserted below.
# ---------------------------------------------------------------------------

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
    # get_domain_size: the one fallback that is actually reachable
    # -----------------------------------------------------------------------
    @testset "get_domain_size nothing domain -> default" begin
        sz = @test_logs (:warn,) get_domain_size(nothing)
        @test sz == (2π, 2π, 2π)
    end

    @testset "get_domain_size / bounds reject non-domains" begin
        # Both take ::Union{Nothing, Domain}. Anything else is a MethodError at
        # the call, not a 2π guess several branches later.
        @test_throws MethodError get_domain_size(42)
        @test_throws MethodError get_domain_size(Dict("bases" => nothing))
        @test_throws MethodError get_domain_bounds(42)
        @test_throws MethodError get_domain_bounds((bases = nothing,))
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
    # get_domain_bounds: the one reachable fallback
    # -----------------------------------------------------------------------
    @testset "get_domain_bounds nothing domain -> default" begin
        b = get_domain_bounds(nothing)
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
