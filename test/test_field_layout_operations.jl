using Test
using Tarang
using Random

# ============================================================================
# Tests for src/core/field/field_layout/field_layout_operations.jl
#
# This file holds the field-data layout *operations* that sit on top of the
# layout-aware storage: fill_random!, the MPI-reproducible random fill helper,
# integrate (quadrature over the domain), and the VectorField/TensorField
# convenience accessors (getindex/setindex!/getproperty/propertynames).
#
# ORACLE NOTE: Every numeric expectation here is derived from the spectral-
# method definitions, NOT from the function's own output:
#   * Layout round-trip: a band-limited Fourier function / low-degree Chebyshev
#     polynomial must survive :g -> :c -> :g exactly (to rounding).
#   * integrate(): the quadrature weights are uniform L/N on a periodic Fourier
#     axis and Clenshaw-Curtis (summing to L) on a Chebyshev axis. Hence the
#     integral of a constant c over an axis of length L is exactly c*L, and a
#     product domain multiplies the per-axis integrals. A band-limited Fourier
#     mode with no zero-frequency content integrates to 0.
#   * fill_random! "uniform" produces values in [-1, 1]; scale=0 zeroes the
#     field; identical seeds give identical fields (determinism).
# ============================================================================

@testset "field_layout_operations.jl" begin

    # ------------------------------------------------------------------
    # 1. ensure_layout! round-trip + idempotence (used by every op here)
    # ------------------------------------------------------------------
    @testset "ensure_layout! round-trip and idempotence" begin
        # --- 1D Fourier ---
        dom1 = PeriodicDomain(16)
        f = ScalarField(dom1, "f")
        # Band-limited: modes 1 and 3 are well within Nyquist for N=16.
        set!(f, (x,) -> sin(x) + 0.5*cos(3x))
        orig = copy(Tarang.get_grid_data(f))
        @test f.current_layout == :g

        ensure_layout!(f, :c)
        @test f.current_layout == :c
        ensure_layout!(f, :g)
        @test f.current_layout == :g
        # Round-trip is exact for a band-limited function (independent oracle:
        # the spectral transform pair is the identity on representable modes).
        @test isapprox(Tarang.get_grid_data(f), orig; rtol=1e-10, atol=1e-12)

        # Idempotence in :g: calling again is a no-op and preserves data.
        snap_g = copy(Tarang.get_grid_data(f))
        ensure_layout!(f, :g)
        @test f.current_layout == :g
        @test Tarang.get_grid_data(f) == snap_g

        # Idempotence in :c.
        ensure_layout!(f, :c)
        snap_c = copy(Tarang.get_coeff_data(f))
        ensure_layout!(f, :c)
        @test f.current_layout == :c
        @test Tarang.get_coeff_data(f) == snap_c

        # --- 2D Fourier x Fourier ---
        dom2 = PeriodicDomain(16, 16)
        g = ScalarField(dom2, "g")
        set!(g, (x,y) -> sin(x)*cos(2y))
        orig2 = copy(Tarang.get_grid_data(g))
        ensure_layout!(g, :c); ensure_layout!(g, :g)
        @test g.current_layout == :g
        @test isapprox(Tarang.get_grid_data(g), orig2; rtol=1e-10, atol=1e-12)

        # --- Mixed Fourier x Chebyshev (channel) ---
        # Low-degree polynomial in z is represented exactly by Chebyshev T_n.
        ch = ChannelDomain(16, 12; Lx=2pi, Lz=2.0)
        h = ScalarField(ch, "h")
        set!(h, (x,z) -> (z^2 - 0.5*z + 1.0) * sin(x))
        orig3 = copy(Tarang.get_grid_data(h))
        ensure_layout!(h, :c); ensure_layout!(h, :g)
        @test h.current_layout == :g
        @test isapprox(Tarang.get_grid_data(h), orig3; rtol=1e-9, atol=1e-10)

        # --- 3D Fourier round-trip ---
        dom3 = PeriodicDomain(8, 8, 8)
        k = ScalarField(dom3, "k")
        set!(k, (x,y,z) -> cos(x)*sin(y+z))
        orig4 = copy(Tarang.get_grid_data(k))
        ensure_layout!(k, :c); ensure_layout!(k, :g)
        @test isapprox(Tarang.get_grid_data(k), orig4; rtol=1e-10, atol=1e-12)

        # --- VectorField ensure_layout! applies to all components ---
        u = VectorField(dom2, "u")
        set!(u, ((x,y)->sin(x), (x,y)->cos(y)))
        ensure_layout!(u, :c)
        @test all(c -> c.current_layout == :c, u.components)
        ensure_layout!(u, :g)
        @test all(c -> c.current_layout == :g, u.components)
    end

    # ------------------------------------------------------------------
    # 2. integrate(field, axes)
    #    Oracle: weights are uniform L/N (Fourier) and Clenshaw-Curtis
    #    summing to L (Chebyshev). integral of const c over axis = c*L.
    #    FULL integration returns a SCALAR (MPI-reduced); PARTIAL integration
    #    returns a ScalarField over the remaining axes.
    # ------------------------------------------------------------------
    @testset "integrate" begin
        Lx = 2pi

        # --- 1D periodic constant: ∫_0^{2π} c dx = c*2π ---
        dom1 = PeriodicDomain(16)               # x in [0, 2π]
        f = ScalarField(dom1, "f")
        set!(f, 3.0)
        I = integrate(f, :)
        @test I isa Number                      # full integral is a scalar
        @test isapprox(I, 3.0 * Lx; rtol=1e-12)

        # --- 1D periodic band-limited mode integrates to ~0 ---
        set!(f, (x,) -> sin(x) + cos(2x))
        Iz = integrate(f, :)
        @test isapprox(Iz, 0.0; atol=1e-12)

        # --- 2D periodic constant: c * Lx * Ly ---
        dom2 = PeriodicDomain(8, 8)             # [0,2π]^2
        g = ScalarField(dom2, "g")
        set!(g, 3.0)
        Ig = integrate(g, :)
        @test Ig isa Number
        @test isapprox(Ig, 3.0 * Lx * Lx; rtol=1e-12)

        # --- 2D integrate over a single axis only (axis 1 = x) ---
        # Partial integration returns a ScalarField over the remaining (y) axis.
        set!(g, 2.0)
        I1 = integrate(g, 1)
        @test I1 isa ScalarField
        ensure_layout!(I1, :g)
        I1d = vec(Array(get_grid_data(I1)))
        @test length(I1d) == 8
        # Each surviving entry equals 2 * Lx (integral over x of the constant).
        @test all(v -> isapprox(v, 2.0 * Lx; rtol=1e-12), I1d)

        # --- Mixed Fourier x Chebyshev (channel): const over [0,2π]x[0,2] ---
        # Chebyshev Clenshaw-Curtis weights sum to L_z exactly, so ∫ c = c*Lx*Lz.
        ch = ChannelDomain(8, 8; Lx=2pi, Lz=2.0)
        cf = ScalarField(ch, "cf")
        set!(cf, 5.0)
        Ic = integrate(cf, :)
        @test Ic isa Number
        @test isapprox(Ic, 5.0 * 2pi * 2.0; rtol=1e-10)

        # --- integrate auto-transforms from :c layout back to :g first ---
        k = ScalarField(dom2, "k")
        set!(k, 4.0)
        ensure_layout!(k, :c)                  # force coefficient layout
        @test k.current_layout == :c
        Ik = integrate(k, :)
        @test isapprox(Ik, 4.0 * Lx * Lx; rtol=1e-12)
        @test k.current_layout == :g           # integrate restored grid layout

        # --- domain === nothing path returns 0.0 (0-D / tau-style field) ---
        coords = CartesianCoordinates("x", "y")
        dist0 = Distributor(coords; dtype=Float64)
        f0 = ScalarField(dist0, "f0", (), Float64)
        @test f0.domain === nothing
        @test integrate(f0) === 0.0
        @test integrate(f0, :) === 0.0
    end

    # ------------------------------------------------------------------
    # 3. fill_random!(ScalarField)
    # ------------------------------------------------------------------
    @testset "fill_random! ScalarField" begin
        dom = PeriodicDomain(8, 8)

        # --- normal distribution fills grid with nonzero values ---
        f = ScalarField(dom, "f")
        ret = fill_random!(f, "g"; seed=42, distribution="normal", scale=1.0)
        @test ret === f                         # returns the field
        @test f.current_layout == :g
        @test !all(Tarang.get_grid_data(f) .== 0.0)

        # --- determinism: identical seed -> identical field ---
        a = ScalarField(dom, "a"); b = ScalarField(dom, "b")
        fill_random!(a, "g"; seed=123)
        fill_random!(b, "g"; seed=123)
        @test Tarang.get_grid_data(a) == Tarang.get_grid_data(b)

        # --- different seeds -> different fields ---
        c = ScalarField(dom, "c")
        fill_random!(c, "g"; seed=999)
        @test Tarang.get_grid_data(a) != Tarang.get_grid_data(c)

        # --- "standard_normal" is accepted (same branch as "normal") ---
        sn = ScalarField(dom, "sn")
        fill_random!(sn, "g"; seed=5, distribution="standard_normal")
        @test !all(Tarang.get_grid_data(sn) .== 0.0)

        # --- uniform distribution: values strictly within [-1, 1] ---
        uf = ScalarField(dom, "uf")
        fill_random!(uf, "g"; seed=7, distribution="uniform", scale=1.0)
        ud = Tarang.get_grid_data(uf)
        @test minimum(ud) >= -1.0
        @test maximum(ud) <= 1.0
        @test !all(ud .== 0.0)

        # --- scale=0 zeroes every entry (scale is a strict multiplier) ---
        z = ScalarField(dom, "z")
        fill_random!(z, "g"; seed=3, distribution="normal", scale=0.0)
        @test all(Tarang.get_grid_data(z) .== 0.0)

        # --- scale multiplies: scale=2 gives exactly 2x the scale=1 field ---
        s1 = ScalarField(dom, "s1"); s2 = ScalarField(dom, "s2")
        fill_random!(s1, "g"; seed=11, distribution="normal", scale=1.0)
        fill_random!(s2, "g"; seed=11, distribution="normal", scale=2.0)
        @test isapprox(Tarang.get_grid_data(s2), 2.0 .* Tarang.get_grid_data(s1); rtol=1e-12)

        # --- filling the "c" (coefficient) layout sets current_layout = :c ---
        m = ScalarField(dom, "m")
        fill_random!(m, "c"; seed=8)
        @test m.current_layout == :c

        # --- unknown distribution throws ArgumentError ---
        bad = ScalarField(dom, "bad")
        @test_throws ArgumentError fill_random!(bad, "g"; seed=1, distribution="bogus")
    end

    # ------------------------------------------------------------------
    # 4. fill_random!(VectorField)
    #    Components get per-component seeds (seed + (i-1)*1000003), so they
    #    must be uncorrelated (distinct) when seeded.
    # ------------------------------------------------------------------
    @testset "fill_random! VectorField" begin
        dom = PeriodicDomain(8, 8)
        u = VectorField(dom, "u")
        ret = fill_random!(u, "g"; seed=100, distribution="normal", scale=1.0)
        @test ret === u
        d1 = Tarang.get_grid_data(u.components[1])
        d2 = Tarang.get_grid_data(u.components[2])
        @test !all(d1 .== 0.0)
        @test !all(d2 .== 0.0)
        @test d1 != d2                          # per-component seed -> uncorrelated

        # determinism for vectors too
        v = VectorField(dom, "v")
        fill_random!(v, "g"; seed=100)
        @test Tarang.get_grid_data(v.components[1]) == d1
        @test Tarang.get_grid_data(v.components[2]) == d2
    end

    # ------------------------------------------------------------------
    # 5. _fill_random_reproducible! — the MPI-reproducible helper.
    #    The full global-index path requires dist.size > 1 (MPI), which is
    #    unreachable in a serial run. We DO cover its domain===nothing
    #    fallback by direct call (it is reachable for 0-D / domainless fields).
    # ------------------------------------------------------------------
    @testset "_fill_random_reproducible! (domainless fallback)" begin
        coords = CartesianCoordinates("x", "y")
        dist0 = Distributor(coords; dtype=Float64)
        f0 = ScalarField(dist0, "f0", (), Float64)
        @test f0.domain === nothing             # triggers the fallback branch

        # normal: fills nonzero, deterministic for a fixed seed
        data = zeros(Float64, 4, 4)
        Tarang._fill_random_reproducible!(data, f0, "g", 99, "normal", 3.0)
        @test !all(data .== 0.0)
        snap = copy(data)
        Tarang._fill_random_reproducible!(data, f0, "g", 99, "normal", 3.0)
        @test snap == data                      # deterministic (same seed)

        # uniform fallback: values within [-1, 1]
        ud = zeros(Float64, 4, 4)
        Tarang._fill_random_reproducible!(ud, f0, "g", 7, "uniform", 1.0)
        @test minimum(ud) >= -1.0 && maximum(ud) <= 1.0

        # scale=0 zeroes the array
        zd = zeros(Float64, 3, 3)
        Tarang._fill_random_reproducible!(zd, f0, "g", 1, "normal", 0.0)
        @test all(zd .== 0.0)
    end

    # ------------------------------------------------------------------
    # 6. VectorField accessors: getindex(Int)/setindex!(Int)/getindex(String)
    # ------------------------------------------------------------------
    @testset "VectorField indexing accessors" begin
        dom = PeriodicDomain(8, 8)
        u = VectorField(dom, "u")

        # integer getindex returns the component (identity, not a copy)
        @test u[1] isa ScalarField
        @test u[1] === u.components[1]
        @test u[2] === u.components[2]

        # integer setindex! replaces the component
        repl = ScalarField(dom, "repl")
        u[1] = repl
        @test u[1] === repl

        # String getindex returns each component's data in that layout
        set!(u, ((x,y)->1.0, (x,y)->2.0))
        gl = u["g"]
        @test length(gl) == 2
        @test all(d -> d isa AbstractArray, gl)
        @test all(gl[1] .== 1.0)
        @test all(gl[2] .== 2.0)
    end

    # ------------------------------------------------------------------
    # 7. VectorField getproperty / propertynames
    # ------------------------------------------------------------------
    @testset "VectorField getproperty and propertynames" begin
        dom = PeriodicDomain(8, 8)            # coords x, y
        u = VectorField(dom, "u")

        # coordinate-name access maps to components in order
        @test u.x === u.components[1]
        @test u.y === u.components[2]

        # struct fields still reachable through getproperty
        @test u.name == "u"
        @test u.dist === dom.dist
        @test u.dtype == Float64
        @test length(u.components) == 2

        # invalid component name throws ArgumentError
        @test_throws ArgumentError u.w

        # Channel coords are x, z
        ch = ChannelDomain(8, 8; Lx=2pi, Lz=2.0)
        v = VectorField(ch, "v")
        @test v.x === v.components[1]
        @test v.z === v.components[2]

        # propertynames includes both struct fields and coordinate names,
        # for both the public and private (private=true) branches.
        pub = Base.propertynames(u)
        @test :x in pub && :y in pub
        @test :name in pub && :components in pub
        priv = Base.propertynames(u, true)
        @test :x in priv && :y in priv
        @test :name in priv && :components in priv
    end

    # ------------------------------------------------------------------
    # 8. TensorField accessors: getindex(i,j) / setindex!(i,j)
    # ------------------------------------------------------------------
    @testset "TensorField indexing accessors" begin
        dom = PeriodicDomain(8, 8)
        S = TensorField(dom, "S")
        @test size(S.components) == (2, 2)

        # getindex returns the (i,j) component (identity)
        @test S[1, 1] isa ScalarField
        @test S[1, 1] === S.components[1, 1]
        @test S[2, 1] === S.components[2, 1]

        # setindex! replaces the (i,j) component
        repl = ScalarField(dom, "repl")
        S[1, 2] = repl
        @test S[1, 2] === repl
        @test S[1, 2] === S.components[1, 2]
    end
end
