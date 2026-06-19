using Test, Tarang, LinearAlgebra

# Coverage tests for src/core/field/field_data/field_data_scales.jl
# Focus: serial-CPU scale management + spectral resampling helpers.
# All expected values are analytic / round-trip / invariant based.

# ---- helpers -------------------------------------------------------------
# Uniform periodic node positions for a RealFourier axis of length M on [0,2pi).
fourier_nodes(M) = [2pi * (i - 1) / M for i in 1:M]
# Band-limited test signal: only modes 2 and 3 present (so a grid with
# Nyquist >= 4, i.e. M >= 8, represents it exactly).
band_signal(x) = sin(2 * x) + 0.5 * cos(3 * x)

@testset "field_data_scales coverage" begin

    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)

    # =====================================================================
    # resample_1d!  (spectral, FFT-based) — round-trips & invariants
    # =====================================================================
    @testset "resample_1d! spectral" begin
        N = 16
        x  = fourier_nodes(N)
        od = band_signal.(x)

        # Upsample 16 -> 32: values land on the finer uniform grid exactly.
        up = zeros(Float64, 32)
        resample_1d!(up, od)
        ana_up = band_signal.(fourier_nodes(32))
        @test maximum(abs.(up .- ana_up)) < 1e-12

        # Downsample 32 -> 16 recovers the original (band-limited) samples.
        back = zeros(Float64, 16)
        resample_1d!(back, up)
        @test maximum(abs.(back .- od)) < 1e-12

        # Direct downsample 16 -> 8 (still band-limited): exact on coarse grid.
        down = zeros(Float64, 8)
        resample_1d!(down, od)
        ana_down = band_signal.(fourier_nodes(8))
        @test maximum(abs.(down .- ana_down)) < 1e-12

        # Odd-N upsampling path (no Nyquist bin; copies highest positive freq).
        Nodd = 15
        odd_old = [sin(2 * t) for t in fourier_nodes(Nodd)]
        odd_up = zeros(Float64, 30)
        resample_1d!(odd_up, odd_old)
        @test maximum(abs.(odd_up .- [sin(2 * t) for t in fourier_nodes(30)])) < 1e-12

        # n_old == 1: fill with the single value.
        single = zeros(Float64, 5)
        resample_1d!(single, [3.7])
        @test all(single .== 3.7)

        # n_new == 1: collapse to first sample.
        one = zeros(Float64, 1)
        resample_1d!(one, od)
        @test one[1] == od[1]

        # Equal sizes: straight copy.
        eq = zeros(Float64, N)
        resample_1d!(eq, od)
        @test eq == od

        # Complex eltype branch (keeps imaginary part).
        cnew = zeros(ComplexF64, 8)
        cold = ComplexF64.(od)
        resample_1d!(cnew, cold)
        @test eltype(cnew) <: Complex
        @test maximum(abs.(cnew .- ComplexF64.(ana_down))) < 1e-12
    end

    # =====================================================================
    # resample_linear_1d!  (linear interpolation fallback)
    # =====================================================================
    @testset "resample_linear_1d!" begin
        # Linear ramp interpolates exactly with linear interpolation.
        ln = zeros(Float64, 7)
        resample_linear_1d!(ln, [0.0, 1.0, 2.0, 3.0])
        @test ln ≈ [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        # Endpoints preserved.
        @test ln[1] == 0.0
        @test ln[end] == 3.0

        # n_old == 1 path: fill.
        s = zeros(Float64, 4)
        resample_linear_1d!(s, [5.0])
        @test all(s .== 5.0)

        # n_new == 1 path: single output gets first value (t maps to start).
        one = zeros(Float64, 1)
        resample_linear_1d!(one, [9.0, 8.0, 7.0])
        @test all(one .== 9.0)
    end

    # =====================================================================
    # resample_2d! / resample_3d!  (separable spectral)
    # =====================================================================
    @testset "resample_2d! / resample_3d!" begin
        sig2(i, j, M) = sin(2pi * (i - 1) / M) * cos(2pi * (j - 1) / M)
        old2 = [sig2(i, j, 8) for i in 1:8, j in 1:8]
        new2 = zeros(Float64, 16, 16)
        resample_2d!(new2, old2)
        ana2 = [sig2(i, j, 16) for i in 1:16, j in 1:16]
        @test maximum(abs.(new2 .- ana2)) < 1e-12

        # Equal sizes copy path.
        eq2 = zeros(Float64, 8, 8)
        resample_2d!(eq2, old2)
        @test eq2 == old2

        sig3(i, j, k, M) = sin(2pi*(i-1)/M) * cos(2pi*(j-1)/M) * sin(2pi*(k-1)/M)
        old3 = [sig3(i, j, k, 4) for i in 1:4, j in 1:4, k in 1:4]
        new3 = zeros(Float64, 8, 8, 8)
        resample_3d!(new3, old3)
        ana3 = [sig3(i, j, k, 8) for i in 1:8, j in 1:8, k in 1:8]
        @test maximum(abs.(new3 .- ana3)) < 1e-12

        eq3 = zeros(Float64, 4, 4, 4)
        resample_3d!(eq3, old3)
        @test eq3 == old3
    end

    # =====================================================================
    # resample_nearest!  (arbitrary dimensions)
    # =====================================================================
    @testset "resample_nearest!" begin
        old4 = reshape(collect(1.0:16.0), 2, 2, 2, 2)
        new4 = zeros(Float64, 4, 4, 4, 4)
        resample_nearest!(new4, old4)
        # Corners map to corners of the source array.
        @test new4[1, 1, 1, 1] == old4[1, 1, 1, 1]
        @test new4[4, 4, 4, 4] == old4[2, 2, 2, 2]
        # Every output value comes from the source set (nearest-neighbor).
        @test all(in(Set(old4)), new4)

        # Degenerate axis (size 1) clamps to index 1.
        old_deg = reshape([2.0, 4.0], 1, 2)
        new_deg = zeros(Float64, 3, 4)
        resample_nearest!(new_deg, old_deg)
        @test all(in(Set(old_deg)), new_deg)
    end

    # =====================================================================
    # resample_grid_data!  (top-level dispatcher)
    # =====================================================================
    @testset "resample_grid_data! dispatch" begin
        # 1D route + spectral accuracy on a band-limited signal.
        od = band_signal.(fourier_nodes(16))
        g1 = zeros(Float64, 32)
        resample_grid_data!(g1, od, (16,), (32,))
        @test maximum(abs.(g1 .- band_signal.(fourier_nodes(32)))) < 1e-12

        # Equal-size copy short-circuit.
        ge = zeros(Float64, 6)
        src = collect(1.0:6.0)
        resample_grid_data!(ge, src, (6,), (6,))
        @test ge == src

        # Dimension mismatch -> zero-fill (and warns).
        gm = ones(Float64, 4, 4)
        resample_grid_data!(gm, collect(1.0:6.0), (6,), (4, 4))
        @test all(gm .== 0)

        # 2D / 3D / 4D-nearest routing.
        g2 = zeros(Float64, 4, 4)
        resample_grid_data!(g2, ones(Float64, 2, 2), (2, 2), (4, 4))
        @test size(g2) == (4, 4)
        @test all(g2 .≈ 1.0)   # constant resamples to constant

        g3 = zeros(Float64, 4, 4, 4)
        resample_grid_data!(g3, ones(Float64, 2, 2, 2), (2, 2, 2), (4, 4, 4))
        @test size(g3) == (4, 4, 4)

        g4 = zeros(Float64, 4, 4, 4, 4)
        resample_grid_data!(g4, fill(7.0, 2, 2, 2, 2), (2, 2, 2, 2), (4, 4, 4, 4))
        @test all(g4 .== 7.0)
    end

    # =====================================================================
    # get_scaled_shape / get_coefficient_shape / dealias_scales
    # =====================================================================
    @testset "scaled-shape helpers" begin
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
        f = ScalarField(Domain(dist, (xb,)), "f")
        ensure_layout!(f, :g)

        @test get_scaled_shape(f) == (16,)
        @test get_scaled_shape(f, (1.5,)) == (24,)       # ceil(16*1.5)
        @test get_scaled_shape(f, nothing) == (16,)      # nothing -> 1.0
        @test get_scaled_shape(f, ()) == (16,)           # missing -> default 1.0
        @test get_scaled_shape(f, (2.0,)) == (32,)

        # RealFourier coefficient shape is N/2+1 on the serial CPU path.
        @test get_coefficient_shape(f) == (9,)

        # Dealias (3/2) scales helper.
        @test dealias_scales(f) == (1.5,)
        apply_dealiasing_scales!(f)
        @test f.scales == (1.5,)
        @test size(get_grid_data(f)) == (24,)
    end

    # =====================================================================
    # set_scales! / change_scales! — Fourier grid-space resample
    # =====================================================================
    @testset "set_scales! Fourier grid-space" begin
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
        f = ScalarField(Domain(dist, (xb,)), "f")
        ensure_layout!(f, :g)
        gd = get_grid_data(f)
        gd .= band_signal.(fourier_nodes(16))

        # Upsample 1.0 -> 2.0 : grid buffer grows and matches analytic samples.
        set_scales!(f, 2.0)
        @test f.scales == (2.0,)
        gd2 = get_grid_data(f)
        @test length(gd2) == 32
        @test maximum(abs.(gd2 .- band_signal.(fourier_nodes(32)))) < 1e-12

        # Downsample 2.0 -> 0.5 : still band-limited (modes 2,3 < Nyquist 4).
        set_scales!(f, 0.5)
        gd3 = get_grid_data(f)
        @test length(gd3) == 8
        @test maximum(abs.(gd3 .- band_signal.(fourier_nodes(8)))) < 1e-12

        # No-op when scales unchanged: returns same field, data untouched.
        before = copy(get_grid_data(f))
        r = set_scales!(f, 0.5)
        @test r === f
        @test get_grid_data(f) == before

        # change_scales! alias drives the same path.
        change_scales!(f, 1.0)
        @test f.scales == (1.0,)
        @test length(get_grid_data(f)) == 16
    end

    # =====================================================================
    # set_scales! — Chebyshev basis-aware (spectral) resample
    # =====================================================================
    @testset "set_scales! Chebyshev basis-aware" begin
        zb = ChebyshevT(coords["x"]; size=16, bounds=(-1.0, 1.0))
        fc = ScalarField(Domain(dist, (zb,)), "fc")
        ensure_layout!(fc, :g)

        # Degree-3 polynomial is exactly representable (size 16 modes).
        zc = Tarang.local_grid(zb, dist, 1.0)
        poly(z) = z^3 - 2z + 1
        get_grid_data(fc) .= poly.(zc)

        # Basis-aware resample: forward -> resize grid -> backward regenerates
        # values at the NEW (scaled) Gauss-Lobatto nodes exactly.
        set_scales!(fc, 1.5)
        @test fc.scales == (1.5,)
        @test fc.current_layout == :g
        gd2 = vec(get_grid_data(fc))
        @test length(gd2) == 24                       # ceil(16*1.5)
        zc2 = vec(Tarang.local_grid(zb, dist, 1.5))   # scaled GL nodes
        @test maximum(abs.(gd2 .- poly.(zc2))) < 1e-10

        # Downscale back to 1.0 reproduces the polynomial on the base nodes.
        set_scales!(fc, 1.0)
        gd1 = vec(get_grid_data(fc))
        @test length(gd1) == 16
        zc1 = vec(Tarang.local_grid(zb, dist, 1.0))
        @test maximum(abs.(gd1 .- poly.(zc1))) < 1e-10
    end

    # =====================================================================
    # set_scales! — coefficient-space branch (scales don't touch coeffs)
    # =====================================================================
    @testset "set_scales! coefficient-space" begin
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
        fk = ScalarField(Domain(dist, (xb,)), "fk")
        ensure_layout!(fk, :g)
        get_grid_data(fk) .= cos.(2 .* fourier_nodes(16))
        forward_transform!(fk)                # -> :c
        @test fk.current_layout == :c
        old_coeff = copy(get_coeff_data(fk))

        set_scales!(fk, 2.0)
        @test fk.scales == (2.0,)
        @test fk.current_layout == :c          # stays in coefficient space
        # Coefficients are storage-invariant under scale change.
        @test maximum(abs.(get_coeff_data(fk) .- old_coeff)) < 1e-14

        # require_scales!: no-op when already at target, else delegates.
        require_scales!(fk, 2.0)
        @test fk.scales == (2.0,)
        require_scales!(fk, 1.0)
        @test fk.scales == (1.0,)
    end

    # =====================================================================
    # VectorField scale propagation
    # =====================================================================
    @testset "VectorField scales" begin
        vcoords = CartesianCoordinates("x", "y")
        vdist = Distributor(vcoords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(vcoords["x"]; size=8, bounds=(0.0, 2pi))
        yb = RealFourier(vcoords["y"]; size=8, bounds=(0.0, 2pi))
        vf = VectorField(vdist, vcoords, "v", (xb, yb))
        for c in vf.components
            ensure_layout!(c, :g)
        end

        preset_scales!(vf, 1.5)
        @test all(c.scales == (1.5, 1.5) for c in vf.components)

        change_scales!(vf, 1.0)
        @test all(c.scales == (1.0, 1.0) for c in vf.components)

        set_scales!(vf, 2.0)
        @test all(c.scales == (2.0, 2.0) for c in vf.components)
    end

    # =====================================================================
    # get_local_data dispatch (Array / Nothing serial paths)
    # =====================================================================
    @testset "get_local_data" begin
        a = [1.0, 2.0, 3.0]
        @test get_local_data(a) === a            # plain Array returns itself
        @test get_local_data(nothing) === nothing
    end
end
