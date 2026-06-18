"""
Test suite for src/extras/flow_tools/flow_tools_streamfunction.jl

This module relates a 2D streamfunction ψ to velocity / vorticity. All
expected values here are INDEPENDENT vector-calculus ground truth computed
analytically on a band-limited periodic ψ (so spectral derivatives are
exact), NEVER the module's own output.

Test field (zero mean, band-limited on a (0,2π)² periodic box, so kx=ky=1):
    ψ(x,y) = sin(x)·cos(y)

Analytic derivatives:
    ∂ψ/∂x =  cos(x)·cos(y)
    ∂ψ/∂y = -sin(x)·sin(y)
    ∇²ψ   = -2·sin(x)·cos(y) = -2ψ

Sign conventions DISCOVERED in the module (adopted, not assumed):

  * perp_grad(ψ)  (== ∇⊥):   u_x = -∂ψ/∂y,  u_y = +∂ψ/∂x
      => for ψ=sin(x)cos(y):  u_x = sin(x)sin(y),  u_y = cos(x)cos(y)
      This is divergence-free: ∂u_x/∂x + ∂u_y/∂y = 0.

  * streamfunction(velocity) [periodic path]:
        ω = curl(velocity) = ∂v/∂x - ∂u/∂y       (2D curl, per Curl operator)
        ψ̂ = -ω̂/k²   (streamfunction_spectral_invert)  =>  solves ∇²ψ = ω
      NOTE: the module DOCSTRING states u=∂ψ/∂y, v=-∂ψ/∂x and ω=∇²ψ, but with
      that velocity convention ω = ∂v/∂x - ∂u/∂y = -∇²ψ, so the inversion
      ∇²ψ=ω is inconsistent by a sign with its own docstring. See the
      round-trip tests below for which convention actually round-trips.

  * sqg_streamfunction(θ): ψ = (-Δ)^(-1/2) θ, i.e. ψ̂ = θ̂/|k|.
      For θ=sin(x)cos(y), |k|=√2, so ψ = sin(x)cos(y)/√2.

  * sqg_velocity(θ) = perp_grad(sqg_streamfunction(θ)).
"""

using Test
using Tarang
using LinearAlgebra

const TWO_PI = 2π

# ---------------------------------------------------------------------------
# Domain / field helpers (periodic (0,2π)² so band-limited ψ has kx=ky=1)
# ---------------------------------------------------------------------------

"""Build a 2D RealFourier-RealFourier periodic domain on (0,2π)²."""
function build_periodic_2d(N::Int=16)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, TWO_PI))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, TWO_PI))
    bases = (xb, yb)
    x = local_grid(xb, dist, 1.0)
    y = local_grid(yb, dist, 1.0)
    return coords, dist, bases, x, y
end

"""Fill a ScalarField in grid space from f(x,y)."""
function fill_scalar!(field, x, y, f)
    Tarang.ensure_layout!(field, :g)
    data = Tarang.get_grid_data(field)
    for i in eachindex(x), j in eachindex(y)
        data[i, j] = f(x[i], y[j])
    end
    field.current_layout = :g
    return field
end

"""Fill a VectorField's two components from (fx, fy)."""
function fill_vector!(vel, x, y, fx, fy)
    fill_scalar!(vel.components[1], x, y, fx)
    fill_scalar!(vel.components[2], x, y, fy)
    return vel
end

"""Grid-space values of f(x,y) as a matrix matching field storage."""
function grid_matrix(x, y, f)
    out = zeros(length(x), length(y))
    for i in eachindex(x), j in eachindex(y)
        out[i, j] = f(x[i], y[j])
    end
    return out
end

# Analytic ψ and its derivatives for ψ = sin(x)·cos(y)
psi_fn(x, y)      = sin(x) * cos(y)
dpsi_dx(x, y)     = cos(x) * cos(y)
dpsi_dy(x, y)     = -sin(x) * sin(y)
lap_psi(x, y)     = -2.0 * sin(x) * cos(y)

@testset "Streamfunction module" begin

    # =======================================================================
    # perp_grad: ∇⊥ψ = (-∂ψ/∂y, +∂ψ/∂x)
    # =======================================================================
    @testset "perp_grad (∇⊥) sign convention and values" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ψ = ScalarField(dist, "psi", bases, Float64)
        fill_scalar!(ψ, x, y, psi_fn)

        u = Tarang.perp_grad(ψ)
        @test isa(u, VectorField)
        @test length(u.components) == 2

        Tarang.ensure_layout!(u.components[1], :g)
        Tarang.ensure_layout!(u.components[2], :g)
        ux = Tarang.get_grid_data(u.components[1])
        uy = Tarang.get_grid_data(u.components[2])

        # Independent oracle: u_x = -∂ψ/∂y = +sin(x)sin(y); u_y = ∂ψ/∂x = cos(x)cos(y)
        expected_ux = grid_matrix(x, y, (a, b) -> -dpsi_dy(a, b))
        expected_uy = grid_matrix(x, y, (a, b) ->  dpsi_dx(a, b))

        @test isapprox(ux, expected_ux; rtol=1e-8, atol=1e-10)
        @test isapprox(uy, expected_uy; rtol=1e-8, atol=1e-10)

        # Unicode alias ∇⊥ must agree exactly with perp_grad
        u2 = Tarang.∇⊥(ψ)
        Tarang.ensure_layout!(u2.components[1], :g)
        Tarang.ensure_layout!(u2.components[2], :g)
        @test isapprox(Tarang.get_grid_data(u2.components[1]), ux; rtol=1e-12, atol=1e-12)
        @test isapprox(Tarang.get_grid_data(u2.components[2]), uy; rtol=1e-12, atol=1e-12)
    end

    @testset "perp_grad requires 2D field" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, TWO_PI))
        ψ1d = ScalarField(dist, "psi1d", (xb,), Float64)
        Tarang.ensure_layout!(ψ1d, :g)
        @test_throws ArgumentError Tarang.perp_grad(ψ1d)
    end

    # =======================================================================
    # Divergence-free property of ∇⊥ψ : ∂u_x/∂x + ∂u_y/∂y ≈ 0
    # =======================================================================
    @testset "perp_grad velocity is divergence-free" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ψ = ScalarField(dist, "psi", bases, Float64)
        fill_scalar!(ψ, x, y, psi_fn)

        u = Tarang.perp_grad(ψ)
        div = Tarang.velocity_divergence(u)
        Tarang.ensure_layout!(div, :g)
        div_data = Tarang.get_grid_data(div)

        # Independent oracle: divergence of a perpendicular gradient is exactly 0.
        @test isapprox(div_data, zeros(size(div_data)); atol=1e-9)
    end

    # =======================================================================
    # vorticity ω = curl(velocity) = ∂v/∂x - ∂u/∂y  (independent oracle)
    # Build a velocity directly (NOT from a streamfunction) and check curl.
    # =======================================================================
    @testset "vorticity via curl matches analytic ∂v/∂x - ∂u/∂y" begin
        coords, dist, bases, x, y = build_periodic_2d()
        u = VectorField(dist, coords, "u", bases, Float64)
        # u = sin(y), v = sin(x)  =>  ω = ∂v/∂x - ∂u/∂y = cos(x) - cos(y)
        fill_vector!(u, x, y, (a, b) -> sin(b), (a, b) -> sin(a))

        ω = Tarang.evaluate_operator(Tarang.curl(u))
        Tarang.ensure_layout!(ω, :g)
        ω_data = Tarang.get_grid_data(ω)

        expected = grid_matrix(x, y, (a, b) -> cos(a) - cos(b))
        @test isapprox(ω_data, expected; rtol=1e-8, atol=1e-10)
    end

    # =======================================================================
    # streamfunction_spectral_invert: solves ∇²ψ = ω  (ψ̂ = -ω̂/k²)
    # Independent oracle: feed ω = -2 sin(x)cos(y); ∇²ψ=ω => ψ = sin(x)cos(y).
    #
    # FIXED 2026-06-02: get_2d_wavenumber_grids now uses fft-frequency ordering
    # (0,1,…,N/2,-(N/2-1),…,-1) for a non-first RealFourier axis (full complex FFT
    # layout) instead of monotonic rfft indices, via _is_first_real_fourier_axis.
    # The Poisson divisor k² is now correct, so the inversion and Laplacian
    # self-consistency hold to ~1e-9.
    # =======================================================================
    @testset "streamfunction_spectral_invert solves ∇²ψ = ω" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ω = ScalarField(dist, "omega", bases, Float64)
        # Choose ω = ∇²ψ_target for ψ_target = sin(x)cos(y):  ∇²ψ = -2 sin(x)cos(y)
        fill_scalar!(ω, x, y, lap_psi)

        ψ = Tarang.streamfunction_spectral_invert(ω, true)
        Tarang.ensure_layout!(ψ, :g)
        ψ_data = Tarang.get_grid_data(ψ)

        # ∇²ψ = ω with ω = -2 sin(x)cos(y) gives ψ = sin(x)cos(y) (k=0 gauged to 0).
        expected = grid_matrix(x, y, psi_fn)
        @test isapprox(ψ_data, expected; rtol=1e-8, atol=1e-9)

        # Self-consistency: ∇²(recovered ψ) must reproduce ω (Laplacian oracle).
        lapψ = Tarang.evaluate_operator(Tarang.lap(ψ))
        Tarang.ensure_layout!(lapψ, :g)
        @test isapprox(Tarang.get_grid_data(lapψ),
                       grid_matrix(x, y, lap_psi); rtol=1e-7, atol=1e-9)
    end

    @testset "streamfunction_spectral_invert gauge removes mean" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ω = ScalarField(dist, "omega", bases, Float64)
        fill_scalar!(ω, x, y, lap_psi)

        ψ = Tarang.streamfunction_spectral_invert(ω, true)
        Tarang.ensure_layout!(ψ, :g)
        ψ_data = Tarang.get_grid_data(ψ)
        # Gauge condition: mean(ψ) ≈ 0
        @test isapprox(sum(ψ_data) / length(ψ_data), 0.0; atol=1e-10)
    end

    # =======================================================================
    # Full streamfunction(velocity) pipeline (periodic path).
    #
    # The module computes ω = curl(u) = ∂v/∂x - ∂u/∂y then inverts ∇²ψ = ω.
    # To get a clean round-trip we must build the velocity with the SAME
    # convention the inversion assumes. Solve for that convention:
    #   want recovered ψ == ψ_target.
    #   inversion gives ∇²ψ = ω = ∂v/∂x - ∂u/∂y.
    #   so we need ∂v/∂x - ∂u/∂y = ∇²ψ_target.
    #   Choose u = -∂ψ/∂y, v = +∂ψ/∂x (the ∇⊥ convention, perp_grad):
    #       ∂v/∂x - ∂u/∂y = ∂²ψ/∂x² + ∂²ψ/∂y² = ∇²ψ_target.  ✓
    # Therefore the velocity from perp_grad(ψ_target) round-trips through
    # streamfunction(velocity) back to ψ_target. (Note this is the OPPOSITE
    # convention from the streamfunction docstring's u=∂ψ/∂y,v=-∂ψ/∂x, which
    # would round-trip to -ψ_target.)
    # =======================================================================
    #
    # FIXED 2026-06-02: streamfunction(velocity::VectorField) used to throw a
    # MethodError because get_fourier_basis_info was ::Vector-only and velocity.bases
    # is a Tuple. The method now accepts Union{Tuple, AbstractVector}, so the two
    # tests below run the real round-trip.
    @testset "streamfunction(perp_grad(ψ)) round-trips to ψ" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ψ_target = ScalarField(dist, "psi_target", bases, Float64)
        fill_scalar!(ψ_target, x, y, psi_fn)

        # velocity = ∇⊥ψ_target  (perp_grad convention). With ω=curl(u)=∇²ψ_target
        # and the ∇²ψ=ω inversion, this SHOULD round-trip to ψ_target.
        u = Tarang.perp_grad(ψ_target)

        # Returns ψ ≈ ψ_target (FIXED: get_fourier_basis_info now accepts a Tuple,
        # and the wavenumber-layout fix makes the Poisson inversion correct).
        ψ_rec = Tarang.streamfunction(u; boundary_condition=:periodic, gauge_condition=true)
        Tarang.ensure_layout!(ψ_rec, :g)
        ψ_rec_data = Tarang.get_grid_data(ψ_rec)
        @test isapprox(ψ_rec_data, grid_matrix(x, y, psi_fn); rtol=1e-7, atol=1e-9)
    end

    @testset "streamfunction returns a ScalarField on periodic domain" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ψ_target = ScalarField(dist, "psi_target", bases, Float64)
        fill_scalar!(ψ_target, x, y, psi_fn)
        u = Tarang.perp_grad(ψ_target)
        @test isa(Tarang.streamfunction(u), ScalarField)
    end

    @testset "streamfunction requires 2D velocity" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, TWO_PI))
        u1d = VectorField(dist, coords, "u1d", (xb,), Float64)
        Tarang.ensure_layout!(u1d.components[1], :g)
        @test_throws ArgumentError Tarang.streamfunction(u1d)
    end

    # =======================================================================
    # get_2d_wavenumber_grids: independent oracle on RealFourier (0,2π)² box.
    # RFFT axis-1 wavenumbers: 0..N/2 ; FFT axis-2 wavenumbers: shifted full set.
    # =======================================================================
    #
    # The FIRST RealFourier axis (rfft layout, size N/2+1) is handled correctly:
    # kx = 0,1,...,N/2. The SECOND RealFourier axis is stored in full complex
    # FFT layout (size N) whose CORRECT wavenumbers are 0,1,...,N/2,-(N/2-1),
    # ...,-1 (max |k| = N/2). The broken assertions below pin the correct
    # fft-layout values that get_2d_wavenumber_grids fails to produce.
    @testset "get_2d_wavenumber_grids RealFourier values" begin
        N = 16
        coords, dist, bases, x, y = build_periodic_2d(N)
        f = ScalarField(dist, "f", bases, Float64)
        Tarang.ensure_layout!(f, :c)

        kx_grid, ky_grid = Tarang.get_2d_wavenumber_grids(f)
        # On (0,2π) the scale 2π/L = 1, so kx index == kx value.
        # X axis (first RealFourier) uses rfft layout: indices 0,1,...,N/2.  (CORRECT)
        kx_vec = vec(kx_grid)
        @test kx_vec[1] == 0.0
        @test kx_vec[2] ≈ 1.0
        @test length(kx_vec) == N ÷ 2 + 1
        @test all(diff(kx_vec) .≈ 1.0)        # rfft: contiguous 0..N/2
        @test maximum(kx_vec) ≈ N / 2          # rfft Nyquist

        # Y axis (second RealFourier, full fft layout): CORRECT wavenumbers have
        # max magnitude N/2, with negative frequencies in the upper half.
        ky_vec = vec(ky_grid)
        @test length(ky_vec) == N
        @test ky_vec[1] == 0.0
        # FIXED: fft layout — max |k| == N/2 and negative frequencies present.
        @test maximum(abs.(ky_vec)) ≈ N / 2
        @test any(ky_vec .< 0.0)
    end

    # =======================================================================
    # validate_streamfunction uses the perp_grad convention u=-∂ψ/∂y, v=∂ψ/∂x
    # (the velocity that streamfunction() itself round-trips). Build velocity via
    # perp_grad(ψ) and assert valid.
    # =======================================================================
    @testset "validate_streamfunction accepts perp_grad convention u=-∂ψ/∂y, v=∂ψ/∂x" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ψ = ScalarField(dist, "psi", bases, Float64)
        fill_scalar!(ψ, x, y, psi_fn)

        # Velocity from the module's own canonical reconstruction:
        #   u = -∂ψ/∂y = +sin(x)sin(y),  v = ∂ψ/∂x = cos(x)cos(y)
        u = Tarang.perp_grad(ψ)

        result = Tarang.validate_streamfunction(u, ψ; tolerance=1e-6)
        @test result.valid
        @test result.max_error < 1e-6
        @test result.u_error < 1e-6
        @test result.v_error < 1e-6
    end

    @testset "validate_streamfunction rejects wrong velocity" begin
        coords, dist, bases, x, y = build_periodic_2d()
        ψ = ScalarField(dist, "psi", bases, Float64)
        fill_scalar!(ψ, x, y, psi_fn)

        # A velocity that does NOT match ∂ψ: constant field.
        u = VectorField(dist, coords, "u", bases, Float64)
        fill_vector!(u, x, y, (a, b) -> 1.0, (a, b) -> 1.0)

        result = Tarang.validate_streamfunction(u, ψ; tolerance=1e-6)
        @test result.valid == false
        @test result.max_error > 1e-6
    end

    @testset "validate_streamfunction returns false for 1D velocity" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, TWO_PI))
        u1d = VectorField(dist, coords, "u1d", (xb,), Float64)
        ψ1d = ScalarField(dist, "psi1d", (xb,), Float64)
        Tarang.ensure_layout!(u1d.components[1], :g)
        Tarang.ensure_layout!(ψ1d, :g)
        @test Tarang.validate_streamfunction(u1d, ψ1d) == false
    end

    # =======================================================================
    # SQG: sqg_streamfunction(θ) = (-Δ)^(-1/2) θ  =>  ψ̂ = θ̂/|k|.
    # For θ = sin(x)cos(y) on (0,2π)², |k| = √(1²+1²) = √2.
    # Independent oracle: ψ = sin(x)cos(y)/√2.
    # =======================================================================
    @testset "sqg_streamfunction = (-Δ)^(-1/2) θ" begin
        coords, dist, bases, x, y = build_periodic_2d()
        θ = ScalarField(dist, "theta", bases, Float64)
        fill_scalar!(θ, x, y, psi_fn)  # θ = sin(x)cos(y)

        ψ = Tarang.sqg_streamfunction(θ)
        Tarang.ensure_layout!(ψ, :g)
        ψ_data = Tarang.get_grid_data(ψ)

        kmag = sqrt(2.0)
        expected = grid_matrix(x, y, (a, b) -> sin(a) * cos(b) / kmag)
        @test isapprox(ψ_data, expected; rtol=1e-7, atol=1e-9)
    end

    # =======================================================================
    # SQG velocity: u = ∇⊥ψ = ∇⊥(-Δ)^(-1/2) θ.
    # Independent oracle: compose analytic ψ = θ/√2 then ∇⊥:
    #   u_x = -∂ψ/∂y = +sin(x)sin(y)/√2 ; u_y = ∂ψ/∂x = cos(x)cos(y)/√2.
    # =======================================================================
    @testset "sqg_velocity = ∇⊥(-Δ)^(-1/2) θ" begin
        coords, dist, bases, x, y = build_periodic_2d()
        θ = ScalarField(dist, "theta", bases, Float64)
        fill_scalar!(θ, x, y, psi_fn)  # θ = sin(x)cos(y)

        u = Tarang.sqg_velocity(θ)
        @test isa(u, VectorField)
        Tarang.ensure_layout!(u.components[1], :g)
        Tarang.ensure_layout!(u.components[2], :g)
        ux = Tarang.get_grid_data(u.components[1])
        uy = Tarang.get_grid_data(u.components[2])

        kmag = sqrt(2.0)
        expected_ux = grid_matrix(x, y, (a, b) ->  sin(a) * sin(b) / kmag)
        expected_uy = grid_matrix(x, y, (a, b) ->  cos(a) * cos(b) / kmag)
        @test isapprox(ux, expected_ux; rtol=1e-7, atol=1e-9)
        @test isapprox(uy, expected_uy; rtol=1e-7, atol=1e-9)

        # Cross-check: sqg_velocity == perp_grad(sqg_streamfunction(θ)) exactly.
        ψ = Tarang.sqg_streamfunction(θ)
        u_ref = Tarang.perp_grad(ψ)
        Tarang.ensure_layout!(u_ref.components[1], :g)
        Tarang.ensure_layout!(u_ref.components[2], :g)
        @test isapprox(ux, Tarang.get_grid_data(u_ref.components[1]); rtol=1e-10, atol=1e-12)
        @test isapprox(uy, Tarang.get_grid_data(u_ref.components[2]); rtol=1e-10, atol=1e-12)

        # SQG velocity must also be divergence-free.
        div = Tarang.velocity_divergence(u)
        Tarang.ensure_layout!(div, :g)
        @test isapprox(Tarang.get_grid_data(div), zeros(size(Tarang.get_grid_data(div))); atol=1e-9)
    end

    # =======================================================================
    # apply_streamfunction_bc!: pure boundary-condition stencil helper.
    # Independent oracle: assert the documented edge values directly.
    # =======================================================================
    @testset "apply_streamfunction_bc!" begin
        @testset ":no_slip zeros boundaries" begin
            psi = ones(Float64, 5, 6)
            Tarang.apply_streamfunction_bc!(psi, :no_slip)
            @test all(psi[1, :] .== 0.0)
            @test all(psi[end, :] .== 0.0)
            @test all(psi[:, 1] .== 0.0)
            @test all(psi[:, end] .== 0.0)
            @test psi[3, 3] == 1.0   # interior untouched
        end

        @testset ":free_slip copies neighbor (zero normal derivative)" begin
            psi = reshape(collect(1.0:30.0), 5, 6)
            # Independent oracle: replicate the documented stencil in the SAME
            # sequential order the module applies it (rows first, then columns;
            # the column copies read the already-updated corner rows).
            expected = copy(psi)
            nx, ny = size(expected)
            expected[1, :]      .= expected[2, :]
            expected[nx, :]     .= expected[nx-1, :]
            expected[:, 1]      .= expected[:, 2]
            expected[:, ny]     .= expected[:, ny-1]

            Tarang.apply_streamfunction_bc!(psi, :free_slip)
            @test psi == expected
            # Zero normal derivative means edge equals its inward neighbor.
            @test psi[:, 1] == psi[:, 2]
            @test psi[:, end] == psi[:, end-1]
        end

        @testset ":periodic is a no-op" begin
            psi = reshape(collect(1.0:30.0), 5, 6)
            ref = copy(psi)
            Tarang.apply_streamfunction_bc!(psi, :periodic)
            @test psi == ref
        end

        @testset "unknown bc throws" begin
            psi = ones(Float64, 4, 4)
            @test_throws ArgumentError Tarang.apply_streamfunction_bc!(psi, :bogus)
        end
    end

    # =======================================================================
    # Basis-info helpers: get_fourier_basis_info / all_periodic_fourier
    # =======================================================================
    @testset "get_fourier_basis_info / all_periodic_fourier" begin
        coords, dist, bases, x, y = build_periodic_2d()
        info = Tarang.get_fourier_basis_info(collect(bases))
        @test length(info) == 2
        @test all(i.is_fourier for i in info)
        @test Tarang.all_periodic_fourier(info) == true

        # Mixed basis (Fourier + Chebyshev) is NOT all-periodic-Fourier.
        coords2 = CartesianCoordinates("x", "z")
        dist2 = Distributor(coords2; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords2["x"]; size=16, bounds=(0.0, TWO_PI))
        zb = ChebyshevT(coords2["z"]; size=16, bounds=(0.0, 1.0))
        info2 = Tarang.get_fourier_basis_info([xb, zb])
        @test info2[1].is_fourier == true
        @test info2[2].is_fourier == false
        @test Tarang.all_periodic_fourier(info2) == false
    end
end

println("All streamfunction tests passed!")
