"""
Test suite for src/core/operators/derivatives/derivatives_polynomial.jl

Targets the polynomial (Jacobi-family) spectral-derivative kernels:
  - evaluate_chebyshev_derivative!  (ChebyshevT path, via Differentiate)
  - evaluate_legendre_derivative!   (Legendre path, via Differentiate)
  - evaluate_legendre_single_derivative!
  - _cheb_deriv_nth_inplace! / _cheb_coeff_convert! / _get_cheb_deriv_plan
  - chebyshev_derivative_1d / chebyshev_derivative_1d!   (low-level kernels)

Oracle policy: every expected value is the ANALYTIC derivative of a polynomial
of degree < N (represented exactly by these bases), never the kernel's own
output. Test polynomial:
    f(z)  = 3z^4 - 2z^2 + z - 0.5
    f'(z) = 12z^3 - 4z + 1
    f''(z)= 36z^2 - 4
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: chebyshev_derivative_1d, chebyshev_derivative_1d!,
               _get_cheb_deriv_plan, _cheb_deriv_nth_inplace!,
               evaluate_chebyshev_derivative!, evaluate_legendre_derivative!,
               evaluate_legendre_single_derivative!,
               get_grid_data, get_coeff_data, ensure_layout!, local_grid

# Analytic oracle polynomial and its derivatives (degree 4, exact for N >= 5)
fpoly(z)   = 3z^4 - 2z^2 + z - 0.5
dfpoly(z)  = 12z^3 - 4z + 1
d2fpoly(z) = 36z^2 - 4

"""Build a ScalarField on `basis` (1D), set grid data to f(z), return (field, z)."""
function setup_field(basis, dist, fz::Function)
    f = ScalarField(dist, "f", (basis,), Float64)
    ensure_layout!(f, :g)
    z = vec(Array(Tarang.create_meshgrid(f.domain)[basis.meta.element_label]))
    get_grid_data(f) .= fz.(z)
    return f, z
end

@testset "derivatives_polynomial.jl" begin

    # ========================================================================
    # Part A: ChebyshevT derivative via the Differentiate operator
    #   exercises evaluate_chebyshev_derivative! -> _evaluate_local_chebyshev_derivative!
    #            -> _cheb_deriv_nth_inplace! -> chebyshev_derivative_1d!
    #            -> _cheb_coeff_convert! (when layout==:c)
    # ========================================================================
    @testset "ChebyshevT first derivative (analytic oracle)" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
        got = vec(Array(get_grid_data(d)))
        @test isapprox(got, dfpoly.(z); rtol=1e-8, atol=1e-8)
    end

    @testset "ChebyshevT second derivative (analytic oracle)" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 2), :g)
        got = vec(Array(get_grid_data(d)))
        @test isapprox(got, d2fpoly.(z); rtol=1e-7, atol=1e-7)
    end

    @testset "ChebyshevT derivative on shifted bounds (0,3) — chain-rule scale" begin
        # On [a,b] the kernel must apply 2/(b-a). f is set on PHYSICAL z, so the
        # returned derivative must equal f'(physical z) with no extra factor.
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = ChebyshevT(coords["z"]; size=20, bounds=(0.0, 3.0))
        f, z   = setup_field(basis, dist, fpoly)
        @test minimum(z) >= 0.0 - 1e-9 && maximum(z) <= 3.0 + 1e-9  # physical grid

        d  = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
        d2 = evaluate(Tarang.Differentiate(f, coords["z"], 2), :g)
        @test isapprox(vec(Array(get_grid_data(d))),  dfpoly.(z);  rtol=1e-6, atol=1e-6)
        @test isapprox(vec(Array(get_grid_data(d2))), d2fpoly.(z); rtol=1e-5, atol=1e-5)
    end

    @testset "ChebyshevT derivative returned in coefficient layout (:c)" begin
        # Exercises the layout==:c branch (_cheb_coeff_convert!).
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 1), :c)
        @test d.current_layout == :c
        # ANALYTIC ChebyshevT coeffs of f'(z)=12z^3-4z+1 on [-1,1]:
        #   12z^3 = 12*(3T1+T3)/4 = 9T1+3T3 ; -4z = -4T1 ; +1 = T0
        #   => c0=1, c1=9-4=5, c2=0, c3=3, rest 0
        # Independently confirmed: forward_transform!(grid-derivative) gives
        #   exactly [1, 5, 0, 3, 0...] (the basis' native coeff convention).
        c = vec(Array(get_coeff_data(d)))
        @test isapprox(c[1], 1.0; atol=1e-7)        # even mode: sign correct
        @test isapprox(c[3], 0.0; atol=1e-7)
        @test all(abs.(c[5:end]) .< 1e-7)
        # _cheb_coeff_convert! now undoes the T_n(-x)=(-1)^n alternation so the
        # odd-mode signs match the basis' native forward_transform! convention
        # (FIXED 2026-06-02; previously c1=-5, c3=-3 and :c did not round-trip).
        @test isapprox(c[2], 5.0; atol=1e-7)
        @test isapprox(c[4], 3.0; atol=1e-7)
    end

    @testset "ChebyshevT order=0 identity and negative-order error" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d0 = evaluate(Tarang.Differentiate(f, coords["z"], 0), :g)
        @test isapprox(vec(Array(get_grid_data(d0))), fpoly.(z); rtol=1e-12, atol=1e-12)

        # negative order is rejected by evaluate_chebyshev_derivative!
        result = ScalarField(dist, "r", (basis,), Float64)
        @test_throws ArgumentError evaluate_chebyshev_derivative!(result, f, 1, -1, :g)
    end

    @testset "ChebyshevT bad-bounds guard (b <= a) throws" begin
        # The basis constructor does NOT reject a>=b, so the b<=a guard inside
        # _evaluate_local_chebyshev_derivative! is reachable. Build a degenerate
        # basis and confirm the kernel rejects it.
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        bad    = ChebyshevT(coords["z"]; size=8, bounds=(2.0, 1.0))   # b < a
        f = ScalarField(dist, "f", (bad,), Float64)
        ensure_layout!(f, :g)
        get_grid_data(f) .= zeros(8)
        result = ScalarField(dist, "r", (bad,), Float64)
        @test_throws ArgumentError evaluate_chebyshev_derivative!(result, f, 1, 1, :g)
    end

    # ========================================================================
    # Part B: 2D ChebyshevT — exercises the dims==2 axis loops
    # ========================================================================
    @testset "ChebyshevT 2D derivative along each axis" begin
        coords = CartesianCoordinates("x", "z")
        dist   = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = ChebyshevT(coords["x"]; size=12, bounds=(-1.0, 1.0))
        zb = ChebyshevT(coords["z"]; size=12, bounds=(-1.0, 1.0))
        f  = ScalarField(dist, "f", (xb, zb), Float64)
        ensure_layout!(f, :g)
        mesh = Tarang.create_meshgrid(f.domain)
        X = Array(mesh["x"]); Z = Array(mesh["z"])
        # g(x,z) = x^3*z + 2x*z^2 ; dg/dx = 3x^2*z + 2z^2 ; dg/dz = x^3 + 4x*z
        get_grid_data(f) .= @. X^3 * Z + 2X * Z^2

        dx = evaluate(Tarang.Differentiate(f, coords["x"], 1), :g)
        dz = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
        @test isapprox(Array(get_grid_data(dx)), @. 3X^2 * Z + 2Z^2; rtol=1e-7, atol=1e-7)
        @test isapprox(Array(get_grid_data(dz)), @. X^3 + 4X * Z;    rtol=1e-7, atol=1e-7)
    end

    # ========================================================================
    # Part C: Legendre derivative via the Differentiate operator
    #   exercises evaluate_legendre_derivative! + evaluate_legendre_single_derivative!
    #   (order>1 takes the temp-field recurrence branch)
    #
    # BUG (all Legendre derivative VALUES are wrong):
    #   evaluate_legendre_single_derivative! applies the UNNORMALIZED Legendre
    #   derivative recurrence  c'[k] = (2k-1) * sum_{j>k, j-k odd} c[j],  but the
    #   Legendre basis + its forward/backward transforms use ORTHONORMAL
    #   polynomials P~_n = sqrt((2n+1)/2) * P_n. The correct orthonormal formula
    #   is  b~[k] = (2k+1)/gamma(k) * sum_{n>k,n-k odd} a[n]*gamma(n)  with
    #   gamma(n)=sqrt((2n+1)/2). The missing per-mode gamma factors corrupt the
    #   result. Verified: forward_transform!(grid of true f') gives
    #   [1.414214, 2.612789, 0, 2.565708, ...]; the orthonormal-correct recurrence
    #   reproduces this exactly, while the code yields
    #   [0.816497, 1.692552, 0, 2.262743, ...].
    # FIXED 2026-06-02: evaluate_legendre_single_derivative! now de-/re-normalizes
    # by γ_n=√((2n+1)/2) so the orthonormal-coefficient derivative is correct.
    # ========================================================================
    @testset "Legendre first derivative (analytic oracle)" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = Legendre(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
        @test isapprox(vec(Array(get_grid_data(d))), dfpoly.(z); rtol=1e-7, atol=1e-7)
    end

    @testset "Legendre second derivative (order>1 recurrence branch)" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = Legendre(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 2), :g)
        @test isapprox(vec(Array(get_grid_data(d))), d2fpoly.(z); rtol=1e-6, atol=1e-6)
    end

    @testset "Legendre third derivative (order>=3 aliasing copy branch)" begin
        # order>=3 reuses the temp field as operand AND result on the last step;
        # exercises the defensive copy() in evaluate_legendre_single_derivative!.
        # f'''(z) = 72z.
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = Legendre(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 3), :g)
        @test isapprox(vec(Array(get_grid_data(d))), (z -> 72z).(z); rtol=1e-6, atol=1e-6)
    end

    @testset "Legendre derivative on shifted bounds (0,3)" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = Legendre(coords["z"]; size=20, bounds=(0.0, 3.0))
        f, z   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
        @test isapprox(vec(Array(get_grid_data(d))), dfpoly.(z); rtol=1e-5, atol=1e-5)
    end

    @testset "Legendre coefficient layout (:c) + guard errors" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = Legendre(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f, _   = setup_field(basis, dist, fpoly)

        d = evaluate(Tarang.Differentiate(f, coords["z"], 1), :c)
        @test d.current_layout == :c

        result = ScalarField(dist, "r", (basis,), Float64)
        # negative-order guard
        @test_throws ArgumentError evaluate_legendre_derivative!(result, f, 1, -1, :c)

        # b<=a guard (constructor does not reject degenerate bounds)
        bad   = Legendre(coords["z"]; size=16, bounds=(2.0, 1.0))
        fbad  = ScalarField(dist, "fbad", (bad,), Float64)
        ensure_layout!(fbad, :g)
        get_grid_data(fbad) .= zeros(16)
        rbad  = ScalarField(dist, "rbad", (bad,), Float64)
        @test_throws ArgumentError evaluate_legendre_derivative!(rbad, fbad, 1, 1, :c)
    end

    # ========================================================================
    # Part D: Low-level chebyshev_derivative_1d / _1d! / plan cache
    #   These operate on raw vectors of grid values at Cheb-Gauss-Lobatto nodes
    #   in ascending order x_k = -cos(pi*k/(N-1)).
    # ========================================================================
    @testset "chebyshev_derivative_1d on raw vector (analytic oracle)" begin
        N = 16
        k = 0:(N-1)
        x = @. -cos(pi * k / (N - 1))     # ascending CGL nodes on [-1,1]
        fvals = fpoly.(x)
        deriv = chebyshev_derivative_1d(fvals, 1.0)   # scale=1 for [-1,1]
        @test isapprox(deriv, dfpoly.(x); rtol=1e-8, atol=1e-8)
    end

    @testset "chebyshev_derivative_1d! in-place equals out-of-place" begin
        N = 16
        k = 0:(N-1)
        x = @. -cos(pi * k / (N - 1))
        fvals = fpoly.(x)
        out = similar(fvals)
        chebyshev_derivative_1d!(out, fvals, 1.0)
        @test isapprox(out, dfpoly.(x); rtol=1e-8, atol=1e-8)
        @test isapprox(out, chebyshev_derivative_1d(fvals, 1.0); rtol=1e-12, atol=1e-12)
    end

    @testset "chebyshev_derivative_1d! with domain scale (0,3)" begin
        # scale = 2/(b-a) = 2/3; nodes mapped to [0,3]; derivative on physical x.
        N = 20
        a, b = 0.0, 3.0
        k = 0:(N-1)
        xnative = @. -cos(pi * k / (N - 1))
        xphys = @. (b - a)/2 * xnative + (b + a)/2
        fvals = fpoly.(xphys)
        out = similar(fvals)
        chebyshev_derivative_1d!(out, fvals, 2.0/(b - a))
        @test isapprox(out, dfpoly.(xphys); rtol=1e-6, atol=1e-6)
    end

    @testset "chebyshev_derivative_1d degenerate N<=1" begin
        @test chebyshev_derivative_1d(Float64[], 1.0) == Float64[]
        @test chebyshev_derivative_1d([3.7], 1.0) == [0.0]
        out = [9.9]
        chebyshev_derivative_1d!(out, [3.7], 1.0)
        @test out == [0.0]
    end

    @testset "chebyshev_derivative_1d! non-Float fallback (Int input)" begin
        # Triggers the AbstractVector fallback method (promotes to Float).
        N = 9
        k = 0:(N-1)
        x = @. -cos(pi * k / (N - 1))
        # integer-valued polynomial sampled but stored as Int vector: use g(z)=z (exact)
        # Build an Int result/input pair; fallback should promote and run.
        finp = Int.(round.(zeros(N)))            # all zeros, derivative of const 0 = 0
        rout = zeros(Int, N)
        chebyshev_derivative_1d!(rout, finp, 1.0)
        @test all(rout .== 0)
    end

    @testset "_get_cheb_deriv_plan caches per (N, T)" begin
        e1 = _get_cheb_deriv_plan(16, Float64)
        e2 = _get_cheb_deriv_plan(16, Float64)
        @test e1 === e2                       # same cached entry
        @test length(e1[2]) == 16             # scratch sized to N
        e3 = _get_cheb_deriv_plan(32, Float64)
        @test e3 !== e1
        @test length(e3[2]) == 32
    end

    @testset "_cheb_deriv_nth_inplace! matches repeated single derivative" begin
        N = 16
        k = 0:(N-1)
        x = @. -cos(pi * k / (N - 1))
        fvals = fpoly.(x)
        out = similar(fvals)
        tmp = similar(fvals)
        _cheb_deriv_nth_inplace!(out, fvals, 1.0, 2, tmp)    # order 2
        @test isapprox(out, d2fpoly.(x); rtol=1e-7, atol=1e-7)
    end

    # ========================================================================
    # Part E: ChebyshevU / ChebyshevV / generic Jacobi differentiation
    #   FIXED 2026-06-02: now routed through a nodal collocation differentiation
    #   matrix (evaluate_jacobi_collocation_derivative!), exact for degree < N.
    #   Oracle = analytic derivative of fpoly.
    # ========================================================================
    @testset "ChebyshevU / ChebyshevV / Jacobi differentiation (nodal collocation)" begin
        coords = CartesianCoordinates("z")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        for B in (ChebyshevU(coords["z"]; size=12, bounds=(-1.0, 1.0)),
                  ChebyshevV(coords["z"]; size=12, bounds=(-1.0, 1.0)),
                  Jacobi(coords["z"]; size=12, bounds=(-1.0, 1.0), a=0.3, b=0.7))
            f, z = setup_field(B, dist, fpoly)
            d1 = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
            @test isapprox(vec(Array(get_grid_data(d1))), dfpoly.(z); rtol=1e-8, atol=1e-8)
            d2 = evaluate(Tarang.Differentiate(f, coords["z"], 2), :g)
            @test isapprox(vec(Array(get_grid_data(d2))), d2fpoly.(z); rtol=1e-7, atol=1e-7)
        end
    end
end
