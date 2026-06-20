# Guard tests for the CPU correctness fixes from the 2026-06-20 audit.
#
# Each testset pins behavior that was previously WRONG, with an independent
# oracle so a mismatch means a real regression (not a tautology). See
# memory/project_cpu_audit_2026_06_20.md for the full analysis.

using Test
using Tarang

# Closed-form Legendre polynomials (independent oracle).
_legP(n, x) = n == 0 ? 1.0 :
              n == 1 ? x :
              n == 2 ? (3x^2 - 1) / 2 :
              n == 3 ? (5x^3 - 3x) / 2 :
              n == 4 ? (35x^4 - 30x^2 + 3) / 8 : error("extend _legP")

@testset "CPU audit 2026-06-20 fixes" begin

    # Bug: Legendre interpolation dropped the orthonormal √((2n+1)/2) per-mode
    # factor (transform stores orthonormal coeffs; Clenshaw evaluates standard Pₙ),
    # so interpolated values were wrong. Fix scales coeffs in interpolate_jacobi.
    @testset "Legendre interpolation reproduces a polynomial" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        N = 12
        basis = Legendre(coords["z"]; size=N, bounds=(-1.0, 1.0))
        f = ScalarField(dist, "f", (basis,), Float64)
        ensure_layout!(f, :g)
        fz(z) = 3z^4 - 2z^2 + z - 0.5             # degree < N → represented exactly
        zj = vec(Array(Tarang.local_grid(basis, dist, 1)))
        Tarang.get_grid_data(f) .= fz.(zj)
        for p in (-0.8, -0.2, 0.0, 0.5, 0.9)
            val = Tarang.evaluate_interpolate(Tarang.Interpolate(f, coords["z"], p), :g)
            @test val ≈ fz(p) atol = 1e-8
        end
        # single-mode oracle (P₃) on a scaled interval too
        b2 = Legendre(coords["z"]; size=N, bounds=(0.0, 2.0))
        g = ScalarField(dist, "g", (b2,), Float64)
        ensure_layout!(g, :g)
        zj2 = vec(Array(Tarang.local_grid(b2, dist, 1)))
        gz(z) = _legP(3, z - 1.0)                 # native coord = z-1 ∈ [-1,1]
        Tarang.get_grid_data(g) .= gz.(zj2)
        for p in (0.3, 1.0, 1.7)
            val = Tarang.evaluate_interpolate(Tarang.Interpolate(g, coords["z"], p), :g)
            @test val ≈ gz(p) atol = 1e-8
        end
    end

    # Bug: multi-D even-N 3/2-dealias Nyquist split used plain conj() at the same
    # other-axis indices instead of a Hermitian reflection, dropping the imaginary
    # part of the Nyquist plane in ≥2D. The pad→truncate round-trip (pure coeff
    # reshuffling, no rescale) must be identity on any Hermitian spectrum.
    @testset "dealias pad→truncate identity on Hermitian 2D spectrum" begin
        N, M = 8, 12
        # Build a Hermitian spectrum X[i,j] = conj(X[ī,j̄]) (i.e. the spectrum of a
        # real field) without FFTW: symmetrize an arbitrary complex array.
        Y = ComplexF64[(i + 2j) + im * (0.5i - 0.3j) for i in 1:N, j in 1:N]
        X = similar(Y)
        for i in 1:N, j in 1:N
            ii = mod(N - (i - 1), N) + 1
            jj = mod(N - (j - 1), N) + 1
            X[i, j] = (Y[i, j] + conj(Y[ii, jj])) / 2
        end
        nyq = N ÷ 2 + 1
        @test count(!iszero, imag.(X[nyq, :])) > 0     # Nyquist row genuinely complex
        padded = zeros(ComplexF64, M, M)
        Tarang._pad_spectral!(padded, X, (N, N), (M, M), [1, 2])
        result = zeros(ComplexF64, N, N)
        Tarang._truncate_spectral!(result, padded, (N, N), (M, M), [1, 2])
        @test result ≈ X atol = 1e-12                  # old conj() collapsed Nyquist row to Re()
    end

    # Bug: FractionalLaplacian silently dropped the non-Fourier axis on a mixed
    # domain (Cheb/Legendre contributes 0 to |k|²). Fix rejects any non-Fourier axis.
    @testset "FractionalLaplacian rejects mixed domains; works all-Fourier" begin
        c2 = CartesianCoordinates("x", "z")
        d2 = Distributor(c2; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(c2["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(c2["z"]; size=8, bounds=(-1.0, 1.0))
        fmix = ScalarField(d2, "fmix", (xb, zb), Float64)
        @test_throws ArgumentError Tarang.evaluate_fractional_laplacian(
            Tarang.FractionalLaplacian(fmix, 1.0), :g)

        c1 = CartesianCoordinates("x")
        d1 = Distributor(c1; mesh=(1,), dtype=Float64)
        b1 = RealFourier(c1["x"]; size=16, bounds=(0.0, 2π))
        fa = ScalarField(d1, "fa", (b1,), Float64)
        ensure_layout!(fa, :g)
        xj = vec(Array(Tarang.local_grid(b1, d1, 1)))
        Tarang.get_grid_data(fa) .= sin.(xj)            # (-Δ)^1 sin x = sin x
        res = Tarang.evaluate_fractional_laplacian(Tarang.FractionalLaplacian(fa, 1.0), :g)
        @test Array(Tarang.get_grid_data(res)) ≈ sin.(xj) atol = 1e-8
    end

    # Bug: divide evaluate fallback ignored the requested layout (returned :g for a
    # :c request). Fix mirrors evaluate_power's _ensure_result_layout!.
    @testset "divide fallback honors requested layout" begin
        c1 = CartesianCoordinates("x")
        d1 = Distributor(c1; mesh=(1,), dtype=Float64)
        b1 = RealFourier(c1["x"]; size=16, bounds=(0.0, 2π))
        u = ScalarField(d1, "u", (b1,), Float64)
        ensure_layout!(u, :g)
        xj = vec(Array(Tarang.local_grid(b1, d1, 1)))
        Tarang.get_grid_data(u) .= 2.0 .+ sin.(xj)      # nonzero everywhere
        res_c = Tarang._divide_result(1.0, u, :c)
        @test res_c isa ScalarField
        @test res_c.current_layout == :c
        @test Tarang.get_coeff_data(res_c) !== nothing
    end

    # Bug: sgs_dissipation diagnostic carried a spurious factor of 2.
    # |S̄| = √(2 S̄ᵢⱼS̄ᵢⱼ) ⇒ εₛₛ = νₑ|S̄|² (the 2 is already in |S̄|).
    @testset "sgs_dissipation has no spurious factor of 2" begin
        smag = SmagorinskyModel(filter_width=(1.0, 1.0, 1.0), field_size=(2, 2, 2))
        fill!(smag.eddy_viscosity, 0.5)
        mags = fill(0.3, (2, 2, 2))
        @test all(sgs_dissipation(smag, mags) .== 0.5 .* mags .^ 2)
        @test mean_sgs_dissipation(smag, mags) ≈ sum(0.5 .* mags .^ 2) / length(mags) atol = 1e-12
    end

    # Bug (conditional): combine_multiply(array, array) used matrix `*` not
    # elementwise `.*` (combine_add already uses elementwise `+`).
    @testset "combine_multiply arrays is elementwise" begin
        @test Tarang.combine_multiply([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == [4.0, 10.0, 18.0]
        @test Tarang.combine_multiply([1.0 2.0; 3.0 4.0], [10.0 20.0; 30.0 40.0]) ==
              [10.0 40.0; 90.0 160.0]
    end
end

# ── Round 2 (under-covered subsystems + resolved leftover candidates) ────────
using LinearAlgebra

@testset "CPU audit 2026-06-20 round 2 fixes" begin

    # Bug: ScalarField * complex Number fell through to a lazy Multiply Future (complex
    # scaling silently never applied). Fix broadens *(::ScalarField, ::Number) + promote.
    @testset "ScalarField * complex Number evaluates" begin
        c = CartesianCoordinates("x"); d = Distributor(c; mesh=(1,), dtype=ComplexF64)
        b = ComplexFourier(c["x"]; size=8, bounds=(0.0, 2π))
        g = ScalarField(d, "g", (b,), ComplexF64); ensure_layout!(g, :g)
        xj = vec(Array(Tarang.local_grid(b, d, 1)))
        Tarang.get_grid_data(g) .= ComplexF64.(cos.(xj))
        r = g * (2 + 3im)
        @test r isa ScalarField
        @test Array(Tarang.get_grid_data(r)) ≈ (2 + 3im) .* cos.(xj) atol = 1e-10
        @test (2 + 3im) * g isa ScalarField
    end

    # Bug: Kronecker (TensorMatMat) used adjoint A₁' instead of transpose A₁ᵀ — wrong sign
    # of the imaginary part for complex factors. vec(C) must equal (A₁⊗A₂)·x.
    @testset "Kronecker product uses transpose (complex-correct)" begin
        A1 = ComplexF64[1 0; 0 im]
        A2 = ComplexF64[1.0 0.5; 0.0 1.0]
        op = Tarang.create_kronecker_operator(AbstractMatrix[A1, A2])
        x = ComplexF64[1, 2, 3, 4]
        C = zeros(ComplexF64, 2, 2)
        Tarang.fast_matmat!(C, op, x)
        @test vec(C) ≈ kron(A1, A2) * x atol = 1e-12
    end

    # Bug: analysis-task axis resolution used a hardcoded x=1/y=2/z=3 table, ignoring the
    # field's real axis order. Now resolves via basis.meta.element_label first.
    @testset "resolve_dimension_index respects field axis order" begin
        c = CartesianCoordinates("z", "x")          # z first, x second
        d = Distributor(c; mesh=(1, 1), dtype=Float64)
        bz = ChebyshevT(c["z"]; size=8, bounds=(-1.0, 1.0))
        bx = RealFourier(c["x"]; size=8, bounds=(0.0, 2π))
        f = ScalarField(d, "f", (bz, bx), Float64)
        @test Tarang.resolve_dimension_index(f, :z) == 1
        @test Tarang.resolve_dimension_index(f, :x) == 2   # hardcoded table would give 1
    end

    # Bug: coerce_constant_value didn't fold constant arithmetic, so operator args like
    # `1/2` (fraclap exponent) or `Lz/2` (BC position) degraded to UnknownOperator.
    @testset "coerce_constant_value folds constant arithmetic" begin
        CO = Tarang.ConstantOperator
        @test Tarang.coerce_constant_value(Tarang.DivideOperator(CO(1.0), CO(2.0))) ≈ 0.5
        @test Tarang.coerce_constant_value(Tarang.DivideOperator(CO(3.0), CO(2.0))) ≈ 1.5
        @test Tarang.coerce_constant_value(CO(2.5)) ≈ 2.5
        @test Tarang.coerce_constant_value(2.5) ≈ 2.5
    end

    # Decision (2026-06-21, user-requested): GQL uses a ZONAL-wavenumber cutoff |kx|≤Λ
    # (Marston-Chini-Tobias 2016), NOT an isotropic √(kx²+ky²) cutoff. At Λ=0 (QL) the
    # large-scale set is the full kx=0 plane (zonal mean), not just the (0,0) mode.
    @testset "GQL uses zonal-wavenumber cutoff" begin
        gql = GQLDecomposition((8, 8), (2π, 2π); Λ=0.0)
        m = Array(gql.large_scale_mask)
        @test all(m[1, :])              # whole kx=0 row retained (zonal mean)
        @test !any(m[2:end, :])         # nothing with kx≠0
        @test count(m) == size(m, 2)    # exactly Ny (isotropic cutoff would give 1)
    end

    # Bug: volume-integral weight/dim mismatch (VectorField) silently misaligned weights.
    # Now errors instead of returning a wrong number.
    @testset "compute_weighted_integral rejects dim mismatch" begin
        @test_throws ArgumentError Tarang.compute_weighted_integral(
            rand(2, 3, 4), Any[collect(1.0:2.0), collect(1.0:3.0)])
        @test Tarang.compute_weighted_integral(ones(3, 4), Any[ones(3), ones(4)]) ≈ 12.0
    end
end
