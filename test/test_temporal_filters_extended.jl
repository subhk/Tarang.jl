"""
Extended tests for temporal filter helper modules with independent analytic oracles:
  - temporal_filters_imex_etd.jl  (ETD/IMEX coefficient precompute, φ-functions)
  - temporal_filters_gql.jl       (GQL wavenumber decomposition)
  - temporal_filters_wave_mean.jl (horizontal-mean / wave-mean decomposition)

Every expected value comes from closed-form math, NOT from running the function
under test on itself.
"""

using Test
using Tarang
using LinearAlgebra
using StaticArrays
using FFTW

# ----------------------------------------------------------------------------
# Analytic φ-function oracles (independent ground truth)
#   φ₀(z) = exp(z)
#   φ₁(z) = (exp(z) - 1)/z,           φ₁(0) = 1
#   φ₂(z) = (exp(z) - 1 - z)/z²,      φ₂(0) = 1/2
# Scalar oracle for a 2×2 matrix function: diagonalize (or use Taylor for the
# matrix forms via dense exp from LinearAlgebra as a fully independent check).
# ----------------------------------------------------------------------------
phi0(z) = exp(z)
phi1(z) = abs(z) < 1e-8 ? 1.0 + z/2 + z^2/6 : (exp(z) - 1) / z
phi2(z) = abs(z) < 1e-6 ? 0.5 + z/6 + z^2/24 : (exp(z) - 1 - z) / z^2

# Independent dense matrix exponential (LinearAlgebra.exp) as oracle for the 2×2 helpers.
dense_exp(M::SMatrix{2,2}) = exp(Matrix(M))
# φ₁(M) = (exp(M) - I) M⁻¹  (matrix version), computed via dense LinearAlgebra ops.
function dense_phi1(M::SMatrix{2,2})
    Md = Matrix(M)
    (exp(Md) - I) * inv(Md)
end


@testset "Temporal Filters Extended" begin

# ============================================================================
# 1. IMEX / ETD coefficient precompute (PURE MATH)
# ============================================================================
@testset "ETD/IMEX coefficients (imex_etd)" begin

    # ---- _matrix_exp_2x2 against dense LinearAlgebra.exp ----
    @testset "_matrix_exp_2x2 vs dense exp" begin
        # Diagonal matrix: exp acts entrywise on the diagonal.
        D = SMatrix{2,2,Float64}(0.3, 0.0, 0.0, -0.7)
        E = Tarang._matrix_exp_2x2(D)
        @test E[1,1] ≈ exp(0.3)
        @test E[2,2] ≈ exp(-0.7)
        @test abs(E[1,2]) < 1e-14
        @test abs(E[2,1]) < 1e-14

        # General matrix with REAL distinct eigenvalues.
        M1 = SMatrix{2,2,Float64}(0.5, 0.2, -0.1, -0.4)   # column-major: [0.5 -0.1; 0.2 -0.4]
        @test Matrix(Tarang._matrix_exp_2x2(M1)) ≈ dense_exp(M1) atol=1e-12

        # Butterworth-type matrix (COMPLEX eigenvalues) — the real use-case branch.
        sqrt2 = sqrt(2.0)
        A = SMatrix{2,2,Float64}(sqrt2 - 1, -1.0, 2 - sqrt2, 1.0)
        L = -0.5 * A
        Ldt = L * 0.3
        @test Matrix(Tarang._matrix_exp_2x2(Ldt)) ≈ dense_exp(Ldt) atol=1e-12

        # Repeated-eigenvalue (near-degenerate) case: scalar multiple of I-ish.
        # exp(c·I) = exp(c)·I exactly.
        cI = SMatrix{2,2,Float64}(0.25, 0.0, 0.0, 0.25)
        Ec = Tarang._matrix_exp_2x2(cI)
        @test Matrix(Ec) ≈ exp(0.25) * I(2) atol=1e-12
    end

    # ---- _matrix_phi1_2x2 against dense (exp(M)-I) M⁻¹ ----
    @testset "_matrix_phi1_2x2 vs dense φ₁" begin
        # Diagonal: φ₁ acts entrywise: φ₁(diag(a,b)) = diag(φ₁(a), φ₁(b)).
        D = SMatrix{2,2,Float64}(0.4, 0.0, 0.0, -0.9)
        P = Tarang._matrix_phi1_2x2(D)
        @test P[1,1] ≈ phi1(0.4)
        @test P[2,2] ≈ phi1(-0.9)
        @test abs(P[1,2]) < 1e-13
        @test abs(P[2,1]) < 1e-13

        # General real-eigenvalue matrix.
        M1 = SMatrix{2,2,Float64}(0.5, 0.2, -0.1, -0.4)
        @test Matrix(Tarang._matrix_phi1_2x2(M1)) ≈ dense_phi1(M1) atol=1e-11

        # Butterworth (complex eigenvalues).
        sqrt2 = sqrt(2.0)
        A = SMatrix{2,2,Float64}(sqrt2 - 1, -1.0, 2 - sqrt2, 1.0)
        Ldt = (-0.5 * A) * 0.3
        @test Matrix(Tarang._matrix_phi1_2x2(Ldt)) ≈ dense_phi1(Ldt) atol=1e-11

        # Near-singular M ⇒ Taylor branch: φ₁(M) → I + M/2 + M²/6 as M → 0.
        # Use a tiny matrix; oracle is the same Taylor series truncation the
        # closed-form φ₁ converges to (independently: φ₁(0)=1 on the diagonal).
        tiny = SMatrix{2,2,Float64}(1e-12, 0.0, 0.0, 1e-12)
        Pt = Tarang._matrix_phi1_2x2(tiny)
        @test Pt[1,1] ≈ 1.0 atol=1e-9
        @test Pt[2,2] ≈ 1.0 atol=1e-9
    end

    # ---- precompute_etd_coefficients for ExponentialMean (scalar φ) ----
    @testset "precompute_etd_coefficients (ExponentialMean)" begin
        α = 0.7
        dt = 0.25
        f = ExponentialMean((8,); α=α)
        c = precompute_etd_coefficients(f, dt)

        z = -α * dt
        # exp_scalar = exp(-α dt)
        @test c.exp_scalar ≈ exp(z)
        # phi1_scalar = (1 - exp(-α dt))/α = φ₁(z)·dt
        @test c.phi1_scalar ≈ (1 - exp(z)) / α
        @test c.phi1_scalar ≈ phi1(z) * dt
        @test c.α ≈ α
        @test c.dt ≈ dt

        # Small-z branch (z → 0 ⇒ φ₁ → 1 ⇒ phi1_scalar → dt).
        # Oracle: the cancellation-free series φ₁(z) = 1 + z/2 + z²/6 + ... (the
        # naive (exp(z)-1)/z loses precision near 0). True value of φ₁(z)·dt.
        α_small = 1e-6
        dt_s = 0.1
        z_s = -α_small * dt_s
        phi1_series = 1 + z_s/2 + z_s^2/6 + z_s^3/24    # accurate near 0
        f_s = ExponentialMean((4,); α=α_small)
        c_s = precompute_etd_coefficients(f_s, dt_s)
        @test c_s.phi1_scalar ≈ phi1_series * dt_s atol=1e-15  # matches analytic φ₁·dt
        @test c_s.phi1_scalar ≈ dt_s atol=1e-7                 # → dt as z → 0
        @test c_s.exp_scalar ≈ 1.0 atol=1e-6
    end

    # ---- precompute_etd_coefficients for ButterworthFilter (matrix φ) ----
    @testset "precompute_etd_coefficients (ButterworthFilter)" begin
        α = 0.5
        dt = 0.3
        f = ButterworthFilter((8,); α=α)
        c = precompute_etd_coefficients(f, dt)

        sqrt2 = sqrt(2.0)
        A = SMatrix{2,2,Float64}(sqrt2 - 1, -1.0, 2 - sqrt2, 1.0)
        L = -α * A
        Ldt = L * dt

        # exp_matrix = exp(L·dt)  (oracle: dense exp)
        @test Matrix(c.exp_matrix) ≈ dense_exp(Ldt) atol=1e-12
        # phi1_matrix = φ₁(L·dt)·dt  (oracle: dense φ₁ × dt)
        @test Matrix(c.phi1_matrix) ≈ dense_phi1(Ldt) * dt atol=1e-11
        @test c.α ≈ α
        @test c.dt ≈ dt
    end

    # ---- precompute_imex_coefficients for ExponentialMean ----
    @testset "precompute_imex_coefficients (ExponentialMean SBDF)" begin
        α = 0.8
        dt = 0.2
        f = ExponentialMean((4,); α=α)

        # SBDF1: c₀ = 1 ⇒ exp_coeff = 1/(1 + α dt)
        c1 = precompute_imex_coefficients(f, dt; scheme=:SBDF1)
        @test c1.exp_coeff ≈ 1 / (1 + α * dt)
        @test c1.scheme == :SBDF1
        @test c1.α ≈ α
        @test c1.dt ≈ dt

        # SBDF2: c₀ = 3/2
        c2 = precompute_imex_coefficients(f, dt; scheme=:SBDF2)
        @test c2.exp_coeff ≈ 1 / (1.5 + α * dt)

        # SBDF3: c₀ = 11/6
        c3 = precompute_imex_coefficients(f, dt; scheme=:SBDF3)
        @test c3.exp_coeff ≈ 1 / (11/6 + α * dt)

        # Unknown scheme throws.
        @test_throws ArgumentError precompute_imex_coefficients(f, dt; scheme=:BOGUS)
    end

    # ---- precompute_imex_coefficients for ButterworthFilter (matrix inverse) ----
    @testset "precompute_imex_coefficients (Butterworth SBDF)" begin
        α = 0.6
        dt = 0.15
        f = ButterworthFilter((4,); α=α)

        sqrt2 = sqrt(2.0)
        A = SMatrix{2,2,Float64}(sqrt2 - 1, -1.0, 2 - sqrt2, 1.0)

        for (scheme, c0) in ((:SBDF1, 1.0), (:SBDF2, 1.5), (:SBDF3, 11/6))
            c = precompute_imex_coefficients(f, dt; scheme=scheme)
            # bw_M_inv should be (c₀·I + α·dt·A)⁻¹  — oracle via dense inverse.
            M = c0 * Matrix(I(2)) + α * dt * Matrix(A)
            @test Matrix(c.bw_M_inv) ≈ inv(M) atol=1e-12
            # Sanity: M_inv * M ≈ I
            @test Matrix(c.bw_M_inv) * M ≈ I(2) atol=1e-12
        end

        @test_throws ArgumentError precompute_imex_coefficients(f, dt; scheme=:NOPE)
    end

    # ---- linear_operator_coefficients ----
    @testset "linear_operator_coefficients" begin
        α = 0.9
        fe = ExponentialMean((4,); α=α)
        @test linear_operator_coefficients(fe) ≈ -α

        fb = ButterworthFilter((4,); α=α)
        sqrt2 = sqrt(2.0)
        A = SMatrix{2,2,Float64}(sqrt2 - 1, -1.0, 2 - sqrt2, 1.0)
        @test Matrix(linear_operator_coefficients(fb)) ≈ -α * Matrix(A) atol=1e-13
    end

    # ---- ETD1 update is EXACT for constant forcing on ExponentialMean ----
    # dh̄/dt = -α h̄ + α h, with h constant ⇒ h̄(t+dt) = h + (h̄₀ - h) e^{-α dt}.
    @testset "update_etd! exactness (ExponentialMean, constant h)" begin
        α = 1.3
        dt = 0.4
        N = 6
        f = ExponentialMean((N,); α=α)
        h0 = 2.0
        f.h̄ .= h0                     # initial mean
        c = precompute_etd_coefficients(f, dt)
        hconst = fill(0.5, N)         # constant forcing value
        update_etd!(f, hconst, c)
        # Analytic exact solution after one step:
        expected = 0.5 .+ (h0 - 0.5) .* exp(-α * dt)
        @test all(abs.(get_mean(f) .- expected) .< 1e-12)
    end

    # ---- IMEX SBDF1 update matches its defining implicit relation ----
    # (1 + α dt) h̄ⁿ⁺¹ = h̄ⁿ + α dt hⁿ  ⇒  h̄ⁿ⁺¹ = (h̄ⁿ + α dt hⁿ)/(1 + α dt)
    @testset "update_imex! SBDF1 (ExponentialMean)" begin
        α = 0.7
        dt = 0.3
        N = 5
        f = ExponentialMean((N,); α=α)
        f.h̄ .= 1.0
        c = precompute_imex_coefficients(f, dt; scheme=:SBDF1)
        hval = fill(2.0, N)
        update_imex!(f, (hval,), c)
        expected = (1.0 + α * dt * 2.0) / (1 + α * dt)
        @test all(abs.(get_mean(f) .- expected) .< 1e-12)
    end
end

# ============================================================================
# 2. GQL wavenumber decomposition
# ============================================================================
@testset "GQL decomposition (gql)" begin

    # Build a 1D field with KNOWN spectral content via rfft.
    # f(x) = cos(1·x) + cos(5·x) on x ∈ [0, 2π), Nx points.
    # rfft mode index i ↔ wavenumber (2π/Lx)*(i-1); here Lx=2π so k = i-1.
    @testset "1D decomposition split & sum" begin
        Nx = 32
        Lx = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        f = cos.(x) .+ cos.(5 .* x)        # energy at k=1 and k=5 only
        f_hat = rfft(f)                    # length Nx÷2+1

        # Cutoff Λ=3: keep |k| ≤ 3 (k=0,1,2,3), drop k≥4.
        gql = GQLDecomposition((Nx,), (Lx,); Λ=3.0)
        f_large, f_small = decompose!(gql, f_hat)

        # f_large + f_small must reconstruct the original spectrum exactly.
        @test f_large .+ f_small ≈ f_hat

        # Known content: k=1 is large, k=5 is small.
        # Mode index for k: i = k+1.
        @test abs(f_large[2]) > 1e-8        # k=1 present in large
        @test abs(f_small[2]) < 1e-12       # k=1 absent from small
        @test abs(f_small[6]) > 1e-8        # k=5 present in small
        @test abs(f_large[6]) < 1e-12       # k=5 absent from large

        # No leakage anywhere: every mode is in exactly one part.
        @test all(abs.(f_large .* f_small) .< 1e-20)
    end

    # mean = k=0 component only (Λ small enough that only DC qualifies).
    @testset "mean (k=0) extraction" begin
        Nx = 16
        Lx = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        meanval = 3.0
        f = meanval .+ cos.(2 .* x)        # DC = meanval, plus k=2
        f_hat = rfft(f)

        # rfft DC component = sum(f) = Nx * mean.  (verify storage convention)
        @test f_hat[1] ≈ Nx * meanval

        # Λ very small (< 1): only k=0 is "large".
        gql = GQLDecomposition((Nx,), (Lx,); Λ=0.5)
        f_large, f_small = decompose!(gql, f_hat)
        @test f_large[1] ≈ f_hat[1]         # DC kept
        @test all(abs.(f_large[2:end]) .< 1e-12)   # nothing else kept
        @test f_small[1] == 0               # DC removed from small
        @test f_large .+ f_small ≈ f_hat
    end

    # 2D decomposition with ZONAL cutoff |kx| ≤ Λ (full ky retained), per GQL.
    @testset "2D zonal cutoff & mode counts" begin
        Nx, Ny = 16, 16
        Lx = Ly = 2π
        gql = GQLDecomposition((Nx, Ny), (Lx, Ly); Λ=4.0)

        # Independently count modes with |kx| ≤ 4 (for ALL ky) over the rfft grid.
        nkx = Nx ÷ 2 + 1
        kx = [(2π / Lx) * i for i in 0:nkx-1]
        expected_large = 0
        for j in 1:Ny, i in 1:nkx
            abs(kx[i]) <= 4.0 && (expected_large += 1)
        end
        expected_total = nkx * Ny

        @test count_large_modes(gql) == expected_large
        @test count_small_modes(gql) == expected_total - expected_large
        @test count_large_modes(gql) + count_small_modes(gql) == expected_total
        @test get_cutoff(gql) ≈ 4.0

        # A single-frequency 2D field: f = cos(3x) ⇒ energy at (kx,ky)=(3,0).
        # |kx| = 3 ≤ 4 ⇒ large; nothing in small.
        x = collect(0:Nx-1) .* (Lx / Nx)
        f2 = [cos(3 * x[i]) for i in 1:Nx, _ in 1:Ny]
        fh2 = rfft(f2)
        fl, fs = decompose!(gql, fh2)
        @test fl .+ fs ≈ fh2
        @test maximum(abs.(fs)) < 1e-9      # all energy is large-scale
    end

    # set_cutoff! rebuilds the mask; counts change monotonically with Λ.
    @testset "set_cutoff! rebuilds mask" begin
        Nx, Ny = 16, 16
        gql = GQLDecomposition((Nx, Ny), (2π, 2π); Λ=2.0)
        n_small = count_large_modes(gql)
        set_cutoff!(gql, 6.0)
        @test get_cutoff(gql) ≈ 6.0
        n_large = count_large_modes(gql)
        @test n_large > n_small             # bigger Λ ⇒ more large modes

        # Λ=0 (QL) ⇒ the zonal mean: kx=0 for ALL ky → the whole kx=0 row (Ny entries),
        # not just the (0,0) DC mode.
        set_cutoff!(gql, 0.0)
        @test count_large_modes(gql) == Ny
    end

    # project_large!/project_small! are complementary in-place projections.
    @testset "project_large!/project_small! complementarity" begin
        Nx = 32
        Lx = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        f = cos.(x) .+ 0.5 .* cos.(7 .* x)
        f_hat = rfft(f)

        gql = GQLDecomposition((Nx,), (Lx,); Λ=3.0)

        a = copy(f_hat); project_large!(gql, a)   # keep |k|≤3
        b = copy(f_hat); project_small!(gql, b)   # keep |k|>3
        @test a .+ b ≈ f_hat                       # exact split
        @test abs(a[2]) > 1e-8 && abs(a[8]) < 1e-12   # k=1 in large, k=7 not
        @test abs(b[8]) > 1e-8 && abs(b[2]) < 1e-12   # k=7 in small, k=1 not
    end

    # Constructor argument validation.
    @testset "constructor validation" begin
        @test_throws ArgumentError GQLDecomposition((16, 16), (2π,); Λ=2.0)  # 2D needs 2 domain dims
    end
end

# ============================================================================
# 3. Wave-mean / horizontal-mean decomposition
# ============================================================================
@testset "Wave-mean decomposition (wave_mean)" begin

    # ---- HorizontalMean: exact horizontal average ----
    # f(x,z) = A(z) + B(z)·cos(kx).  Horizontal mean over x = A(z) (cos averages to 0).
    @testset "HorizontalMean exact average (2D)" begin
        Nx, Nz = 32, 8
        Lx = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        zvals = collect(range(-1.0, 1.0; length=Nz))
        Az = [1.0 + 0.5 * z for z in zvals]        # known profile A(z)
        Bz = [0.3 + z^2 for z in zvals]
        kx = 3
        f = [Az[j] + Bz[j] * cos(kx * x[i]) for i in 1:Nx, j in 1:Nz]

        hmean = HorizontalMean((Nx, Nz); horizontal_dims=(1,))
        update!(hmean, f)
        profile = get_profile(hmean)
        @test length(profile) == Nz
        @test profile ≈ Az atol=1e-12             # cos averages out ⇒ mean = A(z)

        # broadcast_profile repeats the profile along x (constant in x).
        bp = broadcast_profile(hmean)
        @test size(bp) == (Nx, Nz)
        for j in 1:Nz
            @test all(abs.(bp[:, j] .- Az[j]) .< 1e-12)
        end

        # extract_fluctuation!: f - mean has ZERO horizontal mean.
        fluct = similar(f)
        extract_fluctuation!(fluct, hmean, f)
        # Per-column horizontal mean of the fluctuation must be ~0.
        for j in 1:Nz
            @test abs(sum(fluct[:, j]) / Nx) < 1e-12
        end
        # And fluctuation equals B(z)cos(kx) component exactly.
        for j in 1:Nz, i in 1:Nx
            @test fluct[i, j] ≈ Bz[j] * cos(kx * x[i]) atol=1e-12
        end
    end

    # ---- extract_k0_and_fluctuation returns matching profile + fluctuation ----
    @testset "extract_k0_and_fluctuation (3D, avg over x,y)" begin
        Nx, Ny, Nz = 8, 8, 4
        Lx = Ly = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        y = collect(0:Ny-1) .* (Ly / Ny)
        Az = [2.0 - 0.5 * k for k in 1:Nz]          # known z-profile
        f = [Az[k] + cos(2 * x[i]) + sin(3 * y[j]) for i in 1:Nx, j in 1:Ny, k in 1:Nz]

        hmean = HorizontalMean((Nx, Ny, Nz); horizontal_dims=(1, 2))
        profile, fluct = extract_k0_and_fluctuation(hmean, f)
        @test profile ≈ Az atol=1e-12               # cos/sin average to 0 over full periods
        @test size(fluct) == (Nx, Ny, Nz)
        # Fluctuation has zero horizontal (x,y) mean per z-slice.
        for k in 1:Nz
            @test abs(sum(fluct[:, :, k]) / (Nx * Ny)) < 1e-12
        end
    end

    # ---- WaveMeanDecomposition: ETD-filtered mean converges to A(z) for steady input ----
    # With constant-in-time input, repeated ETD updates relax the mean filter toward
    # the horizontal mean A(z).  Oracle = the analytic horizontal mean A(z).
    @testset "WaveMeanDecomposition steady-state mean" begin
        Nx, Nz = 16, 6
        Lx = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        zvals = collect(range(0.0, 1.0; length=Nz))
        Az = [0.5 + z for z in zvals]
        f = [Az[j] + 0.4 * cos(2 * x[i]) for i in 1:Nx, j in 1:Nz]

        decomp = WaveMeanDecomposition((Nx, Nz); α=2.0, horizontal_dims=(1,))
        dt = 0.05
        local mean_profile
        for _ in 1:4000                              # relax filter to steady state
            mean_profile, _ = decompose!(decomp, :u, f, dt)
        end
        # Filtered mean should converge to the horizontal mean A(z).
        @test maximum(abs.(mean_profile .- Az)) < 1e-3
        @test get_mean_profile(decomp, :u) ≈ mean_profile

        # broadcast_mean repeats the converged profile across x.
        bm = broadcast_mean(decomp, :u)
        @test size(bm) == (Nx, Nz)
        for j in 1:Nz
            @test all(abs.(bm[:, j] .- mean_profile[j]) .< 1e-12)
        end
    end

    # ---- update_flux!: horizontally-averaged + filtered Reynolds stress ----
    # Flux product g(x,z) = R(z) + C(z)cos(2x): horizontal mean = R(z);
    # ETD-filtered steady state ⇒ R(z).
    @testset "WaveMeanDecomposition flux filtering" begin
        Nx, Nz = 16, 4
        Lx = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        Rz = [0.1 * k for k in 1:Nz]
        g = [Rz[j] + 0.7 * cos(2 * x[i]) for i in 1:Nx, j in 1:Nz]

        decomp = WaveMeanDecomposition((Nx, Nz); α=2.0, horizontal_dims=(1,))
        add_flux_field!(decomp, :uw)
        dt = 0.05
        for _ in 1:4000
            update_flux!(decomp, :uw, g, dt)
        end
        flux = get_filtered_flux(decomp, :uw)
        @test maximum(abs.(flux .- Rz)) < 1e-3       # converges to horizontal mean R(z)
    end

    # ---- WaveInducedForcing: end-to-end mean / wave / flux for steady fields ----
    @testset "WaveInducedForcing steady-state" begin
        Nx, Ny, Nz = 8, 8, 4
        Lx = Ly = 2π
        x = collect(0:Nx-1) .* (Lx / Nx)
        Uz = [1.0 + 0.2 * k for k in 1:Nz]            # known mean of u
        Wz = [0.5 - 0.1 * k for k in 1:Nz]
        # u = Uz + a·cos(2x);  w = Wz + a·cos(2x)  (same wave so u'w' mean = a²/2)
        a = 0.6
        u = [Uz[k] + a * cos(2 * x[i]) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        w = [Wz[k] + a * cos(2 * x[i]) for i in 1:Nx, j in 1:Ny, k in 1:Nz]

        forcing = WaveInducedForcing((Nx, Ny, Nz); α=2.0)
        add_field!(forcing, :u)
        add_field!(forcing, :w)
        add_flux!(forcing, :uw, :u, :w)
        dt = 0.05
        for _ in 1:4000
            update!(forcing, Dict(:u => u, :w => w), dt)
        end

        # Mean profiles converge to the analytic horizontal means.
        @test maximum(abs.(get_mean(forcing, :u) .- Uz)) < 1e-3
        @test maximum(abs.(get_mean(forcing, :w) .- Wz)) < 1e-3

        # Wave field for u is u - mean(u) = a·cos(2x): zero horizontal mean.
        wave_u = get_wave(forcing, :u)
        for k in 1:Nz
            @test abs(sum(wave_u[:, :, k]) / (Nx * Ny)) < 1e-3
        end

        # Filtered Reynolds stress ⟨u'w'⟩: u'=w'=a cos(2x) ⇒ ⟨u'w'⟩ = a²·⟨cos²⟩ = a²/2.
        flux = get_flux(forcing, :uw)
        @test maximum(abs.(flux .- (a^2 / 2))) < 1e-3

        # get_flux_3d / get_mean_3d broadcast profiles to full size.
        @test size(get_flux_3d(forcing, :uw)) == (Nx, Ny, Nz)
        @test size(get_mean_3d(forcing, :u)) == (Nx, Ny, Nz)

        # Unregistered field access throws.
        @test_throws KeyError get_wave(forcing, :nope)
    end
end

@testset "GQLWaveMeanSystem (combined orchestration)" begin
    # The coupled GQL+ETD update has no clean closed-form oracle, but the GQL
    # decomposition partitions the spectrum, so large+small must reconstruct the
    # input spectral field EXACTLY. This smoke/structural test drives the whole
    # orchestration (construct / add_field! / add_flux! / setup! / update! / getters).
    sys = Tarang.GQLWaveMeanSystem((8, 8, 4), (2π, 2π); Λ = 3.0, α = 0.1)
    Tarang.add_field!(sys, :u)
    Tarang.add_field!(sys, :w)
    Tarang.add_flux!(sys, :uw, :u, :w)
    Tarang.setup!(sys, 0.01)

    ssz = sys.spectral_size
    @test ssz == (5, 8, 4)                       # rfft along dim 1: 8÷2+1
    fh = Dict(:u => rand(ComplexF64, ssz...), :w => rand(ComplexF64, ssz...))
    fp = Dict(:u => rand(8, 8, 4), :w => rand(8, 8, 4))
    Tarang.update!(sys, fh, fp, 0.01)

    # GQL partition reconstructs the input spectrum exactly (independent oracle).
    @test isapprox(Tarang.get_large(sys, :u) .+ Tarang.get_small(sys, :u), fh[:u]; atol = 1e-12)
    @test isapprox(Tarang.get_large(sys, :w) .+ Tarang.get_small(sys, :w), fh[:w]; atol = 1e-12)

    # Mean profile and flux are vertical profiles (length Nz); finite-valued.
    @test size(Tarang.get_mean(sys, :u)) == (4,)
    @test all(isfinite, Tarang.get_mean(sys, :u))
    @test size(Tarang.get_flux(sys, :uw)) == (4,)
    @test all(isfinite, Tarang.get_flux(sys, :uw))

    # Cutoff getter / setter.
    @test Tarang.get_cutoff(sys) == 3.0
    Tarang.set_cutoff!(sys, 5.0)
    @test Tarang.get_cutoff(sys) == 5.0

    # add_flux! name-inference error path (name length != 2).
    @test_throws ArgumentError Tarang.add_flux!(sys, :toolongname)
end

end  # Temporal Filters Extended

println("All extended temporal filter tests passed!")
