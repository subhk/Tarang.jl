# =============================================================================
# Tests for src/core/basis/basis_product_matrices.jl
#
# NCC (non-constant-coefficient) product matrices: ncc_matrix / product_matrix
# for RealFourier and JacobiBasis, plus the Jacobi/Legendre linearization
# helpers.
#
# ORACLE STRATEGY (all expected values are independent ground truth):
#   * mode-0 product matrix == identity (multiplication by the constant 1).
#   * trig identities for Fourier (cos(x)cos(2x) = ½cos(x)+½cos(3x)).
#   * polynomial products for Legendre/Chebyshev/Jacobi: build M from an NCC
#     coefficient vector, apply it to an arg coefficient vector, then verify
#     M*g reconstructs the *grid-space* pointwise product f.*g sampled on a
#     dense node set. The grid product is computed WITHOUT the function under
#     test, so it is a genuine oracle.
#
# COEFFICIENT LAYOUT (verified empirically, NOT from docstrings):
#   * RealFourier product matrices act on the INTERLEAVED REAL layout of
#     length N: index 1 = cos_0 (constant), index 2k = cos_k coefficient,
#     index 2k+1 = msin_k coefficient where msin = -sin (so the +sin(kx)
#     amplitude is stored as the NEGATED value). This is NOT the rfft
#     half-spectrum a ScalarField carries.
#   * JacobiBasis product matrices act on standard P_n^{(a,b)} coefficient
#     vectors of length N (index n+1 ↔ P_n). Chebyshev T/U here use the
#     *standard Jacobi* P_n^{(∓1/2,∓1/2)}, NOT the normalized T_n/U_n.
# =============================================================================

using Test
using Tarang
using SparseArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Helpers (oracles — none of these call the functions under test)
# ---------------------------------------------------------------------------

# Build a RealFourier interleaved coefficient vector from explicit amplitudes:
#   f(x) = c0 + Σ_k [ cosamp[k]·cos(kx) + sinamp[k]·sin(kx) ]
# Storage: v[1]=c0, v[2k]=cosamp[k], v[2k+1] = -sinamp[k]  (msin = -sin).
function build_interleaved(N::Int, c0::Float64,
                           cosamp::Dict{Int,Float64},
                           sinamp::Dict{Int,Float64})
    v = zeros(Float64, N)
    v[1] = c0
    for (k, a) in cosamp
        2k <= N && (v[2k] = a)
    end
    for (k, a) in sinamp
        2k + 1 <= N && (v[2k+1] = -a)   # msin stores the negated sin amplitude
    end
    return v
end

# Evaluate a signal described by an interleaved RealFourier coeff vector.
function eval_interleaved(v::AbstractVector, x::AbstractVector)
    N = length(v)
    y = fill(v[1], length(x))
    k = 1
    j = 2
    while j <= N
        y .+= v[j] .* cos.(k .* x)          # cos_k
        if j + 1 <= N
            y .+= (-v[j+1]) .* sin.(k .* x)  # actual sin amplitude = -(stored msin)
        end
        j += 2
        k += 1
    end
    return y
end

# Evaluate Σ_n c[n+1] P_n^{(a,b)}(x) using the package's own polynomial
# evaluator (this is a basis-function evaluation, independent of the product
# machinery being tested).
function eval_jacobi_coeffs(c::AbstractVector, x::AbstractVector, a::Float64, b::Float64)
    y = zeros(Float64, length(x))
    for n in 0:(length(c)-1)
        if c[n+1] != 0
            y .+= c[n+1] .* Tarang._jacobi_polynomial_values(n, a, b, x)
        end
    end
    return y
end

# Independent Legendre linearization coefficient via Gauss-Legendre projection:
#   P_m P_n = Σ_k A_k P_k,  A_k = (2k+1)/2 ∫_{-1}^{1} P_m P_n P_k dx.
function legendre_lin_oracle(m::Int, n::Int, k::Int)
    nq = m + n + k + 4
    nodes, w = Tarang._gauss_jacobi_quadrature(nq, 0.0, 0.0)
    Pm = Tarang._jacobi_polynomial_values(m, 0.0, 0.0, nodes)
    Pn = Tarang._jacobi_polynomial_values(n, 0.0, 0.0, nodes)
    Pk = Tarang._jacobi_polynomial_values(k, 0.0, 0.0, nodes)
    return (2k + 1) / 2 * sum(w .* Pm .* Pn .* Pk)
end

@testset "NCC product matrices (basis_product_matrices.jl)" begin

    # =======================================================================
    # RealFourier
    # =======================================================================
    @testset "RealFourier product_matrix" begin
        coords = CartesianCoordinates("x")
        N = 16
        basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        # --- mode 0 (constant cos_0 = 1) must be the identity -------------
        M0 = product_matrix(basis, basis, basis, 0)
        @test size(M0) == (N, N)
        @test Matrix(M0) == Matrix(I, N, N)

        # The sin part of k=0 does not exist; the constructor never produces
        # an is_sin mode at ncc_mode 0, so there is no separate zero-matrix
        # branch to exercise from the public mode index. (Covered implicitly.)

        # --- single-mode product matrices: exact trig-identity columns ----
        # f = cos(x)  (ncc_mode = 1).  Column for arg = cos(2x) (index 4):
        #   cos(x)·cos(2x) = ½cos(x) + ½cos(3x)
        #   -> cos_1 (index 2) and cos_3 (index 6), each 0.5.
        M1 = product_matrix(basis, basis, basis, 1)
        col = Vector(M1[:, 4])        # arg = cos_2 = cos(2x)
        expected = zeros(N); expected[2] = 0.5; expected[6] = 0.5
        @test col ≈ expected

        # f = -sin(x) (ncc_mode = 2, the msin_1 storage slot).
        #   (-sin x)·cos(2x) = -½ sin(3x) + ½ sin(x)
        # msin storage = -(sin amplitude): msin_3 = ½, msin_1 = -½.
        #   msin_1 at index 3, msin_3 at index 7.
        M2 = product_matrix(basis, basis, basis, 2)
        col2 = Vector(M2[:, 4])       # arg = cos_2 = cos(2x)
        expected2 = zeros(N); expected2[3] = -0.5; expected2[7] = 0.5
        @test col2 ≈ expected2

        # --- caching returns the same object on the 2nd call --------------
        @test product_matrix(basis, basis, basis, 1) === M1
    end

    @testset "RealFourier ncc_matrix end-to-end (grid oracle)" begin
        coords = CartesianCoordinates("x")
        N = 24
        basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        xs = collect(range(0.0, 2π; length=400))

        # Simple identity: f = cos(x), g = cos(2x) -> ½cos(x)+½cos(3x).
        fc = build_interleaved(N, 0.0, Dict(1=>1.0), Dict{Int,Float64}())
        gc = build_interleaved(N, 0.0, Dict(2=>1.0), Dict{Int,Float64}())
        M = ncc_matrix(basis, basis, basis, fc)
        pc = M * gc
        exp_simple = build_interleaved(N, 0.0, Dict(1=>0.5, 3=>0.5), Dict{Int,Float64}())
        @test pc ≈ exp_simple

        # General mixed NCC (DC + cos + sin) times mixed arg.
        fc2 = build_interleaved(N, 0.5, Dict(1=>0.3, 2=>0.2), Dict(1=>0.4))
        gc2 = build_interleaved(N, 0.0, Dict(3=>1.0), Dict(2=>0.5))
        M2 = ncc_matrix(basis, basis, basis, fc2)
        pc2 = M2 * gc2
        fv = eval_interleaved(fc2, xs)
        gv = eval_interleaved(gc2, xs)
        pcv = eval_interleaved(pc2, xs)
        @test maximum(abs.(pcv .- fv .* gv)) < 1e-12

        # NCC = pure constant (DC only) acts as scalar multiply by that constant.
        fc3 = build_interleaved(N, 2.5, Dict{Int,Float64}(), Dict{Int,Float64}())
        M3 = ncc_matrix(basis, basis, basis, fc3)
        @test Matrix(M3) ≈ 2.5 .* Matrix(I, N, N)

        # All-zero NCC -> empty/zero sparse matrix of the right shape.
        Mz = ncc_matrix(basis, basis, basis, zeros(Float64, N))
        @test size(Mz) == (N, N)
        @test nnz(Mz) == 0

        # cutoff drops small NCC modes: a tiny cos_5 coefficient below the
        # cutoff must not contribute.
        fc4 = build_interleaved(N, 1.0, Dict(5=>1e-9), Dict{Int,Float64}())
        Mcut = ncc_matrix(basis, basis, basis, fc4; cutoff=1e-6)
        @test Matrix(Mcut) ≈ Matrix(I, N, N)   # only the DC=1 survives
    end

    # =======================================================================
    # Legendre  (Jacobi a=b=0)
    # =======================================================================
    @testset "Legendre product_matrix / ncc_matrix" begin
        coords = CartesianCoordinates("z")
        N = 12
        leg = Legendre(coords["z"]; size=N, bounds=(-1.0, 1.0))
        xs = collect(range(-0.99, 0.99; length=400))

        # mode 0 (P_0 = 1) -> identity.
        M0 = product_matrix(leg, leg, leg, 0)
        @test Matrix(M0) == Matrix(I, N, N)

        # P_1·P_1 = ⅓P_0 + ⅔P_2  (x·x = x² in Legendre).
        M1 = product_matrix(leg, leg, leg, 1)
        g = zeros(N); g[2] = 1.0           # P_1
        pc = M1 * g
        exp_p1 = zeros(N); exp_p1[1] = 1/3; exp_p1[3] = 2/3
        @test pc ≈ exp_p1

        # General Legendre NCC * arg, verified on the grid. Product degree
        # 3+4 = 7 < N = 12, so it is exactly representable.
        fc = zeros(N); fc[1]=1.0; fc[2]=2.0; fc[4]=0.5     # 1 + 2P_1 + 0.5P_3
        gc = zeros(N); gc[3]=1.0; gc[5]=-0.3               # P_2 - 0.3P_4
        M = ncc_matrix(leg, leg, leg, fc)
        pc2 = M * gc
        fv = eval_jacobi_coeffs(fc, xs, 0.0, 0.0)
        gv = eval_jacobi_coeffs(gc, xs, 0.0, 0.0)
        pcv = eval_jacobi_coeffs(pc2, xs, 0.0, 0.0)
        @test maximum(abs.(pcv .- fv .* gv)) < 1e-10
    end

    # =======================================================================
    # Chebyshev T / U  (Jacobi a=b=∓1/2) and general non-symmetric Jacobi
    # =======================================================================
    @testset "Chebyshev / Jacobi ncc_matrix (grid oracle)" begin
        coords = CartesianCoordinates("z")
        N = 12
        xs = collect(range(-0.99, 0.99; length=400))

        # ChebyshevT: a=b=-1/2. product matrices use standard P_n^{(-1/2,-1/2)}.
        cheb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        @test product_matrix(cheb, cheb, cheb, 0) == sparse(I, N, N) ||
              Matrix(product_matrix(cheb, cheb, cheb, 0)) == Matrix(I, N, N)
        fc = zeros(N); fc[2]=1.0           # P_1
        gc = zeros(N); gc[3]=1.0           # P_2
        M = ncc_matrix(cheb, cheb, cheb, fc)
        pc = M * gc
        fv = eval_jacobi_coeffs(fc, xs, -0.5, -0.5)
        gv = eval_jacobi_coeffs(gc, xs, -0.5, -0.5)
        pcv = eval_jacobi_coeffs(pc, xs, -0.5, -0.5)
        @test maximum(abs.(pcv .- fv .* gv)) < 1e-10

        # ChebyshevU: a=b=1/2.
        chebu = ChebyshevU(coords["z"]; size=N, bounds=(-1.0, 1.0))
        @test Matrix(product_matrix(chebu, chebu, chebu, 0)) == Matrix(I, N, N)
        fc2 = zeros(N); fc2[3]=1.0
        gc2 = zeros(N); gc2[2]=0.7; gc2[4]=-0.4
        Mu = ncc_matrix(chebu, chebu, chebu, fc2)
        pcu = Mu * gc2
        fv2 = eval_jacobi_coeffs(fc2, xs, 0.5, 0.5)
        gv2 = eval_jacobi_coeffs(gc2, xs, 0.5, 0.5)
        pcv2 = eval_jacobi_coeffs(pcu, xs, 0.5, 0.5)
        @test maximum(abs.(pcv2 .- fv2 .* gv2)) < 1e-10

        # General non-symmetric Jacobi a=1, b=1/2.
        jac = Jacobi(coords["z"]; a=1.0, b=0.5, size=N, bounds=(-1.0, 1.0))
        @test Matrix(product_matrix(jac, jac, jac, 0)) == Matrix(I, N, N)
        fc3 = zeros(N); fc3[1]=1.0; fc3[2]=0.5; fc3[3]=0.25
        gc3 = zeros(N); gc3[2]=1.0; gc3[4]=-0.3
        Mj = ncc_matrix(jac, jac, jac, fc3)
        pcj = Mj * gc3
        fv3 = eval_jacobi_coeffs(fc3, xs, 1.0, 0.5)
        gv3 = eval_jacobi_coeffs(gc3, xs, 1.0, 0.5)
        pcv3 = eval_jacobi_coeffs(pcj, xs, 1.0, 0.5)
        @test maximum(abs.(pcv3 .- fv3 .* gv3)) < 1e-10
    end

    # =======================================================================
    # Jacobi linearization helpers (used internally by _jacobi_product_matrix)
    # =======================================================================
    @testset "_jacobi_linearization_coefficients" begin
        # Multiplication by P_0 (m=0) is the identity in coefficient space:
        # P_0·P_n = P_n.
        for n in 0:5
            c = Tarang._jacobi_linearization_coefficients(0, n, 0.0, 0.0, 10)
            expected = zeros(10); expected[n+1] = 1.0
            @test c ≈ expected
        end
        # Symmetric m<->0 case (n=0): P_m·P_0 = P_m.
        for m in 0:5
            c = Tarang._jacobi_linearization_coefficients(m, 0, 0.0, 0.0, 10)
            expected = zeros(10); expected[m+1] = 1.0
            @test c ≈ expected
        end

        # P_1·P_1 = ⅓P_0 + ⅔P_2 for Legendre (quadrature path).
        c11 = Tarang._jacobi_linearization_coefficients(1, 1, 0.0, 0.0, 10)
        exp11 = zeros(10); exp11[1] = 1/3; exp11[3] = 2/3
        @test c11 ≈ exp11

        # General P_2·P_3 (Legendre) against the independent projection oracle.
        c23 = Tarang._jacobi_linearization_coefficients(2, 3, 0.0, 0.0, 10)
        for k in 0:9
            @test isapprox(c23[k+1], legendre_lin_oracle(2, 3, k); atol=1e-10)
        end
    end

    @testset "_jacobi_polynomial_values (recurrence vs known polynomials)" begin
        x = collect(range(-1.0, 1.0; length=50))
        # Legendre: P_0=1, P_1=x, P_2=(3x²-1)/2, P_3=(5x³-3x)/2.
        @test Tarang._jacobi_polynomial_values(0, 0.0, 0.0, x) ≈ ones(length(x))
        @test Tarang._jacobi_polynomial_values(1, 0.0, 0.0, x) ≈ x
        @test Tarang._jacobi_polynomial_values(2, 0.0, 0.0, x) ≈ (3 .* x.^2 .- 1) ./ 2
        @test Tarang._jacobi_polynomial_values(3, 0.0, 0.0, x) ≈ (5 .* x.^3 .- 3 .* x) ./ 2
    end

    @testset "_jacobi_norm_squared (Legendre h_n = 2/(2n+1))" begin
        for n in 0:6
            @test Tarang._jacobi_norm_squared(n, 0.0, 0.0) ≈ 2 / (2n + 1)
        end
    end

    @testset "_gauss_jacobi_quadrature (integrates polynomials exactly)" begin
        # Legendre weight w(x)=1: ∫_{-1}^1 x^p dx exact for p up to 2N-1.
        nodes, w = Tarang._gauss_jacobi_quadrature(6, 0.0, 0.0)
        @test sum(w) ≈ 2.0                         # ∫ 1 dx
        @test sum(w .* nodes) ≈ 0.0 atol=1e-12     # ∫ x dx
        @test sum(w .* nodes.^2) ≈ 2/3             # ∫ x² dx
        @test sum(w .* nodes.^4) ≈ 2/5             # ∫ x⁴ dx

        # Chebyshev-T special branch (a=b=-1/2): weight w(x)=(1-x²)^{-1/2}.
        # ∫_{-1}^1 (1-x²)^{-1/2} dx = π, and ∫ x²·w dx = π/2.
        nt, wt = Tarang._gauss_jacobi_quadrature(8, -0.5, -0.5)
        @test sum(wt) ≈ π
        @test sum(wt .* nt.^2) ≈ π/2
    end

    # =======================================================================
    # valid_elements / elements_to_groups
    # =======================================================================
    @testset "valid_elements & elements_to_groups" begin
        coords = CartesianCoordinates("x")
        N = 8
        rf = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))

        # In coefficient space the msin part of k=0 is invalid (no sin(0)).
        # Construct elements/groups so element 0 with odd parity is the k=0
        # msin slot. elements_to_groups divides element index by 2.
        elements = ([0, 1, 2, 3, 4, 5],)
        groups = Tarang.elements_to_groups(rf, (false,), elements)
        @test groups == ([0, 0, 1, 1, 2, 2],)

        valid = Tarang.valid_elements(rf, (), (false,), elements)
        # element 1 has group 0 and odd parity -> invalid; all others valid.
        @test valid[1] == true    # element 0 (cos_0), even
        @test valid[2] == false   # element 1 (msin_0), group 0 & odd -> dropped
        @test all(valid[3:end] .== true)

        # In grid space nothing is dropped.
        valid_g = Tarang.valid_elements(rf, (), (true,), elements)
        @test all(valid_g)

        # Jacobi: every element valid, groups == elements.
        leg = Legendre(coords["x"]; size=N, bounds=(-1.0, 1.0))
        je = ([0, 1, 2, 3],)
        @test Tarang.elements_to_groups(leg, (false,), je) == je
        @test all(Tarang.valid_elements(leg, (), (false,), je))
    end

    # =======================================================================
    # BUG: _legendre_linearization_coeff (Clebsch-Gordan helper) is WRONG.
    #
    # This helper is defined but has NO callers in src/ or test/ (the working
    # product path uses the quadrature-based _jacobi_linearization_clenshaw).
    # Its values disagree with the correct Legendre linearization
    # coefficients (verified independently via Gauss-Legendre projection in
    # legendre_lin_oracle and by the exact rationals below).
    #
    # The selection-rule ZEROS are correct (parity / triangle handling is OK),
    # so those are asserted as normal passing tests. Every NONZERO coefficient
    # is wrong, so those are wrapped in @test_broken with the CORRECT expected
    # value asserted.
    #
    # Suspected root cause: the factorial combination in the log-space 3j /
    # Clebsch-Gordan formula (lines ~584-590) is incorrect. Evidence: the
    # m=0 trivial case P_0·P_n = P_n must give coefficient exactly 1.0, but
    # _legendre_linearization_coeff(0,2,2) returns ≈ 2.778 instead of 1.0.
    # =======================================================================
    @testset "_legendre_linearization_coeff (selection-rule zeros — PASS)" begin
        # |m-n| <= k <= m+n and m+n+k even, else zero.
        @test Tarang._legendre_linearization_coeff(1, 1, 1) == 0.0  # parity odd
        @test Tarang._legendre_linearization_coeff(2, 3, 0) == 0.0  # k < |m-n|
        @test Tarang._legendre_linearization_coeff(1, 1, 4) == 0.0  # k > m+n
        @test Tarang._legendre_linearization_coeff(2, 2, 1) == 0.0  # parity odd
    end

    @testset "_legendre_linearization_coeff (nonzero magnitudes — BROKEN)" begin
        # Correct values (exact rationals / projection oracle):
        #   P_0·P_2 -> P_2 : 1            (multiplication by 1)
        #   P_1·P_1 -> P_0 : 1/3 ;  -> P_2 : 2/3
        #   P_2·P_2 -> P_0 : 1/5 ;  -> P_4 : 18/35
        f = Tarang._legendre_linearization_coeff
        @test_broken isapprox(f(0, 2, 2), 1.0;     atol=1e-10)
        @test_broken isapprox(f(1, 1, 0), 1/3;     atol=1e-10)
        @test_broken isapprox(f(1, 1, 2), 2/3;     atol=1e-10)
        @test_broken isapprox(f(2, 2, 0), 1/5;     atol=1e-10)
        @test_broken isapprox(f(2, 2, 4), 18/35;   atol=1e-10)
        @test_broken isapprox(f(2, 3, 1), legendre_lin_oracle(2, 3, 1); atol=1e-10)
    end
end
