using Test, Tarang, LinearAlgebra

# Coverage-focused tests for src/core/basis/basis_operators.jl
#
# Targets the uncovered serial-CPU lines: derivative_basis cold branches,
# the Jacobi recurrence-conversion path (_general_jacobi_conversion, shift
# matrices, quadrature projection), jacobi_polynomial, gauss_jacobi_quadrature,
# the public conversion/differentiation matrices, and the basis dispatcher
# helpers.
#
# Notes on expected values:
# - jacobi_polynomial(x, n, a, b) is the standard (unnormalized) Jacobi P_n^{(a,b)}.
# - The conversion/derivative matrices act on coefficient vectors and must
#   PRESERVE the represented function: if c' = M c then
#       Σ c'_n φ_n^{out}(x) == Σ c_n φ_n^{in}(x)   for all x.
# - The UP-shift and POSITIVE-shift recurrence matrices are correct (verified).
#   The DOWN-shift / negative-shift / _general_jacobi_differentiation_matrix
#   parameter-recurrence paths are known to be mathematically wrong (the source
#   itself replaces them with collocation), so for those we only assert
#   structural invariants (shape / type / banded coefficient pattern), never
#   numerical function-preservation.

const T = Tarang

# Helper: evaluate Σ coeffs[n+1] * P_n^{(a,b)}(x) using the module's jacobi_polynomial
jeval(coeffs, a, b, x) = sum(coeffs[n+1] * T.jacobi_polynomial(x, n, a, b)
                             for n in 0:length(coeffs)-1)

@testset "basis_operators coverage" begin

    coords = CartesianCoordinates("x")
    xc = coords["x"]

    # ------------------------------------------------------------------
    # jacobi_polynomial : three-term recurrence + special cases
    # ------------------------------------------------------------------
    @testset "jacobi_polynomial" begin
        # n == 0 and n == 1 closed forms
        @test T.jacobi_polynomial(0.37, 0, 1.5, 2.5) == 1.0
        # P_1^{(a,b)} = 0.5*(a - b + (a+b+2) x)
        @test T.jacobi_polynomial(0.5, 1, 2.0, 3.0) ≈ 0.5 * (2.0 - 3.0 + (2.0 + 3.0 + 2.0) * 0.5)

        # Legendre (a=b=0): P_2 = (3x^2-1)/2, P_3 = (5x^3-3x)/2
        for x in (-0.8, -0.1, 0.3, 0.95)
            @test T.jacobi_polynomial(x, 2, 0.0, 0.0) ≈ (3x^2 - 1) / 2
            @test T.jacobi_polynomial(x, 3, 0.0, 0.0) ≈ (5x^3 - 3x) / 2
        end

        # Endpoint identity: P_n^{(a,b)}(1) = binom(n+a, n) ; for a=b=0 that is 1
        for n in 0:5
            @test T.jacobi_polynomial(1.0, n, 0.0, 0.0) ≈ 1.0
        end

        # Degenerate three-term branch (c1 ≈ 0): a=b=-1, n>=2 has a vanishing
        # leading coefficient and falls into the P_curr = P_prev1 fallback.
        # Just exercise it and require a finite real result (line 568).
        v = T.jacobi_polynomial(0.5, 3, -1.0, -1.0)
        @test isfinite(v)
    end

    # ------------------------------------------------------------------
    # gauss_jacobi_quadrature : closed-form Chebyshev, Golub-Welsch, N<1
    # ------------------------------------------------------------------
    @testset "gauss_jacobi_quadrature" begin
        # N < 1 guard returns empty vectors (line 586)
        n0, w0 = T.gauss_jacobi_quadrature(0, 0.0, 0.0)
        @test isempty(n0) && isempty(w0)
        @test eltype(n0) == Float64

        # Chebyshev-T special case (a=b=-1/2): nodes are Chebyshev-Gauss points,
        # all weights = π/N, sorted ascending.
        N = 5
        nC, wC = T.gauss_jacobi_quadrature(N, -0.5, -0.5)
        @test length(nC) == N
        @test issorted(nC)
        @test all(w ≈ π / N for w in wC)
        @test sum(wC) ≈ π                     # ∫ (1-x^2)^{-1/2} dx = π
        # Exactness: ∫_{-1}^1 x^2 / sqrt(1-x^2) dx = π/2
        @test sum(wC .* nC.^2) ≈ π / 2

        # Golub-Welsch general path (Legendre a=b=0): Gauss-Legendre quadrature.
        nL, wL = T.gauss_jacobi_quadrature(4, 0.0, 0.0)
        @test length(nL) == 4
        @test sum(wL) ≈ 2.0                   # ∫_{-1}^1 1 dx = 2
        @test sum(wL .* nL.^2) ≈ 2 / 3        # ∫ x^2 = 2/3, exact for 4-pt
        @test sum(wL .* nL.^4) ≈ 2 / 5        # exact up to degree 7
        @test isapprox(sum(wL .* nL), 0.0; atol=1e-12)   # symmetry

        # Golub-Welsch with a=b=1: weight (1-x)(1+x); μ0 = ∫(1-x^2) dx = 4/3
        nJ, wJ = T.gauss_jacobi_quadrature(4, 1.0, 1.0)
        @test sum(wJ) ≈ 4 / 3
    end

    # ------------------------------------------------------------------
    # derivative_basis : chains, order==0, order>1, and error branch
    # ------------------------------------------------------------------
    @testset "derivative_basis" begin
        tb = ChebyshevT(xc; size=10, bounds=(-1.0, 1.0))
        ub = ChebyshevU(xc; size=10, bounds=(-1.0, 1.0))
        vb = ChebyshevV(xc; size=10, bounds=(-1.0, 1.0))
        leb = Legendre(xc; size=10, bounds=(-1.0, 1.0))
        jb = Jacobi(xc; a=1.0, b=2.0, size=10, bounds=(-1.0, 1.0))

        # order == 0 returns the same basis object (lines 50, 99 ...)
        @test T.derivative_basis(tb, 0) === tb
        @test T.derivative_basis(ub, 0) === ub
        @test T.derivative_basis(vb, 0) === vb
        @test T.derivative_basis(leb, 0) === leb
        @test T.derivative_basis(jb, 0) === jb

        # Derivative basis chain types
        @test T.derivative_basis(tb) isa ChebyshevU            # ∂T -> U
        @test T.derivative_basis(ub) isa ChebyshevV            # ∂U -> V
        jv = T.derivative_basis(vb)                            # ∂V -> Jacobi(5/2,5/2)
        @test jv isa Jacobi
        @test jv.a ≈ 2.5 && jv.b ≈ 2.5
        jl = T.derivative_basis(leb)                           # ∂Legendre -> Jacobi(1,1)
        @test jl isa Jacobi
        @test jl.a ≈ 1.0 && jl.b ≈ 1.0
        jj = T.derivative_basis(jb)                            # ∂Jacobi(1,2) -> (2,3)
        @test jj.a ≈ 2.0 && jj.b ≈ 3.0

        # order > 1 recursion (lines 61, 112, ...): T -> U -> V
        @test T.derivative_basis(tb, 2) isa ChebyshevV
        # Legendre order 2: -> Jacobi(1,1) -> Jacobi(2,2)
        jl2 = T.derivative_basis(leb, 2)
        @test jl2 isa Jacobi && jl2.a ≈ 2.0 && jl2.b ≈ 2.0

        # Preserved metadata
        d = T.derivative_basis(tb)
        @test d.meta.size == tb.meta.size
        @test d.meta.bounds == tb.meta.bounds

        # Fourier derivative stays in the same basis
        rf = RealFourier(xc; size=8, bounds=(0.0, 2π))
        @test T.derivative_basis(rf) === rf
        @test T.derivative_basis(rf, 3) === rf

        # Negative order error branch (lines 47, 96, 145, ...)
        @test_throws ArgumentError T.derivative_basis(tb, -1)
        @test_throws ArgumentError T.derivative_basis(ub, -2)
        @test_throws ArgumentError T.derivative_basis(vb, -1)
        @test_throws ArgumentError T.derivative_basis(leb, -1)
        @test_throws ArgumentError T.derivative_basis(jb, -1)
        @test_throws ArgumentError T.derivative_basis(rf, -1)
    end

    # ------------------------------------------------------------------
    # _chebyshev_t_to_u_matrix : exact T_n -> U_n bidiagonal relation
    # ------------------------------------------------------------------
    @testset "_chebyshev_t_to_u_matrix" begin
        N = 6
        M = T._chebyshev_t_to_u_matrix(N)
        @test size(M) == (N, N)
        # Known nonzeros: M[1,1]=1 (T0=U0), M[2,2]=0.5 (T1=U1/2),
        # T_n = (U_n - U_{n-2})/2: diag 0.5, super-super-diag -0.5
        @test M[1, 1] == 1.0
        @test M[2, 2] == 0.5
        for n in 2:(N - 1)
            @test M[n + 1, n + 1] == 0.5
            @test M[n - 1, n + 1] == -0.5
        end

        # Function-preservation: T-coeffs -> U-coeffs represent the same poly.
        cT = [0.5, -1.0, 0.3, 0.7, -0.2, 0.1]
        cU = M * cT
        for x in (-0.7, 0.1, 0.6)
            t = acos(x)
            fT = sum(cT[n + 1] * cos(n * t) for n in 0:N-1)
            fU = sum(cU[n + 1] * sin((n + 1) * t) / sin(t) for n in 0:N-1)
            @test fT ≈ fU
        end

        # N == 1 degenerate (only the T0 -> U0 entry)
        M1 = T._chebyshev_t_to_u_matrix(1)
        @test size(M1) == (1, 1)
        @test M1[1, 1] == 1.0
    end

    # ------------------------------------------------------------------
    # _jacobi_conversion_matrix : top-level recurrence dispatcher
    # ------------------------------------------------------------------
    @testset "_jacobi_conversion_matrix dispatcher" begin
        N = 6
        # identity (same params) -> sparse I  (lines 223-224)
        Iid = T._jacobi_conversion_matrix(N, 0.5, 0.5, 0.5, 0.5)
        @test Matrix(Iid) == Matrix(I, N, N)
        # T -> U special case (lines 228-230) equals _chebyshev_t_to_u_matrix
        Mtu = T._jacobi_conversion_matrix(N, -0.5, -0.5, 0.5, 0.5)
        @test Matrix(Mtu) == Matrix(T._chebyshev_t_to_u_matrix(N))
        # general case dispatch (line 234)
        Mg = T._jacobi_conversion_matrix(N, 0.0, 0.0, 1.0, 1.0)
        @test size(Mg) == (N, N)
    end

    # ------------------------------------------------------------------
    # _general_jacobi_conversion : integer up-shift, identity, quadrature
    # ------------------------------------------------------------------
    @testset "_general_jacobi_conversion" begin
        N = 6
        c0 = [0.2, 0.5, -0.3, 0.1, 0.4, -0.1]

        # Identity short-circuit (lines 282-283)
        Gid = T._general_jacobi_conversion(N, 0.5, 0.5, 0.5, 0.5)
        @test Matrix(Gid) == Matrix(I, N, N)

        # Positive integer shift path (a:0->1, b:0->1) — function preserving.
        G = T._general_jacobi_conversion(N, 0.0, 0.0, 1.0, 1.0)
        cG = G * c0
        for x in (-0.5, 0.2, 0.8)
            @test jeval(c0, 0.0, 0.0, x) ≈ jeval(cG, 1.0, 1.0, x)
        end

        # Non-integer shift -> quadrature projection path (line 305).
        Q = T._general_jacobi_conversion(N, 0.0, 0.0, 0.5, 0.5)
        cQ = Q * c0
        for x in (-0.5, 0.2, 0.8)
            @test jeval(c0, 0.0, 0.0, x) ≈ jeval(cQ, 0.5, 0.5, x)
        end

        # Negative integer shift path is exercised for structure only (it is a
        # known-broken recurrence; assert shape/type, not numerics).
        Gn = T._general_jacobi_conversion(N, 2.0, 1.0, 0.0, 0.0)
        @test size(Gn) == (N, N)
    end

    # ------------------------------------------------------------------
    # _jacobi_conversion_positive_shift : multi-step a then b
    # ------------------------------------------------------------------
    @testset "_jacobi_conversion_positive_shift" begin
        N = 6
        c0 = [0.2, 0.5, -0.3, 0.1, 0.4, -0.1]
        # da=2, db=1 : P^{(0,0)} -> P^{(2,1)} preserving the function
        P = T._jacobi_conversion_positive_shift(N, 0.0, 0.0, 2, 1)
        @test size(P) == (N, N)
        cP = P * c0
        for x in (-0.6, 0.0, 0.7)
            @test jeval(c0, 0.0, 0.0, x) ≈ jeval(cP, 2.0, 1.0, x)
        end
        # da=0, db=0 short circuit -> identity
        P0 = T._jacobi_conversion_positive_shift(N, 0.5, 0.5, 0, 0)
        @test Matrix(P0) == Matrix(I, N, N)
    end

    # ------------------------------------------------------------------
    # _jacobi_conversion_negative_shift : structural exercise only
    # ------------------------------------------------------------------
    @testset "_jacobi_conversion_negative_shift" begin
        N = 6
        Pn = T._jacobi_conversion_negative_shift(N, 2.0, 1.0, 2, 1)
        @test size(Pn) == (N, N)
        @test eltype(Pn) == Float64
        # zero-shift -> identity
        P0 = T._jacobi_conversion_negative_shift(N, 0.5, 0.5, 0, 0)
        @test Matrix(P0) == Matrix(I, N, N)
    end

    # ------------------------------------------------------------------
    # Single-step shift matrices: up-shifts correct, down-shifts structural
    # ------------------------------------------------------------------
    @testset "single-step shift matrices" begin
        N = 6
        a, b = 0.5, 0.5
        c0 = [0.2, 0.5, -0.3, 0.1, 0.4, -0.1]

        # a-shift up: P^{(a,b)} -> P^{(a+1,b)}, function preserving (verified)
        Sa = T._jacobi_a_shift_up_matrix(N, a, b)
        @test size(Sa) == (N, N)
        ca = Sa * c0
        for x in (-0.5, 0.2, 0.8)
            @test jeval(c0, a, b, x) ≈ jeval(ca, a + 1, b, x)
        end
        # Bidiagonal structure: diagonal c1, subdiagonal c2 (no superdiagonal fill)
        for n in 0:(N - 1)
            denom = 2n + a + b + 1
            @test Sa[n + 1, n + 1] ≈ (n + a + b + 1) / denom
            if n > 0
                @test Sa[n, n + 1] ≈ -(n + b) / denom
            end
        end

        # b-shift up: P^{(a,b)} -> P^{(a,b+1)}, function preserving
        Sb = T._jacobi_b_shift_up_matrix(N, a, b)
        cb = Sb * c0
        for x in (-0.5, 0.2, 0.8)
            @test jeval(c0, a, b, x) ≈ jeval(cb, a, b + 1, x)
        end
        for n in 0:(N - 1)
            denom = 2n + a + b + 1
            @test Sb[n + 1, n + 1] ≈ (n + a + b + 1) / denom
            if n > 0
                @test Sb[n, n + 1] ≈ (n + a) / denom
            end
        end

        # Down-shift matrices: structural invariants only (known-broken numerics).
        Da = T._jacobi_a_shift_down_matrix(N, a, b)
        Db = T._jacobi_b_shift_down_matrix(N, a, b)
        @test size(Da) == (N, N) && size(Db) == (N, N)
        @test eltype(Da) == Float64 && eltype(Db) == Float64
        # Upper-bidiagonal pattern: nonzeros only on diag and first superdiagonal.
        for i in 1:N, j in 1:N
            if !(j == i || j == i + 1)
                @test Da[i, j] == 0.0
                @test Db[i, j] == 0.0
            end
        end
    end

    # ------------------------------------------------------------------
    # _jacobi_conversion_quadrature : weighted-projection path
    # ------------------------------------------------------------------
    @testset "_jacobi_conversion_quadrature" begin
        N = 6
        c0 = [0.2, 0.5, -0.3, 0.1, 0.4, -0.1]
        Q = T._jacobi_conversion_quadrature(N, 0.0, 0.0, 0.5, 0.5)
        @test size(Q) == (N, N)
        cQ = Q * c0
        for x in (-0.7, 0.0, 0.4, 0.9)
            @test jeval(c0, 0.0, 0.0, x) ≈ jeval(cQ, 0.5, 0.5, x)
        end
        # Projecting onto the SAME basis is the identity (within quadrature tol).
        Qid = T._jacobi_conversion_quadrature(N, 0.5, 0.5, 0.5, 0.5)
        @test isapprox(Matrix(Qid), Matrix(I, N, N); atol=1e-10)
    end

    # ------------------------------------------------------------------
    # public conversion_matrix : collocation path + caching
    # ------------------------------------------------------------------
    @testset "conversion_matrix (public)" begin
        N = 8
        tb = ChebyshevT(xc; size=N, bounds=(-1.0, 1.0))
        ub = ChebyshevU(xc; size=N, bounds=(-1.0, 1.0))

        # Identity conversion (same basis) -> I
        cm_id = T.conversion_matrix(tb, tb)
        @test Matrix(cm_id) == Matrix(I, N, N)

        # T -> U conversion preserves represented function on the U grid.
        cm = T.conversion_matrix(tb, ub)
        @test size(cm) == (N, N)
        cT = [0.3, -0.5, 0.2, 0.7, -0.1, 0.4, 0.0, 0.15]
        cU = cm * cT
        gridU = T._native_grid(ub, 1.0)
        Bt = Matrix{Float64}(T.evaluate_basis(tb, gridU, 0:N-1))
        Bu = Matrix{Float64}(T.evaluate_basis(ub, gridU, 0:N-1))
        @test maximum(abs.(Bt * cT .- Bu * cU)) < 1e-10

        # Cache returns the identical object on the second call.
        @test T.conversion_matrix(tb, ub) === cm

        # Legendre -> Legendre identity through the general branch.
        leb = Legendre(xc; size=N, bounds=(-1.0, 1.0))
        @test Matrix(T.conversion_matrix(leb, leb)) == Matrix(I, N, N)
    end

    # ------------------------------------------------------------------
    # differentiation_matrix (public) : ChebyshevT/Legendre sparse path,
    # ChebyshevU collocation path, domain scaling, order 0/1/2, caching
    # ------------------------------------------------------------------
    @testset "differentiation_matrix (public)" begin
        N = 8

        # order 0 is identity for the collocation path (ChebyshevU)
        ub = ChebyshevU(xc; size=N, bounds=(-1.0, 1.0))
        D0 = T.differentiation_matrix(ub, 0)
        @test D0 == Matrix(I, N, N)

        # ChebyshevT on a scaled, shifted domain (0,2): spectral derivative must
        # match the nodal derivative of the same function in physical coords.
        tb = ChebyshevT(xc; size=N, bounds=(0.0, 2.0))
        c = [0.1, 0.5, -0.3, 0.2, 0.4, -0.1, 0.05, 0.0]
        D = T.differentiation_matrix(tb, 1)
        @test size(D) == (N, N)
        grid = T._native_grid(tb, 1.0)
        phys = [0.0 + (xx + 1) * (2.0 - 0.0) / 2 for xx in grid]
        B = Matrix{Float64}(T.evaluate_basis(tb, phys, 0:N-1))
        Dn = T._nodal_diff_matrix(phys)
        @test maximum(abs.(Dn * (B * c) .- B * (D * c))) < 1e-9

        # second derivative
        D2 = T.differentiation_matrix(tb, 2)
        @test maximum(abs.(Dn * (Dn * (B * c)) .- B * (D2 * c))) < 1e-7

        # cache
        @test T.differentiation_matrix(tb, 1) === D

        # Legendre sparse recurrence path (reference interval).
        leb = Legendre(xc; size=N, bounds=(-1.0, 1.0))
        DL = T.differentiation_matrix(leb, 1)
        gL = T._native_grid(leb, 1.0)
        BL = Matrix{Float64}(T.evaluate_basis(leb, gL, 0:N-1))
        DnL = T._nodal_diff_matrix(gL)
        cL = [0.2, -0.4, 0.1, 0.3, -0.2, 0.5, 0.0, 0.1]
        @test maximum(abs.(DnL * (BL * cL) .- BL * (DL * cL))) < 1e-9

        # zero-length domain -> ArgumentError
        tbz = ChebyshevT(xc; size=4, bounds=(1.0, 1.0))
        @test_throws ArgumentError T.differentiation_matrix(tbz, 1)
    end

    # ------------------------------------------------------------------
    # low-level differentiation matrices (recurrence path)
    # ------------------------------------------------------------------
    @testset "low-level differentiation matrices" begin
        N = 6
        # _jacobi_differentiation_matrix order 0 -> I
        @test Matrix(T._jacobi_differentiation_matrix(N, -0.5, -0.5, 0)) == Matrix(I, N, N)

        # ChebyshevT recurrence: d/dx T_2 = 4 T_1  (T_2 = 2x^2-1, dT2/dx = 4x = 4 T1)
        Dt = T._jacobi_differentiation_matrix(N, -0.5, -0.5, 1)
        e2 = Float64[0, 0, 1, 0, 0, 0]
        dt = Dt * e2
        @test dt ≈ Float64[0, 4, 0, 0, 0, 0]
        # d/dx T_1 = T_0 (dT1/dx = 1)
        e1 = Float64[0, 1, 0, 0, 0, 0]
        @test Dt * e1 ≈ Float64[1, 0, 0, 0, 0, 0]

        # Legendre recurrence: d/dx P_2 = 3 P_1  (P2=(3x^2-1)/2, dP2=3x=3P1)
        Dl = T._jacobi_differentiation_matrix(N, 0.0, 0.0, 1)
        @test Dl * e2 ≈ Float64[0, 3, 0, 0, 0, 0]
        # d/dx P_3 = 5 P_2 + P_0  (Legendre derivative expansion)
        e3 = Float64[0, 0, 0, 1, 0, 0]
        @test Dl * e3 ≈ Float64[1, 0, 5, 0, 0, 0]

        # _chebyshev_t_differentiation_matrix direct
        Dtc = T._chebyshev_t_differentiation_matrix(N)
        @test Dtc * e2 ≈ Float64[0, 4, 0, 0, 0, 0]
        # N=1 degenerate -> empty/zero matrix (line 762)
        @test T._chebyshev_t_differentiation_matrix(1) == zeros(1, 1)

        # _legendre_differentiation_matrix direct + N=1 degenerate (line 791)
        Dlc = T._legendre_differentiation_matrix(N)
        @test Dlc * e2 ≈ Float64[0, 3, 0, 0, 0, 0]
        @test T._legendre_differentiation_matrix(1) == zeros(1, 1)

        # _general_jacobi_differentiation_matrix: structural only (known-broken).
        # Step-1 raw super-diagonal coeff is (n+a+b+1)/2 BEFORE the (broken)
        # back-conversion; just verify shape and that it produces a matrix.
        Dg = T._general_jacobi_differentiation_matrix(N, 0.5, 0.5)
        @test size(Dg) == (N, N)
        @test eltype(Dg) == Float64
        # N=1 degenerate path returns 1x1 zero (line 818)
        Dg1 = T._general_jacobi_differentiation_matrix(1, 0.5, 0.5)
        @test size(Dg1) == (1, 1)
        @test Dg1[1, 1] == 0.0
    end

    # ------------------------------------------------------------------
    # basis dispatcher helpers
    # ------------------------------------------------------------------
    @testset "basis dispatcher helpers" begin
        # _basis_builder maps each type to its constructor function
        @test T._basis_builder(ChebyshevT) === T._ChebyshevT_constructor
        @test T._basis_builder(ChebyshevU) === T._ChebyshevU_constructor
        @test T._basis_builder(ChebyshevV) === T._ChebyshevV_constructor
        @test T._basis_builder(Legendre) === T._Legendre_constructor
        @test T._basis_builder(Ultraspherical) === T._Ultraspherical_constructor
        @test T._basis_builder(Jacobi) === T._Jacobi_constructor
        @test T._basis_builder(RealFourier) === T._RealFourier_constructor
        @test T._basis_builder(ComplexFourier) === T._ComplexFourier_constructor
        # Unregistered abstract type -> error (line 842)
        @test_throws ErrorException T._basis_builder(Tarang.Basis)

        # dispatch_preprocess: exactly one arg required (line 846)
        @test T.dispatch_preprocess(ChebyshevT, (xc,), NamedTuple()) == ((xc,), NamedTuple())
        @test_throws ArgumentError T.dispatch_preprocess(ChebyshevT, (xc, xc), NamedTuple())
        @test_throws ArgumentError T.dispatch_preprocess(ChebyshevT, (), NamedTuple())

        # dispatch_check: first arg must be a Coordinate (line 854)
        @test T.dispatch_check(ChebyshevT, (xc,), NamedTuple()) === true
        @test_throws ArgumentError T.dispatch_check(ChebyshevT, (5,), NamedTuple())

        # invoke_constructor builds a basis from a Coordinate
        built = T.invoke_constructor(ChebyshevT, (xc,), (; size=12, bounds=(-1.0, 1.0)))
        @test built isa ChebyshevT
        @test built.meta.size == 12
    end

    # ------------------------------------------------------------------
    # Ultraspherical / Jacobi through the public conversion + differentiation
    # (collocation paths) for extra coverage of the JacobiBasis dispatch.
    # ------------------------------------------------------------------
    @testset "Ultraspherical & generic Jacobi (public)" begin
        N = 8
        us = Ultraspherical(xc; alpha=2.0, size=N, bounds=(-1.0, 1.0))
        # differentiation via collocation; compare to nodal derivative.
        Du = T.differentiation_matrix(us, 1)
        g = T._native_grid(us, 1.0)
        Bu = Matrix{Float64}(T.evaluate_basis(us, g, 0:N-1))
        Dn = T._nodal_diff_matrix(g)
        c = [0.1, 0.3, -0.2, 0.5, 0.0, 0.2, -0.1, 0.05]
        @test maximum(abs.(Dn * (Bu * c) .- Bu * (Du * c))) < 1e-8

        # Generic Jacobi conversion to a different Jacobi basis preserves function.
        j0 = Jacobi(xc; a=0.5, b=0.5, size=N, bounds=(-1.0, 1.0))
        j1 = Jacobi(xc; a=1.5, b=1.5, size=N, bounds=(-1.0, 1.0))
        M = T.conversion_matrix(j0, j1)
        @test size(M) == (N, N)
        cj = [0.2, -0.4, 0.1, 0.3, -0.2, 0.5, 0.0, 0.1]
        cj1 = M * cj
        for x in (-0.6, 0.1, 0.7)
            @test jeval(cj, 0.5, 0.5, x) ≈ jeval(cj1, 1.5, 1.5, x)
        end
    end
end
