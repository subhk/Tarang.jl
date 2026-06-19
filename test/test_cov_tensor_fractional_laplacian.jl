"""
Coverage tests for src/core/operators/tensor/tensor_fractional_laplacian.jl

Targets serial-CPU reachable lines that were previously uncovered:
  - evaluate_fractional_laplacian dispatch (ScalarField + VectorField + bad operand)
  - evaluate_scalar_fractional_laplacian (alpha = +1, +0.5, -0.5), 1D / 2D, non-2pi
  - ComplexFourier wavenumber branch in compute_wavenumber_squared_grid
  - "no Fourier basis" ArgumentError
  - matrix_dependence / matrix_coupling / subproblem_matrix (FractionalLaplacian)
  - check_conditions / enforce_conditions / is_linear / operator_order
  - TransposeComponents: _transpose_matrix, subproblem_matrix, deps/coupling/linear

Mathematical contract: on a periodic Fourier field, (-Δ)^s of a single mode
e^{i k x} (or cos(kx)/sin(kx)) equals |k|^{2s} times that mode, where the
physical wavenumber is k = m * 2π/L for integer mode index m on a domain of
length L.
"""

using Test
using Tarang
using LinearAlgebra

const FL = Tarang.FractionalLaplacian
const TC = Tarang.TransposeComponents

# Helper: build a 1D RealFourier scalar field of size N on [0,L).
function make_1d_realfourier(N, L)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
    u = ScalarField(dist, "u", (xb,), Float64)
    mesh = Tarang.create_meshgrid(u.domain)
    return u, mesh["x"], coords, dist, xb
end

@testset "tensor_fractional_laplacian coverage" begin

    # ------------------------------------------------------------------
    # (-Δ)^1 on a single 1D cosine mode == |k|^2 * mode  (alpha=1 branch)
    # ------------------------------------------------------------------
    @testset "1D (-Δ)^1 single mode == k^2 * mode (L=2π)" begin
        N, L = 32, 2π
        u, x, = make_1d_realfourier(N, L)
        m = 3                      # mode index; physical k = m*2π/L = m
        k = m * (2π / L)
        Tarang.get_grid_data(u) .= cos.(m .* x)

        op = FL(u, 1.0)
        res = evaluate(op)         # default layout :g
        ensure_layout!(res, :g)

        expected = (k^2) .* cos.(m .* x)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-9, atol=1e-10)
        # Equivalent to -Laplacian (note sign convention (-Δ)^1 = -Δ)
        lap = evaluate(Laplacian(u)); ensure_layout!(lap, :g)
        @test isapprox(Tarang.get_grid_data(res), -Tarang.get_grid_data(lap);
                       rtol=1e-9, atol=1e-9)
    end

    # ------------------------------------------------------------------
    # (-Δ)^{1/2} on a single mode == |k| * mode  (alpha=0.5, >=0 branch)
    # ------------------------------------------------------------------
    @testset "1D (-Δ)^{1/2} single mode == |k| * mode" begin
        N, L = 32, 2π
        u, x, = make_1d_realfourier(N, L)
        m = 4
        k = m * (2π / L)
        Tarang.get_grid_data(u) .= sin.(m .* x)

        res = evaluate(FL(u, 0.5))
        ensure_layout!(res, :g)
        expected = abs(k) .* sin.(m .* x)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-9, atol=1e-9)
    end

    # ------------------------------------------------------------------
    # (-Δ)^{-1/2} inverse on a single mode == |k|^{-1} * mode (alpha<0 branch).
    # The k=0 (mean) mode is annihilated.
    # ------------------------------------------------------------------
    @testset "1D (-Δ)^{-1/2} inverse single mode == |k|^{-1} * mode, mean killed" begin
        N, L = 32, 2π
        u, x, = make_1d_realfourier(N, L)
        m = 5
        k = m * (2π / L)
        mean_val = 2.7
        Tarang.get_grid_data(u) .= mean_val .+ cos.(m .* x)

        res = evaluate(FL(u, -0.5))
        ensure_layout!(res, :g)
        # mean removed; cosine scaled by 1/|k|
        expected = (1.0 / abs(k)) .* cos.(m .* x)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-8, atol=1e-9)
        # round trip: (-Δ)^{1/2} of the inverse recovers the (mean-removed) field
        back = evaluate(FL(res, 0.5)); ensure_layout!(back, :g)
        @test isapprox(Tarang.get_grid_data(back), cos.(m .* x); rtol=1e-8, atol=1e-8)
    end

    # ------------------------------------------------------------------
    # Non-2π domain: physical k uses k0 = 2π/L scaling.
    # ------------------------------------------------------------------
    @testset "1D non-2π domain k0=2π/L scaling" begin
        N, L = 24, 4.0                 # k0 = 2π/4 = π/2
        u, x, = make_1d_realfourier(N, L)
        m = 3
        k = m * (2π / L)
        Tarang.get_grid_data(u) .= cos.(m .* (2π / L) .* x)

        res = evaluate(FL(u, 1.0)); ensure_layout!(res, :g)
        expected = (k^2) .* cos.(m .* (2π / L) .* x)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-8, atol=1e-9)
    end

    # ------------------------------------------------------------------
    # 2D: (-Δ)^s of cos(k x) cos(l y) == (k^2+l^2)^s * mode.
    # Exercises the non-first RealFourier axis (wavenumbers_fft branch),
    # plus the additive |k|^2 accumulation across axes.
    # ------------------------------------------------------------------
    @testset "2D (-Δ)^1 == (k^2+l^2) * mode" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        u = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x, y = mesh["x"], mesh["y"]

        mk, ml = 2, 3
        k = mk * (2π / L); l = ml * (2π / L)
        Tarang.get_grid_data(u) .= cos.(mk .* x) .* cos.(ml .* y)

        res = evaluate(FL(u, 1.0)); ensure_layout!(res, :g)
        expected = (k^2 + l^2) .* cos.(mk .* x) .* cos.(ml .* y)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-9, atol=1e-9)
    end

    @testset "2D (-Δ)^{1/2} == sqrt(k^2+l^2) * mode" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        u = ScalarField(dist, "u", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(u.domain)
        x, y = mesh["x"], mesh["y"]

        mk, ml = 2, 3
        k = mk * (2π / L); l = ml * (2π / L)
        Tarang.get_grid_data(u) .= cos.(mk .* x) .* cos.(ml .* y)

        res = evaluate(FL(u, 0.5)); ensure_layout!(res, :g)
        expected = sqrt(k^2 + l^2) .* cos.(mk .* x) .* cos.(ml .* y)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-8, atol=1e-9)
    end

    # ------------------------------------------------------------------
    # ComplexFourier branch in compute_wavenumber_squared_grid.
    # ------------------------------------------------------------------
    @testset "ComplexFourier basis (-Δ)^1 single mode" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, L))
        u = ScalarField(dist, "u", (xb,), ComplexF64)
        mesh = Tarang.create_meshgrid(u.domain)
        x = mesh["x"]

        m = 3
        k = m * (2π / L)
        Tarang.get_grid_data(u) .= exp.(im .* m .* x)

        res = evaluate(FL(u, 1.0)); ensure_layout!(res, :g)
        expected = (k^2) .* exp.(im .* m .* x)
        @test isapprox(Tarang.get_grid_data(res), expected; rtol=1e-8, atol=1e-9)
    end

    # ------------------------------------------------------------------
    # Dispatch: VectorField path (evaluate_vector_fractional_laplacian).
    # ------------------------------------------------------------------
    @testset "VectorField (-Δ)^1 applied component-wise" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        v = VectorField(dist, coords, "v", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(v.components[1].domain)
        x, y = mesh["x"], mesh["y"]

        mk, ml = 1, 2
        k = mk * (2π / L); l = ml * (2π / L)
        Tarang.get_grid_data(v.components[1]) .= cos.(mk .* x) .* cos.(ml .* y)
        Tarang.get_grid_data(v.components[2]) .= sin.(mk .* x) .* sin.(ml .* y)

        op = FL(v, 1.0)
        res = evaluate_fractional_laplacian(op, :g)
        @test res isa VectorField
        ensure_layout!(res.components[1], :g)
        ensure_layout!(res.components[2], :g)
        exp1 = (k^2 + l^2) .* cos.(mk .* x) .* cos.(ml .* y)
        exp2 = (k^2 + l^2) .* sin.(mk .* x) .* sin.(ml .* y)
        @test isapprox(Tarang.get_grid_data(res.components[1]), exp1; rtol=1e-9, atol=1e-9)
        @test isapprox(Tarang.get_grid_data(res.components[2]), exp2; rtol=1e-9, atol=1e-9)
    end

    @testset "VectorField (-Δ)^1 :c (coeff) output layout" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        v = VectorField(dist, coords, "v", (xb, yb), Float64)
        mesh = Tarang.create_meshgrid(v.components[1].domain)
        x, y = mesh["x"], mesh["y"]
        Tarang.get_grid_data(v.components[1]) .= cos.(2 .* x)
        Tarang.get_grid_data(v.components[2]) .= cos.(3 .* y)

        # layout = :c exercises the coeff copyto! branch
        res = evaluate_fractional_laplacian(FL(v, 1.0), :c)
        @test res isa VectorField
        # coeff data must be finite and same shape as the per-component coeff data
        cd1 = Tarang.get_coeff_data(res.components[1])
        @test all(isfinite, cd1)
        @test size(cd1) == size(Tarang.get_coeff_data(v.components[1]))
    end

    # ------------------------------------------------------------------
    # Error conditions.
    # ------------------------------------------------------------------
    @testset "ArgumentError: no Fourier basis (pure Chebyshev)" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["x"]; size=16, bounds=(-1.0, 1.0))
        f = ScalarField(dist, "f", (zb,), Float64)
        @test_throws ArgumentError evaluate_fractional_laplacian(FL(f, 0.5), :g)
    end

    # ------------------------------------------------------------------
    # Matrix interface: matrix_dependence / matrix_coupling.
    # ------------------------------------------------------------------
    @testset "matrix_dependence / matrix_coupling (FractionalLaplacian)" begin
        N, L = 16, 2π
        u, x, coords, dist, xb = make_1d_realfourier(N, L)
        w = ScalarField(dist, "w", (xb,), Float64)

        op = FL(u, 0.5)
        dep = Tarang.matrix_dependence(op, u, w)
        @test dep == [true, false]            # depends on u (self), not w
        coup = Tarang.matrix_coupling(op, u, w)
        @test coup == [true, false]           # couples only to itself
        # depends-by-name even for a distinct object with same name
        u2 = ScalarField(dist, "u", (xb,), Float64)
        dep2 = Tarang.matrix_dependence(op, u2)
        @test dep2 == [true]
    end

    # ------------------------------------------------------------------
    # subproblem_matrix (FractionalLaplacian): diagonal |k|^{2α}.
    # ------------------------------------------------------------------
    @testset "subproblem_matrix (FractionalLaplacian) diagonal == |k|^{2α}" begin
        N, L = 16, 2π
        u, x, = make_1d_realfourier(N, L)
        ensure_layout!(u, :c)

        alpha = 1.0
        M = Tarang.subproblem_matrix(FL(u, alpha), nothing)
        ncoeff = length(Tarang.get_coeff_data(u))
        @test size(M) == (ncoeff, ncoeff)
        # Diagonal entries must equal |k|^2 from the coefficient grid
        kgrid = Tarang.compute_wavenumber_squared_grid(u)
        @test isapprox(Vector(diag(M)), vec(kgrid) .^ alpha; rtol=1e-12, atol=1e-12)
        # off-diagonal must be zero (diagonal operator)
        Md = Matrix(M)
        @test all(Md[i, j] == 0.0 for i in 1:ncoeff, j in 1:ncoeff if i != j)
    end

    @testset "subproblem_matrix (FractionalLaplacian) alpha<0 kills k=0" begin
        N, L = 16, 2π
        u, x, = make_1d_realfourier(N, L)
        ensure_layout!(u, :c)
        M = Tarang.subproblem_matrix(FL(u, -0.5), nothing)
        d = Vector(diag(M))
        kgrid = vec(Tarang.compute_wavenumber_squared_grid(u))
        # k=0 entry -> 0; nonzero-k entries -> k^{-1}
        for (i, k2) in enumerate(kgrid)
            if k2 > 1e-14
                @test isapprox(d[i], k2 ^ (-0.5); rtol=1e-10, atol=1e-12)
            else
                @test d[i] == 0.0
            end
        end
    end

    @testset "subproblem_matrix (FractionalLaplacian) rejects VectorField" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        v = VectorField(dist, coords, "v", (xb,), Float64)
        @test_throws ArgumentError Tarang.subproblem_matrix(FL(v, 0.5), nothing)
    end

    # ------------------------------------------------------------------
    # check_conditions / enforce_conditions / is_linear / operator_order.
    # ------------------------------------------------------------------
    @testset "check/enforce conditions + linearity + order (scalar)" begin
        N, L = 16, 2π
        u, x, = make_1d_realfourier(N, L)
        Tarang.get_grid_data(u) .= cos.(2 .* x)     # :g layout

        op = FL(u, 0.5)
        @test Tarang.is_linear(op) == true
        @test Tarang.operator_order(op) == 1.0      # 2*alpha
        @test Tarang.operator_order(FL(u, 1.0)) == 2.0

        # check_conditions returns a Bool for both layouts
        @test Tarang.check_conditions(op) == true   # currently :g
        Tarang.enforce_conditions(op)               # forces :c
        @test u.current_layout == :c
        @test Tarang.check_conditions(op) == true   # now :c
    end

    @testset "check/enforce conditions (vector)" begin
        N, L = 16, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        v = VectorField(dist, coords, "v", (xb, yb), Float64)
        Tarang.get_grid_data(v.components[1]) .= 0.0
        Tarang.get_grid_data(v.components[2]) .= 0.0

        op = FL(v, 0.5)
        @test Tarang.check_conditions(op) == true
        Tarang.enforce_conditions(op)
        @test v.components[1].current_layout == :c
        @test v.components[2].current_layout == :c
    end

    # ------------------------------------------------------------------
    # TransposeComponents matrix methods.
    # ------------------------------------------------------------------
    @testset "TransposeComponents _transpose_matrix + subproblem_matrix" begin
        N, L = 8, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        T = TensorField(dist, coords, "T", (xb, yb), Float64)

        op = TC(T, (1, 2))

        # permutation matrix: dim=2 -> 4x4 swapping T_{ij} <-> T_{ji}.
        P = Tarang._transpose_matrix(op)
        @test size(P) == (4, 4)
        Pd = Matrix(P)
        # flat index map (1-based): (i,j) -> i*dim+j+1 with i,j in 0..1
        #   T11(1)->1, T12(2)->3, T21(3)->2, T22(4)->4
        @test Pd[1, 1] == 1.0
        @test Pd[3, 2] == 1.0    # input idx2 (T12) -> output idx3 (T21)
        @test Pd[2, 3] == 1.0    # input idx3 (T21) -> output idx2 (T12)
        @test Pd[4, 4] == 1.0
        @test sum(Pd) == 4.0     # exactly one 1 per column
        # applying P twice is identity (transpose is an involution)
        @test Matrix(P * P) == Matrix(I, 4, 4)

        # subproblem_matrix kron's in the per-coefficient identity.
        M = Tarang.subproblem_matrix(op, nothing)
        coeff_size = (N) * (N)     # bases.meta.size product (native length)
        @test size(M) == (4 * coeff_size, 4 * coeff_size)
        # block structure: M = kron(P, I_coeff). Verify against a dense kron.
        @test Matrix(M) == kron(Pd, Matrix{Float64}(I, coeff_size, coeff_size))
    end

    @testset "TransposeComponents linear / deps / coupling / conditions" begin
        N, L = 8, 2π
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        T = TensorField(dist, coords, "T", (xb, yb), Float64)
        op = TC(T, (1, 2))

        @test Tarang.is_linear(op) == true
        @test Tarang.check_conditions(op) == true
        @test Tarang.enforce_conditions(op) === nothing
        # deps/coupling: operand has no matrix_dependence method -> falses
        somefield = ScalarField(dist, "z", (xb, yb), Float64)
        @test Tarang.matrix_dependence(op, somefield) == falses(1)
        @test Tarang.matrix_coupling(op, somefield) == falses(1)
    end

    @testset "_transpose_matrix rejects non-tensor operand" begin
        N, L = 8, 2π
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        u = ScalarField(dist, "u", (xb,), Float64)
        # Build a TransposeComponents-like by bypassing the ctor guard is not
        # possible (ctor requires TensorField). Instead verify the ctor itself
        # rejects non-tensors, which is the user-facing guard.
        @test_throws ArgumentError TC(u, (1, 2))
    end

end
