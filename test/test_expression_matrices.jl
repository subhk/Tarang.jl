using Test
using Tarang
using SparseArrays
using LinearAlgebra

@testset "Expression Matrices — Base Cases and Arithmetic" begin

    # ── Setup: build domain, fields, and a mock subproblem ──────────────────
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 4.0))
    zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    domain = Domain(dist, (xb, zb))

    # 2D scalar field on full domain
    b = ScalarField(domain, "b")
    # 2D vector field on full domain (2 components for 2D coords)
    u = VectorField(domain, "u")
    # 1D tau field — Fourier only, no Chebyshev basis
    tau_b = ScalarField(dist, "tau_b", (xb,), Float64)
    # 0D tau field — no bases at all
    tau_p = ScalarField(dist, "tau_p", (), Float64)

    problem = IVP([b, u, tau_b, tau_p])

    # Build a mock subproblem with group (5, nothing):
    #   Fourier mode 5 (separable), Chebyshev fully coupled
    # Reuse the same mock types from test_subsystems.jl
    struct ExprMatSolverBase
        matrix_coupling::Vector{Bool}
    end
    struct ExprMatSolver
        problem::Problem
        base::ExprMatSolverBase
    end
    # Fourier separable (false), Chebyshev coupled (true)
    solver = ExprMatSolver(problem, ExprMatSolverBase([false, true]))
    subsys = Subsystem(solver, (5, nothing))
    sp = Subproblem(solver, (subsys,), (5, nothing))

    Nz = 8  # ChebyshevT size — the coupled dimension

    # ── Base cases ──────────────────────────────────────────────────────────

    @testset "ScalarField in vars -> identity" begin
        vars = [b]
        result = expression_matrices(b, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "ScalarField NOT in vars -> empty" begin
        other = ScalarField(domain, "other_field")
        vars = [other]
        result = expression_matrices(b, sp, vars)
        @test isempty(result)
    end

    @testset "VectorField in vars -> identity" begin
        vars = [u]
        result = expression_matrices(u, sp, vars)
        @test haskey(result, u)
        mat = result[u]
        # VectorField size = 2 * Nz = 16
        n_vec = 2 * Nz
        @test size(mat) == (n_vec, n_vec)
        @test mat ≈ sparse(ComplexF64(1)*I, n_vec, n_vec)
    end

    @testset "0D tau -> 1x1 identity" begin
        vars = [tau_p]
        result = expression_matrices(tau_p, sp, vars)
        @test haskey(result, tau_p)
        mat = result[tau_p]
        @test size(mat) == (1, 1)
        @test mat[1,1] ≈ ComplexF64(1)
    end

    @testset "Number -> empty" begin
        vars = [b]
        result = expression_matrices(42, sp, vars)
        @test isempty(result)
    end

    @testset "Nothing -> empty" begin
        vars = [b]
        result = expression_matrices(nothing, sp, vars)
        @test isempty(result)
    end

    # ── Arithmetic operators ────────────────────────────────────────────────

    @testset "AddOperator -> merges and sums" begin
        vars = [b]
        # b + b should produce 2*I for b
        add_op = AddOperator(b, b)
        result = expression_matrices(add_op, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ 2 * sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "AddOperator with different fields" begin
        other = ScalarField(domain, "c")
        prob2 = IVP([b, other])
        solver2 = ExprMatSolver(prob2, ExprMatSolverBase([false, true]))
        subsys2 = Subsystem(solver2, (5, nothing))
        sp2 = Subproblem(solver2, (subsys2,), (5, nothing))

        vars = [b, other]
        add_op = AddOperator(b, other)
        result = expression_matrices(add_op, sp2, vars)
        @test haskey(result, b)
        @test haskey(result, other)
        @test result[b] ≈ sparse(ComplexF64(1)*I, Nz, Nz)
        @test result[other] ≈ sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "SubtractOperator" begin
        other = ScalarField(domain, "d")
        prob3 = IVP([b, other])
        solver3 = ExprMatSolver(prob3, ExprMatSolverBase([false, true]))
        subsys3 = Subsystem(solver3, (5, nothing))
        sp3 = Subproblem(solver3, (subsys3,), (5, nothing))

        vars = [b, other]
        sub_op = SubtractOperator(b, other)
        result = expression_matrices(sub_op, sp3, vars)
        @test haskey(result, b)
        @test haskey(result, other)
        @test result[b] ≈ sparse(ComplexF64(1)*I, Nz, Nz)
        @test result[other] ≈ -sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "NegateOperator -> negates" begin
        vars = [b]
        neg_op = NegateOperator(b)
        result = expression_matrices(neg_op, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ -sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "MultiplyOperator with scalar constant" begin
        vars = [b]
        # 3.0 * b should produce 3*I for b
        mul_op = MultiplyOperator(3.0, b)
        result = expression_matrices(mul_op, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ 3.0 * sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "MultiplyOperator with scalar constant on right" begin
        vars = [b]
        # b * 2.5 should produce 2.5*I for b
        mul_op = MultiplyOperator(b, 2.5)
        result = expression_matrices(mul_op, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ 2.5 * sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "MultiplyOperator with constant VectorField (b * ez)" begin
        vars = [b]
        # Create a constant unit vector ez = [0; 1] for 2D
        # Unit vector components: ex = [1, 0], ez = [0, 1]
        ez = VectorField(dist, coords, "ez", (), Float64)
        # Set component values: ez_x = 0, ez_z = 1
        # ez has 2 components (x, z) with no bases (0D constant)
        # Component 1 (x): 0.0, Component 2 (z): 1.0
        # For 0D fields, grid data is a 0-dim or 1-element array
        # We need to initialize the data
        if get_grid_data(ez.components[1]) !== nothing
            get_grid_data(ez.components[1]) .= 0.0
        end
        if get_grid_data(ez.components[2]) !== nothing
            get_grid_data(ez.components[2]) .= 1.0
        end

        mul_op = MultiplyOperator(b, ez)
        result = expression_matrices(mul_op, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        # b * ez should produce a (2*Nz x Nz) block expansion:
        # [ 0*I_Nz ]   (ez_x = 0)
        # [ 1*I_Nz ]   (ez_z = 1)
        @test size(mat) == (2*Nz, Nz)
        # Top block (x component) should be zero
        @test mat[1:Nz, :] ≈ spzeros(ComplexF64, Nz, Nz)
        # Bottom block (z component) should be identity
        @test mat[Nz+1:2*Nz, :] ≈ sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "MultiplyOperator nonlinear (both depend on vars) -> empty" begin
        vars = [b]
        # b * b is nonlinear
        mul_op = MultiplyOperator(b, b)
        result = expression_matrices(mul_op, sp, vars)
        @test isempty(result)
    end

    @testset "Generic Operator fallback with operand" begin
        vars = [b]
        # NegateOperator is a single-operand operator with no subproblem_matrix
        # It has its own method, but let's test composition:
        # -(-(b)) should give identity for b
        double_neg = NegateOperator(NegateOperator(b))
        result = expression_matrices(double_neg, sp, vars)
        @test haskey(result, b)
        @test result[b] ≈ sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "Composition: Add + Negate + Multiply" begin
        vars = [b]
        # 3*b + (-2*b) should give (3-2)*I = 1*I for b
        expr = AddOperator(
            MultiplyOperator(3.0, b),
            NegateOperator(MultiplyOperator(2.0, b))
        )
        result = expression_matrices(expr, sp, vars)
        @test haskey(result, b)
        @test result[b] ≈ sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "Composition: Subtract with scalars" begin
        vars = [b]
        # 5*b - 3*b should give 2*I for b
        expr = SubtractOperator(
            MultiplyOperator(5.0, b),
            MultiplyOperator(3.0, b)
        )
        result = expression_matrices(expr, sp, vars)
        @test haskey(result, b)
        @test result[b] ≈ 2.0 * sparse(ComplexF64(1)*I, Nz, Nz)
    end

    # ── Linear operators (Task 3) ──────────────────────────────────────────

    # Precompute expected wavenumber for Fourier mode 5
    Lx = 4.0  # bounds = (0.0, 4.0) for xb
    kx = 5 * 2π / Lx  # Fourier wavenumber for nx=5

    @testset "TimeDerivative(b) -> identity" begin
        vars = [b]
        dt_b = TimeDerivative(b)
        result = expression_matrices(dt_b, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ sparse(ComplexF64(1)*I, Nz, Nz)
    end

    @testset "TimeDerivative(u) -> identity for vector" begin
        vars = [u]
        dt_u = TimeDerivative(u)
        result = expression_matrices(dt_u, sp, vars)
        @test haskey(result, u)
        mat = result[u]
        n_vec = 2 * Nz
        @test size(mat) == (n_vec, n_vec)
        @test mat ≈ sparse(ComplexF64(1)*I, n_vec, n_vec)
    end

    @testset "Differentiate(b, z, 1) -> Chebyshev diff matrix" begin
        vars = [b]
        dz_b = Differentiate(b, coords["z"], 1)
        result = expression_matrices(dz_b, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        # Compare with directly computed Chebyshev diff matrix
        D_z = ComplexF64.(differentiation_matrix(zb, 1))
        @test mat ≈ sparse(D_z)
    end

    @testset "Differentiate(b, x, 1) -> (im*kx)*I" begin
        vars = [b]
        dx_b = Differentiate(b, coords["x"], 1)
        result = expression_matrices(dx_b, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        @test mat ≈ sparse((im * kx) * ComplexF64(1) * I, Nz, Nz)
    end

    @testset "Laplacian(b) -> -kx^2 * I + D_z^2" begin
        vars = [b]
        lap_b = Laplacian(b)
        result = expression_matrices(lap_b, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        @test size(mat) == (Nz, Nz)
        D_z2 = ComplexF64.(differentiation_matrix(zb, 2))
        expected = sparse((-kx^2) * ComplexF64(1) * I, Nz, Nz) + sparse(D_z2)
        @test mat ≈ expected
    end

    @testset "Gradient(b, coords) -> [D_x; D_z]" begin
        vars = [b]
        grad_b = Gradient(b, coords)
        result = expression_matrices(grad_b, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        # Gradient of scalar: (ndim*Nz) x Nz = (2*8) x 8
        @test size(mat) == (2*Nz, Nz)
        # Top block: D_x = (im*kx)*I
        D_x = sparse((im * kx) * ComplexF64(1) * I, Nz, Nz)
        D_z = sparse(ComplexF64.(differentiation_matrix(zb, 1)))
        expected = vcat(D_x, D_z)
        @test mat ≈ expected
    end

    @testset "Divergence(u) -> [D_x, D_z]" begin
        vars = [u]
        div_u = Divergence(u)
        result = expression_matrices(div_u, sp, vars)
        @test haskey(result, u)
        mat = result[u]
        # Divergence of vector: Nz x (ndim*Nz) = 8 x 16
        n_vec = 2 * Nz
        @test size(mat) == (Nz, n_vec)
        D_x = sparse((im * kx) * ComplexF64(1) * I, Nz, Nz)
        D_z = sparse(ComplexF64.(differentiation_matrix(zb, 1)))
        expected = hcat(D_x, D_z)
        @test mat ≈ expected
    end

    @testset "Trace(Gradient(u, coords)) -> div(u) = [D_x, D_z]" begin
        vars = [u]
        # Gradient of VectorField u produces a tensor-like (ndim*n_vec x n_vec) matrix
        # Trace contracts it back to (Nz x n_vec)
        grad_u = Gradient(u, coords)
        trace_grad_u = Trace(grad_u)
        result = expression_matrices(trace_grad_u, sp, vars)
        @test haskey(result, u)
        mat = result[u]
        n_vec = 2 * Nz
        # Trace(Gradient(u)) = Divergence for Cartesian coordinates
        # Result: Nz x n_vec
        @test size(mat) == (Nz, n_vec)
        # Expected: trace selects diagonal blocks of the gradient tensor
        D_x = sparse((im * kx) * ComplexF64(1) * I, Nz, Nz)
        D_z = sparse(ComplexF64.(differentiation_matrix(zb, 1)))
        expected_div = hcat(D_x, D_z)
        @test mat ≈ expected_div
    end

    @testset "Composition: scalar * Laplacian" begin
        vars = [b]
        # nu * Laplacian(b) with nu = 0.1
        nu = 0.1
        expr = MultiplyOperator(nu, Laplacian(b))
        result = expression_matrices(expr, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        D_z2 = ComplexF64.(differentiation_matrix(zb, 2))
        expected = nu * (sparse((-kx^2) * ComplexF64(1) * I, Nz, Nz) + sparse(D_z2))
        @test mat ≈ expected
    end

    @testset "Composition: TimeDerivative(b) - nu*Laplacian(b)" begin
        vars = [b]
        nu = 0.5
        expr = SubtractOperator(
            TimeDerivative(b),
            MultiplyOperator(nu, Laplacian(b))
        )
        result = expression_matrices(expr, sp, vars)
        @test haskey(result, b)
        mat = result[b]
        I_Nz = sparse(ComplexF64(1)*I, Nz, Nz)
        D_z2 = ComplexF64.(differentiation_matrix(zb, 2))
        lap = sparse((-kx^2) * ComplexF64(1) * I, Nz, Nz) + sparse(D_z2)
        expected = I_Nz - nu * lap
        @test mat ≈ expected
    end
end
