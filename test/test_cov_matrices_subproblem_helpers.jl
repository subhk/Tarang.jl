using Test
using Tarang
using LinearAlgebra
using SparseArrays

# Coverage tests for src/core/operators/matrices/matrices_subproblem_helpers.jl
#
# These exercise the per-coordinate subproblem matrix helpers on the serial CPU
# path: Fourier wavenumber lookup, Chebyshev/Fourier differentiation matrices,
# operand-field/coordsys resolution, group-index lookup and the integration
# step matrix (Fourier DC/non-DC + Chebyshev quadrature weights).
#
# A subproblem is built directly using the same lightweight mock-solver pattern
# as test/test_subsystems.jl (a struct exposing `problem` + `base`), then passed
# to the (un-exported) helpers via `Tarang.`.

import Tarang: _subproblem_kx, _subproblem_cheb_basis, _subproblem_diff_matrix,
    _get_operand_coordsys, _resolve_operand_field, _operand_basis_for_coord,
    _subproblem_group_index, _integration_step_matrix,
    differentiation_matrix, wavenumbers_rfft, wavenumbers_fft

# Mock solver carrying just what Subsystem/Subproblem read: `problem` + `base`.
struct _HelperMockBase
    matrix_coupling::Vector{Bool}
end
struct _HelperMockSolver
    problem::Tarang.Problem
    base::_HelperMockBase
end

# Build a (Fourier-x, Chebyshev-z) subproblem for a chosen x mode index.
function _make_xz_sp(xmode; xsize=8, zsize=6, xbounds=(0.0, 2π), zbounds=(-1.0, 1.0))
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=xsize, bounds=xbounds)
    zb = ChebyshevT(coords["z"]; size=zsize, bounds=zbounds)
    domain = Domain(dist, (xb, zb))
    b = ScalarField(domain, "b")
    problem = IVP([b])
    solver = _HelperMockSolver(problem, _HelperMockBase([false, true]))
    subsys = Subsystem(solver, (xmode, nothing))
    sp = Subproblem(solver, (subsys,), (xmode, nothing))
    return (; sp, xb, zb, b, domain, dist)
end

@testset "matrices_subproblem_helpers coverage" begin

    @testset "_subproblem_kx: rfft first-axis wavenumber" begin
        # group (3, nothing): x is the first (and only) RealFourier axis ->
        # rfft layout, so coefficient index 3 has physical wavenumber k=3.
        t = _make_xz_sp(3)
        @test t.sp.group_dict == Dict("nx" => 3)
        @test _subproblem_kx(t.sp, "x") == 3.0
        @test _subproblem_kx(t.sp, "x") == real(wavenumbers_rfft(t.xb)[4])
        # z is not a Fourier axis -> 0; unknown coord -> 0 (haskey miss).
        @test _subproblem_kx(t.sp, "z") == 0.0
        @test _subproblem_kx(t.sp, "nope") == 0.0
    end

    @testset "_subproblem_kx: second Fourier axis uses fft (signed) layout" begin
        # Two RealFourier axes (x,y). The FIRST is rfft; the SECOND is full fft,
        # so a high coefficient index wraps to a NEGATIVE wavenumber.
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        f = ScalarField(Domain(dist, (xb, yb)), "f")
        problem = IVP([f])
        solver = _HelperMockSolver(problem, _HelperMockBase([false, false]))
        # x mode 1, y coefficient index 6 -> fft layout k = 6 - 8 = -2.
        subsys = Subsystem(solver, (1, 6))
        sp = Subproblem(solver, (subsys,), (1, 6))

        @test _subproblem_kx(sp, "x") == 1.0                     # rfft axis
        @test _subproblem_kx(sp, "y") == -2.0                    # fft axis, signed
        @test _subproblem_kx(sp, "y") == real(wavenumbers_fft(yb)[7])
    end

    @testset "_subproblem_cheb_basis" begin
        t = _make_xz_sp(2)
        cb = _subproblem_cheb_basis(t.sp)
        @test cb === t.zb                       # returns the actual ChebyshevT basis
        @test isa(cb, Tarang.JacobiBasis)

        # No Chebyshev/Jacobi basis present -> nothing.
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        f = ScalarField(Domain(dist, (xb,)), "f")
        problem = IVP([f])
        solver = _HelperMockSolver(problem, _HelperMockBase([false]))
        subsys = Subsystem(solver, (nothing,))
        sp = Subproblem(solver, (subsys,), (nothing,))
        @test _subproblem_cheb_basis(sp) === nothing
    end

    @testset "_subproblem_diff_matrix: Chebyshev == differentiation_matrix" begin
        t = _make_xz_sp(2; zsize=6)
        for order in (1, 2)
            D = _subproblem_diff_matrix(t.sp, "z", order, 6)
            @test size(D) == (6, 6)
            @test eltype(D) == ComplexF64
            @test Matrix(D) ≈ ComplexF64.(differentiation_matrix(t.zb, order))
        end
    end

    @testset "_subproblem_diff_matrix: Chebyshev block-diagonal kron path" begin
        # Nz a multiple of the basis size -> kron(I_ncomp, D) (multi-component).
        t = _make_xz_sp(2; zsize=6)
        D1 = Matrix(ComplexF64.(differentiation_matrix(t.zb, 1)))
        Dblock = _subproblem_diff_matrix(t.sp, "z", 1, 12)   # 2 components
        @test size(Dblock) == (12, 12)
        full = Matrix(Dblock)
        @test full[1:6, 1:6] ≈ D1
        @test full[7:12, 7:12] ≈ D1
        @test all(iszero, full[1:6, 7:12])      # block off-diagonals are zero
        @test all(iszero, full[7:12, 1:6])
    end

    @testset "_subproblem_diff_matrix: Fourier (im*kx)^order * I" begin
        t = _make_xz_sp(2)               # kx = 2
        D1 = _subproblem_diff_matrix(t.sp, "x", 1, 6)
        @test size(D1) == (6, 6)
        @test Matrix(D1) ≈ (im * 2.0) * Matrix(I, 6, 6)        # first derivative
        D2 = _subproblem_diff_matrix(t.sp, "x", 2, 6)
        @test Matrix(D2) ≈ (im * 2.0)^2 * Matrix(I, 6, 6)      # -kx^2 * I
        @test real(diag(Matrix(D2))[1]) == -4.0

        # DC mode (kx = 0): first derivative is the zero matrix.
        t0 = _make_xz_sp(0)
        D0 = _subproblem_diff_matrix(t0.sp, "x", 1, 6)
        @test nnz(D0) == 0
        @test size(D0) == (6, 6)
    end

    @testset "_subproblem_diff_matrix: fallback zero matrix for unknown coord" begin
        t = _make_xz_sp(1)
        Z = _subproblem_diff_matrix(t.sp, "no_such_coord", 1, 5)
        @test size(Z) == (5, 5)
        @test nnz(Z) == 0
        @test eltype(Z) == ComplexF64
    end

    @testset "_get_operand_coordsys" begin
        t = _make_xz_sp(1)
        # Gradient carries `coordsys` directly.
        g = grad(t.b)
        cs = _get_operand_coordsys(g)
        @test cs !== nothing
        @test cs.dim == 2

        # A bare number has no coordsys -> nothing.
        @test _get_operand_coordsys(3.0) === nothing
    end

    @testset "_resolve_operand_field walks operator tree to leaf field" begin
        t = _make_xz_sp(1)
        # Field returns itself.
        @test _resolve_operand_field(t.b) === t.b
        # Single-operand operator (.operand) resolves through to the field.
        @test _resolve_operand_field(grad(t.b)) === t.b
        @test _resolve_operand_field(lap(t.b)) === t.b
        # A constant resolves to nothing (no field anywhere).
        @test _resolve_operand_field(2.5) === nothing
    end

    @testset "_operand_basis_for_coord" begin
        t = _make_xz_sp(1)
        @test _operand_basis_for_coord(t.b, "z") === t.zb
        @test _operand_basis_for_coord(t.b, "x") === t.xb
        @test _operand_basis_for_coord(t.b, "missing") === nothing
        # Resolves through an operator to the same leaf bases.
        @test _operand_basis_for_coord(grad(t.b), "z") === t.zb
        # No field anywhere -> nothing.
        @test _operand_basis_for_coord(1.0, "x") === nothing
    end

    @testset "_subproblem_group_index" begin
        t = _make_xz_sp(3)
        # x axis is in dist.coords at axis 1 with group entry 3.
        @test _subproblem_group_index(t.sp, "x") == 3
        # z axis group entry is `nothing` (coupled).
        @test _subproblem_group_index(t.sp, "z") === nothing
        # A coordinate name not in dist.coords -> nothing.
        @test _subproblem_group_index(t.sp, "missing") === nothing

        t0 = _make_xz_sp(0)
        @test _subproblem_group_index(t0.sp, "x") == 0
    end

    @testset "_integration_step_matrix: Fourier non-DC -> zero row" begin
        # Non-DC Fourier mode integrates to zero over the period: returns a
        # 1 x nrows zero row (kept square by later tau-column filtering).
        t = _make_xz_sp(3)
        M = _integration_step_matrix(t.xb, "x", t.sp, 6)
        @test size(M) == (1, 6)
        @test nnz(M) == 0
        @test eltype(M) == ComplexF64
    end

    @testset "_integration_step_matrix: Fourier DC -> L * I" begin
        # DC mode (group 0): integral over the period is L * (mean) -> L * I.
        t = _make_xz_sp(0)
        L = t.xb.meta.bounds[2] - t.xb.meta.bounds[1]
        M = _integration_step_matrix(t.xb, "x", t.sp, 6)
        @test size(M) == (6, 6)
        @test Matrix(M) ≈ ComplexF64(L) * Matrix(I, 6, 6)
        @test real(M[1, 1]) ≈ 2π
    end

    @testset "_integration_step_matrix: Chebyshev quadrature weights integrate exactly" begin
        # On [-1,1]: row of weights w·c gives ∫f over the interval.
        t = _make_xz_sp(0; zsize=8, zbounds=(-1.0, 1.0))
        w = _integration_step_matrix(t.zb, "z", t.sp, 8)
        @test size(w) == (1, 8)

        # ∫ 1 dz = 2  (T_0 coefficient = 1).
        c1 = zeros(ComplexF64, 8); c1[1] = 1.0
        @test real((Matrix(w) * c1)[1]) ≈ 2.0

        # ∫ z^2 dz = 2/3.  z^2 = (T_0 + T_2)/2 -> c0 = c2 = 0.5.
        cz2 = zeros(ComplexF64, 8); cz2[1] = 0.5; cz2[3] = 0.5
        @test real((Matrix(w) * cz2)[1]) ≈ 2/3

        # Odd-degree integrands integrate to zero: ∫ z dz = 0 (z = T_1).
        cz = zeros(ComplexF64, 8); cz[2] = 1.0
        @test abs((Matrix(w) * cz)[1]) < 1e-12
    end

    @testset "_integration_step_matrix: Chebyshev on a scaled domain [0,2]" begin
        # The L/2 Jacobian must be applied. ∫_0^2 z^2 dz = 8/3.
        t = _make_xz_sp(0; zsize=8, zbounds=(0.0, 2.0))
        w = _integration_step_matrix(t.zb, "z", t.sp, 8)
        # z = t + 1 on t∈[-1,1]; z^2 = t^2 + 2t + 1 ->
        #   c0 = 0.5 + 1 = 1.5, c1 = 2, c2 = 0.5.
        c = zeros(ComplexF64, 8); c[1] = 1.5; c[2] = 2.0; c[3] = 0.5
        @test real((Matrix(w) * c)[1]) ≈ 8/3
    end

    @testset "_integration_step_matrix: Chebyshev nrows edge cases" begin
        t = _make_xz_sp(0; zsize=8, zbounds=(-1.0, 1.0))
        # nrows == 0 -> empty 0x0.
        m0 = _integration_step_matrix(t.zb, "z", t.sp, 0)
        @test size(m0) == (0, 0)

        # nrows a multiple of Nz -> block kron of the weight row.
        wsingle = Matrix(_integration_step_matrix(t.zb, "z", t.sp, 8))
        mc = _integration_step_matrix(t.zb, "z", t.sp, 16)   # 2 components
        @test size(mc) == (2, 16)
        full = Matrix(mc)
        @test full[1:1, 1:8] ≈ wsingle
        @test full[2:2, 9:16] ≈ wsingle
        @test all(iszero, full[1:1, 9:16])
        @test all(iszero, full[2:2, 1:8])

        # nrows not a multiple of Nz -> nothing.
        @test _integration_step_matrix(t.zb, "z", t.sp, 7) === nothing
    end
end
