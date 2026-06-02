"""
Test suite for tau-method Lift and basis Convert operators.

Target module: src/core/operators/operations/operations_lift_convert.jl

Functions exercised:
  evaluate_convert, evaluate_lift, apply_lift_nd!,
  _multiply_lift_polynomial!, _set_lift_coefficient!,
  _get_lift_output_bases, _find_basis_axis

Oracle strategy (all independent of the function-under-test's own output):

  Convert  — A Convert preserves the represented FUNCTION while changing the
             spectral basis. Given a coefficient vector c_in in basis B_in,
             Convert produces c_out in B_out. We reconstruct the function from
             BOTH coefficient sets at arbitrary physical points using the
             standard polynomial definitions (`evaluate_basis`, which returns
             un-normalized T_n / U_n / Jacobi values) and assert pointwise
             equality. This oracle does not depend on the transform's
             normalization convention; it is pure math:
                 sum_n c_in[n] B_in_n(x)  ==  sum_n c_out[n] B_out_n(x).

  Lift     — Lift places a lower-D coefficient set into mode `lift_mode` of a
             higher polynomial basis (tau rows for BC enforcement). We assert
             on the coefficient-space placement directly: the operand value
             lands at the resolved 1-indexed mode and all other modes are zero.
             Resolved mode is computed independently from the documented
             convention (n<0 wraps, +1 for 1-indexing).
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: evaluate_convert, evaluate_lift, apply_lift_nd!,
               _multiply_lift_polynomial!, _set_lift_coefficient!,
               _get_lift_output_bases, _find_basis_axis,
               Convert, Lift, evaluate_basis, get_coeff_data, get_grid_data,
               ensure_layout!, conversion_matrix

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

"""Build a 1D field on `basis` whose ChebyshevT/U coefficients are `c` (length N).
Coefficients are written directly into coeff data (layout :c)."""
function field_with_coeffs(dist, name, basis, c::Vector{Float64})
    f = ScalarField(dist, name, (basis,), Float64)
    ensure_layout!(f, :c)
    cd = get_coeff_data(f)
    @assert length(cd) == length(c) "coeff length mismatch: $(length(cd)) vs $(length(c))"
    cd .= c
    return f
end

"""Reconstruct sum_n c[n] * basis_n(x) at physical points `xs` using the
standard (un-normalized) polynomial values returned by evaluate_basis."""
function reconstruct(basis, c::Vector{Float64}, xs::Vector{Float64})
    N = length(c)
    M = evaluate_basis(basis, xs, 0:(N-1))   # n_points x N
    return M * c
end

# ============================================================================
# Convert: function-preservation oracle
# ============================================================================

@testset "Convert (basis conversion)" begin

    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; mesh=(1,), dtype=Float64)

    @testset "ChebyshevT -> ChebyshevU preserves function" begin
        N  = 8
        zt = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        zu = ChebyshevU(coords["z"]; size=N, bounds=(-1.0, 1.0))

        # Arbitrary coefficient vector in T basis (a generic polynomial).
        c_T = [0.5, -1.2, 0.7, 0.3, -0.4, 0.1, 0.0, -0.2]

        f   = field_with_coeffs(dist, "u", zt, c_T)
        res = evaluate_convert(Convert(f, zu), :c)   # stays in coeff space

        @test res.bases[1] === zu
        c_U = vec(Array(get_coeff_data(res)))
        @test length(c_U) == N

        # Independent oracle: reconstruct both at arbitrary interior points.
        xs = collect(range(-0.93, 0.91; length=11))
        vT = reconstruct(zt, c_T, xs)
        vU = reconstruct(zu, c_U, xs)
        @test isapprox(vT, vU; rtol=1e-10, atol=1e-12)
    end

    @testset "Conversion is consistent with conversion_matrix" begin
        # Cross-check evaluate_convert against the raw conversion matrix
        # (matrix correctness itself is covered by test_chebyshev.jl).
        N  = 6
        zt = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        zu = ChebyshevU(coords["z"]; size=N, bounds=(-1.0, 1.0))

        c_T = [1.0, 0.5, -0.5, 0.25, 0.1, -0.1]
        f   = field_with_coeffs(dist, "u", zt, c_T)
        res = evaluate_convert(Convert(f, zu), :c)
        c_U = vec(Array(get_coeff_data(res)))

        C = conversion_matrix(zt, zu)
        @test isapprox(c_U, Matrix(C) * c_T; rtol=1e-12, atol=1e-14)
    end

    @testset "ChebyshevT -> ChebyshevV (two-step shift) preserves function" begin
        # FIXED. conversion_matrix now builds the general (non-identity, non-T->U)
        # connection matrix by collocation from the actual basis functions
        # (evaluate_basis), so it is correct regardless of the Chebyshev-vs-Jacobi
        # normalization. Previously the parameter-only recurrence path applied a
        # P^{(a,b)} conversion to Chebyshev-normalized coefficients and the function
        # was not preserved (err ~10).
        N  = 8
        zt = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        zv = ChebyshevV(coords["z"]; size=N, bounds=(-1.0, 1.0))

        c_T = [0.3, 0.6, -0.2, 0.4, 0.15, -0.05, 0.02, 0.0]
        f   = field_with_coeffs(dist, "u", zt, c_T)
        res = evaluate_convert(Convert(f, zv), :c)
        c_V = vec(Array(get_coeff_data(res)))

        xs = collect(range(-0.9, 0.9; length=9))
        vT = reconstruct(zt, c_T, xs)
        vV = reconstruct(zv, c_V, xs)
        @test isapprox(vT, vV; rtol=1e-9, atol=1e-11)
    end

    @testset "Pure-Jacobi positive integer shifts preserve function" begin
        # Regression guard for the a-shift-up sign fix (DLMF 18.9.5 subdiagonal is
        # negative). These raw P^{(a,b)} conversions previously failed with err ~6.
        N  = 8
        xs = collect(range(-0.85, 0.85; length=7))
        c  = [0.4, -0.3, 0.5, 0.2, -0.1, 0.15, 0.05, -0.2]
        for (a0, b0, a1, b1) in ((0.5, 0.5, 1.5, 0.5),    # pure a-shift
                                 (-0.5, -0.5, 1.5, 1.5),  # a and b shift (T->V params)
                                 (0.0, 0.0, 2.0, 1.0))    # asymmetric multi-step
            bin  = Tarang.Jacobi(coords["z"]; a = a0, b = b0, size = N, bounds = (-1.0, 1.0))
            bout = Tarang.Jacobi(coords["z"]; a = a1, b = b1, size = N, bounds = (-1.0, 1.0))
            res  = evaluate_convert(Convert(field_with_coeffs(dist, "u", bin, c), bout), :c)
            cout = vec(Array(get_coeff_data(res)))
            @test isapprox(reconstruct(bin, c, xs), reconstruct(bout, cout, xs);
                           rtol = 1e-9, atol = 1e-11)
        end
    end

    @testset "Negative integer shifts (down-shift) preserve function" begin
        # FIXED by the collocation conversion (the old _jacobi_*_shift_down_matrix
        # recurrence produced a wrong connection matrix, err ~4-20). Covers both a
        # single down-shift and a two-parameter down-shift.
        N   = 8
        c   = [0.4, -0.3, 0.5, 0.2, -0.1, 0.15, 0.05, -0.2]
        xs  = collect(range(-0.85, 0.85; length=7))
        for (a0, b0, a1, b1) in ((1.5, 0.5, 0.5, 0.5),     # single a down-shift
                                 (1.5, 1.5, -0.5, -0.5))   # both params down
            bin  = Tarang.Jacobi(coords["z"]; a = a0, b = b0, size = N, bounds = (-1.0, 1.0))
            bout = Tarang.Jacobi(coords["z"]; a = a1, b = b1, size = N, bounds = (-1.0, 1.0))
            res  = evaluate_convert(Convert(field_with_coeffs(dist, "u", bin, c), bout), :c)
            cout = vec(Array(get_coeff_data(res)))
            @test isapprox(reconstruct(bin, c, xs), reconstruct(bout, cout, xs);
                           rtol = 1e-9, atol = 1e-11)
        end
    end

    @testset "Identity conversion (same basis) preserves coefficients" begin
        N   = 6
        zt  = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        zt2 = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))  # same params

        c_T = [0.2, -0.3, 0.4, 0.1, 0.0, -0.15]
        f   = field_with_coeffs(dist, "u", zt, c_T)
        res = evaluate_convert(Convert(f, zt2), :c)
        @test isapprox(vec(Array(get_coeff_data(res))), c_T; rtol=1e-12, atol=1e-14)
    end

    @testset "Convert with :g layout preserves the function (c-space and grid)" begin
        # The Convert (in :c) preserves the function — asserted directly. The :g
        # path additionally backward-transforms the U-target field. This used to
        # be BROKEN for ChebyshevU (a no-op: U_0 mapped to grid [1,0,0,...]
        # instead of all-ones) because the transform planner had no branch for
        # ChebyshevU/Jacobi bases and the in-place hot path had no matrix-transform
        # method. Fixed via setup_jacobi_transform! + JacobiTransform in-place
        # methods; the :g grid values now match the function.
        N  = 12
        zt = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        zu = ChebyshevU(coords["z"]; size=N, bounds=(-1.0, 1.0))

        # Smooth function, exactly representable: a low-order polynomial.
        c_T = zeros(N)
        c_T[1] = 0.5; c_T[2] = -0.4; c_T[3] = 0.3; c_T[4] = 0.2

        # (a) c-space conversion preserves the function (PASS — module's job).
        fc   = field_with_coeffs(dist, "u", zt, c_T)
        resc = evaluate_convert(Convert(fc, zu), :c)
        @test resc.bases[1] === zu
        c_U  = vec(Array(get_coeff_data(resc)))
        xs   = collect(range(-0.9, 0.9; length=11))
        @test isapprox(reconstruct(zt, c_T, xs), reconstruct(zu, c_U, xs);
                       rtol=1e-10, atol=1e-12)

        # (b) :g path yields grid values equal to the function at the U
        #     collocation points (ChebyshevU backward transform now correct).
        fg    = field_with_coeffs(dist, "u", zt, c_T)
        resg  = evaluate_convert(Convert(fg, zu), :g)
        zgrid = vec(Array(Tarang.local_grid(zu, dist, 1)))
        gvals = vec(Array(get_grid_data(resg)))
        expected = reconstruct(zt, c_T, zgrid)
        @test isapprox(gvals, expected; rtol=1e-8, atol=1e-9)
    end

    @testset "Non-applicable convert returns a copy" begin
        # When the operand has no Jacobi basis on the target coordinate, the
        # operator falls through to `copy(operand)`.
        fcoords = CartesianCoordinates("x")
        fdist   = Distributor(fcoords; mesh=(1,), dtype=Float64)
        xb      = RealFourier(fcoords["x"]; size=8, bounds=(0.0, 2π))
        zb      = ChebyshevU(coords["z"]; size=8, bounds=(-1.0, 1.0))

        f = ScalarField(fdist, "u", (xb,), Float64)
        ensure_layout!(f, :c)
        get_coeff_data(f) .= 0
        # converting a Fourier-only field toward a Chebyshev basis: no match
        res = evaluate_convert(Convert(f, zb), :c)
        @test res.bases[1] === xb   # unchanged basis (copy of operand)
    end
end

# ============================================================================
# Lift: coefficient-placement oracle
# ============================================================================

@testset "Lift (tau placement)" begin

    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; mesh=(1,), dtype=Float64)

    """Resolve the documented lift convention independently: n<0 wraps from end,
    then convert 0-indexed -> 1-indexed."""
    resolve_mode(n, N) = (n < 0 ? N + n : n) + 1

    @testset "_get_lift_output_bases: substitutes matching basis on same coord" begin
        # Operand already lives on the lift coordinate ("z"): the output basis
        # replaces it in-place (found == true branch), preserving arity.
        zt  = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
        zu  = ChebyshevU(coords["z"]; size=8, bounds=(-1.0, 1.0))
        op  = ScalarField(dist, "tau", (zt,), Float64)
        out = _get_lift_output_bases(op, zu)
        @test length(out) == 1
        @test out[1] === zu
    end

    @testset "_get_lift_output_bases: appends basis when operand lacks coord" begin
        # Operand on coordinate "x" only; lifting into a "z" basis the operand
        # does not have triggers the append branch -> arity grows by one.
        xcoords = CartesianCoordinates("x")
        xdist   = Distributor(xcoords; mesh=(1,), dtype=Float64)
        xb      = RealFourier(xcoords["x"]; size=8, bounds=(0.0, 2π))
        zb      = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
        op      = ScalarField(xdist, "tau", (xb,), Float64)
        out     = _get_lift_output_bases(op, zb)
        @test length(out) == 2          # appended
        @test zb in out
        @test xb in out
        @test _find_basis_axis(out, zb) == 2   # zb appended at end
    end

    @testset "_find_basis_axis locates basis by element label" begin
        zb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
        @test _find_basis_axis((zb,), zb) == 1
        # default-to-1 when not found
        coords2 = CartesianCoordinates("q")
        qb = ChebyshevT(coords2["q"]; size=8, bounds=(-1.0, 1.0))
        @test _find_basis_axis((nothing,), qb) == 1
    end

    @testset "_set_lift_coefficient! sets one mode along an axis" begin
        data = zeros(4, 3)
        _set_lift_coefficient!(data, 1, 2, 7.0)   # axis=1, mode=2
        @test all(data[2, :] .== 7.0)
        data[2, :] .= 0.0
        @test all(data .== 0.0)
    end

    @testset "apply_lift_nd! copies operand mode-1 slice to target mode" begin
        result  = zeros(5, 2)
        operand = zeros(5, 2)
        operand[1, :] .= [3.0, -2.0]
        apply_lift_nd!(result, operand, 1, 4)   # place at mode 4 along axis 1
        @test result[4, :] == [3.0, -2.0]
        result[4, :] .= 0.0
        @test all(result .== 0.0)
    end

    @testset "_multiply_lift_polynomial! 1D places operand[1] at lift_mode" begin
        N        = 8
        result   = zeros(N)
        P_data   = zeros(N)                      # ignored on CPU path
        operand  = zeros(N); operand[1] = 4.2    # operand "tau value"
        lift_mode = 6
        _multiply_lift_polynomial!(result, P_data, operand, 1, lift_mode, nothing)
        @test result[lift_mode] == 4.2
        @test all(result[setdiff(1:N, lift_mode)] .== 0.0)
    end

    @testset "evaluate_lift: 1D constant tau lands at resolved mode (n=-1)" begin
        # Build a 1D operand on a Chebyshev basis with a known mode-1 (constant)
        # coefficient. The Lift takes operand_data[1] and places it at lift_mode.
        N   = 8
        zb  = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        op  = field_with_coeffs(dist, "tau", zb,
                                Float64[2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        res  = evaluate_lift(Lift(op, zb, -1), :c)
        c    = vec(Array(get_coeff_data(res)))
        mode = resolve_mode(-1, N)   # -> N
        @test mode == N
        @test c[mode] == 2.5
        @test all(c[setdiff(1:N, mode)] .== 0.0)
    end

    @testset "evaluate_lift: n=-2 lands at second-to-last mode" begin
        N   = 8
        zb  = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        op  = field_with_coeffs(dist, "tau", zb,
                                Float64[-1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        res  = evaluate_lift(Lift(op, zb, -2), :c)
        c    = vec(Array(get_coeff_data(res)))
        mode = resolve_mode(-2, N)   # -> N-1
        @test mode == N - 1
        @test c[mode] == -1.75
        @test all(c[setdiff(1:N, mode)] .== 0.0)
    end

    @testset "evaluate_lift: non-negative n indexes mode directly" begin
        N   = 8
        zb  = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        op  = field_with_coeffs(dist, "tau", zb,
                                Float64[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        res  = evaluate_lift(Lift(op, zb, 0), :c)
        c    = vec(Array(get_coeff_data(res)))
        mode = resolve_mode(0, N)    # -> 1
        @test mode == 1
        @test c[mode] == 3.0
        @test all(c[setdiff(1:N, mode)] .== 0.0)
    end

    @testset "evaluate_lift: output basis is the lift basis" begin
        N   = 6
        zb  = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        op  = field_with_coeffs(dist, "tau", zb,
                                Float64[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = evaluate_lift(Lift(op, zb, -1), :c)
        @test res.bases[1] === zb
        @test length(get_coeff_data(res)) == N
    end

    @testset "evaluate_lift: out-of-range mode throws" begin
        N  = 8
        zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        op = field_with_coeffs(dist, "tau", zb,
                               Float64[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # n = N (0-indexed) resolves to mode N+1 -> out of bounds
        @test_throws ArgumentError evaluate_lift(Lift(op, zb, N), :c)
    end

    @testset "evaluate_lift: non-scalar operand throws" begin
        N  = 8
        zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        f  = ScalarField(dist, "u", (zb,), Float64)
        vf = VectorField(dist, "v", (zb,), Float64)
        @test_throws ArgumentError evaluate_lift(Lift(vf, zb, -1), :c)
    end
end
