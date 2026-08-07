"""
Test suite for src/core/operators/symbolic_diff.jl

Covers the UNTESTED branches of sym_diff and its helpers. The BASIC cases
(sym_diff base/add/sub/multiply/sin/exp) are already covered in
test_dedalus_features.jl; this file targets the remaining branches:

  - Product rule on f*f (-> 2f), quotient rule (DivideOperator), Negate,
    Power rule (PowerOperator), constant-folding paths.
  - Chain rule for more UFUNCs (cos, tan, tanh, log, sqrt, sinh, cosh,
    atan, abs) beyond sin/exp; GeneralFunction chain rule; error path for
    an unregistered function.
  - Differential-operator guards: two-argument sym_diff rejects operator-valued
    derivatives and directs callers to perturbation-aware Frechet differentiation.
  - Directional Frechet rules for Differentiate, Laplacian, and
    FractionalLaplacian, plus the steady-NLBVP TimeDerivative convention.
  - The _simplify_* helpers unit-tested directly.
  - frechet_differential / build_symbolic_jacobian on simple residuals
    with analytically-known derivatives.

INDEPENDENT ORACLE: every expected value comes from calculus, and the
sym_diff result is *evaluated numerically* and compared to the analytic
derivative array (never compared against sym_diff run on itself).
"""

using Test
using Tarang
using LinearAlgebra
using SparseArrays

sd_real_dofs(v::AbstractVector{<:Number}) = vcat(real.(v), imag.(v))

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# 1D RealFourier field; returns (field, grid x, basis, dist, coords).
function sd_fourier_field(; N=64, L=2π, name="u")
    coords = CartesianCoordinates("x")
    dist   = Distributor(coords; mesh=(1,), dtype=Float64)
    xb     = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
    field  = ScalarField(dist, name, (xb,), Float64)
    mesh   = Tarang.create_meshgrid(field.domain)
    return field, mesh["x"], xb, dist, coords
end

# 2D RealFourier field for Gradient/Divergence/Laplacian passthrough tests.
function sd_fourier_field_2d(; N=16, name="u")
    coords = CartesianCoordinates("x", "y")
    dist   = Distributor(coords; mesh=(1, 1), dtype=Float64)
    xb     = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb     = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    field  = ScalarField(dist, name, (xb, yb), Float64)
    mesh   = Tarang.create_meshgrid(field.domain)
    return field, mesh["x"], mesh["y"], coords, dist
end

# A SQUARE problem for the Jacobian-assembly helpers: one variable, one
# equation, periodic domain. No boundary conditions means no tau variables,
# so n_eqs == n_vars, which is what `build_symbolic_jacobian` requires.
#
# This replaces five hand-written structs (`_SDProblem`, `_EqProb1..3`,
# `_SDProblem2Eq`) that stood in for a problem. They reached the assembler only
# through `_get_equation_data`'s `hasfield(..., :equations)` fallback — a branch
# no `Problem` subtype could take, kept alive by these very tests. That is the
# same arrangement that let a live bug sit in `_get_residual_expression` (it read
# "LHS"/"RHS", which no parser writes) until the test was rewritten to use the
# spellings the parser actually produces.
function sd_square_nlbvp(; N=16)
    fu, _, xb, dist, _ = sd_fourier_field(N=N, name="u")
    ensure_layout!(fu, :g); Tarang.get_grid_data(fu) .= 1.0
    g = ScalarField(dist, "g", (xb,), Float64)
    ensure_layout!(g, :g); Tarang.get_grid_data(g) .= 0.0

    prob = Tarang.NLBVP([fu])
    add_parameters!(prob; g=g)
    Tarang.add_equation!(prob, "lap(u) = u*u + g")
    Tarang.build_matrix_expressions!(prob)   # matrix-free; fills equation_data
    return prob, fu
end

# Reduce a sym_diff result to a grid-data array (or pass through a Number).
# A sym_diff result may already BE a ScalarField (e.g. d(u*g)/du -> g),
# in which case there is nothing to evaluate -- just read its grid data.
function eval_grid(d)
    if d isa Number
        return d
    elseif d isa ScalarField
        ensure_layout!(d, :g)
        return Tarang.get_grid_data(d)
    end
    ev = evaluate(d, :g)
    ensure_layout!(ev, :g)
    return Tarang.get_grid_data(ev)
end

# ============================================================================
@testset "symbolic_diff.jl coverage" begin
# ============================================================================

# -----------------------------------------------------------------------
@testset "Product rule (numeric oracle)" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    fg, _, _, _, _ = sd_fourier_field(name="g")
    Tarang.get_grid_data(fu) .= @. 0.7 + sin(x)          # arbitrary smooth u
    gvals = @. 1.3 + cos(2x)
    Tarang.get_grid_data(fg) .= gvals

    # d(u*g)/du = g  (g independent of u)
    d = sym_diff(Tarang.MultiplyOperator(fu, fg), fu)
    @test isapprox(eval_grid(d), gvals; rtol=1e-10)

    # d(u*u)/du = 2u
    fu2, x2, _, _, _ = sd_fourier_field(name="w")
    uvals = @. 0.9 + sin(3x2)
    Tarang.get_grid_data(fu2) .= uvals
    d2 = sym_diff(Tarang.MultiplyOperator(fu2, fu2), fu2)
    @test isapprox(eval_grid(d2), 2 .* uvals; rtol=1e-10)
end

# -----------------------------------------------------------------------
@testset "Quotient rule (DivideOperator)" begin
    # g independent of u -> short-circuit branch (dg==0): d(u/g)/du = df/g = 1/g.
    # NOTE: the evaluator does not support Number/Field or Field/Field division,
    # so we verify the symbolic STRUCTURE (DivideOperator(1, g)) and compute the
    # numeric value pointwise from grid data (independent calculus oracle: 1/g).
    fu, x, _, _, _ = sd_fourier_field(name="u")
    fg, _, _, _, _ = sd_fourier_field(name="g")
    Tarang.get_grid_data(fu) .= @. 0.5 + sin(x)
    gvals = @. 2.0 + 0.3 * cos(x)        # strictly positive => no /0
    Tarang.get_grid_data(fg) .= gvals

    d = sym_diff(Tarang.DivideOperator(fu, fg), fu)
    @test isa(d, Tarang.DivideOperator)
    @test d.left == 1                    # df = 1
    @test d.right === fg                 # denominator is g
    # Independent oracle: the represented value is 1/g.
    ensure_layout!(d.right, :g)
    @test isapprox(1.0 ./ Tarang.get_grid_data(d.right), 1.0 ./ gvals; rtol=1e-12)

    # General quotient branch (dg != 0): d(u/u)/du = (1*u - u*1)/u^2.
    # df=1, dg=1 -> num = _simplify_sub(g, f) = SubtractOperator(u,u),
    # den = _simplify_mul(g,g). Verify the full quotient structure is built.
    fu2, x2, _, _, _ = sd_fourier_field(name="w")
    uvals = @. 1.5 + 0.4 * sin(2x2)      # strictly positive
    Tarang.get_grid_data(fu2) .= uvals
    dq = sym_diff(Tarang.DivideOperator(fu2, fu2), fu2)
    @test isa(dq, Tarang.DivideOperator)
    @test isa(dq.left, Tarang.SubtractOperator)    # (df*g - f*dg) = (u - u)
    @test isa(dq.right, Tarang.MultiplyOperator)    # g^2 = u*u
    # Independent oracle: numerator (u-u) evaluates to 0 everywhere.
    num_vals = eval_grid(dq.left)
    @test isapprox(num_vals, zeros(length(uvals)); atol=1e-9)
end

# -----------------------------------------------------------------------
@testset "Negate rule" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    Tarang.get_grid_data(fu) .= @. 1.0 + sin(x)

    # d(-u)/du = -1   (operand reduces to 1 -> _simplify_neg(1) = -1)
    d = sym_diff(Tarang.NegateOperator(fu), fu)
    @test d == -1

    # d(-(u*g))/du = -g  (operand reduces to field g -> NegateOperator(g))
    fg, _, _, _, _ = sd_fourier_field(name="g")
    gvals = @. 2.0 + cos(x)
    Tarang.get_grid_data(fg) .= gvals
    d2 = sym_diff(Tarang.NegateOperator(Tarang.MultiplyOperator(fu, fg)), fu)
    @test isa(d2, Tarang.NegateOperator)
    @test isapprox(eval_grid(d2), .-gvals; rtol=1e-10)
end

# -----------------------------------------------------------------------
@testset "Power rule (PowerOperator)" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    uvals = @. 1.2 + 0.5 * sin(x)        # strictly positive
    Tarang.get_grid_data(fu) .= uvals

    # d(u^3)/du = 3 u^2
    d3 = sym_diff(Tarang.PowerOperator(fu, 3), fu)
    @test isapprox(eval_grid(d3), 3 .* uvals .^ 2; rtol=1e-10)

    # d(u^2)/du = 2 u
    d2 = sym_diff(Tarang.PowerOperator(fu, 2), fu)
    @test isapprox(eval_grid(d2), 2 .* uvals; rtol=1e-10)

    # Variable exponent (u^u) must error: dn != 0 branch.
    @test_throws ErrorException sym_diff(Tarang.PowerOperator(fu, fu), fu)
end

# -----------------------------------------------------------------------
@testset "Chain rule for UFUNCs (numeric oracle)" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    uvals = @. 0.6 + 0.3 * sin(x)        # in (0.3, 0.9): safe for all funcs
    Tarang.get_grid_data(fu) .= uvals

    # Each: d(F(u))/du = F'(u), evaluated and compared to analytic F'(uvals).
    cases = [
        (cos,  @. -sin(uvals)),
        (tan,  @. 1 / cos(uvals)^2),
        (tanh, @. 1 - tanh(uvals)^2),
        (sinh, @. cosh(uvals)),
        (cosh, @. sinh(uvals)),
        (atan, @. 1 / (1 + uvals^2)),
        (log,  @. 1 / uvals),
        (sqrt, @. 1 / (2 * sqrt(uvals))),
        (asin, @. 1 / sqrt(1 - uvals^2)),
        (acos, @. -1 / sqrt(1 - uvals^2)),
    ]
    for (fn, analytic) in cases
        op = Tarang.UnaryGridFunction(fu, fn, string(fn))
        d  = sym_diff(op, fu)
        @test isapprox(eval_grid(d), analytic; rtol=1e-10)
    end

    # abs -> sign. Use a strictly-positive field so sign(u)=1 cleanly.
    op_abs = Tarang.UnaryGridFunction(fu, abs, "abs")
    d_abs  = sym_diff(op_abs, fu)
    @test isapprox(eval_grid(d_abs), sign.(uvals); rtol=1e-12)

    # Chain rule with non-trivial inner derivative:
    # d(sin(u*g))/du = cos(u*g) * d(u*g)/du = cos(u*g) * g.
    # The evaluator can't apply a UFUNC to a product expression, so verify the
    # symbolic STRUCTURE: outer derivative is the cos-UnaryGridFunction wrapping
    # the same inner product, multiplied by du = g.
    fg, _, _, _, _ = sd_fourier_field(name="g")
    Tarang.get_grid_data(fg) .= @. 1.0 + 0.0 * x   # ScalarField (indep of u)
    inner = Tarang.MultiplyOperator(fu, fg)
    op_sin = Tarang.UnaryGridFunction(inner, sin, "sin")
    d_sin  = sym_diff(op_sin, fu)
    @test isa(d_sin, Tarang.MultiplyOperator)
    @test isa(d_sin.left, Tarang.UnaryGridFunction)   # cos(u*g)
    @test d_sin.left.func === cos
    @test d_sin.left.operand === inner                # chain: same inner arg
    @test d_sin.right === fg                           # du = d(u*g)/du = g
end

# -----------------------------------------------------------------------
@testset "UFUNC: derivative-of-constant short-circuit (du==0)" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    fv, _, _, _, _ = sd_fourier_field(name="v")
    Tarang.get_grid_data(fu) .= @. 0.5 + sin(x)

    # d(sin(u))/dv = 0  -> hits the `du == 0` early return in UFUNC branch
    op = Tarang.UnaryGridFunction(fu, sin, "sin")
    @test sym_diff(op, fv) == 0
end

# -----------------------------------------------------------------------
@testset "GeneralFunction chain rule + error path" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    uvals = @. 0.7 + 0.2 * sin(x)
    Tarang.get_grid_data(fu) .= uvals

    # GeneralFunction with a REGISTERED func (exp) -> chain rule works.
    gf = Tarang.GeneralFunction(fu, exp, "exp")
    d  = sym_diff(gf, fu)
    @test isa(d, Tarang.GeneralFunction)
    @test isapprox(eval_grid(d), exp.(uvals); rtol=1e-10)

    # GeneralFunction whose func is NOT in UFUNC_DERIVATIVES -> error.
    myfun = (z) -> z + 1            # not registered
    gf_bad = Tarang.GeneralFunction(fu, myfun, "myfun")
    @test_throws ErrorException sym_diff(gf_bad, fu)

    # UnaryGridFunction with unregistered func -> error path.
    uf_bad = Tarang.UnaryGridFunction(fu, myfun, "myfun")
    @test_throws ErrorException sym_diff(uf_bad, fu)
end

# -----------------------------------------------------------------------
@testset "Differentiate: operator-valued derivative guard" begin
    fu, x, _, coords1d_dist, coords = sd_fourier_field(name="u")
    fg, _, _, _, _ = sd_fourier_field(name="g")
    Tarang.get_grid_data(fu) .= @. 0.4 + sin(x)
    gvals = @. 1.0 + cos(2x)
    Tarang.get_grid_data(fg) .= gvals
    cx = coords["x"]

    # A two-argument symbolic partial cannot represent the linear map
    # δu -> d/dx(g*δu). It must direct callers to frechet_differential instead
    # of returning d/dx(g), which silently drops the perturbation inside d/dx.
    expr = Tarang.Differentiate(Tarang.MultiplyOperator(fu, fg), cx, 1)
    @test_throws ArgumentError sym_diff(expr, fu)

    # The same guard applies to the linear map δu -> d/dx(δu).
    @test_throws ArgumentError sym_diff(Tarang.Differentiate(fu, cx, 1), fu)

    # Operand independent of var -> 0 (d_operand == 0 branch).
    fv, _, _, _, _ = sd_fourier_field(name="v")
    @test sym_diff(Tarang.Differentiate(fv, cx, 1), fu) == 0
end

# -----------------------------------------------------------------------
@testset "Laplacian: operator-valued derivative guard" begin
    fu, x, y, coords, _ = sd_fourier_field_2d(name="u")
    fg, _, _, _, _ = sd_fourier_field_2d(name="g")
    Tarang.get_grid_data(fu) .= @. 0.3 + sin(x) * cos(y)
    gvals = @. 1.0 + cos(x) * sin(2y)
    Tarang.get_grid_data(fg) .= gvals

    # The derivative is the map δu -> ∇²(g*δu), not the scalar expression ∇²g.
    expr = Tarang.Laplacian(Tarang.MultiplyOperator(fu, fg))
    @test_throws ArgumentError sym_diff(expr, fu)

    @test_throws ArgumentError sym_diff(Tarang.Laplacian(fu), fu)

    # Operand independent of var -> 0.
    fv, _, _, _, _ = sd_fourier_field_2d(name="v")
    @test sym_diff(Tarang.Laplacian(fv), fu) == 0
end

# -----------------------------------------------------------------------
@testset "FractionalLaplacian: operator-valued derivative guard" begin
    fu, x, y, coords, _ = sd_fourier_field_2d(name="u")
    fg, _, _, _, _ = sd_fourier_field_2d(name="g")
    Tarang.get_grid_data(fu) .= @. 0.3 + sin(x) * cos(y)
    gvals = @. 1.0 + cos(2x) * sin(y)
    Tarang.get_grid_data(fg) .= gvals

    α = 0.5
    # This is likewise an operator-valued derivative requiring a perturbation.
    expr = Tarang.FractionalLaplacian(Tarang.MultiplyOperator(fu, fg), α)
    @test_throws ArgumentError sym_diff(expr, fu)

    @test_throws ArgumentError sym_diff(Tarang.FractionalLaplacian(fu, α), fu)
    # Independent operand -> 0.
    fv, _, _, _, _ = sd_fourier_field_2d(name="v")
    @test sym_diff(Tarang.FractionalLaplacian(fv, α), fu) == 0
end

# -----------------------------------------------------------------------
@testset "Gradient / Divergence operator-valued derivative guards" begin
    fu, x, y, coords, _ = sd_fourier_field_2d(name="u")
    fg, _, _, _, _ = sd_fourier_field_2d(name="g")
    Tarang.get_grid_data(fu) .= @. 0.5 + sin(x) * cos(y)
    gvals = @. 1.0 + cos(x) * sin(y)
    Tarang.get_grid_data(fg) .= gvals

    # Both derivatives are linear maps on a perturbation, not plain expressions.
    gexpr = Tarang.Gradient(Tarang.MultiplyOperator(fu, fg), coords)
    @test_throws ArgumentError sym_diff(gexpr, fu)

    # Gradient of var-independent operand -> 0.
    fv, _, _, _, _ = sd_fourier_field_2d(name="v")
    @test sym_diff(Tarang.Gradient(fv, coords), fu) == 0

    div_expr = Tarang.Divergence(Tarang.Gradient(Tarang.MultiplyOperator(fu, fg), coords))
    @test_throws ArgumentError sym_diff(div_expr, fu)

    # Divergence of var-independent operand -> 0.
    @test sym_diff(Tarang.Divergence(Tarang.Gradient(fv, coords)), fu) == 0
end

# -----------------------------------------------------------------------
@testset "Copy passthrough" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    fg, _, _, _, _ = sd_fourier_field(name="g")
    Tarang.get_grid_data(fu) .= @. 0.5 + sin(x)
    gvals = @. 2.0 + cos(x)
    Tarang.get_grid_data(fg) .= gvals

    # d/du[ Copy(u*g) ] = d/du(u*g) = g  (Copy just forwards to operand).
    d = sym_diff(Tarang.Copy(Tarang.MultiplyOperator(fu, fg)), fu)
    @test isapprox(eval_grid(d), gvals; rtol=1e-10)

    # Copy of a bare var: d/du[ Copy(u) ] = 1.
    @test sym_diff(Tarang.Copy(fu), fu) == 1
end

# -----------------------------------------------------------------------
@testset "TimeDerivative convention" begin
    fu, _, _, _, _ = sd_fourier_field(name="u")
    # NLBVP steady-state convention: ∂(∂t u)/∂u = 0.
    @test sym_diff(Tarang.TimeDerivative(fu), fu) == 0
    @test sym_diff(Tarang.TimeDerivative(fu, 2), fu) == 0
end

# -----------------------------------------------------------------------
@testset "_simplify_* helpers (direct unit tests)" begin
    fu, x, _, _, _ = sd_fourier_field(name="u")
    fv, _, _, _, _ = sd_fourier_field(name="v")

    # _simplify_add
    @test Tarang._simplify_add(0, fu) === fu          # 0 + x -> x
    @test Tarang._simplify_add(fu, 0) === fu          # x + 0 -> x
    @test Tarang._simplify_add(2, 3) == 5             # number fold
    @test isa(Tarang._simplify_add(fu, fv), Tarang.AddOperator)

    # _simplify_sub
    @test Tarang._simplify_sub(fu, 0) === fu          # x - 0 -> x
    @test Tarang._simplify_sub(0, fu) isa Tarang.NegateOperator  # 0 - x -> -x
    @test Tarang._simplify_sub(7, 4) == 3             # number fold
    @test isa(Tarang._simplify_sub(fu, fv), Tarang.SubtractOperator)

    # _simplify_mul
    @test Tarang._simplify_mul(0, fu) == 0            # 0 * x -> 0
    @test Tarang._simplify_mul(fu, 0) == 0            # x * 0 -> 0
    @test Tarang._simplify_mul(1, fu) === fu          # 1 * x -> x
    @test Tarang._simplify_mul(fu, 1) === fu          # x * 1 -> x
    @test Tarang._simplify_mul(2, 3) == 6             # number fold
    @test isa(Tarang._simplify_mul(fu, fv), Tarang.MultiplyOperator)

    # _simplify_div
    @test Tarang._simplify_div(0, fu) == 0            # 0 / x -> 0
    @test Tarang._simplify_div(fu, 1) === fu          # x / 1 -> x
    @test Tarang._simplify_div(6, 3) == 2.0           # number fold (float)
    @test isa(Tarang._simplify_div(fu, fv), Tarang.DivideOperator)

    # _simplify_neg
    @test Tarang._simplify_neg(0) == 0                # -0 -> 0
    @test Tarang._simplify_neg(5) == -5               # -number
    neg = Tarang.NegateOperator(fu)
    @test Tarang._simplify_neg(neg) === fu            # -(-x) -> x
    @test isa(Tarang._simplify_neg(fu), Tarang.NegateOperator)
end

# -----------------------------------------------------------------------
@testset "simplify(expr) recursion" begin
    fu, _, _, _, _ = sd_fourier_field(name="u")
    fv, _, _, _, _ = sd_fourier_field(name="v")

    # Number / ScalarField passthrough.
    @test simplify(3.0) == 3.0
    @test simplify(fu) === fu

    # Nested: ((0 + u) * 1) -> u
    inner = Tarang.AddOperator(0, fu)
    outer = Tarang.MultiplyOperator(inner, 1)
    @test simplify(outer) === fu

    # Subtract recursion: (u - 0) -> u
    @test simplify(Tarang.SubtractOperator(fu, 0)) === fu

    # Divide recursion: (u / 1) -> u
    @test simplify(Tarang.DivideOperator(fu, 1)) === fu

    # Negate recursion: -(0) -> 0
    @test simplify(Tarang.NegateOperator(0)) == 0

    # Unhandled operator type passes through unchanged.
    coords = CartesianCoordinates("x")
    cx = coords["x"]
    diffop = Tarang.Differentiate(fu, cx, 1)
    @test simplify(diffop) === diffop
end

# -----------------------------------------------------------------------
@testset "frechet_differential" begin
    fu, x, _, _, coords = sd_fourier_field(name="u")
    fv, _, _, _, _ = sd_fourier_field(name="v")
    uvals = @. 0.8 + 0.3 * sin(x)
    Tarang.get_grid_data(fu) .= uvals
    gvals = @. 1.0 + 0.2 * cos(x)
    Tarang.get_grid_data(fv) .= gvals

    du = ScalarField(fu.dist, "du", fu.bases, fu.dtype)
    duvals = @. 0.1 + 0.05 * sin(2x)
    ensure_layout!(du, :g); Tarang.get_grid_data(du) .= duvals

    # F = u^2 ; dF(u0).du = 2 u0 * du. Evaluate and compare.
    F = Tarang.PowerOperator(fu, 2)
    dF = frechet_differential(F, [fu], [du])
    @test dF !== 0
    @test isapprox(eval_grid(dF), 2 .* uvals .* duvals; rtol=1e-9)

    # Multi-variable: F = u*v ; dF = v*du + u*dv.
    dv = ScalarField(fv.dist, "dv", fv.bases, fv.dtype)
    dvvals = @. 0.07 + 0.03 * cos(3x)
    ensure_layout!(dv, :g); Tarang.get_grid_data(dv) .= dvvals
    Fuv = Tarang.MultiplyOperator(fu, fv)
    dFuv = frechet_differential(Fuv, [fu, fv], [du, dv])
    expected = @. gvals * duvals + uvals * dvvals
    @test isapprox(eval_grid(dFuv), expected; rtol=1e-9)

    # Linear differential operators act on the perturbation; they are not zero.
    cx = coords["x"]
    d_diff = frechet_differential(Tarang.Differentiate(fu, cx, 1), [fu], [du])
    d_lap = frechet_differential(Tarang.Laplacian(fu), [fu], [du])
    α = 0.5
    d_frac = frechet_differential(Tarang.FractionalLaplacian(fu, α), [fu], [du])
    @test d_diff !== 0
    @test d_lap !== 0
    @test d_frac !== 0
    if d_diff !== 0 && d_lap !== 0 && d_frac !== 0
        @test isapprox(eval_grid(d_diff), eval_grid(Tarang.Differentiate(du, cx, 1)); rtol=1e-9)
        @test isapprox(eval_grid(d_lap), eval_grid(Tarang.Laplacian(du)); rtol=1e-9)
        @test isapprox(eval_grid(d_frac), eval_grid(Tarang.FractionalLaplacian(du, α)); rtol=1e-9)
    end

    # Differential operators must enclose the full directional product rule:
    # d[Δ(u²)]·du = Δ(u*du + du*u), not Δ(2u)*du.
    nested = Tarang.Laplacian(Tarang.MultiplyOperator(fu, fu))
    d_nested = frechet_differential(nested, [fu], [du])
    nested_ref = Tarang.Laplacian(Tarang.AddOperator(
        Tarang.MultiplyOperator(fu, du), Tarang.MultiplyOperator(du, fu)))
    @test isapprox(eval_grid(d_nested), eval_grid(nested_ref); rtol=1e-8, atol=1e-9)

    # Constant residual -> 0 (empty terms branch).
    @test frechet_differential(5.0, [fu], [du]) == 0

    # Length mismatch -> error.
    @test_throws ErrorException frechet_differential(F, [fu, fv], [du])
end

# -----------------------------------------------------------------------
@testset "build_symbolic_jacobian (u^2 residual)" begin
    # This used to run against `_SDProblem`, a hand-written struct with an
    # `equations::Vector{Any}` field, reaching the assembler through a fallback
    # in `_get_equation_data` that no real problem could take. It never managed
    # to produce a Jacobian either — the assertion was `@test_broken built`.
    # Driving a real NLBVP instead covers the production path and produces one.
    prob, fu = sd_square_nlbvp(N=16)
    state = [fu]
    ncoeff = length(Tarang.get_coeff_data(fu))

    J = build_symbolic_jacobian(prob, state)
    @test isa(J, SparseMatrixCSC)
    @test eltype(J) <: Real
    @test size(J) == (2ncoeff, 2ncoeff)
    # Residual is lap(u) - u*u - g at u=1. On Fourier mode k=1 its exact
    # Jacobian eigenvalue is -k² - 2u = -3.
    mode1 = zeros(ComplexF64, ncoeff)
    mode1[2] = 1
    @test isapprox((J * sd_real_dofs(mode1))[2], -3.0; rtol=1e-10, atol=1e-12)

    # A sine perturbation coupled across k=0 exercises the imaginary
    # RealFourier quadrature. The doubled-real Jacobian must match the complete
    # directional residual, not only cosine/real coefficient probes.
    mesh = Tarang.create_meshgrid(fu.domain)
    x = mesh["x"]
    ensure_layout!(fu, :g); Tarang.get_grid_data(fu) .= @. 1.0 + 0.25 * cos(3x)
    sine_pert = ScalarField(fu.dist, "sine_pert", fu.bases, fu.dtype)
    ensure_layout!(sine_pert, :g); Tarang.get_grid_data(sine_pert) .= @. sin(2x)
    sine_delta = Tarang.fields_to_vector([sine_pert])
    residual = Tarang._get_residual_expression(prob.equation_data[1])
    directional = frechet_differential(residual, [fu], [sine_pert])
    expected_field = Tarang.evaluate_solver_expression(
        directional, [sine_pert]; layout=:g, template=fu)
    expected_direction = Tarang.fields_to_vector([expected_field])
    varying_J = build_symbolic_jacobian(prob, state)
    @test isapprox(varying_J * sd_real_dofs(sine_delta),
                   sd_real_dofs(expected_direction); rtol=1e-10, atol=1e-12)
    correction = Tarang._solve_newton_correction(varying_J, expected_direction)
    @test isapprox(correction, -sine_delta; rtol=1e-9, atol=1e-10)

    # Guard: no equation data (IR never built) -> error.
    u_bare = ScalarField(fu.dist, "u_bare", fu.bases, Float64)
    @test_throws ErrorException build_symbolic_jacobian(Tarang.NLBVP([u_bare]), [u_bare])

    # Guard: non-square system (1 equation, 2 variables) -> error.
    @test_throws ErrorException build_symbolic_jacobian(prob, [fu, u_bare])

    # Guard: the assembler takes a Problem, not anything that quacks like one.
    @test_throws MethodError build_symbolic_jacobian(Dict("equations" => []), state)
end

# -----------------------------------------------------------------------
@testset "_evaluate_jacobian_block + _sparse_entries" begin
    fu, x, _, _, _ = sd_fourier_field(name="u", N=8)
    ensure_layout!(fu, :g); Tarang.get_grid_data(fu) .= 2.0
    ensure_layout!(fu, :c)
    ncoef = length(Tarang.get_coeff_data(fu))

    # Number branch: scalar -> scalar * I (diagonal of the constant).
    bnum = Tarang._evaluate_jacobian_block(3.0, fu, 4, 4)
    @test isa(bnum, SparseMatrixCSC)
    @test all(diag(Matrix(bnum)) .== 3.0)
    @test size(bnum) == (8, 8)

    # ScalarField branch: multiplication by a physical constant is that constant
    # on every valid RealFourier degree of freedom. The imaginary DC/Nyquist
    # slots are not physical RFFT degrees of freedom and correctly map to zero.
    bsf = Tarang._evaluate_jacobian_block(fu, fu, ncoef, ncoef)
    @test isa(bsf, SparseMatrixCSC)
    valid_delta = zeros(ComplexF64, ncoef)
    valid_delta[1] = 0.5
    valid_delta[2] = 1 + 2im
    valid_delta[end] = 0.25
    @test isapprox(bsf * sd_real_dofs(valid_delta),
                   2.0 .* sd_real_dofs(valid_delta); rtol=1e-12)

    # An operator-valued coefficient is evaluated and gets the same treatment.
    op = Tarang.MultiplyOperator(2.0, fu)
    bop = Tarang._evaluate_jacobian_block(op, fu, ncoef, ncoef)
    @test isa(bop, SparseMatrixCSC)
    @test isapprox(bop * sd_real_dofs(valid_delta),
                   4.0 .* sd_real_dofs(valid_delta); rtol=1e-10)

    # A varying coefficient couples Fourier modes. Check the full matrix action
    # against an independently evaluated physical-space product.
    ensure_layout!(fu, :g); Tarang.get_grid_data(fu) .= @. 1.0 + 0.25 * cos(x)
    pert = ScalarField(fu.dist, "pert", fu.bases, fu.dtype)
    ensure_layout!(pert, :g); Tarang.get_grid_data(pert) .= @. cos(2x)
    varying_block = Tarang._evaluate_jacobian_block(fu, pert, ncoef, ncoef)
    delta = Tarang.fields_to_vector([pert])
    expected_product = Tarang.fields_to_vector([fu * pert])
    @test isapprox(varying_block * sd_real_dofs(delta),
                   sd_real_dofs(expected_product); rtol=1e-10, atol=1e-12)

    # A RealFourier half-spectrum is only real-linear: multiplication by a
    # varying real field couples each positive mode to its implicit conjugate.
    # Crossing through k=0 flips the sine contribution, which a matrix assembled
    # from cosine-only coefficient probes cannot reproduce.
    q, qx, _, _, _ = sd_fourier_field(name="q", N=16)
    ensure_layout!(q, :g); Tarang.get_grid_data(q) .= @. 1.0 + 0.25 * cos(3qx)
    sine_pert = ScalarField(q.dist, "sine_pert", q.bases, q.dtype)
    ensure_layout!(sine_pert, :g); Tarang.get_grid_data(sine_pert) .= @. sin(2qx)
    sn = length(Tarang.fields_to_vector([sine_pert]))
    sine_block = Tarang._evaluate_jacobian_block(q, sine_pert, sn, sn)
    sine_delta = Tarang.fields_to_vector([sine_pert])
    sine_expected = Tarang.fields_to_vector([q * sine_pert])
    @test isapprox(sine_block * sd_real_dofs(sine_delta),
                   sd_real_dofs(sine_expected); rtol=1e-10, atol=1e-12)

    # Complex coefficient values must survive assembly.
    ccoords = CartesianCoordinates("x")
    cdist = Distributor(ccoords; mesh=(1,), dtype=ComplexF64)
    cbasis = ComplexFourier(ccoords["x"]; size=8, bounds=(0.0, 2π))
    ccoef = ScalarField(cdist, "q", (cbasis,), ComplexF64)
    cvar = ScalarField(cdist, "δu", (cbasis,), ComplexF64)
    set!(ccoef, (x,) -> 1 + 2im)
    cn = length(Tarang.get_coeff_data(ccoef))
    cblock = Tarang._evaluate_jacobian_block(ccoef, cvar, cn, cn)
    cI = Matrix{Float64}(I, cn, cn)
    expected_cblock = [cI -2cI; 2cI cI]
    @test isapprox(Matrix(cblock), expected_cblock; rtol=1e-12)

    # _sparse_entries: SparseMatrixCSC path.
    Sp = spdiagm(0 => [1.0, 2.0, 3.0])
    ents = collect(Tarang._sparse_entries(Sp))
    @test (1, 1, 1.0) in ents && (2, 2, 2.0) in ents && (3, 3, 3.0) in ents

    # _sparse_entries: dense AbstractMatrix path (skips zeros).
    M = [1.0 0.0; 0.0 2.0]
    dents = collect(Tarang._sparse_entries(M))
    @test sort(dents) == sort([(1, 1, 1.0), (2, 2, 2.0)])

    # _sparse_entries: Number path.
    @test collect(Tarang._sparse_entries(5.0)) == [(1, 1, 5.0)]

    # _sparse_entries: unrecognized input -> empty fallback.
    @test isempty(collect(Tarang._sparse_entries("not a matrix")))
end

# -----------------------------------------------------------------------
@testset "Jacobian helpers: _get_equation_data / _get_residual_expression" begin
    fu, x, _, _, _ = sd_fourier_field(name="u", N=8)

    # _get_residual_expression with lhs/rhs -> SubtractOperator(lhs, rhs).
    #
    # These are the keys `build_matrix_expressions!` actually writes
    # (problem_matrices_build.jl). This assertion previously used "LHS"/"RHS",
    # which nothing in the package writes, so it passed while production input
    # never reached the subtraction branch at all — every Newton Jacobian was
    # built from `F` alone, dropping the implicit `L` term. The test was checking
    # the function against invented input rather than against the parser.
    res = Tarang._get_residual_expression(Dict("lhs" => fu, "rhs" => 0.0))
    @test isa(res, Tarang.SubtractOperator)
    @test res.left === fu

    # Canonical keys are case-sensitive and NOT aliased: uppercase must not be
    # mistaken for the real slots, or the same silent fallthrough returns.
    @test Tarang._get_residual_expression(Dict("LHS" => fu, "RHS" => 0.0)) == 0

    # _get_residual_expression with "F" key -> the residual directly.
    F = Tarang.PowerOperator(fu, 2)
    @test Tarang._get_residual_expression(Dict("F" => F)) === F

    # _get_residual_expression with unrecognized data -> 0.
    @test Tarang._get_residual_expression(42) == 0

    # _get_residual_expression with a Dict lacking LHS/RHS/F -> 0.
    @test Tarang._get_residual_expression(Dict("other" => 1)) == 0

    # _get_equation_data returns the problem's own IR, and takes ::Problem only.
    # It used to accept anything and fall back to `problem.equations`, a
    # Vector{String}, where every caller expects equation IR. No Problem subtype
    # can reach that branch; only the duck-typed stand-ins that used to live at
    # the top of this file could, which is why it survived.
    prob = sd_square_nlbvp()[1]
    @test Tarang._get_equation_data(prob) === prob.equation_data
    @test_throws MethodError Tarang._get_equation_data(Dict("F" => F))
    @test_throws MethodError Tarang._get_equation_data(42)
end

# ============================================================================
end  # top-level testset
# ============================================================================
