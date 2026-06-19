# Coverage tests for src/core/solvers/solver_compiled_rhs.jl
#
# Focus: evaluate_solver_expression — the interpreted RHS-expression evaluator
# that walks a parsed operator tree (Add/Subtract/Multiply/Divide/Power/Negate/
# Index/Constant/Zero/Array/Unknown) and produces a ScalarField (grid layout) or
# a numeric scalar. We construct operator trees directly with ScalarField leaves
# and assert the grid result equals the analytic value, plus the error paths.
#
# Also covers the small helpers: _constant_field_from_template,
# _coerce_numeric_operand, _binary_template, create_zero_field,
# create_constant_field, UnrecognizedRHSExpression, get_diff_order.

using Test, Tarang, LinearAlgebra

const T = Tarang

# Top-level helper structs for get_diff_order / get_diff_coordinate tests
# (struct definitions must be at top level, not inside a @testset).
struct _NoOrderExpr end
struct _WithOrderExpr; order::Int; end
struct _BareExpr end

# ---------------------------------------------------------------------------
# Helpers to build a simple serial Fourier field and seed grid data.
# RealFourier with a small size keeps us on the un-padded grid-multiply path
# (the spectral dealiased product only kicks in for prod(size) > 64).
#
# CRITICAL: field arithmetic (a + b, a * b) requires the operands to share the
# SAME basis object — two separately-constructed RealFourier bases are NOT == .
# So we build ONE shared Distributor + basis and allocate all fields from them.
# ---------------------------------------------------------------------------
const _COORDS = T.CartesianCoordinates("x")
const _DIST = T.Distributor(_COORDS; mesh=(1,), dtype=Float64)
const _XB = T.RealFourier(_COORDS["x"]; size=8, bounds=(0.0, 2pi))

function make_field(name; dtype=Float64)
    f = T.ScalarField(_DIST, name, (_XB,), dtype)
    T.ensure_layout!(f, :g)
    return f
end

# Seed a field's grid data from a function of its grid coordinate.
function seed!(f, fn)
    T.ensure_layout!(f, :g)
    g = T.get_grid_data(f)
    mesh = T.create_meshgrid(f.domain)
    x = first(values(mesh))
    g .= fn.(x)
    return f
end

gd(f) = (T.ensure_layout!(f, :g); collect(T.get_grid_data(f)))

# Dense n×n identity for comparing against sparse Jacobian blocks
# (avoids depending on SparseArrays directly; LinearAlgebra's I suffices).
ident(n) = Matrix{ComplexF64}(I, n, n)
dense(m) = Matrix(m)

@testset "solver_compiled_rhs coverage" begin

    @testset "leaf: ScalarField passthrough + layout" begin
        f = make_field("u")
        seed!(f, x -> sin(x))
        out = T.evaluate_solver_expression(f, Dict{String,Any}(); layout=:g)
        @test out === f                       # ScalarField returned as-is
        @test gd(out) ≈ sin.(first(values(T.create_meshgrid(f.domain))))
    end

    @testset "leaf: Number with and without template" begin
        # No template -> raw number returned
        @test T.evaluate_solver_expression(3.5, Dict{String,Any}(); template=nothing) == 3.5
        # With template -> constant field filled with the value
        tmpl = make_field("t")
        out = T.evaluate_solver_expression(2.0, Dict{String,Any}(); layout=:g, template=tmpl)
        @test out isa T.ScalarField
        @test all(gd(out) .≈ 2.0)
    end

    @testset "nothing expression errors" begin
        @test_throws ArgumentError T.evaluate_solver_expression(nothing, Dict{String,Any}())
    end

    @testset "ZeroOperator" begin
        # No template -> integer 0
        @test T.evaluate_solver_expression(T.ZeroOperator(), Dict{String,Any}(); template=nothing) == 0
        # With template -> zero field
        tmpl = make_field("z")
        out = T.evaluate_solver_expression(T.ZeroOperator(), Dict{String,Any}(); template=tmpl)
        @test out isa T.ScalarField
        @test all(gd(out) .≈ 0.0)
    end

    @testset "ConstantOperator" begin
        c = T.ConstantOperator(4.0)
        # No template -> the scalar value
        @test T.evaluate_solver_expression(c, Dict{String,Any}(); template=nothing) == 4.0
        # With template -> constant field
        tmpl = make_field("c")
        out = T.evaluate_solver_expression(c, Dict{String,Any}(); layout=:g, template=tmpl)
        @test all(gd(out) .≈ 4.0)
    end

    @testset "ArrayOperator" begin
        tmpl = make_field("a")
        n = length(T.get_grid_data(tmpl))
        arr = collect(1.0:n)
        ao = T.ArrayOperator(arr)
        # No template -> the raw array
        @test T.evaluate_solver_expression(ao, Dict{String,Any}(); template=nothing) === arr
        # With template -> field whose grid data is copied from the array
        out = T.evaluate_solver_expression(ao, Dict{String,Any}(); layout=:g, template=tmpl)
        @test out isa T.ScalarField
        @test vec(gd(out)) ≈ arr
    end

    @testset "UnknownOperator throws UnrecognizedRHSExpression" begin
        uo = T.UnknownOperator("dx(u)")
        @test_throws T.UnrecognizedRHSExpression T.evaluate_solver_expression(uo, Dict{String,Any}())
        # showerror message mentions the offending expression
        e = T.UnrecognizedRHSExpression("dx(u)")
        buf = IOBuffer()
        showerror(buf, e)
        msg = String(take!(buf))
        @test occursin("dx(u)", msg)
        @test occursin("UnrecognizedRHSExpression", msg)
    end

    @testset "AddOperator: field + field" begin
        a = seed!(make_field("a"), x -> sin(x))
        b = seed!(make_field("b"), x -> cos(x))
        xg = first(values(T.create_meshgrid(a.domain)))
        out = T.evaluate_solver_expression(T.AddOperator(a, b), Dict{String,Any}(); layout=:g)
        @test out isa T.ScalarField
        @test gd(out) ≈ sin.(xg) .+ cos.(xg)
    end

    @testset "AddOperator: number + number" begin
        out = T.evaluate_solver_expression(T.AddOperator(2.0, 3.0), Dict{String,Any}(); template=nothing)
        @test out == 5.0
    end

    @testset "AddOperator: field + number (numeric coerced to const field)" begin
        a = seed!(make_field("a"), x -> sin(x))
        xg = first(values(T.create_meshgrid(a.domain)))
        out = T.evaluate_solver_expression(T.AddOperator(a, 1.5), Dict{String,Any}(); layout=:g)
        @test gd(out) ≈ sin.(xg) .+ 1.5
    end

    @testset "SubtractOperator: field - field, number - number" begin
        a = seed!(make_field("a"), x -> 2 .+ sin(x))
        b = seed!(make_field("b"), x -> sin(x))
        xg = first(values(T.create_meshgrid(a.domain)))
        out = T.evaluate_solver_expression(T.SubtractOperator(a, b), Dict{String,Any}(); layout=:g)
        @test gd(out) ≈ (2 .+ sin.(xg)) .- sin.(xg)
        @test all(gd(out) .≈ 2.0)
        @test T.evaluate_solver_expression(T.SubtractOperator(7.0, 2.0), Dict{String,Any}(); template=nothing) == 5.0
    end

    @testset "MultiplyOperator: field*field, field*number, number*field, number*number" begin
        a = seed!(make_field("a"), x -> sin(x))
        b = seed!(make_field("b"), x -> cos(x))
        xg = first(values(T.create_meshgrid(a.domain)))

        ff = T.evaluate_solver_expression(T.MultiplyOperator(a, b), Dict{String,Any}(); layout=:g)
        @test gd(ff) ≈ sin.(xg) .* cos.(xg)

        a2 = seed!(make_field("a"), x -> sin(x))
        fn = T.evaluate_solver_expression(T.MultiplyOperator(a2, 3.0), Dict{String,Any}(); layout=:g)
        @test gd(fn) ≈ 3.0 .* sin.(xg)

        a3 = seed!(make_field("a"), x -> sin(x))
        nf = T.evaluate_solver_expression(T.MultiplyOperator(2.0, a3), Dict{String,Any}(); layout=:g)
        @test gd(nf) ≈ 2.0 .* sin.(xg)

        @test T.evaluate_solver_expression(T.MultiplyOperator(2.0, 4.0), Dict{String,Any}(); template=nothing) == 8.0
    end

    @testset "DivideOperator: field/number, number/number, field/field" begin
        a = seed!(make_field("a"), x -> 2 .+ sin(x))
        xg = first(values(T.create_meshgrid(a.domain)))

        fn = T.evaluate_solver_expression(T.DivideOperator(a, 2.0), Dict{String,Any}(); layout=:g)
        @test gd(fn) ≈ (2 .+ sin.(xg)) ./ 2.0

        @test T.evaluate_solver_expression(T.DivideOperator(6.0, 3.0), Dict{String,Any}(); template=nothing) == 2.0

        num = seed!(make_field("n"), x -> 2 .+ sin(x))
        den = seed!(make_field("d"), x -> 3 .+ cos(x))
        xg2 = first(values(T.create_meshgrid(num.domain)))
        ff = T.evaluate_solver_expression(T.DivideOperator(num, den), Dict{String,Any}(); layout=:g)
        @test gd(ff) ≈ (2 .+ sin.(xg2)) ./ (3 .+ cos.(xg2))
    end

    @testset "PowerOperator: field^p and number^p, bad exponent throws" begin
        a = seed!(make_field("a"), x -> 1 .+ 0.5 .* sin(x))
        xg = first(values(T.create_meshgrid(a.domain)))
        out = T.evaluate_solver_expression(T.PowerOperator(a, 2.0), Dict{String,Any}(); layout=:g)
        @test gd(out) ≈ (1 .+ 0.5 .* sin.(xg)) .^ 2

        @test T.evaluate_solver_expression(T.PowerOperator(3.0, 2.0), Dict{String,Any}(); template=nothing) == 9.0

        # exponent must be numeric: a field exponent throws
        b = make_field("b")
        @test_throws ArgumentError T.evaluate_solver_expression(
            T.PowerOperator(a, b), Dict{String,Any}(); layout=:g)
    end

    @testset "NegateOperator: number, field" begin
        @test T.evaluate_solver_expression(T.NegateOperator(5.0), Dict{String,Any}(); template=nothing) == -5.0
        a = seed!(make_field("a"), x -> sin(x))
        xg = first(values(T.create_meshgrid(a.domain)))
        out = T.evaluate_solver_expression(T.NegateOperator(a), Dict{String,Any}(); layout=:g)
        @test gd(out) ≈ .-sin.(xg)
    end

    @testset "IndexOperator: field element and array element" begin
        a = seed!(make_field("a"), x -> x)         # grid data == grid coords
        xg = first(values(T.create_meshgrid(a.domain)))
        # field[3] in grid layout
        idx = T.IndexOperator(a, (3,))
        val = T.evaluate_solver_expression(idx, Dict{String,Any}(); layout=:g)
        @test val ≈ xg[3]

        # array index: the indexed operand must itself evaluate to an array, so
        # wrap it in an ArrayOperator (with no template -> returns the raw array).
        arr = collect(10.0:20.0)
        ai = T.IndexOperator(T.ArrayOperator(arr), (2,))
        @test T.evaluate_solver_expression(ai, Dict{String,Any}(); layout=:g, template=nothing) == arr[2]
    end

    @testset "composite expression: 2*u + u^2 - 1" begin
        # f(x) = 2*u + u^2 - 1 with u = sin(x)
        u = seed!(make_field("u"), x -> sin(x))
        xg = first(values(T.create_meshgrid(u.domain)))
        u_for_pow = seed!(make_field("u2"), x -> sin(x))
        expr = T.SubtractOperator(
                  T.AddOperator(
                      T.MultiplyOperator(2.0, u),
                      T.PowerOperator(u_for_pow, 2.0)),
                  1.0)
        out = T.evaluate_solver_expression(expr, Dict{String,Any}(); layout=:g)
        @test gd(out) ≈ 2 .* sin.(xg) .+ sin.(xg).^2 .- 1
    end

    @testset "Operator branch: Differentiate ∂x and Laplacian" begin
        # ∂x sin(x) = cos(x), evaluated through the generic Operator dispatch
        u = seed!(make_field("u"), x -> sin(x))
        xg = first(values(T.create_meshgrid(u.domain)))
        d = T.Differentiate(u, _COORDS["x"], 1)
        rd = T.evaluate_solver_expression(d, Dict{String,Any}(); layout=:g)
        @test rd isa T.ScalarField
        @test gd(rd) ≈ cos.(xg) atol=1e-10

        # lap sin(x) = -sin(x)
        u2 = seed!(make_field("u2"), x -> sin(x))
        lap = T.Laplacian(u2)
        rl = T.evaluate_solver_expression(lap, Dict{String,Any}(); layout=:g)
        @test gd(rl) ≈ .-sin.(xg) atol=1e-10
    end

    @testset "composite with derivative: u*∂x(u) - lap(u)" begin
        # Burgers-style RHS pieces: u·u_x - u_xx with u = sin(x)
        u = seed!(make_field("u"), x -> sin(x))
        ux_field = seed!(make_field("uux"), x -> sin(x))
        ulap_field = seed!(make_field("ulap"), x -> sin(x))
        xg = first(values(T.create_meshgrid(u.domain)))
        expr = T.SubtractOperator(
                  T.MultiplyOperator(u, T.Differentiate(ux_field, _COORDS["x"], 1)),
                  T.Laplacian(ulap_field))
        out = T.evaluate_solver_expression(expr, Dict{String,Any}(); layout=:g)
        # analytic: sin*cos - (-sin) = sin*cos + sin
        @test gd(out) ≈ sin.(xg) .* cos.(xg) .+ sin.(xg) atol=1e-10
    end

    @testset "VectorField branches: scale, add, subtract, negate, scalar*vector" begin
        v = T.VectorField(_DIST, _COORDS, "v", (_XB,), Float64)
        T.ensure_layout!(v.components[1], :g)
        T.get_grid_data(v.components[1]) .= 2.0

        v2 = T.VectorField(_DIST, _COORDS, "v2", (_XB,), Float64)
        T.ensure_layout!(v2.components[1], :g)
        T.get_grid_data(v2.components[1]) .= 5.0

        # number * vector
        out = T.evaluate_solver_expression(T.MultiplyOperator(3.0, v), Dict{String,Any}(); layout=:g)
        @test out isa T.VectorField
        T.ensure_layout!(out.components[1], :g)
        @test all(T.get_grid_data(out.components[1]) .≈ 6.0)

        # vector * number
        out2 = T.evaluate_solver_expression(T.MultiplyOperator(v, 4.0), Dict{String,Any}(); layout=:g)
        T.ensure_layout!(out2.components[1], :g)
        @test all(T.get_grid_data(out2.components[1]) .≈ 8.0)

        # vector + vector
        add = T.evaluate_solver_expression(T.AddOperator(v, v2), Dict{String,Any}(); layout=:g)
        @test add isa T.VectorField
        T.ensure_layout!(add.components[1], :g)
        @test all(T.get_grid_data(add.components[1]) .≈ 7.0)

        # vector - vector
        sub = T.evaluate_solver_expression(T.SubtractOperator(v2, v), Dict{String,Any}(); layout=:g)
        T.ensure_layout!(sub.components[1], :g)
        @test all(T.get_grid_data(sub.components[1]) .≈ 3.0)

        # -vector
        neg = T.evaluate_solver_expression(T.NegateOperator(v), Dict{String,Any}(); layout=:g)
        @test neg isa T.VectorField
        T.ensure_layout!(neg.components[1], :g)
        @test all(T.get_grid_data(neg.components[1]) .≈ -2.0)

        # scalarfield * vectorfield  (scale_vector_field)
        s = seed!(make_field("s"), x -> 2.0)
        sv = T.evaluate_solver_expression(T.MultiplyOperator(s, v), Dict{String,Any}(); layout=:g)
        @test sv isa T.VectorField
        T.ensure_layout!(sv.components[1], :g)
        @test all(T.get_grid_data(sv.components[1]) .≈ 4.0)

        # vectorfield * scalarfield  (other order)
        vs = T.evaluate_solver_expression(T.MultiplyOperator(v, s), Dict{String,Any}(); layout=:g)
        @test vs isa T.VectorField
        T.ensure_layout!(vs.components[1], :g)
        @test all(T.get_grid_data(vs.components[1]) .≈ 4.0)
    end

    @testset "error branches: unsupported operands and negation" begin
        a = seed!(make_field("a"), x -> sin(x))
        v = T.VectorField(_DIST, _COORDS, "v", (_XB,), Float64)
        T.ensure_layout!(v.components[1], :g)
        # Add: ScalarField + VectorField is unsupported -> ArgumentError
        @test_throws ArgumentError T.evaluate_solver_expression(
            T.AddOperator(a, v), Dict{String,Any}(); layout=:g)
        # Subtract: ScalarField - VectorField unsupported
        @test_throws ArgumentError T.evaluate_solver_expression(
            T.SubtractOperator(a, v), Dict{String,Any}(); layout=:g)
    end

    @testset "unsupported expression type errors" begin
        @test_throws ErrorException T.evaluate_solver_expression("not an expr", Dict{String,Any}())
        # IndexOperator on a non-array/non-field operand -> ArgumentError
        bad = T.IndexOperator(T.ConstantOperator(3.0), (1,))
        @test_throws ArgumentError T.evaluate_solver_expression(
            bad, Dict{String,Any}(); layout=:g, template=nothing)
    end

    # -----------------------------------------------------------------------
    # Direct helper coverage
    # -----------------------------------------------------------------------
    @testset "_constant_field_from_template (:g and :c)" begin
        tmpl = make_field("t")
        fg = T._constant_field_from_template(tmpl, 7.0; layout=:g)
        @test all(gd(fg) .≈ 7.0)
        fc = T._constant_field_from_template(tmpl, 0.0; layout=:c)
        T.ensure_layout!(fc, :c)
        @test all(iszero, T.get_coeff_data(fc))
    end

    @testset "_coerce_numeric_operand" begin
        # number with nothing template -> returned unchanged
        @test T._coerce_numeric_operand(3.0, nothing) == 3.0
        # number with template -> constant field
        tmpl = make_field("t")
        cf = T._coerce_numeric_operand(2.0, tmpl; layout=:g)
        @test cf isa T.ScalarField && all(gd(cf) .≈ 2.0)
        # non-number passes through unchanged
        f = make_field("f")
        @test T._coerce_numeric_operand(f, tmpl) === f
    end

    @testset "_binary_template selection" begin
        tmpl = make_field("t")
        a = make_field("a")
        b = make_field("b")
        # template wins when present
        @test T._binary_template(a, b, tmpl) === tmpl
        # left field used when no template
        @test T._binary_template(a, 2.0, nothing) === a
        # right field used when left is not a field
        @test T._binary_template(2.0, b, nothing) === b
        # both numbers, no template -> nothing
        @test T._binary_template(1.0, 2.0, nothing) === nothing
    end

    @testset "create_zero_field and create_constant_field" begin
        tmpl = make_field("t")
        z = T.create_zero_field(tmpl)
        T.ensure_layout!(z, :c)
        @test all(iszero, T.get_coeff_data(z))

        # vector-of-variables overload
        z2 = T.create_zero_field([tmpl])
        @test z2 isa T.ScalarField
        @test_throws ArgumentError T.create_zero_field(T.ScalarField[])

        # constant field from an operator carrying a value
        cf = T.create_constant_field(T.ConstantOperator(3.0), [tmpl])
        T.ensure_layout!(cf, :c)
        @test T.get_coeff_data(cf) !== nothing
        @test_throws ArgumentError T.create_constant_field(T.ConstantOperator(1.0), [])
    end

    @testset "get_diff_order default and explicit" begin
        # struct without :order field -> default 1
        @test T.get_diff_order(_NoOrderExpr()) == 1
        # struct with :order -> max(1, Int(order))
        @test T.get_diff_order(_WithOrderExpr(2)) == 2
        @test T.get_diff_order(_WithOrderExpr(0)) == 1
    end

    @testset "get_diff_coordinate: no coord -> nothing" begin
        @test T.get_diff_coordinate(_BareExpr()) === nothing
    end

    # -----------------------------------------------------------------------
    # Jacobian-block builders. These probe duck-typed expr objects via
    # hasfield(); NamedTuples satisfy hasfield, so we drive every branch
    # directly with NamedTuple "expressions" and a real ScalarField variable.
    # -----------------------------------------------------------------------
    @testset "build_jacobian_block: null / unknown / constant / variable" begin
        var = make_field("v")
        T.ensure_layout!(var, :c)
        n = T.compute_field_vector_size(var)

        # null expression -> 1x1 sparse zero (with warning)
        nj = T.build_jacobian_block(nothing, [var], nothing)
        @test size(nj) == (1, 1)
        @test all(iszero, nj)

        # constant expr -> n x n zero matrix
        cj = T.build_jacobian_block((expr_type="constant",), [var], nothing)
        @test size(cj) == (n, n)
        @test all(iszero, cj)

        # variable expr whose field_ref is the variable -> identity of var size
        vj = T.build_jacobian_block((expr_type="variable", field_ref=var), [var], nothing)
        @test size(vj) == (n, n)
        @test dense(vj) == ident(n)

        # unknown expr_type -> warns, falls through to identity (size = total var size)
        uj = T.build_jacobian_block((expr_type="mystery",), [var], nothing)
        @test size(uj) == (n, n)
        @test dense(uj) == ident(n)

        # expr with NO expr_type field -> fallback identity
        fj = T.build_jacobian_block((foo=1,), [var], nothing)
        @test size(fj) == (n, n)
    end

    @testset "build_variable_jacobian_block edge cases" begin
        var = make_field("v")
        T.ensure_layout!(var, :c)
        # missing :field_ref -> 1x1 identity (warn)
        mj = T.build_variable_jacobian_block((foo=1,), [var])
        @test size(mj) == (1, 1)
        # field_ref not in variable list -> 1x1 identity (warn)
        other = make_field("other")
        nfj = T.build_variable_jacobian_block((field_ref=other,), [var])
        @test size(nfj) == (1, 1)
    end

    @testset "build_operator_jacobian_block: Add / Multiply / Differentiate / unknown" begin
        var = make_field("v")
        T.ensure_layout!(var, :c)
        n = T.compute_field_vector_size(var)
        var_expr = (expr_type="variable", field_ref=var)

        # Add of two variable operands -> sum of two identities == 2*I
        addj = T.build_operator_jacobian_block(
            (operator="Add", operands=[var_expr, var_expr]), [var], nothing)
        @test size(addj) == (n, n)
        @test dense(addj) == 2 .* ident(n)

        # Multiply of two variable operands -> sum of identities (linearized) == 2*I
        mulj = T.build_operator_jacobian_block(
            (operator="Multiply", operands=[var_expr, var_expr]), [var], nothing)
        @test size(mulj) == (n, n)
        @test dense(mulj) == 2 .* ident(n)

        # Multiply with a single operand -> identity fallback
        mul1 = T.build_operator_jacobian_block(
            (operator="Multiply", operands=[var_expr]), [var], nothing)
        @test size(mul1) == (n, n)

        # Differentiate -> returns operand Jacobian (identity here)
        diffj = T.build_operator_jacobian_block(
            (operator="Differentiate", operands=[var_expr]), [var], nothing)
        @test size(diffj) == (n, n)
        @test dense(diffj) == ident(n)

        # Differentiate with no operands -> fallthrough identity
        diff0 = T.build_operator_jacobian_block(
            (operator="Differentiate", operands=Any[]), [var], nothing)
        @test size(diff0) == (n, n)

        # Unknown operator -> warns, fallthrough identity
        unk = T.build_operator_jacobian_block(
            (operator="Bogus", operands=Any[]), [var], nothing)
        @test size(unk) == (n, n)

        # Malformed operator expr (missing :operator/:operands) -> identity
        bad = T.build_operator_jacobian_block((foo=1,), [var], nothing)
        @test size(bad) == (max(n, 1), max(n, 1))
    end

    @testset "build_jacobian_block dispatches operator subtype" begin
        var = make_field("v")
        T.ensure_layout!(var, :c)
        n = T.compute_field_vector_size(var)
        var_expr = (expr_type="variable", field_ref=var)
        # expr_type == "operator" routes to build_operator_jacobian_block
        oj = T.build_jacobian_block(
            (expr_type="operator", operator="Add", operands=[var_expr, var_expr]),
            [var], nothing)
        @test size(oj) == (n, n)
        @test dense(oj) == 2 .* ident(n)
    end
end
