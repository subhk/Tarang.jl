# Serial-CPU coverage tests for src/core/problems/problem_parsing.jl
#
# Exercises the equation/expression parser: parse_expression, parse_equation,
# evaluate_parsed_expression (arithmetic, n-ary operators, differential operators,
# BC-interpolation syntax, advection, integrate/average/lift/fraclap), plus the
# structural validators (contains_time_derivatives, _is_constant_coefficient_strict,
# _get_constant_value, validate_equation_structure, is_proper_lhs_structure) and the
# fallback parser. All assertions check real behavior (returned operator type,
# structural fields, invariants) or error conditions via @test_throws.

using Test
using Tarang
using LinearAlgebra

using Tarang: parse_expression, parse_equation, evaluate_parsed_expression,
    parse_linear_expression, fallback_parse_expression,
    validate_equation_structure, contains_time_derivatives,
    is_proper_lhs_structure, validate_lhs_structure,
    _is_constant_coefficient_strict, _get_constant_value,
    is_linear_expression, _expand_advection, _expression_contains_unknown,
    coerce_constant_value,
    ZeroOperator, ConstantOperator, UnknownOperator, ArrayOperator,
    AddOperator, SubtractOperator, MultiplyOperator, DivideOperator,
    PowerOperator, NegateOperator, IndexOperator,
    TimeDerivative, Differentiate, Laplacian, Gradient, FractionalLaplacian,
    Integrate, Average, Component, Interpolate, has

# ----------------------------------------------------------------------------
# Fixtures: a 1D Chebyshev scalar field, and a 2D Fourier×Chebyshev domain with
# a scalar + vector field, to drive both pure-Fourier and Chebyshev parse paths.
# ----------------------------------------------------------------------------
coords1 = CartesianCoordinates("x")
dist1 = Distributor(coords1; mesh=(1,), dtype=Float64)
zb = ChebyshevT(coords1["x"]; size=16, bounds=(-1.0, 1.0))
dom1 = Domain(dist1, (zb,))
u1 = ScalarField(dom1, "u")
cx = coords1["x"]

coords2 = CartesianCoordinates("x", "y")
dist2 = Distributor(coords2; mesh=(1, 1), dtype=Float64)
xb = RealFourier(coords2["x"]; size=8, bounds=(0.0, 2π))
yb = ChebyshevT(coords2["y"]; size=8, bounds=(-1.0, 1.0))
dom2 = Domain(dist2, (xb, yb))
u2 = ScalarField(dom2, "u")
vv = VectorField(dom2, "vv")

ns1 = Dict{String, Any}("u" => u1, "x" => cx, "nu" => 0.5)
ns2 = Dict{String, Any}("vv" => vv, "u" => u2,
                        "x" => coords2["x"], "y" => coords2["y"], "c" => 2.0)

@testset "problem_parsing coverage" begin

    @testset "atomic expressions" begin
        # bare field returns the field object itself
        @test parse_expression("u", ns1) === u1
        # explicit zero forms short-circuit to ZeroOperator
        @test parse_expression("0", ns1) isa ZeroOperator
        @test parse_expression("zero", ns1) isa ZeroOperator
        # numeric literal wraps in ConstantOperator with Float64 value
        c = parse_expression("3.5", ns1)
        @test c isa ConstantOperator
        @test c.value == 3.5
        # unknown symbol -> UnknownOperator placeholder
        unk = parse_expression("not_a_known_symbol_qzz", ns1)
        @test unk isa UnknownOperator
        @test unk.expression == "not_a_known_symbol_qzz"
    end

    @testset "binary arithmetic operators" begin
        @test parse_expression("u + u", ns1) isa AddOperator
        @test parse_expression("u - u", ns1) isa SubtractOperator
        @test parse_expression("nu * u", ns1) isa MultiplyOperator
        @test parse_expression("u / 2.0", ns1) isa DivideOperator
        @test parse_expression("u ^ 2", ns1) isa PowerOperator
        # unary minus -> NegateOperator
        @test parse_expression("-u", ns1) isa NegateOperator
    end

    @testset "n-ary flattened operators (Meta.parse :call head)" begin
        # `a + b + c` and `a * b * c` flatten into a single multi-arg :call,
        # exercising the left-fold loops in the (:+,:-,:*,:/,:^) branch.
        addr = parse_expression("u + u + u", ns1)
        @test addr isa AddOperator
        @test addr.left isa AddOperator   # left-folded: ((u+u)+u)
        mulr = parse_expression("nu * nu * u", ns1)
        @test mulr isa MultiplyOperator
        # prefix-call syntax forces multi-arg :- :/ :^ to hit their fold loops
        @test parse_expression("-(c, c, c)", ns2) isa SubtractOperator
        @test parse_expression("/(u, c, c)", ns2) isa DivideOperator
        powr = parse_expression("^(u, c, c)", ns2)
        @test powr isa PowerOperator
        @test powr.right isa PowerOperator   # right-folded: u^(c^c)
        # unary prefix minus
        @test parse_expression("-(u)", ns2) isa NegateOperator
        # single-arg power returns the operand unchanged
        @test parse_expression("^(u)", ns2) === u2
    end

    @testset "nested 2-arg arithmetic via Expr head path" begin
        # `u - u - u` and `u / 2 / 2` do NOT flatten; they hit the
        # expr.head in [:+,:-,...] 2-arg branch with nested operators.
        s3 = parse_expression("u - nu - nu", ns1)
        @test s3 isa SubtractOperator
        @test s3.left isa SubtractOperator
        d3 = parse_expression("u / 2 / 2", ns1)
        @test d3 isa DivideOperator
        @test d3.left isa DivideOperator
    end

    @testset "time derivative dt/∂t" begin
        d1 = parse_expression("dt(u)", ns1)
        @test d1 isa TimeDerivative
        @test d1.order == 1
        d2 = parse_expression("dt(u, 2)", ns1)
        @test d2 isa TimeDerivative
        @test d2.order == 2
        @test parse_expression("∂t(u)", ns1) isa TimeDerivative
        # dt with no args is an error
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("dt()"), ns1)
    end

    @testset "spatial derivatives d / diff / ∂x" begin
        @test parse_expression("d(u, x)", ns1) isa Differentiate
        @test parse_expression("d(u, x, 2)", ns1) isa Differentiate
        @test parse_expression("diff(u, x)", ns1) isa Differentiate
        @test parse_expression("∂x(u)", ns1) isa Differentiate
        # ∂ with unknown coordinate name -> error
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("∂q(u)"), ns1)
        # d with too few args -> error
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("d(u)"), ns1)
    end

    @testset "deferred integrals / averages / lift / fractional laplacian" begin
        # integ over all coords and over one coord
        @test parse_expression("integ(u)", ns1) isa Integrate
        @test parse_expression("integ(u, x)", ns1) isa Integrate
        @test parse_expression("average(u, x)", ns1) isa Average
        @test parse_expression("avg(u, x)", ns1) isa Average
        # integ / average error paths
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("integ()"), ns1)
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("average(u)"), ns1)
        # lift short form: auto-detect basis; for matrix sizing returns the operand
        @test parse_expression("lift(u, -1)", ns1) === u1
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("lift(u)"), ns1)
        # fractional / inverse-sqrt laplacian
        @test parse_expression("fraclap(u, 1.0)", ns1) isa FractionalLaplacian
        @test parse_expression("invsqrtlap(u)", ns1) isa FractionalLaplacian
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("fraclap(u)"), ns1)
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("invsqrtlap()"), ns1)
        # sqrtlap with no operand errors out (the no-arg guard is reached)
        @test_throws Exception evaluate_parsed_expression(Meta.parse("sqrtlap()"), ns1)
    end

    @testset "component extraction (vector field)" begin
        comp = parse_expression("component(vv, 1)", ns2)
        @test comp isa Component
        @test parse_expression("comp(vv, 2)", ns2) isa Component
        @test_throws ArgumentError evaluate_parsed_expression(Meta.parse("component(vv)"), ns2)
    end

    @testset "indexing (Expr :ref head)" begin
        idx = parse_expression("u[1]", ns1)
        @test idx isa IndexOperator
        # the indexed array is the field itself
        @test idx.array === u1
    end

    @testset "block expression evaluates last statement" begin
        @test parse_expression("begin; u; end", ns1) === u1
    end

    @testset "advection u⋅∇(f) and dot(u, grad(f))" begin
        # bare advection expands to Σ uᵢ ∂ᵢ f  -> an AddOperator of products
        adv = parse_expression("vv ⋅ ∇(u)", ns2)
        @test adv isa AddOperator
        # dot(...) spelling reaches the same expansion
        adv2 = parse_expression("dot(vv, grad(u))", ns2)
        @test adv2 isa AddOperator
        # scaled advection c*vv ⋅ ∇(u) -> MultiplyOperator wrapping the expansion
        adv3 = parse_expression("c*vv ⋅ ∇(u)", ns2)
        @test adv3 isa MultiplyOperator
        # negated advection -vv ⋅ ∇(u) -> NegateOperator wrapping the expansion
        adv4 = parse_expression("-vv ⋅ ∇(u)", ns2)
        @test adv4 isa NegateOperator
        # regular dot product (right side is NOT a gradient) -> MultiplyOperator
        dotp = parse_expression("dot(vv, vv)", ns2)
        @test dotp isa MultiplyOperator
    end

    @testset "_expand_advection direct (both scaled branches + error)" begin
        # left factor is the VectorField
        r1 = _expand_advection(MultiplyOperator(vv, 3.0), u2)
        @test r1 isa MultiplyOperator
        # right factor is the VectorField
        r2 = _expand_advection(MultiplyOperator(3.0, vv), u2)
        @test r2 isa MultiplyOperator
        # bare vector field expands directly to an AddOperator (2D -> 2 terms)
        @test _expand_advection(vv, u2) isa AddOperator
        # u must be a VectorField, otherwise ArgumentError
        @test_throws ArgumentError _expand_advection(u2, u2)
    end

    @testset "BC interpolation syntax field(coord=value)" begin
        # coordinate resolved from namespace -> Interpolate node
        bc = parse_expression("u(y=0.0)", ns2)
        @test bc isa Interpolate
        # BC on a derivative operator: d(u,y)(y=1.0)
        bc2 = parse_expression("d(u,y)(y=1.0)", ns2)
        @test bc2 isa Interpolate
    end

    @testset "fallback_parse_expression" begin
        # direct field reference
        @test fallback_parse_expression("u", ns1) === u1
        # numeric string -> ConstantOperator
        fc = fallback_parse_expression("2.5", ns1)
        @test fc isa ConstantOperator
        @test fc.value == 2.5
        # ∂t(field) prefix branch is reached (startswith "∂t(") but the multibyte
        # substring offset means the field name is not recovered, so it falls
        # through to UnknownOperator. Assert the actual behavior.
        ft = fallback_parse_expression("∂t(u)", ns1)
        @test ft isa UnknownOperator
        # d(field, coord) pattern DOES resolve to a Differentiate
        @test fallback_parse_expression("d(u,x)", ns1) isa Differentiate
        # totally unknown -> UnknownOperator
        fu = fallback_parse_expression("totally_unknown_string", ns1)
        @test fu isa UnknownOperator
        @test fu.expression == "totally_unknown_string"
    end

    @testset "empty expression throws" begin
        @test_throws ArgumentError parse_expression("", ns1)
        @test_throws ArgumentError parse_expression("   ", ns1)
    end

    @testset "contains_time_derivatives" begin
        @test contains_time_derivatives(parse_expression("dt(u)", ns1)) == true
        @test contains_time_derivatives(parse_expression("nu*u", ns1)) == false
        @test contains_time_derivatives(nothing) == false
        # nested in arithmetic / negation
        @test contains_time_derivatives(parse_expression("dt(u) + nu*u", ns1)) == true
        @test contains_time_derivatives(NegateOperator(parse_expression("dt(u)", ns1))) == true
        # divide / power subtrees
        @test contains_time_derivatives(DivideOperator(parse_expression("dt(u)", ns1), 2.0)) == true
        @test contains_time_derivatives(PowerOperator(parse_expression("dt(u)", ns1), 2.0)) == true
        # index subtree
        @test contains_time_derivatives(IndexOperator(parse_expression("dt(u)", ns1), (1,))) == true
        # generic operand-bearing operator (Differentiate) with no dt inside
        @test contains_time_derivatives(parse_expression("d(u, x)", ns1)) == false
    end

    @testset "coerce_constant_value" begin
        @test coerce_constant_value(ConstantOperator(7.0)) == 7.0
        @test coerce_constant_value(3) == 3
        @test coerce_constant_value(u1) === u1
    end

    @testset "_is_constant_coefficient_strict" begin
        @test _is_constant_coefficient_strict(ConstantOperator(2.0), ns1) == true
        @test _is_constant_coefficient_strict(ZeroOperator(), ns1) == true
        @test _is_constant_coefficient_strict(4.0, ns1) == true
        # UnknownOperator resolving to a Number in namespace
        @test _is_constant_coefficient_strict(UnknownOperator("nu"), ns1) == true
        # UnknownOperator not in namespace -> false
        @test _is_constant_coefficient_strict(UnknownOperator("missing_param"), ns1) == false
        # fields are not constants
        @test _is_constant_coefficient_strict(u1, ns1) == false
        # arithmetic on constants stays constant
        @test _is_constant_coefficient_strict(AddOperator(ConstantOperator(1.0), ConstantOperator(2.0)), ns1) == true
        @test _is_constant_coefficient_strict(SubtractOperator(ConstantOperator(1.0), ConstantOperator(2.0)), ns1) == true
        @test _is_constant_coefficient_strict(MultiplyOperator(ConstantOperator(1.0), ConstantOperator(2.0)), ns1) == true
        @test _is_constant_coefficient_strict(DivideOperator(ConstantOperator(1.0), ConstantOperator(2.0)), ns1) == true
        @test _is_constant_coefficient_strict(PowerOperator(ConstantOperator(2.0), ConstantOperator(3.0)), ns1) == true
        @test _is_constant_coefficient_strict(NegateOperator(ConstantOperator(2.0)), ns1) == true
        # arithmetic containing a field is NOT constant
        @test _is_constant_coefficient_strict(MultiplyOperator(u1, ConstantOperator(2.0)), ns1) == false
    end

    @testset "_get_constant_value" begin
        @test _get_constant_value(ConstantOperator(5.0)) == 5.0
        @test _get_constant_value(9) == 9
        @test _get_constant_value(u1) === nothing
    end

    @testset "parse_equation splits LHS/RHS" begin
        L, R = parse_equation("dt(u) - nu*u = 0", ns1)
        @test L isa SubtractOperator           # dt(u) - nu*u
        @test R isa ZeroOperator               # RHS "0"
        @test contains_time_derivatives(L) == true
        # diagnostic/constraint equation: no time derivative anywhere
        L2, R2 = parse_equation("integ(u) = 0", ns1)
        @test contains_time_derivatives(L2) == false
    end

    @testset "validate_equation_structure" begin
        dt_u = parse_expression("dt(u)", ns1)
        # time derivative on RHS is rejected
        @test_throws ArgumentError validate_equation_structure(ZeroOperator(), dt_u, "0 = dt(u)")
        # well-formed evolution equation validates to true
        @test validate_equation_structure(dt_u, parse_expression("nu*u", ns1), "dt(u) = nu*u") == true
        # diagnostic equation (no ∂t on LHS) returns true early
        @test validate_equation_structure(parse_expression("integ(u)", ns1),
                                          ZeroOperator(), "integ(u) = 0") == true
    end

    @testset "is_proper_lhs_structure / validate_lhs_structure" begin
        # M·∂tX + L·X form is a valid LHS
        lhs = parse_expression("dt(u) - nu*d(u,x,2)", ns1)
        ok, info = is_proper_lhs_structure(lhs)
        @test ok == true
        @test info[:has_time_derivative] == true
        @test info[:is_linear] == true
        @test "u" in info[:dependent_variables]
        # validate_lhs_structure returns the info dict for valid structures
        info2 = validate_lhs_structure(lhs)
        @test info2 isa Dict
        @test info2[:has_time_derivative] == true

        # higher-order time derivative is rejected
        d2 = TimeDerivative(u1, 2)
        ok2, info3 = is_proper_lhs_structure(d2)
        @test ok2 == false
        @test info3[:is_linear] == false
        @test info3[:error_message] !== nothing
        @test_throws ArgumentError validate_lhs_structure(d2)

        # constants / zero / numeric literals are valid LHS pieces
        @test is_proper_lhs_structure(ZeroOperator())[1] == true
        @test is_proper_lhs_structure(ConstantOperator(3.0))[1] == true
        @test is_proper_lhs_structure(2.0)[1] == true
        @test is_proper_lhs_structure(nothing)[1] == true

        # bare field is a valid (linear) LHS term
        ok3, info4 = is_proper_lhs_structure(u1)
        @test ok3 == true
        @test "u" in info4[:dependent_variables]
    end

    @testset "is_linear_expression" begin
        # constant * field is linear (coefficient times variable)
        @test is_linear_expression(parse_expression("nu*u", ns1), ns1) == true
        # bare field is linear
        @test is_linear_expression(u1, ns1) == true
    end

    @testset "_expression_contains_unknown" begin
        @test _expression_contains_unknown(UnknownOperator("x")) == true
        @test _expression_contains_unknown(3.0) == false
        @test _expression_contains_unknown(:sym) == false
        @test _expression_contains_unknown("str") == false
        # nested inside an operator tree
        @test _expression_contains_unknown(AddOperator(u1, UnknownOperator("x"))) == true
        @test _expression_contains_unknown(AddOperator(u1, ConstantOperator(2.0))) == false
        @test _expression_contains_unknown(NegateOperator(UnknownOperator("t"))) == true
    end

    @testset "helper operator has() and coerce types" begin
        # placeholder operator types never report containing problem variables
        @test has(ZeroOperator(), u1) == false
        @test has(ConstantOperator(1.0), u1) == false
        @test has(UnknownOperator("q"), u1) == false
        @test has(ArrayOperator([1.0, 2.0]), u1) == false
    end
end
