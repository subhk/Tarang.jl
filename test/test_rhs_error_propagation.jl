# ============================================================================
# Regression: a term that cannot be evaluated must ABORT the RHS evaluation,
# never silently become zero — and `div(a*grad(u))` must actually evaluate.
# ============================================================================
#
# The bug. Writing the physically standard variable-viscosity term
# `div(nu_e*grad(u))` on an RHS silently zeroed the ENTIRE right-hand side of
# that field, taking unrelated terms down with it:
#
#     ||RHS = 0||                       = 7.997600359963
#     ||-u*∂x(u)||                      = 7.997600357631
#     ||-u*∂x(u) + div(nu_e*grad(u))||  = 7.997600359963   <- identical to RHS = 0
#     max|div-form - RHS=0|             = 0.0   <- the advection went too
#
# `evaluate_divergence` handled only `VectorField` operands and threw
# `ArgumentError` for the `MultiplyOperator` that `nu_e*grad(u)` parses to. The
# interpreted RHS evaluator caught that, `@warn`ed, and left the field's RHS at
# the zero it was initialised to — so the whole sum, advection included, was
# replaced by a physically meaningful and completely wrong value. The run
# completed and reported success.
#
# Two things changed, and both are asserted here:
#
#   1. A failed RHS evaluation re-raises instead of degrading to "RHS = 0"
#      (`_evaluate_rhs_interpreted` in timesteppers/state_utils.jl). Zero is a
#      real right-hand side, so silently substituting it turns an unsupported
#      term into a wrong answer rather than a failure.
#
#   2. `div(a*grad(u))` is supported (`evaluate_divergence` in
#      operators/derivatives/derivatives_eval.jl), expanded with the exact
#      product rule
#
#          ∇·(a ∇u) = a ∇²u + ∇a·∇u = Σₖ (a ∂ₖ²u + ∂ₖa ∂ₖu)
#
#      so every derivative is spectral and only the products are formed on the
#      grid — matching the hand-written spelling `a*lap(u) + ∂x(a)*∂x(u) + …`
#      to round-off.
#
# The manufactured solution used throughout, on a doubly periodic 2π box:
#
#     a(x,y) = 2 + sin(x)          u(x,y) = sin(x)·cos(y)
#
#     ∂ₓu = cos x cos y            ∂ᵧu = -sin x sin y      ∇²u = -2 sin x cos y
#     ∇·(a∇u) = a∇²u + ∂ₓa ∂ₓu
#             = -2(2+sin x) sin x cos y + cos²x cos y
#             = cos y · (cos²x - 2 sin²x - 4 sin x)
#
# Both factors are band-limited to |k| ≤ 1, so the result is band-limited to
# |k| ≤ 2 and a 16-point grid resolves it exactly: the assertions below are at
# round-off (measured ≈ 2e-14), not at truncation error.
#
# Run with:  julia --project=. test/test_rhs_error_propagation.jl

using Test
using Tarang

@testset "RHS error propagation and variable-coefficient diffusion" begin

    N = 16
    coords = CartesianCoordinates("x", "y")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xbasis, ybasis))

    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # Closed-form ∇·(a∇u) and -u ∂ₓu for the manufactured solution above.
    exact_diffusion = @. cos(y) * (cos(x)^2 - 2 * sin(x)^2 - 4 * sin(x))
    exact_advection = @. -(sin(x) * cos(y)) * (cos(x) * cos(y))

    # Fresh (u, nu_e) carrying the manufactured solution. Fresh every time: the
    # solver mutates the fields it is handed.
    function manufactured_fields()
        u = ScalarField(domain, "u")
        u["g"] = @. sin(x) * cos(y)
        nu_e = ScalarField(domain, "nu_e")
        nu_e["g"] = @. 2 + sin(x)
        return u, nu_e
    end

    # Explicit RHS (the F vector) of a single-equation IVP, in grid layout.
    function rhs_of(equation::String)
        u, nu_e = manufactured_fields()
        problem = IVP([u])
        add_parameters!(problem, nu_e=nu_e)
        add_equation!(problem, equation)
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :g)
        return Array(get_grid_data(rhs))
    end

    grid_of(field) = (ensure_layout!(field, :g); Array(get_grid_data(field)))

    # ------------------------------------------------------------------
    # (b) div(a*grad(u)) evaluates correctly
    # ------------------------------------------------------------------
    @testset "div(a*grad(u)) matches the manufactured solution" begin
        # Through the equation parser — the exact path that used to be zeroed.
        F = rhs_of("∂t(u) = div(nu_e*grad(u))")
        @test F ≈ exact_diffusion atol=1e-10 rtol=1e-10

        # Not the all-zero field the pre-fix code produced.
        @test maximum(abs, F) > 1.0
    end

    @testset "div(a*grad(u)) accepts either factor order and both product spellings" begin
        u, a = manufactured_fields()
        gradient = Tarang.Gradient(u, coords)

        # `MultiplyOperator` is what the equation-string parser builds…
        coefficient_first = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(a, gradient)), :g)
        gradient_first = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(gradient, a)), :g)
        # …and the deferred `Multiply` future is what Julia's `*` builds.
        deferred = Tarang.evaluate(Tarang.divergence(a * grad(u)), :g)

        @test grid_of(coefficient_first) ≈ exact_diffusion atol=1e-10 rtol=1e-10
        @test grid_of(gradient_first) ≈ exact_diffusion atol=1e-10 rtol=1e-10
        @test grid_of(deferred) ≈ exact_diffusion atol=1e-10 rtol=1e-10
    end

    @testset "div(a*grad(u)) equals the hand-expanded product rule" begin
        # `a*lap(u) + ∂x(a)*∂x(u) + ∂y(a)*∂y(u)` is the spelling that already
        # worked before the fix; the div() form must agree with it to round-off,
        # not merely to truncation error.
        div_form = rhs_of("∂t(u) = div(nu_e*grad(u))")
        hand_form = rhs_of("∂t(u) = nu_e*lap(u) + ∂x(nu_e)*∂x(u) + ∂y(nu_e)*∂y(u)")
        @test div_form ≈ hand_form atol=1e-11 rtol=1e-11
        @test hand_form ≈ exact_diffusion atol=1e-10 rtol=1e-10
    end

    @testset "a constant coefficient reduces to a*lap(u)" begin
        u, _ = manufactured_fields()
        gradient = Tarang.Gradient(u, coords)
        scaled = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(2.5, gradient)), :g)
        laplacian = Tarang.evaluate(Tarang.Laplacian(u), :g)
        @test grid_of(scaled) ≈ 2.5 .* grid_of(laplacian) atol=1e-12 rtol=1e-12
    end

    @testset "div(a*grad(u)) applies component-wise to a vector field" begin
        # ∇·(a∇u)ⱼ = Σₖ ∂ₖ(a ∂ₖuⱼ): the variable-viscosity momentum term.
        _, a = manufactured_fields()
        v = VectorField(domain, "v")
        v.components[1]["g"] = @. sin(x) * cos(y)
        v.components[2]["g"] = @. cos(x) * sin(y)

        result = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(a, Tarang.Gradient(v, coords))), :g)
        @test result isa VectorField

        # component 2: u₂ = cos x sin y, ∇²u₂ = -2 cos x sin y, ∂ₓa ∂ₓu₂ = -cos x sin x sin y
        exact_second = @. -2 * (2 + sin(x)) * cos(x) * sin(y) - cos(x) * sin(x) * sin(y)
        @test grid_of(result.components[1]) ≈ exact_diffusion atol=1e-10 rtol=1e-10
        @test grid_of(result.components[2]) ≈ exact_second atol=1e-10 rtol=1e-10
    end

    @testset "an expression coefficient survives the derivative buffer rotation" begin
        # `evaluate_differentiate` hands out results from a 16-slot rotating pool.
        # An expression-valued coefficient lands in one of those slots, and a
        # 3-component field in 3D issues 3 coords × 3 derivatives × 3 components
        # = 27 further checkouts — enough to recycle the coefficient's slot out
        # from under the accumulation. Unpinned, components 2 and 3 came back
        # wrong by O(1) (measured 2.0 and 2.9 on O(1) data) with no error raised.
        M = 8
        coords3 = CartesianCoordinates("x", "y", "z")
        dist3 = Distributor(coords3; dtype=Float64, device=CPU())
        bases3 = (RealFourier(coords3["x"]; size=M, bounds=(0.0, 2π)),
                  RealFourier(coords3["y"]; size=M, bounds=(0.0, 2π)),
                  RealFourier(coords3["z"]; size=M, bounds=(0.0, 2π)))
        domain3 = Domain(dist3, bases3)
        mesh3 = Tarang.create_meshgrid(domain3)
        x3, y3, z3 = mesh3["x"], mesh3["y"], mesh3["z"]

        nu = ScalarField(domain3, "nu")
        nu["g"] = @. 3 + cos(y3)                        # a = ∂y(nu) = -sin(y)

        v = VectorField(domain3, "v")
        v.components[1]["g"] = @. sin(x3) * cos(y3)
        v.components[2]["g"] = @. cos(x3) * sin(z3)
        v.components[3]["g"] = @. sin(y3) * cos(z3)

        coefficient = Tarang.Differentiate(nu, coords3["y"], 1)
        result = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(coefficient, Tarang.Gradient(v, coords3))), :g)

        # a∇²uⱼ + ∂ᵧa ∂ᵧuⱼ, with a = -sin y and ∂ᵧa = -cos y (∂ₓa = ∂_z a = 0).
        exact = ((@. -sin(y3) * (-2 * sin(x3) * cos(y3)) + (-cos(y3)) * (-sin(x3) * sin(y3))),
                 (@. -sin(y3) * (-2 * cos(x3) * sin(z3))),
                 (@. -sin(y3) * (-2 * sin(y3) * cos(z3)) + (-cos(y3)) * (cos(y3) * cos(z3))))
        for j in 1:3
            @test grid_of(result.components[j]) ≈ exact[j] atol=1e-10 rtol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # (c) an unrelated term in the same RHS sum is unaffected
    # ------------------------------------------------------------------
    @testset "an unrelated term in the same RHS sum survives" begin
        # THE specific regression: before the fix the div() failure destroyed
        # the advection term sitting next to it, and the combined RHS came back
        # bit-identical to "RHS = 0".
        F_advection = rhs_of("∂t(u) = -u*∂x(u)")
        F_diffusion = rhs_of("∂t(u) = div(nu_e*grad(u))")
        F_both      = rhs_of("∂t(u) = -u*∂x(u) + div(nu_e*grad(u))")

        @test F_advection ≈ exact_advection atol=1e-10 rtol=1e-10
        @test F_both ≈ F_advection .+ F_diffusion atol=1e-10 rtol=1e-10
        @test F_both ≈ exact_advection .+ exact_diffusion atol=1e-10 rtol=1e-10

        # Guard the guard: each term must be big enough that dropping it fails
        # the assertions above (pre-fix, F_both was identically zero).
        @test maximum(abs, F_advection) > 0.1
        @test maximum(abs, F_both .- F_diffusion) > 0.1
        @test maximum(abs, F_both) > 1.0
    end

    # ------------------------------------------------------------------
    # (a) a genuinely broken RHS raises instead of producing a zero RHS
    # ------------------------------------------------------------------
    @testset "a broken RHS term raises instead of silently zeroing the RHS" begin
        # `div(nu_e*nu_e)` is a divergence of a scalar: unevaluable, and exactly
        # the class of failure (an ArgumentError out of evaluate_divergence)
        # that used to be caught, warned about, and turned into RHS = 0.
        @test_throws ArgumentError rhs_of("∂t(u) = div(nu_e*nu_e)")

        # …including when a perfectly good term shares the sum: the failure must
        # not be papered over just because the rest of the RHS evaluated.
        @test_throws ArgumentError rhs_of("∂t(u) = -u*∂x(u) + div(nu_e*nu_e)")
    end

    @testset "unsupported divergence operands throw, naming what was unsupported" begin
        u, a = manufactured_fields()
        v = VectorField(domain, "v")
        v.components[1]["g"] = @. sin(x)
        v.components[2]["g"] = @. cos(y)

        # A non-scalar coefficient has no product-rule expansion here.
        vector_coefficient = Tarang.Divergence(
            Tarang.MultiplyOperator(v, Tarang.Gradient(u, coords)))
        @test_throws ArgumentError Tarang.evaluate(vector_coefficient, :g)
        message = try
            Tarang.evaluate(vector_coefficient, :g)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("SCALAR coefficient", message)
        @test occursin("VectorField", message)

        # ∇· of a scalar is rank-nonsense; it must say so rather than return 0.
        @test_throws ArgumentError Tarang.evaluate(Tarang.Divergence(u), :g)
    end

    @testset "div(grad(u)) still evaluates to the Laplacian" begin
        # The generic path: any operand that evaluates to a VectorField.
        u, _ = manufactured_fields()
        divergence_of_gradient = Tarang.evaluate(
            Tarang.Divergence(Tarang.Gradient(u, coords)), :g)
        laplacian = Tarang.evaluate(Tarang.Laplacian(u), :g)
        @test grid_of(divergence_of_gradient) ≈ grid_of(laplacian) atol=1e-12 rtol=1e-12
    end
end
