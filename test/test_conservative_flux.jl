# ============================================================================
# div(a*u) — the conservative flux divergence ∇·(ρu)
# ============================================================================
#
# `div(a*grad(u))` (a scalar coefficient times a Gradient) was implemented
# earlier; its sibling, a scalar coefficient times a VECTOR FIELD, still threw
#
#     ArgumentError: Cannot multiply ScalarField and VectorField
#
# out of `_multiply_result`, which took down every spelling of the conservative
# form: mass conservation `∂t(ρ) = -div(ρ*u)`, conservative advection
# `div(u*c)`, a variable-coefficient flux `div(κ*q)`. Two changes:
#
#   1. `ScalarField * VectorField` (either order) is now a legal multiply
#      yielding a VectorField (`_multiply_result` in operators/evaluate.jl).
#      It always was legal on the deferred-`Multiply` path (`combine_multiply`
#      → `scale_vector_field`), so the same term evaluated one way inside the
#      solver and threw when the parsed operator tree was evaluated.
#
#   2. `evaluate_divergence` recognises the product and expands it with the
#      exact product rule
#
#          ∇·(a u) = a (∇·u) + u·∇a = Σᵢ (a ∂ᵢuᵢ + uᵢ ∂ᵢa)
#
#      so every derivative is spectral on `a` or `u` themselves and only the
#      pointwise products are formed on the grid — matching the hand-written
#      `a*div(u) + dot(u, grad(a))` to round-off rather than differentiating an
#      assembled product that need not be resolved on the grid its factors are.
#
# The manufactured solution used throughout, on a doubly periodic 2π box:
#
#     a(x,y) = 2 + sin x        u = (sin x cos y, cos x sin y)
#
#     ∇·u  = cos x cos y + cos x cos y = 2 cos x cos y
#     ∇a   = (cos x, 0)                 u·∇a = sin x cos x cos y
#     ∇·(a u) = (2 + sin x)(2 cos x cos y) + sin x cos x cos y
#
# Every factor is band-limited to |k| ≤ 1 and the products to |k| ≤ 2, so a
# 16-point grid resolves the answer exactly: the assertions below sit at
# round-off (measured ≈ 5e-15 against ‖exact‖∞ ≈ 4.76), not at truncation error.
#
# Run with:  julia --project=. test/test_conservative_flux.jl

using Test
using Tarang

@testset "div(a*u) — conservative flux divergence" begin

    N = 16
    coords = CartesianCoordinates("x", "y")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xbasis, ybasis))

    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # Closed-form ∇·(a u) for the manufactured solution above.
    exact_flux = @. (2 + sin(x)) * (2 * cos(x) * cos(y)) + (sin(x) * cos(y)) * cos(x)

    # Fresh (a, u) every time: evaluation transforms the fields it is handed.
    function manufactured_fields()
        a = ScalarField(domain, "a")
        a["g"] = @. 2 + sin(x)
        u = VectorField(domain, "u")
        u.components[1]["g"] = @. sin(x) * cos(y)
        u.components[2]["g"] = @. cos(x) * sin(y)
        return a, u
    end

    grid_of(field) = (ensure_layout!(field, :g); Array(get_grid_data(field)))

    # ------------------------------------------------------------------
    # Manufactured solution
    # ------------------------------------------------------------------
    @testset "matches the manufactured solution to round-off" begin
        a, u = manufactured_fields()
        result = Tarang.evaluate(Tarang.Divergence(Tarang.MultiplyOperator(a, u)), :g)
        @test result isa ScalarField
        @test grid_of(result) ≈ exact_flux atol=1e-11 rtol=1e-11

        # Guard the guard: not the zero field, and not `a*div(u)` with the u·∇a
        # term dropped (that piece is O(1) here, so dropping it fails the above).
        @test maximum(abs, grid_of(result)) > 1.0
    end

    @testset "both factor orders and both product spellings agree" begin
        # `MultiplyOperator` is what the equation-string parser builds…
        a1, u1 = manufactured_fields()
        coefficient_first = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(a1, u1)), :g)
        a2, u2 = manufactured_fields()
        vector_first = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(u2, a2)), :g)
        # …and the deferred `Multiply` future is what Julia's `*` builds.
        a3, u3 = manufactured_fields()
        deferred = Tarang.evaluate(Tarang.divergence(a3 * u3), :g)
        a4, u4 = manufactured_fields()
        deferred_reversed = Tarang.evaluate(Tarang.divergence(u4 * a4), :g)

        for candidate in (coefficient_first, vector_first, deferred, deferred_reversed)
            @test grid_of(candidate) ≈ exact_flux atol=1e-11 rtol=1e-11
        end

        # All four spellings must take the same expansion, so they agree exactly,
        # not merely to truncation error.
        @test grid_of(vector_first) == grid_of(coefficient_first)
        @test grid_of(deferred) ≈ grid_of(coefficient_first) atol=1e-13 rtol=1e-13
        @test grid_of(deferred_reversed) ≈ grid_of(coefficient_first) atol=1e-13 rtol=1e-13
    end

    @testset "equals the hand-expanded a*div(u) + dot(u, grad(a))" begin
        # The spelling a user would write by hand. The div() form must agree with
        # it to round-off, not merely to truncation error.
        a, u = manufactured_fields()
        div_form = grid_of(Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(a, u)), :g))

        # `dot` needs the gradient evaluated first: `dot(u, grad(a))` on the raw
        # Gradient operator is a separate, pre-existing gap.
        ah, uh = manufactured_fields()
        divergence_of_u = Tarang.evaluate(Tarang.Divergence(uh), :g)
        gradient_of_a = Tarang.evaluate(Tarang.Gradient(ah, coords), :g)
        hand_form = grid_of(Tarang.evaluate(
            ah * divergence_of_u + Tarang.dot(uh, gradient_of_a); force=true))

        @test div_form ≈ hand_form atol=1e-13 rtol=1e-13
        @test hand_form ≈ exact_flux atol=1e-11 rtol=1e-11

        # Written out component by component, it must land on the same place.
        component_form = grid_of(ah) .* grid_of(divergence_of_u) .+
                         grid_of(uh.components[1]) .* grid_of(gradient_of_a.components[1]) .+
                         grid_of(uh.components[2]) .* grid_of(gradient_of_a.components[2])
        @test div_form ≈ component_form atol=1e-13 rtol=1e-13
    end

    @testset "a constant coefficient reduces to a*div(u)" begin
        # ∂ᵢa ≡ 0, so ∇·(a u) collapses to a ∇·u exactly.
        _, u = manufactured_fields()
        scaled = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(2.5, u)), :g)
        _, u2 = manufactured_fields()
        divergence_of_u = Tarang.evaluate(Tarang.Divergence(u2), :g)
        @test grid_of(scaled) ≈ 2.5 .* grid_of(divergence_of_u) atol=1e-12 rtol=1e-12

        # A constant ScalarField coefficient must land on the same answer even
        # though it goes down the ∂ᵢa branch (∂ᵢa is computed, and is zero).
        constant_field = ScalarField(domain, "c")
        constant_field["g"] = fill(2.5, size(x))
        _, u3 = manufactured_fields()
        field_scaled = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(constant_field, u3)), :g)
        @test grid_of(field_scaled) ≈ grid_of(scaled) atol=1e-12 rtol=1e-12
    end

    @testset "an expression coefficient evaluates correctly" begin
        # `a` is not a field but an expression that evaluates to one.
        nu = ScalarField(domain, "nu")
        nu["g"] = @. 3 + cos(y)                       # a = ∂y(nu) = -sin y
        _, u = manufactured_fields()

        coefficient = Tarang.Differentiate(nu, coords["y"], 1)
        result = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(coefficient, u)), :g)

        # ∇a = (0, -cos y), so u·∇a = u₂ (-cos y) = -cos x sin y cos y.
        exact = @. (-sin(y)) * (2 * cos(x) * cos(y)) + (cos(x) * sin(y)) * (-cos(y))
        @test grid_of(result) ≈ exact atol=1e-11 rtol=1e-11
    end

    # ------------------------------------------------------------------
    # ScalarField * VectorField as a field operation in its own right
    # ------------------------------------------------------------------
    @testset "ScalarField*VectorField yields the scaled VectorField" begin
        a, u = manufactured_fields()
        product = Tarang.evaluate(Tarang.MultiplyOperator(a, u), :g)
        @test product isa VectorField
        @test grid_of(product.components[1]) ≈ (@. (2 + sin(x)) * sin(x) * cos(y)) atol=1e-13
        @test grid_of(product.components[2]) ≈ (@. (2 + sin(x)) * cos(x) * sin(y)) atol=1e-13

        # Commutative.
        a2, u2 = manufactured_fields()
        reversed = Tarang.evaluate(Tarang.MultiplyOperator(u2, a2), :g)
        @test grid_of(reversed.components[1]) == grid_of(product.components[1])
        @test grid_of(reversed.components[2]) == grid_of(product.components[2])

        # A :c request must come back in coefficient layout, like every other
        # arithmetic result, not flagged :g with stale coefficients.
        a3, u3 = manufactured_fields()
        in_coeff_space = Tarang.evaluate(Tarang.MultiplyOperator(a3, u3), :c)
        @test all(component.current_layout == :c for component in in_coeff_space.components)
        @test grid_of(in_coeff_space.components[1]) ≈ grid_of(product.components[1]) atol=1e-12
    end

    # ------------------------------------------------------------------
    # Through the equation parser: the path an actual solver takes
    # ------------------------------------------------------------------
    @testset "through the equation parser: ∂t(rho) = -div(rho*u)" begin
        _, u = manufactured_fields()
        rho = ScalarField(domain, "rho")
        rho["g"] = @. 2 + sin(x)

        problem = IVP([rho])
        add_parameters!(problem, u=u)
        add_equation!(problem, "∂t(rho) = -div(rho*u)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(F, :g)

        @test Array(get_grid_data(F)) ≈ -exact_flux atol=1e-11 rtol=1e-11
        @test maximum(abs, Array(get_grid_data(F))) > 1.0
    end

    # ------------------------------------------------------------------
    # Derivative-pool recycling
    # ------------------------------------------------------------------
    @testset "an expression coefficient survives the derivative buffer rotation" begin
        # `evaluate_differentiate` hands results out of a 16-slot rotating pool,
        # so an expression-valued coefficient lands in one of those slots and can
        # be recycled out from under a long accumulation. Its sibling
        # `div(a*grad(u))` issues 3 coords × 3 derivatives × 3 components = 27
        # further checkouts and DID come back wrong by 2.0 and 2.9 on O(1) data
        # before the coefficient was pinned into a checked-out field.
        #
        # This loop issues only 2 per coordinate (6 for the 3-component 3D field
        # below), so its margin against the 16-slot pool is wider and the pin is
        # currently insurance rather than a demonstrated fix — measured identical
        # with and without. What is asserted here is the invariant the pin exists
        # to hold, on the largest case the library supports: every component of a
        # 3-component 3D result is exact, and `a` itself is unchanged afterwards.
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
        nu["g"] = @. 3 + cos(y3)                       # a = ∂y(nu) = -sin y
        nu_before = Array(get_grid_data(nu))

        v = VectorField(domain3, "v")
        v.components[1]["g"] = @. sin(x3) * cos(y3)
        v.components[2]["g"] = @. cos(x3) * sin(z3)
        v.components[3]["g"] = @. sin(y3) * cos(z3)
        v_before = [Array(get_grid_data(c)) for c in v.components]

        coefficient = Tarang.Differentiate(nu, coords3["y"], 1)
        result = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(coefficient, v)), :g)

        # ∇·v = cos x cos y + 0 − sin y sin z; a = −sin y; ∇a = (0, −cos y, 0),
        # so v·∇a = v₂ (−cos y) = −cos x sin z cos y.
        divergence_of_v = @. cos(x3) * cos(y3) - sin(y3) * sin(z3)
        exact = @. (-sin(y3)) * divergence_of_v + (cos(x3) * sin(z3)) * (-cos(y3))
        @test grid_of(result) ≈ exact atol=1e-11 rtol=1e-11
        @test maximum(abs, grid_of(result)) > 1.0

        # Neither operand may be written through — a corrupted coefficient is
        # exactly what a recycled buffer looks like from the caller's side.
        @test Array(get_grid_data((ensure_layout!(nu, :g); nu))) ≈ nu_before atol=1e-14
        for (j, component) in enumerate(v.components)
            @test grid_of(component) ≈ v_before[j] atol=1e-14
        end

        # And it must agree with the hand-expanded spelling in 3D too.
        a_field = Tarang.evaluate_differentiate(
            Tarang.Differentiate(nu, coords3["y"], 1), :g)
        a_grid = Array(get_grid_data(a_field))
        divergence_field = Tarang.evaluate(Tarang.Divergence(v), :g)
        gradient_of_a = Tarang.evaluate(
            Tarang.Gradient(Tarang.evaluate_differentiate(
                Tarang.Differentiate(nu, coords3["y"], 1), :g), coords3), :g)
        hand = a_grid .* grid_of(divergence_field)
        for j in 1:3
            hand = hand .+ grid_of(v.components[j]) .* grid_of(gradient_of_a.components[j])
        end
        @test grid_of(result) ≈ hand atol=1e-12 rtol=1e-12
    end

    # ------------------------------------------------------------------
    # Unsupported cases must throw, naming what was unsupported
    # ------------------------------------------------------------------
    @testset "unsupported flux operands throw, naming what was unsupported" begin
        _, u = manufactured_fields()

        # A vector coefficient: `div(v*u)` is ambiguous (dot? cross?), so the
        # product itself must be rejected rather than silently picking one.
        _, v = manufactured_fields()
        vector_coefficient = Tarang.Divergence(Tarang.MultiplyOperator(v, u))
        @test_throws ArgumentError Tarang.evaluate(vector_coefficient, :g)
        message = try
            Tarang.evaluate(vector_coefficient, :g)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Cannot multiply", message)
        @test occursin("VectorField", message)
        @test occursin("dot", message)

        # A tensor coefficient has no product-rule expansion here.
        _, u2 = manufactured_fields()
        tensor_coefficient = Tarang.Divergence(
            Tarang.MultiplyOperator(TensorField(domain, "T"), u2))
        @test_throws ArgumentError Tarang.evaluate(tensor_coefficient, :g)
        tensor_message = try
            Tarang.evaluate(tensor_coefficient, :g)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("SCALAR coefficient", tensor_message)
        @test occursin("TensorField", tensor_message)

        # ∇· of a scalar is rank-nonsense; it must say so rather than return 0.
        a, _ = manufactured_fields()
        @test_throws ArgumentError Tarang.evaluate(Tarang.Divergence(a), :g)

        # And a failing flux term inside a solver RHS must abort the step, not
        # degrade the whole right-hand side to zero.
        @test_throws ArgumentError begin
            _, uu = manufactured_fields()
            rho = ScalarField(domain, "rho")
            rho["g"] = @. 2 + sin(x)
            problem = IVP([rho])
            add_parameters!(problem, u=uu)
            add_equation!(problem, "∂t(rho) = -div(u*u)")
            solver = InitialValueSolver(problem, RK222(); dt=1e-3)
            Tarang.evaluate_rhs(solver, solver.state, 0.0)
        end
    end

    # ------------------------------------------------------------------
    # The sibling case must be untouched
    # ------------------------------------------------------------------
    @testset "div(a*grad(u)) is unaffected" begin
        # The Gradient split is checked before the flux split, so a gradient
        # operand must still take the a∇²u + ∇a·∇u path.
        a, _ = manufactured_fields()
        w = ScalarField(domain, "w")
        w["g"] = @. sin(x) * cos(y)

        result = Tarang.evaluate(
            Tarang.Divergence(Tarang.MultiplyOperator(a, Tarang.Gradient(w, coords))), :g)
        exact_diffusion = @. cos(y) * (cos(x)^2 - 2 * sin(x)^2 - 4 * sin(x))
        @test grid_of(result) ≈ exact_diffusion atol=1e-10 rtol=1e-10

        # A VECTOR coefficient times a Gradient must still report the non-scalar
        # coefficient, not be re-read as a flux whose vector factor is `v`.
        _, v = manufactured_fields()
        message = try
            Tarang.evaluate(
                Tarang.Divergence(Tarang.MultiplyOperator(v, Tarang.Gradient(w, coords))), :g)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("SCALAR coefficient", message)
        @test occursin("grad", message)
    end
end
