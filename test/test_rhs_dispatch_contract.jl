# Guard: RHS expression evaluation dispatches on node type, and keeps dispatching.
#
# `evaluate_solver_expression` used to be a single `if expr isa … elseif …` chain of
# fifteen branches, with a generic `elseif expr isa Operator` at the bottom. Every
# concrete operator is a subtype of `Operator`, so that last branch shadowed all the
# specific ones and only stayed correct because it was written last. Two things
# could go wrong silently:
#
#   * moving the generic branch up, or adding a branch below it, disables the
#     branches it shadows — the expression still evaluates, just through the wrong
#     handler, which is a wrong value rather than an error;
#   * an unsupported operand pairing (say ScalarField + TensorField) fell off the
#     end of an inner chain into whichever `else` came next.
#
# Rewritten as methods, Julia resolves by specificity rather than by source order
# and an unmatched pairing hits a fallback that raises. This file pins those two
# properties so a future edit cannot quietly restore the chain semantics.
#
# It deliberately asserts on the METHOD TABLE as well as on behaviour: the ordering
# hazard is a property of how the code is written, and a purely behavioural test
# passes just as happily against a correctly-ordered chain.

using Test
using Tarang

@testset "RHS evaluation dispatches on node type" begin

    @testset "Specific operator methods beat the generic ::Operator method" begin
        # The whole ordering hazard in one assertion: for each node type that has
        # its own handler, the method Julia selects must be the specific one, not
        # the generic Operator fallback. This is what source order used to decide.
        generic = which(Tarang.evaluate_solver_expression,
                        Tuple{Tarang.Operator, Any})
        for T in (Tarang.AddOperator, Tarang.SubtractOperator, Tarang.MultiplyOperator,
                  Tarang.DivideOperator, Tarang.PowerOperator, Tarang.NegateOperator,
                  Tarang.IndexOperator, Tarang.ZeroOperator, Tarang.ConstantOperator,
                  Tarang.ArrayOperator, Tarang.UnknownOperator)
            @test T <: Tarang.Operator                       # the shadowing premise
            selected = which(Tarang.evaluate_solver_expression, Tuple{T, Any})
            @test selected !== generic
        end
    end

    @testset "The generic Operator method still exists and is reachable" begin
        # 45 of the 56 Operator subtypes are served by it on purpose. Deleting it in
        # a future cleanup would turn all of them into a fallback error, so pin that
        # it is present and is genuinely typed on Operator.
        generic = which(Tarang.evaluate_solver_expression, Tuple{Tarang.Operator, Any})
        @test generic.sig.parameters[2] === Tarang.Operator
    end

    @testset "A node type with no method raises instead of guessing" begin
        # The Any fallback. A String is not an expression node; the old chain ran off
        # the end into the same error, and that behaviour must survive the rewrite.
        @test_throws ErrorException Tarang.evaluate_solver_expression("not a node", [])
        @test_throws ArgumentError Tarang.evaluate_solver_expression(nothing, [])
    end

    @testset "Unsupported operand pairings raise, not fall through" begin
        # Each of these combinations has no method, so the `_rhs_*` fallback fires.
        # Under the old inner chains an unmatched pair reached the trailing `else`,
        # which was correct only as long as every `else` was written correctly.
        @test_throws ArgumentError Tarang._rhs_add("a", 1)
        @test_throws ArgumentError Tarang._rhs_subtract(1, "b")
        @test_throws ArgumentError Tarang._rhs_multiply(nothing, nothing)
        @test_throws ArgumentError Tarang._rhs_negate("c")
        @test_throws ArgumentError Tarang._rhs_index("not indexable", (1,), :g)

        # ...while the supported pairings still compute.
        @test Tarang._rhs_add(2, 3) == 5
        @test Tarang._rhs_subtract(2, 3) == -1
        @test Tarang._rhs_multiply(2, 3) == 6
        @test Tarang._rhs_negate(2) == -2
        @test Tarang._rhs_index([10, 20, 30], (2,), :g) == 20
    end

    @testset "Constant folding dispatches too, and still declines cleanly" begin
        # coerce_constant_value folds only when every leaf is a number. Its `::Any`
        # method returns the node unchanged, which — unlike the evaluation fallback
        # — is a real answer rather than an error, so pin both directions.
        two = Tarang.ConstantOperator(2.0)
        four = Tarang.ConstantOperator(4.0)
        @test Tarang.coerce_constant_value(Tarang.DivideOperator(four, two)) ≈ 2.0
        @test Tarang.coerce_constant_value(Tarang.MultiplyOperator(four, two)) ≈ 8.0
        @test Tarang.coerce_constant_value(Tarang.AddOperator(four, two)) ≈ 6.0
        @test Tarang.coerce_constant_value(Tarang.SubtractOperator(four, two)) ≈ 2.0
        @test Tarang.coerce_constant_value(Tarang.PowerOperator(four, two)) ≈ 16.0
        @test Tarang.coerce_constant_value(Tarang.NegateOperator(four)) ≈ -4.0
        # Nested, to prove the recursion survives the split into methods. This is the
        # `u(z=Lz/2)` shape that used to degrade to UnknownOperator and drop the BC.
        nested = Tarang.DivideOperator(Tarang.AddOperator(four, two), two)
        @test Tarang.coerce_constant_value(nested) ≈ 3.0
        # Not foldable: returned unchanged, not zero and not an error.
        unfoldable = Tarang.UnknownOperator("u")
        @test Tarang.coerce_constant_value(unfoldable) === unfoldable
    end
end
