# Guard: boundary-condition callbacks used to have their ARITY PROBED BY EXCEPTION.
#
# `_evaluate_function_expression` / `_evaluate_space_function_expression` called the user's
# function inside a bare `try`, and on ANY exception moved on to a shorter argument list.
# Two consequences, both silent:
#
#   A. A genuine error raised INSIDE a correctly-shaped callback (a typo, a BoundsError, a
#      unit bug) was indistinguishable from "wrong number of arguments". A time-dependent BC
#      whose body threw would quietly degrade to a lower-arity call, or to `func()`.
#
#   B. The terminal fallback of the space variant was `return func` — the Function OBJECT
#      itself was handed back as if it were a boundary value.
#
# Both are the silent-wrong-value class: no error, plausible-looking result, wrong physics.
# A structural arity mismatch (a MethodError raised by dispatch ON THIS FUNCTION) may still
# fall through to the next candidate; anything else must propagate.

using Test
using Tarang

const _BCA = Tarang

@testset "BC callback arity is not probed by swallowing errors" begin

    coords_1d = Dict{String, Any}("x" => [0.0, 0.5, 1.0])
    coords_2d = Dict{String, Any}("x" => [0.0, 0.5], "y" => [1.0, 2.0])

    @testset "space callback: error inside the body propagates" begin
        boom(x) = error("boom inside user BC")
        @test_throws ErrorException _BCA._evaluate_space_function_expression(boom, coords_1d)
    end

    @testset "space callback: never returns the Function object as a value" begin
        # Accepts no supported signature at all: must raise, not hand back `boom`.
        boom(::Int, ::Int, ::Int, ::Int, ::Int) = 0.0
        result = try
            _BCA._evaluate_space_function_expression(boom, coords_1d)
        catch e
            e
        end
        @test !isa(result, Function)
        @test isa(result, Exception)
    end

    @testset "time callback: error inside the body propagates" begin
        boom(t, x) = error("boom inside user BC")
        @test_throws ErrorException _BCA._evaluate_function_expression(boom, 0.25, coords_1d)
    end

    @testset "time callback: unsupported signature raises a clear error" begin
        boom(::Int, ::Int, ::Int, ::Int, ::Int) = 0.0
        @test_throws Exception _BCA._evaluate_function_expression(boom, 0.25, coords_1d)
    end

    # Regression guards: the shapes that already worked must keep working.
    @testset "supported signatures still evaluate" begin
        @test _BCA._evaluate_space_function_expression(x -> 2 .* x, coords_1d) == [0.0, 1.0, 2.0]
        @test _BCA._evaluate_space_function_expression((x, y) -> x .+ y, coords_2d) == [1.0, 2.5]
        @test _BCA._evaluate_function_expression((t, x) -> t .* x, 2.0, coords_1d) == [0.0, 1.0, 2.0]
        @test _BCA._evaluate_function_expression(t -> 3.0 * t, 2.0, Dict{String, Any}()) == 6.0
    end
end
