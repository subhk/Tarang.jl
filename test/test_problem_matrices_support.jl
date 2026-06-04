"""
Tests for src/core/problems/problem_matrices/problem_matrices_support.jl — the
equation-inclusion / validity predicates and matrix-block helpers used during
matrix assembly. These are pure Dict + size logic with obvious oracles.
"""

using Test
using Tarang
using SparseArrays
using LinearAlgebra
import Tarang: check_equation_condition, is_equation_valid,
               set_equation_condition!, enable_equation!, disable_equation!,
               set_valid_modes!, exclude_k_zero!, get_matrix_expression,
               is_zero_expression, _identity_block, _zero_block, ZeroOperator

# A minimal equation-data dict that PASSES check_equation_condition (has matrix
# content + positive size + no disabling flags).
included_eq() = Dict{String,Any}("L" => 1.0, "equation_size" => 4, "equation_index" => 1)

@testset "problem_matrices_support" begin
    @testset "is_zero_expression" begin
        @test is_zero_expression(ZeroOperator()) == true
        @test is_zero_expression(nothing)        == true
        @test is_zero_expression(1.0)            == false
        @test is_zero_expression("Δ(u)")         == false
    end

    @testset "check_equation_condition: inclusion + exclusion paths" begin
        @test check_equation_condition(included_eq()) == true

        # explicitly disabled
        d = included_eq(); d["enabled"] = false
        @test check_equation_condition(d) == false
        d["enabled"] = true
        @test check_equation_condition(d) == true

        # Bool condition false / true
        d = included_eq(); d["condition"] = false
        @test check_equation_condition(d) == false
        d["condition"] = true
        @test check_equation_condition(d) == true

        # Function condition
        d = included_eq(); d["condition"] = ed -> false
        @test check_equation_condition(d) == false
        d["condition"] = ed -> true
        @test check_equation_condition(d) == true

        # flagged invalid
        d = included_eq(); d["is_invalid"] = true
        @test check_equation_condition(d) == false

        # no matrix content (M/L/F all zero/missing)
        d = Dict{String,Any}("equation_size" => 4)
        @test check_equation_condition(d) == false
        d = Dict{String,Any}("L" => ZeroOperator(), "equation_size" => 4)
        @test check_equation_condition(d) == false

        # non-positive size
        d = included_eq(); d["equation_size"] = 0
        @test check_equation_condition(d) == false

        # valid_modes / current_mode
        d = included_eq(); d["valid_modes"] = Set([1, 2]); d["current_mode"] = 3
        @test check_equation_condition(d) == false
        d["current_mode"] = 2
        @test check_equation_condition(d) == true

        # exclude_k_zero with scalar / tuple wavenumber
        d = included_eq(); d["exclude_k_zero"] = true; d["wavenumber"] = 0
        @test check_equation_condition(d) == false
        d["wavenumber"] = (0, 0)
        @test check_equation_condition(d) == false
        d["wavenumber"] = (1, 0)
        @test check_equation_condition(d) == true
    end

    @testset "equation-condition mutators" begin
        d = Dict{String,Any}()
        enable_equation!(d);            @test d["enabled"] == true
        disable_equation!(d);           @test d["enabled"] == false
        set_equation_condition!(d, true); @test d["condition"] == true
        set_valid_modes!(d, [1, 2, 3]); @test d["valid_modes"] == Set([1, 2, 3])
        set_valid_modes!(d, 4:6);       @test d["valid_modes"] == Set([4, 5, 6])
        exclude_k_zero!(d);             @test d["exclude_k_zero"] == true
        exclude_k_zero!(d, false);      @test d["exclude_k_zero"] == false
    end

    @testset "is_equation_valid" begin
        @test is_equation_valid(Dict{String,Any}())[1] == false              # no equation_string
        d = Dict{String,Any}("equation_string" => "Δ(u)=0")
        @test is_equation_valid(d)[1] == false                                # missing lhs
        d["lhs"] = nothing
        @test is_equation_valid(d) == (true, nothing)                         # lhs present (nothing) → ok
        d2 = Dict{String,Any}("equation_string" => "x", "lhs" => nothing, "parse_error" => "boom")
        ok, msg = is_equation_valid(d2)
        @test ok == false && occursin("Parse error", msg)
    end

    @testset "get_matrix_expression" begin
        d = Dict{String,Any}("L" => "Δ(u)", "M" => ZeroOperator())
        @test get_matrix_expression(d, "L") == "Δ(u)"
        @test get_matrix_expression(d, "M") isa ZeroOperator
        @test get_matrix_expression(d, "F") === nothing
    end

    @testset "_identity_block / _zero_block" begin
        I3 = _identity_block(3, 3)
        @test Matrix(I3) ≈ Matrix{ComplexF64}(I, 3, 3)
        S = _identity_block(3, 3; scale=2.5)
        @test Matrix(S) ≈ 2.5 .* Matrix{ComplexF64}(I, 3, 3)
        # rectangular: min(eqn,var) entries on the diagonal
        R = _identity_block(3, 5; scale=2.0)
        @test size(R) == (3, 5)
        @test nnz(R) == 3 && all(R[i, i] ≈ 2.0 for i in 1:3)
        # degenerate dims → zero block
        @test nnz(_identity_block(0, 4)) == 0
        @test size(_zero_block(2, 3)) == (2, 3) && nnz(_zero_block(2, 3)) == 0
    end
end
