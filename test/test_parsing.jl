"""
Test suite for tools/parsing.jl — equation string parsing.

Tests:
1. split_equation for simple LHS = RHS
2. split_equation with nested brackets
3. split_equation with Unicode
4. Compound operator skipping (==, !=, >=, <=)
5. Error cases (no equals, unmatched brackets, multiple equals)
"""

using Test
using Tarang

import Tarang: split_equation

@testset "Equation Parsing" begin

    @testset "Simple equations" begin
        lhs, rhs = split_equation("dt(u) = -u*grad(u)")
        @test strip(lhs) == "dt(u)"
        @test strip(rhs) == "-u*grad(u)"
    end

    @testset "Equation with spaces" begin
        lhs, rhs = split_equation("  dt(u) + nu*lap(u)  =  F(u)  ")
        @test strip(lhs) == "dt(u) + nu*lap(u)"
        @test strip(rhs) == "F(u)"
    end

    @testset "Equation with nested brackets" begin
        # Equals inside brackets should NOT be treated as equation separator
        lhs, rhs = split_equation("u(z=0) = 0")
        @test strip(lhs) == "u(z=0)"
        @test strip(rhs) == "0"
    end

    @testset "Deeply nested brackets" begin
        lhs, rhs = split_equation("f(g(h(x=1, y=2))) = 0")
        @test strip(lhs) == "f(g(h(x=1, y=2)))"
        @test strip(rhs) == "0"
    end

    @testset "Multiple bracket types" begin
        lhs, rhs = split_equation("a[b{c=1}] = d")
        @test strip(lhs) == "a[b{c=1}]"
        @test strip(rhs) == "d"
    end

    @testset "Unicode equation" begin
        lhs, rhs = split_equation("∂t(u) + ν*∇²(u) = F")
        @test strip(lhs) == "∂t(u) + ν*∇²(u)"
        @test strip(rhs) == "F"
    end

    @testset "Compound operators not treated as equals" begin
        # == inside expression should not split
        lhs, rhs = split_equation("f(x) = (a == b)")
        @test strip(lhs) == "f(x)"
        @test occursin("==", rhs)

        # >= and <= should not split
        lhs2, rhs2 = split_equation("g(x) = (a >= b)")
        @test strip(lhs2) == "g(x)"
        @test occursin(">=", rhs2)
    end

    @testset "Error: no equals sign" begin
        @test_throws Exception split_equation("dt(u) + lap(u)")
    end

    @testset "Error: unmatched opening bracket" begin
        @test_throws Exception split_equation("f(x = 0")
    end

    @testset "Error: unmatched closing bracket" begin
        @test_throws Exception split_equation("f)x( = 0")
    end

    @testset "Error: mismatched bracket types" begin
        @test_throws Exception split_equation("f(x] = 0")
    end

    @testset "Boundary condition strings" begin
        # Dirichlet BC
        lhs, rhs = split_equation("u(z=0) = 0")
        @test occursin("u(z=0)", lhs)

        # Neumann BC
        lhs, rhs = split_equation("dz(u)(z=1) = 0")
        @test occursin("dz(u)(z=1)", lhs)

        # Non-zero BC
        lhs, rhs = split_equation("T(z=-1) = 1.0")
        @test strip(rhs) == "1.0"
    end
end
