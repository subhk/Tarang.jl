"""
The batched mass solve.

`M_min` is rank-deficient in every tau/BC formulation, so the per-mode path
solves `M x = b` with a sparse least-squares (`SPQRSolver`). Measured, though,
`M_min` is a 0/1 PARTIAL PERMUTATION — at most one nonzero per row and per
column, every value exactly 1, the tau columns empty. For such a matrix the
minimum-norm least-squares solution is `x = M' b` for ANY `b`, which is one
kernel rather than `nmodes` sparse solves.

This file pins the structural check that decides whether that shortcut is
legal, because applying it to a genuine mass matrix would be silently wrong
rather than an error.
"""

using Test
using Tarang
using SparseArrays
using LinearAlgebra

@testset "mass_selection_plan" begin
    @testset "accepts a 0/1 partial permutation" begin
        # cols 1,2 empty; col 3 -> row 1, col 4 -> row 4, col 5 -> row 5
        M = sparse([1, 4, 5], [3, 4, 5], ComplexF64[1, 1, 1], 5, 5)
        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing
        src, scale = plan
        @test src == [0, 0, 1, 4, 5]
        @test scale == ComplexF64[1, 1, 1, 1, 1]
    end

    @testset "accepts a SCALED partial permutation" begin
        M = sparse([1, 4], [3, 4], ComplexF64[2.0, 0.5], 4, 4)
        plan = Tarang.mass_selection_plan(M)
        @test plan !== nothing
        src, scale = plan
        @test src == [0, 0, 1, 4]
        @test scale[3] == 2.0
        @test scale[4] == 0.5
    end

    @testset "rejects two nonzeros in one column" begin
        M = sparse([1, 2], [1, 1], ComplexF64[1, 1], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects two nonzeros in one row" begin
        M = sparse([1, 1], [1, 2], ComplexF64[1, 1], 3, 3)
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "rejects a genuine mass matrix" begin
        # tridiagonal — the shape a non-identity basis normalisation produces
        M = spdiagm(-1 => ComplexF64[1, 1], 0 => ComplexF64[4, 4, 4],
                    1 => ComplexF64[1, 1])
        @test Tarang.mass_selection_plan(M) === nothing
    end

    @testset "an explicitly stored zero is not a nonzero" begin
        # A stored zero must not be treated as a mapping, or the plan would
        # divide by it.
        M = SparseMatrixCSC(3, 3, [1, 2, 2, 2], [1], ComplexF64[0.0])
        @test Tarang.mass_selection_plan(M) === nothing
    end
end
