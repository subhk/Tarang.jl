# Characterization tests for `_left_permutation_indices`
# (src/core/subsystems/subproblem_permutations.jl).
#
# The index hierarchy L0[equation][component][coefficient] is RAGGED: equations differ in
# component count and coefficient count, but the regrouping loops run over the MAXIMUM extent
# of each level. Missing entries were detected by catching the resulting `BoundsError` with a
# bare `catch ... continue`, i.e. exceptions used as ragged-array iteration.
#
# That has two costs: any non-BoundsError is swallowed identically (an index silently vanishes
# from the permutation, so a row is never assigned), and a throw/catch pair is paid for every
# ragged hole. These tests pin the exact output so the loops can be rewritten with explicit
# bounds checks and proven equivalent.
#
# Uniquely-prefixed names (spperm_*) — the full suite shares the Main namespace.

using Test
using Tarang

struct SppermBase
    matrix_coupling::Vector{Bool}
end
struct SppermSolver
    problem::Tarang.Problem
    base::SppermBase
end

"""Build a throwaway 1D Subproblem. `_left_permutation_indices` does not read it, but the
signature requires one."""
function spperm_make_sp()
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=Float64)
    xb = ComplexFourier(coords["x"]; size=4, bounds=(0.0, 2π))
    field = ScalarField(dist, "u", (xb,), ComplexF64)
    problem = IVP([field])
    solver = SppermSolver(problem, SppermBase([true]))
    subsys = Tarang.Subsystem(solver, (nothing,))
    return Tarang.Subproblem(solver, (subsys,), (nothing,))
end

@testset "subproblem left permutation over a ragged index hierarchy" begin
    sp = spperm_make_sp()

    @testset "ragged in coefficient count (4-coeff bulk + 2-coeff boundary rows)" begin
        # L0 = [[[0,1,2,3]], [[4,5]]] — equation 2 has no coefficients 3 or 4.
        equations = [
            Dict{String, Any}("tensorsig" => (), "domain_dim" => 2),
            Dict{String, Any}("tensorsig" => (), "domain_dim" => 1),
        ]
        sizes = [4, 2]

        @test Tarang._left_permutation_indices(sp, equations, sizes, true, true) ==
              [5, 6, 1, 2, 3, 4]
        @test Tarang._left_permutation_indices(sp, equations, sizes, false, true) ==
              [1, 2, 3, 4, 5, 6]
        @test Tarang._left_permutation_indices(sp, equations, sizes, true, false) ==
              [5, 6, 1, 2, 3, 4]
    end

    @testset "ragged in component count (2-component vector eq + scalar eq)" begin
        # L0 = [[[0,1],[2,3]], [[4,5]]] — equation 2 has no second component.
        coords2d = CartesianCoordinates("x", "y")
        equations = [
            Dict{String, Any}("tensorsig" => (coords2d,), "domain_dim" => 2),
            Dict{String, Any}("tensorsig" => (), "domain_dim" => 1),
        ]
        sizes = [4, 2]

        @test Tarang._left_permutation_indices(sp, equations, sizes, true, true) ==
              [5, 6, 1, 3, 2, 4]
        @test Tarang._left_permutation_indices(sp, equations, sizes, false, true) ==
              [1, 3, 2, 4, 5, 6]
    end

    @testset "every index appears exactly once — no row is dropped" begin
        coords2d = CartesianCoordinates("x", "y")
        equations = [
            Dict{String, Any}("tensorsig" => (coords2d,), "domain_dim" => 2),
            Dict{String, Any}("tensorsig" => (), "domain_dim" => 1),
            Dict{String, Any}("tensorsig" => (), "domain_dim" => 0),
        ]
        sizes = [6, 3, 0]

        for bc_top in (true, false), interleave in (true, false)
            idx = Tarang._left_permutation_indices(sp, equations, sizes, bc_top, interleave)
            @test sort(idx) == collect(1:sum(sizes))
        end
    end
end
