"""
Tests for subproblem_modes.jl — per-subproblem equation-condition evaluation and
valid-mode masks.

`check_condition` decides whether an equation applies to a given subproblem (one
separable Fourier mode), evaluating a condition string like "nx != 0" or
"kx == 0 && ky == 0" against the subproblem's group dictionary. The compound
(&&/||) path was BROKEN (it recursed into `check_condition` passing a String where
a Subproblem was expected → MethodError); fixed 2026-06-04 by extracting the pure
string evaluator `_eval_condition_str`. These tests pin both.

Uniquely-prefixed names (spm_*) — the full suite shares the Main namespace.
"""

using Test
using Tarang
import Tarang: check_condition, valid_modes, _eval_condition_str

@testset "subproblem_modes" begin
    @testset "_eval_condition_str: simple comparisons" begin
        gd = Dict{String,Any}("nx" => 0, "ny" => 2)
        @test _eval_condition_str("nx != 0", gd) == false
        @test _eval_condition_str("ny != 0", gd) == true
        @test _eval_condition_str("nx == 0", gd) == true
        @test _eval_condition_str("ny == 0", gd) == false
        # Symbol-keyed group dicts are also supported
        gds = Dict{Any,Any}(:nx => 3)
        @test _eval_condition_str("nx == 3", gds) == true
        @test _eval_condition_str("nx != 3", gds) == false
    end

    @testset "_eval_condition_str: compound &&/|| (was broken)" begin
        gd = Dict{String,Any}("nx" => 0, "ny" => 2)
        @test _eval_condition_str("nx == 0 && ny == 2", gd) == true
        @test _eval_condition_str("nx == 0 && ny == 0", gd) == false
        @test _eval_condition_str("nx == 5 || ny == 2", gd) == true
        @test _eval_condition_str("nx == 5 || ny == 9", gd) == false
        # nested-ish: multiple && parts
        @test _eval_condition_str("nx == 0 && ny == 2 && nx != 1", gd) == true
    end

    @testset "_eval_condition_str: unparseable defaults to true" begin
        gd = Dict{String,Any}("nx" => 0)
        @test _eval_condition_str("totally bogus", gd) == true   # warns, includes eqn
        @test _eval_condition_str("nz == 0", gd) == true         # key missing → true
    end

    # Build a real per-mode subproblem (2D Fourier×Chebyshev LBVP) to exercise
    # check_condition / valid_modes against an actual group dictionary.
    function spm_subproblem()
        coords = CartesianCoordinates("x", "z")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
        dom = Domain(dist, (xb, zb))
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (xb,), Float64)
        tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        lb2  = derivative_basis(zb, 2)
        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=1.0, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=1.0) = 0")
        solver = Tarang.BoundaryValueSolver(prob)
        return solver.subproblems[1], u
    end

    @testset "check_condition: literal / Bool branches" begin
        sp, _ = spm_subproblem()
        @test check_condition(sp, Dict{String,Any}("condition" => "true"))  == true
        @test check_condition(sp, Dict{String,Any}("condition" => "false")) == false
        @test check_condition(sp, Dict{String,Any}("condition" => true))    == true
        @test check_condition(sp, Dict{String,Any}("condition" => false))   == false
        @test check_condition(sp, Dict{String,Any}())                       == true  # default
    end

    @testset "check_condition: string condition against group_dict" begin
        sp, _ = spm_subproblem()
        nx = sp.group_dict["nx"]
        @test check_condition(sp, Dict{String,Any}("condition" => "nx == $nx"))  == true
        @test check_condition(sp, Dict{String,Any}("condition" => "nx != $nx"))  == false
        @test check_condition(sp, Dict{String,Any}("condition" => "nx == $(nx + 1)")) == false
    end

    @testset "valid_modes: nothing → all valid" begin
        sp, u = spm_subproblem()
        vm = valid_modes(sp, u, nothing)
        @test eltype(vm) == Bool
        @test all(vm)
        @test length(vm) == Tarang.field_size(sp, u)
    end
end
