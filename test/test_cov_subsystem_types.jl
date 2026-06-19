"""
Serial-CPU coverage tests for src/core/subsystems/subsystem_types.jl.

These pin the behavior of the Subsystem constructor and its supporting
helpers — variable/equation range computation, matrix-group collapsing,
group normalization, separable-dimension sizing, mode-group generation, and
the operator-coupling analyzer — using small LBVP/IVP problems and direct
helper calls. Every assertion checks a real invariant (counts, ranges,
shapes, error conditions), not an arbitrary number.

Mock solver/base mirror the established idiom in test_subsystems.jl so that a
real `Subsystem` can be built cheaply without a full solver build.
"""

using Test
using Tarang
using LinearAlgebra

# ---------------------------------------------------------------------------
# Mock solver/base: a `.base.matrix_coupling` field is all the Subsystem
# constructor and get_matrix_coupling consult on the serial path.
# ---------------------------------------------------------------------------
struct CstBase
    matrix_coupling::Vector{Bool}
end
struct CstSolver
    problem::Tarang.Problem
    base::CstBase
end
# A solver with NO `base` field, to exercise the `hasfield(...,:base)==false`
# fallback paths in the Subsystem constructor and get_matrix_coupling.
struct CstSolverNoBase
    problem::Tarang.Problem
end

@testset "subsystem_types.jl coverage" begin

    # -----------------------------------------------------------------------
    @testset "scalar_components dispatch" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))

        s = ScalarField(domain, "s")
        v = VectorField(domain, "v")   # 2 components for 2D coords

        # Scalar -> singleton vector containing itself
        sc = scalar_components(s)
        @test sc == [s]
        @test length(sc) == 1

        # Vector -> its components list
        vc = scalar_components(v)
        @test vc === v.components
        @test length(vc) == 2
        @test all(c -> c isa Tarang.ScalarField, vc)
    end

    # -----------------------------------------------------------------------
    @testset "scalar_field_dofs counts coefficient DOFs" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        f = ScalarField(dist, "f", (xb,), Float64)
        ensure_layout!(f, :c)
        # RealFourier(8) half-spectrum coeff length = 8/2 + 1 = 5
        @test get_coeff_data(f) !== nothing
        @test scalar_field_dofs(f) == length(get_coeff_data(f))
        @test scalar_field_dofs(f) == 5
    end

    @testset "scalar_field_dofs of a 0-dim tau field" begin
        # A 0-dim tau field with NO bases carries an empty coefficient array
        # (length 0), so its spectral DOF count is 0.
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        tau = ScalarField(dist, "tau", (), Float64)
        cd = get_coeff_data(tau)
        if cd !== nothing
            @test scalar_field_dofs(tau) == length(cd)
            @test scalar_field_dofs(tau) == 0
        else
            # No eagerly-allocated data -> basis-product fallback; empty product = 1.
            @test scalar_field_dofs(tau) == 1
        end
    end

    # -----------------------------------------------------------------------
    @testset "infer_problem_dtype / infer_problem_dist" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])

        # dtype is read off the first scalar component
        @test infer_problem_dtype(prob) == u.dtype

        # With a domain set, dist comes straight from the domain
        @test infer_problem_dist(prob) === domain.dist

        # Without a domain, dist comes from the first component's dist.
        prob_nodom = LBVP([u])
        prob_nodom.domain = nothing
        @test infer_problem_dist(prob_nodom) === u.dist
    end

    @testset "infer_problem_dtype empty-variables fallback" begin
        # No variables -> the loops never return -> ComplexF64 default.
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        prob = LBVP([u])
        empty!(prob.variables)
        @test infer_problem_dtype(prob) == ComplexF64
    end

    @testset "infer_problem_dist throws when no distributor available" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        prob = LBVP([u])
        prob.domain = nothing
        empty!(prob.variables)   # no domain, no variables -> ArgumentError
        @test_throws ArgumentError infer_problem_dist(prob)
    end

    # -----------------------------------------------------------------------
    @testset "compute_variable_ranges packs DOFs contiguously" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))  # 5 coeffs
        a = ScalarField(dist, "a", (xb,), Float64)
        b = ScalarField(dist, "b", (xb,), Float64)
        ensure_layout!(a, :c); ensure_layout!(b, :c)
        prob = LBVP([a, b])

        scalar_ranges, variable_ranges, total = compute_variable_ranges(prob)

        na = scalar_field_dofs(a)
        nb = scalar_field_dofs(b)
        @test total == na + nb
        @test scalar_ranges[a] == 1:na
        @test scalar_ranges[b] == (na + 1):(na + nb)
        # Per-variable range spans its single scalar component.
        @test variable_ranges[a] == 1:na
        @test variable_ranges[b] == (na + 1):(na + nb)
        # Ranges must tile [1, total] with no gaps/overlap.
        @test last(scalar_ranges[a]) + 1 == first(scalar_ranges[b])
    end

    @testset "compute_variable_ranges: zero-DOF variable yields empty range" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        a = ScalarField(dist, "a", (xb,), Float64)
        # 0-dof component path is only taken if dofs==0; build a vector field
        # with an empty component set to force the empty-range branch.
        empty_vec = VectorField(Domain(dist, (xb,)), "ev")
        empty!(empty_vec.components)   # no components -> no DOFs contributed
        prob = LBVP([empty_vec, a])
        scalar_ranges, variable_ranges, total = compute_variable_ranges(prob)
        na = scalar_field_dofs(a)
        @test total == na
        # The empty variable gets a degenerate (empty) range start:(start-1).
        @test isempty(variable_ranges[empty_vec])
        @test scalar_ranges[a] == 1:na
    end

    # -----------------------------------------------------------------------
    @testset "compute_equation_ranges from equation_data" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        prob = LBVP([u])
        # Populate equation_data with explicit equation sizes.
        push!(prob.equation_data, Dict{String,Any}("equation_size" => 3))
        push!(prob.equation_data, Dict{String,Any}("equation_size" => 4))
        eq_ranges, total = compute_equation_ranges(prob)
        @test total == 7
        @test eq_ranges[1] == 1:3
        @test eq_ranges[2] == 4:7
    end

    @testset "compute_equation_ranges from empty equation_data uses equations" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        prob = LBVP([u])
        @test isempty(prob.equation_data)
        # Two raw equation strings, no equation_data -> the `equations` branch.
        push!(prob.equations, "u = 0")
        push!(prob.equations, "dx(u) = 0")
        eq_ranges, total = compute_equation_ranges(prob)
        # compute_field_size(Dict()) == 0, so each range is empty and total==0,
        # but a range entry must exist for every equation index.
        @test total == 0
        @test haskey(eq_ranges, 1)
        @test haskey(eq_ranges, 2)
        @test isempty(eq_ranges[1])
        @test isempty(eq_ranges[2])
    end

    # -----------------------------------------------------------------------
    @testset "compute_matrix_group: dependent axes preserve the group" begin
        # When every axis is dependent (the live Subsystem path marks them so),
        # the matrix group equals the input group unchanged.
        grp = (0, nothing)
        dep = [true, true]
        coupling = [false, true]
        defaults = (1, 1)
        @test compute_matrix_group(grp, dep, coupling, defaults) == grp

        grp2 = (5, nothing)
        @test compute_matrix_group(grp2, dep, coupling, defaults) == grp2
    end

    @testset "compute_matrix_group: non-dependent nonzero axis collapses to default" begin
        # If an axis is neither dependent nor coupled, a nonzero group index is
        # remapped to the default_nonconst_groups entry for that axis. This is
        # the historically-buggy collapse path; exercise it directly.
        grp = (7,)               # nonzero, separable, non-dependent
        dep = [false]            # not matrix-dependent
        coupling = [false]       # not coupled
        defaults = (1,)
        @test compute_matrix_group(grp, dep, coupling, defaults) == (1,)

        # A zero group index is left untouched (the `!= 0` guard).
        grp0 = (0,)
        @test compute_matrix_group(grp0, dep, coupling, defaults) == (0,)

        # If the axis index exceeds default_nonconst_groups length, the entry is
        # left unchanged (the inner `i <= length(defaults)` guard fails).
        grp_long = (7, 9)
        dep2 = [false, false]
        coupling2 = [false, false]
        defaults_short = (1,)    # only one default
        out = compute_matrix_group(grp_long, dep2, coupling2, defaults_short)
        @test out[1] == 1        # remapped
        @test out[2] == 9        # untouched (no default for axis 2)
    end

    # -----------------------------------------------------------------------
    @testset "_normalize_subsystem_group" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)

        # The sentinel global group expands to an all-`nothing` tuple of dim.
        g_global = Tarang._normalize_subsystem_group(dist, (:global,))
        @test g_global == (nothing, nothing)
        @test length(g_global) == dist.dim

        # A correctly-sized explicit group passes through unchanged.
        g_ok = Tarang._normalize_subsystem_group(dist, (3, nothing))
        @test g_ok == (3, nothing)

        # A wrong-length group throws.
        @test_throws ArgumentError Tarang._normalize_subsystem_group(dist, (1,))
        @test_throws ArgumentError Tarang._normalize_subsystem_group(dist, (1, 2, 3))
    end

    # -----------------------------------------------------------------------
    @testset "Subsystem constructor stores ranges/sizes/groups" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])
        solver = CstSolver(prob, CstBase([false, true]))  # x separable, z coupled

        # Default (global) group -> all-nothing.
        sub = Subsystem(solver)
        @test sub.group == (nothing, nothing)
        @test sub.problem === prob
        @test sub.dist === dist
        @test sub.dtype == infer_problem_dtype(prob)
        @test sub.subproblem[] === nothing       # not yet attached
        @test haskey(sub.scalar_ranges, u)
        @test sub.total_variable_size == scalar_field_dofs(u)
        @test sub.total_variable_size > 0

        # Explicit per-mode group is preserved; matrix_group keeps the mode.
        sub2 = Subsystem(solver, (3, nothing))
        @test sub2.group == (3, nothing)
        @test sub2.matrix_group == (3, nothing)   # x marked dependent -> kept

        # A different mode gets a distinct matrix_group (no collapse).
        sub3 = Subsystem(solver, (5, nothing))
        @test sub3.matrix_group == (5, nothing)
        @test sub3.matrix_group != sub2.matrix_group
    end

    @testset "Subsystem constructor without a .base field" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        prob = LBVP([u])
        prob.domain = nothing   # force dist inference via variables
        solver = CstSolverNoBase(prob)
        # No base field -> matrix_coupling defaults to all-true (fill(true,dim)).
        sub = Subsystem(solver)
        @test sub.group == (nothing,)
        @test sub.matrix_group == (nothing,)
        @test sub.total_variable_size == scalar_field_dofs(u)
    end

    # -----------------------------------------------------------------------
    @testset "get_separable_dim_size from domain coefficient shape" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))  # rfft -> 9
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))  # -> 6
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])

        cshape = Tarang.coefficient_shape(domain)
        @test get_separable_dim_size(prob, 1) == cshape[1]
        @test get_separable_dim_size(prob, 2) == cshape[2]
        # First Fourier axis halves under rfft: 16/2 + 1 = 9.
        @test get_separable_dim_size(prob, 1) == 9
    end

    @testset "get_separable_dim_size basis fallback (no domain)" begin
        # With problem.domain == nothing, the per-axis size is reconstructed
        # from the variable bases: the FIRST Fourier axis halves, others don't.
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=10, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])
        prob.domain = nothing   # force the basis-reconstruction fallback

        # Axis 1 is the first Fourier axis -> 16/2 + 1 = 9.
        @test get_separable_dim_size(prob, 1) == 9
        # Axis 2 is a later Fourier axis -> full size 10.
        @test get_separable_dim_size(prob, 2) == 10
    end

    # -----------------------------------------------------------------------
    @testset "generate_mode_groups Cartesian product" begin
        # One separable axis (dim 1) of size 4 in a 2D problem -> 4 groups,
        # each (m, nothing) for m in 0:3.
        groups = generate_mode_groups([1], [4], 2)
        @test length(groups) == 4
        @test groups == [(0, nothing), (1, nothing), (2, nothing), (3, nothing)]

        # Two separable axes -> full Cartesian product, coupled axis stays nothing.
        groups2 = generate_mode_groups([1, 3], [2, 2], 3)
        @test length(groups2) == 4
        @test all(g -> g[2] === nothing, groups2)      # axis 2 coupled
        @test all(g -> g[1] in 0:1 && g[3] in 0:1, groups2)
        @test (0, nothing, 0) in groups2
        @test (1, nothing, 1) in groups2

        # No separable axes -> a single all-nothing group (empty product).
        groups3 = generate_mode_groups(Int[], Int[], 2)
        @test length(groups3) == 1
        @test groups3[1] == (nothing, nothing)
    end

    # -----------------------------------------------------------------------
    @testset "get_matrix_coupling reads solver.base on serial path" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])
        mc = [false, true]
        solver = CstSolver(prob, CstBase(mc))
        out = get_matrix_coupling(solver, prob, dist)
        @test out == mc
        @test out !== mc            # collect() returns a fresh copy
    end

    # -----------------------------------------------------------------------
    @testset "analyze_operator_coupling" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")

        # nothing -> all-false vector of length ndims.
        @test analyze_operator_coupling(nothing, 2) == [false, false]

        # A z-derivative couples only the z axis (index 2).
        dz = Tarang.Differentiate(u, coords["z"], 1)
        c_dz = analyze_operator_coupling(dz, 2)
        @test c_dz == [false, true]

        # An x-derivative couples only the x axis (index 1).
        dx = Tarang.Differentiate(u, coords["x"], 1)
        @test analyze_operator_coupling(dx, 2) == [true, false]

        # Add/Subtract combine operand couplings (here x ∪ z = both).
        add = Tarang.AddOperator(dx, dz)
        @test analyze_operator_coupling(add, 2) == [true, true]
        sub = Tarang.SubtractOperator(dx, dz)
        @test analyze_operator_coupling(sub, 2) == [true, true]

        # Multiply by a numeric constant on the right couples only the left's axes.
        mulc = Tarang.MultiplyOperator(dz, 2.0)
        @test analyze_operator_coupling(mulc, 2) == [false, true]

        # Multiply of two derivative operands unions both axes.
        mul2 = Tarang.MultiplyOperator(dx, dz)
        @test analyze_operator_coupling(mul2, 2) == [true, true]

        # Single-operand wrapper (NegateOperator has an :operand field) recurses.
        neg = Tarang.NegateOperator(dz)
        @test analyze_operator_coupling(neg, 2) == [false, true]
    end

    # -----------------------------------------------------------------------
    @testset "build_subsystems: fully coupled -> single global subsystem" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])
        solver = CstSolver(prob, CstBase([true, true]))   # all coupled

        subs = build_subsystems(solver)
        @test length(subs) == 1
        @test subs[1].group == (nothing, nothing)
    end

    @testset "build_subsystems: one separable axis -> per-mode subsystems" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))  # rfft -> 5 modes
        zb = ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        prob = LBVP([u])
        solver = CstSolver(prob, CstBase([false, true]))  # x separable, z coupled

        subs = build_subsystems(solver)
        nmodes = Tarang.coefficient_shape(domain)[1]   # 5
        @test length(subs) == nmodes
        # Each subsystem is a distinct x-mode with z coupled.
        modes = sort([s.group[1] for s in subs])
        @test modes == collect(0:(nmodes - 1))
        @test all(s -> s.group[2] === nothing, subs)
        # Distinct modes carry distinct matrix groups (no silent collapse).
        @test length(unique(s.matrix_group for s in subs)) == nmodes
    end
end
