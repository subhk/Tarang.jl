using Test
using Tarang
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Coverage tests for src/core/subsystems/subproblem_expr_helpers.jl
#
# These exercise the structural expression-analysis helpers used when sizing
# subproblem matrices: _subproblem_expr_dofs (and its reductions), the
# expression_matrices fallbacks + parent-vector selector path, valid-mode
# masking, and the small predicate helpers. All assertions are invariants or
# known analytic sizes for a 2D Fourier×Chebyshev domain at a single
# separable Fourier mode.
# ---------------------------------------------------------------------------

# Mock solver matching the pattern in test_expression_matrices.jl. The
# Subsystem/Subproblem constructors only read `problem` and
# `base.matrix_coupling`.
struct CovSolverBase
    matrix_coupling::Vector{Bool}
end
struct CovSolver
    problem::Tarang.Problem
    base::CovSolverBase
end

@testset "subproblem_expr_helpers coverage" begin
    # ── Shared 2D Fourier×Chebyshev setup ───────────────────────────────────
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    Nx = 16
    Nz = 8
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 4.0))
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xb, zb))

    b = ScalarField(domain, "b")                # full 2D scalar
    u = VectorField(domain, "u")                # 2-component vector
    tau_b = ScalarField(dist, "tau_b", (xb,), Float64)   # 1D Fourier-only tau
    tau_p = ScalarField(dist, "tau_p", (), Float64)      # 0D tau (no bases)

    problem = IVP([b, u, tau_b, tau_p])
    # Fourier separable (false), Chebyshev coupled (true)
    solver = CovSolver(problem, CovSolverBase([false, true]))
    # group (5, nothing): Fourier mode 5 separable, Chebyshev coupled
    subsys = Subsystem(solver, (5, nothing))
    sp = Subproblem(solver, (subsys,), (5, nothing))

    # The coupled (Chebyshev) dimension contributes Nz DOFs; the separable
    # Fourier dimension contributes 1.  => scalar field size == Nz.
    @test Tarang.subproblem_field_size(sp, b) == Nz

    # ── _coord_name ─────────────────────────────────────────────────────────
    @testset "_coord_name" begin
        cx = coords["x"]
        @test Tarang._coord_name(cx) == "x"
        # Tuple form unwraps the first element.
        @test Tarang._coord_name((cx,)) == "x"
        @test Tarang._coord_name((coords["z"], cx)) == "z"
    end

    # ── _subproblem_expr_dofs: leaf + constant cases ────────────────────────
    @testset "_subproblem_expr_dofs leaves & constants" begin
        @test Tarang._subproblem_expr_dofs(sp, nothing) == 0
        @test Tarang._subproblem_expr_dofs(sp, 3.14) == 0
        @test Tarang._subproblem_expr_dofs(sp, Tarang.ZeroOperator()) == 0
        # ScalarField / VectorField leaves report their field sizes.
        @test Tarang._subproblem_expr_dofs(sp, b) == Nz
        @test Tarang._subproblem_expr_dofs(sp, u) == 2 * Nz
        @test Tarang._subproblem_expr_dofs(sp, tau_p) == 1
    end

    # ── _subproblem_expr_dofs: arithmetic operators (lines 49/52/54) ────────
    @testset "_subproblem_expr_dofs arithmetic" begin
        # Add/Subtract take the max of operand footprints.
        add_op = AddOperator(b, tau_p)            # max(Nz, 1) == Nz
        @test Tarang._subproblem_expr_dofs(sp, add_op) == Nz
        sub_op = SubtractOperator(u, b)           # max(2Nz, Nz) == 2Nz
        @test Tarang._subproblem_expr_dofs(sp, sub_op) == 2 * Nz
        # Multiply/Divide also take the max.
        mul_op = MultiplyOperator(b, b)           # max(Nz, Nz) == Nz
        @test Tarang._subproblem_expr_dofs(sp, mul_op) == Nz
        # Negate preserves the operand footprint.
        neg_op = NegateOperator(u)
        @test Tarang._subproblem_expr_dofs(sp, neg_op) == 2 * Nz
    end

    # ── _subproblem_expr_dofs: Future branch (lazy Add) (lines 56-60) ───────
    @testset "_subproblem_expr_dofs Future" begin
        # A lazy Add Future is a Future but NOT an AddOperator, so it falls
        # through to the Future branch which takes the max over future_args.
        fut = Tarang.Add(b, tau_p)            # args (b, tau_p) -> max(Nz, 1)
        @test isa(fut, Tarang.Future)
        @test !isa(fut, AddOperator)
        @test Tarang._subproblem_expr_dofs(sp, fut) == Nz

        # A lazy Multiply over (b, b) -> max(Nz, Nz) == Nz.
        fmul = Tarang.Multiply(b, b)
        @test isa(fmul, Tarang.Future)
        @test Tarang._subproblem_expr_dofs(sp, fmul) == Nz
    end

    # ── _subproblem_expr_dofs + _subproblem_reduce_dofs: Integrate ──────────
    @testset "_subproblem_expr_dofs Integrate / reduce" begin
        cz = coords["z"]
        cx = coords["x"]

        # Integrate over the coupled Chebyshev axis collapses the Nz-mode axis
        # to a single value: Nz / Nz == 1.
        int_z = Tarang.Integrate(b, cz)
        @test Tarang._subproblem_expr_dofs(sp, int_z) == 1

        # Integrate over the separable Fourier axis: this subproblem is mode 5
        # (a non-zero separable group), so the Fourier integral vanishes => 0.
        int_x = Tarang.Integrate(b, cx)
        @test Tarang._subproblem_expr_dofs(sp, int_x) == 0

        # Compound integrate over (x, z): applies reductions sequentially.
        # x first -> 0 (non-DC Fourier), then z reduction of 0 stays 0.
        int_xz = Tarang.Integrate(b, (cx, cz))
        @test Tarang._subproblem_expr_dofs(sp, int_xz) == 0
    end

    # ── Integrate at the DC Fourier mode (group 0) keeps the inner count ────
    @testset "_subproblem_reduce_dofs Fourier DC mode" begin
        subsys0 = Subsystem(solver, (0, nothing))
        sp0 = Subproblem(solver, (subsys0,), (0, nothing))
        cx = coords["x"]
        # tau_b is Fourier-only; its size at a separable mode is 1.
        @test Tarang.subproblem_field_size(sp0, tau_b) == 1
        # Integrating tau_b over x at the DC mode keeps inner (== 1).
        int_x0 = Tarang.Integrate(tau_b, cx)
        @test Tarang._subproblem_expr_dofs(sp0, int_x0) == 1
        # At a non-zero mode it vanishes.
        int_x5 = Tarang.Integrate(tau_b, cx)
        @test Tarang._subproblem_expr_dofs(sp, int_x5) == 0
    end

    # ── Average reduction (same reduce path as Integrate) ───────────────────
    @testset "_subproblem_expr_dofs Average" begin
        cz = coords["z"]
        avg_z = Tarang.Average(b, cz)
        @test Tarang._subproblem_expr_dofs(sp, avg_z) == 1
    end

    # ── Interpolate reduction (interpolate=true path) ───────────────────────
    @testset "_subproblem_expr_dofs Interpolate" begin
        cz = coords["z"]
        cx = coords["x"]
        # Interpolating onto a point on the coupled Chebyshev axis collapses
        # the Nz coefficient axis: Nz / Nz == 1.
        interp_z = Tarang.Interpolate(b, cz, 0.5)
        @test Tarang._subproblem_expr_dofs(sp, interp_z) == 1
        # Interpolating on a Fourier axis keeps the local mode count (1 here)
        # because interpolate=true short-circuits the DC-mode masking.
        interp_x = Tarang.Interpolate(b, cx, 1.0)
        @test Tarang._subproblem_expr_dofs(sp, interp_x) == Nz
    end

    # ── Trace path: VectorField-resolvable branch (lines 84-85) ─────────────
    @testset "_subproblem_expr_dofs Trace" begin
        # Gradient(u) is a tensor-like expr; its Trace should resolve back to
        # the underlying vector field u and report a single component's size.
        gu = Gradient(u, coords)
        tr = Tarang.Trace(gu)
        # Trace resolves the operand field (VectorField u) and returns the
        # first component's size (Nz).
        @test Tarang._subproblem_expr_dofs(sp, tr) == Nz

        # Trace of a scalar gradient resolves to a ScalarField -> Nz.
        gb = Gradient(b, coords)
        trb = Tarang.Trace(gb)
        @test Tarang._subproblem_expr_dofs(sp, trb) == Nz
    end

    # ── hasfield(:operand) operator-wrapper path via subproblem_matrix ──────
    @testset "_subproblem_expr_dofs operator wrapper (Differentiate/Laplacian)" begin
        # A spatial operator exposes an explicit sparse matrix; the helper uses
        # its row count.  d/dz(b) and lap(b) act on the Nz-mode coupled axis.
        dz_b = Differentiate(b, coords["z"], 1)
        n_dz = Tarang._subproblem_expr_dofs(sp, dz_b)
        @test n_dz == Nz

        lap_b = Laplacian(b)
        n_lap = Tarang._subproblem_expr_dofs(sp, lap_b)
        @test n_lap == Nz

        # Gradient of a scalar: (ndim*Nz) rows.
        grad_b = Gradient(b, coords)
        @test Tarang._subproblem_expr_dofs(sp, grad_b) == 2 * Nz
    end

    # ── hasfield(:operand) wrapper WITHOUT a subproblem_matrix -> child ──────
    @testset "_subproblem_expr_dofs operand-fallback (Grid/Coeff/Copy)" begin
        # Grid/Coeff/Copy are Operators that expose `operand` but have no
        # specialized subproblem_matrix (the catch-all returns nothing), so the
        # helper falls back to the child operand's footprint.
        @test Tarang._subproblem_expr_dofs(sp, Tarang.Grid(b)) == Nz
        @test Tarang._subproblem_expr_dofs(sp, Tarang.Coeff(b)) == Nz
        @test Tarang._subproblem_expr_dofs(sp, Tarang.Copy(b)) == Nz
        # Vector operand propagates its larger footprint.
        @test Tarang._subproblem_expr_dofs(sp, Tarang.Grid(u)) == 2 * Nz
    end

    # ── _has_only_zero_dim_bases ────────────────────────────────────────────
    @testset "_has_only_zero_dim_bases" begin
        # tau_p has no bases at all -> true.
        @test Tarang._has_only_zero_dim_bases(tau_p) == true
        # b has real bases -> false.
        @test Tarang._has_only_zero_dim_bases(b) == false
        # Vector of 0-D scalar components -> true.
        tau_p2 = ScalarField(dist, "tau_p2", (), Float64)
        @test Tarang._has_only_zero_dim_bases(tau_p) == true
        @test Tarang._has_only_zero_dim_bases(tau_p2) == true
    end

    # ── _is_zero_separable_group ────────────────────────────────────────────
    @testset "_is_zero_separable_group" begin
        # group (5, nothing): a non-zero Integer entry -> NOT zero group.
        @test Tarang._is_zero_separable_group(sp) == false
        # group (0, nothing): only zero/nothing entries -> zero group.
        subsys0 = Subsystem(solver, (0, nothing))
        sp0 = Subproblem(solver, (subsys0,), (0, nothing))
        @test Tarang._is_zero_separable_group(sp0) == true
    end

    # ── get_valid_modes (equation valid-mode array) ─────────────────────────
    @testset "get_valid_modes" begin
        # Missing "valid_modes" -> all-true of requested length.
        v = Tarang.get_valid_modes(Dict{Any,Any}(), sp, 5)
        @test v == ones(Bool, 5)
        @test eltype(v) == Bool
        # Provided "valid_modes" is returned (vec'd).
        mask = Bool[true, false, true]
        v2 = Tarang.get_valid_modes(Dict{Any,Any}("valid_modes" => mask), sp, 3)
        @test v2 == mask
    end

    # ── get_valid_modes_var ─────────────────────────────────────────────────
    @testset "get_valid_modes_var" begin
        # Normal field with no valid_modes attribute -> all true of field size.
        v = Tarang.get_valid_modes_var(b, sp)
        @test v == ones(Bool, Nz)

        # 0-D tau in a NON-zero separable group must be fully masked out.
        # tau_p has size 1; sp is group (5, nothing) which is not the zero
        # group, so a 0-D-only field is masked to all-false.
        vt = Tarang.get_valid_modes_var(tau_p, sp)
        @test vt == zeros(Bool, 1)

        # Same 0-D tau in the ZERO separable group is kept (all true).
        subsys0 = Subsystem(solver, (0, nothing))
        sp0 = Subproblem(solver, (subsys0,), (0, nothing))
        vt0 = Tarang.get_valid_modes_var(tau_p, sp0)
        @test vt0 == ones(Bool, 1)
    end

    # ── expression_matrices fallbacks: String / Symbol / Nothing / Number ───
    @testset "expression_matrices scalar fallbacks" begin
        vars = [b]
        @test isempty(expression_matrices("a string", sp, vars))
        @test isempty(expression_matrices(:a_symbol, sp, vars))
        @test isempty(expression_matrices(nothing, sp, vars))
        @test isempty(expression_matrices(7, sp, vars))
    end

    # ── expression_matrices(ScalarField/VectorField) — identity path ────────
    @testset "expression_matrices field-in-vars identity" begin
        # ScalarField b that IS a var -> n×n identity.
        resb = expression_matrices(b, sp, [b])
        @test haskey(resb, b)
        @test size(resb[b]) == (Nz, Nz)
        @test resb[b] ≈ sparse(ComplexF64(1)*I, Nz, Nz)

        # VectorField u that IS a var -> (2Nz)×(2Nz) identity.
        resu = expression_matrices(u, sp, [u])
        @test haskey(resu, u)
        n_vec = 2 * Nz
        @test size(resu[u]) == (n_vec, n_vec)
        @test resu[u] ≈ sparse(ComplexF64(1)*I, n_vec, n_vec)
    end

    # ── expression_matrices(ScalarField) — parent-vector selector path ──────
    @testset "expression_matrices parent-vector selector" begin
        # A component ScalarField of u, when u (the parent vector) is the var,
        # should produce a selector that maps the parent unknown to this comp.
        comp0 = u.components[1]
        comp1 = u.components[2]
        vars = [u]

        res0 = expression_matrices(comp0, sp, vars)
        @test haskey(res0, u)
        sel0 = res0[u]
        n_parent = Tarang.subproblem_field_size(sp, u)   # 2*Nz
        comp_size = div(n_parent, length(u.components))   # Nz
        @test size(sel0) == (comp_size, n_parent)
        # First component selector picks columns 1..comp_size.
        @test sel0[1:comp_size, 1:comp_size] ≈ sparse(ComplexF64(1)*I, comp_size, comp_size)
        @test all(iszero, sel0[:, comp_size+1:end])

        res1 = expression_matrices(comp1, sp, vars)
        sel1 = res1[u]
        # Second component selector picks the trailing block of columns.
        @test all(iszero, sel1[:, 1:comp_size])
        @test sel1[1:comp_size, comp_size+1:end] ≈ sparse(ComplexF64(1)*I, comp_size, comp_size)
    end

    # ── expression_matrices(ScalarField) — not a var, not a component ───────
    @testset "expression_matrices scalar not-in-vars empty" begin
        orphan = ScalarField(domain, "orphan")
        res = expression_matrices(orphan, sp, [b])
        @test isempty(res)
    end

    # ── expression_matrices(VectorField) — not in vars -> empty ─────────────
    @testset "expression_matrices vector not-in-vars empty" begin
        v2 = VectorField(domain, "v2")
        res = expression_matrices(v2, sp, [b])
        @test isempty(res)
    end

    # ── _field_in_vars: identity + name-match branches ──────────────────────
    @testset "_field_in_vars" begin
        @test Tarang._field_in_vars(b, [b]) == true
        # Distinct object with the SAME name matches by name.
        b_twin = ScalarField(domain, "b")
        @test Tarang._field_in_vars(b_twin, [b]) == true
        # Different name, not present -> false.
        c = ScalarField(domain, "c_unique")
        @test Tarang._field_in_vars(c, [b]) == false
        @test Tarang._field_in_vars(b, []) == false
    end

    # ── _find_parent_vector: VectorField + TensorField component lookup ─────
    @testset "_find_parent_vector" begin
        comp0 = u.components[1]
        comp1 = u.components[2]
        # Component found in a VectorField var -> (parent, index).
        pv = Tarang._find_parent_vector(comp0, [u])
        @test pv !== nothing
        @test pv[1] === u
        @test pv[2] == 1
        pv1 = Tarang._find_parent_vector(comp1, [u])
        @test pv1[2] == 2
        # Not a component of any var -> nothing.
        lone = ScalarField(domain, "lone")
        @test Tarang._find_parent_vector(lone, [u]) === nothing
        @test Tarang._find_parent_vector(lone, []) === nothing

        # TensorField branch: a component of a tensor var is found by name.
        T = TensorField(domain, "T")
        tcomps = vec(T.components)
        # First component -> index 1.
        pvt = Tarang._find_parent_vector(tcomps[1], [T])
        @test pvt !== nothing
        @test pvt[1] === T
        @test pvt[2] == 1
        # A LATER component forces the inner loop to iterate past the first
        # entry before matching (exercises the loop-continuation path).
        last_idx = length(tcomps)
        pvt_last = Tarang._find_parent_vector(tcomps[last_idx], [T])
        @test pvt_last !== nothing
        @test pvt_last[1] === T
        @test pvt_last[2] == last_idx
    end

    # ── is_zero_expression methods defined in this module ───────────────────
    @testset "is_zero_expression" begin
        @test Tarang.is_zero_expression(nothing) == true
        @test Tarang.is_zero_expression(0) == true
        @test Tarang.is_zero_expression(0.0) == true
        @test Tarang.is_zero_expression(2.5) == false
    end
end
