"""
Tests for the equation-structure check that `add_equation!` runs.

WHAT THIS PINS
--------------
`validate_equation_structure` detects misplaced terms and prints a "Suggested form"
that tells the user where a term belongs. It used to run only inside
`parse_equation`, which no solver path calls, so in practice nobody ever saw it. It
is now wired into `add_equation!`.

Wiring it in is only safe if its verdict is TRUE. The predicate it used to decide
"is this term implicit-capable?" (`_is_constant_term`) accepted just
Number/ZeroOperator/ConstantOperator — far narrower than what the matrix builder
accepts (`_is_const_or_param`) and narrower still than what the implicit path
supports since field-valued coefficients landed. It therefore reported
`dt(u) - q(z)*lap(u) = 0` and `dt(u) - nu0*lap(u) = 0` as misplaced and advised
moving them to the RHS — advice that is wrong, because those build correctly.

THE INVARIANT TESTED HERE is the equivalence itself, not the warning text:

    the validator declines to warn  <=>  the equation really does build an implicit
                                         operator that carries the term
    the validator warns             <=>  the equation really does fail
                                         (ImplicitNCCError) or the term really is
                                         not in the implicit operator

Every case below asserts BOTH halves: the verdict, and the reality it claims.

Uniquely-prefixed names (eqsv_*) — the full suite shares the Main namespace.
Tarang internals are called fully-qualified (Tarang.foo).
"""

using Test
using Tarang
using LinearAlgebra
using SparseArrays
using Logging

# ---------------------------------------------------------------------------
# Harness: verdict and reality
# ---------------------------------------------------------------------------

"""Add `equation` to `problem`, returning `(warned, messages)` for the structure warning.

Other warnings (the older string-level placement heuristic, matrix-shape notes) are
ignored: this asks only whether `validate_equation_structure` fired.
"""
function eqsv_verdict!(problem, equation::String)
    logger = Test.TestLogger(min_level=Logging.Warn)
    with_logger(logger) do
        Tarang.add_equation!(problem, equation)
    end
    msgs = [string(r.message) for r in logger.logs
            if occursin("misplaced terms", string(r.message))]
    return (!isempty(msgs), msgs)
end

"""Assemble the problem's matrices, reporting the outcome instead of throwing."""
function eqsv_build(problem)
    try
        L, M, F = with_logger(NullLogger()) do
            Tarang.build_matrices(problem)
        end
        return (:ok, L, M)
    catch e
        return (:error, e, nothing)
    end
end

# ---------------------------------------------------------------------------
# Domains and problems
# ---------------------------------------------------------------------------

"""2-D pure-Fourier problem for `u`, initialised to sin(x)cos(y) (lap(u0) = -2u0)."""
function eqsv_fourier_problem(; N::Int=16)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    dom = Domain(dist, (xb, yb))
    u = ScalarField(dom, "u")
    ensure_layout!(u, :g)
    g = Tarang.get_grid_data(u)
    xs = range(0, 2π; length=N + 1)[1:N]
    ys = range(0, 2π; length=N + 1)[1:N]
    for j in 1:N, i in 1:N
        g[i, j] = sin(xs[i]) * cos(ys[j])
    end
    prob = IVP([u]; namespace=Dict("u" => u))
    return u, prob, dom, dist, coords
end

"""Fourier-x by Chebyshev-z channel problem for `u`."""
function eqsv_channel_problem(; Nx::Int=8, Nz::Int=16)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    dom = Domain(dist, (xb, zb))
    u = ScalarField(dom, "u")
    prob = IVP([u]; namespace=Dict("u" => u))
    return u, prob, dom, dist, coords
end

"""Integrate `problem` to `T_end` and return max|u(T)| / max|u(0)|."""
function eqsv_decay(u, problem; T_end::Float64=1.0, dt::Float64=1e-3)
    ensure_layout!(u, :g)
    n0 = maximum(abs, Tarang.get_grid_data(u))
    solver = with_logger(NullLogger()) do
        InitialValueSolver(problem, SBDF2(); dt=dt)
    end
    for _ in 1:round(Int, T_end / dt)
        Tarang.step!(solver, dt)
    end
    ensure_layout!(u, :g)
    return maximum(abs, Tarang.get_grid_data(u)) / n0
end

@testset "equation structure validation: verdict matches reality" begin

    # =====================================================================
    # NO-WARN cases. Each must actually build an implicit operator that
    # carries the term, and (where there is a closed form) solve to it.
    # =====================================================================

    @testset "Number coefficient: no warning, solves to exp(-1)" begin
        u, prob, = eqsv_fourier_problem()
        warned, _ = eqsv_verdict!(prob, "dt(u) - 0.5*lap(u) = 0")
        @test warned == false

        status, L, _ = eqsv_build(prob)
        @test status === :ok
        @test nnz(L) > 0                      # the diffusion term IS implicit
        # lap(u0) = -2*u0 and nu = 0.5  ->  u(1) = exp(-1)*u(0)
        @test eqsv_decay(u, prob) ≈ exp(-1.0) rtol = 1e-4
    end

    @testset "0-D ScalarField parameter: no warning, and it is really used" begin
        # `_is_const_or_param` folds a 0-D / single-point field into a scalar
        # multiplier, so this is as implicit as the literal 0.5 above. The old
        # predicate rejected it and advised moving a genuine constant to the RHS.
        u, prob, _, dist, = eqsv_fourier_problem()
        nu0 = ScalarField(dist, "nu0", (), Float64)
        Tarang.set_grid_data!(nu0, [0.5])
        @test Tarang._is_const_or_param(nu0)
        @test Tarang._extract_scalar(nu0) == 0.5
        Tarang.add_parameters!(prob, nu0 = nu0)

        warned, _ = eqsv_verdict!(prob, "dt(u) - nu0*lap(u) = 0")
        @test warned == false

        status, L, _ = eqsv_build(prob)
        @test status === :ok
        @test nnz(L) > 0
        # Same equation as the literal-0.5 case, so the same closed form.
        @test eqsv_decay(u, prob) ≈ exp(-1.0) rtol = 1e-4
    end

    @testset "Chebyshev-varying coefficient: no warning, and L is the real operator" begin
        # Supported implicitly: one Jacobi axis, constant along every Fourier axis.
        function build_channel(scale)
            u, prob, dom, = eqsv_channel_problem()
            Z = Array(Tarang.create_meshgrid(dom)["z"])
            q = ScalarField(dom, "q")
            ensure_layout!(q, :g)
            Tarang.get_grid_data(q) .= Float64(scale) .* (1.0 .+ Z)
            Tarang.add_parameters!(prob, q = q)
            warned, _ = eqsv_verdict!(prob, "dt(u) - q*lap(u) = 0")
            status, L, _ = eqsv_build(prob)
            return warned, status, L
        end

        warned1, status1, L1 = build_channel(1.0)
        warned3, status3, L3 = build_channel(3.0)
        @test warned1 == false
        @test warned3 == false
        @test status1 === :ok && status3 === :ok
        @test nnz(L1) > 0
        # The operator must RESPOND to the coefficient — a coefficient-free block
        # would satisfy "builds fine" while solving a different equation.
        @test Matrix(L3) ≈ 3 .* Matrix(L1) rtol = 1e-10

        # ... and it is the RIGHT operator: on a 1-D Chebyshev domain L must act
        # as u -> -q(z)*u''(z). Checked against a pointwise oracle.
        N = 24
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, 1.0))
        dom = Domain(dist, (zb,))
        zc = vec(Array(Tarang.create_meshgrid(dom)["z"]))

        u = ScalarField(dom, "u")
        ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= sin.(2π .* zc)
        q = ScalarField(dom, "q")
        ensure_layout!(q, :g)
        Tarang.get_grid_data(q) .= 1.0 .+ zc

        prob = IVP([u]; namespace=Dict("u" => u))
        Tarang.add_parameters!(prob, q = q)
        warned, _ = eqsv_verdict!(prob, "dt(u) - q*lap(u) = 0")
        @test warned == false
        status, L, _ = eqsv_build(prob)
        @test status === :ok

        ensure_layout!(u, :c)
        uhat = ComplexF64.(vec(Array(Tarang.get_coeff_data(u))))
        got = L * uhat
        ref = ScalarField(dom, "ref")
        ensure_layout!(ref, :g)
        Tarang.get_grid_data(ref) .= -((1.0 .+ zc) .* (-(2π)^2 .* sin.(2π .* zc)))
        ensure_layout!(ref, :c)
        refhat = ComplexF64.(vec(Array(Tarang.get_coeff_data(ref))))
        @test maximum(abs, got .- refhat) / maximum(abs, refhat) < 1e-9
    end

    @testset "Division by a constant: no warning, term stays implicit" begin
        # `lap(u)/2.0` is scaled into the block by the matrix builder, but the old
        # predicate had no DivideOperator case at all and called it nonlinear.
        u, prob, = eqsv_fourier_problem()
        warned, _ = eqsv_verdict!(prob, "dt(u) - lap(u)/2.0 = 0")
        @test warned == false
        status, L, _ = eqsv_build(prob)
        @test status === :ok
        @test nnz(L) > 0
        @test eqsv_decay(u, prob) ≈ exp(-1.0) rtol = 1e-4
    end

    # =====================================================================
    # WARN cases. Each must actually fail, or actually leave the flagged
    # term out of the implicit operator.
    # =====================================================================

    @testset "Fourier-varying coefficient: warns, and really raises" begin
        # A coefficient varying along a Fourier axis couples Fourier modes: there is
        # no per-mode matrix for it, so the implicit path raises ImplicitNCCError.
        # Warning at add time is the same verdict, delivered earlier.
        u, prob, dom, = eqsv_fourier_problem()
        X = Array(Tarang.create_meshgrid(dom)["x"])
        nu_e = ScalarField(dom, "nu_e")
        ensure_layout!(nu_e, :g)
        Tarang.get_grid_data(nu_e) .= 0.5 .+ 0.1 .* sin.(X)
        Tarang.add_parameters!(prob, nu_e = nu_e)

        warned, msgs = eqsv_verdict!(prob, "dt(u) - nu_e*lap(u) = 0")
        @test warned == true
        # The advice must be "move it to the RHS", which is what the builder's own
        # error message tells the user to do.
        @test occursin("nu_e", msgs[1])
        @test occursin(r"∂t\(u\)\s*=", msgs[1])

        status, err, _ = eqsv_build(prob)
        @test status === :error
        @test err isa Tarang.ImplicitNCCError
    end

    @testset "Uniform field on a pure-Fourier domain: warns, and really raises" begin
        # Constant VALUE, but no Jacobi axis to build the multiply matrix on, so the
        # implicit path rejects it. The verdict follows the implementation, not the
        # mathematics of the particular data.
        u, prob, dom, = eqsv_fourier_problem()
        nu_e = ScalarField(dom, "nu_e")
        ensure_layout!(nu_e, :g)
        fill!(Tarang.get_grid_data(nu_e), 0.5)
        Tarang.add_parameters!(prob, nu_e = nu_e)

        warned, _ = eqsv_verdict!(prob, "dt(u) - nu_e*lap(u) = 0")
        @test warned == true
        status, err, _ = eqsv_build(prob)
        @test status === :error
        @test err isa Tarang.ImplicitNCCError
    end

    @testset "Fourier-varying coefficient in a channel: warns, and really raises" begin
        # One Jacobi axis is not enough — the coefficient must ALSO be constant along
        # the Fourier axis. This is the case the structural test alone would miss.
        u, prob, dom, = eqsv_channel_problem()
        X = Array(Tarang.create_meshgrid(dom)["x"])
        q = ScalarField(dom, "q")
        ensure_layout!(q, :g)
        Tarang.get_grid_data(q) .= 1.0 .+ 0.3 .* sin.(X)
        Tarang.add_parameters!(prob, q = q)

        warned, _ = eqsv_verdict!(prob, "dt(u) - q*lap(u) = 0")
        @test warned == true
        status, err, _ = eqsv_build(prob)
        @test status === :error
        @test err isa Tarang.ImplicitNCCError
    end

    @testset "Genuinely nonlinear term on the LHS: warns, and is not in L" begin
        u, prob, = eqsv_fourier_problem()
        warned, msgs = eqsv_verdict!(prob, "dt(u) + u*d(u,x) = 0")
        @test warned == true
        @test occursin(r"∂t\(u\)\s*=", msgs[1])          # suggested: move it right

        status, L, M = eqsv_build(prob)
        @test status === :ok
        # Reality behind the warning: the product contributes NOTHING to the implicit
        # operator, so leaving it on the LHS means it is not solved implicitly at all.
        @test nnz(L) == 0
        @test nnz(M) > 0
    end

    @testset "Linear term parked on the RHS: warns, and is not in L" begin
        u, prob, = eqsv_fourier_problem()
        warned, msgs = eqsv_verdict!(prob, "dt(u) = 0.5*lap(u)")
        @test warned == true
        @test occursin("Suggested form", msgs[1])

        status, L, _ = eqsv_build(prob)
        @test status === :ok
        # The diffusion really is absent from the implicit operator here...
        @test nnz(L) == 0

        # ... while the suggested spelling really does put it there.
        u2, prob2, = eqsv_fourier_problem()
        warned2, _ = eqsv_verdict!(prob2, "dt(u) - 0.5*lap(u) = 0")
        @test warned2 == false
        status2, L2, _ = eqsv_build(prob2)
        @test status2 === :ok
        @test nnz(L2) > 0
    end

    # =====================================================================
    # False-positive guards: legitimate usage that must stay quiet.
    # =====================================================================

    @testset "Nonlinear advection on the RHS is not flagged" begin
        # u⋅∇(q) expands to u_x*∂x(q) + u_z*∂z(q). The components of the vector
        # VARIABLE u must not be mistaken for coefficient fields — on a channel each
        # has exactly one Jacobi axis and (before initialisation) no Fourier
        # variation, i.e. it looks exactly like a representable NCC.
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        q = ScalarField(dist, "q", (xb, zb), Float64)
        u = VectorField(dist, coords, "u", (xb, zb), Float64)
        prob = IVP([q, u]; namespace=Dict("q" => q, "u" => u))

        warned, msgs = eqsv_verdict!(prob, "dt(q) - lap(q) = -u⋅∇(q)")
        @test warned == false
        @test isempty(msgs)

        # The predicate itself: a component of a vector variable is not a coefficient.
        vars = Tarang._problem_variable_operands(prob.variables)
        @test Tarang._references_problem_variable(u.components[1], vars)
        @test Tarang._is_implicit_coefficient(u.components[1], vars) == false
    end

    @testset "Explicit variable-coefficient diffusion on the RHS is not flagged" begin
        # `div(nu_e*grad(u))` on the RHS is the spelling the ImplicitNCCError message
        # itself recommends. Judging the term by its outermost node (a Divergence,
        # therefore "linear") would advise moving it to the LHS, where it raises —
        # the exact inverse of the correct advice.
        N = 16
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        dom = Domain(dist, (xb, yb))
        mesh = Tarang.create_meshgrid(dom)
        X, Y = mesh["x"], mesh["y"]

        u = ScalarField(dom, "u")
        u["g"] = @. sin(X) * cos(Y)
        nu_e = ScalarField(dom, "nu_e")
        nu_e["g"] = @. 2 + sin(X)

        prob = IVP([u]; namespace=Dict("u" => u))
        Tarang.add_parameters!(prob, nu_e = nu_e)
        warned, msgs = eqsv_verdict!(prob, "dt(u) = div(nu_e*grad(u))")
        @test warned == false
        @test isempty(msgs)

        # Reality behind "no warning": explicitly, it evaluates to ∇·(a∇u) exactly;
        # implicitly, the same coefficient raises. The RHS really is where it belongs.
        solver = with_logger(NullLogger()) do
            InitialValueSolver(prob, RK222(); dt=1e-3)
        end
        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :g)
        exact = @. cos(Y) * (cos(X)^2 - 2 * sin(X)^2 - 4 * sin(X))
        @test Array(Tarang.get_grid_data(rhs)) ≈ exact atol = 1e-10 rtol = 1e-10

        u2 = ScalarField(dom, "u")
        prob2 = IVP([u2]; namespace=Dict("u" => u2))
        Tarang.add_parameters!(prob2, nu_e = nu_e)
        warned2, _ = eqsv_verdict!(prob2, "dt(u) - nu_e*lap(u) = 0")
        @test warned2 == true
        status2, err2, _ = eqsv_build(prob2)
        @test status2 === :error
        @test err2 isa Tarang.ImplicitNCCError
    end

    @testset "A forcing field on the RHS is not flagged" begin
        # `f` is a field but not a solved-for variable: there is no matrix column to
        # move it into, so "move this to the LHS" would be impossible advice.
        u, prob, dom, = eqsv_fourier_problem()
        f = ScalarField(dom, "f")
        ensure_layout!(f, :g)
        fill!(Tarang.get_grid_data(f), 1.0)
        Tarang.add_parameters!(prob, f = f)

        warned, _ = eqsv_verdict!(prob, "dt(u) - lap(u) = f")
        @test warned == false
        status, L, _ = eqsv_build(prob)
        @test status === :ok
        @test nnz(L) > 0
    end

    @testset "A parameter defined after the equation is not flagged" begin
        # add_equation! now parses at add time, so a name the namespace does not know
        # yet resolves to a placeholder. Guessing that a placeholder is a
        # nonlinearity would report perfectly ordinary code.
        u, prob, = eqsv_fourier_problem()
        warned, _ = eqsv_verdict!(prob, "dt(u) - nu*lap(u) = 0")
        @test warned == false

        # ... and once the parameter exists the equation builds normally.
        Tarang.add_parameters!(prob, nu = 0.5)
        status, L, _ = eqsv_build(prob)
        @test status === :ok
        @test nnz(L) > 0
        @test eqsv_decay(u, prob) ≈ exp(-1.0) rtol = 1e-4
    end

    @testset "A constant unit vector coefficient is not flagged" begin
        # Boussinesq buoyancy: `Ra*Pr*b*ez` is a long-supported implicit term
        # (a constant VectorField is a rank-changing block expansion, not an NCC).
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        b = ScalarField(dist, "b", (xb, zb), Float64)
        u = VectorField(dist, coords, "u", (xb, zb), Float64)
        ez = Tarang.unit_vector_fields(coords, dist)[end]
        @test Tarang._is_constant_term(ez)

        prob = IVP([u, b]; namespace=Dict("u" => u, "b" => b))
        Tarang.add_parameters!(prob, ez = ez, Ra = 1.0e6, Pr = 1.0)
        warned, msgs = eqsv_verdict!(prob, "dt(u) - lap(u) - Ra*Pr*b*ez = 0")
        @test warned == false
        @test isempty(msgs)
    end

    # =====================================================================
    # Predicate unit contract.
    # =====================================================================

    @testset "_is_constant_term agrees with the matrix builder" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        dom = Domain(dist, (xb, zb))

        # Everything `_is_const_or_param` folds into a scalar must be constant here.
        nu0 = ScalarField(dist, "nu0", (), Float64)
        Tarang.set_grid_data!(nu0, [0.25])
        for e in (2.0, Tarang.ConstantOperator(3.0), Tarang.ZeroOperator(), nu0,
                  Tarang.DivideOperator(Tarang.ConstantOperator(1.0),
                                        Tarang.ConstantOperator(4.0)),
                  Tarang.NegateOperator(nu0))
            @test Tarang._is_constant_term(e) == true
            @test Tarang._is_const_or_param(e) == true
        end

        # A field with real spatial extent is NOT a constant, however it is stored.
        q = ScalarField(dom, "q")
        ensure_layout!(q, :g)
        Tarang.get_grid_data(q) .= 1.0
        @test Tarang._is_constant_term(q) == false
        @test Tarang._is_const_or_param(q) == false
    end

    @testset "_is_implicit_coefficient mirrors _implicit_ncc_matrix" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        dom = Domain(dist, (xb, zb))
        Z = Array(Tarang.create_meshgrid(dom)["z"])
        X = Array(Tarang.create_meshgrid(dom)["x"])

        # z-varying: representable.
        qz = ScalarField(dom, "qz")
        ensure_layout!(qz, :g)
        Tarang.get_grid_data(qz) .= 1.0 .+ Z
        @test Tarang._is_implicit_coefficient(qz, nothing) == true
        @test Tarang._implicit_ncc_matrix(qz) isa AbstractMatrix

        # x-varying: not representable.
        qx = ScalarField(dom, "qx")
        ensure_layout!(qx, :g)
        Tarang.get_grid_data(qx) .= 1.0 .+ 0.3 .* sin.(X)
        @test Tarang._is_implicit_coefficient(qx, nothing) == false
        @test Tarang._implicit_ncc_matrix(qx) isa Tarang.ImplicitNCCUnsupported

        # No Jacobi axis at all: not representable, whatever the values are.
        fcoords = CartesianCoordinates("x", "y")
        fdist = Distributor(fcoords; dtype=Float64, device=CPU())
        fx = RealFourier(fcoords["x"]; size=8, bounds=(0.0, 2π))
        fy = ComplexFourier(fcoords["y"]; size=8, bounds=(0.0, 2π))
        nu_e = ScalarField(Domain(fdist, (fx, fy)), "nu_e")
        ensure_layout!(nu_e, :g)
        fill!(Tarang.get_grid_data(nu_e), 0.5)
        @test Tarang._is_implicit_coefficient(nu_e, nothing) == false
        @test Tarang._implicit_ncc_matrix(nu_e) isa Tarang.ImplicitNCCUnsupported

        # A derivative of a field is not a bare field, so it is not a coefficient —
        # resolving through it would build a multiply-by-q matrix for a ∂q/∂z factor.
        dq = Tarang.Differentiate(qz, coords["z"], 1)
        @test Tarang._is_implicit_coefficient(dq, nothing) == false
        @test Tarang._implicit_ncc_matrix(dq) isa Tarang.ImplicitNCCUnsupported
    end

    @testset "inspecting a coefficient never transforms it" begin
        # `add_equation!` must not mutate a user's field as a side effect of
        # validating a string.
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        dom = Domain(dist, (xb, zb))
        Z = Array(Tarang.create_meshgrid(dom)["z"])
        q = ScalarField(dom, "q")
        ensure_layout!(q, :g)
        Tarang.get_grid_data(q) .= 1.0 .+ Z
        before = copy(Array(Tarang.get_grid_data(q)))

        u = ScalarField(dom, "u")
        prob = IVP([u]; namespace=Dict("u" => u))
        Tarang.add_parameters!(prob, q = q)
        eqsv_verdict!(prob, "dt(u) - q*lap(u) = 0")

        @test q.current_layout === :g
        @test Array(Tarang.get_grid_data(q)) == before
    end

    @testset "add_equation! never throws and always stores the equation" begin
        # A structure check must not be able to reject an equation.
        u, prob, = eqsv_fourier_problem()
        with_logger(NullLogger()) do
            Tarang.add_equation!(prob, "0 = dt(u)")              # dt on the RHS
            Tarang.add_equation!(prob, "dt(u) + = = broken")     # unparseable
            Tarang.add_equation!(prob, "u(x=0) = 0")             # boundary condition
        end
        @test length(prob.equations) == 3
        @test prob.equations[1] == "0 = dt(u)"
        @test prob.equations[3] == "u(x=0) = 0"
    end

    @testset "the 3-argument signature still works" begin
        # `parse_equation` calls validate_equation_structure positionally and has no
        # variable list to give it.
        u, prob, = eqsv_fourier_problem()
        ns = prob.namespace
        dt_u = Tarang.parse_expression("dt(u)", ns)
        @test Tarang.validate_equation_structure(dt_u, Tarang.ZeroOperator(),
                                                 "dt(u) = 0") == true
        # A time derivative on the RHS is still an error on that path.
        @test_throws ArgumentError Tarang.validate_equation_structure(
            Tarang.ZeroOperator(), dt_u, "0 = dt(u)")
    end

    @testset "equations without a time derivative are left alone" begin
        # Diagnostics, constraints and BCs have no IMEX splitting to get wrong, and
        # are not parsed at add time at all.
        @test Tarang._has_time_derivative_syntax("dt(u) - lap(u) = 0")
        @test Tarang._has_time_derivative_syntax("∂t(u) - lap(u) = 0")
        @test Tarang._has_time_derivative_syntax("u - skew(grad(psi)) = 0") == false
        @test Tarang._has_time_derivative_syntax("integ(psi) = 0") == false
        # `dt` inside a longer identifier is not a time derivative.
        @test Tarang._has_time_derivative_syntax("sdt(u) = 0") == false

        u, prob, = eqsv_fourier_problem()
        warned, _ = eqsv_verdict!(prob, "u - lap(u) = 0")
        @test warned == false
    end
end
