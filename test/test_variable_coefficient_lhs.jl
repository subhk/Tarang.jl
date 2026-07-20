"""
Tests for FIELD-VALUED (non-constant) coefficients on the IMPLICIT left-hand side,
e.g. `dt(u) - nu_e*lap(u) = 0` where `nu_e` is a ScalarField rather than a number.

BUG THIS PINS (silent wrong answer, fixed 2026-07-20)
-----------------------------------------------------
`build_expression_matrix_block` (problem_matrices_spectral.jl) reduced a product of two
non-constant factors to a ZERO block:

    if _is_const_or_param(expr.left)      -> scale by the constant
    elseif _is_const_or_param(expr.right) -> scale by the constant
    else                                  -> _zero_block(...)   # <-- whole term deleted

A ScalarField coefficient is not `_is_const_or_param` (it has >1 grid point) and neither is
`lap(u)`, so `nu_e*lap(u)` took the else branch and the ENTIRE diffusion term vanished from
the global L matrix. `dt(u) - nu_e*lap(u) = 0` was assembled as `dt(u) = 0` and integrated
as an inviscid run: the answer was exactly 1.0 (no decay) for nu_e = 0.01, 0.5 and 5.0
alike, with zero warnings or errors. Pure-Fourier problems are affected because
`_try_build_subproblems!` deliberately skips them (solver_types.jl:283), so the global
matrix path is the one that runs.

CONTRACT NOW PINNED
-------------------
1. A CONSTANT scalar coefficient is unchanged and still integrates correctly.
2. A field coefficient that IS representable in the implicit operator (varying along a
   single Chebyshev/Jacobi axis, constant along every Fourier axis) is built into a real
   multiply-by-coefficient matrix, and the matrix responds to the coefficient's value.
3. A field coefficient that is NOT representable (pure-Fourier, Fourier-varying, several
   Jacobi axes, or a non-bare-field expression) raises `Tarang.ImplicitNCCError` naming the
   offending term and pointing at the explicit-RHS spelling. An error is the point: it is
   strictly better than the old inviscid run.
4. Nothing on this path silently drops a term — in particular the answer is never
   independent of the coefficient while still reporting success.

Uniquely-prefixed names (vclhs_*) — the full suite shares the Main namespace.
All Tarang internals are called fully-qualified (Tarang.foo).
"""

using Test
using Tarang
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

"""2-D pure-Fourier problem `dt(u) - nu*lap(u) = 0` with u0 = sin(x)cos(y).

`lap(u0) == -2*u0`, so the exact solution decays as `exp(-2*nu*t)`.
`nu_kind == :const` puts a Float64 on the LHS; `:field` puts a uniform ScalarField
there (mathematically identical, and the case that used to be dropped).
"""
function vclhs_fourier_problem(nu_kind::Symbol, nu_val::Real; N::Int=16)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))

    u = ScalarField(dist, "u", (xb, yb), Float64)
    ensure_layout!(u, :g)
    g = Tarang.get_grid_data(u)
    xs = range(0, 2π; length=N + 1)[1:N]
    ys = range(0, 2π; length=N + 1)[1:N]
    for j in 1:N, i in 1:N
        g[i, j] = sin(xs[i]) * cos(ys[j])
    end

    prob = IVP([u]; namespace=Dict("u" => u))
    if nu_kind === :const
        Tarang.add_equation!(prob, "dt(u) - $(nu_val)*lap(u) = 0")
    else
        nf = ScalarField(dist, "nu_e", (xb, yb), Float64)
        ensure_layout!(nf, :g)
        fill!(Tarang.get_grid_data(nf), Float64(nu_val))
        Tarang.add_parameters!(prob, nu_e = nf)
        Tarang.add_equation!(prob, "dt(u) - nu_e*lap(u) = 0")
    end
    return u, prob
end

"""Integrate to `T` and return `max|u(T)| / max|u(0)|`."""
function vclhs_decay(nu_kind::Symbol, nu_val::Real; T::Float64=1.0, dt::Float64=1e-3)
    u, prob = vclhs_fourier_problem(nu_kind, nu_val)
    ensure_layout!(u, :g)
    n0 = maximum(abs, Tarang.get_grid_data(u))
    solver = InitialValueSolver(prob, SBDF2(); dt=dt)
    for _ in 1:round(Int, T / dt)
        Tarang.step!(solver, dt)
    end
    ensure_layout!(u, :g)
    return maximum(abs, Tarang.get_grid_data(u)) / n0
end

"""Run the field-coefficient case, reporting `(:error, msg)` or `(:ok, decay)`.

Used by the regression testset, which has to distinguish "was rejected" from
"ran and produced an answer" without either outcome aborting the test.
"""
function vclhs_field_outcome(nu_val::Real; T::Float64=0.25, dt::Float64=2e-3)
    try
        return (:ok, vclhs_decay(:field, nu_val; T=T, dt=dt))
    catch e
        e isa Tarang.ImplicitNCCError && return (:error, sprint(showerror, e))
        rethrow()
    end
end

"""1-D Chebyshev problem `dt(u) - q*lap(u) = 0` with `q(z) = scale*(1+z)`.
Returns the assembled global stiffness matrix L."""
function vclhs_cheb_L(scale::Real; N::Int=16, Lz::Float64=1.0)
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, Lz))
    dom = Domain(dist, (zb,))
    zc = vec(Array(Tarang.create_meshgrid(dom)["z"]))

    u = ScalarField(dom, "u")
    q = ScalarField(dom, "q")
    ensure_layout!(q, :g)
    Tarang.get_grid_data(q) .= Float64(scale) .* (1.0 .+ zc)

    prob = IVP([u]; namespace=Dict("u" => u))
    Tarang.add_parameters!(prob, q = q)
    Tarang.add_equation!(prob, "dt(u) - q*lap(u) = 0")
    L, _, _ = Tarang.build_matrices(prob)
    return L
end

"""A 1-D Chebyshev ScalarField whose grid values are `f(z)`."""
function vclhs_cheb_field(f; N::Int=16, name::String="q")
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, 1.0))
    dom = Domain(dist, (zb,))
    zc = vec(Array(Tarang.create_meshgrid(dom)["z"]))
    q = ScalarField(dom, name)
    ensure_layout!(q, :g)
    Tarang.get_grid_data(q) .= f.(zc)
    return q, dom, coords
end

@testset "variable coefficient on implicit LHS" begin

    # ---------------------------------------------------------------------
    # 1. Constant coefficient: unchanged, still correct.
    # ---------------------------------------------------------------------
    @testset "constant coefficient still integrates to exp(-1)" begin
        # nu = 0.5 and lap(u0) = -2*u0  ->  u(1) = exp(-1) * u(0)
        decay = vclhs_decay(:const, 0.5; T=1.0, dt=1e-3)
        @test decay ≈ exp(-1.0) rtol = 1e-4

        # ... and it genuinely tracks the coefficient (guards the oracle itself).
        @test vclhs_decay(:const, 0.25; T=1.0, dt=1e-3) ≈ exp(-0.5) rtol = 1e-4
    end

    # ---------------------------------------------------------------------
    # 2. Field coefficient on a pure-Fourier LHS: rejected, not silently run.
    # ---------------------------------------------------------------------
    @testset "field coefficient on pure-Fourier LHS raises" begin
        @test_throws Tarang.ImplicitNCCError vclhs_decay(:field, 0.5; T=0.05, dt=1e-2)

        # The message must name the offending term and hand over a working RHS spelling,
        # so the user can act on it without reading the source.
        err = try
            vclhs_decay(:field, 0.5; T=0.05, dt=1e-2)
            nothing
        catch e
            e
        end
        @test err isa Tarang.ImplicitNCCError
        msg = sprint(showerror, err)
        @test occursin("nu_e*lap(u)", msg)          # names the term
        @test occursin("nu_e", msg)                 # names the coefficient
        @test occursin("∂x(nu_e)*∂x(u)", msg)       # the working explicit-RHS spelling
        @test occursin("right-hand side", msg)
    end

    # ---------------------------------------------------------------------
    # 3. Representable (Chebyshev-varying) coefficient: built, and it responds.
    # ---------------------------------------------------------------------
    @testset "Chebyshev-varying coefficient builds a real matrix" begin
        L1 = vclhs_cheb_L(1.0)
        L3 = vclhs_cheb_L(3.0)

        # The old behaviour was a zero block; a real operator is the whole point.
        @test nnz(L1) > 0
        @test nnz(L3) > 0

        # The matrix must RESPOND to the coefficient. `q -> 3q` is exactly linear in the
        # multiply-by-q operator, so L must scale by 3.
        @test Matrix(L3) ≈ 3 .* Matrix(L1) rtol = 1e-10
        @test !isapprox(Matrix(L3), Matrix(L1); rtol=1e-6)
    end

    @testset "Chebyshev NCC matrix matches a pointwise oracle" begin
        # L must act as u -> -q(z)*u''(z). Check against the spectral coefficients of that
        # product computed independently on the grid, for u = sin(2*pi*z).
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
        Tarang.add_equation!(prob, "dt(u) - q*lap(u) = 0")
        L, _, _ = Tarang.build_matrices(prob)

        ensure_layout!(u, :c)
        uhat = ComplexF64.(vec(Array(Tarang.get_coeff_data(u))))
        got = L * uhat

        # reference: -q(z) * u''(z), with u'' = -(2*pi)^2 * sin(2*pi*z)
        ref = ScalarField(dom, "ref")
        ensure_layout!(ref, :g)
        Tarang.get_grid_data(ref) .= -((1.0 .+ zc) .* (-(2π)^2 .* sin.(2π .* zc)))
        ensure_layout!(ref, :c)
        refhat = ComplexF64.(vec(Array(Tarang.get_coeff_data(ref))))

        @test maximum(abs, got .- refhat) / maximum(abs, refhat) < 1e-9
    end

    @testset "Fourier x Chebyshev: z-varying coefficient is representable" begin
        # The channel-flow case: q constant along the Fourier axis, varying along Chebyshev.
        function build(scale)
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; dtype=Float64, device=CPU())
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
            dom = Domain(dist, (xb, zb))
            Z = Array(Tarang.create_meshgrid(dom)["z"])
            u = ScalarField(dom, "u")
            q = ScalarField(dom, "q")
            ensure_layout!(q, :g)
            Tarang.get_grid_data(q) .= Float64(scale) .* (1.0 .+ Z)
            prob = IVP([u]; namespace=Dict("u" => u))
            Tarang.add_parameters!(prob, q = q)
            Tarang.add_equation!(prob, "dt(u) - q*lap(u) = 0")
            L, _, _ = Tarang.build_matrices(prob)
            return L
        end
        L1 = build(1.0)
        L2 = build(2.0)
        @test nnz(L1) > 0
        @test Matrix(L2) ≈ 2 .* Matrix(L1) rtol = 1e-10
    end

    # ---------------------------------------------------------------------
    # 3b. Same contract via the object-syntax (Future `Multiply`) route.
    # ---------------------------------------------------------------------
    @testset "object-syntax Multiply obeys the same contract" begin
        # Equations built from Julia objects rather than a string produce `Multiply`
        # Futures, which took a SEPARATE branch of build_expression_matrix_block. That
        # branch treated "more than one field factor" as nonlinear and returned a zero
        # block, dropping `q*lap(u)` exactly like the string path did.
        q, dom, _ = vclhs_cheb_field(z -> 1.0 + z; N=16)
        q3, _, _ = vclhs_cheb_field(z -> 3.0 * (1.0 + z); N=16, name="q3")
        u = ScalarField(dom, "u")

        m = Tarang.Multiply(q, Tarang.Laplacian(u))
        @test m isa Tarang.Future
        B = Tarang.build_expression_matrix_block(m, u, 16, 16)
        @test nnz(B) > 0                       # zero here was the bug

        B3 = Tarang.build_expression_matrix_block(
            Tarang.Multiply(q3, Tarang.Laplacian(u)), u, 16, 16)
        @test Matrix(B3) ≈ 3 .* Matrix(B) rtol = 1e-10

        # ... and an unrepresentable coefficient raises on this route too.
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        uf = ScalarField(dist, "u", (xb, yb), Float64)
        nf = ScalarField(dist, "nu_e", (xb, yb), Float64)
        ensure_layout!(nf, :g)
        fill!(Tarang.get_grid_data(nf), 0.5)
        n = Tarang._coeff_space_dofs(uf)
        @test_throws Tarang.ImplicitNCCError Tarang.build_expression_matrix_block(
            Tarang.Multiply(nf, Tarang.Laplacian(uf)), uf, n, n)
    end

    # ---------------------------------------------------------------------
    # 4. Regression: the answer is never silently independent of the coefficient.
    # ---------------------------------------------------------------------
    @testset "regression: result never silently independent of coefficient" begin
        outcomes = [vclhs_field_outcome(nu) for nu in (0.01, 0.5, 5.0)]
        statuses = [o[1] for o in outcomes]

        # Current contract: every unrepresentable coefficient is rejected up front.
        @test all(==(:error), statuses)

        # The historical signature of the bug: all three RAN and returned the SAME number
        # (1.0, no decay at all). Assert that combination can never occur again, phrased so
        # the test still guards correctly if a future change makes these cases solvable.
        if all(==(:ok), statuses)
            vals = [o[2] for o in outcomes]
            @test !(isapprox(vals[1], vals[2]; rtol=1e-9) &&
                    isapprox(vals[2], vals[3]; rtol=1e-9))
            # and in particular none of them may be the inviscid answer
            @test !any(v -> isapprox(v, 1.0; atol=1e-6), vals)
        end
    end

    # ---------------------------------------------------------------------
    # 5. _implicit_ncc_matrix unit contract: reports, never returns `nothing`.
    # ---------------------------------------------------------------------
    @testset "_implicit_ncc_matrix reports unsupported coefficients" begin
        # Representable: single Chebyshev axis -> a real matrix.
        q, _, _ = vclhs_cheb_field(z -> 1.0 + z; N=16)
        Q = Tarang._implicit_ncc_matrix(q)
        @test Q isa AbstractMatrix
        @test size(Q) == (16, 16)
        @test nnz(sparse(Q)) > 0

        # Pure-Fourier coefficient: unsupported, with a reason (never `nothing`).
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        nf = ScalarField(dist, "nu_e", (xb, yb), Float64)
        ensure_layout!(nf, :g)
        fill!(Tarang.get_grid_data(nf), 0.5)
        r = Tarang._implicit_ncc_matrix(nf)
        @test r isa Tarang.ImplicitNCCUnsupported
        @test r !== nothing
        @test occursin("nu_e", r.reason)
        @test occursin("Chebyshev/Jacobi", r.reason)

        # A DERIVATIVE of a field is not a bare field. Resolving through it to the leaf
        # would build a multiply-by-q matrix for a d(q)/dz coefficient — a silently wrong
        # operator — so it must be reported as unsupported instead.
        q2, _, coords2 = vclhs_cheb_field(z -> 1.0 + z; N=16, name="qq")
        dq = Tarang.Differentiate(q2, coords2["z"], 1)
        rd = Tarang._implicit_ncc_matrix(dq)
        @test rd isa Tarang.ImplicitNCCUnsupported
        @test occursin("plain field", rd.reason)

        # An identically-zero coefficient is representable and genuinely zero: it must give
        # a ZERO matrix, not a pass-through (which would silently mean q == 1).
        qz, _, _ = vclhs_cheb_field(_ -> 0.0; N=16, name="qzero")
        Qz = Tarang._implicit_ncc_matrix(qz)
        @test Qz isa AbstractMatrix
        @test nnz(sparse(Qz)) == 0
    end

    @testset "_apply_implicit_ncc raises only when a term would be lost" begin
        unsupported = Tarang.ImplicitNCCUnsupported("test reason")

        # A non-empty operand block would lose its coefficient -> must raise.
        child = Dict{Any, SparseMatrixCSC}("u" => sparse(ComplexF64(1) * I, 4, 4))
        @test_throws Tarang.ImplicitNCCError Tarang._apply_implicit_ncc(unsupported, child, "q*u")

        # Nothing to lose: an empty dict, or blocks that are already structurally zero,
        # must pass through quietly rather than raising a spurious error.
        @test Tarang._apply_implicit_ncc(unsupported, Dict{Any, SparseMatrixCSC}(), "q*u") ==
              Dict{Any, SparseMatrixCSC}()
        zero_child = Dict{Any, SparseMatrixCSC}("u" => spzeros(ComplexF64, 4, 4))
        @test Tarang._apply_implicit_ncc(unsupported, zero_child, "q*u") === zero_child

        # A shape-incompatible multiply matrix must raise too: passing the operand through
        # would drop the coefficient for a block that really carries the term.
        Q = sparse(ComplexF64(2) * I, 3, 3)
        @test_throws Tarang.ImplicitNCCError Tarang._apply_implicit_ncc(Q, child, "q*u")
    end

    # ---------------------------------------------------------------------
    # 6. Term labelling used in the error messages.
    # ---------------------------------------------------------------------
    @testset "_expr_label names terms readably" begin
        q, dom, coords = vclhs_cheb_field(z -> 1.0 + z; N=8)
        u = ScalarField(dom, "u")
        @test Tarang._expr_label(u) == "u"
        @test Tarang._expr_label(Tarang.Laplacian(u)) == "lap(u)"
        @test Tarang._expr_label(Tarang.MultiplyOperator(q, Tarang.Laplacian(u))) == "q*lap(u)"
        @test Tarang._expr_label(Tarang.Differentiate(u, coords["z"], 1)) == "∂z(u)"
        @test Tarang._expr_label(Tarang.NegateOperator(u)) == "-u"
        @test Tarang._expr_label(2.5) == "2.5"
    end
end
