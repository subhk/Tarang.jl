"""
A BVP whose operator cannot be assembled must refuse, not return a confident
wrong answer.

`subproblem_matrix_build.jl` validated each equation×variable block against its
slot and, on a mismatch, logged `@error` and `continue`d — "skip this block to
avoid crash". Skipping drops an entire coupling out of the operator and then
solves what remains, so the caller gets a plausible array back with the failure
buried in the log.

The reachable case: a pure-Fourier LBVP. A periodic axis has no boundary, so a
point boundary condition like `u(x=0) = 0` has nowhere to place its tau row and
produces a full (N,N) block where a single row is expected. `Δu = f` with
`f = sin(x)` then came back as **exactly zero** instead of `-sin(x)` — no error
raised, only a log line. Every pre-existing field-RHS BVP test uses a mixed
Fourier×Chebyshev basis, which has a real boundary and assembles cleanly, which is
why this went unseen.

These tests assert on VALUES and on refusal, because a zero solution is
indistinguishable from a correct one by any shape, type, or did-it-throw check.
"""

using Test
using Tarang

@testset "A pure-Fourier LBVP with a point BC refuses instead of returning zero" begin
    N = 16
    domain = PeriodicDomain(N)
    u = ScalarField(domain, "u")
    f = ScalarField(domain, "f")
    set!(f, (x,) -> sin(x))

    prob = LBVP([u])
    add_parameters!(prob, f = f)
    add_equation!(prob, "lap(u) = f")
    add_bc!(prob, "u(x=0) = 0")

    err = try
        solver = BoundaryValueSolver(prob)
        solve!(solver)
        nothing
    catch e
        e
    end

    @test err !== nothing
    msg = sprint(showerror, err)
    # The message must name what did not fit and why no answer is being returned —
    # the old behaviour logged exactly this text and then returned zero anyway.
    @test occursin("Block size mismatch", msg)
    @test occursin("cannot be assembled", msg)
    @test occursin("periodic", msg)
end

@testset "The refusal does not depend on the RHS form" begin
    # The forcing was dropped for a field, a scaled field, a sum and a negation
    # alike. Whatever the RHS, an unassemblable operator must refuse rather than
    # return whichever wrong answer the truncated system happens to produce.
    N = 16
    for rhs in ("f", "2*f", "c*f", "-f", "f + g")
        domain = PeriodicDomain(N)
        u = ScalarField(domain, "u")
        f = ScalarField(domain, "f"); set!(f, (x,) -> sin(x))
        g = ScalarField(domain, "g"); set!(g, (x,) -> cos(2x))

        prob = LBVP([u])
        add_parameters!(prob, f = f, g = g, c = 3.0)
        add_equation!(prob, "lap(u) = $rhs")
        add_bc!(prob, "u(x=0) = 0")

        @test_throws ErrorException begin
            solver = BoundaryValueSolver(prob)
            solve!(solver)
        end
    end
end

@testset "A coupled Fourier×Chebyshev LBVP still solves to its analytic answer" begin
    # The guard now throws where it used to skip, so pin the case that legitimately
    # assembles: every block fits, and the solve must be unaffected.
    Nx, Nz, Lz = 8, 24, 1.0
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype = Float64, device = CPU())
    xb = RealFourier(coords["x"]; size = Nx, bounds = (0.0, 2π), dealias = 1.0)
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, Lz), dealias = 1.0)
    dom = Domain(dist, (xb, zb))

    u = ScalarField(dom, "u")
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    fld = ScalarField(dom, "f")
    lb2 = derivative_basis(zb, 2)

    xg = vec(Array(Tarang.local_grid(xb, dist, 1)))
    zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
    uex(x, z) = sin(π * z / Lz) * cos(2x)
    λ = (π / Lz)^2 + 4
    ensure_layout!(fld, :g)
    fd = get_grid_data(fld)
    for i in 1:Nx, k in 1:Nz
        fd[i, k] = -λ * uex(xg[i], zg[k])
    end

    prob = LBVP([u, tau1, tau2])
    prob.namespace["f"] = fld
    add_parameters!(prob; Lz = Lz, l1 = lift(tau1, lb2, -1), l2 = lift(tau2, lb2, -2))
    add_equation!(prob, "Δ(u) + l1 + l2 = f")
    add_bc!(prob, "u(z=0) = 0")
    add_bc!(prob, "u(z=Lz) = 0")

    solver = BoundaryValueSolver(prob)
    solve!(solver)
    ensure_layout!(u, :g)
    g = Array(get_grid_data(u))

    @test maximum(abs, g) > 0.5          # non-trivial, so the comparison below means something
    for i in 1:Nx, k in 1:Nz
        @test isapprox(g[i, k], uex(xg[i], zg[k]); atol = 1e-9)
    end
end

@testset "_bvp_bulk_target_vars picks a container for a pure-Fourier state" begin
    # Independent of the assembly guard: the BVP forcing gather chose its per-equation
    # containers from the NON-Fourier variables only, so a pure-Fourier state produced
    # an empty list and no equation ever received its forcing. Selecting on that alone
    # is wrong regardless of whether the operator assembles.
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    @test Tarang._bvp_bulk_target_vars([u]) == [1]

    # A coupled state still prefers the non-Fourier variable, and 0-D tau variables
    # are never valid containers.
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype = Float64, device = CPU())
    xb = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
    zb = ChebyshevT(coords["z"]; size = 12, bounds = (0.0, 1.0))
    dom = Domain(dist, (xb, zb))
    v = ScalarField(dom, "v")
    tau = ScalarField(dist, "tau", (), Float64)
    @test Tarang._bvp_bulk_target_vars([v, tau]) == [1]

    pure = ScalarField(Domain(dist, (xb, RealFourier(coords["z"]; size = 8, bounds = (0.0, 2π)))), "w")
    @test Tarang._bvp_bulk_target_vars([pure, tau]) == [1]
end
