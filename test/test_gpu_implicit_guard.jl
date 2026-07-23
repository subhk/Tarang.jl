"""
CPU-runnable coverage for the single-GPU implicit-operator guard
(`_check_gpu_implicit_compatibility!`, dispatch.jl).

A pure-Fourier GPU IVP skips global-matrix/subproblem assembly, so a standard
IMEX/multistep/ETD scheme would silently drop an implicit LHS operator and
integrate the equation without it (a heat equation runs inviscid). The guard
turns that silent wrong answer into a loud error naming the diagonal-IMEX
schemes.

The guard FIRING needs a GPU field (is_gpu true) and is asserted in
test_gpu_timesteppers.jl on a GPU node. Here — with no GPU — we pin the two
architecture-independent halves: the implicit-term detector, and that the guard
is a no-op on CPU (a CPU implicit heat equation still steps and diffuses).
"""

using Test
using Tarang

@testset "GPU implicit-operator guard (CPU-side logic)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π))

    function _solver(eqn, ts)
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        xs = range(0, 2π, length=17)[1:16]
        g = Tarang.get_grid_data(u)
        for j in 1:16, i in 1:16
            g[i, j] = cos(2 * xs[i])
        end
        prob = IVP([u]; namespace=Dict("u" => u))
        Tarang.add_equation!(prob, eqn)
        return u, InitialValueSolver(prob, ts; dt=1e-3)
    end

    @testset "implicit-term detection" begin
        # implicit LHS operator -> true
        _, s_impl = _solver("dt(u) - 0.5*lap(u) = 0", SBDF2())
        @test Tarang._problem_has_implicit_linear_term(s_impl)
        # explicit RHS diffusion -> false (the term is on the RHS)
        _, s_expl = _solver("dt(u) = 0.5*lap(u)", SBDF2())
        @test !Tarang._problem_has_implicit_linear_term(s_expl)
        # nonlinear RHS only -> false
        _, s_nl = _solver("dt(u) = -u*dx(u)", RK222())
        @test !Tarang._problem_has_implicit_linear_term(s_nl)
    end

    @testset "guard is a no-op on CPU (heat equation still diffuses)" begin
        # A non-diagonal scheme with an implicit LHS operator on CPU must NOT be
        # refused — the CPU global-matrix path solves it. The guard only fires for
        # the GPU field path, so `step!` (which calls the guard) must complete and
        # the equation must diffuse. If the guard wrongly fired, step! would throw.
        u, solver = _solver("dt(u) - 0.5*lap(u) = 0", SBDF2())
        n0 = maximum(abs, Tarang.get_grid_data(u))
        for _ in 1:200
            step!(solver, 1e-3)                    # exercises the guard; must not throw
        end
        ensure_layout!(u, :g)
        ratio = maximum(abs, Tarang.get_grid_data(u)) / n0
        @test isapprox(ratio, exp(-0.5 * 4 * 0.2); rtol=1e-3)   # exp(−ν k² T), k=2
    end

    @testset "coupled 2D vorticity: DiagonalIMEX == global-matrix SBDF2" begin
        # The flagship 2D-turbulence GPU example is a COUPLED system (vorticity
        # transport with implicit viscosity + a streamfunction Poisson constraint +
        # velocity + gauge BC) on a pure-Fourier domain. On GPU that path builds no
        # global matrix, so it must use DiagonalIMEX (which the guard exempts). Pin
        # that DiagonalIMEX_SBDF2 handles the coupled constraints AND applies the
        # viscous term, matching global-matrix SBDF2 — i.e. the example is viscous,
        # not the silently-inviscid run it was with SBDF2 on GPU.
        using Statistics: mean
        function run_coupled(ts; N=32, nu=1e-3, dt=2e-3, nsteps=60)
            u_field = ScalarField
            xb2 = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
            yb2 = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
            dom = Domain(dist, (xb2, yb2))
            q = ScalarField(dom, "q"); psi = ScalarField(dom, "psi")
            vel = VectorField(dom, "u"); tau = ScalarField(dist, "tau", (), Float64)
            prob = IVP([q, psi, vel, tau]); add_parameters!(prob; nu=nu)
            add_equation!(prob, "∂t(q) - nu*Δ(q) = -u⋅∇(q)")
            add_equation!(prob, "Δ(psi) + tau - q = 0")
            add_equation!(prob, "u - skew(grad(psi)) = 0")
            add_bc!(prob, "integ(psi) = 0")
            s = InitialValueSolver(prob, ts; dt=dt)
            xc = Tarang.get_grid_coordinates(dom; on_device=false)["x"]
            yc = Tarang.get_grid_coordinates(dom; on_device=false)["y"]
            q0 = zeros(N, N)
            for kx in 1:6, ky in 1:6
                k = hypot(kx, ky); (4 <= k <= 8) || continue
                @. q0 += sin(kx * xc + 0.3) * cos(ky * yc' + 0.3)
            end
            q0 .*= 10.0 / maximum(abs, q0); q["g"] = q0
            Tarang.evaluate_rhs(s, s.state, 0.0)
            ensure_layout!(q, :g); Z0 = 0.5 * mean(get_grid_data(q) .^ 2)
            for _ in 1:nsteps; step!(s, dt); end
            ensure_layout!(q, :g)
            return Z0, 0.5 * mean(get_grid_data(q) .^ 2)
        end
        Z0_ref, Zf_ref = run_coupled(SBDF2())
        Z0_dia, Zf_dia = run_coupled(DiagonalIMEX_SBDF2())
        @test Zf_ref / Z0_ref < 0.999                      # global-matrix run is viscous
        @test isapprox(Zf_dia, Zf_ref; rtol=1e-8)          # DiagonalIMEX matches it
        @test Zf_dia / Z0_dia < 0.999                      # ... and is therefore viscous too
    end

    @testset "Fourier×Chebyshev channel is exempt from the guard (builds subproblems)" begin
        # A wall-bounded 2D channel (Fourier x × Chebyshev z) is the `_gpu_coupled_state`
        # path: it builds per-Fourier-mode subproblems (Chebyshev BVPs) that DO solve the
        # implicit operator per mode. The GPU implicit-guard must therefore NOT refuse it —
        # its exemption keys on subproblems being present. This pins that the guard does not
        # break wall-bounded 2D GPU IVPs (which, unlike pure-Fourier, need no diagonal-IMEX).
        cheb = CartesianCoordinates("x", "z")
        cdist = Distributor(cheb; dtype=Float64)
        xbc = RealFourier(cheb["x"]; size=16, bounds=(0.0, 2π))
        zbc = ChebyshevT(cheb["z"]; size=16, bounds=(-1.0, 1.0))
        dom = Domain(cdist, (xbc, zbc))
        w = ScalarField(dom, "w")
        prob = IVP([w]; namespace=Dict("w" => w)); add_parameters!(prob; nu=0.1)
        Tarang.add_equation!(prob, "dt(w) - nu*lap(w) = 0")
        s = InitialValueSolver(prob, SBDF2(); dt=1e-3)

        @test haskey(prob.parameters, "subproblems")                 # coupled path built them
        @test prob.parameters["subproblems"] !== nothing
        @test Tarang._problem_has_implicit_linear_term(s)            # it does have an implicit L

        # and it actually diffuses on CPU (baseline correctness of the channel path)
        ensure_layout!(w, :g); g = Tarang.get_grid_data(w)
        zc = Tarang.get_grid_coordinates(dom; on_device=false)["z"]
        for k in axes(g, 2), i in axes(g, 1)
            g[i, k] = sin(2π * (i - 1) / 16) * (1 - zc[k]^2)
        end
        n0 = maximum(abs, g)
        for _ in 1:100; step!(s, 1e-3); end
        ensure_layout!(w, :g)
        @test maximum(abs, Tarang.get_grid_data(w)) / n0 < 0.999      # viscous, not dropped
    end
end
