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
end
