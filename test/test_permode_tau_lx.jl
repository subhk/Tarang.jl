"""
Per-mode scatter→gather must preserve 0-D tau DOFs (zero_dim_stash).

0-D tau fields carry length-0 sentinel storage, so a per-mode scatter has
nowhere to write their solved values; before the fix the next gather silently
ZERO-FILLED those slots. Any vector rebuilt by scatter→re-gather — the per-mode
IMEX steppers' LX stage history, the NLBVP Newton state — therefore dropped
every lift(tau) contribution and integrated a (slightly) different formula than
the global-matrix path. The `zero_dim_stash` on the subproblem runtime now
round-trips those DOFs exactly.

Regression for the 2026-08-20 MPI/parallel review finding V9. Note the review's
headline "2.19 divergence" repro turned out to compare against an invalidly
forced global path; on a resolved 1-D diffusion transient the actual value
impact of the dropped tau is ~1e-13. The roundtrip test below is the
discriminating regression (pre-fix: tau slots come back 0); the analytic checks
pin end-to-end stepper correctness on a tau-heavy transient.
"""

using Test
using Tarang
using LinearAlgebra

function _tau_lx_build(scheme; dt=1e-3)
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
    u = ScalarField(dist, "u", (zb,), Float64)
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    lb = derivative_basis(zb, 2)
    problem = IVP([u, tau1, tau2])
    add_parameters!(problem; lb=lb)
    add_equation!(problem, "∂t(u) - ∂z(∂z(u)) + lift(tau1, lb, -1) + lift(tau2, lb, -2) = 0")
    add_bc!(problem, "u(z=0) = 1")
    add_bc!(problem, "u(z=1) = 0")
    solver = InitialValueSolver(problem, scheme; dt=dt)
    return solver, u, tau1, tau2, dist, zb
end

# u(0,z)=0 with BCs {1,0}: u(t,z) = (1-z) - Σ_{n≥1} (2/(nπ)) e^{-n²π²t} sin(nπz)
function _tau_lx_analytic(z, t)
    s = 1.0 - z
    for n in 1:200
        s -= (2 / (n * pi)) * exp(-n^2 * pi^2 * t) * sin(n * pi * z)
    end
    return s
end

@testset "per-mode 0-D tau DOFs survive scatter→gather" begin
    @testset "roundtrip preserves every slot (incl. tau)" begin
        solver, u, tau1, tau2, dist, zb = _tau_lx_build(RK222())
        sps = Tarang._timestepper_subproblems(solver)
        @test sps !== nothing && length(sps) == 1
        sp = sps[1]
        n = size(sp.L_min, 2)
        x = ComplexF64.(randn(n))
        fields = Any[u, tau1, tau2]
        Tarang.scatter_inputs(sp, x, fields)
        x2 = Tarang.gather_inputs(sp, fields)
        # Pre-fix: the two tau slots came back as exactly 0 (scatter skipped
        # empty storage; gather zero-filled it).
        @test maximum(abs.(x .- x2)) < 1e-12
    end

    @testset "tau-heavy transient matches the analytic solution" begin
        for (name, scheme) in [("RK222", RK222()), ("SBDF2", SBDF2())]
            @testset "$name" begin
                solver, u, _, _, dist, zb = _tau_lx_build(scheme)
                run!(solver; stop_iteration=200, log_interval=10^6)
                ensure_layout!(u, :g)
                g = copy(vec(Tarang.get_grid_data(u)))
                zg = vec(collect(Tarang.local_grid(zb, dist, 1.0)))
                exact = _tau_lx_analytic.(zg, solver.sim_time)
                @test maximum(abs.(g .- exact)) < 1e-5
                @test maximum(abs.(g)) < 1.0 + 1e-6   # maximum principle
            end
        end
    end
end
