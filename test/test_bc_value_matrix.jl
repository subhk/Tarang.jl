"""
Boundary-condition right-hand sides must be enforced with the VALUE they name.

Time- and space-dependent BCs work — measured exact to machine precision — but
nothing tested it. `test_boundary_conditions.jl` checks the `is_time_dependent` /
`is_space_dependent` FLAGS, and `test_bc_regression.jl` covers constant BCs, so a
regression that made `sin(2πt)` freeze at its `t = 0` value, or applied a
space-dependent profile along the wrong axis, would pass the whole suite.

That is the shape of every correctness bug this project has found: the flag is
tested, the value is not, and the failure returns a plausible number. A BC that
silently freezes at its initial value is especially quiet — the solve still runs,
still converges, and is simply solving a different problem.

Each case drives a real Fourier×Chebyshev IVP with a tau/lift formulation, steps
it, and compares the boundary row against the analytic right-hand side evaluated
at the FINAL simulation time. Evaluating at the final time is what makes the test
discriminating: `sin(2πt)` and `0.3t` are both zero at `t = 0`, so a BC pinned to
its first-step value would read as zero here and fail loudly, while a naive
comparison against `t = 0` would pass.

Every case also asserts that no "enforced as ZERO" warning was emitted. The BC
evaluator falls back to zero for right-hand sides it does not recognise, and that
fallback is loud but non-fatal — the run continues with the wrong boundary. Pinning
the absence of the warning catches a form that stops being recognised.
"""

using Test
using Tarang
using Logging

const _BCM_NX = 8
const _BCM_NZ = 16

"""Step a Fourier×Chebyshev diffusion problem with `b(z=0) = bcstr`.

Returns `(boundary_row, final_time, n_zero_warnings)`."""
function _bc_matrix_run(bcstr::AbstractString; nsteps = 50, dt = 0.004)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = _BCM_NX, bounds = (0.0, 2π))
    zb = ChebyshevT(coords["z"]; size = _BCM_NZ, bounds = (0.0, 1.0))
    domain = Domain(dist, (xb, zb))

    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    lb = derivative_basis(zb, 1)
    tau_lift(A) = lift(A, lb, -1)
    _, ez = unit_vector_fields(coords, dist)
    grad_b = grad(b) + ez * tau_lift(tau1)

    prob = IVP([b, tau1, tau2])
    add_parameters!(prob, kappa = 0.1, ez = ez, grad_b = grad_b, tau_lift = tau_lift)
    add_equation!(prob, "dt(b) - kappa*div(grad_b) + tau_lift(tau2) = 0")
    add_bc!(prob, "b(z=0) = $bcstr")
    add_bc!(prob, "b(z=1) = 0")

    local solver
    logs, _ = Test.collect_test_logs() do
        s = InitialValueSolver(prob, RK222(); dt = dt)
        for _ in 1:nsteps
            step!(s, dt)
        end
        solver = s
    end
    zero_warnings = count(logs) do l
        l.level == Logging.Warn && occursin("enforced as ZERO", string(l.message))
    end

    field = solver.state[1]
    ensure_layout!(field, :g)
    return Array(get_grid_data(field))[:, 1], solver.sim_time, zero_warnings
end

_bcm_xs() = [2π * (i - 1) / _BCM_NX for i in 1:_BCM_NX]

@testset "BC right-hand sides are enforced with the value they name" begin
    xs = _bcm_xs()

    # (label, BC string, analytic boundary value as a function of the final time)
    cases = (
        ("constant",          "0.7",                     t -> fill(0.7, _BCM_NX)),
        ("space only",        "1 + 0.5*cos(x)",          t -> 1 .+ 0.5 .* cos.(xs)),
        ("time only",         "sin(2*pi*t)",             t -> fill(sin(2π * t), _BCM_NX)),
        ("time, linear",      "0.3*t",                   t -> fill(0.3t, _BCM_NX)),
        ("time × space",      "(1 + 0.5*cos(x))*sin(t)", t -> (1 .+ 0.5 .* cos.(xs)) .* sin(t)),
        ("time + space",      "cos(x) + 0.4*t",          t -> cos.(xs) .+ 0.4t),
    )

    for (label, bcstr, want_fn) in cases
        @testset "$label — b(z=0) = $bcstr" begin
            got, tfinal, zero_warnings = _bc_matrix_run(bcstr)
            want = want_fn(tfinal)

            @test zero_warnings == 0
            @test tfinal > 0.1                      # actually stepped
            @test got ≈ want atol = 1e-10
        end
    end
end

@testset "A time-dependent BC tracks time rather than freezing at t=0" begin
    # The discriminating check, stated on its own because it is the regression that
    # a value comparison at t = 0 would miss entirely. `sin(2πt)` and `0.3t` are both
    # ZERO at t = 0, so a BC cached once at the first step and never refreshed reads
    # as an all-zero boundary — which is also what a dropped BC looks like.
    for (bcstr, at) in (("sin(2*pi*t)", t -> sin(2π * t)), ("0.3*t", t -> 0.3t))
        got, tfinal, _ = _bc_matrix_run(bcstr)
        @test tfinal > 0.1
        # Non-trivial at the final time, so the comparison below means something.
        @test abs(at(tfinal)) > 0.05
        @test maximum(abs, got) > 0.05          # would be 0 if frozen at t = 0
        @test got ≈ fill(at(tfinal), _BCM_NX) atol = 1e-10
    end
end

@testset "A space-dependent BC is applied along the right axis" begin
    # A profile applied along the wrong axis is constant in x instead of varying,
    # which a mean-based check would not catch. Assert the variation itself.
    got, _, _ = _bc_matrix_run("1 + 0.5*cos(x)")
    xs = _bcm_xs()
    want = 1 .+ 0.5 .* cos.(xs)
    @test maximum(got) - minimum(got) > 0.5     # genuinely varying, not flattened
    @test got ≈ want atol = 1e-10
end
