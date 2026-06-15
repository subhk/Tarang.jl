# ETD multistep (ETD-AB2) correctness guard.
#
# ETD_CNAB2 and ETD_SBDF2 are both the canonical 2nd-order ETD-AB2 update
#   u_{n+1} = exp(hL)·uₙ + h[φ₁·Nₙ + w·φ₂·(Nₙ − Nₙ₋₁)]
# with the same ETDRK2 startup, so on the same IVP they must agree to roundoff.
#
# This guards a real bug: ETD_CNAB2 previously applied a SINGLE φ₁ to the AB2
# extrapolation ((1+w/2)Nₙ − (w/2)Nₙ₋₁), which is 2nd order only as hL→0 and lost
# accuracy for stiff L (the ETD regime — ~10× larger single-step error at |hL|~10).
# If that regression returns, CNAB2 will diverge from SBDF2 and this test fails.

using Test
using Tarang

function _run_etd_stepper(ts; nsteps=12, dt=0.005)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
    dom = Domain(dist, (xb,))
    u = ScalarField(dom, "u")
    ensure_layout!(u, :g)
    xs = collect(0:31) .* (2π / 32)
    get_grid_data(u) .= 0.5 .* sin.(xs) .+ 0.1 .* cos.(2 .* xs)
    prob = IVP([u])
    add_equation!(prob, "∂t(u) = 0.2*lap(u) - u*∂x(u)")   # stiff diffusion + nonlinear advection
    solver = InitialValueSolver(prob, ts; dt=dt)
    for _ in 1:nsteps
        step!(solver, dt)
    end
    ensure_layout!(u, :g)
    return copy(get_grid_data(u))
end

@testset "ETD multistep (ETD-AB2)" begin
    cnab2 = _run_etd_stepper(ETD_CNAB2())
    sbdf2 = _run_etd_stepper(ETD_SBDF2())

    @test all(isfinite, cnab2)
    @test all(isfinite, sbdf2)
    # Same ETD-AB2 update + same startup ⇒ identical to roundoff.
    @test maximum(abs.(cnab2 .- sbdf2)) < 1e-10 * maximum(abs.(sbdf2))
end
