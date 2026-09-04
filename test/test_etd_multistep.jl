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

function _run_etd_stepper(ts; nsteps=12, dt=0.005, return_solver=false)
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
    return_solver && return solver
    ensure_layout!(u, :g)
    return copy(get_grid_data(u))
end

@testset "ETD multistep reuses the startup phi cache" begin
    for (ts, duplicate_key) in ((ETD_CNAB2(), :etd_cnab_phi),
                                (ETD_SBDF2(), :etd_sbdf_phi))
        solver = _run_etd_stepper(ts; nsteps=2, return_solver=true)
        cache = solver.timestepper_state.timestepper_data
        @test haskey(cache, :etd_phi)
        # ETDRK2 startup and the multistep phase use the same L and dt, so a
        # second three-matrix cache would double the dominant retained memory.
        @test !haskey(cache, duplicate_key)
    end
end

@testset "ETD phi cache follows replacement linear operators" begin
    state = Tarang.TimestepperState(ETD_RK222(), 0.1, ScalarField[])
    L1 = ComplexF64[-1 0; 0 -2]
    L2 = ComplexF64[-3 0; 0 -4]

    phi1 = Tarang._get_etd_phi!(state, L1, 0.1)
    phi2 = Tarang._get_etd_phi!(state, L2, 0.1)

    @test phi2 !== phi1
    @test phi2[1] ≈ exp(0.1 .* L2)
    @test state.timestepper_data[:etd_phi_L_source] === L2
end

@testset "ETD multistep (ETD-AB2)" begin
    cnab2 = _run_etd_stepper(ETD_CNAB2())
    sbdf2 = _run_etd_stepper(ETD_SBDF2())

    @test all(isfinite, cnab2)
    @test all(isfinite, sbdf2)
    # Same ETD-AB2 update + same startup ⇒ identical to roundoff.
    @test maximum(abs.(cnab2 .- sbdf2)) < 1e-10 * maximum(abs.(sbdf2))
end

@testset "phi_functions small-z accuracy (cancellation regression)" begin
    # The direct formulas (e^z−1−z)/z² etc. cancel catastrophically for small
    # |z|: with the old 1e-8 Taylor cutoff, φ₂'s relative error reached ~1% at
    # z=−1e-7 and φ₃'s exceeded 1e5(!). Low-|k| modes (z = −dt·ν·k²) sweep this
    # band in essentially every ETD run. The cutoff is now 1e-2 with extended
    # series. Reference values from BigFloat evaluation of the exact formulas.
    setprecision(BigFloat, 256) do
        for z in (-1e-7, -1e-6, -1e-5, -1e-4, -1e-3, -5e-3, -0.5, -5.0)
            φ0, φ1, φ2, φ3 = Tarang.phi_functions(z)
            bz = big(z)
            @test isapprox(φ0, Float64(exp(bz)); rtol=1e-12)
            @test isapprox(φ1, Float64((exp(bz) - 1) / bz); rtol=1e-11)
            @test isapprox(φ2, Float64((exp(bz) - 1 - bz) / bz^2); rtol=1e-11)
            @test isapprox(φ3, Float64((exp(bz) - 1 - bz - bz^2 / 2) / bz^3); rtol=1e-9)
        end
    end
end
