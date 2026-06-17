"""
Test suite for RKSMR — the Spalart–Moser–Rogers semi-implicit (IMEX) RK3.

RKSMR was previously a fully-EXPLICIT SSP-RK3 (Shu-Osher) that ignored the stiff
linear operator L entirely (it only warned). It is now the genuine SMR IMEX
scheme: the nonlinear term F is treated explicitly (3rd order) and the linear
term L implicitly (Crank–Nicolson, 2nd order, stable for stiff diffusion), stored
as a 4-stage ESDIRK additive-Runge–Kutta (ARK) tableau and run through the shared
`step_rk_imex!` driver.

Tests:
1. ARK tableau structure + consistency (row sums = c, b sums = 1, stiffly accurate).
2. Tableau-level order (no Tarang): explicit part 3rd order, implicit part 2nd
   order, and L-stability on a stiff linear problem.
3. End-to-end heat equation in Tarang: 2nd-order accuracy AND stability on a stiff
   mode whose dt·λ is well outside the explicit SSP-RK3 stability region (the old
   explicit RKSMR blew up here).
4. Type instantiation.
"""

using Test
using Tarang

@testset "RKSMR (Spalart–Moser–Rogers IMEX-RK3)" begin

    @testset "ARK tableau structure + consistency" begin
        ts = RKSMR()
        @test ts.stages == 4

        # Stage times are the row sums of each tableau (explicit and implicit agree).
        for s in 1:4
            @test isapprox(sum(ts.A_explicit[s, :]), ts.c_explicit[s]; atol=1e-13)
            @test isapprox(sum(ts.A_implicit[s, :]), ts.c_implicit[s]; atol=1e-13)
        end
        @test isapprox(ts.c_explicit, [0.0, 8/15, 2/3, 1.0]; atol=1e-13)

        # Consistency: both weight vectors sum to 1 (1st-order condition).
        @test isapprox(sum(ts.b_explicit), 1.0; atol=1e-13)
        @test isapprox(sum(ts.b_implicit), 1.0; atol=1e-13)

        # Stiffly accurate: weights equal the last stage row.
        @test ts.b_explicit == ts.A_explicit[4, :]
        @test ts.b_implicit == ts.A_implicit[4, :]

        # Explicit first stage (ERK) and lower-triangular tableaux.
        @test all(ts.A_explicit[1, :] .== 0)
        @test all(ts.A_implicit[1, :] .== 0)
        for s in 1:4, j in (s+1):4
            @test ts.A_explicit[s, j] == 0
            @test ts.A_implicit[s, j] == 0
        end
        # Implicit diagonal carries the SMR β = (37/160, 5/24, 1/6) on stages 2..4.
        @test isapprox([ts.A_implicit[s, s] for s in 2:4], [37/160, 5/24, 1/6]; atol=1e-13)
    end

    @testset "tableau order: explicit 3rd, implicit 2nd, stiff stable" begin
        # Reference ARK integrator for  dX/dt = F(t,X) - L*X  (M = 1).
        ts = RKSMR()
        AE, AI = ts.A_explicit, ts.A_implicit
        bE, bI, c = ts.b_explicit, ts.b_implicit, ts.c_explicit
        function ark_step(Xn, t, dt, L, F)
            X = zeros(4); Fx = zeros(4)
            for s in 1:4
                rhs = Xn
                for j in 1:s-1
                    rhs += dt*AE[s,j]*Fx[j] - dt*AI[s,j]*L*X[j]
                end
                X[s] = rhs / (1 + dt*AI[s,s]*L)
                Fx[s] = F(t + c[s]*dt, X[s])
            end
            Xnp1 = Xn
            for s in 1:4
                Xnp1 += dt*bE[s]*Fx[s] - dt*bI[s]*L*X[s]
            end
            return Xnp1
        end
        solve(dt; L, F, X0, T=1.0) = begin
            X = X0; t = 0.0
            for _ in 1:round(Int, T/dt); X = ark_step(X, t, dt, L, F); t += dt; end
            X
        end
        rate(e1, e2) = log2(e1/e2)

        # Explicit only: dX/dt = cos(t), X0=0 -> sin(t). Expect 3rd order.
        e1 = abs(solve(0.05;  L=0.0, F=(t,x)->cos(t), X0=0.0) - sin(1.0))
        e2 = abs(solve(0.025; L=0.0, F=(t,x)->cos(t), X0=0.0) - sin(1.0))
        @test rate(e1, e2) > 2.7

        # Implicit only: dX/dt = -X (L=1), X0=1 -> exp(-1). Expect ~2nd order (CN).
        e1 = abs(solve(0.05;  L=1.0, F=(t,x)->0.0, X0=1.0) - exp(-1.0))
        e2 = abs(solve(0.025; L=1.0, F=(t,x)->0.0, X0=1.0) - exp(-1.0))
        @test 1.7 < rate(e1, e2) < 2.5

        # Stiff: dX/dt = -100X with dt=0.05 (dt·λ = -5, outside explicit SSP-RK3
        # stability). Implicit treatment must keep it bounded and decaying.
        xs = solve(0.05; L=100.0, F=(t,x)->0.0, X0=1.0)
        @test isfinite(xs) && abs(xs) < 1e-3
    end

    @testset "Tarang heat equation: 2nd-order accuracy + stiff stability" begin
        function heat_run(k, dt, T)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            xs = collect(range(0, 2π, length=33))[1:32]
            Tarang.get_grid_data(u) .= cos.(k .* xs)
            prob = IVP([u])
            add_equation!(prob, "dt(u) - lap(u) = 0")   # dt(u)=Δu ; mode k -> exp(-k^2 t)
            solver = InitialValueSolver(prob, RKSMR(); dt=dt)
            for _ in 1:round(Int, T/dt); step!(solver); end
            ensure_layout!(u, :g)
            return Tarang.get_grid_data(u)[1], cos(k*xs[1])
        end

        T = 0.5; k = 2
        v1, c0 = heat_run(k, 0.05, T)
        v2, _  = heat_run(k, 0.025, T)
        exact = exp(-k^2 * T) * c0
        e1, e2 = abs(v1 - exact), abs(v2 - exact)
        @test e1 < 1e-2
        @test log2(e1 / e2) > 1.7        # ~2nd order (implicit linear treatment)

        # Stiff mode k=10 (λ=-100): dt=0.05 gives dt·λ=-5 — the old explicit RKSMR
        # blew up here. IMEX must stay bounded and decay toward zero.
        vstiff, _ = heat_run(10, 0.05, 0.5)
        @test isfinite(vstiff) && abs(vstiff) < 1e-3
    end

    @testset "RKSMR type instantiation" begin
        @test RKSMR() isa Tarang.TimeStepper
    end
end
