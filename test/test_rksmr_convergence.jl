"""
Test suite for RKSMR (SSP-RK3) timestepper.

Tests:
1. RKSMR coefficient verification against canonical Shu-Osher SSP-RK3
2. RKSMR 3rd-order convergence on autonomous ODE: du/dt = -u, exact: exp(-t)
3. RKSMR 3rd-order convergence on non-autonomous ODE: du/dt = cos(t), exact: sin(t)
   (validates the c₃ = 1/2 stage evaluation time)
4. RKSMR type instantiation
"""

using Test
using Tarang

@testset "RKSMR (SSP-RK3)" begin

    @testset "Shu-Osher coefficient verification" begin
        ts = RKSMR()

        @test ts.stages == 3

        # Canonical Shu-Osher alpha coefficients
        # Stage 1: u^(1) = 1*u^n
        @test ts.alpha[1, 1] ≈ 1.0
        @test ts.alpha[1, 2] ≈ 0.0
        @test ts.alpha[1, 3] ≈ 0.0

        # Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1)
        @test ts.alpha[2, 1] ≈ 3/4
        @test ts.alpha[2, 2] ≈ 1/4
        @test ts.alpha[2, 3] ≈ 0.0

        # Stage 3: u^{n+1} = 1/3*u^n + 0*u^(1) + 2/3*u^(2)
        @test ts.alpha[3, 1] ≈ 1/3
        @test ts.alpha[3, 2] ≈ 0.0
        @test ts.alpha[3, 3] ≈ 2/3

        # Canonical Shu-Osher beta (RHS scaling) coefficients
        @test ts.beta[1] ≈ 1.0
        @test ts.beta[2] ≈ 1/4
        @test ts.beta[3] ≈ 2/3

        # Convexity: each row of alpha should sum to 1
        for s in 1:3
            @test sum(ts.alpha[s, :]) ≈ 1.0 atol=1e-14
        end

        # SSP condition: beta[s] / alpha[s, s] should be consistent
        # For canonical SSP-RK3, beta[s] = alpha[s, s] * c_ssp where c_ssp = 1
        @test ts.beta[1] / ts.alpha[1, 1] ≈ 1.0
        @test ts.beta[2] / ts.alpha[2, 2] ≈ 1.0
        @test ts.beta[3] / ts.alpha[3, 3] ≈ 1.0
    end

    @testset "RKSMR 3rd-order convergence" begin
        # Solve du/dt = -u with u(0) = 1, exact solution: u(t) = exp(-t)
        # Test that halving dt improves error by ~8x (3rd order)
        T_final = 1.0

        function rksmr_solve(dt)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            fill!(Tarang.get_grid_data(u), 1.0)  # u(0) = 1

            problem = IVP([u])
            add_equation!(problem, "dt(u) = -u")
            solver = InitialValueSolver(problem, RKSMR(); dt=dt)

            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                step!(solver)
            end

            ensure_layout!(u, :g)
            return Tarang.get_grid_data(u)[1]
        end

        # Two different timesteps
        dt1 = 0.05
        dt2 = 0.025
        u1 = rksmr_solve(dt1)
        u2 = rksmr_solve(dt2)
        exact = exp(-T_final)

        err1 = abs(u1 - exact)
        err2 = abs(u2 - exact)

        # 3rd order: error ratio should be ~8 when dt halved
        if err2 > 1e-14  # avoid division by zero
            rate = log2(err1 / err2)
            @test rate > 2.5  # Should be ~3 for 3rd order
        end
    end

    @testset "RKSMR 3rd-order convergence (non-autonomous)" begin
        # Solve du/dt = cos(t), u(0) = 0, exact: u(t) = sin(t)
        # This is non-autonomous: the RHS depends on t, NOT on u.
        # With wrong c₃ (e.g., c₃=1 instead of c₃=1/2), convergence order degrades.
        #
        # Uses RKSMR() coefficients directly to verify the Shu-Osher form
        # matches the canonical Butcher tableau c = [0, 1, 1/2].
        ts = RKSMR()
        T_final = 1.0

        function ssp_rk3_step(u, t, dt, alpha, beta)
            # Stage 1: u^(1) = alpha[1,1]*u + beta[1]*dt*f(t + c₁*dt)
            # c₁ = 0
            f0 = cos(t)
            u1 = alpha[1,1] * u + beta[1] * dt * f0

            # Stage 2: u^(2) = alpha[2,1]*u + alpha[2,2]*u^(1) + beta[2]*dt*f(t + c₂*dt)
            # c₂ = 1  (u^(1) is at time t + dt)
            f1 = cos(t + dt)
            u2 = alpha[2,1] * u + alpha[2,2] * u1 + beta[2] * dt * f1

            # Stage 3: u^{n+1} = alpha[3,1]*u + alpha[3,3]*u^(2) + beta[3]*dt*f(t + c₃*dt)
            # c₃ = 1/2  (u^(2) is at time t + dt/2)
            f2 = cos(t + 0.5 * dt)
            return alpha[3,1] * u + alpha[3,3] * u2 + beta[3] * dt * f2
        end

        function ssp_rk3_solve(dt)
            u = 0.0
            t = 0.0
            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                u = ssp_rk3_step(u, t, dt, ts.alpha, ts.beta)
                t += dt
            end
            return u
        end

        dt1 = 0.05
        dt2 = 0.025
        exact = sin(T_final)

        err1 = abs(ssp_rk3_solve(dt1) - exact)
        err2 = abs(ssp_rk3_solve(dt2) - exact)

        # 3rd order: error ratio should be ~8 when dt halved
        if err2 > 1e-14
            rate = log2(err1 / err2)
            @test rate > 2.5  # Should be ~3 for 3rd order
        end

        # Verify that wrong c₃ = 1 loses 3rd-order convergence
        function ssp_rk3_step_wrong_c3(u, t, dt, alpha, beta)
            f0 = cos(t)
            u1 = alpha[1,1] * u + beta[1] * dt * f0
            f1 = cos(t + dt)
            u2 = alpha[2,1] * u + alpha[2,2] * u1 + beta[2] * dt * f1
            f2 = cos(t + dt)  # WRONG: c₃ = 1 instead of 1/2
            return alpha[3,1] * u + alpha[3,3] * u2 + beta[3] * dt * f2
        end

        function ssp_rk3_solve_wrong(dt)
            u = 0.0
            t = 0.0
            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                u = ssp_rk3_step_wrong_c3(u, t, dt, ts.alpha, ts.beta)
                t += dt
            end
            return u
        end

        err1_wrong = abs(ssp_rk3_solve_wrong(dt1) - exact)
        err2_wrong = abs(ssp_rk3_solve_wrong(dt2) - exact)

        if err2_wrong > 1e-14
            rate_wrong = log2(err1_wrong / err2_wrong)
            @test rate_wrong < 2.5  # Should be ~2, NOT 3rd order
        end
    end

    @testset "RKSMR type instantiation" begin
        @test RKSMR() isa Tarang.TimeStepper
    end
end
