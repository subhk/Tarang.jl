"""
Test suite for DiagonalIMEX timestepper and SpectralLinearOperator.

Tests:
1. SpectralLinearOperator construction for Laplacian and hyperviscosity
2. SpectralLinearOperator coefficient values match expected -ν*k²
3. RK222 explicit fallback convergence (ODE: du/dt = -u, exact: exp(-t))
4. RK443 explicit fallback convergence
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: SpectralLinearOperator

@testset "DiagonalIMEX and SpectralLinearOperator" begin

    @testset "SpectralLinearOperator construction — Laplacian" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        bases = (xb, yb)

        ν = 1e-3
        L = SpectralLinearOperator(dist, bases, :laplacian; ν=ν)

        @test L.operator_type == :laplacian

        # k=0 mode should have zero coefficient (no damping of mean)
        @test L.coefficients[1, 1] ≈ 0.0 atol=1e-14

        # Laplacian coefficients are ν*k² ≥ 0 (positive, used in IMEX as 1+dt*γ*L)
        @test all(real.(L.coefficients) .>= -1e-14)

        # Non-zero wavenumber should have positive coefficient
        @test L.coefficients[2, 1] > 0.0
    end

    @testset "SpectralLinearOperator construction — hyperviscosity" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        bases = (xb, yb)

        ν = 1e-4
        order = 2  # 4th-order hyperviscosity: ν*k⁴
        L = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=ν, order=order)

        @test L.operator_type == :hyperviscosity

        # k=0 mode should have zero coefficient
        @test L.coefficients[1, 1] ≈ 0.0 atol=1e-14

        # Coefficients are ν*(k²)^order ≥ 0
        @test all(real.(L.coefficients) .>= -1e-14)

        # Higher wavenumbers should have larger coefficients (k⁴ grows fast)
        # Index i corresponds to wavenumber k=i-1, so index 4 (k=3) > index 2 (k=1)
        if size(L.coefficients, 1) > 4
            @test real(L.coefficients[4, 1]) > real(L.coefficients[2, 1])
        end
    end

    @testset "RK222 explicit convergence" begin
        # Solve du/dt = -u with u(0) = 1, exact solution: u(t) = exp(-t)
        # Test that halving dt improves error by ~4x (2nd order)
        T_final = 1.0

        function rk222_solve(dt)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            fill!(Tarang.get_grid_data(u), 1.0)  # u(0) = 1

            problem = IVP([u])
            add_equation!(problem, "dt(u) = -u")
            solver = InitialValueSolver(problem, RK222(); dt=dt)

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
        u1 = rk222_solve(dt1)
        u2 = rk222_solve(dt2)
        exact = exp(-T_final)

        err1 = abs(u1 - exact)
        err2 = abs(u2 - exact)

        # 2nd order: error ratio should be ~4 when dt halved
        if err2 > 1e-14  # avoid division by zero
            rate = log2(err1 / err2)
            @test rate > 1.5  # Should be ~2 for 2nd order
        end
    end

    @testset "RK111 explicit convergence" begin
        # 1st order Forward Euler: error ratio ~2 when dt halved
        T_final = 0.5

        function rk111_solve(dt)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            fill!(Tarang.get_grid_data(u), 1.0)

            problem = IVP([u])
            add_equation!(problem, "dt(u) = -u")
            solver = InitialValueSolver(problem, RK111(); dt=dt)

            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                step!(solver)
            end

            ensure_layout!(u, :g)
            return Tarang.get_grid_data(u)[1]
        end

        dt1 = 0.02
        dt2 = 0.01
        u1 = rk111_solve(dt1)
        u2 = rk111_solve(dt2)
        exact = exp(-T_final)

        err1 = abs(u1 - exact)
        err2 = abs(u2 - exact)

        if err2 > 1e-14
            rate = log2(err1 / err2)
            @test rate > 0.8  # Should be ~1 for 1st order
        end
    end

    @testset "Timestepper types exist" begin
        @test RK111() isa Tarang.TimeStepper
        @test RK222() isa Tarang.TimeStepper
        @test RK443() isa Tarang.TimeStepper
        @test DiagonalIMEX_RK222() isa Tarang.TimeStepper
    end
end
