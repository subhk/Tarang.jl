using Test
using Tarang

@testset "Documented timestepper stability limits" begin
    @testset "RKGFY has Crank-Nicolson stiff amplification" begin
        for h in (0.1, 10.0, 1.0e4)
            u = ScalarField(PeriodicDomain(8), "u")
            set!(u, 1.0)
            problem = InitialValueProblem([u])
            add_parameters!(problem; decay=h / 0.1)
            add_equation!(problem, "dt(u) + decay*u = 0")
            solver = InitialValueSolver(problem, Tarang.RKGFY(); dt=0.1)
            step!(solver)
            expected = (1 - h / 2) / (1 + h / 2)
            @test maximum(abs, Array(grid_data!(u)) .- expected) < 1e-12
        end
    end

    @testset "CNLF2 explicit decay retains its leapfrog parasitic mode" begin
        h = 0.2
        u = ScalarField(PeriodicDomain(8), "u")
        set!(u, 1.0)
        problem = InitialValueProblem([u])
        add_equation!(problem, "dt(u) = -2*u")
        solver = InitialValueSolver(problem, Tarang.CNLF2(); dt=0.1)
        # y[n+1] = y[n-1] - 2h*y[n], with Euler startup y[1] = 1-h.
        r_plus, r_minus = -h + sqrt(1 + h^2), -h - sqrt(1 + h^2)
        c_plus = (1 - h - r_minus) / (r_plus - r_minus)
        @test abs(r_minus) > 1
        for n in 1:6
            step!(solver)
            expected = c_plus * r_plus^n + (1 - c_plus) * r_minus^n
            @test maximum(abs, Array(grid_data!(u)) .- expected) < 1e-12
        end
    end
end

@testset "Architecture-independent timestepper API" begin
    for name in (:RK222, :RK443, :SBDF2)
        @test name in names(Tarang)
    end
    for name in (:DiagonalIMEX_RK222, :DiagonalIMEX_RK443, :DiagonalIMEX_SBDF2)
        @test !(name in names(Tarang))
        @test !(name in names(Tarang.Timesteppers))
        @test !Tarang.is_public_api(name)
    end
end

@testset "Single scheme per timestepper" begin
    for name in (:WeightedRK222, :AlexanderRK443, :WeightedARS222, :ThetaCNAB2)
        @test !isdefined(Tarang, name)
    end
    @test_throws MethodError RK222(variant=:weighted)
    @test_throws MethodError RK443(variant=:alexander)
    @test_throws MethodError Tarang.RKGFY(variant=:ars_weighted)
    @test_throws MethodError Tarang.MCNAB2(0.5)
    @test_throws MethodError SBDF3(startup=:rk443)
    @test_throws MethodError SBDF4(startup=:rk443)
end

# Reference tableaux include the initial stage.
@testset "Reference timestepper coefficients" begin
    g = 1 - 1 / sqrt(2)
    d = 1 - 1 / (2g)
    references = (
        (RK222(), [0.0 0 0; g 0 0; d 1-d 0],
         [0.0 0 0; 0 g 0; 0 1-g g], [0.0, g, 1]),
        (RK443(), [0.0 0 0 0 0; 1/2 0 0 0 0; 11/18 1/18 0 0 0;
                   5/6 -5/6 1/2 0 0; 1/4 7/4 3/4 -7/4 0],
         [0.0 0 0 0 0; 0 1/2 0 0 0; 0 1/6 1/2 0 0;
          0 -1/2 1/2 1/2 0; 0 3/2 -3/2 1/2 1/2], [0.0, 1/2, 2/3, 1/2, 1]),
        (Tarang.RKGFY(), [0.0 0 0; 1 0 0; 1/2 1/2 0],
         [0.0 0 0; 1/2 1/2 0; 1/2 0 1/2], [0.0, 1, 1]),
    )
    for (ts, A, H, c) in references
        @test ts.A_explicit ≈ A
        @test ts.A_implicit ≈ H
        @test ts.c_explicit ≈ c
        @test ts.b_explicit ≈ A[end, :]
        @test ts.b_implicit ≈ H[end, :]
    end
    for (cpu, gpu) in ((RK222(), Tarang.DiagonalIMEX_RK222()), (RK443(), Tarang.DiagonalIMEX_RK443()))
        @test cpu.A_explicit == gpu.A_explicit
        @test cpu.A_implicit == gpu.A_implicit
        @test cpu.b_explicit == gpu.b_explicit
        @test cpu.b_implicit == gpu.b_implicit
    end
    for (dt, previous) in ((0.1, 0.1), (0.1, 0.07))
        w = dt / previous
        a, b, c = Tarang._mcnab2_coefs(dt, previous)
        @test a == (1/dt, -1/dt)
        @test b == ((8+1/w)/16, (7-1/w)/16, 1/16)
        @test c == (0.0, 1+w/2, -w/2)
    end
end

@testset "RK222 explicit one-step reference" begin
    u = ScalarField(PeriodicDomain(8), "u")
    set!(u, (x,) -> 1.0)
    problem = InitialValueProblem([u])
    add_equation!(problem, "dt(u) = -u")
    solver = InitialValueSolver(problem, RK222(); dt=0.1)
    step!(solver)
    @test Array(grid_data!(u)) ≈ fill(0.905, 8) atol=1e-13
end

@testset "SBDF startup" begin
    for ts in (SBDF3(), SBDF4())
        u = ScalarField(PeriodicDomain(8), "u")
        set!(u, (x,) -> 1.0)
        problem = InitialValueProblem([u])
        add_equation!(problem, "dt(u) = -u")
        solver = InitialValueSolver(problem, ts; dt=0.1)
        step!(solver)
        @test Array(grid_data!(u)) ≈ fill(0.9, 8) atol=1e-13
        step!(solver)
        @test Array(grid_data!(u)) ≈ fill(0.8133333333333334, 8) atol=1e-13
    end
end
