using Test
using Tarang
using Random

@testset "Registered filters consume the completed solution" begin
    for timestepper in (RK222(), CNAB2()), compiled in (true, false)
        domain = PeriodicDomain(8)
        u = ScalarField(domain, "u")
        set!(u, (x,) -> 0.0)
        problem = InitialValueProblem([u])
        add_equation!(problem, "∂t(u) = 1")
        filt = ExponentialMean((8,); α=0.5)
        add_temporal_filter!(problem, :u_mean, filt, :u)
        solver = InitialValueSolver(problem, timestepper; dt=0.1)
        @test solver.rhs_plan.is_compiled
        compiled || (solver.rhs_plan = nothing)

        expected = zeros(8)
        for iteration in 1:3
            step!(solver)
            values = Array(Tarang.grid_data!(u))
            @test values ≈ fill(0.1 * iteration, 8)
            expected .+= 0.05 .* (values .- expected)
            @test get_mean(filt) ≈ expected
        end
    end
end

@testset "Registered filter failures propagate from step!" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> 1.0)
    problem = InitialValueProblem([u])
    add_equation!(problem, "∂t(u) = 0")
    filt = ExponentialMean((8,); α=30.0)
    add_temporal_filter!(problem, :u_mean, filt, :u)
    solver = InitialValueSolver(problem, RK222(); dt=0.1)

    @test_throws ArgumentError step!(solver)
    # The PDE step has completed when its post-step filter rejects the update.
    @test solver.iteration == 1
    @test solver.sim_time == 0.1
    @test all(iszero, get_mean(filt))
end

# Per-step paths for temporal filters and stochastic forcings must not rebuild
# lookup structures or go through Dict{...,Any} values (type-unstable dispatch).

@testset "Filter/forcing per-step type stability" begin
    N = 8
    dt = 0.01

    @testset "temporal filter registration concrete, per-step allocation-free" begin
        domain = PeriodicDomain(N, N)
        u = ScalarField(domain, "u")
        problem = InitialValueProblem([u])
        add_equation!(problem, "∂t(u) = 0")

        filt = ExponentialMean((N, N); α=0.1)
        add_temporal_filter!(problem, :u_mean, filt, :u)

        # Source field resolved once at registration, stored in a concrete struct
        reg = problem.temporal_filters[:u_mean]
        @test reg isa Tarang.TemporalFilterRegistration
        @test reg.filter === filt
        @test reg.source === :u
        @test reg.source_field === u
        @test isconcretetype(typeof(reg))

        solver = InitialValueSolver(problem, RK222(); dt=dt)
        for _ in 1:5
            Tarang._update_temporal_filters!(solver, dt)
        end
        allocs = @allocated Tarang._update_temporal_filters!(solver, dt)
        @test allocs <= 256  # old path rebuilt a Dict{String,Any} + String per step

        # Filter still actually updates
        step!(solver)
        @test get_mean(filt) !== nothing
    end

    @testset "stochastic forcing dict concretely keyed and Forcing-typed" begin
        domain = PeriodicDomain(N, N)
        q = ScalarField(domain, "q")
        problem = InitialValueProblem([q])
        add_equation!(problem, "∂t(q) = 0")

        forcing = StochasticForcing(
            field_size=(N, N), forcing_rate=0.1, k_forcing=3.0,
            dk_forcing=1.0, dt=dt, rng=MersenneTwister(42))
        add_stochastic_forcing!(problem, :q, forcing)

        @test problem.stochastic_forcings isa Dict{Int, Tarang.Forcing}

        solver = InitialValueSolver(problem, RK222(); dt=dt)
        Tarang._update_registered_forcings!(solver, 0.0, dt)
        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        ensure_layout!(rhs[1], :c)
        @test maximum(abs.(get_coeff_data(rhs[1]))) > 0
    end

    @testset "PencilTransformConfig has no Any fields" begin
        @test Tarang.PencilTransformConfig isa UnionAll
        body = Base.unwrap_unionall(Tarang.PencilTransformConfig)
        @test all(t -> t !== Any, body.types)
    end
end
