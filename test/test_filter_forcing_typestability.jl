using Test
using Tarang
using Random

# Per-step paths for temporal filters and stochastic forcings must not rebuild
# lookup structures or go through Dict{...,Any} values (type-unstable dispatch).

@testset "Filter/forcing per-step type stability" begin
    N = 8
    dt = 0.01

    @testset "temporal filter registration concrete, per-step allocation-free" begin
        domain = PeriodicDomain(N, N)
        u = ScalarField(domain, "u")
        problem = IVP([u])
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
        problem = IVP([q])
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
