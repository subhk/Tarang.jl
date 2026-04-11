using Test
using Tarang

@testset "Convenience API" begin
    @testset "grid_data auto-transform" begin
        domain = PeriodicDomain(16)
        T = ScalarField(domain, "T")

        # Set data in grid space
        ensure_layout!(T, :g)
        get_grid_data(T) .= 1.0

        # Transform to coefficient space
        ensure_layout!(T, :c)
        @test T.current_layout == :c

        # grid_data should auto-transform back
        data = grid_data(T)
        @test T.current_layout == :g
        @test data === get_grid_data(T)
    end

    @testset "coeff_data auto-transform" begin
        domain = PeriodicDomain(16)
        T = ScalarField(domain, "T")

        ensure_layout!(T, :g)
        get_grid_data(T) .= 1.0

        # coeff_data should auto-transform to :c
        data = coeff_data(T)
        @test T.current_layout == :c
        @test data === get_coeff_data(T)
    end

    @testset "set! with function 1D" begin
        domain = PeriodicDomain(32)
        T = ScalarField(domain, "T")
        set!(T, (x,) -> sin(x))
        @test T.current_layout == :g
        @test maximum(abs.(get_grid_data(T))) > 0.5
    end

    @testset "set! with function 2D" begin
        domain = PeriodicDomain(16, 16)
        T2 = ScalarField(domain, "T2")
        set!(T2, (x, y) -> sin(x) * cos(y))
        @test T2.current_layout == :g
        @test maximum(abs.(get_grid_data(T2))) > 0.5
    end

    @testset "set! with constant" begin
        domain = PeriodicDomain(16)
        T = ScalarField(domain, "T")
        set!(T, 42.0)
        @test all(get_grid_data(T) .== 42.0)
    end

    @testset "Callback helpers" begin
        cb = on_interval(10) do solver
            nothing
        end
        @test cb isa Pair
        @test cb.first == 10
        @test cb.second isa Function

        cb2 = on_sim_time(0.5) do solver
            nothing
        end
        @test cb2 isa Pair
        @test cb2.first isa AbstractFloat
        @test cb2.first == 0.5
    end

    @testset "@root_only macro" begin
        executed = Ref(false)
        @root_only executed[] = true
        @test executed[]

        result = Ref(0)
        @root_only begin
            result[] = 42
        end
        @test result[] == 42
    end

    @testset "add_parameters!" begin
        domain = PeriodicDomain(16)
        T = ScalarField(domain, "T")
        problem = IVP([T])

        add_parameters!(problem, nu=1e-3, kappa=1e-4, Ra=1e6)

        @test problem.namespace["nu"] == 1e-3
        @test problem.namespace["kappa"] == 1e-4
        @test problem.namespace["Ra"] == 1e6
    end

    @testset "Structured BC helpers" begin
        domain = ChannelDomain(16, 8; Lx=2π, Lz=1.0)
        u = VectorField(domain, "u")
        T = ScalarField(domain, "T")

        problem = IVP([u, T])

        # These should not error and should add BCs
        no_slip!(problem, "u", "z", 0.0)
        no_slip!(problem, "u", "z", 1.0)
        fixed_value!(problem, "T", "z", 0.0, 1.0)
        fixed_value!(problem, "T", "z", 1.0, 0.0)
        free_slip!(problem, "u", "z", 0.5)
        insulating!(problem, "T", "z", 0.5)

        @test length(problem.boundary_conditions) >= 6
    end
end
