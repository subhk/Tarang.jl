using Test
using Tarang

@testset "Pretty Printing" begin
    @testset "Domain pretty printing" begin
        domain = PeriodicDomain(64, 64)

        # Compact form
        s = sprint(show, domain)
        @test contains(s, "Domain")
        @test contains(s, "2D")
        @test contains(s, "64")

        # Rich form
        s = sprint(show, MIME("text/plain"), domain)
        @test contains(s, "Domain")
        @test contains(s, "RealFourier")
        @test contains(s, "64")
        @test contains(s, "Architecture")
        @test contains(s, "CPU")
    end

    @testset "Domain 1D" begin
        domain = PeriodicDomain(32)
        s = sprint(show, domain)
        @test contains(s, "1D")
        @test contains(s, "32")
    end

    @testset "ChannelDomain pretty printing" begin
        domain = ChannelDomain(64, 32; Lx=2π, Lz=1.0)
        s = sprint(show, MIME("text/plain"), domain)
        @test contains(s, "RealFourier")
        @test contains(s, "ChebyshevT")
        @test contains(s, "64")
        @test contains(s, "32")
    end

    @testset "TensorField pretty printing" begin
        domain = PeriodicDomain(16, 16)
        T = TensorField(domain, "stress")

        s = sprint(show, T)
        @test contains(s, "TensorField")
        @test contains(s, "stress")

        s = sprint(show, MIME("text/plain"), T)
        @test contains(s, "TensorField")
        @test contains(s, "Components")
    end

    @testset "CFL pretty printing" begin
        domain = PeriodicDomain(16)
        T = ScalarField(domain, "T")
        problem = IVP([T])
        add_equation!(problem, "dt(T) = 0")
        solver = InitialValueSolver(problem, RK111(); device="cpu")

        cfl_obj = CFL(solver; initial_dt=0.01, safety=0.5, max_dt=0.1)

        s = sprint(show, cfl_obj)
        @test contains(s, "CFL")

        s = sprint(show, MIME("text/plain"), cfl_obj)
        @test contains(s, "CFL")
        @test contains(s, "Safety")
    end
end
