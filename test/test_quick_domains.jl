using Test
using Tarang

@testset "Quick domain constructors" begin
    # 1D periodic domain
    domain1 = PeriodicDomain(8; L=(2π,))
    @test length(domain1.bases) == 1
    @test domain1.bases[1] isa RealFourier
    @test domain1.bases[1].meta.size == 8

    # 2D periodic domain
    domain2 = PeriodicDomain(4, 6; L=(2π, π))
    @test length(domain2.bases) == 2
    @test domain2.bases[1] isa RealFourier
    @test domain2.bases[2] isa RealFourier

    # 2D channel domain (Fourier-Chebyshev)
    domain_channel = ChannelDomain(4, 5; Lx=2π, Lz=1.0, dealias=1.0)
    @test domain_channel.bases[1] isa RealFourier
    @test domain_channel.bases[2] isa ChebyshevT

    # 3D periodic domain
    domain3 = PeriodicDomain(4, 4, 4)
    @test length(domain3.bases) == 3
    @test all(basis -> basis isa RealFourier, domain3.bases)
end
