using Test
using Tarang

@testset "Quick domain constructors" begin
    coords1 = CartesianCoordinates("x")
    dist1 = Distributor(coords1; mesh=(1,), dtype=Float64)
    domain1 = Tarang.create_fourier_domain(dist1, 2π, 8)
    @test domain1.dist === dist1
    @test length(domain1.bases) == 1
    @test domain1.bases[1] isa RealFourier
    @test domain1.bases[1].meta.size == 8

    coords2 = CartesianCoordinates("x", "y")
    dist2 = Distributor(coords2; mesh=(1, 1), dtype=Float64)
    domain2 = Tarang.create_2d_periodic_domain(dist2, 2π, π, 4, 6)
    @test length(domain2.bases) == 2
    @test domain2.bases[1] isa RealFourier
    @test domain2.bases[2] isa RealFourier

    domain_channel = Tarang.create_channel_domain(dist2, 2π, 1.0, 4, 5; dealias=1.0)
    @test domain_channel.bases[1] isa RealFourier
    @test domain_channel.bases[2] isa ChebyshevT

    coords3 = CartesianCoordinates("x", "y", "z")
    dist3 = Distributor(coords3; mesh=(1, 1, 1), dtype=Float64)
    domain3 = Tarang.create_3d_periodic_domain(dist3, 2π, 2π, 2π, 4, 4, 4)
    @test length(domain3.bases) == 3
    @test all(basis -> basis isa RealFourier, domain3.bases)
end
