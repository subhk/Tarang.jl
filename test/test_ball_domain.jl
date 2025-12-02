using Test
using Tarang

const Lib = Tarang.Libraries

@testset "BallDomain boundary conditions" begin
    domain = Lib.BallDomain(1.0, 4, 4, 4; boundary_conditions=Dict(:surface_bc => :dirichlet))
    @test domain.boundary_conditions[:center_regularity] == true
    @test domain.boundary_conditions[:surface_bc] == :dirichlet
    @test domain.coords.radius == 1.0
end
