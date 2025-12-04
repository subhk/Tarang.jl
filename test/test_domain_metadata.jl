using Test
using Tarang

@testset "Domain metadata basics" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1, 1), dtype=Float64)

    x = coords["x"]
    y = coords["y"]

    fourier = RealFourier(x; size=8, dealias=1.5)
    cheb = ChebyshevT(y; size=6, dealias=1.0)

    domain = Domain(dist, (fourier, cheb))

    @test Tarang.dealias(domain) == (1.5, 1.0)
    @test Tarang.constant(domain) == (false, false)
    @test Tarang.nonconstant(domain) == (true, true)
    @test Tarang.mode_dependence(domain) == (true, true)
    full = Tarang.full_bases(domain)
    @test length(full) == dist.dim
    @test full[1] === fourier
    @test full[2] === cheb
    axes_map = Tarang.bases_by_axis(domain)
    @test axes_map[0] === fourier
    @test axes_map[1] === cheb
    @test Tarang.get_basis(domain, x) === fourier
    @test Tarang.get_basis(domain, 1) === cheb
    @test Tarang.get_basis_subaxis(domain, y) == 0
    @test Tarang.dim(domain) == 2
end
