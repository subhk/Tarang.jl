using Test
using Random
using Tarang

@testset "Separable stochastic forcing" begin
    coords = CartesianCoordinates("z")
    zbasis = ChebyshevT(coords["z"]; size=10, bounds=(0.0, 1.0))

    constructor(; profile=(z -> z * (1 - z)), profile_basis=zbasis,
                injection_metric=:direct) = SeparableStochasticForcing(
        fourier_size=(8,),
        chebyshev_basis=profile_basis,
        chebyshev_profile=profile,
        domain_size=(2pi,),
        energy_injection_rate=0.2,
        injection_metric=injection_metric,
        k_forcing=2.0,
        dk_forcing=0.5,
        dt=1e-2,
        architecture=CPU(),
        rng=MersenneTwister(42),
    )

    @testset "construction and profile validation" begin
        forcing = constructor()
        @test forcing isa StochasticForcingType
        @test size(forcing.cached_forcing) == (8, 10)
        @test size(forcing.fourier_realization) == (8,)
        @test length(forcing.chebyshev_profile) == 10
        @test forcing.injection_metric === :direct
        @test forcing.field_size == (8, 10)
        @test forcing.fourier_size == (8,)
        @test forcing.domain_size == (2pi,)
        @test forcing.is_stochastic
        @test !forcing.is_gpu

        @test_throws ArgumentError constructor(injection_metric=:vorticity_kinetic)
        @test_throws ArgumentError constructor(profile=zeros(10))
        @test_throws ArgumentError constructor(profile=fill(Inf, 10))
        @test_throws DimensionMismatch constructor(profile=ones(9))

        other_coords = CartesianCoordinates("z")
        wrong_basis = ChebyshevU(other_coords["z"]; size=10, bounds=(0.0, 1.0))
        @test_throws ArgumentError constructor(profile_basis=wrong_basis)
    end
end
