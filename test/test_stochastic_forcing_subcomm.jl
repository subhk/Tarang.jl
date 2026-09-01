using Test
using Random
using MPI

MPI.Initialized() || MPI.Init()

using Tarang

const FORCING_WORLD = MPI.COMM_WORLD
const FORCING_WORLD_RANK = MPI.Comm_rank(FORCING_WORLD)
const FORCING_WORLD_SIZE = MPI.Comm_size(FORCING_WORLD)

@testset "stochastic forcing RNG synchronization uses the Distributor communicator" begin
    if FORCING_WORLD_SIZE < 4 || isodd(FORCING_WORLD_SIZE)
        @test_skip "requires an even MPI world of at least four ranks"
    else
        subgroup_size = FORCING_WORLD_SIZE ÷ 2
        color = FORCING_WORLD_RANK < subgroup_size ? 0 : 1
        subcomm = MPI.Comm_split(FORCING_WORLD, color, FORCING_WORLD_RANK)

        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; comm=subcomm, mesh=(subgroup_size,),
                           dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (xbasis, ybasis)), "u")
        problem = IVP([u])
        forcing = StochasticForcing(
            field_size=(8, 8),
            k_forcing=1.0,
            dk_forcing=0.5,
            rng=MersenneTwister(10_000 + FORCING_WORLD_RANK),
        )

        add_stochastic_forcing!(problem, :u, forcing)
        token = rand(forcing.rng, UInt64)
        all_tokens = MPI.Allgather(token, FORCING_WORLD)

        first_group = all_tokens[1:subgroup_size]
        second_group = all_tokens[(subgroup_size + 1):end]
        @test all(==(first(first_group)), first_group)
        @test all(==(first(second_group)), second_group)
        @test first(first_group) != first(second_group)

        close(dist)
        MPI.free(subcomm)
    end
end

MPI.Barrier(FORCING_WORLD)
