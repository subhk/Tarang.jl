using Test
using MPI
using PencilArrays
using Tarang

if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "MPI field initialization test requires at least 2 processes"
    exit(0)
end

function local_storage(data)
    return data isa PencilArrays.PencilArray ? parent(data) : data
end

function global_nonzero_count(data)
    local_count = count(!iszero, local_storage(data))
    return MPI.Allreduce(local_count, MPI.SUM, comm)
end

rank == 0 && println("=" ^ 60)
rank == 0 && println("MPI Field Initialization Tests")
rank == 0 && println("=" ^ 60)

@testset "MPI fields start with zeroed buffers (rank=$rank)" begin
    N = 256
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    domain = Domain(dist, (xbasis, ybasis))

    fields = ScalarField[
        ScalarField(domain, "q"),
        ScalarField(domain, "psi"),
        VectorField(domain, "u").components...
    ]

    for field in fields
        @test global_nonzero_count(get_grid_data(field)) == 0
        @test global_nonzero_count(get_coeff_data(field)) == 0
    end
end

rank == 0 && println("MPI field initialization tests completed")
