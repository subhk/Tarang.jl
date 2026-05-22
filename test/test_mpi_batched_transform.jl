using Test
using MPI
using PencilArrays
using Random
using Tarang

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "batched transform test requires >= 2 processes"
    exit(0)
end

_loc(f) = parent(get_grid_data(f))

# Batched backward of k fields must equal per-field backward, bit-for-bit (fp tol).
function run_case(N, k)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    dom = Domain(dist, (bx, by))

    ref = [ScalarField(dom, "ref$i") for i in 1:k]   # per-field backward
    bat = [ScalarField(dom, "bat$i") for i in 1:k]   # batched backward
    for i in 1:k
        ensure_layout!(ref[i], :g); ensure_layout!(bat[i], :g)
        Random.seed!(1000 + i + 17 * rank)
        g = randn(size(_loc(ref[i]))...)
        _loc(ref[i]) .= g
        _loc(bat[i]) .= g                              # identical inputs
        ensure_layout!(ref[i], :c); ensure_layout!(bat[i], :c)
    end

    for f in ref
        backward_transform!(f)                          # reference: per-field
    end
    Tarang._pencil_batched_backward!(bat)               # one batched transpose

    err = 0.0
    for i in 1:k
        err = max(err, maximum(abs.(_loc(bat[i]) .- _loc(ref[i]))))
    end
    return MPI.Allreduce(err, MPI.MAX, comm)
end

@testset "batched backward == per-field (rank=$rank)" begin
    for N in (32, 64), k in (2, 3)
        e = run_case(N, k)
        rank == 0 && println("  N=$N k=$k  max|batched-perfield| = ", e)
        @test e < 1e-12
    end
end

rank == 0 && println("batched transform test completed")
