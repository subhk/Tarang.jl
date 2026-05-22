using Test
using MPI
using PencilArrays
using Tarang

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "dot-product term test requires >= 2 processes"
    exit(0)
end

_loc(f) = get_grid_data(f) isa PencilArrays.PencilArray ? parent(get_grid_data(f)) : get_grid_data(f)

function _assign_local!(field, gdata)
    data = get_grid_data(field)
    if data isa PencilArrays.PencilArray
        ax = PencilArrays.pencil(data).axes_local
        parent(data) .= gdata[ax...]
    else
        data .= gdata
    end
end

# Analytic gate for the operator-tree hot path `evaluate_vector_dot_product`
# (what a parsed `u⋅∇φ` uses): u=(1,0), w=(cos2x, sin3y).
# u·w = 1·cos2x + 0·sin3y = cos2x. Single low modes ⇒ dealiasing no-op ⇒ exact.
function run_dot(N)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    dom = Domain(dist, (xb, yb))

    u = VectorField(dist, coords, "u", (xb, yb), Float64)
    w = VectorField(dist, coords, "w", (xb, yb), Float64)
    for c in (u.components..., w.components...)
        ensure_layout!(c, :g)
    end

    x = [2π * (i - 1) / N for i in 1:N]
    y = [2π * (j - 1) / N for j in 1:N]
    _assign_local!(u.components[1], ones(N, N))
    _assign_local!(u.components[2], zeros(N, N))
    _assign_local!(w.components[1], [cos(2 * x[i]) for i in 1:N, _ in 1:N])
    _assign_local!(w.components[2], [sin(3 * y[j]) for _ in 1:N, j in 1:N])

    res = Tarang.evaluate_vector_dot_product(u, w)
    ensure_layout!(res, :g)

    expg = [cos(2 * x[i]) for i in 1:N, _ in 1:N]
    expf = ScalarField(dist, "exp", (xb, yb), Float64)
    ensure_layout!(expf, :g)
    _assign_local!(expf, expg)

    err = maximum(abs.(_loc(res) .- _loc(expf)))
    return MPI.Allreduce(err, MPI.MAX, comm)
end

@testset "MPI dot-product u·w matches analytic (rank=$rank)" begin
    for N in (32, 64)
        e = run_dot(N)
        rank == 0 && println("  N=$N  max|u·w - analytic| = ", e)
        @test e < 1e-9
    end
end

rank == 0 && println("MPI dot-product term test completed")
