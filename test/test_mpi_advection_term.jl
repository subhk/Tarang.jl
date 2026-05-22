using Test
using MPI
using PencilArrays
using Tarang

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "advection term test requires >= 2 processes"
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

# Analytic gate: u = (1, 0) constant, φ = cos(2x). Single low mode ⇒ dealiasing is
# a no-op, so u·∇φ = ∂φ/∂x = -2 sin(2x) exactly. Drives the real distributed
# advection sum (one product per direction).
function run_advection(N)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    dom = Domain(dist, (xb, yb))

    φ = ScalarField(dist, "phi", (xb, yb), Float64)
    vel = VectorField(dist, coords, "vel", (xb, yb), Float64)
    ensure_layout!(φ, :g)
    ensure_layout!(vel.components[1], :g)
    ensure_layout!(vel.components[2], :g)

    x = [2π * (i - 1) / N for i in 1:N]
    φg  = [cos(2 * x[i]) for i in 1:N, _ in 1:N]
    uxg = ones(N, N)
    uyg = zeros(N, N)
    _assign_local!(φ, φg)
    _assign_local!(vel.components[1], uxg)
    _assign_local!(vel.components[2], uyg)

    op = AdvectionOperator(vel, φ)
    res = evaluate_operator(op)
    ensure_layout!(res, :g)

    expg = [-2 * sin(2 * x[i]) for i in 1:N, _ in 1:N]
    expf = ScalarField(dist, "exp", (xb, yb), Float64)
    ensure_layout!(expf, :g)
    _assign_local!(expf, expg)

    err = maximum(abs.(_loc(res) .- _loc(expf)))
    return MPI.Allreduce(err, MPI.MAX, comm)
end

@testset "MPI advection u·∇φ matches analytic (rank=$rank)" begin
    for N in (32, 64)
        e = run_advection(N)
        rank == 0 && println("  N=$N  max|u·∇φ - analytic| = ", e)
        @test e < 1e-9
    end
end

rank == 0 && println("MPI advection term test completed")
