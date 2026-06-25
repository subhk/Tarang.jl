# Guard: distributed 3D 3/2-rule padded dealiasing == serial (np>=2; np=4 = 2D mesh).
# Exercises the N-D transpose-pad path (evaluate_padded_multiply_distributed) with a
# 3D field — 1 local axis, D-1 decomposed, single-swap transpose cycling. Round-7.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI 3D padded-dealiasing test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const N = 8
const SUMSQ_REF = 139.73759999999993
const MAX_REF   = 2.2499999999999996

@testset "Distributed 3D 3/2 padded dealiasing == serial (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords)
    bs = (RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2),
          RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2),
          RealFourier(coords["z"]; size=N, bounds=(0.0, 2π), dealias=3/2))
    u = ScalarField(dist, "u", bs, Float64)
    ensure_layout!(u, :g)

    xg = [(i-1)*2π/N for i in 1:N]
    u0 = [sin(xg[i])*cos(xg[j]) + 0.5cos(2xg[i])*sin(xg[k]) + 0.3sin(xg[j])*cos(2xg[k])
          for i in 1:N, j in 1:N, k in 1:N]
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        gv = PencilArrays.global_view(gd); for I in CartesianIndices(gv); gv[I] = u0[I]; end
    else
        Tarang.get_cpu_data(gd) .= u0
    end

    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    p = Tarang.evaluate_transform_multiply(u, u, ev; result_layout=:g)
    ensure_layout!(p, :g)
    pg = get_grid_data(p)
    g = pg isa PencilArrays.PencilArray ? PencilArrays.gather(pg) : Array(Tarang.get_cpu_data(pg))

    if rank == 0
        @test isapprox(sum(abs2, g), SUMSQ_REF; rtol=1e-10)
        @test isapprox(maximum(g),  MAX_REF;   rtol=1e-10)
    end
end
