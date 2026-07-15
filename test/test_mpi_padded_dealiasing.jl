# Guard: distributed 3/2-rule padded dealiasing == serial (np>=2).
#
# The distributed nonlinear product used truncation-after-multiply on the decomposed
# Fourier axis (only the non-decomposed axes were 3/2-padded), so distributed
# nonlinear results differed from serial by O(aliasing) on the decomposed axis. The
# round-7 fix (evaluate_padded_multiply_distributed, nonlinear_evaluation.jl)
# transpose-pads the decomposed axis too, matching the serial padded multiply
# (evaluate_padded_multiply) to roundoff. Reference computed serially (np=1).
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI padded-dealiasing test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const N = 12
const SUMSQ_REF = 34.559324999999966
const MAX_REF   = 1.6899999999999995

@testset "Distributed 3/2 padded dealiasing == serial (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    dom = Domain(dist, (xb, yb))
    u = ScalarField(dom, "u")
    ensure_layout!(u, :g)

    xg = [(i-1)*2π/N for i in 1:N]
    u0 = [sin(xg[i])*cos(xg[j]) + 0.5cos(2xg[i])*sin(xg[j]) + 0.3sin(3xg[i])*cos(2xg[j]) for i in 1:N, j in 1:N]
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        gv = PencilArrays.global_view(gd); for I in CartesianIndices(gv); gv[I] = u0[I]; end
    else
        Tarang.get_cpu_data(gd) .= u0
    end

    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    # Construction must be lazy even for the common slab mesh `(nprocs,)`.
    # Prebuilding the legacy catalogue retains >200 MiB of scratch arrays per
    # evaluator/rank before the actual distributed padded path is used.
    @test isempty(ev.pencil_transforms.shape_transforms)
    @test isempty(ev.pencil_transforms.tuple_transforms)
    @test isempty(ev.pencil_transforms.padded_dealiasing)
    @test isempty(ev.pencil_transforms.padded_pencil)
    p = Tarang.evaluate_transform_multiply(u, u, ev; result_layout=:g)
    ensure_layout!(p, :g)
    pg = get_grid_data(p)
    g = pg isa PencilArrays.PencilArray ? PencilArrays.gather(pg) : Array(Tarang.get_cpu_data(pg))

    if rank == 0
        @test isapprox(sum(abs2, g), SUMSQ_REF; rtol=1e-10)
        @test isapprox(maximum(g),  MAX_REF;   rtol=1e-10)
    end

    # Cache-hit / buffer-reuse path: the distributed padded multiply now reuses a
    # cached per-problem workspace (Pencils + scratch PencilArrays + in-place FFT
    # plans + pooled result). The FIRST call above allocates/plans; subsequent calls
    # hit the reused buffers. Re-run several times and assert each still matches the
    # serial reference — a stale-buffer or aliasing bug would surface here, not on
    # the first call.
    for rep in 1:3
        p2 = Tarang.evaluate_transform_multiply(u, u, ev; result_layout=:g)
        ensure_layout!(p2, :g)
        pg2 = get_grid_data(p2)
        g2 = pg2 isa PencilArrays.PencilArray ? PencilArrays.gather(pg2) :
             Array(Tarang.get_cpu_data(pg2))
        if rank == 0
            @test isapprox(sum(abs2, g2), SUMSQ_REF; rtol=1e-10)
            @test isapprox(maximum(g2),  MAX_REF;   rtol=1e-10)
        end
    end
end
