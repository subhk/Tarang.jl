# Guard: 3-D padded dealiasing must also run on a 1-D (slab) process mesh.
#
# `evaluate_padded_multiply_distributed` gated on `length(decomp) == D - 1`. A 3-D
# field on `mesh=(nprocs,)` has ONE decomposed axis and TWO local ones, so it
# failed that test and fell through to the 2/3-rule truncation fallback — while
# serial and a 2-D process mesh kept using 3/2 padding. Every mode between the
# 2/3 cutoff and N/2 was silently deleted, so the same run gave different answers
# depending only on the shape of the process mesh.
#
# u = sin(3x) on N = 16 gives u^2 = (1 - cos(6x))/2 exactly. 3/2 padding resolves
# the 6th harmonic; 2/3-rule truncation (cutoff floor(16/3) = 5) deletes it. The
# amplitude of cos(6x) therefore separates the two paths with no serial reference
# run needed: -1/2 if padded, 0 if truncated.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "3D slab padded-dealiasing test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const N = 16

function _slab_square(mesh)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    bs = ntuple(i -> RealFourier(coords[("x", "y", "z")[i]]; size=N, bounds=(0.0, 2π),
                                 dealias=3/2), 3)
    u = ScalarField(dist, "u", bs, Float64)
    ensure_layout!(u, :g)
    xg = [(i - 1) * 2π / N for i in 1:N]
    u0 = [sin(3 * xg[i]) for i in 1:N, _ in 1:N, _ in 1:N]
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        gv = PencilArrays.global_view(gd)
        for I in CartesianIndices(gv); gv[I] = u0[I]; end
    else
        Tarang.get_cpu_data(gd) .= u0
    end
    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    p = Tarang.evaluate_transform_multiply(u, u, ev; result_layout=:g)
    ensure_layout!(p, :g)
    pg = get_grid_data(p)
    return (pg isa PencilArrays.PencilArray ? PencilArrays.gather(pg) :
            Array(Tarang.get_cpu_data(pg))), xg
end

# amplitude of cos(m*x) in a field that depends on x only
function _cos_amp(g, xg, m)
    N1 = size(g, 1)
    s = 0.0
    for i in 1:N1; s += mean_over_yz(g, i) * cos(m * xg[i]); end
    return 2 * s / N1
end
mean_over_yz(g, i) = sum(@view g[i, :, :]) / (size(g, 2) * size(g, 3))

@testset "3D padded dealiasing on a 1-D slab mesh (np=$nprocs)" begin
    g, xg = _slab_square((nprocs,))
    if rank == 0 && g !== nothing
        # u^2 = 1/2 - 1/2 cos(6x): the 6th harmonic must SURVIVE the dealiasing.
        @test isapprox(_cos_amp(g, xg, 6), -0.5; atol = 1e-10)
        @test isapprox(sum(g) / length(g), 0.5; atol = 1e-10)
        expect = [0.5 - 0.5 * cos(6 * xg[i]) for i in 1:N, _ in 1:N, _ in 1:N]
        @test isapprox(g, expect; atol = 1e-10)
    end
end

# A 2-D process mesh already took the padded path; pin that the two meshes agree.
if nprocs == 4
    @testset "1-D slab mesh == 2-D process mesh (np=4)" begin
        gs, _ = _slab_square((4,))
        gp, _ = _slab_square((2, 2))
        if rank == 0 && gs !== nothing && gp !== nothing
            @test isapprox(gs, gp; atol = 1e-12)
        end
    end
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
