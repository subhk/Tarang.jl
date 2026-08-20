# Guard: reproducible fill_random! must be decomposition-independent for EVERY
# layout (2026-08-20 MPI review, finding TF1/V5).
#
# Bug: `_fill_random_reproducible!` ignored its `layout` argument — rank
# offsets always came from the unscaled GRID pencil. A coeff-layout fill lives
# on a different pencil (other decomposed axis, rfft-halved first Fourier axis,
# permuted storage), so "c" fills seeded from wrong/duplicated global indices
# and silently differed from the np=1 run. Offsets and global sizes now come
# from the filled array's own pencil.
#
# The reference is the SERIAL enumeration itself: filling a full-size plain
# array with the same seed via `_fill_random_global_indexed!` is exactly what
# np=1 produces, so each rank's logical slab must match that array's slab.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

function _check_layout(field, layout, seed)
    fill_random!(field, layout; seed=seed, distribution="normal")
    data = layout == "g" ? get_grid_data(field) : get_coeff_data(field)
    if data isa PencilArrays.PencilArray
        gsize = PencilArrays.size_global(data)
        ax = PencilArrays.range_local(data)
        expected = zeros(eltype(data), gsize)
        Tarang._fill_random_global_indexed!(CPU(), expected, seed,
                                            ntuple(_ -> 0, ndims(expected)),
                                            gsize, "normal", 1.0)
        # global_view: GLOBAL logical indices on the pencil — permutation-safe.
        gv = PencilArrays.global_view(data)
        maxdiff = maximum(abs(gv[I] - expected[I]) for I in CartesianIndices(ax))
    else
        expected = zeros(eltype(data), size(data))
        Tarang._fill_random_global_indexed!(CPU(), expected, seed,
                                            ntuple(_ -> 0, ndims(expected)),
                                            size(expected), "normal", 1.0)
        maxdiff = maximum(abs.(data .- expected))
    end
    return nprocs > 1 ? MPI.Allreduce(maxdiff, MPI.MAX, comm) : maxdiff
end

@testset "fill_random! reproducible = serial enumeration (rank=$rank)" begin
    N = 8
    @testset "pure Fourier 2D (rfft-halved, permuted coeff pencil)" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, architecture=CPU())
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (xb, yb)), "u")
        @test _check_layout(u, "g", 42) == 0.0
        @test _check_layout(u, "c", 42) == 0.0
    end

    @testset "mixed Cheb×Fourier 2D (coeff pencil decomposes the other axis)" begin
        coords = CartesianCoordinates("z", "x")
        dist = Distributor(coords; dtype=Float64, architecture=CPU())
        zb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, 1.0))
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (zb, xb)), "u")
        @test _check_layout(u, "g", 7) == 0.0
        @test _check_layout(u, "c", 7) == 0.0
    end
end
