using Test
using MPI
using PencilArrays
using FFTW
using Random
using Tarang

if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "MPI dealiasing product test requires at least 2 processes"
    exit(0)
end

rank == 0 && println("=" ^ 60)
rank == 0 && println("MPI Dealiasing Product Tests")
rank == 0 && println("=" ^ 60)

# Cutoff exactly as in _apply_spectral_cutoff_distributed! (factor = 3/2).
safe_cutoff(N) = min(floor(Int, N / (2 * (3 / 2))), (N - 1) ÷ 3)

# Integer FFT mode numbers (even N): 0,1,…,N/2,-(N/2-1),…,-1
fmodes(N) = [ i <= (N ÷ 2) + 1 ? i - 1 : i - 1 - N for i in 1:N ]

function _combine(vecs)
    nd = length(vecs)
    nd == 1 && return vecs[1]
    nd == 2 && return vecs[1] .* vecs[2]'
    return reshape(vecs[1], :, 1, 1) .* reshape(vecs[2], 1, :, 1) .* reshape(vecs[3], 1, 1, :)
end

# Deterministic global band-limited real field (identical on every rank for a given seed).
function global_band_limited(dims, K, seed)
    rng = MersenneTwister(seed)
    f = randn(rng, dims...)
    mask = _combine([abs.(fmodes(N)) .<= K for N in dims])
    return real(ifft(fft(f) .* mask))
end

_local(field) = get_grid_data(field) isa PencilArrays.PencilArray ?
                parent(get_grid_data(field)) : get_grid_data(field)

function _assign_local!(field, gdata)
    data = get_grid_data(field)
    if data isa PencilArrays.PencilArray
        ax = PencilArrays.pencil(data).axes_local
        parent(data) .= gdata[ax...]
    else
        data .= gdata
    end
end

function run_case(dims, mesh)
    N = dims[1]
    K = max(safe_cutoff(N) ÷ 2, 1)   # product modes 2K ≤ cutoff, alias-free → equals pointwise product

    coords_names = length(dims) == 2 ? ("x", "y") : ("x", "y", "z")
    coords = CartesianCoordinates(coords_names...)
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    bases = ntuple(i -> RealFourier(coords[coords_names[i]]; size=dims[i], bounds=(0.0, 2π), dealias=3/2),
                   length(dims))
    domain = Domain(dist, bases)

    fa = ScalarField(domain, "fa")
    fb = ScalarField(domain, "fb")
    ensure_layout!(fa, :g); ensure_layout!(fb, :g)

    ga = global_band_limited(dims, K, 101)
    gb = global_band_limited(dims, K, 202)
    _assign_local!(fa, ga)
    _assign_local!(fb, gb)

    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    product = Tarang.evaluate_transform_multiply(fa, fb, ev)
    ensure_layout!(product, :g)

    # Expected: pointwise product of the (band-limited) inputs on this rank's slice.
    expected = _local(fa) .* _local(fb)
    local_err = maximum(abs.(_local(product) .- expected))
    return MPI.Allreduce(local_err, MPI.MAX, comm)
end

@testset "MPI nonlinear product is dealiased correctly (rank=$rank)" begin
    @testset "2D (mesh=($nprocs,))" begin
        err = run_case((32, 32), (nprocs,))
        rank == 0 && println("  2D max error: ", err)
        @test err < 1e-9
    end

    @testset "3D (mesh=($nprocs,))" begin
        err = run_case((16, 16, 16), (nprocs,))
        rank == 0 && println("  3D max error: ", err)
        @test err < 1e-9
    end
end

# `dealias=factor` is the standard Orszag PADDING resolution for alias-free
# quadratic products, NOT a low-pass cutoff. So multiplying any resolved-mode field
# by 1 is the identity, to roundoff, regardless of the dealias factor — MATCHING
# serial. The pre-2026-06-23 distributed path used truncation-after-multiply that
# wrongly low-passed modes |k| > N/(2·factor) (a feature serial never had → it made
# distributed ≠ serial); the round-7 transpose-pad fix
# (evaluate_padded_multiply_distributed) makes distributed == serial to roundoff.
# Aggressive low-pass filtering is a separate explicit operation
# (apply_spectral_cutoff! / low_pass_filter!), not the dealias factor.
function product_x1_err(N, mesh, k, dealias)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=dealias)
    by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=dealias)
    domain = Domain(dist, (bx, by))
    fa = ScalarField(domain, "fa"); fb = ScalarField(domain, "fb")
    ensure_layout!(fa, :g); ensure_layout!(fb, :g)
    x = [2π * (i - 1) / N for i in 1:N]
    _assign_local!(fa, [cos(k * x[i]) for i in 1:N, _ in 1:N])
    _assign_local!(fb, ones(N, N))
    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    product = Tarang.evaluate_transform_multiply(fa, fb, ev)
    ensure_layout!(product, :g)
    # fa × 1 must equal fa (the input) to roundoff for every resolved mode.
    MPI.Allreduce(maximum(abs.(_local(product) .- _local(fa))), MPI.MAX, comm)
end

@testset "dealias=factor is padding resolution, not a low-pass cutoff (rank=$rank)" begin
    for (k, fac) in ((12, 3/2), (12, 3.0), (20, 3/2), (20, 1.0))
        err = product_x1_err(48, (nprocs,), k, fac)
        rank == 0 && println("  k=$k dealias=$fac: |f×1 - f| = ", err)
        @test err < 1e-9   # padding preserves every resolved mode; old path removed them
    end
end

rank == 0 && println("MPI dealiasing product tests completed")
