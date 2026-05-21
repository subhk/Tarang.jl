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

# Per-basis `dealias=` must actually control truncation strength: a field with a
# single mode at k=12 on N=48 survives the default cutoff (3/2 → 15) but is removed
# by a stronger setting (3 → 8). Drives the real distributed multiply path.
function run_factor_case(N, mesh, dealias)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=dealias)
    by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=dealias)
    domain = Domain(dist, (bx, by))

    fa = ScalarField(domain, "fa")
    fb = ScalarField(domain, "fb")
    ensure_layout!(fa, :g); ensure_layout!(fb, :g)

    x = [2π * (i - 1) / N for i in 1:N]
    ga = [cos(12 * x[i]) for i in 1:N, _ in 1:N]
    gb = ones(N, N)
    _assign_local!(fa, ga)
    _assign_local!(fb, gb)

    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    product = Tarang.evaluate_transform_multiply(fa, fb, ev)
    ensure_layout!(product, :g)
    return MPI.Allreduce(maximum(abs.(_local(product))), MPI.MAX, comm)
end

@testset "basis dealias controls truncation strength (rank=$rank)" begin
    amp_keep = run_factor_case(48, (nprocs,), 3/2)   # cutoff 15 ≥ 12 → mode survives
    amp_kill = run_factor_case(48, (nprocs,), 3.0)   # cutoff 8  < 12 → mode removed
    rank == 0 && println("  amp(dealias=3/2)=", amp_keep, "  amp(dealias=3)=", amp_kill)
    @test amp_keep > 0.5
    @test amp_kill < 1e-9
end

# dealias=1 disables dealiasing entirely: a mode beyond the 3/2 cutoff (k=20 > N/3=16)
# is truncated by the default but kept when dealias=1.
function run_mode20_case(N, mesh, dealias)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=dealias)
    by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=dealias)
    domain = Domain(dist, (bx, by))
    fa = ScalarField(domain, "fa"); fb = ScalarField(domain, "fb")
    ensure_layout!(fa, :g); ensure_layout!(fb, :g)
    x = [2π * (i - 1) / N for i in 1:N]
    _assign_local!(fa, [cos(20 * x[i]) for i in 1:N, _ in 1:N])
    _assign_local!(fb, ones(N, N))
    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    product = Tarang.evaluate_transform_multiply(fa, fb, ev)
    ensure_layout!(product, :g)
    return MPI.Allreduce(maximum(abs.(_local(product))), MPI.MAX, comm)
end

@testset "dealias=1 disables dealiasing (rank=$rank)" begin
    amp_default = run_mode20_case(48, (nprocs,), 3/2)   # k=20 > 15 → removed
    amp_off     = run_mode20_case(48, (nprocs,), 1.0)   # no truncation → k=20 kept
    rank == 0 && println("  amp(dealias=3/2)=", amp_default, "  amp(dealias=1)=", amp_off)
    @test amp_default < 1e-9
    @test amp_off > 0.5
end

rank == 0 && println("MPI dealiasing product tests completed")
