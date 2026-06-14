using Test
using MPI
MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)

# Regression guard for the spectral-filter MPI fix. `low_pass_filter!` /
# `apply_spectral_cutoff!` used to filter via a set_scales! resample round-trip,
# which under MPI runs a per-rank LOCAL FFT (not a global spectral operation) and
# produces wrong results. The coefficient-space rewrite uses each rank's GLOBAL
# wavenumbers, so the global high mode is correctly removed regardless of the
# pencil decomposition.
@testset "MPI spectral low-pass filter (np=$nprocs)" begin
    N = 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)

    # global content cos(x) (k=1) + cos(6x) (k=6); keep |kx| ≤ 0.5·N/2 = 4 → drop k=6.
    ensure_layout!(u, :g)
    fillf(gi) = cos(2π * (gi - 1) / N) + cos(6 * 2π * (gi - 1) / N)
    data = get_grid_data(u)
    if isa(data, PencilArrays.PencilArray)
        gv = PencilArrays.global_view(data)
        for I in CartesianIndices(gv); gv[I] = fillf(I[1]); end
    else
        for gi in 1:N, gj in 1:N; data[gi, gj] = fillf(gi); end
    end

    Tarang.low_pass_filter!(u; scales=(0.5, 1.0))
    ensure_layout!(u, :g)

    expf(gi) = cos(2π * (gi - 1) / N)   # only the k=1 component survives
    d2 = get_grid_data(u)
    local_err = if isa(d2, PencilArrays.PencilArray)
        let gv = PencilArrays.global_view(d2)
            maximum(abs(gv[I] - expf(I[1])) for I in CartesianIndices(gv))
        end
    else
        maximum(abs(d2[gi, gj] - expf(gi)) for gi in 1:N, gj in 1:N)
    end
    @test MPI.Allreduce(local_err, MPI.MAX, comm) < 1e-10
end
