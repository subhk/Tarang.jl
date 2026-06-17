# Multi-rank guard for the GLOBAL (dims=nothing) output reduction tasks
# (add_mean_task! / add_variance_task! / add_rms_task! / add_extrema_task!).
#
# Their `postprocess` closures run on the field's LOCAL slab, so under MPI a
# whole-domain reduction must combine across ranks (Allreduce). Previously they
# reduced the local slab only — e.g. for a field whose value is the global x-index
# gi∈1..16 split over 2 ranks, rank 0 wrote mean 4.5 and rank 1 wrote 12.5 instead
# of the true global 8.5. This test pins the MPI-aware global helpers against the
# analytic whole-domain statistics; every rank must agree on the global value.
#
# Launch (np ∈ {1,2,4}) via run_mpi_ci.jl. Registered in MPI_TEST_FILES.
using Test
using MPI
MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# Fill a (possibly distributed) grid array from a function of GLOBAL indices.
function mpired_fill!(field, f, N)
    ensure_layout!(field, :g)
    data = get_grid_data(field)
    if isa(data, PencilArrays.PencilArray)
        gv = PencilArrays.global_view(data)
        for I in CartesianIndices(gv); gv[I] = f(I[1], I[2]); end
    else
        for gj in 1:N, gi in 1:N; data[gi, gj] = f(gi, gj); end
    end
    return field
end

@testset "MPI global output reductions (np=$nprocs)" begin
    N = 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)
    mpired_fill!(u, (gi, gj) -> Float64(gi), N)        # value = gi (1..N), same for all y

    # The handler passes the LOCAL CPU slab to a task's postprocess.
    local_slab = Array(get_grid_data(u))

    # Analytic GLOBAL statistics over the N×N grid of values gi∈1..N (each repeated N times).
    μ   = (N + 1) / 2
    rms = sqrt((N + 1) * (2N + 1) / 6)
    var = N^2 / 12                                     # sample variance SS/(N²-1) = N²/12

    @test isapprox(Tarang._global_mean_val(local_slab, u), μ;   rtol=1e-12)
    @test isapprox(Tarang._global_rms_val(local_slab, u),  rms; rtol=1e-12)
    @test isapprox(Tarang._global_var_val(local_slab, u),  var; rtol=1e-12)
    @test Tarang._global_extremum_val(local_slab, u, MPI.MIN) == 1.0
    @test Tarang._global_extremum_val(local_slab, u, MPI.MAX) == Float64(N)
end

MPI.Finalize()
