# Multi-rank guard for the CFL diffusive limit.
#
# `add_diffusivity!` accepts a per-rank slab (e.g. an LES eddy-viscosity array,
# which is local storage with no communicator). The limiting timestep must come
# from the GLOBAL maximum diffusivity, not this rank's, and every rank must agree
# on the returned dt — otherwise ranks would march at different timesteps.
#
# The diffusive frequency is folded into the SAME batched Allreduce(MAX) as the
# advective term, so this also guards against someone adding a second collective
# or reducing after applying the geometric factor.
#
# Launch (np ∈ {1,2,4}) via run_mpi_ci.jl. Registered in MPI_TEST_FILES.
using Test
using MPI
MPI.Init()
using Tarang

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

@testset "MPI CFL diffusive limit" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 1.0), dtype=Float64)
    yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 1.0), dtype=Float64)

    u = VectorField(dist, coords, "u", (xb, yb), Float64)
    for c in u.components
        ensure_layout!(c, :g)
        fill!(get_grid_data(c), 1e-8)   # negligible advective limit
    end

    problem = InitialValueProblem([u]; namespace=Dict("u" => u))
    Tarang.add_equation!(problem, "∂t(u) = 0")
    solver = InitialValueSolver(problem, RK111(); device="cpu")

    safety = 0.4
    # Only the LAST rank holds the global maximum, so a rank-local reduction on
    # rank 0 would return a dt that is `nu_hi/nu_lo` times too large.
    nu_hi, nu_lo = 1.0, 0.1
    nu_local = rank == nprocs - 1 ? nu_hi : nu_lo
    slab = fill(nu_local, (4, 4))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    Tarang.add_velocity!(cfl, u)
    Tarang.add_diffusivity!(cfl, slab)

    dt = Tarang.compute_timestep(cfl)

    # Both global axes contain modes up to |k| = 16π, even when this rank's
    # coefficient slab does not contain the highest mode.
    spectral_radius = 2 * (16π)^2
    expected_global = 2 * safety / (nu_hi * spectral_radius)

    # dt comes from the GLOBAL max diffusivity.
    @test isapprox(dt, expected_global; rtol=1e-10)

    # Every rank agrees.
    all_dt = MPI.Allgather(dt, comm)
    @test all(isapprox.(all_dt, expected_global; rtol=1e-10))

    # And on a rank that does NOT hold the max, the local-only answer would have
    # been measurably different — proving the reduction actually did something.
    if nprocs > 1 && rank != nprocs - 1
        @test !isapprox(dt, 2 * safety / (nu_lo * spectral_radius); rtol=1e-6)
    end
end

MPI.Barrier(comm)
