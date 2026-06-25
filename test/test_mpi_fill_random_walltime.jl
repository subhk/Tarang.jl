# Guards for two broad-audit MPI fixes (np >= 2):
#
# F1 — fill_random!(reproducible=true) must give the SAME field regardless of MPI
#   decomposition (incl. np=1). Previously the np=1 path used rank-local seeding
#   while np>1 used the global-index algorithm, so serial != distributed.
#
# F10 — proceed()/run! with a finite stop_wall_time must agree on the stop iteration
#   across ranks (rank 0's clock, broadcast). A rank-local wall-clock check could let
#   ranks exit the collective step loop on different iterations and deadlock.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI fill_random/wall-time test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

# Decomposition-independent value for seed=42, 16² RealFourier², reproducible=true.
const FILLRAND_SUMSQ_REF = 218.0074922688368

@testset "fill_random!(reproducible) is decomposition-independent (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)
    Tarang.fill_random!(u, "g"; seed=42, reproducible=true)
    ensure_layout!(u, :g)
    lv = get_grid_data(u) isa PencilArrays.PencilArray ? parent(get_grid_data(u)) : get_grid_data(u)
    sumsq = MPI.Allreduce(sum(abs2, lv), MPI.SUM, comm)
    # Same total as the serial (np=1) reproducible result, to roundoff.
    @test isapprox(sumsq, FILLRAND_SUMSQ_REF; rtol=1e-10)
end

@testset "run! with finite stop_wall_time exits collectively (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    b = ScalarField(Domain(dist, (xb, yb)), "b")
    ensure_layout!(b, :g)
    gd = get_grid_data(b)
    (gd isa PencilArrays.PencilArray ? parent(gd) : gd) .= 0.1
    prob = IVP([b]); add_parameters!(prob; nu=0.05)
    add_equation!(prob, "∂t(b) - nu*lap(b) = 0")
    solver = InitialValueSolver(prob, RK222(); dt=1e-3)
    # Large wall-time so the run stops on iteration; this still exercises the collective
    # wall-time branch in proceed() every step. Must complete (no deadlock) on all ranks.
    Tarang.run!(solver; stop_iteration=5, stop_wall_time=1e6)
    @test solver.iteration == 5
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
