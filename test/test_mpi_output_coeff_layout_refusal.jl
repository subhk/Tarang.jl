# Guard: coefficient-layout output refuses loudly on the PencilArrays path
# (2026-08-20 MPI review, finding O1/V6).
#
# Bug: a layout="c" task at np>1 staged `parent()` of the PERMUTED PencilFFTs
# output pencil (storage order, different decomposed axis) under
# grid-convention start/count metadata — per-rank slabs written transposed
# under wrong-axis offsets, silently scrambled on merge whenever the local
# extents coincide. Until coeff output gets permutation-aware metadata, the
# STAGING refuses whenever the coeff array is a PencilArray. (Task
# registration still succeeds — non-pencil coeff storage, e.g. the metadata
# math pinned by test_mpi_output_audit.jl, is unaffected.)
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "coeff-layout output refusal test requires >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

# Shared output dir: every rank must use the SAME path (the handler's set
# directory is created rank-0-only), so build it from a Bcast pid.
const pid0 = (r = Ref(Int(getpid())); MPI.Bcast!(r, 0, comm); r[])
const outdir = joinpath(tempdir(), "tarang_coeffrefusal_np$(nprocs)_$(pid0)")
rank == 0 && mkpath(outdir)
MPI.Barrier(comm)

@testset "add_task! layout=\"c\" refuses under MPI (rank=$rank)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    u = ScalarField(Domain(dist, (xb, yb)), "u")
    fill_random!(u, "g"; seed=1)

    # Registration succeeds; the refusal fires at STAGING (write) time, where
    # the actual array type is known.
    h = Tarang.NetCDFFileHandler(joinpath(outdir, "snap"), dist,
                                 Dict{String, Any}("u" => u); iter=1)
    add_task!(h, u; layout="c", name="u_c")
    @test get_coeff_data(u) isa PencilArrays.PencilArray  # premise
    err = try
        Tarang.process!(h; iteration=1, sim_time=0.0, wall_time=0.0, timestep=1)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("layout=\"c\"", sprint(showerror, err))

    # Grid-layout output on the same handler class still works.
    h2 = Tarang.NetCDFFileHandler(joinpath(outdir, "snap_g"), dist,
                                  Dict{String, Any}("u" => u); iter=1)
    add_task!(h2, u; layout="g", name="u_g")
    @test Tarang.process!(h2; iteration=1, sim_time=0.0, wall_time=0.0, timestep=1) !== nothing
end
