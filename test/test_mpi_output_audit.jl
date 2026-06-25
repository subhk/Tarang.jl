# Guards for the round-4 MPI output audit (2026-06-23).
#
# #9 — NetCDFFileHandler.process! gated on a per-rank check_schedule whose wall_dt
#   decision diverges across ranks (per-rank wall clock). Divergent ranks would
#   enter/skip the collective writes (MPI.Barrier / MPI.Allreduce) inconsistently
#   → DEADLOCK. The decision is now MPI.Bcast! from rank 0 (mirrors VFH).
# #10 — complex-field checkpoints split data into a leading [real,imag] dim before
#   build_layout_metadata, whose ndims check then dropped all start/count attrs, so
#   the RECONSTRUCT merge mis-inferred a single-axis decomposition (scrambled the
#   restart on a >=2D mesh). The split dim is now accounted.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI output audit test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const pid0 = (r = Ref(Int(getpid())); MPI.Bcast!(r, 0, comm); r[])
const outdir = joinpath(tempdir(), "tarang_outaudit_np$(nprocs)_$(pid0)")
rank == 0 && mkpath(outdir)
MPI.Barrier(comm)

@testset "MPI output audit (np=$nprocs)" begin

    # --- #9: divergent wall_dt schedule must NOT deadlock (Bcast'd decision) ---
    @testset "wall_dt schedule synced across ranks (no deadlock)" begin
        N = 8
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        h = Tarang.NetCDFFileHandler(joinpath(outdir, "chk9"), dist, Dict{String,Any}();
                                     wall_dt=1.0, max_writes=1, parallel="gather")
        Tarang.add_task!(h, u; name="u")
        # Call 1: all ranks at wall=1.5 → div=1 → all schedule (synced).
        Tarang.process!(h; iteration=1, sim_time=0.0, wall_time=1.5, timestep=1)
        MPI.Barrier(comm)
        # Call 2: DIVERGENT — rank 0 crosses to div=2 (schedules), others stay div=1.
        wt2 = rank == 0 ? 2.5 : 1.6
        w2 = Tarang.process!(h; iteration=2, sim_time=1.0, wall_time=wt2, timestep=2)
        MPI.Barrier(comm)   # reached only if call 2 did not deadlock
        # Bcast forces every rank to follow rank 0's "write" decision.
        @test w2 == true
    end

    # --- #10: complex checkpoint metadata accounts for the [real,imag] split dim ---
    @testset "complex build_layout_metadata keeps start/count" begin
        N = 4
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords)
        bx = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        by = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        bz = ComplexFourier(coords["z"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (bx, by, bz), ComplexF64)
        ensure_layout!(u, :c)
        h = Tarang.NetCDFFileHandler(joinpath(outdir, "chk10"), dist, Dict{String,Any}();
                                     iter=1, parallel="gather")
        Tarang.add_task!(h, u; name="u", layout="c")
        Tarang.init_mpi!(h)
        task = h.tasks[1]
        _, lstart, lshape = Tarang.get_data_distribution(h, task)
        # Emulate the leading [real,imag] split (size 2) that write_task_data! prepends.
        split_data = zeros(Float64, 2, Tuple(lshape)...)
        meta = Tarang.build_layout_metadata(task, u, split_data)
        @test meta !== nothing                       # was `nothing` pre-fix
        @test meta.count[1] == 2                      # leading complex-split dim
        @test meta.start[1] == 0
        @test Tuple(meta.count[2:end]) == Tuple(lshape)   # spatial slab preserved
        @test Tuple(meta.start[2:end]) == Tuple(Int.(lstart))
    end
end
