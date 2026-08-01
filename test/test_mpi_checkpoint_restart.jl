# The assertion the whole slab design exists for: a checkpoint written on N ranks
# must load on M, and the result must equal what a serial run produces.
#
# Before this, `load_field!` had rank 0 read the entire global array and scatter
# it, so it could not read the per-rank files the output handler writes at all.

using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NP = MPI.Comm_size(COMM)

if NP < 2
    RANK == 0 && @warn "MPI checkpoint test needs at least two ranks"
    MPI.Finalize()
    exit(0)
end

const NX = 16
# NY=14 (not 12): PencilArrays splits an axis as `start = N*c ÷ P`, remainder on
# the last ranks. NY=12 at np=4 gives starts=[0,3,6,9] counts=[3,3,3,3] -- a
# perfectly even split that never exercises a remainder in the slab overlap
# arithmetic. NY=14 at np=4 gives starts=[0,3,7,10] counts=[3,4,3,4] -- genuinely
# uneven -- while still being even itself (see `_init_value`'s exactness note
# below) and still splitting evenly at np=2 (7/7), so the np=2 testsets are
# unaffected.
const NY = 14

# sin(x)*cos(y) is wavenumber 1 in both x and y; 0.25cos(2x) is wavenumber 2 in
# x only. NX=16 and NY=14 are both even and comfortably above 2x the highest
# wavenumber used (2), so every term is exactly representable in the Fourier
# basis (no aliasing) at either size -- verified empirically for NY=14 by round-
# tripping :g -> :c -> :g and diffing against the original (see task-5-report.md,
# "Uneven-split coverage"). `dt(u) = kappa*lap(u)` then makes each of these exact
# Fourier modes an exact eigenmode, so the decay it produces is exact too, not
# just spectrally accurate.
_init_value(x, y) = sin(x) * cos(y) + 0.25cos(2x)

function _build(stepper, dt; comm=COMM)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU(), comm=comm)
    xb = RealFourier(coords["x"]; size=NX, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=NY, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u")
    problem = IVP([u])
    add_parameters!(problem, kappa=0.02)
    add_equation!(problem, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(problem, stepper; dt)

    xs = [2π * (i - 1) / NX for i in 1:NX]
    ys = [2π * (j - 1) / NY for j in 1:NY]
    initial = [_init_value(x, y) for x in xs, y in ys]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        parent(gd) .= initial[PencilArrays.pencil(gd).axes_local...]
    else
        gd .= initial
    end
    ensure_layout!(u, :c)
    return solver, initial
end

# --- 3-D field on a 2-D process mesh: the production geometry ---
#
# Every other assertion in this file is 2-D on a 1-D process mesh.
# `get_local_start`/`get_local_shape` take a DIFFERENT branch for a 2-D process
# mesh (netcdf_output.jl), and this repo's 2026-06-23 audit records that branch
# having been wrong before (column-major vs MPI-Cart row-major coordinates). A
# 3-D field on a 2x2 mesh at np=4 is the case production actually runs.
const NX3, NY3, NZ3 = 8, 8, 6
_init3d(x, y, z) = sin(x) * cos(y) + 0.25cos(2z)

function _build3d(dt; comm=COMM, mesh=nothing)
    coords = CartesianCoordinates("x", "y", "z")
    dist = mesh === nothing ?
        Distributor(coords; dtype=Float64, architecture=CPU(), comm=comm) :
        Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU(), comm=comm)
    xb = RealFourier(coords["x"]; size=NX3, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=NY3, bounds=(0.0, 2π))
    zb = RealFourier(coords["z"]; size=NZ3, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb, zb))
    u = ScalarField(domain, "u")
    problem = IVP([u])
    add_parameters!(problem, kappa=0.02)
    add_equation!(problem, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(problem, RK222(); dt)

    xs = [2π * (i - 1) / NX3 for i in 1:NX3]
    ys = [2π * (j - 1) / NY3 for j in 1:NY3]
    zs = [2π * (k - 1) / NZ3 for k in 1:NZ3]
    initial = [_init3d(x, y, z) for x in xs, y in ys, z in zs]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        parent(gd) .= initial[PencilArrays.pencil(gd).axes_local...]
    else
        gd .= initial
    end
    ensure_layout!(u, :c)
    return solver
end

"""Advance a 3-D solver on COMM_SELF and return the full global grid."""
function _serial_reference3d(dt, nsteps)
    solver = _build3d(dt; comm=MPI.COMM_SELF)
    for _ in 1:nsteps
        step!(solver, dt)
    end
    u = solver.state[1]
    ensure_layout!(u, :g)
    return Array(get_grid_data(u))
end

"""Advance a serial solver on COMM_SELF and return the full global grid."""
function _serial_reference(stepper, dt, nsteps)
    solver, _ = _build(stepper, dt; comm=MPI.COMM_SELF)
    for _ in 1:nsteps
        step!(solver, dt)
    end
    u = solver.state[1]
    ensure_layout!(u, :g)
    return Array(get_grid_data(u))
end

"""Max |local slab - reference| over all ranks."""
function _diff_against(field, reference)
    ensure_layout!(field, :g)
    gd = get_grid_data(field)
    local_diff = if gd isa PencilArrays.PencilArray
        maximum(abs.(parent(gd) .- reference[PencilArrays.pencil(gd).axes_local...]))
    else
        maximum(abs.(Array(gd) .- reference))
    end
    return MPI.Allreduce(local_diff, MPI.MAX, COMM)
end

"""Reduce a per-rank boolean to a single value every rank agrees on.

Some assertions below are evaluated independently on every rank with no
collective of their own (a `Bool` comparison like `restored.iteration ==
solver.iteration`, or a `gd isa PencilArrays.PencilArray` type check). If a bug
made that condition true on some ranks and false on others, an un-reduced
`@test` would throw `Test.TestSetException` on only the failing ranks while
the rest moved on to the next testset's collectives -- the exact divergent-
failure hazard `_abort_if_any_rank_failed` exists to prevent in
`field_layout_arithmetic_io.jl`, just here in test code instead of production
code. `MPI.Allreduce` with `MPI.MIN` over a 0/1 flag makes `true` the reduced
result only when EVERY rank saw `true`; one rank seeing `false` drags every
rank's `@test` down together, so a real failure fails (or a real pass passes)
the same testset on every rank at once."""
_agree(ok::Bool) = MPI.Allreduce(ok ? 1 : 0, MPI.MIN, COMM) == 1

# Every rank derives the same path independently — no broadcast of a temp-dir name
# (this repo uses no `MPI.bcast` for objects anywhere). Rank 0 creates it, the rest
# wait on the barrier.
const CHK_ROOT = joinpath(tempdir(), "tarang_ckpt_test_np$(NP)")
if RANK == 0
    isdir(CHK_ROOT) && rm(CHK_ROOT; recursive=true, force=true)
    mkpath(CHK_ROOT)
end
MPI.Barrier(COMM)

# Every testset below can fail independently, and per-rank. If a `@test` fails
# on some ranks and not others, Julia throws `Test.TestSetException` on just
# the failing ranks at the end of that testset. Without a `try`/`finally`
# around all of them, those ranks would exit right there while the surviving
# ranks walked into a later testset's collectives (or the barrier + `rm`
# below) and hung forever on a shared box, AND the temp directory would be
# left behind. Wrapping every testset in one `try` with the cleanup in
# `finally` guarantees the barrier and `rm` always run on every rank that gets
# this far. That alone would still deadlock if a testset could fail on some
# ranks but not others -- which is exactly why every assertion below that has
# no collective of its own is routed through `_agree` first: with that in
# place, a real failure fails the SAME testset on every rank at once, so they
# either all reach this `finally` together or none of them do.
try
    @testset "Distributed checkpoint round-trips at the same rank count (rank=$RANK)" begin
        path = joinpath(CHK_ROOT, "same")
        solver, _ = _build(RK222(), 0.02)
        for _ in 1:10
            step!(solver, 0.02)
        end
        save_state(solver, path)
        MPI.Barrier(COMM)

        reference = _serial_reference(RK222(), 0.02, 10)

        restored, _ = _build(RK222(), 0.02)
        load_state!(restored, path)
        @test _diff_against(restored.state[1], reference) < 1e-13
        @test _agree(restored.iteration == solver.iteration)
        @test _agree(restored.sim_time ≈ solver.sim_time)
    end

    @testset "A checkpoint written on $NP ranks loads on one (rank=$RANK)" begin
        # Every rank independently reads the WHOLE global field into a COMM_SELF
        # solver. That is the "restart on fewer ranks" case, and each rank checking it
        # separately makes the assertion independent of which rank holds what.
        path = joinpath(CHK_ROOT, "same")
        reference = _serial_reference(RK222(), 0.02, 10)

        serial_solver, _ = _build(RK222(), 0.02; comm=MPI.COMM_SELF)
        load_state!(serial_solver, path)
        u = serial_solver.state[1]
        ensure_layout!(u, :g)
        @test maximum(abs.(Array(get_grid_data(u)) .- reference)) < 1e-13
        # The equation is time-autonomous, so a clock-restoration bug that only
        # shows up at a different rank count would not perturb these field
        # values at all -- only checking iteration/sim_time here would catch
        # it. The checkpoint at `path` is the one the testset above wrote
        # after 10 steps at dt=0.02 (each `@testset` body is its own scope, so
        # that testset's `solver` isn't reachable here; 10 and 0.02 are the
        # same two literals already reused for `reference` just above).
        @test _agree(serial_solver.iteration == 10)
        @test _agree(serial_solver.sim_time ≈ 10 * 0.02)
    end

    @testset "A serial checkpoint loads on $NP ranks (rank=$RANK)" begin
        # The reverse direction: one writer, many readers.
        path = joinpath(CHK_ROOT, "from_serial")
        if RANK == 0
            writer, _ = _build(RK222(), 0.02; comm=MPI.COMM_SELF)
            for _ in 1:10
                step!(writer, 0.02)
            end
            save_state(writer, path)
        end
        MPI.Barrier(COMM)

        reference = _serial_reference(RK222(), 0.02, 10)
        restored, _ = _build(RK222(), 0.02)
        load_state!(restored, path)
        @test _diff_against(restored.state[1], reference) < 1e-13
    end

    @testset "Restart continues the trajectory (rank=$RANK)" begin
        path = joinpath(CHK_ROOT, "resume")
        reference = _serial_reference(RK222(), 0.02, 20)

        first_half, _ = _build(RK222(), 0.02)
        for _ in 1:10
            step!(first_half, 0.02)
        end
        save_state(first_half, path)
        MPI.Barrier(COMM)

        second_half, _ = _build(RK222(), 0.02)
        load_state!(second_half, path)
        for _ in 1:10
            step!(second_half, 0.02)
        end
        @test _diff_against(second_half.state[1], reference) < 1e-12
    end

    @testset "A missing checkpoint fails on every rank, not just one (rank=$RANK)" begin
        # A one-rank throw with the others still in the collective is a deadlock.
        restored, _ = _build(RK222(), 0.02)
        @test_throws Exception load_state!(restored, joinpath(CHK_ROOT, "absent"))
    end

    @testset "np=4 decomposes the Y axis unevenly, not just evenly (rank=$RANK)" begin
        # This is the assertion that stops a future grid-size edit from silently
        # reverting the coverage NY=14 exists for. Trusting the `start = N*c ÷ P`
        # arithmetic on paper isn't enough -- this asks PencilArrays directly, at
        # runtime, what each rank's Y-axis slab actually looks like.
        #
        # At np=4, NX=16/NY=14 must give counts=[3,4,3,4] (not all equal): a genuine
        # remainder split. At np=2, NY=14 splits 7/7 -- evenly -- so the "not all
        # equal" check only applies at np=4; asserting it unconditionally would make
        # this testset fail at np=2 for a reason that has nothing to do with the
        # coverage gap it exists to guard.
        solver, _ = _build(RK222(), 0.02)
        u = solver.state[1]
        ensure_layout!(u, :g)
        gd = get_grid_data(u)
        @test _agree(gd isa PencilArrays.PencilArray)
        y_range = PencilArrays.pencil(gd).axes_local[2]
        local_y_count = length(y_range)
        @test _agree(size(parent(gd), 2) == local_y_count)
        println("rank=$RANK np=$NP local Y-axis range=$y_range count=$local_y_count")
        all_y_counts = MPI.Allgather(local_y_count, COMM)
        if NP == 4
            @test !all(==(all_y_counts[1]), all_y_counts)
        end
    end

    @testset "re-checkpointing at a LOWER rank count leaves no stale slabs (rank=$RANK)" begin
        # `_slab_output_path!` creates the directory but each rank only ever
        # removed its OWN file, so a np=4 checkpoint followed by a np=2
        # checkpoint to the same path rewrote p0/p1 and left p2/p3 behind. The
        # directory is then self-overlapping and the restart throws "has
        # overlapping slabs" -- loud, but hours later, having silently ruined
        # the checkpoint at write time. Restarting at a different rank count is
        # this feature's headline capability, so checkpoint-restart-checkpoint
        # at a smaller np is a first-week user action.
        if NP == 4
            path = joinpath(CHK_ROOT, "downshift")
            big, _ = _build(RK222(), 0.02)
            for _ in 1:10
                step!(big, 0.02)
            end
            save_state(big, path)
            MPI.Barrier(COMM)
            @test _agree(length(filter(f -> endswith(f, ".nc"), readdir(path))) == 4)

            # Second write from a 2-rank sub-communicator, to the SAME path,
            # holding DIFFERENT data (20 steps, not 10) so a stale p2/p3 that
            # somehow avoided the overlap error would still be caught by value.
            sub = MPI.Comm_split(COMM, RANK < 2 ? 0 : 1, RANK)
            if RANK < 2
                small, _ = _build(RK222(), 0.02; comm=sub)
                for _ in 1:20
                    step!(small, 0.02)
                end
                save_state(small, path)
            end
            MPI.Barrier(COMM)
            @test _agree(length(filter(f -> endswith(f, ".nc"), readdir(path))) == 2)

            reference = _serial_reference(RK222(), 0.02, 20)
            restored, _ = _build(RK222(), 0.02)
            load_state!(restored, path)
            @test _diff_against(restored.state[1], reference) < 1e-13
            @test _agree(restored.iteration == 20)
        else
            @test NP == 2   # placeholder so the testset is non-empty at np=2
        end
    end

    @testset "a 3-D checkpoint on a 2-D process mesh restores on ONE rank (rank=$RANK)" begin
        if NP == 4
            path = joinpath(CHK_ROOT, "mesh2d")
            solver = _build3d(0.02; mesh=(2, 2))
            u = solver.state[1]
            ensure_layout!(u, :g)
            gd = get_grid_data(u)
            @test _agree(gd isa PencilArrays.PencilArray)

            # The writer and the reader BOTH derive their hyperslab from these
            # two helpers, so a wrong offset would be applied consistently and
            # an N->N round trip could not see it. Check them against what
            # PencilArrays actually allocated, on the 2-D-mesh branch.
            axes = PencilArrays.pencil(gd).axes_local
            di = Tarang.get_operator_domain(u)
            helper_start = collect(Int, Tarang.get_local_start(:g, di, u.scales, u.dist.rank))
            helper_shape = collect(Int, Tarang.get_local_shape(:g, di, u.scales, u.dist.rank))
            println("rank=$RANK 3D mesh=(2,2) axes_local=$axes start=$helper_start shape=$helper_shape")
            @test _agree(helper_start == [first(r) - 1 for r in axes])
            @test _agree(helper_shape == [length(r) for r in axes])
            # Both mesh dimensions must genuinely be >1, or this is a 1-D mesh
            # in disguise and tests nothing the testsets above do not.
            @test _agree(length(axes[2]) < NY3 && length(axes[3]) < NZ3)

            for _ in 1:5
                step!(solver, 0.02)
            end
            save_state(solver, path)
            MPI.Barrier(COMM)

            # N -> 1 is the direction that can detect this class. An N -> N
            # round trip re-applies the same offset on read and on write, so a
            # consistently wrong offset cancels and the values match anyway;
            # only a comparison against a serial reference exposes it.
            reference = _serial_reference3d(0.02, 5)
            serial_solver = _build3d(0.02; comm=MPI.COMM_SELF)
            load_state!(serial_solver, path)
            su = serial_solver.state[1]
            ensure_layout!(su, :g)
            @test maximum(abs.(Array(get_grid_data(su)) .- reference)) < 1e-13
            @test _agree(serial_solver.iteration == 5)
            # Oracle sanity: the reference must carry real signal, so an
            # all-zero read cannot pass the comparison above.
            @test maximum(abs, reference) > 0.5
        else
            @test NP == 2
        end
    end

    @testset "a real NetCDFFileHandler output directory reads back under MPI (rank=$RANK)" begin
        # The slab layer's docstring claims the three-attribute rule is what
        # lets a handler directory be opened directly, and the design spec
        # lists this as a test row. Nothing tested it, and it did not work: the
        # handler writes a leading unlimited `sim_time` dimension that the
        # start/count attributes do not describe. This pins the attribute
        # contract against the LIVE writer, under decomposition.
        solver, initial = _build(RK222(), 0.02)
        u = solver.state[1]
        ensure_layout!(u, :g)

        handler = Tarang.NetCDFFileHandler(joinpath(CHK_ROOT, "handler"), u.dist,
                                           Dict{String,Any}(); iter=1, parallel="gather")
        Tarang.add_task!(handler, u; name="u", layout="g")
        Tarang.process!(handler; iteration=0, sim_time=0.0, wall_time=0.0, timestep=0.02)
        MPI.Barrier(COMM)

        setdir = joinpath(CHK_ROOT, "handler_s1")
        @test _agree(isdir(setdir))

        reader, _ = _build(RK222(), 0.02)
        load_field!(reader.state[1], setdir, "u")
        @test _diff_against(reader.state[1], initial) < 1e-13
    end
finally
    MPI.Barrier(COMM)
    RANK == 0 && rm(CHK_ROOT; recursive=true, force=true)
end
MPI.Finalized() || MPI.Finalize()
