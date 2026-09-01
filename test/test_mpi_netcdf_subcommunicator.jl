using Test
using MPI

MPI.Initialized() || MPI.Init()

using Tarang

const WORLD = MPI.COMM_WORLD
const WORLD_RANK = MPI.Comm_rank(WORLD)
const WORLD_SIZE = MPI.Comm_size(WORLD)

if WORLD_SIZE < 2
    WORLD_RANK == 0 && @warn "subcommunicator output test requires at least two ranks"
else
    root_pid = Ref(WORLD_RANK == 0 ? Int(getpid()) : 0)
    MPI.Bcast!(root_pid, WORLD; root=0)
    outdir = joinpath(tempdir(), "tarang_netcdf_subcomm_$(root_pid[])")
    WORLD_RANK == 0 && mkpath(outdir)
    MPI.Barrier(WORLD)

    # Only world rank 0 belongs to the communicator that constructs/processes the
    # handler. Every nonmember blocks in a point-to-point receive, so an accidental
    # COMM_WORLD collective in the handler cannot be matched by test scaffolding.
    subcomm = MPI.Comm_split(WORLD, WORLD_RANK == 0 ? 0 : nothing, WORLD_RANK)
    completion_tag = 731
    root_error = Ref{Any}(nothing)
    subgroup_handler = Ref{Any}(nothing)

    if WORLD_RANK == 0
        try
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; comm=subcomm, mesh=(1,), dtype=Float64)
            basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 2pi))
            u = ScalarField(dist, "u", (basis,), Float64)
            ensure_layout!(u, :g)
            fill!(get_grid_data(u), 4.0)

            subgroup_handler[] = NetCDFFileHandler(
                joinpath(outdir, "subgroup"), dist, Dict("u" => u);
                mode="overwrite")
            add_task!(subgroup_handler[], u; name="u")
            @test subgroup_handler[].comm === subcomm
            @test subgroup_handler[].size == 1
            @test process!(subgroup_handler[]; iteration=0, sim_time=0.0,
                           wall_time=0.0, timestep=0.1)
            @test group_ncread(current_file(subgroup_handler[]), "vars", "u") ==
                  reshape(fill(4.0, 4), 1, 4)
        catch err
            root_error[] = (err, catch_backtrace())
        finally
            try
                if subgroup_handler[] !== nothing
                    close!(subgroup_handler[])
                    # Run and unregister the handler's GC finalizer while its
                    # borrowed communicator is still valid.
                    finalize(subgroup_handler[])
                    subgroup_handler[] = nothing
                end
                subcomm == MPI.COMM_NULL || MPI.free(subcomm)
            catch err
                root_error[] === nothing &&
                    (root_error[] = (err, catch_backtrace()))
            finally
                for destination in 1:(WORLD_SIZE - 1)
                    MPI.Send(Int32(1), WORLD; dest=destination, tag=completion_tag)
                end
            end
        end
    else
        signal = Ref{Int32}(0)
        MPI.Recv!(signal, WORLD; source=0, tag=completion_tag)
        @test signal[] == 1
    end

    MPI.Barrier(WORLD)
    if WORLD_RANK == 0
        @test root_error[] === nothing
        root_error[] === nothing || showerror(stderr, root_error[][1], root_error[][2])
        rm(outdir; recursive=true, force=true)
    end
    MPI.Barrier(WORLD)

    # A failure local to one member must be settled before any member advances to
    # another collective. Rank 0 alone throws in postprocess; every rank must
    # leave process! by exception and still reach the following world barrier.
    failure_dir = joinpath(tempdir(), "tarang_netcdf_collective_failure_$(root_pid[])")
    WORLD_RANK == 0 && mkpath(failure_dir)
    MPI.Barrier(WORLD)

    failure_coords = CartesianCoordinates("x", "y")
    failure_dist = Distributor(failure_coords; comm=WORLD,
                               mesh=(WORLD_SIZE,), dtype=Float64)
    failure_bases = (
        RealFourier(failure_coords["x"]; size=4, bounds=(0.0, 2pi)),
        RealFourier(failure_coords["y"]; size=4, bounds=(0.0, 2pi)),
    )
    failure_field = ScalarField(failure_dist, "u", failure_bases, Float64)
    ensure_layout!(failure_field, :g)
    fill!(get_grid_data(failure_field), Float64(WORLD_RANK + 1))

    failure_handler = NetCDFFileHandler(
        joinpath(failure_dir, "rank_local"), failure_dist,
        Dict("u" => failure_field); mode="overwrite")
    add_task!(failure_handler, failure_field; name="u", postprocess=data -> begin
        WORLD_RANK == 0 && error("injected rank-local postprocess failure")
        data
    end)

    threw = Ref(false)
    try
        process!(failure_handler; iteration=0, sim_time=0.0,
                 wall_time=0.0, timestep=0.1)
    catch
        threw[] = true
    end
    @test threw[]
    @test failure_handler.total_write_num == 0
    @test failure_handler.file_write_num == 0
    @test MPI.Allreduce(threw[] ? 1 : 0, MPI.SUM, WORLD) == WORLD_SIZE
    MPI.Barrier(WORLD)
    close!(failure_handler)
    finalize(failure_handler)
    WORLD_RANK == 0 && rm(failure_dir; recursive=true, force=true)
    MPI.Barrier(WORLD)
end

MPI.Finalized() || MPI.Finalize()
