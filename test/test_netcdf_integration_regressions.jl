using Test
using MPI
using NetCDF
using Tarang

MPI.Initialized() || MPI.Init()

function _serial_output_field(name::String="u")
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 2pi))
    field = ScalarField(dist, name, (basis,), Float64)
    ensure_layout!(field, :g)
    return dist, field
end

function _write_merge_slab(base::String, rank::Int;
                           start::Int, count::Int, global_count::Int,
                           mpi_size::Int, value::Float64=Float64(rank),
                           slab_metadata::Bool=true, grid_space::Int=1,
                           declare_mpi_size::Bool=true)
    setdir = "$(base)_s1"
    mkpath(setdir)
    file = joinpath(setdir, "$(basename(base))_s1_p$(rank).nc")
    Tarang.create_empty_netcdf4_file!(file)
    Tarang.group_nccreate(file, "time", "sim_time", "sim_time", 1; t=Float64)
    Tarang.group_ncwrite([0.0], file, "time", "sim_time")
    attrs = Dict{String,Any}(
        "grid_space" => grid_space,
        "layout" => grid_space == 1 ? "g" : "c",
    )
    if slab_metadata
        attrs["global_shape"] = [global_count]
        attrs["start"] = [start]
        attrs["count"] = [count]
    end
    Tarang.group_nccreate(file, "vars", "q", "sim_time", 1, "x", count;
                          t=Float64, atts=attrs)
    Tarang.group_ncwrite(fill(value, 1, count), file, "vars", "q")
    global_attrs = Dict{String,Any}("processor_rank" => rank)
    declare_mpi_size && (global_attrs["mpi_size"] = mpi_size)
    NetCDF.ncputatt(file, "global", global_attrs)
    return file
end

function _write_metadata_less_scalar(base::String, rank::Int;
                                     mpi_size::Int, value::Float64)
    setdir = "$(base)_s1"
    mkpath(setdir)
    file = joinpath(setdir, "$(basename(base))_s1_p$(rank).nc")
    Tarang.create_empty_netcdf4_file!(file)
    Tarang.group_nccreate(file, "time", "sim_time", "sim_time", 1; t=Float64)
    Tarang.group_ncwrite([0.0], file, "time", "sim_time")
    Tarang.group_nccreate(file, "vars", "q", "sim_time", 1; t=Float64,
                          atts=Dict("grid_space" => 1, "layout" => "g"))
    Tarang.group_ncwrite([value], file, "vars", "q")
    NetCDF.ncputatt(file, "global", Dict{String,Any}(
        "mpi_size" => mpi_size,
        "processor_rank" => rank,
    ))
    return file
end

if get(ENV, "TARANG_TEST_NETCDF_OUTPUT", "1") != "0"
@testset "NetCDF output integration regressions" begin
    @testset "overwrite cleanup matches only the exact handler name" begin
        tmp = mktempdir()
        exact_dir = joinpath(tmp, "snap_s1")
        sibling_dir = joinpath(tmp, "snap_stats_s1")
        similarly_named = joinpath(tmp, "snapshot_s2")
        exact_file = joinpath(tmp, "snap_s2.nc")
        mkpath(exact_dir)
        mkpath(sibling_dir)
        mkpath(similarly_named)
        write(exact_file, "old output")

        dist, _ = _serial_output_field()
        Tarang.NetCDFFileHandler(joinpath(tmp, "snap"), dist, Dict{String,Any}();
                                 mode="overwrite")

        @test !ispath(exact_dir)
        @test !ispath(exact_file)
        @test isdir(sibling_dir)
        @test isdir(similarly_named)
    end

    @testset "append resumes the existing record and counters" begin
        tmp = mktempdir()
        dist, u = _serial_output_field()
        base = joinpath(tmp, "history")

        fill!(get_grid_data(u), 1.0)
        first = Tarang.NetCDFFileHandler(base, dist, Dict("u" => u);
                                         mode="overwrite", max_writes=4)
        Tarang.add_task!(first, u; name="u")
        @test Tarang.process!(first; iteration=0, sim_time=0.0, wall_time=0.0,
                              timestep=0.1)

        fill!(get_grid_data(u), 2.0)
        resumed = Tarang.NetCDFFileHandler(base, dist, Dict("u" => u);
                                           mode="append", max_writes=4)
        Tarang.add_task!(resumed, u; name="u")
        @test resumed.set_num == 1
        @test resumed.file_write_num == 1
        @test resumed.total_write_num == 1
        @test Tarang.process!(resumed; iteration=1, sim_time=1.0, wall_time=1.0,
                              timestep=0.1)

        file = Tarang.current_file(resumed)
        @test Tarang.group_ncread(file, "time", "sim_time") == [0.0, 1.0]
        values = Tarang.group_ncread(file, "vars", "u")
        @test values[1, :] == fill(1.0, 4)
        @test values[2, :] == fill(2.0, 4)
        @test resumed.file_write_num == 2
        @test resumed.total_write_num == 2
    end

    @testset "handler adopts the Distributor communicator" begin
        tmp = mktempdir()
        dist, _ = _serial_output_field()
        handler = Tarang.NetCDFFileHandler(joinpath(tmp, "self_comm"), dist,
                                           Dict{String,Any}(); mode="append")
        Tarang.init_mpi!(handler)
        @test handler.comm === MPI.COMM_SELF
        @test handler.rank == MPI.Comm_rank(MPI.COMM_SELF)
        @test handler.size == MPI.Comm_size(MPI.COMM_SELF)
    end

    @testset "file-creation and postprocess failures restore handler state" begin
        tmp = mktempdir()
        dist, u = _serial_output_field()

        blocker = joinpath(tmp, "not_a_directory")
        write(blocker, "block mkdir")
        bad_path = Tarang.NetCDFFileHandler(joinpath(blocker, "snap"), dist,
                                            Dict("u" => u); sim_dt=1.0)
        @test_throws Exception Tarang.process!(bad_path; iteration=0, sim_time=0.0,
                                               wall_time=0.0, timestep=0.1)
        @test bad_path.total_write_num == 0
        @test bad_path.file_write_num == 0
        @test bad_path.set_num == 1
        @test bad_path.last_sim_div == -1

        bad_post = Tarang.NetCDFFileHandler(joinpath(tmp, "bad_post"), dist,
                                            Dict("u" => u); sim_dt=1.0)
        Tarang.add_task!(bad_post, u; name="u",
                         postprocess=_ -> error("injected postprocess failure"))
        @test_throws ErrorException Tarang.process!(bad_post; iteration=0, sim_time=0.0,
                                                    wall_time=0.0, timestep=0.1)
        @test bad_post.total_write_num == 0
        @test bad_post.file_write_num == 0
        @test bad_post.set_num == 1
        @test bad_post.last_sim_div == -1
        @test isempty(bad_post._created_vars)
    end

    @testset "retry reconciles variables left by a partial append write" begin
        tmp = mktempdir()
        dist, a = _serial_output_field("a")
        b = ScalarField(dist, "b", a.bases, Float64)
        ensure_layout!(b, :g)
        fill!(get_grid_data(a), 3.0)
        fill!(get_grid_data(b), 7.0)

        fail_second = Ref(true)
        handler = Tarang.NetCDFFileHandler(joinpath(tmp, "retry"), dist,
                                           Dict("a" => a, "b" => b);
                                           mode="append")
        Tarang.add_task!(handler, a; name="a")
        Tarang.add_task!(handler, b; name="b", postprocess=data -> begin
            fail_second[] && error("injected second-task failure")
            data
        end)

        @test_throws ErrorException Tarang.process!(handler; iteration=0, sim_time=0.0,
                                                    wall_time=0.0, timestep=0.1)
        @test handler.total_write_num == 0
        @test handler.file_write_num == 0
        fail_second[] = false
        @test Tarang.process!(handler; iteration=0, sim_time=0.0,
                              wall_time=0.0, timestep=0.1)

        file = Tarang.current_file(handler)
        @test Tarang.group_ncread(file, "time", "sim_time") == [0.0]
        @test Tarang.group_ncread(file, "vars", "a") == reshape(fill(3.0, 4), 1, 4)
        @test Tarang.group_ncread(file, "vars", "b") == reshape(fill(7.0, 4), 1, 4)
        @test handler.total_write_num == 1
        @test handler.file_write_num == 1
    end
end
end

if get(ENV, "TARANG_TEST_NETCDF_MERGE", "1") != "0"
@testset "NetCDF merger fails closed" begin
    @testset "missing slab is incomplete and never cleaned up" begin
        tmp = mktempdir()
        cd(tmp) do
            source = _write_merge_slab("missing", 0;
                                       start=0, count=2, global_count=4, mpi_size=2)
            @test !Tarang.merge_netcdf_files("missing"; cleanup=true, verbose=false)
            @test isfile(source)
            @test !isfile(joinpath("missing_s1", "missing_s1.nc"))
        end
    end

    @testset "overlapping slabs are rejected and retained" begin
        tmp = mktempdir()
        cd(tmp) do
            p0 = _write_merge_slab("overlap", 0;
                                   start=0, count=4, global_count=4, mpi_size=2,
                                   value=1.0)
            p1 = _write_merge_slab("overlap", 1;
                                   start=0, count=4, global_count=4, mpi_size=2,
                                   value=2.0)
            @test !Tarang.merge_netcdf_files("overlap"; cleanup=true, verbose=false)
            @test isfile(p0)
            @test isfile(p1)
            @test !isfile(joinpath("overlap_s1", "overlap_s1.nc"))
        end
    end

    @testset "multi-rank reconstruction refuses missing slab metadata" begin
        tmp = mktempdir()
        cd(tmp) do
            p0 = _write_merge_slab("metadata_less", 0;
                                   start=0, count=2, global_count=4, mpi_size=2,
                                   value=1.0, slab_metadata=false)
            p1 = _write_merge_slab("metadata_less", 1;
                                   start=2, count=2, global_count=4, mpi_size=2,
                                   value=2.0, slab_metadata=false)
            @test !Tarang.merge_netcdf_files("metadata_less";
                                              cleanup=true, verbose=false)
            @test isfile(p0)
            @test isfile(p1)
            @test !isfile(joinpath("metadata_less_s1", "metadata_less_s1.nc"))
        end
    end

    @testset "metadata-free replicated scalars are not guessed as reconstruction" begin
        tmp = mktempdir()
        cd(tmp) do
            p0 = _write_metadata_less_scalar("metadata_less_scalar", 0;
                                             mpi_size=2, value=1.0)
            p1 = _write_metadata_less_scalar("metadata_less_scalar", 1;
                                             mpi_size=2, value=2.0)
            @test !Tarang.merge_netcdf_files("metadata_less_scalar";
                                              cleanup=true, verbose=false)
            @test isfile(p0)
            @test isfile(p1)
            @test !isfile(joinpath("metadata_less_scalar_s1",
                                   "metadata_less_scalar_s1.nc"))
        end
    end

    @testset "output cannot alias an input file" begin
        tmp = mktempdir()
        cd(tmp) do
            source = _write_merge_slab("alias", 0;
                                       start=0, count=2, global_count=2, mpi_size=1)
            merger = Tarang.NetCDFMerger("alias"; output_name=source,
                                         cleanup=true, verbose=false)
            @test !Tarang.merge_files!(merger)
            @test isfile(source)
        end
    end

    @testset "filesystem aliases of an input are rejected without mutation" begin
        tmp = mktempdir()
        cd(tmp) do
            source = _write_merge_slab("alias_link", 0;
                                       start=0, count=2, global_count=2, mpi_size=1)
            output_alias = joinpath("alias_link_s1", "linked_output.nc")
            symlink(abspath(source), output_alias)
            merger = Tarang.NetCDFMerger("alias_link"; output_name=output_alias,
                                         cleanup=false, verbose=false)
            @test !Tarang.merge_files!(merger)
            @test isfile(source)
            @test islink(output_alias)
            @test Base.Filesystem.samefile(source, output_alias)
        end
    end

    @testset "coefficient reconstruction rejects even a minor coverage gap" begin
        tmp = mktempdir()
        cd(tmp) do
            p0 = _write_merge_slab("coeff_gap", 0;
                                   start=0, count=10, global_count=20, mpi_size=2,
                                   value=1.0, grid_space=0)
            p1 = _write_merge_slab("coeff_gap", 1;
                                   start=11, count=9, global_count=20, mpi_size=2,
                                   value=2.0, grid_space=0)
            @test !Tarang.merge_netcdf_files(
                "coeff_gap"; merge_mode=Tarang.DOMAIN_DECOMP,
                cleanup=true, verbose=false)
            @test isfile(p0)
            @test isfile(p1)
            @test !isfile(joinpath("coeff_gap_s1", "coeff_gap_s1.nc"))
        end
    end

    @testset "multi-rank inputs must declare their complete rank count" begin
        tmp = mktempdir()
        cd(tmp) do
            p0 = _write_merge_slab("unknown_size", 0;
                                   start=0, count=2, global_count=4, mpi_size=2,
                                   declare_mpi_size=false)
            p1 = _write_merge_slab("unknown_size", 1;
                                   start=2, count=2, global_count=4, mpi_size=2,
                                   declare_mpi_size=false)
            @test !Tarang.merge_netcdf_files(
                "unknown_size"; cleanup=true, verbose=false)
            @test isfile(p0)
            @test isfile(p1)
            @test !isfile(joinpath("unknown_size_s1", "unknown_size_s1.nc"))
        end
    end

    @testset "complete disjoint slabs may be cleaned after verification" begin
        tmp = mktempdir()
        cd(tmp) do
            p0 = _write_merge_slab("complete", 0;
                                   start=0, count=2, global_count=4, mpi_size=2,
                                   value=1.0)
            p1 = _write_merge_slab("complete", 1;
                                   start=2, count=2, global_count=4, mpi_size=2,
                                   value=2.0)
            @test Tarang.merge_netcdf_files("complete"; cleanup=true, verbose=false)
            @test !isfile(p0)
            @test !isfile(p1)
            merged = joinpath("complete_s1", "complete_s1.nc")
            @test isfile(merged)
            @test Tarang.group_ncread(merged, "vars", "q") ==
                  reshape([1.0, 1.0, 2.0, 2.0], 1, 4)
        end
    end
end
end

@testset "NetCDF merger accepts a base path with a directory component" begin
    # The handler's base_path normally carries a directory ("out/snap"); the
    # merger used to build its match pattern from the full path and compare it
    # against basename(file), so it silently found no processor files.
    tmp = mktempdir()
    base = joinpath(tmp, "out", "dirbase")
    mkpath(dirname(base))
    p0 = _write_merge_slab(base, 0; start=0, count=2, global_count=4, mpi_size=2, value=1.0)
    p1 = _write_merge_slab(base, 1; start=2, count=2, global_count=4, mpi_size=2, value=2.0)
    merger = Tarang.NetCDFMerger(base; verbose=false)
    @test basename.(merger.processor_files) == ["dirbase_s1_p0.nc", "dirbase_s1_p1.nc"]
    @test merger.output_file == joinpath(tmp, "out", "dirbase_s1", "dirbase_s1.nc")
    @test Tarang.merge_files!(merger)
    @test vec(Tarang.group_ncread(merger.output_file, "vars", "q")) == [1.0, 1.0, 2.0, 2.0]
    @test isfile(p0) && isfile(p1)
end

@testset "derivative expression task resolves the coordinate from the field" begin
    # `add_task!(handler, "∂x(u)")` is documented syntax; it used to reach
    # `Differentiate(field, :x, 1)` and die with a MethodError because the
    # handler's `vars` carries fields, not coordinates.
    dist, u = _serial_output_field("u")
    xs = vec(collect(local_grids(dist, u.bases[1])[1]))
    grid_data!(u) .= sin.(xs)
    tmp = mktempdir()
    h = NetCDFFileHandler(joinpath(tmp, "expr"), dist, Dict("u" => u); mode="overwrite")
    add_task!(h, "∂x(u)"; name="dux")
    @test process!(h; iteration=0, sim_time=0.0, wall_time=0.0, timestep=0.1)
    f = current_file(h)
    close!(h)
    @test vec(Tarang.group_ncread(f, "vars", "dux")) ≈ cos.(xs) atol=1e-12
    # An unknown coordinate is a clear ArgumentError naming the alternatives.
    err = try
        Tarang.create_differentiate_operator(u, "y", 1, Dict{String, Any}())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("no coordinate named \"y\"", sprint(showerror, err))
end
