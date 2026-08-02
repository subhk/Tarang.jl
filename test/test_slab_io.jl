# Unit + serial tests for the NetCDF slab I/O layer.
#
# `slab_overlap` is pure index math: no files, no MPI. It decides which part of a
# stored slab feeds which part of a rank's local array, so an off-by-one here
# silently loads the wrong region — the failure mode a round-trip test cannot see.

using Test
using Tarang

@testset "slab_overlap: identical boxes" begin
    r = Tarang.slab_overlap([0, 0], [4, 6], [0, 0], [4, 6])
    @test r !== nothing
    @test r.src_offset == [0, 0]
    @test r.dst_offset == [0, 0]
    @test r.extent == [4, 6]
end

@testset "slab_overlap: disjoint in one dimension returns nothing" begin
    @test Tarang.slab_overlap([0], [4], [4], [4]) === nothing
    @test Tarang.slab_overlap([4], [4], [0], [4]) === nothing
    # Overlaps in dim 1, disjoint in dim 2 — the whole box misses.
    @test Tarang.slab_overlap([0, 0], [8, 3], [0, 3], [8, 3]) === nothing
end

@testset "slab_overlap: partial overlap" begin
    # source covers global [2,6), destination [4,9) -> shared [4,6)
    r = Tarang.slab_overlap([2], [4], [4], [5])
    @test r.src_offset == [2]     # 4 - 2
    @test r.dst_offset == [0]     # 4 - 4
    @test r.extent == [2]         # 6 - 4
end

@testset "slab_overlap: destination contained in source" begin
    r = Tarang.slab_overlap([0], [16], [4], [4])
    @test r.src_offset == [4]
    @test r.dst_offset == [0]
    @test r.extent == [4]
end

@testset "slab_overlap: source contained in destination" begin
    r = Tarang.slab_overlap([4], [4], [0], [16])
    @test r.src_offset == [0]
    @test r.dst_offset == [4]
    @test r.extent == [4]
end

@testset "slab_overlap: uneven 4-rank split read back on 2 ranks" begin
    # Written at np=4 on a length-6 axis: starts 0,1,3,4 with counts 1,2,1,2
    # (the real PencilArrays remainder-on-last split, verified on this machine).
    # Reading rank 1 of 2 wants global [3,6).
    src = [([0], [1]), ([1], [2]), ([3], [1]), ([4], [2])]
    hits = [Tarang.slab_overlap(s, c, [3], [3]) for (s, c) in src]
    @test hits[1] === nothing
    @test hits[2] === nothing
    @test hits[3].src_offset == [0] && hits[3].dst_offset == [0] && hits[3].extent == [1]
    @test hits[4].src_offset == [0] && hits[4].dst_offset == [1] && hits[4].extent == [2]
    # The hits must tile the destination exactly.
    @test sum(h.extent[1] for h in hits if h !== nothing) == 3
end

@testset "slab_overlap: dimension mismatch is an error" begin
    @test_throws ArgumentError Tarang.slab_overlap([0, 0], [4], [0, 0], [4, 4])
end

@testset "write_local_slab round-trips a single file" begin
    dir = mktempdir()
    path = joinpath(dir, "one.nc")
    data = reshape(collect(1.0:24.0), 4, 6)

    Tarang.write_local_slab(path, "u", data, [0, 0], [4, 6])
    src = Tarang.open_slab_source(path)

    @test src.files == [path]
    @test src.global_shape["u"] == [4, 6]
    @test length(src.entries["u"]) == 1

    dest = zeros(Float64, 4, 6)
    Tarang.read_local_slab!(dest, src, "u", [0, 0])
    @test dest == data
end

@testset "read_local_slab! assembles a destination from several slab files" begin
    dir = mktempdir()
    global_data = reshape(collect(1.0:48.0), 8, 6)
    # Write as if from 4 ranks splitting the LAST axis 1/2/1/2 (the uneven
    # PencilArrays split this machine produces at np=4 on a length-6 axis).
    starts = [0, 1, 3, 4]
    counts = [1, 2, 1, 2]
    for (r, (s, c)) in enumerate(zip(starts, counts))
        Tarang.write_local_slab(joinpath(dir, "chk_p$(r-1).nc"), "u",
                                global_data[:, (s+1):(s+c)], [0, s], [8, 6])
    end

    src = Tarang.open_slab_source(dir)
    @test length(src.files) == 4
    @test length(src.entries["u"]) == 4

    # Read back on 1 rank: the whole thing.
    whole = zeros(Float64, 8, 6)
    Tarang.read_local_slab!(whole, src, "u", [0, 0])
    @test whole == global_data

    # Read back on 2 ranks: each half must match, and neither may touch the other.
    left = zeros(Float64, 8, 3)
    Tarang.read_local_slab!(left, src, "u", [0, 0])
    @test left == global_data[:, 1:3]

    right = zeros(Float64, 8, 3)
    Tarang.read_local_slab!(right, src, "u", [0, 3])
    @test right == global_data[:, 4:6]
end

@testset "read_local_slab! errors rather than leaving a partly-filled buffer" begin
    dir = mktempdir()
    data = reshape(collect(1.0:24.0), 4, 6)
    # Store only the first half of the last axis but claim a global shape of 6.
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"), "u", data[:, 1:3], [0, 0], [4, 6])
    src = Tarang.open_slab_source(dir)

    dest = zeros(Float64, 4, 6)
    # A silent partial fill would leave zeros in [:, 4:6] — the exact silent-zero
    # class this assertion exists to prevent.
    #
    # The message is checked, not just the type: a coverage GAP and a duplicate
    # OVERLAP both raise a plain `ErrorException` from the same function, so a
    # regression that swapped one branch's message for the other's would leave
    # both `@test_throws ErrorException` assertions passing unchanged.
    err = try
        Tarang.read_local_slab!(dest, src, "u", [0, 0])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("does not span this range", msg)
    @test occursin("covers 12 of 24 elements", msg)
end

@testset "open_slab_source and read_local_slab! report missing variables" begin
    dir = mktempdir()
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"), "u",
                            reshape(collect(1.0:12.0), 4, 3), [0, 0], [4, 3])
    src = Tarang.open_slab_source(dir)
    @test !haskey(src.entries, "nope")
    dest = zeros(Float64, 4, 3)
    @test_throws ErrorException Tarang.read_local_slab!(dest, src, "nope", [0, 0])

    @test_throws ErrorException Tarang.open_slab_source(joinpath(dir, "does_not_exist"))
end

@testset "several variables share one slab file" begin
    dir = mktempdir()
    path = joinpath(dir, "multi.nc")
    a = reshape(collect(1.0:12.0), 4, 3)
    b = reshape(collect(101.0:112.0), 4, 3)
    Tarang.write_local_slab(path, "a", a, [0, 0], [4, 3])
    Tarang.write_local_slab(path, "b", b, [0, 0], [4, 3])

    src = Tarang.open_slab_source(path)
    da = zeros(Float64, 4, 3); Tarang.read_local_slab!(da, src, "a", [0, 0])
    db = zeros(Float64, 4, 3); Tarang.read_local_slab!(db, src, "b", [0, 0])
    @test da == a
    @test db == b
end

@testset "read_local_slab! errors on overlapping duplicate slabs that would fool a coverage sum" begin
    dir = mktempdir()
    data = reshape(collect(1.0:8.0), 4, 2)
    # Two files claim the SAME region of a declared 4x4 global array. A running
    # element-count check would see 8 + 8 = 16 == prod([4, 4]) and pass, even
    # though columns 3:4 are never written while columns 1:2 are written twice.
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"),     "u", data, [0, 0], [4, 4])
    Tarang.write_local_slab(joinpath(dir, "chk_p0_dup.nc"), "u", data, [0, 0], [4, 4])
    src = Tarang.open_slab_source(dir)

    dest = zeros(Float64, 4, 4)
    # The old sum-based check passed here while leaving dest[:, 3:4] silently
    # zero; the coverage-mask check must throw instead. The message is checked
    # so this cannot be satisfied by the coverage-GAP error, which the same
    # function raises with the same `ErrorException` type (see the gap testset
    # above) — this input trips the gap condition too if the overlap check is
    # removed.
    err = try
        Tarang.read_local_slab!(dest, src, "u", [0, 0])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("has overlapping slabs at the region starting at", msg)
    @test occursin("duplicate coverage", msg)
end

@testset "open_slab_source rejects slabs that disagree on global_shape" begin
    dir = mktempdir()
    # Two files claim the same variable but different global shapes — a leftover
    # from a run at another resolution sharing the directory. Last-writer-wins
    # would silently keep whichever file sorted last, and the coverage check
    # downstream would then measure against the wrong global array.
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"), "u",
                            reshape(collect(1.0:8.0), 4, 2), [0, 0], [4, 4])
    Tarang.write_local_slab(joinpath(dir, "chk_p1.nc"), "u",
                            reshape(collect(1.0:8.0), 4, 2), [0, 2], [4, 8])

    err = try
        Tarang.open_slab_source(dir)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("[4, 4]", msg)
    @test occursin("[4, 8]", msg)
    @test occursin("chk_p0.nc", msg)
    @test occursin("chk_p1.nc", msg)
end

# --- Final review wave: a NetCDFFileHandler output directory must actually
# be readable ---
#
# `SlabSource`'s docstring claims the three-attribute rule "is what lets a
# directory written by NetCDFFileHandler be opened directly", and the design
# spec lists reading one as a test row. It did not work and nothing tested it:
# the handler creates its data variables with a leading unlimited `sim_time`
# dimension while `build_layout_metadata` stamps start/count/global_shape
# covering only the component and spatial dims, so the vectors handed to NetCDF
# were one shorter than the variable's rank. The C layer read past the end of
# them — sometimes "NetCDF: Index exceeds dimension bound", sometimes plausible
# garbage, depending on what followed the vectors in memory.

@testset "a real NetCDFFileHandler output directory reads back through the slab layer" begin
    dir = mktempdir()
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)

    handler = Tarang.NetCDFFileHandler(joinpath(dir, "snaps"), dist, Dict{String,Any}();
                                       iter=1, parallel="gather")
    Tarang.add_task!(handler, u; name="u", layout="g")

    # Two writes, with DIFFERENT data: the second must be what comes back, so a
    # reader that silently grabbed index 1 of the unlimited dimension fails here.
    set!(u, (x, y) -> sin(x) + 0.5cos(3y))
    ensure_layout!(u, :g)
    first_write = copy(Array(get_grid_data(u)))
    Tarang.process!(handler; iteration=0, sim_time=0.0, wall_time=0.0, timestep=0.01)

    set!(u, (x, y) -> 2cos(2x) - 0.25sin(y))
    ensure_layout!(u, :g)
    second_write = copy(Array(get_grid_data(u)))
    Tarang.process!(handler; iteration=1, sim_time=0.01, wall_time=0.1, timestep=0.01)

    @test first_write != second_write

    setdir = joinpath(dir, "snaps_s1")
    @test isdir(setdir)

    src = Tarang.open_slab_source(setdir)
    @test haskey(src.entries, "u")
    @test src.global_shape["u"] == [16, 8]
    # The stored variable really does carry the extra leading time axis — if the
    # handler ever stopped writing one, this test would still pass but would no
    # longer be testing the padding it exists for.
    @test length(src.entries["u"][1].dim_lengths) == length(src.entries["u"][1].count) + 1
    @test src.entries["u"][1].dim_lengths[1] == 2   # two writes on the unlimited axis

    dest = zeros(Float64, 16, 8)
    Tarang.read_local_slab!(dest, src, "u", [0, 0])
    @test dest == second_write

    # And through the public field reader, which is how a user would do it.
    v = ScalarField(dist, "v", (xb, yb), Float64)
    load_field!(v, setdir, "u")
    ensure_layout!(v, :g)
    @test Array(get_grid_data(v)) == second_write
end

@testset "write_local_slab refuses complex data" begin
    dir = mktempdir()
    path = joinpath(dir, "complex.nc")
    data = complex.(reshape(collect(1.0:12.0), 4, 3), reshape(collect(1.0:12.0), 4, 3))
    @test eltype(data) <: Complex
    @test_throws ErrorException Tarang.write_local_slab(path, "u", data, [0, 0], [4, 3])
end

@testset "read_local_slab! accepts a non-String AbstractString variable name" begin
    dir = mktempdir()
    data = reshape(collect(1.0:12.0), 4, 3)
    Tarang.write_local_slab(joinpath(dir, "chk_p0.nc"), "u", data, [0, 0], [4, 3])
    src = Tarang.open_slab_source(dir)

    # `split` returns SubString{String} pieces, not String — read_local_slab!
    # delegates to a helper with a ::String-typed parameter, so this would
    # MethodError inside the call unless it converts at the delegation site.
    varname = split("u|other", "|")[1]
    @test varname isa SubString

    dest = zeros(Float64, 4, 3)
    Tarang.read_local_slab!(dest, src, varname, [0, 0])
    @test dest == data
end
