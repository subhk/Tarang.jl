"""
`group_ncread`/`group_ncwrite` must validate their `start`/`count` vectors against
the variable's rank ON DISK before handing them to NetCDF's C API.

`nc_get_vara_*`/`nc_put_vara_*` read exactly `ndims(variable)` entries from the
`start` and `count` pointers, whatever the length of the Julia array behind them.
A short vector therefore makes the C library read past the end of Julia-owned
memory: undefined behaviour that surfaces as silently wrong data on one run and an
"Index exceeds dimension bound" error on the next, depending on what happens to sit
after the array.

These functions are the group-aware hand-rolled path. NetCDF.jl's own root-level
`ncread` already validates ("Length of start (2) must equal the number of variable
dimensions (3)"); this path did not.
"""

using Test
using Tarang
using NetCDF

# A 3-D group variable. `sim_time` first mirrors NetCDFFileHandler's layout, whose
# leading unlimited dimension is what made a caller pass a rank-2 start in the
# first place.
function _make_group_var(dir)
    path = joinpath(dir, "grp.nc")
    Tarang.create_empty_netcdf4_file!(path)
    Tarang.group_nccreate(path, "vars", "u", "sim_time", 2, "x", 4, "y", 3; t = Float64)
    Tarang.group_ncwrite(reshape(collect(1.0:24.0), 2, 4, 3), path, "vars", "u")
    return path
end

@testset "group_ncread rejects a start/count shorter than the variable rank" begin
    path = _make_group_var(mktempdir())

    # Sanity: the full read works and round-trips, so any failure below is the
    # validation firing and not a broken fixture.
    @test Tarang.group_ncread(path, "vars", "u") == reshape(collect(1.0:24.0), 2, 4, 3)

    # rank-2 start/count against a rank-3 variable: the UB case.
    err = try
        Tarang.group_ncread(path, "vars", "u"; start = [1, 1], count = [4, 3])
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("start", msg) || occursin("count", msg)
    @test occursin("3", msg)          # names the variable's true rank

    # A start alone that is too short must be caught too — `count` defaults to the
    # variable's own shape, so the two vectors disagree in length.
    @test_throws Exception Tarang.group_ncread(path, "vars", "u"; start = [1, 1])

    # And one that is too LONG.
    @test_throws Exception Tarang.group_ncread(path, "vars", "u";
                                               start = [1, 1, 1, 1], count = [2, 4, 3, 1])
end

@testset "group_ncread rejects a hyperslab outside the variable" begin
    path = _make_group_var(mktempdir())

    # Correct rank, but reads past the end of dimension 2 (size 4).
    @test_throws Exception Tarang.group_ncread(path, "vars", "u";
                                               start = [1, 3, 1], count = [2, 4, 3])
    # Start below 1 (these indices are 1-based at this layer).
    @test_throws Exception Tarang.group_ncread(path, "vars", "u";
                                               start = [0, 1, 1], count = [2, 4, 3])
    # Negative count that is not the -1 "to the end" sentinel.
    @test_throws Exception Tarang.group_ncread(path, "vars", "u";
                                               start = [1, 1, 1], count = [2, -3, 3])
end

@testset "group_ncread still serves every legitimate hyperslab" begin
    path = _make_group_var(mktempdir())
    full = reshape(collect(1.0:24.0), 2, 4, 3)

    @test Tarang.group_ncread(path, "vars", "u"; start = [1, 1, 1], count = [2, 4, 3]) == full
    @test Tarang.group_ncread(path, "vars", "u"; start = [2, 2, 1], count = [1, 2, 2]) ==
          full[2:2, 2:3, 1:2]
    # -1 means "to the end of this dimension" and must survive validation.
    @test Tarang.group_ncread(path, "vars", "u"; start = [1, 2, 1], count = [-1, -1, -1]) ==
          full[1:2, 2:4, 1:3]
    # Exactly touching the far edge is legal, not off-by-one.
    @test Tarang.group_ncread(path, "vars", "u"; start = [2, 4, 3], count = [1, 1, 1]) ==
          full[2:2, 4:4, 3:3]
end

@testset "group_ncwrite validates against the variable rank, not the array rank" begin
    dir = mktempdir()
    path = _make_group_var(dir)

    # A rank-2 array written into a rank-3 variable: `group_ncwrite`'s own length
    # check compares against `ndims(array)`, so it passes a 2-entry C vector to a
    # call that reads 3 — the same UB as the read path.
    @test_throws Exception Tarang.group_ncwrite(zeros(4, 3), path, "vars", "u")

    # Correct rank still writes and round-trips.
    replacement = reshape(collect(101.0:124.0), 2, 4, 3)
    Tarang.group_ncwrite(replacement, path, "vars", "u")
    @test Tarang.group_ncread(path, "vars", "u") == replacement

    # A correctly-ranked write that would run off the end must be refused.
    @test_throws Exception Tarang.group_ncwrite(reshape(collect(1.0:6.0), 1, 3, 2),
                                                path, "vars", "u"; start = [2, 3, 2])
end
