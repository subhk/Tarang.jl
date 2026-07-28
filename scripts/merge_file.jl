#!/usr/bin/env julia

# Compatibility entry point. The implementation and help text live in one
# place so the two historical script names cannot drift apart.
include(joinpath(@__DIR__, "merge_netcdf.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    @warn "merge_file.jl is deprecated; use merge_netcdf.jl"
    exit(main())
end
