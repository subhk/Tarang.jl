using Aqua
using Tarang

@testset "Aqua.jl" begin
    Aqua.test_all(
        Tarang;
        # Exercise ambiguity detection and report its one known case as broken
        # instead of disabling the check. Piracy is clean and enforced.
        ambiguities=(broken=true,),
        piracies=true,
        persistent_tasks=false,  # skip due to MPI.Init() in __init__
        stale_deps=(ignore=[:ArgParse],),  # ArgParse is used by scripts/merge_netcdf.jl
    )
end
