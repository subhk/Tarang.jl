using Aqua
using Tarang

@testset "Aqua.jl" begin
    Aqua.test_all(
        Tarang;
        ambiguities=false,       # skip due to broad operator dispatch
        piracies=false,          # skip due to Base method extensions
        persistent_tasks=false,  # skip due to MPI.Init() in __init__
    )
end
