using JET
using Tarang

@testset "JET.jl" begin
    JET.test_package(Tarang; target_defined_modules=true)
end
