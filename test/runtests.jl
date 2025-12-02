using Test
using Tarang

const TEST_FILES = [
    "test_cfl.jl",
    "test_domain_metadata.jl",
    "test_solvers.jl",
    "test_flow_tools.jl",
    "test_quick_domains.jl",
    "test_plot_tools.jl",
    "test_ball_domain.jl",
]

for file in TEST_FILES
    include(file)
end
