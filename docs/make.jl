using Documenter

# Try to load Tarang, but don't fail if it errors
tarang_loaded = false
try
    @eval using Tarang
    global tarang_loaded = true
    @info "Successfully loaded Tarang from: $(pathof(Tarang))"
catch e
    @warn "Failed to load Tarang module: $e"
    @warn "Building docs without module - API docs will be incomplete"
end

# Set up documentation metadata only if Tarang loaded
if tarang_loaded
    DocMeta.setdocmeta!(Tarang, :DocTestSetup, :(using Tarang); recursive=true)
end

makedocs(
    sitename = "Tarang.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://subhk.github.io/Tarang.jl",
        assets = ["assets/custom.css"],
        mathengine = MathJax3(),
    ),
    modules = tarang_loaded ? [Tarang] : Module[],
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "getting_started/installation.md",
            "getting_started/first_steps.md",
            "getting_started/running_with_mpi.md",
        ],
        "Tutorials" => [
            "tutorials/overview.md",
            "tutorials/ivp_2d_rbc.md",
            "tutorials/ivp_3d_turbulence.md",
            "tutorials/boundary_conditions.md",
            "tutorials/analysis_and_output.md",
            "tutorials/eigenvalue_problems.md",
            "tutorials/surface_dynamics.md",
        ],
        "Notebooks" => [
            "notebooks/rayleigh_benard.md",
            "notebooks/channel_flow.md",
            "notebooks/taylor_green.md",
        ],
        "User Guide" => [
            "pages/coordinates.md",
            "pages/bases.md",
            "pages/domains.md",
            "pages/fields.md",
            "pages/operators.md",
            "pages/problems.md",
            "pages/solvers.md",
            "pages/timesteppers.md",
            "pages/stochastic_forcing.md",
            "pages/temporal_filters.md",
            "pages/analysis.md",
            "pages/parallelism.md",
            "pages/configuration.md",
        ],
        "Advanced Topics" => [
            "pages/tau_method.md",
            "pages/optimization.md",
            "pages/gpu_computing.md",
            "pages/custom_operators.md",
        ],
        "API Reference" => [
            "api/coordinates.md",
            "api/bases.md",
            "api/domains.md",
            "api/fields.md",
            "api/operators.md",
            "api/problems.md",
            "api/solvers.md",
            "api/timesteppers.md",
            "api/stochastic_forcing.md",
            "api/analysis.md",
            "api/io.md",
        ],
        "Examples" => [
            "examples/gallery.md",
            "examples/fluid_dynamics.md",
            "examples/heat_transfer.md",
            "examples/eigenvalue_analysis.md",
        ],
        "Development" => [
            "pages/contributing.md",
            "pages/architecture.md",
            "pages/testing.md",
        ],
    ],
    repo = "https://github.com/subhk/Tarang.jl/blob/{commit}{path}#{line}",
    authors = "Subhajit Kar",
    checkdocs = :none,  # Temporarily disabled until all docstrings are documented
)

deploydocs(
    repo = "github.com/subhk/Tarang.jl.git",
    devbranch = "main",
    push_preview = true,
)
