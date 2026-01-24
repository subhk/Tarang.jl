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
        collapselevel = 2,
        sidebar_sitename = true,
        footer = "Tarang.jl -- A spectral PDE solver for Julia. [Source](https://github.com/subhk/Tarang.jl)",
    ),
    modules = tarang_loaded ? [Tarang] : Module[],
    # Dedalus-inspired navigation structure
    pages = [
        "Home" => "index.md",

        # Section 1: Installing Tarang (like Dedalus)
        "Installing Tarang" => [
            "getting_started/installation.md",
            "getting_started/first_steps.md",
            "getting_started/running_with_mpi.md",
            "pages/configuration.md",
        ],

        # Section 2: Tutorials & Examples (like Dedalus)
        "Tutorials & Examples" => [
            "Tutorial Notebooks" => [
                "tutorials/overview.md",
                "tutorials/ivp_2d_rbc.md",
                "tutorials/ivp_3d_turbulence.md",
                "tutorials/boundary_conditions.md",
                "tutorials/analysis_and_output.md",
                "tutorials/eigenvalue_problems.md",
                "tutorials/surface_dynamics.md",
                "tutorials/rotating_shallow_water.md",
            ],
            "Example Scripts" => [
                "examples/gallery.md",
                "examples/fluid_dynamics.md",
                "examples/heat_transfer.md",
                "examples/eigenvalue_analysis.md",
            ],
            "Jupyter Notebooks" => [
                "notebooks/rayleigh_benard.md",
                "notebooks/channel_flow.md",
                "notebooks/taylor_green.md",
            ],
        ],

        # Section 3: User Guide & How-To's (like Dedalus)
        "User Guide" => [
            "Core Concepts" => [
                "pages/coordinates.md",
                "pages/bases.md",
                "pages/domains.md",
                "pages/fields.md",
            ],
            "Problem Setup" => [
                "pages/operators.md",
                "pages/problems.md",
                "pages/solvers.md",
                "pages/timesteppers.md",
            ],
            "Physics & Modeling" => [
                "pages/stochastic_forcing.md",
                "pages/temporal_filters.md",
                "pages/les_models.md",
                "pages/gql_approximation.md",
            ],
            "Performance & Parallelism" => [
                "pages/gpu_computing.md",
                "pages/parallelism.md",
                "pages/optimization.md",
            ],
            "Analysis & Output" => [
                "pages/analysis.md",
            ],
            "Advanced Topics" => [
                "pages/tau_method.md",
                "pages/custom_operators.md",
            ],
        ],

        # Section 4: API Reference (like Dedalus)
        "API Reference" => [
            "Core" => [
                "api/coordinates.md",
                "api/bases.md",
                "api/domains.md",
                "api/fields.md",
            ],
            "Operators & Problems" => [
                "api/operators.md",
                "api/problems.md",
                "api/solvers.md",
                "api/timesteppers.md",
            ],
            "GPU & Performance" => [
                "api/gpu.md",
            ],
            "Extras" => [
                "api/stochastic_forcing.md",
                "api/les_models.md",
                "api/analysis.md",
                "api/io.md",
            ],
        ],

        # Development section
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
