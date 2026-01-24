using Documenter

# Set up documentation generation
println("Building Tarang.jl documentation...")

# Try to load Tarang, but don't fail if it errors
tarang_loaded = false
try
    @eval using Tarang
    global tarang_loaded = true
    println("  Tarang.jl loaded from: $(pathof(Tarang))")
catch e
    println("  Failed to load Tarang module: $e")
    println("  Building docs without module - API docs will be incomplete")
end

# Set up documentation metadata only if Tarang loaded
if tarang_loaded
    DocMeta.setdocmeta!(Tarang, :DocTestSetup, :(using Tarang); recursive=true)
end

#####
##### Documentation configuration
#####

# HTML format configuration
format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://subhk.github.io/Tarang.jl/stable",
    assets = [
        "assets/custom.css",
    ],
    mathengine = MathJax3(),
    collapselevel = 2,
    sidebar_sitename = true,
    edit_link = "main",
    repolink = "https://github.com/subhk/Tarang.jl",
    size_threshold = 200 * 1024^2,  # 200 MiB
    size_threshold_warn = 10 * 1024^2   # 10 MiB warning
)

# Documentation pages structure
pages = Any[
    "Home" => "index.md",
    "Getting Started" => Any[
        "Installation" => "getting_started/installation.md",
        "First Steps" => "getting_started/first_steps.md",
        "Running with MPI" => "getting_started/running_with_mpi.md",
        "Configuration" => "pages/configuration.md",
    ],
    "Tutorials & Examples" => Any[
        "Overview" => "tutorials/overview.md",
        "2D Rayleigh-Benard" => "tutorials/ivp_2d_rbc.md",
        "3D Turbulence" => "tutorials/ivp_3d_turbulence.md",
        "Boundary Conditions" => "tutorials/boundary_conditions.md",
        "Analysis & Output" => "tutorials/analysis_and_output.md",
        "Eigenvalue Problems" => "tutorials/eigenvalue_problems.md",
        "Surface Dynamics" => "tutorials/surface_dynamics.md",
        "Rotating Shallow Water" => "tutorials/rotating_shallow_water.md",
        "Examples Gallery" => "examples/gallery.md",
    ],
    "User Guide" => Any[
        "Coordinates" => "pages/coordinates.md",
        "Bases" => "pages/bases.md",
        "Domains" => "pages/domains.md",
        "Fields" => "pages/fields.md",
        "Operators" => "pages/operators.md",
        "Problems" => "pages/problems.md",
        "Solvers" => "pages/solvers.md",
        "Time Steppers" => "pages/timesteppers.md",
        "GPU Computing" => "pages/gpu_computing.md",
        "Parallelism" => "pages/parallelism.md",
        "Stochastic Forcing" => "pages/stochastic_forcing.md",
        "LES Models" => "pages/les_models.md",
        "GQL Approximation" => "pages/gql_approximation.md",
        "Tau Method" => "pages/tau_method.md",
    ],
    "Reference" => Any[
        "API Reference" => Any[
            "Coordinates" => "api/coordinates.md",
            "Bases" => "api/bases.md",
            "Domains" => "api/domains.md",
            "Fields" => "api/fields.md",
            "Operators" => "api/operators.md",
            "Problems" => "api/problems.md",
            "Solvers" => "api/solvers.md",
            "Time Steppers" => "api/timesteppers.md",
            "GPU" => "api/gpu.md",
            "Stochastic Forcing" => "api/stochastic_forcing.md",
            "LES Models" => "api/les_models.md",
            "Analysis" => "api/analysis.md",
            "I/O" => "api/io.md",
        ],
    ],
]

#####
##### Build documentation
#####

println("Generating documentation with Documenter.jl...")

makedocs(;
    modules = tarang_loaded ? [Tarang] : Module[],
    authors = "Subhajit Kar",
    repo = "https://github.com/subhk/Tarang.jl/blob/{commit}{path}#{line}",
    sitename = "Tarang.jl",
    format = format,
    pages = pages,
    clean = true,
    doctest = tarang_loaded,
    linkcheck = false,
    checkdocs = :none,
    warnonly = [:cross_references, :missing_docs, :docs_block],
    draft = false
)

#####
##### Deploy documentation
#####

if get(ENV, "CI", "false") == "true"
    println("Deploying documentation...")

    deploydocs(;
        repo = "github.com/subhk/Tarang.jl.git",
        devbranch = "main",
        target = "build",
        deps = nothing,
        make = nothing,
        versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
        forcepush = false,
        deploy_config = Documenter.GitHubActions(),
        push_preview = true
    )
else
    println("Skipping deployment (not running in CI)")
    println("Documentation built successfully!")
    println("Open docs/build/index.html to view locally")
end

println("Documentation build complete!")
