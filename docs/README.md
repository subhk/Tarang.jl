# Tarang.jl Documentation

This directory contains the source for Tarang.jl documentation, built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

## Documentation Structure

```
docs/
├── make.jl                 # Documentation build script
├── Project.toml            # Documentation dependencies
├── README.md              # This file
└── src/                   # Documentation source
    ├── index.md           # Home page
    ├── assets/            # CSS, images, etc.
    ├── getting_started/   # Installation and first steps
    │   ├── installation.md
    │   ├── first_steps.md
    │   └── running_with_mpi.md
    ├── tutorials/         # Detailed tutorials
    │   ├── overview.md
    │   ├── ivp_2d_rbc.md
    │   ├── ivp_3d_turbulence.md
    │   ├── boundary_conditions.md
    │   └── analysis_and_output.md
    ├── pages/             # User guide and advanced topics
    │   ├── coordinates.md
    │   ├── bases.md
    │   ├── domains.md
    │   ├── fields.md
    │   ├── operators.md
    │   ├── problems.md
    │   ├── solvers.md
    │   ├── timesteppers.md
    │   ├── parallelism.md
    │   ├── configuration.md
    │   ├── optimization.md
    │   └── gpu_computing.md
    ├── api/               # API reference
    │   ├── coordinates.md
    │   ├── bases.md
    │   ├── domains.md
    │   ├── fields.md
    │   ├── operators.md
    │   ├── problems.md
    │   ├── solvers.md
    │   └── analysis.md
    ├── examples/          # Example gallery
    │   ├── gallery.md
    │   ├── fluid_dynamics.md
    │   └── heat_transfer.md
    └── notebooks/         # Jupyter notebook tutorials
        ├── rayleigh_benard.md
        ├── channel_flow.md
        └── taylor_green.md
```

## Building Documentation Locally

### Prerequisites

1. Julia 1.6 or later
2. Tarang.jl installed
3. MPI library (OpenMPI or MPICH)

### Setup

```bash
# Navigate to docs directory
cd docs/

# Install documentation dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Build

```bash
# Build documentation
julia --project=. make.jl
```

This will generate HTML documentation in `docs/build/`.

### Preview

To preview the documentation locally:

```bash
# Serve with Python
cd build/
python3 -m http.server 8000

# Or use Julia LiveServer
julia -e 'using LiveServer; serve(dir="build")'
```

Then open http://localhost:8000 in your browser.

## Automatic Deployment

Documentation is automatically built and deployed using GitHub Actions:

### GitHub Pages Setup

1. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` (created automatically by workflow)
   - Folder: `/root`

2. **Add Documenter Key**:
   ```bash
   # Generate SSH key
   julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="subhajitkar", repo="Tarang.jl")'
   ```

   This creates:
   - **Public key**: Add to GitHub Settings → Deploy Keys (allow write access)
   - **Private key**: Add to GitHub Settings → Secrets as `DOCUMENTER_KEY`

3. **Trigger Build**:
   - Push to `main` branch
   - Create a new tag
   - Open a pull request

### Workflow

The `.github/workflows/Documentation.yml` workflow:

1. Checks out the repository
2. Sets up Julia and dependencies
3. Builds the documentation
4. Deploys to `gh-pages` branch (on main branch pushes)

### Documentation URL

After successful deployment, documentation will be available at:

```
https://subhajitkar.github.io/Tarang.jl/
```

Or with custom domain (if configured):
```
https://tarang.jl.org/
```

## Writing Documentation

### Markdown Files

Write documentation in Markdown with Julia code blocks:

```markdown
# My Section

This is regular markdown text.

\```julia
# Julia code example
using Tarang
x = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))
\```
```

### Docstrings

Document functions using Julia docstrings:

```julia
"""
    my_function(x, y)

Description of function.

# Arguments
- `x`: First argument
- `y`: Second argument

# Returns
- Result description

# Examples
\```julia
result = my_function(1, 2)
\```
"""
function my_function(x, y)
    return x + y
end
```

Include docstrings in documentation with:

```markdown
\```@docs
my_function
\```
```

### Math Equations

Use LaTeX for math:

```markdown
Inline math: $E = mc^2$

Display math:
\```math
\\frac{\\partial u}{\\partial t} = \\nu \\nabla^2 u
\```
```

### Admonitions

Use info boxes for important notes:

```markdown
!!! note "Note Title"
    This is a note with important information.

!!! warning "Be Careful"
    This warns about potential issues.

!!! tip "Helpful Tip"
    This provides helpful tips.
```

Types: `note`, `warning`, `tip`, `info`, `danger`

### Cross-References

Link to other documentation pages:

```markdown
See [Installation](getting_started/installation.md) for details.

Link to API: [ScalarField](@ref)
```

### Code Examples

Include runnable examples:

````markdown
```julia
using Tarang

# Create a field
T = ScalarField(dist, "T", bases)

# Initialize
fill!(T, 1.0)
```
````

## Documentation Style Guide

### Structure

1. **Start with overview**: Brief introduction to the topic
2. **Show examples early**: Practical examples before theory
3. **Progress from simple to complex**: Build up concepts
4. **Include complete examples**: Full working code when possible
5. **Add troubleshooting**: Common issues and solutions

### Writing Style

- **Use active voice**: "Create a field" not "A field is created"
- **Be concise**: Short paragraphs and sentences
- **Code-first**: Show code examples liberally
- **Explain why**: Not just what, but why use this approach
- **Link related topics**: Help users discover related content

### Code Examples

- **Complete**: Include all necessary imports and setup
- **Runnable**: Code should work if copied
- **Annotated**: Comments explaining key points
- **Realistic**: Use meaningful variable names and realistic parameters

### API Documentation

For each function/type:
1. Brief one-line description
2. Detailed description of behavior
3. Arguments with types and descriptions
4. Return value description
5. Examples showing usage
6. Related functions

Example:
```julia
"""
    RealFourier(coord, size, bounds)

Create a real-valued Fourier basis for a periodic coordinate.

Uses sine/cosine representation for real data, saving memory compared to
complex Fourier basis.

# Arguments
- `coord::Coordinate`: Coordinate this basis discretizes
- `size::Int`: Number of modes (must be even)
- `bounds::Tuple{Float64,Float64}`: Domain boundaries (min, max)

# Returns
- `RealFourier`: Configured Fourier basis

# Examples
\```julia
x = Coordinate("x")
basis = RealFourier(x, size=128, bounds=(0.0, 2π))
\```

# See Also
- [`ComplexFourier`](@ref): Complex Fourier basis
- [`ChebyshevT`](@ref): Chebyshev polynomials for bounded domains
"""
```

## Adding New Pages

1. **Create markdown file** in appropriate directory
2. **Add to make.jl** in `pages` array:
   ```julia
   pages = [
       ...
       "New Section" => "path/to/new_page.md",
       ...
   ]
   ```
3. **Build and test** locally
4. **Commit and push** - documentation will auto-deploy

## Maintenance

### Regular Updates

- Keep examples up-to-date with API changes
- Add tutorials for new features
- Update performance benchmarks
- Fix broken links and typos

### Testing

```bash
# Check for broken links
julia --project=docs -e 'using Documenter; doctest(Tarang)'

# Build with strict checking
julia --project=docs docs/make.jl --strict
```

## Contributing

Contributions to documentation are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Documentation PRs

For documentation changes:
1. Fork the repository
2. Create a branch: `git checkout -b docs/my-improvement`
3. Make changes in `docs/src/`
4. Test locally: `julia --project=docs docs/make.jl`
5. Submit pull request

### Guidelines

- Follow the style guide above
- Test all code examples
- Add images/diagrams when helpful
- Check spelling and grammar
- Link to related pages

## Resources

- [Documenter.jl Documentation](https://juliadocs.github.io/Documenter.jl/)
- [Julia Documentation](https://docs.julialang.org/)
- [Dedalus Documentation](https://dedalus-project.readthedocs.io/) (inspiration)
- [Markdown Guide](https://www.markdownguide.org/)

## Questions?

- **GitHub Issues**: Report documentation bugs
- **GitHub Discussions**: Ask questions about docs
- **Pull Requests**: Suggest improvements

---

**Last Updated**: December 2024
