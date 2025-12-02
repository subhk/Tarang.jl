# Tarang.jl Documentation Guide

This guide helps you build, deploy, and maintain the Tarang.jl documentation website modeled after the Dedalus documentation style.

## 📚 Documentation Structure

The documentation is organized into comprehensive sections:

### 1. **Getting Started**
- Installation instructions (Julia, MPI, dependencies)
- First steps tutorial (basic workflow)
- Running with MPI (parallel execution guide)

### 2. **Tutorials**
- Overview and learning path
- 2D Rayleigh-Bénard convection (complete fluid dynamics example)
- 3D turbulence simulations
- Boundary conditions deep-dive
- Analysis and output
- Eigenvalue problems

### 3. **User Guide** (pages/)
- Coordinates and coordinate systems
- Spectral bases (Fourier, Chebyshev, Legendre)
- Domains and discretization
- Fields (scalar, vector, tensor)
- Differential operators
- Problem types (IVP, BVP, EVP)
- Solvers and time steppers
- Parallelism and MPI configuration
- Configuration and optimization
- GPU computing

### 4. **API Reference**
- Complete API documentation with docstrings
- Organized by module
- Usage examples for each function
- Type hierarchies

### 5. **Examples Gallery**
- Categorized by physics (fluid dynamics, heat transfer, etc.)
- Categorized by complexity (beginner, intermediate, advanced)
- Complete runnable code for each example
- Visualization examples

## 🚀 Quick Start: Build Documentation

### Prerequisites

```bash
# Install Julia (1.6+)
# Install MPI (OpenMPI or MPICH)
# Install Tarang.jl
```

### Build Locally

```bash
# 1. Navigate to docs directory
cd docs/

# 2. Install documentation dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Build documentation
julia --project=. make.jl

# 4. Preview (choose one method)
# Method A: Python
cd build/
python3 -m http.server 8000

# Method B: Julia LiveServer
julia -e 'using LiveServer; serve(dir="build")'

# 5. Open in browser
# Navigate to http://localhost:8000
```

## 🌐 Deploy to GitHub Pages

### Step 1: Setup GitHub Pages

1. Go to your repository settings on GitHub
2. Navigate to **Settings → Pages**
3. Set source to: **Deploy from a branch**
4. Select branch: **gh-pages**
5. Folder: **/ (root)**
6. Save

### Step 2: Configure Documenter Key

This enables automatic deployment from GitHub Actions.

```bash
# Generate SSH key pair
julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="subhajitkar", repo="Tarang.jl")'
```

This will output:
- **Public key**: Add to **GitHub Settings → Deploy Keys** (check "Allow write access")
- **Private key**: Add to **GitHub Settings → Secrets → Actions** as `DOCUMENTER_KEY`

### Step 3: Trigger Deployment

Documentation builds automatically when you:

```bash
# Push to main branch
git add .
git commit -m "Update documentation"
git push origin main

# Or create a tag
git tag v0.1.0
git push --tags
```

### Step 4: Access Documentation

After successful deployment (check Actions tab), your documentation will be live at:

```
https://subhajitkar.github.io/Tarang.jl/
```

## 📝 Writing Documentation

### Adding a New Page

1. **Create markdown file** in appropriate directory:
   ```bash
   # Example: Add new tutorial
   touch docs/src/tutorials/my_new_tutorial.md
   ```

2. **Edit make.jl** to include the new page:
   ```julia
   # In docs/make.jl
   pages = [
       ...
       "Tutorials" => [
           ...
           "tutorials/my_new_tutorial.md",
       ],
       ...
   ]
   ```

3. **Write content** following the style guide below

4. **Test locally**:
   ```bash
   julia --project=docs docs/make.jl
   ```

5. **Commit and push** - documentation will auto-deploy

### Markdown Style Guide

#### Headers
```markdown
# Top Level (Page Title)
## Major Section
### Subsection
#### Detail Level
```

#### Code Blocks
````markdown
```julia
using Tarang

# Create a field
T = ScalarField(dist, "T", bases)
```
````

#### Math Equations
```markdown
Inline: $E = mc^2$

Display:
```math
\frac{\partial T}{\partial t} = \kappa \nabla^2 T
```
````

#### Admonitions
```markdown
!!! note "Important Note"
    This is important information.

!!! warning "Be Careful"
    This warns about potential issues.

!!! tip "Helpful Tip"
    This provides helpful tips.
```

#### Tables
```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

#### Links
```markdown
# External link
[Dedalus Project](https://dedalus-project.org)

# Internal link
See [Installation](getting_started/installation.md)

# API reference
See [`ScalarField`](@ref) for details.
```

### Documenting Functions

Add docstrings to your Julia code:

```julia
"""
    my_function(x, y; option=false)

Brief one-line description of the function.

Detailed description explaining what the function does,
any important considerations, and typical use cases.

# Arguments
- `x::Float64`: Description of x parameter
- `y::Int`: Description of y parameter
- `option::Bool`: Optional parameter (default: false)

# Returns
- `result::Float64`: Description of what is returned

# Examples
```julia
# Basic usage
result = my_function(1.0, 2)

# With options
result = my_function(1.0, 2, option=true)
```

# See Also
- [`related_function`](@ref): Related functionality
- [Tutorial](../tutorials/overview.md): Detailed tutorial
"""
function my_function(x, y; option=false)
    # Implementation
    return x + y
end
```

Include in documentation with:
```markdown
```@docs
my_function
```
````

## 🎨 Customization

### Custom CSS

Edit `docs/src/assets/custom.css` to customize appearance:

```css
/* Change primary color */
:root {
    --primary-color: #2c5aa0;  /* Dedalus-inspired blue */
}

/* Customize tables */
table th {
    background-color: var(--primary-color);
    color: white;
}
```

### Logo and Assets

1. Add logo: `docs/src/assets/tarang_logo.svg`
2. Add images: `docs/src/assets/images/`
3. Reference in markdown:
   ```markdown
   ![Tarang Logo](assets/tarang_logo.svg)
   ```

### Navigation

Modify `docs/make.jl` to change navigation:

```julia
pages = [
    "Home" => "index.md",
    "Getting Started" => [
        "getting_started/installation.md",
        "getting_started/first_steps.md",
    ],
    # Add more sections...
]
```

## 🧪 Testing Documentation

### Test Docstrings

```bash
# Run doctests
julia --project=docs -e 'using Documenter, Tarang; doctest(Tarang)'
```

### Check Links

```bash
# Build with strict mode (fails on warnings)
julia --project=docs -e 'ENV["STRICT"] = "true"; include("make.jl")'
```

### Check Examples

```bash
# Test all examples are runnable
cd examples/
for f in *.jl; do
    echo "Testing $f..."
    mpiexec -n 4 julia --project=.. "$f" || echo "FAILED: $f"
done
```

## 📊 Documentation Checklist

When adding documentation, ensure:

- [ ] Page added to `make.jl` navigation
- [ ] All code examples are complete and runnable
- [ ] Links are working (internal and external)
- [ ] Math equations render correctly
- [ ] Images load properly
- [ ] Docstrings follow style guide
- [ ] Cross-references use `@ref` correctly
- [ ] Examples in docstrings work
- [ ] Built locally without errors
- [ ] Previewed in browser
- [ ] Committed and pushed

## 🔧 Troubleshooting

### Build Errors

**Problem**: "LoadError: Package X not found"

**Solution**: Install missing dependency:
```bash
julia --project=docs -e 'using Pkg; Pkg.add("X")'
```

**Problem**: "ERROR: Documenter could not find..."

**Solution**: Check `make.jl` paths match actual file locations

### Deployment Errors

**Problem**: Documentation not updating on website

**Solution**:
1. Check GitHub Actions tab for errors
2. Verify `DOCUMENTER_KEY` is correctly configured
3. Check `gh-pages` branch exists and has content

**Problem**: "Permission denied (publickey)"

**Solution**: Regenerate and reconfigure Documenter key

### Preview Issues

**Problem**: CSS not loading in local preview

**Solution**: Use `prettyurls = false` in `make.jl` for local builds:
```julia
format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    # ...
)
```

## 📈 Maintenance

### Regular Updates

- Update API documentation when code changes
- Add examples for new features
- Keep tutorials current with latest API
- Fix broken links and outdated information
- Update performance benchmarks

### Version Documentation

For versioned documentation:

```julia
# In docs/make.jl
makedocs(
    # ...
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
```

This creates:
- `stable/`: Latest release
- `v1.0/`, `v1.1/`, etc.: Specific versions
- `dev/`: Development (main branch)

## 🎯 Best Practices

1. **Code-First**: Show working examples early
2. **Progressive Complexity**: Start simple, build up
3. **Complete Examples**: Full, runnable code
4. **Visual Aids**: Add diagrams and plots
5. **Cross-Reference**: Link related topics
6. **User-Focused**: Write for your audience
7. **Test Everything**: All code should run
8. **Stay Current**: Update with code changes

## 📚 Resources

- **Documenter.jl**: https://juliadocs.github.io/Documenter.jl/
- **Dedalus Docs** (inspiration): https://dedalus-project.readthedocs.io/
- **Julia Docs Style**: https://docs.julialang.org/
- **Markdown Guide**: https://www.markdownguide.org/

## 🤝 Contributing

See [docs/README.md](docs/README.md) for contributor guidelines.

For questions:
- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions
- **Pull Requests**: Contribute improvements

---

## 📋 Quick Reference

### Build Commands
```bash
# Install deps
julia --project=docs -e 'using Pkg; Pkg.instantiate()'

# Build
julia --project=docs docs/make.jl

# Build with strict checking
STRICT=true julia --project=docs docs/make.jl

# Preview
python3 -m http.server 8000 --directory docs/build
```

### Deployment Commands
```bash
# Generate Documenter key
julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="USER", repo="REPO")'

# Manual deploy
cd docs/
julia --project make.jl

# Check deployment
# Visit: https://USER.github.io/REPO/
```

### Writing Commands
```markdown
# Code block
```julia
code here
```

# Math
$inline$ or ```math display ```

# Docstring
```@docs
FunctionName
```

# Admonition
!!! note "Title"
    Content
```

---

**Your Tarang.jl documentation is now ready!** 🎉

Visit https://subhajitkar.github.io/Tarang.jl/ after deployment.
