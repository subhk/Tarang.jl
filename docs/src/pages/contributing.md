# Contributing to Tarang.jl

We welcome contributions from the community. Whether you're fixing bugs, adding features, improving documentation, or reporting issues, your help makes Tarang.jl better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Development Priorities](#development-priorities)

---

## Code of Conduct

Contributors are expected to maintain a respectful and inclusive environment. Please be considerate in discussions and code reviews.

---

## Getting Started

### Prerequisites

- Julia 1.9 or later
- Git
- MPI installation (OpenMPI or MPICH)
- Optional: CUDA toolkit for GPU development

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Tarang.jl.git
   cd Tarang.jl
   ```

2. **Install dependencies**:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

3. **Verify installation**:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```

4. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/subhajitkar/Tarang.jl.git
   ```

---

## Development Workflow

### Branch Strategy

- `main`: Stable release branch
- `dev`: Development branch for integration
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `docs/*`: Documentation improvements

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Keeping Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

---

## Coding Standards

### General Principles

- Write clear, self-documenting code
- Prefer readability over cleverness
- Follow the principle of least surprise
- Keep functions focused and modular

### Julia Style Guidelines

| Aspect | Convention |
|--------|------------|
| Indentation | 4 spaces (no tabs) |
| Line length | 92 characters maximum |
| Naming | `snake_case` for functions, `CamelCase` for types |
| Constants | `UPPER_SNAKE_CASE` |
| Private functions | Prefix with underscore `_helper_function` |

### Code Organization

```julia
# Good: Clear structure
module MyModule

# Imports first
using LinearAlgebra
using FFTW

# Exports
export MyType, my_function

# Type definitions
struct MyType
    field::Float64
end

# Functions
function my_function(x::MyType)
    # Implementation
end

end # module
```

### Docstrings

All public functions must have docstrings following this format:

```julia
"""
    solve_poisson(rhs::ScalarField; method=:spectral) -> ScalarField

Solve the Poisson equation ∇²φ = f for the potential φ.

# Arguments
- `rhs::ScalarField`: Right-hand side forcing term f
- `method::Symbol=:spectral`: Solution method (`:spectral`, `:multigrid`)

# Returns
- `ScalarField`: Solution field φ

# Examples
```julia
# Create RHS field
rhs = ScalarField(dist, "rhs", bases)
rhs["g"] .= sin.(x) .* cos.(z)

# Solve
phi = solve_poisson(rhs)
```

# See Also
- [`solve_helmholtz`](@ref): For Helmholtz equation
- [`lap`](@ref): Laplacian operator
"""
function solve_poisson(rhs::ScalarField; method=:spectral)
    # Implementation
end
```

### Type Annotations

- Use type annotations for function arguments when it aids clarity
- Avoid over-constraining types; prefer abstract types when possible

```julia
# Good: Uses abstract type
function process(field::AbstractField)
    # Works with any field type
end

# Avoid: Too specific without reason
function process(field::ScalarField{Float64, 3})
    # Unnecessarily restrictive
end
```

---

## Testing Guidelines

### Running Tests

```bash
# Full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Specific test file
julia --project=. test/test_transforms.jl

# With MPI (4 processes)
mpiexec -n 4 julia --project=. test/runtests.jl
```

### Writing Tests

Tests should be:
- **Isolated**: No dependencies between tests
- **Deterministic**: Same result every run
- **Fast**: Unit tests should complete quickly
- **Descriptive**: Clear failure messages

```julia
@testset "Transform Operations" begin
    @testset "Forward-inverse roundtrip" begin
        # Setup
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,))
        basis = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
        field = ScalarField(dist, "f", (basis,), Float64)

        # Initialize with known function
        original = sin.(get_grid(basis))
        field["g"] .= original

        # Transform roundtrip
        Tarang.ensure_layout!(field, :c)
        Tarang.ensure_layout!(field, :g)

        # Verify
        @test field["g"] ≈ original atol=1e-12
    end

    @testset "Derivative accuracy" begin
        # Test spectral derivative against analytical
        # ...
    end
end
```

### Test Categories

| Category | Location | Description |
|----------|----------|-------------|
| Unit tests | `test/unit/` | Individual function tests |
| Integration tests | `test/integration/` | Component interaction |
| Regression tests | `test/regression/` | Known results verification |
| Performance tests | `test/benchmarks/` | Timing and memory |

### Coverage Requirements

- New features must include tests
- Bug fixes should add regression tests
- Aim for >80% code coverage on new code

---

## Documentation

### Building Documentation

```bash
cd docs
julia --project=. make.jl
```

### Documentation Types

1. **Docstrings**: In-code API documentation
2. **Tutorials**: Step-by-step guides in `docs/src/tutorials/`
3. **How-to guides**: Task-oriented guides in `docs/src/pages/`
4. **API Reference**: Auto-generated from docstrings

### Writing Tutorials

- Start with a clear objective
- Include complete, runnable code
- Explain the "why" not just the "how"
- Add visualizations where helpful

---

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```

2. **Check code style**: Review against coding standards

3. **Update documentation**: Add/update docstrings and guides

4. **Write meaningful commits**:
   ```
   feat: Add Chebyshev-Legendre basis conversion

   - Implement spectral_convert() for basis transformations
   - Add tests for accuracy verification
   - Update API documentation
   ```

### Commit Message Format

```
<type>: <short summary>

<detailed description if needed>

<references to issues>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`

### Submitting the PR

1. Push your branch to your fork
2. Open a pull request against `main`
3. Fill out the PR template completely
4. Link related issues
5. Request review from maintainers

### Review Process

- All PRs require at least one approving review
- Address review feedback promptly
- Maintain a constructive dialogue
- Squash commits before merge if requested

---

## Development Priorities

### High Priority

| Area | Description | Skills Needed |
|------|-------------|---------------|
| GPU acceleration | CUDA/ROCm kernel optimization | GPU programming, CUDA.jl |
| Performance | Transform and solver optimization | Profiling, algorithms |
| Spherical coordinates | Full spherical harmonic support | Spectral methods |

### Medium Priority

| Area | Description | Skills Needed |
|------|-------------|---------------|
| Visualization | Plotting integration | Makie.jl, Plots.jl |
| I/O formats | HDF5, parallel I/O | File formats, MPI-IO |
| New physics | MHD, multiphase | Domain expertise |

### Good First Issues

Look for issues labeled `good-first-issue` for entry points:
- Documentation improvements
- Test coverage expansion
- Error message enhancement
- Example scripts

---

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community help
- **Documentation**: [Tarang.jl Documentation](https://subhajitkar.github.io/Tarang.jl/)

---

## Recognition

Contributors are recognized in:
- Release notes
- Contributors list in README
- Annual contributor acknowledgments

We value all contributions, from typo fixes to major features.

---

## License

By contributing to Tarang.jl, you agree that your contributions will be licensed under the MIT License.
