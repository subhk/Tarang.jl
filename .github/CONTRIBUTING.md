# Contributing to Tarang.jl

Thank you for considering contributing to Tarang.jl! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Julia 1.6 or later
- MPI library (OpenMPI or MPICH)
- Git
- GitHub account

### Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Tarang.jl.git
   cd Tarang.jl
   ```

2. **Install dependencies**:
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

3. **Install MPI**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install openmpi-bin libopenmpi-dev

   # macOS
   brew install open-mpi
   ```

4. **Run tests**:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'

   # With MPI
   mpiexec -n 4 julia --project -e 'using Pkg; Pkg.test()'
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check existing [issues](https://github.com/subhajitkar/Tarang.jl/issues)
2. Try the latest version from main branch
3. Collect relevant information

**Create a bug report with**:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Julia version, OS, MPI implementation
- Minimal reproducible example
- Stack trace if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues:
- Use clear, descriptive title
- Explain the use case
- Describe expected behavior
- Provide examples if possible
- Consider implementation approach

### Pull Requests

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-new-feature
   # or
   git checkout -b fix/bug-description
   ```

2. **Make changes**:
   - Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
   - Write tests for new functionality
   - Update documentation
   - Keep commits focused and atomic

3. **Test your changes**:
   ```bash
   # Run tests
   julia --project -e 'using Pkg; Pkg.test()'

   # With MPI
   mpiexec -n 4 julia --project -e 'using Pkg; Pkg.test()'

   # Build documentation
   ./docs/build_docs.sh
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description

   Detailed explanation of changes and why they were needed.

   Fixes #issue_number"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/my-new-feature
   ```
   Then open a Pull Request on GitHub.

## Coding Standards

### Julia Style

Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/):

- Use 4 spaces for indentation (no tabs)
- Line length: 92 characters (soft limit)
- Function names: lowercase with underscores
- Type names: CamelCase
- Constants: UPPERCASE
- Module names: CamelCase

### Code Formatting

Use [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl):

```julia
using JuliaFormatter
format(".")
```

### Documentation

- Document all public functions with docstrings
- Follow Documenter.jl conventions
- Include examples in docstrings
- Update API documentation when adding features

**Docstring template**:
```julia
"""
    function_name(arg1, arg2; kwarg=default)

Brief description of what the function does.

Detailed explanation if needed.

# Arguments
- `arg1::Type`: Description of arg1
- `arg2::Type`: Description of arg2
- `kwarg::Type`: Description of keyword argument (default: `default`)

# Returns
- `ReturnType`: Description of return value

# Examples
```julia
result = function_name(1.0, 2.0)
```

# See Also
- [`related_function`](@ref): Related functionality
"""
function function_name(arg1, arg2; kwarg=default)
    # Implementation
end
```

### Testing

- Write tests for all new functions
- Use descriptive test names
- Test edge cases and error conditions
- Include MPI tests for parallel code

**Test structure**:
```julia
@testset "Feature Name" begin
    @testset "Subfeature 1" begin
        # Tests
        @test result == expected
    end

    @testset "Edge cases" begin
        # Edge case tests
        @test_throws ErrorType function_call(invalid_input)
    end
end
```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `perf/description` - Performance improvements
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

body

footer
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(operators): add custom operator support

Add ability to define custom differential operators that can be
used in symbolic equations.

Closes #123

fix(mpi): correct process mesh validation

Fix validation logic for non-square process meshes. Previously
failed for valid configurations like (4, 2).

Fixes #456
```

### Code Review Process

All submissions require review:
1. Automated CI tests must pass
2. At least one maintainer approval
3. Documentation updated if needed
4. No merge conflicts with main

**Reviewers check**:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance implications
- API compatibility

## Testing Guidelines

### Unit Tests

Test individual functions in isolation:

```julia
@testset "Basis construction" begin
    coords = CartesianCoordinates("x")
    basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))

    @test basis.size == 128
    @test basis.bounds == (0.0, 2π)
    @test typeof(basis) == RealFourier
end
```

### Integration Tests

Test component interactions:

```julia
@testset "2D domain construction" begin
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords, mesh=(2, 2))

    x = RealFourier(coords["x"], size=64, bounds=(0.0, 2π))
    z = ChebyshevT(coords["z"], size=32, bounds=(0.0, 1.0))

    domain = Domain(dist, (x, z))

    @test domain.dim == 2
    @test length(domain.bases) == 2
end
```

### MPI Tests

Test parallel functionality:

```julia
@testset "MPI parallel operations" begin
    MPI.Init()

    # Setup
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords, mesh=(2, 2))

    # Test
    @test MPI.Comm_size(dist.comm) == 4

    MPI.Finalize()
end
```

## Documentation

### Building Documentation

```bash
./docs/build_docs.sh
```

### Adding Examples

Examples go in `docs/src/examples/` or `examples/` directory:

```julia
# examples/my_example.jl
"""
Example: My Feature

This example demonstrates...
"""

using Tarang, MPI

MPI.Init()

# Example code here

MPI.Finalize()
```

### Adding Tutorials

Tutorials go in `docs/src/tutorials/`:
1. Create markdown file
2. Add to `docs/make.jl` navigation
3. Include runnable code examples
4. Add visualizations if applicable

## Performance Considerations

- Profile code before optimization
- Use `@inbounds` and `@simd` where safe
- Minimize allocations in hot loops
- Consider memory layout for MPI
- Document performance characteristics

**Benchmarking**:
```julia
using BenchmarkTools

@benchmark my_function(args)
```

## Release Process

Maintainers follow this process for releases:

1. Update version in `Project.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.x.0`
4. Push tag: `git push --tags`
5. GitHub Actions handles release

## Questions?

- Open an [issue](https://github.com/subhajitkar/Tarang.jl/issues)
- Start a [discussion](https://github.com/subhajitkar/Tarang.jl/discussions)
- Email: subhajitkar19@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 license.

## Code of Conduct

Be respectful, inclusive, and professional. We follow the [Julia Community Standards](https://julialang.org/community/standards/).

Thank you for contributing to Tarang.jl!
