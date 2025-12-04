# Contributing to Tarang.jl

Thank you for your interest in contributing to Tarang.jl!

## Getting Started

### Development Setup

```bash
git clone https://github.com/subhajitkar/Tarang.jl.git
cd Tarang.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Contribution Guidelines

### Code Style

- Use 4-space indentation
- Document public functions with docstrings
- Follow Julia naming conventions
- Add tests for new features

### Docstrings

```julia
"""
    my_function(x, y)

Brief description.

# Arguments
- `x`: First argument
- `y`: Second argument

# Returns
Result description

# Examples
\```julia
result = my_function(1, 2)
\```
"""
function my_function(x, y)
    # ...
end
```

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run tests locally
5. Submit pull request

### Issues

- Use issue templates
- Provide minimal reproducible examples
- Include version information

## Development Areas

### High Priority

- Performance optimization
- GPU acceleration
- Additional coordinate systems

### Medium Priority

- More examples and tutorials
- Enhanced visualization
- Additional basis types

### Documentation

- API documentation
- Tutorials
- Examples

## Testing

### Running Specific Tests

```julia
using Pkg
Pkg.test("Tarang", test_args=["test_specific.jl"])
```

### Writing Tests

```julia
@testset "My Feature" begin
    # Setup
    coords = CartesianCoordinates("x")
    # ...

    # Test
    @test result ≈ expected atol=1e-10
end
```

## Questions?

- Open a GitHub issue
- Start a discussion

Thank you for contributing!
