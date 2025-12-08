# Contributing to Tarang.jl

We welcome contributions! Whether fixing bugs, adding features, or improving documentation.

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Tarang.jl.git
cd Tarang.jl

# Install and test
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# Create a feature branch
git checkout -b feature/your-feature-name
```

## Code Style

- **Indentation**: 4 spaces
- **Line length**: 92 characters max
- **Naming**: `snake_case` for functions, `CamelCase` for types
- Add docstrings to public functions

## Testing

```bash
# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# With MPI
mpiexec -n 4 julia --project=. test/runtests.jl
```

New features must include tests. Bug fixes should add regression tests.

## Pull Requests

1. Ensure tests pass
2. Update documentation if needed
3. Use clear commit messages: `feat:`, `fix:`, `docs:`, `test:`
4. Open PR against `main` branch

## Getting Help

- [GitHub Issues](https://github.com/subhk/Tarang.jl/issues) - Bug reports
- [GitHub Discussions](https://github.com/subhk/Tarang.jl/discussions) - Questions

## License

Contributions are licensed under GPL-3.0.
