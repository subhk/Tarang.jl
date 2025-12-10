# Tarang.jl Documentation

*A flexible spectral PDE solver for Julia with MPI parallelization*

```@raw html
<div style="text-align: center; margin: 2em 0;">
  <img src="assets/logo_light.svg" alt="Tarang.jl" width="600" style="max-width: 100%;" />
</div>
```

## Overview

Tarang.jl is a Julia implementation of spectral methods for solving partial differential equations (PDEs). It provides a flexible framework for solving a wide variety of PDEs using spectral methods with efficient MPI parallelization.

### Key Features

- **Flexible Spectral Methods**: Support for Fourier, Chebyshev, and Legendre bases
- **MPI Parallelization**: Efficient parallel computing using PencilArrays and PencilFFTs
- **Multiple Problem Types**: Initial value problems (IVP), boundary value problems (BVP), and eigenvalue problems (EVP)
- **Advanced Boundary Conditions**: Comprehensive boundary condition system with tau method
- **Symbolic Operators**: Natural mathematical syntax for differential operators
- **High Performance**: Optimized for modern HPC systems with support for GPU acceleration
- **2D and 3D**: Full support for both 2D and 3D problems with optimal parallelization strategies

## Quick Start

```julia
using Tarang, MPI

# Initialize MPI
MPI.Init()

# Create a 2D domain for Rayleigh-Bénard convection
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

# Define spectral bases
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

# Create domain and fields
domain = Domain(dist, (x_basis, z_basis))
u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))

# Set up initial value problem
problem = IVP([u.components[1], u.components[2], p, T])
add_equation!(problem, "∂t(u) - Pr*Δ(u) + ∇(p) = -u⋅∇(u) + Ra*Pr*T*ez")
add_equation!(problem, "div(u) = 0")
add_equation!(problem, "∂t(T) - Δ(T) = -u⋅∇(T)")

# Add parameters
problem.parameters["Pr"] = 1.0
problem.parameters["Ra"] = 1e5

# Add boundary conditions
add_dirichlet_bc!(problem, "u(z=0) = 0")
add_dirichlet_bc!(problem, "u(z=1) = 0")
add_dirichlet_bc!(problem, "T(z=0) = 1")
add_dirichlet_bc!(problem, "T(z=1) = 0")

# Create solver with timestepper
solver = InitialValueSolver(problem, RK222(), dt=1e-3)

# Time stepping
while solver.sim_time < 1.0
    step!(solver, 1e-3)
end

MPI.Finalize()
```

## Why Tarang.jl?

### Performance
- **Native Julia Speed**: Compiled to native machine code with no Python overhead
- **Memory Efficiency**: Zero-copy operations and optimized memory layouts
- **Scalable Parallelism**: Efficient MPI communication with PencilArrays and PencilFFTs

### Flexibility
- **Multiple Coordinate Systems**: Cartesian, spherical, and polar coordinates
- **Rich Operator Library**: grad, div, curl, lap, and custom differential operators
- **Symbolic Equation Parsing**: Natural mathematical notation for PDEs

### Usability
- **Type Safety**: Julia's type system catches errors at compile time
- **Interactive Development**: REPL-driven workflow for rapid prototyping
- **Comprehensive Documentation**: Tutorials, examples, and API reference

## Example Problems

Tarang.jl can solve a wide range of problems in fluid dynamics, heat transfer, and more:

- **Fluid Dynamics**: Navier-Stokes equations, Rayleigh-Bénard convection, channel flow
- **Heat Transfer**: Thermal convection, diffusion problems
- **Linear Stability**: Eigenvalue analysis of fluid flows
- **Magnetohydrodynamics**: MHD equations with magnetic fields
- **Atmospheric/Oceanic Dynamics**: Stratified flows, gravity waves, wave-mean separation
- **Turbulence Modeling**: Large Eddy Simulation (LES) with Smagorinsky and AMD models

### Featured Tutorials

- **[Rotating Shallow Water](tutorials/rotating_shallow_water.md)**: Wave-mean flow separation using temporal filters
- **[Temporal Filters](pages/temporal_filters.md)**: Lagrangian averaging for geophysical flows
- **[LES Models](pages/les_models.md)**: Subgrid-scale modeling for turbulence

## Documentation Structure

```@contents
Pages = [
    "getting_started/installation.md",
    "getting_started/first_steps.md",
    "tutorials/overview.md",
]
Depth = 1
```

## Citing Tarang.jl

If you use Tarang.jl in your research, please cite:

```bibtex
@software{tarang_jl,
  title = {Tarang.jl: A Spectral PDE Solver for Julia},
  author = {Subhajit Kar},
  year = {2024},
  url = {https://github.com/subhk/Tarang.jl},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## Acknowledgments

Tarang.jl builds upon excellent Julia packages including [PencilArrays.jl](https://github.com/jipolanco/PencilArrays.jl), [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl), and [MPI.jl](https://github.com/JuliaParallel/MPI.jl).

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/subhk/Tarang.jl/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/subhk/Tarang.jl/discussions)
- **Contributing**: See the [Contributing Guide](pages/contributing.md) for how to contribute

## License

Tarang.jl is licensed under the MIT license. See [LICENSE](https://github.com/subhk/Tarang.jl/blob/main/LICENSE) for details.

---

```@raw html
<div style="text-align: center; margin-top: 3em; font-size: 0.9em; color: #666;">
  <p>Powered by <a href="https://julialang.org">Julia</a> |
     Documentation built with <a href="https://github.com/JuliaDocs/Documenter.jl">Documenter.jl</a></p>
</div>
```
