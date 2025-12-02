# Tarang.jl

A Julia implementation of the Dedalus spectral PDE framework, designed for high-performance parallel computing with MPI and optimized for 2D problems using PencilFFTs.

## Features

- **Spectral Methods**: Fourier, Chebyshev, and Legendre bases for high-accuracy PDE solutions
- **MPI Parallelization**: Full MPI support with both vertical and horizontal domain decomposition
- **PencilArrays Integration**: Efficient distributed array operations using PencilArrays.jl
- **PencilFFTs Support**: Parallel 2D FFT operations for optimal performance
- **Multiple Problem Types**: Initial Value Problems (IVP), Boundary Value Problems (LBVP/NLBVP), and Eigenvalue Problems (EVP)
- **Flexible Coordinate Systems**: Cartesian, polar, spherical coordinate systems
- **Advanced Time Stepping**: Multiple Runge-Kutta and implicit-explicit schemes
- **Built-in Analysis**: CFL conditions, flow properties, and HDF5 output

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhajitkar/Tarang.jl")
```

### Dependencies

The package requires the following Julia packages:
- MPI.jl
- PencilArrays.jl  
- PencilFFTs.jl
- LinearAlgebra
- SparseArrays
- FFTW.jl
- HDF5.jl

## Quick Start

```julia
using Tarang
using MPI

# Initialize MPI
MPI.Init()

# Create 2D domain for Rayleigh-Bénard convection
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))  # 2×2 process mesh

# Create bases
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

# Create domain and fields
domain = Domain(dist, (x_basis, z_basis))
u = VectorField(dist, coords, "velocity", (x_basis, z_basis))
p = ScalarField(dist, "pressure", (x_basis, z_basis))
b = ScalarField(dist, "buoyancy", (x_basis, z_basis))

# Set up and solve problem
problem = IVP([u.components[1], u.components[2], p, b])
solver = InitialValueSolver(problem, RK222())

# Time stepping
while proceed(solver)
    step!(solver, 0.001)
end

MPI.Finalize()
```

## Examples

### 2D Rayleigh-Bénard Convection

A complete example demonstrating thermal convection:

```bash
cd examples/
mpiexec -n 4 julia rayleigh_benard_2d.jl
```

This example showcases:
- 2D domain with periodic horizontal and no-slip vertical boundaries
- Parallel domain decomposition in both directions
- PencilFFTs for efficient 2D FFT operations
- Adaptive time stepping with CFL condition
- Real-time analysis and HDF5 output

## Key Advantages Over Python Dedalus

### Performance
- **Native Julia Speed**: No Python overhead, compiled to native machine code
- **Optimized Memory Layout**: Column-major arrays matching Julia's native layout
- **Zero-Copy Operations**: Efficient memory usage with minimal allocations

### Parallelization
- **2D Domain Decomposition**: Both vertical and horizontal parallelization for 2D problems
- **PencilFFTs Integration**: State-of-the-art parallel FFT library designed for spectral methods
- **Scalable Architecture**: Efficient scaling to hundreds of MPI processes

### Modern Design
- **Type Safety**: Julia's strong type system prevents many runtime errors
- **Multiple Dispatch**: Elegant operator overloading and method specialization
- **Composability**: Easy integration with Julia's extensive scientific computing ecosystem

## Architecture

### Core Components

- **Coordinate Systems**: `CartesianCoordinates`, `SphericalCoordinates`, `PolarCoordinates`
- **Spectral Bases**: `RealFourier`, `ComplexFourier`, `ChebyshevT`, `ChebyshevU`, `Legendre`
- **Domains**: Multi-dimensional spectral domains with MPI distribution
- **Fields**: `ScalarField`, `VectorField`, `TensorField` with automatic layout management
- **Operators**: `grad`, `div`, `curl`, `lap`, differential operators with symbolic parsing
- **Solvers**: `InitialValueSolver`, `BoundaryValueSolver`, `EigenvalueSolver`
- **Time Steppers**: `RK222`, `RK443`, `CNAB2`, `SBDF2`, etc.

### Parallel Computing Model

```
Process Mesh (2D example):
┌─────────┬─────────┐
│ Rank 0  │ Rank 1  │  ← Horizontal decomposition
├─────────┼─────────┤
│ Rank 2  │ Rank 3  │
└─────────┴─────────┘
     ↑
Vertical decomposition
```

- **Pencil Distribution**: Data distributed along pencil shapes for optimal FFT performance
- **Automatic Load Balancing**: Even distribution of computational work
- **Communication Optimization**: Minimized MPI communication overhead

## Advanced Boundary Condition System

Tarang.jl features a comprehensive boundary condition system following the Dedalus tau method:

### Supported Boundary Condition Types

```julia
# Constant boundary conditions
add_dirichlet_bc!(problem, "velocity", "z", 0.0, 0.0)

# Time-dependent boundary conditions
add_dirichlet_bc!(problem, "velocity", "z", 0.0, "sin(2*π*t)")

# Space-dependent boundary conditions
add_neumann_bc!(problem, "temperature", "x", 0.0, "x^2 + y^2")

# Combined time and space dependent
add_dirichlet_bc!(problem, "pressure", "x", 0.0, "sin(ω*t)*exp(-y^2)")

# Robin (mixed) boundary conditions: α*u + β*du/dcoord = value
add_robin_bc!(problem, "concentration", "x", 0.0, 1.0, 0.5, 2.0)

# Stress-free boundary conditions for fluid mechanics
add_stress_free_bc!(problem, "velocity", "z", 0.0)

# Custom boundary condition expressions
add_custom_bc!(problem, "u(x=0) + 2*dx(u)(x=0) = 1")
```

### Key Features
- **Tau Method Integration**: Automatic tau field generation and lift operator support
- **Time/Space Dependence**: Full support for time and spatially varying boundary conditions
- **Automatic Updates**: Boundary conditions automatically updated during time stepping
- **Type Safety**: Structured boundary condition types with validation
- **Flexible Specification**: Support for symbolic expressions, functions, and field references
- **Error Checking**: Comprehensive validation and consistency checking
- **Legacy Compatibility**: Works alongside traditional string-based BCs
- **Performance Optimized**: Efficient evaluation and caching of dependent expressions

## Supported Problem Types

### Initial Value Problems (IVP)
Time evolution problems like Navier-Stokes equations:

```julia
problem = IVP([u, p, T])
add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) = -u·grad(u)")
add_equation!(problem, "div(u) = 0")
add_equation!(problem, "dt(T) - kappa*lap(T) = -u·grad(T)")

# Add boundary conditions with advanced system
add_dirichlet_bc!(problem, "u", "z", 0.0, 0.0)  # No-slip at bottom
add_dirichlet_bc!(problem, "u", "z", 1.0, 0.0)  # No-slip at top
```

### Boundary Value Problems (LBVP/NLBVP)
Steady-state problems with boundary conditions:

```julia
problem = LBVP([u, p])
add_equation!(problem, "-nu*lap(u) + grad(p) = f")
add_equation!(problem, "div(u) = 0")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = U0")
```

### Eigenvalue Problems (EVP)
Linear stability analysis:

```julia
problem = EVP([u, p], eigenvalue=:sigma)
add_equation!(problem, "sigma*u - nu*lap(u) + grad(p) = -U0·grad(u)")
add_equation!(problem, "div(u) = 0")
```

## Performance Optimization

### Memory Efficiency
- **In-place Operations**: Minimal memory allocations during time stepping
- **Layout Optimization**: Automatic switching between grid and spectral layouts
- **Sparse Matrices**: Efficient storage of differentiation operators

### Computational Efficiency
- **SIMD Vectorization**: Automatic vectorization of array operations
- **Cache Optimization**: Memory access patterns optimized for modern CPUs
- **Parallel FFTs**: PencilFFTs provides optimal parallel FFT performance

### Scalability
- **Weak Scaling**: Problem size increases proportionally with process count
- **Strong Scaling**: Fixed problem size distributed across more processes
- **Communication Overlap**: Computation and communication overlap where possible

## Configuration

Create `tarang.toml` in your project directory:

```toml
[parallelism]
TRANSPOSE_LIBRARY = "PENCIL"
GROUP_TRANSPOSES = true
SYNC_TRANSPOSES = true

[transforms]
GROUP_TRANSFORMS = true
DEALIAS_BEFORE_CONVERTING = true

[transforms-fftw]  
PLANNING_RIGOR = "FFTW_ESTIMATE"

[logging]
LEVEL = "INFO"
FILE = "tarang.log"
```

## Development Status

Tarang.jl is currently in active development. The core functionality is implemented and tested, but some advanced features are still being added.

### Completed Features ✅
- [x] Core spectral bases (Fourier, Chebyshev, Legendre)
- [x] MPI parallelization with PencilArrays
- [x] **Full 3D PencilFFTs integration** for optimal 3D performance
- [x] **3D process mesh optimization** with automatic load balancing
- [x] **Complete 3D differential operators** (grad, div, curl, lap)
- [x] **3D memory optimization** with specialized memory pools
- [x] **Advanced boundary condition handling** with tau method support
- [x] IVP solver with multiple time steppers
- [x] Field I/O and analysis tools
- [x] Configuration system
- [x] 2D Rayleigh-Bénard convection example
- [x] **3D Taylor-Green vortex example**
- [x] **3D turbulent channel flow example**
- [x] **Advanced boundary condition examples**

### In Progress 🚧
- [ ] Nonlinear solver for NLBVP
- [ ] Eigenvalue solver implementation
- [ ] Comprehensive test suite

### Future Plans 📋
- [ ] Spherical coordinate support
- [ ] Advanced analysis tools
- [ ] Performance benchmarks
- [ ] Documentation website
- [ ] Integration with other Julia packages

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Tarang.jl is licensed under the GPL-3.0 license, maintaining compatibility with the original Dedalus project.

## Citation

If you use Tarang.jl in your research, please cite:

```bibtex
@software{tarang_jl,
  title = {Tarang.jl: A Julia Implementation of Spectral PDE Methods},
  author = {Subhajit Kar},
  year = {2024},
  url = {https://github.com/subhajitkar/Tarang.jl}
}
```

## Acknowledgments

This project is inspired by and builds upon the excellent [Dedalus](https://dedalus-project.org) project. We thank the Dedalus developers for creating such a well-designed foundation for spectral methods.

Special thanks to the Julia community and the developers of PencilArrays.jl and PencilFFTs.jl for providing the essential building blocks for high-performance parallel computing in Julia.