# Tarang.jl

**A flexible spectral PDE solver for Julia with GPU acceleration and MPI parallelization**

```@raw html
<div style="text-align: center; margin: 2em 0;">
  <img src="assets/logo_light.svg" alt="Tarang.jl" width="500" style="max-width: 100%;" />
</div>
```

## Overview

Tarang.jl is a high-performance Julia framework for solving partial differential equations (PDEs) using spectral methods. Inspired by [Dedalus](https://dedalus-project.org), it provides a flexible symbolic interface for specifying equations while leveraging Julia's speed and composability.

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3>🔬 Spectral Methods</h3>
    <p>Fourier, Chebyshev, and Legendre bases with automatic differentiation and spectral accuracy.</p>
  </div>
  <div class="feature-card">
    <h3>⚡ GPU Acceleration</h3>
    <p>Native CUDA support with optimized FFT plans, GPU kernels, and automatic CPU/GPU dispatch.</p>
  </div>
  <div class="feature-card">
    <h3>🚀 MPI Parallelization</h3>
    <p>Efficient pencil decomposition using PencilArrays.jl and PencilFFTs.jl for scalable HPC.</p>
  </div>
  <div class="feature-card">
    <h3>📝 Symbolic Equations</h3>
    <p>Natural mathematical syntax for PDEs with automatic parsing and operator construction.</p>
  </div>
</div>
```

## Quick Installation

```julia
using Pkg
Pkg.add("Tarang")
```

For GPU support:
```julia
Pkg.add("CUDA")
```

For MPI parallelization:
```julia
Pkg.add("MPI")
```

See the [Installation Guide](getting_started/installation.md) for detailed instructions.

## Quick Start Example

```julia
using Tarang, MPI

MPI.Init()

# Create 2D domain for Rayleigh-Bénard convection
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

# Set up initial value problem with Dedalus-style syntax
problem = IVP([u.components[1], u.components[2], p, T])
add_equation!(problem, "dt(ux) - Pr*lap(ux) + dx(p) = -u@grad(ux) + Ra*Pr*T*ez")
add_equation!(problem, "dt(uz) - Pr*lap(uz) + dz(p) = -u@grad(uz)")
add_equation!(problem, "div(u) = 0")
add_equation!(problem, "dt(T) - lap(T) = -u@grad(T)")

# Parameters
problem.parameters["Pr"] = 1.0
problem.parameters["Ra"] = 1e5

# Boundary conditions
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "u(z=1) = 0")
add_equation!(problem, "T(z=0) = 1")
add_equation!(problem, "T(z=1) = 0")

# Create solver and integrate
solver = InitialValueSolver(problem, RK222(), dt=1e-3)
while solver.sim_time < 1.0
    step!(solver, 1e-3)
end

MPI.Finalize()
```

## Documentation Sections

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3><a href="getting_started/installation/">Installing Tarang</a></h3>
    <p>Installation instructions, first steps, and configuration options.</p>
  </div>
  <div class="feature-card">
    <h3><a href="tutorials/overview/">Tutorials & Examples</a></h3>
    <p>Step-by-step tutorials, example scripts, and Jupyter notebooks.</p>
  </div>
  <div class="feature-card">
    <h3><a href="pages/coordinates/">User Guide</a></h3>
    <p>Detailed documentation on coordinates, bases, fields, operators, and solvers.</p>
  </div>
  <div class="feature-card">
    <h3><a href="api/coordinates/">API Reference</a></h3>
    <p>Complete API documentation with docstrings for all public functions.</p>
  </div>
</div>
```

## Capabilities

### Problem Types

| Type | Description | Example |
|------|-------------|---------|
| **IVP** | Initial Value Problems | Time-dependent Navier-Stokes |
| **BVP** | Boundary Value Problems | Steady state solutions |
| **EVP** | Eigenvalue Problems | Linear stability analysis |
| **LBVP** | Linear BVPs | Poisson equation |
| **NLBVP** | Nonlinear BVPs | Steady nonlinear systems |

### Spectral Bases

| Basis | Domain | Usage |
|-------|--------|-------|
| `RealFourier` | Periodic | Horizontal directions |
| `ComplexFourier` | Periodic | Complex-valued fields |
| `ChebyshevT` | Non-periodic | Bounded domains, BCs |
| `Legendre` | Non-periodic | Alternative to Chebyshev |

### Physical Applications

- **Fluid Dynamics**: Navier-Stokes, Rayleigh-Bénard convection, channel flow
- **Heat Transfer**: Diffusion, advection-diffusion
- **Magnetohydrodynamics**: MHD equations with magnetic fields
- **Geophysical Flows**: Rotating shallow water, stratified turbulence
- **Turbulence Modeling**: LES with Smagorinsky and AMD models

## Featured Tutorials

- [2D Rayleigh-Bénard Convection](tutorials/ivp_2d_rbc.md) — Classic thermal convection
- [3D Homogeneous Turbulence](tutorials/ivp_3d_turbulence.md) — Isotropic turbulence simulation
- [Eigenvalue Problems](tutorials/eigenvalue_problems.md) — Linear stability analysis
- [Rotating Shallow Water](tutorials/rotating_shallow_water.md) — Geophysical fluid dynamics

## GPU Acceleration

Tarang.jl provides first-class GPU support through CUDA.jl:

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3>CUFFT Integration</h3>
    <p>Automatic FFT/DCT acceleration using NVIDIA's optimized cuFFT library.</p>
  </div>
  <div class="feature-card">
    <h3>Custom Kernels</h3>
    <p>Portable CPU/GPU kernels via KernelAbstractions.jl for element-wise operations.</p>
  </div>
  <div class="feature-card">
    <h3>Multi-GPU MPI</h3>
    <p>Distributed GPU computing with CUDA-aware MPI support for large-scale simulations.</p>
  </div>
  <div class="feature-card">
    <h3>Memory Management</h3>
    <p>GPU memory pools, pinned buffers, and automatic data movement.</p>
  </div>
</div>
```

### Quick GPU Example

```julia
using Tarang, CUDA

# Create GPU-accelerated simulation
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; architecture=GPU(), dtype=Float32)

xbasis = Fourier(coords, "x", 512)
ybasis = Fourier(coords, "y", 512)

field = ScalarField(dist, "u", (xbasis, ybasis))
field["g"] .= CUDA.rand(Float32, 512, 512)

# Transforms automatically use CUFFT
forward_transform!(field)   # GPU FFT
backward_transform!(field)  # GPU IFFT
```

See the [GPU Computing Guide](pages/gpu_computing.md) for detailed documentation.

## Performance

Tarang.jl is designed for high performance:

- **Native Julia Speed**: Compiled to native machine code, no interpreter overhead
- **GPU Acceleration**: CUDA-optimized kernels with automatic memory management
- **Scalable MPI**: Efficient pencil decomposition scales to thousands of cores
- **Memory Efficient**: Zero-copy operations and optimized memory layouts

## Citing Tarang.jl

If you use Tarang.jl in your research, please cite:

```bibtex
@software{tarang_jl,
  title = {Tarang.jl: A Spectral PDE Solver for Julia},
  author = {Subhajit Kar},
  year = {2024},
  url = {https://github.com/subhk/Tarang.jl}
}
```

## Acknowledgments

Tarang.jl builds upon excellent Julia packages:
- [PencilArrays.jl](https://github.com/jipolanco/PencilArrays.jl) — Distributed array infrastructure
- [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl) — Parallel FFT transforms
- [MPI.jl](https://github.com/JuliaParallel/MPI.jl) — MPI bindings for Julia
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) — GPU computing support

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/subhk/Tarang.jl/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/subhk/Tarang.jl/discussions)
- **Contributing**: See the [Contributing Guide](pages/contributing.md)

## License

Tarang.jl is licensed under the MIT License.
