# Tarang.jl

```@raw html
<div class="hero-section">
  <img src="assets/logo_light.svg" alt="Tarang.jl" width="420" style="max-width: 90%;" />
  <p class="hero-tagline">A high-performance spectral PDE solver for Julia with GPU acceleration and MPI parallelization.</p>
</div>
```

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3>Spectral Methods</h3>
    <p>Fourier, Chebyshev, and Legendre bases with spectral accuracy and automatic differentiation.</p>
  </div>
  <div class="feature-card">
    <h3>GPU Acceleration</h3>
    <p>Native CUDA support with cuFFT, GPU kernels via KernelAbstractions.jl, and automatic dispatch.</p>
  </div>
  <div class="feature-card">
    <h3>MPI Parallelization</h3>
    <p>Scalable pencil decomposition using PencilArrays.jl for distributed HPC simulations.</p>
  </div>
  <div class="feature-card">
    <h3>Symbolic Equations</h3>
    <p>Natural mathematical syntax for PDEs with automatic parsing and operator construction.</p>
  </div>
</div>
```

---

## Getting Started

Install Tarang.jl and run your first simulation in three steps:

```julia
# 1. Install the package
using Pkg
Pkg.add(url="https://github.com/subhk/Tarang.jl")

# 2. Set up a 2D convection problem
using Tarang, MPI
MPI.Init()

coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

u = VectorField(dist, coords, "u", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))

# 3. Define equations and solve
problem = IVP([u.components[1], u.components[2], p, T])
add_equation!(problem, "dt(ux) - Pr*lap(ux) + dx(p) = -u@grad(ux) + Ra*Pr*T*ez")
add_equation!(problem, "dt(uz) - Pr*lap(uz) + dz(p) = -u@grad(uz)")
add_equation!(problem, "div(u) = 0")
add_equation!(problem, "dt(T) - lap(T) = -u@grad(T)")

solver = InitialValueSolver(problem, RK222(), dt=1e-3)
```

See the [Installation Guide](getting_started/installation.md) for detailed setup instructions,
or jump to the [Tutorials](tutorials/overview.md) for worked examples.

---

## Capabilities

### Problem Types

| Type | Description | Example |
|------|-------------|---------|
| **IVP** | Initial Value Problems | Time-dependent Navier-Stokes |
| **BVP** | Boundary Value Problems | Steady-state solutions |
| **EVP** | Eigenvalue Problems | Linear stability analysis |
| **LBVP** | Linear BVPs | Poisson equation |
| **NLBVP** | Nonlinear BVPs | Steady nonlinear systems |

### Spectral Bases

| Basis | Domain | Usage |
|-------|--------|-------|
| `RealFourier` | Periodic | Horizontal directions, real-valued fields |
| `ComplexFourier` | Periodic | Complex-valued fields |
| `ChebyshevT` | Bounded | Wall-bounded domains, boundary conditions |
| `Legendre` | Bounded | Alternative to Chebyshev |

### Physics Applications

- **Fluid Dynamics** -- Navier-Stokes, Rayleigh-Benard convection, channel flow
- **Turbulence** -- LES with Smagorinsky and AMD models, stochastic forcing
- **Heat Transfer** -- Diffusion, advection-diffusion
- **Magnetohydrodynamics** -- MHD with magnetic fields
- **Geophysical Flows** -- Rotating shallow water, stratified turbulence

---

## GPU Computing

Tarang.jl provides transparent GPU acceleration through CUDA.jl. Fields and transforms automatically dispatch to GPU when the architecture is set:

```julia
using Tarang, CUDA

coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; architecture=GPU(), dtype=Float32)

xbasis = Fourier(coords, "x", 512)
ybasis = Fourier(coords, "y", 512)

field = ScalarField(dist, "u", (xbasis, ybasis))
forward_transform!(field)   # Uses cuFFT automatically
```

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3>cuFFT Integration</h3>
    <p>Automatic FFT acceleration using NVIDIA's optimized cuFFT library.</p>
  </div>
  <div class="feature-card">
    <h3>Portable Kernels</h3>
    <p>CPU/GPU kernels via KernelAbstractions.jl for element-wise operations.</p>
  </div>
  <div class="feature-card">
    <h3>Multi-GPU MPI</h3>
    <p>CUDA-aware MPI for distributed GPU computing at scale.</p>
  </div>
  <div class="feature-card">
    <h3>Memory Management</h3>
    <p>GPU memory pools, pinned buffers, and automatic data movement.</p>
  </div>
</div>
```

See the [GPU Computing Guide](pages/gpu_computing.md) for details.

---

## Documentation

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3><a href="getting_started/installation/">Installing Tarang</a></h3>
    <p>System requirements, installation, and configuration.</p>
  </div>
  <div class="feature-card">
    <h3><a href="tutorials/overview/">Tutorials</a></h3>
    <p>Step-by-step guides for convection, turbulence, and stability analysis.</p>
  </div>
  <div class="feature-card">
    <h3><a href="pages/coordinates/">User Guide</a></h3>
    <p>Core concepts: coordinates, bases, fields, operators, and solvers.</p>
  </div>
  <div class="feature-card">
    <h3><a href="api/coordinates/">API Reference</a></h3>
    <p>Complete API documentation for all public types and functions.</p>
  </div>
</div>
```

---

## Performance

- **Native Julia** -- Compiled to machine code with zero interpreter overhead
- **GPU Acceleration** -- CUDA-optimized kernels with automatic memory management
- **Scalable MPI** -- Pencil decomposition scales to thousands of cores
- **Memory Efficient** -- Zero-copy operations and optimized memory layouts

---

## Citing Tarang.jl

If you use Tarang.jl in your research, please cite:

```bibtex
@software{tarang_jl,
  title  = {Tarang.jl: A Spectral PDE Solver for Julia},
  author = {Subhajit Kar},
  year   = {2024},
  url    = {https://github.com/subhk/Tarang.jl}
}
```

## Acknowledgments

Tarang.jl builds upon:

- [PencilArrays.jl](https://github.com/jipolanco/PencilArrays.jl) and [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl) -- Distributed FFT infrastructure
- [MPI.jl](https://github.com/JuliaParallel/MPI.jl) -- MPI bindings for Julia
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) -- GPU computing support
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) -- Fast Fourier Transforms

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/subhk/Tarang.jl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/subhk/Tarang.jl/discussions)
- **Contributing**: [Contributing Guide](pages/contributing.md)

Tarang.jl is licensed under the MIT License.
