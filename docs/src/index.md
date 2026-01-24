# Tarang.jl

```@raw html
<div style="background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); color: white; padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;">
    <h1 style="font-size: 2.5rem; margin: 0 0 0.5rem 0; color: white; border: none;">Tarang.jl</h1>
    <p style="font-size: 1.25rem; margin: 0; opacity: 0.95;">A High-Performance Spectral PDE Solver for Julia</p>
    <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">CPU | GPU | MPI Distributed | Symbolic Equations</p>
</div>
```

[![Build Status](https://github.com/subhk/Tarang.jl/workflows/CI/badge.svg)](https://github.com/subhk/Tarang.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/Tarang.jl/stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Features

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; transition: transform 0.2s, box-shadow 0.2s;">
    <h3 style="color: #2563eb; margin-top: 0;">Spectral Methods</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>Fourier, Chebyshev, and Legendre bases</li>
        <li>Spectral accuracy for smooth solutions</li>
        <li>Automatic differentiation operators</li>
    </ul>
</div>

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
    <h3 style="color: #7c3aed; margin-top: 0;">GPU Acceleration</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>Native CUDA support with cuFFT</li>
        <li>KernelAbstractions.jl backend</li>
        <li>Automatic CPU/GPU dispatch</li>
    </ul>
</div>

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
    <h3 style="color: #059669; margin-top: 0;">MPI Parallelization</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>PencilArrays pencil decomposition</li>
        <li>Scalable to thousands of cores</li>
        <li>Efficient distributed FFTs</li>
    </ul>
</div>

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
    <h3 style="color: #dc2626; margin-top: 0;">Symbolic Equations</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>Natural mathematical syntax for PDEs</li>
        <li>Automatic operator construction</li>
        <li>Flexible boundary conditions</li>
    </ul>
</div>

</div>
```

---

## Quick Start

### Installation

```julia
using Pkg

# Basic installation
Pkg.add(url="https://github.com/subhk/Tarang.jl")

# With GPU support
Pkg.add(["CUDA", "KernelAbstractions"])

# With MPI support
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

### Hello World: 2D Rayleigh-Benard Convection

```julia
using Tarang, MPI
MPI.Init()

# Set up coordinate system and parallel distribution
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

# Define spectral bases
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

# Create fields
u = VectorField(dist, coords, "u", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))

# Define equations and solve
problem = IVP([u.components[1], u.components[2], p, T])
add_equation!(problem, "dt(ux) - Pr*lap(ux) + dx(p) = -u@grad(ux) + Ra*Pr*T*ez")
add_equation!(problem, "dt(uz) - Pr*lap(uz) + dz(p) = -u@grad(uz)")
add_equation!(problem, "div(u) = 0")
add_equation!(problem, "dt(T) - lap(T) = -u@grad(T)")

solver = InitialValueSolver(problem, RK222(), dt=1e-3)
```

!!! tip "Pro Tip"
    For large simulations, use GPU acceleration by setting `architecture=GPU()` in the Distributor for significant speedup on NVIDIA hardware.

---

## GPU Example

```julia
using Tarang, CUDA

coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; architecture=GPU(), dtype=Float32)

xbasis = RealFourier(coords["x"], size=512, bounds=(0.0, 2pi))
ybasis = RealFourier(coords["y"], size=512, bounds=(0.0, 2pi))

field = ScalarField(dist, "u", (xbasis, ybasis))
forward_transform!(field)   # Uses cuFFT automatically
```

---

## Scientific Applications

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<div style="background: #eff6ff; border-radius: 8px; padding: 1rem; border-left: 4px solid #2563eb;">
    <strong style="color: #1e40af;">Fluid Dynamics</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Navier-Stokes, Rayleigh-Benard convection, channel flow, jets
    </p>
</div>

<div style="background: #fef3c7; border-radius: 8px; padding: 1rem; border-left: 4px solid #f59e0b;">
    <strong style="color: #92400e;">Turbulence</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        LES models (Smagorinsky, AMD), stochastic forcing, GQL approximation
    </p>
</div>

<div style="background: #f3e8ff; border-radius: 8px; padding: 1rem; border-left: 4px solid #7c3aed;">
    <strong style="color: #5b21b6;">Magnetohydrodynamics</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        MHD with magnetic fields, dynamo problems, magnetic dissipation
    </p>
</div>

<div style="background: #ecfdf5; border-radius: 8px; padding: 1rem; border-left: 4px solid #10b981;">
    <strong style="color: #065f46;">Geophysical Flows</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Rotating shallow water, stratified turbulence, surface dynamics
    </p>
</div>

</div>
```

---

## Problem Types

| Type | Description | Example |
|------|-------------|---------|
| **IVP** | Initial Value Problems | Time-dependent Navier-Stokes |
| **BVP** | Boundary Value Problems | Steady-state solutions |
| **EVP** | Eigenvalue Problems | Linear stability analysis |
| **LBVP** | Linear BVPs | Poisson equation |
| **NLBVP** | Nonlinear BVPs | Steady nonlinear systems |

## Spectral Bases

| Basis | Domain | Usage |
|-------|--------|-------|
| `RealFourier` | Periodic | Horizontal directions, real-valued fields |
| `ComplexFourier` | Periodic | Complex-valued fields |
| `ChebyshevT` | Bounded | Wall-bounded domains, boundary conditions |
| `Legendre` | Bounded | Alternative to Chebyshev |

---

## Documentation

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<a href="getting_started/installation/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>Installation</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Setup and configuration</p>
    </div>
</a>

<a href="tutorials/overview/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>Tutorials</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Step-by-step guides</p>
    </div>
</a>

<a href="pages/gpu_computing/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>GPU Guide</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">CUDA acceleration</p>
    </div>
</a>

<a href="api/coordinates/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>API Reference</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Complete documentation</p>
    </div>
</a>

</div>
```

```@contents
Pages = [
    "getting_started/installation.md",
    "tutorials/overview.md",
    "pages/gpu_computing.md",
    "pages/parallelism.md",
    "api/coordinates.md",
    "examples/gallery.md"
]
Depth = 1
```

---

## Installation Options

| Setup | Command | Use Case |
|-------|---------|----------|
| **Basic** | `Pkg.add(url="...")` | Single CPU, getting started |
| **GPU** | `+ CUDA, KernelAbstractions` | NVIDIA GPU acceleration |
| **MPI** | `+ MPI, PencilArrays, PencilFFTs` | Cluster computing |
| **Full** | All of the above | Maximum flexibility |

!!! note "Requirements"
    - Julia 1.10 or later
    - For GPU: NVIDIA GPU with CUDA support
    - For MPI: OpenMPI or MPICH installed

---

## Contributing

We welcome contributions! See our [GitHub repository](https://github.com/subhk/Tarang.jl) for:
- Bug reports and feature requests
- Documentation improvements
- Pull requests

---

## Citation

If you use Tarang.jl in your research, please cite:

```bibtex
@software{tarang_jl,
  author = {Kar, Subhajit},
  title  = {Tarang.jl: A Spectral PDE Solver for Julia},
  url    = {https://github.com/subhk/Tarang.jl},
  year   = {2024}
}
```

---

## License

Tarang.jl is released under the **MIT License**.

```@raw html
<div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: #f8fafc; border-radius: 8px;">
    <p style="margin: 0; color: #64748b;">
        Made for the scientific computing community
    </p>
</div>
```
