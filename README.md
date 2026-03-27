<h1 align="center">Tarang.jl</h1>

<p align="center">
  <strong>A High-Performance Spectral PDE Solver for Julia</strong>
</p>

<p align="center">
  <a href="https://github.com/subhk/Tarang.jl/actions"><img src="https://github.com/subhk/Tarang.jl/workflows/CI/badge.svg" alt="Build Status"></a>
  <a href="https://codecov.io/gh/subhk/Tarang.jl"><img src="https://codecov.io/gh/subhk/Tarang.jl/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://subhk.github.io/Tarang.jl/stable"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Documentation"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  Solve partial differential equations with spectral accuracy on CPUs, GPUs, and distributed clusters — using natural mathematical syntax.
</p>

---

## Features

- **Spectral Methods** -- Fourier, Chebyshev, and Legendre bases with spectral accuracy for smooth solutions
- **GPU Acceleration** -- Native CUDA support via cuFFT with automatic CPU/GPU dispatch
- **MPI Parallelization** -- Pencil decomposition via PencilArrays, scalable to thousands of cores
- **Symbolic Equations** -- Write PDEs in natural math notation (`"dt(T) - kappa*lap(T) = 0"`)
- **Multiple Problem Types** -- Initial value (IVP), eigenvalue (EVP), linear & nonlinear boundary value problems (LBVP, NLBVP)
- **Advanced Physics** -- LES models, stochastic forcing, temporal filters, surface quasi-geostrophic dynamics
- **13 Time Integrators** -- IMEX Runge-Kutta, multistep, exponential time differencing, diagonal IMEX schemes
- **NetCDF I/O** -- Automatic field output with distributed file merging

## Installation

```julia
using Pkg

# Basic installation
Pkg.add(url="https://github.com/subhk/Tarang.jl")

# With GPU support
Pkg.add(["CUDA", "KernelAbstractions"])

# With MPI support
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

Requires Julia 1.10 or later. GPU support requires an NVIDIA GPU with CUDA. MPI requires OpenMPI or MPICH.

## Quick Start

### 1D Diffusion

```julia
using Tarang

domain = PeriodicDomain(64)                     # 64-point periodic domain [0, 2pi]
T = ScalarField(domain, "T")                    # Temperature field

problem = IVP([T])
add_substitution!(problem, "kappa", 0.01)
add_equation!(problem, "dt(T) - kappa*lap(T) = 0")

solver = InitialValueSolver(problem, RK222(); dt=0.01)
run!(solver; stop_time=1.0)
```

### 2D Rayleigh-Benard Convection

```julia
using Tarang

# Channel domain: periodic in x (Fourier), bounded in z (Chebyshev)
domain = ChannelDomain(256, 64; Lx=4.0, Lz=1.0, dealias=3/2)

p = ScalarField(domain, "p")               # Pressure
b = ScalarField(domain, "b")               # Buoyancy
u = VectorField(domain, "u")               # Velocity

problem = IVP(variables)
add_parameters!(problem, kappa=kappa, nu=nu, Lz=Lz)

add_equation!(problem, "div(u) + tau_p = 0")
add_equation!(problem, "dt(b) - kappa*lap(b) + lift(tau_b1, -1) + lift(tau_b2, -2) = -u dot grad(b)")
add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) + lift(tau_u1, -1) + lift(tau_u2, -2) = -u dot grad(u) + b*ez")

fixed_value!(problem, "b", "z", 0.0, Lz)   # Hot bottom
fixed_value!(problem, "b", "z", Lz, 0.0)   # Cold top
no_slip!(problem, "u", "z", 0.0)            # No-slip bottom
no_slip!(problem, "u", "z", Lz)             # No-slip top

solver = InitialValueSolver(problem, RK222(); dt=0.125)
run!(solver; stop_time=50.0, log_interval=100)
```

See [`examples/`](examples/) for complete runnable scripts including QG turbulence, rotating shallow water, and more.

### GPU

```julia
using Tarang, CUDA

# Add arch=GPU() -- everything else stays the same
domain = PeriodicDomain(512, 512; arch=GPU(), dtype=Float32)
field = ScalarField(domain, "u")
forward_transform!(field)   # Uses cuFFT automatically
```

```bash
# Single GPU
julia --project=. examples/gpu_example.jl

# Multi-GPU with MPI (4 GPUs)
mpiexec -n 4 julia --project=. examples/gpu_example.jl
```

### MPI

```bash
# Run any script in parallel with MPI
mpiexec -n 4 julia --project=. examples/ivp/rayleigh_benard_2d.jl
```

## Spectral Bases

| Basis | Domain | Use Case |
|-------|--------|----------|
| `RealFourier` | Periodic | Horizontal directions, real-valued fields |
| `ComplexFourier` | Periodic | Complex-valued fields |
| `ChebyshevT` | Bounded | Wall-bounded domains, boundary conditions |
| `Legendre` | Bounded | Alternative to Chebyshev |

## Time Integrators

| Family | Schemes |
|--------|---------|
| **IMEX Runge-Kutta** | `RK111`, `RK222`, `RK443`, `RKSMR` |
| **Multistep IMEX** | `CNAB1`, `CNAB2`, `SBDF1`--`SBDF4` |
| **Exponential** | `ETD_RK222`, `ETD_CNAB2`, `ETD_SBDF2` |
| **Diagonal IMEX** | `DiagonalIMEX_RK222`, `DiagonalIMEX_RK443`, `DiagonalIMEX_SBDF2` |

## Scientific Applications

- **Fluid Dynamics** -- Navier-Stokes, Rayleigh-Benard convection, channel flows
- **Turbulence** -- LES (Smagorinsky, AMD), stochastic forcing
- **Geophysical Flows** -- Rotating shallow water, stratified turbulence, SQG dynamics
- **Magnetohydrodynamics** -- MHD with magnetic fields, dynamo problems
- **Stability Analysis** -- Eigenvalue problems for linear stability

## Documentation

Full documentation is available at [subhk.github.io/Tarang.jl](https://subhk.github.io/Tarang.jl/stable).

## Citation

If you use Tarang.jl in your research, please cite:

```bibtex
@software{tarang_jl,
  author = {Kar, Subhajit},
  title  = {Tarang.jl: A Spectral PDE Solver for Julia},
  url    = {https://github.com/subhk/Tarang.jl},
  year   = {2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
