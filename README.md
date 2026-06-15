<h1 align="center">Tarang.jl</h1>

<p align="center">
  <strong>A High-Performance Spectral PDE Solver</strong>
</p>

<p align="center">
  <a href="https://github.com/subhk/Tarang.jl/actions/workflows/CI.yml"><img src="https://github.com/subhk/Tarang.jl/actions/workflows/CI.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/subhk/Tarang.jl/actions/workflows/Documentation.yml"><img src="https://github.com/subhk/Tarang.jl/actions/workflows/Documentation.yml/badge.svg" alt="Documentation build"></a>
  <a href="https://codecov.io/gh/subhk/Tarang.jl"><img src="https://codecov.io/gh/subhk/Tarang.jl/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/subhk/Tarang.jl/releases"><img src="https://img.shields.io/github/v/release/subhk/Tarang.jl?label=release" alt="Latest release"></a>
  <a href="https://julialang.org"><img src="https://img.shields.io/badge/julia-1.10%2B-9558B2.svg" alt="Julia 1.10+"></a>
  <a href="https://github.com/JuliaTesting/Aqua.jl"><img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua QA"></a>
  <a href="https://github.com/aviatesk/JET.jl"><img src="https://img.shields.io/badge/JET.jl-tested-blue.svg" alt="JET"></a>
  <a href="https://cuda.juliagpu.org/stable/"><img src="https://img.shields.io/badge/GPU-CUDA-76B900.svg" alt="CUDA GPU support"></a>
  <a href="https://juliaparallel.org/MPI.jl/stable/"><img src="https://img.shields.io/badge/MPI-enabled-2F6DB3.svg" alt="MPI support"></a>
  <a href="https://subhk.github.io/Tarang.jl/stable"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Documentation"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  Solve partial differential equations with spectral accuracy on CPUs, GPUs, and distributed clusters — using natural mathematical syntax.
</p>

---


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

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # CPU test suite
julia --project=. test/run_mpi_ci.jl 4          # MPI tests across 4 ranks
```

CPU and MPI tests run on GitHub Actions; GPU tests (CUDA + NCCL) run on JuliaGPU
Buildkite (`.buildkite/pipeline.yml`), since GitHub-hosted runners have no GPU.
See the [testing guide](docs/src/pages/testing.md) for running GPU/MPI tests
locally and for enabling GPU CI.
