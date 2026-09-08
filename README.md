<h1 align="center">Tarang.jl</h1>

<p align="center">
  <strong>A High-Performance (Pseudo-)Spectral PDE Solver</strong>
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

# With MPI support
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

Install the optional CUDA extension in Julia's versioned default environment.
It remains available through Julia's environment stack without modifying this
project's `Project.toml` or `Manifest.toml`:

```bash
julia --project=@v#.# -e 'using Pkg; Pkg.add("CUDA")'
```

Requires Julia 1.10 or later. GPU support requires an NVIDIA GPU with CUDA. MPI requires OpenMPI or MPICH.

## Quick Start

### 1D Diffusion

```julia
using Tarang

domain = PeriodicDomain(64)                     # 64-point periodic domain [0, 2pi]
T = ScalarField(domain, "T")                    # Temperature field

problem = InitialValueProblem([T])
add_substitution!(problem, "kappa", 0.01)
add_equation!(problem, "dt(T) - kappa*lap(T) = 0")

solver = InitialValueSolver(problem, RK222(); dt=0.01)
run!(solver; stop_time=1.0)
```

### 2D Rayleigh-Benard Convection

```julia
using Tarang

Lx, Lz = 4.0, 1.0
Nx, Nz = 32, 16
Rayleigh, Prandtl = 2e4, 1.0

coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

# Tau fields supply the degrees of freedom used to enforce the wall conditions.
tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

_, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = InitialValueProblem([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem,
    nu=Prandtl, buoy=Rayleigh * Prandtl, ez=ez,
    grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem,
    "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "T(z=$Lz) = 0")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=$Lz) = 0")
add_bc!(problem, "integ(p) = 0")

# Start from the conduction profile with a small wall-compatible perturbation.
_, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (1.0 .- z')
get_grid_data(T) .+= 1.0 .- z'
ensure_layout!(T, :c)

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
run!(solver; stop_time=1e-3, log_interval=10)
```

See [`examples/`](examples/) for complete runnable scripts including QG turbulence, rotating shallow water, and more.

### GPU

```julia
using Tarang, CUDA

# Allocate on the GPU; select a timestepper supported by your operator/backend.
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
# Run the distributed regression suite with four MPI ranks.
julia --project=. test/run_mpi_ci.jl 4
```

MPI scripts must follow the [domain decomposition rules](docs/src/pages/parallelism.md).
For a bounded domain, keep the Chebyshev axis first so it stays local to each rank.

## Spectral Bases

| Basis | Domain | Use Case |
|-------|--------|----------|
| `RealFourier` | Periodic | Horizontal directions, real-valued fields |
| `ComplexFourier` | Periodic | Complex-valued fields |
| `ChebyshevT` | Bounded | Wall-bounded domains, boundary conditions |
| `Legendre` | Bounded | Alternative to Chebyshev |
