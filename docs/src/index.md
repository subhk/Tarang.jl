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
Pkg.add(url="https://github.com/subhk/Tarang.jl")
```

That is the whole installation. MPI, PencilArrays, PencilFFTs and KernelAbstractions are
**hard dependencies** — they are installed with Tarang, and distributed runs work out of the box.

GPU support is the one opt-in: CUDA is a weak dependency loaded through a package extension, so
add it only if you want it.

```julia
Pkg.add("CUDA")     # enables the GPU backend (TarangCUDAExt)
```

### 1D Diffusion

```julia
using Tarang

domain = PeriodicDomain(64)                     # 64-point periodic [0, 2π]
T = ScalarField(domain, "T")                    # Temperature field

problem = IVP([T])
add_parameters!(problem, kappa=0.01)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

set!(T, x -> sin(x))                            # Initial condition
solver = InitialValueSolver(problem, RK222(); dt=0.01)
run!(solver; stop_time=1.0)                     # That's it!
```

The single mode `sin(x)` decays as `exp(-κt)`; after `t = 1` the solver gives
`max|T| = 0.99004983` against the exact `0.99004983` — a relative error of 4e-12.

### 2D Rayleigh-Benard Convection

A bounded (Chebyshev) direction needs the **tau method**: one tau variable per boundary
condition, lifted into the equations. See [The Tau Method for Boundary Conditions](@ref) for the why.

```julia
using Tarang

Lx, Lz = 4.0, 1.0
Nx, Nz = 64, 32
Rayleigh, Prandtl = 2e4, 1.0

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")                   # Pressure
T = ScalarField(domain, "T")                   # Temperature
u = VectorField(domain, "u")                   # Velocity

# One tau per boundary condition, carrying the Fourier bases (one per x-mode).
tau_p  = ScalarField(dist, "tau_p",  (), Float64)          # pressure gauge
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# First-order reduction: grad_X = ∇X + ẑ·lift(τ)
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem, nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")                                  # continuity
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions go through add_bc!, never add_equation!.
add_bc!(problem, "T(z=0) = 1")                 # hot bottom
add_bc!(problem, "T(z=$Lz) = 0")               # cold top
add_bc!(problem, "u(z=0) = 0")                 # no-slip
add_bc!(problem, "u(z=$Lz) = 0")
add_bc!(problem, "integ(p) = 0")               # pressure gauge

# Conduction profile 1 - z, plus wall-damped noise to seed convection.
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (1.0 .- z')
get_grid_data(T) .+= 1.0 .- z'
ensure_layout!(T, :c)

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
diagnose(solver)                               # Print solver summary
run!(solver; stop_time=0.1, log_interval=100)
```

Time is measured in thermal diffusion times, so `buoy = Ra·Pr` and `nu = Pr`. Over those first
1000 steps the seeded noise grows into rolls (`max|u_z|` reaches 3.76) while the walls stay pinned
to machine precision: `max|T(z=0) − 1| = 3.6e-15`, `max|T(z=Lz)| = 8.1e-15`.

!!! warning "A fixed `dt` will not survive a stiff Rayleigh number"
    This is a Quick Start, not a production run. Push `Rayleigh` to 2e6 with the same fixed
    `dt=1e-4` and the velocities outrun the CFL limit — the run goes to `NaN` within ~100 steps,
    at 64×32 *and* at 128×64. Real runs let a `CFL` controller choose `dt`
    (`run!(solver; cfl=cfl, ...)`); see `examples/ivp/rayleigh_benard_2d.jl` for the full
    256×64, Ra = 2e6 version.

!!! warning "Boundary-condition strings are parsed by Tarang, not by Julia"
    `add_bc!(problem, "T(z=Lz) = 0")` does **not** work: the parser resolves names against the
    problem's variables and `add_parameters!` entries, never your script's globals. A bare global
    warns (`Unknown variable: Lz`) and then fails the matrix build outright. Use a literal
    (`"T(z=1) = 0"`) or interpolate the value in (`"T(z=$Lz) = 0"`) — both enforce the boundary
    exactly.

!!! note "This example is serial"
    Two things here are serial-only. Under MPI the Chebyshev axis must come **first**
    (`Domain(dist, (zbasis, xbasis))`) because a decomposed axis cannot hold a Chebyshev
    transform — and per-mode tau fields (`(xbasis,)`) need the Fourier axis first, so distributed
    runs use bare `()` taus. `-u⋅∇(T)` also puts a Chebyshev derivative on the explicit side, which
    the distributed solver rejects. Pure-Fourier problems parallelize with no changes at all; see
    [Running with MPI](@ref running-with-mpi).

!!! tip "Pro Tip"
    Use `diagnose(solver)` at any time to inspect field layout, compiled RHS status, dealiasing mode, and memory usage.

---

## GPU Example

```julia
using Tarang, CUDA

# Just add device=GPU() — everything else stays the same
domain = PeriodicDomain(512, 512; device=GPU(), dtype=Float32)
field = ScalarField(domain, "u")
forward_transform!(field)   # Uses cuFFT automatically
```

`device` is the only line that changes: the same script with `device=CPU()` runs on the CPU, and
`dtype=Float32` gives `ComplexF32` coefficients either way. `GPU()` needs CUDA.jl loaded — without
it the constructor throws rather than silently falling back.

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
| **LBVP** | Linear Boundary Value Problems | Poisson equation |
| **NLBVP** | Nonlinear Boundary Value Problems | Steady nonlinear systems |
| **EVP** | Eigenvalue Problems | Linear stability analysis |

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
| **Default** | `Pkg.add(url="...")` | Single CPU **and** MPI — nothing else to install |
| **GPU** | `Pkg.add("CUDA")` | NVIDIA GPU acceleration (loads `TarangCUDAExt`) |
| **Cluster MPI** | `MPIPreferences.use_system_binary()` | Bind MPI.jl to the cluster's own MPI |

MPI, MPIPreferences, PencilArrays, PencilFFTs and KernelAbstractions are `[deps]` of Tarang, so a
plain install is already MPI-capable: `mpiexec -n 4 julia --project=. run.jl` works with the MPI
binary that MPI.jl ships. Only CUDA is a `[weakdeps]` package extension.

!!! note "Requirements"
    - Julia 1.10 or later
    - For GPU: NVIDIA GPU with CUDA support
    - For MPI on a cluster: call `MPIPreferences.use_system_binary()` once so MPI.jl uses the
      site's MPI (and its launcher) instead of the bundled one — see
      [Running with MPI](@ref running-with-mpi).

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
