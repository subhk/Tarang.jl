# Example Gallery

This gallery showcases various problems that can be solved with Tarang.jl. Each example includes complete, runnable code and visualization.

## Quick Navigation

### By Physics
- [Fluid Dynamics](#fluid-dynamics)
- [Heat Transfer](#heat-transfer)
- [Stability Analysis](#stability-analysis)
- [Atmospheric Flows](#atmospheric-flows)

### By Complexity
- [Beginner](#beginner-examples) (⭐)
- [Intermediate](#intermediate-examples) (⭐⭐)
- [Advanced](#advanced-examples) (⭐⭐⭐)

### By Features
- [2D Problems](#2d-examples)
- [3D Problems](#3d-examples)
- [Time-Dependent](#time-dependent)
- [Eigenvalue Problems](#eigenvalue-problems)

---

## Fluid Dynamics

### 2D Rayleigh-Bénard Convection (Intermediate)

Thermal convection in a horizontal layer heated from below.

**Physics**: Buoyancy-driven flow, convection cells, heat transfer

**Features**:
- Navier-Stokes equations with buoyancy
- No-slip boundary conditions
- Adaptive time stepping
- Nusselt number analysis

**Code**: `examples/ivp/rayleigh_benard_2d.jl`

**Tutorial**: [2D Rayleigh-Bénard](../tutorials/ivp_2d_rbc.md)

Uses the first-order formulation: divergence and Laplacian operators are
expressed via `trace(grad_f)` and `div(grad_f)`, with lift closures on the
derivative basis to handle tau terms at both walls.

```julia
using Tarang
using Printf

# Parameters
Lx, Lz   = 4.0, 1.0
Nx, Nz   = 256, 64
Rayleigh = 2e6
Prandtl  = 1.0

kappa = (Rayleigh * Prandtl)^(-1/2)
nu    = (Rayleigh / Prandtl)^(-1/2)

# Bases
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

# Fields and tau variables
p = ScalarField(domain, "p");  b = ScalarField(domain, "b")
u = VectorField(domain, "u")
tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), Float64)
tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# First-order reduction with lift closures
ex, ez    = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)

# Problem
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])
add_parameters!(problem, kappa=kappa, nu=nu, Lz=Lz, ez=ez,
                grad_u=grad_u, grad_b=grad_b, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + grad(p) - b*ez + τ_lift(tau_u2) = -u⋅∇(u)")

add_bc!(problem, "b(z=0) = Lz");  add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "b(z=Lz) = 0"); add_bc!(problem, "u(z=Lz) = 0")
add_bc!(problem, "integ(p) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-5)
```

**Parameters to Try**:
- Ra = 10⁴ to 10⁷ (different convection regimes)
- Pr = 0.7 (air), 1.0 (water), 7.0 (oils)
- Aspect ratio = 1, 2, 4, 8

---

### Forced 2D Turbulence (Advanced)

Stochastically forced 2D Navier-Stokes on a doubly-periodic domain, driving
both an inverse energy cascade (energy → large scales) and a forward enstrophy
cascade (enstrophy → small scales).

**Physics**: Inverse energy cascade, k⁻⁵/³ and k⁻³ spectra, 2D turbulence

**Features**:
- Vorticity-streamfunction formulation
- White-in-time ring forcing at wavenumber k_f with prescribed energy injection rate ε
- Hyperviscosity (8th-order) for small-scale dissipation
- Stochastic forcing with Hermitian symmetry for real-valued fields

**Code**: `examples/ivp/forced_2d_turbulence.jl`

The governing equation is:

```
∂t(q) + u·∇(q) = -ν Δ⁴(q) + F
```

where q = Δ(ψ), u = skew(∇ψ), and F is a ring forcing injecting energy at
|k| ∈ [k_f − dk_f, k_f + dk_f].

```julia
using Tarang
using Printf

# Parameters
Nx, Ny  = 512, 512
Lx, Ly  = 2π, 2π
nu      = 1e-20          # Hyperviscosity coefficient (8th-order)
k_f     = 8.0            # Central forcing wavenumber
dk_f    = 2.0            # Forcing bandwidth
ε       = 0.1            # Energy injection rate

domain = PeriodicDomain(Nx, Ny; L=(Lx, Ly))
dist   = domain.dist

q     = ScalarField(domain, "q")          # Vorticity
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

# Ring forcing in wavenumber space
forcing = StochasticForcing(
    field_size            = (Nx, Ny),
    domain_size           = (Lx, Ly),
    energy_injection_rate = ε,
    k_forcing             = k_f,
    dk_forcing            = dk_f,
    dt                    = 5e-3,
    spectrum_type         = :ring,
    enforce_hermitian     = true,
)

problem = IVP([q, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu)

add_equation!(problem, "∂t(q) + nu*Δ⁴(q)  = -u⋅∇(q)")  # PV evolution
add_equation!(problem, "Δ(ψ) + tau_ψ - q  = 0")          # Poisson equation
add_equation!(problem, "u - skew(grad(ψ)) = 0")           # Velocity from ψ
add_bc!(problem, "integ(ψ) = 0")

add_stochastic_forcing!(problem, :q, forcing)

solver = InitialValueSolver(problem, RK222(); dt=5e-3)
```

**To run**:
```bash
julia --project=. examples/ivp/forced_2d_turbulence.jl

# The domain is all-Fourier, so this one does decompose across ranks
mpiexec -n 4 julia --project=. examples/ivp/forced_2d_turbulence.jl
```

---

### Forced SQG Turbulence (Advanced)

Surface quasi-geostrophic (SQG) turbulence: the streamfunction-vorticity
relation uses the fractional Laplacian (−Δ)^{−1/2} instead of Δ^{−1},
producing sharp fronts and filamentary structures in the surface buoyancy field.

**Physics**: Forward energy cascade, k⁻⁵/³ spectrum, surface buoyancy fronts, filaments

**Features**:
- Fractional Laplacian SQG inversion: `fraclap(ψ, 0.5)` for (−Δ)^{1/2}
- Hyperdiffusion with tunable exponent `alpha`
- Stochastic ring forcing on the buoyancy field
- Doubly-periodic domain

**Code**: `examples/ivp/forced_sqg_turbulence.jl`

The governing equation is:

```
∂t(θ) + u·∇(θ) = -ν (−Δ)^α θ + F
```

where ψ = (−Δ)^{−1/2} θ (SQG inversion) and u = skew(∇ψ).

```julia
using Tarang
using Printf

# Parameters
Nx, Ny  = 512, 512
Lx, Ly  = 2π, 2π
nu      = 1e-16          # Hyperdiffusion coefficient
alpha   = 4.0            # Dissipation exponent: (-Δ)^α
k_f     = 10.0           # Central forcing wavenumber
dk_f    = 2.0
ε       = 0.1            # Energy injection rate

domain = PeriodicDomain(Nx, Ny; L=(Lx, Ly))
dist   = domain.dist

θ     = ScalarField(domain, "θ")          # Surface buoyancy
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

forcing = StochasticForcing(
    field_size            = (Nx, Ny),
    domain_size           = (Lx, Ly),
    energy_injection_rate = ε,
    k_forcing             = k_f,
    dk_forcing            = dk_f,
    dt                    = 2e-3,
    spectrum_type         = :ring,
    enforce_hermitian     = true,
)

problem = IVP([θ, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu, alpha=alpha)

# SQG inversion: fraclap(ψ, 0.5) implements (-Δ)^{1/2}
add_equation!(problem, "∂t(θ) + nu*fraclap(θ, alpha) = -u⋅∇(θ)")
add_equation!(problem, "fraclap(ψ, 0.5) + tau_ψ - θ = 0")
add_equation!(problem, "u - skew(grad(ψ)) = 0")
add_bc!(problem, "integ(ψ) = 0")

add_stochastic_forcing!(problem, :θ, forcing)

solver = InitialValueSolver(problem, RK222(); dt=2e-3)
```

**To run**:
```bash
julia --project=. examples/ivp/forced_sqg_turbulence.jl
```

---

### 3D Rotating Rayleigh-Bénard Convection (Advanced)

Standard rotating Rayleigh-Bénard convection (rRBC) non-dimensionalized using
box height H and thermal diffusion time τ_κ = H²/κ, with the E/Pr scaling on
the momentum equation so that the Coriolis term ez×u is O(1).

**Physics**: Geophysical convection, Taylor columns, Coriolis effects, rotating turbulence

**Features**:
- Full 3D (x periodic, y periodic, z Chebyshev)
- E/Pr prefactor on ∂t(u); Coriolis term `curl(u)` = ez×u
- First-order formulation with derivative-basis lift closures (tau_u1/tau_u2, tau_θ1/tau_θ2)
- RK222 IMEX timestepper
- Ekman number, Prandtl number, Rayleigh number as control parameters

**Code**: `examples/ivp/rotating_rayleigh_benard_3d.jl`

The governing equations are:

```
∂t(θ) - Δ(θ)                         = -u⋅∇(θ)
E/Pr * ∂t(u) - E*Δ(u) + ∇(p) + ez×u = Ra*θ*ez - u⋅∇(u)
∇⋅u = 0
```

with no-slip, fixed-temperature walls: θ(z=0)=1, θ(z=1)=0, u(z=0,1)=0.

```julia
using Tarang
using Printf

# Parameters
Lx, Ly, Lz = 2.0, 2.0, 1.0
Nx, Ny, Nz  = 64, 64, 32
Ra  = 1e6;  Pr = 1.0;  Ek = 1e-3
EPr = Ek / Pr          # E/Pr prefactor on ∂t(u)

coords = CartesianCoordinates("x", "y", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=3/2)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, ybasis, zbasis))

p = ScalarField(domain, "p");  θ = ScalarField(domain, "θ")
u = VectorField(domain, "u")
tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_θ1 = ScalarField(dist, "tau_θ1", (xbasis, ybasis), Float64)
tau_θ2 = ScalarField(dist, "tau_θ2", (xbasis, ybasis), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis, ybasis), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis, ybasis), Float64)

ex, ey, ez = unit_vector_fields(coords, dist)
lift_basis  = derivative_basis(zbasis, 1)
τ_lift(A)   = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_θ = grad(θ) + ez * τ_lift(tau_θ1)

problem = IVP([p, θ, u, tau_p, tau_θ1, tau_θ2, tau_u1, tau_u2])
add_parameters!(problem,
    Ra=Ra, Ek=Ek, EPr=EPr, Lz=Lz, ez=ez,
    grad_u=grad_u, grad_θ=grad_θ, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(θ) - div(grad_θ) + τ_lift(tau_θ2) = -u⋅∇(θ)")
add_equation!(problem, "EPr*∂t(u) - Ek*div(grad_u) + ∇(p) + curl(u) + τ_lift(tau_u2) = Ra*θ*ez - u⋅∇(u)")

add_bc!(problem, "θ(z=0) = Lz");  add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "θ(z=Lz) = 0"); add_bc!(problem, "u(z=Lz) = 0")
add_bc!(problem, "integ(p) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-6)
```

**The timestep is set by the buoyancy, not by the CFL condition.** `Ra*θ*ez` sits
on the explicit side, and with `Ra = 10⁶` against the `E/Pr = 10⁻³` prefactor on
`∂t(u)` its effective forcing is `Ra/(E/Pr) = 10⁹`. `dt = 10⁻³` — the value the
shipped script uses — goes to `NaN` within twenty steps, at every resolution and
with every timestepper tried (8³ and 16³; `RK222()` and `ETD_RK222()`). `dt = 10⁻⁶`
integrates stably. Tarang also warns that `θ` appears linearly on the RHS; moving
`- Ra*θ*ez` to the left-hand side gives the same answer but does *not* buy a
larger step. Rescale the buoyancy (or shrink `dt`) before running this at the
parameters above.

The exponential timesteppers are not usable here either: the tau-augmented linear
operator of this first-order formulation is singular, so `ETD_RK222()` cannot
build its matrix exponential and warns
(`ETD-RK222 failed: SingularException(…), falling back to RK222`) on *every*
step. Use `RK222()` directly.

**To run** — serial only. Under MPI the decomposed (trailing) axes must all be
Fourier, so a `(x, y, z_cheb)` domain errors out at construction; and even with
the Chebyshev axis moved to the front, the advective `-u⋅∇(u)` term needs a
Chebyshev derivative on the explicit side, which is not supported distributed
(see [Running with MPI](../getting_started/running_with_mpi.md)).
```bash
julia --project=. examples/ivp/rotating_rayleigh_benard_3d.jl
```

**Parameters to Try**:
- Ek = 10⁻³ to 10⁻⁵ (rotation rate)
- Ra = 10⁵ to 10⁸ (supercriticality)
- Pr = 0.7 (air), 1.0, 7.0 (water)

---

### 3D Taylor-Green Vortex (Advanced)

Canonical test case for 3D turbulence and vortex dynamics.

**Physics**: Vortex stretching, energy cascade, turbulence

**Features**:
- 3D Navier-Stokes
- Periodic boundaries all directions
- Energy spectrum analysis
- Enstrophy tracking

**Code**: See [3D Turbulence Tutorial](../tutorials/ivp_3d_turbulence.md) for a complete
implementation — there is no standalone script for it under `examples/`.

**Visualization**: Vorticity isosurfaces, energy spectra

---

### 3D Channel Flow (Advanced)

Turbulent flow between parallel plates.

**Physics**: Wall-bounded turbulence, Reynolds stress, mean profiles

**Features**:
- 3D Navier-Stokes with walls
- Periodic in x,y; no-slip at z walls
- Mean flow forcing
- Turbulence statistics

**Code**: Example coming soon

**Analysis**: Mean velocity profile, Reynolds stresses, energy spectra

---

### 2D Kelvin-Helmholtz Instability (Intermediate)

Shear flow instability and vortex formation.

**Physics**: Shear instability, vortex roll-up, mixing

**Features**:
- 2D Euler or Navier-Stokes
- Shear layer with perturbation
- Periodic boundaries
- Vorticity evolution

**Code**: Example coming soon

---

## Heat Transfer

### 1D Heat Diffusion (Beginner)

Simple diffusion equation in one dimension.

**Physics**: Heat conduction, diffusion

**Features**:
- Single PDE
- Dirichlet boundary conditions via the tau method
- Exact solution comparison
- Good first example

**Code**: See example below

A bounded (Chebyshev) direction needs one **tau variable per boundary condition**:
the BCs count as equations, so `IVP` must be given as many variables as there are
equations, and the tau variables must be lifted back into the bulk equation. Two
Dirichlet conditions ⇒ two tau variables ⇒ `IVP([T, tau_1, tau_2])`.

```julia
using Tarang, Printf

coords = CartesianCoordinates("x")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = ChebyshevT(coords["x"]; size=64, bounds=(0.0, 1.0))
domain = Domain(dist, (xbasis,))

T = ScalarField(domain, "T")

# One tau variable per boundary condition, lifted onto the derivative basis
tau_1 = ScalarField(dist, "tau_1", (), Float64)
tau_2 = ScalarField(dist, "tau_2", (), Float64)
lift_basis = derivative_basis(xbasis, 1)
l1 = lift(tau_1, lift_basis, -1)
l2 = lift(tau_2, lift_basis, -2)

problem = IVP([T, tau_1, tau_2])
add_parameters!(problem, kappa=0.01, l1=l1, l2=l2)
add_equation!(problem, "∂t(T) - kappa*lap(T) + l1 + l2 = 0")

# Boundary conditions go through add_bc!, never add_equation!
add_bc!(problem, "T(x=0) = 1")
add_bc!(problem, "T(x=1) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Initial condition: the steady profile 1 - x plus one decaying sine mode
x, = local_grids(dist, xbasis)
ensure_layout!(T, :g)
get_grid_data(T) .= 1.0 .- x .+ 0.5 .* sin.(π .* x)
ensure_layout!(T, :c)

# run! drives the loop to stop_time using the solver's fixed dt (no CFL here).
run!(solver; stop_time=0.5)

# Exact solution: T = 1 - x + 0.5 sin(πx) exp(-κπ²t)
ensure_layout!(T, :g)
exact = 1.0 .- x .+ 0.5 .* sin.(π .* x) .* exp(-0.01 * π^2 * 0.5)
@printf("max error = %.2e\n", maximum(abs, Array(get_grid_data(T)) .- exact))
```

Prints `max error = 9.27e-12`, and the boundary values are enforced to machine
precision (`T(x=0) = 1`, `T(x=1) = -5.2e-15`).

---

### 2D Steady-State Heat Conduction (Beginner)

Solve Laplace equation for steady heat distribution.

**Physics**: Steady-state diffusion, thermal equilibrium

**Features**:
- Boundary value problem (LBVP)
- Mixed boundary conditions
- Sparse linear solve

**Code**: Example coming soon

---

### 3D Thermal Convection with Rotation (Advanced)

Rotating Rayleigh-Bénard convection.

**Physics**: Geophysical flows, rotation effects, Taylor-Proudman

**Features**:
- Coriolis force
- 3D buoyancy-driven flow
- Pattern formation

**Code**: See [3D Rotating Rayleigh-Bénard](#3d-rotating-rayleighbénard-convection-advanced) above

---

## Stability Analysis

### Rayleigh-Bénard Linear Stability (Intermediate)

Compute critical Rayleigh number and eigenmodes.

**Physics**: Hydrodynamic stability, critical conditions

**Features**:
- Eigenvalue problem (EVP)
- Growth rates and frequencies
- Critical modes visualization

**Code**: See [Eigenvalue Problems Tutorial](../tutorials/eigenvalue_problems.md) for complete implementation

**Tutorial**: [Eigenvalue Problems](../tutorials/eigenvalue_problems.md)

---

### Plane Poiseuille Stability (Advanced)

Stability of parallel shear flow.

**Physics**: Orr-Sommerfeld equation, Tollmien-Schlichting waves

**Features**:
- Orr-Sommerfeld eigenvalue problem
- Neutral stability curves
- Most unstable modes

**Code**: Example coming soon

---

## Wave Propagation

### 1D Linear Wave Equation (Beginner)

Simple wave propagation.

**Physics**: Hyperbolic PDE, wave dynamics

**Features**:
- Second-order time derivative
- Exact solutions
- Wave reflections

**Code**: Example coming soon

---

### 2D Acoustic Waves (Intermediate)

Sound wave propagation in 2D.

**Physics**: Acoustics, wave scattering

**Features**:
- Linear acoustics
- Point source
- Radiation patterns

**Code**: Example coming soon

---

## Atmospheric Flows

### 2D Gravity Waves (Intermediate)

Internal gravity waves in stratified fluid.

**Physics**: Stratified flows, buoyancy oscillations

**Features**:
- Boussinesq equations
- Stable stratification
- Wave propagation

**Code**: Example coming soon

---

### 2D Quasi-Geostrophic Flow (Advanced)

Large-scale atmospheric/oceanic flows.

**Physics**: Geostrophic balance, Rossby waves

**Features**:
- Potential vorticity equation
- Beta-plane approximation
- Jet formation

**Code**: Example coming soon

---

## Advanced Examples

### MHD (Magnetohydrodynamics) (Advanced)

Fluid dynamics with magnetic fields.

**Physics**: Lorentz force, magnetic induction

**Features**:
- Coupled fluid and magnetic fields
- Alfvén waves
- Magnetic energy

**Code**: Example coming soon

---

### Double-Diffusive Convection (Advanced)

Convection driven by two diffusing species.

**Physics**: Salt-fingering, thermohaline circulation

**Features**:
- Two scalar fields (T and S)
- Different diffusivities
- Layering phenomena

**Code**: Example coming soon

---

### Stratified Shear Flow (Advanced)

Combined shear and stratification.

**Physics**: Richardson number, mixing, turbulence

**Features**:
- Background shear and stratification
- Kelvin-Helmholtz + gravity waves
- Mixing efficiency

**Code**: Example coming soon

---

## Example Structure

Each example follows this structure:

```julia
# 1. Load packages
using Tarang

# 2. Parameters
Ra = 1e6
Pr = 1.0
# ...

# 3. Setup domain
coords = CartesianCoordinates(...)
dist = Distributor(...)
bases = (...)
domain = Domain(...)

# 4. Create fields
u = VectorField(...)
p = ScalarField(...)
# ...

# 5. Define problem
problem = IVP([...])
add_equation!(problem, "...")
add_parameters!(problem, ...)

# 6. Boundary conditions
add_bc!(problem, "...")

# 7. Initial conditions
# ... initialize fields ...

# 8. Solver
solver = InitialValueSolver(...)

# 9. Analysis setup
cfl = CFL(...)
snapshots = add_file_handler(...)

# 10. Time loop
run!(solver;
     stop_time=t_end,
     log_interval=100,
     callbacks=[...])
```

---

## Running Examples

### From Repository

```bash
# Clone repository
git clone https://github.com/subhk/Tarang.jl.git
cd Tarang.jl

# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run an example (serial)
julia --project=. examples/ivp/rayleigh_benard_2d.jl

# Run with MPI — only the all-Fourier examples are distributable; a Chebyshev
# axis must stay local, which rules out the Rayleigh-Bénard / channel examples.
mpiexec -n 4 julia --project=. examples/ivp/forced_2d_turbulence.jl
```

### Modifying Examples

Copy an example and modify:

```bash
cp examples/ivp/rayleigh_benard_2d.jl my_simulation.jl
# Edit my_simulation.jl
julia --project=. my_simulation.jl
```

### Creating New Examples

Start from the structure above and adapt the basis, fields, and equations for
your problem.

---

## Visualization

### During Simulation

Pass a callback to `run!` to plot periodically — an `Int` interval fires every N
iterations:

```julia
using Plots

run!(solver; stop_time=t_end, cfl=cfl,
     callbacks=[
         on_interval(100) do s
             ensure_layout!(T, :g)
             T_grid = Array(get_grid_data(T))
             heatmap(T_grid')
             savefig("output/T_$(s.iteration).png")
         end
     ])
```

### Post-Processing

Snapshot files are NetCDF-4 files whose variables live in **groups** (`vars`,
`time`, `grids`), so a plain `NetCDF.ncread(file, "T")` cannot find them — read
them with `group_ncread`. The leading axis of a task variable is the write index:

```julia
using Plots, Tarang

file   = "snapshots/snapshots_s1/snapshots_s1.nc"
T_data = group_ncread(file, "vars", "T")        # (write, x, z)
t      = group_ncread(file, "time", "sim_time") # (write,)

# Animate
anim = @animate for n in 1:size(T_data, 1)
    heatmap(T_data[n, :, :]', clims=(0, 1), title="t = $(t[n])")
end
gif(anim, "animation.gif", fps=15)
```

`Tarang.group_variable_names(file, "vars")` lists the task names available in a file.

---

## Contributing Examples

We welcome new examples! Guidelines:

1. **Complete and runnable**: Include all necessary code
2. **Well-commented**: Explain physics and numerics
3. **Documented**: Add to this gallery with description
4. **Tested**: Verify runs correctly with typical parameters
5. **Realistic**: Use physically meaningful parameters

See [Contributing Guide](../pages/contributing.md) for details.

---

## Next Steps

- **Learn the basics**: [Getting Started](../getting_started/installation.md)
- **Detailed tutorials**: [Tutorials](../tutorials/overview.md)
- **Understand the API**: [API Reference](../api/fields.md)
- **Ask questions**: [GitHub Discussions](https://github.com/subhk/Tarang.jl/discussions)
