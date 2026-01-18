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

**Code**: See [2D Rayleigh-Bénard Tutorial](../tutorials/ivp_2d_rbc.md) for complete implementation

**Tutorial**: [2D Rayleigh-Bénard](../tutorials/ivp_2d_rbc.md)

```julia
# Quick start
mpiexec -n 4 julia examples/rayleigh_benard_2d.jl
```

**Typical Output**:
```
Starting Rayleigh-Bénard simulation
Ra = 1.0e6, Pr = 1.0
Resolution: 256 × 64
MPI processes: 4
============================================================
t = 0.100, dt = 2.45e-04, Nu = 1.023
t = 0.200, dt = 2.31e-04, Nu = 1.156
t = 1.000, dt = 2.28e-04, Nu = 4.523
```

**Parameters to Try**:
- Ra = 10⁴ to 10⁷ (different convection regimes)
- Pr = 0.7 (air), 1.0 (water), 7.0 (oils)
- Aspect ratio = 1, 2, 4, 8

---

### 3D Taylor-Green Vortex (Advanced)

Canonical test case for 3D turbulence and vortex dynamics.

**Physics**: Vortex stretching, energy cascade, turbulence

**Features**:
- 3D Navier-Stokes
- Periodic boundaries all directions
- Energy spectrum analysis
- Enstrophy tracking

**Code**: See [3D Turbulence Tutorial](../tutorials/ivp_3d_turbulence.md) for complete implementation

```julia
# Run with 8 processes (2×2×2 mesh)
mpiexec -n 8 julia examples/taylor_green_3d.jl
```

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
- Dirichlet boundary conditions
- Exact solution comparison
- Good first example

**Code**: See example below

```julia
using Tarang, MPI

MPI.Init()

coords = CartesianCoordinates("x")
dist = Distributor(coords, mesh=(4,))

x = ChebyshevT(coords["x"], size=64, bounds=(0.0, 1.0))
domain = Domain(dist, (x,))

T = ScalarField(dist, "T", (x,))

problem = IVP([T])
add_equation!(problem, "∂t(T) = kappa*lap(T)")
problem.parameters["kappa"] = 0.01

add_equation!(problem, "T(x=0) = 1")
add_equation!(problem, "T(x=1) = 0")

solver = InitialValueSolver(problem, RK222(), dt=0.001)

while solver.sim_time < 1.0
    step!(solver)
end

MPI.Finalize()
```

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

**Code**: Example coming soon

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
using Tarang, MPI

# 2. Initialize MPI
MPI.Init()

# 3. Parameters
Ra = 1e6
Pr = 1.0
# ...

# 4. Setup domain
coords = CartesianCoordinates(...)
dist = Distributor(...)
bases = (...)
domain = Domain(...)

# 5. Create fields
u = VectorField(...)
p = ScalarField(...)
# ...

# 6. Define problem
problem = IVP([...])
add_equation!(problem, "...")
problem.parameters[...] = ...

# 7. Boundary conditions
add_equation!(problem, "...")

# 8. Initial conditions
# ... initialize fields ...

# 9. Solver
solver = InitialValueSolver(...)

# 10. Analysis setup
cfl = CFL(...)
output_handler = add_netcdf_handler(...)

# 11. Time loop
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
    # ... diagnostics ...
end

# 12. Cleanup
MPI.Finalize()
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

# Run an example
cd examples
mpiexec -n 4 julia rayleigh_benard_2d.jl
```

### Modifying Examples

Copy an example and modify:

```bash
cp examples/rayleigh_benard_2d.jl my_simulation.jl
# Edit my_simulation.jl
mpiexec -n 4 julia my_simulation.jl
```

### Creating New Examples

Start from a template:

```bash
cp examples/template.jl my_new_example.jl
# Follow the structure above
```

---

## Visualization

### During Simulation

```julia
using Plots

if solver.iteration % 100 == 0
    T_grid = get_grid_data(T)
    heatmap(T_grid')
    savefig("output/T_$(solver.iteration).png")
end
```

### Post-Processing

```julia
using Plots, HDF5

# Load saved data
T_data = h5read("output/fields.h5", "T")

# Animate
anim = @animate for t in 1:size(T_data, 3)
    heatmap(T_data[:, :, t]', clims=(0, 1))
end
gif(anim, "animation.gif", fps=15)
```

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
