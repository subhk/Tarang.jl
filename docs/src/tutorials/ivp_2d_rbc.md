# Tutorial: 2D Rayleigh-Bénard Convection

This tutorial demonstrates solving a classic fluid dynamics problem: Rayleigh-Bénard convection. We'll set up a complete simulation including equations, boundary conditions, adaptive time stepping, and output.

## Physical Problem

Rayleigh-Bénard convection occurs when a fluid layer is heated from below. The setup:

- Horizontal layer of fluid between two parallel plates
- Bottom plate at high buoyancy (hot)
- Top plate at zero buoyancy (cold)
- Gravity acts downward

When the buoyancy difference is large enough (high Rayleigh number), the fluid becomes unstable and convection cells form.

### Governing Equations

The Boussinesq equations for buoyancy-driven convection, non-dimensionalized using box height and freefall time:

```math
\begin{aligned}
\partial_t \mathbf{u} - \nu \nabla^2 \mathbf{u} + \nabla p - b \hat{\mathbf{z}} &= -\mathbf{u} \cdot \nabla \mathbf{u} \\
\nabla \cdot \mathbf{u} &= 0 \\
\partial_t b - \kappa \nabla^2 b &= -\mathbf{u} \cdot \nabla b
\end{aligned}
```

where $\kappa = (\text{Ra} \cdot \text{Pr})^{-1/2}$ and $\nu = (\text{Ra} / \text{Pr})^{-1/2}$.

**Dimensionless parameters**:
- **Ra** (Rayleigh number): ratio of buoyancy to viscous forces
- **Pr** (Prandtl number): $\text{Pr} = \nu / \kappa$, ratio of momentum to thermal diffusivity

**Variables**:
- $\mathbf{u} = (u_x, u_z)$: velocity field
- $p$: pressure
- $b$: buoyancy (replaces temperature in freefall scaling)

## Domain and Discretization

### Coordinate System

2D Cartesian domain:
- **x** (horizontal): periodic, length $L_x = 4$
- **z** (vertical): bounded by plates, height $L_z = 1$

### Spectral Bases

```julia
# Periodic horizontal direction → RealFourier
xbasis = RealFourier(coords["x"]; size=256, bounds=(0.0, 4.0), dealias=3/2)

# Bounded vertical direction → ChebyshevT
zbasis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0), dealias=3/2)
```

**Resolution guidelines**:
- Horizontal: 128–512 modes (depends on aspect ratio and Ra)
- Vertical: 32–128 modes (more for higher Ra)
- Aspect ratio $L_x / L_z$: typically 2–4 for 2D

## Complete Implementation

### Setup

```julia
using Tarang
using Printf

# Problem parameters
Lx, Lz   = 4.0, 1.0
Nx, Nz   = 256, 64
Rayleigh = 2e6
Prandtl  = 1.0
dealias  = 3/2
stop_time = 500.0
max_dt   = 1e-5

kappa = (Rayleigh * Prandtl)^(-1/2)
nu    = (Rayleigh / Prandtl)^(-1/2)

@root_only begin
    println("2D Rayleigh-Benard Convection")
    @printf("  Ra=%.2e, Pr=%.1f, κ=%.6e, ν=%.6e\n", Rayleigh, Prandtl, kappa, nu)
    @printf("  Domain: %.1f × %.1f, Resolution: %d × %d\n\n", Lx, Lz, Nx, Nz)
end

# Domain setup
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=dealias)

domain = Domain(dist, (xbasis, zbasis))
```

### Fields

```julia
p = ScalarField(domain, "p")   # Pressure
b = ScalarField(domain, "b")   # Buoyancy
u = VectorField(domain, "u")   # Velocity (2-component)
```

### Tau Fields (for Boundary Conditions)

Tarang.jl uses tau fields for imposing boundary conditions. Each boundary condition requires one tau field. The `lift()` operator injects these into the equations on the **derivative basis**, which is the key to the first-order formulation.

```julia
tau_p  = ScalarField(dist, "tau_p",  (), Float64)                   # Pressure gauge
tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), Float64)           # Buoyancy at z=0
tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), Float64)           # Buoyancy at z=Lz
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)   # Velocity at z=0
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)   # Velocity at z=Lz
```

!!! tip "Vector Tau Fields"
    Using `VectorField` for tau fields allows compact vector notation in equations. The individual components are accessible via `tau_u1.components[1]` (x) and `tau_u1.components[2]` (z).

### First-Order Substitutions

The **first-order formulation** reduces second-order operators to products of first-order ones. This requires computing the gradient of each field and including the tau lift on the **derivative basis**:

```julia
# Unit vectors (used for buoyancy forcing and lift direction)
ex, ez = unit_vector_fields(coords, dist)

# Derivative basis: one order lower than zbasis
lift_basis = derivative_basis(zbasis, 1)

# Lift closure: injects tau into the last Chebyshev mode
τ_lift(A) = lift(A, lift_basis, -1)

# First-order gradient substitutions (tau lifts enforce BCs on the gradient)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)
```

These substitutions replace `Δ(b)` with `div(grad_b)` and `div(u)` with `trace(grad_u)` in the equations.

### Problem Definition

```julia
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])

add_parameters!(problem,
    kappa=kappa, nu=nu, Lz=Lz, ez=ez,
    grad_u=grad_u, grad_b=grad_b,
    τ_lift=τ_lift)

# Continuity: trace(grad_u) + tau_p = 0  (first-order div(u) = 0)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Buoyancy: ∂t(b) - κ div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")

# Momentum: ∂t(u) - ν div(grad_u) + ∇(p) - b*ez + τ_lift(tau_u2) = -u⋅∇(u)
add_equation!(problem, "∂t(u) - nu*div(grad_u) + grad(p) - b*ez + τ_lift(tau_u2) = -u⋅∇(u)")
```

**Key points**:
- `trace(grad_u)` replaces `div(u)` in the continuity equation
- `div(grad_b)` replaces `Δ(b)`, and `div(grad_u)` replaces `Δ(u)`
- `τ_lift(tau_u2)` and `τ_lift(tau_b2)` enforce the top boundary conditions
- `τ_lift(tau_u1)` and `τ_lift(tau_b1)` enter via the gradient substitutions (bottom BCs)
- `b*ez` is the buoyancy force in the vertical direction

### Boundary Conditions

No-slip walls (velocity = 0) and fixed buoyancy at each plate:

```julia
add_bc!(problem, "b(z=0) = Lz")    # Hot bottom (b = Lz in freefall scaling)
add_bc!(problem, "u(z=0) = 0")     # No-slip bottom
add_bc!(problem, "b(z=Lz) = 0")    # Cold top
add_bc!(problem, "u(z=Lz) = 0")    # No-slip top
add_bc!(problem, "integ(p) = 0")   # Pressure gauge (removes degeneracy)
```

!!! note "Equation String Syntax"
    Tarang.jl uses string equations with Unicode operators:
    - `∂t` — time derivative
    - `div`, `grad`, `trace` — first-order vector calculus operators
    - `u⋅∇(b)` — scalar advection; `u⋅∇(u)` — vector advection
    - `ez` — vertical unit vector (passed as parameter via `add_parameters!`)
    - `u(z=0) = 0` — sets all velocity components at a boundary

### Initial Conditions

Add random perturbations to the conductive background profile:

```julia
x, z = local_grids(dist, xbasis, zbasis)

# Small random noise, damped at walls
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(b) .*= z' .* (Lz .- z')    # Damp noise at walls

# Add linear conductive background: b = Lz - z
get_grid_data(b) .+= Lz .- z'

# Pre-compute spectral coefficients for the timestepper
ensure_layout!(b, :c)
```

### Time Stepper and Solver

```julia
solver = InitialValueSolver(problem, RK222(); dt=max_dt)
```

**Timestepper selection**:
- Ra < 10⁵: `RK222` or `RK443` (IMEX Runge-Kutta)
- Ra > 10⁵: `CNAB2` or `SBDF2` (IMEX multistep, more stable for stiff problems)
- Very high Ra: `SBDF3` or `SBDF4`

### Adaptive Time Stepping (CFL)

```julia
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)
```

**CFL parameters**:
- `safety`: Safety factor (0.2–0.5, lower is more conservative)
- `max_change`: Maximum timestep growth factor per interval
- `min_change`: Maximum timestep reduction factor per interval
- `cadence`: Recompute CFL every N iterations

### Output

```julia
snapshots = add_file_handler("snapshots", dist,
    Dict("b" => b, "ux" => u.components[1], "uz" => u.components[2]);
    sim_dt=0.25, max_writes=50)
add_task!(snapshots, b;               name="b" )
add_task!(snapshots, u.components[1]; name="ux")
add_task!(snapshots, u.components[2]; name="uz")
```

### Time-Stepping Loop

```julia
@root_only println("Starting main loop")

run!(solver;
     stop_time=stop_time,
     log_interval=100,
     callbacks=[
         on_interval(10) do s
             ensure_layout!(b, :g)
             max_b = global_max(dist, abs.(get_grid_data(b)))
             @root_only @printf("  iter=%d, t=%.4e, dt=%.4e, max|b|=%.4f\n",
                                 s.iteration, s.sim_time, s.dt, max_b)
         end
     ])

@root_only println("Done!")
```

## Running the Simulation

```bash
# Single process
julia --project=. examples/ivp/rayleigh_benard_2d.jl

# Multiple MPI processes
mpiexec -n 4 julia --project=. examples/ivp/rayleigh_benard_2d.jl
```

**Expected output**:
```
2D Rayleigh-Benard Convection
  Ra=2.00e+06, Pr=1.0, κ=7.07e-04, ν=7.07e-04
  Domain: 4.0 × 1.0, Resolution: 256 × 64

Starting main loop
  iter=10, t=1.0000e-04, dt=1.0000e-05, max|b|=1.0037
  iter=20, t=2.0000e-04, dt=1.0000e-05, max|b|=1.0074
  ...
```

## Physical Interpretation

### Flow Regimes

Different Rayleigh numbers produce different behaviors:

| Ra | Regime | Characteristics |
|----|--------|-----------------|
| < 1708 | Conduction | No flow, pure diffusion |
| 10³–10⁴ | Steady rolls | Regular convection cells |
| 10⁵–10⁶ | Transitional | Time-dependent, irregular |
| > 10⁶ | Turbulent | Chaotic, small-scale structures |

### Nusselt Number

The Nusselt number measures heat transfer enhancement:

```math
\text{Nu} = \frac{\text{Total heat flux}}{\text{Conductive heat flux}}
```

- Nu = 1: Pure conduction (no convection)
- Nu > 1: Enhanced heat transfer due to convection
- Nu ≈ 1.0 for Ra < 1708; Nu ∝ Ra^(1/3) approximately in the turbulent regime

## Visualization

```julia
using Plots

# Extract grid data
b_grid = get_grid_data(b)

# Plot buoyancy field
heatmap(b_grid', xlabel="x", ylabel="z", title="Buoyancy")
savefig("buoyancy.png")
```

## Parameter Studies

### Exploring Rayleigh Number

```julia
Ra_values = [1e4, 1e5, 1e6, 2e6]

for Ra in Ra_values
    kappa = (Ra * Prandtl)^(-1/2)
    nu    = (Ra / Prandtl)^(-1/2)
    # ... rebuild problem and run
end
```

### Aspect Ratio Effects

```julia
aspect_ratios = [2.0, 4.0, 8.0]

for ar in aspect_ratios
    Lx = ar * Lz
    xbasis = RealFourier(coords["x"]; size=Int(64*ar), bounds=(0.0, Lx), dealias=3/2)
    # ... rebuild problem and run
end
```

## Performance Optimization

### Resolution Requirements

For Ra = 2×10⁶:
- Minimum: 128 × 32
- Recommended: 256 × 64
- High-resolution: 512 × 128

**Scaling**: For Ra × 10, increase resolution by ~1.5×.

### Timestep Control

```julia
# For stability at high Ra, reduce safety factor
cfl = CFL(solver; safety=0.3, ...)

# Smooth timestep evolution
cfl = CFL(solver; max_change=1.2, min_change=0.5, ...)
```

## Troubleshooting

### NaN in Solution

**Causes**:
- Timestep too large
- Ra too high for current resolution
- Insufficient dissipation

**Solutions**:
```julia
# Reduce CFL safety factor
cfl = CFL(solver; safety=0.2, ...)

# Use more stable timestepper
solver = InitialValueSolver(problem, CNAB2(); dt=max_dt)

# Increase resolution
Nz = 128   # instead of 64
```

### Simulation Not Converging

**For steady state**:
- Run longer (convection takes time to develop)
- Check initial conditions have small perturbations
- Verify Ra > 1708 (critical Rayleigh number)

### Memory Issues

**Solutions**:
```julia
# Reduce resolution
Nx, Nz = 128, 32

# Launch with more MPI processes
# mpiexec -n 8 julia --project=. ...
```

## Complete Script

The full example is available at `examples/ivp/rayleigh_benard_2d.jl`. Key elements:

1. First-order formulation via `derivative_basis` and `τ_lift` closures
2. `grad_u` and `grad_b` substitutions passed as parameters
3. `trace(grad_u) + tau_p = 0` for continuity
4. `div(grad_b)` and `div(grad_u)` in place of `Δ`
5. `u⋅∇(b)` and `u⋅∇(u)` for advection
6. `unit_vector_fields` for `ez`
7. `add_bc!` with string format, e.g. `"b(z=0) = Lz"`
8. `fill_random!` plus linear background for initial conditions
9. `run!` with callbacks for diagnostics

## Next Steps

- **[3D Rotating RBC](ivp_3d_rbc.md)**: Add rotation (Ekman, Coriolis)
- **[Forced 2D Turbulence](ivp_forced_2d_turbulence.md)**: Stochastic forcing, inverse cascade
- **[Forced SQG Turbulence](ivp_forced_sqg_turbulence.md)**: Fractional Laplacian, surface dynamics
- **[Eigenvalue Analysis](eigenvalue_problems.md)**: Compute linear stability

## References

1. Chandrasekhar, S. (1961). *Hydrodynamic and Hydromagnetic Stability*
2. Tritton, D. J. (1988). *Physical Fluid Dynamics*
3. Burns, K. J. et al. (2020). *Dedalus: A flexible framework for numerical simulations with spectral methods*
