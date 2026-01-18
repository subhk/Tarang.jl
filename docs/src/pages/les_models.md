# Large Eddy Simulation (LES) Models

This page provides a comprehensive introduction to Large Eddy Simulation (LES) and the subgrid-scale (SGS) closure models available in Tarang.jl.

---

## What is Large Eddy Simulation?

### The Turbulence Challenge

Turbulent flows contain a vast range of length scales, from the largest energy-containing eddies down to the smallest dissipative scales (Kolmogorov scales). In a Direct Numerical Simulation (DNS), we resolve **all** these scales, which requires:

```math
N \sim Re^{9/4}
```

grid points in 3D, where $Re$ is the Reynolds number. For atmospheric or oceanic flows with $Re \sim 10^9$, this is computationally impossible.

### The LES Approach

Large Eddy Simulation offers a practical alternative:

1. **Resolve** the large, energy-containing eddies directly
2. **Model** the effect of small, unresolved eddies

```
┌─────────────────────────────────────────────────────────────┐
│                    Energy Spectrum E(k)                      │
│                                                              │
│    E(k)                                                      │
│     │                                                        │
│     │   ╭──╮                                                 │
│     │  ╱    ╲          k^(-5/3)                              │
│     │ ╱      ╲___                                            │
│     │╱            ╲___                                       │
│     │                  ╲___                                  │
│     │                       ╲___                             │
│     └──────────────────────────────────────────────► k       │
│         │                    │                               │
│         │◄── Resolved ──────►│◄── Modeled (SGS) ──►│        │
│         │   (large eddies)   │   (small eddies)    │        │
│                              │                               │
│                          filter cutoff (Δ)                   │
└─────────────────────────────────────────────────────────────┘
```

The **filter width** $\Delta$ (typically the grid spacing) separates resolved from unresolved scales.

### Filtering the Navier-Stokes Equations

We apply a spatial filter to decompose velocity into resolved ($\bar{u}$) and subgrid ($u'$) parts:

```math
u = \bar{u} + u'
```

Filtering the incompressible Navier-Stokes equations gives:

```math
\frac{\partial \bar{u}_i}{\partial t} + \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j} = -\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \nu \nabla^2 \bar{u}_i - \frac{\partial \tau_{ij}}{\partial x_j}
```

where the **subgrid-scale (SGS) stress tensor** appears:

```math
\tau_{ij} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j
```

This tensor represents the effect of unresolved turbulent motions on the resolved flow. Since we cannot compute $\tau_{ij}$ directly (it involves unresolved velocities), we must **model** it.

---

## The Closure Problem

### Why We Need Models

The SGS stress $\tau_{ij}$ contains information about scales we don't resolve. This creates the **closure problem**: our filtered equations have more unknowns than equations.

### The Eddy Viscosity Hypothesis

Most SGS models use the **Boussinesq hypothesis**, which assumes the SGS stress is proportional to the resolved strain rate:

```math
\tau_{ij} - \frac{1}{3}\tau_{kk}\delta_{ij} = -2 \nu_e \bar{S}_{ij}
```

where:
- $\nu_e$ is the **eddy viscosity** (to be modeled)
- $\bar{S}_{ij}$ is the resolved strain rate tensor:

```math
\bar{S}_{ij} = \frac{1}{2}\left(\frac{\partial \bar{u}_i}{\partial x_j} + \frac{\partial \bar{u}_j}{\partial x_i}\right)
```

This transforms the filtered momentum equation into:

```math
\frac{\partial \bar{u}_i}{\partial t} + \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j} = -\frac{1}{\rho}\frac{\partial \bar{p}^*}{\partial x_i} + (\nu + \nu_e) \nabla^2 \bar{u}_i
```

where we've absorbed the isotropic part into a modified pressure $\bar{p}^*$.

**Key insight**: The SGS model effectively adds a spatially-varying viscosity $\nu_e(x,t)$ to the molecular viscosity $\nu$.

---

## Energy Cascade and Dissipation

### Forward Energy Cascade

In 3D turbulence, energy flows from large scales to small scales (forward cascade):

```
Large eddies ──► Medium eddies ──► Small eddies ──► Dissipation
  (production)                                        (ε = 2νSᵢⱼSᵢⱼ)
```

The SGS model must drain energy from resolved scales at the correct rate to maintain physical behavior.

### SGS Dissipation

The rate at which energy is transferred from resolved to unresolved scales is:

```math
\varepsilon_{sgs} = -\tau_{ij} \bar{S}_{ij} = 2\nu_e |\bar{S}|^2
```

where $|\bar{S}| = \sqrt{2\bar{S}_{ij}\bar{S}_{ij}}$ is the strain rate magnitude.

A good SGS model ensures $\varepsilon_{sgs}$ matches the actual energy transfer rate.

---

## Available Models in Tarang.jl

Tarang.jl provides two eddy viscosity models:

| Model | Year | Key Feature | Best For |
|-------|------|-------------|----------|
| **Smagorinsky** | 1963 | Simple, robust | Isotropic grids, fully turbulent flows |
| **AMD** | 2015 | Minimum dissipation, anisotropic | Transitional flows, stretched grids |

---

## Smagorinsky Model

### Physical Basis

Joseph Smagorinsky (1963) proposed the first and most widely-used SGS model. It's based on dimensional analysis and mixing length theory.

**Key assumptions:**
1. The subgrid scales are in equilibrium (production = dissipation)
2. The characteristic length scale is proportional to the filter width $\Delta$
3. The characteristic velocity scale is $\Delta |\bar{S}|$

### Mathematical Formulation

The eddy viscosity is computed as:

```math
\nu_e = (C_s \Delta)^2 |\bar{S}|
```

where:
- $C_s \approx 0.17$ is the **Smagorinsky constant**
- $\Delta = (\Delta_x \Delta_y \Delta_z)^{1/3}$ is the effective filter width
- $|\bar{S}| = \sqrt{2 \bar{S}_{ij} \bar{S}_{ij}}$ is the strain rate magnitude

**Expanded form for 3D:**

```math
|\bar{S}| = \sqrt{2\left[\left(\frac{\partial \bar{u}}{\partial x}\right)^2 + \left(\frac{\partial \bar{v}}{\partial y}\right)^2 + \left(\frac{\partial \bar{w}}{\partial z}\right)^2\right] + \left(\frac{\partial \bar{u}}{\partial y} + \frac{\partial \bar{v}}{\partial x}\right)^2 + \left(\frac{\partial \bar{u}}{\partial z} + \frac{\partial \bar{w}}{\partial x}\right)^2 + \left(\frac{\partial \bar{v}}{\partial z} + \frac{\partial \bar{w}}{\partial y}\right)^2}
```

### Usage in Tarang.jl

```julia
using Tarang

# Grid parameters
N = 128                     # Grid points per direction
L = 2π                      # Domain size
Δ = L / N                   # Grid spacing

# Create the Smagorinsky model
sgs_model = SmagorinskyModel(
    C_s = 0.17,                    # Smagorinsky constant
    filter_width = (Δ, Δ, Δ),      # Grid spacing in each direction
    field_size = (N, N, N)         # Number of grid points
)

# Compute eddy viscosity from velocity gradients
# You need all 9 components of the velocity gradient tensor
compute_eddy_viscosity!(sgs_model,
    ∂u∂x, ∂u∂y, ∂u∂z,    # Gradients of u
    ∂v∂x, ∂v∂y, ∂v∂z,    # Gradients of v
    ∂w∂x, ∂w∂y, ∂w∂z     # Gradients of w
)

# Retrieve the computed eddy viscosity field
νₑ = get_eddy_viscosity(sgs_model)
```

### Choosing the Smagorinsky Constant

The constant $C_s$ is not universal and depends on the flow:

| Flow Type | Recommended $C_s$ | Notes |
|-----------|-------------------|-------|
| Isotropic turbulence | 0.17 - 0.20 | Theoretical value from Lilly (1967) |
| Channel flow | 0.10 - 0.12 | Reduced due to wall effects |
| Mixing layers | 0.10 - 0.14 | Transitional regions need lower values |
| Free shear flows | 0.10 - 0.12 | Similar to channel flow |
| Atmospheric boundary layer | 0.10 - 0.15 | Depends on stability |

**Rule of thumb**: Start with $C_s = 0.17$ and reduce if you observe over-dissipation.

### Limitations

1. **Over-dissipation**: The model is always "on", even in laminar regions
2. **No backscatter**: Cannot represent energy transfer from small to large scales
3. **Wall behavior**: Needs damping functions near solid walls
4. **Isotropic assumption**: Assumes same behavior in all directions

---

## Anisotropic Minimum Dissipation (AMD) Model

### Motivation

The AMD model (Rozema et al., 2015) addresses key limitations of Smagorinsky:

1. **Automatic switch-off**: $\nu_e = 0$ in laminar/transitional regions
2. **Anisotropic grids**: Properly handles $\Delta_x \neq \Delta_y \neq \Delta_z$
3. **Minimum dissipation**: Adds only the dissipation needed for stability

### Mathematical Formulation

The AMD model computes eddy viscosity as:

```math
\nu_e = \max\left(0, -C \frac{\hat{\delta}_{ij}^2 \frac{\partial \bar{u}_i}{\partial x_k} \frac{\partial \bar{u}_j}{\partial x_k} \bar{S}_{ij}}{\frac{\partial \bar{u}_m}{\partial x_n} \frac{\partial \bar{u}_m}{\partial x_n}}\right)
```

where $\hat{\delta}_{ij} = \Delta_i \delta_{ij}$ (no sum) incorporates anisotropic filter widths.

**Simplified form:**

```math
\nu_e = \max\left(0, -C \frac{\text{numerator}}{\text{denominator}}\right)
```

- **Numerator**: Measures alignment between velocity gradients and strain rate
- **Denominator**: Total velocity gradient magnitude squared
- **max(0, ...)**: Ensures non-negative eddy viscosity

### Physical Interpretation

The numerator can be negative (giving $\nu_e > 0$) when:
- Velocity gradients are aligned with the strain rate
- Energy is being transferred to smaller scales

The numerator is positive (giving $\nu_e = 0$) when:
- Flow is laminar or transitional
- No SGS dissipation is needed

### Usage in Tarang.jl

```julia
# Create the AMD model
sgs_model = AMDModel(
    C = 1/12,                      # Poincaré constant (for spectral methods)
    filter_width = (Δx, Δy, Δz),   # Can be different in each direction
    field_size = (Nx, Ny, Nz),
    clip_negative = true           # Ensure νₑ ≥ 0 (recommended)
)

# Compute eddy viscosity (same interface as Smagorinsky)
compute_eddy_viscosity!(sgs_model,
    ∂u∂x, ∂u∂y, ∂u∂z,
    ∂v∂x, ∂v∂y, ∂v∂z,
    ∂w∂x, ∂w∂y, ∂w∂z
)

νₑ = get_eddy_viscosity(sgs_model)
```

### Choosing the AMD Constant

The constant $C$ depends on the numerical discretization:

| Discretization | Recommended $C$ | Notes |
|----------------|-----------------|-------|
| Spectral methods | 1/12 ≈ 0.0833 | Theoretical value |
| 4th-order finite difference | 0.212 | From Verstappen (2011) |
| 2nd-order finite difference | 0.3 | Higher due to numerical diffusion |

### Scalar Transport (Buoyancy)

For buoyancy-driven flows, AMD also provides eddy diffusivity for scalar transport:

```math
\kappa_e = \max\left(0, -C \frac{\Delta_k^2 \frac{\partial \bar{u}_i}{\partial x_k} \frac{\partial \bar{\theta}}{\partial x_k} \frac{\partial \bar{\theta}}{\partial x_i}}{\frac{\partial \bar{\theta}}{\partial x_n} \frac{\partial \bar{\theta}}{\partial x_n}}\right)
```

```julia
# Compute eddy diffusivity for buoyancy/temperature
compute_eddy_diffusivity!(sgs_model,
    ∂w∂x, ∂w∂y, ∂w∂z,    # Vertical velocity gradients
    ∂b∂x, ∂b∂y, ∂b∂z     # Buoyancy/scalar gradients
)

κₑ = get_eddy_diffusivity(sgs_model)
```

---

## Choosing Between Models

### Decision Flowchart

```
Start
  │
  ▼
Is your grid anisotropic (Δx ≠ Δy ≠ Δz)?
  │
  ├── Yes ──► Use AMD (handles anisotropy naturally)
  │
  └── No
       │
       ▼
Does your flow have laminar or transitional regions?
       │
       ├── Yes ──► Use AMD (automatically switches off)
       │
       └── No
            │
            ▼
Is computational cost a primary concern?
            │
            ├── Yes ──► Use Smagorinsky (slightly cheaper)
            │
            └── No ──► Either works; AMD is generally more accurate
```

### Summary Comparison

| Aspect | Smagorinsky | AMD |
|--------|-------------|-----|
| **Complexity** | Simple | Moderate |
| **Cost per timestep** | Low | Slightly higher |
| **Laminar regions** | Over-dissipates | Correctly gives νₑ = 0 |
| **Anisotropic grids** | Needs modification | Native support |
| **Near walls** | Needs damping | Better behavior |
| **Tuning required** | Often yes | Usually not |
| **Scalar transport** | Use Pr_t | Built-in κₑ |

---

## Complete Example: LES of Decaying Turbulence

This example demonstrates a complete LES setup for decaying homogeneous isotropic turbulence.

```julia
using Tarang
using Statistics

# ============================================================
# 1. Physical and Numerical Parameters
# ============================================================

N = 128                     # Grid points per direction
L = 2π                      # Domain size [m]
Δ = L / N                   # Grid spacing [m]
ν = 1e-4                    # Molecular (kinematic) viscosity [m²/s]
dt = 0.001                  # Time step [s]
nsteps = 1000               # Number of time steps

# ============================================================
# 2. Create the SGS Model
# ============================================================

# Option A: Smagorinsky (simple, robust)
sgs = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (Δ, Δ, Δ),
    field_size = (N, N, N)
)

# Option B: AMD (recommended for most applications)
# sgs = AMDModel(
#     C = 1/12,
#     filter_width = (Δ, Δ, Δ),
#     field_size = (N, N, N)
# )

# ============================================================
# 3. Setup Computational Domain
# ============================================================

coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; mesh=(1, 1, 1))  # Single processor

# Fourier bases for periodic domain
xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
zbasis = RealFourier(coords["z"]; size=N, bounds=(0.0, L))

domain = Domain(dist, (xbasis, ybasis, zbasis))

# Create velocity field
u = VectorField(dist, coords, "u", (xbasis, ybasis, zbasis))

# ============================================================
# 4. Initialize with Turbulent Velocity Field
# ============================================================

# (Initialize u with your preferred IC - e.g., random phases
#  with prescribed energy spectrum)

# ============================================================
# 5. Helper Function: Compute All Velocity Gradients
# ============================================================

function compute_velocity_gradients(u)
    # Uses spectral differentiation for high accuracy
    ∂u∂x, ∂u∂y, ∂u∂z = gradient(u.components[1])
    ∂v∂x, ∂v∂y, ∂v∂z = gradient(u.components[2])
    ∂w∂x, ∂w∂y, ∂w∂z = gradient(u.components[3])
    return (∂u∂x, ∂u∂y, ∂u∂z,
            ∂v∂x, ∂v∂y, ∂v∂z,
            ∂w∂x, ∂w∂y, ∂w∂z)
end

# ============================================================
# 6. Time Integration Loop
# ============================================================

for step in 1:nsteps
    # --- Step 1: Compute velocity gradients ---
    grads = compute_velocity_gradients(u)

    # --- Step 2: Update SGS eddy viscosity ---
    compute_eddy_viscosity!(sgs, grads...)

    # --- Step 3: Get effective viscosity ---
    νₑ = get_eddy_viscosity(sgs)
    ν_eff = ν .+ νₑ  # Total viscosity = molecular + SGS

    # --- Step 4: Advance momentum equation ---
    # The filtered Navier-Stokes with SGS model:
    #   ∂ū/∂t + (ū·∇)ū = -∇p̄/ρ + (ν + νₑ)∇²ū
    #
    # In your timestepper, use ν_eff instead of ν
    # for the viscous term

    # --- Step 5: Diagnostics ---
    if step % 100 == 0
        mean_νₑ = mean_eddy_viscosity(sgs)
        max_νₑ = max_eddy_viscosity(sgs)

        println("Step $step:")
        println("  Mean eddy viscosity: $(mean_νₑ)")
        println("  Max eddy viscosity:  $(max_νₑ)")
        println("  Ratio νₑ/ν (mean):   $(mean_νₑ/ν)")
    end
end
```

---

## Complete Example: LES of Rayleigh-Bénard Convection

For buoyancy-driven turbulence, we need both momentum and scalar closures:

```julia
using Tarang

# ============================================================
# Physical Parameters
# ============================================================

Nx, Nz = 256, 128           # Grid resolution
Lx, Lz = 4.0, 1.0           # Domain size
Δx, Δz = Lx/Nx, Lz/Nz       # Grid spacing

Ra = 1e8                    # Rayleigh number
Pr = 1.0                    # Prandtl number
ν = sqrt(Pr / Ra)           # Kinematic viscosity
κ = ν / Pr                  # Thermal diffusivity

# ============================================================
# Create AMD Model (recommended for RBC)
# ============================================================

sgs = AMDModel(
    C = 1/12,
    filter_width = (Δx, Δz),
    field_size = (Nx, Nz)
)

# ============================================================
# In the Time Loop
# ============================================================

for step in 1:nsteps
    # Compute velocity gradients
    ∂u∂x, ∂u∂z = gradient(u)
    ∂w∂x, ∂w∂z = gradient(w)

    # Compute temperature gradients
    ∂T∂x, ∂T∂z = gradient(T)

    # --- Momentum closure ---
    compute_eddy_viscosity!(sgs, ∂u∂x, ∂u∂z, ∂w∂x, ∂w∂z)
    νₑ = get_eddy_viscosity(sgs)
    ν_eff = ν .+ νₑ

    # --- Scalar (temperature) closure ---
    compute_eddy_diffusivity!(sgs, ∂w∂x, ∂w∂z, ∂T∂x, ∂T∂z)
    κₑ = get_eddy_diffusivity(sgs)
    κ_eff = κ .+ κₑ

    # Use ν_eff in momentum equation
    # Use κ_eff in temperature equation
end
```

---

## Diagnostics and Analysis

### SGS Energy Dissipation

The SGS dissipation rate tells you how much energy is being drained by the model:

```julia
# Get strain magnitude (computed during eddy viscosity calculation)
S_mag = sgs.strain_magnitude  # Only for Smagorinsky

# Or compute it yourself
S_mag = sqrt(2 * (S11.^2 + S22.^2 + S33.^2 + 2*S12.^2 + 2*S13.^2 + 2*S23.^2))

# SGS dissipation field: ε_sgs = 2 νₑ |S|²
ε_sgs = sgs_dissipation(sgs, S_mag)

# Domain-averaged SGS dissipation
ε_sgs_mean = mean_sgs_dissipation(sgs, S_mag)
```

### Monitoring Model Behavior

```julia
# Check ratio of SGS to molecular viscosity
νₑ_mean = mean_eddy_viscosity(sgs)
println("νₑ/ν = $(νₑ_mean/ν)")

# If νₑ/ν >> 1: SGS model is dominant (typical for high Re)
# If νₑ/ν << 1: Flow is nearly DNS-resolved
# If νₑ/ν ~ 1: Well-resolved LES

# For AMD: check how often νₑ = 0
νₑ = get_eddy_viscosity(sgs)
fraction_zero = sum(νₑ .== 0) / length(νₑ)
println("Fraction with νₑ = 0: $(fraction_zero)")
# High fraction → flow is largely laminar/transitional
```

---

## Tips for Successful LES

### Resolution Requirements

LES still requires adequate resolution:

```
┌────────────────────────────────────────────────────────────┐
│  Resolution Quality for LES                                 │
├────────────────────────────────────────────────────────────┤
│  80% of TKE resolved      →  Minimum acceptable LES        │
│  90% of TKE resolved      →  Good quality LES              │
│  95%+ of TKE resolved     →  Nearly DNS quality            │
└────────────────────────────────────────────────────────────┘
```

Rule of thumb: Grid spacing should resolve the inertial range, typically $\Delta \lesssim L_{integral}/10$.

### Common Pitfalls

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Simulation blows up | νₑ too small | Increase $C_s$ or $C$ |
| Flow looks too smooth | Over-dissipation | Reduce $C_s$, or use AMD |
| Checkerboard patterns | Aliasing errors | Enable dealiasing (2/3 rule) |
| νₑ unrealistically large | Poor resolution | Refine grid |
| AMD gives νₑ = 0 everywhere | Flow is laminar | This is correct! |

### When LES May Not Be Appropriate

- **Very low Reynolds numbers**: DNS may be feasible
- **Strongly anisotropic turbulence**: May need special treatment
- **Flows with strong backscatter**: Standard models don't capture this
- **Near-wall regions**: May need wall models or finer grids

---

## 2D Flows

Both models support 2D simulations:

```julia
# 2D Smagorinsky
sgs_2d = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (Δx, Δy),
    field_size = (Nx, Ny)
)

# 2D velocity gradients (4 components)
compute_eddy_viscosity!(sgs_2d, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
```

**Note on 2D turbulence**: 2D turbulence has an *inverse* energy cascade (energy flows to large scales), which is fundamentally different from 3D. Standard SGS models may not be appropriate for 2D flows.

---

## API Reference

### Constructors

```julia
SmagorinskyModel(;
    C_s = 0.17,                    # Smagorinsky constant
    filter_width::NTuple{N, Real}, # (Δx, Δy) or (Δx, Δy, Δz)
    field_size::NTuple{N, Int},    # (Nx, Ny) or (Nx, Ny, Nz)
    dtype = Float64                # Precision
)

AMDModel(;
    C = 1/12,                      # Poincaré constant
    filter_width::NTuple{N, Real}, # Can be anisotropic
    field_size::NTuple{N, Int},
    clip_negative = true,          # Ensure νₑ ≥ 0
    dtype = Float64
)
```

### Core Functions

| Function | Description |
|----------|-------------|
| `compute_eddy_viscosity!(model, grads...)` | Compute νₑ from velocity gradients |
| `compute_eddy_diffusivity!(model, grads...)` | Compute κₑ for scalars (AMD only) |
| `get_eddy_viscosity(model)` | Return the νₑ field |
| `get_eddy_diffusivity(model)` | Return the κₑ field (AMD) |

### Analysis Functions

| Function | Description |
|----------|-------------|
| `mean_eddy_viscosity(model)` | Domain-averaged νₑ |
| `max_eddy_viscosity(model)` | Maximum νₑ |
| `sgs_dissipation(model, S_mag)` | SGS dissipation field |
| `mean_sgs_dissipation(model, S_mag)` | Domain-averaged dissipation |

### Utility Functions

| Function | Description |
|----------|-------------|
| `set_constant!(model, C)` | Update model constant |
| `reset!(model)` | Reset νₑ (and κₑ) to zero |
| `get_filter_width(model)` | Return filter width tuple |
| `compute_sgs_stress(model, S...)` | Compute full SGS stress tensor |

---

## Troubleshooting

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Simulation becomes unstable | νₑ too small | Increase $C_s$ or $C$; check CFL condition |
| Flow appears over-damped | νₑ too large | Reduce $C_s$; consider AMD model |
| Spurious oscillations | Aliasing | Enable 2/3 dealiasing rule |
| AMD gives νₑ = 0 everywhere | Flow is laminar | Correct behavior; model switches off |
| Unphysical behavior near walls | Poor wall resolution | Refine near-wall grid; use wall functions |

---

## References

### Original Papers

1. **Smagorinsky, J.** (1963). "General circulation experiments with the primitive equations: I. The basic experiment." *Monthly Weather Review*, 91(3), 99-164.

2. **Rozema, W., Bae, H. J., Moin, P., & Verstappen, R.** (2015). "Minimum-dissipation models for large-eddy simulation." *Physics of Fluids*, 27(8), 085107. [PDF](http://www.its.caltech.edu/~jbae/publications/Rozema_2015.pdf)

3. **Abkar, M., Bae, H. J., & Moin, P.** (2016). "Minimum-dissipation scalar transport model for large-eddy simulation of turbulent flows." *Physical Review Fluids*, 1(4), 041701. [Link](https://link.aps.org/doi/10.1103/PhysRevFluids.1.041701)

### Textbooks and Reviews

4. **Pope, S. B.** (2000). *Turbulent Flows*. Cambridge University Press. (Chapter 13: Large-Eddy Simulation)

5. **Sagaut, P.** (2006). *Large Eddy Simulation for Incompressible Flows*. Springer.

### Related Documentation

- [dedaLES AMD Documentation](https://dedales.readthedocs.io/en/latest/closures/anisotropic_minimum_dissipation.html)

---

## See Also

- [Solvers](solvers.md) - Time integration methods
- [Operators](operators.md) - Gradient and differential operators
- [Stochastic Forcing](stochastic_forcing.md) - Forcing for turbulence
- [API: LES Models](../api/les_models.md) - Complete API reference
