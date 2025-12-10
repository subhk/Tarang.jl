# Large Eddy Simulation (LES) Models

This page describes the subgrid-scale (SGS) closure models available in Tarang.jl for Large Eddy Simulation.

---

## Complete Example: LES of Isotropic Turbulence

```julia
using Tarang
using LinearAlgebra

# ============================================================
# 1. Grid and Physical Parameters
# ============================================================

N = 128                     # Grid points per direction
L = 2π                      # Domain size
Δ = L / N                   # Grid spacing
ν = 1e-4                    # Molecular viscosity
dt = 0.001

# ============================================================
# 2. Create SGS Model
# ============================================================

# Option A: Smagorinsky model (classic, simple)
sgs_model = SmagorinskyModel(
    C_s = 0.17,                        # Smagorinsky constant
    filter_width = (Δ, Δ, Δ),          # Isotropic grid
    field_size = (N, N, N)
)

# Option B: AMD model (modern, handles anisotropic grids)
sgs_model = AMDModel(
    C = 1/12,                          # Poincaré constant (spectral)
    filter_width = (Δ, Δ, Δ),          # Can be anisotropic
    field_size = (N, N, N)
)

# ============================================================
# 3. Setup Domain and Fields
# ============================================================

coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; mesh=(1, 1, 1))

xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
zbasis = RealFourier(coords["z"]; size=N, bounds=(0.0, L))

domain = Domain(dist, (xbasis, ybasis, zbasis))

u = VectorField(dist, coords, "u", (xbasis, ybasis, zbasis))

# ============================================================
# 4. Time Integration with SGS Closure
# ============================================================

function compute_velocity_gradients(u)
    # Compute all 9 velocity gradient components
    # (using spectral differentiation)
    ∂u∂x, ∂u∂y, ∂u∂z = gradient(u.components[1])
    ∂v∂x, ∂v∂y, ∂v∂z = gradient(u.components[2])
    ∂w∂x, ∂w∂y, ∂w∂z = gradient(u.components[3])
    return (∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)
end

for step in 1:1000
    # 1. Compute velocity gradients
    grads = compute_velocity_gradients(u)

    # 2. Compute SGS eddy viscosity
    compute_eddy_viscosity!(sgs_model, grads...)

    # 3. Get effective viscosity: ν_eff = ν + νₑ
    νₑ = get_eddy_viscosity(sgs_model)
    ν_eff = ν .+ νₑ

    # 4. Use ν_eff in your momentum equation
    # (SGS stress divergence is modeled as νₑ ∇²u)

    # 5. Diagnostics
    if step % 100 == 0
        println("Step $step: mean νₑ = $(mean_eddy_viscosity(sgs_model))")
        println("         max νₑ = $(max_eddy_viscosity(sgs_model))")
    end
end
```

---

## Choosing a Model

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Smagorinsky** | Simple flows, quick setup | Simple, robust, cheap | Over-dissipative, poor near walls |
| **AMD** | Anisotropic grids, transitional flows | Automatic switch-off, anisotropic | Slightly more expensive |

**Decision flowchart:**

```
Is your grid anisotropic (Δx ≠ Δy ≠ Δz)?
├── Yes → AMD (handles anisotropy naturally)
└── No  → Does the flow have laminar/transitional regions?
          ├── Yes → AMD (switches off automatically)
          └── No  → Either works, Smagorinsky is simpler
```

---

## Smagorinsky Model

### Mathematical Formulation

The classic Smagorinsky (1963) model computes eddy viscosity as:

```math
\nu_e = (C_s \Delta)^2 |\bar{S}|
```

where:
- $C_s$ is the Smagorinsky constant
- $\Delta = (\Delta_x \Delta_y \Delta_z)^{1/3}$ is the effective filter width
- $|\bar{S}| = \sqrt{2 \bar{S}_{ij} \bar{S}_{ij}}$ is the strain rate magnitude

The strain rate tensor is:

```math
\bar{S}_{ij} = \frac{1}{2}\left(\frac{\partial \bar{u}_i}{\partial x_j} + \frac{\partial \bar{u}_j}{\partial x_i}\right)
```

### Usage

```julia
model = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (Δx, Δy, Δz),
    field_size = (Nx, Ny, Nz)
)

# Compute νₑ from velocity gradients
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z,
                               ∂v∂x, ∂v∂y, ∂v∂z,
                               ∂w∂x, ∂w∂y, ∂w∂z)

νₑ = get_eddy_viscosity(model)
```

### Recommended Constants

| Flow Type | C_s |
|-----------|-----|
| Isotropic turbulence | 0.17 - 0.20 |
| Channel flow | 0.10 - 0.12 |
| Mixing layers | 0.10 - 0.14 |
| Free shear flows | 0.10 - 0.12 |

---

## Anisotropic Minimum Dissipation (AMD) Model

### Mathematical Formulation

The AMD model (Rozema et al., 2015) computes:

```math
\nu_e = \max\left(0, -C \frac{\Delta_k^2 \frac{\partial u_i}{\partial x_k} \frac{\partial u_j}{\partial x_k} S_{ij}}{\frac{\partial u_m}{\partial x_n} \frac{\partial u_m}{\partial x_n}}\right)
```

Key features:
- **Anisotropic filter widths**: Uses $\Delta_k$ in each direction separately
- **Automatic switch-off**: $\nu_e = 0$ in laminar regions (no forcing required)
- **Minimum dissipation**: Provides exactly the dissipation needed

### Usage

```julia
model = AMDModel(
    C = 1/12,                           # For spectral methods
    filter_width = (Δx, Δy, Δz),        # Can be anisotropic
    field_size = (Nx, Ny, Nz),
    clip_negative = true                # Ensure νₑ ≥ 0
)

compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z,
                               ∂v∂x, ∂v∂y, ∂v∂z,
                               ∂w∂x, ∂w∂y, ∂w∂z)
```

### Recommended Constants

| Discretization | C |
|----------------|---|
| Spectral methods | 1/12 ≈ 0.0833 |
| 4th-order finite difference | 0.212 |
| 2nd-order finite difference | 0.3 |

### Scalar Transport

For buoyancy-driven flows, AMD also provides eddy diffusivity:

```julia
compute_eddy_diffusivity!(model, ∂w∂x, ∂w∂y, ∂w∂z,
                                 ∂b∂x, ∂b∂y, ∂b∂z)

κₑ = get_eddy_diffusivity(model)
```

---

## Computing Subgrid Stress

The deviatoric SGS stress is:

```math
\tau_{ij}^d = -2 \nu_e \bar{S}_{ij}
```

```julia
# First compute strain rate components
S11 = ∂u∂x
S22 = ∂v∂y
S33 = ∂w∂z
S12 = 0.5 * (∂u∂y + ∂v∂x)
S13 = 0.5 * (∂u∂z + ∂w∂x)
S23 = 0.5 * (∂v∂z + ∂w∂y)

# Compute SGS stress
(τ11, τ12, τ13, τ22, τ23, τ33) = compute_sgs_stress(
    model, S11, S12, S13, S22, S23, S33
)
```

---

## SGS Dissipation Rate

The SGS dissipation rate is:

```math
\varepsilon_{sgs} = 2 \nu_e |\bar{S}|^2
```

```julia
# Get strain magnitude (stored by Smagorinsky model)
|S̄| = model.strain_magnitude

# Compute dissipation field
ε_sgs = sgs_dissipation(model, |S̄|)

# Domain average
⟨ε_sgs⟩ = mean_sgs_dissipation(model, |S̄|)
```

---

## Anisotropic Grids

The AMD model excels on anisotropic grids (e.g., channel flow with wall refinement):

```julia
# Channel flow: fine near walls, coarse in center
Δx = Lx / Nx
Δy = Ly / Ny  # Uniform in y
Δz_wall = 0.001   # Fine at walls
Δz_center = 0.01  # Coarse in center

# For stretched grids, use local Δz at each grid point
# AMD handles this naturally through Δk² terms
```

Unlike Smagorinsky, AMD:
- Does not require an ad-hoc filter width definition
- Properly reduces eddy viscosity near walls
- Handles grid anisotropy without tuning

---

## 2D Flows

Both models support 2D flows:

```julia
# 2D Smagorinsky
model_2d = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (Δx, Δy),
    field_size = (Nx, Ny)
)

compute_eddy_viscosity!(model_2d, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
```

---

## API Reference

### Constructors

```julia
SmagorinskyModel(;
    C_s = 0.17,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    dtype = Float64
)

AMDModel(;
    C = 1/12,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    clip_negative = true,
    dtype = Float64
)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `compute_eddy_viscosity!(model, grads...)` | Compute νₑ from velocity gradients |
| `compute_eddy_diffusivity!(model, grads...)` | Compute κₑ for scalars (AMD only) |
| `compute_sgs_stress(model, S...)` | Compute τᵢⱼ from strain rate |
| `get_eddy_viscosity(model)` | Get νₑ field |
| `get_eddy_diffusivity(model)` | Get κₑ field (AMD only) |
| `mean_eddy_viscosity(model)` | Domain-averaged νₑ |
| `max_eddy_viscosity(model)` | Maximum νₑ |
| `sgs_dissipation(model, \|S\|)` | SGS dissipation field |
| `set_constant!(model, C)` | Update model constant |
| `reset!(model)` | Reset νₑ to zero |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Simulation blows up | νₑ too small | Increase C_s or C |
| Over-dissipation | C_s too large | Reduce C_s, or switch to AMD |
| νₑ always zero (AMD) | Flow is laminar | This is correct behavior! |
| Checkerboard patterns | Aliasing | Use dealiasing or 2/3 rule |

---

## References

1. Smagorinsky, J. (1963). "General circulation experiments with the primitive equations"
2. [Rozema et al. (2015). "Minimum-dissipation models for large-eddy simulation"](http://www.its.caltech.edu/~jbae/publications/Rozema_2015.pdf), Physics of Fluids 27, 085107.
3. [Abkar et al. (2016). "Minimum-dissipation scalar transport model"](https://link.aps.org/doi/10.1103/PhysRevFluids.1.041701)
4. [dedaLES AMD Documentation](https://dedales.readthedocs.io/en/latest/closures/anisotropic_minimum_dissipation.html)

---

## See Also

- [Solvers](solvers.md) - Time integration
- [Operators](operators.md) - Gradient computation
- [API: LES Models](../api/les_models.md) - Complete API reference
