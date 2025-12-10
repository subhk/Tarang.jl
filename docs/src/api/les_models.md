# LES Models API

## Types

### Abstract Types

```julia
abstract type SGSModel end
abstract type EddyViscosityModel <: SGSModel end
```

### SmagorinskyModel

```@docs
SmagorinskyModel
```

Classic Smagorinsky (1963) subgrid-scale model.

**Type signature:**
```julia
mutable struct SmagorinskyModel{T<:AbstractFloat, N} <: EddyViscosityModel
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `C_s` | `T` | Smagorinsky constant |
| `filter_width` | `NTuple{N, T}` | Filter width (Δx, Δy, Δz) |
| `effective_delta` | `T` | Effective Δ = (Δx Δy Δz)^(1/N) |
| `eddy_viscosity` | `Array{T, N}` | νₑ field |
| `strain_magnitude` | `Array{T, N}` | \|S̄\| field |
| `field_size` | `NTuple{N, Int}` | Grid dimensions |

**Constructor:**

```julia
SmagorinskyModel(;
    C_s = 0.17,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    dtype = Float64
)
```

### AMDModel

```@docs
AMDModel
```

Anisotropic Minimum Dissipation model (Rozema et al., 2015).

**Type signature:**
```julia
mutable struct AMDModel{T<:AbstractFloat, N} <: EddyViscosityModel
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `C` | `T` | Poincaré constant |
| `filter_width` | `NTuple{N, T}` | Anisotropic filter widths |
| `filter_width_sq` | `NTuple{N, T}` | Δₖ² for each direction |
| `eddy_viscosity` | `Array{T, N}` | νₑ field |
| `eddy_diffusivity` | `Array{T, N}` | κₑ field (for scalars) |
| `field_size` | `NTuple{N, Int}` | Grid dimensions |
| `clip_negative` | `Bool` | Whether to clip νₑ < 0 |

**Constructor:**

```julia
AMDModel(;
    C = 1/12,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    clip_negative = true,
    dtype = Float64
)
```

---

## Eddy Viscosity Computation

### compute_eddy_viscosity!

```@docs
compute_eddy_viscosity!
```

Compute eddy viscosity from velocity gradient components.

**2D Signature:**
```julia
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
```

**3D Signature:**
```julia
compute_eddy_viscosity!(model,
    ∂u∂x, ∂u∂y, ∂u∂z,
    ∂v∂x, ∂v∂y, ∂v∂z,
    ∂w∂x, ∂w∂y, ∂w∂z
)
```

**Returns:** The eddy viscosity array `model.eddy_viscosity`

### compute_eddy_diffusivity!

```@docs
compute_eddy_diffusivity!
```

Compute eddy diffusivity for scalar transport (AMD model only).

```julia
compute_eddy_diffusivity!(model::AMDModel,
    ∂w∂x, ∂w∂y, ∂w∂z,
    ∂b∂x, ∂b∂y, ∂b∂z
)
```

**Returns:** The eddy diffusivity array `model.eddy_diffusivity`

---

## Subgrid Stress

### compute_sgs_stress

```@docs
compute_sgs_stress
```

Compute deviatoric SGS stress tensor τᵢⱼ = -2 νₑ S̄ᵢⱼ.

**2D Signature:**
```julia
compute_sgs_stress(model, S11, S12, S22) -> (τ11, τ12, τ22)
```

**3D Signature:**
```julia
compute_sgs_stress(model, S11, S12, S13, S22, S23, S33)
    -> (τ11, τ12, τ13, τ22, τ23, τ33)
```

---

## Accessors

### get_eddy_viscosity

```@docs
get_eddy_viscosity
```

Return the current eddy viscosity field.

```julia
get_eddy_viscosity(model::EddyViscosityModel) -> Array{T, N}
```

### get_eddy_diffusivity

```@docs
get_eddy_diffusivity
```

Return the current eddy diffusivity field (AMD only).

```julia
get_eddy_diffusivity(model::AMDModel) -> Array{T, N}
```

### get_filter_width

```@docs
get_filter_width
```

Return the filter width tuple.

```julia
get_filter_width(model::EddyViscosityModel) -> NTuple{N, T}
```

---

## Statistics

### mean_eddy_viscosity

```@docs
mean_eddy_viscosity
```

Compute domain-averaged eddy viscosity.

```julia
mean_eddy_viscosity(model::EddyViscosityModel) -> T
```

### max_eddy_viscosity

```@docs
max_eddy_viscosity
```

Return maximum eddy viscosity in the domain.

```julia
max_eddy_viscosity(model::EddyViscosityModel) -> T
```

---

## Dissipation Rate

### sgs_dissipation

```@docs
sgs_dissipation
```

Compute SGS dissipation rate field: εₛₛ = 2 νₑ |S̄|²

```julia
sgs_dissipation(model::EddyViscosityModel, strain_magnitude) -> Array{T, N}
```

### mean_sgs_dissipation

```@docs
mean_sgs_dissipation
```

Compute domain-averaged SGS dissipation rate.

```julia
mean_sgs_dissipation(model::EddyViscosityModel, strain_magnitude) -> T
```

---

## Configuration

### set_constant!

```@docs
set_constant!
```

Update the model constant.

```julia
set_constant!(model::SmagorinskyModel, C_s::Real)
set_constant!(model::AMDModel, C::Real)
```

### reset!

Reset eddy viscosity (and diffusivity for AMD) to zero.

```julia
reset!(model::EddyViscosityModel)
reset!(model::AMDModel)  # Also resets eddy_diffusivity
```

---

## Exports

```julia
export SGSModel, EddyViscosityModel
export SmagorinskyModel, AMDModel
export compute_eddy_viscosity!, compute_eddy_diffusivity!
export compute_sgs_stress
export get_eddy_viscosity, get_eddy_diffusivity, get_filter_width
export mean_eddy_viscosity, max_eddy_viscosity
export reset!, set_constant!
export sgs_dissipation, mean_sgs_dissipation
```

---

## Index

```@index
Pages = ["les_models.md"]
```
