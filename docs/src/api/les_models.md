# LES Models API

The SGS models are **array-level utilities**: they consume grid-space velocity-gradient
arrays and produce an eddy-viscosity array. They are not automatically coupled into an
`IVP` — you evaluate the gradients, call the model, and apply the resulting stress
yourself.

## Usage

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
domain = Domain(dist, (xbasis, ybasis))

u = VectorField(domain, "u")
x, y = local_grids(dist, xbasis, ybasis)
ensure_layout!(u, :g)
get_grid_data(u.components[1]) .=  sin.(x) .* cos.(y')
get_grid_data(u.components[2]) .= .-cos.(x) .* sin.(y')
ensure_layout!(u, :c)

# grad(u) is a TensorField; its components are the velocity-gradient tensor in
# component-major order: (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) — exactly the order the models want.
G = evaluate(grad(u))
∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y = (get_grid_data(g) for g in G.components)

Δx, Δy = grid_spacing(domain)
model  = SmagorinskyModel(C_s=0.17, filter_width=(Δx, Δy), field_size=(16, 16))

compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

νₑ = get_eddy_viscosity(model)          # 16×16 Matrix{Float64}
mean_eddy_viscosity(model)              # 0.00352…
max_eddy_viscosity(model)               # 0.00891…

# deviatoric SGS stress τᵢⱼ = -2 νₑ S̄ᵢⱼ
S11 = ∂u∂x
S12 = 0.5 .* (∂u∂y .+ ∂v∂x)
S22 = ∂v∂y
τ11, τ12, τ22 = compute_sgs_stress(model, S11, S12, S22)

mean_sgs_dissipation(model, model.strain_magnitude)   # 0.00643…
```

## Usage under MPI

The models hold **plain per-rank arrays**, so size the model to *this rank's slab*, not to the
global grid. Under MPI `get_grid_data` returns a `PencilArray` whose `size` is already the local
shape; `parent` gives the raw local storage:

```julia
G = evaluate(grad(u))
∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y = (parent(get_grid_data(g)) for g in G.components)

Δx, Δy = grid_spacing(domain)
model  = SmagorinskyModel(C_s=0.17, filter_width=(Δx, Δy),
                          field_size=size(∂u∂x))   # LOCAL slab: (16, 8) at np=2

compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

mean_eddy_viscosity(model)      # 0.00352… — global, Allreduced across ranks
max_eddy_viscosity(model)       # 0.00891… — global
```

`mean_eddy_viscosity`, `max_eddy_viscosity` and `mean_sgs_dissipation` each perform their own
reduction over `MPI.COMM_WORLD`, so they return domain-global values: for the 16×16 case above
they give the same numbers at np = 1, 2 and 4. Being collectives, they must be called on **every**
rank — reaching for one inside an `if rank == 0` block deadlocks. Pass `global_reduce=false` to
skip the reduction and get this rank's slab value, which is safe to call from a subset of ranks:

```julia
# Deadlocks at np > 1 — only rank 0 enters the collective.
rank == 0 && @info "mean νₑ" mean_eddy_viscosity(model)

# Correct: every rank reduces, one rank prints.
ν̄ = mean_eddy_viscosity(model)
rank == 0 && @info "mean νₑ" ν̄

# Or opt out of the reduction entirely.
rank == 0 && @info "local νₑ" mean_eddy_viscosity(model; global_reduce=false)
```

`get_eddy_viscosity(model)` is, by contrast, this rank's slab — broadcast it against the other
`parent(...)` arrays to build the stress locally.

Sizing the model to the global grid is a loud error rather than a silent wrong answer:

```
DimensionMismatch: Gradient array 1 has size (16, 8), expected (16, 16)
```

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
mutable struct SmagorinskyModel{T<:AbstractFloat, N,
                                A<:AbstractArray{T, N},
                                Arch<:AbstractArchitecture} <: EddyViscosityModel
```

The array parameter `A` is the type the model's internal buffers are stored in — an
`Array{T,N}` on `CPU()`, a `CuArray` on `GPU()`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `C_s` | `T` | Smagorinsky constant |
| `filter_width` | `NTuple{N, T}` | Filter width (Δx, Δy, Δz) |
| `effective_delta` | `T` | Effective Δ = (Δx Δy Δz)^(1/N) |
| `eddy_viscosity` | `A` | νₑ field |
| `strain_magnitude` | `A` | \|S̄\| field |
| `field_size` | `NTuple{N, Int}` | Grid dimensions |
| `architecture` | `Arch` | `CPU()` or `GPU()` |

**Constructor:**

```julia
SmagorinskyModel(;
    C_s = 0.17,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    dtype = Float64,
    architecture = CPU()
)
```

### AMDModel

```@docs
AMDModel
```

Anisotropic Minimum Dissipation model (Rozema et al., 2015).

**Type signature:**
```julia
mutable struct AMDModel{T<:AbstractFloat, N,
                        A<:AbstractArray{T, N},
                        Arch<:AbstractArchitecture} <: EddyViscosityModel
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `C` | `T` | Poincaré constant |
| `filter_width` | `NTuple{N, T}` | Anisotropic filter widths |
| `filter_width_sq` | `NTuple{N, T}` | Δₖ² for each direction |
| `eddy_viscosity` | `A` | νₑ field |
| `eddy_diffusivity` | `A` | κₑ field (for scalars) |
| `field_size` | `NTuple{N, Int}` | Grid dimensions |
| `clip_negative` | `Bool` | Whether to clip νₑ < 0 |
| `architecture` | `Arch` | `CPU()` or `GPU()` |

**Constructor:**

```julia
AMDModel(;
    C = 1/12,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    clip_negative = true,
    dtype = Float64,
    architecture = CPU()
)
```

---

## Eddy Viscosity Computation

### compute_eddy_viscosity!

```@docs
compute_eddy_viscosity!
```

Compute eddy viscosity from velocity gradient components. The gradients are plain
grid-space arrays of size `model.field_size`, passed in **component-major order**
(all derivatives of `u`, then all of `v`, then all of `w`).

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

Compute eddy diffusivity for scalar transport (AMD model only). The AMD diffusivity
(Abkar, Bae & Moin 2016, eq. 2.7) is a full double contraction over the scaled-gradient
direction *k* **and** every velocity component *i*:

    κₑ† = -C · [ Σₖ Δₖ² (∂ₖuᵢ)(∂ₖb)(∂ᵢb) ] / [ (∂ₗb)(∂ₗb) ]

so the method needs the **complete velocity-gradient tensor** (2D: 4 components, 3D: 9),
not just the gradient of one velocity component. The scalar gradients follow.

**2D Signature:**
```julia
compute_eddy_diffusivity!(model::AMDModel,
    ∂u∂x, ∂u∂y,
    ∂v∂x, ∂v∂y,
    ∂b∂x, ∂b∂y
)
```

**3D Signature:**
```julia
compute_eddy_diffusivity!(model::AMDModel,
    ∂u∂x, ∂u∂y, ∂u∂z,
    ∂v∂x, ∂v∂y, ∂v∂z,
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

Return the current eddy viscosity field (an `Array` on `CPU()`, a `CuArray` on `GPU()`).

```julia
get_eddy_viscosity(model::EddyViscosityModel) -> A <: AbstractArray{T, N}
```

### get_eddy_diffusivity

```@docs
get_eddy_diffusivity
```

Return the current eddy diffusivity field (AMD only).

```julia
get_eddy_diffusivity(model::AMDModel) -> A <: AbstractArray{T, N}
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

Compute the SGS dissipation rate field: εₛₛ = νₑ |S̄|²

There is **no** extra factor of 2 here. The exact dissipation is
εₛₛ = -τᵢⱼ S̄ᵢⱼ = 2 νₑ S̄ᵢⱼS̄ᵢⱼ, and the `|S̄| = √(2 S̄ᵢⱼS̄ᵢⱼ)` convention used by
`compute_eddy_viscosity!` already carries it, so εₛₛ = νₑ |S̄|².

```julia
sgs_dissipation(model::EddyViscosityModel, strain_magnitude) -> A <: AbstractArray{T, N}
```

The natural argument is the strain magnitude the model just cached,
`model.strain_magnitude` (Smagorinsky only — the AMD model does not cache one).

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
export reset!, set_constant!, set_filter_width!
export sgs_dissipation, mean_sgs_dissipation
```

### Mutating a model

`set_constant!(model, C)` and `set_filter_width!(model, Δ)` validate their argument and keep the
cached derived quantities (`filter_width_sq`, `effective_delta`) in step. Assigning
`model.filter_width` directly is also honoured — the kernels re-derive Δ² and the geometric-mean Δ
on every call — but it leaves those cached fields reading stale, so prefer the setter.

---

## Index

```@index
Pages = ["les_models.md"]
```
