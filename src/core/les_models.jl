"""
# Large Eddy Simulation (LES) Subgrid-Scale Models

This module provides subgrid-scale (SGS) closure models for Large Eddy Simulation:

1. **Smagorinsky Model** (Smagorinsky, 1963): Classic eddy-viscosity model
2. **Anisotropic Minimum Dissipation (AMD)** (Rozema et al., 2015): Modern model
   that handles anisotropic grids and properly switches off in laminar regions

## Mathematical Background

### Filtered Navier-Stokes Equations

LES solves the filtered equations:

    ∂ū_i/∂t + ū_j ∂ū_i/∂x_j = -∂p̄/∂x_i + ν∇²ū_i - ∂τᵢⱼ/∂x_j

where τᵢⱼ = ūᵢūⱼ - (u̅ᵢu̅ⱼ) is the subgrid stress tensor.

### Eddy Viscosity Models

Both models approximate the subgrid stress as:

    τᵢⱼ - (1/3)τₖₖδᵢⱼ = -2νₑS̄ᵢⱼ

where S̄ᵢⱼ = (1/2)(∂ūᵢ/∂xⱼ + ∂ūⱼ/∂xᵢ) is the resolved strain rate tensor.

## References

1. Smagorinsky, J. (1963). "General circulation experiments with the primitive equations"
2. Rozema, W., Bae, H.J., Moin, P., Verstappen, R. (2015). "Minimum-dissipation models
   for large-eddy simulation", Physics of Fluids 27, 085107.
3. Abkar, M., Bae, H.J., Moin, P. (2016). "Minimum-dissipation scalar transport model"
"""

using LinearAlgebra

# ============================================================================
# Abstract Types
# ============================================================================

"""
    SGSModel

Abstract base type for all subgrid-scale models.
"""
abstract type SGSModel end

"""
    EddyViscosityModel <: SGSModel

Abstract type for eddy-viscosity based SGS models.
"""
abstract type EddyViscosityModel <: SGSModel end

# ============================================================================
# Smagorinsky Model
# ============================================================================

"""
    SmagorinskyModel{T, N}

Classic Smagorinsky (1963) subgrid-scale model.

## Mathematical Formulation

The eddy viscosity is:

    νₑ = (Cₛ Δ)² |S̄|

where:
- Cₛ is the Smagorinsky constant (typically 0.1-0.2)
- Δ is the filter width (grid spacing)
- |S̄| = √(2 S̄ᵢⱼ S̄ᵢⱼ) is the strain rate magnitude

## Fields

- `C_s::T`: Smagorinsky constant
- `filter_width::NTuple{N, T}`: Filter width in each direction (Δx, Δy, ...)
- `eddy_viscosity::Array{T, N}`: Cached eddy viscosity field
- `strain_magnitude::Array{T, N}`: Cached |S̄| field

## Example

```julia
# Create model for 256³ domain with Δ = 2π/256
model = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (2π/256, 2π/256, 2π/256),
    field_size = (256, 256, 256)
)

# Compute eddy viscosity from velocity gradients
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)

# Access the result
νₑ = get_eddy_viscosity(model)
```
"""
mutable struct SmagorinskyModel{T<:AbstractFloat, N} <: EddyViscosityModel
    C_s::T                              # Smagorinsky constant
    filter_width::NTuple{N, T}          # (Δx, Δy, Δz)
    effective_delta::T                  # Effective filter width Δ = (Δx Δy Δz)^(1/3)
    eddy_viscosity::Array{T, N}         # νₑ field
    strain_magnitude::Array{T, N}       # |S̄| field
    field_size::NTuple{N, Int}
end

"""
    SmagorinskyModel(;
        C_s = 0.17,
        filter_width,
        field_size,
        dtype = Float64
    )

Create a Smagorinsky SGS model.

## Arguments

- `C_s::Real`: Smagorinsky constant (default: 0.17, suitable for isotropic turbulence)
- `filter_width::NTuple{N, Real}`: Grid spacing (Δx, Δy) or (Δx, Δy, Δz)
- `field_size::NTuple{N, Int}`: Grid dimensions
- `dtype::Type`: Floating point type (default: Float64)

## Recommended Constants

| Flow Type | C_s |
|-----------|-----|
| Isotropic turbulence | 0.17-0.20 |
| Channel flow | 0.10-0.12 |
| Mixing layers | 0.10-0.14 |
| Free shear flows | 0.10-0.12 |
"""
function SmagorinskyModel(;
    C_s::Real = 0.17,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N}

    # Effective filter width: geometric mean
    effective_delta = T(prod(filter_width)^(1/N))

    # Allocate arrays
    eddy_viscosity = zeros(T, field_size)
    strain_magnitude = zeros(T, field_size)

    SmagorinskyModel{T, N}(
        T(C_s),
        T.(filter_width),
        effective_delta,
        eddy_viscosity,
        strain_magnitude,
        field_size
    )
end

"""
    compute_eddy_viscosity!(model::SmagorinskyModel, velocity_gradients...)

Compute eddy viscosity from velocity gradient components.

## 2D Case
```julia
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
```

## 3D Case
```julia
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)
```
"""
function compute_eddy_viscosity!(
    model::SmagorinskyModel{T, 2},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}
) where T

    Δ = model.effective_delta
    C_s = model.C_s

    @inbounds for i in eachindex(model.strain_magnitude)
        # Strain rate tensor components
        S11 = ∂u∂x[i]
        S22 = ∂v∂y[i]
        S12 = T(0.5) * (∂u∂y[i] + ∂v∂x[i])

        # |S̄| = √(2 Sᵢⱼ Sᵢⱼ)
        S_mag = sqrt(2 * (S11^2 + S22^2 + 2*S12^2))
        model.strain_magnitude[i] = S_mag

        # νₑ = (Cₛ Δ)² |S̄|
        model.eddy_viscosity[i] = (C_s * Δ)^2 * S_mag
    end

    return model.eddy_viscosity
end

function compute_eddy_viscosity!(
    model::SmagorinskyModel{T, 3},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T}
) where T

    Δ = model.effective_delta
    C_s = model.C_s

    @inbounds for i in eachindex(model.strain_magnitude)
        # Strain rate tensor components
        S11 = ∂u∂x[i]
        S22 = ∂v∂y[i]
        S33 = ∂w∂z[i]
        S12 = T(0.5) * (∂u∂y[i] + ∂v∂x[i])
        S13 = T(0.5) * (∂u∂z[i] + ∂w∂x[i])
        S23 = T(0.5) * (∂v∂z[i] + ∂w∂y[i])

        # |S̄| = √(2 Sᵢⱼ Sᵢⱼ)
        S_mag = sqrt(2 * (S11^2 + S22^2 + S33^2 + 2*(S12^2 + S13^2 + S23^2)))
        model.strain_magnitude[i] = S_mag

        # νₑ = (Cₛ Δ)² |S̄|
        model.eddy_viscosity[i] = (C_s * Δ)^2 * S_mag
    end

    return model.eddy_viscosity
end

# ============================================================================
# Anisotropic Minimum Dissipation (AMD) Model
# ============================================================================

"""
    AMDModel{T, N}

Anisotropic Minimum Dissipation model (Rozema et al., 2015).

## Mathematical Formulation

The eddy viscosity is:

    νₑ = max(0, νₑ†)

where the predictor is:

    νₑ† = -C (Δₖ² ∂uᵢ/∂xₖ ∂uⱼ/∂xₖ Sᵢⱼ) / (∂uₘ/∂xₙ ∂uₘ/∂xₙ)

Key features:
- Uses **anisotropic filter widths** Δₖ in each direction
- Automatically **switches off** in laminar/transitional regions
- Provides **minimum dissipation** required for subgrid energy transfer
- No explicit filtering or test-filtering required

## Fields

- `C::T`: Poincaré constant (model constant)
- `filter_width::NTuple{N, T}`: Anisotropic filter widths (Δx, Δy, Δz)
- `eddy_viscosity::Array{T, N}`: Cached eddy viscosity field
- `eddy_diffusivity::Array{T, N}`: Cached eddy diffusivity (for scalars)

## Model Constant Recommendations

| Discretization | C |
|----------------|---|
| Spectral methods | 1/12 ≈ 0.0833 |
| 4th-order finite difference | 0.212 |
| 2nd-order finite difference | 0.3 |

## Example

```julia
# Create AMD model for anisotropic grid
model = AMDModel(
    C = 1/12,  # Spectral method
    filter_width = (2π/256, 2π/256, 2π/64),  # Anisotropic
    field_size = (256, 256, 64)
)

# Compute eddy viscosity
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)
```

## References

Rozema, W., Bae, H.J., Moin, P., Verstappen, R. (2015).
"Minimum-dissipation models for large-eddy simulation",
Physics of Fluids 27, 085107.
"""
mutable struct AMDModel{T<:AbstractFloat, N} <: EddyViscosityModel
    C::T                                # Poincaré constant
    filter_width::NTuple{N, T}          # Anisotropic: (Δx, Δy, Δz)
    filter_width_sq::NTuple{N, T}       # (Δx², Δy², Δz²)
    eddy_viscosity::Array{T, N}         # νₑ field
    eddy_diffusivity::Array{T, N}       # κₑ field (for scalars)
    field_size::NTuple{N, Int}
    clip_negative::Bool                 # Whether to clip νₑ < 0
end

"""
    AMDModel(;
        C = 1/12,
        filter_width,
        field_size,
        clip_negative = true,
        dtype = Float64
    )

Create an Anisotropic Minimum Dissipation (AMD) SGS model.

## Arguments

- `C::Real`: Poincaré constant (default: 1/12 for spectral methods)
- `filter_width::NTuple{N, Real}`: Anisotropic grid spacing (Δx, Δy) or (Δx, Δy, Δz)
- `field_size::NTuple{N, Int}`: Grid dimensions
- `clip_negative::Bool`: Clip negative eddy viscosity (default: true)
- `dtype::Type`: Floating point type (default: Float64)
"""
function AMDModel(;
    C::Real = 1/12,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    clip_negative::Bool = true,
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N}

    filter_width_sq = T.(filter_width .^ 2)

    # Allocate arrays
    eddy_viscosity = zeros(T, field_size)
    eddy_diffusivity = zeros(T, field_size)

    AMDModel{T, N}(
        T(C),
        T.(filter_width),
        filter_width_sq,
        eddy_viscosity,
        eddy_diffusivity,
        field_size,
        clip_negative
    )
end

"""
    compute_eddy_viscosity!(model::AMDModel, velocity_gradients...)

Compute AMD eddy viscosity from velocity gradient components.

## 2D Case
```julia
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
```

## 3D Case
```julia
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)
```

The AMD formula uses anisotropic scaling:
    νₑ† = -C (Δₖ² ∂uᵢ/∂xₖ ∂uⱼ/∂xₖ Sᵢⱼ) / (∂uₘ/∂xₙ ∂uₘ/∂xₙ)
"""
function compute_eddy_viscosity!(
    model::AMDModel{T, 2},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}
) where T

    C = model.C
    Δx², Δy² = model.filter_width_sq

    @inbounds for i in eachindex(model.eddy_viscosity)
        # Velocity gradient tensor components
        u_x = ∂u∂x[i]; u_y = ∂u∂y[i]
        v_x = ∂v∂x[i]; v_y = ∂v∂y[i]

        # Strain rate tensor
        S11 = u_x
        S22 = v_y
        S12 = T(0.5) * (u_y + v_x)

        # Denominator: ∂uₘ/∂xₙ ∂uₘ/∂xₙ (trace of gradient tensor squared)
        denom = u_x^2 + u_y^2 + v_x^2 + v_y^2

        # Numerator: -Δₖ² ∂uᵢ/∂xₖ ∂uⱼ/∂xₖ Sᵢⱼ
        # For k=1 (x-direction): Δx² (∂uᵢ/∂x ∂uⱼ/∂x Sᵢⱼ)
        # For k=2 (y-direction): Δy² (∂uᵢ/∂y ∂uⱼ/∂y Sᵢⱼ)

        # k=1: Δx² * (u_x*u_x*S11 + u_x*v_x*S12 + v_x*u_x*S12 + v_x*v_x*S22)
        numer_x = Δx² * (u_x^2 * S11 + 2*u_x*v_x*S12 + v_x^2 * S22)

        # k=2: Δy² * (u_y*u_y*S11 + u_y*v_y*S12 + v_y*u_y*S12 + v_y*v_y*S22)
        numer_y = Δy² * (u_y^2 * S11 + 2*u_y*v_y*S12 + v_y^2 * S22)

        numer = -(numer_x + numer_y)

        # Compute eddy viscosity
        if denom > eps(T)
            νₑ = C * numer / denom
        else
            νₑ = zero(T)
        end

        # Clip negative values if requested
        if model.clip_negative
            νₑ = max(zero(T), νₑ)
        end

        model.eddy_viscosity[i] = νₑ
    end

    return model.eddy_viscosity
end

function compute_eddy_viscosity!(
    model::AMDModel{T, 3},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T}
) where T

    C = model.C
    Δx², Δy², Δz² = model.filter_width_sq

    @inbounds for i in eachindex(model.eddy_viscosity)
        # Velocity gradient tensor components
        u_x = ∂u∂x[i]; u_y = ∂u∂y[i]; u_z = ∂u∂z[i]
        v_x = ∂v∂x[i]; v_y = ∂v∂y[i]; v_z = ∂v∂z[i]
        w_x = ∂w∂x[i]; w_y = ∂w∂y[i]; w_z = ∂w∂z[i]

        # Strain rate tensor
        S11 = u_x
        S22 = v_y
        S33 = w_z
        S12 = T(0.5) * (u_y + v_x)
        S13 = T(0.5) * (u_z + w_x)
        S23 = T(0.5) * (v_z + w_y)

        # Denominator: tr(∇u · ∇uᵀ)
        denom = u_x^2 + u_y^2 + u_z^2 + v_x^2 + v_y^2 + v_z^2 + w_x^2 + w_y^2 + w_z^2

        # Numerator: -Δₖ² ∂uᵢ/∂xₖ ∂uⱼ/∂xₖ Sᵢⱼ
        # This is the key AMD term that uses anisotropic filter widths

        # k=1 (x-direction): Δx² * (∂uᵢ/∂x ∂uⱼ/∂x Sᵢⱼ)
        numer_x = Δx² * (
            u_x^2 * S11 + v_x^2 * S22 + w_x^2 * S33 +
            2 * (u_x*v_x*S12 + u_x*w_x*S13 + v_x*w_x*S23)
        )

        # k=2 (y-direction): Δy² * (∂uᵢ/∂y ∂uⱼ/∂y Sᵢⱼ)
        numer_y = Δy² * (
            u_y^2 * S11 + v_y^2 * S22 + w_y^2 * S33 +
            2 * (u_y*v_y*S12 + u_y*w_y*S13 + v_y*w_y*S23)
        )

        # k=3 (z-direction): Δz² * (∂uᵢ/∂z ∂uⱼ/∂z Sᵢⱼ)
        numer_z = Δz² * (
            u_z^2 * S11 + v_z^2 * S22 + w_z^2 * S33 +
            2 * (u_z*v_z*S12 + u_z*w_z*S13 + v_z*w_z*S23)
        )

        numer = -(numer_x + numer_y + numer_z)

        # Compute eddy viscosity
        if denom > eps(T)
            νₑ = C * numer / denom
        else
            νₑ = zero(T)
        end

        # Clip negative values if requested
        if model.clip_negative
            νₑ = max(zero(T), νₑ)
        end

        model.eddy_viscosity[i] = νₑ
    end

    return model.eddy_viscosity
end

# ============================================================================
# Scalar Eddy Diffusivity (for AMD model)
# ============================================================================

"""
    compute_eddy_diffusivity!(model::AMDModel, velocity_gradients..., scalar_gradients...)

Compute eddy diffusivity for scalar transport using AMD model.

For a scalar field b with gradient ∇b, the eddy diffusivity is:

    κₑ = max(0, κₑ†)

where:
    κₑ† = -C (Δₖ² ∂w/∂xₖ ∂b/∂xₖ) / (∂b/∂xₙ ∂b/∂xₙ)

This is for buoyancy-driven flows where w is vertical velocity.
"""
function compute_eddy_diffusivity!(
    model::AMDModel{T, 3},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T},
    ∂b∂x::AbstractArray{T}, ∂b∂y::AbstractArray{T}, ∂b∂z::AbstractArray{T}
) where T

    C = model.C
    Δx², Δy², Δz² = model.filter_width_sq

    @inbounds for i in eachindex(model.eddy_diffusivity)
        # Scalar gradient magnitude squared
        denom = ∂b∂x[i]^2 + ∂b∂y[i]^2 + ∂b∂z[i]^2

        # Numerator: -Δₖ² ∂w/∂xₖ ∂b/∂xₖ
        numer = -(
            Δx² * ∂w∂x[i] * ∂b∂x[i] +
            Δy² * ∂w∂y[i] * ∂b∂y[i] +
            Δz² * ∂w∂z[i] * ∂b∂z[i]
        )

        # Compute eddy diffusivity
        if denom > eps(T)
            κₑ = C * numer / denom
        else
            κₑ = zero(T)
        end

        # Clip negative values
        if model.clip_negative
            κₑ = max(zero(T), κₑ)
        end

        model.eddy_diffusivity[i] = κₑ
    end

    return model.eddy_diffusivity
end

# ============================================================================
# Subgrid Stress Computation
# ============================================================================

"""
    compute_sgs_stress!(τ, model::EddyViscosityModel, strain_components...)

Compute the deviatoric subgrid stress tensor:

    τᵢⱼᵈ = -2 νₑ S̄ᵢⱼ

## 2D Output
Returns (τ11, τ12, τ22) or modifies pre-allocated arrays.

## 3D Output
Returns (τ11, τ12, τ13, τ22, τ23, τ33) or modifies pre-allocated arrays.
"""
function compute_sgs_stress(
    model::EddyViscosityModel,
    S11::AbstractArray{T}, S12::AbstractArray{T}, S22::AbstractArray{T}
) where T

    νₑ = model.eddy_viscosity

    τ11 = similar(S11)
    τ12 = similar(S12)
    τ22 = similar(S22)

    @inbounds for i in eachindex(νₑ)
        τ11[i] = -2 * νₑ[i] * S11[i]
        τ12[i] = -2 * νₑ[i] * S12[i]
        τ22[i] = -2 * νₑ[i] * S22[i]
    end

    return (τ11, τ12, τ22)
end

function compute_sgs_stress(
    model::EddyViscosityModel,
    S11::AbstractArray{T}, S12::AbstractArray{T}, S13::AbstractArray{T},
    S22::AbstractArray{T}, S23::AbstractArray{T}, S33::AbstractArray{T}
) where T

    νₑ = model.eddy_viscosity

    τ11 = similar(S11); τ12 = similar(S12); τ13 = similar(S13)
    τ22 = similar(S22); τ23 = similar(S23); τ33 = similar(S33)

    @inbounds for i in eachindex(νₑ)
        τ11[i] = -2 * νₑ[i] * S11[i]
        τ12[i] = -2 * νₑ[i] * S12[i]
        τ13[i] = -2 * νₑ[i] * S13[i]
        τ22[i] = -2 * νₑ[i] * S22[i]
        τ23[i] = -2 * νₑ[i] * S23[i]
        τ33[i] = -2 * νₑ[i] * S33[i]
    end

    return (τ11, τ12, τ13, τ22, τ23, τ33)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_eddy_viscosity(model::EddyViscosityModel)

Return the current eddy viscosity field.
"""
get_eddy_viscosity(model::EddyViscosityModel) = model.eddy_viscosity

"""
    get_eddy_diffusivity(model::AMDModel)

Return the current eddy diffusivity field (AMD model only).
"""
get_eddy_diffusivity(model::AMDModel) = model.eddy_diffusivity

"""
    get_filter_width(model::EddyViscosityModel)

Return the filter width(s).
"""
get_filter_width(model::EddyViscosityModel) = model.filter_width

"""
    mean_eddy_viscosity(model::EddyViscosityModel)

Compute the domain-averaged eddy viscosity.
"""
mean_eddy_viscosity(model::EddyViscosityModel) = sum(model.eddy_viscosity) / length(model.eddy_viscosity)

"""
    max_eddy_viscosity(model::EddyViscosityModel)

Return the maximum eddy viscosity in the domain.
"""
max_eddy_viscosity(model::EddyViscosityModel) = maximum(model.eddy_viscosity)

"""
    reset!(model::EddyViscosityModel)

Reset the eddy viscosity field to zero.
"""
function reset!(model::EddyViscosityModel)
    fill!(model.eddy_viscosity, zero(eltype(model.eddy_viscosity)))
    return model
end

function reset!(model::AMDModel)
    fill!(model.eddy_viscosity, zero(eltype(model.eddy_viscosity)))
    fill!(model.eddy_diffusivity, zero(eltype(model.eddy_diffusivity)))
    return model
end

"""
    set_constant!(model::SmagorinskyModel, C_s::Real)

Update the Smagorinsky constant.
"""
function set_constant!(model::SmagorinskyModel{T}, C_s::Real) where T
    model.C_s = T(C_s)
    return model
end

"""
    set_constant!(model::AMDModel, C::Real)

Update the AMD Poincaré constant.
"""
function set_constant!(model::AMDModel{T}, C::Real) where T
    model.C = T(C)
    return model
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray)

Compute the subgrid-scale dissipation rate:

    εₛₛ = 2 νₑ |S̄|²

Returns the dissipation field.
"""
function sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray{T}) where T
    νₑ = model.eddy_viscosity
    ε_sgs = similar(νₑ)

    @inbounds for i in eachindex(ε_sgs)
        ε_sgs[i] = 2 * νₑ[i] * strain_magnitude[i]^2
    end

    return ε_sgs
end

"""
    mean_sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray)

Compute domain-averaged SGS dissipation rate.
"""
function mean_sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray{T}) where T
    νₑ = model.eddy_viscosity
    total = zero(T)

    @inbounds for i in eachindex(νₑ)
        total += 2 * νₑ[i] * strain_magnitude[i]^2
    end

    return total / length(νₑ)
end

# ============================================================================
# Exports
# ============================================================================

export SGSModel, EddyViscosityModel
export SmagorinskyModel, AMDModel
export compute_eddy_viscosity!, compute_eddy_diffusivity!
export compute_sgs_stress
export get_eddy_viscosity, get_eddy_diffusivity, get_filter_width
export mean_eddy_viscosity, max_eddy_viscosity
export reset!, set_constant!
export sgs_dissipation, mean_sgs_dissipation
