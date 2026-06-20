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

## GPU Support

Both models support GPU acceleration through the architecture abstraction.
When created with `architecture = GPU()`, all internal arrays are allocated
on the GPU and computations use GPU-optimized broadcasting.

## References

1. Smagorinsky, J. (1963). "General circulation experiments with the primitive equations"
2. Rozema, W., Bae, H.J., Moin, P., Verstappen, R. (2015). "Minimum-dissipation models
   for large-eddy simulation", Physics of Fluids 27, 085107.
3. Abkar, M., Bae, H.J., Moin, P. (2016). "Minimum-dissipation scalar transport model"
"""

# LinearAlgebra already in Tarang.jl

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
# Validation Helper
# ============================================================================

"""
    _validate_gradient_sizes(expected_size, arrays...)

Validate that all input gradient arrays match the expected field size.
Uses @boundscheck so it can be disabled with @inbounds for performance.
"""
@inline function _validate_gradient_sizes(expected_size::NTuple{N, Int}, arrays...) where N
    @boundscheck begin
        for (i, arr) in enumerate(arrays)
            if size(arr) != expected_size
                throw(DimensionMismatch(
                    "Gradient array $i has size $(size(arr)), expected $expected_size"
                ))
            end
        end
    end
    return nothing
end

"""
    _coerce_arrays_to_architecture(arch, arrays...)

Ensure all gradient arrays live on the target architecture.
CPU models can accept GPU inputs (and vice versa) without scalar indexing,
which is especially important when CUDA scalar indexing is disallowed.
"""
function _coerce_arrays_to_architecture(arch::AbstractArchitecture, arrays::AbstractArray...)
    return tuple((_ensure_array_on_architecture(arch, arr) for arr in arrays)...)
end

@inline function _ensure_array_on_architecture(arch::AbstractArchitecture, arr::AbstractArray)
    if is_gpu(arch)
        return is_gpu_array(arr) ? arr : _move_array_to_gpu(arch, arr)
    else
        return is_gpu_array(arr) ? on_architecture(arch, arr) : arr
    end
end

@inline function _move_array_to_gpu(arch::AbstractArchitecture, arr::AbstractArray)
    if arr isa Array
        return on_architecture(arch, arr)
    else
        return on_architecture(arch, Array(arr))
    end
end

# ============================================================================
# Smagorinsky Model
# ============================================================================

"""
    SmagorinskyModel{T, N, A, Arch}

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
- `eddy_viscosity::A`: Cached eddy viscosity field (Array or CuArray)
- `strain_magnitude::A`: Cached |S̄| field
- `architecture::Arch`: CPU() or GPU() architecture

## Example

```julia
# Create model for 256³ domain with Δ = 2π/256
model = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (2π/256, 2π/256, 2π/256),
    field_size = (256, 256, 256)
)

# Create GPU model
model_gpu = SmagorinskyModel(
    C_s = 0.17,
    filter_width = (2π/256, 2π/256, 2π/256),
    field_size = (256, 256, 256),
    architecture = GPU()
)

# Compute eddy viscosity from velocity gradients
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)

# Access the result
νₑ = get_eddy_viscosity(model)
```
"""
mutable struct SmagorinskyModel{T<:AbstractFloat, N, A<:AbstractArray{T, N}, Arch<:AbstractArchitecture} <: EddyViscosityModel
    C_s::T                              # Smagorinsky constant
    filter_width::NTuple{N, T}          # (Δx, Δy, Δz)
    effective_delta::T                  # Effective filter width Δ = (Δx Δy Δz)^(1/3)
    eddy_viscosity::A                   # νₑ field (Array or CuArray)
    strain_magnitude::A                 # |S̄| field
    field_size::NTuple{N, Int}
    architecture::Arch                  # CPU() or GPU()
end

"""
    SmagorinskyModel(;
        C_s = 0.17,
        filter_width,
        field_size,
        dtype = Float64,
        architecture = CPU()
    )

Create a Smagorinsky SGS model.

## Arguments

- `C_s::Real`: Smagorinsky constant (default: 0.17, suitable for isotropic turbulence)
- `filter_width::NTuple{N, Real}`: Grid spacing (Δx, Δy) or (Δx, Δy, Δz)
- `field_size::NTuple{N, Int}`: Grid dimensions
- `dtype::Type`: Floating point type (default: Float64)
- `architecture::AbstractArchitecture`: CPU() or GPU() (default: CPU())

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
    dtype::Type{T} = Float64,
    architecture::Arch = CPU()
) where {T<:AbstractFloat, N, Arch<:AbstractArchitecture}

    # Effective filter width: geometric mean
    effective_delta = T(prod(filter_width)^(1/N))

    # Allocate arrays on the appropriate architecture
    eddy_viscosity = zeros(architecture, T, field_size...)
    strain_magnitude = zeros(architecture, T, field_size...)

    A = typeof(eddy_viscosity)

    SmagorinskyModel{T, N, A, Arch}(
        T(C_s),
        T.(filter_width),
        effective_delta,
        eddy_viscosity,
        strain_magnitude,
        field_size,
        architecture
    )
end

"""
    compute_eddy_viscosity!(model::SmagorinskyModel, velocity_gradients...)

Compute eddy viscosity from velocity gradient components.

GPU-aware: Uses broadcasting for GPU arrays, optimized SIMD loops for CPU.

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
    model::SmagorinskyModel{T, 2, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}
) where {T, A, Arch}

    # Validate input array sizes
    _validate_gradient_sizes(model.field_size, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) =
        _coerce_arrays_to_architecture(model.architecture, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    # Pre-compute constant factor
    CΔ_sq = (model.C_s * model.effective_delta)^2
    half = T(0.5)
    two = T(2)

    strain_mag = model.strain_magnitude
    eddy_visc = model.eddy_viscosity

    # Check if on GPU - use broadcasting for GPU arrays
    if is_gpu(model.architecture)
        # GPU path: use broadcasting (CUDA.jl handles optimization)
        # S12 = 0.5 * (∂u∂y + ∂v∂x)
        # |S̄| = √(2 * (S11² + S22² + 2*S12²))
        S12_tmp = half .* (∂u∂y .+ ∂v∂x)
        strain_mag .= sqrt.(two .* (∂u∂x.^2 .+ ∂v∂y.^2 .+ two .* S12_tmp.^2))
        eddy_visc .= CΔ_sq .* strain_mag
    else
        # CPU path: use optimized SIMD loops
        @inbounds @simd for i in eachindex(strain_mag)
            # Strain rate tensor components
            S11 = ∂u∂x[i]
            S22 = ∂v∂y[i]
            S12 = half * (∂u∂y[i] + ∂v∂x[i])

            # |S̄| = √(2 Sᵢⱼ Sᵢⱼ)
            S_mag = sqrt(two * (S11^2 + S22^2 + two*S12^2))
            strain_mag[i] = S_mag

            # νₑ = (Cₛ Δ)² |S̄|
            eddy_visc[i] = CΔ_sq * S_mag
        end
    end

    return eddy_visc
end

function compute_eddy_viscosity!(
    model::SmagorinskyModel{T, 3, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T}
) where {T, A, Arch}

    # Validate input array sizes
    _validate_gradient_sizes(model.field_size, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)

    (∂u∂x, ∂u∂y, ∂u∂z,
     ∂v∂x, ∂v∂y, ∂v∂z,
     ∂w∂x, ∂w∂y, ∂w∂z) =
        _coerce_arrays_to_architecture(model.architecture,
                                      ∂u∂x, ∂u∂y, ∂u∂z,
                                      ∂v∂x, ∂v∂y, ∂v∂z,
                                      ∂w∂x, ∂w∂y, ∂w∂z)

    # Pre-compute constant factor
    CΔ_sq = (model.C_s * model.effective_delta)^2
    half = T(0.5)
    two = T(2)

    strain_mag = model.strain_magnitude
    eddy_visc = model.eddy_viscosity

    # Check if on GPU - use broadcasting for GPU arrays
    if is_gpu(model.architecture)
        # GPU path: use broadcasting (CUDA.jl handles optimization)
        S12_tmp = half .* (∂u∂y .+ ∂v∂x)
        S13_tmp = half .* (∂u∂z .+ ∂w∂x)
        S23_tmp = half .* (∂v∂z .+ ∂w∂y)
        strain_mag .= sqrt.(two .* (∂u∂x.^2 .+ ∂v∂y.^2 .+ ∂w∂z.^2 .+
                                     two .* (S12_tmp.^2 .+ S13_tmp.^2 .+ S23_tmp.^2)))
        eddy_visc .= CΔ_sq .* strain_mag
    else
        # CPU path: use optimized SIMD loops
        @inbounds @simd for i in eachindex(strain_mag)
            # Strain rate tensor components
            S11 = ∂u∂x[i]
            S22 = ∂v∂y[i]
            S33 = ∂w∂z[i]
            S12 = half * (∂u∂y[i] + ∂v∂x[i])
            S13 = half * (∂u∂z[i] + ∂w∂x[i])
            S23 = half * (∂v∂z[i] + ∂w∂y[i])

            # |S̄| = √(2 Sᵢⱼ Sᵢⱼ)
            S_mag = sqrt(two * (S11^2 + S22^2 + S33^2 + two*(S12^2 + S13^2 + S23^2)))
            strain_mag[i] = S_mag

            # νₑ = (Cₛ Δ)² |S̄|
            eddy_visc[i] = CΔ_sq * S_mag
        end
    end

    return eddy_visc
end

# ============================================================================
# Anisotropic Minimum Dissipation (AMD) Model
# ============================================================================

"""
    AMDModel{T, N, A, Arch}

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
- `eddy_viscosity::A`: Cached eddy viscosity field (Array or CuArray)
- `eddy_diffusivity::A`: Cached eddy diffusivity (for scalars)
- `architecture::Arch`: CPU() or GPU() architecture

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

# Create GPU AMD model
model_gpu = AMDModel(
    C = 1/12,
    filter_width = (2π/256, 2π/256, 2π/64),
    field_size = (256, 256, 64),
    architecture = GPU()
)

# Compute eddy viscosity
compute_eddy_viscosity!(model, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)
```

## References

Rozema, W., Bae, H.J., Moin, P., Verstappen, R. (2015).
"Minimum-dissipation models for large-eddy simulation",
Physics of Fluids 27, 085107.
"""
mutable struct AMDModel{T<:AbstractFloat, N, A<:AbstractArray{T, N}, Arch<:AbstractArchitecture} <: EddyViscosityModel
    C::T                                # Poincaré constant
    filter_width::NTuple{N, T}          # Anisotropic: (Δx, Δy, Δz)
    filter_width_sq::NTuple{N, T}       # (Δx², Δy², Δz²)
    eddy_viscosity::A                   # νₑ field (Array or CuArray)
    eddy_diffusivity::A                 # κₑ field (for scalars)
    field_size::NTuple{N, Int}
    clip_negative::Bool                 # Whether to clip νₑ < 0
    architecture::Arch                  # CPU() or GPU()
end

"""
    AMDModel(;
        C = 1/12,
        filter_width,
        field_size,
        clip_negative = true,
        dtype = Float64,
        architecture = CPU()
    )

Create an Anisotropic Minimum Dissipation (AMD) SGS model.

## Arguments

- `C::Real`: Poincaré constant (default: 1/12 for spectral methods)
- `filter_width::NTuple{N, Real}`: Anisotropic grid spacing (Δx, Δy) or (Δx, Δy, Δz)
- `field_size::NTuple{N, Int}`: Grid dimensions
- `clip_negative::Bool`: Clip negative eddy viscosity (default: true)
- `dtype::Type`: Floating point type (default: Float64)
- `architecture::AbstractArchitecture`: CPU() or GPU() (default: CPU())
"""
function AMDModel(;
    C::Real = 1/12,
    filter_width::NTuple{N, Real},
    field_size::NTuple{N, Int},
    clip_negative::Bool = true,
    dtype::Type{T} = Float64,
    architecture::Arch = CPU()
) where {T<:AbstractFloat, N, Arch<:AbstractArchitecture}

    filter_width_sq = T.(filter_width .^ 2)

    # Allocate arrays on the appropriate architecture
    eddy_viscosity = zeros(architecture, T, field_size...)
    eddy_diffusivity = zeros(architecture, T, field_size...)

    A = typeof(eddy_viscosity)

    AMDModel{T, N, A, Arch}(
        T(C),
        T.(filter_width),
        filter_width_sq,
        eddy_viscosity,
        eddy_diffusivity,
        field_size,
        clip_negative,
        architecture
    )
end

"""
    compute_eddy_viscosity!(model::AMDModel, velocity_gradients...)

Compute AMD eddy viscosity from velocity gradient components.

GPU-aware: Uses broadcasting for GPU arrays, optimized SIMD loops for CPU.

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
    model::AMDModel{T, 2, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}
) where {T, A, Arch}

    # Validate input array sizes
    _validate_gradient_sizes(model.field_size, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) =
        _coerce_arrays_to_architecture(model.architecture, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    C = model.C
    Δx², Δy² = model.filter_width_sq
    half = T(0.5)
    two = T(2)
    eps_T = T(100) * eps(T)
    clip = model.clip_negative
    eddy_visc = model.eddy_viscosity

    # Check if on GPU - use broadcasting for GPU arrays
    if is_gpu(model.architecture)
        # GPU path: use broadcasting
        S12 = half .* (∂u∂y .+ ∂v∂x)
        denom = ∂u∂x.^2 .+ ∂u∂y.^2 .+ ∂v∂x.^2 .+ ∂v∂y.^2
        numer_x = Δx² .* (∂u∂x.^2 .* ∂u∂x .+ two .* ∂u∂x .* ∂v∂x .* S12 .+ ∂v∂x.^2 .* ∂v∂y)
        numer_y = Δy² .* (∂u∂y.^2 .* ∂u∂x .+ two .* ∂u∂y .* ∂v∂y .* S12 .+ ∂v∂y.^2 .* ∂v∂y)
        numer = .-(numer_x .+ numer_y)
        # Safe division and clipping
        eddy_visc .= ifelse.(denom .> eps_T, C .* numer ./ denom, zero(T))
        if clip
            eddy_visc .= max.(zero(T), eddy_visc)
        end
    else
        # CPU path: use optimized SIMD loops
        @inbounds @simd for i in eachindex(eddy_visc)
            u_x = ∂u∂x[i]; u_y = ∂u∂y[i]
            v_x = ∂v∂x[i]; v_y = ∂v∂y[i]
            S11 = u_x
            S22 = v_y
            S12 = half * (u_y + v_x)
            denom = u_x^2 + u_y^2 + v_x^2 + v_y^2
            numer_x = Δx² * (u_x^2 * S11 + two*u_x*v_x*S12 + v_x^2 * S22)
            numer_y = Δy² * (u_y^2 * S11 + two*u_y*v_y*S12 + v_y^2 * S22)
            numer = -(numer_x + numer_y)
            νₑ = denom > eps_T ? C * numer / denom : zero(T)
            νₑ = ifelse(clip, max(zero(T), νₑ), νₑ)
            eddy_visc[i] = νₑ
        end
    end

    return eddy_visc
end

function compute_eddy_viscosity!(
    model::AMDModel{T, 3, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T}
) where {T, A, Arch}

    # Validate input array sizes
    _validate_gradient_sizes(model.field_size, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)

    (∂u∂x, ∂u∂y, ∂u∂z,
     ∂v∂x, ∂v∂y, ∂v∂z,
     ∂w∂x, ∂w∂y, ∂w∂z) =
        _coerce_arrays_to_architecture(model.architecture,
                                      ∂u∂x, ∂u∂y, ∂u∂z,
                                      ∂v∂x, ∂v∂y, ∂v∂z,
                                      ∂w∂x, ∂w∂y, ∂w∂z)

    C = model.C
    Δx², Δy², Δz² = model.filter_width_sq
    half = T(0.5)
    two = T(2)
    eps_T = T(100) * eps(T)
    clip = model.clip_negative
    eddy_visc = model.eddy_viscosity

    # Check if on GPU - use broadcasting for GPU arrays
    if is_gpu(model.architecture)
        # GPU path: use broadcasting
        S12 = half .* (∂u∂y .+ ∂v∂x)
        S13 = half .* (∂u∂z .+ ∂w∂x)
        S23 = half .* (∂v∂z .+ ∂w∂y)

        denom = ∂u∂x.^2 .+ ∂u∂y.^2 .+ ∂u∂z.^2 .+ ∂v∂x.^2 .+ ∂v∂y.^2 .+ ∂v∂z.^2 .+ ∂w∂x.^2 .+ ∂w∂y.^2 .+ ∂w∂z.^2

        numer_x = Δx² .* (∂u∂x.^2 .* ∂u∂x .+ ∂v∂x.^2 .* ∂v∂y .+ ∂w∂x.^2 .* ∂w∂z .+
                          two .* (∂u∂x .* ∂v∂x .* S12 .+ ∂u∂x .* ∂w∂x .* S13 .+ ∂v∂x .* ∂w∂x .* S23))
        numer_y = Δy² .* (∂u∂y.^2 .* ∂u∂x .+ ∂v∂y.^2 .* ∂v∂y .+ ∂w∂y.^2 .* ∂w∂z .+
                          two .* (∂u∂y .* ∂v∂y .* S12 .+ ∂u∂y .* ∂w∂y .* S13 .+ ∂v∂y .* ∂w∂y .* S23))
        numer_z = Δz² .* (∂u∂z.^2 .* ∂u∂x .+ ∂v∂z.^2 .* ∂v∂y .+ ∂w∂z.^2 .* ∂w∂z .+
                          two .* (∂u∂z .* ∂v∂z .* S12 .+ ∂u∂z .* ∂w∂z .* S13 .+ ∂v∂z .* ∂w∂z .* S23))

        numer = .-(numer_x .+ numer_y .+ numer_z)
        eddy_visc .= ifelse.(denom .> eps_T, C .* numer ./ denom, zero(T))
        if clip
            eddy_visc .= max.(zero(T), eddy_visc)
        end
    else
        # CPU path: use optimized SIMD loops
        @inbounds @simd for i in eachindex(eddy_visc)
            u_x = ∂u∂x[i]; u_y = ∂u∂y[i]; u_z = ∂u∂z[i]
            v_x = ∂v∂x[i]; v_y = ∂v∂y[i]; v_z = ∂v∂z[i]
            w_x = ∂w∂x[i]; w_y = ∂w∂y[i]; w_z = ∂w∂z[i]
            S11 = u_x
            S22 = v_y
            S33 = w_z
            S12 = half * (u_y + v_x)
            S13 = half * (u_z + w_x)
            S23 = half * (v_z + w_y)
            denom = u_x^2 + u_y^2 + u_z^2 + v_x^2 + v_y^2 + v_z^2 + w_x^2 + w_y^2 + w_z^2
            numer_x = Δx² * (u_x^2 * S11 + v_x^2 * S22 + w_x^2 * S33 + two * (u_x*v_x*S12 + u_x*w_x*S13 + v_x*w_x*S23))
            numer_y = Δy² * (u_y^2 * S11 + v_y^2 * S22 + w_y^2 * S33 + two * (u_y*v_y*S12 + u_y*w_y*S13 + v_y*w_y*S23))
            numer_z = Δz² * (u_z^2 * S11 + v_z^2 * S22 + w_z^2 * S33 + two * (u_z*v_z*S12 + u_z*w_z*S13 + v_z*w_z*S23))
            numer = -(numer_x + numer_y + numer_z)
            νₑ = denom > eps_T ? C * numer / denom : zero(T)
            νₑ = ifelse(clip, max(zero(T), νₑ), νₑ)
            eddy_visc[i] = νₑ
        end
    end

    return eddy_visc
end

# ============================================================================
# Scalar Eddy Diffusivity (for AMD model)
# ============================================================================

"""
    compute_eddy_diffusivity!(model::AMDModel, velocity_gradients..., scalar_gradients...)

Compute eddy diffusivity for scalar transport using AMD model.

GPU-aware: Uses broadcasting for GPU arrays, optimized SIMD loops for CPU.

For a scalar field b with gradient ∇b, the AMD eddy diffusivity
(Abkar, Bae & Moin 2016, eq. 2.7) is the FULL double contraction over the
scaled-gradient direction k AND all velocity components i:

    κₑ = max(0, κₑ†),   κₑ† = -C · [ Σₖ δₖ² (∂ₖ uᵢ)(∂ₖ b)(∂ᵢ b) ] / [ (∂ₗ b)(∂ₗ b) ]

i.e. for each direction k form the inner sum Σᵢ (∂ₖ uᵢ)(∂ᵢ b) over ALL velocity
components uᵢ, weight by δₖ²(∂ₖ b), and sum over k. The method therefore needs
every velocity-gradient component ∂uᵢ/∂xₖ (2D: 4 of them; 3D: 9), passed in
component-major order, followed by the scalar gradients ∂b/∂xₖ.
(An earlier version summed only a single velocity component, contracting the
scaled velocity gradient with the SAME scalar-gradient direction twice — that is
NOT the AMD diffusivity and is fixed here.)
"""
function compute_eddy_diffusivity!(
    model::AMDModel{T, 2, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T},
    ∂b∂x::AbstractArray{T}, ∂b∂y::AbstractArray{T}
) where {T, A, Arch}

    # Validate input array sizes
    _validate_gradient_sizes(model.field_size, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y)

    (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y) =
        _coerce_arrays_to_architecture(model.architecture,
                                       ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y)

    C = model.C
    Δx², Δy² = model.filter_width_sq
    eps_T = T(100) * eps(T)
    clip = model.clip_negative
    eddy_diff = model.eddy_diffusivity

    # κₑ† = -[ Δx²(∂ₓb)Σᵢ(∂ₓuᵢ)(∂ᵢb) + Δy²(∂_yb)Σᵢ(∂_yuᵢ)(∂ᵢb) ] / |∇b|²
    if is_gpu(model.architecture)
        # GPU path: use broadcasting
        denom = ∂b∂x.^2 .+ ∂b∂y.^2
        numer = .-(Δx² .* ∂b∂x .* (∂u∂x .* ∂b∂x .+ ∂v∂x .* ∂b∂y) .+
                   Δy² .* ∂b∂y .* (∂u∂y .* ∂b∂x .+ ∂v∂y .* ∂b∂y))
        eddy_diff .= ifelse.(denom .> eps_T, C .* numer ./ denom, zero(T))
        if clip
            eddy_diff .= max.(zero(T), eddy_diff)
        end
    else
        # CPU path: use optimized SIMD loops
        @inbounds @simd for i in eachindex(eddy_diff)
            ax = ∂b∂x[i]; ay = ∂b∂y[i]
            denom = ax^2 + ay^2
            numer = -(Δx² * ax * (∂u∂x[i] * ax + ∂v∂x[i] * ay) +
                      Δy² * ay * (∂u∂y[i] * ax + ∂v∂y[i] * ay))
            κₑ = denom > eps_T ? C * numer / denom : zero(T)
            κₑ = ifelse(clip, max(zero(T), κₑ), κₑ)
            eddy_diff[i] = κₑ
        end
    end

    return eddy_diff
end

function compute_eddy_diffusivity!(
    model::AMDModel{T, 3, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T},
    ∂b∂x::AbstractArray{T}, ∂b∂y::AbstractArray{T}, ∂b∂z::AbstractArray{T}
) where {T, A, Arch}

    # Validate input array sizes
    _validate_gradient_sizes(model.field_size,
                             ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z,
                             ∂w∂x, ∂w∂y, ∂w∂z, ∂b∂x, ∂b∂y, ∂b∂z)

    (∂u∂x, ∂u∂y, ∂u∂z,
     ∂v∂x, ∂v∂y, ∂v∂z,
     ∂w∂x, ∂w∂y, ∂w∂z,
     ∂b∂x, ∂b∂y, ∂b∂z) =
        _coerce_arrays_to_architecture(model.architecture,
                                      ∂u∂x, ∂u∂y, ∂u∂z,
                                      ∂v∂x, ∂v∂y, ∂v∂z,
                                      ∂w∂x, ∂w∂y, ∂w∂z,
                                      ∂b∂x, ∂b∂y, ∂b∂z)

    C = model.C
    Δx², Δy², Δz² = model.filter_width_sq
    eps_T = T(100) * eps(T)
    clip = model.clip_negative
    eddy_diff = model.eddy_diffusivity

    # κₑ† = -[ Σₖ δₖ²(∂ₖb) Σᵢ(∂ₖuᵢ)(∂ᵢb) ] / |∇b|², k,i ∈ {x,y,z}, u=(u,v,w)
    if is_gpu(model.architecture)
        # GPU path: use broadcasting
        denom = ∂b∂x.^2 .+ ∂b∂y.^2 .+ ∂b∂z.^2
        numer = .-(Δx² .* ∂b∂x .* (∂u∂x .* ∂b∂x .+ ∂v∂x .* ∂b∂y .+ ∂w∂x .* ∂b∂z) .+
                   Δy² .* ∂b∂y .* (∂u∂y .* ∂b∂x .+ ∂v∂y .* ∂b∂y .+ ∂w∂y .* ∂b∂z) .+
                   Δz² .* ∂b∂z .* (∂u∂z .* ∂b∂x .+ ∂v∂z .* ∂b∂y .+ ∂w∂z .* ∂b∂z))
        eddy_diff .= ifelse.(denom .> eps_T, C .* numer ./ denom, zero(T))
        if clip
            eddy_diff .= max.(zero(T), eddy_diff)
        end
    else
        # CPU path: use optimized SIMD loops
        @inbounds @simd for i in eachindex(eddy_diff)
            ax = ∂b∂x[i]; ay = ∂b∂y[i]; az = ∂b∂z[i]
            denom = ax^2 + ay^2 + az^2
            numer = -(Δx² * ax * (∂u∂x[i] * ax + ∂v∂x[i] * ay + ∂w∂x[i] * az) +
                      Δy² * ay * (∂u∂y[i] * ax + ∂v∂y[i] * ay + ∂w∂y[i] * az) +
                      Δz² * az * (∂u∂z[i] * ax + ∂v∂z[i] * ay + ∂w∂z[i] * az))
            κₑ = denom > eps_T ? C * numer / denom : zero(T)
            κₑ = ifelse(clip, max(zero(T), κₑ), κₑ)
            eddy_diff[i] = κₑ
        end
    end

    return eddy_diff
end

# ============================================================================
# Subgrid Stress Computation
# ============================================================================

"""
    compute_sgs_stress(model::EddyViscosityModel, strain_components...)

Compute the deviatoric subgrid stress tensor:

    τᵢⱼᵈ = -2 νₑ S̄ᵢⱼ

GPU-aware: Uses broadcasting which works for both CPU and GPU arrays.

## 2D Output
Returns (τ11, τ12, τ22).

## 3D Output
Returns (τ11, τ12, τ13, τ22, τ23, τ33).
"""
function compute_sgs_stress(
    model::EddyViscosityModel,
    S11::AbstractArray{T}, S12::AbstractArray{T}, S22::AbstractArray{T}
) where T

    νₑ = model.eddy_viscosity
    neg_two = T(-2)

    # Use broadcasting - works for both CPU and GPU arrays
    # similar() preserves array type (CuArray for GPU)
    τ11 = neg_two .* νₑ .* S11
    τ12 = neg_two .* νₑ .* S12
    τ22 = neg_two .* νₑ .* S22

    return (τ11, τ12, τ22)
end

function compute_sgs_stress(
    model::EddyViscosityModel,
    S11::AbstractArray{T}, S12::AbstractArray{T}, S13::AbstractArray{T},
    S22::AbstractArray{T}, S23::AbstractArray{T}, S33::AbstractArray{T}
) where T

    νₑ = model.eddy_viscosity
    neg_two = T(-2)

    # Use broadcasting - works for both CPU and GPU arrays
    τ11 = neg_two .* νₑ .* S11
    τ12 = neg_two .* νₑ .* S12
    τ13 = neg_two .* νₑ .* S13
    τ22 = neg_two .* νₑ .* S22
    τ23 = neg_two .* νₑ .* S23
    τ33 = neg_two .* νₑ .* S33

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
function mean_eddy_viscosity(model::EddyViscosityModel)
    n = length(model.eddy_viscosity)
    n == 0 && return zero(eltype(model.eddy_viscosity))
    return sum(model.eddy_viscosity) / n
end

"""
    max_eddy_viscosity(model::EddyViscosityModel)

Return the maximum eddy viscosity in the domain.
"""
max_eddy_viscosity(model::EddyViscosityModel) = maximum(model.eddy_viscosity)

"""
    reset!(model::EddyViscosityModel)

Reset the eddy viscosity field to zero.
GPU-aware: fill!() works for both CPU and GPU arrays.
"""
function reset!(model::EddyViscosityModel)
    fill!(model.eddy_viscosity, zero(eltype(model.eddy_viscosity)))
    return model
end

function reset!(model::AMDModel{T, N, A, Arch}) where {T, N, A, Arch}
    fill!(model.eddy_viscosity, zero(T))
    fill!(model.eddy_diffusivity, zero(T))
    return model
end

"""
    set_constant!(model::SmagorinskyModel, C_s::Real)

Update the Smagorinsky constant.
"""
function set_constant!(model::SmagorinskyModel{T, N, A, Arch}, C_s::Real) where {T, N, A, Arch}
    model.C_s = T(C_s)
    return model
end

"""
    set_constant!(model::AMDModel, C::Real)

Update the AMD Poincaré constant.
"""
function set_constant!(model::AMDModel{T, N, A, Arch}, C::Real) where {T, N, A, Arch}
    model.C = T(C)
    return model
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray)

Compute the subgrid-scale dissipation rate:

    εₛₛ = νₑ |S̄|²

where `|S̄| = √(2 S̄ᵢⱼS̄ᵢⱼ)` (the convention used by `compute_eddy_viscosity!`).
The exact dissipation is εₛₛ = -τᵢⱼ S̄ᵢⱼ = 2 νₑ S̄ᵢⱼS̄ᵢⱼ = νₑ |S̄|² with that `|S̄|`;
an extra factor of 2 here would double-count (the strain magnitude already carries it).

GPU-aware: Uses broadcasting which works for both CPU and GPU arrays.
Returns the dissipation field.
"""
function sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray{T}) where T
    νₑ = model.eddy_viscosity
    # Use broadcasting - works for both CPU and GPU arrays
    return νₑ .* strain_magnitude.^2
end

"""
    mean_sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray)

Compute domain-averaged SGS dissipation rate.
GPU-aware: Uses broadcasting and sum() which work for both CPU and GPU arrays.
"""
function mean_sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray{T}) where T
    νₑ = model.eddy_viscosity
    n = length(νₑ)
    n == 0 && return zero(T)
    # Use broadcasting and sum - works for both CPU and GPU.
    # εₛₛ = νₑ |S̄|² with |S̄| = √(2 S̄ᵢⱼS̄ᵢⱼ); no extra factor of 2 (see sgs_dissipation).
    return sum(νₑ .* strain_magnitude.^2) / n
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
