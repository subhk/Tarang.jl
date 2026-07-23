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
    _validate_gradient_arrays(reference, arrays...)

Validate every input gradient array against `reference` — the model's own output
array, which is what the kernels actually iterate.

This check is deliberately NOT wrapped in `@boundscheck`. It used to be, and that
made it vanish under `--check-bounds=no` (a plausible flag for a production LES
run) while the kernels still ran `@inbounds`: an undersized gradient array was
then read past its end, which segfaults for a large mismatch and silently returns
values read from unowned memory for a small one. One predictable branch per call
is nothing against the O(N) work that follows.

`reference` is the model's array rather than `model.field_size`, so a mutated
`field_size` cannot desync the validated shape from the iterated one.
"""
function _validate_gradient_arrays(reference::AbstractArray, arrays...)
    expected_size = size(reference)
    for (i, arr) in enumerate(arrays)
        if size(arr) != expected_size
            throw(DimensionMismatch(
                "Gradient array $i has size $(size(arr)), expected $expected_size"
            ))
        end
        _reject_nonlocal_array(i, arr)
    end
    return nothing
end

"""
    _reject_nonlocal_array(i, arr)

Reject array types whose element order does not match the model's own array.

The kernels pair cells positionally against a rank-local dense array. A
`PencilArray` reports the same `size` but stores its data in (possibly permuted)
parent order, so mixing one in passes a size check and then silently pairs the
wrong cells — measured at 75% of cells mispaired for a `Permutation(3,2,1)`
pencil. Fail loudly with the fix instead.
"""
@inline function _reject_nonlocal_array(i::Int, arr::AbstractArray)
    if arr isa PencilArrays.PencilArray
        throw(ArgumentError(
            "Gradient array $i is a PencilArray. LES models work on rank-local dense " *
            "arrays and pair cells positionally, which does not match a PencilArray's " *
            "storage order. Pass `get_local_data(field)` (or `parent(array)`) instead."
        ))
    end
    return nothing
end

"""
    _safe_quotient(C, numer, denom)

Return `C * numer / denom`, guarding only the genuine `0/0`.

`denom` is `|∇u|²` (or `|∇b|²`) — a DIMENSIONAL quantity. The previous guard
compared it against an absolute `100*eps(T)`, which made the result depend on the
caller's choice of units and dtype and broke the model's exact invariances: κₑ is
mathematically unchanged by `b → αb`, yet a Float64 scalar scaled by 1e-8 (a trace
species in mixing-ratio units) returned identically zero, and in Float32 an
ordinary weakly-turbulent field had 89% of its cells silently zeroed. No guard
that large is needed: `numer` is `O(denom^1.5)`, so the quotient stays finite for
every `denom > 0` down to the smallest subnormal.

NaN propagates deliberately. Returning zero for a blown-up velocity field would
hide the blow-up at the one place a solver would naturally notice it.
"""
@inline function _safe_quotient(C::T, numer::T, denom::T) where {T}
    isnan(denom) && return T(NaN)
    return denom > zero(T) ? C * numer / denom : zero(T)
end

"""
    _apply_clip(clip, value)

Clip a negative eddy-viscosity/diffusivity predictor to zero when requested.
`max` propagates NaN, so a NaN predictor survives clipping (see `_safe_quotient`).
"""
@inline _apply_clip(clip::Bool, value::T) where {T} = clip ? max(zero(T), value) : value

"""
    _effective_delta(filter_width)

Geometric-mean filter width `(Δ₁ Δ₂ … Δ_N)^(1/N)`, derived on demand so a mutated
`filter_width` can never disagree with it.
"""
# Signature requires at least one element: `NTuple{N,T}` also matches the empty
# tuple, which leaves `T` unbound (Aqua flags it, and `prod(())^(1/0)` is
# meaningless anyway). `N` comes from the tuple length, which is static.
@inline function _effective_delta(filter_width::Tuple{T, Vararg{T}}) where {T}
    return T(prod(filter_width)^(1 / length(filter_width)))
end

"""
    _validate_model_params(constant_name, constant, filter_width, field_size)

Shared constructor validation for the SGS models. Previously absent, which let
`filter_width = (-1,-1,-1)` construct an AMD model silently (the sign vanished in
the squaring) while raising `DomainError` from the geometric mean in Smagorinsky,
and let `C = -5` (anti-dissipative), zero widths, and empty grids through.
"""
function _validate_model_params(constant_name::Symbol, constant::Real,
                                filter_width::NTuple{N, Real},
                                field_size::NTuple{N, Int}) where {N}
    if !isfinite(constant) || constant < 0
        throw(ArgumentError("$constant_name must be finite and non-negative, got $constant"))
    end
    for (d, Δ) in enumerate(filter_width)
        if !isfinite(Δ) || Δ <= 0
            throw(ArgumentError(
                "filter_width[$d] must be finite and positive, got $Δ (filter_width = $filter_width)"
            ))
        end
    end
    for (d, n) in enumerate(field_size)
        if n <= 0
            throw(ArgumentError("field_size[$d] must be positive, got $n (field_size = $field_size)"))
        end
    end
    return nothing
end

"""
    _coerce_arrays_to_architecture(arch, arrays...)

Ensure all gradient arrays live on the target architecture.
CPU inputs may be uploaded to a GPU model. GPU inputs are never downloaded to
a CPU model implicitly; that architecture mismatch is rejected.
"""
function _coerce_arrays_to_architecture(arch::AbstractArchitecture, arrays::AbstractArray...)
    return tuple((_ensure_array_on_architecture(arch, arr) for arr in arrays)...)
end

@inline function _ensure_array_on_architecture(arch::AbstractArchitecture, arr::AbstractArray)
    if is_gpu(arch)
        return is_gpu_array(arr) ? arr : _move_array_to_gpu(arch, arr)
    else
        is_gpu_array(arr) && error(
            "A CPU LES model cannot consume GPU gradient arrays; CPU fallback is disabled. " *
            "Construct the model with architecture=GPU().")
        return arr
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

    _validate_model_params(:C_s, C_s, filter_width, field_size)

    # Effective filter width: geometric mean. Cached for inspection only — the
    # compute kernels re-derive it from `filter_width` so the two cannot desync.
    effective_delta = _effective_delta(T.(filter_width))

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
# ----------------------------------------------------------------------------
# Pointwise kernels
# ----------------------------------------------------------------------------
# One scalar function per formula, broadcast over whatever array type the model
# holds. CPU and GPU previously ran separately hand-written implementations of
# the same algebra: they happened to agree bitwise, but every future edit had to
# be mirrored by hand, and the GPU branch materialised up to eight field-sized
# temporaries per call (≈1 GB at 256³ Float64 for AMD 3-D). Broadcasting one
# scalar kernel needs none and cannot drift.

"""
    _smag_strain(gradients...)

Strain-rate magnitude `|S̄| = √(2 S̄ᵢⱼ S̄ᵢⱼ)` at a point.
"""
@inline function _smag_strain(u_x::T, u_y::T, v_x::T, v_y::T) where {T}
    S12 = T(0.5) * (u_y + v_x)
    return sqrt(T(2) * (u_x^2 + v_y^2 + T(2) * S12^2))
end

@inline function _smag_strain(u_x::T, u_y::T, u_z::T,
                              v_x::T, v_y::T, v_z::T,
                              w_x::T, w_y::T, w_z::T) where {T}
    S12 = T(0.5) * (u_y + v_x)
    S13 = T(0.5) * (u_z + w_x)
    S23 = T(0.5) * (v_z + w_y)
    return sqrt(T(2) * (u_x^2 + v_y^2 + w_z^2 + T(2) * (S12^2 + S13^2 + S23^2)))
end

function compute_eddy_viscosity!(
    model::SmagorinskyModel{T, 2, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}
) where {T, A, Arch}

    strain_mag = model.strain_magnitude
    eddy_visc = model.eddy_viscosity

    _validate_gradient_arrays(eddy_visc, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) =
        _coerce_arrays_to_architecture(model.architecture, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    # Derived per call: a mutated `filter_width` must not leave a stale Δ behind.
    CΔ_sq = (model.C_s * _effective_delta(model.filter_width))^2

    strain_mag .= _smag_strain.(∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
    eddy_visc .= CΔ_sq .* strain_mag

    return eddy_visc
end

function compute_eddy_viscosity!(
    model::SmagorinskyModel{T, 3, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T}
) where {T, A, Arch}

    strain_mag = model.strain_magnitude
    eddy_visc = model.eddy_viscosity

    _validate_gradient_arrays(eddy_visc, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)

    (∂u∂x, ∂u∂y, ∂u∂z,
     ∂v∂x, ∂v∂y, ∂v∂z,
     ∂w∂x, ∂w∂y, ∂w∂z) =
        _coerce_arrays_to_architecture(model.architecture,
                                      ∂u∂x, ∂u∂y, ∂u∂z,
                                      ∂v∂x, ∂v∂y, ∂v∂z,
                                      ∂w∂x, ∂w∂y, ∂w∂z)

    CΔ_sq = (model.C_s * _effective_delta(model.filter_width))^2

    strain_mag .= _smag_strain.(∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)
    eddy_visc .= CΔ_sq .* strain_mag

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

    _validate_model_params(:C, C, filter_width, field_size)

    # Cached for inspection only — the compute kernels re-derive Δ² from
    # `filter_width`, so mutating the struct cannot leave a stale squared width.
    filter_width_sq = T.(filter_width) .^ 2

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
    _amd_nu(C, Δ²..., clip, gradients...)

AMD eddy viscosity at a point:

    νₑ = max(0, -C (Δₖ² ∂uᵢ/∂xₖ ∂uⱼ/∂xₖ S̄ᵢⱼ) / (∂uₘ/∂xₙ ∂uₘ/∂xₙ))

summed over k. Note Δₖ indexes the DERIVATIVE direction, not the velocity
component — that is what makes the model anisotropy-aware, and it is the term
most easily got wrong.
"""
@inline function _amd_nu(C::T, Δx²::T, Δy²::T, clip::Bool,
                         u_x::T, u_y::T, v_x::T, v_y::T) where {T}
    S11 = u_x
    S22 = v_y
    S12 = T(0.5) * (u_y + v_x)
    denom = u_x^2 + u_y^2 + v_x^2 + v_y^2
    numer_x = Δx² * (u_x^2 * S11 + T(2) * u_x * v_x * S12 + v_x^2 * S22)
    numer_y = Δy² * (u_y^2 * S11 + T(2) * u_y * v_y * S12 + v_y^2 * S22)
    numer = -(numer_x + numer_y)
    return _apply_clip(clip, _safe_quotient(C, numer, denom))
end

@inline function _amd_nu(C::T, Δx²::T, Δy²::T, Δz²::T, clip::Bool,
                         u_x::T, u_y::T, u_z::T,
                         v_x::T, v_y::T, v_z::T,
                         w_x::T, w_y::T, w_z::T) where {T}
    S11 = u_x
    S22 = v_y
    S33 = w_z
    S12 = T(0.5) * (u_y + v_x)
    S13 = T(0.5) * (u_z + w_x)
    S23 = T(0.5) * (v_z + w_y)
    denom = u_x^2 + u_y^2 + u_z^2 + v_x^2 + v_y^2 + v_z^2 + w_x^2 + w_y^2 + w_z^2
    numer_x = Δx² * (u_x^2 * S11 + v_x^2 * S22 + w_x^2 * S33 +
                     T(2) * (u_x * v_x * S12 + u_x * w_x * S13 + v_x * w_x * S23))
    numer_y = Δy² * (u_y^2 * S11 + v_y^2 * S22 + w_y^2 * S33 +
                     T(2) * (u_y * v_y * S12 + u_y * w_y * S13 + v_y * w_y * S23))
    numer_z = Δz² * (u_z^2 * S11 + v_z^2 * S22 + w_z^2 * S33 +
                     T(2) * (u_z * v_z * S12 + u_z * w_z * S13 + v_z * w_z * S23))
    numer = -(numer_x + numer_y + numer_z)
    return _apply_clip(clip, _safe_quotient(C, numer, denom))
end

"""
    _amd_kappa(C, Δ²..., clip, velocity_gradients..., scalar_gradients...)

AMD eddy diffusivity at a point (Abkar, Bae & Moin 2016):

    κₑ = max(0, -C (Σₖ Δₖ² (∂ₖb) Σᵢ (∂ₖuᵢ)(∂ᵢb)) / (∂ₗb ∂ₗb))

The inner sum runs over ALL velocity components i, not just one.
"""
@inline function _amd_kappa(C::T, Δx²::T, Δy²::T, clip::Bool,
                            u_x::T, u_y::T, v_x::T, v_y::T,
                            b_x::T, b_y::T) where {T}
    denom = b_x^2 + b_y^2
    numer = -(Δx² * b_x * (u_x * b_x + v_x * b_y) +
              Δy² * b_y * (u_y * b_x + v_y * b_y))
    return _apply_clip(clip, _safe_quotient(C, numer, denom))
end

@inline function _amd_kappa(C::T, Δx²::T, Δy²::T, Δz²::T, clip::Bool,
                            u_x::T, u_y::T, u_z::T,
                            v_x::T, v_y::T, v_z::T,
                            w_x::T, w_y::T, w_z::T,
                            b_x::T, b_y::T, b_z::T) where {T}
    denom = b_x^2 + b_y^2 + b_z^2
    numer = -(Δx² * b_x * (u_x * b_x + v_x * b_y + w_x * b_z) +
              Δy² * b_y * (u_y * b_x + v_y * b_y + w_y * b_z) +
              Δz² * b_z * (u_z * b_x + v_z * b_y + w_z * b_z))
    return _apply_clip(clip, _safe_quotient(C, numer, denom))
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

    eddy_visc = model.eddy_viscosity

    _validate_gradient_arrays(eddy_visc, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) =
        _coerce_arrays_to_architecture(model.architecture, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    C = model.C
    Δx², Δy² = model.filter_width .^ 2   # derived per call; see _effective_delta
    clip = model.clip_negative

    eddy_visc .= _amd_nu.(C, Δx², Δy², clip, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)

    return eddy_visc
end

function compute_eddy_viscosity!(
    model::AMDModel{T, 3, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T}
) where {T, A, Arch}

    eddy_visc = model.eddy_viscosity

    _validate_gradient_arrays(eddy_visc, ∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)

    (∂u∂x, ∂u∂y, ∂u∂z,
     ∂v∂x, ∂v∂y, ∂v∂z,
     ∂w∂x, ∂w∂y, ∂w∂z) =
        _coerce_arrays_to_architecture(model.architecture,
                                      ∂u∂x, ∂u∂y, ∂u∂z,
                                      ∂v∂x, ∂v∂y, ∂v∂z,
                                      ∂w∂x, ∂w∂y, ∂w∂z)

    C = model.C
    Δx², Δy², Δz² = model.filter_width .^ 2
    clip = model.clip_negative

    eddy_visc .= _amd_nu.(C, Δx², Δy², Δz², clip,
                          ∂u∂x, ∂u∂y, ∂u∂z,
                          ∂v∂x, ∂v∂y, ∂v∂z,
                          ∂w∂x, ∂w∂y, ∂w∂z)

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

    eddy_diff = model.eddy_diffusivity

    _validate_gradient_arrays(eddy_diff, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y)

    (∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y) =
        _coerce_arrays_to_architecture(model.architecture,
                                       ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y)

    C = model.C
    Δx², Δy² = model.filter_width .^ 2
    clip = model.clip_negative

    eddy_diff .= _amd_kappa.(C, Δx², Δy², clip, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂b∂x, ∂b∂y)

    return eddy_diff
end

function compute_eddy_diffusivity!(
    model::AMDModel{T, 3, A, Arch},
    ∂u∂x::AbstractArray{T}, ∂u∂y::AbstractArray{T}, ∂u∂z::AbstractArray{T},
    ∂v∂x::AbstractArray{T}, ∂v∂y::AbstractArray{T}, ∂v∂z::AbstractArray{T},
    ∂w∂x::AbstractArray{T}, ∂w∂y::AbstractArray{T}, ∂w∂z::AbstractArray{T},
    ∂b∂x::AbstractArray{T}, ∂b∂y::AbstractArray{T}, ∂b∂z::AbstractArray{T}
) where {T, A, Arch}

    eddy_diff = model.eddy_diffusivity

    _validate_gradient_arrays(eddy_diff,
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
    Δx², Δy², Δz² = model.filter_width .^ 2
    clip = model.clip_negative

    eddy_diff .= _amd_kappa.(C, Δx², Δy², Δz², clip,
                             ∂u∂x, ∂u∂y, ∂u∂z,
                             ∂v∂x, ∂v∂y, ∂v∂z,
                             ∂w∂x, ∂w∂y, ∂w∂z,
                             ∂b∂x, ∂b∂y, ∂b∂z)

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

function get_eddy_diffusivity(model::EddyViscosityModel)
    throw(ArgumentError(
        "$(nameof(typeof(model))) has no eddy diffusivity — only AMDModel models " *
        "scalar transport directly. Use an AMDModel, or derive a diffusivity from " *
        "the eddy viscosity with a turbulent Prandtl number: κₑ = νₑ / Pr_t."
    ))
end

"""
    get_filter_width(model::EddyViscosityModel)

Return the filter width(s).
"""
get_filter_width(model::EddyViscosityModel) = model.filter_width

"""
    mean_eddy_viscosity(model::EddyViscosityModel; global_reduce=true)

Compute the domain-averaged eddy viscosity.

!!! warning "Collective under MPI"
    With `global_reduce=true` (the default) this is a **collective** call on
    `MPI.COMM_WORLD`: every rank must call it, or the ones that do will hang.
    In particular `rank == 0 && @info mean_eddy_viscosity(model)` deadlocks —
    compute on all ranks first, then log on one. Pass `global_reduce=false` for
    this rank's slab only, which is safe to call from a subset of ranks.
"""
function mean_eddy_viscosity(model::EddyViscosityModel; global_reduce::Bool=true)
    n = length(model.eddy_viscosity)
    n == 0 && return zero(eltype(model.eddy_viscosity))
    s = sum(model.eddy_viscosity)
    # LES models hold a per-rank plain Array (no communicator). Under MPI the bare
    # sum/length is only this rank's slab → reduce the global mean as Σs/Σn over
    # COMM_WORLD. Correct whether slabs are DECOMPOSED (tile the domain) or REPLICATED
    # (Σ scales numerator and denominator by nprocs equally).
    if global_reduce && MPI.Initialized() && MPI.Comm_size(MPI.COMM_WORLD) > 1
        s = MPI.Allreduce(s, +, MPI.COMM_WORLD)
        n = MPI.Allreduce(n, +, MPI.COMM_WORLD)
    end
    return s / n
end

"""
    max_eddy_viscosity(model::EddyViscosityModel; global_reduce=true)

Return the maximum eddy viscosity in the domain.

!!! warning "Collective under MPI"
    See [`mean_eddy_viscosity`](@ref) — with `global_reduce=true` every rank must
    call this or the callers hang.
"""
function max_eddy_viscosity(model::EddyViscosityModel; global_reduce::Bool=true)
    m = maximum(model.eddy_viscosity)        # per-rank slab maximum
    # Global maximum under MPI (idempotent → also correct for replicated slabs).
    return (global_reduce && MPI.Initialized() && MPI.Comm_size(MPI.COMM_WORLD) > 1) ?
           MPI.Allreduce(m, MPI.MAX, MPI.COMM_WORLD) : m
end

"""
    reset!(model::EddyViscosityModel)

Reset the eddy viscosity field to zero.
GPU-aware: fill!() works for both CPU and GPU arrays.
"""
function reset!(model::EddyViscosityModel)
    fill!(model.eddy_viscosity, zero(eltype(model.eddy_viscosity)))
    return model
end

# Also clear the cached |S̄|: it feeds sgs_dissipation, so leaving it populated
# after a reset lets a stale strain field flow into a diagnostic.
function reset!(model::SmagorinskyModel{T, N, A, Arch}) where {T, N, A, Arch}
    fill!(model.eddy_viscosity, zero(T))
    fill!(model.strain_magnitude, zero(T))
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
    (isfinite(C_s) && C_s >= 0) ||
        throw(ArgumentError("C_s must be finite and non-negative, got $C_s"))
    model.C_s = T(C_s)
    return model
end

"""
    set_constant!(model::AMDModel, C::Real)

Update the AMD Poincaré constant.
"""
function set_constant!(model::AMDModel{T, N, A, Arch}, C::Real) where {T, N, A, Arch}
    (isfinite(C) && C >= 0) ||
        throw(ArgumentError("C must be finite and non-negative, got $C"))
    model.C = T(C)
    return model
end

"""
    set_filter_width!(model, filter_width)

Update the filter width and its cached derived quantities together.

Assigning `model.filter_width` directly is also safe — the compute kernels derive
Δ² and the geometric-mean Δ from `filter_width` on every call — but it leaves the
cached `filter_width_sq` / `effective_delta` fields reading stale. Use this to
keep every view of the model consistent.
"""
function set_filter_width!(model::AMDModel{T, N, A, Arch},
                           filter_width::NTuple{N, Real}) where {T, N, A, Arch}
    _validate_model_params(:C, model.C, filter_width, model.field_size)
    model.filter_width = T.(filter_width)
    model.filter_width_sq = model.filter_width .^ 2
    return model
end

function set_filter_width!(model::SmagorinskyModel{T, N, A, Arch},
                           filter_width::NTuple{N, Real}) where {T, N, A, Arch}
    _validate_model_params(:C_s, model.C_s, filter_width, model.field_size)
    model.filter_width = T.(filter_width)
    model.effective_delta = _effective_delta(model.filter_width)
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

!!! warning "Collective under MPI"
    See [`mean_eddy_viscosity`](@ref) — with `global_reduce=true` every rank must
    call this or the callers hang.
"""
function mean_sgs_dissipation(model::EddyViscosityModel, strain_magnitude::AbstractArray{T};
                              global_reduce::Bool=true) where T
    νₑ = model.eddy_viscosity
    n = length(νₑ)
    n == 0 && return zero(T)
    # Use broadcasting and sum - works for both CPU and GPU.
    # εₛₛ = νₑ |S̄|² with |S̄| = √(2 S̄ᵢⱼS̄ᵢⱼ); no extra factor of 2 (see sgs_dissipation).
    s = sum(νₑ .* strain_magnitude.^2)
    # Global mean under MPI (Σs/Σn — correct for decomposed or replicated slabs).
    if global_reduce && MPI.Initialized() && MPI.Comm_size(MPI.COMM_WORLD) > 1
        s = MPI.Allreduce(s, +, MPI.COMM_WORLD)
        n = MPI.Allreduce(n, +, MPI.COMM_WORLD)
    end
    return s / n
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
export reset!, set_constant!, set_filter_width!
export sgs_dissipation, mean_sgs_dissipation
