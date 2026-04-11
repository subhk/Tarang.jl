# Stochastic Forcing API

## Types

### Abstract Types

```julia
abstract type Forcing end
abstract type StochasticForcingType <: Forcing end
abstract type DeterministicForcingType <: Forcing end
```

### StochasticForcing

```@docs
StochasticForcing
```

Stochastic forcing in Fourier space with white-noise temporal correlation.

**Type signature:**
```julia
mutable struct StochasticForcing{T<:AbstractFloat, N} <: StochasticForcingType
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `forcing_spectrum` | `Array{T, N}` | √Q̂(k) - amplitude spectrum |
| `energy_injection_rate` | `T` | Target ε |
| `k_forcing` | `T` | Central forcing wavenumber |
| `dk_forcing` | `T` | Forcing bandwidth |
| `dt` | `T` | Current timestep |
| `domain_size` | `NTuple{N, T}` | Domain extent (Lx, Ly, ...) |
| `field_size` | `NTuple{N, Int}` | Grid size (Nx, Ny, ...) |
| `wavenumbers` | `NTuple{N, Vector{T}}` | Wavenumber arrays (kx, ky, ...) |
| `cached_forcing` | `Array{Complex{T}, N}` | Cached F̂ (constant within timestep) |
| `prevsol` | `Union{Nothing, Array{Complex{T}, N}}` | Previous solution for Stratonovich work |
| `rng` | `AbstractRNG` | Random number generator |
| `last_update_time` | `T` | Time of last forcing update |
| `spectrum_type` | `Symbol` | Type of forcing spectrum |

**Constructor:**

```julia
StochasticForcing(;
    field_size::NTuple{N, Int},                    # Required
    domain_size::NTuple{N, Real} = ntuple(i -> 2π, N),
    energy_injection_rate::Real = 1.0,
    k_forcing::Real = 4.0,
    dk_forcing::Real = 1.0,
    dt::Real = 0.01,
    spectrum_type::Symbol = :ring,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N}
```

**Spectrum types:**

| Symbol | Description | Formula |
|--------|-------------|---------|
| `:ring` | Gaussian ring around k_f | exp(-(|k| - k_f)² / 2δ_f²) |
| `:band` | Sharp band [k_f - δ_f, k_f + δ_f] | 1 if |k-k_f| < δ_f, else 0 |
| `:lowk` | Low wavenumber forcing | 1 if |k| < k_f, else 0 |
| `:kolmogorov` | Large-scale Kolmogorov | Smooth large-scale forcing |

### DeterministicForcing

```@docs
DeterministicForcing
```

Deterministic (non-random) forcing.

**Type signature:**
```julia
mutable struct DeterministicForcing{T<:AbstractFloat, N, A<:AbstractArray{T, N}} <: DeterministicForcingType
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `forcing_function` | `Function` | f(x, y, ..., t, params) → forcing |
| `field_size` | `NTuple{N, Int}` | Grid size |
| `cached_forcing` | `A` | Cached forcing values on the chosen architecture |
| `parameters` | `Dict{Symbol, Any}` | Parameters for forcing function |
| `architecture` | `AbstractArchitecture` | CPU() or GPU() backend |

**Constructor:**

```julia
DeterministicForcing(
    forcing_function::Function,
    field_size::NTuple{N, Int};
    parameters::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    dtype::Type{T} = Float64,
    architecture::AbstractArchitecture = CPU()
)
```

---

## Forcing Generation

### generate_forcing!

```@docs
generate_forcing!
```

Generate forcing realization. Returns cached value for substeps > 1.

**Signatures:**

```julia
# Stochastic forcing
generate_forcing!(forcing::StochasticForcing, t::Real, substep::Int=1)
generate_forcing!(forcing::StochasticForcing, t::Real)

# Deterministic forcing
generate_forcing!(forcing::DeterministicForcing, grid, t::Real)
```

**Key behavior:**
- `substep == 1`: Generates new random forcing
- `substep > 1`: Returns cached forcing (same as substep 1)
- Time check: Won't regenerate if already updated at this time

**Returns:** The cached forcing array `forcing.cached_forcing`

### apply_forcing!

```@docs
apply_forcing!
```

Add forcing to a field in spectral space.

**Signature:**

```julia
apply_forcing!(
    rhs::AbstractArray{Complex{T}, N},
    forcing::StochasticForcing{T, N},
    t::Real,
    substep::Int=1
) where {T, N}
```

Equivalent to: `rhs .+= generate_forcing!(forcing, t, substep)`

---

## Configuration

### set_dt!

```@docs
set_dt!
```

Update the timestep used for Stratonovich scaling.

```julia
set_dt!(forcing::StochasticForcing{T, N}, dt::Real) where {T, N}
```

Call this when dt changes (e.g., adaptive timestepping).

### reset_forcing!

```@docs
reset_forcing!
```

Reset forcing cache. Forces regeneration on next `generate_forcing!` call.

```julia
reset_forcing!(forcing::StochasticForcing{T, N}) where {T, N}
```

---

## Work Calculation

### store_prevsol!

```@docs
store_prevsol!
```

Store the current solution for Stratonovich work calculation.

```julia
store_prevsol!(forcing::StochasticForcing{T, N}, sol::AbstractArray{Complex{T}, N})
```

Call at the **beginning** of each timestep, before advancing.

### work_stratonovich

```@docs
work_stratonovich
```

Compute work done using Stratonovich interpretation.

```julia
work_stratonovich(forcing::StochasticForcing{T, N}, sol::AbstractArray{Complex{T}, N}) -> T
```

**Formula:**
```math
W = -\text{Re}\left\langle \frac{\psi^n + \psi^{n+1}}{2} \cdot \hat{F}^* \right\rangle
```

Uses midpoint evaluation (requires `store_prevsol!` called first).

### work_ito

```@docs
work_ito
```

Compute work done using Itô interpretation.

```julia
work_ito(forcing::StochasticForcing{T, N}, sol_prev::AbstractArray{Complex{T}, N}) -> T
```

**Formula:**
```math
W_{\text{Itô}} = -\text{Re}\langle \psi^n \cdot \hat{F}^* \rangle \cdot dt + \varepsilon \cdot dt
```

Uses initial value plus drift correction.

---

## Diagnostics

### mean_energy_injection_rate

```@docs
mean_energy_injection_rate
```

Return the target (mean) energy injection rate ε.

```julia
mean_energy_injection_rate(forcing::StochasticForcing) -> T
```

### instantaneous_power

```@docs
instantaneous_power
```

Compute instantaneous power input P(t) = dW/dt.

```julia
instantaneous_power(forcing::StochasticForcing{T, N}, sol::AbstractArray{Complex{T}, N}) -> T
```

This fluctuates around the mean ε due to randomness.

### forcing_enstrophy_injection_rate

```@docs
forcing_enstrophy_injection_rate
```

Compute mean enstrophy injection rate (for 2D vorticity forcing).

```julia
forcing_enstrophy_injection_rate(forcing::StochasticForcing{T, N}) -> T
```

**Formula:**
```math
\eta = \sum_k |k|^2 \hat{Q}(k) / 2
```

### get_forcing_spectrum

```@docs
get_forcing_spectrum
```

Return the forcing amplitude spectrum √Q̂(k).

```julia
get_forcing_spectrum(forcing::StochasticForcing) -> Array{T, N}
```

### get_cached_forcing

```@docs
get_cached_forcing
```

Return the current cached forcing F̂(k).

```julia
get_cached_forcing(forcing::StochasticForcing) -> Array{Complex{T}, N}
```

---

## Internal Functions

### build_wavenumbers

Build wavenumber arrays for each dimension.

```julia
build_wavenumbers(
    field_size::NTuple{N, Int},
    domain_size::NTuple{N, Real},
    dtype::Type{T}
) -> NTuple{N, Vector{T}}
```

### compute_forcing_spectrum

Compute the forcing amplitude spectrum √Q̂(k).

```julia
compute_forcing_spectrum(
    wavenumbers::NTuple{N, Vector{T}},
    k_f::Real,
    dk_f::Real,
    ε::Real,
    domain_size::NTuple{N, Real},
    spectrum_type::Symbol,
    dtype::Type{T}
) -> Array{T, N}
```

The spectrum is normalized such that energy injection rate equals ε.

---

## Exports

```julia
export Forcing, StochasticForcingType, DeterministicForcingType
export StochasticForcing, DeterministicForcing
export generate_forcing!, apply_forcing!
export reset_forcing!, set_dt!
export store_prevsol!, work_stratonovich, work_ito
export mean_energy_injection_rate, instantaneous_power
export forcing_enstrophy_injection_rate
export get_forcing_spectrum, get_cached_forcing
```

---

## Index

```@index
Pages = ["stochastic_forcing.md"]
```
