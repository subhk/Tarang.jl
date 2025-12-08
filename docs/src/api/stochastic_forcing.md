# Stochastic Forcing API

## Types

### StochasticForcing

```@docs
StochasticForcing
```

**Fields:**
- `spectrum::AbstractArray{T}` - Forcing amplitude spectrum sqrt(Q(k))
- `forcing_rate::T` - Energy injection rate
- `k_forcing::T` - Central forcing wavenumber
- `dk_forcing::T` - Forcing bandwidth
- `dt::T` - Timestep for Stratonovich scaling
- `field_size::Tuple` - Size of the field to force
- `cached_forcing::AbstractArray{Complex{T}}` - Cached forcing (constant within timestep)
- `is_stochastic::Bool` - Flag for stochastic vs deterministic
- `rng::AbstractRNG` - Random number generator
- `last_update_time::T` - Time when forcing was last updated

### DeterministicForcing

```@docs
DeterministicForcing
```

**Fields:**
- `forcing_function::Function` - Function(x, y, t, params) returning forcing
- `field_size::Tuple` - Size of the field
- `cached_forcing::AbstractArray{T}` - Cached forcing values
- `parameters::Dict{Symbol, Any}` - Parameters passed to forcing function

### Abstract Types

```julia
abstract type Forcing end
abstract type StochasticForcingType <: Forcing end
abstract type DeterministicForcingType <: Forcing end
```

## Forcing Generation

### generate_forcing!

```@docs
generate_forcing!
```

Generate forcing realization. Called once per timestep, returns cached value for substeps.

**Signatures:**
```julia
generate_forcing!(forcing::StochasticForcing, current_time::Real)
generate_forcing!(forcing::StochasticForcing, current_time::Real, substep::Int)
generate_forcing!(forcing::DeterministicForcing, grid, current_time::Real)
```

### apply_forcing!

```@docs
apply_forcing!
```

Apply forcing to a field in spectral space.

**Signatures:**
```julia
apply_forcing!(field::AbstractArray{<:Complex}, forcing::StochasticForcing, current_time::Real)
apply_forcing!(field::AbstractArray{<:Complex}, forcing::StochasticForcing, current_time::Real, substep::Int)
```

## Configuration

### set_dt!

```@docs
set_dt!
```

Update the timestep used for Stratonovich scaling.

```julia
set_dt!(forcing::StochasticForcing, dt::Real)
```

### reset_forcing!

```@docs
reset_forcing!
```

Reset forcing cache. Forces regeneration on next access.

```julia
reset_forcing!(forcing::StochasticForcing)
```

## Diagnostics

### energy_injection_rate

```@docs
energy_injection_rate
```

Compute mean energy injection rate.

```julia
energy_injection_rate(forcing::StochasticForcing) -> Real
```

### instantaneous_power

```@docs
instantaneous_power
```

Compute instantaneous power input to a field.

```julia
instantaneous_power(forcing::StochasticForcing, field::AbstractArray) -> Real
```

### get_forcing_real

```@docs
get_forcing_real
```

Get forcing in real space (returns real part of cached forcing).

```julia
get_forcing_real(forcing::StochasticForcing) -> AbstractArray
```

## Spectrum Generation

### compute_forcing_spectrum

```@docs
compute_forcing_spectrum
```

Compute forcing spectrum for different spectrum types.

```julia
compute_forcing_spectrum(field_size, k_forcing, dk_forcing, forcing_rate, spectrum_type, dtype)
```

**Spectrum types:**
- `:ring` - Annular forcing in wavenumber space
- `:isotropic` - Gaussian envelope around k_forcing
- `:bandlimited` - Sharp cutoff band
- `:kolmogorov` - Large-scale forcing

## Timestepper Integration

### set_forcing!

```julia
set_forcing!(state::TimestepperState, forcing)
```

Attach stochastic forcing to a timestepper state.

### update_forcing!

```julia
update_forcing!(state::TimestepperState, sim_time::Float64)
```

Generate new forcing at beginning of timestep.

### reset_forcing_flag!

```julia
reset_forcing_flag!(state::TimestepperState)
```

Reset forcing flag at end of timestep.

### get_cached_forcing

```julia
get_cached_forcing(state::TimestepperState) -> Union{AbstractArray, Nothing}
```

Get cached forcing array from timestepper state.

## Index

```@index
Pages = ["stochastic_forcing.md"]
```
