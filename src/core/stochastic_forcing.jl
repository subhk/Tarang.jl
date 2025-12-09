"""
# Stochastic Forcing for Tarang.jl

Implementation of stochastic forcing following the mathematical framework from
GeophysicalFlows.jl/FourierFlows.jl.

## Mathematical Background

### Forcing Statistics

The stochastic forcing ξ(x,t) has the following statistical properties:
- **Zero mean**: ⟨ξ(x, t)⟩ = 0
- **White in time**: ⟨ξ(x, t) ξ(x', t')⟩ = Q(x - x') δ(t - t')
- **Spatially correlated**: Q(x - x') is the spatial covariance function

In Fourier space, the covariance becomes:
    ⟨ξ̂(k, t) ξ̂*(k', t')⟩ = Q̂(k) δ(k - k') δ(t - t')

where Q̂(k) is the **forcing spectrum** (power spectral density).

### Energy Injection Rate

For a system with kinetic energy E = ½⟨|u|²⟩, the mean energy injection rate is:

    ε = ∫ (d^d k)/(2π)^d · Q̂(k)/(2|k|²)    (for vorticity forcing)

or more generally:

    ε = ∑_k Q̂(k) / (2 * normalization)

### Stratonovich vs Itô Interpretation

We use **Stratonovich calculus** because:
1. The chain rule matches ordinary calculus
2. It works the same for stochastic and deterministic forcing
3. Physical systems with finite correlation time converge to Stratonovich

**Work done by forcing (Stratonovich)**:
    P = -⟨[ψ(tⱼ) + ψ(tⱼ₊₁)]/2 · ξ(tⱼ₊₁)⟩

### Numerical Implementation

For time discretization with step dt, the forcing is:

    F̂(k) = √(Q̂(k)/dt) · exp(2πi · rand())

The √dt scaling ensures correct variance for the discrete-time Wiener process:
    ⟨|F̂|²⟩ · dt = Q̂(k)

### Ring Forcing (Isotropic)

The most common forcing spectrum for 2D turbulence is "ring forcing":

    Q̂(k) ∝ exp(-(|k| - k_f)² / (2δ_f²))

where:
- k_f = forcing wavenumber (ring center)
- δ_f = forcing bandwidth (ring width)

The spectrum is normalized to inject energy at rate ε.

## References

1. GeophysicalFlows.jl: https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/
2. FourierFlows.jl: https://github.com/FourierFlows/FourierFlows.jl
"""

using Random
using LinearAlgebra

# ============================================================================
# Abstract forcing types
# ============================================================================

"""
    Forcing

Abstract base type for all forcing types.
"""
abstract type Forcing end

"""
    StochasticForcingType <: Forcing

Abstract type for stochastic (random) forcing.
"""
abstract type StochasticForcingType <: Forcing end

"""
    DeterministicForcingType <: Forcing

Abstract type for deterministic forcing.
"""
abstract type DeterministicForcingType <: Forcing end

# ============================================================================
# Stochastic Forcing
# ============================================================================

"""
    StochasticForcing{T, N}

Stochastic forcing in Fourier space with white-noise temporal correlation.

## Mathematical Properties

The forcing F̂(k,t) satisfies:
- ⟨F̂(k,t)⟩ = 0                                          (zero mean)
- ⟨F̂(k,t) F̂*(k',t')⟩ = Q̂(k) δ(k-k') δ(t-t')/(dt)     (white noise)

## Implementation

At each timestep, the forcing is computed as:
    F̂(k) = √(Q̂(k)) · ξ(k) / √(dt)

where ξ(k) is complex white noise with |ξ| = 1 and random phase.

## Fields

- `forcing_spectrum::Array{T,N}`: √Q̂(k) - square root of power spectrum
- `energy_injection_rate::T`: Target energy injection rate ε
- `k_forcing::T`: Central forcing wavenumber k_f
- `dk_forcing::T`: Forcing bandwidth δ_f
- `dt::T`: Current timestep (for proper scaling)
- `domain_size::NTuple{N,T}`: Domain size (Lx, Ly, ...)
- `field_size::NTuple{N,Int}`: Grid size (Nx, Ny, ...)
- `wavenumbers::NTuple{N,Vector{T}}`: Wavenumber arrays (kx, ky, ...)
- `cached_forcing::Array{Complex{T},N}`: Cached forcing (constant within timestep)
- `prevsol::Union{Nothing,Array{Complex{T},N}}`: Previous solution (for Stratonovich work)
- `rng::AbstractRNG`: Random number generator
- `last_update_time::T`: Time of last forcing update
- `spectrum_type::Symbol`: Type of forcing spectrum
"""
mutable struct StochasticForcing{T<:AbstractFloat, N} <: StochasticForcingType
    forcing_spectrum::Array{T, N}           # √Q̂(k) - amplitude spectrum
    energy_injection_rate::T                # Target ε
    k_forcing::T                            # Central wavenumber
    dk_forcing::T                           # Bandwidth
    dt::T                                   # Timestep
    domain_size::NTuple{N, T}               # Domain (Lx, Ly, ...)
    field_size::NTuple{N, Int}              # Grid (Nx, Ny, ...)
    wavenumbers::NTuple{N, Vector{T}}       # (kx, ky, ...)
    cached_forcing::Array{Complex{T}, N}    # Cached F̂
    prevsol::Union{Nothing, Array{Complex{T}, N}}  # For Stratonovich work
    rng::AbstractRNG
    last_update_time::T
    spectrum_type::Symbol
end

"""
    StochasticForcing(;
        field_size,
        domain_size = ntuple(i -> 2π, length(field_size)),
        energy_injection_rate = 1.0,
        k_forcing = 4.0,
        dk_forcing = 1.0,
        dt = 0.01,
        spectrum_type = :ring,
        rng = Random.GLOBAL_RNG,
        dtype = Float64
    )

Create a stochastic forcing configuration.

## Arguments

- `field_size::NTuple{N,Int}`: Grid size (Nx, Ny, ...)
- `domain_size::NTuple{N,Real}`: Domain size (Lx, Ly, ...), default 2π in each direction
- `energy_injection_rate::Real`: Target energy injection rate ε (default: 1.0)
- `k_forcing::Real`: Central forcing wavenumber k_f (default: 4.0)
- `dk_forcing::Real`: Forcing bandwidth δ_f (default: 1.0)
- `dt::Real`: Initial timestep (default: 0.01)
- `spectrum_type::Symbol`: Spectrum shape (default: :ring)
    - `:ring` - Gaussian ring in wavenumber space
    - `:band` - Sharp band |k| ∈ [k_f - δ_f, k_f + δ_f]
    - `:lowk` - Low wavenumber forcing |k| < k_f
    - `:kolmogorov` - Forcing at large scales
- `rng::AbstractRNG`: Random number generator
- `dtype::Type`: Floating point type (default: Float64)

## Example

```julia
# Create ring forcing for 2D turbulence
forcing = StochasticForcing(
    field_size = (256, 256),
    domain_size = (2π, 2π),
    energy_injection_rate = 0.1,
    k_forcing = 10.0,   # Force at |k| ≈ 10
    dk_forcing = 2.0,   # Bandwidth
    dt = 0.001
)

# In your simulation loop:
generate_forcing!(forcing, t, substep)
```
"""
function StochasticForcing(;
    field_size::NTuple{N, Int},
    domain_size::NTuple{N, Real} = ntuple(i -> 2π, N),
    energy_injection_rate::Real = 1.0,
    k_forcing::Real = 4.0,
    dk_forcing::Real = 1.0,
    dt::Real = 0.01,
    spectrum_type::Symbol = :ring,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N}

    # Build wavenumber arrays
    wavenumbers = build_wavenumbers(field_size, domain_size, dtype)

    # Compute the forcing spectrum √Q̂(k)
    forcing_spectrum = compute_forcing_spectrum(
        wavenumbers, k_forcing, dk_forcing, energy_injection_rate,
        domain_size, spectrum_type, dtype
    )

    # Allocate cached forcing array
    cached_forcing = zeros(Complex{T}, field_size)

    # Previous solution for Stratonovich work calculation
    prevsol = nothing

    StochasticForcing{T, N}(
        forcing_spectrum,
        T(energy_injection_rate),
        T(k_forcing),
        T(dk_forcing),
        T(dt),
        T.(domain_size),
        field_size,
        wavenumbers,
        cached_forcing,
        prevsol,
        rng,
        T(-Inf),  # Initialize to -Inf so first call always updates
        spectrum_type
    )
end

"""
    build_wavenumbers(field_size, domain_size, dtype)

Build wavenumber arrays for each dimension.
"""
function build_wavenumbers(
    field_size::NTuple{N, Int},
    domain_size::NTuple{N, Real},
    dtype::Type{T}
) where {T<:AbstractFloat, N}

    wavenumbers = ntuple(N) do d
        n = field_size[d]
        L = T(domain_size[d])
        dk = 2π / L

        # Standard FFT wavenumber ordering: 0, 1, ..., n/2-1, -n/2, ..., -1
        k = zeros(T, n)
        for i in 1:n
            if i <= n ÷ 2 + 1
                k[i] = (i - 1) * dk
            else
                k[i] = (i - 1 - n) * dk
            end
        end
        k
    end

    return wavenumbers
end

"""
    compute_forcing_spectrum(wavenumbers, k_f, dk_f, ε, domain_size, spectrum_type, dtype)

Compute the forcing amplitude spectrum √Q̂(k).

The spectrum is normalized such that the energy injection rate equals ε.
"""
function compute_forcing_spectrum(
    wavenumbers::NTuple{N, Vector{T}},
    k_f::Real,
    dk_f::Real,
    ε::Real,
    domain_size::NTuple{N, Real},
    spectrum_type::Symbol,
    dtype::Type{T}
) where {T<:AbstractFloat, N}

    field_size = ntuple(d -> length(wavenumbers[d]), N)
    spectrum = zeros(T, field_size)

    if N == 1
        _fill_spectrum_1d!(spectrum, wavenumbers[1], k_f, dk_f, spectrum_type)
    elseif N == 2
        _fill_spectrum_2d!(spectrum, wavenumbers[1], wavenumbers[2], k_f, dk_f, spectrum_type)
    elseif N == 3
        _fill_spectrum_3d!(spectrum, wavenumbers[1], wavenumbers[2], wavenumbers[3],
                          k_f, dk_f, spectrum_type)
    else
        error("Unsupported dimension: $N")
    end

    # Enforce zero mean (no forcing at k=0)
    spectrum[1] = zero(T)
    if N >= 2
        spectrum[1, 1] = zero(T)
    end
    if N >= 3
        spectrum[1, 1, 1] = zero(T)
    end

    # Normalize to achieve target energy injection rate ε
    # Energy injection rate: ε = ∑_k Q̂(k) / (2 * domain_area)
    domain_area = prod(domain_size)

    # Current unnormalized energy injection
    ε0 = sum(spectrum.^2) / (2 * domain_area)

    if ε0 > 0
        # Scale spectrum to achieve target ε
        spectrum .*= sqrt(T(ε) / ε0)
    end

    return spectrum
end

function _fill_spectrum_1d!(spectrum, kx, k_f, dk_f, spectrum_type)
    for i in eachindex(kx)
        k = abs(kx[i])
        spectrum[i] = _spectrum_amplitude(k, k_f, dk_f, spectrum_type)
    end
end

function _fill_spectrum_2d!(spectrum, kx, ky, k_f, dk_f, spectrum_type)
    Nx, Ny = length(kx), length(ky)
    for j in 1:Ny
        for i in 1:Nx
            k = sqrt(kx[i]^2 + ky[j]^2)
            spectrum[i, j] = _spectrum_amplitude(k, k_f, dk_f, spectrum_type)
        end
    end
end

function _fill_spectrum_3d!(spectrum, kx, ky, kz, k_f, dk_f, spectrum_type)
    Nx, Ny, Nz = length(kx), length(ky), length(kz)
    for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                kmag = sqrt(kx[i]^2 + ky[j]^2 + kz[k]^2)
                spectrum[i, j, k] = _spectrum_amplitude(kmag, k_f, dk_f, spectrum_type)
            end
        end
    end
end

"""
    _spectrum_amplitude(k, k_f, dk_f, spectrum_type)

Compute the (unnormalized) spectrum amplitude at wavenumber k.
"""
function _spectrum_amplitude(k::T, k_f::Real, dk_f::Real, spectrum_type::Symbol) where T
    if k ≈ 0
        return zero(T)
    end

    if spectrum_type == :ring
        # Gaussian ring: concentrated around |k| = k_f
        # Q̂(k) ∝ exp(-(|k| - k_f)² / (2 δ_f²))
        return exp(-((k - k_f)^2) / (2 * dk_f^2))

    elseif spectrum_type == :band
        # Sharp band: |k| ∈ [k_f - δ_f, k_f + δ_f]
        if abs(k - k_f) < dk_f
            return one(T)
        else
            return zero(T)
        end

    elseif spectrum_type == :lowk
        # Low wavenumber forcing: |k| < k_f
        if k < k_f
            return one(T)
        else
            return zero(T)
        end

    elseif spectrum_type == :kolmogorov
        # Large-scale forcing for Kolmogorov cascade
        # Smooth cutoff at k_f
        if k < k_f + dk_f
            return exp(-((k - k_f)^2) / (2 * dk_f^2)) * (k / k_f)
        else
            return zero(T)
        end

    else
        error("Unknown spectrum type: $spectrum_type")
    end
end

# ============================================================================
# Forcing Generation
# ============================================================================

"""
    generate_forcing!(forcing::StochasticForcing, t::Real, substep::Int=1)

Generate stochastic forcing at time t.

## Key Points

1. **Forcing is regenerated only on substep 1** - ensures forcing stays constant
   within a timestep for IMEX and multi-stage methods.

2. **Scaling**: F̂(k) = √Q̂(k) · exp(2πi·rand) / √dt

3. **Zero mean**: The k=0 mode is always set to zero.

## Arguments

- `forcing`: StochasticForcing configuration
- `t`: Current simulation time
- `substep`: Current substep (1 for first substep)

## Returns

The cached forcing array (modified in-place).
"""
function generate_forcing!(forcing::StochasticForcing{T, N}, t::Real, substep::Int=1) where {T, N}
    # Only update forcing on first substep
    if substep > 1
        return forcing.cached_forcing
    end

    # Check if we've already updated at this time
    if t ≈ forcing.last_update_time
        return forcing.cached_forcing
    end

    # Generate complex white noise with unit magnitude and random phase
    # ξ = exp(2πi · rand) has |ξ| = 1
    for i in eachindex(forcing.cached_forcing)
        phase = 2π * rand(forcing.rng, T)
        noise = exp(im * phase)

        # F̂ = √Q̂ · ξ / √dt  (Stratonovich scaling)
        forcing.cached_forcing[i] = forcing.forcing_spectrum[i] * noise / sqrt(forcing.dt)
    end

    # Enforce zero mean
    forcing.cached_forcing[1] = zero(Complex{T})
    if N >= 2
        forcing.cached_forcing[1, 1] = zero(Complex{T})
    end
    if N >= 3
        forcing.cached_forcing[1, 1, 1] = zero(Complex{T})
    end

    forcing.last_update_time = T(t)
    return forcing.cached_forcing
end

"""
    generate_forcing!(forcing::StochasticForcing, t::Real)

Generate forcing without substep tracking. Equivalent to substep=1.
"""
generate_forcing!(forcing::StochasticForcing, t::Real) = generate_forcing!(forcing, t, 1)

# ============================================================================
# Forcing Application and Work Calculation
# ============================================================================

"""
    apply_forcing!(rhs::AbstractArray, forcing::StochasticForcing, t::Real, substep::Int=1)

Add stochastic forcing to the RHS in spectral space.

## Arguments

- `rhs`: Right-hand side array (modified in-place)
- `forcing`: StochasticForcing configuration
- `t`: Current simulation time
- `substep`: Current substep number
"""
function apply_forcing!(
    rhs::AbstractArray{Complex{T}, N},
    forcing::StochasticForcing{T, N},
    t::Real,
    substep::Int=1
) where {T, N}

    F = generate_forcing!(forcing, t, substep)
    rhs .+= F
    return rhs
end

"""
    store_prevsol!(forcing::StochasticForcing, sol::AbstractArray)

Store the current solution for Stratonovich work calculation.

Call this at the **beginning** of each timestep, before advancing.
"""
function store_prevsol!(forcing::StochasticForcing{T, N}, sol::AbstractArray{Complex{T}, N}) where {T, N}
    if forcing.prevsol === nothing
        forcing.prevsol = copy(sol)
    else
        forcing.prevsol .= sol
    end
end

"""
    work_stratonovich(forcing::StochasticForcing, sol::AbstractArray)

Compute work done by forcing using Stratonovich interpretation.

## Formula

    W = -Re⟨(ψⁿ + ψⁿ⁺¹)/2 · F̂*⟩ · dt

This correctly accounts for the correlation between forcing and response.

## Arguments

- `forcing`: StochasticForcing with prevsol stored
- `sol`: Current solution ψⁿ⁺¹

## Returns

Work done during this timestep (scalar).
"""
function work_stratonovich(forcing::StochasticForcing{T, N}, sol::AbstractArray{Complex{T}, N}) where {T, N}
    if forcing.prevsol === nothing
        return zero(T)
    end

    # Stratonovich work: W = -Re⟨(ψⁿ + ψⁿ⁺¹)/2 · F̂*⟩
    domain_area = prod(forcing.domain_size)

    work = zero(T)
    for i in eachindex(sol)
        # Midpoint value (Stratonovich)
        ψ_mid = (forcing.prevsol[i] + sol[i]) / 2
        work -= real(ψ_mid * conj(forcing.cached_forcing[i]))
    end

    # Multiply by dt (forcing was already scaled by 1/√dt, so net effect is √dt)
    # Actually: F = √Q/√dt, and work integrates over dt, so:
    # W = ⟨ψ · F⟩ · dt = ⟨ψ · √Q · ξ/√dt⟩ · dt = ⟨ψ · √Q · ξ⟩ · √dt
    return work * sqrt(forcing.dt) / domain_area
end

"""
    work_ito(forcing::StochasticForcing, sol::AbstractArray)

Compute work done by forcing using Itô interpretation.

## Formula

    W_Itô = -Re⟨ψⁿ · F̂*⟩ · dt + ε · dt

The drift correction ε ensures ⟨W_Itô⟩ = ⟨W_Stratonovich⟩.
"""
function work_ito(forcing::StochasticForcing{T, N}, sol_prev::AbstractArray{Complex{T}, N}) where {T, N}
    domain_area = prod(forcing.domain_size)

    # Itô work (uses previous solution)
    work = zero(T)
    for i in eachindex(sol_prev)
        work -= real(sol_prev[i] * conj(forcing.cached_forcing[i]))
    end

    # Add drift correction (energy injection rate)
    drift = forcing.energy_injection_rate * forcing.dt

    return work * sqrt(forcing.dt) / domain_area + drift
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    mean_energy_injection_rate(forcing::StochasticForcing)

Return the target (mean) energy injection rate ε.

This is the ensemble average of work done per unit time.
"""
mean_energy_injection_rate(forcing::StochasticForcing) = forcing.energy_injection_rate

"""
    instantaneous_power(forcing::StochasticForcing, sol::AbstractArray)

Compute instantaneous power input P(t) = dW/dt.

This fluctuates around the mean ε due to randomness.
"""
function instantaneous_power(forcing::StochasticForcing{T, N}, sol::AbstractArray{Complex{T}, N}) where {T, N}
    domain_area = prod(forcing.domain_size)

    power = zero(T)
    for i in eachindex(sol)
        power -= real(sol[i] * conj(forcing.cached_forcing[i]))
    end

    # Unscale by √dt (since F = √Q/√dt, the power is ⟨ψ·F⟩ which has √dt scaling)
    return power * sqrt(forcing.dt) / domain_area
end

"""
    forcing_enstrophy_injection_rate(forcing::StochasticForcing)

Compute mean enstrophy injection rate (for 2D vorticity forcing).

    η = ∑_k |k|² Q̂(k) / 2
"""
function forcing_enstrophy_injection_rate(forcing::StochasticForcing{T, N}) where {T, N}
    if N != 2
        @warn "Enstrophy injection rate is only meaningful for 2D"
    end

    domain_area = prod(forcing.domain_size)
    kx, ky = forcing.wavenumbers

    η = zero(T)
    for j in eachindex(ky)
        for i in eachindex(kx)
            k2 = kx[i]^2 + ky[j]^2
            η += k2 * forcing.forcing_spectrum[i, j]^2
        end
    end

    return η / (2 * domain_area)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    set_dt!(forcing::StochasticForcing, dt::Real)

Update the timestep for Stratonovich scaling.

Call this when dt changes (e.g., adaptive timestepping).
"""
function set_dt!(forcing::StochasticForcing{T, N}, dt::Real) where {T, N}
    forcing.dt = T(dt)
end

"""
    reset_forcing!(forcing::StochasticForcing)

Reset the forcing cache, causing regeneration on next call.
"""
function reset_forcing!(forcing::StochasticForcing{T, N}) where {T, N}
    forcing.last_update_time = T(-Inf)
    fill!(forcing.cached_forcing, zero(Complex{T}))
    if forcing.prevsol !== nothing
        fill!(forcing.prevsol, zero(Complex{T}))
    end
end

"""
    get_forcing_spectrum(forcing::StochasticForcing)

Return the forcing amplitude spectrum √Q̂(k).
"""
get_forcing_spectrum(forcing::StochasticForcing) = forcing.forcing_spectrum

"""
    get_cached_forcing(forcing::StochasticForcing)

Return the current cached forcing F̂(k).
"""
get_cached_forcing(forcing::StochasticForcing) = forcing.cached_forcing

# ============================================================================
# Deterministic Forcing
# ============================================================================

"""
    DeterministicForcing{T, N}

Deterministic (non-random) forcing.

## Example

```julia
# Sinusoidal forcing
forcing = DeterministicForcing(
    (x, y, t, p) -> p[:A] * sin(p[:k] * x) * cos(p[:ω] * t),
    (64, 64);
    parameters = Dict(:A => 1.0, :k => 4.0, :ω => 1.0)
)
```
"""
mutable struct DeterministicForcing{T<:AbstractFloat, N} <: DeterministicForcingType
    forcing_function::Function
    field_size::NTuple{N, Int}
    cached_forcing::Array{T, N}
    parameters::Dict{Symbol, Any}
end

"""
    DeterministicForcing(forcing_function, field_size; parameters=Dict())

Create deterministic forcing.

## Arguments

- `forcing_function`: Function(x, y, ..., t, params) → forcing value
- `field_size`: Grid size
- `parameters`: Dictionary passed to forcing function
"""
function DeterministicForcing(
    forcing_function::Function,
    field_size::NTuple{N, Int};
    parameters::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N}

    cached_forcing = zeros(T, field_size)

    DeterministicForcing{T, N}(
        forcing_function,
        field_size,
        cached_forcing,
        parameters
    )
end

"""
    generate_forcing!(forcing::DeterministicForcing, grid, t::Real)

Evaluate deterministic forcing at time t on the given grid.
"""
function generate_forcing!(forcing::DeterministicForcing{T, N}, grid, t::Real) where {T, N}
    forcing.cached_forcing .= forcing.forcing_function(grid..., t, forcing.parameters)
    return forcing.cached_forcing
end

# ============================================================================
# Exports
# ============================================================================

export Forcing, StochasticForcingType, DeterministicForcingType
export StochasticForcing, DeterministicForcing
export generate_forcing!, apply_forcing!
export reset_forcing!, set_dt!
export store_prevsol!, work_stratonovich, work_ito
export mean_energy_injection_rate, instantaneous_power
export forcing_enstrophy_injection_rate
export get_forcing_spectrum, get_cached_forcing
