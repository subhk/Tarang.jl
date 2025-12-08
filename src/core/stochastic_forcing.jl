"""
Stochastic forcing implementation for Tarang.jl

Follows the approach from GeophysicalFlows.jl/FourierFlows.jl:
- Forcing is computed once per timestep (not per substep)
- Uses Stratonovich calculus for consistency
- Forcing spectrum determines spatial correlation
- White noise in time, spatially correlated

Reference:
- GeophysicalFlows.jl documentation: https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/
"""

using Random
using LinearAlgebra

# ============================================================================
# Abstract forcing type
# ============================================================================

abstract type Forcing end
abstract type StochasticForcingType <: Forcing end
abstract type DeterministicForcingType <: Forcing end

# ============================================================================
# Stochastic forcing configuration
# ============================================================================

"""
    StochasticForcing{T}

Stochastic forcing configuration following GeophysicalFlows.jl pattern.

The forcing ξ(x,t) is white in time but spatially correlated:
- ⟨ξ(x,t)⟩ = 0  (zero mean)
- ⟨ξ(x,t) ξ(x',t')⟩ = Q(x-x') δ(t-t')  (spatial correlation, white in time)

For numerical implementation using Stratonovich calculus:
    F = sqrt(Q / dt) * ξ
where ξ is complex white noise with unit variance.

# Fields
- `spectrum::AbstractArray{T}`: Forcing amplitude spectrum sqrt(Q(k))
- `forcing_rate::T`: Energy injection rate ε
- `k_forcing::T`: Central forcing wavenumber
- `dk_forcing::T`: Forcing bandwidth
- `dt::T`: Timestep (for proper scaling)
- `field_size::Tuple`: Size of the field to force
- `cached_forcing::AbstractArray{Complex{T}}`: Cached forcing (constant within timestep)
- `is_stochastic::Bool`: Flag for stochastic vs deterministic
- `rng::AbstractRNG`: Random number generator
- `last_update_time::T`: Time when forcing was last updated
"""
mutable struct StochasticForcing{T<:AbstractFloat} <: StochasticForcingType
    spectrum::AbstractArray{T}
    forcing_rate::T
    k_forcing::T
    dk_forcing::T
    dt::T
    field_size::Tuple
    cached_forcing::AbstractArray{Complex{T}}
    is_stochastic::Bool
    rng::AbstractRNG
    last_update_time::T
end

"""
    StochasticForcing(; field_size, forcing_rate=1.0, k_forcing=4.0, dk_forcing=2.0,
                       dt=0.01, is_stochastic=true, spectrum_type=:ring, rng=Random.GLOBAL_RNG)

Create a stochastic forcing configuration.

# Keyword Arguments
- `field_size`: Size of the field (e.g., (Nx, Ny) for 2D)
- `forcing_rate`: Energy injection rate ε
- `k_forcing`: Central forcing wavenumber
- `dk_forcing`: Bandwidth around forcing wavenumber
- `dt`: Timestep for proper Stratonovich scaling
- `is_stochastic`: If true, forcing is stochastic; if false, deterministic
- `spectrum_type`: Type of forcing spectrum (:ring, :isotropic, :bandlimited, :custom)
- `rng`: Random number generator for reproducibility

# Example
```julia
forcing = StochasticForcing(
    field_size=(64, 64),
    forcing_rate=0.1,
    k_forcing=4.0,
    dk_forcing=2.0,
    dt=0.001,
    is_stochastic=true
)
```
"""
function StochasticForcing(;
    field_size::Tuple,
    forcing_rate::Real=1.0,
    k_forcing::Real=4.0,
    dk_forcing::Real=2.0,
    dt::Real=0.01,
    is_stochastic::Bool=true,
    spectrum_type::Symbol=:ring,
    spectrum::Union{Nothing, AbstractArray}=nothing,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    dtype::Type{T}=Float64
) where T<:AbstractFloat

    # Compute forcing spectrum based on type
    if spectrum !== nothing
        forcing_spectrum = convert(Array{T}, spectrum)
    else
        forcing_spectrum = compute_forcing_spectrum(
            field_size, k_forcing, dk_forcing, forcing_rate, spectrum_type, dtype
        )
    end

    # Pre-allocate cached forcing array
    cached_forcing = zeros(Complex{T}, field_size)

    StochasticForcing{T}(
        forcing_spectrum,
        T(forcing_rate),
        T(k_forcing),
        T(dk_forcing),
        T(dt),
        field_size,
        cached_forcing,
        is_stochastic,
        rng,
        T(-Inf)  # Initialize to -Inf so first call always updates
    )
end

"""
    compute_forcing_spectrum(field_size, k_forcing, dk_forcing, forcing_rate, spectrum_type, dtype)

Compute the forcing spectrum sqrt(Q(k)) for different spectrum types.
"""
function compute_forcing_spectrum(
    field_size::Tuple,
    k_forcing::Real,
    dk_forcing::Real,
    forcing_rate::Real,
    spectrum_type::Symbol,
    dtype::Type{T}
) where T<:AbstractFloat

    ndim = length(field_size)

    if ndim == 1
        return _spectrum_1d(field_size[1], k_forcing, dk_forcing, forcing_rate, spectrum_type, dtype)
    elseif ndim == 2
        return _spectrum_2d(field_size, k_forcing, dk_forcing, forcing_rate, spectrum_type, dtype)
    elseif ndim == 3
        return _spectrum_3d(field_size, k_forcing, dk_forcing, forcing_rate, spectrum_type, dtype)
    else
        error("Unsupported dimension: $ndim")
    end
end

function _spectrum_1d(N, kf, dk, ε, spectrum_type, dtype::Type{T}) where T
    spectrum = zeros(T, N)
    kmax = N ÷ 2

    for i in 1:N
        k = i <= kmax ? i - 1 : i - N - 1
        kmag = abs(k)

        if spectrum_type == :ring || spectrum_type == :bandlimited
            # Band-limited forcing around kf
            if abs(kmag - kf) < dk
                spectrum[i] = sqrt(ε / (2 * dk))
            end
        elseif spectrum_type == :isotropic
            # Isotropic forcing with Gaussian envelope
            spectrum[i] = sqrt(ε) * exp(-((kmag - kf)^2) / (2 * dk^2))
        end
    end

    return spectrum
end

function _spectrum_2d(field_size, kf, dk, ε, spectrum_type, dtype::Type{T}) where T
    Nx, Ny = field_size
    spectrum = zeros(T, Nx, Ny)

    kx_max = Nx ÷ 2
    ky_max = Ny ÷ 2

    for j in 1:Ny
        for i in 1:Nx
            kx = i <= kx_max ? i - 1 : i - Nx - 1
            ky = j <= ky_max ? j - 1 : j - Ny - 1
            kmag = sqrt(kx^2 + ky^2)

            if spectrum_type == :ring
                # Ring forcing: force modes in annulus kf-dk < |k| < kf+dk
                if abs(kmag - kf) < dk && kmag > 0
                    # Normalize by number of modes in the ring
                    spectrum[i, j] = sqrt(ε)
                end
            elseif spectrum_type == :bandlimited
                # Same as ring but with sharper cutoff
                if abs(kmag - kf) < dk && kmag > 0
                    spectrum[i, j] = sqrt(ε / dk)
                end
            elseif spectrum_type == :isotropic
                # Gaussian envelope around kf
                if kmag > 0
                    spectrum[i, j] = sqrt(ε) * exp(-((kmag - kf)^2) / (2 * dk^2))
                end
            elseif spectrum_type == :kolmogorov
                # Large-scale forcing with k^0 spectrum
                if kmag > 0 && kmag < kf + dk
                    spectrum[i, j] = sqrt(ε / kf)
                end
            end
        end
    end

    # Normalize so total energy injection rate is ε
    norm_factor = sqrt(sum(spectrum.^2))
    if norm_factor > 0
        spectrum ./= norm_factor
        spectrum .*= sqrt(ε)
    end

    return spectrum
end

function _spectrum_3d(field_size, kf, dk, ε, spectrum_type, dtype::Type{T}) where T
    Nx, Ny, Nz = field_size
    spectrum = zeros(T, Nx, Ny, Nz)

    kx_max = Nx ÷ 2
    ky_max = Ny ÷ 2
    kz_max = Nz ÷ 2

    for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                kx = i <= kx_max ? i - 1 : i - Nx - 1
                ky = j <= ky_max ? j - 1 : j - Ny - 1
                kz = k <= kz_max ? k - 1 : k - Nz - 1
                kmag = sqrt(kx^2 + ky^2 + kz^2)

                if spectrum_type == :ring || spectrum_type == :bandlimited
                    if abs(kmag - kf) < dk && kmag > 0
                        spectrum[i, j, k] = sqrt(ε)
                    end
                elseif spectrum_type == :isotropic
                    if kmag > 0
                        spectrum[i, j, k] = sqrt(ε) * exp(-((kmag - kf)^2) / (2 * dk^2))
                    end
                end
            end
        end
    end

    # Normalize
    norm_factor = sqrt(sum(spectrum.^2))
    if norm_factor > 0
        spectrum ./= norm_factor
        spectrum .*= sqrt(ε)
    end

    return spectrum
end

# ============================================================================
# Forcing generation
# ============================================================================

"""
    generate_forcing!(forcing::StochasticForcing, current_time::Real)

Generate a new forcing realization. This should be called ONCE at the beginning
of each timestep, NOT at each substep.

For stochastic forcing, generates white noise scaled by the spectrum.
For deterministic forcing, returns the cached forcing unchanged.

Following GeophysicalFlows.jl/FourierFlows.jl pattern:
    F = sqrt(spectrum / dt) * exp(2πi * rand())

# Arguments
- `forcing`: StochasticForcing configuration
- `current_time`: Current simulation time

# Returns
The cached forcing array (modified in-place for stochastic forcing).
"""
function generate_forcing!(forcing::StochasticForcing{T}, current_time::Real) where T
    # Only update if time has changed (prevents updates during substeps)
    if current_time ≈ forcing.last_update_time
        return forcing.cached_forcing
    end

    if forcing.is_stochastic
        # Generate complex white noise with unit variance
        # ξ = exp(2πi * rand()) has |ξ| = 1 and random phase
        ξ = exp.(2π * im .* rand(forcing.rng, T, forcing.field_size))

        # Scale by spectrum and timestep (Stratonovich scaling)
        # F = sqrt(Q / dt) * ξ
        @. forcing.cached_forcing = forcing.spectrum * ξ / sqrt(forcing.dt)

        # Ensure zero mean (remove k=0 mode)
        forcing.cached_forcing[1] = zero(Complex{T})
        if ndims(forcing.cached_forcing) >= 2
            forcing.cached_forcing[1, 1] = zero(Complex{T})
        end
        if ndims(forcing.cached_forcing) >= 3
            forcing.cached_forcing[1, 1, 1] = zero(Complex{T})
        end
    end

    forcing.last_update_time = T(current_time)
    return forcing.cached_forcing
end

"""
    generate_forcing!(forcing::StochasticForcing, current_time::Real, substep::Int)

Generate forcing with substep awareness. Only generates new forcing on substep 1.

This is the key function that ensures forcing stays constant within a timestep:
- substep == 1: Generate new forcing
- substep > 1: Return cached forcing

# Arguments
- `forcing`: StochasticForcing configuration
- `current_time`: Current simulation time (at start of timestep)
- `substep`: Current substep number (1, 2, 3, ...)
"""
function generate_forcing!(forcing::StochasticForcing{T}, current_time::Real, substep::Int) where T
    if substep == 1
        # First substep: generate new forcing
        return generate_forcing!(forcing, current_time)
    else
        # Subsequent substeps: return cached forcing (no update)
        return forcing.cached_forcing
    end
end

"""
    reset_forcing!(forcing::StochasticForcing)

Reset the forcing cache. Call this to force regeneration on next access.
"""
function reset_forcing!(forcing::StochasticForcing{T}) where T
    forcing.last_update_time = T(-Inf)
    fill!(forcing.cached_forcing, zero(Complex{T}))
end

"""
    set_dt!(forcing::StochasticForcing, dt::Real)

Update the timestep used for Stratonovich scaling.
Should be called if dt changes during simulation.
"""
function set_dt!(forcing::StochasticForcing{T}, dt::Real) where T
    forcing.dt = T(dt)
end

# ============================================================================
# Forcing application to fields
# ============================================================================

"""
    apply_forcing!(field::AbstractArray, forcing::StochasticForcing, current_time::Real)

Apply stochastic forcing to a field (in spectral space).

# Arguments
- `field`: Field array in spectral space (modified in-place)
- `forcing`: StochasticForcing configuration
- `current_time`: Current simulation time
"""
function apply_forcing!(field::AbstractArray{T}, forcing::StochasticForcing,
                        current_time::Real) where T<:Complex
    F = generate_forcing!(forcing, current_time)
    field .+= F
    return field
end

"""
    apply_forcing!(field::AbstractArray, forcing::StochasticForcing,
                   current_time::Real, substep::Int)

Apply stochastic forcing to a field with substep awareness.
"""
function apply_forcing!(field::AbstractArray{T}, forcing::StochasticForcing,
                        current_time::Real, substep::Int) where T<:Complex
    F = generate_forcing!(forcing, current_time, substep)
    field .+= F
    return field
end

"""
    get_forcing_real(forcing::StochasticForcing)

Get the forcing in real space (for visualization or diagnostics).
Requires FFTW.
"""
function get_forcing_real(forcing::StochasticForcing{T}) where T
    # This requires FFTW - we'll return the cached forcing as-is
    # Users should transform it themselves if needed
    return real.(forcing.cached_forcing)
end

# ============================================================================
# Energy injection diagnostics
# ============================================================================

"""
    energy_injection_rate(forcing::StochasticForcing)

Compute the mean energy injection rate ⟨εf⟩.

For Stratonovich forcing: ⟨εf⟩ = ∑_k |F_k|² / 2
"""
function energy_injection_rate(forcing::StochasticForcing{T}) where T
    # Mean energy injection rate is the sum of squared spectrum
    return sum(forcing.spectrum.^2) / 2
end

"""
    instantaneous_power(forcing::StochasticForcing, field::AbstractArray)

Compute instantaneous power input P = -⟨ψ · ξ⟩ (Stratonovich).

# Arguments
- `forcing`: StochasticForcing configuration
- `field`: Field array in spectral space (e.g., streamfunction ψ̂)
"""
function instantaneous_power(forcing::StochasticForcing{T}, field::AbstractArray) where T
    # P = -Re(⟨ψ̂* · F̂⟩)
    return -real(sum(conj.(field) .* forcing.cached_forcing)) / prod(forcing.field_size)
end

# ============================================================================
# Deterministic forcing (for comparison)
# ============================================================================

"""
    DeterministicForcing{T}

Deterministic (non-stochastic) forcing configuration.
"""
mutable struct DeterministicForcing{T<:AbstractFloat} <: DeterministicForcingType
    forcing_function::Function
    field_size::Tuple
    cached_forcing::AbstractArray{T}
    parameters::Dict{Symbol, Any}
end

"""
    DeterministicForcing(forcing_function, field_size; parameters=Dict())

Create a deterministic forcing configuration.

# Arguments
- `forcing_function`: Function(x, y, t, params) → forcing value
- `field_size`: Size of the field
- `parameters`: Dictionary of parameters passed to forcing function
"""
function DeterministicForcing(
    forcing_function::Function,
    field_size::Tuple;
    parameters::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    dtype::Type{T}=Float64
) where T<:AbstractFloat

    cached_forcing = zeros(T, field_size)

    DeterministicForcing{T}(
        forcing_function,
        field_size,
        cached_forcing,
        parameters
    )
end

"""
    generate_forcing!(forcing::DeterministicForcing, grid, current_time::Real)

Evaluate deterministic forcing at current time.
"""
function generate_forcing!(forcing::DeterministicForcing{T}, grid, current_time::Real) where T
    # Call the forcing function with grid and time
    forcing.cached_forcing .= forcing.forcing_function(grid..., current_time, forcing.parameters)
    return forcing.cached_forcing
end

# ============================================================================
# Exports
# ============================================================================

export Forcing, StochasticForcingType, DeterministicForcingType
export StochasticForcing, DeterministicForcing
export generate_forcing!, reset_forcing!, set_dt!
export apply_forcing!, get_forcing_real
export energy_injection_rate, instantaneous_power
export compute_forcing_spectrum
