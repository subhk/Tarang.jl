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
    W = Re⟨[ψ(tⱼ) + ψ(tⱼ₊₁)]/2 · ΔF*⟩

where ΔF is the forcing increment over dt.

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
    StochasticForcing{T, N, A<:AbstractArray}

Stochastic forcing in Fourier space with white-noise temporal correlation.
Supports both CPU and GPU architectures.

## Mathematical Properties

The forcing F̂(k,t) satisfies:
- ⟨F̂(k,t)⟩ = 0                                          (zero mean)
- ⟨F̂(k,t) F̂*(k',t')⟩ = Q̂(k) δ(k-k') δ(t-t')/(dt)     (white noise)

## Implementation

At each timestep, the forcing is computed as:
    F̂(k) = √(Q̂(k)) · ξ(k) / √(dt)

where ξ(k) is complex white noise with |ξ| = 1 and random phase.

## Fields

- `forcing_spectrum::A`: √Q̂(k) - square root of power spectrum
- `energy_injection_rate::T`: Target energy injection rate ε
- `k_forcing::T`: Central forcing wavenumber k_f
- `dk_forcing::T`: Forcing bandwidth δ_f
- `dt::T`: Current timestep (for proper scaling)
- `domain_size::NTuple{N,T}`: Domain size (Lx, Ly, ...)
- `field_size::NTuple{N,Int}`: Grid size (Nx, Ny, ...)
- `wavenumbers::NTuple{N,Vector{T}}`: Wavenumber arrays (kx, ky, ...)
- `cached_forcing::AbstractArray{Complex{T},N}`: Cached forcing (constant within timestep)
- `prevsol::Union{Nothing,AbstractArray{Complex{T},N}}`: Previous solution (for Stratonovich work)
- `rng::AbstractRNG`: Random number generator (CPU-side)
- `random_phases::AbstractArray{T,N}`: Pre-allocated random phase buffer
- `last_update_time::T`: Time of last forcing update
- `spectrum_type::Symbol`: Type of forcing spectrum
- `architecture::AbstractArchitecture`: CPU() or GPU() architecture
"""
mutable struct StochasticForcing{T<:AbstractFloat, N, A<:AbstractArray{T,N}, CA<:AbstractArray{Complex{T},N}} <: StochasticForcingType
    forcing_spectrum::A                     # √Q̂(k) - amplitude spectrum
    energy_injection_rate::T                # Target ε
    k_forcing::T                            # Central wavenumber
    dk_forcing::T                           # Bandwidth
    dt::T                                   # Timestep
    domain_size::NTuple{N, T}               # Domain (Lx, Ly, ...)
    field_size::NTuple{N, Int}              # Grid (Nx, Ny, ...)
    wavenumbers::NTuple{N, Vector{T}}       # (kx, ky, ...) - kept on CPU for setup
    cached_forcing::CA                      # Cached F̂
    prevsol::Union{Nothing, CA}             # For Stratonovich work
    rng::AbstractRNG                        # CPU-side RNG
    random_phases::A                        # Pre-allocated random phase buffer
    last_update_time::T
    spectrum_type::Symbol
    architecture::AbstractArchitecture
end

function Base.getproperty(forcing::StochasticForcing, name::Symbol)
    if name === :forcing_rate
        return getfield(forcing, :energy_injection_rate)
    elseif name === :spectrum
        return getfield(forcing, :forcing_spectrum)
    elseif name === :is_stochastic
        return true
    elseif name === :is_gpu
        return is_gpu(getfield(forcing, :architecture))
    end
    return getfield(forcing, name)
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
        dtype = Float64,
        architecture = CPU()
    )

Create a stochastic forcing configuration that works on CPU or GPU.

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
- `rng::AbstractRNG`: Random number generator (CPU-side, phases transferred to GPU)
- `dtype::Type`: Floating point type (default: Float64)
- `architecture::AbstractArchitecture`: CPU() or GPU() (default: CPU())

## Example

```julia
# Create ring forcing for 2D turbulence on CPU
forcing = StochasticForcing(
    field_size = (256, 256),
    domain_size = (2π, 2π),
    energy_injection_rate = 0.1,
    k_forcing = 10.0,   # Force at |k| ≈ 10
    dk_forcing = 2.0,   # Bandwidth
    dt = 0.001
)

# Create ring forcing for GPU
using CUDA
forcing_gpu = StochasticForcing(
    field_size = (256, 256),
    energy_injection_rate = 0.1,
    k_forcing = 10.0,
    architecture = GPU()
)

# In your simulation loop:
generate_forcing!(forcing, t, substep)
```
"""
function StochasticForcing(;
    field_size::NTuple{N, Int},
    domain_size::Union{Nothing, NTuple{N, Real}} = nothing,
    energy_injection_rate::Real = 1.0,
    forcing_rate::Union{Nothing, Real} = nothing,
    k_forcing::Real = 4.0,
    dk_forcing::Real = 1.0,
    dt::Real = 0.01,
    spectrum_type::Symbol = :ring,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    dtype::Type{T} = Float64,
    architecture::AbstractArchitecture = CPU()
) where {T<:AbstractFloat, N}

    # Compute default domain_size if not provided
    domain_size = domain_size === nothing ? ntuple(i -> T(2π), N) : T.(domain_size)

    # Use forcing_rate if provided, otherwise use energy_injection_rate
    # Only warn if both appear to be explicitly set (forcing_rate provided AND energy_injection_rate differs from default)
    default_energy_injection_rate = 1.0
    energy = forcing_rate === nothing ? energy_injection_rate : forcing_rate
    if forcing_rate !== nothing && !isapprox(energy_injection_rate, default_energy_injection_rate) && !isapprox(forcing_rate, energy_injection_rate)
        @warn "Both forcing_rate and energy_injection_rate were provided; using forcing_rate"
    end

    spectrum_type = _normalize_spectrum_type(spectrum_type)

    # Build wavenumber arrays (always on CPU for setup)
    wavenumbers = build_wavenumbers(field_size, domain_size, dtype)

    # Compute the forcing spectrum √Q̂(k) on CPU first
    forcing_spectrum_cpu = compute_forcing_spectrum(
        wavenumbers, k_forcing, dk_forcing, energy,
        domain_size, spectrum_type, dtype
    )

    # Move spectrum to target architecture
    forcing_spectrum = on_architecture(architecture, forcing_spectrum_cpu)

    # Allocate cached forcing array on target architecture
    cached_forcing = zeros(architecture, Complex{T}, field_size...)

    # Pre-allocate random phase buffer on target architecture
    random_phases = zeros(architecture, T, field_size...)

    # Previous solution for Stratonovich work calculation
    prevsol = nothing

    # Get the concrete array types for the struct
    A = typeof(forcing_spectrum)
    CA = typeof(cached_forcing)

    StochasticForcing{T, N, A, CA}(
        forcing_spectrum,
        T(energy),
        T(k_forcing),
        T(dk_forcing),
        T(dt),
        T.(domain_size),
        field_size,
        wavenumbers,
        cached_forcing,
        prevsol,
        rng,
        random_phases,
        T(-Inf),  # Initialize to -Inf so first call always updates
        spectrum_type,
        architecture
    )
end

function _normalize_spectrum_type(spectrum_type::Symbol)
    if spectrum_type === :isotropic
        return :ring
    elseif spectrum_type === :bandlimited
        return :band
    end
    return spectrum_type
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

        # Standard FFT wavenumber ordering: 0, 1, ..., n/2, -(n/2-1), ..., -1
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

    spectrum_type = _normalize_spectrum_type(spectrum_type)
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
    spectrum[ntuple(_ -> 1, N)...] = zero(T)

    # Normalize to achieve target energy injection rate ε
    # Energy injection rate: ε = ∑_k Q̂(k) / (2 * domain_area)
    domain_area = prod(T.(domain_size))

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

Generate stochastic forcing at time t. Works on both CPU and GPU.

## Key Points

1. **Forcing is regenerated only on substep 1** - ensures forcing stays constant
   within a timestep for IMEX and multi-stage methods.

2. **Scaling**: F̂(k) = √Q̂(k) · exp(2πi·rand) / √dt

3. **Zero mean**: The k=0 mode is always set to zero.

4. **Hermitian symmetry**: F̂(-k) = F̂(k)* so that ifft produces real fields.

5. **GPU support**: Random phases are generated on CPU and transferred to GPU,
   then combined with the spectrum using broadcasted operations.

## Arguments

- `forcing`: StochasticForcing configuration
- `t`: Current simulation time
- `substep`: Current substep (1 for first substep)

## Returns

The cached forcing array (modified in-place).
"""
function generate_forcing!(forcing::StochasticForcing{T, N, A, CA}, t::Real, substep::Int=1) where {T, N, A, CA}
    # Only update forcing on first substep
    if substep > 1
        return forcing.cached_forcing
    end

    # Check if we've already updated at this time
    if t ≈ forcing.last_update_time
        return forcing.cached_forcing
    end

    # Generate forcing using GPU-compatible method
    _generate_forcing_gpu_compatible!(forcing)

    forcing.last_update_time = T(t)
    return forcing.cached_forcing
end

"""
    _generate_forcing_gpu_compatible!(forcing)

GPU-compatible forcing generation using broadcasted operations.
Random phases are generated on CPU and transferred to GPU.
Hermitian symmetry is enforced using array operations.
"""
function _generate_forcing_gpu_compatible!(forcing::StochasticForcing{T, N, A, CA}) where {T, N, A, CA}
    sqrt_dt = sqrt(forcing.dt)
    field_size = forcing.field_size

    # Generate random phases on CPU
    phases_cpu = 2 * T(π) * rand(forcing.rng, T, field_size...)

    # Transfer to target architecture
    phases = on_architecture(forcing.architecture, phases_cpu)

    # Compute forcing: F = spectrum * exp(i*phase) / sqrt(dt)
    # Using broadcasting for GPU compatibility
    forcing.cached_forcing .= forcing.forcing_spectrum .* exp.(im .* phases) ./ sqrt_dt

    # Enforce Hermitian symmetry for real physical fields
    _enforce_hermitian_symmetry!(forcing.cached_forcing, forcing.architecture)

    # Enforce zero mean (k=0 mode)
    _set_zero_mode!(forcing.cached_forcing, forcing.architecture)

    return forcing.cached_forcing
end

"""
    _enforce_hermitian_symmetry!(data, arch)

Enforce Hermitian symmetry F̂(-k) = F̂(k)* for real physical fields.
For GPU, uses a CPU-based approach with data transfer.
"""
function _enforce_hermitian_symmetry!(data::AbstractArray{Complex{T}, N}, arch::CPU) where {T, N}
    # CPU implementation using scalar indexing
    if N == 1
        _enforce_hermitian_1d!(data)
    elseif N == 2
        _enforce_hermitian_2d!(data)
    elseif N == 3
        _enforce_hermitian_3d!(data)
    end
end

function _enforce_hermitian_symmetry!(data::AbstractArray{Complex{T}, N}, arch::GPU) where {T, N}
    # For GPU: transfer to CPU, enforce symmetry, transfer back
    # This is acceptable because forcing is only generated once per timestep
    data_cpu = Array(data)

    if N == 1
        _enforce_hermitian_1d!(data_cpu)
    elseif N == 2
        _enforce_hermitian_2d!(data_cpu)
    elseif N == 3
        _enforce_hermitian_3d!(data_cpu)
    end

    # Copy back to GPU
    copyto!(data, on_architecture(arch, data_cpu))
end

"""
    _enforce_hermitian_1d!(data)

Enforce F̂(-k) = F̂(k)* for 1D arrays.
"""
function _enforce_hermitian_1d!(data::AbstractVector{Complex{T}}) where T
    n = length(data)

    # DC mode (k=0) must be real
    data[1] = Complex{T}(real(data[1]), zero(T))

    # Nyquist mode (if even n) must be real
    if iseven(n)
        nyq = n ÷ 2 + 1
        data[nyq] = Complex{T}(real(data[nyq]), zero(T))
    end

    # Enforce conjugate symmetry: F(-k) = F(k)*
    last_positive = iseven(n) ? n ÷ 2 : (n + 1) ÷ 2
    for i in 2:last_positive
        j = n + 2 - i  # Corresponding negative frequency
        data[j] = conj(data[i])
    end
end

"""
    _enforce_hermitian_2d!(data)

Enforce F̂(-kx, -ky) = F̂(kx, ky)* for 2D arrays.
"""
function _enforce_hermitian_2d!(data::AbstractMatrix{Complex{T}}) where T
    nx, ny = size(data)

    for j in 1:ny
        for i in 1:nx
            # Find conjugate index
            ci = i == 1 ? 1 : nx + 2 - i
            cj = j == 1 ? 1 : ny + 2 - j

            # Only process if we haven't visited this pair yet
            if (i < ci) || (i == ci && j < cj)
                # Average the value and its conjugate for consistency
                avg = (data[i, j] + conj(data[ci, cj])) / 2
                data[i, j] = avg
                data[ci, cj] = conj(avg)
            elseif i == ci && j == cj
                # Self-conjugate modes must be real
                data[i, j] = Complex{T}(real(data[i, j]), zero(T))
            end
        end
    end
end

"""
    _enforce_hermitian_3d!(data)

Enforce F̂(-k) = F̂(k)* for 3D arrays.
"""
function _enforce_hermitian_3d!(data::AbstractArray{Complex{T}, 3}) where T
    nx, ny, nz = size(data)

    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                # Find conjugate index
                ci = i == 1 ? 1 : nx + 2 - i
                cj = j == 1 ? 1 : ny + 2 - j
                ck = k == 1 ? 1 : nz + 2 - k

                # Only process if we haven't visited this pair yet
                lin_idx = i + (j-1)*nx + (k-1)*nx*ny
                conj_lin_idx = ci + (cj-1)*nx + (ck-1)*nx*ny

                if lin_idx < conj_lin_idx
                    avg = (data[i, j, k] + conj(data[ci, cj, ck])) / 2
                    data[i, j, k] = avg
                    data[ci, cj, ck] = conj(avg)
                elseif lin_idx == conj_lin_idx
                    # Self-conjugate modes must be real
                    data[i, j, k] = Complex{T}(real(data[i, j, k]), zero(T))
                end
            end
        end
    end
end

"""
    _set_zero_mode!(data, arch)

Set the k=0 (DC) mode to zero. GPU-compatible.
"""
function _set_zero_mode!(data::AbstractArray{Complex{T}, N}, arch::CPU) where {T, N}
    data[1] = zero(Complex{T})
end

function _set_zero_mode!(data::AbstractArray{Complex{T}, N}, arch::GPU) where {T, N}
    # For GPU arrays, we need to use fill! with a view or copyto!
    # Using a scalar write that works on GPU via KernelAbstractions pattern
    fill!(view(data, 1:1), zero(Complex{T}))
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

Add stochastic forcing to the RHS in spectral space. Works on both CPU and GPU.

## Arguments

- `rhs`: Right-hand side array (modified in-place)
- `forcing`: StochasticForcing configuration
- `t`: Current simulation time
- `substep`: Current substep number
"""
function apply_forcing!(
    rhs::AbstractArray{Complex{T}, N},
    forcing::StochasticForcing{T, N, A, CA},
    t::Real,
    substep::Int=1
) where {T, N, A, CA}

    F = generate_forcing!(forcing, t, substep)
    F_view = _matched_forcing_view(forcing, size(rhs))
    if F_view === nothing
        throw(ArgumentError("Forcing size $(size(F)) does not match RHS size $(size(rhs))"))
    end

    # Use broadcasting for GPU compatibility
    rhs .+= F_view
    return rhs
end

"""
    store_prevsol!(forcing::StochasticForcing, sol::AbstractArray)

Store the current solution for Stratonovich work calculation.
Works on both CPU and GPU arrays.

Call this at the **beginning** of each timestep, before advancing.
"""
function store_prevsol!(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    if forcing.prevsol === nothing
        # Allocate on the same architecture as forcing
        forcing.prevsol = similar(forcing.cached_forcing)
    end
    # Use copyto! for GPU compatibility
    copyto!(forcing.prevsol, sol)
end

"""
    work_stratonovich(forcing::StochasticForcing, sol::AbstractArray)

Compute work done by forcing using Stratonovich interpretation.
Works on both CPU and GPU arrays.

## Formula

    W = Re⟨(ψⁿ + ψⁿ⁺¹)/2 · ΔF̂*⟩

where ΔF̂ = √Q̂ · ξ · √dt is the forcing increment over dt.

This correctly accounts for the correlation between forcing and response.

## Arguments

- `forcing`: StochasticForcing with prevsol stored
- `sol`: Current solution ψⁿ⁺¹

## Returns

Work done during this timestep (scalar, units of energy).
"""
function work_stratonovich(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    if forcing.prevsol === nothing
        return zero(T)
    end

    # Stratonovich work: W = Re⟨ψ_mid · ΔF̂*⟩ where ΔF̂ = F̂_stored · dt
    # Since F̂_stored = √Q̂ · ξ / √dt, we have ΔF̂ = √Q̂ · ξ · √dt
    domain_area = prod(forcing.domain_size)

    # Use broadcasting for GPU compatibility
    # Compute midpoint value and correlation
    ψ_mid = (forcing.prevsol .+ sol) ./ 2
    work_array = real.(ψ_mid .* conj.(forcing.cached_forcing))
    work = sum(work_array)

    # The cached_forcing stores F̂ = √Q̂ · ξ / √dt
    # The forcing increment is ΔF̂ = F̂ · dt = √Q̂ · ξ · √dt
    # Work = (1/A) · Re Σ ψ_mid · ΔF̂* = (dt/A) · Re Σ ψ_mid · F̂*
    return T(work * forcing.dt / domain_area)
end

"""
    work_ito(forcing::StochasticForcing, sol::AbstractArray)

Compute work done by forcing using Itô interpretation.
Works on both CPU and GPU arrays.

## Formula

    W_Itô = Re⟨ψⁿ · ΔF̂*⟩ + ε · dt

where ΔF̂ = √Q̂ · ξ · √dt is the forcing increment.

The drift correction ε · dt accounts for the Itô-Stratonovich conversion.
In Itô calculus, ψⁿ is independent of Fⁿ⁺¹, so ⟨ψⁿ · ΔF̂⟩ = 0.
The drift ensures ⟨W_Itô⟩ = ⟨W_Stratonovich⟩ = ε · dt.
"""
function work_ito(forcing::StochasticForcing{T, N, A, CA}, sol_prev::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    domain_area = prod(forcing.domain_size)

    # Itô work (uses previous solution, which is independent of current forcing)
    # Use broadcasting for GPU compatibility
    work_array = real.(sol_prev .* conj.(forcing.cached_forcing))
    work = sum(work_array)

    # The Itô integral has zero mean, so we add drift correction
    # to match Stratonovich mean: ⟨W_Itô⟩ = 0 + ε·dt = ε·dt
    drift = forcing.energy_injection_rate * forcing.dt

    # Work = (dt/A) · Re Σ ψ_prev · F̂* (same scaling as Stratonovich)
    return T(work * forcing.dt / domain_area + drift)
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

energy_injection_rate(forcing::StochasticForcing) = forcing.energy_injection_rate

function get_forcing_real(forcing::StochasticForcing)
    return real.(forcing.cached_forcing)
end

"""
    instantaneous_power(forcing::StochasticForcing, sol::AbstractArray)

Compute instantaneous power input P = Re⟨ψ · F̂*⟩ / A.
Works on both CPU and GPU arrays.

This is the correlation between the solution and forcing at a given instant.
For white noise forcing, the expected value depends on which solution is passed:
- ⟨P⟩ = 0 if sol is the solution BEFORE the forcing was applied (independent)
- ⟨P⟩ = ε if sol is the MIDPOINT (ψⁿ + ψⁿ⁺¹)/2
- ⟨P⟩ = 2ε if sol is the solution AFTER forcing (includes full response)

Note: For the Stratonovich-consistent time-averaged power over the timestep,
use `work_stratonovich(forcing, sol) / forcing.dt` instead.

## Returns

Instantaneous power (energy per unit time).
"""
function instantaneous_power(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    domain_area = prod(forcing.domain_size)

    # Use broadcasting for GPU compatibility
    power_array = real.(sol .* conj.(forcing.cached_forcing))
    power = sum(power_array)

    # P = (1/A) · Re Σ ψ · F̂* where F̂ = √Q̂ · ξ / √dt
    return T(power / domain_area)
end

"""
    forcing_enstrophy_injection_rate(forcing::StochasticForcing)

Compute mean enstrophy injection rate (for 2D turbulence).

    η = ∑_k |k|² Q̂(k) / (2 · domain_area)

Note: This assumes the forcing spectrum Q̂(k) corresponds to direct field forcing.
For vorticity forcing, enstrophy injection is simply ε. For streamfunction forcing,
this formula gives the enstrophy injection rate.
"""
function forcing_enstrophy_injection_rate(forcing::StochasticForcing{T, N, A, CA}) where {T, N, A, CA}
    if N != 2
        @warn "Enstrophy injection rate is only meaningful for 2D"
        return zero(T)
    end

    domain_area = prod(forcing.domain_size)
    kx, ky = forcing.wavenumbers

    # Build k² array on CPU (wavenumbers are always on CPU)
    k2_array = zeros(T, forcing.field_size)
    for j in eachindex(ky)
        for i in eachindex(kx)
            k2_array[i, j] = kx[i]^2 + ky[j]^2
        end
    end

    # Move to target architecture and compute
    k2_on_arch = on_architecture(forcing.architecture, k2_array)

    # Get spectrum on CPU for computation (spectrum might be on GPU)
    spectrum_cpu = Array(forcing.forcing_spectrum)

    # Compute enstrophy injection rate
    η = zero(T)
    for j in eachindex(ky)
        for i in eachindex(kx)
            k2 = kx[i]^2 + ky[j]^2
            η += k2 * spectrum_cpu[i, j]^2
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
function set_dt!(forcing::StochasticForcing{T, N, A, CA}, dt::Real) where {T, N, A, CA}
    forcing.dt = T(dt)
end

"""
    reset_forcing!(forcing::StochasticForcing)

Reset the forcing cache, causing regeneration on next call.
Works on both CPU and GPU.
"""
function reset_forcing!(forcing::StochasticForcing{T, N, A, CA}) where {T, N, A, CA}
    forcing.last_update_time = T(-Inf)
    # Use fill! which works on both CPU and GPU arrays
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
# Internal helpers
# ============================================================================

function _matched_forcing_view(forcing::StochasticForcing{T, N, A, CA},
                               target_shape::NTuple{N, Int}) where {T, N, A, CA}
    forcing_shape = size(forcing.cached_forcing)
    if forcing_shape == target_shape
        return forcing.cached_forcing
    end

    ranges = Vector{UnitRange{Int}}(undef, N)
    for d in 1:N
        if forcing_shape[d] == target_shape[d]
            ranges[d] = 1:target_shape[d]
        elseif forcing_shape[d] == 2 * (target_shape[d] - 1) ||
               forcing_shape[d] == 2 * target_shape[d] - 1
            ranges[d] = 1:target_shape[d]
        else
            return nothing
        end
    end

    # view() works on both CPU and GPU arrays
    return view(forcing.cached_forcing, Tuple(ranges)...)
end

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
mutable struct DeterministicForcing{T<:AbstractFloat, N, A<:AbstractArray{T, N}} <: DeterministicForcingType
    forcing_function::Function
    field_size::NTuple{N, Int}
    cached_forcing::A
    parameters::Dict{Symbol, Any}
    architecture::AbstractArchitecture
end

function Base.getproperty(forcing::DeterministicForcing, name::Symbol)
    if name === :is_gpu
        return is_gpu(getfield(forcing, :architecture))
    end
    return getfield(forcing, name)
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
    parameters::AbstractDict{Symbol} = Dict{Symbol, Any}(),
    dtype::Type{T} = Float64,
    architecture::AbstractArchitecture = CPU()
) where {T<:AbstractFloat, N}

    cached_forcing = zeros(architecture, T, field_size...)

    # Convert to Dict{Symbol, Any} for storage
    params = Dict{Symbol, Any}(k => v for (k, v) in parameters)

    DeterministicForcing{T, N, typeof(cached_forcing)}(
        forcing_function,
        field_size,
        cached_forcing,
        params,
        architecture
    )
end

"""
    generate_forcing!(forcing::DeterministicForcing, grid, t::Real)

Evaluate deterministic forcing at time t on the given grid.
"""
function generate_forcing!(forcing::DeterministicForcing{T, N, A}, grid, t::Real) where {T, N, A}
    values = forcing.forcing_function(grid..., t, forcing.parameters)

    if !(values isa AbstractArray)
        throw(ArgumentError("Deterministic forcing function must return an array"))
    end

    if size(values) != forcing.field_size
        throw(ArgumentError("Deterministic forcing output size $(size(values)) does not match field size $(forcing.field_size)"))
    end

    coerced = eltype(values) <: T ? values : T.(values)
    data_on_arch = on_architecture(forcing.architecture, coerced)
    copyto!(forcing.cached_forcing, data_on_arch)
    return forcing.cached_forcing
end

# ============================================================================
# Exports
# ============================================================================

# Export abstract types
export Forcing, StochasticForcingType, DeterministicForcingType

# Export concrete forcing types
export StochasticForcing, DeterministicForcing

# Export forcing generation and application
export generate_forcing!, apply_forcing!
export reset_forcing!, set_dt!

# Export work calculation (Stratonovich calculus)
export store_prevsol!, work_stratonovich, work_ito

# Export diagnostics
export mean_energy_injection_rate, energy_injection_rate, instantaneous_power
export forcing_enstrophy_injection_rate
export get_forcing_spectrum, get_cached_forcing, get_forcing_real

# Export spectrum building utilities (for custom forcing spectra)
export build_wavenumbers, compute_forcing_spectrum
