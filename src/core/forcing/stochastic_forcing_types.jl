"""
    stochastic_forcing_types

Forcing types and construction: the `StochasticForcing` / `SeparableStochasticForcing`
structs, their keyword constructors, wavenumber grids and forcing spectra.

Split out of the former 1767-line `stochastic_forcing.jl`; the sections are
unchanged, only relocated.
"""

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

    F̂(k) = √(Q̂(k)/dt) · exp(i · φ),  φ ∼ Uniform[0, 2π)

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
# LinearAlgebra already in Tarang.jl

# ============================================================================
# Abstract forcing types
# ============================================================================

# NOTE: `abstract type Forcing end` is defined in problems/problem_types.jl
# (which loads earlier) so IVP can type its stochastic_forcings dict.

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
    StochasticForcing{T, N, A<:AbstractArray{T,N}, CA<:AbstractArray{Complex{T},N}}

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
- `injection_metric::Symbol`: Quadratic invariant used to normalize ε
- `k_forcing::T`: Central forcing wavenumber k_f
- `dk_forcing::T`: Forcing bandwidth δ_f
- `dt::T`: Current timestep (for proper scaling)
- `domain_size::NTuple{N,T}`: Domain size (Lx, Ly, ...)
- `field_size::NTuple{N,Int}`: Grid size (Nx, Ny, ...)
- `wavenumbers::NTuple{N,Vector{T}}`: Wavenumber arrays (kx, ky, ...)
- `cached_forcing::AbstractArray{Complex{T},N}`: Cached forcing (constant within timestep)
- `prevsol::Union{Nothing,AbstractArray{Complex{T},N}}`: Previous solution (for Stratonovich work)
- `rng::AbstractRNG`: Random number generator (default: fresh MersenneTwister per instance for thread/parallel safety)
- `random_phases::AbstractArray{T,N}`: Pre-allocated random phase buffer (on target architecture)
- `diagnostic_weights`: Cached backend weights for the active GPU diagnostic layout
- `last_update_time::T`: Time of last forcing update
- `spectrum_type::Symbol`: Type of forcing spectrum
- `enforce_hermitian::Bool`: Enforce Hermitian symmetry for real-valued fields
- `architecture::AbstractArchitecture`: CPU() or GPU() architecture
"""
mutable struct StochasticForcing{T<:AbstractFloat, N, A<:AbstractArray{T,N}, CA<:AbstractArray{Complex{T},N}} <: StochasticForcingType
    forcing_spectrum::A                     # √Q̂(k) - amplitude spectrum
    energy_injection_rate::T                # Target ε
    injection_metric::Symbol                # :direct or :vorticity_kinetic
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
    diagnostic_weights::Union{Nothing, A}   # Cached backend weights for GPU reductions
    diagnostic_global_shape::NTuple{N, Int}
    diagnostic_local_ranges::NTuple{N, UnitRange{Int}}
    diagnostic_metric::Symbol
    last_update_time::T
    spectrum_type::Symbol
    enforce_hermitian::Bool                 # Enforce Hermitian symmetry for real fields
    architecture::AbstractArchitecture
end

"""
    SeparableStochasticForcing

Stochastic forcing for a Fourier product domain with one trailing Chebyshev
dimension. Randomness and spectral localization live only in the Fourier
dimensions; `chebyshev_profile` supplies the fixed Chebyshev dependence.
"""
mutable struct SeparableStochasticForcing{
    T<:AbstractFloat, NF, N,
    A<:AbstractArray{T,NF},
    FCA<:AbstractArray{Complex{T},NF},
    P<:AbstractVector{T},
    HP<:Array{T,NF},
    CA<:AbstractArray{Complex{T},N},
    FRV<:AbstractArray{Complex{T},N},
    PV<:AbstractArray{T,N},
} <: StochasticForcingType
    forcing_spectrum::A
    energy_injection_rate::T
    injection_metric::Symbol
    k_forcing::T
    dk_forcing::T
    dt::T
    domain_size::NTuple{NF,T}
    fourier_size::NTuple{NF,Int}
    field_size::NTuple{N,Int}
    wavenumbers::NTuple{NF,Vector{T}}
    chebyshev_basis::ChebyshevT
    chebyshev_profile::P
    cached_forcing::CA
    prevsol::Union{Nothing,CA}
    rng::AbstractRNG
    random_phases::A
    random_phases_host::HP
    fourier_realization::FCA
    fourier_outer_view::FRV
    profile_outer_view::PV
    last_update_time::T
    spectrum_type::Symbol
    enforce_hermitian::Bool
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

function Base.getproperty(forcing::SeparableStochasticForcing, name::Symbol)
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
        injection_metric = :direct,
        k_forcing = 4.0,
        dk_forcing = 1.0,
        dt = 0.01,
        spectrum_type = :ring,
        rng = Random.MersenneTwister(),
        dtype = Float64,
        enforce_hermitian = true,
        architecture = CPU()
    )

Create a stochastic forcing configuration that works on CPU or GPU.

## Arguments

- `field_size::NTuple{N,Int}`: Grid size (Nx, Ny, ...)
- `domain_size::NTuple{N,Real}`: Domain size (Lx, Ly, ...), default 2π in each direction
- `energy_injection_rate::Real`: Target energy injection rate ε (default: 1.0)
- `injection_metric::Symbol`: Quadratic invariant used to normalize ε. `:direct`
  uses unit Fourier weight; `:vorticity_kinetic` uses `1/|k|²` for nonzero
  modes (default: `:direct`). Use the latter when forcing 2-D vorticity and ε
  denotes kinetic-energy injection.
- `k_forcing::Real`: Central forcing wavenumber k_f (default: 4.0)
- `dk_forcing::Real`: Forcing bandwidth δ_f (default: 1.0)
- `dt::Real`: Initial timestep (default: 0.01)
- `spectrum_type::Symbol`: Spectrum shape (default: :ring)
    - `:ring` - Gaussian ring in wavenumber space
    - `:band` - Sharp band |k| ∈ [k_f - δ_f, k_f + δ_f]
    - `:lowk` - Low wavenumber forcing |k| < k_f
    - `:kolmogorov` - Forcing at large scales
- `rng::AbstractRNG`: Random number generator (default: fresh MersenneTwister per instance for thread/parallel safety)
- `dtype::Type`: Floating point type (default: Float64)
- `enforce_hermitian::Bool`: Enforce Hermitian symmetry (set false for complex-valued fields)
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
    injection_metric::Symbol = :direct,
    k_forcing::Real = 4.0,
    dk_forcing::Real = 1.0,
    dt::Real = 0.01,
    spectrum_type::Symbol = :ring,
    rng::AbstractRNG = Random.MersenneTwister(),
    dtype::Type{T} = Float64,
    enforce_hermitian::Bool = true,
    architecture::AbstractArchitecture = CPU()
) where {T<:AbstractFloat, N}

    # Normalize aliases before validation so :isotropic/:bandlimited are caught
    spectrum_type = _normalize_spectrum_type(spectrum_type)

    _validate_injection_metric(injection_metric)

    # Validate dk_forcing for spectrum types that use it
    if dk_forcing <= 0 && spectrum_type in (:ring, :kolmogorov)
        throw(ArgumentError("dk_forcing must be positive for spectrum_type=:$spectrum_type (got dk_forcing=$dk_forcing). " *
                            "Use spectrum_type=:band for delta-function-like forcing at k_f."))
    end
    if dk_forcing <= 0 && spectrum_type == :band
        throw(ArgumentError("dk_forcing must be positive for spectrum_type=:band (got dk_forcing=$dk_forcing). " *
                            "A non-positive bandwidth produces a zero spectrum with no energy injection."))
    end

    # Compute default domain_size if not provided
    domain_size = domain_size === nothing ? ntuple(i -> T(2π), N) : T.(domain_size)

    # Use forcing_rate if provided, otherwise use energy_injection_rate
    # Only warn if both appear to be explicitly set (forcing_rate provided AND energy_injection_rate differs from default)
    default_energy_injection_rate = 1.0
    energy = forcing_rate === nothing ? energy_injection_rate : forcing_rate
    energy < 0 &&
        throw(ArgumentError("effective energy injection rate must be nonnegative (got $energy)"))
    if forcing_rate !== nothing && !isapprox(energy_injection_rate, default_energy_injection_rate) && !isapprox(forcing_rate, energy_injection_rate)
        @warn "Both forcing_rate and energy_injection_rate were provided; using forcing_rate"
    end

    # Derive the working RNG from a single shared seed. Two goals:
    #  1. Every MPI rank must generate the SAME global random field — the phases
    #     have to match across ranks or the assembled forcing is incoherent. The
    #     Bcast makes rank 0's seed win when ranks carry independent RNG state
    #     (e.g. a default MersenneTwister with per-process OS entropy).
    #  2. A given user seed must reproduce the SAME forcing regardless of the
    #     number of ranks. This derivation runs UNCONDITIONALLY — previously it
    #     was gated on Comm_size > 1, so a serial run kept the raw user RNG while
    #     a parallel run forked to MersenneTwister(rand(rng, UInt64)); the two
    #     then diverged (~169% different) under the same seed, and a serial
    #     trajectory could not be reproduced in parallel.
    seed_buf = Ref(rand(rng, UInt64))
    if MPI.Initialized() && MPI.Comm_size(MPI.COMM_WORLD) > 1
        MPI.Bcast!(seed_buf, 0, MPI.COMM_WORLD)
    end
    rng = Random.MersenneTwister(seed_buf[])

    # Build wavenumber arrays (always on CPU for setup)
    wavenumbers = build_wavenumbers(field_size, domain_size, dtype)

    # Compute the forcing spectrum √Q̂(k) on CPU first
    forcing_spectrum_cpu = compute_forcing_spectrum(
        wavenumbers, k_forcing, dk_forcing, energy,
        domain_size, spectrum_type, dtype;
        injection_metric
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
        injection_metric,
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
        nothing,
        ntuple(_ -> 0, N),
        ntuple(_ -> 1:0, N),
        injection_metric,
        T(-Inf),  # Initialize to -Inf so first call always updates
        spectrum_type,
        enforce_hermitian,
        architecture
    )
end

function _normalized_chebyshev_profile(
    basis::ChebyshevT,
    profile,
    ::Type{T},
) where {T<:AbstractFloat}
    n = basis.meta.size
    transform = ChebyshevTransform(basis)
    setup_chebyshev_cpu_transform!(transform, n, n, 1)

    if profile isa AbstractVector
        length(profile) == n || throw(DimensionMismatch(
            "Chebyshev profile has length $(length(profile)); expected $n coefficients",
        ))
        coeffs = T.(profile)
        values = vec(_chebyshev_backward(coeffs, transform))
    else
        lo, hi = basis.meta.bounds
        reference_grid = _native_grid(basis, 1.0)
        physical_grid = @. T((hi - lo) / 2) * reference_grid + T((hi + lo) / 2)
        values = T[profile(z) for z in physical_grid]
        coeffs = vec(T.(_chebyshev_forward(values, transform)))
    end

    all(isfinite, coeffs) && all(isfinite, values) ||
        throw(ArgumentError("Chebyshev profile must contain only finite values"))

    lo, hi = basis.meta.bounds
    weights = T.(get_integration_weights(basis))
    mean_square = sum(weights .* abs2.(values)) / T(hi - lo)
    isfinite(mean_square) && mean_square > zero(T) ||
        throw(ArgumentError("Chebyshev profile must have a finite, nonzero mean-square norm"))

    coeffs ./= sqrt(mean_square)
    return coeffs
end

function SeparableStochasticForcing(;
    fourier_size::NTuple{NF,Int},
    chebyshev_basis,
    chebyshev_profile,
    domain_size::Union{Nothing,NTuple{NF,Real}}=nothing,
    energy_injection_rate::Real=1.0,
    forcing_rate::Union{Nothing,Real}=nothing,
    injection_metric::Symbol=:direct,
    k_forcing::Real=4.0,
    dk_forcing::Real=1.0,
    dt::Real=0.01,
    spectrum_type::Symbol=:ring,
    rng::AbstractRNG=Random.MersenneTwister(),
    dtype::Type{T}=Float64,
    enforce_hermitian::Bool=true,
    architecture::AbstractArchitecture=CPU(),
) where {T<:AbstractFloat,NF}
    chebyshev_basis isa ChebyshevT || throw(ArgumentError(
        "SeparableStochasticForcing requires a ChebyshevT basis",
    ))
    injection_metric === :direct || throw(ArgumentError(
        "mixed Fourier--Chebyshev forcing supports only injection_metric=:direct",
    ))

    base = StochasticForcing(
        field_size=fourier_size,
        domain_size=domain_size,
        energy_injection_rate=energy_injection_rate,
        forcing_rate=forcing_rate,
        injection_metric=:direct,
        k_forcing=k_forcing,
        dk_forcing=dk_forcing,
        dt=dt,
        spectrum_type=spectrum_type,
        rng=rng,
        dtype=dtype,
        enforce_hermitian=enforce_hermitian,
        architecture=architecture,
    )

    profile_cpu = _normalized_chebyshev_profile(
        chebyshev_basis, chebyshev_profile, T,
    )
    profile = on_architecture(architecture, profile_cpu)
    nz = chebyshev_basis.meta.size
    field_size = (fourier_size..., nz)
    cached_forcing = zeros(architecture, Complex{T}, field_size...)
    fourier_realization = zeros(architecture, Complex{T}, fourier_size...)
    random_phases_host = is_gpu(architecture) ?
        zeros(T, fourier_size...) : base.random_phases
    N = NF + 1
    fourier_outer_view = reshape(fourier_realization, (fourier_size..., 1))
    profile_outer_view = reshape(profile, (ntuple(_ -> 1, NF)..., nz))

    return SeparableStochasticForcing{
        T, NF, N,
        typeof(base.forcing_spectrum),
        typeof(fourier_realization),
        typeof(profile),
        typeof(random_phases_host),
        typeof(cached_forcing),
        typeof(fourier_outer_view),
        typeof(profile_outer_view),
    }(
        base.forcing_spectrum,
        base.energy_injection_rate,
        :direct,
        base.k_forcing,
        base.dk_forcing,
        base.dt,
        base.domain_size,
        fourier_size,
        field_size,
        base.wavenumbers,
        chebyshev_basis,
        profile,
        cached_forcing,
        nothing,
        base.rng,
        base.random_phases,
        random_phases_host,
        fourier_realization,
        fourier_outer_view,
        profile_outer_view,
        T(-Inf),
        base.spectrum_type,
        enforce_hermitian,
        architecture,
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

function _validate_injection_metric(injection_metric::Symbol)
    injection_metric in (:direct, :vorticity_kinetic) ||
        throw(ArgumentError("injection_metric must be :direct or :vorticity_kinetic (got :$injection_metric)"))
    return nothing
end

function _injection_metric_weight(k2::T, injection_metric::Symbol) where T
    if injection_metric === :direct
        return one(T)
    elseif injection_metric === :vorticity_kinetic
        return iszero(k2) ? zero(T) : inv(k2)
    end
    _validate_injection_metric(injection_metric)
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
    compute_forcing_spectrum(wavenumbers, k_f, dk_f, ε, domain_size, spectrum_type, dtype;
                             injection_metric=:direct)

Compute the forcing amplitude spectrum √Q̂(k). With `M` physical grid points,
the full spectrum is normalized so `sum(Q̂(k) * w(k)) / (2M²) == ε`, where
`w=1` for `:direct` and `w=1/|k|²` for `:vorticity_kinetic`. A positive ε
whose selected band contains no representable nonzero mode raises `ArgumentError`.
"""
function compute_forcing_spectrum(
    wavenumbers::NTuple{N, Vector{T}},
    k_f::Real,
    dk_f::Real,
    ε::Real,
    domain_size::NTuple{N, Real},
    spectrum_type::Symbol,
    dtype::Type{T};
    injection_metric::Symbol=:direct,
) where {T<:AbstractFloat, N}

    spectrum_type = _normalize_spectrum_type(spectrum_type)
    _validate_injection_metric(injection_metric)
    ε < 0 && throw(ArgumentError("energy injection rate must be nonnegative (got $ε)"))
    if spectrum_type === :kolmogorov && !(k_f > 0)
        throw(ArgumentError(
            "k_forcing must be positive for spectrum_type=:kolmogorov " *
            "(got k_forcing=$k_f)"
        ))
    end
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

    # Tarang stores unnormalized full FFT coefficients. Parseval therefore
    # contributes M⁻², where M is the number of physical grid points.
    weighted_power = zero(T)
    for I in CartesianIndices(spectrum)
        k2 = injection_metric === :direct ? zero(T) :
             sum(d -> wavenumbers[d][I[d]]^2, 1:N)
        weight = _injection_metric_weight(k2, injection_metric)
        weighted_power += abs2(spectrum[I]) * weight
    end
    M = T(prod(field_size))
    ε0 = weighted_power / (2 * M^2)

    if ε0 > 0
        # Scale spectrum to achieve target ε
        spectrum .*= sqrt(T(ε) / ε0)
    elseif ε > 0
        throw(ArgumentError("forcing spectrum has no representable nonzero modes for positive energy injection rate $ε"))
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
        return exp(-((k - k_f)^2) / (4 * dk_f^2))

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
        # Square root of Q̂(k) ∝ (k/k_f) exp(-(|k|-k_f)²/(2δ_f²)).
        if k < k_f + dk_f
            return exp(-((k - k_f)^2) / (4 * dk_f^2)) * sqrt(k / k_f)
        else
            return zero(T)
        end

    else
        error("Unknown spectrum type: $spectrum_type")
    end
end

