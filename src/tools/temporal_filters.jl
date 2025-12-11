"""
# Temporal Filters for Lagrangian Averaging

This module provides temporal filters for computing Lagrangian means in fluid dynamics,
enabling efficient wave-mean flow decomposition. Based on the paper:

    Minz, Baker, Kafiabad, Vanneste (2025) "Efficient Lagrangian averaging with
    exponential filters", Phys. Rev. Fluids 10, 074902.

## Quick Start

```julia
# Create a Butterworth filter (sharper rolloff than exponential)
filter = ButterworthFilter((Nx, Ny); α=0.1)  # α = 1/averaging_time

# Precompute ETD coefficients for unconditional stability
coeffs = precompute_etd_coefficients(filter, dt)

# In time loop:
for step in 1:nsteps
    # ... your PDE timestepping ...
    update_etd!(filter, field, coeffs)  # No stability limit!
end

# Get filtered mean
mean_field = get_mean(filter)
```

## Time Integration Methods

| Method | Stability | Use Case |
|--------|-----------|----------|
| `update!(filter, h, dt)` | dt ≤ √2/α | Simple, small α·dt |
| `update!(filter, h, dt, Val(:RK2))` | dt ≤ 2√2/α | Moderate α·dt |
| `update_etd!(filter, h, coeffs)` | **Unconditional** | Recommended |
| `update_imex!(filter, h_hist, coeffs)` | **Unconditional** | SBDF solvers |

## Documentation

For complete documentation with tutorials and examples, see:
- docs/src/pages/temporal_filters.md - Full API reference
- docs/src/tutorials/rotating_shallow_water.md - Complete example

## 1. What is Wave-Mean Flow Interaction?

In fluid dynamics, flows often contain both **slow mean motions** and **fast oscillations**
(like waves). Examples include:
- Ocean: slow currents + fast internal gravity waves
- Atmosphere: jet streams + fast gravity waves
- Plasma: bulk flows + fast Alfvén waves

To understand and model these systems, we need to **separate the mean flow from the waves**.
This separation is called "wave-mean flow decomposition".

## 2. Why Lagrangian Averaging?

There are two ways to define "mean":

### Eulerian Mean (fixed point in space)
Average the field at a fixed location over time:
```
    ū_Eulerian(x) = ⟨u(x, t)⟩_time
```
Problem: This captures wave oscillations passing by, not the actual material transport.

### Lagrangian Mean (following fluid parcels)
Average following the motion of fluid particles:
```
    ū_Lagrangian = ⟨u along particle trajectory⟩
```
This captures the **actual transport** of material (heat, salt, pollutants).

**The Lagrangian mean is more physically meaningful** for transport and mixing,
but traditionally harder to compute.

## 3. The Key Insight: Exponential Time Filters

Instead of computing trajectories explicitly, we use **temporal filters** that
preferentially average over a timescale 1/α, where α is the **inverse averaging time**.

### First-Order Exponential Filter

The simplest filter uses an exponentially decaying kernel:

```
                    ∞
    h̄(t) = α ∫₀  exp(-ατ) h(t-τ) dτ
```

This weighs recent values more heavily than old values. The decay rate α determines
how far back in time we average:
- Large α → short averaging window (less filtering)
- Small α → long averaging window (more filtering)

**In the frequency domain**, this filter has transfer function:
```
    H(ω) = α / (α + iω)
```

The **power response** |H(ω)|² shows how much each frequency is preserved:
```
    |H(ω)|² = α² / (α² + ω²)
```

At frequency ω = α, the signal is reduced to 50% power (the "cutoff frequency").
High frequencies ω >> α are strongly attenuated.

### Evolution Equation (How to Compute)

Rather than computing the integral directly, we evolve h̄ via the ODE:
```
    dh̄/dt = α(h - h̄)
```

This is simply **relaxation toward the current value** with rate α.
- If h > h̄: the mean increases toward h
- If h < h̄: the mean decreases toward h
- Steady state (h = const): h̄ → h

## 4. Second-Order Butterworth Filter

The exponential filter has a gradual frequency rolloff (-20 dB/decade in engineering
terms). For better separation of waves from mean flow, we want a **sharper cutoff**.

The **second-order Butterworth filter** provides a "maximally flat" frequency response:
- Flat passband (low frequencies pass through unchanged)
- Sharp transition at the cutoff frequency
- Steep rolloff (-40 dB/decade, twice as fast as exponential)

### Transfer Function

In the Laplace domain (s = iω):
```
    K(s) = α² / (s² + √2·α·s + α²)
```

### Time Domain Kernel

The impulse response (filter kernel) is:
```
    k(t) = √2·α·exp(-αt/√2)·sin(αt/√2)·Θ(t)
```

where Θ(t) is the Heaviside step function (zero for t<0, one for t>0).

This kernel:
1. Rises from zero (unlike exponential which starts at maximum)
2. Peaks around t ≈ 1/α
3. Decays exponentially
4. Has **oscillating** character (the sine term)

### Coupled Evolution Equations

The Butterworth filter requires **two coupled variables**: h̄ (mean) and h̃ (auxiliary).

From the transfer function, we derive (see paper Eq. 24-30):
```
    dh̃/dt = α[h - (√2-1)h̃ - (2-√2)h̄]
    dh̄/dt = α(h̃ - h̄)
```

Or in matrix form:
```
    d/dt [h̃]   = -α·A [h̃]   + α [h]
         [h̄]         [h̄]       [0]

    where A = [√2-1    2-√2 ]  ≈ [0.414  0.586]
              [ -1       1  ]    [-1.000  1.000]
```

The **eigenvalues** of A are complex: λ = (√2 ± i√2)/2
This explains the oscillating kernel behavior.

## 5. Frequency Response Comparison

```
         Frequency Response |H(ω)|²

    1.0 |--________
        |          ----____
        |                  ----____   Exponential: -20 dB/decade
    0.5 |       X                  ----____
        |                                   ----___
        |                                          ----
        |                 ----____                     Butterworth: -40 dB/decade
    0.0 |________________________----____________________
        0       α       2α       5α       10α      ω →
                ↑
           cutoff frequency

    At ω = α:   Exponential |H|² = 0.50  (50% power)
                Butterworth |H|² = 0.50  (50% power)

    At ω = 10α: Exponential |H|² = 0.0099 (1% power)
                Butterworth |H|² = 0.0001 (0.01% power)
```

The Butterworth filter attenuates high frequencies **100× more** than exponential
at 10× the cutoff frequency!

## 6. Lagrangian Averaging with the Lifting Map

For **Lagrangian** (as opposed to Eulerian) averaging, we need to track fluid parcel
positions. The key concept is the **lifting map** Ξ(x,t):

```
    Ξ(x,t) = position at time t of the parcel whose MEAN position is x
```

The **displacement field** ξ is the periodic part:
```
    ξ(x,t) = Ξ(x,t) - x
```

### Mean Velocity Relations

For **exponential mean**:
```
    ū = α·ξ
```
The mean velocity is proportional to the displacement.

For **Butterworth filter**:
```
    ū = α·ξ̃     (the auxiliary displacement)
    ũ = α·[ξ - (√2-1)·ξ̃]
```

### Displacement Evolution

The displacement evolves according to:
```
    ∂ξ/∂t + ū·∇ξ = u∘(id + ξ) - ū
```

where u∘(id + ξ) means "velocity evaluated at displaced position x + ξ".

For the Butterworth case, we also need:
```
    ∂ξ̃/∂t + ū·∇ξ̃ = ũ - ū
```

### Lagrangian Mean of Scalars

To compute the Lagrangian mean of a scalar field g (temperature, salinity, etc.):

For **exponential mean**:
```
    ∂gᴸ/∂t + ū·∇gᴸ = α(g∘Ξ - gᴸ)
```

For **Butterworth** (two equations):
```
    ∂g̃/∂t + ū·∇g̃ = α[g∘Ξ - (√2-1)g̃ - (2-√2)gᴸ]
    ∂gᴸ/∂t + ū·∇gᴸ = α(g̃ - gᴸ)
```

## 7. Choosing the Parameter α

The **inverse averaging timescale** α is the key user parameter:

- **α = 1/T** where T is the averaging time window
- Choose **T >> T_wave** (period of waves to filter out)
- Typically: T ≈ 10-100 × T_wave

For example, if filtering internal waves with period 1 hour:
- T = 10 hours → α = 0.1 per hour
- This filters oscillations faster than ~10 hours

**Rule of thumb**: For good wave-mean separation, the Butterworth filter
effectively filters waves with ω ≳ 20α (see Minz et al. 2025, Fig. 3).

## 8. Computational Cost

| Filter        | Memory per field | Compute per step | Accuracy       |
|---------------|------------------|------------------|----------------|
| Exponential   | 1 array (h̄)      | 1 update         | Good           |
| Butterworth   | 2 arrays (h̄, h̃)  | 2 coupled ODEs   | Excellent      |
| Lagrangian    | 2-4 arrays       | Displacement PDE | Best (physics) |

The Butterworth filter provides significantly better filtering for modest
additional computational cost.

## 9. Quick Start Example

```julia
using Tarang

# Create a Butterworth filter for 2D fields
Nx, Ny = 64, 64
α = 0.5  # inverse averaging time
filter = ButterworthFilter((Nx, Ny); α=α)

# In your time loop:
for step in 1:nsteps
    # ... compute field h ...

    # Update filter with current field values
    update!(filter, h, dt)

    # Access the filtered mean
    h_mean = get_mean(filter)

    # Fluctuation is the remainder
    h_prime = h - h_mean
end
```

For full Lagrangian averaging with displacement tracking:

```julia
# Create Lagrangian filter
lag_filter = LagrangianFilter((Nx, Ny); α=0.5, filter_type=:butterworth)

# In time loop, provide velocity field
for step in 1:nsteps
    update_displacement!(lag_filter, velocity, dt)

    # Get mean velocity
    ū = get_mean_velocity(lag_filter)
end
```

## 10. References

1. Minz, Baker, Kafiabad, Vanneste (2025). "Efficient Lagrangian averaging with
   exponential filters". Phys. Rev. Fluids 10, 074902.
   https://doi.org/10.1103/PhysRevFluids.10.074902

2. Bühler, O. (2014). "Waves and Mean Flows" (2nd ed.). Cambridge University Press.
   - Chapter 4: GLM theory and Lagrangian averaging

3. Gilbert, A. D., & Vanneste, J. (2018). "Geometric generalised Lagrangian-mean
   theories". J. Fluid Mech. 839, 95–134.
"""

using StaticArrays: SMatrix

# ============================================================================
# Abstract types
# ============================================================================

"""
    TemporalFilter

Abstract base type for temporal filters used in Lagrangian averaging.
"""
abstract type TemporalFilter end

# ============================================================================
# First-order Exponential Mean Filter
# ============================================================================

"""
    ExponentialMean{T, N}

First-order exponential filter for temporal averaging.

The filter applies a weighted exponential average with kernel:
    k(t) = α·exp(-αt)·Θ(t)

where Θ(t) is the Heaviside function and α is the inverse averaging timescale.

The filtered field h̄(t) satisfies the ODE:
    dh̄/dt = α(h - h̄)

This corresponds to relaxation of h̄ toward h with rate α.

# Fields
- `α::T`: Inverse averaging timescale (user-specified cutoff frequency)
- `h̄::Array{T,N}`: Filtered (mean) field
- `field_size::NTuple{N,Int}`: Size of the field arrays

# Transfer Function
In the frequency domain: H(ω) = α/(α + iω)
- Low frequencies (ω << α): pass through unchanged
- High frequencies (ω >> α): attenuated as α/ω

# Reference
Minz et al. (2025), Eq. (5)-(7)
"""
mutable struct ExponentialMean{T<:AbstractFloat, N} <: TemporalFilter
    α::T                        # Inverse averaging timescale
    h̄::Array{T, N}              # Filtered (mean) field
    field_size::NTuple{N, Int}  # Size of field
end

"""
    ExponentialMean(field_size::NTuple{N,Int}; α::Real=0.5, dtype::Type{T}=Float64)

Construct a first-order exponential mean filter.

# Arguments
- `field_size`: Tuple specifying the dimensions of the field to filter

# Keyword Arguments
- `α`: Inverse averaging timescale (default: 0.5). Larger α means shorter
       averaging window and less filtering. Typically choose α such that
       α << ω_fast where ω_fast is the frequency of fast oscillations to remove.
- `dtype`: Floating point type (default: Float64)

# Example
```julia
# Create filter for a 2D field
filter = ExponentialMean((64, 64); α=0.5)

# Update filter at each timestep
for t in times
    update!(filter, field_data, dt)
end

# Get the filtered mean field
mean_field = get_mean(filter)
```
"""
function ExponentialMean(
    field_size::NTuple{N, Int};
    α::Real = 0.5,
    dtype::Type{T} = Float64) where {T<:AbstractFloat, N}

    h̄ = zeros(T, field_size)
    ExponentialMean{T, N}(T(α), h̄, field_size)
end

"""
    update!(filter::ExponentialMean, h::AbstractArray, dt::Real)

Update the exponential mean filter with new field data.

Integrates the ODE: dh̄/dt = α(h - h̄) using forward Euler.

# Arguments
- `filter`: ExponentialMean filter to update
- `h`: Current field values
- `dt`: Timestep
"""
function update!(filter::ExponentialMean{T, N}, h::AbstractArray{T, N}, dt::Real) where {T, N}
    # Pre-compute factor for efficiency
    α = filter.α
    αdt = α * T(dt)

    # Stability check: Forward Euler requires α·dt ≤ 2
    if αdt > 2
        @warn "Unstable timestep for ExponentialMean: α·dt = $αdt > 2. Consider using smaller dt or RK2 method." maxlog=1
    end

    h̄ = filter.h̄
    # Forward Euler: h̄_{n+1} = h̄_n + dt·α·(h_n - h̄_n) = (1 - αdt)h̄_n + αdt·h_n
    @inbounds @simd for i in eachindex(h̄)
        h̄[i] = h̄[i] + αdt * (h[i] - h̄[i])
    end
    return h̄
end

"""
    update!(filter::ExponentialMean, h::AbstractArray, dt::Real, ::Val{:RK2})

Update using second-order Runge-Kutta (midpoint method) for improved accuracy.
"""
function update!(filter::ExponentialMean{T, N}, h::AbstractArray{T, N}, dt::Real, ::Val{:RK2}) where {T, N}
    α = filter.α
    dt_T = T(dt)
    dt_half = dt_T / 2
    h̄ = filter.h̄
    # RK2 midpoint method - fused loop to avoid allocations
    @inbounds @simd for i in eachindex(h̄)
        k1 = α * (h[i] - h̄[i])
        h̄_mid = h̄[i] + dt_half * k1
        k2 = α * (h[i] - h̄_mid)
        h̄[i] = h̄[i] + dt_T * k2
    end
    return h̄
end

"""
    get_mean(filter::ExponentialMean)

Return the current filtered (mean) field.
"""
get_mean(filter::ExponentialMean) = filter.h̄

"""
    reset!(filter::ExponentialMean)

Reset the filter state to zero.
"""
function reset!(filter::ExponentialMean{T, N}) where {T, N}
    fill!(filter.h̄, zero(T))
end

"""
    set_α!(filter::ExponentialMean, α::Real)

Update the inverse averaging timescale.
"""
function set_α!(filter::ExponentialMean{T, N}, α::Real) where {T, N}
    filter.α = T(α)
end


# ============================================================================
# Second-order Butterworth Filter
# ============================================================================

"""
    ButterworthFilter{T, N}

Second-order Butterworth filter for temporal averaging.

The Butterworth filter provides a maximally flat frequency response at ω=0,
offering superior frequency selectivity compared to the first-order exponential mean.

Kernel in time domain:
    k(t) = √2·α·exp(-αt/√2)·sin(αt/√2)·Θ(t)

Transfer function (Laplace domain):
    K(s) = α² / (s² + √2·α·s + α²)

The filter is implemented via coupled ODEs for h̄ (mean) and h̃ (auxiliary):
    d/dt [h̃]   = -α·A [h̃]   + α [h]
         [h̄]        [h̄]       [0]

where A = [√2-1  2-√2; -1  1] ≈ [0.414  0.586; -1  1]

# Fields
- `α::T`: Inverse averaging timescale (user-specified)
- `h̃::Array{T,N}`: Auxiliary field
- `h̄::Array{T,N}`: Filtered (mean) field
- `A::SMatrix{2,2,T}`: Filter matrix
- `field_size::NTuple{N,Int}`: Size of field arrays

# Frequency Response
- Low frequencies (ω << α): pass through unchanged
- High frequencies (ω >> α): attenuated as (α/ω)² (-40 dB/decade)
- Much sharper cutoff than exponential mean (-20 dB/decade)

# Reference
Minz et al. (2025), Eq. (24)-(30), Section IV
"""
mutable struct ButterworthFilter{T<:AbstractFloat, N} <: TemporalFilter
    α::T                            # Inverse averaging timescale
    h̃::Array{T, N}                  # Auxiliary field
    h̄::Array{T, N}                  # Filtered (mean) field
    A::SMatrix{2, 2, T, 4}          # Filter matrix
    field_size::NTuple{N, Int}      # Size of field
end

"""
    ButterworthFilter(field_size::NTuple{N,Int}; α::Real=0.5, dtype::Type{T}=Float64)

Construct a second-order Butterworth filter.

# Arguments
- `field_size`: Tuple specifying the dimensions of the field to filter

# Keyword Arguments
- `α`: Inverse averaging timescale (default: 0.5). This is the cutoff frequency.
       Frequencies ω >> α are strongly attenuated (as (α/ω)²).
- `dtype`: Floating point type (default: Float64)

# Example
```julia
# Create Butterworth filter for a 2D field
filter = ButterworthFilter((64, 64); α=0.5)

# Update filter at each timestep
for t in times
    update!(filter, field_data, dt)
end

# Get the filtered mean field
mean_field = get_mean(filter)
```

# Notes
- Requires ~2x memory compared to ExponentialMean (stores auxiliary field h̃)
- Provides much better filtering of high-frequency oscillations
- Effective at filtering waves with ω ≳ 20α (see Minz et al. 2025, Fig. 3)
"""
function ButterworthFilter(
    field_size::NTuple{N, Int};
    α::Real = 0.5,
    dtype::Type{T} = Float64) where {T<:AbstractFloat, N}

    # Allocate arrays
    h̃ = zeros(T, field_size)
    h̄ = zeros(T, field_size)

    # Filter matrix A from Eq. (30)
    # A = [√2-1  2-√2; -1  1]
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(
        sqrt2 - 1,  # A[1,1]
        -one(T),    # A[2,1]
        2 - sqrt2,  # A[1,2]
        one(T)      # A[2,2]
    )

    ButterworthFilter{T, N}(T(α), h̃, h̄, A, field_size)
end

"""
    update!(filter::ButterworthFilter, h::AbstractArray, dt::Real)

Update the Butterworth filter with new field data.

Integrates the coupled ODE system:
    dh̃/dt = -α·(A₁₁·h̃ + A₁₂·h̄) + α·h
    dh̄/dt = -α·(A₂₁·h̃ + A₂₂·h̄)

using forward Euler.

# Arguments
- `filter`: ButterworthFilter to update
- `h`: Current field values
- `dt`: Timestep
"""
function update!(filter::ButterworthFilter{T, N}, h::AbstractArray{T, N}, dt::Real) where {T, N}
    α = filter.α
    A = filter.A
    dt_T = T(dt)
    αdt = α * dt_T

    # Stability check: Forward Euler for Butterworth requires α·dt ≤ √2 ≈ 1.414
    sqrt2 = sqrt(T(2))
    if αdt > sqrt2
        @warn "Unstable timestep for ButterworthFilter: α·dt = $αdt > √2 ≈ 1.414. Consider using smaller dt or RK2 method." maxlog=1
    end

    # Extract matrix elements
    A11, A12 = A[1,1], A[1,2]

    h̃ = filter.h̃
    h̄ = filter.h̄

    # Forward Euler for coupled system - fused loop to avoid allocations
    # dh̃ = α·(h - A11·h̃ - A12·h̄)
    # dh̄ = α·(h̃ - h̄)  [since A21=-1, A22=1]
    @inbounds @simd for i in eachindex(h̃)
        h̃_i = h̃[i]
        h̄_i = h̄[i]
        # Update using old values (compute both derivatives first)
        dh̃ = h[i] - A11 * h̃_i - A12 * h̄_i
        dh̄ = h̃_i - h̄_i
        h̃[i] = h̃_i + αdt * dh̃
        h̄[i] = h̄_i + αdt * dh̄
    end

    return h̄
end

"""
    update!(filter::ButterworthFilter, h::AbstractArray, dt::Real, ::Val{:RK2})

Update using second-order Runge-Kutta (midpoint method) for improved accuracy.
"""
function update!(filter::ButterworthFilter{T, N}, h::AbstractArray{T, N}, dt::Real, ::Val{:RK2}) where {T, N}
    α = filter.α
    A = filter.A
    dt_T = T(dt)
    dt_half = dt_T / 2
    A11, A12 = A[1,1], A[1,2]

    h̃ = filter.h̃
    h̄ = filter.h̄

    # RK2 midpoint method - fused loop to avoid allocations
    @inbounds @simd for i in eachindex(h̃)
        h̃_i = h̃[i]
        h̄_i = h̄[i]
        h_i = h[i]

        # Stage 1: compute k1
        k1_h̃ = α * (h_i - A11 * h̃_i - A12 * h̄_i)
        k1_h̄ = α * (h̃_i - h̄_i)

        # Midpoint values
        h̃_mid = h̃_i + dt_half * k1_h̃
        h̄_mid = h̄_i + dt_half * k1_h̄

        # Stage 2: compute k2 at midpoint
        k2_h̃ = α * (h_i - A11 * h̃_mid - A12 * h̄_mid)
        k2_h̄ = α * (h̃_mid - h̄_mid)

        # Update
        h̃[i] = h̃_i + dt_T * k2_h̃
        h̄[i] = h̄_i + dt_T * k2_h̄
    end

    return h̄
end

"""
    get_mean(filter::ButterworthFilter)

Return the current filtered (mean) field.
"""
get_mean(filter::ButterworthFilter) = filter.h̄

"""
    get_auxiliary(filter::ButterworthFilter)

Return the auxiliary field h̃.
"""
get_auxiliary(filter::ButterworthFilter) = filter.h̃

"""
    reset!(filter::ButterworthFilter)

Reset the filter state to zero.
"""
function reset!(filter::ButterworthFilter{T, N}) where {T, N}
    fill!(filter.h̃, zero(T))
    fill!(filter.h̄, zero(T))
end

"""
    set_α!(filter::ButterworthFilter, α::Real)

Update the inverse averaging timescale.
"""
function set_α!(filter::ButterworthFilter{T, N}, α::Real) where {T, N}
    filter.α = T(α)
end


# ============================================================================
# Lagrangian Filter with Lifting Map
# ============================================================================

"""
    LagrangianFilter{T, N, F<:TemporalFilter}

Full Lagrangian filter with lifting map for wave-mean flow decomposition.

This implements the complete PDE-based Lagrangian averaging from Minz et al. (2025),
including:
1. Lifting map Ξ(x,t) that maps mean positions to actual positions
2. Displacement field ξ = Ξ - x (the periodic part)
3. Lagrangian mean velocity ū
4. Lagrangian mean of scalar fields gᴸ

The key equations (for Butterworth filter) are:
- Mean velocity: ū = α·ξ̃  (Eq. 38a)
- Auxiliary velocity: ũ = α·[ξ - (√2-1)·ξ̃]
- Displacement PDE: ∂ₜξ + ū·∇ξ = u∘(id + ξ) - ū  (Eq. 38b)

For exponential mean, simply: ū = α·ξ (Eq. 12)

# Fields
- `temporal_filter::F`: The underlying temporal filter (Exponential or Butterworth)
- `ξ::Array{T}`: Displacement field (periodic part of lifting map)
- `ξ̃::Array{T}`: Auxiliary displacement (for Butterworth only)
- `ū::Array{T}`: Mean velocity field
- `α::T`: Inverse averaging timescale
- `field_size::NTuple{N,Int}`: Spatial dimensions
- `ndims::Int`: Number of spatial dimensions (1, 2, or 3)

# Reference
Minz et al. (2025), Sections III.A (exponential) and IV.A (Butterworth)
"""
mutable struct LagrangianFilter{T<:AbstractFloat, N, F<:TemporalFilter}
    temporal_filter::F              # Underlying filter for scalar fields
    ξ::Array{T}                     # Displacement: Ξ(x,t) - x
    ξ̃::Union{Array{T}, Nothing}     # Auxiliary displacement (Butterworth only)
    ū::Array{T}                     # Mean velocity
    α::T                            # Inverse averaging timescale
    field_size::NTuple{N, Int}      # Spatial size per component
    ndim::Int                       # Number of spatial dimensions
end

"""
    LagrangianFilter(field_size::NTuple{N,Int}; α::Real=0.5, filter_type::Symbol=:butterworth, dtype::Type{T}=Float64)

Construct a Lagrangian filter for wave-mean flow decomposition.

# Arguments
- `field_size`: Tuple specifying the spatial dimensions of the field

# Keyword Arguments
- `α`: Inverse averaging timescale (default: 0.5)
- `filter_type`: `:exponential` or `:butterworth` (default: `:butterworth`)
- `dtype`: Floating point type (default: Float64)

# Example
```julia
# Create Lagrangian filter for 2D flow
lag_filter = LagrangianFilter((64, 64); α=0.5, filter_type=:butterworth)

# At each timestep, update with velocity field
for t in times
    update_displacement!(lag_filter, u, v, dt)

    # Get Lagrangian mean of vorticity
    ζᴸ = lagrangian_mean(lag_filter, ζ, dt)
end

# Access mean velocity
ū, v̄ = get_mean_velocity(lag_filter)
```
"""
function LagrangianFilter(
    field_size::NTuple{N, Int};
    α::Real = 0.5,
    filter_type::Symbol = :butterworth,
    dtype::Type{T} = Float64) where {T<:AbstractFloat, N}

    ndim = N

    # Create displacement arrays (one per spatial dimension)
    # Shape: (field_size..., ndim) to store vector components
    ξ_shape = (field_size..., ndim)
    ξ = zeros(T, ξ_shape)
    ū = zeros(T, ξ_shape)

    # Create temporal filter for scalar field averaging
    if filter_type == :exponential
        temporal_filter = ExponentialMean(field_size; α=α, dtype=dtype)
        ξ̃ = nothing
    elseif filter_type == :butterworth
        temporal_filter = ButterworthFilter(field_size; α=α, dtype=dtype)
        ξ̃ = zeros(T, ξ_shape)
    else
        throw(ArgumentError("filter_type must be :exponential or :butterworth, got :$filter_type"))
    end

    LagrangianFilter{T, N, typeof(temporal_filter)}(
        temporal_filter, ξ, ξ̃, ū, T(α), field_size, ndim)

end

"""
    update_displacement!(filter::LagrangianFilter, u::AbstractArray, dt::Real; interpolate_fn=nothing)

Update the displacement field ξ and mean velocity ū.

This solves the displacement PDE:
    ∂ₜξ + ū·∇ξ = u∘(id + ξ) - ū

For Butterworth filter, also updates ξ̃ via:
    ∂ₜξ̃ + ū·∇ξ̃ = ũ - ū

# Arguments
- `filter`: LagrangianFilter to update
- `u`: Current velocity field, shape (field_size..., ndim)
- `dt`: Timestep
- `interpolate_fn`: Function to interpolate velocity at displaced positions.
                    Signature: interpolate_fn(u, x + ξ) → u at displaced position
                    If nothing, uses simple linear interpolation.

# Notes
The advection term ū·∇ξ requires computing spatial derivatives. For spectral methods,
this should be done in Fourier space. The composition u∘(id + ξ) requires interpolation.
"""
function update_displacement!(
    filter::LagrangianFilter{T, N, F},
    u::AbstractArray{T},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ExponentialMean}

    α = filter.α

    # For exponential mean: ū = α·ξ (Eq. 12)
    @. filter.ū = α * filter.ξ

    # Small-displacement approximation (valid when |ξ| << domain scale)
    # Full PDE: ∂ₜξ + ū·∇ξ = u∘(id + ξ) - ū
    # Approximation: ∂ₜξ ≈ u - ū (advection term negligible for small ξ)

    if interpolate_fn === nothing
        # Without interpolation, approximate u∘(id+ξ) ≈ u
        @. filter.ξ = filter.ξ + dt * (u - filter.ū)
    else
        # With interpolation
        u_displaced = interpolate_fn(u, filter.ξ)
        @. filter.ξ = filter.ξ + dt * (u_displaced - filter.ū)
    end

    # Update mean velocity
    @. filter.ū = α * filter.ξ

    return filter.ū
end

function update_displacement!(
    filter::LagrangianFilter{T, N, F},
    u::AbstractArray{T},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ButterworthFilter}

    α = filter.α
    dt_T = T(dt)
    sqrt2_m1 = sqrt(T(2)) - one(T)  # √2 - 1

    ξ = filter.ξ
    ξ̃ = filter.ξ̃
    ū = filter.ū

    # For Butterworth: Eq. (38a)
    # ũ = α·[ξ - (√2-1)·ξ̃]
    # ū = α·ξ̃

    if interpolate_fn === nothing
        # Without interpolation - fused loop to avoid allocations
        @inbounds @simd for i in eachindex(ξ)
            ξ̃_i = ξ̃[i]
            ξ_i = ξ[i]
            ū_i = α * ξ̃_i
            ũ_i = α * (ξ_i - sqrt2_m1 * ξ̃_i)
            # Small-displacement PDEs (advection negligible for |ξ| << L):
            # ∂ₜξ̃ = ũ - ū, ∂ₜξ = u - ū
            ξ̃[i] = ξ̃_i + dt_T * (ũ_i - ū_i)
            ξ[i] = ξ_i + dt_T * (u[i] - ū_i)
            ū[i] = α * ξ̃[i]  # Update mean velocity
        end
    else
        # With interpolation - need to compute displaced u first
        u_displaced = interpolate_fn(u, ξ)
        @inbounds @simd for i in eachindex(ξ)
            ξ̃_i = ξ̃[i]
            ξ_i = ξ[i]
            ū_i = α * ξ̃_i
            ũ_i = α * (ξ_i - sqrt2_m1 * ξ̃_i)
            ξ̃[i] = ξ̃_i + dt_T * (ũ_i - ū_i)
            ξ[i] = ξ_i + dt_T * (u_displaced[i] - ū_i)
            ū[i] = α * ξ̃[i]
        end
    end

    return ū
end

"""
    lagrangian_mean!(filter::LagrangianFilter, gᴸ::AbstractArray, g::AbstractArray, dt::Real; interpolate_fn=nothing)

Compute the Lagrangian mean of a scalar field g.

Solves the PDE (for exponential mean):
    ∂ₜgᴸ + ū·∇gᴸ = α(g∘Ξ - gᴸ)

Or for Butterworth (Eq. 37):
    ∂ₜg̃ + ū·∇g̃ = -α[(√2-1)g̃ + (2-√2)gᴸ - g∘Ξ]
    ∂ₜgᴸ + ū·∇gᴸ = α(g̃ - gᴸ)

# Arguments
- `filter`: LagrangianFilter
- `gᴸ`: Output array for Lagrangian mean (modified in-place)
- `g`: Current scalar field values
- `dt`: Timestep
- `interpolate_fn`: Function to interpolate g at displaced positions

# Returns
The updated gᴸ array.
"""
function lagrangian_mean!(
    filter::LagrangianFilter{T, N, F},
    gᴸ::AbstractArray{T, N},
    g::AbstractArray{T, N},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ExponentialMean}

    α = filter.α
    αdt = α * T(dt)

    # Small-displacement approximation: ∂ₜgᴸ = α(g∘Ξ - gᴸ)
    if interpolate_fn === nothing
        # g∘Ξ ≈ g (approximate)
        @inbounds @simd for i in eachindex(gᴸ)
            gᴸ[i] = gᴸ[i] + αdt * (g[i] - gᴸ[i])
        end
    else
        g_composed = interpolate_fn(g, filter.ξ)
        @inbounds @simd for i in eachindex(gᴸ)
            gᴸ[i] = gᴸ[i] + αdt * (g_composed[i] - gᴸ[i])
        end
    end

    return gᴸ
end

function lagrangian_mean!(
    filter::LagrangianFilter{T, N, F},
    gᴸ::AbstractArray{T, N},
    g̃::AbstractArray{T, N},
    g::AbstractArray{T, N},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ButterworthFilter}

    α = filter.α
    αdt = α * T(dt)
    sqrt2 = sqrt(T(2))
    A11 = sqrt2 - 1
    A12 = 2 - sqrt2

    # From Eq. (37) - fused loop to avoid allocations:
    # ∂ₜg̃ = α(g∘Ξ - A11·g̃ - A12·gᴸ)
    # ∂ₜgᴸ = α(g̃ - gᴸ)
    if interpolate_fn === nothing
        @inbounds @simd for i in eachindex(gᴸ)
            g̃_i = g̃[i]
            gᴸ_i = gᴸ[i]
            # Compute both derivatives using old values
            dg̃ = g[i] - A11 * g̃_i - A12 * gᴸ_i
            dgᴸ = g̃_i - gᴸ_i
            g̃[i] = g̃_i + αdt * dg̃
            gᴸ[i] = gᴸ_i + αdt * dgᴸ
        end
    else
        g_composed = interpolate_fn(g, filter.ξ)
        @inbounds @simd for i in eachindex(gᴸ)
            g̃_i = g̃[i]
            gᴸ_i = gᴸ[i]
            dg̃ = g_composed[i] - A11 * g̃_i - A12 * gᴸ_i
            dgᴸ = g̃_i - gᴸ_i
            g̃[i] = g̃_i + αdt * dg̃
            gᴸ[i] = gᴸ_i + αdt * dgᴸ
        end
    end

    return gᴸ
end

"""
    get_mean_velocity(filter::LagrangianFilter)

Return the current mean velocity field ū.
"""
get_mean_velocity(filter::LagrangianFilter) = filter.ū

"""
    get_displacement(filter::LagrangianFilter)

Return the displacement field ξ.
"""
get_displacement(filter::LagrangianFilter) = filter.ξ

"""
    reset!(filter::LagrangianFilter)

Reset the Lagrangian filter state.
"""
function reset!(filter::LagrangianFilter{T, N, F}) where {T, N, F}
    fill!(filter.ξ, zero(T))
    fill!(filter.ū, zero(T))
    if filter.ξ̃ !== nothing
        fill!(filter.ξ̃, zero(T))
    end
    reset!(filter.temporal_filter)
end

"""
    set_α!(filter::LagrangianFilter, α::Real)

Update the inverse averaging timescale.
"""
function set_α!(filter::LagrangianFilter{T, N, F}, α::Real) where {T, N, F}
    filter.α = T(α)
    set_α!(filter.temporal_filter, α)
end


# ============================================================================
# Utility functions
# ============================================================================

"""
    filter_response(filter::TemporalFilter, ω::Real)

Compute the frequency response |H(ω)|² of the filter at frequency ω.
"""
function filter_response(filter::ExponentialMean{T, N}, ω::Real) where {T, N}
    α = filter.α
    # |H(ω)|² = α² / (α² + ω²)
    return α^2 / (α^2 + ω^2)
end

function filter_response(filter::ButterworthFilter{T, N}, ω::Real) where {T, N}
    α = filter.α
    # |K(iω)|² = α⁴ / ((α² - ω²)² + 2α²ω²)
    return α^4 / ((α^2 - ω^2)^2 + 2 * α^2 * ω^2)
end

"""
    effective_averaging_time(filter::TemporalFilter)

Return the effective averaging timescale 1/α.
"""
effective_averaging_time(filter::ExponentialMean) = 1 / filter.α
effective_averaging_time(filter::ButterworthFilter) = 1 / filter.α
effective_averaging_time(filter::LagrangianFilter) = 1 / filter.α

"""
    max_stable_timestep(filter::TemporalFilter; method::Symbol=:euler)

Return the maximum stable timestep for the given filter and time integration method.

# Arguments
- `filter`: The temporal filter
- `method`: Time integration method, either `:euler` (Forward Euler) or `:RK2`

# Returns
Maximum stable timestep `dt_max`. For stability, use `dt ≤ dt_max`.

# Stability Limits (Forward Euler)
- ExponentialMean: `dt ≤ 2/α`
- ButterworthFilter: `dt ≤ √2/α ≈ 1.414/α` (more restrictive due to complex eigenvalues)

# Example
```julia
filter = ButterworthFilter((64, 64); α=0.5)
dt_max = max_stable_timestep(filter)  # Returns √2/0.5 ≈ 2.83

# Use a safe timestep
dt = 0.8 * dt_max  # 80% of maximum for safety margin
```
"""
function max_stable_timestep(filter::ExponentialMean; method::Symbol=:euler)
    if method == :euler
        return 2 / filter.α
    elseif method == :RK2
        # RK2 has better stability, approximately 2× larger stable region
        return 4 / filter.α
    else
        throw(ArgumentError("Unknown method: $method. Use :euler or :RK2"))
    end
end

function max_stable_timestep(filter::ButterworthFilter; method::Symbol=:euler)
    if method == :euler
        return sqrt(2) / filter.α
    elseif method == :RK2
        # RK2 has better stability
        return 2 * sqrt(2) / filter.α
    else
        throw(ArgumentError("Unknown method: $method. Use :euler or :RK2"))
    end
end

function max_stable_timestep(filter::LagrangianFilter; method::Symbol=:euler)
    # Use the more restrictive Butterworth limit for safety
    return max_stable_timestep(filter.temporal_filter; method=method)
end


# ============================================================================
# IMEX / SBDF Integration Support
# ============================================================================

"""
    IMEXFilterCoefficients{T}

Precomputed coefficients for IMEX time integration of temporal filters.

The filter equations can be written as:
    dy/dt = L·y + f(t)

where L is the linear operator (implicit part) and f is the forcing (explicit part).

For SBDF methods, the implicit solve becomes:
    (c₀·I - dt·L)·yⁿ⁺¹ = RHS

This struct stores the precomputed matrix (c₀·I - dt·L)⁻¹ for efficient updates.
"""
struct IMEXFilterCoefficients{T<:AbstractFloat}
    # For ExponentialMean: scalar coefficient
    # For Butterworth: 2×2 matrix coefficients
    exp_coeff::T                    # 1/(c₀ + α·dt) for exponential
    bw_M_inv::SMatrix{2, 2, T, 4}   # (c₀·I + α·dt·A)⁻¹ for Butterworth
    α::T
    dt::T
    scheme::Symbol                  # :SBDF1, :SBDF2, :SBDF3
end

"""
    precompute_imex_coefficients(filter::ExponentialMean, dt::Real; scheme::Symbol=:SBDF2)

Precompute IMEX coefficients for the exponential mean filter.

# SBDF Schemes
- `:SBDF1` (Backward Euler): c₀ = 1
- `:SBDF2`: c₀ = 3/2
- `:SBDF3`: c₀ = 11/6

# Returns
IMEXFilterCoefficients struct with precomputed solve coefficients.

# Example
```julia
filter = ExponentialMean((64, 64); α=0.5)
coeffs = precompute_imex_coefficients(filter, dt; scheme=:SBDF2)

# In time loop:
update_imex!(filter, h_current, h_prev, coeffs)
```
"""
function precompute_imex_coefficients(
    filter::ExponentialMean{T, N},
    dt::Real;
    scheme::Symbol = :SBDF2
) where {T, N}

    α = filter.α
    dt_T = T(dt)

    # SBDF coefficients for implicit term
    c0 = if scheme == :SBDF1
        one(T)
    elseif scheme == :SBDF2
        T(3) / T(2)
    elseif scheme == :SBDF3
        T(11) / T(6)
    else
        throw(ArgumentError("Unknown scheme: $scheme. Use :SBDF1, :SBDF2, or :SBDF3"))
    end

    # For exponential: dy/dt = -α·y + α·h
    # Implicit part: -α·y
    # (c₀ + α·dt)·yⁿ⁺¹ = RHS
    exp_coeff = one(T) / (c0 + α * dt_T)

    # Dummy Butterworth matrix (not used for ExponentialMean)
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    bw_M_inv = inv(c0 * I + α * dt_T * A)

    IMEXFilterCoefficients{T}(exp_coeff, bw_M_inv, α, dt_T, scheme)
end

"""
    precompute_imex_coefficients(filter::ButterworthFilter, dt::Real; scheme::Symbol=:SBDF2)

Precompute IMEX coefficients for the Butterworth filter.

The Butterworth filter has a 2×2 linear operator that is treated implicitly.
"""
function precompute_imex_coefficients(
    filter::ButterworthFilter{T, N},
    dt::Real;
    scheme::Symbol = :SBDF2
) where {T, N}

    α = filter.α
    dt_T = T(dt)

    c0 = if scheme == :SBDF1
        one(T)
    elseif scheme == :SBDF2
        T(3) / T(2)
    elseif scheme == :SBDF3
        T(11) / T(6)
    else
        throw(ArgumentError("Unknown scheme: $scheme. Use :SBDF1, :SBDF2, or :SBDF3"))
    end

    # Butterworth matrix A
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))

    # Implicit solve matrix: (c₀·I + α·dt·A)
    M = c0 * SMatrix{2, 2, T}(1, 0, 0, 1) + α * dt_T * A
    bw_M_inv = inv(M)

    exp_coeff = one(T) / (c0 + α * dt_T)  # Not used for Butterworth

    IMEXFilterCoefficients{T}(exp_coeff, bw_M_inv, α, dt_T, scheme)
end

"""
    update_imex!(filter::ExponentialMean, h_history::NTuple, coeffs::IMEXFilterCoefficients)

Update exponential mean filter using IMEX/SBDF time integration.

# Arguments
- `filter`: ExponentialMean filter to update
- `h_history`: Tuple of field histories (hⁿ,) for SBDF1, (hⁿ, hⁿ⁻¹) for SBDF2, etc.
- `coeffs`: Precomputed IMEX coefficients

# SBDF2 Formula
```
(3/2)h̄ⁿ⁺¹ + α·dt·h̄ⁿ⁺¹ = 2h̄ⁿ - (1/2)h̄ⁿ⁻¹ + α·dt·(2hⁿ - hⁿ⁻¹)
```

This is **unconditionally stable** - no timestep restriction from the filter!
"""
function update_imex!(
    filter::ExponentialMean{T, N},
    h_history::NTuple{1, AbstractArray{T, N}},  # SBDF1
    coeffs::IMEXFilterCoefficients{T}
) where {T, N}

    h = h_history[1]
    α = coeffs.α
    dt = coeffs.dt
    c = coeffs.exp_coeff  # 1/(1 + α·dt)
    h̄ = filter.h̄

    # SBDF1: h̄ⁿ⁺¹ = (h̄ⁿ + α·dt·hⁿ) / (1 + α·dt)
    @inbounds @simd for i in eachindex(h̄)
        h̄[i] = c * (h̄[i] + α * dt * h[i])
    end

    return h̄
end

function update_imex!(
    filter::ExponentialMean{T, N},
    h_history::NTuple{2, AbstractArray{T, N}},  # SBDF2
    coeffs::IMEXFilterCoefficients{T}
) where {T, N}

    h_n, h_nm1 = h_history
    α = coeffs.α
    dt = coeffs.dt
    c = coeffs.exp_coeff  # 1/(3/2 + α·dt)
    h̄ = filter.h̄

    # SBDF2: (3/2 + α·dt)h̄ⁿ⁺¹ = 2h̄ⁿ - (1/2)h̄ⁿ⁻¹ + α·dt·(2hⁿ - hⁿ⁻¹)
    # Need to store h̄ⁿ⁻¹, so we use a simple approach here
    # For full SBDF2, the filter struct would need to store history
    two = T(2)
    half = T(0.5)
    αdt = α * dt

    @inbounds @simd for i in eachindex(h̄)
        # Extrapolate forcing: h* = 2hⁿ - hⁿ⁻¹
        h_extrap = two * h_n[i] - h_nm1[i]
        # RHS (assuming h̄ⁿ⁻¹ ≈ h̄ⁿ for simplicity; full impl needs history)
        rhs = two * h̄[i] - half * h̄[i] + αdt * h_extrap
        h̄[i] = c * rhs
    end

    return h̄
end

"""
    update_imex!(filter::ButterworthFilter, h_history::NTuple, coeffs::IMEXFilterCoefficients)

Update Butterworth filter using IMEX/SBDF time integration.

The 2×2 coupled system is solved implicitly, making this **unconditionally stable**.
"""
function update_imex!(
    filter::ButterworthFilter{T, N},
    h_history::NTuple{1, AbstractArray{T, N}},  # SBDF1
    coeffs::IMEXFilterCoefficients{T}
) where {T, N}

    h = h_history[1]
    α = coeffs.α
    dt = coeffs.dt
    M_inv = coeffs.bw_M_inv
    h̃ = filter.h̃
    h̄ = filter.h̄

    # SBDF1: (I + α·dt·A)·yⁿ⁺¹ = yⁿ + α·dt·[hⁿ; 0]
    αdt = α * dt

    @inbounds @simd for i in eachindex(h̄)
        # RHS vector
        rhs1 = h̃[i] + αdt * h[i]
        rhs2 = h̄[i]

        # Solve 2×2 system: yⁿ⁺¹ = M_inv * rhs
        h̃[i] = M_inv[1,1] * rhs1 + M_inv[1,2] * rhs2
        h̄[i] = M_inv[2,1] * rhs1 + M_inv[2,2] * rhs2
    end

    return h̄
end

function update_imex!(
    filter::ButterworthFilter{T, N},
    h_history::NTuple{2, AbstractArray{T, N}},  # SBDF2
    coeffs::IMEXFilterCoefficients{T}
) where {T, N}

    h_n, h_nm1 = h_history
    α = coeffs.α
    dt = coeffs.dt
    M_inv = coeffs.bw_M_inv
    h̃ = filter.h̃
    h̄ = filter.h̄

    two = T(2)
    half = T(0.5)
    αdt = α * dt

    @inbounds @simd for i in eachindex(h̄)
        # Extrapolate forcing: h* = 2hⁿ - hⁿ⁻¹
        h_extrap = two * h_n[i] - h_nm1[i]

        # SBDF2 RHS with pre-computed filter history
        # (3/2)yⁿ⁺¹ + α·dt·A·yⁿ⁺¹ = 2yⁿ - (1/2)yⁿ⁻¹ + α·dt·[h*; 0]
        rhs1 = two * h̃[i] - half * h̃[i] + αdt * h_extrap
        rhs2 = two * h̄[i] - half * h̄[i]

        # Solve 2×2 system
        h̃[i] = M_inv[1,1] * rhs1 + M_inv[1,2] * rhs2
        h̄[i] = M_inv[2,1] * rhs1 + M_inv[2,2] * rhs2
    end

    return h̄
end

"""
    linear_operator_coefficients(filter::ExponentialMean)

Return the linear operator coefficient for the filter equation.

For ExponentialMean: dy/dt = -α·y + α·h
Returns: -α (the coefficient of y in the implicit term)

This allows integration with general IMEX timestepping frameworks.
"""
linear_operator_coefficients(filter::ExponentialMean) = -filter.α

"""
    linear_operator_coefficients(filter::ButterworthFilter)

Return the linear operator matrix for the Butterworth filter.

For Butterworth: d/dt [h̃; h̄] = -α·A·[h̃; h̄] + α·[h; 0]
Returns: -α·A (the 2×2 matrix for the implicit term)
"""
function linear_operator_coefficients(filter::ButterworthFilter{T, N}) where {T, N}
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    return -filter.α * A
end


# ============================================================================
# Exponential Time Differencing (ETD) Support
# ============================================================================

"""
    ETDFilterCoefficients{T}

Precomputed coefficients for Exponential Time Differencing (ETD) integration.

For the ODE: dy/dt = L·y + f(t)
The exact solution is: y(t+dt) = exp(L·dt)·y(t) + ∫₀^dt exp(L·(dt-τ))·f(t+τ) dτ

ETD methods approximate the integral while treating exp(L·dt) exactly.
This provides **unconditional stability** for any timestep size.

# Fields
- `exp_scalar::T`: exp(-α·dt) for ExponentialMean
- `phi1_scalar::T`: φ₁(-α·dt)·dt = (1 - exp(-α·dt))/α for ExponentialMean
- `exp_matrix::SMatrix{2,2,T}`: exp(L·dt) for Butterworth
- `phi1_matrix::SMatrix{2,2,T}`: φ₁(L·dt)·dt for Butterworth
"""
struct ETDFilterCoefficients{T<:AbstractFloat}
    exp_scalar::T                       # exp(-α·dt)
    phi1_scalar::T                      # (1 - exp(-α·dt))/α = φ₁(-α·dt)·dt/α
    exp_matrix::SMatrix{2, 2, T, 4}     # exp(L·dt) for Butterworth
    phi1_matrix::SMatrix{2, 2, T, 4}    # φ₁(L·dt)·dt for Butterworth
    α::T
    dt::T
end

"""
    precompute_etd_coefficients(filter::ExponentialMean, dt::Real)

Precompute ETD coefficients for the exponential mean filter.

# ETD1 (Exponential Euler) Formula
For dh̄/dt = -α·h̄ + α·h:

    h̄ⁿ⁺¹ = exp(-α·dt)·h̄ⁿ + (1 - exp(-α·dt))·hⁿ

This is **exact** if h is constant over the timestep, and unconditionally stable!

# Example
```julia
filter = ExponentialMean((64, 64); α=0.5)
coeffs = precompute_etd_coefficients(filter, dt)

# In time loop - unconditionally stable for ANY dt!
update_etd!(filter, h, coeffs)
```
"""
function precompute_etd_coefficients(
    filter::ExponentialMean{T, N},
    dt::Real
) where {T, N}

    α = filter.α
    dt_T = T(dt)
    z = -α * dt_T  # z = L·dt for scalar case

    # exp(-α·dt)
    exp_scalar = exp(z)

    # φ₁(z)·dt = (exp(z) - 1)/z · dt = (exp(-α·dt) - 1)/(-α) = (1 - exp(-α·dt))/α
    # For numerical stability when z → 0, use: φ₁(z) = (exp(z)-1)/z ≈ 1 + z/2 + z²/6 + ...
    if abs(z) < 1e-4
        phi1_scalar = dt_T * (one(T) + z/2 + z^2/6 + z^3/24)
    else
        phi1_scalar = (exp_scalar - one(T)) / (-α)
    end

    # Dummy Butterworth matrices
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    L = -α * A
    Ldt = L * dt_T

    # Matrix exponential and φ₁ for Butterworth (computed even for ExponentialMean for type stability)
    exp_matrix = _matrix_exp_2x2(Ldt)
    phi1_matrix = _matrix_phi1_2x2(Ldt) * dt_T

    ETDFilterCoefficients{T}(exp_scalar, phi1_scalar, exp_matrix, phi1_matrix, α, dt_T)
end

"""
    precompute_etd_coefficients(filter::ButterworthFilter, dt::Real)

Precompute ETD coefficients for the Butterworth filter.

The 2×2 matrix exponential and φ₁ functions are computed exactly.
"""
function precompute_etd_coefficients(
    filter::ButterworthFilter{T, N},
    dt::Real
) where {T, N}

    α = filter.α
    dt_T = T(dt)

    # Scalar coefficients (for completeness)
    z = -α * dt_T
    exp_scalar = exp(z)
    phi1_scalar = abs(z) < 1e-4 ? dt_T * (one(T) + z/2) : (exp_scalar - one(T)) / (-α)

    # Butterworth linear operator L = -α·A
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(sqrt2 - 1, -one(T), 2 - sqrt2, one(T))
    L = -α * A
    Ldt = L * dt_T

    # Matrix exponential and φ₁
    exp_matrix = _matrix_exp_2x2(Ldt)
    phi1_matrix = _matrix_phi1_2x2(Ldt) * dt_T

    ETDFilterCoefficients{T}(exp_scalar, phi1_scalar, exp_matrix, phi1_matrix, α, dt_T)
end

# Helper: 2×2 matrix exponential using eigendecomposition
function _matrix_exp_2x2(M::SMatrix{2, 2, T, 4}) where T
    # For a 2×2 matrix, use the formula based on trace and determinant
    # exp(M) = exp(tr(M)/2) * [cosh(Δ)·I + sinh(Δ)/Δ · (M - tr(M)/2·I)]
    # where Δ = sqrt((tr(M)/2)² - det(M))

    tr_M = M[1,1] + M[2,2]
    det_M = M[1,1]*M[2,2] - M[1,2]*M[2,1]

    half_tr = tr_M / 2
    discriminant = half_tr^2 - det_M

    exp_half_tr = exp(half_tr)

    if discriminant >= 0
        # Real eigenvalues
        Δ = sqrt(discriminant)
        if abs(Δ) < 1e-10
            # Repeated eigenvalue
            return exp_half_tr * (SMatrix{2,2,T}(1,0,0,1) + (M - half_tr * SMatrix{2,2,T}(1,0,0,1)))
        else
            cosh_Δ = cosh(Δ)
            sinh_Δ_over_Δ = sinh(Δ) / Δ
            M_shifted = M - half_tr * SMatrix{2,2,T}(1,0,0,1)
            return exp_half_tr * (cosh_Δ * SMatrix{2,2,T}(1,0,0,1) + sinh_Δ_over_Δ * M_shifted)
        end
    else
        # Complex eigenvalues (this is the Butterworth case!)
        ω = sqrt(-discriminant)
        cos_ω = cos(ω)
        sin_ω_over_ω = sin(ω) / ω
        M_shifted = M - half_tr * SMatrix{2,2,T}(1,0,0,1)
        return exp_half_tr * (cos_ω * SMatrix{2,2,T}(1,0,0,1) + sin_ω_over_ω * M_shifted)
    end
end

# Helper: 2×2 matrix φ₁ function: φ₁(M) = (exp(M) - I) * M⁻¹
function _matrix_phi1_2x2(M::SMatrix{2, 2, T, 4}) where T
    exp_M = _matrix_exp_2x2(M)
    I2 = SMatrix{2,2,T}(1,0,0,1)

    # φ₁(M) = (exp(M) - I) * M⁻¹
    # For numerical stability, check if M is nearly singular
    det_M = M[1,1]*M[2,2] - M[1,2]*M[2,1]

    if abs(det_M) < 1e-10
        # M nearly singular, use Taylor series: φ₁(M) ≈ I + M/2 + M²/6 + ...
        M2 = M * M
        return I2 + M/2 + M2/6
    else
        M_inv = inv(M)
        return (exp_M - I2) * M_inv
    end
end

"""
    update_etd!(filter::ExponentialMean, h::AbstractArray, coeffs::ETDFilterCoefficients)

Update exponential mean filter using ETD1 (Exponential Euler).

# Formula
    h̄ⁿ⁺¹ = exp(-α·dt)·h̄ⁿ + (1 - exp(-α·dt))·hⁿ

This is **unconditionally stable** for any timestep size!
"""
function update_etd!(
    filter::ExponentialMean{T, N},
    h::AbstractArray{T, N},
    coeffs::ETDFilterCoefficients{T}
) where {T, N}

    exp_factor = coeffs.exp_scalar
    phi1_factor = coeffs.phi1_scalar * coeffs.α  # (1 - exp(-α·dt))
    h̄ = filter.h̄

    @inbounds @simd for i in eachindex(h̄)
        h̄[i] = exp_factor * h̄[i] + phi1_factor * h[i]
    end

    return h̄
end

"""
    update_etd!(filter::ButterworthFilter, h::AbstractArray, coeffs::ETDFilterCoefficients)

Update Butterworth filter using ETD1 (Exponential Euler).

# Formula
    [h̃; h̄]ⁿ⁺¹ = exp(L·dt)·[h̃; h̄]ⁿ + φ₁(L·dt)·dt·α·[hⁿ; 0]

The matrix exponential handles the complex eigenvalues exactly,
making this **unconditionally stable** for any timestep!
"""
function update_etd!(
    filter::ButterworthFilter{T, N},
    h::AbstractArray{T, N},
    coeffs::ETDFilterCoefficients{T}
) where {T, N}

    exp_M = coeffs.exp_matrix
    phi1_M = coeffs.phi1_matrix
    α = coeffs.α
    h̃ = filter.h̃
    h̄ = filter.h̄

    @inbounds @simd for i in eachindex(h̄)
        h̃_i = h̃[i]
        h̄_i = h̄[i]
        f1 = α * h[i]  # Forcing: α·[h; 0]
        f2 = zero(T)

        # yⁿ⁺¹ = exp(L·dt)·yⁿ + φ₁(L·dt)·dt·f
        h̃[i] = exp_M[1,1]*h̃_i + exp_M[1,2]*h̄_i + phi1_M[1,1]*f1 + phi1_M[1,2]*f2
        h̄[i] = exp_M[2,1]*h̃_i + exp_M[2,2]*h̄_i + phi1_M[2,1]*f1 + phi1_M[2,2]*f2
    end

    return h̄
end


# ============================================================================
# Horizontal Mean (k=0 mode) Extraction
# ============================================================================

"""
    HorizontalMean{T, N, M}

Extracts and stores the horizontal mean (k=0 mode) of a field. The horizontal
mean is computed by averaging over specified dimensions, leaving a profile
that varies only in the remaining (typically vertical) dimension.

# Fields
- `mean_profile::Array{T, M}`: The horizontally-averaged profile
- `horizontal_dims::NTuple{H, Int}`: Dimensions to average over (e.g., (1,2) for x,y)
- `vertical_dim::Int`: The remaining dimension (e.g., 3 for z)
- `field_size::NTuple{N, Int}`: Original field size

# Example
```julia
# For a 3D field (Nx, Ny, Nz), extract the horizontal mean (average over x,y)
hmean = HorizontalMean((64, 64, 32); horizontal_dims=(1, 2))

# Update with current field
update!(hmean, velocity_field)

# Get the k=0 profile (varies only in z)
profile = get_profile(hmean)  # size (32,)

# Broadcast back to full field for subtraction
field_k0 = broadcast_profile(hmean)  # size (64, 64, 32)
fluctuation = velocity_field - field_k0
```
"""
mutable struct HorizontalMean{T<:AbstractFloat, N, M}
    mean_profile::Array{T, M}           # Reduced-dimension mean
    horizontal_dims::NTuple{M, Int}     # Dims to average (where M = N - 1 typically)
    vertical_dim::Int                   # Remaining dimension
    field_size::NTuple{N, Int}
    broadcast_buffer::Array{T, N}       # Preallocated for broadcasting back
end

"""
    HorizontalMean(field_size::NTuple{N, Int}; horizontal_dims, dtype=Float64)

Create a HorizontalMean extractor for fields of the given size.

# Arguments
- `field_size`: Size of the input field, e.g., `(Nx, Ny, Nz)`
- `horizontal_dims`: Tuple of dimensions to average over, e.g., `(1, 2)` for x,y
- `dtype`: Element type (default: Float64)

# Examples
```julia
# 3D: average over x,y to get z-profile
hmean_3d = HorizontalMean((64, 64, 32); horizontal_dims=(1, 2))

# 2D: average over x to get y-profile
hmean_2d = HorizontalMean((128, 64); horizontal_dims=(1,))

# 3D: average over x,z to get y-profile (e.g., channel flow)
hmean_channel = HorizontalMean((64, 32, 64); horizontal_dims=(1, 3))
```
"""
function HorizontalMean(
    field_size::NTuple{N, Int};
    horizontal_dims::NTuple{M, Int},
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N, M}

    # Validate dimensions
    all_dims = Set(1:N)
    hdims_set = Set(horizontal_dims)
    @assert hdims_set ⊆ all_dims "horizontal_dims must be valid dimensions"
    @assert length(hdims_set) == M "horizontal_dims must have unique elements"
    @assert M < N "Must have at least one non-horizontal dimension"

    # Find vertical dimension(s)
    remaining_dims = setdiff(all_dims, hdims_set)
    @assert length(remaining_dims) == 1 "Expected exactly one vertical dimension, got $(length(remaining_dims))"
    vertical_dim = first(remaining_dims)

    # Profile size (only vertical dimension)
    profile_size = field_size[vertical_dim]
    mean_profile = zeros(T, profile_size)

    # Preallocate broadcast buffer
    broadcast_buffer = zeros(T, field_size...)

    HorizontalMean{T, N, M}(
        mean_profile,
        horizontal_dims,
        vertical_dim,
        field_size,
        broadcast_buffer
    )
end

# Convenience constructors for common cases
"""
    HorizontalMean(Nx, Ny, Nz; dtype=Float64)

Create a HorizontalMean for 3D fields, averaging over x,y (dims 1,2).
"""
function HorizontalMean(Nx::Int, Ny::Int, Nz::Int; dtype::Type{T}=Float64) where T
    HorizontalMean((Nx, Ny, Nz); horizontal_dims=(1, 2), dtype=dtype)
end

"""
    HorizontalMean(Nx, Ny; dtype=Float64)

Create a HorizontalMean for 2D fields, averaging over x (dim 1).
"""
function HorizontalMean(Nx::Int, Ny::Int; dtype::Type{T}=Float64) where T
    HorizontalMean((Nx, Ny); horizontal_dims=(1,), dtype=dtype)
end

"""
    update!(hmean::HorizontalMean, field)

Compute the horizontal mean of `field` and store in `hmean`.
"""
function update!(
    hmean::HorizontalMean{T, N, M},
    field::AbstractArray{T, N}
) where {T, N, M}

    @assert size(field) == hmean.field_size "Field size mismatch"

    # Compute mean over horizontal dimensions
    profile = hmean.mean_profile
    vdim = hmean.vertical_dim
    hdims = hmean.horizontal_dims

    # Number of horizontal points for normalization
    n_horizontal = prod(hmean.field_size[d] for d in hdims)

    # Sum over horizontal dimensions
    fill!(profile, zero(T))

    if N == 2 && hdims == (1,)
        # 2D case: average over x (dim 1), profile in y (dim 2)
        Nx, Ny = hmean.field_size
        @inbounds for j in 1:Ny
            s = zero(T)
            @simd for i in 1:Nx
                s += field[i, j]
            end
            profile[j] = s / n_horizontal
        end

    elseif N == 3 && hdims == (1, 2)
        # 3D case: average over x,y (dims 1,2), profile in z (dim 3)
        Nx, Ny, Nz = hmean.field_size
        @inbounds for k in 1:Nz
            s = zero(T)
            for j in 1:Ny
                @simd for i in 1:Nx
                    s += field[i, j, k]
                end
            end
            profile[k] = s / n_horizontal
        end

    elseif N == 3 && hdims == (1, 3)
        # 3D channel: average over x,z (dims 1,3), profile in y (dim 2)
        Nx, Ny, Nz = hmean.field_size
        @inbounds for j in 1:Ny
            s = zero(T)
            for k in 1:Nz
                @simd for i in 1:Nx
                    s += field[i, j, k]
                end
            end
            profile[j] = s / n_horizontal
        end

    else
        # General case using dropdims and mean
        profile .= dropdims(sum(field; dims=hdims); dims=hdims) ./ n_horizontal
    end

    return profile
end

"""
    get_profile(hmean::HorizontalMean)

Return the current horizontal mean profile (1D array).
"""
get_profile(hmean::HorizontalMean) = hmean.mean_profile

"""
    broadcast_profile(hmean::HorizontalMean)

Broadcast the 1D profile back to the full field size.
Returns a preallocated array with the profile repeated along horizontal dimensions.
"""
function broadcast_profile(hmean::HorizontalMean{T, N, M}) where {T, N, M}

    buf = hmean.broadcast_buffer
    profile = hmean.mean_profile
    vdim = hmean.vertical_dim

    if N == 2 && hmean.horizontal_dims == (1,)
        # 2D: profile in y (dim 2)
        Nx, Ny = hmean.field_size
        @inbounds for j in 1:Ny
            pj = profile[j]
            @simd for i in 1:Nx
                buf[i, j] = pj
            end
        end

    elseif N == 3 && hmean.horizontal_dims == (1, 2)
        # 3D: profile in z (dim 3)
        Nx, Ny, Nz = hmean.field_size
        @inbounds for k in 1:Nz
            pk = profile[k]
            for j in 1:Ny
                @simd for i in 1:Nx
                    buf[i, j, k] = pk
                end
            end
        end

    elseif N == 3 && hmean.horizontal_dims == (1, 3)
        # 3D channel: profile in y (dim 2)
        Nx, Ny, Nz = hmean.field_size
        @inbounds for k in 1:Nz
            for j in 1:Ny
                pj = profile[j]
                @simd for i in 1:Nx
                    buf[i, j, k] = pj
                end
            end
        end

    else
        # General case
        # Reshape profile to broadcast correctly
        new_shape = ntuple(i -> i == vdim ? length(profile) : 1, N)
        buf .= reshape(profile, new_shape)
    end

    return buf
end

"""
    extract_fluctuation!(fluctuation, hmean, field)

Compute `fluctuation = field - horizontal_mean(field)` in-place.
Updates hmean and stores result in fluctuation.
"""
function extract_fluctuation!(
    fluctuation::AbstractArray{T, N},
    hmean::HorizontalMean{T, N, M},
    field::AbstractArray{T, N}
) where {T, N, M}

    update!(hmean, field)
    k0_field = broadcast_profile(hmean)
    @. fluctuation = field - k0_field
    return fluctuation
end

"""
    extract_k0_and_fluctuation(hmean, field)

Return both the k=0 profile and the fluctuation field.
"""
function extract_k0_and_fluctuation(
    hmean::HorizontalMean{T, N, M},
    field::AbstractArray{T, N}
) where {T, N, M}

    update!(hmean, field)
    profile = copy(hmean.mean_profile)
    k0_field = broadcast_profile(hmean)
    fluctuation = field .- k0_field
    return profile, fluctuation
end


# ============================================================================
# Combined Temporal + Horizontal Mean for Wave-Mean QL
# ============================================================================

"""
    WaveMeanDecomposition{T, N}

Combined temporal and horizontal averaging for quasi-linear wave-mean flow.

The mean flow is defined as: ⟨·⟩ = temporal_filter(horizontal_mean(·))

This provides:
1. `k=0` extraction (horizontal mean → profile)
2. Temporal filtering of the profile
3. Wave flux computation and filtering

# Example: Quasi-Linear Boussinesq
```julia
# Setup decomposition
decomp = WaveMeanDecomposition((64, 64, 32); α=0.1, horizontal_dims=(1,2))

# In time loop:
for step in 1:nsteps
    # Get mean profile and wave fluctuation
    u_mean_profile, u_wave = decompose!(decomp, :u, u_field, dt)

    # u_mean_profile is 1D (Nz,) - the temporally-filtered horizontal mean
    # u_wave is 3D (Nx, Ny, Nz) - the fluctuation

    # Compute and filter Reynolds stress
    update_flux!(decomp, :uw, u_wave .* w_wave, dt)
    R_uw = get_filtered_flux(decomp, :uw)  # ⟨u'w'⟩ as 1D profile

    # Use in mean equation
    # ∂ū/∂t = ... - ∂⟨u'w'⟩/∂z
end
```
"""
mutable struct WaveMeanDecomposition{T<:AbstractFloat, N, M}
    # Horizontal mean extractors
    hmean::HorizontalMean{T, N, M}

    # Temporal filters for mean profiles
    mean_filters::Dict{Symbol, ButterworthFilter{T, 1}}

    # Temporal filters for wave flux products (horizontally averaged)
    flux_filters::Dict{Symbol, ButterworthFilter{T, 1}}

    # ETD coefficients
    etd_coeffs::Union{ETDFilterCoefficients{T}, Nothing}

    # Parameters
    α::T
    field_size::NTuple{N, Int}
    profile_size::Int

    # Work arrays
    fluctuation::Array{T, N}
    flux_profile::Vector{T}
end

"""
    WaveMeanDecomposition(field_size; α, horizontal_dims=(1,2), dtype=Float64)

Create a wave-mean decomposition system.

# Arguments
- `field_size`: Size of 3D fields, e.g., `(Nx, Ny, Nz)`
- `α`: Temporal filter parameter (inverse averaging time)
- `horizontal_dims`: Dimensions to average over (default: `(1,2)` for x,y)
- `dtype`: Element type
"""
function WaveMeanDecomposition(
    field_size::NTuple{N, Int};
    α::Real,
    horizontal_dims::NTuple{M, Int} = N == 3 ? (1, 2) : (1,),
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N, M}

    # Create horizontal mean extractor
    hmean = HorizontalMean(field_size; horizontal_dims=horizontal_dims, dtype=dtype)
    profile_size = length(hmean.mean_profile)

    # Initialize empty filter dictionaries
    mean_filters = Dict{Symbol, ButterworthFilter{T, 1}}()
    flux_filters = Dict{Symbol, ButterworthFilter{T, 1}}()

    # Work arrays
    fluctuation = zeros(T, field_size...)
    flux_profile = zeros(T, profile_size)

    WaveMeanDecomposition{T, N, M}(
        hmean,
        mean_filters,
        flux_filters,
        nothing,  # ETD coefficients set later
        T(α),
        field_size,
        profile_size,
        fluctuation,
        flux_profile
    )
end

"""
    setup_etd!(decomp::WaveMeanDecomposition, dt)

Precompute ETD coefficients for the given timestep.
"""
function setup_etd!(decomp::WaveMeanDecomposition{T}, dt::Real) where T
    # Create a dummy filter to compute coefficients
    dummy_filter = ButterworthFilter((decomp.profile_size,); α=decomp.α, dtype=T)
    decomp.etd_coeffs = precompute_etd_coefficients(dummy_filter, dt)
    return decomp
end

"""
    add_mean_field!(decomp, name::Symbol)

Register a field for mean flow tracking.
"""
function add_mean_field!(decomp::WaveMeanDecomposition{T}, name::Symbol) where T
    if !haskey(decomp.mean_filters, name)
        decomp.mean_filters[name] = ButterworthFilter(
            (decomp.profile_size,); α=decomp.α, dtype=T
        )
    end
    return decomp
end

"""
    add_flux_field!(decomp, name::Symbol)

Register a wave flux product for filtering (e.g., :uw for ⟨u'w'⟩).
"""
function add_flux_field!(decomp::WaveMeanDecomposition{T}, name::Symbol) where T
    if !haskey(decomp.flux_filters, name)
        decomp.flux_filters[name] = ButterworthFilter(
            (decomp.profile_size,); α=decomp.α, dtype=T
        )
    end
    return decomp
end

"""
    decompose!(decomp, name, field, dt) -> (mean_profile, fluctuation)

Decompose field into temporally-filtered horizontal mean and fluctuation.

Returns:
- `mean_profile`: 1D profile of ⟨field⟩ (temporally + horizontally averaged)
- `fluctuation`: Full field minus the k=0 temporal mean
"""
function decompose!(
    decomp::WaveMeanDecomposition{T, N, M},
    name::Symbol,
    field::AbstractArray{T, N},
    dt::Real
) where {T, N, M}

    # Ensure field is registered
    add_mean_field!(decomp, name)

    # Ensure ETD coefficients are computed
    if decomp.etd_coeffs === nothing
        setup_etd!(decomp, dt)
    end

    # Step 1: Extract horizontal mean profile
    profile = update!(decomp.hmean, field)

    # Step 2: Temporally filter the profile
    filter = decomp.mean_filters[name]
    update_etd!(filter, profile, decomp.etd_coeffs)
    mean_profile = get_mean(filter)

    # Step 3: Compute fluctuation = field - broadcast(mean_profile)
    # Note: we use the FILTERED mean for the fluctuation
    decomp.hmean.mean_profile .= mean_profile
    k0_field = broadcast_profile(decomp.hmean)
    @. decomp.fluctuation = field - k0_field

    return mean_profile, decomp.fluctuation
end

"""
    update_flux!(decomp, name, flux_field, dt)

Update the temporal filter for a wave flux product.
The flux_field should be the PRODUCT of wave fields (e.g., u'*w').
"""
function update_flux!(
    decomp::WaveMeanDecomposition{T, N, M},
    name::Symbol,
    flux_field::AbstractArray{T, N},
    dt::Real
) where {T, N, M}

    # Ensure flux is registered
    add_flux_field!(decomp, name)

    # Ensure ETD coefficients are computed
    if decomp.etd_coeffs === nothing
        setup_etd!(decomp, dt)
    end

    # Step 1: Horizontal average of flux product
    flux_profile = update!(decomp.hmean, flux_field)

    # Step 2: Temporal filter
    filter = decomp.flux_filters[name]
    update_etd!(filter, flux_profile, decomp.etd_coeffs)

    return get_mean(filter)
end

"""
    get_mean_profile(decomp, name) -> Vector

Get the current temporally-filtered mean profile for field `name`.
"""
function get_mean_profile(decomp::WaveMeanDecomposition, name::Symbol)
    @assert haskey(decomp.mean_filters, name) "Field $name not registered"
    return get_mean(decomp.mean_filters[name])
end

"""
    get_filtered_flux(decomp, name) -> Vector

Get the current temporally-filtered flux profile.
"""
function get_filtered_flux(decomp::WaveMeanDecomposition, name::Symbol)
    @assert haskey(decomp.flux_filters, name) "Flux $name not registered"
    return get_mean(decomp.flux_filters[name])
end

"""
    broadcast_mean(decomp, name) -> Array

Broadcast the mean profile back to full field dimensions.
"""
function broadcast_mean(decomp::WaveMeanDecomposition, name::Symbol)
    profile = get_mean_profile(decomp, name)
    decomp.hmean.mean_profile .= profile
    return broadcast_profile(decomp.hmean)
end


# ============================================================================
# Wave-Induced Forcing for PDE RHS
# ============================================================================

"""
    WaveInducedForcing{T, N}

Compute wave-mean decomposition and filtered wave fluxes that can be used
in the RHS of mean flow equations. User applies their own differentiation.

This provides a clean interface for quasi-linear wave-mean flow coupling:

```julia
# Setup
forcing = WaveInducedForcing((Nx, Ny, Nz); α=0.1)

# Register which fields to decompose and which fluxes to compute
add_field!(forcing, :u)
add_field!(forcing, :v)
add_field!(forcing, :w)
add_field!(forcing, :b)
add_flux!(forcing, :uw)  # ⟨u'w'⟩ for ∂ū/∂t equation
add_flux!(forcing, :vw)  # ⟨v'w'⟩ for ∂v̄/∂t equation
add_flux!(forcing, :wb)  # ⟨w'b'⟩ for ∂b̄/∂t equation

# In time loop - update with current fields
update!(forcing, Dict(:u => u, :v => v, :w => w, :b => b), dt)

# Get filtered flux profile (1D)
R_uw = get_flux(forcing, :uw)   # Returns 1D profile ⟨u'w'⟩(z)

# Get flux as 3D field (broadcast profile)
R_uw_3d = get_flux_3d(forcing, :uw)  # Returns 3D array

# User applies their own derivative for forcing term:
# F_u = -∂⟨u'w'⟩/∂z  (use your spectral/FD derivative)
```

The module provides:
- Horizontal averaging (k=0 extraction)
- Temporal filtering (Butterworth with ETD)
- User applies differentiation externally
"""
mutable struct WaveInducedForcing{T<:AbstractFloat, N, M}
    # Wave-mean decomposition system
    decomp::WaveMeanDecomposition{T, N, M}

    # Registered field names
    field_names::Vector{Symbol}

    # Flux specifications: Dict(:uw => (:u, :w)) means ⟨u'w'⟩
    flux_specs::Dict{Symbol, Tuple{Symbol, Symbol}}

    # Cached wave fluctuations for computing products
    wave_fields::Dict{Symbol, Array{T, N}}

    # Parameters
    field_size::NTuple{N, Int}
    profile_size::Int
end

"""
    WaveInducedForcing(field_size; α, horizontal_dims=(1,2), dtype=Float64)

Create a wave-induced forcing calculator.

# Arguments
- `field_size`: Size of 3D fields, e.g., `(Nx, Ny, Nz)`
- `α`: Temporal filter parameter (inverse averaging time)
- `horizontal_dims`: Dimensions to average over (default: `(1,2)` for x,y)
- `dtype`: Element type

# Example
```julia
forcing = WaveInducedForcing((64, 64, 32); α=0.1)
```
"""
function WaveInducedForcing(
    field_size::NTuple{N, Int};
    α::Real,
    horizontal_dims::NTuple{M, Int} = N == 3 ? (1, 2) : (1,),
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N, M}

    decomp = WaveMeanDecomposition(field_size; α=α, horizontal_dims=horizontal_dims, dtype=dtype)
    profile_size = decomp.profile_size

    WaveInducedForcing{T, N, M}(
        decomp,
        Symbol[],
        Dict{Symbol, Tuple{Symbol, Symbol}}(),
        Dict{Symbol, Array{T, N}}(),
        field_size,
        profile_size
    )
end

"""
    add_field!(forcing::WaveInducedForcing, name::Symbol)

Register a field for wave-mean decomposition.
"""
function add_field!(forcing::WaveInducedForcing{T, N}, name::Symbol) where {T, N}
    if !(name in forcing.field_names)
        push!(forcing.field_names, name)
        forcing.wave_fields[name] = zeros(T, forcing.field_size...)
        add_mean_field!(forcing.decomp, name)
    end
    return forcing
end

"""
    add_flux!(forcing::WaveInducedForcing, flux_name::Symbol, field1::Symbol, field2::Symbol)

Register a wave flux product ⟨field1' * field2'⟩.

# Example
```julia
add_flux!(forcing, :uw, :u, :w)  # ⟨u'w'⟩
add_flux!(forcing, :wb, :w, :b)  # ⟨w'b'⟩
```
"""
function add_flux!(
    forcing::WaveInducedForcing{T, N},
    flux_name::Symbol,
    field1::Symbol,
    field2::Symbol
) where {T, N}

    forcing.flux_specs[flux_name] = (field1, field2)
    add_flux_field!(forcing.decomp, flux_name)
    return forcing
end

# Convenience: infer fields from flux name like :uw -> (:u, :w)
function add_flux!(forcing::WaveInducedForcing, flux_name::Symbol)
    name_str = string(flux_name)
    if length(name_str) == 2
        field1 = Symbol(name_str[1])
        field2 = Symbol(name_str[2])
        return add_flux!(forcing, flux_name, field1, field2)
    else
        throw(ArgumentError("Cannot infer fields from flux name :$flux_name. Use add_flux!(forcing, :name, :field1, :field2)"))
    end
end

"""
    setup!(forcing::WaveInducedForcing, dt)

Initialize ETD coefficients. Called automatically on first update.
"""
function setup!(forcing::WaveInducedForcing, dt::Real)
    setup_etd!(forcing.decomp, dt)
    return forcing
end

"""
    update!(forcing::WaveInducedForcing, fields::Dict{Symbol, AbstractArray}, dt)

Update all filters with current field values.

# Arguments
- `fields`: Dictionary mapping field names to their current values
- `dt`: Timestep

# Example
```julia
update!(forcing, Dict(:u => u_field, :v => v_field, :w => w_field, :b => b_field), dt)
```
"""
function update!(
    forcing::WaveInducedForcing{T, N, M},
    fields::Dict{Symbol, <:AbstractArray{T, N}},
    dt::Real
) where {T, N, M}

    # Ensure setup is done
    if forcing.decomp.etd_coeffs === nothing
        setup!(forcing, dt)
    end

    # Step 1: Decompose all registered fields into mean + wave
    for name in forcing.field_names
        if haskey(fields, name)
            _, wave = decompose!(forcing.decomp, name, fields[name], dt)
            forcing.wave_fields[name] .= wave
        end
    end

    # Step 2: Compute and filter all flux products
    for (flux_name, (f1, f2)) in forcing.flux_specs
        if haskey(forcing.wave_fields, f1) && haskey(forcing.wave_fields, f2)
            wave1 = forcing.wave_fields[f1]
            wave2 = forcing.wave_fields[f2]
            # Compute horizontally-averaged, temporally-filtered flux
            update_flux!(forcing.decomp, flux_name, wave1 .* wave2, dt)
        end
    end

    return forcing
end

"""
    get_flux(forcing::WaveInducedForcing, flux_name::Symbol) -> Vector

Get the filtered wave flux profile (1D array).
"""
function get_flux(forcing::WaveInducedForcing, flux_name::Symbol)
    return get_filtered_flux(forcing.decomp, flux_name)
end

"""
    get_flux_3d(forcing::WaveInducedForcing, flux_name::Symbol) -> Array

Get the filtered wave flux broadcast to full 3D field size.
"""
function get_flux_3d(forcing::WaveInducedForcing, flux_name::Symbol)
    flux_profile = get_filtered_flux(forcing.decomp, flux_name)
    forcing.decomp.hmean.mean_profile .= flux_profile
    return broadcast_profile(forcing.decomp.hmean)
end

"""
    get_mean(forcing::WaveInducedForcing, field_name::Symbol) -> Vector

Get the temporally-filtered horizontal mean profile (1D array).
"""
function get_mean(forcing::WaveInducedForcing, field_name::Symbol)
    return get_mean_profile(forcing.decomp, field_name)
end

"""
    get_mean_3d(forcing::WaveInducedForcing, field_name::Symbol) -> Array

Get the temporally-filtered horizontal mean broadcast to full 3D field size.
"""
function get_mean_3d(forcing::WaveInducedForcing, field_name::Symbol)
    mean_profile = get_mean_profile(forcing.decomp, field_name)
    forcing.decomp.hmean.mean_profile .= mean_profile
    return broadcast_profile(forcing.decomp.hmean)
end

"""
    get_wave(forcing::WaveInducedForcing, field_name::Symbol) -> Array

Get the wave (fluctuation) field (3D array).
"""
function get_wave(forcing::WaveInducedForcing, field_name::Symbol)
    if haskey(forcing.wave_fields, field_name)
        return forcing.wave_fields[field_name]
    else
        throw(KeyError("Field :$field_name not registered. Use add_field!() first."))
    end
end


# ============================================================================
# Generalized Quasi-Linear (GQL) Wavenumber Decomposition
# ============================================================================

"""
    GQLDecomposition{T, N}

Generalized Quasi-Linear (GQL) decomposition using Fourier wavenumber cutoff.

Splits a field into "large-scale" (low wavenumber, |k| ≤ Λ) and "small-scale"
(high wavenumber, |k| > Λ) components in spectral space.

This follows the GQL approximation of Marston, Chini & Tobias (2016):
- **QL (Quasi-Linear)**: Λ = 0, only k=0 mode is "large scale"
- **GQL**: 0 < Λ < k_max, intermediate cutoff
- **Full NL**: Λ = k_max, all modes are "large scale" (no approximation)

```julia
# Setup for 3D field with cutoff at |k| = 4
gql = GQLDecomposition((Nx, Ny, Nz), (Lx, Ly); Λ=4)

# Decompose field (requires FFT of field)
f_hat = fft(f)  # User performs FFT
f_large, f_small = decompose!(gql, f_hat)

# f_large: modes with |k| ≤ Λ (includes k=0)
# f_small: modes with |k| > Λ

# For GQL dynamics:
# - Large-scale eqn: ∂f_L/∂t = NL(f_L, f_L) + NL(f_S, f_S)|_L
# - Small-scale eqn: ∂f_S/∂t = NL(f_L, f_S) + NL(f_S, f_L)  [no NL(f_S, f_S)]
```

Reference: Marston, Chini & Tobias (2016), Phys. Rev. Lett. 116, 214501
"""
mutable struct GQLDecomposition{T<:AbstractFloat, N}
    # Cutoff wavenumber
    Λ::T

    # Wavenumber arrays
    kx::Vector{T}
    ky::Vector{T}

    # Precomputed mask: true for |k| ≤ Λ (large scale)
    large_scale_mask::Array{Bool, N}

    # Work arrays for decomposition (spectral space, complex)
    f_large::Array{Complex{T}, N}
    f_small::Array{Complex{T}, N}

    # Grid info
    field_size::NTuple{N, Int}
    spectral_size::NTuple{N, Int}  # Size after rfft
end

"""
    GQLDecomposition(field_size, domain_size; Λ, dtype=Float64)

Create a GQL decomposition with wavenumber cutoff Λ.

# Arguments
- `field_size`: Size of physical space field, e.g., `(Nx, Ny)` or `(Nx, Ny, Nz)`
- `domain_size`: Physical domain size, e.g., `(Lx, Ly)` for horizontal directions
- `Λ`: Cutoff wavenumber. Modes with |k| ≤ Λ are "large scale"
- `dtype`: Element type

# Wavenumber computation
For a periodic domain of size L with N points:
- kx = 2π/L * [0, 1, 2, ..., N/2, -N/2+1, ..., -1] (for full FFT)
- For rfft, only non-negative kx are stored

# Example
```julia
# 2D field 64×64, domain 2π×2π, cutoff at |k|=4
gql = GQLDecomposition((64, 64), (2π, 2π); Λ=4.0)

# 3D field with horizontal cutoff only
gql = GQLDecomposition((64, 64, 32), (2π, 2π); Λ=8.0)
```
"""
function GQLDecomposition(
    field_size::NTuple{N, Int},
    domain_size::NTuple{M, Real};
    Λ::Real,
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N, M}

    # Compute wavenumber arrays for horizontal dimensions
    # Assuming rfft along first dimension
    Nx = field_size[1]
    Lx = T(domain_size[1])

    # For rfft: kx = [0, 1, 2, ..., Nx/2] * 2π/Lx
    nkx = Nx ÷ 2 + 1
    kx = T[(2π / Lx) * i for i in 0:nkx-1]

    # For second dimension (if exists): full FFT wavenumbers
    if N >= 2 && M >= 2
        Ny = field_size[2]
        Ly = T(domain_size[2])
        # ky = [0, 1, ..., Ny/2, -Ny/2+1, ..., -1] * 2π/Ly
        ky = zeros(T, Ny)
        for j in 0:Ny-1
            if j <= Ny ÷ 2
                ky[j+1] = (2π / Ly) * j
            else
                ky[j+1] = (2π / Ly) * (j - Ny)
            end
        end
    else
        ky = T[0]
    end

    # Spectral size (after rfft along first dim)
    spectral_size = if N == 2
        (nkx, field_size[2])
    elseif N == 3
        (nkx, field_size[2], field_size[3])
    else
        (nkx,)
    end

    # Build large-scale mask: |k| ≤ Λ
    Λ_T = T(Λ)
    large_scale_mask = zeros(Bool, spectral_size...)

    if N == 1
        for i in 1:nkx
            k_mag = abs(kx[i])
            large_scale_mask[i] = (k_mag <= Λ_T)
        end
    elseif N == 2
        Ny = field_size[2]
        for j in 1:Ny, i in 1:nkx
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            large_scale_mask[i, j] = (k_mag <= Λ_T)
        end
    elseif N == 3
        Ny = field_size[2]
        Nz = field_size[3]
        # For 3D, cutoff is in horizontal (kx, ky) only
        for k in 1:Nz, j in 1:Ny, i in 1:nkx
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            large_scale_mask[i, j, k] = (k_mag <= Λ_T)
        end
    end

    # Allocate work arrays
    f_large = zeros(Complex{T}, spectral_size...)
    f_small = zeros(Complex{T}, spectral_size...)

    GQLDecomposition{T, N}(
        Λ_T,
        kx, ky,
        large_scale_mask,
        f_large, f_small,
        field_size,
        spectral_size
    )
end

"""
    decompose!(gql::GQLDecomposition, f_hat) -> (f_large, f_small)

Decompose spectral field into large-scale (|k| ≤ Λ) and small-scale (|k| > Λ) parts.

# Arguments
- `f_hat`: Field in spectral space (after rfft)

# Returns
- `f_large`: Large-scale (low-k) component
- `f_small`: Small-scale (high-k) component

Note: Returns references to internal arrays. Copy if you need to store.
"""
function decompose!(gql::GQLDecomposition{T, N}, f_hat::AbstractArray{Complex{T}, N}) where {T, N}
    mask = gql.large_scale_mask

    @inbounds @simd for i in eachindex(f_hat)
        if mask[i]
            gql.f_large[i] = f_hat[i]
            gql.f_small[i] = zero(Complex{T})
        else
            gql.f_large[i] = zero(Complex{T})
            gql.f_small[i] = f_hat[i]
        end
    end

    return gql.f_large, gql.f_small
end

"""
    project_large!(gql::GQLDecomposition, f_hat) -> f_large

Project field onto large-scale modes (|k| ≤ Λ). Modifies f_hat in-place.
"""
function project_large!(gql::GQLDecomposition{T, N}, f_hat::AbstractArray{Complex{T}, N}) where {T, N}
    mask = gql.large_scale_mask

    @inbounds @simd for i in eachindex(f_hat)
        if !mask[i]
            f_hat[i] = zero(Complex{T})
        end
    end

    return f_hat
end

"""
    project_small!(gql::GQLDecomposition, f_hat) -> f_small

Project field onto small-scale modes (|k| > Λ). Modifies f_hat in-place.
"""
function project_small!(gql::GQLDecomposition{T, N}, f_hat::AbstractArray{Complex{T}, N}) where {T, N}
    mask = gql.large_scale_mask

    @inbounds @simd for i in eachindex(f_hat)
        if mask[i]
            f_hat[i] = zero(Complex{T})
        end
    end

    return f_hat
end

"""
    get_cutoff(gql::GQLDecomposition) -> Λ

Get the wavenumber cutoff.
"""
get_cutoff(gql::GQLDecomposition) = gql.Λ

"""
    set_cutoff!(gql::GQLDecomposition, Λ_new)

Update the wavenumber cutoff and rebuild the mask.
"""
function set_cutoff!(gql::GQLDecomposition{T, N}, Λ_new::Real) where {T, N}
    gql.Λ = T(Λ_new)

    kx, ky = gql.kx, gql.ky
    mask = gql.large_scale_mask
    Λ_T = gql.Λ

    if N == 1
        for i in eachindex(kx)
            mask[i] = (abs(kx[i]) <= Λ_T)
        end
    elseif N == 2
        Ny = size(mask, 2)
        for j in 1:Ny, i in eachindex(kx)
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            mask[i, j] = (k_mag <= Λ_T)
        end
    elseif N == 3
        Ny, Nz = size(mask, 2), size(mask, 3)
        for k in 1:Nz, j in 1:Ny, i in eachindex(kx)
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            mask[i, j, k] = (k_mag <= Λ_T)
        end
    end

    return gql
end

"""
    count_large_modes(gql::GQLDecomposition) -> Int

Count number of large-scale (|k| ≤ Λ) modes.
"""
count_large_modes(gql::GQLDecomposition) = sum(gql.large_scale_mask)

"""
    count_small_modes(gql::GQLDecomposition) -> Int

Count number of small-scale (|k| > Λ) modes.
"""
count_small_modes(gql::GQLDecomposition) = sum(.!gql.large_scale_mask)


# ============================================================================
# GQL + Temporal Filter Combined System
# ============================================================================

"""
    GQLWaveMeanSystem{T, N}

Combined GQL wavenumber decomposition with temporal filtering for wave-mean
flow interactions. This is the full Generalized Quasi-Linear system.

Decomposes fields into:
1. **Large-scale (L)**: |k| ≤ Λ, includes mean flow
2. **Small-scale (S)**: |k| > Λ, wave/eddy field

And applies temporal filtering to extract slowly-varying mean from large-scale.

```julia
# Setup
sys = GQLWaveMeanSystem((Nx, Ny, Nz), (Lx, Ly); Λ=4.0, α=0.1)

# Register fields
add_field!(sys, :u)
add_field!(sys, :w)
add_flux!(sys, :uw)

# In time loop (user provides FFT'd fields)
update!(sys, Dict(:u => u_hat, :w => w_hat), dt)

# Get decomposition
u_L = get_large(sys, :u)      # Large-scale (spectral)
u_S = get_small(sys, :u)      # Small-scale (spectral)
u_mean = get_mean(sys, :u)    # Temporally-filtered mean profile (1D)

# Get filtered Reynolds stress
R_uw = get_flux(sys, :uw)     # ⟨u'w'⟩(z) profile
```

Reference: Marston, Chini & Tobias (2016), Phys. Rev. Lett. 116, 214501
"""
mutable struct GQLWaveMeanSystem{T<:AbstractFloat, N, M}
    # GQL spectral decomposition
    gql::GQLDecomposition{T, N}

    # Temporal filter for mean extraction
    decomp::WaveMeanDecomposition{T, N, M}

    # Registered fields
    field_names::Vector{Symbol}

    # Flux specifications
    flux_specs::Dict{Symbol, Tuple{Symbol, Symbol}}

    # Cached spectral decompositions
    large_fields::Dict{Symbol, Array{Complex{T}, N}}
    small_fields::Dict{Symbol, Array{Complex{T}, N}}

    # Physical space wave fields (for flux computation)
    wave_fields_phys::Dict{Symbol, Array{T, N}}

    # Parameters
    field_size::NTuple{N, Int}
    spectral_size::NTuple{N, Int}
end

"""
    GQLWaveMeanSystem(field_size, domain_size; Λ, α, horizontal_dims=(1,2), dtype=Float64)

Create a combined GQL + temporal filtering system.

# Arguments
- `field_size`: Physical space size, e.g., `(Nx, Ny, Nz)`
- `domain_size`: Horizontal domain size, e.g., `(Lx, Ly)`
- `Λ`: GQL wavenumber cutoff
- `α`: Temporal filter parameter (inverse averaging time)
- `horizontal_dims`: Dimensions for horizontal averaging
- `dtype`: Element type
"""
function GQLWaveMeanSystem(
    field_size::NTuple{N, Int},
    domain_size::NTuple{M, Real};
    Λ::Real,
    α::Real,
    horizontal_dims::NTuple{P, Int} = N == 3 ? (1, 2) : (1,),
    dtype::Type{T} = Float64
) where {T<:AbstractFloat, N, M, P}

    gql = GQLDecomposition(field_size, domain_size; Λ=Λ, dtype=dtype)
    decomp = WaveMeanDecomposition(field_size; α=α, horizontal_dims=horizontal_dims, dtype=dtype)

    GQLWaveMeanSystem{T, N, P}(
        gql,
        decomp,
        Symbol[],
        Dict{Symbol, Tuple{Symbol, Symbol}}(),
        Dict{Symbol, Array{Complex{T}, N}}(),
        Dict{Symbol, Array{Complex{T}, N}}(),
        Dict{Symbol, Array{T, N}}(),
        field_size,
        gql.spectral_size
    )
end

"""
    add_field!(sys::GQLWaveMeanSystem, name::Symbol)

Register a field for GQL decomposition.
"""
function add_field!(sys::GQLWaveMeanSystem{T, N}, name::Symbol) where {T, N}
    if !(name in sys.field_names)
        push!(sys.field_names, name)
        sys.large_fields[name] = zeros(Complex{T}, sys.spectral_size...)
        sys.small_fields[name] = zeros(Complex{T}, sys.spectral_size...)
        sys.wave_fields_phys[name] = zeros(T, sys.field_size...)
        add_mean_field!(sys.decomp, name)
    end
    return sys
end

"""
    add_flux!(sys::GQLWaveMeanSystem, flux_name::Symbol, field1::Symbol, field2::Symbol)

Register a wave flux product.
"""
function add_flux!(sys::GQLWaveMeanSystem, flux_name::Symbol, field1::Symbol, field2::Symbol)
    sys.flux_specs[flux_name] = (field1, field2)
    add_flux_field!(sys.decomp, flux_name)
    return sys
end

function add_flux!(sys::GQLWaveMeanSystem, flux_name::Symbol)
    name_str = string(flux_name)
    if length(name_str) == 2
        return add_flux!(sys, flux_name, Symbol(name_str[1]), Symbol(name_str[2]))
    else
        throw(ArgumentError("Cannot infer fields from flux name :$flux_name"))
    end
end

"""
    setup!(sys::GQLWaveMeanSystem, dt)

Initialize ETD coefficients.
"""
function setup!(sys::GQLWaveMeanSystem, dt::Real)
    setup_etd!(sys.decomp, dt)
    return sys
end

"""
    update!(sys::GQLWaveMeanSystem, fields_hat::Dict, fields_phys::Dict, dt)

Update GQL decomposition and temporal filters.

# Arguments
- `fields_hat`: Dict of spectral fields (after rfft)
- `fields_phys`: Dict of physical space fields (for flux computation)
- `dt`: Timestep
"""
function update!(
    sys::GQLWaveMeanSystem{T, N, M},
    fields_hat::Dict{Symbol, <:AbstractArray{Complex{T}, N}},
    fields_phys::Dict{Symbol, <:AbstractArray{T, N}},
    dt::Real
) where {T, N, M}

    # Ensure setup
    if sys.decomp.etd_coeffs === nothing
        setup!(sys, dt)
    end

    # Step 1: GQL decomposition in spectral space
    for name in sys.field_names
        if haskey(fields_hat, name)
            f_L, f_S = decompose!(sys.gql, fields_hat[name])
            sys.large_fields[name] .= f_L
            sys.small_fields[name] .= f_S
        end

        # Store physical space field for flux computation
        if haskey(fields_phys, name)
            sys.wave_fields_phys[name] .= fields_phys[name]
        end
    end

    # Step 2: Temporal filtering for mean profiles
    for name in sys.field_names
        if haskey(fields_phys, name)
            decompose!(sys.decomp, name, fields_phys[name], dt)
        end
    end

    # Step 3: Filter flux products (using physical space small-scale)
    for (flux_name, (f1, f2)) in sys.flux_specs
        if haskey(sys.wave_fields_phys, f1) && haskey(sys.wave_fields_phys, f2)
            # For GQL, compute flux from SMALL-scale fields
            # User should pass irfft(small_fields) as wave_fields_phys
            w1 = sys.wave_fields_phys[f1]
            w2 = sys.wave_fields_phys[f2]
            update_flux!(sys.decomp, flux_name, w1 .* w2, dt)
        end
    end

    return sys
end

# Update variant for when user handles FFT externally
function update!(
    sys::GQLWaveMeanSystem{T, N, M},
    fields_phys::Dict{Symbol, <:AbstractArray{T, N}},
    dt::Real
) where {T, N, M}

    if sys.decomp.etd_coeffs === nothing
        setup!(sys, dt)
    end

    # Only temporal filtering (no spectral decomposition)
    for name in sys.field_names
        if haskey(fields_phys, name)
            _, wave = decompose!(sys.decomp, name, fields_phys[name], dt)
            sys.wave_fields_phys[name] .= wave
        end
    end

    for (flux_name, (f1, f2)) in sys.flux_specs
        if haskey(sys.wave_fields_phys, f1) && haskey(sys.wave_fields_phys, f2)
            w1 = sys.wave_fields_phys[f1]
            w2 = sys.wave_fields_phys[f2]
            update_flux!(sys.decomp, flux_name, w1 .* w2, dt)
        end
    end

    return sys
end

"""
    get_large(sys::GQLWaveMeanSystem, name::Symbol) -> Array{Complex}

Get large-scale (|k| ≤ Λ) spectral component.
"""
function get_large(sys::GQLWaveMeanSystem, name::Symbol)
    return sys.large_fields[name]
end

"""
    get_small(sys::GQLWaveMeanSystem, name::Symbol) -> Array{Complex}

Get small-scale (|k| > Λ) spectral component.
"""
function get_small(sys::GQLWaveMeanSystem, name::Symbol)
    return sys.small_fields[name]
end

"""
    get_mean(sys::GQLWaveMeanSystem, name::Symbol) -> Vector

Get temporally-filtered horizontal mean profile.
"""
function get_mean(sys::GQLWaveMeanSystem, name::Symbol)
    return get_mean_profile(sys.decomp, name)
end

"""
    get_flux(sys::GQLWaveMeanSystem, flux_name::Symbol) -> Vector

Get filtered wave flux profile.
"""
function get_flux(sys::GQLWaveMeanSystem, flux_name::Symbol)
    return get_filtered_flux(sys.decomp, flux_name)
end

"""
    get_cutoff(sys::GQLWaveMeanSystem) -> Real

Get the GQL wavenumber cutoff Λ.
"""
get_cutoff(sys::GQLWaveMeanSystem) = get_cutoff(sys.gql)

"""
    set_cutoff!(sys::GQLWaveMeanSystem, Λ_new)

Update the GQL wavenumber cutoff.
"""
set_cutoff!(sys::GQLWaveMeanSystem, Λ_new::Real) = set_cutoff!(sys.gql, Λ_new)


# ============================================================================
# Exports
# ============================================================================

export TemporalFilter
export ExponentialMean, ButterworthFilter, LagrangianFilter
export update!, get_mean, get_auxiliary, reset!, set_α!
export update_displacement!, lagrangian_mean!, get_mean_velocity, get_displacement
export filter_response, effective_averaging_time, max_stable_timestep
export IMEXFilterCoefficients, precompute_imex_coefficients, update_imex!
export ETDFilterCoefficients, precompute_etd_coefficients, update_etd!
export linear_operator_coefficients

# Horizontal mean exports
export HorizontalMean, get_profile, broadcast_profile
export extract_fluctuation!, extract_k0_and_fluctuation

# Wave-mean decomposition exports
export WaveMeanDecomposition, setup_etd!
export add_mean_field!, add_flux_field!
export decompose!, update_flux!
export get_mean_profile, get_filtered_flux, broadcast_mean

# Wave-induced forcing exports (for PDE RHS)
export WaveInducedForcing
export add_field!, add_flux!, setup!
export get_flux, get_flux_3d, get_mean, get_mean_3d, get_wave

# GQL decomposition exports
export GQLDecomposition
export project_large!, project_small!
export get_cutoff, set_cutoff!
export count_large_modes, count_small_modes

# GQL + Wave-Mean combined system exports
export GQLWaveMeanSystem
export get_large, get_small
