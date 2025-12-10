"""
# Temporal Filters for Lagrangian Averaging

This module provides temporal filters for computing Lagrangian means in fluid dynamics,
enabling efficient wave-mean flow decomposition. Based on the paper:

    Minz, Baker, Kafiabad, Vanneste (2025) "Efficient Lagrangian averaging with
    exponential filters", Phys. Rev. Fluids 10, 074902.

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

    # Simplified update (neglecting advection term for now)
    # Full PDE: ∂ₜξ + ū·∇ξ = u∘(id + ξ) - ū
    # Simplified (small displacement): ∂ₜξ ≈ u - ū

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
            # Simplified PDEs (neglecting advection):
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

    # Simplified (neglecting advection): ∂ₜgᴸ = α(g∘Ξ - gᴸ)
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
# Exports
# ============================================================================

export TemporalFilter
export ExponentialMean, ButterworthFilter, LagrangianFilter
export update!, get_mean, get_auxiliary, reset!, set_α!
export update_displacement!, lagrangian_mean!, get_mean_velocity, get_displacement
export filter_response, effective_averaging_time, max_stable_timestep
