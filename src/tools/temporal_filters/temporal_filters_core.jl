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
    ExponentialMean{T, N, A}

First-order exponential filter for temporal averaging.

The filter applies a weighted exponential average with kernel:
    k(t) = α·exp(-αt)·Θ(t)

where Θ(t) is the Heaviside function and α is the inverse averaging timescale.

The filtered field h̄(t) satisfies the ODE:
    dh̄/dt = α(h - h̄)

This corresponds to relaxation of h̄ toward h with rate α.

# Fields
- `α::T`: Inverse averaging timescale (user-specified cutoff frequency)
- `h̄::A`: Filtered (mean) field (supports CPU Array or GPU CuArray)
- `field_size::NTuple{N,Int}`: Size of the field arrays

# Transfer Function
In the frequency domain: H(ω) = α/(α + iω)
- Low frequencies (ω << α): pass through unchanged
- High frequencies (ω >> α): attenuated as α/ω

# Reference
Minz et al. (2025), Eq. (5)-(7)
"""
mutable struct ExponentialMean{T<:AbstractFloat, N, A<:AbstractArray{T,N}} <: TemporalFilter
    α::T                        # Inverse averaging timescale
    h̄::A                        # Filtered (mean) field
    h̄_prev::A                   # Previous timestep h̄ (for SBDF2)
    field_size::NTuple{N, Int}  # Size of field
end

"""
    ExponentialMean(field_size::NTuple{N,Int}; α::Real=0.5, dtype::Type{T}=Float64, array_like=nothing)

Construct a first-order exponential mean filter.

# Arguments
- `field_size`: Tuple specifying the dimensions of the field to filter

# Keyword Arguments
- `α`: Inverse averaging timescale (default: 0.5). Larger α means shorter
       averaging window and less filtering. Typically choose α such that
       α << ω_fast where ω_fast is the frequency of fast oscillations to remove.
- `dtype`: Floating point type (default: Float64)
- `array_like`: Optional array to match type (for GPU support). If provided,
                allocations will use the same array type (e.g., CuArray).

# Example
```julia
# Create filter for a 2D field (CPU)
filter = ExponentialMean((64, 64); α=0.5)

# Create filter for GPU arrays
using CUDA
gpu_array = CUDA.zeros(Float64, 64, 64)
filter_gpu = ExponentialMean((64, 64); α=0.5, array_like=gpu_array)

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
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing) where {T<:AbstractFloat, N}

    if array_like === nothing
        # CPU arrays
        h̄ = zeros(T, field_size)
        h̄_prev = zeros(T, field_size)
    else
        # Match array type (GPU-compatible)
        h̄ = similar_zeros(array_like, T, field_size...)
        h̄_prev = similar_zeros(array_like, T, field_size...)
    end
    ExponentialMean{T, N, typeof(h̄)}(T(α), h̄, h̄_prev, field_size)
end

"""
    update!(filter::ExponentialMean, h::AbstractArray, dt::Real)

Update the exponential mean filter with new field data.

Integrates the ODE: dh̄/dt = α(h - h̄) using forward Euler.

# Arguments
- `filter`: ExponentialMean filter to update
- `h`: Current field values
- `dt`: Timestep

# Notes
- Uses broadcasting for GPU compatibility
"""
function update!(filter::ExponentialMean{T, N, A}, h::AbstractArray{T, N}, dt::Real) where {T, N, A}
    # Pre-compute factor for efficiency
    α = filter.α
    αdt = α * T(dt)

    # Stability check: Forward Euler is unconditionally unstable for α·dt > 2.
    # The filter WILL diverge, so throw rather than warn.
    if αdt > 2
        throw(ArgumentError("Unstable timestep for ExponentialMean: α·dt = $αdt > 2. " *
            "Use a smaller dt or RK2 method."))
    end

    h̄ = filter.h̄
    h̄_prev = filter.h̄_prev

    # Save current h̄ to h̄_prev before updating
    h̄_prev .= h̄

    # Forward Euler: h̄_{n+1} = h̄_n + dt·α·(h_n - h̄_n) = (1 - αdt)h̄_n + αdt·h_n
    # Using broadcasting for GPU compatibility
    @. h̄ = h̄_prev + αdt * (h - h̄_prev)

    return h̄
end

"""
    update!(filter::ExponentialMean, h::AbstractArray, dt::Real, ::Val{:RK2})

Update using second-order Runge-Kutta (midpoint method) for improved accuracy.
Uses broadcasting for GPU compatibility.
"""
function update!(filter::ExponentialMean{T, N, A}, h::AbstractArray{T, N}, dt::Real, ::Val{:RK2}) where {T, N, A}
    α = filter.α
    dt_T = T(dt)
    dt_half = dt_T / 2
    h̄ = filter.h̄
    h̄_prev = filter.h̄_prev

    # Save current h̄ to h̄_prev before updating
    h̄_prev .= h̄

    # RK2 midpoint method using broadcasting
    # k1 = α * (h - h̄_curr)
    # h̄_mid = h̄_curr + dt_half * k1 = h̄_curr + dt_half * α * (h - h̄_curr)
    # k2 = α * (h - h̄_mid)
    # h̄_new = h̄_curr + dt_T * k2

    # Combined formula:
    # h̄_mid = h̄_prev + dt_half * α * (h - h̄_prev)
    # h̄_new = h̄_prev + dt_T * α * (h - h̄_mid)
    αdt_half = α * dt_half
    αdt = α * dt_T

    # Two-step update (need intermediate value)
    # Step 1: compute h̄_mid in h̄
    @. h̄ = h̄_prev + αdt_half * (h - h̄_prev)
    # Step 2: compute final h̄ using h̄_prev and h̄ (which holds h̄_mid)
    @. h̄ = h̄_prev + αdt * (h - h̄)

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
function reset!(filter::ExponentialMean{T, N, A}) where {T, N, A}
    fill!(filter.h̄, zero(T))
    fill!(filter.h̄_prev, zero(T))
end

"""
    set_α!(filter::ExponentialMean, α::Real)

Update the inverse averaging timescale.
"""
function set_α!(filter::ExponentialMean{T, N, A}, α::Real) where {T, N, A}
    filter.α = T(α)
end


# ============================================================================
# Second-order Butterworth Filter
# ============================================================================

"""
    ButterworthFilter{T, N, A}

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
- `h̃::A`: Auxiliary field (supports CPU Array or GPU CuArray)
- `h̄::A`: Filtered (mean) field
- `A_mat::SMatrix{2,2,T}`: Filter matrix
- `field_size::NTuple{N,Int}`: Size of field arrays

# Frequency Response
- Low frequencies (ω << α): pass through unchanged
- High frequencies (ω >> α): attenuated as (α/ω)² (-40 dB/decade)
- Much sharper cutoff than exponential mean (-20 dB/decade)

# Reference
Minz et al. (2025), Eq. (24)-(30), Section IV
"""
mutable struct ButterworthFilter{T<:AbstractFloat, N, Arr<:AbstractArray{T,N}} <: TemporalFilter
    α::T                            # Inverse averaging timescale
    h̃::Arr                          # Auxiliary field
    h̄::Arr                          # Filtered (mean) field
    h̃_prev::Arr                     # Previous timestep h̃ (for SBDF2)
    h̄_prev::Arr                     # Previous timestep h̄ (for SBDF2)
    A::SMatrix{2, 2, T, 4}          # Filter matrix
    field_size::NTuple{N, Int}      # Size of field
    rk2_workspace::Union{Nothing, Arr}  # Pre-allocated workspace for RK2 (lazy init)
end

"""
    ButterworthFilter(field_size::NTuple{N,Int}; α::Real=0.5, dtype::Type{T}=Float64, array_like=nothing)

Construct a second-order Butterworth filter.

# Arguments
- `field_size`: Tuple specifying the dimensions of the field to filter

# Keyword Arguments
- `α`: Inverse averaging timescale (default: 0.5). This is the cutoff frequency.
       Frequencies ω >> α are strongly attenuated (as (α/ω)²).
- `dtype`: Floating point type (default: Float64)
- `array_like`: Optional array to match type (for GPU support). If provided,
                allocations will use the same array type (e.g., CuArray).

# Example
```julia
# Create Butterworth filter for a 2D field (CPU)
filter = ButterworthFilter((64, 64); α=0.5)

# Create filter for GPU arrays
using CUDA
gpu_array = CUDA.zeros(Float64, 64, 64)
filter_gpu = ButterworthFilter((64, 64); α=0.5, array_like=gpu_array)

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
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing) where {T<:AbstractFloat, N}

    # Allocate arrays
    if array_like === nothing
        # CPU arrays
        h̃ = zeros(T, field_size)
        h̄ = zeros(T, field_size)
        h̃_prev = zeros(T, field_size)
        h̄_prev = zeros(T, field_size)
    else
        # Match array type (GPU-compatible)
        h̃ = similar_zeros(array_like, T, field_size...)
        h̄ = similar_zeros(array_like, T, field_size...)
        h̃_prev = similar_zeros(array_like, T, field_size...)
        h̄_prev = similar_zeros(array_like, T, field_size...)
    end

    # Filter matrix A from Eq. (30)
    # A = [√2-1  2-√2; -1  1]
    sqrt2 = sqrt(T(2))
    A = SMatrix{2, 2, T}(
        sqrt2 - 1,  # A[1,1]
        -one(T),    # A[2,1]
        2 - sqrt2,  # A[1,2]
        one(T)      # A[2,2]
    )

    ButterworthFilter{T, N, typeof(h̄)}(T(α), h̃, h̄, h̃_prev, h̄_prev, A, field_size, nothing)
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

# Notes
- Uses broadcasting for GPU compatibility
"""
function update!(filter::ButterworthFilter{T, N, Arr}, h::AbstractArray{T, N}, dt::Real) where {T, N, Arr}
    α = filter.α
    A = filter.A
    dt_T = T(dt)
    αdt = α * dt_T

    # Stability check: Forward Euler for Butterworth is unconditionally unstable for α·dt > √2.
    sqrt2 = sqrt(T(2))
    if αdt > sqrt2
        throw(ArgumentError("Unstable timestep for ButterworthFilter: α·dt = $αdt > √2 ≈ 1.414. " *
            "Use a smaller dt or RK2 method."))
    end

    # Extract matrix elements
    A11, A12 = A[1,1], A[1,2]

    h̃ = filter.h̃
    h̄ = filter.h̄
    h̃_prev = filter.h̃_prev
    h̄_prev = filter.h̄_prev

    # Save current values to history before updating
    h̃_prev .= h̃
    h̄_prev .= h̄

    # Forward Euler for coupled system using broadcasting (GPU-compatible)
    # dh̃ = α·(h - A11·h̃ - A12·h̄) = h - A11·h̃_prev - A12·h̄_prev
    # dh̄ = α·(h̃ - h̄) = h̃_prev - h̄_prev  [since A21=-1, A22=1]
    @. h̃ = h̃_prev + αdt * (h - A11 * h̃_prev - A12 * h̄_prev)
    @. h̄ = h̄_prev + αdt * (h̃_prev - h̄_prev)

    return h̄
end

"""
    update!(filter::ButterworthFilter, h::AbstractArray, dt::Real, ::Val{:RK2})

Update using second-order Runge-Kutta (midpoint method) for improved accuracy.
Uses broadcasting for GPU compatibility.

Note: RK2 on the coupled Butterworth system requires one temporary allocation
for the h̃_mid value, as both h̃ and h̄ finals depend on both midpoint values.
"""
function update!(filter::ButterworthFilter{T, N, Arr}, h::AbstractArray{T, N}, dt::Real, ::Val{:RK2}) where {T, N, Arr}
    α = filter.α
    A = filter.A
    dt_T = T(dt)
    dt_half = dt_T / 2
    A11, A12 = A[1,1], A[1,2]

    h̃ = filter.h̃
    h̄ = filter.h̄
    h̃_prev = filter.h̃_prev
    h̄_prev = filter.h̄_prev

    # Save current values to history before updating
    h̃_prev .= h̃
    h̄_prev .= h̄

    αdt_half = α * dt_half
    αdt = α * dt_T

    # Stage 1: compute midpoint values (store in h̃, h̄)
    @. h̃ = h̃_prev + αdt_half * (h - A11 * h̃_prev - A12 * h̄_prev)
    @. h̄ = h̄_prev + αdt_half * (h̃_prev - h̄_prev)

    # Stage 2: compute final values using midpoints
    # Save h̃_mid in pre-allocated workspace (lazy init on first use)
    if filter.rk2_workspace === nothing
        filter.rk2_workspace = similar(h̃)
    end
    temp_h̃_mid = filter.rk2_workspace
    temp_h̃_mid .= h̃
    @. h̃ = h̃_prev + αdt * (h - A11 * temp_h̃_mid - A12 * h̄)
    @. h̄ = h̄_prev + αdt * (temp_h̃_mid - h̄)

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
function reset!(filter::ButterworthFilter{T, N, Arr}) where {T, N, Arr}
    fill!(filter.h̃, zero(T))
    fill!(filter.h̄, zero(T))
    fill!(filter.h̃_prev, zero(T))
    fill!(filter.h̄_prev, zero(T))
end

"""
    set_α!(filter::ButterworthFilter, α::Real)

Update the inverse averaging timescale.
"""
function set_α!(filter::ButterworthFilter{T, N, Arr}, α::Real) where {T, N, Arr}
    filter.α = T(α)
end


# ============================================================================
# Lagrangian Filter with Lifting Map
# ============================================================================

"""
    LagrangianFilter{T, N, F<:TemporalFilter, Arr}

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
- `ξ::Arr`: Displacement field (periodic part of lifting map) - supports GPU arrays
- `ξ̃::Union{Arr, Nothing}`: Auxiliary displacement (for Butterworth only)
- `ū::Arr`: Mean velocity field
- `α::T`: Inverse averaging timescale
- `field_size::NTuple{N,Int}`: Spatial dimensions
- `ndims::Int`: Number of spatial dimensions (1, 2, or 3)

# Reference
Minz et al. (2025), Sections III.A (exponential) and IV.A (Butterworth)
"""
mutable struct LagrangianFilter{T<:AbstractFloat, N, F<:TemporalFilter, Arr<:AbstractArray{T}}
    temporal_filter::F              # Underlying filter for scalar fields
    ξ::Arr                          # Displacement: Ξ(x,t) - x
    ξ̃::Union{Arr, Nothing}          # Auxiliary displacement (Butterworth only)
    ū::Arr                          # Mean velocity
    α::T                            # Inverse averaging timescale
    field_size::NTuple{N, Int}      # Spatial size per component
    ndim::Int                       # Number of spatial dimensions
    _scalar_workspace::Union{AbstractArray{T}, Nothing}  # Pre-allocated workspace for lagrangian_mean!
end

"""
    LagrangianFilter(field_size::NTuple{N,Int}; α::Real=0.5, filter_type::Symbol=:butterworth, dtype::Type{T}=Float64, array_like=nothing)

Construct a Lagrangian filter for wave-mean flow decomposition.

# Arguments
- `field_size`: Tuple specifying the spatial dimensions of the field

# Keyword Arguments
- `α`: Inverse averaging timescale (default: 0.5)
- `filter_type`: `:exponential` or `:butterworth` (default: `:butterworth`)
- `dtype`: Floating point type (default: Float64)
- `array_like`: Optional array to match type (for GPU support)

# Example
```julia
# Create Lagrangian filter for 2D flow (CPU)
lag_filter = LagrangianFilter((64, 64); α=0.5, filter_type=:butterworth)

# Create filter for GPU arrays
using CUDA
gpu_array = CUDA.zeros(Float64, 64, 64)
lag_filter_gpu = LagrangianFilter((64, 64); α=0.5, array_like=gpu_array)

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
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing) where {T<:AbstractFloat, N}

    ndim = N

    # Create displacement arrays (one per spatial dimension)
    # Shape: (field_size..., ndim) to store vector components
    ξ_shape = (field_size..., ndim)

    if array_like === nothing
        # CPU arrays
        ξ = zeros(T, ξ_shape)
        ū = zeros(T, ξ_shape)
    else
        # Match array type (GPU-compatible)
        ξ = similar_zeros(array_like, T, ξ_shape...)
        ū = similar_zeros(array_like, T, ξ_shape...)
    end

    # Create temporal filter for scalar field averaging
    if filter_type == :exponential
        temporal_filter = ExponentialMean(field_size; α=α, dtype=dtype, array_like=array_like)
        ξ̃ = nothing
    elseif filter_type == :butterworth
        temporal_filter = ButterworthFilter(field_size; α=α, dtype=dtype, array_like=array_like)
        if array_like === nothing
            ξ̃ = zeros(T, ξ_shape)
        else
            ξ̃ = similar_zeros(array_like, T, ξ_shape...)
        end
    else
        throw(ArgumentError("filter_type must be :exponential or :butterworth, got :$filter_type"))
    end

    # Pre-allocate scalar workspace for lagrangian_mean! (avoids per-step allocation)
    if array_like === nothing
        scalar_ws = zeros(T, field_size)
    else
        scalar_ws = similar_zeros(array_like, T, field_size...)
    end

    LagrangianFilter{T, N, typeof(temporal_filter), typeof(ξ)}(
        temporal_filter, ξ, ξ̃, ū, T(α), field_size, ndim, scalar_ws)

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
    filter::LagrangianFilter{T, N, F, Arr},
    u::AbstractArray{T},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ExponentialMean, Arr}

    α = filter.α

    # For exponential mean: ū = α·ξ (Eq. 12)
    # Uses broadcasting for GPU compatibility
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
    filter::LagrangianFilter{T, N, F, Arr},
    u::AbstractArray{T},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ButterworthFilter, Arr}

    α = filter.α
    dt_T = T(dt)
    sqrt2_m1 = sqrt(T(2)) - one(T)  # √2 - 1

    ξ = filter.ξ
    ξ̃ = filter.ξ̃
    ū = filter.ū

    # For Butterworth: Eq. (38a)
    # ũ = α·[ξ - (√2-1)·ξ̃]
    # ū = α·ξ̃

    # Using broadcasting for GPU compatibility
    # ū_old = α * ξ̃ (current mean velocity)
    # ũ = α * (ξ - sqrt2_m1 * ξ̃)
    # ξ̃_new = ξ̃ + dt * (ũ - ū_old) = ξ̃ + dt * α * (ξ - sqrt2_m1 * ξ̃ - ξ̃)
    #       = ξ̃ + dt * α * (ξ - (sqrt2_m1 + 1) * ξ̃)
    #       = ξ̃ + dt * α * (ξ - sqrt2 * ξ̃)
    # ξ_new = ξ + dt * (u - ū_old) = ξ + dt * (u - α * ξ̃)
    # ū_new = α * ξ̃_new

    sqrt2 = sqrt(T(2))
    αdt = α * dt_T

    if interpolate_fn === nothing
        # Without interpolation - use broadcasting
        # Save old ξ̃ before updating (both equations must use values at the old time level)
        @. ū = ξ̃  # temporarily store old ξ̃ in ū
        @. ξ̃ = ξ̃ + αdt * (ξ - sqrt2 * ū)
        @. ξ = ξ + dt_T * (u - α * ū)
        @. ū = α * ξ̃
    else
        # With interpolation
        u_displaced = interpolate_fn(u, ξ)
        # Save old ξ̃ before updating (both equations must use values at the old time level)
        @. ū = ξ̃  # temporarily store old ξ̃ in ū
        @. ξ̃ = ξ̃ + αdt * (ξ - sqrt2 * ū)
        @. ξ = ξ + dt_T * (u_displaced - α * ū)
        @. ū = α * ξ̃
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
    filter::LagrangianFilter{T, N, F, Arr},
    gᴸ::AbstractArray{T, N},
    g::AbstractArray{T, N},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ExponentialMean, Arr}

    α = filter.α
    αdt = α * T(dt)

    # Small-displacement approximation: ∂ₜgᴸ = α(g∘Ξ - gᴸ)
    # Using broadcasting for GPU compatibility
    if interpolate_fn === nothing
        # g∘Ξ ≈ g (approximate)
        @. gᴸ = gᴸ + αdt * (g - gᴸ)
    else
        g_composed = interpolate_fn(g, filter.ξ)
        @. gᴸ = gᴸ + αdt * (g_composed - gᴸ)
    end

    return gᴸ
end

function lagrangian_mean!(
    filter::LagrangianFilter{T, N, F, Arr},
    gᴸ::AbstractArray{T, N},
    g::AbstractArray{T, N},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ButterworthFilter, Arr}
    throw(ArgumentError(
        "Butterworth LagrangianFilter requires 5 positional arguments: " *
        "lagrangian_mean!(filter, gᴸ, g̃, g, dt). " *
        "The auxiliary state g̃ is required for the two-stage Butterworth filter."))
end

function lagrangian_mean!(
    filter::LagrangianFilter{T, N, F, Arr},
    gᴸ::AbstractArray{T, N},
    g̃::AbstractArray{T, N},
    g::AbstractArray{T, N},
    dt::Real;
    interpolate_fn = nothing) where {T, N, F<:ButterworthFilter, Arr}

    α = filter.α
    αdt = α * T(dt)
    sqrt2 = sqrt(T(2))
    A11 = sqrt2 - 1
    A12 = 2 - sqrt2

    # From Eq. (37) using broadcasting for GPU compatibility:
    # ∂ₜg̃ = α(g∘Ξ - A11·g̃ - A12·gᴸ)
    # ∂ₜgᴸ = α(g̃ - gᴸ)

    # Save g̃ before update since gᴸ depends on old g̃ (use pre-allocated workspace)
    g̃_old = filter._scalar_workspace
    g̃_old .= g̃

    if interpolate_fn === nothing
        @. g̃ = g̃ + αdt * (g - A11 * g̃ - A12 * gᴸ)
        @. gᴸ = gᴸ + αdt * (g̃_old - gᴸ)
    else
        g_composed = interpolate_fn(g, filter.ξ)
        @. g̃ = g̃ + αdt * (g_composed - A11 * g̃ - A12 * gᴸ)
        @. gᴸ = gᴸ + αdt * (g̃_old - gᴸ)
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
function reset!(filter::LagrangianFilter{T, N, F, Arr}) where {T, N, F, Arr}
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
function set_α!(filter::LagrangianFilter{T, N, F, Arr}, α::Real) where {T, N, F, Arr}
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
function filter_response(filter::ExponentialMean{T, N, A}, ω::Real) where {T, N, A}
    α = filter.α
    # |H(ω)|² = α² / (α² + ω²)
    return α^2 / (α^2 + ω^2)
end

function filter_response(filter::ButterworthFilter{T, N, Arr}, ω::Real) where {T, N, Arr}
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
#
# RK2 (midpoint) approximate limits:
- ExponentialMean: `dt ≤ 2/α`
- ButterworthFilter: `dt ≈ 2.18/α`

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
        # Midpoint RK2 has the same linear stability limit as Euler for real negative eigenvalues.
        return 2 / filter.α
    else
        throw(ArgumentError("Unknown method: $method. Use :euler or :RK2"))
    end
end

function max_stable_timestep(filter::ButterworthFilter; method::Symbol=:euler)
    if method == :euler
        return sqrt(2) / filter.α
    elseif method == :RK2
        # Midpoint RK2 stability limit for eigenvalues -α/√2 (1 ± i).
        return 2.183 / filter.α
    else
        throw(ArgumentError("Unknown method: $method. Use :euler or :RK2"))
    end
end

function max_stable_timestep(filter::LagrangianFilter; method::Symbol=:euler)
    # Use the more restrictive Butterworth limit for safety
    return max_stable_timestep(filter.temporal_filter; method=method)
end

