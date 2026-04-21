# ============================================================================
# Horizontal Mean (k=0 mode) Extraction
# ============================================================================

"""
    HorizontalMean{T, N, M, A}

Extracts and stores the horizontal mean (k=0 mode) of a field. The horizontal
mean is computed by averaging over specified dimensions, leaving a profile
that varies only in the remaining (typically vertical) dimension.

# Type Parameters
- `T`: Element type (e.g., Float64)
- `N`: Number of dimensions in the field
- `M`: Number of horizontal dimensions to average over
- `A`: Array type for the broadcast buffer (supports GPU arrays)

# Fields
- `mean_profile::Vector{T}`: The horizontally-averaged profile (1D, on CPU)
- `horizontal_dims::NTuple{M, Int}`: Dimensions to average over (e.g., (1,2) for x,y)
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
mutable struct HorizontalMean{T<:AbstractFloat, N, M, A<:AbstractArray{T, N}}
    mean_profile::Vector{T}             # 1D profile (always reduces to single dimension, on CPU)
    horizontal_dims::NTuple{M, Int}     # Dims to average (where M = N - 1 typically)
    vertical_dim::Int                   # Remaining dimension
    field_size::NTuple{N, Int}
    broadcast_buffer::A                 # Preallocated for broadcasting back (matches input device)
end

"""
    HorizontalMean(field_size::NTuple{N, Int}; horizontal_dims, dtype=Float64, array_like=nothing)

Create a HorizontalMean extractor for fields of the given size.

# Arguments
- `field_size`: Size of the input field, e.g., `(Nx, Ny, Nz)`
- `horizontal_dims`: Tuple of dimensions to average over, e.g., `(1, 2)` for x,y
- `dtype`: Element type (default: Float64)
- `array_like`: Optional array to match type (for GPU compatibility). If provided,
  the broadcast buffer will be allocated on the same device.

# Examples
```julia
# 3D: average over x,y to get z-profile
hmean_3d = HorizontalMean((64, 64, 32); horizontal_dims=(1, 2))

# 2D: average over x to get y-profile
hmean_2d = HorizontalMean((128, 64); horizontal_dims=(1,))

# 3D: average over x,z to get y-profile (e.g., channel flow)
hmean_channel = HorizontalMean((64, 32, 64); horizontal_dims=(1, 3))

# GPU-compatible
hmean_gpu = HorizontalMean((64, 64, 32); horizontal_dims=(1, 2), array_like=cu_field)
```
"""
function HorizontalMean(
    field_size::NTuple{N, Int};
    horizontal_dims::NTuple{M, Int},
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing
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
    mean_profile = zeros(T, profile_size)  # Always on CPU (small 1D array)

    # Preallocate broadcast buffer (GPU-compatible if array_like provided)
    if array_like === nothing
        broadcast_buffer = zeros(T, field_size...)
    else
        broadcast_buffer = similar_zeros(array_like, T, field_size...)
    end

    HorizontalMean{T, N, M, typeof(broadcast_buffer)}(
        mean_profile,
        horizontal_dims,
        vertical_dim,
        field_size,
        broadcast_buffer
    )
end

# Convenience constructors for common cases
"""
    HorizontalMean(Nx, Ny, Nz; dtype=Float64, array_like=nothing)

Create a HorizontalMean for 3D fields, averaging over x,y (dims 1,2).
"""
function HorizontalMean(Nx::Int, Ny::Int, Nz::Int; dtype::Type{T}=Float64, array_like=nothing) where T
    HorizontalMean((Nx, Ny, Nz); horizontal_dims=(1, 2), dtype=dtype, array_like=array_like)
end

"""
    HorizontalMean(Nx, Ny; dtype=Float64, array_like=nothing)

Create a HorizontalMean for 2D fields, averaging over x (dim 1).
"""
function HorizontalMean(Nx::Int, Ny::Int; dtype::Type{T}=Float64, array_like=nothing) where T
    HorizontalMean((Nx, Ny); horizontal_dims=(1,), dtype=dtype, array_like=array_like)
end

"""
    update!(hmean::HorizontalMean, field)

Compute the horizontal mean of `field` and store in `hmean`.
Uses sum with dims argument for GPU compatibility.
"""
function update!(
    hmean::HorizontalMean{T, N, M, A},
    field::AbstractArray{T, N}
) where {T, N, M, A}

    @assert size(field) == hmean.field_size "Field size mismatch"

    # Compute mean over horizontal dimensions
    profile = hmean.mean_profile
    hdims = hmean.horizontal_dims

    # Number of horizontal points for normalization
    n_horizontal = prod(hmean.field_size[d] for d in hdims)

    # Use sum with dims for GPU compatibility
    # The result needs to be squeezed to 1D and copied to profile
    summed = dropdims(sum(field; dims=hdims); dims=hdims)

    # Copy to profile (handles GPU -> CPU transfer if needed for profile storage)
    if profile isa Array && !(summed isa Array)
        # GPU field -> CPU profile: need explicit Array() conversion
        profile .= Array(summed) ./ n_horizontal
    else
        profile .= summed ./ n_horizontal
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
function broadcast_profile(hmean::HorizontalMean{T, N, M, A}) where {T, N, M, A}

    buf = hmean.broadcast_buffer
    profile = hmean.mean_profile
    vdim = hmean.vertical_dim

    # GPU-compatible: use reshape + broadcast for all cases
    # Reshape profile to broadcast correctly along vertical dimension
    new_shape = ntuple(i -> i == vdim ? length(profile) : 1, N)
    profile_dev = on_architecture(architecture(buf), profile)
    buf .= reshape(profile_dev, new_shape)

    return buf
end

"""
    extract_fluctuation!(fluctuation, hmean, field)

Compute `fluctuation = field - horizontal_mean(field)` in-place.
Updates hmean and stores result in fluctuation.
"""
function extract_fluctuation!(
    fluctuation::AbstractArray{T, N},
    hmean::HorizontalMean{T, N, M, A},
    field::AbstractArray{T, N}
) where {T, N, M, A}

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
    hmean::HorizontalMean{T, N, M, A},
    field::AbstractArray{T, N}
) where {T, N, M, A}

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
mutable struct WaveMeanDecomposition{T<:AbstractFloat, N, M, AField<:AbstractArray{T, N}}
    # Horizontal mean extractors
    hmean::HorizontalMean{T, N, M, AField}

    # Temporal filters for mean profiles (1D, always on CPU)
    mean_filters::Dict{Symbol, ButterworthFilter{T, 1, Array{T, 1}}}

    # Temporal filters for wave flux products (horizontally averaged, 1D, always on CPU)
    flux_filters::Dict{Symbol, ButterworthFilter{T, 1, Array{T, 1}}}

    # ETD coefficients
    etd_coeffs::Union{ETDFilterCoefficients{T}, Nothing}

    # Parameters
    α::T
    field_size::NTuple{N, Int}
    profile_size::Int

    # Work arrays (fluctuation matches input device, flux_profile is 1D on CPU)
    fluctuation::AField
    flux_profile::Vector{T}
end

"""
    WaveMeanDecomposition(field_size; α, horizontal_dims=(1,2), dtype=Float64, array_like=nothing)

Create a wave-mean decomposition system.

# Arguments
- `field_size`: Size of 3D fields, e.g., `(Nx, Ny, Nz)`
- `α`: Temporal filter parameter (inverse averaging time)
- `horizontal_dims`: Dimensions to average over (default: `(1,2)` for x,y)
- `dtype`: Element type
- `array_like`: Optional array to match type (for GPU compatibility)
"""
function WaveMeanDecomposition(
    field_size::NTuple{N, Int};
    α::Real,
    horizontal_dims::NTuple{M, Int} = N == 3 ? (1, 2) : (1,),
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing
) where {T<:AbstractFloat, N, M}

    # Create horizontal mean extractor (GPU-compatible)
    hmean = HorizontalMean(field_size; horizontal_dims=horizontal_dims, dtype=dtype, array_like=array_like)
    profile_size = length(hmean.mean_profile)

    # Initialize empty filter dictionaries (1D filters always on CPU)
    mean_filters = Dict{Symbol, ButterworthFilter{T, 1, Array{T, 1}}}()
    flux_filters = Dict{Symbol, ButterworthFilter{T, 1, Array{T, 1}}}()

    # Work arrays (fluctuation on device, flux_profile on CPU)
    if array_like === nothing
        fluctuation = zeros(T, field_size...)
    else
        fluctuation = similar_zeros(array_like, T, field_size...)
    end
    flux_profile = zeros(T, profile_size)

    WaveMeanDecomposition{T, N, M, typeof(fluctuation)}(
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
    decomp::WaveMeanDecomposition{T, N, M, AField},
    name::Symbol,
    field::AbstractArray{T, N},
    dt::Real
) where {T, N, M, AField}

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
    decomp::WaveMeanDecomposition{T, N, M, AField},
    name::Symbol,
    flux_field::AbstractArray{T, N},
    dt::Real
) where {T, N, M, AField}

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
mutable struct WaveInducedForcing{T<:AbstractFloat, N, M, AField<:AbstractArray{T, N}}
    # Wave-mean decomposition system
    decomp::WaveMeanDecomposition{T, N, M, AField}

    # Registered field names
    field_names::Vector{Symbol}

    # Flux specifications: Dict(:uw => (:u, :w)) means ⟨u'w'⟩
    flux_specs::Dict{Symbol, Tuple{Symbol, Symbol}}

    # Cached wave fluctuations for computing products (matches input device)
    wave_fields::Dict{Symbol, AField}

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
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing
) where {T<:AbstractFloat, N, M}

    decomp = WaveMeanDecomposition(field_size; α=α, horizontal_dims=horizontal_dims, dtype=dtype, array_like=array_like)
    profile_size = decomp.profile_size

    # Determine array type for wave_fields
    AFieldType = typeof(decomp.fluctuation)

    WaveInducedForcing{T, N, M, AFieldType}(
        decomp,
        Symbol[],
        Dict{Symbol, Tuple{Symbol, Symbol}}(),
        Dict{Symbol, AFieldType}(),
        field_size,
        profile_size
    )
end

"""
    add_field!(forcing::WaveInducedForcing, name::Symbol)

Register a field for wave-mean decomposition.
"""
function add_field!(forcing::WaveInducedForcing{T, N, M, AField}, name::Symbol) where {T, N, M, AField}
    if !(name in forcing.field_names)
        push!(forcing.field_names, name)
        # Create wave_field matching the device of existing arrays
        forcing.wave_fields[name] = similar_zeros(forcing.decomp.fluctuation, T, forcing.field_size...)
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
    forcing::WaveInducedForcing{T, N, M, AField},
    flux_name::Symbol,
    field1::Symbol,
    field2::Symbol
) where {T, N, M, AField}

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
    forcing::WaveInducedForcing{T, N, M, AField},
    fields::Dict{Symbol, <:AbstractArray{T, N}},
    dt::Real
) where {T, N, M, AField}

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


