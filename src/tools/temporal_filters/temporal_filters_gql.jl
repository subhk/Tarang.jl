# GQL spectral decomposition and combined GQL + wave-mean helpers.

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
mutable struct GQLDecomposition{T<:AbstractFloat, N, AMask<:AbstractArray{Bool, N}, AComplex<:AbstractArray{Complex{T}, N}}
    # Cutoff wavenumber
    Λ::T

    # Wavenumber arrays (stored on CPU for wavenumber lookups)
    kx::Vector{T}
    ky::Vector{T}

    # Precomputed mask: true for |k| ≤ Λ (large scale)
    large_scale_mask::AMask

    # Work arrays for decomposition (spectral space, complex)
    f_large::AComplex
    f_small::AComplex

    # Grid info
    field_size::NTuple{N, Int}
    spectral_size::NTuple{N, Int}  # Size after rfft
end

"""
    GQLDecomposition(field_size, domain_size; Λ, dtype=Float64, array_like=nothing)

Create a GQL decomposition with wavenumber cutoff Λ.

# Arguments
- `field_size`: Size of physical space field, e.g., `(Nx, Ny)` or `(Nx, Ny, Nz)`
- `domain_size`: Physical domain size, e.g., `(Lx, Ly)` for horizontal directions
- `Λ`: Cutoff wavenumber. Modes with |k| ≤ Λ are "large scale"
- `dtype`: Element type
- `array_like`: Optional array to match type (for GPU compatibility). If provided,
  work arrays will be allocated on the same device.

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

# GPU-compatible (pass a CuArray to match device)
gql = GQLDecomposition((64, 64), (2π, 2π); Λ=4.0, array_like=cu_field)
```
"""
function GQLDecomposition(
    field_size::NTuple{N, Int},
    domain_size::NTuple{M, Real};
    Λ::Real,
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing
) where {T<:AbstractFloat, N, M}

    # Validate domain_size dimensions
    # For N=1: need M >= 1 (Lx)
    # For N=2: need M >= 2 (Lx, Ly)
    # For N=3: need M >= 2 (Lx, Ly) - kz not used in horizontal cutoff
    if N >= 2 && M < 2
        throw(ArgumentError("For $(N)D fields, domain_size must have at least 2 elements (Lx, Ly), got $M"))
    end
    if N == 1 && M < 1
        throw(ArgumentError("For 1D fields, domain_size must have at least 1 element (Lx), got $M"))
    end

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

    # Build large-scale mask on CPU first: |k| ≤ Λ
    Λ_T = T(Λ)
    large_scale_mask_cpu = zeros(Bool, spectral_size...)

    if N == 1
        for i in 1:nkx
            k_mag = abs(kx[i])
            large_scale_mask_cpu[i] = (k_mag <= Λ_T)
        end
    elseif N == 2
        Ny = field_size[2]
        for j in 1:Ny, i in 1:nkx
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            large_scale_mask_cpu[i, j] = (k_mag <= Λ_T)
        end
    elseif N == 3
        Ny = field_size[2]
        Nz = field_size[3]
        # For 3D, cutoff is in horizontal (kx, ky) only
        for k in 1:Nz, j in 1:Ny, i in 1:nkx
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            large_scale_mask_cpu[i, j, k] = (k_mag <= Λ_T)
        end
    end

    # Allocate work arrays (GPU-compatible if array_like provided)
    if array_like === nothing
        # CPU arrays
        large_scale_mask = large_scale_mask_cpu
        f_large = zeros(Complex{T}, spectral_size...)
        f_small = zeros(Complex{T}, spectral_size...)
    else
        # Match array type (GPU-compatible)
        # Copy mask to GPU
        large_scale_mask = similar(array_like, Bool, spectral_size...)
        copyto!(large_scale_mask, large_scale_mask_cpu)
        f_large = similar_zeros(array_like, Complex{T}, spectral_size...)
        f_small = similar_zeros(array_like, Complex{T}, spectral_size...)
    end

    GQLDecomposition{T, N, typeof(large_scale_mask), typeof(f_large)}(
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
function decompose!(gql::GQLDecomposition{T, N, AMask, AComplex}, f_hat::AbstractArray{Complex{T}, N}) where {T, N, AMask, AComplex}
    mask = gql.large_scale_mask
    z = zero(Complex{T})

    # GPU-compatible: use broadcasting with ifelse
    @. gql.f_large = ifelse(mask, f_hat, z)
    @. gql.f_small = ifelse(mask, z, f_hat)

    return gql.f_large, gql.f_small
end

"""
    project_large!(gql::GQLDecomposition, f_hat) -> f_large

Project field onto large-scale modes (|k| ≤ Λ). Modifies f_hat in-place.
"""
function project_large!(gql::GQLDecomposition{T, N, AMask, AComplex}, f_hat::AbstractArray{Complex{T}, N}) where {T, N, AMask, AComplex}
    mask = gql.large_scale_mask
    z = zero(Complex{T})

    # GPU-compatible: zero out small-scale modes using broadcasting
    @. f_hat = ifelse(mask, f_hat, z)

    return f_hat
end

"""
    project_small!(gql::GQLDecomposition, f_hat) -> f_small

Project field onto small-scale modes (|k| > Λ). Modifies f_hat in-place.
"""
function project_small!(gql::GQLDecomposition{T, N, AMask, AComplex}, f_hat::AbstractArray{Complex{T}, N}) where {T, N, AMask, AComplex}
    mask = gql.large_scale_mask
    z = zero(Complex{T})

    # GPU-compatible: zero out large-scale modes using broadcasting
    @. f_hat = ifelse(mask, z, f_hat)

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
function set_cutoff!(gql::GQLDecomposition{T, N, AMask, AComplex}, Λ_new::Real) where {T, N, AMask, AComplex}
    gql.Λ = T(Λ_new)

    kx, ky = gql.kx, gql.ky
    mask = gql.large_scale_mask
    Λ_T = gql.Λ

    # Build mask on CPU first (GPU-compatible)
    mask_size = size(mask)
    mask_cpu = zeros(Bool, mask_size...)

    if N == 1
        for i in eachindex(kx)
            mask_cpu[i] = (abs(kx[i]) <= Λ_T)
        end
    elseif N == 2
        Ny = mask_size[2]
        for j in 1:Ny, i in eachindex(kx)
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            mask_cpu[i, j] = (k_mag <= Λ_T)
        end
    elseif N == 3
        Ny, Nz = mask_size[2], mask_size[3]
        for k in 1:Nz, j in 1:Ny, i in eachindex(kx)
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            mask_cpu[i, j, k] = (k_mag <= Λ_T)
        end
    end

    # Copy to device (handles both CPU and GPU)
    copyto!(mask, mask_cpu)

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
mutable struct GQLWaveMeanSystem{T<:AbstractFloat, N, M, AField<:AbstractArray{T, N}, AComplex<:AbstractArray{Complex{T}, N}}
    # GQL spectral decomposition
    gql::GQLDecomposition{T, N, <:AbstractArray{Bool, N}, AComplex}

    # Temporal filter for mean extraction
    decomp::WaveMeanDecomposition{T, N, M, AField}

    # Registered fields
    field_names::Vector{Symbol}

    # Flux specifications
    flux_specs::Dict{Symbol, Tuple{Symbol, Symbol}}

    # Cached spectral decompositions (matches GQL device)
    large_fields::Dict{Symbol, AComplex}
    small_fields::Dict{Symbol, AComplex}

    # Physical space wave fields (for flux computation, matches input device)
    wave_fields_phys::Dict{Symbol, AField}

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
    dtype::Type{T} = Float64,
    array_like::Union{AbstractArray, Nothing} = nothing
) where {T<:AbstractFloat, N, M, P}

    gql = GQLDecomposition(field_size, domain_size; Λ=Λ, dtype=dtype, array_like=array_like)
    decomp = WaveMeanDecomposition(field_size; α=α, horizontal_dims=horizontal_dims, dtype=dtype, array_like=array_like)

    # Determine array types
    AFieldType = typeof(decomp.fluctuation)
    AComplexType = typeof(gql.f_large)

    GQLWaveMeanSystem{T, N, P, AFieldType, AComplexType}(
        gql,
        decomp,
        Symbol[],
        Dict{Symbol, Tuple{Symbol, Symbol}}(),
        Dict{Symbol, AComplexType}(),
        Dict{Symbol, AComplexType}(),
        Dict{Symbol, AFieldType}(),
        field_size,
        gql.spectral_size
    )
end

"""
    add_field!(sys::GQLWaveMeanSystem, name::Symbol)

Register a field for GQL decomposition.
"""
function add_field!(sys::GQLWaveMeanSystem{T, N, M, AField, AComplex}, name::Symbol) where {T, N, M, AField, AComplex}
    if !(name in sys.field_names)
        push!(sys.field_names, name)
        # Create arrays matching the device of existing arrays
        sys.large_fields[name] = similar_zeros(sys.gql.f_large, Complex{T}, sys.spectral_size...)
        sys.small_fields[name] = similar_zeros(sys.gql.f_large, Complex{T}, sys.spectral_size...)
        sys.wave_fields_phys[name] = similar_zeros(sys.decomp.fluctuation, T, sys.field_size...)
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
    sys::GQLWaveMeanSystem{T, N, M, AField, AComplex},
    fields_hat::Dict{Symbol, <:AbstractArray{Complex{T}, N}},
    fields_phys::Dict{Symbol, <:AbstractArray{T, N}},
    dt::Real
) where {T, N, M, AField, AComplex}

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
    sys::GQLWaveMeanSystem{T, N, M, AField, AComplex},
    fields_phys::Dict{Symbol, <:AbstractArray{T, N}},
    dt::Real
) where {T, N, M, AField, AComplex}

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

