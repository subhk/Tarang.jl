# Spectral diagnostics and wavenumber helpers used by flow analysis tools.

# Energy spectra (for Fourier bases)
"""
    energy_spectrum(velocity::VectorField; max_wavenumber=nothing, radial_average=true, binning=LinearBinning())

Calculate kinetic energy spectrum E(k) from velocity field.

Computes the kinetic energy spectrum E(k) = ∫∫ |û_i(k)|² dA(k) where the integral
is over a spherical (3D) or circular (2D) shell of radius k in wavenumber space.

Based on standard spectral turbulence analysis methods and follows Tarang patterns
for spectral field processing with PencilArrays/PencilFFTs integration.

# Arguments
- `velocity::VectorField`: Velocity field with Fourier bases
- `max_wavenumber::Union{Int,Nothing}=nothing`: Maximum wavenumber (default: Nyquist limit)
- `radial_average::Bool=true`: Whether to perform radial averaging in wavenumber space
- `binning::SpectrumBinning=LinearBinning()`: Binning configuration for smoothing

# Returns
- If radial_average=true: NamedTuple (k=wavenumbers, power=spectrum values, bin_counts=modes per bin)
- If radial_average=false: Dict{Tuple, Float64} with (kx, ky, ...) → E(k)

# Examples
```julia
# Standard energy spectrum
result = energy_spectrum(velocity)
plot(result.k, result.power, xscale=:log10, yscale=:log10)

# Smoothed with logarithmic bins
result = energy_spectrum(velocity, binning=LogBinning(bins_per_decade=8))

# Linear bins with width 2
result = energy_spectrum(velocity, binning=LinearBinning(bin_width=2.0))
```
"""
function energy_spectrum(velocity::VectorField;
                         max_wavenumber::Union{Int,Nothing}=nothing,
                         radial_average::Bool=true,
                         binning::SpectrumBinning=LinearBinning())
    # Validate Fourier bases
    fourier_axes, fourier_bases = validate_fourier_bases(velocity)

    if isempty(fourier_axes)
        throw(ArgumentError("Energy spectrum requires at least one Fourier basis"))
    end

    # Ensure all velocity components are in spectral space
    for component in velocity.components
        ensure_layout!(component, :c)  # Coefficient space
    end

    # Get wavenumber grid information
    wavenumber_info = get_wavenumber_info(velocity, fourier_axes, fourier_bases)

    # Determine maximum wavenumber if not specified
    if max_wavenumber === nothing
        max_wavenumber = wavenumber_info.kmax
    else
        max_wavenumber = min(max_wavenumber, wavenumber_info.kmax)
    end

    # Calculate energy spectrum
    if radial_average
        return calculate_radial_energy_spectrum(velocity, wavenumber_info, max_wavenumber, binning)
    else
        return calculate_full_energy_spectrum(velocity, wavenumber_info, max_wavenumber)
    end
end

"""Validate and extract Fourier basis information"""
function validate_fourier_bases(velocity::VectorField)
    fourier_axes = Int[]
    fourier_bases = []
    
    for (i, basis) in enumerate(velocity.bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            push!(fourier_axes, i)
            push!(fourier_bases, basis)
        end
    end
    
    return fourier_axes, fourier_bases
end

"""
    Extract wavenumber grid information from velocity field and bases.
    Handles both 2D and 3D cases with proper PencilArrays integration.

    IMPORTANT: For MPI mode with PencilArrays, this function computes GLOBAL
    wavenumber values for each local grid point using the PencilArray offsets.
    """
function get_wavenumber_info(velocity::VectorField, fourier_axes::Vector{Int}, fourier_bases::Vector)

    # Get domain information
    domain = velocity.domain
    domain_size = get_domain_size(domain)

    # Get local spectral shape
    fourier_shape = get_fourier_shape(velocity, fourier_axes)

    # Get global index offsets for MPI mode
    offsets = _get_pencil_array_offsets(velocity)

    # Get global coefficient shape for proper wavenumber mapping
    global_coeff_shape = _get_global_coeff_shape(velocity, fourier_bases)

    # Calculate wavenumber grids with global indexing
    kx_grid, ky_grid, kz_grid = calculate_wavenumber_grids_global(
        fourier_bases, fourier_shape, domain_size, offsets, global_coeff_shape
    )

    # Calculate wavenumber magnitudes
    k_magnitudes = calculate_k_magnitudes(kx_grid, ky_grid, kz_grid)

    # Determine maximum wavenumber (Nyquist limit) - use global shape
    # CRITICAL: Pass fourier_bases to correctly handle RFFT sizing
    kmax = calculate_kmax_global(global_coeff_shape, fourier_bases)

    return WavenumberInfo(kmax, k_magnitudes, kx_grid, ky_grid, kz_grid, domain_size, fourier_shape)
end

"""
    _get_global_coeff_shape(velocity::VectorField, fourier_bases::Vector)

Get the global coefficient shape for the velocity field.
For serial execution, this is the same as the local shape.
For MPI execution with PencilArrays, this is retrieved from the pencil metadata.
"""
function _get_global_coeff_shape(velocity::VectorField, fourier_bases::Vector)
    first_component = velocity.components[1]
    coeff_data = get_coeff_data(first_component)

    # CRITICAL: Check for null coeff_data - can happen if field hasn't been transformed yet
    if coeff_data === nothing
        error("_get_global_coeff_shape: coefficient data is nothing. " *
              "This typically means the field hasn't been transformed to spectral space yet. " *
              "Call forward_transform! on the field before computing spectral diagnostics.")
    end

    # Try to get global shape from PencilArrays
    if velocity.dist.use_pencil_arrays && velocity.dist.size > 1
        try
            if applicable(PencilArrays.size_global, coeff_data)
                return Tuple(PencilArrays.size_global(coeff_data))
            end
        catch
            # Fallback below
        end
    end

    # For serial or non-PencilArrays, local shape is global shape
    return size(coeff_data)
end

"""
    calculate_kmax_global(global_shape::Tuple, fourier_bases=nothing)

Calculate maximum wavenumber (Nyquist limit) from global coefficient shape.

IMPORTANT: For RFFT (first RealFourier axis), the coefficient shape is N/2+1,
so kmax = shape - 1. For FFT axes, kmax = shape ÷ 2.

# Arguments
- `global_shape`: Global coefficient shape (may have RFFT size on first axis)
- `fourier_bases`: Optional vector of Fourier basis objects to determine RFFT usage

If `fourier_bases` is provided, the function correctly handles RFFT sizing.
Otherwise, it uses the legacy formula (which may be incorrect for RFFT).
"""
function calculate_kmax_global(global_shape::Tuple, fourier_bases=nothing)
    if isempty(global_shape)
        return 0
    end

    # Determine kmax for each dimension
    kmaxes = Int[]

    for i in 1:min(length(global_shape), 3)  # Consider up to 3 dimensions
        shape_i = global_shape[i]

        # Check if this dimension uses RFFT (first RealFourier axis)
        uses_rfft = false
        if fourier_bases !== nothing && i <= length(fourier_bases)
            basis = fourier_bases[i]
            if isa(basis, RealFourier)
                # Check if this is the FIRST RealFourier axis (only first uses RFFT)
                first_rf_idx = findfirst(b -> isa(b, RealFourier), fourier_bases)
                uses_rfft = (first_rf_idx == i)
            end
        end

        if uses_rfft
            # RFFT: coefficient shape is N/2+1, so kmax = shape - 1
            push!(kmaxes, shape_i - 1)
        else
            # FFT: coefficient shape is N, so kmax = N/2
            push!(kmaxes, shape_i ÷ 2)
        end
    end

    if isempty(kmaxes)
        return 0
    elseif length(kmaxes) == 1
        return kmaxes[1]
    else
        return minimum(kmaxes)
    end
end

"""
    Calculate wavenumber grids for each Fourier dimension (serial version).
    DEPRECATED: Use calculate_wavenumber_grids_global for MPI compatibility.
    """
function calculate_wavenumber_grids(fourier_bases::Vector, fourier_shape::Tuple{Vararg{Int}}, domain_size::Tuple{Vararg{Float64}})
    # Call global version with zero offsets for backward compatibility
    offsets = Tuple(zeros(Int, length(fourier_shape)))
    return calculate_wavenumber_grids_global(fourier_bases, fourier_shape, domain_size, offsets, fourier_shape)
end

"""
    calculate_wavenumber_grids_global(fourier_bases, fourier_shape, domain_size, offsets, global_shape)

Calculate wavenumber grids for each Fourier dimension using GLOBAL indexing.

For MPI execution with PencilArrays, this function computes the correct global
wavenumber values for each local grid point by accounting for the local rank's
offset in the global domain.

Arguments:
- fourier_bases: Vector of Fourier basis objects
- fourier_shape: Local shape of the Fourier coefficient array (LOCAL size per dimension)
- domain_size: Physical domain size (L) for each dimension
- offsets: Global index offset for each dimension (from PencilArrays.axes_local)
- global_shape: Global shape of the coefficient array

For RFFT (first RealFourier axis): wavenumbers are [0, 1, 2, ..., N/2], no negative frequencies.
For FFT axes: wavenumbers follow standard FFT ordering [0, 1, ..., N/2-1, -N/2, ..., -1].

In MPI mode, each rank computes wavenumbers for its local portion of the global domain.
"""
function calculate_wavenumber_grids_global(fourier_bases::Vector, fourier_shape::Tuple{Vararg{Int}},
                                            domain_size::Tuple{Vararg{Float64}}, offsets::Tuple,
                                            global_shape::Tuple)
    kx_grid = nothing
    ky_grid = nothing
    kz_grid = nothing

    # X-direction (first Fourier basis)
    if length(fourier_bases) >= 1
        basis_x = fourier_bases[1]
        nx_local = fourier_shape[1]  # LOCAL size
        nx_global = length(global_shape) >= 1 ? global_shape[1] : nx_local
        offset_x = length(offsets) >= 1 ? offsets[1] : 0
        Lx = domain_size[1]
        N_x = basis_x.meta.size  # GLOBAL grid size
        k0_x = 2π / Lx

        kx_1d = zeros(Float64, nx_local)

        if isa(basis_x, RealFourier)
            # Check if this is RFFT layout
            rfft_size = N_x ÷ 2 + 1
            if nx_global == rfft_size
                # RFFT layout: global wavenumbers are [0, 1, 2, ..., N/2]
                # Each local index i maps to global index (i-1 + offset_x), giving wavenumber k
                for i in 1:nx_local
                    global_idx = (i - 1) + offset_x  # 0-based global index
                    kx_1d[i] = k0_x * global_idx
                end
            else
                # FFT fallback (full size N): compute global wavenumber from global index
                for i in 1:nx_local
                    global_idx = (i - 1) + offset_x  # 0-based global index in array
                    k = _fft_index_to_wavenumber(global_idx, nx_global)
                    kx_1d[i] = k0_x * k
                end
            end
        else  # ComplexFourier
            # FFT ordering with global indexing
            for i in 1:nx_local
                global_idx = (i - 1) + offset_x
                k = _fft_index_to_wavenumber(global_idx, nx_global)
                kx_1d[i] = k0_x * k
            end
        end
        kx_grid = reshape(kx_1d, length(kx_1d), 1)
    end

    # Y-direction (second Fourier basis)
    if length(fourier_bases) >= 2
        basis_y = fourier_bases[2]
        ny_local = fourier_shape[2]
        ny_global = length(global_shape) >= 2 ? global_shape[2] : ny_local
        offset_y = length(offsets) >= 2 ? offsets[2] : 0
        Ly = domain_size[2]
        k0_y = 2π / Ly

        ky_1d = zeros(Float64, ny_local)

        # Y-direction uses FFT ordering in PencilFFTs (even for RealFourier)
        for j in 1:ny_local
            global_idx = (j - 1) + offset_y
            k = _fft_index_to_wavenumber(global_idx, ny_global)
            ky_1d[j] = k0_y * k
        end
        ky_grid = reshape(ky_1d, 1, length(ky_1d))
    end

    # Z-direction (third Fourier basis, if present)
    if length(fourier_bases) >= 3
        basis_z = fourier_bases[3]
        nz_local = fourier_shape[3]
        nz_global = length(global_shape) >= 3 ? global_shape[3] : nz_local
        offset_z = length(offsets) >= 3 ? offsets[3] : 0
        Lz = domain_size[3]
        k0_z = 2π / Lz

        kz_1d = zeros(Float64, nz_local)

        # Z-direction uses FFT ordering
        for k in 1:nz_local
            global_idx = (k - 1) + offset_z
            wavenum = _fft_index_to_wavenumber(global_idx, nz_global)
            kz_1d[k] = k0_z * wavenum
        end
        kz_grid = reshape(kz_1d, 1, 1, length(kz_1d))
    end

    return kx_grid, ky_grid, kz_grid
end

"""
    _fft_index_to_wavenumber(idx, N)

Convert 0-based FFT array index to wavenumber.

Standard FFT ordering for an N-point transform:
- Indices 0 to N/2-1 (or (N-1)/2 for odd N): k = idx (non-negative frequencies)
- Indices N/2 to N-1: k = idx - N (negative frequencies)

This is the standard numpy/FFTW convention where:
- output[0] is DC (k=0)
- output[1:N/2] are positive frequencies k=1,2,...,N/2-1
- output[N/2] is Nyquist (k=N/2 or k=-N/2, depending on convention)
- output[N/2+1:N-1] are negative frequencies k=-N/2+1,...,-1
"""
function _fft_index_to_wavenumber(idx::Int, N::Int)
    half_N = N ÷ 2
    if idx <= half_N
        return idx
    else
        return idx - N
    end
end

"""Calculate wavenumber magnitudes |k| = √(kx² + ky² + kz²)"""
function calculate_k_magnitudes(kx_grid::Array{Float64}, ky_grid::Array{Float64}, kz_grid::Union{Array{Float64}, Nothing})
    
    if kz_grid !== nothing
        # 3D case
        k_magnitudes = sqrt.(kx_grid.^2 .+ ky_grid.^2 .+ kz_grid.^2)
    else
        # 2D case
        k_magnitudes = sqrt.(kx_grid.^2 .+ ky_grid.^2)
    end
    
    return k_magnitudes
end

"""Calculate maximum wavenumber (Nyquist limit)"""
function calculate_kmax(fourier_shape::Tuple{Vararg{Int}})
    return min(fourier_shape[1]÷2, fourier_shape[2]÷2)
end

"""
    calculate_radial_energy_spectrum(velocity, wavenumber_info, max_wavenumber, binning)

Calculate radially-averaged energy spectrum E(k) with configurable binning.
Bins energy by wavenumber magnitude and performs proper averaging.

Returns a NamedTuple with:
- `k`: Vector of bin center wavenumbers
- `power`: Vector of energy spectrum values E(k)
- `bin_counts`: Vector of mode counts per bin
- `bin_edges`: Vector of bin edges used
"""
function calculate_radial_energy_spectrum(velocity::VectorField, wavenumber_info::WavenumberInfo,
                                          max_wavenumber::Int, binning::SpectrumBinning=LinearBinning())
    # Calculate kinetic energy density in spectral space
    ke_spectral = calculate_spectral_kinetic_energy(velocity)
    k_magnitudes = wavenumber_info.k_magnitudes
    is_3d = wavenumber_info.kz_grid !== nothing

    # Build bin edges based on binning mode
    bin_edges = _build_bin_edges(binning, max_wavenumber)
    num_bins = length(bin_edges) - 1

    # Initialize spectrum arrays
    spectrum = zeros(Float64, num_bins)
    bin_counts = zeros(Int, num_bins)

    # Bin by wavenumber magnitude
    for idx in CartesianIndices(ke_spectral)
        k_mag = k_magnitudes[idx]
        bin_idx = _find_bin(k_mag, bin_edges)

        if bin_idx > 0 && bin_idx <= num_bins
            spectrum[bin_idx] += ke_spectral[idx]
            bin_counts[bin_idx] += 1
        end
    end

    # Perform MPI reduction across processes using field's communicator
    if MPI.Initialized() && velocity.dist.size > 1
        comm = velocity.components[1].dist.comm
        spectrum = MPI.Allreduce(spectrum, MPI.SUM, comm)
        bin_counts = MPI.Allreduce(bin_counts, MPI.SUM, comm)
    end

    # Calculate bin centers
    bin_centers = _calculate_bin_centers(bin_edges, binning.mode)

    # Normalize by bin counts and apply proper scaling
    for i in 1:num_bins
        if bin_counts[i] > 0
            spectrum[i] /= bin_counts[i]
            k_center = bin_centers[i]
            if k_center > 0
                if is_3d
                    spectrum[i] *= 4π * k_center^2  # Spherical shell
                else
                    spectrum[i] *= 2π * k_center    # Circular shell
                end
            end
        end
    end

    return (k=bin_centers, power=spectrum, bin_counts=bin_counts, bin_edges=bin_edges)
end

"""
    Calculate kinetic energy density in spectral space.
    Returns |û|² + |v̂|² + |ŵ|² with proper normalization.

    IMPORTANT: For RFFT (RealFourier on first axis), the spectral coefficients only
    represent non-negative frequencies. Due to conjugate symmetry, modes with k>0 and k<N/2
    should be counted twice (they represent both +k and -k). This function applies
    the factor-of-2 correction when apply_conjugate_symmetry=true (default).

    - k=0 (DC mode): count once
    - k=1 to k=N/2-1 (interior modes): count twice
    - k=N/2 (Nyquist, only if N even): count once
    """
function calculate_spectral_kinetic_energy(velocity::VectorField; apply_conjugate_symmetry::Bool=true)

    # Get first component to determine array size (GPU-compatible allocation)
    first_component = velocity.components[1]
    ensure_layout!(first_component, :c)
    ke_spectral = similar(get_coeff_data(first_component), Float64)
    fill!(ke_spectral, zero(Float64))

    # Sum |û_i|² over all velocity components
    for component in velocity.components
        ensure_layout!(component, :c)
        ke_spectral .+= abs2.(get_coeff_data(component))
    end

    # Apply conjugate symmetry correction for RFFT
    # Check if first basis is RealFourier and if we're in RFFT layout
    if apply_conjugate_symmetry && length(velocity.bases) > 0
        first_basis = velocity.bases[1]
        if isa(first_basis, RealFourier)
            N = first_basis.meta.size
            coeff_shape = size(ke_spectral)
            rfft_size = N ÷ 2 + 1

            # Check if we're in RFFT layout (global first axis has size N/2+1)
            # Note: For distributed arrays, coeff_shape[1] is local size, not global!
            # Use offsets to determine which global k-indices this rank owns

            # Get global offsets for this rank's portion
            offsets = _get_pencil_array_offsets(velocity)
            offset_k = offsets[1]  # Offset in first (k) dimension

            # Determine local index range that maps to global k indices
            local_k_size = coeff_shape[1]
            global_k_start = offset_k + 1  # 1-based global index
            global_k_end = offset_k + local_k_size

            # CRITICAL: Only apply factor of 2 to interior modes
            # - Global k=0 (DC mode): count once
            # - Global k=N/2 (Nyquist, if N even): count once
            # - All other k: count twice (represent both +k and -k due to conjugate symmetry)
            #
            # IMPORTANT: For odd N, there is NO Nyquist frequency. The highest k = floor(N/2)
            # is a regular interior mode and SHOULD be doubled.

            # Find local indices that correspond to interior global k (should be doubled)
            # Local index i corresponds to global k = offset_k + i
            # Mapping: local index i → wavenumber k = offset_k + i - 1 (0-based k, 1-based i)

            # Determine the maximum interior k that should be doubled
            # For even N: exclude Nyquist at k=N/2, so max_interior_k = N/2 - 1
            # For odd N: no Nyquist exists, so max_interior_k = N÷2 (include highest k)
            max_interior_k = iseven(N) ? (N ÷ 2 - 1) : (N ÷ 2)

            # Build the local index range to double
            # We want k >= 1:  offset_k + i - 1 >= 1  →  i >= 2 - offset_k
            # We want k <= max_interior_k:  offset_k + i - 1 <= max_interior_k  →  i <= max_interior_k - offset_k + 1
            double_start = max(1, 2 - offset_k)  # First local index where k >= 1
            double_end = min(local_k_size, max_interior_k - offset_k + 1)  # Last local index where k <= max_interior_k

            # Only apply doubling if there are valid indices to double
            if double_start <= double_end
                double_range = double_start:double_end
                if ndims(ke_spectral) == 2
                    ke_spectral[double_range, :] .*= 2.0
                elseif ndims(ke_spectral) == 3
                    ke_spectral[double_range, :, :] .*= 2.0
                elseif ndims(ke_spectral) == 1
                    ke_spectral[double_range] .*= 2.0
                end
            end
        end
    end

    # Apply normalization factor (accounts for FFT scaling)
    # Factor of 0.5 for kinetic energy definition: KE = (1/2)|u|²
    ke_spectral .*= 0.5

    return ke_spectral
end

"""
    Calculate full energy spectrum without radial averaging.
    Returns E(kx, ky) or E(kx, ky, kz) for detailed analysis.

    IMPORTANT: In MPI mode with PencilArrays, the returned dictionary uses GLOBAL
    wavenumber indices as keys (accounting for the local rank's offset in the
    distributed decomposition). Each rank returns its portion of the spectrum.

    To combine across ranks, use MPI gather operations (not reduce, since each rank
    has different wavenumbers). The keys are globally unique across ranks.
    """
function calculate_full_energy_spectrum(velocity::VectorField, wavenumber_info::WavenumberInfo, max_wavenumber::Int)

    ke_spectral = calculate_spectral_kinetic_energy(velocity)

    # Get global index offsets for MPI with PencilArrays
    # For serial or non-PencilArrays, offsets are (0, 0, ...)
    offsets = _get_pencil_array_offsets(velocity)

    # Return as dictionary with GLOBAL wavenumber indices as keys
    full_spectrum = Dict{Tuple{Vararg{Int}}, Float64}()

    for idx in CartesianIndices(ke_spectral)
        # Convert local indices to global indices
        if length(idx.I) >= 3
            global_kx = idx.I[1] + offsets[1]
            global_ky = idx.I[2] + offsets[2]
            global_kz = idx.I[3] + offsets[3]
            full_spectrum[(global_kx, global_ky, global_kz)] = ke_spectral[idx]
        elseif length(idx.I) >= 2
            global_kx = idx.I[1] + offsets[1]
            global_ky = idx.I[2] + offsets[2]
            full_spectrum[(global_kx, global_ky)] = ke_spectral[idx]
        else
            global_kx = idx.I[1] + offsets[1]
            full_spectrum[(global_kx,)] = ke_spectral[idx]
        end
    end

    return full_spectrum
end

# ============================================================================
# Power Spectrum Functions for Scalar and Vector Fields
# ============================================================================

"""
    power_spectrum(field::ScalarField; max_wavenumber=nothing, radial_average=true, binning=LinearBinning())

Calculate power spectrum P(k) of a scalar field.

Computes the power spectrum P(k) = |f̂(k)|² where the spectral coefficients are
binned by wavenumber magnitude (if radial_average=true) or returned as a full
multi-dimensional spectrum.

Properly handles:
- 2D and 3D fields
- MPI distributed data (global reduction across ranks)
- RFFT conjugate symmetry (interior modes counted twice)
- Flexible binning options for spectrum smoothing

# Arguments
- `field::ScalarField`: Scalar field with Fourier bases
- `max_wavenumber::Union{Int,Nothing}=nothing`: Maximum wavenumber (default: Nyquist limit)
- `radial_average::Bool=true`: Whether to perform radial (shell) averaging
- `binning::SpectrumBinning=LinearBinning()`: Binning configuration for smoothing

# Returns
- If radial_average=true: NamedTuple (k=wavenumbers, power=spectrum values, bin_counts=modes per bin)
- If radial_average=false: Dict{Tuple, Float64} with (kx, ky, ...) → P(k)

# Examples
```julia
# Standard integer-binned spectrum
result = power_spectrum(temperature_field)
plot(result.k, result.power, xscale=:log10, yscale=:log10)

# Smoothed with linear bins of width 2
result = power_spectrum(field, binning=LinearBinning(bin_width=2.0))

# Logarithmic bins for turbulence spectra
result = power_spectrum(field, binning=LogBinning(bins_per_decade=8))

# Custom bin edges
result = power_spectrum(field, binning=CustomBinning([1,2,4,8,16,32,64]))

# Full 2D spectrum (no binning)
P_full = power_spectrum(scalar_field, radial_average=false)
```
"""
function power_spectrum(field::ScalarField;
                        max_wavenumber::Union{Int,Nothing}=nothing,
                        radial_average::Bool=true,
                        binning::SpectrumBinning=LinearBinning())
    # Validate Fourier bases
    fourier_axes, fourier_bases = validate_fourier_bases_scalar(field)

    if isempty(fourier_axes)
        throw(ArgumentError("Power spectrum requires at least one Fourier basis"))
    end

    # Ensure field is in spectral space
    ensure_layout!(field, :c)

    # Get wavenumber grid information
    wavenumber_info = get_wavenumber_info_scalar(field, fourier_axes, fourier_bases)

    # Determine maximum wavenumber if not specified
    if max_wavenumber === nothing
        max_wavenumber = wavenumber_info.kmax
    else
        max_wavenumber = min(max_wavenumber, wavenumber_info.kmax)
    end

    # Calculate power spectrum
    if radial_average
        return calculate_radial_power_spectrum(field, wavenumber_info, max_wavenumber, binning)
    else
        return calculate_full_power_spectrum(field, wavenumber_info, max_wavenumber)
    end
end

"""
    enstrophy_spectrum(velocity::VectorField; max_wavenumber=nothing, radial_average=true, binning=LinearBinning())

Calculate enstrophy spectrum Z(k) = |ω̂(k)|² from velocity field.

For 2D flows, computes the spectrum of the scalar vorticity ω = ∂v/∂x - ∂u/∂y.
For 3D flows, computes the spectrum of vorticity magnitude |ω|² = |ωx|² + |ωy|² + |ωz|².

Properly handles:
- 2D and 3D velocity fields
- MPI distributed data (global reduction across ranks)
- RFFT conjugate symmetry
- Flexible binning options for spectrum smoothing

# Arguments
- `velocity::VectorField`: Velocity field with Fourier bases
- `max_wavenumber::Union{Int,Nothing}=nothing`: Maximum wavenumber (default: Nyquist limit)
- `radial_average::Bool=true`: Whether to perform radial (shell) averaging
- `binning::SpectrumBinning=LinearBinning()`: Binning configuration for smoothing

# Returns
- If radial_average=true: NamedTuple (k=wavenumbers, power=spectrum values, bin_counts=modes per bin)
- If radial_average=false: Dict{Tuple, Float64} with (kx, ky, ...) → Z(k)

# Examples
```julia
# Standard enstrophy spectrum
result = enstrophy_spectrum(velocity)
plot(result.k, result.power, xscale=:log10, yscale=:log10)

# Logarithmic bins for smoother spectrum
result = enstrophy_spectrum(velocity, binning=LogBinning(bins_per_decade=8))
```
"""
function enstrophy_spectrum(velocity::VectorField;
                            max_wavenumber::Union{Int,Nothing}=nothing,
                            radial_average::Bool=true,
                            binning::SpectrumBinning=LinearBinning())
    dim = velocity.coordsys.dim

    if dim == 2
        # 2D: vorticity is a scalar, ω = ∂v/∂x - ∂u/∂y
        vorticity_field = evaluate_operator(curl(velocity))
        return power_spectrum(vorticity_field; max_wavenumber=max_wavenumber,
                             radial_average=radial_average, binning=binning)

    elseif dim == 3
        # 3D: vorticity is a vector, compute |ω|² = |ωx|² + |ωy|² + |ωz|²
        vorticity = evaluate_operator(curl(velocity))

        # Validate Fourier bases
        fourier_axes, fourier_bases = validate_fourier_bases(velocity)
        if isempty(fourier_axes)
            throw(ArgumentError("Enstrophy spectrum requires at least one Fourier basis"))
        end

        # Ensure vorticity components are in spectral space
        for component in vorticity.components
            ensure_layout!(component, :c)
        end

        # Get wavenumber info
        wavenumber_info = get_wavenumber_info(velocity, fourier_axes, fourier_bases)

        if max_wavenumber === nothing
            max_wavenumber = wavenumber_info.kmax
        else
            max_wavenumber = min(max_wavenumber, wavenumber_info.kmax)
        end

        if radial_average
            return calculate_radial_vector_spectrum(vorticity, wavenumber_info, max_wavenumber, binning)
        else
            return calculate_full_vector_spectrum(vorticity, wavenumber_info, max_wavenumber)
        end
    else
        throw(ArgumentError("Enstrophy spectrum requires 2D or 3D velocity field, got $(dim)D"))
    end
end

"""
    scalar_spectrum(field::ScalarField; max_wavenumber=nothing, radial_average=true, quantity_name="scalar")

Alias for power_spectrum with optional quantity name for logging.
"""
scalar_spectrum(field::ScalarField; kwargs...) = power_spectrum(field; kwargs...)

# ============================================================================
# Helper Functions for Scalar Field Spectra
# ============================================================================

"""
    validate_fourier_bases_scalar(field::ScalarField)

Validate and extract Fourier basis information for a scalar field.
"""
function validate_fourier_bases_scalar(field::ScalarField)
    fourier_axes = Int[]
    fourier_bases = []

    for (i, basis) in enumerate(field.bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            push!(fourier_axes, i)
            push!(fourier_bases, basis)
        end
    end

    return fourier_axes, fourier_bases
end

"""
    get_wavenumber_info_scalar(field::ScalarField, fourier_axes, fourier_bases)

Get wavenumber grid information for a scalar field.
"""
function get_wavenumber_info_scalar(field::ScalarField, fourier_axes::Vector{Int}, fourier_bases::Vector)
    # Get spectral data shape
    coeff_data = get_coeff_data(field)
    if coeff_data === nothing
        error("get_wavenumber_info_scalar: field has no coefficient data. Call forward_transform! first.")
    end

    local_shape = size(coeff_data)
    ndims_field = length(local_shape)

    # Get MPI offsets and global shape
    use_mpi = field.dist.use_pencil_arrays && field.dist.size > 1
    if use_mpi
        global_shape = _get_global_coeff_shape_internal(field)
        offsets = _get_pencil_array_offsets_internal(field)
    else
        global_shape = local_shape
        offsets = Tuple(zeros(Int, ndims_field))
    end

    # Get domain size
    domain_size = _get_domain_size_from_bases(fourier_bases)

    # Build wavenumber grids for each dimension
    k_grids = []
    for (dim_idx, (axis, basis)) in enumerate(zip(fourier_axes, fourier_bases))
        if dim_idx <= length(local_shape)
            local_n = local_shape[dim_idx]
            offset = offsets[dim_idx]
            L = domain_size[dim_idx]

            if isa(basis, RealFourier)
                # RFFT: k = 0, 1, 2, ..., N/2
                k_indices = offset:(offset + local_n - 1)
                k_1d = collect(k_indices)
            else
                # FFT: k = -N/2, ..., -1, 0, 1, ..., N/2-1
                global_n = global_shape[dim_idx]
                global_k = collect(fftshift(-global_n÷2:(global_n÷2-1)))
                k_1d = global_k[(offset+1):(offset+local_n)]
            end
            push!(k_grids, k_1d)
        end
    end

    # Build multi-dimensional k-magnitude grid
    if ndims_field == 2 && length(k_grids) >= 2
        kx, ky = k_grids[1], k_grids[2]
        k_magnitudes = sqrt.(reshape(kx, :, 1).^2 .+ reshape(ky, 1, :).^2)
        kx_grid = reshape(Float64.(kx), :, 1) .* ones(1, length(ky))
        ky_grid = ones(length(kx), 1) .* reshape(Float64.(ky), 1, :)
        kz_grid = nothing
        fourier_shape = (length(kx), length(ky))
    elseif ndims_field >= 3 && length(k_grids) >= 3
        kx, ky, kz = k_grids[1], k_grids[2], k_grids[3]
        k_magnitudes = sqrt.(reshape(kx, :, 1, 1).^2 .+ reshape(ky, 1, :, 1).^2 .+ reshape(kz, 1, 1, :).^2)
        kx_grid = reshape(Float64.(kx), :, 1, 1) .* ones(1, length(ky), length(kz))
        ky_grid = ones(length(kx), 1, 1) .* reshape(Float64.(ky), 1, :, 1) .* ones(1, 1, length(kz))
        kz_grid = ones(length(kx), 1, 1) .* ones(1, length(ky), 1) .* reshape(Float64.(kz), 1, 1, :)
        fourier_shape = (length(kx), length(ky), length(kz))
    elseif ndims_field == 1 && length(k_grids) >= 1
        kx = k_grids[1]
        k_magnitudes = abs.(Float64.(kx))
        kx_grid = Float64.(kx)
        ky_grid = zeros(0)
        kz_grid = nothing
        fourier_shape = (length(kx),)
    else
        error("Unsupported field dimensions: $ndims_field with $(length(k_grids)) Fourier bases")
    end

    # Calculate kmax
    kmax = calculate_kmax_from_bases(fourier_bases, global_shape)

    return WavenumberInfo(kmax, k_magnitudes, kx_grid, ky_grid, kz_grid, domain_size, fourier_shape)
end

"""
    _get_domain_size_from_bases(fourier_bases)

Extract domain size from Fourier basis metadata.
"""
function _get_domain_size_from_bases(fourier_bases)
    sizes = Float64[]
    for basis in fourier_bases
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :bounds)
            bounds = basis.meta.bounds
            if bounds !== nothing && length(bounds) >= 2
                push!(sizes, Float64(bounds[2] - bounds[1]))
            else
                push!(sizes, 2π)  # Default
            end
        else
            push!(sizes, 2π)  # Default
        end
    end
    return Tuple(sizes)
end

"""
    calculate_spectral_power(field::ScalarField; apply_conjugate_symmetry=true)

Calculate power density |f̂|² in spectral space with proper RFFT handling.
"""
function calculate_spectral_power(field::ScalarField; apply_conjugate_symmetry::Bool=true)
    ensure_layout!(field, :c)
    coeff_data = get_coeff_data(field)

    # Compute |f̂|²
    power_spectral = similar(coeff_data, Float64)
    power_spectral .= abs2.(coeff_data)

    # Apply conjugate symmetry correction for RFFT
    if apply_conjugate_symmetry && length(field.bases) > 0
        first_basis = field.bases[1]
        if isa(first_basis, RealFourier)
            N = first_basis.meta.size
            coeff_shape = size(power_spectral)

            # Get offsets for MPI
            if field.dist.use_pencil_arrays && field.dist.size > 1
                offsets = _get_pencil_array_offsets_internal(field)
            else
                offsets = Tuple(zeros(Int, ndims(coeff_data)))
            end
            offset_k = offsets[1]
            local_k_size = coeff_shape[1]

            # Determine max interior k (exclude Nyquist for even N)
            max_interior_k = iseven(N) ? (N ÷ 2 - 1) : (N ÷ 2)

            # Build local index range to double
            double_start = max(1, 2 - offset_k)
            double_end = min(local_k_size, max_interior_k - offset_k + 1)

            if double_start <= double_end
                double_range = double_start:double_end
                if ndims(power_spectral) == 2
                    power_spectral[double_range, :] .*= 2.0
                elseif ndims(power_spectral) == 3
                    power_spectral[double_range, :, :] .*= 2.0
                elseif ndims(power_spectral) == 1
                    power_spectral[double_range] .*= 2.0
                end
            end
        end
    end

    return power_spectral
end

"""
    calculate_radial_power_spectrum(field, wavenumber_info, max_wavenumber, binning)

Calculate radially-averaged power spectrum P(k) for a scalar field with configurable binning.

Returns a NamedTuple with:
- `k`: Vector of bin center wavenumbers
- `power`: Vector of power spectrum values P(k)
- `bin_counts`: Vector of mode counts per bin
- `bin_edges`: Vector of bin edges used
"""
function calculate_radial_power_spectrum(field::ScalarField, wavenumber_info::WavenumberInfo,
                                         max_wavenumber::Int, binning::SpectrumBinning=LinearBinning())
    # Calculate spectral power density
    power_spectral = calculate_spectral_power(field)
    k_magnitudes = wavenumber_info.k_magnitudes
    is_3d = wavenumber_info.kz_grid !== nothing

    # Build bin edges based on binning mode
    bin_edges = _build_bin_edges(binning, max_wavenumber)
    num_bins = length(bin_edges) - 1

    # Initialize spectrum arrays
    spectrum = zeros(Float64, num_bins)
    bin_counts = zeros(Int, num_bins)

    # Bin by wavenumber magnitude
    for idx in CartesianIndices(power_spectral)
        k_mag = k_magnitudes[idx]

        # Find which bin this k belongs to
        bin_idx = _find_bin(k_mag, bin_edges)

        if bin_idx > 0 && bin_idx <= num_bins
            spectrum[bin_idx] += power_spectral[idx]
            bin_counts[bin_idx] += 1
        end
    end

    # MPI reduction
    if MPI.Initialized() && field.dist.size > 1
        comm = field.dist.comm
        spectrum = MPI.Allreduce(spectrum, MPI.SUM, comm)
        bin_counts = MPI.Allreduce(bin_counts, MPI.SUM, comm)
    end

    # Calculate bin centers
    bin_centers = _calculate_bin_centers(bin_edges, binning.mode)

    # Normalize and apply shell scaling
    for i in 1:num_bins
        if bin_counts[i] > 0
            spectrum[i] /= bin_counts[i]
            k_center = bin_centers[i]
            if k_center > 0
                if is_3d
                    spectrum[i] *= 4π * k_center^2  # Spherical shell
                else
                    spectrum[i] *= 2π * k_center    # Circular shell
                end
            end
        end
    end

    return (k=bin_centers, power=spectrum, bin_counts=bin_counts, bin_edges=bin_edges)
end

"""
    _build_bin_edges(binning, max_wavenumber)

Build bin edges based on binning configuration.
"""
function _build_bin_edges(binning::SpectrumBinning, max_wavenumber::Int)
    if binning.mode == :linear
        # Linear bins with specified width
        bin_width = binning.bin_width
        edges = collect(0.0:bin_width:(max_wavenumber + bin_width))
        return edges

    elseif binning.mode == :log
        # Logarithmic bins
        if binning.num_bins < 0
            # Negative means bins_per_decade was specified
            bins_per_decade = -binning.num_bins
            num_decades = log10(max_wavenumber)
            num_bins = max(1, round(Int, bins_per_decade * num_decades))
        else
            num_bins = binning.num_bins
        end

        # Log-spaced edges from 1 to max_wavenumber (k=0 handled separately)
        if max_wavenumber > 1
            log_edges = 10.0 .^ range(0, log10(max_wavenumber), length=num_bins+1)
            # Prepend 0 for the DC mode bin
            edges = vcat([0.0, 0.5], log_edges[2:end])
        else
            edges = [0.0, 0.5, 1.0]
        end
        return edges

    elseif binning.mode == :custom
        return binning.bin_edges

    else
        error("Unknown binning mode: $(binning.mode)")
    end
end

"""
    _find_bin(k, bin_edges)

Find which bin index a wavenumber k belongs to.
Returns 0 if k is outside all bins.
"""
function _find_bin(k::Real, bin_edges::Vector{Float64})
    n = length(bin_edges)
    if n < 2
        return 0
    end

    # Binary search for bin
    if k < bin_edges[1] || k >= bin_edges[end]
        return 0
    end

    # Linear search (can be optimized to binary search for large num_bins)
    for i in 1:(n-1)
        if k >= bin_edges[i] && k < bin_edges[i+1]
            return i
        end
    end

    return 0
end

"""
    _calculate_bin_centers(bin_edges, mode)

Calculate bin center wavenumbers.
Uses geometric mean for log bins, arithmetic mean for linear/custom.
"""
function _calculate_bin_centers(bin_edges::Vector{Float64}, mode::Symbol)
    n = length(bin_edges) - 1
    centers = zeros(Float64, n)

    for i in 1:n
        if mode == :log && bin_edges[i] > 0 && bin_edges[i+1] > 0
            # Geometric mean for log bins
            centers[i] = sqrt(bin_edges[i] * bin_edges[i+1])
        else
            # Arithmetic mean for linear/custom bins
            centers[i] = (bin_edges[i] + bin_edges[i+1]) / 2
        end
    end

    return centers
end

"""
    calculate_full_power_spectrum(field::ScalarField, wavenumber_info, max_wavenumber)

Calculate full (non-averaged) power spectrum for a scalar field.
"""
function calculate_full_power_spectrum(field::ScalarField, wavenumber_info::WavenumberInfo, max_wavenumber::Int)
    power_spectral = calculate_spectral_power(field)

    # Get MPI offsets
    if field.dist.use_pencil_arrays && field.dist.size > 1
        offsets = _get_pencil_array_offsets_internal(field)
    else
        offsets = Tuple(zeros(Int, ndims(power_spectral)))
    end

    full_spectrum = Dict{Tuple{Vararg{Int}}, Float64}()
    ndims_field = ndims(power_spectral)

    for idx in CartesianIndices(power_spectral)
        if ndims_field == 3
            global_kx = round(Int, wavenumber_info.kx_grid[idx])
            global_ky = round(Int, wavenumber_info.ky_grid[idx])
            global_kz = round(Int, wavenumber_info.kz_grid[idx])
            full_spectrum[(global_kx, global_ky, global_kz)] = power_spectral[idx]
        elseif ndims_field == 2
            global_kx = round(Int, wavenumber_info.kx_grid[idx])
            global_ky = round(Int, wavenumber_info.ky_grid[idx])
            full_spectrum[(global_kx, global_ky)] = power_spectral[idx]
        else
            global_kx = round(Int, wavenumber_info.kx_grid[idx[1]])
            full_spectrum[(global_kx,)] = power_spectral[idx]
        end
    end

    return full_spectrum
end

"""
    calculate_radial_vector_spectrum(vector_field, wavenumber_info, max_wavenumber, binning)

Calculate radially-averaged spectrum of a vector field magnitude |v̂|² = Σ|v̂_i|².
Used for 3D enstrophy spectrum. Supports configurable binning for smoothing.

Returns a NamedTuple with:
- `k`: Vector of bin center wavenumbers
- `power`: Vector of spectrum values
- `bin_counts`: Vector of mode counts per bin
- `bin_edges`: Vector of bin edges used
"""
function calculate_radial_vector_spectrum(vector_field::VectorField, wavenumber_info::WavenumberInfo,
                                          max_wavenumber::Int, binning::SpectrumBinning=LinearBinning())
    # Calculate |ω|² = Σ|ω̂_i|² with conjugate symmetry correction
    vector_spectral = calculate_spectral_kinetic_energy(vector_field, apply_conjugate_symmetry=true)
    k_magnitudes = wavenumber_info.k_magnitudes
    is_3d = wavenumber_info.kz_grid !== nothing

    # Build bin edges based on binning mode
    bin_edges = _build_bin_edges(binning, max_wavenumber)
    num_bins = length(bin_edges) - 1

    # Initialize spectrum arrays
    spectrum = zeros(Float64, num_bins)
    bin_counts = zeros(Int, num_bins)

    # Bin by wavenumber magnitude
    for idx in CartesianIndices(vector_spectral)
        k_mag = k_magnitudes[idx]
        bin_idx = _find_bin(k_mag, bin_edges)

        if bin_idx > 0 && bin_idx <= num_bins
            spectrum[bin_idx] += vector_spectral[idx]
            bin_counts[bin_idx] += 1
        end
    end

    # MPI reduction
    if MPI.Initialized() && vector_field.dist.size > 1
        comm = vector_field.components[1].dist.comm
        spectrum = MPI.Allreduce(spectrum, MPI.SUM, comm)
        bin_counts = MPI.Allreduce(bin_counts, MPI.SUM, comm)
    end

    # Calculate bin centers
    bin_centers = _calculate_bin_centers(bin_edges, binning.mode)

    # Normalize and apply shell scaling
    for i in 1:num_bins
        if bin_counts[i] > 0
            spectrum[i] /= bin_counts[i]
            k_center = bin_centers[i]
            if k_center > 0
                if is_3d
                    spectrum[i] *= 4π * k_center^2
                else
                    spectrum[i] *= 2π * k_center
                end
            end
        end
    end

    return (k=bin_centers, power=spectrum, bin_counts=bin_counts, bin_edges=bin_edges)
end

"""
    calculate_full_vector_spectrum(vector_field::VectorField, wavenumber_info, max_wavenumber)

Calculate full (non-averaged) spectrum of a vector field magnitude.
"""
function calculate_full_vector_spectrum(vector_field::VectorField, wavenumber_info::WavenumberInfo, max_wavenumber::Int)
    vector_spectral = calculate_spectral_kinetic_energy(vector_field, apply_conjugate_symmetry=true)

    # Get MPI offsets
    first_component = vector_field.components[1]
    if first_component.dist.use_pencil_arrays && first_component.dist.size > 1
        offsets = _get_pencil_array_offsets(vector_field)
    else
        offsets = Tuple(zeros(Int, ndims(vector_spectral)))
    end

    full_spectrum = Dict{Tuple{Vararg{Int}}, Float64}()
    ndims_field = ndims(vector_spectral)

    for idx in CartesianIndices(vector_spectral)
        if ndims_field == 3
            global_kx = round(Int, wavenumber_info.kx_grid[idx])
            global_ky = round(Int, wavenumber_info.ky_grid[idx])
            global_kz = round(Int, wavenumber_info.kz_grid[idx])
            full_spectrum[(global_kx, global_ky, global_kz)] = vector_spectral[idx]
        elseif ndims_field == 2
            global_kx = round(Int, wavenumber_info.kx_grid[idx])
            global_ky = round(Int, wavenumber_info.ky_grid[idx])
            full_spectrum[(global_kx, global_ky)] = vector_spectral[idx]
        else
            global_kx = round(Int, wavenumber_info.kx_grid[idx[1]])
            full_spectrum[(global_kx,)] = vector_spectral[idx]
        end
    end

    return full_spectrum
end

"""
    calculate_kmax_from_bases(fourier_bases, global_shape)

Calculate maximum wavenumber (Nyquist limit) from Fourier bases.
"""
function calculate_kmax_from_bases(fourier_bases, global_shape)
    if isempty(fourier_bases)
        return 0
    end

    kmaxes = Int[]
    for (i, basis) in enumerate(fourier_bases)
        if i <= length(global_shape)
            N = global_shape[i]
            if isa(basis, RealFourier)
                # RFFT: kmax = N/2
                push!(kmaxes, N ÷ 2)
            else
                # FFT: kmax = N/2
                push!(kmaxes, N ÷ 2)
            end
        end
    end

    return isempty(kmaxes) ? 0 : minimum(kmaxes)
end

"""
    _get_pencil_array_offsets(velocity::VectorField)

Get the global index offsets for this rank's portion of the PencilArray.
Returns a tuple of offsets (one per dimension). For serial execution or
non-PencilArrays, returns zeros.

In PencilArrays, axes_local contains the global index ranges for each dimension
on this rank. The offset is first(range) - 1 for 1-based indexing.
"""
function _get_pencil_array_offsets(velocity::VectorField)
    first_component = velocity.components[1]
    coeff_data = get_coeff_data(first_component)

    # CRITICAL: Check for null coeff_data - can happen if field hasn't been transformed yet
    if coeff_data === nothing
        error("_get_pencil_array_offsets: coefficient data is nothing. " *
              "This typically means the field hasn't been transformed to spectral space yet. " *
              "Call forward_transform! on the field before computing spectral diagnostics.")
    end

    # Check if this is a PencilArray
    # NOTE: Don't use isdefined(Main, :PencilArrays) - it fails when Tarang is imported as a module.
    # Instead, check if coeff_data has the expected PencilArray properties directly.
    if velocity.dist.use_pencil_arrays && velocity.dist.size > 1
        # CRITICAL: For MPI runs with PencilArrays, getting offsets is MANDATORY.
        # Silent fallback to zero offsets would produce incorrect global indexing.

        # Check if this looks like a PencilArray (has pencil property or parent)
        # PencilArrays.axes_local returns global index ranges
        if hasproperty(coeff_data, :pencil)
            # Direct PencilArray access
            try
                axes = PencilArrays.axes_local(coeff_data)
                offsets = Tuple(first(ax) - 1 for ax in axes)
                return offsets
            catch e
                error("_get_pencil_array_offsets: PencilArray.axes_local failed: $e. " *
                      "Cannot compute correct MPI offsets.")
            end
        elseif applicable(PencilArrays.axes_local, coeff_data)
            # Try calling axes_local directly
            try
                axes = PencilArrays.axes_local(coeff_data)
                offsets = Tuple(first(ax) - 1 for ax in axes)
                return offsets
            catch e
                error("_get_pencil_array_offsets: axes_local call failed: $e. " *
                      "Cannot compute correct MPI offsets.")
            end
        else
            # CRITICAL: Expected PencilArray but got plain Array or wrapper
            # This can happen if:
            # 1. Upstream code assigned a plain Array into coeff_data
            # 2. coeff_data is a view/wrapper that doesn't expose pencil property
            # Must throw an error since continuing would produce incorrect results
            error("_get_pencil_array_offsets: MPI spectral diagnostics expected PencilArray " *
                  "but coeff_data is $(typeof(coeff_data)). " *
                  "This can happen if code assigned a plain Array into the coefficient data, " *
                  "or if the data is wrapped in a view. " *
                  "Ensure coefficient data retains its PencilArray wrapper for correct MPI indexing.")
        end
    end

    # Serial execution or non-PencilArrays: no offset
    return Tuple(zeros(Int, ndims(coeff_data)))
end
