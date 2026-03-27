"""
Flow analysis tools

Note: GlobalArrayReducer, reduce_scalar, global_min, global_max, and global_mean
are defined in core/evaluator.jl and reused here.
"""

using Statistics
using LinearAlgebra
using MPI
using FFTW: fftshift

# CFL condition calculation
mutable struct CFL
    solver::InitialValueSolver
    initial_dt::Float64
    safety::Float64
    threshold::Float64
    max_change::Float64
    min_change::Float64
    max_dt::Float64
    cadence::Int
    
    # State
    velocities::Vector{VectorField}
    current_dt::Float64
    iteration_count::Int
    reducer::GlobalArrayReducer
    
    function CFL(solver::InitialValueSolver; 
                initial_dt::Float64=0.01,
                cadence::Int=1,
                safety::Float64=0.4,
                threshold::Float64=0.1, 
                max_change::Float64=2.0,
                min_change::Float64=0.5,
                max_dt::Float64=Inf)
        
        comm = solver.base !== nothing ? solver_comm(solver.problem) : MPI.COMM_WORLD
        reducer = GlobalArrayReducer(comm)
        cfl = new(solver, initial_dt, safety, threshold, max_change, min_change, max_dt, 
                 cadence, VectorField[], initial_dt, 0, reducer)
        return cfl
    end
end

function add_velocity!(cfl::CFL, velocity::VectorField)
    """Add velocity field for CFL calculation"""
    push!(cfl.velocities, velocity)
end

function compute_timestep(cfl::CFL)
    """Compute adaptive timestep based on CFL condition"""

    cfl.iteration_count += 1

    # Only compute every cadence iterations
    if cfl.iteration_count % cfl.cadence != 0
        return cfl.current_dt
    end

    if isempty(cfl.velocities)
        return cfl.current_dt
    end

    min_dt = Inf

    for velocity in cfl.velocities
        # Get domain and grid spacing
        domain = velocity.domain
        spacings = grid_spacing(domain)

        for (i, component) in enumerate(velocity.components)
            ensure_layout!(component, :g)

            # Get velocity magnitude in this direction
            vel_data = abs.(get_grid_data(component))
            max_vel = maximum(vel_data)

            max_vel = global_max(cfl.reducer, max_vel)

            if max_vel > 0
                # CFL condition: dt < dx / |u|
                dt_limit = spacings[i] / max_vel
                min_dt = min(min_dt, dt_limit)
            end
        end
    end

    if min_dt == Inf
        proposed_dt = cfl.initial_dt
    else
        proposed_dt = cfl.safety * min_dt
    end

    # Apply constraints
    proposed_dt = min(proposed_dt, cfl.max_dt)

    # Limit change rate (only after first real CFL computation)
    # Skip rate limiting on first iteration to allow proper initialization
    if cfl.iteration_count > 1 && cfl.current_dt > 0 && cfl.current_dt != cfl.initial_dt
        max_allowed = cfl.current_dt * cfl.max_change
        min_allowed = cfl.current_dt * cfl.min_change
        proposed_dt = clamp(proposed_dt, min_allowed, max_allowed)
    end

    cfl.current_dt = proposed_dt
    return proposed_dt
end

# Reynolds number calculation
function reynolds_number(velocity::VectorField, viscosity::Float64, length_scale::Float64=1.0)
    """Calculate Reynolds number Re = |u| * L / ν"""
    
    # Calculate velocity magnitude (GPU-compatible allocation)
    ensure_layout!(velocity.components[1], :g)
    vel_magnitude_squared = similar_zeros(get_grid_data(velocity.components[1]))
    
    for component in velocity.components
        ensure_layout!(component, :g)
        vel_magnitude_squared .+= abs2.(get_grid_data(component))
    end
    
    vel_magnitude = sqrt.(vel_magnitude_squared)
    max_vel = maximum(vel_magnitude)
    
    reducer = GlobalArrayReducer()
    max_vel = global_max(reducer, max_vel)
    
    return max_vel * length_scale / viscosity
end

# Kinetic energy calculation
function kinetic_energy(velocity::VectorField, density::Float64=1.0)
    """Calculate kinetic energy KE = (1/2) * ρ * |u|²"""
    
    # Calculate |u|² (GPU-compatible allocation)
    ensure_layout!(velocity.components[1], :g)
    vel_squared = similar_zeros(get_grid_data(velocity.components[1]))
    
    for component in velocity.components
        ensure_layout!(component, :g)
        vel_squared .+= abs2.(get_grid_data(component))
    end
    
    ke_field = ScalarField(velocity.dist, "kinetic_energy", velocity.bases, velocity.dtype)
    ensure_layout!(ke_field, :g)
    get_grid_data(ke_field) .= 0.5 * density .* vel_squared
    
    return ke_field
end

function total_kinetic_energy(velocity::VectorField, density::Float64=1.0)
    """Calculate total kinetic energy integrated over domain"""
    
    ke_field = kinetic_energy(velocity, density)
    return integrate(ke_field)
end

# Enstrophy calculation (for 2D flows)
function enstrophy(velocity::VectorField)
    """Calculate enstrophy (vorticity squared) for 2D flow"""
    
    if velocity.coordsys.dim != 2
        throw(ArgumentError("Enstrophy calculation requires 2D velocity field"))
    end
    
    # Calculate vorticity: ω = ∂v/∂x - ∂u/∂y
    vorticity_op = curl(velocity)
    vorticity_field = evaluate_operator(vorticity_op)
    
    # Enstrophy = (1/2) * ω²
    enstrophy_field = ScalarField(velocity.dist, "enstrophy", velocity.bases, velocity.dtype)
    ensure_layout!(vorticity_field, :g)
    ensure_layout!(enstrophy_field, :g)
    
    get_grid_data(enstrophy_field) .= 0.5 .* abs2.(get_grid_data(vorticity_field))
    
    return enstrophy_field
end

function total_enstrophy(velocity::VectorField)
    """Calculate total enstrophy integrated over domain"""
    
    enstrophy_field = enstrophy(velocity)
    return integrate(enstrophy_field)
end

# Energy dissipation rate
function energy_dissipation_rate(velocity::VectorField, viscosity::Float64)
    """Calculate energy dissipation rate ε = ν * |∇u|²"""
    
    # Calculate strain rate tensor components (GPU-compatible allocation)
    ensure_layout!(velocity.components[1], :g)
    strain_rate_squared = similar_zeros(get_grid_data(velocity.components[1]))
    
    for i in 1:velocity.coordsys.dim
        for j in 1:velocity.coordsys.dim
            # ∂u_i/∂x_j
            coord = velocity.coordsys[j]
            du_i_dx_j = evaluate_differentiate(
                Differentiate(velocity.components[i], coord, 1), :g
            )
            ensure_layout!(du_i_dx_j, :g)
            
            # All terms contribute equally: ε = ν * Σ_{i,j} |∂u_i/∂x_j|² = ν * |∇u|²
            strain_rate_squared .+= abs2.(get_grid_data(du_i_dx_j))
        end
    end
    
    dissipation_field = ScalarField(velocity.dist, "dissipation", velocity.bases, velocity.dtype)
    ensure_layout!(dissipation_field, :g)
    get_grid_data(dissipation_field) .= viscosity .* strain_rate_squared
    
    return dissipation_field
end

# Vorticity dynamics
function vorticity_transport(velocity::VectorField, vorticity::ScalarField, viscosity::Float64)
    """Calculate vorticity transport equation terms for 2D flow"""
    
    if velocity.coordsys.dim != 2
        throw(ArgumentError("Vorticity transport requires 2D velocity field"))
    end
    
    # Advection term: u·∇ω
    grad_vorticity = grad(vorticity)
    advection = ScalarField(velocity.dist, "vorticity_advection", velocity.bases, velocity.dtype)
    ensure_layout!(advection, :g)
    fill!(get_grid_data(advection), 0.0)
    
    for i in 1:2
        ensure_layout!(velocity.components[i], :g)
        ensure_layout!(grad_vorticity.components[i], :g)
        get_grid_data(advection) .+= get_grid_data(velocity.components[i]) .* get_grid_data(grad_vorticity.components[i])
    end
    
    # Diffusion term: ν∇²ω
    lap_vorticity = lap(vorticity)
    diffusion = ScalarField(velocity.dist, "vorticity_diffusion", velocity.bases, velocity.dtype)
    ensure_layout!(lap_vorticity, :g)
    ensure_layout!(diffusion, :g)
    get_grid_data(diffusion) .= viscosity .* get_grid_data(lap_vorticity)
    
    return (advection=advection, diffusion=diffusion)
end

# ============================================================================
# Spectrum Binning Types (must be defined before spectrum functions)
# ============================================================================

"""
    SpectrumBinning

Configuration for spectrum binning to smooth power spectra.

# Fields
- `mode::Symbol`: Binning mode - `:linear`, `:log`, or `:custom`
- `bin_width::Float64`: Bin width for linear binning (default: 1.0)
- `num_bins::Int`: Number of bins for logarithmic binning
- `bin_edges::Vector{Float64}`: Custom bin edges (for `:custom` mode)
"""
struct SpectrumBinning
    mode::Symbol
    bin_width::Float64
    num_bins::Int
    bin_edges::Vector{Float64}
end

# Constructors for different binning modes
"""
    LinearBinning(; bin_width=1.0)

Create linear binning configuration with specified bin width.
Default bin_width=1.0 gives integer wavenumber bins (k=0,1,2,...).
Larger bin_width smooths the spectrum by averaging over wider k ranges.

# Example
```julia
# Standard integer bins
P = power_spectrum(field, binning=LinearBinning())

# Smoothed with Δk=2
P = power_spectrum(field, binning=LinearBinning(bin_width=2.0))
```
"""
LinearBinning(; bin_width::Real=1.0) = SpectrumBinning(:linear, Float64(bin_width), 0, Float64[])

"""
    LogBinning(; num_bins=nothing, bins_per_decade=10)

Create logarithmic binning configuration for spectra spanning many decades.
Useful for turbulence spectra where equal spacing in log(k) is desired.

# Arguments
- `num_bins::Int`: Total number of bins (overrides bins_per_decade if specified)
- `bins_per_decade::Int=10`: Number of bins per decade of wavenumber

# Example
```julia
# Logarithmic bins, 10 per decade
P = power_spectrum(field, binning=LogBinning())

# Coarser log bins
P = power_spectrum(field, binning=LogBinning(bins_per_decade=5))

# Fixed number of log bins
P = power_spectrum(field, binning=LogBinning(num_bins=50))
```
"""
function LogBinning(; num_bins::Union{Int,Nothing}=nothing, bins_per_decade::Int=10)
    if num_bins === nothing
        # Will be computed based on kmax later
        return SpectrumBinning(:log, 0.0, -bins_per_decade, Float64[])  # Negative = bins_per_decade
    else
        return SpectrumBinning(:log, 0.0, num_bins, Float64[])
    end
end

"""
    CustomBinning(bin_edges::Vector{<:Real})

Create custom binning with user-specified bin edges.

# Arguments
- `bin_edges`: Vector of bin edges [k0, k1, k2, ...]. Bins are [k0,k1), [k1,k2), etc.

# Example
```julia
# Custom bins: [1,2), [2,4), [4,8), [8,16), ...
edges = [1, 2, 4, 8, 16, 32, 64, 128]
P = power_spectrum(field, binning=CustomBinning(edges))
```
"""
CustomBinning(bin_edges::Vector{<:Real}) = SpectrumBinning(:custom, 0.0, length(bin_edges)-1, Float64.(bin_edges))

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

function validate_fourier_bases(velocity::VectorField)
    """Validate and extract Fourier basis information"""
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

struct WavenumberInfo
    kmax::Int
    k_magnitudes::Array{Float64}
    kx_grid::Array{Float64}
    ky_grid::Array{Float64}
    kz_grid::Union{Array{Float64}, Nothing}
    domain_size::Tuple{Vararg{Float64}}
    fourier_shape::Tuple{Vararg{Int}}
end

function get_wavenumber_info(velocity::VectorField, fourier_axes::Vector{Int}, fourier_bases::Vector)
    """
    Extract wavenumber grid information from velocity field and bases.
    Handles both 2D and 3D cases with proper PencilArrays integration.

    IMPORTANT: For MPI mode with PencilArrays, this function computes GLOBAL
    wavenumber values for each local grid point using the PencilArray offsets.
    """

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

function calculate_wavenumber_grids(fourier_bases::Vector, fourier_shape::Tuple{Vararg{Int}}, domain_size::Tuple{Vararg{Float64}})
    """
    Calculate wavenumber grids for each Fourier dimension (serial version).
    DEPRECATED: Use calculate_wavenumber_grids_global for MPI compatibility.
    """
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

function calculate_k_magnitudes(kx_grid::Array{Float64}, ky_grid::Array{Float64}, kz_grid::Union{Array{Float64}, Nothing})
    """Calculate wavenumber magnitudes |k| = √(kx² + ky² + kz²)"""
    
    if kz_grid !== nothing
        # 3D case
        k_magnitudes = sqrt.(kx_grid.^2 .+ ky_grid.^2 .+ kz_grid.^2)
    else
        # 2D case
        k_magnitudes = sqrt.(kx_grid.^2 .+ ky_grid.^2)
    end
    
    return k_magnitudes
end

function calculate_kmax(fourier_shape::Tuple{Vararg{Int}})
    """Calculate maximum wavenumber (Nyquist limit)"""
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

function calculate_spectral_kinetic_energy(velocity::VectorField; apply_conjugate_symmetry::Bool=true)
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

function calculate_full_energy_spectrum(velocity::VectorField, wavenumber_info::WavenumberInfo, max_wavenumber::Int)
    """
    Calculate full energy spectrum without radial averaging.
    Returns E(kx, ky) or E(kx, ky, kz) for detailed analysis.

    IMPORTANT: In MPI mode with PencilArrays, the returned dictionary uses GLOBAL
    wavenumber indices as keys (accounting for the local rank's offset in the
    distributed decomposition). Each rank returns its portion of the spectrum.

    To combine across ranks, use MPI gather operations (not reduce, since each rank
    has different wavenumbers). The keys are globally unique across ranks.
    """

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

# Helper functions for domain and shape extraction

"""
    get_domain_size(domain) -> Tuple

Extract physical domain size (extent) from domain object.

Returns a tuple of physical lengths for each dimension, computed from
the bounds of each basis in the domain.

# Arguments
- `domain`: Domain object containing bases with bounds information

# Returns
- Tuple of physical extents `(L1, L2, ...)` for each dimension

# Example
```julia
# For a 3D periodic box [0, 2π] × [0, 2π] × [0, 2π]
Lx, Ly, Lz = get_domain_size(domain)  # Returns (2π, 2π, 2π)
```
"""
function get_domain_size(domain)
    if domain === nothing
        @warn "No domain provided, returning default size"
        return (2π, 2π, 2π)
    end

    # Check if domain has bases
    if !hasfield(typeof(domain), :bases) || domain.bases === nothing
        @warn "Domain has no bases, returning default size"
        return (2π, 2π, 2π)
    end

    sizes = Float64[]

    for basis in domain.bases
        if basis === nothing
            push!(sizes, 2π)  # Default for missing basis
            continue
        end

        # Extract bounds from basis metadata
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :bounds)
            bounds = basis.meta.bounds
            if bounds !== nothing && length(bounds) >= 2
                extent = Float64(bounds[2] - bounds[1])
                push!(sizes, extent)
            else
                push!(sizes, 2π)  # Default
            end
        elseif hasfield(typeof(basis), :bounds)
            # Direct bounds field
            bounds = basis.bounds
            if bounds !== nothing && length(bounds) >= 2
                extent = Float64(bounds[2] - bounds[1])
                push!(sizes, extent)
            else
                push!(sizes, 2π)
            end
        else
            push!(sizes, 2π)  # Default for unknown basis type
        end
    end

    if isempty(sizes)
        return (2π, 2π, 2π)
    end

    return Tuple(sizes)
end

"""
    get_domain_bounds(domain) -> Vector{Tuple{Float64, Float64}}

Extract physical domain bounds from domain object.

Returns a vector of (min, max) tuples for each dimension.

# Arguments
- `domain`: Domain object containing bases with bounds information

# Returns
- Vector of `(min, max)` tuples for each dimension
"""
function get_domain_bounds(domain)
    if domain === nothing || !hasfield(typeof(domain), :bases) || domain.bases === nothing
        return [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    bounds_list = Tuple{Float64, Float64}[]

    for basis in domain.bases
        if basis === nothing
            push!(bounds_list, (0.0, 2π))
            continue
        end

        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :bounds)
            bounds = basis.meta.bounds
            if bounds !== nothing && length(bounds) >= 2
                push!(bounds_list, (Float64(bounds[1]), Float64(bounds[2])))
            else
                push!(bounds_list, (0.0, 2π))
            end
        elseif hasfield(typeof(basis), :bounds)
            bounds = basis.bounds
            if bounds !== nothing && length(bounds) >= 2
                push!(bounds_list, (Float64(bounds[1]), Float64(bounds[2])))
            else
                push!(bounds_list, (0.0, 2π))
            end
        else
            push!(bounds_list, (0.0, 2π))
        end
    end

    if isempty(bounds_list)
        return [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
    end

    return bounds_list
end

function get_fourier_shape(velocity::VectorField, fourier_axes::Vector{Int})
    """Extract local Fourier shape from velocity field"""
    first_component = velocity.components[1] 
    ensure_layout!(first_component, :c)
    return size(get_coeff_data(first_component))
end

# Turbulence statistics
function turbulence_statistics(velocity::VectorField)
    """Calculate basic turbulence statistics"""

    stats = Dict{String, Float64}()
    use_mpi = MPI.Initialized() && velocity.dist.size > 1
    comm = velocity.dist.comm

    # RMS velocity - CRITICAL: Use global reduction for MPI
    # Compute local sum of squares and local count, then reduce globally
    vel_sum_squared = 0.0
    local_count = 0
    for component in velocity.components
        ensure_layout!(component, :g)
        data = get_grid_data(component)
        vel_sum_squared += sum(abs2.(data))
        local_count += length(data)
    end

    if use_mpi
        # Global reduction: sum of squares and total count
        global_sum_squared = MPI.Allreduce(vel_sum_squared, MPI.SUM, comm)
        global_count = MPI.Allreduce(local_count, MPI.SUM, comm)
        stats["velocity_rms"] = sqrt(global_sum_squared / global_count)
    else
        stats["velocity_rms"] = sqrt(vel_sum_squared / local_count)
    end

    # Maximum velocity
    max_vel = 0.0
    for component in velocity.components
        ensure_layout!(component, :g)
        max_vel = max(max_vel, maximum(abs.(get_grid_data(component))))
    end

    if use_mpi
        # Use field's communicator, not COMM_WORLD, to support custom communicators
        max_vel = MPI.Allreduce(max_vel, MPI.MAX, comm)
    end
    stats["velocity_max"] = max_vel

    # Calculate derived quantities
    if velocity.coordsys.dim == 2
        # Vorticity RMS for 2D flows - CRITICAL: Use global reduction for MPI
        vort = evaluate_operator(curl(velocity))
        ensure_layout!(vort, :g)
        vort_data = get_grid_data(vort)
        vort_sum_squared = sum(abs2.(vort_data))
        vort_count = length(vort_data)

        if use_mpi
            global_vort_sum = MPI.Allreduce(vort_sum_squared, MPI.SUM, comm)
            global_vort_count = MPI.Allreduce(vort_count, MPI.SUM, comm)
            stats["vorticity_rms"] = sqrt(global_vort_sum / global_vort_count)
        else
            stats["vorticity_rms"] = sqrt(vort_sum_squared / vort_count)
        end

        # Maximum vorticity
        max_vort = maximum(abs.(vort_data))
        if use_mpi
            # Use field's communicator, not COMM_WORLD
            max_vort = MPI.Allreduce(max_vort, MPI.MAX, comm)
        end
        stats["vorticity_max"] = max_vort
    end

    return stats
end

# Flow visualization helpers  
function streamfunction(velocity::VectorField; boundary_condition::Symbol=:periodic, gauge_condition::Bool=true)
    """
    Calculate streamfunction for 2D incompressible flow.
    
    Solves the Poisson equation ∇²ψ = ω to obtain streamfunction from vorticity.
    Based on Tarang LBVP patterns for Poisson equation solving.
    
    For incompressible 2D flow: u = ∂ψ/∂y, v = -∂ψ/∂x
    Vorticity: ω = ∂v/∂x - ∂u/∂y = ∇²ψ
    
    Args:
        velocity: 2D VectorField 
        boundary_condition: :periodic, :no_slip, or :free_slip
        gauge_condition: Apply ∫ψ dA = 0 constraint
    
    Returns:
        ScalarField containing streamfunction ψ
    """
    
    if velocity.coordsys.dim != 2
        throw(ArgumentError("Streamfunction calculation requires 2D velocity field"))
    end
    
    # Calculate vorticity: ω = ∇ × u (2D curl)
    vorticity_field = evaluate_operator(curl(velocity))
    
    # Determine solution method based on basis types
    fourier_bases = get_fourier_basis_info(velocity.bases)
    
    if all_periodic_fourier(fourier_bases)
        # Pure Fourier bases - use direct spectral inversion
        return streamfunction_spectral_invert(vorticity_field, gauge_condition)
    else
        # Mixed/bounded bases - use boundary value problem solver
        return streamfunction_bvp_solve(vorticity_field, boundary_condition, gauge_condition)
    end
end

function get_fourier_basis_info(bases::Vector)
    """Extract information about Fourier vs non-Fourier bases"""
    fourier_info = []
    
    for (i, basis) in enumerate(bases)
        is_fourier = isa(basis, RealFourier) || isa(basis, ComplexFourier)
        push!(fourier_info, (index=i, is_fourier=is_fourier, basis=basis))
    end
    
    return fourier_info
end

function all_periodic_fourier(fourier_info::Vector)
    """Check if all bases are periodic Fourier"""
    return all(info.is_fourier for info in fourier_info)
end

function streamfunction_spectral_invert(vorticity::ScalarField, apply_gauge::Bool=true)
    """
    Direct spectral inversion for periodic domains: ψ̂(k) = -ω̂(k)/|k|²
    Based on Tarang spectral Poisson inversion patterns.
    """
    
    # Ensure vorticity is in spectral space
    ensure_layout!(vorticity, :c)
    
    # Create streamfunction field
    streamfunction_field = ScalarField(vorticity.dist, "streamfunction", vorticity.bases, vorticity.dtype)
    ensure_layout!(streamfunction_field, :c)
    
    # Get wavenumber information for Poisson inversion
    kx_grid, ky_grid = get_2d_wavenumber_grids(vorticity)
    k_squared = kx_grid.^2 .+ ky_grid.^2
    
    # Perform spectral inversion: ψ̂ = -ω̂/k²
    vorticity_spec = get_coeff_data(vorticity)
    streamfunction_spec = similar(vorticity_spec)
    
    for idx in CartesianIndices(vorticity_spec)
        k2 = k_squared[idx]
        
        if k2 > 1e-12  # Avoid division by zero
            # Spectral Poisson inversion: ψ̂ = -ω̂/k²
            streamfunction_spec[idx] = -vorticity_spec[idx] / k2
        else
            # Handle k=0 mode (constant/mean mode)
            if apply_gauge
                # Gauge condition: set mean to zero
                streamfunction_spec[idx] = 0.0
            else
                # Keep arbitrary constant (may need external constraint)
                streamfunction_spec[idx] = 0.0
            end
        end
    end
    
    # Set spectral coefficients
    get_coeff_data(streamfunction_field) .= streamfunction_spec

    # NOTE: MPI reduction is NOT needed here.
    # Each rank computes its local spectral coefficients from local vorticity.
    # The spectral Poisson inversion is fully local (no cross-rank dependencies).
    # Previous code incorrectly used MPI.Allreduce! which would corrupt the data.

    return streamfunction_field
end

function get_2d_wavenumber_grids(field::ScalarField)
    """
    Get 2D wavenumber grids for spectral operations.
    Returns properly scaled kx, ky grids.

    CRITICAL: For MPI/PencilArrays, uses global wavenumber indices with proper offsets.
    Each rank generates wavenumbers corresponding to its local portion of global spectrum.
    """

    # Extract domain size from field bases
    if field.domain !== nothing && length(field.domain.bases) >= 2
        Lx = field.domain.bases[1].meta.bounds[2] - field.domain.bases[1].meta.bounds[1]
        Ly = field.domain.bases[2].meta.bounds[2] - field.domain.bases[2].meta.bounds[1]
    else
        # Fallback for fields without domain info
        Lx = 2π
        Ly = 2π
        @warn "get_2d_wavenumber_grids: domain bounds not available, assuming Lx=Ly=2π" maxlog=1
    end

    # Get spectral shape and MPI offsets
    ensure_layout!(field, :c)
    coeff_data = get_coeff_data(field)
    local_nx, local_ny = size(coeff_data)[1:2]

    # CRITICAL FIX: Get global shape and local offsets for MPI
    # For distributed data, wavenumber indices must account for rank's offset in global spectrum
    use_mpi = field.dist.use_pencil_arrays && field.dist.size > 1
    if use_mpi
        # Get global shape for proper wavenumber indexing
        global_shape = _get_global_coeff_shape_internal(field)
        global_nx, global_ny = global_shape[1:2]

        # Get this rank's offset in global wavenumber space
        offsets = _get_pencil_array_offsets_internal(field)
        offset_x, offset_y = offsets[1:2]
    else
        global_nx, global_ny = local_nx, local_ny
        offset_x, offset_y = 0, 0
    end

    # Generate wavenumber grids based on basis types
    bases = field.bases

    # X-direction wavenumbers (with proper offset for MPI)
    if isa(bases[1], RealFourier)
        # RFFT: wavenumbers are 0, 1, 2, ..., N/2
        kx_indices = offset_x:(offset_x + local_nx - 1)
        kx_1d = 2π/Lx * kx_indices
    else  # ComplexFourier
        # FFT: wavenumbers are -N/2, ..., -1, 0, 1, ..., N/2-1 (after fftshift)
        # For distributed FFT, need to map local indices to global wavenumbers
        global_k_indices = fftshift(-global_nx÷2:(global_nx÷2-1))
        local_k_indices = global_k_indices[(offset_x+1):(offset_x+local_nx)]
        kx_1d = 2π/Lx * local_k_indices
    end

    # Y-direction wavenumbers (with proper offset for MPI)
    if isa(bases[2], RealFourier)
        ky_indices = offset_y:(offset_y + local_ny - 1)
        ky_1d = 2π/Ly * ky_indices
    else  # ComplexFourier
        global_k_indices = fftshift(-global_ny÷2:(global_ny÷2-1))
        local_k_indices = global_k_indices[(offset_y+1):(offset_y+local_ny)]
        ky_1d = 2π/Ly * local_k_indices
    end

    # Create 2D grids
    kx_grid = reshape(collect(kx_1d), length(kx_1d), 1)
    ky_grid = reshape(collect(ky_1d), 1, length(ky_1d))

    return kx_grid, ky_grid
end

"""
    _get_global_coeff_shape_internal(field::ScalarField)

Internal helper to get global coefficient shape. Similar to _get_global_coeff_shape
but works with ScalarField directly.
"""
function _get_global_coeff_shape_internal(field::ScalarField)
    coeff_data = get_coeff_data(field)
    if coeff_data === nothing
        error("_get_global_coeff_shape_internal: coefficient data is nothing")
    end

    if field.dist.use_pencil_arrays && field.dist.size > 1
        try
            if applicable(PencilArrays.size_global, coeff_data)
                return Tuple(PencilArrays.size_global(coeff_data))
            end
        catch
            # Fallback
        end
    end
    return size(coeff_data)
end

"""
    _get_pencil_array_offsets_internal(field::ScalarField)

Internal helper to get PencilArray offsets. Similar to _get_pencil_array_offsets
but works with ScalarField directly.
"""
function _get_pencil_array_offsets_internal(field::ScalarField)
    coeff_data = get_coeff_data(field)
    if coeff_data === nothing
        error("_get_pencil_array_offsets_internal: coefficient data is nothing")
    end

    if field.dist.use_pencil_arrays && field.dist.size > 1
        try
            if hasproperty(coeff_data, :pencil)
                axes = PencilArrays.axes_local(coeff_data)
                return Tuple(first(ax) - 1 for ax in axes)
            elseif applicable(PencilArrays.axes_local, coeff_data)
                axes = PencilArrays.axes_local(coeff_data)
                return Tuple(first(ax) - 1 for ax in axes)
            end
        catch e
            # CRITICAL: Don't silently fall back - error for MPI runs
            error("_get_pencil_array_offsets_internal: failed to get offsets for MPI data: $e")
        end
    end
    return Tuple(zeros(Int, ndims(coeff_data)))
end

function streamfunction_bvp_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool=true)
    """
    Solve streamfunction BVP for bounded/mixed domains.

    Solves ∇²ψ = ω with appropriate boundary conditions using Jacobi iteration.
    This is an iterative solver suitable for domains with physical boundaries.
    """

    streamfunction_field = ScalarField(vorticity.dist, "streamfunction", vorticity.bases, vorticity.dtype)
    ensure_layout!(streamfunction_field, :g)

    # Solve using Jacobi iteration with boundary conditions
    psi = streamfunction_jacobi_solve(vorticity, bc_type, apply_gauge)

    get_grid_data(streamfunction_field) .= psi
    return streamfunction_field
end

function streamfunction_jacobi_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool;
                                   max_iter::Int=1000, tolerance::Float64=1e-8)
    """
    Jacobi iteration solver for the Poisson equation ∇²ψ = ω.

    Uses classical Jacobi relaxation with configurable boundary conditions.
    Converges to tolerance or max_iter, whichever comes first.
    """
    
    ensure_layout!(vorticity, :g)
    omega = get_grid_data(vorticity)
    nx, ny = size(omega)
    
    # Initialize streamfunction
    psi = zeros(Float64, nx, ny)
    psi_new = similar(psi)
    
    # Extract grid spacing from domain, or fall back to 2π assumption
    if vorticity.domain !== nothing && length(vorticity.domain.bases) >= 2
        Lx = vorticity.domain.bases[1].meta.bounds[2] - vorticity.domain.bases[1].meta.bounds[1]
        Ly = vorticity.domain.bases[2].meta.bounds[2] - vorticity.domain.bases[2].meta.bounds[1]
    else
        Lx = 2π
        Ly = 2π
        @warn "streamfunction_jacobi_solve: domain bounds not available, assuming Lx=Ly=2π" maxlog=1
    end
    dx = Lx / nx
    dy = Ly / ny
    dx2 = dx^2
    dy2 = dy^2
    
    # Jacobi iteration coefficients
    alpha = 2.0 / (dx2 + dy2)
    beta_x = 1.0 / dx2
    beta_y = 1.0 / dy2
    
    for iter in 1:max_iter
        # Jacobi update: psi_new[i,j] = (omega[i,j] + beta_x*(psi[i-1,j] + psi[i+1,j]) + beta_y*(psi[i,j-1] + psi[i,j+1])) / alpha
        
        for i in 2:(nx-1), j in 2:(ny-1)
            psi_new[i,j] = (omega[i,j] + 
                           beta_x * (psi[i-1,j] + psi[i+1,j]) + 
                           beta_y * (psi[i,j-1] + psi[i,j+1])) / alpha
        end
        
        # Apply boundary conditions
        apply_streamfunction_bc!(psi_new, bc_type)
        
        # Check convergence
        residual = maximum(abs.(psi_new - psi))
        if residual < tolerance
            break
        end
        
        # Update
        psi, psi_new = psi_new, psi
    end
    
    # Apply gauge condition if requested
    if apply_gauge
        psi .-= mean(psi)  # Set mean to zero
    end

    # NOTE: MPI reduction is NOT needed here.
    # This function operates on a local Array (not a distributed field).
    # The Jacobi iteration works on the local grid; if distributed execution is needed,
    # proper halo exchange should be implemented instead of Allreduce.
    # Previous code incorrectly used MPI.Allreduce! which would corrupt the data.

    return psi
end

function apply_streamfunction_bc!(psi::Array{Float64,2}, bc_type::Symbol)
    """Apply boundary conditions to streamfunction"""
    
    nx, ny = size(psi)
    
    if bc_type == :no_slip
        # No-slip: ψ = 0 on boundaries
        psi[1, :] .= 0.0      # Bottom
        psi[nx, :] .= 0.0     # Top  
        psi[:, 1] .= 0.0      # Left
        psi[:, ny] .= 0.0     # Right
        
    elseif bc_type == :free_slip
        # Free-slip: ∂ψ/∂n = 0 on boundaries (approximate)
        psi[1, :] .= psi[2, :]        # Bottom
        psi[nx, :] .= psi[nx-1, :]    # Top
        psi[:, 1] .= psi[:, 2]        # Left  
        psi[:, ny] .= psi[:, ny-1]    # Right
        
    elseif bc_type == :periodic
        # Periodic: handled naturally by spectral methods
        # No explicit boundary conditions needed
        
    else
        throw(ArgumentError("Unknown boundary condition: $bc_type"))
    end
end

function validate_streamfunction(velocity::VectorField, streamfunction::ScalarField; tolerance::Float64=1e-6)
    """
    Validate streamfunction by checking if it generates the correct velocity field.
    
    For 2D incompressible flow: u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    
    if velocity.coordsys.dim != 2
        return false
    end
    
    # Calculate velocity from streamfunction
    dpsi_dy = evaluate_differentiate(Differentiate(streamfunction, velocity.coordsys[2], 1), :g)
    dpsi_dx = evaluate_differentiate(Differentiate(streamfunction, velocity.coordsys[1], 1), :g)
    
    # Check u = ∂ψ/∂y
    ensure_layout!(velocity.components[1], :g)
    ensure_layout!(dpsi_dy, :g)
    u_error = maximum(abs.(get_grid_data(velocity.components[1]) - get_grid_data(dpsi_dy)))
    
    # Check v = -∂ψ/∂x
    ensure_layout!(velocity.components[2], :g)
    ensure_layout!(dpsi_dx, :g)  
    v_error = maximum(abs.(get_grid_data(velocity.components[2]) + get_grid_data(dpsi_dx)))
    
    # MPI reduction for global error using field's communicator
    if MPI.Initialized()
        comm = velocity.components[1].dist.comm
        u_error = MPI.Allreduce(u_error, MPI.MAX, comm)
        v_error = MPI.Allreduce(v_error, MPI.MAX, comm)
    end

    max_error = max(u_error, v_error)
    is_valid = max_error < tolerance
    
    return (valid=is_valid, u_error=u_error, v_error=v_error, max_error=max_error)
end

function velocity_divergence(velocity::VectorField)
    """Calculate velocity divergence ∇·u"""

    divergence_op = divergence(velocity)
    return evaluate_operator(divergence_op)
end

# ============================================================================
# Surface Quasi-Geostrophic (SQG) Flow Tools
# ============================================================================

"""
    perp_grad(ψ::ScalarField)

Perpendicular gradient: ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)

Used in 2D flows to compute velocity from streamfunction: u = ∇⊥ψ

This gives a divergence-free velocity field: ∇·u = ∇·(∇⊥ψ) = 0

# Example
```julia
ψ = ScalarField(dist, "psi", bases, Float64)
u = perp_grad(ψ)  # Returns VectorField with u_x = -∂ψ/∂y, u_y = ∂ψ/∂x
```
"""
function perp_grad(ψ::ScalarField)
    # Get coordinates
    coords = ψ.dist.coordsys

    if length(ψ.bases) != 2
        throw(ArgumentError("perp_grad requires a 2D field"))
    end

    # Get coordinates via CoordinateSystem indexing
    coord1 = coords[1]  # x
    coord2 = coords[2]  # y

    # Create result vector field
    result = VectorField(ψ.dist, coords, "perp_grad_$(ψ.name)", ψ.bases, ψ.dtype)

    # u_x = -∂ψ/∂y (second coordinate)
    dpsi_dy = evaluate_differentiate(Differentiate(ψ, coord2, 1), :g)
    ensure_layout!(result.components[1], :g)
    get_grid_data(result.components[1]) .= -get_grid_data(dpsi_dy)

    # u_y = ∂ψ/∂x (first coordinate)
    dpsi_dx = evaluate_differentiate(Differentiate(ψ, coord1, 1), :g)
    ensure_layout!(result.components[2], :g)
    get_grid_data(result.components[2]) .= get_grid_data(dpsi_dx)

    return result
end

# Unicode alias for perpendicular gradient
const ∇⊥ = perp_grad

"""
    sqg_streamfunction(θ::ScalarField)

Compute SQG streamfunction from surface buoyancy:

    ψ = (-Δ)^(-1/2) θ

In spectral space: ψ̂(k) = θ̂(k) / |k|

This is the fundamental relation for Surface Quasi-Geostrophic dynamics.

# Example
```julia
θ = ScalarField(dist, "buoyancy", bases, Float64)
ψ = sqg_streamfunction(θ)
u = perp_grad(ψ)  # or u = ∇⊥(ψ)
```
"""
function sqg_streamfunction(θ::ScalarField)
    return evaluate_fractional_laplacian(FractionalLaplacian(θ, -0.5), :g)
end

"""
    sqg_velocity(θ::ScalarField)

Compute SQG velocity directly from surface buoyancy:

    u = ∇⊥ψ = ∇⊥(-Δ)^(-1/2) θ

This combines streamfunction inversion and perpendicular gradient in one step.

# Example
```julia
θ = ScalarField(dist, "buoyancy", bases, Float64)
u = sqg_velocity(θ)  # Directly get velocity from buoyancy
```
"""
function sqg_velocity(θ::ScalarField)
    ψ = sqg_streamfunction(θ)
    return perp_grad(ψ)
end

"""
    sqg_problem_setup(dist::Distributor, bases::Tuple; κ::Real=0.0, α::Real=0.5)

Set up an SQG initial value problem.

The SQG equation is:
    ∂θ/∂t + u·∇θ = κ(-Δ)^α θ

where:
- θ is the surface buoyancy/temperature
- u = ∇⊥(-Δ)^(-1/2)θ is the velocity
- κ is the dissipation coefficient
- α is the dissipation exponent (typically 1/2 for physical SQG)

# Arguments
- `dist`: Distributor for field allocation
- `bases`: Tuple of bases (should be 2D Fourier bases for SQG)
- `κ`: Dissipation coefficient (default: 0 for inviscid)
- `α`: Dissipation exponent (default: 0.5 for physical dissipation)

# Returns
Named tuple with:
- `problem`: IVP problem object
- `θ`: Buoyancy field
- `u`: Velocity field (computed from θ)
- `ψ`: Streamfunction field (computed from θ)

# Example
```julia
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(2, 2))
x_basis = RealFourier(coords["x"]; size=256, bounds=(0.0, 2π))
y_basis = RealFourier(coords["y"]; size=256, bounds=(0.0, 2π))

sqg = sqg_problem_setup(dist, (x_basis, y_basis); κ=1e-4, α=0.5)

# Set initial condition
    get_grid_data(sqg.θ) .= initial_buoyancy_field

# Solve with timestepper
solver = InitialValueSolver(sqg.problem, RK443())
```
"""
function sqg_problem_setup(dist::Distributor, bases::Tuple; κ::Real=0.0, α::Real=0.5)
    coords = dist.coordsys

    # Create fields
    θ = ScalarField(dist, "θ", bases, Float64)  # Buoyancy

    # Create IVP
    problem = IVP([θ])

    # Add parameters
    problem.parameters["κ"] = Float64(κ)
    problem.parameters["α"] = Float64(α)

    # The SQG equation: ∂θ/∂t + u·∇θ = κ(-Δ)^α θ
    # where u = ∇⊥(-Δ)^(-1/2)θ
    #
    # Note: The nonlinear term u·∇θ needs special handling because
    # u depends on θ through the streamfunction inversion.
    # This is typically handled in the timestepper's nonlinear evaluation.

    if κ > 0
        add_equation!(problem, "∂t(θ) = κ*fraclap(θ, α)")
    else
        add_equation!(problem, "∂t(θ) = 0")
    end

    # Note: The advection term -u·∇θ should be added to the RHS
    # as a nonlinear term evaluated at each timestep

    @info "SQG problem setup" κ=κ α=α bases=length(bases)
    @info "Use sqg_velocity(θ) to compute velocity from buoyancy at each timestep"

    return (
        problem = problem,
        θ = θ,
        compute_velocity = () -> sqg_velocity(θ),
        compute_streamfunction = () -> sqg_streamfunction(θ)
    )
end

# ============================================================================
# Full Quasi-Geostrophic (QG) with Surface Buoyancy Dynamics
# ============================================================================

"""
    QGSystem

A coupled Quasi-Geostrophic system with:
- Surface buoyancy dynamics at z=0 and z=H (IVP, time-evolving)
- Interior PV inversion (LBVP, diagnostic at each timestep)

The system solves:

**Interior (LBVP for ψ given q and boundary θ):**
```
∇²ψ + (f₀/N)² ∂²ψ/∂z² = q
```
with Neumann BCs from surface buoyancy:
```
∂ψ/∂z|_{z=0} = (N/f₀) θ_bot
∂ψ/∂z|_{z=H} = (N/f₀) θ_top
```

**Surfaces (IVP for θ):**
```
∂θ/∂t + u·∇θ = κ(-Δ)^α θ
```
where u = ∇⊥ψ evaluated at the surface.
"""
mutable struct QGSystem
    # Domain info
    dist_3d::Distributor          # 3D distributor for interior
    dist_2d_bot::Distributor      # 2D distributor for bottom surface
    dist_2d_top::Distributor      # 2D distributor for top surface

    # Fields
    ψ::ScalarField                # 3D streamfunction (interior)
    q::ScalarField                # 3D potential vorticity (interior)
    θ_bot::ScalarField            # 2D surface buoyancy (bottom)
    θ_top::ScalarField            # 2D surface buoyancy (top)

    # Problems
    interior_bvp::LBVP            # Interior PV inversion problem
    surface_ivp_bot::IVP          # Bottom surface evolution
    surface_ivp_top::IVP          # Top surface evolution

    # Parameters
    f0::Float64                   # Coriolis parameter
    N::Float64                    # Buoyancy frequency
    H::Float64                    # Domain height
    κ::Float64                    # Surface dissipation coefficient
    α::Float64                    # Dissipation exponent

    # Computed fields (cached)
    u_bot::Union{Nothing, VectorField}
    u_top::Union{Nothing, VectorField}
end

"""
    qg_system_setup(;
        Lx, Ly, H,           # Domain size
        Nx, Ny, Nz,          # Resolution
        f0=1.0, N=1.0,       # Physical parameters
        κ=0.0, α=0.5,        # Surface dissipation
        mesh_xy=(1,1),       # MPI mesh for horizontal
        mesh_z=1             # MPI mesh for vertical (usually 1)
    )

Set up a full 3D Quasi-Geostrophic system with surface buoyancy dynamics.

# Physical Setup
- Doubly-periodic in x, y
- Bounded in z ∈ [0, H] with Chebyshev discretization
- Surface buoyancy θ at z=0 (bottom) and z=H (top)
- Interior PV q (can be zero for SQG limit)

# Algorithm (at each timestep)
1. Given θ_bot, θ_top, q → Solve LBVP for ψ
2. Compute u = ∇⊥ψ at surfaces
3. Advance θ_bot, θ_top using surface advection

# Example
```julia
qg = qg_system_setup(
    Lx=2π, Ly=2π, H=1.0,
    Nx=128, Ny=128, Nz=32,
    f0=1.0, N=10.0,
    κ=1e-4, α=0.5
)

# Set initial surface buoyancy
qg.θget_grid_data(_bot) .= initial_buoyancy_bottom
qg.θget_grid_data(_top) .= initial_buoyancy_top

# Set interior PV (zero for SQG limit)
get_grid_data(qg.q) .= 0.0

# Time stepping loop
for iter in 1:nsteps
    qg_step!(qg, dt)
end
```
"""
function qg_system_setup(;
    Lx::Real, Ly::Real, H::Real,
    Nx::Int, Ny::Int, Nz::Int,
    f0::Real=1.0, N::Real=1.0,
    κ::Real=0.0, α::Real=0.5,
    mesh_xy::Tuple{Int,Int}=(1,1),
    mesh_z::Int=1
)
    # 3D coordinates and distributor for interior
    coords_3d = CartesianCoordinates("x", "y", "z")
    dist_3d = Distributor(coords_3d; mesh=(mesh_xy..., mesh_z), dtype=Float64)

    # 3D bases
    x_basis = RealFourier(coords_3d["x"]; size=Nx, bounds=(0.0, Float64(Lx)))
    y_basis = RealFourier(coords_3d["y"]; size=Ny, bounds=(0.0, Float64(Ly)))
    z_basis = ChebyshevT(coords_3d["z"]; size=Nz, bounds=(0.0, Float64(H)))
    bases_3d = (x_basis, y_basis, z_basis)

    # 2D coordinates and distributors for surfaces
    coords_2d = CartesianCoordinates("x", "y")
    dist_2d_bot = Distributor(coords_2d; mesh=mesh_xy, dtype=Float64)
    dist_2d_top = Distributor(coords_2d; mesh=mesh_xy, dtype=Float64)

    # 2D bases for surfaces
    x_basis_2d = RealFourier(coords_2d["x"]; size=Nx, bounds=(0.0, Float64(Lx)))
    y_basis_2d = RealFourier(coords_2d["y"]; size=Ny, bounds=(0.0, Float64(Ly)))
    bases_2d = (x_basis_2d, y_basis_2d)

    # Create 3D fields
    ψ = ScalarField(dist_3d, "ψ", bases_3d, Float64)
    q = ScalarField(dist_3d, "q", bases_3d, Float64)

    # Create 2D surface fields
    θ_bot = ScalarField(dist_2d_bot, "θ_bot", bases_2d, Float64)
    θ_top = ScalarField(dist_2d_top, "θ_top", bases_2d, Float64)

    # Interior LBVP: ∇²ψ + (f₀/N)² ∂²ψ/∂z² = q
    # This is the QG elliptic inversion
    interior_bvp = LBVP([ψ])
    interior_bvp.parameters["f0"] = Float64(f0)
    interior_bvp.parameters["N"] = Float64(N)
    interior_bvp.parameters["S"] = (f0/N)^2  # Burger number squared

    # QG elliptic operator: ∇_h²ψ + S·∂²ψ/∂z² = q
    add_equation!(interior_bvp, "Δ(ψ) + S*∂z(∂z(ψ)) = q")

    # Boundary conditions: ∂ψ/∂z = (N/f₀)θ at surfaces
    # These will be updated at each timestep with current θ values
    add_equation!(interior_bvp, "∂z(ψ)(z=0) = (N/f0)*θ_bot")
    add_equation!(interior_bvp, "∂z(ψ)(z=$H) = (N/f0)*θ_top")

    # Surface IVPs: ∂θ/∂t + u·∇θ = κ(-Δ)^α θ
    surface_ivp_bot = IVP([θ_bot])
    surface_ivp_bot.parameters["κ"] = Float64(κ)
    surface_ivp_bot.parameters["α"] = Float64(α)

    surface_ivp_top = IVP([θ_top])
    surface_ivp_top.parameters["κ"] = Float64(κ)
    surface_ivp_top.parameters["α"] = Float64(α)

    if κ > 0
        add_equation!(surface_ivp_bot, "∂t(θ_bot) - κ*fraclap(θ_bot, α) = 0")
        add_equation!(surface_ivp_top, "∂t(θ_top) - κ*fraclap(θ_top, α) = 0")
    else
        add_equation!(surface_ivp_bot, "∂t(θ_bot) = 0")
        add_equation!(surface_ivp_top, "∂t(θ_top) = 0")
    end

    @info "QG system setup complete" Nx=Nx Ny=Ny Nz=Nz f0=f0 N=N H=H κ=κ α=α

    return QGSystem(
        dist_3d, dist_2d_bot, dist_2d_top,
        ψ, q, θ_bot, θ_top,
        interior_bvp, surface_ivp_bot, surface_ivp_top,
        Float64(f0), Float64(N), Float64(H), Float64(κ), Float64(α),
        nothing, nothing
    )
end

"""
    qg_invert!(qg::QGSystem)

Perform QG PV inversion: given q, θ_bot, θ_top → solve for ψ.

Solves the elliptic problem:
    ∇²ψ + (f₀/N)² ∂²ψ/∂z² = q
with boundary conditions:
    ∂ψ/∂z|_{z=0} = (N/f₀) θ_bot
    ∂ψ/∂z|_{z=H} = (N/f₀) θ_top
"""
function qg_invert!(qg::QGSystem)
    # Update boundary condition values with current surface buoyancy
    # This requires updating the BC namespace with current θ values
    qg.interior_bvp.namespace["θ_bot"] = qg.θ_bot
    qg.interior_bvp.namespace["θ_top"] = qg.θ_top
    qg.interior_bvp.namespace["q"] = qg.q

    # Solve the LBVP
    solver = BoundaryValueSolver(qg.interior_bvp)
    solve!(solver)

    # Solution is in qg.ψ
    return qg.ψ
end

"""
    qg_surface_velocity!(qg::QGSystem)

Compute horizontal velocity at surfaces from streamfunction.
u = ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)
"""
function qg_surface_velocity!(qg::QGSystem)
    # Extract ψ at z=0 (bottom surface)
    ψ_bot = extract_surface(qg.ψ, :z, 0.0)
    qg.u_bot = perp_grad(ψ_bot)

    # Extract ψ at z=H (top surface)
    ψ_top = extract_surface(qg.ψ, :z, qg.H)
    qg.u_top = perp_grad(ψ_top)

    return qg.u_bot, qg.u_top
end

"""
    extract_surface(field_3d::ScalarField, dim::Symbol, position::Real)

Extract a 2D slice from a 3D field at a given position along dimension `dim`.
"""
function extract_surface(field_3d::ScalarField, dim::Symbol, position::Real)
    # Find the dimension index
    coord_names = field_3d.dist.coordsys.names
    dim_idx = findfirst(==(String(dim)), coord_names)

    if dim_idx === nothing
        throw(ArgumentError("Dimension $dim not found in field"))
    end

    # Use interpolation to extract the surface
    coord = field_3d.dist.coordsys[String(dim)]
    interp_op = Interpolate(field_3d, coord, position)

    return evaluate_interpolate(interp_op, :g)
end

"""
    qg_advection_rhs(qg::QGSystem)

Compute the advection terms -u·∇θ for surface buoyancy equations.

Returns (rhs_bot, rhs_top) where each is the RHS for the respective surface.
"""
function qg_advection_rhs(qg::QGSystem)
    # Ensure we have surface velocities
    if qg.u_bot === nothing || qg.u_top === nothing
        qg_surface_velocity!(qg)
    end

    # Compute -u·∇θ at each surface
    # Bottom surface
    ensure_layout!(qg.θ_bot, :g)
    ensure_layout!(qg.u_bot.components[1], :g)
    ensure_layout!(qg.u_bot.components[2], :g)

    coords_2d = qg.θ_bot.dist.coordsys
    coord_x = coords_2d[1]
    coord_y = coords_2d[2]

    dθ_dx_bot = evaluate_differentiate(Differentiate(qg.θ_bot, coord_x, 1), :g)
    dθ_dy_bot = evaluate_differentiate(Differentiate(qg.θ_bot, coord_y, 1), :g)

    rhs_bot = ScalarField(qg.dist_2d_bot, "rhs_bot", qg.θ_bot.bases, Float64)
    ensure_layout!(rhs_bot, :g)
    get_grid_data(rhs_bot) .= -(get_grid_data(qg.u_bot.components[1]) .* dθget_grid_data(_dx_bot) .+
                        get_grid_data(qg.u_bot.components[2]) .* dθget_grid_data(_dy_bot))

    # Top surface
    ensure_layout!(qg.θ_top, :g)
    ensure_layout!(qg.u_top.components[1], :g)
    ensure_layout!(qg.u_top.components[2], :g)

    dθ_dx_top = evaluate_differentiate(Differentiate(qg.θ_top, coord_x, 1), :g)
    dθ_dy_top = evaluate_differentiate(Differentiate(qg.θ_top, coord_y, 1), :g)

    rhs_top = ScalarField(qg.dist_2d_top, "rhs_top", qg.θ_top.bases, Float64)
    ensure_layout!(rhs_top, :g)
    get_grid_data(rhs_top) .= -(get_grid_data(qg.u_top.components[1]) .* dθget_grid_data(_dx_top) .+
                        get_grid_data(qg.u_top.components[2]) .* dθget_grid_data(_dy_top))

    return rhs_bot, rhs_top
end

"""
    qg_step!(qg::QGSystem, dt::Real; timestepper=:RK4)

Perform one timestep of the QG system.

Algorithm:
1. Invert PV: solve LBVP for ψ given q, θ_bot, θ_top
2. Compute surface velocities from ψ
3. Advance surface buoyancy: θ_new = θ + dt * (-u·∇θ + κ(-Δ)^α θ)

# Arguments
- `qg`: QGSystem to advance
- `dt`: Timestep size
- `timestepper`: Time integration method (:RK4, :RK2, :Euler)
"""
function qg_step!(qg::QGSystem, dt::Real; timestepper::Symbol=:RK4)
    if timestepper == :RK4
        qg_step_rk4!(qg, dt)
    elseif timestepper == :RK2
        qg_step_rk2!(qg, dt)
    elseif timestepper == :Euler
        qg_step_euler!(qg, dt)
    else
        throw(ArgumentError("Unknown timestepper: $timestepper"))
    end
end

function qg_step_euler!(qg::QGSystem, dt::Real)
    # 1. Invert for streamfunction
    qg_invert!(qg)

    # 2. Compute surface velocities
    qg_surface_velocity!(qg)

    # 3. Compute advection RHS
    rhs_bot, rhs_top = qg_advection_rhs(qg)

    # 4. Add dissipation if present
    if qg.κ > 0
        frac_lap_bot = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_bot, qg.α), :g)
        frac_lap_top = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_top, qg.α), :g)
        get_grid_data(rhs_bot) .+= qg.κ .* get_grid_data(frac_lap_bot)
        get_grid_data(rhs_top) .+= qg.κ .* get_grid_data(frac_lap_top)
    end

    # 5. Euler step
    qg.θget_grid_data(_bot) .+= dt .* get_grid_data(rhs_bot)
    qg.θget_grid_data(_top) .+= dt .* get_grid_data(rhs_top)
end

function qg_step_rk2!(qg::QGSystem, dt::Real)
    # Save initial state
    θ_bot_0 = copy(qg.θget_grid_data(_bot))
    θ_top_0 = copy(qg.θget_grid_data(_top))

    # Stage 1: Euler step to midpoint
    qg_step_euler!(qg, dt/2)

    # Stage 2: Full step from original using midpoint derivative
    qg_invert!(qg)
    qg_surface_velocity!(qg)
    rhs_bot, rhs_top = qg_advection_rhs(qg)

    if qg.κ > 0
        frac_lap_bot = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_bot, qg.α), :g)
        frac_lap_top = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_top, qg.α), :g)
        get_grid_data(rhs_bot) .+= qg.κ .* get_grid_data(frac_lap_bot)
        get_grid_data(rhs_top) .+= qg.κ .* get_grid_data(frac_lap_top)
    end

    qg.θget_grid_data(_bot) .= θ_bot_0 .+ dt .* get_grid_data(rhs_bot)
    qg.θget_grid_data(_top) .= θ_top_0 .+ dt .* get_grid_data(rhs_top)
end

function qg_step_rk4!(qg::QGSystem, dt::Real)
    # Save initial state
    θ_bot_0 = copy(qg.θget_grid_data(_bot))
    θ_top_0 = copy(qg.θget_grid_data(_top))

    function compute_rhs()
        qg_invert!(qg)
        qg_surface_velocity!(qg)
        rhs_bot, rhs_top = qg_advection_rhs(qg)

        if qg.κ > 0
            frac_lap_bot = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_bot, qg.α), :g)
            frac_lap_top = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_top, qg.α), :g)
            get_grid_data(rhs_bot) .+= qg.κ .* get_grid_data(frac_lap_bot)
            get_grid_data(rhs_top) .+= qg.κ .* get_grid_data(frac_lap_top)
        end

        return copy(get_grid_data(rhs_bot)), copy(get_grid_data(rhs_top))
    end

    # k1
    k1_bot, k1_top = compute_rhs()

    # k2
    qg.θget_grid_data(_bot) .= θ_bot_0 .+ (dt/2) .* k1_bot
    qg.θget_grid_data(_top) .= θ_top_0 .+ (dt/2) .* k1_top
    k2_bot, k2_top = compute_rhs()

    # k3
    qg.θget_grid_data(_bot) .= θ_bot_0 .+ (dt/2) .* k2_bot
    qg.θget_grid_data(_top) .= θ_top_0 .+ (dt/2) .* k2_top
    k3_bot, k3_top = compute_rhs()

    # k4
    qg.θget_grid_data(_bot) .= θ_bot_0 .+ dt .* k3_bot
    qg.θget_grid_data(_top) .= θ_top_0 .+ dt .* k3_top
    k4_bot, k4_top = compute_rhs()

    # Final update
    qg.θget_grid_data(_bot) .= θ_bot_0 .+ (dt/6) .* (k1_bot .+ 2 .* k2_bot .+ 2 .* k3_bot .+ k4_bot)
    qg.θget_grid_data(_top) .= θ_top_0 .+ (dt/6) .* (k1_top .+ 2 .* k2_top .+ 2 .* k3_top .+ k4_top)
end

# ============================================================================
# General Boundary Advection-Diffusion Framework
# ============================================================================

"""
    VelocitySource

Abstract type for specifying how boundary velocity is obtained.
"""
abstract type VelocitySource end

"""
    PrescribedVelocity <: VelocitySource

Velocity is prescribed externally (user sets values directly).
"""
struct PrescribedVelocity <: VelocitySource end

"""
    InteriorDerivedVelocity <: VelocitySource

Velocity derived from an interior field (e.g., u = ∇⊥ψ at surface).

# Fields
- `operator`: How to compute velocity from interior field (:perp_grad, :grad, :curl)
"""
struct InteriorDerivedVelocity <: VelocitySource
    operator::Symbol  # :perp_grad, :grad, :curl
end

"""
    SelfDerivedVelocity <: VelocitySource

Velocity derived from the boundary field itself (e.g., SQG: u = ∇⊥(-Δ)^(-1/2)θ).

# Fields
- `inversion_exponent`: Exponent for fractional Laplacian inversion (e.g., -0.5 for SQG)
- `use_perp_grad`: Whether to use perpendicular gradient (for 2D incompressible)
"""
struct SelfDerivedVelocity <: VelocitySource
    inversion_exponent::Float64
    use_perp_grad::Bool

    SelfDerivedVelocity(; inversion_exponent::Real=-0.5, use_perp_grad::Bool=true) =
        new(Float64(inversion_exponent), use_perp_grad)
end

"""
    DiffusionSpec

Specification for diffusion operator on the boundary.

# Fields
- `type`: Type of diffusion (:laplacian, :fractional, :hyperdiffusion, :none)
- `coefficient`: Diffusion coefficient κ
- `exponent`: Exponent α for fractional/hyper diffusion
- `implicit`: Whether to treat diffusion implicitly in timestepping
"""
struct DiffusionSpec
    type::Symbol
    coefficient::Float64
    exponent::Float64
    implicit::Bool

    function DiffusionSpec(;
        type::Symbol=:laplacian,
        coefficient::Real=0.0,
        exponent::Real=1.0,
        implicit::Bool=false
    )
        valid_types = (:laplacian, :fractional, :hyperdiffusion, :none)
        if !(type in valid_types)
            throw(ArgumentError("Diffusion type must be one of $valid_types"))
        end
        new(type, Float64(coefficient), Float64(exponent), implicit)
    end
end

"""
    BoundarySpec

Specification for a single boundary surface.

# Fields
- `name`: Identifier for this boundary (e.g., "bottom", "top")
- `dimension`: Coordinate dimension (:x, :y, :z)
- `position`: Position along that dimension
- `field_name`: Name for the boundary field
"""
struct BoundarySpec
    name::String
    dimension::Symbol
    position::Float64
    field_name::String

    BoundarySpec(name::String, dim::Symbol, pos::Real; field_name::String=name) =
        new(name, dim, Float64(pos), field_name)
end

"""
    BoundaryAdvectionDiffusion

General framework for solving advection-diffusion equations on domain boundaries,
optionally coupled to interior dynamics.

Solves equations of the form:
    ∂c/∂t + u·∇c = D(c) + S

where:
- c is the boundary scalar field
- u is the advection velocity (from various sources)
- D(c) is the diffusion operator (standard, fractional, or hyper)
- S is an optional source term

# Supported Configurations

1. **Standalone boundary dynamics** (like pure SQG):
   - Velocity derived from boundary field itself
   - No interior coupling

2. **Interior-coupled dynamics** (like full QG):
   - Boundary provides BC for interior problem
   - Velocity extracted from interior solution

3. **Prescribed velocity**:
   - User sets velocity externally each timestep
   - Useful for passive tracer advection

# Example
```julia
# Create SQG-like system
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π,
    Nx=256, Ny=256,
    boundaries=[BoundarySpec("surface", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
)

# Set initial condition
get_grid_data(bad.fields["surface"]) .= initial_condition

# Time step
for i in 1:nsteps
    bad_step!(bad, dt)
end
```
"""
mutable struct BoundaryAdvectionDiffusion
    # Domain information
    dist::Distributor                           # Boundary distributor
    bases::Tuple                                # Basis functions
    coords::CartesianCoordinates               # Coordinate system

    # Boundaries (can have multiple, e.g., top and bottom)
    boundary_specs::Vector{BoundarySpec}

    # Fields on each boundary
    fields::Dict{String, ScalarField}

    # Velocity specification
    velocity_source::VelocitySource
    velocities::Dict{String, VectorField}

    # Diffusion specification
    diffusion::DiffusionSpec

    # Optional interior coupling
    interior_dist::Union{Nothing, Distributor}
    interior_field::Union{Nothing, ScalarField}
    interior_problem::Union{Nothing, LBVP}
    interior_bases::Union{Nothing, Tuple}

    # Source terms (user-defined functions)
    source_terms::Dict{String, Function}

    # Auxiliary fields for computations
    work_fields::Dict{String, ScalarField}

    # Parameters
    params::Dict{String, Float64}

    # State
    time::Float64
    iteration::Int
end

"""
    boundary_advection_diffusion_setup(;
        Lx, Ly,                          # Domain size
        Nx, Ny,                          # Resolution
        boundaries,                       # Vector{BoundarySpec}
        velocity_source,                  # VelocitySource
        diffusion=DiffusionSpec(),        # Diffusion specification
        interior_coupling=nothing,        # Optional interior coupling config
        mesh=(1,1),                       # MPI decomposition
        dtype=Float64                     # Data type
    )

Create a BoundaryAdvectionDiffusion system.

# Arguments
- `Lx, Ly`: Domain extent in x and y
- `Nx, Ny`: Grid resolution
- `boundaries`: Vector of BoundarySpec defining each boundary
- `velocity_source`: How to obtain advection velocity (VelocitySource subtype)
- `diffusion`: Diffusion specification (DiffusionSpec)
- `interior_coupling`: Optional named tuple for interior problem coupling
- `mesh`: MPI mesh decomposition
- `dtype`: Floating point type

# Returns
`BoundaryAdvectionDiffusion` struct ready for time-stepping

# Examples

## Pure SQG (self-derived velocity)
```julia
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π, Nx=256, Ny=256,
    boundaries=[BoundarySpec("theta", :z, 0.0)],
    velocity_source=SelfDerivedVelocity(inversion_exponent=-0.5),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
)
```

## Passive tracer with prescribed velocity
```julia
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π, Nx=128, Ny=128,
    boundaries=[BoundarySpec("concentration", :z, 0.0)],
    velocity_source=PrescribedVelocity(),
    diffusion=DiffusionSpec(type=:laplacian, coefficient=0.01)
)
# Set velocity externally
bad.velocities["concentration"]get_grid_data(.components[1]) .= u_x
bad.velocities["concentration"]get_grid_data(.components[2]) .= u_y
```

## With interior coupling (QG-like)
```julia
bad = boundary_advection_diffusion_setup(
    Lx=2π, Ly=2π, Nx=128, Ny=128,
    boundaries=[
        BoundarySpec("bottom", :z, 0.0),
        BoundarySpec("top", :z, 1.0)
    ],
    velocity_source=InteriorDerivedVelocity(:perp_grad),
    diffusion=DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5),
    interior_coupling=(
        Nz=32,
        H=1.0,
        equation="Δ(ψ) + S*∂z(∂z(ψ)) = q",
        params=Dict("S" => 0.01)
    )
)
```
"""
function boundary_advection_diffusion_setup(;
    Lx::Real, Ly::Real,
    Nx::Int, Ny::Int,
    boundaries::Vector{BoundarySpec},
    velocity_source::VelocitySource,
    diffusion::DiffusionSpec=DiffusionSpec(),
    interior_coupling::Union{Nothing, NamedTuple}=nothing,
    mesh::Tuple{Int,Int}=(1,1),
    dtype::Type=Float64
)
    # Create 2D coordinates and distributor for boundaries
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=mesh, dtype=dtype)

    # Create bases
    x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Float64(Lx)))
    y_basis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Float64(Ly)))
    bases = (x_basis, y_basis)

    # Create fields for each boundary
    fields = Dict{String, ScalarField}()
    velocities = Dict{String, VectorField}()
    work_fields = Dict{String, ScalarField}()

    for bspec in boundaries
        # Main field
        fields[bspec.name] = ScalarField(dist, bspec.field_name, bases, dtype)

        # Velocity field
        velocities[bspec.name] = VectorField(dist, coords, "u_$(bspec.name)", bases, dtype)

        # Work fields for RHS computation
        work_fields["rhs_$(bspec.name)"] = ScalarField(dist, "rhs_$(bspec.name)", bases, dtype)
        work_fields["dfdx_$(bspec.name)"] = ScalarField(dist, "dfdx_$(bspec.name)", bases, dtype)
        work_fields["dfdy_$(bspec.name)"] = ScalarField(dist, "dfdy_$(bspec.name)", bases, dtype)
    end

    # Handle interior coupling if specified
    interior_dist = nothing
    interior_field = nothing
    interior_problem = nothing
    interior_bases = nothing

    if interior_coupling !== nothing
        interior_dist, interior_field, interior_problem, interior_bases =
            setup_interior_coupling(coords, Lx, Ly, Nx, Ny, interior_coupling, mesh, dtype)
    end

    # Parameters dictionary
    params = Dict{String, Float64}(
        "Lx" => Float64(Lx),
        "Ly" => Float64(Ly),
        "κ" => diffusion.coefficient,
        "α" => diffusion.exponent
    )

    @info "BoundaryAdvectionDiffusion setup complete" n_boundaries=length(boundaries) velocity_source=typeof(velocity_source) diffusion=diffusion.type

    return BoundaryAdvectionDiffusion(
        dist, bases, coords,
        boundaries,
        fields,
        velocity_source, velocities,
        diffusion,
        interior_dist, interior_field, interior_problem, interior_bases,
        Dict{String, Function}(),  # source_terms
        work_fields,
        params,
        0.0,  # time
        0     # iteration
    )
end

"""
Set up interior coupling for boundary-interior problems.
"""
function setup_interior_coupling(
    coords_2d::CartesianCoordinates,
    Lx::Real, Ly::Real, Nx::Int, Ny::Int,
    config::NamedTuple,
    mesh_2d::Tuple{Int,Int},
    dtype::Type
)
    Nz = get(config, :Nz, 32)
    H = get(config, :H, 1.0)

    # Create 3D coordinates and distributor
    coords_3d = CartesianCoordinates("x", "y", "z")
    dist_3d = Distributor(coords_3d; mesh=(mesh_2d..., 1), dtype=dtype)

    # Create 3D bases
    x_basis = RealFourier(coords_3d["x"]; size=Nx, bounds=(0.0, Float64(Lx)))
    y_basis = RealFourier(coords_3d["y"]; size=Ny, bounds=(0.0, Float64(Ly)))
    z_basis = ChebyshevT(coords_3d["z"]; size=Nz, bounds=(0.0, Float64(H)))
    bases_3d = (x_basis, y_basis, z_basis)

    # Create interior field (e.g., streamfunction)
    interior_field = ScalarField(dist_3d, "ψ", bases_3d, dtype)

    # Create LBVP for interior
    interior_problem = LBVP([interior_field])

    # Add parameters
    if haskey(config, :params)
        for (k, v) in config.params
            interior_problem.parameters[string(k)] = Float64(v)
        end
    end

    # Add equation if provided
    if haskey(config, :equation)
        add_equation!(interior_problem, config.equation)
    end

    return dist_3d, interior_field, interior_problem, bases_3d
end

"""
    bad_compute_velocity!(bad::BoundaryAdvectionDiffusion)

Compute advection velocities based on the velocity source type.
"""
function bad_compute_velocity!(bad::BoundaryAdvectionDiffusion)
    if isa(bad.velocity_source, PrescribedVelocity)
        # Velocity already set by user, nothing to do
        return

    elseif isa(bad.velocity_source, SelfDerivedVelocity)
        # Derive velocity from boundary field itself
        src = bad.velocity_source

        for bspec in bad.boundary_specs
            field = bad.fields[bspec.name]

            # Apply fractional Laplacian inversion
            if src.inversion_exponent != 0.0
                inverted = evaluate_fractional_laplacian(
                    FractionalLaplacian(field, src.inversion_exponent), :g
                )
            else
                inverted = field
            end

            # Compute velocity
            if src.use_perp_grad
                vel = perp_grad(inverted)
            else
                vel = evaluate_operator(grad(inverted))
            end

            # Copy to stored velocity
            ensure_layout!(bad.velocities[bspec.name].components[1], :g)
            ensure_layout!(bad.velocities[bspec.name].components[2], :g)
            ensure_layout!(vel.components[1], :g)
            ensure_layout!(vel.components[2], :g)

            get_grid_data(bad.velocities[bspec.name].components[1]) .= get_grid_data(vel.components[1])
            get_grid_data(bad.velocities[bspec.name].components[2]) .= get_grid_data(vel.components[2])
        end

    elseif isa(bad.velocity_source, InteriorDerivedVelocity)
        # Derive velocity from interior field
        if bad.interior_field === nothing
            throw(ArgumentError("InteriorDerivedVelocity requires interior coupling"))
        end

        src = bad.velocity_source

        for bspec in bad.boundary_specs
            # Extract interior field at boundary position
            surface_field = extract_surface(bad.interior_field, bspec.dimension, bspec.position)

            # Apply operator to get velocity
            if src.operator == :perp_grad
                vel = perp_grad(surface_field)
            elseif src.operator == :grad
                vel = evaluate_operator(grad(surface_field))
            elseif src.operator == :curl
                vel = evaluate_operator(curl(surface_field))
            else
                throw(ArgumentError("Unknown velocity operator: $(src.operator)"))
            end

            # Copy to stored velocity
            ensure_layout!(bad.velocities[bspec.name].components[1], :g)
            ensure_layout!(bad.velocities[bspec.name].components[2], :g)
            ensure_layout!(vel.components[1], :g)
            ensure_layout!(vel.components[2], :g)

            get_grid_data(bad.velocities[bspec.name].components[1]) .= get_grid_data(vel.components[1])
            get_grid_data(bad.velocities[bspec.name].components[2]) .= get_grid_data(vel.components[2])
        end
    end
end

"""
    bad_compute_rhs!(bad::BoundaryAdvectionDiffusion, boundary_name::String)

Compute the RHS of the advection-diffusion equation for a specific boundary:
    RHS = -u·∇c + D(c) + S
"""
function bad_compute_rhs!(bad::BoundaryAdvectionDiffusion, boundary_name::String)
    field = bad.fields[boundary_name]
    velocity = bad.velocities[boundary_name]
    rhs = bad.work_fields["rhs_$(boundary_name)"]

    ensure_layout!(field, :g)
    ensure_layout!(rhs, :g)

    # Get coordinates
    coord_x = bad.coords[1]
    coord_y = bad.coords[2]

    # Compute gradients of field
    dfdx = evaluate_differentiate(Differentiate(field, coord_x, 1), :g)
    dfdy = evaluate_differentiate(Differentiate(field, coord_y, 1), :g)

    ensure_layout!(velocity.components[1], :g)
    ensure_layout!(velocity.components[2], :g)
    ensure_layout!(dfdx, :g)
    ensure_layout!(dfdy, :g)

    # Advection term: -u·∇c
    get_grid_data(rhs) .= -(get_grid_data(velocity.components[1]) .* get_grid_data(dfdx) .+
                    get_grid_data(velocity.components[2]) .* get_grid_data(dfdy))

    # Diffusion term (explicit part)
    if bad.diffusion.coefficient > 0 && !bad.diffusion.implicit
        diff_term = compute_diffusion_term(bad, field)
        get_grid_data(rhs) .+= get_grid_data(diff_term)
    end

    # Source term (if defined)
    if haskey(bad.source_terms, boundary_name)
        source_func = bad.source_terms[boundary_name]
        source_data = source_func(bad, boundary_name)
        get_grid_data(rhs) .+= source_data
    end

    return rhs
end

"""
Compute diffusion term based on diffusion specification.
"""
function compute_diffusion_term(bad::BoundaryAdvectionDiffusion, field::ScalarField)
    κ = bad.diffusion.coefficient
    α = bad.diffusion.exponent

    result = ScalarField(bad.dist, "diff_term", bad.bases, field.dtype)
    ensure_layout!(result, :g)

    if bad.diffusion.type == :none || κ == 0.0
        fill!(get_grid_data(result), 0.0)

    elseif bad.diffusion.type == :laplacian
        # Standard Laplacian: κΔc
        lap_field = evaluate_operator(lap(field))
        ensure_layout!(lap_field, :g)
        get_grid_data(result) .= κ .* get_grid_data(lap_field)

    elseif bad.diffusion.type == :fractional
        # Fractional Laplacian: κ(-Δ)^α c
        # Note: negative sign convention - dissipative for α > 0
        frac_lap = evaluate_fractional_laplacian(FractionalLaplacian(field, α), :g)
        ensure_layout!(frac_lap, :g)
        get_grid_data(result) .= -κ .* get_grid_data(frac_lap)

    elseif bad.diffusion.type == :hyperdiffusion
        # Hyperdiffusion: -κ(-Δ)^n c (n > 1)
        # Typically n = 2 gives -κΔ²c (biharmonic)
        frac_lap = evaluate_fractional_laplacian(FractionalLaplacian(field, α), :g)
        ensure_layout!(frac_lap, :g)
        get_grid_data(result) .= -κ .* get_grid_data(frac_lap)
    end

    return result
end

"""
    bad_step!(bad::BoundaryAdvectionDiffusion, dt::Real; timestepper=:RK4)

Advance the boundary advection-diffusion system by one timestep.

# Arguments
- `bad`: BoundaryAdvectionDiffusion system
- `dt`: Timestep size
- `timestepper`: Integration method (:Euler, :RK2, :RK4, :SSPRK3)
"""
function bad_step!(bad::BoundaryAdvectionDiffusion, dt::Real; timestepper::Symbol=:RK4)
    # First, solve interior problem if coupled
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end

    # Compute velocities
    bad_compute_velocity!(bad)

    # Time step based on method
    if timestepper == :Euler
        bad_step_euler!(bad, dt)
    elseif timestepper == :RK2
        bad_step_rk2!(bad, dt)
    elseif timestepper == :RK4
        bad_step_rk4!(bad, dt)
    elseif timestepper == :SSPRK3
        bad_step_ssprk3!(bad, dt)
    else
        throw(ArgumentError("Unknown timestepper: $timestepper"))
    end

    bad.time += dt
    bad.iteration += 1
end

"""
Solve interior problem (LBVP) with current boundary values as BCs.
"""
function bad_solve_interior!(bad::BoundaryAdvectionDiffusion)
    if bad.interior_problem === nothing
        return
    end

    # Update interior problem namespace with current boundary fields
    for bspec in bad.boundary_specs
        bad.interior_problem.namespace[bspec.field_name] = bad.fields[bspec.name]
    end

    # Solve LBVP
    solver = BoundaryValueSolver(bad.interior_problem)
    solve!(solver)
end

function bad_step_euler!(bad::BoundaryAdvectionDiffusion, dt::Real)
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        ensure_layout!(bad.fields[bspec.name], :g)
        get_grid_data(bad.fields[bspec.name]) .+= dt .* get_grid_data(rhs)
    end
end

function bad_step_rk2!(bad::BoundaryAdvectionDiffusion, dt::Real)
    # Save initial state
    saved_states = Dict{String, Array}()
    for bspec in bad.boundary_specs
        ensure_layout!(bad.fields[bspec.name], :g)
        saved_states[bspec.name] = copy(get_grid_data(bad.fields[bspec.name]))
    end

    # Stage 1: Euler to midpoint
    bad_step_euler!(bad, dt/2)

    # Recompute velocities at midpoint
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end
    bad_compute_velocity!(bad)

    # Stage 2: Full step using midpoint derivative
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        get_grid_data(bad.fields[bspec.name]) .= saved_states[bspec.name] .+ dt .* get_grid_data(rhs)
    end
end

function bad_step_rk4!(bad::BoundaryAdvectionDiffusion, dt::Real)
    # Save initial state
    saved_states = Dict{String, Array}()
    k1 = Dict{String, Array}()
    k2 = Dict{String, Array}()
    k3 = Dict{String, Array}()
    k4 = Dict{String, Array}()

    for bspec in bad.boundary_specs
        ensure_layout!(bad.fields[bspec.name], :g)
        saved_states[bspec.name] = copy(get_grid_data(bad.fields[bspec.name]))
    end

    # k1
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        k1[bspec.name] = copy(get_grid_data(rhs))
    end

    # k2
    for bspec in bad.boundary_specs
        get_grid_data(bad.fields[bspec.name]) .= saved_states[bspec.name] .+ (dt/2) .* k1[bspec.name]
    end
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end
    bad_compute_velocity!(bad)
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        k2[bspec.name] = copy(get_grid_data(rhs))
    end

    # k3
    for bspec in bad.boundary_specs
        get_grid_data(bad.fields[bspec.name]) .= saved_states[bspec.name] .+ (dt/2) .* k2[bspec.name]
    end
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end
    bad_compute_velocity!(bad)
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        k3[bspec.name] = copy(get_grid_data(rhs))
    end

    # k4
    for bspec in bad.boundary_specs
        get_grid_data(bad.fields[bspec.name]) .= saved_states[bspec.name] .+ dt .* k3[bspec.name]
    end
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end
    bad_compute_velocity!(bad)
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        k4[bspec.name] = copy(get_grid_data(rhs))
    end

    # Final update
    for bspec in bad.boundary_specs
        get_grid_data(bad.fields[bspec.name]) .= saved_states[bspec.name] .+
            (dt/6) .* (k1[bspec.name] .+ 2 .* k2[bspec.name] .+
                       2 .* k3[bspec.name] .+ k4[bspec.name])
    end
end

function bad_step_ssprk3!(bad::BoundaryAdvectionDiffusion, dt::Real)
    """Strong Stability Preserving RK3 (Shu-Osher form)"""

    # Save initial state
    saved_states = Dict{String, Array}()
    for bspec in bad.boundary_specs
        ensure_layout!(bad.fields[bspec.name], :g)
        saved_states[bspec.name] = copy(get_grid_data(bad.fields[bspec.name]))
    end

    # Stage 1: u^(1) = u^n + dt * L(u^n)
    for bspec in bad.boundary_specs
        rhs = bad_compute_rhs!(bad, bspec.name)
        get_grid_data(bad.fields[bspec.name]) .= saved_states[bspec.name] .+ dt .* get_grid_data(rhs)
    end

    # Update velocities for stage 2
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end
    bad_compute_velocity!(bad)

    # Stage 2: u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * L(u^(1)))
    stage1_states = Dict{String, Array}()
    for bspec in bad.boundary_specs
        stage1_states[bspec.name] = copy(get_grid_data(bad.fields[bspec.name]))
        rhs = bad_compute_rhs!(bad, bspec.name)
        get_grid_data(bad.fields[bspec.name]) .= 0.75 .* saved_states[bspec.name] .+
            0.25 .* (stage1_states[bspec.name] .+ dt .* get_grid_data(rhs))
    end

    # Update velocities for stage 3
    if bad.interior_problem !== nothing
        bad_solve_interior!(bad)
    end
    bad_compute_velocity!(bad)

    # Stage 3: u^(n+1) = 1/3 u^n + 2/3 (u^(2) + dt * L(u^(2)))
    for bspec in bad.boundary_specs
        stage2_state = copy(get_grid_data(bad.fields[bspec.name]))
        rhs = bad_compute_rhs!(bad, bspec.name)
        get_grid_data(bad.fields[bspec.name]) .= (1/3) .* saved_states[bspec.name] .+
            (2/3) .* (stage2_state .+ dt .* get_grid_data(rhs))
    end
end

"""
    bad_add_source!(bad::BoundaryAdvectionDiffusion, boundary_name::String, source_func::Function)

Add a source term to a boundary equation.

The source function should have signature:
    source_func(bad::BoundaryAdvectionDiffusion, boundary_name::String) -> Array

# Example
```julia
# Add a Gaussian forcing
function my_source(bad, name)
    x = get_grid(bad, :x)
    y = get_grid(bad, :y)
    return 0.1 * exp.(-((x .- π).^2 .+ (y .- π).^2) ./ 0.5)
end
bad_add_source!(bad, "surface", my_source)
```
"""
function bad_add_source!(bad::BoundaryAdvectionDiffusion, boundary_name::String, source_func::Function)
    bad.source_terms[boundary_name] = source_func
end

"""
    bad_energy(bad::BoundaryAdvectionDiffusion)

Compute total energy (L² norm) of all boundary fields.
"""
function bad_energy(bad::BoundaryAdvectionDiffusion)
    # CRITICAL FIX: Compute global mean correctly for MPI
    # Must reduce sums and counts separately, then divide
    total_sum_squared = 0.0
    total_count = 0

    for bspec in bad.boundary_specs
        field = bad.fields[bspec.name]
        ensure_layout!(field, :g)
        data = get_grid_data(field)
        total_sum_squared += sum(abs2.(data))
        total_count += length(data)
    end

    if MPI.Initialized()
        # Use field's communicator, not COMM_WORLD
        comm = bad.fields[first(keys(bad.fields))].dist.comm
        if MPI.Comm_size(comm) > 1
            # Reduce sum and count globally, then compute mean
            global_sum = MPI.Allreduce(total_sum_squared, MPI.SUM, comm)
            global_count = MPI.Allreduce(total_count, MPI.SUM, comm)
            return global_sum / global_count
        end
    end

    return total_sum_squared / total_count
end

"""
    bad_enstrophy(bad::BoundaryAdvectionDiffusion, boundary_name::String)

Compute enstrophy (squared gradient) of a boundary field.
"""
function bad_enstrophy(bad::BoundaryAdvectionDiffusion, boundary_name::String)
    field = bad.fields[boundary_name]

    coord_x = bad.coords[1]
    coord_y = bad.coords[2]

    dfdx = evaluate_differentiate(Differentiate(field, coord_x, 1), :g)
    dfdy = evaluate_differentiate(Differentiate(field, coord_y, 1), :g)

    ensure_layout!(dfdx, :g)
    ensure_layout!(dfdy, :g)

    # CRITICAL FIX: Compute global mean correctly for MPI
    # Must reduce sums and counts separately, then divide
    data_x = get_grid_data(dfdx)
    data_y = get_grid_data(dfdy)
    local_sum = sum(abs2.(data_x) .+ abs2.(data_y))
    local_count = length(data_x)

    if MPI.Initialized()
        # Use field's communicator, not COMM_WORLD
        comm = field.dist.comm
        if MPI.Comm_size(comm) > 1
            # Reduce sum and count globally, then compute mean
            global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
            global_count = MPI.Allreduce(local_count, MPI.SUM, comm)
            return global_sum / global_count
        end
    end

    return local_sum / local_count
end

"""
    bad_max_velocity(bad::BoundaryAdvectionDiffusion)

Compute maximum velocity magnitude across all boundaries.
"""
function bad_max_velocity(bad::BoundaryAdvectionDiffusion)
    max_vel = 0.0

    for bspec in bad.boundary_specs
        vel = bad.velocities[bspec.name]
        ensure_layout!(vel.components[1], :g)
        ensure_layout!(vel.components[2], :g)

        vel_mag = sqrt.(get_grid_data(vel.components[1]).^2 .+ get_grid_data(vel.components[2]).^2)
        max_vel = max(max_vel, maximum(vel_mag))
    end

    if MPI.Initialized()
        # Use velocity field's communicator, not COMM_WORLD
        first_bspec = first(bad.boundary_specs)
        comm = bad.velocities[first_bspec.name].components[1].dist.comm
        max_vel = MPI.Allreduce(max_vel, MPI.MAX, comm)
    end

    return max_vel
end

"""
    bad_cfl_dt(bad::BoundaryAdvectionDiffusion; safety=0.5)

Compute CFL-limited timestep.
"""
function bad_cfl_dt(bad::BoundaryAdvectionDiffusion; safety::Float64=0.5)
    max_vel = bad_max_velocity(bad)

    if max_vel == 0.0
        return Inf
    end

    # Grid spacing
    Lx = bad.params["Lx"]
    Ly = bad.params["Ly"]
    Nx = bad.bases[1].meta.size
    Ny = bad.bases[2].meta.size

    dx = Lx / Nx
    dy = Ly / Ny
    dmin = min(dx, dy)

    return safety * dmin / max_vel
end

"""
    qg_energy(qg::QGSystem)

Compute total QG energy: E = ∫∫∫ (|∇ψ|² + S|∂ψ/∂z|²) dx dy dz

For surface-only dynamics (SQG limit with q=0), this reduces to
surface integrals involving θ.
"""
function qg_energy(qg::QGSystem)
    # Ensure streamfunction is up to date
    qg_invert!(qg)
    ensure_layout!(qg.ψ, :g)

    coords = qg.ψ.dist.coordsys
    S = (qg.f0 / qg.N)^2

    # Compute gradient components
    coord_x = coords[1]
    coord_y = coords[2]
    coord_z = coords[3]

    dψ_dx = evaluate_differentiate(Differentiate(qg.ψ, coord_x, 1), :g)
    dψ_dy = evaluate_differentiate(Differentiate(qg.ψ, coord_y, 1), :g)
    dψ_dz = evaluate_differentiate(Differentiate(qg.ψ, coord_z, 1), :g)

    # Create energy density field for proper quadrature integration
    energy_field = ScalarField(qg.ψ.dist, "energy_density", qg.ψ.bases, qg.ψ.dtype)
    ensure_layout!(energy_field, :g)
    get_grid_data(energy_field) .= dψget_grid_data(_dx).^2 .+ dψget_grid_data(_dy).^2 .+ S .* dψget_grid_data(_dz).^2

    # Integrate using basis-specific quadrature weights and compute volume average
    # integrate() uses proper weights: uniform for Fourier, Gauss-Legendre for Legendre,
    # Clenshaw-Curtis for Chebyshev
    total_integral = integrate(energy_field)

    # Compute domain volume for averaging
    domain_volume = 1.0
    for basis in qg.ψ.bases
        domain_volume *= (basis.bounds[2] - basis.bounds[1])
    end

    total_energy = total_integral / domain_volume

    if MPI.Initialized() && qg.ψ.dist.size > 1
        # Use field's communicator, not COMM_WORLD
        total_energy = MPI.Allreduce(total_energy, MPI.SUM, qg.ψ.dist.comm)
    end

    return total_energy
end

# ============================================================================
# CFL pretty printing
# ============================================================================

function Base.show(io::IO, cfl::CFL)
    nvel = length(cfl.velocities)
    print(io, "CFL(dt=$(round(cfl.current_dt; sigdigits=3)), safety=$(cfl.safety), $nvel velocities)")
end

function Base.show(io::IO, ::MIME"text/plain", cfl::CFL)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("CFL Controller"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    println(io, _box_text(@sprintf("Current dt:    %.4e", cfl.current_dt)))
    println(io, _box_text(@sprintf("Safety:        %.2f", cfl.safety)))
    println(io, _box_text(@sprintf("Max dt:        %.4e", cfl.max_dt)))
    println(io, _box_text("Cadence:       $(cfl.cadence)"))
    println(io, _box_text(@sprintf("Max change:    %.2f", cfl.max_change)))
    println(io, _box_text(@sprintf("Min change:    %.2f", cfl.min_change)))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    nvel = length(cfl.velocities)
    println(io, _box_text("Velocities:    $nvel registered"))
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# Exports
# ============================================================================

# CFL adaptive timestepping
export CFL, add_velocity!, compute_timestep

# Flow diagnostics
export reynolds_number, kinetic_energy, total_kinetic_energy
export enstrophy, total_enstrophy
export energy_dissipation_rate, vorticity_transport

# Energy spectrum analysis
export energy_spectrum, WavenumberInfo
export validate_fourier_bases, get_wavenumber_info
export calculate_wavenumber_grids, calculate_k_magnitudes, calculate_kmax
export calculate_radial_energy_spectrum, calculate_spectral_kinetic_energy
export calculate_full_energy_spectrum

# Power spectra for scalar and vector fields
export power_spectrum, enstrophy_spectrum, scalar_spectrum
export SpectrumBinning, LinearBinning, LogBinning, CustomBinning
export calculate_spectral_power, calculate_radial_power_spectrum, calculate_full_power_spectrum
export calculate_radial_vector_spectrum, calculate_full_vector_spectrum
export validate_fourier_bases_scalar, get_wavenumber_info_scalar

# Domain utilities
export get_domain_size, get_domain_bounds, get_fourier_shape

# Turbulence statistics
export turbulence_statistics

# Streamfunction utilities
export streamfunction, validate_streamfunction
export get_fourier_basis_info, all_periodic_fourier
export streamfunction_spectral_invert, get_2d_wavenumber_grids
export streamfunction_bvp_solve, streamfunction_jacobi_solve
export apply_streamfunction_bc!

# Velocity utilities
export velocity_divergence, perp_grad, ∇⊥

# Surface Quasi-Geostrophic (SQG) system
export sqg_streamfunction, sqg_velocity, sqg_problem_setup

# Quasi-Geostrophic (QG) system
export QGSystem, qg_system_setup
export qg_invert!, qg_surface_velocity!, extract_surface
export qg_advection_rhs, qg_step!
export qg_step_euler!, qg_step_rk2!, qg_step_rk4!
export qg_energy

# Boundary advection-diffusion system types
export VelocitySource, PrescribedVelocity, InteriorDerivedVelocity, SelfDerivedVelocity
export DiffusionSpec, BoundarySpec
export BoundaryAdvectionDiffusion

# Boundary advection-diffusion system functions
export boundary_advection_diffusion_setup, setup_interior_coupling
export bad_compute_velocity!, bad_compute_rhs!, compute_diffusion_term
export bad_step!, bad_solve_interior!
export bad_step_euler!, bad_step_rk2!, bad_step_rk4!, bad_step_ssprk3!
export bad_add_source!, bad_energy, bad_enstrophy
export bad_max_velocity, bad_cfl_dt
