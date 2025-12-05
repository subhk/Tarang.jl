"""
Flow analysis tools

Translated from dedalus/extras/flow_tools.py

Note: GlobalArrayReducer, reduce_scalar, global_min, global_max, and global_mean
are defined in core/evaluator.jl and reused here.
"""

using Statistics
using LinearAlgebra
using MPI

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

function add_velocity!(cfl::CFL, velocity::VectorField, weight::Float64=1.0)
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
            vel_data = abs.(component.data_g)
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
    
    # Calculate velocity magnitude
    vel_magnitude_squared = zeros(size(velocity.components[1].data_g))
    
    for component in velocity.components
        ensure_layout!(component, :g)
        vel_magnitude_squared .+= abs2.(component.data_g)
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
    
    # Calculate |u|²
    vel_squared = zeros(size(velocity.components[1].data_g))
    
    for component in velocity.components
        ensure_layout!(component, :g)
        vel_squared .+= abs2.(component.data_g)
    end
    
    ke_field = ScalarField(velocity.dist, "kinetic_energy", velocity.bases, velocity.dtype)
    ensure_layout!(ke_field, :g)
    ke_field.data_g .= 0.5 * density .* vel_squared
    
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
    
    enstrophy_field.data_g .= 0.5 .* abs2.(vorticity_field.data_g)
    
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
    
    # Calculate strain rate tensor components
    strain_rate_squared = zeros(size(velocity.components[1].data_g))
    
    for i in 1:velocity.coordsys.dim
        for j in 1:velocity.coordsys.dim
            # ∂u_i/∂x_j
            coord = velocity.coordsys[j]
            du_i_dx_j = evaluate_differentiate(
                Differentiate(velocity.components[i], coord, 1), :g
            )
            ensure_layout!(du_i_dx_j, :g)
            
            if i == j
                # Diagonal terms
                strain_rate_squared .+= abs2.(du_i_dx_j.data_g)
            else
                # Off-diagonal terms contribute twice (symmetry)
                strain_rate_squared .+= 2 .* abs2.(du_i_dx_j.data_g)
            end
        end
    end
    
    dissipation_field = ScalarField(velocity.dist, "dissipation", velocity.bases, velocity.dtype)
    ensure_layout!(dissipation_field, :g)
    dissipation_field.data_g .= viscosity .* strain_rate_squared
    
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
    fill!(advection.data_g, 0.0)
    
    for i in 1:2
        ensure_layout!(velocity.components[i], :g)
        ensure_layout!(grad_vorticity.components[i], :g)
        advection.data_g .+= velocity.components[i].data_g .* grad_vorticity.components[i].data_g
    end
    
    # Diffusion term: ν∇²ω
    lap_vorticity = lap(vorticity)
    diffusion = ScalarField(velocity.dist, "vorticity_diffusion", velocity.bases, velocity.dtype)
    ensure_layout!(lap_vorticity, :g)
    ensure_layout!(diffusion, :g)
    diffusion.data_g .= viscosity .* lap_vorticity.data_g
    
    return (advection=advection, diffusion=diffusion)
end

# Energy spectra (for Fourier bases)
function energy_spectrum(velocity::VectorField; max_wavenumber::Union{Int,Nothing}=nothing, radial_average::Bool=true)
    """
    Calculate kinetic energy spectrum E(k) from velocity field.
    
    Computes the kinetic energy spectrum E(k) = ∫∫ |û_i(k)|² dA(k) where the integral
    is over a spherical (3D) or circular (2D) shell of radius k in wavenumber space.
    
    Based on standard spectral turbulence analysis methods and follows dedalus patterns
    for spectral field processing with PencilArrays/PencilFFTs integration.
    
    Args:
        velocity: VectorField with Fourier bases
        max_wavenumber: Maximum wavenumber for spectrum (default: Nyquist limit)  
        radial_average: Whether to perform radial averaging in wavenumber space
    
    Returns:
        Dictionary with wavenumber bins as keys and energy E(k) as values
    """
    
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
        return calculate_radial_energy_spectrum(velocity, wavenumber_info, max_wavenumber)
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
    """
    
    # Get domain information
    domain = velocity.domain
    domain_size = get_domain_size(domain)
    
    # Get local spectral shape
    fourier_shape = get_fourier_shape(velocity, fourier_axes)
    
    # Calculate wavenumber grids
    kx_grid, ky_grid, kz_grid = calculate_wavenumber_grids(
        fourier_bases, fourier_shape, domain_size
    )
    
    # Calculate wavenumber magnitudes
    k_magnitudes = calculate_k_magnitudes(kx_grid, ky_grid, kz_grid)
    
    # Determine maximum wavenumber (Nyquist limit)
    kmax = calculate_kmax(fourier_shape)
    
    return WavenumberInfo(kmax, k_magnitudes, kx_grid, ky_grid, kz_grid, domain_size, fourier_shape)
end

function calculate_wavenumber_grids(fourier_bases::Vector, fourier_shape::Tuple{Vararg{Int}}, domain_size::Tuple{Vararg{Float64}})
    """
    Calculate wavenumber grids for each Fourier dimension.
    Handles RealFourier vs ComplexFourier indexing properly.
    """
    
    kx_grid = nothing
    ky_grid = nothing  
    kz_grid = nothing
    
    # X-direction (first Fourier basis)
    if length(fourier_bases) >= 1
        basis_x = fourier_bases[1]
        nx = fourier_shape[1]
        Lx = domain_size[1]
        
        if isa(basis_x, RealFourier)
            # RealFourier: [0, 1, 2, ..., nx/2]
            kx_1d = 2π/Lx * (0:(nx÷2))
        else  # ComplexFourier
            # ComplexFourier: [0, 1, ..., nx/2-1, -nx/2, ..., -1]
            kx_1d = 2π/Lx * fftshift(-nx÷2:(nx÷2-1))
        end
        kx_grid = reshape(kx_1d, length(kx_1d), 1)
    end
    
    # Y-direction (second Fourier basis)
    if length(fourier_bases) >= 2
        basis_y = fourier_bases[2]
        ny = fourier_shape[2]
        Ly = domain_size[2]
        
        if isa(basis_y, RealFourier)
            ky_1d = 2π/Ly * (0:(ny÷2))
        else  # ComplexFourier
            ky_1d = 2π/Ly * fftshift(-ny÷2:(ny÷2-1))
        end
        ky_grid = reshape(ky_1d, 1, length(ky_1d))
    end
    
    # Z-direction (third Fourier basis, if present)
    if length(fourier_bases) >= 3
        basis_z = fourier_bases[3]
        nz = fourier_shape[3]
        Lz = domain_size[3]
        
        if isa(basis_z, RealFourier)
            kz_1d = 2π/Lz * (0:(nz÷2))
        else  # ComplexFourier
            kz_1d = 2π/Lz * fftshift(-nz÷2:(nz÷2-1))
        end
        kz_grid = reshape(kz_1d, 1, 1, length(kz_1d))
    end
    
    return kx_grid, ky_grid, kz_grid
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

function calculate_radial_energy_spectrum(velocity::VectorField, wavenumber_info::WavenumberInfo, max_wavenumber::Int)
    """
    Calculate radially-averaged energy spectrum E(k).
    Bins energy by wavenumber magnitude and performs proper averaging.
    """
    
    # Initialize spectrum bins
    spectrum = Dict{Int, Float64}()
    bin_counts = Dict{Int, Int}()
    
    for k in 1:max_wavenumber
        spectrum[k] = 0.0
        bin_counts[k] = 0
    end
    
    # Calculate kinetic energy density in spectral space
    ke_spectral = calculate_spectral_kinetic_energy(velocity)
    
    # Bin energy by wavenumber magnitude
    k_magnitudes = wavenumber_info.k_magnitudes
    
    for idx in CartesianIndices(ke_spectral)
        k_mag = k_magnitudes[idx]
        k_bin = round(Int, k_mag)
        
        if 1 <= k_bin <= max_wavenumber
            spectrum[k_bin] += ke_spectral[idx]
            bin_counts[k_bin] += 1
        end
    end
    
    # Perform MPI reduction across processes
    if MPI.Initialized()
        for k in 1:max_wavenumber
            spectrum[k] = MPI.Allreduce(spectrum[k], MPI.SUM, MPI.COMM_WORLD)
            bin_counts[k] = MPI.Allreduce(bin_counts[k], MPI.SUM, MPI.COMM_WORLD)
        end
    end
    
    # Normalize by bin counts and apply proper scaling
    for k in 1:max_wavenumber
        if bin_counts[k] > 0
            spectrum[k] /= bin_counts[k]
            # Apply 2D/3D shell normalization
            if wavenumber_info.kz_grid !== nothing
                # 3D: multiply by 4πk² (spherical shell area)
                spectrum[k] *= 4π * k^2
            else
                # 2D: multiply by 2πk (circular shell circumference)  
                spectrum[k] *= 2π * k
            end
        end
    end
    
    return spectrum
end

function calculate_spectral_kinetic_energy(velocity::VectorField)
    """
    Calculate kinetic energy density in spectral space.
    Returns |û|² + |v̂|² + |ŵ|² with proper normalization.
    """
    
    # Get first component to determine array size
    first_component = velocity.components[1]
    ensure_layout!(first_component, :c)
    ke_spectral = zeros(Float64, size(first_component.data_c))
    
    # Sum |û_i|² over all velocity components
    for component in velocity.components
        ensure_layout!(component, :c)
        ke_spectral .+= abs2.(component.data_c)
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
    """
    
    ke_spectral = calculate_spectral_kinetic_energy(velocity)
    
    # Perform MPI reduction
    if MPI.Initialized()
        ke_spectral = MPI.Allreduce(ke_spectral, MPI.SUM, MPI.COMM_WORLD)
    end
    
    # Return as dictionary with wavenumber indices as keys
    full_spectrum = Dict{Tuple{Vararg{Int}}, Float64}()
    
    for idx in CartesianIndices(ke_spectral)
        kx_idx, ky_idx = idx.I[1], idx.I[2]
        if length(idx.I) >= 3
            kz_idx = idx.I[3]
            full_spectrum[(kx_idx, ky_idx, kz_idx)] = ke_spectral[idx]
        else
            full_spectrum[(kx_idx, ky_idx)] = ke_spectral[idx]
        end
    end
    
    return full_spectrum
end

# Helper functions for domain and shape extraction (these would need to be implemented)
function get_domain_size(domain)
    """Extract physical domain size from domain object"""
    # Placeholder - would extract from domain.bases
    return (2π, 2π, 2π)  # Default periodic domain
end

function get_fourier_shape(velocity::VectorField, fourier_axes::Vector{Int})
    """Extract local Fourier shape from velocity field"""
    first_component = velocity.components[1] 
    ensure_layout!(first_component, :c)
    return size(first_component.data_c)
end

# Turbulence statistics
function turbulence_statistics(velocity::VectorField)
    """Calculate basic turbulence statistics"""
    
    stats = Dict{String, Float64}()
    
    # RMS velocity
    vel_rms_squared = 0.0
    for component in velocity.components
        ensure_layout!(component, :g)
        vel_rms_squared += mean(abs2.(component.data_g))
    end
    stats["velocity_rms"] = sqrt(vel_rms_squared)
    
    # Maximum velocity
    max_vel = 0.0
    for component in velocity.components
        ensure_layout!(component, :g)
        max_vel = max(max_vel, maximum(abs.(component.data_g)))
    end
    
    if MPI.Initialized()
        max_vel = MPI.Allreduce(max_vel, MPI.MAX, MPI.COMM_WORLD)
    end
    stats["velocity_max"] = max_vel
    
    # Calculate derived quantities
    if velocity.coordsys.dim == 2
        # Vorticity RMS for 2D flows
        vort = evaluate_operator(curl(velocity))
        ensure_layout!(vort, :g)
        stats["vorticity_rms"] = sqrt(mean(abs2.(vort.data_g)))
        
        # Maximum vorticity
        max_vort = maximum(abs.(vort.data_g))
        if MPI.Initialized()
            max_vort = MPI.Allreduce(max_vort, MPI.MAX, MPI.COMM_WORLD)
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
    Based on dedalus LBVP patterns for Poisson equation solving.
    
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
    Based on dedalus spectral Poisson inversion patterns.
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
    vorticity_spec = vorticity.data_c
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
    streamfunction_field.data_c .= streamfunction_spec
    
    # Apply MPI reduction if needed
    if MPI.Initialized()
        MPI.Allreduce!(streamfunction_field.data_c, MPI.SUM, MPI.COMM_WORLD)
    end
    
    return streamfunction_field
end

function get_2d_wavenumber_grids(field::ScalarField)
    """
    Get 2D wavenumber grids for spectral operations.
    Returns properly scaled kx, ky grids.
    """
    
    # Get domain size (this would need to be extracted from field.domain)
    # For now, assume 2π periodic domain
    Lx = 2π
    Ly = 2π
    
    # Get spectral shape
    ensure_layout!(field, :c)
    nx, ny = size(field.data_c)
    
    # Generate wavenumber grids based on basis types
    bases = field.bases
    
    # X-direction wavenumbers
    if isa(bases[1], RealFourier)
        kx_1d = 2π/Lx * (0:(nx-1))
    else  # ComplexFourier
        kx_1d = 2π/Lx * fftshift(-nx÷2:(nx÷2-1))
    end
    
    # Y-direction wavenumbers  
    if isa(bases[2], RealFourier)
        ky_1d = 2π/Ly * (0:(ny-1))
    else  # ComplexFourier
        ky_1d = 2π/Ly * fftshift(-ny÷2:(ny÷2-1))
    end
    
    # Create 2D grids
    kx_grid = reshape(kx_1d, length(kx_1d), 1)
    ky_grid = reshape(ky_1d, 1, length(ky_1d))
    
    return kx_grid, ky_grid
end

function streamfunction_bvp_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool=true)
    """
    Solve streamfunction BVP for bounded/mixed domains.
    
    Based on dedalus LBVP (Linear Boundary Value Problem) approach:
    ∇²ψ = ω with appropriate boundary conditions.
    
    This is a framework - full implementation would require BVP solver infrastructure.
    """
    
    # For now, implement a simplified approach using iterative methods
    # A full implementation would use dedalus-style tau method with LBVP
    
    streamfunction_field = ScalarField(vorticity.dist, "streamfunction", vorticity.bases, vorticity.dtype)
    ensure_layout!(streamfunction_field, :g)
    
    # Simplified Jacobi iteration for demonstration
    # In practice, would use sophisticated LBVP solver
    psi = streamfunction_jacobi_solve(vorticity, bc_type, apply_gauge)
    
    streamfunction_field.data_g .= psi
    return streamfunction_field
end

function streamfunction_jacobi_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool; 
                                   max_iter::Int=1000, tolerance::Float64=1e-8)
    """
    Simplified Jacobi iteration solver for ∇²ψ = ω.
    
    This is a basic implementation - a full dedalus-style implementation 
    would use tau methods and LBVP infrastructure.
    """
    
    ensure_layout!(vorticity, :g)
    omega = vorticity.data_g
    nx, ny = size(omega)
    
    # Initialize streamfunction
    psi = zeros(Float64, nx, ny)
    psi_new = similar(psi)
    
    # Get grid spacing (would extract from domain)
    dx = 2π / nx  # Assuming 2π domain
    dy = 2π / ny
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
    
    # MPI reduction if needed
    if MPI.Initialized()
        MPI.Allreduce!(psi, MPI.SUM, MPI.COMM_WORLD)
    end
    
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
    u_error = maximum(abs.(velocity.components[1].data_g - dpsi_dy.data_g))
    
    # Check v = -∂ψ/∂x
    ensure_layout!(velocity.components[2], :g)
    ensure_layout!(dpsi_dx, :g)  
    v_error = maximum(abs.(velocity.components[2].data_g + dpsi_dx.data_g))
    
    # MPI reduction for global error
    if MPI.Initialized()
        u_error = MPI.Allreduce(u_error, MPI.MAX, MPI.COMM_WORLD)
        v_error = MPI.Allreduce(v_error, MPI.MAX, MPI.COMM_WORLD)
    end
    
    max_error = max(u_error, v_error)
    is_valid = max_error < tolerance
    
    return (valid=is_valid, u_error=u_error, v_error=v_error, max_error=max_error)
end

function velocity_divergence(velocity::VectorField)
    """Calculate velocity divergence ∇·u"""
    
    divergence_op = div(velocity)
    return evaluate_operator(divergence_op)
end
