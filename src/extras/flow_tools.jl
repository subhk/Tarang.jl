"""
Flow analysis tools

Note: GlobalArrayReducer, reduce_scalar, global_min, global_max, and global_mean
are defined in core/evaluator.jl and reused here.
"""

using Statistics
using LinearAlgebra
using MPI
using FFTW: fftshift

include("flow_tools/flow_tools_cfl.jl")
include("flow_tools/flow_tools_diagnostics.jl")
include("flow_tools/flow_tools_spectrum_types.jl")
include("flow_tools/flow_tools_domain_utils.jl")
include("flow_tools/flow_tools_spectra.jl")

# Turbulence statistics
"""Calculate basic turbulence statistics"""
function turbulence_statistics(velocity::VectorField)

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
        ScalarField containing stream"""
function streamfunction(velocity::VectorField; boundary_condition::Symbol=:periodic, gauge_condition::Bool=true)
    
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

"""Extract information about Fourier vs non-Fourier bases"""
function get_fourier_basis_info(bases::Vector)
    fourier_info = []
    
    for (i, basis) in enumerate(bases)
        is_fourier = isa(basis, RealFourier) || isa(basis, ComplexFourier)
        push!(fourier_info, (index=i, is_fourier=is_fourier, basis=basis))
    end
    
    return fourier_info
end

"""Check if all bases are periodic Fourier"""
function all_periodic_fourier(fourier_info::Vector)
    return all(info.is_fourier for info in fourier_info)
end

"""
    Direct spectral inversion for periodic domains: ψ̂(k) = -ω̂(k)/|k|²
    Based on Tarang spectral Poisson inversion patterns.
    """
function streamfunction_spectral_invert(vorticity::ScalarField, apply_gauge::Bool=true)
    
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

"""
    Get 2D wavenumber grids for spectral operations.
    Returns properly scaled kx, ky grids.

    CRITICAL: For MPI/PencilArrays, uses global wavenumber indices with proper offsets.
    Each rank generates wavenumbers corresponding to its local portion of global spectrum.
    """
function get_2d_wavenumber_grids(field::ScalarField)

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

"""
    Solve streamfunction BVP for bounded/mixed domains.

    Solves ∇²ψ = ω with appropriate boundary conditions using Jacobi iteration.
    This is an iterative solver suitable for domains with physical boundaries.
    """
function streamfunction_bvp_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool=true)

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

"""Apply boundary conditions to streamfunction"""
function apply_streamfunction_bc!(psi::Array{Float64,2}, bc_type::Symbol)
    
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

"""
    Validate streamfunction by checking if it generates the correct velocity field.
    
    For 2D incompressible flow: u = ∂ψ/∂y, v = -∂ψ/∂x
    """
function validate_streamfunction(velocity::VectorField, streamfunction::ScalarField; tolerance::Float64=1e-6)
    
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

"""Calculate velocity divergence ∇·u"""
function velocity_divergence(velocity::VectorField)

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

"""Strong Stability Preserving RK3 (Shu-Osher form)"""
function bad_step_ssprk3!(bad::BoundaryAdvectionDiffusion, dt::Real)

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
