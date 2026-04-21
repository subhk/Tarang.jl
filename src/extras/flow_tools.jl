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
include("flow_tools/flow_tools_streamfunction.jl")
include("flow_tools/flow_tools_boundary_advection.jl")

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
