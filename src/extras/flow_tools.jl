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
