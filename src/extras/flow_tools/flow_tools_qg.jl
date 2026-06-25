# Quasi-geostrophic helpers used by flow analysis tools.

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
get_grid_data(qg.θ_bot) .= initial_buoyancy_bottom
get_grid_data(qg.θ_top) .= initial_buoyancy_top

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
    interior_bvp.parameters["S"] = (f0 / N)^2  # Burger number squared

    # QG elliptic operator: ∇_h²ψ + S·∂²ψ/∂z² = q. Use the HORIZONTAL Laplacian
    # explicitly: `Δ` is the full 3D Laplacian (∂xx+∂yy+∂zz), so `Δ(ψ) + S*∂z(∂z(ψ))`
    # would double-count ∂²/∂z² and give a vertical coefficient (1+S) instead of S.
    add_equation!(interior_bvp, "∂x(∂x(ψ)) + ∂y(∂y(ψ)) + S*∂z(∂z(ψ)) = q")

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
        # Dissipation must DECAY high-k: ∂tθ = -κ(-Δ)^α θ. fraclap = (-Δ)^α has positive
        # eigenvalues |k|^{2α}, so the term enters with a PLUS sign on the LHS (moving it
        # to the RHS gives -κ(-Δ)^α θ). A minus sign here would be anti-diffusion (blow-up).
        add_equation!(surface_ivp_bot, "∂t(θ_bot) + κ*fraclap(θ_bot, α) = 0")
        add_equation!(surface_ivp_top, "∂t(θ_top) + κ*fraclap(θ_top, α) = 0")
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
    get_grid_data(rhs_bot) .= -(get_grid_data(qg.u_bot.components[1]) .* get_grid_data(dθ_dx_bot) .+
                        get_grid_data(qg.u_bot.components[2]) .* get_grid_data(dθ_dy_bot))

    # Top surface
    ensure_layout!(qg.θ_top, :g)
    ensure_layout!(qg.u_top.components[1], :g)
    ensure_layout!(qg.u_top.components[2], :g)

    dθ_dx_top = evaluate_differentiate(Differentiate(qg.θ_top, coord_x, 1), :g)
    dθ_dy_top = evaluate_differentiate(Differentiate(qg.θ_top, coord_y, 1), :g)

    rhs_top = ScalarField(qg.dist_2d_top, "rhs_top", qg.θ_top.bases, Float64)
    ensure_layout!(rhs_top, :g)
    get_grid_data(rhs_top) .= -(get_grid_data(qg.u_top.components[1]) .* get_grid_data(dθ_dx_top) .+
                        get_grid_data(qg.u_top.components[2]) .* get_grid_data(dθ_dy_top))

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
        get_grid_data(rhs_bot) .-= qg.κ .* get_grid_data(frac_lap_bot)
        get_grid_data(rhs_top) .-= qg.κ .* get_grid_data(frac_lap_top)
    end

    # 5. Euler step
    get_grid_data(qg.θ_bot) .+= dt .* get_grid_data(rhs_bot)
    get_grid_data(qg.θ_top) .+= dt .* get_grid_data(rhs_top)
end

function qg_step_rk2!(qg::QGSystem, dt::Real)
    # Save initial state
    θ_bot_0 = copy(get_grid_data(qg.θ_bot))
    θ_top_0 = copy(get_grid_data(qg.θ_top))

    # Stage 1: Euler step to midpoint
    qg_step_euler!(qg, dt / 2)

    # Stage 2: Full step from original using midpoint derivative
    qg_invert!(qg)
    qg_surface_velocity!(qg)
    rhs_bot, rhs_top = qg_advection_rhs(qg)

    if qg.κ > 0
        frac_lap_bot = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_bot, qg.α), :g)
        frac_lap_top = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_top, qg.α), :g)
        get_grid_data(rhs_bot) .-= qg.κ .* get_grid_data(frac_lap_bot)
        get_grid_data(rhs_top) .-= qg.κ .* get_grid_data(frac_lap_top)
    end

    get_grid_data(qg.θ_bot) .= θ_bot_0 .+ dt .* get_grid_data(rhs_bot)
    get_grid_data(qg.θ_top) .= θ_top_0 .+ dt .* get_grid_data(rhs_top)
end

function qg_step_rk4!(qg::QGSystem, dt::Real)
    # Save initial state
    θ_bot_0 = copy(get_grid_data(qg.θ_bot))
    θ_top_0 = copy(get_grid_data(qg.θ_top))

    function compute_rhs()
        qg_invert!(qg)
        qg_surface_velocity!(qg)
        rhs_bot, rhs_top = qg_advection_rhs(qg)

        if qg.κ > 0
            frac_lap_bot = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_bot, qg.α), :g)
            frac_lap_top = evaluate_fractional_laplacian(FractionalLaplacian(qg.θ_top, qg.α), :g)
            get_grid_data(rhs_bot) .-= qg.κ .* get_grid_data(frac_lap_bot)
            get_grid_data(rhs_top) .-= qg.κ .* get_grid_data(frac_lap_top)
        end

        return copy(get_grid_data(rhs_bot)), copy(get_grid_data(rhs_top))
    end

    # k1
    k1_bot, k1_top = compute_rhs()

    # k2
    get_grid_data(qg.θ_bot) .= θ_bot_0 .+ (dt / 2) .* k1_bot
    get_grid_data(qg.θ_top) .= θ_top_0 .+ (dt / 2) .* k1_top
    k2_bot, k2_top = compute_rhs()

    # k3
    get_grid_data(qg.θ_bot) .= θ_bot_0 .+ (dt / 2) .* k2_bot
    get_grid_data(qg.θ_top) .= θ_top_0 .+ (dt / 2) .* k2_top
    k3_bot, k3_top = compute_rhs()

    # k4
    get_grid_data(qg.θ_bot) .= θ_bot_0 .+ dt .* k3_bot
    get_grid_data(qg.θ_top) .= θ_top_0 .+ dt .* k3_top
    k4_bot, k4_top = compute_rhs()

    # Final update
    get_grid_data(qg.θ_bot) .= θ_bot_0 .+ (dt / 6) .* (k1_bot .+ 2 .* k2_bot .+ 2 .* k3_bot .+ k4_bot)
    get_grid_data(qg.θ_top) .= θ_top_0 .+ (dt / 6) .* (k1_top .+ 2 .* k2_top .+ 2 .* k3_top .+ k4_top)
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
    get_grid_data(energy_field) .= get_grid_data(dψ_dx).^2 .+ get_grid_data(dψ_dy).^2 .+ S .* get_grid_data(dψ_dz).^2

    # Integrate using basis-specific quadrature weights and compute volume average
    # integrate() uses proper weights: uniform for Fourier, Gauss-Legendre for Legendre,
    # Clenshaw-Curtis for Chebyshev
    total_integral = integrate(energy_field)

    # Compute domain volume for averaging
    domain_volume = 1.0
    for basis in qg.ψ.bases
        domain_volume *= (basis.bounds[2] - basis.bounds[1])
    end

    # `integrate()` already performs the global MPI.Allreduce internally
    # (operations_integrate.jl: _integrate_full_distributed), so total_integral is
    # the COMPLETE global integral, identical on every rank. A second
    # MPI.Allreduce(SUM) here would sum that replicated scalar across all ranks and
    # report nprocs × the true energy (round-4 audit 2026-06-23). Match the sibling
    # diagnostics total_kinetic_energy / total_enstrophy, which integrate() directly.
    total_energy = total_integral / domain_volume

    return total_energy
end
