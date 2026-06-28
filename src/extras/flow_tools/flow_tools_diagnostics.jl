"""Basic flow diagnostics used across the larger flow-tools module."""

# Reynolds number calculation
"""Calculate Reynolds number Re = |u| * L / ν"""
function reynolds_number(velocity::VectorField, viscosity::Float64, length_scale::Float64=1.0)

    # Calculate velocity magnitude (GPU-compatible allocation)
    ensure_layout!(velocity.components[1], :g)
    vel_magnitude_squared = similar_zeros(get_grid_data(velocity.components[1]))

    for component in velocity.components
        ensure_layout!(component, :g)
        vel_magnitude_squared .+= abs2.(get_grid_data(component))
    end

    vel_magnitude = sqrt.(vel_magnitude_squared)
    # `maximum` on a PencilArray is COLLECTIVE (already global); reduce the LOCAL
    # storage (parent) so the single global_max below is the one and only collective
    # (was a redundant double-reduce). Use the field's communicator, not COMM_WORLD,
    # so a sub-communicator run does not deadlock/mismatch.
    max_vel = maximum(parent(vel_magnitude))

    reducer = GlobalArrayReducer(velocity.components[1].dist.comm)
    max_vel = global_max(reducer, max_vel)

    return max_vel * length_scale / viscosity
end

# Kinetic energy calculation
"""Calculate kinetic energy KE = (1/2) * ρ * |u|²"""
function kinetic_energy(velocity::VectorField, density::Float64=1.0)

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

"""Calculate total kinetic energy integrated over domain"""
function total_kinetic_energy(velocity::VectorField, density::Float64=1.0)

    ke_field = kinetic_energy(velocity, density)
    return integrate(ke_field)
end

# Enstrophy calculation (for 2D flows)
"""Calculate enstrophy (vorticity squared) for 2D flow"""
function enstrophy(velocity::VectorField)

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

"""Calculate total enstrophy integrated over domain"""
function total_enstrophy(velocity::VectorField)

    enstrophy_field = enstrophy(velocity)
    return integrate(enstrophy_field)
end

# Energy dissipation rate
"""Calculate energy dissipation rate ε = ν * |∇u|²"""
function energy_dissipation_rate(velocity::VectorField, viscosity::Float64)

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
"""Calculate vorticity transport equation terms for 2D flow"""
function vorticity_transport(velocity::VectorField, vorticity::ScalarField, viscosity::Float64)

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
