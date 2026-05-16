# Boundary condition dependency detection, constructors, and manager setup helpers.

"""Check if value depends on time"""
function is_time_dependent(value)
    if isa(value, String)
        return occursin(r"\bt\b", value) || occursin("∂t(", value) || occursin("dt(", value)
    elseif isa(value, TimeDependentValue) || isa(value, TimeSpaceDependentValue)
        return true
    elseif isa(value, FieldReference)
        return "t" in value.dependencies
    end
    return false
end

"""Check if value depends on spatial coordinates"""
function is_space_dependent(value)
    if isa(value, String)
        # Check for common spatial coordinate patterns
        spatial_patterns = [r"\bx\b", r"\by\b", r"\bz\b", r"\br\b", r"\btheta\b", r"\bphi\b"]
        has_greek = occursin("θ", value) || occursin("φ", value)
        return any(occursin(pat, value) for pat in spatial_patterns) || has_greek
    elseif isa(value, SpaceDependentValue) || isa(value, TimeSpaceDependentValue)
        return true
    elseif isa(value, FieldReference)
        spatial_coords = ["x", "y", "z", "r", "theta", "phi", "θ", "φ"]
        return any(coord in value.dependencies for coord in spatial_coords)
    end
    return false
end

function dirichlet_bc(field::String, coordinate::String, position, value;
                     tau_field::Union{String, Nothing}=nothing,
                     time_dependent::Union{Bool, Nothing}=nothing,
                     space_dependent::Union{Bool, Nothing}=nothing)
    """Create Dirichlet boundary condition: u(coord=pos) = value

    Supports time and spatially dependent values:
    - Constant: value = 1.0
    - Time-dependent: value = "sin(2*π*t)"
    - Space-dependent: value = "x^2 + y^2"
    - Time+Space: value = "sin(2*π*t)*exp(-x^2)"
    """

    is_time_dep = time_dependent !== nothing ? time_dependent : is_time_dependent(value)
    is_space_dep = space_dependent !== nothing ? space_dependent : is_space_dependent(value)

    return DirichletBC(field, coordinate, position, value, tau_field, is_time_dep, is_space_dep)
end

function neumann_bc(field::String, coordinate::String, position, value;
                   derivative_order::Int=1, tau_field::Union{String, Nothing}=nothing,
                   time_dependent::Union{Bool, Nothing}=nothing,
                   space_dependent::Union{Bool, Nothing}=nothing)
    """Create Neumann boundary condition: d^n u/d coord^n (coord=pos) = value

    Supports time and spatially dependent values like dirichlet_bc
    """

    is_time_dep = time_dependent !== nothing ? time_dependent : is_time_dependent(value)
    is_space_dep = space_dependent !== nothing ? space_dependent : is_space_dependent(value)

    return NeumannBC(field, coordinate, position, derivative_order, value, tau_field, is_time_dep, is_space_dep)
end

function robin_bc(field::String, coordinate::String, position, alpha, beta, value;
                 tau_field::Union{String, Nothing}=nothing,
                 time_dependent::Union{Bool, Nothing}=nothing,
                 space_dependent::Union{Bool, Nothing}=nothing)
    """Create Robin boundary condition: alpha*u + beta*du/dcoord (coord=pos) = value

    The RHS value can be time/space dependent. Dynamic alpha/beta coefficients
    are rejected because they change the LHS operator and require rebuilding
    solver matrices.
    """

    if is_time_dependent(alpha) || is_space_dependent(alpha) ||
       is_time_dependent(beta) || is_space_dependent(beta)
        throw(ArgumentError(
            "Time- or space-dependent Robin alpha/beta coefficients are not supported; " *
            "only the Robin RHS value may be dynamic."
        ))
    end

    any_time_dep = any(is_time_dependent(v) for v in [alpha, beta, value])
    any_space_dep = any(is_space_dependent(v) for v in [alpha, beta, value])

    is_time_dep = time_dependent !== nothing ? time_dependent : any_time_dep
    is_space_dep = space_dependent !== nothing ? space_dependent : any_space_dep

    return RobinBC(field, coordinate, position, alpha, beta, value, tau_field, is_time_dep, is_space_dep)
end

"""Create periodic boundary condition"""
function periodic_bc(field::String, coordinate::String)
    return PeriodicBC(field, coordinate)
end

function stress_free_bc(velocity_field::String, coordinate::String, position;
                       tau_fields::Vector{String}=String[],
                       component_coordinates::Vector{String}=String[])
    """Create stress-free boundary condition for velocity field"""
    return StressFreeBC(velocity_field, coordinate, position, tau_fields,
                        component_coordinates)
end

"""Create custom boundary condition from expression"""
function custom_bc(expression::String; tau_fields::Vector{String}=String[])
    return CustomBC(expression, tau_fields)
end

"""Add boundary condition to manager with dependency tracking"""
function add_bc!(manager::BoundaryConditionManager, bc::AbstractBoundaryCondition)
    push!(manager.conditions, bc)

    bc_index = length(manager.conditions)

    if hasfield(typeof(bc), :is_time_dependent) && bc.is_time_dependent
        push!(manager.time_dependent_bcs, bc_index)
        manager.bc_update_required = true
    end

    if hasfield(typeof(bc), :is_space_dependent) && bc.is_space_dependent
        push!(manager.space_dependent_bcs, bc_index)
    end

    empty!(manager.nonconstant_bc_indices)

    return manager
end

function add_dirichlet!(manager::BoundaryConditionManager, field::String,
                       coordinate::String, position, value; kwargs...)
    """Convenient function to add Dirichlet BC"""
    bc = dirichlet_bc(field, coordinate, position, value; kwargs...)
    add_bc!(manager, bc)
    return bc
end

function add_neumann!(manager::BoundaryConditionManager, field::String,
                     coordinate::String, position, value; kwargs...)
    """Convenient function to add Neumann BC"""
    bc = neumann_bc(field, coordinate, position, value; kwargs...)
    add_bc!(manager, bc)
    return bc
end

function add_robin!(manager::BoundaryConditionManager, field::String,
                   coordinate::String, position, alpha, beta, value; kwargs...)
    """Convenient function to add Robin BC"""
    bc = robin_bc(field, coordinate, position, alpha, beta, value; kwargs...)
    add_bc!(manager, bc)
    return bc
end

"""Convenient function to add periodic BC"""
function add_periodic!(manager::BoundaryConditionManager, field::String, coordinate::String)
    bc = periodic_bc(field, coordinate)
    add_bc!(manager, bc)
    return bc
end

function add_stress_free!(manager::BoundaryConditionManager, velocity_field::String,
                         coordinate::String, position; kwargs...)
    """Convenient function to add stress-free BC"""
    bc = stress_free_bc(velocity_field, coordinate, position; kwargs...)
    add_bc!(manager, bc)
    return bc
end

"""Convenient function to add custom BC"""
function add_custom!(manager::BoundaryConditionManager, expression::String; kwargs...)
    bc = custom_bc(expression; kwargs...)
    add_bc!(manager, bc)
    return bc
end

"""Register tau field for boundary condition enforcement"""
function register_tau_field!(manager::BoundaryConditionManager, name::String, field)
    manager.tau_fields[name] = field
    return manager
end

"""Get registered tau field"""
function get_tau_field(manager::BoundaryConditionManager, name::String)
    return get(manager.tau_fields, name, nothing)
end

"""
    validate_tau_fields!(manager::BoundaryConditionManager)

Validate that all boundary conditions on non-periodic bases have explicit tau fields.
Following the spectral approach, users must explicitly create tau fields and add them
to equations using the lift() operator.

Throws an error if any BC is missing a tau field specification.
"""
function validate_tau_fields!(manager::BoundaryConditionManager)
    missing_tau = String[]

    for bc in manager.conditions
        if isa(bc, DirichletBC) && bc.tau_field === nothing
            push!(missing_tau, "DirichletBC on $(bc.field) at $(bc.coordinate)=$(bc.position)")
        elseif isa(bc, NeumannBC) && bc.tau_field === nothing
            push!(missing_tau, "NeumannBC on $(bc.field) at $(bc.coordinate)=$(bc.position)")
        elseif isa(bc, RobinBC) && bc.tau_field === nothing
            push!(missing_tau, "RobinBC on $(bc.field) at $(bc.coordinate)=$(bc.position)")
        end
    end

    if !isempty(missing_tau)
        error_msg = """
        Missing tau field specifications for boundary conditions.

        Tarang.jl follows the spectral approach: you must explicitly create tau fields
        and add them to your equations using the lift() operator.

        Missing tau fields for:
        $(join(["  - " * m for m in missing_tau], "\n"))

        Example fix:
        ```julia
        # 1. Create tau field
        tau_u1 = ScalarField(dist, "tau_u1", (horizontal_basis,))

        # 2. Add tau to problem variables
        problem = IVP([u, tau_u1])

        # 3. Add equation with lift(tau) term
        add_equation!(problem, "dt(u) - lap(u) + lift(tau_u1, zbasis, -1) = 0")

        # 4. Add boundary condition (auto-detected from syntax)
        add_equation!(problem, "u(z=0) = 0")
        ```

        See documentation: https://subhk.github.io/Tarang.jl/pages/tau_method/
        """
        throw(ArgumentError(error_msg))
    end

    return manager
end

function auto_generate_tau_fields!(manager::BoundaryConditionManager, problem, dist)
    @warn """
    auto_generate_tau_fields! is deprecated and will be removed in a future version.

    Tarang.jl now follows the spectral approach: you must explicitly create tau fields
    and add them to your equations using the lift() operator.

    See documentation: https://subhk.github.io/Tarang.jl/pages/tau_method/
    """ maxlog=1

    validate_tau_fields!(manager)
    return manager
end

"""
    Get basis for boundary along specified coordinate.

    For a boundary condition along a given coordinate, the tau field lives on
    a lower-dimensional space (the boundary). This function returns the appropriate
    basis for that boundary space.

    For example:
    - In 2D with bases (Fourier_x, Chebyshev_y), a BC along "y" needs a Fourier_x basis
    - In 3D with bases (Fourier_x, Fourier_y, Chebyshev_z), a BC along "z" needs (Fourier_x, Fourier_y)

    Arguments:
    - manager: The BoundaryConditionManager containing coordinate/basis information
    - coordinate: The coordinate name along which the BC is applied (e.g., "x", "y", "z")

    Returns:
    - Basis or tuple of bases for the boundary, or nothing if not found
    """
function get_boundary_basis(manager::BoundaryConditionManager, coordinate::String)
    if !haskey(manager.coordinate_info, "bases") || !haskey(manager.coordinate_info, "coordinates")
        @debug "No coordinate/basis info registered in BoundaryConditionManager"
        return nothing
    end

    bases = manager.coordinate_info["bases"]
    coordinates = manager.coordinate_info["coordinates"]

    if isempty(bases) || isempty(coordinates)
        return nothing
    end

    coord_axis = nothing
    for (i, coord) in enumerate(coordinates)
        coord_name = isa(coord, String) ? coord :
                     (hasfield(typeof(coord), :name) ? coord.name : string(coord))
        if coord_name == coordinate
            coord_axis = i
            break
        end
    end

    if coord_axis === nothing
        @warn "Coordinate '$coordinate' not found in registered coordinates: $coordinates"
        return nothing
    end

    boundary_bases = []
    for (i, basis) in enumerate(bases)
        if i != coord_axis
            push!(boundary_bases, basis)
        end
    end

    if isempty(boundary_bases)
        return nothing
    elseif length(boundary_bases) == 1
        return boundary_bases[1]
    else
        return tuple(boundary_bases...)
    end
end

function register_coordinate_info!(manager::BoundaryConditionManager,
                                   coordinates::Vector, bases::Vector)
    """
    Register coordinate and basis information for boundary basis lookup.

    Arguments:
    - manager: The BoundaryConditionManager
    - coordinates: Vector of coordinate names or Coordinate objects (e.g., ["x", "y", "z"])
    - bases: Vector of Basis objects corresponding to each coordinate
    """
    if length(coordinates) != length(bases)
        throw(ArgumentError("Number of coordinates ($(length(coordinates))) must match number of bases ($(length(bases)))"))
    end

    manager.coordinate_info["coordinates"] = coordinates
    manager.coordinate_info["bases"] = bases

    @debug "Registered $(length(coordinates)) coordinates with bases for boundary conditions"
    return manager
end

"""
    Register domain information for boundary basis lookup.

    Extracts coordinate and basis information from a Domain object.
    """
function register_domain_info!(manager::BoundaryConditionManager, domain::Domain)
    if !hasfield(typeof(domain), :bases) || domain.bases === nothing
        @warn "Domain has no bases information"
        return manager
    end

    bases = collect(domain.bases)

    coordinates = String[]
    for basis in bases
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :element_label)
            push!(coordinates, basis.meta.element_label)
        else
            push!(coordinates, "coord_$(length(coordinates)+1)")
        end
    end

    manager.coordinate_info["coordinates"] = coordinates
    manager.coordinate_info["bases"] = bases
    manager.coordinate_info["domain"] = domain

    @debug "Registered domain with $(length(bases)) bases for boundary conditions"
    return manager
end

function create_lift_operator(manager::BoundaryConditionManager, tau_field_name::String,
                            target_basis, derivative_order::Int=1)
    """Create lift operator to incorporate tau field into equations"""

    if haskey(manager.lift_operators, tau_field_name)
        return manager.lift_operators[tau_field_name]
    end

    lift_op = Dict(
        "tau_field" => tau_field_name,
        "target_basis" => target_basis,
        "derivative_order" => derivative_order
    )

    manager.lift_operators[tau_field_name] = lift_op
    return lift_op
end

function apply_lift(manager::BoundaryConditionManager, tau_field_name::String,
                   target_basis, derivative_order::Int=-1)
    """Apply lift operator to incorporate tau field"""

    tau_field = get_tau_field(manager, tau_field_name)
    if tau_field === nothing
        throw(ArgumentError("Tau field $tau_field_name not found"))
    end

    basis_str = target_basis isa String ? target_basis :
                target_basis isa Symbol ? string(target_basis) :
                string(target_basis)

    return "lift($tau_field_name, $basis_str, $derivative_order)"
end

function apply_lift(manager::BoundaryConditionManager, tau_field_name::String,
                   derivative_order::Int=-1)
    throw(ArgumentError("apply_lift requires target_basis; use apply_lift(manager, tau_field_name, target_basis, derivative_order)"))
end
