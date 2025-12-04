"""
Advanced Boundary Condition Handling for Tarang.jl

This module implements comprehensive boundary condition support following
the Dedalus tau method approach, with support for:
- Dirichlet boundary conditions
- Neumann boundary conditions  
- Robin (mixed) boundary conditions
- Periodic boundary conditions
- Stress-free boundary conditions
- Custom boundary condition expressions

Translated from dedalus/core/boundary_conditions.py and enhanced
"""

using LinearAlgebra
using SparseArrays

# CPU-only (GPU support removed)

mutable struct BCPerformanceStats
    total_time::Float64
    total_evaluations::Int
    bc_updates::Int
    cache_hits::Int
    cache_misses::Int

    function BCPerformanceStats()
        new(0.0, 0, 0, 0, 0)
    end
end

# Field reference for time/space dependent boundary conditions
struct FieldReference
    name::String
    expression::Union{String, Function}
    dependencies::Vector{String}  # List of variables this depends on (e.g., ["t", "x", "y"])
end

# Time-dependent boundary condition value wrapper
struct TimeDependentValue
    expression::String  # Expression like "sin(2*π*t)" or "exp(-t)*cos(x)"
    variables::Vector{String}  # Variables used: ["t", "x", "y", etc.]
    function_obj::Union{Function, Nothing}  # Compiled function for efficiency
end

# Space-dependent boundary condition value wrapper  
struct SpaceDependentValue
    expression::String  # Expression like "sin(x)*cos(y)" or "r^2"
    coordinates::Vector{String}  # Spatial coordinates used: ["x", "y", "z", "r", "θ", etc.]
    function_obj::Union{Function, Nothing}  # Compiled function for efficiency
end

# Combined time and space dependent value
struct TimeSpaceDependentValue
    expression::String  # Expression like "sin(2*π*t)*cos(x)"
    time_variables::Vector{String}  # Time variables: ["t"]
    space_coordinates::Vector{String}  # Spatial coordinates: ["x", "y", "z"]
    function_obj::Union{Function, Nothing}  # Compiled function for efficiency
end

# Abstract boundary condition types
abstract type AbstractBoundaryCondition end

# Concrete boundary condition types
struct DirichletBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    value::Union{Real, String, Function, FieldReference}
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct NeumannBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    derivative_order::Int
    value::Union{Real, String, Function, FieldReference}
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct RobinBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    alpha::Union{Real, String, Function, FieldReference}  # Coefficient of field
    beta::Union{Real, String, Function, FieldReference}   # Coefficient of derivative
    value::Union{Real, String, Function, FieldReference}
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct PeriodicBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
end

struct StressFreeBC <: AbstractBoundaryCondition
    velocity_field::String
    coordinate::String
    position::Union{Real, String}
    tau_fields::Vector{String}
end

struct CustomBC <: AbstractBoundaryCondition
    expression::String
    tau_fields::Vector{String}
end

# Enhanced boundary condition manager with time/space dependency
mutable struct BoundaryConditionManager
    conditions::Vector{AbstractBoundaryCondition}
    tau_fields::Dict{String, Any}  # Maps tau field names to field objects
    lift_operators::Dict{String, Any}  # Cached lift operators
    coordinate_info::Dict{String, Any}  # Information about coordinate systems
    time_variable::Union{String, Nothing}  # Name of time variable (e.g., "t")
    coordinate_fields::Dict{String, Any}  # Spatial coordinate fields (e.g., "x", "y", "z")
    time_dependent_bcs::Vector{Int}  # Indices of time-dependent BCs
    space_dependent_bcs::Vector{Int}  # Indices of space-dependent BCs
    bc_update_required::Bool  # Flag indicating if BC values need updating
    
    # Workspace and caching
    workspace::Dict{String, AbstractArray}
    bc_cache::Dict{String, AbstractArray}
    performance_stats::BCPerformanceStats

    function BoundaryConditionManager(; device::String="cpu")
        workspace = Dict{String, AbstractArray}()
        bc_cache = Dict{String, AbstractArray}()
        perf_stats = BCPerformanceStats()

        new(AbstractBoundaryCondition[], Dict{String, Any}(),
            Dict{String, Any}(), Dict{String, Any}(), nothing,
            Dict{String, Any}(), Int[], Int[], false,
            workspace, bc_cache, perf_stats)
    end
end

# Utility functions for time/space dependency detection
function is_time_dependent(value)
    """Check if value depends on time"""
    if isa(value, String)
        return occursin(r"\bt\b", value) || occursin("dt(", value)
    elseif isa(value, TimeDependentValue) || isa(value, TimeSpaceDependentValue)
        return true
    elseif isa(value, FieldReference)
        return "t" in value.dependencies
    end
    return false
end

function is_space_dependent(value)
    """Check if value depends on spatial coordinates"""
    if isa(value, String)
        # Check for common spatial coordinate patterns
        spatial_patterns = [r"\bx\b", r"\by\b", r"\bz\b", r"\br\b", r"\btheta\b", r"\bphi\b"]
        return any(occursin(pat, value) for pat in spatial_patterns)
    elseif isa(value, SpaceDependentValue) || isa(value, TimeSpaceDependentValue)
        return true
    elseif isa(value, FieldReference)
        spatial_coords = ["x", "y", "z", "r", "theta", "phi"]
        return any(coord in value.dependencies for coord in spatial_coords)
    end
    return false
end

# Enhanced BC Creation Functions with time/space dependency support
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
    
    # Auto-detect dependencies if not specified
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
    
    All parameters (alpha, beta, value) can be time/space dependent
    """
    
    # Check if any component is time/space dependent
    any_time_dep = any(is_time_dependent(v) for v in [alpha, beta, value])
    any_space_dep = any(is_space_dependent(v) for v in [alpha, beta, value])
    
    is_time_dep = time_dependent !== nothing ? time_dependent : any_time_dep
    is_space_dep = space_dependent !== nothing ? space_dependent : any_space_dep
    
    return RobinBC(field, coordinate, position, alpha, beta, value, tau_field, is_time_dep, is_space_dep)
end

function periodic_bc(field::String, coordinate::String)
    """Create periodic boundary condition"""
    return PeriodicBC(field, coordinate)
end

function stress_free_bc(velocity_field::String, coordinate::String, position; 
                       tau_fields::Vector{String}=String[])
    """Create stress-free boundary condition for velocity field"""
    return StressFreeBC(velocity_field, coordinate, position, tau_fields)
end

function custom_bc(expression::String; tau_fields::Vector{String}=String[])
    """Create custom boundary condition from expression"""
    return CustomBC(expression, tau_fields)
end

# Enhanced BC Management Functions with time/space dependency tracking
function add_bc!(manager::BoundaryConditionManager, bc::AbstractBoundaryCondition)
    """Add boundary condition to manager with dependency tracking"""
    push!(manager.conditions, bc)
    
    # Track time and space dependencies
    bc_index = length(manager.conditions)
    
    if hasfield(typeof(bc), :is_time_dependent) && bc.is_time_dependent
        push!(manager.time_dependent_bcs, bc_index)
        manager.bc_update_required = true
    end
    
    if hasfield(typeof(bc), :is_space_dependent) && bc.is_space_dependent
        push!(manager.space_dependent_bcs, bc_index)
    end
    
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

function add_periodic!(manager::BoundaryConditionManager, field::String, coordinate::String)
    """Convenient function to add periodic BC"""
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

function add_custom!(manager::BoundaryConditionManager, expression::String; kwargs...)
    """Convenient function to add custom BC"""
    bc = custom_bc(expression; kwargs...)
    add_bc!(manager, bc)
    return bc
end

# Tau field management
function register_tau_field!(manager::BoundaryConditionManager, name::String, field)
    """Register tau field for boundary condition enforcement"""
    manager.tau_fields[name] = field
    return manager
end

function get_tau_field(manager::BoundaryConditionManager, name::String)
    """Get registered tau field"""
    return get(manager.tau_fields, name, nothing)
end

function auto_generate_tau_fields!(manager::BoundaryConditionManager, problem, dist)
    """Automatically generate tau fields for boundary conditions that need them"""
    
    tau_counter = 1
    
    for bc in manager.conditions
        tau_name = nothing
        
        if isa(bc, DirichletBC) && bc.tau_field === nothing
            tau_name = "tau_$(bc.field)_dirichlet_$tau_counter"
            tau_counter += 1
            
            # Create scalar tau field on boundary basis
            coord_basis = get_boundary_basis(manager, bc.coordinate)
            if coord_basis !== nothing
                tau_field = ScalarField(dist, tau_name, (coord_basis,))
                # Ensure tau field is on correct device
                tau_field.data_g = tau_field.data_g
                tau_field.data_c = tau_field.data_c
                
                register_tau_field!(manager, tau_name, tau_field)
                bc = DirichletBC(bc.field, bc.coordinate, bc.position, bc.value, tau_name, bc.is_time_dependent, bc.is_space_dependent)
                # Update the BC in the list
                for (i, existing_bc) in enumerate(manager.conditions)
                    if existing_bc === bc
                        manager.conditions[i] = bc
                        break
                    end
                end
            end
            
        elseif isa(bc, NeumannBC) && bc.tau_field === nothing
            tau_name = "tau_$(bc.field)_neumann_$tau_counter"
            tau_counter += 1
            
            coord_basis = get_boundary_basis(manager, bc.coordinate)
            if coord_basis !== nothing
                tau_field = ScalarField(dist, tau_name, (coord_basis,))
                # Ensure tau field is on correct device
                tau_field.data_g = tau_field.data_g
                tau_field.data_c = tau_field.data_c
                
                register_tau_field!(manager, tau_name, tau_field)
                bc = NeumannBC(bc.field, bc.coordinate, bc.position, 
                              bc.derivative_order, bc.value, tau_name, bc.is_time_dependent, bc.is_space_dependent)
                # Update the BC in the list
                for (i, existing_bc) in enumerate(manager.conditions)
                    if existing_bc === bc
                        manager.conditions[i] = bc
                        break
                    end
                end
            end
            
        elseif isa(bc, RobinBC) && bc.tau_field === nothing
            tau_name = "tau_$(bc.field)_robin_$tau_counter"
            tau_counter += 1
            
            coord_basis = get_boundary_basis(manager, bc.coordinate)
            if coord_basis !== nothing
                tau_field = ScalarField(dist, tau_name, (coord_basis,))
                # Ensure tau field is on correct device
                tau_field.data_g = tau_field.data_g
                tau_field.data_c = tau_field.data_c
                
                register_tau_field!(manager, tau_name, tau_field)
                bc = RobinBC(bc.field, bc.coordinate, bc.position,
                            bc.alpha, bc.beta, bc.value, tau_name, bc.is_time_dependent, bc.is_space_dependent)
                # Update the BC in the list
                for (i, existing_bc) in enumerate(manager.conditions)
                    if existing_bc === bc
                        manager.conditions[i] = bc
                        break
                    end
                end
            end
        end
    end
    
    return manager
end

function get_boundary_basis(manager::BoundaryConditionManager, coordinate::String)
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

    # Check if coordinate info has been registered
    if !haskey(manager.coordinate_info, "bases") || !haskey(manager.coordinate_info, "coordinates")
        @debug "No coordinate/basis info registered in BoundaryConditionManager"
        return nothing
    end

    bases = manager.coordinate_info["bases"]
    coordinates = manager.coordinate_info["coordinates"]

    if isempty(bases) || isempty(coordinates)
        return nothing
    end

    # Find which axis corresponds to this coordinate
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

    # Get bases for all OTHER coordinates (the boundary is the remaining dimensions)
    boundary_bases = []
    for (i, basis) in enumerate(bases)
        if i != coord_axis
            push!(boundary_bases, basis)
        end
    end

    # Return appropriate result based on dimensionality
    if isempty(boundary_bases)
        # 1D case: boundary is a point, no basis needed
        # Return a trivial constant basis or nothing
        return nothing
    elseif length(boundary_bases) == 1
        # 2D case: boundary is 1D
        return boundary_bases[1]
    else
        # 3D+ case: boundary is multi-dimensional
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

function register_domain_info!(manager::BoundaryConditionManager, domain::Domain)
    """
    Register domain information for boundary basis lookup.

    Extracts coordinate and basis information from a Domain object.
    """
    if !hasfield(typeof(domain), :bases) || domain.bases === nothing
        @warn "Domain has no bases information"
        return manager
    end

    bases = collect(domain.bases)

    # Extract coordinate names from bases
    coordinates = String[]
    for basis in bases
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :coordsys)
            coordsys = basis.meta.coordsys
            if hasfield(typeof(coordsys), :names) && !isempty(coordsys.names)
                push!(coordinates, coordsys.names[1])
            else
                push!(coordinates, "coord_$(length(coordinates)+1)")
            end
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

# Lift operator support
function create_lift_operator(manager::BoundaryConditionManager, tau_field_name::String, 
                            target_basis, derivative_order::Int=1)
    """Create lift operator to incorporate tau field into equations"""
    
    if haskey(manager.lift_operators, tau_field_name)
        return manager.lift_operators[tau_field_name]
    end
    
    # Create lift operator (placeholder - actual implementation would depend on basis type)
    lift_op = Dict(
        "tau_field" => tau_field_name,
        "target_basis" => target_basis,
        "derivative_order" => derivative_order
    )
    
    manager.lift_operators[tau_field_name] = lift_op
    return lift_op
end

function apply_lift(manager::BoundaryConditionManager, tau_field_name::String, 
                   derivative_order::Int=-1)
    """Apply lift operator to incorporate tau field"""
    
    tau_field = get_tau_field(manager, tau_field_name)
    if tau_field === nothing
        throw(ArgumentError("Tau field $tau_field_name not found"))
    end
    
    # Return symbolic representation (actual implementation would create operator)
    return "lift($tau_field_name, $derivative_order)"
end

# BC to equation conversion
function bc_to_equation(manager::BoundaryConditionManager, bc::DirichletBC)
    """Convert Dirichlet BC to equation string"""
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    val_str = isa(bc.value, String) ? bc.value : string(bc.value)
    
    equation = "$(bc.field)($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        # Include tau term in the equation
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

function bc_to_equation(manager::BoundaryConditionManager, bc::NeumannBC)
    """Convert Neumann BC to equation string"""
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    val_str = isa(bc.value, String) ? bc.value : string(bc.value)
    
    # Create derivative notation
    if bc.derivative_order == 1
        deriv = "d$(bc.coordinate)($(bc.field))"
    else
        deriv = "d$(bc.coordinate)^$(bc.derivative_order)($(bc.field))"
    end
    
    equation = "$deriv($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

function bc_to_equation(manager::BoundaryConditionManager, bc::RobinBC)
    """Convert Robin BC to equation string"""
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    alpha_str = isa(bc.alpha, String) ? bc.alpha : string(bc.alpha)
    beta_str = isa(bc.beta, String) ? bc.beta : string(bc.beta)
    val_str = isa(bc.value, String) ? bc.value : string(bc.value)
    
    equation = "$(alpha_str)*$(bc.field)($(bc.coordinate)=$pos_str) + " *
               "$(beta_str)*d$(bc.coordinate)($(bc.field))($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

function bc_to_equation(manager::BoundaryConditionManager, bc::StressFreeBC)
    """Convert stress-free BC to equations"""
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    
    # Stress-free: u = 0 and d^2u/dz^2 = 0 at boundary
    equations = [
        "$(bc.velocity_field)($(bc.coordinate)=$pos_str) = 0",
        "d$(bc.coordinate)^2($(bc.velocity_field))($(bc.coordinate)=$pos_str) = 0"
    ]
    
    return equations
end

function bc_to_equation(manager::BoundaryConditionManager, bc::CustomBC)
    """Convert custom BC to equation string"""
    return bc.expression
end

# Problem integration
function apply_boundary_conditions!(manager::BoundaryConditionManager, problem)
    """Apply all boundary conditions to problem"""
    
    equations_added = String[]
    
    for bc in manager.conditions
        if isa(bc, StressFreeBC)
            # Stress-free BCs generate multiple equations
            eqs = bc_to_equation(manager, bc)
            for eq in eqs
                add_equation!(problem, eq)
                push!(equations_added, eq)
            end
        else
            # Other BCs generate single equation
            eq = bc_to_equation(manager, bc)
            add_equation!(problem, eq)
            push!(equations_added, eq)
        end
    end
    
    return equations_added
end

function validate_boundary_conditions(manager::BoundaryConditionManager, problem)
    """Validate boundary conditions for consistency and completeness"""
    
    warnings = String[]
    errors = String[]
    
    # Check for required tau fields
    for bc in manager.conditions
        tau_needed = false
        tau_field = nothing
        
        if isa(bc, DirichletBC)
            tau_needed = true
            tau_field = bc.tau_field
        elseif isa(bc, NeumannBC)
            tau_needed = true
            tau_field = bc.tau_field
        elseif isa(bc, RobinBC)
            tau_needed = true
            tau_field = bc.tau_field
        end
        
        if tau_needed && tau_field === nothing
            # This is informational - tau fields are optional for simple problems
            @debug "Boundary condition for $(bc.field) does not have tau field (optional for simple problems)"
        elseif tau_needed && tau_field !== nothing && get_tau_field(manager, tau_field) === nothing
            push!(errors, "Tau field $tau_field not registered")
        end
    end

    # Check for coordinate consistency
    field_coords = Dict{String, Set{String}}()
    for bc in manager.conditions
        field_name = ""
        coord = ""
        
        if isa(bc, DirichletBC)
            field_name = bc.field
            coord = bc.coordinate
        elseif isa(bc, NeumannBC)
            field_name = bc.field
            coord = bc.coordinate
        elseif isa(bc, RobinBC)
            field_name = bc.field
            coord = bc.coordinate
        elseif isa(bc, StressFreeBC)
            field_name = bc.velocity_field
            coord = bc.coordinate
        end
        
        if field_name != "" && coord != ""
            if !haskey(field_coords, field_name)
                field_coords[field_name] = Set{String}()
            end
            push!(field_coords[field_name], coord)
        end
    end
    
    # Report validation results
    if !isempty(warnings)
        @warn "Boundary condition warnings:\n" * join(warnings, "\n")
    end
    
    if !isempty(errors)
        throw(ArgumentError("Boundary condition errors:\n" * join(errors, "\n")))
    end
    
    return true
end

# Utility functions
function get_bc_count_by_type(manager::BoundaryConditionManager)
    """Get count of boundary conditions by type"""
    counts = Dict{String, Int}()
    
    for bc in manager.conditions
        bc_type = string(typeof(bc))
        counts[bc_type] = get(counts, bc_type, 0) + 1
    end
    
    return counts
end

function get_required_tau_fields(manager::BoundaryConditionManager)
    """Get list of required tau field names"""
    tau_fields = String[]
    
    for bc in manager.conditions
        if isa(bc, DirichletBC) && bc.tau_field !== nothing
            push!(tau_fields, bc.tau_field)
        elseif isa(bc, NeumannBC) && bc.tau_field !== nothing
            push!(tau_fields, bc.tau_field)
        elseif isa(bc, RobinBC) && bc.tau_field !== nothing
            push!(tau_fields, bc.tau_field)
        elseif isa(bc, StressFreeBC)
            append!(tau_fields, bc.tau_fields)
        elseif isa(bc, CustomBC)
            append!(tau_fields, bc.tau_fields)
        end
    end
    
    return unique(tau_fields)
end

# Time and space dependency management
function set_time_variable!(manager::BoundaryConditionManager, time_var::String, time_field=nothing)
    """Set the time variable for time-dependent boundary conditions"""
    manager.time_variable = time_var
    if time_field !== nothing
        manager.coordinate_fields[time_var] = time_field
    end
    return manager
end

function add_coordinate_field!(manager::BoundaryConditionManager, coord_name::String, field)
    """Add spatial coordinate field for space-dependent boundary conditions"""
    manager.coordinate_fields[coord_name] = field
    return manager
end

function evaluate_bc_value(manager::BoundaryConditionManager, bc, current_time=0.0, coords=Dict())
    """Evaluate boundary condition value at current time and spatial coordinates"""

    if isa(bc, DirichletBC)
        value = bc.value
    elseif isa(bc, NeumannBC)
        value = bc.value
    elseif isa(bc, RobinBC)
        # For Robin BCs, evaluate all components
        alpha = evaluate_expression(bc.alpha, current_time, coords, manager)
        beta = evaluate_expression(bc.beta, current_time, coords, manager)
        val = evaluate_expression(bc.value, current_time, coords, manager)
        
        # Update performance statistics
        manager.performance_stats.total_evaluations += 3
        
        return (alpha, beta, val)
    else
        return nothing
    end
    
    result = evaluate_expression(value, current_time, coords, manager)
    
    # Update performance statistics
    manager.performance_stats.total_evaluations += 1
    
    return result
end

function evaluate_expression(expr, current_time=0.0, coords=Dict())
    """Evaluate a boundary condition expression with current time and coordinates"""
    
    if isa(expr, Real)
        return expr
    elseif isa(expr, String)
        # Simple expression evaluation
        # This is a placeholder - in practice, you'd want a proper expression parser
        # For now, handle some common cases
        if expr == "0" || expr == "0.0"
            return 0.0
        elseif expr == "1" || expr == "1.0"
            return 1.0
        elseif occursin("sin(", expr) && occursin("t", expr)
            # Simple time-dependent case: sin(ωt)
            if occursin("sin(2*pi*t)", expr)
                return sin(2*π*current_time)
            elseif occursin("sin(t)", expr)
                return sin(current_time)
            end
        elseif occursin("cos(", expr) && occursin("t", expr)
            # Simple time-dependent case: cos(ωt)  
            if occursin("cos(2*pi*t)", expr)
                return cos(2*π*current_time)
            elseif occursin("cos(t)", expr)
                return cos(current_time)
            end
        elseif occursin("exp(", expr) && occursin("t", expr)
            # Exponential time dependence
            if occursin("exp(-t)", expr)
                return exp(-current_time)
            elseif occursin("exp(t)", expr)
                return exp(current_time)
            end
        end
        
        # If we can't parse it, return the string for symbolic processing
        return expr
        
    elseif isa(expr, Function)
        # Evaluate function with available arguments
        try
            if haskey(coords, "x") && haskey(coords, "y")
                return expr(current_time, coords["x"], coords["y"])
            elseif haskey(coords, "x")
                return expr(current_time, coords["x"])
            else
                return expr(current_time)
            end
        catch
            return expr(current_time)
        end
        
    elseif isa(expr, TimeDependentValue) || isa(expr, TimeSpaceDependentValue)
        # Evaluate compiled function if available
        if expr.function_obj !== nothing
            return expr.function_obj(current_time, coords)
        else
            # Fall back to string evaluation
            return evaluate_expression(expr.expression, current_time, coords)
        end
    end
    
    return expr
end

function update_time_dependent_bcs!(manager::BoundaryConditionManager, current_time)
    """Update all time-dependent boundary conditions for current time"""
    
    if isempty(manager.time_dependent_bcs)
        return manager
    end
    
    start_time = time()
    @debug "Updating $(length(manager.time_dependent_bcs)) time-dependent BCs at t=$current_time"
    
    for bc_index in manager.time_dependent_bcs
        bc = manager.conditions[bc_index]
        
        # Update the BC value based on current time
        # Cache evaluated values for repeated use
        cache_key = "bc_$(bc_index)_t_$(current_time)"
        
        if isa(bc, DirichletBC) && bc.is_time_dependent
            if !haskey(manager.bc_cache, cache_key)
                new_value = evaluate_bc_value(manager, bc, current_time)
                if isa(new_value, AbstractArray)
                    manager.bc_cache[cache_key] = new_value
                end
            end
            @debug "Updated Dirichlet BC for $(bc.field)"

        elseif isa(bc, NeumannBC) && bc.is_time_dependent
            if !haskey(manager.bc_cache, cache_key)
                new_value = evaluate_bc_value(manager, bc, current_time)
                if isa(new_value, AbstractArray)
                    manager.bc_cache[cache_key] = new_value
                end
            end
            @debug "Updated Neumann BC for $(bc.field)"

        elseif isa(bc, RobinBC) && bc.is_time_dependent
            if !haskey(manager.bc_cache, cache_key)
                alpha, beta, value = evaluate_bc_value(manager, bc, current_time)
                for (comp_name, comp_value) in [("alpha", alpha), ("beta", beta), ("value", value)]
                    if isa(comp_value, AbstractArray)
                        manager.bc_cache["$(cache_key)_$(comp_name)"] = comp_value
                    end
                end
            end
            @debug "Updated Robin BC for $(bc.field)"
        end
    end
    
    # Update performance statistics
    update_time = time() - start_time
    manager.performance_stats.total_time += update_time
    manager.performance_stats.bc_updates += 1
    
    manager.bc_update_required = false
    return manager
end

function requires_bc_update(manager::BoundaryConditionManager)
    """Check if boundary conditions need updating"""
    return manager.bc_update_required || !isempty(manager.time_dependent_bcs)
end

function has_time_dependent_bcs(manager::BoundaryConditionManager)
    """Check if manager has any time-dependent boundary conditions"""
    return !isempty(manager.time_dependent_bcs)
end

function has_space_dependent_bcs(manager::BoundaryConditionManager)
    """Check if manager has any space-dependent boundary conditions"""
    return !isempty(manager.space_dependent_bcs)
end

function clear_boundary_conditions!(manager::BoundaryConditionManager)
    """Clear all boundary conditions"""
    empty!(manager.conditions)
    empty!(manager.tau_fields)
    empty!(manager.lift_operators)
    empty!(manager.time_dependent_bcs)
    empty!(manager.space_dependent_bcs)
    manager.bc_update_required = false
    return manager
end

function log_bc_performance(manager::BoundaryConditionManager)
    """Log boundary condition performance statistics"""

    stats = manager.performance_stats

    @info "BC performance:"
    @info "  Total evaluations: $(stats.total_evaluations)"
    @info "  BC updates: $(stats.bc_updates)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    if stats.total_evaluations > 0
        @info "  Average evaluation time: $(round(stats.total_time/stats.total_evaluations*1000, digits=3)) ms"
    end
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses ($(round(100*stats.cache_hits/max(stats.cache_hits+stats.cache_misses, 1), digits=1))% hit rate)"
end

function evaluate_space_dependent_bcs!(manager::BoundaryConditionManager, coordinates::Dict)
    """Evaluate space-dependent boundary conditions"""

    if isempty(manager.space_dependent_bcs)
        return manager
    end

    start_time = time()

    @debug "Evaluating $(length(manager.space_dependent_bcs)) space-dependent BCs"

    for bc_index in manager.space_dependent_bcs
        bc = manager.conditions[bc_index]

        coord_hash = hash(coordinates)
        cache_key = "bc_$(bc_index)_spatial_$(coord_hash)"

        if !haskey(manager.bc_cache, cache_key)
            if isa(bc, DirichletBC) && bc.is_space_dependent
                new_value = evaluate_bc_value(manager, bc, 0.0, coordinates)
                if isa(new_value, AbstractArray)
                    manager.bc_cache[cache_key] = new_value
                end
                manager.performance_stats.cache_misses += 1
            elseif isa(bc, NeumannBC) && bc.is_space_dependent
                new_value = evaluate_bc_value(manager, bc, 0.0, coordinates)
                if isa(new_value, AbstractArray)
                    manager.bc_cache[cache_key] = new_value
                end
                manager.performance_stats.cache_misses += 1
            elseif isa(bc, RobinBC) && bc.is_space_dependent
                alpha, beta, value = evaluate_bc_value(manager, bc, 0.0, coordinates)
                for (comp_name, comp_value) in [("alpha", alpha), ("beta", beta), ("value", value)]
                    if isa(comp_value, AbstractArray)
                        manager.bc_cache["$(cache_key)_$(comp_name)"] = comp_value
                    end
                end
                manager.performance_stats.cache_misses += 1
            end
        else
            manager.performance_stats.cache_hits += 1
        end
    end

    manager.performance_stats.total_time += time() - start_time

    return manager
end

function clear_bc_cache!(manager::BoundaryConditionManager)
    """Clear cache for boundary conditions"""
    empty!(manager.bc_cache)
    return manager
end

# Export all public functions
export AbstractBoundaryCondition, DirichletBC, NeumannBC, RobinBC, PeriodicBC,
       StressFreeBC, CustomBC, BoundaryConditionManager,
       dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
       add_bc!, add_dirichlet!, add_neumann!, add_robin!, add_periodic!,
       add_stress_free!, add_custom!,
       register_tau_field!, get_tau_field, auto_generate_tau_fields!,
       create_lift_operator, apply_lift, bc_to_equation,
       apply_boundary_conditions!, validate_boundary_conditions,
       get_bc_count_by_type, get_required_tau_fields, clear_boundary_conditions!,
       TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
       set_time_variable!, add_coordinate_field!, evaluate_bc_value, evaluate_expression,
       update_time_dependent_bcs!, requires_bc_update, has_time_dependent_bcs, has_space_dependent_bcs,
       is_time_dependent, is_space_dependent,
       log_bc_performance, evaluate_space_dependent_bcs!, clear_bc_cache!
