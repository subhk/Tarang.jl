"""
Advanced Boundary Condition Handling for Tarang.jl

This module implements comprehensive boundary condition support using
the tau method approach, with support for:
- Dirichlet boundary conditions
- Neumann boundary conditions
- Robin (mixed) boundary conditions
- Periodic boundary conditions
- Stress-free boundary conditions
- Custom boundary condition expressions
"""

using LinearAlgebra
using SparseArrays

# GPU support: This module works with both CPU and GPU arrays.
# - Expression evaluation uses broadcast() which handles GPU arrays automatically
# - Workspace and cache dictionaries accept AbstractArray (works with CuArray)
# - When coordinate arrays (x, y, z grids) are on GPU, results are on GPU

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
# Union type for BC values that can be constant, expression, or function
const BCValueType = Union{Real, String, Function, FieldReference,
                          TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue}

struct DirichletBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    value::BCValueType
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct NeumannBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    derivative_order::Int
    value::BCValueType
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct RobinBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    alpha::BCValueType  # Coefficient of field
    beta::BCValueType   # Coefficient of derivative
    value::BCValueType
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
mutable struct BoundaryConditionManager{Arch<:AbstractArchitecture}
    conditions::Vector{AbstractBoundaryCondition}
    tau_fields::Dict{String, Any}  # Maps tau field names to field objects
    lift_operators::Dict{String, Any}  # Cached lift operators
    coordinate_info::Dict{String, Any}  # Information about coordinate systems
    time_variable::Union{String, Nothing}  # Name of time variable (e.g., "t")
    coordinate_fields::Dict{String, Any}  # Spatial coordinate fields (e.g., "x", "y", "z")
    time_dependent_bcs::Vector{Int}  # Indices of time-dependent BCs
    space_dependent_bcs::Vector{Int}  # Indices of space-dependent BCs
    bc_update_required::Bool  # Flag indicating if BC values need updating

    # Equation index tracking
    bc_equation_indices::Dict{Int, Int}  # bc_index => equation_index in problem.equations

    # Workspace and caching
    workspace::Dict{String, AbstractArray}
    bc_cache::Dict{String, Any}  # Stores both scalar and array BC values
    performance_stats::BCPerformanceStats

    # Architecture for CPU/GPU support
    architecture::Arch

    function BoundaryConditionManager(; architecture::Arch=CPU()) where {Arch<:AbstractArchitecture}
        workspace = Dict{String, AbstractArray}()
        bc_cache = Dict{String, Any}()
        perf_stats = BCPerformanceStats()

        new{Arch}(AbstractBoundaryCondition[], Dict{String, Any}(),
            Dict{String, Any}(), Dict{String, Any}(), nothing,
            Dict{String, Any}(), Int[], Int[], false,
            Dict{Int, Int}(),
            workspace, bc_cache, perf_stats, architecture)
    end
end

# Architecture helper functions for BoundaryConditionManager
"""
    is_gpu(manager::BoundaryConditionManager)

Check if the boundary condition manager is configured for GPU.
"""
is_gpu(manager::BoundaryConditionManager) = is_gpu(manager.architecture)

"""
    architecture(manager::BoundaryConditionManager)

Get the architecture (CPU or GPU) of the boundary condition manager.
"""
architecture(manager::BoundaryConditionManager) = manager.architecture

# Utility functions for time/space dependency detection
function is_time_dependent(value)
    """Check if value depends on time"""
    if isa(value, String)
        return occursin(r"\bt\b", value) || occursin("∂t(", value) || occursin("dt(", value)
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

"""
    validate_tau_fields!(manager::BoundaryConditionManager)

Validate that all boundary conditions on non-periodic bases have explicit tau fields.
Following the Dedalus approach, users must explicitly create tau fields and add them
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

        Tarang.jl follows the Dedalus approach: you must explicitly create tau fields
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

# Deprecated: auto_generate_tau_fields! is no longer supported
# Users must explicitly create tau fields following the Dedalus approach
function auto_generate_tau_fields!(manager::BoundaryConditionManager, problem, dist)
    @warn """
    auto_generate_tau_fields! is deprecated and will be removed in a future version.

    Tarang.jl now follows the Dedalus approach: you must explicitly create tau fields
    and add them to your equations using the lift() operator.

    See documentation: https://subhk.github.io/Tarang.jl/pages/tau_method/
    """ maxlog=1

    # Call validation instead - this will error if tau fields are missing
    validate_tau_fields!(manager)
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

# Lift operator support
function create_lift_operator(manager::BoundaryConditionManager, tau_field_name::String, 
                            target_basis, derivative_order::Int=1)
    """Create lift operator to incorporate tau field into equations"""
    
    if haskey(manager.lift_operators, tau_field_name)
        return manager.lift_operators[tau_field_name]
    end
    
    # Create lift operator as symbolic descriptor
    # The actual numerical lift matrix is built during system assembly based on basis type
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

    # Return symbolic representation (actual implementation would create operator)
    return "lift($tau_field_name, $basis_str, $derivative_order)"
end

function apply_lift(manager::BoundaryConditionManager, tau_field_name::String,
                   derivative_order::Int=-1)
    throw(ArgumentError("apply_lift requires target_basis; use apply_lift(manager, tau_field_name, target_basis, derivative_order)"))
end

# BC to equation conversion
function _bc_value_to_string(value)
    if isa(value, String)
        return value
    elseif isa(value, FieldReference)
        if isa(value.expression, String)
            return value.expression
        elseif isa(value.expression, Function)
            return _bc_value_to_string(value.expression)
        else
            return value.name
        end
    elseif isa(value, Function)
        return string(Base.nameof(value))
    else
        return string(value)
    end
end

function _bc_derivative_str(field::String, coordinate::String, order::Int)
    if order == 1
        return "d($field, $coordinate)"
    end
    return "d($field, $coordinate, $order)"
end

function bc_to_equation(manager::BoundaryConditionManager, bc::DirichletBC)
    """Convert Dirichlet BC to equation string"""
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    val_str = _bc_value_to_string(bc.value)
    
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
    val_str = _bc_value_to_string(bc.value)
    
    # Create derivative notation
    deriv = _bc_derivative_str(bc.field, bc.coordinate, bc.derivative_order)
    
    equation = "$deriv($(bc.coordinate)=$pos_str) = $val_str"
    
    if bc.tau_field !== nothing
        equation = "$equation  # tau: $(bc.tau_field)"
    end
    
    return equation
end

function bc_to_equation(manager::BoundaryConditionManager, bc::RobinBC)
    """Convert Robin BC to equation string"""
    pos_str = isa(bc.position, String) ? bc.position : string(bc.position)
    alpha_str = _bc_value_to_string(bc.alpha)
    beta_str = _bc_value_to_string(bc.beta)
    val_str = _bc_value_to_string(bc.value)
    
    deriv = _bc_derivative_str(bc.field, bc.coordinate, 1)
    equation = "($(alpha_str))*$(bc.field)($(bc.coordinate)=$pos_str) + " *
               "($(beta_str))*$(deriv)($(bc.coordinate)=$pos_str) = $val_str"
    
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
        "$( _bc_derivative_str(bc.velocity_field, bc.coordinate, 2) )($(bc.coordinate)=$pos_str) = 0"
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
        if isa(bc, PeriodicBC)
            # Periodic BCs are enforced by the Fourier basis (inherently periodic
            # spectral representation) — no equation needed.
            continue
        elseif isa(bc, StressFreeBC)
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
        alpha = evaluate_expression(bc.alpha, current_time, coords)
        beta = evaluate_expression(bc.beta, current_time, coords)
        val = evaluate_expression(bc.value, current_time, coords)
        
        # Update performance statistics
        manager.performance_stats.total_evaluations += 3
        
        return (alpha, beta, val)
    else
        return nothing
    end
    
    result = evaluate_expression(value, current_time, coords)
    
    # Update performance statistics
    manager.performance_stats.total_evaluations += 1
    
    return result
end

"""
    evaluate_expression(expr, current_time=0.0, coords=Dict())

Evaluate a boundary condition expression with current time and coordinates.

Supports:
- Numeric values (returned as-is)
- String expressions with mathematical functions and variables
- Function objects
- TimeDependentValue and TimeSpaceDependentValue wrappers

String expressions can contain:
- Variables: t (time), x, y, z, r, θ, φ (coordinates)
- Constants: pi, π, e
- Functions: sin, cos, tan, exp, log, sqrt, abs, sinh, cosh, tanh
- Operators: +, -, *, /, ^, ()

Examples:
- "sin(2*pi*t)" - time-dependent sinusoid
- "exp(-t)*cos(x)" - exponentially decaying spatial cosine
- "x^2 + y^2" - spatial quadratic
- "sin(t)*sin(pi*x)*sin(pi*y)" - product of temporal and spatial modes
"""
function evaluate_expression(expr, current_time=0.0, coords=Dict())
    if isa(expr, Real)
        return expr
    elseif isa(expr, String)
        return _evaluate_string_expression(expr, current_time, coords)
    elseif isa(expr, Function)
        return _evaluate_function_expression(expr, current_time, coords)
    elseif isa(expr, FieldReference)
        return evaluate_expression(expr.expression, current_time, coords)
    elseif isa(expr, TimeDependentValue) || isa(expr, TimeSpaceDependentValue)
        if expr.function_obj !== nothing
            return _evaluate_function_expression(expr.function_obj, current_time, coords)
        else
            return evaluate_expression(expr.expression, current_time, coords)
        end
    elseif isa(expr, SpaceDependentValue)
        if expr.function_obj !== nothing
            return _evaluate_space_function_expression(expr.function_obj, coords)
        else
            return evaluate_expression(expr.expression, current_time, coords)
        end
    end

    return expr
end

"""
Evaluate a string expression by substituting variables and parsing.
Uses Julia's Meta.parse for safe expression evaluation.
"""
function _evaluate_string_expression(expr::String, current_time, coords)
    # Handle simple constant cases first
    stripped = strip(expr)
    if stripped == "0" || stripped == "0.0"
        return 0.0
    elseif stripped == "1" || stripped == "1.0"
        return 1.0
    elseif stripped == "-1" || stripped == "-1.0"
        return -1.0
    end

    # Try to parse as a simple number
    try
        return parse(Float64, stripped)
    catch
        # Continue with expression parsing
    end

    # Build variable substitution dictionary
    vars = Dict{String, Any}()
    vars["t"] = Float64(current_time)
    vars["pi"] = Float64(π)
    vars["π"] = Float64(π)
    vars["e"] = Float64(ℯ)

    # Add coordinate values
    for (key, val) in coords
        if isa(val, Real)
            vars[string(key)] = Float64(val)
        elseif isa(val, AbstractArray)
            # For array-valued coordinates, we'll need special handling
            vars[string(key)] = val
        end
    end

    # Check if all required variables are present
    # Build the expression with variable substitution
    try
        result = _safe_eval_math_expr(expr, vars)
        return result
    catch e
        @warn "BC expression evaluation failed for '$expr': $e" maxlog=5
        return expr
    end
end

"""
Safely evaluate a mathematical expression string with variable substitutions.
Only allows mathematical operations - no arbitrary code execution.
"""
function _safe_eval_math_expr(expr_str::String, vars::Dict{String, T}) where T
    # Allowed mathematical functions
    allowed_funcs = Dict{String, Function}(
        "sin" => sin,
        "cos" => cos,
        "tan" => tan,
        "sinh" => sinh,
        "cosh" => cosh,
        "tanh" => tanh,
        "asin" => asin,
        "acos" => acos,
        "atan" => atan,
        "exp" => exp,
        "log" => log,
        "log10" => log10,
        "log2" => log2,
        "sqrt" => sqrt,
        "abs" => abs,
        "sign" => sign,
        "floor" => floor,
        "ceil" => ceil,
        "round" => round,
        "rem" => rem,  # Remainder function (same as % operator)
        "min" => min,
        "max" => max,
    )

    # Parse the expression
    parsed = Meta.parse(expr_str)

    # Evaluate with restrictions
    return _eval_safe_ast(parsed, vars, allowed_funcs)
end

"""
Recursively evaluate an AST node with safety restrictions.
Only allows arithmetic operations and whitelisted functions.
"""
function _eval_safe_ast(node, vars::Dict{String, T}, allowed_funcs::Dict{String, Function}) where T
    if isa(node, Number)
        return Float64(node)
    elseif isa(node, Symbol)
        name = string(node)
        if haskey(vars, name)
            return vars[name]
        elseif name == "pi" || name == "π"
            return Float64(π)
        elseif name == "e" || name == "ℯ"
            return Float64(ℯ)
        else
            throw(ArgumentError("Unknown variable: $name"))
        end
    elseif isa(node, Expr)
        if node.head == :call
            func_name = string(node.args[1])
            args = [_eval_safe_ast(arg, vars, allowed_funcs) for arg in node.args[2:end]]

            # Check if it's an allowed function
            if haskey(allowed_funcs, func_name)
                func = allowed_funcs[func_name]
                return _apply_safe_function(func, args)
            # Handle binary operators
            elseif func_name == "+"
                return _apply_safe_operator(+, args)
            elseif func_name == "-"
                if length(args) == 1
                    return _apply_safe_operator(-, args)
                else
                    return _apply_safe_operator(-, args)
                end
            elseif func_name == "*"
                return _apply_safe_operator(*, args)
            elseif func_name == "/"
                return _apply_safe_operator(/, args)
            elseif func_name == "^"
                return _apply_safe_operator(^, args)
            elseif func_name == "%"
                # Julia's % is rem (remainder), not mod
                return _apply_safe_operator(rem, args)
            elseif func_name == "mod"
                return _apply_safe_operator(mod, args)
            else
                throw(ArgumentError("Function not allowed: $func_name"))
            end
        elseif node.head == :block
            # Handle block expressions (multiple statements)
            result = nothing
            for child in node.args
                if !isa(child, LineNumberNode)
                    result = _eval_safe_ast(child, vars, allowed_funcs)
                end
            end
            return result
        elseif node.head == :(=)
            throw(ArgumentError("Assignment not allowed in expressions"))
        else
            throw(ArgumentError("Expression type not allowed: $(node.head)"))
        end
    elseif isa(node, LineNumberNode)
        return nothing
    else
        throw(ArgumentError("Unsupported node type: $(typeof(node))"))
    end
end

function _apply_safe_function(func::Function, args::Vector{Any})
    if any(arg -> arg isa AbstractArray, args)
        return broadcast(func, args...)
    end
    return func(args...)
end

function _apply_safe_operator(op::Function, args::Vector{Any})
    if any(arg -> arg isa AbstractArray, args)
        return broadcast(op, args...)
    end
    return op(args...)
end

"""
Evaluate a function expression with appropriate arguments.
"""
function _coord_value(coords, name::String)
    if haskey(coords, name)
        return coords[name]
    end
    sym = Symbol(name)
    return haskey(coords, sym) ? coords[sym] : nothing
end

function _evaluate_function_expression(func::Function, current_time, coords)
    # Try different argument combinations based on function arity
    try
        return func(current_time, coords)
    catch
        # Continue with component-wise arguments
    end

    x = _coord_value(coords, "x")
    y = _coord_value(coords, "y")
    z = _coord_value(coords, "z")
    r = _coord_value(coords, "r")
    theta = _coord_value(coords, "θ")
    if theta === nothing
        theta = _coord_value(coords, "theta")
    end
    phi = _coord_value(coords, "φ")
    if phi === nothing
        phi = _coord_value(coords, "phi")
    end

    try
        # Try with all available arguments
        if x !== nothing && y !== nothing && z !== nothing
            return func(current_time, x, y, z)
        elseif x !== nothing && y !== nothing
            return func(current_time, x, y)
        elseif r !== nothing && theta !== nothing && phi !== nothing
            return func(current_time, r, theta, phi)
        elseif r !== nothing && theta !== nothing
            return func(current_time, r, theta)
        elseif x !== nothing
            return func(current_time, x)
        else
            return func(current_time)
        end
    catch
        # If that fails, try just time
        try
            return func(current_time)
        catch
            # Last resort: try no arguments
            return func()
        end
    end
end

function _evaluate_space_function_expression(func::Function, coords)
    # Try different argument combinations based on function arity
    try
        return func(coords)
    catch
        # Continue with component-wise arguments
    end

    x = _coord_value(coords, "x")
    y = _coord_value(coords, "y")
    z = _coord_value(coords, "z")
    r = _coord_value(coords, "r")
    theta = _coord_value(coords, "θ")
    if theta === nothing
        theta = _coord_value(coords, "theta")
    end
    phi = _coord_value(coords, "φ")
    if phi === nothing
        phi = _coord_value(coords, "phi")
    end

    try
        if x !== nothing && y !== nothing && z !== nothing
            return func(x, y, z)
        elseif x !== nothing && y !== nothing
            return func(x, y)
        elseif r !== nothing && theta !== nothing && phi !== nothing
            return func(r, theta, phi)
        elseif r !== nothing && theta !== nothing
            return func(r, theta)
        elseif x !== nothing
            return func(x)
        elseif r !== nothing
            return func(r)
        elseif theta !== nothing
            return func(theta)
        elseif phi !== nothing
            return func(phi)
        else
            return func()
        end
    catch
        try
            return func()
        catch
            return func
        end
    end
end

function update_time_dependent_bcs!(manager::BoundaryConditionManager, current_time)
    """Update all time-dependent boundary conditions for current time"""
    
    if isempty(manager.time_dependent_bcs)
        return manager
    end
    
    start_time = time()
    @debug "Updating $(length(manager.time_dependent_bcs)) time-dependent BCs at t=$current_time"
    
    # Use coordinate fields for BCs that are both time and space dependent
    coords = manager.coordinate_fields

    for bc_index in manager.time_dependent_bcs
        bc = manager.conditions[bc_index]

        # Update the BC value based on current time
        # Cache evaluated values for repeated use
        cache_key = "bc_$(bc_index)_t_$(current_time)"

        if isa(bc, DirichletBC) && bc.is_time_dependent
            if !haskey(manager.bc_cache, cache_key)
                new_value = evaluate_bc_value(manager, bc, current_time, coords)
                manager.bc_cache[cache_key] = new_value
            end
            @debug "Updated Dirichlet BC for $(bc.field)"

        elseif isa(bc, NeumannBC) && bc.is_time_dependent
            if !haskey(manager.bc_cache, cache_key)
                new_value = evaluate_bc_value(manager, bc, current_time, coords)
                manager.bc_cache[cache_key] = new_value
            end
            @debug "Updated Neumann BC for $(bc.field)"

        elseif isa(bc, RobinBC) && bc.is_time_dependent
            # For Robin BCs, check component keys since we store alpha, beta, value separately
            robin_cache_key = "$(cache_key)_value"
            if !haskey(manager.bc_cache, robin_cache_key)
                alpha, beta, value = evaluate_bc_value(manager, bc, current_time, coords)
                for (comp_name, comp_value) in [("alpha", alpha), ("beta", beta), ("value", value)]
                    manager.bc_cache["$(cache_key)_$(comp_name)"] = comp_value
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

function get_current_bc_value(manager::BoundaryConditionManager, bc_index::Int, current_time)
    """Retrieve the most recently cached value for a time-dependent BC."""
    cache_key = "bc_$(bc_index)_t_$(current_time)"
    bc = manager.conditions[bc_index]

    if isa(bc, RobinBC)
        alpha = get(manager.bc_cache, "$(cache_key)_alpha", nothing)
        beta = get(manager.bc_cache, "$(cache_key)_beta", nothing)
        value = get(manager.bc_cache, "$(cache_key)_value", nothing)
        return (alpha, beta, value)
    else
        return get(manager.bc_cache, cache_key, nothing)
    end
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
    """Clear all boundary conditions and associated caches"""
    empty!(manager.conditions)
    empty!(manager.tau_fields)
    empty!(manager.lift_operators)
    empty!(manager.time_dependent_bcs)
    empty!(manager.space_dependent_bcs)
    empty!(manager.bc_equation_indices)
    empty!(manager.bc_cache)
    empty!(manager.workspace)
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

function evaluate_space_dependent_bcs!(manager::BoundaryConditionManager, coordinates::Dict,
                                      current_time::Real=0.0)
    """Evaluate space-dependent boundary conditions at the given time and coordinates"""

    if isempty(manager.space_dependent_bcs)
        return manager
    end

    start_time = time()

    @debug "Evaluating $(length(manager.space_dependent_bcs)) space-dependent BCs at t=$current_time"

    for bc_index in manager.space_dependent_bcs
        bc = manager.conditions[bc_index]

        coord_hash = hash((coordinates, current_time))
        cache_key = "bc_$(bc_index)_spatial_$(coord_hash)"

        # For Robin BCs, use component key for cache check since they store at component keys
        check_key = isa(bc, RobinBC) ? "$(cache_key)_value" : cache_key

        if !haskey(manager.bc_cache, check_key)
            if isa(bc, DirichletBC) && bc.is_space_dependent
                new_value = evaluate_bc_value(manager, bc, current_time, coordinates)
                manager.bc_cache[cache_key] = new_value
                manager.performance_stats.cache_misses += 1
            elseif isa(bc, NeumannBC) && bc.is_space_dependent
                new_value = evaluate_bc_value(manager, bc, current_time, coordinates)
                manager.bc_cache[cache_key] = new_value
                manager.performance_stats.cache_misses += 1
            elseif isa(bc, RobinBC) && bc.is_space_dependent
                alpha, beta, value = evaluate_bc_value(manager, bc, current_time, coordinates)
                for (comp_name, comp_value) in [("alpha", alpha), ("beta", beta), ("value", value)]
                    manager.bc_cache["$(cache_key)_$(comp_name)"] = comp_value
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
       register_tau_field!, get_tau_field, auto_generate_tau_fields!, validate_tau_fields!,
       create_lift_operator, apply_lift, bc_to_equation, get_boundary_basis,
       apply_boundary_conditions!, validate_boundary_conditions,
       get_bc_count_by_type, get_required_tau_fields, clear_boundary_conditions!,
       TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
       set_time_variable!, add_coordinate_field!, evaluate_bc_value, evaluate_expression,
       update_time_dependent_bcs!, requires_bc_update, has_time_dependent_bcs, has_space_dependent_bcs,
       is_time_dependent, is_space_dependent, get_current_bc_value,
       register_coordinate_info!, register_domain_info!,
       log_bc_performance, evaluate_space_dependent_bcs!, clear_bc_cache!
