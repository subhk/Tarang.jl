"""
Problem formulation classes
"""

using LinearAlgebra
using SparseArrays

abstract type Problem end

mutable struct IVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    time::Union{Nothing, ScalarField}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
    # Stochastic forcing: Dict mapping variable index to StochasticForcing
    # Forcing is automatically added to RHS during timestepping
    stochastic_forcings::Dict{Any, Any}
    # Temporal filters: Dict mapping variable symbol to (filter, source_symbol)
    # Filters are updated each timestep with the source variable's data
    temporal_filters::Dict{Symbol, Any}
end

mutable struct LBVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

mutable struct NLBVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

mutable struct EVP <: Problem
    variables::Vector{Operand}
    equations::Vector{String}
    boundary_conditions::Vector{String}
    parameters::Dict{String, Any}
    namespace::Dict{String, Any}
    eigenvalue::Union{Nothing, Symbol}
    domain::Union{Nothing, Domain}
    bc_manager::BoundaryConditionManager
    equation_data::Vector{Dict{String, Any}}
end

function build_problem_namespace(variables::Vector{<:Operand}, user_ns::Union{Nothing, AbstractDict{String}})
    ns = Dict{String, Any}()
    
    # Lowest priority: operator registries (parseables first, then aliases)
    for (name, value) in OPERATOR_PARSEABLES
        ns[name] = value
    end
    for (name, value) in OPERATOR_ALIASES
        ns[name] = value
    end
    
    # User-supplied namespace overrides registry entries
    if user_ns !== nothing
        for (name, value) in user_ns
            ns[string(name)] = value
        end
    end
    
    # Highest priority: problem variables by name
    for var in variables
        if hasproperty(var, :name)
            name = getfield(var, :name)
            if name !== nothing && !isempty(name)
                ns[name] = var
            end
        end
    end
    
    return ns
end

function _build_ivp(variables::Vector{<:Operand}; namespace::Union{Nothing, AbstractDict{String}}=nothing)
    vars = Vector{Operand}(variables)  # Convert to Vector{Operand}
    ns = build_problem_namespace(vars, namespace)
    return IVP(vars, String[], String[], Dict{String, Any}(), ns,
               nothing, nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}(),
               Dict{Any, Any}(),      # Empty stochastic_forcings dict
               Dict{Symbol, Any}())   # Empty temporal_filters dict
end

function _build_lbvp(variables::Vector{<:Operand}; namespace::Union{Nothing, AbstractDict{String}}=nothing)
    vars = Vector{Operand}(variables)  # Convert to Vector{Operand}
    ns = build_problem_namespace(vars, namespace)
    return LBVP(vars, String[], String[], Dict{String, Any}(), ns,
                nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

function _build_nlbvp(variables::Vector{<:Operand}; namespace::Union{Nothing, AbstractDict{String}}=nothing)
    vars = Vector{Operand}(variables)  # Convert to Vector{Operand}
    ns = build_problem_namespace(vars, namespace)
    return NLBVP(vars, String[], String[], Dict{String, Any}(), ns,
                 nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

function _build_evp(variables::Vector{<:Operand}; eigenvalue::Union{Nothing, Symbol}=nothing, namespace::Union{Nothing, AbstractDict{String}}=nothing)
    vars = Vector{Operand}(variables)  # Convert to Vector{Operand}
    ns = build_problem_namespace(vars, namespace)
    return EVP(vars, String[], String[], Dict{String, Any}(), ns,
               eigenvalue, nothing, BoundaryConditionManager(), Vector{Dict{String, Any}}())
end

const _IVP_constructor = _build_ivp
const _LBVP_constructor = _build_lbvp
const _NLBVP_constructor = _build_nlbvp
const _EVP_constructor = _build_evp

function IVP(variables::Vector{<:Operand}; kwargs...)
    return multiclass_new(IVP, variables; kwargs...)
end

function LBVP(variables::Vector{<:Operand}; kwargs...)
    return multiclass_new(LBVP, variables; kwargs...)
end

function NLBVP(variables::Vector{<:Operand}; kwargs...)
    return multiclass_new(NLBVP, variables; kwargs...)
end

function EVP(variables::Vector{<:Operand}; kwargs...)
    return multiclass_new(EVP, variables; kwargs...)
end

_problem_builder(::Type{IVP}) = _IVP_constructor
_problem_builder(::Type{LBVP}) = _LBVP_constructor
_problem_builder(::Type{NLBVP}) = _NLBVP_constructor
_problem_builder(::Type{EVP}) = _EVP_constructor
_problem_builder(::Type{T}) where {T<:Problem} = error("No problem builder registered for type $(T)")

function _validate_problem_kwargs(kwargs::NamedTuple)
    if haskey(kwargs, :namespace)
        ns = kwargs[:namespace]
        if !(ns isa AbstractDict || ns === nothing)
            throw(ArgumentError("namespace keyword must be a dictionary or nothing"))
        end
    end
end

function _validate_problem_variables(args::Tuple, kwargs::NamedTuple)
    if isempty(args)
        throw(ArgumentError("Problem constructors require a variables vector"))
    end
    variables = args[1]
    if !(variables isa Vector)
        throw(ArgumentError("Problem variables must be provided as a Vector"))
    end
    if !all(var -> var isa Operand, variables)
        throw(ArgumentError("All problem variables must be Operands"))
    end
    _validate_problem_kwargs(kwargs)
end

function dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Problem}
    if length(args) != 1
        throw(ArgumentError("$(T) expects a single Vector{Operand} argument"))
    end
    return (args, kwargs)
end

function dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Problem}
    _validate_problem_variables(args, kwargs)
    return true
end

function dispatch_check(::Type{EVP}, args::Tuple, kwargs::NamedTuple)
    _validate_problem_variables(args, kwargs)
    if haskey(kwargs, :eigenvalue)
        eigen = kwargs[:eigenvalue]
        if !(eigen === nothing || eigen isa Symbol)
            throw(ArgumentError("Eigenvalue keyword must be a Symbol or nothing"))
        end
    end
    return true
end

function invoke_constructor(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Problem}
    builder = _problem_builder(T)
    variables = args[1]  # Don't copy here, _build_* functions handle conversion
    return builder(variables; kwargs...)
end

# ============================================================================
# Boundary condition string parsing
# ============================================================================

"""
    parse_bc_string(bc_string::String)

Parse Dedalus-style boundary condition string into components.

# Supported formats:
- `"u(z=0) = 0"` → ("u", "z", 0.0, 0.0)
- `"T(z=1.0) = 1"` → ("T", "z", 1.0, 1.0)
- `"u(x=-1) = sin(y)"` → ("u", "x", -1.0, "sin(y)")

Returns: (field_name, coordinate, position, value)
"""
function parse_bc_string(bc_string::String)
    # Remove whitespace
    s = replace(bc_string, " " => "")

    # Match pattern: field(coord=pos)=value
    # e.g., "u(z=0)=0" or "T(z=1.0)=sin(x)"
    pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)=(.+)$"
    m = match(pattern, s)

    if m === nothing
        throw(ArgumentError("Invalid BC string format: '$bc_string'. Expected format: 'field(coord=pos) = value'"))
    end

    field_name = String(m.captures[1])
    coordinate = String(m.captures[2])
    pos_str = String(m.captures[3])
    val_str = String(m.captures[4])

    # Parse position as number
    position = try
        parse(Float64, pos_str)
    catch
        throw(ArgumentError("Cannot parse position '$pos_str' as number in BC: '$bc_string'"))
    end

    # Parse value - could be number or expression string
    value = try
        parse(Float64, val_str)
    catch
        # Keep as string expression for space/time dependent BCs
        val_str
    end

    return (field_name, coordinate, position, value)
end

"""
    parse_neumann_bc_string(bc_string::String)

Parse Neumann BC string like "∂z(u)(z=0) = 0" into components.

Returns: (field_name, coordinate, position, value)
"""
function parse_neumann_bc_string(bc_string::String)
    # Remove whitespace
    s = replace(bc_string, " " => "")

    # Match pattern: ∂<coord>(<field>)(<coord>=<pos>)=<value>
    # e.g., "∂z(u)(z=0)=0"
    pattern = r"^∂([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)=(.+)$"
    m = match(pattern, s)

    if m === nothing
        # Try simpler format: field(coord=pos) = value (same as Dirichlet but caller knows it's Neumann)
        return parse_bc_string(bc_string)
    end

    deriv_coord = String(m.captures[1])
    field_name = String(m.captures[2])
    bc_coord = String(m.captures[3])
    pos_str = String(m.captures[4])
    val_str = String(m.captures[5])

    # Verify derivative coordinate matches BC coordinate
    if deriv_coord != bc_coord
        @warn "Derivative coordinate '$deriv_coord' differs from BC coordinate '$bc_coord'"
    end

    # Parse position
    position = try
        parse(Float64, pos_str)
    catch
        throw(ArgumentError("Cannot parse position '$pos_str' as number in BC: '$bc_string'"))
    end

    # Parse value
    value = try
        parse(Float64, val_str)
    catch
        val_str
    end

    return (field_name, bc_coord, position, value)
end

"""
    parse_robin_bc_string(bc_string::String)

Parse Robin BC string like "1.0*T(z=0) + 0.5*∂z(T)(z=0) = 1.0" into components.

Returns: (field_name, coordinate, position, alpha, beta, value)
"""
function parse_robin_bc_string(bc_string::String)
    # Remove whitespace
    s = replace(bc_string, " " => "")

    # Match pattern: alpha*field(coord=pos)+beta*∂<coord>(field)(<coord>=pos)=value
    # e.g., "1.0*T(z=0)+0.5*∂z(T)(z=0)=1.0"
    pattern = r"^([0-9.eE+-]+)\*([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)\+([0-9.eE+-]+)\*∂([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)=(.+)$"
    m = match(pattern, s)

    if m === nothing
        throw(ArgumentError("Invalid Robin BC format: '$bc_string'. Expected: 'alpha*field(coord=pos) + beta*∂<coord>(field)(coord=pos) = value'"))
    end

    alpha_str = String(m.captures[1])
    field_name1 = String(m.captures[2])
    field_coord = String(m.captures[3])
    field_pos_str = String(m.captures[4])
    beta_str = String(m.captures[5])
    deriv_coord = String(m.captures[6])
    field_name2 = String(m.captures[7])
    deriv_bc_coord = String(m.captures[8])
    deriv_pos_str = String(m.captures[9])
    val_str = String(m.captures[10])

    # Verify field names match
    if field_name1 != field_name2
        throw(ArgumentError("Field names must match in Robin BC: '$field_name1' vs '$field_name2'"))
    end

    # Verify coordinates and positions match
    if field_coord != deriv_bc_coord
        throw(ArgumentError("Coordinates must match in Robin BC: '$field_coord' vs '$deriv_bc_coord'"))
    end
    if field_pos_str != deriv_pos_str
        throw(ArgumentError("Positions must match in Robin BC: '$field_pos_str' vs '$deriv_pos_str'"))
    end

    # Parse coefficients
    alpha = parse(Float64, alpha_str)
    beta = parse(Float64, beta_str)

    # Parse position
    position = try
        parse(Float64, field_pos_str)
    catch
        throw(ArgumentError("Cannot parse position '$field_pos_str' as number in BC: '$bc_string'"))
    end

    # Parse value
    value = try
        parse(Float64, val_str)
    catch
        val_str
    end

    return (field_name1, field_coord, position, alpha, beta, value)
end

"""
    parse_stress_free_bc_string(bc_string::String)

Parse stress-free BC string like "u(z=0)" into components.
The "stress-free" part is implied by the function name and optional in the string.

Returns: (velocity_field, coordinate, position)
"""
function parse_stress_free_bc_string(bc_string::String)
    # Remove extra whitespace but keep single spaces
    s = strip(bc_string)

    # Primary format: field(coord=pos) - simple and clean
    # e.g., "u(z=0)" or "u(z=1)"
    pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)$"
    m = match(pattern, s)

    if m === nothing
        throw(ArgumentError("Invalid stress-free BC format: '$bc_string'. Expected: 'field(coord=pos)', e.g., 'u(z=0)'"))
    end

    velocity_field = String(m.captures[1])
    coordinate = String(m.captures[2])
    pos_str = String(m.captures[3])

    # Parse position
    position = try
        parse(Float64, pos_str)
    catch
        throw(ArgumentError("Cannot parse position '$pos_str' as number in BC: '$bc_string'"))
    end

    return (velocity_field, coordinate, position)
end

# ============================================================================
# Problem building and manipulation
# ============================================================================

function add_equation!(problem::Problem, equation::String)
    """Add equation to problem"""
    push!(problem.equations, equation)
end

"""
    add_stochastic_forcing!(problem::IVP, variable::Symbol, forcing)

Add stochastic forcing to a variable in the IVP. The forcing will be automatically
applied to the RHS during timestepping.

## Why This Exists

Stochastic forcing requires special handling for correct Stratonovich calculus:
- Forcing must be generated ONCE at the beginning of each timestep
- The SAME forcing value must be used across all RK substeps
- This ensures correct statistical properties for white-in-time noise

By registering forcing with `add_stochastic_forcing!`, the timestepper handles
all of this automatically - you don't need to manually call `generate_forcing!`
or `apply_forcing!` in your time loop.

## Arguments

- `problem::IVP`: The initial value problem
- `variable::Symbol`: The variable name (as Symbol) whose RHS receives the forcing
- `forcing`: The stochastic forcing configuration

## Example

```julia
# Create problem
problem = IVP([ω])
add_equation!(problem, "∂t(ω) + μ*ω - ν*Δ(ω) = -J(ψ, ω)")

# Create and register forcing - it will be added to RHS automatically!
forcing = StochasticForcing(
    field_size = (256, 256),
    energy_injection_rate = 0.1,
    k_forcing = 10.0,
    dt = dt
)
add_stochastic_forcing!(problem, :ω, forcing)

# Create solver and run - forcing is handled automatically
solver = InitialValueSolver(problem, RK443(); dt=dt)
for step in 1:nsteps
    step!(solver)  # Forcing is generated and applied internally!
end
```

See also: [`StochasticForcing`](@ref), [`generate_forcing!`](@ref)
"""
function add_stochastic_forcing!(problem::IVP, variable::Symbol, forcing)
    # Find variable index by name
    var_name = String(variable)
    var_idx = nothing
    for (i, var) in enumerate(problem.variables)
        if hasproperty(var, :name) && getfield(var, :name) == var_name
            var_idx = i
            break
        end
    end

    if var_idx === nothing
        throw(ArgumentError("Variable ':$variable' not found in problem variables"))
    end

    # Store in stochastic_forcings dict keyed by variable index
    problem.stochastic_forcings[var_idx] = forcing

    @info "Registered stochastic forcing for variable :$variable (index: $var_idx)"
    return problem
end

"""
    has_stochastic_forcing(problem::IVP) -> Bool

Check if the problem has any registered stochastic forcings.
"""
has_stochastic_forcing(problem::IVP) = !isempty(problem.stochastic_forcings)

"""
    get_stochastic_forcing(problem::IVP, var_index::Int)

Get the stochastic forcing for a variable (by index), or nothing if none registered.
"""
function get_stochastic_forcing(problem::IVP, var_index::Int)
    return get(problem.stochastic_forcings, var_index, nothing)
end

# ============================================================================
# Temporal Filter Integration
# ============================================================================

"""
    add_temporal_filter!(problem::IVP, :filter_name, filter, :source_variable)

Register a temporal filter with the problem. The filter will be automatically
updated each timestep with data from the source variable.

## Arguments

- `problem::IVP`: The initial value problem
- `filter_name::Symbol`: Name to identify this filter (used to access the mean)
- `filter::TemporalFilter`: The temporal filter (ExponentialMean, ButterworthFilter, etc.)
- `source_variable::Symbol`: The variable whose data feeds into the filter

## How It Works

At each timestep, the timestepper automatically:
1. Gets the current data from the source variable
2. Calls `update!(filter, data, dt)` to advance the filter state
3. The filtered mean is available via `get_mean(filter)`

The filtered mean can be accessed in your equations or post-processing via
`get_temporal_filter(problem, :filter_name)`.

## Example

```julia
# Create a problem with vorticity
problem = IVP([ω])
add_equation!(problem, "∂t(ω) + μ*ω - ν*Δ(ω) = -J(ψ, ω)")

# Create a Butterworth filter for computing Lagrangian mean
filter = ButterworthFilter((n, n); α=0.1)  # α = 1/averaging_time

# Register filter to track the filtered mean of ω
add_temporal_filter!(problem, :ω_mean, filter, :ω)

# Create solver and run
solver = InitialValueSolver(problem, RK443(); dt=dt)
for step in 1:nsteps
    step!(solver)  # Filter is updated automatically!

    # Access the filtered mean
    ω_bar = get_mean(get_temporal_filter(problem, :ω_mean))
end
```

See also: [`ExponentialMean`](@ref), [`ButterworthFilter`](@ref), [`get_mean`](@ref)
"""
function add_temporal_filter!(problem::IVP, filter_name::Symbol, filter, source_variable::Symbol)
    # Validate source variable exists
    var_name = String(source_variable)
    found = false
    for var in problem.variables
        if hasproperty(var, :name) && getfield(var, :name) == var_name
            found = true
            break
        end
    end

    if !found
        throw(ArgumentError("Source variable ':$source_variable' not found in problem variables"))
    end

    # Store as (filter, source_symbol)
    problem.temporal_filters[filter_name] = (filter=filter, source=source_variable)

    @info "Registered temporal filter :$filter_name tracking :$source_variable"
    return problem
end

"""
    has_temporal_filters(problem::IVP) -> Bool

Check if the problem has any registered temporal filters.
"""
has_temporal_filters(problem::IVP) = !isempty(problem.temporal_filters)

"""
    get_temporal_filter(problem::IVP, filter_name::Symbol)

Get the temporal filter by name, or nothing if not registered.
Returns a NamedTuple with fields `filter` and `source`.
"""
function get_temporal_filter(problem::IVP, filter_name::Symbol)
    return get(problem.temporal_filters, filter_name, nothing)
end

"""
    get_all_temporal_filters(problem::IVP)

Get all registered temporal filters as a Dict.
"""
get_all_temporal_filters(problem::IVP) = problem.temporal_filters

function add_bc!(problem::Problem, bc::String)
    """Add boundary condition to problem (legacy string interface)"""
    push!(problem.boundary_conditions, bc)
end

# Enhanced boundary condition functions
function add_bc!(problem::Problem, bc::AbstractBoundaryCondition)
    """Add structured boundary condition to problem"""
    add_bc!(problem.bc_manager, bc)
    
    # Also add to legacy string list for compatibility
    eq = bc_to_equation(problem.bc_manager, bc)
    if isa(eq, Vector)
        append!(problem.boundary_conditions, eq)
    else
        push!(problem.boundary_conditions, eq)
    end
    
    return bc
end

# Note: add_dirichlet_bc!, add_neumann_bc!, add_robin_bc!, add_stress_free_bc!, add_no_slip_bc!
# have been removed. Use add_equation!() for all equations including boundary conditions.
# Example: add_equation!(problem, "u(z=0) = 0")  # Dirichlet BC
# Example: add_equation!(problem, "dz(u)(z=1) = 0")  # Neumann BC
# This matches Dedalus behavior where add_equation() handles everything.

function register_tau_field!(problem::Problem, name::String, field)
    """Register tau field for boundary condition enforcement"""
    register_tau_field!(problem.bc_manager, name, field)
    
    # Add to problem variables if not already present
    if !any(var -> (isa(var, ScalarField) && var.name == name), problem.variables)
        push!(problem.variables, field)
    end
    
    return problem
end

function set_parameter!(problem::Problem, name::String, value)
    """Set parameter value"""
    problem.parameters[name] = value
end

function get_parameter(problem::Problem, name::String, default=nothing)
    """Get parameter value"""
    return get(problem.parameters, name, default)
end

# Equation parsing following structure
function parse_equation(equation::String, namespace::Dict{String, Any})
    """
    Parse equation string into operator expressions following formulation requirements.
    
    Requires:
    - LHS: Linear terms only (first-order in time derivatives, linear in variables)
    - RHS: Nonlinear terms, time-dependent terms, non-constant coefficients
    - Form: M·∂tX + L·X = F(X,t)
    
    Following problems:add_equation pattern (problems:65-80).
    """
    
    try
        # Split equation into LHS and RHS expressions
        if isa(equation, String)
            # Parse string-valued equations following pattern
            LHS_str, RHS_str = split_equation(equation)
            
            # Parse LHS (should contain only linear terms)
            LHS = parse_linear_expression(LHS_str, namespace)
            
            # Parse RHS (can contain nonlinear terms)
            RHS = parse_expression(RHS_str, namespace)
            
            # Validate equation structure following requirements
            validate_equation_structure(LHS, RHS, equation)
            
            return LHS, RHS
        else
            throw(ArgumentError("Equation must be a string"))
        end
        
    catch e
        @error "Failed to parse equation: $equation" exception=e
        # Return fallback expressions as proper operator types
        return UnknownOperator(equation), ZeroOperator()
    end
end

function parse_linear_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    """
    Parse LHS expression ensuring it contains only linear terms.
    
    Linear terms allowed on LHS:
    - Time derivatives (first-order): dt(u)
    - Linear spatial derivatives: d(u, x), lap(u)
    - Constant coefficients: 2*dt(u), -viscosity*lap(u)
    - Linear combinations: dt(u) + diffusion*lap(u)
    
    Non-constant coefficients should be moved to RHS.
    """
    
    expr = parse_expression(expr_str, namespace)
    
    # Validate linearity (this would need proper implementation)
    if !is_linear_expression(expr, namespace)
        @warn "Non-linear terms detected on LHS of equation. Consider moving to RHS: $expr_str"
        @warn "Requires: Linear terms (LHS) = Nonlinear terms (RHS)"
    end
    
    return expr
end

function is_linear_expression(expr, namespace::Dict{String, Any})
    """
    Check if expression contains only linear terms suitable for LHS.

    Linear terms are those that can be represented as matrix operations on the
    solution vector. This includes:
    1. Terms linear in dependent variables (∂t(u), ∇²u, ν·u, etc.)
    2. First-order time derivatives only
    3. Constant coefficients (spatially-varying coefficients require NCC treatment)

    Non-linear terms (u·∇u, u², etc.) must go to RHS.
    """

    # Zero and constant expressions are trivially linear
    if isa(expr, ZeroOperator) || isa(expr, ConstantOperator)
        return true
    end

    # Time derivatives
    if isa(expr, TimeDerivative)
        # First-order time derivatives are linear, higher orders need special treatment
        if hasfield(typeof(expr), :order)
            return expr.order == 1
        end
        return true
    end

    # Spatial differential operators are linear
    if isa(expr, Differentiate) || isa(expr, Laplacian) || isa(expr, Gradient) ||
       isa(expr, Divergence) || isa(expr, Curl)
        # The operator itself is linear; check if operand is a variable or linear expr
        if hasfield(typeof(expr), :operand)
            operand = expr.operand
            # Direct variable reference is linear
            if is_variable_reference(operand, namespace)
                return true
            end
            # Linear combination of variables is also linear
            return is_linear_expression(operand, namespace)
        end
        return true
    end

    # Addition/subtraction preserves linearity
    if isa(expr, AddOperator) || isa(expr, SubtractOperator)
        left_linear = is_linear_expression(expr.left, namespace)
        right_linear = is_linear_expression(expr.right, namespace)
        return left_linear && right_linear
    end

    # Multiplication is linear only if one factor is a constant coefficient
    if isa(expr, MultiplyOperator)
        left_constant = is_constant_coefficient(expr.left, namespace)
        right_constant = is_constant_coefficient(expr.right, namespace)
        # At least one factor must be constant for linearity
        if left_constant
            return is_linear_expression(expr.right, namespace)
        elseif right_constant
            return is_linear_expression(expr.left, namespace)
        else
            # Product of two non-constant terms is nonlinear
            return false
        end
    end

    # Direct variable references are linear
    if is_variable_reference(expr, namespace)
        return true
    end

    # Other operations (powers, products of variables, nonlinear functions) are nonlinear
    return false
end

function is_variable_reference(expr, namespace::Dict{String, Any})
    """Check if expression is a direct reference to a problem variable."""
    if hasfield(typeof(expr), :name) && isa(getfield(expr, :name), String)
        var_name = getfield(expr, :name)
        # Check if this name corresponds to a variable in the namespace
        if haskey(namespace, var_name)
            val = namespace[var_name]
            return isa(val, ScalarField) || isa(val, VectorField) || isa(val, TensorField)
        end
    end
    return false
end

function is_constant_coefficient(expr, namespace::Dict{String, Any})
    """
    Check if expression represents a constant coefficient.
    Non-constant coefficients should be moved to RHS per requirements.
    """
    
    if isa(expr, ConstantOperator)
        return true
    elseif isa(expr, UnknownOperator)
        # Check if it's a known constant parameter
        expr_str = expr.expression
        if haskey(namespace, expr_str)
            param = namespace[expr_str]
            # This would need proper type checking for constant parameters
            return isa(param, Number) || isa(param, AbstractFloat) || isa(param, Complex)
        end
    end
    
    return false
end

function validate_equation_structure(LHS, RHS, original_equation::String)
    """
    Validate that equation follows structure requirements.
    
    Requirements:
    1. LHS must be linear in dependent variables
    2. LHS must be first-order in time derivatives  
    3. RHS can contain nonlinear terms
    4. Non-constant coefficients should be on RHS
    """
    
    # Check for temporal derivatives on RHS (not allowed)
    if contains_time_derivatives(RHS)
        throw(ArgumentError("Time derivatives found on RHS of equation: $original_equation. " *
                           "Requires all time derivatives on LHS."))
    end
    
    # Check for proper linear structure on LHS
    is_valid_lhs, lhs_info = is_proper_lhs_structure(LHS)
    if !is_valid_lhs
        @warn "LHS may not follow linear structure: $original_equation"
        @warn "Ensure LHS contains only: dt(vars), linear spatial derivatives, constant coefficients"
        if lhs_info[:error_message] !== nothing
            @warn "Issue: $(lhs_info[:error_message])"
        end
    end
    
    return true
end

function contains_time_derivatives(expr)
    """Check if expression tree contains any time derivatives."""
    if expr === nothing
        return false
    elseif isa(expr, TimeDerivative)
        return true
    elseif isa(expr, AddOperator) || isa(expr, SubtractOperator) || isa(expr, MultiplyOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, DivideOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, PowerOperator)
        return contains_time_derivatives(expr.base) || contains_time_derivatives(expr.exponent)
    elseif isa(expr, NegateOperator)
        return contains_time_derivatives(expr.operand)
    elseif isa(expr, IndexOperator)
        return contains_time_derivatives(expr.array)
    elseif hasfield(typeof(expr), :operand)
        # Handle operators like Laplacian, Differentiate, Gradient, etc.
        return contains_time_derivatives(expr.operand)
    else
        return false
    end
end

function is_proper_lhs_structure(expr)
    """
    Check if LHS has proper structure for matrix formulation.
    Should be of the form: M·∂tX + L·X where M and L are linear operators.

    The LHS of a equation must satisfy:
    1. All terms must be linear in the state variables
    2. Time derivatives must be first-order only (∂t, not ∂t²)
    3. Spatial operators must be linear (derivatives, Laplacian, etc.)
    4. Coefficients must be constant (not space/time-dependent)
    5. No products of state variables (those go to RHS as nonlinear terms)

    Returns a tuple (is_valid::Bool, info::Dict) where info contains:
    - :has_time_derivative => whether the expression contains ∂t terms
    - :has_spatial_operators => whether spatial operators are present
    - :is_linear => whether all terms are linear
    - :error_message => description of any structural issues
    """

    info = Dict{Symbol, Any}(
        :has_time_derivative => false,
        :has_spatial_operators => false,
        :is_linear => true,
        :error_message => nothing,
        :time_derivative_order => 0,
        :dependent_variables => Set{String}()
    )

    # Recursively analyze the expression structure
    is_valid = _analyze_lhs_structure!(expr, info, Dict{String, Any}())

    return (is_valid, info)
end

function _analyze_lhs_structure!(expr, info::Dict{Symbol, Any}, namespace::Dict{String, Any})
    """
    Recursively analyze expression structure for LHS validity.
    """

    # Handle nothing/empty case
    if expr === nothing
        return true
    end

    # Zero operator is always valid
    if isa(expr, ZeroOperator)
        return true
    end

    # Constant operator is valid
    if isa(expr, ConstantOperator)
        return true
    end

    # Numeric literals are valid coefficients
    if isa(expr, Number)
        return true
    end

    # Time derivative: check order and operand
    if isa(expr, TimeDerivative)
        info[:has_time_derivative] = true

        # Only first-order time derivatives allowed on LHS
        if expr.order > 1
            info[:is_linear] = false
            info[:error_message] = "Higher-order time derivatives (order=$(expr.order)) not allowed on LHS"
            return false
        end

        info[:time_derivative_order] = max(info[:time_derivative_order], expr.order)

        # The operand of time derivative should be a state variable or linear operator on it
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    # Spatial differential operators: linear by nature
    if isa(expr, Differentiate)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Laplacian)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Gradient)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Divergence)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    if isa(expr, Curl)
        info[:has_spatial_operators] = true
        return _analyze_lhs_operand!(expr.operand, info, namespace)
    end

    # Addition/subtraction: both sides must be valid
    if isa(expr, AddOperator)
        left_valid = _analyze_lhs_structure!(expr.left, info, namespace)
        right_valid = _analyze_lhs_structure!(expr.right, info, namespace)
        return left_valid && right_valid
    end

    if isa(expr, SubtractOperator)
        left_valid = _analyze_lhs_structure!(expr.left, info, namespace)
        right_valid = _analyze_lhs_structure!(expr.right, info, namespace)
        return left_valid && right_valid
    end

    # Negation: operand must be valid
    if isa(expr, NegateOperator)
        return _analyze_lhs_structure!(expr.operand, info, namespace)
    end

    # Multiplication: exactly one factor must be a constant coefficient
    if isa(expr, MultiplyOperator)
        left_is_const = _is_constant_coefficient_strict(expr.left, namespace)
        right_is_const = _is_constant_coefficient_strict(expr.right, namespace)

        if left_is_const && right_is_const
            # Both constant - this is fine (just a constant)
            return true
        elseif left_is_const
            # Left is constant, right must be linear in variables
            return _analyze_lhs_structure!(expr.right, info, namespace)
        elseif right_is_const
            # Right is constant, left must be linear in variables
            return _analyze_lhs_structure!(expr.left, info, namespace)
        else
            # Neither is constant - this is a nonlinear term (product of variables)
            info[:is_linear] = false
            info[:error_message] = "Product of non-constant terms on LHS (nonlinear)"
            return false
        end
    end

    # Division: numerator must be linear, denominator must be constant
    if isa(expr, DivideOperator)
        if !_is_constant_coefficient_strict(expr.right, namespace)
            info[:is_linear] = false
            info[:error_message] = "Division by non-constant on LHS"
            return false
        end
        return _analyze_lhs_structure!(expr.left, info, namespace)
    end

    # Power operator: only valid if base is constant OR exponent is 1
    if isa(expr, PowerOperator)
        exp_val = _get_constant_value(expr.exponent)
        if exp_val !== nothing && exp_val == 1
            return _analyze_lhs_structure!(expr.base, info, namespace)
        elseif _is_constant_coefficient_strict(expr.base, namespace)
            return true
        else
            info[:is_linear] = false
            info[:error_message] = "Power of non-constant with exponent ≠ 1 on LHS"
            return false
        end
    end

    # ScalarField: this is a state variable - valid as linear term
    if isa(expr, ScalarField)
        push!(info[:dependent_variables], expr.name)
        return true
    end

    # VectorField: state variable
    if isa(expr, VectorField)
        push!(info[:dependent_variables], expr.name)
        return true
    end

    # Index operator on a valid structure
    if isa(expr, IndexOperator)
        return _analyze_lhs_structure!(expr.array, info, namespace)
    end

    # Unknown operator: check if it's in the namespace as a constant
    if isa(expr, UnknownOperator)
        if haskey(namespace, expr.expression)
            val = namespace[expr.expression]
            if isa(val, Number) || isa(val, ConstantOperator)
                return true
            elseif isa(val, ScalarField) || isa(val, VectorField)
                push!(info[:dependent_variables], expr.expression)
                return true
            end
        end
        # Unknown expression - be conservative and mark as potentially invalid
        @debug "Unknown expression in LHS: $(expr.expression)"
        return true  # Allow it but could be stricter
    end

    # Nonlinear operators are not allowed on LHS
    if isa(expr, NonlinearOperator)
        info[:is_linear] = false
        info[:error_message] = "Nonlinear operator on LHS"
        return false
    end

    # Default: if we can't classify it, check if it looks linear
    return is_linear_expression(expr, namespace)
end

function _analyze_lhs_operand!(operand, info::Dict{Symbol, Any}, namespace::Dict{String, Any})
    """
    Analyze the operand of a differential operator.
    """
    if isa(operand, ScalarField)
        push!(info[:dependent_variables], operand.name)
        return true
    elseif isa(operand, VectorField)
        push!(info[:dependent_variables], operand.name)
        return true
    else
        return _analyze_lhs_structure!(operand, info, namespace)
    end
end

function _is_constant_coefficient_strict(expr, namespace::Dict{String, Any})
    """
    Strictly check if expression is a constant coefficient.
    Constants are: numbers, ConstantOperator, or namespace entries that are constant.
    """
    if isa(expr, Number)
        return true
    end

    if isa(expr, ConstantOperator)
        return true
    end

    if isa(expr, ZeroOperator)
        return true
    end

    if isa(expr, UnknownOperator)
        if haskey(namespace, expr.expression)
            val = namespace[expr.expression]
            return isa(val, Number) || isa(val, ConstantOperator)
        end
        return false
    end

    # Arithmetic on constants
    if isa(expr, AddOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, SubtractOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, MultiplyOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, DivideOperator)
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    if isa(expr, NegateOperator)
        return _is_constant_coefficient_strict(expr.operand, namespace)
    end

    if isa(expr, PowerOperator)
        return _is_constant_coefficient_strict(expr.base, namespace) &&
               _is_constant_coefficient_strict(expr.exponent, namespace)
    end

    # Fields are not constants
    if isa(expr, ScalarField) || isa(expr, VectorField)
        return false
    end

    return false
end

function _get_constant_value(expr)
    """
    Extract numeric value from a constant expression.
    Returns nothing if not a simple constant.
    """
    if isa(expr, Number)
        return expr
    end

    if isa(expr, ConstantOperator)
        return expr.value
    end

    return nothing
end

function validate_lhs_structure(expr)
    """
    Validate LHS structure and return a detailed report.
    Throws an error if the structure is invalid.
    """
    is_valid, info = is_proper_lhs_structure(expr)

    if !is_valid
        error_msg = info[:error_message]
        if error_msg === nothing
            error_msg = "LHS structure is invalid for matrix formulation"
        end
        throw(ArgumentError("Invalid LHS structure: $error_msg"))
    end

    return info
end

function parse_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    """
    Parse expression string into operator tree following patterns.
    Uses Julia's Meta.parse for proper AST parsing and operator precedence handling.
    
    This function evaluates mathematical expressions similar to how one would use
    eval(string, namespace) in problems:73-74.
    """
    
    expr_str = strip(expr_str)
    
    # Handle empty expressions
    if isempty(expr_str)
        throw(ArgumentError("Empty expression string"))
    end
    
    # Handle simple zero cases
    if expr_str == "0" || expr_str == "zero"
        return ZeroOperator()
    end
    
    try
        # Use Julia's Meta.parse for proper expression parsing
        parsed = Meta.parse(expr_str)
        
        # Recursively evaluate the parsed expression with namespace
        result = evaluate_parsed_expression(parsed, namespace)
        
        return result
        
    catch e
        if isa(e, Meta.ParseError)
            @warn "Parse error in expression '$expr_str': $(e.msg)"
            return UnknownOperator(expr_str)
        else
            # For evaluation errors, try fallback parsing patterns
            @debug "Error evaluating expression '$expr_str': $e, trying fallback patterns"
            return fallback_parse_expression(expr_str, namespace)
        end
    end
end

function evaluate_parsed_expression(expr, namespace::Dict{String, Any})
    """
    Recursively evaluate parsed expression with namespace substitution.
    Similar to eval(string, namespace) but with proper Julia AST handling.
    """
    
    if isa(expr, Symbol)
        # Variable/function lookup
        var_name = string(expr)
        if haskey(namespace, var_name)
            return namespace[var_name]
        else
            # Try built-in Julia functions/constants
            try
                return eval(expr)
            catch
                @warn "Unknown variable: $var_name"
                return UnknownOperator(var_name)
            end
        end
        
    elseif isa(expr, Number)
        # Numeric constants -> wrap in ConstantOperator for consistency
        return ConstantOperator(Float64(expr))
        
    elseif isa(expr, String)
        # String literals
        return expr
        
    elseif isa(expr, Expr)
        if expr.head == :call
            func_expr = expr.args[1]
            arg_exprs = expr.args[2:end]

            if func_expr isa Symbol && func_expr in (:+, :-, :*, :/, :^)
                op_args = [evaluate_parsed_expression(arg, namespace) for arg in arg_exprs]
                if isempty(op_args)
                    throw(ArgumentError("Operator $(func_expr) requires at least one operand"))
                end

                if func_expr == :+
                    result = op_args[1]
                    for i in 2:length(op_args)
                        result = AddOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :-
                    if length(op_args) == 1
                        return NegateOperator(op_args[1])
                    end
                    result = SubtractOperator(op_args[1], op_args[2])
                    for i in 3:length(op_args)
                        result = SubtractOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :*
                    result = op_args[1]
                    for i in 2:length(op_args)
                        result = MultiplyOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :/
                    if length(op_args) == 1
                        throw(ArgumentError("Division requires at least two operands"))
                    end
                    result = DivideOperator(op_args[1], op_args[2])
                    for i in 3:length(op_args)
                        result = DivideOperator(result, op_args[i])
                    end
                    return result
                elseif func_expr == :^
                    if length(op_args) == 1
                        return op_args[1]
                    end
                    result = PowerOperator(op_args[end - 1], op_args[end])
                    for i in (length(op_args) - 2):-1:1
                        result = PowerOperator(op_args[i], result)
                    end
                    return result
                end
            end

            # Check for Dedalus-style BC syntax: field(coord=value) → Interpolate
            # This handles expressions like u(y=0), T(z=1.0), dy(u)(y=0), etc.
            # In Julia's AST, keyword arguments appear as Expr(:kw, coord, value)
            # Following Dedalus operators.interpolate pattern (operators.py:1053-1062)
            if length(arg_exprs) == 1 && isa(arg_exprs[1], Expr) && arg_exprs[1].head == :kw
                kw_expr = arg_exprs[1]
                coord_name = string(kw_expr.args[1])
                position_expr = kw_expr.args[2]

                # Evaluate the function expression - this could be:
                # 1. A Symbol (field name like :u)
                # 2. An Expr (derivative expression like dy(u))
                operand = nothing
                if func_expr isa Symbol
                    field_name = string(func_expr)
                    if haskey(namespace, field_name)
                        operand = namespace[field_name]
                    end
                elseif func_expr isa Expr
                    # Recursively evaluate - handles dy(u)(y=0), lap(u)(y=0), etc.
                    operand = evaluate_parsed_expression(func_expr, namespace)
                end

                # Check if this is a valid operand for interpolation
                if operand !== nothing && isa(operand, Operand)
                    # Try to find the coordinate
                    coord = _find_coordinate_for_field(operand, coord_name, namespace)

                    # If not found in field, check namespace directly
                    if coord === nothing && haskey(namespace, coord_name)
                        coord_obj = namespace[coord_name]
                        if isa(coord_obj, Coordinate)
                            coord = coord_obj
                        end
                    end

                    if coord !== nothing
                        # Evaluate position value
                        position = coerce_constant_value(evaluate_parsed_expression(position_expr, namespace))

                        @debug "Detected BC syntax: $func_expr($coord_name=$position) → Interpolate"
                        return interpolate(operand, coord, Float64(position))
                    end
                end
            end

            if func_expr isa Symbol
                if func_expr == :dt || func_expr == :∂t
                    if isempty(arg_exprs)
                        throw(ArgumentError("dt/∂t requires at least one argument"))
                    end
                    field = evaluate_parsed_expression(arg_exprs[1], namespace)
                    order = if length(arg_exprs) >= 2
                        val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                        Int(round(val))
                    else
                        1
                    end
                    return dt(field, order)
                elseif func_expr == :d || func_expr == :diff
                    if length(arg_exprs) < 2
                        throw(ArgumentError("d requires at least an operand and a coordinate"))
                    end
                    field = evaluate_parsed_expression(arg_exprs[1], namespace)
                    coord = evaluate_parsed_expression(arg_exprs[2], namespace)
                    order = if length(arg_exprs) >= 3
                        val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                        Int(round(val))
                    else
                        1
                    end
                    return d(field, coord, order)
                elseif func_expr == :lift
                    if length(arg_exprs) < 3
                        throw(ArgumentError("lift requires operand, basis, and polynomial index"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    basis = evaluate_parsed_expression(arg_exprs[2], namespace)
                    n_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                    return lift(operand, basis, Int(round(n_val)))
                elseif func_expr == :fraclap || func_expr == :Δᵅ
                    # Fractional Laplacian: fraclap(f, α) or Δᵅ(f, α)
                    if length(arg_exprs) < 2
                        throw(ArgumentError("fraclap requires operand and exponent α"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    α_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                    return fraclap(operand, Float64(α_val))
                elseif func_expr == :sqrtlap || func_expr == :√Δ || func_expr == :Δ½
                    # Square root Laplacian: (-Δ)^(1/2)
                    if isempty(arg_exprs)
                        throw(ArgumentError("sqrtlap requires an operand"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    return sqrtlap(operand)
                elseif func_expr == :invsqrtlap || func_expr == :Δ⁻½
                    # Inverse square root Laplacian: (-Δ)^(-1/2)
                    if isempty(arg_exprs)
                        throw(ArgumentError("invsqrtlap requires an operand"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    return invsqrtlap(operand)
                end
            end
            
            func = evaluate_parsed_expression(func_expr, namespace)
            evaluated_args = [coerce_constant_value(evaluate_parsed_expression(arg, namespace)) for arg in arg_exprs]
            
            if isa(func, Function)
                return func(evaluated_args...)
            else
                @warn "Unknown function in expression: $(string(func_expr))"
                return UnknownOperator(string(expr))
            end
            
        elseif expr.head in [:+, :-, :*, :/, :^]
            # Arithmetic operations with proper precedence from Meta.parse
            if length(expr.args) == 2
                left = evaluate_parsed_expression(expr.args[1], namespace)
                right = evaluate_parsed_expression(expr.args[2], namespace)
                
                if expr.head == :+
                    return AddOperator(left, right)
                elseif expr.head == :-
                    return SubtractOperator(left, right)
                elseif expr.head == :*
                    return MultiplyOperator(left, right)
                elseif expr.head == :/
                    return DivideOperator(left, right)
                elseif expr.head == :^
                    return PowerOperator(left, right)
                end
            elseif length(expr.args) == 1 && expr.head == :-
                # Unary minus
                operand = evaluate_parsed_expression(expr.args[1], namespace)
                return NegateOperator(operand)
            end
            
        elseif expr.head == :ref
            # Array/field indexing: field[i, j]
            array_expr = expr.args[1]
            indices = expr.args[2:end]
            
            array = evaluate_parsed_expression(array_expr, namespace)
            evaluated_indices = [evaluate_parsed_expression(idx, namespace) for idx in indices]
            
            return IndexOperator(array, evaluated_indices)
            
        elseif expr.head == :block
            # Block expressions - evaluate meaningful statements
            meaningful_args = filter(arg -> !isa(arg, LineNumberNode), expr.args)
            if !isempty(meaningful_args)
                return evaluate_parsed_expression(meaningful_args[end], namespace)
            end
        end
        
        # Fallback for unsupported expression types
        @debug "Unsupported expression type: $(expr.head)"
        return UnknownOperator(string(expr))
        
    else
        # Return other types as-is
        return expr
    end
end

function fallback_parse_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    """
    Fallback parsing for expressions that fail Meta.parse evaluation.
    Uses simple string pattern matching as backup.
    """
    
    # Handle simple field references
    if haskey(namespace, expr_str)
        return namespace[expr_str]
    end
    
    # Handle simple numeric constants
    try
        val = parse(Float64, expr_str)
        return ConstantOperator(val)
    catch
        # Continue with pattern matching
    end
    
    # Pattern-based parsing for common PDE operators (as fallback)
    # Unicode time derivative: ∂t(field)
    if startswith(expr_str, "∂t(") && endswith(expr_str, ")")
        field_name = expr_str[nextind(expr_str, 0, 3):end-1]  # Skip "∂t(" which is 3 chars
        if haskey(namespace, field_name)
            return TimeDerivative(namespace[field_name], 1)
        end
    end
    
    if startswith(expr_str, "d(") && endswith(expr_str, ")")
        args_str = expr_str[3:end-1]
        args = [strip(arg) for arg in split(args_str, ',')]
        if length(args) >= 2 && haskey(namespace, args[1]) && haskey(namespace, args[2])
            return Differentiate(namespace[args[1]], namespace[args[2]], 1)
        end
    end
    
    # Return as unknown operator if all parsing fails
    return UnknownOperator(expr_str)
end

# Helper operator types for parsing (following operator structure)
struct ZeroOperator <: Operator end
struct ConstantOperator <: Operator
    value::Float64
end
struct UnknownOperator <: Operator
    expression::String
end

@inline coerce_constant_value(x) = x isa ConstantOperator ? x.value : x

# Note: AddOperator, SubtractOperator, MultiplyOperator are defined in operators.jl

# Additional arithmetic operators for equation parsing
struct DivideOperator <: Operator
    left::Any
    right::Any
end

struct PowerOperator <: Operator
    base::Any
    exponent::Any
end

struct NegateOperator <: Operator
    operand::Any
end

struct IndexOperator <: Operator
    array::Any
    indices::Vector{Any}
end

"""
    _find_coordinate_for_field(field, coord_name, namespace)

Find a Coordinate object for the given coordinate name that is associated with the field.
Used for automatic BC syntax parsing: field(coord=value) → Interpolate(field, coord, value)

Following Dedalus operators.interpolate pattern (operators.py:1053-1059):
- If coord is already a Coordinate, use it directly
- If coord is a string, look it up in the field's domain or namespace
"""
function _find_coordinate_for_field(field::Operand, coord_name::String, namespace::Dict{String, Any})
    # First check namespace for coordinate objects
    if haskey(namespace, coord_name)
        coord = namespace[coord_name]
        if isa(coord, Coordinate)
            return coord
        end
    end

    # Try to find coordinate from field's domain/bases
    if isa(field, ScalarField) && hasfield(typeof(field), :dist)
        dist = field.dist
        if dist !== nothing && hasfield(typeof(dist), :coords)
            for c in dist.coords
                if c.name == coord_name
                    return c
                end
            end
            # Try coordinate-system indexing if available
            if hasfield(typeof(dist), :coordsys)
                try
                    return dist.coordsys[coord_name]
                catch
                    # Coordinate not found in coordsys
                end
            end
        end
    end

    # Try to construct coordinate from field's bases
    if hasfield(typeof(field), :bases)
        for basis in field.bases
            if basis !== nothing && hasfield(typeof(basis), :meta)
                if basis.meta.element_label == coord_name
                    # Create coordinate from basis metadata
                    if hasfield(typeof(basis.meta), :coordsys) && basis.meta.coordsys !== nothing
                        coordsys = basis.meta.coordsys
                        if hasfield(typeof(coordsys), :coords)
                            for c in coordsys.coords
                                if c.name == coord_name
                                    return c
                                end
                            end
                        end
                        # Try subscript access
                        try
                            return coordsys[coord_name]
                        catch
                            # Not found
                        end
                    end
                end
            end
        end
    end

    @debug "Could not find coordinate '$coord_name' for field"
    return nothing
end

function build_matrices(problem::Problem)
    """
    Build system matrices for problem following structure.
    Following subsystems:build_subproblem_matrices (subsystems:72-81) and
    Subproblem.build_matrices (subsystems:497-576).
    """
    
    if length(problem.equations) == 0
        throw(ArgumentError("No equations specified"))
    end
    
    # Build matrix expressions from equations (following problems:_build_matrix_expressions)
    build_matrix_expressions!(problem)
    
    # Compute field sizes for each equation and variable
    eqn_sizes = [compute_field_size(eq_data) for eq_data in problem.equation_data]
    var_sizes = [compute_field_size(var) for var in problem.variables]

    total_rows = sum(eqn_sizes)  # Total rows
    total_cols = sum(var_sizes)  # Total columns

    @debug "Building matrices: equations=$total_rows, variables=$total_cols"

    # Matrix names to build (following convention)
    matrix_names = ["M", "L"]  # M = mass matrix, L = stiffness matrix

    # Build sparse matrices following subsystems:513-537 pattern
    matrices = Dict{String, Any}()
    for name in matrix_names
        # Collect sparse matrix entries (ComplexF64 for spectral methods)
        data, rows, cols = ComplexF64[], Int[], Int[]
        
        i0 = 0  # Row offset
        for (eq_idx, eq_data) in enumerate(problem.equation_data)
            eqn_size = eqn_sizes[eq_idx]
            if eqn_size > 0 && check_equation_condition(eq_data)
                # Get expression matrix blocks for this equation
                expr = get_matrix_expression(eq_data, name)
                if expr !== nothing && !is_zero_expression(expr)
                    # Build expression matrices for each variable
                    j0 = 0  # Column offset
                    for (var_idx, var) in enumerate(problem.variables)
                        var_size = var_sizes[var_idx]
                        if var_size > 0
                            # Get matrix block for this variable
                            block = build_expression_matrix_block(expr, var, eqn_size, var_size)
                            if !isempty(block.nzval)
                                # Add to sparse matrix data
                                # SparseMatrixCSC stores: rowval (row indices), colptr (column pointers), nzval (values)
                                # We need to expand colptr to get column indices for each non-zero
                                block_rows, block_cols, block_vals = findnz(block)
                                append!(data, block_vals)
                                append!(rows, i0 .+ block_rows)
                                append!(cols, j0 .+ block_cols)
                            end
                        end
                        j0 += var_size
                    end
                end
            end
            i0 += eqn_size
        end
        
        # Create sparse matrix
        if !isempty(data)
            # Filter small entries (following entry_cutoff pattern)
            entry_cutoff = 1e-14
            significant = abs.(data) .>= entry_cutoff
            data = data[significant]
            rows = rows[significant]
            cols = cols[significant]
            
            matrices[name] = sparse(rows, cols, data, total_rows, total_cols)
        else
            # Empty matrix
            matrices[name] = spzeros(ComplexF64, total_rows, total_cols)
        end

        @debug "Built matrix $name: size=($total_rows, $total_cols), nnz=$(nnz(matrices[name]))"
    end
    
    # Build forcing vector (RHS terms)
    F_vector = build_forcing_vector(problem, eqn_sizes, total_rows)
    
    # Return matrices in standard format
    L_matrix = matrices["L"]
    M_matrix = matrices["M"] 
    
    @info "Matrix building completed: L=$(size(L_matrix)), M=$(size(M_matrix)), F=$(length(F_vector))"
    
    return L_matrix, M_matrix, F_vector
end

function build_matrix_expressions!(problem::Problem)
    """
    Build matrix expressions from parsed equations.
    Following problems:_build_matrix_expressions patterns.
    """
    
    problem.equation_data = Dict{String, Any}[]
    
    for (i, equation_str) in enumerate(problem.equations)
        try
            # Parse equation
            lhs, rhs = parse_equation(equation_str, problem.namespace)
            
            # Build matrix expressions following problem type
            eq_data = build_equation_expressions(lhs, rhs, problem.variables)
            eq_data["equation_index"] = i
            eq_data["equation_string"] = equation_str
            if !haskey(eq_data, "equation_size")
                vars = get(eq_data, "equation_variables", get(eq_data, "variables", problem.variables))
                if isa(vars, Vector)
                    eq_sz = sum(field_dofs(var) for var in vars)
                    if eq_sz <= 0
                        eq_sz = sum(field_dofs(var) for var in problem.variables)
                    end
                    eq_data["equation_size"] = eq_sz
                end
            end

            push!(problem.equation_data, eq_data)
            
        catch e
            @error "Failed to build matrix expressions for equation $i: $equation_str" exception=e
            # Create fallback equation data
            fallback_data = Dict(
                "M" => nothing,
                "L" => UnknownOperator(equation_str),
                "F" => ZeroOperator(),
                "equation_index" => i,
                "equation_string" => equation_str,
                "equation_size" => 0
            )
            push!(problem.equation_data, fallback_data)
        end
    end
end

function build_equation_expressions(lhs, rhs, variables::Vector)
    """
    Build matrix expressions from LHS and RHS operators.
    Following _build_matrix_expressions patterns.
    """
    
    eq_data = Dict{String, Any}()
    
    # Split LHS into mass matrix (time derivatives) and stiffness matrix (spatial) terms
    # Following IVP pattern: M.dt(X) + L.X = F (problems:328)
    M_terms, L_terms = split_time_spatial_operators(lhs)
    
    # Store matrix expressions
    eq_data["M"] = combine_operators(M_terms)      # Mass matrix terms
    eq_data["L"] = combine_operators(L_terms)      # Stiffness matrix terms  
    eq_data["F"] = rhs                             # Forcing terms

    # Determine which variables participate in this equation
    eq_vars = _detect_equation_variables(lhs, variables)
    if isempty(eq_vars)
        # Some constraint equations (e.g., BCs) only reference variables on RHS
        eq_vars = _detect_equation_variables(rhs, variables)
    end
    if isempty(eq_vars)
        # Fall back to all variables to keep matrix sizes consistent
        eq_vars = copy(variables)
    end

    eq_data["equation_variables"] = eq_vars
    eq_size = sum(field_dofs(var) for var in eq_vars)
    if eq_size <= 0
        eq_size = sum(field_dofs(var) for var in variables)
    end
    eq_data["equation_size"] = eq_size
    
    # Metadata
    eq_data["variables"] = variables
    eq_data["lhs"] = lhs
    eq_data["rhs"] = rhs
    
    return eq_data
end

function _detect_equation_variables(expr, variables::Vector{<:Operand})
    found = Operand[]
    _collect_equation_variables!(found, expr, variables)
    return found
end

function _collect_equation_variables!(found::Vector{Operand}, expr, variables::Vector{<:Operand})
    expr === nothing && return

    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        for var in variables
            if _operand_matches_variable(expr, var)
                _maybe_add_variable!(found, var)
                break
            end
        end
    end

    if hasfield(typeof(expr), :left)
        _collect_equation_variables!(found, getfield(expr, :left), variables)
    end
    if hasfield(typeof(expr), :right)
        _collect_equation_variables!(found, getfield(expr, :right), variables)
    end
    if hasfield(typeof(expr), :operand)
        _collect_equation_variables!(found, getfield(expr, :operand), variables)
    end
    if hasfield(typeof(expr), :operands)
        ops = getfield(expr, :operands)
        if ops !== nothing
            for op in ops
                _collect_equation_variables!(found, op, variables)
            end
        end
    end
    if hasfield(typeof(expr), :array)
        _collect_equation_variables!(found, getfield(expr, :array), variables)
    end
    if hasfield(typeof(expr), :indices)
        idxs = getfield(expr, :indices)
        if idxs !== nothing
            for idx in idxs
                _collect_equation_variables!(found, idx, variables)
            end
        end
    end
    if hasfield(typeof(expr), :base)
        _collect_equation_variables!(found, getfield(expr, :base), variables)
    end
    if hasfield(typeof(expr), :exponent)
        _collect_equation_variables!(found, getfield(expr, :exponent), variables)
    end
end

function _maybe_add_variable!(found::Vector{Operand}, var::Operand)
    for existing in found
        if existing === var
            return
        end
        existing_name = _operand_name(existing)
        new_name = _operand_name(var)
        if existing_name !== nothing && existing_name == new_name
            return
        end
    end
    push!(found, var)
end

@inline function _operand_matches_variable(expr, var::Operand)
    (expr === var) && return true
    expr_name = _operand_name(expr)
    var_name = _operand_name(var)
    return expr_name !== nothing && expr_name == var_name
end

@inline function _operand_name(var)
    return hasfield(typeof(var), :name) ? getfield(var, :name) : nothing
end

function split_time_spatial_operators(operator)
    """
    Split operator into time derivative (mass matrix) and spatial (stiffness) terms.
    Following operators split pattern.
    """
    
    M_terms = []  # Time derivative terms
    L_terms = []  # Spatial terms
    empty_namespace = Dict{String, Any}()
    
    if isa(operator, TimeDerivative)
        # Pure time derivative
        push!(M_terms, operator)
        
    elseif isa(operator, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Pure spatial operator
        push!(L_terms, operator)
        
    elseif isa(operator, AddOperator)
        # Split addition terms recursively
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)
        append!(M_terms, left_M)
        append!(L_terms, left_L)
        append!(M_terms, right_M)
        append!(L_terms, right_L)
        
    elseif isa(operator, SubtractOperator)
        # Split subtraction terms recursively
        # For A - B, we split both and negate the right side terms
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)

        # Add left terms directly
        append!(M_terms, left_M)
        append!(L_terms, left_L)

        # Right terms need negation - wrap in NegateOperator or multiply by -1
        for term in right_M
            push!(M_terms, NegateOperator(term))
        end
        for term in right_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, NegateOperator)
        inner_M, inner_L = split_time_spatial_operators(operator.operand)
        for term in inner_M
            push!(M_terms, NegateOperator(term))
        end
        for term in inner_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, MultiplyOperator)
        coeff = nothing
        inner = nothing

        if _is_constant_coefficient_strict(operator.left, empty_namespace) &&
           !_is_constant_coefficient_strict(operator.right, empty_namespace)
            coeff = operator.left
            inner = operator.right
        elseif _is_constant_coefficient_strict(operator.right, empty_namespace) &&
               !_is_constant_coefficient_strict(operator.left, empty_namespace)
            coeff = operator.right
            inner = operator.left
        end

        if inner !== nothing
            scaled = MultiplyOperator(coeff, inner)
            if isa(inner, TimeDerivative)
                push!(M_terms, scaled)
            elseif isa(inner, Union{Laplacian, Gradient, Divergence, Differentiate}) || hasfield(typeof(inner), :name)
                push!(L_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end

    elseif isa(operator, DivideOperator)
        if _is_constant_coefficient_strict(operator.right, empty_namespace)
            scaled = DivideOperator(operator.left, operator.right)
            if isa(operator.left, TimeDerivative)
                push!(M_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end
        
    elseif hasfield(typeof(operator), :name)
        # Direct variable reference -> identity in L
        push!(L_terms, operator)
        
    else
        # Other operators go to L by default
        push!(L_terms, operator)
    end
    
    return M_terms, L_terms
end

function combine_operators(terms::Vector)
    """Combine operator terms into single expression"""
    if isempty(terms)
        return ZeroOperator()
    elseif length(terms) == 1
        return terms[1]
    else
        # Combine with addition
        result = terms[1]
        for i in 2:length(terms)
            result = AddOperator(result, terms[i])
        end
        return result
    end
end

# Supporting functions for matrix building

function field_dofs(field::ScalarField)
    if get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    else
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    end
end

field_dofs(field::VectorField) = sum(field_dofs(comp) for comp in field.components)
field_dofs(field::TensorField) = sum(field_dofs(comp) for comp in vec(field.components))

function compute_field_size(field_or_data)
    """Compute size (degrees of freedom) of field or equation data"""
    if isa(field_or_data, Dict)
        if haskey(field_or_data, "equation_size")
            return field_or_data["equation_size"]
        elseif haskey(field_or_data, "equation_variables")
            vars = field_or_data["equation_variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        elseif haskey(field_or_data, "variables")
            vars = field_or_data["variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        end
        return 0
    elseif isa(field_or_data, ScalarField)
        return field_dofs(field_or_data)
    elseif isa(field_or_data, VectorField) || isa(field_or_data, TensorField)
        return field_dofs(field_or_data)
    elseif hasfield(typeof(field_or_data), :buffers) && get_coeff_data(field_or_data) !== nothing
        return length(get_coeff_data(field_or_data))
    elseif hasfield(typeof(field_or_data), :buffers) && get_grid_data(field_or_data) !== nothing
        return length(get_grid_data(field_or_data))
    else
        return 0
    end
end

function check_equation_condition(eq_data::Dict)
    """
    Check if equation should be included in matrix assembly.

    An equation is included if:
    1. It has valid matrix expressions (M, L, or F)
    2. It is marked as enabled (if "enabled" key exists)
    3. It has a valid condition (if "condition" key exists)
    4. It references at least one problem variable
    5. The equation is well-formed (not flagged as invalid)

    Following patterns where equations can be conditionally
    included/excluded based on wavenumber, problem parameters, etc.
    """

    # Check if equation is explicitly disabled
    if haskey(eq_data, "enabled") && !eq_data["enabled"]
        @debug "Equation excluded: explicitly disabled" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has a condition function that evaluates to false
    if haskey(eq_data, "condition")
        condition = eq_data["condition"]
        if isa(condition, Bool)
            if !condition
                @debug "Equation excluded: condition is false" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        elseif isa(condition, Function)
            # Condition is a function - evaluate it
            try
                result = condition(eq_data)
                if !result
                    @debug "Equation excluded: condition function returned false" eq_index=get(eq_data, "equation_index", 0)
                    return false
                end
            catch e
                @warn "Equation condition evaluation failed, including equation" exception=e
            end
        end
    end

    # Check if equation is flagged as invalid
    if get(eq_data, "is_invalid", false)
        @debug "Equation excluded: flagged as invalid" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has any matrix content
    has_M = haskey(eq_data, "M") && !is_zero_expression(eq_data["M"])
    has_L = haskey(eq_data, "L") && !is_zero_expression(eq_data["L"])
    has_F = haskey(eq_data, "F") && !is_zero_expression(eq_data["F"])

    if !has_M && !has_L && !has_F
        @debug "Equation excluded: no matrix content (M, L, F all zero/missing)" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check equation size
    eq_size = get(eq_data, "equation_size", 0)
    if eq_size <= 0
        @debug "Equation excluded: equation_size <= 0" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check wavenumber conditions (for spectral problems)
    if haskey(eq_data, "valid_modes")
        valid_modes = eq_data["valid_modes"]
        current_mode = get(eq_data, "current_mode", nothing)
        if current_mode !== nothing && !in(current_mode, valid_modes)
            @debug "Equation excluded: mode not in valid_modes" current_mode valid_modes
            return false
        end
    end

    # Check for wavenumber-based conditions (k=0 special handling, etc.)
    if haskey(eq_data, "exclude_k_zero") && eq_data["exclude_k_zero"]
        wavenumber = get(eq_data, "wavenumber", nothing)
        if wavenumber !== nothing
            # Check if all wavenumber components are zero
            if isa(wavenumber, Number) && wavenumber == 0
                @debug "Equation excluded: k=0 mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            elseif isa(wavenumber, Tuple) && all(k -> k == 0, wavenumber)
                @debug "Equation excluded: k=(0,...,0) mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        end
    end

    # Check for gauge conditions (pressure gauge, etc.)
    if haskey(eq_data, "is_gauge_condition") && eq_data["is_gauge_condition"]
        # Gauge conditions may have special handling
        gauge_mode = get(eq_data, "gauge_mode", nothing)
        current_mode = get(eq_data, "current_mode", nothing)

        if gauge_mode !== nothing && current_mode !== nothing
            if gauge_mode != current_mode
                # Only include gauge condition for specific mode
                return false
            end
        end
    end

    # Check if this is a boundary condition equation
    if haskey(eq_data, "is_boundary_condition") && eq_data["is_boundary_condition"]
        # Boundary conditions are always included if they're valid
        bc_valid = get(eq_data, "bc_valid", true)
        if !bc_valid
            @debug "Equation excluded: boundary condition marked invalid"
            return false
        end
    end

    # All checks passed
    return true
end

function is_equation_valid(eq_data::Dict)
    """
    Check if equation data is structurally valid.
    Returns (is_valid::Bool, error_message::Union{String,Nothing})
    """

    # Must have equation string
    if !haskey(eq_data, "equation_string")
        return (false, "Missing equation_string")
    end

    # Must have LHS
    if !haskey(eq_data, "lhs")
        return (false, "Missing LHS expression")
    end

    # Check for parse errors
    if haskey(eq_data, "parse_error")
        return (false, "Parse error: $(eq_data["parse_error"])")
    end

    # Check LHS structure if we have the expression
    lhs = eq_data["lhs"]
    if lhs !== nothing
        is_valid_lhs, lhs_info = is_proper_lhs_structure(lhs)
        if !is_valid_lhs
            return (false, "Invalid LHS structure: $(lhs_info[:error_message])")
        end
    end

    return (true, nothing)
end

function set_equation_condition!(eq_data::Dict, condition::Union{Bool, Function})
    """
    Set a condition for equation inclusion in matrix assembly.
    """
    eq_data["condition"] = condition
end

function enable_equation!(eq_data::Dict)
    """Enable an equation for matrix assembly."""
    eq_data["enabled"] = true
end

function disable_equation!(eq_data::Dict)
    """Disable an equation from matrix assembly."""
    eq_data["enabled"] = false
end

function set_valid_modes!(eq_data::Dict, modes::Union{Vector, Set, AbstractRange})
    """
    Set the valid wavenumber modes for this equation.
    The equation will only be included for these modes.
    """
    eq_data["valid_modes"] = Set(modes)
end

function exclude_k_zero!(eq_data::Dict, exclude::Bool=true)
    """
    Exclude this equation from k=0 (homogeneous) mode.
    Useful for gauge conditions in incompressible flow problems.
    """
    eq_data["exclude_k_zero"] = exclude
end

function get_matrix_expression(eq_data::Dict, matrix_name::String)
    """Get matrix expression from equation data"""
    return get(eq_data, matrix_name, nothing)
end

function is_zero_expression(expr)
    """Check if expression is effectively zero"""
    return isa(expr, ZeroOperator) || expr === nothing
end

@inline _zero_block(eqn_size::Int, var_size::Int) = spzeros(ComplexF64, eqn_size, var_size)

function _identity_block(eqn_size::Int, var_size::Int; scale::Number=1.0)
    if eqn_size == 0 || var_size == 0
        return _zero_block(eqn_size, var_size)
    end
    diag_len = min(eqn_size, var_size)
    vals = fill(ComplexF64(scale), diag_len)
    return spdiagm(eqn_size, var_size, 0 => vals)
end

function build_expression_matrix_block(expr, var, eqn_size::Int, var_size::Int)
    """
    Build matrix block for expression acting on variable.
    Following expression_matrices pattern.
    """
    
    if isa(expr, TimeDerivative) && _operand_matches_variable(expr.operand, var)
        # Time derivative of this variable -> identity block
        return _identity_block(eqn_size, var_size)

    elseif isa(expr, Laplacian) && _operand_matches_variable(expr.operand, var)
        # Laplacian: ∇² = Σ_i ∂²/∂x_i²
        # In spectral space for Fourier bases: Δ̂ = -|k|² (diagonal)
        # For Chebyshev/Legendre: use second derivative matrix D²
        # Here we return a diagonal approximation using -|k|² scaling
        # The actual matrix construction happens in subsystems.jl via expression_matrices
        return _identity_block(eqn_size, var_size; scale=-1.0)

    elseif isa(expr, Union{Gradient, Divergence, Differentiate}) && _operand_matches_variable(expr.operand, var)
        # First-order spatial derivatives
        # Gradient/Differentiate: ∂/∂x_i -> ik_i in Fourier, D matrix in Chebyshev
        # Divergence: ∇·v = Σ_i ∂v_i/∂x_i
        # Returns identity matrix here as the marker for variable participation.
        # Actual spectral differentiation matrices with basis-specific coefficients
        # are constructed in operators.jl and subsystems.jl during system assembly.
        return _identity_block(eqn_size, var_size)

    elseif _operand_matches_variable(expr, var)
        # Direct variable reference -> identity
        return _identity_block(eqn_size, var_size)
        
    elseif isa(expr, AddOperator)
        # Sum of operators
        left_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        right_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        return left_block + right_block
        
    elseif isa(expr, SubtractOperator)
        # Difference of operators
        left_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        right_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        return left_block - right_block
        
    elseif isa(expr, MultiplyOperator)
        # Constant multiplication (either side)
        if isa(expr.right, ConstantOperator) || isa(expr.right, Number)
            coeff = isa(expr.right, ConstantOperator) ? expr.right.value : expr.right
            base_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
            return ComplexF64(coeff) * base_block
        elseif isa(expr.left, ConstantOperator) || isa(expr.left, Number)
            coeff = isa(expr.left, ConstantOperator) ? expr.left.value : expr.left
            base_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
            return ComplexF64(coeff) * base_block
        else
            @debug "Non-constant multiplication in matrix block: $(typeof(expr.left)) * $(typeof(expr.right))"
            return _zero_block(eqn_size, var_size)
        end
        
    elseif isa(expr, DivideOperator)
        # Constant division
        if isa(expr.right, ConstantOperator) || isa(expr.right, Number)
            denom = isa(expr.right, ConstantOperator) ? expr.right.value : expr.right
            base_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
            return (ComplexF64(1) / ComplexF64(denom)) * base_block
        else
            @debug "Non-constant division in matrix block: $(typeof(expr.right))"
            return _zero_block(eqn_size, var_size)
        end
        
    elseif isa(expr, NegateOperator)
        return -build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
        
    elseif isa(expr, ConstantOperator)
        # Constant expression -> zero block (constants don't depend on variables)
        return _zero_block(eqn_size, var_size)
        
    elseif isa(expr, ZeroOperator)
        # Zero expression -> zero block
        return _zero_block(eqn_size, var_size)

    elseif isa(expr, Interpolate) && _operand_matches_variable(expr.operand, var)
        # BC interpolation constraint: field evaluated at boundary
        # Return identity block to mark variable participation in this BC equation
        return _identity_block(eqn_size, var_size)

    else
        # Unknown expression -> zero block
        @debug "Unknown expression type for matrix block: $(typeof(expr))"
        return _zero_block(eqn_size, var_size)
    end
end

function build_forcing_vector(problem::Problem, eqn_sizes::Vector{Int}, total_size::Int)
    """Build forcing vector from RHS terms"""
    
    F_vector = zeros(ComplexF64, total_size)
    
    i0 = 0
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        eqn_size = eqn_sizes[eq_idx]
        if eqn_size > 0
            rhs_expr = get(eq_data, "F", ZeroOperator())
            
            # Evaluate RHS expression to get forcing values
            if isa(rhs_expr, ConstantOperator)
                F_vector[i0+1:i0+eqn_size] .= rhs_expr.value
            elseif isa(rhs_expr, ZeroOperator)
                F_vector[i0+1:i0+eqn_size] .= 0.0
            else
                # Complex RHS expressions would need proper evaluation
                @debug "Complex RHS expression not fully supported: $(typeof(rhs_expr))"
                F_vector[i0+1:i0+eqn_size] .= 0.0
            end
        end
        i0 += eqn_size
    end
    
    return F_vector
end

# Legacy functions (kept for compatibility)

function process_lhs_operator!(L_matrix::Matrix, M_matrix::Matrix, lhs_op, eq_idx::Int, variables::Vector)
    """
    Process LHS operator and extract contributions to system matrices.
    Following pattern where time derivatives go to M_matrix,
    spatial operators go to L_matrix.
    """
    
    if isa(lhs_op, TimeDerivative)
        # Time derivative terms go to mass matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            M_matrix[eq_idx, var_idx] = 1.0
        else
            @debug "Unknown variable in time derivative"
        end
        
    elseif isa(lhs_op, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Spatial operators go to linear operator matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            # Store operator type marker - actual spectral matrix coefficients
            # are computed during subproblem matrix assembly based on basis type
            if isa(lhs_op, Laplacian)
                L_matrix[eq_idx, var_idx] = -1.0  # Typical Laplacian sign
            else
                L_matrix[eq_idx, var_idx] = 1.0
            end
        else
            @debug "Unknown variable in spatial operator"
        end
        
    elseif isa(lhs_op, AddOperator)
        # Recursively process addition terms
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.right, eq_idx, variables)
        
    elseif isa(lhs_op, SubtractOperator)
        # Process left term normally, right term with negative sign
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        # Would need to negate contributions from right side
        # This requires more sophisticated matrix coefficient tracking
        @debug "Subtraction operator needs more sophisticated handling"
        
    elseif isa(lhs_op, MultiplyOperator)
        # Handle coefficient multiplication
        if isa(lhs_op.right, ConstantOperator)
            coeff = lhs_op.right.value
            # Apply coefficient to left operand contributions
            # This would require modifying matrix entries by coefficient
            @debug "Coefficient multiplication needs coefficient tracking: $coeff"
            process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        else
            @debug "General multiplication not yet supported"
        end
        
    elseif hasfield(typeof(lhs_op), :name)
        # Direct variable reference
        var_idx = find_variable_index(lhs_op, variables)
        if var_idx !== nothing
            L_matrix[eq_idx, var_idx] = 1.0
        end
        
    elseif isa(lhs_op, ZeroOperator)
        # Zero contribution
        nothing
        
    elseif isa(lhs_op, ConstantOperator)
        # Constant terms shouldn't appear in LHS typically
        @debug "Constant term in LHS: $(lhs_op.value)"
        
    else
        @debug "Unhandled LHS operator type: $(typeof(lhs_op))"
    end
end

function process_rhs_operator!(F_vector::Vector, rhs_op, eq_idx::Int, variables::Vector)
    """
    Process RHS operator and extract contributions to forcing vector.
    Following pattern where RHS represents known terms/forcing.

    Recursively evaluates composite operators (Add, Subtract, Multiply) to
    compute the scalar forcing value for each equation.
    """

    if isa(rhs_op, ConstantOperator)
        # Constant forcing term
        F_vector[eq_idx] = rhs_op.value

    elseif isa(rhs_op, ZeroOperator)
        # Zero RHS (homogeneous equation)
        F_vector[eq_idx] = 0.0

    elseif isa(rhs_op, AddOperator)
        # Sum of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value + right_value

    elseif isa(rhs_op, SubtractOperator)
        # Difference of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value - right_value

    elseif isa(rhs_op, MultiplyOperator)
        # Product of RHS terms
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        if isa(rhs_op.right, Number)
            F_vector[eq_idx] = left_value * rhs_op.right
        else
            right_value = evaluate_rhs_scalar(rhs_op.right, variables)
            F_vector[eq_idx] = left_value * right_value
        end

    elseif isa(rhs_op, Number)
        # Direct numeric value
        F_vector[eq_idx] = Float64(real(rhs_op))

    elseif isa(rhs_op, String) && (rhs_op == "0" || rhs_op == "zero")
        # String representation of zero
        F_vector[eq_idx] = 0.0

    else
        @debug "Unhandled RHS operator type: $(typeof(rhs_op)), using zero"
        F_vector[eq_idx] = 0.0
    end
end

"""
    evaluate_rhs_scalar(op, variables::Vector) -> Float64

Recursively evaluate an operator expression to obtain a scalar value.
Used for extracting forcing terms from composite RHS expressions.

Returns the scalar value of the expression, or 0.0 for unhandled types.
"""
function evaluate_rhs_scalar(op, variables::Vector)
    if isa(op, ConstantOperator)
        return Float64(op.value)

    elseif isa(op, ZeroOperator)
        return 0.0

    elseif isa(op, Number)
        return Float64(real(op))

    elseif isa(op, AddOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val + right_val

    elseif isa(op, SubtractOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val - right_val

    elseif isa(op, MultiplyOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        if isa(op.right, Number)
            return left_val * op.right
        else
            right_val = evaluate_rhs_scalar(op.right, variables)
            return left_val * right_val
        end

    elseif isa(op, ScalarField)
        # For field-valued RHS, we need to evaluate at specific points
        # For now, return the mean value if available
        if get_grid_data(op) !== nothing && length(get_grid_data(op)) > 0
            return real(sum(get_grid_data(op)) / length(get_grid_data(op)))
        elseif get_coeff_data(op) !== nothing && length(get_coeff_data(op)) > 0
            # First coefficient is often the mean for spectral methods
            # Use GPU-safe indexing to avoid scalar indexing on GPU arrays
            if is_gpu_array(get_coeff_data(op))
                # Copy first element to CPU to avoid GPU scalar indexing
                first_coef = Array(@view get_coeff_data(op)[1:1])[1]
            else
                first_coef = get_coeff_data(op)[1]
            end
            return real(first_coef)
        else
            return 0.0
        end

    elseif isa(op, String)
        # Try to parse as number
        if op == "0" || op == "zero"
            return 0.0
        end
        try
            return parse(Float64, op)
        catch
            return 0.0
        end

    else
        @debug "evaluate_rhs_scalar: unhandled type $(typeof(op)), returning 0.0"
        return 0.0
    end
end

function find_variable_index(operand, variables::Vector)
    """Find index of variable in problem variable list"""
    
    # Handle direct variable reference
    for (i, var) in enumerate(variables)
        if operand === var
            return i
        end
    end
    
    # Handle by name if operand has name field
    if hasfield(typeof(operand), :name)
        for (i, var) in enumerate(variables)
            if hasfield(typeof(var), :name) && operand.name == var.name
                return i
            end
        end
    end
    
    return nothing
end

# Domain setup
function setup_domain!(problem::Problem)
    """Setup domain for problem based on variables"""
    
    if length(problem.variables) == 0
        throw(ArgumentError("No variables specified"))
    end
    
    # Get distributor and bases from first variable
    first_var = problem.variables[1]
    if isa(first_var, ScalarField)
        problem.domain = first_var.domain
    elseif isa(first_var, VectorField)
        problem.domain = first_var.domain
    elseif isa(first_var, TensorField)
        problem.domain = first_var.domain
    else
        throw(ArgumentError("Unknown variable type: $(typeof(first_var))"))
    end
    
    # Verify all variables have compatible domains
    for var in problem.variables
        var_domain = nothing
        if isa(var, ScalarField)
            var_domain = var.domain
        elseif isa(var, VectorField)
            var_domain = var.domain
        elseif isa(var, TensorField)
            var_domain = var.domain
        end
        
        if var_domain !== nothing && var_domain != problem.domain
            @warn "Variable $(var.name) has incompatible domain"
        end
    end
end

# Problem validation
function validate_problem(problem::Problem)
    """Validate problem formulation"""

    errors = String[]

    if length(problem.variables) == 0
        push!(errors, "No variables specified")
    end

    if length(problem.equations) == 0
        push!(errors, "No equations specified")
    end

    # For IVPs/EVPs: equations should match variables exactly
    # For BVPs: equations should be >= variables (includes BCs)
    if isa(problem, LBVP) || isa(problem, NLBVP)
        # BVPs can have extra equations for boundary conditions
        if length(problem.equations) < length(problem.variables)
            push!(errors, "Number of equations ($(length(problem.equations))) is less than number of variables ($(length(problem.variables)))")
        end
    else
        # IVP/EVP: strict match
        if length(problem.equations) != length(problem.variables)
            push!(errors, "Number of equations ($(length(problem.equations))) does not match number of variables ($(length(problem.variables)))")
        end
    end

    # Check for required boundary conditions in boundary value problems
    # Note: BCs can be embedded in equations via field(coord=value) syntax
    if isa(problem, LBVP) || isa(problem, NLBVP)
        # For BVPs, we either need explicit BCs or equations > variables (implicit BCs)
        has_explicit_bcs = length(problem.boundary_conditions) > 0 || length(problem.bc_manager.conditions) > 0
        has_implicit_bcs = length(problem.equations) > length(problem.variables)
        if !has_explicit_bcs && !has_implicit_bcs
            push!(errors, "Boundary value problem requires boundary conditions")
        end
    end
    
    # Validate boundary conditions if using advanced BC system
    if length(problem.bc_manager.conditions) > 0
        try
            validate_boundary_conditions(problem.bc_manager, problem)
        catch e
            push!(errors, "Boundary condition validation failed: $e")
        end
    end
    
    if length(errors) > 0
        throw(ArgumentError("Problem validation failed:\n" * join(errors, "\n")))
    end
    
    return true
end

# Substitutions and namespace management
function add_substitution!(problem::Problem, name::String, expression)
    """Add substitution to problem namespace"""
    problem.namespace[name] = expression
end

function expand_substitutions!(problem::Problem)
    """
    Expand substitutions in equations following pattern.
    Following expand(*vars) methods (arithmetic:319-329, operators:704-739).
    """
    
    # Get all variables for expansion
    variables = problem.variables
    
    # Expand equation expressions if they exist
    if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
        # Expand parsed equation data
        for eq_data in problem.equation_data
            try
                # Expand each matrix expression
                for matrix_name in ["M", "L", "F"]
                    if haskey(eq_data, matrix_name) && eq_data[matrix_name] !== nothing
                        expr = eq_data[matrix_name]
                        expanded_expr = expand_expression(expr, variables)
                        eq_data[matrix_name] = expanded_expr
                        @debug "Expanded $matrix_name expression" original=expr expanded=expanded_expr
                    end
                end
                
                # Expand LHS and RHS if they exist
                if haskey(eq_data, "lhs") && eq_data["lhs"] !== nothing
                    eq_data["lhs"] = expand_expression(eq_data["lhs"], variables)
                end
                if haskey(eq_data, "rhs") && eq_data["rhs"] !== nothing
                    eq_data["rhs"] = expand_expression(eq_data["rhs"], variables)
                end
                
            catch e
                @warn "Failed to expand substitutions in equation" exception=e
            end
        end
    else
        # Expand string equations if no equation data exists
        for (i, equation_str) in enumerate(problem.equations)
            try
                # Parse and expand the equation
                lhs, rhs = parse_equation(equation_str, problem.namespace)
                expanded_lhs = expand_expression(lhs, variables)
                expanded_rhs = expand_expression(rhs, variables)
                
                # Reconstruct expanded equation string
                expanded_equation = reconstruct_equation_string(expanded_lhs, expanded_rhs)
                problem.equations[i] = expanded_equation
                
                @debug "Expanded equation $i" original=equation_str expanded=expanded_equation
                
            catch e
                @warn "Failed to expand equation $i: $equation_str" exception=e
            end
        end
    end
    
    # Expand namespace substitutions
    expand_namespace_substitutions!(problem)
    
    @info "Substitution expansion completed for $(length(problem.equations)) equations"
    return problem
end

function expand_expression(expr, variables::Vector)
    """
    Expand expression over specified variables following pattern.
    Following operators:expand and arithmetic:expand methods.
    """
    
    if expr === nothing || isa(expr, String)
        return expr
    end
    
    # Check if expression contains any of the specified variables
    if !has_variables(expr, variables)
        return expr
    end
    
    # Expand based on expression type
    if isa(expr, AddOperator)
        # Expand addition: sum(expand(arg) for arg in args)
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return combine_add_expressions(left_expanded, right_expanded)
        
    elseif isa(expr, MultiplyOperator)
        # Expand multiplication with distribution over addition
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return expand_multiply_expressions(left_expanded, right_expanded, variables)
        
    elseif isa(expr, SubtractOperator)
        # Expand subtraction
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return SubtractOperator(left_expanded, right_expanded)

    elseif isa(expr, DivideOperator)
        # Expand division (only expand numerator, denominator stays as-is)
        left_expanded = expand_expression(expr.left, variables)
        right_expanded = expand_expression(expr.right, variables)
        return DivideOperator(left_expanded, right_expanded)

    elseif isa(expr, PowerOperator)
        # Expand power (expand base, exponent stays as-is typically)
        base_expanded = expand_expression(expr.base, variables)
        exp_expanded = expand_expression(expr.exponent, variables)
        return PowerOperator(base_expanded, exp_expanded)

    elseif isa(expr, NegateOperator)
        # Expand negation
        expanded_operand = expand_expression(expr.operand, variables)
        return NegateOperator(expanded_operand)

    elseif isa(expr, IndexOperator)
        # Expand indexed expression
        expanded_array = expand_expression(expr.array, variables)
        return IndexOperator(expanded_array, expr.indices)

    elseif isa(expr, Union{TimeDerivative, Laplacian, Gradient, Divergence, Differentiate})
        # Expand operators: distribute over operand
        expanded_operand = expand_expression(expr.operand, variables)
        return distribute_operator_over_operand(expr, expanded_operand, variables)

    elseif isa(expr, ConstantOperator) || isa(expr, ZeroOperator)
        # Constants don't expand
        return expr

    else
        # Unknown expression types
        @debug "Unknown expression type in expansion: $(typeof(expr))"
        return expr
    end
end

function has_variables(expr, variables::Vector)
    """Check if expression contains any of the specified variables"""
    
    if expr === nothing || isa(expr, String)
        return false
    end
    
    # Check direct variable reference
    for var in variables
        if expr === var
            return true
        end
    end
    
    # Check by name if possible
    if hasfield(typeof(expr), :name)
        for var in variables
            if hasfield(typeof(var), :name) && expr.name == var.name
                return true
            end
        end
    end
    
    # Recursively check compound expressions
    if isa(expr, Union{AddOperator, SubtractOperator, MultiplyOperator})
        return has_variables(expr.left, variables) || has_variables(expr.right, variables)
    elseif isa(expr, DivideOperator)
        return has_variables(expr.left, variables) || has_variables(expr.right, variables)
    elseif isa(expr, PowerOperator)
        return has_variables(expr.base, variables) || has_variables(expr.exponent, variables)
    elseif isa(expr, NegateOperator)
        return has_variables(expr.operand, variables)
    elseif isa(expr, IndexOperator)
        return has_variables(expr.array, variables)
    elseif isa(expr, Union{TimeDerivative, Laplacian, Gradient, Divergence, Differentiate})
        return has_variables(expr.operand, variables)
    elseif hasfield(typeof(expr), :operand)
        # Generic operand-based operators
        return has_variables(expr.operand, variables)
    end

    return false
end

function combine_add_expressions(left, right)
    """Combine two expressions with addition, flattening nested additions"""
    
    if isa(left, ZeroOperator)
        return right
    elseif isa(right, ZeroOperator)
        return left
    else
        return AddOperator(left, right)
    end
end

function expand_multiply_expressions(left, right, variables::Vector)
    """
    Expand multiplication with distribution over addition.
    Following arithmetic:expand multiplication pattern.
    """
    
    # If either operand is addition involving variables, distribute
    if isa(left, AddOperator) && has_variables(left, variables)
        # (a + b) * c = a*c + b*c
        term1 = expand_multiply_expressions(left.left, right, variables)
        term2 = expand_multiply_expressions(left.right, right, variables)
        return combine_add_expressions(term1, term2)
        
    elseif isa(right, AddOperator) && has_variables(right, variables)
        # a * (b + c) = a*b + a*c
        term1 = expand_multiply_expressions(left, right.left, variables)
        term2 = expand_multiply_expressions(left, right.right, variables)
        return combine_add_expressions(term1, term2)
        
    else
        # Simple multiplication
        return MultiplyOperator(left, right)
    end
end

function distribute_operator_over_operand(operator, expanded_operand, variables::Vector)
    """
    Distribute operator over expanded operand.
    Following operators:_expand_add pattern.
    """
    
    if isa(expanded_operand, AddOperator) && has_variables(expanded_operand, variables)
        # Op(a + b) = Op(a) + Op(b) for linear operators
        left_term = distribute_operator_over_operand(operator, expanded_operand.left, variables)
        right_term = distribute_operator_over_operand(operator, expanded_operand.right, variables)
        return combine_add_expressions(left_term, right_term)
        
    else
        # Apply operator to single operand
        return create_similar_operator(operator, expanded_operand)
    end
end

function create_similar_operator(operator, new_operand)
    """Create new operator of same type with different operand"""
    
    if isa(operator, TimeDerivative)
        return TimeDerivative(new_operand, operator.order)
    elseif isa(operator, Laplacian)
        return Laplacian(new_operand)
    elseif isa(operator, Gradient)
        return Gradient(new_operand, operator.coordsys)
    elseif isa(operator, Divergence)
        return Divergence(new_operand)
    elseif isa(operator, Differentiate)
        return Differentiate(new_operand, operator.coord, operator.order)
    else
        @debug "Unknown operator type for recreation: $(typeof(operator))"
        return operator  # Return original if can't recreate
    end
end

function reconstruct_equation_string(lhs, rhs)
    """Reconstruct equation string from expanded expressions"""
    
    lhs_str = expression_to_string(lhs)
    rhs_str = expression_to_string(rhs)
    
    return "$lhs_str = $rhs_str"
end

function expression_to_string(expr)
    """Convert expression back to string representation"""

    if isa(expr, String)
        return expr
    elseif isa(expr, Number)
        return string(expr)
    elseif isa(expr, ZeroOperator)
        return "0"
    elseif isa(expr, ConstantOperator)
        return string(expr.value)
    elseif isa(expr, AddOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str + $right_str)"
    elseif isa(expr, SubtractOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str - $right_str)"
    elseif isa(expr, MultiplyOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str * $right_str)"
    elseif isa(expr, DivideOperator)
        left_str = expression_to_string(expr.left)
        right_str = expression_to_string(expr.right)
        return "($left_str / $right_str)"
    elseif isa(expr, PowerOperator)
        base_str = expression_to_string(expr.base)
        exp_str = expression_to_string(expr.exponent)
        return "($base_str ^ $exp_str)"
    elseif isa(expr, NegateOperator)
        operand_str = expression_to_string(expr.operand)
        return "(-$operand_str)"
    elseif isa(expr, TimeDerivative)
        operand_str = expression_to_string(expr.operand)
        return "∂t($operand_str)"
    elseif isa(expr, Laplacian)
        operand_str = expression_to_string(expr.operand)
        return "Δ($operand_str)"
    elseif isa(expr, Differentiate)
        operand_str = expression_to_string(expr.operand)
        coord_name = hasfield(typeof(expr.coord), :name) ? expr.coord.name : string(expr.coord)
        return "d($operand_str, $coord_name)"
    elseif isa(expr, Gradient)
        operand_str = expression_to_string(expr.operand)
        return "∇($operand_str)"
    elseif isa(expr, Divergence)
        operand_str = expression_to_string(expr.operand)
        return "∇·($operand_str)"
    elseif hasfield(typeof(expr), :name)
        return expr.name
    else
        return string(expr)
    end
end

"""
    expand_namespace_substitutions!(problem::Problem)

Expand any substitution definitions in problem namespace.

Substitutions are defined in the namespace as strings of the form "pattern = replacement".
They are applied to all equations using word-boundary-aware replacement to avoid
partial matches within variable names.

# Example
```julia
problem.namespace["sub1"] = "nu = 1e-3"
problem.namespace["sub2"] = "f = sin(x)"
# "nu*lap(u)" becomes "1e-3*lap(u)"
# "f + g" becomes "sin(x) + g" but "freq" stays unchanged
```
"""
function expand_namespace_substitutions!(problem::Problem)
    # Look for substitution patterns in namespace
    substitutions = Dict{String, String}()

    for (key, value) in problem.namespace
        if isa(value, String) && contains(value, "=")
            # Potential substitution definition
            try
                lhs, rhs = split_equation(value)
                lhs_clean = strip(lhs)
                rhs_clean = strip(rhs)
                if !isempty(lhs_clean) && !isempty(rhs_clean)
                    substitutions[lhs_clean] = rhs_clean
                end
            catch
                # Not a valid substitution, skip
            end
        end
    end

    if isempty(substitutions)
        return
    end

    @debug "Found substitutions in namespace" substitutions

    # Sort substitutions by pattern length (longest first) to avoid partial replacements
    sorted_patterns = sort(collect(keys(substitutions)), by=length, rev=true)

    # Apply substitutions to equations
    for (i, equation) in enumerate(problem.equations)
        modified_equation = apply_substitutions(equation, substitutions, sorted_patterns)
        if modified_equation != equation
            problem.equations[i] = modified_equation
            @debug "Applied substitution to equation $i" original=equation modified=modified_equation
        end
    end

    # Also apply to boundary conditions if they are strings
    for (i, bc) in enumerate(problem.boundary_conditions)
        if isa(bc, String)
            modified_bc = apply_substitutions(bc, substitutions, sorted_patterns)
            if modified_bc != bc
                problem.boundary_conditions[i] = modified_bc
                @debug "Applied substitution to boundary condition $i"
            end
        end
    end
end

"""
    apply_substitutions(text::String, substitutions::Dict{String,String},
                       sorted_patterns::Vector{String}) -> String

Apply substitutions to text using word-boundary-aware replacement.

Only replaces patterns that appear as whole words (not as substrings of longer
identifiers). Uses regex word boundaries to ensure correct matching.

# Arguments
- `text`: The string to perform substitutions on
- `substitutions`: Dict mapping patterns to their replacements
- `sorted_patterns`: Patterns sorted by length (longest first)
"""
function apply_substitutions(text::String, substitutions::Dict{String,String},
                            sorted_patterns::Vector{String})
    result = text

    for pattern in sorted_patterns
        replacement = substitutions[pattern]

        # Build word-boundary-aware regex pattern
        # Match pattern only when surrounded by non-identifier characters
        # This prevents "nu" from matching inside "enumerate"
        escaped_pattern = escape_regex_chars(pattern)

        # Use word boundaries: pattern must be preceded and followed by
        # non-word characters (or start/end of string)
        regex_pattern = Regex("(?<![a-zA-Z0-9_])$(escaped_pattern)(?![a-zA-Z0-9_])")

        result = replace(result, regex_pattern => replacement)
    end

    return result
end

"""
    escape_regex_chars(s::String) -> String

Escape special regex characters in a string for literal matching.
"""
function escape_regex_chars(s::String)
    # Characters that have special meaning in regex
    special_chars = raw"\.^$*+?{}[]|()"

    result = IOBuffer()
    for c in s
        if c in special_chars
            write(result, '\\')
        end
        write(result, c)
    end

    return String(take!(result))
end

"""
    apply_substitution_recursive(expr, substitutions::Dict)

Apply substitutions to an operator expression tree recursively.
Used when equations are already parsed into operator form.
Returns a new expression tree with substitutions applied (immutable-safe).
"""
function apply_substitution_recursive(expr, substitutions::Dict)
    if expr === nothing
        return expr
    end

    # Handle different expression types
    if isa(expr, ScalarField) && haskey(substitutions, expr.name)
        # Can't directly replace a field, but we can note it
        @debug "Found field $(expr.name) in substitutions but cannot replace parsed field"
        return expr

    elseif isa(expr, AddOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = apply_substitution_recursive(expr.right, substitutions)
        return AddOperator(new_left, new_right)

    elseif isa(expr, SubtractOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = apply_substitution_recursive(expr.right, substitutions)
        return SubtractOperator(new_left, new_right)

    elseif isa(expr, MultiplyOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = isa(expr.right, Number) ? expr.right : apply_substitution_recursive(expr.right, substitutions)
        return MultiplyOperator(new_left, new_right)

    elseif isa(expr, DivideOperator)
        new_left = apply_substitution_recursive(expr.left, substitutions)
        new_right = apply_substitution_recursive(expr.right, substitutions)
        return DivideOperator(new_left, new_right)

    elseif isa(expr, NegateOperator)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return NegateOperator(new_operand)

    elseif isa(expr, PowerOperator)
        new_base = apply_substitution_recursive(expr.base, substitutions)
        new_exp = apply_substitution_recursive(expr.exponent, substitutions)
        return PowerOperator(new_base, new_exp)

    elseif isa(expr, TimeDerivative)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return TimeDerivative(new_operand, expr.order)

    elseif isa(expr, Laplacian)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return Laplacian(new_operand)

    elseif isa(expr, Differentiate)
        new_operand = apply_substitution_recursive(expr.operand, substitutions)
        return Differentiate(new_operand, expr.coord, expr.order)

    elseif hasfield(typeof(expr), :operand)
        # Generic operand-based operator
        new_operand = apply_substitution_recursive(getfield(expr, :operand), substitutions)
        # Try to construct a new instance with the same type
        return expr  # Fallback: return original if can't reconstruct

    elseif hasfield(typeof(expr), :left) && hasfield(typeof(expr), :right)
        # Generic binary operator - return original since we can't reconstruct unknown types
        return expr
    end

    return expr
end

# Problem metadata
function get_variable_names(problem::Problem)
    """Get names of all variables"""
    names = String[]
    for var in problem.variables
        if isa(var, ScalarField)
            push!(names, var.name)
        elseif isa(var, VectorField)
            for comp in var.components
                push!(names, comp.name)
            end
        elseif isa(var, TensorField)
            for i in 1:size(var.components, 1), j in 1:size(var.components, 2)
                push!(names, var.components[i,j].name)
            end
        end
    end
    return names
end

function get_equation_count(problem::Problem)
    """Get total number of equations including boundary conditions"""
    return length(problem.equations) + length(problem.boundary_conditions)
end

function get_variable_count(problem::Problem)
    """Get total number of scalar variables"""
    count = 0
    for var in problem.variables
        if isa(var, ScalarField)
            count += 1
        elseif isa(var, VectorField)
            count += length(var.components)
        elseif isa(var, TensorField)
            count += length(var.components)
        end
    end
    return count
end

# ============================================================================
# Convenience Macros
# ============================================================================

"""
    @equations problem begin
        "equation1"
        "equation2"
        ...
    end

Add multiple equations to a problem with cleaner syntax.

# Example
```julia
problem = IVP([u, v, p])

@equations problem begin
    "∂t(u) - ν*Δ(u) + ∇(p) = -u⋅∇(u)"
    "∂t(v) - ν*Δ(v) + ∇(p) = -u⋅∇(v)"
    "div(u) = 0"
end
```

Equivalent to:
```julia
add_equation!(problem, "∂t(u) - ν*Δ(u) + ∇(p) = -u⋅∇(u)")
add_equation!(problem, "∂t(v) - ν*Δ(v) + ∇(p) = -u⋅∇(v)")
add_equation!(problem, "div(u) = 0")
```
"""
macro equations(problem, block)
    if block.head != :block
        error("@equations requires a begin...end block")
    end

    calls = Expr[]
    for arg in block.args
        # Skip LineNumberNodes
        if arg isa LineNumberNode
            continue
        end
        # Each argument should be a string (equation)
        push!(calls, :(add_equation!($(esc(problem)), $(esc(arg)))))
    end

    return Expr(:block, calls...)
end

"""
    @bcs problem begin
        "bc1"
        "bc2"
        ...
    end

Add multiple boundary conditions to a problem with cleaner syntax.

# Example
```julia
@bcs problem begin
    "u(z=0) = 0"
    "u(z=Lz) = 0"
    "integ(p) = 0"
end
```

Equivalent to:
```julia
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=Lz) = 0")
add_bc!(problem, "integ(p) = 0")
```
"""
macro bcs(problem, block)
    if block.head != :block
        error("@bcs requires a begin...end block")
    end

    calls = Expr[]
    for arg in block.args
        # Skip LineNumberNodes
        if arg isa LineNumberNode
            continue
        end
        push!(calls, :(add_bc!($(esc(problem)), $(esc(arg)))))
    end

    return Expr(:block, calls...)
end

"""
    @substitutions problem begin
        "name1" => value1
        "name2" => value2
        ...
    end

Add multiple substitutions to a problem with cleaner syntax.

# Example
```julia
@substitutions problem begin
    "nu" => 1e-3
    "kappa" => 1e-4
    "Ra" => 1e6
end
```

Equivalent to:
```julia
add_substitution!(problem, "nu", 1e-3)
add_substitution!(problem, "kappa", 1e-4)
add_substitution!(problem, "Ra", 1e6)
```
"""
macro substitutions(problem, block)
    if block.head != :block
        error("@substitutions requires a begin...end block")
    end

    calls = Expr[]
    for arg in block.args
        # Skip LineNumberNodes
        if arg isa LineNumberNode
            continue
        end
        # Expect pair syntax: "name" => value
        if arg isa Expr && arg.head == :call && arg.args[1] == :(=>)
            name = arg.args[2]
            value = arg.args[3]
            push!(calls, :(add_substitution!($(esc(problem)), $(esc(name)), $(esc(value)))))
        else
            error("@substitutions expects \"name\" => value pairs")
        end
    end

    return Expr(:block, calls...)
end

# ============================================================================
# Exports
# ============================================================================

# Export problem types
export Problem, IVP, LBVP, NLBVP, EVP

# Export core API functions
export add_equation!, add_bc!, add_substitution!
export @equations, @bcs, @substitutions
export set_parameter!, get_parameter, register_tau_field!

# Export stochastic forcing integration
export add_stochastic_forcing!, has_stochastic_forcing, get_stochastic_forcing

# Export temporal filter integration
export add_temporal_filter!, has_temporal_filters, get_temporal_filter, get_all_temporal_filters

# Export problem query functions
export get_variable_names, get_equation_count, get_variable_count

# Export problem building/validation functions
export build_problem_namespace, build_matrices, build_matrix_expressions!
export validate_problem, parse_equation, parse_expression

# Export BC string parsing functions
export parse_bc_string, parse_neumann_bc_string, parse_robin_bc_string, parse_stress_free_bc_string

# Export equation data functions
export set_equation_condition!, set_valid_modes!, get_matrix_expression
export is_equation_valid, check_equation_condition
