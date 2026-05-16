"""
Problem formulation classes
"""

# LinearAlgebra, SparseArrays already in Tarang.jl

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

    # Register coordinate names (e.g. "x", "y", "z") from every basis in
    # every variable's bases so that boundary-condition RHS strings like
    # `T(z=0) = 1 + 0.1*sin(2*pi*x/Lx)` parse without "Unknown variable: x"
    # warnings. The placeholder is an `UnknownOperator` carrying the coord
    # name — the parser treats it as a generic opaque symbol, and at
    # solver-build time `_apply_bc_values_to_equations!` OVERWRITES the BC's
    # `equation_data["F"]` slot with an `ArrayOperator` evaluated by the
    # dedicated spatial-BC evaluator against the coordinate grid arrays, so
    # the placeholder value is never consulted by the solver at runtime.
    for var in variables
        if hasproperty(var, :bases)
            bases = getfield(var, :bases)
            if bases !== nothing
                for basis in bases
                    basis === nothing && continue
                    if hasproperty(basis, :meta) && hasproperty(basis.meta, :element_label)
                        label = String(basis.meta.element_label)
                        if !isempty(label) && !haskey(ns, label)
                            ns[label] = UnknownOperator(label)
                        end
                    end
                end
            end
        end
    end

    # Register the default time variable name `"t"` as a placeholder so
    # time-dependent BC expressions like `"T(z=0) = sin(2*pi*t)"` parse
    # without "Unknown variable: t" warnings. The same precedence rules
    # apply — user-supplied namespace entries and problem variables named
    # `t` will override the placeholder. At solver-build time,
    # `_apply_bc_values_to_equations!` evaluates the BC's original string
    # against the current time via `evaluate_bc_value`, so the placeholder
    # is never consulted at runtime.
    #
    # If you use a different time-variable name (via `set_time_variable!`
    # with a custom string), pass it through your `namespace=Dict(...)`
    # constructor argument or via `add_parameters!`, and it will override
    # this default entry before any equation is parsed.
    if !haskey(ns, "t")
        ns["t"] = UnknownOperator("t")
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

Parse boundary condition string into components.

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
        # Match canonical operator syntax emitted by `bc_to_equation`:
        # d(<field>,<coord>)(<coord>=<pos>)=<value>
        # e.g., "d(u,z)(z=0)=0"
        pattern = r"^d\(([a-zA-Z_][a-zA-Z0-9_]*),([a-zA-Z_][a-zA-Z0-9_]*)(?:,([0-9]+))?\)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)=(.+)$"
        m = match(pattern, s)
    end

    if m === nothing
        # Match compact derivative aliases used in some docs/comments:
        # d<coord>(<field>)(<coord>=<pos>)=<value>
        # e.g., "dz(u)(z=0)=0"
        pattern = r"^d([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)\(([a-zA-Z_][a-zA-Z0-9_]*)=([^)]+)\)=(.+)$"
        m = match(pattern, s)
        if m !== nothing
            deriv_coord = String(m.captures[1])
            field_name = String(m.captures[2])
            bc_coord = String(m.captures[3])
            pos_str = String(m.captures[4])
            val_str = String(m.captures[5])
            return _finish_neumann_bc_parse(bc_string, deriv_coord, field_name, bc_coord, pos_str, val_str)
        end
    end

    if m === nothing
        # Try simpler format: field(coord=pos) = value (same as Dirichlet but caller knows it's Neumann)
        return parse_bc_string(bc_string)
    end

    if length(m.captures) == 6
        field_name = String(m.captures[1])
        deriv_coord = String(m.captures[2])
        order_str = m.captures[3]
        if order_str !== nothing && parse(Int, order_str) != 1
            throw(ArgumentError("Only first-order Neumann BC strings are supported: '$bc_string'"))
        end
        bc_coord = String(m.captures[4])
        pos_str = String(m.captures[5])
        val_str = String(m.captures[6])
    else
        deriv_coord = String(m.captures[1])
        field_name = String(m.captures[2])
        bc_coord = String(m.captures[3])
        pos_str = String(m.captures[4])
        val_str = String(m.captures[5])
    end

    return _finish_neumann_bc_parse(bc_string, deriv_coord, field_name, bc_coord, pos_str, val_str)
end

function _finish_neumann_bc_parse(bc_string::String, deriv_coord::String, field_name::String,
                                  bc_coord::String, pos_str::String, val_str::String)
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

"""
    add_equation!(problem, equation::String)

Add an equation to the problem.  The system matrix row count for each equation
is determined automatically from the expression's output type (scalar, vector,
etc.) — equation ordering does not matter.

# Example
```julia
problem = IVP([q, ψ, u, tau_ψ])

add_equation!(problem, "∂t(q) + nu*Δ⁴(q) = -u⋅∇(q)")   # scalar eq  → D rows
add_equation!(problem, "Δ(ψ) + tau_ψ - q = 0")           # scalar eq  → D rows
add_equation!(problem, "u - skew(grad(ψ)) = 0")           # vector eq  → 2D rows
add_bc!(problem, "integ(ψ) = 0")                           # constraint → 1 row
```
"""
function add_equation!(problem::Problem, equation::String)
    push!(problem.equations, equation)
    # Quick string-level check for common mistakes (linear terms on RHS)
    _check_equation_placement(equation, problem)
end

# Linear operator patterns that should be on the LHS
const _LINEAR_OP_PATTERNS = [r"Δ\(", r"lap\(", r"∇\(", r"div\(", r"curl\(", r"∇²\("]

"""Check if equation string has linear operator terms or parameter*field terms on the RHS."""
function _check_equation_placement(equation::String, problem::Problem)
    # Skip BCs like "u(z=0) = 0"
    occursin(r"\w+\(\w+=", equation) && return

    parts = split(equation, "="; limit=2)
    length(parts) != 2 && return
    lhs, rhs = strip(parts[1]), strip(parts[2])

    # Mask nonlinear advection terms (u⋅∇(q)) so ∇( doesn't false-positive
    rhs_check = replace(rhs, "⋅∇(" => "⋅_advect_(")

    # Check for linear operators on RHS
    for pat in _LINEAR_OP_PATTERNS
        if occursin(pat, rhs_check)
            @warn "Linear operator on RHS: \"$equation\"\n" *
                  "  Move linear terms to LHS for correct IMEX time splitting.\n" *
                  "  Example: \"∂t(u) = nu*Δ(u)\" → \"∂t(u) - nu*Δ(u) = 0\""
            return
        end
    end

    # Check for parameter*field patterns on RHS (e.g., Ra*Pr*T)
    field_names = String[f.name for f in problem.variables if hasfield(typeof(f), :name)]

    for fname in field_names
        # Skip if field name doesn't appear on RHS at all
        occursin(fname, rhs) || continue

        # Skip if field is inside a function call like ∂x(u), Δ(u) — that's an operator, already caught above
        occursin(fname * "(", rhs) && continue
        occursin("(" * fname * ")", rhs) && continue

        # Skip if field is part of a nonlinear product:
        #   field*field: u*T, u*v
        #   field*operator(field): u*∂x(T), u⋅∇(T)
        is_nonlinear = false
        for f2 in field_names
            f2 == fname && continue
            # u*T or T*u
            if occursin(f2 * "*" * fname, rhs) || occursin(fname * "*" * f2, rhs)
                is_nonlinear = true; break
            end
            # u*∂x(T) or u*Δ(T) etc
            if occursin(fname * "*∂", rhs) || occursin(fname * "*Δ", rhs) ||
               occursin(fname * "*∇", rhs) || occursin(fname * "*lap", rhs) ||
               occursin(fname * "⋅∇", rhs) || occursin(fname * "*div", rhs)
                is_nonlinear = true; break
            end
        end
        is_nonlinear && continue

        # Field appears on RHS without another field multiplying it → linear term
        @warn "Linear term on RHS: \"$equation\"\n" *
              "  '$fname' appears linearly on RHS (multiplied only by parameters).\n" *
              "  Move to LHS for correct IMEX time splitting."
        return
    end
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

"""
    add_bc!(problem, bc::String)

Add a boundary condition using string syntax.

# Examples
```julia
add_bc!(problem, "u(z=0) = 0")           # Dirichlet
add_bc!(problem, "integ(p) = 0")          # Integral constraint
```

For common physical BCs, prefer the named helpers:
```julia
no_slip!(problem, "u", "z", 0.0)         # u = 0 (Dirichlet)
fixed_value!(problem, "T", "z", 0.0, 1.0) # T = 1 (Dirichlet)
free_slip!(problem, "u", "z", 0.0)        # du/dz = 0 (Neumann)
insulating!(problem, "T", "z", 1.0)       # dT/dz = 0 (Neumann)
```

For structured BC objects, pass an `AbstractBoundaryCondition` directly:
```julia
add_bc!(problem, dirichlet_bc("u", "z", 0.0, 0.0))
```
"""
function add_bc!(problem::Problem, bc::String)
    # Always store the raw string so `_merge_boundary_conditions!` can push it
    # into `problem.equations` during solver build.
    push!(problem.boundary_conditions, bc)

    # ALSO register a concrete `DirichletBC` / `NeumannBC` / ... in the BC
    # manager with auto-detected time / space dependency flags. Without this
    # step, a string like `"T(z=0) = 1 + 0.0001*sin(2*pi*x/4)"` would never
    # enter `bc_manager.space_dependent_bcs`, so `_apply_bc_values_to_equations!`
    # would never fire and the equation_data["F"] slot would retain the
    # parser's symbolic garbage — silently degrading to a zero-F BC and
    # causing max|T| to decay from the conduction profile to zero even
    # though a `T(z=0) = 1 + ε*sin(x)` BC was declared.
    try
        _register_string_bc!(problem, bc)
    catch err
        @debug "add_bc!: could not register string BC with bc_manager; string-only path may miss space/time-dependent handling" bc err
    end
end

"""
    _register_string_bc!(problem, bc_string)

Parse a BC string like `"T(z=0) = <value>"` (Dirichlet) or
`"∂z(T)(z=0) = <value>"` (Neumann), and register a concrete
`DirichletBC` / `NeumannBC` in `problem.bc_manager` so that time- and
space-dependency detection runs on the value expression. This lets
`_apply_bc_values_to_equations!` find and refresh the BC at solver build
and on every step.

Gracefully returns without registering if the string doesn't match any
known BC pattern (in which case the raw-string path in
`_merge_boundary_conditions!` still pushes the BC equation into the
system, but time/space dependency handling is disabled for it — the
user would see wrong enforcement for non-constant values in that case).
"""
function _register_string_bc!(problem::Problem, bc_string::String)
    # Detect Neumann first (has `∂coord(field)(...)` prefix). Fall back to
    # Dirichlet for the usual `field(coord=pos) = value` form.
    stripped = replace(bc_string, " " => "")
    is_neumann = startswith(stripped, "∂") ||
                 startswith(stripped, "d(") ||
                 occursin(r"^d[a-zA-Z_][a-zA-Z0-9_]*\(", stripped)

    if is_neumann
        parts = try
            parse_neumann_bc_string(bc_string)
        catch
            return  # unparseable — fall back to raw-string path
        end
        field_name, coord, position, value = parts
        bc_obj = neumann_bc(field_name, coord, position, value)
        add_bc!(problem.bc_manager, bc_obj)
        return
    end

    # Dirichlet path.
    parts = try
        parse_bc_string(bc_string)
    catch
        return  # unparseable (e.g. `integ(p) = 0` — handled via raw string)
    end
    field_name, coord, position, value = parts

    # Some "BC-like" strings (e.g. `integ(p) = 0`) slip through `parse_bc_string`
    # with a bogus `field_name` / `coord`. Skip those — they're not actual
    # Dirichlet BCs and the raw-string path handles them as plain algebraic
    # constraint equations.
    if field_name == "integ" || field_name == "average" || field_name == "avg"
        return
    end

    bc_obj = dirichlet_bc(field_name, coord, position, value)
    add_bc!(problem.bc_manager, bc_obj)
    return
end
