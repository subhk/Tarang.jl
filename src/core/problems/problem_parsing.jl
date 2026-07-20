
"""
    add_bc!(problem, bc::AbstractBoundaryCondition)

Add a structured boundary condition object to the problem.

See also: [`no_slip!`](@ref), [`fixed_value!`](@ref), [`free_slip!`](@ref),
[`insulating!`](@ref), [`dirichlet_bc`](@ref), [`neumann_bc`](@ref)
"""
function add_bc!(problem::Problem, bc::AbstractBoundaryCondition)
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

# Boundary conditions can also be passed to add_equation!() following convention:
#   add_equation!(problem, "u(z=0) = 0")  # Dirichlet BC
#   add_equation!(problem, "dz(u)(z=1) = 0")  # Neumann BC

"""Register tau field for boundary condition enforcement"""
function register_tau_field!(problem::Problem, name::String, field)
    register_tau_field!(problem.bc_manager, name, field)
    
    # Add to problem variables if not already present
    if !any(var -> (isa(var, ScalarField) && var.name == name), problem.variables)
        push!(problem.variables, field)
    end
    
    return problem
end

"""Set parameter value"""
function set_parameter!(problem::Problem, name::String, value)
    problem.parameters[name] = value
end

"""Get parameter value"""
function get_parameter(problem::Problem, name::String, default=nothing)
    return get(problem.parameters, name, default)
end

# Equation parsing following structure
"""
    Parse equation string into operator expressions following formulation requirements.
    
    Requires:
    - LHS: Linear terms only (first-order in time derivatives, linear in variables)
    - RHS: Nonlinear terms, time-dependent terms, non-constant coefficients
    - Form: M·∂tX + L·X = F(X,t)
    
    Following problems:add_equation pattern (problems:65-80).
    """
function parse_equation(equation::String, namespace::Dict{String, Any})
    
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

"""
    Parse LHS expression ensuring it contains only linear terms.
    
    Linear terms allowed on LHS:
    - Time derivatives (first-order): dt(u)
    - Linear spatial derivatives: d(u, x), lap(u)
    - Constant coefficients: 2*dt(u), -viscosity*lap(u)
    - Linear combinations: dt(u) + diffusion*lap(u)
    
    Non-constant coefficients should be moved to RHS.
    """
function parse_linear_expression(expr_str::AbstractString, namespace::Dict{String, Any})

    expr = parse_expression(expr_str, namespace)
    
    return expr
end

# Note: has_variables() is now replaced by the proper has() method from field.jl/operators.jl
# following the spectral pattern for variable linearity checking

"""
    is_linear_expression(expr, namespace::Dict{String, Any}) -> Bool

Check if expression is linear in the problem variables.
Follows the spectral approach:

- Multiplication is nonlinear ONLY if BOTH factors contain problem variables
- Addition: all addends must be linear
- Linear operators (grad, div, lap, skew, etc.): delegate to operand
- Nonlinear operators (power, sqrt applied to variables): nonlinear

Key insight: `c * u` is linear (c is coefficient),
but `u * v` is nonlinear (both are variables).

Uses the proper has() and require_linearity() methods from field.jl/operators.jl
which follow the spectral pattern exactly.
"""
function is_linear_expression(expr, namespace::Dict{String, Any})
    # Extract problem variables from namespace
    variables = collect_problem_variables(namespace)

    # Use require_linearity() following spectral pattern
    # allow_affine=true because constant terms are allowed on LHS
    try
        require_linearity(expr, variables...; allow_affine=true)
        return true
    catch e
        @debug "Nonlinearity detected" exception=e expr=expr
        return false
    end
end

"""Collect all problem variables (fields) from namespace."""
function collect_problem_variables(namespace::Dict{String, Any})
    variables = []
    for (name, val) in namespace
        if isa(val, ScalarField) || isa(val, VectorField) || isa(val, TensorField)
            push!(variables, val)
        end
    end
    return variables
end

"""Check if expression is a direct reference to a problem variable."""
function is_variable_reference(expr, namespace::Dict{String, Any})
    if hasfield(typeof(expr), :name) && isa(getfield(expr, :name), String)
        var_name = getfield(expr, :name)
        if haskey(namespace, var_name)
            val = namespace[var_name]
            return isa(val, ScalarField) || isa(val, VectorField) || isa(val, TensorField)
        end
    end
    # Also check if it's directly a field
    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        return true
    end
    return false
end

"""
    Check if expression represents a constant coefficient.
    Non-constant coefficients should be moved to RHS per requirements.
    """
function is_constant_coefficient(expr, namespace::Dict{String, Any})
    
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

"""
    Validate that equation follows structure requirements.

    Requirements:
    1. LHS must be linear in dependent variables
    2. LHS must be first-order in time derivatives
    3. RHS can contain nonlinear terms
    4. Coefficients that the implicit operator cannot represent belong on the RHS

    `variables` is the problem's list of solved-for variables (`problem.variables`).
    It is what tells a COEFFICIENT factor (`q` in `q*lap(u)`) apart from a
    variable-dependent factor (`u` in `u*dx(u)`) — the same distinction the matrix
    builder makes via `_references_variable`. Pass it whenever it is available;
    without it the RHS "move this to the LHS" check is skipped rather than guessed,
    because a forcing field on the RHS is indistinguishable from a misplaced linear
    term once the variable list is gone.
    """
function validate_equation_structure(LHS, RHS, original_equation::String;
                                     variables=nothing)

    # Check for temporal derivatives on RHS (not allowed)
    if contains_time_derivatives(RHS)
        throw(ArgumentError("Time derivatives found on RHS of equation: \"$original_equation\". " *
                           "All time derivatives must be on LHS."))
    end

    # IMEX placement only matters for evolution equations (those with ∂t).
    # Diagnostic/constraint equations (e.g. "u - skew(grad(ψ)) = 0", "integ(ψ) = 0")
    # have no time splitting, so skip the check.
    if !contains_time_derivatives(LHS)
        return true
    end

    vars = _problem_variable_operands(variables)

    # Collect misplaced terms and suggest correction
    lhs_terms = _collect_addends(LHS)
    rhs_terms = _collect_addends(RHS)

    # A RHS term belongs on the LHS only when it is (a) linear AND implicit-capable,
    # (b) not a bare constant, and (c) actually about a solved-for variable. Without
    # (c) every constant-in-time forcing field on the RHS — a completely legitimate
    # explicit term with no matrix column to move into — would be reported. With no
    # variable list at all, (c) is unanswerable and the check is skipped rather than
    # guessed: `u*dx(u)` and `q*dx(u)` are indistinguishable without it, and the
    # advice for one is exactly wrong for the other.
    _moves_to_lhs(t) = _references_problem_variable(t, vars) &&
                       _is_linear_term(t, vars) && !_is_constant_term(t)

    nonlinear_on_lhs = filter(t -> !_is_linear_term(t, vars), lhs_terms)
    linear_on_rhs = filter(_moves_to_lhs, rhs_terms)

    if !isempty(nonlinear_on_lhs) || !isempty(linear_on_rhs)
        # Build suggested equation. `0` addends are dropped: an equation written
        # "... = 0" would otherwise be suggested back as "0 - (-(q * Δ(u)))".
        new_lhs_terms = filter(t -> _is_linear_term(t, vars) && !isa(t, ZeroOperator), lhs_terms)
        # Move linear RHS terms to LHS (flip sign)
        for t in linear_on_rhs
            push!(new_lhs_terms, _negate_term(t))
        end
        new_rhs_terms = filter(t -> !_moves_to_lhs(t) && !isa(t, ZeroOperator), rhs_terms)
        # Move nonlinear LHS terms to RHS (flip sign)
        for t in nonlinear_on_lhs
            push!(new_rhs_terms, _negate_term(t))
        end

        suggested_lhs = _format_sum(new_lhs_terms)
        suggested_rhs = isempty(new_rhs_terms) ? "0" : _format_sum(new_rhs_terms)

        @warn "Equation has misplaced terms: \"$original_equation\"\n" *
              "  Suggested form: $suggested_lhs = $suggested_rhs"
    end

    return true
end

"""
    validate_added_equation(problem, equation::String) -> Bool

Run [`validate_equation_structure`](@ref) on an equation as it is ADDED to the
problem, so the guidance lands at the point of the mistake instead of at solve
time (or, as was the case before, never — `validate_equation_structure` only ran
inside `parse_equation`, which no solver path calls).

Equations are stored as strings and parsed later, so this parses into a throwaway
tree. Three properties keep that safe:

* Only equations that actually contain `dt(`/`∂t(` are parsed at all. Boundary
  conditions, constraints and diagnostic equations are skipped without being
  touched, and `validate_equation_structure` would have returned early for them
  anyway.
* The parse runs under a null logger. A parameter that has not been added to the
  namespace yet (equation written before `add_parameters!`) would otherwise emit
  "Unknown variable" warnings that no user sees today, because the real parse
  happens later, after the namespace is complete. Nothing is lost: that parse
  still runs, with logging, at build time.
* Anything unexpected is swallowed. A structure check must never be able to stop
  an equation from being added.
"""
function validate_added_equation(problem, equation::String)
    _has_time_derivative_syntax(equation) || return true
    try
        lhs_str, rhs_str = split_equation(equation)
        variables = hasfield(typeof(problem), :variables) ? problem.variables : nothing
        namespace = hasfield(typeof(problem), :namespace) ? problem.namespace : Dict{String, Any}()
        LHS = nothing
        RHS = nothing
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            LHS = parse_expression(strip(lhs_str), namespace)
            RHS = parse_expression(strip(rhs_str), namespace)
        end
        validate_equation_structure(LHS, RHS, equation; variables=variables)
    catch e
        if isa(e, ArgumentError)
            # e.g. a time derivative on the RHS. Report it, but never throw: the
            # equation has already been stored and add_equation! has never thrown.
            @warn "Equation structure: $(e.msg)"
        else
            @debug "Structure check skipped for equation" equation exception=e
        end
    end
    return true
end

"""True when the equation string contains a time derivative (`dt(...)`/`∂t(...)`)."""
function _has_time_derivative_syntax(equation::AbstractString)
    return occursin(r"(?:^|[^A-Za-z0-9_])(?:dt|∂t)\s*\(", equation)
end

"""Flip a term's sign for the suggested form, collapsing `-(-x)` back to `x`."""
_negate_term(t) = isa(t, NegateOperator) ? t.operand : NegateOperator(t)

"""Collect top-level addends from an expression tree (split by +/-)."""
function _collect_addends(expr)
    if isa(expr, AddOperator)
        return vcat(_collect_addends(expr.left), _collect_addends(expr.right))
    elseif isa(expr, SubtractOperator)
        return vcat(_collect_addends(expr.left), Any[NegateOperator(t) for t in _collect_addends(expr.right)])
    else
        return Any[expr]
    end
end

"""
    _is_linear_term(expr, variables=nothing) -> Bool

True when `expr` is a term the IMPLICIT (left-hand) side can hold: linear in the
solved-for variables *and* built from coefficients the implicit operator can
actually represent.

Nonlinearity is decided by exclusion rather than by an allowlist of linear
operator types — an allowlist silently mis-reports every operator nobody
remembered to add to it (`Δ⁴`, `Copy`, `Grid`, …). A term is nonlinear only when
it is one of the constructs that genuinely cannot become a matrix:

* a product whose factors are BOTH variable-dependent (`u*dx(u)`), or whose
  coefficient factor is not implicit-capable (`nu_e*lap(u)` on a Fourier domain);
* a division by anything other than a constant (`lap(u)/q` — the matrix builder
  raises on the reciprocal of a field);
* a power other than `^1` of a variable-dependent base (`u^2`);
* a nonlinear function of a variable-dependent operand (`sin(u)`).
"""
function _is_linear_term(expr, variables=nothing)
    if isa(expr, NegateOperator)
        return _is_linear_term(expr.operand, variables)
    elseif isa(expr, AddOperator) || isa(expr, SubtractOperator)
        return _is_linear_term(expr.left, variables) && _is_linear_term(expr.right, variables)
    elseif isa(expr, MultiplyOperator)
        # c * L(u) is linear when one side is a coefficient the implicit operator
        # can represent — a number, a constant parameter, or a field-valued NCC
        # the matrix builder actually knows how to multiply by.
        left_is_coeff = _is_implicit_coefficient(expr.left, variables)
        right_is_coeff = _is_implicit_coefficient(expr.right, variables)
        if left_is_coeff && right_is_coeff
            return true                       # coefficient * coefficient: a constant term
        elseif left_is_coeff
            return _is_linear_term(expr.right, variables)
        elseif right_is_coeff
            return _is_linear_term(expr.left, variables)
        else
            return false                      # variable * variable = nonlinear
        end
    elseif isa(expr, DivideOperator)
        # Only division by a CONSTANT scales a matrix block. A field-valued
        # denominator raises in the matrix builder (1/q is not an implicit operator),
        # so it must be reported, not accepted.
        return _is_constant_term(expr.right) && _is_linear_term(expr.left, variables)
    elseif isa(expr, PowerOperator)
        _is_constant_term(expr) && return true
        exponent = _get_constant_value(expr.right)
        return exponent !== nothing && exponent == 1 &&
               _is_linear_term(expr.left, variables)
    elseif isa(expr, Union{GeneralFunction, UnaryGridFunction, NonlinearOperator})
        # sin(u), exp(u), … are linear only if their operand holds no variable.
        return !_references_problem_variable(expr, variables)
    elseif isa(expr, Future)
        return _is_linear_future_term(expr, variables)
    elseif hasfield(typeof(expr), :operand)
        # A single-operand operator is exactly as linear as what it wraps. Stopping
        # at the outer node instead would call `div(nu_e*grad(u))` an implicit term
        # because the outermost thing is a Divergence — and then advise moving an
        # explicitly-written variable-viscosity diffusion onto the LHS, where it
        # raises.
        return _is_linear_term(getfield(expr, :operand), variables)
    else
        # Bare fields, constants and unresolved symbols. Reporting an unrecognised
        # node as nonlinear would be a guess, and a wrong guess produces wrong advice.
        return true
    end
end

"""Linearity for the object-syntax (`Future`) arithmetic nodes."""
function _is_linear_future_term(expr, variables)
    args = collect(Any, future_args(expr))
    if isa(expr, Multiply)
        dependent = filter(a -> _references_problem_variable(a, variables), args)
        length(dependent) > 1 && return false
        coefficients = filter(a -> !_references_problem_variable(a, variables), args)
        all(a -> _is_implicit_coefficient(a, variables), coefficients) || return false
        return all(a -> _is_linear_term(a, variables), dependent)
    elseif isa(expr, Divide)
        length(args) >= 2 || return true
        return _is_constant_term(args[2]) && _is_linear_term(args[1], variables)
    elseif isa(expr, Add) || isa(expr, Subtract) || isa(expr, Negate)
        return all(a -> _is_linear_term(a, variables), args)
    end
    # DotProduct / CrossProduct / Outer / Power and friends: nonlinear as soon as
    # more than one operand carries a variable (and Power is nonlinear with even one).
    dependent = count(a -> _references_problem_variable(a, variables), args)
    return dependent == 0
end

"""
    _is_constant_term(expr) -> Bool

True for a CONSTANT scalar coefficient — precisely the set the matrix builder
folds into a scalar multiplier (`_is_const_or_param`, problem_matrices_spectral.jl):
numbers, `ConstantOperator`/`ZeroOperator`, arithmetic whose leaves are all
constant, and a `ScalarField` that is 0-D or holds a single value (a tau variable
or a scalar parameter stored as a field).

Also accepts a constant `VectorField`/`TensorField` — the unit vectors `ez` in
`Ra*Pr*b*ez`. Those are rank-changing block expansions rather than scalar
multipliers, but they are long-supported implicit coefficients
(`_build_constant_field_matrix`), so calling them nonlinear would flag standard
Boussinesq equations.
"""
function _is_constant_term(expr)
    isa(expr, NegateOperator) && return _is_constant_term(expr.operand)
    _is_const_or_param(expr) && return true
    if isa(expr, AddOperator) || isa(expr, SubtractOperator) ||
       isa(expr, MultiplyOperator) || isa(expr, DivideOperator) || isa(expr, PowerOperator)
        return _is_constant_term(expr.left) && _is_constant_term(expr.right)
    end
    if isa(expr, VectorField) || isa(expr, TensorField)
        comps = expr.components
        return !isempty(comps) && all(_is_const_or_param, comps)
    end
    return false
end

"""
    _is_implicit_coefficient(expr, variables=nothing) -> Bool

True when `expr` can be the COEFFICIENT factor of an implicit (LHS) product — that
is, when the matrix builder turns it into an operator instead of raising
`ImplicitNCCError`. This mirrors, in order, the three tests the builder applies:

1. `_references_variable` — a factor that mentions a solved-for variable is not a
   coefficient at all, it is the variable-dependent side of the product.
2. `_is_const_or_param` — numbers, constant parameters, 0-D/single-point fields
   (see [`_is_constant_term`](@ref)).
3. `_implicit_ncc_matrix` — a bare field coefficient is representable only when it
   varies along exactly one Jacobi/Chebyshev axis and is constant along every
   Fourier axis. A coefficient that varies along a Fourier axis couples Fourier
   modes, has no per-mode matrix, and genuinely does NOT belong on the LHS.

The narrower predicate this replaces accepted only `Number`/`ZeroOperator`/
`ConstantOperator`, so it reported `dt(u) - q(z)*lap(u) = 0` and
`dt(u) - nu0*lap(u) = 0` as misplaced and advised moving them to the RHS — advice
that is wrong for terms the implicit path builds correctly.
"""
function _is_implicit_coefficient(expr, variables=nothing)
    _references_problem_variable(expr, variables) && return false
    isa(expr, NegateOperator) && return _is_implicit_coefficient(expr.operand, variables)
    _is_constant_term(expr) && return true
    # A symbol that is not in the namespace yet (the equation was written before
    # `add_parameters!`) parses to a placeholder. Assume it is the scalar parameter
    # it looks like; treating it as a nonlinearity would invent a problem.
    isa(expr, UnknownOperator) && return true

    field = _ncc_direct_field(expr)      # sees through Grid/Coeff/Convert/Copy only
    field === nothing && return false
    isempty(field.bases) && return true
    any(b -> b === nothing, field.bases) && return false
    count(b -> isa(b, JacobiBasis), field.bases) == 1 || return false
    return !_coefficient_varies_along_fourier(field)
end

"""
    _coefficient_varies_along_fourier(field) -> Bool

True when a coefficient field's data varies along one of its Fourier/periodic axes,
which is what makes it unrepresentable in the per-mode implicit operator.

Reads whichever layout the field is ALREADY in — never transforms. `add_equation!`
must not mutate a user's field as a side effect of validating a string, and a field
whose data is not set yet is zero, which the matrix builder itself treats as a
representable (identically zero) coefficient. Any surprise is reported as "does not
vary", because the cost of guessing wrong here is a false warning.
"""
function _coefficient_varies_along_fourier(field)
    fourier_axes = findall(b -> isa(b, FourierBasis), field.bases)
    isempty(fourier_axes) && return false
    try
        layout = hasfield(typeof(field), :current_layout) ?
                 getfield(field, :current_layout) : :g
        data = layout === :c ? get_coeff_data(field) : get_grid_data(field)
        data === nothing && return false
        arr = Array(data)
        (isempty(arr) || ndims(arr) != length(field.bases)) && return false
        scale = maximum(abs, arr)
        scale == 0 && return false          # identically zero: genuinely representable
        for axis in fourier_axes
            size(arr, axis) > 1 || continue
            if layout === :c
                # Constant along a Fourier axis <=> only the DC coefficient is nonzero.
                non_dc = selectdim(arr, axis, 2:size(arr, axis))
                (!isempty(non_dc) && maximum(abs, non_dc) > 1e-8 * scale) && return true
            else
                reference = selectdim(arr, axis, 1)
                for k in 2:size(arr, axis)
                    maximum(abs, selectdim(arr, axis, k) .- reference) > 1e-8 * scale &&
                        return true
                end
            end
        end
    catch e
        @debug "Could not inspect coefficient field for Fourier variation" exception=e
        return false
    end
    return false
end

"""
    _problem_variable_operands(variables) -> Vector{Operand}

The problem's variables, plus the scalar components of every vector/tensor
variable. `_detect_equation_variables` matches by NAME, so the component `u_x`
produced by expanding `u⋅∇(q)` does not match the `VectorField` `u` it came from —
without the components in the list, `u_x` looks like an innocent coefficient field
and `u⋅∇(q)` is mistaken for a linear term.
"""
function _problem_variable_operands(variables)
    result = Operand[]
    variables === nothing && return result
    for var in variables
        isa(var, Operand) || continue
        push!(result, var)
        if isa(var, VectorField) || isa(var, TensorField)
            for comp in var.components
                isa(comp, Operand) && push!(result, comp)
            end
        end
    end
    return result
end

"""
    _references_problem_variable(expr, variables) -> Bool

True when `expr` mentions one of the solved-for variables. Mirrors
`_references_variable` in the matrix builder. An empty/absent variable list means
"cannot tell", and the answer is `false` so that no factor is wrongly demoted from
coefficient to variable-dependent.
"""
function _references_problem_variable(expr, variables)
    (variables === nothing || isempty(variables)) && return false
    try
        vars = eltype(variables) <: Operand ? variables : Operand[v for v in variables if isa(v, Operand)]
        isempty(vars) && return false
        return !isempty(_detect_equation_variables(expr, vars))
    catch e
        @debug "Variable-reference check failed" exception=e
        return false
    end
end

"""Format a list of addends as a readable sum string, handling signs cleanly."""
function _format_sum(terms)
    isempty(terms) && return "0"
    parts = String[]
    for (i, t) in enumerate(terms)
        s = if isa(t, NegateOperator)
            "- " * expression_to_string(t.operand)
        else
            (i == 1 ? "" : "+ ") * expression_to_string(t)
        end
        push!(parts, s)
    end
    return join(parts, " ")
end

"""Check if expression tree contains any time derivatives."""
function contains_time_derivatives(expr)
    if expr === nothing
        return false
    elseif isa(expr, TimeDerivative)
        return true
    elseif isa(expr, AddOperator) || isa(expr, SubtractOperator) || isa(expr, MultiplyOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, DivideOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
    elseif isa(expr, PowerOperator)
        return contains_time_derivatives(expr.left) || contains_time_derivatives(expr.right)
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
function is_proper_lhs_structure(expr)

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

"""
    Recursively analyze expression structure for LHS validity.
    """
function _analyze_lhs_structure!(expr, info::Dict{Symbol, Any}, namespace::Dict{String, Any})

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
        exp_val = _get_constant_value(expr.right)
        if exp_val !== nothing && exp_val == 1
            return _analyze_lhs_structure!(expr.left, info, namespace)
        elseif _is_constant_coefficient_strict(expr.left, namespace)
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

"""
    Analyze the operand of a differential operator.
    """
function _analyze_lhs_operand!(operand, info::Dict{Symbol, Any}, namespace::Dict{String, Any})
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

"""
    Strictly check if expression is a constant coefficient.
    Constants are: numbers, ConstantOperator, or namespace entries that are constant.
    """
function _is_constant_coefficient_strict(expr, namespace::Dict{String, Any})
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
        return _is_constant_coefficient_strict(expr.left, namespace) &&
               _is_constant_coefficient_strict(expr.right, namespace)
    end

    # Fields are not constants
    if isa(expr, ScalarField) || isa(expr, VectorField)
        return false
    end

    return false
end

"""
    Extract numeric value from a constant expression.
    Returns nothing if not a simple constant.
    """
function _get_constant_value(expr)
    if isa(expr, Number)
        return expr
    end

    if isa(expr, ConstantOperator)
        return expr.value
    end

    return nothing
end

"""
    Validate LHS structure and return a detailed report.
    Throws an error if the structure is invalid.
    """
function validate_lhs_structure(expr)
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

"""
    Parse expression string into operator tree following patterns.
    Uses Julia's Meta.parse for proper AST parsing and operator precedence handling.
    
    This function evaluates mathematical expressions similar to how one would use
    eval(string, namespace) in problems:73-74.
    """
function parse_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    
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
            @warn "Error evaluating expression '$expr_str': $e" maxlog=1
            return fallback_parse_expression(expr_str, namespace)
        end
    end
end

"""
    Recursively evaluate parsed expression with namespace substitution.
    Similar to eval(string, namespace) but with proper Julia AST handling.
    """
function evaluate_parsed_expression(expr, namespace::Dict{String, Any})
    
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

            # ── Advection operator: u⋅∇(f) or dot(u, grad(f)) ──
            # style: u⋅∇(f) expands to Σᵢ uᵢ ∂ᵢf, handling
            # both scalar f (→ scalar result) and vector f (→ vector result).
            if func_expr === :⋅ || func_expr === :dot
                left = evaluate_parsed_expression(arg_exprs[1], namespace)
                right_expr = length(arg_exprs) >= 2 ? arg_exprs[2] : nothing
                # Detect ∇(f) or grad(f) on the right
                if right_expr isa Expr && right_expr.head == :call &&
                   right_expr.args[1] in (:∇, :grad, :gradient)
                    f = evaluate_parsed_expression(right_expr.args[2], namespace)
                    return _expand_advection(left, f)
                else
                    # Regular dot product
                    right = evaluate_parsed_expression(right_expr, namespace)
                    return MultiplyOperator(left, right)
                end
            end

            # Check for BC syntax: field(coord=value) → Interpolate
            # This handles expressions like u(y=0), T(z=1.0), dy(u)(y=0), etc.
            # In Julia's AST, keyword arguments appear as Expr(:kw, coord, value)
            # Following spectral methods pattern)
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
                elseif startswith(string(func_expr), "∂")
                    if isempty(arg_exprs)
                        throw(ArgumentError("$(func_expr) requires an operand"))
                    end
                    coord_name = replace(string(func_expr), r"^∂" => "")
                    field = evaluate_parsed_expression(arg_exprs[1], namespace)
                    coord = field isa Operand ? _find_coordinate_for_field(field, coord_name, namespace) : nothing
                    if coord === nothing && haskey(namespace, coord_name) && isa(namespace[coord_name], Coordinate)
                        coord = namespace[coord_name]
                    end
                    coord === nothing && throw(ArgumentError("Unknown derivative coordinate '$coord_name' in $(func_expr)"))
                    return d(field, coord, 1)
                elseif func_expr == :component || func_expr == :comp
                    if length(arg_exprs) < 2
                        throw(ArgumentError("component requires an operand and component index"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    index_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                    return component(operand, Int(round(index_val)))
                elseif func_expr == :d || func_expr == :diff
                    if length(arg_exprs) < 2
                        throw(ArgumentError("d requires at least an operand and a coordinate"))
                    end
                    field = evaluate_parsed_expression(arg_exprs[1], namespace)
                    coord_expr = arg_exprs[2]
                    coord = evaluate_parsed_expression(coord_expr, namespace)
                    if !(coord isa Coordinate) && coord_expr isa Symbol && field isa Operand
                        coord_name = string(coord_expr)
                        resolved_coord = _find_coordinate_for_field(field, coord_name, namespace)
                        resolved_coord !== nothing && (coord = resolved_coord)
                    end
                    order = if length(arg_exprs) >= 3
                        val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                        Int(round(val))
                    else
                        1
                    end
                    return d(field, coord, order)
                elseif func_expr == :lift
                    if length(arg_exprs) == 2
                        # Short form: lift(tau, -1) — auto-detect basis
                        operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                        n_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[2], namespace))
                        n_int = Int(round(n_val))
                        try
                            return lift(operand, n_int)
                        catch
                            # Auto-detection failed; return operand directly.
                            # For matrix sizing the Lift is just a shape-preserving
                            # wrapper, so the operand alone is sufficient.
                            return operand
                        end
                    elseif length(arg_exprs) >= 3
                        # Full form: lift(tau, basis, -1)
                        operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                        basis = evaluate_parsed_expression(arg_exprs[2], namespace)
                        n_val = coerce_constant_value(evaluate_parsed_expression(arg_exprs[3], namespace))
                        return lift(operand, basis, Int(round(n_val)))
                    else
                        throw(ArgumentError("lift requires (operand, n) or (operand, basis, n)"))
                    end
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
                elseif func_expr in (:integ, :integrate)
                    # Deferred integration — must NOT eagerly evaluate to a number.
                    # integ(f) → integrate over all coords; integ(f, coord) → one coord.
                    if isempty(arg_exprs)
                        throw(ArgumentError("integ requires at least one argument"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    if length(arg_exprs) >= 2
                        coord = evaluate_parsed_expression(arg_exprs[2], namespace)
                        return integrate(operand, coord)
                    elseif isa(operand, Operand) && hasfield(typeof(operand), :dist) &&
                           operand.dist !== nothing && hasfield(typeof(operand.dist), :coordsys)
                        # Construct Integrate node directly to avoid dispatch
                        # ambiguity with integrate(::ScalarField, axes=:)
                        all_coords = tuple(coords(operand.dist.coordsys)...)
                        return Integrate(operand, all_coords)
                    else
                        return Integrate(operand, ())
                    end
                elseif func_expr in (:avg, :average)
                    # Deferred average — same pattern as integ.
                    if length(arg_exprs) < 2
                        throw(ArgumentError("average requires (operand, coord)"))
                    end
                    operand = evaluate_parsed_expression(arg_exprs[1], namespace)
                    coord = evaluate_parsed_expression(arg_exprs[2], namespace)
                    return average(operand, coord)
                end
            end
            
            func = evaluate_parsed_expression(func_expr, namespace)
            evaluated_args = [coerce_constant_value(evaluate_parsed_expression(arg, namespace)) for arg in arg_exprs]

            if isa(func, Function)
                # Short-circuit ONLY when an argument contains an
                # `UnknownOperator` placeholder (typically a coordinate or
                # time variable like `x` / `t` in a space- or time-dependent
                # BC RHS that can't be bound at parse time). In that case
                # the function call would fail at the Julia level (e.g.
                # `sin(::UnknownOperator)` → MethodError). The spatial /
                # temporal BC evaluator in `_apply_bc_values_to_equations!`
                # re-evaluates the BC's original string at runtime against
                # actual coordinate grid arrays / current time, and
                # overwrites `equation_data[eq_idx]["F"]` with the result —
                # so the parser's symbolic placeholder is never consulted.
                #
                # For normal field-operator expressions like `trace(grad_u)`
                # where `grad_u` is a stored substitution (an `AddOperator`
                # / `Future` tree of concrete fields), no `UnknownOperator`
                # is present and the function call proceeds as normal,
                # letting operators like `trace` algebraically simplify to
                # `divergence(u) + lift(...)`.
                if any(_expression_contains_unknown, evaluated_args)
                    return UnknownOperator(string(expr))
                end
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

            return IndexOperator(array, Tuple(evaluated_indices))
            
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

"""
    Fallback parsing for expressions that fail Meta.parse evaluation.
    Uses simple string pattern matching as backup.
    """
function fallback_parse_expression(expr_str::AbstractString, namespace::Dict{String, Any})
    
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

"""
    _expand_advection(u, f) → expression tree for u⋅∇(f) = Σᵢ uᵢ ∂ᵢf

Expands the advection operator into scalar derivative terms.
Works for both scalar f (returns scalar) and vector f (returns vector via
component-wise differentiation).
"""
function _expand_advection(u, f)
    # Handle -u⋅∇(f) → -(u⋅∇(f)):  Julia parses -u⋅∇(f) as (-u)⋅∇(f)
    if isa(u, NegateOperator) && isa(u.operand, VectorField)
        return NegateOperator(_expand_advection(u.operand, f))
    end
    # Handle c*u⋅∇(f) → c*(u⋅∇(f))
    if isa(u, MultiplyOperator)
        if isa(u.left, VectorField)
            return MultiplyOperator(u.right, _expand_advection(u.left, f))
        elseif isa(u.right, VectorField)
            return MultiplyOperator(u.left, _expand_advection(u.right, f))
        end
    end
    if !isa(u, VectorField)
        throw(ArgumentError("Advection u⋅∇(f) requires u to be a VectorField, got $(typeof(u))"))
    end
    coordsys = u.coordsys
    coord_list = coords(coordsys)
    ndim = min(length(coord_list), length(u.components))

    result = nothing
    for i in 1:ndim
        ui = u.components[i]
        ci = coord_list[i]
        # uᵢ * ∂ᵢf — Differentiate handles both ScalarField and VectorField
        term = MultiplyOperator(ui, Differentiate(f, ci, 1))
        result = result === nothing ? term : AddOperator(result, term)
    end
    return result
end

"""
    _expression_contains_unknown(expr) -> Bool

Return `true` if an expression tree (either an `Operator` subtype or a
`Future` node) contains an `UnknownOperator` placeholder anywhere in its
subtree. Used by the equation parser's function-call path to decide
whether to short-circuit `func(args...)` into a symbolic
`UnknownOperator(string(expr))` — which it does when an argument carries
an unbound coordinate / time placeholder — versus proceeding with the
call, which is correct for normal operator expressions built from
concrete fields.

The walk handles:
- `UnknownOperator` directly
- Operator subtypes via `operand` / `left` / `right` fields
- `Future` subtypes via `future_args`
- `Number` and literal values (short-circuit false)
"""
function _expression_contains_unknown(expr)
    isa(expr, UnknownOperator) && return true
    (isa(expr, Number) || isa(expr, Symbol) || isa(expr, AbstractString)) && return false
    # Operator subtypes commonly store children in :operand / :left / :right.
    for f in (:operand, :left, :right)
        if hasfield(typeof(expr), f)
            child = getfield(expr, f)
            _expression_contains_unknown(child) && return true
        end
    end
    # Future subtypes store children via `future_args`.
    if isa(expr, Future)
        for arg in future_args(expr)
            _expression_contains_unknown(arg) && return true
        end
    end
    return false
end

# Helper operator types for parsing (following operator structure)
struct ZeroOperator <: Operator end
struct ConstantOperator <: Operator
    value::Float64
end
struct ArrayOperator <: Operator
    value::AbstractArray
end
struct UnknownOperator <: Operator
    expression::String
end

# has() definitions for ZeroOperator, ConstantOperator, ArrayOperator, UnknownOperator
# These types never contain problem variables (following spectral pattern)
has(::ZeroOperator, vars...) = false
has(::ConstantOperator, vars...) = false
has(::ArrayOperator, vars...) = false
has(::UnknownOperator, vars...) = false  # Conservative: unknowns don't contain tracked vars

# Symbolic derivatives of constant operators are zero (they hold no variables).
# Needed by frechet_differential when an NLBVP RHS has constant/parameter terms
# (e.g. ∂(u²+g)/∂tau = 0) or is differentiated w.r.t. a non-appearing variable.
sym_diff(::ZeroOperator, ::ScalarField) = 0
sym_diff(::ConstantOperator, ::ScalarField) = 0
sym_diff(::ArrayOperator, ::ScalarField) = 0

coerce_constant_value(x::Number) = x
coerce_constant_value(x::ConstantOperator) = x.value
# Constant-fold arithmetic over numbers / constants so an operator argument written as an
# expression resolves to a NUMBER instead of degrading to UnknownOperator downstream. This
# fixes e.g. a BC position `u(z=Lz/2)` (was parsed as UnknownOperator → BC silently dropped)
# and a fractional exponent `fraclap(u, 1/2)` (was a DivideOperator → Float64() threw → term
# dropped). Only folds when every leaf reduces to a Number; otherwise returns x unchanged.
function coerce_constant_value(x)
    if x isa DivideOperator
        l = coerce_constant_value(x.left); r = coerce_constant_value(x.right)
        return (l isa Number && r isa Number) ? l / r : x
    elseif x isa MultiplyOperator
        l = coerce_constant_value(x.left); r = coerce_constant_value(x.right)
        return (l isa Number && r isa Number) ? l * r : x
    elseif x isa AddOperator
        l = coerce_constant_value(x.left); r = coerce_constant_value(x.right)
        return (l isa Number && r isa Number) ? l + r : x
    elseif x isa SubtractOperator
        l = coerce_constant_value(x.left); r = coerce_constant_value(x.right)
        return (l isa Number && r isa Number) ? l - r : x
    elseif x isa PowerOperator
        l = coerce_constant_value(x.left); r = coerce_constant_value(x.right)
        return (l isa Number && r isa Number) ? l ^ r : x
    elseif x isa NegateOperator
        v = coerce_constant_value(x.operand)
        return v isa Number ? -v : x
    end
    return x
end

# Note: AddOperator, SubtractOperator, MultiplyOperator, DivideOperator, PowerOperator,
# NegateOperator, IndexOperator are defined in operators.jl

"""
    _find_coordinate_for_field(field, coord_name, namespace)

Find a Coordinate object for the given coordinate name that is associated with the field.
Used for automatic BC syntax parsing: field(coord=value) → Interpolate(field, coord, value)

Following spectral methods pattern):
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

    # Operators such as d(T, z) wrap the original field. Boundary interpolation
    # on those operators still needs the wrapped field's coordinate metadata.
    if hasfield(typeof(field), :operand)
        operand = getfield(field, :operand)
        if isa(operand, Operand)
            coord = _find_coordinate_for_field(operand, coord_name, namespace)
            coord !== nothing && return coord
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
