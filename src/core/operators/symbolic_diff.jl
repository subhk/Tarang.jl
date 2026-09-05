"""
    Symbolic Differentiation for Operator Expressions

Provides pointwise symbolic partial derivatives through `sym_diff(expr, var)`
and perturbation-aware directional derivatives through
`frechet_differential(expr, vars, perts)`. The latter is used by the NLBVP
Jacobian path because differential operators produce linear maps rather than
scalar derivative coefficients.

Key components:
- UFUNC_DERIVATIVES: lookup table for derivatives of standard math functions
- sym_diff(): pointwise symbolic differentiation following chain/product/sum rules
- simplify(): basic algebraic simplification (0+x→x, 1*x→x, 0*x→0)
- frechet_differential(): linearization dF(X0).dX = Σ (∂F/∂uⱼ) * δuⱼ
- build_symbolic_jacobian(): assemble Jacobian matrix for NLBVP Newton iteration
"""

# ============================================================================
# Derivative Lookup Table for Unary Grid Functions
# ============================================================================

"""
Mapping from standard Julia math functions to their derivatives.
Each value is a function that takes the inner argument and returns the
derivative of the outer function evaluated at that argument.
"""
const UFUNC_DERIVATIVES = Dict{Function, Function}(
    sin   => cos,
    cos   => x -> -sin(x),
    tan   => x -> 1 / cos(x)^2,
    exp   => exp,
    log   => x -> 1 / x,
    sqrt  => x -> 1 / (2 * sqrt(x)),
    abs   => sign,
    tanh  => x -> 1 - tanh(x)^2,
    sinh  => cosh,
    cosh  => sinh,
    asin  => x -> 1 / sqrt(1 - x^2),
    acos  => x -> -1 / sqrt(1 - x^2),
    atan  => x -> 1 / (1 + x^2),
)

# ============================================================================
# Symbolic Differentiation: sym_diff(expr, var)
# ============================================================================

"""
    sym_diff(expr, var)

Compute the symbolic derivative of `expr` with respect to field `var`.
Returns an operator expression tree representing the derivative.

Rules:
- ScalarField: d(var)/d(var) = 1, d(other)/d(var) = 0
- Number: d(const)/d(var) = 0
- Add/Subtract: linearity
- Multiply: product rule
- Divide: quotient rule
- Negate: d(-f)/d(var) = -d(f)/d(var)
- Power: d(f^n)/d(var) = n*f^(n-1)*d(f)/d(var)
- Differential operators: their derivatives are linear maps and require an
  explicit perturbation; use `frechet_differential` for those expressions
- UnaryGridFunction: chain rule using UFUNC_DERIVATIVES
"""
function sym_diff end

# Base cases
function sym_diff(f::ScalarField, var::ScalarField)
    return (f === var) ? 1 : 0
end

sym_diff(::Number, ::ScalarField) = 0

# Arithmetic operators
function sym_diff(op::AddOperator, var::ScalarField)
    dl = sym_diff(op.left, var)
    dr = sym_diff(op.right, var)
    return _simplify_add(dl, dr)
end

function sym_diff(op::SubtractOperator, var::ScalarField)
    dl = sym_diff(op.left, var)
    dr = sym_diff(op.right, var)
    return _simplify_sub(dl, dr)
end

function sym_diff(op::MultiplyOperator, var::ScalarField)
    # Product rule: d(f*g) = f*dg + df*g
    f = op.left
    g = op.right
    df = sym_diff(f, var)
    dg = sym_diff(g, var)
    term1 = _simplify_mul(f, dg)
    term2 = _simplify_mul(df, g)
    return _simplify_add(term1, term2)
end

function sym_diff(op::DivideOperator, var::ScalarField)
    # Quotient rule: d(f/g) = (df*g - f*dg) / g^2
    f = op.left
    g = op.right
    df = sym_diff(f, var)
    dg = sym_diff(g, var)
    if dg == 0 || dg === 0
        # g is constant w.r.t. var: d(f/g) = df/g
        return _simplify_div(df, g)
    end
    num = _simplify_sub(_simplify_mul(df, g), _simplify_mul(f, dg))
    den = _simplify_mul(g, g)
    return _simplify_div(num, den)
end

function sym_diff(op::NegateOperator, var::ScalarField)
    d = sym_diff(op.operand, var)
    return _simplify_neg(d)
end

function sym_diff(op::PowerOperator, var::ScalarField)
    # d(f^n)/d(var) = n * f^(n-1) * df/d(var)
    f = op.left
    n = op.right
    df = sym_diff(f, var)
    dn = sym_diff(n, var)
    if dn == 0 || dn === 0
        # Exponent is constant: power rule
        return _simplify_mul(_simplify_mul(n, PowerOperator(f, _simplify_sub(n, 1))), df)
    else
        # General case: d(f^g) = f^g * (g'*log(f) + g*f'/f)
        # Not commonly needed for PDEs, provide basic support
        error("sym_diff for variable exponents not yet supported")
    end
end

# A two-argument scalar partial cannot represent an operator-valued derivative.
# For example, d[Δ(u)]/du is the map δu -> Δ(δu), not the scalar zero and not
# Δ(1). Returning either silently loses where the perturbation belongs. Keep the
# independent-operand zero case, but require the directional Frechet API whenever
# this operator actually depends on `var`.
function _operator_valued_symdiff(op::Operator, var::ScalarField)
    has(op.operand, var) || return 0
    throw(ArgumentError(
        "sym_diff cannot represent the operator-valued derivative of " *
        "$(nameof(typeof(op))) with two arguments. Use " *
        "frechet_differential(expr, [var], [perturbation]) instead."))
end

sym_diff(op::Differentiate, var::ScalarField) = _operator_valued_symdiff(op, var)
sym_diff(op::Laplacian, var::ScalarField) = _operator_valued_symdiff(op, var)
sym_diff(op::FractionalLaplacian, var::ScalarField) = _operator_valued_symdiff(op, var)

# Chain rule for UnaryGridFunction
function sym_diff(op::UnaryGridFunction, var::ScalarField)
    f_prime = get(UFUNC_DERIVATIVES, op.func, nothing)
    if f_prime === nothing
        error("No symbolic derivative registered for function '$(op.name)'. " *
              "Add it to UFUNC_DERIVATIVES.")
    end
    du = sym_diff(op.operand, var)
    if du == 0 || du === 0
        return 0
    end
    # Chain rule: d(f(u))/d(var) = f'(u) * du/d(var)
    outer_deriv = UnaryGridFunction(op.operand, f_prime, "d_$(op.name)")
    return _simplify_mul(outer_deriv, du)
end

# GeneralFunction: same chain rule logic
function sym_diff(op::GeneralFunction, var::ScalarField)
    f_prime = get(UFUNC_DERIVATIVES, op.func, nothing)
    if f_prime === nothing
        error("No symbolic derivative registered for function '$(op.name)'. " *
              "Add it to UFUNC_DERIVATIVES.")
    end
    du = sym_diff(op.operand, var)
    if du == 0 || du === 0
        return 0
    end
    outer_deriv = GeneralFunction(op.operand, f_prime, "d_$(op.name)")
    return _simplify_mul(outer_deriv, du)
end

# Copy: derivative passes through
function sym_diff(op::Copy, var::ScalarField)
    return sym_diff(op.operand, var)
end

# Gradient and divergence are operator-valued for the same reason.
sym_diff(op::Gradient, var::ScalarField) = _operator_valued_symdiff(op, var)
sym_diff(op::Divergence, var::ScalarField) = _operator_valued_symdiff(op, var)

# TimeDerivative: for NLBVP (steady-state), ∂t(u) = 0 so ∂(∂t(u))/∂u = 0
function sym_diff(op::TimeDerivative, var::ScalarField)
    return 0
end

# Fallback for unknown operator types
function sym_diff(op::Operator, var::ScalarField)
    if hasfield(typeof(op), :operand)
        d = sym_diff(op.operand, var)
        if d == 0 || d === 0
            return 0
        end
    end
    error("sym_diff not implemented for operator type $(typeof(op))")
end

# ============================================================================
# Simplification Helpers
# ============================================================================

"""Simplify addition: 0+x → x, x+0 → x."""
function _simplify_add(a, b)
    if a === 0 || a == 0
        return b
    elseif b === 0 || b == 0
        return a
    elseif isa(a, Number) && isa(b, Number)
        return a + b
    else
        return AddOperator(a, b)
    end
end

"""Simplify subtraction: x-0 → x, 0-x → -x."""
function _simplify_sub(a, b)
    if b === 0 || b == 0
        return a
    elseif a === 0 || a == 0
        return _simplify_neg(b)
    elseif isa(a, Number) && isa(b, Number)
        return a - b
    else
        return SubtractOperator(a, b)
    end
end

"""Simplify multiplication: 0*x → 0, 1*x → x, x*1 → x, x*0 → 0."""
function _simplify_mul(a, b)
    if a === 0 || a == 0 || b === 0 || b == 0
        return 0
    elseif a === 1 || a == 1
        return b
    elseif b === 1 || b == 1
        return a
    elseif isa(a, Number) && isa(b, Number)
        return a * b
    else
        return MultiplyOperator(a, b)
    end
end

"""Simplify division: x/1 → x, 0/x → 0."""
function _simplify_div(a, b)
    if a === 0 || a == 0
        return 0
    elseif b === 1 || b == 1
        return a
    elseif isa(a, Number) && isa(b, Number)
        return a / b
    else
        return DivideOperator(a, b)
    end
end

"""Simplify negation: -0 → 0, -(-x) → x, -(number) → -number."""
function _simplify_neg(a)
    if a === 0 || a == 0
        return 0
    elseif isa(a, Number)
        return -a
    elseif isa(a, NegateOperator)
        return a.operand
    else
        return NegateOperator(a)
    end
end

"""
    simplify(expr)

Apply basic algebraic simplifications to an operator expression tree.
Collapses 0+x→x, 1*x→x, 0*x→0 recursively.
"""
function simplify(expr)
    if isa(expr, Number) || isa(expr, ScalarField)
        return expr
    elseif isa(expr, AddOperator)
        l = simplify(expr.left)
        r = simplify(expr.right)
        return _simplify_add(l, r)
    elseif isa(expr, SubtractOperator)
        l = simplify(expr.left)
        r = simplify(expr.right)
        return _simplify_sub(l, r)
    elseif isa(expr, MultiplyOperator)
        l = simplify(expr.left)
        r = simplify(expr.right)
        return _simplify_mul(l, r)
    elseif isa(expr, DivideOperator)
        l = simplify(expr.left)
        r = simplify(expr.right)
        return _simplify_div(l, r)
    elseif isa(expr, NegateOperator)
        inner = simplify(expr.operand)
        return _simplify_neg(inner)
    else
        return expr
    end
end

# ============================================================================
# Frechet Differentiation for NLBVP
# ============================================================================

@inline _direction_iszero(x) = x === 0 || (x isa Number && iszero(x))

"""
    _directional_diff(expr, var, pert)

Construct the directional derivative `d(expr)[var]·pert`. Unlike `sym_diff`,
this keeps the perturbation inside differential operators, which is required for
field-valued functional derivatives such as `d[Δ(u)]·δu = Δ(δu)`.
"""
_directional_diff(f::ScalarField, var::ScalarField, pert::ScalarField) =
    f === var ? pert : 0
_directional_diff(::Number, ::ScalarField, ::ScalarField) = 0

function _directional_diff(op::AddOperator, var::ScalarField, pert::ScalarField)
    return _simplify_add(
        _directional_diff(op.left, var, pert),
        _directional_diff(op.right, var, pert))
end

function _directional_diff(op::SubtractOperator, var::ScalarField, pert::ScalarField)
    return _simplify_sub(
        _directional_diff(op.left, var, pert),
        _directional_diff(op.right, var, pert))
end

function _directional_diff(op::MultiplyOperator, var::ScalarField, pert::ScalarField)
    dl = _directional_diff(op.left, var, pert)
    dr = _directional_diff(op.right, var, pert)
    return _simplify_add(_simplify_mul(dl, op.right), _simplify_mul(op.left, dr))
end

function _directional_diff(op::DivideOperator, var::ScalarField, pert::ScalarField)
    dl = _directional_diff(op.left, var, pert)
    dr = _directional_diff(op.right, var, pert)
    _direction_iszero(dr) && return _simplify_div(dl, op.right)
    numerator = _simplify_sub(
        _simplify_mul(dl, op.right), _simplify_mul(op.left, dr))
    return _simplify_div(numerator, _simplify_mul(op.right, op.right))
end

function _directional_diff(op::NegateOperator, var::ScalarField, pert::ScalarField)
    return _simplify_neg(_directional_diff(op.operand, var, pert))
end

function _directional_diff(op::PowerOperator, var::ScalarField, pert::ScalarField)
    df = _directional_diff(op.left, var, pert)
    dn = _directional_diff(op.right, var, pert)
    _direction_iszero(dn) || error(
        "Directional differentiation of variable exponents is not supported")
    return _simplify_mul(
        _simplify_mul(op.right, PowerOperator(op.left, _simplify_sub(op.right, 1))), df)
end

function _directional_diff(op::UnaryGridFunction, var::ScalarField, pert::ScalarField)
    du = _directional_diff(op.operand, var, pert)
    _direction_iszero(du) && return 0
    f_prime = get(UFUNC_DERIVATIVES, op.func, nothing)
    f_prime === nothing && error(
        "No symbolic derivative registered for function '$(op.name)'. " *
        "Add it to UFUNC_DERIVATIVES.")
    outer = UnaryGridFunction(op.operand, f_prime, "d_$(op.name)")
    return _simplify_mul(outer, du)
end

function _directional_diff(op::GeneralFunction, var::ScalarField, pert::ScalarField)
    du = _directional_diff(op.operand, var, pert)
    _direction_iszero(du) && return 0
    f_prime = get(UFUNC_DERIVATIVES, op.func, nothing)
    f_prime === nothing && error(
        "No symbolic derivative registered for function '$(op.name)'. " *
        "Add it to UFUNC_DERIVATIVES.")
    outer = GeneralFunction(op.operand, f_prime, "d_$(op.name)")
    return _simplify_mul(outer, du)
end

_directional_diff(op::Copy, var::ScalarField, pert::ScalarField) =
    _directional_diff(op.operand, var, pert)

function _directional_diff(op::Differentiate, var::ScalarField, pert::ScalarField)
    inner = _directional_diff(op.operand, var, pert)
    return _direction_iszero(inner) ? 0 : Differentiate(inner, op.coord, op.order)
end

function _directional_diff(op::Laplacian, var::ScalarField, pert::ScalarField)
    inner = _directional_diff(op.operand, var, pert)
    return _direction_iszero(inner) ? 0 : Laplacian(inner)
end

function _directional_diff(op::FractionalLaplacian, var::ScalarField, pert::ScalarField)
    inner = _directional_diff(op.operand, var, pert)
    return _direction_iszero(inner) ? 0 : FractionalLaplacian(inner, op.α)
end

function _directional_diff(op::Gradient, var::ScalarField, pert::ScalarField)
    inner = _directional_diff(op.operand, var, pert)
    return _direction_iszero(inner) ? 0 : Gradient(inner, op.coordsys)
end

function _directional_diff(op::Divergence, var::ScalarField, pert::ScalarField)
    inner = _directional_diff(op.operand, var, pert)
    return _direction_iszero(inner) ? 0 : Divergence(inner)
end

_directional_diff(::TimeDerivative, ::ScalarField, ::ScalarField) = 0

function _directional_diff(op::Operator, var::ScalarField, pert::ScalarField)
    has(op, var) || return 0
    error("Directional differentiation is not implemented for operator type $(typeof(op))")
end

"""
    frechet_differential(F, vars, perts)

Compute the Frechet differential (linearization) of expression F:
    dF(X₀)·δX = Σⱼ (∂F/∂uⱼ) * δuⱼ

Arguments:
- F: operator expression (residual of an equation)
- vars: vector of ScalarField variables [u₁, u₂, ...]
- perts: vector of perturbation fields [δu₁, δu₂, ...]

Returns an operator expression representing the linearized operator.
"""
function frechet_differential(F, vars::Vector, perts::Vector)
    length(vars) == length(perts) || error("vars and perts must have same length")

    terms = Any[]
    for (var, pert) in zip(vars, perts)
        direction = _directional_diff(F, var, pert)
        _direction_iszero(direction) && continue
        push!(terms, direction)
    end
    isempty(terms) && return 0
    return foldl(_simplify_add, terms)
end

"""
    build_symbolic_jacobian(problem, state_fields)

Build the Jacobian matrix for an NLBVP by constructing the directional
derivative of each equation residual and applying it to both quadratures of
each coefficient-space basis vector at the current state.

Returns a doubled-real sparse matrix acting on `[real(x); imag(x)]`. This is
required for `RealFourier` half-spectra: nonlinear physical-space operations are
real-linear in their packed complex coefficients, but are not complex-linear.

Applying the actual directional expression preserves differential operators and
the convolution induced by field-valued pointwise multipliers. This CPU fallback
favours correctness over the diagonal approximation previously used here.
"""
function build_symbolic_jacobian(problem::Problem, state_fields)
    any(_field_uses_gpu, state_fields) && error(
        "GPU symbolic Jacobian assembly is unsupported; CPU fallback is disabled.")
    vars = state_fields

    # Get equation data
    eq_data_list = _get_equation_data(problem)
    if isempty(eq_data_list)
        error("No equation data available for symbolic Jacobian construction")
    end

    # Build block Jacobian
    # Each equation i and variable j gives a block
    n_eqs = length(eq_data_list)
    n_vars = length(vars)

    if n_eqs != n_vars
        error("Symbolic Jacobian requires n_eqs ($n_eqs) == n_vars ($n_vars). " *
              "Non-square systems are not supported.")
    end

    # Determine block sizes (one per variable/equation pair)
    block_sizes = Int[]
    for field in state_fields
        push!(block_sizes, length(coeff_data!(field)))
    end

    total_size = sum(block_sizes)
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    row_offset = 0
    for (i, eq_data) in enumerate(eq_data_list)
        # Get the residual expression F_i = LHS_i - RHS_i
        F_i = _get_residual_expression(eq_data)

        col_offset = 0
        for (j, var) in enumerate(vars)
            block = _directional_jacobian_block(
                F_i, var, state_fields[i], block_sizes[i], block_sizes[j])

            # A block is locally ordered as [real rows; imag rows] ×
            # [real cols; imag cols]. Map it into the global ordering
            # [real(all fields); imag(all fields)].
            for (bi, bj, bv) in _sparse_entries(block)
                global_i = bi <= block_sizes[i] ? row_offset + bi :
                           total_size + row_offset + bi - block_sizes[i]
                global_j = bj <= block_sizes[j] ? col_offset + bj :
                           total_size + col_offset + bj - block_sizes[j]
                push!(I_idx, global_i)
                push!(J_idx, global_j)
                push!(V_val, bv)
            end

            col_offset += block_sizes[j]
        end
        row_offset += block_sizes[i]
    end

    for field in state_fields
        coeff_data!(field)
    end
    return sparse(I_idx, J_idx, V_val, 2total_size, 2total_size)
end

"""
    _get_equation_data(problem::Problem) -> Vector{EquationIR}

The problem's equation IR.

This used to be a `hasfield` chain that fell back to `problem.equations` when
`:equation_data` was missing. All four `Problem` subtypes declare
`equation_data`, so the fallback was unreachable from the package — and it would
have been wrong if reached, because `equations` is a `Vector{String}` while
every caller treats the result as equation IR. The only thing exercising it was
a duck-typed stand-in in the test file.
"""
_get_equation_data(problem::Problem) = problem.equation_data

"""Extract residual expression (LHS - RHS) from equation data."""
function _get_residual_expression(eq_data)
    if isa(eq_data, AbstractDict)
        # The parser writes these under LOWERCASE "lhs"/"rhs"
        # (problem_matrices_build.jl). This read "LHS"/"RHS", which nothing in the
        # package ever writes, so the subtraction branch was unreachable and every
        # residual fell through to `F` — a Newton Jacobian missing the entire
        # implicit `L` contribution, with no error to say so.
        lhs = get(eq_data, "lhs", nothing)
        rhs = get(eq_data, "rhs", nothing)
        if lhs !== nothing && rhs !== nothing
            return SubtractOperator(lhs, rhs)
        end
        # `haskey` is true for every canonical EquationIR slot whether or not it
        # holds anything, so test the value rather than the key.
        forcing = get(eq_data, "F", nothing)
        forcing === nothing || return forcing
    end
    return 0
end

"""
    _directional_jacobian_block(residual, var, template, nrows, ncols)

Assemble one Jacobian block by applying the exact directional expression to the
real and imaginary quadratures of every coefficient basis vector. Rebuilding
the expression for every probe keeps its perturbation field live inside nested
differential operators.
"""
function _directional_jacobian_block(residual, var::ScalarField,
                                     template::ScalarField, nrows::Int, ncols::Int)
    pert = ScalarField(var.dist, "δ_$(var.name)", var.bases, var.dtype)
    basis_vector = zeros(ComplexF64, ncols)
    real_actions = zeros(ComplexF64, nrows, ncols)
    imag_actions = zeros(ComplexF64, nrows, ncols)

    for col in 1:ncols
        for (probe, actions) in ((1.0 + 0.0im, real_actions),
                                 (0.0 + 1.0im, imag_actions))
            fill!(basis_vector, 0)
            basis_vector[col] = probe
            copy_solution_to_fields!([pert], basis_vector)

            direction = _directional_diff(residual, var, pert)
            _direction_iszero(direction) && continue
            result = direction isa ScalarField ? direction :
                     evaluate_solver_expression(
                         direction, [pert]; layout=:g, template=template)
            result isa ScalarField || error(
                "Directional Jacobian action for $(typeof(residual)) returned " *
                "$(typeof(result)); expected ScalarField")
            values = fields_to_vector([result])
            length(values) == nrows || throw(DimensionMismatch(
                "Directional Jacobian action has $(length(values)) coefficients, expected $nrows"))
            @views actions[:, col] .= values
        end
    end

    return _doubled_real_matrix(real_actions, imag_actions)
end

"""Build the real matrix mapping `[Re(x); Im(x)]` to `[Re(y); Im(y)]`."""
function _doubled_real_matrix(real_actions::AbstractMatrix,
                              imag_actions::AbstractMatrix)
    size(real_actions) == size(imag_actions) || throw(DimensionMismatch(
        "Real and imaginary Jacobian probes must have matching sizes"))
    return sparse([real.(real_actions) real.(imag_actions);
                   imag.(real_actions) imag.(imag_actions)])
end

"""Doubled-real representation of a complex-linear matrix."""
function _complex_linear_to_doubled_real(matrix::AbstractMatrix)
    return sparse([real.(matrix) -imag.(matrix);
                   imag.(matrix)  real.(matrix)])
end

function _scalar_to_doubled_real(value::Number, nrows::Int, ncols::Int)
    n = min(nrows, ncols)
    identity_block = spdiagm(nrows, ncols, 0 => ones(Float64, n))
    a = Float64(real(value))
    b = Float64(imag(value))
    return [a .* identity_block  -b .* identity_block;
            b .* identity_block   a .* identity_block]
end

"""
Evaluate a Jacobian block (derivative expression) to a matrix.
For scalar-valued expressions, returns their exact coefficient-space
multiplication matrix.
For operator-valued expressions (Laplacian), returns the operator matrix.

GPU-valued Jacobian blocks are rejected until sparse Jacobian assembly is
device-native; field data is never downloaded implicitly.
"""
function _field_multiplication_matrix(coefficient::ScalarField, var::ScalarField,
                                      nrows::Int, ncols::Int)
    coefficient.bases == var.bases || throw(ArgumentError(
        "Jacobian coefficient field and perturbation field must have matching bases"))
    pert = ScalarField(var.dist, "δ_$(var.name)", var.bases, var.dtype)
    basis_vector = zeros(ComplexF64, ncols)
    real_actions = zeros(ComplexF64, nrows, ncols)
    imag_actions = zeros(ComplexF64, nrows, ncols)

    for col in 1:ncols
        for (probe, actions) in ((1.0 + 0.0im, real_actions),
                                 (0.0 + 1.0im, imag_actions))
            fill!(basis_vector, 0)
            basis_vector[col] = probe
            copy_solution_to_fields!([pert], basis_vector)
            product = coefficient * pert
            values = fields_to_vector([product])
            length(values) == nrows || throw(DimensionMismatch(
                "Coefficient product has $(length(values)) coefficients, expected $nrows"))
            @views actions[:, col] .= values
        end
    end
    return _doubled_real_matrix(real_actions, imag_actions)
end


function _evaluate_jacobian_block(expr, var::ScalarField, nrows::Int, ncols::Int)
    _field_uses_gpu(var) && error(
        "GPU symbolic Jacobian assembly is unsupported; CPU fallback is disabled.")
    if isa(expr, Number)
        return _scalar_to_doubled_real(expr, nrows, ncols)
    elseif isa(expr, ScalarField)
        data = grid_data!(expr)
        is_gpu_array(data) && error(
            "GPU symbolic Jacobian assembly is unsupported; CPU fallback is disabled.")
        return _field_multiplication_matrix(expr, var, nrows, ncols)
    elseif isa(expr, Operator)
        # Try to evaluate as a matrix
        try
            result = evaluate(expr, :g)
            if isa(result, ScalarField)
                data = grid_data!(result)
                is_gpu_array(data) && error(
                    "GPU symbolic Jacobian assembly is unsupported; CPU fallback is disabled.")
                return _field_multiplication_matrix(result, var, nrows, ncols)
            elseif isa(result, AbstractMatrix)
                is_gpu_array(result) && error(
                    "GPU symbolic Jacobian matrices are unsupported; CPU fallback is disabled.")
                return _complex_linear_to_doubled_real(result)
            elseif isa(result, Number)
                return _scalar_to_doubled_real(result, nrows, ncols)
            end
        catch e1
            # Fallback: try to get operator matrix from infrastructure
            try
                return _complex_linear_to_doubled_real(subproblem_matrix(expr))
            catch e2
                # Last resort: identity block — warn so Newton convergence issues are diagnosable
                @warn "Jacobian block for $(typeof(expr)) could not be evaluated; using identity fallback. " *
                      "Newton convergence may be degraded or incorrect." evaluate_error=e1 matrix_error=e2 maxlog=3
                return _scalar_to_doubled_real(1, nrows, ncols)
            end
        end
    end
    @warn "Unrecognized expression type $(typeof(expr)) in Jacobian block; using identity fallback." maxlog=3
    return _scalar_to_doubled_real(1, nrows, ncols)
end

"""Extract sparse entries (i, j, v) from a matrix."""
function _sparse_entries(M)
    if isa(M, SparseMatrixCSC)
        I_m, J_m, V_m = findnz(M)
        return zip(I_m, J_m, V_m)
    elseif isa(M, AbstractMatrix)
        entries = Tuple{Int, Int, Any}[]
        for j in 1:size(M, 2), i in 1:size(M, 1)
            v = M[i, j]
            if v != 0
                push!(entries, (i, j, v))
            end
        end
        return entries
    elseif isa(M, Number)
        return [(1, 1, M)]
    end
    return Tuple{Int, Int, Any}[]
end

# ============================================================================
# Exports
# ============================================================================

export UFUNC_DERIVATIVES, sym_diff, simplify
export frechet_differential, build_symbolic_jacobian
