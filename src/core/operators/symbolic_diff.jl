"""
    Symbolic Differentiation for Operator Expressions

Provides `sym_diff(expr, var)` for computing symbolic derivatives of operator
expressions with respect to field variables. Used by the NLBVP solver to
build analytical Jacobians via Frechet differentiation.

Key components:
- UFUNC_DERIVATIVES: lookup table for derivatives of standard math functions
- sym_diff(): recursive symbolic differentiation following chain/product/sum rules
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
- Differentiate: commutes with sym_diff (d/dx commutes with d/du)
- Laplacian: commutes with sym_diff
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

# Differential operators commute with sym_diff
function sym_diff(op::Differentiate, var::ScalarField)
    d_operand = sym_diff(op.operand, var)
    if d_operand == 0 || d_operand === 0
        return 0
    end
    if isa(d_operand, Number)
        return 0  # Derivative of constant is 0
    end
    return Differentiate(d_operand, op.coord, op.order)
end

function sym_diff(op::Laplacian, var::ScalarField)
    d_operand = sym_diff(op.operand, var)
    if d_operand == 0 || d_operand === 0
        return 0
    end
    if isa(d_operand, Number)
        return 0
    end
    return Laplacian(d_operand)
end

function sym_diff(op::FractionalLaplacian, var::ScalarField)
    d_operand = sym_diff(op.operand, var)
    if d_operand == 0 || d_operand === 0
        return 0
    end
    if isa(d_operand, Number)
        return 0
    end
    return FractionalLaplacian(d_operand, op.α)
end

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

# Gradient, Divergence, Curl pass through
function sym_diff(op::Gradient, var::ScalarField)
    d_operand = sym_diff(op.operand, var)
    if d_operand == 0 || d_operand === 0
        return 0
    end
    return Gradient(d_operand, op.coordsys)
end

function sym_diff(op::Divergence, var::ScalarField)
    d_operand = sym_diff(op.operand, var)
    if d_operand == 0 || d_operand === 0
        return 0
    end
    return Divergence(d_operand)
end

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

    dF = nothing
    for (var, pert) in zip(vars, perts)
        dF_dvar = sym_diff(F, var)
        if dF_dvar === 0 || dF_dvar == 0
            continue
        end
        term = _simplify_mul(dF_dvar, pert)
        dF = (dF === nothing) ? term : _simplify_add(dF, term)
    end
    return dF === nothing ? 0 : dF
end

"""
    build_symbolic_jacobian(problem, state_fields)

Build the Jacobian matrix for an NLBVP by symbolically differentiating
each equation residual with respect to each state variable, then
evaluating the derivatives at the current state.

Returns a sparse matrix J where J[i,j] = ∂F_i/∂u_j evaluated at current state.

For operator-valued derivatives (e.g., Laplacian), the existing matrix
infrastructure is used. For field-valued derivatives (e.g., 2*u from u²),
diagonal matrices are constructed from grid values.
"""
function build_symbolic_jacobian(problem, state_fields)
    n = length(fields_to_vector(state_fields))
    vars = state_fields

    # Get equation data
    eq_data_list = _get_equation_data(problem)
    if eq_data_list === nothing || isempty(eq_data_list)
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
        ensure_layout!(field, :c)
        push!(block_sizes, length(get_coeff_data(field)))
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
            # Compute ∂F_i/∂u_j symbolically
            dF_duj = sym_diff(F_i, var)

            if dF_duj !== 0 && dF_duj != 0
                # Evaluate the derivative at the current state
                block = _evaluate_jacobian_block(dF_duj, var, block_sizes[i], block_sizes[j])

                # Insert non-zeros into sparse arrays
                for (bi, bj, bv) in _sparse_entries(block)
                    push!(I_idx, row_offset + bi)
                    push!(J_idx, col_offset + bj)
                    push!(V_val, bv)
                end
            end

            col_offset += block_sizes[j]
        end
        row_offset += block_sizes[i]
    end

    return sparse(I_idx, J_idx, V_val, total_size, total_size)
end

"""Extract equation data list from problem."""
function _get_equation_data(problem)
    if hasfield(typeof(problem), :equation_data) && problem.equation_data !== nothing
        return problem.equation_data
    elseif hasfield(typeof(problem), :equations)
        return problem.equations
    end
    return nothing
end

"""Extract residual expression (LHS - RHS) from equation data."""
function _get_residual_expression(eq_data)
    if isa(eq_data, Dict)
        lhs = get(eq_data, "LHS", nothing)
        rhs = get(eq_data, "RHS", nothing)
        if lhs !== nothing && rhs !== nothing
            return SubtractOperator(lhs, rhs)
        elseif haskey(eq_data, "F")
            return eq_data["F"]
        end
    end
    return 0
end

"""
Evaluate a Jacobian block (derivative expression) to a matrix.
For scalar-valued expressions (field * perturbation), returns a diagonal matrix.
For operator-valued expressions (Laplacian), returns the operator matrix.

GPU-compatible: Field data is transferred to CPU before constructing sparse
matrices (spdiagm requires CPU arrays for scalar indexing).
"""
function _evaluate_jacobian_block(expr, var::ScalarField, nrows::Int, ncols::Int)
    if isa(expr, Number)
        # Constant: scalar * identity
        return expr * sparse(I, min(nrows, ncols), min(nrows, ncols))
    elseif isa(expr, ScalarField)
        # Field-valued: diagonal matrix from coefficient values
        ensure_layout!(expr, :c)
        data = get_coeff_data(expr)
        # Transfer to CPU if on GPU (sparse matrix construction requires CPU)
        cpu_data = is_gpu_array(data) ? Array(data) : data
        n = min(length(cpu_data), nrows, ncols)
        return spdiagm(0 => real.(cpu_data[1:n]))
    elseif isa(expr, Operator)
        # Try to evaluate as a matrix
        try
            result = evaluate(expr, :c)
            if isa(result, ScalarField)
                ensure_layout!(result, :c)
                data = get_coeff_data(result)
                # Transfer to CPU if on GPU
                cpu_data = is_gpu_array(data) ? Array(data) : data
                n = min(length(cpu_data), nrows, ncols)
                return spdiagm(0 => real.(cpu_data[1:n]))
            elseif isa(result, AbstractMatrix)
                # Ensure matrix is on CPU
                return is_gpu_array(result) ? Array(result) : result
            elseif isa(result, Number)
                return result * sparse(I, min(nrows, ncols), min(nrows, ncols))
            end
        catch e1
            # Fallback: try to get operator matrix from infrastructure
            try
                return subproblem_matrix(expr)
            catch e2
                # Last resort: identity block — warn so Newton convergence issues are diagnosable
                @warn "Jacobian block for $(typeof(expr)) could not be evaluated; using identity fallback. " *
                      "Newton convergence may be degraded or incorrect." evaluate_error=e1 matrix_error=e2 maxlog=3
                return sparse(I, min(nrows, ncols), min(nrows, ncols))
            end
        end
    end
    @warn "Unrecognized expression type $(typeof(expr)) in Jacobian block; using identity fallback." maxlog=3
    return sparse(I, min(nrows, ncols), min(nrows, ncols))
end

"""Extract sparse entries (i, j, v) from a matrix."""
function _sparse_entries(M)
    if isa(M, SparseMatrixCSC)
        I_m, J_m, V_m = findnz(M)
        return zip(I_m, J_m, V_m)
    elseif isa(M, AbstractMatrix)
        entries = Tuple{Int, Int, Float64}[]
        for j in 1:size(M, 2), i in 1:size(M, 1)
            v = M[i, j]
            if v != 0
                push!(entries, (i, j, Float64(v)))
            end
        end
        return entries
    elseif isa(M, Number)
        return [(1, 1, Float64(M))]
    end
    return Tuple{Int, Int, Float64}[]
end

# ============================================================================
# Exports
# ============================================================================

export UFUNC_DERIVATIVES, sym_diff, simplify
export frechet_differential, build_symbolic_jacobian
