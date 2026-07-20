"""
    Expression matrix composition

This file contains the compositional `expression_matrices` entry points and
their helper functions for linear operator assembly.
"""

# ============================================================================
# Expression Matrices for Matrix Assembly
# ============================================================================

"""
    subproblem_matrix(op::Operator, sp; kwargs...) -> Union{SparseMatrixCSC, Nothing}

Default fallback: returns `nothing` (no operator matrix).
Specific operators override this to return their sparse matrix representation.
"""
subproblem_matrix(op::Operator, sp; kwargs...) = nothing

# ============================================================================
# Helper functions for compositional expression_matrices
# ============================================================================

"""
    _depends_on_vars(expr, vars) -> Bool

Recursively check if `expr` references any variable in `vars`.
Used for linearity analysis in MultiplyOperator.
"""
function _depends_on_vars(expr, vars)
    # Direct field match
    if hasfield(typeof(expr), :name)
        for v in vars
            if v === expr
                return true
            end
            if hasfield(typeof(v), :name) && v.name == expr.name
                return true
            end
        end
    end
    # ScalarField: check if it's a component of a VectorField/TensorField in vars
    if isa(expr, ScalarField)
        for v in vars
            if isa(v, VectorField)
                for comp in v.components
                    if comp === expr || (hasfield(typeof(comp), :name) && comp.name == expr.name)
                        return true
                    end
                end
            elseif isa(v, TensorField)
                for comp in v.components
                    if comp === expr || (hasfield(typeof(comp), :name) && comp.name == expr.name)
                        return true
                    end
                end
            end
        end
    end
    # VectorField: check components
    if isa(expr, VectorField)
        for comp in expr.components
            _depends_on_vars(comp, vars) && return true
        end
    end
    # Recurse into operator children
    for f in (:operand, :left, :right)
        hasfield(typeof(expr), f) || continue
        child = getfield(expr, f)
        child === nothing && continue
        _depends_on_vars(child, vars) && return true
    end
    if hasfield(typeof(expr), :operands)
        ops = getfield(expr, :operands)
        if ops !== nothing
            for child in ops
                _depends_on_vars(child, vars) && return true
            end
        end
    end
    if isa(expr, Future)
        for child in future_args(expr)
            _depends_on_vars(child, vars) && return true
        end
    end
    return false
end

"""
    _is_const_or_param_local(expr) -> Bool

Check if an expression is a constant (number, parameter, ConstantOperator,
or a ScalarField with fixed data). Local version for matrices.jl.
"""
function _is_const_or_param_local(expr)
    isa(expr, Number) && return true
    isa(expr, ConstantOperator) && return true
    isa(expr, ZeroOperator) && return true
    isa(expr, NegateOperator) && return _is_const_or_param_local(expr.operand)
    # Constant ScalarField: 0D or single-element data (unit vectors, tau vars)
    if isa(expr, ScalarField)
        isempty(expr.bases) && return true
        gdata = get_grid_data(expr)
        gdata !== nothing && length(gdata) == 1 && return true
        cdata = get_coeff_data(expr)
        cdata !== nothing && length(cdata) == 1 && return true
    end
    return false
end

"""
    _extract_scalar_local(expr) -> Number

Extract scalar value from a constant expression. Local version for matrices.jl.
"""
function _extract_scalar_local(expr)
    isa(expr, Number) && return expr
    isa(expr, ConstantOperator) && return expr.value
    isa(expr, ZeroOperator) && return 0.0
    isa(expr, NegateOperator) && return -_extract_scalar_local(expr.operand)
    if isa(expr, ScalarField)
        gdata = get_grid_data(expr)
        gdata !== nothing && length(gdata) >= 1 && return real(gdata[1])
        cdata = get_coeff_data(expr)
        cdata !== nothing && length(cdata) >= 1 && return real(cdata[1])
        return 0.0  # uninitialized constant -> zero
    end
    return 1.0  # fallback
end

"""
    _build_constant_field_matrix(expr, sp) -> Union{SparseMatrixCSC, Nothing}

For a constant VectorField like `ez = [0; 1]`, build a `(ndim*Nz x Nz)` block
expansion matrix from component values. Each component's scalar value becomes
a scaled identity block, stacked vertically.
"""
function _build_constant_field_matrix(expr, sp)
    if !isa(expr, VectorField)
        return nothing
    end
    components = expr.components
    ndim = length(components)
    if ndim == 0
        return nothing
    end
    # Get the per-component block size from the first scalar component
    # For a constant VectorField, each component is a constant ScalarField
    # The block size is determined by what this VectorField would multiply
    # We need to infer the block size from the subproblem
    # Each component should have a single scalar value
    vals = ComplexF64[]
    for comp in components
        if _is_const_or_param_local(comp)
            push!(vals, ComplexF64(_extract_scalar_local(comp)))
        else
            return nothing  # Not all components are constant
        end
    end
    return vals  # Return component values; caller builds the block matrix
end

function _merge_expression_mats(left_mats::Dict, right_mats::Dict; right_scale::ComplexF64=ComplexF64(1))
    result = Dict{Any, SparseMatrixCSC}()
    for (var, mat) in left_mats
        result[var] = mat
    end
    for (var, mat) in right_mats
        scaled = right_scale * mat
        if haskey(result, var)
            left = result[var]
            if size(left, 1) != size(scaled, 1)
                nrows = max(size(left, 1), size(scaled, 1))
                left = _promote_expression_rows(left, nrows)
                scaled = _promote_expression_rows(scaled, nrows)
            end
            result[var] = left + scaled
        else
            result[var] = scaled
        end
    end
    return _normalize_expression_rows(result)
end

_scale_expression_mats(mats::Dict, coeff::ComplexF64) =
    Dict{Any, SparseMatrixCSC}(var => coeff * mat for (var, mat) in mats)

function _promote_expression_rows(mat::SparseMatrixCSC, target_rows::Int)
    size(mat, 1) == target_rows && return mat
    nrows, _ = size(mat)
    nrows == 0 && return spzeros(ComplexF64, target_rows, size(mat, 2))
    target_rows % nrows == 0 || return mat

    # Place scalar/low-rank contributions at the FIRST row block, not the last.
    # Placing at the last row conflicts with Lift operators (which also target
    # the highest Chebyshev mode), creating rank deficiencies.
    rows = collect(1:nrows)
    cols = collect(1:nrows)
    P = sparse(rows, cols, ones(ComplexF64, nrows), target_rows, nrows)
    return P * mat
end

function _normalize_expression_rows(mats::Dict)
    isempty(mats) && return Dict{Any, SparseMatrixCSC}()
    target_rows = maximum(size(mat, 1) for mat in values(mats))
    result = Dict{Any, SparseMatrixCSC}()
    for (var, mat) in mats
        result[var] = _promote_expression_rows(mat, target_rows)
    end
    return result
end

function _expand_constant_vector_product(vec_expr, child_mats::Dict, sp)
    comp_vals = _build_constant_field_matrix(vec_expr, sp)
    comp_vals === nothing && return Dict{Any, SparseMatrixCSC}()
    isempty(child_mats) && return Dict{Any, SparseMatrixCSC}()

    result = Dict{Any, SparseMatrixCSC}()
    for (var, mat) in child_mats
        n = size(mat, 1)
        ndim = length(comp_vals)
        blocks = [comp_vals[i] * sparse(ComplexF64(1) * I, n, n) for i in 1:ndim]
        expansion = vcat(blocks...)
        result[var] = expansion * mat
    end
    return result
end

"""
    expression_matrices(expr::Future, sp, vars; kwargs...)

Handle Future hierarchy types (Add, Subtract, Multiply, Negate, Divide).
These have args accessed via future_args(expr), not left/right fields.
"""
function expression_matrices(expr::Future, sp, vars; kwargs...)
    @debug "expression_matrices(::Future) called for $(typeof(expr)) with $(length(future_args(expr))) args"
    return _expression_matrices_future(expr, sp, vars; kwargs...)
end

function _expression_matrices_future(expr::Future, sp, vars; kwargs...)
    args = collect(Any, future_args(expr))

    if isa(expr, Add)
        result = Dict{Any, SparseMatrixCSC}()
        for arg in args
            result = _merge_expression_mats(result, expression_matrices(arg, sp, vars; kwargs...))
        end
        return result
    end

    if isa(expr, Subtract)
        isempty(args) && return Dict{Any, SparseMatrixCSC}()
        result = expression_matrices(args[1], sp, vars; kwargs...)
        for arg in args[2:end]
            result = _merge_expression_mats(result, expression_matrices(arg, sp, vars; kwargs...);
                                            right_scale=ComplexF64(-1))
        end
        return result
    end

    if isa(expr, Negate)
        isempty(args) && return Dict{Any, SparseMatrixCSC}()
        return _scale_expression_mats(expression_matrices(args[1], sp, vars; kwargs...), ComplexF64(-1))
    end

    if isa(expr, Divide)
        length(args) < 2 && return Dict{Any, SparseMatrixCSC}()
        denom = args[2]
        (_is_const_or_param_local(denom) || isa(denom, Number)) || return Dict{Any, SparseMatrixCSC}()
        coeff = ComplexF64(1) / ComplexF64(_extract_scalar_local(denom))
        return _scale_expression_mats(expression_matrices(args[1], sp, vars; kwargs...), coeff)
    end

    if isa(expr, Multiply)
        vec_idx = findfirst(arg -> isa(arg, VectorField) && !_depends_on_vars(arg, vars), args)
        if vec_idx !== nothing
            other_args = Any[arg for (idx, arg) in enumerate(args) if idx != vec_idx]
            child = isempty(other_args) ? nothing : (length(other_args) == 1 ? other_args[1] : Multiply(other_args...))
            child_mats = child === nothing ? Dict{Any, SparseMatrixCSC}() :
                         expression_matrices(child, sp, vars; kwargs...)
            return _expand_constant_vector_product(args[vec_idx], child_mats, sp)
        end

        scalar_coeff = ComplexF64(1)
        dependent = Any[]
        ncc_factors = Any[]      # non-constant FIELD coefficients (NCC), e.g. q(z) in q*u
        for arg in args
            if _is_const_or_param_local(arg) || isa(arg, Number)
                scalar_coeff *= ComplexF64(_extract_scalar_local(arg))
            elseif _depends_on_vars(arg, vars)
                push!(dependent, arg)
            else
                # Non-constant field coefficient (NCC) on the implicit side — handled below
                # via the SAME multiply-by-q builder the string MultiplyOperator path uses,
                # instead of being dropped. (Previously this object-syntax path warned+dropped.)
                push!(ncc_factors, arg)
            end
        end

        length(dependent) == 1 || return Dict{Any, SparseMatrixCSC}()
        child_mats = expression_matrices(only(dependent), sp, vars; kwargs...)
        # Apply each NCC factor as a multiply-by-coefficient matrix (mirrors the
        # MultiplyOperator Case-3 string path). `_implicit_ncc_matrix` reports
        # `ImplicitNCCUnsupported` for coefficients that cannot live in the implicit
        # operator, and `_apply_implicit_ncc` turns that into a descriptive error — the
        # two construction routes behave identically, and neither drops a term.
        for ncc in ncc_factors
            # Non-scalar (VectorField/TensorField) factors are rank-changing block expansions,
            # not scalar coefficients — a separate, pre-existing gap left untouched here.
            _is_scalar_coefficient(ncc) || continue
            child_mats = _apply_implicit_ncc(_implicit_ncc_matrix(ncc), child_mats, _expr_label(expr))
        end
        return _scale_expression_mats(child_mats, scalar_coeff)
    end

    return Dict{Any, SparseMatrixCSC}()
end

# ============================================================================
# Compositional expression_matrices methods
# ============================================================================

"""
    expression_matrices(op::Operator, sp, vars; kwargs...)

Generic Operator fallback for compositional expression_matrices.
For single-operand operators, recurse into operand then left-multiply by
the operator's subproblem_matrix.

Returns Dict mapping variables to sparse matrices.
"""
function expression_matrices(op::Operator, sp, vars; kwargs...)
    if hasfield(typeof(op), :operand)
        operand_mats = expression_matrices(op.operand, sp, vars; kwargs...)
        if isempty(operand_mats); return operand_mats; end
        op_mat = subproblem_matrix(op, sp; kwargs...)
        if op_mat === nothing; return operand_mats; end
        return Dict(var => op_mat * mat for (var, mat) in operand_mats)
    end
    return Dict{Any, SparseMatrixCSC}()
end

"""
    expression_matrices(op::AddOperator, sp, vars; kwargs...)

Merge child dicts, summing matrices for shared keys.
"""
function expression_matrices(op::AddOperator, sp, vars; kwargs...)
    left_mats = expression_matrices(op.left, sp, vars; kwargs...)
    right_mats = expression_matrices(op.right, sp, vars; kwargs...)
    return _merge_expression_mats(left_mats, right_mats)
end

"""
    expression_matrices(op::SubtractOperator, sp, vars; kwargs...)

Merge child dicts, subtracting right from left.
"""
function expression_matrices(op::SubtractOperator, sp, vars; kwargs...)
    left_mats = expression_matrices(op.left, sp, vars; kwargs...)
    right_mats = expression_matrices(op.right, sp, vars; kwargs...)
    return _merge_expression_mats(left_mats, right_mats; right_scale=ComplexF64(-1))
end

"""
    expression_matrices(op::NegateOperator, sp, vars; kwargs...)

Negate all matrices in child's dict.
"""
function expression_matrices(op::NegateOperator, sp, vars; kwargs...)
    child_mats = expression_matrices(op.operand, sp, vars; kwargs...)
    return Dict(var => -mat for (var, mat) in child_mats)
end

"""
    expression_matrices(op::MultiplyOperator, sp, vars; kwargs...)

Handle three cases:
1. Scalar constant x expression -> scalar multiply all matrices
2. Constant VectorField x expression (e.g., b * ez) -> block-expansion
3. One side depends on vars, other doesn't -> extract var-dependent side's matrices
"""
function expression_matrices(op::MultiplyOperator, sp, vars; kwargs...)
    left = op.left
    right = op.right

    left_const = _is_const_or_param_local(left)
    right_const = _is_const_or_param_local(right)

    # Case 1: Scalar constant on the left
    if left_const && !isa(left, VectorField)
        coeff = ComplexF64(_extract_scalar_local(left))
        child_mats = expression_matrices(right, sp, vars; kwargs...)
        return Dict(var => coeff * mat for (var, mat) in child_mats)
    end

    # Case 1: Scalar constant on the right
    if right_const && !isa(right, VectorField)
        coeff = ComplexF64(_extract_scalar_local(right))
        child_mats = expression_matrices(left, sp, vars; kwargs...)
        return Dict(var => coeff * mat for (var, mat) in child_mats)
    end

    # Case 2: Constant VectorField on the left (e.g., ez * b)
    if isa(left, VectorField) && !_depends_on_vars(left, vars)
        comp_vals = _build_constant_field_matrix(left, sp)
        if comp_vals !== nothing
            child_mats = expression_matrices(right, sp, vars; kwargs...)
            if isempty(child_mats); return child_mats; end
            result = Dict{Any, SparseMatrixCSC}()
            for (var, mat) in child_mats
                n = size(mat, 1)
                ndim = length(comp_vals)
                # Build block expansion matrix: stack ndim blocks of (val_i * I_n)
                blocks = [comp_vals[i] * sparse(ComplexF64(1)*I, n, n) for i in 1:ndim]
                expansion = vcat(blocks...)
                result[var] = expansion * mat
            end
            return result
        end
    end

    # Case 2: Constant VectorField on the right (e.g., b * ez)
    if isa(right, VectorField) && !_depends_on_vars(right, vars)
        comp_vals = _build_constant_field_matrix(right, sp)
        if comp_vals !== nothing
            child_mats = expression_matrices(left, sp, vars; kwargs...)
            if isempty(child_mats); return child_mats; end
            result = Dict{Any, SparseMatrixCSC}()
            for (var, mat) in child_mats
                n = size(mat, 1)
                ndim = length(comp_vals)
                # Build block expansion matrix: stack ndim blocks of (val_i * I_n)
                blocks = [comp_vals[i] * sparse(ComplexF64(1)*I, n, n) for i in 1:ndim]
                expansion = vcat(blocks...)
                result[var] = expansion * mat
            end
            return result
        end
    end

    # Case 3: one side depends on vars, the other is a (non-constant) coefficient.
    # The coefficient side is a spatially-varying FIELD (NCC) — `q(z)*u`. It must
    # become a multiply-by-q matrix that left-multiplies the var side's block.
    # Previously this branch returned the var side alone, SILENTLY DROPPING q; a
    # coefficient that cannot be represented implicitly now RAISES (never drops).
    left_dep = _depends_on_vars(left, vars)
    right_dep = _depends_on_vars(right, vars)

    if left_dep && !right_dep
        child = expression_matrices(left, sp, vars; kwargs...)
        _is_scalar_coefficient(right) || return child   # rank-changing factor; see below
        return _apply_implicit_ncc(_implicit_ncc_matrix(right), child, _expr_label(op))
    elseif right_dep && !left_dep
        child = expression_matrices(right, sp, vars; kwargs...)
        _is_scalar_coefficient(left) || return child
        return _apply_implicit_ncc(_implicit_ncc_matrix(left), child, _expr_label(op))
    end

    # Both depend on vars (nonlinear) or neither depends -> empty
    return Dict{Any, SparseMatrixCSC}()
end

"""
    ImplicitNCCUnsupported

Result of `_implicit_ncc_matrix` when a field-valued (non-constant) coefficient on the
implicit LHS cannot be turned into a multiply-by-coefficient matrix. It carries the
reason so the caller can raise a descriptive error.

This type exists because the alternative — returning `nothing` and letting the caller
pass the operand through unchanged — DISCARDS the coefficient, which silently turns
e.g. `dt(u) - nu(x)*lap(u) = 0` into an inviscid run. A dropped term is never an
acceptable outcome here, so every unsupported case must be reported, not swallowed.
"""
struct ImplicitNCCUnsupported
    reason::String
end

"""
    _ncc_direct_field(expr) -> ScalarField | Nothing

The ScalarField that `expr` *is*, seeing through value-preserving wrappers only
(`Grid`/`Coeff`/`Convert`/`Copy` are representation changes, not value changes).

Deliberately NOT `_resolve_operand_field`, which walks to the first leaf field anywhere
in the tree: for a coefficient like `∂z(q)` that would hand back `q` and build a
multiply-by-`q` matrix — a silently WRONG operator. Anything other than a bare field is
reported as unsupported instead.
"""
function _ncc_direct_field(expr)
    isa(expr, ScalarField) && return expr
    if isa(expr, Union{Grid, Coeff, Convert, Copy}) && hasfield(typeof(expr), :operand)
        return _ncc_direct_field(expr.operand)
    end
    return nothing
end

"""
    _is_scalar_coefficient(expr) -> Bool

True when `expr` is a SCALAR-valued coefficient factor, i.e. one whose only possible matrix
form is "multiply by a scalar field". Such a factor must either be built into the implicit
operator or reported — never dropped.

Deliberately FALSE for VectorField/TensorField-valued factors such as the `ez` in `b*ez`.
Those are rank-CHANGING block expansions handled separately (`expression_matrices` Case 2,
`_build_constant_field_matrix`), and the global matrix path has never implemented them at
all. Routing them into the NCC error path would raise on long-working Boussinesq and
rotating-frame equations — a pre-existing gap, distinct from the dropped-scalar-coefficient
bug this path exists to prevent.
"""
@inline _is_scalar_coefficient(expr) = _resolve_operand_field(expr) isa ScalarField

"""
    _implicit_ncc_matrix(ncc_operand) -> SparseMatrixCSC | ImplicitNCCUnsupported

Multiply-by-coefficient matrix for a non-constant FIELD coefficient on the implicit
side (`q*u`), built pseudospectrally through the coupled basis's own transform.

Supported (returns the matrix): the coefficient is a bare field varying along a SINGLE
Jacobi (Chebyshev/Legendre/…) axis and CONSTANT along every Fourier/periodic axis.
Everything else returns `ImplicitNCCUnsupported` with a reason — the caller must raise,
never drop. A coefficient that is identically zero returns an explicit ZERO matrix
(the term really is zero; passing the operand through would wrongly imply q ≡ 1).
"""
function _implicit_ncc_matrix(ncc_operand)
    field = _ncc_direct_field(ncc_operand)
    if field === nothing
        return ImplicitNCCUnsupported(
            "the coefficient is the expression `$(_expr_label(ncc_operand))` rather than a plain " *
            "field; only a bare field coefficient can be built into an implicit multiply matrix")
    end
    if isempty(field.bases)
        return ImplicitNCCUnsupported(
            "the coefficient field `$(field.name)` has no spatial bases, so it has no " *
            "coefficient-space representation to multiply by")
    end
    if any(b -> b === nothing, field.bases)
        return ImplicitNCCUnsupported(
            "the coefficient field `$(field.name)` has an undefined (nothing) basis on one of its axes")
    end

    jac_axes  = findall(b -> isa(b, JacobiBasis), field.bases)
    four_axes = findall(b -> isa(b, FourierBasis), field.bases)

    if length(jac_axes) != 1
        detail = isempty(jac_axes) ?
            "it has no Chebyshev/Jacobi axis at all (on a fully periodic/Fourier domain a " *
            "varying coefficient couples every Fourier mode, so there is no per-mode matrix)" :
            "it varies along $(length(jac_axes)) Chebyshev/Jacobi axes and only a single " *
            "coupled axis can be represented"
        return ImplicitNCCUnsupported(
            "the coefficient field `$(field.name)` is not representable: $detail")
    end
    jax = jac_axes[1]
    coupled_basis = field.bases[jax]
    Nc = coupled_basis.meta.size

    # A coefficient that varies along a FOURIER axis couples Fourier modes and so cannot be
    # a per-subproblem (single-mode) matrix. Only a coefficient CONSTANT along every Fourier
    # axis (its non-DC Fourier coefficients vanish) is representable here — e.g. a z-dependent
    # diffusivity in a Fourier-x × Chebyshev-z channel.
    if !isempty(four_axes)
        ensure_layout!(field, :c)
        coeffs = get_coeff_data(field)
        if coeffs === nothing
            return ImplicitNCCUnsupported(
                "the coefficient field `$(field.name)` has no coefficient-space data available")
        end
        maxabs = maximum(abs, coeffs)
        # Identically zero coefficient: the term genuinely vanishes. Return a real zero
        # matrix rather than `nothing`, which would have left the operand unscaled (q ≡ 1).
        maxabs == 0 && return spzeros(ComplexF64, Nc, Nc)
        for fax in four_axes
            if size(coeffs, fax) > 1
                nonDC = selectdim(coeffs, fax, 2:size(coeffs, fax))
                if !isempty(nonDC) && maximum(abs, nonDC) > 1e-8 * maxabs
                    fb = field.bases[fax]
                    axis_label = hasfield(typeof(fb), :meta) ? string(fb.meta.element_label) : "?"
                    return ImplicitNCCUnsupported(
                        "the coefficient field `$(field.name)` varies along the Fourier/periodic " *
                        "axis `$(axis_label)`, which couples Fourier modes and therefore has no " *
                        "per-mode implicit matrix")
                end
            end
        end
    end

    # q on the coupled-axis grid. Because q is constant along the Fourier directions, the
    # fiber at the first index of every other axis is the entire coefficient profile.
    ensure_layout!(field, :g)
    g = Array(get_grid_data(field))
    idx = ntuple(d -> (d == jax ? Colon() : 1), ndims(g))
    qfiber = vec(g[idx...])
    if length(qfiber) != Nc
        return ImplicitNCCUnsupported(
            "the coefficient field `$(field.name)` has $(length(qfiber)) grid points along its " *
            "coupled axis but the basis has $Nc coefficients")
    end
    if !(eltype(qfiber) <: Real)
        return ImplicitNCCUnsupported(
            "the coefficient field `$(field.name)` has complex grid data; only real-valued " *
            "implicit coefficients are supported")
    end
    # All-zero profile: the term is genuinely zero (see the maxabs branch above).
    any(!=(0), qfiber) || return spzeros(ComplexF64, Nc, Nc)

    # Build the multiply-by-q matrix PSEUDOSPECTRALLY through the coupled basis's OWN transform:
    # Q = F · diag(q) · B (column k = forward(q .* backward(eₖ))). Convention-EXACT — it uses
    # the actual Chebyshev/Legendre transform, sidestepping the standard-Jacobi-vs-normalized
    # coefficient mismatch that makes the raw `ncc_matrix`/`product_matrix` disagree with
    # Tarang's stored coeffs. Built on a fresh 1D domain so it is independent of the (possibly
    # multi-dimensional) parent distributor and applies per Fourier-mode subproblem.
    tmp = _ncc_temp_field(coupled_basis)
    if tmp === nothing
        return ImplicitNCCUnsupported(
            "the coupled basis type $(nameof(typeof(coupled_basis))) of coefficient field " *
            "`$(field.name)` cannot be rebuilt as a stand-alone 1-D basis, so its " *
            "multiply-by-coefficient transform cannot be assembled")
    end
    Q = zeros(ComplexF64, Nc, Nc)
    for k in 1:Nc
        ensure_layout!(tmp, :c)
        cd = get_coeff_data(tmp); fill!(cd, 0); cd[k] = 1
        backward_transform!(tmp)                            # eₖ → grid
        gg = get_grid_data(tmp); gg .= vec(gg) .* qfiber    # pointwise × q
        forward_transform!(tmp)                             # → coeffs
        @views Q[:, k] .= vec(Array(get_coeff_data(tmp)))
    end
    return sparse(Q)
end

# Fresh 1D ScalarField on a stand-alone copy of the coupled Jacobi basis, used only to run
# the basis's forward/backward transform when assembling the pseudospectral NCC matrix.
# Returns `nothing` for basis types we don't reconstruct (caller falls back).
function _ncc_temp_field(coupled_basis)
    lo, hi = coupled_basis.meta.bounds
    N      = coupled_basis.meta.size
    cname  = coupled_basis.meta.element_label
    coord  = CartesianCoordinates(cname)
    dist   = Distributor(coord; dtype=Float64)
    b1     = _rebuild_jacobi_1d(coupled_basis, coord[cname], N, Float64(lo), Float64(hi))
    b1 === nothing && return nothing
    return ScalarField(Domain(dist, (b1,)), "_ncc_tmp")
end

_rebuild_jacobi_1d(::ChebyshevT, coord, N, lo, hi) = ChebyshevT(coord; size=N, bounds=(lo, hi))
_rebuild_jacobi_1d(::ChebyshevU, coord, N, lo, hi) = ChebyshevU(coord; size=N, bounds=(lo, hi))
_rebuild_jacobi_1d(::Legendre,   coord, N, lo, hi) = Legendre(coord;   size=N, bounds=(lo, hi))
_rebuild_jacobi_1d(::Basis,      coord, N, lo, hi) = nothing

"""
    ImplicitNCCError

Raised when an implicit (LHS) term carries a field-valued coefficient that cannot be
represented in the implicit operator. Assembling the term without its coefficient would
silently change the equation being solved, so assembly stops instead.
"""
struct ImplicitNCCError <: Exception
    term::String
    reason::String
end

function Base.showerror(io::IO, e::ImplicitNCCError)
    print(io, """
    Tarang: the implicit (left-hand-side) term `$(e.term)` has a field-valued (non-constant)
    coefficient that cannot be represented in the implicit operator, because $(e.reason).

    This is an ERROR rather than a warning because the only alternative is to assemble the
    term without its coefficient, which silently solves a different equation (for a diffusion
    term, an inviscid run that shows no dependence on the coefficient at all).

    Fix: move the term to the EXPLICIT right-hand side and expand the product yourself, e.g.

        dt(u) = nu_e*lap(u) + ∂x(nu_e)*∂x(u) + ∂y(nu_e)*∂y(u)

    Alternatively, keep a CONSTANT scalar coefficient on the LHS (so the implicit solve stays
    well conditioned) and put only the varying part on the RHS. Tarang deliberately does not
    move the term for you: that would change the implicit/explicit splitting, and with it the
    stability properties of your timestepper, behind your back.

    Field coefficients are supported in the implicit operator only when the coefficient varies
    along a single Chebyshev/Jacobi axis and is constant along every Fourier/periodic axis.""")
end

# `_implicit_ncc_matrix` never returns `nothing`; an unsupported coefficient is reported as
# `ImplicitNCCUnsupported` and raised here. Dropping it is never an option — see the struct docs.
function _apply_implicit_ncc(u::ImplicitNCCUnsupported, child, term_label::AbstractString="<implicit term>")
    # If the variable side contributes nothing anyway, the coefficient is irrelevant: there is
    # no term to lose, so do not raise on a block that was always going to be empty.
    (isempty(child) || all(mat -> nnz(mat) == 0, values(child))) && return child
    throw(ImplicitNCCError(String(term_label), u.reason))
end

function _apply_implicit_ncc(Q::AbstractMatrix, child, term_label::AbstractString="<implicit term>")
    result = Dict{Any, SparseMatrixCSC}()
    for (var, mat) in child
        if size(Q, 2) == size(mat, 1)
            result[var] = sparse(Q * mat)
        elseif nnz(mat) == 0
            result[var] = mat   # empty block: nothing to scale, nothing to lose
        else
            # Size mismatch (truncated/tau-augmented block). Passing `mat` through would drop
            # the coefficient for a block that really does carry the term — that is the exact
            # silent-wrong-answer this path exists to prevent, so raise.
            throw(ImplicitNCCError(String(term_label),
                "its multiply-by-coefficient matrix $(size(Q)) does not match the " *
                "$(size(mat)) operand block for variable `$(_expr_label(var))`"))
        end
    end
    return result
end

"""
    _expr_label(expr) -> String

Short human-readable rendering of an expression tree, used to name the offending term in
implicit-NCC errors. Best-effort: falls back to the operator's type name.
"""
function _expr_label(expr)
    isa(expr, AbstractString) && return String(expr)
    isa(expr, Number) && return string(expr)
    isa(expr, ConstantOperator) && return string(expr.value)
    isa(expr, ZeroOperator) && return "0"
    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        return string(expr.name)
    end
    isa(expr, NegateOperator) && return "-" * _expr_label(expr.operand)
    isa(expr, AddOperator)      && return _expr_label(expr.left) * " + " * _expr_label(expr.right)
    isa(expr, SubtractOperator) && return _expr_label(expr.left) * " - " * _expr_label(expr.right)
    isa(expr, MultiplyOperator) && return _expr_label(expr.left) * "*" * _expr_label(expr.right)
    isa(expr, DivideOperator)   && return _expr_label(expr.left) * "/" * _expr_label(expr.right)
    isa(expr, PowerOperator)    && return _expr_label(expr.left) * "^" * _expr_label(expr.right)
    isa(expr, TimeDerivative)   && return "dt(" * _expr_label(expr.operand) * ")"
    isa(expr, Laplacian)        && return "lap(" * _expr_label(expr.operand) * ")"
    isa(expr, Gradient)         && return "grad(" * _expr_label(expr.operand) * ")"
    isa(expr, Divergence)       && return "div(" * _expr_label(expr.operand) * ")"
    isa(expr, Curl)             && return "curl(" * _expr_label(expr.operand) * ")"
    if isa(expr, Differentiate)
        cname = hasfield(typeof(expr), :coord) && expr.coord !== nothing ? string(expr.coord.name) : "?"
        ord = hasfield(typeof(expr), :order) ? expr.order : 1
        suffix = ord == 1 ? "" : "^$ord"
        return "∂$(cname)$(suffix)(" * _expr_label(expr.operand) * ")"
    end
    if isa(expr, Future)
        args = future_args(expr)
        inner = join((_expr_label(a) for a in args), ", ")
        return string(nameof(typeof(expr))) * "(" * inner * ")"
    end
    if hasfield(typeof(expr), :operand)
        return string(nameof(typeof(expr))) * "(" * _expr_label(expr.operand) * ")"
    end
    return string(nameof(typeof(expr)))
end
