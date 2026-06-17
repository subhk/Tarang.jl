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
        for arg in args
            if _is_const_or_param_local(arg) || isa(arg, Number)
                scalar_coeff *= ComplexF64(_extract_scalar_local(arg))
            elseif _depends_on_vars(arg, vars)
                push!(dependent, arg)
            else
                return Dict{Any, SparseMatrixCSC}()
            end
        end

        length(dependent) == 1 || return Dict{Any, SparseMatrixCSC}()
        child_mats = expression_matrices(only(dependent), sp, vars; kwargs...)
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
    # Previously this branch returned the var side alone, SILENTLY DROPPING q.
    left_dep = _depends_on_vars(left, vars)
    right_dep = _depends_on_vars(right, vars)

    if left_dep && !right_dep
        child = expression_matrices(left, sp, vars; kwargs...)
        return _apply_implicit_ncc(_implicit_ncc_matrix(right), child)
    elseif right_dep && !left_dep
        child = expression_matrices(right, sp, vars; kwargs...)
        return _apply_implicit_ncc(_implicit_ncc_matrix(left), child)
    end

    # Both depend on vars (nonlinear) or neither depends -> empty
    return Dict{Any, SparseMatrixCSC}()
end

"""
    _implicit_ncc_matrix(ncc_operand) -> SparseMatrixCSC | Nothing

Multiply-by-coefficient matrix for a non-constant FIELD coefficient on the implicit
side (`q*u`). Built via the basis-aware `ncc_matrix`/`product_matrix` (correct for
Chebyshev/Legendre/Jacobi).

Supported (returns the matrix): the coefficient is a field varying along a SINGLE
Jacobi (Chebyshev/Legendre/…) axis with NO Fourier/periodic axis. Unsupported
(returns `nothing` after a one-time warning, so the coefficient is never DROPPED
silently): any Fourier-axis dependence (couples Fourier modes → not representable per
subproblem) or more than one Jacobi axis. Those belong in the explicit RHS.
"""
function _implicit_ncc_matrix(ncc_operand)
    field = _resolve_operand_field(ncc_operand)
    (field isa ScalarField && !isempty(field.bases)) || return nothing

    jac_axes  = findall(b -> isa(b, JacobiBasis), field.bases)
    four_axes = findall(b -> isa(b, FourierBasis), field.bases)

    if length(jac_axes) != 1
        @warn "Tarang: implicit NCC is supported only for a coefficient varying along a single " *
              "Chebyshev/Jacobi direction (this one has $(length(jac_axes)) Jacobi axes). The " *
              "term is being dropped from the implicit matrix — move it to the explicit RHS." maxlog=1
        return nothing
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
        coeffs === nothing && return nothing
        maxabs = maximum(abs, coeffs)
        maxabs == 0 && return nothing
        for fax in four_axes
            if size(coeffs, fax) > 1
                nonDC = selectdim(coeffs, fax, 2:size(coeffs, fax))
                if !isempty(nonDC) && maximum(abs, nonDC) > 1e-8 * maxabs
                    @warn "Tarang: a non-constant FIELD coefficient that varies along a " *
                          "Fourier/periodic axis couples Fourier modes and is not supported in the " *
                          "implicit operator. Dropping it — move the term to the explicit RHS." maxlog=1
                    return nothing
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
    (length(qfiber) == Nc && eltype(qfiber) <: Real && any(!=(0), qfiber)) || return nothing

    # Build the multiply-by-q matrix PSEUDOSPECTRALLY through the coupled basis's OWN transform:
    # Q = F · diag(q) · B (column k = forward(q .* backward(eₖ))). Convention-EXACT — it uses
    # the actual Chebyshev/Legendre transform, sidestepping the standard-Jacobi-vs-normalized
    # coefficient mismatch that makes the raw `ncc_matrix`/`product_matrix` disagree with
    # Tarang's stored coeffs. Built on a fresh 1D domain so it is independent of the (possibly
    # multi-dimensional) parent distributor and applies per Fourier-mode subproblem.
    tmp = _ncc_temp_field(coupled_basis)
    tmp === nothing && return nothing
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

# No NCC matrix (constant coefficient already handled, or unsupported+warned) → child blocks unchanged.
_apply_implicit_ncc(::Nothing, child) = child
function _apply_implicit_ncc(Q::AbstractMatrix, child)
    result = Dict{Any, SparseMatrixCSC}()
    for (var, mat) in child
        if size(Q, 2) == size(mat, 1)
            result[var] = sparse(Q * mat)
        else
            # Size mismatch (truncated/tau-augmented block) — don't crash; fall back.
            @warn "Tarang: NCC multiply matrix $(size(Q)) incompatible with operand block " *
                  "$(size(mat)); coefficient dropped for this block." maxlog=1
            result[var] = mat
        end
    end
    return result
end
