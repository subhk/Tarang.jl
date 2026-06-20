"""
    Problem matrix spectral builders

This file contains spectral operator matrix builders and the recursive
`build_expression_matrix_block` path used to assemble problem-level sparse
matrices.
"""

# ============================================================================
# Spectral operator matrices (replaces identity markers for ETD accuracy)
# ============================================================================

"""
    _spectral_operator_matrix(expr, var, eqn_size, var_size)

Try to build the actual spectral operator matrix for `expr` acting on `var`.
Returns the matrix if successful, or `nothing` to fall back to identity markers.

Uses existing infrastructure:
- `build_operator_differentiation_matrix` for ∂/∂x (Fourier diagonal, Chebyshev dense)
- Kronecker products for multi-dimensional operators
"""
function _spectral_operator_matrix(expr, var, eqn_size::Int, var_size::Int)
    field = _get_matching_scalar_field(expr, var)
    field === nothing && return nothing

    if isa(expr, Laplacian)
        # Composed operand (e.g. Δ(∂x(u))): build Δ as a square axis-operator and
        # left-multiply by the operand's matrix so inner derivatives are NOT dropped.
        if _operand_matches_variable(expr.operand, var)
            return _spectral_laplacian(field, eqn_size, var_size)
        end
        return _compose_outer_with_operand(
            _spectral_laplacian(field, var_size, var_size),
            expr.operand, var, eqn_size, var_size)
    elseif isa(expr, FractionalLaplacian)
        return _spectral_fractional_laplacian(field, expr.α, eqn_size, var_size)
    elseif isa(expr, Differentiate)
        # Composed operand (e.g. ∂x(∂x(u)), ∂x(∂y(u)), ∂x(lap(u))): the old code
        # walked straight to the innermost field and applied ONLY the outer
        # coord/order, silently dropping every intermediate operator — so ∂x(∂x(u))
        # was assembled as (ik)¹ instead of (ik)² = -k², turning an implicit
        # diffusion term into an advection-like one. Recurse instead: build the
        # outer 1D derivative as a square axis-operator and left-multiply by the
        # operand's matrix.
        if _operand_matches_variable(expr.operand, var)
            return _spectral_differentiate(field, expr.coord, expr.order, eqn_size, var_size)
        end
        return _compose_outer_with_operand(
            _spectral_differentiate(field, expr.coord, expr.order, var_size, var_size),
            expr.operand, var, eqn_size, var_size)
    elseif isa(expr, Gradient)
        return _spectral_gradient(expr, var, eqn_size, var_size)
    end
    return nothing
end

"""
    _compose_outer_with_operand(D_outer, operand, var, eqn_size, var_size)

Compose an outer (square, `var_size × var_size`) spectral axis-operator with the
matrix of a (possibly composed) inner `operand`: returns `D_outer · M(operand)`.
Used when a differential operator wraps another operator so that intermediate
derivatives are not dropped. Because every block here lives in the SAME coefficient
basis (all `var_size × var_size`), operator composition is just matrix
multiplication. Returns `nothing` if either piece can't be built or shapes mismatch.
"""
function _compose_outer_with_operand(D_outer, operand, var, eqn_size::Int, var_size::Int)
    D_outer === nothing && return nothing
    size(D_outer) == (var_size, var_size) || return nothing
    B = build_expression_matrix_block(operand, var, var_size, var_size)
    B === nothing && return nothing
    result = D_outer * B
    return size(result) == (eqn_size, var_size) ? result : nothing
end

"""Get the ScalarField that the operator acts on (matching var)."""
function _get_matching_scalar_field(expr, var)
    op = expr
    while hasfield(typeof(op), :operand)
        op = op.operand
    end
    if isa(op, ScalarField) && _operand_matches_variable(op, var)
        return op
    end
    return nothing
end

"""
    _coeff_shape(field)

Coefficient-space array shape, matching the actual state vector layout.
First RealFourier dimension is halved (rFFT); Chebyshev/Legendre keep N.
"""
function _coeff_shape(field::ScalarField)
    shape = Int[]
    first_rf = true
    for basis in field.bases
        basis === nothing && continue
        if isa(basis, RealFourier) && first_rf
            push!(shape, div(basis.meta.size, 2) + 1)
            first_rf = false
        else
            push!(shape, basis.meta.size)
        end
    end
    return tuple(shape...)
end

"""
    _spectral_laplacian(field, eqn_size, var_size)

Spectral Laplacian Δ = Σᵢ ∂²/∂xᵢ² in the complex coefficient representation.

- Fourier dimensions: diagonal -(2πk/L)²
- Chebyshev dimensions: dense D² matrix (from differentiation_matrix)
- Combined: Kronecker product matching the state vector layout
"""
function _spectral_laplacian(field::ScalarField, eqn_size::Int, var_size::Int)
    cshape = _coeff_shape(field)
    prod(cshape) != var_size && return nothing

    # Build 1D second-derivative operators for each dimension
    ops_1d = AbstractMatrix[]
    for (dim, basis) in enumerate(field.bases)
        basis === nothing && continue
        Nk = cshape[dim]

        if isa(basis, FourierBasis)
            # Fourier: diagonal with -k²
            N = basis.meta.size
            L = basis.meta.bounds[2] - basis.meta.bounds[1]
            k0 = 2π / L
            is_rfft = isa(basis, RealFourier) && dim == findfirst(b -> isa(b, RealFourier), field.bases)

            vals = zeros(ComplexF64, Nk)
            for j in 1:Nk
                if is_rfft
                    k = (j - 1) * k0
                else
                    k = j <= N÷2+1 ? (j-1)*k0 : (j-N-1)*k0
                end
                vals[j] = -k^2
            end
            push!(ops_1d, spdiagm(0 => vals))
        else
            # Chebyshev/Legendre: dense D² matrix
            D2 = try
                ComplexF64.(differentiation_matrix(basis, 2))
            catch
                nothing
            end
            D2 === nothing && return nothing
            push!(ops_1d, sparse(D2))
        end
    end

    isempty(ops_1d) && return nothing

    # Sum of Kronecker products: Σ_d (I ⊗ ... ⊗ D²_d ⊗ ... ⊗ I)
    ndims = length(ops_1d)
    result = spzeros(ComplexF64, var_size, var_size)
    for d in 1:ndims
        # Build Kronecker factors: identity for all dims except d
        mats = [i == d ? ops_1d[d] : sparse(ComplexF64(1)*I, size(ops_1d[i],1), size(ops_1d[i],1))
                for i in 1:ndims]
        # Kronecker in reverse order for column-major layout
        term = mats[end]
        for i in (ndims-1):-1:1
            term = kron(term, mats[i])
        end
        result += term
    end

    return size(result) == (eqn_size, var_size) ? result : nothing
end

"""Spectral fractional Laplacian: (-Δ)^α — uses |k|^{2α} for fully Fourier domains."""
function _spectral_fractional_laplacian(field::ScalarField, α::Float64, eqn_size::Int, var_size::Int)
    all_fourier = all(b -> b !== nothing && isa(b, FourierBasis), field.bases)
    !all_fourier && return nothing

    lap = _spectral_laplacian(field, eqn_size, var_size)
    lap === nothing && return nothing

    # Laplacian is diagonal for Fourier: entries are -|k|²
    # (-Δ)^α = |k|^{2α}
    diag_vals = diag(lap)
    if α >= 0
        frac_vals = [abs(v)^α for v in diag_vals]
    else
        threshold = 1e-14
        frac_vals = [abs(v) > threshold ? abs(v)^α : 0.0 for v in diag_vals]
    end
    n = min(eqn_size, var_size, length(frac_vals))
    return spdiagm(eqn_size, var_size, 0 => ComplexF64.(frac_vals[1:n]))
end

"""
    _try_div_grad_spectral(div_expr, var, eqn_size, var_size)

Recognize div(grad(f)) or div(grad(f) + ...) as a Laplacian acting on `f`.
The first-order substitution writes div(grad_f) where
grad_f = grad(f) + ez*lift(tau). The Gradient(f) term produces the
spectral Laplacian; the lift terms are handled separately.
"""
function _try_div_grad_spectral(div_expr::Divergence, var, eqn_size::Int, var_size::Int)
    inner = div_expr.operand

    # Search all addends for Gradient(var) — handles both
    # div(grad(f)) and div(grad(f) + ez*lift(tau))
    addends = isa(inner, Gradient) ? [inner] : _collect_all_addends(inner)
    for addend in addends
        actual = isa(addend, NegateOperator) ? addend.operand : addend
        if isa(actual, Gradient) && _operand_matches_variable(actual.operand, var)
            return _spectral_laplacian_for_var(var, eqn_size, var_size)
        end
    end

    return nothing
end

"""Build spectral Laplacian sized to match a variable (scalar or block-diagonal for vector)."""
function _spectral_laplacian_for_var(var, eqn_size::Int, var_size::Int)
    if isa(var, ScalarField)
        return _spectral_laplacian(var, eqn_size, var_size)
    elseif isa(var, VectorField)
        # Block-diagonal: Δ ⊕ Δ ⊕ ... ⊕ Δ (one block per component)
        n_comp = length(var.components)
        comp_size = var_size ÷ n_comp
        lap = _spectral_laplacian(var.components[1], comp_size, comp_size)
        lap === nothing && return nothing
        # Build block-diagonal via kron(I_ncomp, lap)
        result = kron(sparse(ComplexF64(1)*I, n_comp, n_comp), lap)
        return size(result) == (eqn_size, var_size) ? result : nothing
    end
    return nothing
end

"""Collect all addends from nested Add/AddOperator/Subtract/Future chains."""
function _collect_all_addends(expr)
    result = Any[]
    if isa(expr, AddOperator)
        append!(result, _collect_all_addends(expr.left))
        append!(result, _collect_all_addends(expr.right))
    elseif isa(expr, SubtractOperator)
        append!(result, _collect_all_addends(expr.left))
        for a in _collect_all_addends(expr.right)
            push!(result, NegateOperator(a))
        end
    elseif isa(expr, Add) || isa(expr, Subtract)
        args = future_args(expr)
        if isa(expr, Add)
            for arg in args
                append!(result, _collect_all_addends(arg))
            end
        elseif length(args) >= 2
            append!(result, _collect_all_addends(args[1]))
            for a in _collect_all_addends(args[2])
                push!(result, NegateOperator(a))
            end
        end
    else
        push!(result, expr)
    end
    return result
end

"""Spectral differentiation: d^n/dx^n in complex coefficient representation."""
function _spectral_differentiate(field::ScalarField, coord::Coordinate, order::Int,
                                  eqn_size::Int, var_size::Int)
    cshape = _coeff_shape(field)
    prod(cshape) != var_size && return nothing

    # Find which dimension matches this coordinate
    dim_idx = nothing
    for (i, basis) in enumerate(field.bases)
        basis === nothing && continue
        if basis.meta.element_label == coord.name
            dim_idx = i
            break
        end
    end
    dim_idx === nothing && return nothing

    basis = field.bases[dim_idx]
    Nk = cshape[dim_idx]

    # Build 1D differentiation operator
    D1d = nothing
    if isa(basis, FourierBasis)
        N = basis.meta.size
        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        k0 = 2π / L
        is_rfft = isa(basis, RealFourier) && dim_idx == findfirst(b -> isa(b, RealFourier), field.bases)

        vals = zeros(ComplexF64, Nk)
        for j in 1:Nk
            k = is_rfft ? (j-1)*k0 : (j <= N÷2+1 ? (j-1)*k0 : (j-N-1)*k0)
            vals[j] = (im * k)^order
        end
        D1d = spdiagm(0 => vals)
    else
        D_dense = try ComplexF64.(differentiation_matrix(basis, order)) catch; nothing end
        D_dense === nothing && return nothing
        D1d = sparse(D_dense)
    end

    # Kronecker product for multi-dimensional
    ndims = length(cshape)
    if ndims == 1
        return size(D1d) == (eqn_size, var_size) ? D1d : nothing
    end

    mats = [i == dim_idx ? D1d : sparse(ComplexF64(1)*I, cshape[i], cshape[i])
            for i in 1:ndims]
    result = mats[end]
    for i in (ndims-1):-1:1
        result = kron(result, mats[i])
    end
    return size(result) == (eqn_size, var_size) ? result : nothing
end

"""Spectral gradient of a scalar field, stacked as [∂x f; ∂y f; ...]."""
function _spectral_gradient(expr::Gradient, var, eqn_size::Int, var_size::Int)
    field = expr.operand
    isa(field, ScalarField) || return nothing
    _operand_matches_variable(field, var) || return nothing

    coordsys = expr.coordsys
    coordsys === nothing && return nothing
    ndim = length(coordsys.names)
    eqn_size == ndim * var_size || return nothing

    blocks = SparseMatrixCSC{ComplexF64, Int}[]
    for coord_name in coordsys.names
        coord = coordsys[coord_name]
        block = _spectral_differentiate(field, coord, 1, var_size, var_size)
        block === nothing && return nothing
        push!(blocks, block)
    end
    return vcat(blocks...)
end

function _is_2d_vector_skew_operand(operand)
    if isa(operand, VectorField)
        return length(operand.components) == 2
    elseif isa(operand, Gradient)
        return isa(operand.operand, ScalarField) &&
               operand.coordsys !== nothing &&
               length(operand.coordsys.names) == 2
    end
    return false
end

"""Matrix block for 2D vector skew: skew(vx, vy) = (-vy, vx)."""
function _spectral_skew_block(expr::Skew, var, eqn_size::Int, var_size::Int)
    _is_2d_vector_skew_operand(expr.operand) || return nothing
    iseven(eqn_size) || return nothing

    operand_block = build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
    nnz(operand_block) == 0 && return operand_block

    scalar_size = eqn_size ÷ 2
    rotation = kron(
        sparse(ComplexF64[0 -1; 1 0]),
        sparse(ComplexF64(1) * I, scalar_size, scalar_size),
    )
    return rotation * operand_block
end

"""
    Build matrix block for expression acting on variable.
    Following expression_matrices pattern.
    """
function build_expression_matrix_block(expr, var, eqn_size::Int, var_size::Int)

    # ── 1. Direct variable reference ────────────────────────────
    if _operand_matches_variable(expr, var)
        return _identity_block(eqn_size, var_size)
    end

    # ── 2. Arithmetic combinators ───────────────────────────────
    if isa(expr, AddOperator)
        return build_expression_matrix_block(expr.left, var, eqn_size, var_size) +
               build_expression_matrix_block(expr.right, var, eqn_size, var_size)
    end
    if isa(expr, SubtractOperator)
        return build_expression_matrix_block(expr.left, var, eqn_size, var_size) -
               build_expression_matrix_block(expr.right, var, eqn_size, var_size)
    end
    if isa(expr, NegateOperator)
        return -build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
    end
    if isa(expr, MultiplyOperator)
        # const * expr  or  expr * const  → scale the block
        if _is_const_or_param(expr.left)
            coeff = _extract_scalar(expr.left)
            return ComplexF64(coeff) * build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        elseif _is_const_or_param(expr.right)
            coeff = _extract_scalar(expr.right)
            return ComplexF64(coeff) * build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        else
            # Nonlinear product (field * field) → belongs on RHS, zero in L
            return _zero_block(eqn_size, var_size)
        end
    end
    if isa(expr, DivideOperator)
        if _is_const_or_param(expr.right)
            coeff = _extract_scalar(expr.right)
            return (ComplexF64(1) / ComplexF64(coeff)) *
                   build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        else
            return _zero_block(eqn_size, var_size)
        end
    end
    if isa(expr, PowerOperator)
        # x^const is nonlinear unless x is itself const → zero
        return _zero_block(eqn_size, var_size)
    end

    # ── 2b. Future arithmetic types (from Julia-level expression building) ──
    # The first-order substitutions (grad_u = grad(u) + ez*lift(tau_u1))
    # produce Add/Multiply/Negate/Subtract Future types, not AddOperator etc.
    if isa(expr, Future)
        args = future_args(expr)
        if isa(expr, Add)
            # Add flattens to N args: sum all blocks
            result = _zero_block(eqn_size, var_size)
            for arg in args
                result = result + build_expression_matrix_block(arg, var, eqn_size, var_size)
            end
            return result
        elseif isa(expr, Subtract) && length(args) >= 2
            return build_expression_matrix_block(args[1], var, eqn_size, var_size) -
                   build_expression_matrix_block(args[2], var, eqn_size, var_size)
        elseif isa(expr, Negate) && length(args) >= 1
            return -build_expression_matrix_block(args[1], var, eqn_size, var_size)
        elseif isa(expr, Multiply) && length(args) >= 2
            # Separate scalar constants from field/operator factors
            scalars = filter(a -> _is_const_or_param(a) || isa(a, Number), args)
            fields  = filter(a -> !(_is_const_or_param(a) || isa(a, Number)), args)
            if length(fields) > 1
                # Multiple field factors → nonlinear product → zero in L
                return _zero_block(eqn_size, var_size)
            end
            scalar_val = isempty(scalars) ? ComplexF64(1) :
                         prod(ComplexF64(_extract_scalar(s)) for s in scalars)
            if isempty(fields)
                return _zero_block(eqn_size, var_size)  # pure scalar product
            end
            return scalar_val * build_expression_matrix_block(fields[1], var, eqn_size, var_size)
        else
            # Unknown Future type (DotProduct, CrossProduct, Power, etc.)
            # These are generally nonlinear → zero in L
            return _zero_block(eqn_size, var_size)
        end
    end

    # ── 3. Constants & zeros ────────────────────────────────────
    if isa(expr, ZeroOperator) || isa(expr, ConstantOperator) ||
       isa(expr, Number) || isa(expr, AbstractArray)
        return _zero_block(eqn_size, var_size)
    end
    if isa(expr, UnknownOperator) || isa(expr, ArrayOperator)
        return _zero_block(eqn_size, var_size)
    end

    # ── 4. Spectral operator matrices ─────────────────────────────
    #
    # For Laplacian, FractionalLaplacian, Differentiate: try to build
    # the actual spectral matrix (Fourier diagonal, Chebyshev dense).
    # Falls back to identity markers if spectral matrix can't be built.
    # Accurate spectral matrices are essential for ETD time integration.

    if isa(expr, Laplacian)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size; scale=-1.0)
    end

    if isa(expr, FractionalLaplacian)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    if isa(expr, Differentiate)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    if isa(expr, Gradient)
        spec = _spectral_operator_matrix(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    if isa(expr, Skew)
        spec = _spectral_skew_block(expr, var, eqn_size, var_size)
        spec !== nothing && return spec
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    # ── 4b. div(grad(f)) → spectral Laplacian (first-order form) ──
    # The first-order substitution turns Δ(f) into div(grad_f).
    # When the Divergence operand contains a Gradient of the target variable,
    # build the spectral Laplacian matrix (sum of second-derivative operators).
    if isa(expr, Divergence)
        lap_matrix = _try_div_grad_spectral(expr, var, eqn_size, var_size)
        if lap_matrix !== nothing
            return lap_matrix
        end
    end

    if isa(expr, Union{
            # Time derivative
            TimeDerivative,
            # Differential operators (spectral matrices attempted in section 4 above;
            # this fallback handles cases where spectral matrix couldn't be built)
            FractionalLaplacian, Gradient, Divergence, Curl, Differentiate,
            # Algebraic operators
            Skew, Trace, TransposeComponents,
            # Projection / extraction
            Component, RadialComponent, AngularComponent, AzimuthalComponent,
            # Reduction operators
            Integrate, Average, Interpolate,
            # Conversion / representation
            Convert, Grid, Coeff, Lift, Copy,
            # Spectral operators
            HilbertTransform,
            # Function operators (sin(u), exp(u), etc. — nonlinear in general,
            # but we still mark participation so the matrix isn't structurally
            # singular; the implicit solve is an approximation for these)
            GeneralFunction, UnaryGridFunction,
            # CFL (diagnostic only, but handle gracefully)
            AdvectiveCFL})
        return _recurse_operand(expr, var, eqn_size, var_size)
    end

    # ── 5. Multi-operand operators ──────────────────────────────
    if isa(expr, Outer)
        # Outer product: left ⊗ right — nonlinear if both depend on variables.
        # Treat like Multiply: if one side is const, recurse the other.
        if _is_const_or_param(expr.left)
            return build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        elseif _is_const_or_param(expr.right)
            return build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        else
            return _zero_block(eqn_size, var_size)
        end
    end
    if isa(expr, IndexOperator)
        return hasfield(typeof(expr), :array) ?
               build_expression_matrix_block(expr.array, var, eqn_size, var_size) :
               _zero_block(eqn_size, var_size)
    end

    # ── 6. Fallback ─────────────────────────────────────────────
    @debug "Unhandled expression type in matrix block: $(typeof(expr))"
    return _zero_block(eqn_size, var_size)
end

# ── Helpers ─────────────────────────────────────────────────────

"""Recurse into a single-operand operator to build its matrix block."""
function _recurse_operand(expr, var, eqn_size::Int, var_size::Int; scale::Number=1.0)
    # Recursing operand-as-identity is CORRECT only for TimeDerivative (M = I) and
    # basis-conversion no-ops (Convert/Grid/Coeff/Copy). For genuine operators with no
    # spectral matrix builder (HilbertTransform, Integrate, Average, Curl, Gradient,
    # Divergence, Differentiate, FractionalLaplacian, …) this SILENTLY approximates the
    # operator as identity in the global L/M matrix — wrong for the implicit solve. Warn
    # loudly so it is not a silent correctness loss (the per-subproblem path is unaffected).
    if !(expr isa Union{TimeDerivative, Convert, Grid, Coeff, Copy})
        @warn "Implicit global matrix: operator $(typeof(expr)) has no spectral matrix " *
              "builder and is approximated as IDENTITY-on-operand; the implicit solve will " *
              "be wrong for this term. Add a builder or use the per-subproblem solve path." maxlog=3
    end
    inner = build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
    return scale == 1.0 ? inner : ComplexF64(scale) * inner
end

"""
Check if an expression is a constant (number, parameter, ConstantOperator,
or a ScalarField with fixed data — e.g. unit vector components).

A ScalarField is "constant" if it has no spatial bases (0D tau variable)
or has uniform single-element data (unit vector component created by
`unit_vector_fields`).
"""
function _is_const_or_param(expr)
    isa(expr, Number) && return true
    isa(expr, ConstantOperator) && return true
    isa(expr, ZeroOperator) && return true
    isa(expr, NegateOperator) && return _is_const_or_param(expr.operand)
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

"""Extract scalar value from a constant expression."""
function _extract_scalar(expr)
    isa(expr, Number) && return expr
    isa(expr, ConstantOperator) && return expr.value
    isa(expr, ZeroOperator) && return 0.0
    isa(expr, NegateOperator) && return -_extract_scalar(expr.operand)
    if isa(expr, ScalarField)
        gdata = get_grid_data(expr)
        gdata !== nothing && length(gdata) >= 1 && return real(gdata[1])
        cdata = get_coeff_data(expr)
        cdata !== nothing && length(cdata) >= 1 && return real(cdata[1])
        return 0.0  # uninitialized constant → zero
    end
    return 1.0  # fallback
end

"""Build forcing vector from RHS terms"""
