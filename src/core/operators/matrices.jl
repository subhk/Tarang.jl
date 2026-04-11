"""
    Expression matrices

This file contains the expression_matrices() functions for building
sparse matrices for implicit solvers, along with helper functions
for differentiation matrices and lift matrices.
"""

# LinearAlgebra, SparseArrays already in Tarang.jl

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

    block = div(target_rows, nrows)
    rows = [i * block for i in 1:nrows]
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

    # Case 3: One side depends on vars, other doesn't -> linearity
    left_dep = _depends_on_vars(left, vars)
    right_dep = _depends_on_vars(right, vars)

    if left_dep && !right_dep
        return expression_matrices(left, sp, vars; kwargs...)
    elseif right_dep && !left_dep
        return expression_matrices(right, sp, vars; kwargs...)
    end

    # Both depend on vars (nonlinear) or neither depends -> empty
    return Dict{Any, SparseMatrixCSC}()
end

# ============================================================================
# Subproblem helper functions for compositional operator matrices
# ============================================================================

"""
    _subproblem_kx(sp) -> Float64

Get the Fourier wavenumber kx for a subproblem.
Looks up the Fourier mode index from sp.group_dict and computes kx = nx * 2π/Lx
using the domain length from the first FourierBasis found in problem variables.
"""
function _subproblem_kx(sp)
    # Find the Fourier coordinate name and mode index from group_dict
    fourier_basis = nothing
    nx = nothing
    for var in sp.problem.variables
        if !hasfield(typeof(var), :bases)
            continue
        end
        for (i, basis) in enumerate(var.bases)
            if basis !== nothing && isa(basis, FourierBasis)
                coord_name = basis.meta.element_label
                key = "n" * coord_name
                if haskey(sp.group_dict, key)
                    fourier_basis = basis
                    nx = sp.group_dict[key]
                    break
                end
            end
        end
        if fourier_basis !== nothing
            break
        end
    end

    if fourier_basis === nothing || nx === nothing
        return 0.0
    end

    Lx = fourier_basis.meta.bounds[2] - fourier_basis.meta.bounds[1]
    return nx * 2π / Lx
end

"""
    _subproblem_cheb_basis(sp) -> Union{JacobiBasis, Nothing}

Get the ChebyshevT (or JacobiBasis) from a subproblem's problem variables.
Returns the first JacobiBasis found.
"""
function _subproblem_cheb_basis(sp)
    for var in sp.problem.variables
        if !hasfield(typeof(var), :bases)
            continue
        end
        for basis in var.bases
            if basis !== nothing && isa(basis, JacobiBasis)
                return basis
            end
        end
        # Also check components for VectorField
        if isa(var, VectorField)
            for comp in var.components
                for basis in comp.bases
                    if basis !== nothing && isa(basis, JacobiBasis)
                        return basis
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _subproblem_diff_matrix(sp, coord_name::String, order::Int, Nz::Int) -> SparseMatrixCSC{ComplexF64, Int64}

Get per-subproblem differentiation matrix for a coordinate.
- If coord is Chebyshev (JacobiBasis): returns ComplexF64.(differentiation_matrix(cheb_basis, order))
- If coord is Fourier: returns (im*kx)^order * I_Nz
"""
function _subproblem_diff_matrix(sp, coord_name::String, order::Int, Nz::Int)
    # Check if this coordinate corresponds to a Fourier basis
    for var in sp.problem.variables
        if !hasfield(typeof(var), :bases)
            continue
        end
        for basis in var.bases
            if basis === nothing
                continue
            end
            if basis.meta.element_label == coord_name
                if isa(basis, FourierBasis)
                    # Fourier coordinate: use (im*kx)^order * I
                    kx = _subproblem_kx(sp)
                    coeff = (im * kx)^order
                    return sparse(ComplexF64(coeff) * I, Nz, Nz)
                elseif isa(basis, JacobiBasis)
                    # Chebyshev/Jacobi coordinate: use differentiation matrix
                    D = sparse(ComplexF64.(differentiation_matrix(basis, order)))
                    n_basis = size(D, 1)
                    if n_basis == Nz
                        return D
                    else
                        # Multi-component: block-diagonal kron(I_ncomp, D)
                        n_comp = div(Nz, n_basis)
                        return kron(sparse(ComplexF64(1) * I, n_comp, n_comp), D)
                    end
                end
            end
        end
        # Also check VectorField components
        if isa(var, VectorField)
            for comp in var.components
                for basis in comp.bases
                    if basis === nothing
                        continue
                    end
                    if basis.meta.element_label == coord_name
                        if isa(basis, FourierBasis)
                            kx = _subproblem_kx(sp)
                            coeff = (im * kx)^order
                            return sparse(ComplexF64(coeff) * I, Nz, Nz)
                        elseif isa(basis, JacobiBasis)
                            D = sparse(ComplexF64.(differentiation_matrix(basis, order)))
                            n_basis = size(D, 1)
                            if n_basis == Nz
                                return D
                            else
                                n_comp = div(Nz, n_basis)
                                return kron(sparse(ComplexF64(1) * I, n_comp, n_comp), D)
                            end
                        end
                    end
                end
            end
        end
    end
    # Fallback: zero matrix
    return spzeros(ComplexF64, Nz, Nz)
end

"""
    _get_operand_coordsys(operand) -> Union{CoordinateSystem, Nothing}

Extract coordinate system from an operand (field or operator).
"""
function _get_operand_coordsys(operand)
    # VectorField has coordsys directly
    if isa(operand, VectorField)
        return operand.coordsys
    end
    # Gradient has coordsys
    if hasfield(typeof(operand), :coordsys)
        return operand.coordsys
    end
    # ScalarField: get from dist
    if hasfield(typeof(operand), :dist)
        dist = operand.dist
        if hasfield(typeof(dist), :coordsys)
            return dist.coordsys
        end
    end
    return nothing
end

"""
    _resolve_operand_field(operand) -> Union{ScalarField, VectorField, Nothing}

Walk the operator tree to find the leaf field.
"""
function _resolve_operand_field(operand)
    if isa(operand, ScalarField) || isa(operand, VectorField) || isa(operand, TensorField)
        return operand
    end
    if hasfield(typeof(operand), :operand)
        field = _resolve_operand_field(operand.operand)
        field !== nothing && return field
    end
    for field_name in (:left, :right, :base, :exponent)
        hasfield(typeof(operand), field_name) || continue
        field = _resolve_operand_field(getfield(operand, field_name))
        field !== nothing && return field
    end
    if hasfield(typeof(operand), :operands)
        ops = getfield(operand, :operands)
        if ops !== nothing
            for op in ops
                field = _resolve_operand_field(op)
                field !== nothing && return field
            end
        end
    end
    if isa(operand, Future)
        for op in future_args(operand)
            field = _resolve_operand_field(op)
            field !== nothing && return field
        end
    end
    return nothing
end

function _operand_basis_for_coord(operand, coord_name::String)
    field = _resolve_operand_field(operand)
    field === nothing && return nothing

    if hasfield(typeof(field), :bases)
        for basis in field.bases
            basis === nothing && continue
            label = isa(basis.meta.element_label, Symbol) ? String(basis.meta.element_label) : String(basis.meta.element_label)
            label == coord_name && return basis
        end
    end

    if isa(field, VectorField)
        for comp in field.components
            for basis in comp.bases
                basis === nothing && continue
                label = isa(basis.meta.element_label, Symbol) ? String(basis.meta.element_label) : String(basis.meta.element_label)
                label == coord_name && return basis
            end
        end
    end

    return nothing
end

function _subproblem_group_index(sp, coord_name::String)
    if !(hasfield(typeof(sp), :dist) && sp.dist !== nothing && hasfield(typeof(sp.dist), :coords))
        return nothing
    end

    for (axis, group_entry) in enumerate(sp.group)
        axis > length(sp.dist.coords) && continue
        dist_coord = sp.dist.coords[axis]
        label = isa(dist_coord.name, Symbol) ? String(dist_coord.name) : String(dist_coord.name)
        if label == coord_name
            return group_entry
        end
    end
    return nothing
end

function _integration_step_matrix(basis, coord_name::String, sp, nrows::Int)
    if basis isa FourierBasis
        group_entry = _subproblem_group_index(sp, coord_name)
        if !(group_entry isa Integer)
            return nothing
        elseif group_entry != 0
            # Non-DC mode: Fourier integral is zero. Return zero row(s) to keep
            # the matrix square — valid mode filtering will remove these rows
            # along with the corresponding tau variable columns.
            return spzeros(ComplexF64, 1, nrows)
        end

        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        return sparse(ComplexF64(L) * I, nrows, nrows)
    elseif basis isa JacobiBasis
        Nz = basis.meta.size
        z_min, z_max = basis.meta.bounds[1], basis.meta.bounds[2]
        L = z_max - z_min

        w = zeros(ComplexF64, 1, Nz)
        for n in 0:(Nz-1)
            if n % 2 == 0
                w[1, n+1] = ComplexF64(L / 2.0 * 2.0 / (1.0 - n^2))
            end
        end

        if nrows == 0
            return spzeros(ComplexF64, 0, 0)
        elseif nrows == Nz
            return sparse(w)
        elseif nrows % Nz == 0
            n_comp = div(nrows, Nz)
            return kron(sparse(ComplexF64(1) * I, n_comp, n_comp), sparse(w))
        else
            return nothing
        end
    end

    return nothing
end

# ============================================================================
# subproblem_matrix implementations for linear operators
# ============================================================================

"""
    subproblem_matrix(op::TimeDerivative, sp; kwargs...)

Time derivative: identity matrix. The derivative order is handled by the
timestepping scheme, not the mass matrix.
"""
function subproblem_matrix(op::TimeDerivative, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    return sparse(ComplexF64(1) * I, n, n)
end

"""
    subproblem_matrix(op::Differentiate, sp; kwargs...)

Spatial differentiation: returns the per-subproblem differentiation matrix
for the specified coordinate and order.
"""
function subproblem_matrix(op::Differentiate, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    coord_name = isa(op.coord.name, Symbol) ? String(op.coord.name) : op.coord.name
    return _subproblem_diff_matrix(sp, coord_name, op.order, n)
end

"""
    subproblem_matrix(op::Laplacian, sp; kwargs...)

Laplacian: sum of second derivatives across all coordinates.
For 2D Fourier-Chebyshev: -kx² * I_Nz + D_z²
"""
function subproblem_matrix(op::Laplacian, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    lap_mat = spzeros(ComplexF64, n, n)

    # Sum second derivatives over all coordinates in the distributor
    if hasfield(typeof(sp), :dist) && sp.dist !== nothing && hasfield(typeof(sp.dist), :coords)
        for coord in sp.dist.coords
            coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
            D2 = _subproblem_diff_matrix(sp, coord_name, 2, n)
            lap_mat = lap_mat + D2
        end
    end
    return lap_mat
end

"""
    subproblem_matrix(op::Gradient, sp; kwargs...)

Gradient: stacks per-coordinate differentiation matrices vertically.
For scalar operand: (ndim*Nz × Nz).
For vector operand: (ndim*n_vec × n_vec).
"""
function subproblem_matrix(op::Gradient, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    coordsys = op.coordsys

    blocks = SparseMatrixCSC{ComplexF64, Int64}[]
    for coord in coordsys.coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        D = _subproblem_diff_matrix(sp, coord_name, 1, n)
        push!(blocks, D)
    end

    if isempty(blocks)
        return nothing
    end
    return vcat(blocks...)
end

"""
    subproblem_matrix(op::Divergence, sp; kwargs...)

Divergence: concatenates per-coordinate differentiation matrices horizontally.
Size: (Nz × ndim*Nz) for vector operand.
"""
function subproblem_matrix(op::Divergence, sp; kwargs...)
    # Get coordinate system from operand or from dist
    coordsys = _get_operand_coordsys(op.operand)
    if coordsys === nothing && hasfield(typeof(sp), :dist) && sp.dist !== nothing
        coordsys = sp.dist.coordsys
    end
    if coordsys === nothing
        return nothing
    end

    # For divergence, we need the scalar component size (Nz), not the full vector size
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end

    ndim = coordsys.dim
    field_size = subproblem_field_size(sp, field)
    input_size = try
        _subproblem_expr_dofs(sp, op.operand)
    catch
        field_size
    end

    # Vector divergence: div(u) maps (dim * Nz) -> Nz.
    # Tensor divergence: div(grad(u)) maps (dim * Nu) -> Nu where Nu = dim * Nz.
    if isa(field, VectorField) && input_size == field_size
        block_size = isempty(field.components) ? field_size : subproblem_field_size(sp, field.components[1])
    elseif input_size == ndim * field_size
        block_size = field_size
    elseif isa(field, VectorField) && !isempty(field.components)
        block_size = subproblem_field_size(sp, field.components[1])
    else
        block_size = field_size
    end

    blocks = SparseMatrixCSC{ComplexF64, Int64}[]
    for coord in coordsys.coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        D = _subproblem_diff_matrix(sp, coord_name, 1, block_size)
        push!(blocks, D)
    end

    @debug "Divergence subproblem_matrix: field_size=$field_size, input_size=$input_size, block_size=$block_size, ndim=$ndim, result=$(size(hcat(blocks...)))"

    if isempty(blocks)
        return nothing
    end
    return hcat(blocks...)
end

"""
    subproblem_matrix(op::Trace, sp; kwargs...)

Trace of a tensor: contracts diagonal components.
For 2D: trace_vec = [1, 0, 0, 1] (from ravel(eye(dim))).
Size: (Nz × dim²*Nz).
"""
function subproblem_matrix(op::Trace, sp; kwargs...)
    # Determine dimensionality from the operand or from dist
    coordsys = _get_operand_coordsys(op.operand)
    if coordsys === nothing && hasfield(typeof(sp), :dist) && sp.dist !== nothing
        coordsys = sp.dist.coordsys
    end
    if coordsys === nothing
        return nothing
    end
    dim = coordsys.dim

    # Get scalar block size from the operand
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end

    # For Trace, the operand is typically a tensor or a composed expression
    # that produces dim²*Nz rows. We need the scalar Nz.
    if isa(field, VectorField) && !isempty(field.components)
        Nz = subproblem_field_size(sp, field.components[1])
    elseif isa(field, ScalarField)
        Nz = subproblem_field_size(sp, field)
    else
        # TensorField or other
        Nz_total = subproblem_field_size(sp, field)
        Nz = div(Nz_total, dim * dim)
    end

    # Build trace vector: ravel(eye(dim)) — e.g., [1,0,0,1] for dim=2
    eye_flat = zeros(ComplexF64, dim * dim)
    for i in 1:dim
        eye_flat[(i-1)*dim + i] = ComplexF64(1.0)
    end

    # Trace matrix: kron(transpose(trace_vec), I_Nz)
    # This selects and sums diagonal blocks: (Nz × dim²*Nz)
    trace_vec = sparse(reshape(eye_flat, dim*dim, 1))
    return kron(sparse(transpose(trace_vec)), sparse(ComplexF64(1) * I, Nz, Nz))
end

"""
    subproblem_matrix(op::Lift, sp; kwargs...)

Per-subproblem lift matrix for tau method boundary conditions.

For a tau variable with `n_tau` DOFs per subproblem, the lift places each
tau DOF at a specific Chebyshev mode. The result is a `(n_comp * Nz) × n_tau`
matrix (block-diagonal of Nz×1 lift columns for each component).

The lift mode index follows the convention:
- n >= 0: sets mode n (0-indexed → Julia 1-indexed)
- n < 0: wraps around (n = -1 → last mode, n = -2 → second-to-last)
"""
function subproblem_matrix(op::Lift, sp; kwargs...)
    cheb_basis = _subproblem_cheb_basis(sp)
    if cheb_basis === nothing
        return nothing
    end
    Nz = cheb_basis.meta.size

    # Resolve lift mode index
    lift_mode = op.n
    if lift_mode < 0
        lift_mode = Nz + lift_mode  # -1 → Nz-1 (0-indexed)
    end
    lift_mode += 1  # 0-indexed → 1-indexed

    if lift_mode < 1 || lift_mode > Nz
        @warn "Lift mode $(op.n) resolved to $lift_mode, out of range [1, $Nz]" maxlog=1
        return nothing
    end

    # Build Nz×1 lift column: e_{lift_mode}
    e_lift = spzeros(ComplexF64, Nz, 1)
    e_lift[lift_mode, 1] = ComplexF64(1)

    # Determine number of components in the tau operand
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return e_lift
    end

    n_comp = 1
    if isa(field, VectorField)
        n_comp = length(field.components)
    end

    if n_comp == 1
        return e_lift  # (Nz × 1)
    else
        # Block-diagonal: one lift column per component
        # Result: (n_comp * Nz) × n_comp
        blocks = [e_lift for _ in 1:n_comp]
        return blockdiag(blocks...)
    end
end

# ── Interpolate: evaluation at a point (BC constraints) ──
function subproblem_matrix(op::Interpolate, sp; kwargs...)
    cheb_basis = _subproblem_cheb_basis(sp)
    if cheb_basis === nothing
        return nothing
    end
    coord_name = isa(op.coord.name, Symbol) ? String(op.coord.name) : op.coord.name
    cheb_label = String(cheb_basis.meta.element_label)

    if coord_name != cheb_label
        # Interpolation in Fourier direction: handled by mode selection, just identity
        return nothing
    end

    Nz = cheb_basis.meta.size
    z0 = Float64(op.position)
    z_min, z_max = cheb_basis.meta.bounds[1], cheb_basis.meta.bounds[2]

    # Map physical position to canonical [-1, 1]
    xi = 2.0 * (z0 - z_min) / (z_max - z_min) - 1.0

    # Chebyshev evaluation: T_n(xi) row vector
    T = zeros(ComplexF64, 1, Nz)
    if Nz >= 1; T[1, 1] = 1.0; end
    if Nz >= 2; T[1, 2] = xi; end
    for n in 3:Nz
        T[1, n] = 2.0 * xi * T[1, n-1] - T[1, n-2]
    end

    # For vector operands, apply to each component: kron(I_ncomp, T_row)
    field = _resolve_operand_field(op.operand)
    n_comp = 1
    if field !== nothing && isa(field, VectorField)
        n_comp = length(field.components)
    end
    if n_comp > 1
        return kron(sparse(ComplexF64(1)*I, n_comp, n_comp), sparse(T))
    end
    return sparse(T)  # (1 × Nz) for scalar
end

# ── Integrate: integration over a coordinate (constraints like integ(p)=0) ──
function subproblem_matrix(op::Integrate, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    field === nothing && return nothing
    n = subproblem_field_size(sp, field)
    coords = isa(op.coord, Tuple) ? collect(op.coord) : [op.coord]

    # For multi-coordinate integration (integ over full domain):
    # Check if any separable (Fourier) coordinate gives zero for this mode.
    # If so, the entire integral is zero — return a zero row.
    for coord in coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        basis = _operand_basis_for_coord(op.operand, coord_name)
        if basis !== nothing && basis isa FourierBasis
            group_entry = _subproblem_group_index(sp, coord_name)
            if group_entry isa Integer && group_entry != 0
                # Non-DC Fourier mode: integral over x is zero
                # Return a zero row to keep the system square; valid mode
                # filtering will detect and remove it.
                return spzeros(ComplexF64, 1, n)
            end
        end
    end

    # All Fourier coordinates are DC (or none are Fourier) — compute integration weights.
    # For Chebyshev direction, build the integration weight row vector.
    cheb_basis = nothing
    for coord in coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        basis = _operand_basis_for_coord(op.operand, coord_name)
        if basis !== nothing && basis isa JacobiBasis
            cheb_basis = basis
            break
        end
    end

    if cheb_basis === nothing
        return sparse(ComplexF64(1) * I, 1, n)  # No Chebyshev: scalar identity
    end

    Nz = cheb_basis.meta.size
    z_min, z_max = cheb_basis.meta.bounds[1], cheb_basis.meta.bounds[2]
    L = z_max - z_min

    # Chebyshev integration weights: ∫_{-1}^{1} T_n(x) dx = 2/(1-n²) for even n
    w = zeros(ComplexF64, 1, Nz)
    for k in 0:(Nz-1)
        if k % 2 == 0
            w[1, k+1] = ComplexF64(L / 2.0 * 2.0 / (1.0 - k^2))
        end
    end

    # Scale by Fourier domain length for DC mode (x-integration gives Lx)
    for coord in coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        basis = _operand_basis_for_coord(op.operand, coord_name)
        if basis !== nothing && basis isa FourierBasis
            Lx = basis.meta.bounds[2] - basis.meta.bounds[1]
            w .*= Lx
        end
    end

    # For vector operands
    n_comp = 1
    if field !== nothing && isa(field, VectorField)
        n_comp = length(field.components)
    end
    if n_comp > 1
        return kron(sparse(ComplexF64(1)*I, n_comp, n_comp), sparse(w))
    end
    return sparse(w)  # (1 × Nz)
end

# ============================================================================
# expression_matrices for operators not yet migrated to subproblem_matrix
# ============================================================================

"""
    expression_matrices(op::Convert, sp, vars; kwargs...)

Basis conversion matrices.
Following operators Convert.expression_matrices.
"""
function expression_matrices(op::Convert, sp, vars; kwargs...)
    operand = op.operand
    out_basis = op.basis
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Get input basis and build conversion matrix
            if hasfield(typeof(var), :bases) && !isempty(var.bases)
                in_basis = var.bases[1]
                if in_basis !== nothing && isa(in_basis, JacobiBasis) && isa(out_basis, JacobiBasis)
                    conv_mat = conversion_matrix(in_basis, out_basis)
                    result[var] = conv_mat
                end
            end
        end
    end

    return result
end

# ============================================================================
# Helper Functions for Building Operator Matrices
# ============================================================================

"""
    build_operator_differentiation_matrix(var, coord, order; kwargs...)

Build differentiation matrix for variable with respect to coordinate.
"""
function build_operator_differentiation_matrix(var, coord::Coordinate, order::Int; kwargs...)
    if !hasfield(typeof(var), :bases)
        return nothing
    end

    # Find the basis corresponding to this coordinate
    basis_idx = nothing
    target_basis = nothing

    for (i, basis) in enumerate(var.bases)
        if basis !== nothing && basis.meta.element_label == coord.name
            basis_idx = i
            target_basis = basis
            break
        end
    end

    if target_basis === nothing
        return nothing
    end

    n_total = field_dofs(var)
    n_basis = target_basis.meta.size

    # Build 1D differentiation matrix based on basis type
    D1d = nothing

    if isa(target_basis, JacobiBasis)
        D1d = differentiation_matrix(target_basis, order)
    elseif isa(target_basis, FourierBasis)
        D1d = fourier_differentiation_matrix(target_basis, order)
    end

    if D1d === nothing
        return nothing
    end

    # For multi-dimensional fields, apply Kronecker product
    if length(var.bases) == 1
        return D1d
    else
        # Build identity matrices for other dimensions
        matrices = AbstractMatrix[]
        for (i, basis) in enumerate(var.bases)
            if basis === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, D1d)
            else
                push!(matrices, sparse(I, basis.meta.size, basis.meta.size))
            end
        end

        # Kronecker product in reverse order for Julia's column-major (Fortran) layout:
        # For bases (B1, B2, ..., Bn), the multi-dimensional operator is
        #   M_n ⊗ ... ⊗ M_2 ⊗ M_1
        # which acts on vectorized data stored in column-major order where
        # the first axis varies fastest.
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end

        return result
    end
end

"""
    fourier_differentiation_matrix(basis::FourierBasis, order::Int)

Build Fourier differentiation matrix.
"""
function fourier_differentiation_matrix(basis::RealFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # RealFourier differentiation follows spectral convention:
    # Modes: [cos(0x), cos(1x), -sin(1x), cos(2x), -sin(2x), ...]
    # Note: Using -sin (msin) convention
    #
    # For differentiation with (ik)^order factor:
    # d/dx cos(kx) = -k sin(kx) = k * (-sin(kx))  -> k * msin
    # d/dx (-sin(kx)) = -k cos(kx)                -> -k * cos
    #
    # The 2x2 block for each wavenumber k is:
    # | 0  -k |   (maps: cos <- -k*msin, msin <- k*cos)
    # | k   0 |
    #
    # For order n, we apply this matrix n times, or equivalently compute (ik)^n
    # and extract the real/imaginary parts for the rotation.

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # DC mode (k=0) derivative is zero — implicitly zero in sparse matrix

    k_max = (N - 1) ÷ 2

    for k in 1:k_max
        cos_idx = 2*k      # 1-indexed: mode 2 is cos(1x), mode 4 is cos(2x), etc.
        sin_idx = 2*k + 1  # 1-indexed: mode 3 is -sin(1x), mode 5 is -sin(2x), etc.
        k_phys = k0 * k

        if cos_idx <= N && sin_idx <= N
            # Compute (ik)^order = k^order * i^order
            # i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i, i^4 = 1, ...
            # For real representation: (ik)^n = k^n * (cos(npi/2) + i*sin(npi/2))
            #
            # The 2x2 derivative block D^n for the (cos, msin) pair:
            # D^1 = | 0  -k |    D^2 = |-k^2  0  |    D^3 = | 0   k^3 |   D^4 = |k^4  0 |
            #       | k   0 |          | 0  -k^2 |          |-k^3  0  |         | 0  k^4|
            #
            # Pattern: D^n has form k^n * |cos(npi/2)  -sin(npi/2)|
            #                             |sin(npi/2)   cos(npi/2)|

            kn = k_phys^order
            phase = order * pi / 2
            c = cos(phase)
            s = sin(phase)

            # Matrix entries for the 2x2 block:
            # cos_out = c * k^n * cos_in - s * k^n * msin_in
            # msin_out = s * k^n * cos_in + c * k^n * msin_in

            # (cos_idx, cos_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, cos_idx); push!(V_list, c * kn)
            end
            # (cos_idx, sin_idx): -s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, sin_idx); push!(V_list, -s * kn)
            end
            # (sin_idx, cos_idx): s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, cos_idx); push!(V_list, s * kn)
            end
            # (sin_idx, sin_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, sin_idx); push!(V_list, c * kn)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

function fourier_differentiation_matrix(basis::ComplexFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # ComplexFourier: diagonal matrix with (ik)^order (works for both even and odd N)
    k_native = [k <= N÷2 ? k : k - N for k in 0:N-1]
    k_phys = k0 .* k_native

    diag_vals = (im .* k_phys).^order

    return spdiagm(0 => diag_vals)
end

"""
    build_lift_matrix(var, basis, n; kwargs...)

Build lifting matrix for tau method boundary conditions.
Following the standard basis LiftJacobi implementation (lines 790-814).

the standard convention:
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)

Example: For N=10 modes
- Lift(tau, basis, 0) -> sets mode 1 (Julia 1-indexed)
- Lift(tau, basis, -1) -> sets mode N (last mode)
- Lift(tau, basis, -2) -> sets mode N-1 (second-to-last mode)

Following spectral methods pattern):
The matrix places the tau variable's coefficient at mode n in the solution.
For LBVP solvers, this creates the "tau polynomial" that adds boundary
condition enforcement terms to the highest modes.
"""
function build_lift_matrix(var, basis, n::Int; kwargs...)
    N = basis.meta.size

    # Resolve mode index: negative wrap-around (spectral convention)
    # n < 0: index from end (e.g., -1 -> N-1 in 0-indexed -> N in 1-indexed)
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode
    end
    lift_mode += 1  # Convert 0-indexed to 1-indexed Julia convention

    if lift_mode < 1 || lift_mode > N
        @warn "Lift mode $n (resolved to $lift_mode) out of range [1, $N] for basis $(basis.meta.element_label)"
        tau_dofs = max(1, field_dofs(var))
        full_dofs = N * tau_dofs  # Full field DOFs = lift dimension × tau DOFs
        return spzeros(Float64, full_dofs, tau_dofs)
    end

    # Build the 1D lift column vector: e_{lift_mode} of size (N, 1)
    e_lift = sparse([lift_mode], [1], [1.0], N, 1)

    # If var has no bases or only the lift basis, return 1D lift vector
    if !hasfield(typeof(var), :bases) || isempty(var.bases) || all(b -> b === nothing, var.bases)
        return e_lift
    end

    # Find which var basis (if any) matches the lift basis coordinate
    lift_coord = basis.meta.element_label
    basis_idx = nothing
    for (i, b) in enumerate(var.bases)
        if b !== nothing && b.meta.element_label == lift_coord
            basis_idx = i
            break
        end
    end

    # Multi-dimensional case: build Kronecker product
    if basis_idx !== nothing
        # Tau variable already has the lift basis - rare case
        var_basis_size = var.bases[basis_idx].meta.size
        lift_1d = sparse([lift_mode], [lift_mode], [1.0], N, var_basis_size)

        if length(var.bases) == 1
            return lift_1d
        end

        # Kronecker product for multi-dimensional lift operator.
        # Reverse iteration matches Julia's column-major layout: M_n ⊗ ... ⊗ M_1
        # where axis 1 varies fastest in the vectorized representation.
        matrices = AbstractMatrix[]
        for (i, b) in enumerate(var.bases)
            if b === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, lift_1d)
            else
                push!(matrices, sparse(LinearAlgebra.I, b.meta.size, b.meta.size))
            end
        end

        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end
        return result
    else
        # Tau variable does NOT have the lift basis - standard tau method case.
        # Build Kronecker factors in coordinate order, placing e_lift at the
        # correct position (not always last) based on the coordinate system.
        coordsys = basis.meta.coordsys

        # Map tau variable bases by coordinate name for lookup
        var_basis_map = Dict{String, Basis}()
        for b in var.bases
            if b !== nothing
                var_basis_map[b.meta.element_label] = b
            end
        end

        if isempty(var_basis_map)
            return e_lift
        end

        # Build matrices in coordinate order: e_lift at the lift dimension,
        # identity matrices for the tau variable's tangential dimensions
        matrices = AbstractMatrix[]
        for coord in coordsys.coords
            if coord.name == lift_coord
                push!(matrices, e_lift)
            elseif haskey(var_basis_map, coord.name)
                b = var_basis_map[coord.name]
                push!(matrices, sparse(LinearAlgebra.I, b.meta.size, b.meta.size))
            end
        end

        if isempty(matrices)
            return e_lift
        end

        if length(matrices) == 1
            return matrices[1]
        end

        # Reverse Kronecker product for column-major layout (see note above)
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end
        return result
    end
end

"""
    get_coord_for_basis(basis::Basis)

Get coordinate associated with a basis.
"""
function get_coord_for_basis(basis::Basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :coordsys)
        coordsys = basis.meta.coordsys
        coord_name = basis.meta.element_label
        if hasfield(typeof(coordsys), :coords)
            for coord in coordsys.coords
                if coord.name == coord_name
                    return coord
                end
            end
        end
    end
    return nothing
end

"""
    field_dofs(field)

Get total degrees of freedom for a field.
"""
function field_dofs(field)
    if hasfield(typeof(field), :buffers) && get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif hasfield(typeof(field), :buffers) && get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    elseif hasfield(typeof(field), :bases)
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    elseif hasfield(typeof(field), :components)
        # VectorField or TensorField
        return sum(field_dofs(comp) for comp in field.components)
    end
    return 0
end
