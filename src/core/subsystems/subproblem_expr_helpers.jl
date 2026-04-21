function _coord_name(coord)
    isa(coord, Tuple) && !isempty(coord) && return _coord_name(coord[1])
    return isa(coord.name, Symbol) ? String(coord.name) : String(coord.name)
end

function _subproblem_reduce_dofs(sp::Subproblem, inner::Int, operand, coord; interpolate::Bool=false)
    coord_name = _coord_name(coord)
    basis = _operand_basis_for_coord(operand, coord_name)
    basis === nothing && return inner

    if basis isa FourierBasis
        if interpolate
            return inner
        end
        group_entry = _subproblem_group_index(sp, coord_name)
        return (group_entry isa Integer && group_entry == 0) ? inner : 0
    end

    bsize = _basis_coeff_size(basis)
    if bsize <= 0 || inner == 0
        return inner
    end
    return inner % bsize == 0 ? div(inner, bsize) : inner
end

function _subproblem_expr_dofs(sp::Subproblem, expr)
    expr === nothing && return 0
    (isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator)) && return 0
    isa(expr, ScalarField) && return subproblem_field_size(sp, expr)
    isa(expr, VectorField) && return subproblem_field_size(sp, expr)
    isa(expr, TensorField) && return subproblem_field_size(sp, expr)

    if isa(expr, AddOperator) || isa(expr, SubtractOperator)
        return max(_subproblem_expr_dofs(sp, expr.left), _subproblem_expr_dofs(sp, expr.right))
    end
    if isa(expr, MultiplyOperator) || isa(expr, DivideOperator)
        return max(_subproblem_expr_dofs(sp, expr.left), _subproblem_expr_dofs(sp, expr.right))
    end
    isa(expr, NegateOperator) && return _subproblem_expr_dofs(sp, expr.operand)

    if isa(expr, Future)
        args = future_args(expr)
        return isempty(args) ? 0 : maximum(_subproblem_expr_dofs(sp, arg) for arg in args)
    end

    if isa(expr, Integrate) || isa(expr, Average)
        inner = _subproblem_expr_dofs(sp, expr.operand)
        coords = isa(expr.coord, Tuple) ? expr.coord : (expr.coord,)
        for coord in coords
            inner = _subproblem_reduce_dofs(sp, inner, expr.operand, coord; interpolate=false)
        end
        return inner
    end

    if isa(expr, Interpolate)
        return _subproblem_reduce_dofs(sp, _subproblem_expr_dofs(sp, expr.operand),
                                       expr.operand, expr.coord; interpolate=true)
    end

    if isa(expr, Trace)
        field = _resolve_operand_field(expr.operand)
        if isa(field, VectorField) && !isempty(field.components)
            return subproblem_field_size(sp, field.components[1])
        elseif isa(field, ScalarField)
            return subproblem_field_size(sp, field)
        end

        inner = _subproblem_expr_dofs(sp, expr.operand)
        ndim = _infer_ndim(expr.operand)
        if ndim > 0 && inner % (ndim * ndim) == 0
            return div(inner, ndim * ndim)
        elseif ndim > 0 && inner % ndim == 0
            return div(inner, ndim)
        end
        return inner
    end

    if hasfield(typeof(expr), :operand)
        mat = subproblem_matrix(expr, sp)
        if mat !== nothing
            return size(mat, 1)
        end
        return _subproblem_expr_dofs(sp, getfield(expr, :operand))
    end

    field = _resolve_operand_field(expr)
    field !== nothing && return subproblem_field_size(sp, field)
    return 0
end

function _has_only_zero_dim_bases(field::ScalarField)
    isempty(field.bases) && return true
    return all(b -> b === nothing, field.bases)
end

function _has_only_zero_dim_bases(field::VectorField)
    return all(_has_only_zero_dim_bases, field.components)
end

function _has_only_zero_dim_bases(field::TensorField)
    return all(_has_only_zero_dim_bases, vec(field.components))
end

function _is_zero_separable_group(sp::Subproblem)
    for group_entry in sp.group
        if group_entry isa Integer && group_entry != 0
            return false
        end
    end
    return true
end

"""
    get_valid_modes(eq_data, sp, num_modes)

Get valid modes array for equation.
"""
function get_valid_modes(eq_data::Dict, sp::Subproblem, num_modes::Int)
    valid = get(eq_data, "valid_modes", nothing)
    if valid === nothing
        return ones(Bool, num_modes)
    end
    return vec(valid)
end

"""
    get_valid_modes_var(var, sp)

Get valid modes array for variable.
"""
function get_valid_modes_var(var, sp::Subproblem)
    local_size = subproblem_field_size(sp, var)
    if local_size > 0 && _has_only_zero_dim_bases(var) && !_is_zero_separable_group(sp)
        return zeros(Bool, local_size)
    end
    if hasfield(typeof(var), :valid_modes) && var.valid_modes !== nothing
        valid = vec(var.valid_modes)
        if length(valid) == local_size
            return valid
        end
    end
    return ones(Bool, local_size)
end

"""
    expression_matrices(expr, sp, vars; kwargs...)

Build expression matrices for each variable.
"""
# Fallbacks for non-Operator expression types passed to expression_matrices.
# NOTE: Do NOT define a catch-all `expression_matrices(expr, sp::Subproblem, ...)` here —
# it would create a method ambiguity with `expression_matrices(op::Operator, sp, ...)`
# in matrices.jl, causing Julia MethodError for Operator+Subproblem calls.
expression_matrices(::Nothing, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(::Number, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(expr::Future, sp::Subproblem, vars; kwargs...) =
    _expression_matrices_future(expr, sp, vars; kwargs...)
function expression_matrices(field::ScalarField, sp::Subproblem, vars; kwargs...)
    if _field_in_vars(field, vars)
        n = subproblem_field_size(sp, field)
        return Dict{Any, SparseMatrixCSC}(field => sparse(ComplexF64(1) * I, n, n))
    end

    # Check if this ScalarField is a component of a VectorField in vars
    parent_info = _find_parent_vector(field, vars)
    if parent_info !== nothing
        parent, comp_idx = parent_info
        n_parent = subproblem_field_size(sp, parent)
        n_comp = length(parent.components)
        comp_size = div(n_parent, n_comp)
        # Build selector: (comp_size × n_parent) with identity block at comp_idx
        rows = collect(1:comp_size)
        cols = collect((comp_idx - 1) * comp_size + 1 : comp_idx * comp_size)
        vals = ones(ComplexF64, comp_size)
        selector = sparse(rows, cols, vals, comp_size, n_parent)
        return Dict{Any, SparseMatrixCSC}(parent => selector)
    end

    return Dict{Any, SparseMatrixCSC}()
end

function expression_matrices(field::VectorField, sp::Subproblem, vars; kwargs...)
    if _field_in_vars(field, vars)
        n = subproblem_field_size(sp, field)
        return Dict{Any, SparseMatrixCSC}(field => sparse(ComplexF64(1) * I, n, n))
    end
    return Dict{Any, SparseMatrixCSC}()
end

"""
    _field_in_vars(field, vars) -> Bool

Check if `field` is in `vars` by object identity or name matching.
"""
function _field_in_vars(field, vars)
    for v in vars
        v === field && return true
        if hasfield(typeof(v), :name) && hasfield(typeof(field), :name) && v.name == field.name
            return true
        end
    end
    return false
end

"""
    _find_parent_vector(field::ScalarField, vars) -> (parent::VectorField, comp_idx::Int) or nothing

Find the parent VectorField in `vars` that contains this ScalarField as a component.
Returns (parent, component_index) or nothing if no parent found.
"""
function _find_parent_vector(field::ScalarField, vars)
    for v in vars
        if isa(v, VectorField)
            for (ci, comp) in enumerate(v.components)
                if comp === field || (hasfield(typeof(comp), :name) && comp.name == field.name)
                    return (v, ci)
                end
            end
        end
        if isa(v, TensorField)
            for (ci, comp) in enumerate(v.components)
                if comp === field || (hasfield(typeof(comp), :name) && comp.name == field.name)
                    return (v, ci)
                end
            end
        end
    end
    return nothing
end
expression_matrices(::String, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()
expression_matrices(::Symbol, sp::Subproblem, vars; kwargs...) = Dict{Any, SparseMatrixCSC}()

# Note: is_zero_expression is defined in problems.jl
# Add additional methods here for type dispatch
is_zero_expression(::Nothing) = true
is_zero_expression(x::Number) = x == 0
