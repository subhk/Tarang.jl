# ═════════════════════════════════════════════════════════════════════════════
# Lazy RHS evaluation — type-specialized Future expression trees
# ═════════════════════════════════════════════════════════════════════════════
#
# Builds a type-parametric expression tree from equation RHS expressions, then
# evaluates it via multiple-dispatch on concrete LazyFuture subtypes. The JIT
# specializes the entire evaluate! call chain at first use, eliminating dynamic
# dispatch overhead.
#
# Workspace design: holds a pool of pre-allocated ScalarField objects. Each
# binary operation (Add, Mul, etc.) borrows two scratch fields, evaluates its
# children into them, and combines them into the output. Differentiation uses
# the ScalarField's transform machinery (ensure_layout :c → apply D → :g).
#
# If translation fails (unsupported operator), the caller falls back to the
# compiled_rhs path, then interpreted evaluation.

# ── Abstract type hierarchy ──────────────────────────────────────────────────

"""Abstract lazy future — a deferred operation that produces a field value."""
abstract type LazyFuture end

# ── Leaf types ───────────────────────────────────────────────────────────────

"""Reference to a state field by index in solver.state."""
struct LazyStateField <: LazyFuture
    idx::Int
end

"""Reference to a parameter / non-state field."""
struct LazyParamField{F} <: LazyFuture
    field::F
end

"""Scalar numeric constant."""
struct LazyConst <: LazyFuture
    value::Float64
end

# ── Arithmetic combinators ───────────────────────────────────────────────────

struct LazyAdd{L<:LazyFuture, R<:LazyFuture} <: LazyFuture
    left::L
    right::R
end

struct LazySub{L<:LazyFuture, R<:LazyFuture} <: LazyFuture
    left::L
    right::R
end

struct LazyMul{L<:LazyFuture, R<:LazyFuture} <: LazyFuture
    left::L
    right::R
end

struct LazyNegate{T<:LazyFuture} <: LazyFuture
    operand::T
end

struct LazyScale{T<:LazyFuture} <: LazyFuture
    operand::T
    coeff::Float64
end

# ── Differentiation ──────────────────────────────────────────────────────────

struct LazyDiff{T<:LazyFuture} <: LazyFuture
    operand::T
    coord::Coordinate
    order::Int
end

# ── Workspace ────────────────────────────────────────────────────────────────

"""
    LazyWorkspace

Pool of pre-allocated ScalarField objects used as scratch buffers during
evaluation. Each `evaluate!` borrows fields from the pool (round-robin) and
returns them via `ws.next_idx` rewinding between sibling calls.
"""
mutable struct LazyWorkspace
    fields::Vector{ScalarField}
    next_idx::Int     # index of next free field (grows as needed)
    template::Union{Nothing, ScalarField}
end

LazyWorkspace() = LazyWorkspace(ScalarField[], 1, nothing)

"""Get a scratch ScalarField from the pool, allocating if needed."""
function _borrow_scratch!(ws::LazyWorkspace)
    if ws.next_idx > length(ws.fields)
        # Allocate a new scratch field matching the template
        template = ws.template
        if template === nothing
            throw(ArgumentError("LazyWorkspace has no template — cannot allocate scratch"))
        end
        new_field = ScalarField(template.dist,
                                "_lazy_scratch_$(ws.next_idx)",
                                template.bases,
                                template.dtype)
        push!(ws.fields, new_field)
    end
    field = ws.fields[ws.next_idx]
    ws.next_idx += 1
    return field
end

"""Save/restore workspace position around a subtree evaluation."""
@inline function _with_scratch_scope(f, ws::LazyWorkspace)
    saved = ws.next_idx
    try
        return f()
    finally
        ws.next_idx = saved
    end
end

# ── Translation from existing expression trees ──────────────────────────────

"""
    translate_to_lazy(expr, state; target=nothing) -> LazyFuture or nothing

Recursively translate an existing operator expression tree into a LazyFuture.
Returns `nothing` if any sub-expression contains an unsupported operator type.

`target` is the per-component target field (e.g. `u_x` for the u_x equation).
When set, VectorField references are resolved to the matching component —
this handles vector advection `u⋅∇(u)` which expands to `Differentiate(u, x)`
with a VectorField operand.
"""
function translate_to_lazy(expr, state; target=nothing)
    # Leaf: state field
    if isa(expr, ScalarField)
        idx = _lazy_find_state_index(expr, state)
        if idx !== nothing
            return LazyStateField(idx)
        end
        return LazyParamField(expr)
    end

    # VectorField → resolve to matching component when a target is given
    if isa(expr, VectorField)
        if target !== nothing
            comp = _vector_component_for_target(expr, target)
            if comp !== nothing
                return translate_to_lazy(comp, state; target=target)
            end
        end
        return LazyParamField(expr)
    end

    # Leaf: numbers / constants
    if isa(expr, Number)
        return LazyConst(Float64(real(expr)))
    end
    if isa(expr, ConstantOperator)
        return LazyConst(Float64(expr.value))
    end
    if isa(expr, ZeroOperator)
        return LazyConst(0.0)
    end

    # Binary ops
    if isa(expr, AddOperator)
        l = translate_to_lazy(expr.left, state; target=target)
        r = translate_to_lazy(expr.right, state; target=target)
        (l === nothing || r === nothing) && return nothing
        return LazyAdd(l, r)
    end
    if isa(expr, SubtractOperator)
        l = translate_to_lazy(expr.left, state; target=target)
        r = translate_to_lazy(expr.right, state; target=target)
        (l === nothing || r === nothing) && return nothing
        return LazySub(l, r)
    end
    if isa(expr, MultiplyOperator)
        l = translate_to_lazy(expr.left, state; target=target)
        r = translate_to_lazy(expr.right, state; target=target)
        (l === nothing || r === nothing) && return nothing
        if isa(l, LazyConst)
            return LazyScale(r, l.value)
        elseif isa(r, LazyConst)
            return LazyScale(l, r.value)
        end
        return LazyMul(l, r)
    end

    # Unary ops
    if isa(expr, NegateOperator)
        op = translate_to_lazy(expr.operand, state; target=target)
        op === nothing && return nothing
        return LazyNegate(op)
    end

    # Differentiation
    if isa(expr, Differentiate)
        op_expr = expr.operand
        # Resolve VectorField operand to target component
        if isa(op_expr, VectorField) && target !== nothing
            comp = _vector_component_for_target(op_expr, target)
            if comp !== nothing
                op_expr = comp
            end
        end
        op = translate_to_lazy(op_expr, state; target=target)
        op === nothing && return nothing
        return LazyDiff(op, expr.coord, expr.order)
    end

    # Future hierarchy
    if isa(expr, Future)
        return _translate_future_to_lazy(expr, state; target=target)
    end

    return nothing
end

"""Find the VectorField component matching `target` by identity or name."""
function _vector_component_for_target(vf::VectorField, target::ScalarField)
    for comp in vf.components
        if comp === target
            return comp
        end
        if hasfield(typeof(comp), :name) && comp.name == target.name
            return comp
        end
    end
    # Fall back to name-suffix matching (_x, _y, _z)
    target_name = String(target.name)
    for comp in vf.components
        cname = String(comp.name)
        if endswith(target_name, "_x") && endswith(cname, "_x")
            return comp
        elseif endswith(target_name, "_y") && endswith(cname, "_y")
            return comp
        elseif endswith(target_name, "_z") && endswith(cname, "_z")
            return comp
        end
    end
    return nothing
end

function _lazy_find_state_index(field::ScalarField, state)
    for (i, s) in enumerate(state)
        if s === field
            return i
        end
        if s.name == field.name
            return i
        end
    end
    return nothing
end

function _translate_future_to_lazy(expr::Future, state; target=nothing)
    args = future_args(expr)
    isempty(args) && return LazyConst(0.0)

    if isa(expr, Add)
        result = translate_to_lazy(args[1], state; target=target)
        result === nothing && return nothing
        for a in args[2:end]
            rhs = translate_to_lazy(a, state; target=target)
            rhs === nothing && return nothing
            result = LazyAdd(result, rhs)
        end
        return result
    end

    if isa(expr, Subtract)
        length(args) < 2 && return nothing
        result = translate_to_lazy(args[1], state; target=target)
        result === nothing && return nothing
        for a in args[2:end]
            rhs = translate_to_lazy(a, state; target=target)
            rhs === nothing && return nothing
            result = LazySub(result, rhs)
        end
        return result
    end

    if isa(expr, Negate)
        length(args) < 1 && return nothing
        op = translate_to_lazy(args[1], state; target=target)
        op === nothing && return nothing
        return LazyNegate(op)
    end

    if isa(expr, Multiply)
        scalar_coeff = 1.0
        deps = Any[]
        for a in args
            if isa(a, Number)
                scalar_coeff *= Float64(real(a))
            elseif isa(a, ConstantOperator)
                scalar_coeff *= Float64(a.value)
            else
                push!(deps, a)
            end
        end

        isempty(deps) && return LazyConst(scalar_coeff)

        result = translate_to_lazy(deps[1], state; target=target)
        result === nothing && return nothing
        for a in deps[2:end]
            rhs = translate_to_lazy(a, state; target=target)
            rhs === nothing && return nothing
            result = LazyMul(result, rhs)
        end

        if scalar_coeff != 1.0
            result = LazyScale(result, scalar_coeff)
        end
        return result
    end

    return nothing
end

# ── Evaluation — writes into a ScalarField output ───────────────────────────

"""
    evaluate_lazy!(out::ScalarField, expr, state, ws::LazyWorkspace)

Compute `expr` and store the result in `out` (in grid layout).
Type-specialized via multiple dispatch; the JIT inlines the whole chain.
"""
@inline function evaluate_lazy!(out::ScalarField, expr::LazyStateField, state, ws::LazyWorkspace)
    src = state[expr.idx]
    ensure_layout!(src, :g)
    src_data = get_local_data(get_grid_data(src))
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    if src_data !== nothing && out_data !== nothing && size(src_data) == size(out_data)
        copyto!(out_data, src_data)
    elseif out_data !== nothing
        fill!(out_data, zero(eltype(out_data)))
    end
    out.current_layout = :g
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazyParamField, state, ws::LazyWorkspace)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    out_data === nothing && return out
    if isa(expr.field, ScalarField)
        ensure_layout!(expr.field, :g)
        src_data = get_local_data(get_grid_data(expr.field))
        if src_data !== nothing && size(src_data) == size(out_data)
            copyto!(out_data, src_data)
        else
            fill!(out_data, zero(eltype(out_data)))
        end
    else
        fill!(out_data, zero(eltype(out_data)))
    end
    out.current_layout = :g
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazyConst, state, ws::LazyWorkspace)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    if out_data !== nothing
        fill!(out_data, eltype(out_data)(expr.value))
    end
    out.current_layout = :g
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazyAdd, state, ws::LazyWorkspace)
    _with_scratch_scope(ws) do
        a = _borrow_scratch!(ws)
        b = _borrow_scratch!(ws)
        evaluate_lazy!(a, expr.left, state, ws)
        evaluate_lazy!(b, expr.right, state, ws)
        _fused_binary!(out, a, b, +)
    end
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazySub, state, ws::LazyWorkspace)
    _with_scratch_scope(ws) do
        a = _borrow_scratch!(ws)
        b = _borrow_scratch!(ws)
        evaluate_lazy!(a, expr.left, state, ws)
        evaluate_lazy!(b, expr.right, state, ws)
        _fused_binary!(out, a, b, -)
    end
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazyMul, state, ws::LazyWorkspace)
    _with_scratch_scope(ws) do
        a = _borrow_scratch!(ws)
        b = _borrow_scratch!(ws)
        evaluate_lazy!(a, expr.left, state, ws)
        evaluate_lazy!(b, expr.right, state, ws)
        _fused_binary!(out, a, b, *)
    end
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazyNegate, state, ws::LazyWorkspace)
    evaluate_lazy!(out, expr.operand, state, ws)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    if out_data !== nothing
        @. out_data = -out_data
    end
    out.current_layout = :g
    return out
end

@inline function evaluate_lazy!(out::ScalarField, expr::LazyScale, state, ws::LazyWorkspace)
    evaluate_lazy!(out, expr.operand, state, ws)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    coeff = eltype(out_data) <: Real ? real(expr.coeff) : expr.coeff
    if out_data !== nothing
        @. out_data = coeff * out_data
    end
    out.current_layout = :g
    return out
end

function evaluate_lazy!(out::ScalarField, expr::LazyDiff, state, ws::LazyWorkspace)
    # Evaluate operand into out, then apply differentiation in-place
    evaluate_lazy!(out, expr.operand, state, ws)
    _apply_lazy_diff!(out, expr.coord, expr.order)
    return out
end

function evaluate_lazy!(out::ScalarField, expr::LazyFuture, state, ws::LazyWorkspace)
    throw(ArgumentError("LazyRHS: no evaluate_lazy! method for $(typeof(expr))"))
end

# ── Fused binary op: writes op.(a_data, b_data) into out_data ───────────────

@inline function _fused_binary!(out::ScalarField, a::ScalarField, b::ScalarField, op)
    ensure_layout!(out, :g)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    out_data = get_local_data(get_grid_data(out))
    a_data = get_local_data(get_grid_data(a))
    b_data = get_local_data(get_grid_data(b))
    if out_data !== nothing && a_data !== nothing && b_data !== nothing
        @. out_data = op(a_data, b_data)
    elseif out_data !== nothing
        fill!(out_data, zero(eltype(out_data)))
    end
    out.current_layout = :g
    return out
end

# ── Differentiation using the field's transform machinery ───────────────────

function _apply_lazy_diff!(field::ScalarField, coord::Coordinate, order::Int)
    coord_name = isa(coord.name, Symbol) ? String(coord.name) : String(coord.name)

    # Find the axis matching this coordinate
    axis = nothing
    target_basis = nothing
    for (i, basis) in enumerate(field.bases)
        if basis !== nothing && String(basis.meta.element_label) == coord_name
            axis = i
            target_basis = basis
            break
        end
    end

    if target_basis === nothing || axis === nothing
        return field  # Not in this field's bases — skip
    end

    # Switch to coefficient space
    ensure_layout!(field, :c)
    coeff_data = get_local_data(get_coeff_data(field))
    coeff_data === nothing && return field

    # Get the 1D differentiation matrix
    D = if isa(target_basis, JacobiBasis)
        differentiation_matrix(target_basis, order)
    elseif isa(target_basis, FourierBasis)
        fourier_differentiation_matrix(target_basis, order)
    else
        return field
    end

    # Apply D along the target axis
    _apply_1d_matrix!(coeff_data, D, axis)

    field.current_layout = :c
    # Transform back to grid space for downstream operations
    ensure_layout!(field, :g)
    return field
end

"""Apply a 1D matrix `D` along `axis` of multi-dimensional array `data` in place."""
function _apply_1d_matrix!(data::AbstractArray, D::AbstractMatrix, axis::Int)
    if ndims(data) == 1
        data .= D * data
        return data
    end

    if ndims(data) == 2
        if axis == 1
            # result[i,j] = Σ_k D[i,k] * data[k,j]  →  D * data
            tmp = D * data
            copyto!(data, tmp)
        elseif axis == 2
            # result[i,j] = Σ_k data[i,k] * D[j,k]  →  data * D'
            tmp = data * transpose(D)
            copyto!(data, tmp)
        end
        return data
    end

    # Higher-dimensional: use permutedims to move axis to front
    dims = [axis; [i for i in 1:ndims(data) if i != axis]]
    permuted = permutedims(data, dims)
    reshaped = reshape(permuted, size(permuted, 1), :)
    transformed = D * reshaped
    reshaped_back = reshape(transformed, size(permuted)...)
    unpermuted = permutedims(reshaped_back, invperm(dims))
    copyto!(data, unpermuted)
    return data
end

# ── Plan structure ───────────────────────────────────────────────────────────

"""
    LazyRHSPlan

A pre-built lazy RHS evaluation plan.
- `exprs[i]`: LazyFuture for the i-th state field's RHS, or nothing
- `result_fields[i]`: pre-allocated output ScalarField for state field i
- `workspace`: shared scratch pool
- `is_compiled`: true if all non-trivial equations translated successfully
"""
mutable struct LazyRHSPlan
    exprs::Vector{Union{LazyFuture, Nothing}}
    result_fields::Vector{Union{ScalarField, Nothing}}
    workspaces::Vector{LazyWorkspace}
    is_compiled::Bool
end

function LazyRHSPlan(n_state::Int)
    LazyRHSPlan(
        fill(nothing, n_state),
        fill(nothing, n_state),
        [LazyWorkspace() for _ in 1:n_state],
        false,
    )
end

"""
    build_lazy_rhs_plan!(solver) -> LazyRHSPlan

Walk the problem's equation data and translate each equation's F expression
into a LazyFuture tree. Returns a plan with `is_compiled=true` if successful.
"""
function build_lazy_rhs_plan!(solver)
    problem = solver.problem
    state = solver.state
    plan = LazyRHSPlan(length(state))

    if !hasfield(typeof(problem), :equation_data) || isempty(problem.equation_data)
        plan.is_compiled = true
        return plan
    end

    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        M_expr = get(eq_data, "M", nothing)
        if M_expr === nothing || _is_zero_m_term(M_expr)
            continue
        end

        target_indices = _find_time_derivative_targets(M_expr, state, problem.variables)
        isempty(target_indices) && continue

        expr = if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
            eq_data["F_expr"]
        else
            get(eq_data, "F", nothing)
        end
        expr === nothing && continue

        # For vector equations with multiple targets, translate per-component:
        # each component's RHS resolves VectorField references to its own target.
        for state_idx in target_indices
            if state_idx > length(state)
                continue
            end
            template = state[state_idx]
            lazy = translate_to_lazy(expr, state; target=template)
            if lazy === nothing
                @info "LazyRHS: cannot translate eq$eq_idx F for target $(template.name); falling back" maxlog=3
                return plan  # is_compiled stays false
            end
            plan.exprs[state_idx] = lazy
            plan.result_fields[state_idx] = _lazy_allocate_result(template)
            plan.workspaces[state_idx].template = template
        end
    end

    plan.is_compiled = true
    n_eqs = count(!isnothing, plan.exprs)
    @info "LazyRHS: built type-specialized plan for $n_eqs equations"
    return plan
end

function _lazy_allocate_result(template::ScalarField)
    return ScalarField(template.dist,
                       "_lazy_result_" * template.name,
                       template.bases,
                       template.dtype)
end

"""
    execute_lazy_rhs!(plan::LazyRHSPlan, state, solver) -> Vector{ScalarField}

Execute the compiled lazy plan and return result fields (one per state).
Fields without an RHS (no time derivative) return zero fields.
"""
function execute_lazy_rhs!(plan::LazyRHSPlan, state, solver)
    results = Vector{ScalarField}(undef, length(state))
    for idx in 1:length(state)
        expr = plan.exprs[idx]
        if expr === nothing
            # No RHS → zero field
            results[idx] = _lazy_zero_field(state[idx])
            continue
        end
        result_field = plan.result_fields[idx]
        ws = plan.workspaces[idx]
        ws.next_idx = 1  # reset between equations
        evaluate_lazy!(result_field, expr, state, ws)
        results[idx] = result_field
    end
    return results
end

function _lazy_zero_field(template::ScalarField)
    field = ScalarField(template.dist,
                        "_lazy_zero_" * template.name,
                        template.bases,
                        template.dtype)
    ensure_layout!(field, :g)
    gd = get_local_data(get_grid_data(field))
    if gd !== nothing
        fill!(gd, zero(eltype(gd)))
    end
    field.current_layout = :g
    return field
end
