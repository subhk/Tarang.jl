# ═════════════════════════════════════════════════════════════════════════════
# Lazy RHS runtime.
# ═════════════════════════════════════════════════════════════════════════════
#
# This file is an optimization layer for `evaluate_rhs`, not a separate solver
# architecture. It tries to translate each equation RHS into a typed
# `LazyFuture` tree once, then reuses that compiled tree on every stage.
#
# If translation succeeds, Julia specializes the `evaluate_lazy!` call chain
# and removes most dynamic dispatch from RHS evaluation. If translation fails
# for any operator, runtime falls back to the interpreted
# `evaluate_solver_expression` path and correctness is preserved.
#
# Workspace design: a pool of scratch `ScalarField`s reused while walking the
# expression tree. Binary operators borrow scratch fields for their children,
# then combine them into the caller-provided output field.

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

"""Pointwise field/field division (grid space, undealiased) — mirrors
`divide_operands(::ScalarField, ::ScalarField)`. Field/scalar division is folded
into `LazyScale` at translation time."""
struct LazyDiv{L<:LazyFuture, R<:LazyFuture} <: LazyFuture
    left::L
    right::R
end

"""Pointwise field power with a constant real exponent (grid space, undealiased)
— mirrors `power_operands(::ScalarField, ::Real)`."""
struct LazyPow{T<:LazyFuture} <: LazyFuture
    operand::T
    exponent::Float64
end

"""Pointwise unary grid function (sin, exp, tanh, …) — mirrors
`UnaryGridFunction`. `func` is applied elementwise on grid data."""
struct LazyUnaryFunc{T<:LazyFuture, F} <: LazyFuture
    operand::T
    func::F
end

# ── Differentiation ──────────────────────────────────────────────────────────

struct LazyDiff{T<:LazyFuture} <: LazyFuture
    operand::T
    coord::Coordinate
    order::Int
    axis::Int   # basis axis resolved at translation time (0 = resolve at runtime)
end

"""A SUM of derivatives of ONE shared operand — `Σᵢ ∂^{oᵢ}u/∂xᵢ^{oᵢ}` (i.e. `lap`).

Each `LazyDiff` node costs a full grid→coeff→grid round-trip (two spectral transforms, and
under MPI each is an all-to-all). Measured: every extra ∂ node adds exactly 6 transforms per
RK222 step. Expanding `lap` into a chain of `LazyAdd(LazyDiff, LazyDiff, …)` therefore pays
that round-trip PER AXIS, and re-evaluates the shared operand subtree per axis on top.

This node evaluates the operand ONCE, does ONE forward transform, applies every axis's
multiplier inside that single `:c` visit, and does ONE backward transform — 2 transforms
regardless of dimension."""
struct LazyMultiDiff{T<:LazyFuture} <: LazyFuture
    operand::T
    orders::Vector{Int}
    axes::Vector{Int}
end

# ── Workspace ────────────────────────────────────────────────────────────────

"""
    LazyWorkspace

Pool of pre-allocated ScalarField objects used as scratch buffers during
evaluation. Each `evaluate!` borrows fields from the pool (round-robin) and
returns them via `ws.next_idx` rewinding between sibling calls.
"""
mutable struct LazyWorkspace{F<:ScalarField}
    fields::Vector{F}
    next_idx::Int     # index of next free field (grows as needed)
    template::Union{Nothing, F}
end

LazyWorkspace{F}() where {F<:ScalarField} = LazyWorkspace{F}(F[], 1, nothing)
LazyWorkspace() = LazyWorkspace{ScalarField}()

"""Get a scratch ScalarField from the pool, allocating if needed."""
function _borrow_scratch!(ws::LazyWorkspace{F}) where {F<:ScalarField}
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
        # LazyConst/LazyScale store a real Float64; a complex coefficient (e.g. `2im`)
        # must NOT be silently truncated to its real part (that would zero an imaginary
        # term — `2im*u` would compile to `0.0*u`). Bail so the interpreted RHS path,
        # which carries complex coefficients correctly, takes over.
        imag(expr) != 0 && return nothing
        return LazyConst(Float64(real(expr)))
    end
    if isa(expr, ConstantOperator)
        imag(expr.value) != 0 && return nothing
        return LazyConst(Float64(real(expr.value)))
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
    if isa(expr, DivideOperator)
        l = translate_to_lazy(expr.left, state; target=target)
        r = translate_to_lazy(expr.right, state; target=target)
        (l === nothing || r === nothing) && return nothing
        if isa(r, LazyConst)
            # field / constant → scale by reciprocal (mirrors divide_operands(::ScalarField, ::Number)).
            r.value == 0.0 && return nothing
            return LazyScale(l, 1.0 / r.value)
        end
        return LazyDiv(l, r)
    end
    if isa(expr, PowerOperator)
        # Only a constant real exponent is supported (mirrors power_operands(::ScalarField, ::Real)).
        r = translate_to_lazy(expr.right, state; target=target)
        (r === nothing || !isa(r, LazyConst)) && return nothing
        l = translate_to_lazy(expr.left, state; target=target)
        l === nothing && return nothing
        return LazyPow(l, r.value)
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
        axis = target !== nothing ? _resolve_diff_axis(expr.coord, target.bases) : 0
        return LazyDiff(op, expr.coord, expr.order, axis)
    end

    # Vector/tensor operators that reduce to a SCALAR field. Without these the whole
    # solver falls back to the interpreted RHS evaluator whenever an explicit term is
    # written with lap/div (see `build_lazy_rhs_plan!`). Each expands into the same
    # per-axis derivative sum the interpreted evaluator uses (evaluate_scalar_laplacian
    # / evaluate_divergence), so the compiled and interpreted paths agree by construction.
    #
    # `grad` and `curl` are vector-valued and have no scalar LazyFuture representation;
    # they are translated only inside `div(grad(·))`, and otherwise fall through to
    # `nothing` (interpreted fallback) rather than being silently mistranslated.
    if isa(expr, Laplacian)
        return _translate_laplacian_to_lazy(expr.operand, state, target)
    end

    if isa(expr, Divergence)
        inner = expr.operand
        # div(grad(u)) == lap(u): the idiomatic spelling of a diffusion term.
        isa(inner, Gradient) && return _translate_laplacian_to_lazy(inner.operand, state, target)
        return _translate_divergence_to_lazy(inner, state, target)
    end

    # Pointwise unary grid function (sin, exp, tanh, …)
    if isa(expr, UnaryGridFunction)
        op = translate_to_lazy(expr.operand, state; target=target)
        op === nothing && return nothing
        return LazyUnaryFunc(op, expr.func)
    end

    # Future hierarchy
    if isa(expr, Future)
        return _translate_future_to_lazy(expr, state; target=target)
    end

    return nothing
end

"""True when `_apply_lazy_diff!` can differentiate along `basis` CORRECTLY for `field`.

The lazy derivative is not a drop-in for the interpreted one on every basis, and where it
is not, the honest move is to decline the translation so the solver keeps its existing
(correct) interpreted fallback — NOT to compile and then fail:

  * DISTRIBUTED non-Fourier axis. The lazy derivative works in COEFFICIENT space, where the
    coupled axis is precisely the decomposed one (`pencil_fft_output`), so the global
    differentiation matrix does not conform to a rank's slab. The interpreted derivative
    works in GRID space, where the non-Fourier axis is LOCAL — so it is correct there and we
    must fall back to it. (Verified at np=2: interpreted err 1.3e-12; lazy is unserviceable.)
  * NON-CHEBYSHEV JACOBI (Legendre, general Jacobi). `differentiation_matrix` returns the
    classical recurrence for UNNORMALIZED Pₙ, but the transform stores ORTHONORMAL P̃ₙ = γₙPₙ
    coefficients. The interpreted path applies that normalization; the lazy one does not, so
    it is silently wrong (measured: error 5.3 on an amplitude-11.8 answer, wrong sign).
    ChebyshevT/ChebyshevU carry no such normalization and agree to ~5e-13."""
function _lazy_diff_axis_supported(field::ScalarField, basis)
    isa(basis, FourierBasis) && return true
    isa(basis, JacobiBasis) || return false
    (isa(basis, ChebyshevT) || isa(basis, ChebyshevU)) || return false   # Legendre & friends
    dist = field.dist
    return !(dist !== nothing && dist.size > 1)                          # distributed ⇒ decline
end

"""True when `operand` is differentiable into `target`'s buffer axis-for-axis.

The translators resolve axes against TARGET.bases, but `LazyParamField` copies an operand's
array on a size-only check — so an operand whose bases are in a different ORDER than the
target's (a directly-constructed VectorField does not get the coordsys canonicalization that
`Domain` applies) would be silently transposed rather than rejected."""
function _lazy_bases_match(operand::ScalarField, target::ScalarField)
    length(operand.bases) == length(target.bases) || return false
    for (ob, tb) in zip(operand.bases, target.bases)
        ob === tb || return false
    end
    return true
end

"""Expand `lap(u)` into `Σᵢ ∂²u/∂xᵢ²` over the target's bases.

Mirrors `evaluate_scalar_laplacian`, which sums `Differentiate(u, coord, 2)` over the
operand's bases with `coord = basis.meta.coordsys[basis.meta.element_label]`. A
VectorField operand is resolved to the component matching `target` (the vector
Laplacian is diagonal: `lap(u)ᵢ == lap(uᵢ)`). Returns `nothing` — i.e. leaves the
solver on its existing interpreted fallback — for anything it cannot map exactly."""
function _translate_laplacian_to_lazy(operand, state, target)
    target === nothing && return nothing
    if isa(operand, VectorField)
        comp = _vector_component_for_target(operand, target)
        comp === nothing && return nothing
        operand = comp
    end
    isa(operand, ScalarField) && !_lazy_bases_match(operand, target) && return nothing
    # Decline the WHOLE laplacian if any axis is one the lazy derivative cannot serve.
    for basis in target.bases
        basis === nothing && continue
        _lazy_diff_axis_supported(target, basis) || return nothing
    end
    op = translate_to_lazy(operand, state; target=target)
    op === nothing && return nothing

    axes = Int[]
    for (axis, basis) in enumerate(target.bases)
        basis === nothing && continue
        push!(axes, axis)
    end
    isempty(axes) && return nothing

    # One axis: a plain LazyDiff is already optimal (a fused node would add a copy for nothing).
    # Two or more: fuse, so the operand is evaluated once and the whole Laplacian costs a single
    # forward + backward transform instead of one round-trip PER AXIS.
    length(axes) == 1 && return LazyDiff(op, _coord_of_axis(target, axes[1]), 2, axes[1])
    return LazyMultiDiff(op, fill(2, length(axes)), axes)
end

"""The Coordinate owning `axis` of `field` (as `evaluate_scalar_laplacian` resolves it)."""
_coord_of_axis(field::ScalarField, axis::Int) =
    (b = field.bases[axis]; b.meta.coordsys[b.meta.element_label])

"""Expand `div(v)` into `Σᵢ ∂vᵢ/∂xᵢ`.

Mirrors `evaluate_divergence`: component `i` is differentiated along `coordsys.names[i]`.
Bails (`nothing`) unless every coordinate resolves to an axis of the target, so a
coordinate the target has no basis for falls back to the interpreted path instead of
silently contributing an undifferentiated component (`_apply_lazy_diff!` is a no-op at
`axis == 0`)."""
function _translate_divergence_to_lazy(operand, state, target)
    target === nothing && return nothing
    isa(operand, VectorField) || return nothing
    coordsys = operand.coordsys
    comps = operand.components
    length(comps) == length(coordsys.names) || return nothing
    acc = nothing
    for (i, coord_name) in enumerate(coordsys.names)
        coord = coordsys[coord_name]
        axis = _resolve_diff_axis(coord, target.bases)
        axis == 0 && return nothing
        basis = target.bases[axis]
        # Decline the axes the lazy derivative cannot serve correctly (distributed
        # non-Fourier, non-Chebyshev Jacobi) so the solver keeps the interpreted fallback.
        (basis !== nothing && _lazy_diff_axis_supported(target, basis)) || return nothing
        comp = comps[i]
        isa(comp, ScalarField) && !_lazy_bases_match(comp, target) && return nothing
        c = translate_to_lazy(comp, state; target=target)
        c === nothing && return nothing
        term = LazyDiff(c, coord, 1, axis)
        acc = acc === nothing ? term : LazyAdd(acc, term)
    end
    return acc
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
                # A complex coefficient (e.g. `2im`) must NOT have its imaginary
                # part discarded — folding only the real part would silently zero
                # an imaginary term (`2im*u` → `0.0*u`). Bail so the interpreted
                # RHS, which carries complex coefficients correctly, takes over.
                imag(a) != 0 && return nothing
                scalar_coeff *= Float64(real(a))
            elseif isa(a, ConstantOperator)
                imag(a.value) != 0 && return nothing
                scalar_coeff *= Float64(real(a.value))
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
    f = expr.field
    if !isa(f, ScalarField)
        error("LazyRHS: parameter of type $(typeof(f)) cannot be evaluated as a field term. " *
              "This used to be silently replaced by ZERO, dropping the term from the RHS.")
    end

    ensure_layout!(f, :g)
    src_data = get_local_data(get_grid_data(f))
    src_data === nothing && error("LazyRHS: parameter field `$(f.name)` has no grid data.")

    if size(src_data) == size(out_data)
        copyto!(out_data, src_data)
    elseif isempty(f.bases) || length(src_data) == 1
        # A 0-D / CONSTANT field — e.g. a component of `unit_vector_fields`, whose components are
        # 0-D fields holding a single 1.0 or 0.0. Broadcast its value across the target's grid.
        #
        # This branch used to fall into `fill!(out_data, 0)` because the shapes differ, so a term
        # like `dpdx*ex` on an explicit RHS was silently ZEROED: a pressure-gradient-driven flow
        # never got forced, the fluid stayed at rest, and `is_compiled` still reported `true`.
        @inbounds fill!(out_data, convert(eltype(out_data), real(first(src_data))))
    else
        error("LazyRHS: parameter field `$(f.name)` has local grid size $(size(src_data)), which " *
              "does not match the target field's $(size(out_data)) and is not a 0-D constant. " *
              "Zeroing it (the old behavior) would silently drop the term from the RHS.")
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
        # A LazyMul is a genuine field·field product (field·scalar becomes LazyScale),
        # i.e. a nonlinear term. A raw grid multiply aliases; dealias it via the same
        # 3/2-padded (serial) / 2/3-truncation (MPI) machinery used elsewhere, unless
        # every Fourier axis has dealias ≤ 1 (dealiasing disabled).
        if _any_axis_dealias(out.bases, 1.5)
            _dealiased_lazy_product!(out, a, b)
        else
            _fused_binary!(out, a, b, *)
        end
    end
    return out
end

"""Dealiased field·field product for the lazy RHS, written into `out` (grid layout)."""
function _dealiased_lazy_product!(out::ScalarField, a::ScalarField, b::ScalarField)
    evaluator = _get_evaluator(out.dist)
    product = evaluate_transform_multiply(a, b, evaluator)
    ensure_layout!(product, :g)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    prod_data = get_local_data(get_grid_data(product))
    if out_data !== nothing && prod_data !== nothing && size(out_data) == size(prod_data)
        copyto!(out_data, prod_data)
    elseif out_data !== nothing
        fill!(out_data, zero(eltype(out_data)))
    end
    out.current_layout = :g
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
    _apply_lazy_diff!(out, expr.coord, expr.order, expr.axis)
    return out
end

# Σᵢ ∂ᵢ^{oᵢ}(operand) in ONE coefficient-space visit: 2 spectral transforms total, instead of
# 2 per axis. The per-axis multiplier application is the SAME code the single-axis path uses
# (`_apply_lazy_diff_coeff!`), so rfft-vs-full-fft layout, the pencil permutation and the
# distributed local index ranges are all handled identically — no new index math.
function evaluate_lazy!(out::ScalarField, expr::LazyMultiDiff, state, ws::LazyWorkspace)
    _with_scratch_scope(ws) do
        src = _borrow_scratch!(ws)
        evaluate_lazy!(src, expr.operand, state, ws)
        ensure_layout!(src, :c)              # ONE forward transform, shared by every axis
        src_coeff = get_local_data(get_coeff_data(src))

        # First axis: seed `out` from the operand's coefficients (a copy, not a transform).
        out_coeff = get_local_data(get_coeff_data(out))
        copyto!(out_coeff, src_coeff)
        out.current_layout = :c
        _apply_lazy_diff_coeff!(out, expr.orders[1], expr.axes[1])

        # Remaining axes: differentiate a copy of the SAME coefficients and accumulate.
        tmp = length(expr.axes) > 1 ? _borrow_scratch!(ws) : nothing
        if tmp !== nothing
            tmp_coeff = get_local_data(get_coeff_data(tmp))
            @inbounds for i in 2:length(expr.axes)
                copyto!(tmp_coeff, src_coeff)
                tmp.current_layout = :c
                _apply_lazy_diff_coeff!(tmp, expr.orders[i], expr.axes[i])
                out_c = get_local_data(get_coeff_data(out))
                @. out_c += tmp_coeff
            end
        end

        out.current_layout = :c
        ensure_layout!(out, :g)              # ONE backward transform

        # Hand the scratch fields back flagged :g. They are pooled and reused round-robin by
        # sibling subtrees, whose writers all begin with `ensure_layout!(scratch, :g)` — a
        # :c-flagged scratch would make that a full BACKWARD transform of coefficients the
        # writer immediately overwrites. (Measured: leaving them :c re-added exactly the 3
        # backward transforms per step this node had just removed.) Every evaluate_lazy!
        # writer fully overwrites the grid array, so the :g flag costs nothing and is safe.
        src.current_layout = :g
        tmp === nothing || (tmp.current_layout = :g)
    end
    return out
end

# Pointwise field/field division in grid space (undealiased), mirroring
# `divide_operands(::ScalarField, ::ScalarField)`.
@inline function evaluate_lazy!(out::ScalarField, expr::LazyDiv, state, ws::LazyWorkspace)
    _with_scratch_scope(ws) do
        a = _borrow_scratch!(ws)
        b = _borrow_scratch!(ws)
        evaluate_lazy!(a, expr.left, state, ws)
        evaluate_lazy!(b, expr.right, state, ws)
        _fused_binary!(out, a, b, /)
    end
    return out
end

# Pointwise field power with constant exponent, mirroring
# `power_operands(::ScalarField, ::Real)` (grid space, undealiased).
@inline function evaluate_lazy!(out::ScalarField, expr::LazyPow, state, ws::LazyWorkspace)
    evaluate_lazy!(out, expr.operand, state, ws)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    p = expr.exponent
    if out_data !== nothing
        @. out_data = out_data ^ p
    end
    out.current_layout = :g
    return out
end

# Pointwise unary grid function (sin/exp/…), mirroring `UnaryGridFunction`.
@inline function evaluate_lazy!(out::ScalarField, expr::LazyUnaryFunc, state, ws::LazyWorkspace)
    evaluate_lazy!(out, expr.operand, state, ws)
    ensure_layout!(out, :g)
    out_data = get_local_data(get_grid_data(out))
    f = expr.func
    if out_data !== nothing
        @. out_data = f(out_data)
    end
    out.current_layout = :g
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

"""Resolve which basis axis a coordinate maps to (0 if not present in `bases`)."""
function _resolve_diff_axis(coord::Coordinate, bases)
    coord_name = String(coord.name)
    for (i, basis) in enumerate(bases)
        if basis !== nothing && String(basis.meta.element_label) == coord_name
            return i
        end
    end
    return 0
end

function _apply_lazy_diff!(field::ScalarField, coord::Coordinate, order::Int, axis_hint::Int=0)
    # Axis is precomputed at translation time; fall back to a runtime lookup only
    # when the hint is absent (0), avoiding per-call string allocations otherwise.
    axis = axis_hint == 0 ? _resolve_diff_axis(coord, field.bases) : axis_hint
    axis == 0 && return field  # Not in this field's bases — skip

    # Switch to coefficient space
    ensure_layout!(field, :c)
    _apply_lazy_diff_coeff!(field, order, axis)
    # Transform back to grid space for downstream operations
    ensure_layout!(field, :g)
    return field
end

"""Coefficient-space core of the lazy derivative: `field` must ALREADY be `:c`, and is left `:c`.

Split out of `_apply_lazy_diff!` so a fused multi-axis node (`LazyMultiDiff`) can apply several
derivatives inside a SINGLE `:c` visit instead of paying a grid→coeff→grid round-trip per axis."""
function _apply_lazy_diff_coeff!(field::ScalarField, order::Int, axis::Int)
    axis == 0 && return field
    target_basis = field.bases[axis]
    target_basis === nothing && return field

    coeff_storage = get_coeff_data(field)
    coeff_storage === nothing && return field

    # Apply differentiation in coefficient space.
    # Chebyshev/Jacobi: Nz×Nz differentiation matrix (applied via matmul)
    # Fourier: diagonal scaling per mode (im*k*k0)^order — the RFFT
    #          representation is complex, so matrix form doesn't apply
    if isa(target_basis, JacobiBasis)
        coeff_data = get_local_data(coeff_storage)
        coeff_data === nothing && return field
        D = differentiation_matrix(target_basis, order)
        _apply_1d_matrix!(coeff_data, D, _jacobi_diff_axis(coeff_storage, axis, target_basis, D),
                          target_basis)
    elseif isa(target_basis, FourierBasis)
        _apply_lazy_fourier_diff!(coeff_storage, field, target_basis, axis, order)
    end

    field.current_layout = :c
    return field
end

"""Physical (parent-array) axis to apply a Jacobi differentiation matrix along.

The Jacobi branch multiplies the GLOBAL Nz×Nz differentiation matrix into the LOCAL
coefficient array, which is only meaningful when this rank owns the whole non-Fourier
axis. Two things can break that under MPI, and both used to be silent:

  * the axis is DECOMPOSED — the rank holds a slab, so the global matrix does not
    conform. Observed at np=2 on a Chebyshev×Fourier domain with `dt(q) = nu*lap(q)`:
    either a `DimensionMismatch`, or (when the local extent coincidentally matches Nz)
    an RHS wrong by more than its own amplitude, integrated forever with no error.
  * the pencil is PERMUTED — `parent()` reorders the dims, so the logical axis is not
    the parent axis (the Fourier branch already handles this; this one did not).

A distributed non-Fourier derivative is only correct in the solve layout (the transpose
machinery in `subproblem_io.jl`), which the lazy RHS does not enter — so refuse loudly
rather than return garbage. Serial fields are unaffected."""
function _jacobi_diff_axis(coeff_storage, axis::Int, basis, D)
    coeff_storage isa PencilArrays.PencilArray || return axis

    pencil = PencilArrays.pencil(coeff_storage)
    n_local = length(pencil.axes_local[axis])
    # DECOMPOSITION is the thing that makes the local matmul wrong. Test it against this
    # array's own global extent rather than size(D,1), so a padded/rescaled coefficient
    # array that is nonetheless fully local is not rejected for the wrong reason.
    n_global = PencilArrays.size_global(pencil)[axis]
    if n_local != n_global
        error("Lazy RHS: cannot differentiate along the non-Fourier axis $axis " *
              "($(nameof(typeof(basis)))) of a DISTRIBUTED field — this rank owns only " *
              "$n_local of the $n_global coefficients on that axis, so the global " *
              "differentiation matrix does not apply to its local slab. Under MPI a " *
              "non-Fourier derivative is only correct in the solve layout. Move the term " *
              "to the implicit (L) side of the equation, or run in serial.")
    end

    # perm_tuple[physical] = logical. Tuple(NoPermutation()) is `nothing`, NOT the identity.
    perm_raw = Tuple(PencilArrays.permutation(coeff_storage))
    perm_raw === nothing && return axis
    physical = findfirst(==(axis), perm_raw)
    return physical === nothing ? axis : physical
end

"""
Apply Fourier differentiation as a diagonal scaling in coefficient space.
For RealFourier transforms, only the first RealFourier axis is in RFFT layout;
later RealFourier axes are full FFT axes and must use negative wavenumbers in
FFT order.
"""
function _apply_lazy_fourier_diff!(coeff_storage, field::ScalarField,
                                   basis::FourierBasis, axis::Int, order::Int)
    uses_rfft = isa(basis, RealFourier) && _is_first_real_fourier_axis(field.bases, axis)

    if isa(coeff_storage, PencilArrays.PencilArray)
        _apply_spectral_derivative_distributed!(coeff_storage, basis, axis, order,
                                                field.dist; uses_rfft=uses_rfft)
        return coeff_storage
    end

    data = get_local_data(coeff_storage)
    data === nothing && return coeff_storage

    deriv_mult = _get_cached_lazy_deriv_mult(basis, order, uses_rfft)
    if length(deriv_mult) != size(data, axis)
        error("Lazy Fourier derivative coefficient size mismatch on axis $axis: " *
              "local coefficient size is $(size(data, axis)) but expected $(length(deriv_mult)) " *
              "(basis=$(typeof(basis)), uses_rfft=$uses_rfft).")
    end

    # `data` is a concrete array (SerialFieldStorage is parametrized on the array
    # type) and `deriv_mult` is a concrete Vector{ComplexF64}, so this broadcast is
    # type-stable inline — no function barrier needed.
    mult_shape = ntuple(i -> i == axis ? length(deriv_mult) : 1, ndims(data))
    data .*= reshape(deriv_mult, mult_shape...)
    return coeff_storage
end

"""
Cache `(ik)^order` for the lazy Fourier derivative in `basis.transforms`.
Keyed by `(order, uses_rfft)` since the basis (hence N and L) is fixed; this
mirrors `_get_cached_deriv_mult` for the non-lazy path and avoids reallocating
the wavenumber-multiplier array on every RHS evaluation.
"""
function _get_cached_lazy_deriv_mult(basis::FourierBasis, order::Int, uses_rfft::Bool)
    cache_key = (:lazy_deriv_mult, order, uses_rfft)
    cached = get(basis.transforms, cache_key, nothing)
    cached !== nothing && return cached::Vector{ComplexF64}

    k_axis = if isa(basis, RealFourier)
        uses_rfft ? wavenumbers_rfft(basis) : wavenumbers_fft(basis)
    else
        wavenumbers(basis)
    end
    deriv_mult = ComplexF64.((im .* k_axis) .^ order)
    basis.transforms[cache_key] = deriv_mult
    return deriv_mult
end

"""Apply a 1D matrix `D` along `axis` of multi-dimensional array `data` in place.

The 1D/2D cases (the common Chebyshev/Jacobi spectral derivative) reuse a scratch
buffer cached in `basis.transforms`, keyed by `(size, eltype)`, instead of
allocating the matmul output every call. Safe because lazy RHS evaluation is
sequential — the buffer is filled and consumed within a single call before any
other derivative runs (same contract as the `_DERIV_FFT_WS` FFT buffers)."""
function _apply_1d_matrix!(data::AbstractArray, D::AbstractMatrix, axis::Int, basis)
    nd = ndims(data)
    if nd == 1 || nd == 2
        tmp = _diff_matmul_buffer(basis, data)
        _matmul_axis_into!(data, D, tmp, axis)
        return data
    end

    # Higher-dimensional (rare): permutedims path, left allocating.
    dims = [axis; [i for i in 1:ndims(data) if i != axis]]
    permuted = permutedims(data, dims)
    reshaped = reshape(permuted, size(permuted, 1), :)
    transformed = D * reshaped
    reshaped_back = reshape(transformed, size(permuted)...)
    unpermuted = permutedims(reshaped_back, invperm(dims))
    copyto!(data, unpermuted)
    return data
end

"""Reusable matmul scratch for `_apply_1d_matrix!`, cached per basis by shape+eltype."""
function _diff_matmul_buffer(basis, data::AbstractArray)
    key = (:diff_matmul_tmp, size(data), eltype(data))
    buf = get(basis.transforms, key, nothing)
    if buf === nothing
        buf = similar(data)
        basis.transforms[key] = buf
    end
    return buf
end

# Function barrier: `tmp` arrives `Any`-typed from the `Dict{Any,Any}` basis.transforms
# cache (`data` is already concrete post storage parametrization), so dispatching here
# recovers `tmp`'s concrete type and keeps `mul!`/`copyto!` allocation-free.
@inline function _matmul_axis_into!(data::AbstractArray, D::AbstractMatrix, tmp::AbstractArray, axis::Int)
    if ndims(data) == 1 || axis == 1
        mul!(tmp, D, data)            # (D * data) along first axis
    else  # 2D, axis == 2
        mul!(tmp, data, transpose(D)) # (data * Dᵀ) along second axis
    end
    copyto!(data, tmp)
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
mutable struct LazyRHSPlan{F<:ScalarField}
    exprs::Vector{Union{LazyFuture, Nothing}}
    result_fields::Vector{Union{F, Nothing}}
    output_fields::Vector{F}
    workspaces::Vector{LazyWorkspace{F}}
    is_compiled::Bool
end

function LazyRHSPlan{F}(n_state::Int) where {F<:ScalarField}
    LazyRHSPlan{F}(
        fill(nothing, n_state),
        fill(nothing, n_state),
        Vector{F}(undef, n_state),
        [LazyWorkspace{F}() for _ in 1:n_state],
        false,
    )
end

LazyRHSPlan(n_state::Int) = LazyRHSPlan{ScalarField}(n_state)

_lazy_plan_field_type(state) = _lazy_plan_field_type(eltype(state))
_lazy_plan_field_type(::Type{F}) where {F<:ScalarField{<:Any, <:SerialFieldStorage}} =
    isconcretetype(F) ? F : ScalarField
_lazy_plan_field_type(::Type) = ScalarField

# When set (via `require_lazy_rhs!(true)`), `build_lazy_rhs_plan!` throws instead
# of silently dropping the whole solver onto the ~100×-slower interpreted RHS
# path. Off by default to keep the correctness-preserving fallback; turn it on in
# CI / perf-sensitive runs to catch a degradation early.
const REQUIRE_LAZY_RHS = Ref(false)

"""
    require_lazy_rhs!(on::Bool=true) -> Bool

When `on`, RHS-compilation failure throws instead of silently falling back to the
slower interpreted evaluator. Returns the previous setting.
"""
function require_lazy_rhs!(on::Bool=true)
    prev = REQUIRE_LAZY_RHS[]
    REQUIRE_LAZY_RHS[] = on
    return prev
end

"""
    build_lazy_rhs_plan!(solver) -> LazyRHSPlan

Walk the problem's equation data and translate each equation's F expression
into a LazyFuture tree. Returns a plan with `is_compiled=true` if successful.
"""
function build_lazy_rhs_plan!(solver)
    problem = solver.problem
    state = solver.state
    F = _lazy_plan_field_type(state)
    plan = LazyRHSPlan{F}(length(state))

    for (idx, template) in enumerate(state)
        zero_field = _lazy_zero_field(template)
        plan.result_fields[idx] = zero_field
        plan.output_fields[idx] = zero_field
        plan.workspaces[idx].template = template
    end

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
                distributed = template.dist !== nothing && template.dist.size > 1
                # A DELIBERATE decline, not a defect: the lazy derivative operates in
                # coefficient space, where a non-Fourier axis is the DECOMPOSED one, so it
                # cannot serve a derivative along it. The interpreted evaluator differentiates
                # in GRID space, where that axis is local — it is the CORRECT path here, just
                # slower. Do not cry wolf about it (see `_lazy_diff_axis_supported`).
                declined_non_fourier = distributed && any(
                    b -> b !== nothing && !isa(b, FourierBasis), template.bases)

                msg = "LazyRHS: cannot compile the RHS of equation $eq_idx for `$(template.name)` " *
                      "(expression: $(string(expr))). The whole solver falls back to the " *
                      "~100×-slower interpreted RHS evaluator."
                if declined_non_fourier
                    msg *= " This is EXPECTED and CORRECT for a distributed non-Fourier " *
                           "(Chebyshev/Jacobi) axis: the compiled path cannot differentiate " *
                           "along a decomposed coefficient axis, so the interpreted evaluator " *
                           "handles it — accurately, but slowly. Move the term to the implicit " *
                           "(L) side to keep the compiled RHS."
                else
                    msg *= " Usual cause: an unsupported or mistyped operator in this term."
                    # For an all-Fourier distributed field the fallback is not merely slow: the
                    # distributed interpreted derivative has been observed to return zero,
                    # leaving the field FROZEN with no error raised.
                    if distributed
                        msg *= " UNDER MPI THIS PATH IS ALSO KNOWN TO PRODUCE SILENTLY WRONG " *
                               "RESULTS (the distributed interpreted derivative can evaluate to " *
                               "zero, freezing the field). Rewrite the term using supported " *
                               "operators, or call `require_lazy_rhs!()` to turn this into a hard error."
                    end
                end
                REQUIRE_LAZY_RHS[] && error(msg * " [require_lazy_rhs! is set — aborting.]")
                @warn msg maxlog=5
                return plan  # is_compiled stays false
            end
            plan.exprs[state_idx] = lazy
            result_field = _lazy_allocate_result(template)
            plan.result_fields[state_idx] = result_field
            plan.output_fields[state_idx] = result_field
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

function execute_lazy_rhs_buffered!(plan::LazyRHSPlan, state, solver)
    for idx in eachindex(plan.output_fields, state)
        expr = plan.exprs[idx]
        if expr === nothing
            _reset_lazy_forced_rhs_field!(plan.output_fields[idx], solver.problem, idx)
            continue
        end
        result_field = plan.output_fields[idx]
        ws = plan.workspaces[idx]
        ws.next_idx = 1  # reset between equations
        evaluate_lazy!(result_field, expr, state, ws)
    end
    _add_registered_forcings_to_lazy_rhs!(plan.output_fields, solver.problem)
    return plan.output_fields
end

function _reset_lazy_forced_rhs_field!(field::ScalarField, problem::Problem, var_idx::Int)
    hasfield(typeof(problem), :stochastic_forcings) || return field
    haskey(problem.stochastic_forcings, var_idx) || return field
    isempty(field.bases) && return field

    ensure_layout!(field, :c)
    coeff_data = get_coeff_data(field)
    coeff_data === nothing && return field
    fill!(coeff_data, zero(eltype(coeff_data)))
    return field
end

function _add_registered_forcings_to_lazy_rhs!(rhs_fields, problem::Problem)
    hasfield(typeof(problem), :stochastic_forcings) || return rhs_fields
    isempty(problem.stochastic_forcings) && return rhs_fields

    for (var_idx, forcing) in problem.stochastic_forcings
        var_idx <= length(rhs_fields) || continue
        rhs_field = rhs_fields[var_idx]
        isempty(rhs_field.bases) && continue

        ensure_layout!(rhs_field, :c)
        coeff_data = get_coeff_data(rhs_field)
        coeff_data === nothing && continue

        F_view = _matched_forcing_view(forcing, coeff_data)
        if F_view !== nothing
            coeff_data .+= F_view
        else
            @warn "Forcing size doesn't match RHS size for state field $var_idx"
        end
    end

    return rhs_fields
end

"""
    execute_lazy_rhs!(plan::LazyRHSPlan, state, solver) -> Vector{ScalarField}

Execute the compiled lazy plan and return result fields (one per state).
Fields without an RHS (no time derivative) return zero fields.
"""
function execute_lazy_rhs!(plan::LazyRHSPlan, state, solver)
    return execute_lazy_rhs_buffered!(plan, state, solver)
end

function _lazy_zero_field(template::ScalarField)
    field = ScalarField(template.dist,
                        "_lazy_zero_" * template.name,
                        template.bases,
                        template.dtype)
    layout = template.current_layout === :c ? :c : :g
    ensure_layout!(field, layout)
    data = layout === :c ? get_coeff_data(field) : get_grid_data(field)
    local_data = data === nothing ? nothing : get_local_data(data)
    if local_data !== nothing
        fill!(local_data, zero(eltype(local_data)))
    end
    field.current_layout = layout
    return field
end

# Note: _is_zero_m_term and _find_time_derivative_targets are defined in
# timesteppers/state_utils.jl (they were moved there to consolidate helpers).
