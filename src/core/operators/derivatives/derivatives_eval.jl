# Derivative evaluator entry points for gradient, divergence, and Differentiate.
#
# These functions are the boundary between the symbolic operator tree and the
# numerical spectral-differentiation kernels. They resolve the operand's field
# rank and basis, allocate the correctly-typed result, and delegate the actual
# per-basis math to `evaluate_fourier_derivative!`, `evaluate_chebyshev_derivative!`,
# and `evaluate_legendre_derivative!` in the sibling `derivatives_*.jl` files.

# ============================================================================
# Gradient and Divergence Evaluation
# ============================================================================

"""
    evaluate_gradient(grad_op, layout=:g) -> VectorField | TensorField

Evaluate `∇` by differentiating along every coordinate of the operand's system.
The result rank goes up by one: a `ScalarField` yields a `VectorField`
(`∂f/∂xᵢ`); a `VectorField` yields a `TensorField` Jacobian (`Tᵢⱼ = ∂uⱼ/∂xᵢ`).
`layout` selects whether components are returned in grid (`:g`) or coefficient
(`:c`) space. Throws `ArgumentError` for unsupported operand ranks.
"""
function evaluate_gradient(grad_op::Gradient, layout::Symbol=:g)
    operand = grad_op.operand
    coordsys = grad_op.coordsys

    if isa(operand, ScalarField)
        # Scalar → VectorField (∂f/∂xᵢ for each i)
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end
        return result

    elseif isa(operand, VectorField)
        # Vector → TensorField (Jacobian: Tᵢⱼ = ∂uⱼ/∂xᵢ)
        ndim = length(coordsys.names)
        result = TensorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            for j in 1:length(operand.components)
                result.components[i, j] = evaluate_differentiate(
                    Differentiate(operand.components[j], coord, 1), layout)
            end
        end
        return result

    else
        throw(ArgumentError("Gradient not implemented for operand type $(typeof(operand))"))
    end
end

"""
    evaluate_divergence(div_op, layout=:g) -> ScalarField | VectorField

Evaluate `∇·(…)`, lowering the operand's rank by one. Four operand forms are
supported:

1. **`VectorField`** — sums `∂uᵢ/∂xᵢ` over all coordinates into a single scalar
   result.

2. **A scalar coefficient times a `Gradient`** — `div(a*grad(u))` (either factor
   order), the standard variable-coefficient diffusion term. It is expanded with
   the exact product rule

       ∇·(a ∇u) = a ∇²u + ∇a·∇u = Σₖ (a ∂ₖ²u + ∂ₖa ∂ₖu)

   so every derivative is taken spectrally on `a` or `u` themselves and only the
   pointwise products are formed on the grid — numerically identical (to
   round-off) to writing `a*lap(u) + ∂x(a)*∂x(u) + …` by hand. `u` may be a
   `ScalarField` (result: `ScalarField`) or a `VectorField` (result:
   `VectorField`; the identity applied component-wise, since `∇·(a∇u)ⱼ =
   Σₖ ∂ₖ(a ∂ₖuⱼ)`).

3. **A scalar coefficient times a `VectorField`** — `div(a*u)` (either factor
   order), the *conservative flux* form: `∇·(ρu)` in mass conservation, `∇·(uc)`
   in conservative advection, `∇·(κ q)` for a variable-coefficient flux. Expanded
   with the same kind of exact product rule,

       ∇·(a u) = a (∇·u) + u·∇a = Σᵢ (a ∂ᵢuᵢ + uᵢ ∂ᵢa)

   so, again, derivatives are spectral on `a` and `u` themselves and only the
   pointwise products touch the grid. Writing it this way rather than
   differentiating the assembled product `a uᵢ` keeps the answer identical to the
   hand-written `a*div(u) + dot(u, grad(a))` and avoids differentiating a product
   that may not be resolved on the grid its factors are resolved on.

4. **Any other expression that evaluates to a `VectorField`** — e.g.
   `div(grad(u))`, `div(skew(grad(ψ)))`, `div(-u)`, `div(a*curl(A))`. The operand
   is evaluated and its divergence taken as in case 1.

Accumulation is done directly on the result's field data (not via the symbolic
`+` tree) to avoid building intermediate operator nodes. The result buffer is
taken from the field pool and zero-initialized; PencilArray buffers are zeroed
via their `parent`.

Anything else throws an `ArgumentError` naming the unsupported operand. A
divergence must never quietly evaluate to zero because its operand was not
recognized: callers treat a zero field as a real value, so a silent fallback
turns an unsupported term into a wrong answer rather than a failure.
"""
function evaluate_divergence(div_op::Divergence, layout::Symbol=:g)
    operand = div_op.operand

    isa(operand, VectorField) && return _evaluate_vector_divergence(operand, layout)

    # div(a*grad(u)): expand symbolically before evaluating, so the coefficient
    # and the gradient are never multiplied into an intermediate vector field.
    factors = _divergence_product_factors(operand)
    if factors !== nothing
        coeff_expr, grad_op = factors
        return _evaluate_variable_coefficient_divergence(coeff_expr, grad_op, layout)
    end

    # div(a*u): the conservative flux form, expanded the same way.
    flux_factors = _divergence_flux_factors(operand)
    if flux_factors !== nothing
        coeff_expr, vector_field = flux_factors
        return _evaluate_conservative_flux_divergence(coeff_expr, vector_field, layout)
    end

    # Any other expression: evaluate it; a vector result has a well-defined
    # divergence. Errors raised while evaluating the operand propagate as-is —
    # they name the real problem better than a generic message here could.
    evaluated = _eval_operand(operand, :g)
    isa(evaluated, VectorField) && return _evaluate_vector_divergence(evaluated, layout)

    evaluated_note = evaluated === operand ? "" : " (evaluates to $(typeof(evaluated)))"
    throw(ArgumentError(
        "Divergence not implemented for operand type $(typeof(operand))$(evaluated_note). " *
        "∇· lowers rank by one, so its operand must be vector-valued. Supported: a " *
        "VectorField, a scalar coefficient times a gradient (`div(a*grad(u))`), a scalar " *
        "coefficient times a vector field (`div(a*u)`), or any expression that evaluates " *
        "to a VectorField."))
end

"""
    _evaluate_vector_divergence(operand::VectorField, layout) -> ScalarField

`∇·u = Σᵢ ∂uᵢ/∂xᵢ`, accumulated straight into the result's field data.
Component `i` is differentiated along `coordsys.names[i]`.
"""
function _evaluate_vector_divergence(operand::VectorField, layout::Symbol)
    coordsys = operand.coordsys
    template = operand.components[1]
    result = _divergence_result_field(template, "div_$(operand.name)", layout)

    for (i, coord_name) in enumerate(coordsys.names)
        coord = coordsys[coord_name]
        # Add d(u_i)/d(x_i) — accumulate into field data directly, not via symbolic +
        component_deriv = evaluate_differentiate(Differentiate(operand.components[i], coord, 1), layout)
        if layout == :g
            get_grid_data(result) .+= get_grid_data(component_deriv)
        else
            get_coeff_data(result) .+= get_coeff_data(component_deriv)
        end
    end

    return result
end

"""
    _divergence_result_field(template, name, layout) -> ScalarField

Check out a result field shaped like `template` (preserving PencilArray
structure via `copy_field_data!`), put it in `layout`, and zero it. Data is
allocated explicitly if the copy did not provide it.
"""
function _divergence_result_field(template::ScalarField, name::AbstractString, layout::Symbol)
    result = checkout_or_alloc(template.bases, template.dtype, template.dist)
    copy_field_data!(result, template)
    result.current_layout = template.current_layout
    result.name = name
    ensure_layout!(result, layout)

    if layout == :g
        grid_data = get_grid_data(result)
        if grid_data === nothing
            # Allocate grid data if not present (copy may not have provided it)
            set_grid_data!(result, zeros(eltype(get_grid_data(template)),
                                         size(get_grid_data(template))))
        else
            _fill_zeros!(grid_data)
        end
    else
        coeff_data = get_coeff_data(result)
        if coeff_data === nothing
            set_coeff_data!(result, zeros(eltype(get_coeff_data(template)),
                                          size(get_coeff_data(template))))
        else
            _fill_zeros!(coeff_data)
        end
    end

    return result
end

"""Zero an array, going through `parent` for PencilArray buffers."""
@inline function _fill_zeros!(data::AbstractArray)
    if isa(data, PencilArrays.PencilArray)
        fill!(parent(data), zero(eltype(data)))
    else
        fill!(data, zero(eltype(data)))
    end
    return data
end

"""
    _divergence_product_factors(operand) -> (coefficient_expr, ::Gradient) | nothing

Split a symbolic product into its `Gradient` factor and the coefficient
multiplying it, accepting either operand order (`a*grad(u)` and `grad(u)*a`) and
both product spellings: `MultiplyOperator` (built by the equation-string parser)
and the deferred `Multiply` future (built by the Julia `*` operators).

Returns `nothing` when the operand is not such a product — including when it
holds more than one `Gradient`, where "the coefficient" is not well defined.
"""
_divergence_product_factors(operand) = nothing

function _divergence_product_factors(operand::MultiplyOperator)
    left, right = operand.left, operand.right
    if isa(right, Gradient) && !isa(left, Gradient)
        return (left, right)
    elseif isa(left, Gradient) && !isa(right, Gradient)
        return (right, left)
    end
    return nothing
end

function _divergence_product_factors(operand::Multiply)
    args = future_args(operand)
    gradient_positions = findall(arg -> isa(arg, Gradient), args)
    length(gradient_positions) == 1 || return nothing
    k = only(gradient_positions)
    rest = [arg for (i, arg) in enumerate(args) if i != k]
    coefficient = if isempty(rest)
        1
    elseif length(rest) == 1
        only(rest)
    else
        Multiply(rest...)
    end
    return (coefficient, args[k])
end

"""
    _divergence_flux_factors(operand) -> (coefficient_expr, ::VectorField) | nothing

Split a symbolic product into its `VectorField` factor and the coefficient
multiplying it — the conservative flux `a u` — accepting either operand order
(`a*u` and `u*a`) and both product spellings, exactly as
[`_divergence_product_factors`](@ref) does for `a*grad(u)`.

Returns `nothing` unless exactly one factor is a `VectorField`; with none there
is no flux, and with two (`v*u`) "the coefficient" is not well defined and the
product itself is ambiguous (dot? cross?). Both cases fall through to the generic
branch of `evaluate_divergence`, which raises the error that names them.

Note this is checked *after* the `Gradient` split, so `div(v*grad(u))` with a
vector `v` still reports the unsupported non-scalar coefficient rather than being
re-read as a flux whose coefficient happens to be a gradient.
"""
_divergence_flux_factors(operand) = nothing

function _divergence_flux_factors(operand::MultiplyOperator)
    left, right = operand.left, operand.right
    if isa(right, VectorField) && !isa(left, VectorField)
        return (left, right)
    elseif isa(left, VectorField) && !isa(right, VectorField)
        return (right, left)
    end
    return nothing
end

function _divergence_flux_factors(operand::Multiply)
    args = future_args(operand)
    vector_positions = findall(arg -> isa(arg, VectorField), args)
    length(vector_positions) == 1 || return nothing
    k = only(vector_positions)
    rest = [arg for (i, arg) in enumerate(args) if i != k]
    coefficient = if isempty(rest)
        1
    elseif length(rest) == 1
        only(rest)
    else
        Multiply(rest...)
    end
    return (coefficient, args[k])
end

"""
    _evaluate_conservative_flux_divergence(coeff_expr, u, layout) -> ScalarField

`∇·(a u)` via the exact product rule `Σᵢ (a ∂ᵢuᵢ + uᵢ ∂ᵢa)`. Coordinates come
from the vector field's own coordinate system, matching
`_evaluate_vector_divergence`, so component `i` is differentiated along
`coordsys.names[i]`.

Throws `ArgumentError` when the coefficient is not scalar-valued, or when the
vector field has a different number of components than the coordinate system has
coordinates (there is then no `∂ᵢuᵢ` pairing to sum).
"""
function _evaluate_conservative_flux_divergence(coeff_expr, u::VectorField, layout::Symbol)
    coefficient = _eval_operand(coeff_expr, :g)
    if !(isa(coefficient, ScalarField) || isa(coefficient, Number))
        throw(ArgumentError(
            "div(a*u) requires a SCALAR coefficient `a` — a ScalarField, a number, or an " *
            "expression evaluating to one. Got $(typeof(coeff_expr)) evaluating to " *
            "$(typeof(coefficient)), which is not scalar-valued. A vector or tensor " *
            "coefficient has no product-rule expansion here: `div(v*u)` is ambiguous " *
            "(dot or cross?), and a tensor flux must be assembled explicitly and passed " *
            "to div() as a vector."))
    end

    coordsys = u.coordsys
    if length(u.components) != length(coordsys.names)
        coordinate_list = join(coordsys.names, ", ")
        throw(ArgumentError(
            "div(a*u): VectorField '$(u.name)' has $(length(u.components)) components but " *
            "its coordinate system has $(length(coordsys.names)) coordinates " *
            "($(coordinate_list)). ∇·(a u) = Σᵢ ∂ᵢ(a uᵢ) needs one component per " *
            "coordinate."))
    end

    # Same hazard as in `_evaluate_variable_coefficient_divergence`: an expression
    # coefficient can come back in one of the rotating derivative-pool buffers,
    # which is recycled after `_DERIV_RESULT_POOL_SIZE` further derivatives. Pin
    # it into a checked-out field, which is never handed out again while it is
    # held, so `a` cannot change under us mid-accumulation.
    #
    # The margin here is wider than in the gradient sibling: this loop issues 2
    # checkouts per coordinate (6 in 3D) against that path's 3 per coordinate per
    # component (27 for a 3D vector, which does wrap 16 and did corrupt
    # components 2 and 3 by 2.0 and 2.9 before it was pinned). So this pin is not
    # currently load-bearing — measured identical with and without at ndim ≤ 3 —
    # but the count scales with ndim while the pool size is a constant that has
    # already had to be raised once, and the copy is one field per div().
    if isa(coefficient, ScalarField) && !isa(coeff_expr, ScalarField)
        pinned = checkout_or_alloc(coefficient.bases, coefficient.dtype, coefficient.dist)
        copy_field_data!(pinned, coefficient)
        pinned.current_layout = coefficient.current_layout
        pinned.name = coefficient.name
        coefficient = pinned
    end

    result = _divergence_result_field(u.components[1], "div_flux_$(u.name)", :g)
    _accumulate_conservative_flux_divergence!(result, coefficient, u, coordsys)
    ensure_layout!(result, layout)
    return result
end

"""
    _accumulate_conservative_flux_divergence!(result, coefficient, u, coordsys)

Write `Σᵢ (a ∂ᵢuᵢ + uᵢ ∂ᵢa)` into `result`'s grid data. `result` is zeroed first.
Derivatives are spectral; only the products are formed pointwise on the grid, so
the result matches the hand-written `a*div(u) + dot(u, grad(a))` to round-off.

A constant (`Number`) coefficient contributes only `a ∂ᵢuᵢ` — `∂ᵢa` is
identically zero — so it reduces exactly to `a*div(u)`.
"""
function _accumulate_conservative_flux_divergence!(result::ScalarField, coefficient,
                                                   u::VectorField, coordsys)
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)
    result_data === nothing && throw(ArgumentError(
        "div(a*u): result field $(result.name) has no grid data to accumulate into."))
    _fill_zeros!(result_data)

    coefficient_data = nothing
    if isa(coefficient, ScalarField)
        ensure_layout!(coefficient, :g)
        coefficient_data = get_grid_data(coefficient)
        coefficient_data === nothing && throw(ArgumentError(
            "div(a*u): coefficient field $(coefficient.name) has no grid data."))
        if size(coefficient_data) != size(result_data)
            throw(ArgumentError(
                "div(a*u): coefficient $(coefficient.name) has grid shape " *
                "$(size(coefficient_data)) but $(u.name) has $(size(result_data)). " *
                "The coefficient must live on the same grid as the field it multiplies."))
        end
    end

    for (i, coord_name) in enumerate(coordsys.names)
        coord = coordsys[coord_name]
        component = u.components[i]

        # a ∂ᵢuᵢ
        component_derivative = evaluate_differentiate(Differentiate(component, coord, 1), :g)
        if coefficient_data === nothing
            result_data .+= coefficient .* get_grid_data(component_derivative)
        else
            result_data .+= coefficient_data .* get_grid_data(component_derivative)

            # uᵢ ∂ᵢa (identically zero for a constant coefficient, hence the branch)
            coefficient_derivative = evaluate_differentiate(
                Differentiate(coefficient, coord, 1), :g)
            ensure_layout!(component, :g)
            result_data .+= get_grid_data(component) .*
                            get_grid_data(coefficient_derivative)
        end
    end

    result.current_layout = :g
    return result
end

"""
    _evaluate_variable_coefficient_divergence(coeff_expr, grad_op, layout)

`∇·(a ∇u)` via the product rule `Σₖ (a ∂ₖ²u + ∂ₖa ∂ₖu)`. Coordinates are taken
from the gradient's own coordinate system, matching `evaluate_gradient` and
`_evaluate_vector_divergence`, so a coordinate with no basis simply contributes
nothing (its derivatives are zero).

Throws `ArgumentError` when the coefficient is not scalar-valued or when `u` is
neither a `ScalarField` nor a `VectorField`.
"""
function _evaluate_variable_coefficient_divergence(coeff_expr, grad_op::Gradient, layout::Symbol)
    coefficient = _eval_operand(coeff_expr, :g)
    if !(isa(coefficient, ScalarField) || isa(coefficient, Number))
        throw(ArgumentError(
            "div(a*grad(u)) requires a SCALAR coefficient `a` — a ScalarField, a number, " *
            "or an expression evaluating to one. Got $(typeof(coeff_expr)) evaluating to " *
            "$(typeof(coefficient)), which is not scalar-valued; a vector or tensor " *
            "coefficient has no product-rule expansion here, so write that term out " *
            "explicitly (for example as div() of an explicitly assembled flux vector)."))
    end

    # An expression coefficient can come back in a rotating derivative buffer,
    # which is recycled after `_DERIV_RESULT_POOL_SIZE` further derivatives —
    # and the loops below issue three per coordinate per component, which a 3D
    # vector `u` exceeds. Pin it into a checked-out field, which is never handed
    # out again, so `a` cannot change under us mid-accumulation.
    if isa(coefficient, ScalarField) && !isa(coeff_expr, ScalarField)
        pinned = checkout_or_alloc(coefficient.bases, coefficient.dtype, coefficient.dist)
        copy_field_data!(pinned, coefficient)
        pinned.current_layout = coefficient.current_layout
        pinned.name = coefficient.name
        coefficient = pinned
    end

    field = grad_op.operand
    coordsys = grad_op.coordsys

    if isa(field, ScalarField)
        result = _divergence_result_field(field, "div_coeff_grad_$(field.name)", :g)
        _accumulate_coeff_gradient_divergence!(result, coefficient, field, coordsys)
        ensure_layout!(result, layout)
        return result

    elseif isa(field, VectorField)
        # ∇·(a∇u) for vector u: the identity applies to each component separately.
        result = VectorField(field.dist, field.coordsys,
                             "div_coeff_grad_$(field.name)", field.bases, field.dtype)
        for (j, component) in enumerate(field.components)
            component_result = _divergence_result_field(
                component, "div_coeff_grad_$(component.name)", :g)
            _accumulate_coeff_gradient_divergence!(component_result, coefficient,
                                                   component, coordsys)
            ensure_layout!(component_result, layout)
            result.components[j] = component_result
        end
        return result
    end

    throw(ArgumentError(
        "div(a*grad(u)) is implemented for a ScalarField or VectorField `u`; got " *
        "grad($(typeof(field)))."))
end

"""
    _accumulate_coeff_gradient_divergence!(result, coefficient, u, coordsys)

Write `Σₖ (a ∂ₖ²u + ∂ₖa ∂ₖu)` into `result`'s grid data. `result` is zeroed
first. Derivatives are spectral; only the products are formed pointwise on the
grid, so the result matches the hand-written `a*lap(u) + Σₖ ∂ₖ(a)*∂ₖ(u)` to
round-off.
"""
function _accumulate_coeff_gradient_divergence!(result::ScalarField, coefficient,
                                                u::ScalarField, coordsys)
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)
    result_data === nothing && throw(ArgumentError(
        "div(a*grad(u)): result field $(result.name) has no grid data to accumulate into."))
    _fill_zeros!(result_data)

    coefficient_data = nothing
    if isa(coefficient, ScalarField)
        ensure_layout!(coefficient, :g)
        coefficient_data = get_grid_data(coefficient)
        coefficient_data === nothing && throw(ArgumentError(
            "div(a*grad(u)): coefficient field $(coefficient.name) has no grid data."))
        if size(coefficient_data) != size(result_data)
            throw(ArgumentError(
                "div(a*grad(u)): coefficient $(coefficient.name) has grid shape " *
                "$(size(coefficient_data)) but $(u.name) has $(size(result_data)). " *
                "The coefficient must live on the same grid as the differentiated field."))
        end
    end

    for coord_name in coordsys.names
        coord = coordsys[coord_name]

        # a ∂ₖ²u
        second_derivative = evaluate_differentiate(Differentiate(u, coord, 2), :g)
        if coefficient_data === nothing
            result_data .+= coefficient .* get_grid_data(second_derivative)
        else
            result_data .+= coefficient_data .* get_grid_data(second_derivative)

            # ∂ₖa ∂ₖu (identically zero for a constant coefficient)
            coefficient_derivative = evaluate_differentiate(
                Differentiate(coefficient, coord, 1), :g)
            first_derivative = evaluate_differentiate(Differentiate(u, coord, 1), :g)
            result_data .+= get_grid_data(coefficient_derivative) .*
                            get_grid_data(first_derivative)
        end
    end

    result.current_layout = :g
    return result
end

# ============================================================================
# Differentiate Evaluation
# ============================================================================

"""
    evaluate_differentiate(diff_op, layout=:g) -> ScalarField | VectorField

Evaluate `∂ⁿf/∂(coord)ⁿ` along a single coordinate. `VectorField` operands are
differentiated component-wise; `ScalarField` operands are differentiated by
locating the basis whose `element_label` matches `coord` and dispatching to the
matching spectral kernel (Fourier / Chebyshev / Legendre).

Handles two degenerate cases without touching a kernel: `order == 0` returns a
copy (identity), and a coordinate absent from the operand's bases (a constant
dimension) returns a zeroed field. Note the result basis can differ from the
operand's — e.g. Chebyshev differentiation maps `ChebyshevT → ChebyshevU` — so
the result is built from the differentiated component's bases, not the operand's.
"""
# Rotating pool of derivative-result buffers, keyed by (bases, dtype). Reuses
# fields across calls instead of allocating a fresh ScalarField per derivative.
# Uses N=16 distinct buffers (vs the global FieldPool's single-buffer reuse that
# caused silent corruption — see step!), giving each of several simultaneously
# live derivative results (e.g. the components of a gradient) its own buffer.
# Must be ≥ ndim² so a full 3D vector gradient (3×3 = 9 components evaluated in
# one loop) never aliases buffer 1 with buffer 9 (8 was too small → T[3,3]
# overwrote T[1,1]).
const _DERIV_RESULT_POOL_SIZE = 16
const _DERIV_RESULT_POOL = Dict{Tuple, Vector{ScalarField}}()
const _DERIV_RESULT_IDX = Ref(0)

function _checkout_deriv_result!(bases::Tuple, dtype::DataType, dist)
    key = (hash(bases), dtype)
    bufs = get!(() -> Vector{ScalarField}(undef, _DERIV_RESULT_POOL_SIZE),
                _DERIV_RESULT_POOL, key)
    i = (_DERIV_RESULT_IDX[] % _DERIV_RESULT_POOL_SIZE) + 1
    _DERIV_RESULT_IDX[] += 1
    if !isassigned(bufs, i)
        bufs[i] = ScalarField(dist, "deriv_tmp", bases, dtype)
    end
    return bufs[i]
end

function evaluate_differentiate(diff_op::Differentiate, layout::Symbol=:g)
    operand = diff_op.operand
    coord = diff_op.coord
    order = diff_op.order

    # VectorField: differentiate each component, return VectorField
    if isa(operand, VectorField)
        diff_comps = [evaluate_differentiate(Differentiate(c, coord, order), layout)
                      for c in operand.components]
        # Create result with differentiated component bases (may differ from
        # original for Chebyshev: ChebyshevT → ChebyshevU after differentiation)
        result = VectorField(operand.dist, operand.coordsys,
                             "d$(order)_$(operand.name)_d$(coord.name)",
                             diff_comps[1].bases, operand.dtype)
        for (i, dc) in enumerate(diff_comps)
            copy_field_data!(result.components[i], dc)
            result.components[i].current_layout = dc.current_layout
        end
        return result
    end

    if !isa(operand, ScalarField)
        throw(ArgumentError(
            "Differentiation requires ScalarField or VectorField, got $(typeof(operand))"))
    end

    # Short-circuit for zero-order derivative (identity operation)
    if order == 0
        result = checkout_or_alloc(operand.bases, operand.dtype, operand.dist)
        copy_field_data!(result, operand)
        result.current_layout = operand.current_layout
        result.name = "d0_$(operand.name)"
        ensure_layout!(result, layout)
        return result
    end

    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        # Coordinate not present in bases (constant dimension): derivative is zero
        result = checkout_or_alloc(operand.bases, operand.dtype, operand.dist)
        copy_field_data!(result, operand)
        result.current_layout = operand.current_layout
        result.name = "d$(order)_$(operand.name)_d$(coord.name)$(order)"
        ensure_layout!(result, layout)

        # Zero out the data
        if layout == :g
            grid_data = get_grid_data(result)
            if grid_data !== nothing
                if isa(grid_data, PencilArrays.PencilArray)
                    fill!(parent(grid_data), zero(eltype(grid_data)))
                else
                    fill!(grid_data, zero(eltype(grid_data)))
                end
            end
        else
            coeff_data = get_coeff_data(result)
            if coeff_data !== nothing
                if isa(coeff_data, PencilArrays.PencilArray)
                    fill!(parent(coeff_data), zero(eltype(coeff_data)))
                else
                    fill!(coeff_data, zero(eltype(coeff_data)))
                end
            end
        end

        return result
    end

    basis = operand.bases[basis_index]
    result = _checkout_deriv_result!(operand.bases, operand.dtype, operand.dist)
    copy_field_data!(result, operand)
    result.current_layout = operand.current_layout
    result.name = "d$(order)_$(operand.name)_d$(coord.name)$(order)"

    # Apply differentiation based on basis type
    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        evaluate_fourier_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, ChebyshevT)
        evaluate_chebyshev_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, Legendre)
        evaluate_legendre_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, JacobiBasis)
        # ChebyshevU/ChebyshevV/Ultraspherical/generic Jacobi: nodal collocation
        # derivative (no dedicated spectral kernel). Matched after ChebyshevT and
        # Legendre, which have their own faster kernels above.
        evaluate_jacobi_collocation_derivative!(result, operand, basis_index, order, layout)
    else
        throw(ArgumentError(
            "Differentiation not implemented for basis type $(typeof(basis)). " *
            "Supported: RealFourier, ComplexFourier, ChebyshevT, Legendre, and other " *
            "JacobiBasis (ChebyshevU/ChebyshevV/Ultraspherical/Jacobi) via nodal collocation. " *
            "Check that the coordinate '$(coord.name)' has a valid basis assigned."))
    end

    return result
end
