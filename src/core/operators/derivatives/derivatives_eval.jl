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
    evaluate_divergence(div_op, layout=:g) -> ScalarField

Evaluate `∇·u` for a `VectorField`, lowering rank by one: sums `∂uᵢ/∂xᵢ` over
all coordinates into a single scalar result. Accumulation is done directly on
the result's field data (not via the symbolic `+` tree) to avoid building
intermediate operator nodes. The result buffer is taken from the field pool and
zero-initialized in the requested `layout`; PencilArray buffers are zeroed via
their `parent`. Throws `ArgumentError` for non-vector operands.
"""
function evaluate_divergence(div_op::Divergence, layout::Symbol=:g)
    operand = div_op.operand

    if isa(operand, VectorField)
        # Sum partial derivatives of components
        coordsys = operand.coordsys

        # Create result field from pool, then copy data to preserve PencilArray structure
        result = checkout_or_alloc(operand.components[1].bases, operand.components[1].dtype, operand.components[1].dist)
        copy_field_data!(result, operand.components[1])
        result.current_layout = operand.components[1].current_layout
        result.name = "div_$(operand.name)"

        # Initialize result to zero — ensure data is allocated even if copy didn't provide it
        ensure_layout!(result, layout)
        if layout == :g
            grid_data = get_grid_data(result)
            if grid_data === nothing
                # Allocate grid data if not present (copy may not have provided it)
                set_grid_data!(result, zeros(eltype(get_grid_data(operand.components[1])),
                                             size(get_grid_data(operand.components[1]))))
            elseif isa(grid_data, PencilArrays.PencilArray)
                fill!(parent(grid_data), zero(eltype(grid_data)))
            else
                fill!(grid_data, zero(eltype(grid_data)))
            end
        else
            coeff_data = get_coeff_data(result)
            if coeff_data === nothing
                set_coeff_data!(result, zeros(eltype(get_coeff_data(operand.components[1])),
                                              size(get_coeff_data(operand.components[1]))))
            elseif isa(coeff_data, PencilArrays.PencilArray)
                fill!(parent(coeff_data), zero(eltype(coeff_data)))
            else
                fill!(coeff_data, zero(eltype(coeff_data)))
            end
        end

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
    else
        throw(ArgumentError("Divergence not implemented for operand type $(typeof(operand))"))
    end
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
# Uses N=8 distinct buffers (vs the global FieldPool's single-buffer reuse that
# caused silent corruption — see step!), giving each of several simultaneously
# live derivative results (e.g. the components of a gradient) its own buffer.
const _DERIV_RESULT_POOL_SIZE = 8
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
