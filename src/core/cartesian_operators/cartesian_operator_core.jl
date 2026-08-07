# Common Cartesian operator traits, component extraction, and tensor helpers.

# ============================================================================
# Abstract operator infrastructure for multiclass dispatch
# ============================================================================

"""
    AbstractLinearOperator

Base type for linear operators that support matrix representations.
"""
abstract type AbstractLinearOperator <: Operator end

"""
    OperatorConditions

Mixin trait for operators with layout conditions.
"""
abstract type OperatorConditions end

# ============================================================================
# CartesianComponent - Extract component from vector/tensor by coordinate
# ============================================================================

"""
    CartesianComponent <: AbstractLinearOperator

Extract a specific component of a vector or tensor field in Cartesian coordinates.

Following operators:3270-3412 CartesianComponent implementation.

# Arguments
- `operand`: Vector or tensor field
- `index`: Tensor index to extract from (0-based following standard)
- `comp`: Coordinate object specifying which component

# Example
```julia
coords = CartesianCoordinates("x", "y", "z")
u = VectorField(dist, coords, "u")
u_x = CartesianComponent(u, index=0, comp=coords["x"])  # x-component
```
"""
struct CartesianComponent <: AbstractLinearOperator
    operand::Operand
    index::Int           # Tensor index (0-based like standard)
    comp::Coordinate     # Coordinate for component selection
    coordsys::CoordinateSystem
    comp_subaxis::Int    # Component subaxis within coordsys
    tensorsig::Tuple     # Output tensor signature

    function CartesianComponent(operand::Operand; index::Int=0, comp::Coordinate)
        # Get the coordinate system at the specified tensor index
        if !isa(operand, VectorField) && !isa(operand, TensorField)
            throw(ArgumentError("CartesianComponent requires a VectorField or TensorField"))
        end

        # Get tensorsig from operand
        tensorsig = get_tensorsig(operand)

        if index < 0 || index >= length(tensorsig)
            throw(ArgumentError("Invalid tensor index $index for operand with $(length(tensorsig)) tensor indices"))
        end

        coordsys = tensorsig[index + 1]  # 1-based Julia indexing

        if !isa(coordsys, CartesianCoordinates) && !isa(coordsys, Coordinate)
            throw(ArgumentError("CartesianComponent only works with CartesianCoordinates"))
        end

        # Find component subaxis
        comp_subaxis = -1
        if isa(coordsys, CartesianCoordinates)
            for (i, c) in enumerate(coordsys.coords)
                if c == comp || c.name == comp.name
                    comp_subaxis = i - 1  # 0-based
                    break
                end
            end
        elseif isa(coordsys, Coordinate)
            if coordsys == comp || coordsys.name == comp.name
                comp_subaxis = 0
            end
        end

        if comp_subaxis < 0
            throw(ArgumentError("Component coordinate $(comp.name) not found in coordinate system"))
        end

        # Output tensorsig removes the extracted index
        out_tensorsig = (tensorsig[1:index]..., tensorsig[index+2:end]...)

        new(operand, index, comp, coordsys, comp_subaxis, out_tensorsig)
    end
end

const _CartesianComponent_constructor = CartesianComponent

# Convenience constructor
function cartesian_component(operand::Operand; index::Int=0, comp::Coordinate)
    return CartesianComponent(operand; index=index, comp=comp)
end

# Register operator
register_operator_alias!(cartesian_component, "cartesian_component", "cart_comp")
register_operator_parseable!(cartesian_component, "cartesian_component", "cart_comp")

"""
    get_tensorsig(operand)

Get tensor signature from an operand.
"""
function get_tensorsig(operand)
    if hasfield(typeof(operand), :tensorsig)
        return operand.tensorsig
    elseif isa(operand, VectorField)
        return (operand.coordsys,)
    elseif isa(operand, TensorField)
        return (operand.coordsys, operand.coordsys)
    else
        return ()
    end
end

# ============================================================================
# Matrix operation methods for CartesianComponent
# ============================================================================

"""
    matrix_dependence(op::CartesianComponent, vars...)

Determine which variables the operator matrix depends on.
Component extraction doesn't add new dependencies beyond the operand's own dependencies.
"""
function matrix_dependence(op::CartesianComponent, vars...)
    # Dispatch to operand's matrix_dependence (falls back to Operand method if no specific method)
    return matrix_dependence(op.operand, vars...)
end

"""
    matrix_coupling(op::CartesianComponent, vars...)

Determine which variables couple through the operator.
Component extraction doesn't add new coupling beyond the operand's own coupling.
"""
function matrix_coupling(op::CartesianComponent, vars...)
    # Dispatch to operand's matrix_coupling (falls back to Operand method if no specific method)
    return matrix_coupling(op.operand, vars...)
end

"""
    subproblem_matrix(op::CartesianComponent, subproblem)

Build operator matrix for a specific subproblem.
"""
function subproblem_matrix(op::CartesianComponent, subproblem)
    # Build selection matrix that extracts the specified component
    # The matrix has shape (output_size, input_size)

    coordsys = op.coordsys
    dim = isa(coordsys, CartesianCoordinates) ? coordsys.dim : 1

    # Build identity matrix for scalar part
    scalar_size = get_scalar_size(op.operand, subproblem)
    I_scalar = sparse(I, scalar_size, scalar_size)

    if isa(op.operand, TensorField)
        # Build tensor index selection matrix for rank-2 tensor.
        # Components are flattened column-major: idx = (j-1)*dim + i.
        comp_idx = op.comp_subaxis + 1
        selector = zeros(Float64, dim, dim * dim)
        if op.index == 0
            # Select row comp_idx -> output components correspond to column index.
            for j in 1:dim
                idx_in = (j - 1) * dim + comp_idx
                selector[j, idx_in] = 1.0
            end
        elseif op.index == 1
            # Select column comp_idx -> output components correspond to row index.
            for i in 1:dim
                idx_in = (comp_idx - 1) * dim + i
                selector[i, idx_in] = 1.0
            end
        else
            throw(ArgumentError("Tensor index $(op.index) out of bounds for rank-2 tensor"))
        end
        return kron(sparse(selector), I_scalar)
    end

    # Vector case: select one component from dim components.
    index_factor = zeros(Float64, 1, dim)
    index_factor[1, op.comp_subaxis + 1] = 1.0  # 1-based Julia indexing
    return kron(sparse(index_factor), I_scalar)
end

"""
    get_scalar_size(operand, subproblem) -> Int

Degrees of freedom of the SCALAR part of `operand` — the block size the
identity/selection matrices in `subproblem_matrix` are built from. For a vector
or tensor operand that is ONE component, not the sum over components (which is
what `field_dofs` returns for those types).

The unit is the GRID size, `prod(basis.meta.size)` — the size the Cartesian
operator matrices are built at. `build_operator_differentiation_matrix` on a
`RealFourier(16) x RealFourier(16)` field returns a 256x256 matrix, not the
144x144 its coefficient array would suggest, because a `RealFourier` axis stores
only the `rfft_len` half-spectrum. Blocks assembled here have to compose with
those matrices.

WHY THIS IS NO LONGER A `hasfield` CHAIN. The previous body tried
`hasfield(typeof(operand), :buffers)` to reach `length(get_coeff_data(operand))`
BEFORE falling back to the basis product. No field type has a `:buffers` field —
`TransposableField` is the only `Operand` that does, and it has no
`get_coeff_data` method — so those two branches never ran. Which is the only
reason the function was right: had they run they would have returned 144 where
the surrounding `kron` needs 256, and the selection matrix would not have
composed. A dead branch that is also wrong is worse than either alone, because
its deadness is what makes the code work and nothing says so.
"""
get_scalar_size(operand::ScalarField, subproblem) = _bases_dof_product(operand.bases)

get_scalar_size(operand::VectorField, subproblem) =
    isempty(operand.components) ? _bases_dof_product(operand.bases) :
                                  get_scalar_size(operand.components[1], subproblem)

get_scalar_size(operand::TensorField, subproblem) =
    isempty(operand.components) ? _bases_dof_product(operand.bases) :
                                  get_scalar_size(operand.components[1], subproblem)

# Operators and non-field operands carry no scalar block of their own.
get_scalar_size(operand, subproblem) = 1

"""Grid-space DOF count of a basis tuple: the block size the Cartesian operator
matrices are assembled at. Distinct from `field_dofs`, which counts COEFFICIENT
DOFs and is smaller on any `RealFourier` axis."""
_bases_dof_product(bases) = isempty(bases) ? 1 : prod(b.meta.size for b in bases)

# ============================================================================
# Layout condition check/enforce for CartesianComponent
# ============================================================================

"""
    check_conditions(op::CartesianComponent)

Check that operands are in a proper layout for operation.

CartesianComponent extraction works in both grid and coefficient layouts,
so this always returns true. The layout of the output matches the operand.
"""
function check_conditions(op::CartesianComponent)
    # Component extraction is purely algebraic - it just selects a subarray
    # from the operand's data. Works in any layout.
    operand = op.operand

    # Vector/tensor components should share a layout
    if isa(operand, VectorField)
        layouts = [comp.current_layout for comp in operand.components if comp.current_layout !== nothing]
        return isempty(layouts) || length(Set(layouts)) <= 1
    elseif isa(operand, TensorField)
        layouts = [comp.current_layout for comp in operand.components if comp.current_layout !== nothing]
        return isempty(layouts) || length(Set(layouts)) <= 1
    end

    # Verify operand has valid data in at least one layout (scalar fields)
    layout = operand_layout(operand)
    if layout !== nothing
        if layout == :g
            return get_grid_data(operand) !== nothing
        elseif layout == :c
            return get_coeff_data(operand) !== nothing
        end
    end

    # For non-field operands or missing layout info, assume OK
    return true
end

"""
    enforce_conditions(op::CartesianComponent)

Require operands to be in a proper layout.

Since CartesianComponent works in any layout, this only ensures
the operand has valid data allocated in its current layout.
"""
function enforce_conditions(op::CartesianComponent)
    operand = op.operand

    # Ensure operand has data in its current layout
    if isa(operand, VectorField)
        if isempty(operand.components)
            return nothing
        end
        layout = operand.components[1].current_layout
        ensure_layout!(operand, layout)
    elseif isa(operand, TensorField)
        if isempty(operand.components)
            return nothing
        end
        layout = operand.components[1, 1].current_layout
        ensure_layout!(operand, layout)
    elseif operand isa ScalarField
        # This branch used to be gated on `hasfield(typeof(operand), :buffers)`,
        # which `ScalarField` answers FALSE to — `TransposableField` is the only
        # `Operand` with a `:buffers` field — so it never ran and a scalar
        # operand got no enforcement at all. `ensure_layout!` is idempotent when
        # the data is already there, so making it live only adds the repair the
        # branch was written to perform.
        layout = operand.current_layout
        if layout == :g && get_grid_data(operand) === nothing
            get_coeff_data(operand) === nothing || ensure_layout!(operand, :g)
        elseif layout == :c && get_coeff_data(operand) === nothing
            get_grid_data(operand) === nothing || ensure_layout!(operand, :c)
        end
    end

    return nothing
end

"""
    operate(op::CartesianComponent, out)

Perform the component extraction operation.
Following operators:3405-3411 CartesianComponent.operate.
"""
function operate(op::CartesianComponent, out)
    operand = op.operand

    # Prefer the output layout if provided, otherwise infer from operand
    layout = if operand_layout(out) !== nothing
        operand_layout(out)
    elseif isa(operand, VectorField)
        operand.components[op.comp_subaxis + 1].current_layout
    elseif isa(operand, TensorField)
        operand.components[1, 1].current_layout
    elseif operand_layout(operand) !== nothing
        operand_layout(operand)
    else
        :g
    end
    if layout === nothing
        layout = :g
    end

    # Extract component based on operand type
    if isa(operand, VectorField)
        # For vector field, extract the component field
        comp_idx = op.comp_subaxis + 1  # 1-based Julia indexing
        comp_field = operand.components[comp_idx]

        ensure_layout!(comp_field, layout)
        if out isa ScalarField
            ensure_layout!(out, layout)
        end
        if layout == :g
            copyto!(get_grid_data(out), get_grid_data(comp_field))
        else
            copyto!(get_coeff_data(out), get_coeff_data(comp_field))
        end
    elseif isa(operand, TensorField)
        # For tensor field, extract along the specified tensor index
        if !isa(out, VectorField)
            throw(ArgumentError("Tensor component extraction requires VectorField output"))
        end
        extract_tensor_component!(out, operand, op.index, op.comp_subaxis, layout)
    end

    return out
end

"""
    extract_tensor_component!(out, tensor, index, comp_subaxis, layout)

Extract a component from a tensor field along specified index.
"""
function extract_tensor_component!(out::VectorField, tensor::TensorField, index::Int, comp_subaxis::Int, layout::Symbol)
    comp_idx = comp_subaxis + 1

    ensure_layout!(tensor, layout)
    ensure_layout!(out, layout)

    n1, n2 = size(tensor.components)
    if index == 0
        if length(out.components) != n2
            throw(ArgumentError("Output VectorField has $(length(out.components)) components, expected $n2"))
        end
        for j in 1:n2
            src = tensor.components[comp_idx, j]
            dst = out.components[j]
            ensure_layout!(src, layout)
            ensure_layout!(dst, layout)
            if layout == :g
                copyto!(get_grid_data(dst), get_grid_data(src))
            else
                copyto!(get_coeff_data(dst), get_coeff_data(src))
            end
        end
    elseif index == 1
        if length(out.components) != n1
            throw(ArgumentError("Output VectorField has $(length(out.components)) components, expected $n1"))
        end
        for i in 1:n1
            src = tensor.components[i, comp_idx]
            dst = out.components[i]
            ensure_layout!(src, layout)
            ensure_layout!(dst, layout)
            if layout == :g
                copyto!(get_grid_data(dst), get_grid_data(src))
            else
                copyto!(get_coeff_data(dst), get_coeff_data(src))
            end
        end
    else
        throw(ArgumentError("Tensor index $index out of bounds for rank-2 tensor"))
    end
    return out
end

# ============================================================================
# Evaluate function for CartesianComponent
# ============================================================================

"""
    evaluate_cartesian_component(op::CartesianComponent, layout::Symbol=:g)

Evaluate CartesianComponent operator.

!!! note "Return Value Semantics"
    For VectorField operands, this returns a **reference** to the component
    (not a copy). Modifications to the returned ScalarField will affect
    the original VectorField. For TensorField operands, data is copied
    to a new VectorField.
"""
function evaluate_cartesian_component(op::CartesianComponent, layout::Symbol=:g)
    operand = op.operand

    if isa(operand, VectorField)
        # Extract the component field directly (returns reference, not copy)
        comp_idx = op.comp_subaxis + 1  # 1-based Julia indexing
        result = operand.components[comp_idx]

        if layout == :g
            ensure_layout!(result, :g)
        else
            ensure_layout!(result, :c)
        end

        return result
    elseif isa(operand, TensorField)
        # Extract a row or column from the tensor to get a VectorField
        # index determines which tensor axis to extract from (0-based)
        # comp_subaxis determines which component along that axis (0-based)
        comp_idx = op.comp_subaxis + 1  # 1-based Julia indexing

        # Create result VectorField with the remaining coordinate system
        result = VectorField(operand.dist, operand.coordsys,
                            "$(operand.name)_comp$(comp_idx)",
                            operand.bases, operand.dtype)

        # Extract components based on which index we're extracting
        # Copy data to ensure result is independent of operand
        if op.index == 0
            # Extracting first index: T[comp_idx, :] -> vector
            for j in 1:size(operand.components, 2)
                src = operand.components[comp_idx, j]
                dst = result.components[j]
                ensure_layout!(src, layout)
                ensure_layout!(dst, layout)
                if layout == :g
                    copyto!(get_grid_data(dst), get_grid_data(src))
                else
                    copyto!(get_coeff_data(dst), get_coeff_data(src))
                end
            end
        else
            # Extracting second index: T[:, comp_idx] -> vector
            for i in 1:size(operand.components, 1)
                src = operand.components[i, comp_idx]
                dst = result.components[i]
                ensure_layout!(src, layout)
                ensure_layout!(dst, layout)
                if layout == :g
                    copyto!(get_grid_data(dst), get_grid_data(src))
                else
                    copyto!(get_coeff_data(dst), get_coeff_data(src))
                end
            end
        end

        return result
    else
        throw(ArgumentError("CartesianComponent requires VectorField or TensorField"))
    end
end
