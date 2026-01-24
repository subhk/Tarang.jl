"""
Cartesian coordinate system specific operators

This module implements Cartesian-specific operator variants following the
MultiClass dispatch pattern. Each operator has:
- CartesianX version for Cartesian coordinates
- DirectProductX version for direct product coordinate systems
- Matrix operation methods for implicit solvers
- Layout condition check/enforce patterns

Key features:
- CartesianComponent: Extract vector component by coordinate
- CartesianGradient: Gradient in Cartesian coordinates
- CartesianDivergence: Divergence in Cartesian coordinates
- CartesianCurl: Curl in 3D Cartesian coordinates
- CartesianLaplacian: Laplacian in Cartesian coordinates
- CartesianTrace: Trace of tensor in Cartesian coordinates
- CartesianSkew: Skew operation for 2D vectors
"""

using LinearAlgebra
using SparseArrays

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
    get_scalar_size(operand, subproblem)

Get the size of the scalar portion of an operand for matrix assembly.
"""
function get_scalar_size(operand, subproblem)
    if hasfield(typeof(operand), :components) && !isempty(operand.components)
        return get_scalar_size(operand.components[1], subproblem)
    elseif hasfield(typeof(operand), :buffers) && get_coeff_data(operand) !== nothing
        return length(get_coeff_data(operand))
    elseif hasfield(typeof(operand), :buffers) && get_grid_data(operand) !== nothing
        return length(get_grid_data(operand))
    elseif hasfield(typeof(operand), :bases)
        size = 1
        for basis in operand.bases
            if basis !== nothing
                size *= basis.meta.size
            end
        end
        return size
    else
        return 1
    end
end

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
    if hasfield(typeof(operand), :current_layout)
        layout = operand.current_layout
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
    elseif hasfield(typeof(operand), :current_layout) && hasfield(typeof(operand), :buffers)
        layout = operand.current_layout
        if layout == :g && get_grid_data(operand) === nothing
            # Need to transform from coefficient space to grid space
            if hasfield(typeof(operand), :buffers) && get_coeff_data(operand) !== nothing
                # Trigger transform - this depends on the field's transform implementation
                ensure_layout!(operand, :g)
            end
        elseif layout == :c && get_coeff_data(operand) === nothing
            if hasfield(typeof(operand), :buffers) && get_grid_data(operand) !== nothing
                ensure_layout!(operand, :c)
            end
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
    layout = if hasfield(typeof(out), :current_layout)
        out.current_layout
    elseif isa(operand, VectorField)
        operand.components[op.comp_subaxis + 1].current_layout
    elseif isa(operand, TensorField)
        operand.components[1, 1].current_layout
    elseif hasfield(typeof(operand), :current_layout)
        operand.current_layout
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
        if hasfield(typeof(out), :current_layout)
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

# ============================================================================
# CartesianGradient - Gradient in Cartesian coordinates
# Following operators:2340-2412 CartesianGradient
# ============================================================================

"""
    CartesianGradient <: AbstractLinearOperator

Gradient operator specialized for Cartesian coordinates.

For scalar field f, gradient is a vector field:
∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)

For vector field u, gradient is a tensor field:
(∇u)_ij = ∂u_j/∂x_i  (first index = derivative direction)
"""
struct CartesianGradient <: AbstractLinearOperator
    operand::Operand
    coordsys::CartesianCoordinates
    args::Vector{Any}  # Partial derivative operators for each coordinate

    function CartesianGradient(operand::Operand, coordsys::CoordinateSystem)
        # Handle single coordinate case
        if isa(coordsys, Coordinate)
            coordsys = CartesianCoordinates(coordsys.name)
        end

        if !isa(coordsys, CartesianCoordinates)
            throw(ArgumentError("CartesianGradient requires CartesianCoordinates"))
        end

        # Build partial derivative operators for each coordinate
        args = [Differentiate(operand, coord, 1) for coord in coordsys.coords]

        new(operand, coordsys, args)
    end
end

# ============================================================================
# Matrix operations for CartesianGradient
# ============================================================================

function matrix_dependence(op::CartesianGradient, vars...)
    # Linear operator; dependence follows the operand
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::CartesianGradient, vars...)
    result = falses(length(vars))
    operand = op.operand

    if isa(operand, VectorField)
        # For vector operand, check coupling of each component against vars
        for comp in operand.components
            for (i, var) in enumerate(vars)
                if comp === var ||
                   (hasfield(typeof(comp), :name) && hasfield(typeof(var), :name) &&
                    comp.name == var.name)
                    result[i] = true
                end
            end
        end
        # Also check the VectorField itself
        for (i, var) in enumerate(vars)
            if operand === var ||
               (hasfield(typeof(operand), :name) && hasfield(typeof(var), :name) &&
                operand.name == var.name)
                result[i] = true
            end
        end
    else
        # Scalar operand: delegate to derivative args
        for arg in op.args
            arg_coupling = matrix_coupling(arg, vars...)
            result .|= arg_coupling
        end
    end

    return result
end

function subproblem_matrix(op::CartesianGradient, subproblem)
    """Build operator matrix for a specific subproblem."""
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # Scalar → Vector: stack differentiation matrices vertically
        matrices = SparseMatrixCSC[]
        for arg in op.args
            mat = expression_matrices(arg, subproblem, [operand])
            if haskey(mat, operand)
                push!(matrices, mat[operand])
            end
        end

        if isempty(matrices)
            return spzeros(Float64, 0, 0)
        end

        return vcat(matrices...)

    elseif isa(operand, VectorField)
        # Vector → Tensor: T[i,j] = ∂u_j/∂x_i
        # Build block matrix with dim² row blocks and dim column blocks
        dim = coordsys.dim
        comps = operand.components
        n_per = field_dofs(comps[1])
        zero_block = spzeros(Float64, n_per, n_per)

        rows = SparseMatrixCSC[]
        for i in 1:dim  # derivative direction
            for j in 1:dim  # vector component
                blocks = SparseMatrixCSC[]
                for k in 1:dim
                    if k == j
                        D = build_operator_differentiation_matrix(comps[j], coordsys.coords[i], 1)
                        push!(blocks, D === nothing ? zero_block : D)
                    else
                        push!(blocks, zero_block)
                    end
                end
                push!(rows, hcat(blocks...))
            end
        end

        return vcat(rows...)

    else
        return spzeros(Float64, 0, 0)
    end
end

function check_conditions(op::CartesianGradient)
    """Check that operands are in a proper layout."""
    operand = op.operand

    if isa(operand, VectorField)
        # For vector operand, check that all components are in a consistent layout
        layouts = Symbol[]
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout) && comp.current_layout !== nothing
                push!(layouts, comp.current_layout)
            end
        end
        return length(Set(layouts)) <= 1
    else
        # Scalar operand: check layout consistency across derivative args
        layouts = [arg.operand.current_layout for arg in op.args
                   if hasfield(typeof(arg.operand), :current_layout)]
        return length(Set(layouts)) <= 1
    end
end

function enforce_conditions(op::CartesianGradient)
    """Require operands to be in coefficient layout."""
    operand = op.operand

    if isa(operand, VectorField)
        # For vector operand, ensure all components are in coefficient layout
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout)
                ensure_layout!(comp, :c)
            end
        end
    else
        for arg in op.args
            if hasfield(typeof(arg), :operand) && hasfield(typeof(arg.operand), :current_layout)
                ensure_layout!(arg.operand, :c)
            end
        end
    end
end

# ============================================================================
# CartesianDivergence - Divergence in Cartesian coordinates
# Following operators:3438-3495 CartesianDivergence
# ============================================================================

"""
    CartesianDivergence <: AbstractLinearOperator

Divergence operator specialized for Cartesian coordinates.

Following operators:3438-3495 CartesianDivergence implementation.

For vector field u = (u_x, u_y, u_z):
∇·u = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
"""
struct CartesianDivergence <: AbstractLinearOperator
    operand::Operand
    index::Int
    coordsys::CoordinateSystem
    arg::Any  # Sum of component derivatives

    function CartesianDivergence(operand::Operand; index::Int=0)
        if !isa(operand, VectorField)
            throw(ArgumentError("CartesianDivergence requires a VectorField"))
        end

        tensorsig = get_tensorsig(operand)
        coordsys = tensorsig[index + 1]

        # Handle single coordinate case
        if isa(coordsys, Coordinate)
            coordsys = CartesianCoordinates(coordsys.name)
        end

        if !isa(coordsys, CartesianCoordinates)
            throw(ArgumentError("CartesianDivergence requires CartesianCoordinates"))
        end

        # Build sum of component derivatives
        # ∇·u = Σ ∂u_i/∂x_i
        comps = Any[]
        for coord in coordsys.coords
            comp_op = CartesianComponent(operand; index=index, comp=coord)
            deriv_op = Differentiate(comp_op, coord, 1)
            push!(comps, deriv_op)
        end

        # Sum of derivatives (conceptually)
        arg = comps  # Store as array; evaluate sums them

        new(operand, index, coordsys, arg)
    end
end

function matrix_dependence(op::CartesianDivergence, vars...)
    # Linear operator; dependence follows the operand
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::CartesianDivergence, vars...)
    if isa(op.arg, AbstractArray)
        result = falses(length(vars))
        for comp in op.arg
            if hasmethod(matrix_coupling, Tuple{typeof(comp), Vararg})
                result .|= matrix_coupling(comp, vars...)
            end
        end
        return result
    end
    return matrix_coupling(op.arg, vars...)
end

function subproblem_matrix(op::CartesianDivergence, subproblem)
    """Build operator matrix for a specific subproblem."""
    operand = op.operand
    if !isa(operand, VectorField)
        return spzeros(Float64, 0, 0)
    end

    coordsys = op.coordsys
    n_comp = length(operand.components)
    n_per_comp = n_comp > 0 ? field_dofs(operand.components[1]) : 0

    blocks = SparseMatrixCSC[]
    for (i, coord) in enumerate(coordsys.coords)
        comp = operand.components[i]
        D = build_operator_differentiation_matrix(comp, coord, 1)
        push!(blocks, D === nothing ? spzeros(Float64, n_per_comp, n_per_comp) : D)
    end

    return isempty(blocks) ? spzeros(Float64, 0, 0) : hcat(blocks...)
end

function check_conditions(op::CartesianDivergence)
    """Check that operands are in a proper layout for divergence computation."""
    operand = op.operand

    # For VectorField, check that all components are in a consistent layout
    if isa(operand, VectorField)
        layouts = Symbol[]
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout) && comp.current_layout !== nothing
                push!(layouts, comp.current_layout)
            end
        end
        # All components should be in the same layout
        return length(Set(layouts)) <= 1
    end

    # For other operand types, check if they have valid data
    if hasfield(typeof(operand), :current_layout)
        layout = operand.current_layout
        if layout == :g
            return hasfield(typeof(operand), :buffers) && get_grid_data(operand) !== nothing
        elseif layout == :c
            return hasfield(typeof(operand), :buffers) && get_coeff_data(operand) !== nothing
        end
    end

    return true
end

function enforce_conditions(op::CartesianDivergence)
    """Ensure operands are in coefficient layout for differentiation."""
    operand = op.operand

    # For VectorField, ensure all components are in coefficient layout
    if isa(operand, VectorField)
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout)
                ensure_layout!(comp, :c)
            end
        end
    elseif hasfield(typeof(operand), :current_layout)
        ensure_layout!(operand, :c)
    end

    return nothing
end

# ============================================================================
# CartesianCurl - Curl in 3D Cartesian coordinates
# Following operators:3689-3749 CartesianCurl
# ============================================================================

"""
    CartesianCurl <: AbstractLinearOperator

Curl operator specialized for 3D Cartesian coordinates.

Following operators:3689-3749 CartesianCurl implementation.

For vector field u = (u_x, u_y, u_z):
∇×u = (∂u_z/∂y - ∂u_y/∂z, ∂u_x/∂z - ∂u_z/∂x, ∂u_y/∂x - ∂u_x/∂y)

Note: For 2D, use skew gradient instead of curl.
"""
struct CartesianCurl <: AbstractLinearOperator
    operand::Operand
    index::Int
    coordsys::CartesianCoordinates
    arg::Any  # Constructed curl expression

    function CartesianCurl(operand::Operand; index::Int=0)
        if !isa(operand, VectorField)
            throw(ArgumentError("CartesianCurl requires a VectorField"))
        end

        tensorsig = get_tensorsig(operand)
        coordsys = tensorsig[index + 1]

        if !isa(coordsys, CartesianCoordinates)
            throw(ArgumentError("CartesianCurl requires CartesianCoordinates"))
        end

        if coordsys.dim != 3
            throw(ArgumentError("CartesianCurl is only implemented for 3D vector fields. For 2D, use skew gradient."))
        end

        # Extract coordinates
        cx, cy, cz = coordsys.coords

        # Get vector components
        # comps[i] extracts component i
        # Following: curl = ex*(∂uz/∂y - ∂uy/∂z) + ey*(∂ux/∂z - ∂uz/∂x) + ez*(∂uy/∂x - ∂ux/∂y)

        # Store the component expressions for evaluation
        # x-component: ∂uz/∂y - ∂uy/∂z
        # y-component: ∂ux/∂z - ∂uz/∂x
        # z-component: ∂uy/∂x - ∂ux/∂y

        # Build actual derivative operators for curl computation
        # curl = (∂uz/∂y - ∂uy/∂z, ∂ux/∂z - ∂uz/∂x, ∂uy/∂x - ∂ux/∂y)
        # For each output component, store (positive_deriv, negative_deriv)

        comp_x = CartesianComponent(operand; index=index, comp=cx)
        comp_y = CartesianComponent(operand; index=index, comp=cy)
        comp_z = CartesianComponent(operand; index=index, comp=cz)

        # x-component: ∂uz/∂y - ∂uy/∂z
        curl_x = (Differentiate(comp_z, cy, 1), Differentiate(comp_y, cz, 1))
        # y-component: ∂ux/∂z - ∂uz/∂x
        curl_y = (Differentiate(comp_x, cz, 1), Differentiate(comp_z, cx, 1))
        # z-component: ∂uy/∂x - ∂ux/∂y
        curl_z = (Differentiate(comp_y, cx, 1), Differentiate(comp_x, cy, 1))

        arg = Dict(
            :x => curl_x,
            :y => curl_y,
            :z => curl_z,
            :components => (comp_x, comp_y, comp_z),
            :right_handed => coordsys.right_handed
        )

        new(operand, index, coordsys, arg)
    end
end

function matrix_dependence(op::CartesianCurl, vars...)
    """Determine which variables the curl operator matrix depends on."""
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::CartesianCurl, vars...)
    """Determine which variables couple through the curl operator."""
    result = falses(length(vars))

    # Check coupling from all derivative operators
    for key in (:x, :y, :z)
        pos_deriv, neg_deriv = op.arg[key]
        if hasmethod(matrix_coupling, Tuple{typeof(pos_deriv), Vararg})
            result .|= matrix_coupling(pos_deriv, vars...)
        end
        if hasmethod(matrix_coupling, Tuple{typeof(neg_deriv), Vararg})
            result .|= matrix_coupling(neg_deriv, vars...)
        end
    end

    return result
end

function subproblem_matrix(op::CartesianCurl, subproblem)
    """Build operator matrix for curl in a specific subproblem."""
    operand = op.operand
    if !isa(operand, VectorField) || op.coordsys.dim != 3
        return spzeros(Float64, 0, 0)
    end

    coords = op.coordsys.coords
    comps = operand.components
    n_per = field_dofs(comps[1])
    zero_block = spzeros(Float64, n_per, n_per)

    D = Array{SparseMatrixCSC}(undef, 3, 3)
    for i in 1:3, j in 1:3
        mat = build_operator_differentiation_matrix(comps[i], coords[j], 1)
        D[i, j] = mat === nothing ? zero_block : mat
    end

    row1 = hcat(zero_block, -D[2, 3], D[3, 2])
    row2 = hcat(D[1, 3], zero_block, -D[3, 1])
    row3 = hcat(-D[1, 2], D[2, 1], zero_block)
    result = vcat(row1, row2, row3)

    if op.coordsys.right_handed === false
        result = -result
    end

    return result
end

function check_conditions(op::CartesianCurl)
    """Check that operands are in a proper layout for curl computation."""
    operand = op.operand

    # For VectorField, check that all components are in a consistent layout
    if isa(operand, VectorField)
        layouts = Symbol[]
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout) && comp.current_layout !== nothing
                push!(layouts, comp.current_layout)
            end
        end
        # All components should be in the same layout
        return length(Set(layouts)) <= 1
    end

    return true
end

function enforce_conditions(op::CartesianCurl)
    """Ensure operands are in coefficient layout for differentiation."""
    operand = op.operand

    # For VectorField, ensure all components are in coefficient layout
    if isa(operand, VectorField)
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout)
                ensure_layout!(comp, :c)
            end
        end
    elseif hasfield(typeof(operand), :current_layout)
        ensure_layout!(operand, :c)
    end

    return nothing
end

# ============================================================================
# CartesianLaplacian - Laplacian in Cartesian coordinates
# Following operators:4016-4062 CartesianLaplacian
# ============================================================================

"""
    CartesianLaplacian <: AbstractLinearOperator

Laplacian operator specialized for Cartesian coordinates.

Following operators:4016-4062 CartesianLaplacian implementation.

For scalar field f:
∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

For vector field u:
(∇²u)_i = ∂²u_i/∂x² + ∂²u_i/∂y² + ∂²u_i/∂z²
"""
struct CartesianLaplacian <: AbstractLinearOperator
    operand::Operand
    coordsys::CoordinateSystem
    arg::Any  # Sum of second derivatives

    function CartesianLaplacian(operand::Operand, coordsys::CoordinateSystem)
        # Handle single coordinate case
        if isa(coordsys, Coordinate)
            coordsys = CartesianCoordinates(coordsys.name)
        end

        if !isa(coordsys, CartesianCoordinates)
            throw(ArgumentError("CartesianLaplacian requires CartesianCoordinates"))
        end

        # Build sum of second derivatives
        # ∇²f = Σ ∂²f/∂x_i²
        parts = [Differentiate(operand, c, 2) for c in coordsys.coords]

        new(operand, coordsys, parts)
    end
end

function matrix_dependence(op::CartesianLaplacian, vars...)
    # Linear operator; dependence follows the operand
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::CartesianLaplacian, vars...)
    result = falses(length(vars))
    for part in op.arg
        result .|= matrix_coupling(part, vars...)
    end
    return result
end

function subproblem_matrix(op::CartesianLaplacian, subproblem)
    """Build operator matrix for a specific subproblem."""
    # Sum of second derivative matrices
    result = nothing
    for part in op.arg
        mat = expression_matrices(part, subproblem, [op.operand])
        if haskey(mat, op.operand)
            if result === nothing
                result = mat[op.operand]
            else
                result = result + mat[op.operand]
            end
        end
    end

    return result === nothing ? spzeros(Float64, 0, 0) : result
end

function check_conditions(op::CartesianLaplacian)
    """Check that operands are in a proper layout for Laplacian computation."""
    operand = op.operand

    # For VectorField, check that all components are in a consistent layout
    if isa(operand, VectorField)
        layouts = Symbol[]
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout) && comp.current_layout !== nothing
                push!(layouts, comp.current_layout)
            end
        end
        return length(Set(layouts)) <= 1
    end

    # For ScalarField or similar
    if hasfield(typeof(operand), :current_layout)
        layout = operand.current_layout
        if layout == :g
            return hasfield(typeof(operand), :buffers) && get_grid_data(operand) !== nothing
        elseif layout == :c
            return hasfield(typeof(operand), :buffers) && get_coeff_data(operand) !== nothing
        end
    end

    return true
end

function enforce_conditions(op::CartesianLaplacian)
    """Ensure operands are in coefficient layout for differentiation."""
    operand = op.operand

    # For VectorField, ensure all components are in coefficient layout
    if isa(operand, VectorField)
        for comp in operand.components
            if hasfield(typeof(comp), :current_layout)
                ensure_layout!(comp, :c)
            end
        end
    elseif hasfield(typeof(operand), :current_layout)
        ensure_layout!(operand, :c)
    end

    return nothing
end

# ============================================================================
# CartesianTrace - Trace in Cartesian coordinates
# ============================================================================

"""
    CartesianTrace <: AbstractLinearOperator

Trace operator for tensor fields in Cartesian coordinates.

For tensor T:
trace(T) = Σ T_ii = T_xx + T_yy + T_zz
"""
struct CartesianTrace <: AbstractLinearOperator
    operand::Operand

    function CartesianTrace(operand::Operand)
        if !isa(operand, TensorField)
            throw(ArgumentError("CartesianTrace requires a TensorField"))
        end
        new(operand)
    end
end

function matrix_dependence(op::CartesianTrace, vars...)
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::CartesianTrace, vars...)
    return matrix_coupling(op.operand, vars...)
end

function subproblem_matrix(op::CartesianTrace, subproblem)
    """Build trace matrix."""
    # Get tensor dimension
    tensorsig = get_tensorsig(op.operand)
    if length(tensorsig) < 2
        throw(ArgumentError("Trace requires rank-2 or higher tensor"))
    end

    dim = tensorsig[1].dim
    scalar_size = get_scalar_size(op.operand, subproblem)

    # Build trace selection matrix
    # Selects diagonal elements: (0,0), (1,1), (2,2), ...
    I_scalar = sparse(I, scalar_size, scalar_size)

    trace_factor = zeros(Float64, 1, dim * dim)
    for i in 1:dim
        # Diagonal element (i-1, i-1) in row-major flattening
        idx = (i - 1) * dim + i
        trace_factor[1, idx] = 1.0
    end

    return kron(sparse(trace_factor), I_scalar)
end

function check_conditions(op::CartesianTrace)
    return true
end

function enforce_conditions(op::CartesianTrace)
    return nothing
end

# ============================================================================
# CartesianSkew - Skew operation for 2D vectors
# Following operators:2098-2123 CartesianSkew
# ============================================================================

"""
    CartesianSkew <: AbstractLinearOperator

Skew operator for 2D vector fields in Cartesian coordinates.

Following operators:2098-2123 Skew implementation.

For 2D vector u = (u_x, u_y):
skew(u) = (-u_y, u_x)

This is equivalent to rotation by 90 degrees and is used for
2D curl operations (z-component of 3D curl).
"""
struct CartesianSkew <: AbstractLinearOperator
    operand::Operand
    index::Int
    coordsys::CoordinateSystem

    function CartesianSkew(operand::Operand; index::Int=0)
        if !isa(operand, VectorField)
            throw(ArgumentError("CartesianSkew requires a VectorField"))
        end

        tensorsig = get_tensorsig(operand)
        coordsys = tensorsig[index + 1]

        if !isa(coordsys, CartesianCoordinates) && !isa(coordsys, Coordinate)
            throw(ArgumentError("CartesianSkew requires Cartesian coordinates"))
        end

        dim = isa(coordsys, CartesianCoordinates) ? coordsys.dim : 1
        if dim != 2
            throw(ArgumentError("CartesianSkew is only implemented for 2D vectors"))
        end

        new(operand, index, coordsys)
    end
end

function matrix_dependence(op::CartesianSkew, vars...)
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::CartesianSkew, vars...)
    return matrix_coupling(op.operand, vars...)
end

function subproblem_matrix(op::CartesianSkew, subproblem)
    """Build skew matrix."""
    # For 2D: skew(u) = [-u_y, u_x]
    # Matrix: [0 -1; 1 0]

    scalar_size = get_scalar_size(op.operand, subproblem)
    I_scalar = sparse(I, scalar_size, scalar_size)

    skew_factor = [0.0 -1.0; 1.0 0.0]

    return kron(sparse(skew_factor), I_scalar)
end

function check_conditions(op::CartesianSkew)
    return true
end

function enforce_conditions(op::CartesianSkew)
    return nothing
end

# ============================================================================
# Evaluation functions for Cartesian operators
# ============================================================================

"""
    evaluate_cartesian_gradient(op::CartesianGradient, layout::Symbol=:g)

Evaluate Cartesian gradient operator.
"""
function evaluate_cartesian_gradient(op::CartesianGradient, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # Scalar → Vector: (∇f)_i = ∂f/∂x_i
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)

        for (i, coord) in enumerate(coordsys.coords)
            deriv_op = Differentiate(operand, coord, 1)
            result.components[i] = evaluate_differentiate(deriv_op, layout)
        end

        return result

    elseif isa(operand, VectorField)
        # Vector → Tensor: (∇u)_ij = ∂u_j/∂x_i (first index = derivative direction)
        result = TensorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)

        for (i, coord) in enumerate(coordsys.coords)
            for (j, comp) in enumerate(operand.components)
                deriv_op = Differentiate(comp, coord, 1)
                result.components[i, j] = evaluate_differentiate(deriv_op, layout)
            end
        end

        return result

    else
        throw(ArgumentError("CartesianGradient requires ScalarField or VectorField operand, got $(typeof(operand))"))
    end
end

"""
    evaluate_cartesian_divergence(op::CartesianDivergence, layout::Symbol=:g)

Evaluate Cartesian divergence operator.
"""
function evaluate_cartesian_divergence(op::CartesianDivergence, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("CartesianDivergence requires VectorField"))
    end

    # Create scalar field for result
    result = ScalarField(operand.dist, "div_$(operand.name)", operand.components[1].bases, operand.dtype)
    ensure_layout!(result, layout)

    # Initialize to zero
    if layout == :g
        fill!(get_grid_data(result), 0.0)
    else
        fill!(get_coeff_data(result), 0.0)
    end

    # Sum partial derivatives of components
    for (i, coord) in enumerate(coordsys.coords)
        comp = operand.components[i]
        deriv_op = Differentiate(comp, coord, 1)
        comp_deriv = evaluate_differentiate(deriv_op, layout)

        if layout == :g
            get_grid_data(result) .+= get_grid_data(comp_deriv)
        else
            get_coeff_data(result) .+= get_coeff_data(comp_deriv)
        end
    end

    return result
end

"""
    evaluate_cartesian_curl(op::CartesianCurl, layout::Symbol=:g)

Evaluate Cartesian curl operator (3D only).
"""
function evaluate_cartesian_curl(op::CartesianCurl, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("CartesianCurl requires VectorField"))
    end

    if coordsys.dim != 3
        throw(ArgumentError("CartesianCurl is only for 3D. For 2D, use skew(grad(f))."))
    end

    # Create result vector field
    result = VectorField(operand.dist, coordsys, "curl_$(operand.name)", operand.components[1].bases, operand.dtype)

    cx, cy, cz = coordsys.coords
    ux, uy, uz = operand.components

    # curl_x = ∂uz/∂y - ∂uy/∂z
    duz_dy = evaluate_differentiate(Differentiate(uz, cy, 1), layout)
    duy_dz = evaluate_differentiate(Differentiate(uy, cz, 1), layout)
    result.components[1] = field_subtract(duz_dy, duy_dz, layout)

    # curl_y = ∂ux/∂z - ∂uz/∂x
    dux_dz = evaluate_differentiate(Differentiate(ux, cz, 1), layout)
    duz_dx = evaluate_differentiate(Differentiate(uz, cx, 1), layout)
    result.components[2] = field_subtract(dux_dz, duz_dx, layout)

    # curl_z = ∂uy/∂x - ∂ux/∂y
    duy_dx = evaluate_differentiate(Differentiate(uy, cx, 1), layout)
    dux_dy = evaluate_differentiate(Differentiate(ux, cy, 1), layout)
    result.components[3] = field_subtract(duy_dx, dux_dy, layout)

    # Handle handedness
    if coordsys.right_handed === false
        for comp in result.components
            if layout == :g
                get_grid_data(comp) .*= -1
            else
                get_coeff_data(comp) .*= -1
            end
        end
    end

    return result
end

"""
    field_subtract(a, b, layout)

Subtract two scalar fields.
"""
function field_subtract(a::ScalarField, b::ScalarField, layout::Symbol)
    result = ScalarField(a.dist, "sub", a.bases, a.dtype)
    ensure_layout!(result, layout)
    ensure_layout!(a, layout)
    ensure_layout!(b, layout)

    if layout == :g
        get_grid_data(result) .= get_grid_data(a) .- get_grid_data(b)
    else
        get_coeff_data(result) .= get_coeff_data(a) .- get_coeff_data(b)
    end

    return result
end

"""
    evaluate_cartesian_laplacian(op::CartesianLaplacian, layout::Symbol=:g)

Evaluate Cartesian Laplacian operator.
"""
function evaluate_cartesian_laplacian(op::CartesianLaplacian, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # Create result field
        result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(result, layout)

        # Initialize to zero
        if layout == :g
            fill!(get_grid_data(result), 0.0)
        else
            fill!(get_coeff_data(result), 0.0)
        end

        # Sum second derivatives
        for coord in coordsys.coords
            d2_op = Differentiate(operand, coord, 2)
            d2 = evaluate_differentiate(d2_op, layout)

            if layout == :g
                get_grid_data(result) .+= get_grid_data(d2)
            else
                get_coeff_data(result) .+= get_coeff_data(d2)
            end
        end

        return result
    elseif isa(operand, VectorField)
        # Apply Laplacian to each component
        result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                            operand.components[1].bases, operand.dtype)

        for (i, comp) in enumerate(operand.components)
            lap_op = CartesianLaplacian(comp, coordsys)
            result.components[i] = evaluate_cartesian_laplacian(lap_op, layout)
        end

        return result
    else
        throw(ArgumentError("CartesianLaplacian not implemented for $(typeof(operand))"))
    end
end

"""
    evaluate_cartesian_trace(op::CartesianTrace, layout::Symbol=:g)

Evaluate Cartesian trace operator.
"""
function evaluate_cartesian_trace(op::CartesianTrace, layout::Symbol=:g)
    operand = op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("CartesianTrace requires TensorField"))
    end

    ensure_layout!(operand, layout)

    # Sum diagonal elements
    dim = size(operand.components, 1)  # Get dimension from matrix size

    # Check for empty tensor
    if dim == 0 || isempty(operand.components)
        throw(ArgumentError("CartesianTrace requires non-empty TensorField"))
    end

    # Get first component to create result
    first_comp = operand.components[1, 1]
    result = ScalarField(first_comp.dist, "trace_$(operand.name)", first_comp.bases, first_comp.dtype)
    ensure_layout!(result, layout)

    # Initialize to zero
    if layout == :g
        fill!(get_grid_data(result), 0.0)
        for i in 1:dim
            get_grid_data(result) .+= get_grid_data(operand.components[i, i])
        end
    else
        fill!(get_coeff_data(result), 0.0)
        for i in 1:dim
            get_coeff_data(result) .+= get_coeff_data(operand.components[i, i])
        end
    end

    return result
end

"""
    evaluate_cartesian_skew(op::CartesianSkew, layout::Symbol=:g)

Evaluate Cartesian skew operator (2D vector rotation by 90°).
"""
function evaluate_cartesian_skew(op::CartesianSkew, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("CartesianSkew requires VectorField"))
    end

    ensure_layout!(operand, layout)

    # skew(u_x, u_y) = (-u_y, u_x)
    result = VectorField(operand.dist, coordsys, "skew_$(operand.name)",
                        operand.components[1].bases, operand.dtype)

    for comp in result.components
        ensure_layout!(comp, layout)
    end

    # skew(u_x, u_y) = (-u_y, u_x): first component is negated second, second is first
    if layout == :g
        get_grid_data(result.components[1]) .= .-get_grid_data(operand.components[2])
        copyto!(get_grid_data(result.components[2]), get_grid_data(operand.components[1]))
    else
        get_coeff_data(result.components[1]) .= .-get_coeff_data(operand.components[2])
        copyto!(get_coeff_data(result.components[2]), get_coeff_data(operand.components[1]))
    end

    return result
end

# ============================================================================
# DirectProduct operator variants
# ============================================================================

"""
    DirectProductGradient <: AbstractLinearOperator

Gradient operator for DirectProduct coordinate systems.
"""
struct DirectProductGradient <: AbstractLinearOperator
    operand::Operand
    coordsys::DirectProduct
    args::Vector{Any}  # Gradient operators for each subsystem

    function DirectProductGradient(operand::Operand, coordsys::DirectProduct)
        # Build gradient for each coordinate subsystem
        args = [Gradient(operand, cs) for cs in coordsys.coordsystems]
        new(operand, coordsys, args)
    end
end

"""
    DirectProductDivergence <: AbstractLinearOperator

Divergence operator for DirectProduct coordinate systems.

Following operators:3497-3544 DirectProductDivergence.
"""
struct DirectProductDivergence <: AbstractLinearOperator
    operand::Operand
    index::Int
    coordsys::DirectProduct
    args::Vector{Any}  # Divergence operators for each subsystem

    function DirectProductDivergence(operand::Operand; index::Int=0)
        tensorsig = get_tensorsig(operand)
        coordsys = tensorsig[index + 1]

        if !isa(coordsys, DirectProduct)
            throw(ArgumentError("DirectProductDivergence requires DirectProduct coordinates"))
        end

        # Build divergence for each coordinate subsystem
        args = Any[]
        for cs in coordsys.coordsystems
            # Extract component for this subsystem
            comp_op = DirectProductComponent(operand, index=index, comp=cs)
            div_op = Divergence(comp_op)
            push!(args, div_op)
        end

        new(operand, index, coordsys, args)
    end
end

"""
    DirectProductLaplacian <: AbstractLinearOperator

Laplacian operator for DirectProduct coordinate systems.

Following operators:4064-4106 DirectProductLaplacian.
"""
struct DirectProductLaplacian <: AbstractLinearOperator
    operand::Operand
    coordsys::DirectProduct
    args::Vector{Any}  # Laplacian operators for each subsystem

    function DirectProductLaplacian(operand::Operand, coordsys::DirectProduct)
        # Build Laplacian for each coordinate subsystem
        args = [Laplacian(operand, cs) for cs in coordsys.coordsystems]
        new(operand, coordsys, args)
    end
end

"""
    DirectProductComponent

Extract component corresponding to a coordinate subsystem from DirectProduct.
"""
struct DirectProductComponent <: AbstractLinearOperator
    operand::Operand
    index::Int
    comp::CoordinateSystem  # Subsystem to extract

    function DirectProductComponent(operand::Operand; index::Int=0, comp::CoordinateSystem)
        new(operand, index, comp)
    end
end

# ============================================================================
# Multiclass dispatch integration
# Following MultiClass metaclass pattern
# ============================================================================

"""
    dispatch_cartesian_operator(OpType, operand, args...; kwargs...)

Dispatch to Cartesian-specific or DirectProduct-specific operator variant
based on coordinate system type.
"""
function dispatch_cartesian_operator(::Type{Gradient}, operand, coordsys; kwargs...)
    if isa(coordsys, CartesianCoordinates) || isa(coordsys, Coordinate)
        return CartesianGradient(operand, coordsys)
    elseif isa(coordsys, DirectProduct)
        return DirectProductGradient(operand, coordsys)
    else
        throw(ArgumentError("Gradient not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

function dispatch_cartesian_operator(::Type{Divergence}, operand; index::Int=0, kwargs...)
    tensorsig = get_tensorsig(operand)
    if isempty(tensorsig)
        throw(ArgumentError("Divergence requires a tensor operand"))
    end

    coordsys = tensorsig[index + 1]

    if isa(coordsys, CartesianCoordinates) || isa(coordsys, Coordinate)
        return CartesianDivergence(operand; index=index)
    elseif isa(coordsys, DirectProduct)
        return DirectProductDivergence(operand; index=index)
    else
        throw(ArgumentError("Divergence not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

function dispatch_cartesian_operator(::Type{Curl}, operand; index::Int=0, kwargs...)
    tensorsig = get_tensorsig(operand)
    if isempty(tensorsig)
        throw(ArgumentError("Curl requires a tensor operand"))
    end

    coordsys = tensorsig[index + 1]

    if isa(coordsys, CartesianCoordinates)
        return CartesianCurl(operand; index=index)
    else
        throw(ArgumentError("Curl not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

function dispatch_cartesian_operator(::Type{Laplacian}, operand, coordsys; kwargs...)
    if isa(coordsys, CartesianCoordinates) || isa(coordsys, Coordinate)
        return CartesianLaplacian(operand, coordsys)
    elseif isa(coordsys, DirectProduct)
        return DirectProductLaplacian(operand, coordsys)
    else
        throw(ArgumentError("Laplacian not implemented for coordinate system type $(typeof(coordsys))"))
    end
end

# ============================================================================
# Generic matrix operation fallbacks
# ============================================================================

"""
    matrix_dependence(operand, vars...)

Default matrix dependence for operands (not operators).
"""
function matrix_dependence(operand::Operand, vars...)
    # Check if operand is one of the variables
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if operand === var ||
           (hasfield(typeof(operand), :name) && hasfield(typeof(var), :name) && operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(op::Differentiate, vars...)

Matrix coupling for Differentiate operator.
A linear operator that couples its operand variable into the equation.
"""
function matrix_coupling(op::Differentiate, vars...)
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if op.operand === var ||
           (hasfield(typeof(op.operand), :name) && hasfield(typeof(var), :name) &&
            op.operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(operand, vars...)

Default matrix coupling for operands (not operators).
"""
function matrix_coupling(operand::Operand, vars...)
    # Single variable doesn't couple to others by default
    return falses(length(vars))
end

# ============================================================================
# Integration with evaluate dispatcher
# ============================================================================

# Extend the main evaluate function to handle Cartesian operators
function evaluate(op::CartesianComponent, layout::Symbol=:g)
    return evaluate_cartesian_component(op, layout)
end

function evaluate(op::CartesianGradient, layout::Symbol=:g)
    return evaluate_cartesian_gradient(op, layout)
end

function evaluate(op::CartesianDivergence, layout::Symbol=:g)
    return evaluate_cartesian_divergence(op, layout)
end

function evaluate(op::CartesianCurl, layout::Symbol=:g)
    return evaluate_cartesian_curl(op, layout)
end

function evaluate(op::CartesianLaplacian, layout::Symbol=:g)
    return evaluate_cartesian_laplacian(op, layout)
end

function evaluate(op::CartesianTrace, layout::Symbol=:g)
    return evaluate_cartesian_trace(op, layout)
end

function evaluate(op::CartesianSkew, layout::Symbol=:g)
    return evaluate_cartesian_skew(op, layout)
end

# DirectProduct evaluation methods
function evaluate(op::DirectProductGradient, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        for (i, coord) in enumerate(coordsys.coords)
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end
        return result
    end

    throw(ArgumentError("DirectProductGradient evaluation only implemented for scalar fields"))
end

function evaluate(op::DirectProductDivergence, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("DirectProductDivergence requires VectorField"))
    end

    result = ScalarField(operand.dist, "div_$(operand.name)", operand.components[1].bases, operand.dtype)
    ensure_layout!(result, layout)

    if layout == :g
        fill!(get_grid_data(result), 0.0)
        for (i, coord) in enumerate(coordsys.coords)
            comp = operand.components[i]
            deriv = evaluate_differentiate(Differentiate(comp, coord, 1), layout)
            get_grid_data(result) .+= get_grid_data(deriv)
        end
    else
        fill!(get_coeff_data(result), 0.0)
        for (i, coord) in enumerate(coordsys.coords)
            comp = operand.components[i]
            deriv = evaluate_differentiate(Differentiate(comp, coord, 1), layout)
            get_coeff_data(result) .+= get_coeff_data(deriv)
        end
    end

    return result
end

function evaluate(op::DirectProductLaplacian, layout::Symbol=:g)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
        ensure_layout!(result, layout)

        if layout == :g
            fill!(get_grid_data(result), 0.0)
            for coord in coordsys.coords
                d2 = evaluate_differentiate(Differentiate(operand, coord, 2), layout)
                get_grid_data(result) .+= get_grid_data(d2)
            end
        else
            fill!(get_coeff_data(result), 0.0)
            for coord in coordsys.coords
                d2 = evaluate_differentiate(Differentiate(operand, coord, 2), layout)
                get_coeff_data(result) .+= get_coeff_data(d2)
            end
        end
        return result
    elseif isa(operand, VectorField)
        result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                            operand.components[1].bases, operand.dtype)
        for (i, comp) in enumerate(operand.components)
            lap_comp = evaluate(DirectProductLaplacian(comp, coordsys), layout)
            result.components[i] = lap_comp
        end
        return result
    end

    throw(ArgumentError("DirectProductLaplacian not implemented for $(typeof(operand))"))
end

"""
    evaluate(op::DirectProductComponent, layout::Symbol=:g)

Extract a sub-vector from a DirectProduct VectorField.

!!! note "Return Value Semantics"
    The returned VectorField contains **references** to the original operand's
    components (not copies). Modifications to the returned components will
    affect the original VectorField.
"""
function evaluate(op::DirectProductComponent, layout::Symbol=:g)
    operand = op.operand
    if !isa(operand, VectorField)
        throw(ArgumentError("DirectProductComponent requires VectorField"))
    end

    if !isa(operand.coordsys, DirectProduct)
        throw(ArgumentError("DirectProductComponent requires DirectProduct coordinates"))
    end

    coordsys = operand.coordsys
    subaxis = get(coordsys.subaxis_by_cs, op.comp, nothing)
    if subaxis === nothing
        idx = findfirst(cs -> cs.names == op.comp.names, coordsys.coordsystems)
        if idx === nothing
            throw(ArgumentError("Component coordinate system not found in DirectProduct"))
        end
        subaxis = coordsys.subaxis_by_cs[coordsys.coordsystems[idx]]
    end

    dim = op.comp.dim
    start_idx = subaxis + 1
    end_idx = start_idx + dim - 1

    comp_tag = isempty(op.comp.names) ? "component" : join(op.comp.names, "")
    result = VectorField(operand.dist, op.comp, "$(operand.name)_$(comp_tag)", operand.bases, operand.dtype)
    for i in 1:dim
        result.components[i] = operand.components[start_idx + i - 1]
        ensure_layout!(result.components[i], layout)
    end

    return result
end

# ============================================================================
# Exports
# ============================================================================

# Export types
export AbstractLinearOperator, OperatorConditions,
       CartesianComponent, CartesianGradient, CartesianDivergence, CartesianCurl,
       CartesianLaplacian, CartesianTrace, CartesianSkew,
       DirectProductGradient, DirectProductDivergence, DirectProductLaplacian,
       DirectProductComponent

# Export functions
export cartesian_component, get_tensorsig, get_scalar_size,
       matrix_dependence, matrix_coupling, subproblem_matrix,
       check_conditions, enforce_conditions, operate,
       evaluate_cartesian_component, evaluate_cartesian_gradient,
       evaluate_cartesian_divergence, evaluate_cartesian_curl,
       evaluate_cartesian_laplacian, evaluate_cartesian_trace,
       evaluate_cartesian_skew, field_subtract,
       dispatch_cartesian_operator
