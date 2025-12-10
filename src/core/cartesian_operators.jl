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
register_operator_alias!(cartesian_component, "cartesian_component", "comp")
register_operator_parseable!(cartesian_component, "cartesian_component", "comp")

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
        return operand.tensorsig
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
"""
function matrix_dependence(op::CartesianComponent, vars...)
    # Component extraction doesn't add new dependencies
    return matrix_dependence(op.operand, vars...)
end

"""
    matrix_coupling(op::CartesianComponent, vars...)

Determine which variables couple through the operator.
"""
function matrix_coupling(op::CartesianComponent, vars...)
    # Component extraction doesn't add new coupling
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

    # Build tensor index selection matrix
    # For a vector, this selects row comp_subaxis from a dim×1 "vector"
    index_factor = zeros(Float64, 1, dim)
    index_factor[1, op.comp_subaxis + 1] = 1.0  # 1-based Julia indexing

    # Full matrix is Kronecker product
    return kron(sparse(index_factor), I_scalar)
end

"""
    get_scalar_size(operand, subproblem)

Get the size of the scalar portion of an operand for matrix assembly.
"""
function get_scalar_size(operand, subproblem)
    if hasfield(typeof(operand), :bases)
        size = 1
        for basis in operand.bases
            if basis !== nothing
                size *= basis.meta.size
            end
        end
        return size
    elseif hasfield(typeof(operand), :components) && !isempty(operand.components)
        return get_scalar_size(operand.components[1], subproblem)
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

    # Verify operand has valid data in at least one layout
    if hasfield(typeof(operand), :current_layout)
        layout = operand.current_layout
        if layout == :g
            return operand.data_g !== nothing
        elseif layout == :c
            return operand.data_c !== nothing
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
    if hasfield(typeof(operand), :current_layout) && hasfield(typeof(operand), :data_g)
        layout = operand.current_layout
        if layout == :g && operand.data_g === nothing
            # Need to transform from coefficient space to grid space
            if hasfield(typeof(operand), :data_c) && operand.data_c !== nothing
                # Trigger transform - this depends on the field's transform implementation
                ensure_layout!(operand, :g)
            end
        elseif layout == :c && operand.data_c === nothing
            if hasfield(typeof(operand), :data_g) && operand.data_g !== nothing
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

    # Get operand data
    if hasfield(typeof(operand), :current_layout)
        layout = operand.current_layout
    else
        layout = :g
    end

    # Extract component based on operand type
    if isa(operand, VectorField)
        # For vector field, extract the component field
        comp_idx = op.comp_subaxis + 1  # 1-based Julia indexing
        comp_field = operand.components[comp_idx]

        if layout == :g
            out.data_g = copy(comp_field.data_g)
        else
            out.data_c = copy(comp_field.data_c)
        end
    elseif isa(operand, TensorField)
        # For tensor field, extract along the specified tensor index
        extract_tensor_component!(out, operand, op.index, op.comp_subaxis, layout)
    end

    return out
end

"""
    extract_tensor_component!(out, tensor, index, comp_subaxis, layout)

Extract a component from a tensor field along specified index.
"""
function extract_tensor_component!(out, tensor::TensorField, index::Int, comp_subaxis::Int, layout::Symbol)
    # Tensor components are stored as nested structure
    # Navigate to the correct component
    if layout == :g && hasfield(typeof(tensor), :data_g) && tensor.data_g !== nothing
        # Extract from grid data
        out.data_g = selectdim(tensor.data_g, index + 1, comp_subaxis + 1)
    elseif hasfield(typeof(tensor), :data_c) && tensor.data_c !== nothing
        # Extract from coefficient data
        out.data_c = selectdim(tensor.data_c, index + 1, comp_subaxis + 1)
    end
end

# ============================================================================
# Evaluate function for CartesianComponent
# ============================================================================

"""
    evaluate_cartesian_component(op::CartesianComponent, layout::Symbol=:g)

Evaluate CartesianComponent operator.
"""
function evaluate_cartesian_component(op::CartesianComponent, layout::Symbol=:g)
    operand = op.operand

    if isa(operand, VectorField)
        # Extract the component field directly
        comp_idx = op.comp_subaxis + 1  # 1-based Julia indexing
        result = copy(operand.components[comp_idx])

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
        if op.index == 0
            # Extracting first index: T[comp_idx, :] -> vector
            for j in 1:size(operand.components, 2)
                src_field = operand.components[comp_idx, j]
                result.components[j] = copy(src_field)
            end
        else
            # Extracting second index: T[:, comp_idx] -> vector
            for i in 1:size(operand.components, 1)
                src_field = operand.components[i, comp_idx]
                result.components[i] = copy(src_field)
            end
        end

        # Ensure proper layout
        for comp in result.components
            if layout == :g
                ensure_layout!(comp, :g)
            else
                ensure_layout!(comp, :c)
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

Following operators:2340-2412 CartesianGradient implementation.

For scalar field f, gradient is:
∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)

For vector field u, gradient is tensor:
(∇u)_ij = ∂u_i/∂x_j
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
    # Gradient depends on all coordinates in the system
    result = falses(length(vars))
    for arg in op.args
        arg_dep = matrix_dependence(arg, vars...)
        result .|= arg_dep
    end
    return result
end

function matrix_coupling(op::CartesianGradient, vars...)
    result = falses(length(vars))
    for arg in op.args
        arg_coupling = matrix_coupling(arg, vars...)
        result .|= arg_coupling
    end
    return result
end

function subproblem_matrix(op::CartesianGradient, subproblem)
    """Build operator matrix for a specific subproblem."""
    # Stack differentiation matrices vertically for vector output
    matrices = [expression_matrices(arg, subproblem, [op.operand])[op.operand]
                for arg in op.args if haskey(expression_matrices(arg, subproblem, [op.operand]), op.operand)]

    if isempty(matrices)
        return spzeros(Float64, 0, 0)
    end

    return vcat(matrices...)
end

function check_conditions(op::CartesianGradient)
    """Check that operands are in a proper layout."""
    # Require all args to be in same layout
    layouts = [arg.operand.current_layout for arg in op.args
               if hasfield(typeof(arg.operand), :current_layout)]
    return length(Set(layouts)) <= 1
end

function enforce_conditions(op::CartesianGradient)
    """Require operands to be in coefficient layout."""
    for arg in op.args
        if hasfield(typeof(arg), :operand) && hasfield(typeof(arg.operand), :current_layout)
            ensure_layout!(arg.operand, :c)
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
        comps = []
        for (i, coord) in enumerate(coordsys.coords)
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
    if isa(op.arg, AbstractArray)
        result = falses(length(vars))
        for comp in op.arg
            if hasmethod(matrix_dependence, Tuple{typeof(comp), Vararg})
                result .|= matrix_dependence(comp, vars...)
            end
        end
        return result
    end
    return matrix_dependence(op.arg, vars...)
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
    # Horizontal concatenation of derivative matrices for each component
    matrices = []
    for (i, comp_deriv) in enumerate(op.arg)
        mat = expression_matrices(comp_deriv, subproblem, [op.operand])
        if haskey(mat, op.operand)
            push!(matrices, mat[op.operand])
        end
    end

    if isempty(matrices)
        return spzeros(Float64, 0, 0)
    end

    return hcat(matrices...)
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
            return hasfield(typeof(operand), :data_g) && operand.data_g !== nothing
        elseif layout == :c
            return hasfield(typeof(operand), :data_c) && operand.data_c !== nothing
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
    result = falses(length(vars))

    # Check dependencies from all derivative operators
    for key in (:x, :y, :z)
        pos_deriv, neg_deriv = op.arg[key]
        if hasmethod(matrix_dependence, Tuple{typeof(pos_deriv), Vararg})
            result .|= matrix_dependence(pos_deriv, vars...)
        end
        if hasmethod(matrix_dependence, Tuple{typeof(neg_deriv), Vararg})
            result .|= matrix_dependence(neg_deriv, vars...)
        end
    end

    return result
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
    # Each output component is a difference of two derivatives
    # Stack vertically for vector output
    matrices = []

    for key in (:x, :y, :z)
        pos_deriv, neg_deriv = op.arg[key]

        # Get matrices for positive and negative terms
        pos_mat = expression_matrices(pos_deriv, subproblem, [op.operand])
        neg_mat = expression_matrices(neg_deriv, subproblem, [op.operand])

        if haskey(pos_mat, op.operand) && haskey(neg_mat, op.operand)
            # Curl component = positive - negative
            comp_mat = pos_mat[op.operand] - neg_mat[op.operand]
            push!(matrices, comp_mat)
        end
    end

    if isempty(matrices)
        return spzeros(Float64, 0, 0)
    end

    return vcat(matrices...)
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
        parts = [Differentiate(Differentiate(operand, c, 1), c, 1) for c in coordsys.coords]

        new(operand, coordsys, parts)
    end
end

function matrix_dependence(op::CartesianLaplacian, vars...)
    result = falses(length(vars))
    for part in op.arg
        result .|= matrix_dependence(part, vars...)
    end
    return result
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
            return hasfield(typeof(operand), :data_g) && operand.data_g !== nothing
        elseif layout == :c
            return hasfield(typeof(operand), :data_c) && operand.data_c !== nothing
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
    scalar_size = get_scalar_size(op.operand, subproblem) ÷ (dim * dim)

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

    scalar_size = get_scalar_size(op.operand, subproblem) ÷ 2
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
        # Create vector field for result
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)

        # Compute partial derivatives for each component
        for (i, coord) in enumerate(coordsys.coords)
            deriv_op = Differentiate(operand, coord, 1)
            result.components[i] = evaluate_differentiate(deriv_op, layout)
        end

        return result
    else
        throw(ArgumentError("CartesianGradient evaluation only implemented for scalar fields"))
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
        fill!(result.data_g, 0.0)
    else
        fill!(result.data_c, 0.0)
    end

    # Sum partial derivatives of components
    for (i, coord) in enumerate(coordsys.coords)
        comp = operand.components[i]
        deriv_op = Differentiate(comp, coord, 1)
        comp_deriv = evaluate_differentiate(deriv_op, layout)

        if layout == :g
            result.data_g .+= comp_deriv.data_g
        else
            result.data_c .+= comp_deriv.data_c
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
                comp.data_g .*= -1
            else
                comp.data_c .*= -1
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

    if layout == :g
        result.data_g = a.data_g .- b.data_g
    else
        result.data_c = a.data_c .- b.data_c
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
            fill!(result.data_g, 0.0)
        else
            fill!(result.data_c, 0.0)
        end

        # Sum second derivatives
        for coord in coordsys.coords
            d2_op = Differentiate(Differentiate(operand, coord, 1), coord, 1)
            d2 = evaluate_differentiate(d2_op, layout)

            if layout == :g
                result.data_g .+= d2.data_g
            else
                result.data_c .+= d2.data_c
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

    # Get first component to create result
    first_comp = operand.components[1, 1]
    result = ScalarField(first_comp.dist, "trace_$(operand.name)", first_comp.bases, first_comp.dtype)
    ensure_layout!(result, layout)

    # Initialize to zero
    if layout == :g
        fill!(result.data_g, 0.0)
        for i in 1:dim
            result.data_g .+= operand.components[i, i].data_g
        end
    else
        fill!(result.data_c, 0.0)
        for i in 1:dim
            result.data_c .+= operand.components[i, i].data_c
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

    if layout == :g
        result.components[1].data_g = -operand.components[2].data_g
        result.components[2].data_g = copy(operand.components[1].data_g)
    else
        result.components[1].data_c = -operand.components[2].data_c
        result.components[2].data_c = copy(operand.components[1].data_c)
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
        args = []
        for (i, cs) in enumerate(coordsys.coordsystems)
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
