# Differential Cartesian operators and their matrix/layout behavior.

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

"""Build operator matrix for a specific subproblem."""
function subproblem_matrix(op::CartesianGradient, subproblem)
    operand = op.operand
    coordsys = op.coordsys

    if isa(operand, ScalarField)
        # Scalar → Vector: stack differentiation matrices vertically
        matrices = SparseMatrixCSC[]
        n_per = get_scalar_size(operand, subproblem)
        zero_block = spzeros(Float64, n_per, n_per)

        for arg in op.args
            mat = expression_matrices(arg, subproblem, [operand])
            if haskey(mat, operand)
                push!(matrices, mat[operand])
            else
                push!(matrices, zero_block)
            end
        end

        return isempty(matrices) ? spzeros(Float64, 0, 0) : vcat(matrices...)

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

"""Check that operands are in a proper layout."""
function check_conditions(op::CartesianGradient)
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

"""Require operands to be in coefficient layout."""
function enforce_conditions(op::CartesianGradient)
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

"""Build operator matrix for a specific subproblem."""
function subproblem_matrix(op::CartesianDivergence, subproblem)
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

"""Check that operands are in a proper layout for divergence computation."""
function check_conditions(op::CartesianDivergence)
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

"""Ensure operands are in coefficient layout for differentiation."""
function enforce_conditions(op::CartesianDivergence)
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

"""Determine which variables the curl operator matrix depends on."""
function matrix_dependence(op::CartesianCurl, vars...)
    return matrix_dependence(op.operand, vars...)
end

"""Determine which variables couple through the curl operator."""
function matrix_coupling(op::CartesianCurl, vars...)
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

"""Build operator matrix for curl in a specific subproblem."""
function subproblem_matrix(op::CartesianCurl, subproblem)
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

"""Check that operands are in a proper layout for curl computation."""
function check_conditions(op::CartesianCurl)
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

"""Ensure operands are in coefficient layout for differentiation."""
function enforce_conditions(op::CartesianCurl)
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

"""Build operator matrix for a specific subproblem."""
function subproblem_matrix(op::CartesianLaplacian, subproblem)
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

"""Check that operands are in a proper layout for Laplacian computation."""
function check_conditions(op::CartesianLaplacian)
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

"""Ensure operands are in coefficient layout for differentiation."""
function enforce_conditions(op::CartesianLaplacian)
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

"""Build trace matrix."""
function subproblem_matrix(op::CartesianTrace, subproblem)
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

"""Build skew matrix."""
function subproblem_matrix(op::CartesianSkew, subproblem)
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
