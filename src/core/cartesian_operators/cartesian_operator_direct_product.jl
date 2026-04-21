# Direct-product operator variants built on the Cartesian operator layer.

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
    DirectProductTrace <: AbstractLinearOperator

Trace operator for DirectProduct coordinate systems.
Following spectral methods pattern DirectProductTrace.

For a tensor T indexed by DirectProduct coordinates, the trace extracts
diagonal blocks for each subsystem and sums their traces.
"""
struct DirectProductTrace <: AbstractLinearOperator
    operand::Operand
    index::Int
    coordsys::DirectProduct
    args::Vector{Any}  # Trace operators for each subsystem's diagonal block

    function DirectProductTrace(operand::Operand; index::Int=0)
        tensorsig = get_tensorsig(operand)
        if isempty(tensorsig)
            throw(ArgumentError("DirectProductTrace requires a tensor operand"))
        end

        coordsys = tensorsig[index + 1]
        if !isa(coordsys, DirectProduct)
            throw(ArgumentError("DirectProductTrace requires DirectProduct coordinate system"))
        end

        # Build trace for each coordinate subsystem
        # For each cs in coordsystems:
        #   1. Extract component for first tensor index (index=0) corresponding to cs
        #   2. Extract component for second tensor index (index=1) corresponding to cs
        #   3. Take trace of the resulting sub-tensor
        args = Any[]
        for cs in coordsys.coordsystems
            # Double component extraction to get diagonal block
            comp1 = DirectProductComponent(operand, index=0, comp=cs)
            comp2 = DirectProductComponent(comp1, index=1, comp=cs)
            # Take trace of the block
            push!(args, Trace(comp2))
        end

        new(operand, index, coordsys, args)
    end
end

"""
    DirectProductCurl <: AbstractLinearOperator

Curl operator for 3D DirectProduct coordinate systems (product of 1D and 2D subsystems).

Following spectral methods pattern DirectProductCurl.

For a 3D vector field u on DirectProduct(cs1, cs2):
  curl(u)_i = ε_ijk ∂u_k/∂x_j
"""
struct DirectProductCurl <: AbstractLinearOperator
    operand::Operand
    index::Int
    coordsys::DirectProduct

    function DirectProductCurl(operand::Operand; index::Int=0)
        if !isa(operand, VectorField)
            throw(ArgumentError("DirectProductCurl requires a VectorField"))
        end

        tensorsig = get_tensorsig(operand)
        coordsys = tensorsig[index + 1]

        if !isa(coordsys, DirectProduct)
            throw(ArgumentError("DirectProductCurl requires DirectProduct coordinates"))
        end

        if coordsys.dim != 3
            throw(ArgumentError("DirectProductCurl requires 3D (product of 1D and 2D). Got dim=$(coordsys.dim)"))
        end

        # Validate 1D+2D decomposition
        dims = [cs.dim for cs in coordsys.coordsystems]
        if sort(dims) != [1, 2]
            throw(ArgumentError("DirectProductCurl requires product of 1D and 2D subsystems. Got dims=$dims"))
        end

        new(operand, index, coordsys)
    end
end

function matrix_dependence(op::DirectProductCurl, vars...)
    return matrix_dependence(op.operand, vars...)
end

function matrix_coupling(op::DirectProductCurl, vars...)
    result = falses(length(vars))
    for comp in op.operand.components
        deriv_coupling = matrix_coupling(comp, vars...)
        result .|= deriv_coupling
    end
    return result
end

function subproblem_matrix(op::DirectProductCurl, subproblem)
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

    # curl_1 = ∂u_3/∂x_2 - ∂u_2/∂x_3
    # curl_2 = ∂u_1/∂x_3 - ∂u_3/∂x_1
    # curl_3 = ∂u_2/∂x_1 - ∂u_1/∂x_2
    row1 = hcat(zero_block, -D[2, 3], D[3, 2])
    row2 = hcat(D[1, 3], zero_block, -D[3, 1])
    row3 = hcat(-D[1, 2], D[2, 1], zero_block)
    result = vcat(row1, row2, row3)

    if op.coordsys.right_handed === false
        result = -result
    end

    return result
end

function check_conditions(op::DirectProductCurl)
    if isa(op.operand, VectorField)
        layouts = Symbol[]
        for comp in op.operand.components
            if hasfield(typeof(comp), :current_layout) && comp.current_layout !== nothing
                push!(layouts, comp.current_layout)
            end
        end
        return length(Set(layouts)) <= 1
    end
    return true
end

function enforce_conditions(op::DirectProductCurl)
    if isa(op.operand, VectorField)
        for comp in op.operand.components
            if hasfield(typeof(comp), :current_layout)
                ensure_layout!(comp, :c)
            end
        end
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
