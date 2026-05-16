# ---------------------------------------------------------------------------
# Permutation matrices
# ---------------------------------------------------------------------------
#
# Subproblem matrices are assembled in a natural parser order, but the solve
# path expects Dedalus-style grouping by spectral mode, tensor component, and
# equation/variable dimension. The helpers below produce sparse permutation
# matrices instead of moving dense blocks by hand.

"""
    left_permutation(sp, equations, bc_top, interleave_components)

Left permutation acting on equations.
Following subsystems:614-675.

bc_top determines if lower-dimensional equations are placed at the top or bottom.

Input ordering: Equations > Components > Modes
Output ordering with interleave_components=true: Modes > Components > Equations
Output ordering with interleave_components=false: Modes > Equations > Components
"""
function _left_permutation_indices(sp::Subproblem, equations, eqn_sizes::AbstractVector{<:Integer}, bc_top::Bool, interleave_components::Bool)
    # Build a three-level hierarchy:
    #   equation -> component -> coefficient/mode
    # This mirrors the input order and lets the second pass invert the nesting.
    i = 0
    L0 = Vector{Vector{Vector{Int}}}()

    for (eq_data, eq_size) in zip(equations, eqn_sizes)
        L1 = Vector{Vector{Int}}()

        if eq_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end

        # Tensor signatures determine how many component blocks the equation
        # contributes. Scalars are treated as one-component tensors.
        tensorsig = get(eq_data, "tensorsig", ())
        n_comps = max(1, prod((cs isa CoordinateSystem ? cs.dim : 1 for cs in tensorsig); init=1))
        n_coeffs = eq_size ÷ n_comps

        for comp in 1:n_comps
            L2 = Int[]
            for coeff in 1:n_coeffs
                push!(L2, i)
                i += 1
            end
            push!(L1, L2)
        end
        push!(L0, L1)
    end

    # Reverse list hierarchy, grouping by equation domain dimension. The
    # `bc_top` flag controls whether lower-dimensional boundary rows appear
    # before or after bulk rows.
    indices_by_dim = Dict{Int, Vector{Int}}()
    n1max = length(L0)
    n2max = maximum(length(L1) for L1 in L0; init=1)
    n3max = maximum(length(L2) for L1 in L0 for L2 in L1; init=1)

    if interleave_components
        for n3 in 1:n3max
            for n2 in 1:n2max
                for n1 in 1:n1max
                    dim = get(equations[n1], "domain_dim", 0)
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    else
        for n3 in 1:n3max
            for n1 in 1:n1max
                dim = get(equations[n1], "domain_dim", 0)
                for n2 in 1:n2max
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    end

    # Combine indices by dimension
    dims = sort(collect(keys(indices_by_dim)))
    if !bc_top
        dims = reverse(dims)
    end

    indices = Int[]
    for dim in dims
        append!(indices, indices_by_dim[dim])
    end

    return indices .+ 1  # 1-based indexing
end

function left_permutation(sp::Subproblem, equations, eqn_sizes::AbstractVector{<:Integer}, bc_top::Bool, interleave_components::Bool)
    indices = _left_permutation_indices(sp, equations, eqn_sizes, bc_top, interleave_components)
    return perm_matrix(indices, length(indices))
end

function left_permutation(sp::Subproblem, equations, bc_top::Bool, interleave_components::Bool)
    eqn_sizes = [_subproblem_expr_dofs(sp, get(eq, "lhs", nothing)) for eq in equations]
    return left_permutation(sp, equations, eqn_sizes, bc_top, interleave_components)
end

"""
    right_permutation(sp, variables, tau_left, interleave_components)

Right permutation acting on variables.
Following subsystems:678-739.

tau_left determines if lower-dimensional variables are placed at the left or right.

Input ordering: Variables > Components > Modes
Output ordering with interleave_components=true: Modes > Components > Variables
Output ordering with interleave_components=false: Modes > Variables > Components
"""
function _right_permutation_indices(sp::Subproblem, variables, tau_left::Bool, interleave_components::Bool)
    # Mirror `_left_permutation_indices`, but for variable columns. Tau and
    # lower-dimensional variables can be placed left or right of bulk columns.
    i = 0
    L0 = Vector{Vector{Vector{Int}}}()

    for var in variables
        L1 = Vector{Vector{Int}}()
        var_size = subproblem_field_size(sp, var)

        if var_size == 0
            push!(L1, Int[])
            push!(L0, L1)
            continue
        end

        # A ScalarField has one component; VectorField/TensorField components
        # are flattened by `scalar_components`.
        components = scalar_components(var)
        n_comps = length(components)
        n_coeffs = var_size ÷ n_comps

        for comp in 1:n_comps
            L2 = Int[]
            for coeff in 1:n_coeffs
                push!(L2, i)
                i += 1
            end
            push!(L1, L2)
        end
        push!(L0, L1)
    end

    # Reorder columns into mode-major order, optionally interleaving components
    # inside each mode before moving to the next variable.
    indices_by_dim = Dict{Int, Vector{Int}}()
    n1max = length(L0)
    n2max = maximum(length(L1) for L1 in L0; init=1)
    n3max = maximum(length(L2) for L1 in L0 for L2 in L1; init=1)

    if interleave_components
        for n3 in 1:n3max
            for n2 in 1:n2max
                for n1 in 1:n1max
                    var = variables[n1]
                    dim = get_var_dim(var)
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    else
        for n3 in 1:n3max
            for n1 in 1:n1max
                var = variables[n1]
                dim = get_var_dim(var)
                for n2 in 1:n2max
                    try
                        idx = L0[n1][n2][n3]
                        if !haskey(indices_by_dim, dim)
                            indices_by_dim[dim] = Int[]
                        end
                        push!(indices_by_dim[dim], idx)
                    catch
                        continue
                    end
                end
            end
        end
    end

    # Combine indices by dimension
    dims = sort(collect(keys(indices_by_dim)))
    if !tau_left
        dims = reverse(dims)
    end

    indices = Int[]
    for dim in dims
        append!(indices, indices_by_dim[dim])
    end

    return indices .+ 1  # 1-based indexing
end

function right_permutation(sp::Subproblem, variables, tau_left::Bool, interleave_components::Bool)
    indices = _right_permutation_indices(sp, variables, tau_left, interleave_components)
    return perm_matrix(indices, length(indices))
end

"""
    get_var_dim(var)

Get dimension of variable's domain.
"""
function get_var_dim(var)
    if hasfield(typeof(var), :domain) && var.domain !== nothing
        return var.domain.dim
    end
    if hasfield(typeof(var), :bases)
        return length(var.bases)
    end
    return 0
end

"""
    perm_matrix(indices, n)

Create permutation matrix from index array.
Following tools/array perm_matrix.
"""
function perm_matrix(indices::Vector{Int}, n::Int)
    m = length(indices)
    if m == 0 || n == 0
        return spzeros(Float64, m, n)
    end

    # Row i selects source column `indices[i]`. Invalid indices are filtered so
    # partially masked mode lists produce a valid sparse matrix rather than an
    # out-of-bounds construction error.
    rows = collect(1:m)
    cols = indices
    vals = ones(Float64, m)

    # Filter valid indices
    valid = (cols .>= 1) .& (cols .<= n)
    rows = rows[valid]
    cols = cols[valid]
    vals = vals[valid]

    return sparse(rows, cols, vals, m, n)
end
