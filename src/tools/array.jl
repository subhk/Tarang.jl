"""
Array manipulation utilities
"""

using LinearAlgebra
using SparseArrays
using Arpack

# Array copying and manipulation
function copyto!(dest::AbstractArray, src::AbstractArray)
    """Copy array data with proper type handling"""
    dest .= src
    return dest
end

function reshape_vector(arr::AbstractArray, target_shape::Tuple)
    """Reshape array to target shape"""
    if prod(size(arr)) != prod(target_shape)
        throw(ArgumentError("Cannot reshape array of size $(size(arr)) to $target_shape"))
    end
    return reshape(arr, target_shape)
end

# Axis utilities
function axindex(ndim::Int, axis::Int, index)
    """Create indexing tuple for specific axis"""
    indices = [Colon() for _ in 1:ndim]
    indices[axis] = index
    return tuple(indices...)
end

function axslice(ndim::Int, axis::Int, slice_range)
    """Create slicing tuple for specific axis"""
    indices = [Colon() for _ in 1:ndim]
    indices[axis] = slice_range
    return tuple(indices...)
end

# Matrix operations along axes
function apply_matrix(matrix::AbstractMatrix, array::AbstractArray, axis::Int)
    """Apply matrix multiplication along specified axis"""
    
    # Move target axis to the front
    perm = [axis; setdiff(1:ndims(array), axis)]
    permuted_array = permutedims(array, perm)
    
    # Reshape to 2D: (axis_size, other_dims)
    original_shape = size(permuted_array)
    axis_size = original_shape[1]
    other_size = prod(original_shape[2:end])
    
    reshaped = reshape(permuted_array, axis_size, other_size)
    
    # Apply matrix
    result_reshaped = matrix * reshaped
    
    # Reshape back
    result_shape = (size(matrix, 1), original_shape[2:end]...)
    result_permuted = reshape(result_reshaped, result_shape)
    
    # Permute back to original axis order
    inv_perm = invperm(perm)
    result = permutedims(result_permuted, inv_perm)
    
    return result
end

function apply_dense(matrix::AbstractMatrix, array::AbstractArray)
    """Apply dense matrix to flattened array"""
    flat_array = vec(array)
    result_flat = matrix * flat_array
    return reshape(result_flat, size(array))
end

function apply_sparse(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int=1)
    """Apply sparse matrix along specified axis"""
    return apply_matrix(matrix, array, axis)
end

# Kronecker products
function kron(A::AbstractMatrix, B::AbstractMatrix)
    """Kronecker product of two matrices"""
    return LinearAlgebra.kron(A, B)
end

function kron_multi(matrices::AbstractMatrix...)
    """Kronecker product of multiple matrices"""
    result = matrices[1]
    for i in 2:length(matrices)
        result = kron(result, matrices[i])
    end
    return result
end

# Sparse matrix utilities
function sparse_block_diag(matrices::AbstractMatrix...)
    """Create block diagonal sparse matrix"""
    # Calculate dimensions
    total_rows = sum(size(M, 1) for M in matrices)
    total_cols = sum(size(M, 2) for M in matrices)
    
    # Build index arrays
    I_indices = Int[]
    J_indices = Int[]
    values = eltype(matrices[1])[]
    
    row_offset = 0
    col_offset = 0
    
    for matrix in matrices
        m, n = size(matrix)
        
        if isa(matrix, SparseMatrixCSC)
            # Handle sparse matrices
            rows, cols, vals = findnz(matrix)
            append!(I_indices, rows .+ row_offset)
            append!(J_indices, cols .+ col_offset)
            append!(values, vals)
        else
            # Handle dense matrices
            for j in 1:n, i in 1:m
                if matrix[i, j] != 0
                    push!(I_indices, i + row_offset)
                    push!(J_indices, j + col_offset)
                    push!(values, matrix[i, j])
                end
            end
        end
        
        row_offset += m
        col_offset += n
    end
    
    return sparse(I_indices, J_indices, values, total_rows, total_cols)
end

function add_sparse(A::SparseMatrixCSC, B::SparseMatrixCSC)
    """Add two sparse matrices"""
    return A + B
end

# Permutation matrices
function perm_matrix(permutation::Vector{Int}, n::Int=length(permutation))
    """Create permutation matrix from permutation vector"""
    P = spzeros(n, n)
    for (i, j) in enumerate(permutation)
        P[i, j] = 1
    end
    return P
end

function permute_axis(array::AbstractArray, axis::Int, permutation::Vector{Int})
    """Permute array along specified axis"""
    
    # Create permutation matrix
    P = perm_matrix(permutation)
    
    # Apply permutation
    return apply_matrix(P, array, axis)
end

# Matrix interleaving
function interleave_matrices(matrices::AbstractMatrix...)
    """Interleave matrices (block-wise interleaving)"""
    
    if length(matrices) == 0
        throw(ArgumentError("At least one matrix required"))
    end
    
    # Check dimensions are compatible
    m, n = size(matrices[1])
    for matrix in matrices[2:end]
        if size(matrix) != (m, n)
            throw(ArgumentError("All matrices must have the same size"))
        end
    end
    
    num_matrices = length(matrices)
    result = zeros(eltype(matrices[1]), m * num_matrices, n)
    
    for (k, matrix) in enumerate(matrices)
        rows = k:num_matrices:m*num_matrices
        result[rows, :] = matrix
    end
    
    return result
end

# Eigenvalue utilities for sparse matrices
function scipy_sparse_eigs(A::SparseMatrixCSC, B::Union{SparseMatrixCSC, Nothing}=nothing;
                          nev::Int=6, which::Symbol=:LM, sigma=nothing)
    """Compute eigenvalues of sparse matrix using Arpack"""

    if B === nothing
        # Standard eigenvalue problem
        if sigma === nothing
            if which == :LM
                return eigs(A, nev=nev, which=:LM)
            elseif which == :SM
                return eigs(A, nev=nev, which=:SM)
            elseif which == :LR
                return eigs(A, nev=nev, which=:LR)
            elseif which == :SR
                return eigs(A, nev=nev, which=:SR)
            else
                return eigs(A, nev=nev, which=which)
            end
        else
            # Shift-invert mode
            return eigs(A, nev=nev, sigma=sigma)
        end
    else
        # Generalized eigenvalue problem
        if sigma === nothing
            return eigs(A, B, nev=nev, which=which)
        else
            return eigs(A, B, nev=nev, sigma=sigma)
        end
    end
end

# Linear solver utilities
function solve_upper_sparse(A::SparseMatrixCSC, b::AbstractVector)
    """Solve upper triangular sparse system"""
    return A \ b
end

function solve_lower_sparse(A::SparseMatrixCSC, b::AbstractVector)
    """Solve lower triangular sparse system"""
    return A \ b
end

# Array broadcasting utilities
function broadcast_arrays(arrays::AbstractArray...)
    """Broadcast arrays to common shape"""
    return broadcast(identity, arrays...)
end

function expand_dims(arr::AbstractArray, axis::Int)
    """Add singleton dimension at specified axis"""
    current_shape = size(arr)
    if axis < 1 || axis > ndims(arr) + 1
        throw(BoundsError("Invalid axis $axis for array with $(ndims(arr)) dimensions"))
    end
    
    new_shape = tuple(
        current_shape[1:axis-1]..., 
        1, 
        current_shape[axis:end]...
    )
    
    return reshape(arr, new_shape)
end

function squeeze_dims(arr::AbstractArray, dims::Union{Int, Tuple{Vararg{Int}}})
    """Remove singleton dimensions"""
    if isa(dims, Int)
        dims = (dims,)
    end
    
    current_shape = size(arr)
    new_shape = []
    
    for (i, s) in enumerate(current_shape)
        if i in dims
            if s != 1
                throw(ArgumentError("Cannot squeeze dimension $i with size $s"))
            end
        else
            push!(new_shape, s)
        end
    end
    
    return reshape(arr, tuple(new_shape...))
end

# Memory layout utilities
function is_contiguous(arr::AbstractArray)
    """Check if array has contiguous memory layout"""
    # Julia arrays are column-major contiguous by default
    return arr isa Array
end

function make_contiguous(arr::AbstractArray)
    """Ensure array has contiguous memory layout"""
    if is_contiguous(arr)
        return arr
    else
        return collect(arr)
    end
end

# Type conversion utilities
function safe_cast(arr::AbstractArray{T}, target_type::Type{U}) where {T, U}
    """Safely cast array to target type"""
    try
        return convert.(target_type, arr)
    catch InexactError
        @warn "Inexact conversion from $T to $U, data may be truncated"
        return convert.(target_type, arr)
    end
end

function promote_arrays(arrays::AbstractArray...)
    """Promote arrays to common element type"""
    common_type = promote_type((eltype(arr) for arr in arrays)...)
    return (convert.(common_type, arr) for arr in arrays)
end