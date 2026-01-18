"""
Array manipulation utilities

## GPU Compatibility

Most functions in this module work with both CPU and GPU arrays:

- `reshape_vector`, `axindex`, `axslice`: GPU-compatible (use standard Julia operations)
- `apply_matrix`, `apply_dense`: GPU-compatible (use broadcasting/matmul)
- `expand_dims`, `squeeze_dims`: GPU-compatible (use reshape/dropdims)
- `broadcast_arrays`: GPU-compatible (uses similar())
- `safe_cast`, `promote_arrays`: GPU-compatible (use broadcasting)
- `is_contiguous`, `make_contiguous`: GPU-aware (handle CuArrays correctly)
- `interleave_matrices`: GPU-aware (preserves input array type)

Sparse matrix operations are CPU-only by design:
- `sparse_block_diag`, `add_sparse`, `perm_matrix_square`: CPU sparse matrices
- `apply_sparse_via_matrix`, `permute_axis`: CPU sparse matrix operations
- `scipy_sparse_eigs`: Uses Arpack (CPU only)
- `solve_upper_sparse`, `solve_lower_sparse`: CPU sparse solvers
"""

using LinearAlgebra
using SparseArrays
using Arpack

# Array copying and manipulation
# Note: Use Base.copyto! directly for array copying.
# This module provides additional array utilities beyond Base.

"""Reshape array to target shape."""
function reshape_vector(arr::AbstractArray, target_shape::Tuple)
    if prod(size(arr)) != prod(target_shape)
        throw(ArgumentError("Cannot reshape array of size $(size(arr)) to $target_shape"))
    end
    return reshape(arr, target_shape)
end

# Axis utilities

"""
    axindex(ndim::Int, axis::Int, index)

Create indexing tuple for specific axis.

Returns a tuple of length `ndim` with `Colon()` at all positions except
`axis`, which contains `index`.

# Example
```julia
axindex(3, 2, 5)  # Returns (Colon(), 5, Colon())
arr[axindex(ndims(arr), 2, 5)...]  # Equivalent to arr[:, 5, :]
```
"""
function axindex(ndim::Int, axis::Int, index)
    if axis < 1 || axis > ndim
        throw(ArgumentError("Axis $axis out of bounds for $ndim dimensions (must be 1 ≤ axis ≤ $ndim)"))
    end
    indices = [Colon() for _ in 1:ndim]
    indices[axis] = index
    return tuple(indices...)
end

"""
    axslice(ndim::Int, axis::Int, slice_range)

Create slicing tuple for specific axis.

Returns a tuple of length `ndim` with `Colon()` at all positions except
`axis`, which contains `slice_range`.

# Example
```julia
axslice(3, 2, 1:10)  # Returns (Colon(), 1:10, Colon())
arr[axslice(ndims(arr), 2, 1:10)...]  # Equivalent to arr[:, 1:10, :]
```
"""
function axslice(ndim::Int, axis::Int, slice_range)
    if axis < 1 || axis > ndim
        throw(ArgumentError("Axis $axis out of bounds for $ndim dimensions (must be 1 ≤ axis ≤ $ndim)"))
    end
    indices = [Colon() for _ in 1:ndim]
    indices[axis] = slice_range
    return tuple(indices...)
end

# Matrix operations along axes

"""
    apply_matrix(matrix::AbstractMatrix, array::AbstractArray, axis::Int)

Apply matrix multiplication along specified axis.

Multiplies `matrix` with slices of `array` along `axis`. The matrix columns
must match the size of `array` along `axis`.

GPU-aware: Works with GPU arrays if both `matrix` and `array` are on the same
device (both CPU or both GPU). Uses `permutedims`, `reshape`, and matrix
multiplication which are all GPU-compatible.

# Algorithm
1. Permute `array` to move `axis` to the first position
2. Reshape to 2D: (axis_size, other_dims_product)
3. Apply matrix: result = matrix * reshaped
4. Reshape and permute back to original axis order

# Example
```julia
M = rand(4, 3)           # 4×3 matrix
A = rand(3, 5, 6)        # Array with size 3 along axis 1
result = apply_matrix(M, A, 1)  # Returns array of size (4, 5, 6)
```
"""
function apply_matrix(matrix::AbstractMatrix, array::AbstractArray, axis::Int)
    if axis < 1 || axis > ndims(array)
        throw(ArgumentError("Axis $axis out of bounds for array with $(ndims(array)) dimensions (must be 1 ≤ axis ≤ $(ndims(array)))"))
    end

    # Move target axis to the front
    perm = [axis; setdiff(1:ndims(array), axis)]
    permuted_array = permutedims(array, perm)

    # Reshape to 2D: (axis_size, other_dims)
    original_shape = size(permuted_array)
    axis_size = original_shape[1]
    other_size = prod(original_shape[2:end])

    if size(matrix, 2) != axis_size
        throw(DimensionMismatch("Matrix columns ($(size(matrix, 2))) must match axis size ($axis_size)"))
    end
    
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

"""
    apply_dense(matrix, array)

Apply dense matrix to flattened array.

For a matrix of size (m, n) and array with length(array) == n,
returns a vector of length m.

For square matrices (m == n), can optionally reshape back to original shape
if the array was multi-dimensional.
"""
function apply_dense(matrix::AbstractMatrix, array::AbstractArray)
    flat_array = vec(array)
    n = length(flat_array)
    m, matrix_n = size(matrix)

    if matrix_n != n
        throw(DimensionMismatch("Matrix columns ($matrix_n) must match array length ($n)"))
    end

    result_flat = matrix * flat_array

    # Only reshape back if matrix is square and preserves dimensions
    if m == n && ndims(array) > 1
        return reshape(result_flat, size(array))
    else
        return result_flat
    end
end

# Note: apply_sparse is defined in subsystems.jl
# This wrapper delegates sparse matrix application to the general apply_matrix function

"""Apply sparse matrix along specified axis (alternative implementation)."""
function apply_sparse_via_matrix(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int=1)
    return apply_matrix(matrix, array, axis)
end

# Kronecker products
# Note: Use LinearAlgebra.kron directly for two-matrix Kronecker products.
# kron_multi provides Kronecker product of multiple matrices.

"""Kronecker product of multiple matrices."""
function kron_multi(matrices::AbstractMatrix...)
    if isempty(matrices)
        throw(ArgumentError("At least one matrix required"))
    end
    result = matrices[1]
    for i in 2:length(matrices)
        result = kron(result, matrices[i])
    end
    return result
end

# Sparse matrix utilities
# Note: These operations are CPU-only by design. SparseMatrixCSC is a CPU data structure.
# For GPU sparse operations, use CUDA.CUSPARSE directly.

"""Create block diagonal sparse matrix (CPU only)."""
function sparse_block_diag(matrices::AbstractMatrix...)
    if isempty(matrices)
        throw(ArgumentError("At least one matrix required"))
    end
    # Calculate dimensions
    total_rows = sum(size(M, 1) for M in matrices)
    total_cols = sum(size(M, 2) for M in matrices)
    
    # Build index arrays
    I_indices = Int[]
    J_indices = Int[]
    values = promote_type((eltype(M) for M in matrices)...)[]    
    
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

"""Add two sparse matrices."""
function add_sparse(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return A + B
end

# Note: perm_matrix is defined in subsystems.jl
# This is an alternative implementation for square permutation matrices

"""Create square permutation matrix from permutation vector."""
function perm_matrix_square(permutation::Vector{Int}, n::Int=length(permutation))
    if n < length(permutation)
        throw(ArgumentError("Permutation length exceeds matrix size"))
    end
    if any(j -> j < 1 || j > n, permutation)
        throw(ArgumentError("Permutation entries must be between 1 and $n"))
    end
    if length(unique(permutation)) != length(permutation)
        throw(ArgumentError("Permutation entries must be unique"))
    end
    P = spzeros(n, n)
    for (i, j) in enumerate(permutation)
        P[i, j] = 1
    end
    return P
end

"""Permute array along specified axis."""
function permute_axis(array::AbstractArray, axis::Int, permutation::Vector{Int})
    # Create permutation matrix (uses perm_matrix from subsystems.jl or local perm_matrix_square)
    n = length(permutation)
    P = perm_matrix_square(permutation, n)

    # Apply permutation
    return apply_matrix(P, array, axis)
end

# Matrix interleaving

"""
    interleave_matrices(matrices...)

Interleave matrices by rows (block-wise row interleaving).

Given k matrices of size (m, n), produces a matrix of size (m*k, n) where
rows are interleaved: row 1 of matrix 1, row 1 of matrix 2, ..., row 1 of matrix k,
row 2 of matrix 1, row 2 of matrix 2, etc.

GPU-aware: If input matrices are GPU arrays, the result will also be a GPU array.

# Arguments
- `matrices`: Two or more matrices of the same size

# Returns
A matrix of size (m * num_matrices, n) with interleaved rows.

# Example
```julia
A = [1 2; 3 4]      # 2×2
B = [5 6; 7 8]      # 2×2
interleave_matrices(A, B)
# Returns: [1 2; 5 6; 3 4; 7 8]  # 4×2
```
"""
function interleave_matrices(matrices::AbstractMatrix...)
    if isempty(matrices)
        throw(ArgumentError("At least one matrix required"))
    end

    # Check dimensions are compatible
    m, n = size(matrices[1])
    for matrix in matrices[2:end]
        if size(matrix) != (m, n)
            throw(ArgumentError("All matrices must have the same size, got $(size(matrices[1])) and $(size(matrix))"))
        end
    end

    num_matrices = length(matrices)

    # Use promote_type to handle mixed element types correctly
    # This prevents data truncation when mixing e.g., Int and Float64
    common_eltype = promote_type((eltype(M) for M in matrices)...)

    # GPU-aware allocation: use similar() to preserve array type
    # Then fill with zeros
    result = similar(matrices[1], common_eltype, m * num_matrices, n)
    fill!(result, zero(common_eltype))

    for (k, matrix) in enumerate(matrices)
        rows = k:num_matrices:m*num_matrices
        result[rows, :] .= matrix
    end

    return result
end

# Eigenvalue utilities for sparse matrices

"""Compute eigenvalues of sparse matrix using Arpack."""
function scipy_sparse_eigs(A::SparseMatrixCSC, B::Union{SparseMatrixCSC, Nothing}=nothing;
                          nev::Int=6, which::Symbol=:LM, sigma=nothing)
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

"""Solve upper triangular sparse system."""
function solve_upper_sparse(A::SparseMatrixCSC, b::AbstractVector)
    return A \ b
end

"""Solve lower triangular sparse system."""
function solve_lower_sparse(A::SparseMatrixCSC, b::AbstractVector)
    return A \ b
end

# Array broadcasting utilities

"""
    broadcast_arrays(arrays...)

Broadcast arrays to common shape, returning a tuple of arrays.

Each array is expanded to the common broadcast shape using Julia's
broadcasting rules (singleton dimensions are expanded).

GPU-aware: Uses `similar()` to preserve array type. If inputs are GPU arrays,
outputs will also be GPU arrays.

Example:
    a = [1, 2, 3]           # (3,)
    b = [1 2; 3 4; 5 6]     # (3, 2)
    a_bc, b_bc = broadcast_arrays(a, b)
    # a_bc is (3, 2), b_bc is (3, 2)
"""
function broadcast_arrays(arrays::AbstractArray...)
    if isempty(arrays)
        return ()
    end

    # Compute the common broadcast shape
    common_shape = Base.Broadcast.broadcast_shape(map(size, arrays)...)

    result = map(arrays) do arr
        out = similar(arr, common_shape)
        out .= arr
        out
    end

    return Tuple(result)
end

"""
    expand_dims(arr::AbstractArray, axis::Int)

Add singleton dimension at specified axis position.

Inserts a new dimension of size 1 at position `axis`. The original dimensions
are shifted to accommodate the new dimension.

# Arguments
- `arr`: Input array
- `axis`: Position for new dimension (1 ≤ axis ≤ ndims(arr) + 1)

# Example
```julia
arr = rand(3, 4)              # Size (3, 4)
expand_dims(arr, 1)           # Size (1, 3, 4)
expand_dims(arr, 2)           # Size (3, 1, 4)
expand_dims(arr, 3)           # Size (3, 4, 1)
```
"""
function expand_dims(arr::AbstractArray, axis::Int)
    current_shape = size(arr)
    max_axis = ndims(arr) + 1
    if axis < 1 || axis > max_axis
        throw(ArgumentError("Invalid axis $axis for array with $(ndims(arr)) dimensions (must be 1 ≤ axis ≤ $max_axis)"))
    end

    new_shape = tuple(
        current_shape[1:axis-1]...,
        1,
        current_shape[axis:end]...
    )

    return reshape(arr, new_shape)
end

"""Remove singleton dimensions."""
function squeeze_dims(arr::AbstractArray, dims::Union{Int, Tuple{Vararg{Int}}})
    if isa(dims, Int)
        dims = (dims,)
    end

    nd = ndims(arr)
    if any(d -> d < 1 || d > nd, dims)
        throw(ArgumentError("Dims must be between 1 and $nd"))
    end
    if length(unique(dims)) != length(dims)
        throw(ArgumentError("Dims must be unique"))
    end

    for d in dims
        s = size(arr, d)
        if s != 1
            throw(ArgumentError("Cannot squeeze dimension $d with size $s"))
        end
    end

    return dropdims(arr; dims=dims)
end

# Memory layout utilities

"""
    is_contiguous(arr::AbstractArray)

Check if array has contiguous memory layout.

GPU-aware: Returns true for both CPU Arrays and GPU CuArrays,
as both have contiguous memory layouts by default.
"""
function is_contiguous(arr::AbstractArray)
    # Julia Arrays are column-major contiguous by default
    # GPU arrays (CuArray) are also contiguous in device memory
    # Check for standard Array or GPU array types
    if arr isa Array
        return true
    end
    # Check if it's a GPU array by checking for the is_gpu_array function
    # This avoids direct dependency on CUDA.jl
    if is_gpu_array(arr)
        return true
    end
    # For other AbstractArray types (views, reshapes, etc.), check if parent is contiguous
    if hasproperty(arr, :parent)
        return is_contiguous(arr.parent)
    end
    return false
end

"""
    make_contiguous(arr::AbstractArray)

Ensure array has contiguous memory layout.

GPU-aware: Preserves array architecture. If input is a GPU array,
the output will also be a GPU array. Uses `similar` for allocation
to maintain the same array type.
"""
function make_contiguous(arr::AbstractArray)
    if is_contiguous(arr)
        return arr
    else
        # Use similar to preserve array type (CPU or GPU)
        # Then copy data
        result = similar(arr)
        copyto!(result, arr)
        return result
    end
end

# Type conversion utilities

"""
    safe_cast(arr, target_type)

Safely cast array to target type with truncation for inexact conversions.

For floating-point to integer conversions, uses `trunc` to avoid InexactError.
For other conversions, attempts direct conversion first.
"""
function safe_cast(arr::AbstractArray{T}, target_type::Type{U}) where {T, U}
    # Check if we need special handling for float-to-int conversion
    if T <: AbstractFloat && U <: Integer
        @warn "Converting from $T to $U using truncation"
        return U.(trunc.(arr))
    end

    # Check for complex-to-real conversion
    if T <: Complex && !(U <: Complex)
        @warn "Converting from $T to $U, imaginary parts will be discarded"
        return convert.(U, real.(arr))
    end

    # Standard conversion
    try
        return convert.(U, arr)
    catch e
        if e isa InexactError
            @warn "Inexact conversion from $T to $U, using rounding"
            # For numeric types, try rounding
            if U <: Integer && T <: Real
                return U.(round.(arr))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
end

"""Promote arrays to common element type, returning a tuple of converted arrays."""
function promote_arrays(arrays::AbstractArray...)
    if isempty(arrays)
        return ()
    end
    common_type = promote_type((eltype(arr) for arr in arrays)...)
    return Tuple(convert.(common_type, arr) for arr in arrays)
end

# ============================================================================
# Exports
# ============================================================================

# Array reshaping and indexing
export reshape_vector, axindex, axslice

# Matrix operations along axes
export apply_matrix, apply_dense, apply_sparse_via_matrix

# Kronecker products
export kron_multi

# Sparse matrix utilities
export sparse_block_diag, add_sparse, perm_matrix_square

# Array permutation
export permute_axis

# Matrix interleaving
export interleave_matrices

# Eigenvalue utilities
export scipy_sparse_eigs

# Linear solver utilities
export solve_upper_sparse, solve_lower_sparse

# Array broadcasting utilities
export broadcast_arrays, expand_dims, squeeze_dims

# Memory layout utilities
export is_contiguous, make_contiguous

# Type conversion utilities
export safe_cast, promote_arrays
