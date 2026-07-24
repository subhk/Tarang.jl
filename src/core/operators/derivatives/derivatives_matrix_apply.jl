# Axis-wise dense and sparse matrix application helpers used by derivatives.

# ============================================================================
# Matrix Application Helpers
# ============================================================================

"""
    apply_matrix_along_axis(matrix, array, axis; out=nothing)

Apply matrix along any axis of an array.
Following array:77-82 and apply_dense:104-126 implementation.
"""
function apply_matrix_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    if issparse(matrix)
        return apply_sparse_along_axis(matrix, array, axis; out=out)
    else
        return apply_dense_along_axis(matrix, array, axis; out=out)
    end
end

"""
    apply_dense_along_axis(matrix, array, axis; out=nothing)

Apply dense matrix along any axis of an array.
Following apply_dense implementation in array:104-126.
"""
function apply_dense_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    ndim = ndims(array)
    use_gpu = is_gpu_array(array)
    arch = architecture(array)

    # Resolve wraparound axis
    axis = mod1(axis, ndim)

    out_is_gpu = out !== nothing && is_gpu_array(out)
    out !== nothing && use_gpu != out_is_gpu && throw(ArgumentError(
        "Input and output must use the same architecture; implicit CPU/GPU staging is disabled"))

    if out === nothing
        out_shape = ntuple(d -> d == axis ? size(matrix, 1) : size(array, d), ndim)
        # Promote so a complex matrix applied to a real array (or vice versa)
        # yields a buffer that can hold the result (else copyto! → InexactError).
        out_data = similar(array, promote_type(eltype(array), eltype(matrix)), out_shape)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_data = out
    end

    # The matrix is configuration data, not field state. Upload it once for a
    # GPU multiplication; the field array itself never leaves its device.
    matrix_data = use_gpu ? on_architecture(arch, Matrix(matrix)) : matrix
    work_array = array

    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        work_array = permutedims(work_array, perm)
    end

    array_shape = size(work_array)

    # Flatten later axes for matrix multiplication
    if ndim > 2
        work_array = reshape(work_array, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply matrix multiplication
    temp = matrix_data * work_array

    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end

    # Move axis back
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end

    copyto!(out_data, temp)
    return out_data
end

"""
    apply_sparse_along_axis(matrix, array, axis; out=nothing, check_shapes=false)

Apply sparse matrix along any axis of an array.
For GPU arrays the sparse matrix is densified as configuration data and uploaded;
the field array and result remain on-device.
Following apply_sparse implementation in array:171-203.
Note: Uses SparseMatrixCSC (Julia's sparse format) instead of CSR.
"""
function apply_sparse_along_axis(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int; out=nothing, check_shapes=false)
    ndim = ndims(array)

    use_gpu = is_gpu_array(array)
    out_is_gpu = out !== nothing && is_gpu_array(out)
    out !== nothing && use_gpu != out_is_gpu && throw(ArgumentError(
        "Input and output must use the same architecture; implicit CPU/GPU staging is disabled"))
    if use_gpu
        return apply_dense_along_axis(Matrix(matrix), array, axis; out=out)
    end
    array_cpu = array

    axis = mod1(axis, ndim)

    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        # Promote so a complex matrix × real array (or vice versa) fits the buffer.
        out_cpu = zeros(promote_type(eltype(array_cpu), eltype(matrix)), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = out
    end

    if check_shapes
        if !(1 <= axis <= ndim)
            throw(BoundsError("Axis out of bounds"))
        end
        if size(matrix, 2) != size(array_cpu, axis) || size(matrix, 1) != size(out_cpu, axis)
            throw(DimensionMismatch("Matrix shape mismatch"))
        end
    end

    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array_cpu = permutedims(array_cpu, perm)
    end

    array_shape = size(array_cpu)

    # Flatten later axes
    if ndim > 2
        array_cpu = reshape(array_cpu, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply sparse matrix multiplication
    temp = matrix * array_cpu

    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end

    # Move axis back
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end

    copyto!(out_cpu, temp)

    return out_cpu
end
