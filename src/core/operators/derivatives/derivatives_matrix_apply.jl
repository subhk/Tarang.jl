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
    array_cpu = use_gpu ? Array(array) : array

    # Resolve wraparound axis
    axis = mod1(axis, ndim)

    out_is_gpu = out !== nothing && is_gpu_array(out)

    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        out_cpu = zeros(eltype(array_cpu), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = out_is_gpu ? Array(out) : out
    end

    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array_cpu = permutedims(array_cpu, perm)
    end

    array_shape = size(array_cpu)

    # Flatten later axes for matrix multiplication
    if ndim > 2
        array_cpu = reshape(array_cpu, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply matrix multiplication
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

    if use_gpu || out_is_gpu
        if out === nothing
            return on_architecture(arch, out_cpu)
        else
            if out_is_gpu
                out .= copy_to_device(out_cpu, out)
            else
                copyto!(out, out_cpu)
            end
            return out
        end
    else
        return out_cpu
    end
end

"""
    apply_sparse_along_axis(matrix, array, axis; out=nothing, check_shapes=false)

Apply sparse matrix along any axis of an array.
Supports both CPU and GPU arrays (GPU arrays are copied to CPU for sparse operations).
Following apply_sparse implementation in array:171-203.
Note: Uses SparseMatrixCSC (Julia's sparse format) instead of CSR.
"""
function apply_sparse_along_axis(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int; out=nothing, check_shapes=false)
    ndim = ndims(array)

    use_gpu = is_gpu_array(array)
    if use_gpu
        array_cpu = Array(array)
    else
        array_cpu = array
    end

    axis = mod1(axis, ndim)

    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        out_cpu = zeros(eltype(array_cpu), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = use_gpu ? Array(out) : out
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

    if use_gpu
        if out === nothing
            return copy_to_device(out_cpu, array)
        else
            out .= copy_to_device(out_cpu, out)
            return out
        end
    else
        return out_cpu
    end
end
