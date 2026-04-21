# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

"""
    drop_empty_rows(A)

Remove empty rows from sparse matrix.
Following tools/array drop_empty_rows.
"""
function drop_empty_rows(A::SparseMatrixCSC)
    m, n = size(A)
    if nnz(A) == 0
        return spzeros(eltype(A), 0, n)
    end
    # Use sparse structure directly instead of dense sum(abs.(A), dims=2)
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    has_nonzero = falses(m)
    @inbounds for i in eachindex(rows)
        if vals[i] != zero(eltype(A))
            has_nonzero[rows[i]] = true
        end
    end
    non_empty = findall(has_nonzero)
    if isempty(non_empty)
        return spzeros(eltype(A), 0, n)
    end
    if length(non_empty) == m
        return A  # All rows non-empty, no slicing needed
    end
    return A[non_empty, :]
end

"""
    zeros_with_pattern(matrices...)

Create zero matrix with combined sparsity pattern.
Following tools/array zeros_with_pattern.

Note: Julia's sparse format doesn't store explicit zeros, so we use
a tiny value (eps) to preserve the structural pattern. This is needed
for IMEX recombination where the sparsity pattern must be preserved.
"""
function zeros_with_pattern(matrices...)
    if isempty(matrices)
        return spzeros(ComplexF64, 0, 0)
    end

    # Get combined shape
    m = size(first(matrices), 1)
    n = size(first(matrices), 2)

    # Collect all non-zero positions
    rows = Int[]
    cols = Int[]

    for mat in matrices
        r, c, _ = findnz(mat)
        append!(rows, r)
        append!(cols, c)
    end

    if isempty(rows)
        return spzeros(ComplexF64, m, n)
    end

    # Use tiny non-zero values to preserve structural pattern
    # (Julia's sparse format optimizes away explicit zeros)
    # The sparse() call also handles duplicate indices by summing values
    placeholder_vals = fill(eps(Float64) * (1.0 + 0.0im), length(rows))
    return sparse(rows, cols, placeholder_vals, m, n)
end

"""
    expand_pattern(A, B)

Expand A to have same sparsity pattern as B.
Following tools/array expand_pattern.
"""
function expand_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    # Get pattern of B
    rows_b, cols_b, _ = findnz(B)

    # Get values of A at pattern positions
    vals = zeros(eltype(A), length(rows_b))
    for (i, (r, c)) in enumerate(zip(rows_b, cols_b))
        if r <= size(A, 1) && c <= size(A, 2)
            vals[i] = A[r, c]
        end
    end

    return sparse(rows_b, cols_b, vals, size(B)...)
end

"""
    apply_sparse(A, x; axis=1, out=nothing)

Apply sparse matrix along specified axis.
Following tools/array apply_sparse.

For axis=1: Applies A to x normally (A * x)
For axis=2: Applies A along the second axis of x (for 2D arrays: (A * x')')
            For 1D vectors, axis=2 is treated same as axis=1.
"""
function apply_sparse(A::SparseMatrixCSC, x::AbstractArray; axis::Int=1, out::Union{Nothing, AbstractArray}=nothing)
    if axis == 1
        result = A * x
    elseif axis == 2
        if ndims(x) == 1
            # For 1D vectors, axis=2 is equivalent to axis=1
            result = A * x
        else
            # For 2D+ arrays, multiply along second axis
            # This applies A to each row of x
            result = (A * x')'
        end
    else
        throw(ArgumentError("axis must be 1 or 2"))
    end

    if out !== nothing
        copyto!(out, result)
        return out
    end
    return result
end

