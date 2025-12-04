"""
Linear Algebra Operations for Spectral Methods

This module implements high-performance matrix-vector and matrix-matrix operations
for spectral PDE solvers, with particular focus on:
- Sparse matrix operations for differential operators
- Dense BLAS operations for transforms and nonlinear terms
- Memory-efficient tensor operations
- SIMD and threading implementations
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using LoopVectorization
using StaticArrays
using BenchmarkTools

# Import BLAS symbols for direct calls
import LinearAlgebra.BLAS: gemm!, gemv!, axpy!, scal!

# Matrix-vector multiplication types
abstract type MatVecOp end

struct SparseMatVec <: MatVecOp
    matrix::SparseMatrixCSC
    workspace::Vector{Float64}
    
    function SparseMatVec(matrix::SparseMatrixCSC{T}) where T
        workspace = Vector{T}(undef, size(matrix, 1))
        new(matrix, workspace)
    end
end

struct DenseMatVec <: MatVecOp
    matrix::Matrix{Float64}
    transposed::Bool
    
    function DenseMatVec(matrix::Matrix{T}; transposed::Bool=false) where T
        new(matrix, transposed)
    end
end

struct BlockSparseMatVec <: MatVecOp
    blocks::Vector{SparseMatrixCSC}
    block_structure::Matrix{Int}  # Maps to block indices
    workspace::Vector{Vector{Float64}}
end

# Matrix-matrix multiplication types
abstract type MatMatOp end

struct SparseDenseMatMat <: MatMatOp
    sparse_matrix::SparseMatrixCSC
    dense_workspace::Matrix{Float64}
end

struct DenseDenseMatMat <: MatMatOp
    use_blas::Bool
    use_threads::Bool
    block_size::Int
    
    function DenseDenseMatMat(; use_blas::Bool=true, use_threads::Bool=true, block_size::Int=64)
        new(use_blas, use_threads, block_size)
    end
end

struct TensorMatMat <: MatMatOp
    # For operations like A ⊗ B * vec(C) common in spectral methods
    kronecker_factors::Vector{Matrix{Float64}}
    temp_matrices::Vector{Matrix{Float64}}
end

# Performance monitoring
mutable struct LinalgPerformanceStats
    matvec_calls::Int
    matvec_time::Float64
    matmat_calls::Int  
    matmat_time::Float64
    sparse_ops::Int
    dense_ops::Int
    blas_ops::Int
    
    function LinalgPerformanceStats()
        new(0, 0.0, 0, 0.0, 0, 0, 0)
    end
end

const GLOBAL_LINALG_STATS = LinalgPerformanceStats()

# Matrix-vector operations
function fast_matvec!(y::AbstractVector, op::SparseMatVec, x::AbstractVector, α::Real=1.0, β::Real=0.0)
    """Sparse matrix-vector multiplication: y = α*A*x + β*y"""
    
    start_time = time()
    
    # Use sparse matrix-vector multiplication
    if β == 0.0
        # y = α*A*x (can skip the β*y term)
        if α == 1.0
            # Most common case: y = A*x
            mul!(y, op.matrix, x)
        else
            # y = α*A*x
            mul!(y, op.matrix, x)
            scal!(length(y), α, y, 1)
        end
    else
        # General case: y = α*A*x + β*y
        if β != 1.0
            scal!(length(y), β, y, 1)
        end
        mul!(op.workspace, op.matrix, x)
        axpy!(α, op.workspace, y)
    end
    
    # Update statistics
    GLOBAL_LINALG_STATS.matvec_calls += 1
    GLOBAL_LINALG_STATS.matvec_time += time() - start_time
    GLOBAL_LINALG_STATS.sparse_ops += 1
    
    return y
end

function fast_matvec!(y::AbstractVector, op::DenseMatVec, x::AbstractVector, α::Real=1.0, β::Real=0.0)
    """Dense matrix-vector multiplication using BLAS"""
    
    start_time = time()
    
    # Use BLAS GEMV for optimal performance
    if op.transposed
        # y = α*A^T*x + β*y
        gemv!('T', α, op.matrix, x, β, y)
    else
        # y = α*A*x + β*y  
        gemv!('N', α, op.matrix, x, β, y)
    end
    
    # Update statistics
    GLOBAL_LINALG_STATS.matvec_calls += 1
    GLOBAL_LINALG_STATS.matvec_time += time() - start_time
    GLOBAL_LINALG_STATS.blas_ops += 1
    
    return y
end

function fast_matvec!(y::AbstractVector, op::BlockSparseMatVec, x::AbstractVector, α::Real=1.0, β::Real=0.0)
    """Block sparse matrix-vector multiplication for structured problems"""
    
    start_time = time()
    
    # Initialize result
    if β == 0.0
        fill!(y, 0.0)
    elseif β != 1.0
        scal!(length(y), β, y, 1)
    end
    
    # Process each block
    for (block_idx, block) in enumerate(op.blocks)
        if block !== nothing
            # Determine block ranges
            i_range, j_range = get_block_ranges(op.block_structure, block_idx)
            
            # Extract sub-vectors
            x_sub = view(x, j_range)
            y_sub = view(y, i_range)
            
            # Block multiplication: y_sub += α * block * x_sub
            mul!(op.workspace[block_idx], block, x_sub)
            axpy!(α, op.workspace[block_idx], y_sub)
        end
    end
    
    # Update statistics
    GLOBAL_LINALG_STATS.matvec_calls += 1
    GLOBAL_LINALG_STATS.matvec_time += time() - start_time
    GLOBAL_LINALG_STATS.sparse_ops += 1
    
    return y
end

# Matrix-matrix operations
function fast_matmat!(C::AbstractMatrix, op::SparseDenseMatMat, A_is_sparse::Bool, A::AbstractMatrix, B::AbstractMatrix, α::Real=1.0, β::Real=0.0)
    """Sparse-dense matrix multiplication"""
    
    start_time = time()
    
    if A_is_sparse
        # C = α*A_sparse*B_dense + β*C where A is sparse
        if β == 0.0
            mul!(C, op.sparse_matrix, B)
            if α != 1.0
                C .*= α
            end
        else
            # Use workspace to avoid overwriting C
            mul!(op.dense_workspace, op.sparse_matrix, B)
            # C = α*workspace + β*C
            @. C = α * op.dense_workspace + β * C
        end
    else
        # C = α*A_dense*B_sparse + β*C where B is sparse  
        if β == 0.0
            mul!(C, A, op.sparse_matrix)
            if α != 1.0
                C .*= α
            end
        else
            mul!(op.dense_workspace, A, op.sparse_matrix)
            @. C = α * op.dense_workspace + β * C
        end
    end
    
    # Update statistics
    GLOBAL_LINALG_STATS.matmat_calls += 1
    GLOBAL_LINALG_STATS.matmat_time += time() - start_time
    GLOBAL_LINALG_STATS.sparse_ops += 1
    
    return C
end

function fast_matmat!(C::AbstractMatrix, op::DenseDenseMatMat, A::AbstractMatrix, B::AbstractMatrix, α::Real=1.0, β::Real=0.0)
    """Dense matrix multiplication with BLAS"""
    
    start_time = time()
    
    if op.use_blas && size(A, 1) * size(B, 2) * size(A, 2) > 1000  # Use BLAS for large matrices
        # Use BLAS GEMM: C = α*A*B + β*C
        gemm!('N', 'N', α, A, B, β, C)
        GLOBAL_LINALG_STATS.blas_ops += 1
        
    elseif op.use_threads && size(A, 1) > op.block_size
        # Use threaded block multiplication for medium matrices
        threaded_block_matmat!(C, A, B, α, β, op.block_size)
        
    else
        # Use vectorized multiplication for small matrices
        vectorized_matmat!(C, A, B, α, β)
    end
    
    # Update statistics
    GLOBAL_LINALG_STATS.matmat_calls += 1
    GLOBAL_LINALG_STATS.matmat_time += time() - start_time
    GLOBAL_LINALG_STATS.dense_ops += 1
    
    return C
end

function fast_matmat!(C::AbstractMatrix, op::TensorMatMat, vec_C::AbstractVector, α::Real=1.0, β::Real=0.0)
    """Kronecker product matrix multiplication: C = α*(A₁⊗A₂⊗...)*vec(C) + β*C"""
    
    start_time = time()
    
    # Reshape vector to matrix form for Kronecker operations
    n_factors = length(op.kronecker_factors)
    
    if n_factors == 2
        # Most common case: A₁ ⊗ A₂
        A1, A2 = op.kronecker_factors[1], op.kronecker_factors[2]
        m1, n1 = size(A1)
        m2, n2 = size(A2)
        
        # Reshape vec(C) to matrix form
        C_mat = reshape(vec_C, (n1, n2))
        result_mat = op.temp_matrices[1]  # Pre-allocated workspace
        
        # Kronecker product multiplication: (A₁ ⊗ A₂) * vec(C) = vec(A₂ * C * A₁ᵀ)
        # Step 1: temp = A₂ * C  
        mul!(op.temp_matrices[2], A2, C_mat)
        # Step 2: result = temp * A₁ᵀ
        mul!(result_mat, op.temp_matrices[2], A1')
        
        # Apply scaling and accumulation
        if β == 0.0
            vec(C) .= α .* vec(result_mat)
        else
            vec(C) .= α .* vec(result_mat) .+ β .* vec(C)
        end
        
    else
        # General case for multiple Kronecker factors
        error("General Kronecker products with >2 factors not yet implemented")
    end
    
    # Update statistics  
    GLOBAL_LINALG_STATS.matmat_calls += 1
    GLOBAL_LINALG_STATS.matmat_time += time() - start_time
    GLOBAL_LINALG_STATS.dense_ops += 1
    
    return C
end

# Specialized optimizations
@inline function vectorized_matmat!(C, A, B, α, β)
    """SIMD small matrix multiplication"""
    
    m, n, k = size(A, 1), size(B, 2), size(A, 2)
    
    if β == 0.0
        @avx for i in 1:m, j in 1:n
            Cᵢⱼ = zero(eltype(C))
            for l in 1:k
                Cᵢⱼ += A[i,l] * B[l,j]
            end
            C[i,j] = α * Cᵢⱼ
        end
    else
        @avx for i in 1:m, j in 1:n
            Cᵢⱼ = zero(eltype(C))
            for l in 1:k
                Cᵢⱼ += A[i,l] * B[l,j]
            end
            C[i,j] = α * Cᵢⱼ + β * C[i,j]
        end
    end
end

function threaded_block_matmat!(C, A, B, α, β, block_size)
    """Threaded block matrix multiplication"""
    
    m, n, k = size(A, 1), size(B, 2), size(A, 2)
    
    Threads.@threads for i_block in 1:block_size:m
        i_end = min(i_block + block_size - 1, m)
        i_range = i_block:i_end
        
        for j_block in 1:block_size:n
            j_end = min(j_block + block_size - 1, n)
            j_range = j_block:j_end
            
            # Block multiplication
            C_block = view(C, i_range, j_range)
            A_block = view(A, i_range, :)
            B_block = view(B, :, j_range)
            
            # Use BLAS for block
            gemm!('N', 'N', α, A_block, B_block, β, C_block)
        end
    end
end

# Memory-efficient operations for large systems
function streaming_matvec!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, chunk_size::Int=1024)
    """Memory-efficient matrix-vector multiplication for very large matrices"""
    
    n, m = size(A)
    fill!(y, 0.0)
    
    for i_start in 1:chunk_size:n
        i_end = min(i_start + chunk_size - 1, n)
        i_range = i_start:i_end
        
        # Process chunk
        y_chunk = view(y, i_range)
        A_chunk = view(A, i_range, :)
        
        mul!(y_chunk, A_chunk, x)
    end
    
    return y
end

function cache_efficient_matmat!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, cache_size::Int=32768)
    """Cache-efficient matrix multiplication using blocking"""
    
    m, n, k = size(A, 1), size(B, 2), size(A, 2)
    
    # Determine block sizes based on cache size
    block_size = Int(sqrt(cache_size ÷ sizeof(eltype(A))))
    
    fill!(C, 0.0)
    
    for kk in 1:block_size:k
        k_end = min(kk + block_size - 1, k)
        k_range = kk:k_end
        
        for jj in 1:block_size:n
            j_end = min(jj + block_size - 1, n)
            j_range = jj:j_end
            
            B_block = view(B, k_range, j_range)
            
            for ii in 1:block_size:m
                i_end = min(ii + block_size - 1, m)
                i_range = ii:i_end
                
                A_block = view(A, i_range, k_range)
                C_block = view(C, i_range, j_range)
                
                # Accumulate: C_block += A_block * B_block
                mul!(C_block, A_block, B_block, 1.0, 1.0)
            end
        end
    end
    
    return C
end

# Integration with spectral operators
function create_operator(matrix::AbstractMatrix, operation_type::Symbol)
    """Create operator for spectral method matrices"""
    
    if issparse(matrix)
        if operation_type == :matvec
            return SparseMatVec(matrix)
        elseif operation_type == :matmat
            workspace = Matrix{eltype(matrix)}(undef, size(matrix, 1), 64)  # Default columns
            return SparseDenseMatMat(matrix, workspace)
        end
    else
        if operation_type == :matvec
            return DenseMatVec(matrix)
        elseif operation_type == :matmat
            return DenseDenseMatMat()
        end
    end
end

function create_kronecker_operator(factors::Vector{<:AbstractMatrix})
    """Create Kronecker product operator"""
    
    # Pre-allocate workspace matrices
    temp_matrices = Vector{Matrix{Float64}}()
    
    if length(factors) == 2
        A1, A2 = factors[1], factors[2]
        push!(temp_matrices, Matrix{Float64}(undef, size(A1, 1), size(A2, 2)))
        push!(temp_matrices, Matrix{Float64}(undef, size(A2, 1), size(A1, 2)))
    end
    
    return TensorMatMat(factors, temp_matrices)
end

# Utility functions
function get_block_ranges(block_structure::Matrix{Int}, block_idx::Int;
                          block_sizes::Union{Nothing, Vector{Tuple{Int,Int}}}=nothing,
                          row_sizes::Union{Nothing, Vector{Int}}=nothing,
                          col_sizes::Union{Nothing, Vector{Int}}=nothing,
                          default_block_size::Int=64)
    """
    Get index ranges for a block in a block sparse matrix.

    The block_structure matrix defines the layout of blocks where:
    - block_structure[i,j] contains the block index for block at position (i,j)
    - Zero or negative values indicate empty/missing blocks
    - block_idx specifies which block we want the ranges for

    Arguments:
    - block_structure: Matrix mapping (block_row, block_col) -> block_index
    - block_idx: The block index to find ranges for
    - block_sizes: Optional vector of (rows, cols) for each block index
    - row_sizes: Optional vector of row sizes for each block row
    - col_sizes: Optional vector of column sizes for each block column
    - default_block_size: Size to use when no size info available

    Returns:
    - (i_range, j_range): Tuple of UnitRange for row and column indices

    Example:
    ```julia
    # 2x2 block structure with 4 blocks
    block_structure = [1 2; 3 4]
    row_sizes = [32, 64]  # First block row has 32 rows, second has 64
    col_sizes = [32, 64]  # First block col has 32 cols, second has 64

    i_range, j_range = get_block_ranges(block_structure, 4,
                                        row_sizes=row_sizes, col_sizes=col_sizes)
    # Returns (33:96, 33:96) for block 4 at position (2,2)
    ```
    """

    # Find the position of the requested block in the block structure
    block_pos = findfirst(x -> x == block_idx, block_structure)

    if block_pos === nothing
        throw(ArgumentError("Block index $block_idx not found in block structure"))
    end

    # Get block matrix dimensions
    num_block_rows, num_block_cols = size(block_structure)

    # Convert CartesianIndex to (row, col) coordinates
    block_row, block_col = Tuple(block_pos)

    # Determine block sizes for each block row and column
    if row_sizes !== nothing && col_sizes !== nothing
        # Use provided sizes directly
        actual_row_sizes = row_sizes
        actual_col_sizes = col_sizes
    elseif block_sizes !== nothing
        # Derive row/col sizes from block_sizes
        actual_row_sizes = zeros(Int, num_block_rows)
        actual_col_sizes = zeros(Int, num_block_cols)

        for i in 1:num_block_rows
            for j in 1:num_block_cols
                idx = block_structure[i, j]
                if idx > 0 && idx <= length(block_sizes)
                    if actual_row_sizes[i] == 0
                        actual_row_sizes[i] = block_sizes[idx][1]
                    end
                    if actual_col_sizes[j] == 0
                        actual_col_sizes[j] = block_sizes[idx][2]
                    end
                end
            end
        end

        # Fill any remaining zeros with default
        for i in 1:num_block_rows
            if actual_row_sizes[i] == 0
                actual_row_sizes[i] = default_block_size
            end
        end
        for j in 1:num_block_cols
            if actual_col_sizes[j] == 0
                actual_col_sizes[j] = default_block_size
            end
        end
    else
        # Use uniform default sizes
        actual_row_sizes = fill(default_block_size, num_block_rows)
        actual_col_sizes = fill(default_block_size, num_block_cols)
    end

    # Calculate cumulative offsets
    row_offsets = cumsum([0; actual_row_sizes[1:end-1]])
    col_offsets = cumsum([0; actual_col_sizes[1:end-1]])

    # Get ranges for this block
    i_start = row_offsets[block_row] + 1
    i_end = row_offsets[block_row] + actual_row_sizes[block_row]
    i_range = i_start:i_end

    j_start = col_offsets[block_col] + 1
    j_end = col_offsets[block_col] + actual_col_sizes[block_col]
    j_range = j_start:j_end

    return (i_range, j_range)
end

function get_all_block_ranges(block_structure::Matrix{Int};
                              row_sizes::Union{Nothing, Vector{Int}}=nothing,
                              col_sizes::Union{Nothing, Vector{Int}}=nothing,
                              default_block_size::Int=64)
    """
    Get index ranges for all blocks in a block sparse matrix.

    Returns a Dict mapping block_idx -> (i_range, j_range)
    """
    ranges = Dict{Int, Tuple{UnitRange{Int}, UnitRange{Int}}}()

    for idx in unique(block_structure)
        if idx > 0
            ranges[idx] = get_block_ranges(block_structure, idx;
                                           row_sizes=row_sizes,
                                           col_sizes=col_sizes,
                                           default_block_size=default_block_size)
        end
    end

    return ranges
end

function get_total_matrix_size(block_structure::Matrix{Int};
                               row_sizes::Union{Nothing, Vector{Int}}=nothing,
                               col_sizes::Union{Nothing, Vector{Int}}=nothing,
                               default_block_size::Int=64)
    """
    Get the total size of the full matrix from block structure.

    Returns (total_rows, total_cols)
    """
    num_block_rows, num_block_cols = size(block_structure)

    if row_sizes === nothing
        row_sizes = fill(default_block_size, num_block_rows)
    end
    if col_sizes === nothing
        col_sizes = fill(default_block_size, num_block_cols)
    end

    return (sum(row_sizes), sum(col_sizes))
end

function get_block_ranges(blocks::Vector{SparseMatrixCSC}, block_structure::Matrix{Int}, block_idx::Int)
    """
    Enhanced version that uses actual block sizes from the sparse matrices.
    This is more accurate than the size estimation approach above.
    """
    
    # Find the position of the requested block in the block structure
    block_pos = findfirst(x -> x == block_idx, block_structure)
    
    if block_pos === nothing
        throw(ArgumentError("Block index $block_idx not found in block structure"))
    end
    
    # Convert linear index to (i,j) coordinates
    block_row, block_col = Tuple(block_pos)
    
    # Calculate cumulative sizes to get actual ranges
    # This assumes blocks are ordered consistently with the block_structure
    
    # Get row ranges by summing block heights
    row_start = 1
    for i in 1:(block_row-1)
        # Find a block in this row to get its height
        for j in 1:size(block_structure, 2)
            idx = block_structure[i, j]
            if idx > 0 && idx <= length(blocks) && blocks[idx] !== nothing
                row_start += size(blocks[idx], 1)
                break
            end
        end
    end
    
    # Get the height of the current block
    current_block = blocks[block_idx]
    if current_block === nothing
        throw(ArgumentError("Block $block_idx is nothing"))
    end
    
    row_height = size(current_block, 1)
    i_range = row_start:(row_start + row_height - 1)
    
    # Get column ranges by summing block widths
    col_start = 1
    for j in 1:(block_col-1)
        # Find a block in this column to get its width
        for i in 1:size(block_structure, 1)
            idx = block_structure[i, j]
            if idx > 0 && idx <= length(blocks) && blocks[idx] !== nothing
                col_start += size(blocks[idx], 2)
                break
            end
        end
    end
    
    col_width = size(current_block, 2)
    j_range = col_start:(col_start + col_width - 1)
    
    return (i_range, j_range)
end

function benchmark_linalg_operations(sizes::Vector{Int}=[100, 500, 1000, 2000])
    """Benchmark various linear algebra operations"""
    
    println("Benchmarking linear algebra operations...")
    println("=" ^ 60)
    
    for n in sizes
        println("Matrix size: $n × $n")
        
        # Generate test matrices
        A_dense = randn(n, n)
        A_sparse = sprandn(n, n, 0.1)  # 10% sparsity
        x = randn(n)
        B = randn(n, n ÷ 2)
        
        # Create operators
        dense_matvec_op = DenseMatVec(A_dense)
        sparse_matvec_op = SparseMatVec(A_sparse)
        dense_matmat_op = DenseDenseMatMat()
        
        # Benchmark matrix-vector
        y_dense = similar(x)
        y_sparse = similar(x)
        
        t_dense_mv = @belapsed fast_matvec!($y_dense, $dense_matvec_op, $x)
        t_sparse_mv = @belapsed fast_matvec!($y_sparse, $sparse_matvec_op, $x)
        t_stdlib_mv = @belapsed mul!($y_dense, $A_dense, $x)
        
        # Benchmark matrix-matrix
        C = Matrix{Float64}(undef, n, size(B, 2))
        t_fast_mm = @belapsed fast_matmat!($C, $dense_matmat_op, $A_dense, $B)
        t_stdlib_mm = @belapsed mul!($C, $A_dense, $B)

        println("  Dense MatVec:    $(round(t_dense_mv*1e6, digits=1))μs ($(round(t_stdlib_mv/t_dense_mv, digits=2))x stdlib)")
        println("  Sparse MatVec:   $(round(t_sparse_mv*1e6, digits=1))μs")
        println("  Dense MatMat:    $(round(t_fast_mm*1e6, digits=1))μs ($(round(t_stdlib_mm/t_fast_mm, digits=2))x stdlib)")
        println()
    end
end

# Performance monitoring and statistics
function reset_linalg_stats!()
    """Reset global linear algebra statistics"""
    global GLOBAL_LINALG_STATS
    GLOBAL_LINALG_STATS.matvec_calls = 0
    GLOBAL_LINALG_STATS.matvec_time = 0.0
    GLOBAL_LINALG_STATS.matmat_calls = 0
    GLOBAL_LINALG_STATS.matmat_time = 0.0
    GLOBAL_LINALG_STATS.sparse_ops = 0
    GLOBAL_LINALG_STATS.dense_ops = 0
    GLOBAL_LINALG_STATS.blas_ops = 0
end

function print_linalg_stats()
    """Print linear algebra performance statistics"""
    stats = GLOBAL_LINALG_STATS
    
    println("Linear Algebra Performance Statistics:")
    println("=" ^ 45)
    println("Matrix-Vector Operations:")
    println("  Total calls: $(stats.matvec_calls)")
    println("  Total time:  $(round(stats.matvec_time, digits=3))s")
    if stats.matvec_calls > 0
        println("  Average time: $(round(stats.matvec_time/stats.matvec_calls*1000, digits=3))ms per call")
    end
    
    println("\nMatrix-Matrix Operations:")
    println("  Total calls: $(stats.matmat_calls)")
    println("  Total time:  $(round(stats.matmat_time, digits=3))s")
    if stats.matmat_calls > 0
        println("  Average time: $(round(stats.matmat_time/stats.matmat_calls*1000, digits=3))ms per call")
    end
    
    println("\nOperation Breakdown:")
    println("  Sparse operations: $(stats.sparse_ops)")
    println("  Dense operations:  $(stats.dense_ops)")
    println("  BLAS operations:   $(stats.blas_ops)")
    
    total_ops = stats.matvec_calls + stats.matmat_calls
    if total_ops > 0
        total_time = stats.matvec_time + stats.matmat_time
        println("\nOverall Performance:")
        println("  Total operations: $total_ops")
        println("  Total compute time: $(round(total_time, digits=3))s")
        println("  Average per operation: $(round(total_time/total_ops*1000, digits=3))ms")
    end
end

# Export main interface
export MatVecOp, MatMatOp, SparseMatVec, DenseMatVec, BlockSparseMatVec
export SparseDenseMatMat, DenseDenseMatMat, TensorMatMat
export fast_matvec!, fast_matmat!
export create_operator, create_kronecker_operator
export streaming_matvec!, cache_efficient_matmat!
export benchmark_linalg_operations, reset_linalg_stats!, print_linalg_stats
export LinalgPerformanceStats, GLOBAL_LINALG_STATS