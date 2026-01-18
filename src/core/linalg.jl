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

struct SparseMatVec{T, Ti} <: MatVecOp
    matrix::SparseMatrixCSC{T, Ti}
    workspace::Vector{T}
    
    function SparseMatVec(matrix::SparseMatrixCSC{T, Ti}) where {T, Ti}
        workspace = Vector{T}(undef, size(matrix, 1))
        new{T, Ti}(matrix, workspace)
    end
end

struct DenseMatVec{T, M<:AbstractMatrix{T}} <: MatVecOp
    matrix::M
    transposed::Bool
end

DenseMatVec(matrix::AbstractMatrix{T}; transposed::Bool=false) where T =
    DenseMatVec{T, typeof(matrix)}(matrix, transposed)

struct BlockSparseMatVec{T, Ti} <: MatVecOp
    blocks::Vector{Union{SparseMatrixCSC{T, Ti}, Nothing}}
    block_structure::Matrix{Int}  # Maps to block indices
    workspace::Vector{Vector{T}}
end

function BlockSparseMatVec(blocks::Vector{Union{SparseMatrixCSC{T, Ti}, Nothing}},
                            block_structure::Matrix{Int}) where {T, Ti}
    workspace = Vector{Vector{T}}(undef, length(blocks))
    for i in eachindex(blocks)
        block = blocks[i]
        workspace[i] = block === nothing ? Vector{T}(undef, 0) : Vector{T}(undef, size(block, 1))
    end
    return BlockSparseMatVec{T, Ti}(blocks, block_structure, workspace)
end

function BlockSparseMatVec(blocks::Vector{SparseMatrixCSC{T, Ti}},
                            block_structure::Matrix{Int}) where {T, Ti}
    blocks_union = Vector{Union{SparseMatrixCSC{T, Ti}, Nothing}}(undef, length(blocks))
    for i in eachindex(blocks)
        blocks_union[i] = blocks[i]
    end
    return BlockSparseMatVec(blocks_union, block_structure)
end

# Matrix-matrix multiplication types
abstract type MatMatOp end

struct SparseDenseMatMat{T, Ti} <: MatMatOp
    sparse_matrix::SparseMatrixCSC{T, Ti}
    dense_workspace::Matrix{T}
end

struct DenseDenseMatMat <: MatMatOp
    use_blas::Bool
    use_threads::Bool
    block_size::Int
    
    function DenseDenseMatMat(; use_blas::Bool=true, use_threads::Bool=true, block_size::Int=64)
        new(use_blas, use_threads, block_size)
    end
end

struct TensorMatMat{T} <: MatMatOp
    # For operations like A ⊗ B * vec(C) common in spectral methods
    kronecker_factors::Vector{AbstractMatrix{T}}
    temp_matrices::Vector{Matrix{T}}
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

const BlasTypes = Union{Float32, Float64, ComplexF32, ComplexF64}

@inline function scale_vector!(y::AbstractVector, α::Real)
    if α == 1
        return y
    elseif α == 0
        fill!(y, zero(eltype(y)))
        return y
    end
    # Use BLAS scal! only for CPU strided vectors, otherwise use broadcasting
    if !is_gpu_array(y) && y isa StridedVector{<:BlasTypes}
        scal!(length(y), eltype(y)(α), y, 1)
    else
        y .*= α  # Broadcasting works for both CPU and GPU
    end
    return y
end

@inline function axpy_vector!(α::Real, x::AbstractVector, y::AbstractVector)
    # Use BLAS axpy! only for CPU strided vectors, otherwise use broadcasting
    if !is_gpu_array(x) && !is_gpu_array(y) &&
       x isa StridedVector{<:BlasTypes} && y isa StridedVector{<:BlasTypes} && eltype(x) === eltype(y)
        axpy!(eltype(y)(α), x, y)
    else
        @. y = y + α * x  # Broadcasting works for both CPU and GPU
    end
    return y
end

@inline function blas_compatible(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    return A isa StridedMatrix{<:BlasTypes} &&
           B isa StridedMatrix{<:BlasTypes} &&
           C isa StridedMatrix{<:BlasTypes}
end

# Matrix-vector operations
function fast_matvec!(y::AbstractVector, op::SparseMatVec, x::AbstractVector, α::Real=1.0, β::Real=0.0)
    """Sparse matrix-vector multiplication: y = α*A*x + β*y"""

    # Sparse matrices are CPU-only in Julia - check for GPU arrays
    if is_gpu_array(x) || is_gpu_array(y)
        throw(ArgumentError("SparseMatVec does not support GPU arrays. " *
            "Julia's SparseMatrixCSC is CPU-only. Use dense operations for GPU."))
    end

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
            scale_vector!(y, α)
        end
    else
        # General case: y = α*A*x + β*y
        if β != 1.0
            scale_vector!(y, β)
        end
        mul!(op.workspace, op.matrix, x)
        axpy_vector!(α, op.workspace, y)
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

    matrix = op.matrix
    matrix_gpu = is_gpu_array(matrix)
    x_gpu = is_gpu_array(x)
    y_gpu = is_gpu_array(y)
    target_gpu = x_gpu || y_gpu

    if target_gpu
        arch = y_gpu ? architecture(y) : architecture(x)
        matrix_dev = matrix_gpu ?
            (architecture(matrix) == arch ? matrix : on_architecture(arch, matrix)) :
            on_architecture(arch, matrix)
        x_dev = x_gpu && architecture(x) == arch ? x : on_architecture(arch, x)
        y_dev = y_gpu && architecture(y) == arch ? y : on_architecture(arch, y)

        if β == 0.0
            mul!(y_dev, matrix_dev, x_dev)
            if α != 1.0
                y_dev .*= α
            end
        else
            tmp = similar(y_dev)
            mul!(tmp, matrix_dev, x_dev)
            @. y_dev = α * tmp + β * y_dev
        end

        if !y_gpu
            copyto!(y, Array(y_dev))
        end

        GLOBAL_LINALG_STATS.matvec_calls += 1
        GLOBAL_LINALG_STATS.matvec_time += time() - start_time
        GLOBAL_LINALG_STATS.dense_ops += 1
        return y
    elseif matrix_gpu
        matrix = Array(matrix)
    end

    use_blas = matrix isa StridedMatrix{<:BlasTypes} &&
               x isa StridedVector{<:BlasTypes} && y isa StridedVector{<:BlasTypes}

    if use_blas
        # Use BLAS GEMV for optimal performance (CPU only)
        if op.transposed
            # y = α*A^T*x + β*y
            gemv!('T', eltype(y)(α), matrix, x, eltype(y)(β), y)
        else
            # y = α*A*x + β*y
            gemv!('N', eltype(y)(α), matrix, x, eltype(y)(β), y)
        end
        GLOBAL_LINALG_STATS.blas_ops += 1
    else
        # Generic fallback - works for both CPU and GPU arrays
        A = op.transposed ? transpose(matrix) : matrix
        if β == 0.0
            mul!(y, A, x)
            if α != 1.0
                y .*= α  # Use broadcasting for GPU compatibility
            end
        else
            tmp = similar(y)
            mul!(tmp, A, x)
            @. y = α * tmp + β * y
        end
        GLOBAL_LINALG_STATS.dense_ops += 1
    end

    # Update statistics
    GLOBAL_LINALG_STATS.matvec_calls += 1
    GLOBAL_LINALG_STATS.matvec_time += time() - start_time

    return y
end

function fast_matvec!(y::AbstractVector, op::BlockSparseMatVec, x::AbstractVector, α::Real=1.0, β::Real=0.0)
    """Block sparse matrix-vector multiplication for structured problems"""

    # Sparse matrices are CPU-only in Julia - check for GPU arrays
    if is_gpu_array(x) || is_gpu_array(y)
        throw(ArgumentError("BlockSparseMatVec does not support GPU arrays. " *
            "Julia's SparseMatrixCSC is CPU-only. Use dense operations for GPU."))
    end

    start_time = time()

    # Initialize result
    if β == 0.0
        fill!(y, zero(eltype(y)))
    elseif β != 1.0
        scale_vector!(y, β)
    end
    
    # Compute block sizes from actual blocks for accurate range computation
    num_block_rows, num_block_cols = size(op.block_structure)
    row_sizes = zeros(Int, num_block_rows)
    col_sizes = zeros(Int, num_block_cols)

    for i in 1:num_block_rows
        for j in 1:num_block_cols
            idx = op.block_structure[i, j]
            if idx > 0 && idx <= length(op.blocks) && op.blocks[idx] !== nothing
                block = op.blocks[idx]
                if row_sizes[i] == 0
                    row_sizes[i] = size(block, 1)
                end
                if col_sizes[j] == 0
                    col_sizes[j] = size(block, 2)
                end
            end
        end
    end

    # Process each block
    for (block_idx, block) in enumerate(op.blocks)
        if block !== nothing
            # Determine block ranges using actual block sizes
            i_range, j_range = get_block_ranges(op.block_structure, block_idx;
                                                 row_sizes=row_sizes, col_sizes=col_sizes)
            
            # Extract sub-vectors
            x_sub = view(x, j_range)
            y_sub = view(y, i_range)
            
            # Block multiplication: y_sub += α * block * x_sub
            block_workspace = block_idx <= length(op.workspace) ? op.workspace[block_idx] : Vector{eltype(block)}(undef, size(block, 1))
            if length(block_workspace) != size(block, 1)
                block_workspace = Vector{eltype(block)}(undef, size(block, 1))
            end
            mul!(block_workspace, block, x_sub)
            axpy_vector!(α, block_workspace, y_sub)
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

    # Sparse matrices are CPU-only in Julia - check for GPU arrays
    if is_gpu_array(A) || is_gpu_array(B) || is_gpu_array(C)
        throw(ArgumentError("SparseDenseMatMat does not support GPU arrays. " *
            "Julia's SparseMatrixCSC is CPU-only. Use DenseDenseMatMat for GPU."))
    end

    start_time = time()
    workspace = size(op.dense_workspace) == size(C) ? op.dense_workspace : similar(C)
    
    if A_is_sparse
        # C = α*A_sparse*B_dense + β*C where A is sparse
        if β == 0.0
            mul!(C, op.sparse_matrix, B)
            if α != 1.0
                C .*= α
            end
        else
            # Use workspace to avoid overwriting C
            mul!(workspace, op.sparse_matrix, B)
            # C = α*workspace + β*C
            @. C = α * workspace + β * C
        end
    else
        # C = α*A_dense*B_sparse + β*C where B is sparse  
        if β == 0.0
            mul!(C, A, op.sparse_matrix)
            if α != 1.0
                C .*= α
            end
        else
            mul!(workspace, A, op.sparse_matrix)
            @. C = α * workspace + β * C
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

    is_gpu = is_gpu_array(A) || is_gpu_array(B) || is_gpu_array(C)
    if is_gpu
        arch = is_gpu_array(C) ? architecture(C) : (is_gpu_array(A) ? architecture(A) : architecture(B))
        A_dev = is_gpu_array(A) && architecture(A) == arch ? A : on_architecture(arch, A)
        B_dev = is_gpu_array(B) && architecture(B) == arch ? B : on_architecture(arch, B)
        C_dev = is_gpu_array(C) && architecture(C) == arch ? C : on_architecture(arch, C)

        if β == 0.0
            mul!(C_dev, A_dev, B_dev)
            if α != 1.0
                C_dev .*= α
            end
        else
            tmp = similar(C_dev)
            mul!(tmp, A_dev, B_dev)
            @. C_dev = α * tmp + β * C_dev
        end

        if !is_gpu_array(C)
            copyto!(C, Array(C_dev))
        end

        GLOBAL_LINALG_STATS.matmat_calls += 1
        GLOBAL_LINALG_STATS.matmat_time += time() - start_time
        GLOBAL_LINALG_STATS.dense_ops += 1
        return C
    end

    use_blas = op.use_blas && blas_compatible(A, B, C)

    if use_blas && size(A, 1) * size(B, 2) * size(A, 2) > 1000  # Use BLAS for large matrices
        # Use BLAS GEMM: C = α*A*B + β*C
        gemm!('N', 'N', eltype(C)(α), A, B, eltype(C)(β), C)
        GLOBAL_LINALG_STATS.blas_ops += 1

    elseif !is_gpu && op.use_threads && use_blas && size(A, 1) > op.block_size
        # Use threaded block multiplication for medium matrices (CPU only)
        threaded_block_matmat!(C, A, B, α, β, op.block_size)

    else
        # Generic fallback - works for both CPU and GPU arrays
        if β == 0.0
            mul!(C, A, B)
            if α != 1.0
                C .*= α
            end
        else
            tmp = similar(C)
            mul!(tmp, A, B)
            @. C = α * tmp + β * C
        end
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

        if length(vec_C) != n1 * n2
            throw(DimensionMismatch("Expected vec_C length $(n1 * n2), got $(length(vec_C))"))
        end
        if size(C, 1) != m2 || size(C, 2) != m1
            throw(DimensionMismatch("Output C should be $(m2)×$(m1), got $(size(C, 1))×$(size(C, 2))"))
        end

        # Reshape vec(C) to matrix form (C is n2 x n1)
        is_gpu = is_gpu_array(C) || is_gpu_array(vec_C)
        if is_gpu
            arch = is_gpu_array(C) ? architecture(C) : architecture(vec_C)
            vec_C_dev = is_gpu_array(vec_C) && architecture(vec_C) == arch ? vec_C : on_architecture(arch, vec_C)
            C_mat = reshape(vec_C_dev, (n2, n1))
            temp = create_array(arch, eltype(vec_C_dev), m2, n1)
            result_mat = create_array(arch, eltype(vec_C_dev), m2, m1)
            A1_dev = is_gpu_array(A1) && architecture(A1) == arch ? A1 : on_architecture(arch, A1)
            A2_dev = is_gpu_array(A2) && architecture(A2) == arch ? A2 : on_architecture(arch, A2)
            C_dev = is_gpu_array(C) && architecture(C) == arch ? C : on_architecture(arch, C)
        else
            C_mat = reshape(vec_C, (n2, n1))
            if length(op.temp_matrices) < 2
                throw(ArgumentError("TensorMatMat requires 2 temp matrices for 2-factor Kronecker product"))
            end
            temp = op.temp_matrices[1]   # m2 x n1
            result_mat = op.temp_matrices[2]  # m2 x m1
            A1_dev = A1
            A2_dev = A2
            C_dev = C
        end

        # Kronecker product multiplication: (A₁ ⊗ A₂) * vec(C) = vec(A₂ * C * A₁ᵀ)
        # Step 1: temp = A₂ * C
        mul!(temp, A2_dev, C_mat)
        # Step 2: result = temp * A₁ᵀ
        mul!(result_mat, temp, A1_dev')

        # Apply scaling and accumulation
        if β == 0.0
            vec(C_dev) .= α .* vec(result_mat)
        else
            vec(C_dev) .= α .* vec(result_mat) .+ β .* vec(C_dev)
        end

        if is_gpu && !is_gpu_array(C)
            copyto!(C, Array(C_dev))
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
    """SIMD small matrix multiplication - CPU only, uses generic fallback for GPU"""

    # GPU arrays don't support @turbo SIMD - use generic mul! instead
    if is_gpu_array(A) || is_gpu_array(B) || is_gpu_array(C)
        if β == 0.0
            mul!(C, A, B)
            if α != 1.0
                C .*= α
            end
        else
            tmp = similar(C)
            mul!(tmp, A, B)
            @. C = α * tmp + β * C
        end
        return
    end

    m, n, k = size(A, 1), size(B, 2), size(A, 2)

    if β == 0.0
        @turbo for i in 1:m, j in 1:n
            Cᵢⱼ = zero(eltype(C))
            for l in 1:k
                Cᵢⱼ += A[i,l] * B[l,j]
            end
            C[i,j] = α * Cᵢⱼ
        end
    else
        @turbo for i in 1:m, j in 1:n
            Cᵢⱼ = zero(eltype(C))
            for l in 1:k
                Cᵢⱼ += A[i,l] * B[l,j]
            end
            C[i,j] = α * Cᵢⱼ + β * C[i,j]
        end
    end
end

function threaded_block_matmat!(C, A, B, α, β, block_size)
    """Threaded block matrix multiplication - CPU only, uses BLAS"""

    # This function uses CPU BLAS - check for GPU arrays
    if is_gpu_array(A) || is_gpu_array(B) || is_gpu_array(C)
        throw(ArgumentError("threaded_block_matmat! does not support GPU arrays. " *
            "Use fast_matmat! with DenseDenseMatMat which has GPU fallback."))
    end

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
            gemm!('N', 'N', eltype(C_block)(α), A_block, B_block, eltype(C_block)(β), C_block)
        end
    end
end

# Memory-efficient operations for large systems
function streaming_matvec!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, chunk_size::Int=1024)
    """Memory-efficient matrix-vector multiplication for very large matrices"""

    # GPU arrays don't benefit from CPU-style chunking - use direct mul!
    # GPU memory management and kernel launch overhead make chunking counterproductive
    if is_gpu_array(y) || is_gpu_array(A) || is_gpu_array(x)
        mul!(y, A, x)
        return y
    end

    n, m = size(A)
    fill!(y, zero(eltype(y)))

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

    # GPU arrays don't benefit from CPU cache blocking - use direct mul!
    if is_gpu_array(A) || is_gpu_array(B) || is_gpu_array(C)
        mul!(C, A, B)
        return C
    end

    m, n, k = size(A, 1), size(B, 2), size(A, 2)

    # Determine block sizes based on cache size
    block_size = Int(sqrt(cache_size ÷ sizeof(eltype(A))))

    fill!(C, zero(eltype(C)))

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

    if operation_type != :matvec && operation_type != :matmat
        throw(ArgumentError("Invalid operation_type: $operation_type. Must be :matvec or :matmat"))
    end

    if issparse(matrix)
        if operation_type == :matvec
            return SparseMatVec(matrix)
        else  # :matmat
            workspace = Matrix{eltype(matrix)}(undef, size(matrix, 1), 64)  # Default columns
            return SparseDenseMatMat(matrix, workspace)
        end
    else
        if operation_type == :matvec
            return DenseMatVec(matrix)
        else  # :matmat
            return DenseDenseMatMat()
        end
    end
end

function create_kronecker_operator(factors::Vector{<:AbstractMatrix})
    """Create Kronecker product operator"""

    if isempty(factors)
        throw(ArgumentError("Cannot create Kronecker operator with empty factors vector"))
    end

    # Pre-allocate workspace matrices
    T = promote_type(map(eltype, factors)...)
    factor_mats = Vector{AbstractMatrix{T}}()
    for factor in factors
        if eltype(factor) === T
            push!(factor_mats, factor)
        else
            push!(factor_mats, Array{T}(factor))
        end
    end

    temp_matrices = Vector{Matrix{T}}()
    
    if length(factor_mats) == 2
        A1, A2 = factor_mats[1], factor_mats[2]
        # temp: A2 * C -> (m2 x n1), result: (m2 x m1)
        push!(temp_matrices, Matrix{T}(undef, size(A2, 1), size(A1, 2)))
        push!(temp_matrices, Matrix{T}(undef, size(A2, 1), size(A1, 1)))
    end
    
    return TensorMatMat{T}(factor_mats, temp_matrices)
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

function get_block_ranges(blocks::Vector{<:Union{SparseMatrixCSC, Nothing}}, block_structure::Matrix{Int}, block_idx::Int)
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

        speedup_mv = t_dense_mv > 0 ? round(t_stdlib_mv/t_dense_mv, digits=2) : NaN
        speedup_mm = t_fast_mm > 0 ? round(t_stdlib_mm/t_fast_mm, digits=2) : NaN
        println("  Dense MatVec:    $(round(t_dense_mv*1e6, digits=1))μs ($(speedup_mv)x stdlib)")
        println("  Sparse MatVec:   $(round(t_sparse_mv*1e6, digits=1))μs")
        println("  Dense MatMat:    $(round(t_fast_mm*1e6, digits=1))μs ($(speedup_mm)x stdlib)")
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
export get_block_ranges, get_all_block_ranges, get_total_matrix_size
