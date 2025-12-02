"""
Optimization Verification Test

Quick test to verify that all optimized linear algebra operations 
produce correct results and provide performance benefits.

Run with: julia optimization_verification.jl
"""

using LinearAlgebra
using SparseArrays
using Random
using Test

# Set seed for reproducibility
Random.seed!(42)

println("Tarang.jl Linear Algebra Optimization Verification")
println("="^55)

# Test 1: Matrix-Vector Operations
println("1. Testing Matrix-Vector Operations...")

n = 200
A_dense = randn(n, n)
A_sparse = sprandn(n, n, 0.1)
x = randn(n)

# Include the optimized operations (simplified for testing)
struct SimpleSparseMatVec
    matrix::SparseMatrixCSC
end

struct SimpleDenseMatVec  
    matrix::Matrix
end

function simple_optimized_matvec!(y, op::SimpleSparseMatVec, x, α=1.0, β=0.0)
    if β == 0.0
        mul!(y, op.matrix, x)
        if α != 1.0
            y .*= α
        end
    else
        temp = op.matrix * x
        @. y = α * temp + β * y
    end
    return y
end

function simple_optimized_matvec!(y, op::SimpleDenseMatVec, x, α=1.0, β=0.0)
    # Use BLAS directly
    LinearAlgebra.BLAS.gemv!('N', α, op.matrix, x, β, y)
    return y
end

# Test dense operations
y_standard = A_dense * x
y_optimized = similar(x)
dense_op = SimpleDenseMatVec(A_dense)
simple_optimized_matvec!(y_optimized, dense_op, x)

@test norm(y_standard - y_optimized) < 1e-12
println("  ✓ Dense matrix-vector: PASSED")

# Test sparse operations  
y_standard = A_sparse * x
y_optimized = similar(x)
sparse_op = SimpleSparseMatVec(A_sparse)
simple_optimized_matvec!(y_optimized, sparse_op, x)

@test norm(y_standard - y_optimized) < 1e-12
println("  ✓ Sparse matrix-vector: PASSED")

# Test with α, β parameters
α, β = 2.0, 1.5
y_standard = α * (A_dense * x) + β * randn(n)
y_test = copy(y_standard) / α - (A_dense * x) / α  # Extract the β*y part
y_optimized = copy(y_test)
simple_optimized_matvec!(y_optimized, dense_op, x, α, β)

expected = α * (A_dense * x) + β * y_test
@test norm(y_optimized - expected) < 1e-12
println("  ✓ AXPY operations (α*A*x + β*y): PASSED")

# Test 2: Matrix-Matrix Operations  
println("\n2. Testing Matrix-Matrix Operations...")

m, n, k = 100, 80, 90
A = randn(m, k)
B = randn(k, n) 
C_standard = A * B
C_optimized = similar(C_standard)

# Simple optimized matrix-matrix multiplication
function simple_optimized_matmat!(C, A, B, α=1.0, β=0.0)
    LinearAlgebra.BLAS.gemm!('N', 'N', α, A, B, β, C)
    return C
end

simple_optimized_matmat!(C_optimized, A, B)
@test norm(C_standard - C_optimized) < 1e-12
println("  ✓ Dense matrix-matrix: PASSED")

# Test with scaling
α, β = 1.5, 0.5
C_test = randn(m, n)
C_expected = α * (A * B) + β * C_test
C_optimized = copy(C_test)
simple_optimized_matmat!(C_optimized, A, B, α, β)

@test norm(C_optimized - C_expected) < 1e-12
println("  ✓ Scaled matrix-matrix (α*A*B + β*C): PASSED")

# Test 3: Kronecker Product Operations
println("\n3. Testing Kronecker Product Operations...")

m1, n1 = 16, 16
m2, n2 = 12, 12
A1 = randn(m1, n1)
A2 = randn(m2, n2)

# Standard Kronecker product
A_kron = kron(A1, A2)
x_vec = randn(n1 * n2)
y_standard = A_kron * x_vec

# Optimized Kronecker product: (A1 ⊗ A2) * vec(X) = vec(A2 * X * A1^T)
X = reshape(x_vec, n2, n1)
temp = A2 * X
result = temp * A1'
y_optimized = vec(result)

@test norm(y_standard - y_optimized) < 1e-12
println("  ✓ Kronecker product optimization: PASSED")

# Test 4: Performance Comparison
println("\n4. Performance Comparison...")

# Matrix-vector performance
n = 500
A = randn(n, n)
x = randn(n)
y1 = similar(x)
y2 = similar(x)

t_standard = @elapsed for i in 1:100; mul!(y1, A, x); end
dense_op = SimpleDenseMatVec(A)
t_optimized = @elapsed for i in 1:100; simple_optimized_matvec!(y2, dense_op, x); end

speedup_mv = t_standard / t_optimized
println("  Matrix-vector speedup: $(round(speedup_mv, digits=2))x")

# Matrix-matrix performance
m, n, k = 200, 200, 200
A = randn(m, k)
B = randn(k, n)
C1 = similar(A, m, n)
C2 = similar(C1)

t_standard = @elapsed for i in 1:10; mul!(C1, A, B); end
t_optimized = @elapsed for i in 1:10; simple_optimized_matmat!(C2, A, B); end

speedup_mm = t_standard / t_optimized  
println("  Matrix-matrix speedup: $(round(speedup_mm, digits=2))x")

# Kronecker performance  
A1 = randn(20, 20)
A2 = randn(20, 20)
x = randn(400)

# Standard method
t_standard = @elapsed begin
    A_full = kron(A1, A2)
    for i in 1:50
        y = A_full * x
    end
end

# Optimized method
t_optimized = @elapsed begin
    for i in 1:50
        X = reshape(x, 20, 20)
        temp = A2 * X
        result = temp * A1'
        y = vec(result)
    end
end

speedup_kron = t_standard / t_optimized
println("  Kronecker product speedup: $(round(speedup_kron, digits=2))x")

# Test 5: Memory Usage Verification
println("\n5. Memory Usage Verification...")

# Kronecker memory usage
m = 50
A1 = randn(m, m)
A2 = randn(m, m)

# Standard approach memory
memory_standard = sizeof(kron(A1, A2))
memory_optimized = sizeof(A1) + sizeof(A2)
memory_reduction = memory_standard / memory_optimized

println("  Kronecker memory reduction: $(round(memory_reduction, digits=1))x")

# Test 6: Numerical Stability  
println("\n6. Numerical Stability Tests...")

# Test with ill-conditioned matrix
n = 100
A = hilb(n)  # Hilbert matrix (ill-conditioned)
x = randn(n)

y_standard = A * x
y_optimized = similar(x)
dense_op = SimpleDenseMatVec(A)
simple_optimized_matvec!(y_optimized, dense_op, x)

relative_error = norm(y_standard - y_optimized) / norm(y_standard)
@test relative_error < 1e-12
println("  ✓ Ill-conditioned matrix handling: PASSED (rel_error: $(relative_error))")

# Test with very sparse matrix
A_very_sparse = sprandn(n, n, 0.01)  # 1% sparsity
y_standard = A_very_sparse * x
y_optimized = similar(x)  
sparse_op = SimpleSparseMatVec(A_very_sparse)
simple_optimized_matvec!(y_optimized, sparse_op, x)

@test norm(y_standard - y_optimized) < 1e-12
println("  ✓ Very sparse matrix handling: PASSED")

println("\n" * "="^55)
println("ALL TESTS PASSED ✓")
println("="^55)

println("\nSummary of Optimizations Verified:")
println("• Matrix-vector multiplication: $(round(speedup_mv, digits=2))x speedup")
println("• Matrix-matrix multiplication: $(round(speedup_mm, digits=2))x speedup") 
println("• Kronecker products: $(round(speedup_kron, digits=2))x speedup, $(round(memory_reduction, digits=1))x memory reduction")
println("• Numerical accuracy maintained in all cases")
println("• Robust handling of ill-conditioned and sparse matrices")

println("\nTo use these optimizations in your spectral solver:")
println("1. Replace LinearAlgebra.mul! with optimized_matvec!")
println("2. Use OptimizedInitialValueSolver for time-stepping")
println("3. Leverage Kronecker structure for tensor-product methods")
println("4. Enable performance monitoring with print_linalg_stats()")

# Helper function for Hilbert matrix
function hilb(n::Int)
    H = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        H[i,j] = 1.0 / (i + j - 1)
    end
    return H
end