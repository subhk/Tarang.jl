"""
Test suite for linalg.jl — linear algebra operations.

Tests:
1. SparseMatVec: y = A*x, y = α*A*x + β*y
2. DenseMatVec: y = A*x with BLAS
3. BlockSparseMatVec
4. TensorMatMat Kronecker identity: (A₁⊗A₂)vec(C) = vec(A₂*C*A₁ᵀ)
5. fftfreq against known values
6. scale_vector! and axpy_vector!
"""

using Test
using LinearAlgebra
using SparseArrays
using Tarang

const SparseMatVec = Tarang.SparseMatVec
const DenseMatVec = Tarang.DenseMatVec
const DenseDenseMatMat = Tarang.DenseDenseMatMat
const TensorMatMat = Tarang.TensorMatMat
const fast_matvec! = Tarang.fast_matvec!
const fast_matmat! = Tarang.fast_matmat!
const create_kronecker_operator = Tarang.create_kronecker_operator
const scale_vector! = Tarang.scale_vector!
const axpy_vector! = Tarang.axpy_vector!
const fftfreq = Tarang.fftfreq
const BlockSparseMatVec = Tarang.BlockSparseMatVec
const SparseDenseMatMat = Tarang.SparseDenseMatMat
const create_operator = Tarang.create_operator
const streaming_matvec! = Tarang.streaming_matvec!
const cache_efficient_matmat! = Tarang.cache_efficient_matmat!
const get_block_ranges = Tarang.get_block_ranges
const get_total_matrix_size = Tarang.get_total_matrix_size

@testset "Linear Algebra" begin

    @testset "fftfreq" begin
        # Even n
        f8 = fftfreq(8, 1.0)
        expected_8 = [0, 1, 2, 3, -4, -3, -2, -1] ./ 8.0
        @test f8 ≈ expected_8

        # Odd n
        f7 = fftfreq(7, 1.0)
        expected_7 = [0, 1, 2, 3, -3, -2, -1] ./ 7.0
        @test f7 ≈ expected_7

        # With sample rate (AbstractFFTs convention: fftfreq(n, fs) = k * fs / n)
        f4 = fftfreq(4, 0.5)
        expected_4 = [0, 1, -2, -1] .* (0.5 / 4)
        @test f4 ≈ expected_4
    end

    @testset "scale_vector!" begin
        y = [1.0, 2.0, 3.0]

        # α = 1: no-op
        scale_vector!(copy(y), 1.0)

        # α = 0: zero out
        y_zero = copy(y)
        scale_vector!(y_zero, 0.0)
        @test all(y_zero .== 0.0)

        # α = 2: double
        y_scaled = copy(y)
        scale_vector!(y_scaled, 2.0)
        @test y_scaled ≈ [2.0, 4.0, 6.0]
    end

    @testset "axpy_vector!" begin
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0]

        # y = y + 2*x
        y_result = copy(y)
        axpy_vector!(2.0, x, y_result)
        @test y_result ≈ [12.0, 24.0, 36.0]
    end

    @testset "SparseMatVec: y = A*x" begin
        A = sparse([1.0 2.0; 3.0 4.0])
        x = [1.0, 1.0]

        op = SparseMatVec(A)
        y = zeros(2)

        # y = A*x (α=1, β=0)
        fast_matvec!(y, op, x)
        @test y ≈ [3.0, 7.0]
    end

    @testset "SparseMatVec: y = α*A*x + β*y" begin
        A = sparse([1.0 0.0; 0.0 2.0])
        x = [3.0, 4.0]

        op = SparseMatVec(A)
        y = [10.0, 10.0]

        # y = 2*A*x + 0.5*y = 2*[3,8] + 0.5*[10,10] = [6,16] + [5,5] = [11,21]
        fast_matvec!(y, op, x, 2.0, 0.5)
        @test y ≈ [11.0, 21.0]
    end

    @testset "DenseMatVec: y = A*x" begin
        A = [1.0 2.0; 3.0 4.0]
        x = [1.0, 1.0]

        op = DenseMatVec(A)
        y = zeros(2)

        fast_matvec!(y, op, x)
        @test y ≈ [3.0, 7.0]
    end

    @testset "DenseMatVec: transposed" begin
        A = [1.0 2.0; 3.0 4.0]
        x = [1.0, 1.0]

        op = DenseMatVec(A; transposed=true)
        y = zeros(2)

        # y = Aᵀ*x = [1 3; 2 4] * [1; 1] = [4, 6]
        fast_matvec!(y, op, x)
        @test y ≈ [4.0, 6.0]
    end

    @testset "DenseDenseMatMat: C = A*B" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]
        C = zeros(2, 2)

        op = DenseDenseMatMat()
        fast_matmat!(C, op, A, B)
        @test C ≈ A * B
    end

    @testset "DenseDenseMatMat: C = α*A*B + β*C" begin
        A = [1.0 0.0; 0.0 2.0]
        B = [3.0 4.0; 5.0 6.0]
        C = ones(2, 2)

        op = DenseDenseMatMat()
        # C = 2*A*B + 3*C = 2*[3 4; 10 12] + 3*[1 1; 1 1] = [6 8; 20 24] + [3 3; 3 3] = [9 11; 23 27]
        fast_matmat!(C, op, A, B, 2.0, 3.0)
        @test C ≈ [9.0 11.0; 23.0 27.0]
    end

    @testset "TensorMatMat Kronecker identity" begin
        # (A₁ ⊗ A₂) * vec(C) = vec(A₂ * C * A₁ᵀ)
        A1 = [1.0 2.0; 3.0 4.0]
        A2 = [5.0 6.0; 7.0 8.0]

        op = create_kronecker_operator([A1, A2])

        # Input: C is 2×2, vec(C) is 4-element
        C_input = [1.0 2.0; 3.0 4.0]
        vec_C = vec(C_input)

        # Expected: A₂ * C * A₁ᵀ
        expected = A2 * C_input * A1'

        # Output matrix
        C_out = zeros(2, 2)
        fast_matmat!(C_out, op, vec_C)

        @test C_out ≈ expected
    end

    @testset "TensorMatMat matches full Kronecker" begin
        A1 = randn(3, 3)
        A2 = randn(4, 4)

        op = create_kronecker_operator([A1, A2])

        # Random input
        C_input = randn(4, 3)
        vec_C = vec(C_input)

        # Full Kronecker product reference
        K = kron(A1, A2)
        expected_vec = K * vec_C

        C_out = zeros(4, 3)
        fast_matmat!(C_out, op, vec_C)

        @test vec(C_out) ≈ expected_vec rtol=1e-10
    end

    @testset "BlockSparseMatVec: block-diagonal" begin
        A11 = sprand(5, 4, 0.5); A22 = sprand(3, 6, 0.5)
        blocks = Union{SparseMatrixCSC{Float64, Int}, Nothing}[A11, A22]
        bs = [1 0; 0 2]                          # (1,1)→blk1, (2,2)→blk2
        op = BlockSparseMatVec(blocks, bs)
        x = rand(4 + 6); y = zeros(5 + 3)
        fast_matvec!(y, op, x)
        @test y ≈ vcat(A11 * x[1:4], A22 * x[5:10])
    end

    @testset "SparseDenseMatMat: C = α*A_sparse*B + β*C" begin
        A = sprand(10, 8, 0.4); B = rand(8, 5)
        op = SparseDenseMatMat(A, Matrix{Float64}(undef, 10, 5))
        C = zeros(10, 5); fast_matmat!(C, op, true, A, B);  @test C ≈ A * B
        C = rand(10, 5); C0 = copy(C); fast_matmat!(C, op, true, A, B, 2.0, 0.5)
        @test C ≈ 2.0 .* (A * B) .+ 0.5 .* C0
    end

    @testset "create_operator dispatch + validation" begin
        Asp = sprand(6, 6, 0.4); Ad = rand(6, 6)
        @test create_operator(Asp, :matvec) isa SparseMatVec
        @test create_operator(Asp, :matmat) isa SparseDenseMatMat
        @test create_operator(Ad,  :matvec) isa DenseMatVec
        @test create_operator(Ad,  :matmat) isa DenseDenseMatMat
        @test_throws ArgumentError create_operator(Ad, :bogus)
        @test_throws ArgumentError create_kronecker_operator(AbstractMatrix[])
    end

    @testset "streaming_matvec! / cache_efficient_matmat!" begin
        A = rand(40, 30); x = rand(30); y = zeros(40)
        streaming_matvec!(y, A, x, 8);            @test y ≈ A * x
        A2 = rand(20, 16); B2 = rand(16, 12); C = zeros(20, 12)
        cache_efficient_matmat!(C, A2, B2);       @test C ≈ A2 * B2   # default cache_size
        # Regression: a non-perfect-square cache_size used to throw InexactError
        # (Int(√(cache_size÷8)) with √32 ≈ 5.66). Now floored.
        C2 = zeros(20, 12); cache_efficient_matmat!(C2, A2, B2, 256)
        @test C2 ≈ A2 * B2
        C3 = zeros(20, 12); cache_efficient_matmat!(C3, A2, B2, 8)   # tiny → block_size ≥ 1
        @test C3 ≈ A2 * B2
    end

    @testset "block range / total-size helpers" begin
        bs = [1 0; 0 2]; rs = [5, 3]; cs = [4, 6]
        @test get_block_ranges(bs, 1; row_sizes=rs, col_sizes=cs) == (1:5, 1:4)
        @test get_block_ranges(bs, 2; row_sizes=rs, col_sizes=cs) == (6:8, 5:10)
        @test get_total_matrix_size(bs; row_sizes=rs, col_sizes=cs) == (8, 10)
    end
end
