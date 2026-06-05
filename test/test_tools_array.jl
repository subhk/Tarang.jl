# Tests for src/tools/array.jl — pure array/matrix utility functions.
# Functions are included directly into the Tarang module, so they are called
# as fully-qualified `Tarang.foo(...)` to avoid name collisions in the shared
# Main namespace used by the full test suite.

using Test
using Tarang
using LinearAlgebra
using SparseArrays

@testset "tools/array.jl" begin

    @testset "reshape_vector" begin
        let arr = collect(1:12)
            r = Tarang.reshape_vector(arr, (3, 4))
            @test size(r) == (3, 4)
            # column-major: reshape preserves linear order
            @test vec(r) == arr
            @test r == reshape(arr, (3, 4))
        end
        let arr = reshape(collect(1:24), (2, 3, 4))
            r = Tarang.reshape_vector(arr, (4, 6))
            @test size(r) == (4, 6)
            @test vec(r) == vec(arr)
        end
        # size mismatch must throw
        @test_throws ArgumentError Tarang.reshape_vector(collect(1:10), (3, 4))
    end

    @testset "axindex" begin
        # BUG (src/tools/array.jl:59): `indices = [Colon() for _ in 1:ndim]`
        # infers Vector{Colon}, which cannot store the Int `index`. Every
        # functional call throws MethodError(convert, (Colon, index)).
        # Expected: a tuple with `index` at position `axis`, Colon() elsewhere.
        @test Tarang.axindex(3, 2, 5) == (Colon(), 5, Colon())
        @test Tarang.axindex(1, 1, 7) == (7,)
        @test Tarang.axindex(4, 4, 2) == (Colon(), Colon(), Colon(), 2)
        # functional check: indexing an array via axindex should match manual slice
        let A = reshape(collect(1:24), (2, 3, 4))
            @test A[Tarang.axindex(3, 2, 2)...] == A[:, 2, :]
        end
        # out-of-bounds axis is checked before the broken assignment, so these work
        @test_throws ArgumentError Tarang.axindex(3, 0, 1)
        @test_throws ArgumentError Tarang.axindex(3, 4, 1)
    end

    @testset "axslice" begin
        # BUG (src/tools/array.jl:82): same Vector{Colon} eltype problem as
        # axindex — the slice range cannot be stored, so functional calls throw
        # MethodError(convert, (Colon, slice_range)).
        @test Tarang.axslice(3, 2, 1:2) == (Colon(), 1:2, Colon())
        @test Tarang.axslice(2, 1, 2:3) == (2:3, Colon())
        let A = reshape(collect(1:24), (2, 3, 4))
            @test A[Tarang.axslice(3, 3, 2:3)...] == A[:, :, 2:3]
        end
        @test_throws ArgumentError Tarang.axslice(3, 0, 1:2)
        @test_throws ArgumentError Tarang.axslice(3, 5, 1:2)
    end

    @testset "apply_matrix" begin
        # axis 1: ordinary matrix multiply on a matrix
        let M = [1.0 2.0; 3.0 4.0; 5.0 6.0], A = [1.0 0.0; 0.0 1.0]
            # 3x2 * 2x2 = 3x2
            @test Tarang.apply_matrix(M, A, 1) ≈ M * A
        end
        # axis 1 on a 3D array, oracle via mapslices
        let M = reshape(collect(1.0:12.0), (4, 3)),   # 4x3
            A = reshape(collect(1.0:30.0), (3, 5, 2)) # size 3 along axis 1
            R = Tarang.apply_matrix(M, A, 1)
            @test size(R) == (4, 5, 2)
            @test R ≈ mapslices(x -> M * x, A; dims=1)
        end
        # axis 2: apply along the second axis
        let M = reshape(collect(1.0:6.0), (2, 3)),    # 2x3
            A = reshape(collect(1.0:24.0), (4, 3, 2)) # size 3 along axis 2
            R = Tarang.apply_matrix(M, A, 2)
            @test size(R) == (4, 2, 2)
            @test R ≈ mapslices(x -> M * x, A; dims=2)
        end
        # axis 3
        let M = [2.0 0.0; 0.0 3.0],
            A = reshape(collect(1.0:24.0), (3, 4, 2))
            R = Tarang.apply_matrix(M, A, 3)
            @test size(R) == (3, 4, 2)
            @test R ≈ mapslices(x -> M * x, A; dims=3)
        end
        # error paths
        let A = reshape(collect(1.0:6.0), (3, 2))
            @test_throws ArgumentError Tarang.apply_matrix([1.0 2.0; 3.0 4.0], A, 0)
            @test_throws ArgumentError Tarang.apply_matrix([1.0 2.0; 3.0 4.0], A, 3)
            # matrix cols must match axis size (axis-1 size is 3, matrix has 2 cols)
            @test_throws DimensionMismatch Tarang.apply_matrix([1.0 2.0; 3.0 4.0], A, 1)
        end
    end

    @testset "apply_dense" begin
        # rectangular: returns a flat vector
        let M = reshape(collect(1.0:6.0), (2, 3)), v = [1.0, 2.0, 3.0]
            r = Tarang.apply_dense(M, v)
            @test r ≈ M * v
            @test r isa AbstractVector
        end
        # square matrix + multidim array -> reshaped back to input shape
        let M = Matrix{Float64}(I, 6, 6), A = reshape(collect(1.0:6.0), (2, 3))
            r = Tarang.apply_dense(M, A)
            @test size(r) == (2, 3)
            @test r ≈ A
        end
        # square but 1D input -> stays a vector
        let M = [0.0 1.0; 1.0 0.0], v = [10.0, 20.0]
            r = Tarang.apply_dense(M, v)
            @test r == [20.0, 10.0]
        end
        # dimension mismatch
        @test_throws DimensionMismatch Tarang.apply_dense(reshape(collect(1.0:6.0), (2, 3)), [1.0, 2.0])
    end

    @testset "apply_sparse_via_matrix" begin
        # delegates to apply_matrix; compare to dense apply_matrix
        let M = sparse([2.0 0.0; 0.0 3.0]),
            A = reshape(collect(1.0:8.0), (2, 4))
            R = Tarang.apply_sparse_via_matrix(M, A, 1)
            @test R ≈ Matrix(M) * A
            @test R ≈ Tarang.apply_matrix(Matrix(M), A, 1)
        end
        # default axis is 1
        let M = sparse([1.0 1.0; 0.0 1.0]),
            A = reshape(collect(1.0:6.0), (2, 3))
            R = Tarang.apply_sparse_via_matrix(M, A)
            @test R ≈ Matrix(M) * A
        end
    end

    @testset "kron_multi" begin
        let A = [1 2; 3 4], B = [0 1; 1 0]
            @test Tarang.kron_multi(A, B) == kron(A, B)
        end
        let A = [1.0 2.0], B = [1.0; 1.0;;], C = [2.0 0.0; 0.0 2.0]
            @test Tarang.kron_multi(A, B, C) ≈ kron(kron(A, B), C)
            @test Tarang.kron_multi(A, B, C) ≈ kron(A, B, C)
        end
        # single matrix returns itself
        let A = [1 2; 3 4]
            @test Tarang.kron_multi(A) == A
        end
        @test_throws ArgumentError Tarang.kron_multi()
    end

    @testset "sparse_block_diag" begin
        let A = [1.0 2.0; 3.0 4.0], B = [5.0 6.0; 7.0 8.0]
            R = Tarang.sparse_block_diag(A, B)
            @test R isa SparseMatrixCSC
            @test Matrix(R) == Matrix(blockdiag(sparse(A), sparse(B)))
        end
        # rectangular blocks
        let A = reshape(collect(1.0:6.0), (2, 3)), B = reshape(collect(1.0:4.0), (4, 1))
            R = Tarang.sparse_block_diag(A, B)
            @test size(R) == (6, 4)
            @test Matrix(R) == Matrix(blockdiag(sparse(A), sparse(B)))
        end
        # mixed sparse + dense input
        let A = sparse([1.0 0.0; 0.0 2.0]), B = [3.0 4.0; 5.0 6.0]
            R = Tarang.sparse_block_diag(A, B)
            @test Matrix(R) == Matrix(blockdiag(sparse(A), sparse(B)))
        end
        # three blocks
        let A = [1.0;;], B = [2.0 0.0; 0.0 2.0], C = [3.0;;]
            R = Tarang.sparse_block_diag(A, B, C)
            @test Matrix(R) == Matrix(blockdiag(sparse(A), sparse(B), sparse(C)))
        end
        @test_throws ArgumentError Tarang.sparse_block_diag()
    end

    @testset "add_sparse" begin
        let A = sparse([1.0 0.0; 2.0 3.0]), B = sparse([0.0 4.0; 1.0 1.0])
            R = Tarang.add_sparse(A, B)
            @test R == A + B
            @test Matrix(R) == Matrix(A) + Matrix(B)
        end
    end

    @testset "perm_matrix_square" begin
        # P[i, perm[i]] = 1; applying P*v gathers v[perm]
        let perm = [3, 1, 2]
            P = Tarang.perm_matrix_square(perm)
            @test P isa SparseMatrixCSC
            @test size(P) == (3, 3)
            v = [10.0, 20.0, 30.0]
            @test Matrix(P) * v == v[perm]
            # row i has its single 1 in column perm[i]
            for (i, j) in enumerate(perm)
                @test P[i, j] == 1
            end
            @test sum(P) == 3
        end
        # n larger than permutation length
        let P = Tarang.perm_matrix_square([2, 1], 4)
            @test size(P) == (4, 4)
            @test P[1, 2] == 1 && P[2, 1] == 1
        end
        # errors
        @test_throws ArgumentError Tarang.perm_matrix_square([1, 2, 3], 2)   # length > n
        @test_throws ArgumentError Tarang.perm_matrix_square([1, 1, 2])      # not unique
        @test_throws ArgumentError Tarang.perm_matrix_square([0, 1, 2])      # out of range
        @test_throws ArgumentError Tarang.perm_matrix_square([1, 2, 4])      # out of range high
    end

    @testset "permute_axis" begin
        # Uses perm_matrix_square -> apply_matrix. P*v = v[perm], so along an
        # axis this gathers slices: result[i,...] = array[perm[i],...].
        let perm = [3, 1, 2], A = reshape(collect(1.0:9.0), (3, 3))
            R = Tarang.permute_axis(A, 1, perm)
            @test R ≈ A[perm, :]
        end
        let perm = [2, 1], A = reshape(collect(1.0:6.0), (3, 2))
            R = Tarang.permute_axis(A, 2, perm)
            @test R ≈ A[:, perm]
        end
        # 3D, axis 3
        let perm = [2, 3, 1], A = reshape(collect(1.0:24.0), (2, 4, 3))
            R = Tarang.permute_axis(A, 3, perm)
            @test R ≈ A[:, :, perm]
        end
    end

    @testset "interleave_matrices" begin
        # documented example
        let A = [1 2; 3 4], B = [5 6; 7 8]
            R = Tarang.interleave_matrices(A, B)
            @test R == [1 2; 5 6; 3 4; 7 8]
            @test size(R) == (4, 2)
        end
        # three matrices: row r of each, in order
        let A = [1 2], B = [3 4], C = [5 6]
            R = Tarang.interleave_matrices(A, B, C)
            @test R == [1 2; 3 4; 5 6]
        end
        let A = reshape(collect(1:6), (3, 2)),
            B = reshape(collect(7:12), (3, 2))
            R = Tarang.interleave_matrices(A, B)
            @test size(R) == (6, 2)
            # rows k:num:end come from matrix k
            @test R[1:2:end, :] == A
            @test R[2:2:end, :] == B
        end
        # mixed eltype promotes (Int + Float64 -> Float64), no truncation
        let A = [1 2; 3 4], B = [0.5 1.5; 2.5 3.5]
            R = Tarang.interleave_matrices(A, B)
            @test eltype(R) == Float64
            @test R == [1.0 2.0; 0.5 1.5; 3.0 4.0; 2.5 3.5]
        end
        # single matrix returns equivalent matrix
        let A = [1 2; 3 4]
            @test Tarang.interleave_matrices(A) == A
        end
        # size mismatch
        @test_throws ArgumentError Tarang.interleave_matrices([1 2; 3 4], [1 2 3])
        @test_throws ArgumentError Tarang.interleave_matrices()
    end

    @testset "scipy_sparse_eigs" begin
        # Arpack constraint: ncv <= N and ncv - nev >= 2, with default
        # ncv = max(20, 2*nev+1). Use N >= 20 so default ncv fits comfortably.
        N = 24
        # Diagonal matrix: eigenvalues are exactly the diagonal entries.
        let d = Float64.(1:N),
            A = sparse(Diagonal(d))
            vals, _ = Tarang.scipy_sparse_eigs(A; nev=3, which=:LM)
            got = sort(abs.(real.(vals)))
            expected = sort(abs.(d))[end-2:end]   # 3 largest magnitudes
            @test got ≈ expected
        end
        # Symmetric tridiagonal matrix: compare LM eigenvalues vs dense eigvals.
        let M = Matrix(SymTridiagonal(Float64.(1:N), fill(0.5, N - 1))),
            A = sparse(M)
            vals, _ = Tarang.scipy_sparse_eigs(A; nev=2, which=:LM)
            dense = eigvals(Symmetric(M))
            got = sort(abs.(real.(vals)))
            expected = sort(abs.(dense))[end-1:end]
            @test got ≈ expected
        end
        # Generalized eigenproblem A x = λ B x with B = I reduces to standard.
        let M = Diagonal(Float64.(1:N)),
            A = sparse(M), B = sparse(Matrix{Float64}(I, N, N))
            vals, _ = Tarang.scipy_sparse_eigs(A, B; nev=2, which=:LM)
            got = sort(abs.(real.(vals)))
            @test got ≈ [Float64(N - 1), Float64(N)]
        end
    end

    @testset "solve_upper_sparse" begin
        # Genuine upper-triangular system: oracle is U \ b.
        let U = sparse([2.0 1.0 0.0; 0.0 3.0 1.0; 0.0 0.0 4.0]),
            b = [5.0, 6.0, 8.0]
            x = Tarang.solve_upper_sparse(U, b)
            @test x ≈ UpperTriangular(Matrix(U)) \ b
            @test U * x ≈ b
        end
        # The function only uses the upper triangle: lower entries are ignored.
        let A = sparse([2.0 1.0; 99.0 4.0]), b = [3.0, 8.0]
            x = Tarang.solve_upper_sparse(A, b)
            @test x ≈ UpperTriangular(Matrix(A)) \ b
            # equivalently, solving with the lower part zeroed
            U = sparse([2.0 1.0; 0.0 4.0])
            @test x ≈ U \ b
        end
    end

end
