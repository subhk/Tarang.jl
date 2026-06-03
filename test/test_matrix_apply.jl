"""
Test suite for src/core/operators/derivatives/derivatives_matrix_apply.jl

Targets the axis-wise matrix-application helpers:
  - apply_matrix_along_axis(matrix, array, axis; out=nothing)   (dispatcher)
  - apply_dense_along_axis(matrix, array, axis; out=nothing)
  - apply_sparse_along_axis(matrix, array, axis; out=nothing, check_shapes=false)

Oracle policy: every expected value comes from an INDEPENDENT hand-computed
reference, never from the function's own output. The semantics of
"apply M along axis a" is: for every 1-D slice `s` along axis `a`, the output
slice is `M*s`. The independent oracle is therefore
    mapslices(s -> M*s, array; dims=a)
(plus exact closed-form values for small integer M/arrays).
"""

using Test
using LinearAlgebra
using SparseArrays
using Random

using Tarang
import Tarang: apply_matrix_along_axis, apply_dense_along_axis,
               apply_sparse_along_axis

# Independent oracle: apply M along `axis` by acting on every 1-D slice.
oracle(M, A, axis) = mapslices(s -> M * s, A; dims=axis)

@testset "apply_matrix_along_axis" begin

    # ========================================================================
    # 1-D arrays (vectors): apply along axis 1 == ordinary M*v
    # ========================================================================
    @testset "1D dense" begin
        M = [1 2; 3 4]          # 2x2, exact integers
        v = [5, 6]              # length-2 vector

        # Hand-computed: M*v = [1*5+2*6, 3*5+4*6] = [17, 39]
        expected = [17, 39]
        out = apply_dense_along_axis(M, v, 1)
        @test out == expected
        @test out == M * v                 # cross-check ordinary matvec
        @test out == vec(oracle(M, v, 1))  # mapslices oracle

        # Dispatcher routes dense -> same result
        @test apply_matrix_along_axis(M, v, 1) == expected

        # Negative / wraparound axis (mod1): axis -1 wraps to ndim for a vector -> 1
        @test apply_matrix_along_axis(M, v, 0) == expected   # mod1(0,1) == 1
        @test apply_matrix_along_axis(M, v, 3) == expected   # mod1(3,1) == 1
    end

    @testset "1D rectangular (3x2 applied to length-2 axis)" begin
        M = [1 0; 0 1; 1 1]    # 3x2: identity stacked over sum-row
        v = [7, 9]
        # M*v = [7, 9, 16]
        expected = [7, 9, 16]
        out = apply_matrix_along_axis(M, v, 1)
        @test size(out) == (3,)
        @test out == expected
        @test out == vec(oracle(M, v, 1))
    end

    # ========================================================================
    # 2-D arrays
    # ========================================================================
    @testset "2D dense axis 1 (columns)" begin
        M = [1 2; 3 4]
        A = [1 2 3;
             4 5 6]            # 2x3
        # axis 1: each COLUMN multiplied by M.
        # col1=[1,4] -> [1+8, 3+16]   = [9, 19]
        # col2=[2,5] -> [2+10, 6+20]  = [12, 26]
        # col3=[3,6] -> [3+12, 9+24]  = [15, 33]
        expected = [ 9 12 15;
                    19 26 33]
        out = apply_matrix_along_axis(M, A, 1)
        @test size(out) == (2, 3)
        @test out == expected
        @test out == M * A                  # axis-1 dense apply == M*A
        @test out == oracle(M, A, 1)
    end

    @testset "2D dense axis 2 (rows)" begin
        M = [1 2; 3 4]
        # axis 2 needs the axis length to equal size(M,2)=2, so A is 3x2.
        A = [1 4;
             2 5;
             3 6]              # 3x2
        # axis 2: each ROW multiplied by M.
        # row1=[1,4] -> [1+8, 3+16]   = [9, 19]
        # row2=[2,5] -> [2+10, 6+20]  = [12, 26]
        # row3=[3,6] -> [3+12, 9+24]  = [15, 33]
        expected = [ 9 19;
                    12 26;
                    15 33]
        out = apply_matrix_along_axis(M, A, 2)
        @test size(out) == (3, 2)
        @test out == expected
        @test out == (M * A')'              # axis-2 dense apply == (M*Aᵀ)ᵀ
        @test out == oracle(M, A, 2)
    end

    @testset "2D rectangular axis 1 (3x2 -> grows axis)" begin
        M = [1 0; 0 1; 1 1]   # 3x2
        A = [1 2 3;
             4 5 6]           # 2x3
        out = apply_matrix_along_axis(M, A, 1)
        @test size(out) == (3, 3)
        @test out == M * A
        @test out == oracle(M, A, 1)
    end

    @testset "2D rectangular axis 2 (2x3 -> shrinks axis)" begin
        M = [1 0 1; 0 1 0]    # 2x3
        A = [1 2 3;
             4 5 6;
             7 8 9]           # 3x3, axis-2 length 3 == size(M,2)
        out = apply_matrix_along_axis(M, A, 2)
        @test size(out) == (3, 2)
        @test out == oracle(M, A, 2)
        @test out == (M * A')'
    end

    # ========================================================================
    # 3-D arrays — each axis via the mapslices oracle
    # ========================================================================
    @testset "3D dense each axis (random, ≈)" begin
        rng = MersenneTwister(0xBEEF)
        A = rand(rng, 3, 4, 5)
        for (axis, m) in ((1, 3), (2, 4), (3, 5))
            M = rand(rng, m, m)
            out = apply_matrix_along_axis(M, A, axis)
            @test size(out) == size(A)
            @test out ≈ oracle(M, A, axis)
        end
    end

    @testset "3D dense rectangular each axis" begin
        rng = MersenneTwister(0xCAFE)
        A = rand(rng, 3, 4, 5)
        # Rectangular per axis: rows differ from cols -> output shape changes.
        for (axis, rows, cols) in ((1, 2, 3), (2, 6, 4), (3, 2, 5))
            M = rand(rng, rows, cols)
            out = apply_matrix_along_axis(M, A, axis)
            expshape = collect(size(A)); expshape[axis] = rows
            @test size(out) == Tuple(expshape)
            @test out ≈ oracle(M, A, axis)
        end
    end

    @testset "3D dense exact integers axis 2" begin
        M = [2 0; 0 3]                       # diagonal scaling
        A = reshape(collect(1:8), (2, 2, 2)) # axis-2 length 2
        out = apply_matrix_along_axis(M, A, 2)
        @test out == oracle(M, A, 2)
        # Spot-check: scaling rows of each 2x2 slice along axis 2 doubles
        # the first axis-2 layer and triples the second.
        @test out[:, 1, :] == 2 .* A[:, 1, :]
        @test out[:, 2, :] == 3 .* A[:, 2, :]
    end

    @testset "3D wraparound axis (mod1)" begin
        rng = MersenneTwister(7)
        A = rand(rng, 3, 4, 5)
        M = rand(rng, 5, 5)
        # axis 0 wraps to 3; axis -1 (==6) wraps to 3 as well (mod1(6,3)=3).
        @test apply_matrix_along_axis(M, A, 0) ≈ oracle(M, A, 3)
        @test apply_matrix_along_axis(M, A, 6) ≈ oracle(M, A, 3)
    end

    # ========================================================================
    # out= in-place buffer form
    # ========================================================================
    @testset "out= buffer (2D dense)" begin
        M = [1 2; 3 4]
        A = [1 2 3;
             4 5 6]
        expected = M * A
        buf = zeros(Int, 2, 3)
        ret = apply_matrix_along_axis(M, A, 1; out=buf)
        @test ret === buf                # returns the very buffer passed in
        @test buf == expected            # written in place
        @test buf == oracle(M, A, 1)
    end

    @testset "out= buffer rectangular shape" begin
        M = [1 0; 0 1; 1 1]              # 3x2
        A = [1 2 3;
             4 5 6]                      # 2x3 -> out 3x3
        buf = zeros(Int, 3, 3)
        ret = apply_dense_along_axis(M, A, 1; out=buf)
        @test ret === buf
        @test buf == oracle(M, A, 1)
    end

    @testset "out= matches allocating form (3D random)" begin
        rng = MersenneTwister(99)
        A = rand(rng, 3, 4, 2)
        M = rand(rng, 4, 4)
        alloc = apply_matrix_along_axis(M, A, 2)
        buf = similar(alloc)
        ret = apply_matrix_along_axis(M, A, 2; out=buf)
        @test ret === buf
        @test buf ≈ alloc
        @test buf ≈ oracle(M, A, 2)
    end

    @testset "out === array errors (cannot apply in place)" begin
        M = [1 2; 3 4]
        A = [1 2; 3 4]
        @test_throws ArgumentError apply_dense_along_axis(M, A, 1; out=A)
        @test_throws ArgumentError apply_matrix_along_axis(M, A, 1; out=A)
        Asp = sparse(M)
        @test_throws ArgumentError apply_sparse_along_axis(Asp, A, 1; out=A)
    end

    # ========================================================================
    # eltype interaction: real array with complex matrix and vice versa
    # ========================================================================
    @testset "complex array, real matrix" begin
        M = [1 2; 3 4]
        v = ComplexF64[1+2im, 3-1im]
        # M*v exactly:
        #   row1: 1*(1+2im)+2*(3-1im) = (1+2im)+(6-2im) = 7+0im
        #   row2: 3*(1+2im)+4*(3-1im) = (3+6im)+(12-4im)= 15+2im
        expected = ComplexF64[7+0im, 15+2im]
        out = apply_matrix_along_axis(M, v, 1)
        @test out == expected
        @test out == M * v
        @test eltype(out) == ComplexF64
    end

    @testset "real array, complex matrix (eltype promotion)" begin
        # FIXED 2026-06-03: the auto-allocated output buffer now uses
        # promote_type(eltype(array), eltype(matrix)) (derivatives_matrix_apply.jl),
        # so a complex matrix applied to a real array returns the correct complex
        # result instead of throwing InexactError.
        M = ComplexF64[1+0im 0+1im; 0-1im 2+0im]   # 2x2
        v = Float64[1.0, 2.0]
        # M*v: row1 = 1+2im, row2 = 4-1im
        expected = ComplexF64[1+2im, 4-1im]
        @test M * v == expected            # confirm the oracle itself
        @test apply_matrix_along_axis(M, v, 1) == expected

        # A correctly-typed `out=` buffer DOES let the complex result through,
        # confirming the only defect is the auto-allocated buffer's eltype.
        buf = zeros(ComplexF64, 2)
        ret = apply_matrix_along_axis(M, v, 1; out=buf)
        @test ret === buf
        @test buf == expected
    end

    # ========================================================================
    # SPARSE matrix path (apply_sparse_along_axis + dispatcher routing)
    # ========================================================================
    @testset "sparse 1D == dense" begin
        Md = [1 2; 3 4]
        Ms = sparse(Md)
        v = [5, 6]
        expected = [17, 39]
        # Dispatcher must route sparse to sparse helper.
        @test apply_matrix_along_axis(Ms, v, 1) == expected
        @test apply_sparse_along_axis(Ms, v, 1) == expected
        @test apply_sparse_along_axis(Ms, v, 1) == oracle(Md, v, 1) |> vec
    end

    @testset "sparse 2D axis 1 and 2" begin
        Md = [2 0; 0 3]
        Ms = sparse(Md)
        A = [1 2 3;
             4 5 6]                # 2x3
        @test apply_sparse_along_axis(Ms, A, 1) == oracle(Md, A, 1)
        @test apply_matrix_along_axis(Ms, A, 1) == oracle(Md, A, 1)

        B = [1 4; 2 5; 3 6]        # 3x2, axis-2 length 2
        @test apply_sparse_along_axis(Ms, B, 2) == oracle(Md, B, 2)
    end

    @testset "sparse 3D each axis (random, ≈)" begin
        rng = MersenneTwister(0xF00D)
        A = rand(rng, 3, 4, 5)
        for (axis, m) in ((1, 3), (2, 4), (3, 5))
            Md = rand(rng, m, m)
            Ms = sparse(Md)
            out = apply_sparse_along_axis(Ms, A, axis)
            @test size(out) == size(A)
            @test out ≈ oracle(Md, A, axis)
            @test apply_matrix_along_axis(Ms, A, axis) ≈ oracle(Md, A, axis)
        end
    end

    @testset "sparse rectangular (changes axis length)" begin
        Md = [1 0; 0 1; 1 1]      # 3x2
        Ms = sparse(Md)
        A = [1 2 3;
             4 5 6]               # 2x3 -> out 3x3
        out = apply_sparse_along_axis(Ms, A, 1)
        @test size(out) == (3, 3)
        @test out == oracle(Md, A, 1)
    end

    @testset "sparse out= buffer" begin
        Md = [2 0; 0 3]
        Ms = sparse(Md)
        A = [1 2 3;
             4 5 6]
        buf = zeros(Int, 2, 3)
        ret = apply_sparse_along_axis(Ms, A, 1; out=buf)
        @test ret === buf
        @test buf == oracle(Md, A, 1)
    end

    @testset "sparse wraparound axis (mod1)" begin
        Md = [2 0 0; 0 3 0; 0 0 4]
        Ms = sparse(Md)
        rng = MersenneTwister(11)
        A = rand(rng, 3, 4, 3)
        @test apply_sparse_along_axis(Ms, A, 0) ≈ oracle(Md, A, 3)  # mod1(0,3)=3
    end

    # ========================================================================
    # check_shapes validation branch (sparse only)
    # ========================================================================
    @testset "sparse check_shapes validation" begin
        Md = [1 2; 3 4]
        Ms = sparse(Md)
        A = [1 2 3; 4 5 6]        # 2x3, axis-1 length 2 == size(M,2)

        # Valid: matches shapes, should pass through.
        @test apply_sparse_along_axis(Ms, A, 1; check_shapes=true) == oracle(Md, A, 1)

        # Mismatch: axis length (3) != size(M,2) (2) on axis 2 of a 2x3.
        @test_throws DimensionMismatch apply_sparse_along_axis(Ms, A, 2; check_shapes=true)

        # Mismatch via wrong-size out buffer (size(M,1) != size(out,axis)).
        badbuf = zeros(Int, 3, 3)  # axis-1 should be size(M,1)=2, not 3
        @test_throws DimensionMismatch apply_sparse_along_axis(Ms, A, 1; out=badbuf, check_shapes=true)
    end

    # ========================================================================
    # CPU is_gpu_array == false sanity (GPU-staging branches are CPU-unreachable)
    # ========================================================================
    @testset "CPU arrays are not GPU arrays" begin
        @test Tarang.is_gpu_array(rand(3)) == false
        @test Tarang.is_gpu_array(rand(2, 2)) == false
    end
end
