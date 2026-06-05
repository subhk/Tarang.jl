# Tests for the deterministic, index-addressable random array utilities in
# src/tools/random_arrays.jl.
#
# These provide reproducible pseudo-random streams without materialising whole
# arrays: chunked_rng / ChunkedRNG (chunk iterator), rng_element / rng_elements
# (direct stream access), IndexArray + linear_index + expand_selection (the
# multi-dim -> linear-stream-position mapping) and ChunkedRandomArray (lazy
# fixed-shape container).
#
# An RNG has no closed-form oracle, so the assertions are DETERMINISM and
# CONSISTENCY oracles: the same (seed, chunk_size, distribution, index) must
# reproduce bit-for-bit across calls and across fresh objects; the batch path
# (rng_elements / ChunkedRandomArray) must agree with the single-element path;
# and the pure index math (linear_index) is checked against LinearIndices and
# hand-computed row/column-major strides. Distribution kwargs are checked for
# range / shape / dtype, not for a particular drawn value.
#
# All names are fully qualified (Tarang.foo) because the full suite shares one
# Main namespace.

using Test
using Tarang
using Random

@testset "random_arrays" begin

    # ---------------------------------------------------------------------
    # chunked_rng + ChunkedRNG iteration
    # ---------------------------------------------------------------------
    @testset "chunked_rng construction and iteration" begin
        r = Tarang.chunked_rng(UInt64(42), 8, :uniform)
        @test r isa Tarang.ChunkedRNG
        @test r.seed == UInt64(42)
        @test r.chunk_size == 8
        @test r.distribution == :uniform

        # chunk_size is clamped to >= 1
        rclamp = Tarang.chunked_rng(7, 0, :uniform)
        @test rclamp.chunk_size == 1

        # String distribution is normalised to Symbol
        rstr = Tarang.chunked_rng(7, 4, "normal")
        @test rstr.distribution == :normal

        # Keyword-only constructor
        rkw = Tarang.chunked_rng(seed=11, chunk_size=4, distribution=:uniform)
        @test rkw.seed == UInt64(11)
        @test rkw.chunk_size == 4

        # Iterating yields (chunk_index, data) tuples starting at chunk 0.
        it = iterate(r)
        @test it !== nothing
        (chunk0, data0), state = it
        @test chunk0 == 0
        @test length(data0) == 8
        (chunk1, data1), _ = iterate(r, state)
        @test chunk1 == 1
        @test length(data1) == 8

        # Re-iterating the SAME ChunkedRNG reseeds: first chunk is identical.
        (chunk0b, data0b), _ = iterate(r)
        @test data0b == data0

        # Two independently constructed RNGs with the same seed agree.
        r2 = Tarang.chunked_rng(UInt64(42), 8, :uniform)
        (_, data0_r2), _ = iterate(r2)
        @test data0_r2 == data0
    end

    # ---------------------------------------------------------------------
    # rng_element: determinism + reproducibility
    # ---------------------------------------------------------------------
    @testset "rng_element determinism" begin
        seed = UInt64(2024)
        cs = 1024

        # Same (seed, chunk_size, distribution, index) -> identical value,
        # across repeated independent calls.
        for i in (0, 1, 7, 100, 1023, 5000)
            a = Tarang.rng_element(i, seed, cs, :uniform)
            b = Tarang.rng_element(i, seed, cs, :uniform)
            @test a == b
        end

        # Different indices (almost surely) give different values.
        v0 = Tarang.rng_element(0, seed, cs, :uniform)
        v1 = Tarang.rng_element(1, seed, cs, :uniform)
        @test v0 != v1

        # Different seeds (almost surely) give different streams.
        @test Tarang.rng_element(0, UInt64(1), cs, :uniform) !=
              Tarang.rng_element(0, UInt64(2), cs, :uniform)

        # Negative index is rejected.
        @test_throws ArgumentError Tarang.rng_element(-1, seed, cs, :uniform)
    end

    # ---------------------------------------------------------------------
    # rng_elements: consistency with rng_element + shape preservation
    # ---------------------------------------------------------------------
    @testset "rng_elements consistency" begin
        seed = UInt64(777)

        # The batch path must equal the per-element path for every chunk_size,
        # because both define the same reproducible stream. Sweep several
        # chunk sizes (incl. ones smaller than the indices) and index sets.
        for cs in (1, 2, 3, 7, 16, 100, 1024)
            for idxs in ([0, 1, 2, 3, 4, 5, 10, 15], [3, 7, 11], [0], [10, 2, 5])
                singles = [Tarang.rng_element(i, seed, cs, :uniform) for i in idxs]
                batch = Tarang.rng_elements(idxs, seed, cs, :uniform)
                @test batch == singles
            end
        end

        # Output shape mirrors the shape of `indices`.
        mat = [0 1 2; 3 4 5]
        out = Tarang.rng_elements(mat, seed, 1024, :uniform)
        @test size(out) == size(mat)
        flat = Tarang.rng_elements(vec(mat), seed, 1024, :uniform)
        @test vec(out) == flat

        # Empty selection -> zero-filled array of the right element type/shape.
        empt = Tarang.rng_elements(Int[], seed, 1024, :uniform)
        @test isempty(empt)
        @test eltype(empt) == Float64

        # Negative index rejected.
        @test_throws ArgumentError Tarang.rng_elements([0, -2], seed, 1024, :uniform)
    end

    # ---------------------------------------------------------------------
    # expand_selection: 1-based per-dimension selection -> index vector
    # ---------------------------------------------------------------------
    @testset "expand_selection" begin
        @test Tarang.expand_selection(Colon(), 4) == [1, 2, 3, 4]
        @test Tarang.expand_selection(2, 4) == [2]
        @test Tarang.expand_selection(2:3, 4) == [2, 3]
        @test Tarang.expand_selection(1:4, 4) == [1, 2, 3, 4]

        # Out-of-bounds integer and range are rejected.
        @test_throws ArgumentError Tarang.expand_selection(5, 4)
        @test_throws ArgumentError Tarang.expand_selection(0, 4)
        @test_throws ArgumentError Tarang.expand_selection(3:5, 4)
        # Unsupported selection type.
        @test_throws ArgumentError Tarang.expand_selection("x", 4)
    end

    # ---------------------------------------------------------------------
    # linear_index: pure index math, checked against independent oracles
    # ---------------------------------------------------------------------
    @testset "linear_index" begin
        shape = (2, 3)

        # :F is column-major (fastest dim first). Oracle: LinearIndices is
        # column-major and 1-based, so subtract 1 for the zero-based result.
        L = LinearIndices(shape)
        for i in 1:2, j in 1:3
            @test Tarang.linear_index((i, j), shape, :F) == L[i, j] - 1
        end

        # :C is row-major (last dim fastest). Hand-computed oracle.
        rowmajor((i, j), (_, n)) = (i - 1) * n + (j - 1)
        for i in 1:2, j in 1:3
            @test Tarang.linear_index((i, j), shape, :C) == rowmajor((i, j), shape)
        end

        # Spot checks of the two orders on the same coordinate.
        @test Tarang.linear_index((2, 1), shape, :F) == 1   # down a column
        @test Tarang.linear_index((1, 2), shape, :C) == 1   # across a row

        # 3D sanity against LinearIndices for :F.
        shape3 = (2, 3, 4)
        L3 = LinearIndices(shape3)
        for i in 1:2, j in 1:3, k in 1:4
            @test Tarang.linear_index((i, j, k), shape3, :F) == L3[i, j, k] - 1
        end
    end

    # ---------------------------------------------------------------------
    # IndexArray: virtual array of linear stream positions
    # ---------------------------------------------------------------------
    @testset "IndexArray" begin
        ia = Tarang.IndexArray((2, 3), :C)
        @test size(ia) == (2, 3)
        @test length(ia) == 6
        @test ndims(ia) == 2

        # Full selection enumerates every linear index in row-major order.
        @test ia[:, :] == [0 1 2; 3 4 5]
        @test ia[1, :] == reshape([0, 1, 2], 1, 3)

        # Column-major variant.
        iaF = Tarang.IndexArray((2, 3), :F)
        @test iaF[:, :] == [0 2 4; 1 3 5]

        # Partial indexing: supplying FEWER indices than dimensions pads the
        # trailing dims with Colon (fixed 2026-06-05 — the padding used to append
        # the colon tuple as a single nested element and threw; now it spreads).
        @test ia[1] == ia[1, :]

        # Order keyword constructor; invalid orders fall back to :C.
        @test Tarang.IndexArray((2, 2); order=:F).order == :F
        @test Tarang.IndexArray((2, 2), :bogus).order == :C

        # Dimension validation and too-many-indices.
        @test_throws ArgumentError Tarang.IndexArray((0, 2))
        @test_throws ArgumentError ia[1, 2, 3]
    end

    # ---------------------------------------------------------------------
    # ChunkedRandomArray: lazy fixed-shape reproducible container
    # ---------------------------------------------------------------------
    @testset "ChunkedRandomArray" begin
        shape = (4, 5)

        cra = Tarang.ChunkedRandomArray(shape; seed=99, chunk_size=2^20,
                                        distribution=:uniform, order=:C)
        @test size(cra) == shape
        @test length(cra) == 20
        @test ndims(cra) == 2
        @test axes(cra) == (Base.OneTo(4), Base.OneTo(5))
        @test eltype(cra) == Float64

        # Determinism: a freshly constructed array with the same seed is equal.
        cra2 = Tarang.ChunkedRandomArray(shape; seed=99, chunk_size=2^20,
                                         distribution=:uniform, order=:C)
        @test cra[] == cra2[]
        # The no-index getindex equals the full :,: selection.
        @test cra[] == cra[:, :]

        # Indexing matches rng_element of the corresponding linear index.
        for (i, j) in ((1, 1), (2, 3), (4, 5), (3, 2))
            lin = Tarang.linear_index((i, j), shape, :C)
            expected = Tarang.rng_element(lin, UInt64(99), 2^20, :uniform)
            @test cra[i, j][1] == expected
        end

        # Order affects the mapping: :F vs :C give different element positions.
        craF = Tarang.ChunkedRandomArray(shape; seed=99, chunk_size=2^20,
                                         distribution=:uniform, order=:F)
        linF = Tarang.linear_index((2, 3), shape, :F)
        @test craF[2, 3][1] == Tarang.rng_element(linF, UInt64(99), 2^20, :uniform)

        # Different seeds (almost surely) give different data.
        craB = Tarang.ChunkedRandomArray(shape; seed=100, chunk_size=2^20)
        @test cra[] != craB[]

        # Dimension validation.
        @test_throws ArgumentError Tarang.ChunkedRandomArray((0, 2))
    end

    # ---------------------------------------------------------------------
    # Distributions: range / shape / dtype behaviour
    # ---------------------------------------------------------------------
    @testset "distributions" begin
        seed = UInt64(5)
        idxs = collect(0:9999)

        # Uniform respects low/high bounds.
        u = Tarang.rng_elements(idxs, seed, 2^20, :uniform; low=2.0, high=5.0)
        @test all(2.0 .<= u .< 5.0)

        # low > high is swapped to a valid range.
        u2 = Tarang.rng_elements(idxs, seed, 2^20, :uniform; low=5.0, high=2.0)
        @test all(2.0 .<= u2 .< 5.0)

        # Default uniform is [0, 1).
        ud = Tarang.rng_elements(idxs, seed, 2^20, :uniform)
        @test all(0.0 .<= ud .< 1.0)

        # Normal: mean and (positive) sigma approximately recovered.
        n = Tarang.rng_elements(idxs, seed, 2^20, :normal; mean=10.0, sigma=2.0)
        @test abs(sum(n) / length(n) - 10.0) < 0.2
        # :gaussian alias behaves identically.
        g = Tarang.rng_elements(idxs, seed, 2^20, :gaussian; mean=10.0, sigma=2.0)
        @test g == n
        # Negative sigma is equivalent to its absolute value.
        nneg = Tarang.rng_elements(idxs, seed, 2^20, :normal; mean=10.0, sigma=-2.0)
        @test nneg == n

        # complex_normal yields complex values; eltype reported as ComplexF64.
        c = Tarang.rng_elements(collect(0:9), seed, 2^20, :complex_normal)
        @test eltype(c) == Complex{Float64}
        crac = Tarang.ChunkedRandomArray((3,); seed=7, distribution=:complex_normal)
        @test eltype(crac) == Complex{Float64}
        @test eltype(crac[]) == Complex{Float64}

        # Integer dtype for uniform produces that integer type.
        ui = Tarang.rng_elements(collect(0:9), seed, 2^20, :uniform;
                                 low=0.0, high=10.0, dtype=Int)
        @test eltype(ui) == Int

        # :normal with an integer dtype is an error (randn needs floats).
        @test_throws ArgumentError Tarang.rng_elements(collect(0:3), seed, 2^20,
                                                       :normal; dtype=Int)

        # Unsupported distribution name is rejected.
        @test_throws ArgumentError Tarang.draw_distribution!(
            Random.MersenneTwister(1), :bogus, 4, NamedTuple())
    end

end
